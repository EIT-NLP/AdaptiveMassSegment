# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states

@dataclass
class RKVPress(ScorerPress):
    """
    R-KV: Redundancy-aware KV Cache Compression.
    Integrates Attention Importance (Max Pool) and Cosine Similarity Redundancy.
    """
    window_size: int = 8
    kernel_size: int = 7
    mix_lambda: float = 0.07        # Attention/redundancy mixing weight.
    retain_ratio: float = 0.1       # Fraction used by the redundancy filter.
    retain_direction: str = "last"  # "last", "first", "last_percent", "first_percent"
    similarity_threshold: float = 0.5 

    def cal_similarity_batched(self, key_states: torch.Tensor) -> torch.Tensor:
        """
        Batched redundancy score.

        Input: [batch, heads, seq_len, head_dim]
        Output: [batch, heads, seq_len]
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        # 1. Normalize keys and compute cosine similarity.
        k_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
        similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
        
        # Remove self-similarity on the diagonal.
        similarity_cos.diagonal(dim1=-2, dim2=-1).fill_(0.0)
        
        # 2. Keep only similarities above the redundancy threshold.
        similarity_mask = similarity_cos > self.similarity_threshold
        k_top = int(seq_len * self.retain_ratio)
        
        # 3. Find entries to suppress.
        indices = torch.where(
            similarity_mask,
            torch.arange(seq_len, device=key_states.device).expand_as(similarity_mask),
            torch.zeros_like(similarity_mask, dtype=torch.long),
        )

        if self.retain_direction == "last":
            similarity_retain = torch.max(indices, dim=-1)[0]
        elif self.retain_direction == "first":
            similarity_retain = torch.min(indices, dim=-1)[0]
        elif self.retain_direction == "last_percent":
            similarity_retain = torch.topk(indices, k=k_top, dim=-1)[0][:, :, :, 0]
        elif self.retain_direction == "first_percent":
            similarity_retain = torch.topk(indices, k=k_top, dim=-1, largest=False)[0][:, :, :, -1]
        else:
            similarity_retain = torch.max(indices, dim=-1)[0]

        # 4. Suppress retained redundant positions with scatter.
        similarity_cos.scatter_(dim=-1, index=similarity_retain.unsqueeze(-1), value=0.0)

        # 5. Average over source-token similarity and normalize.
        return similarity_cos.mean(dim=2).softmax(dim=-1)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        num_heads = module.config.num_attention_heads
        query_group_size = num_heads // num_kv_heads

        # Protect all tokens if the cache is shorter than the scoring window.
        if k_len <= self.window_size:
            return torch.ones((bsz, num_kv_heads, k_len), device=keys.device)

        # 1. Build recent RoPE-aligned query states.
        actual_window = min(self.window_size, hidden_states.shape[1])
        recent_hidden_states = hidden_states[:, -actual_window:]
        query_states = get_prerope_query_states(module, recent_hidden_states)

        if hasattr(module, "rotary_emb"):
            position_ids = torch.arange(k_len - actual_window, k_len, dtype=torch.long, device=keys.device).unsqueeze(0)
            cos, sin = module.rotary_emb(values, position_ids)
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)

        # 2. Compute recent attention scores.
        query_states_view = query_states.view(bsz, num_kv_heads, query_group_size, actual_window, head_dim)
        key_states_view = keys.unsqueeze(2) # [bsz, kv_heads, 1, seq_len, head_dim]
        attn_weights = torch.matmul(query_states_view, key_states_view.transpose(3, 4)) / math.sqrt(head_dim)
        
        # Exclude the protected recent suffix, then average GQA groups and queries.
        attn_slice = attn_weights[..., :-actual_window]
        attn_weights_sum = F.softmax(attn_slice, dim=-1, dtype=torch.float32).mean(dim=2).mean(dim=-2)
        
        # 3. Max-pool attention importance.
        attn_cache = F.max_pool1d(
            attn_weights_sum, 
            kernel_size=self.kernel_size, 
            padding=self.kernel_size // 2, 
            stride=1
        )

        # 4. Compute cosine-similarity redundancy over discardable history.
        historical_keys = keys[:, :, :-actual_window, :]
        similarity_cos = self.cal_similarity_batched(historical_keys)

        # 5. Combine attention utility and redundancy penalty.
        final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)

        # 6. Build the final score tensor and protect prefix/recent tokens.
        scores = torch.zeros((bsz, num_kv_heads, k_len), device=keys.device)
        
        # Insert historical scores.
        scores[..., :-actual_window] = final_score
        
        # Absolute protection for recent tokens and prompt prefix.
        max_val = scores.max().item() + 10.0
        scores[..., -actual_window:] = max_val
        scores[..., :4] = max_val 

        return scores
