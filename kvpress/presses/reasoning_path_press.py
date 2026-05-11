# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states

@dataclass
class ReasoningPathPress(ScorerPress):
    """
    Reasoning Path Compression (RPC) for KVPRESS.
    Computes smoothed attention scores using a window of recent queries.
    """
    window_size: int = 32  # Recent queries used as reasoning-path selectors.
    kernel_size: int = 7   # 1D smoothing window for aggregated path scores.

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
        num_kv_groups = num_heads // num_kv_heads

        # Protect all tokens if the cache is shorter than the selector window.
        if k_len <= self.window_size:
            return torch.ones((bsz, num_kv_heads, k_len), device=keys.device)

        # 1. Use the most recent hidden states as selectors.
        actual_window = min(self.window_size, hidden_states.shape[1])
        recent_hidden_states = hidden_states[:, -actual_window:]
        query_states = get_prerope_query_states(module, recent_hidden_states)

        # 2. Reconstruct RoPE-aligned queries for decoding-time scoring.
        if hasattr(module, "rotary_emb"):
            # Use the trailing cache positions for the selector queries.
            position_ids = torch.arange(k_len - actual_window, k_len, dtype=torch.long, device=keys.device).unsqueeze(0)
            cos, sin = module.rotary_emb(values, position_ids)
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)

        # 3. Compute the selector-to-cache attention score matrix.
        key_states = repeat_kv(keys, num_kv_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        # Causal safety mask.
        mask = torch.ones_like(attn_weights) * float("-inf")
        mask = torch.triu(mask, diagonal=k_len - actual_window + 1)
        attn_weights += mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 4. Aggregate across selectors and smooth along the sequence.
        attn_sum = attn_weights.sum(dim=-2) 
        
        # Average query heads into KV heads for grouped-query attention.
        attn_sum = attn_sum.view(bsz, num_kv_heads, num_kv_groups, k_len)
        scores = attn_sum.mean(dim=2)
        
        # Smooth scores with average pooling.
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

        # 5. Protect recent tokens and the prompt prefix.
        max_score = scores.max().item() + 1.0
        scores[..., -actual_window:] = max_score
        
        scores[..., :4] = max_score

        return scores
