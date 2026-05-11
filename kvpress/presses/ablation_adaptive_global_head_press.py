# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.snapkv_press import SnapKVPress

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveGlobalHeadPress(ScorerPress):
    """
    Ablation that removes AMS segment quotas and performs global top-k per KV head.

    Compared with `AdaptiveMassSegmentWrapperHeadPress`, this variant still
    computes recent-window attention evidence and can maintain EMA credit, but
    it does not build segments or allocate per-segment quotas.
    """

    base_press: Optional[ScorerPress] = None
    compression_ratio: float = 0.0

    # Recent-window attention evidence.
    window_size: int = 32
    window_kernel_size: int = 5

    # Must-keep guards.
    n_sink: int = 0
    always_keep_last: int = 0
    sort_indices: bool = True

    # Selection score source.
    use_window_scores_for_selection: bool = False

    # EMA credit is tracked for parity with AMS but is not used for segmentation.
    use_credit: bool = True
    credit_decay: float = 0.8
    mix_beta: float = 0.7

    def __post_init__(self):
        assert 0.0 <= float(self.compression_ratio) <= 1.0, "compression_ratio must be in [0,1]"
        assert self.window_size >= 1
        assert 0.0 <= float(self.credit_decay) < 1.0
        assert 0.0 <= float(self.mix_beta) <= 1.0

        if self.base_press is None and not self.use_window_scores_for_selection:
            raise ValueError(
                "base_press must be provided unless use_window_scores_for_selection=True"
            )

        # The wrapper controls the global compression ratio.
        if self.base_press is not None and hasattr(self.base_press, "compression_ratio"):
            try:
                setattr(self.base_press, "compression_ratio", 0.0)
            except Exception:
                pass

        self._compress_event_count: Dict[int, int] = {}       # layer_idx -> count
        self._credit_by_layer: Dict[int, torch.Tensor] = {}   # layer_idx -> [B, H, T]

    # --------- shared helpers ---------
    def reset(self):
        """Reset state before a new context."""
        self._compress_event_count.clear()
        self._credit_by_layer.clear()
        if self.base_press is not None and hasattr(self.base_press, "reset"):
            self.base_press.reset()

    def post_init_from_model(self, model):
        if self.base_press is not None and hasattr(self.base_press, "post_init_from_model"):
            self.base_press.post_init_from_model(model)  # type: ignore[call-arg]

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs,
    ) -> torch.Tensor:
        if self.base_press is None:
            raise RuntimeError("score() called but base_press is None.")
        return self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)

    def _get_layer_idx(self, module: nn.Module) -> int:
        return int(getattr(module, "layer_idx", -1))

    def _next_event_id(self, layer_idx: int) -> int:
        self._compress_event_count[layer_idx] = self._compress_event_count.get(layer_idx, 0) + 1
        return self._compress_event_count[layer_idx]

    # --------- window attention -> scores ---------
    def _compute_window_attn_weights(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs: dict,
        W: int,
    ) -> torch.Tensor:
        """
        Return attention from the last W queries to the preceding T-W keys.

        Shape: [B, Hq, W, T-W].
        """
        if attentions is not None:
            # attentions: [B, Hq, q_len, k_len]
            return attentions[..., -W:, :-W]

        if "position_embeddings" not in kwargs:
            raise KeyError("position_embeddings missing in kwargs; required for window attention computation.")

        return SnapKVPress.compute_window_attention(
            module, hidden_states, keys, W, kwargs["position_embeddings"]
        )

    def _window_scores_from_attn(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs: dict,
        W: int,
    ) -> torch.Tensor:
        """
        Convert window attention into KV-head-aligned scores: [B, H_kv, T].
        """
        bsz, num_kv_heads, k_len, _ = keys.shape
        num_heads = module.config.num_attention_heads
        num_groups = num_heads // num_kv_heads

        attn_w = self._compute_window_attn_weights(module, hidden_states, keys, attentions, kwargs, W)
        # [B, Hq, W, k_len - W]

        scores_q = attn_w.mean(dim=-2)  # [B, Hq, k_len-W]

        if self.window_kernel_size and self.window_kernel_size > 1:
            scores_q = F.avg_pool1d(
                scores_q,
                kernel_size=self.window_kernel_size,
                padding=self.window_kernel_size // 2,
                stride=1,
            )

        # Aggregate query heads into KV heads.
        scores_q = scores_q.view(bsz, num_kv_heads, num_groups, -1).mean(2)  # [B, H_kv, k_len-W]

        # Assign a high default score to the recent W tokens.
        scores = F.pad(scores_q, (0, W), value=float(scores_q.max().item()))
        return scores  # [B, H_kv, k_len]

    def _to_mass_headwise(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Normalize each (batch, head) curve into a distribution for credit tracking.
        """
        x = torch.relu(x)
        x = x + eps
        return x / x.sum(dim=-1, keepdim=True)

    # --------- main logic: no segments, global top-k ---------
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if float(self.compression_ratio) <= 0.0:
            return keys, values

        bsz, num_heads_kv, k_len, head_dim = keys.shape
        if k_len <= 1:
            return keys, values

        # Global keep length.
        t_keep = int(round(k_len * (1.0 - float(self.compression_ratio))))
        t_keep = max(1, min(k_len, t_keep))
        if t_keep >= k_len:
            return keys, values

        layer_idx = self._get_layer_idx(module)
        self._next_event_id(layer_idx)

        # Window size.
        q_len = int(hidden_states.shape[1])
        W = int(min(self.window_size, max(q_len - 1, 1), k_len - 1, t_keep))
        W = max(1, W)

        # 1) Window evidence.
        win_scores = self._window_scores_from_attn(
            module, hidden_states, keys, attentions, kwargs, W
        )  # [B, H_kv, T]

        # 2) Choose selection scores.
        if self.use_window_scores_for_selection:
            sel_scores = win_scores
        else:
            base_scores = self.score(module, hidden_states, keys, values, attentions, kwargs)  # [B, H_kv, T]
            sel_scores = base_scores

        # 3) Update credit for parity with AMS; selection remains pure top-k.
        mass_current = self._to_mass_headwise(win_scores)  # [B, H_kv, T]
        if self.use_credit:
            credit = self._credit_by_layer.get(layer_idx, None)
            if credit is None:
                credit = torch.zeros_like(mass_current)
            else:
                if credit.shape != mass_current.shape:
                    credit = torch.zeros_like(mass_current)

            credit = float(self.credit_decay) * credit + (1.0 - float(self.credit_decay)) * mass_current
            self._credit_by_layer[layer_idx] = credit
        # Keep this as a pure "remove segmentation" ablation by not mixing credit
        # into the selection scores.

        # 4) Must-keep guards: sink tokens and recent suffix.
        eff_last = min(int(self.always_keep_last), t_keep)
        eff_sink = min(int(self.n_sink), max(0, t_keep - eff_last))

        must_keep_list = []
        if eff_sink > 0:
            must_keep_list.append(
                torch.arange(0, eff_sink, device=keys.device, dtype=torch.long)
            )
        if eff_last > 0:
            must_keep_list.append(
                torch.arange(k_len - eff_last, k_len, device=keys.device, dtype=torch.long)
            )
        must_keep = (
            torch.unique(torch.cat(must_keep_list))
            if must_keep_list
            else torch.empty((0,), device=keys.device, dtype=torch.long)
        )

        # 5) Per-head global top-k.
        indices = torch.empty((bsz, num_heads_kv, t_keep), device=keys.device, dtype=torch.long)

        for b in range(bsz):
            for h in range(num_heads_kv):
                need = t_keep - int(must_keep.numel())

                if need < 0:
                    idx_bh = torch.arange(
                        k_len - t_keep, k_len, device=keys.device, dtype=torch.long
                    )
                else:
                    scores_all = sel_scores[b, h].clone()  # [T]
                    if must_keep.numel() > 0:
                        scores_all[must_keep] = float("-inf")

                    if need > 0:
                        topk = torch.topk(scores_all, k=min(need, k_len), dim=-1).indices
                        take = topk[:need]
                        idx_bh = torch.cat([must_keep, take], dim=0)
                    else:
                        idx_bh = must_keep.clone()

                # Pad or trim to exactly t_keep.
                if idx_bh.numel() < t_keep:
                    pad = torch.full(
                        (t_keep - idx_bh.numel(),),
                        k_len - 1,
                        device=keys.device,
                        dtype=torch.long,
                    )
                    idx_bh = torch.cat([idx_bh, pad], dim=0)
                elif idx_bh.numel() > t_keep:
                    idx_bh = idx_bh[-t_keep:]

                if self.sort_indices:
                    idx_bh, _ = torch.sort(idx_bh)

                indices[b, h] = idx_bh

        # 6) Gather KV.
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B,H,t_keep,D]
        keys_new = keys.gather(2, gather_idx).contiguous()
        values_new = values.gather(2, gather_idx).contiguous()

        # 7) Gather credit to the compressed length.
        if self.use_credit:
            credit = self._credit_by_layer.get(layer_idx, None)
            if credit is not None:
                credit_new = credit.gather(2, indices)  # [B,H,t_keep]
                self._credit_by_layer[layer_idx] = credit_new

        return keys_new, values_new
