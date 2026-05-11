# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.snapkv_press import SnapKVPress

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveMassSegmentWrapperHeadPress_NoMass(ScorerPress):
    """
    NoMass ablation:

    - Do not normalize evidence scores into a probability mass curve.
    - Use `segment_mass` as a fractional step over cumulative non-negative
      score evidence instead: cumsum(score_pos) >= segment_mass * total_score_pos.
    - Allocate budgets in proportion to each segment's raw non-negative score sum.
    - Keep in-segment top-k, must-keep guards, fallback, and KV gather unchanged.
    """

    base_press: Optional[ScorerPress] = None
    compression_ratio: float = 0.0

    # window evidence / segmentation
    window_size: int = 32
    segment_mass: float = 0.02
    min_seg_len: int = 64
    max_seg_len: int = 2048

    # quotas / guards
    min_keep_per_segment: int = 8
    n_sink: int = 0
    always_keep_last: int = 0
    sort_indices: bool = True

    # selection score source
    use_window_scores_for_selection: bool = False
    window_kernel_size: int = 5

    # stability
    use_credit: bool = True
    credit_decay: float = 0.8
    mix_beta: float = 0.7        # score_used = beta*score_current + (1-beta)*credit

    # Optional recording / visualization.
    record_segment_stats: bool = False
    record_max_points: int = 2048
    visualize: bool = False
    visualize_layer: int = 0
    visualize_every: int = 1
    visualize_dir: Optional[str] = None

    def __post_init__(self):
        assert 0.0 <= float(self.compression_ratio) <= 1.0, "compression_ratio must be in [0,1]"
        assert self.window_size >= 1, "window_size must be >= 1"
        assert 0.0 < float(self.segment_mass) <= 1.0, "segment_mass must be in (0,1]"
        assert self.min_seg_len >= 1 and self.max_seg_len >= 1 and self.min_seg_len <= self.max_seg_len
        assert self.min_keep_per_segment >= 0
        assert self.n_sink >= 0 and self.always_keep_last >= 0
        assert 0.0 <= float(self.credit_decay) < 1.0
        assert 0.0 <= float(self.mix_beta) <= 1.0

        if self.base_press is None and not self.use_window_scores_for_selection:
            raise ValueError("base_press must be provided unless use_window_scores_for_selection=True")

        # wrapper controls global compression_ratio; base_press compression_ratio is ignored
        if self.base_press is not None and hasattr(self.base_press, "compression_ratio"):
            try:
                if float(getattr(self.base_press, "compression_ratio")) not in (0.0, 0):
                    logger.warning(
                        f"[AdaptiveMassSegmentWrapperPress_NoMass] base_press.compression_ratio="
                        f"{getattr(self.base_press,'compression_ratio')} will be ignored."
                    )
                setattr(self.base_press, "compression_ratio", 0.0)
            except Exception:
                pass

        self._compress_event_count: Dict[int, int] = {}         # layer_idx -> count
        self._credit_by_layer: Dict[int, torch.Tensor] = {}     # layer_idx -> [B, H, T]
        self._press_records = []

        if self.visualize and not self.visualize_dir:
            self.visualize_dir = "./press_viz"

    def reset(self):
        self._compress_event_count.clear()
        self._credit_by_layer.clear()
        self._press_records.clear()
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

    # ------------------------- helpers -------------------------

    def _get_layer_idx(self, module: nn.Module) -> int:
        return int(getattr(module, "layer_idx", -1))

    def _next_event_id(self, layer_idx: int) -> int:
        self._compress_event_count[layer_idx] = self._compress_event_count.get(layer_idx, 0) + 1
        return self._compress_event_count[layer_idx]

    def _compute_window_attn_weights(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs: dict,
        W: int,
    ) -> torch.Tensor:
        if attentions is not None:
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
        window evidence scores aligned to KV heads: [B, H_kv, T]

        Scores are generally non-negative, but they are intentionally not
        normalized to sum to one in this ablation.
        """
        bsz, num_kv_heads, k_len, _ = keys.shape
        num_heads = module.config.num_attention_heads
        num_groups = num_heads // num_kv_heads

        attn_w = self._compute_window_attn_weights(module, hidden_states, keys, attentions, kwargs, W)
        scores_q = attn_w.mean(dim=-2)  # [B, Hq, k_len-W]

        if self.window_kernel_size and self.window_kernel_size > 1:
            scores_q = F.avg_pool1d(
                scores_q,
                kernel_size=self.window_kernel_size,
                padding=self.window_kernel_size // 2,
                stride=1,
            )

        scores_q = scores_q.view(bsz, num_kv_heads, num_groups, k_len - W).mean(2)
        scores = F.pad(scores_q, (0, W), value=float(scores_q.max().item()))
        return scores  # [B, H_kv, k_len]

    def _allocate_budgets_by_mass(
        self,
        t_keep: int,
        segments: List[Tuple[int, int]],
        seg_mass: List[float],
        seg_lengths: List[int],
    ) -> List[int]:
        n = len(segments)
        if n == 0:
            return []

        budgets = [0] * n
        for i, L in enumerate(seg_lengths):
            if L <= 0:
                budgets[i] = 0
            else:
                budgets[i] = min(L, min(self.min_keep_per_segment, L))

        base = sum(budgets)
        if base >= t_keep:
            cur = base
            while cur > t_keep:
                for i in range(n):
                    if budgets[i] > 0 and budgets[i] > 1:
                        budgets[i] -= 1
                        cur -= 1
                        if cur == t_keep:
                            break
                if cur == t_keep:
                    break
            return budgets

        remain = t_keep - base
        total_mass = float(sum(seg_mass)) + 1e-12

        raw_add = [int(round(remain * (m / total_mass))) for m in seg_mass]
        for i in range(n):
            budgets[i] = min(seg_lengths[i], budgets[i] + raw_add[i])

        cur = sum(budgets)

        while cur > t_keep:
            changed = False
            for i in range(n):
                min_i = min(self.min_keep_per_segment, seg_lengths[i]) if seg_lengths[i] > 0 else 0
                if budgets[i] > min_i:
                    budgets[i] -= 1
                    cur -= 1
                    changed = True
                    if cur == t_keep:
                        break
            if not changed:
                break

        while cur < t_keep:
            changed = False
            for i in range(n):
                if budgets[i] < seg_lengths[i]:
                    budgets[i] += 1
                    cur += 1
                    changed = True
                    if cur == t_keep:
                        break
            if not changed:
                break

        return budgets

    def _downsample_1d(self, x: torch.Tensor) -> torch.Tensor:
        T = x.numel()
        if T <= self.record_max_points:
            return x
        idx = torch.linspace(0, T - 1, self.record_max_points, device=x.device).long()
        return x.index_select(0, idx)

    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def _visualize_once(
        self,
        layer_idx: int,
        event_id: int,
        curve_1d: torch.Tensor,
        segments: List[Tuple[int, int]],
        kept_indices_1d: torch.Tensor,
    ):
        try:
            import matplotlib.pyplot as plt

            out_dir = self.visualize_dir or "./press_viz"
            self._ensure_dir(out_dir)

            curve_ds = self._downsample_1d(curve_1d.detach().float().cpu())
            T_ds = curve_ds.numel()

            if curve_1d.numel() <= self.record_max_points:
                x = torch.arange(curve_1d.numel()).cpu().numpy()
                y = curve_1d.detach().float().cpu().numpy()
                seg_lines = [s for s, _ in segments]
                kept_x = kept_indices_1d.detach().cpu().numpy()
            else:
                x = torch.arange(T_ds).cpu().numpy()
                y = curve_ds.numpy()
                scale = float(curve_1d.numel() - 1) / float(max(T_ds - 1, 1))
                seg_lines = [int(round(s / scale)) for s, _ in segments]
                kept_x = (kept_indices_1d.detach().float().cpu() / scale).round().long().numpy()

            plt.figure()
            plt.plot(x, y)
            for s in seg_lines:
                if 0 <= s < len(x):
                    plt.axvline(s, linestyle="--", linewidth=0.8)
            plt.scatter(kept_x, [0.0] * len(kept_x), s=6)

            plt.title(f"AdaptiveSegment (NoMass) | layer={layer_idx} event={event_id}")
            plt.xlabel("token position (approx)")
            plt.ylabel("score (not normalized)")

            out_path = os.path.join(out_dir, f"amseg_nomass_layer{layer_idx}_event{event_id}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=160)
            plt.close()
        except Exception as e:
            logger.debug(f"[AdaptiveMassSegmentWrapperPress_NoMass] visualize failed: {e}")

    def _build_segments_by_score_searchsorted(self, score_1d: torch.Tensor, k_len: int) -> List[Tuple[int, int]]:
        """
        Split by equal fractions of cumulative raw score evidence.

        `score_pos = relu(score) + eps` keeps the cumulative curve monotonic.
        """
        score_1d = score_1d[:k_len]
        score_pos = torch.relu(score_1d) + 1e-12
        cum = torch.cumsum(score_pos, dim=0)
        total = cum[-1]

        step = float(self.segment_mass)
        if step <= 0 or float(total.item()) <= 0:
            return [(0, k_len)]

        targets = torch.arange(step, 1.0, step, device=score_pos.device, dtype=score_pos.dtype) * total
        if targets.numel() == 0:
            return [(0, k_len)]

        cut_pos = torch.searchsorted(cum, targets).long()
        cut_pos = torch.unique(cut_pos).clamp(min=1, max=k_len - 1)

        cuts = [0] + cut_pos.tolist() + [k_len]
        segments = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]

        # Split segments that exceed the maximum length.
        fixed = []
        for s, e in segments:
            while e - s > self.max_seg_len:
                fixed.append((s, s + self.max_seg_len))
                s += self.max_seg_len
            fixed.append((s, e))
        segments = fixed

        # Merge segments that are shorter than the minimum length.
        merged = []
        i = 0
        while i < len(segments):
            s, e = segments[i]
            while (e - s) < self.min_seg_len and (i + 1) < len(segments):
                _, e2 = segments[i + 1]
                e = e2
                i += 1
            merged.append((s, e))
            i += 1

        return merged

    # ------------------------- main compress -------------------------

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

        t_keep = int(round(k_len * (1.0 - float(self.compression_ratio))))
        t_keep = max(1, min(k_len, t_keep))
        if t_keep >= k_len:
            return keys, values

        layer_idx = self._get_layer_idx(module)
        event_id = self._next_event_id(layer_idx)

        q_len = int(hidden_states.shape[1])
        W = int(min(self.window_size, max(q_len - 1, 1), k_len - 1, t_keep))
        W = max(1, W)

        # 1) window evidence scores [B, H_kv, T]
        win_scores = self._window_scores_from_attn(
            module, hidden_states, keys, attentions, kwargs, W
        )

        # 2) selection scores for segment topk
        if self.use_window_scores_for_selection:
            sel_scores = win_scores
        else:
            base_scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
            sel_scores = base_scores

        # 3) segmentation/budget evidence score (NoMass): use window evidence
        score_current = win_scores  # [B, H, T]

        # 4) optional EMA credit on score (NoMass)
        if self.use_credit:
            credit = self._credit_by_layer.get(layer_idx, None)
            if credit is None:
                credit = torch.zeros((bsz, num_heads_kv, k_len), device=keys.device, dtype=score_current.dtype)
            else:
                if credit.shape[-1] < k_len:
                    pad = k_len - credit.shape[-1]
                    credit = torch.cat(
                        [credit, torch.zeros((bsz, num_heads_kv, pad), device=keys.device, dtype=credit.dtype)],
                        dim=-1,
                    )
                elif credit.shape[-1] > k_len:
                    credit = credit[..., :k_len]

                if credit.shape[0] != bsz or credit.shape[1] != num_heads_kv:
                    logger.warning("[AdaptiveSegment_NoMass] batch/head mismatch; reset credit")
                    credit = torch.zeros((bsz, num_heads_kv, k_len), device=keys.device, dtype=score_current.dtype)

            credit = float(self.credit_decay) * credit + (1.0 - float(self.credit_decay)) * score_current
            score_used = float(self.mix_beta) * score_current + (1.0 - float(self.mix_beta)) * credit

            self._credit_by_layer[layer_idx] = credit
        else:
            score_used = score_current

        # 5) build per-head segments+budgets using mean score over batch
        score_mean = score_used.mean(dim=0)  # [H, T]

        segments_per_head: List[List[Tuple[int, int]]] = []
        budgets_per_head: List[List[int]] = []

        for h in range(num_heads_kv):
            score_1d_h = score_mean[h]  # [T] (not normalized)
            segments_h = self._build_segments_by_score_searchsorted(score_1d_h, k_len)

            seg_lengths_h = [e - s for (s, e) in segments_h]
            # segment weights: sum of non-negative evidence in segment
            seg_w_h = [float(torch.relu(score_1d_h[s:e]).sum().item()) for (s, e) in segments_h]

            budgets_h = self._allocate_budgets_by_mass(t_keep, segments_h, seg_w_h, seg_lengths_h)

            segments_per_head.append(segments_h)
            budgets_per_head.append(budgets_h)

        # 6) segment-wise topk -> candidates per head
        cand_idx_per_head: List[torch.Tensor] = []

        for h in range(num_heads_kv):
            cand_list = []
            for (start, end), budget in zip(segments_per_head[h], budgets_per_head[h]):
                L = end - start
                if L <= 0 or budget <= 0:
                    continue

                seg_score = sel_scores[:, h, start:end]  # [B, L]
                if budget >= L:
                    idx = torch.arange(start, end, device=keys.device).view(1, L).expand(bsz, L)
                else:
                    idx_local = torch.topk(seg_score, k=budget, dim=-1).indices
                    idx = idx_local + start

                cand_list.append(idx)

            if cand_list:
                cand_idx_h = torch.cat(cand_list, dim=-1)
            else:
                cand_idx_h = torch.full((bsz, 1), k_len - 1, device=keys.device, dtype=torch.long)

            cand_idx_per_head.append(cand_idx_h)

        # 7) must-keep
        eff_last = min(int(self.always_keep_last), t_keep)
        eff_sink = min(int(self.n_sink), max(0, t_keep - eff_last))

        must_keep_list = []
        if eff_sink > 0:
            must_keep_list.append(torch.arange(0, eff_sink, device=keys.device, dtype=torch.long))
        if eff_last > 0:
            must_keep_list.append(torch.arange(k_len - eff_last, k_len, device=keys.device, dtype=torch.long))
        must_keep = (
            torch.unique(torch.cat(must_keep_list))
            if must_keep_list
            else torch.empty((0,), device=keys.device, dtype=torch.long)
        )

        # 8) final indices per (B,H)
        indices_bh = torch.empty((bsz, num_heads_kv, t_keep), device=keys.device, dtype=torch.long)

        for b in range(bsz):
            for h in range(num_heads_kv):
                cand = torch.unique(cand_idx_per_head[h][b])

                # remove must_keep
                if must_keep.numel() > 0:
                    try:
                        cand = cand[~torch.isin(cand, must_keep)]
                    except Exception:
                        mask = torch.ones_like(cand, dtype=torch.bool)
                        for mk in must_keep:
                            mask &= (cand != mk)
                        cand = cand[mask]

                need = t_keep - int(must_keep.numel())
                if need < 0:
                    idx_bh = torch.arange(k_len - t_keep, k_len, device=keys.device, dtype=torch.long)
                    indices_bh[b, h] = idx_bh
                    continue

                # rank by head-specific selection score
                if cand.numel() > 0 and need > 0:
                    sc = sel_scores[b, h].index_select(0, cand)
                    order = torch.argsort(sc, descending=True)
                    cand = cand.index_select(0, order)

                # fallback global topk if insufficient
                if need > 0 and cand.numel() < need:
                    all_score = sel_scores[b, h]
                    if must_keep.numel() > 0:
                        all_score = all_score.clone()
                        all_score[must_keep] = float("-inf")
                    extra = torch.topk(all_score, k=min(need, k_len), dim=-1).indices
                    extra = torch.unique(extra)

                    if cand.numel() > 0:
                        try:
                            extra = extra[~torch.isin(extra, cand)]
                        except Exception:
                            for x in cand:
                                extra = extra[extra != x]
                    cand = torch.unique(torch.cat([cand, extra], dim=0))

                    sc = sel_scores[b, h].index_select(0, cand)
                    order = torch.argsort(sc, descending=True)
                    cand = cand.index_select(0, order)

                take = cand[:max(0, need)] if need > 0 else torch.empty((0,), device=keys.device, dtype=torch.long)
                idx_bh = torch.cat([must_keep, take], dim=0)

                # pad/trim
                if idx_bh.numel() < t_keep:
                    pad = torch.full((t_keep - idx_bh.numel(),), k_len - 1, device=keys.device, dtype=torch.long)
                    idx_bh = torch.cat([idx_bh, pad], dim=0)
                elif idx_bh.numel() > t_keep:
                    idx_bh = idx_bh[:t_keep]

                if self.sort_indices:
                    idx_bh, _ = torch.sort(idx_bh)

                indices_bh[b, h] = idx_bh

        indices_h = indices_bh  # [B, H, t_keep]

        # 9) gather KV
        gather_idx = indices_h.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        keys_new = keys.gather(2, gather_idx).contiguous()
        values_new = values.gather(2, gather_idx).contiguous()

        # 10) gather credit to match new tokens
        if self.use_credit:
            credit = self._credit_by_layer.get(layer_idx, None)
            if credit is not None:
                credit_new = credit.gather(2, indices_h)
                self._credit_by_layer[layer_idx] = credit_new

        # 11) optional record/visualize
        if self.record_segment_stats:
            with torch.no_grad():
                h0 = 0
                segments_h0 = segments_per_head[h0]
                budgets_h0 = budgets_per_head[h0]
                score_1d_h0 = score_mean[h0]  # [T] not normalized
                kept0 = indices_h[0, h0]

                kept_mask0 = torch.zeros((k_len,), device=keys.device, dtype=torch.bool)
                kept_mask0[kept0] = True

                total = float(torch.relu(score_1d_h0).sum().item() + 1e-12)

                seg_stats = []
                for seg_id, ((s, e), bud) in enumerate(zip(segments_h0, budgets_h0)):
                    L = e - s
                    if L <= 0:
                        continue
                    seg_w = float(torch.relu(score_1d_h0[s:e]).sum().item())
                    seg_frac = seg_w / total
                    seg_kept = int(kept_mask0[s:e].sum().item())
                    seg_stats.append(
                        {
                            "segment_id": int(seg_id),
                            "start": int(s),
                            "end": int(e),
                            "length": int(L),
                            "seg_weight_frac": float(seg_frac),
                            "budget": int(bud),
                            "kept_count_batch0": int(seg_kept),
                            "keep_ratio_batch0": float(seg_kept / max(L, 1)),
                        }
                    )

                curve_ds = self._downsample_1d(score_1d_h0.detach())
                kept_ds = kept0.detach().clone()
                if score_1d_h0.numel() > self.record_max_points:
                    scale = float(score_1d_h0.numel() - 1) / float(max(curve_ds.numel() - 1, 1))
                    kept_ds = (kept_ds.float() / scale).round().long()

                self._press_records.append(
                    {
                        "type": "adaptive_segment_headwise_nomass",
                        "layer_idx": int(layer_idx),
                        "event_id": int(event_id),
                        "k_len": int(k_len),
                        "t_keep": int(t_keep),
                        "compression_ratio": float(self.compression_ratio),
                        "W": int(W),
                        "segment_mass": float(self.segment_mass),
                        "min_seg_len": int(self.min_seg_len),
                        "max_seg_len": int(self.max_seg_len),
                        "n_segments": int(len(segments_h0)),
                        "budgets_sum": int(sum(budgets_h0)),
                        "seg_stats": seg_stats,
                        "score_curve_downsampled": curve_ds.detach().float().cpu().tolist(),
                        "kept_indices_batch0_head0": kept_ds.detach().cpu().tolist(),
                    }
                )

        if self.visualize and layer_idx == int(self.visualize_layer):
            if (event_id % int(self.visualize_every)) == 0:
                h0 = 0
                curve_1d = score_mean[h0]
                segments_h0 = segments_per_head[h0]
                kept0 = indices_h[0, h0]
                self._visualize_once(layer_idx, event_id, curve_1d, segments_h0, kept0)

        return keys_new, values_new
