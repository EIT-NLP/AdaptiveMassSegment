# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.snapkv_press import SnapKVPress

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveMassSegmentWrapperHeadPress_No_credit(ScorerPress):
    """
    Adaptive Mass Segment Wrapper (decode-friendly KV compression):

    - Compatible with decodepress compression_interval trigger (this class doesn't change trigger logic).
    - Every compress() recomputes segmentation & pruning on the full current cache (history + newly appended tokens).
    - Importance evidence: attention over the last W queries (window evidence) -> token-level scores -> mass.
    - Segmentation: cumulative mass threshold (segment_mass) + min/max segment length guards.
    - Budget allocation: proportional to segment mass (not length), with per-segment minimum keep.
    - Selection: segment-wise top-k on shared_score (head-shared indices), then gather KV.
    - Optional stability: EMA credit to smooth across compression events.
    - Debug: record stats + optional single-figure visualization (avoid per-layer/head spam).

    Parameters (key)
    ----------------
    base_press : ScorerPress
        Used to score tokens inside each segment (segment-wise top-k). If you prefer, you can set
        use_window_scores_for_selection=True to directly use window-derived scores for top-k.
    compression_ratio : float
        Global ratio to remove. Keep count = round(T * (1 - compression_ratio)).
    window_size : int
        Number of recent queries to use as evidence (W). Will be clamped by available hidden_states length.
    segment_mass : float
        Mass threshold to cut a segment; smaller -> more segments.
    min_seg_len / max_seg_len : int
        Guards to avoid too short/too long segments.
    min_keep_per_segment : int
        Minimum tokens kept in each segment (subject to segment length and global keep budget).
    n_sink : int
        Optional: keep first n_sink tokens as "sinks" (common in KV pruning). Guaranteed best-effort.
    always_keep_last : int
        Optional: guarantee keeping at least the last K tokens (best-effort under global keep budget).

    use_credit : bool
        Enable EMA credit to stabilize across intervals.
    credit_decay : float
        EMA decay λ in [0,1). Larger -> more memory.
    mix_beta : float
        mass_used = normalize( mix_beta*mass_current + (1-mix_beta)*normalize(credit) )

    record_segment_stats : bool
        Record per-compress stats into self._press_records.
    record_max_points : int
        Downsample mass curve in records (and visualization) to avoid giant logs.
    visualize : bool
        Save a single png visualization for a chosen layer every visualize_every compress events.
    visualize_layer : int
        Only visualize when module.layer_idx == visualize_layer.
    visualize_every : int
        Visualize every Nth compress event for that layer.
    visualize_dir : str
        Where to save pngs if visualize=True.
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
    use_window_scores_for_selection: bool = False  # if True: segment topk uses window-derived shared_score
    window_kernel_size: int = 5  # optional smoothing like SnapKV (avg_pool1d)

    # stability
    use_credit: bool = False
    credit_decay: float = 0.8
    mix_beta: float = 0.7

    # recording / visualization
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
        assert self.use_credit == False

        if self.base_press is None and not self.use_window_scores_for_selection:
            raise ValueError("base_press must be provided unless use_window_scores_for_selection=True")

        # wrapper controls global compression_ratio; base_press compression_ratio is ignored
        if self.base_press is not None and hasattr(self.base_press, "compression_ratio"):
            try:
                if float(getattr(self.base_press, "compression_ratio")) not in (0.0, 0):
                    logger.warning(
                        f"[AdaptiveMassSegmentWrapperPress] base_press.compression_ratio="
                        f"{getattr(self.base_press,'compression_ratio')} will be ignored."
                    )
                setattr(self.base_press, "compression_ratio", 0.0)
            except Exception:
                pass

        # internal state
        self._compress_event_count: Dict[int, int] = {}   # layer_idx -> count
        self._credit_by_layer: Dict[int, torch.Tensor] = {}  # layer_idx -> [B, T]
        self._press_records = []

        if self.visualize and not self.visualize_dir:
            self.visualize_dir = "./press_viz"

    def reset(self):
        """Reset per-sample/context state to avoid cross-sample leakage."""
        self._compress_event_count.clear()
        self._credit_by_layer.clear()
        self._press_records.clear()
        if self.base_press is not None and hasattr(self.base_press, "reset"):
            self.base_press.reset()

    def post_init_from_model(self, model):
        if self.base_press is not None and hasattr(self.base_press, "post_init_from_model"):
            self.base_press.post_init_from_model(model)  # type: ignore[call-arg]

    # Keep score() delegate for compatibility (some pipelines may call press.score)
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

    # ------------------------- core helpers -------------------------

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
        """
        Returns attention weights of shape [B, Hq, W, (T - W)] for the last W queries
        attending to the first (T - W) keys.
        """
        if attentions is not None:
            # attentions: [B, Hq, q_len, k_len] typically
            # take last W queries to first k_len-W keys
            return attentions[..., -W:, :-W]

        # fallback: compute attention weights for last W queries
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
        Convert window attention weights into scores aligned to KV heads: [B, H_kv, T]
        Similar to SnapKV scoring, but W is configurable here.
        """
        bsz, num_kv_heads, k_len, _ = keys.shape
        num_heads = module.config.num_attention_heads
        num_groups = num_heads // num_kv_heads

        attn_w = self._compute_window_attn_weights(module, hidden_states, keys, attentions, kwargs, W)
        # attn_w: [B, Hq, W, k_len - W]

        scores_q = attn_w.mean(dim=-2)  # avg over W queries -> [B, Hq, k_len-W]

        # optional smoothing (like SnapKV)
        if self.window_kernel_size and self.window_kernel_size > 1:
            scores_q = F.avg_pool1d(
                scores_q,
                kernel_size=self.window_kernel_size,
                padding=self.window_kernel_size // 2,
                stride=1,
            )

        # group-average to kv heads: [B, H_kv, k_len-W]
        scores_q = scores_q.view(bsz, num_kv_heads, num_groups, k_len - W).mean(2)

        # add back observation window, pad with max to discourage pruning recent tokens
        scores = F.pad(scores_q, (0, W), value=float(scores_q.max().item()))
        # scores: [B, H_kv, k_len]
        return scores

    def _to_mass(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        x: [B, T] -> mass: [B, T], non-negative and normalized
        """
        x = torch.relu(x)
        x = x + eps
        return x / x.sum(dim=-1, keepdim=True)

    def _build_segments_by_mass(
        self,
        mass_1d: torch.Tensor,
        k_len: int,
    ) -> List[Tuple[int, int]]:
        """
        mass_1d: [T] (already normalized)
        Returns segments [(start,end), ...] covering [0,T)
        """
        segments: List[Tuple[int, int]] = []
        start = 0
        cum = 0.0
        for t in range(k_len):
            cum += float(mass_1d[t].item())
            seg_len = (t + 1) - start

            force_cut = seg_len >= self.max_seg_len
            reach_mass = cum >= float(self.segment_mass) and seg_len >= self.min_seg_len

            if force_cut or reach_mass:
                end = t + 1
                segments.append((start, end))
                start = end
                cum = 0.0

        if start < k_len:
            segments.append((start, k_len))
        return segments

    def _allocate_budgets_by_mass(
        self,
        t_keep: int,
        segments: List[Tuple[int, int]],
        seg_mass: List[float],
        seg_lengths: List[int],
    ) -> List[int]:
        """
        Allocate budgets proportional to seg_mass, with min_keep_per_segment, and cap by seg length.
        """
        n = len(segments)
        if n == 0:
            return []

        # baseline min keep (best-effort)
        budgets = [0] * n
        for i, L in enumerate(seg_lengths):
            if L <= 0:
                budgets[i] = 0
            else:
                budgets[i] = min(L, min(self.min_keep_per_segment, L))

        base = sum(budgets)
        if base >= t_keep:
            # not enough room for more; shrink if needed
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

        # distribute remaining by mass
        raw_add = [int(round(remain * (m / total_mass))) for m in seg_mass]
        for i in range(n):
            budgets[i] = min(seg_lengths[i], budgets[i] + raw_add[i])

        # adjust sum exactly to t_keep
        cur = sum(budgets)

        # if too large: reduce from segments with slack above min_keep
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

        # if too small: add where possible
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
        """
        Downsample to at most record_max_points for logging/plotting.
        """
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
        mass_1d: torch.Tensor,
        segments: List[Tuple[int, int]],
        kept_indices_1d: torch.Tensor,
    ):
        """
        Save a single figure: mass curve + segment boundaries + kept points.
        Only called when visualize=True and layer matches visualize_layer.
        """
        try:
            import matplotlib.pyplot as plt  # optional dependency

            out_dir = self.visualize_dir or "./press_viz"
            self._ensure_dir(out_dir)

            mass_ds = self._downsample_1d(mass_1d.detach().float().cpu())
            T_ds = mass_ds.numel()

            # map segments and kept indices into downsampled coordinate space (approx)
            # For readability, keep it simple: plot in original scale if not too large; else downsample x.
            if mass_1d.numel() <= self.record_max_points:
                x = torch.arange(mass_1d.numel()).cpu().numpy()
                y = mass_1d.detach().float().cpu().numpy()
                seg_lines = [s for s, _ in segments]
                kept_x = kept_indices_1d.detach().cpu().numpy()
            else:
                # downsample x
                x = torch.arange(T_ds).cpu().numpy()
                y = mass_ds.numpy()
                # approximate mapping:
                scale = float(mass_1d.numel() - 1) / float(max(T_ds - 1, 1))
                seg_lines = [int(round(s / scale)) for s, _ in segments]
                kept_x = (kept_indices_1d.detach().float().cpu() / scale).round().long().numpy()

            plt.figure()
            plt.plot(x, y)
            for s in seg_lines:
                if 0 <= s < len(x):
                    plt.axvline(s, linestyle="--", linewidth=0.8)
            # plot kept indices as markers near y=0 (just for positions)
            plt.scatter(kept_x, [0.0] * len(kept_x), s=6)

            plt.title(f"AdaptiveMassSegment KV pruning | layer={layer_idx} event={event_id}")
            plt.xlabel("token position (approx)")
            plt.ylabel("mass (normalized)")

            out_path = os.path.join(out_dir, f"amseg_layer{layer_idx}_event{event_id}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=160)
            plt.close()
        except Exception as e:
            logger.debug(f"[AdaptiveMassSegmentWrapperPress] visualize failed: {e}")


    def _to_mass_headwise(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Normalize each (batch, KV-head) curve into a distribution.
        """
        x = torch.relu(x)
        x = x + eps
        return x / x.sum(dim=-1, keepdim=True)

    def _build_segments_by_mass_searchsorted(self, mass_1d: torch.Tensor, k_len: int) -> List[Tuple[int,int]]:
        """
        Build equal-mass segments with lightweight length corrections.

        `mass_1d` is normalized and sums to one.
        """
        mass_1d = mass_1d[:k_len]
        cum = torch.cumsum(mass_1d, dim=0)

        step = float(self.segment_mass)
        if step <= 0:
            return [(0, k_len)]

        # Target cut points on the cumulative mass curve.
        targets = torch.arange(step, 1.0, step, device=mass_1d.device, dtype=mass_1d.dtype)
        if targets.numel() == 0:
            return [(0, k_len)]

        cut_pos = torch.searchsorted(cum, targets).to(torch.long)  # [n_cuts]
        cut_pos = torch.unique(cut_pos).clamp(min=1, max=k_len - 1)

        # Convert cuts to half-open segments.
        cuts = cut_pos.tolist()
        cuts = [0] + cuts + [k_len]
        segments = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b > a:
                segments.append((a, b))

        # Enforce approximate min/max lengths.
        # 1) Split segments that are too long.
        fixed = []
        for s, e in segments:
            while e - s > self.max_seg_len:
                fixed.append((s, s + self.max_seg_len))
                s = s + self.max_seg_len
            fixed.append((s, e))
        segments = fixed

        # 2) Merge segments that are too short with following segments.
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

        # global keep
        t_keep = int(round(k_len * (1.0 - float(self.compression_ratio))))
        t_keep = max(1, min(k_len, t_keep))
        if t_keep >= k_len:
            return keys, values

        layer_idx = self._get_layer_idx(module)
        event_id = self._next_event_id(layer_idx)
        

        # clamp W by available hidden_states length
        q_len = int(hidden_states.shape[1])
        W = int(min(self.window_size, max(q_len - 1, 1), k_len - 1, t_keep))
        W = max(1, W)
        
        
        # Optional debug logging kept disabled for release runs.
        # if (event_id <= 2) or (event_id % 50 == 0):
        #     logger.info(
        #         f"[HeadSegPress] compress trigger | layer={layer_idx} event={event_id} "
        #         f"k_len={k_len} -> t_keep={t_keep} (ratio={self.compression_ratio:.3f}) W={W} "
        #         f"H_kv={num_heads_kv} B={bsz}"
        #     )

        


        # # 1) window-derived scores [B, H_kv, T]
        win_scores = self._window_scores_from_attn(
            module, hidden_states, keys, attentions, kwargs, W
        )  # [B, H_kv, T]

        # # 2) selection scores (segment top-k): either base_press score or window score
        # Keep head-specific selection scores; do not average heads early.
        if self.use_window_scores_for_selection:
            sel_scores = win_scores                         # [B, H_kv, T]
        else:
            base_scores = self.score(module, hidden_states, keys, values, attentions, kwargs)  # [B, H_kv, T]
            sel_scores = base_scores                        # [B, H_kv, T]


        # 3) mass_current from window evidence (head-wise)
        mass_current = self._to_mass_headwise(win_scores)      # [B, H_kv, T]


        if self.use_credit:
            credit = self._credit_by_layer.get(layer_idx, None)
            if credit is None:
                credit = torch.zeros((bsz, num_heads_kv, k_len), device=keys.device, dtype=mass_current.dtype)
            else:
                # Align sequence length.
                if credit.shape[-1] < k_len:
                    pad = k_len - credit.shape[-1]
                    credit = torch.cat(
                        [credit, torch.zeros((bsz, num_heads_kv, pad), device=keys.device, dtype=credit.dtype)],
                        dim=-1
                    )
                elif credit.shape[-1] > k_len:
                    credit = credit[..., :k_len]

                # Align batch/head dimensions defensively.
                if credit.shape[0] != bsz or credit.shape[1] != num_heads_kv:
                    logger.warning("[AdaptiveSegment] batch/head mismatch; resetting credit")
                    credit = torch.zeros((bsz, num_heads_kv, k_len), device=keys.device, dtype=mass_current.dtype)

            # EMA: credit <- λ credit + (1-λ) mass_current
            credit = float(self.credit_decay) * credit + (1.0 - float(self.credit_decay)) * mass_current

            # Mix with normalized credit for numerical stability.
            credit_mass = self._to_mass_headwise(credit)
            mass_used = self._to_mass_headwise(float(self.mix_beta) * mass_current + (1.0 - float(self.mix_beta)) * credit_mass)

            self._credit_by_layer[layer_idx] = credit
        else:
            mass_used = mass_current


        # 5) build a single segmentation (shared across batch) using mean mass
        # mass_used: [B, H, T]
        mass_mean = mass_used.mean(dim=0)   # [H, T]

        segments_per_head = []
        budgets_per_head = []

        for h in range(num_heads_kv):
            mass_1d_h = mass_mean[h]  # [T]
            mass_1d_h = self._to_mass(mass_1d_h) 

            # Use the searchsorted implementation for efficient boundary lookup.
            segments_h = self._build_segments_by_mass_searchsorted(mass_1d_h, k_len)

            seg_lengths_h = [e - s for (s, e) in segments_h]
            seg_mass_h = [float(mass_1d_h[s:e].sum().item()) for (s, e) in segments_h]

            budgets_h = self._allocate_budgets_by_mass(t_keep, segments_h, seg_mass_h, seg_lengths_h)

            segments_per_head.append(segments_h)
            budgets_per_head.append(budgets_h)
            
            
        # Optional per-head segment statistics for debugging.
        # if (event_id <= 2) or (event_id % 50 == 0):
        #     seg_counts = [len(segs) for segs in segments_per_head]
        #     bud_sums = [sum(buds) for buds in budgets_per_head]

            # logger.info(
            #     f"[HeadSegPress] head stats | layer={layer_idx} event={event_id} "
            #     f"segments(min/avg/max)={min(seg_counts)}/{(sum(seg_counts)/len(seg_counts)):.1f}/{max(seg_counts)} "
            #     f"budget_sum(min/avg/max)={min(bud_sums)}/{(sum(bud_sums)/len(bud_sums)):.1f}/{max(bud_sums)} "
            #     f"target={t_keep}"
            # )

        
        
        # 6) segment-wise topk -> candidate indices per batch
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
                cand_idx_h = torch.cat(cand_list, dim=-1)  # [B, ~t_keep]
            else:
                cand_idx_h = torch.full((bsz, 1), k_len - 1, device=keys.device, dtype=torch.long)

            cand_idx_per_head.append(cand_idx_h)


        # 7) enforce must-keep: sinks + last tokens (best-effort under t_keep)
        # effective keeps (avoid impossible constraints)
        eff_last = min(int(self.always_keep_last), t_keep)
        eff_sink = min(int(self.n_sink), max(0, t_keep - eff_last))

        must_keep_list = []
        if eff_sink > 0:
            must_keep_list.append(torch.arange(0, eff_sink, device=keys.device, dtype=torch.long))
        if eff_last > 0:
            must_keep_list.append(torch.arange(k_len - eff_last, k_len, device=keys.device, dtype=torch.long))
        must_keep = torch.unique(torch.cat(must_keep_list)) if must_keep_list else torch.empty((0,), device=keys.device, dtype=torch.long)

        # 8) build final indices per batch:
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

                # rank by head-specific score
                if cand.numel() > 0 and need > 0:
                    score = sel_scores[b, h].index_select(0, cand)
                    order = torch.argsort(score, descending=True)
                    cand = cand.index_select(0, order)

                # fallback global topk for this head if insufficient
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
                    cand = torch.cat([cand, extra], dim=0)
                    cand = torch.unique(cand)

                    score = sel_scores[b, h].index_select(0, cand)
                    order = torch.argsort(score, descending=True)
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
        gather_idx = indices_h.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B,H,t_keep,D]
        keys_new = keys.gather(2, gather_idx).contiguous()
        values_new = values.gather(2, gather_idx).contiguous()


        # 10) gather credit to match new tokens (if enabled)
        if self.use_credit:
            credit = self._credit_by_layer.get(layer_idx, None)
            if credit is not None:
                # indices_h: [B, H, t_keep]
                credit_new = credit.gather(2, indices_h)   # [B, H, t_keep]
                self._credit_by_layer[layer_idx] = credit_new


        # 11) record stats + optional visualization (single figure)
        if self.record_segment_stats:
            with torch.no_grad():
                # Record head 0 as a representative example.
                h0 = 0
                segments_h0 = segments_per_head[h0]
                budgets_h0 = budgets_per_head[h0]
                mass_1d_h0 = mass_mean[h0]          # [T]
                kept0 = indices_h[0, h0]

                kept_mask0 = torch.zeros((k_len,), device=keys.device, dtype=torch.bool)
                kept_mask0[kept0] = True

                seg_stats = []
                total_mass = float(mass_1d_h0.sum().item() + 1e-12)
                for seg_id, ((s, e), bud) in enumerate(zip(segments_h0, budgets_h0)):
                    L = e - s
                    if L <= 0:
                        continue
                    seg_m = float(mass_1d_h0[s:e].sum().item()) / total_mass
                    seg_kept = int(kept_mask0[s:e].sum().item())
                    seg_stats.append(
                        {
                            "segment_id": int(seg_id),
                            "start": int(s),
                            "end": int(e),
                            "length": int(L),
                            "seg_mass": float(seg_m),
                            "budget": int(bud),
                            "kept_count_batch0": int(seg_kept),
                            "keep_ratio_batch0": float(seg_kept / max(L, 1)),
                        }
                    )

                mass_ds = self._downsample_1d(mass_1d_h0.detach())
                kept_ds = kept0.detach().clone()
                if mass_1d_h0.numel() > self.record_max_points:
                    scale = float(mass_1d_h0.numel() - 1) / float(max(mass_ds.numel() - 1, 1))
                    kept_ds = (kept_ds.float() / scale).round().long()

                self._press_records.append(
                    {
                        "type": "adaptive_mass_segment_headwise",
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
                        "mass_curve_downsampled": mass_ds.detach().float().cpu().tolist(),
                        "kept_indices_batch0_head0": kept_ds.detach().cpu().tolist(),
                    }
                )


        if self.visualize and layer_idx == int(self.visualize_layer):
            if (event_id % int(self.visualize_every)) == 0:
                h0 = 0
                mass_1d_h0 = mass_mean[h0]
                segments_h0 = segments_per_head[h0]
                kept0 = indices_h[0, h0]
                self._visualize_once(layer_idx, event_id, mass_1d_h0, segments_h0, kept0)

        return keys_new, values_new
