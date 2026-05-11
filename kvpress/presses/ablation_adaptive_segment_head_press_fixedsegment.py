# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .adaptive_segment_head_press import AdaptiveMassSegmentWrapperHeadPress


@dataclass
class FixedSegmentWrapperHeadPress(AdaptiveMassSegmentWrapperHeadPress):
    """
    Fixed-segment ablation for AMS.

    It reuses scoring, EMA credit, quota allocation, and in-segment top-k from
    `AdaptiveMassSegmentWrapperHeadPress`, but replaces mass-derived adaptive
    boundaries with fixed-length segments.
    """

    fixed_seg_len: int = 128

    def _build_segments_by_mass_searchsorted(self, mass_1d: torch.Tensor, k_len: int) -> List[Tuple[int, int]]:
        """Ignore the mass curve and partition [0, k_len) by fixed length."""
        L = int(self.fixed_seg_len)
        assert L >= 1, "fixed_seg_len must be >= 1"

        segments: List[Tuple[int, int]] = []
        start = 0
        while start < k_len:
            end = min(start + L, k_len)
            segments.append((start, end))
            start = end
        return segments
