# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public evaluation registries for the AMS paper release.

This file intentionally contains only the datasets and methods used by the
paper-facing evaluation scripts. Local exploratory methods were removed so that
importing the registry is side-effect free and does not depend on private paths.
"""

from __future__ import annotations

import os
from copy import deepcopy

from .benchmarks.aime25.calculate_metrics import calculate_metrics as aime_scorer
from .benchmarks.gsm8k.calculate_metrics import calculate_metrics as gsm8k_scorer
from .benchmarks.math500.calculate_metrics import calculate_metrics as math500_scorer

from kvpress import (
    AdaKVPress,
    ChunkKVPress,
    DecodingPress,
    ExpectedAttentionPress,
    KeyDiffPress,
    KnormPress,
    PyramidKVPress,
    StreamingLLMPress,
    TOVAPress,
)
from kvpress.presses.ablation_adaptive_global_head_press import AdaptiveGlobalHeadPress
from kvpress.presses.ablation_adaptive_segment_head_press_fixedsegment import FixedSegmentWrapperHeadPress
from kvpress.presses.ablation_adaptive_segment_head_press_no_mass import AdaptiveMassSegmentWrapperHeadPress_NoMass
from kvpress.presses.ablation_adaptive_segment_no_credit import AdaptiveMassSegmentWrapperHeadPress_No_credit
from kvpress.presses.adaptive_segment_head_press import AdaptiveMassSegmentWrapperHeadPress
from kvpress.presses.reasoning_path_press import ReasoningPathPress
from kvpress.presses.rkv_press import RKVPress


def _env_or_default(env_name: str, default: str) -> str:
    """Resolve a dataset path from an env var while keeping a portable default."""
    return os.environ.get(env_name, default)


# For local reproduction, set these environment variables to parquet/json/jsonl
# files or directories. If unset, the loader falls back to public HF dataset IDs
# for the datasets where stable IDs are available.
DATASET_REGISTRY = {
    "math500": _env_or_default("MATH500_DATA", "HuggingFaceH4/MATH-500"),
    "aime24": _env_or_default("AIME24_DATA", "./data/aime24.jsonl"),
    "aime25": _env_or_default("AIME25_DATA", "opencompass/AIME2025"),
    "gsm8k": _env_or_default("GSM8K_DATA", "openai/gsm8k"),
}

SCORER_REGISTRY = {
    "math500": math500_scorer,
    "aime24": aime_scorer,
    "aime25": aime_scorer,
    "gsm8k": gsm8k_scorer,
}


AMS_DEFAULTS = {
    "window_size": 32,
    "segment_mass": 0.02,
    "min_seg_len": 32,
    "max_seg_len": 1024,
    "min_keep_per_segment": 8,
    "n_sink": 4,
    "always_keep_last": 32,
    "sort_indices": True,
    "use_credit": True,
    "credit_decay": 0.8,
    "mix_beta": 0.7,
    "record_segment_stats": False,
    "visualize": False,
}


def _decoding(base_press, *, hs_buffer: int = 256) -> DecodingPress:
    """Create a decoding-time compressor with paper defaults.

    The evaluator overwrites ``target_size`` and ``compression_interval`` from
    CLI/config values, so the registry only fixes the base selection policy and
    the hidden-state buffer required by attention-based scorers.
    """
    return DecodingPress(
        base_press=base_press,
        compression_interval=512,
        target_size=1024,
        hidden_states_buffer_size=hs_buffer,
    )


def _ams(base_press, **overrides) -> AdaptiveMassSegmentWrapperHeadPress:
    kwargs = deepcopy(AMS_DEFAULTS)
    kwargs.update(overrides)
    return AdaptiveMassSegmentWrapperHeadPress(base_press=base_press, compression_ratio=0.0, **kwargs)


PRESS_REGISTRY = {
    # Generic / no-compression entries.
    "no_press": None,
    "knorm": KnormPress(),
    "expected_attention": ExpectedAttentionPress(epsilon=1e-2),

    # Main decoding-time baselines.
    "decoding_streaming_llm": _decoding(StreamingLLMPress(), hs_buffer=0),
    "decoding_tova": _decoding(TOVAPress(), hs_buffer=0),
    "decoding_keydiff": _decoding(KeyDiffPress(), hs_buffer=0),
    "decoding_pyramidkv": _decoding(PyramidKVPress(), hs_buffer=256),
    "decoding_adakv_expected_attention_e2": _decoding(
        AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
        hs_buffer=256,
    ),
    "decoding_chunkkv_expected_attention": _decoding(
        ChunkKVPress(press=ExpectedAttentionPress(epsilon=1e-2), chunk_length=20),
        hs_buffer=256,
    ),
    "decoding_rkv": _decoding(
        RKVPress(window_size=8, kernel_size=7, mix_lambda=0.07, retain_ratio=0.1),
        hs_buffer=256,
    ),
    "decoding_rpc": _decoding(
        ReasoningPathPress(window_size=32, kernel_size=7),
        hs_buffer=64,
    ),

    # AMS variants reported in the paper.
    "decoding_adaptivesegmenthead_tova": _decoding(_ams(TOVAPress()), hs_buffer=256),
    "decoding_adaptive_segment_head_expected_attention": _decoding(
        _ams(ExpectedAttentionPress(epsilon=1e-2)),
        hs_buffer=256,
    ),
    "decoding_ams_keydiff": _decoding(_ams(KeyDiffPress()), hs_buffer=256),
    "decoding_ams_rkv": _decoding(
        _ams(RKVPress(window_size=8, kernel_size=7, mix_lambda=0.07, retain_ratio=0.1)),
        hs_buffer=256,
    ),

    # Ablations used for the component analysis table.
    "decoding_adaptive_global_head_tova": _decoding(
        AdaptiveGlobalHeadPress(base_press=TOVAPress(compression_ratio=0.0)),
        hs_buffer=256,
    ),
    "decoding_adaptivesegmenthead_tova_fixedsegment": _decoding(
        FixedSegmentWrapperHeadPress(base_press=TOVAPress(compression_ratio=0.0), **AMS_DEFAULTS),
        hs_buffer=256,
    ),
    "decoding_adaptivesegmenthead_tova_no_credit": _decoding(
        AdaptiveMassSegmentWrapperHeadPress_No_credit(
            base_press=TOVAPress(compression_ratio=0.0),
            **{**AMS_DEFAULTS, "use_credit": False},
        ),
        hs_buffer=256,
    ),
    "decoding_adaptivesegmenthead_tova_no_mass": _decoding(
        AdaptiveMassSegmentWrapperHeadPress_NoMass(base_press=TOVAPress(compression_ratio=0.0), **AMS_DEFAULTS),
        hs_buffer=256,
    ),
}

# Backward-compatible aliases for names that appeared in early drafts.
PRESS_REGISTRY.update(
    {
        "decoding_TOVA": PRESS_REGISTRY["decoding_tova"],
        "decoding_adaptivesegmenthead_TOVA": PRESS_REGISTRY["decoding_adaptivesegmenthead_tova"],
        "decoding_adaptiveGlobalHeadPress": PRESS_REGISTRY["decoding_adaptive_global_head_tova"],
        "decoding_AdaptiveMassSegmentWrapperHeadPress_No_credit": PRESS_REGISTRY[
            "decoding_adaptivesegmenthead_tova_no_credit"
        ],
        "decoding_AdaptiveMassSegmentWrapperHeadPress_No_Mass": PRESS_REGISTRY[
            "decoding_adaptivesegmenthead_tova_no_mass"
        ],
    }
)
