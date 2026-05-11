# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Method aliases for reproducible paper runs.

The evaluator accepts either ``press_name`` directly or a higher-level
``method_name``.  ``method_name`` is just a stable alias that points to a
registered press and keeps command lines short in scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional


RunnerType = Literal["kvpress_hf"]
CacheType = Literal["dynamic"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    runner: RunnerType
    press_name: Optional[str] = None
    cache_type: CacheType = "dynamic"


METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "full_kv": MethodSpec("full_kv", runner="kvpress_hf", press_name="no_press"),
    "streaming_llm": MethodSpec("streaming_llm", runner="kvpress_hf", press_name="decoding_streaming_llm"),
    "tova": MethodSpec("tova", runner="kvpress_hf", press_name="decoding_tova"),
    "keydiff": MethodSpec("keydiff", runner="kvpress_hf", press_name="decoding_keydiff"),
    "pyramidkv": MethodSpec("pyramidkv", runner="kvpress_hf", press_name="decoding_pyramidkv"),
    "adakv_expe2": MethodSpec(
        "adakv_expe2",
        runner="kvpress_hf",
        press_name="decoding_adakv_expected_attention_e2",
    ),
    "chunkkv_expected": MethodSpec(
        "chunkkv_expected",
        runner="kvpress_hf",
        press_name="decoding_chunkkv_expected_attention",
    ),
    "rkv": MethodSpec("rkv", runner="kvpress_hf", press_name="decoding_rkv"),
    "rpc": MethodSpec("rpc", runner="kvpress_hf", press_name="decoding_rpc"),
    "ams_tova": MethodSpec(
        "ams_tova",
        runner="kvpress_hf",
        press_name="decoding_adaptivesegmenthead_tova",
    ),
    "ams_expected": MethodSpec(
        "ams_expected",
        runner="kvpress_hf",
        press_name="decoding_adaptive_segment_head_expected_attention",
    ),
    "ams_keydiff": MethodSpec("ams_keydiff", runner="kvpress_hf", press_name="decoding_ams_keydiff"),
    "ams_rkv": MethodSpec("ams_rkv", runner="kvpress_hf", press_name="decoding_ams_rkv"),

    # Ablations.
    "ams_tova_global_head": MethodSpec(
        "ams_tova_global_head",
        runner="kvpress_hf",
        press_name="decoding_adaptive_global_head_tova",
    ),
    "ablation_global_head_tova": MethodSpec(
        "ablation_global_head_tova",
        runner="kvpress_hf",
        press_name="decoding_adaptive_global_head_tova",
    ),
    "ams_tova_fixed_segment": MethodSpec(
        "ams_tova_fixed_segment",
        runner="kvpress_hf",
        press_name="decoding_adaptivesegmenthead_tova_fixedsegment",
    ),
    "ablation_fixed_segment_tova": MethodSpec(
        "ablation_fixed_segment_tova",
        runner="kvpress_hf",
        press_name="decoding_adaptivesegmenthead_tova_fixedsegment",
    ),
    "ams_tova_no_credit": MethodSpec(
        "ams_tova_no_credit",
        runner="kvpress_hf",
        press_name="decoding_adaptivesegmenthead_tova_no_credit",
    ),
    "ablation_no_credit_tova": MethodSpec(
        "ablation_no_credit_tova",
        runner="kvpress_hf",
        press_name="decoding_adaptivesegmenthead_tova_no_credit",
    ),
    "ams_tova_no_mass": MethodSpec(
        "ams_tova_no_mass",
        runner="kvpress_hf",
        press_name="decoding_adaptivesegmenthead_tova_no_mass",
    ),
    "ablation_no_mass_tova": MethodSpec(
        "ablation_no_mass_tova",
        runner="kvpress_hf",
        press_name="decoding_adaptivesegmenthead_tova_no_mass",
    ),
}
