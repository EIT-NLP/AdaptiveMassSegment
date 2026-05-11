# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small adapter layer for ``method_name`` based evaluation."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from transformers import DynamicCache

from .evaluate_registry import PRESS_REGISTRY
from .method_registry import METHOD_REGISTRY, MethodSpec


def get_method_spec(method_name: str) -> MethodSpec:
    if method_name not in METHOD_REGISTRY:
        raise KeyError(
            f"Unknown method_name={method_name!r}. Available methods: {sorted(METHOD_REGISTRY)}"
        )
    return METHOD_REGISTRY[method_name]


def resolve_method_extra_config(config: Any, method: Optional[MethodSpec]) -> Dict[str, Any]:
    """Compatibility hook kept for the evaluator; paper methods need no extras."""
    return {}


def build_cache_from_method(method: MethodSpec, model_config: Optional[Any] = None):
    """Return a fresh cache object for one sample/group."""
    if method.cache_type != "dynamic":
        raise ValueError(f"Unsupported cache_type={method.cache_type!r}")
    return DynamicCache()


def build_press_from_method(config: Any, method: MethodSpec):
    if method.press_name is None:
        return None
    if method.press_name not in PRESS_REGISTRY:
        raise KeyError(
            f"press_name={method.press_name!r} not found. Available presses: {sorted(PRESS_REGISTRY)}"
        )

    press = copy.deepcopy(PRESS_REGISTRY[method.press_name])
    if press is None:
        return None

    # DecodingPress parameters are intentionally controlled by the evaluator so
    # every method is compared under the same cache budget and interval.
    if hasattr(press, "compression_interval") and getattr(config, "compression_interval", None) is not None:
        press.compression_interval = config.compression_interval
    if hasattr(press, "target_size") and getattr(config, "target_size", None) is not None:
        press.target_size = config.target_size
    if hasattr(press, "hidden_states_buffer_size") and getattr(config, "hidden_states_buffer_size", None) is not None:
        press.hidden_states_buffer_size = config.hidden_states_buffer_size
    if hasattr(press, "compression_ratio"):
        press.compression_ratio = getattr(config, "compression_ratio", 1.0)
    return press


def apply_method_model_patch(model: Any, config: Any, method: Optional[MethodSpec]) -> None:
    """No-op retained for API compatibility with older experiment scripts."""
    return None


def prepare_kvpress_hf_components(config: Any):
    if not getattr(config, "method_name", None):
        raise ValueError("config.method_name is required for method-based execution.")
    method = get_method_spec(config.method_name)
    return method, build_press_from_method(config, method), build_cache_from_method(method)


def dispatch_method(config: Any) -> str:
    return "kvpress_hf" if getattr(config, "method_name", None) else "legacy_press"
