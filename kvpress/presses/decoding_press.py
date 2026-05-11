# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.cache_utils import QuantizedCache

from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.base_press import BasePress
from kvpress.presses.chunkkv_press import ChunkKVPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class DecodingPress(BasePress):
    """
    Apply a scorer-style KV compressor periodically during autoregressive decoding.

    `DecodingPress` keeps a short buffer of recent hidden states for each layer,
    invokes the wrapped base press every `compression_interval` generated tokens,
    and physically gathers the KV cache to `target_size` tokens.
    """

    base_press: ScorerPress | AdaKVPress | ChunkKVPress
    compression_interval: int = 512
    target_size: int = 2048
    hidden_states_buffer_size: int = 256

    # Optional length-based trigger for benchmarks where generation can grow
    # faster than the step counter captures.
    trigger_on_k_len: bool = False
    k_len_trigger_margin: int = 0
    min_steps_before_k_len_trigger: int = 0

    def __post_init__(self):
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)

        assert self.compression_interval > 0, "compression_interval must be greater than 0"
        assert self.target_size > 0, "target_size must be greater than 0"

        if getattr(self.base_press, "compression_ratio", 0):
            logger.warning(
                "base_press.compression_ratio=%s will be overridden by DecodingPress.",
                self.base_press.compression_ratio,
            )

    def post_init_from_model(self, model):
        self.base_press.post_init_from_model(model)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delegate one compression event to the wrapped base press."""
        k_len = keys.shape[2]
        target_compression_ratio = self._find_target_compression_ratio(k_len, self.target_size)
        logger.debug("Compressing %s tokens to %s with ratio %.6f", k_len, self.target_size, target_compression_ratio)

        original_compression_ratio = self.base_press.compression_ratio
        self.base_press.compression_ratio = target_compression_ratio
        result = self.base_press.compress(module, hidden_states, keys, values, attentions, kwargs)
        self.base_press.compression_ratio = original_compression_ratio
        return result

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Accumulate decoding hidden states and compress the current layer when due.

        The hook skips prefill calls, stores recent decoding hidden states, and
        clears the buffer after a successful cache gather so the hidden-state
        window remains aligned with the physical KV cache.
        """
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx

        if self._is_prefill_call(kwargs, q_len):
            return output

        self.hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())

        keys, values = extract_keys_and_values(cache, module.layer_idx)
        k_len = keys.shape[2]

        self.layer_step_counts[layer_idx] += 1
        step_count = self.layer_step_counts[layer_idx]

        interval_trigger = step_count >= self.compression_interval
        target_trigger = q_len >= self.target_size
        k_len_threshold = self.target_size + self.k_len_trigger_margin
        length_trigger = (
            self.trigger_on_k_len
            and k_len >= k_len_threshold
            and step_count >= self.min_steps_before_k_len_trigger
        )

        if interval_trigger or target_trigger or length_trigger:
            logger.debug(
                "[DecodingPress] layer=%s step=%s q_len=%s k_len=%s target=%s interval=%s length_trigger=%s",
                layer_idx,
                step_count,
                q_len,
                k_len,
                self.target_size,
                self.compression_interval,
                length_trigger,
            )

            cache_layer = cache.layers[module.layer_idx]
            k_len_before = int(keys.shape[2])
            attentions = output[1] if len(output) > 1 and output[1] is not None else None

            buffered_hidden_states = torch.cat(self.hidden_states_buffer[layer_idx], dim=1)
            keys, values = self.compress(module, buffered_hidden_states, keys, values, attentions, kwargs)

            logger.debug(
                "[DecodingPress] layer=%s cache length %s -> %s",
                layer_idx,
                k_len_before,
                int(keys.shape[2]),
            )

            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
                cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
                cache_layer.cumulative_length = keys.shape[2]
            else:
                cache_layer.keys = keys
                cache_layer.values = values

            self.layer_step_counts[layer_idx] = 0
            self.hidden_states_buffer[layer_idx] = []

        self.hidden_states_buffer[layer_idx] = (
            self.hidden_states_buffer[layer_idx][-self.hidden_states_buffer_size :]
            if self.hidden_states_buffer_size > 0
            else []
        )
        return output

    def reset(self):
        """Reset per-context state before a new sample."""
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)
        if hasattr(self.base_press, "reset"):
            self.base_press.reset()

    @staticmethod
    def _is_prefill_call(kwargs: dict, q_len: int) -> bool:
        if "cache_position" in kwargs and kwargs["cache_position"] is not None:
            return bool(kwargs["cache_position"][-1] < q_len)
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            return bool(kwargs["position_ids"][0, -1] < q_len)
        return q_len > 1

    def _find_target_compression_ratio(self, q_len: int, target_tokens: int) -> float:
        """Find a ratio whose integer rounding keeps exactly `target_tokens`."""
        if q_len <= target_tokens:
            return 0.0

        ratio = 1.0 - (target_tokens / q_len)
        low, high = 0.0, 1.0

        for _ in range(20):
            n_kept = int(q_len * (1 - ratio))
            if n_kept == target_tokens:
                break
            if n_kept > target_tokens:
                low = ratio
                ratio = (ratio + high) / 2
            else:
                high = ratio
                ratio = (low + ratio) / 2

        final_n_kept = int(q_len * (1 - ratio))
        if final_n_kept != target_tokens:
            logger.warning(
                "Binary search failed: q_len=%s target=%s got=%s ratio=%s",
                q_len,
                target_tokens,
                final_n_kept,
                ratio,
            )

        return ratio
