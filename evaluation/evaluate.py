# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation entry point for AMS decoding-time KV compression.

The public release keeps this script intentionally focused on the paper setup:
Hugging Face Transformers + KVPress hooks + math reasoning datasets.  Historical
private experiment grids, vLLM prototypes, and quantization scouts were removed
so that ``python -m evaluation.evaluate`` behaves like a normal CLI.
"""

from __future__ import annotations

import glob
import json
import logging
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from datasets import load_dataset
from datasets.exceptions import DatasetGenerationError
from fire import Fire
from tqdm import tqdm
from transformers import DynamicCache, pipeline

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "evaluation"

from .evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from .method_registry import METHOD_REGISTRY
from .method_runner import build_cache_from_method, build_press_from_method, dispatch_method, get_method_spec

from kvpress import DecodingPress

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = {
    "math500": 4096,
    "aime24": 32768,
    "aime25": 32768,
    "gsm8k": 1024,
}


def _clean_dict(values: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


def _read_parquet_file_or_dir(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.is_file():
        files = [path]
        root = path.parent
    elif path.is_dir():
        files = [Path(x) for x in sorted(glob.glob(str(path / "**" / "*.parquet"), recursive=True))]
        root = path
    else:
        raise FileNotFoundError(f"Parquet path not found: {path_str}")

    dfs = []
    for file_path in files:
        one = pq.read_table(str(file_path)).to_pandas()
        try:
            rel = file_path.relative_to(root)
            one["subset"] = rel.parts[0] if len(rel.parts) > 1 else rel.stem
        except Exception:
            one["subset"] = file_path.parent.name
        dfs.append(one)
    return pd.concat(dfs, ignore_index=True)


def _load_yaml_config(path: str | Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        logger.warning("Config file not found at %s. Using CLI/default values.", path)
        return {}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_main_metric(metrics: Any) -> float:
    if isinstance(metrics, dict):
        for key in ("accuracy", "pass@1", "score", "correct"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        overall = metrics.get("overall")
        if isinstance(overall, (int, float)):
            return float(overall)
        if isinstance(overall, dict):
            return _load_main_metric(overall)
    return float("nan")


def _upsert_summary(summary_path: Path, row: dict, key_cols: list[str]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        df = pd.DataFrame([row])
    else:
        for key in row:
            if key not in df.columns:
                df[key] = np.nan
        mask = np.ones(len(df), dtype=bool)
        for key in key_cols:
            if key in df.columns:
                mask &= df[key].astype(str) == str(row.get(key, ""))
        if mask.any():
            idx = np.where(mask)[0][0]
            for key, value in row.items():
                df.at[idx, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(summary_path, index=False)


@dataclass
class EvaluationConfig:
    dataset: str = "math500"
    data_dir: Optional[str] = None
    model: str = ""
    device: Optional[str] = None

    # Use either method_name (recommended) or press_name.
    method_name: Optional[str] = None
    press_name: str = "decoding_adaptive_segment_head_expected_attention"
    compression_ratio: float = 1.0

    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    compress_questions: bool = False

    compression_interval: Optional[int] = 512
    target_size: Optional[int] = 1024
    hidden_states_buffer_size: Optional[int] = None

    output_dir: str = "./results"
    log_level: str = "INFO"
    model_kwargs: Optional[Dict[str, Any]] = None
    seed: int = 42

    resume_mode: str = "validate"  # "skip" | "validate" | "force"
    write_done_flag: bool = True
    write_failed_flag: bool = True
    write_summary: bool = True
    summary_csv: Optional[str] = None
    record_time: bool = True
    record_peak_memory: bool = True

    press_init_command: Optional[str] = None

    def __post_init__(self):
        if self.dataset not in DATASET_REGISTRY:
            raise KeyError(f"Unknown dataset={self.dataset!r}. Available: {sorted(DATASET_REGISTRY)}")
        if self.dataset not in SCORER_REGISTRY:
            raise KeyError(f"No scorer registered for dataset={self.dataset!r}")
        if self.method_name is not None and self.method_name not in METHOD_REGISTRY:
            raise KeyError(f"Unknown method_name={self.method_name!r}. Available: {sorted(METHOD_REGISTRY)}")
        if self.method_name is None and self.press_name not in PRESS_REGISTRY:
            raise KeyError(f"Unknown press_name={self.press_name!r}. Available: {sorted(PRESS_REGISTRY)}")
        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.method_name is None and self.press_name == "no_press":
            self.compression_ratio = 0.0
        if not (0.0 <= self.compression_ratio <= 1.0):
            raise ValueError("compression_ratio must be in [0, 1]")
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError("fraction must be in (0, 1]")
        if self.resume_mode not in {"skip", "validate", "force"}:
            raise ValueError("resume_mode must be one of: skip, validate, force")

    def _dir_components(self) -> list[str]:
        method_or_press = self.method_name or self.press_name
        components = [
            self.dataset,
            str(self.data_dir) if self.data_dir else "",
            self.model.replace("/", "--"),
            method_or_press,
            f"{self.compression_ratio:.2f}",
        ]
        if self.target_size is not None:
            components.append(f"target{int(self.target_size)}")
        if self.compression_interval is not None:
            components.append(f"interval{int(self.compression_interval)}")
        if self.hidden_states_buffer_size is not None:
            components.append(f"hsbuf{int(self.hidden_states_buffer_size)}")
        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        return components

    def get_experiment_dir(self, output_dir: Path) -> Path:
        return output_dir / "__".join(filter(None, self._dir_components()))

    def get_results_dir(self, output_dir: Path) -> Path:
        base = self.get_experiment_dir(output_dir)
        if not base.exists():
            base.mkdir(parents=True)
            return base
        index = 1
        while (base / str(index)).exists():
            index += 1
        path = base / str(index)
        path.mkdir(parents=True)
        return path

    def save_config(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(self), handle, sort_keys=False)


class EvaluationRunner:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.pipeline = None
        self.press = None
        self.method_spec = None
        self.df: Optional[pd.DataFrame] = None
        self._setup_logging()
        self._setup_seeds()
        logger.info("Initialized evaluation with config:\n%s", json.dumps(asdict(config), indent=2))

    def _setup_logging(self) -> None:
        level = self.config.log_level.upper()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(level)

    def _setup_seeds(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _setup_press(self) -> None:
        runner_type = dispatch_method(self.config)
        if runner_type == "kvpress_hf":
            self.method_spec = get_method_spec(self.config.method_name)  # type: ignore[arg-type]
            self.press = build_press_from_method(self.config, self.method_spec)
        else:
            import copy

            self.press = copy.deepcopy(PRESS_REGISTRY[self.config.press_name])
            if isinstance(self.press, DecodingPress):
                if self.config.compression_interval is not None:
                    self.press.compression_interval = self.config.compression_interval
                if self.config.target_size is not None:
                    self.press.target_size = self.config.target_size
                if self.config.hidden_states_buffer_size is not None:
                    self.press.hidden_states_buffer_size = self.config.hidden_states_buffer_size
            elif hasattr(self.press, "compression_ratio"):
                self.press.compression_ratio = self.config.compression_ratio

        self.config.press_init_command = str(self.press)
        logger.info("Prepared press: %s", self.config.press_init_command)

    def _make_cache(self):
        if self.method_spec is not None:
            return build_cache_from_method(self.method_spec)
        if isinstance(self.press, DecodingPress):
            return DynamicCache()
        return None

    def _setup_model_pipeline(self) -> None:
        if not self.config.model:
            raise ValueError("config.model is required.")

        device = self.config.device
        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"

        model_kwargs = dict(self.config.model_kwargs or {})
        model_kwargs.setdefault("dtype", "auto")

        logger.info("Loading model=%s device=%s model_kwargs=%s", self.config.model, device, model_kwargs)
        kwargs = {
            "model": self.config.model,
            "model_kwargs": _clean_dict(model_kwargs),
            "trust_remote_code": True,
        }
        if device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device"] = device
        self.pipeline = pipeline("kv-press-text-generation", **kwargs)
        self.pipeline.model.eval()

    def _load_dataset(self) -> None:
        dataset_name = self.config.dataset
        ds_spec = DATASET_REGISTRY[dataset_name]
        data_dir = self.config.data_dir
        if dataset_name == "gsm8k" and data_dir is None and ds_spec == "openai/gsm8k":
            data_dir = "main"

        logger.info("Loading dataset=%s spec=%s data_dir=%s", dataset_name, ds_spec, data_dir)
        if ds_spec.endswith(".parquet") or (Path(ds_spec).exists() and Path(ds_spec).is_dir()):
            df = _read_parquet_file_or_dir(ds_spec)
        elif ds_spec.endswith(".jsonl") or ds_spec.endswith(".json"):
            df = load_dataset("json", data_files={"test": ds_spec}, split="test").to_pandas()
        elif ds_spec.endswith(".csv"):
            df = load_dataset("csv", data_files={"test": ds_spec}, split="test").to_pandas()
        else:
            try:
                df = load_dataset(ds_spec, data_dir=data_dir, split="test").to_pandas()
            except DatasetGenerationError:
                ds = load_dataset(ds_spec, data_dir=data_dir, split="test", streaming=True)
                df = pd.DataFrame(list(ds))

        if "context" not in df.columns:
            df["context"] = ""
        if "question" not in df.columns:
            for alias in ("problem", "prompt", "input", "query", "question_text"):
                if alias in df.columns:
                    df["question"] = df[alias].astype(str)
                    break
            else:
                raise KeyError(f"Dataset has no question/problem/prompt column. columns={list(df.columns)}")
        if "answer_prefix" not in df.columns:
            df["answer_prefix"] = ""
        if "max_new_tokens" not in df.columns:
            df["max_new_tokens"] = int(self.config.max_new_tokens or DEFAULT_MAX_NEW_TOKENS.get(dataset_name, 512))

        if self.config.fraction < 1.0:
            task_col = next((c for c in ("dataset", "task", "sub_dataset", "subset", "source") if c in df.columns), None)
            if task_col is None:
                df = df.sample(frac=self.config.fraction, random_state=self.config.seed)
            else:
                df = (
                    df.groupby(task_col, group_keys=False)
                    .apply(lambda x: x.sample(frac=self.config.fraction, random_state=self.config.seed))
                    .reset_index(drop=True)
                )

        if self.config.compress_questions:
            df["context"] = df["context"].astype(str) + df["question"].astype(str)
            df["question"] = ""

        self.df = df.reset_index(drop=True)
        logger.info("Loaded %d examples.", len(self.df))

    @torch.inference_mode()
    def _run_inference(self) -> None:
        if self.df is None or self.pipeline is None:
            raise RuntimeError("Dataset and model must be loaded before inference.")

        self.df["predicted_answer"] = None
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running inference"):
            if hasattr(self.press, "reset"):
                self.press.reset()
            output = self.pipeline(
                row["context"],
                question=row["question"],
                answer_prefix=row["answer_prefix"],
                press=self.press,
                cache=self._make_cache(),
                max_new_tokens=int(self.config.max_new_tokens or row["max_new_tokens"]),
                max_context_length=self.config.max_context_length,
            )
            self.df.loc[index, "predicted_answer"] = output["answer"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _save_results(self, path: Path) -> None:
        assert self.df is not None
        columns = [c for c in self.df.columns if c != "context"]
        self.df[columns].to_csv(path, index=False)
        logger.info("Saved predictions to %s", path)

    def _calculate_and_save_metrics(self, path: Path) -> dict:
        assert self.df is not None
        metrics = SCORER_REGISTRY[self.config.dataset](self.df)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        logger.info("Metrics: %s", json.dumps(metrics, indent=2))
        return metrics

    def run_evaluation(self) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_dir = (
            self.config.get_results_dir(output_dir)
            if self.config.resume_mode == "force"
            else self.config.get_experiment_dir(output_dir)
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = results_dir / "predictions.csv"
        metrics_path = results_dir / "metrics.json"
        config_path = results_dir / "config.yaml"
        done_path = results_dir / ".DONE"
        failed_path = results_dir / ".FAILED"

        if self.config.resume_mode in {"skip", "validate"} and done_path.exists():
            logger.info("Found %s; skipping existing run.", done_path)
            return

        t0 = time.time() if self.config.record_time else None
        peak_alloc_gb = float("nan")

        try:
            self._setup_press()
            self._setup_model_pipeline()
            self._load_dataset()

            if self.config.record_peak_memory and self.config.device and str(self.config.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(self.config.device))

            self._run_inference()
            metrics = self._calculate_and_save_metrics(metrics_path)
            self._save_results(predictions_path)
            self.config.save_config(config_path)

            if self.config.record_peak_memory and self.config.device and str(self.config.device).startswith("cuda"):
                peak_alloc_gb = torch.cuda.max_memory_allocated(torch.device(self.config.device)) / 1024**3
            total_time = time.time() - t0 if t0 is not None else float("nan")

            if self.config.write_done_flag:
                _write_text(done_path, "ok\n")
            if failed_path.exists():
                failed_path.unlink()

            if self.config.write_summary:
                num_samples = len(self.df) if self.df is not None else 0
                summary_path = Path(self.config.summary_csv) if self.config.summary_csv else output_dir / "summary.csv"
                row = {
                    "dataset": self.config.dataset,
                    "model": self.config.model,
                    "method_name": self.config.method_name or "",
                    "press_name": self.config.press_name,
                    "target_size": self.config.target_size,
                    "compression_interval": self.config.compression_interval,
                    "fraction": self.config.fraction,
                    "num_samples": num_samples,
                    "metric_main": _load_main_metric(metrics),
                    "total_time_sec": total_time,
                    "avg_time_per_sample_sec": total_time / num_samples if num_samples else float("nan"),
                    "peak_alloc_gb": peak_alloc_gb,
                    "results_dir": str(results_dir),
                }
                _upsert_summary(
                    summary_path,
                    row,
                    key_cols=["dataset", "model", "method_name", "press_name", "target_size", "fraction"],
                )
        except Exception:
            if self.config.write_failed_flag:
                _write_text(failed_path, traceback.format_exc())
            raise


class CliEntryPoint:
    def __call__(self, config_file: Optional[str] = None, **cli_overrides):
        config_path = config_file or str(Path(__file__).with_name("evaluate_config.yaml"))
        args = asdict(EvaluationConfig())
        args.update(_load_yaml_config(config_path))
        args.update({k: v for k, v in cli_overrides.items() if v is not None})
        EvaluationRunner(EvaluationConfig(**args)).run_evaluation()


if __name__ == "__main__":
    Fire(CliEntryPoint())
