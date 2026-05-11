# Evaluation

This directory contains the public evaluation entry point for the AMS release.
It is intentionally small: method construction lives in `evaluate_registry.py`
and stable paper aliases live in `method_registry.py`.

## Basic Command

```bash
python -m evaluation.evaluate \
  --model /path/to/model \
  --dataset math500 \
  --method_name ams_expected \
  --target_size 512 \
  --compression_interval 512 \
  --device cuda:0
```

The CLI also accepts a YAML config:

```bash
python -m evaluation.evaluate --config evaluation/evaluate_config.yaml
```

Command-line arguments override config values.

## Datasets

Supported dataset keys:

```text
math500
aime24
aime25
gsm8k
```

Set local paths with environment variables when you do not want to rely on the
default public dataset IDs:

```bash
export MATH500_DATA=/path/to/math500.parquet
export AIME24_DATA=/path/to/aime24.jsonl
export AIME25_DATA=/path/to/aime25.jsonl
export GSM8K_DATA=/path/to/gsm8k
```

Local files may be `.json`, `.jsonl`, `.csv`, or `.parquet`.

## Methods

Prefer `--method_name` over raw `--press_name`. Current stable names:

```text
full_kv
streaming_llm
tova
keydiff
pyramidkv
adakv_expe2
chunkkv_expected
rkv
rpc
ams_tova
ams_expected
ams_keydiff
ams_rkv
ablation_global_head_tova
ablation_fixed_segment_tova
ablation_no_credit_tova
ablation_no_mass_tova
```

The evaluator maps these aliases to `DecodingPress` instances and overwrites
`target_size`, `compression_interval`, and `hidden_states_buffer_size` from the
CLI/config.

## Outputs

Each run writes:

```text
predictions.csv
metrics.json
config.yaml
.DONE or .FAILED
```

With `write_summary: true`, a compact `summary.csv` is also updated under the
output directory.
