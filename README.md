# AMS KVPress

Adaptive Mass-Segmented KV compression for long-context reasoning.

This repository contains the code used for the paper draft **"Adaptive
Mass-Segmented KV Compression for Long-Context Reasoning"**. It is a research
fork of [NVIDIA KVPress](https://github.com/NVIDIA/kvpress) with the AMS
decoding-time compression methods, paper baselines, ablations, and a lightweight
evaluation CLI.

AMS is training-free. It wraps an existing token scorer, builds adaptive
segments from recent attention mass, allocates region-wise quotas, and then runs
top-k selection inside each segment. The release keeps the original KVPress
press abstractions so AMS can be used as a drop-in decoding-time compressor.

## Repository Layout

```text
kvpress/
  presses/
    adaptive_segment_head_press.py        # AMS implementation
    decoding_press.py                     # periodic decoding-time compression
    ablation_*.py                         # AMS ablations from the paper
    reasoning_path_press.py, rkv_press.py # reasoning baselines
evaluation/
  evaluate.py                             # paper-facing evaluation CLI
  evaluate_registry.py                    # datasets and press registry
  method_registry.py                      # stable method aliases
  benchmarks/                             # metric scripts
tests/                                    # upstream KVPress smoke/unit tests
```

Exploratory notebooks, generated caches, local result folders, and private
debug scripts have been removed from this release branch.

## Installation

Use Python 3.10 or newer. For a fresh reproducible environment, use the bundled
conda environment file:

```bash
conda env create -f environment.yml
conda activate kvpress-ams
```

If you already have a CUDA-compatible PyTorch environment, install the package
directly from this repository:

```bash
# Recommended if the environment previously had an editable upstream KVPress install.
pip uninstall -y kvpress kvpress-ams
pip install -e ".[eval]"
```

For FlashAttention-backed models, install the matching `flash-attn` wheel for
your CUDA/PyTorch stack separately, then use model loading options such as
`attn_implementation=flash_attention_2`.

## AMS Usage

```python
from kvpress import DecodingPress, ExpectedAttentionPress
from kvpress import AdaptiveMassSegmentWrapperHeadPress

base_scorer = ExpectedAttentionPress(epsilon=1e-2)
ams = AdaptiveMassSegmentWrapperHeadPress(
    base_press=base_scorer,
    window_size=32,
    segment_mass=0.02,
    min_seg_len=32,
    max_seg_len=1024,
    min_keep_per_segment=8,
    n_sink=4,
    always_keep_last=32,
    use_credit=True,
    credit_decay=0.8,
    mix_beta=0.7,
)

press = DecodingPress(
    base_press=ams,
    compression_interval=512,
    target_size=512,
    hidden_states_buffer_size=256,
)
```

Use `press` with the standard KVPress context manager or with the provided
evaluation CLI. The wrapper physically gathers the KV cache to `target_size` at
each compression event.

## Evaluation

The main entry point is:

```bash
python -m evaluation.evaluate \
  --model /path/to/DeepSeek-R1-Distill-Qwen-7B \
  --dataset math500 \
  --method_name ams_expected \
  --target_size 512 \
  --compression_interval 512 \
  --device cuda:0
```

Results are written under `results/<dataset>/<model>/<method>/...` and include
`predictions.csv`, `metrics.json`, `config.yaml`, and optional timing/memory
fields.

Stable method names are defined in `evaluation/method_registry.py`:

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

Datasets can be loaded from public Hugging Face IDs where available, or from
local files/directories via environment variables:

```bash
export MATH500_DATA=/path/to/math500.parquet
export AIME24_DATA=/path/to/aime24.jsonl
export AIME25_DATA=/path/to/aime25.jsonl
export GSM8K_DATA=/path/to/gsm8k
```

See [evaluation/README.md](evaluation/README.md) for more CLI examples.

## Development Checks

Fast syntax/import checks:

```bash
python -m compileall -q kvpress evaluation tests
python - <<'PY'
from evaluation.evaluate_registry import PRESS_REGISTRY, DATASET_REGISTRY
from kvpress import AdaptiveMassSegmentWrapperHeadPress, DecodingPress
print(len(PRESS_REGISTRY), sorted(DATASET_REGISTRY))
print(AdaptiveMassSegmentWrapperHeadPress.__name__, DecodingPress.__name__)
PY
```

If dev dependencies are installed, run:

```bash
pytest tests
```

## License and Attribution

This repository keeps the Apache-2.0 license and SPDX headers from upstream
KVPress. AMS-specific additions are released under the same license.

If you use this repository, please cite the AMS paper once available and also
cite the upstream KVPress / Expected Attention work when using the inherited
framework or baselines.
