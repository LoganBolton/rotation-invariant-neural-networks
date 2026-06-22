# Benchmarks

This directory contains three toy geometric datasets and shared scripts for
training or sweeping HIP-NN/HIP-HOP models on them.

## Scripts

- `visualize.py`: generate HTML viewers for any benchmark dataset.
- `k_chain/generate_data/kchains.py`: generate or save k-chain tensors.
- `incompleteness/generate_data/incompleteness.py`: local-neighborhood
  incompleteness dataset helpers.
- `rotating_ring/generate_data/rotating_ring_dataset.py`: rotating-ring
  dataset helpers.
- `run_models/train.py`: train one model.
- `run_models/sweep.py`: run repeated trainings over datasets, seeds, cutoffs,
  and model configs.

Generated viewers, plots, sweep logs, JSON results, and caches are ignored by
git.

## Important Params

Dataset selection:

- `--dataset {k_chain,incompleteness,rotating_ring}`
- k-chain: `--k`
- incompleteness: `--counterexample`
- rotating ring: `--ring-n-graphs`, `--ring-n-inner`, `--ring-n-outer`,
  `--ring-seed`, `--ring-outer-3d-rotation-deg`

Model/training:

- `--model {hipnn,hipnnvec,hiphop}`
- `--epochs`, `--seed`, `--learning-rate`
- `--n-interaction-layers`, `--n-atom-layers`, `--n-features`,
  `--n-sensitivities`
- HIP-HOP: `--l-max`, `--n-max`
- neighborhoods: `--neighborhood-cutoff {cutoff,edges}`
- distances: `--dist-hard-max`, `--dist-soft-min`, `--dist-soft-max`

Sweep-specific:

- `--seeds`
- `--interaction-layers`
- `--hard-cutoffs`
- `--model-configs default`
- `--ring-graph-configs all_2d`, `all_3d`, or explicit values like `2d:3:4`
- `--output-dir`

## Examples

Generate a k-chain tensor file:

```bash
uv run python benchmarks/k_chain/generate_data/kchains.py \
  --k 4 \
  --output benchmarks/k_chain/generated/k4.pt
```

Generate HTML viewers:

```bash
uv run python benchmarks/visualize.py --dataset k_chain --k 4
```

```bash
uv run python benchmarks/visualize.py \
  --dataset incompleteness \
  --counterexample three_body
```

```bash
uv run python benchmarks/visualize.py \
  --dataset rotating_ring \
  --ring-n-graphs 100 \
  --ring-seed 7 \
  --ring-n-inner 3 \
  --ring-n-outer 4
```

Train one model:

```bash
uv run python benchmarks/run_models/train.py \
  --dataset incompleteness \
  --counterexample all \
  --model hiphop \
  --neighborhood-cutoff edges \
  --epochs 5000 \
  --l-max 2 \
  --n-max 3
```

Run a sweep and save markdown plus JSON results:

```bash
uv run python benchmarks/run_models/sweep.py \
  --dataset rotating_ring \
  --ring-n-graphs 100 \
  --ring-graph-configs all_2d \
  --epochs 2000 \
  --seeds 0 1 2 \
  --hard-cutoffs 4 5 \
  --model-configs default \
  --output-dir benchmarks/rotating_ring/results/2d_sweep
```
