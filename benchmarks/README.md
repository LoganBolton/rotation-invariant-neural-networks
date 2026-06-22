# Benchmarks

This directory contains three toy geometric datasets and shared scripts for
training or sweeping HIP-NN/HIP-HOP models on them.

## Scripts

- `visualize.py`: generate HTML viewers for any benchmark dataset.
- `k_chain/generate_data/kchains.py`: generate k-chain dataset
- `incompleteness/generate_data/incompleteness.py`: local-neighborhood
  incompleteness dataset helpers.
- `rotating_ring/generate_data/rotating_ring_dataset.py`: rotating-ring
  dataset helpers.
- `run_models/train.py`: train one model.
- `run_models/sweep.py`: run repeated trainings over datasets, seeds, cutoffs,
  and model configs.

## Training

- To determine if pairs are created thorough radius cutoffs or 
through predefined edges, use: `--neighborhood-cutoff {cutoff,edges}`
- Select your dataset through: `--dataset {k_chain,incompleteness,rotating_ring}`
- By default the different model configs and dataset configs are 
determined through `benchmarks/run_models/sweep_configs.py`, but you
can also manually pass in different model configs


To train a single model:

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

## Visualize Dataset Examples

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
