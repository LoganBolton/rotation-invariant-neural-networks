# Benchmarks

This directory contains the small geometric benchmark datasets and shared model
training utilities. The PR-facing surface is:

- dataset generators under `*/generate_data/`
- `run_models/train.py` for one training run
- `run_models/sweep.py` for hyperparameter sweeps
- `run_models/sweep_configs.py` and `run_models/sweep_results.py` for sweep
  configuration parsing and JSON result output

Generated plots, HTML viewers, sweep logs, JSON results, and caches are ignored
by git.

## Datasets

### k-chain

Two canonical k-chain geometries test whether a model can distinguish endpoint
placement around a chain.

Generate and inspect the tensors:

```bash
uv run python benchmarks/k_chain/generate_data/kchains.py --k 4
```

Optionally save the HIP-NN tensor dictionary:

```bash
uv run python benchmarks/k_chain/generate_data/kchains.py \
  --k 4 \
  --output benchmarks/k_chain/generated/k4.pt
```

Train one model:

```bash
uv run python benchmarks/run_models/train.py --dataset k_chain --k 4 --epochs 5000
```

Sweep depths, cutoffs, and seeds:

```bash
uv run python benchmarks/run_models/sweep.py \
  --dataset k_chain \
  --k 3 4 5 \
  --epochs 2000 \
  --seeds 0 1 2 \
  --interaction-layers 3 4 \
  --hard-cutoffs 6.5 30
```

### Local-neighborhood incompleteness

The incompleteness dataset contains four local-neighborhood counterexample
pairs:

- `two_body`
- `three_body`
- `four_body_nonchiral`
- `four_body_chiral`

Each pair has two star-shaped local environments centered at node `0`, with
binary labels `0` and `1`. The tensors use the usual HIP-NN names:

- `Z`: integer species, with `0` reserved for padding and `1` for each real toy
  node
- `R`: Cartesian positions
- `T`: scalar binary targets
- `edge_indices`: per-system explicit center-leaf edges for fixed-topology
  message passing

Verify the dataset:

```bash
uv run python benchmarks/incompleteness/generate_data/verify_dataset.py
```

Verify one pair:

```bash
uv run python benchmarks/incompleteness/generate_data/verify_dataset.py --counterexample three_body
```

Train with normal cutoff-built neighborhoods:

```bash
uv run python benchmarks/run_models/train.py \
  --dataset incompleteness \
  --counterexample all \
  --epochs 5000
```

Train with dataset-defined star edges:

```bash
uv run python benchmarks/run_models/train.py \
  --dataset incompleteness \
  --neighborhood-cutoff edges \
  --epochs 5000
```

Sweep the standard model bundle and write one markdown and JSON result file per
model config:

```bash
uv run python benchmarks/run_models/sweep.py \
  --dataset incompleteness \
  --neighborhood-cutoff edges \
  --hard-cutoffs 5 10 14 \
  --model-configs default \
  --output-dir benchmarks/incompleteness/results/system_edges
```

### Rotating ring

The rotating-ring dataset creates graphs with one center node, an inner ring,
and an outer ring. Labels are assigned by sorting graphs on the minimum nearest
outer-node distance across inner-ring nodes.

Generate an HTML viewer:

```bash
uv run python benchmarks/rotating_ring/generate_data/rotating_ring_dataset.py \
  --n-graphs 100 \
  --seed 7 \
  --n-inner 3 \
  --n-outer 4 \
  --html benchmarks/rotating_ring/generate_data/visualizations/2d_3inner_4outer_100.html
```

Generate a 3D-tilted viewer:

```bash
uv run python benchmarks/rotating_ring/generate_data/rotating_ring_dataset.py \
  --n-graphs 100 \
  --seed 7 \
  --n-inner 4 \
  --n-outer 4 \
  --outer-3d-rotation-deg 360 \
  --outer-3d-axis-deg 360 \
  --html benchmarks/rotating_ring/generate_data/visualizations/3d_4inner_4outer_100.html
```

Train one model:

```bash
uv run python benchmarks/run_models/train.py \
  --dataset rotating_ring \
  --ring-n-graphs 100 \
  --ring-n-inner 3 \
  --ring-n-outer 4 \
  --epochs 2000
```

Sweep rotating-ring graph sizes:

```bash
uv run python benchmarks/run_models/sweep.py \
  --dataset rotating_ring \
  --ring-n-graphs 100 \
  --ring-graph-configs all_2d \
  --model-configs default \
  --output-dir benchmarks/rotating_ring/results/2d_sweep
```

## Shared Training Options

`train.py` supports:

- `--dataset {k_chain,incompleteness,rotating_ring}`
- `--model {hipnn,hipnnvec,hiphop}`
- `--neighborhood-cutoff {cutoff,edges}`
- `--n-interaction-layers`, `--n-atom-layers`, `--n-features`,
  `--n-sensitivities`
- HIP-HOP-specific `--l-max` and `--n-max`
- distance sensitivity parameters `--dist-soft-min`, `--dist-soft-max`, and
  `--dist-hard-max`

`--neighborhood-cutoff edges` uses the dataset-provided `edge_indices` instead
of HIP-NN cutoff-built neighbors. This is useful for experiments that need a
fixed topology, such as center-leaf star neighborhoods.

## Sweep Outputs

When `--output-dir` is set, `sweep.py` writes markdown logs to that directory
and JSON files under `<output-dir>/json_results`. Without `--output-dir`, it
prints the sweep log to stdout and writes JSON to `sweep_json_results/`.

Useful examples:

```bash
uv run python benchmarks/run_models/sweep.py \
  --dataset k_chain \
  --hard-cutoffs 5 10 14 \
  --model-configs default \
  --output-dir benchmarks/k_chain/results/system_node
```

```bash
uv run python benchmarks/run_models/sweep.py \
  --dataset incompleteness \
  --counterexamples two_body three_body four_body_nonchiral four_body_chiral \
  --epochs 2000 \
  --seeds 0 1 2 \
  --interaction-layers 1 2 3 \
  --hard-cutoffs 5 10 14 18
```

## Focused Tests

```bash
uv run pytest \
  tests/test_k_chain_dataset.py \
  tests/test_incompleteness_dataset.py \
  tests/test_edge_neighborhood.py
```
