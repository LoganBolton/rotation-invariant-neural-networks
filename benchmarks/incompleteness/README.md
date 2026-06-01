# Local-neighborhood incompleteness

This directory contains the four local-neighborhood counterexample pairs from
the geometric GNN dojo
[incompleteness notebook](https://github.com/chaitjo/geometric-gnn-dojo/blob/main/experiments/incompleteness.ipynb):

- `two_body`
- `three_body`
- `four_body_nonchiral`
- `four_body_chiral`

Each pair has two star-shaped local environments centered at node `0`, with
labels `0` and `1`. The diagnostic `edge_index` stores the star graph only for
inspection

The first step is dataset verification:

```bash
uv run python benchmarks/incompleteness/generate_data/verify_dataset.py
```

To verify one pair:

```bash
uv run python benchmarks/incompleteness/generate_data/verify_dataset.py --counterexample three_body
```

To also save 3D plots of the counterexample pairs:

```bash
uv run python benchmarks/incompleteness/generate_data/verify_dataset.py --plot
```

The tensors intended for HIP-NN use the usual names:

- `Z`: integer species, with `0` reserved for padding and `1` for every real toy node.
- `R`: Cartesian positions centered per environment when stacked for HIP-NN.
- `T`: binary labels shaped as scalar targets.
- `central_atom_mask`: padded atom mask selecting node `0` in each environment.

## Training

The shared training script lives at the benchmark root and accepts a dataset
flag:

```bash
uv run python benchmarks/run_models/train.py --dataset incompleteness --epochs 5000
```

To use the central-atom-only hierarchical readout:

```bash
uv run python benchmarks/run_models/train.py --dataset incompleteness --readout central --epochs 5000
```

The shared sweep script accepts the same dataset flag:

```bash
uv run python benchmarks/run_models/sweep.py --dataset incompleteness --counterexamples two_body three_body four_body_nonchiral four_body_chiral --epochs 2000 --seeds 0 1 2 --interaction-layers 1 2 3 --hard-cutoffs 5 10 14 18
```

To run the standard central-readout HIP-NN/HIP-HOP comparison in one command
and write one markdown log per model config:

```bash
uv run python benchmarks/run_models/sweep.py --dataset incompleteness --hard-cutoffs 5 10 14 --model-configs default --output-dir benchmarks/incompleteness/results/central_node
```

The default config bundle writes:

- `l0_n1.md` for HIP-NN
- `l1_n2.md`
- `l2_n2.md`
- `l2_n3.md`
- `l3_n2.md`
- `l3_n4.md`
