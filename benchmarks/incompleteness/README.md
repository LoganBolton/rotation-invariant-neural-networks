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
inspection; HIP-NN-style models can use the stacked `Z`, `R`, and `T` arrays.

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

The verifier checks that each pair has matching center-relative body-order
distance fingerprints at the body order named by the counterexample. This is a
local diagnostic for the dataset construction, not a claim that a particular
model will or will not distinguish the pair after training.
