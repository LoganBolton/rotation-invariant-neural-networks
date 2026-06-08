# Local-neighborhood incompleteness

This directory contains four local-neighborhood counterexample pairs based on
the geometric GNN dojo
[incompleteness notebook](https://github.com/chaitjo/geometric-gnn-dojo/blob/main/experiments/incompleteness.ipynb):

- `two_body`
- `three_body`
- `four_body_nonchiral`
- `four_body_chiral`

Each pair has two star-shaped local environments centered at node `0`, with
labels `0` and `1`.

By default, the dataset uses the `v2` coordinate set, whose center-leaf distances
are smaller than leaf-leaf distances. The original notebook-style coordinates
remain available with `coordinate_set="original"` or `--coordinate-set original`.

The first step is dataset verification:

```bash
uv run python benchmarks/incompleteness/generate_data/verify_dataset.py
```

To verify one pair:

```bash
uv run python benchmarks/incompleteness/generate_data/verify_dataset.py --counterexample three_body
```

To verify the original coordinates:

```bash
uv run python benchmarks/incompleteness/generate_data/verify_dataset.py --coordinate-set original
```

To also save 3D plots of the counterexample pairs:

```bash
uv run python benchmarks/incompleteness/generate_data/verify_dataset.py --plot
```

The default plot output for the v2 coordinates is
`benchmarks/incompleteness/generate_data/v2_visualizations`.

The tensors intended for HIP-NN use the usual names:

- `Z`: integer species, with `0` reserved for padding and `1` for every real toy node.
- `R`: Cartesian positions centered per environment when stacked for HIP-NN.
- `T`: binary labels shaped as scalar targets.
- `edge_index`: compressed atom-indexed center-leaf edges for explicit edge
  based message passing.

## Training

The shared training script lives at the benchmark root and accepts a dataset
flag:

```bash
uv run python benchmarks/run_models/train.py --dataset incompleteness --epochs 5000
```

To train on the original coordinates instead:

```bash
uv run python benchmarks/run_models/train.py --dataset incompleteness --coordinate-set original --epochs 5000
```

To restrict message passing to the stored center-leaf star graph instead of
HIP-NN's cutoff-built neighbor graph:

```bash
uv run python benchmarks/run_models/train.py --dataset incompleteness --neighborhood-cutoff edges --epochs 5000
```

## Explicit Edge Neighborhoods

The `--neighborhood-cutoff edges` mode is the path that tests HIP-NN/HIP-HOP
when leaf nodes cannot communicate directly with other leaf nodes. It does not
change the final readout by itself. It changes only the message-passing
neighborhood used inside the interaction layers.

In this dataset every environment is a star graph:

```text
leaf 1 <-> center <-> leaf 2
leaf 3 <-> center <-> leaf 4
...
```

There are edges from the center to every leaf and from every leaf back to the
center. There are no leaf-leaf edges, even if two leaves are close in Cartesian
space.

The important files are:

- `benchmarks/incompleteness/generate_data/incompleteness.py`: creates the
  dataset and stores `edge_index`.
- `benchmarks/run_models/train.py`: converts `edge_index` into HIP-NN pair
  tensors and builds the model with those pairs as graph inputs.
- `benchmarks/run_models/sweep.py`: passes `--neighborhood-cutoff edges`
  through to each training run.
- `external/hippynn/...`: provides the normal HIP-NN/HIP-HOP network layers and
  pair-tensor interface. The dataset-specific star graph is not hard-coded in
  the external hippynn source.

### Dataset Side

Each `IncompletenessEnvironment` stores:

- `Z`: node species.
- `R`: node positions.
- `T`: the binary label, added when environments are stacked into HIP-NN arrays.
- `central_atom_local_index`: currently node `0`.
- `edge_index`: the local star-graph edges for that environment.

`star_edge_index(n_nodes)` builds a bidirectional center-leaf edge list for one
environment. For example, with three nodes it returns:

```text
0 -> 1
0 -> 2
1 -> 0
2 -> 0
```

When multiple environments are batched together, `center_leaf_edge_tensors`
converts those local edges into one global compressed atom index. "Compressed"
means the indices count only real atoms, after padding atoms have been removed.
For padded batches, this matters because the HIP-NN pair tensors operate over
the flattened real-atom representation, not over padded `[system, atom]`
coordinates.

The final arrays include:

```python
{
    "Z": species,
    "R": positions,
    "T": targets,
    "edge_index": edge_index,
}
```

### Training Side

The training script has two neighborhood modes:

- `--neighborhood-cutoff cutoff`: the default HIP-NN behavior. The graph receives
  `Z` and `R`, and hippynn builds neighbors from interatomic distances and the
  hard cutoff.
- `--neighborhood-cutoff edges`: the benchmark behavior for this experiment. The
  graph receives `Z`, `pair_first`, `pair_second`, `pair_dist`, and, for HIP-HOP
  or HIPNNVec, `pair_coord`.

`edge_pair_tensors(arrays)` is the bridge between the dataset and HIP-NN. It:

1. Checks that `Z`, `R`, and `edge_index` are present.
2. Builds the flattened list of real atoms from `Z != 0`.
3. Verifies that `edge_index` points only to valid real atoms.
4. Splits `edge_index` into `pair_first` and `pair_second`.
5. Computes `pair_coord = R[pair_first] - R[pair_second]`.
6. Computes `pair_dist = ||pair_coord||`.

Those tensors are then passed directly into the graph as `InputNode`s with
`IdxType.Pairs`. This bypasses hippynn's usual cutoff-based pair construction.

For scalar HIP-NN, the network parents are:

```python
indexed_features, pair_first, pair_second, pair_dist
```

For HIP-HOP and HIPNNVec, the network also needs directional information:

```python
indexed_features, pair_first, pair_second, pair_dist, pair_coord
```

### Why The Hard Cutoff Is Still Set In Edges Mode

Even in `edges` mode, the HIP-NN/HIP-HOP network still has sensitivity functions
that use `dist_hard_max` and `dist_soft_max`. In cutoff mode, `dist_hard_max`
does two jobs:

1. It decides which atom pairs exist.
2. It shapes the distance sensitivity cutoff inside the model.

In edges mode, job 1 is handled by `edge_index`, so distance should not remove an
edge. To avoid accidentally suppressing long explicit edges, `train.py` resolves
the hard cutoff to a large value for edge neighborhoods:

```python
EDGE_NEIGHBORHOOD_DIST_HARD_MAX = 1.0e6
```

That makes the sensitivity cutoff effectively nonrestrictive for these toy
geometries. The swept `hard_cutoff` values can still appear in the markdown logs,
but in edge mode they are no longer deciding which messages exist.

### Sweep Side

The sweep script exposes the same flag:

```bash
uv run python benchmarks/run_models/sweep.py \
  --dataset incompleteness \
  --coordinate-set original \
  --neighborhood-cutoff edges \
  --output-dir benchmarks/incompleteness/results/system_original_edges
```

When `--output-dir` is provided without `--model-configs`, the sweep runs the
default model bundle and writes one markdown file per model config.

In edge mode, `sweep.py` passes `dist_soft_max=None` into `train.py`, so the
training script can apply the edge-mode defaults consistently.

### What This Tests

The purpose of this mode is to block direct leaf-leaf communication while still
allowing every leaf to talk to the center. With more than one interaction layer,
information can travel along paths such as:

```text
leaf A -> center -> leaf B
```

but there is still no direct message:

```text
leaf A -> leaf B
```

Edge neighborhoods change which atoms can exchange information inside each
interaction layer. They do not change the final system-level readout.

### Tests

The edge behavior is covered in `tests/test_incompleteness_dataset.py`.

The tests check that:

- `edge_index` has shape `[2, n_edges]` and integer dtype.
- Every edge stays inside a single environment.
- Every edge touches the central atom.
- No edge connects one leaf to another leaf.
- Padding atoms are never referenced.
- `edge_pair_tensors` produces `pair_first`, `pair_second`, `pair_coord`, and
  `pair_dist` consistent with `edge_index` and `R`.

The most useful command for this area is:

```bash
uv run pytest tests/test_incompleteness_dataset.py tests/test_edge_neighborhood.py
```

The shared sweep script accepts the same dataset flag:

```bash
uv run python benchmarks/run_models/sweep.py --dataset incompleteness --counterexamples two_body three_body four_body_nonchiral four_body_chiral --epochs 2000 --seeds 0 1 2 --interaction-layers 1 2 3 --hard-cutoffs 5 10 14 18
```

To run the standard HIP-NN/HIP-HOP comparison in one command
and write one markdown log per model config:

```bash
uv run python benchmarks/run_models/sweep.py --dataset incompleteness --hard-cutoffs 5 10 14 --model-configs default --output-dir benchmarks/incompleteness/results/system_node
```

The default config bundle writes:

- `l0_n1.md` for HIP-NN
- `l1_n2.md`
- `l2_n2.md`
- `l2_n3.md`
- `l3_n2.md`
- `l3_n4.md`
