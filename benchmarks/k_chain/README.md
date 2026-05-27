# k-chain distinguishability

This directory contains a small toy dataset for testing whether HIP-NN can
distinguish the two canonical k-chain geometries.

The first step is just dataset verification:

```bash
uv run python benchmarks/k_chain/generate_data/verify_dataset.py --k 4
```

To also save a visual plot and verify that the arrays can pass through a small
HIP-NN scalar readout:

```bash
uv run python benchmarks/k_chain/generate_data/verify_dataset.py --k 4 --plot
```

The tensors intended for HIP-NN use the usual names:

- `Z`: integer species, with `0` reserved for padding and `1` for every real toy node.
- `R`: centered Cartesian positions.
- `T`: binary labels shaped as scalar targets.

The reference `edge_index` is included only for diagnostics. Standard HIP-NN
models build interactions from `R` and ignore graph edges.

## Geometry diameters

These are Euclidean endpoint-to-endpoint diameters for the two centered
geometries.

| k | class 0 diameter | class 1 diameter |
|---|-----------------:|-----------------:|
| 2 | 13.60 | 11.00 |
| 3 | 17.89 | 16.00 |
| 4 | 22.47 | 21.00 |
| 5 | 27.20 | 26.00 |
| 6 | 32.02 | 31.00 |

With the default local cutoff, HIP-NN sees the equal-distance path neighbors.
The optional forward check only verifies input compatibility and output shape;
it is not evidence that the model can distinguish the two geometries.

To train the scalar HIP-NN output as a binary logit:

```bash
uv run python benchmarks/k_chain/run_models/train_kchain.py --k 4 --epochs 5000
```

To try HIP-HOP-NN instead:

```bash
uv run python benchmarks/k_chain/run_models/train_kchain.py --model hiphop --k 4 --epochs 5000 --l-max 2 --n-max 3
```

To try HIP-NN-TS with vector features:

```bash
uv run python benchmarks/k_chain/run_models/train_kchain.py --model hipnnvec --k 4 --epochs 5000
```

This uses the two canonical graphs as one full batch and optimizes
`BCEWithLogitsLoss` against labels `0` and `1`.

For the local-chain experiment, keep the default cutoff near the neighbor
spacing:

```bash
uv run python benchmarks/k_chain/run_models/train_kchain.py --k 4 --n-interaction-layers 3 --dist-hard-max 6.5
```

For a quick sanity check that the model can see nonlocal distances, try a much
larger cutoff:

```bash
uv run python benchmarks/k_chain/run_models/train_kchain.py --k 4 --dist-hard-max 30
```

To compare depths, cutoffs, and random seeds:

```bash
uv run python benchmarks/k_chain/run_models/sweep_kchain.py --k 3 4 5 --epochs 2000 --seeds 0 1 2 --interaction-layers 3 4 --hard-cutoffs 6.5 30
```

The training and sweep scripts also accept `--dataset incompleteness` for the
local-neighborhood counterexamples:

```bash
uv run python benchmarks/k_chain/run_models/train_kchain.py --dataset incompleteness --epochs 5000
```

By default this trains on all four incompleteness counterexamples; pass
`--counterexample three_body` to train on only one pair.

The sweep also accepts `--model hiphop`:

```bash
uv run python benchmarks/k_chain/run_models/sweep_kchain.py --model hiphop --k 3 4 5 --epochs 2000 --seeds 0 1 2 --interaction-layers 3 4 --hard-cutoffs 6.5 30
```

or `--model hipnnvec`:

```bash
uv run python benchmarks/k_chain/run_models/sweep_kchain.py --model hipnnvec --k 3 4 5 --epochs 2000 --seeds 0 1 2 --interaction-layers 3 4 --hard-cutoffs 6.5 30
```

The sweep reports both plain sign accuracy and margin accuracy. Margin accuracy
is the safer success criterion here because logits very close to zero can look
correct by thresholding while still not being a robust separation.

To plot a markdown sweep log as accuracy grids:

```bash
uv run python benchmarks/k_chain/results/plot_sweep_results.py benchmarks/k_chain/results/hiphopnn/training.md
```

The default plot uses final accuracy, where `0.5` is red and `1.0` is green.
Use `--metric margin_accuracy` or `--metric success_rate` to visualize the
stricter criteria.


# Logan Notes

model is able to distinguish with k=3 but NOT with k = 4


k	class 0 diameter	class 1 diameter    max size
2	13.60	11.00   14
3	17.89	16.00   18
4	22.47	21.00   23
5	27.20	26.00
6	32.02	31.00


counterexample	class 0	class 1	max
two_body	5.00	10.00	10.00
three_body	14.14	14.14	14.14
four_body_nonchiral	10.66	10.66	10.66
four_body_chiral	10.00	10.00	10.00