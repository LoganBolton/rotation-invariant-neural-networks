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

To train the scalar HIP-NN output as a binary logit:

```bash
uv run python benchmarks/run_models/train.py --k 4 --epochs 5000
```

To keep normal HIP-NN message passing but read out only from the designated atom:

```bash
uv run python benchmarks/run_models/train.py --k 4 --readout central --epochs 5000
```

To compare depths, cutoffs, and random seeds:

```bash
uv run python benchmarks/run_models/sweep.py --k 3 4 5 --epochs 2000 --seeds 0 1 2 --interaction-layers 3 4 --hard-cutoffs 6.5 30
```

To plot a markdown sweep log as accuracy grids:

```bash
uv run python benchmarks/k_chain/results/plot_sweep_results.py benchmarks/k_chain/results/hiphopnn/training.md
```


# Dataset Sizes


k	class 0 diameter	class 1 diameter
2	13.60	11.00   14
3	17.89	16.00   18
4	22.47	21.00   23
5	27.20	26.00
6	32.02	31.00


counterexample	class 0	class 1
two_body	5.00	10.00
three_body	14.14	14.14
four_body_nonchiral	10.66	10.66
four_body_chiral	10.00	10.00