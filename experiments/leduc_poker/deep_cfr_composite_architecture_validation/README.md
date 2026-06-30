# Leduc Deep CFR Composite Architecture Validation

This experiment explicitly tests the architecture proposed after the Leduc
network-architecture ablations. It keeps the average-policy network fixed at
the baseline 2x32 MLP and varies only the advantage-network architecture.

The default run uses five fixed seeds:

```text
1234, 2025, 31415, 27182, 16180
```

## Variants

- `baseline_direct_2x32`: original Experiment 1 baseline, using direct 2x32 MLP
  advantage networks.
- `centered_advantage_8x32`: strongest single factorised-head intervention,
  using an 8x32 centred action-advantage head.
- `composite_res_ln_centered_advantage_8x32`: proposed combined configuration,
  using an 8x32 residual LayerNorm trunk with a mean-centred action-advantage
  output head.

## Run

From `leduc_poker_deep_cfr/leduc-poker-deep-cfr-experiments`:

```bash
python -m experiments.leduc_poker.deep_cfr_composite_architecture_validation.run
```

The shared architecture-ablation runner writes per-seed summaries, aggregate
tables, paired differences, curve CSVs, NPZ arrays, and plots including
exploitability by iteration and exploitability by nodes touched.

For a cheaper dry run:

```bash
python -m experiments.leduc_poker.deep_cfr_composite_architecture_validation.run \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --seeds 1234
```
