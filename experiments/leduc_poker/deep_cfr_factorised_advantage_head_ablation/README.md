# Experiment 16: Factorised Advantage-Output Heads

This experiment tests whether structuring the advantage-network output improves
Deep CFR performance. The average-policy network is held fixed at the
Experiment 1 two-layer, width-32 MLP so that the comparison isolates the
advantage-network output head.

The design compares three advantage-output heads at depths `2`, `4`, and `8`,
all at hidden width `32`:

- `direct`: the baseline MLP head that predicts one value per action.
- `centered`: an action head whose outputs are centred to zero mean for each
  information state.
- `dueling`: a scalar state-value head plus an action-advantage head centred to
  zero mean, following the dueling-network decomposition.

Each variant uses the same solver settings as the Experiment 1 baseline unless
overridden on the command line. The default run uses three matched seeds to keep
the full grid feasible under the 48-hour Batch runtime limit.

```bash
python -m experiments.leduc_poker.deep_cfr_factorised_advantage_head_ablation.run
```

Quick local smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_factorised_advantage_head_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids direct_advantage_layers2_width32,centered_advantage_layers2_width32,dueling_advantage_layers2_width32 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

The runner writes the same thesis-facing artefacts as the other architecture
ablations, including per-seed summaries, checkpoint curves, variant summaries,
paired differences against the direct-head baseline, and diagnostic figures.
