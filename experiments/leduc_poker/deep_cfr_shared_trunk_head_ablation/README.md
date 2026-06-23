# Experiment 15: Shared Trunk With Player/Action Heads

This experiment tests whether the Deep CFR advantage approximators benefit from
a shared hidden representation with separate player/action output heads. The
average-policy network is held fixed at the Experiment 1 two-layer, width-32 MLP
so that the comparison isolates the advantage-network representation.

The design compares independent per-player advantage MLPs against a shared
advantage trunk with player-specific action heads at depths `2`, `4`, and `8`,
all at hidden width `32`. Each variant uses the same solver settings as the
Experiment 1 baseline unless overridden on the command line. The default run
uses three matched seeds to keep the full grid feasible under the 48-hour Batch
runtime limit.

```bash
python -m experiments.leduc_poker.deep_cfr_shared_trunk_head_ablation.run
```

Quick local smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_shared_trunk_head_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids independent_advantage_layers2_width32,shared_trunk_advantage_layers2_width32 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

The runner writes the same thesis-facing artefacts as the other architecture
ablations, including per-seed summaries, checkpoint curves, variant summaries,
paired differences against the baseline, and publication-style diagnostic
figures.
