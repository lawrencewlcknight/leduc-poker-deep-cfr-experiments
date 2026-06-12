# Leduc Poker Deep CFR Replay and Average-Strategy Weighting Ablation

This experiment tests how average-strategy samples are weighted when training
the average-policy network.

The default configuration is aligned with the Experiment 2 baseline. The only
treatment variable in the thesis-facing run is:

- `average_strategy_weighting`: `linear` or `uniform`

The baseline arm is `uniform_replay_linear_avg_exp2_baseline`, which matches
the existing Experiment 2 setup.

Priority replay variants remain defined and can be selected with
`--variant-ids`, but they are excluded from the default run because the current
priority sampler is too compute intensive for the standard experiment suite.

## Run

```bash
python -m experiments.leduc_poker.deep_cfr_replay_averaging_ablation.run
```

Quick smoke run:

```bash
python -m experiments.leduc_poker.deep_cfr_replay_averaging_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids uniform_replay_linear_avg_exp2_baseline,uniform_replay_uniform_avg \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

Useful overrides:

- `--variant-ids`: comma-separated subset of variant ids
- `--priority-alpha`: exponent applied to priority replay scores
- `--priority-epsilon`: positive priority floor
- `--save-final-checkpoints`: save final full solver checkpoints

## Outputs

Each run writes to:

```text
outputs/leduc_poker_deep_cfr_replay_and_averaging_ablation_<timestamp>/
```

Core thesis artifacts:

- `seed_summary.csv`: one row per `(variant, seed)`
- `checkpoint_curves.csv`: checkpoint-level metrics and diagnostics
- `aggregate_summary.json`: per-variant aggregate statistics
- `paired_differences_vs_baseline.csv`: seed-paired deltas against the baseline
- `paired_difference_summary.json`: aggregate paired-difference statistics
- `ablation_curves.npz`: NumPy arrays for replotting curves
- `experiment_metadata.json`: full config and software metadata

Figures:

- `exploitability_by_iteration.png`
- `exploitability_by_nodes.png`
- `average_policy_value_by_iteration.png`
- `average_policy_value_by_nodes.png`
- `final_exploitability_by_variant.png`
- `final_average_policy_value_by_variant.png`
- `paired_deltas_vs_baseline.png`
- `policy_value_error_by_iteration.png`
- `advantage_target_variance_diagnostic.png`
- `policy_loss_diagnostic.png`
- `policy_entropy_diagnostic.png`
- `priority_effective_sample_size.png`

## Interpretation

Exploitability is the primary outcome. Negative paired deltas mean a variant
improved over the uniform-replay, linear-average baseline for the same seed.
Target variance, policy loss, policy entropy, and optional priority effective
sample size are mechanism diagnostics and should be reported separately from
strategic quality.
