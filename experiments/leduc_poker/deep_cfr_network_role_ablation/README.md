# Leduc Poker Deep CFR Network-Role Ablation

Experiment 14 tests whether Deep CFR is more sensitive to architecture in the
average-policy network or the advantage networks. Unlike Experiments 11-13, the
policy and advantage networks are not always varied together.

The baseline is `policy 2x32, advantage 2x32`. The treatment arms shrink or
deepen only one network role at a time:

- policy `2x8`, advantage `2x32`
- policy `8x32`, advantage `2x32`
- policy `2x32`, advantage `2x8`
- policy `2x32`, advantage `8x32`

```bash
python -m experiments.leduc_poker.deep_cfr_network_role_ablation.run
```

Quick smoke run:

```bash
python -m experiments.leduc_poker.deep_cfr_network_role_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids baseline_policy2x32_advantage2x32,small_policy_baseline_advantage,baseline_policy_small_advantage \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

Core outputs include `seed_summary.csv`, `checkpoint_curves.csv`,
`aggregate_summary.json`, `variant_summary.csv`,
`paired_differences_vs_baseline.csv`, `ablation_curves.npz`, and thesis-facing
PNG figures.

