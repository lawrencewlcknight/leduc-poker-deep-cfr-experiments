# Leduc Poker Deep CFR Composite Replay-Memory HP Ablation

This experiment tests replay freshness around the current best baseline. The
baseline keeps a large `10,000,000`-sample reservoir. The variants lower that
capacity while holding the network architecture, traversal budget, target
processing, replay sampling rule, and average-strategy weighting fixed.

| Variant | Change from baseline | Rationale |
| --- | --- | --- |
| `composite_best_baseline` | `10,000,000` capacity | Large-reservoir reference. |
| `memory_capacity_1m` | `1,000,000` capacity | Aggressive freshness test. |
| `memory_capacity_2m` | `2,000,000` capacity | Intermediate freshness/coverage trade-off. |
| `memory_capacity_5m` | `5,000,000` capacity | Conservative freshness test. |

## Run

```bash
python -m experiments.leduc_poker.deep_cfr_composite_replay_memory_ablation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_composite_replay_memory_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids composite_best_baseline,memory_capacity_1m \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

## Outputs

The runner writes the standard experiment artefacts and paired differences
against `composite_best_baseline`. For thesis figures, prefer the
`*_by_nodes.png` longitudinal charts.
