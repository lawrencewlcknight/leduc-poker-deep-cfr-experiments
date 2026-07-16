# Leduc Poker Deep CFR Parallel-Equivalence Ablation

This experiment compares the current best Deep CFR configuration with the same
learner using Ray-parallel traversal collection. The learner, networks, target
processing, replay settings, average-strategy weighting, optimiser settings,
and evaluation protocol are held fixed; only the traversal execution backend
changes.

| Variant | Change from baseline | Rationale |
| --- | --- | --- |
| `composite_best_sequential` | Existing sequential traversal loop | Reference implementation and current production path. |
| `composite_best_ray_parallel` | Three Ray traversal workers plus one central learner | Tests whether ESCHER-style parallel traversal collection preserves learning quality while reducing runtime. |

The default run uses three matched seeds: `1234,2025,31415`.

## Run

```bash
python -m experiments.leduc_poker.deep_cfr_parallel_equivalence_ablation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_parallel_equivalence_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --parallel-num-workers 2 \
  --output-root outputs/smoke_tests
```

GCP Batch smoke test:

Set `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL` first; see
[`docs/GCP_BATCH_EXPERIMENTS.md`](../../../docs/GCP_BATCH_EXPERIMENTS.md).

```bash
./gcp/submit_batch_experiment.sh \
  "smoke-exp25-parallel-equivalence-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_parallel_equivalence_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --evaluation-interval 1 \
    --policy-network-train-every 1 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --policy-network-layers 8,8 \
    --advantage-network-layers 8,8 \
    --batch-size-advantage 2 \
    --batch-size-strategy 2 \
    --memory-capacity 256 \
    --parallel-num-workers 2 \
    --output-root outputs/cloud/smoke-exp25-parallel-equivalence" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

## Outputs

The runner writes the standard experiment artefacts plus:

- `paired_parallel_equivalence_and_timing.csv` — paired seed deltas and timing speedups.
- `parallel_equivalence_summary.json` — practical-equivalence summaries for final exploitability and final policy value.
- `runtime_by_variant.png` — training-loop, end-to-end, and traversal-collection runtime bars.
- `traversal_collection_seconds_by_nodes.png` — cumulative traversal-collection time by nodes touched.

For thesis figures, prefer the longitudinal `*_by_nodes.png` charts.
