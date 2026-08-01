# Experiment 27: Final-Candidate Checkpoint Head-to-Head

## Research question

Does the reduction in exploitability observed during training of the best Leduc
Deep CFR configuration correspond to progressively stronger direct-play
performance?

## Design

Each of five fixed seeds trains one uninterrupted instance of the final
candidate selected in Experiment 26. Lightweight average-policy snapshots are
captured at 20%, 40%, 60%, 80%, and 100% of the 1,050-iteration budget:

```text
iterations:     210, 420, 630, 840, 1050
expected nodes:  3M,  6M,  9M, 12M,  15M
seeds:          1234, 2025, 31415, 27182, 16180
```

Actual nodes touched are recorded for every snapshot and used on every
longitudinal chart. The callback-based snapshot mechanism does not split or
restart training and does not serialize the replay buffers.

The final-candidate configuration is retained exactly: a two-layer width-32
policy MLP; an eight-layer width-32 residual LayerNorm centred-advantage
network; standardised, unclipped targets; uniform advantage replay and
average-strategy weighting; warm-started advantage networks; policy fitting
every 10 iterations; advantage minibatch 2048; replay capacity 5M; and constant
learning rate 0.004.

## Evaluation and inference

Leduc is small enough to evaluate policies exactly. For each seed, every pair
of checkpoints is evaluated with OpenSpiel in both seat assignments. If `A`
denotes the later policy and `B` the earlier policy, the reported effect is

```text
0.5 * (value of A as player 0 against B + value of A as player 1 against B).
```

There is consequently no Monte Carlo match noise and no arbitrary game-count
choice. The independent training seed, rather than each of the ten correlated
checkpoint pairs within a seed, is the primary inferential unit. The primary
estimand is the mean later-versus-earlier exact EV within each seed, aggregated
across the five seeds. Adjacent-checkpoint and final-versus-first contrasts test
the shape and endpoint of the progression separately. The analysis reports 95%
t intervals and exact one-sided sign-flip tests. All ten checkpoint-specific
comparisons are secondary and use Holm correction for family-wise error control.

With five seeds, the smallest attainable one-sided exact sign-flip p-value is
`1/32 = 0.03125`. The experiment can therefore detect a completely consistent
directional effect, but effect magnitudes and confidence intervals remain more
informative than a binary significance declaration.

## Run

From the repository root:

```bash
# Five-seed default run: train and analyse
python -m experiments.leduc_poker.deep_cfr_final_candidate_checkpoint_head_to_head.run

# Re-run exact analysis against existing snapshots
python -m experiments.leduc_poker.deep_cfr_final_candidate_checkpoint_head_to_head.run analyse \
  --run-dir outputs/leduc_poker_deep_cfr_final_candidate_checkpoint_head_to_head_<timestamp>

# Local smoke test
python -m experiments.leduc_poker.deep_cfr_final_candidate_checkpoint_head_to_head.run \
  --seeds 1234 \
  --iterations 10 \
  --checkpoint-schedule 2,4,6,8,10 \
  --traversals 4 \
  --evaluation-interval 2 \
  --policy-network-train-every 2 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

## GCP Batch

Set `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL` first; see
[`docs/GCP_BATCH_EXPERIMENTS.md`](../../../docs/GCP_BATCH_EXPERIMENTS.md).

```bash
# GCP smoke test
./gcp/submit_batch_experiment.sh \
  "smoke-exp27-final-checkpoint-h2h-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_final_candidate_checkpoint_head_to_head.run \
    --seeds 1234 \
    --iterations 10 \
    --checkpoint-schedule 2,4,6,8,10 \
    --traversals 4 \
    --evaluation-interval 2 \
    --policy-network-train-every 2 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --policy-network-layers 8,8 \
    --advantage-network-layers 8,8 \
    --batch-size-advantage 2 \
    --batch-size-strategy 2 \
    --memory-capacity 256 \
    --output-root outputs/cloud/smoke-exp27-final-checkpoint-h2h" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Five-seed full run, approximately 15M nodes per seed
./gcp/submit_batch_experiment.sh \
  "leduc-deep-cfr-exp27-final-checkpoint-h2h-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_final_candidate_checkpoint_head_to_head.run \
    --output-root outputs/cloud/leduc-deep-cfr-exp27-final-checkpoint-h2h" \
  "n2-standard-4" \
  "172800" \
  "4000" \
  "16000"
```

Based on the completed Experiment 26 candidate runs, the five training seeds
should take approximately 14--18 hours sequentially on the same machine class;
exact checkpoint analysis adds comparatively little time. The 48-hour Batch
timeout leaves substantial headroom.

## Principal outputs

| File | Contents |
| --- | --- |
| `training_stage_metrics.csv` | Actual nodes, elapsed time, replay sizes, and policy-fit counts at each snapshot. |
| `checkpoint_exploitability_metrics.csv` | Exact NashConv/2, self-play value, and value error by seed and checkpoint. |
| `head_to_head_pairwise.csv` | Exact two-seat EV for every ordered checkpoint pair. |
| `head_to_head_primary_effect_by_seed.csv` | One independent later-versus-earlier summary effect per seed. |
| `head_to_head_inference_summary.csv` | Primary and final-versus-first estimates, confidence intervals, and exact p-values. |
| `head_to_head_pairwise_inference.csv` | Secondary pair-specific estimates with Holm-adjusted p-values. |
| `aggregate_summary.json` | Machine-readable statement of the estimands and inferential protocol. |
| `head_to_head_later_vs_earlier.png` | Lower-triangular exact-EV matrix labelled by mean nodes touched. |
| `head_to_head_strength_vs_earlier_by_nodes.png` | Mean EV against all earlier checkpoints over nodes. |
| `head_to_head_strength_vs_previous_by_nodes.png` | Adjacent-checkpoint EV over nodes. |
| `exploitability_by_nodes.png` | Exact exploitability at the five snapshots. |
| `average_policy_value_by_nodes.png` | Average-policy value at the five snapshots. |
| `head_to_head_primary_effect_by_seed.png` | Primary seed-level effects and their cross-seed mean. |
| `snapshots/seed_<seed>_iter_<iter>_snapshot.pt` | Lightweight average-policy snapshots used for exact analysis. |
