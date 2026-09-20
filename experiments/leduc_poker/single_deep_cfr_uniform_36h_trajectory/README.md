# Uniform SD-CFR 36-hour trajectory (Experiment 29)

## Research question

Does the uniformly weighted SD-CFR configuration selected by Experiment 28
continue to improve over 36 active training hours, and is its long-horizon
behaviour stable across the five thesis seeds?

## Frozen design

- Seeds `1234`, `2025`, `31415`, `27182`, and `16180`.
- One `n2-standard-8` VM per seed, with all five seeds running in parallel.
- 36 active training hours per seed.
- The Experiment 28 advantage learner unchanged: 320 traversals per player and
  iteration; 200 advantage updates; batch size 2,048; a five-million-row
  reservoir per player; constant learning rate 0.004; standardised targets;
  and continuously warm-started eight-layer width-32 residual LayerNorm
  centred-advantage networks.
- No average-policy fitting, strategy replay, or exploitability calculation in
  the timed learner. These are not components of standalone SD-CFR.
- Every fitted historical advantage network is retained. Archive-prefix
  checkpoints are recorded every 30 active-training minutes, at the first
  completed iteration crossing 15 million nodes, and at the final completed
  iteration after the 36-hour boundary.
- Exact reach-weighted uniform SD-CFR is the primary output; canonical linear
  SD-CFR is evaluated from the same archive as a secondary diagnostic.

Evaluation is deferred to five parallel post-training workers. The primary
endpoint is exact uniform-SD-CFR exploitability at 36 hours.

## Cloud execution

From the repository root, with `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL`
already set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp29-sdcfr36h-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_single_deep_cfr_uniform_36h_trajectory.sh smoke-local
./gcp/run_single_deep_cfr_uniform_36h_trajectory.sh run
```

The remote controller runs a cloud smoke test, launches five parallel training
workers, launches five parallel deferred-evaluation workers, and aggregates
the analysis. The laptop may be disconnected after submission.

```bash
./gcp/run_single_deep_cfr_uniform_36h_trajectory.sh status
./gcp/run_single_deep_cfr_uniform_36h_trajectory.sh resume
```

Principal analysis artifacts:

- `analysis/exploitability_by_training_time.png`;
- `analysis/exploitability_by_nodes.png`;
- `analysis/exploitability_by_iteration.png`;
- `analysis/uniform_minus_linear_by_training_time.png`;
- `analysis/endpoint_seed_metrics.csv`;
- `analysis/endpoint_aggregate_summary.csv`;
- `analysis/trajectory_by_training_time_summary.csv`; and
- `analysis/trajectory_by_nodes_summary.csv`.

