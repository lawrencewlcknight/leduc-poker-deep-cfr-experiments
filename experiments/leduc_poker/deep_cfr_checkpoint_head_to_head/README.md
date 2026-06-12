# Experiment 2: Leduc Poker Deep CFR Checkpoint Head-to-Head

## Question

> Do later Deep CFR average-policy checkpoints consistently outperform earlier checkpoints in direct play, and how does that relationship compare with exploitability?

## Design

For each seed, the experiment trains a single Deep CFR run that pauses at every milestone in the checkpoint schedule and writes a *policy snapshot* (the average-policy network plus a small metadata header). Training resumes from the previous milestone's full checkpoint, so the total compute matches a single straight-through 1,500-iteration run.

The saved snapshots are then loaded back into an OpenSpiel `Policy` object and evaluated:

1. exact pairwise expected value between every pair of checkpoints (seat-averaged);
2. exact NashConv / exploitability per checkpoint;
3. monotonicity statistics summarising whether later checkpoints beat earlier checkpoints;
4. an optional Monte Carlo cross-check, for use as a template on larger games where exact evaluation is unavailable.

## Default protocol

- Environment: `leduc_poker`
- Algorithm: Deep CFR (the canonical `deep_cfr_poker.solver.DeepCFRSolver`, identical to experiment 1 except where overridden by config)
- Checkpoint schedule: `[100, 300, 500, 750, 1000, 1250, 1500]`
- Seeds: 3 fixed seeds (`1234, 2025, 31415`) by default
- Policy training cadence: every 10 iterations during stage 1, every 100 iterations afterwards (so the very first checkpoint has a fitted average-policy network)
- Equivalence threshold: pairwise EVs with magnitude `< 1e-3` are treated as practical ties
- Primary metric: per-seed monotonicity of head-to-head EV
- Secondary metric: exploitability over the schedule, plus the equilibrium-quality vs head-to-head-strength scatter

## Run

From the repository root:

```bash
# Train and analyse in one go
python -m experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.run

# Quick smoke test: two seeds, abbreviated schedule
python -m experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.run \
  --seeds 1234,2025 \
  --checkpoint-schedule 100,300,500 \
  --output-root outputs/smoke_tests

# Re-run analysis against an existing run directory without retraining
python -m experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.run analyse \
  --run-dir outputs/leduc_poker_deep_cfr_checkpoint_head_to_head_20260508_120000

# Optional Monte Carlo validation on adjacent pairs only (cheap)
python -m experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.run \
  --run-monte-carlo-validation true \
  --mc-pair-mode adjacent \
  --num-mc-episodes 20000
```

## Outputs

The run directory contains every artefact a thesis chapter needs.

| File | Contents |
| --- | --- |
| `experiment_metadata.json` | Full configuration, software versions, completed seeds. |
| `experiment.log` | Full log mirroring stdout. |
| `failed_seeds.json` | Present only if any seed failed; contains the per-seed traceback. |
| `training_stage_metrics.csv` | One row per (seed, stage) — wall-clock, final exploitability at the milestone, last losses, buffer sizes. |
| `checkpoint_inventory.csv` | Every snapshot file discovered, with size on disk. |
| `loaded_policy_inventory.csv` | Per-loaded-policy metadata (declared vs filename iteration, etc.). |
| `checkpoint_exploitability_metrics.csv` | NashConv, exploitability, average-policy value, signed and absolute value error per (seed, checkpoint). |
| `head_to_head_pairwise.csv` | Long-form pairwise EVs (seat 0, seat 1, seat-averaged). |
| `head_to_head_mean_matrix.csv` | Across-seed mean of seat-averaged EV. |
| `head_to_head_seed_win_fraction_matrix.csv` | Fraction of seeds for which row checkpoint clearly beats column checkpoint. |
| `head_to_head_monotonicity_by_seed.csv` | Per-seed monotonicity statistics. |
| `head_to_head_strength_by_checkpoint.csv` | Per-(seed, checkpoint) head-to-head strength columns, joined with exploitability. |
| `head_to_head_strength_aggregate.csv` | Across-seed mean / SE per checkpoint, plus exploitability and average-policy-value mean / SE. |
| `best_checkpoint_summary.csv` | Best checkpoint per seed under each criterion. |
| `head_to_head_monte_carlo.csv` | Optional Monte Carlo validation rows (only when enabled). |
| `head_to_head_mean_matrix.png` | Across-seed heatmap of pairwise EV. |
| `head_to_head_later_vs_earlier.png` | Lower-triangular view of the heatmap (the central monotonicity figure). |
| `head_to_head_seed_win_fraction.png` | Heatmap of the seed-win-fraction matrix. |
| `head_to_head_strength_vs_earlier.png` | Cross-seed mean EV vs all earlier checkpoints, with SE bars. |
| `head_to_head_strength_vs_previous.png` | Adjacent-checkpoint improvement, with SE bars. |
| `exploitability_by_checkpoint.png` | Mean exploitability per milestone, with SE bars. |
| `average_policy_value_by_checkpoint.png` | Mean average-policy value per milestone, with SE bars. |
| `strength_vs_exploitability.png` | Annotated scatter linking equilibrium quality and head-to-head strength. |
| `strength_vs_average_policy_value.png` | Annotated scatter linking average-policy value and head-to-head strength. |
| `checkpoints/seed_<seed>_iter_<iter>_full.pt` | Full Deep CFR checkpoint (resumable). |
| `snapshots/seed_<seed>_iter_<iter>_snapshot.pt` | Lightweight policy snapshot. |

## Interpretation guidance for the thesis

The headline figure is `head_to_head_later_vs_earlier.png`. If every cell is non-negative, later checkpoints monotonically dominate earlier ones in direct play. Sign changes indicate non-monotonic checkpoint quality.

`exploitability_by_checkpoint.png` should be reported separately. A checkpoint may be lower-exploitability while losing to specific earlier checkpoints in head-to-head play — that contrast, where it occurs, is exactly what this experiment is designed to surface.

The Monte Carlo validation is intentionally off by default. Leduc poker is small enough that exact EV is the cleaner result; the MC code is preserved as a template for larger games where exact evaluation is intractable.
