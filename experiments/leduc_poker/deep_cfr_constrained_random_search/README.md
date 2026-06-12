# Leduc Poker Deep CFR Constrained Random Search

This experiment performs a bounded hyperparameter search around the Experiment
2 Deep CFR baseline. It is not intended to be an exhaustive optimiser. The aim
is to identify promising configurations while keeping the Leduc poker runtime
practical and preserving matched-seed comparisons against the baseline.

The default design has two stages:

| Stage | Default protocol |
| --- | --- |
| `screening` | Experiment 2 baseline plus 6 sampled candidates, run for 500 iterations over seeds `1234,2025`. |
| `confirmation` | Experiment 2 baseline plus the top 2 screening candidates, run for 1,500 iterations over seeds `1234,2025,31415`. |

The constrained search space varies learning rate, traversals, network
architecture, replay-memory size, advantage-network batch size, supervised
training steps, and advantage-network reinitialisation. Average-policy
batch size is held fixed to avoid confounding policy-extraction memory
pressure with the tuned advantage-training hyperparameters. Policy-network
training cadence and evaluation cadence remain aligned with Experiment 2
because they have their own dedicated ablations.

## Run

From the repository root:

```bash
python -m experiments.leduc_poker.deep_cfr_constrained_random_search.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_constrained_random_search.run \
  --quick-test \
  --output-root outputs/smoke_tests
```

Useful CLI options:

- `--screening-random-configs`, `--confirmation-top-k`
- `--screening-iterations`, `--confirmation-iterations`
- `--screening-seeds`, `--confirmation-seeds`
- `--use-extended-confirmation-seeds`
- `--master-seed` to reproduce or change the sampled candidates
- baseline overrides such as `--traversals`, `--learning-rate`,
  `--policy-network-layers`, `--advantage-network-layers`,
  `--batch-size-advantage`, `--batch-size-strategy`, and training-step counts

## Outputs

The run directory follows `docs/OUTPUT_CONVENTIONS.md`:

- `seed_summary.csv` — one row per completed `(stage, config_label, seed)`.
- `checkpoint_curves.csv` — one row per checkpoint with `stage`,
  `config_label`, exploitability, policy-value error, learning rate, losses,
  replay sizes, and gradient diagnostics.
- `screening_seed_summary.csv` and `confirmation_seed_summary.csv` — staged
  subsets of `seed_summary.csv`.
- `screening_config_summary.csv` and `confirmation_config_summary.csv` —
  across-seed means, standard deviations, standard errors, and ranks by
  configuration.
- `search_configurations.csv` — all sampled configurations and whether each
  was selected for confirmation.
- `paired_differences_vs_baseline.csv` — confirmation-stage per-seed candidate
  minus baseline differences.
- `paired_difference_summary.json` — aggregate paired differences by confirmed
  candidate.
- `aggregate_summary.json` — nested screening and confirmation summaries by
  `config_label`.
- `search_curves.npz` — compact curve arrays grouped by stage and
  configuration.
- `traces/` — per-run JSON traces containing config, summary, and checkpoint
  curves.
- PNG plots:
  - `screening_final_exploitability_by_config.png`
  - `screening_final_average_policy_value_by_config.png`
  - `confirmation_final_exploitability_by_config.png`
  - `confirmation_final_average_policy_value_by_config.png`
  - `confirmation_exploitability_by_iteration.png`
  - `confirmation_exploitability_by_nodes.png`
  - `confirmation_average_policy_value_by_iteration.png`
  - `confirmation_average_policy_value_by_nodes.png`
  - `confirmation_policy_value_error_by_iteration.png`
  - `confirmation_paired_difference_vs_baseline.png`

Paired differences are reported as `candidate - baseline`; negative
exploitability, AUC, or policy-value-error deltas are improvements.
