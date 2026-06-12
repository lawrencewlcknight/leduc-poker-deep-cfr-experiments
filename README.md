# Leduc Poker Deep CFR Experiments

This repository contains reproducible experiments for evaluating Deep Counterfactual Regret Minimisation (Deep CFR) on Leduc poker using DeepMind's OpenSpiel library.

The immediate aim is to validate that the Deep CFR implementation can learn a low-exploitability average policy in a small two-player zero-sum imperfect-information game with known game value, and to characterise its behaviour under controlled diagnostic conditions. This is part of a broader MPhil thesis programme on neural CFR methods for poker, where Leduc poker is used as a diagnostic environment before moving to larger games such as no-limit hold'em abstractions.

The repository is organised so that each experiment can be run independently while sharing as much code as possible. The shared `deep_cfr_poker` package contains a single canonical Deep CFR solver, replay buffer, snapshot utilities, evaluation primitives, and plotting helpers. Each experiment lives in its own package under `experiments/leduc_poker/<experiment_name>/`.

## Repository structure

```text
.
├── deep_cfr_poker/                                   # Shared reusable code
│   ├── solver.py                                     # Canonical Deep CFR solver
│   ├── networks.py                                   # MLP and Sonnet-style linear layers
│   ├── replay.py                                     # Reservoir replay buffer
│   ├── snapshots.py                                  # Policy snapshots + LoadedPolicy
│   ├── evaluation.py                                 # Head-to-head + monotonicity analysis
│   ├── experiment_utils.py                           # Shared run-dir lifecycle helpers
│   ├── plotting.py                                   # Curves, heatmaps, scatter plots
│   ├── constants.py                                  # Leduc value and thresholds
│   └── seeding.py                                    # Reproducibility helpers
├── experiments/
│   └── leduc_poker/
│       ├── deep_cfr_multiseed_validation/            # Experiment 1
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_checkpoint_head_to_head/         # Experiment 2
│       │   ├── config.py
│       │   ├── train.py
│       │   ├── analyse.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_policy_training_frequency_ablation/ # Experiment 3
│       │   ├── config.py
│       │   ├── plotting.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_final_only_policy_training_ablation/ # Experiment 4
│       │   ├── config.py
│       │   ├── plotting.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_advantage_reinitialisation_ablation/ # Experiment 5
│       │   ├── config.py
│       │   ├── plotting.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_warm_start_fair_ablation/       # Experiment 6
│       │   ├── config.py
│       │   ├── plotting.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_lr_schedule_ablation/           # Experiment 7
│       │   ├── config.py
│       │   ├── plotting.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_constrained_random_search/      # Experiment 8
│       │   ├── config.py
│       │   ├── plotting.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_target_processing_ablation/     # Experiment 9
│       │   ├── config.py
│       │   ├── plotting.py
│       │   ├── run.py
│       │   └── README.md
│       └── deep_cfr_replay_averaging_ablation/      # Experiment 10
│           ├── config.py
│           ├── plotting.py
│           ├── run.py
│           └── README.md
├── tests/                                            # pytest suite
├── docs/
│   └── OUTPUT_CONVENTIONS.md                         # Naming + layout contract
├── outputs/                                          # Experiment outputs (gitignored)
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── TESTING.md
└── CODE_REVIEW.md
```

The contract every experiment satisfies — file names, run-dir layout, CLI flags — is documented in [`docs/OUTPUT_CONVENTIONS.md`](docs/OUTPUT_CONVENTIONS.md). Read that before adding a new experiment.

## Experiments

### 1. Leduc poker Deep CFR multi-seed validation

[`experiments/leduc_poker/deep_cfr_multiseed_validation/`](experiments/leduc_poker/deep_cfr_multiseed_validation/README.md)

Trains Deep CFR on `leduc_poker` for a fixed budget of 1,500 iterations across 10 random seeds. The primary metric is exploitability (NashConv/2). Diagnostics include policy-value error from the known Leduc value, nodes touched, wall-clock, policy loss, gradient norms, replay-buffer sizes, policy entropy, legal-action mass before masking, and advantage-target variance.

**Question:** under a fixed training budget, does the Deep CFR implementation reliably learn a low-exploitability average policy in Leduc poker, and how variable is the result across random seeds?

### 2. Leduc poker Deep CFR checkpoint head-to-head

[`experiments/leduc_poker/deep_cfr_checkpoint_head_to_head/`](experiments/leduc_poker/deep_cfr_checkpoint_head_to_head/README.md)

Trains Deep CFR with a checkpoint schedule of `[100, 300, 500, 750, 1000, 1250, 1500]` iterations, saving a lightweight policy snapshot at every milestone. The saved snapshots are then evaluated against each other in exact pairwise head-to-head play, giving a per-(seed, checkpoint) head-to-head matrix and an across-seed monotonicity summary.

**Question:** do later Deep CFR average-policy checkpoints consistently outperform earlier checkpoints in direct play, and how does that relationship compare with exploitability?

The solver configuration is intentionally identical to experiment 1 so that any difference in results is attributable to the schedule, not to a different solver setup.

### 3. Leduc poker Deep CFR policy-training-frequency ablation

[`experiments/leduc_poker/deep_cfr_policy_training_frequency_ablation/`](experiments/leduc_poker/deep_cfr_policy_training_frequency_ablation/README.md)

Runs a controlled ablation over `policy_network_train_every = [10, 25, 50, 100]` while holding total CFR iterations, traversals, seeds, architecture, optimiser settings, replay capacity, advantage training, and evaluation checkpoints fixed. Evaluation remains at every 25 CFR iterations for all arms, so exploitability and policy-value curves are directly comparable.

**Question:** does training the average-policy network more frequently improve the stability or quality of the evaluated average policy, and how much extra supervised update budget does that require?

### 4. Leduc poker Deep CFR final-only policy-training ablation

[`experiments/leduc_poker/deep_cfr_final_only_policy_training_ablation/`](experiments/leduc_poker/deep_cfr_final_only_policy_training_ablation/README.md)

Compares the intermittent average-policy training baseline against final-only average-policy extraction. The default variants are `intermittent_every_25`, `final_only_200_steps`, and `final_only_matched_steps`; the matched arm trains once at the end with the same total policy-gradient-step budget as the intermittent baseline.

**Question:** is the average-policy network needed during CFR training for final evaluated policy quality, or can it be trained only once after policy memory has been collected?

### 5. Leduc poker Deep CFR advantage-network reinitialisation ablation

[`experiments/leduc_poker/deep_cfr_advantage_reinitialisation_ablation/`](experiments/leduc_poker/deep_cfr_advantage_reinitialisation_ablation/README.md)

Compares two otherwise identical Deep CFR regimes: warm-starting the advantage networks across CFR iterations versus resetting them before each advantage-network training phase. The default protocol uses the same ten seeds as the multi-seed validation experiment and reports paired `True - False` differences.

**Question:** does Brown-style advantage-network reinitialisation improve regret approximation and final average-policy quality under the fixed neural optimisation budget used in these Leduc poker experiments?

### 6. Leduc poker Deep CFR fair warm-start ablation

[`experiments/leduc_poker/deep_cfr_warm_start_fair_ablation/`](experiments/leduc_poker/deep_cfr_warm_start_fair_ablation/README.md)

Compares a continuous cold-start baseline with a matched warm-start arm that trains to an intermediate boundary, saves a full solver checkpoint, reloads into a fresh solver, and continues to the same total iteration budget. The default boundary is 100 iterations within a 1,500-iteration run.

**Question:** does the checkpoint/resume mechanism itself preserve or change Deep CFR learning behaviour under matched seeds and matched compute?

### 7. Leduc poker Deep CFR learning-rate schedule ablation

[`experiments/leduc_poker/deep_cfr_lr_schedule_ablation/`](experiments/leduc_poker/deep_cfr_lr_schedule_ablation/README.md)

Compares the Experiment 2 constant-learning-rate baseline against matched learning-rate schedule arms. The default protocol uses `constant_baseline_exp2` and `cosine_decay_to_10pct`, while optional linear and step-decay arms can be enabled from the CLI.

**Question:** does introducing a learning-rate schedule improve Deep CFR stability or final average-policy quality when every other core training parameter is held fixed?

### 8. Leduc poker Deep CFR constrained random search

[`experiments/leduc_poker/deep_cfr_constrained_random_search/`](experiments/leduc_poker/deep_cfr_constrained_random_search/README.md)

Runs a bounded two-stage hyperparameter search around the Experiment 2 baseline. The screening stage evaluates sampled candidates under a shortened budget; the confirmation stage reruns the baseline plus the strongest screened candidates at the full configured budget over matched seeds.

**Question:** can a constrained random search identify promising Deep CFR configurations without the cost and interpretability problems of a full Cartesian grid search?

### 9. Leduc poker Deep CFR target-processing ablation

[`experiments/leduc_poker/deep_cfr_target_processing_ablation/`](experiments/leduc_poker/deep_cfr_target_processing_ablation/README.md)

Compares raw advantage targets with standardized targets, clipped targets, and standardized-then-clipped targets. Replay buffers retain raw sampled regrets; target processing is applied only to the supervised advantage-network loss.

**Question:** can simple target-processing methods reduce optimisation instability and improve final Deep CFR average-policy quality under the Experiment 2-aligned training budget?

### 10. Leduc poker Deep CFR replay and average-strategy weighting ablation

[`experiments/leduc_poker/deep_cfr_replay_averaging_ablation/`](experiments/leduc_poker/deep_cfr_replay_averaging_ablation/README.md)

Compares the Experiment 2 baseline, `uniform_replay_linear_avg_exp2_baseline`, with uniform average-strategy weighting. Uniform average weighting removes the baseline CFR-style iteration weighting from the average-policy supervised loss. Priority replay variants remain available as optional exploratory arms, but are excluded from the default run because the current priority sampler is too compute intensive for the standard experiment suite.

**Question:** does average-strategy target weighting improve Deep CFR stability or final average-policy quality when every other core training parameter is held fixed?

## Setup

The package metadata requires Python 3.9 or newer (`requires-python = ">=3.9"` in `pyproject.toml`).

Create and activate a virtual environment. The repository contains a placeholder `venv/` directory, but the actual environment is not committed.

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
# .\venv\Scripts\Activate.ps1   # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
# For the test suite and lint config:
pip install -r requirements-dev.txt
```

OpenSpiel installation can vary by platform. If `pip install -r requirements.txt` fails on `open_spiel`, install OpenSpiel following the official instructions for your platform.

## Running the experiments

From the repository root:

```bash
# Experiment 1 — multi-seed validation
python -m experiments.leduc_poker.deep_cfr_multiseed_validation.run

# Experiment 1 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_multiseed_validation.run \
  --seeds 1234,2025 \
  --iterations 100 \
  --output-root outputs/smoke_tests

# Experiment 2 — checkpoint head-to-head (train + analyse)
python -m experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.run

# Experiment 2 — re-run analysis against an existing run dir
python -m experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.run analyse \
  --run-dir outputs/leduc_poker_deep_cfr_checkpoint_head_to_head_20260508_120000

# Experiment 3 — policy-training-frequency ablation
python -m experiments.leduc_poker.deep_cfr_policy_training_frequency_ablation.run

# Experiment 3 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_policy_training_frequency_ablation.run \
  --seeds 1234 \
  --iterations 25 \
  --traversals 8 \
  --evaluation-interval 5 \
  --policy-train-every-variants 1,5 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --output-root outputs/smoke_tests

# Experiment 4 — final-only policy-training ablation
python -m experiments.leduc_poker.deep_cfr_final_only_policy_training_ablation.run

# Experiment 4 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_final_only_policy_training_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --variant-ids intermittent_every_25,final_only_200_steps \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 0 \
  --batch-size-strategy 0 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 5 — advantage-network reinitialisation ablation
python -m experiments.leduc_poker.deep_cfr_advantage_reinitialisation_ablation.run

# Experiment 5 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_advantage_reinitialisation_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 0 \
  --batch-size-strategy 0 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 6 — fair warm-start ablation
python -m experiments.leduc_poker.deep_cfr_warm_start_fair_ablation.run

# Experiment 6 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_warm_start_fair_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --warm-start-boundary 1 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 0 \
  --batch-size-strategy 0 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 7 — learning-rate schedule ablation
python -m experiments.leduc_poker.deep_cfr_lr_schedule_ablation.run

# Experiment 7 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_lr_schedule_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 0 \
  --batch-size-strategy 0 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 8 — constrained random hyperparameter search
python -m experiments.leduc_poker.deep_cfr_constrained_random_search.run

# Experiment 8 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_constrained_random_search.run \
  --quick-test \
  --output-root outputs/smoke_tests

# Experiment 9 — target-processing ablation
python -m experiments.leduc_poker.deep_cfr_target_processing_ablation.run

# Experiment 9 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_target_processing_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids raw_targets_exp2_baseline,standardized_clipped_targets \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 0 \
  --batch-size-strategy 0 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 10 — replay and average-strategy weighting ablation
python -m experiments.leduc_poker.deep_cfr_replay_averaging_ablation.run

# Experiment 10 — quick smoke test
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

Each CLI exposes overrides for the most commonly varied configuration values. See `--help` for the per-experiment flag list, and the experiment's own README for the full output catalogue.

## Outputs

Every experiment writes to a timestamped directory under `outputs/`:

```
outputs/<experiment_name>_<YYYYMMDD_HHMMSS>/
```

with a uniform layout:

- `experiment_metadata.json` — full configuration, software versions, completed seeds
- `experiment.log` — full log file mirroring stdout
- `failed_seeds.json` — present only if at least one seed errored, with traceback
- experiment-specific CSVs / JSONs / NPZs / PNGs (see each experiment's README)
- `checkpoints/` and `snapshots/` subdirectories where applicable

Naming and layout are governed by the contract in [`docs/OUTPUT_CONVENTIONS.md`](docs/OUTPUT_CONVENTIONS.md).
Use [`scripts/promote_thesis_artifacts.py`](scripts/promote_thesis_artifacts.py)
to copy only lightweight thesis-facing figures, tables, and summaries from
downloaded cloud outputs into the tracked `thesis_artifacts/` tree. The workflow
is documented in [`docs/THESIS_ARTIFACTS.md`](docs/THESIS_ARTIFACTS.md).

## Academic interpretation

Exploitability is the primary equilibrium-quality metric. Policy-value error and neural-network losses are useful diagnostics, but they should not be interpreted as evidence of Nash-equilibrium convergence on their own. Head-to-head expected value (experiment 2) is a separate, complementary signal: a low-exploitability checkpoint may still lose to specific earlier checkpoints in direct play. Policy-training frequency and final-only extraction (experiments 3 and 4) change the supervised fitting budget and timing for the average-policy network, advantage-network reinitialisation (experiment 5) changes the regret-approximation optimisation path, the fair warm-start ablation (experiment 6) tests checkpoint/resume fidelity, the learning-rate schedule ablation (experiment 7) changes the optimiser trajectory while holding the Deep CFR data-generation protocol fixed, the constrained random search (experiment 8) screens multiple implementation and optimisation choices under a practical compute budget, the target-processing ablation (experiment 9) changes only the supervised advantage-network targets seen during fitting, and the replay/averaging ablation (experiment 10) changes only average-policy target weighting by default. In the thesis, report exploitability, head-to-head strength, supervised update budget, checkpoint fidelity, optimiser schedule, search-stage uncertainty, target-processing diagnostics, optional replay diagnostics, and paired ablation differences as distinct quantities, and treat contrasts between them as empirical results rather than failure modes.
