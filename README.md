# Leduc Poker Deep CFR Experiments

This repository contains reproducible experiments for evaluating Deep Counterfactual Regret Minimisation (Deep CFR) on Leduc poker using DeepMind's OpenSpiel library.

The immediate aim is to validate that the Deep CFR implementation can learn a low-exploitability average policy in a small two-player zero-sum imperfect-information game with known game value, and to characterise its behaviour under controlled diagnostic conditions. This is part of a broader MPhil thesis programme on neural CFR methods for poker, where Leduc poker is used as a diagnostic environment before moving to larger games such as no-limit hold'em abstractions.

The repository is organised so that each experiment can be run independently while sharing as much code as possible. The shared `deep_cfr_poker` package contains a single canonical Deep CFR solver, replay buffer, snapshot utilities, evaluation primitives, and plotting helpers. Each experiment lives in its own package under `experiments/leduc_poker/<experiment_name>/`.

## Repository structure

```text
.
├── deep_cfr_poker/                                   # Shared reusable code
│   ├── solver.py                                     # Canonical Deep CFR solver
│   ├── networks.py                                   # MLP variants and Sonnet-style linear layers
│   ├── replay.py                                     # Reservoir replay buffer
│   ├── snapshots.py                                  # Policy snapshots + LoadedPolicy
│   ├── evaluation.py                                 # Head-to-head + monotonicity analysis
│   ├── experiment_utils.py                           # Shared run-dir lifecycle helpers
│   ├── plotting.py                                   # Curves, heatmaps, scatter plots
│   ├── constants.py                                  # Leduc value and thresholds
│   └── seeding.py                                    # Reproducibility helpers
├── experiments/
│   └── leduc_poker/
│       ├── architecture_ablation_common.py           # Shared architecture-ablation runner
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
│       ├── deep_cfr_replay_averaging_ablation/      # Experiment 10
│       │   ├── config.py
│       │   ├── plotting.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_network_size_ablation/          # Experiment 11
│       │   ├── config.py
│       │   ├── plotting.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_residual_network_ablation/      # Experiment 12
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_layer_norm_network_ablation/    # Experiment 13
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_network_role_ablation/          # Experiment 14
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_shared_trunk_head_ablation/     # Experiment 15
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_factorised_advantage_head_ablation/ # Experiment 16
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_dropout_ablation/              # Experiment 17
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_composite_architecture_validation/ # Experiment 18
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_composite_target_processing_ablation/ # Experiment 19
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_composite_standardized_replay_averaging_ablation/ # Experiment 20
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_composite_policy_extraction_ablation/ # Experiment 21
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_composite_advantage_fitting_ablation/ # Experiment 22
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── deep_cfr_composite_replay_memory_ablation/ # Experiment 23
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       └── deep_cfr_composite_learning_rate_ablation/ # Experiment 24
│           ├── config.py
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

### 11. Leduc poker Deep CFR network-size ablation

[`experiments/leduc_poker/deep_cfr_network_size_ablation/`](experiments/leduc_poker/deep_cfr_network_size_ablation/README.md)

Runs a controlled architecture grid anchored to Experiment 1. Both the advantage networks and average-policy network use hidden depths of `2`, `4`, or `8` layers and hidden widths of `8`, `16`, or `32` units, with all other solver parameters fixed. The `layers2_width32` arm is the Experiment 1 architecture and is used as the paired baseline. The default run uses three matched seeds so that the full grid completes within the 48-hour Batch runtime limit.

**Question:** how sensitive is Deep CFR performance in Leduc poker to neural-network depth and width when the data-generation budget, optimiser, replay capacity, and average-policy training schedule are held fixed?

### 12. Leduc poker Deep CFR residual-network ablation

[`experiments/leduc_poker/deep_cfr_residual_network_ablation/`](experiments/leduc_poker/deep_cfr_residual_network_ablation/README.md)

Compares plain MLPs against residual MLPs at fixed width `32` and hidden depths `2`, `4`, and `8`. Both the advantage networks and average-policy network use the same architecture within each variant.

**Question:** do skip connections improve Deep CFR optimisation stability or final average-policy quality for deeper networks?

### 13. Leduc poker Deep CFR layer-normalisation network ablation

[`experiments/leduc_poker/deep_cfr_layer_norm_network_ablation/`](experiments/leduc_poker/deep_cfr_layer_norm_network_ablation/README.md)

Compares plain MLPs, layer-normalised MLPs, and residual layer-normalised MLPs at fixed width `32` and hidden depths `2`, `4`, and `8`.

**Question:** does normalising hidden activations improve Deep CFR supervised fitting stability under the same data-generation and optimiser budget?

### 14. Leduc poker Deep CFR network-role ablation

[`experiments/leduc_poker/deep_cfr_network_role_ablation/`](experiments/leduc_poker/deep_cfr_network_role_ablation/README.md)

Varies policy-network capacity and advantage-network capacity separately. The baseline is policy `2x32`, advantage `2x32`; treatment arms shrink or deepen only one of the two network roles at a time.

**Question:** is Deep CFR performance more sensitive to the average-policy network architecture or to the advantage-network architecture?

### 15. Leduc poker Deep CFR shared-trunk head ablation

[`experiments/leduc_poker/deep_cfr_shared_trunk_head_ablation/`](experiments/leduc_poker/deep_cfr_shared_trunk_head_ablation/README.md)

Holds the average-policy network fixed at the Experiment 1 `2x32` MLP architecture and compares independent per-player advantage MLPs with a shared advantage trunk and separate player/action heads. The comparison is run at hidden depths `2`, `4`, and `8`, all at width `32`.

**Question:** can shared representations across player-specific advantage approximators improve sample efficiency or optimisation stability without weakening the per-player action-value heads?

### 16. Leduc poker Deep CFR factorised advantage-head ablation

[`experiments/leduc_poker/deep_cfr_factorised_advantage_head_ablation/`](experiments/leduc_poker/deep_cfr_factorised_advantage_head_ablation/README.md)

Holds the average-policy network fixed at the Experiment 1 `2x32` MLP architecture and compares direct action outputs with centred action-advantage outputs and dueling-style state-value-plus-action-advantage outputs. The comparison is run at hidden depths `2`, `4`, and `8`, all at width `32`.

**Question:** does imposing a value/advantage factorisation on the advantage approximator improve Deep CFR optimisation stability or final average-policy quality?

### 17. Leduc poker Deep CFR dropout ablation

[`experiments/leduc_poker/deep_cfr_dropout_ablation/`](experiments/leduc_poker/deep_cfr_dropout_ablation/README.md)

Runs a negative/control-style regularisation test. The average-policy network is fixed at the Experiment 1 `2x32` MLP architecture, while dropout is applied only inside the advantage networks during supervised fitting. The comparison uses dropout probabilities `0.00`, `0.05`, `0.10`, and `0.20` at advantage-network depths `2` and `8`.

**Question:** does dropout regularisation improve advantage fitting and final average-policy quality, or does it mainly add noise to an already stochastic Deep CFR training signal?

### 18. Leduc poker Deep CFR composite architecture validation

[`experiments/leduc_poker/deep_cfr_composite_architecture_validation/`](experiments/leduc_poker/deep_cfr_composite_architecture_validation/README.md)

Tests the proposed architecture that combines the strongest architecture-ablation signals. The average-policy network remains the baseline `2x32` MLP, while the advantage networks compare the original direct `2x32` MLP, the best centred-head `8x32` intervention, and the proposed `8x32` residual LayerNorm centred-advantage network over five matched seeds.

**Question:** do the architectural interventions that looked promising in isolation remain complementary when combined into a single candidate baseline?

### 19. Leduc poker Deep CFR composite target-processing ablation

[`experiments/leduc_poker/deep_cfr_composite_target_processing_ablation/`](experiments/leduc_poker/deep_cfr_composite_target_processing_ablation/README.md)

Mirrors the target-processing ablation on the composite architecture baseline. Raw, standardised, clipped, and standardised-then-clipped advantage targets are compared while the composite architecture, traversal budget, optimiser settings, replay capacity, and average-policy training schedule are held fixed.

**Question:** does advantage-target standardisation still improve Deep CFR after the network architecture has already been strengthened?

### 20. Leduc poker Deep CFR composite standardised replay and average-strategy weighting ablation

[`experiments/leduc_poker/deep_cfr_composite_standardized_replay_averaging_ablation/`](experiments/leduc_poker/deep_cfr_composite_standardized_replay_averaging_ablation/README.md)

Repeats the replay and average-strategy weighting ablation on the current best candidate baseline: composite advantage architecture plus standardised advantage targets. The thesis-facing default contrasts linear CFR-style average-strategy weighting against uniform weighting over five matched seeds; priority-replay variants remain available as optional exploratory arms.

**Question:** after architecture and target processing have been improved, is there additional gain from changing the average-policy target weighting?

### 21. Leduc poker Deep CFR composite policy-extraction HP ablation

[`experiments/leduc_poker/deep_cfr_composite_policy_extraction_ablation/`](experiments/leduc_poker/deep_cfr_composite_policy_extraction_ablation/README.md)

Starts from the current best candidate baseline from experiment 20 and varies only average-policy extraction settings: policy-training cadence, policy-gradient steps, and strategy minibatch size.

**Question:** is the best baseline still limited by how accurately or how often the average-policy network is fitted to the sampled average-strategy targets?

### 22. Leduc poker Deep CFR composite advantage-fitting HP ablation

[`experiments/leduc_poker/deep_cfr_composite_advantage_fitting_ablation/`](experiments/leduc_poker/deep_cfr_composite_advantage_fitting_ablation/README.md)

Starts from the current best candidate baseline and varies only advantage-network fitting effort: supervised gradient steps and advantage minibatch size, including one high-effort combined arm.

**Question:** do the deeper residual LayerNorm advantage networks benefit from more settled supervised regret fitting, or is the 200-step, 1024-batch baseline already sufficient?

### 23. Leduc poker Deep CFR composite replay-memory HP ablation

[`experiments/leduc_poker/deep_cfr_composite_replay_memory_ablation/`](experiments/leduc_poker/deep_cfr_composite_replay_memory_ablation/README.md)

Starts from the current best candidate baseline and varies only replay reservoir capacity, testing `1M`, `2M`, and `5M` against the `10M` baseline.

**Question:** does a stronger Deep CFR baseline prefer fresher replay samples, or does it still need the broader historical coverage of the large reservoir?

### 24. Leduc poker Deep CFR composite learning-rate HP ablation

[`experiments/leduc_poker/deep_cfr_composite_learning_rate_ablation/`](experiments/leduc_poker/deep_cfr_composite_learning_rate_ablation/README.md)

Starts from the current best candidate baseline and varies only constant learning-rate magnitude. This deliberately differs from experiment 7, which tested learning-rate schedules on the older baseline.

**Question:** is the improved baseline sensitive to a modest constant learning-rate change, despite the earlier negative result for scheduled decay?

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

# Experiment 11 — network-size ablation
python -m experiments.leduc_poker.deep_cfr_network_size_ablation.run

# Experiment 11 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_network_size_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --depths 2,4 \
  --widths 8 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 12 — residual-network ablation
python -m experiments.leduc_poker.deep_cfr_residual_network_ablation.run

# Experiment 12 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_residual_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids plain_layers2_width32,residual_layers2_width32 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 13 — layer-normalisation network ablation
python -m experiments.leduc_poker.deep_cfr_layer_norm_network_ablation.run

# Experiment 13 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_layer_norm_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids plain_layers2_width32,layer_norm_layers2_width32,residual_layer_norm_layers2_width32 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 14 — network-role ablation
python -m experiments.leduc_poker.deep_cfr_network_role_ablation.run

# Experiment 14 — quick smoke test
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

# Experiment 15 — shared-trunk head ablation
python -m experiments.leduc_poker.deep_cfr_shared_trunk_head_ablation.run

# Experiment 15 — quick smoke test
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

# Experiment 16 — factorised advantage-head ablation
python -m experiments.leduc_poker.deep_cfr_factorised_advantage_head_ablation.run

# Experiment 16 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_factorised_advantage_head_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids direct_advantage_layers2_width32,centered_advantage_layers2_width32,dueling_advantage_layers2_width32 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 17 — dropout ablation
python -m experiments.leduc_poker.deep_cfr_dropout_ablation.run

# Experiment 17 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_dropout_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids dropout_p00_advantage_layers2_width32,dropout_p05_advantage_layers2_width32 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 18 — composite architecture validation
python -m experiments.leduc_poker.deep_cfr_composite_architecture_validation.run

# Experiment 18 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_composite_architecture_validation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids baseline_direct_2x32,composite_res_ln_centered_advantage_8x32 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 19 — composite target-processing ablation
python -m experiments.leduc_poker.deep_cfr_composite_target_processing_ablation.run

# Experiment 19 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_composite_target_processing_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids composite_raw_targets_baseline,composite_standardized_targets \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 20 — composite standardised replay and averaging ablation
python -m experiments.leduc_poker.deep_cfr_composite_standardized_replay_averaging_ablation.run

# Experiment 20 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_composite_standardized_replay_averaging_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids composite_std_uniform_replay_linear_avg_baseline,composite_std_uniform_replay_uniform_avg \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 21 — composite policy-extraction HP ablation
python -m experiments.leduc_poker.deep_cfr_composite_policy_extraction_ablation.run

# Experiment 21 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_composite_policy_extraction_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids composite_best_baseline,policy_train_every_10 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 22 — composite advantage-fitting HP ablation
python -m experiments.leduc_poker.deep_cfr_composite_advantage_fitting_ablation.run

# Experiment 22 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_composite_advantage_fitting_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids composite_best_baseline,advantage_batch_512 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests

# Experiment 23 — composite replay-memory HP ablation
python -m experiments.leduc_poker.deep_cfr_composite_replay_memory_ablation.run

# Experiment 23 — quick smoke test
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

# Experiment 24 — composite learning-rate HP ablation
python -m experiments.leduc_poker.deep_cfr_composite_learning_rate_ablation.run

# Experiment 24 — quick smoke test
python -m experiments.leduc_poker.deep_cfr_composite_learning_rate_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids composite_best_baseline,learning_rate_0_002 \
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

Exploitability is the primary equilibrium-quality metric. Policy-value error and neural-network losses are useful diagnostics, but they should not be interpreted as evidence of Nash-equilibrium convergence on their own. Head-to-head expected value (experiment 2) is a separate, complementary signal: a low-exploitability checkpoint may still lose to specific earlier checkpoints in direct play. Policy-training frequency and final-only extraction (experiments 3 and 4) change the supervised fitting budget and timing for the average-policy network, advantage-network reinitialisation (experiment 5) changes the regret-approximation optimisation path, the fair warm-start ablation (experiment 6) tests checkpoint/resume fidelity, the learning-rate schedule ablation (experiment 7) changes the optimiser trajectory while holding the Deep CFR data-generation protocol fixed, the constrained random search (experiment 8) screens multiple implementation and optimisation choices under a practical compute budget, the target-processing ablation (experiment 9) changes only the supervised advantage-network targets seen during fitting, the replay/averaging ablation (experiment 10) changes only average-policy target weighting by default, the network-size ablation (experiment 11) changes only the policy and advantage MLP architecture, and experiments 12-17 isolate residual connections, layer normalisation, policy-vs-advantage architecture roles, shared advantage trunks, factorised advantage heads, and dropout. Experiments 18-20 test whether the strongest architecture signals combine into a better candidate baseline, then re-test target processing and average-strategy weighting on that improved baseline rather than assuming effects transfer unchanged from the original configuration. Experiments 21-24 run targeted hyperparameter searches around that best baseline, isolating policy extraction, advantage fitting, replay freshness, and constant learning-rate magnitude. In the thesis, report exploitability, head-to-head strength, supervised update budget, checkpoint fidelity, optimiser schedule, search-stage uncertainty, target-processing diagnostics, optional replay diagnostics, architecture-size diagnostics, and paired ablation differences as distinct quantities, and treat contrasts between them as empirical results rather than failure modes.
