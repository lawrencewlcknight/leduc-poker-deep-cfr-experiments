# Experiment 1: Leduc Poker Deep CFR Multi-Seed Validation

This experiment validates the Deep CFR implementation in OpenSpiel's `leduc_poker` environment.

The experiment trains one Deep CFR model per random seed for a fixed training budget. It evaluates the learned average policy periodically using exact exploitability, reported as NashConv divided by two. The known Leduc poker value for player 0, `-0.085606424078`, is used only as a secondary diagnostic via policy-value error.

## Default protocol

- Environment: `leduc_poker`
- Algorithm: Deep CFR
- Iterations: 1,500
- Traversals per iteration: 320
- Seeds: 10 fixed seeds
- Evaluation interval: every 25 iterations
- Primary metric: exploitability
- Secondary metrics: policy-value error, nodes touched, wall-clock time, policy loss, gradient norms, replay sizes, entropy, legal-action mass, advantage-target variance

## Run

From the repository root:

```bash
python -m experiments.leduc_poker.deep_cfr_multiseed_validation.run
```

For a quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_multiseed_validation.run   --seeds 1234,2025   --iterations 100   --output-root outputs/smoke_tests
```

The experiment writes a timestamped directory containing CSV, JSON, NPZ, and PNG outputs.
The headline PNGs include exploitability, average policy value, and policy-value
error curves by iteration / nodes touched.
