#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

JOB_NAME="${1:-smoke-exp25-parallel-equivalence-$(date +%Y%m%d-%H%M%S)}"

exec "${SCRIPT_DIR}/submit_batch_experiment.sh" \
  "${JOB_NAME}" \
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
    --replay-buffer-type compact \
    --output-root outputs/cloud/smoke-exp25-parallel-equivalence" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
