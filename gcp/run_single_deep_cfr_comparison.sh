#!/usr/bin/env bash
set -Eeuo pipefail

# One-command, laptop-independent orchestration for Experiment 28:
# cloud smoke -> five parallel seed workers -> aggregate analysis.

ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILDER="$SCRIPT_DIR/single_deep_cfr_comparison_batch.py"

if [[ "$ACTION" == "smoke-local" ]]; then
  cd "$REPO_DIR"
  exec python3 -m experiments.leduc_poker.single_deep_cfr_comparison.run \
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
    --run-dir /tmp/deep-cfr-exp28-sdcfr-smoke
fi

: "${PROJECT_ID:?Set PROJECT_ID to the Google Cloud project}"
: "${REGION:?Set REGION to the Google Cloud Batch region}"
: "${BUCKET:?Set BUCKET to a bucket name or gs:// bucket/prefix}"
: "${SA_EMAIL:?Set SA_EMAIL to the Batch service account}"
: "${REPO_REF:?Set REPO_REF to the pushed Deep CFR repository commit SHA}"

RUN_ID="${RUN_ID:-exp28-sdcfr-$(date -u '+%Y%m%d-%H%M%S')}"
if [[ ${#RUN_ID} -gt 40 || ! "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ ]]; then
  echo "RUN_ID must be 2-40 lowercase letters, digits or hyphens; start with a letter and end alphanumeric" >&2
  exit 2
fi
PARALLELISM="${PARALLELISM:-5}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-STANDARD}"
if [[ "$PROVISIONING_MODEL" != "STANDARD" && "$PROVISIONING_MODEL" != "SPOT" ]]; then
  echo "PROVISIONING_MODEL must be STANDARD or SPOT" >&2
  exit 2
fi
if [[ -n "${MAX_RETRIES:-}" ]]; then
  RETRIES="$MAX_RETRIES"
elif [[ "$PROVISIONING_MODEL" == "SPOT" ]]; then
  RETRIES=2
else
  RETRIES=0
fi
if [[ "$BUCKET" == gs://* ]]; then
  BUCKET_ROOT="${BUCKET%/}"
else
  BUCKET_ROOT="gs://${BUCKET%/}"
fi

SMOKE_JOB="${RUN_ID}-smoke"
TRAIN_JOB="${RUN_ID}-train"
AGGREGATE_JOB="${RUN_ID}-aggregate"
CONTROLLER_JOB="${RUN_ID}-controller"
CONTROLLER_ACTION="orchestrate"
if [[ "$ACTION" == "resume" ]]; then
  RESUME_TAG="${RESUME_TAG:-$(date -u '+%H%M%S')}"
  CONTROLLER_JOB="${RUN_ID}-controller-resume-${RESUME_TAG}"
  CONTROLLER_ACTION="orchestrate-resume"
elif [[ "$ACTION" == "orchestrate-resume" ]]; then
  RESUME_TAG="${RESUME_TAG:-$(date -u '+%H%M%S')}"
  SMOKE_JOB="${RUN_ID}-smoke-retry-${RESUME_TAG}"
  TRAIN_JOB="${RUN_ID}-train-retry-${RESUME_TAG}"
  AGGREGATE_JOB="${RUN_ID}-reaggregate-${RESUME_TAG}"
  CONTROLLER_ACTION="orchestrate-resume"
fi

TEMP_DIR="$(mktemp -d /tmp/deep-cfr-exp28-batch.XXXXXX)"
trap 'rm -rf "$TEMP_DIR"' EXIT

build_json() {
  local kind="$1"
  local output="$2"
  local model="$PROVISIONING_MODEL"
  local retries="$RETRIES"
  if [[ "$kind" != "train" ]]; then
    model=STANDARD
    retries=0
  fi
  python3 "$BUILDER" \
    --kind "$kind" \
    --output "$output" \
    --run-id "$RUN_ID" \
    --bucket-root "$BUCKET_ROOT" \
    --service-account "$SA_EMAIL" \
    --repo-ref "$REPO_REF" \
    --parallelism "$PARALLELISM" \
    --provisioning-model "$model" \
    --max-retries "$retries" \
    --project-id "$PROJECT_ID" \
    --region "$REGION" \
    --controller-action "$CONTROLLER_ACTION"
}

submit_job() {
  gcloud batch jobs submit "$1" \
    --project "$PROJECT_ID" --location "$REGION" --config "$2"
}

job_state() {
  gcloud batch jobs describe "$1" \
    --project "$PROJECT_ID" --location "$REGION" \
    --format='value(status.state)'
}

wait_for_job() {
  local name="$1"
  local state
  while true; do
    state="$(job_state "$name")"
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $name: $state"
    case "$state" in
      SUCCEEDED) return 0 ;;
      FAILED|DELETION_IN_PROGRESS) return 1 ;;
    esac
    sleep 30
  done
}

ensure_job_succeeds() {
  local name="$1"
  local config="$2"
  local state
  if state="$(job_state "$name" 2>/dev/null)"; then
    if [[ "$state" == "SUCCEEDED" ]]; then return 0; fi
    if [[ "$state" == "FAILED" || "$state" == "DELETION_IN_PROGRESS" ]]; then
      echo "$name is terminal with state $state" >&2
      return 1
    fi
    wait_for_job "$name"
    return
  fi
  submit_job "$name" "$config"
  wait_for_job "$name"
}

complete_or_retry() {
  local primary="$1"
  local retry="$2"
  local config="$3"
  local state
  if state="$(job_state "$primary" 2>/dev/null)"; then
    if [[ "$state" == "SUCCEEDED" ]]; then return 0; fi
    if [[ "$state" != "FAILED" && "$state" != "DELETION_IN_PROGRESS" ]]; then
      if wait_for_job "$primary"; then return 0; fi
    fi
  fi
  submit_job "$retry" "$config"
  wait_for_job "$retry"
}

build_json controller "$TEMP_DIR/controller.json"
build_json smoke "$TEMP_DIR/smoke.json"
build_json train "$TEMP_DIR/train.json"
build_json aggregate "$TEMP_DIR/aggregate.json"

case "$ACTION" in
  dry-run)
    cp "$TEMP_DIR/controller.json" "$REPO_DIR/exp28_controller_job.json"
    cp "$TEMP_DIR/smoke.json" "$REPO_DIR/exp28_smoke_job.json"
    cp "$TEMP_DIR/train.json" "$REPO_DIR/exp28_train_job.json"
    cp "$TEMP_DIR/aggregate.json" "$REPO_DIR/exp28_aggregate_job.json"
    ;;
  status)
    gcloud batch jobs list \
      --project "$PROJECT_ID" --location "$REGION" \
      --filter="name:${RUN_ID}" \
      --format='table(name.basename(),status.state,createTime)'
    echo "Artifacts: $BUCKET_ROOT/$RUN_ID/"
    ;;
  smoke-cloud)
    submit_job "$SMOKE_JOB" "$TEMP_DIR/smoke.json"
    ;;
  run)
    submit_job "$CONTROLLER_JOB" "$TEMP_DIR/controller.json"
    echo "Remote controller submitted: $CONTROLLER_JOB"
    echo "The laptop may now be disconnected or switched off."
    echo "Artifacts: $BUCKET_ROOT/$RUN_ID/"
    ;;
  resume)
    submit_job "$CONTROLLER_JOB" "$TEMP_DIR/controller.json"
    echo "Remote recovery controller submitted: $CONTROLLER_JOB"
    ;;
  orchestrate)
    [[ "${EXP28_REMOTE_CONTROLLER:-}" == "1" ]] || exit 2
    ensure_job_succeeds "$SMOKE_JOB" "$TEMP_DIR/smoke.json"
    ensure_job_succeeds "$TRAIN_JOB" "$TEMP_DIR/train.json"
    ensure_job_succeeds "$AGGREGATE_JOB" "$TEMP_DIR/aggregate.json"
    ;;
  orchestrate-resume)
    [[ "${EXP28_REMOTE_CONTROLLER:-}" == "1" ]] || exit 2
    complete_or_retry "${RUN_ID}-smoke" "$SMOKE_JOB" "$TEMP_DIR/smoke.json"
    complete_or_retry "${RUN_ID}-train" "$TRAIN_JOB" "$TEMP_DIR/train.json"
    complete_or_retry \
      "${RUN_ID}-aggregate" "$AGGREGATE_JOB" "$TEMP_DIR/aggregate.json"
    ;;
  *)
    echo "Usage: $0 [run|resume|status|smoke-local|smoke-cloud|dry-run]" >&2
    exit 2
    ;;
esac
