#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILDER="$SCRIPT_DIR/single_deep_cfr_paper_aligned_36h_trajectory_batch.py"
MODULE="experiments.leduc_poker.single_deep_cfr_paper_aligned_36h_trajectory.cloud"

if [[ "$ACTION" == "smoke-local" ]]; then
  SMOKE_OUTPUT="${SMOKE_OUTPUT:-/tmp/deep-cfr-exp30-sdcfr36h-smoke}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/deep-cfr-exp30-matplotlib}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/deep-cfr-exp30-cache}"
  mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
  cd "$REPO_DIR"
  exec python3 -m "$MODULE" smoke --output-root "$SMOKE_OUTPUT" --no-resume
fi

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed Experiment 30 commit SHA}"

if ! git -C "$REPO_DIR" cat-file -e "${REPO_REF}^{commit}" 2>/dev/null; then
  echo "REPO_REF is not a commit in this local checkout: $REPO_REF" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-exp30-paper-sdcfr36h-$(date -u '+%Y%m%d-%H%M%S')}"
if [[ ${#RUN_ID} -gt 40 || ! "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ ]]; then
  echo "RUN_ID must be 2-40 lowercase letters, digits or hyphens" >&2
  exit 2
fi
if [[ "$BUCKET" == gs://* ]]; then BUCKET_ROOT="${BUCKET%/}"; else BUCKET_ROOT="gs://${BUCKET%/}"; fi
PARALLELISM="${PARALLELISM:-5}"
if [[ ! "$PARALLELISM" =~ ^[1-5]$ ]]; then
  echo "PARALLELISM must be between 1 and 5" >&2
  exit 2
fi

SMOKE_JOB="${RUN_ID}-smoke"
TRAIN_JOB="${RUN_ID}-train"
EVALUATE_JOB="${RUN_ID}-evaluate"
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
  EVALUATE_JOB="${RUN_ID}-evaluate-retry-${RESUME_TAG}"
  AGGREGATE_JOB="${RUN_ID}-reaggregate-${RESUME_TAG}"
  CONTROLLER_ACTION="orchestrate-resume"
fi

TEMP_DIR="$(mktemp -d /tmp/deep-cfr-exp30-batch.XXXXXX)"
trap 'rm -rf "$TEMP_DIR"' EXIT

build_json() {
  python3 "$BUILDER" --kind "$1" --output "$2" --run-id "$RUN_ID" \
    --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" \
    --repo-ref "$REPO_REF" --project-id "$PROJECT_ID" --region "$REGION" \
    --parallelism "$PARALLELISM" --controller-action "$CONTROLLER_ACTION"
}
submit_job() {
  gcloud batch jobs submit "$1" --project "$PROJECT_ID" --location "$REGION" --config "$2"
}
verify_controller_iam() {
  local member="serviceAccount:${SA_EMAIL}"
  local batch_role self_act_as
  batch_role="$(gcloud projects get-iam-policy "$PROJECT_ID" \
    --flatten='bindings[].members' \
    --filter="bindings.role=roles/batch.jobsEditor AND bindings.members=${member}" \
    --format='value(bindings.role)' --limit=1)"
  self_act_as="$(gcloud iam service-accounts get-iam-policy "$SA_EMAIL" \
    --project "$PROJECT_ID" --flatten='bindings[].members' \
    --filter="bindings.role=roles/iam.serviceAccountUser AND bindings.members=${member}" \
    --format='value(bindings.role)' --limit=1)"
  if [[ "$batch_role" != "roles/batch.jobsEditor" || "$self_act_as" != "roles/iam.serviceAccountUser" ]]; then
    cat >&2 <<EOF
The remote controller service account is missing required child-job permissions.
Run these one-time commands, then submit again:

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="$member" --role="roles/batch.jobsEditor"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project="$PROJECT_ID" --member="$member" --role="roles/iam.serviceAccountUser"
EOF
    return 1
  fi
}
job_state() {
  gcloud batch jobs describe "$1" --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)'
}
wait_for_job() {
  local state
  while true; do
    if ! state="$(job_state "$1" 2>&1)"; then
      echo "Unable to inspect child Batch job $1: $state" >&2
      return 1
    fi
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $1: $state"
    case "$state" in SUCCEEDED) return 0 ;; FAILED|DELETION_IN_PROGRESS) return 1 ;; esac
    sleep 30
  done
}
ensure_job_succeeds() {
  local state
  if state="$(job_state "$1" 2>/dev/null)"; then
    [[ "$state" == "SUCCEEDED" ]] && return 0
    [[ "$state" == "FAILED" || "$state" == "DELETION_IN_PROGRESS" ]] && return 1
    wait_for_job "$1"; return
  fi
  submit_job "$1" "$2"; wait_for_job "$1"
}
complete_or_retry() {
  local state
  if state="$(job_state "$1" 2>/dev/null)"; then
    [[ "$state" == "SUCCEEDED" ]] && return 0
    if [[ "$state" != "FAILED" && "$state" != "DELETION_IN_PROGRESS" ]]; then
      wait_for_job "$1" && return 0
    fi
  fi
  submit_job "$2" "$3"; wait_for_job "$2"
}

build_json controller "$TEMP_DIR/controller.json"
build_json smoke "$TEMP_DIR/smoke.json"
build_json train "$TEMP_DIR/train.json"
build_json evaluate "$TEMP_DIR/evaluate.json"
build_json aggregate "$TEMP_DIR/aggregate.json"

case "$ACTION" in
  dry-run)
    cp "$TEMP_DIR/controller.json" "$REPO_DIR/exp30_controller_job.json"
    cp "$TEMP_DIR/smoke.json" "$REPO_DIR/exp30_smoke_job.json"
    cp "$TEMP_DIR/train.json" "$REPO_DIR/exp30_train_job.json"
    cp "$TEMP_DIR/evaluate.json" "$REPO_DIR/exp30_evaluate_job.json"
    cp "$TEMP_DIR/aggregate.json" "$REPO_DIR/exp30_aggregate_job.json"
    ;;
  status)
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
      --filter="name:${RUN_ID}" --format='table(name.basename(),status.state,createTime)'
    echo "Artifacts: $BUCKET_ROOT/$RUN_ID/"
    ;;
  smoke-cloud)
    submit_job "$SMOKE_JOB" "$TEMP_DIR/smoke.json"
    ;;
  run|resume)
    verify_controller_iam
    submit_job "$CONTROLLER_JOB" "$TEMP_DIR/controller.json"
    echo "Remote Deep CFR Experiment 30 controller submitted: $CONTROLLER_JOB"
    echo "The laptop may now be disconnected or switched off."
    ;;
  orchestrate)
    [[ "${EXP30_REMOTE_CONTROLLER:-}" == "1" ]] || { echo "Internal action" >&2; exit 2; }
    ensure_job_succeeds "$SMOKE_JOB" "$TEMP_DIR/smoke.json" || {
      echo "Cloud smoke failed; production was not submitted." >&2; exit 1;
    }
    ensure_job_succeeds "$TRAIN_JOB" "$TEMP_DIR/train.json"
    ensure_job_succeeds "$EVALUATE_JOB" "$TEMP_DIR/evaluate.json"
    ensure_job_succeeds "$AGGREGATE_JOB" "$TEMP_DIR/aggregate.json"
    ;;
  orchestrate-resume)
    [[ "${EXP30_REMOTE_CONTROLLER:-}" == "1" ]] || { echo "Internal action" >&2; exit 2; }
    complete_or_retry "${RUN_ID}-smoke" "$SMOKE_JOB" "$TEMP_DIR/smoke.json"
    complete_or_retry "${RUN_ID}-train" "$TRAIN_JOB" "$TEMP_DIR/train.json"
    complete_or_retry "${RUN_ID}-evaluate" "$EVALUATE_JOB" "$TEMP_DIR/evaluate.json"
    complete_or_retry "${RUN_ID}-aggregate" "$AGGREGATE_JOB" "$TEMP_DIR/aggregate.json"
    ;;
  *)
    echo "Usage: $0 [run|resume|smoke-local|smoke-cloud|status|dry-run]" >&2
    exit 2
    ;;
esac
