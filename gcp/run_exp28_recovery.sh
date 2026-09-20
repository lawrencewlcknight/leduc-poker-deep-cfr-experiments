#!/usr/bin/env bash
set -Eeuo pipefail

# Recover the analysable portion of the original failed Experiment 28 run on a
# cloud VM. The generic Batch wrapper installs the repository environment and
# uploads the lightweight recovered analysis after this command completes.

ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${PROJECT_ID:?Set PROJECT_ID to the Google Cloud project}"
: "${REGION:?Set REGION to the Google Cloud Batch region}"
: "${BUCKET:?Set BUCKET to a bucket name or gs:// bucket/prefix}"
: "${SA_EMAIL:?Set SA_EMAIL to the Batch service account}"
: "${REPO_REF:?Set REPO_REF to the pushed Deep CFR repository commit SHA}"

SOURCE_RUN_ID="${SOURCE_RUN_ID:-leduc-deep-cfr-exp28-sdcfr-20260920-021836}"
SOURCE_INNER_RUN="${SOURCE_INNER_RUN:-leduc_poker_single_deep_cfr_comparison_20260920_022034}"
RECOVERY_JOB="${RECOVERY_JOB:-exp28-recovery-$(date -u '+%Y%m%d-%H%M%S')}"
if [[ ${#RECOVERY_JOB} -gt 63 || ! "$RECOVERY_JOB" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ ]]; then
  echo "RECOVERY_JOB must be 2-63 lowercase letters, digits or hyphens" >&2
  exit 2
fi
if [[ "$BUCKET" == gs://* ]]; then
  BUCKET_ROOT="${BUCKET%/}"
else
  BUCKET_ROOT="gs://${BUCKET%/}"
fi

SOURCE_ROOT="$BUCKET_ROOT/$SOURCE_RUN_ID/outputs/cloud/$SOURCE_RUN_ID/$SOURCE_INNER_RUN"
RESULT_ROOT="$BUCKET_ROOT/$RECOVERY_JOB/outputs/cloud/$RECOVERY_JOB"

case "$ACTION" in
  run)
    RECOVERY_COMMAND="set -Eeuo pipefail
INPUT_ROOT=/workspace/exp28-recovery-input
OUTPUT_ROOT=outputs/cloud/$RECOVERY_JOB
mkdir -p \"\$INPUT_ROOT/policy_snapshots\" \"\$INPUT_ROOT/sd_cfr_archives\" \"\$OUTPUT_ROOT\"
gcloud storage rsync --recursive '$SOURCE_ROOT/policy_snapshots' \"\$INPUT_ROOT/policy_snapshots\"
gcloud storage rsync --recursive '$SOURCE_ROOT/sd_cfr_archives' \"\$INPUT_ROOT/sd_cfr_archives\"
python -m experiments.leduc_poker.single_deep_cfr_comparison.recover_failed_run \\
  --input-dir \"\$INPUT_ROOT\" \\
  --output-dir \"\$OUTPUT_ROOT\""
    "$SCRIPT_DIR/submit_batch_experiment.sh" \
      "$RECOVERY_JOB" \
      "$RECOVERY_COMMAND" \
      "n2-standard-4" \
      "14400" \
      "4000" \
      "16000"
    echo "Recovery submitted. The laptop may now be disconnected."
    echo "Results will be uploaded to: $RESULT_ROOT/"
    ;;
  status)
    gcloud batch jobs describe "$RECOVERY_JOB" \
      --project "$PROJECT_ID" \
      --location "$REGION" \
      --format='table(name.basename(),status.state,createTime,status.runDuration)'
    echo "Results: $RESULT_ROOT/"
    ;;
  *)
    echo "Usage: $0 [run|status]" >&2
    exit 2
    ;;
esac
