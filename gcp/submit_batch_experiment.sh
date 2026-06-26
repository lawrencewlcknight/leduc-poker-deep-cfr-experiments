#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

# Usage:
#   ./gcp/submit_batch_experiment.sh \
#     JOB_NAME \
#     "PYTHON_EXPERIMENT_COMMAND" \
#     MACHINE_TYPE \
#     MAX_RUN_SECONDS \
#     CPU_MILLI \
#     MEMORY_MIB
#
# Examples:
#   n2-standard-2: CPU_MILLI=2000 MEMORY_MIB=8000
#   n2-standard-4: CPU_MILLI=4000 MEMORY_MIB=16000
#   n2-standard-8: CPU_MILLI=8000 MEMORY_MIB=32000

JOB_NAME="$1"
EXPERIMENT_COMMAND="$2"
MACHINE_TYPE="${3:-n2-standard-4}"
MAX_RUN_SECONDS="${4:-21600}"
CPU_MILLI="${5:-4000}"
MEMORY_MIB="${6:-16000}"

: "${PROJECT_ID:?Set PROJECT_ID first}"
: "${REGION:?Set REGION first}"
: "${BUCKET:?Set BUCKET first}"
: "${SA_EMAIL:?Set SA_EMAIL first}"

JOB_JSON="$(mktemp "/tmp/${JOB_NAME}.XXXXXX.json")"

export JOB_NAME
export EXPERIMENT_COMMAND
export MACHINE_TYPE
export MAX_RUN_SECONDS
export CPU_MILLI
export MEMORY_MIB
export BUCKET
export SA_EMAIL
export JOB_JSON

python3 <<'PY'
import json
import os

job_json_path = os.environ["JOB_JSON"]
job_name = os.environ["JOB_NAME"]
experiment_command = os.environ["EXPERIMENT_COMMAND"]
machine_type = os.environ["MACHINE_TYPE"]
max_run_seconds = os.environ["MAX_RUN_SECONDS"]
cpu_milli = int(os.environ["CPU_MILLI"])
memory_mib = int(os.environ["MEMORY_MIB"])
bucket = os.environ["BUCKET"]
service_account = os.environ["SA_EMAIL"]

script = f"""#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "Starting job: {job_name}"
echo "Experiment command: {experiment_command}"
echo "Requested CPU milli: {cpu_milli}"
echo "Requested memory MiB: {memory_mib}"

if command -v sudo >/dev/null 2>&1; then
  SUDO=sudo
else
  SUDO=
fi

$SUDO apt-get update
$SUDO apt-get install -y git python3-pip python3-dev python3-venv

WORKDIR=/workspace
mkdir -p "$WORKDIR"
cd "$WORKDIR"

git clone --depth 1 https://github.com/lawrencewlcknight/leduc-poker-deep-cfr-experiments.git
cd leduc-poker-deep-cfr-experiments

export HOME="${{HOME:-/root}}"
export TMPDIR="/tmp"
export PIP_CACHE_DIR="/tmp/pip-cache"
export PATH="/usr/local/bin:$PATH"

mkdir -p "$HOME" "$TMPDIR" "$PIP_CACHE_DIR"

# Log basic machine information for later VM right-sizing.
echo "Machine information:"
nproc || true
free -h || true
df -h || true
lscpu | head -30 || true

# Keep experiment dependencies isolated from the Google Cloud CLI Python runtime.
# This avoids breaking Cloud SDK commands after the experiment has finished.
python3 -m venv --copies /tmp/leduc-venv
source /tmp/leduc-venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-cache-dir --no-build-isolation -e .
python -m pip install --no-cache-dir "google-cloud-storage>=2.16,<4.0"
python -m pip check || true

PYTHON_BIN="$(command -v python)"
UPLOAD_DESTINATION="{bucket}/{job_name}/"
UPLOAD_INTERVAL_SECONDS="${{UPLOAD_INTERVAL_SECONDS:-1800}}"

upload_outputs() {{
  if [ -d outputs ]; then
    echo "Uploading outputs to Cloud Storage: $UPLOAD_DESTINATION"
    "$PYTHON_BIN" scripts/upload_outputs_to_gcs.py outputs "$UPLOAD_DESTINATION" || true
  else
    echo "No outputs directory found yet; skipping upload."
  fi
}}

periodic_upload_outputs() {{
  while true; do
    sleep "$UPLOAD_INTERVAL_SECONDS" || break
    echo "Periodic output upload."
    upload_outputs
  done
}}

cleanup_uploads() {{
  status=$?
  if [ -n "${{UPLOAD_PID:-}}" ]; then
    kill "$UPLOAD_PID" 2>/dev/null || true
    wait "$UPLOAD_PID" 2>/dev/null || true
  fi
  echo "Final output upload before exit. Script status: $status"
  upload_outputs
  exit "$status"
}}

periodic_upload_outputs &
UPLOAD_PID="$!"
trap cleanup_uploads EXIT

mkdir -p "outputs/cloud/{job_name}"

{experiment_command}

echo "Experiment completed."

deactivate

echo "Done."
"""

job = {
    "taskGroups": [
        {
            "taskSpec": {
                "runnables": [
                    {
                        "script": {
                            "text": script
                        }
                    }
                ],
                "computeResource": {
                    "cpuMilli": cpu_milli,
                    "memoryMib": memory_mib,
                },
                "maxRetryCount": 0,
                "maxRunDuration": f"{max_run_seconds}s",
            },
            "taskCount": 1,
            "parallelism": 1,
        }
    ],
    "allocationPolicy": {
        "serviceAccount": {
            "email": service_account
        },
        "instances": [
            {
                "policy": {
                    "machineType": machine_type,
                    "provisioningModel": "STANDARD",
                }
            }
        ],
    },
    "logsPolicy": {
        "destination": "CLOUD_LOGGING"
    },
}

with open(job_json_path, "w", encoding="utf-8") as f:
    json.dump(job, f, indent=2)
PY

echo "Submitting Batch job: ${JOB_NAME}"
echo "Machine type: ${MACHINE_TYPE}"
echo "Max run duration: ${MAX_RUN_SECONDS}s"
echo "CPU milli: ${CPU_MILLI}"
echo "Memory MiB: ${MEMORY_MIB}"
echo "Job config: ${JOB_JSON}"

echo
echo "Script that will run inside Batch:"
echo "-----------------------------------"
python3 - "$JOB_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    job = json.load(f)

print(job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"])
PY
echo "-----------------------------------"
echo

gcloud batch jobs submit "${JOB_NAME}" \
  --location "${REGION}" \
  --config "${JOB_JSON}"

echo "Submitted."
echo "Monitor with:"
echo "  gcloud batch jobs describe ${JOB_NAME} --location ${REGION}"
echo "Outputs will be copied to:"
echo "  ${BUCKET}/${JOB_NAME}/"
