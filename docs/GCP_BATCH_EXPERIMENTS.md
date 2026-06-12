# Running the Deep CFR experiments on Google Cloud Batch

This guide explains how to run the repository experiments on Google Cloud using Google Batch. The workflow is designed to be repeatable and mostly command-line driven:

1. configure Google Cloud locally;
2. create a Cloud Storage bucket for outputs;
3. create a service account for Batch jobs;
4. create the Batch submission script;
5. run a smoke test;
6. run full experiments with configurable CPU and memory;
7. inspect logs and retrieve outputs.

The Batch job creates a temporary VM, clones this GitHub repository, creates an isolated Python virtual environment, installs the repo dependencies, runs the selected experiment, copies outputs to Cloud Storage, and then exits. Batch handles VM lifecycle management, so there is no persistent VM to shut down after a successful job.

---

## 1. Prerequisites

You need:

- a Google Cloud project with billing enabled;
- the Google Cloud CLI installed on your local machine;
- permission to create service accounts, IAM bindings, Batch jobs, and Cloud Storage buckets;
- this GitHub repository available to the Batch VM.

If the repository is public, the script can clone it directly with HTTPS. If the repository is private, adapt the `git clone` step to use an authenticated method such as a deploy key, GitHub token, or a pre-built container image.

---

## 2. One-time local Google Cloud setup

Authenticate and select your project:

```bash
gcloud init
gcloud auth login

export PROJECT_ID="your-gcp-project-id"
gcloud config set project "$PROJECT_ID"
```

For UK-based use where latency is not important and cost is a consideration, `europe-west1` is a sensible default region:

```bash
export REGION="europe-west1"
export ZONE="europe-west1-b"
```

Enable the required APIs:

```bash
gcloud services enable \
  compute.googleapis.com \
  batch.googleapis.com \
  logging.googleapis.com \
  storage.googleapis.com
```

---

## 3. Create a Cloud Storage bucket for experiment outputs

Create a regional bucket in the same region as the Batch jobs:

```bash
export BUCKET_NAME="${PROJECT_ID}-leduc-poker-results"
export BUCKET="gs://${BUCKET_NAME}"

gcloud storage buckets create "$BUCKET" \
  --location="$REGION" \
  --uniform-bucket-level-access
```

Check the bucket exists:

```bash
gcloud storage buckets describe "$BUCKET"
```

If the bucket exists and is accessible, this command will print metadata such as the bucket name, creation time, location, storage class, and storage URL.

---

## 4. Create a service account for Batch jobs

Create a dedicated service account:

```bash
export SA_NAME="leduc-experiment-runner"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --display-name="Leduc poker experiment runner" \
  --project="$PROJECT_ID"
```

Grant the service account permission to write logs:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter"
```

Grant the service account permission to report Batch agent status:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/batch.agentReporter"
```

Grant the service account permission to write experiment outputs to the bucket:

```bash
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"
```

Allow your user account to run jobs as this service account. Replace the email address with your Google account email:

```bash
export YOUR_EMAIL="your-email@example.com"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/iam.serviceAccountUser"
```

If you need to inspect logs from your local account, make sure your user has log-viewing permission:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/logging.viewer"
```

---

## 5. Environment variables to set in each new terminal

Before submitting jobs from a new shell session, set:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"
export BUCKET="gs://${PROJECT_ID}-leduc-poker-results"
export SA_EMAIL="leduc-experiment-runner@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
```

Check the values:

```bash
echo "$PROJECT_ID"
echo "$REGION"
echo "$BUCKET"
echo "$SA_EMAIL"
```

---

## 6. Create the Batch submission script

Create a `gcp` directory in the repository root:

```bash
mkdir -p gcp
```

Create the script:

```bash
nano gcp/submit_batch_experiment.sh
```

Paste the following content into the file:

```bash
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

mkdir -p "outputs/cloud/{job_name}"

{experiment_command}

echo "Experiment completed. Copying outputs to Cloud Storage."
python scripts/upload_outputs_to_gcs.py outputs "{bucket}/{job_name}/"

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
```

Make the script executable and check its syntax:

```bash
chmod +x gcp/submit_batch_experiment.sh
bash -n gcp/submit_batch_experiment.sh
```

`bash -n` should return silently. If it prints an error, fix the script before submitting a Batch job.

---

## 7. Run a smoke test

Before running a full experiment, submit a very small smoke test:

```bash
./gcp/submit_batch_experiment.sh \
  "smoke-exp1-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_multiseed_validation.run \
    --seeds 1234 \
    --iterations 10 \
    --traversals 4 \
    --evaluation-interval 5 \
    --output-root outputs/cloud/smoke-exp1" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

The script prints the Batch script before submission. Check that the upload line uses the repository Python uploader, for example:

```bash
python scripts/upload_outputs_to_gcs.py outputs "gs://your-project-id-leduc-poker-results/smoke-exp1-.../"
```

To see if the experiment is running, list the Batch jobs in the configured region:

```bash
gcloud batch jobs list --location "$REGION"
```

---

## 8. Monitor a Batch job

List jobs:

```bash
gcloud batch jobs list --location "$REGION"
```

Describe one job:

```bash
gcloud batch jobs describe JOB_NAME --location "$REGION"
```

After an experiment has completed, review this command's output to see how long
the Batch job took to run. The `status.runDuration` field reports the time the
job spent in the `RUNNING` state.

Possible states include:

- `QUEUED`
- `SCHEDULED`
- `RUNNING`
- `SUCCEEDED`
- `FAILED`

If the job succeeds, list the uploaded outputs:

```bash
gcloud storage ls -r "$BUCKET/JOB_NAME/"
```

Copy outputs back to your local machine:

```bash
mkdir -p cloud_outputs/JOB_NAME
gcloud storage cp -r "$BUCKET/JOB_NAME/*" "cloud_outputs/JOB_NAME/"
```

The downloaded directory is a scratch copy of the complete Batch output. To
copy only lightweight thesis-facing artifacts such as figures, CSV tables, and
aggregate summaries into the tracked repository tree, run:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME
```

This writes curated files under `thesis_artifacts/<experiment_name>/<run>/`.
See [`docs/THESIS_ARTIFACTS.md`](THESIS_ARTIFACTS.md) for the full workflow,
including dry-run and overwrite options.

---

## 9. Read logs for a failed job

First describe the job and find its `uid`:

```bash
gcloud batch jobs describe JOB_NAME --location "$REGION"
```

Then query task logs using the job UID:

```bash
gcloud logging read \
  'logName="projects/YOUR_PROJECT_ID/logs/batch_task_logs" AND labels.job_uid="JOB_UID"' \
  --limit=500 \
  --format="value(timestamp,severity,textPayload,jsonPayload.message)"
```

For this project, the command has the form:

```bash
gcloud logging read \
  'logName="projects/clever-overview-399515/logs/batch_task_logs" AND labels.job_uid="JOB_UID"' \
  --limit=500 \
  --format="value(timestamp,severity,textPayload,jsonPayload.message)"
```

Useful things to look for:

- `Completed ... seeds` means the experiment itself completed;
- `All outputs saved to:` shows where the experiment wrote local VM outputs;
- `No space left on device` indicates disk pressure;
- `Killed`, `exit code 137`, or `Out of memory` usually indicates memory pressure;
- `maxRunDuration` means the job hit the time limit;
- `Invalid machine type` or resource errors usually mean the requested CPU/memory does not fit the selected machine type.

---

## 10. Run the full Experiment 1

After the smoke test succeeds, run the full multi-seed validation experiment.

A safe starting configuration is `n2-standard-4`, 4 vCPUs, 16 GiB memory, and a 12-hour runtime cap:

```bash
./gcp/submit_batch_experiment.sh \
  "exp1-validation-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_multiseed_validation.run \
    --output-root outputs/cloud/exp1-validation" \
  "n2-standard-4" \
  "43200" \
  "4000" \
  "16000"
```

To collect process-level runtime and memory diagnostics, wrap the experiment command with `/usr/bin/time -v`:

```bash
./gcp/submit_batch_experiment.sh \
  "exp1-validation-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.deep_cfr_multiseed_validation.run \
    --output-root outputs/cloud/exp1-validation" \
  "n2-standard-4" \
  "43200" \
  "4000" \
  "16000"
```

In the logs, look for:

- `Elapsed (wall clock) time`;
- `Percent of CPU this job got`;
- `Maximum resident set size`.

These help determine whether the VM is over- or under-sized.

---

## 11. Changing CPU and memory

The final three arguments control the VM and Batch task resources:

```bash
MACHINE_TYPE MAX_RUN_SECONDS CPU_MILLI MEMORY_MIB
```

Examples:

| Machine type | CPU milli | Memory MiB | Approximate resources |
|---|---:|---:|---|
| `n2-standard-2` | `2000` | `8000` | 2 vCPUs, about 8 GiB RAM |
| `n2-standard-4` | `4000` | `16000` | 4 vCPUs, about 16 GiB RAM |
| `n2-standard-8` | `8000` | `32000` | 8 vCPUs, about 32 GiB RAM |

The Batch resource request must fit inside the selected machine type. For example, do not request `4000` CPU milli and `16000` MiB memory on an `n2-standard-2` machine.

### Smaller VM test

```bash
./gcp/submit_batch_experiment.sh \
  "exp1-validation-small-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_multiseed_validation.run \
    --output-root outputs/cloud/exp1-validation-small" \
  "n2-standard-2" \
  "43200" \
  "2000" \
  "8000"
```

### Larger VM test

```bash
./gcp/submit_batch_experiment.sh \
  "exp1-validation-large-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_multiseed_validation.run \
    --output-root outputs/cloud/exp1-validation-large" \
  "n2-standard-8" \
  "43200" \
  "8000" \
  "32000"
```

---

## 12. Choosing a VM size

Use evidence rather than guessing.

Start with:

```text
n2-standard-4, CPU_MILLI=4000, MEMORY_MIB=16000
```

Then compare with:

```text
n2-standard-2, CPU_MILLI=2000, MEMORY_MIB=8000
n2-standard-8, CPU_MILLI=8000, MEMORY_MIB=32000
```

A VM may be too large if:

- memory usage is far below the allocation;
- CPU utilisation is consistently low;
- doubling vCPUs does not materially reduce runtime;
- the job is bottlenecked by Python/OpenSpiel traversal rather than CPU capacity.

A VM may be too small if:

- the job fails with `Killed`, `Out of memory`, or exit code `137`;
- the job hits `maxRunDuration`;
- logs show `No space left on device`;
- the job is much slower than expected.

---

## 13. Runtime limits

`MAX_RUN_SECONDS` is a safety cap. For example:

| Seconds | Duration |
|---:|---:|
| `3600` | 1 hour |
| `21600` | 6 hours |
| `43200` | 12 hours |
| `86400` | 24 hours |

If a job exceeds `MAX_RUN_SECONDS`, Batch stops the task and marks the job as failed. If the task is stopped before the final Python upload step, outputs that exist only on the VM may be lost.

For full experiments, use a generous cap such as 12 or 24 hours unless you are deliberately testing runtime behaviour.

---

## 14. Running other experiments

Use the same submission pattern but change the Python module and `--output-root`.

Template:

```bash
./gcp/submit_batch_experiment.sh \
  "JOB_PREFIX-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.EXPERIMENT_MODULE.run \
    --output-root outputs/cloud/JOB_PREFIX" \
  "n2-standard-4" \
  "43200" \
  "4000" \
  "16000"
```

Example:

```bash
./gcp/submit_batch_experiment.sh \
  "exp6-warm-start-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_warm_start_fair_ablation.run \
    --output-root outputs/cloud/exp6-warm-start" \
  "n2-standard-4" \
  "43200" \
  "4000" \
  "16000"
```

---

## 15. Cleaning up

Batch-created VMs are temporary and should terminate when the job completes or fails. Failed job records can be deleted from Batch if desired:

```bash
gcloud batch jobs delete JOB_NAME --location "$REGION" --quiet
```

Do not delete the Cloud Storage bucket unless you are sure you no longer need any outputs.

To inspect bucket contents:

```bash
gcloud storage ls -r "$BUCKET/"
```

To delete a specific job output folder:

```bash
gcloud storage rm -r "$BUCKET/JOB_NAME/"
```

---

## 16. Notes on dependency installation

The script deliberately creates a virtual environment under `/tmp/leduc-venv` rather than installing packages into the system Python. This is important because installing experiment dependencies into the system Python can interfere with the Google Cloud CLI on the Batch VM.

The script uses:

```bash
python3 -m venv --copies /tmp/leduc-venv
source /tmp/leduc-venv/bin/activate
```

and copies outputs to Cloud Storage from inside that same virtual environment:

```bash
python -m pip install --no-cache-dir "google-cloud-storage>=2.16,<4.0"
python scripts/upload_outputs_to_gcs.py outputs "$BUCKET_PATH"
deactivate
```

This avoids relying on the system Google Cloud SDK Python runtime after the
experiment has installed its own dependencies. The uploaded object layout is
unchanged: local `outputs/...` files are copied under
`$BUCKET_PATH/outputs/...`.
