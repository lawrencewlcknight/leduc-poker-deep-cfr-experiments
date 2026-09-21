#!/usr/bin/env python3
"""Build Google Cloud Batch jobs for Deep CFR Experiment 30."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


REPO_URL = "https://github.com/lawrencewlcknight/leduc-poker-deep-cfr-experiments.git"
MODULE = "experiments.leduc_poker.single_deep_cfr_paper_aligned_36h_trajectory.cloud"
TASK_COUNT = 5


def _q(value) -> str:
    return shlex.quote(str(value))


def _bootstrap(args, kind: str) -> str:
    return f"""
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/cache
export PIP_CACHE_DIR=/tmp/pip-cache

REPO_URL={_q(args.repo_url)}
REPO_REF={_q(args.repo_ref)}
BUCKET_ROOT={_q(args.bucket_root.rstrip('/'))}
RUN_ID={_q(args.run_id)}
RETRY_ATTEMPT="${{BATCH_TASK_RETRY_ATTEMPT:-0}}"
TASK_SLOT="${{BATCH_TASK_INDEX:-0}}"
WORK_ROOT="/workspace/exp30-{kind}-$TASK_SLOT-$RETRY_ATTEMPT"
REPOSITORY="$WORK_ROOT/repository"
OUTPUT_ROOT="$WORK_ROOT/output"
VENV_ROOT="/tmp/exp30-{kind}-venv-$TASK_SLOT-$RETRY_ATTEMPT"

if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git ca-certificates python3-pip python3-dev python3-venv
mkdir -p "$WORK_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
git clone --filter=blob:none "$REPO_URL" "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach "$REPO_REF"
# Google Batch images may place a launcher named ``python3`` earlier on PATH
# without a usable sys._base_executable. Use Debian's installed interpreter
# explicitly so venv can always locate the executable it must copy.
unset PYTHONHOME PYTHONPATH __PYVENV_LAUNCHER__
SYSTEM_PYTHON=/usr/bin/python3
"$SYSTEM_PYTHON" -m venv --copies "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.0,<3.0" "torchvision>=0.15,<1.0"
python -m pip install --no-cache-dir --no-build-isolation -r "$REPOSITORY/requirements.txt"
python -m pip install --no-cache-dir --no-build-isolation -e "$REPOSITORY"
python -m pip check
cd "$REPOSITORY"
""".strip()


def _task_metadata() -> str:
    return f"""
TASK_INDEX="${{BATCH_TASK_INDEX:?Google Batch did not set BATCH_TASK_INDEX}}"
TASK_METADATA="$(python - "$TASK_INDEX" <<'PY' | sed -n 's/^EXP30_TASK_METADATA //p' | tail -n 1
import importlib
import sys
module = importlib.import_module("{MODULE}")
index = int(sys.argv[1])
seeds = tuple(module.PRODUCTION_SEEDS)
print("EXP30_TASK_METADATA", int(seeds[index]), module.task_name(index, seeds))
PY
)"
read -r SOURCE_SEED TASK_NAME EXTRA_METADATA <<< "$TASK_METADATA"
EXPECTED_TASK_NAME="task_$(printf '%03d' "$TASK_INDEX")_seed_$SOURCE_SEED"
if [[ ! "$SOURCE_SEED" =~ ^[0-9]+$ || "$TASK_NAME" != "$EXPECTED_TASK_NAME" || -n "$EXTRA_METADATA" ]]; then
  echo "Invalid Experiment 30 task metadata: $TASK_METADATA" >&2
  exit 2
fi
""".strip()


def _controller_script(args) -> str:
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
REPO_URL={_q(args.repo_url)}
REPO_REF={_q(args.repo_ref)}
CONTROLLER_ACTION={_q(args.controller_action)}
WORK_ROOT="/workspace/exp30-controller-${{BATCH_TASK_RETRY_ATTEMPT:-0}}"
REPOSITORY="$WORK_ROOT/repository"

if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git ca-certificates python3
mkdir -p "$WORK_ROOT"
git clone --filter=blob:none "$REPO_URL" "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach "$REPO_REF"
cd "$REPOSITORY"

export PROJECT_ID={_q(args.project_id)}
export REGION={_q(args.region)}
export BUCKET={_q(args.bucket_root.rstrip('/'))}
export SA_EMAIL={_q(args.service_account)}
export REPO_REF={_q(args.repo_ref)}
export RUN_ID={_q(args.run_id)}
export PARALLELISM={_q(args.parallelism)}
export EXP30_REMOTE_CONTROLLER=1

gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --limit 1 \
  --format='value(name)' >/dev/null
exec bash gcp/run_single_deep_cfr_paper_aligned_36h_trajectory.sh "$CONTROLLER_ACTION"
"""


def _script(args) -> str:
    if args.kind == "controller":
        return _controller_script(args)
    bootstrap = _bootstrap(args, args.kind)
    if args.kind == "smoke":
        action = f"""
python -m {MODULE} smoke --output-root "$OUTPUT_ROOT" --no-resume
gcloud storage rsync --recursive "$OUTPUT_ROOT" "$BUCKET_ROOT/$RUN_ID/smoke"
""".strip()
    elif args.kind == "train":
        action = f"""
{_task_metadata()}
REMOTE_TASK="$BUCKET_ROOT/$RUN_ID/training/$TASK_NAME"
mkdir -p "$OUTPUT_ROOT/training/$TASK_NAME"
if gcloud storage ls "$REMOTE_TASK/training_worker_result.json" >/dev/null 2>&1; then
  echo "Seed $SOURCE_SEED is already complete; no retraining is required."
  exit 0
fi
upload_worker() {{
  gcloud storage rsync --recursive "$OUTPUT_ROOT/training/$TASK_NAME" "$REMOTE_TASK" || true
}}
trap upload_worker EXIT
python -m {MODULE} train --task-index "$TASK_INDEX" --output-root "$OUTPUT_ROOT"
""".strip()
    elif args.kind == "evaluate":
        action = f"""
{_task_metadata()}
REMOTE_TRAINING="$BUCKET_ROOT/$RUN_ID/training/$TASK_NAME"
REMOTE_EVALUATION="$BUCKET_ROOT/$RUN_ID/evaluation/$TASK_NAME"
mkdir -p "$OUTPUT_ROOT/training/$TASK_NAME" "$OUTPUT_ROOT/evaluation/$TASK_NAME"
gcloud storage rsync --recursive "$REMOTE_TRAINING" "$OUTPUT_ROOT/training/$TASK_NAME"
if gcloud storage ls "$REMOTE_EVALUATION/evaluation_worker_result.json" >/dev/null 2>&1; then
  gcloud storage rsync --recursive "$REMOTE_EVALUATION" "$OUTPUT_ROOT/evaluation/$TASK_NAME"
fi
upload_worker() {{
  gcloud storage rsync --recursive "$OUTPUT_ROOT/evaluation/$TASK_NAME" "$REMOTE_EVALUATION" || true
}}
trap upload_worker EXIT
python -m {MODULE} evaluate --task-index "$TASK_INDEX" \
  --training-root "$OUTPUT_ROOT" --output-root "$OUTPUT_ROOT"
""".strip()
    elif args.kind == "aggregate":
        action = f"""
mkdir -p "$OUTPUT_ROOT/evaluation"
gcloud storage rsync --recursive "$BUCKET_ROOT/$RUN_ID/evaluation" "$OUTPUT_ROOT/evaluation"
python -m {MODULE} aggregate --evaluation-root "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/analysis"
gcloud storage rsync --recursive "$OUTPUT_ROOT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
""".strip()
    else:  # pragma: no cover
        raise ValueError(args.kind)
    return f"#!/usr/bin/env bash\nset -Eeuo pipefail\n{bootstrap}\n{action}\n"


def build_job(args) -> dict:
    task_count = TASK_COUNT if args.kind in {"train", "evaluate"} else 1
    parallelism = min(task_count, int(args.parallelism))
    if args.kind == "train":
        duration, cpu, memory, machine, disk, retries = (
            "172800s", 8000, 30000, "n2-standard-8", 200, 0
        )
    elif args.kind == "evaluate":
        duration, cpu, memory, machine, disk, retries = (
            "72000s", 8000, 30000, "n2-standard-8", 200, 0
        )
    elif args.kind == "smoke":
        duration, cpu, memory, machine, disk, retries = (
            "7200s", 8000, 30000, "n2-standard-8", 100, 0
        )
    elif args.kind == "controller":
        duration, cpu, memory, machine, disk, retries = (
            "604800s", 1000, 1500, "e2-small", 30, 2
        )
    else:
        duration, cpu, memory, machine, disk, retries = (
            "7200s", 4000, 15000, "n2-standard-4", 100, 0
        )
    return {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [{"script": {"text": _script(args)}}],
                    "computeResource": {"cpuMilli": cpu, "memoryMib": memory},
                    "maxRetryCount": retries,
                    "maxRunDuration": duration,
                },
                "taskCount": task_count,
                "parallelism": parallelism,
                "taskCountPerNode": 1,
            }
        ],
        "allocationPolicy": {
            "serviceAccount": {"email": args.service_account},
            "instances": [
                {
                    "policy": {
                        "machineType": machine,
                        "provisioningModel": "STANDARD",
                        "bootDisk": {"sizeGb": disk, "type": "pd-balanced"},
                    }
                }
            ],
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"experiment": "deep-cfr-exp30-paper-sdcfr", "stage": args.kind},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=("controller", "smoke", "train", "evaluate", "aggregate"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket-root", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--repo-ref", required=True)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--parallelism", type=int, default=TASK_COUNT)
    parser.add_argument("--project-id", default="")
    parser.add_argument("--region", default="")
    parser.add_argument(
        "--controller-action", choices=("orchestrate", "orchestrate-resume"), default="orchestrate"
    )
    args = parser.parse_args()
    if args.parallelism < 1 or args.parallelism > TASK_COUNT:
        parser.error(f"--parallelism must be between 1 and {TASK_COUNT}")
    if args.kind == "controller" and (not args.project_id or not args.region):
        parser.error("controller jobs require --project-id and --region")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(build_job(args), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
