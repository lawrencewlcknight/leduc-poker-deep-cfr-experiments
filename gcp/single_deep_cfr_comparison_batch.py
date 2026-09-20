#!/usr/bin/env python3
"""Build Google Cloud Batch jobs for Experiment 28's resilient parallel run."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


REPO_URL = "https://github.com/lawrencewlcknight/leduc-poker-deep-cfr-experiments.git"


def _q(value) -> str:
    return shlex.quote(str(value))


def _bootstrap(args) -> str:
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
WORK_ROOT=/workspace/single-deep-cfr-comparison
REPO="$WORK_ROOT/repository"
OUTPUT_ROOT="$WORK_ROOT/output"

if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git python3-pip python3-dev python3-venv
mkdir -p "$WORK_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
git clone --filter=blob:none "$REPO_URL" "$REPO"
git -C "$REPO" checkout --detach "$REPO_REF"
cd "$REPO"
echo "Resolved repository commit: $(git rev-parse HEAD)"

python3 -m venv --copies /tmp/leduc-venv
source /tmp/leduc-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.0,<3.0" "torchvision>=0.15,<1.0"
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-cache-dir --no-build-isolation -e .
python -m pip check
python - <<'PY'
import numpy as np
from deep_cfr_poker.experiment_utils import normalised_auc
print("NumPy:", np.__version__)
print("AUC compatibility check:", normalised_auc([0.0, 1.0], [1.0, 2.0]))
PY
""".strip()


def _controller_script(args) -> str:
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
REPO_URL={_q(args.repo_url)}
REPO_REF={_q(args.repo_ref)}
CONTROLLER_ACTION={_q(args.controller_action)}
WORK_ROOT="/workspace/exp28-controller-${{BATCH_TASK_RETRY_ATTEMPT:-0}}"
REPO="$WORK_ROOT/repository"

if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git ca-certificates python3
mkdir -p "$WORK_ROOT"
git clone --filter=blob:none "$REPO_URL" "$REPO"
git -C "$REPO" checkout --detach "$REPO_REF"
cd "$REPO"

export PROJECT_ID={_q(args.project_id)}
export REGION={_q(args.region)}
export BUCKET={_q(args.bucket_root.rstrip('/'))}
export SA_EMAIL={_q(args.service_account)}
export REPO_REF={_q(args.repo_ref)}
export RUN_ID={_q(args.run_id)}
export PARALLELISM={_q(args.parallelism)}
export PROVISIONING_MODEL={_q(args.provisioning_model)}
export MAX_RETRIES={_q(args.max_retries)}
export EXP28_REMOTE_CONTROLLER=1

exec bash gcp/run_single_deep_cfr_comparison.sh "$CONTROLLER_ACTION"
"""


def _script(args) -> str:
    if args.kind == "controller":
        return _controller_script(args)

    bootstrap = _bootstrap(args)
    if args.kind == "smoke":
        action = """
SMOKE_OUTPUT="$OUTPUT_ROOT/smoke"
python -m experiments.leduc_poker.single_deep_cfr_comparison.run \
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
  --run-dir "$SMOKE_OUTPUT"
test -s "$SMOKE_OUTPUT/seed_results/seed_1234_result.json"
test -s "$SMOKE_OUTPUT/checkpoint_curves.csv"
gcloud storage rsync --recursive "$SMOKE_OUTPUT" "$BUCKET_ROOT/$RUN_ID/smoke"
""".strip()
    elif args.kind == "train":
        action = """
TASK_INDEX="${BATCH_TASK_INDEX:?Google Batch did not set BATCH_TASK_INDEX}"
SEED="$(python - "$TASK_INDEX" <<'PY'
import sys
from experiments.leduc_poker.single_deep_cfr_comparison.config import DEFAULT_SEEDS
print(DEFAULT_SEEDS[int(sys.argv[1])])
PY
)"
TASK_NAME="$(printf 'task_%03d_seed_%s' "$TASK_INDEX" "$SEED")"
LOCAL_TASK="$OUTPUT_ROOT/workers/$TASK_NAME"
REMOTE_TASK="$BUCKET_ROOT/$RUN_ID/workers/$TASK_NAME"
mkdir -p "$LOCAL_TASK"

if gcloud storage ls "$REMOTE_TASK/SUCCESS.json" >/dev/null 2>&1; then
  echo "Seed $SEED is already complete; no retraining is required."
  exit 0
fi

upload_worker() {
  if [[ -d "$LOCAL_TASK" ]]; then
    gcloud storage rsync --recursive "$LOCAL_TASK" "$REMOTE_TASK" || true
  fi
}
trap upload_worker EXIT

python -m experiments.leduc_poker.single_deep_cfr_comparison.run \
  --seeds "$SEED" \
  --run-dir "$LOCAL_TASK"
test -s "$LOCAL_TASK/seed_results/seed_${SEED}_result.json"
python - "$LOCAL_TASK/SUCCESS.json" "$SEED" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"status": "complete", "seed": int(sys.argv[2])}, indent=2))
PY
""".strip()
    elif args.kind == "aggregate":
        action = """
mkdir -p "$OUTPUT_ROOT/workers" "$OUTPUT_ROOT/analysis"
gcloud storage rsync --recursive \
  "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT_ROOT/workers"
python -m experiments.leduc_poker.single_deep_cfr_comparison.run \
  --aggregate-workers-root "$OUTPUT_ROOT/workers" \
  --run-dir "$OUTPUT_ROOT/analysis"
gcloud storage rsync --recursive \
  "$OUTPUT_ROOT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
""".strip()
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(args.kind)
    return f"#!/usr/bin/env bash\nset -Eeuo pipefail\n{bootstrap}\n{action}\n"


def build_job(args) -> dict:
    task_count = 5 if args.kind == "train" else 1
    parallelism = min(task_count, args.parallelism)
    if args.kind == "train":
        max_duration = "28800s"
        cpu_milli = 4000
        memory_mib = 15000
        machine_type = "n2-standard-4"
        boot_disk_size = 100
        retries = args.max_retries
    elif args.kind == "controller":
        max_duration = "86400s"
        cpu_milli = 1000
        memory_mib = 1500
        machine_type = "e2-small"
        boot_disk_size = 30
        retries = 2
    elif args.kind == "aggregate":
        max_duration = "7200s"
        cpu_milli = 4000
        memory_mib = 15000
        machine_type = "n2-standard-4"
        boot_disk_size = 100
        retries = 0
    else:
        max_duration = "3600s"
        cpu_milli = 4000
        memory_mib = 15000
        machine_type = "n2-standard-4"
        boot_disk_size = 50
        retries = 0

    return {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [{"script": {"text": _script(args)}}],
                    "computeResource": {
                        "cpuMilli": cpu_milli,
                        "memoryMib": memory_mib,
                    },
                    "maxRetryCount": retries,
                    "maxRunDuration": max_duration,
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
                        "machineType": machine_type,
                        "provisioningModel": (
                            "STANDARD"
                            if args.kind == "controller"
                            else args.provisioning_model
                        ),
                        "bootDisk": {
                            "sizeGb": boot_disk_size,
                            "type": "pd-balanced",
                        },
                    }
                }
            ],
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"experiment": "deep-cfr-exp28-sdcfr", "stage": args.kind},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=("controller", "smoke", "train", "aggregate"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket-root", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--repo-ref", required=True)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--parallelism", type=int, default=5)
    parser.add_argument(
        "--provisioning-model", choices=("STANDARD", "SPOT"), default="STANDARD"
    )
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--project-id", default="")
    parser.add_argument("--region", default="")
    parser.add_argument(
        "--controller-action",
        choices=("orchestrate", "orchestrate-resume"),
        default="orchestrate",
    )
    args = parser.parse_args()
    if args.parallelism < 1:
        parser.error("--parallelism must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(build_job(args), handle, indent=2)


if __name__ == "__main__":
    main()
