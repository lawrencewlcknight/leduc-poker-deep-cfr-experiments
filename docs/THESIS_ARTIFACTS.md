# Promoting thesis artifacts

The full `outputs/` tree from an experiment is working data. It can include
logs, curve arrays, checkpoints, snapshots, and other files that are useful for
debugging or re-analysis but should not usually be committed to the repository.

For thesis writing, keep only the lightweight artifacts you are likely to cite
or include directly:

- graph images (`*.png`);
- result tables (`*.csv`);
- aggregate JSON summaries;
- `experiment_metadata.json` for provenance.

The repository stores these curated files under:

```text
thesis_artifacts/<experiment_name>/<run_directory_name>/
```

This keeps complete cloud output downloads separate from the smaller set of
tracked artifacts used by the thesis.

## Recommended workflow

After a Google Batch job has succeeded, download the full output tree into a
scratch directory:

```bash
mkdir -p cloud_outputs/JOB_NAME
gcloud storage cp -r "$BUCKET/JOB_NAME/*" "cloud_outputs/JOB_NAME/"
```

Then promote the lightweight thesis artifacts:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME
```

The script searches below `cloud_outputs/JOB_NAME` for experiment run
directories, copies the selected files into `thesis_artifacts/`, and writes a
`promotion_manifest.json` alongside the promoted files.

To preview what would be copied:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME --dry-run
```

To re-promote a run and replace existing promoted artifacts:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME --overwrite
```

You can also point the script at a specific run directory:

```bash
python scripts/promote_thesis_artifacts.py \
  cloud_outputs/JOB_NAME/outputs/cloud/exp1-validation/RUN_DIRECTORY
```

## Included files

By default the promotion script copies:

- `*.png`;
- `*.csv`;
- `aggregate_summary.json`;
- `paired_difference_summary.json`;
- `paired_aggregate_summary.json`;
- `best_checkpoint_summary.json`;
- `experiment_metadata.json`.

It deliberately excludes heavy or scratch files:

- model checkpoints and snapshots (`*.pt`, `*.pth`, `checkpoints/`, `snapshots/`);
- NumPy curve archives (`*.npz`);
- logs (`*.log`);
- failed-run traceback files;
- random-search trace JSON files under `traces/`.

If a particular experiment produces an extra lightweight artifact you want to
track, include it by name or glob:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME \
  --include "extra_table.json,appendix_*.png"
```

If you want to leave out an otherwise selected file:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME \
  --exclude "checkpoint_curves.csv"
```

## Why this is not automatic inside Batch

The Batch VM clones this repository to run the experiment, then uploads outputs
to Cloud Storage. Having that VM push files back into git would require storing
write credentials in the cloud job environment and deciding automatically which
research outputs are worth committing. That is fragile and risky.

The safer pattern is:

1. Batch runs the experiment and uploads everything to Cloud Storage.
2. You download the completed job outputs locally.
3. The promotion script copies only the lightweight thesis-facing files into
   the tracked `thesis_artifacts/` tree.
4. You review the promoted files and commit the ones you want to keep.
