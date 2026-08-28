# Revision-v2 GitHub update checklist

## Branch boundary

1. Work only on branch `revision-v2-comparative`.
2. Confirm the branch is based on `main` and is ahead of `main` by the intended
   revision-v2 commits.
3. Do not merge the branch into `main` and do not move, delete or recreate tag
   `v1.0.0` while the comparative experiments are incomplete.
4. Do not commit raw datasets, Parquet checkpoints, fitted models, credentials
   or per-flow prediction arrays.

## Upload through the GitHub website

1. Select branch `revision-v2-comparative` before uploading.
2. Extract the supplied update ZIP locally.
3. Upload the files at the repository root while preserving the `protocols/`,
   `scripts/` and `tests/` paths.
4. Replace files with the same paths when GitHub asks; do not create a second
   nested repository directory.
5. Use the commit message:

   ```text
   Add restartable revision-v2 comparative runner
   ```

6. Select **Commit directly to the revision-v2-comparative branch**.
7. Do not select **Create a new branch** and do not open or merge a pull request.

## Validation before training

Run from the repository root in the pinned environment:

```bash
python scripts/validate_revision_v2_protocol.py --check
python -m unittest discover -s tests -v
python scripts/run_revision_v2.py plan --direction forward
python scripts/run_revision_v2.py train \
  --direction forward \
  --job-id forward_matched_complete_baseline_seed_2027 \
  --dry-run
python scripts/verify_repository.py --require-figures --require-tables
```

All validation commands must pass before any revision-v2 training begins.
