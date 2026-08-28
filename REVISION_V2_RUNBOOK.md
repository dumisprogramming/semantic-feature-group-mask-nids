# Revision-v2 execution runbook

This runbook executes the frozen comparative protocol without altering the
submitted `v1.0.0` record. All generated artifacts are written below
`NIDS_CHECKPOINT_DIR/revision_v2`.

## 1. Prepare the environment

Check out branch `revision-v2-comparative`, install the pinned dependencies and
choose a persistent checkpoint directory. In Colab, place the checkpoint
directory on mounted Google Drive so completed jobs survive a runtime reset.

```bash
python -m pip install -r requirements.txt
export NIDS_CHECKPOINT_DIR=/path/to/NIDS_Research_Checkpoints
python scripts/validate_revision_v2_protocol.py --check
python -m unittest discover -s tests -v
```

Do not put raw data or checkpoint outputs inside the Git repository.

## 2. Verify the job plan

```bash
python scripts/run_revision_v2.py plan --direction forward
```

The forward plan must report 50 jobs: 45 newly trained jobs and five verified
reuses of the corrected v1 semantic-augmentation models. Planning and dry-run
commands do not train a model.

```bash
python scripts/run_revision_v2.py train \
  --direction forward \
  --job-id forward_matched_complete_baseline_seed_2027 \
  --dry-run
```

## 3. Run forward training restartably

Use one explicit job ID from `plan` at a time:

```bash
python scripts/run_revision_v2.py train \
  --direction forward \
  --job-id JOB_ID
```

Rerunning a completed job verifies and reuses its completion marker. The
runner never silently selects all jobs; `--all` must be explicit.

After every forward job is complete, freeze policies and evaluate the source
test before opening the sealed target:

```bash
python scripts/run_revision_v2.py freeze-policies --direction forward
python scripts/run_revision_v2.py evaluate-source --direction forward --all
python scripts/run_revision_v2.py gate --direction forward
```

Inspect the frozen continuation-gate report. Do not prepare or train the
reverse direction unless the report says `PASS`.

## 4. Evaluate the forward target

After the source evaluation and gate are frozen:

```bash
python scripts/run_revision_v2.py evaluate-target --direction forward --all
python scripts/audit_revision_v2_run.py --direction forward --write-report
```

Each completed job must produce 43 non-selective condition rows and 129
selective-policy rows for each evaluated dataset.

## 5. Conditional reverse direction

Only after a forward gate `PASS`, create the reverse source and sealed-target
artifacts:

```bash
python scripts/prepare_revision_v2_reverse.py \
  --cicids2017-dir /path/to/CICIDS2017 \
  --cse-cic-ids2018-dir /path/to/CSE-CIC-IDS2018
python scripts/run_revision_v2.py plan --direction reverse
```

Run the 40 reverse jobs by explicit job ID, then execute the same locked stages:

```bash
python scripts/run_revision_v2.py train --direction reverse --job-id JOB_ID
python scripts/run_revision_v2.py freeze-policies --direction reverse
python scripts/run_revision_v2.py evaluate-source --direction reverse --all
python scripts/run_revision_v2.py evaluate-target --direction reverse --all
python scripts/audit_revision_v2_run.py --direction reverse --write-report
```

## 6. Interpretation boundary

Do not claim a semantic-group advantage unless the frozen forward source-only
gate passes. A gate failure is a valid result and must be reported without
using target performance to override the decision.
