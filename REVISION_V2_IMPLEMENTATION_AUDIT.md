# Revision-v2 implementation audit

Audit date: 2026-08-28 UTC  
Status: **PASS — ready for precommitted forward execution**

This audit validates the runner implementation only. It does not contain or
claim any revision-v2 research result, and no research-dataset training was
performed during the audit.

## Verified invariants

- The frozen protocol and derived files pass their deterministic hash audit.
- The forward matrix contains 50 jobs: 45 new jobs and five artifact-verified
  model reuses.
- The conditional reverse matrix contains 40 new jobs.
- Every model uses the same five source-only development conditions: complete
  telemetry and four semantic singleton losses.
- Independent dropout applies an exact per-row feature-loss budget.
- Each evaluated model/dataset combination contains 43 unique non-selective
  conditions and 129 selective-policy rows.
- The source-only continuation gate has tested `PASS` and `FAIL` paths.
- The target evaluation command reports `target_loaded: false` during dry-run
  and is stage-gated behind frozen source evaluation.
- Reverse preparation and execution are blocked unless the forward gate is
  frozen as `PASS`.
- Generated artifacts are confined to
  `NIDS_CHECKPOINT_DIR/revision_v2`; the submitted `v1.0.0` paths are not
  overwritten.

## Executed checks

```text
python -m ruff check ...                                      PASS
python -m ruff format --check ...                             PASS
python -m unittest discover -s tests -v                       PASS (8 tests)
python scripts/validate_revision_v2_protocol.py --check       PASS
python scripts/verify_repository.py --require-figures \
  --require-tables                                            PASS
python scripts/run_revision_v2.py plan --direction forward    PASS
python scripts/run_revision_v2.py train ... --dry-run         DRY_RUN_PASS
python -m compileall -q scripts tests                         PASS
```

The synthetic end-to-end test trained a small 77-feature XGBoost model, saved
and reloaded the model/calibrator/policy bundle, and evaluated all 43 telemetry
conditions. It produced exactly 43 non-selective and 129 selective rows. The
synthetic artifacts were created in an isolated temporary directory and are
not part of this repository.

## Execution boundary

Start with the forward direction and follow `REVISION_V2_RUNBOOK.md`. Do not
start reverse preparation or training after a source-only gate failure. Do not
merge this branch into `main` until the comparative analysis and its audit are
complete.
