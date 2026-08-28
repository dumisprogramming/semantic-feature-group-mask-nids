# Revision-v2 frozen comparative protocol

This directory defines the pre-training protocol created after the JISA desk
rejection. It does not contain revision-v2 results and does not alter the
submitted `v1.0.0` release.

## Scientific purpose

The submitted experiment showed that semantic singleton-mask augmentation can
improve average unseen pairwise-loss performance. It did not establish that
semantic grouping itself caused the benefit, because it lacked matched
independent-feature and random-group controls. Revision v2 addresses that gap.

The primary comparison is:

1. matched complete-only training;
2. independent feature dropout with the same mask-size distribution;
3. random non-semantic group masking with the same group sizes;
4. expert semantic singleton masking.

Exhaustive singleton training is a budget sensitivity analysis. Pairwise-seen
training is an explicitly labelled upper bound and cannot support an unseen-loss
claim.

## Files

- `revision_v2_protocol_frozen.json`: authoritative scientific and execution
  specification.
- `random_partitions_frozen.json`: five deterministic random partitions created
  from the frozen predictor order.
- `revision_v2_job_matrix.csv`: every precommitted forward and conditional
  reverse training job.
- `revision_v2_protocol_audit.json`: deterministic protocol and matrix audit.
- `scripts/run_revision_v2.py`: stage-gated training and evaluation runner.
- `scripts/prepare_revision_v2_reverse.py`: conditional reverse-direction
  preparation and artifact freezing.
- `scripts/audit_revision_v2_run.py`: read-only execution and result audit.

The three derived files are created and then verified by:

```bash
python scripts/validate_revision_v2_protocol.py --materialize
python scripts/validate_revision_v2_protocol.py --check
```

## Mandatory execution order

1. Create the Git branch `revision-v2-comparative` from the recorded main
   commit. Do not move or recreate tag `v1.0.0`.
2. Verify every forward v1 artifact against
   `protocols/replication_protocol_frozen.json`.
3. Dry-run the matched comparative runner and inspect its 50 forward jobs.
4. Freeze all forward source-only models, calibrators and policies.
5. Evaluate the forward source test and apply the source-only continuation
   gate.
6. Open the forward sealed target only after the source evaluation and gate are
   frozen.
7. Run the precommitted reverse direction only if the source-only gate passes.
8. Audit all job markers and aggregate outputs before changing the manuscript.

## Runner commands

Install the pinned environment and verify the frozen specification:

```bash
python -m pip install -r requirements.txt
python scripts/validate_revision_v2_protocol.py --check
python scripts/run_revision_v2.py plan --direction forward
python -m unittest discover -s tests -v
```

Dry-run one precommitted job without loading experiment data:

```bash
python scripts/run_revision_v2.py train \
  --direction forward \
  --job-id forward_matched_complete_baseline_seed_2027 \
  --dry-run
```

Execute one forward job after setting `NIDS_CHECKPOINT_DIR`:

```bash
python scripts/run_revision_v2.py train \
  --direction forward \
  --job-id forward_matched_complete_baseline_seed_2027
```

Repeat with each job ID returned by `plan`. The explicit `--all` flag is
available for an uninterrupted environment, but is intentionally never the
default. After all forward jobs finish, run the locked stages in order:

```bash
python scripts/run_revision_v2.py freeze-policies --direction forward
python scripts/run_revision_v2.py evaluate-source --direction forward --all
python scripts/run_revision_v2.py gate --direction forward
python scripts/run_revision_v2.py evaluate-target --direction forward --all
python scripts/audit_revision_v2_run.py --direction forward --write-report
```

Reverse preparation and execution are blocked unless the forward gate is
`PASS`. If it passes, prepare the reverse artifacts before reverse training:

```bash
python scripts/prepare_revision_v2_reverse.py \
  --cicids2017-dir /path/to/CICIDS2017 \
  --cse-cic-ids2018-dir /path/to/CSE-CIC-IDS2018
python scripts/run_revision_v2.py plan --direction reverse
```

All outputs are written under `NIDS_CHECKPOINT_DIR/revision_v2`; v1 model,
result and release paths are read-only.

## Interpretation rule

If semantic masking does not outperform both matched independent dropout and
the average of all five random partitions under the source-only gate, the
revision must not claim that expert semantic grouping is the causal advantage.
The work should instead be framed as a robustness benchmark and transfer-failure
study.
