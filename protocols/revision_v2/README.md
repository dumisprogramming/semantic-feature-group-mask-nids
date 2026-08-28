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
3. Implement and dry-run the new matched comparative runner.
4. Freeze all forward source-only models, calibrators and policies.
5. Evaluate the forward source test and apply the source-only continuation
   gate.
6. Open the forward sealed target only after step 4 is complete.
7. Run the precommitted reverse direction only if the source-only gate passes.
8. Audit all job markers and aggregate outputs before changing the manuscript.

## Interpretation rule

If semantic masking does not outperform both matched independent dropout and
the average of all five random partitions under the source-only gate, the
revision must not claim that expert semantic grouping is the causal advantage.
The work should instead be framed as a robustness benchmark and transfer-failure
study.
