"""Audit revision-v2 execution markers and result-table invariants.

The audit never trains or evaluates a model.  It reports completed and missing
stages, validates per-job row counts and condition uniqueness, and verifies
that target evaluation markers reference the frozen source-policy markers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from run_revision_v2 import (
    DEFAULT_CHECKPOINT_DIR,
    EXPECTED_CONDITION_ROWS,
    EXPECTED_SELECTIVE_ROWS,
    atomic_json,
    direction_dir,
    job_dir,
    jobs_for_direction,
    load_json,
    load_protocol_bundle,
    resolved_artifacts,
    sha256_file,
    source_policy_complete,
    utc_now,
)

KEY_COLUMNS = ["severity", "missing_groups", "replacement"]


def inspect_table(path: Path, expected_rows: int) -> tuple[list[str], dict]:
    problems = []
    details = {"exists": path.exists(), "expected_rows": expected_rows}
    if not path.exists():
        return problems, details
    frame = pd.read_csv(path)
    details.update(
        {
            "rows": len(frame),
            "sha256": sha256_file(path),
            "columns": list(frame.columns),
        }
    )
    if len(frame) != expected_rows:
        problems.append(f"{path}:expected {expected_rows} rows; found {len(frame)}")
    missing_columns = [name for name in KEY_COLUMNS if name not in frame.columns]
    if missing_columns:
        problems.append(f"{path}:missing key columns {missing_columns}")
    else:
        uniqueness_columns = list(KEY_COLUMNS)
        if expected_rows == EXPECTED_SELECTIVE_ROWS:
            if "desired_policy_coverage" not in frame.columns:
                problems.append(f"{path}:missing desired_policy_coverage")
                uniqueness_columns = []
            else:
                uniqueness_columns.append("desired_policy_coverage")
        duplicate_count = (
            int(frame.duplicated(uniqueness_columns).sum()) if uniqueness_columns else 0
        )
        details["duplicate_condition_keys"] = duplicate_count
        if duplicate_count:
            problems.append(f"{path}:{duplicate_count} duplicate condition keys")
    return problems, details


def audit_job(checkpoint_dir: Path, job: dict) -> tuple[list[str], dict]:
    problems = []
    output_dir = job_dir(checkpoint_dir, job)
    policy_ok, policy_reason = source_policy_complete(checkpoint_dir, job)
    row = {
        "job_id": job["job_id"],
        "direction": job["direction"],
        "variant": job["variant"],
        "partition_seed": job["partition_seed"],
        "model_seed": job["model_seed"],
        "source_policy_frozen": policy_ok,
        "source_policy_note": policy_reason,
    }
    if policy_ok:
        try:
            artifacts = resolved_artifacts(checkpoint_dir, job)
            row["artifact_hashes"] = {
                name: sha256_file(path) for name, path in artifacts.items()
            }
        except (FileNotFoundError, KeyError, RuntimeError) as error:
            problems.append(f"{job['job_id']}:artifact resolution:{error}")

    for role in ["source", "target"]:
        prefix = role.upper()
        marker_path = output_dir / f"{prefix}_EVALUATION_COMPLETE.json"
        nonselective_path = output_dir / f"{role}_nonselective.csv"
        selective_path = output_dir / f"{role}_selective.csv"
        ns_problems, ns_details = inspect_table(
            nonselective_path, EXPECTED_CONDITION_ROWS
        )
        s_problems, s_details = inspect_table(selective_path, EXPECTED_SELECTIVE_ROWS)
        problems.extend(ns_problems + s_problems)
        marker_exists = marker_path.exists()
        role_details = {
            "marker_exists": marker_exists,
            "nonselective": ns_details,
            "selective": s_details,
        }
        if marker_exists:
            marker = load_json(marker_path)
            source_policy_marker = output_dir / "SOURCE_POLICY_FROZEN.json"
            current_hash = (
                sha256_file(source_policy_marker)
                if source_policy_marker.exists()
                else None
            )
            recorded_hash = marker.get("source_policy_marker_sha256")
            role_details["source_policy_reference_pass"] = current_hash == recorded_hash
            if current_hash != recorded_hash:
                problems.append(
                    f"{job['job_id']}:{role}:source-policy marker hash mismatch"
                )
            if not nonselective_path.exists() or not selective_path.exists():
                problems.append(f"{job['job_id']}:{role}:marker without tables")
        elif nonselective_path.exists() or selective_path.exists():
            problems.append(f"{job['job_id']}:{role}:tables without completion marker")
        row[f"{role}_evaluation"] = role_details
    return problems, row


def stage_summary(rows: list[dict], direction_root: Path) -> dict:
    return {
        "jobs": len(rows),
        "source_policies_frozen": sum(row["source_policy_frozen"] for row in rows),
        "source_evaluations_complete": sum(
            row["source_evaluation"]["marker_exists"] for row in rows
        ),
        "target_evaluations_complete": sum(
            row["target_evaluation"]["marker_exists"] for row in rows
        ),
        "all_source_policies_marker": (
            direction_root / "ALL_SOURCE_POLICIES_FROZEN.json"
        ).exists(),
        "all_source_evaluations_marker": (
            direction_root / "ALL_SOURCE_EVALUATIONS_COMPLETE.json"
        ).exists(),
        "all_target_evaluations_marker": (
            direction_root / "ALL_TARGET_EVALUATIONS_COMPLETE.json"
        ).exists(),
    }


def audit(checkpoint_dir: Path, requested_direction: str) -> dict:
    protocol, protocol_audit, _, jobs = load_protocol_bundle()
    directions = (
        ["forward", "reverse"]
        if requested_direction == "both"
        else [requested_direction]
    )
    all_problems = []
    direction_reports = {}
    for direction in directions:
        rows = []
        direction_problems = []
        for job in jobs_for_direction(jobs, direction):
            problems, row = audit_job(checkpoint_dir, job)
            direction_problems.extend(problems)
            rows.append(row)
        root = direction_dir(checkpoint_dir, direction)
        summary = stage_summary(rows, root)
        if direction == "forward":
            gate_path = root / "FORWARD_SOURCE_ONLY_CONTINUATION_GATE.json"
            summary["forward_gate_exists"] = gate_path.exists()
            summary["forward_gate_status"] = (
                load_json(gate_path).get("status") if gate_path.exists() else None
            )
        direction_reports[direction] = {
            "summary": summary,
            "problems": direction_problems,
            "jobs": rows,
        }
        all_problems.extend(direction_problems)

    any_execution = any(
        report["summary"]["source_policies_frozen"]
        or report["summary"]["source_evaluations_complete"]
        or report["summary"]["target_evaluations_complete"]
        for report in direction_reports.values()
    )
    all_complete = all(
        report["summary"]["target_evaluations_complete"] == report["summary"]["jobs"]
        for report in direction_reports.values()
    )
    status = (
        "FAIL"
        if all_problems
        else "PASS"
        if all_complete
        else "IN_PROGRESS"
        if any_execution
        else "NOT_STARTED"
    )
    return {
        "status": status,
        "created_utc": utc_now(),
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_audit["protocol_sha256"],
        "requested_direction": requested_direction,
        "directions": direction_reports,
        "problems": all_problems,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--direction", choices=["forward", "reverse", "both"], default="both"
    )
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.checkpoint_dir, args.direction)
    if args.write_report:
        destination = (
            args.checkpoint_dir / "revision_v2" / ("REVISION_V2_EXECUTION_AUDIT.json")
        )
        atomic_json(report, destination)
        print("Audit written to", destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
