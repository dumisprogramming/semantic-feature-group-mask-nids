"""Prepare and freeze the secondary Random Forest replication.

Run in Google Colab with:
    %run "/content/drive/MyDrive/research/researchdata/NIDS_Research/prepare_random_forest_replication.py"

This script does not train or evaluate a model. It verifies the seven frozen
artifacts used by the XGBoost confirmatory study and then creates an immutable
protocol for a five-seed Random Forest sensitivity analysis. Target labels are
never used for training, calibration, threshold selection, or policy selection.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CHECKPOINT_DIR = Path(
    os.environ.get(
        "NIDS_CHECKPOINT_DIR",
        "/content/drive/MyDrive/research/researchdata/"
        "NIDS_Research/NIDS_Research_Checkpoints",
    )
)
SOURCE_PROTOCOL_PATH = (
    CHECKPOINT_DIR
    / "five_seed_confirmatory_replication"
    / "replication_protocol_frozen.json"
)
OUTPUT_DIR = (
    CHECKPOINT_DIR
    / "five_seed_confirmatory_replication"
    / "random_forest_model_family_replication"
)
OUTPUT_PROTOCOL_PATH = OUTPUT_DIR / "random_forest_protocol_frozen.json"

SEEDS = [2027, 2028, 2029, 2030, 2031]
EXPECTED_ROWS = {
    "source_train.parquet": 598043,
    "source_validation.parquet": 152964,
    "source_calibration.parquet": 99084,
    "source_test.parquet": 150178,
    "target_test_sealed.parquet": 999143,
}

# Fixed before observing any Random Forest result. The depth and leaf limits
# constrain memory on standard Colab CPU runtimes while retaining nonlinear
# interactions. The model is a bagged ensemble, distinct from boosted XGBoost.
RF_CONFIGURATION = {
    "estimator": "sklearn.ensemble.RandomForestClassifier",
    "n_estimators": 300,
    "criterion": "gini",
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "bootstrap": True,
    "max_samples": 0.7,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": "confirmatory seed",
}


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError:
        return
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def verify_authoritative_files(source_protocol: dict) -> list[dict]:
    rows = []
    failures = []
    frozen_files = source_protocol["frozen_files"]
    for name, expected in frozen_files.items():
        path = CHECKPOINT_DIR / name
        exists = path.exists()
        actual_size = path.stat().st_size if exists else None
        actual_hash = sha256(path) if exists else None
        size_match = exists and actual_size == int(expected["size_bytes"])
        hash_match = exists and actual_hash == expected["sha256"]
        status = "PASS" if exists and size_match and hash_match else "FAIL"
        row = {
            "file": name,
            "status": status,
            "exists": exists,
            "expected_size": int(expected["size_bytes"]),
            "actual_size": actual_size,
            "size_match": bool(size_match),
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual_hash,
            "sha256_match": bool(hash_match),
        }
        rows.append(row)
        if status != "PASS":
            failures.append(row)
    if failures:
        names = [row["file"] for row in failures]
        raise RuntimeError(f"Frozen-file verification failed: {names}")
    return rows


def inspect_parquet_files(metadata: dict) -> list[dict]:
    predictors = metadata["predictors"]
    reports = []
    for name, expected_rows in EXPECTED_ROWS.items():
        path = CHECKPOINT_DIR / name
        frame = pd.read_parquet(path)
        missing_predictors = sorted(set(predictors) - set(frame.columns))
        additional_columns = sorted(set(frame.columns) - set(predictors))
        report = {
            "file": name,
            "rows": int(len(frame)),
            "expected_rows": expected_rows,
            "row_count_match": len(frame) == expected_rows,
            "columns": int(len(frame.columns)),
            "missing_predictors": missing_predictors,
            "additional_columns": additional_columns,
            "target_present": "target" in frame.columns,
        }
        reports.append(report)
        if len(frame) != expected_rows:
            raise RuntimeError(
                f"Unexpected row count for {name}: {len(frame)} != {expected_rows}"
            )
        if missing_predictors or "target" not in frame.columns:
            raise RuntimeError(f"Schema verification failed for {name}: {report}")
        del frame
    return reports


def stable_protocol_payload(source_protocol: dict, metadata: dict) -> dict:
    return {
        "protocol_name": "Random Forest model-family confirmatory replication",
        "protocol_version": "1.0",
        "status": "FROZEN_BEFORE_TRAINING",
        "relationship_to_primary_study": (
            "Secondary model-family sensitivity analysis; the original five-seed "
            "XGBoost analysis remains primary."
        ),
        "confirmatory_seeds": SEEDS,
        "source_dataset": source_protocol["source_dataset"],
        "external_target_dataset": source_protocol["external_target_dataset"],
        "predictor_count": len(metadata["predictors"]),
        "semantic_group_count": len(metadata["semantic_groups"]),
        "partition_rows": EXPECTED_ROWS,
        "fixed_model_configuration": RF_CONFIGURATION,
        "ordinary_baseline_training": "Complete source training data only",
        "group_aware_training": {
            "complete_copy_per_training_row": 1,
            "random_single_group_mask_copy_per_row": 1,
            "group_availability_indicators": 4,
            "pairwise_masks_used_during_training": False,
            "mask_assignment_seed": "confirmatory seed",
        },
        "calibration": {
            "method": "Platt scaling",
            "data": "Fixed source_calibration.parquet only",
            "split_method": "Stratified 50:50 calibrator-fit/policy split",
            "split_random_state": 2026,
            "target_labels_used": False,
        },
        "classification_threshold": "Source-policy F1 maximization",
        "selective_coverages": [0.8, 0.9, 0.95],
        "evaluated_conditions": {
            "complete": 1,
            "single_group_loss": 4,
            "unseen_pairwise_group_loss": 6,
        },
        "primary_endpoints": [
            "Mean source unseen-pairwise F1",
            "Mean target unseen-pairwise F1",
            "Mean source unseen-pairwise AURC",
            "Mean target unseen-pairwise AURC",
        ],
        "target_result_rules": source_protocol["target_result_rules"],
        "frozen_authoritative_files": source_protocol["frozen_files"],
    }


def freeze_protocol(payload: dict) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PROTOCOL_PATH.exists():
        existing = load_json(OUTPUT_PROTOCOL_PATH)
        existing_payload = {
            key: value for key, value in existing.items() if key != "frozen_on_utc"
        }
        if existing_payload != payload:
            raise RuntimeError(
                "A different Random Forest protocol already exists. Nothing was "
                "overwritten; inspect the existing protocol before continuing."
            )
        return "EXISTING_PROTOCOL_VERIFIED"

    stored = dict(payload)
    stored["frozen_on_utc"] = datetime.now(timezone.utc).isoformat()
    with OUTPUT_PROTOCOL_PATH.open("w", encoding="utf-8") as file:
        json.dump(stored, file, indent=2)
    return "NEW_PROTOCOL_FROZEN"


def main() -> None:
    mount_drive()
    if not SOURCE_PROTOCOL_PATH.exists():
        raise FileNotFoundError(
            f"Authoritative replication protocol not found: {SOURCE_PROTOCOL_PATH}"
        )
    metadata = load_json(CHECKPOINT_DIR / "experiment_metadata.json")
    source_protocol = load_json(SOURCE_PROTOCOL_PATH)

    if metadata["predictor_count"] != 77:
        raise RuntimeError("The metadata does not contain the expected 77 predictors.")
    if len(metadata["semantic_groups"]) != 4:
        raise RuntimeError("The metadata does not contain the expected four groups.")

    print("Verifying the seven frozen authoritative files...")
    file_audit = verify_authoritative_files(source_protocol)
    print("Inspecting all five frozen Parquet partitions...")
    parquet_audit = inspect_parquet_files(metadata)

    payload = stable_protocol_payload(source_protocol, metadata)
    freeze_status = freeze_protocol(payload)

    audit = {
        "status": "PASS",
        "freeze_status": freeze_status,
        "authoritative_file_audit": file_audit,
        "parquet_audit": parquet_audit,
    }
    audit_path = OUTPUT_DIR / "random_forest_pretraining_audit.json"
    with audit_path.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)

    print("\nRandom Forest replication preparation")
    print("=" * 68)
    print("Status: PASS")
    print("Protocol:", freeze_status)
    print("Seeds:", SEEDS)
    print("Predictors:", len(metadata["predictors"]))
    print("Semantic groups:", len(metadata["semantic_groups"]))
    print("Frozen files verified:", len(file_audit), "/", len(file_audit))
    print("Parquet partitions verified:", len(parquet_audit), "/", len(parquet_audit))
    print("Model: RandomForestClassifier")
    print("Trees:", RF_CONFIGURATION["n_estimators"])
    print("Maximum depth:", RF_CONFIGURATION["max_depth"])
    print("Target use: evaluation only")
    print("\nProtocol saved:", OUTPUT_PROTOCOL_PATH)
    print("Audit saved:", audit_path)
    print("\nPRETRAINING_PROTOCOL_COMPLETE")


if __name__ == "__main__":
    main()
