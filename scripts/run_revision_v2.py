"""Restartable runner for the frozen revision-v2 comparative NIDS study.

The runner is intentionally stage-gated:

1. ``train`` loads source development data only and freezes model policies.
2. ``freeze-policies`` verifies every job in a direction.
3. ``evaluate-source`` evaluates source test data without loading the target.
4. ``gate`` computes the frozen forward source-only continuation decision.
5. ``evaluate-target`` is allowed only after the source evaluation and gate.

No command silently executes all training jobs.  Use ``--job-id`` for one job
or the explicit ``--all`` flag.  Outputs are written below
``NIDS_CHECKPOINT_DIR/revision_v2`` and never overwrite v1 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations, pairwise
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:  # Planning and audits do not require XGBoost.
    xgb = None


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "protocols" / "revision_v2"
PROTOCOL_PATH = PROTOCOL_DIR / "revision_v2_protocol_frozen.json"
PROTOCOL_AUDIT_PATH = PROTOCOL_DIR / "revision_v2_protocol_audit.json"
JOB_MATRIX_PATH = PROTOCOL_DIR / "revision_v2_job_matrix.csv"
RANDOM_PARTITIONS_PATH = PROTOCOL_DIR / "random_partitions_frozen.json"
REPOSITORY_METADATA_PATH = ROOT / "config" / "experiment_metadata.json"
REPOSITORY_GROUPS_PATH = ROOT / "config" / "semantic_feature_groups.json"
REUSE_AUDIT_PATH = (
    ROOT / "protocols" / "component_ablation" / "corrected_ablation_manifest_audit.csv"
)
V1_PROTOCOL_PATH = ROOT / "protocols" / "replication_protocol_frozen.json"

DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get(
        "NIDS_CHECKPOINT_DIR",
        "/content/drive/MyDrive/research/researchdata/"
        "NIDS_Research/NIDS_Research_Checkpoints",
    )
)
CALIBRATOR_RATIO = 48_889 / 99_084
T_CRITICAL_DF4_95 = 2.7764451051977987
EXPECTED_CONDITION_ROWS = 43
EXPECTED_SELECTIVE_ROWS = 129


def require_xgboost() -> None:
    if xgb is None:
        raise RuntimeError(
            "XGBoost is required for model training, reuse verification, and "
            "evaluation. Install repository requirements first: "
            "python -m pip install -r requirements.txt"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def atomic_joblib(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(payload, temporary, compress=1)
    temporary.replace(destination)


def save_xgboost(model: xgb.XGBClassifier, destination: Path) -> None:
    require_xgboost()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.json")
    model.save_model(str(temporary))
    temporary.replace(destination)


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 15
) -> float:
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for lower, upper in pairwise(edges):
        selected = (
            (probability >= lower) & (probability <= upper)
            if upper == 1
            else (probability >= lower) & (probability < upper)
        )
        count = int(selected.sum())
        if count:
            value += (count / len(y_true)) * abs(
                float(np.mean(y_true[selected])) - float(np.mean(probability[selected]))
            )
    return float(value)


def select_f1_threshold(
    y_true: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        raise RuntimeError("Threshold selection requires both source classes.")
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    position = int(np.nanargmax(f1_values))
    return float(thresholds[position]), float(f1_values[position])


def confidence_cutoffs(
    probability: np.ndarray, coverages: list[float]
) -> dict[str, float]:
    confidence = np.maximum(probability, 1 - probability)
    return {
        str(coverage): float(np.quantile(confidence, 1 - coverage, method="lower"))
        for coverage in coverages
    }


def load_protocol_bundle() -> tuple[dict, dict, dict, list[dict]]:
    protocol = load_json(PROTOCOL_PATH)
    audit = load_json(PROTOCOL_AUDIT_PATH)
    partitions = load_json(RANDOM_PARTITIONS_PATH)
    if audit.get("status") != "PASS" or audit.get("problems"):
        raise RuntimeError("The committed revision-v2 protocol audit is not PASS.")
    if sha256_file(PROTOCOL_PATH) != audit["protocol_sha256"]:
        raise RuntimeError("Frozen revision-v2 protocol hash mismatch.")
    if sha256_file(RANDOM_PARTITIONS_PATH) != audit["random_partitions_sha256"]:
        raise RuntimeError("Frozen random-partition hash mismatch.")
    if sha256_file(JOB_MATRIX_PATH) != audit["job_matrix_sha256"]:
        raise RuntimeError("Frozen revision-v2 job-matrix hash mismatch.")
    jobs = []
    with JOB_MATRIX_PATH.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            job = dict(raw)
            job["model_seed"] = int(job["model_seed"])
            job["partition_seed"] = (
                None if not job["partition_seed"] else int(job["partition_seed"])
            )
            jobs.append(job)
    if len(jobs) != protocol["expected_training_jobs"]["grand_total_if_gate_passes"]:
        raise RuntimeError("Job matrix count differs from the frozen protocol.")
    return protocol, audit, partitions, jobs


def jobs_for_direction(jobs: list[dict], direction: str) -> list[dict]:
    return [job for job in jobs if job["direction"] == direction]


def data_dir_for(checkpoint_dir: Path, direction: str) -> Path:
    if direction == "forward":
        return checkpoint_dir
    return checkpoint_dir / "revision_v2" / "reverse" / "artifacts"


def direction_dir(checkpoint_dir: Path, direction: str) -> Path:
    return checkpoint_dir / "revision_v2" / direction


def job_dir(checkpoint_dir: Path, job: dict) -> Path:
    return checkpoint_dir / job["output_subdir"]


def verify_forward_artifacts(checkpoint_dir: Path) -> dict:
    protocol = load_json(V1_PROTOCOL_PATH)
    expected = protocol["frozen_files"]
    rows = []
    problems = []
    for name, specification in expected.items():
        path = checkpoint_dir / name
        exists = path.exists()
        size = path.stat().st_size if exists else None
        digest = sha256_file(path) if exists else None
        size_pass = exists and size == specification["size_bytes"]
        hash_pass = exists and digest == specification["sha256"]
        rows.append(
            {
                "file": name,
                "exists": exists,
                "size_bytes": size,
                "sha256": digest,
                "size_pass": size_pass,
                "hash_pass": hash_pass,
            }
        )
        if not size_pass or not hash_pass:
            problems.append(name)
    report = {
        "status": "PASS" if not problems else "FAIL",
        "direction": "forward",
        "checked_utc": utc_now(),
        "authoritative_protocol": str(V1_PROTOCOL_PATH.relative_to(ROOT)),
        "files": rows,
        "problems": problems,
        "target_content_inspected": False,
    }
    destination = direction_dir(checkpoint_dir, "forward") / (
        "FORWARD_ARTIFACT_AUDIT.json"
    )
    atomic_json(report, destination)
    if problems:
        raise RuntimeError(f"Forward frozen artifact mismatch: {problems}")
    return report


def verify_reverse_artifacts(checkpoint_dir: Path) -> dict:
    data_dir = data_dir_for(checkpoint_dir, "reverse")
    manifest_path = data_dir / "REVISION_V2_REVERSE_ARTIFACT_MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Reverse artifacts are not frozen. Run "
            "scripts/prepare_revision_v2_reverse.py after the forward gate passes."
        )
    manifest = load_json(manifest_path)
    problems = []
    for name, specification in manifest["files"].items():
        path = data_dir / name
        if not path.exists():
            problems.append(f"missing:{name}")
            continue
        if path.stat().st_size != specification["size_bytes"]:
            problems.append(f"size:{name}")
        if sha256_file(path) != specification["sha256"]:
            problems.append(f"hash:{name}")
    report = {
        "status": "PASS" if not problems else "FAIL",
        "direction": "reverse",
        "checked_utc": utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "problems": problems,
        "target_content_inspected": False,
    }
    atomic_json(
        report,
        direction_dir(checkpoint_dir, "reverse") / "REVERSE_ARTIFACT_AUDIT.json",
    )
    if problems:
        raise RuntimeError(f"Reverse artifact mismatch: {problems}")
    return report


def verify_direction_artifacts(checkpoint_dir: Path, direction: str) -> dict:
    return (
        verify_forward_artifacts(checkpoint_dir)
        if direction == "forward"
        else verify_reverse_artifacts(checkpoint_dir)
    )


def load_definition(data_dir: Path) -> tuple[dict, object]:
    metadata_path = data_dir / "experiment_metadata.json"
    imputer_path = data_dir / "source_median_imputer.joblib"
    if not metadata_path.exists() or not imputer_path.exists():
        raise FileNotFoundError(f"Missing metadata or source imputer in {data_dir}.")
    metadata = load_json(metadata_path)
    expected_metadata = load_json(REPOSITORY_METADATA_PATH)
    expected_groups = load_json(REPOSITORY_GROUPS_PATH)["semantic_groups"]
    predictors = metadata["predictors"]
    groups = metadata["semantic_groups"]
    if predictors != expected_metadata["predictors"]:
        raise RuntimeError("Direction predictor order differs from the frozen order.")
    if groups != expected_groups:
        raise RuntimeError("Direction semantic groups differ from the frozen groups.")
    feature_index = {name: position for position, name in enumerate(predictors)}
    group_names = list(groups)
    group_indices = {
        group: [feature_index[name] for name in groups[group]] for group in group_names
    }
    imputer = joblib.load(imputer_path)
    medians = np.asarray(imputer.statistics_, dtype=np.float32)
    if len(predictors) != 77 or [len(group_indices[g]) for g in group_names] != [
        32,
        23,
        12,
        10,
    ]:
        raise RuntimeError("Frozen feature or group-size boundary changed.")
    return {
        "metadata": metadata,
        "predictors": predictors,
        "groups": groups,
        "group_names": group_names,
        "group_indices": group_indices,
        "medians": medians,
    }, imputer


def load_frame(
    path: Path, definition: dict, imputer: object
) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    x_data = imputer.transform(frame[definition["predictors"]]).astype("float32")
    y_data = frame["target"].to_numpy(dtype="int8")
    if not np.isfinite(x_data).all():
        raise RuntimeError(f"Non-finite values remain after imputation: {path}")
    return x_data, y_data


def load_source_development(checkpoint_dir: Path, direction: str) -> dict:
    data_dir = data_dir_for(checkpoint_dir, direction)
    definition, imputer = load_definition(data_dir)
    output = {"definition": definition}
    for key, filename in {
        "train": "source_train.parquet",
        "validation": "source_validation.parquet",
        "calibration": "source_calibration.parquet",
    }.items():
        output[key] = load_frame(data_dir / filename, definition, imputer)

    x_calibration, y_calibration = output.pop("calibration")
    fit_rows = round(len(y_calibration) * CALIBRATOR_RATIO)
    all_indices = np.arange(len(y_calibration))
    fit_indices, policy_indices = train_test_split(
        all_indices,
        train_size=fit_rows,
        test_size=len(y_calibration) - fit_rows,
        stratify=y_calibration,
        random_state=2026,
        shuffle=True,
    )
    output["calibrator"] = (
        x_calibration[fit_indices],
        y_calibration[fit_indices],
    )
    output["policy"] = (
        x_calibration[policy_indices],
        y_calibration[policy_indices],
    )
    output["calibration_split"] = {
        "method": "stratified train_test_split",
        "seed": 2026,
        "calibrator_rows": len(fit_indices),
        "policy_rows": len(policy_indices),
        "calibrator_ratio": CALIBRATOR_RATIO,
    }
    return output


def mask_groups(
    x_data: np.ndarray,
    groups: tuple[str, ...],
    definition: dict,
    replacement: str = "source_training_median",
) -> np.ndarray:
    output = np.asarray(x_data, dtype=np.float32).copy()
    for group in groups:
        columns = definition["group_indices"][group]
        if replacement == "source_training_median":
            output[:, columns] = definition["medians"][columns]
        elif replacement == "zero":
            output[:, columns] = 0.0
        elif replacement == "native_nan":
            output[:, columns] = np.nan
        else:
            raise ValueError(f"Unknown replacement: {replacement}")
    return output


def matched_known_conditions(
    x_data: np.ndarray, y_data: np.ndarray, definition: dict
) -> tuple[np.ndarray, np.ndarray]:
    conditions = [()] + [(group,) for group in definition["group_names"]]
    x_parts = [mask_groups(x_data, group, definition) for group in conditions]
    return np.vstack(x_parts), np.tile(y_data, len(conditions))


def iid_masked_copy(
    x_data: np.ndarray,
    seed: int,
    budgets: list[int],
    medians: np.ndarray,
    chunk_rows: int = 50_000,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    masked = np.asarray(x_data, dtype=np.float32).copy()
    budget_assignments = rng.choice(np.asarray(budgets), size=len(masked))
    for start in range(0, len(masked), chunk_rows):
        stop = min(start + chunk_rows, len(masked))
        chunk_budget = budget_assignments[start:stop]
        for budget in budgets:
            local_rows = np.flatnonzero(chunk_budget == budget)
            if not len(local_rows):
                continue
            scores = rng.random((len(local_rows), masked.shape[1]), dtype=np.float32)
            columns = np.argpartition(scores, budget - 1, axis=1)[:, :budget]
            global_rows = start + local_rows
            masked[global_rows[:, None], columns] = medians[columns]
    counts = Counter(int(value) for value in budget_assignments)
    return masked, {str(key): int(counts[key]) for key in sorted(counts)}


def build_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    job: dict,
    definition: dict,
    random_partitions: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    variant = job["variant"]
    seed = int(job["model_seed"])
    if variant == "matched_complete_baseline":
        return x_train, y_train, {"complete": len(y_train)}

    if variant == "iid_matched_feature_dropout":
        masked, counts = iid_masked_copy(
            x_train,
            seed,
            [32, 23, 12, 10],
            definition["medians"],
        )
        return (
            np.vstack([x_train, masked]),
            np.concatenate([y_train, y_train]),
            {"masked_feature_budget_counts": counts},
        )

    rng = np.random.default_rng(seed)
    if variant == "random_partition_group_mask":
        partition_seed = str(job["partition_seed"])
        groups = random_partitions["partitions"][partition_seed]
        feature_index = {
            name: position for position, name in enumerate(definition["predictors"])
        }
        names = list(groups)
        indices = {
            name: [feature_index[feature] for feature in groups[name]] for name in names
        }
        assignments = rng.integers(0, len(names), size=len(x_train))
        masked = x_train.copy()
        counts = {}
        for number, name in enumerate(names):
            rows = np.flatnonzero(assignments == number)
            columns = indices[name]
            masked[np.ix_(rows, columns)] = definition["medians"][columns]
            counts[name] = len(rows)
        return (
            np.vstack([x_train, masked]),
            np.concatenate([y_train, y_train]),
            {"partition_seed": int(partition_seed), "mask_counts": counts},
        )

    if variant == "semantic_uniform_singleton":
        names = definition["group_names"]
        assignments = rng.integers(0, len(names), size=len(x_train))
        masked = x_train.copy()
        counts = {}
        for number, name in enumerate(names):
            rows = np.flatnonzero(assignments == number)
            columns = definition["group_indices"][name]
            masked[np.ix_(rows, columns)] = definition["medians"][columns]
            counts[name] = len(rows)
        return (
            np.vstack([x_train, masked]),
            np.concatenate([y_train, y_train]),
            {"semantic_singleton_counts": counts},
        )

    if variant == "semantic_exhaustive_singleton":
        x_parts = [x_train]
        for name in definition["group_names"]:
            x_parts.append(mask_groups(x_train, (name,), definition))
        return (
            np.vstack(x_parts),
            np.tile(y_train, 5),
            {"copies": ["complete"] + definition["group_names"]},
        )

    if variant == "seen_pairwise_oracle_upper_bound":
        pairs = list(combinations(definition["group_names"], 2))
        assignments = rng.integers(0, len(pairs), size=len(x_train))
        masked = x_train.copy()
        counts = {}
        for number, pair in enumerate(pairs):
            rows = np.flatnonzero(assignments == number)
            for name in pair:
                columns = definition["group_indices"][name]
                masked[np.ix_(rows, columns)] = definition["medians"][columns]
            counts["+".join(pair)] = len(rows)
        return (
            np.vstack([x_train, masked]),
            np.concatenate([y_train, y_train]),
            {"seen_pairwise_counts": counts},
        )
    raise ValueError(f"Unsupported variant: {variant}")


def model_configuration(protocol: dict, seed: int, device: str) -> dict:
    configuration = dict(protocol["model"]["fixed_configuration"])
    configuration["random_state"] = seed
    configuration["device"] = device
    return configuration


def new_job_artifacts(output_dir: Path) -> dict[str, Path]:
    return {
        "model": output_dir / "model_xgboost.json",
        "calibrator": output_dir / "platt.joblib",
        "policy": output_dir / "policy.json",
        "model_manifest": output_dir / "MODEL_TRAINED.json",
        "completion": output_dir / "SOURCE_POLICY_FROZEN.json",
    }


def reuse_source_dir(checkpoint_dir: Path, seed: int) -> Path:
    return (
        checkpoint_dir
        / "five_seed_confirmatory_replication"
        / "component_ablation_corrected"
        / f"seed_{seed}"
        / "augmentation_only"
    )


def verify_reused_job(checkpoint_dir: Path, job: dict) -> dict:
    require_xgboost()
    seed = int(job["model_seed"])
    source_dir = reuse_source_dir(checkpoint_dir, seed)
    source_paths = {
        "model": source_dir / "augmentation_only_xgboost.json",
        "calibrator": source_dir / "augmentation_only_platt.joblib",
        "policy": source_dir / "augmentation_only_policy.json",
        "model_manifest": source_dir / "MODEL_TRAINED.json",
        "training_completion": source_dir / "TRAINING_COMPLETE.json",
    }
    missing = [name for name, path in source_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Reusable v1 augmentation-only artifacts missing for seed {seed}: {missing}"
        )
    audit = pd.read_csv(REUSE_AUDIT_PATH)
    row = audit[(audit["seed"] == seed) & (audit["variant"] == "augmentation_only")]
    if len(row) != 1:
        raise RuntimeError(f"No unique corrected reuse audit row for seed {seed}.")
    expected_manifest = str(row.iloc[0]["manifest_sha256"])
    actual_manifest = sha256_file(source_paths["model_manifest"])
    if expected_manifest != actual_manifest:
        raise RuntimeError(f"Reusable manifest hash mismatch for seed {seed}.")
    policy = load_json(source_paths["policy"])
    if policy.get("target_data_used") is not False:
        raise RuntimeError(f"Reusable policy target-use violation for seed {seed}.")
    expected_conditions = ["complete", "each single-group loss"]
    if policy.get("calibration_training_conditions") != expected_conditions:
        raise RuntimeError(f"Reusable calibration-condition mismatch for seed {seed}.")
    if policy.get("policy_selection_conditions") != expected_conditions:
        raise RuntimeError(f"Reusable policy-condition mismatch for seed {seed}.")
    manifest = load_json(source_paths["model_manifest"])
    if manifest.get("variant") != "augmentation_only":
        raise RuntimeError(f"Reusable variant mismatch for seed {seed}.")
    if manifest.get("training_rows") != 1_196_086:
        raise RuntimeError(f"Reusable training-row mismatch for seed {seed}.")
    if manifest.get("validation_rows") != 764_820:
        raise RuntimeError(f"Reusable validation-row mismatch for seed {seed}.")
    expected_configuration = load_json(V1_PROTOCOL_PATH)["fixed_model_configuration"]
    manifest_configuration = manifest.get("xgboost_parameters", {})
    for key, value in expected_configuration.items():
        if manifest_configuration.get(key) != value:
            raise RuntimeError(
                f"Reusable XGBoost parameter mismatch for seed {seed}: {key}."
            )
    if manifest_configuration.get("random_state") != seed:
        raise RuntimeError(f"Reusable random-state mismatch for seed {seed}.")
    model = xgb.XGBClassifier()
    model.load_model(str(source_paths["model"]))
    if int(model.get_booster().num_features()) != 77:
        raise RuntimeError(f"Reusable model feature mismatch for seed {seed}.")
    if int(model.best_iteration) != int(row.iloc[0]["best_iteration"]):
        raise RuntimeError(f"Reusable best-iteration mismatch for seed {seed}.")
    hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    marker = {
        "status": "reused_v1_augmentation_only_verified",
        "verified_utc": utc_now(),
        "job_id": job["job_id"],
        "seed": seed,
        "source_directory": str(source_dir),
        "source_artifacts": {name: str(path) for name, path in source_paths.items()},
        "sha256": hashes,
        "corrected_manifest_sha256": expected_manifest,
        "target_data_used": False,
    }
    output_dir = job_dir(checkpoint_dir, job)
    atomic_json(marker, output_dir / "REUSE_VERIFIED.json")
    atomic_json(
        {
            "status": "source_policy_frozen_by_verified_reuse",
            "job_id": job["job_id"],
            "seed": seed,
            "variant": job["variant"],
            "reuse_marker_sha256": sha256_file(output_dir / "REUSE_VERIFIED.json"),
            "target_data_used": False,
        },
        output_dir / "SOURCE_POLICY_FROZEN.json",
    )
    return marker


def resolved_artifacts(checkpoint_dir: Path, job: dict) -> dict[str, Path]:
    output_dir = job_dir(checkpoint_dir, job)
    if str(job["action"]).startswith("reuse"):
        marker_path = output_dir / "REUSE_VERIFIED.json"
        if not marker_path.exists():
            raise FileNotFoundError(
                f"Reuse verification is missing for {job['job_id']}."
            )
        marker = load_json(marker_path)
        return {
            key: Path(marker["source_artifacts"][key])
            for key in ["model", "calibrator", "policy", "model_manifest"]
        }
    return new_job_artifacts(output_dir)


def fit_calibrator(
    model: xgb.XGBClassifier,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    seed: int,
) -> tuple[LogisticRegression, dict]:
    raw = model.predict_proba(x_fit)[:, 1]
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1_000, random_state=seed)
    calibrator.fit(probability_to_logit(raw), y_fit)
    calibrated = calibrator.predict_proba(probability_to_logit(raw))[:, 1]
    metrics = {
        "rows": len(y_fit),
        "raw_brier": float(brier_score_loss(y_fit, raw)),
        "raw_logloss": float(log_loss(y_fit, raw)),
        "raw_ece": expected_calibration_error(y_fit, raw),
        "calibrated_brier": float(brier_score_loss(y_fit, calibrated)),
        "calibrated_logloss": float(log_loss(y_fit, calibrated)),
        "calibrated_ece": expected_calibration_error(y_fit, calibrated),
    }
    return calibrator, metrics


def predict_calibrated(
    model: xgb.XGBClassifier,
    calibrator: LogisticRegression,
    x_data: np.ndarray,
) -> np.ndarray:
    raw = model.predict_proba(x_data)[:, 1]
    return calibrator.predict_proba(probability_to_logit(raw))[:, 1]


def train_one_job(
    checkpoint_dir: Path,
    job: dict,
    source: dict,
    protocol: dict,
    random_partitions: dict,
    device: str,
) -> None:
    require_xgboost()
    output_dir = job_dir(checkpoint_dir, job)
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(job["action"]).startswith("reuse"):
        verify_reused_job(checkpoint_dir, job)
        print(job["job_id"], "| verified v1 reuse")
        return

    artifacts = new_job_artifacts(output_dir)
    required = [
        artifacts["model"],
        artifacts["calibrator"],
        artifacts["policy"],
        artifacts["model_manifest"],
        artifacts["completion"],
    ]
    if all(path.exists() for path in required):
        marker = load_json(artifacts["completion"])
        if marker.get("target_data_used") is False:
            print(job["job_id"], "| source policy already frozen")
            return

    definition = source["definition"]
    x_train, y_train = source["train"]
    x_validation, y_validation = source["validation_known"]
    seed = int(job["model_seed"])
    started = time.time()

    if artifacts["model"].exists():
        model = xgb.XGBClassifier()
        model.load_model(str(artifacts["model"]))
        training_seconds = None
        training_details = {"restored_existing_model": True}
    else:
        x_variant, y_variant, training_details = build_training_data(
            x_train,
            y_train,
            job,
            definition,
            random_partitions,
        )
        model = xgb.XGBClassifier(**model_configuration(protocol, seed, device))
        training_started = time.time()
        model.fit(
            x_variant,
            y_variant,
            eval_set=[(x_validation, y_validation)],
            verbose=50,
        )
        training_seconds = time.time() - training_started
        save_xgboost(model, artifacts["model"])
        atomic_json(
            {
                "job_id": job["job_id"],
                "direction": job["direction"],
                "variant": job["variant"],
                "partition_seed": job["partition_seed"],
                "model_seed": seed,
                "training_seconds": training_seconds,
                "training_rows": len(y_variant),
                "matched_validation_rows": len(y_validation),
                "best_iteration": int(model.best_iteration),
                "best_score": float(model.best_score),
                "training_details": training_details,
                "xgboost_parameters": model_configuration(protocol, seed, device),
                "pairwise_seen_during_training": job["variant"]
                == "seen_pairwise_oracle_upper_bound",
                "target_data_used": False,
                "model_sha256": sha256_file(artifacts["model"]),
            },
            artifacts["model_manifest"],
        )
        if x_variant is not x_train:
            del x_variant, y_variant
            gc.collect()

    x_fit, y_fit = source["calibrator_known"]
    calibration_path = output_dir / "CALIBRATION_METRICS.json"
    if artifacts["calibrator"].exists():
        calibrator = joblib.load(artifacts["calibrator"])
    else:
        calibrator, metrics = fit_calibrator(model, x_fit, y_fit, seed)
        atomic_joblib(calibrator, artifacts["calibrator"])
        metrics.update(
            {
                "job_id": job["job_id"],
                "conditions": "complete plus four semantic singleton losses",
                "target_data_used": False,
            }
        )
        atomic_json(metrics, calibration_path)

    if not artifacts["policy"].exists():
        x_policy, y_policy = source["policy_known"]
        probability = predict_calibrated(model, calibrator, x_policy)
        threshold, policy_f1 = select_f1_threshold(y_policy, probability)
        coverages = protocol["matched_development_schedule"]["selective_coverages"]
        policy = {
            "job_id": job["job_id"],
            "direction": job["direction"],
            "variant": job["variant"],
            "model_seed": seed,
            "classification_threshold": threshold,
            "source_policy_f1": policy_f1,
            "confidence_cutoffs": confidence_cutoffs(probability, coverages),
            "confidence_definition": "max(p_attack,1-p_attack)",
            "calibration_method": "Platt scaling on source-only logits",
            "development_conditions": (
                "complete plus four semantic singleton losses with "
                "source-training-median replacement"
            ),
            "calibration_split": source["calibration_split"],
            "pairwise_loss_used_for_policy": False,
            "target_data_used": False,
        }
        atomic_json(policy, artifacts["policy"])

    marker = {
        "status": "source_policy_frozen",
        "created_utc": utc_now(),
        "job_id": job["job_id"],
        "direction": job["direction"],
        "variant": job["variant"],
        "partition_seed": job["partition_seed"],
        "model_seed": seed,
        "elapsed_seconds_this_run": time.time() - started,
        "artifacts": {
            name: str(path.name)
            for name, path in artifacts.items()
            if name != "completion"
        },
        "sha256": {
            name: sha256_file(path)
            for name, path in artifacts.items()
            if name != "completion" and path.exists()
        },
        "target_data_used": False,
    }
    atomic_json(marker, artifacts["completion"])
    print(job["job_id"], "| SOURCE POLICY FROZEN")


def source_policy_complete(checkpoint_dir: Path, job: dict) -> tuple[bool, str]:
    marker = job_dir(checkpoint_dir, job) / "SOURCE_POLICY_FROZEN.json"
    if not marker.exists():
        return False, "missing marker"
    payload = load_json(marker)
    if payload.get("target_data_used") is not False:
        return False, "target-use violation"
    try:
        artifacts = resolved_artifacts(checkpoint_dir, job)
    except (FileNotFoundError, KeyError) as error:
        return False, str(error)
    missing = [name for name, path in artifacts.items() if not path.exists()]
    return (not missing, "" if not missing else f"missing:{missing}")


def freeze_direction_policies(
    checkpoint_dir: Path, direction: str, jobs: list[dict], protocol_hash: str
) -> dict:
    problems = []
    selected = jobs_for_direction(jobs, direction)
    for job in selected:
        complete, reason = source_policy_complete(checkpoint_dir, job)
        if not complete:
            problems.append(f"{job['job_id']}:{reason}")
    if problems:
        raise RuntimeError(
            f"Cannot freeze {direction} policies; incomplete jobs: {problems}"
        )
    marker = {
        "status": "all_source_policies_frozen",
        "direction": direction,
        "created_utc": utc_now(),
        "job_count": len(selected),
        "job_ids": [job["job_id"] for job in selected],
        "protocol_sha256": protocol_hash,
        "target_loaded": False,
    }
    destination = direction_dir(checkpoint_dir, direction) / (
        "ALL_SOURCE_POLICIES_FROZEN.json"
    )
    atomic_json(marker, destination)
    print(direction, "| ALL SOURCE POLICIES FROZEN")
    return marker


def evaluation_conditions(group_names: list[str]) -> list[dict]:
    conditions = [
        {
            "condition_type": "complete",
            "severity": 0,
            "missing_groups": (),
            "replacement": "none",
        }
    ]
    for severity, group_sets in [
        (1, combinations(group_names, 1)),
        (2, combinations(group_names, 2)),
        (3, combinations(group_names, 3)),
    ]:
        for groups in group_sets:
            for replacement in [
                "source_training_median",
                "zero",
                "native_nan",
            ]:
                conditions.append(
                    {
                        "condition_type": {
                            1: "single",
                            2: "pairwise",
                            3: "triple",
                        }[severity],
                        "severity": severity,
                        "missing_groups": tuple(groups),
                        "replacement": replacement,
                    }
                )
    if len(conditions) != EXPECTED_CONDITION_ROWS:
        raise AssertionError(f"Expected 43 conditions; found {len(conditions)}")
    return conditions


def evaluate_probability(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict:
    prediction = (probability >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "rows": len(y_true),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)),
        "brier": float(brier_score_loss(y_true, probability)),
        "logloss": float(log_loss(y_true, probability)),
        "ece": expected_calibration_error(y_true, probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def calculate_aurc(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> float:
    prediction = (probability >= threshold).astype("int8")
    confidence = np.maximum(probability, 1 - probability)
    errors = (prediction != y_true).astype("float64")
    order = np.argsort(-confidence)
    cumulative_errors = np.cumsum(errors[order])
    accepted_count = np.arange(1, len(y_true) + 1)
    return float(np.mean(cumulative_errors / accepted_count))


def evaluate_selective(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    confidence_cutoff: float,
) -> dict:
    prediction = (probability >= threshold).astype("int8")
    confidence = np.maximum(probability, 1 - probability)
    accepted = confidence >= confidence_cutoff
    benign = y_true == 0
    attack = y_true == 1
    accepted_rows = int(accepted.sum())
    result = {
        "actual_coverage": float(accepted.mean()),
        "benign_coverage": float(accepted[benign].mean()),
        "attack_coverage": float(accepted[attack].mean()),
        "accepted_rows": accepted_rows,
        "abstained_rows": int((~accepted).sum()),
        "selective_risk": np.nan,
        "accepted_accuracy": np.nan,
        "accepted_precision": np.nan,
        "accepted_recall": np.nan,
        "accepted_f1": np.nan,
        "attack_abstention_rate": float((~accepted[attack]).mean()),
        "accepted_false_negative_rate": float(
            (accepted & attack & (prediction == 0)).sum() / attack.sum()
        ),
    }
    if accepted_rows:
        accepted_y = y_true[accepted]
        accepted_prediction = prediction[accepted]
        result.update(
            {
                "selective_risk": float(np.mean(accepted_prediction != accepted_y)),
                "accepted_accuracy": float(
                    accuracy_score(accepted_y, accepted_prediction)
                ),
                "accepted_precision": float(
                    precision_score(accepted_y, accepted_prediction, zero_division=0)
                ),
                "accepted_recall": float(
                    recall_score(accepted_y, accepted_prediction, zero_division=0)
                ),
                "accepted_f1": float(
                    f1_score(accepted_y, accepted_prediction, zero_division=0)
                ),
            }
        )
    return result


def load_test_split(
    checkpoint_dir: Path, direction: str, dataset_role: str
) -> tuple[dict, np.ndarray, np.ndarray]:
    data_dir = data_dir_for(checkpoint_dir, direction)
    definition, imputer = load_definition(data_dir)
    filename = (
        "source_test.parquet"
        if dataset_role == "source"
        else "target_test_sealed.parquet"
    )
    x_data, y_data = load_frame(data_dir / filename, definition, imputer)
    return definition, x_data, y_data


def evaluate_one_job(
    checkpoint_dir: Path,
    job: dict,
    dataset_role: str,
    definition: dict,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> None:
    require_xgboost()
    output_dir = job_dir(checkpoint_dir, job)
    prefix = dataset_role.upper()
    marker_path = output_dir / f"{prefix}_EVALUATION_COMPLETE.json"
    nonselective_path = output_dir / f"{dataset_role}_nonselective.csv"
    selective_path = output_dir / f"{dataset_role}_selective.csv"
    if all(path.exists() for path in [marker_path, nonselective_path, selective_path]):
        print(job["job_id"], "|", dataset_role, "evaluation already complete")
        return

    artifacts = resolved_artifacts(checkpoint_dir, job)
    model = xgb.XGBClassifier()
    model.load_model(str(artifacts["model"]))
    calibrator = joblib.load(artifacts["calibrator"])
    policy = load_json(artifacts["policy"])
    threshold = float(policy["classification_threshold"])
    cutoffs = {
        float(key): float(value) for key, value in policy["confidence_cutoffs"].items()
    }
    nonselective_rows = []
    selective_rows = []
    started = time.time()
    conditions = evaluation_conditions(definition["group_names"])
    for number, condition in enumerate(conditions, start=1):
        missing_groups = condition["missing_groups"]
        if condition["severity"] == 0:
            x_condition = np.asarray(x_data, dtype=np.float32)
        else:
            x_condition = mask_groups(
                x_data,
                missing_groups,
                definition,
                condition["replacement"],
            )
        probability = predict_calibrated(model, calibrator, x_condition)
        metrics = evaluate_probability(y_data, probability, threshold)
        common = {
            "job_id": job["job_id"],
            "direction": job["direction"],
            "dataset_role": dataset_role,
            "variant": job["variant"],
            "partition_seed": job["partition_seed"],
            "model_seed": job["model_seed"],
            "condition_type": condition["condition_type"],
            "severity": condition["severity"],
            "missing_groups": ("+".join(missing_groups) if missing_groups else "None"),
            "replacement": condition["replacement"],
            "pairwise_seen_during_training": job["variant"]
            == "seen_pairwise_oracle_upper_bound",
            "generalization_status": (
                "seen_upper_bound"
                if job["variant"] == "seen_pairwise_oracle_upper_bound"
                and condition["severity"] == 2
                else "unseen_or_not_applicable"
            ),
        }
        metrics.update(common)
        nonselective_rows.append(metrics)
        aurc = calculate_aurc(y_data, probability, threshold)
        for desired_coverage, cutoff in cutoffs.items():
            selective = evaluate_selective(y_data, probability, threshold, cutoff)
            selective.update(common)
            selective.update(
                {
                    "desired_policy_coverage": desired_coverage,
                    "confidence_cutoff": cutoff,
                    "aurc": aurc,
                }
            )
            selective_rows.append(selective)
        if condition["severity"]:
            del x_condition
        del probability
        gc.collect()
        print(
            job["job_id"],
            "|",
            dataset_role,
            f"{number}/{len(conditions)}",
            condition["condition_type"],
            common["missing_groups"],
            condition["replacement"],
        )

    if len(nonselective_rows) != EXPECTED_CONDITION_ROWS:
        raise RuntimeError("Non-selective condition-row count mismatch.")
    if len(selective_rows) != EXPECTED_SELECTIVE_ROWS:
        raise RuntimeError("Selective condition-row count mismatch.")
    atomic_csv(pd.DataFrame(nonselective_rows), nonselective_path)
    atomic_csv(pd.DataFrame(selective_rows), selective_path)
    source_policy_marker = output_dir / "SOURCE_POLICY_FROZEN.json"
    marker = {
        "status": f"{dataset_role}_evaluation_complete",
        "created_utc": utc_now(),
        "job_id": job["job_id"],
        "direction": job["direction"],
        "dataset_role": dataset_role,
        "elapsed_seconds": time.time() - started,
        "rows": len(y_data),
        "nonselective_rows": len(nonselective_rows),
        "selective_rows": len(selective_rows),
        "source_policy_marker_sha256": sha256_file(source_policy_marker),
        "target_used_for": "evaluation only" if dataset_role == "target" else None,
    }
    atomic_json(marker, marker_path)
    print(job["job_id"], "|", dataset_role, "EVALUATION COMPLETE")
    del model, calibrator
    gc.collect()


def combine_evaluations(
    checkpoint_dir: Path,
    direction: str,
    dataset_role: str,
    jobs: list[dict],
) -> dict | None:
    nonselective = []
    selective = []
    missing = []
    for job in jobs_for_direction(jobs, direction):
        output_dir = job_dir(checkpoint_dir, job)
        nonselective_path = output_dir / f"{dataset_role}_nonselective.csv"
        selective_path = output_dir / f"{dataset_role}_selective.csv"
        marker = output_dir / f"{dataset_role.upper()}_EVALUATION_COMPLETE.json"
        if not all(
            path.exists() for path in [nonselective_path, selective_path, marker]
        ):
            missing.append(job["job_id"])
            continue
        nonselective.append(pd.read_csv(nonselective_path))
        selective.append(pd.read_csv(selective_path))
    if missing:
        print(
            direction,
            dataset_role,
            "| combined output pending; missing jobs:",
            len(missing),
        )
        return None
    combined_nonselective = pd.concat(nonselective, ignore_index=True)
    combined_selective = pd.concat(selective, ignore_index=True)
    expected_nonselective = len(nonselective) * EXPECTED_CONDITION_ROWS
    expected_selective = len(selective) * EXPECTED_SELECTIVE_ROWS
    if len(combined_nonselective) != expected_nonselective:
        raise RuntimeError("Combined non-selective row count mismatch.")
    if len(combined_selective) != expected_selective:
        raise RuntimeError("Combined selective row count mismatch.")
    output_dir = direction_dir(checkpoint_dir, direction)
    nonselective_path = output_dir / f"combined_{dataset_role}_nonselective.csv"
    selective_path = output_dir / f"combined_{dataset_role}_selective.csv"
    atomic_csv(combined_nonselective, nonselective_path)
    atomic_csv(combined_selective, selective_path)
    marker = {
        "status": f"all_{dataset_role}_evaluations_complete",
        "created_utc": utc_now(),
        "direction": direction,
        "job_count": len(nonselective),
        "nonselective_rows": len(combined_nonselective),
        "selective_rows": len(combined_selective),
        "nonselective_sha256": sha256_file(nonselective_path),
        "selective_sha256": sha256_file(selective_path),
    }
    atomic_json(
        marker,
        output_dir / f"ALL_{dataset_role.upper()}_EVALUATIONS_COMPLETE.json",
    )
    print(direction, dataset_role, "| COMBINED EVALUATION COMPLETE")
    return marker


def t_interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    half_width = T_CRITICAL_DF4_95 * sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "wins": int((values > 0).sum()),
        "differences_by_seed": [float(value) for value in values],
    }


def calculate_forward_gate(frame: pd.DataFrame) -> dict:
    pairwise = frame[
        (frame["direction"] == "forward")
        & (frame["dataset_role"] == "source")
        & (frame["severity"] == 2)
        & (frame["replacement"] == "source_training_median")
    ].copy()
    per_job = pairwise.groupby(
        ["model_seed", "variant", "partition_seed"],
        dropna=False,
        as_index=False,
    )["f1"].mean()
    semantic = (
        per_job[per_job["variant"] == "semantic_uniform_singleton"]
        .set_index("model_seed")["f1"]
        .sort_index()
    )
    iid = (
        per_job[per_job["variant"] == "iid_matched_feature_dropout"]
        .set_index("model_seed")["f1"]
        .sort_index()
    )
    random = (
        per_job[per_job["variant"] == "random_partition_group_mask"]
        .groupby("model_seed")["f1"]
        .mean()
        .sort_index()
    )
    expected_seeds = [2027, 2028, 2029, 2030, 2031]
    for name, series in [("semantic", semantic), ("iid", iid), ("random", random)]:
        if list(series.index) != expected_seeds:
            raise RuntimeError(f"Gate {name} seeds incomplete: {list(series.index)}")
    comparisons = {
        "semantic_minus_iid": t_interval(semantic.to_numpy() - iid.to_numpy()),
        "semantic_minus_random_partition_average": t_interval(
            semantic.to_numpy() - random.to_numpy()
        ),
    }
    for comparison in comparisons.values():
        comparison["passes_positive_mean"] = comparison["mean_difference"] > 0
        comparison["passes_ci_lower_above_zero"] = comparison["ci95_lower"] > 0
        comparison["passes_four_of_five_wins"] = comparison["wins"] >= 4
        comparison["passes"] = all(
            [
                comparison["passes_positive_mean"],
                comparison["passes_ci_lower_above_zero"],
                comparison["passes_four_of_five_wins"],
            ]
        )

    complete = frame[
        (frame["direction"] == "forward")
        & (frame["dataset_role"] == "source")
        & (frame["severity"] == 0)
    ]
    complete_means = complete.groupby("variant")["f1"].mean()
    required = {"semantic_uniform_singleton", "matched_complete_baseline"}
    if not required.issubset(set(complete_means.index)):
        raise RuntimeError(
            "Complete-condition values required for the gate are missing."
        )
    baseline_mean = float(complete_means["matched_complete_baseline"])
    semantic_mean = float(complete_means["semantic_uniform_singleton"])
    penalty = baseline_mean - semantic_mean
    complete_check = {
        "matched_complete_baseline_mean_f1": baseline_mean,
        "semantic_uniform_singleton_mean_f1": semantic_mean,
        "baseline_minus_semantic_penalty": penalty,
        "maximum_allowed_penalty": 0.005,
        "passes": penalty <= 0.005,
    }
    gate_pass = (
        all(value["passes"] for value in comparisons.values())
        and complete_check["passes"]
    )
    return {
        "status": "PASS" if gate_pass else "FAIL",
        "decision_basis": (
            "Forward source-test pairwise-average attack-class F1 with "
            "source-training-median replacement only"
        ),
        "forward_target_results_used": False,
        "comparisons": comparisons,
        "complete_telemetry_penalty": complete_check,
        "reverse_training_authorized": gate_pass,
        "interpretation": (
            "Proceed with reverse direction and method-comparison framing."
            if gate_pass
            else "Do not make a causal semantic-grouping claim or start reverse training."
        ),
    }


def freeze_forward_gate(checkpoint_dir: Path) -> dict:
    forward_dir = direction_dir(checkpoint_dir, "forward")
    source_marker = forward_dir / "ALL_SOURCE_EVALUATIONS_COMPLETE.json"
    source_results = forward_dir / "combined_source_nonselective.csv"
    if not source_marker.exists() or not source_results.exists():
        raise FileNotFoundError("Complete forward source evaluation is required.")
    frame = pd.read_csv(source_results)
    gate = calculate_forward_gate(frame)
    gate.update(
        {
            "created_utc": utc_now(),
            "source_results_sha256": sha256_file(source_results),
            "source_completion_marker_sha256": sha256_file(source_marker),
        }
    )
    destination = forward_dir / "FORWARD_SOURCE_ONLY_CONTINUATION_GATE.json"
    if destination.exists():
        existing = load_json(destination)
        comparable_existing = {
            key: value for key, value in existing.items() if key != "created_utc"
        }
        comparable_new = {
            key: value for key, value in gate.items() if key != "created_utc"
        }
        if comparable_existing != comparable_new:
            raise RuntimeError(
                "Frozen forward gate differs from recomputation; do not overwrite it."
            )
        return existing
    atomic_json(gate, destination)
    print("FORWARD SOURCE-ONLY GATE:", gate["status"])
    return gate


def require_forward_gate(checkpoint_dir: Path, must_pass: bool) -> dict:
    path = (
        direction_dir(checkpoint_dir, "forward")
        / "FORWARD_SOURCE_ONLY_CONTINUATION_GATE.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            "The frozen forward source-only continuation gate does not exist."
        )
    gate = load_json(path)
    if must_pass and gate.get("status") != "PASS":
        raise RuntimeError(
            "Reverse execution is blocked because the forward gate failed."
        )
    return gate


def select_jobs(args: argparse.Namespace, jobs: list[dict]) -> list[dict]:
    candidates = jobs_for_direction(jobs, args.direction)
    if args.job_id:
        selected = [job for job in candidates if job["job_id"] == args.job_id]
        if not selected:
            raise ValueError(f"Unknown {args.direction} job id: {args.job_id}")
        return selected
    if args.all:
        return candidates
    raise ValueError("Select exactly one --job-id or explicitly provide --all.")


def plan_rows(checkpoint_dir: Path, jobs: list[dict], direction: str) -> list[dict]:
    rows = []
    for job in jobs_for_direction(jobs, direction):
        complete, reason = source_policy_complete(checkpoint_dir, job)
        output_dir = job_dir(checkpoint_dir, job)
        rows.append(
            {
                **job,
                "source_policy_frozen": complete,
                "status_note": reason,
                "source_evaluated": (
                    output_dir / "SOURCE_EVALUATION_COMPLETE.json"
                ).exists(),
                "target_evaluated": (
                    output_dir / "TARGET_EVALUATION_COMPLETE.json"
                ).exists(),
            }
        )
    return rows


def print_plan(checkpoint_dir: Path, jobs: list[dict], direction: str) -> None:
    rows = plan_rows(checkpoint_dir, jobs, direction)
    print(
        json.dumps(
            {
                "direction": direction,
                "jobs": len(rows),
                "new": sum(row["action"] == "train_new" for row in rows),
                "reuse": sum(str(row["action"]).startswith("reuse") for row in rows),
                "source_policies_frozen": sum(
                    row["source_policy_frozen"] for row in rows
                ),
                "source_evaluated": sum(row["source_evaluated"] for row in rows),
                "target_evaluated": sum(row["target_evaluated"] for row in rows),
                "job_ids": [row["job_id"] for row in rows],
            },
            indent=2,
        )
    )


def prepare_source_known_conditions(source: dict) -> None:
    definition = source["definition"]
    for key in ["validation", "calibrator", "policy"]:
        x_data, y_data = source[key]
        source[f"{key}_known"] = matched_known_conditions(x_data, y_data, definition)


def run_training(
    args: argparse.Namespace,
    protocol: dict,
    audit: dict,
    random_partitions: dict,
    jobs: list[dict],
) -> None:
    if args.direction == "reverse":
        require_forward_gate(args.checkpoint_dir, must_pass=True)
    selected = select_jobs(args, jobs)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "direction": args.direction,
                    "selected_jobs": [job["job_id"] for job in selected],
                    "protocol_sha256": audit["protocol_sha256"],
                    "target_loaded": False,
                },
                indent=2,
            )
        )
        return
    verify_direction_artifacts(args.checkpoint_dir, args.direction)
    source = load_source_development(args.checkpoint_dir, args.direction)
    prepare_source_known_conditions(source)
    print(
        args.direction,
        "| source development shapes:",
        source["train"][0].shape,
        source["validation_known"][0].shape,
        source["calibrator_known"][0].shape,
        source["policy_known"][0].shape,
    )
    for job in selected:
        train_one_job(
            args.checkpoint_dir,
            job,
            source,
            protocol,
            random_partitions,
            args.device,
        )
    del source
    gc.collect()


def run_evaluation(
    args: argparse.Namespace,
    jobs: list[dict],
    dataset_role: str,
) -> None:
    selected = select_jobs(args, jobs)
    direction_root = direction_dir(args.checkpoint_dir, args.direction)
    policies_marker = direction_root / "ALL_SOURCE_POLICIES_FROZEN.json"
    if not policies_marker.exists():
        raise FileNotFoundError(
            f"Freeze every {args.direction} source policy before evaluation."
        )
    if args.direction == "reverse":
        require_forward_gate(args.checkpoint_dir, must_pass=True)
    if dataset_role == "target":
        source_marker = direction_root / "ALL_SOURCE_EVALUATIONS_COMPLETE.json"
        if not source_marker.exists():
            raise FileNotFoundError(
                "All source evaluations must finish before target loading."
            )
        if args.direction == "forward":
            require_forward_gate(args.checkpoint_dir, must_pass=False)
    definition, x_data, y_data = load_test_split(
        args.checkpoint_dir, args.direction, dataset_role
    )
    print(
        args.direction,
        dataset_role,
        "| rows:",
        len(y_data),
        "| attack rate:",
        float(y_data.mean()),
    )
    for job in selected:
        evaluate_one_job(
            args.checkpoint_dir,
            job,
            dataset_role,
            definition,
            x_data,
            y_data,
        )
    combine_evaluations(
        args.checkpoint_dir,
        args.direction,
        dataset_role,
        jobs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "plan",
            "train",
            "freeze-policies",
            "evaluate-source",
            "gate",
            "evaluate-target",
        ],
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--direction", choices=["forward", "reverse"], default="forward"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--job-id")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol, audit, random_partitions, jobs = load_protocol_bundle()
    if args.command == "plan":
        print_plan(args.checkpoint_dir, jobs, args.direction)
        return
    if args.command == "train":
        run_training(args, protocol, audit, random_partitions, jobs)
        return
    if args.command == "freeze-policies":
        if args.direction == "reverse":
            require_forward_gate(args.checkpoint_dir, must_pass=True)
        freeze_direction_policies(
            args.checkpoint_dir,
            args.direction,
            jobs,
            audit["protocol_sha256"],
        )
        return
    if args.command == "evaluate-source":
        run_evaluation(args, jobs, "source")
        return
    if args.command == "gate":
        if args.direction != "forward":
            raise ValueError("The frozen continuation gate is forward-only.")
        freeze_forward_gate(args.checkpoint_dir)
        return
    if args.command == "evaluate-target":
        run_evaluation(args, jobs, "target")
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
