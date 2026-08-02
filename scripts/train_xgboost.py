"""Restartable five-seed source-only XGBoost training.

For every confirmatory seed, this script trains (1) an ordinary complete-flow
baseline and (2) the proposed semantic-group-aware model.  The proposed model
receives one complete copy and one randomly single-group-masked copy of every
source-training row, plus four group-availability indicators.  Target data are
never loaded by this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_recall_curve,
)


DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get(
        "NIDS_CHECKPOINT_DIR",
        "/content/drive/MyDrive/research/researchdata/"
        "NIDS_Research/NIDS_Research_Checkpoints",
    )
)
DEFAULT_SEEDS = [2027, 2028, 2029, 2030, 2031]
SELECTIVE_COVERAGES = [0.80, 0.90, 0.95]


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 15
) -> float:
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1:
            selected = (probability >= lower) & (probability <= upper)
        else:
            selected = (probability >= lower) & (probability < upper)
        count = int(selected.sum())
        if count:
            ece += (count / len(y_true)) * abs(
                float(np.mean(y_true[selected]))
                - float(np.mean(probability[selected]))
            )
    return float(ece)


def fit_platt_and_policy(
    model: xgb.XGBClassifier,
    x_calibrator: np.ndarray,
    y_calibrator: np.ndarray,
    x_policy: np.ndarray,
    y_policy: np.ndarray,
    seed: int,
) -> tuple[LogisticRegression, dict, dict]:
    raw_calibrator = model.predict_proba(x_calibrator)[:, 1]
    calibrator = LogisticRegression(
        C=1_000_000,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    calibrator.fit(probability_to_logit(raw_calibrator), y_calibrator)
    raw_policy = model.predict_proba(x_policy)[:, 1]
    calibrated_policy = calibrator.predict_proba(probability_to_logit(raw_policy))[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_policy, calibrated_policy)
    if len(thresholds) == 0:
        raise RuntimeError("Threshold selection requires both classes in policy data.")
    f1_values = (
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_position = int(np.argmax(f1_values))
    confidence = np.maximum(calibrated_policy, 1 - calibrated_policy)
    policy = {
        "classification_threshold": float(thresholds[best_position]),
        "policy_f1": float(f1_values[best_position]),
        "confidence_cutoffs": {
            str(coverage): float(np.quantile(confidence, 1 - coverage))
            for coverage in SELECTIVE_COVERAGES
        },
        "calibration_method": "Platt scaling",
        "seed": seed,
        "target_data_used": False,
    }
    metrics = {
        "raw": {
            "brier": float(brier_score_loss(y_policy, raw_policy)),
            "logloss": float(log_loss(y_policy, raw_policy)),
            "ece": expected_calibration_error(y_policy, raw_policy),
        },
        "platt_calibrated": {
            "brier": float(brier_score_loss(y_policy, calibrated_policy)),
            "logloss": float(log_loss(y_policy, calibrated_policy)),
            "ece": expected_calibration_error(y_policy, calibrated_policy),
        },
    }
    return calibrator, policy, metrics


def create_group_training_data(
    x_values: np.ndarray,
    y_values: np.ndarray,
    seed: int,
    group_names: list[str],
    group_indices: dict[str, list[int]],
    medians: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.int8)
    row_count = len(x_values)
    group_count = len(group_names)
    assigned = np.random.default_rng(seed).integers(0, group_count, size=row_count)
    masked = x_values.copy()
    masked_availability = np.ones((row_count, group_count), dtype=np.float32)
    assignment_counts: dict[str, int] = {}
    for group_number, group in enumerate(group_names):
        rows = np.where(assigned == group_number)[0]
        columns = group_indices[group]
        masked[np.ix_(rows, columns)] = medians[columns]
        masked_availability[rows, group_number] = 0.0
        assignment_counts[group] = int(len(rows))
    complete_availability = np.ones((row_count, group_count), dtype=np.float32)
    x_augmented = np.vstack(
        (
            np.hstack((x_values, complete_availability)),
            np.hstack((masked, masked_availability)),
        )
    )
    return x_augmented, np.concatenate((y_values, y_values)), assignment_counts


def create_all_known_conditions(
    x_values: np.ndarray,
    y_values: np.ndarray,
    group_names: list[str],
    group_indices: dict[str, list[int]],
    medians: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.int8)
    row_count = len(x_values)
    group_count = len(group_names)
    versions = [np.hstack((x_values, np.ones((row_count, group_count), np.float32)))]
    labels = [y_values]
    for group_number, group in enumerate(group_names):
        masked = x_values.copy()
        masked[:, group_indices[group]] = medians[group_indices[group]]
        availability = np.ones((row_count, group_count), dtype=np.float32)
        availability[:, group_number] = 0.0
        versions.append(np.hstack((masked, availability)))
        labels.append(y_values)
    return np.vstack(versions), np.concatenate(labels)


def build_model(seed: int, configuration: dict, device: str) -> xgb.XGBClassifier:
    model_configuration = dict(configuration)
    model_configuration["device"] = device
    model_configuration["random_state"] = seed
    return xgb.XGBClassifier(**model_configuration)


def load_source_data(checkpoint_dir: Path) -> dict:
    required = [
        "source_train.parquet",
        "source_validation.parquet",
        "source_calibration.parquet",
        "experiment_metadata.json",
        "source_median_imputer.joblib",
    ]
    missing = [name for name in required if not (checkpoint_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts: {missing}")
    with (checkpoint_dir / "experiment_metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    predictors = metadata["predictors"]
    groups = metadata["semantic_groups"]
    group_names = list(groups)
    feature_index = {feature: position for position, feature in enumerate(predictors)}
    group_indices = {
        group: [feature_index[feature] for feature in groups[group]]
        for group in group_names
    }
    train = pd.read_parquet(checkpoint_dir / "source_train.parquet")
    validation = pd.read_parquet(checkpoint_dir / "source_validation.parquet")
    calibration = pd.read_parquet(checkpoint_dir / "source_calibration.parquet")
    imputer = joblib.load(checkpoint_dir / "source_median_imputer.joblib")
    x_train = imputer.transform(train[predictors]).astype("float32")
    x_validation = imputer.transform(validation[predictors]).astype("float32")
    x_calibration = imputer.transform(calibration[predictors]).astype("float32")
    y_train = train["target"].to_numpy(dtype="int8")
    y_validation = validation["target"].to_numpy(dtype="int8")
    y_calibration = calibration["target"].to_numpy(dtype="int8")
    split_hash = pd.util.hash_pandas_object(
        calibration[predictors], index=False
    ).to_numpy(dtype="uint64")
    calibrator_mask = split_hash % 2 == 0
    medians = np.asarray(imputer.statistics_, dtype=np.float32)
    return {
        "predictors": predictors,
        "group_names": group_names,
        "group_indices": group_indices,
        "medians": medians,
        "x_train": x_train,
        "y_train": y_train,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "x_calibrator": x_calibration[calibrator_mask],
        "y_calibrator": y_calibration[calibrator_mask],
        "x_policy": x_calibration[~calibrator_mask],
        "y_policy": y_calibration[~calibrator_mask],
    }


def train_seed(
    seed: int,
    data: dict,
    replication_dir: Path,
    configuration: dict,
    device: str,
) -> None:
    seed_dir = replication_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    completion = seed_dir / "TRAINING_COMPLETE.json"
    if completion.exists():
        print(f"Seed {seed}: already complete")
        return
    group_names = data["group_names"]
    group_indices = data["group_indices"]
    medians = data["medians"]
    x_group_validation, y_group_validation = create_all_known_conditions(
        data["x_validation"],
        data["y_validation"],
        group_names,
        group_indices,
        medians,
    )
    x_group_calibrator, y_group_calibrator = create_all_known_conditions(
        data["x_calibrator"],
        data["y_calibrator"],
        group_names,
        group_indices,
        medians,
    )
    x_group_policy, y_group_policy = create_all_known_conditions(
        data["x_policy"], data["y_policy"], group_names, group_indices, medians
    )
    summary = {"seed": seed, "target_data_used": False}
    print("\n" + "=" * 70 + f"\nCONFIRMATORY SEED: {seed}\n" + "=" * 70)

    start = time.time()
    baseline = build_model(seed, configuration, device)
    baseline.fit(
        data["x_train"],
        data["y_train"],
        eval_set=[(data["x_validation"], data["y_validation"])],
        verbose=50,
    )
    baseline_time = time.time() - start
    baseline.save_model(str(seed_dir / "ordinary_baseline_xgboost.json"))
    calibrator, policy, metrics = fit_platt_and_policy(
        baseline,
        data["x_calibrator"],
        data["y_calibrator"],
        data["x_policy"],
        data["y_policy"],
        seed,
    )
    joblib.dump(calibrator, seed_dir / "ordinary_baseline_platt.joblib")
    (seed_dir / "ordinary_baseline_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    (seed_dir / "ordinary_baseline_calibration.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    summary["ordinary_baseline"] = {
        "training_time_seconds": baseline_time,
        "best_iteration": int(baseline.best_iteration),
        "best_validation_logloss": float(baseline.best_score),
        "classification_threshold": policy["classification_threshold"],
        "policy_f1": policy["policy_f1"],
    }

    x_group_train, y_group_train, counts = create_group_training_data(
        data["x_train"],
        data["y_train"],
        seed,
        group_names,
        group_indices,
        medians,
    )
    start = time.time()
    group_model = build_model(seed, configuration, device)
    group_model.fit(
        x_group_train,
        y_group_train,
        eval_set=[(x_group_validation, y_group_validation)],
        verbose=50,
    )
    group_time = time.time() - start
    group_model.save_model(str(seed_dir / "group_aware_xgboost.json"))
    del x_group_train, y_group_train
    gc.collect()
    calibrator, policy, metrics = fit_platt_and_policy(
        group_model,
        x_group_calibrator,
        y_group_calibrator,
        x_group_policy,
        y_group_policy,
        seed,
    )
    joblib.dump(calibrator, seed_dir / "group_aware_platt.joblib")
    (seed_dir / "group_aware_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    (seed_dir / "group_aware_calibration.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    summary["group_aware"] = {
        "training_time_seconds": group_time,
        "best_iteration": int(group_model.best_iteration),
        "best_validation_logloss": float(group_model.best_score),
        "classification_threshold": policy["classification_threshold"],
        "policy_f1": policy["policy_f1"],
        "single_mask_assignment_counts": counts,
    }
    (seed_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    completion.write_text(
        json.dumps(
            {"seed": seed, "status": "training_complete", "target_data_used": False},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Seed {seed}: source-only training complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = (
        args.checkpoint_dir
        / "five_seed_confirmatory_replication"
        / "replication_protocol_frozen.json"
    )
    with protocol_path.open(encoding="utf-8") as stream:
        protocol = json.load(stream)
    allowed = set(protocol["confirmatory_seeds"])
    if not set(args.seeds).issubset(allowed):
        raise ValueError(f"Seeds must be selected from the frozen set: {sorted(allowed)}")
    configuration = protocol["fixed_model_configuration"]
    data = load_source_data(args.checkpoint_dir)
    print(
        "Source-only preparation:",
        data["x_train"].shape,
        data["x_validation"].shape,
        data["x_calibrator"].shape,
        data["x_policy"].shape,
    )
    if args.dry_run:
        print("DRY_RUN_PASS")
        return
    replication_dir = args.checkpoint_dir / "five_seed_confirmatory_replication"
    for seed in args.seeds:
        train_seed(seed, data, replication_dir, configuration, args.device)


if __name__ == "__main__":
    main()
