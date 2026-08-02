"""Restartable five-seed Random Forest training for the NIDS study.

Run in Google Colab with:
    %run "/content/drive/MyDrive/research/researchdata/NIDS_Research/train_random_forest_five_seed.py"

The script trains an ordinary Random Forest and the full group-aware Random
Forest for each frozen seed. Models are checkpointed after every 50 trees.
Calibration, threshold selection, and confidence policy selection use source
data only. The sealed target is never loaded by this training script.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


CHECKPOINT_DIR = Path(
    os.environ.get(
        "NIDS_CHECKPOINT_DIR",
        "/content/drive/MyDrive/research/researchdata/"
        "NIDS_Research/NIDS_Research_Checkpoints",
    )
)
RF_DIR = (
    CHECKPOINT_DIR
    / "five_seed_confirmatory_replication"
    / "random_forest_model_family_replication"
)
PROTOCOL_PATH = RF_DIR / "random_forest_protocol_frozen.json"
SEEDS = [2027, 2028, 2029, 2030, 2031]
TREE_CHECKPOINTS = [50, 100, 150, 200, 250, 300]
CALIBRATION_SPLIT_SEED = 2026
SELECTIVE_COVERAGES = [0.8, 0.9, 0.95]


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError:
        return
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def atomic_json_dump(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    os.replace(temporary, path)


def atomic_joblib_dump(payload, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(payload, temporary, compress=1)
    os.replace(temporary, path)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability), 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


def select_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    # A fixed dense grid avoids choosing among hundreds of thousands of sample-
    # specific thresholds and makes the policy exactly reproducible.
    thresholds = np.linspace(0.001, 0.999, 999, dtype=np.float64)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        score = f1_score(y_true, probability >= threshold, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def confidence_cutoffs(probability: np.ndarray) -> dict[str, float]:
    confidence = np.maximum(probability, 1 - probability)
    cutoffs = {}
    for coverage in SELECTIVE_COVERAGES:
        cutoff = np.quantile(confidence, 1 - coverage, method="higher")
        cutoffs[str(coverage)] = float(cutoff)
    return cutoffs


def fit_platt_and_policy(
    model: RandomForestClassifier,
    x_calibrator: np.ndarray,
    y_calibrator: np.ndarray,
    x_policy: np.ndarray,
    y_policy: np.ndarray,
    seed: int,
) -> tuple[LogisticRegression, dict]:
    raw_calibrator = model.predict_proba(x_calibrator)[:, 1]
    calibrator = LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=seed
    )
    calibrator.fit(probability_to_logit(raw_calibrator), y_calibrator)

    raw_policy = model.predict_proba(x_policy)[:, 1]
    calibrated_policy = calibrator.predict_proba(
        probability_to_logit(raw_policy)
    )[:, 1]
    threshold, policy_f1 = select_f1_threshold(y_policy, calibrated_policy)
    policy = {
        "classification_threshold": threshold,
        "source_policy_f1": policy_f1,
        "confidence_cutoffs": confidence_cutoffs(calibrated_policy),
        "selective_coverages": SELECTIVE_COVERAGES,
        "calibration_method": "Platt scaling",
        "threshold_selection": "source-policy F1 maximization on fixed grid",
        "target_labels_used": False,
    }
    return calibrator, policy


def build_forest(seed: int, tree_count: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=tree_count,
        criterion="gini",
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        max_samples=0.7,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
        warm_start=True,
        verbose=0,
    )


def verify_forest(model: RandomForestClassifier, seed: int) -> None:
    check_is_fitted(model)
    parameters = model.get_params()
    expected = {
        "criterion": "gini",
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "bootstrap": True,
        "max_samples": 0.7,
        "class_weight": "balanced_subsample",
        "random_state": seed,
    }
    mismatches = {
        key: {"expected": value, "actual": parameters.get(key)}
        for key, value in expected.items()
        if parameters.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Random Forest parameter mismatch: {mismatches}")
    if len(model.estimators_) > 300:
        raise RuntimeError("The stored forest has more than the frozen 300 trees.")


def train_forest_restartable(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_path: Path,
    label: str,
) -> RandomForestClassifier:
    if model_path.exists():
        print(f"Loading {label} checkpoint: {model_path.name}")
        model = joblib.load(model_path)
        verify_forest(model, seed)
        completed_trees = len(model.estimators_)
    else:
        model = build_forest(seed, TREE_CHECKPOINTS[0])
        completed_trees = 0

    if completed_trees == 300:
        print(f"{label}: all 300 trees already trained.")
        return model

    for tree_count in TREE_CHECKPOINTS:
        if tree_count <= completed_trees:
            continue
        print(f"{label}: training trees {completed_trees + 1}-{tree_count}...")
        started = time.time()
        model.set_params(n_estimators=tree_count)
        model.fit(x_train, y_train)
        verify_forest(model, seed)
        atomic_joblib_dump(model, model_path)
        completed_trees = tree_count
        print(
            f"{label}: {tree_count}/300 trees saved in "
            f"{time.time() - started:.1f} seconds."
        )
    return model


def make_group_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    medians: np.ndarray,
    group_names: list[str],
    group_indices: dict[str, list[int]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rows, predictor_count = x_train.shape
    group_count = len(group_names)
    output = np.empty(
        (rows * 2, predictor_count + group_count), dtype=np.float32
    )
    output[:rows, :predictor_count] = x_train
    output[rows:, :predictor_count] = x_train
    output[:, predictor_count:] = 1.0

    rng = np.random.default_rng(seed)
    assignments = rng.integers(0, group_count, size=rows)
    counts = {}
    for group_number, group in enumerate(group_names):
        local_rows = np.flatnonzero(assignments == group_number)
        augmented_rows = local_rows + rows
        columns = np.asarray(group_indices[group], dtype=np.int64)
        output[np.ix_(augmented_rows, columns)] = medians[columns]
        output[augmented_rows, predictor_count + group_number] = 0.0
        counts[group] = int(len(local_rows))
    output_y = np.concatenate([y_train, y_train]).astype(np.int8, copy=False)
    return output, output_y, counts


def expand_complete_and_single_masks(
    x: np.ndarray,
    y: np.ndarray,
    medians: np.ndarray,
    group_names: list[str],
    group_indices: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    rows, predictor_count = x.shape
    group_count = len(group_names)
    copies = group_count + 1
    output = np.empty(
        (rows * copies, predictor_count + group_count), dtype=np.float32
    )
    output_y = np.tile(y, copies).astype(np.int8, copy=False)
    for copy_number in range(copies):
        start = copy_number * rows
        stop = start + rows
        output[start:stop, :predictor_count] = x
        output[start:stop, predictor_count:] = 1.0
        if copy_number:
            group_number = copy_number - 1
            group = group_names[group_number]
            columns = np.asarray(group_indices[group], dtype=np.int64)
            output[start:stop, columns] = medians[columns]
            output[start:stop, predictor_count + group_number] = 0.0
    return output, output_y


def load_source_training_data() -> dict:
    metadata = load_json(CHECKPOINT_DIR / "experiment_metadata.json")
    predictors = metadata["predictors"]
    groups = metadata["semantic_groups"]
    group_names = list(groups)
    feature_index = {name: index for index, name in enumerate(predictors)}
    group_indices = {
        group: [feature_index[feature] for feature in groups[group]]
        for group in group_names
    }
    imputer = joblib.load(CHECKPOINT_DIR / "source_median_imputer.joblib")
    medians = np.asarray(imputer.statistics_, dtype=np.float32)

    train_df = pd.read_parquet(CHECKPOINT_DIR / "source_train.parquet")
    validation_df = pd.read_parquet(CHECKPOINT_DIR / "source_validation.parquet")
    calibration_df = pd.read_parquet(CHECKPOINT_DIR / "source_calibration.parquet")

    x_train = imputer.transform(train_df[predictors]).astype(np.float32)
    y_train = train_df["target"].to_numpy(dtype=np.int8)
    x_validation = imputer.transform(validation_df[predictors]).astype(np.float32)
    y_validation = validation_df["target"].to_numpy(dtype=np.int8)
    x_calibration = imputer.transform(calibration_df[predictors]).astype(np.float32)
    y_calibration = calibration_df["target"].to_numpy(dtype=np.int8)
    del train_df, validation_df, calibration_df
    gc.collect()

    if any(
        np.isnan(array).any()
        for array in (x_train, x_validation, x_calibration)
    ):
        raise RuntimeError("NaN values remain after source-median imputation.")

    indices = np.arange(len(y_calibration))
    fit_indices, policy_indices = train_test_split(
        indices,
        test_size=0.5,
        random_state=CALIBRATION_SPLIT_SEED,
        stratify=y_calibration,
    )
    split_path = RF_DIR / "calibration_policy_split_indices.npz"
    if split_path.exists():
        saved = np.load(split_path)
        if not (
            np.array_equal(saved["fit_indices"], fit_indices)
            and np.array_equal(saved["policy_indices"], policy_indices)
        ):
            raise RuntimeError("Saved calibration split does not match the protocol.")
    else:
        np.savez_compressed(
            split_path,
            fit_indices=fit_indices,
            policy_indices=policy_indices,
        )

    print("Source training:", x_train.shape)
    print("Source validation:", x_validation.shape)
    print("Source calibration:", x_calibration.shape)
    print("Calibrator-fit rows:", len(fit_indices))
    print("Policy-selection rows:", len(policy_indices))
    print("Predictors:", len(predictors))
    print("Semantic groups:", len(group_names))

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "x_calibration": x_calibration,
        "y_calibration": y_calibration,
        "fit_indices": fit_indices,
        "policy_indices": policy_indices,
        "medians": medians,
        "group_names": group_names,
        "group_indices": group_indices,
    }


def calibrated_validation_metrics(
    model: RandomForestClassifier,
    calibrator: LogisticRegression,
    policy: dict,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> dict:
    raw = model.predict_proba(x_validation)[:, 1]
    probability = calibrator.predict_proba(probability_to_logit(raw))[:, 1]
    threshold = float(policy["classification_threshold"])
    return {
        "rows": int(len(y_validation)),
        "f1": float(f1_score(y_validation, probability >= threshold)),
        "logloss": float(log_loss(y_validation, probability)),
    }


def train_baseline(seed: int, seed_dir: Path, data: dict) -> dict:
    completion_path = seed_dir / "BASELINE_TRAINING_COMPLETE.json"
    required = [
        seed_dir / "ordinary_random_forest.joblib",
        seed_dir / "ordinary_platt.joblib",
        seed_dir / "ordinary_policy.json",
    ]
    if completion_path.exists() and all(path.exists() for path in required):
        print("Ordinary baseline: COMPLETE; skipping.")
        return load_json(completion_path)

    started = time.time()
    model_path = required[0]
    model = train_forest_restartable(
        data["x_train"], data["y_train"], seed, model_path, "Ordinary baseline"
    )
    fit = data["fit_indices"]
    policy_indices = data["policy_indices"]
    calibrator, policy = fit_platt_and_policy(
        model,
        data["x_calibration"][fit],
        data["y_calibration"][fit],
        data["x_calibration"][policy_indices],
        data["y_calibration"][policy_indices],
        seed,
    )
    atomic_joblib_dump(calibrator, required[1])
    atomic_json_dump(policy, required[2])
    validation = calibrated_validation_metrics(
        model,
        calibrator,
        policy,
        data["x_validation"],
        data["y_validation"],
    )
    manifest = {
        "seed": seed,
        "status": "complete",
        "model": "ordinary_random_forest",
        "trees": len(model.estimators_),
        "training_rows": int(len(data["y_train"])),
        "training_features": int(data["x_train"].shape[1]),
        "validation": validation,
        "elapsed_seconds": time.time() - started,
        "target_loaded": False,
        "files": {path.name: sha256(path) for path in required},
    }
    atomic_json_dump(manifest, completion_path)
    print("Ordinary baseline training complete.")
    del model, calibrator
    gc.collect()
    return manifest


def train_group_aware(seed: int, seed_dir: Path, data: dict) -> dict:
    completion_path = seed_dir / "GROUP_AWARE_TRAINING_COMPLETE.json"
    required = [
        seed_dir / "group_aware_random_forest.joblib",
        seed_dir / "group_aware_platt.joblib",
        seed_dir / "group_aware_policy.json",
    ]
    if completion_path.exists() and all(path.exists() for path in required):
        print("Group-aware model: COMPLETE; skipping.")
        return load_json(completion_path)

    started = time.time()
    print("Creating group-aware training data...")
    x_augmented, y_augmented, assignment_counts = make_group_training_data(
        data["x_train"],
        data["y_train"],
        data["medians"],
        data["group_names"],
        data["group_indices"],
        seed,
    )
    print("Group-aware training shape:", x_augmented.shape)
    print("Mask assignment counts:", assignment_counts)
    model_path = required[0]
    model = train_forest_restartable(
        x_augmented,
        y_augmented,
        seed,
        model_path,
        "Group-aware model",
    )
    del x_augmented, y_augmented
    gc.collect()

    fit = data["fit_indices"]
    policy_indices = data["policy_indices"]
    x_fit, y_fit = expand_complete_and_single_masks(
        data["x_calibration"][fit],
        data["y_calibration"][fit],
        data["medians"],
        data["group_names"],
        data["group_indices"],
    )
    x_policy, y_policy = expand_complete_and_single_masks(
        data["x_calibration"][policy_indices],
        data["y_calibration"][policy_indices],
        data["medians"],
        data["group_names"],
        data["group_indices"],
    )
    calibrator, policy = fit_platt_and_policy(
        model, x_fit, y_fit, x_policy, y_policy, seed
    )
    del x_fit, y_fit, x_policy, y_policy
    gc.collect()
    atomic_joblib_dump(calibrator, required[1])
    atomic_json_dump(policy, required[2])

    x_validation, y_validation = expand_complete_and_single_masks(
        data["x_validation"],
        data["y_validation"],
        data["medians"],
        data["group_names"],
        data["group_indices"],
    )
    validation = calibrated_validation_metrics(
        model, calibrator, policy, x_validation, y_validation
    )
    del x_validation, y_validation
    gc.collect()
    manifest = {
        "seed": seed,
        "status": "complete",
        "model": "group_aware_random_forest",
        "trees": len(model.estimators_),
        "training_rows": int(len(data["y_train"]) * 2),
        "training_features": int(data["x_train"].shape[1] + 4),
        "mask_assignment_counts": assignment_counts,
        "validation_complete_plus_single_masks": validation,
        "elapsed_seconds": time.time() - started,
        "target_loaded": False,
        "files": {path.name: sha256(path) for path in required},
    }
    atomic_json_dump(manifest, completion_path)
    print("Group-aware training complete.")
    del model, calibrator
    gc.collect()
    return manifest


def train_seed(seed: int, data: dict) -> None:
    seed_dir = RF_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    final_marker = seed_dir / "TRAINING_COMPLETE.json"
    if final_marker.exists():
        marker = load_json(final_marker)
        if marker.get("status") == "training_complete":
            print(f"Seed {seed}: TRAINING COMPLETE; skipping.")
            return

    print("\n" + "=" * 72)
    print("RANDOM FOREST CONFIRMATORY SEED:", seed)
    print("=" * 72)
    started = time.time()
    baseline = train_baseline(seed, seed_dir, data)
    group = train_group_aware(seed, seed_dir, data)
    final = {
        "seed": seed,
        "status": "training_complete",
        "ordinary_baseline": baseline,
        "group_aware": group,
        "elapsed_seconds_this_invocation": time.time() - started,
        "target_loaded": False,
    }
    atomic_json_dump(final, final_marker)
    print(f"Seed {seed} training and source-only policy creation complete.")


def verify_protocol() -> dict:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(
            "Frozen Random Forest protocol not found. Run "
            "prepare_random_forest_replication.py first."
        )
    protocol = load_json(PROTOCOL_PATH)
    if protocol.get("status") != "FROZEN_BEFORE_TRAINING":
        raise RuntimeError("Random Forest protocol is not frozen.")
    if protocol.get("confirmatory_seeds") != SEEDS:
        raise RuntimeError("Seed list differs from the frozen protocol.")
    configuration = protocol.get("fixed_model_configuration", {})
    if configuration.get("n_estimators") != 300:
        raise RuntimeError("The frozen tree count is not 300.")
    if configuration.get("max_depth") != 20:
        raise RuntimeError("The frozen maximum depth is not 20.")
    return protocol


def main() -> None:
    mount_drive()
    RF_DIR.mkdir(parents=True, exist_ok=True)
    verify_protocol()
    data = load_source_training_data()
    for seed in SEEDS:
        train_seed(seed, data)
    print("\nAll five Random Forest seeds have completed training.")
    print("RANDOM_FOREST_FIVE_SEED_TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
