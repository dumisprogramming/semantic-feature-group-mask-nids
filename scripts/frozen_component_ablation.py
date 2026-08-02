"""Corrected frozen, restartable component ablation for the NIDS study.

Run in Google Colab after uploading this file:

    %run /content/frozen_component_ablation.py

This corrected script trains only the two missing ablation variants:

1. indicators_only: no masked-copy augmentation; four availability indicators.
2. augmentation_only: single-group masked-copy augmentation; no indicators.

It reuses the already completed ordinary baseline and full group-aware results.
The script is restartable: models, calibrators, policies, and evaluations are
saved separately for every seed. Completed work is skipped after interruption.

It uses the exact XGBoost hyperparameters recorded in the original frozen
five-seed protocol. Earlier exploratory ablation outputs are not overwritten.
The target dataset is not loaded until all ablation models, calibrators,
thresholds, and selective policies are frozen using source data only.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
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


CHECKPOINT_DIR = Path(
    os.environ.get(
        "NIDS_CHECKPOINT_DIR",
        "/content/drive/MyDrive/research/researchdata/"
        "NIDS_Research/NIDS_Research_Checkpoints",
    )
)
REPLICATION_DIR = CHECKPOINT_DIR / "five_seed_confirmatory_replication"
ABLATION_DIR = REPLICATION_DIR / "component_ablation_corrected"

SEEDS = [2027, 2028, 2029, 2030, 2031]
VARIANTS = ["indicators_only", "augmentation_only"]
DESIRED_COVERAGES = [0.80, 0.90, 0.95]
CALIBRATOR_FIT_ROWS = 48_889
POLICY_SELECTION_ROWS = 50_195
CALIBRATION_SPLIT_SEED = 2026
N_ESTIMATORS = 1_200
EARLY_STOPPING_ROUNDS = 50
T_CRITICAL_DF4_95 = 2.7764451051977987


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError:
        return
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    temporary.replace(destination)


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def protocol_payload() -> dict:
    return {
        "study": "five-seed component ablation",
        "status": "corrected and frozen before ablation retraining",
        "correction_reason": (
            "The exploratory ablation reconstructed default booster settings "
            "from loaded models. This corrected run uses the exact parameters "
            "stored in replication_protocol_frozen.json."
        ),
        "exploratory_results_retained_at": "component_ablation",
        "confirmatory_seeds": SEEDS,
        "source_dataset": "CICIDS2017",
        "target_dataset": "CSE-CIC-IDS2018",
        "task": "binary attack detection",
        "variants": {
            "ordinary_baseline": {
                "masked_copy_augmentation": False,
                "availability_indicators": False,
                "action": "reuse completed confirmatory results",
            },
            "indicators_only": {
                "masked_copy_augmentation": False,
                "availability_indicators": True,
                "training_conditions": ["complete only"],
            },
            "augmentation_only": {
                "masked_copy_augmentation": True,
                "availability_indicators": False,
                "training_conditions": [
                    "complete",
                    "one randomly assigned single-group-loss copy per row",
                ],
            },
            "full_method": {
                "masked_copy_augmentation": True,
                "availability_indicators": True,
                "action": "reuse completed group-aware confirmatory results",
            },
        },
        "semantic_groups": 4,
        "pairwise_loss": {
            "used_for_training": False,
            "used_for_early_stopping": False,
            "used_for_calibration": False,
            "used_for_threshold_selection": False,
            "used_for_selective_policy": False,
            "used_for_evaluation": True,
        },
        "calibration": {
            "method": "Platt scaling on source-only model logits",
            "split_seed": CALIBRATION_SPLIT_SEED,
            "calibrator_fit_rows": CALIBRATOR_FIT_ROWS,
            "policy_selection_rows": POLICY_SELECTION_ROWS,
            "stratified": True,
        },
        "classification_threshold": "maximise source policy-partition F1",
        "selective_confidence": "max(p_attack, 1-p_attack)",
        "desired_coverages": DESIRED_COVERAGES,
        "missing_group_replacement": "source-training median",
        "target_usage": "evaluation only after all source-only policies freeze",
        "primary_condition": "average across six unseen pairwise losses per seed",
        "primary_metric": "attack-class F1",
        "xgboost": {
            "parameter_source": (
                "replication_protocol_frozen.json/fixed_model_configuration"
            ),
            "objective": "binary:logistic",
            "n_estimators": N_ESTIMATORS,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "evaluation_metric": "logloss",
            "training_device": "cpu",
        },
    }


def freeze_protocol() -> None:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    protocol_path = ABLATION_DIR / "component_ablation_protocol_frozen.json"
    proposed = protocol_payload()
    if protocol_path.exists():
        with open(protocol_path) as file:
            existing = json.load(file)
        if existing != proposed:
            raise RuntimeError(
                "The existing frozen ablation protocol differs from this "
                "script. Do not overwrite it; inspect the discrepancy."
            )
        print("Frozen component-ablation protocol verified:", protocol_path)
    else:
        atomic_json(proposed, protocol_path)
        print("Frozen component-ablation protocol saved:", protocol_path)


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


def evaluate_probability(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict:
    prediction = (probability >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()
    return {
        "rows": int(len(y_true)),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "mcc": matthews_corrcoef(y_true, prediction),
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
        "brier": brier_score_loss(y_true, probability),
        "logloss": log_loss(y_true, probability),
        "ece": expected_calibration_error(y_true, probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


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
                "selective_risk": float(
                    np.mean(accepted_prediction != accepted_y)
                ),
                "accepted_accuracy": accuracy_score(
                    accepted_y, accepted_prediction
                ),
                "accepted_precision": precision_score(
                    accepted_y, accepted_prediction, zero_division=0
                ),
                "accepted_recall": recall_score(
                    accepted_y, accepted_prediction, zero_division=0
                ),
                "accepted_f1": f1_score(
                    accepted_y, accepted_prediction, zero_division=0
                ),
            }
        )
    return result


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


def load_metadata_and_imputer() -> tuple[dict, object]:
    required = [
        CHECKPOINT_DIR / "experiment_metadata.json",
        CHECKPOINT_DIR / "source_median_imputer.joblib",
        REPLICATION_DIR / "replication_protocol_frozen.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required frozen files: {missing}")
    with open(CHECKPOINT_DIR / "experiment_metadata.json") as file:
        metadata = json.load(file)
    imputer = joblib.load(CHECKPOINT_DIR / "source_median_imputer.joblib")
    return metadata, imputer


def feature_definition(metadata: dict, imputer: object) -> dict:
    predictors = metadata["predictors"]
    semantic_groups = metadata["semantic_groups"]
    group_names = list(semantic_groups)
    feature_index = {name: index for index, name in enumerate(predictors)}
    group_indices = {
        group: [feature_index[name] for name in semantic_groups[group]]
        for group in group_names
    }
    medians = np.asarray(imputer.statistics_, dtype=np.float32)
    if len(predictors) != 77 or len(group_names) != 4:
        raise ValueError(
            f"Expected 77 predictors and 4 groups; received "
            f"{len(predictors)} and {len(group_names)}."
        )
    return {
        "predictors": predictors,
        "group_names": group_names,
        "group_indices": group_indices,
        "medians": medians,
    }


def load_source_development_data(
    metadata: dict, imputer: object, definition: dict
) -> dict:
    files = {
        "train": CHECKPOINT_DIR / "source_train.parquet",
        "validation": CHECKPOINT_DIR / "source_validation.parquet",
        "calibration": CHECKPOINT_DIR / "source_calibration.parquet",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source development files: {missing}")

    output = {}
    for name, path in files.items():
        frame = pd.read_parquet(path)
        x_data = imputer.transform(
            frame[definition["predictors"]]
        ).astype("float32")
        y_data = frame["target"].to_numpy(dtype="int8")
        if not np.isfinite(x_data).all():
            raise ValueError(f"Non-finite values remain in source {name} data.")
        output[name] = (x_data, y_data)
        del frame

    x_calibration, y_calibration = output["calibration"]
    expected = CALIBRATOR_FIT_ROWS + POLICY_SELECTION_ROWS
    if len(y_calibration) != expected:
        raise ValueError(
            f"Expected {expected} calibration rows; found "
            f"{len(y_calibration)}."
        )
    all_indices = np.arange(len(y_calibration))
    fit_indices, policy_indices = train_test_split(
        all_indices,
        train_size=CALIBRATOR_FIT_ROWS,
        test_size=POLICY_SELECTION_ROWS,
        stratify=y_calibration,
        random_state=CALIBRATION_SPLIT_SEED,
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
    del output["calibration"], x_calibration, y_calibration

    print("\nSource-only development data")
    print("Training:", output["train"][0].shape)
    print("Validation:", output["validation"][0].shape)
    print("Calibrator fit:", output["calibrator"][0].shape)
    print("Policy selection:", output["policy"][0].shape)
    return output


def load_test_data(imputer: object, definition: dict) -> dict:
    files = {
        "source": CHECKPOINT_DIR / "source_test.parquet",
        "target": CHECKPOINT_DIR / "target_test_sealed.parquet",
    }
    output = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        x_data = imputer.transform(
            frame[definition["predictors"]]
        ).astype("float32")
        y_data = frame["target"].to_numpy(dtype="int8")
        if not np.isfinite(x_data).all():
            raise ValueError(f"Non-finite values remain in {name} test data.")
        output[name] = (x_data, y_data)
        del frame
        print(
            f"{name.title()} test: {x_data.shape} | "
            f"attack rate: {float(y_data.mean()):.6f}"
        )
    return output


def replace_groups(
    x_data: np.ndarray,
    missing_groups: tuple[str, ...],
    definition: dict,
) -> np.ndarray:
    output = np.asarray(x_data, dtype=np.float32).copy()
    for group in missing_groups:
        columns = definition["group_indices"][group]
        output[:, columns] = definition["medians"][columns]
    return output


def transform_condition(
    x_data: np.ndarray,
    missing_groups: tuple[str, ...],
    variant: str,
    definition: dict,
) -> np.ndarray:
    output = replace_groups(x_data, missing_groups, definition)
    if variant == "augmentation_only":
        return output
    if variant == "indicators_only":
        availability = np.ones(
            (len(output), len(definition["group_names"])), dtype=np.float32
        )
        for group in missing_groups:
            position = definition["group_names"].index(group)
            availability[:, position] = 0.0
        return np.hstack([output, availability])
    raise ValueError(f"Unknown ablation variant: {variant}")


def build_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    variant: str,
    definition: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if variant == "indicators_only":
        availability = np.ones(
            (len(x_train), len(definition["group_names"])), dtype=np.float32
        )
        return (
            np.hstack([x_train, availability]),
            y_train.copy(),
            {"complete": int(len(y_train))},
        )

    if variant != "augmentation_only":
        raise ValueError(variant)
    rng = np.random.default_rng(seed)
    assignments = rng.integers(
        0, len(definition["group_names"]), size=len(x_train)
    )
    masked = x_train.copy()
    counts = {}
    for group_number, group in enumerate(definition["group_names"]):
        selected = assignments == group_number
        columns = definition["group_indices"][group]
        masked[np.ix_(selected, columns)] = definition["medians"][columns]
        counts[group] = int(selected.sum())
    return (
        np.vstack([x_train, masked]),
        np.concatenate([y_train, y_train]),
        counts,
    )


def build_known_condition_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    variant: str,
    definition: dict,
) -> tuple[np.ndarray, np.ndarray]:
    if variant == "indicators_only":
        return (
            transform_condition(x_data, tuple(), variant, definition),
            y_data.copy(),
        )
    conditions = [tuple()] + [
        (group,) for group in definition["group_names"]
    ]
    x_parts = [
        transform_condition(x_data, condition, variant, definition)
        for condition in conditions
    ]
    return np.vstack(x_parts), np.tile(y_data, len(conditions))


def recursively_find(mapping, key: str):
    if isinstance(mapping, dict):
        if key in mapping:
            return mapping[key]
        for value in mapping.values():
            result = recursively_find(value, key)
            if result is not None:
                return result
    elif isinstance(mapping, list):
        for value in mapping:
            result = recursively_find(value, key)
            if result is not None:
                return result
    return None


def reference_xgb_parameters(seed: int) -> dict:
    protocol_path = REPLICATION_DIR / "replication_protocol_frozen.json"
    with open(protocol_path) as file:
        frozen_protocol = json.load(file)
    frozen = frozen_protocol["fixed_model_configuration"]
    expected = {
        "objective": "binary:logistic",
        "n_estimators": 1200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "device": "cpu",
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
        "n_jobs": -1,
    }
    if frozen != expected:
        raise RuntimeError(
            "The frozen model configuration does not match the corrected "
            f"ablation specification. Found: {frozen}"
        )
    return {
        "objective": frozen["objective"],
        "n_estimators": N_ESTIMATORS,
        "learning_rate": frozen["learning_rate"],
        "max_depth": frozen["max_depth"],
        "min_child_weight": frozen["min_child_weight"],
        "subsample": frozen["subsample"],
        "colsample_bytree": frozen["colsample_bytree"],
        "reg_alpha": frozen["reg_alpha"],
        "reg_lambda": frozen["reg_lambda"],
        "tree_method": frozen["tree_method"],
        "device": frozen["device"],
        "eval_metric": frozen["eval_metric"],
        "random_state": seed,
        "n_jobs": frozen["n_jobs"],
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }


def select_f1_threshold(
    y_true: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        y_true, probability
    )
    if len(thresholds) == 0:
        return 0.5, 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.nanargmax(f1_values))
    return float(thresholds[best]), float(f1_values[best])


def confidence_cutoffs(probability: np.ndarray) -> dict[str, float]:
    confidence = np.maximum(probability, 1 - probability)
    return {
        str(coverage): float(
            np.quantile(confidence, 1 - coverage, method="lower")
        )
        for coverage in DESIRED_COVERAGES
    }


def predict_calibrated(
    model: xgb.XGBClassifier,
    calibrator: LogisticRegression,
    x_data: np.ndarray,
) -> np.ndarray:
    raw_probability = model.predict_proba(x_data)[:, 1]
    return calibrator.predict_proba(
        probability_to_logit(raw_probability)
    )[:, 1]


def train_variant(
    seed: int,
    variant: str,
    source: dict,
    definition: dict,
) -> None:
    output_dir = ABLATION_DIR / f"seed_{seed}" / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "TRAINING_COMPLETE.json"
    model_path = output_dir / f"{variant}_xgboost.json"
    calibrator_path = output_dir / f"{variant}_platt.joblib"
    policy_path = output_dir / f"{variant}_policy.json"

    if all(
        path.exists()
        for path in [completion_path, model_path, calibrator_path, policy_path]
    ):
        print(f"Seed {seed} | {variant}: training already complete; skipping.")
        return

    started = time.time()
    x_train, y_train = source["train"]
    x_validation, y_validation = source["validation"]

    if model_path.exists():
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        training_seconds = None
        mask_counts = "restored existing model"
        print(f"Seed {seed} | {variant}: restored saved model.")
    else:
        x_train_variant, y_train_variant, mask_counts = build_training_data(
            x_train, y_train, seed, variant, definition
        )
        x_validation_variant, y_validation_variant = (
            build_known_condition_data(
                x_validation, y_validation, variant, definition
            )
        )
        print("\n" + "=" * 72)
        print(f"TRAINING SEED {seed} | {variant}")
        print("=" * 72)
        print("Training shape:", x_train_variant.shape)
        print("Validation shape:", x_validation_variant.shape)
        print("Mask counts:", mask_counts)
        model = xgb.XGBClassifier(**reference_xgb_parameters(seed))
        training_started = time.time()
        model.fit(
            x_train_variant,
            y_train_variant,
            eval_set=[(x_validation_variant, y_validation_variant)],
            verbose=50,
        )
        training_seconds = time.time() - training_started
        model.save_model(str(model_path))
        atomic_json(
            {
                "seed": seed,
                "variant": variant,
                "training_seconds": training_seconds,
                "best_iteration": int(model.best_iteration),
                "best_score": float(model.best_score),
                "training_rows": int(len(y_train_variant)),
                "validation_rows": int(len(y_validation_variant)),
                "mask_counts": mask_counts,
                "xgboost_parameters": reference_xgb_parameters(seed),
            },
            output_dir / "MODEL_TRAINED.json",
        )
        del (
            x_train_variant,
            y_train_variant,
            x_validation_variant,
            y_validation_variant,
        )
        gc.collect()
        print(
            f"Seed {seed} | {variant}: model saved after "
            f"{training_seconds:.1f} seconds."
        )

    x_fit, y_fit = source["calibrator"]
    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
        print(f"Seed {seed} | {variant}: restored saved calibrator.")
    else:
        x_fit_variant, y_fit_variant = build_known_condition_data(
            x_fit, y_fit, variant, definition
        )
        raw_fit = model.predict_proba(x_fit_variant)[:, 1]
        calibrator = LogisticRegression(
            solver="lbfgs", max_iter=1_000, random_state=seed
        )
        calibrator.fit(probability_to_logit(raw_fit), y_fit_variant)
        joblib.dump(calibrator, calibrator_path)
        del x_fit_variant, y_fit_variant, raw_fit
        gc.collect()
        print(f"Seed {seed} | {variant}: source-only calibrator saved.")

    if policy_path.exists():
        print(f"Seed {seed} | {variant}: restored saved policy.")
    else:
        x_policy, y_policy = source["policy"]
        x_policy_variant, y_policy_variant = build_known_condition_data(
            x_policy, y_policy, variant, definition
        )
        policy_probability = predict_calibrated(
            model, calibrator, x_policy_variant
        )
        threshold, selected_f1 = select_f1_threshold(
            y_policy_variant, policy_probability
        )
        cutoffs = confidence_cutoffs(policy_probability)
        policy = {
            "seed": seed,
            "variant": variant,
            "classification_threshold": threshold,
            "source_policy_f1": selected_f1,
            "threshold_metric": "F1",
            "threshold_partition": "source policy partition",
            "calibration_method": "Platt scaling",
            "calibration_training_conditions": (
                ["complete"]
                if variant == "indicators_only"
                else ["complete", "each single-group loss"]
            ),
            "policy_selection_conditions": (
                ["complete"]
                if variant == "indicators_only"
                else ["complete", "each single-group loss"]
            ),
            "confidence_definition": "max(p_attack, 1-p_attack)",
            "confidence_cutoffs": cutoffs,
            "pairwise_loss_used": False,
            "target_data_used": False,
        }
        atomic_json(policy, policy_path)
        del x_policy_variant, y_policy_variant, policy_probability
        gc.collect()
        print(
            f"Seed {seed} | {variant}: source-only policy saved; "
            f"threshold={threshold:.6f}."
        )

    elapsed = time.time() - started
    with open(policy_path) as file:
        policy = json.load(file)
    marker = {
        "seed": seed,
        "variant": variant,
        "status": "training_calibration_and_policy_complete",
        "elapsed_seconds_this_run": elapsed,
        "model_file": model_path.name,
        "calibrator_file": calibrator_path.name,
        "policy_file": policy_path.name,
        "classification_threshold": policy["classification_threshold"],
        "pairwise_loss_used": False,
        "target_data_used": False,
    }
    atomic_json(marker, completion_path)
    del model, calibrator
    gc.collect()
    print(f"Seed {seed} | {variant}: TRAINING COMPLETE.")


def verify_all_training_complete() -> None:
    missing = []
    for seed in SEEDS:
        for variant in VARIANTS:
            output_dir = ABLATION_DIR / f"seed_{seed}" / variant
            required = [
                output_dir / "TRAINING_COMPLETE.json",
                output_dir / f"{variant}_xgboost.json",
                output_dir / f"{variant}_platt.joblib",
                output_dir / f"{variant}_policy.json",
            ]
            if not all(path.exists() for path in required):
                missing.append(f"{seed}/{variant}")
    if missing:
        raise RuntimeError(f"Incomplete ablation training: {missing}")
    atomic_json(
        {
            "status": "all source-only ablation policies frozen",
            "seeds": SEEDS,
            "variants": VARIANTS,
            "target_loaded": False,
        },
        ABLATION_DIR / "ALL_SOURCE_POLICIES_FROZEN.json",
    )
    print("\nAll ten ablation models and source-only policies are frozen.")


def evaluation_conditions(group_names: list[str]) -> list[tuple]:
    conditions = [("complete", tuple())]
    conditions.extend(("single", (group,)) for group in group_names)
    conditions.extend(
        ("pairwise_unseen", pair) for pair in combinations(group_names, 2)
    )
    return conditions


def evaluate_variant(
    seed: int,
    variant: str,
    tests: dict,
    definition: dict,
) -> None:
    output_dir = ABLATION_DIR / f"seed_{seed}" / variant
    completion_path = output_dir / "EVALUATION_COMPLETE.json"
    nonselective_path = output_dir / "ablation_nonselective_results.csv"
    selective_path = output_dir / "ablation_selective_results.csv"
    if all(
        path.exists()
        for path in [completion_path, nonselective_path, selective_path]
    ):
        print(f"Seed {seed} | {variant}: evaluation complete; skipping.")
        return

    model = xgb.XGBClassifier()
    model.load_model(str(output_dir / f"{variant}_xgboost.json"))
    calibrator = joblib.load(output_dir / f"{variant}_platt.joblib")
    with open(output_dir / f"{variant}_policy.json") as file:
        policy = json.load(file)
    threshold = float(policy["classification_threshold"])
    cutoffs = {
        float(key): float(value)
        for key, value in policy["confidence_cutoffs"].items()
    }

    nonselective_rows = []
    selective_rows = []
    started = time.time()
    print("\n" + "=" * 72)
    print(f"EVALUATING SEED {seed} | {variant}")
    print("=" * 72)

    for dataset_name in ["source", "target"]:
        x_data, y_data = tests[dataset_name]
        for condition_type, missing_groups in evaluation_conditions(
            definition["group_names"]
        ):
            condition_started = time.time()
            condition_name = (
                "+".join(missing_groups) if missing_groups else "None"
            )
            x_condition = transform_condition(
                x_data, missing_groups, variant, definition
            )
            probability = predict_calibrated(
                model, calibrator, x_condition
            )
            metrics = evaluate_probability(y_data, probability, threshold)
            metrics.update(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "model": variant,
                    "condition_type": condition_type,
                    "missing_groups": condition_name,
                }
            )
            nonselective_rows.append(metrics)
            aurc = calculate_aurc(y_data, probability, threshold)
            for desired_coverage, cutoff in cutoffs.items():
                selective = evaluate_selective(
                    y_data, probability, threshold, cutoff
                )
                selective.update(
                    {
                        "seed": seed,
                        "dataset": dataset_name,
                        "model": variant,
                        "condition_type": condition_type,
                        "missing_groups": condition_name,
                        "desired_policy_coverage": desired_coverage,
                        "confidence_cutoff": cutoff,
                        "aurc": aurc,
                    }
                )
                selective_rows.append(selective)
            del x_condition, probability
            gc.collect()
            print(
                dataset_name,
                "|",
                condition_type,
                "|",
                condition_name,
                "|",
                round(time.time() - condition_started, 1),
                "seconds",
            )

    atomic_csv(pd.DataFrame(nonselective_rows), nonselective_path)
    atomic_csv(pd.DataFrame(selective_rows), selective_path)
    elapsed = time.time() - started
    atomic_json(
        {
            "seed": seed,
            "variant": variant,
            "status": "evaluation_complete",
            "elapsed_seconds": elapsed,
            "nonselective_rows": len(nonselective_rows),
            "selective_rows": len(selective_rows),
            "target_used_for": "evaluation only",
        },
        completion_path,
    )
    del model, calibrator
    gc.collect()
    print(
        f"Seed {seed} | {variant}: evaluation completed in "
        f"{elapsed:.1f} seconds."
    )


def load_existing_confirmatory_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    nonselective_path = REPLICATION_DIR / "five_seed_nonselective_all.csv"
    selective_path = REPLICATION_DIR / "five_seed_selective_all.csv"
    if nonselective_path.exists() and selective_path.exists():
        return pd.read_csv(nonselective_path), pd.read_csv(selective_path)

    nonselective_parts = []
    selective_parts = []
    for seed in SEEDS:
        seed_dir = REPLICATION_DIR / f"seed_{seed}"
        nonselective_parts.append(
            pd.read_csv(seed_dir / "confirmatory_nonselective_results.csv")
        )
        selective_parts.append(
            pd.read_csv(seed_dir / "confirmatory_selective_results.csv")
        )
    return (
        pd.concat(nonselective_parts, ignore_index=True),
        pd.concat(selective_parts, ignore_index=True),
    )


def summarize_primary(
    nonselective: pd.DataFrame, selective: pd.DataFrame
) -> None:
    pairwise = nonselective[
        nonselective["condition_type"] == "pairwise_unseen"
    ].copy()
    per_seed = (
        pairwise.groupby(["seed", "dataset", "model"], as_index=False)[
            [
                "f1",
                "precision",
                "recall",
                "false_positive_rate",
                "ece",
                "brier",
            ]
        ]
        .mean()
    )
    atomic_csv(
        per_seed, ABLATION_DIR / "component_ablation_pairwise_seed_means.csv"
    )

    summary_rows = []
    metrics = [
        "f1",
        "precision",
        "recall",
        "false_positive_rate",
        "ece",
        "brier",
    ]
    for (dataset, model), group in per_seed.groupby(["dataset", "model"]):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            half_width = T_CRITICAL_DF4_95 * sd / math.sqrt(len(values))
            summary_rows.append(
                {
                    "analysis": "nonselective pairwise average",
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "seeds": len(values),
                    "mean": mean,
                    "sd": sd,
                    "ci95_lower": mean - half_width,
                    "ci95_upper": mean + half_width,
                }
            )

    selective_pairwise = selective[
        (selective["condition_type"] == "pairwise_unseen")
        & np.isclose(selective["desired_policy_coverage"], 0.90)
    ].copy()
    selective_per_seed = (
        selective_pairwise.groupby(
            ["seed", "dataset", "model"], as_index=False
        )[
            [
                "actual_coverage",
                "attack_coverage",
                "selective_risk",
                "accepted_f1",
                "aurc",
            ]
        ]
        .mean()
    )
    atomic_csv(
        selective_per_seed,
        ABLATION_DIR / "component_ablation_selective90_seed_means.csv",
    )
    for (dataset, model), group in selective_per_seed.groupby(
        ["dataset", "model"]
    ):
        for metric in [
            "actual_coverage",
            "attack_coverage",
            "selective_risk",
            "accepted_f1",
            "aurc",
        ]:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.nanmean(values))
            sd = float(np.nanstd(values, ddof=1))
            valid_n = int(np.isfinite(values).sum())
            half_width = (
                T_CRITICAL_DF4_95 * sd / math.sqrt(valid_n)
                if valid_n == 5
                else np.nan
            )
            summary_rows.append(
                {
                    "analysis": "selective pairwise average at nominal 90%",
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "seeds": valid_n,
                    "mean": mean,
                    "sd": sd,
                    "ci95_lower": mean - half_width,
                    "ci95_upper": mean + half_width,
                }
            )

    atomic_csv(
        pd.DataFrame(summary_rows),
        ABLATION_DIR / "component_ablation_primary_summary.csv",
    )


def combine_all_results() -> None:
    existing_nonselective, existing_selective = (
        load_existing_confirmatory_results()
    )
    existing_nonselective = existing_nonselective[
        existing_nonselective["model"].isin(
            ["ordinary_baseline", "group_aware"]
        )
    ].copy()
    existing_selective = existing_selective[
        existing_selective["model"].isin(
            ["ordinary_baseline", "group_aware"]
        )
    ].copy()

    new_nonselective = []
    new_selective = []
    missing = []
    for seed in SEEDS:
        for variant in VARIANTS:
            output_dir = ABLATION_DIR / f"seed_{seed}" / variant
            nonselective_path = output_dir / "ablation_nonselective_results.csv"
            selective_path = output_dir / "ablation_selective_results.csv"
            if not nonselective_path.exists() or not selective_path.exists():
                missing.append(f"{seed}/{variant}")
                continue
            new_nonselective.append(pd.read_csv(nonselective_path))
            new_selective.append(pd.read_csv(selective_path))
    if missing:
        raise RuntimeError(f"Missing component evaluations: {missing}")

    all_nonselective = pd.concat(
        [existing_nonselective] + new_nonselective, ignore_index=True
    )
    all_selective = pd.concat(
        [existing_selective] + new_selective, ignore_index=True
    )
    all_nonselective["model"] = all_nonselective["model"].replace(
        {"group_aware": "full_method"}
    )
    all_selective["model"] = all_selective["model"].replace(
        {"group_aware": "full_method"}
    )

    expected_nonselective = 5 * 4 * 2 * 11
    expected_selective = expected_nonselective * 3
    if len(all_nonselective) != expected_nonselective:
        raise ValueError(
            f"Expected {expected_nonselective} non-selective rows; found "
            f"{len(all_nonselective)}."
        )
    if len(all_selective) != expected_selective:
        raise ValueError(
            f"Expected {expected_selective} selective rows; found "
            f"{len(all_selective)}."
        )

    atomic_csv(
        all_nonselective,
        ABLATION_DIR / "component_ablation_nonselective_all.csv",
    )
    atomic_csv(
        all_selective,
        ABLATION_DIR / "component_ablation_selective_all.csv",
    )
    summarize_primary(all_nonselective, all_selective)
    atomic_json(
        {
            "status": "component_ablation_complete",
            "seeds": SEEDS,
            "models": [
                "ordinary_baseline",
                "indicators_only",
                "augmentation_only",
                "full_method",
            ],
            "nonselective_rows": len(all_nonselective),
            "selective_rows": len(all_selective),
            "primary_summary": "component_ablation_primary_summary.csv",
        },
        ABLATION_DIR / "COMPONENT_ABLATION_COMPLETE.json",
    )
    print("\nCombined component-ablation outputs saved:")
    print("Non-selective rows:", len(all_nonselective))
    print("Selective rows:", len(all_selective))


def main() -> None:
    started = time.time()
    mount_drive()
    metadata, imputer = load_metadata_and_imputer()
    definition = feature_definition(metadata, imputer)
    freeze_protocol()

    # Source development data only. Target data is intentionally not loaded.
    source = load_source_development_data(metadata, imputer, definition)
    for seed in SEEDS:
        for variant in VARIANTS:
            train_variant(seed, variant, source, definition)
    verify_all_training_complete()

    # Release training arrays before target evaluation.
    del source
    gc.collect()

    # Target labels become accessible only after all source policies are frozen.
    tests = load_test_data(imputer, definition)
    for seed in SEEDS:
        for variant in VARIANTS:
            evaluate_variant(seed, variant, tests, definition)
    combine_all_results()
    elapsed = time.time() - started
    print(
        "\nFIVE-SEED COMPONENT ABLATION COMPLETE in "
        f"{elapsed / 60:.1f} minutes."
    )
    print(
        "Please provide component_ablation_primary_summary.csv and "
        "component_ablation_pairwise_seed_means.csv for interpretation."
    )


if __name__ == "__main__":
    main()
