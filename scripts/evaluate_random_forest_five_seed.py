"""Restartable five-seed Random Forest evaluation for the NIDS study.

Run in Google Colab with:
    %run "/content/drive/MyDrive/research/researchdata/NIDS_Research/evaluate_random_forest_five_seed.py"

The script evaluates the frozen ordinary and group-aware Random Forest models
on source and sealed target data. Results are saved after every condition, so a
disconnection resumes after the last completed condition. No model is trained,
calibrated, tuned, or selected in this script.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
EXPECTED_SOURCE_ROWS = 150178
EXPECTED_TARGET_ROWS = 999143
PREDICTION_BATCH_SIZE = 100000


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


def atomic_csv_dump(rows: list[dict], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
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
    y_true = np.asarray(y_true)
    prediction = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "precision": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
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


def evaluate_selective(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    confidence_cutoff: float,
) -> dict:
    y_true = np.asarray(y_true)
    prediction = (probability >= threshold).astype(np.int8)
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
                "accepted_accuracy": float(
                    accuracy_score(accepted_y, accepted_prediction)
                ),
                "accepted_precision": float(
                    precision_score(
                        accepted_y, accepted_prediction, zero_division=0
                    )
                ),
                "accepted_recall": float(
                    recall_score(
                        accepted_y, accepted_prediction, zero_division=0
                    )
                ),
                "accepted_f1": float(
                    f1_score(
                        accepted_y, accepted_prediction, zero_division=0
                    )
                ),
            }
        )
    return result


def calculate_aurc(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> float:
    prediction = (probability >= threshold).astype(np.int8)
    confidence = np.maximum(probability, 1 - probability)
    errors = (prediction != y_true).astype(np.float64)
    order = np.argsort(-confidence, kind="stable")
    cumulative_errors = np.cumsum(errors[order])
    accepted_count = np.arange(1, len(y_true) + 1)
    return float(np.mean(cumulative_errors / accepted_count))


def verify_protocol() -> dict:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(
            "Frozen Random Forest protocol not found. Run the preparation "
            "script before evaluation."
        )
    protocol = load_json(PROTOCOL_PATH)
    if protocol.get("status") != "FROZEN_BEFORE_TRAINING":
        raise RuntimeError("The Random Forest protocol is not frozen.")
    if protocol.get("confirmatory_seeds") != SEEDS:
        raise RuntimeError("The seed list differs from the frozen protocol.")
    configuration = protocol.get("fixed_model_configuration", {})
    if configuration.get("n_estimators") != 300:
        raise RuntimeError("The frozen Random Forest tree count is not 300.")
    return protocol


def load_fixed_test_data() -> dict:
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

    source_df = pd.read_parquet(CHECKPOINT_DIR / "source_test.parquet")
    target_df = pd.read_parquet(CHECKPOINT_DIR / "target_test_sealed.parquet")
    x_source = imputer.transform(source_df[predictors]).astype(np.float32)
    y_source = source_df["target"].to_numpy(dtype=np.int8)
    x_target = imputer.transform(target_df[predictors]).astype(np.float32)
    y_target = target_df["target"].to_numpy(dtype=np.int8)
    del source_df, target_df
    gc.collect()

    if x_source.shape != (EXPECTED_SOURCE_ROWS, 77):
        raise RuntimeError(f"Unexpected source shape: {x_source.shape}")
    if x_target.shape != (EXPECTED_TARGET_ROWS, 77):
        raise RuntimeError(f"Unexpected target shape: {x_target.shape}")
    if np.isnan(x_source).any() or np.isnan(x_target).any():
        raise RuntimeError("NaN values remain after source-median imputation.")

    print("Source test:", x_source.shape)
    print("Target test:", x_target.shape)
    print("Source attack rate:", round(float(y_source.mean()), 6))
    print("Target attack rate:", round(float(y_target.mean()), 6))
    print("Target use: final evaluation only")
    return {
        "source": (x_source, y_source),
        "target": (x_target, y_target),
        "medians": medians,
        "group_names": group_names,
        "group_indices": group_indices,
    }


def verify_forest(
    model: RandomForestClassifier,
    seed: int,
    expected_features: int,
    label: str,
) -> None:
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
        raise RuntimeError(f"{label} parameter mismatch: {mismatches}")
    if len(model.estimators_) != 300:
        raise RuntimeError(
            f"{label} contains {len(model.estimators_)} trees instead of 300."
        )
    if int(model.n_features_in_) != expected_features:
        raise RuntimeError(
            f"{label} expects {model.n_features_in_} features instead of "
            f"{expected_features}."
        )


def verify_policy(policy: dict, label: str) -> None:
    threshold = float(policy["classification_threshold"])
    if not 0 < threshold < 1:
        raise RuntimeError(f"Invalid classification threshold for {label}.")
    expected_coverages = {"0.8", "0.9", "0.95"}
    if set(policy["confidence_cutoffs"]) != expected_coverages:
        raise RuntimeError(f"Invalid confidence policies for {label}.")
    if policy.get("target_labels_used") is not False:
        raise RuntimeError(f"Target-use declaration is invalid for {label}.")


def predict_condition_probability(
    model: RandomForestClassifier,
    calibrator,
    x: np.ndarray,
    missing_groups: tuple[str, ...],
    group_names: list[str],
    group_indices: dict[str, list[int]],
    medians: np.ndarray,
    group_aware: bool,
    batch_size: int = PREDICTION_BATCH_SIZE,
) -> np.ndarray:
    raw_probability = np.empty(len(x), dtype=np.float32)
    predictor_count = x.shape[1]
    group_count = len(group_names)
    missing_group_numbers = [group_names.index(group) for group in missing_groups]
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        batch = np.array(x[start:stop], dtype=np.float32, copy=True)
        for group in missing_groups:
            columns = np.asarray(group_indices[group], dtype=np.int64)
            batch[:, columns] = medians[columns]
        if group_aware:
            extended = np.empty(
                (len(batch), predictor_count + group_count), dtype=np.float32
            )
            extended[:, :predictor_count] = batch
            extended[:, predictor_count:] = 1.0
            for group_number in missing_group_numbers:
                extended[:, predictor_count + group_number] = 0.0
            raw_probability[start:stop] = model.predict_proba(extended)[:, 1]
            del extended
        else:
            raw_probability[start:stop] = model.predict_proba(batch)[:, 1]
        del batch
    calibrated = calibrator.predict_proba(
        probability_to_logit(raw_probability)
    )[:, 1]
    return np.asarray(calibrated, dtype=np.float64)


def condition_key(
    dataset: str, condition_type: str, missing_groups: str
) -> tuple[str, str, str]:
    return dataset, condition_type, missing_groups


def condition_result_path(
    condition_dir: Path,
    dataset: str,
    condition_type: str,
    missing_groups: str,
) -> Path:
    safe_groups = missing_groups.replace("+", "__")
    return condition_dir / f"{dataset}__{condition_type}__{safe_groups}.json"


def restore_condition_results(
    condition_dir: Path,
    expected_conditions: list[tuple[str, str, str]],
) -> tuple[list[dict], list[dict], list[dict], set[tuple[str, str, str]]]:
    nonselective_rows: list[dict] = []
    selective_rows: list[dict] = []
    paired_rows: list[dict] = []
    completed: set[tuple[str, str, str]] = set()
    for dataset, condition_type, missing_groups in expected_conditions:
        path = condition_result_path(
            condition_dir, dataset, condition_type, missing_groups
        )
        if not path.exists():
            continue
        payload = load_json(path)
        expected_key = [dataset, condition_type, missing_groups]
        if payload.get("condition_key") != expected_key:
            raise RuntimeError(f"Condition key mismatch in {path.name}.")
        condition_nonselective = payload.get("nonselective", [])
        condition_selective = payload.get("selective", [])
        condition_paired = payload.get("paired", [])
        if not (
            len(condition_nonselective) == 2
            and len(condition_selective) == 6
            and len(condition_paired) == 1
        ):
            raise RuntimeError(f"Incomplete condition result in {path.name}.")
        nonselective_rows.extend(condition_nonselective)
        selective_rows.extend(condition_selective)
        paired_rows.extend(condition_paired)
        completed.add((dataset, condition_type, missing_groups))
    return nonselective_rows, selective_rows, paired_rows, completed


def evaluate_seed(seed: int, fixed: dict) -> None:
    seed_dir = RF_DIR / f"seed_{seed}"
    completion_path = seed_dir / "EVALUATION_COMPLETE.json"
    final_paths = {
        "nonselective": seed_dir / "random_forest_nonselective_results.csv",
        "selective": seed_dir / "random_forest_selective_results.csv",
        "paired": seed_dir / "random_forest_paired_counts.csv",
    }
    if completion_path.exists() and all(path.exists() for path in final_paths.values()):
        print(f"Seed {seed}: evaluation COMPLETE; skipping.")
        return

    training_marker = seed_dir / "TRAINING_COMPLETE.json"
    if not training_marker.exists():
        raise FileNotFoundError(f"Training marker missing for seed {seed}.")
    marker = load_json(training_marker)
    if marker.get("status") != "training_complete":
        raise RuntimeError(f"Training marker is invalid for seed {seed}.")

    baseline_model = joblib.load(seed_dir / "ordinary_random_forest.joblib")
    group_model = joblib.load(seed_dir / "group_aware_random_forest.joblib")
    verify_forest(baseline_model, seed, 77, "ordinary baseline")
    verify_forest(group_model, seed, 81, "group-aware model")
    baseline_platt = joblib.load(seed_dir / "ordinary_platt.joblib")
    group_platt = joblib.load(seed_dir / "group_aware_platt.joblib")
    baseline_policy = load_json(seed_dir / "ordinary_policy.json")
    group_policy = load_json(seed_dir / "group_aware_policy.json")
    verify_policy(baseline_policy, "ordinary baseline")
    verify_policy(group_policy, "group-aware model")

    baseline_threshold = float(baseline_policy["classification_threshold"])
    group_threshold = float(group_policy["classification_threshold"])
    baseline_cutoffs = {
        float(key): float(value)
        for key, value in baseline_policy["confidence_cutoffs"].items()
    }
    group_cutoffs = {
        float(key): float(value)
        for key, value in group_policy["confidence_cutoffs"].items()
    }

    group_names = fixed["group_names"]
    conditions = [("complete", tuple())]
    conditions.extend(("single", (group,)) for group in group_names)
    conditions.extend(
        ("pairwise_unseen", pair) for pair in combinations(group_names, 2)
    )
    expected_conditions = []
    for dataset_name in ("source", "target"):
        for condition_type, missing_groups in conditions:
            condition_name = "+".join(missing_groups) if missing_groups else "None"
            expected_conditions.append(
                (dataset_name, condition_type, condition_name)
            )
    condition_dir = seed_dir / "condition_results"
    condition_dir.mkdir(parents=True, exist_ok=True)
    (
        nonselective_rows,
        selective_rows,
        paired_rows,
        completed,
    ) = restore_condition_results(condition_dir, expected_conditions)
    started = time.time()
    print("\n" + "=" * 72)
    print("EVALUATING RANDOM FOREST SEED:", seed)
    print("Completed conditions restored:", len(completed), "/ 22")
    print("=" * 72)

    for dataset_name in ("source", "target"):
        x_data, y_data = fixed[dataset_name]
        for condition_type, missing_groups in conditions:
            condition_name = "+".join(missing_groups) if missing_groups else "None"
            key = condition_key(dataset_name, condition_type, condition_name)
            if key in completed:
                print(dataset_name, "|", condition_type, "|", condition_name, "| SKIP")
                continue
            condition_started = time.time()
            nonselective_start = len(nonselective_rows)
            selective_start = len(selective_rows)
            paired_start = len(paired_rows)
            baseline_probability = predict_condition_probability(
                baseline_model,
                baseline_platt,
                x_data,
                missing_groups,
                group_names,
                fixed["group_indices"],
                fixed["medians"],
                group_aware=False,
            )
            baseline_metrics = evaluate_probability(
                y_data, baseline_probability, baseline_threshold
            )
            baseline_metrics.update(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "model": "ordinary_random_forest",
                    "condition_type": condition_type,
                    "missing_groups": condition_name,
                }
            )
            nonselective_rows.append(baseline_metrics)
            baseline_aurc = calculate_aurc(
                y_data, baseline_probability, baseline_threshold
            )
            for desired_coverage, cutoff in baseline_cutoffs.items():
                row = evaluate_selective(
                    y_data, baseline_probability, baseline_threshold, cutoff
                )
                row.update(
                    {
                        "seed": seed,
                        "dataset": dataset_name,
                        "model": "ordinary_random_forest",
                        "condition_type": condition_type,
                        "missing_groups": condition_name,
                        "desired_policy_coverage": desired_coverage,
                        "confidence_cutoff": cutoff,
                        "aurc": baseline_aurc,
                    }
                )
                selective_rows.append(row)
            baseline_prediction = (
                baseline_probability >= baseline_threshold
            ).astype(np.int8)

            group_probability = predict_condition_probability(
                group_model,
                group_platt,
                x_data,
                missing_groups,
                group_names,
                fixed["group_indices"],
                fixed["medians"],
                group_aware=True,
            )
            group_metrics = evaluate_probability(
                y_data, group_probability, group_threshold
            )
            group_metrics.update(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "model": "group_aware_random_forest",
                    "condition_type": condition_type,
                    "missing_groups": condition_name,
                }
            )
            nonselective_rows.append(group_metrics)
            group_aurc = calculate_aurc(
                y_data, group_probability, group_threshold
            )
            for desired_coverage, cutoff in group_cutoffs.items():
                row = evaluate_selective(
                    y_data, group_probability, group_threshold, cutoff
                )
                row.update(
                    {
                        "seed": seed,
                        "dataset": dataset_name,
                        "model": "group_aware_random_forest",
                        "condition_type": condition_type,
                        "missing_groups": condition_name,
                        "desired_policy_coverage": desired_coverage,
                        "confidence_cutoff": cutoff,
                        "aurc": group_aurc,
                    }
                )
                selective_rows.append(row)
            group_prediction = (
                group_probability >= group_threshold
            ).astype(np.int8)

            baseline_correct = baseline_prediction == y_data
            group_correct = group_prediction == y_data
            paired_rows.append(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "condition_type": condition_type,
                    "missing_groups": condition_name,
                    "both_correct": int((baseline_correct & group_correct).sum()),
                    "baseline_only_correct": int(
                        (baseline_correct & ~group_correct).sum()
                    ),
                    "group_only_correct": int(
                        (~baseline_correct & group_correct).sum()
                    ),
                    "both_wrong": int((~baseline_correct & ~group_correct).sum()),
                }
            )

            condition_payload = {
                "condition_key": [dataset_name, condition_type, condition_name],
                "nonselective": nonselective_rows[nonselective_start:],
                "selective": selective_rows[selective_start:],
                "paired": paired_rows[paired_start:],
            }
            condition_path = condition_result_path(
                condition_dir, dataset_name, condition_type, condition_name
            )
            atomic_json_dump(condition_payload, condition_path)
            completed.add(key)
            del (
                baseline_probability,
                group_probability,
                baseline_prediction,
                group_prediction,
                baseline_correct,
                group_correct,
            )
            gc.collect()
            print(
                dataset_name,
                "|",
                condition_type,
                "|",
                condition_name,
                "|",
                round(time.time() - condition_started, 1),
                "seconds | saved",
            )

    if not (
        len(nonselective_rows) == 44
        and len(selective_rows) == 132
        and len(paired_rows) == 22
    ):
        raise RuntimeError(
            "Unexpected per-seed result counts: "
            f"nonselective={len(nonselective_rows)}, "
            f"selective={len(selective_rows)}, paired={len(paired_rows)}."
        )
    atomic_csv_dump(nonselective_rows, final_paths["nonselective"])
    atomic_csv_dump(selective_rows, final_paths["selective"])
    atomic_csv_dump(paired_rows, final_paths["paired"])
    elapsed = time.time() - started
    completion = {
        "seed": seed,
        "status": "evaluation_complete",
        "elapsed_seconds_this_invocation": elapsed,
        "source_rows": EXPECTED_SOURCE_ROWS,
        "target_rows": EXPECTED_TARGET_ROWS,
        "target_used_for": "final evaluation only",
        "models_verified": 2,
        "trees_per_model": 300,
        "completed_conditions": 22,
        "result_rows": {
            "nonselective": len(nonselective_rows),
            "selective": len(selective_rows),
            "paired": len(paired_rows),
        },
        "result_sha256": {
            path.name: sha256(path) for path in final_paths.values()
        },
    }
    atomic_json_dump(completion, completion_path)
    print(f"Seed {seed} evaluation completed in {elapsed:.1f} seconds.")
    del baseline_model, group_model, baseline_platt, group_platt
    gc.collect()


def combine_results() -> None:
    nonselective = []
    selective = []
    paired = []
    missing = []
    for seed in SEEDS:
        seed_dir = RF_DIR / f"seed_{seed}"
        paths = {
            "nonselective": seed_dir / "random_forest_nonselective_results.csv",
            "selective": seed_dir / "random_forest_selective_results.csv",
            "paired": seed_dir / "random_forest_paired_counts.csv",
        }
        if not all(path.exists() for path in paths.values()):
            missing.append(seed)
            continue
        nonselective.append(pd.read_csv(paths["nonselective"]))
        selective.append(pd.read_csv(paths["selective"]))
        paired.append(pd.read_csv(paths["paired"]))
    if missing:
        raise RuntimeError(f"Evaluation outputs are missing for seeds: {missing}")

    combined_nonselective = pd.concat(nonselective, ignore_index=True)
    combined_selective = pd.concat(selective, ignore_index=True)
    combined_paired = pd.concat(paired, ignore_index=True)
    if not (
        len(combined_nonselective) == 220
        and len(combined_selective) == 660
        and len(combined_paired) == 110
    ):
        raise RuntimeError("Combined Random Forest result counts are invalid.")
    output_paths = {
        "nonselective": RF_DIR / "random_forest_five_seed_nonselective_all.csv",
        "selective": RF_DIR / "random_forest_five_seed_selective_all.csv",
        "paired": RF_DIR / "random_forest_five_seed_paired_counts_all.csv",
    }
    combined_nonselective.to_csv(output_paths["nonselective"], index=False)
    combined_selective.to_csv(output_paths["selective"], index=False)
    combined_paired.to_csv(output_paths["paired"], index=False)
    marker = {
        "status": "complete",
        "seeds": SEEDS,
        "nonselective_rows": len(combined_nonselective),
        "selective_rows": len(combined_selective),
        "paired_rows": len(combined_paired),
        "target_used_for": "final evaluation only",
        "result_sha256": {
            path.name: sha256(path) for path in output_paths.values()
        },
    }
    atomic_json_dump(
        marker, RF_DIR / "RANDOM_FOREST_FIVE_SEED_EVALUATION_COMPLETE.json"
    )
    print("\nCombined Random Forest outputs saved:")
    print("Non-selective rows:", len(combined_nonselective))
    print("Selective rows:", len(combined_selective))
    print("Paired rows:", len(combined_paired))


def main() -> None:
    mount_drive()
    verify_protocol()
    fixed = load_fixed_test_data()
    for seed in SEEDS:
        evaluate_seed(seed, fixed)
    combine_results()
    print("\nRandom Forest five-seed evaluation is complete.")
    print("RANDOM_FOREST_FIVE_SEED_EVALUATION_COMPLETE")


if __name__ == "__main__":
    main()
