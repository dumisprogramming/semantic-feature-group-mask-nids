"""Restartable five-seed confirmatory evaluation for the NIDS study.

Run in Google Colab with:
    %run /content/confirmatory_five_seed_evaluation.py

The script loads previously trained models from Google Drive, evaluates source
and target test sets, saves per-seed results, skips completed seeds, and creates
combined result tables. It never trains or tunes a model.
"""

from __future__ import annotations

import gc
import json
import os
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
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


CHECKPOINT_DIR = Path(
    os.environ.get(
        "NIDS_CHECKPOINT_DIR",
        "/content/drive/MyDrive/research/researchdata/"
        "NIDS_Research/NIDS_Research_Checkpoints",
    )
)
REPLICATION_DIR = CHECKPOINT_DIR / "five_seed_confirmatory_replication"
SEEDS = [2027, 2028, 2029, 2030, 2031]


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError:
        return
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 15
) -> float:
    y_true = np.asarray(y_true)
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
    y_true = np.asarray(y_true)
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


def load_fixed_data() -> dict:
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {CHECKPOINT_DIR}")

    with open(CHECKPOINT_DIR / "experiment_metadata.json") as file:
        metadata = json.load(file)

    predictor_columns = metadata["predictors"]
    semantic_groups = metadata["semantic_groups"]
    group_names = list(semantic_groups)
    feature_index = {
        feature: position for position, feature in enumerate(predictor_columns)
    }
    group_indices = {
        group: [feature_index[feature] for feature in semantic_groups[group]]
        for group in group_names
    }

    imputer = joblib.load(CHECKPOINT_DIR / "source_median_imputer.joblib")
    training_medians = np.asarray(imputer.statistics_, dtype=np.float32)

    source_df = pd.read_parquet(CHECKPOINT_DIR / "source_test.parquet")
    target_df = pd.read_parquet(CHECKPOINT_DIR / "target_test_sealed.parquet")
    x_source = imputer.transform(source_df[predictor_columns]).astype("float32")
    y_source = source_df["target"].to_numpy(dtype="int8")
    x_target = imputer.transform(target_df[predictor_columns]).astype("float32")
    y_target = target_df["target"].to_numpy(dtype="int8")

    if np.isnan(x_source).sum() or np.isnan(x_target).sum():
        raise ValueError("Unexpected missing values remain after imputation.")

    print("Source test:", x_source.shape)
    print("Target test:", x_target.shape)
    print("Source attack rate:", round(float(y_source.mean()), 6))
    print("Target attack rate:", round(float(y_target.mean()), 6))

    return {
        "group_names": group_names,
        "group_indices": group_indices,
        "training_medians": training_medians,
        "source": (x_source, y_source),
        "target": (x_target, y_target),
    }


def baseline_condition(
    x: np.ndarray,
    missing_groups: tuple[str, ...],
    group_indices: dict,
    medians: np.ndarray,
) -> np.ndarray:
    output = np.asarray(x, dtype=np.float32).copy()
    for group in missing_groups:
        columns = group_indices[group]
        output[:, columns] = medians[columns]
    return output


def group_condition(
    x: np.ndarray,
    missing_groups: tuple[str, ...],
    group_names: list[str],
    group_indices: dict,
    medians: np.ndarray,
) -> np.ndarray:
    output = np.asarray(x, dtype=np.float32).copy()
    availability = np.ones((len(output), len(group_names)), dtype=np.float32)
    for group in missing_groups:
        group_number = group_names.index(group)
        columns = group_indices[group]
        output[:, columns] = medians[columns]
        availability[:, group_number] = 0.0
    return np.hstack([output, availability])


def evaluate_seed(seed: int, fixed: dict) -> None:
    seed_dir = REPLICATION_DIR / f"seed_{seed}"
    completion_file = seed_dir / "EVALUATION_COMPLETE.json"
    if completion_file.exists():
        print(f"Seed {seed}: evaluation already complete; skipping.")
        return

    baseline_model = xgb.XGBClassifier()
    baseline_model.load_model(str(seed_dir / "ordinary_baseline_xgboost.json"))
    group_model = xgb.XGBClassifier()
    group_model.load_model(str(seed_dir / "group_aware_xgboost.json"))
    baseline_platt = joblib.load(seed_dir / "ordinary_baseline_platt.joblib")
    group_platt = joblib.load(seed_dir / "group_aware_platt.joblib")

    with open(seed_dir / "ordinary_baseline_policy.json") as file:
        baseline_policy = json.load(file)
    with open(seed_dir / "group_aware_policy.json") as file:
        group_policy = json.load(file)

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
    group_indices = fixed["group_indices"]
    medians = fixed["training_medians"]
    conditions = [("complete", tuple())]
    conditions.extend(("single", (group,)) for group in group_names)
    conditions.extend(
        ("pairwise_unseen", pair) for pair in combinations(group_names, 2)
    )

    nonselective_rows = []
    selective_rows = []
    paired_rows = []
    started = time.time()
    print("\n" + "=" * 70)
    print("EVALUATING SEED:", seed)
    print("=" * 70)

    for dataset_name in ["source", "target"]:
        x_data, y_data = fixed[dataset_name]
        for condition_type, missing_groups in conditions:
            condition_name = "+".join(missing_groups) if missing_groups else "None"
            condition_started = time.time()

            x_baseline = baseline_condition(
                x_data, missing_groups, group_indices, medians
            )
            baseline_raw = baseline_model.predict_proba(x_baseline)[:, 1]
            baseline_probability = baseline_platt.predict_proba(
                probability_to_logit(baseline_raw)
            )[:, 1]
            baseline_metrics = evaluate_probability(
                y_data, baseline_probability, baseline_threshold
            )
            baseline_metrics.update(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "model": "ordinary_baseline",
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
                        "model": "ordinary_baseline",
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
            ).astype("int8")
            del x_baseline, baseline_raw
            gc.collect()

            x_group = group_condition(
                x_data,
                missing_groups,
                group_names,
                group_indices,
                medians,
            )
            group_raw = group_model.predict_proba(x_group)[:, 1]
            group_probability = group_platt.predict_proba(
                probability_to_logit(group_raw)
            )[:, 1]
            group_metrics = evaluate_probability(
                y_data, group_probability, group_threshold
            )
            group_metrics.update(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "model": "group_aware",
                    "condition_type": condition_type,
                    "missing_groups": condition_name,
                }
            )
            nonselective_rows.append(group_metrics)
            group_aurc = calculate_aurc(y_data, group_probability, group_threshold)
            for desired_coverage, cutoff in group_cutoffs.items():
                row = evaluate_selective(
                    y_data, group_probability, group_threshold, cutoff
                )
                row.update(
                    {
                        "seed": seed,
                        "dataset": dataset_name,
                        "model": "group_aware",
                        "condition_type": condition_type,
                        "missing_groups": condition_name,
                        "desired_policy_coverage": desired_coverage,
                        "confidence_cutoff": cutoff,
                        "aurc": group_aurc,
                    }
                )
                selective_rows.append(row)
            group_prediction = (group_probability >= group_threshold).astype("int8")

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

            del (
                x_group,
                group_raw,
                baseline_probability,
                group_probability,
                baseline_prediction,
                group_prediction,
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
                "seconds",
            )

    pd.DataFrame(nonselective_rows).to_csv(
        seed_dir / "confirmatory_nonselective_results.csv", index=False
    )
    pd.DataFrame(selective_rows).to_csv(
        seed_dir / "confirmatory_selective_results.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        seed_dir / "confirmatory_paired_counts.csv", index=False
    )
    elapsed = time.time() - started
    with open(completion_file, "w") as file:
        json.dump(
            {
                "seed": seed,
                "status": "evaluation_complete",
                "elapsed_seconds": elapsed,
                "source_rows": int(len(fixed["source"][1])),
                "target_rows": int(len(fixed["target"][1])),
                "target_used_for": "evaluation only",
            },
            file,
            indent=2,
        )
    print(f"Seed {seed} evaluation completed in {elapsed:.1f} seconds.")
    del baseline_model, group_model, baseline_platt, group_platt
    gc.collect()


def combine_results() -> None:
    nonselective = []
    selective = []
    paired = []
    missing = []
    for seed in SEEDS:
        seed_dir = REPLICATION_DIR / f"seed_{seed}"
        paths = {
            "nonselective": seed_dir / "confirmatory_nonselective_results.csv",
            "selective": seed_dir / "confirmatory_selective_results.csv",
            "paired": seed_dir / "confirmatory_paired_counts.csv",
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
    combined_nonselective.to_csv(
        REPLICATION_DIR / "five_seed_nonselective_all.csv", index=False
    )
    combined_selective.to_csv(
        REPLICATION_DIR / "five_seed_selective_all.csv", index=False
    )
    combined_paired.to_csv(
        REPLICATION_DIR / "five_seed_paired_counts_all.csv", index=False
    )
    with open(REPLICATION_DIR / "FIVE_SEED_EVALUATION_COMPLETE.json", "w") as file:
        json.dump(
            {
                "status": "complete",
                "seeds": SEEDS,
                "nonselective_rows": int(len(combined_nonselective)),
                "selective_rows": int(len(combined_selective)),
                "paired_rows": int(len(combined_paired)),
            },
            file,
            indent=2,
        )
    print("\nCombined outputs saved:")
    print("Non-selective rows:", len(combined_nonselective))
    print("Selective rows:", len(combined_selective))
    print("Paired rows:", len(combined_paired))


def main() -> None:
    mount_drive()
    fixed = load_fixed_data()
    for seed in SEEDS:
        evaluate_seed(seed, fixed)
    combine_results()
    print("\nFive-seed confirmatory evaluation is complete.")


if __name__ == "__main__":
    main()
