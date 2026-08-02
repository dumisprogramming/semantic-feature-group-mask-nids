"""Read-only audit of the five-seed NIDS component-ablation models.

Run in Google Colab:

    %run /content/audit_component_ablation_models.py

The audit does not train, calibrate, or evaluate a model. It checks saved
artifacts, model/tree hashes, feature counts, boosted rounds, best iterations,
random seeds, sampling parameters, policy thresholds, and completion markers.
It writes two small files to the component_ablation directory:

    component_ablation_training_audit.csv
    component_ablation_audit_report.json
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb


CHECKPOINT_DIR = Path(
    os.environ.get(
        "NIDS_CHECKPOINT_DIR",
        "/content/drive/MyDrive/research/researchdata/"
        "NIDS_Research/NIDS_Research_Checkpoints",
    )
)
REPLICATION_DIR = CHECKPOINT_DIR / "five_seed_confirmatory_replication"
ABLATION_DIR = REPLICATION_DIR / "component_ablation"
SEEDS = [2027, 2028, 2029, 2030, 2031]

MODEL_SPECS = {
    "ordinary_baseline": {
        "features": 77,
        "model": lambda seed: (
            REPLICATION_DIR
            / f"seed_{seed}"
            / "ordinary_baseline_xgboost.json"
        ),
        "calibrator": lambda seed: (
            REPLICATION_DIR / f"seed_{seed}" / "ordinary_baseline_platt.joblib"
        ),
        "policy": lambda seed: (
            REPLICATION_DIR / f"seed_{seed}" / "ordinary_baseline_policy.json"
        ),
        "manifest": lambda seed: None,
        "completion": lambda seed: (
            REPLICATION_DIR / f"seed_{seed}" / "TRAINING_COMPLETE.json"
        ),
    },
    "full_method": {
        "features": 81,
        "model": lambda seed: (
            REPLICATION_DIR / f"seed_{seed}" / "group_aware_xgboost.json"
        ),
        "calibrator": lambda seed: (
            REPLICATION_DIR / f"seed_{seed}" / "group_aware_platt.joblib"
        ),
        "policy": lambda seed: (
            REPLICATION_DIR / f"seed_{seed}" / "group_aware_policy.json"
        ),
        "manifest": lambda seed: None,
        "completion": lambda seed: (
            REPLICATION_DIR / f"seed_{seed}" / "TRAINING_COMPLETE.json"
        ),
    },
    "indicators_only": {
        "features": 81,
        "model": lambda seed: (
            ABLATION_DIR
            / f"seed_{seed}"
            / "indicators_only"
            / "indicators_only_xgboost.json"
        ),
        "calibrator": lambda seed: (
            ABLATION_DIR
            / f"seed_{seed}"
            / "indicators_only"
            / "indicators_only_platt.joblib"
        ),
        "policy": lambda seed: (
            ABLATION_DIR
            / f"seed_{seed}"
            / "indicators_only"
            / "indicators_only_policy.json"
        ),
        "manifest": lambda seed: (
            ABLATION_DIR
            / f"seed_{seed}"
            / "indicators_only"
            / "MODEL_TRAINED.json"
        ),
        "completion": lambda seed: (
            ABLATION_DIR
            / f"seed_{seed}"
            / "indicators_only"
            / "TRAINING_COMPLETE.json"
        ),
    },
    "augmentation_only": {
        "features": 77,
        "model": lambda seed: (
            ABLATION_DIR
            / f"seed_{seed}"
            / "augmentation_only"
            / "augmentation_only_xgboost.json"
        ),
        "calibrator": lambda seed: (
            ABLATION_DIR
            / f"seed_{seed}"
            / "augmentation_only"
            / "augmentation_only_platt.joblib"
        ),
        "policy": lambda seed: (
            ABLATION_DIR
            / f"seed_{seed}"
            / "augmentation_only"
            / "augmentation_only_policy.json"
        ),
        "manifest": lambda seed: (
            ABLATION_DIR
            / f"seed_{seed}"
            / "augmentation_only"
            / "MODEL_TRAINED.json"
        ),
        "completion": lambda seed: (
            ABLATION_DIR
            / f"seed_{seed}"
            / "augmentation_only"
            / "TRAINING_COMPLETE.json"
        ),
    },
}


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


def tree_hash(booster: xgb.Booster) -> str:
    trees = booster.get_dump(dump_format="json", with_stats=True)
    return hashlib.sha256("\n".join(trees).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    with open(path) as file:
        return json.load(file)


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


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def config_parameters(config: dict) -> dict:
    return {
        "config_seed": as_int(recursively_find(config, "seed")),
        "learning_rate": as_float(recursively_find(config, "eta")),
        "max_depth": as_int(recursively_find(config, "max_depth")),
        "min_child_weight": as_float(
            recursively_find(config, "min_child_weight")
        ),
        "subsample": as_float(recursively_find(config, "subsample")),
        "colsample_bytree": as_float(
            recursively_find(config, "colsample_bytree")
        ),
        "colsample_bylevel": as_float(
            recursively_find(config, "colsample_bylevel")
        ),
        "colsample_bynode": as_float(
            recursively_find(config, "colsample_bynode")
        ),
        "reg_alpha": as_float(recursively_find(config, "reg_alpha")),
        "reg_lambda": as_float(recursively_find(config, "reg_lambda")),
        "scale_pos_weight": as_float(
            recursively_find(config, "scale_pos_weight")
        ),
        "max_bin": as_int(recursively_find(config, "max_bin")),
    }


def policy_threshold(policy: dict):
    value = policy.get("classification_threshold")
    return None if value is None else float(value)


def audit_one(seed: int, model_name: str, spec: dict) -> dict:
    model_path = spec["model"](seed)
    calibrator_path = spec["calibrator"](seed)
    policy_path = spec["policy"](seed)
    manifest_path = spec["manifest"](seed)
    completion_path = spec["completion"](seed)

    paths = [model_path, calibrator_path, policy_path, completion_path]
    if manifest_path is not None:
        paths.append(manifest_path)
    missing = [str(path) for path in paths if not path.exists()]
    base = {
        "seed": seed,
        "model": model_name,
        "expected_features": spec["features"],
        "required_files_present": not missing,
        "missing_files": " | ".join(missing),
    }
    if missing:
        return base

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    parameters = config_parameters(config)
    policy = read_json(policy_path)
    completion = read_json(completion_path)

    manifest = {}
    manifest_parameters = {}
    if manifest_path is not None:
        manifest = read_json(manifest_path)
        manifest_parameters = manifest.get("xgboost_parameters", {})

    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    row = {
        **base,
        "model_sha256": sha256_file(model_path),
        "tree_sha256": tree_hash(booster),
        "calibrator_sha256": sha256_file(calibrator_path),
        "policy_sha256": sha256_file(policy_path),
        "actual_features": int(booster.num_features()),
        "feature_count_pass": int(booster.num_features()) == spec["features"],
        "boosted_rounds": int(booster.num_boosted_rounds()),
        "best_iteration_model": (
            None if best_iteration is None else int(best_iteration)
        ),
        "best_score_model": None if best_score is None else float(best_score),
        "best_iteration_manifest": manifest.get("best_iteration"),
        "best_score_manifest": manifest.get("best_score"),
        "manifest_random_state": manifest_parameters.get("random_state"),
        "manifest_subsample": manifest_parameters.get("subsample"),
        "manifest_colsample_bytree": manifest_parameters.get(
            "colsample_bytree"
        ),
        "manifest_colsample_bylevel": manifest_parameters.get(
            "colsample_bylevel"
        ),
        "manifest_colsample_bynode": manifest_parameters.get(
            "colsample_bynode"
        ),
        "classification_threshold": policy_threshold(policy),
        "completion_status": completion.get("status"),
        "training_target_used": completion.get("target_data_used"),
    }
    row.update({f"config_{key}": value for key, value in parameters.items()})
    return row


def audit_fairness(frame: pd.DataFrame) -> list[dict]:
    comparisons = []
    parameter_columns = [
        "config_learning_rate",
        "config_max_depth",
        "config_min_child_weight",
        "config_subsample",
        "config_colsample_bytree",
        "config_colsample_bylevel",
        "config_colsample_bynode",
        "config_reg_alpha",
        "config_reg_lambda",
        "config_scale_pos_weight",
        "config_max_bin",
    ]
    for seed in SEEDS:
        seed_frame = frame[frame["seed"] == seed].set_index("model")
        if "ordinary_baseline" not in seed_frame.index:
            continue
        reference = seed_frame.loc["ordinary_baseline"]
        for model_name in ["indicators_only", "augmentation_only"]:
            if model_name not in seed_frame.index:
                continue
            candidate = seed_frame.loc[model_name]
            differences = []
            for column in parameter_columns:
                left = reference.get(column)
                right = candidate.get(column)
                if pd.isna(left) and pd.isna(right):
                    continue
                if pd.isna(left) or pd.isna(right) or float(left) != float(right):
                    differences.append(f"{column}:{left}!={right}")
            comparisons.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "matches_baseline_hyperparameters": not differences,
                    "differences": " | ".join(differences),
                }
            )
    return comparisons


def build_report(frame: pd.DataFrame) -> dict:
    errors = []
    warnings = []

    if len(frame) != 20:
        errors.append(f"Expected 20 model rows; found {len(frame)}.")
    if not bool(frame["required_files_present"].all()):
        errors.append("One or more required model artifacts are missing.")
    valid_feature_rows = frame["feature_count_pass"].dropna()
    if not valid_feature_rows.empty and not bool(valid_feature_rows.all()):
        errors.append("At least one model has the wrong feature count.")

    fairness = audit_fairness(frame)
    unfair = [row for row in fairness if not row["matches_baseline_hyperparameters"]]
    if unfair:
        errors.append(
            "Ablation and ordinary-baseline booster hyperparameters differ."
        )

    uniqueness = {}
    for model_name in MODEL_SPECS:
        selected = frame[frame["model"] == model_name]
        uniqueness[model_name] = {
            "model_hashes": int(selected["model_sha256"].nunique()),
            "tree_hashes": int(selected["tree_sha256"].nunique()),
            "calibrator_hashes": int(selected["calibrator_sha256"].nunique()),
            "policy_hashes": int(selected["policy_sha256"].nunique()),
        }

    indicator_unique = uniqueness.get("indicators_only", {}).get(
        "tree_hashes", 0
    )
    if indicator_unique == 1:
        indicators = frame[frame["model"] == "indicators_only"]
        random_states = indicators["manifest_random_state"].dropna().tolist()
        sampling_columns = [
            "config_subsample",
            "config_colsample_bytree",
            "config_colsample_bylevel",
            "config_colsample_bynode",
        ]
        stochastic = False
        for column in sampling_columns:
            values = indicators[column].dropna().astype(float)
            if not values.empty and bool((values < 1.0).any()):
                stochastic = True
        if stochastic:
            warnings.append(
                "Indicators-only trees are identical across seeds despite at "
                "least one sampling fraction below 1.0. Review before paper use."
            )
        else:
            warnings.append(
                "Indicators-only trees are identical across seeds. This is "
                "consistent with deterministic complete-only training and "
                "constant all-one indicator columns, but it should be disclosed."
            )
        if random_states != SEEDS:
            errors.append(
                "Indicators-only manifest random states do not match the five "
                "confirmatory seeds."
            )

    status = "PASS" if not errors else "REVIEW_REQUIRED"
    return {
        "audit_status": status,
        "models_expected": 20,
        "models_found": int(len(frame)),
        "errors": errors,
        "warnings": warnings,
        "hash_uniqueness_across_five_seeds": uniqueness,
        "hyperparameter_fairness": fairness,
        "interpretation_rule": (
            "Identical indicators-only trees are acceptable only when source "
            "training is deterministic; constant availability indicators cannot "
            "create a useful split without masked examples."
        ),
    }


def main() -> None:
    mount_drive()
    if not ABLATION_DIR.exists():
        raise FileNotFoundError(ABLATION_DIR)

    rows = []
    for seed in SEEDS:
        for model_name, spec in MODEL_SPECS.items():
            print(f"Auditing seed {seed} | {model_name}")
            rows.append(audit_one(seed, model_name, spec))

    frame = pd.DataFrame(rows)
    csv_path = ABLATION_DIR / "component_ablation_training_audit.csv"
    frame.to_csv(csv_path, index=False)
    report = build_report(frame)
    report_path = ABLATION_DIR / "component_ablation_audit_report.json"
    with open(report_path, "w") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 72)
    print("COMPONENT-ABLATION AUDIT:", report["audit_status"])
    print("=" * 72)
    print("Models audited:", report["models_found"], "/ 20")
    print("Errors:", len(report["errors"]))
    for item in report["errors"]:
        print("ERROR:", item)
    print("Warnings:", len(report["warnings"]))
    for item in report["warnings"]:
        print("WARNING:", item)
    print("\nHash uniqueness across five seeds:")
    for model_name, values in report[
        "hash_uniqueness_across_five_seeds"
    ].items():
        print(model_name, values)
    print("\nSaved:", csv_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
