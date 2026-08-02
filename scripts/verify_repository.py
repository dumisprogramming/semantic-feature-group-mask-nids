"""Audit configuration, provenance, aggregate results, tables and figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = {2027, 2028, 2029, 2030, 2031}


def check(condition: bool, message: str, problems: list[str]) -> None:
    if not condition:
        problems.append(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, problems: list[str]) -> dict:
    if not path.exists():
        problems.append(f"Missing {path.relative_to(ROOT)}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        problems.append(f"Cannot read {path.relative_to(ROOT)}: {error}")
        return {}


def audit_configuration(problems: list[str]) -> dict:
    metadata = load_json(ROOT / "config" / "experiment_metadata.json", problems)
    groups_payload = load_json(
        ROOT / "config" / "semantic_feature_groups.json", problems
    )
    renaming = load_json(ROOT / "config" / "feature_renaming_map.json", problems)
    protocol = load_json(
        ROOT / "protocols" / "replication_protocol_frozen.json", problems
    )
    groups = groups_payload.get("semantic_groups", {})
    flattened = [feature for group in groups.values() for feature in group]
    check(len(flattened) == 77, "Semantic groups do not contain 77 assignments.", problems)
    check(
        len(flattened) == len(set(flattened)),
        "Semantic groups contain duplicate assignments.",
        problems,
    )
    check(
        set(flattened) == set(metadata.get("predictors", [])),
        "Semantic groups do not match metadata predictors.",
        problems,
    )
    check(
        [len(group) for group in groups.values()] == [32, 23, 12, 10],
        "Unexpected semantic-group sizes.",
        problems,
    )
    check(
        set(protocol.get("confirmatory_seeds", [])) == EXPECTED_SEEDS,
        "Frozen XGBoost seed set mismatch.",
        problems,
    )
    check(protocol.get("predictor_count") == 77, "Frozen predictor count mismatch.", problems)
    check(protocol.get("semantic_group_count") == 4, "Frozen group count mismatch.", problems)
    expected_rows = {
        "training": 598043,
        "validation": 152964,
        "calibration": 99084,
        "source_test": 150178,
        "target_test": 999143,
    }
    check(
        metadata.get("partition_rows") == expected_rows,
        "Metadata partition sizes changed.",
        problems,
    )
    mapping = renaming.get("target_to_source_column_names", {})
    check(len(mapping) == 50, "Feature-renaming map should contain 50 entries.", problems)
    check(
        renaming.get("target_columns_dropped")
        == ["Protocol", "Timestamp", "Flow ID", "Src IP", "Dst IP", "Src Port"],
        "Unexpected target-column drop list.",
        problems,
    )
    return {
        "predictors": len(flattened),
        "group_sizes": [len(group) for group in groups.values()],
        "renamed_target_columns": len(mapping),
        "seeds": sorted(EXPECTED_SEEDS),
    }


def audit_result_family(
    folder: Path,
    nonselective_name: str,
    selective_name: str,
    paired_name: str,
    models: set[str],
    problems: list[str],
) -> dict:
    paths = [folder / nonselective_name, folder / selective_name, folder / paired_name]
    if not all(path.exists() for path in paths):
        for path in paths:
            check(path.exists(), f"Missing {path.relative_to(ROOT)}", problems)
        return {}
    nonselective, selective, paired = [pd.read_csv(path) for path in paths]
    check(len(nonselective) == 220, f"{nonselective_name}: expected 220 rows.", problems)
    check(len(selective) == 660, f"{selective_name}: expected 660 rows.", problems)
    check(len(paired) == 110, f"{paired_name}: expected 110 rows.", problems)
    for frame, name in (
        (nonselective, nonselective_name),
        (selective, selective_name),
        (paired, paired_name),
    ):
        check(set(frame.seed.astype(int)) == EXPECTED_SEEDS, f"{name}: seed mismatch.", problems)
        check(set(frame.dataset) == {"source", "target"}, f"{name}: dataset mismatch.", problems)
    check(set(nonselective.model) == models, f"{nonselective_name}: model mismatch.", problems)
    check(set(selective.model) == models, f"{selective_name}: model mismatch.", problems)
    expected_conditions = {"complete", "single", "pairwise_unseen"}
    check(
        set(nonselective.condition_type) == expected_conditions,
        f"{nonselective_name}: condition types mismatch.",
        problems,
    )
    check(
        set(selective.desired_policy_coverage.round(2)) == {0.8, 0.9, 0.95},
        f"{selective_name}: coverage policies mismatch.",
        problems,
    )
    return {
        "nonselective_rows": len(nonselective),
        "selective_rows": len(selective),
        "paired_rows": len(paired),
    }


def audit_ablation(problems: list[str]) -> dict:
    folder = ROOT / "results" / "ablation"
    nonselective = pd.read_csv(folder / "component_ablation_nonselective_all.csv")
    selective = pd.read_csv(folder / "component_ablation_selective_all.csv")
    summary = pd.read_csv(folder / "component_ablation_primary_summary.csv")
    expected_models = {
        "ordinary_baseline",
        "indicators_only",
        "augmentation_only",
        "full_method",
    }
    check(len(nonselective) == 440, "Ablation non-selective row count mismatch.", problems)
    check(len(selective) == 1320, "Ablation selective row count mismatch.", problems)
    check(set(nonselective.seed.astype(int)) == EXPECTED_SEEDS, "Ablation seed mismatch.", problems)
    check(set(nonselective.model) == expected_models, "Ablation model mismatch.", problems)
    check(len(summary) == 88, "Ablation summary row count mismatch.", problems)
    return {
        "nonselective_rows": len(nonselective),
        "selective_rows": len(selective),
        "summary_rows": len(summary),
    }


def audit_provenance(problems: list[str]) -> dict:
    component_dir = ROOT / "protocols" / "component_ablation"
    component_protocol = load_json(
        component_dir / "component_ablation_protocol_frozen.json", problems
    )
    component_complete = load_json(
        component_dir / "COMPONENT_ABLATION_COMPLETE.json", problems
    )
    component_audit = load_json(
        component_dir / "corrected_ablation_manifest_audit.json", problems
    )
    check(
        component_protocol.get("status") == "corrected and frozen before ablation retraining",
        "Corrected component protocol does not report the frozen status.",
        problems,
    )
    check(
        component_complete.get("status") == "component_ablation_complete",
        "Component-ablation completion marker mismatch.",
        problems,
    )
    check(
        component_audit.get("audit_status") == "PASS",
        "Corrected component-ablation manifest audit did not pass.",
        problems,
    )
    for name, details in component_audit.get("combined_outputs", {}).items():
        result_path = ROOT / "results" / "ablation" / name
        check(result_path.exists(), f"Missing audited ablation output {name}.", problems)
        if result_path.exists():
            check(
                sha256(result_path) == details.get("sha256"),
                f"Ablation SHA-256 mismatch for {name}.",
                problems,
            )

    rf_dir = ROOT / "protocols" / "random_forest"
    rf_protocol = load_json(rf_dir / "random_forest_protocol_frozen.json", problems)
    rf_pretraining = load_json(rf_dir / "random_forest_pretraining_audit.json", problems)
    rf_complete = load_json(
        rf_dir / "RANDOM_FOREST_FIVE_SEED_EVALUATION_COMPLETE.json", problems
    )
    check(
        rf_protocol.get("status") == "FROZEN_BEFORE_TRAINING",
        "Random Forest protocol was not frozen before training.",
        problems,
    )
    check(rf_pretraining.get("status") == "PASS", "Random Forest pretraining audit failed.", problems)
    check(rf_complete.get("status") == "complete", "Random Forest completion marker mismatch.", problems)
    for name, expected_hash in rf_complete.get("result_sha256", {}).items():
        result_path = ROOT / "results" / "random_forest" / name
        check(result_path.exists(), f"Missing audited Random Forest output {name}.", problems)
        if result_path.exists():
            check(
                sha256(result_path) == expected_hash,
                f"Random Forest SHA-256 mismatch for {name}.",
                problems,
            )
    return {
        "component_ablation_audit": component_audit.get("audit_status"),
        "component_models_audited": component_audit.get("model_rows_found"),
        "random_forest_pretraining_audit": rf_pretraining.get("status"),
        "random_forest_frozen_artifacts": len(rf_pretraining.get("authoritative_file_audit", [])),
    }


def audit_tables(require_tables: bool, problems: list[str]) -> dict:
    if not require_tables:
        return {"checked": False}
    expected_rows = {6: 8, 7: 6, 8: 6, 9: 6, 10: 4}
    frames: dict[int, pd.DataFrame] = {}
    for number, rows in expected_rows.items():
        path = ROOT / "manuscript_tables" / f"Table_{number}.csv"
        check(path.exists(), f"Missing reconstructed Table_{number}.csv.", problems)
        if path.exists():
            frame = pd.read_csv(path)
            frames[number] = frame
            check(len(frame) == rows, f"Table_{number}: expected {rows} rows.", problems)

    if 6 in frames:
        table = frames[6]
        lookup = table.set_index(["Dataset", "Model"])
        checks = {
            ("Source", "Baseline", "Pairwise F1"): 0.629341337296,
            ("Source", "Mask augmentation", "Pairwise F1"): 0.891013362723,
            ("Target", "Baseline", "Pairwise F1"): 0.118917120836,
            ("Target", "Mask augmentation", "Pairwise F1"): 0.250810435150,
        }
        for (dataset, model, column), expected in checks.items():
            value = float(lookup.loc[(dataset, model), column])
            check(math.isclose(value, expected, abs_tol=5e-10), f"Table 6 mismatch: {dataset}/{model}/{column}", problems)
    if 7 in frames:
        first = frames[7].iloc[0]
        check(math.isclose(float(first["Mean difference"]), 0.261672025427, abs_tol=5e-10), "Table 7 primary effect mismatch.", problems)
        check(math.isclose(float(first["Exact Wilcoxon p"]), 0.0625, abs_tol=1e-12), "Table 7 Wilcoxon value mismatch.", problems)
    if 8 in frames:
        lookup = frames[8].set_index("Loss")
        check(math.isclose(float(lookup.loc["G2+G4", "Target delta"]), -0.311373416775, abs_tol=5e-10), "Table 8 negative target reversal mismatch.", problems)
    if 10 in frames:
        table = frames[10].set_index(["Model", "Dataset"])
        check(math.isclose(float(table.loc[("Random Forest", "Target"), "F1 difference"]), -0.0351322932022, abs_tol=5e-10), "Table 10 Random Forest target result mismatch.", problems)
    return {"checked": True, "row_counts": expected_rows}


def audit_files(require_figures: bool, problems: list[str]) -> dict:
    expected_scripts = {
        "prepare_data.py",
        "train_xgboost.py",
        "evaluate_xgboost.py",
        "frozen_component_ablation.py",
        "audit_component_ablation_models.py",
        "prepare_random_forest_replication.py",
        "train_random_forest_five_seed.py",
        "evaluate_random_forest_five_seed.py",
        "aggregate_analysis.py",
        "generate_figures.py",
        "verify_repository.py",
    }
    present_scripts = {path.name for path in (ROOT / "scripts").glob("*.py")}
    check(
        expected_scripts <= present_scripts,
        f"Missing scripts: {sorted(expected_scripts - present_scripts)}",
        problems,
    )
    figure_files = []
    if require_figures:
        for number in range(1, 6):
            for extension in ("pdf", "png"):
                path = ROOT / "figures" / f"Figure_{number}.{extension}"
                check(path.exists() and path.stat().st_size > 10_000, f"Missing or invalid {path.name}.", problems)
                if path.exists():
                    figure_files.append(path.name)
    return {"scripts": len(present_scripts), "figures_checked": figure_files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-figures", action="store_true")
    parser.add_argument("--require-tables", action="store_true")
    args = parser.parse_args()
    problems: list[str] = []
    report = {
        "configuration": audit_configuration(problems),
        "xgboost": audit_result_family(
            ROOT / "results" / "xgboost",
            "five_seed_nonselective_all.csv",
            "five_seed_selective_all.csv",
            "five_seed_paired_counts_all.csv",
            {"ordinary_baseline", "group_aware"},
            problems,
        ),
        "random_forest": audit_result_family(
            ROOT / "results" / "random_forest",
            "random_forest_five_seed_nonselective_all.csv",
            "random_forest_five_seed_selective_all.csv",
            "random_forest_five_seed_paired_counts_all.csv",
            {"ordinary_random_forest", "group_aware_random_forest"},
            problems,
        ),
        "ablation": audit_ablation(problems),
        "provenance": audit_provenance(problems),
        "tables": audit_tables(args.require_tables, problems),
        "files": audit_files(args.require_figures, problems),
    }
    report["status"] = "PASS" if not problems else "FAIL"
    report["problems"] = problems
    (ROOT / "REPOSITORY_AUDIT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
