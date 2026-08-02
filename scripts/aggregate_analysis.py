"""Reconstruct manuscript Tables 6--10 from committed five-seed results.

Outputs are written to ``manuscript_tables/``. The calculations follow the
manuscript's fixed order: condition values are first averaged within seed, and
five seed-level values are then used for inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "manuscript_tables"
SEEDS = [2027, 2028, 2029, 2030, 2031]
T_CRITICAL_DF4_95 = float(stats.t.ppf(0.975, df=4))

MODEL_LABELS = {
    "ordinary_baseline": "Baseline",
    "indicators_only": "Indicators only",
    "augmentation_only": "Mask augmentation",
    "full_method": "Aug. + indicators",
}


def require_five(values: pd.Series, label: str) -> pd.Series:
    values = values.astype(float).sort_index()
    if len(values) != 5:
        raise ValueError(f"{label}: expected five seed-level values, found {len(values)}")
    return values


def pairwise_seed_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        frame.loc[frame["condition_type"].eq("pairwise_unseen")]
        .groupby(["seed", "dataset", "model"], as_index=False)[metric]
        .mean()
    )


def table_6(ablation: pd.DataFrame) -> pd.DataFrame:
    seed_condition = (
        ablation.groupby(
            ["seed", "dataset", "model", "condition_type"], as_index=False
        )["f1"]
        .mean()
    )
    means = (
        seed_condition.groupby(["dataset", "model", "condition_type"])["f1"]
        .mean()
        .unstack("condition_type")
        .reset_index()
    )
    order = ["ordinary_baseline", "indicators_only", "augmentation_only", "full_method"]
    means["model_order"] = means["model"].map({value: i for i, value in enumerate(order)})
    means["dataset_order"] = means["dataset"].map({"source": 0, "target": 1})
    means = means.sort_values(["dataset_order", "model_order"])
    means["Dataset"] = means["dataset"].map({"source": "Source", "target": "Target"})
    means["Model"] = means["model"].map(MODEL_LABELS)
    return means[["Dataset", "Model", "complete", "single", "pairwise_unseen"]].rename(
        columns={
            "complete": "Complete F1",
            "single": "Single-loss F1",
            "pairwise_unseen": "Pairwise F1",
        }
    )


def paired_comparison(
    seed_means: pd.DataFrame,
    dataset: str,
    left: str,
    right: str,
    label: str,
) -> dict:
    selected = seed_means.loc[seed_means["dataset"].eq(dataset)]
    pivot = selected.pivot(index="seed", columns="model", values="f1")
    differences = require_five(pivot[left] - pivot[right], f"{dataset}/{label}")
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    margin = T_CRITICAL_DF4_95 * sd / np.sqrt(5)
    t_result = stats.ttest_1samp(differences, popmean=0.0)
    w_result = stats.wilcoxon(differences, zero_method="wilcox", method="exact")
    return {
        "Dataset": dataset.capitalize(),
        "Comparison": label,
        "Mean difference": mean,
        "CI95 lower": mean - margin,
        "CI95 upper": mean + margin,
        "Paired t p": float(t_result.pvalue),
        "Exact Wilcoxon p": float(w_result.pvalue),
        "Wins": int((differences > 0).sum()),
        "Seeds": 5,
    }


def table_7(ablation: pd.DataFrame) -> pd.DataFrame:
    seed_means = pairwise_seed_metric(ablation, "f1")
    comparisons = [
        ("augmentation_only", "ordinary_baseline", "Aug. - baseline"),
        ("indicators_only", "ordinary_baseline", "Ind. - baseline"),
        ("full_method", "augmentation_only", "Full - augmentation"),
    ]
    rows = []
    for left, right, label in comparisons:
        for dataset in ["source", "target"]:
            rows.append(paired_comparison(seed_means, dataset, left, right, label))
    comparison_order = {
        label: i for i, (_, _, label) in enumerate(comparisons)
    }
    for row in rows:
        row["_order"] = comparison_order[row["Comparison"]] * 2 + (
            0 if row["Dataset"] == "Source" else 1
        )
    return pd.DataFrame(rows).sort_values("_order").drop(columns="_order")


def table_8(ablation: pd.DataFrame) -> pd.DataFrame:
    selected = ablation.loc[ablation["condition_type"].eq("pairwise_unseen")]
    means = selected.groupby(["dataset", "missing_groups", "model"])["f1"].mean()
    pair_order = list(dict.fromkeys(selected["missing_groups"].tolist()))
    short = {
        name: "+".join(part.split("_")[0] for part in name.split("+"))
        for name in pair_order
    }
    rows = []
    for pair in pair_order:
        source_baseline = float(means.loc[("source", pair, "ordinary_baseline")])
        source_aug = float(means.loc[("source", pair, "augmentation_only")])
        target_baseline = float(means.loc[("target", pair, "ordinary_baseline")])
        target_aug = float(means.loc[("target", pair, "augmentation_only")])
        rows.append(
            {
                "Loss": short[pair],
                "Source baseline": source_baseline,
                "Source augmentation": source_aug,
                "Source delta": source_aug - source_baseline,
                "Target baseline": target_baseline,
                "Target augmentation": target_aug,
                "Target delta": target_aug - target_baseline,
            }
        )
    return pd.DataFrame(rows)


def table_9(selective: pd.DataFrame) -> pd.DataFrame:
    selected = selective.loc[
        selective["condition_type"].eq("pairwise_unseen")
        & selective["desired_policy_coverage"].round(2).eq(0.90)
        & selective["model"].isin(
            ["ordinary_baseline", "augmentation_only", "full_method"]
        )
    ]
    metrics = [
        "actual_coverage",
        "attack_coverage",
        "selective_risk",
        "accepted_f1",
        "aurc",
    ]
    seed_means = selected.groupby(["seed", "dataset", "model"], as_index=False)[metrics].mean()
    means = seed_means.groupby(["dataset", "model"], as_index=False)[metrics].mean()
    means["dataset_order"] = means["dataset"].map({"source": 0, "target": 1})
    means["model_order"] = means["model"].map(
        {"ordinary_baseline": 0, "augmentation_only": 1, "full_method": 2}
    )
    means = means.sort_values(["dataset_order", "model_order"])
    means["Dataset"] = means["dataset"].map({"source": "Source", "target": "Target"})
    means["Model"] = means["model"].map(MODEL_LABELS)
    return means[
        ["Dataset", "Model", *metrics]
    ].rename(
        columns={
            "actual_coverage": "Coverage",
            "attack_coverage": "Attack coverage",
            "selective_risk": "Risk",
            "accepted_f1": "Accepted F1",
            "aurc": "AURC",
        }
    )


def family_summary(
    nonselective: pd.DataFrame,
    selective: pd.DataFrame,
    family: str,
    baseline_model: str,
    group_model: str,
) -> list[dict]:
    f1_seed = pairwise_seed_metric(nonselective, "f1")
    aurc_seed = (
        selective.loc[selective["condition_type"].eq("pairwise_unseen")]
        .groupby(["seed", "dataset", "model"], as_index=False)["aurc"]
        .mean()
    )
    rows = []
    for dataset in ["source", "target"]:
        f1 = f1_seed.loc[f1_seed["dataset"].eq(dataset)].pivot(
            index="seed", columns="model", values="f1"
        )
        aurc = aurc_seed.loc[aurc_seed["dataset"].eq(dataset)].pivot(
            index="seed", columns="model", values="aurc"
        )
        delta = require_five(f1[group_model] - f1[baseline_model], f"{family}/{dataset}")
        rows.append(
            {
                "Model": family,
                "Dataset": dataset.capitalize(),
                "Baseline F1": float(f1[baseline_model].mean()),
                "Group-aware F1": float(f1[group_model].mean()),
                "F1 difference": float(delta.mean()),
                "AURC improvement": float((aurc[baseline_model] - aurc[group_model]).mean()),
                "F1 wins": int((delta > 0).sum()),
                "Seeds": 5,
            }
        )
    return rows


def table_10(
    xgb_nonselective: pd.DataFrame,
    xgb_selective: pd.DataFrame,
    rf_nonselective: pd.DataFrame,
    rf_selective: pd.DataFrame,
) -> pd.DataFrame:
    rows = family_summary(
        xgb_nonselective,
        xgb_selective,
        "XGBoost",
        "ordinary_baseline",
        "group_aware",
    )
    rows.extend(
        family_summary(
            rf_nonselective,
            rf_selective,
            "Random Forest",
            "ordinary_random_forest",
            "group_aware_random_forest",
        )
    )
    return pd.DataFrame(rows)


def write_table(frame: pd.DataFrame, number: int) -> None:
    path = OUTPUT / f"Table_{number}.csv"
    frame.to_csv(path, index=False, float_format="%.12g")
    print(f"Saved {path.relative_to(ROOT)} ({len(frame)} rows)")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ablation_nonselective = pd.read_csv(
        RESULTS / "ablation" / "component_ablation_nonselective_all.csv"
    )
    ablation_selective = pd.read_csv(
        RESULTS / "ablation" / "component_ablation_selective_all.csv"
    )
    xgb_nonselective = pd.read_csv(
        RESULTS / "xgboost" / "five_seed_nonselective_all.csv"
    )
    xgb_selective = pd.read_csv(
        RESULTS / "xgboost" / "five_seed_selective_all.csv"
    )
    rf_nonselective = pd.read_csv(
        RESULTS / "random_forest" / "random_forest_five_seed_nonselective_all.csv"
    )
    rf_selective = pd.read_csv(
        RESULTS / "random_forest" / "random_forest_five_seed_selective_all.csv"
    )

    tables = {
        6: table_6(ablation_nonselective),
        7: table_7(ablation_nonselective),
        8: table_8(ablation_nonselective),
        9: table_9(ablation_selective),
        10: table_10(
            xgb_nonselective,
            xgb_selective,
            rf_nonselective,
            rf_selective,
        ),
    }
    for number, frame in tables.items():
        write_table(frame, number)

    summary = {
        "status": "PASS",
        "seed_order": SEEDS,
        "aggregation_rule": "average conditions within seed, then analyze five seed-level values",
        "generated_tables": {f"Table_{number}.csv": len(frame) for number, frame in tables.items()},
    }
    (OUTPUT / "RECONSTRUCTION_COMPLETE.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
