"""Reconstruct the five figures reported in the JISA manuscript.

The script uses only committed aggregate result tables. It performs no model
training and never reads the source or sealed target flow records. Both vector
PDF and 300-dpi PNG files are written to ``figures/``.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nids-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "figures"

GROUPS = [
    "G1_Packet_Byte_Magnitude",
    "G2_Temporal_Activity",
    "G3_TCP_Control_Flags",
    "G4_Rate_Header_Flow_Context",
]
PAIRWISE = [
    f"{GROUPS[0]}+{GROUPS[1]}",
    f"{GROUPS[0]}+{GROUPS[2]}",
    f"{GROUPS[0]}+{GROUPS[3]}",
    f"{GROUPS[1]}+{GROUPS[2]}",
    f"{GROUPS[1]}+{GROUPS[3]}",
    f"{GROUPS[2]}+{GROUPS[3]}",
]
CONDITIONS = [None, *GROUPS, *PAIRWISE]
CONDITION_LABELS = [
    "Complete",
    "G1",
    "G2",
    "G3",
    "G4",
    "G1+G2",
    "G1+G3",
    "G1+G4",
    "G2+G3",
    "G2+G4",
    "G3+G4",
]

MODEL_ORDER = ["ordinary_baseline", "augmentation_only", "full_method"]
MODEL_LABELS = {
    "ordinary_baseline": "Baseline",
    "indicators_only": "Indicators only",
    "augmentation_only": "Mask augmentation",
    "full_method": "Augmentation + indicators",
}
COLORS = {
    "ordinary_baseline": "#4C78A8",
    "indicators_only": "#9C755F",
    "augmentation_only": "#F28E2B",
    "full_method": "#59A14F",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(figure: plt.Figure, number: int) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        final_path = OUTPUT / f"Figure_{number}.{extension}"
        temporary_path = OUTPUT / f".Figure_{number}.{extension}.tmp"
        with temporary_path.open("wb") as stream:
            figure.savefig(
                stream,
                format=extension,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, final_path)
    plt.close(figure)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def figure_1_workflow() -> None:
    """Figure 1: source-only development and frozen evaluation protocol."""

    figure, axis = plt.subplots(figsize=(10.8, 2.9))
    axis.set_xlim(0, 10.8)
    axis.set_ylim(0, 3.0)
    axis.axis("off")

    boxes = [
        (0.25, "Source flow\nCICIDS2017", "Source-only\ndevelopment", "#6C63FF"),
        (2.35, "Schema and\ngroups", "77 predictors\n4 semantic groups", "#333333"),
        (4.45, "Mask\naugmentation", "Complete and one\nrandom single-group mask", "#E69F00"),
        (6.55, "Model and policy", "XGBoost, Platt\nscaling, source thresholds", "#2CA02C"),
        (8.65, "Frozen evaluation", "Complete, single\nand unseen pairwise losses", "#4C6FFF"),
    ]
    width, height, y = 1.75, 1.25, 0.80
    for x, title, body, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05,rounding_size=0.16",
            linewidth=1.4,
            edgecolor="#9AA0A6",
            facecolor="white",
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + 0.82,
            title,
            ha="center",
            va="center",
            color=color,
            weight="bold",
            fontsize=8.5,
        )
        axis.text(x + width / 2, y + 0.30, body, ha="center", va="center", fontsize=7)
    for left, right in zip(boxes[:-1], boxes[1:]):
        axis.annotate(
            "",
            xy=(right[0], y + height / 2),
            xytext=(left[0] + width, y + height / 2),
            arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "#3C4043"},
        )

    axis.text(
        5.4,
        2.73,
        "Target labels never enter training, calibration, classification-threshold selection, "
        "or abstention-policy selection",
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
    )
    axis.annotate(
        "",
        xy=(9.52, 2.08),
        xytext=(9.52, 2.56),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#3C4043"},
    )
    axis.text(
        9.52,
        0.46,
        "External target: CSE-CIC-IDS2018 evaluation only",
        ha="center",
        va="center",
        fontsize=7.2,
        weight="bold",
    )
    save_figure(figure, 1)


def condition_mask(frame: pd.DataFrame, condition: str | None) -> pd.Series:
    if condition is None:
        return frame["condition_type"].eq("complete")
    return frame["missing_groups"].eq(condition)


def figure_2_condition_level_f1() -> None:
    """Figure 2: every complete, single-loss, and pairwise-loss condition."""

    path = RESULTS / "ablation" / "component_ablation_nonselective_all.csv"
    frame = pd.read_csv(path)
    require_columns(
        frame,
        {"seed", "dataset", "model", "condition_type", "missing_groups", "f1"},
        path.name,
    )
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 6.6), sharex=True)
    x = np.arange(len(CONDITIONS))
    width = 0.25
    offsets = [-width, 0, width]

    for axis, dataset, title in zip(
        axes,
        ["source", "target"],
        ["CICIDS2017 source test", "CSE-CIC-IDS2018 target test"],
    ):
        for offset, model in zip(offsets, MODEL_ORDER):
            means, errors = [], []
            for condition in CONDITIONS:
                values = frame.loc[
                    frame["dataset"].eq(dataset)
                    & frame["model"].eq(model)
                    & condition_mask(frame, condition),
                    "f1",
                ].astype(float)
                if len(values) != 5:
                    raise ValueError(
                        f"Expected five values for {dataset}/{model}/{condition}; found {len(values)}"
                    )
                means.append(values.mean())
                errors.append(values.std(ddof=1))
            axis.bar(
                x + offset,
                means,
                width,
                yerr=errors,
                capsize=2,
                color=COLORS[model],
                edgecolor="white",
                linewidth=0.4,
                label=MODEL_LABELS[model],
            )
        axis.set_title(title, loc="left", weight="bold")
        axis.set_ylabel("Attack-class F1")
        axis.set_ylim(0, 1.04 if dataset == "source" else 0.62)
        axis.grid(axis="y", alpha=0.25)
        axis.axvline(4.5, color="#999999", linestyle="--", linewidth=0.8)

    axes[0].legend(loc="lower left", ncol=3, frameon=False)
    axes[1].set_xticks(x, CONDITION_LABELS, rotation=30, ha="right")
    axes[1].set_xlabel("Unavailable semantic feature groups")
    figure.suptitle(
        "Complete, single-loss and unseen pairwise-loss performance",
        weight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.005,
        "Bars show five-seed means; error bars show standard deviation. Dashed lines separate complete, single and pairwise conditions.",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_figure(figure, 2)


def pairwise_seed_means(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.loc[frame["condition_type"].eq("pairwise_unseen")]
        .groupby(["seed", "dataset", "model"], as_index=False)["f1"]
        .mean()
    )


def figure_3_component_ablation() -> None:
    """Figure 3: four-component ablation over the unseen pairwise endpoint."""

    frame = pd.read_csv(
        RESULTS / "ablation" / "component_ablation_nonselective_all.csv"
    )
    seed_means = pairwise_seed_means(frame)
    models = [
        "ordinary_baseline",
        "indicators_only",
        "augmentation_only",
        "full_method",
    ]
    datasets = ["source", "target"]
    x = np.arange(2)
    width = 0.19
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    figure, axis = plt.subplots(figsize=(9.6, 4.7))

    for offset, model in zip(offsets, models):
        means, errors = [], []
        for dataset in datasets:
            values = seed_means.loc[
                seed_means["dataset"].eq(dataset) & seed_means["model"].eq(model),
                "f1",
            ].astype(float)
            if len(values) != 5:
                raise ValueError(f"Expected five seed means for {dataset}/{model}.")
            means.append(values.mean())
            errors.append(values.std(ddof=1))
        axis.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=COLORS[model],
            edgecolor="white",
            label=MODEL_LABELS[model],
        )

    axis.set_xticks(x, ["Source", "External target"])
    axis.set_ylabel("Mean attack-class F1 across six pairwise losses")
    axis.set_ylim(0, 1.03)
    axis.set_title(
        "Component ablation under unseen pairwise feature-group loss",
        weight="bold",
    )
    axis.legend(loc="upper center", ncol=2, frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.5,
        0.015,
        "Bars show the mean of six pairwise conditions within each seed, then the mean across five seeds; error bars show standard deviation across seeds.",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    save_figure(figure, 3)


def figure_4_condition_delta_heatmap() -> None:
    """Figure 4: augmentation-only minus baseline F1 for each pairwise loss."""

    frame = pd.read_csv(
        RESULTS / "ablation" / "component_ablation_nonselective_all.csv"
    )
    selected = frame.loc[frame["condition_type"].eq("pairwise_unseen")].copy()
    pivot = selected.pivot_table(
        index=["seed", "dataset", "missing_groups"],
        columns="model",
        values="f1",
        aggfunc="mean",
    ).reset_index()
    pivot["delta"] = pivot["augmentation_only"] - pivot["ordinary_baseline"]
    means = pivot.groupby(["dataset", "missing_groups"])["delta"].mean()
    matrix = pd.DataFrame(
        [[means.loc[(dataset, pair)] for pair in PAIRWISE] for dataset in ["source", "target"]],
        index=["Source", "External target"],
        columns=CONDITION_LABELS[5:],
    )

    figure, axis = plt.subplots(figsize=(9.8, 2.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="+.3f",
        cmap="RdBu_r",
        center=0,
        vmin=-0.34,
        vmax=0.54,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "F1 difference (augmentation - baseline)"},
        ax=axis,
    )
    axis.set_xlabel("Unseen pairwise missing-group condition")
    axis.set_ylabel("")
    axis.set_title(
        "Mask-augmentation benefit is condition dependent under dataset shift",
        weight="bold",
    )
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    save_figure(figure, 4)


def selective_pairwise_curve(frame: pd.DataFrame) -> pd.DataFrame:
    pairwise = frame.loc[frame["condition_type"].eq("pairwise_unseen")].copy()
    return (
        pairwise.groupby(
            ["seed", "dataset", "model", "desired_policy_coverage"], as_index=False
        )[
            ["actual_coverage", "selective_risk", "accepted_f1"]
        ]
        .mean()
        .groupby(["dataset", "model", "desired_policy_coverage"], as_index=False)
        .mean(numeric_only=True)
    )


def figure_5_selective_prediction() -> None:
    """Figure 5: source-selected selective policies under pairwise loss."""

    path = RESULTS / "ablation" / "component_ablation_selective_all.csv"
    frame = pd.read_csv(path)
    require_columns(
        frame,
        {
            "seed",
            "dataset",
            "model",
            "condition_type",
            "desired_policy_coverage",
            "actual_coverage",
            "selective_risk",
            "accepted_f1",
        },
        path.name,
    )
    curves = selective_pairwise_curve(frame)
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    panels = [
        (axes[0, 0], "source", "selective_risk", "Source risk-coverage", "Selective risk (log scale)"),
        (axes[0, 1], "source", "accepted_f1", "Source accepted F1", "Accepted attack-class F1"),
        (axes[1, 0], "target", "selective_risk", "External target risk-coverage", "Selective risk (log scale)"),
        (axes[1, 1], "target", "accepted_f1", "External target accepted F1", "Accepted attack-class F1"),
    ]
    for axis, dataset, metric, title, ylabel in panels:
        for model in MODEL_ORDER:
            part = curves.loc[
                curves["dataset"].eq(dataset) & curves["model"].eq(model)
            ].sort_values("desired_policy_coverage")
            axis.plot(
                part["actual_coverage"],
                part[metric],
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=COLORS[model],
                label=MODEL_LABELS[model],
            )
        if metric == "selective_risk":
            axis.set_yscale("log")
        axis.set_title(title, weight="bold")
        axis.set_xlabel("Actual coverage")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)

    axes[0, 0].legend(loc="best", frameon=False)
    figure.suptitle(
        "Source-selected abstention policies do not transfer reliable attack detection",
        weight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(figure, 5)


def main() -> None:
    configure_style()
    figure_1_workflow()
    figure_2_condition_level_f1()
    figure_3_component_ablation()
    figure_4_condition_delta_heatmap()
    figure_5_selective_prediction()
    print("Generated manuscript-matching Figure_1 through Figure_5 in", OUTPUT)


if __name__ == "__main__":
    main()
