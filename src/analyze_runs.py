from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
# Update this if your project root lives somewhere else.
PROJECT_ROOT = Path(r"C:/Users/kille/Documents/Spring 2026/CSCE_Project/mtg_milligan_modeling")
RUNS_DIR = PROJECT_ROOT / "models" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "ablation_analysis"

RUN_IDS = [
    "run_20260421_202628_463b7101",
    "run_20260421_202631_e133f3f1",
    "run_20260421_202636_3f13735a",
    "run_20260421_202637_daae015d",
    "run_20260421_202637_e8e3681a",
    "run_20260421_202639_12636c65",
    "run_20260421_202641_a8720d78",
    "run_20260421_202642_7e40fb89",
    "run_20260421_202643_25361885",
    "run_20260421_202647_992ca428",
    "run_20260421_202648_cda7e86b",
    "run_20260421_202649_afb1d35f",
]

MODEL_ORDER = ["logreg_l1", "xgboost", "lightgbm"]
EXPERIMENT_ORDER = ["full", "no_mana_buckets", "land_only", "cards_only"]


# =========================================================
# IO HELPERS
# =========================================================
def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# DATA INGEST
# =========================================================
def load_run_record(run_id: str) -> dict:
    run_dir = RUNS_DIR / run_id
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "metadata.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for {run_id}: {metrics_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json for {run_id}: {metadata_path}")

    metrics = read_json(metrics_path)
    metadata = read_json(metadata_path)

    record = {
        "run_id": run_id,
        "dataset_id": metadata.get("dataset_id", metrics.get("dataset_id")),
        "model_type": metadata.get("model_type", metrics.get("model_type")),
        "experiment_id": metadata.get("experiment_id", metrics.get("experiment_id")),
        "n_rows": metrics.get("n_rows"),
        "n_features": metrics.get("n_features"),
        "accuracy": metrics.get("accuracy"),
        "roc_auc": metrics.get("roc_auc"),
        "log_loss": metrics.get("log_loss"),
        "accuracy_mean": metrics.get("accuracy_mean"),
        "accuracy_std": metrics.get("accuracy_std"),
        "roc_auc_mean": metrics.get("roc_auc_mean"),
        "roc_auc_std": metrics.get("roc_auc_std"),
        "log_loss_mean": metrics.get("log_loss_mean"),
        "log_loss_std": metrics.get("log_loss_std"),
        "oof_log_loss": metrics.get("oof_log_loss"),
        "balanced_accuracy_at_chosen_threshold": metrics.get("balanced_accuracy_at_chosen_threshold"),
        "chosen_threshold": metrics.get("chosen_threshold"),
        "baseline_log_loss": metrics.get("baseline_log_loss"),
        "baseline_keep_rate": metrics.get("baseline_keep_rate"),
        "n_selected_features": metrics.get("n_selected_features", metadata.get("n_selected_features")),
        "n_zero_features": metrics.get("n_zero_features", metadata.get("n_zero_features")),
        "feature_report_kind": metrics.get("feature_report_kind", metadata.get("feature_report_kind")),
    }

    selected_feature_columns = metrics.get("selected_feature_columns")
    if isinstance(selected_feature_columns, list):
        record["selected_feature_columns_count"] = len(selected_feature_columns)
    else:
        record["selected_feature_columns_count"] = None

    return record


def load_all_runs(run_ids: Iterable[str]) -> pd.DataFrame:
    rows = [load_run_record(run_id) for run_id in run_ids]
    df = pd.DataFrame(rows)

    df["model_type"] = pd.Categorical(df["model_type"], categories=MODEL_ORDER, ordered=True)
    df["experiment_id"] = pd.Categorical(df["experiment_id"], categories=EXPERIMENT_ORDER, ordered=True)
    df = df.sort_values(["model_type", "experiment_id"]).reset_index(drop=True)
    return df


# =========================================================
# TEXT REPORTING
# =========================================================
def format_num(x, decimals: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{decimals}f}"


def build_summary_text(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("Ablation Analysis Summary")
    lines.append("=========================")
    lines.append("")

    if df.empty:
        lines.append("No runs loaded.")
        return "\n".join(lines)

    lines.append(f"Loaded {len(df)} runs")
    lines.append(f"Dataset IDs: {', '.join(sorted(df['dataset_id'].dropna().astype(str).unique()))}")
    lines.append("")

    # Overall rankings
    best_auc = df.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_logloss = df.sort_values("log_loss_mean", ascending=True).iloc[0]
    best_balacc = df.sort_values("balanced_accuracy_at_chosen_threshold", ascending=False).iloc[0]

    lines.append("Top runs by metric")
    lines.append("------------------")
    lines.append(
        f"Best ROC-AUC mean: {best_auc['model_type']} / {best_auc['experiment_id']} "
        f"({format_num(best_auc['roc_auc_mean'])})"
    )
    lines.append(
        f"Best log loss mean: {best_logloss['model_type']} / {best_logloss['experiment_id']} "
        f"({format_num(best_logloss['log_loss_mean'])})"
    )
    lines.append(
        f"Best balanced accuracy: {best_balacc['model_type']} / {best_balacc['experiment_id']} "
        f"({format_num(best_balacc['balanced_accuracy_at_chosen_threshold'])})"
    )
    lines.append("")

    # Per-model best experiment
    lines.append("Best experiment within each model family")
    lines.append("---------------------------------------")
    for model_type in MODEL_ORDER:
        sub = df[df["model_type"] == model_type].copy()
        if sub.empty:
            continue
        best = sub.sort_values("roc_auc_mean", ascending=False).iloc[0]
        lines.append(
            f"{model_type}: best by ROC-AUC is {best['experiment_id']} "
            f"(AUC={format_num(best['roc_auc_mean'])}, "
            f"logloss={format_num(best['log_loss_mean'])}, "
            f"bal_acc={format_num(best['balanced_accuracy_at_chosen_threshold'])})"
        )
    lines.append("")

    # Per-experiment best model
    lines.append("Best model within each experiment")
    lines.append("-------------------------------")
    for experiment_id in EXPERIMENT_ORDER:
        sub = df[df["experiment_id"] == experiment_id].copy()
        if sub.empty:
            continue
        best = sub.sort_values("roc_auc_mean", ascending=False).iloc[0]
        lines.append(
            f"{experiment_id}: best by ROC-AUC is {best['model_type']} "
            f"(AUC={format_num(best['roc_auc_mean'])}, "
            f"logloss={format_num(best['log_loss_mean'])}, "
            f"bal_acc={format_num(best['balanced_accuracy_at_chosen_threshold'])})"
        )
    lines.append("")

    # Logistic-specific ablation deltas relative to full
    lines.append("Delta vs full feature set")
    lines.append("-------------------------")
    for model_type in MODEL_ORDER:
        sub = df[df["model_type"] == model_type].copy()
        if sub.empty:
            continue
        full_rows = sub[sub["experiment_id"] == "full"]
        if full_rows.empty:
            continue
        full = full_rows.iloc[0]
        lines.append(f"{model_type}:")
        for _, row in sub.iterrows():
            if row["experiment_id"] == "full":
                continue
            delta_auc = row["roc_auc_mean"] - full["roc_auc_mean"]
            delta_logloss = row["log_loss_mean"] - full["log_loss_mean"]
            delta_balacc = row["balanced_accuracy_at_chosen_threshold"] - full["balanced_accuracy_at_chosen_threshold"]
            lines.append(
                f"  {row['experiment_id']}: "
                f"ΔAUC={delta_auc:+.3f}, "
                f"Δlogloss={delta_logloss:+.3f}, "
                f"Δbal_acc={delta_balacc:+.3f}"
            )
        lines.append("")

    return "\n".join(lines)


# =========================================================
# TABLES
# =========================================================
def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model_type",
        "experiment_id",
        "roc_auc_mean",
        "log_loss_mean",
        "accuracy_mean",
        "balanced_accuracy_at_chosen_threshold",
        "n_features",
        "n_selected_features",
        "chosen_threshold",
        "run_id",
    ]
    out = df[cols].copy()
    out = out.rename(columns={
        "model_type": "model",
        "experiment_id": "experiment",
        "roc_auc_mean": "cv_auc",
        "log_loss_mean": "cv_log_loss",
        "accuracy_mean": "cv_accuracy",
        "balanced_accuracy_at_chosen_threshold": "balanced_accuracy",
    })
    return out


# =========================================================
# PLOTTING
# =========================================================
def _save_plot(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_by_experiment(df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    pivot = df.pivot(index="experiment_id", columns="model_type", values=metric)
    pivot = pivot.reindex(EXPERIMENT_ORDER)
    pivot = pivot[MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"{ylabel} by Ablation and Model")
    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    _save_plot(fig, filename)


def plot_metric_by_model(df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    pivot = df.pivot(index="model_type", columns="experiment_id", values=metric)
    pivot = pivot.reindex(MODEL_ORDER)
    pivot = pivot[EXPERIMENT_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"{ylabel} by Model and Ablation")
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.legend(title="Experiment")
    ax.grid(axis="y", alpha=0.3)
    _save_plot(fig, filename)


def plot_auc_vs_logloss(df: pd.DataFrame, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_type in MODEL_ORDER:
        sub = df[df["model_type"] == model_type]
        ax.scatter(sub["roc_auc_mean"], sub["log_loss_mean"], label=model_type, s=70)
        for _, row in sub.iterrows():
            ax.annotate(
                str(row["experiment_id"]),
                (row["roc_auc_mean"], row["log_loss_mean"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_title("Ablation Runs: ROC-AUC vs Log Loss")
    ax.set_xlabel("Cross-validated ROC-AUC")
    ax.set_ylabel("Cross-validated Log Loss")
    ax.legend(title="Model")
    ax.grid(alpha=0.3)
    _save_plot(fig, filename)


def plot_selected_features(df: pd.DataFrame, filename: str) -> None:
    sub = df.dropna(subset=["n_selected_features"]).copy()
    if sub.empty:
        return

    pivot = sub.pivot(index="experiment_id", columns="model_type", values="n_selected_features")
    pivot = pivot.reindex(EXPERIMENT_ORDER)
    pivot = pivot[MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Selected / Nonzero Features by Ablation and Model")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Selected features")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    _save_plot(fig, filename)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_runs(RUN_IDS)
    comparison = build_comparison_table(df)
    summary_text = build_summary_text(df)

    # Save tables
    df.to_csv(OUTPUT_DIR / "ablation_runs_raw.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "ablation_comparison_table.csv", index=False)

    # Save text summary
    with open(OUTPUT_DIR / "ablation_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Print to terminal
    print(summary_text)
    print("\nComparison table\n================")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(comparison.to_string(index=False))

    # Save plots
    plot_metric_by_experiment(df, "roc_auc_mean", "Cross-validated ROC-AUC", "auc_by_experiment.png")
    plot_metric_by_experiment(df, "log_loss_mean", "Cross-validated Log Loss", "logloss_by_experiment.png")
    plot_metric_by_experiment(df, "balanced_accuracy_at_chosen_threshold", "Balanced Accuracy", "balanced_accuracy_by_experiment.png")
    plot_metric_by_model(df, "roc_auc_mean", "Cross-validated ROC-AUC", "auc_by_model.png")
    plot_auc_vs_logloss(df, "auc_vs_logloss_scatter.png")
    plot_selected_features(df, "selected_features_by_experiment.png")

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
