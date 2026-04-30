from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(r"C:/Users/kille/Documents/Spring 2026/CSCE_Project/mtg_milligan_modeling")
RUNS_DIR = PROJECT_ROOT / "models" / "runs"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "ablation_analysis"
FIGURE_DIR = PROJECT_ROOT / "figure"

N_MOST_RECENT_RUNS = 12

MODEL_ORDER = ["logreg_l1", "xgboost", "lightgbm"]
EXPERIMENT_ORDER = ["full", "no_mana_buckets", "land_only", "cards_only"]


# =========================================================
# IO HELPERS
# =========================================================
def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_plot(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()

    out1 = OUTPUT_DIR / filename
    out2 = FIGURE_DIR / filename

    fig.savefig(out1, dpi=300, bbox_inches="tight")
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_most_recent_run_ids(n: int = 12) -> list[str]:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"RUNS_DIR does not exist: {RUNS_DIR}")

    run_dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")]

    if not run_dirs:
        raise FileNotFoundError(f"No run folders found in {RUNS_DIR}")

    # Sort by modified time first, newest first.
    run_dirs = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)

    return [p.name for p in run_dirs[:n]]


def find_first_existing(run_dir: Path, candidates: list[str]) -> Optional[Path]:
    for name in candidates:
        path = run_dir / name
        if path.exists():
            return path
    return None


# =========================================================
# RUN LOADING
# =========================================================
def load_run_record(run_id: str) -> dict:
    run_dir = RUNS_DIR / run_id

    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "metadata.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for {run_id}: {metrics_path}")

    metrics = read_json(metrics_path)
    metadata = read_json(metadata_path) if metadata_path.exists() else {}

    record = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "dataset_id": metadata.get("dataset_id", metrics.get("dataset_id")),
        "model_type": metadata.get("model_type", metrics.get("model_type")),
        "experiment_id": metadata.get("experiment_id", metrics.get("experiment_id")),

        "n_rows": metrics.get("n_rows"),
        "n_features": metrics.get("n_features"),

        "accuracy": metrics.get("accuracy"),
        "roc_auc": metrics.get("roc_auc"),
        "log_loss": metrics.get("log_loss"),

        "accuracy_mean": metrics.get("accuracy_mean", metrics.get("accuracy")),
        "accuracy_std": metrics.get("accuracy_std"),
        "roc_auc_mean": metrics.get("roc_auc_mean", metrics.get("roc_auc")),
        "roc_auc_std": metrics.get("roc_auc_std"),
        "log_loss_mean": metrics.get("log_loss_mean", metrics.get("log_loss")),
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

    return record


def load_all_runs(run_ids: Iterable[str]) -> pd.DataFrame:
    rows = []

    for run_id in run_ids:
        try:
            rows.append(load_run_record(run_id))
        except Exception as e:
            print(f"Skipping {run_id}: {e}")

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No runs could be loaded.")

    df["model_type"] = pd.Categorical(df["model_type"], categories=MODEL_ORDER, ordered=True)
    df["experiment_id"] = pd.Categorical(df["experiment_id"], categories=EXPERIMENT_ORDER, ordered=True)

    df = df.sort_values(["model_type", "experiment_id"]).reset_index(drop=True)
    return df


# =========================================================
# TABLES / TEXT
# =========================================================
def format_num(x, decimals: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{decimals}f}"


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

    existing_cols = [c for c in cols if c in df.columns]
    out = df[existing_cols].copy()

    out = out.rename(columns={
        "model_type": "model",
        "experiment_id": "experiment",
        "roc_auc_mean": "cv_auc",
        "log_loss_mean": "cv_log_loss",
        "accuracy_mean": "cv_accuracy",
        "balanced_accuracy_at_chosen_threshold": "balanced_accuracy",
    })

    return out


def build_summary_text(df: pd.DataFrame) -> str:
    lines = []
    lines.append("Ablation Analysis Summary")
    lines.append("=========================")
    lines.append("")
    lines.append(f"Loaded {len(df)} most recent runs")
    lines.append(f"Dataset IDs: {', '.join(sorted(df['dataset_id'].dropna().astype(str).unique()))}")
    lines.append("")

    best_auc = df.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_logloss = df.sort_values("log_loss_mean", ascending=True).iloc[0]

    lines.append("Top runs")
    lines.append("--------")
    lines.append(
        f"Best ROC-AUC: {best_auc['model_type']} / {best_auc['experiment_id']} "
        f"({format_num(best_auc['roc_auc_mean'])})"
    )
    lines.append(
        f"Best log loss: {best_logloss['model_type']} / {best_logloss['experiment_id']} "
        f"({format_num(best_logloss['log_loss_mean'])})"
    )
    lines.append("")

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

            lines.append(
                f"  {row['experiment_id']}: "
                f"ΔAUC={delta_auc:+.3f}, "
                f"Δlogloss={delta_logloss:+.3f}"
            )

        lines.append("")

    return "\n".join(lines)


# =========================================================
# CORE VISUALS
# =========================================================
def plot_ablation_auc(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="experiment_id", columns="model_type", values="roc_auc_mean")
    pivot = pivot.reindex(EXPERIMENT_ORDER)
    pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Ablation Study: ROC-AUC by Feature Set")
    ax.set_xlabel("Feature Set")
    ax.set_ylabel("Cross-validated ROC-AUC")
    ax.set_ylim(0.45, 1.00)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)

    save_plot(fig, "ablation_auc.png")


def plot_ablation_logloss(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="experiment_id", columns="model_type", values="log_loss_mean")
    pivot = pivot.reindex(EXPERIMENT_ORDER)
    pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Ablation Study: Log Loss by Feature Set")
    ax.set_xlabel("Feature Set")
    ax.set_ylabel("Cross-validated Log Loss")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)

    save_plot(fig, "ablation_logloss.png")


def plot_full_model_comparison(df: pd.DataFrame) -> None:
    full = df[df["experiment_id"] == "full"].copy()
    full = full.sort_values("model_type")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(full["model_type"].astype(str), full["roc_auc_mean"])

    ax.set_title("Full Feature Set Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Cross-validated ROC-AUC")
    ax.set_ylim(0.80, 1.00)
    ax.grid(axis="y", alpha=0.3)

    save_plot(fig, "model_comparison_auc.png")


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
def load_feature_importance_for_run(run_id: str) -> Optional[pd.DataFrame]:
    run_dir = RUNS_DIR / run_id

    candidates = [
        "feature_importance.csv",
        "feature_importances.csv",
        "feature_weights.csv",
        "coefficients.csv",
        "top_features.csv",
        "significant_features.csv",
    ]

    path = find_first_existing(run_dir, candidates)

    if path is None:
        csvs = list(run_dir.glob("*.csv"))
        for p in csvs:
            lowered = p.name.lower()
            if "feature" in lowered or "coef" in lowered or "importance" in lowered:
                path = p
                break

    if path is None:
        return None

    df = pd.read_csv(path)

    # Normalize likely column names
    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if c in {"feature", "feature_name", "name", "term"}:
            rename_map[col] = "feature"
        elif c in {"coef", "coefficient", "weight"}:
            rename_map[col] = "importance"
        elif c in {"importance", "gain", "split", "score"}:
            rename_map[col] = "importance"

    df = df.rename(columns=rename_map)

    if "feature" not in df.columns:
        df = df.rename(columns={df.columns[0]: "feature"})

    if "importance" not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return None
        df = df.rename(columns={numeric_cols[0]: "importance"})

    df = df[["feature", "importance"]].dropna()
    df["abs_importance"] = df["importance"].abs()

    return df.sort_values("abs_importance", ascending=False)


def choose_feature_importance_run(df: pd.DataFrame) -> Optional[str]:
    # Prefer full logistic regression for interpretability.
    preferred = df[(df["model_type"] == "logreg_l1") & (df["experiment_id"] == "full")]
    if not preferred.empty:
        return str(preferred.iloc[0]["run_id"])

    # Otherwise any full run.
    full = df[df["experiment_id"] == "full"]
    if not full.empty:
        return str(full.iloc[0]["run_id"])

    return str(df.iloc[0]["run_id"]) if not df.empty else None


def plot_feature_importance(df: pd.DataFrame, top_n: int = 15) -> None:
    run_id = choose_feature_importance_run(df)
    if run_id is None:
        print("No run available for feature importance.")
        return

    imp = load_feature_importance_for_run(run_id)
    if imp is None or imp.empty:
        print(f"No feature importance file found for run {run_id}. Skipping feature importance plot.")
        return

    top = imp.head(top_n).copy()
    top = top.sort_values("abs_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])

    ax.set_title("Top Feature Weights / Importances")
    ax.set_xlabel("Importance / Coefficient")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)

    save_plot(fig, "feature_importance.png")


# =========================================================
# PREDICTION-BASED VISUALS
# =========================================================
def load_predictions_for_run(run_id: str) -> Optional[pd.DataFrame]:
    run_dir = RUNS_DIR / run_id

    candidates = [
        "predictions.csv",
        "oof_predictions.csv",
        "test_predictions.csv",
        "evaluation_predictions.csv",
        "prediction_analysis.csv",
    ]

    path = find_first_existing(run_dir, candidates)

    if path is None:
        csvs = list(run_dir.glob("*.csv"))
        for p in csvs:
            lowered = p.name.lower()
            if "pred" in lowered or "oof" in lowered:
                path = p
                break

    if path is None:
        return None

    df = pd.read_csv(path)

    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if c in {"y_true", "true", "label", "actual", "actual_label", "decision_binary"}:
            rename_map[col] = "y_true"
        elif c in {"y_pred", "pred", "prediction", "predicted_label"}:
            rename_map[col] = "y_pred"
        elif c in {"y_prob", "prob", "probability", "keep_probability", "predicted_probability", "keep_prob"}:
            rename_map[col] = "y_prob"

    df = df.rename(columns=rename_map)

    if "y_true" not in df.columns:
        return None

    if "y_prob" not in df.columns and "y_pred" not in df.columns:
        return None

    return df


def choose_prediction_run(df: pd.DataFrame) -> Optional[str]:
    # Prefer full logistic regression because it supports your interpretability story.
    preferred = df[(df["model_type"] == "logreg_l1") & (df["experiment_id"] == "full")]
    if not preferred.empty:
        return str(preferred.iloc[0]["run_id"])

    full = df[df["experiment_id"] == "full"]
    if not full.empty:
        return str(full.iloc[0]["run_id"])

    return str(df.iloc[0]["run_id"]) if not df.empty else None


def plot_confusion_matrix_from_predictions(df: pd.DataFrame) -> None:
    run_id = choose_prediction_run(df)
    if run_id is None:
        print("No run available for confusion matrix.")
        return

    preds = load_predictions_for_run(run_id)
    if preds is None:
        print(f"No predictions file found for run {run_id}. Skipping confusion matrix.")
        return

    run_row = df[df["run_id"] == run_id].iloc[0]
    threshold = run_row.get("chosen_threshold")
    if pd.isna(threshold):
        threshold = 0.5

    y_true = preds["y_true"].astype(int)

    if "y_pred" in preds.columns:
        y_pred = preds["y_pred"].astype(int)
    else:
        y_pred = (preds["y_prob"].astype(float) >= float(threshold)).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm)

    ax.set_title(f"Confusion Matrix at Threshold {float(threshold):.2f}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mulligan", "Keep"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Mulligan", "Keep"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_plot(fig, "confusion_matrix.png")


def plot_calibration_curve_from_predictions(df: pd.DataFrame) -> None:
    run_id = choose_prediction_run(df)
    if run_id is None:
        print("No run available for calibration curve.")
        return

    preds = load_predictions_for_run(run_id)
    if preds is None or "y_prob" not in preds.columns:
        print(f"No probability predictions found for run {run_id}. Skipping calibration curve.")
        return

    y_true = preds["y_true"].astype(int)
    y_prob = preds["y_prob"].astype(float)

    mask = y_prob.between(0, 1)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_true) < 20:
        print("Not enough prediction rows for calibration curve. Skipping.")
        return

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    ax.set_title("Calibration Curve")
    ax.set_xlabel("Predicted Keep Probability")
    ax.set_ylabel("Observed Keep Frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    save_plot(fig, "calibration_curve.png")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    run_ids = get_most_recent_run_ids(N_MOST_RECENT_RUNS)

    print("Using most recent run IDs:")
    for run_id in run_ids:
        print(f"  {run_id}")
    print()

    df = load_all_runs(run_ids)
    comparison = build_comparison_table(df)
    summary_text = build_summary_text(df)

    df.to_csv(OUTPUT_DIR / "ablation_runs_raw.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "ablation_comparison_table.csv", index=False)

    with open(OUTPUT_DIR / "ablation_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print("\nComparison table\n================")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(comparison.to_string(index=False))

    plot_ablation_auc(df)
    plot_ablation_logloss(df)
    plot_full_model_comparison(df)
    plot_feature_importance(df)
    plot_confusion_matrix_from_predictions(df)
    plot_calibration_curve_from_predictions(df)

    print("\nSaved outputs to:")
    print(f"  {OUTPUT_DIR}")
    print(f"  {FIGURE_DIR}")


if __name__ == "__main__":
    main()