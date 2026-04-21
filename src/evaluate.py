from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    log_loss,
    balanced_accuracy_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "mulligan_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "logreg.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

N_SPLITS = 5
RANDOM_STATE = 42


# =========================
# Helper Functions
# =========================

def label_to_name(value):
    return "keep" if int(value) == 1 else "mulligan"


def reconstruct_hand_from_input(row):
    card_cols = [f"card{i}" for i in range(1, 8)]
    cards = []

    for col in card_cols:
        card = row.get(col, None)
        if pd.notna(card):
            cards.append(str(card).strip())

    return " | ".join(cards)


# =========================
# Main
# =========================

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model + feature names
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    X = df[feature_columns].copy()
    y = df["keep"].copy()

    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # =========================
    # Cross Validation
    # =========================

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "neg_log_loss": "neg_log_loss",
    }

    cv_results = cross_validate(
        clone(model),
        X_np,
        y_np,
        cv=cv,
        scoring=scoring,
    )

    # Out-of-fold probabilities
    probs = cross_val_predict(
        clone(model),
        X_np,
        y_np,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    # =========================
    # Threshold Sweep
    # =========================

    thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_bal_acc = -np.inf
    threshold_rows = []

    for t in thresholds:
        preds_t = (probs >= t).astype(int)
        bal_acc = balanced_accuracy_score(y_np, preds_t)

        tn, fp, fn, tp = confusion_matrix(y_np, preds_t).ravel()

        threshold_rows.append({
            "threshold": float(t),
            "balanced_accuracy": float(bal_acc),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = float(t)

    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        ["balanced_accuracy", "threshold"],
        ascending=[False, True]
    )

    threshold = best_threshold
    preds = (probs >= threshold).astype(int)

    # =========================
    # Coefficients / Selected Features
    # =========================

    final_model = clone(model)
    final_model.fit(X_np, y_np)

    coef_df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": final_model.coef_[0]
    }).sort_values("coefficient", ascending=False)

    selected_coef_df = coef_df[coef_df["coefficient"] != 0].copy()
    selected_coef_df["abs_coefficient"] = selected_coef_df["coefficient"].abs()
    selected_coef_df = selected_coef_df.sort_values(
        "abs_coefficient",
        ascending=False
    ).drop(columns=["abs_coefficient"])

    # =========================
    # Metrics
    # =========================

    metrics = {
        "cv_n_splits": int(N_SPLITS),
        "cv_shuffle": True,
        "cv_random_state": int(RANDOM_STATE),
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),

        "baseline_keep_rate": float(y_np.mean()),

        "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "accuracy_std": float(np.std(cv_results["test_accuracy"], ddof=1)),

        "roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "roc_auc_std": float(np.std(cv_results["test_roc_auc"], ddof=1)),

        "log_loss_mean": float(-np.mean(cv_results["test_neg_log_loss"])),
        "log_loss_std": float(np.std(-cv_results["test_neg_log_loss"], ddof=1)),

        "oof_log_loss": float(log_loss(y_np, probs)),

        "chosen_threshold": float(threshold),
        "balanced_accuracy_at_chosen_threshold": float(best_bal_acc),

        "n_selected_features": int((final_model.coef_[0] != 0).sum()),
        "n_zero_features": int((final_model.coef_[0] == 0).sum()),
    }

    baseline_probs = np.full(len(y_np), y_np.mean())
    metrics["baseline_log_loss"] = float(log_loss(y_np, baseline_probs))

    # =========================
    # Confusion Matrix
    # =========================

    cm = confusion_matrix(y_np, preds)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_mulligan", "actual_keep"],
        columns=["pred_mulligan", "pred_keep"]
    )

    # =========================
    # Prediction Analysis
    # =========================

    pred_df = df.copy()

    pred_df["actual"] = y_np
    pred_df["actual_label"] = pred_df["actual"].map(label_to_name)

    pred_df["pred_prob"] = probs
    pred_df["pred_class"] = preds
    pred_df["pred_label"] = pred_df["pred_class"].map(label_to_name)

    pred_df["correct"] = pred_df["actual"] == pred_df["pred_class"]

    pred_df["error_type"] = np.where(
        pred_df["correct"],
        "correct",
        np.where(
            (pred_df["actual"] == 0) & (pred_df["pred_class"] == 1),
            "false_keep",
            "false_mulligan"
        )
    )

    pred_df["hand"] = pred_df.apply(
        reconstruct_hand_from_input,
        axis=1
    )

    # =========================
    # Misclassified Hands
    # =========================

    misclassified_df = pred_df[~pred_df["correct"]].copy()

    misclassified_df = misclassified_df[[
        "hand",
        "on_play",
        "pred_prob",
        "actual_label",
        "pred_label",
        "error_type"
    ]].copy()

    misclassified_df["play_draw"] = misclassified_df["on_play"].map(
        {1: "play", 0: "draw"}
    )

    misclassified_df = misclassified_df.drop(columns=["on_play"])

    misclassified_df = misclassified_df.rename(columns={
        "pred_prob": "score"
    })

    misclassified_df = misclassified_df.sort_values("score", ascending=False)

    false_keeps = misclassified_df[
        misclassified_df["error_type"] == "false_keep"
    ][["hand", "play_draw", "score", "actual_label", "pred_label"]].copy()

    false_mulls = misclassified_df[
        misclassified_df["error_type"] == "false_mulligan"
    ][["hand", "play_draw", "score", "actual_label", "pred_label"]].copy()

    # =========================
    # Save Outputs
    # =========================

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    threshold_df.to_csv(REPORTS_DIR / "threshold_sweep.csv", index=False)
    cm_df.to_csv(REPORTS_DIR / "confusion_matrix.csv")

    pred_df.to_csv(REPORTS_DIR / "predictions.csv", index=False)
    misclassified_df.to_csv(REPORTS_DIR / "misclassified.csv", index=False)

    false_keeps.to_csv(REPORTS_DIR / "false_keeps.csv", index=False)
    false_mulls.to_csv(REPORTS_DIR / "false_mulligans.csv", index=False)

    coef_df.to_csv(REPORTS_DIR / "top_features.csv", index=False)
    selected_coef_df.to_csv(REPORTS_DIR / "selected_features.csv", index=False)

    with open(REPORTS_DIR / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write("Cross-validated evaluation summary\n")
        f.write("=================================\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print("\nDone.")
    print(f"Best threshold: {threshold}")
    print(f"Balanced accuracy: {best_bal_acc}")
    print(f"Selected features: {metrics['n_selected_features']}")
    print(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()