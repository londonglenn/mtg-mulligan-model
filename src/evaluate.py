from __future__ import annotations

from pathlib import Path
import traceback

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

from registry import (
    get_latest_run_id,
    get_run,
    get_run_paths,
    update_run,
)
from utils import read_json, write_json, iso_now


PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None or str(run_id).strip().lower() == "latest":
        return get_latest_run_id()

    run_id = str(run_id).strip()
    run_entry = get_run(run_id)
    if run_entry is None:
        raise KeyError(f"Unknown run_id: {run_id}")

    return run_id


def resolve_dataset_path(run_entry: dict, metadata: dict) -> Path:
    dataset_path = (
        run_entry.get("dataset_path")
        or metadata.get("dataset_path")
    )

    if not dataset_path:
        raise KeyError("No dataset_path stored in run entry or metadata.")

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    return dataset_path


# =========================
# Main
# =========================

def main(run_id: str | None = None) -> str:
    resolved_run_id = resolve_run_id(run_id)
    run_entry = get_run(resolved_run_id)
    paths = get_run_paths(resolved_run_id)

    if run_entry is None:
        raise KeyError(f"Run not found in registry: {resolved_run_id}")

    update_run(resolved_run_id, {
        "status": "evaluating",
        "evaluation_started_at": iso_now(),
    })

    try:
        if not paths["model_path"].exists():
            raise FileNotFoundError(f"Model file not found: {paths['model_path']}")

        if not paths["feature_columns_path"].exists():
            raise FileNotFoundError(
                f"Feature columns file not found: {paths['feature_columns_path']}"
            )

        model = joblib.load(paths["model_path"])
        feature_columns = read_json(paths["feature_columns_path"])

        metadata = read_json(paths["metadata_path"], default={})
        prior_metrics = read_json(paths["metrics_path"], default={})

        dataset_path = resolve_dataset_path(run_entry, metadata)
        df = pd.read_csv(dataset_path)

        missing_features = [c for c in feature_columns if c not in df.columns]
        if missing_features:
            raise ValueError(
                "Dataset is missing expected feature columns required by this run: "
                f"{missing_features[:10]}"
                + (" ..." if len(missing_features) > 10 else "")
            )

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
        evaluation_metrics = {
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
        evaluation_metrics["baseline_log_loss"] = float(log_loss(y_np, baseline_probs))

        metrics = dict(prior_metrics)
        metrics.update(evaluation_metrics)
        metrics["run_id"] = resolved_run_id
        metrics["dataset_id"] = run_entry.get("dataset_id")
        metrics["experiment_id"] = run_entry.get("experiment_id")
        metrics["evaluation_completed_at"] = iso_now()

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

        pred_df["hand"] = pred_df.apply(reconstruct_hand_from_input, axis=1)

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
        misclassified_df = misclassified_df.rename(columns={"pred_prob": "score"})
        misclassified_df = misclassified_df.sort_values("score", ascending=False)

        false_keeps = misclassified_df[
            misclassified_df["error_type"] == "false_keep"
        ][["hand", "play_draw", "score", "actual_label", "pred_label"]].copy()

        false_mulls = misclassified_df[
            misclassified_df["error_type"] == "false_mulligan"
        ][["hand", "play_draw", "score", "actual_label", "pred_label"]].copy()

        # =========================
        # Save Outputs to Run Folder
        # =========================
        write_json(paths["metrics_path"], metrics)

        threshold_df.to_csv(paths["threshold_sweep_path"], index=False)
        cm_df.to_csv(paths["confusion_matrix_path"])

        pred_df.to_csv(paths["predictions_path"], index=False)
        misclassified_df.to_csv(paths["misclassified_path"], index=False)

        false_keeps.to_csv(paths["false_keeps_path"], index=False)
        false_mulls.to_csv(paths["false_mulligans_path"], index=False)

        coef_df.to_csv(paths["top_features_path"], index=False)
        selected_coef_df.to_csv(paths["selected_features_path"], index=False)

        with open(paths["run_summary_path"], "w", encoding="utf-8") as f:
            f.write("Cross-validated evaluation summary\n")
            f.write("=================================\n")
            f.write(f"run_id: {resolved_run_id}\n")
            f.write(f"dataset_id: {run_entry.get('dataset_id')}\n")
            f.write(f"experiment_id: {run_entry.get('experiment_id')}\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        metadata.update({
            "evaluation_completed_at": iso_now(),
            "chosen_threshold": float(threshold),
            "n_selected_features": int((final_model.coef_[0] != 0).sum()),
            "n_zero_features": int((final_model.coef_[0] == 0).sum()),
        })
        write_json(paths["metadata_path"], metadata)

        update_run(resolved_run_id, {
            "status": "completed",
            "metrics_path": str(paths["metrics_path"]),
            "metadata_path": str(paths["metadata_path"]),
            "threshold_sweep_path": str(paths["threshold_sweep_path"]),
            "confusion_matrix_path": str(paths["confusion_matrix_path"]),
            "predictions_path": str(paths["predictions_path"]),
            "misclassified_path": str(paths["misclassified_path"]),
            "false_keeps_path": str(paths["false_keeps_path"]),
            "false_mulligans_path": str(paths["false_mulligans_path"]),
            "top_features_path": str(paths["top_features_path"]),
            "selected_features_path": str(paths["selected_features_path"]),
            "run_summary_path": str(paths["run_summary_path"]),
            "chosen_threshold": float(threshold),
            "balanced_accuracy_at_chosen_threshold": float(best_bal_acc),
            "accuracy_mean": metrics["accuracy_mean"],
            "roc_auc_mean": metrics["roc_auc_mean"],
            "oof_log_loss": metrics["oof_log_loss"],
            "evaluation_completed_at": metrics["evaluation_completed_at"],
        })

        print("\nDone.")
        print(f"Run ID: {resolved_run_id}")
        print(f"Dataset path: {dataset_path}")
        print(f"Best threshold: {threshold}")
        print(f"Balanced accuracy: {best_bal_acc}")
        print(f"Selected features: {metrics['n_selected_features']}")
        print(f"Reports saved to: {paths['run_dir']}")

        return resolved_run_id

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"

        try:
            update_run(resolved_run_id, {
                "status": "evaluation_failed",
                "evaluation_error": error_message,
                "evaluation_traceback": traceback.format_exc(),
            })
        except Exception:
            pass

        print(f"\nEvaluation failed for run {resolved_run_id}")
        print(error_message)
        raise


if __name__ == "__main__":
    main()