from __future__ import annotations

from pathlib import Path
import json
import traceback

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from utils import make_id, iso_now, ensure_dir, write_json
from registry import get_run_paths, register_run, update_run, set_latest_run


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "mulligan_data.csv"


def main() -> None:
    # =========================================================
    # Config
    # =========================================================
    dataset_id = "legacy_global_dataset"
    experiment_id = "logreg_l1_default"

    test_size = 0.2
    random_state = 42

    model_params = {
        "penalty": "l1",
        "solver": "saga",
        "max_iter": 5000,
        "random_state": random_state,
    }

    # =========================================================
    # Create run + register early
    # =========================================================
    run_id = make_id("run", extra_text=experiment_id)
    paths = get_run_paths(run_id)
    ensure_dir(paths["run_dir"])

    register_run({
        "run_id": run_id,
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "created_at": iso_now(),
        "run_dir": str(paths["run_dir"]),
        "status": "training",
        "data_path": str(DATA_PATH),
        "notes": "Training run created from legacy global dataset path.",
    })

    try:
        # =========================================================
        # Load data
        # =========================================================
        df = pd.read_csv(DATA_PATH)

        drop_cols = [c for c in ["timestamp", "source_file"] if c in df.columns]
        raw_card_cols = [f"card{i}" for i in range(1, 8) if f"card{i}" in df.columns]

        X = df.drop(columns=drop_cols + raw_card_cols + ["keep"])
        y = df["keep"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        # =========================================================
        # Train model
        # =========================================================
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)

        print("\nIterations used:", model.n_iter_)
        print("Max iterations:", model.max_iter)

        # =========================================================
        # Evaluate on holdout
        # =========================================================
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        metrics = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "experiment_id": experiment_id,
            "accuracy": float(accuracy_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "log_loss": float(log_loss(y_test, probs)),
            "n_rows": int(len(df)),
            "n_features": int(X.shape[1]),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "baseline_keep_rate": float(y_train.mean()),
            "test_size_fraction": float(test_size),
            "random_state": int(random_state),
            "iterations_used": [int(x) for x in model.n_iter_],
            "max_iter": int(model.max_iter),
        }

        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        # =========================================================
        # Save artifacts into run folder
        # =========================================================
        joblib.dump(model, paths["model_path"])

        write_json(paths["feature_columns_path"], list(X.columns))
        write_json(paths["metrics_path"], metrics)

        metadata = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "experiment_id": experiment_id,
            "created_at": iso_now(),
            "status": "completed",
            "data_path": str(DATA_PATH),
            "run_dir": str(paths["run_dir"]),
            "model_path": str(paths["model_path"]),
            "feature_columns_path": str(paths["feature_columns_path"]),
            "metrics_path": str(paths["metrics_path"]),
            "model_type": "LogisticRegression",
            "model_params": model_params,
            "target_column": "keep",
            "dropped_columns": drop_cols + raw_card_cols,
            "n_rows": int(len(df)),
            "n_features": int(X.shape[1]),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        }
        write_json(paths["metadata_path"], metadata)

        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coefficient": model.coef_[0],
        }).sort_values("coefficient", ascending=False)

        coef_df.to_csv(paths["top_features_path"], index=False)

        selected_coef_df = coef_df[coef_df["coefficient"] != 0].copy()
        selected_coef_df["abs_coefficient"] = selected_coef_df["coefficient"].abs()
        selected_coef_df = (
            selected_coef_df
            .sort_values("abs_coefficient", ascending=False)
            .drop(columns=["abs_coefficient"])
        )
        selected_coef_df.to_csv(paths["selected_features_path"], index=False)

        with open(paths["run_summary_path"], "w", encoding="utf-8") as f:
            f.write("Training run summary\n")
            f.write("====================\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        # =========================================================
        # Update registry + latest pointer
        # =========================================================
        update_run(run_id, {
            "status": "completed",
            "model_path": str(paths["model_path"]),
            "feature_columns_path": str(paths["feature_columns_path"]),
            "metrics_path": str(paths["metrics_path"]),
            "metadata_path": str(paths["metadata_path"]),
            "top_features_path": str(paths["top_features_path"]),
            "selected_features_path": str(paths["selected_features_path"]),
            "run_summary_path": str(paths["run_summary_path"]),
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "log_loss": metrics["log_loss"],
            "n_rows": metrics["n_rows"],
            "n_features": metrics["n_features"],
            "train_size": metrics["train_size"],
            "test_size": metrics["test_size"],
        })

        set_latest_run(run_id)

        print(f"\nRun ID: {run_id}")
        print(f"Saved model to: {paths['model_path']}")
        print(f"Saved metrics to: {paths['metrics_path']}")
        print(f"Saved feature importances to: {paths['top_features_path']}")
        print(f"Saved selected features to: {paths['selected_features_path']}")
        print(f"Run directory: {paths['run_dir']}")
        print("Set as latest run.")

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"

        try:
            update_run(run_id, {
                "status": "failed",
                "error": error_message,
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass

        print(f"\nTraining failed for run {run_id}")
        print(error_message)
        raise


if __name__ == "__main__":
    main()