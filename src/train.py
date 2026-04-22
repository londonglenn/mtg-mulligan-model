from __future__ import annotations

import traceback

import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils import (
    make_id,
    iso_now,
    write_json,
    read_json,
    build_dataset_dir,
    ensure_dir,
)
from registry import (
    get_run_paths,
    register_run,
    update_run,
    set_latest_run,
)
from features import get_model_feature_columns


MANA_BUCKET_FEATURES = {
    "0_drops",
    "1_drops",
    "2_drops",
    "3_drops",
    "4_drops",
    "5_drops",
    "6_plus_drops",
}

LAND_COUNT_FEATURES = {
    "num_lands",
    "num_lands_sq",
}

EXPERIMENT_CONFIGS = {
    "full": {
        "exclude_exact": set(),
        "exclude_prefixes": [],
    },
    "no_mana_buckets": {
        "exclude_exact": set(MANA_BUCKET_FEATURES),
        "exclude_prefixes": [],
    },
    "land_only": {
        "exclude_exact": set(),
        "exclude_prefixes": [],
        "include_exact": {"on_play", "num_lands", "num_lands_sq"},
    },
    "cards_only": {
        "exclude_exact": set(MANA_BUCKET_FEATURES) | set(LAND_COUNT_FEATURES) | {"on_play"},
        "exclude_prefixes": [],
    },
}

SUPPORTED_MODEL_TYPES = {
    "logreg_l1",
    "xgboost",
    "lightgbm",
}


def filter_feature_columns(
    feature_columns: list[str],
    exclude_exact: set[str] | None = None,
    exclude_prefixes: list[str] | None = None,
    include_exact: set[str] | None = None,
) -> list[str]:
    exclude_exact = exclude_exact or set()
    exclude_prefixes = exclude_prefixes or []

    filtered = []
    for col in feature_columns:
        if include_exact is not None and col not in include_exact:
            continue

        if col in exclude_exact:
            continue

        if any(col.startswith(prefix) for prefix in exclude_prefixes):
            continue

        filtered.append(col)

    return filtered


def make_lightgbm_feature_name_map(feature_columns: list[str]) -> dict[str, str]:
    """
    LightGBM rejects some special characters in feature names.
    Map original names to safe generic names only for LightGBM training/eval/predict.
    """
    return {orig: f"f_{i:04d}" for i, orig in enumerate(feature_columns)}


def build_logreg_search(random_state: int = 42) -> RandomizedSearchCV:
    base_model = LogisticRegression(
        solver="saga",
        penalty="l1",
        max_iter=5000,
        random_state=random_state,
    )

    # Smaller C => stronger regularization
    param_dist = {
        "C": np.logspace(-3, 2, 30),
    }

    return RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_log_loss",
        cv=5,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=1,
    )


def build_xgboost_search(random_state: int = 42) -> RandomizedSearchCV:
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    param_dist = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [200, 400, 800],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    }

    return RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring="neg_log_loss",
        cv=5,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=1,
    )


def build_lightgbm_search(random_state: int = 42) -> RandomizedSearchCV:
    base_model = LGBMClassifier(
        objective="binary",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    param_dist = {
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [200, 400, 800],
        "min_child_samples": [10, 20, 50],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    }

    return RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring="neg_log_loss",
        cv=5,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=1,
    )


def build_search(model_type: str, random_state: int = 42) -> RandomizedSearchCV:
    if model_type == "logreg_l1":
        return build_logreg_search(random_state=random_state)

    if model_type == "xgboost":
        return build_xgboost_search(random_state=random_state)

    if model_type == "lightgbm":
        return build_lightgbm_search(random_state=random_state)

    raise ValueError(
        f"Unknown model_type: {model_type}. "
        f"Valid options: {sorted(SUPPORTED_MODEL_TYPES)}"
    )


def main(
    dataset_id: str,
    experiment_id: str = "full",
    model_type: str = "logreg_l1",
    set_latest: bool = True,
) -> str:
    dataset_dir = build_dataset_dir(".", dataset_id)
    dataset_path = dataset_dir / "mulligan_data.csv"
    dataset_metadata_path = dataset_dir / "metadata.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset_metadata = read_json(dataset_metadata_path, default={})

    test_size = 0.2
    random_state = 42

    if experiment_id not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment_id: {experiment_id}. "
            f"Valid options: {sorted(EXPERIMENT_CONFIGS.keys())}"
        )

    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Valid options: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    df = pd.read_csv(dataset_path)

    all_feature_columns = get_model_feature_columns(df)
    experiment_config = EXPERIMENT_CONFIGS[experiment_id]

    feature_columns = filter_feature_columns(
        all_feature_columns,
        exclude_exact=experiment_config.get("exclude_exact", set()),
        exclude_prefixes=experiment_config.get("exclude_prefixes", []),
        include_exact=experiment_config.get("include_exact"),
    )

    if not feature_columns:
        raise ValueError(
            f"No features selected for experiment_id={experiment_id}"
        )

    X = df[feature_columns].copy()
    y = df["keep"].copy()

    feature_name_map = None
    if model_type == "lightgbm":
        feature_name_map = make_lightgbm_feature_name_map(feature_columns)
        X = X.rename(columns=feature_name_map)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    run_id = make_id(
        "run",
        extra_text=f"{model_type}|{experiment_id}|{dataset_id}",
    )
    paths = get_run_paths(run_id)

    ensure_dir(paths["run_dir"])

    register_run({
        "run_id": run_id,
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "created_at": iso_now(),
        "run_dir": str(paths["run_dir"]),
        "status": "training",
        "dataset_path": str(dataset_path),
        "dataset_metadata_path": str(dataset_metadata_path),
        "notes": (
            f"Tuned training run using model_type={model_type}, "
            f"experiment_id={experiment_id}, scoring=neg_log_loss"
        ),
    })

    try:
        search = build_search(model_type=model_type, random_state=random_state)
        search.fit(X_train, y_train)

        model = search.best_estimator_
        best_params = search.best_params_
        best_cv_log_loss = float(-search.best_score_)

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        if hasattr(model, "coef_"):
            n_selected_features = int((model.coef_[0] != 0).sum())
            n_zero_features = int((model.coef_[0] == 0).sum())
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            n_selected_features = int((importances > 0).sum())
            n_zero_features = int((importances == 0).sum())
        else:
            n_selected_features = None
            n_zero_features = None

        metrics = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "experiment_id": experiment_id,
            "model_type": model_type,
            "accuracy": float(accuracy_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "log_loss": float(log_loss(y_test, probs)),
            "tuning_metric": "neg_log_loss",
            "best_cv_log_loss": best_cv_log_loss,
            "best_params": best_params,
            "n_rows": int(len(df)),
            "n_features": int(X.shape[1]),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "baseline_keep_rate": float(y_train.mean()),
            "test_size_fraction": float(test_size),
            "random_state": int(random_state),
            "selected_feature_columns": feature_columns,
        }

        if hasattr(model, "n_iter_"):
            try:
                metrics["iterations_used"] = [int(x) for x in model.n_iter_]
                metrics["max_iter"] = int(model.max_iter)
            except Exception:
                pass

        if n_selected_features is not None:
            metrics["n_selected_features"] = int(n_selected_features)
        if n_zero_features is not None:
            metrics["n_zero_features"] = int(n_zero_features)

        joblib.dump(model, paths["model_path"])
        write_json(paths["feature_columns_path"], feature_columns)
        write_json(paths["metrics_path"], metrics)

        metadata = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "experiment_id": experiment_id,
            "model_type": model_type,
            "created_at": iso_now(),
            "status": "completed",
            "dataset_path": str(dataset_path),
            "dataset_metadata_path": str(dataset_metadata_path),
            "dataset_feature_schema_version": dataset_metadata.get("feature_schema_version"),
            "run_dir": str(paths["run_dir"]),
            "model_path": str(paths["model_path"]),
            "feature_columns_path": str(paths["feature_columns_path"]),
            "metrics_path": str(paths["metrics_path"]),
            "target_column": "keep",
            "n_rows": int(len(df)),
            "n_features": int(X.shape[1]),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "feature_name_map": feature_name_map,
            "tuning_metric": "neg_log_loss",
            "best_cv_log_loss": best_cv_log_loss,
            "best_params": best_params,
        }

        if hasattr(model, "get_params"):
            metadata["model_params"] = model.get_params()

        if n_selected_features is not None:
            metadata["n_selected_features"] = int(n_selected_features)
        if n_zero_features is not None:
            metadata["n_zero_features"] = int(n_zero_features)

        write_json(paths["metadata_path"], metadata)

        update_run(run_id, {
            "status": "completed",
            "dataset_path": str(dataset_path),
            "dataset_metadata_path": str(dataset_metadata_path),
            "model_path": str(paths["model_path"]),
            "feature_columns_path": str(paths["feature_columns_path"]),
            "metrics_path": str(paths["metrics_path"]),
            "metadata_path": str(paths["metadata_path"]),
            "model_type": model_type,
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "log_loss": metrics["log_loss"],
            "best_cv_log_loss": best_cv_log_loss,
            "n_rows": metrics["n_rows"],
            "n_features": metrics["n_features"],
            "train_size": metrics["train_size"],
            "test_size": metrics["test_size"],
        })

        if n_selected_features is not None:
            update_run(run_id, {"n_selected_features": int(n_selected_features)})
        if n_zero_features is not None:
            update_run(run_id, {"n_zero_features": int(n_zero_features)})

        if set_latest:
            set_latest_run(run_id)

        print("\nTraining complete.")
        print(f"Run ID: {run_id}")
        print(f"Dataset ID: {dataset_id}")
        print(f"Model Type: {model_type}")
        print(f"Experiment ID: {experiment_id}")
        print(f"Best CV log loss: {best_cv_log_loss:.6f}")
        print(f"Best params: {best_params}")
        print(f"Saved model to: {paths['model_path']}")
        print(f"Saved metrics to: {paths['metrics_path']}")
        print(f"Run directory: {paths['run_dir']}")
        if "n_selected_features" in metrics:
            print(f"Selected features: {metrics['n_selected_features']}")
        if "n_zero_features" in metrics:
            print(f"Zeroed features: {metrics['n_zero_features']}")
        if set_latest:
            print("Set as latest run.")

        return run_id

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