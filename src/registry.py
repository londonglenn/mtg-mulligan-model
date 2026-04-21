from __future__ import annotations

from pathlib import Path
from typing import Any

from utils import ensure_dir, read_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = MODELS_DIR / "runs"
REGISTRY_PATH = MODELS_DIR / "registry.json"
LATEST_PATH = MODELS_DIR / "latest.json"


def init_registry() -> None:
    """
    Ensure the models directory, runs directory, and registry files exist.
    """
    ensure_dir(MODELS_DIR)
    ensure_dir(RUNS_DIR)

    if not REGISTRY_PATH.exists():
        write_json(REGISTRY_PATH, {"runs": []})

    if not LATEST_PATH.exists():
        write_json(LATEST_PATH, {"latest_run_id": None})


def load_registry() -> dict[str, Any]:
    """
    Load the full registry JSON.
    """
    init_registry()
    return read_json(REGISTRY_PATH, default={"runs": []})


def save_registry(registry: dict[str, Any]) -> None:
    """
    Save the full registry JSON.
    """
    init_registry()
    write_json(REGISTRY_PATH, registry)


def register_run(entry: dict[str, Any]) -> None:
    """
    Add a run entry to the registry.

    Required keys:
        - run_id
        - dataset_id
        - experiment_id
        - created_at
        - run_dir

    Optional keys:
        - status
        - metrics_path
        - model_path
        - feature_columns_path
        - metadata_path
        - notes
        - anything else useful
    """
    required_keys = {
        "run_id",
        "dataset_id",
        "experiment_id",
        "created_at",
        "run_dir",
    }

    missing = required_keys - set(entry.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required registry keys: {missing_str}")

    registry = load_registry()

    existing_ids = {run["run_id"] for run in registry.get("runs", [])}
    if entry["run_id"] in existing_ids:
        raise ValueError(f"Run ID already exists in registry: {entry['run_id']}")

    registry["runs"].append(entry)
    save_registry(registry)


def set_latest_run(run_id: str) -> None:
    """
    Set the latest/default run used by predict.py when no run_id is specified.
    """
    init_registry()

    run = get_run(run_id)
    if run is None:
        raise KeyError(f"Cannot set latest run. Unknown run_id: {run_id}")

    write_json(LATEST_PATH, {"latest_run_id": run_id})


def get_latest_run_id() -> str:
    """
    Return the latest/default run ID.
    """
    init_registry()
    latest = read_json(LATEST_PATH, default={"latest_run_id": None})
    run_id = latest.get("latest_run_id")

    if not run_id:
        raise FileNotFoundError("No latest run has been set yet.")

    return str(run_id)


def get_latest_run() -> dict[str, Any]:
    """
    Return the registry entry for the latest/default run.
    """
    run_id = get_latest_run_id()
    run = get_run(run_id)

    if run is None:
        raise KeyError(f"Latest run_id '{run_id}' was not found in the registry.")

    return run


def get_run(run_id: str) -> dict[str, Any] | None:
    """
    Return one run entry by run_id, or None if it does not exist.
    """
    registry = load_registry()

    for run in registry.get("runs", []):
        if run.get("run_id") == run_id:
            return run

    return None


def list_runs(
    dataset_id: str | None = None,
    experiment_id: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return all runs, optionally filtered by dataset_id, experiment_id, and/or status.
    """
    registry = load_registry()
    runs = registry.get("runs", [])

    filtered = []
    for run in runs:
        if dataset_id is not None and run.get("dataset_id") != dataset_id:
            continue
        if experiment_id is not None and run.get("experiment_id") != experiment_id:
            continue
        if status is not None and run.get("status") != status:
            continue
        filtered.append(run)

    return filtered


def update_run(run_id: str, updates: dict[str, Any]) -> None:
    """
    Update fields on an existing run entry.
    """
    registry = load_registry()
    runs = registry.get("runs", [])

    for i, run in enumerate(runs):
        if run.get("run_id") == run_id:
            updated = dict(run)
            updated.update(updates)
            runs[i] = updated
            registry["runs"] = runs
            save_registry(registry)
            return

    raise KeyError(f"Run ID not found: {run_id}")


def remove_run(run_id: str) -> None:
    """
    Remove a run from the registry.

    Note:
        This only removes the registry entry.
        It does NOT delete files from disk.
    """
    registry = load_registry()
    runs = registry.get("runs", [])

    new_runs = [run for run in runs if run.get("run_id") != run_id]

    if len(new_runs) == len(runs):
        raise KeyError(f"Run ID not found: {run_id}")

    registry["runs"] = new_runs
    save_registry(registry)

    latest = read_json(LATEST_PATH, default={"latest_run_id": None})
    if latest.get("latest_run_id") == run_id:
        write_json(LATEST_PATH, {"latest_run_id": None})


def get_run_dir(run_id: str) -> Path:
    """
    Return the canonical run directory path for a run_id.

    This trusts the folder convention:
        models/runs/<run_id>
    """
    return RUNS_DIR / run_id


def get_run_paths(run_id: str) -> dict[str, Path]:
    """
    Return the standard file paths inside a run directory.
    """
    run_dir = get_run_dir(run_id)

    return {
        "run_dir": run_dir,
        "model_path": run_dir / "model.pkl",
        "feature_columns_path": run_dir / "feature_columns.json",
        "metrics_path": run_dir / "metrics.json",
        "metadata_path": run_dir / "metadata.json",
        "threshold_sweep_path": run_dir / "threshold_sweep.csv",
        "confusion_matrix_path": run_dir / "confusion_matrix.csv",
        "predictions_path": run_dir / "predictions.csv",
        "misclassified_path": run_dir / "misclassified.csv",
        "false_keeps_path": run_dir / "false_keeps.csv",
        "false_mulligans_path": run_dir / "false_mulligans.csv",
        "top_features_path": run_dir / "top_features.csv",
        "selected_features_path": run_dir / "selected_features.csv",
        "run_summary_path": run_dir / "run_summary.txt",
    }