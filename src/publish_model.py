from __future__ import annotations

from pathlib import Path
import hashlib
import os
import traceback

import requests

from registry import get_latest_run_id, get_run, get_run_paths
from utils import read_json, write_json


# =========================================================
# CONFIG
# =========================================================
SERVER_BASE_URL = os.environ.get(
    "MODEL_SERVER_BASE_URL",
    "https://mulligan-server.onrender.com",
)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")

UPLOAD_URL = f"{SERVER_BASE_URL}/model/upload"
SET_LATEST_URL = f"{SERVER_BASE_URL}/model/set-latest"


# =========================================================
# HELPERS
# =========================================================
def sha256_file(path: Path, chunk_size: int = 8192) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None or str(run_id).strip().lower() == "latest":
        return get_latest_run_id()

    run_id = str(run_id).strip()
    run_entry = get_run(run_id)
    if run_entry is None:
        raise KeyError(f"Unknown run_id: {run_id}")

    return run_id


def build_bundle_version(metadata: dict, run_id: str) -> str:
    """
    Use an explicit model_version if present, otherwise fall back to run_id.
    """
    return str(
        metadata.get("model_version")
        or metadata.get("bundle_version")
        or run_id
    )


def build_manifest(run_id: str, paths: dict[str, Path]) -> dict:
    metadata = read_json(paths["metadata_path"], default={})
    metrics = read_json(paths["metrics_path"], default={})

    bundle_version = build_bundle_version(metadata, run_id)

    manifest = {
        "bundle_version": bundle_version,
        "run_id": run_id,
        "dataset_id": metadata.get("dataset_id"),
        "experiment_id": metadata.get("experiment_id"),
        "feature_schema_version": metadata.get("dataset_feature_schema_version"),
        "threshold": metrics.get("chosen_threshold", metrics.get("threshold", 0.5)),
        "files": {
            # These names are placeholders for upload metadata.
            # The server rewrites them to stored filenames.
            "model": paths["model_path"].name,
            "feature_columns": paths["feature_columns_path"].name,
            "metadata": paths["metadata_path"].name,
            "metrics": paths["metrics_path"].name,
        },
        "sha256": {
            "model.pkl": sha256_file(paths["model_path"]),
            "feature_columns.json": sha256_file(paths["feature_columns_path"]),
            "metadata.json": sha256_file(paths["metadata_path"]),
            "metrics.json": sha256_file(paths["metrics_path"]),
        },
    }

    return manifest


# =========================================================
# MAIN
# =========================================================
def main(run_id: str | None = None, set_latest: bool = True) -> dict:
    if not ADMIN_TOKEN:
        raise EnvironmentError(
            "ADMIN_TOKEN is not set. Export ADMIN_TOKEN before publishing."
        )

    resolved_run_id = resolve_run_id(run_id)
    run_entry = get_run(resolved_run_id)
    if run_entry is None:
        raise KeyError(f"Run not found in registry: {resolved_run_id}")

    paths = get_run_paths(resolved_run_id)

    required_paths = {
        "model_path": paths["model_path"],
        "feature_columns_path": paths["feature_columns_path"],
        "metadata_path": paths["metadata_path"],
        "metrics_path": paths["metrics_path"],
    }

    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Run {resolved_run_id} is missing required artifacts: {missing}"
        )

    manifest = build_manifest(resolved_run_id, paths)
    manifest_path = paths["run_dir"] / "manifest.json"
    write_json(manifest_path, manifest)

    headers = {
        "Authorization": f"Bearer {ADMIN_TOKEN}",
    }

    try:
        with open(manifest_path, "rb") as manifest_f, \
             open(paths["model_path"], "rb") as model_f, \
             open(paths["feature_columns_path"], "rb") as feature_f, \
             open(paths["metadata_path"], "rb") as metadata_f, \
             open(paths["metrics_path"], "rb") as metrics_f:

            files = {
                "manifest": (manifest_path.name, manifest_f, "application/json"),
                "model": (paths["model_path"].name, model_f, "application/octet-stream"),
                "feature_columns": (paths["feature_columns_path"].name, feature_f, "application/json"),
                "metadata": (paths["metadata_path"].name, metadata_f, "application/json"),
                "metrics": (paths["metrics_path"].name, metrics_f, "application/json"),
            }

            upload_response = requests.post(
                UPLOAD_URL,
                headers=headers,
                files=files,
                timeout=120,
            )

        upload_response.raise_for_status()
        upload_payload = upload_response.json()

        result = {
            "status": "uploaded",
            "run_id": resolved_run_id,
            "bundle_version": manifest["bundle_version"],
            "upload_response": upload_payload,
        }

        if set_latest:
            latest_response = requests.post(
                SET_LATEST_URL,
                headers={
                    **headers,
                    "Content-Type": "application/json",
                },
                json={"bundle_version": manifest["bundle_version"]},
                timeout=60,
            )
            latest_response.raise_for_status()
            result["set_latest_response"] = latest_response.json()
            result["status"] = "uploaded_and_set_latest"

        print("\n=== PUBLISH COMPLETE ===")
        print(f"Run ID: {resolved_run_id}")
        print(f"Bundle version: {manifest['bundle_version']}")
        print(f"Server: {SERVER_BASE_URL}")
        print(f"Manifest saved to: {manifest_path}")
        if set_latest:
            print("Bundle marked as latest.")

        return result

    except Exception as e:
        print(f"\nPublish failed for run {resolved_run_id}")
        print(f"{type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()