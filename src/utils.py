from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any
import json
import hashlib


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist and return it as a Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp_str() -> str:
    """
    Return a compact timestamp string for IDs and filenames.
    Example: 20260419_213015
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_id(prefix: str, extra_text: str = "") -> str:
    """
    Build a stable-looking ID like:
        ds_20260419_213015_ab12cd34
        run_20260419_213015_ef56gh78

    The short hash helps avoid collisions when multiple IDs are created
    close together or when you want to include context.
    """
    ts = timestamp_str()
    seed = f"{prefix}|{ts}|{extra_text}"
    short = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{ts}_{short}"


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    """
    Write a Python object to JSON, creating parent directories if needed.
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def read_json(path: str | Path, default: Any | None = None) -> Any:
    """
    Read JSON from disk. If the file does not exist and a default is provided,
    return the default instead.
    """
    path = Path(path)

    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_json_list(path: str | Path, item: dict[str, Any], root_key: str = "items") -> None:
    """
    Append one dictionary entry into a JSON file shaped like:
        { "items": [ ... ] }

    If the file does not exist yet, it will be created.
    """
    path = Path(path)
    data = read_json(path, default={root_key: []})

    if root_key not in data:
        data[root_key] = []

    data[root_key].append(item)
    write_json(path, data)


def iso_now() -> str:
    """
    Return the current local timestamp in ISO format.
    Example: 2026-04-19T21:30:15
    """
    return datetime.now().isoformat(timespec="seconds")


def safe_stem(text: str) -> str:
    """
    Convert a string into something safer for folder/file names.
    Keeps letters, numbers, underscore, hyphen.
    Replaces spaces and other characters with underscores.
    """
    cleaned = []
    for ch in str(text).strip():
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")

    out = "".join(cleaned)

    while "__" in out:
        out = out.replace("__", "_")

    return out.strip("_")


def file_sha256(path: str | Path, chunk_size: int = 8192) -> str:
    """
    Compute the SHA-256 hash of a file.
    Useful for dataset fingerprints, duplicate detection, or provenance.
    """
    path = Path(path)
    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def build_dataset_dir(project_root: str | Path, dataset_id: str) -> Path:
    """
    Return the canonical folder for a processed dataset snapshot.
    """
    project_root = Path(project_root)
    return project_root / "data" / "processed" / "datasets" / dataset_id


def build_run_dir(project_root: str | Path, run_id: str) -> Path:
    """
    Return the canonical folder for a trained model run.
    """
    project_root = Path(project_root)
    return project_root / "models" / "runs" / run_id