from __future__ import annotations

from pathlib import Path
import traceback

import pandas as pd

from utils import (
    make_id,
    iso_now,
    ensure_dir,
    write_json,
    build_dataset_dir,
    file_sha256,
)
from features import (
    CARD_COLS,
    LAND_COUNT_FEATURES,
    MANA_VALUE_BUCKET_FEATURES,
    FEATURE_SCHEMA_VERSION,
    SCRYFALL_CACHE_PATH,
    build_card_info_lookup,
    build_card_count_matrix,
    build_step_encoded_matrix,
    add_land_features,
    add_mana_value_features,
)

# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LEGACY_FINAL_DATA_PATH = PROJECT_ROOT / "data" / "mulligan_data.csv"

SUPPORTED_EXTS = {".csv", ".xlsx", ".xls"}


# =========================================================
# HELPERS
# =========================================================
def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_decision(value):
    if pd.isna(value):
        return None
    v = str(value).strip().lower()
    if v in {"keep", "k", "1", "yes", "y"}:
        return 1
    if v in {"mulligan", "mull", "m", "0", "no", "n"}:
        return 0
    return None


def normalize_play_draw(value):
    if pd.isna(value):
        return None
    v = str(value).strip().lower()
    if v == "play":
        return 1
    if v == "draw":
        return 0
    return None


def list_supported_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []

    return [
        path
        for path in sorted(raw_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]


def load_all_raw_files(raw_dir: Path) -> pd.DataFrame:
    all_frames = []
    raw_files = list_supported_raw_files(raw_dir)

    for path in raw_files:
        df = load_table(path)

        for col in CARD_COLS:
            if col not in df.columns:
                df[col] = None

        if "timestamp" not in df.columns:
            df["timestamp"] = None

        if "decision" not in df.columns:
            df["decision"] = None

        if "play_draw" not in df.columns:
            df["play_draw"] = None

        keep_cols = ["timestamp", "play_draw"] + CARD_COLS + ["decision"]
        df = df[keep_cols].copy()
        df["source_file"] = path.name

        all_frames.append(df)

    if not all_frames:
        raise ValueError(f"No supported raw files found in {raw_dir}")

    return pd.concat(all_frames, ignore_index=True)


def clean_combined_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in CARD_COLS:
        df[col] = df[col].astype("string").str.strip()

    df["keep"] = df["decision"].apply(normalize_decision)
    df["on_play"] = df["play_draw"].apply(normalize_play_draw)

    df = df[df["on_play"].notna()].copy()
    df["on_play"] = df["on_play"].astype(int)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df[df["keep"].notna()].copy()
    df["keep"] = df["keep"].astype(int)

    df = df.dropna(subset=CARD_COLS)
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


def build_final_dataset(clean_df, step_matrix, land_features, mana_value_features):
    metadata = clean_df[["timestamp", "source_file", "on_play"]].reset_index(drop=True)
    raw_hand = clean_df[CARD_COLS].reset_index(drop=True)
    labels = clean_df[["keep"]].reset_index(drop=True)

    return pd.concat(
        [
            metadata,
            raw_hand,
            step_matrix.reset_index(drop=True),
            land_features.reset_index(drop=True),
            mana_value_features.reset_index(drop=True),
            labels,
        ],
        axis=1,
    )


def build_dataset_metadata(
    dataset_id: str,
    dataset_dir: Path,
    final_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    raw_files: list[Path],
    final_data_path: Path,
) -> dict:
    raw_file_info = []
    for path in raw_files:
        try:
            raw_file_info.append({
                "name": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": int(path.stat().st_size),
            })
        except Exception:
            raw_file_info.append({
                "name": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": None,
            })

    metadata = {
        "dataset_id": dataset_id,
        "created_at": iso_now(),
        "dataset_dir": str(dataset_dir),
        "final_data_path": str(final_data_path),
        "legacy_final_data_path": str(LEGACY_FINAL_DATA_PATH),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "n_rows": int(len(clean_df)),
        "n_columns": int(final_df.shape[1]),
        "n_unique_source_files": int(clean_df["source_file"].nunique()) if "source_file" in clean_df.columns else 0,
        "land_count_features": LAND_COUNT_FEATURES,
        "mana_value_bucket_features": MANA_VALUE_BUCKET_FEATURES,
        "raw_dir": str(RAW_DIR),
        "raw_files": raw_file_info,
        "notes": "Processed dataset snapshot using num_lands / num_lands_sq and shared features.py feature definitions.",
    }

    if final_data_path.exists():
        metadata["dataset_sha256"] = file_sha256(final_data_path)

    return metadata


# =========================================================
# MAIN
# =========================================================
def main(dataset_id: str | None = None, update_legacy_global: bool = True) -> str:
    ensure_dir(PROCESSED_DIR)

    raw_files = list_supported_raw_files(RAW_DIR)
    if not raw_files:
        raise ValueError(f"No supported raw files found in {RAW_DIR}")

    if dataset_id is None:
        dataset_id = make_id("ds", extra_text=f"{len(raw_files)}_files")

    dataset_dir = build_dataset_dir(PROJECT_ROOT, dataset_id)
    ensure_dir(dataset_dir)

    final_data_path = dataset_dir / "mulligan_data.csv"
    metadata_path = dataset_dir / "metadata.json"

    print("Loading raw files...")
    combined = load_all_raw_files(RAW_DIR)

    print("Cleaning data...")
    clean_df = clean_combined_data(combined)

    print("Building card matrix...")
    card_matrix = build_card_count_matrix(clean_df)

    print("Building step matrix...")
    step_matrix = build_step_encoded_matrix(card_matrix)

    print("Building card info lookup from Scryfall...")
    unique_cards = pd.unique(clean_df[CARD_COLS].values.ravel("K"))
    unique_cards = [c for c in unique_cards if pd.notna(c)]
    card_info_lookup = build_card_info_lookup(unique_cards, SCRYFALL_CACHE_PATH)

    print("Building land features...")
    land_features = add_land_features(clean_df, card_info_lookup)

    print("Building mana value features...")
    mana_value_features = add_mana_value_features(clean_df, card_info_lookup)

    print("Building final dataset...")
    final_df = build_final_dataset(clean_df, step_matrix, land_features, mana_value_features)

    print("Saving dataset snapshot...")
    final_df.to_csv(final_data_path, index=False)

    metadata = build_dataset_metadata(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        final_df=final_df,
        clean_df=clean_df,
        raw_files=raw_files,
        final_data_path=final_data_path,
    )
    write_json(metadata_path, metadata)

    if update_legacy_global:
        ensure_dir(LEGACY_FINAL_DATA_PATH.parent)
        final_df.to_csv(LEGACY_FINAL_DATA_PATH, index=False)

    print("\nDone.")
    print(f"Dataset ID: {dataset_id}")
    print(f"Rows: {len(clean_df)}")
    print(f"Dataset snapshot saved to: {final_data_path}")
    print(f"Metadata saved to: {metadata_path}")

    if update_legacy_global:
        print(f"Legacy global dataset updated at: {LEGACY_FINAL_DATA_PATH}")

    return dataset_id


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nPreprocessing failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise