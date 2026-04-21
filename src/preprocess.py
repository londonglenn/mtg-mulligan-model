from __future__ import annotations

from pathlib import Path
import json
import time
import traceback

import requests
import pandas as pd

from utils import (
    make_id,
    iso_now,
    ensure_dir,
    write_json,
    build_dataset_dir,
    file_sha256,
)

# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Legacy compatibility output
LEGACY_FINAL_DATA_PATH = PROJECT_ROOT / "data" / "mulligan_data.csv"

SUPPORTED_EXTS = {".csv", ".xlsx", ".xls"}
CARD_COLS = [f"card{i}" for i in range(1, 8)]

SCRYFALL_CACHE_PATH = PROCESSED_DIR / "card_info_cache.json"
SCRYFALL_NAMED_URL = "https://api.scryfall.com/cards/named"

FEATURE_SCHEMA_VERSION = "v1_step_land_curve"


# =========================================================
# HELPERS
# =========================================================
def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    else:
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


def load_card_info_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_card_info_cache(cache: dict, cache_path: Path) -> None:
    ensure_dir(cache_path.parent)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def fetch_card_info(card_name: str) -> dict:
    response = requests.get(
        SCRYFALL_NAMED_URL,
        params={"exact": card_name},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    return {
        "type_line": data.get("type_line", ""),
        "is_land": ("Land" in data.get("type_line", "")),
        "cmc": float(data.get("cmc", 0.0)),
    }


def build_card_info_lookup(unique_cards, cache_path: Path) -> dict:
    cache = load_card_info_cache(cache_path)
    updated = False

    for card in sorted(unique_cards):
        if not card or pd.isna(card):
            continue

        card = str(card).strip()
        if not card:
            continue

        if card in cache:
            continue

        try:
            info = fetch_card_info(card)
            cache[card] = info
            updated = True
            time.sleep(0.1)  # be polite to Scryfall
        except Exception as e:
            print(f"Warning: could not fetch Scryfall data for '{card}': {e}")
            cache[card] = {
                "type_line": "",
                "is_land": False,
                "cmc": 0.0,
            }
            updated = True

    if updated:
        save_card_info_cache(cache, cache_path)

    return cache


def list_supported_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []

    return [
        path
        for path in sorted(raw_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]


# =========================================================
# STEP 1: LOAD AND STANDARDIZE RAW FILES
# =========================================================
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


# =========================================================
# STEP 2: CLEAN DATA
# =========================================================
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


# =========================================================
# STEP 3: CARD FEATURES
# =========================================================
def build_card_count_matrix(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in df.iterrows():
        counts = {}
        for col in CARD_COLS:
            card = row[col]
            counts[card] = counts.get(card, 0) + 1
        records.append(counts)

    return pd.DataFrame(records).fillna(0).astype(int)


def build_step_encoded_matrix(card_matrix: pd.DataFrame) -> pd.DataFrame:
    step_features = {}

    for card in card_matrix.columns:
        max_count = int(card_matrix[card].max())

        for k in range(1, max_count + 1):
            feature_name = f"{card}_{k}"
            step_features[feature_name] = (card_matrix[card] >= k).astype(int)

    return pd.DataFrame(step_features, index=card_matrix.index)


# =========================================================
# STEP 4: LAND BUCKET FEATURES
# =========================================================
def add_land_features(clean_df: pd.DataFrame, card_info_lookup: dict) -> pd.DataFrame:
    num_lands = []

    for _, row in clean_df.iterrows():
        count = 0
        for col in CARD_COLS:
            card = str(row[col]).strip()
            if card_info_lookup.get(card, {}).get("is_land", False):
                count += 1
        num_lands.append(count)

    num_lands = pd.Series(num_lands)

    land_df = pd.DataFrame({
        "lands_0_1": (num_lands <= 1).astype(int),
        "lands_2_4": ((num_lands >= 2) & (num_lands <= 4)).astype(int),
        "lands_5_plus": (num_lands >= 5).astype(int),
    })

    return land_df.reset_index(drop=True)


# =========================================================
# STEP 5: MANA VALUE / CURVE FEATURES
# =========================================================
def cmc_to_bucket(cmc: float) -> str:
    cmc_int = int(round(cmc))
    if cmc_int <= 0:
        return "0_drops"
    if cmc_int == 1:
        return "1_drops"
    if cmc_int == 2:
        return "2_drops"
    if cmc_int == 3:
        return "3_drops"
    if cmc_int == 4:
        return "4_drops"
    if cmc_int == 5:
        return "5_drops"
    return "6_plus_drops"


def add_mana_value_features(clean_df: pd.DataFrame, card_info_lookup: dict) -> pd.DataFrame:
    rows = []

    for _, row in clean_df.iterrows():
        counts = {
            "0_drops": 0,
            "1_drops": 0,
            "2_drops": 0,
            "3_drops": 0,
            "4_drops": 0,
            "5_drops": 0,
            "6_plus_drops": 0,
        }

        for col in CARD_COLS:
            card = str(row[col]).strip()
            info = card_info_lookup.get(card, {})

            # Skip lands entirely
            if info.get("is_land", False):
                continue

            cmc = float(info.get("cmc", 0.0))
            bucket = cmc_to_bucket(cmc)
            counts[bucket] += 1

        rows.append(counts)

    return pd.DataFrame(rows).reset_index(drop=True)


# =========================================================
# STEP 6: FINAL DATASET
# =========================================================
def build_final_dataset(clean_df, step_matrix, land_features, mana_value_features):
    metadata = clean_df[["timestamp", "source_file", "on_play"]].reset_index(drop=True)
    raw_hand = clean_df[[f"card{i}" for i in range(1, 8)]].reset_index(drop=True)
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
        "land_bucket_features": ["lands_0_1", "lands_2_4", "lands_5_plus"],
        "mana_value_bucket_features": [
            "0_drops",
            "1_drops",
            "2_drops",
            "3_drops",
            "4_drops",
            "5_drops",
            "6_plus_drops",
        ],
        "raw_dir": str(RAW_DIR),
        "raw_files": raw_file_info,
        "notes": (
            "Processed dataset snapshot using step encoding, land buckets, "
            "and nonland mana-value buckets."
        ),
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

    # Backward compatibility for current train.py / evaluate.py
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

    print("Land buckets: ['lands_0_1', 'lands_2_4', 'lands_5_plus']")
    print("Mana value buckets: ['0_drops', '1_drops', '2_drops', '3_drops', '4_drops', '5_drops', '6_plus_drops']")

    return dataset_id


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nPreprocessing failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise