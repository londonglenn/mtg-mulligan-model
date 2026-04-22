from __future__ import annotations

from pathlib import Path
from collections import Counter
import json
import time

import pandas as pd
import requests

from utils import ensure_dir


# =========================================================
# CONFIG / SCHEMA
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CARD_COLS = [f"card{i}" for i in range(1, 8)]
CARD_COUNT = len(CARD_COLS)

LAND_COUNT_FEATURES = [
    "num_lands",
    "num_lands_sq",
]

MANA_VALUE_BUCKET_FEATURES = [
    "0_drops",
    "1_drops",
    "2_drops",
    "3_drops",
    "4_drops",
    "5_drops",
    "6_plus_drops",
]

FEATURE_SCHEMA_VERSION = "v3_num_lands_shared_features"

SCRYFALL_CACHE_PATH = PROCESSED_DIR / "card_info_cache.json"
SCRYFALL_NAMED_URL = "https://api.scryfall.com/cards/named"


# =========================================================
# CARD INFO CACHE HELPERS
# =========================================================
def load_card_info_cache(cache_path: Path = SCRYFALL_CACHE_PATH) -> dict:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_card_info_cache(cache: dict, cache_path: Path = SCRYFALL_CACHE_PATH) -> None:
    ensure_dir(cache_path.parent)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def fetch_card_info(card_name: str) -> dict:
    response = requests.get(
        SCRYFALL_NAMED_URL,
        params={"exact": str(card_name).strip()},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    type_line = data.get("type_line", "")
    return {
        "type_line": type_line,
        "is_land": ("Land" in type_line),
        "cmc": float(data.get("cmc", 0.0)),
        "lookup_ok": True,
    }


def get_unknown_card_info() -> dict:
    return {
        "type_line": "",
        "is_land": None,
        "cmc": None,
        "lookup_ok": False,
    }


def get_card_info(card_name: str, cache: dict, save_cache: bool = True) -> dict:
    card_name = str(card_name).strip()

    if not card_name:
        return get_unknown_card_info()

    if card_name not in cache:
        try:
            cache[card_name] = fetch_card_info(card_name)
        except Exception as e:
            print(f"Warning: could not fetch Scryfall data for '{card_name}': {e}")
            cache[card_name] = get_unknown_card_info()

        if save_cache:
            save_card_info_cache(cache)

    return cache[card_name]


def build_card_info_lookup(
    unique_cards,
    cache_path: Path = SCRYFALL_CACHE_PATH,
    sleep_seconds: float = 0.1,
) -> dict:
    cache = load_card_info_cache(cache_path)
    updated = False

    for card in sorted(unique_cards):
        if pd.isna(card):
            continue

        card = str(card).strip()
        if not card:
            continue

        if card in cache:
            continue

        try:
            cache[card] = fetch_card_info(card)
            updated = True
            time.sleep(sleep_seconds)
        except Exception as e:
            print(f"Warning: could not fetch Scryfall data for '{card}': {e}")
            cache[card] = get_unknown_card_info()
            updated = True

    if updated:
        save_card_info_cache(cache, cache_path)

    return cache


# =========================================================
# SHARED FEATURE LOGIC
# =========================================================
def cmc_to_bucket(cmc: float) -> str:
    cmc_int = int(round(float(cmc)))

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


def count_lands_in_cards(cards: list[str], card_info_lookup: dict) -> int:
    count = 0
    for card in cards:
        card = str(card).strip()
        info = card_info_lookup.get(card, {})
        if info.get("is_land") is True:
            count += 1
    return count


def build_land_count_dict(num_lands: int) -> dict[str, float]:
    return {
        "num_lands": float(num_lands),
        "num_lands_sq": float(num_lands ** 2),
    }


def build_mana_value_bucket_dict(cards: list[str], card_info_lookup: dict) -> dict[str, int]:
    counts = {name: 0 for name in MANA_VALUE_BUCKET_FEATURES}

    for card in cards:
        card = str(card).strip()
        info = card_info_lookup.get(card, {})

        if info.get("is_land") is True:
            continue

        if info.get("lookup_ok") is False:
            continue

        cmc = info.get("cmc", None)
        if cmc is None:
            continue

        bucket = cmc_to_bucket(cmc)
        counts[bucket] += 1

    return counts


def build_card_count_matrix(df: pd.DataFrame, card_cols: list[str] = CARD_COLS) -> pd.DataFrame:
    records = []

    for _, row in df.iterrows():
        counts = {}
        for col in card_cols:
            card = str(row[col]).strip()
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


def add_land_features(
    clean_df: pd.DataFrame,
    card_info_lookup: dict,
    card_cols: list[str] = CARD_COLS,
) -> pd.DataFrame:
    rows = []

    for _, row in clean_df.iterrows():
        cards = [str(row[col]).strip() for col in card_cols]
        num_lands = count_lands_in_cards(cards, card_info_lookup)
        rows.append(build_land_count_dict(num_lands))

    return pd.DataFrame(rows).reset_index(drop=True)


def add_mana_value_features(
    clean_df: pd.DataFrame,
    card_info_lookup: dict,
    card_cols: list[str] = CARD_COLS,
) -> pd.DataFrame:
    rows = []

    for _, row in clean_df.iterrows():
        cards = [str(row[col]).strip() for col in card_cols]
        rows.append(build_mana_value_bucket_dict(cards, card_info_lookup))

    return pd.DataFrame(rows).reset_index(drop=True)


def build_feature_row(
    hand: list[str],
    on_play: int,
    feature_columns: list[str],
    card_info_cache: dict,
) -> pd.DataFrame:
    if len(hand) != CARD_COUNT:
        raise ValueError(f"Expected exactly {CARD_COUNT} cards, got {len(hand)}")

    hand = [str(card).strip() for card in hand]
    row = {feature: 0 for feature in feature_columns}

    if "on_play" in row:
        row["on_play"] = int(on_play)

    counts = Counter(hand)
    for card, count in counts.items():
        for k in range(1, count + 1):
            feature_name = f"{card}_{k}"
            if feature_name in row:
                row[feature_name] = 1

    for card in hand:
        get_card_info(card, card_info_cache, save_cache=True)

    num_lands = count_lands_in_cards(hand, card_info_cache)
    land_dict = build_land_count_dict(num_lands)
    mana_dict = build_mana_value_bucket_dict(hand, card_info_cache)

    for feature_name, value in {**land_dict, **mana_dict}.items():
        if feature_name in row:
            row[feature_name] = value

    return pd.DataFrame([row], columns=feature_columns)


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "source_file", "keep", *CARD_COLS}
    return [col for col in df.columns if col not in excluded]