from __future__ import annotations

from pathlib import Path
import json
from collections import Counter

import joblib
import pandas as pd
import requests

from registry import get_latest_run_id, get_run_paths, get_run
from utils import read_json


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARD_INFO_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "card_info_cache.json"
SCRYFALL_NAMED_URL = "https://api.scryfall.com/cards/named"

CARD_COUNT = 7


# =========================================================
# RUN / ARTIFACT HELPERS
# =========================================================
def resolve_run_id(run_id: str | None = None) -> str:
    """
    Resolve which run to use for prediction.

    If run_id is None or "latest", use the latest registered run.
    Otherwise validate that the requested run exists.
    """
    if run_id is None or str(run_id).strip().lower() == "latest":
        return get_latest_run_id()

    run_id = str(run_id).strip()
    run_entry = get_run(run_id)
    if run_entry is None:
        raise KeyError(f"Unknown run_id: {run_id}")

    return run_id


def load_run_artifacts(run_id: str | None = None):
    """
    Load model artifacts for the selected run.
    """
    resolved_run_id = resolve_run_id(run_id)
    paths = get_run_paths(resolved_run_id)

    model = joblib.load(paths["model_path"])
    feature_columns = read_json(paths["feature_columns_path"])
    metrics = read_json(paths["metrics_path"], default={})
    metadata = read_json(paths["metadata_path"], default={})

    return {
        "run_id": resolved_run_id,
        "paths": paths,
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "metadata": metadata,
    }


def load_threshold(metrics: dict, default: float = 0.5) -> float:
    """
    Load the prediction threshold from a run's metrics file.

    If evaluation has not yet written a chosen threshold, fall back to:
      - metrics['chosen_threshold']
      - metrics['threshold']
      - default
    """
    try:
        return float(
            metrics.get(
                "chosen_threshold",
                metrics.get("threshold", default)
            )
        )
    except Exception:
        return float(default)


# =========================================================
# CARD INFO HELPERS
# =========================================================
def load_card_info_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_card_info_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
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

    type_line = data.get("type_line", "")
    return {
        "type_line": type_line,
        "is_land": ("Land" in type_line),
        "cmc": float(data.get("cmc", 0.0)),
    }


def get_card_info(card_name: str, cache: dict) -> dict:
    card_name = str(card_name).strip()

    if not card_name:
        return {
            "type_line": "",
            "is_land": False,
            "cmc": 0.0,
        }

    if card_name not in cache:
        try:
            cache[card_name] = fetch_card_info(card_name)
            save_card_info_cache(cache, CARD_INFO_CACHE_PATH)
        except Exception as e:
            print(f"Warning: could not fetch Scryfall data for '{card_name}': {e}")
            cache[card_name] = {
                "type_line": "",
                "is_land": False,
                "cmc": 0.0,
            }
            save_card_info_cache(cache, CARD_INFO_CACHE_PATH)

    return cache[card_name]


# =========================================================
# FEATURE HELPERS
# =========================================================
def cmc_to_bucket(cmc: float) -> str:
    cmc_int = int(cmc)

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


def build_feature_row(hand, on_play, feature_columns, card_info_cache):
    """
    Build a single-row DataFrame in the same feature space used in training.

    Includes:
      - on_play
      - step/card count features
      - land bucket features
      - mana value features (nonlands only)
    """
    row = {feature: 0 for feature in feature_columns}

    # metadata
    if "on_play" in row:
        row["on_play"] = int(on_play)

    # card counts / step features
    counts = Counter(hand)
    for card, count in counts.items():
        for k in range(1, count + 1):
            feature_name = f"{card}_{k}"
            if feature_name in row:
                row[feature_name] = 1

    # card info lookup for hand
    card_infos = []
    for card in hand:
        info = get_card_info(card, card_info_cache)
        card_infos.append((card, info))

    # land bucket features
    num_lands = sum(1 for _, info in card_infos if info.get("is_land", False))

    if "lands_0_1" in row:
        row["lands_0_1"] = int(num_lands <= 1)
    if "lands_2_4" in row:
        row["lands_2_4"] = int(2 <= num_lands <= 4)
    if "lands_5_plus" in row:
        row["lands_5_plus"] = int(num_lands >= 5)

    # mana value / curve features
    mana_counts = {
        "0_drops": 0,
        "1_drops": 0,
        "2_drops": 0,
        "3_drops": 0,
        "4_drops": 0,
        "5_drops": 0,
        "6_plus_drops": 0,
    }

    for _, info in card_infos:
        # Exclude lands from mana curve buckets
        if info.get("is_land", False):
            continue

        cmc = float(info.get("cmc", 0.0))
        bucket = cmc_to_bucket(cmc)
        mana_counts[bucket] += 1

    for feature_name, value in mana_counts.items():
        if feature_name in row:
            row[feature_name] = value

    return pd.DataFrame([row], columns=feature_columns)


def explain_top_contributors(model, X_row, top_n=10):
    """
    Show the largest positive/negative contributions to the logit score.
    """
    coef = model.coef_[0]
    feature_names = list(X_row.columns)
    values = X_row.iloc[0].values

    contribs = []
    for f, v, c in zip(feature_names, values, coef):
        contribution = float(v) * float(c)
        if v != 0:
            contribs.append((f, v, c, contribution))

    contribs.sort(key=lambda x: x[3], reverse=True)

    top_positive = contribs[:top_n]
    top_negative = sorted(contribs, key=lambda x: x[3])[:top_n]

    return top_positive, top_negative


# =========================================================
# MAIN PREDICT FUNCTION
# =========================================================
def predict_hand(hand, on_play, run_id: str | None = None):
    if len(hand) != CARD_COUNT:
        raise ValueError(f"Expected exactly {CARD_COUNT} cards, got {len(hand)}")

    hand = [str(card).strip() for card in hand]

    if int(on_play) not in {0, 1}:
        raise ValueError("on_play must be 1 (play) or 0 (draw)")

    artifacts = load_run_artifacts(run_id=run_id)
    resolved_run_id = artifacts["run_id"]
    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]
    metrics = artifacts["metrics"]
    metadata = artifacts["metadata"]

    card_info_cache = load_card_info_cache(CARD_INFO_CACHE_PATH)
    X_row = build_feature_row(hand, on_play, feature_columns, card_info_cache)

    prob_keep = float(model.predict_proba(X_row)[0, 1])

    if hasattr(model, "decision_function"):
        logit_score = float(model.decision_function(X_row)[0])
    else:
        logit_score = None

    threshold = load_threshold(metrics=metrics, default=0.5)
    pred_class = int(prob_keep >= threshold)

    print(f"\nRun ID: {resolved_run_id}")
    print("Hand:")
    for card in hand:
        print(f" - {card}")

    print(f"\non_play: {on_play} ({'play' if int(on_play) == 1 else 'draw'})")
    print(f"Threshold: {threshold:.4f}")

    if logit_score is not None:
        print(f"Logit score (w·x + b): {logit_score:.4f}")
    else:
        print("Logit score: unavailable for this model type")

    print(f"Predicted class: {'KEEP' if pred_class == 1 else 'MULLIGAN'}")
    print(f"Keep probability: {prob_keep:.4f}")
    print(f"Mulligan probability: {1 - prob_keep:.4f}")

    top_positive, top_negative = explain_top_contributors(model, X_row, top_n=5)

    print("\nTop positive contributors:")
    if top_positive:
        for f, v, c, contrib in top_positive:
            print(f"  {f}: value={v}, coef={c:.4f}, contribution={contrib:.4f}")
    else:
        print("  None")

    print("\nTop negative contributors:")
    if top_negative:
        for f, v, c, contrib in top_negative:
            print(f"  {f}: value={v}, coef={c:.4f}, contribution={contrib:.4f}")
    else:
        print("  None")

    return {
        "run_id": resolved_run_id,
        "experiment_id": metadata.get("experiment_id"),
        "dataset_id": metadata.get("dataset_id"),
        "hand": hand,
        "on_play": int(on_play),
        "threshold": threshold,
        "logit_score": logit_score,
        "prob_keep": prob_keep,
        "prob_mulligan": 1 - prob_keep,
        "pred_class": pred_class,
        "pred_label": "KEEP" if pred_class == 1 else "MULLIGAN",
        "top_positive": top_positive,
        "top_negative": top_negative,
    }


# =========================================================
# EXAMPLE USAGE
# =========================================================
if __name__ == "__main__":
    hand = [
        "Sacred Foundry",
        "Sacred Foundry",
        "Goblin Bombardment",
        "Ragavan, Nimble Pilferer",
        "Lightning Bolt",
        "Arid Mesa",
        "Mountain",
    ]

    on_play = 1

    # Uses latest run by default
    predict_hand(hand, on_play)

    # Or use a specific run:
    # predict_hand(hand, on_play, run_id="run_20260420_104000_aa11bb22")