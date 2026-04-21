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


def load_run_artifacts(run_id: str | None = None) -> dict:
    """
    Load model artifacts for the selected run from the local run registry.
    """
    resolved_run_id = resolve_run_id(run_id)
    paths = get_run_paths(resolved_run_id)

    model = joblib.load(paths["model_path"])
    feature_columns = read_json(paths["feature_columns_path"])
    metrics = read_json(paths["metrics_path"], default={})
    metadata = read_json(paths["metadata_path"], default={})

    model_version = (
        metadata.get("model_version")
        or metadata.get("bundle_version")
        or resolved_run_id
    )

    return {
        "source": "local_run",
        "run_id": resolved_run_id,
        "bundle_version": model_version,
        "paths": paths,
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "metadata": metadata,
    }


def load_threshold(metrics: dict, default: float = 0.5) -> float:
    """
    Load the prediction threshold from metrics.
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


def load_bundle_threshold(bundle: dict, default: float = 0.5) -> float:
    """
    Load threshold from a generic bundle dict.
    """
    metrics = bundle.get("metrics", {})
    metadata = bundle.get("metadata", {})

    try:
        return float(
            metrics.get(
                "chosen_threshold",
                metrics.get(
                    "threshold",
                    metadata.get(
                        "chosen_threshold",
                        metadata.get("threshold", default)
                    )
                )
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

    if "on_play" in row:
        row["on_play"] = int(on_play)

    counts = Counter(hand)
    for card, count in counts.items():
        for k in range(1, count + 1):
            feature_name = f"{card}_{k}"
            if feature_name in row:
                row[feature_name] = 1

    card_infos = []
    for card in hand:
        info = get_card_info(card, card_info_cache)
        card_infos.append((card, info))

    num_lands = sum(1 for _, info in card_infos if info.get("is_land", False))

    if "lands_0_1" in row:
        row["lands_0_1"] = int(num_lands <= 1)
    if "lands_2_4" in row:
        row["lands_2_4"] = int(2 <= num_lands <= 4)
    if "lands_5_plus" in row:
        row["lands_5_plus"] = int(num_lands >= 5)

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
    Return the largest positive/negative contributions to the logit score.
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
# HUMAN-READABLE EXPLANATION HELPERS
# =========================================================
def humanize_feature_name(feature_name: str, value, contribution: float) -> str:
    """
    Convert a model feature into user-facing explanation text.
    """
    direction = "supports KEEP" if contribution >= 0 else "supports MULLIGAN"

    if feature_name == "on_play":
        return "You are on the play" if int(value) == 1 else "You are on the draw"

    if feature_name == "lands_0_1":
        return "This hand has 0-1 lands"
    if feature_name == "lands_2_4":
        return "This hand has 2-4 lands"
    if feature_name == "lands_5_plus":
        return "This hand has 5 or more lands"

    if feature_name in {
        "0_drops", "1_drops", "2_drops", "3_drops", "4_drops", "5_drops", "6_plus_drops"
    }:
        bucket_label = feature_name.replace("_", " ")
        return f"{int(value)} cards in the {bucket_label} bucket ({direction})"

    if feature_name.endswith(("_1", "_2", "_3", "_4")):
        parts = feature_name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            card_name, copies = parts
            return f"Contains at least {copies} copy/copies of {card_name}"

    return f"{feature_name} = {value} ({direction})"


def build_reason_strings(top_positive, top_negative, max_reasons: int = 3) -> list[str]:
    """
    Build a short list of readable reasons for the GUI.
    """
    reasons = []

    combined = []
    combined.extend(top_positive[:max_reasons])
    combined.extend(top_negative[:max_reasons])

    # sort by absolute impact
    combined = sorted(combined, key=lambda x: abs(x[3]), reverse=True)

    seen = set()
    for feature_name, value, coef, contribution in combined:
        text = humanize_feature_name(feature_name, value, contribution)
        if text not in seen:
            reasons.append(text)
            seen.add(text)
        if len(reasons) >= max_reasons:
            break

    return reasons


# =========================================================
# SHARED PREDICTION CORE
# =========================================================
def predict_from_bundle(hand, on_play, bundle: dict) -> dict:
    """
    Predict from a preloaded bundle dict.

    Expected keys:
      - model
      - feature_columns
      - metrics
      - metadata
      - optional run_id
      - optional bundle_version
    """
    if len(hand) != CARD_COUNT:
        raise ValueError(f"Expected exactly {CARD_COUNT} cards, got {len(hand)}")

    hand = [str(card).strip() for card in hand]

    if int(on_play) not in {0, 1}:
        raise ValueError("on_play must be 1 (play) or 0 (draw)")

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    metrics = bundle.get("metrics", {})
    metadata = bundle.get("metadata", {})

    run_id = bundle.get("run_id")
    bundle_version = (
        bundle.get("bundle_version")
        or metadata.get("model_version")
        or metadata.get("bundle_version")
        or run_id
        or "unknown"
    )

    card_info_cache = load_card_info_cache(CARD_INFO_CACHE_PATH)
    X_row = build_feature_row(hand, on_play, feature_columns, card_info_cache)

    prob_keep = float(model.predict_proba(X_row)[0, 1])

    if hasattr(model, "decision_function"):
        logit_score = float(model.decision_function(X_row)[0])
    else:
        logit_score = None

    threshold = load_bundle_threshold(bundle=bundle, default=0.5)
    pred_class = int(prob_keep >= threshold)

    top_positive, top_negative = explain_top_contributors(model, X_row, top_n=5)
    reasons = build_reason_strings(top_positive, top_negative, max_reasons=3)

    return {
        "enabled": True,
        "model_version": bundle_version,
        "run_id": run_id,
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
        "reasons": reasons,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }


# =========================================================
# GUI-FRIENDLY ENTRY POINT
# =========================================================
def predict_hand_for_app(hand, on_play, bundle: dict) -> dict:
    """
    GUI-friendly prediction call.

    The GUI should pass in a preloaded bundle to avoid reloading the model
    from disk every hand.
    """
    return predict_from_bundle(hand=hand, on_play=on_play, bundle=bundle)


# =========================================================
# EXISTING LOCAL-RUN ENTRY POINT
# =========================================================
def predict_hand(hand, on_play, run_id: str | None = None):
    """
    Local CLI/testing entry point using the registry + local run folder.
    """
    artifacts = load_run_artifacts(run_id=run_id)
    result = predict_from_bundle(hand=hand, on_play=on_play, bundle=artifacts)

    print(f"\nRun ID: {result.get('run_id')}")
    print("Hand:")
    for card in result["hand"]:
        print(f" - {card}")

    print(f"\non_play: {result['on_play']} ({'play' if int(result['on_play']) == 1 else 'draw'})")
    print(f"Threshold: {result['threshold']:.4f}")

    if result["logit_score"] is not None:
        print(f"Logit score (w·x + b): {result['logit_score']:.4f}")
    else:
        print("Logit score: unavailable for this model type")

    print(f"Predicted class: {result['pred_label']}")
    print(f"Keep probability: {result['prob_keep']:.4f}")
    print(f"Mulligan probability: {result['prob_mulligan']:.4f}")

    print("\nReasons:")
    if result["reasons"]:
        for reason in result["reasons"]:
            print(f"  - {reason}")
    else:
        print("  None")

    print("\nTop positive contributors:")
    if result["top_positive"]:
        for f, v, c, contrib in result["top_positive"]:
            print(f"  {f}: value={v}, coef={c:.4f}, contribution={contrib:.4f}")
    else:
        print("  None")

    print("\nTop negative contributors:")
    if result["top_negative"]:
        for f, v, c, contrib in result["top_negative"]:
            print(f"  {f}: value={v}, coef={c:.4f}, contribution={contrib:.4f}")
    else:
        print("  None")

    return result


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