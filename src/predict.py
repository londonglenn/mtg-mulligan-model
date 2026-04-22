import joblib

from registry import get_run_paths, get_latest_run_id
from utils import read_json
from features import build_feature_row, load_card_info_cache


def predict(hand, on_play=1, run_id=None):
    if run_id is None:
        run_id = get_latest_run_id()

    paths = get_run_paths(run_id)

    model = joblib.load(paths["model_path"])
    feature_cols = read_json(paths["feature_columns_path"])
    metadata = read_json(paths["metadata_path"], default={})
    feature_name_map = metadata.get("feature_name_map")

    cache = load_card_info_cache()
    X = build_feature_row(hand, on_play, feature_cols, cache)

    if feature_name_map:
        X = X.rename(columns=feature_name_map)

    prob = model.predict_proba(X)[0, 1]

    return {
        "keep_probability": float(prob),
        "decision": int(prob >= 0.5),
    }