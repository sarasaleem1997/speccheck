"""
services/scoring.py

Computes a 0-100 match score for each product given a user profile.
Uses the trained GradientBoostingRegressor model as a base, then applies
use-case weight adjustments on top.
"""

import pickle, json, os
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR  = ROOT / "data"

_model_cache = {}
_meta_cache  = {}


def _load_model(category: str):
    key = category.lower()
    if key not in _model_cache:
        path = MODEL_DIR / f"model_{key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _model_cache[key] = pickle.load(f)
        else:
            _model_cache[key] = None
    return _model_cache[key]


def _load_meta():
    if not _meta_cache:
        path = DATA_DIR / "metadata.json"
        if path.exists():
            with open(path) as f:
                _meta_cache.update(json.load(f))
    return _meta_cache


def compute_scores(
    df: pd.DataFrame,
    category: str,
    use_case: str,
    budget_range: tuple,
    portability: str,
    priorities: list = None,
) -> pd.DataFrame:
    """
    Returns df with a 'match_score' column (0-100), sorted descending.
    Also adds 'score_breakdown' dict column for the radar chart.
    priorities: ranked list of user-selected feature priorities (① first = most important)
    """
    meta = _load_meta()
    model = _load_model(category)

    feature_cols = ["cpu_norm","gpu_norm","display_norm","battery_norm",
                    "weight_norm","ram_norm","price_norm","pos_pct"]

    df = df.copy()

    # filter by budget
    lo, hi = budget_range
    df = df[(df["price"] >= lo) & (df["price"] <= hi)].copy()
    if df.empty:
        return df

    # model base score
    X = df[feature_cols].fillna(50)
    if model is not None:
        base_scores = model.predict(X)
    else:
        base_scores = df["base_score"].values

    # use-case weight adjustment
    weights = (meta.get("use_case_weights", {})
                   .get(category, {})
                   .get(use_case, {}))

    adjustments = np.zeros(len(df))
    for col, w in weights.items():
        norm_col = col if col.endswith("_norm") else (
            "battery_norm" if col == "battery_h" else
            "weight_norm"  if col == "weight_kg" else
            "price_norm"   if col == "price"     else
            "ram_norm"     if col == "ram_gb"    else
            col + "_norm"  if col + "_norm" in df.columns else col
        )
        if norm_col in df.columns:
            adjustments += df[norm_col].fillna(50).values * w * 0.4

    # portability modifier
    port_mod = {"Mainly at desk": 0, "Occasional travel": 0.5, "Always on the go": 1.0}
    p = port_mod.get(portability, 0)
    if "weight_norm" in df.columns:
        adjustments += df["weight_norm"].fillna(50).values * p * 0.15

    # priority boost: reward products praised for the user's chosen priorities
    # ① gets 3x weight, ② gets 2x, ③ gets 1x
    if priorities:
        priority_weights = [3.0, 2.0, 1.0]
        pos_topics_col = df["pos_topics"].fillna("").str.lower()
        for i, prio in enumerate(priorities[:3]):
            boost = priority_weights[i]
            keyword = prio.lower().strip()
            # check if the keyword appears in pos_topics for each product
            matches = pos_topics_col.str.contains(keyword, regex=False).astype(float)
            adjustments += matches.values * boost * 4.0

    raw = base_scores + adjustments
    lo_r, hi_r = raw.min(), raw.max()
    if hi_r > lo_r:
        match_scores = 55 + (raw - lo_r) / (hi_r - lo_r) * 40
    else:
        match_scores = np.full(len(df), 75.0)

    df["match_score"] = match_scores.round(0).astype(int)

    # breakdown for radar chart (5 axes)
    def breakdown(row):
        return {
            "Performance": int(row.get("cpu_norm", 50) * 0.6 + row.get("gpu_norm", 50) * 0.4),
            "Battery":     int(row.get("battery_norm", 50)),
            "Portability": int(row.get("weight_norm", 50)),
            "Display":     int(row.get("display_norm", 50)),
            "Value":       int(row.get("price_norm", 50)),
        }

    df["score_breakdown"] = df.apply(breakdown, axis=1)
    return df.sort_values("match_score", ascending=False).reset_index(drop=True)
