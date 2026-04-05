"""
services/icecat.py

Fetches product specs from the Open Icecat API (free tier).
Register at https://icecat.us to get credentials, then add to .env:
    ICECAT_USERNAME=your_username
    ICECAT_PASSWORD=your_password

Usage:
    from services.icecat import get_product_specs
    specs = get_product_specs("Apple", "MacBook Air M3", "Laptops", 2024, 1299)
"""

import os, re, time, requests
from typing import Optional

ICECAT_USER = os.getenv("ICECAT_USERNAME", "")
ICECAT_PASS = os.getenv("ICECAT_PASSWORD", "")

_SEARCH  = "https://live.icecat.biz/api/search"
_PRODUCT = "https://live.icecat.biz/api"


# ── Auth & request helpers ────────────────────────────────────────────────────

def _auth():
    return (ICECAT_USER, ICECAT_PASS) if ICECAT_PASS else None

def _base_params(**kw):
    p = {"UserName": ICECAT_USER, "Language": "en"}
    p.update(kw)
    return p


# ── API calls ─────────────────────────────────────────────────────────────────

def _search(brand: str, model: str) -> Optional[str]:
    """Return the first Icecat product_id match for brand + model, or None."""
    try:
        r = requests.get(
            _SEARCH,
            params=_base_params(Brand=brand, Product=model),
            auth=_auth(),
            timeout=10,
        )
        if r.status_code == 200:
            hits = r.json().get("data", [])
            if hits:
                return str(hits[0].get("product_id", ""))
    except Exception as e:
        print(f"    [Icecat] search error ({brand} {model}): {e}")
    return None


def _fetch(product_id: str) -> Optional[dict]:
    """Fetch full product JSON by Icecat product_id."""
    try:
        r = requests.get(
            _PRODUCT,
            params=_base_params(
                icecat_id=product_id,
                Content="FeatureGroups,ReviewSummary,ReasonsToBy",
            ),
            auth=_auth(),
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"    [Icecat] fetch error (id={product_id}): {e}")
    return None


# ── Feature parsing helpers ───────────────────────────────────────────────────

def _features(raw: dict) -> dict:
    """
    Flatten all FeatureGroups from Icecat JSON into
    {feature_name_lower: raw_value_str}.
    """
    out = {}
    data = raw.get("data", raw)
    for fg in data.get("FeatureGroups", []):
        fg_data = fg.get("FeatureGroup", fg)
        for f in fg_data.get("Features", []):
            feat = f.get("Feature", f)
            name = feat.get("Name", {})
            name = name.get("Value", name) if isinstance(name, dict) else name
            val  = feat.get("Value", "")
            if name and val:
                out[str(name).lower().strip()] = str(val).strip()
    return out


def _num(features: dict, *keys) -> Optional[float]:
    """Return the first numeric value found for any of the given feature keys."""
    for k in keys:
        v = features.get(k.lower())
        if v:
            m = re.search(r"[\d.]+", v)
            if m:
                try:
                    return float(m.group())
                except ValueError:
                    pass
    return None


def _display_score(features: dict) -> float:
    """Derive a 0–100 display quality score from resolution + panel type."""
    res_w = res_h = None

    res_w = _num(features,
                 "display resolution width", "horizontal resolution",
                 "native resolution width", "screen resolution width")
    res_h = _num(features,
                 "display resolution height", "vertical resolution",
                 "native resolution height", "screen resolution height")

    # Fall back: parse "3840 x 2160" style from any resolution-named feature
    if not res_w:
        for k, v in features.items():
            if "resolution" in k:
                m = re.search(r"(\d{3,5})\s*[x×]\s*(\d{3,5})", v)
                if m:
                    res_w, res_h = float(m.group(1)), float(m.group(2))
                    break

    if res_w and res_h:
        pixels = res_w * res_h
        # Reference: 1080p≈2.1M→70, 1440p≈3.7M→82, 4K≈8.3M→95
        score = min(100, 40 + (pixels / 8_300_000) * 60)
    else:
        score = 65.0

    # Panel type bonus
    panel = (features.get("display technology", "") or
             features.get("panel type", "")).lower()
    if "oled" in panel or "amoled" in panel:
        score = min(100, score + 8)
    elif "ips" in panel or "qled" in panel:
        score = min(100, score + 4)

    return round(score, 1)


def _cpu_score(features: dict) -> float:
    """Derive a relative CPU score from clock frequency (GHz)."""
    freq = _num(features,
                "processor frequency", "cpu frequency",
                "processor clock speed", "processor speed",
                "maximum turbo frequency", "base frequency")
    if freq:
        # 1 GHz → ~30, 3 GHz → ~70, 5 GHz → ~100
        return round(min(100, max(20, freq * 18)), 1)
    return 65.0


_GPU_SCORES = {
    "rtx 4090": 100, "rtx 4080": 95, "rtx 4070 ti": 90,
    "rtx 4070": 85,  "rtx 4060 ti": 80, "rtx 4060": 74,
    "rtx 3080": 88,  "rtx 3070": 80,  "rtx 3060": 70,
    "rtx 4050": 65,  "rx 7900": 92,   "rx 7800": 84,
    "rx 7700": 76,   "rx 7600": 68,   "rx 6700": 72,
    "radeon 890m": 55, "radeon 780m": 48,
}

def _gpu_score(features: dict) -> float:
    """Derive GPU score from discrete GPU model name."""
    model = (
        features.get("discrete graphics adapter", "") or
        features.get("graphics controller model", "") or
        features.get("graphics adapter", "")
    ).lower()
    if not model:
        return 0.0
    for name, score in _GPU_SCORES.items():
        if name in model:
            return float(score)
    if any(k in model for k in ("nvidia", "geforce", "rtx", "gtx", "amd", "radeon", "rx ")):
        return 55.0
    return 0.0


# ── Main public function ──────────────────────────────────────────────────────

def get_product_specs(
    brand: str,
    model: str,
    category: str,
    year: int,
    price: float,
) -> Optional[dict]:
    """
    Search Icecat for brand + model, fetch specs, return a dict matching
    SpecCheck's product schema. Returns None if Icecat is unavailable or
    the product is not found.
    """
    if not ICECAT_USER:
        return None

    product_id = _search(brand, model)
    if not product_id:
        print(f"    [Icecat] not found: {brand} {model}")
        return None

    time.sleep(0.3)   # respect rate limits on the free tier
    raw = _fetch(product_id)
    if not raw:
        return None

    data     = raw.get("data", raw)
    features = _features(raw)

    # ── RAM ──────────────────────────────────────────────────────────────────
    ram_gb = _num(features,
                  "ram capacity", "ram", "memory capacity",
                  "internal memory", "system memory") or 8.0
    if ram_gb > 256:        # value was in MB
        ram_gb /= 1024

    # ── Battery ──────────────────────────────────────────────────────────────
    battery_h = _num(features,
                     "battery life", "battery run time",
                     "battery operating time", "playback time")
    if not battery_h:
        mah = _num(features,
                   "battery capacity", "battery energy content",
                   "typical battery capacity")
        if mah:
            # Rough mAh → hours conversion per category
            divisor = {"Smartphones": 180, "Headphones": 45,
                       "Laptops": 550, "Monitors": 0}.get(category, 550)
            battery_h = mah / divisor if divisor else 0.0
    battery_h = battery_h or 8.0

    # ── Weight ───────────────────────────────────────────────────────────────
    weight_kg = _num(features,
                     "weight", "product weight",
                     "weight with stand", "weight without stand")
    if weight_kg and weight_kg > 20:    # value was in grams
        weight_kg /= 1000
    weight_kg = weight_kg or (0.2 if category in ("Smartphones", "Headphones") else 1.5)

    # ── Derived scores ────────────────────────────────────────────────────────
    cpu_score_val     = _cpu_score(features)
    gpu_score_val     = _gpu_score(features) if category in ("Laptops", "Monitors") else 0.0
    display_score_val = _display_score(features)

    # ── Review data via TestSeek (embedded in Icecat) ─────────────────────────
    review    = data.get("ReviewSummary") or {}
    ts_score  = float(review.get("TestSeekScore") or 0)
    ts_count  = int(review.get("TestSeekReviewsCount") or 0)
    avg_rating   = round(ts_score / 20, 1) if ts_score else 4.2
    review_count = ts_count or 500
    pos_pct      = ts_score or 78.0

    # ── Reasons to buy → pos_topics ──────────────────────────────────────────
    reasons = data.get("ReasonsToBy") or []
    pos_topics = ",".join(
        r.get("ReasonToBuy", {}).get("Value", "")
        for r in reasons[:4]
        if r.get("ReasonToBuy", {}).get("Value")
    ) or "performance,value"
    neg_topics = ""     # Icecat does not provide negative feedback

    # ── Year from Icecat release date ─────────────────────────────────────────
    release = data.get("ReleaseDate") or ""
    if release:
        try:
            year = int(release[:4])
        except ValueError:
            pass

    return {
        "name":          model,
        "brand":         brand,
        "year":          year,
        "price":         price,
        "cpu_score":     cpu_score_val,
        "ram_gb":        ram_gb,
        "battery_h":     battery_h,
        "weight_kg":     weight_kg,
        "display_score": display_score_val,
        "gpu_score":     gpu_score_val,
        "avg_rating":    avg_rating,
        "review_count":  review_count,
        "pos_pct":       pos_pct,
        "pos_topics":    pos_topics,
        "neg_topics":    neg_topics,
    }
