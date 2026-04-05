"""
services/specs.py

Fetches product data from SerpAPI Google Shopping (European locale) and
uses Claude to extract structured specs from the results.

Requires in .env:
    SERPAPI_KEY       — for live retail data (falls back to Claude-only if missing)
    ANTHROPIC_API_KEY — for spec extraction

Results are cached to data/specs_cache.json so the pipeline never re-fetches
a product it has already processed.
"""

import os, json, time, requests
import anthropic
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY   = os.getenv("SERPAPI_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

ROOT       = Path(__file__).parent.parent
CACHE_FILE = ROOT / "data" / "specs_cache.json"

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    return _client


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# ── SerpAPI fetch ─────────────────────────────────────────────────────────────

def _fetch_serpapi(query: str, gl: str = "de") -> str:
    """
    Search Google Shopping and return a plain-text summary of the top results.
    gl = country code for European locale (de, fr, es, it, nl …)
    """
    if not SERPAPI_KEY:
        return ""
    try:
        r = requests.get(
            "https://serpapi.com/search",
            params={
                "engine":  "google_shopping",
                "q":       query,
                "api_key": SERPAPI_KEY,
                "gl":      gl,
                "hl":      "en",
                "num":     5,
            },
            timeout=10,
        )
        if r.status_code != 200:
            return ""

        lines = []
        for item in r.json().get("shopping_results", [])[:5]:
            lines.append(f"Title: {item.get('title', '')}")
            if item.get("price"):
                lines.append(f"Price: {item['price']}")
            if item.get("rating"):
                lines.append(f"Rating: {item['rating']}/5 ({item.get('reviews', '?')} reviews)")
            if item.get("snippet"):
                lines.append(f"Description: {item['snippet']}")
            exts = item.get("extensions", [])
            if exts:
                lines.append(f"Specs: {', '.join(exts)}")
            lines.append("")
        return "\n".join(lines).strip()

    except Exception as e:
        print(f"    [SerpAPI] error for '{query}': {e}")
        return ""


# ── Claude extraction ─────────────────────────────────────────────────────────

def _extract_with_claude(model: str, brand: str, category: str, raw_data: str) -> Optional[dict]:
    """
    Ask Claude to extract structured specs from raw retail data.
    Falls back to Claude's own product knowledge when data is sparse.
    Uses Haiku for speed and low cost (~$0.001 per product).
    """
    if not ANTHROPIC_KEY:
        return None

    data_section = (
        f"Use the following data from European retail sites where available:\n\n{raw_data}"
        if raw_data
        else "No retail data was found. Use your knowledge of this product."
    )

    prompt = f"""You are a product data extraction assistant for a consumer electronics comparison platform.

Product: {brand} {model} ({category})

{data_section}

Return a JSON object with EXACTLY these fields. Use your product knowledge to fill any gaps not covered by the data above.

{{
  "cpu_score":     <integer 0-100: CPU performance relative to other {category} on the market>,
  "ram_gb":        <float: RAM in GB; use 0 if not applicable for this category>,
  "battery_h":     <float: battery life in hours; use 0 if not applicable>,
  "weight_kg":     <float: weight in kg>,
  "display_score": <integer 0-100: display quality considering resolution, panel type, and size>,
  "gpu_score":     <integer 0-100: GPU performance; 0 if integrated-only or not applicable>,
  "avg_rating":    <float 0-5: typical user rating for this product>,
  "review_count":  <integer: approximate number of user reviews online>,
  "pos_pct":       <integer 0-100: estimated % of reviews that are positive>,
  "pos_topics":    <string: comma-separated list of 2-4 things users commonly praise>,
  "neg_topics":    <string: comma-separated list of 1-3 things users commonly criticise>
}}

Return ONLY the raw JSON object. No markdown, no explanation."""

    try:
        msg = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        # Strip markdown code fences if Claude added them
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        print(f"    [Claude] extraction error for {brand} {model}: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_product_specs(
    brand: str,
    model: str,
    category: str,
    year: int,
    price: float,
    gl: str = "de",
) -> Optional[dict]:
    """
    Full pipeline: SerpAPI fetch → Claude extraction → cache.

    Returns a product dict matching SpecCheck's schema, or None if both
    SERPAPI_KEY and ANTHROPIC_API_KEY are missing.

    Cache key is brand|model|category so re-running the pipeline is free
    after the first run.
    """
    if not SERPAPI_KEY and not ANTHROPIC_KEY:
        return None

    cache     = _load_cache()
    cache_key = f"{brand}|{model}|{category}"

    if cache_key in cache:
        print(f"    [cache] {brand} {model}")
        specs = dict(cache[cache_key])
        specs.update({"name": model, "brand": brand, "year": year, "price": price})
        return specs

    # Step 1 — fetch European retail data
    query    = f"{brand} {model} specs"
    print(f"    [SerpAPI] {query} (gl={gl})")
    raw_data = _fetch_serpapi(query, gl=gl)
    if raw_data:
        time.sleep(0.5)     # be polite to SerpAPI rate limits

    # Step 2 — extract structured specs with Claude
    print(f"    [Claude] extracting specs …")
    specs = _extract_with_claude(model, brand, category, raw_data)
    if specs is None:
        return None

    # Step 3 — validate and clamp all numeric fields
    specs["cpu_score"]     = max(0, min(100, int(specs.get("cpu_score",     65))))
    specs["display_score"] = max(0, min(100, int(specs.get("display_score", 65))))
    specs["gpu_score"]     = max(0, min(100, int(specs.get("gpu_score",      0))))
    specs["avg_rating"]    = max(0.0, min(5.0, float(specs.get("avg_rating", 4.2))))
    specs["pos_pct"]       = max(0, min(100, int(specs.get("pos_pct",       78))))
    specs["review_count"]  = max(0, int(specs.get("review_count",          500)))
    specs["ram_gb"]        = max(0.0, float(specs.get("ram_gb",            0.0)))
    specs["battery_h"]     = max(0.0, float(specs.get("battery_h",        0.0)))
    specs["weight_kg"]     = max(0.0, float(specs.get("weight_kg",        0.0)))
    specs.setdefault("pos_topics", "performance,value")
    specs.setdefault("neg_topics", "")

    # Cache without the per-run fields (name/brand/year/price)
    cache[cache_key] = {k: v for k, v in specs.items()
                        if k not in ("name", "brand", "year", "price")}
    _save_cache(cache)

    specs.update({"name": model, "brand": brand, "year": year, "price": price})
    return specs
