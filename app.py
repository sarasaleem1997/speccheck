"""SpecCheck — AI-powered product comparison. Run: streamlit run app.py"""

import os, json, base64, requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API key compatibility: .env (local) → st.secrets (Streamlit Cloud) ────────
# If keys aren't in the environment (no .env file), try Streamlit secrets
import streamlit as st
for _key in ("ANTHROPIC_API_KEY", "SERPAPI_KEY"):
    if not os.getenv(_key):
        try:
            _val = st.secrets.get(_key, "")
            if _val:
                os.environ[_key] = _val
        except Exception:
            pass

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"

st.set_page_config(page_title="SpecCheck", page_icon="⬡", layout="centered",
                   initial_sidebar_state="collapsed")

from services.scoring import compute_scores
from services.prices  import get_prices_batch
from services.llm     import generate_verdict, stream_chat

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif !important; }
.stApp { background:#f6f7fb !important; }
#MainMenu, footer, header { visibility:hidden; }
section[data-testid="stSidebar"] { display:none !important; }

/* ── Card hover lift ── */
.sc-card {
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    cursor: pointer;
}
.sc-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 12px 32px rgba(0,0,0,0.10) !important;
}

/* ── Layout ── */
.block-container {
    max-width: 1170px !important;
    margin: 0 auto !important;
    padding-top: 0 !important;
    padding-bottom: 80px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ── Buttons ── */
div.stButton > button {
    font-family:'Inter',sans-serif !important;
    font-size:14px !important;
    font-weight:500 !important;
    border-radius:4px !important;
    padding:10px 20px !important;
    min-height:44px !important;
    height:auto !important;
    line-height:1.4 !important;
    white-space:normal !important;
    transition:all 0.15s ease !important;
    background:#ffffff !important;
    color:#161616 !important;
    border:1.5px solid #dfdfdf !important;
    box-shadow:none !important;
}
div.stButton > button:hover {
    border-color:#9ca3af !important;
    background:#f9fafb !important;
}
div.stButton > button[kind="primary"] {
    background:#3c59fc !important;
    color:#ffffff !important;
    border-color:#3c59fc !important;
    font-weight:600 !important;
}
div.stButton > button[kind="primary"]:hover {
    background:#2a45e8 !important;
    border-color:#2a45e8 !important;
}
div.stButton > button:disabled { opacity:0.4 !important; }

/* ── Text input ── */
div.stTextInput > div > div > input {
    font-family:'Inter',sans-serif !important;
    border:1.5px solid #dfdfdf !important;
    border-radius:4px !important;
    font-size:14px !important;
    color:#161616 !important;
    background:#fff !important;
    padding:10px 14px !important;
    box-shadow:none !important;
}
div.stTextInput > div > div > input:focus {
    border-color:#3c59fc !important;
    box-shadow: 0 0 0 3px rgba(60,89,252,0.12) !important;
}
div.stTextInput > label { display:none !important; }

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    background:transparent !important;
    border-bottom:1px solid #e7e8e7 !important;
    gap:0 !important; padding:0 !important;
}
button[data-baseweb="tab"] {
    font-family:'Inter',sans-serif !important;
    font-size:14px !important; font-weight:500 !important;
    color:#616161 !important;
    background:transparent !important; border:none !important;
    border-bottom:2px solid transparent !important;
    border-radius:0 !important;
    padding:12px 20px !important; margin-bottom:-1px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color:#161616 !important;
    border-bottom-color:#3c59fc !important;
    font-weight:600 !important;
    background:transparent !important;
}
div[data-baseweb="tab-panel"] { padding:0 !important; background:transparent !important; }

/* ── Chat input ── */
div[data-testid="stChatInput"] { background:transparent !important; }
div[data-testid="stChatInput"] > div {
    border:1.5px solid #dfdfdf !important;
    border-radius:4px !important;
    background:#ffffff !important;
    padding:4px !important;
}
div[data-testid="stChatInput"] textarea {
    background:#ffffff !important; color:#161616 !important;
    font-family:'Inter',sans-serif !important; font-size:14px !important;
}
div[data-testid="stChatInput"] * { background-color:transparent !important; }

.stSpinner > div { border-top-color:#3c59fc !important; }

/* ── Budget slider + checkbox: blue, value text black ── */
div[data-testid="stSlider"] > div > div > div > div {
    background:#3c59fc !important;
}
div[data-testid="stSlider"] > div > div > div > div > div {
    background:#3c59fc !important;
    border-color:#3c59fc !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background:#3c59fc !important;
    border-color:#3c59fc !important;
    box-shadow:0 0 0 4px rgba(60,89,252,0.15) !important;
}
div[data-testid="stSlider"] p,
div[data-testid="stSlider"] span { color:#1c1c1a !important; }
/* Checkbox — transparent backgrounds throughout */
div[data-testid="stCheckbox"],
div[data-testid="stCheckbox"] label {
    background-color: transparent !important;
    background: transparent !important;
}

/* Kill Streamlit's red focus ring */
div[data-testid="stCheckbox"] label:focus,
div[data-testid="stCheckbox"] label:focus-visible,
div[data-testid="stCheckbox"] label > span:focus,
div[data-testid="stCheckbox"] label > span:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}

/* Text: always black, transparent bg, no change on interaction */
div[data-testid="stCheckbox"] label > div,
div[data-testid="stCheckbox"] label > div * {
    color: #1c1c1a !important;
    background-color: transparent !important;
}

/* Visual square (StyledCheckmark span) — always black outline */
div[data-testid="stCheckbox"] label > span:first-child {
    border: 1.5px solid #1c1c1a !important;
    border-radius: 4px !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

/* Unchecked: hide the tick SVG but keep the span's size so border stays visible */
div[data-testid="stCheckbox"] label:not(:has(input:checked)) > span:first-child svg {
    visibility: hidden !important;
}

/* Checked: fill square black, show white tick */
div[data-testid="stCheckbox"] label:has(input:checked) > span:first-child {
    background-color: #1c1c1a !important;
    border-color: #1c1c1a !important;
}
div[data-testid="stCheckbox"] label:has(input:checked) > span:first-child svg {
    visibility: visible !important;
    fill: #ffffff !important;
}
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:#dfdfdf; border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ── Tokens ────────────────────────────────────────────────────────────────────
INK   = "#161616"; INK2  = "#616161"; INK3  = "#858585"
LINE  = "#e7e8e7"; SURF  = "#f6f7fb"; WHITE = "#ffffff"
SEL   = "#3c59fc"; UNSEL = "#ffffff"
NAV   = "#161616"   # dark navbar (versus.com style)

PRIMARY    = "#3c59fc"
PRIMARY_BG = "#eef1ff"

CAT_PAL = {
    "Laptops":     {"accent": "#7c3aed", "bg": "#f5f3ff", "text": "#5b21b6"},
    "Smartphones": {"accent": "#0284c7", "bg": "#f0f9ff", "text": "#075985"},
    "Headphones":  {"accent": "#d97706", "bg": "#fffbeb", "text": "#92400e"},
    "Monitors":    {"accent": "#059669", "bg": "#ecfdf5", "text": "#065f46"},
}
CAT_ICON = {"Laptops":"💻","Smartphones":"📱","Headphones":"🎧","Monitors":"🖥️"}

# SVG icon used for Monitors category card (emoji renders poorly on dark bg)
_MONITOR_SVG = (
    '<svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="2" y="3" width="34" height="23" rx="3" stroke="white" stroke-width="2.5"/>'
    '<path d="M13 26l-2.5 7h15l-2.5-7" stroke="white" stroke-width="2.5" stroke-linejoin="round"/>'
    '<line x1="10" y1="33" x2="28" y2="33" stroke="white" stroke-width="2.5" stroke-linecap="round"/>'
    '<line x1="12" y1="10" x2="26" y2="10" stroke="white" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>'
    '<line x1="12" y1="15" x2="22" y2="15" stroke="white" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>'
    '</svg>'
)

@st.cache_data
def _get_prod_count(cat):
    p = DATA_DIR / f"products_{cat.lower()}.parquet"
    return len(pd.read_parquet(p)) if p.exists() else 0

# Normalize similar/variant topic strings to a single canonical label.
# Order matters — more specific patterns must come before broader ones.
_TOPIC_NORM = [
    # Noise / ANC
    ("active noise",    "Noise Cancellation"),
    ("noise cancell",   "Noise Cancellation"),
    ("noise isol",      "Noise Cancellation"),
    # Comfort
    ("comfort",         "Comfort"),
    # Sound
    ("sound quality",   "Sound Quality"),
    ("audio quality",   "Sound Quality"),
    # Camera
    ("camera qual",     "Camera Quality"),
    ("camera perf",     "Camera Quality"),
    ("camera",          "Camera Quality"),
    # Display
    ("display qual",    "Display Quality"),
    ("oled display",    "Display Quality"),
    ("sharp.*display",  "Display Quality"),
    ("display",         "Display Quality"),
    # Battery
    ("battery life",    "Battery Life"),
    ("long battery",    "Battery Life"),
    ("battery",         "Battery Life"),
    # Charging
    ("fast charg",      "Fast Charging"),
    ("charg speed",     "Fast Charging"),
    ("charging",        "Fast Charging"),
    # Value
    ("value for mon",   "Value for Money"),
    ("affordable",      "Value for Money"),
    ("value",           "Value for Money"),
    # Build / Durability
    ("build quality",   "Build Quality"),
    ("premium build",   "Build Quality"),
    ("durabl",          "Build Quality"),
    ("build",           "Build Quality"),
    # Gaming / Performance
    ("gaming perf",     "Gaming Performance"),
    ("gaming",          "Gaming Performance"),
    ("decent perf",     "Performance"),
    ("performance",     "Performance"),
    # Portability / Weight
    ("portab",          "Portability"),
    ("weight",          "Portability"),
    # Monitor-specific
    ("color accur",     "Color Accuracy"),
    ("ips panel",       "Color Accuracy"),
    ("refresh rate",    "High Refresh Rate"),
    ("hz refresh",      "High Refresh Rate"),
    (" hz",             "High Refresh Rate"),
    ("hz ",             "High Refresh Rate"),
    ("4k resol",        "4K Resolution"),
    ("wqhd",            "4K Resolution"),
    ("1440p",           "4K Resolution"),
    ("resolution",      "High Resolution"),
    ("response time",   "Low Response Time"),
    ("input lag",       "Low Response Time"),
    ("hdr",             "HDR Support"),
    ("ultrawide",       "Ultrawide Display"),
    ("curved",          "Curved Display"),
    ("usb",             "Connectivity"),
    ("ergon",           "Ergonomics"),
    # Laptop-specific
    ("keyboard",        "Keyboard"),
    ("2-in-1",          "Versatile Design"),
    ("versatil",        "Versatile Design"),
    ("everyday",        "Everyday Use"),
    # Monitor verbose labels
    ("oled panel",      "Display Quality"),
    ("full hd",         "High Resolution"),
    ("hd image",        "High Resolution"),
    ("pivot",           "Ergonomics"),
    ("height adjust",   "Ergonomics"),
    ("flicker",         "Eye Comfort"),
    ("compact design",  "Design"),
    # Misc
    ("ai feature",      "AI Features"),
    ("software",        "Software & Updates"),
    ("update",          "Software & Updates"),
    ("android",         "Software & Updates"),
]

def _normalize_topic(raw: str) -> str:
    t = raw.lower()
    for pattern, canonical in _TOPIC_NORM:
        if pattern in t:
            return canonical
    return raw.strip().title()

@st.cache_data
def _get_dynamic_priorities(cat):
    """Build priority options from the most-praised topics in the dataset."""
    from collections import Counter
    df = load_products(cat)
    if df.empty:
        return PRIORITIES.get(cat, [])
    topics = []
    for val in df["pos_topics"].dropna():
        for t in val.split(","):
            t = t.strip()
            if t:
                topics.append(_normalize_topic(t))
    counts = Counter(topics)
    seen, result = set(), []
    for t, _ in counts.most_common(20):
        if t.lower() not in seen and len(t) > 2:
            seen.add(t.lower())
            result.append(t)
    if "Other" not in result:
        result.append("Other")
    return result or PRIORITIES.get(cat, [])

# Use-case → keywords that boost relevant priorities to the top
_UC_BOOST = {
    "Laptops": {
        "Gaming":               ["gaming","gpu","performance","display","cooling","refresh"],
        "Work & productivity":  ["battery","keyboard","display","portability","performance"],
        "Travel & portability": ["weight","battery","portability","build"],
        "University":           ["battery","value","weight","portability"],
        "Programming":          ["display","keyboard","performance","battery"],
        "Creative work":        ["display","gpu","color","performance","ram"],
    },
    "Smartphones": {
        "Photography":          ["camera","display","processing","photo"],
        "Gaming":               ["performance","display","battery","cooling"],
        "Long battery life":    ["battery","charging","endurance"],
        "Everyday use":         ["battery","performance","display","value"],
        "Business":             ["software","security","display","battery"],
        "Value for money":      ["value","price","battery","performance"],
    },
    "Headphones": {
        "Music & hi-fi":        ["sound","audio","quality","driver"],
        "Office calls":         ["mic","call","anc","comfort"],
        "Travel & commute":     ["anc","portability","battery","noise"],
        "Gaming":               ["sound","mic","comfort","positioning"],
        "Gym & sport":          ["portability","fit","weight"],
    },
    "Monitors": {
        "Design & color work":  ["color","accuracy","resolution","calibration"],
        "Gaming":               ["refresh","response","gaming","hz"],
        "Office & coding":      ["ergonomics","eye","comfort","resolution"],
        "Home cinema":          ["color","resolution","contrast","hdr"],
        "Content creation":     ["color","resolution","accuracy","size"],
    },
}

def _ordered_priorities(cat, sel_uses):
    """Return dataset-derived priorities, reordered by relevance to selected use cases."""
    options = _get_dynamic_priorities(cat)
    if not sel_uses:
        return options
    boost_kws = set()
    for use in sel_uses:
        for key, kws in _UC_BOOST.get(cat, {}).items():
            if use.lower() in key.lower() or key.lower() in use.lower():
                boost_kws.update(kws)
    if not boost_kws:
        return options
    def _score(p):
        return sum(1 for kw in boost_kws if kw in p.lower())
    pinned   = [p for p in options if _score(p) > 0]
    rest     = [p for p in options if _score(p) == 0]
    return pinned + rest

OK="#065f46"; OK_BG="#ecfdf5"; WARN="#92400e"; WARN_BG="#fffbeb"; BAD="#991b1b"; BAD_BG="#fef2f2"

BUDGET_RANGES = {
    "Laptops":     [("Under $800",(0,800)),("$800-$1,400",(800,1400)),("$1,400-$2,000",(1400,2000)),("No limit",(0,99999))],
    "Smartphones": [("Under $400",(0,400)),("$400-$800",(400,800)),("$800-$1,200",(800,1200)),("No limit",(0,99999))],
    "Headphones":  [("Under $150",(0,150)),("$150-$300",(150,300)),("$300+",(300,99999)),("No limit",(0,99999))],
    "Monitors":    [("Under $400",(0,400)),("$400-$800",(400,800)),("$800+",(800,99999)),("No limit",(0,99999))],
}
# Slider config: (min, max, step, default) per category
BUDGET_SLIDER = {
    "Laptops":     (300,  3000, 50,  1500),
    "Smartphones": (100,  1500, 25,   800),
    "Headphones":  (30,    600, 10,   300),
    "Monitors":    (100,  2500, 50,   800),
}
USE_CASES = {
    "Laptops":     ["Work & productivity","Creative work","Gaming","University","Travel & portability","Programming","Other"],
    "Smartphones": ["Everyday use","Photography","Gaming","Business","Long battery life","Value for money","Other"],
    "Headphones":  ["Music & hi-fi","Office calls","Travel & commute","Gaming","Gym & sport","Casual listening","Other"],
    "Monitors":    ["Design & color work","Gaming","Office & coding","Home cinema","Content creation","Other"],
}
PRIORITIES = {
    "Laptops":     ["Performance","Battery life","Display quality","Light & portable","Value for money","Build quality","Keyboard feel","Other"],
    "Smartphones": ["Camera quality","Battery life","Performance","Display","Compact size","Software updates","Value","Other"],
    "Headphones":  ["Sound quality","Noise cancellation","Comfort","Battery life","Mic quality","Portability","Value","Other"],
    "Monitors":    ["Color accuracy","Refresh rate","Resolution","Screen size","Connectivity","Ergonomics","Value","Other"],
}
PORTABILITY = {
    "Laptops":     ["Mainly at my desk","A few times a week","Daily commuter","Other"],
    "Smartphones": ["Not important","Somewhat important","Very important","Other"],
    "Headphones":  ["Home use only","Occasional travel","Daily on-the-go","Other"],
    "Monitors":    ["Fixed desk setup","Moved occasionally","Needs to be portable","Other"],
}

@st.cache_data
def _logo_b64():
    p = ROOT / "logo.png"
    if p.exists():
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return None

FEATURED_PRODUCTS = [
    {"name":"MacBook Air M3",         "brand":"Apple",   "year":2024,"price":1299,"category":"Laptops"},
    {"name":"Dell XPS 15 9530",       "brand":"Dell",    "year":2023,"price":1499,"category":"Laptops"},
    {"name":"iPhone 15 Pro",          "brand":"Apple",   "year":2023,"price":999, "category":"Smartphones"},
    {"name":"Samsung Galaxy S24 Ultra","brand":"Samsung","year":2024,"price":1299,"category":"Smartphones"},
    {"name":"Sony WH-1000XM5",        "brand":"Sony",    "year":2022,"price":299, "category":"Headphones"},
    {"name":"Apple AirPods Pro 2",    "brand":"Apple",   "year":2022,"price":249, "category":"Headphones"},
    {"name":"LG 27GP850-B",           "brand":"LG",      "year":2021,"price":349, "category":"Monitors"},
    {"name":"Dell UltraSharp U2723DE","brand":"Dell",    "year":2022,"price":699, "category":"Monitors"},
]

def _init():
    for k,v in dict(step=0,category="Laptops",sel_uses=[],sel_prios=[],
                    sel_port="",sel_budget="",results=None,prices={},
                    verdict="",chat_history=[],prices_fetched=False,
                    verdict_fetched=False,budget_warning=None,
                    featured_product=None,fp_prices={},
                    browse_selected=[],browse_comparing=False,browse_chat=[],browse_search="",browse_prefill=None).items():
        if k not in st.session_state:
            st.session_state[k]=v
_init()

_IMG_CACHE_FILE = ROOT / "data" / "image_cache.json"

def _load_img_cache() -> dict:
    if _IMG_CACHE_FILE.exists():
        try:
            return json.loads(_IMG_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}

def _save_img_cache(cache: dict):
    _IMG_CACHE_FILE.parent.mkdir(exist_ok=True)
    _IMG_CACHE_FILE.write_text(json.dumps(cache, indent=2))

@st.cache_data(ttl=86400)
def _fetch_product_image(product_name: str) -> str | None:
    """Fetch a product image URL via SerpAPI Google Images. Returns URL or None."""
    serpapi_key = os.getenv("SERPAPI_KEY", "")
    if not serpapi_key:
        return None

    cache = _load_img_cache()
    if product_name in cache:
        return cache[product_name]

    try:
        r = requests.get(
            "https://serpapi.com/search",
            params={
                "engine":  "google_images",
                "q":       f"{product_name} official product",
                "api_key": serpapi_key,
                "num":     3,
                "safe":    "active",
            },
            timeout=8,
        )
        if r.status_code == 200:
            results = r.json().get("images_results", [])
            for item in results:
                url = item.get("original") or item.get("thumbnail")
                if url and url.startswith("http"):
                    cache[product_name] = url
                    _save_img_cache(cache)
                    return url
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def load_products(cat):
    p = DATA_DIR / f"products_{cat.lower()}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

def _bkdn(row):
    return {
        "Performance": int(row.get("cpu_norm",50)*.6 + row.get("gpu_norm",50)*.4),
        "Battery": int(row.get("battery_norm",50)),
        "Portability": int(row.get("weight_norm",50)),
        "Display": int(row.get("display_norm",50)),
        "Value": int(row.get("price_norm",50)),
    }

def topbar():
    cat = st.session_state.category
    step = st.session_state.step
    pal = CAT_PAL[cat]
    crumbs = ["Category", "Your needs", "Results"]
    parts = []
    for i, s in enumerate(crumbs):
        if i < step:
            c, w = INK3, "400"
            dot = f'<span style="width:6px;height:6px;border-radius:50%;background:#3c59fc;display:inline-block;margin-right:6px;vertical-align:middle"></span>'
        elif i == step:
            c, w = INK, "600"
            dot = ""
        else:
            c, w = INK3, "400"
            dot = ""
        parts.append(
            f'<span style="font-size:15px;color:{c};font-weight:{w}">{dot}{s}</span>'
        )
    sep = f'<span style="color:{LINE};margin:0 10px;font-size:15px">›</span>'
    st.markdown(f"""
    <div style="background:{WHITE};border-bottom:1px solid {LINE};padding:0;margin:0 -2rem 0 -2rem">
      <div style="padding:0 2rem;height:80px;display:flex;align-items:center;justify-content:space-between">
        <div style="display:flex;align-items:center">
          {f'<img src="{_logo_b64()}" style="height:80px;width:auto;display:block">' if _logo_b64() else '<span style="font-family:Inter,sans-serif;font-size:15px;font-weight:700;color:{INK}">SpecCheck</span>'}
        </div>
        <div style="display:flex;align-items:center">{sep.join(parts)}</div>
        <div style="width:100px"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Sub-nav: back / home buttons — only shown when there's somewhere to go
    if step > 0:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        nav_cols = st.columns([1, 1, 8])
        with nav_cols[0]:
            if st.button("⌂  Home", key="nav_home", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                _init()
                st.rerun()
        with nav_cols[1]:
            if step > 1:
                if st.button("←  Back", key="nav_back", use_container_width=True):
                    st.session_state.step = step - 1
                    st.rerun()
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

def section_head(text):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:28px 0 14px">
      <span style="font-size:11px;font-family:'JetBrains Mono',monospace;letter-spacing:.08em;
                   color:{INK3};text-transform:uppercase;white-space:nowrap;font-weight:500">{text}</span>
      <div style="flex:1;height:1px;background:{LINE}"></div>
    </div>""", unsafe_allow_html=True)

def rank_select(options, selected_list, key_prefix, accent_color=None):
    """Multi-select that tracks selection order as priority rank (① ② ③ …)."""
    RANKS = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩"]
    sel = list(selected_list)
    changed = False
    cols_per_row = 3

    for row_start in range(0, len(options), cols_per_row):
        chunk = options[row_start:row_start + cols_per_row]
        row_cols = st.columns(len(chunk), gap="small")
        for col, opt in zip(row_cols, chunk):
            with col:
                is_sel = opt in sel
                if is_sel:
                    rank = sel.index(opt)
                    badge = RANKS[rank] if rank < len(RANKS) else f"#{rank+1}"
                    label = f"{badge}  {opt}"
                else:
                    label = opt
                if st.button(label, key=f"{key_prefix}_{opt}",
                             use_container_width=True,
                             type="primary" if is_sel else "secondary"):
                    if is_sel:
                        sel.remove(opt)
                    else:
                        sel.append(opt)
                    changed = True

    return sel, changed

def pill_select(options, selected_list, key_prefix, multi=True, accent_color=None):
    sel = list(selected_list)
    changed = False
    cols_per_row = 3

    for row_start in range(0, len(options), cols_per_row):
        chunk = options[row_start:row_start + cols_per_row]
        row_cols = st.columns(len(chunk), gap="small")

        for col, opt in zip(row_cols, chunk):
            with col:
                is_sel = opt in sel
                label = f"✓  {opt}" if is_sel else opt

                if st.button(
                    label,
                    key=f"{key_prefix}_{opt}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                ):
                    if multi:
                        if is_sel:
                            sel.remove(opt)
                        else:
                            sel.append(opt)
                    else:
                        sel = [] if is_sel else [opt]
                    changed = True

    return sel, changed

def render_step0():
    topbar()

    n_sel       = len(st.session_state.browse_selected)
    browse_open = st.session_state.get("browse_open", False)
    browse_mode = n_sel > 0

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin:0 -2rem 32px -2rem;padding:56px 2rem 52px;
                background:linear-gradient(135deg,#0f0c29 0%,#1a1560 45%,#24243e 100%);
                text-align:center;position:relative;overflow:hidden">
      <!-- decorative blobs -->
      <div style="position:absolute;top:-60px;left:-80px;width:320px;height:320px;
                  border-radius:50%;background:radial-gradient(circle,rgba(60,89,252,0.25) 0%,transparent 70%);
                  pointer-events:none"></div>
      <div style="position:absolute;bottom:-80px;right:-60px;width:280px;height:280px;
                  border-radius:50%;background:radial-gradient(circle,rgba(124,58,237,0.2) 0%,transparent 70%);
                  pointer-events:none"></div>
      <!-- content -->
      <div style="position:relative;z-index:1">
        <div style="display:inline-block;font-size:11px;font-family:'JetBrains Mono',monospace;
                    letter-spacing:.14em;color:rgba(255,255,255,0.45);text-transform:uppercase;
                    margin-bottom:18px;border:1px solid rgba(255,255,255,0.12);
                    padding:4px 12px;border-radius:20px;background:rgba(255,255,255,0.06)">
          Step 1 of 3
        </div>
        <div style="font-size:42px;font-weight:800;letter-spacing:-.04em;color:#ffffff;
                    margin-bottom:14px;line-height:1.1">
          What are you shopping for?
        </div>
        <div style="font-size:16px;color:rgba(255,255,255,0.6);max-width:460px;margin:0 auto;line-height:1.6">
          Choose a category and answer a few questions — or pick specific products to compare side by side.
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Gradient definitions per category
    CAT_GRAD = {
        "Laptops":     "linear-gradient(135deg,#4c1d95 0%,#7c3aed 100%)",
        "Smartphones": "linear-gradient(135deg,#0c4a6e 0%,#0284c7 100%)",
        "Headphones":  "linear-gradient(135deg,#78350f 0%,#d97706 100%)",
        "Monitors":    "linear-gradient(135deg,#064e3b 0%,#059669 100%)",
    }

    # ── Guided category flow — hidden when browse products are selected ───────
    if not browse_mode:
        cols = st.columns(4, gap="small")
        for col, cat in zip(cols, ["Laptops","Smartphones","Headphones","Monitors"]):
            with col:
                sel = st.session_state.category == cat
                pal = CAT_PAL[cat]
                outer_border = f"2px solid {pal['accent']}" if sel else f"1.5px solid {LINE}"
                card_bg      = WHITE
                sel_ring     = f"box-shadow:0 0 0 3px {pal['accent']}30;" if sel else ""
                st.markdown(f"""
                <div class="sc-card" style="border:{outer_border};border-radius:16px;overflow:hidden;
                            background:{card_bg};margin-bottom:8px;min-height:178px;
                            display:flex;flex-direction:column;justify-content:space-between;
                            {sel_ring}">
                  <div style="background:{CAT_GRAD[cat]};padding:24px 12px;text-align:center;
                              position:relative;overflow:hidden">
                    <div style="position:absolute;top:-20px;right:-20px;width:80px;height:80px;
                                border-radius:50%;background:rgba(255,255,255,0.08)"></div>
                    <div style="display:inline-flex;align-items:center;justify-content:center;
                                width:60px;height:60px;border-radius:50%;
                                background:rgba(255,255,255,0.15);
                                position:relative;z-index:1">
                      {'<span style="font-size:30px;line-height:1">' + CAT_ICON[cat] + '</span>'
                       if cat != "Monitors" else _MONITOR_SVG}
                    </div>
                  </div>
                  <div style="padding:12px 10px 16px 10px;text-align:center">
                    <div style="font-size:13px;font-weight:700;color:{INK}">{cat}</div>
                    <div style="font-size:10px;color:{INK3};font-family:'JetBrains Mono',monospace;
                                margin-top:3px">{_get_prod_count(cat):,} products</div>
                  </div>
                </div>""", unsafe_allow_html=True)
                if st.button("✓ Selected" if sel else "Select", key=f"c_{cat}",
                             use_container_width=True, type="primary" if sel else "secondary"):
                    st.session_state.category   = cat
                    st.session_state.sel_uses   = []
                    st.session_state.sel_prios  = []
                    st.session_state.sel_port   = ""
                    st.session_state.sel_budget = ""
                    st.rerun()

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1,2,1])
        with mid:
            if st.button(f"Continue with {st.session_state.category}  →",
                         type="primary", use_container_width=True):
                st.session_state.step = 1
                st.rerun()

    else:
        # Browse mode active — show what's selected + escape hatch
        sel_cat = st.session_state.browse_selected[0]["category"]
        pal_sel = CAT_PAL[sel_cat]
        names_preview = ", ".join(
            " ".join(p["name"].split()[:2]) for p in st.session_state.browse_selected
        )
        st.markdown(f"""
        <div style="background:{pal_sel['bg']};border:1.5px solid {pal_sel['accent']}50;
                    border-radius:12px;padding:14px 18px;margin-bottom:16px;
                    display:flex;align-items:center;justify-content:space-between">
          <div>
            <div style="font-size:10px;font-family:'JetBrains Mono',monospace;letter-spacing:.08em;
                        color:{pal_sel['text']};text-transform:uppercase;margin-bottom:3px">
              {sel_cat} · {n_sel}/3 selected
            </div>
            <div style="font-size:14px;font-weight:600;color:{INK}">{names_preview}</div>
          </div>
        </div>""", unsafe_allow_html=True)
        _, mid_sw, _ = st.columns([1,2,1])
        with mid_sw:
            if st.button("← Use guided category flow instead", use_container_width=True):
                st.session_state.browse_selected  = []
                st.session_state.browse_open      = False
                st.session_state.browse_comparing = False
                st.session_state.browse_chat      = []
                st.rerun()

    # ── Browse & compare specific products (collapsible) ─────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:{'28px' if not browse_mode else '16px'} 0 8px">
      <div style="flex:1;height:1px;background:{LINE}"></div>
      <span style="font-size:11px;font-family:'JetBrains Mono',monospace;color:{INK3}">
        or compare specific products
      </span>
      <div style="flex:1;height:1px;background:{LINE}"></div>
    </div>""", unsafe_allow_html=True)

    if n_sel > 0:
        toggle_label = f"✓ {n_sel} selected · {'Hide' if browse_open else 'Edit selection ↓'}"
    else:
        toggle_label = f"{'Hide browser ↑' if browse_open else 'Browse & pick products to compare ↓'}"

    _, mid_br, _ = st.columns([1,2,1])
    with mid_br:
        if st.button(toggle_label, use_container_width=True, key="browse_toggle"):
            st.session_state.browse_open = not browse_open
            st.rerun()

    # Show compare CTA when products selected and browser is collapsed
    if n_sel >= 2 and not browse_open:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _, cta_m, _ = st.columns([1,2,1])
        with cta_m:
            names_str = " vs ".join(p["name"].split()[0] for p in st.session_state.browse_selected)
            if st.button(f"Compare: {names_str}  →", type="primary", use_container_width=True):
                st.session_state.browse_comparing = True
                st.rerun()

    # ── Everything below only renders when open ───────────────────────────────
    if browse_open:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Search filter — key is managed by Streamlit, no value= override
        # We read from session_state AFTER the widget so the filter applies immediately
        sc1, sc2 = st.columns([5,1], gap="small")
        with sc1:
            st.text_input(
                "search_products",
                placeholder='Filter — e.g. "MacBook" or "Sony"',
                label_visibility="collapsed",
                key="browse_search"          # Streamlit stores value here directly
            )
        with sc2:
            if st.button("✕ Clear", use_container_width=True, key="clear_search",
                         disabled=not bool(st.session_state.get("browse_search", ""))):
                st.session_state.browse_search = ""
                st.rerun()

        # Read current search value straight from session_state (no rerun needed)
        sq = st.session_state.get("browse_search", "").strip().lower()
        # Reset pagination when user starts/changes a search
        if sq:
            for _cat in ["Laptops", "Smartphones", "Headphones", "Monitors"]:
                st.session_state[f"browse_page_{_cat}"] = 1

        sel_names = [p["name"] for p in st.session_state.browse_selected]
        sel_cat   = st.session_state.browse_selected[0]["category"] if st.session_state.browse_selected else None

        st.markdown(f'<div style="font-size:11px;color:{INK3};margin:8px 0 12px;text-align:center">'
                    f'Select 2–3 products from the same category (max 3)</div>', unsafe_allow_html=True)

        tab_labels = [f"{CAT_ICON[c]}  {c}" for c in ["Laptops","Smartphones","Headphones","Monitors"]]
        tabs = st.tabs(tab_labels)

        for tab, browse_cat in zip(tabs, ["Laptops","Smartphones","Headphones","Monitors"]):
            with tab:
                df = load_products(browse_cat)
                if df.empty:
                    st.markdown(f'<p style="color:{INK3};font-size:13px">No products found.</p>',
                                unsafe_allow_html=True)
                    continue
                pal = CAT_PAL[browse_cat]

                # Apply live filter from session_state directly
                if sq:
                    mask = (df["name"].str.lower().str.contains(sq, na=False, regex=False) |
                            df["brand"].str.lower().str.contains(sq, na=False, regex=False))
                    filtered = df[mask]
                else:
                    filtered = df

                prods = filtered.sort_values("base_score", ascending=False).to_dict("records")

                if not prods:
                    st.markdown(f'<p style="color:{INK3};font-size:13px;padding:8px 0">'
                                f'No matches for "{sq}"</p>', unsafe_allow_html=True)
                    continue

                # Pagination — 8 per page (2 rows × 4); disabled during search
                PAGE_SIZE = 8
                page_key  = f"browse_page_{browse_cat}"
                if page_key not in st.session_state:
                    st.session_state[page_key] = 1
                visible = prods if sq else prods[: st.session_state[page_key] * PAGE_SIZE]

                cols = st.columns(4, gap="small")
                for i, prod in enumerate(visible):
                    name     = prod["name"]
                    is_sel   = name in sel_names
                    locked   = bool(sel_cat and sel_cat != browse_cat)
                    at_max   = len(sel_names) >= 3 and not is_sel
                    disabled = locked or at_max

                    rank_nums = ["①","②","③"]
                    btn_label = (f"{rank_nums[sel_names.index(name)]}  {name}"
                                 if is_sel else name)
                    is_new    = prod.get("year", 0) >= 2024
                    new_badge = (
                        f'<span style="font-size:8px;background:#eef1ff;color:#3c59fc;'
                        f'border:1px solid #3c59fc40;padding:1px 5px;border-radius:8px;'
                        f'margin-left:4px;font-weight:600">2024</span>'
                    ) if is_new else ""

                    with cols[i % 4]:
                        img_url  = _fetch_product_image(name)
                        img_html = (
                            f'<div style="height:100px;overflow:hidden;background:{pal["bg"]};'
                            f'display:flex;align-items:center;justify-content:center">'
                            f'<img src="{img_url}" style="width:100%;height:100px;'
                            f'object-fit:contain;padding:8px"></div>'
                        ) if img_url else (
                            f'<div style="background:linear-gradient(135deg,{pal["bg"]},'
                            f'{pal["accent"]}18);height:100px;display:flex;'
                            f'align-items:center;justify-content:center;'
                            f'font-size:28px">{CAT_ICON[browse_cat]}</div>'
                        )
                        border = f"2px solid {pal['accent']}" if is_sel else f"1px solid {LINE}"
                        bg     = pal["bg"] if is_sel else WHITE
                        sel_shadow = f"box-shadow:0 0 0 3px {pal['accent']}25;" if is_sel else ""
                        st.markdown(
                            f'<div class="sc-card" style="border:{border};border-radius:10px;overflow:hidden;'
                            f'background:{bg};margin-bottom:4px;{sel_shadow}">'
                            f'{img_html}'
                            f'<div style="padding:10px 10px 4px">'
                            f'<div style="font-size:11px;font-weight:600;color:{INK};'
                            f'line-height:1.3;margin-bottom:2px">{name}{new_badge}</div>'
                            f'<div style="font-size:10px;color:{INK3};margin-bottom:4px">'
                            f'{prod["brand"]} · {prod["year"]}</div>'
                            f'<div style="font-size:12px;font-weight:600;color:{pal["accent"]};'
                            f'font-family:JetBrains Mono,monospace">${prod["price"]:,}</div>'
                            f'</div></div>',
                            unsafe_allow_html=True
                        )
                        if not disabled:
                            if st.button("✓ Remove" if is_sel else "+ Select",
                                         key=f"br_{browse_cat}_{i}",
                                         use_container_width=True,
                                         type="primary" if is_sel else "secondary"):
                                if is_sel:
                                    st.session_state.browse_selected = [
                                        p for p in st.session_state.browse_selected
                                        if p["name"] != name
                                    ]
                                else:
                                    row = prod.copy()
                                    row["category"] = browse_cat
                                    st.session_state.browse_selected.append(row)
                                st.rerun()
                        else:
                            reason = "Max 3 selected" if at_max else f"Need {sel_cat}"
                            st.button(reason, key=f"br_{browse_cat}_{i}",
                                      use_container_width=True, disabled=True)

                # "See more" button — only when not searching and more products exist
                if not sq and len(visible) < len(prods):
                    remaining = len(prods) - len(visible)
                    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                    _, btn_col, _ = st.columns([1, 2, 1])
                    with btn_col:
                        if st.button(f"See more  ({remaining} remaining)",
                                     key=f"see_more_{browse_cat}",
                                     use_container_width=True):
                            st.session_state[page_key] += 1
                            st.rerun()

        n_sel2 = len(st.session_state.browse_selected)
        if n_sel2 >= 2:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            _, cta_m2, _ = st.columns([1,2,1])
            with cta_m2:
                names_str = " vs ".join(p["name"].split()[0] for p in st.session_state.browse_selected)
                if st.button(f"Compare {n_sel2} products: {names_str}  →",
                             type="primary", use_container_width=True, key="compare_cta2"):
                    st.session_state.browse_comparing = True
                    st.rerun()
        elif n_sel2 == 1:
            st.markdown(
                f'<p style="text-align:center;font-size:12px;color:{INK3};margin-top:12px">'
                f'Select 1 or 2 more to compare (max 3)</p>', unsafe_allow_html=True
            )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


def _run_search(query):
    for cat in ["Laptops","Smartphones","Headphones","Monitors"]:
        df = load_products(cat)
        if df.empty:
            continue
        mask = (
            df["name"].str.lower().str.contains(query.lower(), na=False) |
            df["brand"].str.lower().str.contains(query.lower(), na=False)
        )
        m = df[mask].copy()
        if not m.empty:
            m["match_score"] = m["base_score"].round(0).astype(int)
            m["score_breakdown"] = m.apply(_bkdn, axis=1)
            st.session_state.category = cat
            st.session_state.sel_uses = [USE_CASES[cat][0]]
            st.session_state.sel_budget = BUDGET_RANGES[cat][-1][0]
            st.session_state.results = m.head(3).to_dict("records")
            st.session_state.prices_fetched = False
            st.session_state.verdict_fetched = False
            st.session_state.chat_history = []
            st.session_state.step = 2
            st.rerun()
    st.warning(f"No products found for '{query}'.")

def render_step1():
    topbar()
    cat = st.session_state.category
    pal = CAT_PAL[cat]
    ac = "#3c59fc"  # fixed blue throughout results

    st.markdown(f"""
    <div style="background:#eef1ff;border:1px solid #3c59fc20;
                border-radius:16px;padding:28px 32px;margin-bottom:28px">
      <div style="font-size:11px;font-family:'JetBrains Mono',monospace;letter-spacing:.1em;
                  color:#3c59fc;text-transform:uppercase;margin-bottom:10px">
        Personalise your search · {cat}
      </div>
      <div style="font-size:26px;font-weight:600;letter-spacing:-.03em;color:{INK};margin-bottom:6px">
        Tell us what you need
      </div>
      <div style="font-size:14px;color:{INK2};line-height:1.6">
        Select as many options as apply across each section —
        the more you share, the better your matches.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom:10px">
      <div style="font-size:18px;font-weight:600;color:{INK};letter-spacing:-.02em;margin-bottom:4px">
        1 · What will you use it for?
      </div>
      <div style="font-size:13px;color:{INK2}">Select all that apply</div>
    </div>""", unsafe_allow_html=True)

    nu, ch = pill_select(USE_CASES[cat], st.session_state.sel_uses, "use", multi=True, accent_color=ac)
    if ch:
        st.session_state.sel_uses = nu
        st.rerun()

    st.markdown(f"""
    <div style="margin:28px 0 10px">
      <div style="font-size:18px;font-weight:600;color:{INK};letter-spacing:-.02em;margin-bottom:4px">
        2 · What matters most to you?
      </div>
      <div style="font-size:13px;color:{INK2}">Select in order of importance — first tap = highest priority</div>
    </div>""", unsafe_allow_html=True)

    np2, ch2 = rank_select(_ordered_priorities(cat, st.session_state.sel_uses), st.session_state.sel_prios, "pri", accent_color=ac)
    if ch2:
        st.session_state.sel_prios = np2
        st.rerun()

    port_q = {
        "Laptops":"3 · How much will you carry it?",
        "Smartphones":"3 · How important is size & portability?",
        "Headphones":"3 · Where will you use them most?",
        "Monitors":"3 · Is your desk setup fixed?"
    }[cat]

    st.markdown(f"""
    <div style="margin:28px 0 10px">
      <div style="font-size:18px;font-weight:600;color:{INK};letter-spacing:-.02em;margin-bottom:4px">
        {port_q}
      </div>
      <div style="font-size:13px;color:{INK2}">Choose one</div>
    </div>""", unsafe_allow_html=True)

    cur_port = [st.session_state.sel_port] if st.session_state.sel_port else []
    np3, ch3 = pill_select(PORTABILITY[cat], cur_port, "port", multi=False, accent_color=ac)
    if ch3:
        st.session_state.sel_port = np3[0] if np3 else ""
        st.rerun()

    st.markdown(f"""
    <div style="margin:28px 0 10px">
      <div style="font-size:18px;font-weight:600;color:{INK};letter-spacing:-.02em;margin-bottom:4px">
        4 · What's your budget?
      </div>
      <div style="font-size:13px;color:{INK2}">Drag to set your maximum</div>
    </div>""", unsafe_allow_html=True)

    sl_min, sl_max, sl_step, sl_default = BUDGET_SLIDER[cat]
    # Parse stored value or use default
    try:
        cur_slider = int(st.session_state.sel_budget) if st.session_state.sel_budget else sl_default
    except (ValueError, TypeError):
        cur_slider = sl_default
    cur_slider = max(sl_min, min(sl_max, cur_slider))

    new_slider = st.slider(
        "budget_slider",
        min_value=sl_min, max_value=sl_max, value=cur_slider,
        step=sl_step, label_visibility="collapsed",
        key=f"bslider_{cat}"
    )
    # Show selected value with "No limit" label at max
    if new_slider >= sl_max:
        bud_display = f"No limit (showing all under ${sl_max:,}+)"
        budget_range_val = (0, 99999)
    else:
        bud_display = f"Up to ${new_slider:,}"
        budget_range_val = (0, new_slider)

    # "No limit" checkbox underneath the slider
    no_limit = st.checkbox("No budget limit", value=(st.session_state.get("_no_limit", False)), key=f"nolimit_{cat}")
    if no_limit:
        bud_display      = "No limit"
        budget_range_val = (0, 99999)
        st.session_state["_no_limit"]      = True
        st.session_state.sel_budget        = str(sl_max)
        st.session_state["_budget_range"]  = budget_range_val
    else:
        st.session_state["_no_limit"] = False
        if str(new_slider) != st.session_state.sel_budget or st.session_state.get("_budget_range") != budget_range_val:
            st.session_state.sel_budget       = str(new_slider)
            st.session_state["_budget_range"] = budget_range_val
    # Show selected value
    display_text = "No limit — showing all products" if no_limit else f"Up to ${new_slider:,}"
    st.markdown(
        f'<div style="text-align:center;font-size:15px;font-weight:600;color:#1c1c1a;'
        f'margin-top:2px;margin-bottom:2px">{display_text}</div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    can_uses   = bool(st.session_state.sel_uses)
    can_budget = True   # slider always has a value
    can        = can_uses
    bc, nc = st.columns([1,3], gap="small")
    with bc:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 0
            st.rerun()
    with nc:
        if st.button("Find my best matches →", type="primary",
                     use_container_width=True, disabled=not can):
            _run_results()

    if not can:
        missing = []
        if not can_uses: missing.append("a use-case above")
        hint = "Select " + " and ".join(missing) + " to continue"
        st.markdown(
            f'<div style="font-size:12px;color:{INK3};margin-top:6px;text-align:center">{hint}</div>',
            unsafe_allow_html=True
        )

def _run_results():
    cat      = st.session_state.category
    use_case = st.session_state.sel_uses[0] if st.session_state.sel_uses else USE_CASES[cat][0]
    # Budget: slider stores integer string, e.g. "1200"; session has precomputed range
    bmap = {b[0]: b[1] for b in BUDGET_RANGES[cat]}
    if st.session_state.get("_budget_range"):
        budget_range = st.session_state["_budget_range"]
    elif st.session_state.sel_budget and st.session_state.sel_budget.isdigit():
        sv = int(st.session_state.sel_budget)
        sl_max = BUDGET_SLIDER[cat][1]
        budget_range = (0, 99999) if sv >= sl_max else (0, sv)
    else:
        budget_range = bmap.get(st.session_state.sel_budget, (0, 99999))

    # If the user came via "Add preferences" from browse-compare, re-score those
    # specific products against their chosen preferences instead of the full dataset
    prefill = st.session_state.get("browse_prefill")
    if prefill:
        # Filter full dataset to just the hand-picked products, then re-score
        prefill_names = {p["name"] for p in prefill}
        df_full = load_products(cat)
        if not df_full.empty:
            df_pf = df_full[df_full["name"].isin(prefill_names)].copy()
            if df_pf.empty:
                df_pf = df_full   # fallback: score everything
        else:
            import pandas as _pd
            df_pf = _pd.DataFrame(prefill)
        scored = compute_scores(df_pf, cat, use_case, (0, 99999),
                                st.session_state.sel_port, st.session_state.sel_prios)
        if scored.empty:
            scored = df_pf.copy()
            scored["match_score"] = scored["base_score"].round(0).astype(int)
            import numpy as np
            scored["score_breakdown"] = scored.apply(
                lambda r: {"Performance": int(r.get("cpu_norm",50)*.6+r.get("gpu_norm",50)*.4),
                           "Battery": int(r.get("battery_norm",50)),
                           "Portability": int(r.get("weight_norm",50)),
                           "Display": int(r.get("display_norm",50)),
                           "Value": int(r.get("price_norm",50))}, axis=1)
        st.session_state.browse_prefill  = None
        st.session_state.results         = scored.head(3).to_dict("records")
        st.session_state.budget_warning  = None
        st.session_state.prices_fetched  = False
        st.session_state.verdict_fetched = False
        st.session_state.chat_history    = []
        st.session_state.step = 2
        st.rerun()
        return

    df = load_products(cat)
    if df.empty:
        st.error("Dataset not found. Run: python pipeline/build_dataset.py")
        return
    scored = compute_scores(df, cat, use_case, budget_range, st.session_state.sel_port, st.session_state.sel_prios)
    budget_warning = None
    if scored.empty:
        all_scored = compute_scores(df, cat, use_case, (0, 99999), st.session_state.sel_port, st.session_state.sel_prios)
        if all_scored.empty:
            st.warning("No products found.")
            return
        cheapest = all_scored.sort_values("price").head(3)
        scored = cheapest
        lo, hi = budget_range
        budget_label = st.session_state.sel_budget
        min_price = int(scored["price"].min())
        budget_warning = (
            f"No {cat.lower()} found under your budget ({budget_label}). "
            f"Showing the closest options — starting from ${min_price:,}."
        )
    st.session_state.results = scored.head(3).to_dict("records")
    st.session_state.budget_warning = budget_warning
    st.session_state.prices_fetched = False
    st.session_state.verdict_fetched = False
    st.session_state.chat_history = []
    st.session_state.step = 2
    st.rerun()

def render_step2():
    # Scroll to top of page immediately when results load
    st.markdown(
        '<script>window.scrollTo(0,0);</script>',
        unsafe_allow_html=True
    )
    topbar()
    products = st.session_state.results or []
    if not products:
        st.error("No results. Go back.")
        return

    cat = st.session_state.category
    pal = CAT_PAL[cat]
    uses = st.session_state.sel_uses
    # Budget display for results header
    _sb = st.session_state.sel_budget
    if _sb and _sb.isdigit():
        sl_max = BUDGET_SLIDER[cat][1]
        budget = "No limit" if int(_sb) >= sl_max else f"Up to ${int(_sb):,}"
    else:
        budget = _sb or "Any budget"
    use_case = uses[0] if uses else ""
    winner = products[0]

    if not st.session_state.prices_fetched or not st.session_state.verdict_fetched:
        from concurrent.futures import ThreadPoolExecutor
        with st.spinner("Finding your best matches…"):
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_prices  = ex.submit(get_prices_batch, products, cat)
                fut_verdict = ex.submit(generate_verdict, products, cat, use_case,
                                        budget, st.session_state.sel_port,
                                        st.session_state.sel_prios)
                st.session_state.prices         = fut_prices.result()
                st.session_state.verdict        = fut_verdict.result()
                st.session_state.prices_fetched  = True
                st.session_state.verdict_fetched = True

    pref_str = " · ".join(filter(None, [", ".join(uses[:2]), budget]))

    h1, h2 = st.columns([5, 1], gap="small")
    with h1:
        st.markdown(f"""
        <div style="margin-bottom:16px">
          <div style="font-size:28px;font-weight:800;color:{INK};letter-spacing:-.03em;
                      line-height:1.15;margin-bottom:6px">
            Your top matches
          </div>
          <div style="font-size:12px;color:{INK3};font-family:'JetBrains Mono',monospace">{pref_str}</div>
        </div>""", unsafe_allow_html=True)
    with h2:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("← Edit", use_container_width=True, key="edit_prefs"):
            st.session_state.step = 1
            st.rerun()

    st.markdown(f"""
    <div style="background:{WHITE};border:1px solid {LINE};border-radius:8px;
                padding:20px 24px;margin-bottom:24px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);
                display:flex;gap:16px;align-items:flex-start">
      <div style="width:36px;height:36px;border-radius:6px;background:#3c59fc;
                  display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M8 2l1.5 4H14l-3.5 2.5 1.5 4L8 10.5 4 12.5l1.5-4L2 6h4.5z" fill="white"/>
        </svg>
      </div>
      <div style="flex:1">
        <div style="font-size:11px;font-family:'JetBrains Mono',monospace;letter-spacing:.08em;
                    color:#3c59fc;text-transform:uppercase;font-weight:600;margin-bottom:8px">
          AI recommendation
        </div>
        <div style="font-size:14px;color:{INK};line-height:1.75">{st.session_state.verdict}</div>
        <div style="display:inline-flex;align-items:center;gap:6px;margin-top:10px;
                    padding:4px 10px;background:#eef1ff;border-radius:4px">
          <div style="width:8px;height:8px;border-radius:50%;background:#3c59fc"></div>
          <span style="font-size:12px;font-weight:600;color:#3c59fc">
            Top pick: {winner['name']} — {winner['match_score']}/100
          </span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.get("budget_warning"):
        st.markdown(f"""
        <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:6px;
                    padding:12px 16px;margin-bottom:16px;display:flex;align-items:center;gap:10px">
          <span style="font-size:16px">⚠️</span>
          <span style="font-size:13px;color:#92400e">{st.session_state.budget_warning}</span>
        </div>""", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["📊  Comparison","🎯  Score radar","🛒  Where to buy","💬  Spec advisor"])
    with t1:
        _tab_compare(products, cat, pal)
    with t2:
        _tab_radar(products, cat)
    with t3:
        _tab_prices(products, cat, pal)
    with t4:
        _tab_chat(products, cat, use_case, winner, pal)

    st.markdown(f"""
    <div style="background:{WHITE};border:1px solid {LINE};border-radius:14px;
                padding:22px 24px;margin-top:28px;text-align:center">
      <div style="font-size:15px;font-weight:600;color:{INK};margin-bottom:4px">Made your decision?</div>
      <div style="font-size:13px;color:{INK2};margin-bottom:18px">
        Jump to the best price, save this comparison, or start fresh.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3, gap="small")
    with e1:
        if st.button("Start new comparison", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init()
            st.rerun()
    with e2:
        if st.button("← Adjust preferences", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with e3:
        retailers = st.session_state.prices.get(winner["name"], [])
        best_link = retailers[0]["link"] if retailers else "#"
        nm = " ".join(winner["name"].split()[:2])
        st.markdown(f"""
        <a href="{best_link}" target="_blank"
           style="display:block;text-align:center;padding:10px 16px;background:{INK};
                  color:white;border-radius:8px;font-size:13px;font-weight:600;
                  text-decoration:none;font-family:'Inter',sans-serif">
          Buy {nm} →
        </a>""", unsafe_allow_html=True)

def _tab_compare(products, cat, pal):
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Build interleaved columns: product | vs | product | vs | product
    n = len(products)
    if n == 3:
        col_widths = [12, 1, 12, 1, 12]
    elif n == 2:
        col_widths = [12, 1, 12]
    else:
        col_widths = [1]
    all_cols = st.columns(col_widths, gap="small")
    prod_cols = all_cols[::2]   # even indices = product columns
    vs_cols   = all_cols[1::2]  # odd indices  = "vs" divider columns

    # "vs" dividers
    for vc in vs_cols:
        with vc:
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:center;height:100%;
                        padding-top:60px">
              <div style="width:32px;height:32px;border-radius:50%;background:{LINE};
                          display:flex;align-items:center;justify-content:center">
                <span style="font-size:10px;font-weight:700;color:{INK2};
                             font-family:'JetBrains Mono',monospace">vs</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # Product cards
    rank_colors = ["#3c59fc", INK2, INK3]  # fixed blue for score circles
    for i, (col, prod) in enumerate(zip(prod_cols, products)):
        score_color = rank_colors[i] if i < len(rank_colors) else INK3
        is_winner   = (i == 0)
        card_shadow = "0 4px 16px rgba(0,0,0,0.10)" if is_winner else "0 2px 8px rgba(0,0,0,0.06)"
        card_border = f"2px solid #3c59fc" if is_winner else f"1px solid {LINE}"

        with col:
            badge = (
                f'<div style="display:inline-block;font-size:10px;font-weight:600;'
                f'background:#3c59fc;color:white;padding:3px 10px;'
                f'border-radius:2px;margin-bottom:12px;letter-spacing:.04em">BEST MATCH</div>'
                if is_winner else '<div style="height:22px"></div>'
            )
            bar_w = prod.get("match_score", 0)
            # Product image — reuse cached image URLs from browse view
            img_url = _fetch_product_image(prod["name"])
            if img_url:
                img_html = (
                    f'<div style="height:100px;background:{pal["bg"]};border-radius:6px;'
                    f'display:flex;align-items:center;justify-content:center;margin-bottom:14px;overflow:hidden">'
                    f'<img src="{img_url}" style="max-height:90px;max-width:100%;object-fit:contain;padding:6px"></div>'
                )
            else:
                img_html = (
                    f'<div style="height:100px;background:linear-gradient(135deg,{pal["bg"]},'
                    f'{pal["accent"]}18);border-radius:6px;display:flex;align-items:center;'
                    f'justify-content:center;margin-bottom:14px;font-size:32px">{CAT_ICON[cat]}</div>'
                )
            st.markdown(f"""
            <div style="background:{WHITE};border:{card_border};border-radius:8px;
                        padding:20px;box-shadow:{card_shadow};margin-bottom:8px">
              {badge}
              {img_html}
              <div style="font-size:14px;font-weight:700;color:{INK};margin-bottom:2px;
                          line-height:1.3">{prod['name']}</div>
              <div style="font-size:12px;color:{INK3};margin-bottom:16px">
                {prod['brand']} · {prod['year']}
              </div>
              <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px">
                <div style="width:56px;height:56px;border-radius:50%;background:{score_color};
                            display:flex;align-items:center;justify-content:center;flex-shrink:0">
                  <span style="color:white;font-size:18px;font-weight:800;
                               letter-spacing:-.02em">{prod['match_score']}</span>
                </div>
                <div>
                  <div style="font-size:11px;color:{INK3};margin-bottom:2px">match score</div>
                  <div style="height:5px;width:80px;background:{LINE};border-radius:3px;overflow:hidden">
                    <div style="height:100%;width:{bar_w}%;background:{score_color};border-radius:3px"></div>
                  </div>
                </div>
              </div>
              <div style="font-size:13px;color:{INK2}">
                ⭐ {prod['avg_rating']} &nbsp;·&nbsp; ${prod['price']:,.0f}
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Why this score? explainer ─────────────────────────────────────────────
    section_head("Why these scores?")
    why_cols = st.columns(len(products), gap="small")
    for wcol, prod in zip(why_cols, products):
        with wcol:
            bd = prod.get("score_breakdown", {})
            sorted_bd = sorted(bd.items(), key=lambda x: x[1], reverse=True)
            strengths = sorted_bd[:2]
            weakness  = next(((a,v) for a,v in reversed(sorted_bd) if v < 62), None)
            nm = " ".join(prod["name"].split()[:2])
            is_new = prod.get("year", 0) >= 2024
            new_pill = (
                f'<span style="font-size:9px;background:#eef1ff;color:#3c59fc;'
                f'border:1px solid #3c59fc40;padding:1px 7px;border-radius:10px;'
                f'margin-left:5px;vertical-align:middle;font-weight:600">2024</span>'
            ) if is_new else ""
            str_parts = " & ".join(
                f'<span style="color:#3c59fc;font-weight:600">{a.lower()}</span> ({v}/100)'
                for a, v in strengths
            )
            weak_text = (
                f' Weaker on <span style="color:{BAD}">{weakness[0].lower()}</span>'
                f' ({weakness[1]}/100).'
            ) if weakness else ""
            st.markdown(f"""
            <div style="background:{WHITE};border:1px solid {LINE};border-radius:10px;
                        padding:12px 14px;font-size:12px;color:{INK2};line-height:1.7">
              <div style="font-size:11px;font-weight:600;color:{INK};margin-bottom:5px">
                {nm}{new_pill}
              </div>
              Stands out on {str_parts} for your profile.{weak_text}
            </div>""", unsafe_allow_html=True)

    _spec_table(products, cat, pal)

    section_head(f"User sentiment · {sum(p['review_count'] for p in products):,} reviews analysed")
    sc = st.columns(len(products), gap="small")
    for col, prod in zip(sc, products):
        with col:
            _sent_card(prod, pal)

def _spec_table(products, cat, pal):
    section_head("Spec breakdown")
    specs = {
        "Laptops":[("Price","price",lambda v:f"${v:,.0f}",True,True),
                   ("CPU","cpu_score",lambda v:f"{v}/100",False,False),
                   ("RAM","ram_gb",lambda v:f"{v} GB",False,False),
                   ("Battery","battery_h",lambda v:f"{v} h",False,False),
                   ("Weight","weight_kg",lambda v:f"{v} kg",True,True),
                   ("Display","display_score",lambda v:f"{v}/100",False,False),
                   ("GPU","gpu_score",lambda v:f"{v}/100",False,False)],
        "Smartphones":[("Price","price",lambda v:f"${v:,.0f}",True,True),
                       ("CPU","cpu_score",lambda v:f"{v}/100",False,False),
                       ("RAM","ram_gb",lambda v:f"{v} GB",False,False),
                       ("Battery","battery_h",lambda v:f"{v} h",False,False),
                       ("Weight","weight_kg",lambda v:f"{v*1000:.0f} g",True,True),
                       ("Display","display_score",lambda v:f"{v}/100",False,False),
                       ("Camera","gpu_score",lambda v:f"{v}/100",False,False)],
        "Headphones":[("Price","price",lambda v:f"${v:,.0f}",True,True),
                      ("ANC","cpu_score",lambda v:f"{v}/100",False,False),
                      ("Battery","battery_h",lambda v:f"{v} h",False,False),
                      ("Weight","weight_kg",lambda v:f"{v*1000:.0f} g",True,True),
                      ("Sound","display_score",lambda v:f"{v}/100",False,False)],
        "Monitors":[("Price","price",lambda v:f"${v:,.0f}",True,True),
                    ("Refresh","cpu_score",lambda v:f"{v}/100",False,False),
                    ("Color acc.","display_score",lambda v:f"{v}/100",False,False),
                    ("Weight","weight_kg",lambda v:f"{v} kg",True,True)],
    }[cat]

    th = (
        f"text-align:left;padding:10px 12px;font-size:10px;font-family:'JetBrains Mono',monospace;"
        f"letter-spacing:.06em;color:{INK3};font-weight:500;background:{SURF};"
        f"border-bottom:1px solid {LINE}"
    )
    hds = f'<th style="{th}">Spec</th>'
    for p in products:
        nm = " ".join(p["name"].split()[:2])
        hds += f'<th style="{th}">{nm}</th>'

    rows = ""
    for lbl, key, fmt, inv, _ in specs:
        vals = [p.get(key) for p in products if p.get(key) is not None]
        bv = (min(vals) if inv else max(vals)) if vals else None
        wv = (max(vals) if inv else min(vals)) if vals else None
        row = (
            f'<td style="padding:10px 12px;border-bottom:1px solid {LINE};'
            f'font-size:11px;color:{INK3};font-family:\'JetBrains Mono\',monospace">{lbl}</td>'
        )
        for p in products:
            v = p.get(key)
            if v is None:
                row += f'<td style="padding:10px 12px;border-bottom:1px solid {LINE};font-size:11px;color:{INK3}">—</td>'
                continue
            # Treat 0 as missing data for scored specs (not price)
            is_no_data = (v == 0 and key != "price" and key != "ram_gb")
            if is_no_data:
                row += (
                    f'<td style="padding:10px 12px;border-bottom:1px solid {LINE};'
                    f'font-size:11px;color:{INK3};">'
                    f'<span style="font-size:10px;color:{INK3}">—</span>'
                    f'<span style="font-size:9px;color:{INK3};margin-left:4px">no data</span>'
                    f'</td>'
                )
                continue
            ib = v == bv
            iw = v == wv and len(products) > 2
            vc = "#3c59fc" if ib else (INK3 if iw else INK)
            fw = "600" if ib else "400"
            # For inv specs (lower=better: price, weight): worst value = "priciest"/"heaviest"
            # For non-inv specs (higher=better): worst value shown only when 3 products
            # Specific labels only for Price & Weight (unambiguous direction)
            # Everything else uses plain 'best'/'lowest' to avoid '0/100: best ANC' absurdity
            BEST_LABEL  = {"Price": "cheapest", "Weight": "lightest"}
            WORST_LABEL = {"Price": "priciest", "Weight": "heaviest"}
            if ib:
                badge = (
                    f'<span style="font-size:9px;font-family:\'JetBrains Mono\',monospace;'
                    f'background:#eef1ff;color:#3c59fc;padding:1px 6px;'
                    f'border-radius:3px;margin-left:5px">{BEST_LABEL.get(lbl, "best")}</span>'
                )
            elif iw:
                badge = (
                    f'<span style="font-size:9px;font-family:\'JetBrains Mono\',monospace;'
                    f'background:{BAD_BG};color:{BAD};padding:1px 5px;'
                    f'border-radius:3px;margin-left:5px">{WORST_LABEL.get(lbl, "lowest")}</span>'
                )
            else:
                badge = ""
            row += (
                f'<td style="padding:10px 12px;border-bottom:1px solid {LINE};'
                f'font-size:12px;font-weight:{fw};color:{vc}">{fmt(v)}{badge}</td>'
            )
        rows += f"<tr style='transition:background .1s'>{row}</tr>"

    st.markdown(f"""
    <div style="border:1px solid {LINE};border-radius:12px;overflow:hidden;background:{WHITE}">
      <table style="width:100%;border-collapse:collapse;table-layout:fixed">
        <thead><tr>{hds}</tr></thead><tbody>{rows}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

def _sent_card(prod, pal):
    pct = prod["pos_pct"]
    fill = "#3c59fc" if pct >= 80 else ("#c8920a" if pct >= 65 else BAD)
    pc   = "#3c59fc" if pct >= 80 else (WARN if pct >= 65 else BAD)
    bg   = "#eef1ff" if pct >= 80 else (WARN_BG if pct >= 65 else BAD_BG)
    pos = [t.strip() for t in prod["pos_topics"].split(",")][:3]
    neg = [t.strip() for t in prod["neg_topics"].split(",")][:2]
    ptag = "".join(
        f'<span style="font-size:9px;padding:2px 6px;border-radius:3px;'
        f'font-family:\'JetBrains Mono\',monospace;background:#eef1ff;'
        f'color:#3c59fc;margin:1px">{t}</span>' for t in pos
    )
    ntag = "".join(
        f'<span style="font-size:9px;padding:2px 6px;border-radius:3px;'
        f'font-family:\'JetBrains Mono\',monospace;background:{BAD_BG};'
        f'color:{BAD};margin:1px">{t}</span>' for t in neg
    )
    nm = " ".join(prod["name"].split()[:2])
    st.markdown(f"""
    <div style="border:1px solid {LINE};border-radius:12px;overflow:hidden;background:{WHITE}">
      <div style="background:{bg};padding:12px 14px;display:flex;align-items:baseline;
                  justify-content:space-between">
        <span style="font-size:22px;font-weight:600;color:{pc}">{pct}%</span>
        <span style="font-size:10px;color:{pc};opacity:.75">positive</span>
      </div>
      <div style="padding:10px 14px">
        <div style="font-size:12px;font-weight:600;color:{INK};margin-bottom:1px">{nm}</div>
        <div style="font-size:10px;color:{INK3};font-family:'JetBrains Mono',monospace;
                    margin-bottom:8px">{prod['review_count']:,} reviews</div>
        <div style="height:3px;background:{LINE};border-radius:2px;overflow:hidden;margin-bottom:8px">
          <div style="height:100%;width:{pct}%;background:{fill};border-radius:2px"></div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:3px">{ptag}{ntag}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def _tab_radar(products, cat):
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    pal = CAT_PAL[cat]
    axes = ["Performance","Battery","Portability","Display","Value"]
    colors = ["#3c59fc", "#6b6b63", "#b0afa8"]
    fig = go.Figure()
    for i, p in enumerate(products[:3]):
        bd = p.get("score_breakdown", {a:50 for a in axes})
        vals = [bd.get(a,50) for a in axes] + [bd.get(axes[0],50)]
        nm = " ".join(p["name"].split()[:2])
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=axes+[axes[0]],
            fill="toself",
            name=nm,
            line=dict(color=colors[i], width=2),
            fillcolor=colors[i],
            opacity=0.18 if i == 0 else 0.10
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(size=9, color=INK3), gridcolor=LINE),
            angularaxis=dict(tickfont=dict(size=11, color=INK2, family="DM Sans"), gridcolor=LINE),
            bgcolor=WHITE
        ),
        showlegend=True,
        legend=dict(font=dict(size=11, family="DM Sans", color=INK2), orientation="h", y=-0.22),
        margin=dict(t=20, b=70, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

def _tab_prices(products, cat, pal):
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    prices_dict = st.session_state.prices

    saved = sum(
        (max((r["price"] for r in prices_dict.get(p["name"],[]) if r["price"] > 0), default=0) -
         min((r["price"] for r in prices_dict.get(p["name"],[]) if r["price"] > 0), default=0))
        for p in products
    )
    if saved > 5:
        st.markdown(f"""
        <div style="background:{WARN_BG};border:1px solid #e8c97e;border-radius:12px;
                    padding:12px 18px;margin-bottom:18px;display:flex;align-items:center;gap:12px">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M10 2C5.58 2 2 5.58 2 10s3.58 8 8 8 8-3.58 8-8-3.58-8-8-8zm.75 12h-1.5v-5h1.5v5zm0-6.5h-1.5v-1.5h1.5v1.5z" fill="{WARN}"/>
          </svg>
          <span style="font-size:13px;font-weight:600;color:{WARN}">
            Compare retailers — save up to ${saved:.0f} across this shortlist
          </span>
        </div>""", unsafe_allow_html=True)

    for i, prod in enumerate(products):
        name = prod["name"]
        is_win = i == 0
        expected = prod.get("price", 0)
        retailers = [
            r for r in prices_dict.get(name, [])
            if r.get("price", 0) > 0
            and (expected == 0 or r["price"] >= expected * 0.4)
        ]
        border = f"2px solid #3c59fc" if is_win else f"1px solid {LINE}"
        badge = (
            f'<span style="font-size:9px;font-family:\'JetBrains Mono\',monospace;'
            f'background:#3c59fc;color:white;padding:2px 8px;border-radius:4px;'
            f'margin-left:8px">TOP PICK</span>' if is_win else ""
        )

        rows_html = ""
        for r in retailers:
            is_low = r["is_lowest"]
            rc = "#3c59fc" if is_low else INK
            rb_bg = "#eef1ff" if is_low else SURF
            lowest_tag = (
                f'<span style="font-size:9px;font-family:JetBrains Mono,monospace;'
                f'background:#eef1ff;color:#3c59fc;'
                f'padding:2px 6px;border-radius:3px">Best price</span>' if is_low else ''
            )
            rows_html += f"""<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid {LINE};">
        <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:8px;height:8px;border-radius:50%;background:{r.get("logo_color","#999")};"></div>
        <span style="font-size:13px;color:{INK};font-weight:500;">{r["retailer"]}</span>
        {"<span style='font-size:9px;background:#eef1ff;color:#3c59fc;padding:2px 6px;border-radius:3px;font-family:JetBrains Mono,monospace;'>Best price</span>" if is_low else ""}
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-size:14px;font-weight:600;font-family:JetBrains Mono,monospace;color:#3c59fc;">${r["price"]:,.0f}</span>
        <a href="{r.get("link","#")}" target="_blank" style="padding:6px 12px;background:#1c1c1a;color:white;border-radius:6px;font-size:11px;font-weight:600;text-decoration:none;white-space:nowrap;">Buy →</a>
        </div>
        </div>"""
        
        if not rows_html:
            rows_html = f'<div style="padding:14px;font-size:12px;color:{INK3}">No live price data available.</div>'

        # Price spread note
        valid_prices = [r["price"] for r in retailers if r.get("price", 0) > 0]
        if len(valid_prices) >= 2:
            spread_pct = (max(valid_prices) - min(valid_prices)) / min(valid_prices) * 100
            if spread_pct >= 10:
                spread_note = (
                    f'<span style="font-size:10px;color:#3c59fc;margin-left:10px;font-weight:500">'
                    f'💡 Up to {spread_pct:.0f}% price difference — check all retailers</span>'
                )
            else:
                spread_note = f'<span style="font-size:10px;color:{INK3};margin-left:10px">Consistent pricing</span>'
        else:
            spread_note = ""

        st.markdown(f"""
        <div style="background:{WHITE};border:{border};border-radius:14px;
                    overflow:hidden;margin-bottom:14px">
          <div style="padding:14px 16px;border-bottom:1px solid {LINE};
                      display:flex;align-items:center;flex-wrap:wrap">
            <span style="font-size:14px;font-weight:600;color:{INK}">{name}</span>
            {badge}{spread_note}
          </div>
          {rows_html}
        </div>""", unsafe_allow_html=True)

def _tab_chat(products, cat, use_case, winner, pal):
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


    for msg in st.session_state.chat_history:
        is_user = msg["role"] == "user"
        bg = SEL if is_user else WHITE
        align = "flex-end" if is_user else "flex-start"
        st.markdown(f"""
        <div style="display:flex;justify-content:{align};margin-bottom:10px">
          <div style="max-width:82%;background:{bg};border:1px solid {LINE};
                      border-radius:12px;padding:11px 15px;font-size:13px;
                      color:{INK};line-height:1.65">{msg['content']}</div>
        </div>""", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        section_head("Suggested questions")
        nm0 = winner['name'].split()
        sugs = [
            "Which holds up best after 2 years?",
            f"Is the {nm0[0]} {nm0[1] if len(nm0)>1 else ''} worth the price premium?",
            "Which has the best resale value?",
            "Which is the safest long-term buy?"
        ]
        sc1, sc2 = st.columns(2, gap="small")
        for i, s in enumerate(sugs):
            with (sc1 if i % 2 == 0 else sc2):
                if st.button(s, key=f"sug_{i}", use_container_width=True):
                    _do_chat(s, products, cat, use_case)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    ci1, ci2 = st.columns([5, 1], gap="small")
    with ci1:
        chat_val = st.text_input("chat", label_visibility="collapsed",
                                  placeholder=f"Ask about these {cat.lower()}...",
                                  key="main_chat_input")
    with ci2:
        if st.button("Send →", key="main_chat_send", type="primary",
                     use_container_width=True):
            if chat_val.strip():
                _do_chat(chat_val.strip(), products, cat, use_case)

def _do_chat(prompt, products, cat, use_case):
    st.session_state.chat_history.append({"role":"user","content":prompt})
    chunks = []
    with st.spinner(""):
        for chunk in stream_chat(prompt, products, cat, use_case, st.session_state.chat_history[:-1]):
            chunks.append(chunk)
    st.session_state.chat_history.append({"role":"assistant","content":"".join(chunks)})
    st.rerun()

def render_product_page():
    prod = st.session_state.featured_product
    cat  = prod["category"]
    pal  = CAT_PAL[cat]

    # ── Nav bar ──────────────────────────────────────────────────────────────
    logo_html = (
        f'<img src="{_logo_b64()}" style="height:80px;width:auto;display:block">'
        if _logo_b64() else
        f'<span style="font-size:15px;font-weight:700;color:{INK}">SpecCheck</span>'
    )
    st.markdown(f"""
    <div style="background:{WHITE};border-bottom:1px solid {LINE};
                padding:0;margin:0 -2rem 0 -2rem">
      <div style="padding:0 2rem;height:80px;display:flex;align-items:center;
                  justify-content:space-between">
        <div style="display:flex;align-items:center">{logo_html}</div>
        <div style="font-size:15px;color:{INK2}">Where to buy</div>
        <div style="width:100px"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("← Back to home", key="fp_back"):
        st.session_state.featured_product = None
        st.rerun()
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Product header ───────────────────────────────────────────────────────
    img_url = _fetch_product_image(prod["name"])
    if img_url:
        thumb_html = (
            f'<div style="width:90px;height:90px;border-radius:12px;overflow:hidden;'
            f'background:#eef1ff;border:1px solid #3c59fc30;flex-shrink:0;'
            f'display:flex;align-items:center;justify-content:center">'
            f'<img src="{img_url}" style="width:90px;height:90px;object-fit:contain;padding:6px">'
            f'</div>'
        )
    else:
        thumb_html = (
            f'<div style="width:90px;height:90px;border-radius:12px;'
            f'background:linear-gradient(135deg,#eef1ff,#3c59fc40);'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:40px;border:1px solid #3c59fc30;flex-shrink:0">'
            f'{CAT_ICON[cat]}</div>'
        )
    st.markdown(f"""
    <div style="background:#eef1ff;border:1px solid #3c59fc20;
                border-radius:16px;padding:24px 28px;margin-bottom:28px;
                display:flex;align-items:center;gap:20px">
      {thumb_html}
      <div>
        <div style="font-size:11px;font-family:'JetBrains Mono',monospace;
                    letter-spacing:.1em;color:#1e3a8a;text-transform:uppercase;
                    margin-bottom:4px">{cat}</div>
        <div style="font-size:24px;font-weight:700;color:{INK};
                    letter-spacing:-.03em;margin-bottom:2px">{prod['name']}</div>
        <div style="font-size:14px;color:{INK2}">{prod['brand']} · {prod['year']}</div>
      </div>
      <div style="margin-left:auto;text-align:right">
        <div style="font-size:11px;color:{INK3};margin-bottom:2px">from</div>
        <div style="font-size:28px;font-weight:700;color:#3c59fc;
                    font-family:'JetBrains Mono',monospace">${prod['price']:,}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Fetch prices ─────────────────────────────────────────────────────────
    # Flush stale cache if any stored link is a Google redirect
    cached = st.session_state.fp_prices.get(prod["name"], [])
    if any("google.com" in r.get("link", "") for r in cached):
        st.session_state.fp_prices = {}

    if not st.session_state.fp_prices:
        with st.spinner("Fetching live prices…"):
            st.session_state.fp_prices = {
                prod["name"]: get_prices_batch([prod], cat).get(prod["name"], [])
            }

    retailers = st.session_state.fp_prices.get(prod["name"], [])
    expected = prod.get("price", 0)
    retailers = [
        r for r in retailers
        if r.get("price", 0) > 0
        and (expected == 0 or r["price"] >= expected * 0.4)
    ]

    # ── Price table ──────────────────────────────────────────────────────────
    section_head("Retailers & live prices")

    if not retailers:
        st.markdown(f"""
        <div style="background:{SURF};border:1px solid {LINE};border-radius:10px;
                    padding:20px;text-align:center;color:{INK3};font-size:13px">
          No live price data available for this product right now.
        </div>""", unsafe_allow_html=True)
    else:
        rows = []
        for r in retailers:
            is_low = r.get("is_lowest", False)
            lowest_tag = (
                f'<span style="font-size:9px;font-family:JetBrains Mono,monospace;background:#eef1ff;color:#3c59fc;padding:2px 8px;border-radius:3px;margin-left:8px">Best price</span>'
            ) if is_low else ""
            price_color = "#3c59fc" if is_low else INK
            link = r.get("link", "#")
            rows.append(
                f'<div style="display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid {LINE}">'
                f'<div style="display:flex;align-items:center;gap:10px">'
                f'<div style="width:8px;height:8px;border-radius:50%;background:{r.get("logo_color","#999")}"></div>'
                f'<span style="font-size:14px;color:{INK};font-weight:500">{r["retailer"]}</span>'
                f'{lowest_tag}</div>'
                f'<div style="display:flex;align-items:center;gap:14px">'
                f'<span style="font-size:16px;font-weight:700;font-family:JetBrains Mono,monospace;color:{price_color}">${r["price"]:,.0f}</span>'
                f'<a href="{link}" target="_blank" style="padding:8px 16px;background:{INK};color:white;border-radius:6px;font-size:12px;font-weight:600;text-decoration:none;white-space:nowrap">Buy →</a>'
                f'</div></div>'
            )
        rows_html = "".join(rows)
        st.markdown(
            f'<div style="background:{WHITE};border:1px solid {LINE};border-radius:14px;overflow:hidden;margin-bottom:20px">{rows_html}</div>',
            unsafe_allow_html=True,
        )

    # ── CTA ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{WHITE};border:1px solid {LINE};border-radius:12px;
                padding:20px 24px;text-align:center;margin-top:8px">
      <div style="font-size:14px;color:{INK2};margin-bottom:14px">
        Want to compare <strong>{prod['name']}</strong> against other {cat.lower()}?
      </div>
    </div>""", unsafe_allow_html=True)

    _, cta_col, _ = st.columns([1,2,1])
    with cta_col:
        if st.button(f"Compare {cat} →", type="primary", use_container_width=True):
            st.session_state.featured_product = None
            st.session_state.category = cat
            st.session_state.step = 1
            st.rerun()


def _tab_chat_browse(products, cat, pal):
    """Streaming chat tab for the direct browse-compare flow (no use-case context)."""
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    for msg in st.session_state.browse_chat:
        is_user = msg["role"] == "user"
        bg = "#e8e7e2" if is_user else WHITE
        align = "flex-end" if is_user else "flex-start"
        st.markdown(f"""
        <div style="display:flex;justify-content:{align};margin-bottom:10px">
          <div style="max-width:82%;background:{bg};border:1px solid {LINE};
                      border-radius:12px;padding:11px 15px;font-size:13px;
                      color:{INK};line-height:1.65">{msg['content']}</div>
        </div>""", unsafe_allow_html=True)

    if not st.session_state.browse_chat:
        section_head("Suggested questions")
        nm0 = products[0]["name"].split()
        nm1 = products[1]["name"].split() if len(products) > 1 else nm0
        sugs = [
            f"How do these compare for everyday use?",
            f"Is the {nm0[0]} {nm0[1] if len(nm0)>1 else ''} worth the price premium?",
            "Which has the best long-term reliability?",
            f"Which is the better value for money?",
        ]
        sc1, sc2 = st.columns(2, gap="small")
        for i, s in enumerate(sugs):
            with (sc1 if i % 2 == 0 else sc2):
                if st.button(s, key=f"bsug_{i}", use_container_width=True):
                    _do_browse_chat(s, products, cat)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    bi1, bi2 = st.columns([5, 1], gap="small")
    with bi1:
        browse_val = st.text_input("chat_b", label_visibility="collapsed",
                                    placeholder=f"Ask about these {cat.lower()}...",
                                    key="browse_chat_input")
    with bi2:
        if st.button("Send →", key="browse_chat_send", type="primary",
                     use_container_width=True):
            if browse_val.strip():
                _do_browse_chat(browse_val.strip(), products, cat)


def _do_browse_chat(prompt, products, cat):
    st.session_state.browse_chat.append({"role": "user", "content": prompt})
    chunks = []
    with st.spinner(""):
        for chunk in stream_chat(prompt, products, cat, "general comparison",
                                 st.session_state.browse_chat[:-1]):
            chunks.append(chunk)
    st.session_state.browse_chat.append({"role": "assistant", "content": "".join(chunks)})
    st.rerun()


def render_browse_compare():
    st.markdown('<script>window.scrollTo(0,0);</script>', unsafe_allow_html=True)
    products = st.session_state.browse_selected
    cat      = products[0]["category"]
    pal      = CAT_PAL[cat]

    # Ensure required fields
    for p in products:
        if "match_score" not in p:
            p["match_score"] = int(p.get("base_score", 50))
        if "score_breakdown" not in p:
            p["score_breakdown"] = _bkdn(p)

    # Nav bar
    logo_html = (
        f'<img src="{_logo_b64()}" style="height:80px;width:auto;display:block">'
        if _logo_b64() else
        f'<span style="font-size:15px;font-weight:700;color:{INK}">SpecCheck</span>'
    )
    st.markdown(f"""
    <div style="background:{WHITE};border-bottom:1px solid {LINE};
                padding:0;margin:0 -2rem 0 -2rem">
      <div style="padding:0 2rem;height:80px;display:flex;align-items:center;
                  justify-content:space-between">
        <div style="display:flex;align-items:center">{logo_html}</div>
        <div style="font-size:13px;color:{INK2};font-family:'JetBrains Mono',monospace;
                    letter-spacing:.06em">Direct comparison · {cat}</div>
        <div style="width:100px"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    nav_c1, nav_c2, nav_c3 = st.columns([1,1,4], gap="small")
    with nav_c1:
        if st.button("⌂  Home", key="bc_home", use_container_width=True):
            st.session_state.browse_selected  = []
            st.session_state.browse_comparing = False
            st.session_state.browse_chat      = []
            st.session_state.prices           = {}
            st.session_state.prices_fetched   = False
            st.rerun()
    with nav_c2:
        if st.button("← Back", key="bc_back", use_container_width=True):
            st.session_state.browse_comparing = False
            st.rerun()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Header
    names = " vs ".join(p["name"] for p in products)

    st.markdown(f"""
    <div style="margin-bottom:20px">
      <div style="font-size:11px;font-family:'JetBrains Mono',monospace;letter-spacing:.1em;
                  color:#1e3a8a;text-transform:uppercase;margin-bottom:6px">{cat}</div>
      <div style="font-size:24px;font-weight:700;color:{INK};letter-spacing:-.03em">{names}</div>
    </div>""", unsafe_allow_html=True)

    # Fetch prices + generate verdict in parallel
    winner = products[0]
    if not st.session_state.prices_fetched or not st.session_state.verdict_fetched:
        from concurrent.futures import ThreadPoolExecutor
        use_case = "general comparison"
        with st.spinner("Finding best prices…"):
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_p = ex.submit(get_prices_batch, products, cat)
                fut_v = ex.submit(generate_verdict, products, cat, use_case,
                                  "Any budget", "", [])
                st.session_state.prices          = fut_p.result()
                st.session_state.verdict         = fut_v.result()
                st.session_state.prices_fetched  = True
                st.session_state.verdict_fetched = True

    # AI verdict strip
    st.markdown(f"""
    <div style="background:{WHITE};border:1px solid {LINE};border-radius:8px;
                padding:16px 20px;margin-bottom:20px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);
                display:flex;gap:14px;align-items:flex-start">
      <div style="width:32px;height:32px;border-radius:6px;background:#3c59fc;
                  display:flex;align-items:center;justify-content:center;flex-shrink:0">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <path d="M8 2l1.5 4H14l-3.5 2.5 1.5 4L8 10.5 4 12.5l1.5-4L2 6h4.5z" fill="white"/>
        </svg>
      </div>
      <div style="flex:1">
        <div style="font-size:10px;font-family:'JetBrains Mono',monospace;letter-spacing:.08em;
                    color:#3c59fc;text-transform:uppercase;font-weight:600;margin-bottom:6px">
          AI verdict
        </div>
        <div style="font-size:13px;color:{INK};line-height:1.75">{st.session_state.verdict}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Tabs
    t1, t2, t3, t4 = st.tabs(["📊  Comparison", "🎯  Score radar", "🛒  Where to buy", "💬  Spec advisor"])
    with t1:
        _tab_compare(products, cat, pal)
    with t2:
        _tab_radar(products, cat)
    with t3:
        _tab_prices(products, cat, pal)
    with t4:
        _tab_chat_browse(products, cat, pal)

    # ── Footer actions ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{WHITE};border:1px solid {LINE};border-radius:14px;
                padding:22px 24px;margin-top:28px;text-align:center">
      <div style="font-size:15px;font-weight:600;color:{INK};margin-bottom:4px">
        Made your decision?
      </div>
      <div style="font-size:13px;color:{INK2};margin-bottom:18px">
        Jump to the best price, refine with guided preferences, or start fresh.
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap="small")
    with f1:
        if st.button("Start new comparison", use_container_width=True):
            st.session_state.browse_selected  = []
            st.session_state.browse_comparing = False
            st.session_state.prices           = {}
            st.session_state.prices_fetched   = False
            st.session_state.verdict_fetched  = False
            st.session_state.browse_chat      = []
            st.rerun()
    with f2:
        if st.button("Add preferences & re-rank →", use_container_width=True, type="primary"):
            # Send user to step 1 with category pre-set; results will be re-ranked after
            st.session_state.category         = cat
            st.session_state.sel_uses         = []
            st.session_state.sel_prios        = []
            st.session_state.sel_port         = ""
            st.session_state.sel_budget       = ""
            st.session_state.browse_comparing = False
            # Store browse products so step1→step2 can re-score them
            st.session_state.browse_prefill   = [dict(p) for p in products]
            st.session_state.step             = 1
            st.rerun()
    with f3:
        retailers = st.session_state.prices.get(winner["name"], [])
        best_link = retailers[0]["link"] if retailers else "#"
        nm = " ".join(winner["name"].split()[:2])
        st.markdown(f"""
        <a href="{best_link}" target="_blank"
           style="display:block;text-align:center;padding:10px 16px;background:{INK};
                  color:white;border-radius:8px;font-size:13px;font-weight:600;
                  text-decoration:none;font-family:'Inter',sans-serif">
          Buy {nm} →
        </a>""", unsafe_allow_html=True)



def main():
    if st.session_state.browse_comparing:
        render_browse_compare()
        return
    if st.session_state.featured_product is not None:
        render_product_page()
        return
    s = st.session_state.step
    if s == 0:
        render_step0()
    elif s == 1:
        render_step1()
    elif s == 2:
        render_step2()

if __name__=="__main__":
    main()
