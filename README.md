# SpecCheck — AI-powered product comparison

A 3-step Streamlit app that matches you to the best laptop, smartphone, headphone, or monitor for your needs — using a trained Gradient Boosting model, Claude AI for reasoning, and live retailer prices via SerpAPI.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API keys
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY (required) and SERPAPI_KEY (optional)

# 3. Build dataset + train models (run once)
python pipeline/build_dataset.py

# 4. Launch
streamlit run app.py
```

## How it works

**Step 1 — Category:** Pick Laptops, Smartphones, Headphones, or Monitors. Use the search bar to jump to a specific product. Expand "Browse & compare" to hand-pick 2–3 products for a direct side-by-side.

**Step 2 — Your needs:** Select use-cases, rank priorities (① = most important), portability, and budget. Both use-case and budget are required to continue. Priorities reorder dynamically based on your use-case selection.

**Step 3 — Results:**
- **Comparison** — match score cards, spec table with best/lowest highlights, review sentiment
- **Score radar** — Plotly radar chart across 5 dimensions
- **Where to buy** — live prices via SerpAPI or hardcoded fallback, sorted cheapest-first
- **Ask the data** — streaming Claude chat grounded in your shortlisted products' specs

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit `layout="centered"`, Inter + JetBrains Mono |
| ML | scikit-learn `GradientBoostingRegressor` (no system deps) |
| AI | Anthropic Claude `claude-sonnet-4-6`, streaming |
| Prices | SerpAPI Google Shopping, session-cached |
| Data | Synthetic dataset — 98 products across 4 categories |

## API keys

- **ANTHROPIC_API_KEY** (required) — AI verdict + chat. Falls back to template text without it.
- **SERPAPI_KEY** (optional) — Live retailer prices. Falls back to hardcoded prices with real links.

### Local development
```bash
cp .env.example .env
# Edit .env — add your keys
```

### Streamlit Cloud deployment
1. Push to GitHub (without `.env` — it's in `.gitignore`)
2. In Streamlit Cloud → App settings → **Secrets**, paste:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
SERPAPI_KEY = "your_serpapi_key"
```
The app auto-detects whether keys come from `.env` or `st.secrets`.

## Dataset

98 products (33 laptops, 27 smartphones, 19 headphones, 19 monitors) with realistic specs normalised 0–100. A `GradientBoostingRegressor` is trained per category, then adjusted at query time by use-case weights and ranked priority boosts.
