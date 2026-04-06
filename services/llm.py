"""
services/llm.py

Two functions:
  generate_verdict()  — one-shot structured summary of the top products
  stream_chat()       — streaming generator for the "Ask the data" chat
"""

import os
import anthropic

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    return _client


def _build_product_context(products: list[dict], category: str) -> str:
    lines = [f"Category: {category}\n"]
    for p in products:
        lines.append(f"Product: {p['name']} by {p['brand']} ({p['year']})")
        lines.append(f"  Price: ${p['price']}")
        lines.append(f"  Match score: {p.get('match_score', p.get('base_score', 'N/A')):.0f}/100")
        lines.append(f"  Avg rating: {p['avg_rating']}/5 from {p['review_count']:,} reviews")
        lines.append(f"  Positive sentiment: {p['pos_pct']}%")
        lines.append(f"  Praised for: {p['pos_topics']}")
        lines.append(f"  Criticised for: {p['neg_topics']}")
        if category == "Laptops":
            lines.append(f"  CPU score: {p['cpu_score']}, RAM: {p['ram_gb']} GB, "
                         f"Battery: {p['battery_h']} h, Weight: {p['weight_kg']} kg, "
                         f"Display score: {p['display_score']}/100, GPU: {p['gpu_score']}/100")
        elif category == "Smartphones":
            lines.append(f"  CPU: {p['cpu_score']}/100, RAM: {p['ram_gb']} GB, "
                         f"Battery: {p['battery_h']} h, Display: {p['display_score']}/100")
        lines.append("")
    return "\n".join(lines)


def generate_verdict(
    products: list[dict],
    category: str,
    use_case: str,
    budget: str,
    portability: str,
    priorities: list = None,
) -> str:
    """
    Returns a 2-3 sentence plain-English verdict grounded in the product data.
    Synchronous — call inside st.spinner().
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        # graceful fallback for demo without key
        top = products[0]["name"]
        second = products[1]["name"] if len(products) > 1 else ""
        parts = [f"The {top} leads for {use_case.lower()} with the strongest overall match."]
        if second:
            parts.append(f" The {second} is worth a close look as a solid alternative.")
        parts.append(" Compare specs and live prices in the tabs below.")
        return "".join(parts)

    context = _build_product_context(products, category)
    priorities_line = ""
    if priorities:
        ranked = " > ".join(f"#{i+1} {p}" for i, p in enumerate(priorities))
        priorities_line = f"- Feature priorities (in order): {ranked}\n"

    prompt = f"""You are the AI engine behind SpecCheck, a product comparison service.

User profile:
- Category: {category}
- Primary use: {use_case}
- Budget: {budget}
- Portability preference: {portability}
{priorities_line}
Product data (ranked by match score):
{context}

Write a verdict of exactly 2-3 sentences. Rules:
- Name the top product and explain WHY it wins for this specific user profile, directly addressing their #1 priority if one was given
- Mention one concrete advantage of the 2nd product (a reason someone might prefer it)
- End with a one-line note on the 3rd product if there are 3+
- Use specific numbers from the data (scores, hours, kg, %) — no vague claims
- Plain text only, no markdown, no bullet points
- Tone: confident, honest, like a trusted friend who knows tech
- Do NOT start with "Based on" or "According to"
"""

    try:
        msg = _get_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception:
        top = products[0]["name"]
        second = products[1]["name"] if len(products) > 1 else ""
        parts = [f"The {top} leads the shortlist for {use_case.lower()}."]
        if second:
            parts.append(f" The {second} is a strong alternative.")
        parts.append(" Compare specs and live prices in the tabs below.")
        return "".join(parts)


def stream_chat(
    question: str,
    products: list[dict],
    category: str,
    use_case: str,
    history: list[dict],
):
    """
    Yields text chunks for streaming display in Streamlit.
    history format: [{"role": "user"|"assistant", "content": "..."}]
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        yield "Add your ANTHROPIC_API_KEY to .env to enable the AI chat feature."
        return

    context = _build_product_context(products, category)

    system = f"""You are the AI assistant inside SpecCheck, a product comparison tool and a knowledgeable tech advisor.
You have access to real spec data and review sentiment for the products the user is comparing, and you can also draw on your broader knowledge of tech products, brands, real-world performance, longevity, resale value, and industry trends.
Use both the product data below AND your own knowledge to give genuinely useful, insightful answers.
Where relevant, cite numbers from the data. Where the question calls for broader expertise (e.g. reliability over time, value for money, ecosystem fit), answer confidently from your knowledge.
Keep answers under 200 words.

Current comparison data:
{context}

User's use case: {use_case}"""

    messages = []
    for h in history[-6:]:  # keep last 3 turns to avoid context bloat
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})

    try:
        with _get_client().messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception:
        yield "AI chat is unavailable right now. Check your ANTHROPIC_API_KEY and credit balance."
