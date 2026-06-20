"""
Generates high-quality research outputs for Africa Pension Watch:
policy briefs, articles, comparative country reports, issue notes, etc.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import anthropic

import config
from src import knowledge_base as kb


OUTPUT_TYPES = {
    "policy_brief": {
        "label": "Policy Brief",
        "description": "2–4 page evidence-based brief for policymakers with clear, actionable recommendations",
        "sections": ["Executive Summary", "Background & Context", "Key Findings", "Policy Recommendations", "Implementation Considerations", "Conclusion"],
    },
    "article": {
        "label": "Thought Leadership Article",
        "description": "800–1,200 word article for africapensionwatch.org or external outlets (AllAfrica, BusinessDay, FT Adviser)",
        "sections": ["Opening Hook", "The Problem", "Evidence & Analysis", "International Precedent", "What Needs to Change", "Conclusion"],
    },
    "advocacy_position": {
        "label": "Advocacy Position Paper",
        "description": "Formal position paper setting out Africa Pension Watch's stance on a policy issue, with supporting evidence and calls to action",
        "sections": ["APW Position Statement", "Background", "Evidence Base", "Analysis of Current Approach", "International Best Practice", "APW Recommendations", "Call to Action"],
    },
    "interview_guide": {
        "label": "Stakeholder Interview Guide",
        "description": "Structured interview guide for pension regulators, fund managers, trustees, members, or government officials",
        "sections": ["Interview Objectives", "Respondent Background Questions", "Core Thematic Questions", "Probing Questions", "Closing Questions", "Notes on Methodology"],
    },
    "stakeholder_questions": {
        "label": "Stakeholder Question Set",
        "description": "Sharp, targeted questions for parliamentary hearings, regulatory consultations, or public forums",
        "sections": ["Context & Framing", "Questions for Regulators", "Questions for Fund Managers & Trustees", "Questions for Government", "Questions for Members & Beneficiaries", "Data & Accountability Questions"],
    },
    "comparative_report": {
        "label": "Comparative Country Report",
        "description": "Structured comparison of pension systems across specified countries",
        "sections": ["Overview", "Regulatory Framework Comparison", "Investment Rules", "Coverage and Adequacy", "Reform Priorities", "Summary Matrix"],
    },
    "regulatory_summary": {
        "label": "Regulatory Summary",
        "description": "Summary of pension laws, regulations, and investment guidelines for a jurisdiction",
        "sections": ["Legislative Framework", "Regulatory Authority", "Contribution Rules", "Investment Guidelines", "Member Protections", "Recent Developments"],
    },
    "issue_note": {
        "label": "Issue Note",
        "description": "Short focused note (500-800 words) on a specific pension reform issue",
        "sections": ["The Issue", "Evidence", "International Practice", "Recommended Action"],
    },
    "research_agenda": {
        "label": "Research Agenda",
        "description": "Prioritised list of research questions and data gaps for Africa Pension Watch",
        "sections": ["Strategic Research Themes", "Priority Questions by Theme", "Data Gaps", "Methodology Notes", "Suggested Outputs"],
    },
    "article_ideas": {
        "label": "Article Ideas",
        "description": "List of article and research ideas with hooks, angles, and key questions",
        "sections": ["High Priority Ideas", "Medium Priority Ideas", "Data-Driven Ideas", "Interview Angles"],
    },
}


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _build_context_from_kb(topic: str, countries: list[str]) -> str:
    """Pull relevant documents from the knowledge base to ground the output."""
    results = kb.search(topic, limit=8)
    if countries:
        for country in countries[:3]:
            results += kb.search(country, limit=3, jurisdiction=country)

    seen = set()
    unique = []
    for r in results:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)

    if not unique:
        return "No documents currently in the knowledge base on this topic."

    ctx_parts = []
    for r in unique[:10]:
        ctx_parts.append(
            f"Source: {r.get('source_name', 'Unknown')} | {r.get('date_published', 'n/d')}\n"
            f"Title: {r['title']}\n"
            f"Summary: {r.get('summary', 'No summary available.')}\n"
        )
    return "\n---\n".join(ctx_parts)


def _load_jurisdiction_data(countries: list[str]) -> str:
    """Load structured jurisdiction profiles for specified countries."""
    jdata_path = config.DATA_DIR / "jurisdictions.json"
    if not jdata_path.exists():
        return ""
    jdata = json.loads(jdata_path.read_text(encoding="utf-8"))
    profiles = []
    for j in jdata.get("jurisdictions", []):
        if not countries or j["country"] in countries:
            profiles.append(json.dumps(j, indent=2))
    return "\n\n".join(profiles[:5]) if profiles else ""


def _build_prompts(
    output_type: str, topic: str, countries: list[str],
    audience: str, additional_context: str,
) -> tuple[str, str, dict]:
    """Return (system_prompt, user_prompt, otype_dict)."""
    otype = OUTPUT_TYPES[output_type]
    kb_context = _build_context_from_kb(topic, countries)
    jurisdiction_context = _load_jurisdiction_data(countries)
    sections_str = "\n".join(f"- {s}" for s in otype["sections"])
    countries_str = ", ".join(countries) if countries else "Africa broadly"

    system = f"""You are a senior research analyst and policy writer at {config.ORG_NAME}.
{config.ORG_DESCRIPTION}

Your writing is analytical, evidence-based, precise, and suitable for senior pension industry
professionals, regulators, and policymakers. You draw on international best practice while
remaining grounded in African realities. You identify gaps, contradictions, and reform opportunities.
You never pad content — every sentence earns its place.

Today's date: {datetime.now().strftime("%B %Y")}"""

    user_prompt = f"""Produce a {otype['label']} on the following topic for Africa Pension Watch.

TOPIC: {topic}
COUNTRIES/FOCUS: {countries_str}
AUDIENCE: {audience}
OUTPUT TYPE: {otype['description']}

REQUIRED SECTIONS:
{sections_str}

KNOWLEDGE BASE CONTEXT (use this as your primary evidence base):
{kb_context}

JURISDICTION DATA:
{jurisdiction_context}

ADDITIONAL INSTRUCTIONS:
{additional_context}

Write the full {otype['label']} now. Use clear headings for each section.
Where the knowledge base lacks specific data, say so and recommend what research would fill the gap.
Do not fabricate statistics — if you cite a number, it must be grounded in the context above or clearly
qualified as an approximation."""

    return system, user_prompt, otype


def generate_streaming(
    output_type: str,
    topic: str,
    countries: list[str] | None = None,
    audience: str = "pension policymakers and practitioners",
    additional_context: str = "",
    use_deep_model: bool = False,
):
    """
    Stream a research output. Yields {"type":"text","content":chunk} for each text delta,
    then a final {"type":"done","title":...,"filename":...,"model_used":...,"content":...}.
    """
    if output_type not in OUTPUT_TYPES:
        raise ValueError(f"Unknown output type '{output_type}'")

    countries = countries or []
    model = config.DEEP_MODEL if use_deep_model else config.MODEL
    system, user_prompt, otype = _build_prompts(output_type, topic, countries, audience, additional_context)

    full_text: list[str] = []
    with _client().messages.stream(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_prompt}],
        system=system,
    ) as stream:
        for event in stream:
            if getattr(event, "type", None) == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and getattr(delta, "type", None) == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        full_text.append(text)
                        yield {"type": "text", "content": text}

    content = "".join(full_text).strip()
    title_line = content.split("\n")[0].lstrip("#").strip()
    if not title_line or len(title_line) > 120:
        title_line = f"{otype['label']}: {topic}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r"[^\w\s-]", "", topic)[:40].strip().replace(" ", "_")
    filename = f"{output_type}_{safe_topic}_{timestamp}.md"
    output_path = config.OUTPUTS_DIR / filename
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    yield {
        "type": "done",
        "title": title_line,
        "filename": filename,
        "model_used": model,
        "content": content,
    }


def generate(
    output_type: str,
    topic: str,
    countries: list[str] | None = None,
    audience: str = "pension policymakers and practitioners",
    additional_context: str = "",
    use_deep_model: bool = False,
) -> dict:
    """
    Generate a research output document.

    Args:
        output_type: Key from OUTPUT_TYPES dict
        topic: The specific topic or question to address
        countries: List of African countries to focus on (optional)
        audience: Target audience description
        additional_context: Any extra instructions or context
        use_deep_model: Use Opus (deeper analysis) vs Sonnet (faster)

    Returns:
        dict with keys: title, content, output_type, topic, countries, model_used, timestamp
    """
    if output_type not in OUTPUT_TYPES:
        raise ValueError(f"Unknown output type '{output_type}'. Choose from: {list(OUTPUT_TYPES)}")

    countries = countries or []
    model = config.DEEP_MODEL if use_deep_model else config.MODEL
    system, user_prompt, otype = _build_prompts(output_type, topic, countries, audience, additional_context)

    response = _client().messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_prompt}],
        system=system,
    )

    content = response.content[0].text.strip()

    title_line = content.split("\n")[0].lstrip("#").strip()
    if not title_line or len(title_line) > 120:
        title_line = f"{otype['label']}: {topic}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r"[^\w\s-]", "", topic)[:40].strip().replace(" ", "_")
    filename = f"{output_type}_{safe_topic}_{timestamp}.md"
    output_path = config.OUTPUTS_DIR / filename

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    return {
        "title": title_line,
        "content": content,
        "output_type": output_type,
        "topic": topic,
        "countries": countries,
        "model_used": model,
        "timestamp": timestamp,
        "saved_to": str(output_path),
    }


def list_output_types() -> list[dict]:
    return [
        {"id": k, "label": v["label"], "description": v["description"]}
        for k, v in OUTPUT_TYPES.items()
    ]
