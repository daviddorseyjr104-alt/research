"""
Uses Claude to summarize, classify, and enrich documents in the knowledge base.
"""

import json
import sqlite3

import anthropic

import config
from src import knowledge_base as kb


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def summarize_document(doc_id: int) -> dict:
    """
    Generate a structured summary for a document using Claude.
    Updates the document in the knowledge base.
    Returns the enrichment dict.
    """
    doc = kb.get_document(doc_id)
    if not doc:
        return {"error": "Document not found"}

    content_preview = (doc.get("content") or "")[:8000]
    if not content_preview.strip():
        return {"error": "No content to summarize"}

    client = _client()
    prompt = f"""You are a pension research analyst at Africa Pension Watch.

Analyze the following document and return a JSON object with these fields:
- "summary": 3-5 sentence executive summary capturing the key findings or content
- "key_points": array of 3-7 bullet point key takeaways
- "doc_type": one of: research_paper, regulation, annual_report, policy_brief, news_article, consultation_paper, draft_legislation, actuarial_report, market_commentary, investment_guideline, speech, press_release, data_release, comparative_study
- "jurisdiction": primary country or region covered (use ISO country name, or "global", "Sub-Saharan Africa", "East Africa", etc.)
- "topics": array of relevant topic tags from: coverage, governance, investment, regulation, benefits, actuarial, infrastructure, transparency, reform, market_data, annuitization, offshore
- "relevance_score": integer 1-10 for relevance to African pension reform

Document title: {doc.get('title', '')}
Document URL: {doc.get('url', '')}

Content:
{content_preview}

Return ONLY valid JSON, no markdown fences."""

    try:
        response = client.messages.create(
            model=config.MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        enrichment = json.loads(raw)
    except Exception as exc:
        return {"error": f"Claude processing failed: {exc}"}

    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            UPDATE documents
            SET summary = ?, doc_type = ?, jurisdiction = ?, topics = ?, is_summarized = 1
            WHERE id = ?
            """,
            (
                enrichment.get("summary", ""),
                enrichment.get("doc_type", doc.get("doc_type", "")),
                enrichment.get("jurisdiction", doc.get("jurisdiction", "global")),
                json.dumps(enrichment.get("topics", [])),
                doc_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return enrichment


def process_unsummarized(max_docs: int = 20) -> int:
    """Process all documents that haven't been summarized yet. Returns count processed."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id FROM documents WHERE is_summarized = 0 AND content != '' LIMIT ?",
            (max_docs,),
        ).fetchall()
        doc_ids = [r["id"] for r in rows]
    finally:
        conn.close()

    processed = 0
    for doc_id in doc_ids:
        result = summarize_document(doc_id)
        if "error" not in result:
            processed += 1
    return processed
