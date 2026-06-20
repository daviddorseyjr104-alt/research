"""
Africa Pension Watch Research Intelligence Agent.
Claude-powered agent with tools for knowledge base access,
jurisdiction comparison, and research output generation.
"""

import json
from datetime import datetime
from typing import Any

import anthropic

import config
from src import knowledge_base as kb, web_scanner, report_generator, document_processor, web_search


def _build_system_prompt() -> str:
    return f"""You are the Africa Pension Watch Research Intelligence Agent — an AI research
assistant for {config.ORG_NAME}, an independent advocacy group and think tank.

{config.ORG_DESCRIPTION}

Today's date: {datetime.now().strftime("%d %B %Y")}

RESEARCH WORKFLOW — follow this order for every substantive question:
1. search_knowledge_base — check stored research first
2. search_web — ALWAYS search the live web for recent news and research (you have Tavily access)
3. get_jurisdiction_profile or compare_jurisdictions — pull structured country data
4. Synthesise findings into a clear, analytical response with source attribution

CAPABILITIES:
- Live web search for current pension news, regulatory updates, and research (Tavily)
- Full knowledge base search across ingested documents
- 16 structured African pension jurisdiction profiles
- Generate 10 output types: policy briefs, articles, advocacy positions, interview guides,
  stakeholder questions, comparative reports, regulatory summaries, issue notes, research
  agendas, and article idea lists
- Ingest any URL (web page or PDF) into the knowledge base

KEY FOCUS AREAS for Africa Pension Watch:
- Coverage expansion — particularly informal sector inclusion (Africa average: ~10%)
- Governance and trustee capacity building
- Investment guidelines — offshore limits, alternative assets, infrastructure
- Regulatory frameworks and enforcement gaps
- Member protection: portability, preservation, benefit adequacy
- Annuitization and longevity risk management
- Pension fund transparency, disclosure, and accountability
- Role of pension capital in African infrastructure and development
- Actuarial soundness and long-term sustainability
- Cross-country regulatory arbitrage and best-practice transfer

ANALYTICAL STANDARDS:
- Always cite sources (URL, publication, date) when stating facts
- Distinguish law from regulatory practice from on-the-ground reality
- Flag data that is outdated, unreliable, or unverified
- Identify gaps, contradictions, and implementation failures
- Recommend practical, politically feasible reforms — not just aspirational ones
- Compare Africa to international best practice (OECD, WB, ILO standards)
- Be direct and analytical. Avoid hedging on matters where evidence is clear.

WRITING STYLE for generated outputs:
- Publication-quality prose suitable for senior policymakers and pension professionals
- Evidence-based, with specific country examples and data points
- Clearly structured with actionable conclusions
- Africa Pension Watch voice: independent, rigorous, reform-oriented"""


TOOLS: list[dict] = [
    {
        "name": "search_knowledge_base",
        "description": "Full-text search across research papers, regulations, reports, and articles in the knowledge base. Use this before answering any research question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "jurisdiction": {"type": "string", "description": "Filter by country/region (optional)"},
                "doc_type": {"type": "string", "description": "Filter by doc type: research_paper, regulation, annual_report, policy_brief, news_article, etc. (optional)"},
                "limit": {"type": "integer", "description": "Max results (default 8)", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document",
        "description": "Retrieve the full content of a specific document from the knowledge base by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "integer", "description": "Document ID from search results"},
            },
            "required": ["doc_id"],
        },
    },
    {
        "name": "list_recent_documents",
        "description": "List the most recently added documents in the knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of results (default 15)"},
                "jurisdiction": {"type": "string", "description": "Filter by country (optional)"},
                "doc_type": {"type": "string", "description": "Filter by doc type (optional)"},
            },
        },
    },
    {
        "name": "get_knowledge_base_stats",
        "description": "Get statistics about the knowledge base: total documents, breakdown by type and jurisdiction.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "fetch_and_ingest_url",
        "description": "Fetch a URL (web page or PDF), extract its content, and add it to the knowledge base. Use when the user provides a specific URL to research.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch"},
                "jurisdiction": {"type": "string", "description": "Country or region this document relates to"},
                "doc_type": {"type": "string", "description": "Document type classification"},
                "source_name": {"type": "string", "description": "Name of the publishing organization"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_jurisdiction_profile",
        "description": "Get the detailed pension system profile for a specific African country, including regulator, legislation, contribution rates, investment rules, AUM, coverage, key challenges, and reform status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "Country name (e.g., Nigeria, Kenya, South Africa)"},
            },
            "required": ["country"],
        },
    },
    {
        "name": "list_jurisdictions",
        "description": "List all African pension jurisdictions tracked in the database with key summary metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Filter by region: West Africa, East Africa, Southern Africa, North Africa, Central Africa (optional)"},
            },
        },
    },
    {
        "name": "compare_jurisdictions",
        "description": "Generate a structured comparison of pension systems across specified countries on a given dimension (e.g., offshore investment limits, contribution rates, governance, coverage).",
        "input_schema": {
            "type": "object",
            "properties": {
                "countries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of country names to compare",
                },
                "dimension": {
                    "type": "string",
                    "description": "What to compare: 'offshore_limits', 'contribution_rates', 'coverage', 'legislation', 'investment_rules', 'challenges', 'reforms', or a custom dimension",
                },
            },
            "required": ["countries", "dimension"],
        },
    },
    {
        "name": "generate_research_output",
        "description": "Generate a high-quality research output document for Africa Pension Watch. Outputs are saved to the outputs/ folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "output_type": {
                    "type": "string",
                    "enum": ["policy_brief", "article", "comparative_report", "regulatory_summary", "issue_note", "research_agenda", "article_ideas", "advocacy_position", "interview_guide", "stakeholder_questions"],
                    "description": "Type of output to generate",
                },
                "topic": {"type": "string", "description": "Specific topic or research question to address"},
                "countries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Countries to focus on (optional, leave empty for Africa-wide)",
                },
                "audience": {"type": "string", "description": "Target audience (default: pension policymakers and practitioners)"},
                "additional_context": {"type": "string", "description": "Any specific angles, instructions, or context for the output"},
                "use_deep_model": {"type": "boolean", "description": "Use Claude Opus for deeper analysis (slower, higher quality). Default false."},
            },
            "required": ["output_type", "topic"],
        },
    },
    {
        "name": "scan_source",
        "description": "Scan a specific source from the sources registry to ingest new documents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source ID from sources.json (e.g., pencom_nigeria, worldbank_pensions)"},
            },
            "required": ["source_id"],
        },
    },
    {
        "name": "list_sources",
        "description": "List all monitored sources in the registry, optionally filtered by type or country.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_type": {"type": "string", "description": "Filter by type: regulator, multilateral, national_fund, development_finance, consulting, professional_body (optional)"},
                "country": {"type": "string", "description": "Filter by country (optional)"},
            },
        },
    },
    {
        "name": "search_web",
        "description": (
            "Search the live internet for current pension news, research papers, regulatory updates, "
            "and market commentary. Use when the knowledge base lacks up-to-date information or when "
            "the user asks about recent developments. Requires TAVILY_API_KEY to be configured."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query — be specific, e.g. 'Nigeria PenCom investment regulations 2024'"},
                "max_results": {"type": "integer", "description": "Number of results (default 8)", "default": 8},
                "deep": {"type": "boolean", "description": "Use deep search for more thorough results (slower)", "default": False},
                "restrict_to_pension_sources": {"type": "boolean", "description": "Restrict to known pension/finance domains", "default": False},
            },
            "required": ["query"],
        },
    },
]


def _load_jurisdictions() -> list[dict]:
    path = config.DATA_DIR / "jurisdictions.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("jurisdictions", [])


def _load_sources() -> list[dict]:
    path = config.DATA_DIR / "sources.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("sources", [])


def _execute_tool(name: str, inputs: dict) -> Any:
    """Dispatch a tool call and return a serializable result."""

    if name == "search_knowledge_base":
        results = kb.search(
            query=inputs["query"],
            limit=inputs.get("limit", 8),
            jurisdiction=inputs.get("jurisdiction", ""),
            doc_type=inputs.get("doc_type", ""),
        )
        if not results:
            return {"message": "No documents found. The knowledge base may be empty — try running a scan or ingesting a URL first."}
        return {"count": len(results), "results": results}

    if name == "get_document":
        doc = kb.get_document(inputs["doc_id"])
        if not doc:
            return {"error": f"Document {inputs['doc_id']} not found"}
        return doc

    if name == "list_recent_documents":
        docs = kb.list_recent(
            limit=inputs.get("limit", 15),
            jurisdiction=inputs.get("jurisdiction", ""),
            doc_type=inputs.get("doc_type", ""),
        )
        return {"count": len(docs), "documents": docs}

    if name == "get_knowledge_base_stats":
        return kb.stats()

    if name == "fetch_and_ingest_url":
        result = web_scanner.ingest_url(
            url=inputs["url"],
            jurisdiction=inputs.get("jurisdiction", "global"),
            doc_type=inputs.get("doc_type", ""),
            source_name=inputs.get("source_name", ""),
        )
        if result["success"]:
            doc_id = result["doc_id"]
            enrichment = document_processor.summarize_document(doc_id)
            result["enrichment"] = enrichment
        return result

    if name == "get_jurisdiction_profile":
        country = inputs["country"]
        jurisdictions = _load_jurisdictions()
        for j in jurisdictions:
            if j["country"].lower() == country.lower():
                kb_docs = kb.search(country, limit=5)
                j["knowledge_base_documents"] = kb_docs
                return j
        return {"error": f"No profile found for '{country}'. Available: {[j['country'] for j in jurisdictions]}"}

    if name == "list_jurisdictions":
        jurisdictions = _load_jurisdictions()
        region = inputs.get("region", "")
        if region:
            jurisdictions = [j for j in jurisdictions if j.get("region", "").lower() == region.lower()]
        summary = [
            {
                "country": j["country"],
                "region": j.get("region", ""),
                "regulator": j.get("regulator", ""),
                "system_type": j.get("system_type", ""),
                "estimated_aum_usd_bn": j.get("estimated_aum_usd_bn"),
                "coverage_rate_pct": j.get("coverage_rate_pct"),
                "offshore_limit_pct": j.get("offshore_limit_pct"),
            }
            for j in jurisdictions
        ]
        return {"count": len(summary), "jurisdictions": summary}

    if name == "compare_jurisdictions":
        countries = inputs["countries"]
        dimension = inputs["dimension"]
        jurisdictions = _load_jurisdictions()
        country_map = {j["country"].lower(): j for j in jurisdictions}

        comparison = {}
        for c in countries:
            profile = country_map.get(c.lower())
            if not profile:
                comparison[c] = {"error": "Profile not found"}
                continue

            dim = dimension.lower()
            if "offshore" in dim:
                comparison[c] = {
                    "offshore_limit_pct": profile.get("offshore_limit_pct"),
                    "africa_offshore_limit_pct": profile.get("africa_offshore_limit_pct"),
                    "investment_framework": profile.get("investment_framework", ""),
                }
            elif "contribution" in dim:
                comparison[c] = {
                    "total_contribution_rate_pct": profile.get("total_contribution_rate_pct"),
                    "employer_contribution_pct": profile.get("employer_contribution_pct"),
                    "employee_contribution_pct": profile.get("employee_contribution_pct"),
                    "mandatory_for": profile.get("mandatory_for", ""),
                }
            elif "coverage" in dim:
                comparison[c] = {
                    "coverage_rate_pct": profile.get("coverage_rate_pct"),
                    "system_type": profile.get("system_type", ""),
                    "mandatory_for": profile.get("mandatory_for", ""),
                }
            elif "legislation" in dim or "law" in dim or "regulation" in dim:
                comparison[c] = {
                    "primary_legislation": profile.get("primary_legislation", ""),
                    "legislation_year": profile.get("legislation_year"),
                    "amendments": profile.get("amendments", []),
                    "regulator": profile.get("regulator", ""),
                    "pending_reforms": profile.get("pending_reforms", []),
                }
            elif "challenge" in dim:
                comparison[c] = {
                    "key_challenges": profile.get("key_challenges", []),
                    "key_strengths": profile.get("key_strengths", []),
                }
            elif "reform" in dim:
                comparison[c] = {
                    "recent_reforms": profile.get("recent_reforms", []),
                    "pending_reforms": profile.get("pending_reforms", []),
                }
            else:
                fields = [
                    "system_type", "total_contribution_rate_pct", "coverage_rate_pct",
                    "offshore_limit_pct", "estimated_aum_usd_bn", "legislation_year",
                    "retirement_age", "primary_legislation",
                ]
                comparison[c] = {k: profile.get(k) for k in fields}

        return {"dimension": dimension, "comparison": comparison}

    if name == "generate_research_output":
        result = report_generator.generate(
            output_type=inputs["output_type"],
            topic=inputs["topic"],
            countries=inputs.get("countries", []),
            audience=inputs.get("audience", "pension policymakers and practitioners"),
            additional_context=inputs.get("additional_context", ""),
            use_deep_model=inputs.get("use_deep_model", False),
        )
        return {
            "title": result["title"],
            "output_type": result["output_type"],
            "saved_to": result["saved_to"],
            "model_used": result["model_used"],
            "preview": result["content"][:800] + "\n\n[... saved to file ...]",
        }

    if name == "scan_source":
        sources = _load_sources()
        source_id = inputs["source_id"]
        source = next((s for s in sources if s["id"] == source_id), None)
        if not source:
            return {"error": f"Source '{source_id}' not found. Use list_sources to see available sources."}
        count = web_scanner.scan_source(source)
        return {"source": source["name"], "documents_ingested": count}

    if name == "list_sources":
        sources = _load_sources()
        stype = inputs.get("source_type", "")
        country = inputs.get("country", "")
        if stype:
            sources = [s for s in sources if s.get("type", "") == stype]
        if country:
            sources = [s for s in sources if s.get("country", "").lower() == country.lower()]
        summary = [
            {
                "id": s["id"],
                "name": s["name"],
                "type": s.get("type", ""),
                "country": s.get("country", "global"),
                "priority": s.get("priority", ""),
                "url": s.get("url", ""),
            }
            for s in sources
        ]
        return {"count": len(summary), "sources": summary}

    if name == "search_web":
        result = web_search.search(
            query=inputs["query"],
            max_results=inputs.get("max_results", 8),
            deep=inputs.get("deep", False),
            include_domains=web_search.PENSION_DOMAINS if inputs.get("restrict_to_pension_sources") else None,
        )
        # Auto-ingest top results into the knowledge base so they're searchable later
        for r in result.get("results", [])[:4]:
            if r.get("url") and r.get("content") and len(r["content"]) > 100:
                kb.add_document(
                    title=r.get("title", r["url"]),
                    content=r["content"],
                    url=r["url"],
                    summary=r["content"][:400],
                    doc_type="news_article",
                    jurisdiction="global",
                    source_name="Web Search / Tavily",
                    date_published=r.get("published_date", ""),
                )
        return result

    return {"error": f"Unknown tool: {name}"}


def _tool_status(name: str) -> str:
    labels = {
        "search_knowledge_base": "Searching knowledge base…",
        "search_web": "Searching the web…",
        "get_jurisdiction_profile": "Loading jurisdiction profile…",
        "list_jurisdictions": "Loading jurisdiction data…",
        "compare_jurisdictions": "Comparing jurisdictions…",
        "generate_research_output": "Generating research output…",
        "fetch_and_ingest_url": "Fetching document…",
        "list_recent_documents": "Loading recent documents…",
        "get_knowledge_base_stats": "Loading statistics…",
        "get_document": "Retrieving document…",
        "scan_source": "Scanning source…",
        "list_sources": "Loading sources…",
    }
    return labels.get(name, f"Using {name.replace('_', ' ')}…")


class Agent:
    """Stateful agent that maintains conversation history."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.history: list[dict] = []

    def _repair_history(self):
        """Remove any trailing assistant message that has tool_use blocks without
        a following tool_result message. Prevents 400 errors after partial failures."""
        while self.history:
            last = self.history[-1]
            if last["role"] != "assistant":
                break
            content = last.get("content", "")
            if not isinstance(content, list):
                break
            has_tool_use = any(
                getattr(b, "type", None) == "tool_use" or
                (isinstance(b, dict) and b.get("type") == "tool_use")
                for b in content
            )
            if has_tool_use:
                self.history.pop()
            else:
                break

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, executing any tool calls."""
        self._repair_history()
        self.history.append({"role": "user", "content": user_message})

        while True:
            response = self.client.messages.create(
                model=config.MODEL,
                max_tokens=4096,
                system=_build_system_prompt(),
                tools=TOOLS,
                messages=self.history,
            )

            if response.stop_reason == "end_turn":
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text
                self.history.append({"role": "assistant", "content": response.content})
                return text

            if response.stop_reason == "tool_use":
                self.history.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        try:
                            result = _execute_tool(block.name, block.input)
                        except Exception as exc:
                            result = {"error": str(exc)}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, indent=2, default=str),
                        })

                self.history.append({"role": "user", "content": tool_results})
                continue

            break

        return "(No response generated)"

    def chat_streaming(self, user_message: str):
        """
        Generator that yields SSE-ready dicts for streaming responses.
        Handles tool calls transparently, yielding status and text chunks.
        Caller must save messages to KB — this only manages self.history.
        """
        self._repair_history()
        self.history.append({"role": "user", "content": user_message})

        while True:
            try:
                with self.client.messages.stream(
                    model=config.MODEL,
                    max_tokens=4096,
                    system=_build_system_prompt(),
                    tools=TOOLS,
                    messages=self.history,
                ) as stream:
                    for event in stream:
                        etype = getattr(event, "type", None)
                        if etype == "content_block_start":
                            cb = getattr(event, "content_block", None)
                            if cb and getattr(cb, "type", None) == "tool_use":
                                yield {"type": "status", "content": _tool_status(cb.name)}
                        elif etype == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta and getattr(delta, "type", None) == "text_delta":
                                text = getattr(delta, "text", "")
                                if text:
                                    yield {"type": "text", "content": text}
                    message = stream.get_final_message()
            except Exception as exc:
                yield {"type": "error", "content": str(exc)}
                return

            self.history.append({"role": "assistant", "content": message.content})

            if message.stop_reason == "end_turn":
                break

            if message.stop_reason == "tool_use":
                tool_results = []
                for block in message.content:
                    if getattr(block, "type", None) == "tool_use":
                        try:
                            result = _execute_tool(block.name, dict(block.input))
                        except Exception as exc:
                            result = {"error": str(exc)}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, indent=2, default=str),
                        })
                self.history.append({"role": "user", "content": tool_results})
            else:
                break

    def reset(self):
        """Clear conversation history to start a new session."""
        self.history = []
