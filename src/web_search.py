"""
Live web search via Tavily API.
Falls back gracefully if TAVILY_API_KEY is not set.
"""

import os
import requests
import config

TAVILY_URL = "https://api.tavily.com/search"

PENSION_DOMAINS = [
    "worldbank.org", "oecd.org", "ilo.org", "imf.org", "afdb.org",
    "fsdafrica.org", "pencom.gov.ng", "npra.gov.gh", "rba.go.ke",
    "fsca.co.za", "urbra.or.ug", "ssra.go.tz", "nbfira.org.bw",
    "iopsweb.org", "reuters.com", "bloomberg.com", "ft.com",
    "businessdayonline.com", "myjoyonline.com", "businessdailyafrica.com",
]


def search(
    query: str,
    max_results: int = 8,
    include_domains: list[str] | None = None,
    deep: bool = False,
) -> dict:
    """
    Search the web for pension research and news.
    Returns a dict with 'answer', 'results', and optionally 'error'.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return {
            "error": "TAVILY_API_KEY not configured — add it to your .env file for live web search.",
            "results": [],
            "answer": "",
        }

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced" if deep else "basic",
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains

    try:
        resp = requests.post(TAVILY_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return {
            "query": query,
            "answer": data.get("answer", ""),
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:1500],
                    "published_date": r.get("published_date", ""),
                    "score": round(r.get("score", 0), 3),
                }
                for r in data.get("results", [])
            ],
        }
    except requests.HTTPError as exc:
        return {"error": f"Tavily API error: {exc}", "results": [], "answer": ""}
    except Exception as exc:
        return {"error": str(exc), "results": [], "answer": ""}


def search_pension_news(country: str = "", topic: str = "") -> dict:
    """Convenience wrapper for pension-specific news search."""
    parts = ["Africa pension"]
    if country:
        parts.append(country)
    if topic:
        parts.append(topic)
    parts.append("2024 2025")
    return search(" ".join(parts), max_results=10, include_domains=PENSION_DOMAINS)


def search_regulations(country: str) -> dict:
    """Search for recent regulatory updates for a specific country."""
    return search(
        f"{country} pension regulation law reform 2024 2025",
        max_results=8,
        deep=True,
    )
