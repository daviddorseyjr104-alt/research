"""
Web scanner: fetches pages from trusted pension research sources,
extracts relevant content, and stores it in the knowledge base.
"""

import json
import time
import re
import io
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import config
from src import knowledge_base as kb


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT})
    return s


def _is_pdf(response: requests.Response) -> bool:
    ct = response.headers.get("content-type", "")
    return "pdf" in ct or response.url.lower().endswith(".pdf")


def _extract_text_from_html(html: str, base_url: str) -> tuple[str, list[str]]:
    """Return (main_text, list_of_links) from an HTML page."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.startswith("javascript"):
            continue
        full = urljoin(base_url, href)
        if urlparse(full).scheme in ("http", "https"):
            links.append(full)

    return text[:config.MAX_CONTENT_LENGTH], links


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber, fall back to pypdf."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = pdf.pages[:config.MAX_PDF_PAGES]
            return "\n\n".join(p.extract_text() or "" for p in pages)
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        pages = reader.pages[:config.MAX_PDF_PAGES]
        return "\n\n".join(p.extract_text() or "" for p in pages)
    except Exception:
        return ""


PENSION_KEYWORDS = {
    "pension", "retirement", "provident fund", "annuity", "contribution",
    "social security", "RSA", "NSSF", "PenCom", "NPRA", "RBA", "FSCA",
    "actuarial", "trustee", "fiduciary", "defined benefit", "defined contribution",
    "longevity", "superannuation", "retraite", "prévoyance",
    "fund manager", "beneficiary", "withdrawal", "vesting", "gratuity",
    "occupational scheme", "mandatory scheme", "voluntary scheme",
}

# URL fragments that suggest high-value content (PDFs, reports, publications)
_PRIORITY_TERMS = frozenset({
    "pdf", "report", "publication", "annual", "research", "paper",
    "policy", "regulation", "circular", "guideline", "press", "release",
    "statistics", "data", "download", "document", "bulletin", "gazette",
    "act", "bill", "amendment", "directive", "notice", "framework",
})


def _link_priority(link: str) -> int:
    """Return sort key: 0=PDF, 1=high-value URL, 2=other. Lower = first."""
    l = link.lower()
    if l.endswith(".pdf"):
        return 0
    if any(t in l for t in _PRIORITY_TERMS):
        return 1
    return 2


def _is_relevant(text: str, title: str = "") -> bool:
    combined = (title + " " + text[:2000]).lower()
    return any(kw in combined for kw in PENSION_KEYWORDS)


def fetch_url(
    url: str,
    source_id: str = "",
    session: requests.Session | None = None,
) -> dict | None:
    """
    Fetch a URL and return a dict with keys:
        title, content, is_pdf, url, raw_links
    Returns None on failure.
    """
    sess = session or _session()
    try:
        resp = sess.get(url, timeout=config.REQUEST_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        kb.log_scan(source_id, url, "error", error_msg=str(exc))
        return None

    if _is_pdf(resp):
        text = _extract_pdf_text(resp.content)
        title = urlparse(url).path.split("/")[-1].replace("-", " ").replace("_", " ")
        return {"title": title, "content": text, "is_pdf": True, "url": url, "raw_links": []}

    try:
        text, links = _extract_text_from_html(resp.text, url)
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else urlparse(url).path

    return {"title": title, "content": text, "is_pdf": False, "url": url, "raw_links": links}


def scan_source(source: dict, max_pages: int = 30) -> int:
    """
    Scan a source entry (from sources.json), follow links on the landing page,
    and ingest pension-relevant content. Returns count of documents added.
    """
    sess = _session()
    base_url = source["url"]
    source_id = source.get("id", "")
    source_name = source.get("name", "")
    country = source.get("country", "global")

    result = fetch_url(base_url, source_id=source_id, session=sess)
    if not result:
        return 0

    pages_to_process = [base_url]
    if not result["is_pdf"]:
        base_domain = urlparse(base_url).netloc
        candidate_links = [
            l for l in result["raw_links"][:500]
            if urlparse(l).netloc == base_domain and l != base_url
        ]
        # PDFs and publication/report/regulation links first — find real docs, not nav
        candidate_links.sort(key=_link_priority)
        pages_to_process = [base_url] + candidate_links

    pages_to_process = list(dict.fromkeys(pages_to_process))[:max_pages]

    added = 0
    for i, page_url in enumerate(pages_to_process):
        if i > 0:
            time.sleep(config.REQUEST_DELAY)
        page = result if (i == 0) else fetch_url(page_url, source_id=source_id, session=sess)
        if not page:
            continue
        if not _is_relevant(page["content"], page["title"]):
            continue

        doc_type = "research_paper"
        lower_title = page["title"].lower()
        lower_url = page_url.lower()
        if any(k in lower_url for k in (".pdf", "report", "publication")):
            doc_type = "annual_report" if "annual" in lower_title else "research_paper"
        if any(k in lower_title for k in ("regulation", "guideline", "circular", "directive")):
            doc_type = "regulation"
        if any(k in lower_title for k in ("bill", "draft", "amendment", "act")):
            doc_type = "draft_legislation" if "draft" in lower_title else "regulation"
        if any(k in lower_title for k in ("news", "press release", "media")):
            doc_type = "press_release"

        doc_id = kb.add_document(
            title=page["title"],
            content=page["content"],
            url=page["url"],
            doc_type=doc_type,
            jurisdiction=country,
            source_name=source_name,
        )
        if doc_id:
            added += 1

    kb.log_scan(source_id, base_url, "success", items_found=added)
    return added


def scan_all_sources(priority_filter: str | None = None) -> dict:
    """
    Scan all sources from sources.json.
    priority_filter: 'high', 'medium', or 'low' — if given, only that priority.
    Returns a summary dict.
    """
    sources_path = config.DATA_DIR / "sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))["sources"]

    if priority_filter:
        sources = [s for s in sources if s.get("priority") == priority_filter]

    results = {}
    for source in sources:
        print(f"  Scanning {source['name']}...")
        count = scan_source(source)
        results[source["id"]] = count
        print(f"    → {count} document(s) ingested")
        time.sleep(config.REQUEST_DELAY)

    return results


def ingest_url(url: str, jurisdiction: str = "global", doc_type: str = "", source_name: str = "") -> dict:
    """
    Fetch and ingest a single user-supplied URL into the knowledge base.
    Returns a result dict.
    """
    page = fetch_url(url)
    if not page:
        return {"success": False, "error": "Failed to fetch URL"}

    doc_id = kb.add_document(
        title=page["title"],
        content=page["content"],
        url=url,
        doc_type=doc_type or ("annual_report" if page["is_pdf"] else "research_paper"),
        jurisdiction=jurisdiction,
        source_name=source_name,
    )
    return {
        "success": True,
        "doc_id": doc_id,
        "title": page["title"],
        "is_pdf": page["is_pdf"],
        "content_length": len(page["content"]),
    }
