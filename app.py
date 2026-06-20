"""
Africa Pension Watch — Web Application
Run: python app.py
Then open: http://localhost:5464
"""

import asyncio
import concurrent.futures
import json
import os
import queue as sync_queue
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import config
from src import knowledge_base as kb, web_scanner, document_processor, report_generator
from src.agent import Agent

# ── Init ────────────────────────────────────────────────────────────────────

kb.initialize()

app = FastAPI(title="Africa Pension Watch Research Agent", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")

config.OUTPUTS_DIR.mkdir(exist_ok=True)

_agents: dict[int, Agent] = {}          # session_id → Agent instance
_scan_status = {"running": False, "last_result": None, "completed": 0, "total": 0, "current": ""}


def get_agent(session_id: int) -> Agent:
    """Return (or create) the Agent for a given session, with history pre-loaded."""
    if session_id not in _agents:
        agent = Agent()
        msgs = kb.get_messages(session_id)
        agent.history = [{"role": m["role"], "content": m["content"]} for m in msgs]
        _agents[session_id] = agent
    return _agents[session_id]


# ── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None

class RenameRequest(BaseModel):
    name: str

class IngestRequest(BaseModel):
    url: str
    jurisdiction: str = "global"
    doc_type: str = ""
    source_name: str = ""

class GenerateRequest(BaseModel):
    output_type: str
    topic: str
    countries: list[str] = []
    audience: str = "pension policymakers and practitioners"
    additional_context: str = ""
    use_deep_model: bool = False

class SearchRequest(BaseModel):
    query: str
    jurisdiction: str = ""
    doc_type: str = ""
    limit: int = 10

class ScanRequest(BaseModel):
    source_id: str = ""
    priority: str = "high"


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path("static/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found — place index.html in static/</h1>")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set. Add it to your .env file.")
    try:
        sid = req.session_id if req.session_id else kb.create_session()
        is_first = len(kb.get_messages(sid)) == 0
        agent = get_agent(sid)
        response = agent.chat(req.message)
        kb.add_message(sid, "user", req.message)
        kb.add_message(sid, "assistant", response)
        if is_first:
            auto_name = req.message[:50].strip().rstrip("?") + ("…" if len(req.message) > 50 else "")
            kb.rename_session(sid, auto_name)
        return {"response": response, "session_id": sid}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming endpoint. Yields status, text chunks, and done events."""
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set.")

    async def generate():
        try:
            if req.session_id and kb.session_exists(req.session_id):
                sid = req.session_id
            else:
                sid = kb.create_session()
            is_first = len(kb.get_messages(sid)) == 0
            agent = get_agent(sid)

            yield f"data: {json.dumps({'type': 'session_id', 'session_id': sid})}\n\n"

            loop = asyncio.get_running_loop()
            aqueue: asyncio.Queue = asyncio.Queue()

            def run_agent():
                try:
                    for chunk in agent.chat_streaming(req.message):
                        asyncio.run_coroutine_threadsafe(aqueue.put(chunk), loop)
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(
                        aqueue.put({"type": "error", "content": str(exc)}), loop
                    )
                finally:
                    asyncio.run_coroutine_threadsafe(aqueue.put(None), loop)

            loop.run_in_executor(None, run_agent)

            full_text: list[str] = []
            while True:
                chunk = await aqueue.get()
                if chunk is None:
                    break
                if chunk.get("type") == "text":
                    full_text.append(chunk["content"])
                yield f"data: {json.dumps(chunk)}\n\n"

            full_response = "".join(full_text)
            kb.add_message(sid, "user", req.message)
            kb.add_message(sid, "assistant", full_response)
            if is_first and req.message:
                auto_name = req.message[:50].strip().rstrip("?") + ("…" if len(req.message) > 50 else "")
                kb.rename_session(sid, auto_name)

            yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/status")
async def status():
    anthropic_ok = bool(config.ANTHROPIC_API_KEY)
    tavily_ok = bool(os.environ.get("TAVILY_API_KEY", ""))
    return {
        "anthropic": anthropic_ok,
        "tavily": tavily_ok,
        "model": config.MODEL,
        "deep_model": config.DEEP_MODEL,
        "kb_documents": kb.stats()["total_documents"],
    }


# ── Session endpoints ─────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": kb.list_sessions()}


@app.post("/api/sessions")
async def create_session():
    sid = kb.create_session()
    return {"session_id": sid, "name": "New Chat"}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: int):
    kb.delete_session(session_id)
    _agents.pop(session_id, None)
    return {"status": "ok"}


@app.patch("/api/sessions/{session_id}")
async def rename_session_ep(session_id: int, req: RenameRequest):
    kb.rename_session(session_id, req.name)
    return {"status": "ok"}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: int):
    return {"session_id": session_id, "messages": kb.get_messages(session_id)}


@app.post("/api/reset")
async def reset_chat():
    sid = kb.create_session()
    return {"status": "ok", "session_id": sid}


@app.get("/api/stats")
async def stats():
    return kb.stats()


@app.post("/api/search")
async def search(req: SearchRequest):
    results = kb.search(
        query=req.query,
        limit=req.limit,
        jurisdiction=req.jurisdiction,
        doc_type=req.doc_type,
    )
    return {"count": len(results), "results": results}


@app.get("/api/recent")
async def recent_docs(limit: int = 20, jurisdiction: str = "", doc_type: str = ""):
    docs = kb.list_recent(limit=limit, jurisdiction=jurisdiction, doc_type=doc_type)
    return {"count": len(docs), "documents": docs}


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    if not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    result = web_scanner.ingest_url(
        url=req.url,
        jurisdiction=req.jurisdiction,
        doc_type=req.doc_type,
        source_name=req.source_name,
    )
    if result["success"] and config.ANTHROPIC_API_KEY:
        enrichment = document_processor.summarize_document(result["doc_id"])
        result["summary"] = enrichment.get("summary", "")
        result["topics"] = enrichment.get("topics", [])
    return result


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set.")
    try:
        result = report_generator.generate(
            output_type=req.output_type,
            topic=req.topic,
            countries=req.countries,
            audience=req.audience,
            additional_context=req.additional_context,
            use_deep_model=req.use_deep_model,
        )
        return {
            "title": result["title"],
            "content": result["content"],
            "saved_to": Path(result["saved_to"]).name,
            "output_type": result["output_type"],
            "model_used": result["model_used"],
            "timestamp": result["timestamp"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate/stream")
async def generate_stream_ep(req: GenerateRequest):
    """SSE streaming endpoint for report generation — yields text chunks then a done event."""
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set.")

    async def gen():
        try:
            loop = asyncio.get_running_loop()
            aqueue: asyncio.Queue = asyncio.Queue()

            def run():
                try:
                    for chunk in report_generator.generate_streaming(
                        output_type=req.output_type,
                        topic=req.topic,
                        countries=req.countries,
                        audience=req.audience,
                        additional_context=req.additional_context,
                        use_deep_model=req.use_deep_model,
                    ):
                        asyncio.run_coroutine_threadsafe(aqueue.put(chunk), loop)
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(
                        aqueue.put({"type": "error", "content": str(exc)}), loop
                    )
                finally:
                    asyncio.run_coroutine_threadsafe(aqueue.put(None), loop)

            loop.run_in_executor(None, run)

            while True:
                chunk = await aqueue.get()
                if chunk is None:
                    break
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/output-types")
async def output_types():
    return {"output_types": report_generator.list_output_types()}


@app.get("/api/outputs")
async def list_outputs():
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    files = sorted(
        config.OUTPUTS_DIR.glob("*.md"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return {
        "outputs": [
            {
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "created": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
            for f in files[:50]
        ]
    }


@app.get("/api/outputs/{filename}/content")
async def get_output_content(filename: str):
    path = config.OUTPUTS_DIR / filename
    if not path.exists() or path.suffix != ".md":
        raise HTTPException(status_code=404, detail="File not found")
    return {"filename": filename, "content": path.read_text(encoding="utf-8")}


@app.get("/api/outputs/{filename}")
async def download_output(filename: str):
    path = config.OUTPUTS_DIR / filename
    if not path.exists() or not path.suffix == ".md":
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/markdown", filename=filename)


@app.get("/api/jurisdictions")
async def jurisdictions(region: str = ""):
    jdata_path = config.DATA_DIR / "jurisdictions.json"
    if not jdata_path.exists():
        return {"count": 0, "jurisdictions": []}
    data = json.loads(jdata_path.read_text(encoding="utf-8"))
    jurs = data.get("jurisdictions", [])
    if region:
        jurs = [j for j in jurs if j.get("region", "").lower() == region.lower()]
    summary = [
        {
            "country": j["country"],
            "region": j.get("region", ""),
            "regulator": j.get("regulator", ""),
            "system_type": j.get("system_type", ""),
            "estimated_aum_usd_bn": j.get("estimated_aum_usd_bn"),
            "coverage_rate_pct": j.get("coverage_rate_pct"),
            "offshore_limit_pct": j.get("offshore_limit_pct"),
            "primary_legislation": j.get("primary_legislation", ""),
            "legislation_year": j.get("legislation_year"),
        }
        for j in jurs
    ]
    return {"count": len(summary), "jurisdictions": summary}


@app.get("/api/jurisdictions/{country}")
async def jurisdiction_profile(country: str):
    jdata_path = config.DATA_DIR / "jurisdictions.json"
    if not jdata_path.exists():
        raise HTTPException(status_code=404, detail="Jurisdiction data not found")
    data = json.loads(jdata_path.read_text(encoding="utf-8"))
    for j in data.get("jurisdictions", []):
        if j["country"].lower() == country.lower():
            return j
    raise HTTPException(status_code=404, detail=f"No profile for '{country}'")


@app.get("/api/sources")
async def sources(source_type: str = "", country: str = ""):
    sources_path = config.DATA_DIR / "sources.json"
    if not sources_path.exists():
        return {"count": 0, "sources": []}
    data = json.loads(sources_path.read_text(encoding="utf-8"))
    srcs = data.get("sources", [])
    if source_type:
        srcs = [s for s in srcs if s.get("type", "") == source_type]
    if country:
        srcs = [s for s in srcs if s.get("country", "").lower() == country.lower()]
    return {"count": len(srcs), "sources": srcs}


def _do_scan(source_id: str, priority: str):
    global _scan_status
    _scan_status["running"] = True
    try:
        sources_path = config.DATA_DIR / "sources.json"
        all_sources = json.loads(sources_path.read_text(encoding="utf-8"))["sources"]

        if source_id:
            source = next((s for s in all_sources if s["id"] == source_id), None)
            if source:
                count = web_scanner.scan_source(source)
                _scan_status["last_result"] = {
                    "sources_scanned": 1,
                    "documents_found": count,
                    "completed_at": datetime.now().isoformat(),
                }
        else:
            filtered = [s for s in all_sources if not priority or s.get("priority") == priority]
            results = {}
            _scan_status["total"] = len(filtered)
            _scan_status["completed"] = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(web_scanner.scan_source, src): src for src in filtered}
                for fut in concurrent.futures.as_completed(futures):
                    src = futures[fut]
                    results[src["id"]] = fut.result() if not fut.exception() else 0
                    _scan_status["completed"] += 1
                    _scan_status["current"] = src["name"]
            total = sum(results.values())
            if config.ANTHROPIC_API_KEY:
                document_processor.process_unsummarized(max_docs=30)
            _scan_status["last_result"] = {
                "sources_scanned": len(results),
                "documents_found": total,
                "completed_at": datetime.now().isoformat(),
            }
    finally:
        _scan_status["running"] = False


@app.post("/api/scan")
async def scan(req: ScanRequest, background_tasks: BackgroundTasks):
    if _scan_status["running"]:
        return {"status": "already_running", "message": "A scan is already in progress."}
    background_tasks.add_task(_do_scan, req.source_id, req.priority)
    return {"status": "started", "message": "Scan started in background."}


@app.get("/api/scan/status")
async def scan_status():
    return {
        "running": _scan_status["running"],
        "last_result": _scan_status["last_result"],
        "completed": _scan_status["completed"],
        "total": _scan_status["total"],
        "current": _scan_status["current"],
    }


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5464))
    print("\n  Africa Pension Watch Research Agent")
    print("  " + "-" * 37)
    print(f"  Open: http://localhost:{port}\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
