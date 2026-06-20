"""
SQLite knowledge base with full-text search for pension research documents.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import config


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize():
    """Create all tables if they don't exist."""
    conn = _connect()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                url         TEXT UNIQUE,
                content     TEXT,
                summary     TEXT,
                doc_type    TEXT,
                jurisdiction TEXT,
                topics      TEXT,
                source_name TEXT,
                author      TEXT,
                date_published TEXT,
                date_added  TEXT DEFAULT (datetime('now')),
                language    TEXT DEFAULT 'en',
                is_summarized INTEGER DEFAULT 0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title, summary, content, jurisdiction, topics,
                content=documents,
                content_rowid=id
            );

            CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, title, summary, content, jurisdiction, topics)
                VALUES (new.id, new.title, new.summary, new.content, new.jurisdiction, new.topics);
            END;

            CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, summary, content, jurisdiction, topics)
                VALUES ('delete', old.id, old.title, old.summary, old.content, old.jurisdiction, old.topics);
                INSERT INTO documents_fts(rowid, title, summary, content, jurisdiction, topics)
                VALUES (new.id, new.title, new.summary, new.content, new.jurisdiction, new.topics);
            END;

            CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, summary, content, jurisdiction, topics)
                VALUES ('delete', old.id, old.title, old.summary, old.content, old.jurisdiction, old.topics);
            END;

            CREATE TABLE IF NOT EXISTS scan_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id   TEXT,
                url         TEXT,
                status      TEXT,
                items_found INTEGER DEFAULT 0,
                scanned_at  TEXT DEFAULT (datetime('now')),
                error_msg   TEXT
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL DEFAULT 'New Chat',
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            );
        """)
        conn.commit()
    finally:
        conn.close()


def add_document(
    title: str,
    content: str = "",
    url: str = "",
    summary: str = "",
    doc_type: str = "research_paper",
    jurisdiction: str = "global",
    topics: list[str] | None = None,
    source_name: str = "",
    author: str = "",
    date_published: str = "",
) -> int:
    """
    Insert a document into the knowledge base.
    Returns the new document id, or the existing id if URL already exists.
    """
    topics_str = json.dumps(topics or [])
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO documents
                (title, url, content, summary, doc_type, jurisdiction, topics,
                 source_name, author, date_published, is_summarized)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(url) DO UPDATE SET
                title        = excluded.title,
                summary      = excluded.summary,
                doc_type     = excluded.doc_type,
                jurisdiction = excluded.jurisdiction,
                topics       = excluded.topics,
                source_name  = excluded.source_name
            RETURNING id
            """,
            (
                title, url or None, content[:config.MAX_CONTENT_LENGTH],
                summary, doc_type, jurisdiction, topics_str,
                source_name, author, date_published,
                1 if summary else 0,
            ),
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
        return doc_id
    finally:
        conn.close()


def search(query: str, limit: int = 10, jurisdiction: str = "", doc_type: str = "") -> list[dict]:
    """Full-text search across the knowledge base."""
    conn = _connect()
    try:
        where_clauses = ["documents_fts MATCH ?"]
        params: list = [query]

        if jurisdiction:
            where_clauses.append("d.jurisdiction = ?")
            params.append(jurisdiction)
        if doc_type:
            where_clauses.append("d.doc_type = ?")
            params.append(doc_type)

        params.append(limit)
        sql = f"""
            SELECT d.id, d.title, d.url, d.summary, d.doc_type,
                   d.jurisdiction, d.topics, d.source_name, d.date_published,
                   d.date_added, rank
            FROM documents_fts
            JOIN documents d ON documents_fts.rowid = d.id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY rank
            LIMIT ?
        """
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_document(doc_id: int) -> Optional[dict]:
    """Retrieve a single document by ID including full content."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_recent(limit: int = 20, jurisdiction: str = "", doc_type: str = "") -> list[dict]:
    """Return the most recently added documents."""
    conn = _connect()
    try:
        where = []
        params: list = []
        if jurisdiction:
            where.append("jurisdiction = ?")
            params.append(jurisdiction)
        if doc_type:
            where.append("doc_type = ?")
            params.append(doc_type)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT id, title, url, summary, doc_type, jurisdiction,
                   topics, source_name, date_published, date_added
            FROM documents {clause}
            ORDER BY date_added DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_by_topic(topic: str, limit: int = 20) -> list[dict]:
    """Return documents tagged with a specific topic."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, title, url, summary, doc_type, jurisdiction,
                   topics, source_name, date_published, date_added
            FROM documents
            WHERE topics LIKE ?
            ORDER BY date_added DESC
            LIMIT ?
            """,
            (f'%{topic}%', limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def log_scan(source_id: str, url: str, status: str, items_found: int = 0, error_msg: str = ""):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO scan_log (source_id, url, status, items_found, error_msg) VALUES (?,?,?,?,?)",
            (source_id, url, status, items_found, error_msg),
        )
        conn.commit()
    finally:
        conn.close()


def stats() -> dict:
    """Return summary statistics about the knowledge base."""
    conn = _connect()
    try:
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        by_type = conn.execute(
            "SELECT doc_type, COUNT(*) as n FROM documents GROUP BY doc_type ORDER BY n DESC"
        ).fetchall()
        by_jurisdiction = conn.execute(
            "SELECT jurisdiction, COUNT(*) as n FROM documents GROUP BY jurisdiction ORDER BY n DESC LIMIT 15"
        ).fetchall()
        recent_scan = conn.execute(
            "SELECT scanned_at FROM scan_log ORDER BY scanned_at DESC LIMIT 1"
        ).fetchone()
        return {
            "total_documents": total,
            "by_type": [dict(r) for r in by_type],
            "by_jurisdiction": [dict(r) for r in by_jurisdiction],
            "last_scan": recent_scan[0] if recent_scan else "Never",
        }
    finally:
        conn.close()


# ── Chat session management ───────────────────────────────────────────────────

def session_exists(session_id: int) -> bool:
    conn = _connect()
    try:
        row = conn.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        return row is not None
    finally:
        conn.close()


def create_session(name: str = "New Chat") -> int:
    conn = _connect()
    try:
        cur = conn.execute("INSERT INTO chat_sessions (name) VALUES (?)", (name,))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_sessions() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute("""
            SELECT s.id, s.name, s.created_at, s.updated_at,
                   COUNT(m.id) as message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            LIMIT 40
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_session(session_id: int):
    conn = _connect()
    try:
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


def rename_session(session_id: int, name: str):
    conn = _connect()
    try:
        conn.execute(
            "UPDATE chat_sessions SET name = ?, updated_at = datetime('now') WHERE id = ?",
            (name, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def add_message(session_id: int, role: str, content: str):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?,?,?)",
            (session_id, role, content),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = datetime('now') WHERE id = ?",
            (session_id,),
        )
        conn.commit()
    finally:
        conn.close()


def get_messages(session_id: int) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM chat_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
