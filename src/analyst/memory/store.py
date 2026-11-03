"""The agent's memory, kept in the app's SQLite database (state/app.db) next to the catalog.

Only the user writes memory (UI buttons, API calls); the agent reads it. That keeps a wrong
answer from teaching itself.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

FactKind = Literal["glossary", "correction"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verified_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    question TEXT NOT NULL,
    sql TEXT NOT NULL,
    answer TEXT,
    created_at TEXT NOT NULL,
    stale_reason TEXT
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    term TEXT,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversation_messages (
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (conversation_id, ordinal)
);
"""


class VerifiedQuery(BaseModel):
    id: int
    source: str
    question: str
    sql: str
    answer: str | None = None
    created_at: dt.datetime
    stale_reason: str | None = None  # set when the SQL stopped running after a schema change


class Fact(BaseModel):
    id: int
    source: str
    kind: FactKind
    term: str | None = None  # glossary only
    text: str
    created_at: dt.datetime


class Conversation(BaseModel):
    id: int
    source: str
    title: str
    started_at: dt.datetime
    updated_at: dt.datetime


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            with con:  # commit on success, roll back on error
                yield con
        finally:
            con.close()

    # ---- verified queries ----------------------------------------------------------------
    def add_verified(self, source: str, question: str, sql: str, answer: str | None) -> int:
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO verified_queries (source, question, sql, answer, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (source, question.strip(), sql.strip(), answer, _now()),
            )
            return int(cur.lastrowid)

    def verified(self, source: str, *, include_stale: bool = False) -> list[VerifiedQuery]:
        sql = "SELECT * FROM verified_queries WHERE source = ?"
        if not include_stale:
            sql += " AND stale_reason IS NULL"
        with self._connect() as con:
            rows = con.execute(sql + " ORDER BY id", (source,)).fetchall()
        return [VerifiedQuery(**dict(r)) for r in rows]

    def set_stale(self, query_id: int, reason: str | None) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE verified_queries SET stale_reason = ? WHERE id = ?", (reason, query_id)
            )

    def delete_verified(self, query_id: int) -> None:
        self._delete("verified_queries", query_id)

    # ---- glossary and corrections --------------------------------------------------------
    def add_fact(self, source: str, kind: FactKind, text: str, term: str | None = None) -> int:
        if kind == "glossary" and not (term and term.strip()):
            raise ValueError("a glossary entry needs a term")
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO facts (source, kind, term, text, created_at) VALUES (?, ?, ?, ?, ?)",
                (source, kind, term.strip() if term else None, text.strip(), _now()),
            )
            return int(cur.lastrowid)

    def facts(self, source: str, kind: FactKind | None = None) -> list[Fact]:
        sql, args = "SELECT * FROM facts WHERE source = ?", [source]
        if kind:
            sql, args = sql + " AND kind = ?", [*args, kind]
        with self._connect() as con:
            rows = con.execute(sql + " ORDER BY id", args).fetchall()
        return [Fact(**dict(r)) for r in rows]

    def delete_fact(self, fact_id: int) -> None:
        self._delete("facts", fact_id)

    # ---- conversations -------------------------------------------------------------------
    def start_conversation(self, source: str, title: str) -> int:
        now = _now()
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO conversations (source, title, started_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (source, title.strip()[:80] or "Untitled", now, now),
            )
            return int(cur.lastrowid)

    def add_message(self, conversation_id: int, message: dict[str, Any]) -> None:
        with self._connect() as con:
            (count,) = con.execute(
                "SELECT COUNT(*) FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            con.execute(
                "INSERT INTO conversation_messages VALUES (?, ?, ?)",
                (conversation_id, count, json.dumps(message, default=str)),
            )
            con.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id)
            )

    def conversations(self, source: str, limit: int = 20) -> list[Conversation]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM conversations WHERE source = ? ORDER BY updated_at DESC LIMIT ?",
                (source, limit),
            ).fetchall()
        return [Conversation(**dict(r)) for r in rows]

    def messages(self, conversation_id: int) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT payload FROM conversation_messages WHERE conversation_id = ? "
                "ORDER BY ordinal",
                (conversation_id,),
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def delete_conversation(self, conversation_id: int) -> None:
        self._delete("conversations", conversation_id)

    def _delete(self, table: str, row_id: int) -> None:
        with self._connect() as con:
            if con.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,)).rowcount == 0:
                raise KeyError(f"{table} {row_id}")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
