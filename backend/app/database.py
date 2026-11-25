"""SQLite persistence layer for chat sessions, messages, and reply metadata."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterator


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _normalize_sqlite_path(database_url: str) -> str:
    """Extract a filesystem path (or :memory:) from a DATABASE_URL string."""
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///") :]
    if database_url.startswith("sqlite://"):
        return database_url[len("sqlite://") :]
    return database_url


class Database:
    """SQLite store for multi-turn session history and assistant reply metadata."""

    def __init__(self, database_url: str) -> None:
        self.db_path = _normalize_sqlite_path(database_url)
        self._is_memory = self.db_path == ":memory:"
        self._memory_conn: sqlite3.Connection | None = None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured SQLite connection inside a transaction."""
        if self._is_memory:
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_conn.row_factory = sqlite3.Row
            conn = self._memory_conn
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create required SQLite tables and indexes if they do not exist."""
        with self._connect() as conn:
            if not self._is_memory:
                conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    system_prompt TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                    ON messages (session_id, id);
                """
            )

    def check_health(self) -> bool:
        """Return True when SQLite responds to a lightweight query."""
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def get_session_system_prompt(self, session_id: str) -> str | None:
        """Return the stored system prompt override for a session, if any."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return row["system_prompt"]

    def get_history_for_llm(
        self, session_id: str, limit: int | None = None
    ) -> list[dict[str, str]]:
        """Return stored user/assistant messages for a session in chronological order."""
        with self._connect() as conn:
            if limit is not None:
                rows = conn.execute(
                    """
                    SELECT role, content
                    FROM (
                        SELECT id, role, content
                        FROM messages
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()

        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def save_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        metadata: dict,
        system_prompt: str | None = None,
    ) -> None:
        """Atomically persist both user and assistant messages after a successful reply."""
        now = _utc_now()
        metadata_json = json.dumps(metadata)

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT session_id, system_prompt FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, system_prompt, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, system_prompt, now, now),
                )
            else:
                effective_prompt = (
                    system_prompt if system_prompt is not None else existing["system_prompt"]
                )
                conn.execute(
                    """
                    UPDATE sessions
                    SET system_prompt = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (effective_prompt, now, session_id),
                )

            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'user', ?, NULL, ?)
                """,
                (session_id, user_content, now),
            )
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'assistant', ?, ?, ?)
                """,
                (session_id, assistant_content, metadata_json, now),
            )

    def get_session_messages(self, session_id: str) -> list[dict] | None:
        """Return ordered history for a session without system prompt, or None if missing."""
        with self._connect() as conn:
            session_row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            rows = conn.execute(
                """
                SELECT role, content, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        if session_row is None and not rows:
            return None

        messages: list[dict] = []
        for row in rows:
            item: dict = {
                "role": row["role"],
                "content": row["content"],
            }
            if row["role"] == "assistant" and row["metadata"]:
                item["metadata"] = json.loads(row["metadata"])
            messages.append(item)
        return messages
