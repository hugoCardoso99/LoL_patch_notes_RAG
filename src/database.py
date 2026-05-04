"""Database connection and vector operations using pgvector."""

from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from src.config import config


def get_connection():
    """Create a new database connection with pgvector support."""
    conn = psycopg2.connect(
        host=config.db.host,
        port=config.db.port,
        user=config.db.user,
        password=config.db.password,
        dbname=config.db.database,
    )
    register_vector(conn)
    return conn


@contextmanager
def get_cursor(commit: bool = True):
    """Context manager for database cursor with automatic commit/rollback."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def insert_document(source_url: str, patch_version: str, title: str, raw_content: str) -> int:
    """Insert a document and return its ID. Skips if patch_version already exists."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (source_url, patch_version, title, raw_content)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (patch_version) DO UPDATE SET raw_content = EXCLUDED.raw_content
            RETURNING id
            """,
            (source_url, patch_version, title, raw_content),
        )
        return cur.fetchone()["id"]


def insert_chunks(document_id: int, chunks: list[dict]):
    """Bulk insert chunks with embeddings for a document."""
    with get_cursor() as cur:
        # Clear existing chunks for this document (in case of re-ingestion)
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        # Insert new chunks
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks (document_id, chunk_index, content, embedding)
            VALUES %s
            """,
            [
                (document_id, c["chunk_index"], c["content"], c["embedding"])
                for c in chunks
            ],
            template="(%s, %s, %s, %s::vector)",
        )


def search_similar(query_embedding, top_k: int = 5, patch_filter: Optional[str] = None) -> list[dict]:
    """Find the top-k most similar chunks to the query embedding."""
    with get_cursor(commit=False) as cur:
        if patch_filter:
            cur.execute(
                """
                SELECT c.id, c.content, c.chunk_index,
                       d.patch_version, d.title,
                       1 - (c.embedding <=> %s::vector) AS similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.patch_version = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, patch_filter, query_embedding, top_k),
            )
        else:
            cur.execute(
                """
                SELECT c.id, c.content, c.chunk_index,
                       d.patch_version, d.title,
                       1 - (c.embedding <=> %s::vector) AS similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
        return [dict(row) for row in cur.fetchall()]


def get_all_documents() -> list[dict]:
    """List all ingested documents."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, patch_version, title, scraped_at FROM documents ORDER BY patch_version DESC"
        )
        return [dict(row) for row in cur.fetchall()]


def get_document_chunk_count(document_id: int) -> int:
    """Get the number of chunks for a document."""
    with get_cursor(commit=False) as cur:
        cur.execute("SELECT COUNT(*) as cnt FROM chunks WHERE document_id = %s", (document_id,))
        return cur.fetchone()["cnt"]
