"""Text chunking strategies for document ingestion."""

import re
import logging

from pydantic import BaseModel
from src.config import config

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    content: str
    chunk_index: int
    metadata: dict | None = None


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks using a section-aware strategy.

    The chunker respects document structure (headers/sections) and avoids
    splitting mid-sentence when possible.
    """
    chunk_size = chunk_size or config.rag.chunk_size
    chunk_overlap = chunk_overlap or config.rag.chunk_overlap

    # Split by sections first (double newlines or headers)
    sections = re.split(r"\n(?=##)", text)

    chunks = []
    current_chunk = ""
    current_section_header = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Track section headers for context
        header_match = re.match(r"^(#{2,4}\s+.+)", section)
        if header_match:
            current_section_header = header_match.group(1).strip()

        # If section fits in a chunk, try to append
        if len(current_chunk) + len(section) + 1 <= chunk_size:
            current_chunk = f"{current_chunk}\n{section}".strip()
        else:
            # Save current chunk if non-empty
            if current_chunk:
                chunks.append(current_chunk)

            # If section itself is too large, split it further
            if len(section) > chunk_size:
                sub_chunks = _split_long_section(
                    section, chunk_size, chunk_overlap, current_section_header
                )
                chunks.extend(sub_chunks[:-1])  # Add all but last
                current_chunk = sub_chunks[-1] if sub_chunks else ""
            else:
                current_chunk = section

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Apply overlap between consecutive chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, chunk_overlap)

    # Convert to Chunk objects
    return [
        Chunk(content=c, chunk_index=i)
        for i, c in enumerate(chunks)
        if len(c.strip()) > 20  # Skip tiny fragments
    ]


def _split_long_section(
    text: str, chunk_size: int, overlap: int, header: str = ""
) -> list[str]:
    """Split a long section into chunks, trying to break at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sub_chunks = []
    current = header + "\n" if header else ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > chunk_size and current.strip():
            sub_chunks.append(current.strip())
            # Keep header context for continuity
            current = header + "\n" if header else ""
        current += sentence + " "

    if current.strip() and current.strip() != header:
        sub_chunks.append(current.strip())

    return sub_chunks if sub_chunks else [text[:chunk_size]]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlap from the end of each chunk to the start of the next."""
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        # Find a word boundary in the overlap region
        space_idx = prev_tail.find(" ")
        if space_idx > 0:
            prev_tail = prev_tail[space_idx + 1:]
        overlapped.append(f"...{prev_tail}\n{chunks[i]}")
    return overlapped
