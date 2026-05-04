"""Ingestion pipeline: scrape → chunk → embed → store in pgvector."""

import gc
import json
import logging
from pathlib import Path

from tqdm import tqdm

from src.scraper import scrape_all, PatchNote, RAW_DATA_DIR
from src.chunking import chunk_text
from src.database import insert_document, insert_chunks

logger = logging.getLogger(__name__)


def ingest_patch_note(patch: PatchNote) -> dict:
    """Ingest a single patch note: chunk, embed, and store."""
    from src.embeddings import embed_texts

    # 1. Store the raw document
    doc_id = insert_document(
        source_url=patch.url,
        patch_version=patch.patch_version,
        title=patch.title,
        raw_content=patch.content,
    )

    # 2. Chunk the content
    chunks = chunk_text(patch.content)
    if not chunks:
        logger.warning(f"No chunks generated for patch {patch.patch_version}")
        return {"document_id": doc_id, "chunks": 0}

    # 3. Embed all chunks in batch
    texts = [c.content for c in chunks]
    embeddings = embed_texts(texts, show_progress=False)

    # 4. Store chunks with embeddings
    chunk_records = [
        {
            "chunk_index": c.chunk_index,
            "content": c.content,
            "embedding": embeddings[i].tolist(),
        }
        for i, c in enumerate(chunks)
    ]
    insert_chunks(doc_id, chunk_records)

    logger.info(
        f"Ingested patch {patch.patch_version}: "
        f"{len(chunks)} chunks, {len(patch.content)} chars"
    )
    return {"document_id": doc_id, "chunks": len(chunks)}


def run_ingestion(max_patches: int | None = None, delay: float = 2.0):
    """
    Full ingestion pipeline.

    1. Scrape patch notes from the web (with caching to disk)
    2. Free scraper memory
    3. Load patches one-by-one from cache, chunk, embed, store
    """
    logger.info("Starting ingestion pipeline")

    # Phase 1: Scrape (saves to disk cache)
    patches = scrape_all(max_patches=max_patches, delay=delay)
    if not patches:
        logger.error("No patch notes scraped. Check your network connection.")
        return

    # Save just the metadata we need, then free the heavy content from memory
    patch_versions = [p.patch_version for p in patches]
    logger.info(f"Scraped {len(patch_versions)} patch notes, freeing scraper memory...")
    del patches
    gc.collect()

    # Phase 2: Ingest from disk cache one at a time
    total_chunks = 0
    for version in tqdm(patch_versions, desc="Ingesting"):
        cache_path = RAW_DATA_DIR / f"patch_{version.replace('.', '_')}.json"
        if not cache_path.exists():
            logger.warning(f"Cache file not found for patch {version}, skipping")
            continue

        with open(cache_path) as f:
            patch_data = json.load(f)
        patch = PatchNote(**patch_data)

        result = ingest_patch_note(patch)
        total_chunks += result["chunks"]

        # Free memory between patches
        del patch, patch_data
        gc.collect()

    logger.info(
        f"Ingestion complete: {len(patch_versions)} documents, {total_chunks} total chunks"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion(max_patches=5)
