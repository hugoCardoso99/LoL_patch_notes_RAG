"""RAG query engine: retrieve relevant chunks and generate answers."""

import logging
import time

from pydantic import BaseModel
from src.embeddings import embed_query
from src.database import search_similar
from src.llm import generate_answer, LLMResponse
from src.config import config

logger = logging.getLogger(__name__)


class RAGResult(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    source_patches: list[str]
    retrieval_scores: list[float]
    retrieval_latency_seconds: float
    generation_latency_seconds: float
    total_latency_seconds: float
    prompt_tokens: int
    completion_tokens: int


def query(
    question: str,
    top_k: int | None = None,
    patch_filter: str | None = None,
) -> RAGResult:
    """
    Full RAG pipeline: embed query → retrieve chunks → generate answer.

    Args:
        question: Natural language question about LoL patches.
        top_k: Number of chunks to retrieve (default from config).
        patch_filter: Optional patch version to restrict search to.

    Returns:
        RAGResult with answer, contexts, and performance metrics.
    """
    top_k = top_k or config.rag.top_k
    total_start = time.time()

    # 1. Retrieve
    retrieval_start = time.time()
    query_embedding = embed_query(question)
    chunks = search_similar(
        query_embedding=query_embedding.tolist(),
        top_k=top_k,
        patch_filter=patch_filter,
    )
    retrieval_latency = time.time() - retrieval_start

    if not chunks:
        return RAGResult(
            question=question,
            answer="No relevant patch notes found in the database. Make sure you've run the ingestion pipeline first.",
            contexts=[],
            source_patches=[],
            retrieval_scores=[],
            retrieval_latency_seconds=retrieval_latency,
            generation_latency_seconds=0,
            total_latency_seconds=time.time() - total_start,
            prompt_tokens=0,
            completion_tokens=0,
        )

    logger.info(
        f"Retrieved {len(chunks)} chunks "
        f"(best similarity: {chunks[0]['similarity']:.4f})"
    )

    # 2. Generate
    llm_response: LLMResponse = generate_answer(question, chunks)

    total_latency = time.time() - total_start

    return RAGResult(
        question=question,
        answer=llm_response.answer,
        contexts=[c["content"] for c in chunks],
        source_patches=list({c["patch_version"] for c in chunks}),
        retrieval_scores=[float(c["similarity"]) for c in chunks],
        retrieval_latency_seconds=retrieval_latency,
        generation_latency_seconds=llm_response.latency_seconds,
        total_latency_seconds=total_latency,
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
    )


def query_batch(questions: list[str], **kwargs) -> list[RAGResult]:
    """Run multiple queries and return results."""
    results = []
    for q in questions:
        logger.info(f"Processing: {q}")
        results.append(query(q, **kwargs))
    return results
