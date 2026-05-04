"""Embedding model wrapper using sentence-transformers."""

import logging

from src.config import config

logger = logging.getLogger(__name__)

_model = None


def get_embedding_model():
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {config.model.embedding_model}")
        _model = SentenceTransformer(config.model.embedding_model)
        logger.info(f"Embedding dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = False):
    """Embed a list of texts and return numpy array of shape (n, dim)."""
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # For cosine similarity
    )
    return embeddings


def embed_query(query: str):
    """Embed a single query string."""
    return embed_texts([query])[0]
