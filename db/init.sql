-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Document metadata table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    source_url TEXT NOT NULL,
    patch_version TEXT NOT NULL,
    title TEXT,
    raw_content TEXT NOT NULL,
    scraped_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(patch_version)
);

-- Chunks table with vector embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),  -- dimension for all-MiniLM-L6-v2
    created_at TIMESTAMP DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Index for document lookups
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_patch ON documents(patch_version);
