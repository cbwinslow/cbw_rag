-- CBW RAG Schema v2 - Simple, current-only

-- Files: rich metadata + current state
CREATE TABLE files (
    id BIGSERIAL PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_extension TEXT,
    file_size BIGINT,
    file_mode INTEGER,
    file_owner TEXT,
    file_group TEXT,
    mime_type TEXT,
    file_category TEXT,
    detected_language TEXT,
    encoding TEXT,
    line_count INTEGER,
    file_content_hash TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    git_repo_root TEXT,
    git_branch TEXT,
    git_last_commit TEXT,
    git_status TEXT,
    file_created_at TIMESTAMPTZ,
    file_modified_at TIMESTAMPTZ,
    index_status TEXT DEFAULT 'pending',
    indexed_at TIMESTAMPTZ DEFAULT NOW(),
    tags TEXT[] DEFAULT '{}',
    extra JSONB DEFAULT '{}'
);

-- Chunks: text content split into pieces
CREATE TABLE document_chunks (
    id BIGSERIAL PRIMARY KEY,
    file_id BIGINT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    content_fts tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    content_fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    content_hash TEXT NOT NULL,
    token_count INTEGER,
    char_count INTEGER,
    start_line INTEGER,
    end_line INTEGER,
    UNIQUE(file_id, chunk_index)
);

-- Embeddings: 768-dim vectors (nomic-embed-text)
CREATE TABLE embeddings (
    id BIGSERIAL PRIMARY KEY,
    chunk_id BIGINT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    file_id BIGINT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(chunk_id, model_name)
);

-- Embeddings large: 4096-dim vectors (qwen3-embedding)
CREATE TABLE embeddings_large (
    id BIGSERIAL PRIMARY KEY,
    chunk_id BIGINT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    file_id BIGINT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    embedding vector(4096) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(chunk_id, model_name)
);

-- Indexes
CREATE INDEX idx_files_path ON files(source_path);
CREATE INDEX idx_files_category ON files(file_category);
CREATE INDEX idx_files_language ON files(detected_language);
CREATE INDEX idx_files_hash ON files(file_content_hash);
CREATE INDEX idx_files_status ON files(index_status);
CREATE INDEX idx_files_tags ON files USING gin(tags);
CREATE INDEX idx_files_extra ON files USING gin(extra);
CREATE INDEX idx_chunks_file ON document_chunks(file_id);
CREATE INDEX idx_chunks_fts ON document_chunks USING gin(to_tsvector('english', content));
CREATE INDEX idx_emb_file ON embeddings(file_id);
CREATE INDEX idx_emb_model ON embeddings(model_name);
CREATE INDEX idx_emb_hnsw ON embeddings USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX idx_emb_large_file ON embeddings_large(file_id);

-- Search view
CREATE VIEW v_searchable AS
SELECT
    dc.id AS chunk_id, dc.file_id,
    f.source_path, f.file_name, f.file_extension, f.file_category, f.detected_language,
    dc.content, dc.chunk_index, dc.total_chunks, dc.start_line, dc.end_line,
    e.embedding AS embedding_768
FROM document_chunks dc
JOIN files f ON f.id = dc.file_id
LEFT JOIN embeddings e ON e.chunk_id = dc.id AND e.model_name = 'nomic-embed-text';

GRANT ALL ON ALL TABLES IN SCHEMA public TO cbwinslow;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO cbwinslow;
