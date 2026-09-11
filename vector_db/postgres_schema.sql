CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
	document_id BIGSERIAL PRIMARY KEY,
	source_path TEXT NOT NULL UNIQUE,
	filename TEXT NOT NULL,
	title TEXT,
	authors TEXT,
	publication_year INTEGER,
	page_count INTEGER NOT NULL DEFAULT 0,
	source_dataset TEXT,
	metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
	content_sha256 CHAR(64) NOT NULL,
	created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
	chunk_id BIGSERIAL PRIMARY KEY,
	document_id BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
	chunk_number INTEGER NOT NULL,
	page_start INTEGER,
	page_end INTEGER,
	content TEXT NOT NULL,
	metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
	embedding vector(384),
	created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
	UNIQUE(document_id, chunk_number)
);

CREATE INDEX IF NOT EXISTS documents_year_idx ON documents(publication_year);
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING GIN(metadata);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING GIN(metadata);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
	ON chunks USING hnsw (embedding vector_cosine_ops)
	WHERE embedding IS NOT NULL;