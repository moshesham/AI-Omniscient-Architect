-- Omniscient Architect Database Initialization Script
-- This script sets up the database schema and required extensions

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create RAG schema
CREATE SCHEMA IF NOT EXISTS rag;

-- Set search path
SET search_path TO rag, public;

-- Grant permissions
GRANT ALL ON SCHEMA rag TO omniscient;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA rag TO omniscient;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA rag TO omniscient;
ALTER DEFAULT PRIVILEGES IN SCHEMA rag GRANT ALL ON TABLES TO omniscient;
ALTER DEFAULT PRIVILEGES IN SCHEMA rag GRANT ALL ON SEQUENCES TO omniscient;

-- Create documents table
CREATE TABLE IF NOT EXISTS rag.documents (
    id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create chunks table with vector embedding and tsvector for full-text search
CREATE TABLE IF NOT EXISTS rag.chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES rag.documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768),  -- 768 dimensions for nomic-embed-text
    metadata JSONB DEFAULT '{}',
    start_char INTEGER DEFAULT 0,
    end_char INTEGER DEFAULT 0,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for chunks
CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
ON rag.chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx 
ON rag.chunks 
USING GIN (tsv);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx 
ON rag.chunks (document_id);

-- Create knowledge questions table
CREATE TABLE IF NOT EXISTS rag.knowledge_questions (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES rag.documents(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    topic TEXT DEFAULT 'general',
    difficulty TEXT DEFAULT 'medium',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create knowledge scores table
CREATE TABLE IF NOT EXISTS rag.knowledge_scores (
    id UUID PRIMARY KEY,
    retrieval_precision FLOAT NOT NULL,
    answer_accuracy FLOAT NOT NULL,
    coverage_ratio FLOAT NOT NULL,
    questions_evaluated INTEGER DEFAULT 0,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Learning System Tables

-- Learned facts table
CREATE TABLE IF NOT EXISTS rag.learned_facts (
    id UUID PRIMARY KEY,
    topic TEXT NOT NULL,
    fact TEXT NOT NULL,
    source_question TEXT,
    source_answer TEXT,
    confidence REAL DEFAULT 0.5,
    usage_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Reasoning chains table
CREATE TABLE IF NOT EXISTS rag.reasoning_chains (
    id UUID PRIMARY KEY,
    question TEXT NOT NULL,
    steps JSONB NOT NULL,
    final_answer TEXT NOT NULL,
    was_correct BOOLEAN DEFAULT FALSE,
    feedback_score REAL DEFAULT 0.0,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Query refinements table
CREATE TABLE IF NOT EXISTS rag.query_refinements (
    id UUID PRIMARY KEY,
    original_query TEXT NOT NULL,
    refined_query TEXT NOT NULL,
    improvement_score REAL DEFAULT 0.0,
    times_applied INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- User feedback table
CREATE TABLE IF NOT EXISTS rag.user_feedback (
    id UUID PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    feedback_type TEXT NOT NULL,
    correction TEXT,
    rating REAL DEFAULT 0.0,
    context_chunks UUID[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Indexes for learning tables
CREATE INDEX IF NOT EXISTS idx_facts_embedding 
ON rag.learned_facts 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_chains_embedding 
ON rag.reasoning_chains 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_facts_confidence 
ON rag.learned_facts (confidence DESC);

CREATE INDEX IF NOT EXISTS idx_facts_topic 
ON rag.learned_facts (topic);

CREATE INDEX IF NOT EXISTS idx_facts_usage 
ON rag.learned_facts (usage_count DESC);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION rag.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to documents table
DROP TRIGGER IF EXISTS update_documents_updated_at ON rag.documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON rag.documents
    FOR EACH ROW
    EXECUTE FUNCTION rag.update_updated_at_column();

-- Apply trigger to learned_facts table
DROP TRIGGER IF EXISTS update_learned_facts_updated_at ON rag.learned_facts;
CREATE TRIGGER update_learned_facts_updated_at
    BEFORE UPDATE ON rag.learned_facts
    FOR EACH ROW
    EXECUTE FUNCTION rag.update_updated_at_column();

-- Grant all permissions again after table creation
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA rag TO omniscient;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA rag TO omniscient;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Database initialization complete for Omniscient Architect';
END $$;
