import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LlamaIndex imports
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pgvector import PgvectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import HybridRetriever

app = FastAPI()

# ----------------------------------------------------------------------
# Configuration – reuse environment variables (same as indexer)
# ----------------------------------------------------------------------
DB_DSN = os.getenv(
    "CBW_RAG_DATABASE",
    "postgresql://cbwinslow:123qweasd@localhost:5432/cbw_rag",
)
OLLAMA_URL = os.getenv("CBW_RAG_OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("CBW_RAG_EMBEDDING_MODEL", "nomic-embed-text")

# ----------------------------------------------------------------------
# Build the LlamaIndex vector store (lazy singleton)
# ----------------------------------------------------------------------
_vector_store: PgvectorStore | None = None
_index: VectorStoreIndex | None = None

def _get_index() -> VectorStoreIndex:
    global _vector_store, _index
    if _index is None:
        # Initialise pgvector store – it will read from the existing tables
        _vector_store = PgvectorStore.from_params(
            uri=DB_DSN,
            collection_name="rag_docs",
            embed_dim=768,
        )
        # SimpleRetriever – we let LlamaIndex build a query engine on the fly
        _index = VectorStoreIndex.from_vector_store(_vector_store)
    return _index

# ----------------------------------------------------------------------
# Request model
# ----------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    # optional: hybrid flag – for now we always use the default LlamaIndex retriever (vector+full‑text)

# ----------------------------------------------------------------------
# API endpoint
# ----------------------------------------------------------------------
@app.post("/query")
async def query_rag(req: QueryRequest):
    try:
        index = _get_index()
        # Retrieve relevant chunks (LlamaIndex handles hybrid search internally)
        retriever = index.as_retriever(similarity_top_k=req.top_k)
        nodes = retriever.retrieve(req.query)
        # Concatenate the chunk texts for a simple response
        context = "\n---\n".join([node.node.get_text() for node in nodes])
        return {"query": req.query, "results": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------------------------
# Health check (optional)
# ----------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
