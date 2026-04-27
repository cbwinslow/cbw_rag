import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Settings import
from .settings import settings

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, PromptHelper
from llama_index.vector_stores.pgvector import PgvectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import HybridRetriever

app = FastAPI()

# ----------------------------------------------------------------------
# Configuration – use Settings singleton
# ----------------------------------------------------------------------
DB_DSN = settings.db_dsn
OLLAMA_URL = settings.ollama_url
EMBEDDING_MODEL = settings.embedding_model
LLM_MODEL = settings.llm_model

# ----------------------------------------------------------------------
# Lazy singleton for vector store and index
# ----------------------------------------------------------------------
_vector_store: PgvectorStore | None = None
_index: VectorStoreIndex | None = None

def _get_index() -> VectorStoreIndex:
    global _vector_store, _index
    if _index is None:
        _vector_store = PgvectorStore.from_params(
            uri=DB_DSN,
            collection_name="rag_docs",
            embed_dim=768,
        )
        _index = VectorStoreIndex.from_vector_store(_vector_store)
    return _index

# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class AnswerRequest(BaseModel):
    query: str
    top_k: int = 5
    stream: bool = False

# ----------------------------------------------------------------------
# Helper to run LLM on retrieved context
# ----------------------------------------------------------------------
def _run_llm(context: str, query: str) -> str:
    llm = Ollama(base_url=OLLAMA_URL, model=LLM_MODEL)
    prompt = f"Context:\n{context}\n\nAnswer the following question concisely:\n{query}"
    return llm.complete(prompt).text

def _run_llm_stream(context: str, query: str):
    llm = Ollama(base_url=OLLAMA_URL, model=LLM_MODEL, streaming=True)
    prompt = f"Context:\n{context}\n\nAnswer the following question concisely:\n{query}"
    return llm.stream_complete(prompt)

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@record_metrics("query")
@app.post("/query")
async def query_rag(req: QueryRequest):
    try:
        index = _get_index()
        retriever = index.as_retriever(similarity_top_k=req.top_k)
        nodes = retriever.retrieve(req.query)
        context = "\n---\n".join([node.node.get_text() for node in nodes])
        return {"query": req.query, "results": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@record_metrics("answer")
@app.post("/answer")
async def answer_rag(req: AnswerRequest):
    try:
        index = _get_index()
        retriever = index.as_retriever(similarity_top_k=req.top_k)
        nodes = retriever.retrieve(req.query)
        context = "\n---\n".join([node.node.get_text() for node in nodes])
        answer = _run_llm(context, req.query)
        return {"query": req.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@record_metrics("answer_stream")
@app.post("/answer/stream")
async def answer_rag_stream(req: AnswerRequest):
    try:
        index = _get_index()
        retriever = index.as_retriever(similarity_top_k=req.top_k)
        nodes = retriever.retrieve(req.query)
        context = "\n---\n".join([node.node.get_text() for node in nodes])
        generator = _run_llm_stream(context, req.query)
        return StreamingResponse(generator, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------------------------
# Metrics endpoint (Prometheus)
# ----------------------------------------------------------------------
from .metrics import REQ_COUNTER, REQ_LATENCY, metrics_response

@app.get("/metrics")
async def metrics():
    return StreamingResponse(iter([metrics_response()]), media_type="text/plain")

# Helper decorators to record metrics
def record_metrics(endpoint_name: str):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            REQ_COUNTER.labels(endpoint=endpoint_name).inc()
            with REQ_LATENCY.labels(endpoint=endpoint_name).time():
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# ----------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "domain": settings.domain}
