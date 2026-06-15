"""
Variant C: FastAPI + Uvicorn + scikit-learn (cosine_similarity).

POST /encode computes cosine similarity between a request-derived query
vector and a pre-built reference corpus via
sklearn.metrics.pairwise.cosine_similarity, which calls numpy/OpenBLAS
single-precision GEMV/GEMM internally for float32 arrays - the same BLAS thread pool as Variant B (RQ3 / H6).

Thread count controlled via OPENBLAS_NUM_THREADS environment variable.
  Serialized mode : OPENBLAS_NUM_THREADS=1
  Parallel mode   : unset, defaults to number of available cores

Run with:
    WORKERS=4 uvicorn app.server_sklearn:app --host 0.0.0.0 --port 8000 --workers 4
"""
import hashlib
import os

import anyio.to_thread
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel
from threadpoolctl import threadpool_info

# Pre-built reference corpus; simulates an embedding store.
# Initialized once per worker process at import time.
_rng = np.random.default_rng(42)
_CORPUS = _rng.standard_normal((2048, 384)).astype(np.float32)
_norms = np.linalg.norm(_CORPUS, axis=1, keepdims=True)
_CORPUS_NORMED = _CORPUS / (_norms + 1e-8)   # (2048, 384), pre-normalized

app = FastAPI()


@app.on_event("startup")
async def configure_fastapi_threadpool() -> None:
    """Control the AnyIO thread limiter used by FastAPI sync endpoints."""
    limit = int(os.getenv("FASTAPI_THREAD_LIMIT", "40"))
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = limit
    app.state.fastapi_thread_limit = limit


class EncodeRequest(BaseModel):
    text: str


@app.post("/encode")
def encode(req: EncodeRequest):
    """
    CPU-bound endpoint. Calls OpenBLAS via sklearn cosine_similarity.

    Query vector is derived deterministically from the request text hash
    so that each distinct text produces a distinct query, avoiding trivial
    cache effects while being safe for concurrent FastAPI thread-pool calls.
    """
    seed = int(hashlib.md5(req.text.encode()).hexdigest()[:8], 16)
    local_rng = np.random.default_rng(seed)
    query = local_rng.standard_normal((1, 384)).astype(np.float32)

    # cosine_similarity calls np.dot internally -> OpenBLAS SGEMM/SGEMV
    scores = cosine_similarity(query, _CORPUS_NORMED)   # (1, 2048)
    top_k = int(np.argmax(scores))
    return {"top_k": top_k, "score": float(scores[0, top_k]), "library": "sklearn"}


@app.get("/health")
def health():
    """Lightweight I/O-floor endpoint. Produces voluntary context switches."""
    return {"status": "ok"}


@app.get("/runtime")
def runtime():
    """Return runtime thread-pool settings for reproducibility diagnostics."""
    return {
        "library": "sklearn",
        "fastapi_thread_limit": getattr(app.state, "fastapi_thread_limit", None),
        "threadpools": threadpool_info(),
    }

