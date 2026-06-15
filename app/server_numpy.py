"""
Variant B: FastAPI + Uvicorn + numpy / OpenBLAS.

POST /encode performs a two-matrix multiply via np.dot, calling OpenBLAS
single-precision GEMM internally. OpenBLAS spawns a thread pool that inherits the worker
process's CPU affinity mask via POSIX pthread_create semantics - the same
mechanism as PyTorch (RQ3 / H6).

Thread count controlled via OPENBLAS_NUM_THREADS environment variable.
  Serialized mode : OPENBLAS_NUM_THREADS=1
  Parallel mode   : unset, defaults to number of available cores

Run with:
    WORKERS=4 uvicorn app.server_numpy:app --host 0.0.0.0 --port 8000 --workers 4
"""
import os

import anyio.to_thread
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from threadpoolctl import threadpool_info

# Pre-allocated weight matrices simulating fixed model weights.
# Initialized once per worker process at import time.
_rng = np.random.default_rng(42)
_W1 = _rng.standard_normal((512, 384)).astype(np.float32)   # (512, 384)
_W2 = _rng.standard_normal((384, 512)).astype(np.float32)   # (384, 512)

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
    CPU-bound endpoint. Calls OpenBLAS SGEMM via np.dot.
    np.dot(_W1, _W2) -> (512, 512) matrix via BLAS SGEMM.
    L2 norm via np.linalg.norm -> single-precision BLAS norm routines.
    Both calls release the GIL and use the OpenBLAS thread pool.
    """
    h = np.dot(_W1, _W2)                                # (512, 512), BLAS SGEMM
    norms = np.linalg.norm(h, axis=1, keepdims=True)    # BLAS norm routine
    h_normed = h / (norms + 1e-8)
    return {"dim": int(h_normed.shape[1]), "library": "numpy"}


@app.get("/health")
def health():
    """Lightweight I/O-floor endpoint. Produces voluntary context switches."""
    return {"status": "ok"}


@app.get("/runtime")
def runtime():
    """Return runtime thread-pool settings for reproducibility diagnostics."""
    return {
        "library": "numpy",
        "fastapi_thread_limit": getattr(app.state, "fastapi_thread_limit", None),
        "threadpools": threadpool_info(),
    }
