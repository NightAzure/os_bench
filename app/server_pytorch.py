"""
Variant A: FastAPI + Uvicorn + PyTorch + SentenceTransformer.

POST /encode calls model.encode() synchronously, releasing the GIL and
using PyTorch CPU intra-op parallelism. Thread count is controlled via the
PYTORCH_NUM_THREADS environment variable and is applied at worker-process
startup before model execution.

Thread count control:
  Serialized mode : PYTORCH_NUM_THREADS=1  OMP_NUM_THREADS=1
  Parallel mode   : leave PYTORCH_NUM_THREADS unset (runtime default)
  Optional        : TORCH_NUM_INTEROP_THREADS=<n>

Run with:
    WORKERS=4 uvicorn app.server_pytorch:app --host 0.0.0.0 --port 8000 --workers 4
"""
from __future__ import annotations

import os

import anyio.to_thread
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from threadpoolctl import threadpool_info


def _configure_torch_threads_from_env() -> None:
    """Apply PyTorch thread settings before model execution.

    PyTorch recommends calling torch.set_num_threads() before running eager,
    JIT, or autograd code. This module is imported separately in each Uvicorn
    worker process, so applying the setting here makes the worker-level thread
    cap effective before the first inference call.
    """
    num_threads = os.getenv("PYTORCH_NUM_THREADS")
    if num_threads:
        try:
            torch.set_num_threads(int(num_threads))
        except Exception:
            pass

    interop_threads = os.getenv("TORCH_NUM_INTEROP_THREADS")
    if interop_threads:
        try:
            torch.set_num_interop_threads(int(interop_threads))
        except Exception:
            pass


_configure_torch_threads_from_env()

# Load model once per worker process at import time.
model = SentenceTransformer("all-MiniLM-L6-v2")

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
    """CPU-bound endpoint. GIL is released during native compute."""
    embedding = model.encode(req.text, convert_to_numpy=True)
    return {
        "dim": int(len(embedding)),
        "library": "pytorch",
        "torch_num_threads": int(torch.get_num_threads()),
    }


@app.get("/health")
def health():
    """Lightweight endpoint for readiness checks and mixed workload generation."""
    return {
        "status": "ok",
        "library": "pytorch",
        "torch_num_threads": int(torch.get_num_threads()),
    }


@app.get("/runtime")
def runtime():
    """Return runtime thread-pool settings for reproducibility diagnostics."""
    return {
        "library": "pytorch",
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "fastapi_thread_limit": getattr(app.state, "fastapi_thread_limit", None),
        "threadpools": threadpool_info(),
    }
