"""Jev-compatible HTTP server: ``POST /v1/systemone`` and ``GET /v1/models``.

The official ``typesafe-sdk`` works against it with only a ``base_url`` change::

    TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="any2jev-latest")
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date

from .model import DecisionModel
from .schema import SystemOneRequest

DEFAULT_MODEL_NAME = "any2jev-latest"


def create_app(model: DecisionModel, model_name: str = DEFAULT_MODEL_NAME, aliases: tuple[str, ...] = ("jev-latest",)):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="any2jev", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    lock = threading.Lock()
    stats = {"requests": 0, "questions": 0, "latency_ms_total": 0.0}

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        # Any model name is accepted and routed to the one model we serve (Jev-style aliasing).
        with lock:
            t0 = time.perf_counter()
            answers, n_in = model.decide(req)
            latency_ms = (time.perf_counter() - t0) * 1000
        stats["requests"] += 1
        stats["questions"] += len(req.questions)
        stats["latency_ms_total"] += latency_ms
        n_out = len(model.tok(json.dumps(answers), add_special_tokens=False).input_ids)  # billing-style, not generated
        return {"model": model_name, "answers": answers, "usage": {"input_tokens": n_in, "output_tokens": n_out},
                "latency_ms": round(latency_ms, 1)}

    @app.get("/v1/models")
    def models():
        base = model.meta.get("base", "unknown")
        return {"models": [{"name": name, "description": f"any2jev decision model on {base}",
                            "release_date": date.today().isoformat()} for name in (model_name, *aliases)]}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "base": model.meta.get("base"), "mode": model.mode,
                "temperature": model.temperature, "device": str(model.device), **stats}

    return app


def serve(model_dir: str, host: str = "127.0.0.1", port: int = 8009, device: str | None = None,
          dtype: str | None = None, attn: str | None = None, model_name: str = DEFAULT_MODEL_NAME):
    import uvicorn

    model = DecisionModel.load(model_dir, device=device, dtype=dtype, attn=attn)
    app = create_app(model, model_name)
    print(f"any2jev: serving {model_dir} ({model.meta.get('base')}) on http://{host}:{port}  mode={model.mode} device={model.device}")
    uvicorn.run(app, host=host, port=port, log_level="info")
