import argparse
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

CONTEXT_INTRO = "Use the following extracts from the Privacy Act 1988 (Cth) to answer the question."


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    arm: str


def load_settings(path: str) -> dict:
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(Path(settings["generation"]["train_config"]).read_text(encoding="utf-8"))
    settings["system_prompt"] = train_cfg["system_prompt"]
    return settings


def build_user_content(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return question
    blocks = [f"[{i}] {c['display']} {c['title']}\n{c['text']}" for i, c in enumerate(chunks, 1)]
    return "\n\n".join([CONTEXT_INTRO, *blocks, f"Question: {question}"])


def build_messages(system_prompt: str, question: str, chunks: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_content(question, chunks)},
    ]


def load_retriever(settings: dict):
    from ..rag.retrieve import Retriever

    rag_cfg = yaml.safe_load(Path(settings["retrieval"]["config"]).read_text(encoding="utf-8"))
    return Retriever.load(rag_cfg)


def create_app(settings: dict, retriever=None, transport=None) -> FastAPI:
    state = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["client"] = httpx.AsyncClient(transport=transport, timeout=settings["generation"]["timeout_seconds"])
        needs_retrieval = any(a["retrieval"] for a in settings["arms"].values())
        state["retriever"] = retriever if retriever is not None or not needs_retrieval else load_retriever(settings)
        yield
        await state["client"].aclose()

    app = FastAPI(title="privacy-act-lora-vs-rag", lifespan=lifespan)

    @app.get("/arms")
    async def arms() -> dict:
        return {"arms": settings["arms"], "retrieval": settings["retrieval"]}

    @app.get("/health")
    async def health() -> dict:
        backends = {}
        for name, url in settings["backends"].items():
            try:
                r = await state["client"].get(f"{url}/health")
                backends[name] = r.status_code == 200
            except httpx.HTTPError:
                backends[name] = False
        return {"ok": all(backends.values()), "backends": backends}

    @app.post("/answer")
    async def answer(req: AnswerRequest) -> dict:
        arm = settings["arms"].get(req.arm)
        if arm is None:
            raise HTTPException(status_code=422, detail=f"unknown arm: {req.arm}")
        start = time.perf_counter()
        chunks, retrieval_ms = [], 0.0
        if arm["retrieval"]:
            rc = settings["retrieval"]
            result = await run_in_threadpool(state["retriever"].search, req.question, rc["method"], rc["k"])
            chunks, retrieval_ms = result["chunks"], result["retrieval_ms"]
        body = {
            "messages": build_messages(settings["system_prompt"], req.question, chunks),
            "temperature": 0,
            "top_k": 1,
            "max_tokens": settings["generation"]["max_tokens"],
            "seed": 0,
        }
        url = settings["backends"][arm["backend"]]
        try:
            resp = await state["client"].post(f"{url}/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"backend {arm['backend']} unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"backend {arm['backend']} returned {resp.status_code}")
        payload = resp.json()
        timings = payload.get("timings", {})
        usage = payload.get("usage", {})
        return {
            "arm": req.arm,
            "backend": arm["backend"],
            "answer": payload["choices"][0]["message"]["content"].strip(),
            "retrieved": [{"chunk_id": c["chunk_id"], "uid": c["uid"], "display": c["display"]} for c in chunks],
            "latency": {
                "retrieval_ms": retrieval_ms,
                "prefill_ms": timings.get("prompt_ms"),
                "generation_ms": timings.get("predicted_ms"),
                "total_ms": (time.perf_counter() - start) * 1000,
            },
            "tokens": {"prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens")},
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve all experiment arms behind one answer endpoint")
    parser.add_argument("--config", default="configs/api.yaml")
    args = parser.parse_args()

    import uvicorn

    settings = load_settings(args.config)
    uvicorn.run(create_app(settings), host=settings["server"]["host"], port=settings["server"]["port"])


if __name__ == "__main__":
    main()
