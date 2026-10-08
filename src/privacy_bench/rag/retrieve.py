import json
import time
from pathlib import Path

import numpy as np

from ..train.data import read_jsonl
from .bm25 import BM25


def top_indices(scores, k: int) -> list[int]:
    scores = np.asarray(scores)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in idx[np.argsort(-scores[idx], kind="stable")]]


def rrf(rankings: list[list[int]], rrf_k: int) -> list[int]:
    fused = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused, key=lambda i: (-fused[i], i))


class SentenceEncoder:
    def __init__(self, model_name: str, query_prefix: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        if is_query:
            texts = [self.query_prefix + t for t in texts]
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class Retriever:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray, encoder, cfg: dict):
        self.chunks = chunks
        self.embeddings = embeddings
        self.encoder = encoder
        self.cfg = cfg
        self.bm25 = BM25([c["index_text"] for c in chunks], cfg["bm25"]["k1"], cfg["bm25"]["b"])

    @classmethod
    def load(cls, cfg: dict, encoder=None) -> "Retriever":
        index_dir = Path(cfg["index_dir"])
        chunks = read_jsonl(index_dir / "chunks.jsonl")
        embeddings = np.load(index_dir / "embeddings.npy")
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["dense_model"] != cfg["dense"]["model"]:
            raise ValueError("index was built with a different dense model")
        if encoder is None:
            encoder = SentenceEncoder(cfg["dense"]["model"], cfg["dense"]["query_prefix"], cfg["dense"]["batch_size"])
        return cls(chunks, embeddings, encoder, cfg)

    def rank(self, query: str, method: str, k: int) -> list[int]:
        if method == "bm25":
            return top_indices(self.bm25.scores(query), k)
        if method == "dense":
            q = self.encoder.encode([query], is_query=True)[0]
            return top_indices(self.embeddings @ q, k)
        if method == "hybrid":
            n = self.cfg["hybrid"]["candidates"]
            fused = rrf([self.rank(query, "bm25", n), self.rank(query, "dense", n)], self.cfg["hybrid"]["rrf_k"])
            return fused[:k]
        raise ValueError(f"unknown retrieval method: {method}")

    def search(self, query: str, method: str, k: int) -> dict:
        start = time.perf_counter()
        idx = self.rank(query, method, k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"chunks": [self.chunks[i] for i in idx], "retrieval_ms": elapsed_ms}
