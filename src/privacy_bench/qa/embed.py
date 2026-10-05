import hashlib
from pathlib import Path

import numpy as np


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.vectors: dict[str, np.ndarray] = {}
        if path.exists():
            data = np.load(path)
            self.vectors = dict(zip(data["keys"].tolist(), data["vectors"]))

    def missing(self, texts: list[str]) -> list[str]:
        return sorted({t for t in texts if text_key(t) not in self.vectors})

    def add(self, texts: list[str], vectors: list[list[float]]) -> None:
        for text, vec in zip(texts, vectors):
            self.vectors[text_key(text)] = np.asarray(vec, dtype=np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.vectors)
        np.savez(self.path, keys=np.array(keys), vectors=np.stack([self.vectors[k] for k in keys]))

    def matrix(self, texts: list[str]) -> np.ndarray:
        m = np.stack([self.vectors[text_key(t)] for t in texts])
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def embed_texts(texts: list[str], cache: EmbeddingCache, client, model: str, batch_size: int) -> np.ndarray:
    todo = cache.missing(texts)
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        cache.add(batch, [d.embedding for d in resp.data])
    if todo:
        cache.save()
    return cache.matrix(texts)