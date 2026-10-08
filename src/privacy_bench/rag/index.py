import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..train.data import read_jsonl, sha256_file, write_jsonl
from .chunk import chunk_corpus
from .retrieve import SentenceEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk the full Act by section and build the dense and BM25 retrieval index")
    parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    sections = read_jsonl(cfg["corpus"]["sections"])
    chunks = chunk_corpus(sections, cfg["chunking"]["max_chars"], cfg["chunking"]["overlap_chars"])
    encoder = SentenceEncoder(cfg["dense"]["model"], cfg["dense"]["query_prefix"], cfg["dense"]["batch_size"])
    embeddings = encoder.encode([c["index_text"] for c in chunks], is_query=False)

    out = Path(cfg["index_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "chunks.jsonl", chunks)
    np.save(out / "embeddings.npy", embeddings)
    lengths = [len(c["text"]) for c in chunks]
    manifest = {
        "sections_sha256": sha256_file(cfg["corpus"]["sections"]),
        "sections": len(sections),
        "chunks": len(chunks),
        "max_chars": cfg["chunking"]["max_chars"],
        "overlap_chars": cfg["chunking"]["overlap_chars"],
        "chunk_chars_mean": sum(lengths) / len(lengths),
        "chunk_chars_max": max(lengths),
        "dense_model": cfg["dense"]["model"],
        "embedding_dim": int(embeddings.shape[1]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
