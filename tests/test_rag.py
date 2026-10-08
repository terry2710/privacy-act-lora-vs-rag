import numpy as np

from privacy_bench.rag.bm25 import BM25, tokenize
from privacy_bench.rag.chunk import chunk_section, pack, split_long
from privacy_bench.rag.eval_retrieval import metrics_for, select
from privacy_bench.rag.retrieve import Retriever, rrf, top_indices

SECTION = {"uid": "s:6", "display": "s 6", "title": "Interpretation", "unit": "Part II",
           "text": "(1) First paragraph about agencies.\n(2) Second paragraph about records.\n(3) Third paragraph."}
CFG = {"bm25": {"k1": 1.5, "b": 0.75}, "hybrid": {"rrf_k": 60, "candidates": 10}}


def test_chunks_stay_within_section_and_limit():
    chunks = chunk_section(SECTION, 70, 40)
    assert all(c["uid"] == "s:6" and len(c["text"]) <= 70 for c in chunks)
    assert chunks[0]["chunk_id"] == "s:6#c000"
    assert chunks[0]["index_text"].startswith("s 6 Interpretation\n")
    joined = "\n".join(c["text"] for c in chunks)
    assert all(p in joined for p in SECTION["text"].split("\n"))


def test_split_long_and_pack_overlap():
    assert all(len(p) <= 20 for p in split_long("a" * 50, 20))
    assert pack(["aaaa", "bbbb", "cccc"], 9, 4) == ["aaaa\nbbbb", "bbbb\ncccc"]
    assert pack(["aaaa", "bbbb", "cccc"], 9, 2) == ["aaaa\nbbbb", "cccc"]


def test_bm25_prefers_matching_document():
    bm = BM25(["privacy act agency records", "tax file number rules", "credit reporting body"])
    scores = bm.scores("tax file number")
    assert int(np.argmax(scores)) == 1
    assert tokenize("Section 26WA(1)") == ["section", "26wa", "1"]


def test_rrf_and_top_indices():
    assert top_indices([0.1, 0.9, 0.5], 2) == [1, 2]
    assert rrf([[0, 1, 2], [2, 0, 1]], 60)[0] == 0


class Fake:
    def encode(self, texts, is_query):
        vecs = np.array([[t.count("tax"), t.count("credit"), 1.0] for t in texts], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_retriever_methods_and_metrics():
    chunks = [
        {"uid": "s:1", "index_text": "tax file number tax", "text": ""},
        {"uid": "s:2", "index_text": "credit reporting credit", "text": ""},
        {"uid": "s:3", "index_text": "agency records", "text": ""},
    ]
    r = Retriever(chunks, Fake().encode([c["index_text"] for c in chunks], False), Fake(), CFG)
    for method in ("bm25", "dense", "hybrid"):
        assert r.search("credit reporting", method, 2)["chunks"][0]["uid"] == "s:2"
    m = metrics_for(["s:2", "s:3"], [[chunks[1], chunks[0]], [chunks[0], chunks[2]]], [1, 2])
    assert m["recall@1"] == 0.5 and m["recall@2"] == 1.0 and m["mrr"] == 0.75


def test_select_rule():
    results = {
        "bm25": {"recall@1": 0.6, "recall@3": 0.85, "recall@5": 0.9, "mrr": 0.7},
        "hybrid": {"recall@1": 0.7, "recall@3": 0.92, "recall@5": 0.93, "mrr": 0.8},
    }
    assert select(results, 5, [1, 3, 5], 0.02) == {"method": "hybrid", "k": 3, "recall_at_k": 0.92}
