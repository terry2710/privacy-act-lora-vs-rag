import random
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from privacy_bench.qa.build_eval import (
    cross_similarity,
    greedy_dedup,
    item_similarity,
    load_neg,
    provision_refs,
    stratified_sample,
    annotate_parallels,
)
from privacy_bench.qa.embed import EmbeddingCache, embed_texts

SECTIONS = {
    "s:6": {"unit": "Part II", "part": "II", "division": "1", "schedule": None},
    "s:26WA": {"unit": "Part IIIC", "part": "IIIC", "division": "1", "schedule": None},
    "sch1:app6": {"unit": "Sch 1", "part": "3", "division": None, "schedule": "1"},
}


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_item_similarity_takes_max_over_phrasings():
    m = np.stack([_unit([1, 0]), _unit([0, 1]), _unit([0, 1]), _unit([1, 1])])
    sim = item_similarity(m, [2, 1, 1])
    assert sim.shape == (3, 3)
    assert sim[0, 1] == pytest.approx(1.0)
    assert sim[1, 2] == pytest.approx(np.sqrt(0.5))


def test_cross_similarity_shape_and_values():
    a = np.stack([_unit([1, 0]), _unit([0, 1])])
    b = np.stack([_unit([1, 0])])
    sim = cross_similarity(a, [1, 1], b, [1])
    assert sim.shape == (1, 2) and sim[0, 0] == pytest.approx(1.0) and sim[0, 1] == pytest.approx(0.0)


def test_greedy_dedup_keeps_first_occurrence():
    sim = np.array([[1.0, 0.99, 0.1], [0.99, 1.0, 0.2], [0.1, 0.2, 1.0]])
    kept, dropped = greedy_dedup(sim, 0.97)
    assert kept == [0, 2] and dropped == [(1, 0, 0.99)]


def test_stratified_sample_is_deterministic_and_disjoint():
    items = [{"fact_id": f"f{i:03d}", "unit": "A" if i < 60 else "B"} for i in range(100)]
    first = stratified_sample(items, 10, random.Random(1), set())
    again = stratified_sample(items, 10, random.Random(1), set())
    assert first == again
    assert sum(it["unit"] == "A" for it in first) == 6
    second = stratified_sample(items, 10, random.Random(2), {it["fact_id"] for it in first})
    assert not {it["fact_id"] for it in first} & {it["fact_id"] for it in second}


def test_provision_refs_distinguish_existing_and_missing():
    refs = dict(provision_refs("Compare APP 6, APP 14, section 6, section 150, Part IIIC, Part X and Schedule 3.", SECTIONS))
    assert refs == {"APP 6": True, "APP 14": False, "s 6": True, "s 150": False, "Part IIIC": True, "Part X": False, "Schedule 3": False}
    refs = dict(provision_refs("What does Division 6 of Part IIIC require?", SECTIONS))
    assert refs == {"Division 6 of Part IIIC": False}


def test_load_neg_rejects_existing_provisions(tmp_path):
    def write(name, rows):
        path = tmp_path / name
        path.write_text(yaml.safe_dump({"items": rows}))
        return path

    good = write("good.yaml", [{"category": "nonexistent_provision", "question": "What does section 150 provide?", "premise": "p"}])
    assert load_neg(good, SECTIONS)[0]["provision_refs"] == ["s 150"]
    bad = write("bad.yaml", [
        {"category": "nonexistent_provision", "question": "What does section 6 provide?", "premise": "p"},
        {"category": "foreign_law", "question": "How does APP 6 adopt the GDPR?", "premise": "p"},
    ])
    with pytest.raises(ValueError, match="item 1.*\n.*item 2"):
        load_neg(bad, SECTIONS)


def test_embedding_cache_only_requests_missing_texts(tmp_path):
    calls = []

    class FakeEmbeddings:
        def create(self, model, input):
            calls.append(list(input))
            return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in input])

    client = SimpleNamespace(embeddings=FakeEmbeddings())
    path = tmp_path / "cache.npz"
    m = embed_texts(["ab", "abc"], EmbeddingCache(path), client, "m", 1)
    assert m.shape == (2, 2) and np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert sorted(len(c) for c in calls) == [1, 1]
    calls.clear()
    embed_texts(["ab", "abcd"], EmbeddingCache(path), client, "m", 8)
    assert calls == [["abcd"]]


def test_annotate_parallels_tags_and_validates():
    records = [{"fact_id": "s:26WU#00-008"}, {"fact_id": "s:26WU#00-001"}]
    parallels = {"s:26WU#00-008": {"train_fact": "s:44#00-004", "reason": "r"}}
    annotate_parallels(records, parallels, {"s:44#00-004"})
    assert [r["parallel_to_train"] for r in records] == ["s:44#00-004", None]
    with pytest.raises(ValueError, match="s:26WU#00-008"):
        annotate_parallels(records, parallels, set())