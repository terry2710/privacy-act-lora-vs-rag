import json

import numpy as np
import pytest
import yaml

from privacy_bench.analysis import run
from privacy_bench.analysis.stats import bootstrap, holm, mcnemar_exact, mean_diff, median_diff


def test_holm_is_monotone_and_capped():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
    assert holm({"a": 0.9, "b": 0.8})["a"] == 1.0


def test_mcnemar_exact():
    out = mcnemar_exact(np.array([1, 1, 1, 0]), np.array([0, 0, 0, 0]))
    assert out["x_only"] == 3 and out["y_only"] == 0 and out["p_two_sided"] == pytest.approx(0.25)
    assert mcnemar_exact(np.array([1, 0]), np.array([1, 0]))["p_two_sided"] == 1.0


def test_bootstrap_paired_and_independent():
    x = np.ones(50)
    y = np.zeros(50)
    out = bootstrap(mean_diff, [x, y], 500, 1)
    assert out["estimate"] == 1.0 and out["ci95"] == [1.0, 1.0] and out["p_two_sided"] < 0.01
    same = bootstrap(mean_diff, [np.array([0.0, 1.0] * 25), np.array([1.0, 0.0] * 25)], 500, 1)
    assert same["ci95"][0] < 0 < same["ci95"][1]
    ind = bootstrap(median_diff, [np.arange(10.0), np.arange(5.0)], 500, 1, paired=False)
    assert ind["estimate"] == pytest.approx(2.5)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def answer(eval_id, set_name, arm, hit, tag=True, wrong=False, total=100.0):
    return {"eval_id": eval_id, "set": set_name, "arm": arm, "answer": "x",
            "latency": {"retrieval_ms": 5.0 if arm in ("A2", "A3") else None, "prefill_ms": total / 4, "generation_ms": total / 2, "total_ms": total},
            "tokens": {"completion": 10},
            "citation": {"tag_present": tag, "cited": ["s:44"] if tag else [], "hit": hit, "mention_hit": hit,
                         "nonexistent": [], "has_nonexistent": False, "wrong_existing": wrong}}


def judgement(eval_id, set_name, arm, judge, label):
    verdict = {"premise_corrected": label == "pass", "invents_content": False} if set_name == "s_neg" else {"contradicts_reference": False}
    return {"eval_id": eval_id, "set": set_name, "arm": arm, "judge": judge, "label": label, "verdict": verdict}


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    acc = {"A0": 0, "A1": 1, "A2": 2, "A3": 2}
    sets = {
        "s_mem": [{"eval_id": f"m{i}", "fact_id": f"f{i}"} for i in range(6)],
        "s_para": [{"eval_id": f"p{i}", "fact_id": f"f{i // 2}"} for i in range(12)],
        "h1": [{"eval_id": f"h{i}", "fact_id": f"g{i}", "parallel_to_train": "s:44#00-001" if i == 0 else None} for i in range(6)],
        "h2": [{"eval_id": f"k{i}", "fact_id": f"k{i}"} for i in range(6)],
        "s_neg": [{"eval_id": f"n{i}"} for i in range(6)],
    }
    for set_name, items in sets.items():
        write(tmp_path / f"eval/{set_name}.jsonl", items)
        for arm in run.ARMS:
            write(tmp_path / f"answers/{set_name}/{arm}.jsonl",
                  [answer(i["eval_id"], set_name, arm, hit=arm == "A2", wrong=arm == "A1", total=200.0 if arm == "A2" else 100.0) for i in items])
            for judge in ("gpt", "claude"):
                label = ("pass" if arm == "A2" else "fail") if set_name == "s_neg" else (2 if set_name == "s_mem" and arm == "A1" else acc[arm])
                write(tmp_path / f"judgements/{judge}/{set_name}/{arm}.jsonl", [judgement(i["eval_id"], set_name, arm, judge, label) for i in items])
    cfg = {"seed": 1, "resamples": 200, "alpha": 0.05, "judges": ["gpt", "claude"], "eval_dir": "eval",
           "answers_dir": "answers", "judgements_dir": "judgements", "out_json": "out/analysis.json", "out_md": "out/analysis.md"}
    (tmp_path / "analysis.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert run.main(["--config", "analysis.yaml"]) == 0
    result = json.loads((tmp_path / "out/analysis.json").read_text(encoding="utf-8"))
    assert result["set_sizes"]["h1_clean"] == 5 and result["set_sizes"]["h1_parallel"] == 1
    fam = result["primary"]["family"]
    assert fam["H1a_s_para_A1_gt_A0"]["supported"] and fam["H1b_s_para_A2_gt_A1"]["supported"]
    assert result["primary"]["H1_supported"] and result["primary"]["H3_supported"]
    assert result["secondary"]["H2_memorisation_gap_A1"]["estimate"] == pytest.approx(1.0)
    assert result["secondary"]["H5_latency_A2_vs_A1"]["total_median_diff"]["estimate"] == pytest.approx(100.0)
    assert result["exploratory"]["H7_s_neg_pass_A2_gt_A1"]["mean_x"] == 1.0
    assert result["descriptive"]["h2"]["A1"]["bad_citation_given_tag"] == 1.0
    assert "Primary hypotheses" in (tmp_path / "out/analysis.md").read_text(encoding="utf-8")
