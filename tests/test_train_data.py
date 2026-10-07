import pytest

from privacy_bench.train.bundle import check_inputs
from privacy_bench.train.data import apply_overrides, flat_defaults, join_example, pad_batch, probe_items, sft_rows

CFG = {
    "base_model": {"dtype": "float16"},
    "data": {"phrasings": 2, "max_length": 512},
    "lora": {"r": 16, "alpha_multiplier": 2, "dropout": 0.05, "target": "attention"},
    "train": {"learning_rate": 2e-4, "epochs": 3, "batch_size": 8, "grad_accum": 2,
              "warmup_ratio": 0.05, "weight_decay": 0.0, "seed": 1},
}
FACT = {"fact_id": "s:6#00-001", "uid": "s:6", "questions": ["Q1?", "Q2?"], "answer": "A. [Source: s 6]"}


def test_overrides_apply_and_validate():
    out = apply_overrides(CFG, {"lora_r": 32, "learning_rate": 1e-4, "phrasings": 1})
    assert out["lora"]["r"] == 32 and out["train"]["learning_rate"] == 1e-4 and out["data"]["phrasings"] == 1
    assert CFG["lora"]["r"] == 16
    assert flat_defaults(CFG)["lora_target"] == "attention"
    with pytest.raises(KeyError):
        apply_overrides(CFG, {"nope": 1})
    with pytest.raises(ValueError):
        apply_overrides(CFG, {"lora_target": "mlp"})
    with pytest.raises(ValueError):
        apply_overrides(CFG, {"base_dtype": "bfloat16"})


def test_sft_rows_respect_phrasings():
    two = sft_rows([FACT], "SYS", 2)
    one = sft_rows([FACT], "SYS", 1)
    assert [r["phrasing"] for r in two] == [0, 1] and len(one) == 1
    assert [m["role"] for m in two[1]["messages"]] == ["system", "user", "assistant"]
    assert two[1]["messages"][1]["content"] == "Q2?"


def test_join_and_pad_mask_prompt_and_padding():
    ex = join_example([1, 2, 3], [4, 5], 10)
    assert ex["labels"] == [-100, -100, -100, 4, 5]
    assert join_example([1, 2, 3], [4, 5], 4) is None
    batch = pad_batch([ex, join_example([1], [9], 10)], 0)
    assert batch["input_ids"][1] == [1, 9, 0, 0, 0]
    assert batch["attention_mask"][1] == [1, 1, 0, 0, 0]
    assert batch["labels"][1] == [-100, 9, -100, -100, -100]


def test_check_inputs_rejects_holdout_and_unknown_dev_fact():
    provisions = [{"uid": "s:6", "display": "s 6"}, {"uid": "s:26WA", "display": "s 26WA"}]
    dev = [{"eval_id": "d1", "fact_id": "s:6#00-001", "uid": "s:6", "reference_answer": "B. [Source: s 6]"}]
    counts = check_inputs([FACT], dev, provisions, ["^s:26W[A-Z]"])
    assert counts["train_pairs_two_phrasings"] == 2
    leaked = {**FACT, "fact_id": "s:26WA#00-001", "uid": "s:26WA", "answer": "A. [Source: s 26WA]"}
    with pytest.raises(ValueError):
        check_inputs([FACT, leaked], dev, provisions, ["^s:26W[A-Z]"])
    with pytest.raises(ValueError):
        check_inputs([FACT], [{**dev[0], "fact_id": "s:7#00-001"}], provisions, [])


def test_probe_items_exclude_and_are_deterministic():
    facts = [{**FACT, "fact_id": f"s:6#00-{i:03d}", "display": "s 6"} for i in range(10)]
    excluded = {"s:6#00-000", "s:6#00-001"}
    a = probe_items(facts, excluded, 5, 7)
    b = probe_items(list(reversed(facts)), excluded, 5, 7)
    assert a == b and len(a) == 5
    assert not {p["fact_id"] for p in a} & excluded
    assert a[0]["eval_id"] == "probe-0001" and a[0]["question"] == "Q1?"
    assert len(probe_items(facts, excluded, 50, 7)) == 8


def test_overrides_coerce_string_numbers_to_default_types():
    out = apply_overrides(CFG, {"learning_rate": "5e-4", "lora_r": "32", "epochs": 5.0})
    assert out["train"]["learning_rate"] == 5e-4 and isinstance(out["train"]["learning_rate"], float)
    assert out["lora"]["r"] == 32 and isinstance(out["lora"]["r"], int)
    assert out["train"]["epochs"] == 5 and isinstance(out["train"]["epochs"], int)