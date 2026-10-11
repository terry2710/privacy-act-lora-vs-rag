import json
from pathlib import Path

from .rubric import RUBRICS, label


def call_openai(client, cfg: dict, kind: str, user: str) -> tuple[str, str, str, int, int]:
    rubric = RUBRICS[kind]
    extra = {"reasoning": {"effort": cfg["reasoning_effort"]}} if cfg.get("reasoning_effort") else {}
    resp = client.responses.create(
        model=cfg["model"],
        instructions=rubric["system"],
        input=user,
        text={"format": {"type": "json_schema", "name": f"judge_{kind}", "schema": rubric["schema"], "strict": True}},
        max_output_tokens=cfg["max_output_tokens"],
        store=False,
        **extra,
    )
    text = resp.output_text if resp.status == "completed" else ""
    return text, resp.model, resp.status, resp.usage.input_tokens, resp.usage.output_tokens


def call_anthropic(client, cfg: dict, kind: str, user: str) -> tuple[str, str, str, int, int]:
    rubric = RUBRICS[kind]
    output_config = {"format": {"type": "json_schema", "schema": rubric["schema"]}}
    if cfg.get("effort"):
        output_config["effort"] = cfg["effort"]
    resp = client.messages.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        system=rubric["system"],
        output_config=output_config,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.model, resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens


CALLERS = {"anthropic": call_anthropic, "openai": call_openai}


def make_client(cfg: dict, max_retries: int):
    if cfg["provider"] == "anthropic":
        import anthropic

        return anthropic.Anthropic(max_retries=max_retries)
    import openai

    return openai.OpenAI(max_retries=max_retries)


def judge_one(name: str, cfg: dict, client, task: dict, cache_dir: Path, phash: str, caller=None, attempts: int = 2) -> dict:
    path = cache_dir / name / phash / f"{task['key']}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("answer_hash") == task["answer_hash"] and cached.get("label") is not None:
            return cached
    caller = caller or CALLERS[cfg["provider"]]
    record = {
        "key": task["key"], "eval_id": task["eval_id"], "set": task["set"], "arm": task["arm"], "kind": task["kind"],
        "judge": name, "prompt_hash": phash, "answer_hash": task["answer_hash"],
        "verdict": None, "label": None, "input_tokens": 0, "output_tokens": 0, "calls": 0,
    }
    for _ in range(attempts):
        text, model, status, tin, tout = caller(client, cfg, task["kind"], task["user"])
        record.update(model=model, status=status)
        record["input_tokens"] += tin
        record["output_tokens"] += tout
        record["calls"] += 1
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if label(task["kind"], parsed) is not None:
            record.update(verdict=parsed, label=label(task["kind"], parsed))
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record
