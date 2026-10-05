#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: palr <command> [args]

  test [args]        Run the test suite (extra args passed to pytest)
  parse              Parse the Act DOCX into section records
  split              Assign splits, allocate quotas and mask held-out content
  gen-dry [args]     Plan QA generation without calling the API
  gen [args]         Run QA generation (for example: palr gen --split train)
  report             Summarise the latest QA generation report
  models-anthropic   List Anthropic models available to the API key
  models-openai      List OpenAI models available to the API key
  status             Show git status and the latest commits
  commit <name>      Commit staged changes with .commit_msgs/<name>.txt
EOF
}

need_py() {
  if [[ ! -x "$PY" ]]; then
    echo "Virtual environment not found at $ROOT/.venv" >&2
    exit 1
  fi
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  test)
    need_py
    "$PY" -m pytest "$@"
    ;;
  parse)
    need_py
    "$PY" -m privacy_bench.data.parse_act \
      --toc data/raw/toc.txt \
      --docx data/raw/privacy_act_1988.docx \
      --out data/processed/sections.jsonl \
      --report data/processed/parse_report.json
    ;;
  split)
    need_py
    "$PY" -m privacy_bench.data.build_split "$@"
    ;;
  gen-dry)
    need_py
    "$PY" -m privacy_bench.qa.generate --dry-run "$@"
    ;;
  gen)
    need_py
    "$PY" -m privacy_bench.qa.generate "$@"
    ;;
  report)
    need_py
    "$PY" - <<'EOF'
import json
r = json.load(open("data/audit/generation_report.json"))
print(f"model={r['generator_model']} prompt={r['prompt_version']} hash={r['prompt_hash']}")
print(f"requested={r['requested']}")
print(f"accepted={r['accepted']}")
print(f"rejected_by_reason={r['rejected_by_reason']}")
print(f"shortfall_jobs={len(r['shortfall'])} missing={sum(s['quota'] - s['accepted'] for s in r['shortfall'])}")
print(f"usage={r['usage']}")
EOF
    ;;
  models-anthropic)
    need_py
    "$PY" -c "import anthropic, dotenv; dotenv.load_dotenv(); print('\n'.join(m.id for m in anthropic.Anthropic().models.list().data))"
    ;;
  models-openai)
    need_py
    "$PY" -c "import openai, dotenv; dotenv.load_dotenv(); print('\n'.join(sorted(m.id for m in openai.OpenAI().models.list().data)))"
    ;;
  status)
    git status --short
    git log --oneline -5
    ;;
  commit)
    if [[ $# -ne 1 ]]; then
      echo "Usage: palr commit <name>" >&2
      exit 1
    fi
    msg=".commit_msgs/$1.txt"
    if [[ ! -f "$msg" ]]; then
      echo "Commit message file not found: $msg" >&2
      exit 1
    fi
    git commit -F "$msg"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
