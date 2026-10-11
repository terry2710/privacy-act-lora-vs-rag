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
  verify-dry [args]  Plan QA verification without calling the API
  verify [args]      Run QA verification with the second LLM
  verify-report      Summarise the latest verification report
  eval-dry           Validate S-neg and plan evaluation set building
  eval-build         Deduplicate, audit semantic leakage and build evaluation sets
  para-dry [args]    Plan paraphrase test question generation
  para [args]        Generate S-para and dev-para paraphrase questions
  spot-export        Export the blind human spot-check sheet
  audit [args]       Run dual-model review of the spot-check sample (use --limit N for a pilot)
  audit-score        Combine model agreement and human adjudication into the final report
  bundle [args]      Build the Kaggle training bundle zip (use --owner <kaggle username>)
  merge a0|a1        Merge the LoRA adapter into the pinned base model (a0 saves the base model)
  gguf a0|a1         Convert the merged model to GGUF with llama.cpp (needs LLAMA_CPP_DIR)
  parity [args]      Compare GGUF outputs from llama-server with the PEFT dev-para outputs
  rag-index          Chunk the Act by section and build the retrieval index
  rag-eval           Score BM25, dense and hybrid retrieval on dev-para and select method and k
  api                Serve all arms behind the FastAPI answer endpoint (needs both llama-server backends)
  registry <cmd>     MLflow registry: register a0|a1, gate [--dry-run], promote <v>, rollback, resolve [alias]
  mlflow-ui          Open the MLflow UI on port 5000 against the local SQLite store
  serve-prod <port>  Start llama-server with the GGUF that currently holds the production alias
  serve <gguf> <port>  Start llama-server for a GGUF file on the given port (both arguments required)
  answers [args]     Collect answers for every set and arm from the answer service (resumable; --sets/--arms to narrow)
  judge-pilot [n]    Judge n calibration answers with both judges and project token use (default 20)
  judge-calibrate    Run both judges on the calibration sample and write the blind adjudication sheet
  judge-score        Combine judge agreement and adjudication into the calibration report and gate
  judge-run [args]   Judge all answers with the primary judge once calibration has passed
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
  verify-dry)
    need_py
    "$PY" -m privacy_bench.qa.verify --dry-run "$@"
    ;;
  verify)
    need_py
    "$PY" -m privacy_bench.qa.verify "$@"
    ;;
  verify-report)
    need_py
    "$PY" - <<'EOF'
import json
r = json.load(open("data/audit/verification_report.json"))
print(f"model={r['verifier_model']} prompt={r['verifier_prompt_version']} hash={r['verifier_prompt_hash']}")
for split, stats in r["by_split"].items():
    print(f"{split}: {stats}")
print(f"issues={r['issues']}")
low = {u: s["pass_rate"] for u, s in r["by_unit"].items() if s["pass_rate"] < 0.9}
print(f"units_below_90pct={low}")
print(f"usage={r['usage']}")
EOF
    ;;
  eval-dry)
    need_py
    "$PY" -m privacy_bench.qa.build_eval --dry-run "$@"
    ;;
  eval-build)
    need_py
    "$PY" -m privacy_bench.qa.build_eval "$@"
    ;;
  para-dry)
    need_py
    "$PY" -m privacy_bench.qa.paraphrase --dry-run "$@"
    ;;
  para)
    need_py
    "$PY" -m privacy_bench.qa.paraphrase "$@"
    ;;
  spot-export)
    need_py
    "$PY" -m privacy_bench.qa.spot_check export "$@"
    ;;
  audit)
    need_py
    "$PY" -m privacy_bench.qa.audit run "$@"
    ;;
  audit-score)
    need_py
    "$PY" -m privacy_bench.qa.audit score "$@"
    ;;
  bundle)
    need_py
    "$PY" -m privacy_bench.train.bundle "$@"
    ;;
  merge)
    need_py
    "$PY" -m privacy_bench.serve.merge --model "${1:?usage: palr merge a0|a1}"
    ;;
  gguf)
    need_py
    "$PY" -m privacy_bench.serve.gguf --model "${1:?usage: palr gguf a0|a1}"
    ;;
  parity)
    need_py
    "$PY" -m privacy_bench.serve.parity "$@"
    ;;
  rag-index)
    need_py
    "$PY" -m privacy_bench.rag.index "$@"
    ;;
  rag-eval)
    need_py
    "$PY" -m privacy_bench.rag.eval_retrieval "$@"
    ;;
  serve)
    if [[ $# -ne 2 ]]; then
      echo "usage: palr serve <gguf path> <port>" >&2
      exit 1
    fi
    model="$1"
    port="$2"
    if [[ ! -f "$model" ]]; then
      echo "model file not found: $model" >&2
      exit 1
    fi
    if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
      echo "port must be an integer between 1024 and 65535: $port" >&2
      exit 1
    fi
    command -v llama-server >/dev/null || { echo "llama-server not on PATH" >&2; exit 1; }
    exec llama-server -m "$model" --port "$port" --jinja -ngl 99 -c 8192
    ;;
  api)
    need_py
    HF_HUB_OFFLINE=1 "$PY" -m privacy_bench.serve.app "$@"
    ;;
  answers)
    need_py
    "$PY" -m privacy_bench.evaluate.answers "$@"
    ;;
  judge-pilot)
    need_py
    "$PY" -m privacy_bench.judge.run pilot --limit "${1:-20}"
    ;;
  judge-calibrate)
    need_py
    "$PY" -m privacy_bench.judge.run calibrate "$@"
    ;;
  judge-score)
    need_py
    "$PY" -m privacy_bench.judge.run calibrate-score "$@"
    ;;
  judge-run)
    need_py
    "$PY" -m privacy_bench.judge.run run "$@"
    ;;
  registry)
    need_py
    "$PY" -m privacy_bench.registry.registry "$@"
    ;;
  mlflow-ui)
    need_py
    "$ROOT/.venv/bin/mlflow" ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
    ;;
  serve-prod)
    need_py
    if [[ $# -ne 1 ]]; then
      echo "usage: palr serve-prod <port>" >&2
      exit 1
    fi
    gguf="$("$PY" -m privacy_bench.registry.registry resolve production | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["gguf"])')"
    exec "$0" serve "$gguf" "$1"
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
