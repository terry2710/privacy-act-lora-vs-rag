import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from ..train.data import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a merged Hugging Face directory to GGUF with llama.cpp")
    parser.add_argument("--model", required=True, choices=["a0", "a1"])
    parser.add_argument("--config", default="configs/serve.yaml")
    parser.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP_DIR"))
    args = parser.parse_args()
    if not args.llama_cpp:
        sys.exit("set LLAMA_CPP_DIR or pass --llama-cpp with the llama.cpp source directory")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mc = cfg["models"][args.model]
    script = Path(args.llama_cpp) / "convert_hf_to_gguf.py"
    out = Path(mc["gguf"])
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), mc["hf_dir"], "--outtype", cfg["gguf"]["outtype"], "--outfile", str(out)]
    subprocess.run(cmd, check=True)
    commit = subprocess.run(["git", "-C", args.llama_cpp, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    record = {
        "model": args.model,
        "gguf": str(out),
        "outtype": cfg["gguf"]["outtype"],
        "bytes": out.stat().st_size,
        "sha256": sha256_file(out),
        "llama_cpp_commit": commit,
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()