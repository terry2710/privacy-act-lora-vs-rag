import argparse
import json
from pathlib import Path

import yaml

from ..train.data import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the pinned base model and save a Hugging Face directory")
    parser.add_argument("--model", required=True, choices=["a0", "a1"])
    parser.add_argument("--config", default="configs/serve.yaml")
    args = parser.parse_args()

    import torch
    import transformers
    from packaging import version
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bm = cfg["base_model"]
    mc = cfg["models"][args.model]
    out = Path(mc["hf_dir"])
    out.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, bm["merge_dtype"])
    dtype_key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    tokenizer = AutoTokenizer.from_pretrained(bm["name"], revision=bm["revision"])
    model = AutoModelForCausalLM.from_pretrained(bm["name"], revision=bm["revision"], **{dtype_key: dtype})

    adapter_sha = None
    if mc["adapter"]:
        from peft import PeftModel

        adapter_dir = Path(mc["adapter"])
        adapter_sha = sha256_file(adapter_dir / "adapter_model.safetensors")
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()

    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    manifest = {
        "model": args.model,
        "base_model": bm["name"],
        "base_revision": bm["revision"],
        "merge_dtype": bm["merge_dtype"],
        "adapter": mc["adapter"],
        "adapter_sha256": adapter_sha,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "files": {p.name: sha256_file(p) for p in sorted(out.glob("*.safetensors"))},
    }
    (out / "merge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()