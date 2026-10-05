# privacy-act-lora-vs-rag

A controlled comparison of LoRA fine-tuning and retrieval-augmented generation (RAG) for question answering over the Australian *Privacy Act 1988* (Cth), using Qwen2.5-1.5B-Instruct.

**Status:** Phase 1 — dataset construction (work in progress).

## Design summary

- Training data is generated only from selected Parts of the Act. Part IIIC (Notification of eligible data breaches) and Schedule 2 (Statutory tort) are held out entirely as unseen-provision test sets; Part VIA is excluded because it structurally mirrors Part IIIC Division 5.
- A separate paraphrase test set uses held-out phrasings of trained facts to distinguish learning from memorisation.
- Four arms are evaluated: base model zero-shot, LoRA, base + RAG, and LoRA + RAG.
- Experiments are tracked in Weights & Biases; evaluation is orchestrated with n8n.

The full split specification is in `configs/split_manifest.yaml`.

## Reproducing Phase 1

    pip install -e ".[dev]"
    pytest
    python -m privacy_bench.data.parse_act --toc data/raw/toc.txt --docx data/raw/privacy_act_1988.docx --out data/processed/sections.jsonl --report data/processed/parse_report.json

## Source and licence

Legislative text is sourced from the Federal Register of Legislation (legislation.gov.au): *Privacy Act 1988* (Cth), Compilation No. 104 (C2026C00227), compilation date 4 June 2026. The file checksum is recorded in `configs/split_manifest.yaml`. Both the LoRA training data and the RAG index are built from this single source file.
