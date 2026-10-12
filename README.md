# privacy-act-lora-vs-rag

A controlled comparison of LoRA fine-tuning and retrieval-augmented generation (RAG) for question answering over the Australian *Privacy Act 1988* (Cth), using Qwen2.5-1.5B-Instruct. The study asks when fine-tuning helps, when retrieval helps, and what happens when the two are combined, using test sets that separate memorisation, generalisation across phrasings, unseen provisions and false premises.

## Results

- **Fine-tuning memorises phrasings more than facts.** The LoRA model scores 1.99 of 2 on trained questions but 1.27 when the same facts are rephrased, statistically tied with the base model plus retrieval (1.22).
- **Fine-tuning suppresses the use of retrieved text.** Given the same retrieved extracts as the base model, the fine-tuned model scores 0.74–0.82 points lower on provisions it was not trained on.
- **Fine-tuning teaches the citation format, not the knowledge.** Outside its training data the fine-tuned model cites a provision in every answer and every citation is wrong; it never corrects a question about a provision that does not exist.

| Set | A0 base | A1 LoRA | A2 base + RAG | A3 LoRA + RAG |
|---|---|---|---|---|
| S-mem | 0.29 | **1.99** | 1.36 | 1.80 |
| S-para | 0.33 | 1.27 | 1.22 | **1.30** |
| H1-clean | 0.30 | 0.22 | **1.18** | 0.44 |
| H2 | 0.34 | 0.36 | **1.26** | 0.45 |
| S-neg, premise corrected | 10% | 0% | **22%** | 0% |

Mean accuracy on a 0–2 scale from a calibrated LLM judge; every ranking holds under a second judge from another model family. Hypotheses, intervals, latency and limitations are in `FINDINGS.md`. The [W&B report](https://forge.coreweave.com/wandb/arlives/privacy-act-lora-vs-rag/reports/LoRA-vs-RAG-on-the-Australian-Privacy-Act-a-pre-registered-comparison--VmlldzoxODA5NDM1NA?accessToken=l3hvlmabr1k96qiu3tmid3jkr0lcp6vzabrmtpmr49fcs4xyc599777lb286686d) shows the hyperparameter search, the evaluation tables and the promotion gate.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Dataset construction, split design, QA generation and verification, human spot check | Complete |
| 2 | LoRA training on Kaggle, staged hyperparameter search tracked in Weights & Biases, seeds and ablations | Complete |
| 3 | GGUF conversion, retrieval, answer service, model registry, calibrated LLM judging, statistical analysis, n8n orchestration | Complete |
| 4 | Findings, model card, public W&B report | In progress |

## Design

The experiment plan, including hypotheses, metrics and statistics, was fixed before training and is recorded in `EXPERIMENT_PLAN.md`; every later change is logged under *Deviations* with its date and reason.

| Arm | Model | Retrieval |
|---|---|---|
| A0 | Base model | None |
| A1 | LoRA fine-tuned model | None |
| A2 | Base model | Top 8 retrieved extracts |
| A3 | LoRA fine-tuned model | Top 8 retrieved extracts |

| Set | Items | Purpose |
|---|---|---|
| S-mem | 150 | Trained facts, trained phrasing: memorisation ceiling |
| S-para | 298 | Trained facts, held-out phrasings: generalisation across phrasings |
| H1 | 143 | Part IIIC, never seen in fine-tuning |
| H2 | 76 | Schedule 2, never seen in fine-tuning and enacted after the base model's release |
| S-neg | 50 | False-premise questions about provisions the Act does not contain |
| dev-para | 100 | Held-out phrasings used only for model selection |

Part VIA is excluded from all sets because it structurally mirrors Part IIIC Division 5. The full split specification is in `configs/split_manifest.yaml` and the data pipeline is documented in `DATA_CARD.md`.

## Pipeline

1. **Data.** The official DOCX is parsed into 353 sections and clauses. 1,013 facts from the training Parts are turned into question–answer pairs with two phrasings each by one model family and verified by another; evaluation sets are deduplicated and audited for semantic leakage, and a stratified sample was checked by two LLM reviewers with human adjudication of disagreements.
2. **Training.** LoRA adapters are trained on Kaggle T4 GPUs with a custom fp16 loop. A Bayesian search followed by a grid selected all-linear LoRA with rank 64, alpha 128, learning rate 5e-4, 8 epochs and two phrasings per fact, chosen on dev-para citation hit alone (85%). Across three seeds the selected configuration reached 85.7% ± 2.1, against 71.0% ± 4.6 with one phrasing and 76.0% ± 3.0 for an exposure-matched one-phrasing control. The selection record is in `MODEL_SELECTION.md`.
3. **Serving.** The adapter is merged into the pinned base model and converted to Q8_0 GGUF with llama.cpp. The GGUF model matches the PEFT model on dev-para (86% against 85% citation hit, 95% item agreement) and answers with a median latency of 0.59 s on an Apple silicon Mac mini. A FastAPI service exposes all four arms behind one endpoint and reports retrieval, prefill and generation latency separately.
4. **Retrieval.** The Act is split into 669 section-bounded chunks. A hybrid of BM25 and bge-small-en-v1.5 dense retrieval fused by reciprocal rank reached recall@5 of 95%, recall@8 of 100% and MRR of 0.808 on dev-para, with about 6 ms per query.
5. **Registry.** Model versions are registered in MLflow with a promotion gate covering conversion parity, a quality floor, format compliance and latency. The fine-tuned model passed the gate against the base model (citation hit 86% against 7%, p95 latency 889 ms against 3,047 ms) and holds the `production` alias; a rollback drill is recorded.
6. **Judging and analysis.** Accuracy is scored 0, 1 or 2 against a reference answer and the provision text, and S-neg answers are scored on whether they correct the false premise without inventing content. The primary judge (GPT-5.4 mini) is from a different model family from the QA generator. On a 150-answer calibration sample it agreed with a second judge (Claude Opus 5.5) at weighted κ 0.92 on accuracy and κ 0.78 on S-neg, and with the adjudicated labels on 89.7% of accuracy scores. The second judge also scored all 2,868 answers as a registered sensitivity analysis. Hypotheses are tested with paired bootstrap intervals, McNemar's test and Holm–Bonferroni correction.
7. **Orchestration.** n8n, running in Docker, drives two workflows through a local job service that runs only allow-listed pipeline steps behind a bearer token: an evaluation workflow from answer collection through judging and analysis to a W&B run, and a promotion workflow that gates a staged candidate on dev-para and promotes it only if every check passes.

## Repository layout

| Path | Contents |
|---|---|
| `src/privacy_bench/data`, `qa` | Parsing, splitting, QA generation, verification, evaluation set building and audits |
| `src/privacy_bench/train` | Training data, LoRA training loop, citation scoring and Kaggle bundle |
| `src/privacy_bench/serve` | Adapter merge, GGUF conversion, parity check and the FastAPI answer service |
| `src/privacy_bench/rag` | Chunking, BM25, dense and hybrid retrieval, retrieval evaluation |
| `src/privacy_bench/registry` | MLflow registration, staging, promotion gate and rollback |
| `src/privacy_bench/evaluate`, `judge` | Answer collection, judge rubrics, calibration, scoring, W&B logging and the W&B report |
| `src/privacy_bench/analysis` | Pre-registered statistical analysis |
| `src/privacy_bench/ops` | Job service used by the n8n workflows |
| `n8n/` | Docker Compose file and the evaluation, promotion and job workflows |
| `configs/` | Every configuration file, including sweeps and the split manifest |
| `data/` | Parsed Act, QA data and evaluation sets |
| `results/` | Selection, parity, retrieval, registry, answers, calibration, judgements and analysis |

## Reproduction

    python -m venv .venv
    .venv/bin/pip install -e ".[dev]"
    scripts/palr.sh test

`scripts/palr.sh help` lists every pipeline step. The main stages after data construction are:

    scripts/palr.sh bundle --owner <kaggle username>
    scripts/palr.sh merge a1
    scripts/palr.sh gguf a1
    scripts/palr.sh rag-index
    scripts/palr.sh serve <base gguf> 8080
    scripts/palr.sh serve <lora gguf> 8081
    scripts/palr.sh api
    scripts/palr.sh answers
    scripts/palr.sh judge-calibrate
    scripts/palr.sh judge-score
    scripts/palr.sh judge-run
    scripts/palr.sh analyze

To run the same steps from n8n:

    scripts/palr.sh ops
    scripts/palr.sh n8n-up

Training runs on Kaggle from the bundle; GGUF conversion needs llama.cpp built locally with `LLAMA_CPP_DIR` set. API keys and the job-service token are read from a local `.env` file, which is not committed.

## Documents

- `FINDINGS.md`: results, hypothesis tests, latency, judge reliability and limitations
- `MODEL_CARD.md`: the production model, its training, evaluation and known failure modes
- `EXPERIMENT_PLAN.md`: pre-registered design, metrics, statistics and deviations
- `DATA_CARD.md`: data sources, split, pipeline, quality controls and limitations
- `MODEL_SELECTION.md`: hyperparameter search, selected configuration, seeds and ablations

## Source and licence

The code is released under the MIT License (see `LICENSE`). The legislative text and the data derived from it are licensed separately, as set out below.

Legislative text is sourced from the Federal Register of Legislation (legislation.gov.au): *Privacy Act 1988* (Cth), Compilation No. 104 (C2026C00227), compilation date 4 June 2026. The file checksum is recorded in `configs/split_manifest.yaml`. Both the LoRA training data and the RAG index are built from this single source file.

`data/raw/privacy_act_1988.docx` is sourced from the Federal Register of Legislation at 2 October 2026. The questions, answers, splits and retrieval index built from it are based on content from the Federal Register of Legislation at 2 October 2026. For the latest information on Australian Government law please go to https://www.legislation.gov.au. This content is licensed under the [Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/). The Federal Register of Legislation does not endorse this project.
