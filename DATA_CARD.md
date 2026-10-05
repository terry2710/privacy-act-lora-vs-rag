# Data Card: Privacy Act 1988 QA Benchmark

Synthetic question–answer data built from the *Privacy Act 1988* (Cth) to compare LoRA fine-tuning and retrieval-augmented generation under a controlled split. All counts below are produced by the scripts in this repository and can be reproduced from the files under `data/`.

## Final artefacts

| File | Role | Items |
|---|---|---|
| `data/qa/final/train.jsonl` | LoRA training data | 1,013 facts × 2 phrasings = 2,026 QA pairs |
| `data/qa/eval/s_mem.jsonl` | Trained facts, trained phrasing (memorisation ceiling) | 150 |
| `data/qa/eval/s_para.jsonl` | Trained facts, held-out phrasings | 298 (lexical 98, syntactic 100, scenario 100) |
| `data/qa/eval/dev_para.jsonl` | Held-out phrasings for model selection only | 100 |
| `data/qa/eval/h1.jsonl` | Part IIIC, unseen in fine-tuning | 143 (48 in Division 5; 5 tagged parallel to s 44) |
| `data/qa/eval/h2.jsonl` | Schedule 2, unseen in fine-tuning and enacted after the base model's release | 76 |
| `data/qa/eval/s_neg.jsonl` | False-premise questions | 50 |

## Source

- *Privacy Act 1988* (Cth), Compilation No. 104 (C2026C00227), compilation date 4 June 2026, Federal Register of Legislation.
- The official DOCX is pinned by SHA-256 in `configs/split_manifest.yaml` and parsed against its table of contents; all 353 sections and clauses aligned with no skipped or empty headings.
- The same DOCX is the only source for both the QA data and the RAG index.

## Split design

| Unit | Role | Reason |
|---|---|---|
| Parts I–IIIB, IV–VI, VIB–IX; Schedule 1 | Training (287 sections) | Covers definitions, the APPs, credit reporting, codes, the Commissioner, investigations and enforcement |
| Part IIIC | Holdout H1 (28 sections) | Notifiable data breaches; unseen in fine-tuning, likely present in pre-training data |
| Schedule 2 | Holdout H2 (26 clauses) | Statutory tort; enacted after the base model's release |
| Part VIA | Excluded (11 sections) | Structurally mirrors Part IIIC Division 5 |
| s 94A | Excluded | Pointer provision to Schedule 2 |

Leakage controls applied before generation:

- 32 lines in 11 training sections that mention held-out content (for example the definition of *eligible data breach* in s 6, s 13(4A) and s 96(2A)–(2D)) are removed from the text shown to the generator. One flagged section (s 7) is a recorded false positive: it refers to Schedule 2 of the Freedom of Information Act 1982.
- Every generated training item is checked again for held-out identifiers and terms.

## Pipeline and counts

| Stage | Train | H1 | H2 |
|---|---|---|---|
| Quota (allocated by character count, guide sections down-weighted) | 1,076 | 150 | 80 |
| Generated and passing programmatic checks | 1,072 | 150 | 80 |
| Passing cross-model verification | 1,015 | 143 | 76 |
| After near-duplicate removal (cosine ≥ 0.97) | 1,013 | 143 | 76 |

Programmatic checks reject an item when its evidence is not a verbatim excerpt of the source, its questions contain a provision identifier or a pointer such as "this section", its answer breaks the citation format or opens with Yes/No for a non-polar question, or a training item mentions held-out content. Rejections during generation: yes/no mismatch 106, identifier in question 53, non-verbatim evidence 18, held-out mention 2, pointer 1.

Verification failures (68 in total): incomplete 48, unsupported 13, question mismatch 3, incorrect 2, trivial 2.

Training question types: scope 280, obligation 238, exception 164, definition 157, procedure 119, enforcement 55.

## Models and prompts

| Step | Model | Prompt version / hash |
|---|---|---|
| QA generation | `claude-sonnet-5-5` (default effort) | `qa-gen-v2` / `8020c1f91dd0` |
| Verification | `gpt-5.4-mini-2026-03-17` (reasoning effort low) | `qa-verify-v1` / `a7717183dc26` |
| Paraphrase test questions | `gpt-5.5-2026-04-23` | `para-v1` / `ec00382187c4` |
| Embeddings for deduplication and leakage audit | `text-embedding-3-small` | n/a |

The model that writes test paraphrases belongs to a different family from the model that wrote the training phrasings, so that test questions do not share the training data's stylistic fingerprint. Generation and verification also use different model families.

## Item format

Training items carry `fact_id`, `uid`, `display`, `unit`, `question_type`, `statement`, `questions` (two phrasings), `answer` ending in `[Source: <provision>]`, a verbatim `evidence` excerpt, and generation and verification metadata. Evaluation records carry `eval_id`, `set`, `question` and `reference_answer`, plus set-specific fields: `paraphrase_type` and overlap metrics for S-para and dev-para, `division`, `tagged` and `parallel_to_train` for H1, and `premise` and `expected_behaviour` for S-neg.

## Quality controls

- **Cross-model verification.** In a 95-item pilot, verification without reasoning wrongly failed two clearly correct items; enabling low reasoning effort removed those errors. On review of 20 of the 68 final failures, roughly half were judged acceptable, so the verifier errs towards strictness. This discards some valid data but does not admit invalid data.
- **Paraphrase separation.** Held-out phrasings must stay at or below a 4-gram Jaccard of 0.5 and a ROUGE-L of 0.6 against every training phrasing of the same fact (means 0.049 and 0.38). The ROUGE-L limit was raised from 0.5 because unavoidable multi-word legal terms inflate token overlap.
- **Semantic leakage audit.** Every H1 and H2 question was compared with all training questions by embedding similarity. Only one pair exceeded the 0.85 review threshold, but provision-level review then showed that five s 26WU items in H1 restate s 44 in training. These items are tagged `parallel_to_train`, excluded from the headline H1 metric and reported separately. H2's highest similarity to training was 0.77, all thematic.
- **S-neg.** Fifty false-premise questions were written by hand. Twenty that cite non-existent provisions are validated programmatically against the parsed table of contents. The remaining thirty (foreign-law concepts and fabricated mechanisms) are covered by the human spot check. Topics under active law-reform discussion were deliberately avoided.
- **Human spot check.** A blind, stratified sample of 120 items was fixed and committed before labelling. Results: *to be added after labelling* (`data/audit/spot_check/results.json`).

## Known limitations

1. The QA pairs are synthetic. Coverage is limited to the facts the generator extracted, and the answer-length rule (one to three sentences) trades completeness for concision; incompleteness is the main verification failure.
2. The verifier is a small model and is stricter than a human reviewer; residual error rates are estimated by the human spot check rather than assumed.
3. Very short provisions could not always yield further non-duplicate facts: 4 training facts and 2 S-para paraphrases are missing against quota.
4. Leakage controls are lexical and semantic, but parallel provisions with different wording can fall below any similarity threshold; the s 26WU / s 44 case was found by provision-level review.
5. Some scenario paraphrases state conditions from the answer in order to set the scene, which makes them easier than direct questions about those conditions.
6. The H1/H2 contrast assumes that Part IIIC text appears in the base model's pre-training data and Schedule 2 text does not. This is tested empirically by the base model's zero-shot scores rather than assumed.
7. The data reflect one compilation of one Act. They are not legal advice and will drift as the law changes.

## API usage

| Step | Provider | Input tokens | Output tokens | Calls |
|---|---|---|---|---|
| Generation (including top-up) | Anthropic | 924,454 | 396,844 | 477 |
| Verification | OpenAI | 400,874 | 100,675 | 351 |
| Paraphrases | OpenAI | 68,805 | 51,837 | 50 |

Embedding calls are negligible. Pilot runs under earlier prompt versions are not included above.

## Licence

Legislative text is from the Federal Register of Legislation and is used with attribution under its licence terms; verify the current terms before redistribution.

## Reproduction

    palr parse
    palr split
    palr gen
    palr verify
    palr eval-build
    palr para
    palr spot-export

Every API step caches its results under a hash of the prompt and model, so re-running a step reuses earlier calls.