# Experiment Plan (pre-registered)

This plan was committed before any fine-tuning run or evaluation of any arm. Changes made after this commit are recorded under *Deviations* with a date and a reason; the original text is not edited.

## Question

On a fixed, verified QA benchmark built from the *Privacy Act 1988* (Cth), what do LoRA fine-tuning and retrieval-augmented generation each contribute to accuracy, faithful citation and latency for a 1.5B-parameter instruction model, on facts it was trained on, on held-out phrasings of those facts, and on provisions it never saw?

## Arms

| Arm | Model | Retrieval |
|---|---|---|
| A0 | Qwen2.5-1.5B-Instruct | none |
| A1 | Qwen2.5-1.5B-Instruct + LoRA | none |
| A2 | Qwen2.5-1.5B-Instruct | top-k chunks from the full Act |
| A3 (optional) | Qwen2.5-1.5B-Instruct + LoRA | top-k chunks from the full Act |

All arms share one system prompt, chat template, decoding configuration (greedy where the runtime supports it, maximum 256 new tokens), quantisation level and hardware. The RAG index contains the whole Act, including H1 and H2; access to text absent from the weights is the property under test.

## Evaluation sets

| Set | Role |
|---|---|
| S-mem (150) | Trained facts, trained phrasing |
| S-para (298) | Trained facts, held-out phrasing; paired with S-mem by fact |
| dev-para (100) | Model selection only; never used for reported results |
| H1-clean (138) | Part IIIC excluding the five items tagged `parallel_to_train` |
| H1-parallel (5) | Reported separately |
| H2 (76) | Schedule 2 |
| S-neg (50) | False premises |

## Metrics

- **Accuracy**: 0 (wrong), 1 (partly correct), 2 (correct and complete), scored by an LLM judge against the reference answer.
- **Groundedness and fabricated content**: judged per answer.
- **Citation hit**: the provision in the answer's `[Source: …]` tag equals the reference provision (programmatic).
- **Non-existent citation rate**: any provision cited, in the tag or in the text, that does not exist in the parsed table of contents (programmatic).
- **Wrong-existing citation rate**: the tagged provision exists but is not the reference provision.
- **Premise correction (S-neg)**: the answer states that the Act contains no such provision or mechanism and does not invent one.
- **Latency**: server-side retrieval, prefill, generation and total time; reported as median and 95th percentile.
- **Memorisation gap**: Δmem = Acc(S-mem) − Acc(S-para), computed over facts present in both sets.

## Hypotheses

Primary:

- **H1.** On S-para, accuracy orders as A0 < A1 < A2.
- **H3.** On H1-clean and H2 combined, A1's rate of non-existent or wrong-existing citations exceeds A0's: fine-tuning teaches the citation format without the knowledge behind it.

Secondary:

- **H2.** A1's memorisation gap is positive, and is larger for the one-phrasing ablation than for the two-phrasing model.
- **H4.** A0 scores higher on H1-clean than on H2, reflecting pre-training exposure to Part IIIC.
- **H5.** A2's total latency exceeds A1's, and the difference is dominated by prefill of the retrieved context rather than by retrieval itself.

Exploratory (reported descriptively):

- **H6.** On H1-parallel, A1 reproduces the s 44 content but cites s 44 rather than s 26WU.
- **H7.** On S-neg, A2 corrects false premises more often than A1.
- A3 relative to A1 and A2 on every set.

## Model selection

LoRA hyperparameters are chosen by citation-hit rate on dev-para, with validation loss as a secondary signal. No evaluation set other than dev-para is scored before the final configuration is fixed.

## Judging

The primary judge belongs to a different model family from the QA generator. A second judge from another family scores a stratified calibration sample of about 150 answers, drawn first from A0 and A2 and topped up with A1 answers once the fine-tuned model exists. The full run proceeds only if inter-judge Cohen's κ ≥ 0.6 on that sample. The author adjudicates the disagreements, and the adjudicated sample is used to estimate the primary judge's error rate. If κ < 0.6, the rubric is revised and recalibrated, and the revision is recorded under *Deviations*. The same dual-review procedure, with author adjudication of disagreements, is used for the data spot check described in `DATA_CARD.md`.

## Statistics

- Paired bootstrap over items (10,000 resamples) for accuracy differences, with 95% confidence intervals.
- McNemar's test for binary outcomes (citation hit, premise correction).
- Holm–Bonferroni correction across the primary hypotheses; secondary and exploratory results are reported without correction and labelled as such.
- A hypothesis is supported when the 95% interval of the relevant difference excludes zero in the predicted direction, after correction where it applies.

## Model registry and promotion

The served model is versioned and promoted through an MLflow Model Registry backed by a local tracking server with a SQLite store. W&B remains the record of training runs and sweeps; MLflow holds deployable model versions only, and no training run is logged to both.

- **Registered model**: `palr-privacy-act-qa`. Version 1 is A0, the base model converted to GGUF, and is the initial `production` version. Each version stores the GGUF file served by llama.cpp, with tags for the W&B run ID, the bundle git commit, the adapter and GGUF SHA-256 digests, the quantisation level and the system-prompt hash.
- **Aliases**: a new version is registered with the alias `candidate`. The serving layer resolves only the alias `production`.
- **Promotion gate**: the n8n pipeline moves `production` to a candidate only if all of the following hold, measured on dev-para through the serving endpoint:
  1. *Conversion parity*: the candidate's citation-hit rate is within 3 percentage points of the PEFT model it was converted from, and the two agree on citation hit for at least 90% of items.
  2. *Quality floor*: the candidate's citation-hit rate is strictly higher than that of the current `production` version.
  3. *Format compliance*: a `[Source: …]` tag is present in at least 95% of the candidate's answers.
  4. *Operational*: no request fails or times out, and the candidate's 95th-percentile total latency is at most 1.2 times that of the current `production` version.
- **Rollback**: rollback moves `production` back to the previous version. It is exercised once and the drill is recorded.
- **Separation from the study**: the gate uses dev-para only, consistent with *Model selection*. A failed gate is recorded and does not lead to further tuning against any evaluation set. Every arm is evaluated and reported regardless of the promotion outcome.

## Deviations

- **2026-10-06, before any training run.** Added *Model registry and promotion*. Reason: to cover versioning, gated promotion and rollback of the served model. The addition does not change the arms, evaluation sets, metrics, hypotheses, model selection or statistics.
- **2026-10-06, after the first baseline run and before any sweep.** Added a training-fit probe: 100 training facts, excluding every fact in S-mem and dev-para, asked with their first trained phrasing and scored with the same citation metrics. Reason: to tell under-fitting apart from failure to generalise across phrasings. The probe is diagnostic only; model selection still uses dev-para citation hit alone.
- **2026-10-08, after the final configuration was fixed.** Added an exposure-matched control for the one-phrasing ablation: one phrasing for 16 epochs, three seeds, so that each fact is seen as often and the optimiser takes as many steps as in the two-phrasing model. Reason: the 8-epoch ablation confounds phrasing diversity with exposure count. Exploratory; it does not change arm A1 or any primary hypothesis.
- **2026-10-08, before any evaluation set other than dev-para was scored.** Added an exploratory *mention hit* metric: the reference provision appears anywhere in the answer, in the `[Source: …]` tag or in the text. Reason: a smoke test of the answer service showed the base model with retrieved extracts answering without the tag, so tag-based citation hit alone mixes format compliance with knowledge of the source. Citation hit remains the pre-registered metric; mention hit and tag presence are reported alongside it.
- **2026-10-09, before any answer was judged.** Specified how the judging gate is computed. Accuracy agreement uses quadratic-weighted Cohen's κ, because the 0/1/2 scale is ordinal; unweighted κ is reported alongside it. S-neg agreement uses binary κ on pass (premise corrected and no invented content); if both judges give a single class, so that κ is undefined, raw agreement of at least 0.9 is required instead. Groundedness is operationalised as a per-answer flag for any statement that contradicts the reference answer or the provision excerpt. The calibration sample is 50 answers from each of A0, A1 and A2. The thresholds, arms and sets are otherwise as planned.
- **2026-10-11, after judge calibration and before the full judging run.** Added a judge sensitivity analysis: the secondary judge also scores every answer, and the arm comparisons are repeated on its scores. Reason: on the adjudicated calibration sample the primary judge scored A0 and A1 about 0.07 above the final labels and A2 slightly below them, so it may understate differences between arms. The primary judge's scores remain the pre-registered result; the secondary judge's scores are exploratory and reported alongside them.
- 
