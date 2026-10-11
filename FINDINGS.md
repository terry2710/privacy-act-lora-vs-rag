# Findings

This document reports the results of the pre-registered comparison described in `EXPERIMENT_PLAN.md`. Every number below is produced by `scripts/palr.sh analyze` from the committed answers and judgements, and is recorded in `results/phase3/analysis.json` and `results/phase3/analysis.md`. Accuracy is the primary judge's 0–2 score unless stated otherwise; intervals are 95% paired bootstrap intervals over items with 10,000 resamples.

## Summary

1. **Fine-tuning memorises, and most of its advantage disappears when the question is rephrased.** On trained facts asked in the trained phrasing, the LoRA model (A1) is almost perfect (1.99 of 2). On the same facts asked in held-out phrasings it scores 1.27, statistically indistinguishable from the base model with retrieval (A2, 1.22). The pre-registered ordering A0 < A1 < A2 on paraphrases is therefore not supported: A1 and A2 tie.
2. **Fine-tuning suppresses the use of retrieved text.** On provisions never seen in fine-tuning, giving the LoRA model the same retrieved extracts as the base model (A3) recovers little: A3 scores 0.44 on Part IIIC and 0.45 on Schedule 2, against 1.18 and 1.26 for A2, although both arms receive identical extracts that contain the relevant provision about 90% of the time.
3. **Fine-tuning teaches the citation format without the knowledge behind it.** On unseen provisions, A1 attaches a `[Source: …]` tag to every answer and every tag is wrong, against 77% wrong for the base model, which omits the tag on about a quarter of answers. When the base model does tag an answer, it is also always wrong; fine-tuning changes how often the model asserts a provision, not whether it knows it.
4. **Neither fine-tuned arm ever corrects a false premise.** On 50 questions about provisions the Act does not contain, A1 and A3 corrected the premise 0 times and described the non-existent provision in 80–98% of answers. The base model with retrieval corrected 22%.

All arm rankings are identical under a second judge from a different model family, which scored every answer as a registered sensitivity analysis.

## Arms and sets

| Arm | Model | Retrieval |
|---|---|---|
| A0 | Qwen2.5-1.5B-Instruct | None |
| A1 | + LoRA (all-linear, rank 64, two phrasings per fact) | None |
| A2 | Qwen2.5-1.5B-Instruct | Hybrid BM25 + dense, top 8 extracts |
| A3 | + LoRA | Hybrid BM25 + dense, top 8 extracts |

| Set | Items | What it measures |
|---|---|---|
| S-mem | 150 | Trained facts, trained phrasing |
| S-para | 298 | Trained facts, held-out phrasings |
| H1-clean | 138 | Part IIIC, never seen in fine-tuning |
| H1-parallel | 5 | Part IIIC items mirroring a trained provision (s 44) |
| H2 | 76 | Schedule 2, never seen in fine-tuning and enacted after the base model's release |
| S-neg | 50 | Questions built on a false premise |

## Accuracy

Mean accuracy on a 0–2 scale; S-neg is the share of answers that correct the premise without inventing content.

| Set | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| S-mem | 0.29 | **1.99** | 1.36 | 1.80 |
| S-para | 0.33 | 1.27 | 1.22 | **1.30** |
| H1-clean | 0.30 | 0.22 | **1.18** | 0.44 |
| H2 | 0.34 | 0.36 | **1.26** | 0.45 |
| S-neg (pass rate) | 10% | 0% | **22%** | 0% |

## Hypotheses

| Hypothesis | Type | Result | Evidence |
|---|---|---|---|
| H1: on S-para, A0 < A1 < A2 | Primary | **Not supported** | A1 − A0 = +0.94 [+0.83, +1.05], Holm p < 0.001; A2 − A1 = −0.06 [−0.18, +0.07], Holm p = 0.38 |
| H3: on H1-clean and H2, A1's wrong or non-existent citation rate exceeds A0's | Primary | **Supported** | 100% against 77%, difference +0.23 [+0.18, +0.29], McNemar Holm p < 10⁻¹⁴ |
| H2: A1's memorisation gap is positive | Secondary | **Supported** | S-mem − S-para per fact = +0.73 [+0.62, +0.84]; the comparison with the one-phrasing ablation was not tested |
| H4: A0 scores higher on H1-clean than on H2 | Secondary | **Not supported** | −0.04 [−0.20, +0.12] |
| H5: A2 is slower than A1, mainly through prefill rather than retrieval | Secondary | **Supported, with qualification** | Median total +1.20 s [+1.13, +1.26]; mean components: retrieval +22 ms, prefill +375 ms, generation +793 ms |
| H6: on H1-parallel, A1 cites s 44 rather than s 26WU | Exploratory | Partly | s 44 on 3 of 5 items; s 26WU on none |
| H7: on S-neg, A2 corrects premises more often than A1 | Exploratory | Supported | 22% against 0%, McNemar p = 0.001 |

The primary family (H1's two comparisons and H3) is corrected with Holm–Bonferroni; secondary and exploratory results are uncorrected.

## Finding 1: fine-tuning memorises phrasings more than facts

A1 answers trained questions almost perfectly, but its score falls by 0.73 points when the same fact is asked differently. The gap is close to zero for the base model (−0.04) and small for the base model with retrieval (+0.15), so it is a property of fine-tuning rather than of the question sets.

On S-para, the fine-tuned model and the retrieval model reach the same accuracy (1.27 against 1.22, difference −0.06 [−0.18, +0.07]), and the second judge agrees (−0.04 [−0.16, +0.08]). Combining the two (A3, 1.30) adds nothing measurable over either alone. For facts inside the training data, retrieval over the full Act therefore matches a fine-tuned model once questions are phrased the way real users phrase them, without any training.

This is consistent with the training record: during selection, training on two phrasings per fact rather than one raised dev-para citation hit from 71% to 86%, and an exposure-matched control recovered only part of that gain (`MODEL_SELECTION.md`). Phrasing diversity, not repetition, is what generalises.

## Finding 2: fine-tuning suppresses the use of retrieved text

| | H1-clean | H2 |
|---|---|---|
| Recall@8 of the retriever (both arms) | 0.90 | 0.93 |
| A2 accuracy | 1.18 | 1.26 |
| A3 accuracy | 0.44 | 0.45 |
| A3 − A2 | −0.74 [−0.89, −0.59] | −0.82 [−1.03, −0.61] |
| A2 citation hit | 12% | 26% |
| A3 citation hit | 15% | 0% |

The two arms receive the same extracts, so the difference lies in how the model uses them. Retrieval lifts the fine-tuned model only slightly on unseen provisions (A3 − A1 = +0.23 on H1-clean and +0.09 on H2, the latter not significant). On Schedule 2, which postdates the base model, A3 never cites the correct clause even though it is retrieved for 93% of questions, while A2 cites it for 26%.

The same pattern appears, more weakly, on trained facts. Adding retrieval to the fine-tuned model lowers S-mem accuracy from 1.99 to 1.80 and S-para citation hit from 86% to 80% (McNemar p = 0.03): the model answers from its weights and the extracts act as noise rather than evidence.

A plausible mechanism is a mismatch between training and use. The adapter was trained closed-book, with no extracts in the prompt, so it learned to answer every question from memory. Training on prompts that include retrieved extracts, some relevant and some not, is the obvious way to test this, and was not part of this study.

## Finding 3: the citation format without the knowledge

| Set | Arm | Tag present | Correct citation | Wrong or non-existent | Wrong or non-existent, when tagged |
|---|---|---|---|---|---|
| H1-clean | A0 | 78% | 0% | 80% | 100% |
| H1-clean | A1 | 100% | 0% | 100% | 100% |
| H1-clean | A2 | 24% | 12% | 12% | 48% |
| H2 | A0 | 71% | 0% | 71% | 100% |
| H2 | A1 | 100% | 0% | 100% | 100% |
| H2 | A2 | 37% | 26% | 12% | 29% |

H3 is supported, but the table shows what drives it. Conditional on producing a tag, the base model and the fine-tuned model are equally wrong on unseen provisions. Fine-tuning raised the tagging rate to 100%, so every answer about an unseen provision now carries a confident, wrong citation. A citation from the fine-tuned model is therefore not evidence that it knows the provision.

The five H1-parallel items show the same behaviour in a sharper form: three of five A1 answers cite s 44, the trained provision whose wording the questions mirror, and none cites the correct s 26WU.

The base model with retrieval tags only a quarter to a third of its answers, but its tags are far more often correct. Measured by whether the reference provision appears anywhere in the answer, A2 reaches 51% on H1-clean and 60% on S-para, so tag-based citation hit understates its grounding.

## Finding 4: false premises

| Arm | Premise corrected without invention | Describes the non-existent provision |
|---|---|---|
| A0 | 10% | 90% |
| A1 | 0% | 98% |
| A2 | 22% | 62% |
| A3 | 0% | 80% |

Both fine-tuned arms answered every false-premise question as if the provision existed. The training data contained no questions about provisions the Act lacks, so the adapter learned that every question has an answer in the Act. Retrieval helps the base model, presumably because the absence of a matching extract is itself a signal, but a 1.5B model rarely uses that signal: even A2 describes the non-existent provision in most answers.

## Latency

Medians and 95th percentiles of total server-side time on S-para, on an Apple silicon Mac mini with Q8_0 GGUF models served by llama.cpp.

| Arm | Median | p95 | Median answer length (tokens) |
|---|---|---|---|
| A0 | 1.13 s | 2.71 s | 86 |
| A1 | 0.54 s | 0.84 s | 41 |
| A2 | 1.82 s | 3.41 s | 94 |
| A3 | 1.13 s | 1.58 s | 42 |

H5 holds in the sense registered: prefill of the extracts (about 375 ms) costs far more than retrieval itself (about 20 ms). The largest component of the A2–A1 difference, however, is generation (about 790 ms), because the base model writes answers twice as long as the fine-tuned model. Fine-tuning made answers shorter and more uniform, which halves latency independently of retrieval.

## Judge reliability

- **Rubric**: accuracy 0, 1 or 2 against a reference answer and an excerpt of the provision; citations are removed before judging so the judge cannot see which arm produced an answer, and citation accuracy is scored separately by program.
- **Calibration**: on 150 answers from A0, A1 and A2, the primary judge (GPT-5.4 mini) and a second judge (Claude Opus 5.5) agreed at quadratic-weighted κ 0.92 on accuracy and κ 0.78 on S-neg, above the pre-registered threshold of 0.6. The author adjudicated the 16 disagreements; against the adjudicated labels the primary judge agreed on 89.7% of accuracy scores and scored A0 and A1 about 0.07 points above them.
- **Sensitivity**: because that bias differed between arms, the second judge also scored all 2,868 answers. Every comparison reported above has the same sign and the same significance under both judges; the second judge is stricter on low-scoring answers (for example A0 on H2: 0.18 against 0.34) and close to the primary judge on high-scoring ones.
- **Agreement on the full run**: weighted κ 0.85–0.93 across the QA sets and raw agreement 95.5% on S-neg, where κ is 0.64 because almost every answer fails.

## Limitations

- **One model and one domain.** Results are for a 1.5B model and a single statute; larger models may use retrieved text differently and may correct false premises more often.
- **Closed-book fine-tuning.** The adapter never saw retrieved extracts in training, which may explain Finding 2. The comparison between A3 and A2 measures this training recipe, not fine-tuning in general.
- **No negative training examples.** Finding 4 may largely reflect the absence of unanswerable questions from the training data.
- **Synthetic questions.** Questions and reference answers were generated by one LLM family and verified by another, with a human-adjudicated spot check; they are not drawn from real user queries.
- **LLM judging.** Accuracy rests on LLM judges. The calibration and the second-judge analysis bound this risk but do not remove it. The author is not legally trained; disagreements were adjudicated against the reference answer and the provision text, with AI assistance in locating and quoting provisions. The assisting model belongs to the same family as the second judge.
- **H2 incomplete.** The comparison of memorisation gaps with the one-phrasing ablation was not run, because the ablation was evaluated only on dev-para during selection.
- **Small subsets.** H1-parallel has five items and S-neg fifty; their results are descriptive.

## Implications

- For a fixed body of text that users query in their own words, retrieval over that text is the stronger default: it matches a fine-tuned model on trained facts, far exceeds it on anything outside the training data, needs no training and stays current when the text changes.
- Fine-tuning still buys format and speed: every answer carries a citation and answers are about half as long and twice as fast. Those benefits come with confident wrong citations outside the training data and no ability to refuse.
- Combining the two is not free. A model fine-tuned closed-book learned to ignore its context; a combined system needs training data that includes retrieved extracts and unanswerable questions.

## Reproduction

    scripts/palr.sh analyze

Inputs are the committed answers under `results/phase3/answers/` and judgements under `results/phase3/judgements/`; the analysis configuration is `configs/analysis.yaml`. The pipeline that produced the inputs is described in `README.md`.
