# Model Selection Record

Recorded on 2026-10-07, before any evaluation set other than dev-para was scored.

## Procedure

- Stage 1: Bayesian sweep over learning rate (1e-4 to 1e-3, log-uniform), LoRA rank (8, 16, 32, 64) and target modules (attention, all linear layers) at three epochs. Ten runs completed and two were lost to a terminated Kaggle session. All-linear targets dominated, and dev-para citation hit tracked training-set fit almost monotonically; three epochs left every configuration under-fitted.
- Stage 2: grid over rank (16, 32, 64) and epochs (5, 8) with all-linear targets and learning rate 5e-4. The planned epoch set of 2, 3 and 5 was replaced by 5 and 8 because stage 1 showed three epochs to be insufficient.
- Selection rule: highest dev-para citation hit; configurations within 3 percentage points of the best are compared on dev-para loss.

## Run-to-run variance

The configuration rank 32, 5 epochs was trained twice with the same seed, data order and hyperparameters and scored 86% and 79% dev-para citation hit. Non-deterministic GPU kernels make repeated runs diverge once the model fits the training set closely, so differences of a few points between top configurations are not meaningful on their own.

## Selected configuration

| Setting | Value |
|---|---|
| Target modules | all linear layers |
| Rank / alpha | 64 / 128 |
| Learning rate | 5e-4 |
| Epochs | 8 |
| Phrasings per fact | 2 |
| Seed | 20261005 |
| W&B run | fresh-sweep-6 (sweep fb3xh2wj) |
| Dev-para citation hit | 85% |
| Dev-para loss | 0.157 |

Rank 32 with 8 epochs scored 87% citation hit and lies within 3 points; rank 64 with 8 epochs is selected on lower dev-para loss.

## Commitments

- Arm A1 is the fresh-sweep-6 adapter. Additional seeds estimate training variance and do not replace it.
- The one-phrasing ablation uses the same configuration with three seeds.

## Stage 3: seeds and one-phrasing ablation (added 2026-10-08)

Selected configuration on dev-para, three seeds per condition.

| Phrasings | Seed | Citation hit | Dev loss | Probe hit |
|---|---|---|---|---|
| 2 | 20261005 (A1) | 85% | 0.157 | 100% |
| 2 | 20261006 | 84% | 0.151 | 100% |
| 2 | 20261007 | 88% | 0.150 | 100% |
| 1 | 20261005 | 67% | 0.431 | 100% |
| 1 | 20261006 | 70% | 0.431 | 100% |
| 1 | 20261007 | 76% | 0.444 | 99% |

- Two phrasings: 85.7% ± 2.1 citation hit (mean ± standard deviation over seeds); one phrasing: 71.0% ± 4.6.
- Paired by seed, two phrasings lead by 18, 14 and 12 points. Both conditions fit the training phrasing almost perfectly.
- The A1 run is the median of the three two-phrasing seeds.
- The one-phrasing runs see each fact 8 times rather than 16 and take half the optimiser steps, so this comparison mixes phrasing diversity with exposure count. An exposure-matched control (one phrasing, 16 epochs) is recorded under Deviations in `EXPERIMENT_PLAN.md`.

## Exposure-matched control (added 2026-10-08)

One phrasing for 16 epochs, so that each fact is seen 16 times and the optimiser takes as many steps as in the two-phrasing model.

| Condition | Exposures per fact | Seed 20261005 | Seed 20261006 | Seed 20261007 | Citation hit (mean ± sd) | Dev loss (mean) |
|---|---|---|---|---|---|---|
| Two phrasings, 8 epochs | 16 | 85% | 84% | 88% | 85.7% ± 2.1 | 0.153 |
| One phrasing, 16 epochs | 16 | 73% | 79% | 76% | 76.0% ± 3.0 | 0.422 |
| One phrasing, 8 epochs | 8 | 67% | 70% | 76% | 71.0% ± 4.6 | 0.435 |

- Matching exposure recovers about 5 of the 14.7-point gap; about 10 points remain with phrasing diversity (paired by seed: +12, +5, +12).
- Dev-para loss barely moves with extra exposure (0.435 to 0.422) but falls to 0.153 with a second phrasing.
- Exploratory: three seeds on 100 items; a paired t-test on the seed-level differences gives t ≈ 4.1 with 2 degrees of freedom (p ≈ 0.05, two-sided).
