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