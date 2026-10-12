# Commit map

The commit history was rewritten once, before the repository was made public, to replace the author email on early commits. Only the author and committer email changed: every rewritten commit has the same file tree, message and author date as the original. Training runs in W&B record the commit their Kaggle bundle was built from (`bundle_git_commit`) under its original hash. This table maps each original hash to the commit in this repository.

| Original | Current | Referenced by W&B | Message |
|---|---|---|---|
| `1408e5d` | `66fa9d4` |  | first commit |
| `f66f9e8` | `f1ef374` |  | Initialise repository skeleton, split manifest and Act parser |
| `b2f7d4d` | `11a3e92` |  | Add split assignment, quota allocation and source-level masking |
| `ba43199` | `58d4969` |  | Add LLM-based QA generation with programmatic validation |
| `e0f01fd` | `39f4540` |  | Generate QA dataset, relax identifier check and add task script |
| `33b49c3` | `32b0704` |  | Add cross-model verification of generated QA items |
| `3c93027` | `2f89f14` |  | Build evaluation sets with deduplication and semantic leakage audit |
| `e88d49c` | `8f18b77` |  | Generate held-out paraphrase test sets |
| `1890048` | `e5b6b8d` |  | Add data card, pre-registered experiment plan and blind spot check |
| `5b853e5` | `a5e46a6` |  | Add missing plan and data card content and dual-model spot-check review |
| `0f7d80d` | `60359a8` |  | Complete phase 1 spot check and record results in the data card |
| `48786ad` | `a278f34` |  | ## Model registry and promotion |
| `a7e6a0a` | `4a5ee5b` | 3 training runs | Add LoRA training pipeline and Kaggle bundle for phase 2 |
| `6d51123` | `8542015` | 2 training runs | Add training-fit probe to separate under-fitting from phrasing generalisation |
| `0ffda59` | `1c6fa25` | 12 training runs | Add stage 1 W&B sweep and coerce config overrides to default types |
| `71bc642` | `44bfb3e` | 6 training runs | Add stage 2 grid sweep over LoRA rank and epochs |
| `5d28110` | `7e552dd` |  | Record model selection and add stage 3 seed and ablation sweeps |
| `34a89f5` | `01e8393` | 5 training runs | Record model selection and add stage 3 seed and ablation sweeps |
| `3a7d7cd` | `15a4e65` |  | Add adapter merge, GGUF conversion and llama.cpp parity check |
| `74d8788` | `380aeab` |  | Record GGUF conversion and parity results for A0 and A1 |
| `009ba32` | `eec03f9` | 3 training runs | Record stage 3 seed variance and one-phrasing ablation |
| `cf1016d` | `1395a36` |  | Add section-bounded RAG index and dev-para retrieval selection |
| `f21e747` | `973e986` |  | Add section-bounded RAG index and dev-para retrieval selection |
| `de9c37c` | `02df356` |  | Add FastAPI answer service for all four arms |
| `4b60b62` | `6710f32` |  | Warm up retrieval at startup and add exploratory mention hit metric |
| `5975728` | `3a6320e` |  | Record exposure-matched control for the one-phrasing ablation |
| `6d05709` | `59308db` |  | Add MLflow model registry with promotion gate and rollback |
| `5054a28` | `51d7340` |  | Record promotion gate and rollback drill, start the answer service offline |
| `77ee33a` | `c5f4c60` |  | Add answer collection and calibrated two-judge scoring |
| `9a89abe` | `bec9314` |  | Record evaluation answers and judge calibration |
| `eea1d93` | `58c905a` |  | Add a secondary-judge sensitivity run |
| `4b72654` | `cf2b18b` |  | Record full judging runs with both judges |
| `8f16f75` | `aa7cfc8` |  | README updated to phase3 |
| `7ba764f` | `4a8b0be` |  | Add the pre-registered statistical analysis |
| `733fe20` | `e91fce9` |  | Add FINDINGS |
| `d092732` | `b91737f` |  | Orchestrate evaluation and promotion with n8n |
| `678cfcf` | `1b946c1` |  | Add MODEL_CARD and summarise results in README |
| `0682873` | `1d36755` |  | Stop uploading host metadata to W&B runs |
| `4e2ed0e` | `e4a6480` |  | Add the W&B report and link it from README and MODEL_CARD |
| `c76398f` | `bb0c967` |  | Add the MIT licence and attribute the legislative source under CC BY 4.0 |
