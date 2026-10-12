import argparse
import os

ENTITY = "arlives"
PROJECT = "privacy-act-lora-vs-rag"
SWEEPS = {"stage1": "egec9zjd", "stage2": "fb3xh2wj", "seeds": "84ofrhtf", "ablation": "8b3s0fxh", "control": "m1ng5lnv"}
SELECTED_RUN = "fresh-sweep-6"

INTRO = (
    "This report accompanies the GitHub repository privacy-act-lora-vs-rag. It compares LoRA fine-tuning and "
    "retrieval-augmented generation for question answering over the Australian Privacy Act 1988 (Cth) with "
    "Qwen2.5-1.5B-Instruct, under an experiment plan fixed before training."
)
FINDINGS = [
    "Fine-tuning memorises phrasings more than facts: the LoRA model scores 1.99 of 2 on trained questions but 1.27 "
    "when they are rephrased, tied with the base model plus retrieval (1.22).",
    "Fine-tuning suppresses the use of retrieved text: with identical retrieved extracts, the fine-tuned model scores "
    "0.74 to 0.82 points below the base model on provisions it was not trained on.",
    "Fine-tuning teaches the citation format, not the knowledge: outside its training data, every answer from the "
    "fine-tuned model carries a wrong citation, and it never corrects a question about a provision that does not exist.",
]
SELECTION = (
    "Hyperparameters were chosen on dev-para citation hit only, a held-out set of rephrased training questions never "
    "used for reported results. A Bayesian search over learning rate, rank and target modules at three epochs showed "
    "that every configuration was under-fitted and that adapting all linear layers dominated. A grid over rank and "
    "epochs then selected all-linear LoRA with rank 64, alpha 128, learning rate 5e-4, eight epochs and two phrasings "
    "per fact (85% dev-para citation hit)."
)
DIVERSITY = (
    "Phrasing diversity, not exposure, is what generalises. Across three seeds the selected configuration reached "
    "85.7% ± 2.1; training on one phrasing per fact reached 71.0% ± 4.6, and an exposure-matched control with one "
    "phrasing for twice as many epochs recovered only part of the gap (76.0% ± 3.0)."
)
EVALUATION = (
    "Each arm answered 717 questions across five sets. Accuracy was scored 0 to 2 by GPT-5.4 mini, calibrated against "
    "Claude Opus 5.5 on 150 answers (weighted kappa 0.92) with human adjudication of the 16 disagreements. Claude Opus "
    "5.5 also scored every answer as a registered sensitivity analysis; every ranking holds under both judges. "
    "Hypotheses were tested with paired bootstrap intervals, McNemar's test and Holm-Bonferroni correction."
)
EVALUATION_RESULT = (
    "The pre-registered ordering A0 < A1 < A2 on rephrased questions was not supported: the fine-tuned and retrieval "
    "arms tied. Retrieval dominated on provisions outside the training data, while adding retrieval to the fine-tuned "
    "model recovered little. The full analysis, including limitations, is in FINDINGS.md in the repository."
)
PROMOTION = (
    "Served models are versioned in an MLflow registry. A candidate replaces the production model only if it passes "
    "four checks on dev-para through the serving endpoint: conversion parity with the PEFT model, a strictly higher "
    "citation hit than production, a citation tag on at least 95% of answers, and no more than 1.2 times production's "
    "95th-percentile latency. The gate runs from an n8n workflow; in the run below the fine-tuned model passed all four "
    "checks against the base model (citation hit 86% against 7%, p95 latency 889 ms against 3,047 ms) and was promoted."
)


def build(wr, expr):
    def runset(name, filters, groupby=None):
        return wr.Runset(entity=ENTITY, project=PROJECT, name=name, filters=filters, groupby=groupby or [])

    def sweep(key):
        return expr.Metric("Sweep") == SWEEPS[key]

    stage1 = wr.PanelGrid(
        runsets=[runset("Stage 1: Bayesian search", [sweep("stage1")])],
        panels=[
            wr.ParallelCoordinatesPlot(
                title="Stage 1: learning rate, rank and target modules against dev-para citation hit",
                columns=[wr.ParallelCoordinatesPlotColumn(metric=wr.Config(k)) for k in ("learning_rate", "lora_r", "lora_target")]
                + [wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric("dev/citation_hit"))],
                layout=wr.Layout(w=24, h=9),
            )
        ],
    )
    stage2 = wr.PanelGrid(
        runsets=[runset("Stage 2: rank by epochs grid", [sweep("stage2")])],
        panels=[
            wr.ParallelCoordinatesPlot(
                title="Stage 2: rank and epochs against dev-para citation hit and loss",
                columns=[wr.ParallelCoordinatesPlotColumn(metric=wr.Config(k)) for k in ("lora_r", "epochs")]
                + [wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric(m)) for m in ("dev/citation_hit", "dev/loss")],
                layout=wr.Layout(w=12, h=9),
            ),
            wr.LinePlot(
                title="Stage 2 runs: training loss",
                x="Step",
                y=["train/loss"],
                layout=wr.Layout(x=12, w=12, h=9),
            ),
        ],
    )
    stage3 = wr.PanelGrid(
        runsets=[
            runset(
                "Stage 3: seeds, ablation and control",
                [expr.Metric("Sweep").isin([SWEEPS["seeds"], SWEEPS["ablation"], SWEEPS["control"]])],
                groupby=["config.phrasings", "config.epochs"],
            ),
            runset("Selected run", [expr.Metric("Name") == SELECTED_RUN], groupby=["config.phrasings", "config.epochs"]),
        ],
        panels=[
            wr.BarPlot(
                title="Dev-para citation hit by phrasings and epochs (mean and standard deviation over seeds)",
                metrics=["dev/citation_hit"],
                groupby_aggfunc="mean",
                groupby_rangefunc="stddev",
                layout=wr.Layout(w=24, h=8),
            )
        ],
    )
    evaluation = wr.PanelGrid(
        runsets=[runset("Evaluation", [expr.Metric("JobType") == "evaluation"])],
        panels=[
            wr.WeavePanelSummaryTable(table_name="accuracy", layout=wr.Layout(w=24, h=12)),
            wr.WeavePanelSummaryTable(table_name="hypotheses", layout=wr.Layout(y=12, w=24, h=8)),
            wr.WeavePanelSummaryTable(table_name="citations_latency", layout=wr.Layout(y=20, w=24, h=12)),
        ],
    )
    gate = wr.PanelGrid(
        runsets=[runset("Promotion gate", [expr.Metric("JobType") == "gate"])],
        panels=[wr.RunComparer(layout=wr.Layout(w=24, h=12))],
    )
    blocks = [
        wr.P(INTRO),
        wr.P("Main findings:"),
        wr.UnorderedList(items=FINDINGS),
        wr.P("The sections below show how the model was selected, how it was evaluated and how it is promoted to production."),
        wr.H2("1. Model selection"),
        wr.P(SELECTION),
        stage1,
        stage2,
        wr.P(DIVERSITY),
        stage3,
        wr.H2("2. Evaluation"),
        wr.P(EVALUATION),
        evaluation,
        wr.P(EVALUATION_RESULT),
        wr.H2("3. Promotion to production"),
        wr.P(PROMOTION),
        gate,
    ]
    return wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="LoRA vs RAG on the Australian Privacy Act: a pre-registered comparison",
        description="Qwen2.5-1.5B-Instruct, four arms, five evaluation sets, calibrated LLM judging and a gated promotion pipeline.",
        width="fixed",
        blocks=blocks,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create the W&B report for the study as a draft, optionally publishing it with a share link")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("WANDB_ENTITY", ENTITY)
    import wandb_workspaces.expr as expr
    import wandb_workspaces.reports.v2 as wr

    report = build(wr, expr).save(draft=not args.publish)
    print(report.url)
    if args.publish:
        print(report.enable_share_link())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
