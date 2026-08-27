"""Section 8: interpret the pipeline's results and write a markdown report + plots.

Reads the CSVs written by `pipeline.run_pipeline()` (run that first). Compares
Feature Set A (metadata-only) against Feature Set B (metadata + sentiment)
per model, checks statistical significance (McNemar's test, already computed
by the pipeline), and classifies the headline result against the dossier's
three possible outcomes (Section 8):

  A - significant improvement:  F1 delta >= F1_IMPROVEMENT_THRESHOLD and
                                 p < SIGNIFICANCE_ALPHA, for every model
  C - no improvement (null):    no model shows any F1 improvement at all
  B - mixed / marginal:         anything in between

This classification is a heuristic, not a substitute for reading the actual
numbers - treat the generated report as a first draft to check, not a final
verdict.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
KEY_SENTIMENT_FEATURES = {"polarity_std", "pct_extreme"}

OUTCOME_LABELS = {
    "A": "Outcome A - sentiment consistency significantly distinguishes AI-operated from human-operated accounts",
    "B": "Outcome B - marginal or mixed evidence for the sentiment-consistency signal",
    "C": "Outcome C - sentiment consistency does not help distinguish AI-operated accounts (null result)",
}

NARRATIVES = {
    "A": (
        "Sentiment-based features - particularly sentiment variance and proportion of "
        "extreme-sentiment posts - provide statistically significant improvements when "
        "distinguishing AI-operated accounts from human-operated ones, on top of traditional "
        "metadata features. This supports the hypothesis that AI-operated accounts exhibit "
        "less natural emotional variability (higher sentiment consistency) than human-operated "
        "accounts."
    ),
    "B": (
        "Results are mixed: sentiment features improved {improved_models} performance but not "
        "{unimproved_models}. This suggests that the relationship between sentiment consistency "
        "and account-operation type is non-linear, favouring tree-based and kernel-based methods "
        "over linear ones. Future work should explore sentiment feature engineering that "
        "linearises these relationships."
    ),
    "C": (
        "Contrary to prior findings by Dickerson et al. (2014), this study found no "
        "statistically significant improvement from adding sentiment features on the "
        "Cresci-2017 dataset (used here as a proxy for human- vs. AI-operated accounts) using "
        "traditional ML classifiers. One hypothesis is that the social spambot subcategory in "
        "Cresci-2017 has been specifically engineered to mimic natural human sentiment, "
        "rendering sentiment features ineffective against this class of AI-operated accounts - "
        "pointing to an arms race between automation sophistication and detection methods."
    ),
}


def load_results(tables_dir=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = Path(tables_dir) if tables_dir is not None else Path(config.RESULTS_TABLES_DIR)
    summary_path = tables_dir / "experiment_summary.csv"
    mcnemar_path = tables_dir / "mcnemar_significance.csv"
    importance_path = tables_dir / "feature_importance.csv"

    missing = [p for p in (summary_path, mcnemar_path, importance_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing result file(s): "
            + ", ".join(str(p) for p in missing)
            + ". Run `python -m src.pipeline` first."
        )

    return (
        pd.read_csv(summary_path),
        pd.read_csv(mcnemar_path),
        pd.read_csv(importance_path),
    )


def compute_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per model: Feature Set A vs. B, and the delta, for each metric."""
    wide = summary.pivot(index="model", columns="feature_set", values=METRIC_COLUMNS)
    rows = []
    for model in wide.index:
        row = {"model": model}
        for metric in METRIC_COLUMNS:
            a = wide.loc[model, (metric, "metadata_only")]
            b = wide.loc[model, (metric, "metadata_plus_sentiment")]
            row[f"{metric}_metadata_only"] = a
            row[f"{metric}_metadata_plus_sentiment"] = b
            row[f"{metric}_delta"] = b - a
        rows.append(row)
    return pd.DataFrame(rows)


def classify_models(deltas: pd.DataFrame, mcnemar: pd.DataFrame) -> pd.DataFrame:
    """Per-model verdict, joining each model's F1 delta with its McNemar p-value."""
    merged = deltas.merge(mcnemar[["model", "p_value"]], on="model", how="left")

    def _label(row) -> str:
        significant = row["p_value"] < config.SIGNIFICANCE_ALPHA
        meaningful = row["f1_delta"] >= config.F1_IMPROVEMENT_THRESHOLD
        if significant and meaningful:
            return "significant_improvement"
        if significant and row["f1_delta"] > 0:
            return "significant_but_small"
        if row["f1_delta"] > 0:
            return "improvement_not_significant"
        return "no_improvement"

    merged["verdict"] = merged.apply(_label, axis=1)
    return merged


def classify_overall(classified: pd.DataFrame) -> str:
    verdicts = classified["verdict"]
    if (verdicts == "significant_improvement").all():
        return "A"
    if (verdicts == "no_improvement").all():
        return "C"
    return "B"


def feature_importance_note(importance: pd.DataFrame, top_n: int = 5) -> str:
    top_features = importance.head(top_n)["feature"].tolist()
    hits = [f for f in top_features if f in KEY_SENTIMENT_FEATURES]
    if hits:
        return (
            f"{', '.join(hits)} rank in the top {top_n} features by Random Forest importance - "
            "per the dossier, this is a major finding: it confirms that emotional variability, "
            "not just average sentiment, is what separates AI-operated accounts from "
            "human-operated ones."
        )
    return (
        f"Neither polarity_std nor pct_extreme made the top {top_n} features by importance; "
        f"the top {top_n} were: {', '.join(top_features)}."
    )


def _narrative_kwargs(classified: pd.DataFrame) -> dict:
    improved = classified.loc[classified["verdict"] != "no_improvement", "model"].tolist()
    not_improved = classified.loc[classified["verdict"] == "no_improvement", "model"].tolist()
    return {
        "improved_models": ", ".join(improved) if improved else "none",
        "unimproved_models": ", ".join(not_improved) if not_improved else "none",
    }


def _df_block(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string(index=False) + "\n```"


def render_report(
    deltas: pd.DataFrame,
    classified: pd.DataFrame,
    importance: pd.DataFrame,
    overall_outcome: str,
    feature_note: str,
) -> str:
    narrative = NARRATIVES[overall_outcome]
    if overall_outcome == "B":
        narrative = narrative.format(**_narrative_kwargs(classified))

    lines = [
        "# Results Interpretation",
        "",
        f"**{OUTCOME_LABELS[overall_outcome]}**",
        "",
        narrative,
        "",
        "## Per-model comparison (metadata-only vs. metadata+sentiment)",
        "",
        _df_block(deltas),
        "",
        "## Statistical significance (McNemar's test)",
        "",
        _df_block(classified[["model", "p_value", "verdict"]]),
        "",
        "## Feature importance",
        "",
        feature_note,
        "",
        _df_block(importance.head(10)),
        "",
    ]
    return "\n".join(lines)


def plot_metric_comparison(summary: pd.DataFrame, out_path) -> None:
    pivot = summary.pivot(index="model", columns="feature_set", values="f1")
    ax = pivot.plot(kind="bar", figsize=(7, 4), ylim=(0, 1))
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score by model and feature set")
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150)
    plt.close(ax.figure)


def plot_feature_importance(importance: pd.DataFrame, out_path, top_n: int = 10) -> None:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Random Forest importance")
    ax.set_title(f"Top {top_n} features (Feature Set B)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_report(tables_dir=None, figures_dir=None) -> dict:
    tables_dir = Path(tables_dir) if tables_dir is not None else Path(config.RESULTS_TABLES_DIR)
    figures_dir = Path(figures_dir) if figures_dir is not None else Path(config.RESULTS_FIGURES_DIR)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary, mcnemar, importance = load_results(tables_dir)
    deltas = compute_deltas(summary)
    classified = classify_models(deltas, mcnemar)
    overall_outcome = classify_overall(classified)
    feature_note = feature_importance_note(importance)

    report_text = render_report(deltas, classified, importance, overall_outcome, feature_note)
    report_path = tables_dir / "interpretation_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved interpretation report to %s", report_path)

    plot_metric_comparison(summary, figures_dir / "f1_comparison.png")
    plot_feature_importance(importance, figures_dir / "feature_importance.png")
    logger.info("Saved comparison plots to %s", figures_dir)

    return {
        "outcome": overall_outcome,
        "deltas": deltas,
        "classified": classified,
        "report_path": report_path,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = generate_report()
    print(f"Headline outcome: {OUTCOME_LABELS[result['outcome']]}")
    print(f"Full report: {result['report_path']}")
