# Run 3 — Metadata-Constrained + New Sentiment Features

**Date:** 2026-08-28
**Dataset:** same as Runs 1-2 (6,553 accounts: 1,082 human, 5,471 bot).
**Methodology changes from Run 2:**
1. Removed `followers_count`, `friends_count`, `statuses_count`,
   `account_age_days` from Feature Set A - only `verified` and
   `listed_count` remain. Simulates a metadata-constrained scenario (modern
   LLM agent accounts mimicking normal account statistics).
2. Added three new sentiment features: `contextual_consistency_score`
   (std of mean sentiment across 4 keyword-matched topic buckets),
   `temporal_sentiment_drift` (std of mean sentiment across 3 chronological
   windows), `sentiment_topic_alignment_score` (proportion of topic-tagged
   tweets whose sentiment breaks an "expected pattern" per topic - see
   `_topic_mismatch` in `src/features.py` for the exact, explicitly
   interpreted rules, since the original brief didn't fully specify them).
3. Scaling (StandardScaler, from Run 2) unchanged.

See `2026-08-28_run1_unscaled_features.md` and `_run2_scaled_features.md`
for the full-metadata baselines this compares against.

## Headline result: Outcome B, but the clearest confirmation yet for Random Forest

## Per-model comparison (metadata-only vs. metadata+sentiment)

| model | f1 (metadata only) | f1 (+sentiment) | f1 delta | roc_auc delta | p_value | verdict |
|---|---|---|---|---|---|---|
| logistic_regression | 0.9094 | 0.8630 | -0.0465 | +0.0866 | 2.58e-4 | no_improvement (see caveat below) |
| random_forest | 0.7781 | 0.9571 | **+0.1791** | +0.2030 | 1.98e-60 | significant_improvement |
| svm | 0.9113 | 0.9406 | +0.0293 | +0.1654 | 1.87e-6 | significant_but_small |

## Random Forest: the clearest confirmation of the Run 3 hypothesis

With only `verified`/`listed_count`, RF's metadata-only F1 collapses to 0.778
(recall only 0.668 - it misses a third of actual bots). Adding sentiment
features lifts F1 to 0.957, recall to 0.948 - an enormous, overwhelming
statistically significant jump (p=1.98e-60). This is exactly what "removing
reliable metadata should make sentiment essential" predicts, and RF shows it
cleanly.

## Logistic Regression: F1 dropped but ROC-AUC improved - not a real regression

LR's metadata-only F1 (0.909) is deceptively high: with bots at ~84% of the
test set, a classifier that always predicts "bot" gets F1 ~= 0.91 from the
base rate alone - suspiciously close to LR's actual number, meaning with
only 2 weak features LR was behaving close to that trivial classifier.
Adding sentiment features made LR actually use them, sometimes correctly
predicting "human" - this lowers bot-recall (hence lower F1 at the default
0.5 threshold: 0.794 vs. 0.940) but genuinely improves the model's
underlying discrimination (ROC-AUC 0.774 -> 0.860, a real +8.7 point gain).
Sentiment didn't hurt LR; it stopped LR from defaulting to the majority
class. `report.py`'s current classifier logic buckets any negative F1-delta
as "no_improvement" without distinguishing this from a true null result -
this specific verdict undersells what's actually happening. Worth a report.py
refinement if this distinction matters for the write-up.

## SVM: solid, real improvement

+2.9 F1 points (p=1.87e-6) and a large ROC-AUC gain (+16.5 points) -
consistent, genuine improvement, smaller than RF's but clearly real.

## Feature importance (Random Forest, Feature Set B, full ranking)

| rank | feature | importance |
|---|---|---|
| 1 | listed_count | 0.3011 |
| 2 | pct_positive | 0.1221 |
| 3 | pct_neutral | 0.1025 |
| 4 | mean_polarity | 0.0747 |
| **5** | **polarity_std** | **0.0696** |
| 6 | mean_subjectivity | 0.0663 |
| 7 | pct_negative | 0.0631 |
| 8 | pct_extreme | 0.0606 |
| 9 | subjectivity_std | 0.0478 |
| 10 | contextual_consistency_score | 0.0338 |
| 11 | sentiment_topic_alignment_score | 0.0310 |
| 12 | temporal_sentiment_drift | 0.0272 |
| 13 | verified | 0.0001 |

**`polarity_std` reaches the top 5 for the first time across all three
runs.** This is the exact finding Dickerson et al. (2014) predicted -
sentiment variance as a top discriminator - which was being drowned out by
metadata dominance in Runs 1 and 2. Once the dominant metadata features are
removed, it surfaces clearly. `listed_count` (the one metadata feature kept)
is still the single strongest predictor overall, though.

**The three new Run 3 features rank last (10th-12th of 13).** This likely
confirms the topic-bucket sparsity concern flagged before implementation for
`contextual_consistency_score` and `sentiment_topic_alignment_score` (both
depend on the narrow 4-keyword-bucket classifier). `temporal_sentiment_drift`
does NOT depend on topic buckets at all and still ranked last - a genuine,
somewhat surprising finding rather than an artifact, worth reporting as-is
rather than glossed over.

## Bottom line

The metadata-constrained scenario is the strongest evidence yet for the
core hypothesis: when reliable account metadata isn't available (as with
sophisticated bots/LLM agents mimicking normal statistics), sentiment
features become the dominant usable signal, and models that can exploit
them (Random Forest especially, SVM to a real but smaller degree) show
large, highly significant gains. The three new features built specifically
for this run did not turn out to be the strongest contributors - the
existing, simpler sentiment features (`pct_positive`, `pct_neutral`,
`mean_polarity`, `polarity_std`) still carried most of the signal.
