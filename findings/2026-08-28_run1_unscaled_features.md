# Run 1 — Unscaled Features (baseline)

**Date:** 2026-08-28
**Dataset:** Cresci-2017 — genuine_accounts + social_spambots_1-3 + traditional_spambots_1-4
(fake_followers and crowdflower_results excluded)
**Accounts after cleaning (>=10 tweets):** 6,553 (1,082 human, 5,471 bot)
**Methodology:** As specified in the v2/v3 dossier, with no feature scaling applied
before training — this is the gap this run exposes.

This file is a preserved snapshot of the first successful real-data run, taken
*before* adding feature scaling, so the "did scaling matter, and how much"
question has a clean before/after to compare against. `results/tables/` and
`results/figures/` are gitignored and get overwritten on every pipeline run,
so this is the only durable record of this specific configuration's numbers.

## Headline result: Outcome B — mixed / marginal improvement

Sentiment features improved logistic_regression and random_forest performance
but not svm. Per `src/report.py`'s classification: the relationship between
sentiment consistency and account-operation type is inconsistent across model
types in this run.

## Per-model comparison (metadata-only vs. metadata+sentiment)

| model | f1 (metadata only) | f1 (+sentiment) | delta | accuracy delta | roc_auc delta |
|---|---|---|---|---|---|
| logistic_regression | 0.9067 | 0.9448 | **+0.0381** | +0.0580 | +0.0022 |
| random_forest | 0.9801 | 0.9862 | +0.0061 | +0.0099 | +0.0041 |
| svm | 0.9415 | 0.9415 | **0.0000** | 0.0000 | -0.0010 |

## Statistical significance (McNemar's test)

| model | p_value | verdict |
|---|---|---|
| logistic_regression | 8.62e-13 | significant_improvement |
| random_forest | 0.0209 | significant_but_small |
| svm | 1.0000 | no_improvement |

## Feature importance (Random Forest, Feature Set B, top 10)

| rank | feature | importance |
|---|---|---|
| 1 | statuses_count | 0.2405 |
| 2 | account_age_days | 0.1766 |
| 3 | friends_count | 0.1466 |
| 4 | followers_count | 0.1176 |
| 5 | pct_positive | 0.0541 |
| 6 | mean_subjectivity | 0.0508 |
| 7 | pct_neutral | 0.0416 |
| 8 | listed_count | 0.0404 |
| 9 | polarity_std | 0.0401 |
| 10 | pct_extreme | 0.0339 |

Neither `polarity_std` nor `pct_extreme` — the features Dickerson et al.
(2014) identified as the top discriminators — made the top 5 in this run.
`pct_positive` (general sentiment positivity) was the strongest sentiment
signal instead.

## Known methodological gap in this run

**No feature scaling was applied before training.** Metadata features like
`followers_count` range into the millions, while sentiment features sit on a
roughly [-1, 1] or [0, 1] scale. This almost certainly explains SVM's exact-zero
result: SVM's RBF kernel is distance-based, so unscaled large-magnitude
metadata features would dominate the kernel computation, mathematically
drowning out any contribution from the (much smaller-magnitude) sentiment
features — svm's metrics are identical to six decimal places whether or not
sentiment features are included, which is the signature of a feature being
functionally invisible to the model, not a real null result.

Random Forest's split-based decisions are scale-invariant, so its numbers
here should NOT change materially when scaling is added. Logistic Regression
is also somewhat scale-sensitive (regularization penalizes large- and
small-magnitude coefficients differently), so its numbers may shift too.

**Next step:** add `StandardScaler` (fit on training data only, applied
before SMOTE) and re-run, to get a fair read on whether SVM shows a real
effect once features are on comparable scales. See
`2026-08-28_run2_scaled_features.md` (once that run completes) for the
comparison.
