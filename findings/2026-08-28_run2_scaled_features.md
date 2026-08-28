# Run 2 — Scaled Features

**Date:** 2026-08-28
**Dataset:** same as Run 1 (1,082 human, 5,471 bot, from the cached engineered
features — no re-run of loading/cleaning/sentiment scoring needed).
**Methodology change from Run 1:** `preprocessing.scale_features()` added -
`StandardScaler` fit on the training split only, applied to both Feature Set A
and B, *before* SMOTE (so SMOTE's own k-NN interpolation isn't distorted by
the same scale issue).

See `2026-08-28_run1_unscaled_features.md` for the baseline this compares against.

## Headline result: still Outcome B, but a materially different story

## Per-model comparison (metadata-only vs. metadata+sentiment)

| model | f1 (metadata only) | f1 (+sentiment) | delta | p_value | verdict |
|---|---|---|---|---|---|
| logistic_regression | 0.9062 | 0.9549 | **+0.0487** | 2.32e-15 | significant_improvement |
| random_forest | 0.9838 | 0.9866 | +0.0028 | 0.263 | improvement_not_significant |
| svm | 0.9544 | 0.9755 | **+0.0211** | 6.35e-6 | significant_but_small |

## What changed vs. Run 1 (unscaled), and why

| model | Run 1 delta F1 (p) | Run 2 delta F1 (p) | |
|---|---|---|---|
| logistic_regression | +0.0381 (8.62e-13) | +0.0487 (2.32e-15) | stronger, as expected (LR is moderately scale-sensitive) |
| random_forest | +0.0061 (0.0209, significant) | +0.0028 (0.263, **not** significant) | weaker - see below |
| svm | **0.0000 (1.0, no effect)** | **+0.0211 (6.35e-6, real effect)** | **completely unmasked** |

**SVM's result changed exactly as predicted.** In Run 1, SVM's metrics were
identical to six decimal places whether or not sentiment features were
included - the signature of a feature being mathematically invisible to a
distance-based model when metadata dominates the unscaled feature space. With
scaling, SVM shows a real, statistically significant improvement from
sentiment features (+2.1 F1 points, p=6.35e-6). This was the whole point of
the methodology change and confirms the original hypothesis for this model.

**Random Forest's own improvement shrank and lost significance** - this is
not a contradiction, and RF itself is still scale-invariant (its splits don't
care about magnitude). What changed is *upstream*: scaling now happens
before SMOTE, so SMOTE generates its synthetic minority (human) samples using
correctly-scaled distances instead of metadata-dominated ones. That produced
better-quality synthetic samples, which lifted RF's metadata-only baseline
itself from 98.0% to 98.4% F1 - leaving less headroom for sentiment features
to add on top. RF was already near-ceiling; scaling mostly helped the
*baseline* catch up rather than the sentiment features specifically.

**Logistic Regression got a bigger, still extremely significant boost**
(+4.9 vs +3.8 points, p even smaller) - consistent with LR being moderately
scale-sensitive via its regularization term.

## Feature importance (Random Forest, Feature Set B, top 10)

| rank | feature | importance |
|---|---|---|
| 1 | statuses_count | 0.2637 |
| 2 | account_age_days | 0.1641 |
| 3 | friends_count | 0.1438 |
| 4 | followers_count | 0.1211 |
| 5 | listed_count | 0.0579 |
| 6 | pct_positive | 0.0456 |
| 7 | pct_neutral | 0.0433 |
| 8 | polarity_std | 0.0409 |
| 9 | mean_subjectivity | 0.0325 |
| 10 | pct_extreme | 0.0295 |

Same broad story as Run 1: metadata still dominates (top 4 unchanged),
`polarity_std`/`pct_extreme` still rank low (8th/10th). `listed_count`
displaced `pct_positive` from the #5 spot, but the core finding is unchanged -
this study does not replicate Dickerson et al.'s specific "sentiment variance
is a top discriminator" claim, even though sentiment features overall do help.

## Bottom line

Scaling was the right call. It didn't change the headline outcome label
(still B, mixed), but it changed *which* result was trustworthy: Run 1's
"SVM shows nothing" was a methodological artifact, not a finding. With
correct scaling, all three models move in the same direction (sentiment
helps), two of three reach statistical significance, and the one that
doesn't (Random Forest) has a defensible explanation (already near-ceiling,
baseline itself improved from better SMOTE sampling) rather than looking
like a contradiction.
