# Sentiment Consistency as a Signal for Detecting AI-Operated Accounts

> **DCIT 316 Final Course Project – submission**
> **Student:** Sonpon Ye-shua Chief · **ID:** 22082706
> **Course:** DCIT 316 Computational Models for Social Media Mining, University of Ghana
> **Submitted version:** tag `v1.0-final` (final commit on `main`, 15 Oct 2026)

## Submission evidence map

| Rubric evidence | Where |
|---|---|
| Manuscript (PDF) | uploaded to Sakai: `FINAL_PROJECT_22082706_SonponYeshuaChief.pdf` |
| Reproduction steps | this README → **Quick reproduction** below |
| Source code | `src/` (pipeline), `tests/` (14 tests), `scripts/` |
| Notebooks with visible outputs | `notebooks/results_summary.ipynb` (executed; no raw data needed). Pipeline runners: `notebooks/run_pipeline_local.ipynb`, `notebooks/run_pipeline_colab.ipynb` (need the dataset) |
| Saved metrics | `findings/` (archived report per run), `results/metrics/` (CSV) |
| Figures | `results/figures/fig1_f1_by_run.png`, `fig2_feature_importance_run3.png`, `fig3_confusion_run3.png` |
| Literature matrix | `docs/literature_matrix.md` |
| Data card / model card | `docs/data_card.md`, `docs/model_card.md` |
| Ethics statement | `docs/ethics_statement.md` |
| AI-use disclosure | `docs/ai_use_disclosure.md` |
| Failure log | `docs/failure_log.md` |
| Reproducibility checklist | `docs/reproducibility_checklist.md` |

**Safe data access.** The Cresci-2017 dataset is research-only and is **not** in this repository. Download it from the Bot Repository (https://botometer.osome.iu.edu/bot-repository/datasets.html) and extract the subsets into `data/raw/` (layout under **The dataset** below). No passwords, API keys or tokens are used or stored.

## Quick reproduction

```bash
git clone https://github.com/SonponYe/project_86_-sentiment-bot-detection
cd project_86_-sentiment-bot-detection
python -m venv venv && source venv/bin/activate      # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# 1. Tests (no data needed, ~15 s)
python -m pytest -q                                   # expect: 14 passed

# 2. Figures, baseline and error analysis from the saved metrics (no data needed)
jupyter nbconvert --to notebook --execute --inplace notebooks/results_summary.ipynb

# 3. Full experiment (needs Cresci-2017 in data/raw/; ~30-40 min, mostly sentiment scoring)
python -m src.pipeline        # Run 3 configuration (current code)
python -m src.report          # outcome classification + plots
```

Runs 1 and 2 use the full metadata feature list; see **Results → Run 3** below for how to restore it.

Research question: does sentiment consistency (polarity, variance, extremity
across an account's post history) meaningfully distinguish human-operated
accounts from AI-operated ones, on top of account metadata alone?

This is a **v3 reframe** of the original "bot detection" framing (see
`## Framing history` below) — the underlying pipeline, features, models, and
dataset are unchanged; only the research question and terminology shifted
from "bot vs. human" to "AI-operated vs. human-operated," which better
matches 2026's landscape of LLM-agent-run accounts.

Course: Computational Models for Social Media Mining. Full background,
literature review, and evaluation plan live in the local planning documents
(`Sentiment_Bot_Detection_Proposal.docx`, `Bot_Detection_Research_Dossier.docx`,
`Bot_Detection_Research_Dossier_v3.docx`) — kept out of version control via
`.gitignore` since they're personal planning material, not project code.

## Framing history

- **v1/v2**: "Sentiment-Assisted Bot Detection" — binary bot-vs-human framing.
- **v3** (current): "Sentiment Consistency as a Signal for Detecting
  AI-Operated Accounts" — reframed because "bot" now conflates malicious spam
  accounts with legitimate AI agents, and LLM-generated text defeats
  metadata-only detection. The new framing is regulatory-relevant (EU AI Act
  disclosure requirements) rather than purely adversarial.
- Operational definitions used from v3 onward: **human-operated account**
  (posting driven by real-time individual human decisions), **AI-operated
  account** (content substantially generated/managed by an LLM-based system,
  disclosed or not), **sentiment consistency** (how stable an account's
  emotional polarity/intensity is across posts — high consistency is the
  AI-operation signal). Cresci-2017 is used as a proxy: `genuine_accounts` =
  human-operated, spambot subsets = the automated extreme.
- Only `src/report.py`'s auto-generated wording changed for this reframe
  (bot/human → AI-operated/human-operated in the narrative text). Every other
  file - config, data loading, features, modeling, evaluation, pipeline
  orchestration, the Colab notebook, and the dataset composition - is
  identical to v2.

## Status

Pipeline code is written, passes a smoke test on synthetic data, and has
been run against the real Cresci-2017 dataset three times (locally — see
`## Results` below). Each run changed one methodological variable; full
numbers are archived in `findings/`.

## Results

Three runs completed so far, each isolating one methodological change.
`results/` itself is gitignored and gets overwritten on every run, so
`findings/` is the durable record — each linked file has the full tables
(complete McNemar output, full feature-importance ranking, precision/recall
detail) behind the summaries below.

### Run 1 — Unscaled features (baseline)

Feature Set A: 6 metadata features. Feature Set B: A + 8 sentiment features.
No feature scaling. [Full write-up →](findings/2026-08-28_run1_unscaled_features.md)

| model | F1 (A) | F1 (B) | Δ F1 | p-value | verdict |
|---|---|---|---|---|---|
| logistic_regression | 0.9067 | 0.9448 | +0.0381 | 8.62e-13 | significant_improvement |
| random_forest | 0.9801 | 0.9862 | +0.0061 | 0.0209 | significant_but_small |
| svm | 0.9415 | 0.9415 | **0.0000** | 1.0000 | no_improvement |

**Outcome B (mixed).** SVM's exact-zero delta (identical metrics to 6
decimals) turned out to be a scaling artifact, not a real finding — see Run 2.

### Run 2 — Scaled features

Same feature sets as Run 1, plus `StandardScaler` (fit on train only,
applied before SMOTE so its k-NN interpolation isn't distorted by the same
scale issue). [Full write-up →](findings/2026-08-28_run2_scaled_features.md)

| model | F1 (A) | F1 (B) | Δ F1 | p-value | verdict |
|---|---|---|---|---|---|
| logistic_regression | 0.9062 | 0.9549 | +0.0487 | 2.32e-15 | significant_improvement |
| random_forest | 0.9838 | 0.9866 | +0.0028 | 0.2632 | improvement_not_significant |
| svm | 0.9544 | 0.9755 | **+0.0211** | 6.35e-6 | significant_but_small |

**Confirms scaling was the fix.** SVM now shows a real, significant effect.
Random Forest's own delta shrank and lost significance — not a contradiction:
scaling now happens before SMOTE too, so SMOTE's synthetic samples improved,
lifting RF's already-near-ceiling metadata-only baseline (0.980→0.984)
rather than leaving room for sentiment to add more on top.

### Run 3 — Metadata-constrained + 3 new sentiment features

Removed the 4 metadata features that dominated importance rankings
(`followers_count`, `friends_count`, `statuses_count`, `account_age_days`) —
simulating accounts whose metadata looks normal (e.g. sophisticated
bots/LLM agents). Added `contextual_consistency_score`,
`temporal_sentiment_drift`, `sentiment_topic_alignment_score`.
[Full write-up →](findings/2026-08-28_run3_metadata_constrained.md)

| model | F1 (A) | F1 (B) | Δ F1 | ROC-AUC Δ | p-value | verdict |
|---|---|---|---|---|---|---|
| logistic_regression | 0.9094 | 0.8630 | -0.0465 | +0.0866 | 2.58e-4 | no_improvement* |
| random_forest | 0.7781 | 0.9571 | **+0.1791** | +0.2030 | 1.98e-60 | significant_improvement |
| svm | 0.9113 | 0.9406 | +0.0293 | +0.1654 | 1.87e-6 | significant_but_small |

**Strongest confirmation yet.** With only `verified`/`listed_count` to work
with, Random Forest barely functions (F1 0.778) — adding sentiment nearly
doubles its effective performance (F1 0.957, p=1.98e-60). Feature
importance: `polarity_std` reaches the top 5 for the first time across all
three runs (previously ranks 8-9, now rank 5) — the exact effect Dickerson
et al. (2014) predicted, drowned out by metadata dominance until now. The
three new features underperform, ranking last (10th-12th of 13 features).

*LR's F1 *drop* is misleading, not a real regression: with only 2 weak
features, LR's metadata-only F1 (0.909) was inflated by mimicking the ~84%
bot base rate; adding sentiment made LR actually use it (lower bot-recall,
hence lower F1 at the default threshold) while genuinely improving
discrimination (ROC-AUC +8.7 points). See the linked write-up for the full
explanation.

**The current code reflects Run 3's configuration** — Feature Set A is
metadata-constrained (`verified`, `listed_count` only) and Feature Set B
includes all 11 sentiment features. To reproduce Run 1 or 2's full-metadata
results, restore `followers_count`, `friends_count`, `statuses_count`,
`account_age_days` to `METADATA_FEATURES` in `src/config.py` (see commit
`a4786f5` for that exact prior state).

## Repository layout

```
src/
  config.py        paths, random seed, feature lists
  data_loader.py    load Cresci-2017 users.csv/tweets.csv per class
  preprocessing.py  drop low-activity accounts, feature scaling, optional
                     SMOTE (train-only)
  features.py       Feature Set A (metadata) and B (metadata + sentiment)
  modeling.py       Logistic Regression / Random Forest / SVM, 5-fold CV
  evaluation.py     accuracy/precision/recall/F1/ROC-AUC, McNemar's test,
                     Random Forest feature importance
  pipeline.py       orchestrates the full run and writes results/
  report.py         interprets results/, classifies the headline outcome,
                     writes interpretation_report.md + comparison plots
tests/
  test_pipeline_smoke.py    runs the full pipeline + report on tiny synthetic data
  test_data_loader.py       regression tests for real-data edge cases hit so far
                             (invalid UTF-8 bytes, an ID dtype mismatch)
  test_features_run3.py     unit tests for the Run 3 feature helpers
findings/
  2026-08-28_run1_unscaled_features.md    archived numbers per run - see
  2026-08-28_run2_scaled_features.md      `## Results` above. Durable
  2026-08-28_run3_metadata_constrained.md record; results/ is gitignored.
data/
  raw/              put the downloaded Cresci-2017 subsets here (gitignored)
  processed/        engineered_features.csv - cached feature-engineering
                     output (gitignored). Lets pipeline.py skip the slow
                     load+clean+sentiment-scoring steps on re-runs; delete
                     it after changing feature engineering logic, since a
                     stale cache with a different schema will crash the run.
results/
  tables/           experiment_summary.csv, mcnemar_significance.csv,
                     feature_importance.csv, interpretation_report.md
                     (gitignored, regenerated by pipeline.py / report.py)
  figures/          confusion matrices, f1_comparison.png, feature_importance.png
                     (gitignored, regenerated by pipeline.py / report.py)
```

## Environment setup

A virtual environment already exists at `venv/` with all dependencies
installed. To recreate it from scratch:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependencies: pandas, numpy, scikit-learn, vaderSentiment, textblob,
imbalanced-learn, statsmodels, matplotlib, seaborn, jupyter, pytest.

> Note: this machine's project folder is inside OneDrive, which makes
> `pip install` and large package imports noticeably slower than usual
> (cloud-sync file interception). This is expected, not a broken install.

## The dataset

This project uses **Cresci-2017** (12,736 accounts, ~6.6M tweets), the
benchmark used by every paper in the literature review. It's downloaded
locally as zips under `cresci-2017.csv/datasets_full.csv/` (gitignored —
research-only license). Each subset's zip already contains the exact
`users.csv` / `tweets.csv` layout the loader expects:

```
data/raw/                          <- extract zips to here
  genuine_accounts.csv/
    users.csv
    tweets.csv
  social_spambots_1.csv/ ... _3.csv/
    users.csv
    tweets.csv
  traditional_spambots_1.csv/ ... _4.csv/
    users.csv
    tweets.csv
```

We use genuine_accounts vs. social_spambots_1-3 + traditional_spambots_1-4
(matching the proposal's stated "3 bot types" composition).
`fake_followers.csv.zip` and `crowdflower_results.csv.zip` are also in the
download but not used: fake_followers is an optional extra bot class for a
follow-up experiment, and crowdflower_results isn't an accounts/tweets
dataset at all (it's crowdsourced label-verification data).

The loader (`src/data_loader.py`) labels any subfolder containing "genuine"
as human (0) and everything else as bot (1); it's tolerant of `id` vs.
`user_id` column naming.

`Twibot_20.json` (secondary/comparison dataset) and `botometer-feedback-2019/`
(supplementary) are also on disk (gitignored) but use different schemas the
current loader doesn't support. **`Twibot_20.json` was checked and isn't
directly usable even if a loader were written for it: it's only 100 records
(the real TwiBot-20 has ~229,573) and has zero labeled accounts (no `label`
field on any record) — it appears to be an unlabeled sample, not the
dataset itself.** Both are optional later steps, not needed
for the primary experiment.

## Run the pipeline — on Google Colab (recommended)

This machine's project folder lives inside OneDrive, which makes disk I/O
(package installs, large file reads) noticeably slower than normal. Rather
than fight that locally, run the real dataset through
[`notebooks/run_pipeline_colab.ipynb`](notebooks/run_pipeline_colab.ipynb):

1. Upload `cresci-2017.csv/datasets_full.csv/` (the `.zip` files) to Google
   Drive, e.g. `My Drive/cresci-2017/datasets_full.csv/`.
2. Open the notebook in Colab (upload it, or open it from GitHub once pushed).
3. Run the cells top to bottom: mount Drive, clone the repo, install the
   extra dependencies, extract the zips into `data/raw/`, run the pipeline,
   inspect results, copy results back to Drive.

Note: Colab's fast disk doesn't fully offset the sentiment-scoring cost —
VADER and TextBlob run once per tweet and are CPU-bound. By default,
`build_sentiment_features` scores only the most recent 200 tweets per
account (`config.MAX_TWEETS_PER_ACCOUNT_FOR_SENTIMENT`), which keeps runtime
to minutes instead of hours without materially changing the variance signal.
Set that to `None` for a full, uncapped run once the pipeline is validated.

## Run the pipeline — locally

Once `data/raw/` is populated (see above):

```powershell
.\venv\Scripts\python.exe -m src.pipeline
```

This will:
1. Load and clean the accounts (drop accounts with <10 tweets, dedupe) —
   or, if `data/processed/engineered_features.csv` already exists from a
   prior run, skip straight to step 6 using the cached features (delete
   that file first if you've changed feature-engineering logic, since a
   stale cache with a different column schema will crash the run).
2. Build Feature Set A (as of Run 3: `verified`, `listed_count` only — a
   metadata-constrained scenario, see `## Results`) and Feature Set B
   (Feature Set A + 11 sentiment features: VADER polarity mean/variance/
   extremity/class-distribution, TextBlob subjectivity mean/variance, plus
   the three Run 3 additions — `contextual_consistency_score`,
   `temporal_sentiment_drift`, `sentiment_topic_alignment_score`).
3. Cache the engineered features to `data/processed/engineered_features.csv`.
4. Take one stratified 80/20 train/test split shared by both feature sets
   (`random_state=42`), so the two feature sets are compared on identical
   examples.
5. Scale both feature sets (`StandardScaler`, fit on the training split
   only), then apply SMOTE if the class balance warrants it — fit on the
   (scaled) training portion only, never the test set.
6. Train Logistic Regression, Random Forest, and SVM on each feature set
   (6 experiments total), with 5-fold stratified CV (F1) on the training
   set.
7. Write `results/tables/experiment_summary.csv` (accuracy, precision,
   recall, F1, ROC-AUC, CV F1 per experiment), `mcnemar_significance.csv`
   (per-model significance test of metadata-only vs. metadata+sentiment),
   `feature_importance.csv` (Random Forest importances for Feature Set B),
   and confusion matrix plots per experiment under `results/figures/`.

## Interpret the results

```powershell
.\venv\Scripts\python.exe -m src.report
```

Reads the three tables above, computes the Feature-Set-A-vs-B delta per
model, checks it against each model's McNemar p-value, and classifies the
headline result against the dossier's three possible outcomes (Section 8):
**A** — every model improves by ≥3% F1 with p < 0.05; **C** — no model
improves at all; **B** — anything in between. Writes
`results/tables/interpretation_report.md` (the delta table, the
significance verdict per model, and which narrative applies) plus
`results/figures/f1_comparison.png` and `feature_importance.png`.

This is a heuristic first draft, not a final verdict — read the actual
numbers yourself before trusting the auto-picked outcome label.

## Run the tests

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

The smoke test builds a small synthetic dataset shaped like Cresci-2017
(clearly polarized "bot" text vs. varied "human" text) and runs it through
the entire pipeline plus the report generation, to confirm the code is
correct independent of having the real dataset downloaded.

## Key references

The verified reference list is in the manuscript and in `docs/literature_matrix.md` (14 sources, each checked against its publisher or arXiv page).

Full literature review, gap analysis, evaluation plan, and publication
strategy are in `Bot_Detection_Research_Dossier.docx` (v1/v2) and
`Bot_Detection_Research_Dossier_v3.docx` (reframe rationale, definitions,
paper text changes) — both local only.
