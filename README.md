# Sentiment Consistency as a Signal for Detecting AI-Operated Accounts

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
`findings/` is the durable record — full tables (McNemar significance,
feature importance, per-model deltas) are in each linked file.

| Run | What changed | Headline finding |
|---|---|---|
| [1 — unscaled](findings/2026-08-28_run1_unscaled_features.md) | Baseline: Feature Set A (6 metadata features) vs. B (+8 sentiment features), no feature scaling | Outcome B (mixed). Logistic Regression and Random Forest improve significantly from sentiment; SVM shows *exactly zero* effect (identical metrics to 6 decimals) — the signature of a feature being invisible to the model, not a real null result. |
| [2 — scaled](findings/2026-08-28_run2_scaled_features.md) | Added `StandardScaler` (fit on train only, applied before SMOTE) | Confirmed Run 1's SVM null was a scaling artifact: unscaled metadata (`followers_count` in the millions) was drowning out sentiment features (~[-1,1]) in SVM's distance-based kernel. Scaled, SVM shows a real, significant improvement (+2.1 F1, p=6.4e-6). |
| [3 — metadata-constrained](findings/2026-08-28_run3_metadata_constrained.md) | Removed the 4 metadata features that dominated importance rankings (`followers_count`, `friends_count`, `statuses_count`, `account_age_days`) — simulating accounts whose metadata looks normal (e.g. sophisticated bots/LLM agents). Added 3 new sentiment features: `contextual_consistency_score`, `temporal_sentiment_drift`, `sentiment_topic_alignment_score`. | Strongest confirmation yet: Random Forest F1 jumps from 0.778 to 0.957 (+17.9 points, p=1.98e-60) once sentiment is the only usable signal. `polarity_std` reaches the top-5 most important features for the first time — the exact effect Dickerson et al. (2014) predicted, previously drowned out by metadata dominance. The three new features underperform, ranking last (10th-12th of 13). |

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
current loader doesn't support — they're optional later steps, not needed
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
1. Load and clean the accounts (drop accounts with <10 tweets, dedupe).
2. Build Feature Set A (metadata only) and Feature Set B (metadata +
   sentiment: mean/variance of VADER polarity, % extreme-sentiment tweets,
   sentiment class distribution, TextBlob subjectivity mean/variance).
3. Take one stratified 80/20 train/test split shared by both feature sets
   (`random_state=42`), so the two feature sets are compared on identical
   examples. SMOTE, if the class balance warrants it, is fit on the
   training portion only.
4. Train Logistic Regression, Random Forest, and SVM on each feature set
   (6 experiments total), with 5-fold stratified CV (F1) on the training
   set.
5. Write `results/tables/experiment_summary.csv` (accuracy, precision,
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

v1/v2 backbone:
- Dickerson, Kagan & Subrahmanian (2014), "Using Sentiment to Detect Bots
  on Twitter," IEEE/ACM ASONAM — origin of the sentiment-variance signal.
- Xia et al. (2022), "A Method of Machine Learning for Social Bot Detection
  Combined with Sentiment Analysis," ACM MLNLP — closest related method.
- Heidari & Jones (2020), "Using BERT to Extract Topic-Independent
  Sentiment Features for Social Media Bot Detection," IEEE UEMCON — same
  dataset (Cresci-2017), a direct benchmark.

v3 reframe — theoretical backbone and gap justification:
- Li et al. (2025), "Model-Agnostic Sentiment Distribution Stability
  Analysis for Robust LLM-Generated Texts Detection," arXiv:2508.06913 —
  LLM outputs show emotionally consistent patterns; humans show greater
  variability. Direct theoretical grounding for this study's core signal.
- Di Paolo, Spognardi & Petrocchi (2025), "Detection of LLM-Powered Bots
  Using Image Classification," First Monday — establishes the gap: very
  few studies address LLM-powered bot/account detection specifically.
- Zeng, Li & Wang (2025), "Emoji-Driven Sentiment Analysis for Social Bot
  Detection with Relational Graph Convolutional Networks (ESA-BotRGCN),"
  Sensors (MDPI) — sentiment as a legitimate, effective detection signal.
- BotSSCL (2025), "Social Bot Detection with Self-Supervised Contrastive
  Learning," ScienceDirect — peer-reviewed confirmation that bot posting
  has minimal sentiment change due to focused, consistent objectives.
- Staufer et al. (2026), "The 2025 AI Agent Index," ACM FAccT — documents
  agentic AI systems deployed without standardised transparency, motivating
  why account-level AI verification matters now.
- Cresci et al. (2017), "The Paradigm-Shift of Social Spambots," WWW '17 —
  the foundational Cresci-2017 dataset paper; cite whenever the dataset is
  referenced.

Full literature review, gap analysis, evaluation plan, and publication
strategy are in `Bot_Detection_Research_Dossier.docx` (v1/v2) and
`Bot_Detection_Research_Dossier_v3.docx` (reframe rationale, definitions,
paper text changes) — both local only.
