# Model card – sentiment-consistency account classifier

**Task:** binary classification of a social media account as human-operated (0) or automated / AI-operated (1).

**Models:** scikit-learn Logistic Regression, Random Forest (300 trees), SVM (RBF kernel). Each trained on Feature Set A (metadata only) and Feature Set B (metadata + sentiment). StandardScaler fitted on the training split only (Runs 2–3); SMOTE on the training split only.

**Inputs:** Feature Set A – Runs 1–2: `followers_count, friends_count, statuses_count, listed_count, verified, account_age_days`; Run 3: `verified, listed_count`. Feature Set B adds 8 sentiment statistics (VADER polarity, TextBlob subjectivity) and, in Run 3, 3 experimental consistency features.

**Recommended configuration:** Random Forest, Feature Set B, Run 3 (metadata-constrained), because it does not depend on easily faked profile counts.

**Performance (test set, 1,311 accounts):**

| Configuration | F1 (automated) | Balanced accuracy | Humans wrongly flagged |
|---|---|---|---|
| Majority-class baseline | 0.910 | 0.500 | 216 / 216 (100%) |
| RF, Run 3, Set A | 0.778 | 0.711 | 53 / 216 (24.5%) |
| **RF, Run 3, Set B** | **0.957** | **0.891** | **36 / 216 (16.7%)** |
| RF, Run 2, Set B (full metadata) | 0.987 | not archived | not archived |

Source: `results/metrics/`, `findings/`, `notebooks/results_summary.ipynb`.

**Intended use:** research and teaching; a *screening signal* that flags accounts for human review, or supports AI-disclosure auditing.

**Out-of-scope use:** automatic suspension, banning or public labelling of accounts; use on non-English text; claims about any specific named person.

**Ethical considerations:** In the best configuration about 1 in 6 genuine human accounts is still flagged as automated. People who post in a steady, neutral or professional tone (journalists, organisations, people writing in a second language) are the most likely to be wrongly flagged. A human must review every flag.

**Caveats:** trained and tested on one 2017 dataset of pre-LLM spambots; not validated on real LLM-operated accounts; one train/test split; lexicon-based sentiment.
