# Failure log

Problems found during the project, in the order they occurred, with the fix and the evidence.

| # | Date | What went wrong | How it was found | Fix | Commit / evidence |
|---|---|---|---|---|---|
| 1 | Aug 2026 | Loader crashed on bytes that are not valid UTF-8 in `genuine_accounts/tweets.csv` | First real-data run crashed | Replace only invalid bytes instead of failing | `4a7450c`, `tests/test_data_loader.py` |
| 2 | Aug 2026 | Sentiment step projected to take 3.5+ hours | Progress bars (`e4f0a5c`) showed very slow per-account timing | pandas could not infer Twitter's date format and parsed row-by-row, repeatedly inside the loop. Parse once, with explicit format. 2.84M tweets now take 36 s | `8864a2b` |
| 3 | Aug 2026 | Scoring all 6.6M tweets too slow on a laptop / Colab | Runtime estimate | Cap at 200 most recent tweets per account (documented limitation) | `75ca5be` |
| 4 | Aug 2026 | **All human accounts silently disappeared**; training failed with "data contains only one class" | Added class-balance logging | One missing `user_id` made pandas store IDs as floats (678033 → 678033.0), so no genuine profile matched its tweets. Normalised ID dtype; overlap went 0 → 1,083 | `4f74776`, `a4786f5`, regression test |
| 5 | 28 Aug 2026 | Run 1: SVM metrics identical to 6 decimals with and without sentiment | Suspicious exact-zero change | Unscaled metadata (millions) swamped [-1, 1] sentiment in the RBF kernel. Added StandardScaler before SMOTE (Run 2) | `2464a67`, `findings/…run2…` |
| 6 | 28 Aug 2026 | `report.py` gave the wrong Outcome-B narrative when all models improved | Read the generated report | Fixed narrative logic | `aa79dc0` |
| 7 | 28 Aug 2026 | Run 3: `report.py` labels Logistic Regression "no improvement" because F1 fell, although ROC-AUC rose 0.774 → 0.860 | Manual check of the numbers | Not fixed in code. Documented in findings and manuscript; listed as future work | `findings/…run3…` |
| 8 | Aug 2026 | TwiBot-20 / TwiBot-22 unusable as a second dataset | Inspection of files | Documented as a limitation | README |
| 9 | Oct 2026 | Pipeline figures (`results/`) are overwritten each run and were never committed; notebooks were committed without outputs | Preparing final submission | Added `results/metrics/` (saved metrics) and the executed `notebooks/results_summary.ipynb`, which regenerates the figures without raw data | final submission commit |
| 10 | Oct 2026 | No trivial baseline had been reported; Logistic Regression on metadata alone (F1 0.907) turned out to be slightly *below* the majority-class F1 (0.910) | Added majority-class baseline | Reported in manuscript; error analysis uses balanced accuracy and human recall as well as F1 | `results/metrics/baseline_majority.csv` |
