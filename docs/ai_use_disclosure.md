# AI-use disclosure

| Stage | AI tool use | What I did myself / how I verified it |
|---|---|---|
| Research question and framing | None for the original question. Discussed the "bots → AI-operated accounts" reframe with an AI assistant. | The question, definitions and decision to reframe are mine (commit `f38a6b0`). |
| Literature review | AI assistant used to help find and summarise papers. | Every reference was checked against its publisher or arXiv page. Two unsupported claims from earlier drafts were corrected (an uncited "Bi-LSTM study" is now cited as Kudugunta & Ferrara 2018; Heidari & Jones 2020 now has a full reference; Staufer et al. 2026 is cited as the arXiv paper). |
| Code | An AI coding assistant helped write the pipeline in `src/` and the tests. | I ran every run myself on the real data, read the logs, and diagnosed the three data bugs (`docs/failure_log.md`). 14 tests pass (`pytest -q`). |
| Interpretation | AI assistant helped draft explanations. | The decisions to add scaling (Run 2) and remove dominant metadata (Run 3), and the reading of the SVM and Logistic Regression results, were checked against the archived numbers in `findings/`. |
| Manuscript, slides and documentation | Drafted and revised with AI assistance (Claude) from my repository, findings and commit history. The final revision added the majority-class baseline, the error analysis, the figures, and these documentation files. | Every number in the manuscript is traceable to `findings/` or `results/metrics/`. `notebooks/results_summary.ipynb` re-computes the baseline and confusion matrices and asserts they match the manuscript. |

I can explain and defend every decision, line of code and result in this repository.
