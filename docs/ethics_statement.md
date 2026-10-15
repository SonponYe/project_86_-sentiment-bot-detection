# Ethics statement

1. **Data.** I used only the public Cresci-2017 research dataset under its research-only terms. I did not collect new data, contact users, or use any platform API. Raw data is excluded from this repository (`.gitignore`) and must be downloaded from the original source.
2. **Privacy.** Tweets and profiles belong to real people. I report only aggregate statistics. No account ID, handle or tweet text appears in the manuscript, figures or saved metrics.
3. **Harm from errors.** Even the best model wrongly flags 16.7% of genuine human accounts in the metadata-constrained setting. Being labelled a bot or AI can damage a person's reputation or lead to account removal. This work must not be used to sanction accounts automatically.
4. **Fairness.** People with naturally steady or neutral writing (professional, institutional or second-language writers) are at greater risk of false flags. Lexicon tools are English-centric and would perform worse for Ghanaian and other non-English users. This was not measured and is a limitation.
5. **Dual use.** Knowing which sentiment features expose automation could help operators make their accounts harder to detect. The features used are already described in published work (Dickerson et al., 2014), so this project adds little new risk.
6. **Secrets.** The repository contains no passwords, API keys or tokens.
7. **AI assistance.** Disclosed in `docs/ai_use_disclosure.md`.
