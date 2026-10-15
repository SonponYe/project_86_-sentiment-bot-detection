# Data card – Cresci-2017 (as used in this project)

**Source:** Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. (2017). *The paradigm-shift of social spambots.* WWW '17 Companion. Distributed by the authors and mirrored by the Bot Repository (Botometer / OSoMe, Indiana University).

**Licence and access:** Research use only. **The raw data is not in this repository and must not be uploaded.** To reproduce, download `cresci-2017.csv.zip` from the Bot Repository (https://botometer.osome.iu.edu/bot-repository/datasets.html), unzip, and place the subset folders in `data/raw/` (see README, "Reproducing"). `data/raw/` and `data/processed/` are git-ignored.

| Item | Value |
|---|---|
| Platform / period | Twitter, collected 2009–2017 by the dataset authors |
| Full dataset | 12,736 accounts, ~6.6 million tweets |
| Subsets used | `genuine_accounts` → label 0 (human-operated); `social_spambots_1–3`, `traditional_spambots_1–4` → label 1 (automated) |
| Subsets excluded | `fake_followers` (different phenomenon), `crowdflower_results` (annotation data, not accounts) |
| Cleaning | Duplicate accounts/tweets removed; accounts with < 10 tweets removed |
| After cleaning | **6,553 accounts: 1,082 human (16.5%), 5,471 automated (83.5%)** |
| Split | Stratified 80/20, seed 42 → test set 1,311 accounts (216 human, 1,095 automated) |
| Text used | 200 most recent tweets per account, English lexicon scoring |
| Personal data | Public tweets and profile fields of real users. Results are reported only in aggregate; no account is named. |

**Known issues (see `docs/failure_log.md`):** invalid UTF-8 bytes in the genuine tweets file; one missing `user_id` that turned the ID column into floats and silently removed all human accounts from the join; slow date parsing. All fixed in code, with regression tests.

**Known biases and limits:**
- The automated accounts are pre-LLM, rule-based spambots. They are a **proxy** for AI-operated accounts, not examples of them.
- Human and spambot subsets were collected at different times, so time-dependent features (e.g. `account_age_days`) can leak collection date.
- Mostly English-language accounts; lexicon tools do not transfer to Ghanaian languages, Pidgin or code-switched text.
- Strong class imbalance (≈5:1).

**Other datasets considered:** TwiBot-20 (only a 100-record unlabelled sample was available) and TwiBot-22 (labels without matching profile/tweet data). Neither was usable.
