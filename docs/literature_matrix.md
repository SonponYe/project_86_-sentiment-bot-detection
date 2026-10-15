# Literature matrix

Every source below was checked against its publisher or arXiv page on 15 Oct 2026. "Use in this project" says exactly which decision each source supports.

| # | Source | Venue | Data | Method | Key finding | Use in this project |
|---|---|---|---|---|---|---|
| 1 | Dickerson, Kagan & Subrahmanian (2014) | IEEE/ACM ASONAM, pp. 620–627 | Twitter, 2014 Indian election | Sentiment + other features, supervised classifiers | Humans are more opinionated and vary their sentiment more than bots; sentiment features improve detection | Main hypothesis (bots are "emotionally flatter"); `polarity_std`, `pct_extreme` features |
| 2 | Cresci, Di Pietro, Petrocchi, Spognardi & Tesconi (2017) | WWW '17 Companion, pp. 963–972 | Cresci-2017 (genuine, social & traditional spambots) | Benchmarks existing detectors, crowdsourcing | "Social spambots" imitate humans and evade older detectors | Dataset; motivation for not relying on metadata alone |
| 3 | Varol, Ferrara, Davis, Menczer & Flammini (2017) | ICWSM, pp. 280–289 | ~14M Twitter accounts | Botometer: 1,000+ features (metadata, network, timing, content, sentiment) | Sentiment is one of several useful feature families | Shows sentiment is used in production detectors; the "many features" contrast to my small model |
| 4 | Kudugunta & Ferrara (2018) | Information Sciences 467, 312–322 | Cresci-2017 | Contextual LSTM on tweet text + account metadata | Combining text representations with metadata gives very high accuracy | Closest design to mine (text-derived signal + metadata), but with deep text models instead of lexicon sentiment |
| 5 | Heidari & Jones (2020) | IEEE UEMCON 2020 | Cresci-2017 | BERT-based, topic-independent sentiment features | Sentiment features that do not depend on topic help bot detection | Motivates topic-independent sentiment statistics; future work (transformer sentiment) |
| 6 | Akhtar et al. (2024) | arXiv:2402.03740 | Several bot datasets | Self-supervised contrastive learning (BotSSCL) | Many detectors rely on easily manipulated profile features and transfer poorly | Motivation for Run 3 (remove dominant metadata) |
| 7 | Zeng, Li & Wang (2025) | Sensors 25(13), 4179 | TwiBot-20 | Emoji-to-text + 7 sentiment features + relational GCN (ESA-BotRGCN) | 87.5% accuracy; sentiment adds to graph models | Recent evidence sentiment still matters; graph models out of scope here |
| 8 | Li et al. (2025) | arXiv:2508.06913; AAAI 40 | LLM vs human text corpora | SentiDetect: stability of sentiment under perturbation | LLM-generated text keeps more stable sentiment than human text | Link from "bots" to "AI-operated" accounts |
| 9 | Di Paolo, Petrocchi & Spognardi (2025) | First Monday 30(5) | LLM-powered bot accounts | Behaviour encoded as images, CNN classification | LLM-powered bots can be detected; little prior work exists | Shows the gap my reframe addresses |
| 10 | Staufer et al. (2026) | arXiv:2602.17753 | 30 deployed AI agents | Structured documentation index | Most deployed agents disclose little about how they work | Motivation (undisclosed AI agents) |
| 11 | Regulation (EU) 2024/1689, Art. 50 | Official Journal of the EU | – | Law | Users must be told when interacting with an AI system | Practical / regulatory motivation |
| 12 | Hutto & Gilbert (2014) | ICWSM, pp. 216–225 | Social media text | VADER lexicon + rules | Strong sentiment accuracy on social media text | Polarity scoring tool |
| 13 | Chawla et al. (2002) | JAIR 16, 321–357 | – | SMOTE oversampling | Synthetic minority samples improve imbalanced learning | Class imbalance handling (train set only) |
| 14 | McNemar (1947) | Psychometrika 12(2), 153–157 | – | Paired test of proportions | Tests disagreement between two classifiers on the same items | Significance test A vs B |
