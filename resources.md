# Resources: News from the Future

**Research Topic:** A website that combines large language models (LLMs) with prediction markets can generate plausible news articles about future events.

**Domain:** Artificial Intelligence

**Date:** 2026-02-10

---

## 1. Papers (23 total)

### 1.1 Core Papers (Deep-Read)

| # | Citation | Venue | Key Contribution | Local Path |
|---|----------|-------|------------------|------------|
| 1 | Halawi, D., Zhang, F., Yueh-Han, C., & Steinhardt, J. (2024). Approaching Human-Level Forecasting with Language Models. | NeurIPS 2024 | RAG-based forecasting system achieving human-competitive Brier scores across 5 platforms | `papers/halawi2024_approaching_human_level_forecasting.pdf` |
| 2 | Zou, A. et al. (2022). Forecasting Future World Events with Neural Networks. | NeurIPS 2022 D&B | AutoCast benchmark: 6,707 questions + 200GB news corpus with temporal train/test splits | `papers/zou2022_autocast_forecasting_future_world_events.pdf` |
| 3 | Smith, L.N. (2025). Evaluating Epistemic Calibration via Prediction Markets (KalshiBench). | NeurIPS 2024 | 1,531 Kalshi questions revealing systematic LLM overconfidence (15-62% error at 90%+ confidence) | `papers/smith2025_kalshibench_prediction_markets.pdf` |
| 4 | Jain, C. & Flanigan, J. (2024). Future Language Modeling from Temporal Document History. | ICLR 2024 | Temporal bias + gating mechanism for generating plausible future text; 63% vs 46% quality on human eval | `papers/jain2024_future_language_modeling.pdf` |
| 5 | Shahabi, S., Graham, S., & Isah, H. (2026). TruthTensor: Evaluating LLMs through Human Imitation on Prediction Market. | arXiv | 4-layer eval framework; 876K forecasting decisions showing all 8 frontier models have negative P&L | `papers/chen2026_truthtensor_prediction_market.pdf` |
| 6 | Schoenegger, P. et al. (2024). Wisdom of the Silicon Crowd. | JEP:General | LLM ensemble (12 models) median matches human crowd accuracy (Brier 0.20 vs 0.19) | `papers/schoenegger2024_wisdom_silicon_crowd.pdf` |
| 7 | Turtel, B. et al. (2026). Future-as-Label: Scalable Supervision from Real-World Outcomes. | arXiv | Foresight Learning: RL with proper scoring rules on resolved events; 27% Brier improvement, halved ECE | `papers/wu2026_future_as_label.pdf` |

### 1.2 Benchmarks and Evaluation

| # | Citation | Venue | Local Path |
|---|----------|-------|------------|
| 8 | Karger, E. et al. (2024). ForecastBench: Dynamic Benchmark of AI Forecasting. | ICLR 2025 | `papers/karger2024_forecastbench.pdf` |
| 9 | Sun, Z. et al. (2025). PROPHET: Prompting LLMs for Future Forecasting. | arXiv | `papers/sun2025_prophet_future_forecasting.pdf` |
| 10 | Nakano, R. et al. (2025). Scaling Open-Ended Reasoning (OpenForesight). | arXiv | `papers/nakano2025_scaling_open_ended_reasoning.pdf` |
| 11 | Teal, J. et al. (2025). Pitfalls of Evaluating LLM Forecasters. | arXiv | `papers/teal2025_pitfalls_evaluating_forecasters.pdf` |
| 12 | Marois, A. et al. (2024). Consistency Checks for Forecasters. | arXiv | `papers/marois2024_consistency_checks_forecasters.pdf` |
| 13 | Kang, J. et al. (2025). Bench to the Future: Pastcasting. | arXiv | `papers/kang2025_bench_to_the_future_pastcasting.pdf` |

### 1.3 Forecasting Methods

| # | Citation | Venue | Local Path |
|---|----------|-------|------------|
| 14 | Halawi, D. et al. (2023). Forecasting with LLMs (earlier version). | arXiv | `papers/halawi2023_forecasting_with_llms.pdf` |
| 15 | Schoenegger, P. et al. (2023). Large Language Models in Forecasting Tournaments. | arXiv | `papers/schoenegger2023_llm_prediction_tournament.pdf` |
| 16 | Xia, C. et al. (2023). AutoCast++. | arXiv | `papers/xia2023_autocast_plus_plus.pdf` |
| 17 | Zhang, Y. et al. (2024). Can Language Models Use Forecasting Strategies? | arXiv | `papers/zhang2024_can_lm_use_forecasting_strategies.pdf` |
| 18 | Stanton, S. et al. (2025). Outcome-Based RL to Predict the Future. | arXiv | `papers/stanton2025_outcome_rl_predict_future.pdf` |
| 19 | Navarro, D. et al. (2025). Navigating Tomorrow. | arXiv | `papers/navarro2025_navigating_tomorrow.pdf` |

### 1.4 Prediction Markets and LLM Agents

| # | Citation | Venue | Local Path |
|---|----------|-------|------------|
| 20 | Choi, S. et al. (2024). LLMs as Prediction Market Agents. | arXiv | `papers/choi2024_llm_prediction_market_agents.pdf` * |

\* **Note:** This file was found to contain a different paper ("Document Haystacks" by Jun Chen et al.) during deep reading. The correct paper may need to be re-downloaded.

### 1.5 News Generation and Factuality

| # | Citation | Venue | Local Path |
|---|----------|-------|------------|
| 21 | Huertas, P. et al. (2024). News Generation with LLMs. | arXiv | `papers/huertas2024_news_generation.pdf` |
| 22 | Wang, Y. et al. (2023). Survey on Factuality in LLMs. | arXiv | `papers/wang2023_survey_factuality_llm.pdf` |
| 23 | Zheng, L. et al. (2023). Judging LLM-as-a-Judge. | arXiv | `papers/zheng2023_judging_llm_as_judge.pdf` |

---

## 2. Datasets

### 2.1 Downloaded Datasets

| Dataset | Size | Records | Source | Local Path |
|---------|------|---------|--------|------------|
| Forecasting (Halawi et al., 2024) | 11 MB | 5,516 questions (3,762 train / 840 val / 914 test) | [HuggingFace: YuehHanChen/forecasting](https://huggingface.co/datasets/YuehHanChen/forecasting) | `datasets/forecasting_halawi2024/` |
| AutoCast (Zou et al., 2022) | 3.2 MB | 1,364 test + 340 filtered questions | [HuggingFace: valory/autocast](https://huggingface.co/datasets/valory/autocast) | `datasets/autocast/` |
| KalshiBench v2 (Smith, 2025) | 201 KB | 1,531 binary questions | [HuggingFace: 2084Collective/kalshibench-v2](https://huggingface.co/datasets/2084Collective/kalshibench-v2) | `datasets/kalshibench/` |

**Forecasting Dataset Fields:** background, resolution_criteria, question, resolution_date, created_date, source (Metaculus, GJOpen, INFER, Polymarket, Manifold), binary resolution (0/1)

**AutoCast Dataset Fields:** question_id, question, background, qtype (T/F, MCQ, numerical), choices, answer, crowd forecasts, publish_time, close_time, resolution_time

**KalshiBench Fields:** id, question, description, category (Sports, Politics, Entertainment, Companies, Elections, Crypto, Climate, Financials, World, Economics, Social), close_time, ground_truth

### 2.2 Additional Datasets (Not Downloaded - Available via API/Web)

| Dataset | Size | Access | URL |
|---------|------|--------|-----|
| Metaculus Questions API | 10,000+ questions, 3.49M+ predictions | REST API | https://www.metaculus.com/api2/questions/ |
| ForecastBench (Dynamic) | 1,000 questions per biweekly round | GitHub | [forecastingresearch/forecastbench-datasets](https://github.com/forecastingresearch/forecastbench-datasets) |
| Polymarket Trading Data | 1.1B records, 268K+ markets (107 GB) | HuggingFace | [SII-WANGZJ/Polymarket_data](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data) |
| Good Judgment Project (IARPA ACE) | ~500 questions, 1M+ forecasts (2011-2015) | Harvard Dataverse | [doi:10.7910/DVN/BPCDH5](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BPCDH5) |
| YuehHanChen/forecasting_raw | 48,754 questions, 7.17M forecasts (2015-2024) | HuggingFace | [YuehHanChen/forecasting_raw](https://huggingface.co/datasets/YuehHanChen/forecasting_raw) |

---

## 3. Code Repositories

### 3.1 Cloned Repositories (10)

| Repository | Source | Size | Description | Local Path |
|------------|--------|------|-------------|------------|
| AutoCast | [andyzoujm/autocast](https://github.com/andyzoujm/autocast) | 4.0 MB | AutoCast forecasting benchmark + news corpus scripts (NeurIPS 2022) | `code/autocast/` |
| LLM Forecasting | [dannyallover/llm_forecasting](https://github.com/dannyallover/llm_forecasting) | 6.4 MB | Approaching Human-Level Forecasting system (NeurIPS 2024) | `code/llm_forecasting/` |
| ForecastBench | [forecastingresearch/forecastbench](https://github.com/forecastingresearch/forecastbench) | 11 MB | Dynamic contamination-free forecasting benchmark (ICLR 2025) | `code/forecastbench/` |
| PROPHET | [TZWwww/PROPHET](https://github.com/TZWwww/PROPHET) | 231 KB | Polymarket-based future forecasting benchmark | `code/PROPHET/` |
| Forecasting Tools | [Metaculus/forecasting-tools](https://github.com/Metaculus/forecasting-tools) | 9.0 MB | Official Metaculus API wrapper + forecasting bot framework | `code/forecasting-tools/` |
| Polymarket Agents | [Polymarket/agents](https://github.com/Polymarket/agents) | 516 KB | Official Polymarket AI trading agent framework | `code/polymarket-agents/` |
| Future Language Modeling | [jlab-nlp/Future-Language-Modeling](https://github.com/jlab-nlp/Future-Language-Modeling) | 713 MB | Temporal document generation from history (ICLR 2024) | `code/Future-Language-Modeling/` |
| Prediction Market Agent | [gnosis/prediction-market-agent](https://github.com/gnosis/prediction-market-agent) | 13 MB | Cross-platform prediction market agent (Manifold, Polymarket, Omen, Metaculus) | `code/prediction-market-agent/` |
| FActScore | [shmsw25/FActScore](https://github.com/shmsw25/FActScore) | 285 KB | Fine-grained factual accuracy evaluation (EMNLP 2023) | `code/factscore/` |
| Scoring Rules | [frazane/scoringrules](https://github.com/frazane/scoringrules) | 1.1 MB | Proper scoring rules library (Brier, CRPS, Log scores) | `code/scoringrules/` |

### 3.2 Additional Relevant Repositories (Not Cloned)

| Repository | Description |
|------------|-------------|
| [getdatachimp/llm-superforecaster](https://github.com/getdatachimp/llm-superforecaster) | Third-party LLM forecasting for Polymarket/Metaculus/Manifold |
| [Polymarket/py-clob-client](https://github.com/Polymarket/py-clob-client) | Official Polymarket CLOB API client |
| [forecastingresearch/forecastbench-datasets](https://github.com/forecastingresearch/forecastbench-datasets) | ForecastBench question sets and resolution data |
| [PredictionXBT/PredictOS](https://github.com/PredictionXBT/PredictOS) | Multi-agent prediction market trading system |
| [aarora4/Awesome-Prediction-Market-Tools](https://github.com/aarora4/Awesome-Prediction-Market-Tools) | Curated list of prediction market tools |
| [KVishnuVardhanR/News-Article-Generation-using-LLM-s](https://github.com/KVishnuVardhanR/News-Article-Generation-using-LLM-s) | Llama 2 news article generation |
| [IAAR-Shanghai/CTGSurvey](https://github.com/IAAR-Shanghai/CTGSurvey) | Controllable text generation survey |
| [eth-sri/language-model-arithmetic](https://github.com/eth-sri/language-model-arithmetic) | Controlled text generation via LM arithmetic (ICLR 2024) |
| [WanzhengZhu/GRUEN](https://github.com/WanzhengZhu/GRUEN) | Linguistic quality evaluation metric |
| [yecchen/MIRAI](https://github.com/yecchen/MIRAI) | LLM agents for event forecasting |

---

## 4. Prediction Market APIs

| Platform | Type | API Access | Notes |
|----------|------|------------|-------|
| Metaculus | Crowd median/mean (non-monetary) | REST API: `https://www.metaculus.com/api2/questions/` | Most used in academic research; `pip install forecasting-tools` |
| Polymarket | Real-money, blockchain-based | REST API + CLOB client | Largest crypto prediction market; high liquidity |
| Kalshi | CFTC-regulated, real-money | REST API | Legally binding resolutions; used in KalshiBench |
| Manifold | Play-money | REST API | High question diversity; accessible for experimentation |
| Good Judgment Open | Crowd forecasting | Web-based | Superforecaster benchmarks; IARPA ACE historical data |

---

## 5. Evaluation Tools and Metrics

### 5.1 Forecasting Accuracy Metrics

| Metric | Formula / Description | Use Case |
|--------|----------------------|----------|
| Brier Score | BS = (1/N) Σ(pᵢ - yᵢ)² | Standard probabilistic forecasting accuracy (0 = perfect, 0.25 = uninformed) |
| Brier Skill Score | BSS = 1 - (BS / BS_baseline) | Improvement over baseline (positive = better than baseline) |
| Expected Calibration Error | Mean |confidence - accuracy| across bins | Whether stated probabilities match empirical frequencies |
| Log Score | -Σ[yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)] | Strictly proper scoring rule; used in Foresight Learning |
| Overconfidence Rate (OCR@τ) | Error rate among predictions above confidence threshold τ | Measures systematic overconfidence at high-confidence levels |
| CRPS | Continuous Ranked Probability Score | For continuous probability distributions |

### 5.2 Text Quality Metrics

| Metric | Tool/Library | Use Case |
|--------|-------------|----------|
| FActScore | `code/factscore/` | Fine-grained factual accuracy via atomic fact verification |
| Content Perplexity (CPL) | Jain & Flanigan (2024) | Fluency over content words (not function words) |
| Content Meteor | Jain & Flanigan (2024) | Topical alignment with real future documents |
| GRUEN | [WanzhengZhu/GRUEN](https://github.com/WanzhengZhu/GRUEN) | Linguistic quality (grammaticality, non-redundancy, focus, structure/coherence) |
| LLM-as-Judge | Zheng et al. (2023) | Automated evaluation using LLMs as evaluators |
| Proper Scoring Rules | `code/scoringrules/` | Library for Brier, CRPS, Log score computation |

---

## 6. Key Findings Summary

### What Works

1. **RAG-based forecasting** achieves human-competitive Brier scores when combining retrieval with structured reasoning (Halawi et al., 2024).
2. **LLM ensembles** (median of 12 models) match human crowd accuracy on aggregate (Schoenegger et al., 2024).
3. **Foresight Learning** (RL with proper scoring rules on resolved events) achieves 27% Brier improvement and halved calibration error, with 32B model outperforming 7x larger base model (Turtel et al., 2026).
4. **Temporal bias modeling** improves future text generation quality from 46% to 63% on human evaluation (Jain & Flanigan, 2024).
5. **Prediction market prices as context** improve LLM forecasting by 17-28% (Schoenegger et al., 2024).

### Open Challenges

1. **Systematic overconfidence**: All LLMs show 15-62% error rates at 90%+ stated confidence (KalshiBench).
2. **No model beats the market**: All 8 frontier models had negative P&L as live Polymarket agents (TruthTensor).
3. **Tail events**: Forecasting tools are unsuitable for black swan events (Zou et al., 2022).
4. **Novel entities**: Future news contains people/organizations/concepts absent from training data.
5. **Temporal granularity**: Benchmarks operate on weeks-to-months horizons; compelling news requires daily resolution.
6. **Ethical risks**: Plausible future news could enable market manipulation, misinformation, or false certainty.

---

## 7. Workspace Structure

```
/workspaces/news-from-future-ai-aed4-claude/
├── literature_review.md          # Comprehensive literature review
├── resources.md                  # This file - complete resource catalog
├── pyproject.toml                # Workspace configuration
├── papers/                       # 23 downloaded academic papers
│   ├── README.md                 # Paper inventory and metadata
│   ├── pages/                    # Chunked PDFs for deep reading
│   └── *.pdf                     # Individual papers
├── datasets/                     # Downloaded datasets
│   ├── README.md                 # Dataset documentation
│   ├── forecasting_halawi2024/   # 5,516 forecasting questions (11 MB)
│   ├── autocast/                 # 1,704 forecasting questions (3.2 MB)
│   └── kalshibench/              # 1,531 Kalshi questions (201 KB)
├── code/                         # 10 cloned code repositories
│   ├── README.md                 # Repository documentation
│   ├── autocast/                 # AutoCast benchmark code
│   ├── llm_forecasting/          # Halawi et al. forecasting system
│   ├── forecastbench/            # ForecastBench benchmark
│   ├── PROPHET/                  # PROPHET benchmark
│   ├── forecasting-tools/        # Metaculus API wrapper
│   ├── polymarket-agents/        # Polymarket agent framework
│   ├── Future-Language-Modeling/  # Temporal text generation
│   ├── prediction-market-agent/  # Cross-platform market agent
│   ├── factscore/                # Factual accuracy evaluation
│   └── scoringrules/             # Proper scoring rules library
├── paper_search_results/         # Search logs and surveys
│   └── github_repos_survey.md    # Detailed repository survey
└── logs/                         # Execution logs
```

---

## 8. Suggested Next Steps for Research

1. **Prototype pipeline**: Combine `code/llm_forecasting/` (RAG forecasting) with `code/forecasting-tools/` (Metaculus API) to build a probability estimation module.
2. **Calibration training**: Implement Foresight Learning (Turtel et al., 2026) using `code/scoringrules/` and resolved market data as training labels.
3. **Temporal generation**: Adapt `code/Future-Language-Modeling/` from academic abstracts to news articles using CommonCrawl News corpus.
4. **Evaluation framework**: Use `code/factscore/` for factual accuracy and `code/scoringrules/` for probability calibration assessment.
5. **Market integration**: Use `code/polymarket-agents/` or `code/prediction-market-agent/` for live market data feeds and probability anchoring.
6. **Benchmark validation**: Test forecasting accuracy on downloaded datasets (`datasets/forecasting_halawi2024/`, `datasets/kalshibench/`) before generating news articles.
