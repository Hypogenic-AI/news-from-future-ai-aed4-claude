# GitHub Repository Survey: LLMs + Prediction Markets + Future News Generation

*Compiled: 2026-02-10*

---

## 1. Forecasting with LLMs

### 1.1 Approaching Human-Level Forecasting (Halawi et al.)

**Repository:** [dannyallover/llm_forecasting](https://github.com/dannyallover/llm_forecasting)
- **Description:** Official code for the paper "Approaching Human-Level Forecasting with Language Models" (Halawi, Zhang, Yueh-Han, Steinhardt, 2024). Retrieval-augmented LM system that automatically searches for relevant information, generates forecasts, and aggregates predictions.
- **Key Features:**
  - 5-stage pipeline: search query generation, article relevancy assessment, summarization, reasoning+prediction, forecast aggregation
  - Uses GPT-4-1106 and GPT-3.5
  - Dataset from 5 forecasting platforms (Metaculus, Good Judgment Open, INFER, Polymarket, Manifold)
  - Approaches human crowd-level Brier score; surpasses crowd in selective setting
  - Dataset on HuggingFace
- **Language:** Python
- **Relevance:** Core reference implementation for LLM-based forecasting systems. Directly applicable architecture.

**Third-party implementation:** [getdatachimp/llm-superforecaster](https://github.com/getdatachimp/llm-superforecaster)
- Generates forecasts for Yes/No questions like those on Polymarket, Metaculus, and Manifold.

---

### 1.2 AutoCast / AutoCast++

**Repository:** [andyzoujm/autocast](https://github.com/andyzoujm/autocast)
- **Description:** "Forecasting Future World Events with Neural Networks" (NeurIPS 2022). Dataset of thousands of forecasting questions from forecasting tournaments plus accompanying news corpus.
- **Key Features:**
  - Questions include: id, question, background, qtype, status, choices, answer, crowd forecasts, publish/close times
  - News corpus organized by date (avoids future leakage)
  - IntervalQA dataset for numerical calibration
  - Demonstrated LLM performance far below human expert baseline at time of publication
- **Language:** Python
- **Relevance:** Foundational benchmark dataset for forecasting. The news corpus + question pairing is directly relevant to generating future news conditioned on forecasting questions.

**Competition fork:** [smritae01/autocast_competition](https://github.com/smritae01/autocast_competition)

---

### 1.3 ForecastBench

**Repository:** [forecastingresearch/forecastbench](https://github.com/forecastingresearch/forecastbench)
- **Description:** A dynamic, contamination-free benchmark for LLM forecasting accuracy with human comparison groups. Published at ICLR 2025.
- **Stars:** Active project with regular updates
- **Key Features:**
  - Nightly data ingestion (00:00 UTC); biweekly question sampling (1000 for LLMs, 200 for humans)
  - Contamination-free: only questions about future events with no known answer at submission time
  - Two leaderboards: Baseline (out-of-box) and Tournament (any enhancement allowed)
  - Difficulty-adjusted Brier scores
  - Key finding: GPT-4.5 achieves 0.101 Brier vs superforecasters' 0.081; projected parity in late 2026
- **Language:** Python
- **Website:** [forecastbench.org](https://www.forecastbench.org/)
- **Relevance:** Gold-standard benchmark for evaluating LLM forecasting. Provides ongoing calibration data.

**Datasets repo:** [forecastingresearch/forecastbench-datasets](https://github.com/forecastingresearch/forecastbench-datasets)

---

### 1.4 LLM Forecasting Agent Frameworks

**Repository:** [yecchen/MIRAI](https://github.com/yecchen/MIRAI)
- **Description:** "MIRAI: Evaluating LLM Agents for Event Forecasting." Uses GDELT event database for relational prediction tasks with varying horizons.
- **Key Features:**
  - Evaluates agents on: sourcing information from global databases, writing code with domain-specific APIs, reasoning over historical knowledge
  - Short-term to long-term forecasting assessment
- **Language:** Python
- **Relevance:** Agent-based event forecasting evaluation framework.

**Repository:** [Y-Research-SBU/TimeSeriesScientist](https://github.com/Y-Research-SBU/TimeSeriesScientist)
- **Description:** First LLM-driven agentic framework for general time series forecasting using multi-agent workflow (Curator, Planner, Forecaster, Reporter) orchestrated by LangGraph.
- **Language:** Python
- **Relevance:** Multi-agent architecture pattern applicable to forecasting pipeline design.

---

## 2. Prediction Market APIs / Tools

### 2.1 Metaculus

**Repository:** [Metaculus/forecasting-tools](https://github.com/Metaculus/forecasting-tools)
- **Description:** Official framework for building AI Forecasting Bots for Metaculus. Python + Streamlit.
- **Key Features:**
  - Metaculus API wrapper (pydantic objects for Binary, Multiple Choice, Numeric, Date questions)
  - Pre-built bot types: MainBot (most accurate) and TemplateBot (simpler/cheaper)
  - Benchmarking against community predictions using expected baseline scores
  - Smart Searcher (Exa.ai + GPT powered), Key Factor Analysis, Base Rate Researcher, Fermi Estimator
  - 30-minute setup tutorial; pip installable (`forecasting-tools`)
  - Tournament IDs for Q4 2024, Q1 2025 AI Benchmarking
- **Language:** Python
- **Relevance:** Most directly useful for building forecasting bots that interact with Metaculus. The API wrapper and bot architecture are ready to use.

**Repository:** [Metaculus/metaculus](https://github.com/Metaculus/metaculus)
- **Description:** Main Metaculus platform codebase (Django, Postgres, Redis).
- **Relevance:** Reference for understanding data models and API internals.

**Repository:** [oughtinc/ergo](https://github.com/oughtinc/ergo)
- **Description:** Python library for integrating model-based and judgmental forecasting. Direct Metaculus API integration. No longer actively developed.
- **Relevance:** Historical reference; some useful abstractions for distribution manipulation.

---

### 2.2 Polymarket

**Repository:** [Polymarket/agents](https://github.com/Polymarket/agents)
- **Description:** Trade autonomously on Polymarket using AI Agents. Official Polymarket project.
- **Stars:** ~2,100
- **Key Features:**
  - Interact with Polymarket API, retrieve news, query local data, send to LLMs, execute trades
  - Modular architecture with Chroma DB for vectorizing news, GammaMarketClient for market metadata
  - MIT License
- **Language:** Python
- **Relevance:** Reference architecture for LLM-powered prediction market agents.

**Repository:** [Polymarket/py-clob-client](https://github.com/Polymarket/py-clob-client)
- **Description:** Official Python client for Polymarket CLOB (Central Limit Order Book).
- **Stars:** ~773
- **Key Features:** Get prices, trades, manage API credentials, Polygon chain integration
- **Language:** Python
- **Relevance:** Core API client for reading Polymarket data.

**Repository:** [warproxxx/poly_data](https://github.com/warproxxx/poly_data)
- **Description:** Comprehensive data pipeline for fetching, processing, and analyzing Polymarket trading data.
- **Language:** Python
- **Relevance:** Useful for historical market data analysis.

---

### 2.3 Kalshi

**Official:** [Kalshi/kalshi-starter-code-python](https://github.com/Kalshi/kalshi-starter-code-python)
- **Description:** Official Kalshi API starter code with RSA authentication, rate limiting, HTTP handlers for exchange/markets/portfolio endpoints.
- **Language:** Python

**Repository:** [ArshKA/kalshi-client (pykalshi)](https://github.com/ArshKA/kalshi-client)
- **Description:** Unofficial Python client with WebSocket streaming, automatic retries, pandas integration.
- **Relevance:** Production-ready client with real-time data capabilities.

**Repository:** [the-odds-company/aiokalshi](https://github.com/the-odds-company/aiokalshi)
- **Description:** Asyncio-native Kalshi client with typed query parameters and Pydantic response models.
- **Language:** Python
- **Relevance:** Clean async interface for high-throughput data collection.

---

### 2.4 Manifold Markets

**Repository:** [manifoldmarkets/manifold](https://github.com/manifoldmarkets/manifold)
- **Description:** Main Manifold Markets monorepo (Vercel-hosted, migrating from Firebase to Supabase/SQL).
- **Language:** TypeScript
- **Relevance:** Platform source code; useful for understanding data models.

**Repository:** [bcongdon/PyManifold](https://github.com/bcongdon/PyManifold)
- **Description:** Python API client for Manifold Markets (work in progress).
- **Language:** Python

**Repository:** [vluzko/manifoldpy](https://github.com/vluzko/manifoldpy)
- **Description:** Python tools for Manifold Markets. No longer maintained; Manifold now releases full data dumps.
- **Language:** Python
- **Relevance:** Data dump approach may be more reliable than API for historical analysis.

**Repository:** [bmorphism/manifold-mcp-server](https://github.com/bmorphism/manifold-mcp-server)
- **Description:** MCP (Model Context Protocol) server for interacting with Manifold Markets prediction markets.
- **Relevance:** Modern LLM-tool integration pattern.

---

### 2.5 Cross-Platform / Aggregators

**Repository:** [gnosis/prediction-market-agent](https://github.com/gnosis/prediction-market-agent)
- **Description:** Library for AI agent frameworks applied to prediction market betting. Supports Manifold, Polymarket, Omen, and Metaculus.
- **Key Features:**
  - Multiple agent types: think_thoroughly_prophet, microchain, metaculus_bot_tournament_agent, prophet_gpt4o
  - Built on gnosis/prediction-market-agent-tooling
- **Language:** Python
- **Relevance:** Multi-platform agent framework; closest to a unified prediction market agent library.

**Repository:** [PredictionXBT/PredictOS](https://github.com/PredictionXBT/PredictOS)
- **Description:** Multi-agent AI system with Supervised and Autonomous modes. Cross-platform arbitrage detection between Polymarket and Kalshi.
- **Key Features:** Multiple AI agents, Bookmaker Agent aggregation, Dome unified API
- **Language:** Python
- **Relevance:** Advanced multi-agent trading framework with cross-platform capabilities.

**Repository:** [aarora4/Awesome-Prediction-Market-Tools](https://github.com/aarora4/Awesome-Prediction-Market-Tools)
- **Description:** Curated list of prediction market tools (AI agents, analytics, APIs, dashboards).
- **Relevance:** Meta-resource for discovering additional tools. Lists PolyRouter (unified API), Dome, PMXT, Oddpool, and many others.

**Repository:** [0xperp/awesome-prediction-markets](https://github.com/0xperp/awesome-prediction-markets)
- **Description:** Curated list of prediction market platforms and resources.

---

## 3. News Generation

### 3.1 LLM-Based News Article Generation

**Repository:** [KVishnuVardhanR/News-Article-Generation-using-LLM-s](https://github.com/KVishnuVardhanR/News-Article-Generation-using-LLM-s)
- **Description:** Uses Llama 2 for news article generation based on MediaStack API data. Fine-tuning with QLoRA.
- **Language:** Python
- **Relevance:** Direct example of fine-tuning an LLM for news article generation.

**Repository:** [JsnDg/EMNLP23-LLM-headline](https://github.com/JsnDg/EMNLP23-LLM-headline)
- **Description:** EMNLP 2023 paper on "Harnessing the Power of LLMs: Evaluating Human-AI Text Co-Creation through the Lens of News Headline Generation."
- **Key Features:** 840 headlines from 6 conditions; 2400 ratings dataset
- **Language:** Python
- **Relevance:** Evaluation methodology for LLM-generated news content quality.

**Repository:** [rokbenko/crew-news](https://github.com/rokbenko/crew-news)
- **Description:** CrewNews generates unbiased news articles using Llama 3.1, CrewAI agents, Exa search, and Firecrawl scraping.
- **Language:** Python
- **Relevance:** Multi-agent news generation pattern using CrewAI.

**Repository:** [dmgolembiowski/AI-news](https://github.com/dmgolembiowski/AI-news)
- **Description:** Generate realistic news articles mimicking CNN, NYTimes, Fox News styles. Uses LSTM/RNN architecture.
- **Relevance:** Focused on generating plausible news articles; includes plans for human detection surveys.

---

### 3.2 Controllable Text Generation

**Repository:** [IAAR-Shanghai/CTGSurvey](https://github.com/IAAR-Shanghai/CTGSurvey)
- **Description:** Survey paper repo: "Controllable Text Generation for Large Language Models." Covers training-stage and inference-stage methods.
- **Key Features:** Categorized papers by Type, Phase, Classification. Covers sentiment, style, and safety control.
- **Relevance:** Comprehensive overview of techniques for controlling LLM output style/content.

**Repository:** [eth-sri/language-model-arithmetic](https://github.com/eth-sri/language-model-arithmetic)
- **Description:** "Controlled Text Generation via Language Model Arithmetic" (ICLR 2024). Combine prompts, models, and classifiers to create precisely controlled LLMs.
- **Language:** Python
- **Relevance:** Technique for fine-grained control over generated text attributes.

**Repository:** [jxzhangjhu/awesome-LLM-controlled-decoding-generation](https://github.com/jxzhangjhu/awesome-LLM-controlled-decoding-generation)
- **Description:** Curated paper list for controllable/constrained decoding in LLMs. Includes FUDGE, PPLM, and many 2024 papers.
- **Relevance:** Reference list for controllable generation techniques.

---

### 3.3 Fake/Synthetic News Generation & Detection

**Repository:** [RobinSmits/FakeNews-Generator-And-Detector](https://github.com/RobinSmits/FakeNews-Generator-And-Detector)
- **Description:** Train T5 to generate fake news articles from headlines; RoBERTa classifier to detect them (99% accuracy).
- **Language:** Python
- **Relevance:** Generator-detector pair useful for plausibility evaluation of generated news.

**Repository:** [KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
- **Description:** Dataset for fake news detection research. PolitiFact and GossipCop sources with social media data.
- **Relevance:** Benchmark dataset for evaluating generated news plausibility.

---

## 4. Evaluation Tools

### 4.1 Brier Score / Calibration

**Repository:** [properscoring/properscoring](https://github.com/properscoring/properscoring)
- **Description:** Proper scoring rules in Python. CRPS and Brier Score implementations. Originally from The Climate Corporation.
- **Key Features:** Brier score, threshold Brier score, CRPS; Numba acceleration (20x speedup)
- **Language:** Python (NumPy, SciPy)
- **Relevance:** Standard library for forecast evaluation with proper scoring rules.

**Repository:** [frazane/scoringrules](https://github.com/frazane/scoringrules)
- **Description:** Lightweight probabilistic forecast evaluation library. Python port of R's `scoringRules`.
- **Key Features:** Brier, Log, Ranked Probability, CRPS, Dawid-Sebastiani, Energy, Variogram, Gaussian Kernel scores. Supports numpy/numba, jax, pytorch, tensorflow backends.
- **Language:** Python (>=3.10)
- **Relevance:** Most comprehensive scoring library; multi-backend support useful for integration.

**Repository:** [flimao/briercalc](https://github.com/flimao/briercalc)
- **Description:** Calculates Brier scores, Brier skill scores, calibration, resolution, and uncertainty for multiple classes.
- **Language:** Python
- **Relevance:** Focused tool for Brier score decomposition.

**Repository:** [yhoiseth/python-prediction-scorer](https://github.com/yhoiseth/python-prediction-scorer)
- **Description:** Python library to score predictions. Also available as REST API.
- **Language:** Python
- **Relevance:** Lightweight prediction scoring alternative.

---

### 4.2 Factuality / Hallucination Evaluation

**Repository:** [shmsw25/FActScore](https://github.com/shmsw25/FActScore)
- **Description:** "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation" (EMNLP 2023). Breaks generation into atomic facts and checks against knowledge source.
- **Key Features:**
  - Decomposes text into atomic facts, verifies each against knowledge base (e.g., Wikipedia)
  - retrieval+ChatGPT and retrieval+llama+npm modes (0.99 Pearson correlation)
  - Custom knowledge sources supported (.jsonl format)
  - pip installable (`factscore`)
- **Language:** Python
- **Relevance:** Directly applicable to evaluating factual accuracy of generated future news articles.

**Repository:** [lflage/OpenFActScore](https://github.com/lflage/OpenFActScore)
- **Description:** Open-source variant of FActScore using open-usage models. Modular and extensible.
- **Language:** Python
- **Relevance:** FActScore without proprietary model dependency.

**Repository:** [amazon-science/RefChecker](https://github.com/amazon-science/RefChecker)
- **Description:** Detects hallucinations by extracting knowledge triplets (subject, predicate, object) and comparing against references.
- **Key Features:** Modular 3-stage pipeline; fine-grained claim extraction
- **Language:** Python
- **Relevance:** Structured factuality checking with knowledge graph-style representation.

**Repository:** [intuit/sac3](https://github.com/intuit/sac3)
- **Description:** SAC3: Reliable Hallucination Detection via Semantic-aware Cross-check Consistency. Works with black-box LLMs.
- **Relevance:** Cross-checking approach useful when multiple sources/models are involved.

**Repository:** [zjunlp/FactCHD](https://github.com/zjunlp/FactCHD)
- **Description:** Fact-Conflicting Hallucination Detection benchmark (IJCAI 2024). 51,383 training + 6,960 evaluation samples across multiple factuality patterns.
- **Relevance:** Benchmark for evaluating fact-checking capabilities.

**Repository:** [EdinburghNLP/awesome-hallucination-detection](https://github.com/EdinburghNLP/awesome-hallucination-detection)
- **Description:** Curated list of hallucination detection papers covering factuality and faithfulness.
- **Relevance:** Meta-resource for staying current on detection methods.

---

### 4.3 Text Quality / Plausibility Evaluation

**Repository:** [WanzhengZhu/GRUEN](https://github.com/WanzhengZhu/GRUEN)
- **Description:** GRUEN metric for evaluating linguistic quality of generated text (EMNLP 2020 Findings). Captures four linguistic dimensions.
- **Key Features:** Correlates well with human judgments on 13 datasets across 5 NLG tasks
- **Language:** Python
- **Relevance:** Automated metric for evaluating linguistic quality of generated news articles.

**Repository:** [thu-coai/OpenMEVA](https://github.com/thu-coai/OpenMEVA)
- **Description:** Benchmark for evaluating open-ended story generation metrics (ACL 2021). Extensible toolkit with data perturbation techniques.
- **Language:** Python
- **Relevance:** Evaluation framework adaptable to news story generation quality assessment.

---

## Summary: Recommended Stack for Research Project

| Component | Recommended Repo(s) | Purpose |
|-----------|---------------------|---------|
| Forecasting Pipeline | dannyallover/llm_forecasting | Core LLM forecasting architecture |
| Forecasting Benchmark | forecastingresearch/forecastbench | Evaluation against human forecasters |
| Forecasting Dataset | andyzoujm/autocast | Training/eval data with news corpus |
| Metaculus Integration | Metaculus/forecasting-tools | API wrapper + bot framework |
| Polymarket Data | Polymarket/py-clob-client + agents | Market data + agent framework |
| Multi-platform Agent | gnosis/prediction-market-agent | Cross-platform prediction agent |
| Calibration Scoring | properscoring or frazane/scoringrules | Brier score + proper scoring rules |
| Factuality Eval | shmsw25/FActScore | Atomic fact verification |
| Text Quality | WanzhengZhu/GRUEN | Linguistic quality assessment |
| Controllable Gen | eth-sri/language-model-arithmetic | Style/attribute control for generation |
