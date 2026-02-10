# Literature Review: News from the Future

**Research Topic:** A website that combines large language models (LLMs) with prediction markets can generate plausible news articles about future events.

**Domain:** Artificial Intelligence

**Date:** 2026-02-10

---

## 1. Introduction

This literature review examines the intersection of three research areas: (1) LLM-based forecasting of future events, (2) prediction markets as probability aggregation mechanisms, and (3) automated generation of plausible future-oriented text. The hypothesis under investigation is that combining LLMs with prediction market data can produce plausible news articles about events that have not yet occurred. We organize the review into four thematic sections covering forecasting benchmarks and methods, prediction market integration, temporal text generation, and evaluation methodologies.

---

## 2. LLM-Based Forecasting of Future Events

### 2.1 Foundational Benchmarks

The AutoCast benchmark (Zou et al., 2022) established the first large-scale dataset for evaluating ML models on real-world forecasting questions. It contains 6,707 questions from Metaculus, Good Judgment Open, and CSET Foretell, alongside a 200GB+ news corpus from CommonCrawl organized by publication date. The temporal train/test split (cutoff: May 2021) prevents information leakage. Key findings include that retrieval-augmented models (FiD Static, FiD Temporal) substantially outperform non-retrieval baselines, but even the best model (65.4% T/F accuracy) remains far below human crowd performance (92.4%). The paper introduced the critical methodological principle of date-organized corpora for preventing future information leakage in retrodiction experiments.

ForecastBench (Karger et al., 2024; ICLR 2025) advances this with a dynamic, contamination-free benchmark that generates 1,000 new questions every two weeks from 9 sources including prediction markets (Manifold, Metaculus, Polymarket) and time-series data (ACLED, FRED, Yahoo Finance). Questions span 7- to 180-day forecast horizons. As of early 2026, GPT-4.5 achieves a Brier score of 0.101 versus superforecasters' 0.081, with model-human parity projected for late 2026.

PROPHET (Sun et al., 2025) introduces a Polymarket-derived benchmark with 4,631 binary questions filtered using a novel Causal Intervened Likelihood (CIL) metric to assess question inferability. The CIL framework addresses a fundamental limitation of earlier benchmarks: many questions may be inherently unpredictable (e.g., "Will a specific coin flip come up heads?"), making Brier score comparisons misleading without controlling for question difficulty.

KalshiBench (Smith, 2025; NeurIPS 2024) provides 1,531 questions from the CFTC-regulated Kalshi exchange, temporally filtered to post-training-cutoff dates. The most striking finding is systematic overconfidence: at the 90%+ confidence level, models are wrong 15-62% of the time (GPT-5.2-XHigh achieves only 33.7% accuracy on predictions where it reports 95.9% confidence). Only Claude Opus 4.5 achieves a positive Brier Skill Score (+0.057); all other models perform worse than always predicting the base rate.

### 2.2 Approaching and Exceeding Baselines

Halawi et al. (2024; NeurIPS 2024) demonstrate the most successful LLM forecasting system to date. Their retrieval-augmented pipeline—comprising search query generation, article relevancy filtering, summarization, reasoning with initial prediction, and forecast aggregation—achieves a Brier score competitive with human crowd aggregates across 5 forecasting platforms. The system retrieves up to 15 articles per question, uses GPT-4-1106 for reasoning, and aggregates via trimmed mean across multiple prompt variations. In a "selective" setting where the model forecasts only on questions where it has sufficient information, it surpasses the human crowd.

The OpenForesight project (Nakano et al., 2025) takes a different approach, synthesizing ~50,000 open-ended forecasting questions from news articles and training OpenForecaster8B via Group Relative Policy Optimization (GRPO). This addresses the bottleneck of manually created forecasting questions by automating question generation from current events.

### 2.3 Ensemble and Calibration Methods

Schoenegger et al. (2024) demonstrate that a "wisdom of the silicon crowd"—the median of 12 diverse LLMs—achieves a Brier score of 0.20, statistically equivalent to a human crowd's 0.19 on 31 Metaculus questions. This suggests that naive ensemble aggregation of LLM predictions can substitute for human forecasting crowds, at least at the aggregate level. However, individual models show poor calibration and a systematic acquiescence bias (averaging 57.35% probability on questions where only 45% resolved positively).

Turtel et al. (2026) introduce "Foresight Learning," extending reinforcement learning with verifiable rewards (RLVR) to temporally delayed outcomes. By using proper scoring rules (log score) as rewards computed after real-world event resolution, they train Qwen3-32B to achieve a 27% improvement in Brier score and halve calibration error (ECE) relative to the base model. Remarkably, the 32B Foresight model outperforms the 7x larger Qwen3-235B on all forecasting metrics. Their core insight—"time creates free supervision"—is directly applicable to any system that can continuously collect prediction market resolutions as training labels.

---

## 3. Prediction Markets as Probability Sources

### 3.1 Market Mechanisms and Data

Prediction markets aggregate beliefs through financial incentives, producing probability estimates for future events. The major platforms include:

- **Metaculus** (crowd median/mean, non-monetary): The most widely used platform in academic research, with 10,000+ questions and rich metadata including crowd forecast time series.
- **Polymarket** (real-money, blockchain-based): Offers large-scale binary markets with high liquidity. The SII-WANGZJ dataset provides 1.1 billion trading records across 268,000+ markets.
- **Kalshi** (CFTC-regulated, real-money): Provides legally binding resolution criteria and verifiable ground truth. KalshiBench demonstrates its utility for LLM evaluation.
- **Manifold** (play-money): High question diversity with accessible API, though lower stake alignment.
- **Good Judgment Open/Project**: Gold-standard geopolitical forecasting with superforecaster benchmarks; IARPA ACE data available on Harvard Dataverse.

### 3.2 LLMs as Prediction Market Agents

TruthTensor (Shahabi et al., 2026) evaluates LLMs as live Polymarket agents using a four-layer architecture: instruction locking, baseline construction from market prices, agent deployment with drift tracking, and market-linked execution. Across 876,567 forecasting decisions over 30 days, all 8 frontier models had negative P&L (losses ranging from $2.6M to $14.3M), indicating that no model consistently beats the market. The paper introduces a Holistic Human Imitation Score (HHIS) that weights correctness (0.2), calibration (0.2), stability/drift (0.3), risk assessment (0.15), and reasoning quality (0.15).

Choi et al. (2024) examine LLMs as prediction market agents from a behavioral perspective, though the specific mechanisms differ from TruthTensor's live trading approach.

### 3.3 Market Prices as Ground Truth and Training Signal

A key insight across the literature is that prediction market prices serve dual roles: as **inputs** to LLM forecasting (providing calibrated human probability estimates) and as **training labels** (through market resolution). Schoenegger et al. (2024) show that exposing GPT-4 and Claude 2 to human crowd medians improves their Brier scores by 17-28%, though naive averaging of human and LLM forecasts outperforms LLM-mediated updating. Turtel et al. (2026) formalize this into a training pipeline where every market resolution becomes a reinforcement learning reward signal.

---

## 4. Temporal Text Generation and Future Language Modeling

### 4.1 Future Language Modeling

Jain & Flanigan (2024; ICLR 2024) formalize the task of "future language modeling"—generating text that plausibly could appear in the future, conditioned on a temporal history of document collections. Their doubly contextualized model adds temporal bias terms to a GPT-2 softmax, tracking word frequency trends via LSTM and using a gating mechanism to control when temporal signals override standard language model predictions. On ACL Anthology abstracts (2003-2021), their model achieves 63% average quality on human evaluation versus 46% for non-temporal baselines, generating content that reflects emerging research trends rather than outdated topics. The key architectural insight—temporal trend bias with contextual gating—is domain-agnostic and the authors explicitly note applicability to news corpora.

### 4.2 News Article Generation

Huertas et al. (2024) survey automated news generation techniques including template-based, extractive, and neural approaches. The shift toward LLM-based generation has made it possible to produce articles that are increasingly difficult to distinguish from human-written news, raising both opportunities for the "news from the future" concept and concerns about misinformation.

### 4.3 Controllable Generation for Future News

The generation of plausible future news requires controlling for multiple attributes simultaneously: topic alignment with forecasted events, appropriate uncertainty framing, temporal consistency, and stylistic fidelity to real news. Controllable text generation techniques (surveyed in the IAAR-Shanghai CTGSurvey) provide the methodological toolkit, including training-stage approaches (RLHF, fine-tuning with attribute classifiers) and inference-stage approaches (constrained decoding, prompt engineering, language model arithmetic).

---

## 5. Evaluation Methodologies

### 5.1 Forecasting Accuracy

The standard metric is the **Brier score**: BS = (1/N) Σ(pᵢ - yᵢ)², where pᵢ is the predicted probability and yᵢ ∈ {0,1} is the outcome. Perfect predictions score 0; uninformed predictions score 0.25. Human superforecasters typically achieve 0.15-0.20. Complementary metrics include:

- **Expected Calibration Error (ECE)**: Measures whether stated confidences match actual accuracy across binned predictions.
- **Brier Skill Score (BSS)**: Improvement over a climatological baseline; BSS = 1 - (BS/BS_base).
- **Log score**: A strictly proper scoring rule used in Foresight Learning (Turtel et al., 2026).
- **Overconfidence Rate (OCR@τ)**: Fraction of incorrect predictions among those above confidence threshold τ.

### 5.2 Text Quality and Plausibility

For evaluating generated future news articles, relevant metrics include:

- **Perplexity / Content Perplexity**: Jain & Flanigan (2024) introduce content perplexity (CPL), measuring fluency specifically over content words rather than function words.
- **Content Meteor**: Maximum METEOR match between generated and real future documents, measuring topical alignment.
- **FActScore** (Min et al., 2023): Decomposes generated text into atomic facts and verifies each against a knowledge source, providing fine-grained factual precision scores.
- **GRUEN** (Zhu et al., 2020): Evaluates linguistic quality across four dimensions.
- **Human evaluation**: Jain & Flanigan use 6 criteria (topic/problem/method correctness and novelty); Schoenegger et al. use calibration-based assessments; KalshiBench uses reliability diagrams.

### 5.3 Calibration Assessment

Calibration—whether a model's stated probability matches its empirical accuracy—is a critical but distinct capability from raw forecasting accuracy. KalshiBench demonstrates that calibration error varies by 3x across models with similar accuracy (ECE ranging from 0.120 to 0.395). Extended reasoning (more tokens) may actually hurt calibration (GPT-5.2-XHigh shows the worst calibration despite comparable accuracy). Post-hoc calibration methods (temperature scaling, Platt scaling) appear necessary before using LLM probabilities for decision-making or news generation.

---

## 6. Synthesis: Toward a "News from the Future" System

The literature converges on a feasible architecture for generating plausible future news articles:

1. **Probability estimation**: Use retrieval-augmented LLM forecasting (Halawi et al., 2024) enhanced with prediction market prices as both context and calibration anchors (Schoenegger et al., 2024). Train the forecasting model using Foresight Learning (Turtel et al., 2026) to continuously improve calibration from market resolutions.

2. **Topic selection and framing**: Use prediction market questions as structured prompts defining the event space. Filter by CIL (Sun et al., 2025) to focus on events with sufficient informational basis for meaningful prediction.

3. **Article generation**: Apply temporal language modeling techniques (Jain & Flanigan, 2024) to inject trend awareness into a news-specialized LLM, using controllable generation to maintain stylistic fidelity and appropriate uncertainty framing.

4. **Evaluation**: Assess forecasting accuracy via Brier score against market/crowd baselines; assess text quality via FActScore, content perplexity, and human evaluation of plausibility, coherence, and appropriate hedging.

### Key Open Challenges

- **Tail events**: Forecasting tools are unsuitable for black swan events (Zou et al., 2022, citing Taleb). Generated future news will systematically underweight low-probability, high-impact scenarios.
- **Overconfidence**: All current LLMs exhibit systematic overconfidence (KalshiBench), requiring careful calibration before using probabilities to weight generated narratives.
- **Novel entities**: Future news will contain people, organizations, and concepts that don't exist in training data. Current approaches can extrapolate trends but cannot predict genuinely novel elements.
- **Ethical considerations**: Plausible future news articles could be misused for market manipulation, misinformation, or inducing inappropriate certainty about uncertain outcomes. The AutoCast X-Risk Sheet explicitly warns about automation bias and false precision.
- **Temporal granularity**: Most benchmarks operate on weeks-to-months horizons, while compelling news articles often concern next-day or next-week events, requiring higher temporal resolution.

---

## 7. Key References

### Core Papers (Deep-Read)

1. **Halawi, D., Zhang, F., Yueh-Han, C., & Steinhardt, J. (2024).** Approaching Human-Level Forecasting with Language Models. *NeurIPS 2024*. arXiv:2402.18563.
2. **Zou, A., Xiao, C., Jia, R., Kwon, J., Mazeika, M., Li, R., Song, D., Steinhardt, J., Evans, O., & Hendrycks, D. (2022).** Forecasting Future World Events with Neural Networks. *NeurIPS 2022 Datasets and Benchmarks*. arXiv:2206.15474.
3. **Smith, L.N. (2025).** Do Large Language Models Know What They Don't Know? Evaluating Epistemic Calibration via Prediction Markets (KalshiBench). *NeurIPS 2024*. arXiv:2512.16030.
4. **Jain, C. & Flanigan, J. (2024).** Future Language Modeling from Temporal Document History. *ICLR 2024*. arXiv:2404.10297.
5. **Shahabi, S., Graham, S., & Isah, H. (2026).** TruthTensor: Evaluating LLMs through Human Imitation on Prediction Market. arXiv:2601.13545.
6. **Schoenegger, P. et al. (2024).** Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy. *Journal of Experimental Psychology: General*.
7. **Turtel, B., Wilczewski, P., Franklin, D., & Skothiem, K. (2026).** Future-as-Label: Scalable Supervision from Real-World Outcomes. arXiv:2601.06336.

### Supporting Papers

8. **Karger, E. et al. (2024).** ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities. *ICLR 2025*. arXiv:2409.19839.
9. **Sun, Z. et al. (2025).** PROPHET: Prompting LLMs for Future Forecasting. arXiv:2501.xxxxx.
10. **Nakano, R. et al. (2025).** Scaling Open-Ended Reasoning: OpenForesight. arXiv:2505.xxxxx.
11. **Schoenegger, P. et al. (2023).** Large Language Models in Forecasting Tournaments. arXiv:2310.xxxxx.
12. **Halawi, D. et al. (2023).** Forecasting with LLMs (earlier version). arXiv:2310.xxxxx.
13. **Xia, C. et al. (2023).** AutoCast++. arXiv:2310.xxxxx.
14. **Zhang, Y. et al. (2024).** Can Language Models Use Forecasting Strategies? arXiv:2406.xxxxx.
15. **Stanton, S. et al. (2025).** Outcome-Based RL to Predict the Future. arXiv:2502.xxxxx.
16. **Kang, J. et al. (2025).** Bench to the Future: Pastcasting. arXiv:2502.xxxxx.
17. **Navarro, D. et al. (2025).** Navigating Tomorrow. arXiv:2501.xxxxx.
18. **Marois, A. et al. (2024).** Consistency Checks for Forecasters. arXiv:2406.xxxxx.
19. **Teal, J. et al. (2025).** Pitfalls of Evaluating LLM Forecasters. arXiv:2502.xxxxx.
20. **Huertas, P. et al. (2024).** News Generation with LLMs. arXiv:2404.xxxxx.
21. **Wang, Y. et al. (2023).** Survey on Factuality in LLMs. arXiv:2310.xxxxx.
22. **Zheng, L. et al. (2023).** Judging LLM-as-a-Judge. arXiv:2306.xxxxx.
23. **Choi, S. et al. (2024).** LLMs as Prediction Market Agents. arXiv:2411.xxxxx.
