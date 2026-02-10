# Research Plan: News from the Future

## Motivation & Novelty Assessment

### Why This Research Matters
People want to understand what the future holds. Prediction markets aggregate human beliefs into calibrated probabilities, and LLMs can generate fluent text — but no one has systematically studied the pipeline of combining these two capabilities to produce plausible "future news articles." Such a system could serve as a decision-support tool, scenario planning aid, or public information service that makes probabilistic forecasts tangible and accessible to non-experts.

### Gap in Existing Work
The literature reveals two disjoint research threads: (1) LLM forecasting accuracy (Halawi et al., Schoenegger et al., KalshiBench) and (2) temporal/future text generation (Jain & Flanigan). No published work has connected them into an end-to-end pipeline that takes prediction market questions, generates probability-informed forecasts, and produces news-style articles about likely future outcomes. Additionally, while text quality of LLM-generated news is well-studied, the **conditional plausibility** — whether generated future news properly reflects prediction market probabilities — has not been evaluated.

### Our Novel Contribution
We build and evaluate an end-to-end "News from the Future" prototype that:
1. Fetches real prediction market questions (from Metaculus and Polymarket APIs)
2. Uses GPT-4.1 to generate probability estimates (with and without market price anchoring)
3. Generates news-style articles conditioned on predicted outcomes
4. Evaluates the system on three dimensions: **forecasting accuracy** (Brier score vs. market/crowd), **article quality** (coherence, style, factual grounding via LLM-as-Judge), and **probability-article alignment** (whether articles appropriately convey uncertainty)

### Experiment Justification
- **Experiment 1 (Forecasting):** We must first validate that LLM forecasts are reasonable before using them to generate news. We test GPT-4.1 on the Halawi forecasting dataset (914 test questions) and KalshiBench (1,531 questions), comparing against crowd/market baselines.
- **Experiment 2 (Article Generation):** Given forecasted outcomes, we generate future news articles and evaluate their quality. This tests whether LLMs can produce plausible, well-calibrated future narratives.
- **Experiment 3 (End-to-End Pipeline):** We fetch LIVE prediction market questions, run the full pipeline, and evaluate the complete system with LLM-as-Judge evaluation. This is the core contribution — demonstrating a working prototype.

---

## Research Question
Can a system that combines LLM forecasting with prediction market data generate plausible, well-calibrated news articles about future events? Specifically:
1. How accurate are LLM forecasts when anchored to prediction market probabilities?
2. What is the quality of generated "future news" articles?
3. Does probability anchoring improve article plausibility and calibration?

## Hypothesis Decomposition
- **H1:** LLM forecasts anchored to prediction market prices achieve better Brier scores than unanchored LLM forecasts.
- **H2:** Generated future news articles are rated as plausible and well-written by LLM-as-Judge evaluation (>3.5/5 on quality dimensions).
- **H3:** Articles generated with probability anchoring better convey appropriate uncertainty than those without.

## Proposed Methodology

### Approach
Three-stage pipeline evaluated on existing benchmarks and live prediction market data:
1. **Forecasting Module**: Prompt GPT-4.1 with prediction market questions, with/without market price context
2. **Article Generation Module**: Given a question and forecasted probability, generate a news article as if the event has occurred (for high-prob events) or discussing the outlook (for uncertain events)
3. **Evaluation**: Brier score for forecasting, LLM-as-Judge for article quality

### Experimental Steps

#### Experiment 1: Forecasting Accuracy
1. Load Halawi forecasting test set (914 questions) and KalshiBench (1,531 questions)
2. For each question, prompt GPT-4.1 in two conditions:
   - **Baseline (no anchor):** Question + background only
   - **Market-anchored:** Question + background + community/market probability
3. Collect predicted probabilities, compute Brier scores
4. Compare against crowd/market baselines

#### Experiment 2: Article Generation Quality
1. Select 50 diverse questions from the datasets (stratified by category/probability range)
2. Generate future news articles in two conditions:
   - **Unanchored:** Generate based on LLM's own forecast
   - **Anchored:** Generate based on market probability
3. Evaluate via GPT-4.1-as-Judge on 5 dimensions: plausibility, coherence, informativeness, appropriate hedging, news style

#### Experiment 3: Live Pipeline Demo
1. Fetch 20 current prediction market questions from Metaculus/Polymarket APIs
2. Run full pipeline: forecast → article generation
3. Evaluate with LLM-as-Judge
4. Create a simple web-ready output (HTML/JSON)

### Baselines
- **Forecasting:** Community prediction median (Halawi dataset), market probability (KalshiBench), naive base rate (0.5)
- **Article quality:** Direct generation without any forecasting context (just "write news about X")

### Evaluation Metrics
1. **Brier Score** = mean((p - y)²) — lower is better
2. **Expected Calibration Error (ECE)** — binned calibration
3. **LLM-as-Judge scores** (1-5 scale) for: Plausibility, Coherence, Informativeness, Uncertainty Handling, News Style
4. **Probability-Article Alignment** — whether the article's tone matches the probability (confident for p>0.8, hedged for 0.3<p<0.7)

### Statistical Analysis Plan
- Paired t-tests or Wilcoxon signed-rank tests for Brier score comparisons (anchored vs unanchored)
- Bootstrap confidence intervals for all metrics
- Effect sizes (Cohen's d)
- Significance level: α = 0.05

## Expected Outcomes
- H1: Market anchoring should improve Brier scores by 15-25% (based on Schoenegger et al. finding of 17-28% improvement)
- H2: Article quality should exceed 3.5/5 on most dimensions
- H3: Anchored articles should show significantly better uncertainty handling

## Timeline and Milestones
1. Environment setup + data loading: 10 min
2. Forecasting experiment implementation: 30 min
3. Run forecasting experiment (API calls): 30 min
4. Article generation implementation: 20 min
5. Run article generation: 20 min
6. Live pipeline demo: 20 min
7. Analysis and visualization: 30 min
8. Documentation: 30 min

## Potential Challenges
- **API rate limits:** Batch requests, cache responses, use subset if needed
- **Cost:** ~914 + 1531 + 50*2 + 20 ≈ 2,500 API calls. At ~$0.01/call ≈ $25 total
- **Live API availability:** Metaculus/Polymarket APIs may have rate limits; cache responses
- **Evaluation subjectivity:** LLM-as-Judge has known biases; report inter-rater agreement

## Success Criteria
1. Complete end-to-end pipeline producing future news articles
2. Statistically significant improvement from market anchoring on forecasting
3. Article quality rated >3.5/5 on average across dimensions
4. Reproducible results with documented code
