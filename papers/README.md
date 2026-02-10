# Papers

Downloaded academic papers relevant to the "News from the Future" research topic.

## Paper Inventory

### Core Papers (Deep-Read)

| File | Authors | Year | Title | Venue | Key Contribution |
|------|---------|------|-------|-------|-----------------|
| `halawi2024_approaching_human_level_forecasting.pdf` | Halawi, Zhang, Yueh-Han, Steinhardt | 2024 | Approaching Human-Level Forecasting with Language Models | NeurIPS 2024 | RAG-based forecasting system achieving human-competitive Brier scores across 5 platforms |
| `zou2022_autocast_forecasting_future_world_events.pdf` | Zou et al. | 2022 | Forecasting Future World Events with Neural Networks | NeurIPS 2022 D&B | AutoCast benchmark: 6,707 questions + 200GB news corpus |
| `smith2025_kalshibench_prediction_markets.pdf` | Smith (Nel) | 2025 | Evaluating Epistemic Calibration via Prediction Markets (KalshiBench) | NeurIPS 2024 | 1,531 Kalshi questions showing systematic LLM overconfidence |
| `jain2024_future_language_modeling.pdf` | Jain, Flanigan | 2024 | Future Language Modeling from Temporal Document History | ICLR 2024 | Temporal bias + gating mechanism for generating plausible future text |
| `chen2026_truthtensor_prediction_market.pdf` | Shahabi, Graham, Isah | 2026 | TruthTensor: Evaluating LLMs through Human Imitation on Prediction Market | arXiv | 4-layer eval framework for LLMs as live Polymarket agents |
| `schoenegger2024_wisdom_silicon_crowd.pdf` | Schoenegger et al. | 2024 | Wisdom of the Silicon Crowd | JEP:General | LLM ensemble (12 models) matches human crowd accuracy |
| `wu2026_future_as_label.pdf` | Turtel, Wilczewski, Franklin, Skothiem | 2026 | Future-as-Label: Scalable Supervision from Real-World Outcomes | arXiv | Foresight Learning: RL with proper scoring rules on resolved events |

### Benchmarks and Evaluation

| File | Authors | Year | Title | Venue |
|------|---------|------|-------|-------|
| `karger2024_forecastbench.pdf` | Karger et al. | 2024 | ForecastBench: Dynamic Benchmark of AI Forecasting | ICLR 2025 |
| `sun2025_prophet_future_forecasting.pdf` | Sun et al. | 2025 | PROPHET: Prompting LLMs for Future Forecasting | arXiv |
| `nakano2025_scaling_open_ended_reasoning.pdf` | Nakano et al. | 2025 | Scaling Open-Ended Reasoning (OpenForesight) | arXiv |
| `teal2025_pitfalls_evaluating_forecasters.pdf` | Teal et al. | 2025 | Pitfalls of Evaluating LLM Forecasters | arXiv |
| `marois2024_consistency_checks_forecasters.pdf` | Marois et al. | 2024 | Consistency Checks for Forecasters | arXiv |
| `kang2025_bench_to_the_future_pastcasting.pdf` | Kang et al. | 2025 | Bench to the Future: Pastcasting | arXiv |

### Forecasting Methods

| File | Authors | Year | Title | Venue |
|------|---------|------|-------|-------|
| `halawi2023_forecasting_with_llms.pdf` | Halawi et al. | 2023 | Forecasting with LLMs (earlier version) | arXiv |
| `schoenegger2023_llm_prediction_tournament.pdf` | Schoenegger et al. | 2023 | LLMs in Forecasting Tournaments | arXiv |
| `xia2023_autocast_plus_plus.pdf` | Xia et al. | 2023 | AutoCast++ | arXiv |
| `zhang2024_can_lm_use_forecasting_strategies.pdf` | Zhang et al. | 2024 | Can Language Models Use Forecasting Strategies? | arXiv |
| `stanton2025_outcome_rl_predict_future.pdf` | Stanton et al. | 2025 | Outcome-Based RL to Predict the Future | arXiv |
| `navarro2025_navigating_tomorrow.pdf` | Navarro et al. | 2025 | Navigating Tomorrow | arXiv |

### Prediction Markets and LLM Agents

| File | Authors | Year | Title | Venue |
|------|---------|------|-------|-------|
| `choi2024_llm_prediction_market_agents.pdf` | Choi et al. | 2024 | LLMs as Prediction Market Agents | arXiv |

### News Generation and Factuality

| File | Authors | Year | Title | Venue |
|------|---------|------|-------|-------|
| `huertas2024_news_generation.pdf` | Huertas et al. | 2024 | News Generation with LLMs | arXiv |
| `wang2023_survey_factuality_llm.pdf` | Wang et al. | 2023 | Survey on Factuality in LLMs | arXiv |
| `zheng2023_judging_llm_as_judge.pdf` | Zheng et al. | 2023 | Judging LLM-as-a-Judge | arXiv |

## Notes

- **choi2024_llm_prediction_market_agents.pdf**: This file was found to contain a different paper ("Document Haystacks" by Jun Chen et al.) during deep reading. The correct paper may need to be re-downloaded.
- **pages/**: Contains chunked versions of papers (3 pages per chunk) used for deep reading by analysis agents.
- Total: 23 papers downloaded, 7 deep-read with detailed summaries.

## How to Re-Download

Papers were downloaded from arXiv using their API. Example:
```bash
curl -L "https://arxiv.org/pdf/2402.18563" -o halawi2024_approaching_human_level_forecasting.pdf
```
