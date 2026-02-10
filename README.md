# News from the Future

**Can LLMs + Prediction Markets generate plausible news articles about future events?**

## Key Findings

- **Market-anchored GPT-4.1 achieves Brier score 0.060** — a 29.2% improvement over unanchored forecasting (p=0.041) and nearly matching the human crowd (0.058)
- **Generated articles score 4.3-5.0/5.0** across plausibility, coherence, informativeness, uncertainty handling, and news style
- **Probability anchoring significantly improves uncertainty handling** in generated articles (+0.52, p<0.001, large effect)
- **Live pipeline works end-to-end** with real Metaculus and Polymarket data, producing a complete "Future Times" website

## How It Works

```
Prediction Markets  →  Market Probability (calibrated)
        +                      ↓
    LLM (GPT-4.1)  →  Combined Forecast (60% market + 40% LLM)
                               ↓
                    News Article Generation
                    (tone matched to probability)
                               ↓
                    "The Future Times" Website
```

## Quick Start

```bash
# Set up environment
source .venv/bin/activate

# Run all experiments
python src/run_all.py

# Or run individually:
python src/experiment1_forecasting.py   # Forecasting accuracy
python src/experiment2_article_generation.py  # Article quality
python src/experiment3_live_pipeline.py  # Live pipeline demo
python src/analysis.py                  # Generate plots
```

Requires: `OPENAI_API_KEY` environment variable set.

## Project Structure

```
src/                          # Experiment code
  experiment1_forecasting.py  # Brier score evaluation on Halawi + KalshiBench
  experiment2_article_generation.py  # Article generation + LLM-as-Judge evaluation
  experiment3_live_pipeline.py       # Live Metaculus/Polymarket pipeline
  analysis.py                 # Statistical analysis and plotting
results/
  plots/                      # All visualizations
  live_pipeline/
    future_times.html         # Generated news website
  forecasts/                  # Forecasting experiment data
  articles/                   # Article generation data
datasets/                     # Halawi, KalshiBench, AutoCast
papers/                       # 23 research papers
```

## Full Report

See [REPORT.md](REPORT.md) for comprehensive methodology, results, and analysis.

## Dependencies

- Python 3.12+
- openai, numpy, scipy, matplotlib, seaborn, requests, pandas, tqdm
- See `pyproject.toml` for full list
