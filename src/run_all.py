"""
Main runner script for all experiments.
Executes experiments 1-3, then analysis.
"""
import sys
import os
import json
import time
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = "/workspaces/news-from-future-ai-aed4-claude"


def main():
    start = time.time()
    print("=" * 70)
    print("NEWS FROM THE FUTURE: Complete Research Pipeline")
    print(f"Started: {datetime.datetime.now().isoformat()}")
    print("=" * 70)

    # Environment info
    print(f"\nPython: {sys.version}")
    print(f"OpenAI key: {'SET' if os.environ.get('OPENAI_API_KEY') else 'MISSING'}")

    # Experiment 1: Forecasting
    print("\n\n" + "=" * 70)
    print("PHASE 1: Forecasting Accuracy Experiment")
    print("=" * 70)
    from experiment1_forecasting import main as exp1_main
    exp1_results = exp1_main()

    # Experiment 2: Article Generation
    print("\n\n" + "=" * 70)
    print("PHASE 2: Article Generation Experiment")
    print("=" * 70)
    from experiment2_article_generation import main as exp2_main
    exp2_results = exp2_main()

    # Experiment 3: Live Pipeline
    print("\n\n" + "=" * 70)
    print("PHASE 3: Live Pipeline Demo")
    print("=" * 70)
    from experiment3_live_pipeline import main as exp3_main
    exp3_results = exp3_main()

    # Analysis
    print("\n\n" + "=" * 70)
    print("PHASE 4: Analysis & Visualization")
    print("=" * 70)
    from analysis import main as analysis_main
    analysis_results = analysis_main()

    elapsed = time.time() - start
    print(f"\n\nTotal time: {elapsed/60:.1f} minutes")
    print("=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
