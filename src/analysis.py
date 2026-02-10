"""
Analysis and Visualization for all experiments.
Generates plots and statistical analyses for the REPORT.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE_DIR = "/workspaces/news-from-future-ai-aed4-claude"
RESULTS_DIR = f"{BASE_DIR}/results"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150


def analyze_experiment1():
    """Analyze forecasting results."""
    print("=" * 60)
    print("Analyzing Experiment 1: Forecasting Accuracy")
    print("=" * 60)

    results = {}
    for dataset in ["halawi", "kalshi"]:
        path = f"{RESULTS_DIR}/forecasts/{dataset}_results.json"
        if os.path.exists(path):
            with open(path) as f:
                results[dataset] = json.load(f)

    if not results:
        print("  No results found. Skipping.")
        return {}

    analysis = {}
    for dataset, data in results.items():
        forecasts = data["forecasts"]
        n = len(forecasts)

        preds_base = np.array([f["pred_baseline"] for f in forecasts])
        preds_anchor = np.array([f["pred_anchored"] for f in forecasts])
        outcomes = np.array([f["resolution"] for f in forecasts])
        community = np.array([f["community_prediction"] for f in forecasts
                              if f["community_prediction"] is not None])
        outcomes_comm = np.array([f["resolution"] for f in forecasts
                                  if f["community_prediction"] is not None])

        # Brier scores
        bs_base = np.mean((preds_base - outcomes) ** 2)
        bs_anchor = np.mean((preds_anchor - outcomes) ** 2)
        bs_naive = 0.25  # always predict 0.5
        bs_community = np.mean((community - outcomes_comm) ** 2) if len(community) > 0 else None

        # Bootstrap confidence intervals
        def bootstrap_brier(preds, outs, n_boot=1000):
            scores = []
            rng = np.random.RandomState(42)
            for _ in range(n_boot):
                idx = rng.choice(len(preds), len(preds), replace=True)
                scores.append(np.mean((preds[idx] - outs[idx]) ** 2))
            return np.percentile(scores, [2.5, 97.5])

        ci_base = bootstrap_brier(preds_base, outcomes)
        ci_anchor = bootstrap_brier(preds_anchor, outcomes)

        # Paired test: baseline vs anchored
        diffs = (preds_base - outcomes) ** 2 - (preds_anchor - outcomes) ** 2
        t_stat, p_value = stats.ttest_rel(
            (preds_base - outcomes) ** 2,
            (preds_anchor - outcomes) ** 2
        )
        effect_size = np.mean(diffs) / (np.std(diffs) + 1e-10)

        # Calibration
        def compute_calibration_curve(preds, outs, n_bins=10):
            bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = []
            bin_accuracies = []
            bin_counts = []
            for i in range(n_bins):
                mask = (preds >= bins[i]) & (preds < bins[i + 1])
                if mask.sum() > 0:
                    bin_centers.append(preds[mask].mean())
                    bin_accuracies.append(outs[mask].mean())
                    bin_counts.append(mask.sum())
            return bin_centers, bin_accuracies, bin_counts

        cal_base = compute_calibration_curve(preds_base, outcomes)
        cal_anchor = compute_calibration_curve(preds_anchor, outcomes)

        ece_base = data["metrics"].get("ece_baseline", 0)
        ece_anchor = data["metrics"].get("ece_anchored", 0)

        analysis[dataset] = {
            "n": n,
            "brier_baseline": float(bs_base),
            "brier_anchored": float(bs_anchor),
            "brier_community": float(bs_community) if bs_community is not None else None,
            "brier_naive": float(bs_naive),
            "ci_baseline": [float(x) for x in ci_base],
            "ci_anchored": [float(x) for x in ci_anchor],
            "improvement_pct": float((bs_base - bs_anchor) / bs_base * 100),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "effect_size_d": float(effect_size),
            "ece_baseline": float(ece_base),
            "ece_anchored": float(ece_anchor),
            "calibration_baseline": cal_base,
            "calibration_anchored": cal_anchor,
        }

        print(f"\n{dataset.upper()} Results (n={n}):")
        print(f"  Brier (baseline):  {bs_base:.4f} [{ci_base[0]:.4f}, {ci_base[1]:.4f}]")
        print(f"  Brier (anchored):  {bs_anchor:.4f} [{ci_anchor[0]:.4f}, {ci_anchor[1]:.4f}]")
        if bs_community is not None:
            print(f"  Brier (community): {bs_community:.4f}")
        print(f"  Brier (naive 0.5): {bs_naive:.4f}")
        print(f"  Improvement:       {(bs_base - bs_anchor) / bs_base * 100:.1f}%")
        print(f"  t-test:            t={t_stat:.3f}, p={p_value:.4f}")
        print(f"  Effect size (d):   {effect_size:.3f}")
        print(f"  ECE (baseline):    {ece_base:.4f}")
        print(f"  ECE (anchored):    {ece_anchor:.4f}")

    # --- PLOTS ---

    # Plot 1: Brier Score Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (dataset, a) in enumerate(analysis.items()):
        ax = axes[idx]
        methods = ["Naive (0.5)", "GPT-4.1\n(no anchor)", "GPT-4.1\n(anchored)"]
        scores = [a["brier_naive"], a["brier_baseline"], a["brier_anchored"]]
        colors = ["#95a5a6", "#e74c3c", "#27ae60"]

        if a["brier_community"] is not None:
            methods.insert(2, "Community/\nMarket")
            scores.insert(2, a["brier_community"])
            colors.insert(2, "#3498db")

        bars = ax.bar(methods, scores, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Brier Score (lower is better)")
        ax.set_title(f"{dataset.capitalize()} Dataset (n={a['n']})")
        ax.set_ylim(0, 0.35)

        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{score:.3f}", ha="center", va="bottom", fontsize=10)

        # Significance annotation
        if a["p_value"] < 0.05:
            sig = "***" if a["p_value"] < 0.001 else "**" if a["p_value"] < 0.01 else "*"
            ax.annotate(sig, xy=(0.7, max(scores) + 0.02), fontsize=14, ha="center", color="red")

    plt.suptitle("Experiment 1: Forecasting Accuracy — Brier Score Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/exp1_brier_comparison.png", bbox_inches="tight")
    plt.close()

    # Plot 2: Calibration Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (dataset, a) in enumerate(analysis.items()):
        ax = axes[idx]
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

        cal = a["calibration_baseline"]
        ax.plot(cal[0], cal[1], "o-", color="#e74c3c", label=f"Baseline (ECE={a['ece_baseline']:.3f})")

        cal = a["calibration_anchored"]
        ax.plot(cal[0], cal[1], "s-", color="#27ae60", label=f"Anchored (ECE={a['ece_anchored']:.3f})")

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title(f"{dataset.capitalize()} Calibration")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle("Experiment 1: Calibration Curves", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/exp1_calibration.png", bbox_inches="tight")
    plt.close()

    # Plot 3: Prediction distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (dataset, data) in enumerate(results.items()):
        ax = axes[idx]
        forecasts = data["forecasts"]
        preds_b = [f["pred_baseline"] for f in forecasts]
        preds_a = [f["pred_anchored"] for f in forecasts]

        ax.hist(preds_b, bins=20, alpha=0.5, color="#e74c3c", label="Baseline", density=True)
        ax.hist(preds_a, bins=20, alpha=0.5, color="#27ae60", label="Anchored", density=True)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title(f"{dataset.capitalize()} Prediction Distribution")
        ax.legend()

    plt.suptitle("Experiment 1: Distribution of Predictions", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/exp1_prediction_distribution.png", bbox_inches="tight")
    plt.close()

    with open(f"{RESULTS_DIR}/experiment1_analysis.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert(o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError
        json.dump(analysis, f, indent=2, default=convert)

    return analysis


def analyze_experiment2():
    """Analyze article generation results."""
    print("\n" + "=" * 60)
    print("Analyzing Experiment 2: Article Quality")
    print("=" * 60)

    path = f"{RESULTS_DIR}/articles/experiment2_results.json"
    if not os.path.exists(path):
        print("  No results found. Skipping.")
        return {}

    with open(path) as f:
        data = json.load(f)

    articles = data["articles"]
    dims = ["plausibility", "coherence", "informativeness", "uncertainty_handling", "news_style"]

    anchored_scores = {d: [] for d in dims}
    baseline_scores = {d: [] for d in dims}

    for a in articles:
        if a.get("eval_anchored"):
            for d in dims:
                anchored_scores[d].append(a["eval_anchored"].get(d, 3))
        if a.get("eval_baseline"):
            for d in dims:
                baseline_scores[d].append(a["eval_baseline"].get(d, 3))

    analysis = {"dimensions": {}}
    for d in dims:
        anc = np.array(anchored_scores[d])
        bas = np.array(baseline_scores[d])
        if len(anc) > 0 and len(bas) > 0:
            t_stat, p_val = stats.ttest_rel(anc[:min(len(anc), len(bas))],
                                             bas[:min(len(anc), len(bas))])
            analysis["dimensions"][d] = {
                "anchored_mean": float(np.mean(anc)),
                "anchored_std": float(np.std(anc)),
                "baseline_mean": float(np.mean(bas)),
                "baseline_std": float(np.std(bas)),
                "diff": float(np.mean(anc) - np.mean(bas)),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "effect_size": float((np.mean(anc) - np.mean(bas)) / (np.std(np.concatenate([anc, bas])) + 1e-10)),
            }

    # Print results
    print("\nArticle Quality Comparison (Anchored vs Baseline):")
    print(f"{'Dimension':<25} {'Anchored':>10} {'Baseline':>10} {'Diff':>8} {'p-value':>10}")
    print("-" * 65)
    for d in dims:
        if d in analysis["dimensions"]:
            a = analysis["dimensions"][d]
            sig = "*" if a["p_value"] < 0.05 else ""
            print(f"  {d:<23} {a['anchored_mean']:>8.2f}   {a['baseline_mean']:>8.2f}   "
                  f"{a['diff']:>+6.2f}   {a['p_value']:>8.4f}{sig}")

    # Plot: Article quality comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dims))
    width = 0.35

    anc_means = [analysis["dimensions"][d]["anchored_mean"] for d in dims if d in analysis["dimensions"]]
    anc_stds = [analysis["dimensions"][d]["anchored_std"] for d in dims if d in analysis["dimensions"]]
    bas_means = [analysis["dimensions"][d]["baseline_mean"] for d in dims if d in analysis["dimensions"]]
    bas_stds = [analysis["dimensions"][d]["baseline_std"] for d in dims if d in analysis["dimensions"]]
    valid_dims = [d for d in dims if d in analysis["dimensions"]]

    bars1 = ax.bar(x[:len(valid_dims)] - width / 2, anc_means, width, yerr=anc_stds,
                   label="Anchored", color="#27ae60", capsize=3, alpha=0.85)
    bars2 = ax.bar(x[:len(valid_dims)] + width / 2, bas_means, width, yerr=bas_stds,
                   label="Baseline", color="#e74c3c", capsize=3, alpha=0.85)

    ax.set_ylabel("Score (1-5)")
    ax.set_title("Experiment 2: Article Quality — Anchored vs Baseline")
    ax.set_xticks(x[:len(valid_dims)])
    ax.set_xticklabels([d.replace("_", "\n") for d in valid_dims], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3.5, color="gray", linestyle="--", alpha=0.5, label="Target (3.5)")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # Add significance markers
    for i, d in enumerate(valid_dims):
        if analysis["dimensions"][d]["p_value"] < 0.05:
            max_h = max(anc_means[i], bas_means[i]) + max(anc_stds[i], bas_stds[i])
            ax.text(i, max_h + 0.3, "*", ha="center", fontsize=14, color="red")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/exp2_article_quality.png", bbox_inches="tight")
    plt.close()

    # Plot: Quality by probability range
    prob_ranges = [(0, 0.3, "Low (0-30%)"), (0.3, 0.7, "Medium (30-70%)"), (0.7, 1.01, "High (70-100%)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (lo, hi, label) in enumerate(prob_ranges):
        ax = axes[idx]
        subset = [a for a in articles if lo <= (a.get("probability", 0.5) or 0.5) < hi]
        if not subset:
            ax.set_title(f"{label}\n(no data)")
            continue

        anc_by_dim = []
        for d in dims:
            scores = [a["eval_anchored"][d] for a in subset
                      if a.get("eval_anchored") and d in a["eval_anchored"]]
            anc_by_dim.append(np.mean(scores) if scores else 0)

        ax.bar(range(len(dims)), anc_by_dim, color=["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"])
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([d[:8] for d in dims], rotation=45, fontsize=8)
        ax.set_ylim(0, 5.5)
        ax.set_title(f"{label}\n(n={len(subset)})")
        ax.set_ylabel("Score (1-5)")

    plt.suptitle("Experiment 2: Article Quality by Probability Range (Anchored)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/exp2_quality_by_probability.png", bbox_inches="tight")
    plt.close()

    with open(f"{RESULTS_DIR}/experiment2_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis


def analyze_experiment3():
    """Analyze live pipeline results."""
    print("\n" + "=" * 60)
    print("Analyzing Experiment 3: Live Pipeline")
    print("=" * 60)

    path = f"{RESULTS_DIR}/live_pipeline/live_pipeline_results.json"
    if not os.path.exists(path):
        print("  No results found. Skipping.")
        return {}

    with open(path) as f:
        data = json.load(f)

    # LLM vs Market forecast comparison
    market_probs = [r["market_probability"] for r in data if r.get("market_probability")]
    llm_probs = [r["llm_forecast"] for r in data if r.get("llm_forecast")]
    combined_probs = [r["combined_probability"] for r in data if r.get("combined_probability")]

    diffs = [abs(m - l) for m, l in zip(market_probs, llm_probs)]

    analysis = {
        "n_articles": len(data),
        "n_evaluated": sum(1 for r in data if r.get("evaluation")),
        "mean_market_prob": float(np.mean(market_probs)) if market_probs else 0,
        "mean_llm_prob": float(np.mean(llm_probs)) if llm_probs else 0,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else 0,
        "correlation": float(np.corrcoef(market_probs, llm_probs)[0, 1]) if len(market_probs) > 2 else 0,
    }

    # Quality scores
    evals = [r["evaluation"] for r in data if r.get("evaluation")]
    dims = ["plausibility", "coherence", "informativeness", "uncertainty_handling", "news_style"]
    for d in dims:
        scores = [e[d] for e in evals if d in e]
        if scores:
            analysis[f"{d}_mean"] = float(np.mean(scores))
            analysis[f"{d}_std"] = float(np.std(scores))

    # Print
    print(f"\nLive Pipeline Summary (n={analysis['n_articles']}):")
    print(f"  Mean |LLM - Market| diff: {analysis['mean_abs_diff']:.3f}")
    print(f"  LLM-Market correlation: {analysis['correlation']:.3f}")
    print(f"\n  Article Quality:")
    for d in dims:
        m = analysis.get(f"{d}_mean", 0)
        s = analysis.get(f"{d}_std", 0)
        print(f"    {d}: {m:.2f} +/- {s:.2f}")

    # Plot: LLM vs Market scatter
    if market_probs and llm_probs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.scatter(market_probs, llm_probs, alpha=0.6, color="#3498db", s=60, edgecolors="white")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect agreement")
        ax.set_xlabel("Market Probability")
        ax.set_ylabel("LLM Forecast")
        ax.set_title(f"LLM vs Market Forecasts (r={analysis['correlation']:.2f})")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax = axes[1]
        if evals:
            overall = [np.mean([e[d] for d in dims if d in e]) for e in evals]
            ax.bar(range(len(dims)),
                   [analysis.get(f"{d}_mean", 0) for d in dims],
                   color=["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"],
                   edgecolor="white")
            ax.set_xticks(range(len(dims)))
            ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
            ax.set_ylabel("Score (1-5)")
            ax.set_title("Live Pipeline Article Quality")
            ax.set_ylim(0, 5.5)
            ax.axhline(y=3.5, color="gray", linestyle="--", alpha=0.5)

        plt.suptitle("Experiment 3: Live Pipeline Results", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/exp3_live_pipeline.png", bbox_inches="tight")
        plt.close()

    with open(f"{RESULTS_DIR}/experiment3_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis


def main():
    print("\n" + "=" * 60)
    print("RUNNING ALL ANALYSES")
    print("=" * 60)

    exp1 = analyze_experiment1()
    exp2 = analyze_experiment2()
    exp3 = analyze_experiment3()

    # Save combined analysis
    combined = {
        "experiment1": exp1,
        "experiment2": exp2,
        "experiment3": exp3,
    }
    with open(f"{RESULTS_DIR}/combined_analysis.json", "w") as f:
        def convert(o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError
        json.dump(combined, f, indent=2, default=convert)

    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print(f"Plots saved to: {PLOTS_DIR}/")
    print("=" * 60)

    return combined


if __name__ == "__main__":
    main()
