"""
Experiment 1: LLM Forecasting Accuracy
Tests GPT-4.1 forecasting with and without market price anchoring
on Halawi forecasting dataset and KalshiBench.
"""
import json
import os
import random
import time
import numpy as np
from openai import OpenAI
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR = "/workspaces/news-from-future-ai-aed4-claude"
RESULTS_DIR = f"{BASE_DIR}/results/forecasts"


def load_halawi_test(max_n=None):
    """Load Halawi forecasting test questions."""
    data = []
    with open(f"{BASE_DIR}/datasets/forecasting_halawi2024/test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("question_type") == "binary" and d.get("resolution") is not None:
                # Get last community prediction as anchor
                cp = json.loads(d.get("community_predictions", "[]"))
                last_community = cp[-1][1] if cp else None
                data.append({
                    "id": d.get("url", ""),
                    "question": d["question"],
                    "background": d.get("background", "")[:1500],
                    "resolution_criteria": d.get("resolution_criteria", "")[:500],
                    "resolution": float(d["resolution"]),
                    "community_prediction": last_community,
                    "source": "halawi",
                })
            if max_n and len(data) >= max_n:
                break
    return data


def load_kalshibench(max_n=None):
    """Load KalshiBench questions."""
    data = []
    with open(f"{BASE_DIR}/datasets/kalshibench/train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            gt = d.get("ground_truth", "")
            if gt in ("yes", "no"):
                resolution = 1.0 if gt == "yes" else 0.0
                data.append({
                    "id": d.get("id", ""),
                    "question": d["question"],
                    "background": d.get("description", "")[:1500],
                    "resolution_criteria": d.get("description", "")[:500],
                    "resolution": resolution,
                    "community_prediction": d.get("market_probability"),
                    "category": d.get("category", ""),
                    "source": "kalshi",
                })
            if max_n and len(data) >= max_n:
                break
    return data


def forecast_question(question_data, use_anchor=False, model="gpt-4.1"):
    """Get LLM probability forecast for a binary question."""
    q = question_data["question"]
    bg = question_data["background"]
    rc = question_data["resolution_criteria"]

    system_prompt = (
        "You are an expert forecaster. Given a question about a future event, "
        "estimate the probability that the event will resolve YES. "
        "Respond with ONLY a number between 0.0 and 1.0 representing your "
        "probability estimate. Do not include any other text."
    )

    user_prompt = f"Question: {q}\n\nBackground: {bg}\n\nResolution Criteria: {rc}"

    if use_anchor:
        if question_data.get("community_prediction") is not None:
            anchor = question_data["community_prediction"]
            user_prompt += (
                f"\n\nNote: The prediction market/crowd currently estimates this "
                f"at {anchor:.2f} probability. Consider this as a reference point, "
                f"but form your own judgment."
            )
        else:
            # No market price available — use chain-of-thought reasoning prompt
            system_prompt = (
                "You are an expert superforecaster trained in probabilistic reasoning. "
                "Use the following strategy: (1) consider base rates, (2) identify key "
                "factors that could push the probability up or down, (3) consider "
                "multiple perspectives, (4) give your final probability estimate. "
                "Respond with ONLY a number between 0.0 and 1.0."
            )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=20,
        )
        text = response.choices[0].message.content.strip()
        # Parse probability
        prob = float(text)
        return max(0.01, min(0.99, prob))
    except Exception as e:
        print(f"  Error: {e}, text='{text if 'text' in dir() else 'N/A'}'")
        return 0.5  # fallback


def brier_score(predictions, outcomes):
    """Compute Brier score."""
    preds = np.array(predictions)
    outs = np.array(outcomes)
    return np.mean((preds - outs) ** 2)


def calibration_error(predictions, outcomes, n_bins=10):
    """Compute Expected Calibration Error."""
    preds = np.array(predictions)
    outs = np.array(outcomes)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = outs[mask].mean()
        bin_conf = preds[mask].mean()
        ece += mask.sum() / len(preds) * abs(bin_acc - bin_conf)
    return ece


def run_forecasting_experiment(dataset_name, data, sample_size=200):
    """Run forecasting experiment on a dataset sample."""
    print(f"\n{'='*60}")
    print(f"Running forecasting on {dataset_name} (n={len(data)}, sample={sample_size})")
    print(f"{'='*60}")

    # Sample if needed
    if len(data) > sample_size:
        random.seed(SEED)
        data = random.sample(data, sample_size)

    results = {
        "dataset": dataset_name,
        "n_questions": len(data),
        "forecasts": [],
    }

    for i, q in enumerate(tqdm(data, desc=f"{dataset_name}")):
        # Baseline forecast (no anchor)
        pred_base = forecast_question(q, use_anchor=False)
        time.sleep(0.1)  # rate limit

        # Anchored forecast (with community/market price)
        pred_anchor = forecast_question(q, use_anchor=True)
        time.sleep(0.1)

        results["forecasts"].append({
            "id": q["id"],
            "question": q["question"][:200],
            "resolution": q["resolution"],
            "community_prediction": q.get("community_prediction"),
            "pred_baseline": pred_base,
            "pred_anchored": pred_anchor,
            "category": q.get("category", ""),
        })

        if (i + 1) % 20 == 0:
            # Intermediate checkpoint
            with open(f"{RESULTS_DIR}/{dataset_name}_checkpoint.json", "w") as f:
                json.dump(results, f, indent=2)

    # Save final results
    with open(f"{RESULTS_DIR}/{dataset_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Compute metrics
    preds_base = [r["pred_baseline"] for r in results["forecasts"]]
    preds_anchor = [r["pred_anchored"] for r in results["forecasts"]]
    outcomes = [r["resolution"] for r in results["forecasts"]]
    community = [r["community_prediction"] for r in results["forecasts"]
                 if r["community_prediction"] is not None]
    outcomes_with_community = [r["resolution"] for r in results["forecasts"]
                               if r["community_prediction"] is not None]

    metrics = {
        "brier_baseline": brier_score(preds_base, outcomes),
        "brier_anchored": brier_score(preds_anchor, outcomes),
        "ece_baseline": calibration_error(preds_base, outcomes),
        "ece_anchored": calibration_error(preds_anchor, outcomes),
    }

    if community:
        metrics["brier_community"] = brier_score(community, outcomes_with_community)
        metrics["ece_community"] = calibration_error(community, outcomes_with_community)

    metrics["brier_naive"] = brier_score([0.5] * len(outcomes), outcomes)

    results["metrics"] = metrics

    # Save with metrics
    with open(f"{RESULTS_DIR}/{dataset_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults for {dataset_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 1: LLM Forecasting Accuracy")
    print("=" * 60)

    # Load datasets
    halawi_data = load_halawi_test()
    kalshi_data = load_kalshibench()
    print(f"Loaded {len(halawi_data)} Halawi questions, {len(kalshi_data)} KalshiBench questions")

    # Run on samples (200 each for cost/time balance)
    halawi_results = run_forecasting_experiment("halawi", halawi_data, sample_size=200)
    kalshi_results = run_forecasting_experiment("kalshi", kalshi_data, sample_size=200)

    # Save combined summary
    summary = {
        "halawi_metrics": halawi_results["metrics"],
        "kalshi_metrics": kalshi_results["metrics"],
    }
    with open(f"{RESULTS_DIR}/experiment1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    main()
