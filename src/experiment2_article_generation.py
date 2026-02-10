"""
Experiment 2: Future News Article Generation & Quality Evaluation
Generates news articles about future events conditioned on prediction probabilities,
then evaluates quality using GPT-4.1-as-Judge.
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
RESULTS_DIR = f"{BASE_DIR}/results/articles"


def select_diverse_questions(halawi_data, kalshi_data, n=50):
    """Select diverse questions across datasets and probability ranges."""
    selected = []

    # From Halawi: stratify by probability range
    for data, source in [(halawi_data, "halawi"), (kalshi_data, "kalshi")]:
        n_per = n // 2
        # Group by probability range
        low = [q for q in data if q.get("community_prediction") and q["community_prediction"] < 0.3]
        mid = [q for q in data if q.get("community_prediction") and 0.3 <= q["community_prediction"] <= 0.7]
        high = [q for q in data if q.get("community_prediction") and q["community_prediction"] > 0.7]
        no_pred = [q for q in data if not q.get("community_prediction")]

        for group, count in [(low, n_per // 4), (mid, n_per // 4),
                             (high, n_per // 4), (no_pred, n_per // 4)]:
            if group:
                random.seed(SEED)
                selected.extend(random.sample(group, min(count, len(group))))

    # Pad to n if needed
    all_data = halawi_data + kalshi_data
    while len(selected) < n:
        random.seed(SEED + len(selected))
        q = random.choice(all_data)
        if q not in selected:
            selected.append(q)

    return selected[:n]


def generate_future_article(question_data, probability, condition="anchored", model="gpt-4.1"):
    """Generate a future news article based on a prediction question and probability."""
    q = question_data["question"]
    bg = question_data["background"]

    if probability > 0.75:
        framing = "likely to happen"
        style_hint = "Write as if reporting on something expected to occur."
    elif probability > 0.5:
        framing = "more likely than not"
        style_hint = "Write with measured optimism, acknowledging uncertainty."
    elif probability > 0.25:
        framing = "possible but uncertain"
        style_hint = "Write with significant hedging and uncertainty language."
    else:
        framing = "unlikely"
        style_hint = "Write about why this event is not expected, while acknowledging the possibility."

    system_prompt = (
        "You are a professional news journalist writing for a major publication. "
        "Your task is to write a news article about a future event based on current "
        "forecasting data. The article should be realistic, well-sourced in tone, "
        "and appropriately convey the level of certainty/uncertainty about the outcome. "
        "Write 200-400 words. Include a headline."
    )

    if condition == "anchored":
        user_prompt = (
            f"Write a news article about the following forecasting question:\n\n"
            f"Question: {q}\n\n"
            f"Background: {bg}\n\n"
            f"Current forecast probability: {probability:.0%} ({framing})\n\n"
            f"{style_hint}\n\n"
            f"Write a plausible news article as if you were reporting on the current "
            f"outlook for this event. Use appropriate hedging language that matches "
            f"the probability level."
        )
    else:  # baseline - no probability info
        user_prompt = (
            f"Write a news article about the following topic:\n\n"
            f"Question: {q}\n\n"
            f"Background: {bg}\n\n"
            f"Write a plausible news article discussing the outlook for this event."
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Generation error: {e}")
        return None


def evaluate_article(question, article, probability, model="gpt-4.1"):
    """Evaluate a generated article using LLM-as-Judge on 5 dimensions."""
    eval_prompt = f"""Evaluate the following news article about a future event on 5 dimensions.
Rate each dimension from 1 (poor) to 5 (excellent).

**Forecasting Question:** {question}
**Forecast Probability:** {probability:.0%}

**Article:**
{article}

Rate on these dimensions:
1. **Plausibility** (1-5): Does the article sound like a real news article? Is the scenario reasonable?
2. **Coherence** (1-5): Is the article well-structured, logical, and internally consistent?
3. **Informativeness** (1-5): Does the article provide useful context, analysis, or information?
4. **Uncertainty Handling** (1-5): Does the article appropriately convey the level of certainty/uncertainty matching the {probability:.0%} probability? (High prob = more confident tone, low prob = more hedging)
5. **News Style** (1-5): Does the article follow professional news writing conventions (headline, lead, quotes, AP style)?

Respond in JSON format only:
{{"plausibility": X, "coherence": X, "informativeness": X, "uncertainty_handling": X, "news_style": X, "brief_comment": "..."}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        # Parse JSON from response
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"  Eval error: {e}")
        return {"plausibility": 3, "coherence": 3, "informativeness": 3,
                "uncertainty_handling": 3, "news_style": 3, "brief_comment": "parse_error"}


def run_article_experiment(questions, n_articles=50):
    """Run article generation and evaluation experiment."""
    print(f"\n{'='*60}")
    print(f"Running article generation experiment (n={n_articles})")
    print(f"{'='*60}")

    results = {"articles": [], "n_articles": n_articles}

    for i, q in enumerate(tqdm(questions[:n_articles], desc="Generating articles")):
        prob = q.get("community_prediction", 0.5) or 0.5

        # Generate anchored article
        article_anchored = generate_future_article(q, prob, condition="anchored")
        time.sleep(0.2)

        # Generate baseline article (no probability info)
        article_baseline = generate_future_article(q, prob, condition="baseline")
        time.sleep(0.2)

        # Evaluate both
        eval_anchored = None
        eval_baseline = None
        if article_anchored:
            eval_anchored = evaluate_article(q["question"], article_anchored, prob)
            time.sleep(0.2)
        if article_baseline:
            eval_baseline = evaluate_article(q["question"], article_baseline, prob)
            time.sleep(0.2)

        results["articles"].append({
            "question": q["question"][:200],
            "probability": prob,
            "resolution": q.get("resolution"),
            "source": q.get("source", ""),
            "category": q.get("category", ""),
            "article_anchored": article_anchored,
            "article_baseline": article_baseline,
            "eval_anchored": eval_anchored,
            "eval_baseline": eval_baseline,
        })

        # Checkpoint every 10
        if (i + 1) % 10 == 0:
            with open(f"{RESULTS_DIR}/articles_checkpoint.json", "w") as f:
                json.dump(results, f, indent=2)

    # Compute aggregate metrics
    anchored_scores = {k: [] for k in ["plausibility", "coherence", "informativeness",
                                        "uncertainty_handling", "news_style"]}
    baseline_scores = {k: [] for k in anchored_scores}

    for a in results["articles"]:
        if a["eval_anchored"]:
            for k in anchored_scores:
                anchored_scores[k].append(a["eval_anchored"].get(k, 3))
        if a["eval_baseline"]:
            for k in baseline_scores:
                baseline_scores[k].append(a["eval_baseline"].get(k, 3))

    metrics = {}
    for k in anchored_scores:
        if anchored_scores[k]:
            metrics[f"anchored_{k}_mean"] = np.mean(anchored_scores[k])
            metrics[f"anchored_{k}_std"] = np.std(anchored_scores[k])
        if baseline_scores[k]:
            metrics[f"baseline_{k}_mean"] = np.mean(baseline_scores[k])
            metrics[f"baseline_{k}_std"] = np.std(baseline_scores[k])

    results["metrics"] = metrics

    with open(f"{RESULTS_DIR}/experiment2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nArticle Quality Scores (Anchored vs Baseline):")
    for k in ["plausibility", "coherence", "informativeness", "uncertainty_handling", "news_style"]:
        am = metrics.get(f"anchored_{k}_mean", 0)
        bm = metrics.get(f"baseline_{k}_mean", 0)
        print(f"  {k}: anchored={am:.2f}, baseline={bm:.2f}, diff={am-bm:+.2f}")

    return results


def main():
    from experiment1_forecasting import load_halawi_test, load_kalshibench

    print("=" * 60)
    print("EXPERIMENT 2: Article Generation & Quality")
    print("=" * 60)

    halawi_data = load_halawi_test()
    kalshi_data = load_kalshibench()

    questions = select_diverse_questions(halawi_data, kalshi_data, n=50)
    print(f"Selected {len(questions)} diverse questions")

    results = run_article_experiment(questions, n_articles=50)

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
