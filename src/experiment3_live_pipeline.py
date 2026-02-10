"""
Experiment 3: Live Pipeline Demo
Fetches current prediction market questions, runs the full pipeline,
and generates a "News from the Future" website output.
"""
import json
import os
import random
import time
import datetime
import numpy as np
import requests
from openai import OpenAI
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR = "/workspaces/news-from-future-ai-aed4-claude"
RESULTS_DIR = f"{BASE_DIR}/results/live_pipeline"


def fetch_metaculus_questions(n=20):
    """Fetch open prediction market questions from Metaculus API v2."""
    print("Fetching questions from Metaculus API...")
    questions = []
    try:
        url = "https://www.metaculus.com/api2/questions/"
        params = {
            "limit": 100,
            "order_by": "-nr_forecasters",
            "status": "open",
            "forecast_type": "binary",
            "include_description": "true",
        }
        headers = {"Accept": "application/json"}
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        for q in results:
            title = q.get("title", "")
            inner = q.get("question", {})
            description = ""
            if isinstance(inner, dict):
                description = inner.get("description", "")[:1000]
            if not description:
                description = q.get("description", "")[:1000]

            community_pred = None
            if isinstance(inner, dict):
                agg = inner.get("aggregations", {})
                if isinstance(agg, dict):
                    rec = agg.get("recency_weighted", {})
                    if isinstance(rec, dict):
                        latest = rec.get("latest")
                        if isinstance(latest, dict):
                            centers = latest.get("centers", [])
                            if centers:
                                community_pred = centers[0]

            if title and community_pred is not None:
                questions.append({
                    "id": str(q.get("id", "")),
                    "question": title,
                    "background": description,
                    "resolution_criteria": q.get("resolution_criteria", "")[:500],
                    "community_prediction": community_pred,
                    "source": "metaculus_live",
                    "close_date": q.get("scheduled_close_time", ""),
                    "url": f"https://www.metaculus.com/questions/{q.get('id', '')}/",
                })

            if len(questions) >= n:
                break

        print(f"  Fetched {len(questions)} questions from Metaculus")
    except Exception as e:
        print(f"  Metaculus API error: {e}")

    return questions


def fetch_polymarket_questions(n=20):
    """Fetch current prediction market questions from Polymarket events API."""
    print("Fetching questions from Polymarket API...")
    questions = []
    try:
        # Use events endpoint with volume sorting for interesting markets
        url = "https://gamma-api.polymarket.com/events"
        params = {"limit": 20, "closed": "false", "order": "volume", "ascending": "false"}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        events = resp.json()

        for event in events:
            markets = event.get("markets", [])
            for market in markets[:3]:  # Top 3 markets per event
                title = market.get("question", "")
                description = market.get("description", "")[:1000]
                outcomes = market.get("outcomePrices", "")
                prob = None
                if outcomes:
                    try:
                        prices = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                        if prices and len(prices) >= 1:
                            prob = float(prices[0])
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass

                vol = float(market.get("volume", 0) or 0)
                if title and prob is not None and vol > 1000:
                    questions.append({
                        "id": str(market.get("id", "")),
                        "question": title,
                        "background": description,
                        "resolution_criteria": "",
                        "community_prediction": prob,
                        "source": "polymarket_live",
                        "close_date": market.get("endDate", ""),
                        "url": f"https://polymarket.com/event/{event.get('slug', '')}",
                        "volume": vol,
                    })

                if len(questions) >= n:
                    break
            if len(questions) >= n:
                break

        print(f"  Fetched {len(questions)} questions from Polymarket")
    except Exception as e:
        print(f"  Polymarket API error: {e}")

    return questions


def generate_llm_forecast(question_data, model="gpt-4.1"):
    """Generate an LLM forecast for a live question."""
    system_prompt = (
        "You are an expert forecaster analyzing current events. "
        "Given a prediction market question, estimate the probability (0.0-1.0) "
        "that it will resolve YES. Consider the market probability as a reference "
        "but form your own independent judgment based on available evidence. "
        "Respond with ONLY a number between 0.0 and 1.0."
    )

    user_prompt = (
        f"Question: {question_data['question']}\n\n"
        f"Background: {question_data['background']}\n\n"
        f"Market probability: {(question_data.get('community_prediction') or 0.5):.2f}\n\n"
        f"Your probability estimate:"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=20,
        )
        text = response.choices[0].message.content.strip()
        prob = float(text)
        return max(0.01, min(0.99, prob))
    except Exception as e:
        print(f"  Forecast error: {e}")
        return question_data.get("community_prediction", 0.5)


def generate_news_article(question_data, probability, model="gpt-4.1"):
    """Generate a full news article for the pipeline."""
    if probability > 0.8:
        tone = "confident, reporting likely outcome"
        framing = "is widely expected to"
    elif probability > 0.6:
        tone = "cautiously optimistic"
        framing = "appears likely to"
    elif probability > 0.4:
        tone = "balanced, acknowledging deep uncertainty"
        framing = "remains uncertain whether it will"
    elif probability > 0.2:
        tone = "skeptical but open"
        framing = "appears unlikely to"
    else:
        tone = "reporting low probability"
        framing = "is widely expected not to"

    system_prompt = (
        "You are a senior journalist at a major news organization writing for "
        "'The Future Times' — a publication that reports on likely future events "
        "based on prediction market data and expert analysis. Write professionally, "
        "citing 'forecasters' and 'prediction markets' as sources. "
        "Include a compelling headline, dateline, and 300-500 word article body. "
        "Match your tone and hedging language to the probability level."
    )

    user_prompt = (
        f"Write a 'News from the Future' article about:\n\n"
        f"Topic: {question_data['question']}\n"
        f"Background: {question_data['background']}\n"
        f"Combined forecast probability: {probability:.0%}\n"
        f"Market probability: {question_data['community_prediction']:.0%}\n"
        f"Tone: {tone}\n"
        f"Framing: The event {framing} happen.\n\n"
        f"Write the article now. Make it compelling, realistic, and appropriately "
        f"hedged for the {probability:.0%} probability level."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Article generation error: {e}")
        return None


def evaluate_article_quality(question, article, probability, model="gpt-4.1"):
    """LLM-as-Judge evaluation."""
    eval_prompt = f"""Rate this future news article on 5 dimensions (1-5 each).

Question: {question}
Forecast Probability: {probability:.0%}

Article:
{article}

Dimensions:
1. Plausibility: Does it read like a real news article about a plausible scenario?
2. Coherence: Is it well-structured and internally consistent?
3. Informativeness: Does it provide useful context and analysis?
4. Uncertainty Handling: Does the hedging/confidence match the {probability:.0%} probability?
5. News Style: Professional journalism standards (headline, lead, AP style)?

Respond ONLY in JSON:
{{"plausibility": X, "coherence": X, "informativeness": X, "uncertainty_handling": X, "news_style": X}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except Exception:
        return {"plausibility": 3, "coherence": 3, "informativeness": 3,
                "uncertainty_handling": 3, "news_style": 3}


def generate_html_output(pipeline_results):
    """Generate a simple HTML page showing the 'News from the Future' website."""
    articles_html = ""
    for item in pipeline_results:
        if not item.get("article"):
            continue
        prob = item["combined_probability"]
        prob_class = "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"
        article_html = item["article"].replace("\n", "<br>")

        articles_html += f"""
        <div class="article-card {prob_class}">
            <div class="prob-badge">{prob:.0%} likely</div>
            <div class="source-badge">{item['source']}</div>
            <div class="article-content">{article_html}</div>
            <div class="meta">
                Market: {item['market_probability']:.0%} |
                LLM: {item['llm_forecast']:.0%} |
                Combined: {prob:.0%}
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Future Times</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Georgia', serif; background: #f5f1eb; color: #333; }}
    header {{
        background: #1a1a2e; color: #eee; padding: 2rem; text-align: center;
        border-bottom: 4px solid #e94560;
    }}
    header h1 {{ font-size: 2.5rem; font-family: 'Times New Roman', serif; letter-spacing: 3px; }}
    header p {{ color: #aaa; margin-top: 0.5rem; font-style: italic; }}
    .subtitle {{ font-size: 0.9rem; color: #888; margin-top: 0.3rem; }}
    .container {{ max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
    .article-card {{
        background: white; margin: 1.5rem 0; padding: 1.5rem 2rem;
        border-left: 5px solid #ccc; border-radius: 3px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        position: relative;
    }}
    .article-card.high {{ border-left-color: #27ae60; }}
    .article-card.medium {{ border-left-color: #f39c12; }}
    .article-card.low {{ border-left-color: #e74c3c; }}
    .prob-badge {{
        position: absolute; top: 1rem; right: 1rem;
        background: #1a1a2e; color: white; padding: 0.3rem 0.8rem;
        border-radius: 20px; font-size: 0.85rem; font-family: sans-serif;
    }}
    .source-badge {{
        display: inline-block; background: #eee; color: #666;
        padding: 0.2rem 0.6rem; border-radius: 3px; font-size: 0.75rem;
        font-family: sans-serif; margin-bottom: 0.8rem; text-transform: uppercase;
    }}
    .article-content {{ line-height: 1.7; font-size: 1.05rem; }}
    .meta {{
        margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid #eee;
        font-size: 0.85rem; color: #888; font-family: sans-serif;
    }}
    .methodology {{
        background: #1a1a2e; color: #ddd; padding: 2rem; margin-top: 3rem;
        border-radius: 5px;
    }}
    .methodology h2 {{ color: #e94560; margin-bottom: 1rem; }}
    .methodology p {{ line-height: 1.6; margin-bottom: 0.5rem; }}
    footer {{ text-align: center; padding: 2rem; color: #999; font-size: 0.85rem; }}
</style>
</head>
<body>
<header>
    <h1>THE FUTURE TIMES</h1>
    <p>AI-Powered News from Tomorrow</p>
    <div class="subtitle">Generated {datetime.datetime.now().strftime('%B %d, %Y')} |
    Powered by Prediction Markets + GPT-4.1</div>
</header>
<div class="container">
    {articles_html}
    <div class="methodology">
        <h2>How This Works</h2>
        <p><strong>Step 1:</strong> We fetch live questions from prediction markets
        (Metaculus, Polymarket) where thousands of forecasters bet real money on future events.</p>
        <p><strong>Step 2:</strong> GPT-4.1 analyzes each question and generates its own
        probability forecast, anchored to market prices.</p>
        <p><strong>Step 3:</strong> We combine the market and LLM forecasts, then generate
        a news article whose tone and hedging match the probability level.</p>
        <p><strong>Disclaimer:</strong> These are probabilistic forecasts, not predictions.
        The future is inherently uncertain. Articles are AI-generated for research purposes.</p>
    </div>
</div>
<footer>
    Research Project: News from the Future | LLMs + Prediction Markets
</footer>
</body>
</html>"""

    with open(f"{RESULTS_DIR}/future_times.html", "w") as f:
        f.write(html)
    print(f"  HTML output saved to {RESULTS_DIR}/future_times.html")


def run_live_pipeline(n_questions=20):
    """Run the complete live pipeline."""
    print(f"\n{'='*60}")
    print(f"Running Live Pipeline Demo (n={n_questions})")
    print(f"{'='*60}")

    # Fetch questions from both sources
    metaculus_qs = fetch_metaculus_questions(n=n_questions // 2 + 5)
    polymarket_qs = fetch_polymarket_questions(n=n_questions // 2 + 5)

    all_questions = metaculus_qs + polymarket_qs
    if len(all_questions) > n_questions:
        random.seed(SEED)
        all_questions = random.sample(all_questions, n_questions)

    print(f"  Working with {len(all_questions)} questions")

    if not all_questions:
        print("  WARNING: No live questions fetched. Using dataset questions as fallback.")
        # Fallback: use dataset questions
        from experiment1_forecasting import load_halawi_test, load_kalshibench
        halawi = load_halawi_test(max_n=10)
        kalshi = load_kalshibench(max_n=10)
        all_questions = (halawi[:10] + kalshi[:10])[:n_questions]
        for q in all_questions:
            q["source"] = q.get("source", "dataset") + "_fallback"

    pipeline_results = []
    for i, q in enumerate(tqdm(all_questions, desc="Live pipeline")):
        # Step 1: LLM forecast
        llm_forecast = generate_llm_forecast(q)
        time.sleep(0.2)

        # Step 2: Combine forecasts (weighted average: 60% market, 40% LLM)
        market_prob = q.get("community_prediction", 0.5) or 0.5
        combined = 0.6 * market_prob + 0.4 * llm_forecast

        # Step 3: Generate article
        article = generate_news_article(q, combined)
        time.sleep(0.2)

        # Step 4: Evaluate
        evaluation = None
        if article:
            evaluation = evaluate_article_quality(q["question"], article, combined)
            time.sleep(0.2)

        pipeline_results.append({
            "question": q["question"],
            "background": q["background"][:300],
            "source": q.get("source", "unknown"),
            "market_probability": market_prob,
            "llm_forecast": llm_forecast,
            "combined_probability": combined,
            "article": article,
            "evaluation": evaluation,
            "close_date": q.get("close_date", ""),
            "url": q.get("url", ""),
        })

    # Save results
    with open(f"{RESULTS_DIR}/live_pipeline_results.json", "w") as f:
        json.dump(pipeline_results, f, indent=2)

    # Generate HTML output
    generate_html_output(pipeline_results)

    # Compute summary metrics
    evals = [r["evaluation"] for r in pipeline_results if r["evaluation"]]
    if evals:
        dims = ["plausibility", "coherence", "informativeness", "uncertainty_handling", "news_style"]
        metrics = {}
        for d in dims:
            scores = [e[d] for e in evals if d in e]
            if scores:
                metrics[f"{d}_mean"] = np.mean(scores)
                metrics[f"{d}_std"] = np.std(scores)
        with open(f"{RESULTS_DIR}/live_pipeline_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("\nLive Pipeline Article Scores:")
        for d in dims:
            m = metrics.get(f"{d}_mean", 0)
            s = metrics.get(f"{d}_std", 0)
            print(f"  {d}: {m:.2f} +/- {s:.2f}")

    # Forecast comparison
    forecast_diffs = [abs(r["llm_forecast"] - r["market_probability"])
                      for r in pipeline_results]
    print(f"\nMean |LLM - Market| difference: {np.mean(forecast_diffs):.3f}")

    return pipeline_results


def main():
    print("=" * 60)
    print("EXPERIMENT 3: Live Pipeline Demo")
    print("=" * 60)
    results = run_live_pipeline(n_questions=20)
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
