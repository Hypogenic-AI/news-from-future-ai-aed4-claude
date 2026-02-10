# Datasets

Datasets relevant to the "News from the Future" research topic: combining LLMs with prediction markets to generate plausible future news articles.

## Downloaded Datasets

### 1. Forecasting Dataset (Halawi et al., 2024)
- **Directory:** `forecasting_halawi2024/`
- **Source:** [HuggingFace: YuehHanChen/forecasting](https://huggingface.co/datasets/YuehHanChen/forecasting)
- **Paper:** "Approaching Human-Level Forecasting with Language Models" (NeurIPS 2024)
- **Size:** 5,516 binary forecasting questions (3,762 train / 840 validation / 914 test)
- **Sources:** Metaculus, Good Judgment Open, INFER, Polymarket, Manifold
- **Fields:** Background description, resolution criteria, timestamps, binary resolution
- **Download:**
  ```python
  from datasets import load_dataset
  ds = load_dataset("YuehHanChen/forecasting")
  ```

### 2. AutoCast (Zou et al., 2022)
- **Directory:** `autocast/`
- **Source:** [HuggingFace: valory/autocast](https://huggingface.co/datasets/valory/autocast) | [GitHub](https://github.com/andyzoujm/autocast)
- **Paper:** "Forecasting Future World Events with Neural Networks" (NeurIPS 2022)
- **Size:** 1,364 test questions + 340 filtered questions
- **Fields:** Question ID, text, background, type (T/F, MCQ, numerical), choices, answer, crowd forecasts, timestamps
- **Note:** Full dataset also includes a 200GB+ news corpus from CommonCrawl (not downloaded here; see GitHub repo for download scripts)
- **Download:**
  ```python
  from huggingface_hub import hf_hub_download
  hf_hub_download("valory/autocast", "autocast_competition_test_set.json", repo_type="dataset")
  ```

### 3. KalshiBench v2 (Smith, 2025)
- **Directory:** `kalshibench/`
- **Source:** [HuggingFace: 2084Collective/kalshibench-v2](https://huggingface.co/datasets/2084Collective/kalshibench-v2)
- **Paper:** "KalshiBench: Evaluating Epistemic Calibration via Prediction Markets" (NeurIPS 2024)
- **Size:** 1,531 binary prediction market questions from Kalshi
- **Fields:** ID, question, description, category, close_time, ground_truth
- **Categories:** Sports, Politics, Entertainment, Companies, Elections, Crypto, Climate, Financials, World, Economics, Social
- **Download:**
  ```python
  from datasets import load_dataset
  ds = load_dataset("2084Collective/kalshibench-v2")
  ```

## Additional Datasets (Not Downloaded - Available via API/Web)

### 4. Metaculus Questions API
- **URL:** https://www.metaculus.com/api2/questions/
- **Size:** 10,000+ questions, 3.49M+ total predictions
- **Access:** REST API, paginated queries
- **Python:** `pip install forecasting-tools` (official Metaculus package)

### 5. ForecastBench (Dynamic)
- **URL:** https://www.forecastbench.org/datasets/
- **GitHub:** [forecastingresearch/forecastbench-datasets](https://github.com/forecastingresearch/forecastbench-datasets)
- **Size:** 1,000 questions per biweekly round
- **License:** MIT

### 6. Polymarket Trading Data
- **HuggingFace:** [SII-WANGZJ/Polymarket_data](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data)
- **Size:** 1.1B trading records across 268K+ markets (107GB)
- **License:** MIT

### 7. Good Judgment Project (IARPA ACE)
- **URL:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BPCDH5
- **Size:** ~500 geopolitical questions, 1M+ forecasts (2011-2015)
- **License:** Public/Academic

### 8. YuehHanChen/forecasting_raw
- **HuggingFace:** [YuehHanChen/forecasting_raw](https://huggingface.co/datasets/YuehHanChen/forecasting_raw)
- **Size:** 48,754 questions with 7,174,607 forecasts (2015-2024)
- **Note:** Uncurated version of the Halawi et al. dataset
