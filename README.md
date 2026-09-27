# Zepto Data & AI Platform

## Overview

This repository contains one connected capstone project with three modules:

1. `/data_pipeline` — web scraping, cleaning, currency conversion, SQLite normalization, SQL querying, and pandas validation.
2. `/analytics` — Titanic profiling, EDA, predictive classification, imbalance handling, hyperparameter tuning, and regression.
3. `/support_assistant` — local document embeddings, ChromaDB retrieval, LangGraph routing, structured responses, and FastAPI.

## Project requirements

## Python version

This project is prepared for **Python 3.8.1**. Use Python 3.8.1 for the assessment environment.

Verify with `python --version`. Expected: `Python 3.8.1`.

Install the consolidated dependencies:

```bash
pip install -r requirements.txt
```

The project uses the required fixed conversion baseline:

```text
1 GBP = 105.50 INR
```

No paid service or API key is required for the graded baseline.

## Module 1 — Data Pipeline

From the repository root:

```bash
python data_pipeline/scrape_pipeline.py
python data_pipeline/run_queries.py
```

The scraper uses `requests` and `BeautifulSoup`, collects the first five pages of the public Books to Scrape catalogue, cleans the fields, converts GBP to INR using 105.50, and creates a normalized SQLite database.

The database is created at:

```text
data_pipeline/output/books.db
```

The query script saves SQL and result outputs under:

```text
data_pipeline/output/
```

## Module 2 — Analytics

The analytics module is deliberately split into two ordered scripts:

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

`01_eda.py` loads the Titanic dataset once with:

```python
sns.load_dataset("titanic")
```

and immediately saves:

```text
analytics/titanic.csv
```

The modeling script then reads that same committed CSV and does not independently call `sns.load_dataset`.

The EDA script produces charts and a text report in:

```text
analytics/output/
```

The modeling script saves the complete fitted preprocessing + estimator pipeline as:

```text
analytics/output/model_pipeline.joblib
```

## Module 3 — Support Assistant

First create the local vector store:

```bash
python support_assistant/ingest.py
```

Then run the API:

```bash
uvicorn support_assistant.main:app --reload
```

The API endpoint is:

```text
POST /ask
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"How long do I have to report a damaged item?\"}"
```

On Linux/macOS, use:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"How long do I have to report a damaged item?"}'
```

The graded baseline leaves `MOCK_LLM` unset or sets it to `1`. No LLM API call is made in this mode.

## Docker

From the repository root:

```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support-assistant
```

Then send a request to:

```text
http://127.0.0.1:7860/ask
```

## Git workflow

The repository should contain a feature branch with at least two commits and a merge back into `main`, as required by the capstone rubric.

## Academic integrity

This repository is a technical implementation scaffold. Before submission, review the code, run every module yourself, verify the generated outputs, and rewrite the interpretations/recommendations in your own words where required by the assessment.
