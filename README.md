# Zepto Data & AI Platform — Capstone

One repository containing three graded modules:
- `data_pipeline/` — web scraping, cleaning, SQLite, SQL and pandas.
- `analytics/` — Titanic EDA, classification, imbalance handling, tuning and regression.
- `support_assistant/` — local embeddings + ChromaDB + LangGraph + FastAPI.

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

The project uses the assignment's fixed currency baseline: **1 GBP = 105.50 INR**.

## Module 1 — Data Pipeline

```bash
cd data_pipeline
python pipeline.py
```

This scrapes the first five catalogue pages from `books.toscrape.com` (100 books), cleans the fields, converts GBP to INR using 105.50, creates `zepto.db`, executes the required SQL queries, saves outputs to `output.txt`, and demonstrates `pd.read_sql` and `pd.merge`.

## Module 2 — Analytics

Run the EDA first:

```bash
cd analytics
python 01_eda.py
```

This downloads Seaborn's Titanic dataset once and commits the offline fallback `titanic.csv`.

Then:

```bash
python 02_modeling.py
```

The modeling script reads the committed CSV rather than calling `sns.load_dataset` again. It creates model metrics, plots, `model_pipeline.joblib`, and text/CSV outputs under `outputs/`.

## Module 3 — Support Assistant

```bash
cd support_assistant
python ingest.py
uvicorn main:app --host 0.0.0.0 --port 7860
```

Default mode is `MOCK_LLM=1`, so no LLM API key is required. Open:
`http://127.0.0.1:7860/docs`

Example:

```bash
curl -X POST http://127.0.0.1:7860/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the delivery fee?\"}"
```

General-question example:

```bash
curl -X POST http://127.0.0.1:7860/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the capital of India?\"}"
```

### Docker

From `support_assistant/`:

```bash
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

## Architecture

### Data pipeline
Scrape → clean → fixed-rate currency conversion → normalized SQLite → SQL queries → pandas verification.

### Analytics
Load raw Titanic once → save `titanic.csv` → profile/clean → EDA → stratified split → train-only preprocessing → classifiers → evaluation/tuning → regression → save complete fitted pipeline.

### RAG support assistant
Ingestion → chunking → local `all-MiniLM-L6-v2` embeddings → ChromaDB → LangGraph intent classification → retrieval for policy questions → mock/optional LLM generation → Pydantic response → FastAPI.

`MOCK_LLM` branches only the generation/classification LLM-dependent portions. The graded default is deterministic mock mode; retrieval still uses real local embeddings and ChromaDB.

## Design decisions

- Scraping uses `requests` and `BeautifulSoup`.
- Fixed `GBP_TO_INR = 105.50` is used exactly as required.
- Numeric parse failures use median imputation; rows with missing essential text/category fields are skipped.
- SQLite uses `categories(category_id)` and `books(category_id)` as a PK/FK relationship.
- Analytics preprocessing is placed inside scikit-learn pipelines so fitting happens only on training data.
- SMOTE is applied only to training data.
- The saved classifier artifact contains preprocessing plus estimator in one `Pipeline`.
- The support assistant uses deterministic mock mode by default, avoiding paid services or API keys.
