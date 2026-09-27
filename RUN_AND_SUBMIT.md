# Run and Submit Checklist

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Data pipeline

```bash
python data_pipeline/scrape_pipeline.py
python data_pipeline/run_queries.py
```

Check:

- at least 60 rows
- at least 3 categories
- `data_pipeline/output/books.db`
- `data_pipeline/output/sql_results.txt`
- `data_pipeline/output/pandas_join_validation.txt`

## 3. Analytics

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

Check:

- `analytics/titanic.csv`
- EDA charts
- `analytics/output/eda_report.txt`
- classifier comparison
- imbalance comparison
- GridSearch/OOB result
- regression metrics
- `analytics/output/model_pipeline.joblib`
- reload validation

## 4. Support assistant

```bash
python support_assistant/ingest.py
uvicorn support_assistant.main:app --reload
```

Test one policy query:

```json
{"query":"How long do I have to report a damaged item?"}
```

Test one general query:

```json
{"query":"What is the capital of France?"}
```

Record the actual JSON responses in the README after running them.

## 5. Git

Create a feature branch and make at least two commits:

```bash
git checkout -b feature/capstone
git add .
git commit -m "Add capstone implementation"
git add .
git commit -m "Add validation and documentation"
git checkout main
git merge feature/capstone
```

Then push:

```bash
git remote add origin https://github.com/YOUR-USERNAME/zepto-data-ai-platform.git
git branch -M main
git push -u origin main
```

## 6. Masai answer box

After verifying the GitHub repository is public, submit only:

```text
https://github.com/YOUR-USERNAME/zepto-data-ai-platform
```

Replace `YOUR-USERNAME` with your actual GitHub username.
