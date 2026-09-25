# Submission checklist

- [ ] Run `data_pipeline/pipeline.py`; confirm >=60 rows and >=3 categories.
- [ ] Confirm `data_pipeline/zepto.db` and `output.txt`.
- [ ] Run `analytics/01_eda.py`.
- [ ] Run `analytics/02_modeling.py`.
- [ ] Confirm `analytics/titanic.csv` and `model_pipeline.joblib`.
- [ ] Run `support_assistant/ingest.py`.
- [ ] Run FastAPI and test one policy + one general query.
- [ ] Test Docker build/run.
- [ ] Update README with actual generated metrics and raw API responses if desired.
- [ ] Git feature branch with at least 2 commits, then merge to main.
- [ ] Make the GitHub repository public.
- [ ] Submit exactly one repository URL.
