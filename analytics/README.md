# Python 3.8.1

Use Python 3.8.1 for this module.

# Analytics Pipeline

Run the scripts in order:

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

The first script loads Titanic once using Seaborn, immediately saves `titanic.csv`, cleans the same DataFrame, and creates the EDA outputs.

The second script reads the same `titanic.csv` and performs train/test splitting, train-only preprocessing, three classifiers, imbalance comparison, Random Forest GridSearchCV, and multivariate fare regression.

Generated artifacts are placed in `analytics/output/`.
