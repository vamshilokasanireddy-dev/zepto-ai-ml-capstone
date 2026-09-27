# Python 3.8.1

Use Python 3.8.1 for this module.

# Data Pipeline

## Purpose

This module implements:

```text
Books to Scrape
    -> requests + BeautifulSoup
    -> cleaning
    -> fixed GBP/INR conversion
    -> normalized SQLite
    -> SQL queries
    -> pandas read_sql / merge validation
```

The required project conversion rate is:

```text
1 GBP = 105.50 INR
```

Run:

```bash
python data_pipeline/scrape_pipeline.py
python data_pipeline/run_queries.py
```

The scripts write generated artifacts under `data_pipeline/output/`.
