# Data Pipeline

Run `python pipeline.py`.

The script scrapes five catalogue pages, producing 100 books across multiple categories. It uses the fixed assignment conversion rate of 1 GBP = 105.50 INR. The SQLite schema is normalized into `categories` and `books`, with `books.category_id` referencing `categories.category_id`.

`output.txt` records SQL queries, their results, and the `pd.read_sql` versus `pd.merge` JOIN comparison.
