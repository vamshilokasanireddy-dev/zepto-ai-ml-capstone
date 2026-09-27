from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
DB_PATH = OUTPUT / "books.db"


QUERIES = {
    "01_select_where": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
        LIMIT 10;
    """,
    "02_order_by_limit": """
        SELECT title, price_inr, rating
        FROM books
        ORDER BY price_inr DESC
        LIMIT 10;
    """,
    "03_distinct_categories": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    "04_between": """
        SELECT title, price_gbp, price_inr
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp;
    """,
    "05_in": """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
        LIMIT 15;
    """,
    "06_join": """
        SELECT
            b.title,
            b.rating,
            b.price_inr,
            c.category_name
        FROM books b
        JOIN categories c
          ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_inr DESC
        LIMIT 10;
    """,
}


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "books.db does not exist. Run scrape_pipeline.py first."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        with open(OUTPUT / "sql_results.txt", "w", encoding="utf-8") as fh:
            for name, query in QUERIES.items():
                print(f"\n===== {name} =====")
                print(query.strip())
                df = pd.read_sql(query, conn)
                print(df.to_string(index=False))

                fh.write(f"\n===== {name} =====\n")
                fh.write(query.strip() + "\n")
                fh.write(df.to_string(index=False) + "\n")

            # Read two SQL results into pandas.
            q1_df = pd.read_sql(QUERIES["01_select_where"], conn)
            join_sql_df = pd.read_sql(QUERIES["06_join"], conn)

            # Reproduce the join directly in memory with pandas.
            books_df = pd.read_sql(
                """
                SELECT book_id, title, price_gbp, price_inr, rating,
                       in_stock, category_id
                FROM books
                """,
                conn,
            )
            categories_df = pd.read_sql(
                """
                SELECT category_id, category_name
                FROM categories
                """,
                conn,
            )

    merge_df = books_df.merge(
        categories_df,
        on="category_id",
        how="inner",
    )[
        ["title", "rating", "price_inr", "category_name"]
    ].sort_values(
        ["rating", "price_inr"],
        ascending=[False, False],
    ).head(10).reset_index(drop=True)

    sql_compare = join_sql_df.reset_index(drop=True).copy()
    merge_compare = merge_df.reset_index(drop=True).copy()

    equivalent = sql_compare.equals(merge_compare)

    print("\n===== pd.read_sql vs pd.merge =====")
    print("Equivalent:", equivalent)
    print("\npd.read_sql result:")
    print(sql_compare.to_string(index=False))
    print("\npd.merge result:")
    print(merge_compare.to_string(index=False))

    with open(OUTPUT / "pandas_join_validation.txt", "w", encoding="utf-8") as fh:
        fh.write("pd.read_sql result:\n")
        fh.write(sql_compare.to_string(index=False))
        fh.write("\n\npd.merge result:\n")
        fh.write(merge_compare.to_string(index=False))
        fh.write(f"\n\nEquivalent: {equivalent}\n")

    if not equivalent:
        raise AssertionError("pd.read_sql JOIN and pd.merge results are not equivalent.")


if __name__ == "__main__":
    main()
