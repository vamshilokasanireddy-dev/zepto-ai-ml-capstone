import sqlite3
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

BASE = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
ROOT = Path(__file__).resolve().parent
DB = ROOT / "zepto.db"
OUTPUT = ROOT / "output.txt"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def scrape():
    rows = []
    for page in range(1, 6):
        url = BASE if page == 1 else f"{BASE}catalogue/page-{page}.html"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for card in soup.select("article.product_pod"):
            title = card.h3.a.get("title", "").strip()
            price = card.select_one(".price_color").get_text(strip=True)
            rating_class = next(
                (c for c in card.select_one("p.star-rating").get("class", [])
                 if c in RATING_MAP), None
            )
            availability = card.select_one(".availability").get_text(" ", strip=True)
            category = "Unknown"
            link = card.h3.a.get("href", "")
            detail_url = requests.compat.urljoin(url, link)
            dr = requests.get(detail_url, timeout=30)
            if dr.ok:
                ds = BeautifulSoup(dr.text, "html.parser")
                breadcrumb = ds.select("ul.breadcrumb li a")
                if breadcrumb:
                    category = breadcrumb[-1].get_text(strip=True)
            rows.append({
                "title": title,
                "price": price,
                "star_rating": rating_class,
                "availability": availability,
                "category": category,
            })
    return pd.DataFrame(rows)

def clean(df):
    out = df.copy()
    out["price_gbp"] = pd.to_numeric(
        out["price"].astype(str).str.replace("£", "", regex=False),
        errors="coerce"
    )
    out["rating"] = out["star_rating"].map(RATING_MAP)
    out["in_stock"] = out["availability"].str.contains(
        "In stock", case=False, na=False
    ).astype(bool)

    # Numeric parse failures: median imputation as required.
    for col in ["price_gbp", "rating"]:
        out[col] = out[col].fillna(out[col].median())
    out = out.dropna(subset=["title", "category"]).copy()
    out["price_inr"] = out["price_gbp"] * GBP_TO_INR
    return out[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]

def create_db(df):
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("""CREATE TABLE categories(
        category_id INTEGER PRIMARY KEY,
        category_name TEXT UNIQUE NOT NULL
    )""")
    cur.execute("""CREATE TABLE books(
        book_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        price_gbp REAL,
        price_inr REAL,
        rating INTEGER,
        in_stock INTEGER,
        category_id INTEGER NOT NULL,
        FOREIGN KEY(category_id) REFERENCES categories(category_id)
    )""")
    cats = sorted(df["category"].unique())
    cur.executemany("INSERT INTO categories(category_name) VALUES (?)",
                    [(c,) for c in cats])
    cat_map = {name: i+1 for i, name in enumerate(cats)}
    rows = []
    for i, r in df.reset_index(drop=True).iterrows():
        rows.append((i+1, r.title, float(r.price_gbp), float(r.price_inr),
                     int(r.rating), int(r.in_stock), cat_map[r.category]))
    cur.executemany("""INSERT INTO books
        (book_id,title,price_gbp,price_inr,rating,in_stock,category_id)
        VALUES (?,?,?,?,?,?,?)""", rows)
    con.commit()
    return con

def run_queries(con, df):
    queries = [
        ("Q1 SELECT/WHERE", "SELECT * FROM books WHERE rating >= 4;"),
        ("Q2 ORDER BY", "SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 10;"),
        ("Q3 LIMIT", "SELECT * FROM books LIMIT 10;"),
        ("Q4 DISTINCT", "SELECT DISTINCT category_name FROM categories ORDER BY category_name;"),
        ("Q5 BETWEEN", "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30;"),
        ("Q6 JOIN", """SELECT b.title,b.rating,b.price_inr,c.category_name
                       FROM books b JOIN categories c
                       ON b.category_id=c.category_id
                       ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;""")
    ]
    with OUTPUT.open("w", encoding="utf-8") as f:
        f.write(f"Rows scraped/cleaned: {len(df)}\n")
        f.write(f"Categories: {df['category'].nunique()}\n")
        f.write("Fixed rate: 1 GBP = 105.50 INR\n\n")
        for name, sql in queries:
            f.write(f"=== {name} ===\n{sql}\n")
            result = pd.read_sql(sql, con)
            f.write(result.to_string(index=False) + "\n\n")

        q_join = queries[-1][1]
        sql_df = pd.read_sql(q_join, con)
        categories_df = pd.read_sql("SELECT * FROM categories", con)
        books_df = pd.read_sql("SELECT * FROM books", con)
        merged = books_df.merge(categories_df, on="category_id")
        merged = merged[["title","rating","price_inr","category_name"]].sort_values(
            ["rating","price_inr"], ascending=[False,False]
        ).head(10).reset_index(drop=True)
        sql_cmp = sql_df.reset_index(drop=True)
        f.write("=== pd.read_sql JOIN vs pd.merge JOIN ===\n")
        f.write("pd.read_sql:\n" + sql_cmp.to_string(index=False) + "\n\n")
        f.write("pd.merge:\n" + merged.to_string(index=False) + "\n\n")
        f.write("Equivalent values: " + str(sql_cmp.equals(merged)) + "\n")

def main():
    raw = scrape()
    cleaned = clean(raw)
    if len(cleaned) < 60 or cleaned["category"].nunique() < 3:
        raise RuntimeError("Acceptance criterion failed: need >=60 rows and >=3 categories.")
    con = create_db(cleaned)
    run_queries(con, cleaned)
    con.close()
    print(f"Completed: {len(cleaned)} books, {cleaned['category'].nunique()} categories")
    print(f"Database: {DB}")
    print(f"SQL output: {OUTPUT}")

if __name__ == "__main__":
    main()
