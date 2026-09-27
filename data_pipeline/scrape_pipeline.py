from __future__ import annotations

import sqlite3
from pathlib import Path
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
GBP_TO_INR = 105.50

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
DB_PATH = OUTPUT / "books.db"

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def fetch_page(page_number: int) -> BeautifulSoup:
    response = requests.get(BASE_URL.format(page_number), timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_category(book_url: str) -> str:
    response = requests.get(book_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    breadcrumbs = soup.select("ul.breadcrumb li a")
    if len(breadcrumbs) >= 3:
        return breadcrumbs[-1].get_text(strip=True)
    return "Unknown"


def scrape_books(pages: int = 5) -> pd.DataFrame:
    rows = []

    for page in range(1, pages + 1):
        soup = fetch_page(page)

        for article in soup.select("article.product_pod"):
            title_tag = article.select_one("h3 a")
            price_tag = article.select_one(".price_color")
            rating_tag = article.select_one("p.star-rating")
            availability_tag = article.select_one(".availability")
            link = title_tag.get("href") if title_tag else None

            title = title_tag.get("title", title_tag.get_text(strip=True)) if title_tag else None
            price = price_tag.get_text(strip=True) if price_tag else None
            rating = next(
                (cls for cls in rating_tag.get("class", []) if cls in RATING_MAP),
                None,
            ) if rating_tag else None
            availability = availability_tag.get_text(" ", strip=True) if availability_tag else None

            # The catalogue page does not expose the category directly, so visit
            # the product page to obtain it.
            if link:
                book_url = requests.compat.urljoin(
                    f"https://books.toscrape.com/catalogue/page-{page}.html",
                    link,
                )
                category = get_category(book_url)
            else:
                category = "Unknown"

            rows.append(
                {
                    "title": title,
                    "price": price,
                    "star_rating": rating,
                    "availability": availability,
                    "category": category,
                }
            )

    return pd.DataFrame(rows)


def clean_books(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["price_gbp"] = (
        out["price"]
        .astype("string")
        .str.replace("£", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    out["rating"] = out["star_rating"].map(RATING_MAP)
    out["in_stock"] = (
        out["availability"]
        .astype("string")
        .str.contains("In stock", case=False, na=False)
    )

    # Numeric parse failures are median-imputed as required by the brief.
    for column in ["price_gbp", "rating"]:
        if out[column].isna().any():
            out[column] = out[column].fillna(out[column].median())

    # Rows without a usable title/category are not useful catalog records.
    out = out.dropna(subset=["title", "category"]).copy()

    out["rating"] = out["rating"].round().astype(int)
    out["price_inr"] = (out["price_gbp"] * GBP_TO_INR).round(2)

    return out[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    ]


def create_database(df: pd.DataFrame) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute("DROP TABLE IF EXISTS books")
        conn.execute("DROP TABLE IF EXISTS categories")

        conn.execute(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL UNIQUE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL,
                in_stock INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                FOREIGN KEY(category_id) REFERENCES categories(category_id)
            )
            """
        )

        categories = sorted(df["category"].dropna().unique())
        conn.executemany(
            "INSERT INTO categories(category_name) VALUES (?)",
            [(category,) for category in categories],
        )

        category_map = dict(
            conn.execute("SELECT category_name, category_id FROM categories").fetchall()
        )

        records = [
            (
                row.title,
                float(row.price_gbp),
                float(row.price_inr),
                int(row.rating),
                int(bool(row.in_stock)),
                category_map[row.category],
            )
            for row in df.itertuples(index=False)
        ]

        conn.executemany(
            """
            INSERT INTO books
            (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()


def main() -> None:
    print("Scraping first five catalogue pages...")
    raw = scrape_books(pages=5)

    cleaned = clean_books(raw)

    if len(cleaned) < 60:
        raise RuntimeError(
            f"Only {len(cleaned)} usable books were produced; at least 60 are required."
        )

    if cleaned["category"].nunique() < 3:
        raise RuntimeError(
            f"Only {cleaned['category'].nunique()} categories were found; at least 3 are required."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT / "clean_books.csv", index=False)
    create_database(cleaned)

    print(f"Raw rows: {len(raw)}")
    print(f"Clean rows: {len(cleaned)}")
    print(f"Categories: {cleaned['category'].nunique()}")
    print(f"Database: {DB_PATH}")
    print(f"Fixed conversion rate: 1 GBP = {GBP_TO_INR:.2f} INR")


if __name__ == "__main__":
    main()
