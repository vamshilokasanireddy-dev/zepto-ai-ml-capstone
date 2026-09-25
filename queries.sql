SELECT * FROM books WHERE rating >= 4;

SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 10;

SELECT * FROM books LIMIT 10;

SELECT DISTINCT category_name FROM categories ORDER BY category_name;

SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30;

SELECT b.title,b.rating,b.price_inr,c.category_name
FROM books b
JOIN categories c ON b.category_id=c.category_id
ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;
