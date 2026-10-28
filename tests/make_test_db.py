"""Builds a small SQLite file with the same tables as the supplied database.

Tests use this so they never touch data/shopsphere.db. TODAY is 2026-07-01, so
"delivered N days ago" is easy to read in the fixtures below.
"""
import sqlite3
from datetime import date, timedelta

TODAY = date(2026, 7, 1)

SCHEMA = """
CREATE TABLE customers (customer_id TEXT PRIMARY KEY, full_name TEXT NOT NULL, email TEXT NOT NULL,
    city TEXT NOT NULL, plus_member INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT '2025-01-01');
CREATE TABLE products (product_id TEXT PRIMARY KEY, name TEXT NOT NULL, brand TEXT NOT NULL DEFAULT 'B',
    category TEXT NOT NULL, price REAL NOT NULL, returnable TEXT NOT NULL, warranty_class TEXT NOT NULL);
CREATE TABLE orders (order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, order_date TEXT NOT NULL,
    status TEXT NOT NULL, subtotal REAL NOT NULL DEFAULT 0, discount REAL NOT NULL DEFAULT 0,
    shipping_fee REAL NOT NULL DEFAULT 0, total REAL NOT NULL, payment_method TEXT NOT NULL DEFAULT 'upi');
CREATE TABLE order_items (item_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, product_id TEXT NOT NULL,
    qty INTEGER NOT NULL DEFAULT 1, unit_price REAL NOT NULL, opened INTEGER NOT NULL DEFAULT 0,
    defect_reported INTEGER NOT NULL DEFAULT 0);
CREATE TABLE shipments (order_id TEXT PRIMARY KEY, carrier TEXT NOT NULL DEFAULT 'Carrier',
    tracking_no TEXT NOT NULL DEFAULT 'T1', status TEXT NOT NULL, shipped_at TEXT, delivered_at TEXT,
    eta TEXT, original_eta TEXT);
CREATE TABLE policy_parameters (key TEXT PRIMARY KEY, value TEXT NOT NULL, unit TEXT NOT NULL,
    source_doc_id TEXT NOT NULL, source_section TEXT NOT NULL);
CREATE TABLE support_tickets (ticket_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, customer_id TEXT NOT NULL,
    ticket_type TEXT NOT NULL, summary TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE return_requests (request_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, customer_id TEXT NOT NULL,
    item_ids TEXT NOT NULL, reason TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
"""

PRODUCTS = [
    ("P1", "Nimbra Smart Speaker", "smart_home", 4499, "yes", "standard"),
    ("P2", "Aero Earbuds", "in_ear", 2999, "unopened_only", "standard"),
    ("P3", "Gift Card 1000", "gift_card", 1000, "no", "none"),
    ("P4", "Voltix ProBook 14", "laptop", 74990, "yes", "extended"),
    ("P5", "Volt Charger", "accessory", 999, "yes", "standard"),
    ("P6", "Photo Editor Licence", "software", 2000, "no", "none"),
]

# order_id, customer, status, days since delivery (None = not delivered), items [(item_id, product, opened)]
ORDERS = [
    ("SS-1", "C1001", "delivered", 10, [("SS-1-1", "P1", 0)]),
    ("SS-2", "C1001", "delivered", 30, [("SS-2-1", "P1", 1)]),
    ("SS-3", "C1001", "delivered", 31, [("SS-3-1", "P1", 0)]),
    ("SS-4", "C1001", "delivered", 5, [("SS-4-1", "P2", 1), ("SS-4-2", "P1", 0)]),
    ("SS-5", "C1001", "delivered", 2, [("SS-5-1", "P3", 0)]),
    ("SS-6", "C1001", "processing", None, [("SS-6-1", "P5", 0)]),
    ("SS-7", "C1001", "shipped", None, [("SS-7-1", "P5", 0)]),
    ("SS-8", "C1001", "delivered", 5, [("SS-8-1", "P2", 0)]),
    ("SS-9", "C1002", "delivered", 3, [("SS-9-1", "P1", 0)]),
    ("SS-10", "C1001", "cancelled", None, [("SS-10-1", "P5", 0)]),
    ("SS-11", "C1001", "delivered", 400, [("SS-11-1", "P5", 0)]),
    ("SS-12", "C1001", "delivered", 20, [("SS-12-1", "P6", 0)]),
]


def default_parameters():
    return {
        "return_window_days": "30",
        "standard_warranty_months": "12",
        "extended_warranty_months": "24",
    }


def make_db(path, parameters=None, orders=ORDERS, today=TODAY):
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.executemany(
        "INSERT INTO customers (customer_id, full_name, email, city) VALUES (?, ?, ?, 'Pune')",
        [("C1001", "Asha Rao", "asha@example.com"), ("C1002", "Ravi Nair", "ravi@example.com")],
    )
    for pid, name, category, price, returnable, warranty in PRODUCTS:
        connection.execute(
            "INSERT INTO products (product_id, name, category, price, returnable, warranty_class)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (pid, name, category, price, returnable, warranty),
        )
    for key, value in (parameters or default_parameters()).items():
        connection.execute(
            "INSERT INTO policy_parameters VALUES (?, ?, 'x', 'doc', '§1')", (key, value)
        )
    for order_id, customer, status, days, items in orders:
        delivered_at = None
        if days is not None:
            delivered_at = (today - timedelta(days=days)).isoformat() + "T10:00:00"
        connection.execute(
            "INSERT INTO orders (order_id, customer_id, order_date, status, total)"
            " VALUES (?, ?, '2026-05-01T09:00:00', ?, 1000)",
            (order_id, customer, status),
        )
        if status != "cancelled":
            shipment_status = "delivered" if days is not None else status
            connection.execute(
                "INSERT INTO shipments (order_id, status, delivered_at, eta, original_eta)"
                " VALUES (?, ?, ?, '2026-07-04', '2026-07-02')",
                (order_id, shipment_status, delivered_at),
            )
        for item_id, product_id, opened in items:
            connection.execute(
                "INSERT INTO order_items (item_id, order_id, product_id, unit_price, opened)"
                " VALUES (?, ?, ?, 100, ?)",
                (item_id, order_id, product_id, opened),
            )
    connection.commit()
    connection.close()
    return str(path)


def count_rows(db_path, table):
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        connection.close()
