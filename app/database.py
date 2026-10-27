"""SQLite access. Every query is parameterized; ownership rules live in rules/tools."""
import sqlite3
from contextlib import closing


def connect(db_path):
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row  # lets us read columns by name
    return connection


def customer_exists(db_path, customer_id):
    with closing(connect(db_path)) as connection:
        row = connection.execute(
            "SELECT 1 FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
    return row is not None


def load_policy_parameters(db_path):
    """Return {key: value} as text. Values like '5–7' are not numbers, so callers convert."""
    with closing(connect(db_path)) as connection:
        rows = connection.execute("SELECT key, value FROM policy_parameters").fetchall()
    return {row["key"]: row["value"] for row in rows}
