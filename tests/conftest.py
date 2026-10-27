import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app


@pytest.fixture
def small_db(tmp_path):
    """A tiny database so tests do not depend on the real data/ folder."""
    path = tmp_path / "test.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE customers (customer_id TEXT PRIMARY KEY, name TEXT)")
    connection.execute("INSERT INTO customers VALUES ('C1001', 'Asha'), ('C1002', 'Ravi')")
    connection.commit()
    connection.close()
    return str(path)


@pytest.fixture
def client(small_db):
    # An env dict with no API key at all: the service must still work
    settings = load_settings({"DB_PATH": small_db})
    return TestClient(create_app(settings))


def chat(client, body, customer="C1001"):
    headers = {"X-Customer-Id": customer} if customer else {}
    return client.post("/chat", json=body, headers=headers)
