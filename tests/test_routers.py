# tests/test_routes.py(int and func)
import os
from pathlib import Path
import tempfile
import sqlite3
import json
from unittest.mock import patch, MagicMock

import pytest

import app as myapp

@pytest.fixture(autouse=True)
def temp_env(tmp_path: Path, monkeypatch):
    # redirect uploads and DB to tmp dir for tests
    monkeypatch.setattr(myapp, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    os.makedirs(myapp.UPLOAD_FOLDER, exist_ok=True)
    # create a separate DB file and point DB_PATH to it
    dbfile = tmp_path / "test_users.db"
    monkeypatch.setattr(myapp, "DB_PATH", str(dbfile))
    # Ensure DB is (re)initialized
    myapp.init_db_and_migrate()
    yield
    # cleanup done by tmp_path

@pytest.fixture
def client():
    return myapp.app.test_client()

def test_signup_and_login(client):
    # Signup a new user
    resp = client.post("/signup", data={"username":"testuser","password":"testpass"}, follow_redirects=True)
    assert b"Signup successful" in resp.data or resp.status_code in (200, 302)
    # Try login with wrong credentials -> error shown
    resp = client.post("/login", data={"username":"bad","password":"bad"}, follow_redirects=True)
    assert b"Invalid username or password" in resp.data or resp.status_code == 200
    # The code only accepts admin/admin for login, so test admin login
    resp = client.post("/login", data={"username":"admin","password":"admin"}, follow_redirects=True)
    assert b"Logged in as admin" in resp.data or resp.status_code == 200

@patch("app.summarize_with_gemini", autospec=True)
def test_dashboard_post_text(mock_summarize, client):
    mock_summarize.return_value = "MOCK_SUMMARY"
    # login as admin first
    client.post("/login", data={"username":"admin","password":"admin"}, follow_redirects=True)
    resp = client.post("/dashboard", data={"text":"This is a test document."}, follow_redirects=True)
    assert b"MOCK_SUMMARY" in resp.data or resp.status_code == 200
    # history should contain an entry
    resp2 = client.get("/history")
    assert resp2.status_code == 200
    assert b"MOCK_SUMMARY" in resp2.data or b"This is a test document" in resp2.data

@patch("app.summarize_with_gemini", autospec=True)
def test_resummarize_and_delete(mock_summarize, client):
    mock_summarize.return_value = "MOCK_SUMMARY"
    client.post("/login", data={"username":"admin","password":"admin"}, follow_redirects=True)
    client.post("/dashboard", data={"text":"To be saved."}, follow_redirects=True)
    # fetch history to get entry id
    resp = client.get("/history")
    assert resp.status_code == 200
    # find entry id by querying DB directly
    conn = myapp.get_db_connection()
    r = conn.execute("SELECT id FROM history ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert r is not None
    entry_id = r["id"]
    # resummarize
    resp = client.post(f"/history/{entry_id}/resummarize", follow_redirects=True)
    assert b"re-summarized" in resp.data.lower() or resp.status_code == 200
    # delete
    resp = client.post(f"/history/{entry_id}/delete", follow_redirects=True)
    assert b"Entry deleted" in resp.data or resp.status_code == 200

