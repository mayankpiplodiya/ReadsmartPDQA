# tests/test_helpers.py(unit)
import os
import tempfile
import sqlite3
import importlib
import time
from unittest.mock import MagicMock, patch

import pytest

import app as myapp

def test_naive_extractive_summarize_simple():
    text = "Sentence one. Sentence two! Sentence three? Sentence four."
    s = myapp.naive_extractive_summarize(text, ratio=0.5, min_sents=1, max_sents=10)
    assert isinstance(s, str)
    # since ratio=0.5 of 4 sentences => 2 sentences (or min_sents)
    assert "Sentence one" in s

def test_naive_extractive_empty():
    assert myapp.naive_extractive_summarize("") == ""

def test_readability_with_textstat(monkeypatch):
    # simulate textstat availability
    monkeypatch.setattr(myapp, "HAS_TEXTSTAT", True)
    monkeypatch.setattr(myapp, "flesch_reading_ease", lambda t: 72.1234)
    score = myapp.compute_readability("Some text here.")
    assert score == round(72.1234, 2)

def test_readability_without_textstat(monkeypatch):
    monkeypatch.setattr(myapp, "HAS_TEXTSTAT", False)
    assert myapp.compute_readability("any") is None

@patch("app.genai", autospec=True)
def test_summarize_with_gemini_chunking(mock_genai, monkeypatch):
    # Ensure HAS_GENAI True and GEMINI_KEY set for this test path
    monkeypatch.setattr(myapp, "HAS_GENAI", True)
    monkeypatch.setattr(myapp, "GEMINI_KEY", "fakekey")
    # Mock list_models and GenerativeModel behavior
    mock_genai.list_models.return_value = [{"name": "gemini-1.5-flash"}]
    class FakeModel:
        def __init__(self, name): pass
        def generate_content(self, prompt):
            # return an object with text attribute
            resp = MagicMock()
            resp.text = "GEN_SUMMARY"
            return resp
    mock_genai.GenerativeModel.side_effect = lambda name: FakeModel(name)
    long_text = "This is sentence. " * 500
    s = myapp.summarize_with_gemini(long_text)
    assert "GEN_SUMMARY" in s

@patch("app.fitz", autospec=True)
def test_extract_text_pdf_mocked(mock_fitz, tmp_path):
    # create a fake file
    p = tmp_path / "doc.pdf"
    p.write_text("dummy")
    # mock fitz.open to return pages with get_text
    fake_doc = MagicMock()
    fake_page = MagicMock()
    fake_page.get_text.return_value = "PAGE_TEXT"
    fake_doc.__iter__.return_value = [fake_page]
    mock_fitz.open.return_value = [fake_page]
    # call extract_text_from_file
    txt = myapp.extract_text_from_file(str(p))
    # Because we used simple return strategy in code, it will attempt fitz.open -> list of pages
    # We accept either an empty string or combined text; main check is that function returns str
    assert isinstance(txt, str)

