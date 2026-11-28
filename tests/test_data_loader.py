"""Tests for DataLoader (src/data_loader.py)."""
import re
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataLoader


# ───────────── __init__ ─────────────
def test_init_converts_to_path():
    assert isinstance(DataLoader("x.csv").csv_path, Path)


def test_init_respects_given_path():
    assert DataLoader("a.csv").csv_path != DataLoader("b.csv").csv_path


# ───────────── prepare ─────────────
def test_prepare_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "nothing.csv"
    with pytest.raises(FileNotFoundError, match="nothing.csv"):
        DataLoader(missing).prepare()


def test_prepare_keeps_all_rows_and_adds_columns(prepared):
    assert len(prepared) == 10
    for column in ["year", "genres_list", "keywords_list", "directors_list", "text"]:
        assert column in prepared.columns


# ───────────── _clean ─────────────
def test_clean_turns_zero_runtime_into_missing(prepared):
    assert prepared["runtime"].isna().sum() == 1
    assert str(prepared["runtime"].dtype) == "Int64"


def test_clean_extracts_year(prepared):
    years = dict(zip(prepared["id"], prepared["year"]))
    assert years[1] == 2010
    assert years[7] == 1920
    assert str(prepared["year"].dtype) == "Int64"


def test_clean_does_not_modify_input(loader, csv_path):
    raw = pd.read_csv(csv_path)
    result = loader._clean(raw)
    assert (raw["runtime"] == 0).sum() == 1   # original untouched
    assert "year" not in raw.columns
    assert result is not raw


# ───────────── _split_list ─────────────
@pytest.mark.parametrize("value, expected", [
    ("Action, Science Fiction", ["Action", "Science Fiction"]),
    ("a,,b", ["a", "b"]),
    ("  spaced ,  out  ", ["spaced", "out"]),
    ("", []),
    (float("nan"), []),
    (None, []),
])
def test_split_list(loader, value, expected):
    assert loader._split_list(value) == expected


def test_lists_are_empty_for_missing_values(prepared):
    row6 = prepared[prepared["id"] == 6].iloc[0]
    row7 = prepared[prepared["id"] == 7].iloc[0]
    assert row6["keywords_list"] == []
    assert row7["directors_list"] == []


# ───────────── _build_text_features ─────────────
def test_text_is_always_a_string_without_missing_markers(prepared):
    assert prepared["text"].map(type).eq(str).all()
    assert not prepared["text"].str.contains(r"\b(?:nan|None)\b", flags=re.IGNORECASE).any()


def test_text_contains_all_parts(prepared):
    text = prepared[prepared["id"] == 1].iloc[0]["text"]
    assert "thief" in text            # overview
    assert "Science Fiction" in text  # genres
    assert "heist" in text            # keywords
    assert "Nolan" in text            # director


def test_text_is_valid_for_movie_without_keywords_or_directors(prepared):
    assert prepared[prepared["id"] == 6].iloc[0]["text"].strip() != ""
    assert prepared[prepared["id"] == 7].iloc[0]["text"].strip() != ""
