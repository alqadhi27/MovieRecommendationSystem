"""Tests for MovieRecommender (src/recommender.py)."""
import numpy as np
import pytest

from src.data_loader import DataLoader
from src.recommender import MovieRecommender


def ids(frame):
    return list(frame["id"])


# ───────────── __init__ and fit ─────────────
def test_init_does_not_load_anything(csv_path):
    rec = MovieRecommender(DataLoader(csv_path))
    assert rec.movies is None and rec.vectorizer is None and rec.matrix is None


def test_fit_returns_self_and_builds_model(csv_path):
    rec = MovieRecommender(DataLoader(csv_path))
    assert rec.fit() is rec
    assert rec.matrix.shape[0] == len(rec.movies) == 10
    assert rec.matrix.shape[1] > 10
    assert "the" not in rec.vectorizer.vocabulary_   # stop words removed


# ───────────── get_movies, list_genres, search_titles ─────────────
def test_get_movies(recommender):
    assert len(recommender.get_movies()) == 10


def test_list_genres_sorted_and_unique(recommender):
    genres = recommender.list_genres()
    assert genres == sorted(genres)
    assert len(genres) == len(set(genres))
    assert "Science Fiction" in genres   # two-word genre stays one genre


def test_search_titles_is_case_insensitive(recommender):
    assert ids(recommender.search_titles("DREAM")) == [1, 3]


def test_search_titles_returns_both_duplicated_titles(recommender):
    assert sorted(ids(recommender.search_titles("echo"))) == [4, 5]


def test_search_titles_limit_and_no_match(recommender):
    assert len(recommender.search_titles("a", limit=2)) == 2
    assert recommender.search_titles("zzzz").empty


# ───────────── _top_n ─────────────
def test_top_n_orders_by_score_and_skips_zeros(recommender):
    scores = np.zeros(10)
    scores[0], scores[3], scores[5] = 0.5, 0.9, 0.7
    top = recommender._top_n(scores, 2, recommender.movies.index)
    assert ids(top) == [4, 6]                         # highest scores first, rows 3 and 5
    assert recommender._top_n(np.zeros(10), 5, recommender.movies.index).empty


def test_top_n_respects_exclude_and_candidates(recommender):
    scores = np.zeros(10)
    scores[0], scores[3], scores[5] = 0.5, 0.9, 0.7
    movies = recommender.movies
    assert ids(recommender._top_n(scores, 5, movies.index, exclude=3)) == [6, 1]
    assert ids(recommender._top_n(scores, 5, movies.index[[0, 5]])) == [6, 1]


# ───────────── recommend_by_title ─────────────
def test_recommend_by_title_excludes_the_chosen_movie(recommender):
    result = recommender.recommend_by_title(1)
    assert 1 not in ids(result)
    assert 0 < len(result) <= 5


def test_recommend_by_title_finds_a_similar_movie(recommender):
    assert 3 in ids(recommender.recommend_by_title(1))   # shares "dream" and "heist"


def test_recommend_by_title_never_exposes_the_score(recommender):
    result = recommender.recommend_by_title(1)
    assert list(result.columns) == list(recommender.movies.columns)


def test_recommend_by_title_respects_n(recommender):
    assert len(recommender.recommend_by_title(1, n=2)) <= 2


# ───────────── filter_movies ─────────────
def test_filter_none_returns_everything(recommender):
    assert len(recommender.filter_movies(None)) == 10
    assert len(recommender.filter_movies({})) == 10


def test_filter_genres_is_or(recommender):
    result = recommender.filter_movies({"genres": ["Horror", "Romance"]})
    assert sorted(ids(result)) == [5, 8, 10]


def test_filter_years_and_rating(recommender):
    years = recommender.filter_movies({"year_min": 2000, "year_max": 2010})
    assert sorted(ids(years)) == [1, 8, 10]
    rated = recommender.filter_movies({"min_rating": 7.5})
    assert sorted(ids(rated)) == [1, 2, 7]


def test_filters_combine_with_and(recommender):
    result = recommender.filter_movies({"genres": ["Science Fiction"], "min_rating": 8.5})
    assert ids(result) == [2]


# ───────────── filters inside recommendations ─────────────
def test_filters_are_applied_before_picking_top_n(recommender):
    result = recommender.recommend_by_title(1, filters={"year_min": 2000})
    assert len(result) > 0
    assert (result["year"] >= 2000).all()


def test_fewer_than_n_returns_only_what_is_available(recommender):
    result = recommender.recommend_by_title(1, filters={"min_rating": 8.5})
    assert ids(result) == [2]          # not padded, filter not relaxed


# ───────────── recommend_by_text ─────────────
def test_recommend_by_text_finds_the_matching_movie(recommender):
    result = recommender.recommend_by_text("a thief who enters dreams")
    assert ids(result)[0] == 1


def test_recommend_by_text_does_not_refit_the_vectorizer(recommender):
    before = len(recommender.vectorizer.vocabulary_)
    recommender.recommend_by_text("a brand new unknown description")
    assert len(recommender.vectorizer.vocabulary_) == before


def test_recommend_by_text_unknown_words_gives_empty(recommender):
    assert recommender.recommend_by_text("zzzz qqqq").empty


def test_recommend_by_text_applies_filters(recommender):
    result = recommender.recommend_by_text("dream heist", filters={"year_min": 2012})
    assert ids(result) == [3]


# ───────────── explanations ─────────────
def test_explain_by_title_lists_only_shared_items(recommender):
    shared = recommender.explain_by_title(1, 2)
    assert shared["genres"] == ["Science Fiction"]
    assert shared["directors"] == ["Christopher Nolan"]
    assert shared["keywords"] == []


def test_explain_by_title_with_nothing_in_common(recommender):
    shared = recommender.explain_by_title(7, 10)
    assert shared == {"genres": [], "keywords": [], "directors": []}


def test_explain_by_text_returns_real_shared_words(recommender):
    words = recommender.explain_by_text("a thief who enters dreams", 1)
    assert "thief" in words
    assert "who" not in words                  # stop word
    assert len(words) == len(set(words))       # no duplicates
    assert recommender.explain_by_text("zzzz", 1) == []
