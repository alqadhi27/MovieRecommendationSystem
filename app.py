"""Movie Recommendation System: Streamlit interface (presentation layer only).

No data cleaning and no similarity computation here: everything comes from
DataLoader and MovieRecommender.
Run from the project root:  streamlit run app.py
"""
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import DataLoader
from src.recommender import MovieRecommender

DATA_PATH = Path(__file__).resolve().parent / "data" / "movies_filtered.csv"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"
MAX_RESULTS = 5      # recommendations per request
SEARCH_LIMIT = 20    # title search matches shown
HOME_SIZE = 12       # movies on the start page
GRID_COLUMNS = 4     # cards per row on browse pages

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    [data-testid="stImage"] img { border-radius: 10px; }
    .block-container { padding-top: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ───────────────────────── Model loading (once per server) ─────────────────────────
@st.cache_resource(show_spinner="Loading movies and building the TF-IDF model...")
def load_recommender():
    return MovieRecommender(DataLoader(DATA_PATH)).fit()


# ───────────────────────── Small helpers ─────────────────────────
def safe(text):
    """Escape characters that Markdown would interpret inside a title."""
    return re.sub(r"([\\`*_$\[\]])", r"\\\1", str(text))


def year_text(year):
    return "?" if pd.isna(year) else str(int(year))


def runtime_text(runtime):
    return "Unknown" if pd.isna(runtime) else f"{int(runtime)} min"


def poster_url(value):
    """Full poster URL, or None when the movie has no poster."""
    if pd.isna(value) or not str(value).strip():
        return None
    value = str(value).strip()
    if value.startswith("http"):
        return value
    return POSTER_BASE + (value if value.startswith("/") else "/" + value)


def title_explanation(shared):
    """Lines built only from what the two movies really share."""
    lines = []
    if shared["genres"]:
        lines.append("Shared genres: " + ", ".join(shared["genres"]))
    if shared["keywords"]:
        lines.append("Shared keywords: " + ", ".join(shared["keywords"]))
    if shared["directors"]:
        lines.append("Same director: " + ", ".join(shared["directors"]))
    return lines or ["No shared genre, keyword or director; the similarity comes from other words in the texts."]


def text_explanation(words):
    if words:
        return ["Words from your description found in this movie: " + ", ".join(words)]
    return ["No word from your description appears in this movie's text."]


def build_filters(genres, year_range, year_bounds, min_rating):
    """Only the filters the user actually narrowed; None means no filtering."""
    filters = {}
    if genres:
        filters["genres"] = genres
    if year_range[0] > year_bounds[0]:
        filters["year_min"] = year_range[0]
    if year_range[1] < year_bounds[1]:
        filters["year_max"] = year_range[1]
    if min_rating > 0:
        filters["min_rating"] = min_rating
    return filters or None


# ───────────────────────── Session state ─────────────────────────
st.session_state.setdefault("selected_id", None)
st.session_state.setdefault("desc_query", "")
st.session_state.setdefault("seed", 0)


def select_movie(movie_id):
    st.session_state.selected_id = movie_id


def clear_selection():
    st.session_state.selected_id = None


def shuffle():
    st.session_state.seed += 1


# ───────────────────────── Cards and grids ─────────────────────────
def render_card(row, key_prefix, explanation=None, similar_button=True):
    """Compact movie card: poster, title, rating, and a details expander."""
    with st.container(border=True):
        url = poster_url(getattr(row, "poster_path", None))
        if url:
            st.image(url)
        else:
            st.markdown("🎞️ *No poster available*")
        st.markdown(f"**{safe(row.title)}** ({year_text(row.year)})")
        st.caption(f"⭐ {row.vote_average:.1f} · " + (", ".join(row.genres_list[:2]) or "No genre"))
        with st.expander("Details & why" if explanation else "Details"):
            st.caption(" · ".join([
                ", ".join(row.genres_list) or "No genre",
                f"Language: {row.original_language}",
                f"Runtime: {runtime_text(row.runtime)}",
            ]))
            st.write(row.overview if pd.notna(row.overview) else "No overview available.")
            if explanation:
                st.markdown("**Why this recommendation?**")
                for line in explanation:
                    st.markdown(f"- {line}")
        if similar_button:
            st.button("Find similar", key=f"{key_prefix}_{row.id}",
                      on_click=select_movie, args=(int(row.id),))


def render_grid(movies, per_row, key_prefix, explain=None, similar_button=True):
    rows = list(movies.itertuples())
    for start in range(0, len(rows), per_row):
        columns = st.columns(per_row)
        for column, row in zip(columns, rows[start:start + per_row]):
            with column:
                explanation = explain(int(row.id)) if explain else None
                render_card(row, key_prefix, explanation, similar_button)


def render_recommendations(results, key_prefix, explain, similar_button=True):
    if results.empty:
        st.warning("No movie matches your request and filters. Try other words or relax the filters.")
        return
    if len(results) < MAX_RESULTS:
        st.info(f"Only {len(results)} movie(s) match your request and filters (no filter was relaxed).")
    render_grid(results, MAX_RESULTS, key_prefix, explain, similar_button)


def render_selected(row):
    """Large card for the movie the recommendations are based on."""
    with st.container(border=True):
        poster_col, info_col = st.columns([1, 4], gap="large")
        with poster_col:
            url = poster_url(getattr(row, "poster_path", None))
            if url:
                st.image(url)
            else:
                st.markdown("🎞️ *No poster available*")
        with info_col:
            st.subheader(f"{row.title} ({year_text(row.year)})")
            st.caption(" · ".join([
                ", ".join(row.genres_list) or "No genre",
                f"Language: {row.original_language}",
                f"Runtime: {runtime_text(row.runtime)}",
                f"⭐ {row.vote_average:.1f}",
            ]))
            st.write(row.overview if pd.notna(row.overview) else "No overview available.")
            st.button("← Back to all movies", on_click=clear_selection)


# ───────────────────────── Page ─────────────────────────
st.title("🎬 Movie Recommender")
st.caption("Content-Based Recommendation System using TF-IDF and Cosine Similarity.")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}. Generate it with scripts/prepare_dataset.py.")
    st.stop()

rec = load_recommender()
movies = rec.get_movies()
year_bounds = (int(movies["year"].min()), int(movies["year"].max()))

with st.sidebar:
    st.header("Filters (optional)")
    genres = st.multiselect("Genres (any of)", rec.list_genres())
    year_range = st.slider("Release year", year_bounds[0], year_bounds[1], year_bounds)
    min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, step=0.5)
    st.caption("Filters apply to the movies shown and to recommendations. "
               "They are applied before picking the top 5 and never change the ranking.")
    st.divider()
    st.caption("The system compares words, not meanings, and is not a trained model.")

filters = build_filters(genres, year_range, year_bounds, min_rating)

tab_title, tab_text = st.tabs(["🔎 Discover & similar movies", "✍️ From a description"])

with tab_title:
    query = st.text_input("Search a movie by title", key="query", on_change=clear_selection,
                          placeholder="e.g. Inception").strip()
    selected_id = st.session_state.selected_id

    if selected_id is not None:
        chosen = next(movies[movies["id"] == selected_id].itertuples())
        render_selected(chosen)
        st.markdown("### Recommended for you")
        results = rec.recommend_by_title(selected_id, n=MAX_RESULTS, filters=filters)
        render_recommendations(results, "rec",
                               lambda rid: title_explanation(rec.explain_by_title(selected_id, rid)))

    elif query:
        found = rec.search_titles(query, limit=SEARCH_LIMIT)
        if found.empty:
            st.warning(f'No movie title contains "{query}".')
        else:
            st.markdown(f"### Results for “{safe(query)}”")
            if len(found) == SEARCH_LIMIT:
                st.caption(f"Showing the first {SEARCH_LIMIT} matches. Type more of the title to narrow down.")
            render_grid(found, GRID_COLUMNS, "search")

    else:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("### Start with a movie")
        with right:
            mode = st.radio("Show", ["Most voted", "Random"], horizontal=True, label_visibility="collapsed")
        pool = rec.filter_movies(filters)
        if pool.empty:
            st.warning("No movie matches the current filters.")
        elif mode == "Most voted":
            render_grid(pool.nlargest(HOME_SIZE, "vote_count"), GRID_COLUMNS, "home")
        else:
            st.button("🔀 Shuffle", on_click=shuffle)
            sample = pool.sample(min(HOME_SIZE, len(pool)), random_state=st.session_state.seed)
            render_grid(sample, GRID_COLUMNS, "home")

with tab_text:
    description = st.text_area(
        "Describe the movie you want (English only)", height=120,
        placeholder="e.g. a thief who enters people's dreams to steal secrets",
    )
    if st.button("Find movies", type="primary"):
        if description.strip():
            st.session_state.desc_query = description.strip()
        else:
            st.session_state.desc_query = ""
            st.warning("Please write a description first.")

    text = st.session_state.desc_query
    if text:
        st.markdown("### Recommended for you")
        results = rec.recommend_by_text(text, n=MAX_RESULTS, filters=filters)
        render_recommendations(results, "desc", lambda rid: text_explanation(rec.explain_by_text(text, rid)),
                               similar_button=False)
