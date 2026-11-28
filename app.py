#"""Movie Recommendation System: Streamlit interface (presentation layer only).
#
#No data cleaning and no similarity computation here: everything comes from
#DataLoader and MovieRecommender.
#Run from the project root:  streamlit run app.py
#"""
#import re
#from pathlib import Path
#
#import pandas as pd
#import streamlit as st
#
#from src.data_loader import DataLoader
#from src.recommender import MovieRecommender
#
#DATA_PATH = Path(__file__).resolve().parent / "data" / "movies_filtered.csv"
#POSTER_BASE = "https://image.tmdb.org/t/p/w342"
#MAX_RESULTS = 5      # recommendations per request
#SEARCH_LIMIT = 20    # title search matches shown
#HOME_SIZE = 12       # movies on the start page
#GRID_COLUMNS = 4     # cards per row on browse pages
#
#st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide",
#                   initial_sidebar_state="expanded")
#
#st.markdown(
#    """
#    <style>
#    [data-testid="stImage"] img { border-radius: 10px; }
#    .block-container { padding-top: 2.5rem; }
#    </style>
#    """,
#    unsafe_allow_html=True,
#)
#
#
## ───────────────────────── Model loading (once per server) ─────────────────────────
#@st.cache_resource(show_spinner="Loading movies and building the TF-IDF model...")
#def load_recommender():
#    return MovieRecommender(DataLoader(DATA_PATH)).fit()
#
#
## ───────────────────────── Small helpers ─────────────────────────
#def safe(text):
#    """Escape characters that Markdown would interpret inside a title."""
#    return re.sub(r"([\\`*_$\[\]])", r"\\\1", str(text))
#
#
#def year_text(year):
#    return "?" if pd.isna(year) else str(int(year))
#
#
#def runtime_text(runtime):
#    return "Unknown" if pd.isna(runtime) else f"{int(runtime)} min"
#
#
#def poster_url(value):
#    """Full poster URL, or None when the movie has no poster."""
#    if pd.isna(value) or not str(value).strip():
#        return None
#    value = str(value).strip()
#    if value.startswith("http"):
#        return value
#    return POSTER_BASE + (value if value.startswith("/") else "/" + value)
#
#
#def title_explanation(shared):
#    """Lines built only from what the two movies really share."""
#    lines = []
#    if shared["genres"]:
#        lines.append("Shared genres: " + ", ".join(shared["genres"]))
#    if shared["keywords"]:
#        lines.append("Shared keywords: " + ", ".join(shared["keywords"]))
#    if shared["directors"]:
#        lines.append("Same director: " + ", ".join(shared["directors"]))
#    return lines or ["No shared genre, keyword or director; the similarity comes from other words in the texts."]
#
#
#def text_explanation(words):
#    if words:
#        return ["Words from your description found in this movie: " + ", ".join(words)]
#    return ["No word from your description appears in this movie's text."]
#
#
#def build_filters(genres, year_range, year_bounds, min_rating):
#    """Only the filters the user actually narrowed; None means no filtering."""
#    filters = {}
#    if genres:
#        filters["genres"] = genres
#    if year_range[0] > year_bounds[0]:
#        filters["year_min"] = year_range[0]
#    if year_range[1] < year_bounds[1]:
#        filters["year_max"] = year_range[1]
#    if min_rating > 0:
#        filters["min_rating"] = min_rating
#    return filters or None
#
#
## ───────────────────────── Session state ─────────────────────────
#st.session_state.setdefault("selected_id", None)
#st.session_state.setdefault("desc_query", "")
#st.session_state.setdefault("seed", 0)
#
#
#def select_movie(movie_id):
#    st.session_state.selected_id = movie_id
#
#
#def clear_selection():
#    st.session_state.selected_id = None
#
#
#def shuffle():
#    st.session_state.seed += 1
#
#
## ───────────────────────── Cards and grids ─────────────────────────
#def render_card(row, key_prefix, explanation=None, similar_button=True):
#    """Compact movie card: poster, title, rating, and a details expander."""
#    with st.container(border=True):
#        url = poster_url(getattr(row, "poster_path", None))
#        if url:
#            st.image(url)
#        else:
#            st.markdown("🎞️ *No poster available*")
#        st.markdown(f"**{safe(row.title)}** ({year_text(row.year)})")
#        st.caption(f"⭐ {row.vote_average:.1f} · " + (", ".join(row.genres_list[:2]) or "No genre"))
#        with st.expander("Details & why" if explanation else "Details"):
#            st.caption(" · ".join([
#                ", ".join(row.genres_list) or "No genre",
#                f"Language: {row.original_language}",
#                f"Runtime: {runtime_text(row.runtime)}",
#            ]))
#            st.write(row.overview if pd.notna(row.overview) else "No overview available.")
#            if explanation:
#                st.markdown("**Why this recommendation?**")
#                for line in explanation:
#                    st.markdown(f"- {line}")
#        if similar_button:
#            st.button("Find similar", key=f"{key_prefix}_{row.id}",
#                      on_click=select_movie, args=(int(row.id),))
#
#
#def render_grid(movies, per_row, key_prefix, explain=None, similar_button=True):
#    rows = list(movies.itertuples())
#    for start in range(0, len(rows), per_row):
#        columns = st.columns(per_row)
#        for column, row in zip(columns, rows[start:start + per_row]):
#            with column:
#                explanation = explain(int(row.id)) if explain else None
#                render_card(row, key_prefix, explanation, similar_button)
#
#
#def render_recommendations(results, key_prefix, explain, similar_button=True):
#    if results.empty:
#        st.warning("No movie matches your request and filters. Try other words or relax the filters.")
#        return
#    if len(results) < MAX_RESULTS:
#        st.info(f"Only {len(results)} movie(s) match your request and filters (no filter was relaxed).")
#    render_grid(results, MAX_RESULTS, key_prefix, explain, similar_button)
#
#
#def render_selected(row):
#    """Large card for the movie the recommendations are based on."""
#    with st.container(border=True):
#        poster_col, info_col = st.columns([1, 4], gap="large")
#        with poster_col:
#            url = poster_url(getattr(row, "poster_path", None))
#            if url:
#                st.image(url)
#            else:
#                st.markdown("🎞️ *No poster available*")
#        with info_col:
#            st.subheader(f"{row.title} ({year_text(row.year)})")
#            st.caption(" · ".join([
#                ", ".join(row.genres_list) or "No genre",
#                f"Language: {row.original_language}",
#                f"Runtime: {runtime_text(row.runtime)}",
#                f"⭐ {row.vote_average:.1f}",
#            ]))
#            st.write(row.overview if pd.notna(row.overview) else "No overview available.")
#            st.button("← Back to all movies", on_click=clear_selection)
#
#
## ───────────────────────── Page ─────────────────────────
#st.title("🎬 Movie Recommender")
#st.caption("Content-Based Recommendation System using TF-IDF and Cosine Similarity.")
#
#if not DATA_PATH.exists():
#    st.error(f"Dataset not found: {DATA_PATH}. Generate it with scripts/prepare_dataset.py.")
#    st.stop()
#
#rec = load_recommender()
#movies = rec.get_movies()
#year_bounds = (int(movies["year"].min()), int(movies["year"].max()))
#
#with st.sidebar:
#    st.header("Filters (optional)")
#    genres = st.multiselect("Genres (any of)", rec.list_genres())
#    year_range = st.slider("Release year", year_bounds[0], year_bounds[1], year_bounds)
#    min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, step=0.5)
#    st.caption("Filters apply to the movies shown and to recommendations. "
#               "They are applied before picking the top 5 and never change the ranking.")
#    st.divider()
#    st.caption("The system compares words, not meanings, and is not a trained model.")
#
#filters = build_filters(genres, year_range, year_bounds, min_rating)
#
#tab_title, tab_text = st.tabs(["🔎 Discover & similar movies", "✍️ From a description"])
#
#with tab_title:
#    query = st.text_input("Search a movie by title", key="query", on_change=clear_selection,
#                          placeholder="e.g. Inception").strip()
#    selected_id = st.session_state.selected_id
#
#    if selected_id is not None:
#        chosen = next(movies[movies["id"] == selected_id].itertuples())
#        render_selected(chosen)
#        st.markdown("### Recommended for you")
#        results = rec.recommend_by_title(selected_id, n=MAX_RESULTS, filters=filters)
#        render_recommendations(results, "rec",
#                               lambda rid: title_explanation(rec.explain_by_title(selected_id, rid)))
#
#    elif query:
#        found = rec.search_titles(query, limit=SEARCH_LIMIT)
#        if found.empty:
#            st.warning(f'No movie title contains "{query}".')
#        else:
#            st.markdown(f"### Results for “{safe(query)}”")
#            if len(found) == SEARCH_LIMIT:
#                st.caption(f"Showing the first {SEARCH_LIMIT} matches. Type more of the title to narrow down.")
#            render_grid(found, GRID_COLUMNS, "search")
#
#    else:
#        left, right = st.columns([3, 2])
#        with left:
#            st.markdown("### Start with a movie")
#        with right:
#            mode = st.radio("Show", ["Most voted", "Random"], horizontal=True, label_visibility="collapsed")
#        pool = rec.filter_movies(filters)
#        if pool.empty:
#            st.warning("No movie matches the current filters.")
#        elif mode == "Most voted":
#            render_grid(pool.nlargest(HOME_SIZE, "vote_count"), GRID_COLUMNS, "home")
#        else:
#            st.button("🔀 Shuffle", on_click=shuffle)
#            sample = pool.sample(min(HOME_SIZE, len(pool)), random_state=st.session_state.seed)
#            render_grid(sample, GRID_COLUMNS, "home")
#
#with tab_text:
#    description = st.text_area(
#        "Describe the movie you want (English only)", height=120,
#        placeholder="e.g. a thief who enters people's dreams to steal secrets",
#    )
#    if st.button("Find movies", type="primary"):
#        if description.strip():
#            st.session_state.desc_query = description.strip()
#        else:
#            st.session_state.desc_query = ""
#            st.warning("Please write a description first.")
#
#    text = st.session_state.desc_query
#    if text:
#        st.markdown("### Recommended for you")
#        results = rec.recommend_by_text(text, n=MAX_RESULTS, filters=filters)
#        render_recommendations(results, "desc", lambda rid: text_explanation(rec.explain_by_text(text, rid)),
#                               similar_button=False)
#



"""Movie Recommendation System – Streamlit presentation layer.

All data loading and similarity logic live in DataLoader + MovieRecommender.
Run from project root:  streamlit run app.py
"""

import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import DataLoader
from src.recommender import MovieRecommender

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parent / "data" / "movies_filtered.csv"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"
MAX_RESULTS = 5
SEARCH_LIMIT = 20
HOME_SIZE = 12
GRID_COLUMNS = 4

EXAMPLE_PROMPTS = [
    "Mind-bending sci-fi about dreams and reality",
    "Warm animated adventure about friendship and family",
    "Slow-burn detective mystery with a plot twist",
    "Epic space survival journey with emotional depth",
]

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLING — Cozy Low-Glare Cinema Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-main: #12151c;
    --bg-sidebar: #161a23;
    --bg-card: #1b202b;
    --bg-card-hover: #212736;
    --bg-elevated: #242b3a;
    --border-soft: rgba(255, 255, 255, 0.07);
    --border-hover: rgba(226, 168, 86, 0.35);
    --text-primary: #e8ebf2;
    --text-secondary: #9aa3b5;
    --text-muted: #6f788c;
    --accent: #e2a856;
    --accent-hover: #ecc07a;
    --accent-soft: rgba(226, 168, 86, 0.14);
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: radial-gradient(circle at top right, #1a2030 0%, var(--bg-main) 52%);
    color: var(--text-primary);
}

[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border-soft);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}

/* Main container */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1220px;
}

/* Inputs & TextArea */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 14px !important;
    color: var(--text-primary) !important;
    padding: 0.7rem 1rem !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stTextInput > div > div > input::placeholder,
.stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
}

.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

/* Multiselect */
.stMultiSelect > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.52rem 1rem !important;
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-soft) !important;
    transition: all 0.22s cubic-bezier(0.22, 1, 0.36, 1) !important;
}

.stButton > button:hover {
    background: #2b3345 !important;
    border-color: rgba(255, 255, 255, 0.16) !important;
    transform: translateY(-1px);
}

.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #14171f !important;
    border: none !important;
    padding: 0.6rem 1.4rem !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--accent-hover) !important;
    box-shadow: 0 6px 20px rgba(226, 168, 86, 0.22);
}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card);
    border: 1px solid var(--border-soft) !important;
    border-radius: 18px !important;
    padding: 0.2rem !important;
    transition: transform 0.25s cubic-bezier(0.22, 1, 0.36, 1),
                border-color 0.25s ease,
                box-shadow 0.25s ease,
                background 0.25s ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-hover) !important;
    transform: translateY(-3px);
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.28);
}

/* Posters */
[data-testid="stImage"] img {
    border-radius: 12px;
    aspect-ratio: 2 / 3;
    object-fit: cover;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: transparent;
    border-bottom: 1px solid var(--border-soft);
    padding-bottom: 10px;
    margin-bottom: 1.2rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 8px 18px;
    background: transparent;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 0.92rem;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg-card);
    color: var(--text-primary);
}

.stTabs [aria-selected="true"] {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    border: 1px solid rgba(226, 168, 86, 0.3) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

/* Expander */
details {
    border: none !important;
    background: transparent !important;
}

.streamlit-expanderHeader {
    font-size: 0.84rem !important;
    color: var(--text-secondary) !important;
    padding-left: 0 !important;
}

/* Typography & Custom Blocks */
.hero-wrap {
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -0.6px;
    margin: 0 0 6px 0;
    color: var(--text-primary);
}

.hero-sub {
    color: var(--text-secondary);
    font-size: 0.98rem;
    margin: 0;
    line-height: 1.5;
}

.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 1.4rem 0 0.85rem 0;
    letter-spacing: -0.2px;
}

.card-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0.45rem 0 0.25rem 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.card-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.83rem;
    color: var(--text-secondary);
    margin-bottom: 0.45rem;
}

.rating-pill {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    background: var(--accent-soft);
    color: var(--accent);
    font-weight: 700;
    font-size: 0.78rem;
    padding: 2px 8px;
    border-radius: 999px;
}

.genre-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 0.35rem 0 0.6rem 0;
    min-height: 24px;
}

.pill {
    display: inline-block;
    background: rgba(255, 255, 255, 0.055);
    color: var(--text-secondary);
    font-size: 0.76rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.why-box {
    background: rgba(226, 168, 86, 0.07);
    border: 1px solid rgba(226, 168, 86, 0.18);
    border-radius: 12px;
    padding: 0.65rem 0.9rem;
    margin-top: 0.75rem;
    font-size: 0.85rem;
    color: #d8ccb8;
    line-height: 1.55;
}

.overview-text {
    color: #b7bfcf;
    font-size: 0.92rem;
    line-height: 1.65;
    margin: 0.5rem 0 0.3rem 0;
}

.poster-placeholder {
    aspect-ratio: 2 / 3;
    background: #232938;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 0.85rem;
    gap: 6px;
}

.stat-chip {
    display: inline-block;
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    color: var(--text-secondary);
    font-size: 0.82rem;
    padding: 5px 12px;
    border-radius: 999px;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading movies & building TF-IDF model...")
def load_recommender():
    return MovieRecommender(DataLoader(DATA_PATH)).fit()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def safe(text: str) -> str:
    return re.sub(r"([\\`*_$\[\]])", r"\\\1", str(text))


def esc(text: str) -> str:
    return html.escape(str(text))


def year_text(year) -> str:
    return "—" if pd.isna(year) else str(int(year))


def runtime_text(runtime) -> str:
    return "Unknown" if pd.isna(runtime) else f"{int(runtime)} min"


def poster_url(value) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    value = str(value).strip()
    if value.startswith("http"):
        return value
    return POSTER_BASE + (value if value.startswith("/") else "/" + value)


def title_explanation(shared: dict) -> list[str]:
    lines = []
    if shared.get("genres"):
        lines.append("Shared genres: " + ", ".join(shared["genres"]))
    if shared.get("keywords"):
        lines.append("Shared keywords: " + ", ".join(shared["keywords"]))
    if shared.get("directors"):
        lines.append("Same director: " + ", ".join(shared["directors"]))
    return lines or ["Similarity comes from overall plot and textual features."]


def text_explanation(words: list) -> list[str]:
    if words:
        return ["Matched words: " + ", ".join(words)]
    return ["Matched based on overall textual similarity."]


def build_filters(genres, year_range, year_bounds, min_rating) -> dict | None:
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


def render_poster(url: str | None):
    if url:
        st.image(url, use_container_width=True)
    else:
        st.markdown(
            '<div class="poster-placeholder">🎞️<span>No poster</span></div>',
            unsafe_allow_html=True,
        )


def genres_html(genres_list, limit: int | None = None) -> str:
    items = genres_list[:limit] if limit else genres_list
    if not items:
        return '<div class="genre-row"><span class="pill">Uncategorized</span></div>'
    pills = "".join(f'<span class="pill">{esc(g)}</span>' for g in items)
    return f'<div class="genre-row">{pills}</div>'


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
st.session_state.setdefault("selected_id", None)
st.session_state.setdefault("desc_query", "")
st.session_state.setdefault("desc_input", "")
st.session_state.setdefault("seed", 0)


def select_movie(movie_id: int):
    st.session_state.selected_id = movie_id


def clear_selection():
    st.session_state.selected_id = None


def shuffle():
    st.session_state.seed += 1


def apply_example(prompt_text: str):
    st.session_state.desc_input = prompt_text
    st.session_state.desc_query = prompt_text


# ─────────────────────────────────────────────
# CARD COMPONENTS
# ─────────────────────────────────────────────
def render_browse_card(row, key_prefix: str):
    """Clean, uniform vertical card for browsing and search grids."""
    with st.container(border=True):
        render_poster(poster_url(getattr(row, "poster_path", None)))

        title_clean = esc(row.title)
        st.markdown(
            f'<div class="card-title" title="{title_clean}">{title_clean}</div>'
            f'<div class="card-meta">'
            f'<span class="rating-pill">★ {row.vote_average:.1f}</span>'
            f'<span>{year_text(row.year)}</span>'
            f'<span>·</span>'
            f'<span>{runtime_text(row.runtime)}</span>'
            f'</div>'
            f'{genres_html(row.genres_list, limit=2)}',
            unsafe_allow_html=True,
        )

        with st.expander("Overview"):
            overview = row.overview if pd.notna(row.overview) else "No overview available."
            st.markdown(f'<p class="overview-text">{esc(overview)}</p>', unsafe_allow_html=True)
            st.caption(f"Language: {str(row.original_language).upper()}")

        st.button(
            "Find similar",
            key=f"{key_prefix}_{row.id}",
            on_click=select_movie,
            args=(int(row.id),),
            use_container_width=True,
        )


def render_grid(movies, per_row: int, key_prefix: str):
    rows = list(movies.itertuples())
    for start in range(0, len(rows), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, row in zip(cols, rows[start : start + per_row]):
            with col:
                render_browse_card(row, key_prefix)


def render_recommendation_card(row, rank: int, key_prefix: str, explanation: list[str], similar_button: bool = True):
    """Spacious horizontal card so recommendations are effortless to read."""
    with st.container(border=True):
        poster_col, info_col = st.columns([1, 4.5], gap="large")

        with poster_col:
            render_poster(poster_url(getattr(row, "poster_path", None)))

        with info_col:
            header_left, header_right = st.columns([4, 1.2])
            with header_left:
                st.markdown(
                    f'<div style="font-size:1.18rem;font-weight:700;color:#e8ebf2;margin-bottom:4px;">'
                    f'<span style="color:#e2a856;margin-right:8px;">#{rank}</span>'
                    f'{esc(row.title)} <span style="color:#9aa3b5;font-weight:500;">({year_text(row.year)})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with header_right:
                if similar_button:
                    st.button(
                        "Similar →",
                        key=f"{key_prefix}_{row.id}",
                        on_click=select_movie,
                        args=(int(row.id),),
                        use_container_width=True,
                    )

            st.markdown(
                f'<div class="card-meta">'
                f'<span class="rating-pill">★ {row.vote_average:.1f}</span>'
                f'<span>{runtime_text(row.runtime)}</span>'
                f'<span>·</span>'
                f'<span>Lang: {esc(str(row.original_language).upper())}</span>'
                f'</div>'
                f'{genres_html(row.genres_list)}',
                unsafe_allow_html=True,
            )

            overview = row.overview if pd.notna(row.overview) else "No overview available."
            st.markdown(f'<p class="overview-text">{esc(overview)}</p>', unsafe_allow_html=True)

            if explanation:
                reasons_html = "<br>".join(f"• {esc(line)}" for line in explanation)
                st.markdown(
                    f'<div class="why-box"><strong>Why this match:</strong><br>{reasons_html}</div>',
                    unsafe_allow_html=True,
                )


def render_recommendations(results, key_prefix: str, explain, similar_button: bool = True):
    if results.empty:
        st.warning("No movies match your request and current filters. Try relaxing the sidebar filters.")
        return
    if len(results) < MAX_RESULTS:
        st.info(f"Showing {len(results)} matching movie(s) under your current filters.")

    for idx, row in enumerate(results.itertuples(), start=1):
        explanation = explain(int(row.id)) if explain else []
        render_recommendation_card(row, idx, key_prefix, explanation, similar_button)


def render_selected(row):
    """Featured spotlight card for the currently selected movie."""
    st.button("← Back to browse", on_click=clear_selection)

    with st.container(border=True):
        poster_col, info_col = st.columns([1.1, 3.8], gap="large")

        with poster_col:
            render_poster(poster_url(getattr(row, "poster_path", None)))

        with info_col:
            st.markdown(
                f'<div style="font-size:1.55rem;font-weight:800;letter-spacing:-0.4px;margin-bottom:6px;">'
                f'{esc(row.title)} <span style="color:#9aa3b5;font-weight:500;">({year_text(row.year)})</span>'
                f'</div>'
                f'<div class="card-meta" style="margin-bottom:0.6rem;">'
                f'<span class="rating-pill">★ {row.vote_average:.1f}</span>'
                f'<span>{runtime_text(row.runtime)}</span>'
                f'<span>·</span>'
                f'<span>Language: {esc(str(row.original_language).upper())}</span>'
                f'</div>'
                f'{genres_html(row.genres_list)}',
                unsafe_allow_html=True,
            )
            overview = row.overview if pd.notna(row.overview) else "No overview available."
            st.markdown(f'<p class="overview-text">{esc(overview)}</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
st.markdown(
    '<div class="hero-wrap">'
    '<p class="hero-title">🎬 CineMatch</p>'
    '<p class="hero-sub">Relaxed movie discovery powered by TF-IDF &amp; Cosine Similarity</p>'
    '</div>',
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: `{DATA_PATH}`\n\nGenerate it with `scripts/prepare_dataset.py`.")
    st.stop()

rec = load_recommender()
movies = rec.get_movies()
year_bounds = (int(movies["year"].min()), int(movies["year"].max()))

# Initialize filter defaults cleanly
st.session_state.setdefault("filter_genres", [])
st.session_state.setdefault("filter_years", year_bounds)
st.session_state.setdefault("filter_rating", 0.0)


def reset_filters():
    st.session_state.filter_genres = []
    st.session_state.filter_years = year_bounds
    st.session_state.filter_rating = 0.0


# ── Sidebar filters ──
with st.sidebar:
    st.markdown("### Filters")
    st.caption("Refine both browse results and recommendations.")

    genres = st.multiselect("Genres", rec.list_genres(), key="filter_genres")
    year_range = st.slider("Release year", year_bounds[0], year_bounds[1], key="filter_years")
    min_rating = st.slider("Minimum rating", 0.0, 10.0, step=0.5, key="filter_rating")

    filters = build_filters(genres, year_range, year_bounds, min_rating)
    filtered_pool = rec.filter_movies(filters)

    st.markdown(
        f'<div class="stat-chip">🎞️ {len(filtered_pool):,} of {len(movies):,} movies available</div>',
        unsafe_allow_html=True,
    )

    if filters:
        st.button("Reset filters", on_click=reset_filters, use_container_width=True)

    st.divider()
    st.caption("Matches shared words, genres, keywords, and directors using TF-IDF.")

# ── Tabs ──
tab_discover, tab_describe = st.tabs(["🔎 Discover & Similar", "✍️ Describe a Movie"])

# ═════════════════════════════════════════════
# TAB 1 – Discover & Similar
# ═════════════════════════════════════════════
with tab_discover:
    query = st.text_input(
        "Search by title",
        key="query",
        placeholder="Search for a movie… (e.g. Inception, Interstellar, Spirited Away)",
        on_change=clear_selection,
        label_visibility="collapsed",
    ).strip()

    selected_id = st.session_state.selected_id

    # ── Selected movie + recommendations ──
    if selected_id is not None:
        chosen_df = movies[movies["id"] == selected_id]
        if chosen_df.empty:
            clear_selection()
            st.rerun()

        chosen = next(chosen_df.itertuples())
        render_selected(chosen)

        st.markdown('<p class="section-header">Recommended for you</p>', unsafe_allow_html=True)
        results = rec.recommend_by_title(selected_id, n=MAX_RESULTS, filters=filters)
        render_recommendations(
            results,
            "rec",
            lambda rid: title_explanation(rec.explain_by_title(selected_id, rid)),
        )

    # ── Search results ──
    elif query:
        found = rec.search_titles(query, limit=SEARCH_LIMIT)
        if found.empty:
            st.warning(f'No title contains “{query}”.')
        else:
            st.markdown(
                f'<p class="section-header">Results for “{esc(query)}”</p>',
                unsafe_allow_html=True,
            )
            if len(found) == SEARCH_LIMIT:
                st.caption(f"Showing the first {SEARCH_LIMIT} matches. Type a bit more to narrow down.")
            render_grid(found, GRID_COLUMNS, "search")

    # ── Home / browse ──
    else:
        left, right = st.columns([3, 1.3], vertical_alignment="center")
        with left:
            st.markdown('<p class="section-header" style="margin-top:0.6rem;">Start exploring</p>', unsafe_allow_html=True)
        with right:
            mode = st.radio(
                "Show",
                ["Most voted", "Random"],
                horizontal=True,
                label_visibility="collapsed",
            )

        if filtered_pool.empty:
            st.warning("No movies match the current filters.")
        elif mode == "Most voted":
            render_grid(filtered_pool.nlargest(HOME_SIZE, "vote_count"), GRID_COLUMNS, "home")
        else:
            st.button("🔀 Shuffle selection", on_click=shuffle)
            sample = filtered_pool.sample(min(HOME_SIZE, len(filtered_pool)), random_state=st.session_state.seed)
            render_grid(sample, GRID_COLUMNS, "home")

# ═════════════════════════════════════════════
# TAB 2 – From a description
# ═════════════════════════════════════════════
with tab_describe:
    st.caption("Try an idea or write your own mood, plot, or keywords:")
    ex_cols = st.columns(len(EXAMPLE_PROMPTS))
    for i, (col, prompt) in enumerate(zip(ex_cols, EXAMPLE_PROMPTS)):
        with col:
            st.button(
                prompt[:26] + "…",
                key=f"ex_{i}",
                on_click=apply_example,
                args=(prompt,),
                use_container_width=True,
                help=prompt,
            )

    description = st.text_area(
        "Describe the kind of movie you want",
        key="desc_input",
        height=120,
        placeholder="Example: a thief who enters people's dreams to steal secrets, mind-bending, sci-fi",
        label_visibility="collapsed",
    )

    if st.button("Find movies", type="primary"):
        if description.strip():
            st.session_state.desc_query = description.strip()
        else:
            st.session_state.desc_query = ""
            st.warning("Please write a short description first.")

    text = st.session_state.desc_query
    if text:
        st.markdown('<p class="section-header">Best matches for your description</p>', unsafe_allow_html=True)
        results = rec.recommend_by_text(text, n=MAX_RESULTS, filters=filters)
        render_recommendations(
            results,
            "desc",
            lambda rid: text_explanation(rec.explain_by_text(text, rid)),
            similar_button=False,
        )