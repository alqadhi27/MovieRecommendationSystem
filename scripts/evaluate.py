"""Measures loading and response times, and prints a qualitative sample of recommendations.

Run from the project root:  python scripts/evaluate.py
Results are printed and saved to docs/evaluation_results.txt (for the report).
"""
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import DataLoader  # noqa: E402
from src.recommender import MovieRecommender  # noqa: E402

DATA_PATH = ROOT / "data" / "movies_filtered.csv"
OUTPUT_PATH = ROOT / "docs" / "evaluation_results.txt"
REPEATS = 20

KNOWN_TITLES = ["Inception", "Toy Story", "The Dark Knight", "Titanic", "The Matrix", "Frozen"]
DESCRIPTIONS = [
    "A movie about space exploration and survival.",
    "A detective investigates a murder in a small town.",
    "A funny animated adventure for the whole family.",
]

lines = []


def emit(text=""):
    print(text)
    lines.append(text)


def timed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def repeat_ms(function):
    """Average and worst time in milliseconds over REPEATS calls."""
    durations = [timed(function)[1] * 1000 for _ in range(REPEATS)]
    return statistics.mean(durations), max(durations)


def show_title_case(rec, title):
    movies = rec.get_movies()
    matches = movies[movies["title"].str.lower() == title.lower()]
    if matches.empty:
        emit(f"- {title}: not found in the dataset")
        return
    chosen = matches.sort_values("vote_count", ascending=False).iloc[0]
    emit(f"- {chosen['title']} ({chosen['year']}) | {', '.join(chosen['genres_list'])}")
    for row in rec.recommend_by_title(int(chosen["id"])).itertuples():
        shared = rec.explain_by_title(int(chosen["id"]), int(row.id))
        common = "; ".join(f"{key}: {', '.join(values)}" for key, values in shared.items() if values)
        emit(f"    -> {row.title} ({row.year}) | {', '.join(row.genres_list)} | shared: {common or 'nothing'}")


def show_text_case(rec, text):
    emit(f'- "{text}"')
    results = rec.recommend_by_text(text)
    if results.empty:
        emit("    -> no result")
    for row in results.itertuples():
        words = rec.explain_by_text(text, int(row.id))
        emit(f"    -> {row.title} ({row.year}) | {', '.join(row.genres_list)} | words: {', '.join(words)}")


def main(titles=KNOWN_TITLES, descriptions=DESCRIPTIONS):
    emit("=== PERFORMANCE ===")
    _, prepare_time = timed(lambda: DataLoader(DATA_PATH).prepare())
    rec, fit_time = timed(lambda: MovieRecommender(DataLoader(DATA_PATH)).fit())
    emit(f"Movies: {len(rec.get_movies())} | TF-IDF matrix: {rec.matrix.shape[0]} x {rec.matrix.shape[1]}")
    emit(f"DataLoader.prepare(): {prepare_time:.2f} s")
    emit(f"Full startup (prepare + TF-IDF fit): {fit_time:.2f} s")

    movie_id = int(rec.get_movies().iloc[0]["id"])
    filters = {"genres": [rec.list_genres()[0]], "min_rating": 6.0}
    for label, function in [
        ("search_titles('the')", lambda: rec.search_titles("the")),
        ("recommend_by_title", lambda: rec.recommend_by_title(movie_id)),
        ("recommend_by_title + filters", lambda: rec.recommend_by_title(movie_id, filters=filters)),
        ("recommend_by_text", lambda: rec.recommend_by_text(descriptions[0])),
    ]:
        average, worst = repeat_ms(function)
        emit(f"{label}: average {average:.1f} ms, worst {worst:.1f} ms ({REPEATS} runs)")

    emit()
    emit("=== QUALITATIVE SAMPLE: by title ===")
    for title in titles:
        show_title_case(rec, title)

    emit()
    emit("=== QUALITATIVE SAMPLE: by description ===")
    for text in descriptions:
        show_text_case(rec, text)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
