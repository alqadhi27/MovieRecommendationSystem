"""Shared fixtures: a tiny movie CSV built on the fly, so tests never touch the real dataset."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # lets `from src...` work with plain `pytest` too

from src.data_loader import DataLoader  # noqa: E402
from src.recommender import MovieRecommender  # noqa: E402

COLUMNS = [
    "id", "title", "release_date", "overview", "genres", "keywords",
    "directors", "original_language", "runtime", "vote_average", "vote_count",
]

# Edge cases on purpose: runtime 0 (id 6), missing keywords (id 6), missing
# directors (id 7), duplicated title "Echo" (ids 4 and 5), a two-word genre,
# and one director shared by several movies.
MOVIES = [
    (1, "Dream Thief", "2010-07-16", "A thief enters dreams to steal secrets from a sleeping mind.",
     "Action, Science Fiction", "dream, heist", "Christopher Nolan", "en", 148, 8.4, 5000),
    (2, "Space Voyage", "2014-11-05", "Astronauts travel through a wormhole to save humanity.",
     "Science Fiction, Drama", "space, wormhole", "Christopher Nolan", "en", 169, 8.6, 4000),
    (3, "Dream Heist Two", "2015-05-01", "A crew plans another heist inside a dream.",
     "Action, Thriller", "heist, dream", "Jane Roe", "en", 120, 7.0, 900),
    (4, "Echo", "1999-03-10", "A sound travels across the valley and returns changed.",
     "Drama, Mystery", "sound, valley", "Ana Lopez", "fr", 95, 6.8, 300),
    (5, "Echo", "2018-10-26", "A young woman hears a mysterious echo in an empty house.",
     "Mystery, Horror", "house, echo", "Mike Doe", "en", 101, 6.1, 450),
    (6, "Quiet Garden", "1985-06-01", "An old gardener keeps a peaceful garden alone.",
     "Drama", None, "Ana Lopez", "en", 0, 6.0, 120),
    (7, "Silent Film", "1920-03-01", "A clown struggles in a world without sound.",
     "Comedy", "silent", None, "en", 70, 7.5, 210),
    (8, "Romantic Paris", "2005-02-14", "Two strangers fall in love in Paris.",
     "Romance, Comedy", "paris, love", "Jane Roe", "fr", 110, 6.9, 800),
    (9, "Space Pirates", "2020-08-08", "Pirates chase treasure across the galaxy.",
     "Science Fiction, Adventure", "space, pirates", "Mike Doe", "en", 130, 6.5, 700),
    (10, "Haunted House", "2001-10-31", "A family discovers ghosts living in their house.",
     "Horror", "ghost, house", "Jane Roe", "en", 98, 5.5, 650),
]


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "movies_sample.csv"
    pd.DataFrame(MOVIES, columns=COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def loader(csv_path):
    return DataLoader(csv_path)


@pytest.fixture(scope="session")
def prepared(loader):
    return loader.prepare()


@pytest.fixture(scope="session")
def recommender(csv_path):
    return MovieRecommender(DataLoader(csv_path)).fit()
