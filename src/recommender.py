import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, loader):
        self.loader = loader
        self.movies = None
        self.vectorizer = None
        self.matrix = None

    def fit(self):
        self.movies = self.loader.prepare()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.movies["text"])
        return self

    def get_movies(self):
        return self.movies

    def list_genres(self):
        return sorted({genre for genres in self.movies["genres_list"] for genre in genres})

    def search_titles(self, query, limit=20):
        mask = self.movies["title"].str.contains(query, case=False, regex=False)
        return self.movies[mask].head(limit)

    def _top_n(self, scores, n, candidates, exclude=None):
        ranked = pd.Series(scores, index=self.movies.index).loc[candidates]
        if exclude is not None:
            ranked = ranked.drop(exclude, errors="ignore")
        ranked = ranked[ranked > 0].sort_values(ascending=False, kind="stable")
        return self.movies.loc[ranked.head(n).index]

    def recommend_by_title(self, movie_id, n=5, filters=None):
        position = self.movies.index[self.movies["id"] == movie_id][0]
        scores = cosine_similarity(self.matrix[position], self.matrix).ravel()
        candidates = self.filter_movies(filters).index
        return self._top_n(scores, n, candidates, exclude=position)

    def filter_movies(self, filters=None):
        movies = self.movies
        if not filters:
            return movies
        mask = pd.Series(True, index=movies.index)
        if filters.get("genres"):
            wanted = set(filters["genres"])
            mask &= movies["genres_list"].apply(lambda g: bool(wanted & set(g)))
        if filters.get("year_min") is not None:
            mask &= movies["year"] >= filters["year_min"]
        if filters.get("year_max") is not None:
            mask &= movies["year"] <= filters["year_max"]
        if filters.get("min_rating") is not None:
            mask &= movies["vote_average"] >= filters["min_rating"]
        return movies[mask]

    def recommend_by_text(self, text, n=5, filters=None):
        vector = self.vectorizer.transform([text])
        scores = cosine_similarity(vector, self.matrix).ravel()
        candidates = self.filter_movies(filters).index
        return self._top_n(scores, n, candidates)

    def explain_by_title(self, movie_id, recommended_id):
        a = self.movies[self.movies["id"] == movie_id].iloc[0]
        b = self.movies[self.movies["id"] == recommended_id].iloc[0]
        return {
            "genres": [x for x in a["genres_list"] if x in b["genres_list"]],
            "keywords": [x for x in a["keywords_list"] if x in b["keywords_list"]],
            "directors": [x for x in a["directors_list"] if x in b["directors_list"]],
        }

    def explain_by_text(self, text, recommended_id):
        movie = self.movies[self.movies["id"] == recommended_id].iloc[0]
        analyzer = self.vectorizer.build_analyzer()
        movie_words = set(analyzer(movie["text"]))
        return [word for word in dict.fromkeys(analyzer(text)) if word in movie_words]
