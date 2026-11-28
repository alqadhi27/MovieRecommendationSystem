# Movie Recommendation System

A content-based movie recommendation application built with Python and Streamlit, using TF-IDF vectorization and cosine similarity.

## Project Overview

This is a small individual academic project developed as part of a university Software Engineering course. It helps users discover movies based on their content and metadata.

## Features

- **Recommendations by movie:** Search for a movie by title and receive up to five similar movies.
- **Recommendations by description:** Enter an English description to find movies with related content.
- **Optional filters:** Filter recommendations by genre, release year, and minimum rating.
- **Recommendation explanations:** View shared genres, keywords, directors, or matching words from the entered description.
- **Movie details:** Display movie overviews, metadata, ratings, and poster images when available.

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Pytest

## Dataset

The project uses movie metadata including:

- Movie title and overview
- Genres and keywords
- Directors
- Release date
- Original language
- Runtime
- Rating and vote count
- Poster path (for displaying movie posters)

The application loads the prepared dataset from:

`data/movies_filtered.csv`

## Recommendation Method

The system uses **Content-Based Filtering**:

1. Movie metadata (overview, genres, keywords, and directors) is combined into text features.
2. TF-IDF converts the text features into numerical vectors.
3. Cosine similarity measures the similarity between movies or between a user description and the movie dataset.
4. The system ranks eligible candidates and returns up to five recommendations.

Filters are applied to the candidate movies before selecting the top results. They do not modify the similarity scores.

## Project Structure

```text
MovieRecommendationSystem/
├── app.py
├── requirements.txt
├── data/
│   └── movies_filtered.csv
├── scripts/
│   └── prepare_dataset.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   └── recommender.py
├── tests/
│   ├── Conftest.py
│   ├── test_data_loader.py
│   └── test_recommender.py
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/alqadhi27/MovieRecommendationSystem.git
cd MovieRecommendationSystem
```

Create and activate a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

From the project root:

```bash
python -m streamlit run app.py
```

Open the local URL provided by Streamlit, usually:

`http://localhost:8501`

## Run Tests

Run the test suite from the project root:

```bash
python -m pytest
```

## Project Status

The coding and testing phases have been completed for the current academic project version.

## Academic Context

Developed as an individual academic project for a university Software Engineering course.
