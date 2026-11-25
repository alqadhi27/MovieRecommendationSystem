# Movie Recommendation System

## Project Overview

A small academic project developed using Python and Streamlit to recommend movies based on their content and metadata.

## Objective

Implement a Content-Based Filtering recommendation system to help users discover movies similar to a selected movie.

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

The project uses a movie dataset containing:

- Movie title
- Overview
- Genres
- Keywords
- Directors
- Release date
- Original language
- Runtime
- Rating and vote count

## Recommendation Method

The recommendation system uses:

1. TF-IDF to convert movie metadata into numerical vectors.
2. Cosine Similarity to measure similarity between movies.
3. Similarity scores to identify relevant recommendations.

## Project Structure

    MovieRecommendationSystem/
    ├── app.py
    ├── requirements.txt
    ├── data/
    ├── scripts/
    │   └── prepare_dataset.py
    ├── src/
    │   ├── __init__.py
    │   ├── data_loader.py
    │   └── recommender.py
    └── README.md

## Installation

Clone the repository:

    git clone https://github.com/alqadhi27/MovieRecommendationSystem.git
    cd MovieRecommendationSystem

Install dependencies:

    pip install -r requirements.txt

## Run the Application

    streamlit run app.py

## Project Status

Under development.

## Academic Project

Developed as part of a university Software Engineering course.

