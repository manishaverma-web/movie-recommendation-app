# Movie Recommendation App

A movie recommendation web application built using Python, Flask, Pandas, NumPy and Machine Learning.

## Features

- Search for a movie by title
- Get similar movie recommendations
- Content-based recommendation system
- Simple and user-friendly web interface

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- HTML
- CSS

## How It Works

The application uses a content-based recommendation approach. Movie information is converted into numerical features using `CountVectorizer`, and `Cosine Similarity` is used to find movies that are similar to the selected movie.

## Dataset

The project uses the TMDB 5000 Movies dataset.

## Project Structure

```text
movie-recommendation-app/
│
├── app.py
├── movie_recommender.py
├── tmdb_5000_movies.csv
└── templates/
    └── index.html
