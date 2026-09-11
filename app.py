from flask import Flask, render_template, request
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# =========================
# OMDb API KEY
# =========================
API_KEY = "36a47de5"


# =========================
# LOAD 5000 MOVIES DATASET
# =========================
movies = pd.read_csv("tmdb_5000_movies.csv")

movies = movies[['title', 'overview', 'popularity']]
movies.dropna(inplace=True)

# Convert movie descriptions into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['overview']).toarray()

similarity = cosine_similarity(vectors)


# =========================
# DATASET RECOMMENDATIONS
# =========================
def recommend(movie):

    movie = movie.lower()

    matches = movies[
        movies['title'].str.lower().str.contains(movie, na=False)
    ]

    if matches.empty:
    # Fallback recommendations
      return movies['title'].head(5).tolist()

    movie_index = matches.index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movie_list:
        recommendations.append(movies.iloc[i[0]]['title'])

    return recommendations


# =========================
# OMDb MOVIE SEARCH
# =========================
def search_omdb(movie_name):

    url = "https://www.omdbapi.com/"

    # पहले exact title search
    params = {
        "apikey": API_KEY,
        "t": movie_name
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("Response") == "True":
        return data, []

    # Exact movie नहीं मिली तो similar movies search
    params = {
        "apikey": API_KEY,
        "s": movie_name
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("Response") == "True":
        return None, data.get("Search", [])

    return None, []


# =========================
# HOME PAGE
# =========================
@app.route('/', methods=['GET', 'POST'])
def home():

    recommendations = []
    movie_name = ""
    movie_info = None
    search_results = []
    message = ""

    if request.method == 'POST':

        movie_name = request.form.get('movie_name', '').strip()

        if movie_name:

            # OMDb search
            movie_info, search_results = search_omdb(movie_name)

            # Dataset recommendations
            recommendations = recommend(movie_name)

            # कहीं भी movie नहीं मिली
            if not movie_info and not search_results and not recommendations:
                message = "Movie not found. Please check the spelling."

    return render_template(
        'index.html',
        recommendations=recommendations,
        movie_name=movie_name,
        movie_info=movie_info,
        search_results=search_results,
        message=message
    )


if __name__ == '__main__':
    app.run(debug=True)