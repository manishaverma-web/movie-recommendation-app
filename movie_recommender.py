import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("tmdb_5000_movies.csv")

# Sirf zaroori columns
movies = movies[['title', 'overview']]

# Empty overview remove
movies.dropna(inplace=True)

# Text ko numbers/vectors me convert karna
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['overview']).toarray()

# Similarity calculate
similarity = cosine_similarity(vectors)

print("Movie Recommendation System Ready!")
print(movies[['title']].head())


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\nRecommended Movies:")

    for i in movie_list:
        print(movies.iloc[i[0]].title)


movie_name = input("Enter a movie name: ")
recommend(movie_name)