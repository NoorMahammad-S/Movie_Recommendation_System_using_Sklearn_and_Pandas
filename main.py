import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load the MovieLens dataset (you can replace this with your own dataset)
# Download the dataset from: https://grouplens.org/datasets/movielens/
movies = pd.read_csv('db/movies.csv')
ratings = pd.read_csv('db/ratings.csv')

# Merge the movies and ratings dataframes
df = pd.merge(ratings, movies, on='movieId')

# Create a user-item matrix
user_movie_ratings = df.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values with 0 (assuming no rating means a rating of 0)
user_movie_ratings = user_movie_ratings.fillna(0)

# Transpose the matrix to have movies as rows and users as columns
movie_user_ratings = user_movie_ratings.transpose()

# Build a movie similarity matrix using cosine similarity
movie_similarity = cosine_similarity(movie_user_ratings)

# Convert the similarity matrix into a Pandas DataFrame
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

# Function to get movie recommendations based on user ratings
def get_movie_recommendations(movie_title, user_ratings):
    similar_scores = movie_similarity_df[movie_title] * (user_ratings - 2.5)
    similar_scores = similar_scores.sort_values(ascending=False)
    return similar_scores

# Example: Get recommendations for a user who rated 'The Dark Knight' highly
user_ratings = user_movie_ratings.loc[1].fillna(0)  # Replace NaN with 0
user_ratings['The Dark Knight'] = 5.0  # User rates 'The Dark Knight' as 5.0

# Get movie recommendations
recommendations = get_movie_recommendations('The Dark Knight', user_ratings)

print("Top 5 movie recommendations:")
print(recommendations.head(5))
