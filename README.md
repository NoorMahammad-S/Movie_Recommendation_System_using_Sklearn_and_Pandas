# Movie_Recommendation_System_using_Sklearn_and_Pandas
A movie recommendation system using data science techniques implemented in Python using collaborative filtering and cosine similarity. It uses the MovieLens dataset, but you can replace it with your own dataset.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NoorMahammad-S/movie-recommendation-system.git
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   Download the MovieLens dataset from [here](https://grouplens.org/datasets/movielens/) and place the `movies.csv` and `ratings.csv` files in the project root directory.

4. **Run the recommendation script:**
   ```bash
   python movie_recommendation.py
   ```

## How it Works

- The code uses collaborative filtering to recommend movies based on user ratings.
- The user-item matrix is created from the MovieLens dataset.
- Cosine similarity is used to build a movie similarity matrix.
- Given a movie title and user ratings, the script provides movie recommendations.

## Example

```python
# Example: Get recommendations for a user who rated 'The Dark Knight' highly
user_ratings = user_movie_ratings.loc[1].fillna(0)  # Replace NaN with 0
user_ratings['The Dark Knight'] = 5.0  # User rates 'The Dark Knight' as 5.0

# Get movie recommendations
recommendations = get_movie_recommendations('The Dark Knight', user_ratings)

print("Top 5 movie recommendations:")
print(recommendations.head(5))
```

## Contributing

If you have suggestions or find issues, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
