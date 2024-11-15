# Movie-Recommender-
This repository contains the code for a movie recommender system. The system recommends movies based on their similarity to a given movie. This is achieved using a combination of natural language processing and cosine similarity.
## Getting Started
These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites
Ensure you have the following libraries installed:
* pandas
* numpy
* scikit-learn
* nltk

You can install the required libraries using pip:
```
pip install pandas numpy scikit-learn nltk
```
### Installing
Clone the repository to your local machine:
```
git clone https://github.com/yourusername/movierecommender.git
cd movierecommender
```
## Data Preparation
The movie dataset is assumed to be preprocessed and consists of the following columns:

* movie_id: Unique identifier for the movie
* title: The title of the movie
* genres: List of genres associated with the movie
* cast: List of main cast members
* crew: List of main crew members
* overview: Short description of the movie
* keywords: List of keywords associated with the movie

These columns are combined into a single tags column which is used to calculate the similarity between movies.
```
movies['tags'] = movies['genres'] + movies['cast'] + movies['crew'] + movies['overview'] + movies['keywords']
```
## Feature Extraction
Natural language processing techniques are used to process the text data:

* Tokenization: Splitting text into words.
* Stemming: Reducing words to their root form.

A CountVectorizer is used to convert the text data into vectors.
```
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
```
## Similarity Calculation
The similarity between movies is calculated using cosine similarity.
```
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
```
## Recommendation Function
The recommendation function takes a movie title as input and returns the top 5 most similar movies.
```
def recommend(movie):
    mindex = movies[movies['title'] == movie].index[0]
    distance = similarity[mindex]
    mlist = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
    for i in mlist:
        print(movies.iloc[i[0]].title)
```
## Example
```
recommend('Batman Begins')
# Output:
# The Dark Knight
# Batman
# Batman
# The Dark Knight Rises
# 10th & Wolf
```

![WhatsApp Image 2024-11-15 at 08 51 42_06270dd4](https://github.com/user-attachments/assets/0a12b392-8f27-4d50-b947-7270a76844f7)

## Running the Tests
To test the recommendation system, you can call the recommend function with a movie title and verify the output.
