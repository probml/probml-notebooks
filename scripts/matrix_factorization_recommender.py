# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/Nirzu97/pyprobml/blob/matrix-factorization/notebooks/matrix_factorization_recommender.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="NTsyA3nxgIhT"
# # Matrix Factorization for Movie Lens Recommendations 
#
# This notebook is based on  code from Nick Becker 
#
# https://github.com/beckernick/matrix_factorization_recommenders/blob/master/matrix_factorization_recommender.ipynb
#
#
#
#
#
#

# + [markdown] id="nf5GiG3YgIhd"
# # Setting Up the Ratings Data
#
# We read the data directly from MovieLens website, since they don't allow redistribution. We want to include the metadata (movie titles, etc), not just the ratings matrix.
#

# + id="aH_UwaAsh1LP"
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt


# + colab={"base_uri": "https://localhost:8080/"} id="0Pa5k76tYztd" outputId="15e6753a-4b12-4459-fcbe-87262c71c2b7"
# !wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
# !ls
# !unzip ml-100k
folder = 'ml-100k'

# + colab={"base_uri": "https://localhost:8080/"} id="THfvnkzah3nv" outputId="e7310704-fb63-49bd-cf4b-ab568d65532e"
# !wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
# !unzip ml-1m
# !ls
folder = 'ml-1m'


# + id="J_zij7tJgIhd"

ratings_list = [ [int(x) for x in i.strip().split("::")] for i in open(os.path.join(folder,'ratings.dat'), 'r').readlines()]
users_list = [i.strip().split("::") for i in open(os.path.join(folder, 'users.dat'), 'r').readlines()]
movies_list = [i.strip().split("::") for i in open(os.path.join(folder, 'movies.dat'), 'r',  encoding="latin-1").readlines()]



# + id="R8JnjoDVgIhe"
ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

# + id="L06ZLb4CgIhf" colab={"base_uri": "https://localhost:8080/", "height": 196} outputId="21273c16-64a4-4ef7-ae6b-bc2544284b6c"
movies_df.head()


# + colab={"base_uri": "https://localhost:8080/"} id="Tv9rqPfoxvXo" outputId="70b15305-7226-4008-e4f8-13e0c06117e0"
def get_movie_name(movies_df, movie_id_str):
  ndx = (movies_df['MovieID']==int(movie_id_str))
  name = movies_df['Title'][ndx].to_numpy()[0]
  return name

print(get_movie_name(movies_df, 1))
print(get_movie_name(movies_df, "527"))


# + colab={"base_uri": "https://localhost:8080/"} id="mrqetJo14NEe" outputId="f7276962-607a-47c0-a6d2-4322a4dab187"
def get_movie_genres(movies_df, movie_id_str):
  ndx = (movies_df['MovieID']==int(movie_id_str))
  name = movies_df['Genres'][ndx].to_numpy()[0]
  return name

print(get_movie_genres(movies_df, 1))
print(get_movie_genres(movies_df, "527"))

# + id="a3fua44igIhg" colab={"base_uri": "https://localhost:8080/", "height": 196} outputId="ee59b580-a2fd-4917-d7fa-93c70b2d71af"
ratings_df.head()

# + [markdown] id="Qmf6YmHEgIhh"
# These look good, but I want the format of my ratings matrix to be one row per user and one column per movie. I'll `pivot` `ratings_df` to get that and call the new variable `R`.

# + id="Jmysfzc4gIhh" colab={"base_uri": "https://localhost:8080/", "height": 275} outputId="600d38df-73df-4fbb-db65-b2cfcd2d62f1"
R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
R_df.head()

# + [markdown] id="h_4z9YWTgIhh"
# The last thing I need to do is de-mean the data (normalize by each users mean) and convert it from a dataframe to a numpy array.

# + id="k3GGGqwAgIhi" colab={"base_uri": "https://localhost:8080/"} outputId="7d350a7c-0d61-432c-fdfc-2db708b046eb"
R = R_df.to_numpy()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

print(R.shape)
print(np.count_nonzero(R))

# + [markdown] id="ktEjpdh2gIhi"
# # Singular Value Decomposition
#
# Scipy and Numpy both have functions to do the singular value decomposition. I'm going to use the Scipy function `svds` because it let's me choose how many latent factors I want to use to approximate the original ratings matrix (instead of having to truncate it after).

# + id="DMFgd5IIgIhi"
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)
sigma = np.diag(sigma)

# + id="arTEARPGgIhj" colab={"base_uri": "https://localhost:8080/"} outputId="6576c695-c993-4843-8dfd-2b429e3d66b4"
latents = [10, 20, 50]
errors = []
for latent_dim in latents:
  U, sigma, Vt = svds(R_demeaned, k = latent_dim)
  sigma = np.diag(sigma)
  Rpred = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
  Rpred[Rpred < 0] = 0
  Rpred[Rpred > 5] = 5
  err = (np.sqrt(np.sum(np.power(R - Rpred, 2))))
  errors.append(err)

print(errors)


# + [markdown] id="bhBscFmXgIhk"
# # Making Predictions from the Decomposed Matrices
#
# I now have everything I need to make movie ratings predictions for every user. I can do it all at once by following the math and matrix multiply $U$, $\Sigma$, and $V^{T}$ back to get the rank $k=50$ approximation of $R$.
#
# I also need to add the user means back to get the actual star ratings prediction.

# + id="gQyqTbUCgIhk"
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

# + [markdown] id="1bZkdk_GgIhk"
# # Making Movie Recommendations
# Finally, it's time. With the predictions matrix for every user, I can build a function to recommend movies for any user. All I need to do is return the movies with the highest predicted rating that the specified user hasn't already rated. Though I didn't use actually use any explicit movie content features (such as genre or title), I'll merge in that information to get a more complete picture of the recommendations.
#
# I'll also return the list of movies the user has already rated, for the sake of comparison.

# + id="NWmGciBegIhl" colab={"base_uri": "https://localhost:8080/", "height": 245} outputId="577f2bdc-214d-4b22-e62f-2ea9aecbd126"
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
preds_df.head()


# + id="ggAv-Y_GgIhl"
def recommend_movies(preds_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) # UserID starts at 1
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'MovieID',
               right_on = 'MovieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


# + id="T6wmnxuTgIhl" colab={"base_uri": "https://localhost:8080/"} outputId="2a4d693e-7497-4200-af26-9282fd9b7266"
already_rated, predictions = recommend_movies(preds_df, 837, movies_df, ratings_df, 10)

# + [markdown] id="XdIpIY9ZgIhm"
# So, how'd I do?

# + id="PfP2cSPMgIhm" colab={"base_uri": "https://localhost:8080/", "height": 345} outputId="e28e4c9e-6ac3-4e64-bab4-5de77931b6fc"
already_rated.head(10)

# + colab={"base_uri": "https://localhost:8080/", "height": 345} id="7uNLhyK3Z95t" outputId="a385ec2b-e18b-4bd3-cc0e-1336654bc3d1"


df = already_rated[['MovieID', 'Title', 'Genres']].copy()
df.head(10)

# + id="eFx8wgwYgIhn" colab={"base_uri": "https://localhost:8080/", "height": 345} outputId="ed30c0d3-685e-4f39-cb48-73e0efba0108"
predictions

# + [markdown] id="u2ZnPxdzgIhn"
# Pretty cool! These look like pretty good recommendations. It's also good to see that, though I didn't actually use the genre of the movie as a feature, the truncated matrix factorization features "picked up" on the underlying tastes and preferences of the user. I've recommended some film-noirs, crime, drama, and war movies - all of which were genres of some of this user's top rated movies.

# + [markdown] id="fKyoDci9tu8K"
# # Visualizing true and predicted ratings matrix 

# + colab={"base_uri": "https://localhost:8080/"} id="46qng2bFwYXf" outputId="3cb85f4a-9ef5-493c-bb8d-73d8f44e5658"
Rpred = all_user_predicted_ratings
Rpred[Rpred < 0] = 0
Rpred[Rpred > 5] = 5

print(np.linalg.norm(R - Rpred, ord='fro'))

print(np.sqrt(np.sum(np.power(R - Rpred, 2))))


# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="pSk8UdyetzUk" outputId="421a5b15-c691-4464-db0f-155a902e67bc"
import matplotlib.pyplot as plt

nusers = 20
nitems = 20

plt.figure(figsize=(10,10))
plt.imshow(R[:nusers, :nitems], cmap='jet')
plt.xlabel('item')
plt.ylabel('user')
plt.title('True ratings')
plt.colorbar()


plt.figure(figsize=(10,10))
plt.imshow(Rpred[:nusers, :nitems], cmap='jet')
plt.xlabel('item')
plt.ylabel('user')
plt.title('Predcted ratings')
plt.colorbar()
