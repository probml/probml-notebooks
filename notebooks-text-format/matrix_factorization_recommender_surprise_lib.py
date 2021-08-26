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
# <a href="https://colab.research.google.com/github/Nirzu97/pyprobml/blob/matrix-factorization/notebooks/matrix_factorization_recommender_surprise_lib.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="NTsyA3nxgIhT"
# # Matrix Factorization for Movie Lens Recommendations using Surprise library
#
#
#
#
#
#
#
#

# + id="q4cTdhEoaE7-"
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt


# + [markdown] id="EA6T5KISbY2f"
# # Surprise library for collaborative filtering
#
# http://surpriselib.com/
# Simple Python RecommendatIon System Engine
#

# + colab={"base_uri": "https://localhost:8080/"} id="jN4o9omlboLi" outputId="12cc5b65-8f89-477d-c58e-f1db88aab5bb"
# !pip install surprise

# + id="O_YvO1dmihg3"
import surprise

# + id="Bd60v17sbfOo"
from surprise import Dataset
data = Dataset.load_builtin('ml-1m')


# + id="_pu0cS1zb3v4"
trainset = data.build_full_trainset()


# + colab={"base_uri": "https://localhost:8080/"} id="jm3VA17gdn8S" outputId="714ea61f-7133-4420-9a92-206f1908a4e4"
print([trainset.n_users, trainset.n_items, trainset.n_ratings])

# + [markdown] id="zco5p3ACaE79"
# # Setting Up the Ratings Data
#
# We read the data directly from MovieLens website, since they don't allow redistribution. We want to include the metadata (movie titles, etc), not just the ratings matrix.
#

# + colab={"base_uri": "https://localhost:8080/"} id="Uxq9yLDyaE7_" outputId="fb2e6f79-388a-44d3-dea7-dde7d61d7bd4"
# !wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
# !unzip ml-1m
# !ls
folder = 'ml-1m'


# + id="3WPGyhw0aE8C"

ratings_list = [ [int(x) for x in i.strip().split("::")] for i in open(os.path.join(folder,'ratings.dat'), 'r').readlines()]
users_list = [i.strip().split("::") for i in open(os.path.join(folder, 'users.dat'), 'r').readlines()]
movies_list = [i.strip().split("::") for i in open(os.path.join(folder, 'movies.dat'), 'r',  encoding="latin-1").readlines()]



# + id="hEiTNglAaE8D"
ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

# + colab={"base_uri": "https://localhost:8080/", "height": 196} id="Is3-4FqOaE8E" outputId="6496c0e5-39ca-4224-c58a-8d315e9f0eca"
movies_df.head()


# + colab={"base_uri": "https://localhost:8080/"} id="s5CvzqNraE8G" outputId="5f839117-430c-4bb2-c07c-ea7a06d69632"
def get_movie_name(movies_df, movie_id_str):
  ndx = (movies_df['MovieID']==int(movie_id_str))
  name = movies_df['Title'][ndx].to_numpy()[0]
  return name

print(get_movie_name(movies_df, 1))
print(get_movie_name(movies_df, "527"))


# + colab={"base_uri": "https://localhost:8080/"} id="M-YAL-AIaE8I" outputId="5c1b2235-c321-4c0d-c2f7-3773e33d4cf9"
def get_movie_genres(movies_df, movie_id_str):
  ndx = (movies_df['MovieID']==int(movie_id_str))
  name = movies_df['Genres'][ndx].to_numpy()[0]
  return name

print(get_movie_genres(movies_df, 1))
print(get_movie_genres(movies_df, "527"))

# + colab={"base_uri": "https://localhost:8080/", "height": 196} id="m4t_0EFtgdiA" outputId="c70c37e5-3d3a-4117-c4ce-a9cbe9853738"
ratings_df.head()

# + colab={"base_uri": "https://localhost:8080/"} id="f_j4TpupfUAl" outputId="0168bb54-520c-4bb8-cef5-018a6d576795"
iter = trainset.all_ratings()
nshow = 5
counter = 0
for item in iter:
  #print(item)
  (uid_inner, iid_inner, rating)  = item
  # Raw ids are strings that match the external ratings file
  uid_raw = trainset.to_raw_uid(uid_inner)
  iid_raw = trainset.to_raw_iid(iid_inner)
  print('uid inner {}, raw {}, iid inner {}, raw {}, rating {}'.format(
      uid_inner, uid_raw, iid_inner, iid_raw, rating))
  counter += 1
  if counter > nshow: break


# + id="TGpMKj7glcx1" colab={"base_uri": "https://localhost:8080/"} outputId="3cadd35e-0926-46c6-a242-4e42d0865e98"
iid_raw = str(1318) 

items_raw = list(trainset.to_raw_iid(i) for i in trainset.all_items())
print(items_raw[:10])
print(type(items_raw[0]))
print(len(np.unique(items_raw)))

# + id="HxGk3RFrmTZo" colab={"base_uri": "https://localhost:8080/"} outputId="c486433a-6558-4a25-b717-e6a5bf8992cc"
users_raw = list(trainset.to_raw_uid(i) for i in trainset.all_users())
print(users_raw[:10])
print(len(np.unique(users_raw)))

# + id="2yV7RNqtmsL6" colab={"base_uri": "https://localhost:8080/"} outputId="208c4f5c-b950-4b55-b9e5-c7b2a5167020"
# inspect user ratings for user 837
uid_raw = str(837)
uid_inner = trainset.to_inner_uid(uid_raw)
user_ratings = trainset.ur[uid_inner]
print(len(user_ratings))
print(user_ratings)
rated_raw = [trainset.to_raw_iid(iid) for (iid, rating) in user_ratings]
print(rated_raw)
unrated_raw = list(set(items_raw) - set(rated_raw))
print(len(unrated_raw))


# + [markdown] id="Z4U41wxO7qDJ"
# # Join with meta data

# + id="i0RcPJAWzp3b"
def get_true_ratings(uid_raw, trainset):
  uid_inner = trainset.to_inner_uid(uid_raw)
  user_ratings = trainset.ur[uid_inner]
  item_list = [trainset.to_raw_iid(iid) for (iid, rating) in user_ratings] 
  rating_list = [rating for (iid, rating) in user_ratings]  
  item_list = np.array(item_list)
  rating_list = np.array(rating_list)
  ndx = np.argsort([-r for r in rating_list]) # largest (most negative) first
  return item_list[ndx], rating_list[ndx]

def make_predictions(algo, uid_raw, trainset):
  uid_inner = trainset.to_inner_uid(uid_raw)
  user_ratings = trainset.ur[uid_inner]
  rated_raw = [trainset.to_raw_iid(iid) for (iid, rating) in user_ratings]  
  items_raw = list(trainset.to_raw_iid(i) for i in trainset.all_items())
  unrated_raw = list(set(items_raw) - set(rated_raw))
  item_list = []
  rating_list = []
  for iid_raw in unrated_raw:
    pred = algo.predict(uid_raw, iid_raw,  verbose=False)
    uid_raw, iid_raw, rating_true, rating_pred, details =  pred
    item_list.append(iid_raw)
    rating_list.append(rating_pred)
  item_list = np.array(item_list)
  rating_list = np.array(rating_list)
  ndx = np.argsort([-r for r in rating_list]) # largest (most negative) first
  return item_list[ndx], rating_list[ndx]

def make_df(movies_df, item_list_raw, rating_list):
  name_list = []
  genre_list = []
  for i in range(len(item_list_raw)):
    item_raw = item_list_raw[i]
    name = get_movie_name(movies_df, item_raw)
    genre = get_movie_genres(movies_df, item_raw)
    name_list.append(name)
    genre_list.append(genre)
  df = pd.DataFrame({'name': name_list, 'genre': genre_list, 'rating': rating_list, 'iid': item_list_raw})
  return df





# + id="SypyFupC62f4" colab={"base_uri": "https://localhost:8080/", "height": 345} outputId="c9d8627e-8df9-4030-a392-576dd22bb737"
uid_raw = str(837)

item_list_raw, rating_list = get_true_ratings(uid_raw, trainset)
df = make_df(movies_df, item_list_raw, rating_list)
df.head(10)




# + [markdown] id="xWwi50q27xvN"
# # Fit/ predict

# + id="IlkjbU5L7zgG" colab={"base_uri": "https://localhost:8080/"} outputId="f6dab9d4-7203-48fd-c0e6-3aef39eed0ef"
# https://surprise.readthedocs.io/en/stable/matrix_factorization.html
algo = surprise.SVD(n_factors=50, biased=True, n_epochs=20, random_state=42, verbose=True)
algo.fit(trainset)

# + id="xtM1C-Rd63I0" colab={"base_uri": "https://localhost:8080/", "height": 345} outputId="bf02d880-c463-46da-90c3-760cf528e1ec"
uid_raw = str(837)

item_list_raw, rating_list = make_predictions(algo, uid_raw, trainset)
df = make_df(movies_df, item_list_raw, rating_list)
df.head(10)


# + [markdown] id="2M4FLANA8Vzg"
# # Visualize matrix of predictions

# + id="dGS8ihSPAhMF" colab={"base_uri": "https://localhost:8080/"} outputId="e263af01-2457-4314-e4fb-173abe515dcf"
# inspect user ratings for user 837
uid_raw = str(837)
uid_inner = trainset.to_inner_uid(uid_raw)
user_ratings = trainset.ur[uid_inner]
print(len(user_ratings))
print(user_ratings)
ratings_raw = [rating for (iid, rating) in user_ratings]
rated_raw = [trainset.to_raw_iid(iid) for (iid, rating) in user_ratings]
print(rated_raw)
print(trainset.to_raw_iid(1231))
print(ratings_raw[0])


# + id="ZjzdR9KRC5Ry" colab={"base_uri": "https://localhost:8080/"} outputId="6c29447f-58aa-4405-dd80-7e24092c6833"
def get_rating(trainset, uid_raw, iid_raw):
  uid_inner = trainset.to_inner_uid(uid_raw)
  user_ratings = trainset.ur[uid_inner]
  rated_iid_raw = np.array([trainset.to_raw_iid(iid) for (iid, rating) in user_ratings])
  ratings = np.array([rating for (iid, rating) in user_ratings])
  ndx = np.where(rated_iid_raw == iid_raw)[0]
  if len(ndx)>0:
    return ratings[ndx][0]
  else:
    return 0


print(get_rating(trainset, '837', '1201'))
print(get_rating(trainset, '837', '0'))

# + id="Qkkc8AQnvMUy" colab={"base_uri": "https://localhost:8080/", "height": 537} outputId="96eccf71-d3a7-437e-b1ca-3490ceb32d82"


users_raw = list(trainset.to_raw_uid(i) for i in trainset.all_users())
items_raw = list(trainset.to_raw_iid(i) for i in trainset.all_items())

users_raw = ['837'] + users_raw
items_raw = [str(i) for i in range(1200, 1300)]
nusers = 20
nitems = 20

Rtrue = np.zeros((nusers, nitems))
Rpred = np.zeros((nusers, nitems))
for ui in range(nusers):
  for ii in range(nitems):
    uid = users_raw[ui]
    iid = items_raw[ii]
    pred = algo.predict(uid, iid,  verbose=False)
    uid_raw, iid_raw, _, rating_pred, details =  pred
    Rpred[ui, ii] = rating_pred
    Rtrue[ui, ii] = get_rating(trainset, uid_raw, iid_raw)


plt.figure(); plt.imshow(Rtrue, cmap='jet'); plt.colorbar()
plt.figure(); plt.imshow(Rpred, cmap='jet'); plt.colorbar()
