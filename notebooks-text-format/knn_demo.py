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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/exemplars/knn_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="POrA585UFLms"
# ##### Copyright and License.
#
# ![GitHub](https://img.shields.io/github/license/probml/pyprobml)
#
#
# Authors: Kevin P. Murphy (murphyk@gmail.com) and Mahmoud Soliman (mjs@aucegypt.edu)
#

# + id="I3CEU8u0FQR0"
# Attribution 
# This notebook is based on the following: 
# https://github.com/probml/pyprobml/blob/master/scripts/knn_classify_demo.py

# + id="qEYlbLuzFh_b" colab={"base_uri": "https://localhost:8080/"} outputId="5c92d965-8d24-4ced-e413-8f58076656f5"
# Imports

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score


from sklearn.datasets.samples_generator import make_blobs
from  IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile

from tqdm import tqdm

# + id="V4juJBTsQ4E3" cellView="both"
# In this notebook we will walk through KNN clustering technique
# Here we generate isotropic Gaussian blobs by using the make_blob function from sklearn
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=6, random_state=42)
ntrain = 100
x_train = X[:ntrain]
y_train = y[:ntrain]
x_test = X[ntrain:]
y_test = y[ntrain:]

# + id="ZCwHCZ1jQ456" colab={"base_uri": "https://localhost:8080/", "height": 281} outputId="c201eab1-b4c0-445d-bfe8-97fa035ec0d1"
# Plotting the generated training dataset by class in a scatter plot
plt.figure()
y_unique = np.unique(y_train)
markers = '*x+'
colors = 'bgr'
for i in range(len(y_unique)):
  plt.scatter(x_train[y_train == y_unique[i], 0],
             x_train[y_train == y_unique[i], 1],
             marker=markers[i],
             c=colors[i])
plt.title('train')

plt.show()

# + id="oWxuJCXkimN6" colab={"base_uri": "https://localhost:8080/", "height": 281} outputId="61dc13db-35fd-4011-a5a6-0021c9eba323"
# Plotting the generated test dataset by class in a scatter plot
plt.figure()
for i in range(len(y_unique)):
  plt.scatter(x_test[y_test == y_unique[i], 0],
             x_test[y_test == y_unique[i], 1],
             marker=markers[i],
             c=colors[i])
plt.title('test')

plt.show()

# + id="Ge9TeKKwipXK" colab={"base_uri": "https://localhost:8080/", "height": 808} outputId="3de3a31f-fb43-48f0-82c7-7d076996735e"
x = np.linspace(np.min(x_test[:, 0]), np.max(x_test[:, 0]), 200)
y = np.linspace(np.min(x_test[:, 1]), np.max(x_test[:, 1]), 200)
xx, yy = np.meshgrid(x, y)
xy = np.c_[xx.ravel(), yy.ravel()]

# Train a knn model and use the knn model to predict
for k in [1, 2, 5]:
  knn = KNN(n_neighbors=k)
  knn.fit(x_train, y_train)
  plt.figure()
  y_predicted = knn.predict(xy)
  
  plt.pcolormesh(xx, yy, y_predicted.reshape(200, 200), cmap='jet', alpha=0.2)
  for i in range(len(y_unique)):
      plt.scatter(x_train[y_train == y_unique[i], 0],
                 x_train[y_train == y_unique[i], 1],
                 marker=markers[i],
                 c=colors[i])
  plt.title('k=%s' % (k))
 
  plt.show()


# + id="mjkc0KH7iy4G" colab={"base_uri": "https://localhost:8080/", "height": 278} outputId="2717a589-d870-4e0f-b0da-72a715ec6cd7"
# plot train err and test err with different k
# ks = [int(n) for n in np.linspace(1, ntrain, 10)]
ks = [1, 5, 10, 20, 50, 70, 79]
train_errs = []
test_errs = []
for k in ks:
  knn = KNN(n_neighbors=k)
  knn.fit(x_train, y_train)
  train_errs.append(1 - knn.score(x_train, y_train))
  test_errs.append(1 - knn.score(x_test, y_test))
plt.figure()
plt.plot(ks, train_errs, 'bs:', label='train')
plt.plot(ks, test_errs, 'rx-', label='test')
plt.legend()
plt.xlabel('k')
plt.ylabel('misclassification rate')

plt.show()


# + id="ZIAU_GIzjAdk" colab={"base_uri": "https://localhost:8080/", "height": 311} outputId="ba33ffc9-b09a-4711-ceb0-714a395cdbcf"
#cross_validate
scores = []
for k in ks:
    knn = KNN(n_neighbors=k)
    score = cross_val_score(knn, x_train, y_train, cv=5)
    scores.append(1 - score.mean())
plt.figure()
plt.plot(ks, scores, 'ko-')
min_k = ks[np.argmin(scores)]
plt.plot([min_k, min_k], [0, 1.0], 'b-')
plt.xlabel('k')
plt.ylabel('misclassification rate')
plt.title('5-fold cross validation, n-train = 200')


# + id="L0_9Mlv5i1By" colab={"base_uri": "https://localhost:8080/", "height": 808} outputId="32b03f83-f6b9-4184-d460-df98347467b9"

#draw hot-map to show the probability of different class
knn = KNN(n_neighbors=10)
knn.fit(x_train, y_train)
xy_predic = knn.predict_proba(xy)
levels = np.arange(0, 1.01, 0.1)
for i in range(3):
    plt.figure()
    plt.contourf(xy_predic[:, i].ravel().reshape(200, 200), levels)
    plt.colorbar()
    plt.title('p(y=%s | data, k=10)' % (i))
plt.show()
