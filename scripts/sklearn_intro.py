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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/intro/sklearn_intro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="N8swpNugjZ2m"
# # Introduction to sklearn
#
# [Scikit-learn](http://scikit-learn.org) is a widely used Python machine learning library. There are several good tutorials on it, some of which we list below. 
#
#
# | Name | Notes |
# | ---- | ---- | 
# |[Python data science handbook](https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.02-Introducing-Scikit-Learn.ipynb)|  by Jake VanderPlas. Covers many python libraries. |
# |[Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow v2](https://github.com/ageron/handson-ml2)| by Aurelion Geron. Covers sklearn and TF2.|
# |[Python Machine Learning v3](https://sebastianraschka.com/books.html) | by Sebastian Raschka. Covers sklearn and TF2. |
#
# In the sections below, we just give a few examples of how to use it.
#
#
# If you want to scale up sklearn to handle datasets that do not fit into memory, and/or you want to run slow jobs in parallel (e.g., for grid search over model hyper-parameters) on multiple cores of your laptop or in the cloud, you should use [ML-dask](https://ml.dask.org/).

# + [markdown] id="uCLsMbr6jXHp"
# # Install necessary libraries
#
#
#

# + id="ds8fKSWOjX6J"
# Standard Python libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

from IPython import display

import sklearn

import seaborn as sns;
sns.set(style="ticks", color_codes=True)

import pandas as pd
pd.set_option('precision', 2) # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100) # wide windows



# + [markdown] id="7Tr1OoImahkV"
# # Estimators
#
# Most of sklearn is designed around the concept of "estimators", which are objects that can transform data. That is, we can think of an estimator as a function of the form $f(x,\theta)$, where $x$ is the input, and $\theta$ is the internal state (e.g., model parameters) of the object. Each estimator has two main methods:  ```fit``` and ```predict```. The fit method has the form ```f=fit(f,data)```, and updates the internal state (e.g., by computing the maximum likelihood estimate of the parameters). The predict method has the form ```y=predict(f,x)```. We can also have stateless estimators (with no internal parameters), which do things like preprocess the data. We give examples of all this below.

# + [markdown] id="AreGt3B0ssxu"
# # Logistic regression 
#
# We illustrate how to fit a logistic regression model using the Iris dataset.

# + colab={"base_uri": "https://localhost:8080/"} id="GDzMgv-5sx_T" outputId="63435d9e-82da-4211-e69c-6224e7134564"
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()

# use 2 features and all 3 classes
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

#softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", penalty="none")
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=1000, random_state=42)
softmax_reg.fit(X, y)

# Get predictive distribution for a single example
X = [[2.5, 3.0]] # (1,2) array
y_probs = softmax_reg.predict_proba(X)
print(np.round(y_probs, 2))

# + colab={"base_uri": "https://localhost:8080/"} id="dFF7PzN2tOmy" outputId="e353a8a5-a37e-4f68-81e3-0ed1b480b079"
# Fit model and evaluate on separate test set

from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features to make problem harder
#X = iris.data # use all data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# compute MLE (penalty=None means do not use regularization)
logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', penalty='none')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test) # categorical labels
errs = (y_pred != y_test)
nerrs = np.sum(errs)
print("Made {} errors out of {}, on instances {}".format(nerrs, len(y_pred), np.where(errs)))
# With ndims=2: Made 10 errors out of 50, on instances
#  (array([ 4, 15, 21, 32, 35, 36, 40, 41, 42, 48]),)


from sklearn.metrics import zero_one_loss
err_rate_test = zero_one_loss(y_test, y_pred)
assert np.isclose(err_rate_test, nerrs / len(y_pred))
err_rate_train =  zero_one_loss(y_train, logreg.predict(X_train))
print("Error rates on train {:0.3f} and test {:0.3f}".format(
    err_rate_train, err_rate_test))
#Error rates on train 0.180 and test 0.200

# + [markdown] id="1jMdzyTFnBJT"
# # Data preprocessing <a class="anchor" id="preprocess"></a>
#
# We often have to preprocess data before feeding it to an ML model. 
# We give some examples below.

# + [markdown] id="Mga3-OTEnTg-"
# ### Standardizing numeric features in Boston housing <a class="anchor" id="preprocess-boston"></a>

# + id="l_W4Catpm-Od" colab={"base_uri": "https://localhost:8080/", "height": 742} outputId="d9b32388-3738-41c8-f755-0a1f8c4db0f9"
import sklearn.datasets
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split


boston = sklearn.datasets.load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

scaler = sklearn.preprocessing.StandardScaler()
scaler = scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X) # entire dataset

# scatter plot of response vs each feature.
# The shape of the data looks the same as the unscaled case, but the x-axis of each feature is changed.
nrows = 3; ncols = 4;
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=[15,10])
plt.tight_layout()
plt.clf()
for i in range(0,12):
    plt.subplot(nrows, ncols, i+1)
    plt.scatter(X_scaled[:,i], y)
    plt.xlabel(boston.feature_names[i])
    plt.ylabel("house price")
    plt.grid()
#save_fig("boston-housing-scatter-scaled.pdf")
plt.show()

# + [markdown] id="JR9qx9pKndcW"
# ### One-hot encoding for Autompg <a class="anchor" id="preprocess-onehot"></a>
#
# We need to convert categorical inputs (aka factors) to one-hot vectors. We illustrate this below.
#

# + id="bQt8IZUNnd-t" colab={"base_uri": "https://localhost:8080/", "height": 435} outputId="59271c3f-2d98-45c3-98ea-57731fb6d788"
# Get data 
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Year', 'Origin', 'Name']
df = pd.read_csv(url, names=column_names, sep='\s+', na_values="?")

# The last column (name) is a unique id for the car, so we drop it
df = df.drop(columns=['Name'])

# Ensure same number of rows for all features.
df = df.dropna()

# Convert origin integer to categorical factor
df['Origin'] = df.Origin.replace([1,2,3],['USA','Europe','Japan'])
df['Origin'] = df['Origin'].astype('category')


df.info()

df.tail()

# + id="8foO-n6Znhoi" colab={"base_uri": "https://localhost:8080/"} outputId="faf78c1b-dcbf-40d7-9f78-cc5a58fb6fc0"
# Convert origin factor to integer
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
origin_cat = df['Origin']

print('before transform')
print(origin_cat)

origin_int = encoder.fit_transform(origin_cat)
print('after transform')
print(origin_int)

# Make sure we can decode back to strings
print('class names are {}'.format(encoder.classes_))
origin_cat2 = encoder.inverse_transform(origin_int)
print(origin_cat2)

# + id="DcXmfUdPnlX6" colab={"base_uri": "https://localhost:8080/"} outputId="e2101669-deba-4567-de1e-d9a5fa6f96e6"
# Convert integer encoding to one-hot vectors
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
origin_onehot_sparse = encoder.fit_transform(origin_int.reshape(-1,1)) # Sparse array
origin_onehot_dense = origin_onehot_sparse.toarray()
print(origin_onehot_dense[-5:,:])

# + id="fabhz1wWnq86" colab={"base_uri": "https://localhost:8080/", "height": 51} outputId="532ead5d-d22a-4b9e-b8c1-666038e9162d"
# We should be able to combine LabelEncoder and OneHotEncoder together
# using a Pipeline. However this fails due to known bug: https://github.com/scikit-learn/scikit-learn/issues/3956
# TypeError: fit_transform() takes 2 positional arguments but 3 were given

'''
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('str2int', LabelEncoder()),
    ('int2onehot', OneHotEncoder())
])
origin_onehot2 = pipeline.fit_transform(df['Origin'])
'''

# However, as of sckit v0.20, we can now convert Categorical to OneHot directly.
# https://jorisvandenbossche.github.io/blog/2017/11/20/categorical-encoder/
# https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696

'''
from sklearn.preprocessing import CategoricalEncoder # not available :(
encoder = CategoricalEncoder()
origin_onehot2 = encoder.fit_transform(df['Origin'])
print(origin_onehot2)
'''


# + id="CQ1yTfK8nucx" colab={"base_uri": "https://localhost:8080/", "height": 191} outputId="c1657339-785a-4346-ef42-c6f829d2b98e"
# Function to add one-hot encoding as extra columns to a dataframe

# See also sklearn-pandas library
#https://github.com/scikit-learn-contrib/sklearn-pandas#transformation-mapping

def one_hot_encode_dataframe_col(df, colname):
  encoder = OneHotEncoder(sparse=False)
  data = df[[colname]] # Extract column as (N,1) matrix
  data_onehot = encoder.fit_transform(data)
  df = df.drop(columns=[colname])
  ncats = np.size(encoder.categories_)
  for c in range(ncats):
    colname_c = '{}:{}'.format(colname, c)
    df[colname_c] = data_onehot[:,c]
  return df, encoder

df_onehot, encoder_origin = one_hot_encode_dataframe_col(df, 'Origin')

df_onehot.tail()

# + [markdown] id="wxk5g7Srn7p1"
# ### Feature crosses for Autompg <a class="anchor" id="preprocess-feature-cross"></a>
#
# We will use the [Patsy](https://patsy.readthedocs.io/en/latest/) library, which provides R-like syntax for specifying feature interactions.
#

# + id="TGHOBmDGnxVp" colab={"base_uri": "https://localhost:8080/"} outputId="2dc43bb0-79e6-4722-c42e-c14472b0b8db"
# Simple example of feature cross
import patsy

cylinders = pd.Series([4,   2,    3,   2,   4], dtype='int')
colors = pd.Series(['R', 'R', 'G', 'B', 'R'], dtype='category')
origin = pd.Series(['U', 'J', 'J', 'U', 'U'], dtype='category')
data = {'Cyl': cylinders, 'C': colors, 'O': origin}
df0 = pd.DataFrame(data=data)
print(df0)

df_cross0 = patsy.dmatrix('Cyl + C + O + C:O', df0, return_type='dataframe')
print(df_cross0.tail())

# + id="K-L0BxwEoCmw" colab={"base_uri": "https://localhost:8080/"} outputId="79267ce6-3281-49d3-f80f-e75f67df782c"
# Create feature crosses for AutoMPG

# For demo purposes, replace integer year with binary decade (70s and 80s)
year = df.pop('Year')
decade = [ 70 if (y>=70 and y<=79) else 80 for y in year ]
df['Decade'] =  pd.Series(decade, dtype='category')

# Make feature cross between #decades and origin (2*3 values)
y = df.pop("MPG") # Remove target column from dataframe and store
df.columns = ['Cyl', 'Dsp', 'HP', 'Wgt', 'Acc',  'O', 'D'] # Shorten names
df['O'] = df['O'].replace(['USA','Europe','Japan'], ['U','E','J'])
df_cross = patsy.dmatrix('D:O + Cyl + Dsp + HP + Wgt + Acc', df, return_type='dataframe')
print(df_cross.tail())

# + id="5B3bHDgXoE84"

