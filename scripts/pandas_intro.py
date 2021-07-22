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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/intro/pandas_intro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="N8swpNugjZ2m"
# # Pandas
#
# [Pandas](https://pandas.pydata.org/) is a widely used Python library for storing and manipulating tabular data, where feature columns may be of different types (e.g., scalar, ordinal, categorical, text). We give some examples of how to use it below.
#
# For very large datasets, you might want to use [modin](https://github.com/modin-project/modin), which provides the same pandas API but scales to multiple cores, by using [dask](https://github.com/dask/dask) or [ray](https://github.com/ray-project/ray) on the backend.

# + [markdown] id="uCLsMbr6jXHp"
# ### Install necessary libraries
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

from IPython.display import display, HTML

import sklearn

import seaborn as sns;
sns.set(style="ticks", color_codes=True)

import pandas as pd
pd.set_option('precision', 2) # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100) # wide windows

import xarray as xr

# + [markdown] id="fW9crfnZlUmU"
# ### Auto-mpg dataset <a class="anchor" id="EDA-autompg"></a>

# + id="k6SpViZcjhAQ" colab={"base_uri": "https://localhost:8080/"} outputId="b7d6e7ff-391e-453d-d956-0b15892957b2"
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Year', 'Origin', 'Name']
df = pd.read_csv(url, names=column_names, sep='\s+', na_values="?")

# The last column (name) is a unique id for the car, so we drop it
df = df.drop(columns=['Name'])

df.info()

# + [markdown] id="ryYgM3I6l1VH"
# We notice that there are only 392 horsepower rows, but 398 of the others.
# This is because the HP column has 6 **missing values** (also called NA, or
# not available).
# There are 3 main ways to deal with this:
# - Drop the rows with any missing values using dropna()
# - Drop any columns with any missing values using drop()
# - Replace the missing vales with some other valye (eg the median) using fillna. (This is called missing value imputation.)
# For simplicity, we adopt the first approach.
#

# + id="yq2XTXk4lwFk" colab={"base_uri": "https://localhost:8080/"} outputId="4ebb0303-3bbc-4525-8b8a-1fb8e6a3b4b2"
# Ensure same number of rows for all features.
df = df.dropna()
df.info()

# + id="BdwRK6ovl4Qy" colab={"base_uri": "https://localhost:8080/", "height": 277} outputId="98f0fb44-f4bd-46af-cc51-50d321ec42cc"
# Summary statistics
df.describe(include='all')

# + id="mJES73Pil6da" colab={"base_uri": "https://localhost:8080/"} outputId="b1b43078-10b3-40d7-cf3e-7e69b666dd0d"
# Convert Origin feature from int to categorical factor
df['Origin'] = df.Origin.replace([1,2,3],['USA','Europe','Japan'])
df['Origin'] = df['Origin'].astype('category')

# Let us check the categories (levels)
print(df['Origin'].cat.categories)

# Let us check the datatypes of all the features
print(df.dtypes)

# + id="ElSW55EVl8mE" colab={"base_uri": "https://localhost:8080/", "height": 191} outputId="1d8737cc-7ff7-4c19-f3f3-e768bac32454"
# Let us inspect the data. We see meaningful names for Origin.
df.tail()

# + id="aKkS27k5l-2K" colab={"base_uri": "https://localhost:8080/"} outputId="f8b5dab0-1ef5-4a87-cb82-b3b77081e6d5"
# Create latex table from first 5 rows 
tbl = df[-5:].to_latex(index=False, escape=False)
print(tbl)

# + id="rrv5YoXkmFUJ" colab={"base_uri": "https://localhost:8080/", "height": 285} outputId="bc9e42f2-a82d-41b5-88a7-5b046f2569d9"
# Plot mpg distribution for cars from different countries of origin
data = pd.concat( [df['MPG'], df['Origin']], axis=1)
fig, ax = plt.subplots()
ax = sns.boxplot(x='Origin', y='MPG', data=data)
ax.axhline(data.MPG.mean(), color='r', linestyle='dashed', linewidth=2)
#plt.savefig(os.path.join(figdir, 'auto-mpg-origin-boxplot.pdf'))
plt.show()

# + id="Zhd7OaXwmHPJ" colab={"base_uri": "https://localhost:8080/", "height": 285} outputId="98aaf055-7058-4690-b971-e4535e003357"
# Plot mpg distribution for cars from different years
data = pd.concat( [df['MPG'], df['Year']], axis=1)
fig, ax = plt.subplots()
ax = sns.boxplot(x='Year', y='MPG', data=data)
ax.axhline(data.MPG.mean(), color='r', linestyle='dashed', linewidth=2)
#plt.savefig(os.path.join(figdir, 'auto-mpg-year-boxplot.pdf'))
plt.show()

# + [markdown] id="KpKG_NgomPmd"
# ### Iris dataset <a class="anchor" id="EDA-iris"></a>

# + id="vZwptsuWmJIY" colab={"base_uri": "https://localhost:8080/"} outputId="2686df6a-2587-447c-ade4-5b16ddf41a84"
# Get the iris dataset and look at it
from sklearn.datasets import load_iris
iris = load_iris()
# show attributes of this object
print(dir(iris))

# Extract numpy arrays
X = iris.data 
y = iris.target
print(np.shape(X)) # (150, 4)
print(np.c_[X[0:3,:], y[0:3]]) # concatenate columns

# + id="iAdAEXxrmWIA" colab={"base_uri": "https://localhost:8080/"} outputId="520cbfa9-d235-430b-d775-5092e0dd7c2d"
# The data is sorted by class. Let's shuffle the rows.
N = np.shape(X)[0]
rng = np.random.RandomState(42)
perm = rng.permutation(N)
X = X[perm]
y = y[perm]
print(np.c_[X[0:3,:], y[0:3]])

# + id="4eWOuMkJmYCg" colab={"base_uri": "https://localhost:8080/", "height": 363} outputId="301f3c15-dec8-4a2f-836c-83b8399439cd"
# Convert to pandas dataframe 
df = pd.DataFrame(data=X, columns=['sl', 'sw', 'pl', 'pw'])
# create column for labels
df['label'] = pd.Series(iris.target_names[y], dtype='category')

# Summary statistics
df.describe(include='all')

# + id="zbHZKIdRmZpA" colab={"base_uri": "https://localhost:8080/", "height": 191} outputId="89063ae9-319b-4356-ec48-0fc38399abb6"
# Peak at the data
df.head()

# + id="OIgZSoOambhw" colab={"base_uri": "https://localhost:8080/"} outputId="200edc2c-9def-4fe7-998d-08df9d71f271"
# Create latex table from first 5 rows 
tbl = df[:6].to_latex(index=False, escape=False)
print(tbl)

# + id="1EMJx8FOmdOf" colab={"base_uri": "https://localhost:8080/", "height": 547} outputId="74107c7c-f4a1-420c-88db-5da5ac302b88"
# 2d scatterplot
#https://seaborn.pydata.org/generated/seaborn.pairplot.html
import seaborn as sns;
sns.set(style="ticks", color_codes=True)
# Make a dataframe with nicer labels for printing
#iris_df = sns.load_dataset("iris")
iris_df = df.copy()
iris_df.columns = iris['feature_names'] + ['label'] 
g = sns.pairplot(iris_df, vars = iris_df.columns[0:3] , hue="label")
#save_fig("iris-scatterplot.pdf")
plt.show()

# + [markdown] id="b18tmIORmjmw"
# ### Boston housing dataset <a class="anchor" id="EDA-boston"></a>

# + id="1IBp-q42me-o" colab={"base_uri": "https://localhost:8080/", "height": 277} outputId="75464b36-c127-489b-add1-ac1f9d1e633c"
# Load data (creates numpy arrays)
boston = sklearn.datasets.load_boston()
X = boston.data
y = boston.target

# Convert to Pandas format
df = pd.DataFrame(X)
df.columns = boston.feature_names
df['MEDV'] = y.tolist()

df.describe()

# + id="2I1RZQJyml2v" colab={"base_uri": "https://localhost:8080/", "height": 300} outputId="1375eb86-5ba6-4cc4-8854-006a56a2d093"
# plot marginal histograms of each column (13 features, 1 response)
plt.figure()
df.hist()
plt.show()

# + id="TvM6U_lYmzF-" colab={"base_uri": "https://localhost:8080/", "height": 737} outputId="a00e988c-d376-4ce1-e058-14b8c6799105"
# scatter plot of response vs each feature 
nrows = 3; ncols = 4;
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=[15, 10])
plt.tight_layout()
plt.clf()
for i in range(0,12):
    plt.subplot(nrows, ncols, i+1)
    plt.scatter(X[:,i], y)
    plt.xlabel(boston.feature_names[i])
    plt.ylabel("house price")
    plt.grid()
#save_fig("boston-housing-scatter.pdf")
plt.show()


# + [markdown] id="H-KJ-inpAnGi"
# # Xarray
#
# [Xarray](http://xarray.pydata.org/en/stable/quick-overview.html) generalizes pandas to multi-dimensional indexing. Put another way, xarray is a way to create multi-dimensional numpy arrays, where each dimension has a label (instead of having to remember axis ordering), and each value along each dimension can also have a specified set of allowable values (instead of having to be an integer index). This allows for easier slicing and dicing of data. We give some examples below.
#

# + id="JvQPxOjCBMT3"
import xarray as xr

# + [markdown] id="w2m5WTa1pRdh"
# ## DataArray
#
#
# A data-array is for storing a single, multiply-indexed variable. It is a generalization of a Pandas series.

# + [markdown] id="lj_1sc4pBYom"
# We create a 2d DataArray, where the first dimension is labeled 'gender' and has values 'male', 'female' and 'other' for its coordinates; the second dimension is labeled 'age', and has integer coordinates. We also associate some arbitrary attributes to the array.

# + colab={"base_uri": "https://localhost:8080/", "height": 236} id="c6Jug1-BAoop" outputId="4692eaaf-1a4d-4d89-e5db-4e667be83566"
X = np.reshape(np.arange(15), (3,5))
print(X)
attrs = {'authors': ['John', 'Mary'], 'date': '2021-01-29'}
data = xr.DataArray(X,
                    dims=("gender", "age"),
                    coords={"gender": ["male", "female", "other"]},
                    attrs = attrs)
data

# + colab={"base_uri": "https://localhost:8080/", "height": 201} id="MtD8hTRjB-IT" outputId="965a9fcf-1eb3-4ef4-cb98-7da6a899e348"
# select on dimension name and coordinate label
data.sel(gender="female") 

# + colab={"base_uri": "https://localhost:8080/"} id="JUyA35LDDy3Q" outputId="8f6d0fe9-172d-4b78-b1ef-f2d4e209cc79"
v = data.sel(gender="female").values
print(v)
assert np.all(v == X[1,:])


# + id="DEflsVfPM0-S" colab={"base_uri": "https://localhost:8080/", "height": 183} outputId="cb3491b0-82d7-4dfe-87bd-1faac809b6d4"
# the dict indexing method is equivalent to  data.sel(gender="other")
data.loc[dict(gender="other")] 
data

# + id="RL4muFaPLSaa" colab={"base_uri": "https://localhost:8080/", "height": 183} outputId="d62b07d8-436d-4aa5-ccc3-8a3f41606f1b"
# For assignment, we need to use the dict indexing method
data.loc[dict(gender="other")] = 42
data

# + colab={"base_uri": "https://localhost:8080/", "height": 183} id="sj2-e_0dC89_" outputId="3e6ef850-57a7-4023-b291-36118c266a51"
# select on dimension name and coordinate value
data.sel(age=3) 


# + colab={"base_uri": "https://localhost:8080/"} id="rmRwUXMbDi5A" outputId="f32e90c8-6a38-44b3-9dce-70ad57704754"
v = data.sel(age=3).values
print(v)
assert np.all(v == X[:,3])

# + colab={"base_uri": "https://localhost:8080/", "height": 183} id="UB7-hyXsD_Rv" outputId="e99aaae1-141d-497b-d3ba-e014404a99c6"
# select on dimension name and integer index
data.isel(gender=1)

# + colab={"base_uri": "https://localhost:8080/"} id="89Tr7KUJEV8U" outputId="e5429eb6-0a2f-45a2-dd08-982a764d3eeb"
# regular numpy indexing
data[1,:].values

# + [markdown] id="50gRESr-EiCj"
# We can also do [broadcasting](http://xarray.pydata.org/en/stable/computation.html#broadcasting-by-dimension-name) on xarrays.

# + colab={"base_uri": "https://localhost:8080/", "height": 150} id="UIxZiguRFqTt" outputId="6148efd4-02cf-4630-e896-19c8531946be"
a = xr.DataArray([1, 2], [("x", ["a", "b"])])
a

# + colab={"base_uri": "https://localhost:8080/", "height": 150} id="ZXLoJFYiJTGa" outputId="131deed1-4e59-4acc-890c-8305b3055902"
b = xr.DataArray([-1, -2, -3], [("y", [10, 20, 30])])
b

# + colab={"base_uri": "https://localhost:8080/", "height": 188} id="nonx9ZF-JYjx" outputId="008253dd-56ce-4221-df2a-4bb19ab7505b"

c = a*b
print(c.shape)
c

# + colab={"base_uri": "https://localhost:8080/", "height": 150} id="QrXOmKt_Jfx3" outputId="05b8face-a188-454f-a5ca-e565e92cd832"
data2 = xr.DataArray([10,20,30],dims=("gender"), coords={"gender": ["male", "female", "other"]})
data2

# + colab={"base_uri": "https://localhost:8080/", "height": 256} id="C--JRtYkJrp8" outputId="c02dadca-0542-4c6b-a437-2c895b6d138d"
c = data + data2
print(c.shape)
print(c.sel(gender="female"))
c

# + id="dc_r0HKlJswB"


# + [markdown] id="kaVvJ-nDpToC"
# ## DataSet
#
# An xarray DataSet is a collection of related DataArrays.
#

# + id="nI-uACUWpVNx"

