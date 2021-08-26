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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/opt_flax.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="b520E1nCIBHc"
# # Optimization using Flax
#
#
# [Flax](https://colab.research.google.com/giathub/probml/pyprobml/blob/master/book1/mlp/flax_intro.ipynb) is a JAX library for creating deep neural networks. It also has a simple optimization library built in.
# Below we show how to fit a multi-class logistic regression model using flax. 
#
#
#

# + id="UeuOgABaIENZ"
import sklearn
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import itertools
import time
from functools import partial
import os

import numpy as np
#np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

# + id="TNQHpyKLIx_P" colab={"base_uri": "https://localhost:8080/"} outputId="b2e9b02d-bbad-4672-b6d2-8881f558b8b5"

import jax
import jax.numpy as jnp

from jax.scipy.special import logsumexp
from jax import grad, hessian, jacfwd, jacrev, jit, vmap
print("jax version {}".format(jax.__version__))



# + [markdown] id="RUICitLqjkrR"
# ## Import code

# + id="HHI0RPrPblpY" colab={"base_uri": "https://localhost:8080/"} outputId="72d1fffb-7252-4681-d83d-1af3b83e3790"
# Install Flax at head:
# !pip install --upgrade -q git+https://github.com/google/flax.git

# + id="jyv9ODiCf_aH"
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import optim

from jax.config import config
config.enable_omnistaging() # Linen requires enabling omnistaging

# + colab={"base_uri": "https://localhost:8080/"} id="pBzM5HwiiuM6" outputId="28028867-9e59-4c2f-df22-17c8a6a3675c"
# Book code
# !git clone https://github.com/probml/pyprobml

# + colab={"base_uri": "https://localhost:8080/"} id="4SMa9njai3Qt" outputId="2e816ef5-a5d2-4cd8-d8a6-04a334315ca5"


import pyprobml.scripts.fit_flax as ff

ff.test()

# + [markdown] id="YWp-tBzfdXHe"
# Now we show the source code for the fitting function in the file editor on the RHS.
#
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 17} id="eT5_wY4SdacY" outputId="329d4db2-b8c4-462c-8c3b-a87817b50d0c"
from google.colab import files
files.view('pyprobml/scripts/fit_flax.py')

# + [markdown] id="wHnVMv3zjnt3"
# ## Data
#
# We use the [tensorflow datasets](https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/intro/datasets.ipynb) library to make it easy to create minibatches.
#
# We switch to the multi-class version of Iris.

# + colab={"base_uri": "https://localhost:8080/"} id="0a-tDJOfjIf7" outputId="2e44c3f8-aace-49e2-9acc-ed8df501e993"
import tensorflow as tf
import tensorflow_datasets as tfds

import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split

def get_datasets_iris():
  iris = sklearn.datasets.load_iris()
  X = iris["data"]
  y = iris["target"] 
  N, D = X.shape # 150, 4
  X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.33, random_state=42)
  train_data = {'X': X_train, 'y': y_train}
  test_data = {'X': X_test, 'y': y_test}
  return train_data, test_data

def load_dataset_iris(split, batch_size=None):
  train_ds, test_ds = get_datasets_iris()
  if split == tfds.Split.TRAIN:
    ds = tf.data.Dataset.from_tensor_slices({"X": train_ds["X"], "y": train_ds["y"]})
  elif split == tfds.Split.TEST:
    ds = tf.data.Dataset.from_tensor_slices({"X": test_ds["X"], "y": test_ds["y"]})
  if batch_size is not None:
    ds = ds.shuffle(buffer_size=batch_size)
    ds = ds.batch(batch_size)
  else:
    N = len(train_ds['X'])
    ds = ds.batch(N)
  ds = ds.prefetch(buffer_size=5)
  ds = ds.repeat() # make infinite stream of data
  return iter(tfds.as_numpy(ds)) # python iterator


batch_size = 30
train_ds = load_dataset_iris(tfds.Split.TRAIN, batch_size)
batch = next(train_ds)
print(batch['X'].shape)
print(batch['y'].shape)

test_ds = load_dataset_iris(tfds.Split.TEST, None) # load full test set
batch = next(test_ds)
print(batch['X'].shape)
print(batch['y'].shape)


# + [markdown] id="VrzcCrmsjpi-"
# ## Model

# + id="P5JQ3iovjqGS"
class Model(nn.Module):
  nhidden: int
  nclasses: int

  @nn.compact
  def __call__(self, x):
    if self.nhidden > 0:
      x = nn.Dense(self.nhidden)(x)
      x = nn.relu(x)
    x = nn.Dense(self.nclasses)(x)
    x = nn.log_softmax(x)
    return x


# + [markdown] id="bwuGK8GJjxy_"
# ## Training loop
#

# + colab={"base_uri": "https://localhost:8080/", "height": 497} id="fN29jn7XjzG1" outputId="6bb61444-f4e8-48e4-d0f1-767459a98cf8"
from flax import optim

make_optimizer = optim.Momentum(learning_rate=0.1, beta=0.9)

model = Model(nhidden = 0, nclasses=3) # no hidden units ie logistic regression

batch_size = 100 # 30 # full batch training
train_ds = load_dataset_iris(tfds.Split.TRAIN, batch_size)
test_ds = load_dataset_iris(tfds.Split.TEST, batch_size)

rng = jax.random.PRNGKey(0)
num_steps = 200

  
params, history =  ff.fit_model(
    model, rng, num_steps, train_ds, test_ds, print_every=20)

display(history)

# + colab={"base_uri": "https://localhost:8080/", "height": 278} id="-NzU_wMAkut-" outputId="1f95b91a-4b7e-41a8-a3dc-41e48c54a0aa"
plt.figure()
plt.plot(history['step'], history['test_accuracy'], 'o-', label='test accuracy')
plt.xlabel('num. minibatches')
plt.legend()
plt.show()

# + [markdown] id="um91hW0ikzfe"
# ## Compare to sklearn
#

# + colab={"base_uri": "https://localhost:8080/"} id="1XPa5V5hk0vd" outputId="7a319aba-817e-4371-ec72-ed42bb3c1a1c"
train_ds, test_ds = get_datasets_iris()
from sklearn.linear_model import LogisticRegression

# We set C to a large number to turn off regularization.
log_reg = LogisticRegression(solver="lbfgs", C=1e3, fit_intercept=True)
log_reg.fit(train_ds['X'], train_ds['y'])

w_sklearn = np.ravel(log_reg.coef_)
print(w_sklearn)
b_sklearn = np.ravel(log_reg.intercept_)
print(b_sklearn)

yprob_sklearn = log_reg.predict_proba(test_ds['X'])
print(yprob_sklearn.shape)
print(yprob_sklearn[:10,:])


ypred_sklearn = jnp.argmax(yprob_sklearn, axis=-1)
print(ypred_sklearn.shape)
print(ypred_sklearn[:10])

# + colab={"base_uri": "https://localhost:8080/"} id="I_QxgKCilBrn" outputId="8b3918cb-feca-40d7-d4fd-6aa23aea3a99"
# Flax version
print(params)

train_ds, test_ds = get_datasets_iris()
Xtest = test_ds['X']
logits = model.apply({'params': params}, Xtest)
yprob = nn.softmax(logits)
print(yprob.shape)
print(yprob[:10,:])
print(np.allclose(yprob_sklearn, yprob, atol=1e-0)) # very loose numerical tolerance

ypred = jnp.argmax(yprob, axis=-1)
print(ypred[:10])
print(np.allclose(ypred_sklearn, ypred))
