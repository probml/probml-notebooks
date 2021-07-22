# -*- coding: utf-8 -*-
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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/intro/datasets.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="cRiaabSUb5Sz"
# # Manipulating datasets
#
# In this colab, we briefly discuss ways to access and manipulate common datasets that are used in the ML literature. Most of these are used for supervised learning experiments.
#

# + id="02aUbbJ7d7u-"
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

# + [markdown] id="3pApO0WqcYLK"
# # Tabular datasets
#
# The [UCI ML repository](https://archive.ics.uci.edu/ml/index.php) contains many smallish datasets, mostly tabular.
#
# [Kaggle](https://www.kaggle.com/datasets) also hosts many interesting datasets.
#
# [Sklearn](https://scikit-learn.org/0.16/datasets/index.html) has many small datasets builtin, making them easy to use for prototyping, as we illustrate below.

# + colab={"base_uri": "https://localhost:8080/"} id="29RGlnbTeBYk" outputId="849ad93f-72d1-41fb-eb37-9f525229c19d"

from sklearn import datasets

iris = datasets.load_iris()
print(iris.keys())

X = iris['data']
y = iris['target'] # class labels
print(X.shape)
print(iris['feature_names']) # meaning of each feature
print(iris['target_names']) # meaning of each class




# + [markdown] id="xX4GSX3Fwt1S"
# # Tensorflow datasets
#
# [TFDS](https://www.tensorflow.org/datasets) is a handy way to handle large datasets as a stream of minibatches, suitable for large scale training and parallel evaluation. It can be used by tensorflow and JAX code, as we illustrate below. (See the [official colab](https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/overview.ipynb) for details.)
#
#
#

# + id="s7H7qrB8xT8J"
# Standard Python libraries
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Any, Iterator, Mapping, NamedTuple, Sequence, Tuple

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display
import sklearn


# + id="kYPihPWaXaSv" colab={"base_uri": "https://localhost:8080/"} outputId="965efbef-7ab6-4ae4-ae99-f573f84c6ec3"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
import tensorflow_datasets as tfds

print("tf version {}".format(tf.__version__))


# + id="cNVBYUoYQPrO"
import jax
from typing import Any, Callable, Sequence, Optional, Dict, Tuple
import jax.numpy as jnp

rng = jax.random.PRNGKey(0)

# + id="80Ltp1RTqA_K"
# Useful type aliases

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


# + [markdown] id="ckj0R3UbPDPV"
# ## Minibatching without using TFDS
#
# We first illustrate how to make streams of minibatches using vanilla numpy code. TFDS will then let us eliminate a lot of this boilerplate. As an example, let's package some small labeled datasets into two dictionaries, for train and test.

# + colab={"base_uri": "https://localhost:8080/"} id="bOkb5HqSPQfd" outputId="57391c65-d7c3-4ca7-ff27-1435d4616c6e"
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
  train_ds = {'X': X_train, 'y': y_train}
  test_ds = {'X': X_test, 'y': y_test}
  return train_ds, test_ds

train_ds, test_ds = get_datasets_iris()
print(train_ds['X'].shape)
print(train_ds['y'].shape)

# + colab={"base_uri": "https://localhost:8080/"} id="xz5fOdLxuIEz" outputId="3e9f6602-f730-4614-a2ba-8aa92211ca64"
iris = sklearn.datasets.load_iris()
print(iris.feature_names)
print(iris.target_names)


# + [markdown] id="_GBwrYnGSWjg"
# Now we make one pass (epoch) over the data, computing random minibatches of size 30. There are 100 examples total, but with a batch size of 30,
# we don't use all the data. We can solve such "boundary effects" later. 

# + colab={"base_uri": "https://localhost:8080/"} id="4oVJlk5rQIef" outputId="42f4ce69-93bf-4515-89c4-cdfcd2f73a51"
def extract_batch(ds, ndx):
  batch = {k: v[ndx, ...] for k, v in ds.items()}
  #batch = {'X': ds['X'][ndx,:],  'y': ds['y'][ndx]}
  return batch

def process_epoch(train_ds, batch_size, rng):
  train_ds_size = len(train_ds['X'])
  steps_per_epoch = train_ds_size // batch_size
  perms = jax.random.permutation(rng, len(train_ds['X']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size)) # perms[i,:] is list of  data indices for step i
  for step, perm in enumerate(perms):
    batch = extract_batch(train_ds, perm)
    print('processing batch {} X shape {}, y shape {}'.format(
        step, batch['X'].shape, batch['y'].shape))

batch_size = 30
process_epoch(train_ds, batch_size, rng)


# + [markdown] id="jo-p6Lr9cnB1"
# ## Minibatching with TFDS
#
# Below we show how to convert a numpy array into a TFDS.
# We shuffle the records and convert to minibatches, and then repeat these batches indefinitely to create an infinite stream,
# which we can convert to a python iterator. We pass this iterator of batches to our training loop.
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="jzX34Vv4cqUQ" outputId="e711b297-55ff-4f7f-da94-69b236d40b55"


def load_dataset_iris(split: str, batch_size: int) -> Iterator[Batch]:
  train_ds, test_ds = get_datasets_iris()
  if split == tfds.Split.TRAIN:
    ds = tf.data.Dataset.from_tensor_slices({"X": train_ds["X"], "y": train_ds["y"]})
  elif split == tfds.Split.TEST:
    ds = tf.data.Dataset.from_tensor_slices({"X": test_ds["X"], "y": test_ds["y"]})
  ds = ds.shuffle(buffer_size=1 * batch_size)
  ds = ds.batch(batch_size)
  ds = ds.cache()
  ds = ds.repeat() # make infinite stream of batches
  return iter(tfds.as_numpy(ds)) # python iterator

batch_size = 30
train_ds = load_dataset_iris(tfds.Split.TRAIN, batch_size)
valid_ds = load_dataset_iris(tfds.Split.TEST, batch_size)

print(train_ds)

training_steps = 5
for step in range(training_steps):
  batch = next(train_ds)
  print('processing batch {} X shape {}, y shape {}'.format(
        step, batch['X'].shape, batch['y'].shape))

# + [markdown] id="NfX0AeRpQewX"
# ## Preprocessing the data
#
# We can process the data before creating minibatches.
# We can also use pre-fetching to speed things up (see
# [this TF tutorial](https://www.tensorflow.org/guide/data_performance) for details.)
# We illustrate this below for MNIST.
#

# + id="V-dJ2J0kQkH4" outputId="fcdadd2d-95a9-4d5b-e78d-539a61b25f3e" colab={"base_uri": "https://localhost:8080/"}


def process_record(batch):
  image = batch['image']
  label = batch['label']
  # reshape image to standard size, just for fun
  image = tf.image.resize(image, (32, 32))
  # flatten image to vector
  shape = image.get_shape().as_list()
  D = np.prod(shape) # no batch dimension
  image = tf.reshape(image, (D,))
  # rescale to -1..+1
  image = tf.cast(image, dtype=tf.float32)
  image = ((image / 255.) - .5) * 2. 
  # convert to standard names
  return {'X': image, 'y': label} 

def load_mnist(split, batch_size):
  dataset, info = tfds.load("mnist", split=split, with_info=True)
  dataset = dataset.map(process_record)
  if split=="train":
    dataset = dataset.shuffle(10*batch_size, seed=0)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  dataset = dataset.cache()
  dataset = dataset.repeat()
  dataset = tfds.as_numpy(dataset) # leave TF behind
  num_examples = info.splits[split].num_examples
  return iter(dataset), num_examples


batch_size = 100
train_iter, num_train = load_mnist("train", batch_size)
test_iter, num_test = load_mnist("test", batch_size)

num_epochs = 3
num_steps = num_train // batch_size 
print(f'{num_epochs} epochs with batch size {batch_size} will take {num_steps} steps')

batch = next(train_iter)
print(batch['X'].shape)
print(batch['y'].shape)

# + [markdown] id="3w95_gDKij4F"
# # Vision datasets

# + [markdown] id="NIeQtCs1X6vh"
# ## MNIST
#
# There are many standard versions of MNIST,
# some of which are available from https://www.tensorflow.org/datasets. We give some examples below.
#

# + colab={"base_uri": "https://localhost:8080/"} id="Hr2NWROhhxNf" outputId="94c71f41-584d-481f-fded-f85ef5d29f41"
ds, info = tfds.load("binarized_mnist", split=tfds.Split.TRAIN, shuffle_files=True, with_info=True)
print(ds)
print(info)

# + colab={"base_uri": "https://localhost:8080/"} id="zfraK_18jp9q" outputId="74a7112e-a048-4420-9f49-2a05df903e3b"
train_ds, info = tfds.load("mnist", split=tfds.Split.TRAIN, shuffle_files=True, with_info=True)
print(train_ds)
print(info)

# + colab={"base_uri": "https://localhost:8080/", "height": 300, "referenced_widgets": ["e7267b0caf0b4567b9f75f2173c01ca9", "aa035dda94174d0291fd317977a79187", "507695c8b55d4d09838718faac6c018d", "36294079f5f34e1bb4aea5b31715e53b", "f26c7e4299b444aa97544068211da32d", "867e98ba8daa4caf8bf632bbce7279cc", "0384f357bfd742e1bad0b037870526a3", "dc9833915c4c4b1e8b48b7afb74acfff"]} id="RMijYNrVrkD8" outputId="2962e10e-81c9-4982-d6a1-c4c8845f07ac"
ds = tfds.load('mnist', split='train')
print(type(ds))
ds = ds.take(1)  # Only take a single example
print(type(ds))

for example in ds:  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
  print(list(example.keys()))
  image = example["image"]
  label = example["label"]
  print(image.shape, label)

# + colab={"base_uri": "https://localhost:8080/", "height": 365} id="L758Mpe8tGTC" outputId="5eef6fda-c15c-4b79-fc3e-aa3a7570aef2"
ds, info = tfds.load('mnist', split='train', with_info=True)
fig = tfds.show_examples(ds, info, rows=2, cols=5)

# This function is not well documented. But source code for show_examples is here:
# https://github.com/tensorflow/datasets/blob/v4.2.0/tensorflow_datasets/core/visualization/image_visualizer.py

# + id="YIcG-dUxNL9d"


# + [markdown] id="V1MEEmlJirie"
# ## CIFAR
#
# The CIFAR dataset is commonly used for prototyping.
# The CIFAR-10 version consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. There is also a 100 class version.
#
# An easy way to get this data is to use TFDS, as we show below.

# + colab={"base_uri": "https://localhost:8080/", "height": 723, "referenced_widgets": ["d526703337764075a7e6059ac6a46690", "4ee1f2714adb4a5eb67fd098ff132d95", "b9fdb63414f54a69910aa0853730e612", "1608d59a041340628684dff6e86372cf", "7ec5aa0ef97246289d50c2bbe65c68af", "7f92228eb07d4cb0875ae9c2ba20f50d", "56e7d78acb15479cb51f1176f71baaa8", "dc694382bd9147fe8af853794d60d6a3", "735a974553864ce49cf421893afc0fdd", "fea3adfca4394d7d9060c22a8dbb1971", "aa44b7d8111b4153bed9a40f1eace33d", "1404c3c099164e538806ac5eef9a5815", "deb690d31eae453d8534d2d627090bea", "80f5726191294a88be90a395e52b3ba1", "f7987104aebb4f4aa44670caf248c63a", "175e6d0844fe4c24a7c8c375793ccc67", "d8550a6424fd4488ac31925bac69ea68", "ebe231c9d2e8465b8664ec9ab2d9eda3", "635adf30b26f45c0a373a7def2308960", "9bd96821dce04471b5815930c8a9b03b", "41e055c0367141f398a3e0008caea0ad", "2a25e73a769447cc93d9e2265778e32b", "9ae04b505d614ff8876a787de7607d9a", "e3c634aa4a2c422c84f665d08301f2af", "3edf0eb13759411899c9c4b47693e6bb", "611ec94216164ec2a4c92c0d2fb46862", "07c8e2f409ff44439e03b495f573c2e7", "84231104b5cf4c14b2cc55e4bb359b2e", "7209f1021c7b4e80bde42ad6403dc68e", "a93dd47e9c2747fea237cf32b1325cbf", "b0835c04a435434681402c86822ab60b", "e14a614ea2da48ad972fe6a10af26fee", "07782bf0d647451a98cd06cc7b13c6d7", "16a5c001caf349e6bfe33e785ebc1ae2", "8c62556d792a404e8bd07c400408bd39", "46291d658720446c8a863518d4f95472", "b32c825dedcf4db39503b66228b70572", "893649147aa843aba91d85095561a2f0", "7e888e157bcc4fa4b2ca7c7ee6b55e05", "8ce3827639e54bee92d3aa55b3870d12", "17a7de6d743d4c9abaa7c0d0d4d2abc5", "0ba1bc4de2664563b020939e8809e2bd", "0f0a111781ea4745a41b912a78a7c728", "86bdb3ead122406581645f8b44826b57", "c20e6ed8a0e54598adcc6c450e530deb", "f2ab837c539149f39dbebc9bccd8d4f3", "72fecff90376435b87cfa102380c1383", "1c46e7f22fa74117802e32a6aec2a7db", "c19ed67332bd49b692c93add9e93456a", "0f52e93d3cae404493d8d06792ecc3c4", "5b4763b0b59d4ae1a1a4a259922fd6d6", "ef7fc697c1da4d218ea17e96e1834c36", "6d66709ce91d47a2bb7c01cec6893d4e", "7e020206852440ecab58892b51df895a", "9825bd31ea6644bda8e61e4588e30510", "66c5cf8d45c54a019a18ce85fb1d6308"]} id="u4IAzN4xwD8h" outputId="bd882907-de4f-4651-fb0f-18e479c4a8cf"
ds, info = tfds.load('cifar10', split='train', with_info=True)
fig = tfds.show_examples(ds, info, rows=2, cols=5)

# + [markdown] id="mvIhzkmlh0r5"
# ## Imagenet
#
# A lot of vision experiments use the Imagenet dataset, with 1000 classes and ~1M images.
# However, this takes a long time to download and process.
# The FastAI team made a smaller version called [ImageNette](https://github.com/fastai/imagenette), that only has 10 classes of size 160 or 320 pixels (largest dimension). This is good for prototyping, and the images tend to be easier to interpret that CIFAR. A version of the raw data, in a more convenient format (all images 224x224, no dependence on FastAI library) can be found [here](https://github.com/thunderInfy/imagenette). It is also bundled into TFDS, as we show below.

# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["1ce46b8c30574a118dcd0150e6820cdc", "3aca7d5f4b7949d8b2b6bf9533e7ebdc", "a3014e88235b47f286df45dc50acc93d", "18cbc4a8ec8849b7a5e030e71a66c995", "bbea7a031b5c4994b2b296c42cbc7cf9", "bdb2e63b11334faab83191bd15534c39", "173c88df3cc94db5b2d211f0fa6726ac", "c57e5d63e7f749aa938199a1ec1c7515", "61cce561be5b423cb2d8395b5c7bdd1b", "be0074f3f9f440fc8fd9f401c041f0e5", "f1a5873f03324f79a88671eba289ede0", "aad163d9632e4548a74ffe77c1261db9", "c16eb024010a4d9eba894f4e1dfc423d", "d7e3322442514f698d23a7e12b703e18", "33c5387c4f494eb1963346c3f8a299bc", "24b9f59d9d154047a9628cfa05691288", "9f306c58d3f741b28231abcb269583e7", "8912b52cd1a14b8a9bf57064f499e0b3", "bcbcb7c61cc44c009bfb17c9c7c15a2a", "f7854e040c104feb9d0f47ec79f04efc", "e44dfb54170743b5a2f77b6649f92b7f", "a79d8122071f4a8fa4ee4c885927daff", "dcd38a0abf384aa7a3a75173b9d2d497", "777c7a6b76c24a64b26051f8399d6fb6", "4c3441f4dbd044109787d883e8e5e631", "e484951ed5ff4e8098676658547014a3", "457e762b546e462d94a4665db9f3bc39", "dfd7a3c95e064fa3a0ca2095dc9ba0bf", "4e85850ed9ad4dcb8c2b746814c940fd", "1def96be1b814c5b95a49459d0502d1a", "1d59c42579db4a708bc930df8b3440ba", "97256ad1f9074e158216e532518579e7", "81dd63aed41441d7aafd6f1b756121f9", "688bbd80b2ec419b83a12dd9d12dc917", "4a76833d453d447facd4eb3ea41f1829", "955dea391d3c4387b361ef6bdcc3a853", "e3eb7a31bbe647a3a8520b268b236034", "8c41324381944a05aee3db1184a9dfba", "05d868c646524b228e745e66ce0a58ba", "aaca37f4b47946d68e89153b48bdcfcd", "4bf87c8a121a4916a9e371efcc579dbf", "1de21984c33e468c9f76ac322e41953e", "391c43486f6a4b7fa202cf28b2af1808", "c6be09e3ae8245c687e8cf6ea2528f70", "b8a556fec1844be9a615603faa101ace", "b248c85e461a4a98a25157308847f8a9", "daf15971b6b74adf87fc79b334a93964", "26807ca2840a4a34b69d0e50aafccadf", "fa5c188ccb2349e38205e5840a899700", "f87fb78be8e04977a7684df39528b86e", "d27e8f7b15964d498055ce8639d7ffee", "499bbb9f8b0b4b92901b447c9508df28", "63446d8755f1416fbd411c7013f9e875", "e3b4f58db500462c8419cc7db82e4d1a", "0d14b06c46ff4f1c9321b84ccc0c1fe2", "90dd1c0b5a0b403a8a06c633fcc8318c"]} id="z-UVh5ZWkBzw" outputId="5a6c114f-74a5-42a0-e212-b606d67ab2ca"
import tensorflow_datasets as tfds

imagenette_builder = tfds.builder("imagenette/full-size")
imagenette_info = imagenette_builder.info
print(imagenette_info)

imagenette_builder.download_and_prepare()


# + id="Rl2GabIAlEM9"
datasets = imagenette_builder.as_dataset(as_supervised=True)

# + colab={"base_uri": "https://localhost:8080/"} id="mn0QiyH6ka2X" outputId="9992352e-f6c6-4657-c30e-615d5c25df97"

train_examples = imagenette_info.splits['train'].num_examples
validation_examples = imagenette_info.splits['validation'].num_examples
print('ntrain', train_examples, 'nvalidation', validation_examples)

train, test = datasets['train'], datasets['validation']

import tensorflow as tf
batch_size = 32

train_batch = train.map(
    lambda image, label: (tf.image.resize(image, (448, 448)), label)).shuffle(100).batch(batch_size).repeat()

validation_batch = test.map(
    lambda image, label: (tf.image.resize(image, (448, 448)), label)
).shuffle(100).batch(batch_size).repeat()



# + colab={"base_uri": "https://localhost:8080/"} id="XwZfF8scljgI" outputId="8630e466-07e2-47f1-c73e-a6aea38f7980"
i = 0
for X, y in train_batch:
  #print(b)
  #X = b['image']
  #y = b['label']
  print('image {}, X shape {}, y shape {}'.format(i, X.shape, y.shape))
  i += 1
  if i > 1: break

# + colab={"base_uri": "https://localhost:8080/", "height": 357} id="vNPXpR2Fmqu9" outputId="d875a486-ad80-4888-e21f-1250af08881c"
fig = tfds.show_examples(train, imagenette_info, rows=2, cols=5)


# + [markdown] id="z4E9okpEoTTD"
# # Language datasets
#
# Various datasets are used in the natural language processing (NLP) communities.
#
# TODO: fill in.

# + [markdown] id="fUu0R4_rSJ6O"
# # Graveyard
#
# Here we store some scratch code that you can ignore,

# + id="ZtLBfzVM5hVd"
def get_datasets_mnist():
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds_all = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds_all = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

  num_train = len(train_ds_all['image'])
  train_ds['X'] = jnp.reshape(jnp.float32(train_ds_all['image']) / 255., (num_train, -1))
  train_ds['y'] = train_ds_all['label']

  num_test = len(test_ds_all['image'])
  test_ds['X'] = jnp.reshape(jnp.float32(test_ds['image']) / 255., (num_test, -1))
  test_ds['y'] = test_ds_all['label']

  return train_ds, test_ds


# + colab={"base_uri": "https://localhost:8080/"} id="kaMqUR3ggBlx" outputId="7e091b7a-2729-4404-a2c0-127cd1ecd109"
dataset = load_dataset_iris(tfds.Split.TRAIN, 30)
batches = dataset.repeat().batch(batch_size)

step = 0
num_minibatches = 5
for batch in batches:
    if step >= num_minibatches:
        break
    X, y = batch['image'], batch['label']
    print('processing batch {} X shape {}, y shape {}'.format(
        step, X.shape, y.shape))
    step = step + 1

# + colab={"base_uri": "https://localhost:8080/"} id="fFBvntfOgYpy" outputId="ff4eafed-1998-4751-f348-4e954d3df98d"
print('batchified version v2')
batch_stream = batches.as_numpy_iterator()
for step in range(num_minibatches):
  batch = batch_stream.next()
  X, y = batch['image'], batch['label'] # convert to canonical names
  print('processing batch {} X shape {}, y shape {}'.format(
        step, X.shape, y.shape))
  step = step + 1

# + id="H7QpYkxsgs6r"
ds=tfds.as_numpy(train_ds)
print(ds)
for i, batch in enumerate(ds):
  print(type(batch))
  X = batch['image']
  y = batch['label']
  print(X.shape)
  print(y.shape)
  i += 1
  if i > 2: break



ds = tfds.load('mnist', split='train')
ds = ds.take(100)
#ds = tfds.as_numpy(ds)

batches = ds.repeat(2).batch(batch_size)
print(type(batches))
print(batches)

batch_stream = batches.as_numpy_iterator()
print(type(batch_stream))
print(batch_stream)

b = next(batch_stream)
print(type(b))
print(b['image'].shape)

b = batch_stream.next()
print(type(b))
print(b['image'].shape)


ds = tfds.load('mnist', split='train')
batches = ds.repeat().batch(batch_size)
batch_stream = batches.as_numpy_iterator()


def process_stream(stream):
  b = next(stream)
  X = b['image']
  y = b['label']
  d = {'X': X, 'y': y}
  yield d
  
my_stream = process_stream(batch_stream)

b = next(my_stream)
print(type(b))
print(b['X'].shape)

b = my_stream.next()
print(type(b))
print(b['X'].shape)


# + id="4jmj_K2KSLwh"

def sample_categorical(N, C):
  p = (1/C)*np.ones(C);
  y = np.random.choice(C, size=N, p=p);
  return y

def get_datasets_rnd():
  Ntrain = 1000; Ntest = 1000; D = 5; C = 10;
  train_ds = {'X': np.random.randn(Ntrain, D), 'y': sample_categorical(Ntrain, C)}
  test_ds = {'X': np.random.randn(Ntest, D), 'y': sample_categorical(Ntest, C)}
  return train_ds, test_ds


def get_datasets_logreg(key):
  Ntrain = 1000; Ntest = 1000; D = 5; C = 10;
  W = jax.random.normal(key, (D,C))

  Xtrain = jax.random.normal(key, (Ntrain, D))
  logits = jnp.dot(Xtrain, W)
  ytrain = jax.random.categorical(key, logits)

  Xtest = jax.random.normal(key, (Ntest, D))
  logits = jnp.dot(Xtest, W)
  ytest = jax.random.categorical(key, logits)

  train_ds = {'X': Xtrain, 'y': ytrain}
  test_ds = {'X': Xtest, 'y': ytest}
  return train_ds, test_ds
