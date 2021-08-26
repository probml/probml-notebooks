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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/mlp/mlp_imdb_tf.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="kxiXt-gMpkL2"
# # MLP applied to IMDB movie reviews (binary sentiment analysis) <a class="anchor" id="imdb-keras-mlp"></a>
#
# We use the IMDB movie review dataset, where the task is to classify the sentiment of the review as positive or negative. We use the preprocessed version of the dataset from
# https://www.tensorflow.org/datasets

# + id="l3Gn9enu4hNY"

try:
    # # %tensorflow_version only exists in Colab.
    # %tensorflow_version 2.x
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. DNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

# + id="CmUmao_Vregq"
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
from time import time

# + id="zK91S3jWrww3" colab={"base_uri": "https://localhost:8080/"} outputId="c4c54645-0fea-4d57-bd8e-b691e08b05cf"
#We can also use the version that ships with keras (this does not require an additional download)
imdb = keras.datasets.imdb

vocab_size = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
print(np.shape(train_data)) # (25000)
print(train_data[0])
# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941...]

word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

for i in range(2):
  print('example {}, label {}'.format(i, train_labels[i]))
  print(decode_review(train_data[i]))

# + id="2XbwNZw3r0En" colab={"base_uri": "https://localhost:8080/"} outputId="2a12a2f0-7387-4283-e687-bd2c8558af26"
# Keras padding - every example in the dataset has fixed length

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

print(train_data.shape)
print(train_data[0])

# + id="OMKgm6Mtr2St" colab={"base_uri": "https://localhost:8080/"} outputId="d3c5446f-618d-41d7-fded-08bb8217de68"

embed_size = 16
def make_model(embed_size):
  tf.random.set_seed(42)
  np.random.seed(42)
  model = keras.Sequential()
  model.add(keras.layers.Embedding(vocab_size, embed_size))
  model.add(keras.layers.GlobalAveragePooling1D())
  model.add(keras.layers.Dense(16, activation=tf.nn.relu))
  model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
  model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
  return model

model = make_model(embed_size)
model.summary()

# + id="QhoUkXBWr4ds" colab={"base_uri": "https://localhost:8080/"} outputId="1ea2c56a-fe24-4b55-df42-019befe399c9"
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

history = model.fit(x_train,
                    y_train,
                    epochs=50,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# + id="8lydqd93r6xw" colab={"base_uri": "https://localhost:8080/"} outputId="53a99ed0-ad1a-4d09-e0cb-f3753c3e34c8"
history_dict = history.history
print(history_dict.keys())

results = model.evaluate(test_data, test_labels)
print(results)

# + id="78JEUWz5r9BD" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="796ac88e-8a21-41ba-bbef-587b8529cffd"
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
fig, ax = plt.subplots()
plt.plot(epochs, loss, 'bo', label='Training loss') 
plt.plot(epochs, val_loss, 'r-', label='Validation loss') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#save_fig("imdb-loss.pdf")
plt.show()


# + id="9PgxvslSr_5n" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="e69cdacf-9738-4882-8ce1-d2f6547092e9"
fig, ax = plt.subplots()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#save_fig("imdb-acc.pdf")
plt.show()


# + id="wyzlGm5DsCdg"
# Now turn on early stopping
# https://chrisalbon.com/deep_learning/keras/neural_network_early_stopping/

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
    
callbacks = [PrintDot(),
             keras.callbacks.EarlyStopping(monitor='val_acc', patience=2),
             keras.callbacks.ModelCheckpoint(filepath='imdb_keras_best_model.ckpt',
                                             monitor='val_acc', save_best_only=True)]

# + id="atzh4weFsEsx" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="cea4fdd0-3158-4a63-94e1-1085ac8ddc68"
# Reset parameters to a new random state
model = make_model(embed_size)
history = model.fit(
    x_train, y_train, epochs=50, batch_size=512, 
    validation_data=(x_val, y_val), verbose=0, callbacks=callbacks)

history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
fig, ax = plt.subplots()
plt.plot(epochs, loss, 'bo', label='Training loss') 
plt.plot(epochs, val_loss, 'r-', label='Validation loss') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#save_fig("imdb-loss-early-stop.pdf")
plt.show()


# + [markdown] id="_JOfkbubpZYg"
#
