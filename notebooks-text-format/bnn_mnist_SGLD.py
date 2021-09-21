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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/bnn_mnist_SGLD.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="JSZw_Iv0t5Aw"
# # Bayesian MLP for MNIST using preconditioned SGLD
#
# We use the [Jax Bayes](https://github.com/jamesvuc/jax-bayes) library 
# by  James Vuckovic 
# to fit an MLP to MNIST using SGD, and SGLD (with RMS preconditioning).
# Code is based on:
#
#
#
# 1.   https://github.com/jamesvuc/jax-bayes/blob/master/examples/deep/mnist/mnist.ipynb
# 2.   https://github.com/jamesvuc/jax-bayes/blob/master/examples/deep/mnist/mnist_mcmc.ipynb
#

# + [markdown] id="YwuRJZKAuoQs"
# # Setup

# + id="PmY4LL3AuL0j"
# %%capture
# !pip install git+https://github.com/deepmind/dm-haiku
# !pip install git+https://github.com/jamesvuc/jax-bayes

# + id="2Mz0kTuHwKqy"
import haiku as hk

import jax.numpy as jnp
from jax.experimental import optimizers
import jax

import jax_bayes

import sys, os, math, time
import numpy as onp
import numpy as np
from functools import partial 
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow_datasets as tfds


# + [markdown] id="oyxSjxhZwU2_"
# # Data

# + id="2HDMRDZwv9-d"
def load_dataset(split, is_training, batch_size):
  ds = tfds.load('mnist:3.*.*', split=split).cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  #return tfds.as_numpy(ds)
  return iter(tfds.as_numpy(ds))
 



# + id="WcjAlEjtwVr1"

# load the data into memory and create batch iterators
train_batches = load_dataset("train", is_training=True, batch_size=1_000)
val_batches = load_dataset("train", is_training=False, batch_size=10_000)
test_batches = load_dataset("test", is_training=False, batch_size=10_000)

# + [markdown] id="rE5YVKsawZj-"
# # Model

# + id="-QMiyBlEwY5q"
nclasses = 10


def net_fn(batch, sig):
  """ Standard LeNet-300-100 MLP """
  x = batch["image"].astype(jnp.float32) / 255.
  # x has size (1000, 28, 28, 1)
  D = np.prod(x.shape[1:]) # 784
  # To match initialization of linear layer
  # sigma = 1/sqrt(fan-in)
  # https://dm-haiku.readthedocs.io/en/latest/api.html#id1
  #w_init = hk.initializers.TruncatedNormal(stddev=stddev)
  sizes = [D, 300, 100, nclasses]
  sigmas = [sig/jnp.sqrt(fanin) for fanin in sizes]
  mlp = hk.Sequential([
    hk.Flatten(),
    hk.Linear(sizes[1], 
              w_init=hk.initializers.TruncatedNormal(stddev=sigmas[0]),
              b_init=jnp.zeros), 
    jax.nn.relu, 
    hk.Linear(sizes[2], 
              w_init=hk.initializers.TruncatedNormal(stddev=sigmas[1]),
              b_init=jnp.zeros), 
    jax.nn.relu, 
    hk.Linear(sizes[3],
              w_init=hk.initializers.TruncatedNormal(stddev=sigmas[2]),
              b_init=jnp.zeros)
    ])

  return mlp(x)

# L2 regularizer will be added to loss
reg = 1e-4

# + [markdown] id="vhHWqpc8wkE3"
# # SGD

# + id="Erm8G2YZwkcq"
net = hk.transform(partial(net_fn, sig=1))

lr = 1e-3
opt_init, opt_update, opt_get_params = optimizers.rmsprop(lr)

# instantiate the model parameters --- requires a sample batch to get size
params_init = net.init(jax.random.PRNGKey(42), next(train_batches))

# intialize the optimzier state
opt_state = opt_init(init_params)

def loss(params, batch):
  logits = net.apply(params, None, batch)
  labels = jax.nn.one_hot(batch['label'], 10)

  l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) 
            for p in jax.tree_leaves(params))
  
  softmax_crossent = - jnp.mean(labels * jax.nn.log_softmax(logits))

  return softmax_crossent + reg * l2_loss

@jax.jit
def accuracy(params, batch):
  preds = net.apply(params, None, batch)
  return jnp.mean(jnp.argmax(preds, axis=-1) == batch['label'])

@jax.jit
def train_step(i, opt_state, batch):
  params = opt_get_params(opt_state)
  dx = jax.grad(loss)(params, batch)
  opt_state = opt_update(i, dx, opt_state)
  return opt_state


# + colab={"base_uri": "https://localhost:8080/"} id="vfW2P3jX_yYu" outputId="97ceed56-51fa-40c9-a316-78877b9ceadc"
print(params_init['linear']['w'].shape)


# + id="r54UuI0n1f_l"
def callback(step, params, train_eval, test_eval, print_every=500):
  if step % print_every == 0:
    # Periodically evaluate classification accuracy on train & test sets.
    train_accuracy = accuracy(params, next(train_eval))
    test_accuracy = accuracy(params, next(test_eval))
    train_accuracy, test_accuracy = jax.device_get(
        (train_accuracy, test_accuracy))
    print(f"[Step {step}] Train / Test accuracy: "
          f"{train_accuracy:.3f} / {test_accuracy:.3f}.")


# + colab={"base_uri": "https://localhost:8080/"} id="b0hKy5QgzfJM" outputId="1cfe1801-4685-4992-e63b-a03e6f278742"
# %%time

nsteps = 5000
for step in range(nsteps+1):
  opt_state = train_step(step, opt_state, next(train_batches))
  params_sgd = opt_get_params(opt_state)
  callback(step, params_sgd, val_batches, test_batches)

  

# + [markdown] id="OlXgvVn56ICo"
# # SGLD

# + id="08j-W-vF6Im5"
lr = 5e-3

num_samples = 10 # number of samples to approximate the posterior
init_stddev =  0.01 #0.1 # params sampled around params_init

# we initialize all weights to 0 since we will be sampling them anyway
#net_bayes = hk.transform(partial(net_fn, sig=0))

sampler_fns = jax_bayes.mcmc.rms_langevin_fns
seed = 0
key = jax.random.PRNGKey(seed)
sampler_init, sampler_propose, sampler_update, sampler_get_params = \
  sampler_fns(key, num_samples=num_samples, step_size=lr, init_stddev=init_stddev)



# + id="UqMB8wbX6nD1"
@jax.jit
def accuracy_bayes(params_samples, batch):
  # average the logits over the parameter samples
  pred_fn = jax.vmap(net.apply, in_axes=(0, None, None))
  preds = jnp.mean(pred_fn(params_samples, None, batch), axis=0)
  return jnp.mean(jnp.argmax(preds, axis=-1) == batch['label'])


#the log-probability is the negative of the loss
logprob = lambda p,b : - loss(p, b)

#build the mcmc step. This is like the opimization step, but for sampling
@jax.jit
def mcmc_step(i, sampler_state, sampler_keys, batch):
  #extract parameters
  params = sampler_get_params(sampler_state)
  
  #form a partial eval of logprob on the data
  logp = lambda p: logprob(p,  batch)
  
  # evaluate *per-sample* gradients
  fx, dx = jax.vmap(jax.value_and_grad(logp))(params)

  # generat proposal states for the Markov chains
  sampler_prop_state, new_keys = sampler_propose(i, dx, sampler_state, 
                          sampler_keys)
  
  #we don't need to re-compute gradients for the accept stage (unadjusted Langevin)
  fx_prop, dx_prop = fx, dx

  # accept the proposal states for the markov chain
  sampler_state, new_keys = sampler_update(i, fx, fx_prop, 
                                           dx, sampler_state, 
                                           dx_prop, sampler_prop_state, 
                                           new_keys)
  
  return jnp.mean(fx), sampler_state, new_keys


# + id="QgNjQqeh7IS_"
def callback_bayes(step, params, val_batches, test_batches, print_every=500):
  if step % print_every == 0:
    val_acc = accuracy_bayes(params, next(val_batches))
    test_acc = accuracy_bayes(params, next(test_batches))
    print(f"step = {step}"
        f" | val acc = {val_acc:.3f}"
        f" | test acc = {test_acc:.3f}")
    



# + colab={"base_uri": "https://localhost:8080/"} id="7aaaOUbD7F7F" outputId="a61d9467-2936-46c4-9955-664a86cd85a3"

# %%time

#get a single sample of the params using the normal hk.init(...)
params_init = net.init(jax.random.PRNGKey(42), next(train_batches))

# get a SamplerState object with `num_samples` params along dimension 0
# generated by adding Gaussian noise (see sampler_fns(..., init_dist='normal'))
sampler_state, sampler_keys = sampler_init(params_init)

# iterate the the Markov chain
nsteps = 5000
for step in range(nsteps+1):
  train_logprob, sampler_state, sampler_keys = \
    mcmc_step(step, sampler_state, sampler_keys, next(train_batches))
  params_samples = sampler_get_params(sampler_state)
  callback_bayes(step, params_samples, val_batches, test_batches)


# + colab={"base_uri": "https://localhost:8080/"} id="N4S3nB9tAc4X" outputId="83327308-7c4d-4d32-cfdc-818346105429"
print(params_samples['linear']['w'].shape) # 10 samples of the weights for first layer

# + [markdown] id="gC1EIJpH8_Bb"
# # Uncertainty analysis

# + [markdown] id="e684Q2Tw9BTS"
# We select the predictions above a confidence threshold, and compute the predictive accuracy on that subset. As we increase the threshold, the accuracy should increase, but fewer examples will be selected.

# + id="xMenkseG9wQv"
test_batch = next(test_batches)
from jax_bayes.utils import entropy, certainty_acc


# + id="NtzhVPR-9ANv"
def plot_acc_vs_confidence(predict_fn, test_batch):
  # plot how accuracy changes as we increase the required level of certainty
  preds = predict_fn(test_batch) #(batch_size, n_classes) array of probabilities
  acc, mask = certainty_acc(preds, test_batch['label'], cert_threshold=0)
  thresholds = [0.1 * i for i in range(11)] 
  cert_accs, pct_certs = [], []
  for t in thresholds:
    cert_acc, cert_mask = certainty_acc(preds, test_batch['label'], cert_threshold=t)
    cert_accs.append(cert_acc)
    pct_certs.append(cert_mask.mean())

  fig, ax = plt.subplots(1)
  line1 = ax.plot(thresholds, cert_accs, label='accuracy at certainty', marker='x')
  line2 = ax.axhline(y=acc, label='regular accuracy', color='black')
  ax.set_ylabel('accuracy')
  ax.set_xlabel('certainty threshold')

  axb = ax.twinx()
  line3 = axb.plot(thresholds, pct_certs, label='pct of certain preds', 
                  color='green', marker='x')
  axb.set_ylabel('pct certain')

  lines = line1 + [line2] + line3
  labels = [l.get_label() for l in lines]
  ax.legend(lines, labels, loc=6)

  return fig, ax


# + [markdown] id="iut61vJp9w5h"
# ## SGD
#
# For the plugin estimate, the model is very confident on nearly all of the points.

# + id="JDSDCJcfExTh"
# plugin approximation to  posterior predictive 
@jax.jit
def posterior_predictive_plugin(params, batch):
  logit_pp = net.apply(params, None, batch)
  return jax.nn.softmax(logit_pp, axis=-1)


# + colab={"base_uri": "https://localhost:8080/", "height": 279} id="MSV8qbLX9zca" outputId="77d38ed6-a43d-4f7a-ba8d-fc217da1f0a3"

def pred_fn(batch):
  return posterior_predictive_plugin(params_sgd, batch)

fig, ax = plot_acc_vs_confidence(pred_fn, test_batch)
plt.savefig('acc-vs-conf-sgd.pdf')
plt.show()


# + [markdown] id="bArOtdLYC3re"
# ## SGLD

# + id="30wKQSDNC4l9"

def posterior_predictive_bayes(params_sampled, batch):
  """computes the posterior_predictive P(class = c | inputs, params) using a histogram
  """
  pred_fn = lambda p:net.apply(p, jax.random.PRNGKey(0), batch) 
  pred_fn = jax.vmap(pred_fn)

  logit_samples = pred_fn(params_sampled) # n_samples x batch_size x n_classes
  pred_samples = jnp.argmax(logit_samples, axis=-1) #n_samples x batch_size

  n_classes = logit_samples.shape[-1]
  batch_size = logit_samples.shape[1]
  probs = np.zeros((batch_size, n_classes))
  for c in range(n_classes):
    idxs = pred_samples == c
    probs[:,c] = idxs.sum(axis=0)

  return probs / probs.sum(axis=1, keepdims=True)


# + colab={"base_uri": "https://localhost:8080/", "height": 279} id="NrUeEvkoDdo-" outputId="a9f101c0-baff-4be1-e548-e8626e12dcf7"
def pred_fn(batch):
  return posterior_predictive_bayes(params_samples, batch)

fig, ax = plot_acc_vs_confidence(pred_fn, test_batch)
plt.savefig('acc-vs-conf-sgld.pdf')
plt.show()

# + [markdown] id="GqINlARzETGs"
# # Distribution shift
#
# We now examine the behavior of the models on the Fashion MNIST dataset.
# We expect the predictions to be much less confident, since the inputs are now 'out of distribution'. We will see that this is true for the Bayesian approach, but not for the plugin approximation. 

# + id="5NkcBnZ-EhMG"

fashion_ds = tfds.load('fashion_mnist:3.*.*', split="test").cache().repeat()
fashion_test_batches = tfds.as_numpy(fashion_ds.batch(10_000))
fashion_test_batches = iter(fashion_test_batches)

fashion_batch = next(fashion_test_batches)


# + [markdown] id="ztiLGAp7Eqxt"
# ## SGD
#
# We see that the plugin estimate is confident (but wrong!) on many of the predictions, which is undesirable.
# If consider a confidence threshold of 0.6, 
# the plugin approach predicts on about 80% of the examples,
# even though the accuracy is only about 6% on these.

# + colab={"base_uri": "https://localhost:8080/", "height": 279} id="CISeZjWmEre2" outputId="285ac006-24ba-4409-fb18-2157e7bd96ca"

def pred_fn(batch):
  return posterior_predictive_plugin(params_sgd, batch)

fig, ax = plot_acc_vs_confidence(pred_fn, fashion_batch)
plt.savefig('acc-vs-conf-sgd-fashion.pdf')
plt.show()


# + [markdown] id="qwBM58OJGUgO"
# ## SGLD
#
# If consider a confidence threshold of 0.6, 
# the Bayesian approach predicts on less than 20% of the examples,
# on which the accuracy is ~4%.
#

# + colab={"base_uri": "https://localhost:8080/", "height": 279} id="8uJGI2D7GVNG" outputId="e330dabb-91f0-422c-fd59-103f53aae8f4"
def pred_fn(batch):
  return posterior_predictive_bayes(params_samples, batch)

fig, ax = plot_acc_vs_confidence(pred_fn, fashion_batch)
plt.savefig('acc-vs-conf-sgld-fashion.pdf')
plt.show()
