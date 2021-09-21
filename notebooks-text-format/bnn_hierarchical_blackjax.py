# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: blackjax
#     language: python
#     name: blackjax
# ---

# + [markdown] id="0qS2Qh3_E6sv"
# # Hierarchical Bayesian neural networks
#
# Code is based on  [This blog post](https://twiecki.io/blog/2018/08/13/hierarchical_bayesian_neural_network/) by Thomas Wiecki.
# [Original PyMC3 Notebook](https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/bayesian_neural_network_hierarchical.ipynb). Converted to Blackjax by Aleyna Kara (@karalleyna) and Kevin Murphy (@murphyk). (For a Numpyro version, see [here](https://github.com/probml/probml-notebooks/blob/main/notebooks/bnn_hierarchical_numpyro.ipynb).)
#
# We create T=18 different versions of the "two moons" dataset, each rotated by a different amount. These correspond to T different nonlinear binary classification "tasks" that we have to solve. We only get 50 samples from each each task, so solving them separately (with T independent multi layer perceptrons) will result in poor performance. If we pool all the data, and fit a single MLP, we also get poor performance. But if we use a hierarchical Bayesian model, with one MLP per task, and one learned prior MLP,  we will get better results, as we will see.
#
#
#
#
#

# + [markdown] id="slZekdiXVd-r"
# ## Setup

# + colab={"base_uri": "https://localhost:8080/"} id="8237SKPQf5IT" outputId="44a8e51e-4590-4b27-8489-f698ee6cccbd"
# !pip install blackjax
# !pip install distrax

# + id="XUdWX2RJFQ-2"
from warnings import filterwarnings

import jax
from jax import vmap, jit
import jax.numpy as jnp
from jax.random import PRNGKey, split, normal
import jax.random as random
import numpy as np

import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup

import distrax
import tensorflow_probability.substrates.jax.distributions as tfd

import sklearn
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial

filterwarnings('ignore')
sns.set_style('white')

cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
cmap_uncertainty = sns.cubehelix_palette(light=1, as_cmap=True)

# + [markdown] id="CSRICnGGJl2x"
# ## Data
#
# We create T=18 different versions of the "two moons" dataset, each rotated by a different amount. These correspond to T different binary classification "tasks" that we have to solve. 
#

# + colab={"base_uri": "https://localhost:8080/", "height": 266} id="WmId3FYXsdDe" outputId="c4284396-96bf-4406-8e94-62c710910a5f"
X, Y = make_moons(noise=0.3, n_samples=1000)
plt.scatter(X[Y==0, 0], X[Y==0, 1], label='Class 0')
plt.scatter(X[Y==1, 0], X[Y==1, 1], color='r', label='Class 1')
sns.despine(); plt.legend();

# + id="R4AUifO4rp0Z"

n_groups = 18 

n_grps_sq = int(np.sqrt(n_groups))
n_samples = 100


# + id="edzKr4JQsG37"
def rotate(X, deg):
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix([[c, -s], [s, c]])

    X = X.dot(R)
    
    return np.asarray(X)


# + id="grOX0OwPsdDf"
np.random.seed(31)

Xs, Ys = [], []
for i in range(n_groups):
    # Generate data with 2 classes that are not linearly separable
    X, Y = make_moons(noise=0.3, n_samples=n_samples)
    X = scale(X)
    
    # Rotate the points randomly for each category
    rotate_by = np.random.randn() * 90.
    X = rotate(X, rotate_by)
    Xs.append(X)
    Ys.append(Y)

# + id="fv_rIsxFrtCD" colab={"base_uri": "https://localhost:8080/"} outputId="23e4b7df-8e7c-4b01-c445-16745da2e78d"
Xs = jnp.stack(Xs)
Ys = jnp.stack(Ys)

Xs_train = Xs[:, :n_samples // 2, :]
Xs_test = Xs[:, n_samples // 2:, :]
Ys_train = Ys[:, :n_samples // 2]
Ys_test = Ys[:, n_samples // 2:]

# + id="12wnn6bjFQ-2" colab={"base_uri": "https://localhost:8080/", "height": 730} outputId="05f5d0f1-33de-49c4-b145-c6e2232df8aa"
fig, axs = plt.subplots(figsize=(15, 12), nrows=n_grps_sq, ncols=n_grps_sq, 
                        sharex=True, sharey=True)
axs = axs.flatten()
for i, (X, Y, ax) in enumerate(zip(Xs_train, Ys_train, axs)):
    ax.scatter(X[Y==0, 0], X[Y==0, 1], label='Class 0')
    ax.scatter(X[Y==1, 0], X[Y==1, 1], color='r', label='Class 1')
    sns.despine(); ax.legend()
    ax.set(title='Category {}'.format(i + 1), xlabel='X1', ylabel='X2')

# + id="Y5ZjQ2yFQQqR"
grid = jnp.mgrid[-3:3:100j, -3:3:100j].reshape((2, -1)).T
grid_3d = jnp.repeat(grid[None, ...], n_groups, axis=0)


# + [markdown] id="Rz2HftvwYgsx"
# ## Utility functions for training and testing

# + id="comparative-trinity"
def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


# + id="7vhCkG63zfDR"
def get_predictions(model, samples, X, n_hidden_layers, rng_key, num_samples):
  samples_flattened, tree_def = jax.tree_flatten(samples)
  keys = random.split(rng_key, num_samples)
  predictions = []
  
  for i, key in enumerate(keys):
    params = {}
    for j, k in enumerate(samples.keys()):
      params[k] = samples_flattened[j][i]

    z = model(params, X, n_hidden_layers)
    Y = distrax.Bernoulli(logits=z).sample(seed=key)
    predictions.append(Y[None, ...])

  return jnp.vstack(predictions)


# + id="0wD5Rxsg0aLv"
def get_mean_predictions(predictions, threshold=0.5):
  # compute mean prediction and confidence interval around median
  mean_prediction = jnp.mean(predictions, axis=0)
  return mean_prediction > threshold


# + id="P_PO9q7fOByr"
def fit_and_eval(rng_key, model, potential_fn, X_train, Y_train, X_test, grid, n_groups=None):
    init_key, warmup_key, inference_key, train_key, test_key, grid_key = split(rng_key, 6)
    
    # initialization
    potential = partial(potential_fn, X=X_train, Y=Y_train, model=model, n_hidden_layers=n_hidden_layers)
    initial_position = init_bnn_params(layer_widths, init_key) if n_groups is None else init_hierarchical_params(layer_widths, n_groups, init_key) 
    initial_state = nuts.new_state(initial_position, potential)
    
    kernel_generator = lambda step_size, inverse_mass_matrix: nuts.kernel(potential, step_size, inverse_mass_matrix)
    
    # warm up
    final_state, (step_size, inverse_mass_matrix), info = stan_warmup.run(
        warmup_key,
        kernel_generator,
        initial_state,
        num_warmup)
        
    # inference
    nuts_kernel = jax.jit(nuts.kernel(potential, step_size, inverse_mass_matrix))
    states = inference_loop(inference_key, nuts_kernel, initial_state, num_samples)
    samples = states.position
    
    # evaluation
    predictions = get_predictions(model, samples, X_train, n_hidden_layers , train_key, num_samples)
    Y_pred_train = get_mean_predictions(predictions)

    predictions = get_predictions(model, samples, X_test, n_hidden_layers, test_key, num_samples)
    Y_pred_test = get_mean_predictions(predictions)

    pred_grid = get_predictions(model, samples, grid, n_hidden_layers, grid_key, num_samples)

    return Y_pred_train, Y_pred_test, pred_grid


# + [markdown] id="PjWMMOkbaZFZ"
# ## Hyperparameters
#
# We use an MLP with 2 hidden layers, each with 5 hidden units.
#

# + id="1jkXKkRLLZq9"
# MLP params
layer_widths = [Xs_train.shape[-1], 5, 5, 1]
n_hidden_layers = len(layer_widths) - 2

# + id="f8f1951c"
# MCMC params

num_warmup = 1000
num_samples = 500 


# + [markdown] id="lOEOkbQ9Y-mY"
# ## Fit separate MLPs, one per task
#
# Let $w_{tijl}$ be  the weight for node $i$ to node $j$ in layer $l$ in task $t$. We assume
# $$
# w_{tijl} \sim N(0,1) 
# $$
# and compute the posterior for all the weights.
#

# + id="07MMp8LzObLp"
def init_bnn_params(layer_widths, rng_key):
  rng_key, *keys = split(rng_key, len(layer_widths))
  params = {}
  
  for i, (n_in, n_out, key) in enumerate(zip(layer_widths[:-1], layer_widths[1:], keys)):
    params[f"w_{i}"] = distrax.Normal(0,1).sample(seed=key, sample_shape=(n_in, n_out))
  
  return params



# + id="_o1WI4nawfGx"
def bnn(params, X, n_hidden_layers):
    z = X
    
    for i in range(n_hidden_layers + 1):
      z = z @ params[f"w_{i}"]
      z = jax.nn.tanh(z) if i != n_hidden_layers else z

    z = z.squeeze(-1)
    return z


# + id="dxETp5eVwN8v"
def potential_fn_of_bnn(params, X, Y, model, n_hidden_layers):
    log_joint = 0

    for i in range(n_hidden_layers + 1):
        log_joint += distrax.Normal(0., 1.).log_prob(params[f"w_{i}"]).sum()
    
    z = model(params, X, n_hidden_layers)
    loglikelihood = distrax.Bernoulli(logits=z).log_prob(Y).sum()
    log_joint += loglikelihood
 
    return -jnp.sum(log_joint)


# + id="MVf9MDs9QUrk"
rng_key = PRNGKey(0)
keys = split(rng_key, n_groups)

def fit_and_eval_single_mlp(key, X_train, Y_train, X_test):
  return fit_and_eval(key, bnn, potential_fn_of_bnn, X_train, Y_train, X_test, grid, n_groups=None)

Ys_pred_train, Ys_pred_test, ppc_grid_single = vmap(fit_and_eval_single_mlp)(keys, Xs_train, Ys_train, Xs_test)

# + [markdown] id="V7SCr0O5MwnT"
# ### Results
#
# Accuracy is reasonable, but the decision boundaries are nearly linear (thanks to the Bayes Occam's razor effect), and have not captured the underlying Z pattern in the data, due to having too little data per task.

# + colab={"base_uri": "https://localhost:8080/"} id="WkLfyqSySBP_" outputId="30791a27-031b-42e3-d578-e88c7ff18642"
print ("Train accuracy = {:.2f}%".format(100*jnp.mean(Ys_pred_train == Ys_train)))

# + colab={"base_uri": "https://localhost:8080/"} id="-8Fi_c4USBQA" outputId="65afe0dc-bd2f-4546-9825-703d40c1c9fc"
print ("Test accuracy = {:.2f}%".format(100*jnp.mean(Ys_pred_test == Ys_test)))


# + id="8EE3dHbu9rcq"
def plot_decision_surfaces_non_hierarchical(nrows=2, ncols=2):
  fig, axes = plt.subplots(figsize=(15, 12), nrows=nrows, ncols=ncols, sharex=True, sharey=True)
  axes = axes.flatten()
  for i, (X, Y_pred, Y_true, ax) in enumerate(zip(Xs_train, Ys_pred_train, Ys_train, axes)):
      contour = ax.contourf(grid[:, 0].reshape(100, 100), grid[:, 1].reshape(100, 100), ppc_grid_single[i, ...].mean(axis=0).reshape(100, 100), cmap=cmap)
      ax.scatter(X[Y_true == 0, 0], X[Y_true == 0, 1], label='Class 0')
      ax.scatter(X[Y_true == 1, 0], X[Y_true == 1, 1], color='r', label='Class 1')
      sns.despine(); ax.legend()


# + [markdown] id="VL4IP1tzKO_R"
# Below we show that the decision boundaries do not look reasonable, since there is not enough data to fit each model separately.
#

# + colab={"base_uri": "https://localhost:8080/", "height": 704} id="s8xZ4szTxXhE" outputId="68d66ff2-8186-493b-f3a4-0b28086d01b4"
plot_decision_surfaces_non_hierarchical(nrows=n_grps_sq, ncols=n_grps_sq)

# + colab={"base_uri": "https://localhost:8080/", "height": 704} id="T4dE9O-3-Afd" outputId="c083e0ec-004f-4a26-8920-25d57bd1199a"
plot_decision_surfaces_non_hierarchical()


# + [markdown] id="b7ymwtSpWJoH"
# ## Hierarchical Model
#
# Now we use a hierarchical Bayesian model, which has a common Gaussian prior for all the weights, but allows each task to have its own task-specific parameters. More precisely, let $w_{tijl}$ be  the weight for node $i$ to node $j$ in layer $l$ in task $t$. We assume
# $$
# w_{tijl} \sim N(\mu_{ijl}, \sigma_l) 
# $$
#
# $$
# \mu_{ijl} \sim N(0,1) 
# $$
#
# $$
# \sigma_l \sim N_+(0,1)
# $$
#
# or, in non-centered form,
# $$
# w_{tijl} = \mu_{ijl} + \epsilon_{tijl} \sigma_l
# $$

# + id="W5VibtfvJf_y"
def init_hierarchical_params(layer_widths, n_groups, rng_key):
  half_normal = distrax.as_distribution(tfd.HalfNormal(1.0))
  rng_key, *keys = split(rng_key, len(layer_widths))
  params = {}
  for i, (n_in, n_out, key) in enumerate(zip(layer_widths[:-1], layer_widths[1:], keys)):
    mu_key, std_key, eps_key = split(key, 3)
    params[f"w_{i}_mu"] = distrax.Normal(0,1).sample(seed=mu_key, sample_shape=(n_in, n_out))
    params[f"w_{i}_std"] = half_normal.sample(seed=std_key, sample_shape=(1,))
    params[f"w_{i}_eps"] = distrax.Normal(0,1).sample(seed=eps_key, sample_shape=(n_groups, n_in, n_out))
    
  return params


# + id="QfaIM3ofx4dT"
def hierarchical_model(params, X, n_hidden_layers):
    n_groups, _, input_dim = X.shape
    output_dim = 1

    z = X
    
    for i in range(n_hidden_layers + 1):
      w = params[f"w_{i}_mu"] +  params[f"w_{i}_eps"] * params[f"w_{i}_std"]
      z = z @ w
      z = jax.nn.tanh(z) if i != n_hidden_layers else z

    z = z.squeeze(-1)
    return z


# + id="UBjAfS0x9ZdQ"
def potential_fn_of_hierarchical_model(params, X, Y, model, n_hidden_layers):
    log_joint = 0
    half_normal = distrax.as_distribution(tfd.HalfNormal(1.0))
    
    for i in range(n_hidden_layers + 1):
        log_joint += distrax.Normal(0., 1.0).log_prob(params[f"w_{i}_mu"]).sum()
        log_joint += half_normal.log_prob(params[f"w_{i}_std"]).sum()
        log_joint += distrax.Normal(0., 1.).log_prob(params[f"w_{i}_eps"]).sum()
    
    z = hierarchical_model(params, X, n_hidden_layers)
    loglikelihood = distrax.Bernoulli(logits=z).log_prob(Y).sum()
    log_joint += loglikelihood
 
    return -jnp.sum(log_joint)


# + id="tq9AGXQXaRjA"
rng_key = PRNGKey(0)
Ys_hierarchical_pred_train, Ys_hierarchical_pred_test, ppc_grid = fit_and_eval(rng_key, hierarchical_model, potential_fn_of_hierarchical_model, Xs_train, Ys_train, Xs_test, grid_3d, n_groups=n_groups)

# + [markdown] id="lraVhcUhMmt6"
# ### Results
#
# We see that the train and test accuracy are higher, and the decision boundaries all have the shared "Z" shape, as desired.
#

# + colab={"base_uri": "https://localhost:8080/"} id="qp6cgurS6Vbk" outputId="4162e1a8-c45a-46fb-f258-64f9564d48b4"
print ("Train accuracy = {:.2f}%".format(100*jnp.mean(Ys_hierarchical_pred_train == Ys_train)))

# + id="MHD0BU_iFQ-3" colab={"base_uri": "https://localhost:8080/"} outputId="0df37165-75f0-4e2a-bfe9-5fb6ed220050"
print ("Test accuracy = {:.2f}%".format(100*jnp.mean(Ys_hierarchical_pred_test == Ys_test)))


# + id="4ScIKzIkZ-hE"
def plot_decision_surfaces_hierarchical(nrows=2, ncols=2):
  fig, axes = plt.subplots(figsize=(15, 12), nrows=nrows, ncols=ncols, sharex=True, sharey=True)

  for i, (X, Y_pred, Y_true, ax) in enumerate(zip(Xs_train, Ys_hierarchical_pred_train, Ys_train, axes.flatten())):
      contour = ax.contourf(grid[:, 0].reshape((100, 100)), grid[:, 1].reshape((100, 100)), ppc_grid[:, i, :].mean(axis=0).reshape(100, 100), cmap=cmap)
      ax.scatter(X[Y_true == 0, 0], X[Y_true == 0, 1], label='Class 0')
      ax.scatter(X[Y_true == 1, 0], X[Y_true == 1, 1], color='r', label='Class 1')
      sns.despine(); ax.legend()


# + colab={"base_uri": "https://localhost:8080/", "height": 704} id="kuej5KRzxf_I" outputId="3f402c0f-eef9-4626-d14f-6bf88146992b"
plot_decision_surfaces_hierarchical(nrows=n_grps_sq, ncols=n_grps_sq)

# + colab={"base_uri": "https://localhost:8080/", "height": 704} id="67UtkWu-aAer" outputId="924a146e-ee0c-44b8-abcd-a7f4646a2b15"
plot_decision_surfaces_hierarchical()
