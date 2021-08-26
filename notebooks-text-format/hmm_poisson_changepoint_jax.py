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

# + [markdown] id="CerscXVN31Wi"
# # HMM with Poisson observations for detecting changepoints in the rate of a signal

# + [markdown] id="WlLkQCk_34d6"
# This notebook is based on the
# [Multiple Changepoint Detection and Bayesian Model Selection Notebook of TensorFlow](https://www.tensorflow.org/probability/examples/Multiple_changepoint_detection_and_Bayesian_model_selection)

# + id="HctuFIVMz5BA"
from IPython.utils import io
with io.capture_output() as captured:
  # !pip install distrax
  # !pip install flax

# + id="75-fa7vO506c"
# !git clone https://github.com/probml/pyprobml /pyprobml &> /dev/null
# %cd -q /pyprobml/scripts

# + id="n6VSJM1gzlM0"
import logging
logging.getLogger('absl').setLevel(logging.CRITICAL)

# + id="eceu-i1bPOHZ"
from hmm_lib import HMM, hmm_forwards, hmm_forwards_backwards, hmm_viterbi
import numpy as np

import jax
from jax.random import split, PRNGKey
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.experimental import optimizers

import tensorflow_probability as tfp

from matplotlib import pylab as plt
# %matplotlib inline
import scipy.stats

import distrax

# + [markdown] id="U5D7a3SA4lBP"
# ## Data

# + [markdown] id="jFyTZElr4p-K"
# The synthetic data corresponds to a single time series of counts, where the rate of the underlying generative process changes at certain points in time.

# + colab={"base_uri": "https://localhost:8080/", "height": 265} id="8TQh8kvg3Pnn" outputId="0cdf2b23-f9f4-496f-9860-d012aa8e47e8"
true_rates = [40, 3, 20, 50]
true_durations = [10, 20, 5, 35]
random_state = 0

observed_counts = jnp.concatenate([
  scipy.stats.poisson(rate).rvs(num_steps, random_state=random_state)
    for (rate, num_steps) in zip(true_rates, true_durations)
]).astype(jnp.float32)

plt.plot(observed_counts);


# + [markdown] id="ZFR_nwenIqfZ"
# ## Model with fixed $K$

# + [markdown] id="wIfy7113It50"
# To model the changing Poisson rate, we use an HMM. We initially assume the number of states is known to be $K=4$. Later we will try comparing HMMs with different $K$.
#
# We fix the initial state distribution to be uniform, and fix the transition matrix to be the following, where we set $p=0.05$:
#
# $$ \begin{align*} z_1 &\sim \text{Categorical}\left(\left\{\frac{1}{4}, \frac{1}{4}, \frac{1}{4}, \frac{1}{4}\right\}\right)\\ z_t | z_{t-1} &\sim \text{Categorical}\left(\left\{\begin{array}{cc}p & \text{if } z_t = z_{t-1} \\ \frac{1-p}{4-1} & \text{otherwise}\end{array}\right\}\right) \end{align*}$$

# + id="N-qzSZ5Pf_ff"
def build_latent_state(num_states, max_num_states, daily_change_prob):
  # Give probability 0 to states outside of the current model.
  def prob(s):
    return jnp.where(s < num_states + 1, 1/num_states, 0.)
 
  states = jnp.arange(1, max_num_states+1)
  initial_state_probs = vmap(prob)(states)
 
  # Build a transition matrix that transitions only within the current
  # `num_states` states.
  def transition_prob(i, s):
      return jnp.where((s <= num_states) & (i <= num_states) & (1<num_states), 
                      jnp.where(s == i, 1-daily_change_prob, daily_change_prob/(num_states-1)), 
                      jnp.where(s == i, 1, 0))
      
  transition_probs = vmap(transition_prob, in_axes=(None, 0))(states, states)
 
  return initial_state_probs, transition_probs


# + colab={"base_uri": "https://localhost:8080/"} id="9kj3K0ZNMgMt" outputId="9f122556-3f6d-447b-cfe8-3fa707266ebf"
num_states = 4
daily_change_prob = 0.05

initial_state_probs, transition_probs = build_latent_state(num_states, num_states, daily_change_prob)
print("Initial state probs:\n{}".format(initial_state_probs))
print("Transition matrix:\n{}".format(transition_probs))


# + [markdown] id="pFN6ke-9I8W9"
# Now we create an HMM where the observation distribution is a Poisson with learnable parameters. We specify the parameters in log space and initialize them to random values around the log of the overall mean count (to set the scal

# + id="CI6WIVwZm8Jg"
def make_hmm(trainable_log_rates, transition_probs, initial_state_probs):
  hmm = HMM(
    obs_dist=distrax.as_distribution(tfp.substrates.jax.distributions.Poisson(log_rate=trainable_log_rates)),
    trans_dist=distrax.Categorical(probs=transition_probs),
    init_dist=distrax.Categorical(probs=initial_state_probs))
  return hmm


# + id="jt6SzFviNWt4"
rng_key = PRNGKey(0)
rng_key, rng_normal, rng_poisson = split(rng_key, 3)

# Define variable to represent the unknown log rates.
trainable_log_rates = jnp.log(jnp.mean(observed_counts)) + jax.random.normal(rng_normal, (num_states,))
hmm = make_hmm(trainable_log_rates, transition_probs, initial_state_probs)


# + [markdown] id="wXJ7uij9JFLF"
# ## Model fitting using Gradient Descent

# + [markdown] id="yPZ-z0caJScE"
# We compute a MAP estimate of the Poisson rates $\lambda$ using batch gradient descent, using the Adam optimizer applied to the log likelihood (from the HMM) plus the log prior for $p(\lambda)$.

# + id="Er0w6zaaThZg"
def loss_fn(trainable_log_rates, transition_probs, initial_state_probs):
  cur_hmm = make_hmm(trainable_log_rates, transition_probs, initial_state_probs)
  return -(jnp.sum(rate_prior.log_prob(jnp.exp(trainable_log_rates))) + hmm_forwards(cur_hmm, observed_counts)[0])

def update(i, opt_state, transition_probs, initial_state_probs):
  params = get_params(opt_state)
  loss, grads = jax.value_and_grad(loss_fn)(params, transition_probs, initial_state_probs)
  return opt_update(i, grads, opt_state), loss 

def fit(trainable_log_rates, transition_probs, initial_state_probs, n_steps):
  opt_state = opt_init(trainable_log_rates)
  def train_step(opt_state, step):
    opt_state, loss = update(step, opt_state, transition_probs, initial_state_probs)
    return opt_state, loss

  steps = jnp.arange(n_steps)
  opt_state, losses = lax.scan(train_step, opt_state, steps)
  
  return get_params(opt_state), losses


# + colab={"base_uri": "https://localhost:8080/"} id="gSjyTtkDrOHu" outputId="93bba946-c418-40ff-c191-7dab6895db91"
rate_prior = distrax.LogStddevNormal(5, 5)
opt_init, opt_update, get_params = optimizers.adam(1e-1)

n_steps = 201
params, losses = fit(trainable_log_rates, transition_probs, initial_state_probs, n_steps)
rates = jnp.exp(params)
hmm = hmm.replace(obs_dist = distrax.as_distribution(tfp.substrates.jax.distributions.Poisson(log_rate=params)))

# + colab={"base_uri": "https://localhost:8080/"} id="TljgKrxri_Px" outputId="a27a9881-489c-41d2-c912-9b541d551f23"
print("Inferred rates: {}".format(rates))
print("True rates: {}".format(true_rates))

# + id="1qEQfk0WJqGW" colab={"base_uri": "https://localhost:8080/", "height": 265} outputId="ea886c94-c35b-4a31-9878-e96b3dcc393e"
plt.plot(losses)
plt.ylabel('Negative log marginal likelihood');

# + [markdown] id="2E6o_kGKJ81Z"
# We see that the method learned a good approximation to the true (generating) parameters, up to a permutation of the states (since the labels are unidentifiable). However, results can vary with different random seeds. We may find that the rates are the same for some states, which means those states are being treated as identical, and are therefore redundant.

# + [markdown] id="fM_JX-feJ_pG"
# ## Plotting the posterior over states

# + id="1EO0gw2klz5Z" colab={"base_uri": "https://localhost:8080/"} outputId="fe0f4a78-88a7-4d89-bd0c-33917d58cbd8"
_, _, posterior_probs, _ = hmm_forwards_backwards(hmm, observed_counts)


# + id="oZ7C937t-Xh3" colab={"base_uri": "https://localhost:8080/", "height": 729} outputId="f9f40200-33f0-4ce3-bf0f-4d287885b5ef"
def plot_state_posterior(ax, state_posterior_probs, title):
  ln1 = ax.plot(state_posterior_probs, c='tab:blue', lw=3, label='p(state | counts)')
  ax.set_ylim(0., 1.1)
  ax.set_ylabel('posterior probability')
  ax2 = ax.twinx()
  ln2 = ax2.plot(observed_counts, c='black', alpha=0.3, label='observed counts')
  ax2.set_title(title)
  ax2.set_xlabel("time")
  lns = ln1+ln2
  labs = [l.get_label() for l in lns]
  ax.legend(lns, labs, loc=4)
  ax.grid(True, color='white')
  ax2.grid(False)

fig = plt.figure(figsize=(10, 10))
plot_state_posterior(fig.add_subplot(2, 2, 1),
                     posterior_probs[:, 0],
                     title="state 0 (rate {:.2f})".format(rates[0]))
plot_state_posterior(fig.add_subplot(2, 2, 2),
                     posterior_probs[:, 1],
                     title="state 1 (rate {:.2f})".format(rates[1]))
plot_state_posterior(fig.add_subplot(2, 2, 3),
                     posterior_probs[:, 2],
                     title="state 2 (rate {:.2f})".format(rates[2]))
plot_state_posterior(fig.add_subplot(2, 2, 4),
                     posterior_probs[:, 3],
                     title="state 3 (rate {:.2f})".format(rates[3]))
plt.tight_layout()

# + id="oBhqUUBUKV7Q" colab={"base_uri": "https://localhost:8080/"} outputId="5baa7f49-4330-4857-82de-752e6a75b21f"
print(rates)

# + id="GSFx4aCb0lZU"
# max marginals
most_probable_states = np.argmax(posterior_probs, axis=1)
most_probable_rates = rates[most_probable_states]

# + id="84hUzqYn2Nky" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="63a056d7-a8df-4b84-b6c6-076c65fb4166"
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1)
ax.plot(most_probable_rates, c='tab:green', lw=3, label='inferred rate')
ax.plot(observed_counts, c='black', alpha=0.3, label='observed counts')
ax.set_ylabel("latent rate")
ax.set_xlabel("time")
ax.set_title("Inferred latent rate over time")
ax.legend(loc=4);

# + id="Cv5_vI8d2Pkf"
# max probaility trajectory (Viterbi)
most_probable_states = hmm_viterbi(hmm, observed_counts)
most_probable_rates = rates[most_probable_states]

# + id="g8kKmL5f2vAf" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="00f91e89-24b1-4905-f365-7dfe7deeec2c"
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1)
color_list = np.array(['tab:red', 'tab:green', 'tab:blue', 'k'])
colors = color_list[most_probable_states]
for i in range(len(colors)):
  ax.plot(i, most_probable_rates[i], '-o', c=colors[i], lw=3, alpha=0.75)
ax.plot(observed_counts, c='black', alpha=0.3, label='observed counts')
ax.set_ylabel("latent rate")
ax.set_xlabel("time")
ax.set_title("Inferred latent rate over time");

# + [markdown] id="_RnpDkTKK4el"
# ## Model with unknown $K$

# + [markdown] id="pd-iztMNLBIb"
# In general we don't know the true number of states. One way to select the 'best' model is to compute the one with the maximum marginal likelihood. Rather than summing over both discrete latent states and integrating over the unknown parameters $\lambda$, we just maximize over the parameters (empirical Bayes approximation).
#
# $$p(x_{1:T}|K) \approx \max_\lambda \int p(x_{1:T}, z_{1:T} | \lambda, K) dz$$
# We can do this by fitting a bank of separate HMMs in parallel, one for each value of $K$. We need to make them all the same size so we can batch them efficiently. To do this, we pad the transition matrices (and other paraemeter vectors) so they all have the same shape, and then use masking.

# + id="628zP3kx2hg6" colab={"base_uri": "https://localhost:8080/"} outputId="67f73e9c-4e9f-4d01-9793-767666c21322"
max_num_states = 6
states = jnp.arange(1, max_num_states + 1)

# For each candidate model, build the initial state prior and transition matrix.
batch_initial_state_probs, batch_transition_probs = vmap(build_latent_state, in_axes=(0, None, None))(states, max_num_states, daily_change_prob)
 
print("Shape of initial_state_probs: {}".format(batch_initial_state_probs.shape))
print("Shape of transition probs: {}".format(batch_transition_probs.shape))
print("Example initial state probs for num_states==3:\n{}".format(batch_initial_state_probs[2, :]))
print("Example transition_probs for num_states==3:\n{}".format(batch_transition_probs[2, :, :]))

# + id="4_via5B93S0T"
rng_key, rng_normal = split(rng_key)

# Define variable to represent the unknown log rates.
trainable_log_rates = jnp.log(jnp.mean(observed_counts)) + jax.random.normal(rng_normal, (max_num_states,))

# + [markdown] id="kElabi3wjiRf"
# ## Model fitting with gradient descent

# + id="3rOAqjPdjLWV"
n_steps = 201
params, losses = vmap(fit, in_axes=(None, 0, 0, None))(trainable_log_rates, batch_transition_probs, batch_initial_state_probs, n_steps)
rates = jnp.exp(params)

# + colab={"base_uri": "https://localhost:8080/", "height": 266} id="IuqR_1P5kBBX" outputId="58b320e8-36b1-4a85-f7f9-a2ca721ac8a3"
plt.plot(losses.T);
plt.ylabel('Negative log marginal likelihood');

# + [markdown] id="7FTjbb4Qj1pQ"
# ## Plot marginal likelihood of each model

# + id="fe7CQTO9OuA8" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="e0bf5eab-449b-4bf2-9b0a-5a5fe91d0088"
plt.plot(-losses[:, -1])
plt.ylim([-400, -200])
plt.ylabel("marginal likelihood $\\tilde{p}(x)$")
plt.xlabel("number of latent states")
plt.title("Model selection on latent states");

# + [markdown] id="7aJLgl4NkegW"
# ## Plot posteriors

# + id="Lljaow1JOw1H" colab={"base_uri": "https://localhost:8080/"} outputId="617b5559-8d3a-4a90-8279-4731b648e937"
for i, learned_model_rates in enumerate(rates):
  print("rates for {}-state model: {}".format(i+1, learned_model_rates[:i+1]))


# + id="y7OEgXe6P5ro" colab={"base_uri": "https://localhost:8080/"} outputId="71fe63b7-7e6b-44e7-af28-a23fde5efb48"
def posterior_marginals(trainable_log_rates, initial_state_probs, transition_probs):
  hmm = make_hmm(trainable_log_rates, transition_probs, initial_state_probs)
  return hmm_forwards_backwards(hmm, observed_counts)[2]

posterior_probs = vmap(posterior_marginals, in_axes=(0, 0, 0))(params, batch_initial_state_probs, batch_transition_probs)
most_probable_states = jnp.argmax(posterior_probs, axis=-1)

# + id="r8aaKsNiRAxs" colab={"base_uri": "https://localhost:8080/", "height": 462} outputId="f7a975c2-f9ae-483f-fbcd-42a829900d9f"
fig = plt.figure(figsize=(14, 12))
for i, learned_model_rates in enumerate(rates):
  ax = fig.add_subplot(4, 3, i+1)
  ax.plot(learned_model_rates[most_probable_states[i]], c='green', lw=3, label='inferred rate')
  ax.plot(observed_counts, c='black', alpha=0.3, label='observed counts')
  ax.set_ylabel("latent rate")
  ax.set_xlabel("time")
  ax.set_title("{}-state model".format(i+1))
  ax.legend(loc=4)
plt.tight_layout()
