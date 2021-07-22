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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/vb_gmm_tfp.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="0oC36OzLSJ2f"
# # Variational Bayes for Gaussian Mixture Models using TFP
#
# Code is written by [Dave Moore](https://davmre.github.io/), with some tweaks by [Kevin Murphy](https://www.cs.ubc.ca/~murphyk/).
#
# We use a diagonal Gaussian approximation to the posterior (after transforming the variables) using  SVI objective, optimized with full batch gradient descent. See [here](https://github.com/probml/pyprobml/blob/master/scripts/variational_mixture_gaussians_demo.py) for code that implements full-batch VBEM using a conjugate prior.
#

# + colab={"base_uri": "https://localhost:8080/"} id="lua_1gsqSFbK" outputId="bb8d5fe0-766e-4279-8825-e9ae1f0e4742"
import functools
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
#from matplotlib import pylab as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__, tfp.__version__)

# + [markdown] id="lT1ANmCvb7pz"
# # Plotting code

# + id="oAVkh4N6SO7p"


from matplotlib.patches import Ellipse

def plot_loc_scale(weight_, loc_, scale_tril_, color, ax):
  cov = np.dot(scale_tril_, scale_tril_.T)
  w, v = np.linalg.eig(cov)
  angle = np.arctan2(v[1, 0], v[1, 1]) * 360 / (2*np.pi)
  height = 3 * np.sqrt(w[1])  # minor axis
  width = 3 * np.sqrt(w[0])  # major axis

  e = Ellipse(xy=loc_,
              width=width,
              height=height,
              angle=angle)
  ax.add_artist(e)
  e.set_clip_box(ax.bbox)
  e.set_alpha(weight_)
  e.set_facecolor(color)
  e.set_edgecolor("black")

def plot_posterior_with_data(mix_, loc_, scale_tril_, data, ax, facecolors=None):
  ax.plot(data[:, 0], data[:, 1], 'k.', markersize=3);
  ax.plot(loc_[:, 0], loc_[:, 1], 'r^');

  num_components = len(mix_)
  np.random.seed(420)
  if facecolors is None:
    facecolors = sns.color_palette('deep', n_colors=num_components)

  weights_ = np.power(mix_,0.8) # larger power means less emphasis on low weights
  weights_ = weights_ * (0.5 / np.max(weights_))
  for i, (weight_, l_, st_) in enumerate(zip(weights_, loc_, scale_tril_)):
    plot_loc_scale(weight_, l_, st_, color=facecolors[i], ax=ax)

def plot_posterior_sample(surrogate_posterior, data):
  fig = plt.figure(figsize=(10, 6), constrained_layout=True)
  gs = fig.add_gridspec(4, 4)
  mix, loc, _, _, scale_tril = surrogate_posterior.sample()
  num_components = len(mix)
  plot_posterior_with_data(mix.numpy(),
                          loc.numpy(),
                          scale_tril.numpy(),
                          data=data,
                          ax=fig.add_subplot(gs[:, :3]))

  ax = fig.add_subplot(gs[:1, 3])
  sns.barplot(x=np.arange(num_components), y=mix.numpy(), ax=ax, palette='deep')
  ax.set_title('Mixture component weights')


# + [markdown] id="BOVmUarAcmeh"
# # Data
#
# We use a datset of erruption times from the "Old Faithful" geyser in Yellowstone National Park. 

# + colab={"base_uri": "https://localhost:8080/"} id="QXaWed2hdW2X" outputId="2570df23-8466-48fa-d821-605952bb91a3"
url = 'https://raw.githubusercontent.com/probml/probml-data/main/data/faithful.txt'
#df = pd.read_csv(url, sep='\t', header=None, columns=['eruptions', 'waiting'])
# !wget $url
data = np.array(np.loadtxt("faithful.txt"))
print(data.shape)

# + colab={"base_uri": "https://localhost:8080/", "height": 296} id="m_BeW_JKb-OY" outputId="4a1c9e1f-e5d4-454c-e2a5-9806d9b81fc7"
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Eruption duration (mins)')
plt.ylabel('Waiting time (mins)')

# + id="994VsIGzhC9N"
# Standardize the data (to simplify model fitting)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data_normalized = (data - mean) / std

# + colab={"base_uri": "https://localhost:8080/", "height": 284} id="53IHUPx9hQQ4" outputId="8014b843-497f-453c-f773-c7618346dc4a"
plt.figure()
plt.scatter(data_normalized[:,0], data_normalized[:,1])


# + [markdown] id="enk_u7Gve9VF"
# # Model
#
# We put a Gaussian prior on each mean vector, an LKJ prior on each correlation matrix, and a half-normal prior on each scale vector. (This is not a conjugate prior.) 

# + id="Kaef8hQFe-YY"
def bayesian_gaussian_mixture_model(num_observations, dims, components):
  mixture_probs = yield tfd.Dirichlet(
      concentration=tf.ones(components, dtype=tf.float32) / components,
      name='mixture_probs')
  loc = yield tfd.Normal(loc=tf.zeros([components, dims]),
                         scale=1,
                         name='loc')
  
  scale = yield tfd.HalfNormal(scale=2 * tf.ones([components, dims]),
                               name='scale')
  correlation_tril = yield tfd.CholeskyLKJ(
      dimension=dims,
      concentration=tf.ones([components]),
      name='correlation_tril')
  scale_tril = yield tfd.Deterministic(
      scale[..., tf.newaxis] * correlation_tril,
      name='scale_tril')

  observations = yield tfd.Sample(
      tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=mixture_probs),
        components_distribution=tfd.MultivariateNormalTriL(
            loc=loc,
            scale_tril=scale_tril)),
      sample_shape=num_observations,
      name='observations')
  


  


# + id="5xXCofGKfnND"
ncomponents = 10
ndata = data.shape[0]
ndims = data.shape[1]

bgmm = tfd.JointDistributionCoroutineAutoBatched(
    functools.partial(bayesian_gaussian_mixture_model,
                      dims=ndims,
                      components=ncomponents,
                      num_observations=ndata))

# + colab={"base_uri": "https://localhost:8080/"} id="q5rEnpI1Gm60" outputId="3bffab34-ab73-411d-c638-b7bf90d9fc77"
print(bgmm.event_shape)
print(bgmm.event_shape._fields)

# + colab={"base_uri": "https://localhost:8080/"} id="jGXsIXG3ckRy" outputId="f991d4fe-57a9-469e-b727-aa19080e3725"
# Sample from the prior predictive joint distribution
x = bgmm.sample() 
print(type(x))
#print(x)
print(x.mixture_probs.shape)
print(x.mixture_probs)
print(x.loc.shape)
print(x.scale.shape)
print(x.correlation_tril.shape)
print(x.scale_tril.shape)
print(x.observations.shape)
print('sample data')
print(x.observations[:5,:])

# + colab={"base_uri": "https://localhost:8080/"} id="P_tSNCjMgM5V" outputId="3fcccf9b-8a5d-4a33-995a-3f828fd43f26"
print(bgmm.log_prob(x))

# + id="HTwE_nJ3gCPp"
# Clamp the observations
pinned = bgmm.experimental_pin(observations=data_normalized)

# + colab={"base_uri": "https://localhost:8080/"} id="wiacMtZNp47v" outputId="415b715c-1fc1-4750-ec81-2af095ae65d8"
print(type(pinned))
print(pinned)

# + colab={"base_uri": "https://localhost:8080/"} id="D6QiiVCMhbdC" outputId="7558877c-e0dd-4782-bd44-4e8dc4d7623e"
# Sample from clamped model

#x = pinned.sample() # does not work 
x = pinned.sample_unpinned() # sample from unnormalized joint
print(x._fields) # observations is excluded
print(x.mixture_probs)

print(pinned.unnormalized_log_prob(x))
#print(bgmm.unnormalized_log_prob(x))

# + [markdown] id="HlEEyrBBh2iG"
# # Fitting a point mass posterior (MAP estimate)
#
# This marginalizes over the discrete latent indicators (as part of MixtureSameFamily logprob computation), but uses point estimates for model parameters, similar to standard EM. Thus there is no "Bayes Occam's razor" penalty factor when choosing too many mixture components.

# + id="Z_UvdSFshkj-"
def trainable_point_estimate(initial_loc, initial_scale, event_ndims, validate_args):
  return tfd.Independent(
      tfd.Deterministic(tf.Variable(initial_loc), validate_args=validate_args),
      reinterpreted_batch_ndims=event_ndims)

point_mass_posterior = tfp.experimental.vi.build_factored_surrogate_posterior(
    pinned.event_shape,
    bijector=pinned.experimental_default_event_space_bijector(),
    trainable_distribution_fn=trainable_point_estimate)

# + colab={"base_uri": "https://localhost:8080/"} id="t8cwgnE3iXju" outputId="0e126753-a178-434f-88a2-bfef79fdc95e"
import time
t0 = time.time()
num_steps = 1000
losses = tfp.vi.fit_surrogate_posterior(
    pinned.unnormalized_log_prob,
    point_mass_posterior,
    optimizer=tf.optimizers.Adam(3e-2),
    num_steps=int(num_steps))
t1 = time.time()
print("{} variational steps finished in {:.3f}s".format(num_steps, t1-t0))

# + colab={"base_uri": "https://localhost:8080/", "height": 298} id="vecaE7mLibTr" outputId="182407ee-3573-453c-9248-89e42b6d1d5d"
# The negative log data likelihood goes down monotonically, as EM theory predicts
plt.plot(losses)
plt.title("Training loss curve")

# + colab={"base_uri": "https://localhost:8080/"} id="6_aEzfBAFu3M" outputId="abb2d6b1-84ba-43a5-a0ff-f85db85bb78e"
print(type(point_mass_posterior))


# + colab={"base_uri": "https://localhost:8080/"} id="vmo-qqebjaeg" outputId="170759e6-b6a1-4ba6-a2c9-f614b1fed5c8"
print(point_mass_posterior)


# + colab={"base_uri": "https://localhost:8080/"} id="2I69tOPnFOPD" outputId="320815cb-09ef-4d06-d62a-ce3919eeff37"
# unconstrained parameters  (before applying bijector e.g., Softplus for a positive-valued scale parameter).
print(point_mass_posterior.trainable_variables)


# + colab={"base_uri": "https://localhost:8080/"} id="BKBbkIWXGGlm" outputId="2a98ac93-dcfd-49a4-bbfd-08bf706da8de"
mix_log_weights = point_mass_posterior.trainable_variables[0]
print(tf.nn.softmax(mix_log_weights))

# + [markdown] id="wCYNnRI7ilcH"
# Samples from the posterior predictive distribution should be constant across sampling runs, since we use a point estimate of the parameters.

# + colab={"base_uri": "https://localhost:8080/"} id="uqM0Ui_XlO_w" outputId="45ec8c95-f207-41e6-a516-8539c3c72b64"
params = point_mass_posterior.sample()
print(params.mixture_probs)

# + colab={"base_uri": "https://localhost:8080/"} id="--7imbQio83J" outputId="d10c6a71-75c4-493f-c486-bdac9f07e9cd"
params = point_mass_posterior.sample()
print(params.mixture_probs)

# + colab={"base_uri": "https://localhost:8080/", "height": 457} id="4wFbLoVhigEn" outputId="7cb1079d-27bf-40dc-e917-6b38a44c1c24"
plot_posterior_sample(point_mass_posterior, data=data_normalized)
plt.savefig('vb_gmm_map_sample.pdf')
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 457} id="9yLKJn9SmgOu" outputId="2508e048-880b-4674-dfd7-c439fc7988c1"
plot_posterior_sample(point_mass_posterior, data=data_normalized)

# + [markdown] id="P6XiifcpmY7r"
# # Fitting a diagonal Gaussian posterior
#
# Construct and fit a surrogate posterior using stochastic gradient VI. The surrogate is a diagonal Gaussian that is transformed into the support of the model's parameters using appropriate bijectors. (The transformed vector is then split into tensors for each of the models RVs, and these are pushed through constraining bijectors as needed.)
# For details, see
# https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/vi/build_affine_surrogate_posterior
#
# The event space for this distribution is derived from the pinned distribution. For details, see https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/distributions/JointDistributionPinned
#

# + id="IwCQc8DXjbnI"
surrogate_posterior = tfp.experimental.vi.build_affine_surrogate_posterior(
    pinned.event_shape,
    bijector=pinned.experimental_default_event_space_bijector(),
    operators='diag')

# Use operators='tril' for full covariance Gaussian


# + colab={"base_uri": "https://localhost:8080/"} id="jrcykGZcmtvD" outputId="08f23f26-3965-453b-8580-822818850e29"
import time
t0 = time.time()
num_steps = 1000
losses = tfp.vi.fit_surrogate_posterior(
    pinned.unnormalized_log_prob,
    surrogate_posterior,
    optimizer=tf.optimizers.Adam(2e-2),
    num_steps=int(num_steps))
t1 = time.time()
print("{} variational steps finished in {:.3f}s".format(num_steps, t1-t0))

# + colab={"base_uri": "https://localhost:8080/", "height": 298} id="GFYqmgsZmue6" outputId="68496bff-f0aa-4b15-cfc3-b13a60102bce"
plt.plot(losses)
plt.title("Training loss curve")

# + colab={"base_uri": "https://localhost:8080/", "height": 457} id="5J9to4m_m9jm" outputId="64dfe3de-c502-48ff-cb43-6d7e81495b74"
plot_posterior_sample(surrogate_posterior, data=data_normalized)
plt.savefig('vb_gmm_bayes_sample1.pdf')
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 457} id="XjqWbW0nnCIF" outputId="89b779c6-0e05-42df-f84f-bdc1057681ef"
plot_posterior_sample(surrogate_posterior, data=data_normalized)
plt.savefig('vb_gmm_bayes_sample2.pdf')
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 457} id="vIY0XgwMRMTW" outputId="d40f57df-a19b-4ab5-eea3-73e69bfb200f"
plot_posterior_sample(surrogate_posterior, data=data_normalized)
plt.savefig('vb_gmm_bayes_sample3.pdf')
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} id="oY0_fXwJm-wu" outputId="4be7b268-82ce-4f43-b33f-f489183614cd"
params = surrogate_posterior.sample()
print(params.mixture_probs)

# + colab={"base_uri": "https://localhost:8080/"} id="XX-WL8Gion0D" outputId="b3ea947f-cea9-4b2d-f17a-6ca61ea2223f"
params = surrogate_posterior.sample()
print(params.mixture_probs)
