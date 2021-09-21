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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/flow_2d_mlp.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="UNsR5KGRU3HI"
# # Mapping a 2d standard Gaussian to a more complex distribution using an invertible MLP
#
# Author: George Papamakarios
#
# Based on the example by Eric Jang from
# https://blog.evjang.com/2018/01/nf1.html
#
# Reproduces Figure 23.1 of the book *Probabilistic Machine Learning: Advanced Topics* by Kevin P. Murphy

# + [markdown] id="ygG6LSeF4m2t"
# ## Imports and definitions

# + id="aGna32BcyeTI"
from typing import Sequence

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

Array = jnp.ndarray
PRNGKey = Array

prng = hk.PRNGSequence(42)


# + [markdown] id="EdCdDC-A4qdn"
# ## Create flow model

# + id="JibqwcduyKTU"
class Parameter(hk.Module):
  """Helper Haiku module for defining model parameters."""

  def __init__(self,
               module_name: str,
               param_name: str,
               shape: Sequence[int],
               init: hk.initializers.Initializer):
    """Initializer.

    Args:
      module_name: name of the module.
      param_name: name of the parameter.
      shape: shape of the parameter.
      init: initializer of the parameter value.
    """
    super().__init__(name=module_name)
    self._param = hk.get_parameter(param_name, shape=shape, init=init)

  def __call__(self) -> Array:
    return self._param


class LeakyRelu(distrax.Lambda):
  """Leaky ReLU elementwise bijector."""

  def __init__(self, slope: Array):
    """Initializer.

    Args:
      slope: the slope for x < 0. Must be positive.
    """
    forward = lambda x: jnp.where(x >= 0., x, x * slope)
    inverse = lambda y: jnp.where(y >= 0., y, y / slope)
    forward_log_det_jacobian = lambda x: jnp.where(x >= 0., 0., jnp.log(slope))
    inverse_log_det_jacobian = lambda y: jnp.where(y >= 0., 0., -jnp.log(slope))
    super().__init__(
        forward=forward,
        inverse=inverse,
        forward_log_det_jacobian=forward_log_det_jacobian,
        inverse_log_det_jacobian=inverse_log_det_jacobian,
        event_ndims_in=0)


def make_model() -> distrax.Transformed:
  """Creates the flow model."""
  num_layers = 6

  layers = []
  for _ in range(num_layers - 1):
    # Each intermediate layer is an affine transformation followed by a leaky
    # ReLU nonlinearity.
    matrix = Parameter(
        'affine',
        'matrix',
        shape=[2, 2],
        init=hk.initializers.Identity())()
    bias = Parameter(
        'affine',
        'bias',
        shape=[2],
        init=hk.initializers.TruncatedNormal(2.))()
    affine = distrax.UnconstrainedAffine(matrix, bias)
    slope = Parameter('nonlinearity', 'slope', shape=[2], init=jnp.ones)()
    nonlinearity = distrax.Block(LeakyRelu(slope), 1)
    layers.append(distrax.Chain([nonlinearity, affine]))

  # The final layer is just an affine transformation.
  matrix = Parameter(
      'affine',
      'matrix',
      shape=[2, 2],
      init=hk.initializers.Identity())()
  bias = Parameter(
      'affine',
      'bias',
      shape=[2],
      init=jnp.zeros)()
  affine = distrax.UnconstrainedAffine(matrix, bias)
  layers.append(affine)

  flow = distrax.Chain(layers[::-1])
  base = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(2),
      scale_diag=jnp.ones(2))
  return distrax.Transformed(base, flow)


@hk.without_apply_rng
@hk.transform
def model_log_prob(x: Array) -> Array:
  model = make_model()
  return model.log_prob(x)


@hk.without_apply_rng
@hk.transform
def model_sample(key: PRNGKey, num_samples: int) -> Array:
  model = make_model()
  return model.sample(seed=key, sample_shape=[num_samples])


# + [markdown] id="uq8dBvLz6aVK"
# ## Define target distribution

# + colab={"height": 281} id="V9SGQ83H1DO4" outputId="4f4f7100-5d2a-44e2-9b51-86f6e2e9f517"
def target_sample(key: PRNGKey, num_samples: int) -> Array:
  """Generates samples from target distribution.
  
  Args:
    key: a PRNG key.
    num_samples: number of samples to generate.

  Returns:
    An array of shape [num_samples, 2] containing the samples.
  """
  key1, key2 = jax.random.split(key)
  x = 0.6 * jax.random.normal(key1, [num_samples])
  y = 0.8 * x ** 2 + 0.2 * jax.random.normal(key2, [num_samples])
  return jnp.concatenate([y[:, None], x[:, None]], axis=-1)

# Plot samples from target distribution.
data = target_sample(next(prng), num_samples=1000)
plt.plot(data[:, 0], data[:, 1], '.', color='red', label='Target')
plt.axis('equal')
plt.title('Samples from target distribution')
plt.legend();

# + [markdown] id="zPFHR0Sd8joE"
# ## Train model

# + colab={"height": 281} id="gsnjWDi90tw1" outputId="0791fd77-a7e5-4d28-a272-6e8a2267bf4d"
# Initialize model parameters.
params = model_sample.init(next(prng), next(prng), num_samples=1)

# Plot samples from the untrained model.
x = target_sample(next(prng), num_samples=1000)
y = model_sample.apply(params, next(prng), num_samples=1000)
plt.plot(x[:, 0], x[:, 1], '.', color='red', label='Target')
plt.plot(y[:, 0], y[:, 1], '.', color='green', label='Model')
plt.axis('equal')
plt.title('Samples from untrained model')
plt.legend();

# + id="ZRQaTdDN1F7K" outputId="f551b6c7-698d-457d-89f6-440b535d5a82"
# Loss function is negative log likelihood.
loss_fn = jax.jit(lambda params, x: -jnp.mean(model_log_prob.apply(params, x)))

# Optimizer.
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Training loop.
for i in range(5000):
  data = target_sample(next(prng), num_samples=100)
  loss, g = jax.value_and_grad(loss_fn)(params, data)
  updates, opt_state = optimizer.update(g, opt_state)
  params = optax.apply_updates(params, updates)

  if i % 100 == 0:
    print(f'Step {i}, loss = {loss:.3f}')

# + colab={"height": 281} id="VMuj1oH11MOu" outputId="0aeb750e-e4d3-453a-ca75-9f572c383a5e"
# Plot samples from the trained model.
x = target_sample(next(prng), num_samples=1000)
y = model_sample.apply(params, next(prng), num_samples=1000)
plt.plot(x[:, 0], x[:, 1], '.', color='red', label='Target')
plt.plot(y[:, 0], y[:, 1], '.', color='green', label='Model')
plt.axis('equal')
plt.title('Samples from trained model')
plt.legend();


# + [markdown] id="XAlCxXqq_cqj"
# ## Create plot with intermediate distributions

# + id="_8kGzlUO1Oli"
@hk.without_apply_rng
@hk.transform
def model_sample_intermediate(key: PRNGKey, num_samples: int) -> Array:
  model = make_model()
  samples = []
  x = model.distribution.sample(seed=key, sample_shape=[num_samples])
  samples.append(x)
  for layer in model.bijector.bijectors[::-1]:
    x = layer.forward(x)
    samples.append(x)
  return samples

xs = model_sample_intermediate.apply(params, next(prng), num_samples=2000)

# + colab={"height": 237} id="NbjnETx-1Q67" outputId="89171275-4371-40a8-875c-96fec3119f59"
plt.rcParams['figure.figsize'] = [2 * len(xs), 3]
fig, axs = plt.subplots(1, len(xs))
fig.tight_layout()

color = xs[0][:, 1]
cm = plt.cm.get_cmap('gnuplot2')

for i, (x, ax) in enumerate(zip(xs, axs)):
  ax.scatter(x[:, 0], x[:, 1], s=10, cmap=cm, c=color)
  ax.axis('equal')
  if i == 0:
    title = 'Base distribution'
  else:
    title = f'Layer {i}'
  ax.set_title(title)
