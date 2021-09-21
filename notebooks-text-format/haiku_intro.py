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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/haiku_intro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="IbJqnWzn9L6S"
# # An introduction to haiku (neural network library in JAX)
#
# https://github.com/deepmind/dm-haiku
#
# Haiku is a JAX version of the [Sonnet](https://github.com/deepmind/sonnet) neural network library (which was written in Tensorflow2). The main thing it does is to provide a way to convert object-oriented (stateful) code into functionally pure code, which can then be processed by JAX transformations like jit and grad. In addition it has implementations of common neural net building blocks.
#
# Below we give a brief introduction, based on the offical docs.
#
#

# + id="Yi0DMWY89LKZ"
# %%capture
# !pip install git+https://github.com/deepmind/dm-haiku
import haiku as hk

# + id="xDcW57Ynd5jX"
# %%capture
# !pip install git+git://github.com/deepmind/optax.git
import optax

# + id="JpbH4Hhr9N_O"
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


# + [markdown] id="-JOlff_nHlz6"
# # Haiku function transformations
#
# The main thing haiku offers is a way to let the user write a function that defines and accesses mutable parameters inside the function, and then to transform this into a function that takes the parameters as explicit arguments. (The advantage of the implicit method will become clearer later, when we consider modules, which let the user define parameters using nested objects.)

# + id="e9-i6pOFIPFL"
# Here is a function that takes in data x, and meta-data output_size, 
# but creates its mutable parameters internally.
# The parameters define an affine mapping, f1(x) = b + W*x
def f1(x, output_size):
  j, k = x.shape[-1], output_size
  w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
  w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
  b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.ones)
  return jnp.dot(x, w) + b




# + colab={"base_uri": "https://localhost:8080/"} id="_Y6HcwKyJyIB" outputId="638ec341-1c5a-4a8f-fbf6-e6b597aa6f2f"
# transform will convert f1 to a function that explicitly uses parameters, which we call f2.
# (We explain the rng part later.)
f2 = hk.without_apply_rng(hk.transform(f1))

# f2 is a struct with two functions, init and apply
print(f2)

# + colab={"base_uri": "https://localhost:8080/"} id="YbXmx9yYI9fJ" outputId="305dda0b-a415-4ff4-912b-e1c07823557b"
# The init function creates an initial random set of parameters
# by calling f1 on some data x (the values don't matter, just the shape)
# and using the RNG. 
# The params are stoerd in a haiku FlatMap (like a FrozenDict)
output_size = 2
dummy_x = jnp.array([[1., 2., 3.]])
rng_key = jax.random.PRNGKey(42)
#params = f2.init(rng=rng_key, x=dummy_x, output_size = output_size)
params = f2.init(rng_key, dummy_x, output_size)
print(params)



# + colab={"base_uri": "https://localhost:8080/"} id="uxcFt7GiKocY" outputId="3883cca5-9621-4ec6-a798-86a88c57a7e7"
p = params['~']
print(p['b'])

# + colab={"base_uri": "https://localhost:8080/", "height": 164} id="5X2uAetOLcpU" outputId="d0f1fd5b-b2b4-4e0f-b743-b8f614eb39b0"
# params are frozen
params['~']['b'] = jnp.array([2.0, 2.0])

# + colab={"base_uri": "https://localhost:8080/"} id="hmEWyvtQKBJD" outputId="f21f0396-738d-47b5-a76d-c4be8c28f9e0"
# The apply function takes a param FlatMap and injects it into the original f1 function
sample_x = jnp.array([[1., 2., 3.]])
output_1 = f2.apply(params=params, x=sample_x, output_size = output_size)
print(output_1)


# + [markdown] id="_KVtU8v2Q9NG"
# # Transforming stateful functions
#
# We can create a function with internal state that is mutated on each call,
# but is treated separately from the fixed parameters (which are usually mutated by an external optimizer). Below we illustrate this for a simple counter example, that gets incremented on each call.
#

# + colab={"base_uri": "https://localhost:8080/"} id="HXCW0UNSRZaF" outputId="383dc9d0-8f1e-4d70-f6bf-ddd464d74683"
def stateful_f(x):
  counter = hk.get_state("counter", shape=[], dtype=jnp.int32, init=jnp.ones)
  multiplier = hk.get_parameter('multiplier', shape=[1,], dtype=x.dtype, init=jnp.ones)
  hk.set_state("counter", counter + 1)
  output = x + multiplier * counter
  return output

stateful_forward = hk.without_apply_rng(hk.transform_with_state(stateful_f))
sample_x = jnp.array([[5., ]])
params, state = stateful_forward.init(x=sample_x, rng=rng_key)
print(f'Initial params:\n{params}\nInitial state:\n{state}')
print('##########')
for i in range(3):
  output, state = stateful_forward.apply(params, state, x=sample_x)
  print(f'After {i+1} iterations:\nOutput: {output}\nState: {state}')
  print('##########')


# + [markdown] id="KwvRhDUlLsSq"
# # Modules

# + [markdown] id="7jDgDevPNTPS"
# Creating a single dict of parameters and passing it as an argument is easy,
# and haiku is overkill for such cases. However we often have nested parameterized functions, each of which has metadata (like `output_sizes` above) that needs to specified. In such cases it is easier to work with haiku modules. These are just like regular Python classes (no required methods), but typically have a `__init__` constructor and a `__call__` method that can be invoked when calling the module. Below we reimplement the affine function f1 as a module.
#

# + id="9ySaGWV3IFDX"
class MyLinear1(hk.Module):

  def __init__(self, output_size, name=None):
    super().__init__(name=name)
    self.output_size = output_size

  def __call__(self, x):
    j, k = x.shape[-1], self.output_size
    w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
    w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
    b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.ones)
    return jnp.dot(x, w) + b

# + id="maCs7NjiIIWW"


def _forward_fn_linear1(x):
  module = MyLinear1(output_size=2)
  return module(x)


forward_linear1 = hk.without_apply_rng(hk.transform(_forward_fn_linear1))

# + colab={"base_uri": "https://localhost:8080/"} id="3SUOKgTdIqPk" outputId="571be59f-bb55-420a-af4c-604999ccafab"
dummy_x = jnp.array([[1., 2., 3.]])
rng_key = jax.random.PRNGKey(42)

params = forward_linear1.init(rng=rng_key, x=dummy_x)
print(params)

sample_x = jnp.array([[1., 2., 3.]])

output_1 = forward_linear1.apply(params=params, x=sample_x)
print(output_1)

# + [markdown] id="s2eSwM2ER1tA"
# # Nested and built-in modules
#
# We can nest modules inside of each other. This allows us to create complex functions. Haiku ships with [many common layers](https://dm-haiku.readthedocs.io/en/latest/api.html#common-modules), as well as a 
# [small number of common models](https://dm-haiku.readthedocs.io/en/latest/api.html#module-haiku.nets), like MLPs and Resnets. (A model is just multiple layers.)

# + colab={"base_uri": "https://localhost:8080/"} id="5dLnlxW-I2jh" outputId="50175ddc-dfbc-4d60-9dd0-1d62debdf320"


class MyModuleCustom(hk.Module):
  def __init__(self, output_size=2, name='custom_linear'):
    super().__init__(name=name)
    self._internal_linear_1 = hk.nets.MLP(output_sizes=[2, 3], name='hk_internal_linear')
    self._internal_linear_2 = MyLinear1(output_size=output_size, name='old_linear')

  def __call__(self, x):
    return self._internal_linear_2(self._internal_linear_1(x))

def _custom_forward_fn(x):
  module = MyModuleCustom()
  return module(x)

custom_forward_without_rng = hk.without_apply_rng(hk.transform(_custom_forward_fn))
params = custom_forward_without_rng.init(rng=rng_key, x=sample_x)
params


# + [markdown] id="57eKZjlNSGlW"
# # Stochastic modules
#
#
# If the module is stochastic, we have to pass the RNG to the apply function (as well as the init function), as we show below. We can use `hk.next_rng_key()` to derive a new key from the one that the user passes to `apply`. This is useful for when we have nested modules.
#

# + colab={"base_uri": "https://localhost:8080/"} id="jZhc9ttJSIP7" outputId="58338735-523d-43d4-d973-2d6f69bcfada"
class HkRandom2(hk.Module):
  def __init__(self, rate=0.5):
    super().__init__()
    self.rate = rate

  def __call__(self, x):
    key1 = hk.next_rng_key()
    return jax.random.bernoulli(key1, 1.0 - self.rate, shape=x.shape)


class HkRandomNest(hk.Module):
  def __init__(self, rate=0.5):
    super().__init__()
    self.rate = rate
    self._another_random_module = HkRandom2()

  def __call__(self, x):
    key2 = hk.next_rng_key()
    p1 = self._another_random_module(x)
    p2 = jax.random.bernoulli(key2, 1.0 - self.rate, shape=x.shape)
    print(f'Bernoullis are  : {p1, p2}')

# Note that the modules that are stochastic cannot be wrapped with hk.without_apply_rng()
forward = hk.transform(lambda x: HkRandomNest()(x))

x = jnp.array(1.)
params = forward.init(rng_key, x=x)
# The 2 Bernoullis can be difference, since they use key1 and key2
# But across the 5 iterations the answers should be the same,
# since they are all produced by passing in the same rng_key to apply.
for i in range(5):
  print(f'\n Iteration {i+1}')
  prediction = forward.apply(params, x=x, rng=rng_key)

# + [markdown] id="_3EaBFJ9TP2C"
# # Combining JAX Function transformations and Haiku
#
# We cannot apply JAX function transformations, like jit and grad, inside of a haiku module, since modules are impure. So we have to use `hk.jit`, `hk.grad`, etc.
# See [this page](https://dm-haiku.readthedocs.io/en/latest/notebooks/transforms.html) for details. However, after transforming the haiku code to be pure, we can apply JAX transformations as usual.
#
#
# (See also the [equinox libary](https://github.com/patrick-kidger/equinox) for an alternative approach to this problem.)
#

# + id="42HeSZhgTeHF"


# + [markdown] id="zt7ArYx7am0T"
# # Example: MLP on MNIST
#
# This example is modified from https://github.com/deepmind/dm-haiku/blob/main/examples/mnist.py

# + id="Jxr-Yn5Xaq-P"
from typing import Generator, Mapping, Tuple

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

Batch = Mapping[str, np.ndarray]




# + colab={"base_uri": "https://localhost:8080/", "height": 205, "referenced_widgets": ["190095d5de254f0f92add36689a2f37c", "af7c5fbbc4324da9b459ab5a214f2eed", "ea6a64028530455385fbf0faddf33934", "b4e39f75f93a4e49852b027c96a45261", "67cfa9b71c2245dcad60c57eccdf3882", "58b64255af8347fc95ddea22107f7ae9", "a4f93609a2c54ab5a0ca979a557ee62f", "3a1114b5331e4d7f8874acad1011f08e", "40709ad839df4e2898e5138e2896e645", "308b203c34054dfabd2e9fcdb35604f9", "2a3d89b100f24c6790fec5ef5c0be25e"]} id="r9u3nVb8bUGj" outputId="756e1cf2-10ac-41dc-d9e6-8b2933625dc9"
# Data
def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
) -> Generator[Batch, None, None]:
  """Loads the dataset as a generator of batches."""
  ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds))

# Make datasets.
train = load_dataset("train", is_training=True, batch_size=1000)
train_eval = load_dataset("train", is_training=False, batch_size=10000)
test_eval = load_dataset("test", is_training=False, batch_size=10000)

# + id="LkgyFvG0a07d"
# Model
NCLASSES = 10
def net_fn(batch: Batch) -> jnp.ndarray:
  """Standard LeNet-300-100 MLP network."""
  x = batch["image"].astype(jnp.float32) / 255.
  mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(300), jax.nn.relu,
      hk.Linear(100), jax.nn.relu,
      hk.Linear(NCLASSES),
  ])
  return mlp(x)

net = hk.without_apply_rng(hk.transform(net_fn))
L2_REGULARIZER = 1e-4


# + id="faFP5Efga_zZ"

# Metrics

# Training loss (cross-entropy).
def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
  """Compute the loss of the network, including L2."""
  logits = net.apply(params, batch)
  labels = jax.nn.one_hot(batch["label"], NCLASSES)

  l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
  softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
  softmax_xent /= labels.shape[0]

  return softmax_xent + L2_REGULARIZER * l2_loss

# Evaluation metric (classification accuracy).
@jax.jit
def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
  predictions = net.apply(params, batch)
  return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

@jax.jit
def update(
    params: hk.Params,
    opt_state: optax.OptState,
    batch: Batch,
) -> Tuple[hk.Params, optax.OptState]:
  """Learning rule (stochastic gradient descent)."""
  grads = jax.grad(loss)(params, batch)
  updates, opt_state = opt.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  return new_params, opt_state

# We maintain avg_params, the exponential moving average of the "live" params.
# avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
@jax.jit
def ema_update(params, avg_params):
  return optax.incremental_update(params, avg_params, step_size=0.001)


# + colab={"base_uri": "https://localhost:8080/"} id="M3--0utzbQze" outputId="a2383453-27c4-4735-ed57-e19a5cc2824b"
# Optimzier

LR =1e-3
opt = optax.adam(LR)

# Initialize network and optimiser; note we draw an input to get shapes.
params = avg_params = net.init(jax.random.PRNGKey(42), next(train))
opt_state = opt.init(params)

# Train/eval loop.
nsteps = 500
print_every = 100

def callback(step, avg_params, train_eval, test_eval):
  if step % print_every == 0:
    # Periodically evaluate classification accuracy on train & test sets.
    train_accuracy = accuracy(avg_params, next(train_eval))
    test_accuracy = accuracy(avg_params, next(test_eval))
    train_accuracy, test_accuracy = jax.device_get(
        (train_accuracy, test_accuracy))
    print(f"[Step {step}] Train / Test accuracy: "
          f"{train_accuracy:.3f} / {test_accuracy:.3f}.")
    

for step in range(nsteps+1):
  params, opt_state = update(params, opt_state, next(train))
  avg_params = ema_update(params, avg_params)
  callback(step, avg_params, train_eval, test_eval)

