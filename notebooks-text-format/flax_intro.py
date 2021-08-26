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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/mlp/flax_intro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="rF208fIxvq8m"
# # Introduction to neural networks using Flax
#
#
#
# Flax / Linen is a neural net library, built on top of JAX, "designed to offer an implicit variable management API to save the user from having to manually thread thousands of variables through a complex tree of functions." To handle both current and future JAX transforms (configured and composed in any way), Linen Modules are defined as explicit functions of the form
# $$
# f(v_{in}, x) \rightarrow v_{out}, y
# $$
# Where $v_{in}$ is the collection of variables (eg. parameters) and PRNG state used by the model, $v_{out}$ the mutated output variable collections, $x$ the input data and $y$ the output data. We illustrate this below. Our tutorial is based on the official [flax intro](https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html) and [linen colab](https://github.com/google/flax/blob/master/docs/notebooks/linen_intro.ipynb). Details are in the [flax source code](https://flax.readthedocs.io/en/latest/_modules/index.html). Note: please be sure to read our [JAX tutorial](https://github.com/probml/pyprobml/blob/master/book1/intro/jax_intro.ipynb) first.
#

# + id="uRAzAXYXvztz"
import numpy as np
#np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

import matplotlib.pyplot as plt


# + colab={"base_uri": "https://localhost:8080/"} id="1ob8P9ALvkcM" outputId="573415d3-eba0-4d6a-e51a-674aea067217"
# Install the latest JAXlib version.
# #!pip install --upgrade -q pip jax jaxlib

# + id="68kI74E1vvEI" colab={"base_uri": "https://localhost:8080/"} outputId="4525fc58-0c43-4909-dad9-1172ef516cbf"
import jax
from jax import lax, random, numpy as jnp
key = random.PRNGKey(0)


# + id="N3dXu6XY6U0H"
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

# Useful type aliases
Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any

# + colab={"base_uri": "https://localhost:8080/"} id="7pcNcE9_Qj_l" outputId="107ed9cb-1d4b-46ab-8032-60836e4b083e"
# Install Flax at head:
# !pip install --upgrade -q git+https://github.com/google/flax.git

# + id="s80k9sonQfDi"
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import optim

from jax.config import config
config.enable_omnistaging() # Linen requires enabling omnistaging


# + [markdown] id="aGrUFJYxjyL7"
# # MLP in vanilla JAX
#
# We construct a simple MLP with L hidden layers (relu activation), and scalar output (linear activation).
#
# Note: JAX and Flax, like NumPy, are row-based systems, meaning that vectors are represented as row vectors and not column vectors. 
#

# + id="mWQGVJMP0VMB"
# We define the parameter initializers using a signature that is flax-compatible
# https://flax.readthedocs.io/en/latest/_modules/jax/_src/nn/initializers.html

def weights_init(key, shape, dtype=jnp.float32):
  return random.normal(key, shape, dtype)
  #return jnp.ones(shape, dtype)

def bias_init(key, shape, dtype=jnp.float32):
  return jnp.zeros(shape, dtype)

def relu(a):
  return jnp.maximum(a, 0)


# + id="GepkhhTh-9b-"
# A minimal MLP class

class MLP0():
  features: Sequence[int] # number of features in each layer

  def __init__(self, features): # class constructor
    self.features = features

  def init(self, key, x): # initialize parameters
    in_size = np.shape(x)[1]
    sizes = np.concatenate( ([in_size], self.features) )
    nlayers = len(sizes)
    params = {}
    for i in range(nlayers-1):
      in_size = sizes[i]
      out_size = sizes[i+1]
      subkey1, subkey2, key = random.split(key, num=3)
      W = weights_init(subkey1, (in_size, out_size) )
      b = bias_init(subkey2, out_size)
      params[f'W{i}'] = W
      params[f'b{i}'] = b
    return params

  def apply(self, params, x): # forwards pass
    activations = x
    nhidden_layers = len(self.features)-1
    for i in range(nhidden_layers):
      W = params[f'W{i}'];
      b = params[f'b{i}'];
      outputs = jnp.dot(activations, W) + b
      activations = relu(outputs)
    # for final layer, no activation function
    i = nhidden_layers
    outputs = jnp.dot(activations, params[f'W{i}']) + params[f'b{i}']
    return outputs



# + colab={"base_uri": "https://localhost:8080/"} id="8jFS4SNO0V_I" outputId="72d9d449-742e-45d1-9641-a4f3a6390e26"
key = random.PRNGKey(0)
D = 3
N = 2
x = random.normal(key, (N,D,))
layer_sizes = [3,1] # 1 hidden layer of size 3, 1 scalar output

model0 = MLP0(layer_sizes)
params0 = model0.init(key, x)

print('params')
for k,v in params0.items():
  print(k, v.shape)
  print(v)


y0 = model0.apply(params0, x)
print('\noutput')
print(y0)


# + [markdown] id="rBtPT-drBkGA"
# # Our first flax model
#
# Here we recreate the vanilla model in flax. Since we don't specify how the parameters are initialized, the behavior will not be identical to the vanilla model --- we will fix this below, but for now, we focus on model construction.
#
# We see that the model is a subclass of `nn.Module`, which is a subclass of Python's dataclass. The child class (written by the user) must define a `model.call(inputs)` method, that applies the function to the input, and a `model.setup()` method, that creates the modules inside this model.
#
# The module (parent) class defines two main methods: `model.apply(variables, input`, that applies the function to the input (and variables) to generate an output; and `model.init(key, input)`, that initializes the variables and returns them as a "frozen dictionary". This dictionary can contain multiple *kinds* of variables. In the example below, the only kind are parameters, which are immutable variables (that will usually get updated in an external optimization loop, as we show later). The parameters are  automatically named after the corresponding module (here, dense0, dense1, etc).  In this example, both modules are dense layers, so their parameters are a weight matrix (called 'kernel') and a bias vector.
#
# The hyper-parameters (in this case, the size of each layer) are stored as attributes of the class, and are specified when the module is constructed.

# + id="3zueDo1r0Qav"
class MLP(nn.Module):
  features: Sequence[int]
  default_attr: int = 42

  def setup(self):
    print('setup')
    self.layers = [nn.Dense(feat) for feat in self.features]

  def __call__(self, inputs):
    print('call')
    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)
    return x



# + colab={"base_uri": "https://localhost:8080/"} id="OoYDn8lX7_ZH" outputId="9aed2723-3248-417b-9a25-408ef49763d7"
key = random.PRNGKey(0)
D = 3
N = 2
x = random.normal(key, (N,D,))
layer_sizes = [3,1] # 1 hidden layer of size 3, 1 scalar output

print('calling constructor')
model = MLP(layer_sizes) # just initialize attributes of the object
print('OUTPUT')
print(model)

print('\ncalling init')
variables = model.init(key, x)  # calls setup then __call___
print('OUTPUT')
print(variables)


print('Calling apply')
y = model.apply(variables, x) # calls setup then __call___
print(y)


# + [markdown] id="5lwM1j1WksDG"
# # Compact modules
#
# To reduce the amount of boiler plate code, flax makes it possible to define a module just by writing the `call` method, avoiding the need to write a `setup` function. The corresponding layers will be created when the `init` funciton is called, so the input shape can be inferred lazily (when passed an input). 

# + colab={"base_uri": "https://localhost:8080/"} id="Akq_iXXdktwb" outputId="2827db63-b8a0-4500-dd78-bd6b49f72065"
class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat)(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x


model = MLP(layer_sizes)
print(model)

params = model.init(key, x)
print(params)

y = model.apply(params, x)
print(y)


# + [markdown] id="dNiuZ54yB7Gj"
# # Explicit parameter initialization
#
# We can control the initialization of the random parameters in each submodule by specifying an init function. Below we show how to initialize our MLP to match the vanilla JAX model. We then check both methods give the same outputs.

# + id="5W_lEFsU4t04"
def make_const_init(x):
  def init_params(key, shape, dtype=jnp.float32):
    return x
  return init_params

class MLP_init(nn.Module):
  features: Sequence[int]
  params_init: Dict

  def setup(self):
    nlayers = len(self.features)
    layers = []
    for i in range(nlayers):
      W = self.params_init[f'W{i}'];
      b = self.params_init[f'b{i}']; 
      weights_init = make_const_init(W)
      bias_init = make_const_init(b)
      layer = nn.Dense(self.features[i], kernel_init=weights_init, bias_init=bias_init)
      layers.append(layer)
    self.layers = layers

  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)
    return x



# + colab={"base_uri": "https://localhost:8080/"} id="R27PhzrLY_zJ" outputId="adaba9d1-90a4-444b-95be-cef4da0768fe"
params_init = params0
model = MLP_init(layer_sizes, params_init)
print(model)

variables = model.init(key, x)
params = variables['params']
print(params)

W0 = params0['W0']
W = params['layers_0']['kernel']
assert np.allclose(W, W0)

y = model.apply(variables, x)
print(y)
assert np.allclose(y, y0)


# + [markdown] id="Rf8avaA_nGJ1"
# # Creating your own modules
#
# Now we illustrate how to create a module with its own parameters, instead of relying on composing built-in primitives. As an example, we write our own dense layer class.

# + colab={"base_uri": "https://localhost:8080/"} id="WUJ98XpSnS8F" outputId="ccdc09ef-f87f-4234-d64e-0bd583406851"
class SimpleDense(nn.Module):
  features: int # num output features for this layer
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    features_in = inputs.shape[-1] # infer shape from input
    features_out = self.features
    kernel = self.param('kernel', self.kernel_init, (features_in, features_out))
    bias = self.param('bias', self.bias_init, (features_out,))
    outputs = jnp.dot(inputs, kernel) + bias
    return outputs


model = SimpleDense(features=3)
print(model)

vars = model.init(key, x)
print(vars)

y = model.apply(vars, x)
print(y)


# + [markdown] id="BJpBh933-GTW"
# # Stochastic layers
#
# Some layers may need a source of randomness. If so, we must pass them a PRNG in the `init` and `apply` functions, in addition to the PRNG used for parameter initialization. We illustrate this below using dropout. We construct two versions, one which is stochastic (for training), and one which is deterministic (for evaluation). 

# + colab={"base_uri": "https://localhost:8080/"} id="tMSpLucO-Yfj" outputId="9726f1f3-ca25-4946-f9da-29692b1b034c"
class Block(nn.Module):
  features: int
  training: bool
  @nn.compact
  def __call__(self, inputs):
    x = nn.Dense(self.features)(inputs)
    x = nn.Dropout(rate=0.5)(x, deterministic=not self.training)
    return x

N = 1; D = 2;
x = random.uniform(key, (N,D))

model = Block(features=3, training=True)
key = random.PRNGKey(0)
variables = model.init({'params': key, 'dropout': key}, x)
#variables = model.init(key, x) # cannot share the rng
print('variables', variables)

# Apply stochastic model
for i in range(2):
  key, subkey = random.split(key)
  y = model.apply(variables, x, rngs={'dropout': subkey})
  print(f'train output {i}, ', y)

# Now make a deterministic version
eval_model = Block(features=3, training=False)
key = random.PRNGKey(0)
#variables = eval_model.init({'params': key, 'dropout': key}, x)
for i in range(2):
  key, subkey = random.split(key)
  y = eval_model.apply(variables, x, rngs={'dropout': subkey})
  print(f'eval output {i}, ', y)



# + [markdown] id="CHieB2aAumdg"
# # Stateful layers
#
# In addition to parameters, linen modules can contain other kinds of variables, which may be mutable as we illustrate below.
# Indeed, parameters are just a special case of variable.
# In particular, this line
# ```
# p = self.param('param_name', init_fn, shape, dtype)
# ```
# is a convenient shorthand for this:
# ```
# p = self.variable('params', 'param_name', lambda s, d: init_fn(self.make_rng('params'), s, d), shape, dtype).value
# ```
#

# + [markdown] id="EAQxx2Tu8xln"
# ## Example: counter

# + colab={"base_uri": "https://localhost:8080/"} id="BeGNa8zaut41" outputId="eb8923e0-ed62-46f9-f11b-80dec053e31a"
class Counter(nn.Module):
  @nn.compact
  def __call__(self):
    # variable(collection, name, init_fn, *init_args)
    counter1 = self.variable('counter', 'count1', lambda: jnp.zeros((), jnp.int32))
    counter2 = self.variable('counter', 'count2', lambda: jnp.zeros((), jnp.int32))
    is_initialized = self.has_variable('counter', 'count1')
    if is_initialized:
      counter1.value += 1
      counter2.value += 2
    return counter1.value, counter2.value


model = Counter()
print(model)

init_variables = model.init(key) # calls the `call` method
print('initialized variables:\n', init_variables)
counter = init_variables['counter']['count1']
print('counter 1 value', counter)

y, mutated_variables = model.apply(init_variables, mutable=['counter'])
print('mutated variables:\n', mutated_variables)
print('output:\n', y)


# + [markdown] id="1IaC2RT1v65t"
# ## Combining mutable variables and immutable parameters
#
# We can combine mutable variables with immutable parameters.
# As an example, consider a simplified version of batch normalization, which 
#  computes the running mean of its inputs, and adds an optimzable offset (bias) term. 
#
#

# + id="NXP19telv_Y_"
class BiasAdderWithRunningMean(nn.Module):
  decay: float = 0.99

  @nn.compact
  def __call__(self, x):
    is_initialized = self.has_variable('params', 'bias')

    # variable(collection, name, init_fn, *init_args)
    ra_mean = self.variable('batch_stats', 'mean', lambda s: jnp.zeros(s), x.shape[1:])

    dummy_mutable = self.variable('mutables', 'dummy', lambda s: 42, 0)

    # param(name, init_fn, *init_args)
    bias = self.param('bias', lambda rng, shape: jnp.ones(shape), x.shape[1:]) 

    if is_initialized:
      ra_mean.value = self.decay * ra_mean.value + (1.0 - self.decay) * jnp.mean(x, axis=0, keepdims=True)

    return x - ra_mean.value + bias



# + [markdown] id="x_WsMGY8xA_x"
#
# The intial variables are:
# params = (bias=1), batch_stats=(mean=0)
#
# If we pass in x=ones(N,D), the  running average becomes
# $$
# 0.99*0 + (1-0.99)*1 = 0.01
# $$
# and the output becomes
# $$
# 1 - 0.01 + 1 = 1.99
# $$
#
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="dvXKCE8yxiTu" outputId="4ddb1117-32a0-481d-d8bb-876010d0e821"
key = random.PRNGKey(0)
N = 2
D = 5
x = jnp.ones((N,D))
model = BiasAdderWithRunningMean()

variables = model.init(key, x)
print('initial variables:\n', variables)
nonstats, stats = variables.pop('batch_stats')
print('nonstats', nonstats)
print('stats', stats)


# + colab={"base_uri": "https://localhost:8080/"} id="Ytr2_w9U12PT" outputId="30555a51-9b09-4ef2-8222-98c9b52e4a47"
y, mutables = model.apply(variables, x, mutable=['batch_stats'])
print('output', y)
print('mutables', mutables)

# + [markdown] id="B1g2GW3f3B-Z"
# To call the function with the updated batch stats, we have to stitch together the new mutated state with the old state, as shown below.
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="cpBb21A72Bdj" outputId="d5ce7521-90c4-48a0-a180-4f02be5fe5f8"

variables = unfreeze(nonstats)
print(variables)
variables['batch_stats'] = mutables['batch_stats']
variables = freeze(variables)
print(variables)

# + [markdown] id="sa7nH74Y5-Lg"
# If we pass in x=2*ones(N,D), the running average gets updated to
# $$
# 0.99 * 0.01 + (1-0.99) * 2.0 = 0.0299
# $$
# and the output becomes
# $$
# 2- 0.0299 + 1 = 2.9701
# $$

# + colab={"base_uri": "https://localhost:8080/"} id="t2dF2si51QN5" outputId="6177ee1c-41b7-40e6-d954-d0c1c85b0180"

x = 2*jnp.ones((N,D))
y, mutables = model.apply(variables, x, mutable=['batch_stats'])
print('output', y)
print('batch_stats', mutables)

assert np.allclose(y, 2.9701)
assert np.allclose(mutables['batch_stats']['mean'], 0.0299)

# + [markdown] id="cnBmgGxOoPKU"
# # Optimization
#
# Flax has several built-in (first-order) optimizers, as we illustrate below on a random linear function. (Note that we can also fit a model defined in flax using some other kind of optimizer, such as that provided by the [optax library](https://github.com/deepmind/optax).)

# + colab={"base_uri": "https://localhost:8080/"} id="OTHgj_pMra3H" outputId="d142c2bb-c725-47e6-8cba-5b55f9f16b48"
D = 5
key = jax.random.PRNGKey(0)
params = {'w': jax.random.normal(key, (D,))}
print(params)

x = jax.random.normal(key, (D,))

def loss(params):
  w = params['w']
  return jnp.dot(x, w)

loss_grad_fn = jax.value_and_grad(loss)
v, g = loss_grad_fn(params)
print(v)
print(g)

# + id="7KdmBHa8oWFY" colab={"base_uri": "https://localhost:8080/"} outputId="3695c5f5-339c-4e27-d80e-30d100ddae66"
from flax import optim
optimizer_def = optim.Momentum(learning_rate=0.1, beta=0.9)
print(optimizer_def)

optimizer = optimizer_def.create(params) 
print(optimizer)

# + colab={"base_uri": "https://localhost:8080/"} id="1JpgauX_ox_w" outputId="c649c61c-93ba-41a0-a834-2c254a53a243"
for i in range(10):
  params = optimizer.target
  loss_val, grad = loss_grad_fn(params)
  optimizer = optimizer.apply_gradient(grad)
  params = optimizer.target
  print('step {}, loss {:0.3f}, params {}'.format(i, loss_val, params))

# + [markdown] id="_ITTDWT2ECxC"
# # Worked example: MLP for MNIST 
#
# We demonstrate how to fit a shallow MLP to MNIST using Flax.
# We use this function:
# https://github.com/probml/pyprobml/blob/master/scripts/fit_flax.py
# To allow us to edit this file locally (in colab), and push commits back to github, we sync this colab with github. (For details see [this colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/intro/colab_intro.ipynb), the cell labeled "Working with github".)
#
#

# + [markdown] id="vavamofruHS_"
# ## Import code

# + colab={"base_uri": "https://localhost:8080/"} id="L3KR7bMQWQ2A" outputId="6d6f555c-4da0-46dd-975c-ce7daeb24564"
# !ls

# + colab={"base_uri": "https://localhost:8080/"} id="3OhoJQUnuE1Q" outputId="d40b0ab5-8fb3-4286-a925-51e41be70d40"
from google.colab import drive
drive.mount('/content/drive')
# !ls /content/drive/MyDrive/ssh/

# + colab={"base_uri": "https://localhost:8080/"} id="ItaIH9dyocZA" outputId="aebdafd4-9c3c-4bb5-a9a7-f72dbe21f979"
# !rm -rf probml_tools*.*
# !wget https://raw.githubusercontent.com/probml/pyprobml/master/scripts/probml_tools.py   
import probml_tools as pml

# + colab={"base_uri": "https://localhost:8080/"} id="rUYY3-N4uWYh" outputId="a9e455f0-0b49-4429-ad1d-689f5a6936cc"

# !rm -rf pyprobml
pml.git_ssh("git clone https://github.com/probml/pyprobml.git")


# + id="fjUVc9yKvFFP"
# %load_ext autoreload
# %autoreload 2

# + colab={"base_uri": "https://localhost:8080/", "height": 17} id="OQ7d5hfulh7r" outputId="744eedb7-72d1-4912-fe55-bd79c3fbd3fe"
from google.colab import files
files.view('/content/pyprobml/scripts/fit_flax.py')

# + colab={"base_uri": "https://localhost:8080/"} id="gKnFd4MFu_jE" outputId="0d715cf0-8b5c-495d-a1c4-4357d3fe1038"
import pyprobml.scripts.fit_flax as ff
ff.test()

# + [markdown] id="AMjR542vpGAQ"
# Edit the file, then commit changes.

# + colab={"base_uri": "https://localhost:8080/"} id="YJXwfqz0-_XJ" outputId="01fe932f-10f2-4486-c8bd-80ef64c06e68"
# If made any local changes to fit_flax.py, save them to github
# %cd /content/pyprobml
pml.git_ssh("git add scripts; git commit -m 'push from colab'; git push")
# %cd /content

# + [markdown] id="E_xSZi3v03pC"
# ## Data

# + colab={"base_uri": "https://localhost:8080/"} id="l4uNqjBIW0we" outputId="566ff9c6-ca6f-42a7-dbf4-abf2c78f6d53"


def process_record(batch):
  image = batch['image']
  label = batch['label']
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


# + [markdown] id="rLiWUSjR05BQ"
# ## Model
#
#

# + id="cLwAwqd4Nzvy"
class Model(nn.Module):
  nhidden: int
  nclasses: int

  @nn.compact
  def __call__(self, x):
    if self.nhidden > 0:
      x = nn.Dense(self.nhidden)(x)
      x = nn.relu(x)
    x = nn.Dense(self.nclasses)(x) # logits
    x = nn.log_softmax(x) # log probabilities
    return x


# + [markdown] id="9JsVFGfU628j"
# ## Training loop
#

# + colab={"base_uri": "https://localhost:8080/", "height": 497} id="KDAJthPTvxI7" outputId="3f277974-c0db-4c0e-c39d-04648ae16f8f"


model = Model(nhidden = 128, nclasses=10) 
rng = jax.random.PRNGKey(0)
num_steps = 200

params, history =  ff.fit_model(
    model, rng, num_steps, train_iter, test_iter, print_every=20)
  
display(history)

# + colab={"base_uri": "https://localhost:8080/", "height": 278} id="RWv2Sspl8EAN" outputId="88ac8b9a-9bc6-4921-8e0c-71ec0d61210d"
plt.figure()
plt.plot(history['step'], history['test_accuracy'], 'o-', label='test accuracy')
plt.xlabel('num. minibatches')
plt.legend()
plt.show()

# + id="oWe69Z51Q3Kz"

