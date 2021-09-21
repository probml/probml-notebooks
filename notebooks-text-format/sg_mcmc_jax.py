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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/sg_mcmc_jax.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="CQdfeQ_pOh9S"
# # SG-MCMC-JAX library
#
# https://github.com/jeremiecoullon/SGMCMCJax
#
#

# + [markdown] id="VQ5U7hR8QKbp"
# #Setup

# + id="mJawbDo3g5Dx"
# If running in TPU mode
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()

# + id="0fopeFuiObaN"
# %%capture 
# !pip install sgmcmcjax

# + colab={"base_uri": "https://localhost:8080/"} id="Qc4tvflueuqM" outputId="7c7b0c1a-7d41-4bca-b148-c92c59695906"
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, random

print(jax.__version__)
print(jax.devices())

# + id="v4WVl4nqPD1w"


from sgmcmcjax.samplers import build_sgld_sampler
from sgmcmcjax.kernels import build_sgld_kernel, build_psgld_kernel, build_sgldAdam_kernel, build_sghmc_kernel

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

key = random.PRNGKey(0)

# + [markdown] id="nm1WIaOCOzod"
# # Gaussian posterior
#
# https://sgmcmcjax.readthedocs.io/en/latest/nbs/gaussian.html

# + id="PY7HiHcAOicY"


# define model in JAX
def loglikelihood(theta, x):
    return -0.5*jnp.dot(x-theta, x-theta)

def logprior(theta):
    return -0.5*jnp.dot(theta, theta)*0.01

# generate dataset
N, D = 10_000, 100
key = random.PRNGKey(0)
mu_true = random.normal(key, shape=(D,))
X_data = random.normal(key, shape=(N, D)) + mu_true

# build sampler
batch_size = int(0.1*N)
dt = 1e-5
my_sampler = build_sgld_sampler(dt, loglikelihood, logprior, (X_data,), batch_size)



# + colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["06ea1f92111d42c59f5743f7f19ddecd", "31d5332f37ac479d9af2fb057f62919a", "08104e55ab654e6e879a690db22c330c", "044100d715714b419e185d40d82e466e", "143541dd2f7a4f1190c31124c2e1a4b9", "a600713bb198461392c65c1f4f81b885", "0e9463b210f1443fb6f2f4d63f738448", "3551fc911d954ad8b20c25d8957b3a6e", "a87ad53eea084f00ad0cd510f1283859", "8e2c1822d9884a76b54d0b8377baac64", "692f3d95b0664943817257244324af24"]} id="C4saWTbOPCsa" outputId="9a1f211f-3661-4538-92bc-a0e91ef8805f"
# %%time 
Nsamples = 10_000
samples = my_sampler(key, Nsamples, jnp.zeros(D))




# + colab={"base_uri": "https://localhost:8080/"} id="uuwhhcWGPHtg" outputId="3c61568a-eb0b-474b-b994-b3ec3a918e54"
print(samples.shape)
mu_est = jnp.mean(samples, axis=0)
print(jnp.allclose(mu_true, mu_est, atol=1e-1))
print(mu_true[:10])
print(mu_est[:10])

# + colab={"base_uri": "https://localhost:8080/", "height": 341} id="D68_hyWjXaZY" outputId="52b8c09c-1e01-4742-fafa-eb78a5634d95"
data = (X_data,)
init_fn, sgld_kernel, get_params = build_sgld_kernel(dt, loglikelihood, logprior, data, batch_size)


key, subkey = random.split(key)
params = random.normal(subkey, shape=(D,))

key, subkey = random.split(key)
state = init_fn(subkey, params)
print(state)

# + [markdown] id="IE268RvOQaRn"
# # Logistic regression
#
# https://sgmcmcjax.readthedocs.io/en/latest/nbs/logistic_regression.html

# + colab={"base_uri": "https://localhost:8080/", "height": 368} id="mz6uwMYfP6h7" outputId="b2575f27-03f3-4a42-a26d-69530aba264c"
from models.logistic_regression import gen_data, loglikelihood, logprior

key = random.PRNGKey(42)
dim = 10
Ndata = 100000

theta_true, X, y_data = gen_data(key, dim, Ndata)

# + [markdown] id="dh7cMKRaQknp"
# # Bayesian neural network
#
# https://sgmcmcjax.readthedocs.io/en/latest/nbs/BNN.html

# + colab={"base_uri": "https://localhost:8080/", "height": 368} id="KkvqqmuPQelA" outputId="dc5c931e-6842-4245-e6f1-08a2513002a8"
from models.bayesian_NN.NN_data import X_train, X_test, y_train, y_test
from models.bayesian_NN.NN_model import init_network, loglikelihood, logprior, accuracy

from sgmcmcjax.kernels import build_sgld_kernel
from tqdm.auto import tqdm

# + [markdown] id="9hchex74Qwf9"
# # FLAX CNN
#
# https://sgmcmcjax.readthedocs.io/en/latest/nbs/Flax_MNIST.html

# + id="mOK2K-J2REyH"
# %%capture
# !pip install --upgrade git+https://github.com/google/flax.git

# + id="iy3WEKDyQmfw"

import tensorflow_datasets as tfds

from flax import linen as nn


# + id="D-g1gVeSRqa1"
class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


# + colab={"base_uri": "https://localhost:8080/", "height": 273, "referenced_widgets": ["0fa5bf6d413d49da90cfdf6bc1c93ad7", "836a784d8eff4de490a1466c51145e5c", "72149a1caaa340ecb0bd1541ed3966af", "d3d0416899f349e79e40a35ca96304aa", "e2edef161ef14aac91f1df4aa97b82e4", "e00004a6fdf14570b808fa4cd2e45696", "8ab28b5ef74d4cd9933430b42796e760", "30c26b16d89341bfa7f66d6beaa83177", "3c127d0bc41c42ca8bad12c7efe2cf83", "ab14c268fc6842d48546e3b5cfd4a562", "fc4c9953df514699bd4a07551a8587af"]} id="e2WL0nyNRqse" outputId="aec3d2b1-39b2-4921-b35c-7fc185ee1bb1"
cnn = CNN()

def loglikelihood(params, x, y):
    x = x[jnp.newaxis] # add an axis so that it works for a single data point
    logits = cnn.apply({'params':(params)}, x)
    label = jax.nn.one_hot(y, num_classes=10)
    return jnp.sum(logits*label)

def logprior(params):
    return 1.

@jit
def accuracy_cnn(params, X, y):
    target_class = y
    predicted_class = jnp.argmax(cnn.apply({'params':(params)}, X), axis=1)
    return jnp.mean(predicted_class == target_class)


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds


train_ds, test_ds = get_datasets()

X_train_s = train_ds['image']
y_train_s = jnp.array(train_ds['label'])
X_test_s = test_ds['image']
y_test_s = jnp.array(test_ds['label'])


data = (X_train_s, y_train_s)
batch_size = int(0.01*data[0].shape[0])
print(batch_size)

# + colab={"base_uri": "https://localhost:8080/"} id="4FyicuEFTUJu" outputId="00c2509c-6457-4142-fb23-d6acb32af628"
params = cnn.init(key, jnp.ones([1,28,28,1]))
print(params.keys())
params = params['params']
print(params['Dense_0']['kernel'].shape)
print(params['Dense_1']['kernel'].shape)


# + id="WjzaI5agRy8P"
def run_sgmcmc(key, Nsamples, init_fn, my_kernel, get_params,
               record_accuracy_every=50):
    "Run SGMCMC sampler and return the test accuracy list"
    accuracy_list = []
    params = cnn.init(key, jnp.ones([1,28,28,1]))['params']
    key, subkey = random.split(key)
    state = init_fn(subkey, params)

    for i in tqdm(range(Nsamples)):
        key, subkey = random.split(key)
        state = my_kernel(i, subkey, state)
        if i % record_accuracy_every==0:
          test_acc = accuracy_cnn(get_params(state), X_test_s, y_test_s)
          accuracy_list.append((i, test_acc))

    return accuracy_list


# + id="7zV76Cc3co-M"
kernel_dict = {}

# SGLD
init_fn, kernel, get_params = build_sgld_kernel(5e-6, loglikelihood, logprior, data, batch_size)
kernel = jit(kernel)
kernel_dict['sgld'] = (init_fn, kernel, get_params)

# SG-HMC
init_fn, kernel, get_params = build_sghmc_kernel(
    1e-6, 4, loglikelihood, logprior, data, batch_size, compiled_leapfrog=False)
kernel = jit(kernel)
kernel_dict['sghmc'] = (init_fn, kernel, get_params)

# PSGLD
init_fn, kernel, get_params = build_psgld_kernel(1e-3, loglikelihood, logprior, data, batch_size)
kernel = jit(kernel)
kernel_dict['psgld'] = (init_fn, kernel, get_params)

# SGLD-Adam
init_fn, kernel, get_params = build_sgldAdam_kernel(1e-2, loglikelihood, logprior, data, batch_size)
kernel = jit(kernel)
kernel_dict['sgldAdam'] = (init_fn, kernel, get_params)


# + colab={"base_uri": "https://localhost:8080/", "height": 301, "referenced_widgets": ["515eee086faa4af192a4f15b732cdc12", "11d6cf780e094a74820cb1f4a59f82c2", "ccacbc5cf85c4f58ba6b04c1bd10f44a", "27463680b4964bb9b7cbfd6e3810a0a7", "a98650350f0245d4ad43eeb67db4d475", "725b55b0905d45c18bb7fa464c3c854a", "d74b51168bfc468ab6077ac1a3e9b510", "956879b98b6543a3b77edb3f13519d2a", "8040702a4a0d49c797ddb4108df97ab6", "c9f56d7307fa4cc69ec42eabae759d3f", "0d85e43f2cf8466c8620e59ae2150873", "00ce0f32a72d47b6b9ff3ea81baaba06", "37a1fa07d5114025a67808fe91d63d10", "948bda00470142a6bca036445a0915e4", "533fc883a3514a1ebad42b986435185e", "9c571f90046642299445f1a368e418ba", "131c370f746140a79ffb25d88f0f441f", "1d265306d5404362aa45616f5ba0b2c9", "1e6755b199c548e7a793ba287c563673", "5ac5e621d9b743678efd266c2a183ddd", "94b2c890d292439ea25c2c3b8fa8220c", "40f11285fdd54ee1b86bf97e6f57541b", "78771cfa75b54d5fb94e9d3874426398", "14e9bf1c01bc4873b33564334f539e74", "a35a1332bef54135a81e8180b9598ffc", "30d2304c9d4d43fca97feae501a52f1b", "ad591e13829b42d192cdd10644af31c4", "3dd51604e7e941eb99644b251ae461e6", "60998fdf80ef4cad96dadda247ae038b", "f3e3453c306144f1ad37c12d78998c86", "ea7c18e53f174f4f9fb6f13797f851f9", "d65822d65636431cb874b54bfcf77601", "d0173557582c4cd189745d4700ec42ef", "c20aaacdea57496cb7a1be74a524966a", "5370f4ac8b5c4fc986d6f5a162d8d0ef", "e8408ed20c1446fcb9e848d6db2cf39e", "e98f2e686d184706ae55f1a756c3112a", "0f985e720e414ffeb606d9c4ccb81f7f", "998a9482e0f5451cba8142d1c053a1c4", "22f8e96a941b4fe591734c48755468be", "7a8c03f483ac44f796d4795c96ea5b46", "0dfb815cdcea47d3adcb5604ef0a1241", "e5450ca369044b3480f42568a05003ef", "798ce80bf7104f55affa9dd149f19c4e"]} id="vcKM1NGCdSmm" outputId="8de4010e-90a0-404b-ad9d-c2ba9b95e937"
Nsamples = 500
acc_dict = {}

for (name, fns) in kernel_dict.items():
  print(name)
  init_fn, kernel, get_params = fns
  accuracy_list = run_sgmcmc(random.PRNGKey(0), Nsamples, init_fn, kernel, get_params)
  print(accuracy_list)
  acc_dict[name] = accuracy_list


# + colab={"base_uri": "https://localhost:8080/", "height": 321} id="q11jhQ2IjCdw" outputId="941e23ba-d27d-468e-827c-3a8c66435f86"
plt.figure()
for name, acc_list in acc_dict.items():
  steps = [s for (s,a) in acc_list]
  accs = [a for (s,a) in acc_list]
  plt.plot(steps, accs, '-o', label=name)

plt.legend(fontsize=16)
plt.title("Test accuracy of SGMCMC samplers on MNIST with a CNN")
plt.xlabel("Iterations", size=20)
plt.ylabel("Test accuracy", size=20)
