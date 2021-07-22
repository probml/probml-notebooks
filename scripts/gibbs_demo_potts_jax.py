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

# + [markdown] id="wB8eX0M9r9a3"
# # The math

# + [markdown] id="pJSNoqiKiqNK"
# The potts model
# $$p(x) = \frac{1}{Z}\exp{-\mathcal{E}(x)}\\
# \mathcal{E}(x) = - J\sum_{i\sim j}\mathbb{I}(x_i = x_j)\\
# p(x_i = k | x_{-i}) = \frac{\exp(J\sum_{n\in \text{nbr}}\mathbb{I}(x_n = k))}{\sum_{k'}\exp(J\sum_{n\in \text{nbr}}\mathbb{I}(x_n = k))}$$ 

# + [markdown] id="Ha2X3MkZjVoL"
# In order to efficiently compute 
# $$
# \sum_{n\in \text{nbr}}$$ 
# for all the different states in our potts model we use a convolution. The idea is to first reperesent each potts model state as a one-hot state and then apply a convolution to compute the logits. 

# + [markdown] id="mZuX38K4mTy_"
# $$\begin{pmatrix}
# S_{11} & S_{12} & \ldots & S_{1n} \\
# S_{21} & S_{22} & \ldots & S_{2n} \\
# \vdots & &\ddots & \vdots\\
# S_{n1} & S_{n2} & \ldots & S_{nn} \\
#  \end{pmatrix} \underset{\longrightarrow}{\text{padding}} \begin{pmatrix}
#  0 & \ldots & 0 & \ldots & 0 & 0\\
# 0 & S_{11} & S_{12} & \ldots & S_{1n} & 0 \\
# 0 & S_{21} & S_{22} & \ldots & S_{2n}&0 \\
# \vdots & &\ddots & \vdots\\
# 0 & S_{n1} & S_{n2} & \ldots & S_{nn} & 0 \\
# 0 & \ldots & 0 & \ldots & 0 & 0\\
#  \end{pmatrix} \underset{\longrightarrow}{\text{convolution}} \begin{pmatrix}
# E_{11} & E_{12} & \ldots & E_{1n} \\
# E_{21} & E_{22} & \ldots & E_{2n} \\
# \vdots & &\ddots & \vdots\\
# E_{n1} & E_{n2} & \ldots & E_{nn} \\
#  \end{pmatrix} $$ 

# + [markdown] id="pjkZLlz8qepC"
# An example
# $$\begin{pmatrix}
# 1 & 1 & 1 \\
# 1 & 1 & 1 \\
# 1 & 1 & 1 
#  \end{pmatrix} \underset{\longrightarrow}{\text{padding}} \begin{pmatrix}
#  0 & 0 & 0 & 0 & 0\\
# 0 & 1 & 1 & 1 & 0 \\
# 0 & 1 & 1 & 1 & 0\\
# 0 & 1 & 1 & 1 & 0 \\
# 0 & 0 & 0 & 0 & 0
#  \end{pmatrix} \underset{\longrightarrow}{\text{convolution}} \begin{pmatrix}
# 2 & 3  & 2 \\
# 3 & 4 & 3 \\
# 2 & 3  & 2
#  \end{pmatrix} $$ 

# + [markdown] id="XARcMmGZrPGF"
# Where the matrix $$\begin{pmatrix}
# 2 & 3  & 2 \\
# 3 & 4 & 3 \\
# 2 & 3  & 2
#  \end{pmatrix} $$ correspond to the number of neighbours with the same value around in the matrix \begin{pmatrix}
# 1 & 1 & 1 \\
# 1 & 1 & 1 \\
# 1 & 1 & 1 
#  \end{pmatrix} 

# + [markdown] id="RYfQg2U_rkZk"
# For more than 2 states, we represent the abobe matrix as a 3d tensor which you can imagine as the state matrix but with each element as a one hot vector. 

# + [markdown] id="EHJcunQyiGk7"
# # Import libaries

# + id="85GN9Dk3l9DB"
import jax
import jax.numpy as jnp 
from jax import lax
from jax import vmap
from jax import random
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# + [markdown] id="canIPYzdiJZ1"
# # The K number of states and size of the board

# + id="gwydLmZbe6bw"
K= 10
ix = 128
iy = 128

# + [markdown] id="6CL6cxWoiTOT"
# # The key and the kernel

# + id="EAvOkb4XksB5"
key = random.PRNGKey(12234)

# + id="e_FUJlaNYtSP"
kernel = jnp.zeros((3, 3, 1, 1), dtype=jnp.float32)
kernel += jnp.array([[0, 1, 0],
                     [1, 0,1],
                     [0,1,0]])[:, :, jnp.newaxis, jnp.newaxis]

dn = lax.conv_dimension_numbers((K, ix, iy, 1),     # only ndim matters, not shape
                                 kernel.shape,  # only ndim matters, not shape 
                                ('NHWC', 'HWIO', 'NHWC'))  # the important bit

# + [markdown] id="U696TCZ3iXjB"
# # Creating the checkerboard 

# + id="96V2oEqqWti3"
mask = jnp.indices((K, iy, ix, 1)).sum(axis=0) % 2


# + id="9D_sZ2pRWuj1"
def checkerboard_pattern1(x):
  return mask[0, :, : , 0]

def checkerboard_pattern2(x):
  return mask[1, :, : , 0]

def make_checkerboard_pattern1():
  arr = vmap(checkerboard_pattern1, in_axes=0)(jnp.array(K*[1]))
  return jnp.expand_dims(arr, -1)

def make_checkerboard_pattern2():
  arr = vmap(checkerboard_pattern2, in_axes=0)(jnp.array(K*[1]))
  return jnp.expand_dims(arr, -1)


# + id="pS0O_5ztel-l"
def test_state_mat_update(state_mat_update):
  """
  Checking the checkerboard pattern is the same for each channel
  """
  mask = make_checkerboard_pattern1()
  inverse_mask = make_checkerboard_pattern2()
  state_mat = jnp.zeros((K, 128, 128, 1))
  sample = jnp.ones((K, 128, 128, 1))
  new_state = state_mat_update(mask, inverse_mask, sample, state_mat)
  assert jnp.array_equal(new_state[0, :, :, 0], new_state[1, :, :, 0])

def test_state_mat_update2(state_mat_update):
  """
  Checking the checkerboard pattern is the same for each channel
  """
  mask = make_checkerboard_pattern1()
  inverse_mask = make_checkerboard_pattern2()
  state_mat = jnp.ones((K, 128, 128, 1))
  sample = jnp.zeros((K, 128, 128, 1))
  new_state = state_mat_update(mask, inverse_mask, sample, state_mat)
  assert jnp.array_equal(new_state[0, :, :, 0], new_state[1, :, :, 0])

def test_energy(energy):
  """
  If you give the convolution all ones, it will produce the number of edges 
  it is connected to on a grid i.e the number of neighbours around it. 
  """
  X = jnp.ones((3, 3))
  state_mat = jax.nn.one_hot(X, K, axis=0)[:, :, :, jnp.newaxis]
  energy = energy(state_mat, 1)
  assert np.array_equal(energy[1,:,:,0], 
                        jnp.array([[2,3,2], [3, 4, 3], [2, 3, 2]]))


# + id="aoL_FB675xcE"
def sampler(K, key, logits):
  # Sample from the energy using gumbel trick
  u = random.uniform(key, shape=(K, ix, iy, 1))
  sample = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=0)
  sample = jax.nn.one_hot(sample, K, axis=0)
  return sample

def state_mat_update(mask, inverse_mask, sample, state_mat):
  # Update the state_mat using masking
  masked_sample = mask*sample 
  masked_state_mat = inverse_mask*state_mat 
  state_mat  = masked_state_mat+masked_sample
  return state_mat

def energy(state_mat, jvalue):
  # Calculate energy
  logits = lax.conv_general_dilated(state_mat, jvalue*kernel, 
                                    (1,1), 'SAME', (1,1), (1,1), dn)  
  return logits

def gibbs_sampler(key, jvalue, niter=1):
  key, key2 = random.split(key)
  
  X = random.randint(key, shape=(ix, iy), minval=0, maxval=K)
  state_mat = jax.nn.one_hot(X, K, axis=0)[:, :, :, jnp.newaxis]

  mask = make_checkerboard_pattern1()
  inverse_mask = make_checkerboard_pattern2()
  
  @jit
  def state_update(key, state_mat, mask, inverse_mask):
    logits = energy(state_mat, jvalue)  
    sample = sampler(K, key, logits)
    state_mat = state_mat_update(mask, inverse_mask, sample, state_mat)
    return state_mat

  for iter in tqdm(range(niter)):
    key, key2 = random.split(key2)
    state_mat = state_update(key, state_mat, mask, inverse_mask )
    mask, inverse_mask = inverse_mask, mask
      
  return jnp.squeeze(jnp.argmax(state_mat, axis=0), axis=-1)


# + [markdown] id="3cCjF0RTr3Dv"
# # Running the test

# + id="4VX58dRNg_pF"
test_state_mat_update(state_mat_update)
test_state_mat_update2(state_mat_update)
test_energy(energy)

# + [markdown] id="G1jxMXGgr5LO"
# # Running the model

# + id="2poygXwns9Gu"
Jvals = [1.42, 1.43, 1.44]

# + colab={"base_uri": "https://localhost:8080/", "height": 185, "referenced_widgets": ["59382bbeee02433b823c3f58099783cb", "ff4fefdf30b94aaa8eb4bb5cb1a35ea7", "bc93265d7bc644a29a8993efc365442a", "6237cb298df2434682b8bb5692f69d21", "da8f0af268284e7fbc6ee35ff8771baa", "64f188f566464b8c935f2ad9008608ed", "05ae5a4d5a6b410a864e8b6324668cc1", "a5c38bfc6ff44cdabb9af0bae8a3373b"]} id="4DLUNREjz2gz" outputId="12acd6b4-32cf-4c8d-9b14-3f269db84cd8"
gibbs_sampler(key, 1, niter=2)

# + colab={"base_uri": "https://localhost:8080/", "height": 391, "referenced_widgets": ["6bf23f5839b047f8a4c6cd4e29084212", "a307cfb915bd45fc885cabfb21d5e7d4", "8841ff5f33154e26abbd56e336573b92", "e1b7ba83610b4999a94cdc8ad57881c7", "9bcad2121a974028883b6e8059b895ac", "b846949aa3cf42c1b91fe3c1dc2fe166", "016407df18d44895b665e89c593ab1cb", "c745769c904b4ba485130b65b676a3b7", "69a87679be3a4279976a4a5bdcfd232f", "6229f7b060e7452d93fcc2584447192c", "9780fc3c915544919336b745e73a44ff", "2a4775131d944b3db962a7ff8937ff63", "4c58c34db15848fcbf508ad146d14bd2", "98ea1535e3664c358c837c0f004572da", "0f4b4c6bc0934db0b820c2fd547da488", "6d1df69e990948fcbb05c173ed7f333f", "6370fbea05e8471abe37311912316ea3", "da89e7576f714ad08e6383d663f880b5", "41f1eb0bedda4c899ee40241309153b2", "4eb980c4fc5b453595c6ce69e11baa43", "fc7cc73c7d934e54aea993d719b0c65f", "441623ef7c8942dbbb4968690b40108a", "40a705d0bde1443da0b51d203fce65c1", "5691975b221147b48de5050e97f585ba", "9669df99d61145298168425b0ec56f6f", "08e5d00299c94751bfcb1092aed772b4", "7b7700554cf0401ea51cb1389aabebde", "ef94fdbbd91d4cfabb784c8680b8eb71", "2ee2acb5e7934239ae986e96076fe90d", "527042ec355343ce99a1ea76d4d96b19", "68203780498147d4b0fb3d0e6d5da0ee", "9a727e64b3714807bfb885acb19c3cda"]} id="78pi2s6mtABW" outputId="2f9ce70b-9d21-4c10-a980-daaf85d10711"
dfig, axs = plt.subplots(1, len(Jvals), figsize=(8, 8))
for t in tqdm(range(len(Jvals))):
  arr = gibbs_sampler(key, Jvals[t], niter=8000)
  axs[t].imshow(arr, cmap='Accent', interpolation="nearest")
  axs[t].set_title(f"J = {Jvals[t]}")

# + id="gAQtirK6kGja"

