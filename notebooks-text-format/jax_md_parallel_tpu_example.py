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

# + [markdown] id="gn_7TYq4nJ1y"
# # Parallel simulation of a bubble raft
# **Important**: This code also works on multi-host TPU setup without any changes !! The key thing to do with a multi-host TPU setup is to ssh the file and run it on all the host at the same time. In order to do that please refer to this [notebook](https://github.com/probml/probml-notebooks/blob/main/notebooks/tpu_colab_tutorial.ipynb).

# + [markdown] id="0_YmOEQHmdYp"
# This notebook is based on the first example from the [JAX MD cookbook](https://colab.research.google.com/github/google/jax-md/blob/master/notebooks/jax_md_cookbook.ipynb) i.e the simulating bubble raft example ![alt text](https://upload.wikimedia.org/wikipedia/commons/8/89/Bubblerraft2.jpg)

# + [markdown] id="TLcqQSFtUwKT"
#  ## Installation

# + colab={"base_uri": "https://localhost:8080/"} id="k1BbpnW5TnzC" outputId="06bced29-8e56-409f-b9fd-8bbdb9d69c3b"
# !pip install -q git+https://www.github.com/google/jax-md

# + colab={"base_uri": "https://localhost:8080/"} id="yd1MA81jeCL3" outputId="bde1b03e-2a29-4415-a645-c31f6fbf9598"
import jax
try:
  import jax.tools.colab_tpu
  jax.tools.colab_tpu.setup_tpu()
except KeyError:
  import os 
jax.devices()

# + id="D-VfDPa0TRuZ"
import os
import jax.numpy as np

from jax import jit
from jax import vmap, pmap
import jax.numpy as jnp
from jax import random
from jax import lax


from jax_md import space
from jax_md import simulate
from jax_md import energy
os.environ['XLA_USE_32BIT_LONG'] = '1'


# + colab={"base_uri": "https://localhost:8080/"} id="dpv-UiZWidCo" outputId="b1f270db-2326-4d54-ecae-253f4a30cd45"
jax.local_device_count()

# + [markdown] id="w8r01QugUyU6"
# ## Hyperparameters

# + id="9wVq3mNdTiIl"
f32 = np.float32
ensemble_size = 1000
nlocal=8
N = 32
dt = 1e-1
simulation_steps = np.arange(1000)
key = random.PRNGKey(0)


# + [markdown] id="TLztCRKBU0T0"
# ## Defining the box and the energy function

# + id="YUzPX3FPTt18"
def box_size_at_number_density(particle_count, number_density):
  return f32((particle_count / number_density) ** 0.5)

box_size = box_size_at_number_density(particle_count=N, number_density=1)
displacement, shift = space.periodic(box_size)
energy_fun = energy.soft_sphere_pair(displacement)


# + [markdown] id="ZvO5aX2FU446"
# ## Defining the solution

# + id="18Vj_izVTwUa"
def simulation(key, temperature):
  pos_key, sim_key = random.split(key)

  R = random.uniform(pos_key, (N, 2), maxval=box_size)

  init_fn, apply_fn = simulate.brownian(energy_fun, shift, dt, temperature)
  state = init_fn(sim_key, R)

  do_step = lambda state, t: (apply_fn(state, t=t), t)
  state, _ = lax.scan(do_step, state, simulation_steps)
  return state.position



# + [markdown] id="rEi7woP6U7Q_"
# ## Parallelsing the simulation

# + id="jJ2UJkZCT0p2"
vectorized_simulation = vmap(simulation, in_axes=(0, None))
parallel_vectorized_simulation = pmap(vectorized_simulation, in_axes=(0, None))

# + id="A1yzxGoFUbOe"
vectorized_energy = vmap(energy_fun)
parallel_vectorized_energy = pmap(vectorized_energy)

# + [markdown] id="8x5m4GiCVB1F"
# ## Getting the random keys

# + id="WXTSjQaSUBrC"
simulation_keys_lst = []
for i in range(nlocal):
  key, *simulation_keys = random.split(key, ensemble_size+1)
  simulation_keys = jnp.stack(simulation_keys)
  simulation_keys_lst.append(simulation_keys)
simulation_keys = jnp.stack(simulation_keys_lst)

# + [markdown] id="5PwlRUx3VETH"
# ## Running the simulation

# + id="bSLQye23UHM5" colab={"base_uri": "https://localhost:8080/"} outputId="2de80034-2443-4897-8e32-1a14c843238c"
bubble_positions = parallel_vectorized_simulation(simulation_keys, 1e-5)
bubble_energies = parallel_vectorized_energy(bubble_positions)

# + colab={"base_uri": "https://localhost:8080/", "height": 291} id="_tmfwOr9lx5a" outputId="77d9f9bd-7f65-4f02-f2e6-b070dbddd9d5"
import numpy as onp 
import matplotlib.pyplot as plt 

def format_plot(x, y):  
  plt.xlabel(x, fontsize=20)
  plt.ylabel(y, fontsize=20)

bubble_energies = jax.pmap(lambda x: jax.lax.all_gather(x, 'i'), axis_name='i')(bubble_energies)[0]
counts, bins = onp.histogram(bubble_energies, bins=50)
plt.plot(bins[:-1] * 10 ** 5, counts, 'o')
format_plot('$E\\times 10 ^{-5}$', '$P(E)$')
plt.savefig("plot.png")

# + id="8sctHDSUlU6d"

