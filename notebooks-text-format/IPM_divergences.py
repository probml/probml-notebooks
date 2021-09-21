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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/IPM_divergences.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="FygybCkXMl9W"
# # Critics in IPMs variational bounds on $f$-divergences
#
# Author: Mihaela Rosca 
#
# This colab uses a simple example (two 1-d distributions) to show how the critics of various IPMs (Wasserstein distance and MMD) look like. We also look at how smooth estimators (neural nets) can estimte density ratios which are not 
# smooth, and how that can be useful in providing a good learning signal for a model.

# + id="tBtObwhMwgbb"
import jax 
import random
import numpy as np
import jax.numpy as jnp

import seaborn as sns
import matplotlib.pyplot as plt

import scipy

# + colab={"base_uri": "https://localhost:8080/"} id="O6tXOZ88iOVq" outputId="ff17d796-da30-4f45-bfd5-c101ca78bb24"
# !pip install dm-haiku
# !pip install optax

# + id="icX1eUTBihny"
import haiku as hk
import optax

# + id="2iP2urBbxHD9"
sns.set(rc={"lines.linewidth": 2.8}, font_scale=2)
sns.set_style("whitegrid")

# + [markdown] id="CNTZXm0GjqRU"
# # KL and non overlapping distributions
#
# * non overlapping distributions (visual)
# * explain ratio will be infinity - integral
# * move the distributions closer and they will not have signal

# + id="Y-MuCBTdlE5P"
import scipy.stats

# + id="xEJ2TtFBlSW_"
 from scipy.stats import truncnorm
 from scipy.stats import beta


# + id="KxacaTuxD14p"
#  We allow a displacement from 0 of the beta distribution.
class TranslatedBeta():

  def __init__(self, a, b, expand_dims=False, displacement=0):
    self._a = a
    self._b = b
    self.expand_dims = expand_dims
    self.displacement = displacement

  def rvs(self, size):
    val = beta.rvs(self._a, self._b, size=size) + self.displacement
    return np.expand_dims(val, axis=1) if self.expand_dims else val
  
  def pdf(self, x):
    return beta.pdf(x - self.displacement, self._a, self._b)


# + id="f5jtHWY8EQOx"
p_param1 = 3
p_param2 = 5

q_param1 = 2
q_param2 = 3


start_p = 0
start_r = 1
start_q = 2

p_dist = TranslatedBeta(p_param1, p_param2, displacement=start_p)
q_dist = TranslatedBeta(q_param1, q_param2, displacement=start_q)
r_dist = TranslatedBeta(q_param1, q_param2, displacement=start_r)

# + id="mdtdEstVjuUR" colab={"base_uri": "https://localhost:8080/", "height": 592} outputId="514f5e7e-88a8-4495-bc45-cbe4699ff5fc"
plt.figure(figsize=(14,10))

p_x_samples = p_dist.rvs(size=15)
q_x_samples = q_dist.rvs(size=15)

p_linspace_x = np.linspace(start_p, start_p + 1, 100)
p_x_pdfs = p_dist.pdf(p_linspace_x)

q_linspace_x = np.linspace(start_q, start_q + 1, 100)
q_x_pdfs = q_dist.pdf(q_linspace_x)

plt.plot(p_linspace_x, p_x_pdfs, 'b', label=r'$p_1(x)$')
plt.plot(p_x_samples, [0] * len(p_x_samples), 'bo', ms=10)

plt.plot(q_linspace_x, q_x_pdfs, 'r', label=r'$p_2(x)$')
plt.plot(q_x_samples, [0] * len(q_x_samples), 'rd', ms=10)

plt.ylim(-0.5, 2.7)
plt.xlim(-0.2, 3.5)
plt.axis('off')
plt.legend()
plt.xticks([])
plt.yticks([])

# + colab={"base_uri": "https://localhost:8080/", "height": 483} id="1djktHYWhpo6" outputId="437bd3b5-0633-469d-b906-a28b2d0d0201"
plt.figure(figsize=(14,8))

local_start_p = 0
local_start_r = 1.2
local_start_q = 2.4

local_p_dist = TranslatedBeta(p_param1, p_param2, displacement=local_start_p)
local_q_dist = TranslatedBeta(q_param1, q_param2, displacement=local_start_q)
local_r_dist = TranslatedBeta(q_param1, q_param2, displacement=local_start_r)

p_linspace_x = np.linspace(local_start_p, local_start_p + 1, 100)
q_linspace_x = np.linspace(local_start_q, local_start_q + 1, 100)
r_linspace_x = np.linspace(local_start_r, local_start_r + 1, 100)

p_x_pdfs = local_p_dist.pdf(p_linspace_x)
q_x_pdfs = local_q_dist.pdf(q_linspace_x)
r_x_pdfs = local_r_dist.pdf(r_linspace_x)

plt.plot(p_linspace_x, p_x_pdfs, 'b')
plt.plot(q_linspace_x, q_x_pdfs, 'r')
plt.plot(r_linspace_x, r_x_pdfs, 'g')

num_samples = 15
plt.plot(local_p_dist.rvs(size=num_samples), [0] * num_samples, 'bo', ms=10, label=r'$p^*$')
plt.plot(local_q_dist.rvs(size=num_samples), [0] * num_samples, 'rd', ms=10, label=r'$q(\theta_1)$')
plt.plot(local_r_dist.rvs(size=num_samples), [0] * num_samples, 'gd', ms=10, label=r'$q(\theta_2)$')

plt.ylim(-0.5, 2.7)
plt.xlim(-0.2, 3.5)
plt.axis('off')
plt.legend(framealpha=0)
plt.xticks([])
plt.yticks([])

# + [markdown] id="74amMJnQVdSH"
# # Approximation of the ratio using the f-gan approach

# + id="cmRk0kHTZnhl"
model_transform = hk.without_apply_rng(hk.transform(lambda *args, **kwargs: hk.Sequential([
        hk.Linear(10),
        jax.nn.relu,
        hk.Linear(10),
        jax.nn.tanh,
        hk.Linear(40),
        hk.Linear(1)])(*args, **kwargs)))

# + id="eKdV_0NECYtK"
BATCH_SIZE = 100
NUM_UPDATES = 1000

# + id="mTzpe00TV3oi"
dist1 = TranslatedBeta(p_param1, p_param2, expand_dims=True, displacement=start_p)
dist2 = TranslatedBeta(q_param1, q_param2, expand_dims=True, displacement=start_q)


# + id="watywW01czZZ"
@jax.jit
def estimate_kl(params, dist1_batch, dist2_batch):
  dist1_logits = model_transform.apply(params, dist1_batch)
  dist2_logits = model_transform.apply(params, dist2_batch)
  return jnp.mean(dist1_logits - jnp.exp(dist2_logits -1))


# + id="FRqoQjs9d-0D"
def update(params, opt_state, dist1_batch, dist2_batch):
    model_loss = lambda *args: - estimate_kl(*args)
    loss, grads = jax.value_and_grad(model_loss, has_aux=False)(params, dist1_batch, dist2_batch)
    params_update, new_opt_state = optim.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, params_update)
    return loss, new_params, new_opt_state


# + id="9QaVK5_IblRq"
NUM_UPDATES = 200

# + id="_i3rEevWclf7" colab={"base_uri": "https://localhost:8080/"} outputId="eda90b11-5dae-400d-d2f2-967798ed599a"
rng = jax.random.PRNGKey(1)
init_model_params = model_transform.init(rng, dist1.rvs(BATCH_SIZE))

# + id="MUmsbCTI8uhB" colab={"base_uri": "https://localhost:8080/"} outputId="23a7e428-1e51-45c4-b320-e9b44c5954cc"
params = init_model_params
optim = optax.adam(learning_rate=0.0005, b1=0.9, b2=0.999)
opt_state = optim.init(init_model_params)

for i in range(NUM_UPDATES):
   # Get a new batch of data
   x = dist1.rvs(BATCH_SIZE)
   y = dist2.rvs(BATCH_SIZE)
   loss, params, opt_state = update(params, opt_state, x, y)

   if i % 50 == 0:
    print('Loss at {}'.format(i))
    print(loss)

# + id="n_TWrjUCXnsM"
plotting_x = np.expand_dims(np.linspace(-1., 3.5, 100), axis=1)

#  TODO: how do you get the ratio values form the estimate - need to check the fgan paper
ratio_values = model_transform.apply(params, plotting_x)
# ratio_values = 1 + np.log(model_transform.apply(params, plotting_x))

# + id="21jJhmc_W5yU" colab={"base_uri": "https://localhost:8080/", "height": 551} outputId="7255c930-e8a4-45fc-b229-1a2cf564b72d"
plt.figure(figsize=(14,8))


p_linspace_x = np.linspace(start_p, start_p + 1, 100)
q_linspace_x = np.linspace(start_q, start_q + 1, 100)

plt.plot(p_linspace_x, p_x_pdfs, 'b', label=r'$p^*$')
plt.plot(p_x_samples, [0] * len(p_x_samples), color='b', marker=10, linestyle="None", ms=18)

plt.plot(q_linspace_x, q_x_pdfs, 'g', label=r'$q(\theta)$')
plt.plot(q_x_samples, [0] * len(q_x_samples), color='g', marker=11, linestyle="None", ms=18)

x = np.linspace(-1, 3.5, 200)

ratio = p_dist.pdf(x) / q_dist.pdf(x) 

plt.hlines(6.1, -0.6, start_q, linestyles='--', color='r')
plt.hlines(6.1, start_q+1, 3.5, linestyles='--', color='r')
plt.text(3.4, 5.6, r'$\infty$')

plt.plot(x, ratio, 'r', label=r'$\frac{p^*}{q(\theta)}$', linewidth=4)
plt.plot(plotting_x, ratio_values[:, 0].T, color='darkgray', label=r'MLP approx to $\frac{p^*}{q(\theta)}$', linewidth=4)

plt.ylim(-2.5, 8)
plt.xlim(-0.2, 3.5)
plt.axis('off')

plt.legend(loc='upper center', bbox_to_anchor=(0.35, 0., 0.25, 1.), ncol=4, framealpha=0)
plt.xticks([])
plt.yticks([])

# + [markdown] id="rSW8c5dLO9NO"
# ## Gradients 
#
#
# In order to see why the learned density ratio has useful properties for learning, we can plot the gradients of the learned density ratio across the input space

# + colab={"base_uri": "https://localhost:8080/", "height": 534} id="BZl2hXzwQ_4O" outputId="41eaf8f7-beaf-4d93-905e-bf612a4ba389"
plt.figure(figsize=(14,8))

grad_fn = jax.grad(lambda x: model_transform.apply(params, x)[0])

grad_values = jax.vmap(grad_fn)(plotting_x)


plt.figure(figsize=(14,8))


p_linspace_x = np.linspace(start_p, start_p + 1, 100)
q_linspace_x = np.linspace(start_q, start_q + 1, 100)

plt.plot(p_linspace_x, p_x_pdfs, 'b', label=r'$p^*$')
plt.plot(p_x_samples, [0] * len(p_x_samples), color='b', marker=10, linestyle="None", ms=18)

plt.plot(q_linspace_x, q_x_pdfs, 'g', label=r'$q(\theta)$')
plt.plot(q_x_samples, [0] * len(q_x_samples), color='g', marker=11, linestyle="None", ms=18)

x = np.linspace(-1, 3.5, 200)

ratio = p_dist.pdf(x) / q_dist.pdf(x) 

plt.hlines(5.8, -0.6, start_q, linestyles='--', color='r')
plt.hlines(5.8, start_q+1, 3.5, linestyles='--', color='r')
plt.text(3.4, 5.4, r'$\infty$')

plt.plot(x, ratio, 'r', label=r'$\frac{p^*}{q(\theta)}$', linewidth=4)
plt.plot(plotting_x, ratio_values[:, 0].T, color='darkgray', label=r'$f_{\phi}$ approximating $\frac{p^*}{q(\theta)}$', linewidth=4)
plt.plot(plotting_x, grad_values[:, 0].T, color='orange', label=r'$\nabla_{x} f_{\phi}(x)$', linewidth=4, ls='-.')


plt.ylim(-2.5, 8)
plt.xlim(-0.2, 3.5)
plt.axis('off')

plt.legend(loc='upper center', bbox_to_anchor=(0.35, 0., 0.25, 1.), ncol=4, framealpha=0)
plt.xticks([])
plt.yticks([])

# + [markdown] id="tYJsW_d0NqAn"
# # Wasserstein distance for the same two distributions
#
#
# Computing the Wasserstein critic in 1 dimension. Reminder that the Wasserstein distance is defined as:
# $$
# W(p, q) = \sup_{\|\|f\|\|_{Lip} \le 1} E_p(x) f(x) - E_q(x) f(x)
# $$
#
# The below code finds the values of f evaluated at the samples of the two distributions. This vector is computed to maximise the empirical (Monte Carlo) estimate of the IPM:
# $$
#   \frac{1}{n}\sum_{i=1}^n f(x_i) - \frac{1}{m}\sum_{i=1}^m f(y_j)
# $$
#
# where $x_i$ are samples from the first distribution, while $y_j$ are samples
# from the second distribution. Since we want the function $f$ to be 1-Lipschitz, 
# an inequality constraint is added to ensure that for all two choices of samples 
# in the two distributions, $\forall x \in \{x_1, ... x_n, y_1, ... y_m\}, \forall y \in \{x_1, ... x_n, y_1, ... y_m\}$ 
# $$
#   f(x) - f(y) \le |x - y| \\
#   f(y) - f(x) \le |x - y| \\
# $$
#
# This maximisation needs to occur under the constraint that the function $f$
# is 1-Lipschitz, which is ensured uisng the constraint on the linear program.
#
# Note: This approach does not scale to large datasets.
#
# Thank you to Arthur Gretton and Dougal J Sutherland for this version of the code.
#

# + id="HlnCHi1fRIsC"
from scipy.optimize import linprog


# + id="M6jLecerRNul"
def get_W_witness_spectrum(p_samples, q_samples):
    n = len(p_samples)
    m = len(q_samples)
    X = np.concatenate([p_samples, q_samples], axis=0)

    ## AG:  repeat [-1/n] n times 
    c = np.array(n*[-1/n] + m*[1/m])
    A_ub, b_ub = [], []
    for i in range(n + m):
        for j in range(n + m):
            if i == j:
                continue
            z = np.zeros(n + m)
            z[i] = 1
            z[j] = -1
            A_ub.append(z)
            b_ub.append(np.abs(X[i] - X[j]))

    ## AG: Minimize: c^T * x
    ## Subject to: A_ub * x <= b_ub
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, method='simplex', options={'tol':1e-5})
    a = res['x']


    ## AG:  second argument xs to be passed into the internal
    ## function.
    def witness_spectrum(x):
        diff = np.abs(x - X[:, np.newaxis])
        one = np.min(a[:, np.newaxis] + diff, axis=0)
        two = np.max(a[:, np.newaxis] - diff, axis=0)
        return one, two
    return witness_spectrum

# + id="Ly0-9XUETI1S"
x = np.linspace(-1, 3.5, 100)

wass_estimate = get_W_witness_spectrum(p_x_samples + start_p, q_x_samples + start_q)(x)

wa, wb = wass_estimate
w = (wa + wb) / 2
w -= w.mean()

# + id="ekEdkT9jRWmn" colab={"base_uri": "https://localhost:8080/", "height": 474} outputId="50639461-853a-4928-cf24-60ad237ac7c1"
plt.figure(figsize=(14,6))

display_offset = 0.8

plt.plot(p_linspace_x, display_offset + p_x_pdfs, 'b', label=r'$p^*$')
plt.plot(p_x_samples, [display_offset] * len(p_x_samples), color='b', marker=10, linestyle="None", ms=18)

plt.plot(q_linspace_x, display_offset + q_x_pdfs, 'g', label=r'$q(\theta)$')
plt.plot(q_x_samples, [display_offset] * len(q_x_samples), color='g', marker=11, linestyle="None", ms=18)

x = np.linspace(-1, 3.5, 100)
plt.plot(x, w + display_offset, 'r', label=r'$f^{\star}$', linewidth=4)

plt.ylim(-2.5, 8)
plt.xlim(-0.2, 3.5)
plt.axis('off')
plt.legend(loc='upper center', bbox_to_anchor=(0.35, 0., 0.5, 1.34), ncol=3, framealpha=0)
plt.xticks([])
plt.yticks([])


# + [markdown] id="SWkGFs0w0GSR"
# ## MMD computation
#
# The MMD is an IPM defined as:
# $$
# MMD(p, q) = \sup_{\|\|f\|\|_{\mathcal{H}} \le 1} E_p(x) f(x) - E_q(x) f(x)
# $$
#
# where $\mathcal{H}$ is a RKHS. Using the mean embedding operators in an RKHS, we can write:
# $$
#  E_p(x) f(x) = \langle f, \mu_p \rangle \\
#  E_q(x) f(x) = \langle f, \mu_q \rangle \\
# $$
#
# replacing in the MMD:
#
# $$
# MMD(p, q) = \sup_{\|\|f\|\|_{\mathcal{H}} \le 1} \langle f, \mu_p - \mu_q \rangle
# $$
#
# which means that 
# $$
# f = \frac{\mu_p - \mu_q}{\|\|\mu_p - \mu_q\|\|_{\mathcal{H}}}
# $$
#
# To obtain an estimate of $f$ evaluated at $x$ we use that:
# $$
# f(x) = \frac{\mathbb{E}_{p(y)} k(x, y) - \mathbb{E}_{q(y)} k(x, y)}{\|\|\mu_p - \mu_q\|\|_{\mathcal{H}}}
# $$
#
# to estimate $\|\|\mu_p - \mu_q\|\|_{\mathcal{H}}$ we use:
#
# $$
# \|\|\mu_p - \mu_q\|\|_{\mathcal{H}} = \langle \mu_p - \mu_q, \mu_p - \mu_q \rangle = \langle \mu_p, \mu_p \rangle  + \langle \mu_q, \mu_q \rangle 
#     - 2  \langle \mu_p, \mu_q \rangle 
# $$
#
#
# To estimate the dot products, we use:
# $$
# \langle \mu_p, \mu_p \rangle = E_p(x) \mu_p(x) =  E_p(x) \langle \mu_p, k(x, \cdot) \rangle =  E_p(x) E_p(x') k(x, x')
# $$
#
# For more details see the slides here: http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture5_distribEmbed_1.pdf
#

# + id="kPPQVoQ0Ulhn"
def covariance(kernel_fn, X, Y):
  num_rows = len(X)
  num_cols = len(Y)
  K = np.zeros((num_rows, num_cols))
  for i in range(num_rows):
    for j in range(num_cols):
      K[i, j] = kernel_fn(X[i], Y[j])

  return K



# + id="5chKdIkSVLjp"
def gaussian_kernel(x1, x2, gauss_var=0.1, height=2.2):
  return height * np.exp(- np.linalg.norm(x1 - x2) ** 2 / gauss_var)


# + colab={"base_uri": "https://localhost:8080/", "height": 508} id="JqRHavAuzol7" outputId="180d6f40-de20-4e2d-ed8d-d0fe5f6969b2"
def evaluate_mmd_critic(p_samples, q_samples):
  n = p_samples.shape[0]
  m = q_samples.shape[0]

  p_cov = covariance(gaussian_kernel, p_samples, p_samples)
  print('indices')
  print(np.diag_indices(n))
  p_samples_norm = np.sum(p_cov) - np.sum(p_cov[np.diag_indices(n)])
  p_samples_norm /= n * (n-1)

  q_cov = covariance(gaussian_kernel, q_samples, q_samples)
  q_samples_norm = np.sum(q_cov) - np.sum(q_cov[np.diag_indices(m)])
  q_samples_norm /= m * (m-1)

  p_q_cov = covariance(gaussian_kernel, p_samples, q_samples)
  p_q_norm = np.sum(p_q_cov)
  p_q_norm /= n * m

  norm = p_samples_norm + q_samples_norm - 2 * p_q_norm
  def critic(x):
    p_val = np.mean([gaussian_kernel(x, y) for y in p_samples])
    q_val = np.mean([gaussian_kernel(x, y) for y in q_samples])
    return (p_val - q_val) / norm
  return critic

critic_fn = evaluate_mmd_critic(p_x_samples, q_x_samples)

plt.figure(figsize=(14,6))

display_offset = 0

plt.plot(p_linspace_x, display_offset + p_x_pdfs, 'b', label=r'$p^*$')
plt.plot(p_x_samples, [display_offset] * len(p_x_samples), color='b', marker=10, linestyle="None", ms=18)

plt.plot(q_linspace_x, display_offset + q_x_pdfs, 'g', label=r'$q(\theta)$')
plt.plot(q_x_samples, [display_offset] * len(q_x_samples), color='g', marker=11, linestyle="None", ms=18)

x = np.linspace(-1, 3.5, 100)
plt.plot(start_p + x,  np.array([critic_fn(x_val) for x_val in x]) + display_offset, 'r', label=r'$f^{\star}$', linewidth=4)

plt.ylim(-2.5, 8)
plt.xlim(-0.2, 3.5)
plt.axis('off')
plt.legend(loc='upper center', bbox_to_anchor=(0.35, 0., 0.5, 1.34), ncol=3, framealpha=0)
plt.xticks([])
plt.yticks([])

# + [markdown] id="QMT_TILiMjcH"
#

# + id="ixlViVFMinJR"

