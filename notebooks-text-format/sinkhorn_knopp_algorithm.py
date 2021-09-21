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

# + [markdown] id="LjcyL23CJ7wB"
# # Installing packages

# + [markdown] id="CsaZop7W8_gN"
# The code is from https://michielstock.github.io/posts/2017/2017-11-5-OptimalTransport/

# + id="eFm_J1v8zhXL"
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange

from sklearn.datasets import make_circles
from scipy.spatial import distance_matrix


# + [markdown] id="rxS5_ilRJ3-5"
# # JAX implementation of sinkhorn algorithm

# + id="SrZnIIuP2Eqw"
@jit
def scale_cols_and_rows(P):
  P *= (r / P.sum(1)).reshape((-1, 1))
  P *= (c / P.sum(0)).reshape((1, -1))
  return P


# + id="4PxeX0VA0DX8"
def sinkhorn_knopp_jax(M, r, c, lam, niter=100000):
  M = jnp.array(M)
  n, m = M.shape
  P = jnp.exp(- lam * M)
  P /= P.sum()
  # normalize this matrix
  for i in trange(niter):
      P = scale_cols_and_rows(P)
  return P, jnp.sum(P * M)

def sinkhorn_knopp_np(M, r, c, lam, niter=100000):
  n, m = M.shape
  P = np.exp(- lam * M)
  P /= P.sum()
  # normalize this matrix
  for i in trange(niter):
      P *= (r / P.sum(1))
      P *= (c / P.sum(0))
  return P, np.sum(P * M)


# + id="pjNiCX-a1WYl"
n_samples = 10000
X, y = make_circles(n_samples=n_samples, noise=0.05,
                    factor=0.5, shuffle=False)
X1 = X[y==0]
X2 = X[y==1]

n, m = len(X1), len(X2)

# + [markdown] id="FwcIO__5J0E6"
# # Comparing numpy and JAX

# + colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["f1b59fec4690469d8a1f40dda9925933", "f5b1350cd9c948e98872791a65aaa00f", "05a3ea37f4ea4224bb63469cdde4107e", "8560cdc13b8f4ec5ba0ee6ca41eb020b", "35ce1fc17d544aa7a7de8611600c3e16", "0db880b3f6134dfa9d95f81c0ee12e41", "f587d3c9158847889ee8ac10adec92ff", "fe253ac8b68f44789e2a15804fda4741"]} id="nkygpkAc2sx0" outputId="fd2f2b53-e188-42af-fc7a-d558a8709738"
r = np.ones(n) / n
c = np.ones(m) / m

M = jnp.array(distance_matrix(X1, X2))

P, d = sinkhorn_knopp_jax(M, r, c, lam=30, niter=1000)

# + colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["55ca0aa0edbd4e1389460d0ad335ef7f", "71f6e7139e96400bbe3bc543f30d785d", "91dffb97288f45d2b182d82ea8ac3100", "f0ad9d6fe1b44a37b07e105f2883810e", "a117a60496f24216a86da919372df79f", "f6c14dceebcc4f4792f276530be23eb9", "8a1fad5c8f90416691d9ebdbb533a602", "d26b39c75ca940b2910d71198f6df47e"]} id="o9_6tRbr5BVm" outputId="e4d66dd9-bc24-44ea-8cf1-4f9fd02100f3"
P, d = sinkhorn_knopp_np(M, r, c, lam=30, niter=1000)

# + [markdown] id="ZZepBvuoJlMo"
# # Visualisation of sinkorn algorithm
# **Warning**: Only taking a subset of points to map otherwise the visualisation will look cluttered on matplotlib

# + colab={"base_uri": "https://localhost:8080/", "height": 347, "referenced_widgets": ["4550c99eb14844ef9c5966e6ae900238", "ca6631f26f3f480ea5a7bfda3c5c345a", "248080fe7f5d4e34b0c5df5d0e0ef757", "622cd7dd9b54424baae4dba05e27b508", "0725f04fcab543c0a55cb9bca38326a9", "eb4a66608cae4da198e8595848cecf91", "52aec0ae47694d8b8e7e9e8fc7b797b7", "ef4f57422ef1435a953ece13f918be8a"]} id="zihTTVlA2xs3" outputId="8a2d5e5d-f043-402b-e70e-fbe69f1421bf"
sampling_factor = 100
plt.scatter(X1[:,0], X1[:,1], color="blue")
plt.scatter(X2[:,0], X2[:,1], color="red")
for i in trange(0, n, sampling_factor):
    for j in range(0, m, sampling_factor):
        plt.plot([X1[i,0], X2[j,0]], [X1[i,1], X2[j,1]], color="green",
                alpha=float(P[i,j] * n)*sampling_factor)
plt.title('Optimal matching')

# + id="n6oMAmiR6lnG"

