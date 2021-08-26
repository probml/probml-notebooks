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
# <a href="https://colab.research.google.com/github/Nirzu97/pyprobml/blob/linalg/notebooks/linalg.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="enjVnybrwKxI"
# # Linear algebra
#
# ## TOC:
# * [Basics](#basics)
# * [Sparse matrices](#sparse)
# * [Broadcasting](#broadcasting)
# * [Einstein summation](#einstein)
# * [Eigenvalue decomposition](#EVD)
# * [Singular value decomposition](#SVD)
# * [Other decompositions](#decomp)
# * [Matrix calculus](#calculus)
# * [Linear systems of equations](#linear)
#

# + id="EUUp6AYzyoXp"
# Standard Python libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import numpy as np
np.set_printoptions(precision=3)
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

from IPython import display

import sklearn

import seaborn as sns;
sns.set(style="ticks", color_codes=True)

import pandas as pd
pd.set_option('precision', 2) # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100) # wide windows



# + id="khgbUtr18Srr" colab={"base_uri": "https://localhost:8080/"} outputId="b401998a-5fd5-4bac-9b4e-e8f30e68c8b9"
# https://github.com/google/jax
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import grad, hessian, jacfwd, jacrev, jit, vmap
from jax.experimental import optimizers
print("jax version {}".format(jax.__version__))

# + [markdown] id="btFqRlnXwKxT"
# ## Basics <a class="anchor" id="basics"></a>

# + id="7vg6-g5TwKxU" colab={"base_uri": "https://localhost:8080/"} outputId="62a90cee-0d5b-4409-9aed-bc80e73b7e2f"
# Create 1d vector
v = jnp.array([0,1,2]) # 1d vector
print(v.ndim) ## 1
print(v.shape) ## (3,)


# + id="LwJTGOpawKxW" colab={"base_uri": "https://localhost:8080/"} outputId="505ba2c8-ff8f-4603-b1eb-aef2643a2a19"
# Note that Python uses 0-indexing, not 1-indexing.
# Thus the elements are accessed as follows:
print(v[0], v[1], v[2]) ## 0 1 2

# + id="Ohu3VF56wKxZ" colab={"base_uri": "https://localhost:8080/"} outputId="8578deaa-54ae-440d-e3ff-017dba6f4c76"
# Create 2d array
A = jnp.array([ [0,1,2], [3,4,5] ]) 
print(A)
## [[0, 1, 2],
##  [3, 4, 5]])
print(A.ndim) ## 2
print(A.shape) ## (2,3)
print(A.size) ## 6
print(A.T.shape) ## (3,2)

# + id="Dt8D8bkcwKxc" colab={"base_uri": "https://localhost:8080/"} outputId="9726e656-f7b4-4b0c-ec63-b74d73700f95"
# If we want to make a vector into a matrix with one row, we can use any of the following:

x = jnp.array([1,2]) # vector
X1 = jnp.array([x]) # matrix with one row
X2 = jnp.reshape(x, (1,-1))
X3 = x[None, :]
X4 = x[jnp.newaxis, :]
assert jnp.array_equal(X1, X2)
assert jnp.array_equal(X1, X3)
print(jnp.shape(X1)) ## (1,2)


# + id="w5HA6oTtwKxe" colab={"base_uri": "https://localhost:8080/"} outputId="dad8efd7-c81d-4363-b89d-ff9ccfe638ca"
# If we want to make a vector into a matrix with one column, we can use any of the following:
x = jnp.array([1,2]) # vector
X1 = jnp.array([x]).T # matrix with one column
X2 = jnp.reshape(x, (-1,1))
X3 = x[:, None]
X4 = x[:, jnp.newaxis]
assert jnp.array_equal(X1, X2)
assert jnp.array_equal(X1, X3)
print(jnp.shape(X1)) ## (2,1)


# + id="XWYnykp-wKxg" colab={"base_uri": "https://localhost:8080/"} outputId="298c3667-c884-4573-83aa-0ceeccfa9d93"
# Here is how to create a one-hot encoding of integers.

def one_hot(x, k, dtype=jnp.float32):
  return jnp.array(x[:, None] == jnp.arange(k), dtype)


# Example
x = jnp.array([1,2,0,2]);
X = one_hot(x, 3)
print(X)



# + id="waqYvW57wKxj"

# We can construct arrays from a list of column vectors as follows:
A1 = jnp.array([ [0,1,2], [3,4,5] ]) 
col0 = A1[:,0]; col1 = A1[:,1]; col2=A1[:,2];
A2 = jnp.stack([col0,col1,col2],axis=1)
assert jnp.array_equal(A1, A2)

# We can construct arrays from a list of row vectors as follows:
row0=A1[0,:]; row1=A1[1,:];
A2 = jnp.stack([row0,row1],axis=0)
assert jnp.array_equal(A1, A2)

# + id="yg3iV66SwKxm" colab={"base_uri": "https://localhost:8080/"} outputId="28373128-ece6-46f9-ab5b-3ae60c942bb6"
# We can construct arrays from a list of arrays
# using the hstack or vstack functions,
# which stack horizontally or vertically,  as illustrated below.

M = jnp.array([[9,8,7],[6,5,4]])
C = jnp.array([[99], [99]])
A1 = jnp.concatenate([M, C], axis=1)
A2 = jnp.hstack([M, C])
#A3 = jnp.c_[M, C] # c_ does not work in jax
assert jnp.array_equal(A1, A2)
#assert jnp.array_equal(A1, A3)
print(A1)



# + id="iAsvZGtGwKxq" colab={"base_uri": "https://localhost:8080/"} outputId="b20c8c52-bd9b-45f5-89a5-af53d54b28fa"

R = jnp.array([[1,2,3]])
A1 = jnp.concatenate([R, M], axis=0)
A2 = jnp.vstack([R, M])
assert jnp.array_equal(A1, A2)
print(A1)


# + id="nTTeMiMhwKxt" colab={"base_uri": "https://localhost:8080/"} outputId="09e34343-d695-4301-c40e-f12bf35420e6"
# A very common idiom  is to add a column of 1s to a datamatrix.
# We can do this using horizontal stacking (along the columns) as follows.

X = jnp.array([[9,8,7],[6,5,4]])
N = jnp.shape(X)[0] # num. rows
X1 = jnp.hstack([jnp.ones((N,1)), X])
print(X1)

# + id="-QNsSTE8wKxw" colab={"base_uri": "https://localhost:8080/"} outputId="43fdb8bc-c096-4083-a3ee-b9ba0dc076b5"

# We can flatten a matrix to a vector (concatenating its rows, one by one) using ravel

A = jnp.reshape(jnp.arange(6),(2,3))
print(A.ravel()) ##  [0 1 2 3 4 5]


# + [markdown] id="1Gjmm2tNwKxy"
# In numpy,  arrays are layed out in memory
# such that, if we iterate over neighboring elements,
# the rightmost index changes the fastest.
# This is called row-major order,
# and is used by other languages such as C++, Eigen and PyTorch.
# By contrast, other languages (such as Julia, Matlab, R and Fortran)
# use column-major order.
# See below for an illustration of the difference.
#
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/4/4d/Row_and_column_major_order.svg" width="200">
#
#
# (Source: https://commons.wikimedia.org/wiki/File:Row_and_column_major_order.svg)
#
# Thus in numpy, for speed reasons, we should always write loops like this:
# ```
# A = jnp.reshape(jnp.arange(6),(2,3))
# d1, d2 = jnp.shape(A)
# for i in range(d1):
#   for j in range(d2):
#     # Do something with A[i,j]
#  ```
#
# For similar reasons, data matrices are usually stored
# in the form $(N,D)$, where $N$ is the batchsize (number of examples),
# so that we can efficiently extract minibatches by slicing blocks of consecutive memory.

# + id="npnWnwQNwKxz" colab={"base_uri": "https://localhost:8080/"} outputId="6391105a-f230-48a2-bc04-73792064f9e2"
## We can create a tensor in numpy as in this example:

#T = jnp.ndarray([2,3,4]) # fill with random values # does not work with jax
T = jnp.zeros((2,3,4))
T = jnp.reshape(jnp.arange(24),(2,3,4)) # fill with 0..23
print(jnp.shape(T))
print(T)



# + id="EKNM6Tp4wKx2" colab={"base_uri": "https://localhost:8080/"} outputId="7f54b135-856f-4e82-b620-d94873cbe2b4"
#We can permute the order of the dimensions using jnp.transpose.

x = jnp.ones((1, 2, 3))
print(jnp.transpose(x, (1, 0, 2)).shape) ## (2, 1, 3)

#Note that this does not actually move the data in memory
#(which would be slow),
#it merely provides a different \keywordDef{view} of the same data,
#i.e., it changes the mapping from $n$-dimensional vectors of
#subscripts to 1d integers.

# + id="nsRCZoT-wKx5" colab={"base_uri": "https://localhost:8080/"} outputId="bd2bef8d-0b1b-4c85-c115-fb7f84f5bf6f"
# matrix multiplication 

A = np.random.rand(2,3);
B = np.random.rand(3,4);
C = jnp.dot(A,B)
assert jnp.shape(C) == (2,4)
print(C)
C2 = A.dot(B)
C3 = A @ B
assert jnp.allclose(C, C2)
assert jnp.allclose(C, C3)

#Note that we need to use jnp.dot(A,B)
#if we use A * B, Python tries to compute the elementwise product,
#which is invalid, since $A$ and $B$ have incompatible shapes.


# + id="6GBtWrfqwKx7" colab={"base_uri": "https://localhost:8080/"} outputId="f92355a8-eb4a-4a18-9507-03147b91d72c"
# Outer products

x = jnp.arange(1,3); y = jnp.arange(1,3); 
A = jnp.outer(x,y);
print(A)


# + id="Di2E6jQtwKx9" colab={"base_uri": "https://localhost:8080/"} outputId="86118a39-d98e-4706-c5d4-d660ea6b1573"
# We can sum across the rows

X = jnp.reshape(jnp.arange(6), (2,3))
XS = jnp.dot(jnp.ones((1,2)), X)
print(XS)
XS2 = jnp.sum(X, axis=0)
assert jnp.allclose(XS, XS2)


# + id="6R2b8hI1wKx_" colab={"base_uri": "https://localhost:8080/"} outputId="d6874669-11f9-4064-ccfc-3bd5f5996852"
# We can sum across the columns 

X = jnp.reshape(jnp.arange(6), (2,3))
XS = jnp.dot(X, jnp.ones((3,1)))
print(XS)
XS2 = jnp.sum(X, axis=1).reshape(-1, 1)
assert jnp.allclose(XS, XS2)


# + id="5Iz8FBE_wKyC"
# We can sum across all entries

X = jnp.reshape(jnp.arange(6), (2,3))
S1 = jnp.dot(jnp.ones((1,2)), jnp.dot(X, jnp.ones((3,1))))[0]
S2 = jnp.sum(X)
assert jnp.allclose(S1, S2)


# + id="lr74QJDxwKyE" colab={"base_uri": "https://localhost:8080/"} outputId="462d6bb1-ff54-43a4-b22c-0d76bb8fa520"
# Kronecker product

jnp.kron(jnp.eye(2), jnp.ones((2,2)))


# + id="WlzujAkMwKyH" colab={"base_uri": "https://localhost:8080/"} outputId="c3f494bc-15c8-405a-a66d-edf54edd7a1e"
# Vector Norms
x = jnp.arange(6)
print(jnp.linalg.norm(x, 2) ** 2)
print(jnp.sum(jnp.power(x, 2)))
print(jnp.linalg.norm(x, jnp.inf))

# Matrix norms
A = np.random.randn(4,4)
print(np.linalg.norm(A, ord=2)) # not supported by jax
print(np.linalg.norm(A, ord='nuc')) # not supported by jax
print(jnp.linalg.norm(A, ord='fro'))



# + id="Epv5_4ZPwKyJ" colab={"base_uri": "https://localhost:8080/"} outputId="4f80cc08-9fe9-4e4a-efba-4f68cbc4a6b1"
# Size of a matrix

print(jnp.trace(A))
print(np.linalg.det(A)) # not supported by jax
print(np.linalg.cond(A)) # not supported by jax

# + [markdown] id="3ccnkeh4wKyL"
# ## Sparse matrices  <a class="anchor" id="sparse"></a>

# + id="oEGMPCvXwKyL" colab={"base_uri": "https://localhost:8080/"} outputId="37cd91b0-ede0-4044-b6b4-3a3824ebfa3c"
from scipy.sparse import diags
A = diags([1,2,3])
print(A)
print(A.toarray())


# + id="1MPMovF-wKyN" colab={"base_uri": "https://localhost:8080/"} outputId="b577341f-8874-4904-bd77-dae0cd5669a8"
# Block diagonal

from scipy.linalg import block_diag
block_diag([2, 3], [[4, 5], [6, 7]])



# + [markdown] id="vU_3xBrHwKyP"
# Band diagonal
#
# See (https://pypi.org/project/bandmat)

# + [markdown] id="_-8Q8ljmwKyQ"
# ## Broadcasting  <a class="anchor" id="broadcasting"></a>

# + [markdown] id="E8_wwsN_wKyQ"
# In numpy, the command A * B computes the elementwise multiplication of arrays or tensors A and B.
# If these arrays have different shapes,
# they will be automatically converted to have compatible shapes by
# implictly replicating  certain dimensions; this is called
# **broadcasting**. The following conversion rules are applied
# in order:
#
# * If the two arrays differ in their number of dimensions, the
#    shape of the one with fewer dimensions is padded with ones on the
#    left side. For example, a scalar will be converted to a vector,
#    and a vector to a matrix with one row.
# * If the shape of the two arrays does not match in any dimension,
#    the array with shape equal to 1 in that dimension is stretched to
#    match the other shape, by replicating the corresponding contents.
# * If in any dimension the sizes disagree and neither is equal to
#    1, an error is raised.
#
#
#
# <img src="https://github.com/probml/pyprobml/blob/master/book1/linalg/figures/broadcasting.png?raw=True" width="400">
#
#
#
# Figure made by [broadcasting_fig.py](https://github.com/probml/pyprobml/blob/master/scripts/broadcasting_fig.py) by Jake VanderPlas.
#

# + id="aiKsKgofwKyR" colab={"base_uri": "https://localhost:8080/"} outputId="69b39d87-bbbf-4f6c-f21f-107f2fc3a02c"
# Example: scaling each column
X = jnp.reshape(jnp.arange(6), (2,3))
s = jnp.array([1,2,3])
XS = X * s 
print(XS)
XS2 = jnp.dot(X, jnp.diag(s)) # post-multiply by diagonal
assert jnp.allclose(XS, XS2)


# + id="sCAJAFcCwKyT" colab={"base_uri": "https://localhost:8080/"} outputId="ff43cb7f-13e8-4d55-d28a-34869672ea69"
# Example: scaling each row
X = jnp.reshape(jnp.arange(6), (2,3))
s  = jnp.array([1,2])
XS = X *  jnp.reshape(s, (-1,1)) 
print(XS)
XS2 = jnp.dot(jnp.diag(s), X) # pre-multiply by diagonal
assert jnp.allclose(XS, XS2)

# + [markdown] id="VzY3i3rawKyV"
# ## Einstein summation  <a class="anchor" id="broadcasting"></a>

# + [markdown] id="lgZTVXRdwKyW"
# Einstein summation lets us write formula such as  inputs -> outputs, which name the dimensions 
# of the input tensor and output tensors; dimensions which are not named in the output are summed over - this is called **tensor contraction**.
#

# + id="SDEulwVMwKyW"
# Sample data
a = jnp.arange(3)
b = jnp.arange(3)
A = jnp.arange(6).reshape(2,3)
B = jnp.arange(15).reshape(3,5)
S = jnp.arange(9).reshape(3,3)
T = np.random.randn(2,2,2,2)

# + [markdown] id="4Ey4afi4wKyY"
# Now consider einsum with  a single tensor.

# + id="iuZxLeoQwKyZ"


# Matrix transpose
assert jnp.allclose(A.T, jnp.einsum('ij->ji', A))

# Sum all elements
assert jnp.allclose(jnp.sum(A), jnp.einsum('ij->', A))

# Sum across rows
assert jnp.allclose(jnp.sum(A, axis=0), jnp.einsum('ij->j', A))

# Sum across columns
assert jnp.allclose(jnp.sum(A, axis=1), jnp.einsum('ij->i', A))

# Sum specific axis of tensor
assert jnp.allclose(jnp.sum(T, axis=1), jnp.einsum('ijkl->ikl', T))
assert jnp.allclose(jnp.sum(jnp.sum(T, axis=0), axis=0), jnp.einsum('ijkl->kl', T))

# repeated indices with one arg extracts diagonals
assert jnp.allclose(jnp.diag(S), jnp.einsum('ii->i', S))
          
# Trace
assert jnp.allclose(jnp.trace(S), jnp.einsum('ii->', S))


# + [markdown] id="OfScBTbPwKyb"
# Now consider einsum with 2 tensors.

# + id="i0_4DJJnwKyb"


# Matrix vector multiplication
assert jnp.allclose(jnp.dot(A, b), jnp.einsum('ik,k->i', A, b))

# Matrix matrix multiplication
assert jnp.allclose(jnp.dot(A, B), jnp.einsum('ik,kj->ij', A, B))
assert jnp.allclose(jnp.matmul(A, B), jnp.einsum('ik,kj->ij', A, B))

# Inner product 
assert jnp.allclose(jnp.dot(a, b), jnp.einsum('i,i->', a, b))
assert jnp.allclose(jnp.inner(a, b), jnp.einsum('i,i->', a, b))

# Outer product
assert jnp.allclose(jnp.outer(a, b), jnp.einsum('i,j->ij', a, b))

# Elementwise product
assert jnp.allclose(a * a, jnp.einsum('i,i->i', a, a))
assert jnp.allclose(A * A, jnp.einsum('ij,ij->ij', A, A))
assert jnp.allclose(jnp.multiply(A, A), jnp.einsum('ij,ij->ij', A, A))


# + [markdown] id="t1W41QVfwKyd"
#  As a more complex example,
#  suppose we have a 3d tensor $S_{ntk}$ where $n$ indexes examples in the
#  batch, $t$ indexes locations in the sequence, and $k$ indexes words
#  in a one-hot representation.
#  Let $W_{kd}$ be an embedding matrix that maps sparse one-hot vectors
#  $R^k$  to dense vectors in $R^d$.
#  We can convert the batch of sequences of one-hots
#  to a batch of sequences of embeddings as follows:
# $$
# E_{ntd} = \sum_k S_{ntk} W_{kd}
# $$
# We can compute the sum of the embedding vectors for
# each sequence (to get a global representation
# of each bag of words) as follows:
# $$
# E_{nd} = \sum_k \sum_t S_{ntk} W_{kd}
# $$
# Finally we can pass each sequence's vector representation
# through another linear transform $V_{dc}$ to map to the logits over a
# classifier
# with $c$ labels:
# $$
# L_{nc} = \sum_d E_{nd} V_{dc}
# = \sum_d \sum_k \sum_t S_{ntk} W_{kd} V_{dc}
# $$
# In einsum notation, we have
# $$
# L_{nc} = S_{ntk} W_{kd} V_{dc}
# $$
# We sum  over $k$ and $d$  because those
# indices occur twice on the RHS.
# We sum over $t$  because that index does not occur
# on the LHS.

# + id="o-E5ICHawKye"
# sentence embedding example in code

N = 2; C = 3; D = 4; K = 5; T = 6;
S = np.random.randn(N, T, K)
W = np.random.randn(K, D)
V = np.random.randn(D, C)
Lfast = jnp.einsum('ntk,kd,dc->nc', S, W, V)
# Compare to brute force way of computing L below.
# We can only do elementwise assignment to L in original numpy, not jax
L = np.zeros((N,C))
for n in range(N):
    for c in range(C):
        s = 0
        for d in range(D):
            for k in range(K):
                for t in range(T):
                    s += S[n,t,k] * W[k,d] * V[d,c]
        L[n,c] = s # does not work in jax
assert jnp.allclose(L, Lfast)

# + id="iNTpGq-CwKyg"
# Optimization

path = jnp.einsum_path('ntk,kd,dc->nc', S, W, V, optimize='optimal')[0]
assert jnp.allclose(L, jnp.einsum('ntk,kd,dc->nc', S, W, V, optimize=path))


# + [markdown] id="SO9iXr1cwKyi"
# ## Eigenvalue decomposition (EVD)<a class="anchor" id="EVD"></a>

# + id="9BMsh3NewKyi" colab={"base_uri": "https://localhost:8080/"} outputId="2a274813-d6c7-47f7-d129-8f160586461d"
np.random.seed(42)
M = np.random.randn(4, 4)
A = M  + M.T # ensure symmetric
assert (A == A.T).all() # check symmetric
evals, evecs = jnp.linalg.eigh(A) # tell JAX matrix is symmetric
#evals, evecs = np.linalg.eig(A)
print(evals)
print(evecs)

# Sort columns so one with largest evals (absolute value) are first
idx = jnp.argsort(jnp.abs(evals))[::-1] # largest first
evecs = evecs[:, idx] # sort columns

evals = evals[idx]
print(evals)
print(evecs)

# + [markdown] id="2Mct64wpwKyk"
# ### Example: Diagonalizing a rotation matrix <a class="anchor" id="EVD-rotation"></a>
#
# As an example, let us construct $A$
# by combining
# a rotation by 45 degrees about the $z$ axis,
# a scaling by $\diag(1,2,3)$, followed by another rotation of -45 degrees.
# These components can be recovered from $A$ using EVD, as we show below.
#

# + id="lnk1g3WFwKyl" colab={"base_uri": "https://localhost:8080/"} outputId="2e36636d-d81b-4ea6-ae79-9cf248ed2ef4"
a = (45/180) * jnp.pi
R = jnp.array(
        [[jnp.cos(a), -jnp.sin(a), 0],
          [jnp.sin(a), jnp.cos(a), 0],
          [0, 0, 1]])
print(R)

S = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
A = jnp.dot(jnp.dot(R, S), R.T) # Rotate, scale, then unrotate


evals, evecs = jnp.linalg.eig(A)
idx = jnp.argsort(jnp.abs(evals)) # smallest first
U = evecs[:, idx] # sort columns

evals2 = evals[idx]
D = jnp.diag(evals2)

assert jnp.allclose(A, jnp.dot(U, jnp.dot(D, U.T))) # eigen decomposition
assert jnp.allclose(jnp.abs(R), jnp.abs(U)) # Recover rotation
assert jnp.allclose(D, S)  # Recover scale



# + [markdown] id="NdZJgaMNwKyn"
# ### Example: checking for positive definitness
#
# A symmetric matrix is positive definite iff all its eigenvalues are positive.
#

# + id="-90h9LOUwKyo" colab={"base_uri": "https://localhost:8080/"} outputId="18014945-cdea-4f36-d6d7-457156d793fa"
np.random.seed(42)
M = np.random.randn(3, 4)
A = jnp.dot(M, M.T) # ensure A is positive definite

def is_symmetric(A):
  return (A == A.T).all()

def isposdef(A):
  if not is_symmetric(A):
    return False
  evals, evecs = jnp.linalg.eigh(A)
  return jnp.all(evals > 0)

print(isposdef(A))

# + [markdown] id="mVq9ktIKwKyq"
# ### Power method

# + id="NkpWGvB6wKyq"
from numpy.linalg import norm

np.random.seed(0)

def power_method(A, max_iter=100, tol=1e-5):
    n = jnp.shape(A)[0]
    u = np.random.rand(n)
    converged = False
    iter = 0
    while (not converged) and (iter < max_iter):
        old_u = u
        u = jnp.dot(A, u)
        u = u / norm(u)
        lam = jnp.dot(u, jnp.dot(A, u))
        converged = (norm(u - old_u) < tol)
        iter += 1
    return lam, u

X = np.random.randn(10, 5)
A = jnp.dot(X.T, X) # psd matrix
lam, u = power_method(A)

evals, evecs = jnp.linalg.eigh(A)
idx = jnp.argsort(jnp.abs(evals))[::-1] # largest first
evals = evals[idx]
evecs = evecs[:,idx]

tol = 1e-3
assert jnp.allclose(evecs[:,0], u, tol)

# + [markdown] id="WzDG4o8_wKys"
# ## Singular value decomposition (SVD) <a class="anchor" id="SVD"></a>

# + id="sdCHdICywKyv" colab={"base_uri": "https://localhost:8080/"} outputId="35a02d1f-53b1-4da1-aa1b-0107906b8553"
np.random.seed(0)

A = np.random.randn(10, 5)

U, S, V = jnp.linalg.svd(A,full_matrices=False)
print("Full=False: shape of U {}, S {}, V {}".format(U.shape, S.shape, V.shape))

U, S, V = jnp.linalg.svd(A,full_matrices=True)
print("Full=True: shape of U {}, S {}, V {}".format(U.shape, S.shape, V.shape))


# + id="0Z8X_GD4wKyt" colab={"base_uri": "https://localhost:8080/"} outputId="9db97424-f175-40f8-e669-b412e52ff9d5"
np.random.seed(0)

def make_random_low_rank(D, K):
  A = np.zeros((D, D), dtype=jnp.float32) # we use np so we can mutate A in place
  for i in range(K):
    x = np.random.randn(D)
    A = A + jnp.outer(x, x)
  return A


A = make_random_low_rank(10, 3)
U, S, V = jnp.linalg.svd(A,full_matrices=False) 
print(jnp.sum(S > 1e-5))
print(np.linalg.matrix_rank(A))


# + [markdown] id="WNtYeurTwKy0"
# ## Low rank approximation to an image <a class="anchor" id="SVD-image"></a>

# + id="uPFmJliQwKy1" colab={"base_uri": "https://localhost:8080/", "height": 820} outputId="3c9d9330-915a-4b01-a284-4aa9c3804db7"
import matplotlib.image

def rgb2gray(rgb):
  #Y' = 0.2989 R + 0.5870 G + 0.1140 B 
  return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

#url = 'https://github.com/probml/pyprobml/blob/master/data/clown.png'
#img = matplotlib.image.imread(url) # invalid png file, apparently...
#X = rgb2gray(img)    

import scipy
racoon = scipy.misc.face().astype(np.float)
X = rgb2gray(racoon)
plt.gray()
plt.imshow(X)
plt.show()

r = np.linalg.matrix_rank(X)
print(r)

U, sigma, V = jnp.linalg.svd(X, full_matrices=True)
ranks = [1, 2, 5, 10, 20, r]
R = len(ranks)

fig, axes = plt.subplots(2, 3, figsize=[15, 10])
axes = axes.reshape(-1)
for i in range(R):
    k = ranks[i] 
    x_hat = jnp.dot(jnp.dot(U[:, :k], jnp.diag(sigma[:k])), V[:k, :]) 
    ax = axes[i]
    #plt.subplot(2, 3, i+1)
    ax.imshow(x_hat, cmap='gray')
    ax.set_title("rank {}".format(k))
    #save_fig("svdImageDemoClown{}.pdf".format(k))
plt.show()



# + id="yd2kIqHcwKy3" colab={"base_uri": "https://localhost:8080/", "height": 285} outputId="c8dd8d04-76f1-4349-cd27-e5993dbe46d3"
# Plot singular value spectrum
k = 100
plt.figure()
plt.plot(jnp.log(sigma[:k]), 'r-', linewidth=4, label="Original")
plt.ylabel(r"$log(\sigma_i)$")
plt.xlabel("i")


# Compare this to a random shuffled version of the image
x2 = np.random.permutation(X)
# so we convert to a 1d vector, permute, and convert back
x1d = X.ravel()
np.random.shuffle(x1d) # ijnplace
x2 = x1d.reshape(X.shape)
U, sigma2, V = jnp.linalg.svd(x2, full_matrices = False)
plt.plot(jnp.log(sigma2[:k]), 'g:', linewidth=4, label="Randomized")
plt.legend()
#save_fig("svdImageDemoClownSigmaScrambled.pdf")
plt.show()

# + [markdown] id="JUfJbRZcwKy4"
# ## Other matrix decompositions <a class="anchor" id="decomp"></a>
#
# In this section, we illustrate a few other matrix decompositions.

# + [markdown] id="_6OoeWB4wKy5"
# ### LU decomposition <a class="anchor" id="decomp-LU"></a>

# + id="Buo-Erf5wKy5" colab={"base_uri": "https://localhost:8080/"} outputId="0adeb08f-e156-42ec-d729-d38029f371fc"

np.random.seed(42)
A = np.random.randn(5,5)
L, U = scipy.linalg.lu(A, True)
print(L)
print(U)

# + [markdown] id="z3N8dgekwKy9"
# ### QR decomposition <a class="anchor" id="decomp-QR"></a>
#
#

# + id="pHUvH-BDwKy-" colab={"base_uri": "https://localhost:8080/"} outputId="40823ec4-fbf1-4012-c80c-8f31d98375e7"
# Economy vs full mode

np.random.seed(42)
A = np.random.randn(5,3)
Q, R = scipy.linalg.qr(A, mode='economic')
print("economy: Q shape {}, R shape {}".format(Q.shape, R.shape))
print(Q)
print(R)


Q, R = scipy.linalg.qr(A, mode='full')
print("full: Q shape {}, R shape {}".format(Q.shape, R.shape))
print(Q)
print(R)
assert jnp.allclose(jnp.eye(5), jnp.dot(Q, Q.T), atol=1e-3)

# + [markdown] id="Ev-GV30xwKzA"
# ### Cholesky decomposition <a class="anchor" id="decomp-chol"></a>

# + id="l3Gv23uDwKzA" colab={"base_uri": "https://localhost:8080/"} outputId="0d06b168-b121-4058-c572-8c1cfcd9e82b"
# Sample from multivariate Gaussian

from scipy.stats import multivariate_normal as mvn

def sample_mvn(mu, Sigma, N):
    L = jnp.linalg.cholesky(Sigma)
    D = len(mu)
    Z = np.random.randn(N, D)
    X = jnp.dot(Z, L.T) + jnp.reshape(mu, (-1,D))
    return X

D = 5
np.random.seed(42)
mu = np.random.randn(D)
A = np.random.randn(D,D)
Sigma = jnp.dot(A, A.T)
N = 10000
X = sample_mvn(mu, Sigma, N)
mu_hat = jnp.mean(X)
C = np.cov(X, rowvar=False) # not yet implemented by jax
print(C)
print(Sigma)
assert jnp.allclose(C, Sigma, 1e-0) # not that close, even after 10k samples...

dist = mvn(mu, Sigma)
X = dist.rvs(size=N)
C = np.cov(X, rowvar=False)
assert jnp.allclose(C, Sigma, 1e-0)
    

# + [markdown] id="Iu5cuTfKwKzD"
# ## Matrix calculus <a class="anchor" id="calculus"></a>

# + [markdown] id="wITRY5j1wKzD"
# ### Automatic differentiation in Jax  <a class="anchor" id="AD-jax"></a>
#
# In this section, we show how to use Jax to compute gradients, Jacobians and Hessians
# of some simple convex functions.

# + id="sTPT_7ltS6nI"
from jax import grad, hessian, jacfwd, jacrev, vmap, jit

# + [markdown] id="vtLg-LWc9ubs"
# Linear function: multi-input, scalar output.
#
# $$
# \begin{align}
# f(x; a) &= a^T x\\
# \nabla_x f(x;a) &= a
# \end{align}
# $$
#

# + id="AzYGzC24wKzF"
# We construct a single output linear function.
# In this case, the Jacobian and gradient are the same.
Din = 3; Dout = 1;
np.random.seed(42)
a = np.random.randn(Dout, Din)
def fun1d(x):
    return jnp.dot(a, x)[0]
x = np.random.randn(Din)
g = grad(fun1d)(x)
assert jnp.allclose(g, a)
J = jacrev(fun1d)(x)
assert jnp.allclose(J, g)


# + [markdown] id="sQsnY_oD-EmG"
# Linear function: multi-input, multi-output.
#
# $$
# \begin{align}
# f(x;A) &= A x \\
# \nabla_x f(x;A) &= A
# \end{align}
# $$

# + id="Uxq54zWCwKzG"
# We construct a multi-output linear function.
# We check forward and reverse mode give same Jacobians.
Din = 3; Dout = 4;
A = np.random.randn(Dout, Din)
def fun(x):
    return jnp.dot(A, x)
x = np.random.randn(Din)
Jf = jacfwd(fun)(x)
Jr = jacrev(fun)(x)
assert jnp.allclose(Jf, Jr)
assert jnp.allclose(Jf, A)

# + [markdown] id="gw4tFSqb-YU-"
# Quadratic form.
#
# $$
# \begin{align}
# f(x;A) &= x^T A x \\
# \nabla_x f(x;A) &= (A+A^T) x \\
# \nabla^2 x^2 f(x;A) &= A + A^T
# \end{align}
# $$

# + id="9b5fug19BX1W"

D = 4
A = np.random.randn(D, D)
x = np.random.randn(D)
quadfun = lambda x: jnp.dot(x, jnp.dot(A, x))

J = jacfwd(quadfun)(x)
assert jnp.allclose(J, jnp.dot(A+A.T, x))

H1 = hessian(quadfun)(x)
assert jnp.allclose(H1, A+A.T)

def my_hessian(fun):
  return jacfwd(jacrev(fun))
H2 = my_hessian(quadfun)(x)
assert jnp.allclose(H1, H2)

# + [markdown] id="_bRM7ATKwKzU"
# ## Solving linear systems of equations <a class="anchor" id="linear"></a>

# + [markdown] id="5xaLp82twKzU"
# ### Square systems with unique solution <a class="anchor" id="linear-systems-square"></a>

# + id="-KucygV_wKzV" colab={"base_uri": "https://localhost:8080/"} outputId="628cfd85-76f5-4424-d7e6-e25c383b9df6"
A = jnp.array([[3,2,-1], [2, -2, 4], [-1, 0.5, -1]])
b = jnp.array([1, -2, 0])
x  = jax.scipy.linalg.solve(A,b)
print(x)
print(jnp.dot(A, x) - b)

# + id="YpaEHzfmwKzX" colab={"base_uri": "https://localhost:8080/"} outputId="83aa04b3-97f6-4474-9b36-d0071f2affe8"
# Now use LU decomposition and backsubstitution.

L, U = jax.scipy.linalg.lu(A, permute_l=True)
print(L)
print(U)
y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
x = jax.scipy.linalg.solve_triangular(U, y, lower=False)
print(x)

# + [markdown] id="jrqvwopEwKzY"
# ### Underconstrained systems: least norm solution <a class="anchor" id="linear-systems-under"></a>
#
# We will compute the minimum norm solution.

# + id="P3t1BsKzwKzZ" colab={"base_uri": "https://localhost:8080/"} outputId="203e5bfa-a647-4dd2-9760-47a234129452"
np.random.seed(42)
m = 3
n = 4
A = np.random.randn(m, n)
x = np.random.randn(n)
b = jnp.dot(A, x)

#x_least_norm = jax.scipy.linalg.lstsq(A, b)[0]
x_least_norm = scipy.linalg.lstsq(A, b)[0]
print(x_least_norm)
print(jnp.dot(A, x_least_norm) - b)
print(jnp.linalg.norm(x_least_norm, 2))


# + [markdown] id="2Sx56_wvwKza"
# If you look at the [source code for scipy.linalg.lstsq](https://github.com/scipy/scipy/blob/v0.19.0/scipy/linalg/basic.py#L892-L1058),
# you will see that it just a Python wrapper
# to some LAPACK code written in Fortran. LAPACK offers multiple methods for solving
# linear systems, including `gelsd` (default), `gelss` , and `gelsy`. The first two methods use SVD, the latter uses QR decomposition. 
#
# A lot of numpy and scipy functions are just wrappers to legacy libraries,
#   written in Fortran or C++, since Python itself is too slow for
#   efficient numerical computing.
#   Confusingly, sometimes numpy and scipy offer different wrappers to the same
#   underlying LAPACK functions, but with different interfaces.
#   For example, as of 2018, `jnp.linalg.lstsq` and
#   `scipy.linalg.lstsq` have been modified to behave the same.
#   However,
#   `jnp.linalg.qr` and `scipy.linalg.qr`
#   have slightly different optional arguments and may give different
#   results.
#   
# Jax does not yet implement lstsq, but does implement some of the underlying methods 
#  [here](https://github.com/google/jax/blob/master/jax/scipy/linalg.py).
#  Unlike the legacy code, this can run fast on GPUs and TPUs.

# + [markdown] id="zweT26QfwKzc"
# ### Overconstrained systems: least squares solution <a class="anchor" id="linear-systems-over"></a>

# + id="KGsrWmfbwKzc" colab={"base_uri": "https://localhost:8080/"} outputId="79a12650-b3e3-4273-e390-1d01b333f854"
def naive_solve(A, b):
    return jax.numpy.linalg.inv(A.T @ A) @ A.T @ b
    #return jjnp.linalg.inv(A.T @ A) @ A.T @ b

def qr_solve(A, b):
    Q, R = jnp.linalg.qr(A) 
    Qb = jnp.dot(Q.T,b) 
    return jax.scipy.linalg.solve_triangular(R, Qb)

def lstsq_solve(A, b):
    return scipy.linalg.lstsq(A, b, rcond=None)[0]

def pinv_solve(A, b):
    return jnp.dot(jnp.linalg.pinv(A), b)
    
np.random.seed(42)
m = 4
n = 3
A = np.random.randn(m, n)
x = np.random.randn(n)
b = jnp.dot(A, x)


methods = list()
solns = list()
    
methods.append('naive')
solns.append(naive_solve(A, b))

#methods.append('pinv')
#solns.append(pinv_solve(A,b)) # pinv not yet implemented by jax

#methods.append('lstsq')
#solns.append(lstsq_solve(A, b)) # lstsq not yet implemented by jax

methods.append('qr')
solns.append(qr_solve(A, b))


for (method, soln) in zip(methods, solns):
    residual = b -  jnp.dot(A, soln)
    print('method {}, norm {:0.5f}, residual {:0.5f}'.format(method, jnp.linalg.norm(soln), jnp.linalg.norm(residual)))
    print(soln.T)
    print('\n')

# + id="lYBi9w70wKzd"

