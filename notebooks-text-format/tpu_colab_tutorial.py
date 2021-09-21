# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="IKu4IVYuxRaj"
# # TPUs in Colab
# **Authors**
#
# * Gerardo Durán-Martín
# * Mahmoud Soliman
# * Kevin Murphy

# + [markdown] id="rSC4o4M6Wxc5"
# Before start this tutorial, make sure to configure your session correctly.
#
# ### 1. First we authenticate GCP to our current session

# + id="gbQ4xcYrOVzl"
from google.colab import auth
auth.authenticate_user()

# + [markdown] id="IuKMgyQSXICm"
# ### 2. Next, we install GCloud SDK

# + id="LOnrbTH7XO4O" outputId="d6296602-4351-4147-f36b-883feafd3aed" colab={"base_uri": "https://localhost:8080/"}
# !curl -S https://sdk.cloud.google.com | bash

# + [markdown] id="9VlUkDu9XVVt"
# ### 3. Finally, we initialise all the variables we will be using throughout this tutorial.
#
# We will create a `.sh` file that must be called at every cell that begins with `%%bash` as follows:
#
# ```bash
# # # %%bash
# source /content/commands.sh
# # ... rest of the commands
# ```

# + colab={"base_uri": "https://localhost:8080/"} id="eG-7t5ZTXfkO" outputId="8e6779fd-d0e8-4432-f396-d6a04b8c72c7"
# %%writefile commands.sh
gcloud="/root/google-cloud-sdk/bin/gcloud"
gtpu="gcloud alpha compute tpus tpu-vm"
instance_name="murphyk-v3-8" # "probml-01-gerdm" # Modify for your instance name 
tpu_zone="us-central1-a" #"us-east1-d"
jax_install="pip install 'jax[tpu]>=0.2.16' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

# + [markdown] id="K2upn6cBWozc"
# # gcloud
#
# This first section introduces the gloud command line. We can work in the cloud in one of two ways:
#
# 1. Using the command line (this tutorial)
# 2. Using the google cloud console ([console.cloud.google.com](https://console.cloud.google.com/))

# + [markdown] id="xEDGVNT9zdef"
# ## Setup
#
# Our first step is to install `gcloud alpha`.
#
# - Installing `gcloud alpha`
#
#     We begin by installing the `gcloud alpha` command line. This will allow us to work with TPUs at Google cloud. Run the following command

# + colab={"base_uri": "https://localhost:8080/"} id="FrXfY-WhRONk" outputId="64ff2377-e91d-414a-f822-51fdc4206abe" language="bash"
# source /content/commands.sh
#
# $gcloud components install alpha

# + [markdown] id="Zk33hqJtYeRz"
# Next, we set the project to `probml` 

# + id="nAhn8jntYmnd" outputId="4bac0c9c-54fb-40fb-b222-50b4166169c5" colab={"base_uri": "https://localhost:8080/"} language="bash"
# source /content/commands.sh
#
# $gcloud config set project probml

# + [markdown] id="dlq-_S2KYoKr"
# - Verify installation
#
# Finally, we verify that you've successfully installed `gcloud alpha` by running the following command. Make sure to have version `alpha 2021.06.25` or later.

# + colab={"base_uri": "https://localhost:8080/"} id="wGpoCBwTY9k5" outputId="69799c70-149a-4ad5-ef69-1b30a8001b0c" language="bash"
# source /content/commands.sh
#
# $gcloud -v 

# + [markdown] id="KZnbePahZHfw"
# # TPUS
#
# ## The basics

# + [markdown] id="F4vE-BUTaFa7"
# ### Creating an instance
#
# Each GSoC member obtains 8 v3-32 cores (or a Slice) when following the instructions outlined below.
#
# To create our first TPU instance, we run the following command. Note that `instance_name` should be unique (it was defined at the top of this tutorial)

# + colab={"base_uri": "https://localhost:8080/"} id="z_r9Eyti0toh" outputId="69c80177-50cc-4364-fccd-acdc4695b2dd" language="bash"
# source /content/commands.sh
# $gtpu create $instance_name \
#     --accelerator-type v3-8 \
#     --version v2-alpha \
#     --zone $tpu_zone

# + [markdown] id="-X3e-AwvZgDV"
# You can verify whether your instance has been created by running the following cell

# + colab={"base_uri": "https://localhost:8080/"} id="dg49Jo99OzOM" outputId="34f25786-97b0-4d5c-90f4-1f7a0d31c0cd" language="bash"
# source /content/commands.sh
# $gcloud alpha compute tpus list --zone $tpu_zone

# + [markdown] id="RYn8VUkOZr33"
# ### Deleting an instance
#
# To avoid extra costs, it is important to delete the instance after use (training, testing experimenting, etc.).
#
# To delete an instance, we create and run a cell with the following content
#
# ```bash
# # # %%bash
# source /content/commands.sh
#
# $gtpu delete --quiet $instance_name --zone=$tpu_zone
# ```
#
# **Make sure to delete your instance once you finish!!**

# + [markdown] id="Q_HAWg-EaZNI"
# # Jax

# + [markdown] id="5RBZsCE6ajyJ"
# ### Installing Jax
#
# When connecting to an instance directly via ssh, it is important to note that running any Jax command will wait for the other hosts to be active. To void this, we have to run the desired code simultaneously on all the hosts.
#
# > To run JAX code on a TPU Pod slice, you must run the code **on each host in the TPU Pod slice.**
#
# In the next cell, we install Jax on each host of our slice.

# + id="bvEUfHCQUBSG" language="bash"
# source /content/commands.sh
# $gtpu ssh $instance_name \
#     --zone $tpu_zone \
#     --command "$jax_install" \
#     --worker all # or machine instance 1..3

# + [markdown] id="VVXbpgxEarZd"
# ### Example 1: Hello, TPUs!
#
# In this example, we create a `hello_tpu.sh` that asserts whether we can connect to all of the hosts. First, we create the `.sh` file that will be run **in each of the workers**.

# + colab={"base_uri": "https://localhost:8080/"} id="h0Rm7QhHUGqx" outputId="aee29b2f-2707-4134-925b-7984909ca62a"
# %%writefile hello_tpu.sh
# #!/bin/bash
# file: hello_tpu.sh

export gist_url="https://gist.github.com/1e8d226e7a744d22d010ca4980456c3a.git"
git clone $gist_url hello_gsoc
python3 hello_gsoc/hello_tpu.py

# + [markdown] id="NWxXh1Gga_qJ"
# The content of `$gist_url` is the following
#
# You do not need to store the following file. Our script `hello_tpu.sh` will download the file to each of the hosts and run it.
#
# ```python
# # Taken from https://cloud.google.com/tpu/docs/jax-pods
# # To be used by the Pyprobml GSoC 2021 team
# # The following code snippet will be run on all TPU hosts
# import jax
#
# # The total number of TPU cores in the pod
# device_count = jax.device_count()
# # The number of TPU cores attached to this host
# local_device_count = jax.local_device_count()
#
# # The psum is performed over all mapped devices across the pod
# xs = jax.numpy.ones(jax.local_device_count())
# r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
#
# # Print from a single host to avoid duplicated output
# if jax.process_index() == 0:
#     print('global device count:', jax.device_count())
#     print('local device count:', jax.local_device_count())
#     print('pmap result:', r)%
# ```
#
# Next, we run the code across all workers

# + colab={"base_uri": "https://localhost:8080/"} id="C0Svilz_U3PE" outputId="9fb1668e-f9b6-4883-a9b5-a8e3fbe8ca6a" language="bash"
# source /content/commands.sh
# $gtpu ssh $instance_name \
#     --zone $tpu_zone \
#     --command "$(<./hello_tpu.sh)" \
#     --worker all

# + [markdown] id="FDKJQwnLbQzD"
# ### Example 2: 🚧K-nearest neighbours🚧
#
# In this example we train the MNIST dataset using the KNN algorithm `pmap`. Our program clones a Github gist into each of the hosts. We use the multi-device availability of our slice to delegate a part of the training to each of the workers.
#
# First, we create the script that will be run on each of the workers

# + colab={"base_uri": "https://localhost:8080/"} id="gD0_Vbr_VFWg" outputId="3a948227-b745-4d8a-c782-da0756770996"
# %%writefile knn_tpu.sh
# #!/bin/bash
# file: knn_tpu.sh

export gist_url="https://gist.github.com/716a7bfd4c5c0c0e1949072f7b2e03a6.git"
pip3 install -q tensorflow_datasets
git clone $gist_url demo
python3 demo/knn_tpu.py

# + [markdown] id="HbuCmPMLbgzv"
# Next, we run the script

# + colab={"base_uri": "https://localhost:8080/"} id="4xBAeKlpVedt" outputId="6ff3a3fb-f799-4404-b0f3-7899dec65fc1" language="bash"
# source /content/commands.sh
#
# $gtpu ssh $instance_name \
#     --zone $tpu_zone \
#     --command "$(<./knn_tpu.sh)" \
#     --worker all

# + [markdown] id="4mZQ2T2ocEOV"
# # 🔪TPUs - The Sharp Bits🔪

# + [markdown] id="CUTrUrLHbvNX"
#
# ## Service accounts
#
# Before creating a new TPU instance, make sure that the Admin of the project grants the correct IAM user/group roles for your service account
#
# - `TPU Admin`
# - `Service Account User`
#
# This prevents you from running into the following error
#
# ![error](https://imgur.com/sMAV2A5.png)
#
# ## Running Jax on a Pod
#
# When creating an instance, we obtain different *slices*. Running a parallel operation on a single slice will not perform any computation until all of the slices have been run in sync. In Jax, this is done using `jax.pmap` function
#
# ## `pmap`ing a function
#
# > *The mapped axis size must be less than or equal to the number of local XLA devices available, as returned by jax.local_device_count() (unless devices is specified, [...])*
#
# ## Misc
#
# - [Padding can tank your performance](https://github.com/google/jax/tree/main/cloud_tpu_colabs#padding)

# + [markdown] id="UE0c56Z2cLM4"
# # References
#
# - gcloud
#     - [gcloud CLI cheatsheet](https://cloud.google.com/sdk/docs/cheatsheet)
#     - [gcloud update components](https://cloud.google.com/sdk/gcloud/reference/components/update)
# - TPUs
#     - [Jax cloud TPU](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)
#     - [TPU VM User's guide](https://cloud.google.com/tpu/docs/users-guide-tpu-vm)
#     - [Jax TPUs on Slices](https://cloud.google.com/tpu/docs/jax-pods)
# - Jax
#     - [MNIST example with Flax](https://github.com/google/flax/tree/master/examples/mnist)
#     - [Parallelism in Jax](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html)
#     - [Jax multi-hosts](https://jax.readthedocs.io/en/latest/multi_process.html)
#     - [ColCollective communication operations](https://colab.research.google.com/github/google/jax/blob/main/cloud_tpu_colabs/JAX_demo.ipynb#scrollTo=f-FBsWeo1AXE&uniqifier=1)
