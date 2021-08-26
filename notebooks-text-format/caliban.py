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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/intro/caliban.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="0-yTlCjR1xm3"
# # Running parallel jobs on Google Cloud using Caliban 
#
# [Caliban](https://github.com/google/caliban) is a package that makes it easy to run embarassingly parallel jobs on Google Cloud Platform (GCP) from your laptop.  (Caliban bundles your code into a Docker image, and then runs it on  [Cloud AI Platform](https://cloud.google.com/ai-platform), which is a VM on top of GCP.)
#

# + id="VpDcr6kBW4Za"
import json
import pandas as pd
import glob
from IPython.display import display
import numpy as np
import matplotlib as plt

# + [markdown] id="_puU0I-07XJ0"
# # Installation
#
# The details on how to install and run Caliban can be found [here](https://github.com/google/caliban). Below we give a very brief summary. Do these steps on your laptop, **outside of this colab**.

# + [markdown] id="22HUS_6b9NDz"
# - [install docker](https://github.com/google/caliban#docker) and test using ```docker run hello-world```
#
# - ```pip install caliban```
#
# - [setup GCP](https://caliban.readthedocs.io/en/latest/getting_started/cloud.html)
#

# + [markdown] id="9oupB5yUPOpA"
# # Launch jobs on GCP 
#
# Do these steps on your laptop, **outside of this colab**.
#

# + [markdown] id="ZN-__YAZPZXB"
#
# - create a requirements.txt file containing packages you need to be installed in GCP Docker image. Example:
#
# ```
# numpy
# scipy
# #sympy
# matplotlib
# #torch # 776MB  slow
# #torchvision
# tensorflow_datasets
# jupyter
# ipywidgets
# seaborn
# pandas
# keras
# sklearn
# #ipympl 
# jax
# flax
#  
# # below is jaxlib with GPU support
#  
# # CUDA 10.0
# #tensorflow-gpu==2.0
# #https://storage.googleapis.com/jax-releases/cuda100/jaxlib-0.1.47-cp36-none-linux_x86_64.whl
# #https://storage.googleapis.com/jax-releases/cuda100/jaxlib-0.1.47-cp37-none-linux_x86_64.whl
#  
# # CUDA 10.1
# #tensorflow-gpu==2.1
# #https://storage.googleapis.com/jax-releases/cuda101/jaxlib-0.1.47-cp37-none-linux_x86_64.whl
#  
# tensorflow==2.1  # 421MB slow
# https://storage.googleapis.com/jax-releases/cuda101/jaxlib-0.1.60+cuda101-cp37-none-manylinux2010_x86_64.whl
#  
# # jaxlib with CPU support
# #tensorflow
# #jaxlib
# ```
#
# - create script that you want to run in parallel, eg [caliban_test.py](https://github.com/probml/pyprobml/blob/master/scripts/caliban_test.py)
#
#
# - create config.json file with the list of flag combinations you want to pass to the script. For example the following file says to run 2 versions of the script, with flags ```--ndims 10 --prefix "***"``` and ```--ndims 100 --prefix "***"```. (The prefix flag is for pretty printing.)
# ```
# {"ndims": [10, 100],
# "prefix": "***" }
# ```
#
# - launch jobs on GCP, giving them a common name using the xgroup flag. 
# ```
# # # cp ~/github/pyprobml/scripts/caliban_test.py .
# caliban cloud --experiment_config config.json --xgroup mygroup --gpu_spec 2xV100  caliban_test.py
# ```
# You can specify the kind of machines you want to use as explained [here](https://caliban.readthedocs.io/en/latest/cloud/gpu_specs.html). If you omit "--gpu_spec", it defaults to n1-standard-8 with a single P100 GPU.
#
#
# - open the URL that it prints to monitor progress. Example:
# ```
# Visit https://console.cloud.google.com/ai-platform/jobs/?projectId=probml to see the status of all jobs.
#  ```
# You should see something like this:
# <img src="https://github.com/probml/pyprobml/blob/
# master/book1/intro/figures/GCP-jobs.png?raw=true">
#
# - Monitor your jobs by clicking on 'view logs'.   You should see something like this:
# <img src="https://github.com/probml/pyprobml/blob/
# master/book1/intro/figures/GCP-logs-GPU.png?raw=true">
#
# - When jobs are done,  download  the log files using [caliban_save_logs.py](https://github.com/probml/pyprobml/blob/master/scripts/caliban_save_logs.py). Example:
# ```
# python ~/github/pyprobml/scripts/caliban_save_logs.py --xgroup mygroup 
# ```
#
# - Upload the log files to Google drive and parse them  inside colab using python code below.
#

# + [markdown] id="MyY8uW4j13k3"
# # Parse the log files

# + colab={"base_uri": "https://localhost:8080/"} id="wlVvzb_yPDkW" outputId="54902e5e-a736-47e1-8317-6216ce11c468"
# !rm -rf pyprobml # Remove any old local directory to ensure fresh install
# !git clone https://github.com/probml/pyprobml


# + colab={"base_uri": "https://localhost:8080/"} id="HNvDkoDjPPDd" outputId="91492cbf-c9d0-43ac-e07e-92f265233ae0"
import pyprobml.scripts.probml_tools as pml
pml.test()


# + id="-ddv_l5fRp_k"
import pyprobml.scripts.caliban_logs_parse as parse

# + colab={"base_uri": "https://localhost:8080/"} id="nQM7oDPIQlcw" outputId="6203ae97-c341-48d2-e1a5-4cf7ebb403df"
import glob
logdir = 'https://github.com/probml/pyprobml/tree/master/data/Logs'
fnames = glob.glob(f'{logdir}/*.config')
print(fnames) # empty

# + colab={"base_uri": "https://localhost:8080/"} id="CVMC4JaiVcA0" outputId="7ee0c0df-4f32-4d9f-f102-f64bcebca5a5"
from google.colab import drive
drive.mount('/content/gdrive')

logdir = '/content/gdrive/MyDrive/Logs'
fnames = glob.glob(f'{logdir}/*.config')
print(fnames)

# + colab={"base_uri": "https://localhost:8080/", "height": 212} id="-NnEroyxPUCh" outputId="e9819fa8-1562-4d35-e184-a4ab7c18d5fd"
configs_df = parse.parse_configs(logdir)
display(configs_df)

for n in [1,2]:
  print(get_args(configs_df, n))

# + colab={"base_uri": "https://localhost:8080/", "height": 403} id="Afk0WLWetAmS" outputId="6be0e180-c5c6-4e8a-c01b-a76c92e750c3"
logdir = '/content/gdrive/MyDrive/Logs'
#df1 = log_file_to_pandas('/content/gdrive/MyDrive/Logs/caliban_kpmurphy_20210208_194505_1.log')
logs_df = parse.parse_logs(logdir)
display(logs_df.sample(n=5))


# + colab={"base_uri": "https://localhost:8080/"} id="c905D4VR-kgv" outputId="dd8bdb27-330d-4b7c-fa29-978874b11e70"
print(parse.get_log_messages(logs_df, 1))

# + colab={"base_uri": "https://localhost:8080/"} id="z26JN7y21ffg" outputId="daebf3b4-6e48-41ec-ec22-f27cbd5b9113"
print(parse.get_log_messages(logs_df, 2))

# + id="GNBhwnty-7uW"

