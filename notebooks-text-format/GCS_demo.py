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
# <a href="https://colab.research.google.com/github/always-newbie161/pyprobml/blob/hermissue_gcs/notebooks/GCS_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="G7sHV8tjjy-S"
# ### Mounting to GCS bucket
#
# and read/write data from/to it.

# + [markdown] id="xFPk2qWsjrqO"
# **Replace bucket_name with name of the your GCS bucket**
#
# Note: you should first create the bucket in GCP if its not yet.
#
# This [colab](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=7Z2jcRKwUHqV) shows you how to create a bucket within a project(go to GCS block)

# + id="IdZ7N_hLjL95"
bucket_name="gsoc_bucket"


# + [markdown] id="VtP4SSFkpCuT"
# This code mounts your bucket to the current session

# + id="V4adu90SjfwF"
def auth_and_mount(bucket_name):
  from google.colab import auth
  auth.authenticate_user()
  # !echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" > /etc/apt/sources.list.d/gcsfuse.list
  # !curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
  # !apt -qq update
  # !apt -qq install gcsfuse
  # !mkdir $bucket_name
  # !gcsfuse $bucket_name /content/$bucket_name


# + colab={"base_uri": "https://localhost:8080/"} id="4aerqvlMlcP_" outputId="7ab4f58e-e9c8-474a-fd23-9da44fb53ec2"
auth_and_mount(bucket_name)

# + colab={"base_uri": "https://localhost:8080/"} id="Wcr20Zy6ls9h" outputId="f048c8d9-a03c-48b0-9137-c372d2d06605"
# cd /content/$bucket_name

# + id="RJp9oEoql6GA"
# !mkdir test_folder

# + id="X8gVcAhvmA7j"
with open('./test_folder/test_file.txt', 'w') as f:
  f.write('this file get saved in the test_folder you just created')

# + [markdown] id="92Pdgij2m5vv"
# You can check in your cloud platform that these changes are rendered in your GCS bucket

# + [markdown] id="72G9xEy9nMCT"
# Reverting the changes..

# + id="t-mz_MyanG2O"
# !rm ./test_folder/test_file.txt

# + [markdown] id="eE7kaNQroXlY"
# `-rf` is used because even though you clear the directory, `.ipynb_checkpoints` will be present in the folder (in GCP), So you have to force delete the directory.

# + id="pkIx80pfnTHz"
# !rm -rf test_folder
