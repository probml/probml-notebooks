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

# + [markdown] id="VAcj8IZxaUth"
# ### Source:
# 1) https://github.com/eitanrich/torch-mfa 
#
# 2) https://github.com/eitanrich/gans-n-gmms
#
#
#

# + [markdown] id="o_vaqr1IEdWV"
# ## Get the CelebA dataset

# + id="_2KaFVMC5OsQ"
# !wget -q https://raw.githubusercontent.com/sayantanauddy/vae_lightning/main/data.py 

# + [markdown] id="hzck8LfvVmmS"
# ## Get helpers

# + id="PUsycQy8VqYM"
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/mfa_celeba_helpers.py

# + [markdown] id="91H5S1c0EkN6"
# ## Get the Kaggle api token and upload it to colab. Follow the instructions [here](https://github.com/Kaggle/kaggle-api#api-credentials).

# + colab={"base_uri": "https://localhost:8080/"} id="vbKcXwCedIN8" outputId="4b1aa883-7b65-4b3f-a65e-c57777e67331"
# !pip install kaggle

# + colab={"resources": {"http://localhost:8080/nbextensions/google.colab/files.js": {"data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7CgpmdW5jdGlvbiBfdXBsb2FkRmlsZXMoaW5wdXRJZCwgb3V0cHV0SWQpIHsKICBjb25zdCBzdGVwcyA9IHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCk7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICAvLyBDYWNoZSBzdGVwcyBvbiB0aGUgb3V0cHV0RWxlbWVudCB0byBtYWtlIGl0IGF2YWlsYWJsZSBmb3IgdGhlIG5leHQgY2FsbAogIC8vIHRvIHVwbG9hZEZpbGVzQ29udGludWUgZnJvbSBQeXRob24uCiAgb3V0cHV0RWxlbWVudC5zdGVwcyA9IHN0ZXBzOwoKICByZXR1cm4gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpOwp9CgovLyBUaGlzIGlzIHJvdWdobHkgYW4gYXN5bmMgZ2VuZXJhdG9yIChub3Qgc3VwcG9ydGVkIGluIHRoZSBicm93c2VyIHlldCksCi8vIHdoZXJlIHRoZXJlIGFyZSBtdWx0aXBsZSBhc3luY2hyb25vdXMgc3RlcHMgYW5kIHRoZSBQeXRob24gc2lkZSBpcyBnb2luZwovLyB0byBwb2xsIGZvciBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcC4KLy8gVGhpcyB1c2VzIGEgUHJvbWlzZSB0byBibG9jayB0aGUgcHl0aG9uIHNpZGUgb24gY29tcGxldGlvbiBvZiBlYWNoIHN0ZXAsCi8vIHRoZW4gcGFzc2VzIHRoZSByZXN1bHQgb2YgdGhlIHByZXZpb3VzIHN0ZXAgYXMgdGhlIGlucHV0IHRvIHRoZSBuZXh0IHN0ZXAuCmZ1bmN0aW9uIF91cGxvYWRGaWxlc0NvbnRpbnVlKG91dHB1dElkKSB7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICBjb25zdCBzdGVwcyA9IG91dHB1dEVsZW1lbnQuc3RlcHM7CgogIGNvbnN0IG5leHQgPSBzdGVwcy5uZXh0KG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSk7CiAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShuZXh0LnZhbHVlLnByb21pc2UpLnRoZW4oKHZhbHVlKSA9PiB7CiAgICAvLyBDYWNoZSB0aGUgbGFzdCBwcm9taXNlIHZhbHVlIHRvIG1ha2UgaXQgYXZhaWxhYmxlIHRvIHRoZSBuZXh0CiAgICAvLyBzdGVwIG9mIHRoZSBnZW5lcmF0b3IuCiAgICBvdXRwdXRFbGVtZW50Lmxhc3RQcm9taXNlVmFsdWUgPSB2YWx1ZTsKICAgIHJldHVybiBuZXh0LnZhbHVlLnJlc3BvbnNlOwogIH0pOwp9CgovKioKICogR2VuZXJhdG9yIGZ1bmN0aW9uIHdoaWNoIGlzIGNhbGxlZCBiZXR3ZWVuIGVhY2ggYXN5bmMgc3RlcCBvZiB0aGUgdXBsb2FkCiAqIHByb2Nlc3MuCiAqIEBwYXJhbSB7c3RyaW5nfSBpbnB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIGlucHV0IGZpbGUgcGlja2VyIGVsZW1lbnQuCiAqIEBwYXJhbSB7c3RyaW5nfSBvdXRwdXRJZCBFbGVtZW50IElEIG9mIHRoZSBvdXRwdXQgZGlzcGxheS4KICogQHJldHVybiB7IUl0ZXJhYmxlPCFPYmplY3Q+fSBJdGVyYWJsZSBvZiBuZXh0IHN0ZXBzLgogKi8KZnVuY3Rpb24qIHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IGlucHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlucHV0SWQpOwogIGlucHV0RWxlbWVudC5kaXNhYmxlZCA9IGZhbHNlOwoKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIG91dHB1dEVsZW1lbnQuaW5uZXJIVE1MID0gJyc7CgogIGNvbnN0IHBpY2tlZFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgaW5wdXRFbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsIChlKSA9PiB7CiAgICAgIHJlc29sdmUoZS50YXJnZXQuZmlsZXMpOwogICAgfSk7CiAgfSk7CgogIGNvbnN0IGNhbmNlbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2J1dHRvbicpOwogIGlucHV0RWxlbWVudC5wYXJlbnRFbGVtZW50LmFwcGVuZENoaWxkKGNhbmNlbCk7CiAgY2FuY2VsLnRleHRDb250ZW50ID0gJ0NhbmNlbCB1cGxvYWQnOwogIGNvbnN0IGNhbmNlbFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgY2FuY2VsLm9uY2xpY2sgPSAoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9OwogIH0pOwoKICAvLyBXYWl0IGZvciB0aGUgdXNlciB0byBwaWNrIHRoZSBmaWxlcy4KICBjb25zdCBmaWxlcyA9IHlpZWxkIHsKICAgIHByb21pc2U6IFByb21pc2UucmFjZShbcGlja2VkUHJvbWlzZSwgY2FuY2VsUHJvbWlzZV0pLAogICAgcmVzcG9uc2U6IHsKICAgICAgYWN0aW9uOiAnc3RhcnRpbmcnLAogICAgfQogIH07CgogIGNhbmNlbC5yZW1vdmUoKTsKCiAgLy8gRGlzYWJsZSB0aGUgaW5wdXQgZWxlbWVudCBzaW5jZSBmdXJ0aGVyIHBpY2tzIGFyZSBub3QgYWxsb3dlZC4KICBpbnB1dEVsZW1lbnQuZGlzYWJsZWQgPSB0cnVlOwoKICBpZiAoIWZpbGVzKSB7CiAgICByZXR1cm4gewogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgICAgfQogICAgfTsKICB9CgogIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcykgewogICAgY29uc3QgbGkgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdsaScpOwogICAgbGkuYXBwZW5kKHNwYW4oZmlsZS5uYW1lLCB7Zm9udFdlaWdodDogJ2JvbGQnfSkpOwogICAgbGkuYXBwZW5kKHNwYW4oCiAgICAgICAgYCgke2ZpbGUudHlwZSB8fCAnbi9hJ30pIC0gJHtmaWxlLnNpemV9IGJ5dGVzLCBgICsKICAgICAgICBgbGFzdCBtb2RpZmllZDogJHsKICAgICAgICAgICAgZmlsZS5sYXN0TW9kaWZpZWREYXRlID8gZmlsZS5sYXN0TW9kaWZpZWREYXRlLnRvTG9jYWxlRGF0ZVN0cmluZygpIDoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ24vYSd9IC0gYCkpOwogICAgY29uc3QgcGVyY2VudCA9IHNwYW4oJzAlIGRvbmUnKTsKICAgIGxpLmFwcGVuZENoaWxkKHBlcmNlbnQpOwoKICAgIG91dHB1dEVsZW1lbnQuYXBwZW5kQ2hpbGQobGkpOwoKICAgIGNvbnN0IGZpbGVEYXRhUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICAgIGNvbnN0IHJlYWRlciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHJlYWRlci5vbmxvYWQgPSAoZSkgPT4gewogICAgICAgIHJlc29sdmUoZS50YXJnZXQucmVzdWx0KTsKICAgICAgfTsKICAgICAgcmVhZGVyLnJlYWRBc0FycmF5QnVmZmVyKGZpbGUpOwogICAgfSk7CiAgICAvLyBXYWl0IGZvciB0aGUgZGF0YSB0byBiZSByZWFkeS4KICAgIGxldCBmaWxlRGF0YSA9IHlpZWxkIHsKICAgICAgcHJvbWlzZTogZmlsZURhdGFQcm9taXNlLAogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbnRpbnVlJywKICAgICAgfQogICAgfTsKCiAgICAvLyBVc2UgYSBjaHVua2VkIHNlbmRpbmcgdG8gYXZvaWQgbWVzc2FnZSBzaXplIGxpbWl0cy4gU2VlIGIvNjIxMTU2NjAuCiAgICBsZXQgcG9zaXRpb24gPSAwOwogICAgZG8gewogICAgICBjb25zdCBsZW5ndGggPSBNYXRoLm1pbihmaWxlRGF0YS5ieXRlTGVuZ3RoIC0gcG9zaXRpb24sIE1BWF9QQVlMT0FEX1NJWkUpOwogICAgICBjb25zdCBjaHVuayA9IG5ldyBVaW50OEFycmF5KGZpbGVEYXRhLCBwb3NpdGlvbiwgbGVuZ3RoKTsKICAgICAgcG9zaXRpb24gKz0gbGVuZ3RoOwoKICAgICAgY29uc3QgYmFzZTY0ID0gYnRvYShTdHJpbmcuZnJvbUNoYXJDb2RlLmFwcGx5KG51bGwsIGNodW5rKSk7CiAgICAgIHlpZWxkIHsKICAgICAgICByZXNwb25zZTogewogICAgICAgICAgYWN0aW9uOiAnYXBwZW5kJywKICAgICAgICAgIGZpbGU6IGZpbGUubmFtZSwKICAgICAgICAgIGRhdGE6IGJhc2U2NCwKICAgICAgICB9LAogICAgICB9OwoKICAgICAgbGV0IHBlcmNlbnREb25lID0gZmlsZURhdGEuYnl0ZUxlbmd0aCA9PT0gMCA/CiAgICAgICAgICAxMDAgOgogICAgICAgICAgTWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCk7CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPSBgJHtwZXJjZW50RG9uZX0lIGRvbmVgOwoKICAgIH0gd2hpbGUgKHBvc2l0aW9uIDwgZmlsZURhdGEuYnl0ZUxlbmd0aCk7CiAgfQoKICAvLyBBbGwgZG9uZS4KICB5aWVsZCB7CiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICB9CiAgfTsKfQoKc2NvcGUuZ29vZ2xlID0gc2NvcGUuZ29vZ2xlIHx8IHt9OwpzY29wZS5nb29nbGUuY29sYWIgPSBzY29wZS5nb29nbGUuY29sYWIgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYi5fZmlsZXMgPSB7CiAgX3VwbG9hZEZpbGVzLAogIF91cGxvYWRGaWxlc0NvbnRpbnVlLAp9Owp9KShzZWxmKTsK", "ok": true, "headers": [["content-type", "application/javascript"]], "status": 200, "status_text": ""}}, "base_uri": "https://localhost:8080/", "height": 72} id="qncitxB3oyVF" outputId="4a9347a8-b78f-4eda-edef-58054e048e52"
from google.colab import files
uploaded = files.upload() 

# + id="Yjc9kbpaoyki"
# !mkdir /root/.kaggle

# + id="I3hCPht5pjwa"
# !cp kaggle.json /root/.kaggle/kaggle.json

# + id="81GmoSPCpj4T"
# !chmod 600 /root/.kaggle/kaggle.json

# + [markdown] id="jpW1offQ-eRh"
# ## Getting the checkpoint of the model from buckets. 

# + id="x2vQduXW7K__"
from google.colab import auth
auth.authenticate_user()

# + id="g43d9dfRxTif"
bucket_name = 'probml_data' 

# + id="eV0nGz0jKxIt"
# !mkdir /content/models 

# + id="2Evj8R-wKxSt" colab={"base_uri": "https://localhost:8080/"} outputId="84964809-81ec-4326-c253-a328cf0b2d1b"
# !gsutil cp -r gs://{bucket_name}/mix_PPCA /content/models/

# + [markdown] id="DWU2S_TslNHS"
# # Main

# + colab={"base_uri": "https://localhost:8080/"} id="ydVN046dBGsP" outputId="6d711c17-6588-4897-a223-c286fdfa862d"
# !pip install pytorch-lightning 

# + colab={"base_uri": "https://localhost:8080/"} id="hs-gd9gwBSbY" outputId="983c08b7-f077-427d-c441-5c3e12568037"
# !pip install torchvision

# + id="mOIYhDs5kgYh"
import sys, os
import torch
from torchvision.datasets import CelebA, MNIST
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split, SequentialSampler, RandomSampler
import numpy as np
from matplotlib import pyplot as plt
from imageio import imwrite
from packaging import version
from tqdm import tqdm
from data import CelebADataset,  CelebADataModule
from mfa_celeba_helpers import *
from IPython.display import Image

def main(argv):
    assert version.parse(torch.__version__) >= version.parse('1.2.0')

    dataset = argv[1] if len(argv) == 2 else 'celeba'
    print('Preparing dataset and parameters for', dataset, '...')

    if dataset == 'celeba':
        image_shape = [64, 64, 3]       # The input image shape
        n_components = 300              # Number of components in the mixture model
        n_factors = 10                  # Number of factors - the latent dimension (same for all components)
        batch_size = 1000               # The EM batch size
        num_iterations = 30             # Number of EM iterations (=epochs)
        feature_sampling = 0.2          # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0              # Perform additional training with diagonal (per-pixel) covariance, using SGD
        init_method = 'rnd_samples'     # Initialize each component from few random samples using PPCA
        trans = transforms.Compose([CropTransform((25, 50, 25+128, 50+128)), transforms.Resize(image_shape[0]), transforms.ToTensor(),  ReshapeTransform([-1])])
        train_set = CelebADataset(root='./data', split='train', transform=trans, download=True) 
        test_set = CelebADataset(root='./data', split='test', transform=trans, download=True) 
        
    elif dataset == 'mnist':
        image_shape = [28, 28]          # The input image shape
        n_components = 50               # Number of components in the mixture model
        n_factors = 6                   # Number of factors - the latent dimension (same for all components)
        batch_size = 1000               # The EM batch size
        num_iterations = 1              # Number of EM iterations (=epochs)
        feature_sampling = False        # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0              # Perform additional training with diagonal (per-pixel) covariance, using SGD
        init_method = 'kmeans'          # Initialize by using k-means clustering
        trans = transforms.Compose([transforms.ToTensor(),  ReshapeTransform([-1])])
        train_set = MNIST(root='./data', train=True, transform=trans, download=True)
        test_set = MNIST(root='./data', train=False, transform=trans, download=True)
    else:
        assert False, 'Unknown dataset: ' + dataset



# + [markdown] id="yixzHO5DlSMk"
# # Inference

# + [markdown] id="2GrWR4FnBqCc"
# ### Preparing dataset

# + id="wFnoD89Nkgf7" colab={"base_uri": "https://localhost:8080/"} outputId="bf436f87-1e20-4643-b0ae-888feea5bcfa"
"""
Examples for inference using the trained MFA model - likelihood evaluation and (conditional) reconstruction
"""

if __name__ == "__main__":
    dataset = 'celeba'
    find_outliers = True
    reconstruction = True
    inpainting = True  

    print('Preparing dataset and parameters for', dataset, '...')
    if dataset == 'celeba':
        image_shape = [64, 64, 3]       # The input image shape
        n_components = 300              # Number of components in the mixture model
        n_factors = 10                  # Number of factors - the latent dimension (same for all components)
        batch_size = 128                # The EM batch size
        num_iterations = 30             # Number of EM iterations (=epochs)
        feature_sampling = 0.2          # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0              # Perform additional training with diagonal (per-pixel) covariance, using SGD
        trans = transforms.Compose([CropTransform((25, 50, 25+128, 50+128)), transforms.Resize(image_shape[0]),
                                    transforms.ToTensor(),  ReshapeTransform([-1])])
        test_dataset = CelebADataset(root='./data', split='test', transform=trans, download=True)
        # The train set has more interesting outliers...
        # test_dataset = CelebA(root='./data', split='train', transform=trans, download=True)
    else:
        assert False, 'Unknown dataset: ' + dataset


# + id="G6vik7_bz-Fi"
def samples_to_mosaic_any_size(gird_size, samples, image_shape=[64, 64, 3]):
    images = samples_to_np_images(samples, image_shape)
    num_images = images.shape[0]
    num_cols = gird_size[1] 
    num_rows = gird_size[0]
    rows = []
    for i in range(num_rows): 
        rows.append(np.hstack([images[j] for j in range(i*num_cols, (i+1)*num_cols)])) 
    return np.vstack(rows)


# + [markdown] id="MrFtqWH7BvU8"
# ### Loading pre-trained MFA model

# + colab={"base_uri": "https://localhost:8080/"} id="ZAmlhaIaNJt3" outputId="f6285284-aeee-4304-e818-560d46cc1fad"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_dir = './models/' + 'mix_PPCA' 
figures_dir = './figures/' + dataset
os.makedirs(figures_dir, exist_ok=True)

print('Loading pre-trained MFA model...')
model = MFA(n_components=n_components, n_features=np.prod(image_shape), n_factors=n_factors).to(device=device)
model.load_state_dict(torch.load(os.path.join(model_dir, 'model_c_300_l_10_init_rnd_samples.pth'))) 


# + [markdown] id="eVHJT8PTJxjF"
# ### Samples 

# + colab={"base_uri": "https://localhost:8080/"} id="Qs_-lpF0nnE1" outputId="a29aab48-faa8-42d6-afef-d87fa6f09b1f"
gird_size = [int(x) for x in input("Enter gird size: ").split()] 

# + id="3QevXnfKJzzq" colab={"base_uri": "https://localhost:8080/", "height": 443} outputId="5667c3d3-e5ef-4589-98dc-eb79cfb2c620"
 print('Visualizing the trained model...')
model_image = visualize_model(model, image_shape=image_shape, end_component=10)
fname = os.path.join(figures_dir, 'model.jpg')
imwrite(fname, model_image)
display(Image(fname))

# + colab={"base_uri": "https://localhost:8080/", "height": 243} id="nkJA9mhou8vT" outputId="7ebc63e2-08dd-4c6e-abd1-e198c35337ef"
print('Generating random samples...')
rnd_samples, _ = model.sample(gird_size[0]*gird_size[1], with_noise=False) #100->n #gird_size[0]*gird_size[1]
mosaic = samples_to_mosaic_any_size(gird_size, samples=rnd_samples, image_shape=image_shape)
fname = os.path.join(figures_dir, 'samples.jpg')
imwrite(fname, mosaic)
display(Image(fname))

# + [markdown] id="ge7NxeG2B0qG"
# ### Showing outliers

# + colab={"base_uri": "https://localhost:8080/"} id="sdIX39iGoAyf" outputId="7a21c4eb-d2f2-43b0-bc4d-7baf94247a6e"
gird_size = [int(x) for x in input("Enter gird size: ").split()] 

# + id="IrG06uhNM5_X" colab={"base_uri": "https://localhost:8080/", "height": 186} outputId="25a2d072-4d67-4d46-f9c8-f0f78bdf046e"
if find_outliers:
    print('Finding dataset outliers...')
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    all_ll = []
    for batch_x, _ in tqdm(loader):
        all_ll.append(model.log_prob(batch_x.to(device)))
    all_ll = torch.cat(all_ll, dim=0)
    ll_sorted = torch.argsort(all_ll).cpu().numpy()

    all_keys = [key for key in SequentialSampler(test_dataset)]
    outlier_samples, _ = zip(*[test_dataset[all_keys[ll_sorted[i]]] for i in range(gird_size[0]*gird_size[1])]) 
    mosaic = samples_to_mosaic_any_size(gird_size, torch.stack(outlier_samples), image_shape=image_shape)
    fname = os.path.join(figures_dir, 'outliers.jpg')
    imwrite(fname, mosaic)
    display(Image(fname))

# + [markdown] id="M_VziRVnCB5C"
# ### Reconstructing original masked images 

# + colab={"base_uri": "https://localhost:8080/"} id="krXEz__xohjb" outputId="d1a188ce-e2d2-43c2-ee31-c56c132d8c6e"
mask_type = input("Enter the type of mask from following options: (a)centre (b)bottom (c)right (d)left (e)top: ") 
gird_size = [int(x) for x in input("Enter gird size: ").split()] 

# + colab={"base_uri": "https://localhost:8080/", "height": 452} id="jSZ7MDigOg6Y" outputId="3f44ec7f-9e30-4610-c37b-eb848e5df3bd"
if reconstruction:
    print('Reconstructing images from the trained model...')
    n = gird_size[0]*gird_size[1]
    random_samples, _ = zip(*[test_dataset[k] for k in RandomSampler(test_dataset, replacement=True, num_samples=n)]) #num_samples -> gird_size = [m1, m2]
    random_samples = torch.stack(random_samples)

    if inpainting:
        
        w = image_shape[0]
        mask = np.ones([3, w, w], dtype=np.float32)   # Hide part of each image

        if (mask_type=="centre"):
          mask[:, w//4:-w//4, w//4:-w//4] = 0         #Masking centre
        elif (mask_type=="bottom"):
          mask[:, w//2:, :] = 0                       #Masking bottom half
        elif (mask_type=="right"):
          mask[:, :, w//2:] = 0                       #Masking right half
        elif (mask_type=="left"):
          mask[:, :, :w//2] = 0                       #Masking left half
        else:
          mask[:, :w//2, :] = 0                       #Masking top half

        mask = torch.from_numpy(mask.flatten()).reshape([1, -1])
        random_samples *= mask
        used_features = torch.nonzero(mask.flatten()).flatten()
        reconstructed_samples = model.conditional_reconstruct(random_samples.to(device), observed_features=used_features).cpu()
    else:
        reconstructed_samples = model.reconstruct(random_samples.to(device)).cpu()

    if inpainting:
        reconstructed_samples = random_samples * mask + reconstructed_samples * (1 - mask)

    mosaic_original = samples_to_mosaic_any_size(gird_size, random_samples, image_shape=image_shape)
    fname = os.path.join(figures_dir, 'original_samples.jpg')
    imwrite(fname, mosaic_original)
    display(Image(fname))

    mosaic_recontructed = samples_to_mosaic_any_size(gird_size, reconstructed_samples, image_shape=image_shape)
    fname = os.path.join(figures_dir, 'reconstructed_samples.jpg')
    imwrite(fname, mosaic_recontructed)
    display(Image(fname))

