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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/text_autoencoders_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="Gmm_TyNcTdLD"
# # Adversarial autoencoders for text
#
# Code is from
# https://github.com/shentianxiao/text-autoencoders
#
# Paper is here: https://arxiv.org/pdf/1905.12777.pdf

# + [markdown] id="dV3lykF6TvwL"
# # Setup

# + colab={"base_uri": "https://localhost:8080/"} id="G29OhPQcTdoO" outputId="1bfdb063-5b30-4e65-f1a8-1a7a11e00e9a"
import torch
from multiprocessing import cpu_count
print(cpu_count())
print(torch.cuda.is_available())

# + colab={"base_uri": "https://localhost:8080/"} id="5JtFQPuqTaw2" outputId="cba5e1dc-40be-4a5f-cb9b-777765800836"
# !git clone https://github.com/shentianxiao/text-autoencoders.git

# + colab={"base_uri": "https://localhost:8080/"} id="rSKVhddyUUm2" outputId="423007d3-743f-4dc3-e3e7-ec5af238261a"
# !ls

# + colab={"base_uri": "https://localhost:8080/"} id="qh7fYGg9UWcz" outputId="c2b179ad-ecd6-43a5-85e4-03720518dc08"
# %cd text-autoencoders

# + colab={"base_uri": "https://localhost:8080/"} id="dTmgH_QUUYId" outputId="f1fb7a20-2a84-4f50-af88-58ff3eaff4c4"
# !ls

# + id="jrmT7wGeB6zm"


# + [markdown] id="u-FxMmZ3Txlk"
# # Data

# + colab={"base_uri": "https://localhost:8080/"} id="PJqGHr-KTo9W" outputId="7f168732-337f-4fec-c41c-5a013abe1828"
# !bash download_data.sh

# + [markdown] id="RBD4mGZ_Uqjr"
# # Train

# + colab={"base_uri": "https://localhost:8080/"} id="aPw6uP1MU0qh" outputId="908c9949-75f1-4498-de65-1bc0c823e860"
# !python train.py -h

# + colab={"base_uri": "https://localhost:8080/", "height": 154} id="6rTRlHtdXwwX" outputId="59cc4db5-b540-4d04-a644-a61fa274f213"
'''
# Path arguments
parser.add_argument('--train', metavar='FILE', required=True,
                    help='path to training file')
parser.add_argument('--valid', metavar='FILE', required=True,
                    help='path to validation file')
parser.add_argument('--save-dir', default='checkpoints', metavar='DIR',
                    help='directory to save checkpoints and outputs')
parser.add_argument('--load-model', default='', metavar='FILE',
                    help='path to load checkpoint if specified')
# Architecture arguments
parser.add_argument('--vocab-size', type=int, default=10000, metavar='N',
                    help='keep N most frequent words in vocabulary')
parser.add_argument('--dim_z', type=int, default=128, metavar='D',
                    help='dimension of latent variable z')
parser.add_argument('--dim_emb', type=int, default=512, metavar='D',
                    help='dimension of word embedding')
parser.add_argument('--dim_h', type=int, default=1024, metavar='D',
                    help='dimension of hidden state per layer')
parser.add_argument('--nlayers', type=int, default=1, metavar='N',
                    help='number of layers')
parser.add_argument('--dim_d', type=int, default=512, metavar='D',
                    help='dimension of hidden state in AAE discriminator')
# Model arguments
parser.add_argument('--model_type', default='dae', metavar='M',
                    choices=['dae', 'vae', 'aae'],
                    help='which model to learn')
parser.add_argument('--lambda_kl', type=float, default=0, metavar='R',
                    help='weight for kl term in VAE')
parser.add_argument('--lambda_adv', type=float, default=0, metavar='R',
                    help='weight for adversarial loss in AAE')
parser.add_argument('--lambda_p', type=float, default=0, metavar='R',
                    help='weight for L1 penalty on posterior log-variance')
parser.add_argument('--noise', default='0,0,0,0', metavar='P,P,P,K',
                    help='word drop prob, blank prob, substitute prob'
                         'max word shuffle distance')
# Training arguments
parser.add_argument('--dropout', type=float, default=0.5, metavar='DROP',
                    help='dropout probability (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate')
#parser.add_argument('--clip', type=float, default=0.25, metavar='NORM',
#                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size')
# Others
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
'''

# + colab={"base_uri": "https://localhost:8080/"} id="G3i3cj-0Up3_" outputId="fdf7d39c-2c73-4f18-9eb1-33d82dc0fbde"
NUM_EPOCHS = 1 # debugging
# !python train.py --epochs $NUM_EPOCHS --train data/yelp/train.txt --valid data/yelp/valid.txt --model_type aae --lambda_adv 10 --noise 0.3,0,0,0 --save-dir checkpoints/yelp/daae


# + colab={"base_uri": "https://localhost:8080/"} id="1mh7nfgidNdJ" outputId="da0ebbbe-2e7a-499f-af30-22169349f034"
NUM_EPOCHS = 10 # debugging
# !python train.py --epochs $NUM_EPOCHS --train data/yelp/train.txt --valid data/yelp/valid.txt --model_type aae --lambda_adv 10 --noise 0.3,0,0,0 --save-dir checkpoints/yelp/daae


# + colab={"base_uri": "https://localhost:8080/"} id="NaNeyq3ZYhe_" outputId="e3b4c7b6-e944-4d20-c48f-21f0d68765f9"
# !ls

# + colab={"base_uri": "https://localhost:8080/"} id="W_OdNhuwYlLa" outputId="0d3af920-5798-4986-9ccf-e4789456e166"
# !ls checkpoints/yelp/daae

# + id="e1Hq9eseq7dv"
from google.colab import files
#files.download('checkpoints/yelp/daae/model.pt')

# + [markdown] id="fSRiEn32rCL7"
# # Upload pretrained model

# + id="Y3AqiUXFrEcN"
from google.colab import files
#uploaded = files.upload() # store it in checkpoints/yelp/daae/model.pt

# + [markdown] id="MoCvv4mfYSA9"
# # Reconstruction

# + id="uKXVJuyxYLWD"
# !python test.py --reconstruct --data data/yelp/test.txt --output test.rec --checkpoint checkpoints/yelp/daae/



# + colab={"base_uri": "https://localhost:8080/"} id="llD_hqk7Y3Rm" outputId="791098fd-50af-4656-aead-6efd74c08dd9"
# !ls checkpoints/yelp/daae

# + colab={"base_uri": "https://localhost:8080/"} id="BzBL_6uPY3kz" outputId="7cb96ac3-8f59-4b4b-ad27-e3ea5f8ab481"
# !head checkpoints/yelp/daae/test.rec.rec

# + colab={"base_uri": "https://localhost:8080/"} id="krtESrsPZ3y_" outputId="0284615d-54d2-4a69-c34e-d67bae5fc3bd"
# !head checkpoints/yelp/daae/test.rec.z

# + colab={"base_uri": "https://localhost:8080/"} id="aVAGRBBmZ_Wo" outputId="e57cc98a-becd-42cb-e737-b3a6ca123cc5"
# !head data/yelp/test.txt

# + [markdown] id="iYP4zjFCaS2Q"
# # Sample

# + id="oRtLTTRNaEKT"
# !python test.py --sample --n 10 --output sample --checkpoint checkpoints/yelp/daae/


# + colab={"base_uri": "https://localhost:8080/"} id="yg1MEw8uaUnY" outputId="a341b76e-dfaa-4677-bdee-33ef47b5f36c"
# !ls checkpoints/yelp/daae

# + colab={"base_uri": "https://localhost:8080/"} id="Q-EF4QvYaYqz" outputId="70f0acd9-172b-4a74-e853-6a37d099a839"
# !head checkpoints/yelp/daae/sample

# + [markdown] id="sr-THLamamjI"
# # Arithmetic
#
# The difference between the average latent representation of the first two data files will be applied to the third file (separated by commas), and k denotes the scaling factor.
#

# + [markdown] id="_Tz_Y_Txg55c"
# ## Tense

# + id="Y6cqFAmxab3j"
# !python test.py --arithmetic --data data/yelp/tense/valid.past,data/yelp/tense/valid.present,data/yelp/tense/test.past --output test.past2present --checkpoint checkpoints/yelp/daae/



# + colab={"base_uri": "https://localhost:8080/"} id="UtOj8o5AbSEh" outputId="517effc9-bcd6-436e-8cfa-831e20eae74e"
# !head data/yelp/tense/valid.past

# + colab={"base_uri": "https://localhost:8080/"} id="FqwU42qvbnTP" outputId="101396ac-1b37-444e-ee6e-5c0e6310d596"
# !head data/yelp/tense/valid.present

# + colab={"base_uri": "https://localhost:8080/"} id="Sl91aiLWbql1" outputId="084def78-c9c8-4850-f1aa-b4845eca557b"
# !head data/yelp/tense/test.past

# + colab={"base_uri": "https://localhost:8080/"} id="Df37XG-xbKin" outputId="04e98445-73dc-44dd-8cbf-2568bc8c79b6"
# !head checkpoints/yelp/daae/test.past2present

# + [markdown] id="z5nbF9p_g7pQ"
# ## Sentiment

# + id="vZPQ6sZoatXK"
# !python test.py --arithmetic --k 2 --data data/yelp/sentiment/100.neg,data/yelp/sentiment/100.pos,data/yelp/sentiment/1000.neg --output 1000.neg2pos --checkpoint checkpoints/yelp/daae/

# + colab={"base_uri": "https://localhost:8080/"} id="i1o_A_3rgs79" outputId="c9d1b8a0-ec27-4a8f-8a05-545aa124c56a"
# !head data/yelp/sentiment/100.neg

# + colab={"base_uri": "https://localhost:8080/"} id="l9vw2Pwtgxlx" outputId="7dd93048-0ac3-4a0b-da35-8f7997a5db73"
# !head data/yelp/sentiment/100.pos

# + colab={"base_uri": "https://localhost:8080/"} id="xh9fTPahg19l" outputId="64c5da57-40a8-4b88-a0ca-611d1a57086b"
# !head data/yelp/sentiment/1000.neg

# + colab={"base_uri": "https://localhost:8080/"} id="Z8LiQoRZhGLo" outputId="f8c36c27-59ee-4789-be54-66a11c1003f7"
# !head checkpoints/yelp/daae/1000.neg2pos

# + [markdown] id="327Ahgmgb86B"
# # Interpolation
#
# Sentence interpolation between two data files (separated by a comma), 
#

# + id="tGekWTjnb9lc"
# !python test.py --interpolate --data data/yelp/interpolate/example.long,data/yelp/interpolate/example.short --output example.int --checkpoint checkpoints/yelp/daae/


# + colab={"base_uri": "https://localhost:8080/"} id="Q3u-mKAhcbnO" outputId="63c8fd4c-3d15-4f9d-a68b-d96d8f5a1513"
# !head checkpoints/yelp/daae/example.int

# + colab={"base_uri": "https://localhost:8080/"} id="sbO3hWLIckRQ" outputId="8f22487f-fa94-4e26-c23b-543d2bb45e84"
# !head data/yelp/interpolate/example.long

# + colab={"base_uri": "https://localhost:8080/"} id="sEXD-qXIcpzw" outputId="be940779-9bf6-403b-dc68-bfc8631d06e0"
# !head data/yelp/interpolate/example.short

# + id="dqI6ngticulp"


# + [markdown] id="OrgQaMimDIrJ"
# # Optiponal: Reproduce fig 1 (toy dataset)
#
# The code below was sent to me by Tianxiao.
# It is not part of the repo.

# + [markdown] id="DSlNrC08EcBQ"
# ## Make dataset

# + id="skM4oJUGDOIv"
import os
import random

random.seed(1)


def gen(path, n=5, m=100, l=50, p=0.2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for i in range(n):
            c = [random.randint(0, 1) for _ in range(l)]
            for j in range(m):
                b = c.copy()
                for k in range(l):
                    if random.random() < p:
                        b[k] = 1 - b[k]
                f.write(' '.join([str(x) for x in b]) + '\n')


gen('data/toy/data.txt')

# + [markdown] id="kUG8fRVeEd77"
# ## Train

# + id="wnW6LfqRDS38" outputId="92afa290-0781-4420-b5a3-0263396a8543" colab={"base_uri": "https://localhost:8080/"}
# !python train.py --train data/toy/data.txt --valid data/toy/data.txt --model_type aae --lambda_adv 10 --dim_z 2 --save-dir checkpoints/toy/aae/ --epochs 1

# + id="As6Q8UsbDT1m" outputId="04f373f9-6038-406f-f144-84b34f485f21" colab={"base_uri": "https://localhost:8080/"}
# !python train.py --train data/toy/data.txt --valid data/toy/data.txt --model_type dae --lambda_adv 10 --dim_z 2 --noise 0,0,0.2,0 --save-dir checkpoints/toy/daae/ --epochs 1

# + id="BMqllw1KEQdL" outputId="6f95b32b-4fb2-4663-da21-2d2613afa3af" colab={"base_uri": "https://localhost:8080/"}
# !ls checkpoints/toy/aae

# + [markdown] id="I93ajRDCEf-N"
# ## Compute latent representations

# + id="OTURry3wEhoB"
# !python test.py --reconstruct --data data/toy/data.txt --output data --max-len 55 --checkpoint checkpoints/toy/aae/

# + id="36Oh5lewEnmy"
# !python test.py --reconstruct --data data/toy/data.txt --output data --max-len 55 --checkpoint checkpoints/toy/daae/

# + [markdown] id="O3jHgjW4EkUj"
# ## Plot

# + id="oq8lP-8_DyoR"
import sys
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE

def plot_z(filename):
  x = []
  with open(filename) as f:
      for line in f:
          parts = line.split()
          x.append([float(p) for p in parts])
  x = np.array(x)
  #x = TSNE().fit_transform(x)

  n, m = 5, 100
  for i in range(n):
      l, r = i*m, (i+1)*m
      plt.scatter(x[l:r, 0], x[l:r, 1])

  #plt.title('DAAE', fontsize=22)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.show()


# + id="SZ3tcCrPEFhT" outputId="704aae9b-928b-4917-ba76-fdd6d05c0529" colab={"base_uri": "https://localhost:8080/", "height": 279}
plot_z('checkpoints/toy/aae/data.z')

# + id="1dKqBy9ZELE4" outputId="d13fdc09-04cf-42ca-f770-fe08de7565ab" colab={"base_uri": "https://localhost:8080/", "height": 273}
plot_z('checkpoints/toy/daae/data.z')

# + id="PLGYhD9uEuRt"

