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

# + [markdown] id="MgaMOfMt0rLz"
# # Adversarial autoencoders for text
#
# ### Code: https://github.com/shentianxiao/text-autoencoders
#
# ### Paper: https://arxiv.org/pdf/1905.12777.pdf
#
# ### GCP account creation: https://cloud.google.com/apigee/docs/hybrid/v1.1/precog-gcpaccount
#
#

# + [markdown] id="YUYM4RbZ0nQh"
# # Set up

# + colab={"base_uri": "https://localhost:8080/"} id="5YDwUFHfIPUy" outputId="2fcd5db5-9df4-4e4d-f262-39a80448beb9"
import torch
from multiprocessing import cpu_count
print(cpu_count())
print(torch.cuda.is_available())

# + colab={"base_uri": "https://localhost:8080/"} id="OBTDYua1IPYw" outputId="4827c45d-0ec5-4020-a2fb-41b693132a78"
# !git clone https://github.com/shentianxiao/text-autoencoders.git 

# + colab={"base_uri": "https://localhost:8080/"} id="82JBs11jNxv1" outputId="5aa49ed8-f6c0-4b27-b6e8-40875307429b"
# %cd text-autoencoders 

# + colab={"base_uri": "https://localhost:8080/"} id="eIy1JfPPNx5x" outputId="37d48979-76c4-4ad0-b492-fe92e0b15d7c"
# !bash download_data.sh

# + [markdown] id="rG97oAJdvCup"
# # Imports

# + id="JeOW9eT0vBse"
import os
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from vocab import Vocab
from model import *
from utils import *
from batchify import get_batches
from train import evaluate

# + [markdown] id="AoLQcAN-wZh6"
# # Getting the checkpoints of trained model from probml-data bucket 

# + id="xs_rWp-5HEmU"
# Authentication is required to access a protected bucket. This is not required for a public one.
from google.colab import auth
auth.authenticate_user()

# + id="mai41aZaIO52"
bucket_name = 'probml_data' 

# + id="doylC-Z0wmJP"
# !mkdir /content/text-autoencoders/checkpoints

# + [markdown] id="69TeHhTC9ckp"
# How to use [gsutil](https://cloud.google.com/storage/docs/gsutil/commands/help)  

# + colab={"base_uri": "https://localhost:8080/"} id="SWU46UisvJEC" outputId="9252019e-9c70-4853-858f-c92c3c4ad191"
# !gsutil cp -r gs://{bucket_name}/text-autoencoders/vocab.txt /content/text-autoencoders/checkpoints/

# + colab={"base_uri": "https://localhost:8080/"} id="dwl-tQNfvtsj" outputId="2696d44b-728d-4e17-ff47-39363dfef3ab"
# !gsutil cp -r gs://{bucket_name}/text-autoencoders/text_ae_yelp_30_epochs.pt /content/text-autoencoders/checkpoints/

# + [markdown] id="gD8bPHl9yJYX"
# # Creating vocab

# + id="J5PB57qlRRQe"
vocab = Vocab('/content/text-autoencoders/checkpoints/vocab.txt') #os.path.join(args.checkpoint, 'vocab.txt')

# + id="vPxgjNEERRX3"
seed = 1111
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
batch_size = 100
max_len = 35

# + id="gnHUEdKDSP0o"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# + [markdown] id="8Plc0VoUyNzt"
# # Loading checkpoints
#

# + id="kNdOXogwSdkK"
ckpt = torch.load('/content/text-autoencoders/checkpoints/text_ae_yelp_30_epochs.pt')

# + id="vAMhD8pwSdrq"
train_args = ckpt['args']

# + [markdown] id="7q3AmmtjyRsI"
# # Selecting AAE model

# + id="Q1kNfaynSdy8"
model = {'dae': DAE, 'vae': VAE, 'aae': AAE}['aae'](vocab, train_args).to(device) 

# + colab={"base_uri": "https://localhost:8080/"} id="hYOyYNEpSd6l" outputId="5d58c4ed-3f05-411f-a2ab-dab12f8dba47"
model.load_state_dict(ckpt['model'])
model.flatten()
model.eval()


# + id="-5_jDM1wVg23"
def encode(sents, enc='mu'):
    assert enc == 'mu' or enc == 'z'
    batches, order = get_batches(sents, vocab, batch_size, device) 
    z = []
    for inputs, _ in batches:
        mu, logvar = model.encode(inputs)
        if enc == 'mu': 
            zi = mu
        else:
            zi = reparameterize(mu, logvar)
        z.append(zi.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z
    return z_


# + id="LNkoBqX5Vkg-"
def decode(z, dec='greedy'):
    sents = []
    i = 0
    while i < len(z):
        zi = torch.tensor(z[i: i+batch_size], device=device) 
        outputs = model.generate(zi, max_len, dec).t() 
        for s in outputs:
            sents.append([vocab.idx2word[id] for id in s[1:]])  
        i += batch_size 
    return strip_eos(sents)


# + [markdown] id="H_557yqSyZng"
# # Reconstruction

# + id="nQLJp0EdTn5i"
n = 5
sents = load_sent('/content/text-autoencoders/data/yelp/test.txt') 
z = encode(sents)
sents_rec = decode(z)
write_z(z, '/content/text-autoencoders/checkpoints/test.z') 
write_sent(sents_rec, '/content/text-autoencoders/checkpoints/test.rec') 

# + colab={"base_uri": "https://localhost:8080/"} id="yhiM2rEx2J21" outputId="04aa001e-fe6d-4230-a0f0-0014cb9fb3e9"
for i in range(n):
  sentence = ""
  rec = ""
  for word in sents[i]:
    
    sentence = sentence + word + ' '
  print("Original sentence: " + sentence)

  for word in sents_rec[i]:
    rec = rec + word + ' '
  print("Reconstructed sentence: " + rec)
  print('\n')

# + [markdown] id="8DS5hxZdzb67"
# # Sample

# + id="nW0Sv7GFYD_1"
n = 10
dim_z = 128

# + id="gghye6JGYEHi"
z = np.random.normal(size=(n, dim_z)).astype('f')
sents = decode(z)
write_sent(sents, '/content/text-autoencoders/checkpoints/sample')

# + colab={"base_uri": "https://localhost:8080/"} id="M-3zBeI2_z3E" outputId="6581c3f9-512a-40b8-c1c5-e0dd6fc6066d"
for i in range(n):
  sample = ""
  for word in sents[i]:
    sample = sample + word + ' '
  print(sample)

# + [markdown] id="0SZaVG6OZcgy"
# # Tense

# + id="QJxMcNonQNj6"
n = 10

# + id="jsStGpwcYEbr"
k = 1

# + id="B86vq6kWYEjj"
fa, fb, fc = '/content/text-autoencoders/data/yelp/tense/valid.past', '/content/text-autoencoders/data/yelp/tense/valid.present', '/content/text-autoencoders/data/yelp/tense/test.past'                     
sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
za, zb, zc = encode(sa), encode(sb), encode(sc)
zd = zc + k * (zb.mean(axis=0) - za.mean(axis=0))
sd = decode(zd)
write_sent(sd, '/content/text-autoencoders/checkpoints/test.past2present') 

# + colab={"base_uri": "https://localhost:8080/"} id="1l9vAKRFL_Se" outputId="1e57c2af-cbb5-492c-9881-b4ba4e2e9acd"
for i in range(n):
  tense = ""
  for word in sd[i]:
    tense = tense + word + ' '
  print(tense)

# + [markdown] id="VrGkOBNDcLFp"
# # Sentiment

# + id="kXXfrWvjHhRm"
n = 10

# + [markdown] id="R4-np7Cx2FRB"
# k = 2

# + id="x6h6GNiXYE5B"
k = 2

# + id="ntq-qvZdcRV8"
fa, fb, fc = '/content/text-autoencoders/data/yelp/sentiment/100.neg', '/content/text-autoencoders/data/yelp/sentiment/100.pos', '/content/text-autoencoders/data/yelp/sentiment/1000.neg'                     
sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
za, zb, zc = encode(sa), encode(sb), encode(sc)
zd = zc + k * (zb.mean(axis=0) - za.mean(axis=0))
sd = decode(zd)
write_sent(sd, '/content/text-autoencoders/checkpoints/2_1000.neg2pos') 

# + colab={"base_uri": "https://localhost:8080/"} id="qQ_r-23kMcWV" outputId="e78c080f-48d0-45c9-8352-1f563964badc"
for i in range(n):
  sentiment = ""
  for word in sd[i]:
    sentiment = sentiment + word + ' '
  print(sentiment)

# + [markdown] id="O7PDM8JE2OoJ"
# k = 1.5

# + id="AZ-LtsKW1tuL"
k = 1.5

# + id="_NM3bsqB1uXM"
fa, fb, fc = '/content/text-autoencoders/data/yelp/sentiment/100.neg', '/content/text-autoencoders/data/yelp/sentiment/100.pos', '/content/text-autoencoders/data/yelp/sentiment/1000.neg'                     
sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
za, zb, zc = encode(sa), encode(sb), encode(sc)
zd = zc + k * (zb.mean(axis=0) - za.mean(axis=0))
sd = decode(zd)
write_sent(sd, '/content/text-autoencoders/checkpoints/1_5_1000.neg2pos') 

# + colab={"base_uri": "https://localhost:8080/"} id="PfKqgDafMfyF" outputId="7315c162-db25-427e-ce6b-4d91a5c1b47c"
for i in range(n):
  sentiment = ""
  for word in sd[i]:
    sentiment = sentiment + word + ' '
  print(sentiment)

# + [markdown] id="Iyb7AuaM2ScS"
# k =1

# + id="GdK4IvDn2UWb"
k = 1

# + id="89WEKt-72Uf1"
fa, fb, fc = '/content/text-autoencoders/data/yelp/sentiment/100.neg', '/content/text-autoencoders/data/yelp/sentiment/100.pos', '/content/text-autoencoders/data/yelp/sentiment/1000.neg'                     
sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
za, zb, zc = encode(sa), encode(sb), encode(sc)
zd = zc + k * (zb.mean(axis=0) - za.mean(axis=0))
sd = decode(zd)
write_sent(sd, '/content/text-autoencoders/checkpoints/1_1000.neg2pos') 

# + colab={"base_uri": "https://localhost:8080/"} id="CWCC__ZtMh5W" outputId="a5203bea-f372-4493-db60-1138e14ee526"
for i in range(n):
  sentiment = ""
  for word in sd[i]:
    sentiment = sentiment + word + ' '
  print(sentiment)

# + [markdown] id="EOmjd8wBiaYq"
# # Interpolation

# + id="t1Ofzdc2cRuC"
f1, f2 = '/content/text-autoencoders/data/yelp/interpolate/example.long', '/content/text-autoencoders/data/yelp/interpolate/example.short'        
s1, s2 = load_sent(f1), load_sent(f2)
z1, z2 = encode(s1), encode(s2)
zi = [interpolate(z1_, z2_, n) for z1_, z2_ in zip(z1, z2)]
zi = np.concatenate(zi, axis=0)
si = decode(zi)
write_doc(si, '/content/text-autoencoders/checkpoints/example.int')                          

# + id="hDiTPEXAcRlg"
n = 10

# + colab={"base_uri": "https://localhost:8080/"} id="VUW5a_tzMntC" outputId="f04411ed-04d3-4dbd-8d56-f151a0997714"
for i in range(n):
  interpolation = ""
  for word in si[i]:
    interpolation = interpolation + word + ' '
  print(interpolation)
