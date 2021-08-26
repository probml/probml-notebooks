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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/mlp_cifar_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="b520E1nCIBHc"
#
# # MLP for image classification using PyTorch
#
# In this section, we follow Chap. 7 of the [Deep Learning With PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf) book, and illustrate how to fit an MLP to a two-class version of CIFAR. (We modify the code from [here](https://github.com/deep-learning-with-pytorch/dlwpt-code/tree/master/p1ch7).)
#
#

# + id="UeuOgABaIENZ"
import sklearn
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import itertools
import time
from functools import partial

import os

import numpy as np
from scipy.special import logsumexp
np.set_printoptions(precision=3)




# + id="GPozRwDAKFb8" colab={"base_uri": "https://localhost:8080/"} outputId="982cc6c0-054f-4b26-8d54-4b4d65e6b440"


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
print("torch version {}".format(torch.__version__))
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
  print("current device {}".format(torch.cuda.current_device()))
else:
  print("Torch cannot find GPU")

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#torch.backends.cudnn.benchmark = True

# + [markdown] id="Ds9B4oxBzQ4I"
# ## Get the CIFAR dataset

# + colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["4ac2dfbb2573483a9292fd9abdd35f65", "ba188765b5ba49fbb0e21731d0561f6a", "b55fe94d59b04923ba841a96013288e6", "6633979a8e5f48df8f51013a625068f8", "3e4c3b7d9aee4060940b27b60972c7ff", "4cb38cdfd61e4aa0a474727e280e3c44", "308fedacacbb4ada9e749b7d00187f45", "f1159f0269504f7082735948433b8158"]} id="SglcKAXPyZaC" outputId="55ae6233-e431-457d-b0df-ac817fc73e55"
from torchvision import datasets

folder = 'data'
cifar10 = datasets.CIFAR10(folder, train=True, download=True)
cifar10_val = datasets.CIFAR10(folder, train=False, download=True)

# + colab={"base_uri": "https://localhost:8080/"} id="ruAhO94LzT3k" outputId="16ae2539-1e8d-41ff-a4a2-7f91072190d2"
print(type(cifar10))
print(type(cifar10).__mro__) # module resolution order shows class hierarchy


# + colab={"base_uri": "https://localhost:8080/", "height": 317} id="v65yrJkxzw5s" outputId="133a2fed-822e-4155-8513-a40889ff020d"
print(len(cifar10))
img, label = cifar10[99]
print(type(img))
print(img)
plt.imshow(img)
plt.show()


# + colab={"base_uri": "https://localhost:8080/", "height": 213} id="gqx19_tX0EcX" outputId="4b445696-e835-445a-fac9-4b7236afd78a"
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

fig = plt.figure(figsize=(8,3))
num_classes = 10
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.set_title(class_names[i])
    img = next(img for img, label in cifar10 if label == i)
    plt.imshow(img)
plt.show()

# + [markdown] id="O3HCDpb01rWU"
# ## Convert to tensors

# + colab={"base_uri": "https://localhost:8080/"} id="NYahB-cP0Ix4" outputId="ec04f25a-3ab3-4002-a95b-66d95fb7ca91"
# Now we want to convert this to a tensor
from torchvision import transforms

to_tensor = transforms.ToTensor()

img, label = cifar10[99]
img_t = to_tensor(img)
print(type(img))
#print(img.shape)
print(type(img_t))
print(img_t.shape) # channels * height * width, here channels=3 (RGB)
print(img_t.min(), img_t.max()) # pixel values are rescaled to 0..1


# + id="0R4C9E5e0pNE"
# transform the whole dataset to tensors
cifar10 = datasets.CIFAR10(folder, train=True, download=False,
                          transform=transforms.ToTensor())

# + colab={"base_uri": "https://localhost:8080/", "height": 283} id="cMAdGB1x1FF3" outputId="f36d2f9c-9fc7-4c99-a0c3-9fb8a1c3063a"
img, label = cifar10[99]
print(type(img))
plt.imshow(img.permute(1, 2, 0)) # matplotlib expects H*W*C
plt.show()


# + [markdown] id="RjVR6T0P1tu7"
# ## Standardize the inputs
#
# We standardize the features by  computing the mean and std of each channel, averaging across all pixels and all images. This will help optimization.

# + colab={"base_uri": "https://localhost:8080/"} id="U82Rd6F91WwE" outputId="b4cd54b4-3648-424d-d86e-ece9fd885359"
# we load the whole training set as a batch, of size 3*H*W*N

imgs = torch.stack([img for img, _ in cifar10], dim=3)
print(imgs.shape)


# + colab={"base_uri": "https://localhost:8080/"} id="QWswsfLi2XLw" outputId="c404a14f-5b06-4369-c3a7-100205dbfca3"
imgs_flat =  imgs.view(3, -1) # reshape by keeping first 3 channels, but flatten all others 
print(imgs_flat.shape)
mu = imgs_flat.mean(dim=1) # average over second dimension (H*W*N) to get one mean per channel
sigma = imgs_flat.std(dim=1)
print(mu)
print(sigma)

# + id="ZfR76Z2K2hHU"
cifar10 = datasets.CIFAR10(folder, train=True, download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mu, sigma)
                          ]))

cifar10_val = datasets.CIFAR10(folder, train=False, download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mu, sigma),
                          ]))

# + colab={"base_uri": "https://localhost:8080/", "height": 283} id="s2RfiP5_29y9" outputId="95e1f145-e207-4de2-9a1f-48aac92e8b3c"
# rescaled data is harder to visualize
img, _ = cifar10[99]

plt.imshow(img.permute(1, 2, 0))
plt.show()


# + [markdown] id="4NRd9JhE3UBa"
# ## Create two-class version of dataset
#
# We extract data which correspond to airplane or bird.
# The result object is a list of pairs.
# This "acts like" an object of type torch.utilts.data.dataset.Dataset, since it implements the len() and item index methods.

# + id="eVFSJNot3FfY"
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

label_map = {0: 0, 2: 1} # 0(airplane)->0, 2(bird)->1
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]

# + colab={"base_uri": "https://localhost:8080/"} id="UuYEk0Mu3sRA" outputId="0c3e4032-3953-4eca-fc52-9465753c7954"
print(len(cifar2))
print(len(cifar2_val))


# + [markdown] id="HwzZkD7rIWL2"
# ## A shallow, fully connected model

# + id="4X2yWXzUOVrB"
img, label = cifar10[0]
img = img.view(-1)
ninputs = len(img)
nhidden = 512
nclasses = 2

# + colab={"base_uri": "https://localhost:8080/"} id="ubw-BPamIVg_" outputId="71996fd9-e9ba-404e-c2bb-4cf0158f99f2"
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(ninputs, nhidden),
            nn.Tanh(),
            nn.Linear(nhidden, nclasses),
            nn.LogSoftmax(dim=1))
print(model)



# + [markdown] id="NEfAypQtSGFz"
# We can name the layers so we can access their activations and/or parameters more easily.

# + colab={"base_uri": "https://localhost:8080/"} id="43MK_CqkN31h" outputId="2a76b5db-e884-4439-c0d4-570c71faa21e"
torch.manual_seed(0)
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
            ('hidden_linear', nn.Linear(ninputs, nhidden)),
            ('activation', nn.Tanh()),
            ('output_linear', nn.Linear(nhidden, nclasses)),
            ('softmax', nn.LogSoftmax(dim=1))
            ]))
print(model)

# + [markdown] id="PPlKZgTDJfKL"
# Let's test the model.

# + colab={"base_uri": "https://localhost:8080/"} id="9gt1iSSaJhtm" outputId="eda20f15-3ad8-40ab-f2d5-b02bf24c986e"
img, label = cifar2[0]
img_batch = img.view(-1).unsqueeze(0)
print(img_batch.shape)
logprobs = model(img_batch)
print(logprobs.shape)
print(logprobs)
probs = torch.exp(logprobs) # elementwise
print(probs)
print(probs.sum(1))


# + [markdown] id="_1hMQhDQLSHI"
#  Negative log likelihood loss.

# + colab={"base_uri": "https://localhost:8080/"} id="WbMn-lhoJ6_t" outputId="8aedc99a-6424-48f0-e53f-5a9dabe62b25"

loss_fn = nn.NLLLoss()
loss = loss_fn(logprobs, torch.tensor([label]))
print(loss)

# + [markdown] id="Z9ZSuffoPG3k"
# Let's access the output of the logit layer directly, bypassing the final log softmax.
# (We borrow a trick from [here](https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6)).

# + colab={"base_uri": "https://localhost:8080/"} id="C6bM48-YNj13" outputId="4322fe09-99d1-4f11-b123-11d51b85d20c"
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.output_linear.register_forward_hook(get_activation('output_linear'))

logprobs = model(img_batch).detach().numpy()
logits = activation['output_linear']
logprobs2 = F.log_softmax(logits).detach().numpy()

print(logprobs)
print(logprobs2)
assert(np.allclose(logprobs, logprobs2))


# + [markdown] id="yUtXCruaLZCy"
# We can also modify the model to return logits.

# + colab={"base_uri": "https://localhost:8080/"} id="z21Yx0ZtLWjf" outputId="6c7b1bd4-f19d-4896-c304-57f587154204"
torch.manual_seed(0)
model_logits = nn.Sequential(
            nn.Linear(ndims_input, nhidden),
            nn.Tanh(),
            nn.Linear(nhidden, nclasses))

logits2 = model_logits(img_batch)
print(logits)
print(logits2)
torch.testing.assert_allclose(logits, logits2)

# + [markdown] id="poTLbKR_SmvQ"
# In this case, we need to modify the loss to take in logits.

# + colab={"base_uri": "https://localhost:8080/"} id="QH8CTYcvS-Bu" outputId="2a86971b-1259-47b6-fac5-74c77ac63676"
logprobs = model(img_batch)
loss = nn.NLLLoss()(logprobs, torch.tensor([label]))

logits = model_logits(img_batch)
loss2 = nn.CrossEntropyLoss()(logits, torch.tensor([label]))

print(loss)
print(loss2)
torch.testing.assert_allclose(loss, loss2)

# + [markdown] id="f0wBb1r8Vv8l"
# We can also use the functional API to specify the model. This avoids having to create stateless layers (i.e., layers with no adjustable parameters), such as the tanh or softmax layers.

# + colab={"base_uri": "https://localhost:8080/"} id="Euo9zE2ITNi_" outputId="2291f5a8-0309-41eb-820c-f8081c60e327"

  
class MLP(nn.Module):
  def __init__(self, ninputs, nhidden, nclasses):
    super().__init__()
    self.fc1 = nn.Linear(ninputs, nhidden)
    self.fc2 = nn.Linear(nhidden, nclasses)

  def forward(self, x):
    out = F.tanh(self.fc1(x))
    out = self.fc2(out)
    return out # logits

torch.manual_seed(0)
model = MLP(ninputs, nhidden, nclasses)
logits = model(img_batch)
logits2 = model_logits(img_batch)
print(logits)
print(logits2)
torch.testing.assert_allclose(logits, logits2)

#print(list(model.named_parameters()))
nparams = [p.numel() for p in model.parameters() if p.requires_grad == True]
print(nparams)
# weights1, bias1, weights2, bias2
print([ninputs*nhidden, nhidden, nhidden*nclasses, nclasses])


# + [markdown] id="0sz_8jyeXvYT"
# ## Evaluation pre-training

# + colab={"base_uri": "https://localhost:8080/"} id="PV2nM6f3XxR5" outputId="6264aac5-4103-4e0e-a94c-0e866fc9b1c8"
def compute_accuracy(model, loader):
  correct = 0
  total = 0
  with torch.no_grad():
      for imgs, labels in loader:
          outputs = model(imgs.view(imgs.shape[0], -1))
          _, predicted = torch.max(outputs, dim=1)
          total += labels.shape[0]
          correct += int((predicted == labels).sum())
  return correct / total

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

torch.manual_seed(0)
model = MLP(ninputs, nhidden, nclasses)

acc_train = compute_accuracy(model, train_loader)
acc_val = compute_accuracy(model, val_loader)
print([acc_train, acc_val])

# + [markdown] id="JuYNB_huVV_G"
# ## Training loop

# + colab={"base_uri": "https://localhost:8080/"} id="BKUIE0f4VYgM" outputId="8ace862d-301e-4716-debf-140890de28d8"
torch.manual_seed(0)
model = MLP(ninputs, nhidden, nclasses)

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 20

for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        outputs = model(imgs.view(imgs.shape[0], -1))
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # At end of each epoch
    acc_val = compute_accuracy(model, val_loader)
    loss_train_batch = float(loss)
    print(f"Epoch {epoch}, Batch Loss {loss_train_batch}, Val acc {acc_val}")
