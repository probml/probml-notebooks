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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/cnn_cifar_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="wwEzgnhWgIVr"
#
# # CNN for image classification using PyTorch
#
# In this section, we follow Chap. 8 of the [Deep Learning With PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf) book, and illustrate how to fit a CNN to a two-class version of CIFAR. We modify the code from [here](https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p1ch8/1_convolution.ipynb)
#

# + id="RtO_TmeDgRUp"
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from functools import partial
import os
import collections



# + colab={"base_uri": "https://localhost:8080/"} id="wTzpnH_bgjD_" outputId="5d00f8b1-85a7-4ac9-912b-78e34d02bd84"

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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# + id="ePXi5pgOvzcf"

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


# + [markdown] id="I-OIHr6Ahgdj"
# # Get the data
#
# We standardize the data. (The mean/std of the RGB channels are precomputed in the MLP version of this colab.)
#

# + colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["8ffa87f9b83f4decabddedc7b11b42d1", "64d2f5290589473bb799f9745a591c25", "f8c0fc1c00c442b2b6d89bb34141afb3", "094ba7067c6f4fedb1a79fe59ee81cca", "3a0040c40cd34465ac9f790332b71542", "809c869b098b42fc898afacb83c53e91", "1f88cab621f541bf825acda9acf1df42", "717fa68813bc459584b0ed360f94d2a5"]} id="rJkOHhx8hnAy" outputId="f6888f6c-796e-4ab1-9c1b-c6a2cfba0481"
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

from torchvision import datasets, transforms
data_path = 'data'
cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

# + id="iqZRbS5Ih9pN"
label_map = {0: 0, 2: 1}
class_names = ['plane', 'bird']
nclasses = 2
cifar2 = [(img, label_map[label])
          for img, label in cifar10
          if label in [0, 2]]
cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val
              if label in [0, 2]]

# + [markdown] id="YNUUJrQugmNE"
# # Basics of convolution
#
# Lets apply a set of `nfeatures` convolutional kernels to a gray scale image.

# + colab={"base_uri": "https://localhost:8080/"} id="DXyymfW-gjXf" outputId="d4146b17-693a-462e-d35d-3aaac25e85c1"
img, label = cifar2[0]
img_t  = img.unsqueeze(0)
print(img_t.shape)

nfeatures = 16
kernel_size = 3
conv = nn.Conv2d(3, nfeatures, kernel_size=kernel_size) 
output = conv(img_t)
print(output.shape)


# + [markdown] id="CbnqS7Cui9Kt"
# Now we adding padding to ensure output size is same as input.

# + colab={"base_uri": "https://localhost:8080/"} id="GLCDqJ03i03q" outputId="8eece124-fd12-41d2-e106-5b2a3b13118d"
nfeatures = 16
kernel_size = 3
pad = kernel_size // 2
conv = nn.Conv2d(3, nfeatures, kernel_size=kernel_size, padding=pad) 

output = conv(img_t)
print(output.shape)


# + colab={"base_uri": "https://localhost:8080/", "height": 216} id="IbnKy4jHmQTG" outputId="54a374f7-3f12-4e5f-d2f1-c6b4b351b846"
def show_results(img_batch, output_batch, index=0, channel=0):
  ax1 = plt.subplot(1, 2, 1)  
  img = img_batch[index] 
  plt.imshow(img.mean(0), cmap='gray')  
  plt.title('input')   
  plt.subplot(1, 2, 2) #, sharex=ax1, sharey=ax1) 
  plt.title('output')   
  out = output_batch[index, channel]
  plt.imshow(out.detach(), cmap='gray')
  plt.show()

show_results(img_t, output)

# + [markdown] id="JVVXo_IOn6fV"
# Currently the filter parameters are random.

# + colab={"base_uri": "https://localhost:8080/"} id="LZq-NEKKn9o3" outputId="c2691cdc-349c-4d22-bc63-2b997d812919"
print(conv)
print(conv.weight.shape) # nfeatures x nchannels x kernel x kernel
print(conv.bias.shape) # nfeatures 
print(conv.weight[0,:,:,:])

# + [markdown] id="Sgu8ZdFOnKhH"
# Let's make the first filter just be an averaging operation.

# + colab={"base_uri": "https://localhost:8080/", "height": 403} id="RMEzbPVImrGc" outputId="0d426a8a-c73f-45f2-df39-0b3e8a08575f"

with torch.no_grad():
    conv.bias.zero_()
    
with torch.no_grad():
    conv.weight.fill_(1.0 / 9.0)
print(conv.weight[0,:,:,:])

output = conv(img_t)
show_results(img_t, output)

# + [markdown] id="2nDkp2_2oIEg"
# Let's make the first filter be a vertical edge detector.

# + colab={"base_uri": "https://localhost:8080/", "height": 216} id="0jFq8d13nfdx" outputId="60cd9994-7d0e-4eef-9930-96b1580af73a"

conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

with torch.no_grad():
    conv.weight[:] = torch.tensor([[-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0]])
    conv.bias.zero_()

output = conv(img_t)
show_results(img_t, output)

# + [markdown] id="Hna7FLWDpdey"
# # Max-pooling
#
# We can reduce the size of the internal feature maps by using max-pooling.

# + colab={"base_uri": "https://localhost:8080/", "height": 233} id="3g4_2jqEoTuB" outputId="610d4668-0dcb-46be-8f36-d63b3a716524"
pool = nn.MaxPool2d(2)
output = pool(img_t)
print(output.shape)

show_results(img_t, output)

# + [markdown] id="PkA_bONKrA0x"
# # Making our first CNN

# + colab={"base_uri": "https://localhost:8080/"} id="p-j2s3Y5q0jC" outputId="63cca1c3-ec98-4ad6-a8b2-6c86919a17b8"
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, nclasses, img, nchans1=16, nhidden=32):
        super().__init__()
        nchannels, nrows, ncols = img.shape
        self.nchans1 = nchans1
        self.nchans2 = nchans1//2
        self.nhidden = nhidden
        self.nclasses = nclasses
        self.conv1 = nn.Conv2d(nchannels, self.nchans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(nchans1, self.nchans2, kernel_size=3, padding=1)
        # size of input to fc1 will be  8 * nrows/4 * ncols/4, 
        # We divide by 4 since we apply 2 maxpooling layers with size 2
        # For a 32x32 image, this becomes 8x8 times 8 channnels.
        self.nflat = nrows//4 * ncols//4
        self.fc1 = nn.Linear(self.nchans2 * self.nflat, self.nhidden)
        self.fc2 = nn.Linear(self.nhidden, self.nclasses)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, self.nchans2 * self.nflat)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

set_seed(0)
img_batch = img_t.to(device=device)
model = Net(nclasses, img_batch[0]).to(device=device)
out_batch = model(img_batch)
print(out_batch.shape)
print(out_batch)


# + [markdown] id="78RzWsB3nF2o"
# # Training loop

# + id="jnzRkdaLsOtm"
import datetime  

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, 
                  l2_regularizer=0, print_every=5):
    for epoch in range(1, n_epochs + 1):  
        loss_train = 0.0
        for imgs, labels in train_loader: 
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)  
            loss = loss_fn(outputs, labels) 

            l2_norm = sum(p.pow(2.0).sum()
                          for p in model.parameters())
            loss = loss + l2_regularizer * l2_norm

            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step() 
            loss_train += loss.item() 

        if epoch == 1 or epoch % print_every == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))  


# + colab={"base_uri": "https://localhost:8080/"} id="urZLNnYhnVNG" outputId="84581110-9fd7-4c7a-baac-b8290dc31547"

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                           shuffle=True)  

model = Net(nclasses, img_batch[0]).to(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2)  
loss_fn = nn.CrossEntropyLoss() 

training_loop(  
    n_epochs = 50,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
)

# + [markdown] id="x4zPGA_XntHO"
# # Validation accuracy

# + colab={"base_uri": "https://localhost:8080/"} id="weB--S4FnVcQ" outputId="8930c375-6cd5-4c23-a3ee-9e5ee731dacb"
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64,
                                         shuffle=False)

def accuracy(model, loader):
  correct = 0
  total = 0
  with torch.no_grad(): 
      for imgs, labels in loader:
          imgs = imgs.to(device=device)
          labels = labels.to(device=device)
          outputs = model(imgs)
          _, predicted = torch.max(outputs, dim=1) 
          total += labels.shape[0]  # batch size
          correct += int((predicted == labels).sum())  
  accuracy =  correct / total
  return accuracy


train_acc = accuracy(model, train_loader)
val_acc = accuracy(model, val_loader)
print([train_acc, val_acc])

# + colab={"base_uri": "https://localhost:8080/"} id="j3ldjlvGuR2h" outputId="0e3e348e-7027-495a-9548-d9823bfd919a"
# Apply the model to a minibatch
set_seed(0)
dataiter = iter(train_loader)
img_batch, label_batch = dataiter.next()
print(img_batch.shape)
img_batch = img_batch.to(device=device)
outputs = model(img_batch)
_, predicted = torch.max(outputs, dim=1) 


# + colab={"base_uri": "https://localhost:8080/", "height": 388} id="WlTmlw2tuonS" outputId="a4439d41-e160-4719-a123-fd973ace2b96"
def imshow(img, ax):
  #img = img / 2 + 0.5     # unnormalize from -1..1 to 0..1
  npimg = img.cpu().numpy()
  ax.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def plot_results(images, labels, pred_labels, nrows, ncols):
  fig, axs = plt.subplots(nrows, ncols) 
  k = 0
  for i in range(nrows):
    for j in range(ncols):
      ax = axs[i,j]
      imshow(images[k], ax)
      ttl = f'{class_names[labels[k]]}, est {class_names[pred_labels[k]]}'
      ax.set_title(ttl, fontsize=8)
      k += 1

plot_results(img_batch, label_batch, predicted, 2,4)

# + [markdown] id="CNy1PDlropBX"
# # Save/load model

# + id="NfkpyYR3op6H"
out = model(img_batch)
fname = os.path.join(data_path, 'birds_vs_airplanes.pt')
torch.save(model.state_dict(), fname)




# + id="6NONMXTDv_6I"

loaded_model = Net(nclasses, img_batch[0]).to(device=device)
loaded_model.load_state_dict(torch.load(fname, map_location=device))
out2 = loaded_model(img_batch)
torch.testing.assert_allclose(out, out2)

# + [markdown] id="-BVLFB4V6JNn"
# # Dropout
#
# We can use dropout as a form of regularization. Let's see how it works for a single convolutional layer. We pass in a single image of size 1x3x32x32 and get back a tensor of size 1x10x16x16, where 10 is the number of filters we choose, and size 16 arises because we use maxpool of 2.

# + colab={"base_uri": "https://localhost:8080/", "height": 267} id="xJHxImyr6Sfv" outputId="20b3410a-bef3-4508-be8d-ed4894ea8eb2"


class NetDropout(nn.Module):
    def __init__(self, nclasses, img, nchans1=10, dropout_prob=0.4):
        super().__init__()
        nchannels, nrows, ncols = img.shape
        self.conv1 = nn.Conv2d(nchannels, nchans1, kernel_size=3, padding=1)
        self.conv1_dropout = nn.Dropout2d(dropout_prob)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = self.conv1_dropout(out)
        return out

set_seed(0)
print(img_t.shape)
img_batch = img_t.to(device=device) # single image
model = NetDropout(nclasses, img_batch[0]).to(device=device)
output = model(img_batch).cpu()
print(output.shape)
print(type(output))
show_results(img_t, output)

# + [markdown] id="MrumaZWx7kiL"
# In training model, the model is stochastic.

# + colab={"base_uri": "https://localhost:8080/"} id="foZcp_iJ7niU" outputId="c6e484c3-c04a-4ab3-a7e1-930810337779"
model.train()
set_seed(0)
out_batch1 = model(img_batch).detach().cpu().numpy()
out_batch2 = model(img_batch).detach().cpu().numpy()
np.allclose(out_batch1, out_batch2)

# + [markdown] id="nw00sPpw72Tl"
# In testing model, the model is deterministic, since dropout is turned off.
# This is controlled by the fact that the dropout layer inherits state from the parent nn.Module.

# + colab={"base_uri": "https://localhost:8080/"} id="mJaqFXEe8ASC" outputId="24e5bb62-5858-47a3-a155-2ce6c8b1e78a"
model.eval()
set_seed(0)
out_batch1 = model(img_batch).detach().cpu().numpy()
out_batch2 = model(img_batch).detach().cpu().numpy()
np.allclose(out_batch1, out_batch2)


# + [markdown] id="uLNTRb_Cxv2o"
# # MNIST
#
# We've written the model definition to work with images with any number of input channels, including just 1 (i.e., gray-scale images). It can also handle any number of output classes. Let's check that it works on 10-class gray-scale MNIST.
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 383, "referenced_widgets": ["d94dc1a1722d412486777816774d2d8d", "23f97b3ca11745ca86da99644f232422", "72b70df3a70a4eb99b8955a88d3d23d5", "55d3805aa7be4c7aabf3898838d8d02c", "97f44fecf5e14b9fbb240e0ebec6a64b", "9b810b90351e463ab05c79c8809a16d9", "0a4982b2eb73461cace57a6500dcafe7", "f5f235e3053642a1bb5d49d5a7f454b8", "f1a5938297014f72a35e375c543f2778", "ddfab70242d84a39ada141ad6435fe61", "0689fc9445db4711883a2d7318d927fb", "22605743ca9147ebbe9c670cbada3713", "453bf0a288a9499cb66b7f5d63d18f26", "b986f9012b7643d5b5d2240cbe4cff77", "18de83b1829646d0be71bf68583db6cb", "e30caf368bbf4da7b9feca8ca6fccdf3", "2476160c1873406f82a67c192c647eb5", "e93f832e09824741a34364253377d837", "699136ff86264a5db465cc458d0344c4", "5f46d062c441434882e8ae3e0a35eb63", "c75b8ab87f534424a0df9bc570e723eb", "e3609f57f8cd4609b85699439ae8480d", "5861ef24fd3c40df85d773a816371ff8", "b43a4e7f7c0c43ed89146ab7c3c982cd", "08836a8f047f46f4b8d9e9a2510dab3a", "fdd62f0666c146109182abd9d2cb143a", "b0199da3d6a24649ad4ec3d086c30e87", "5f78b6d2c2184c24bb6d62d486de068c", "6c21485c9daf4ebf98cc591691261715", "66b3cfce0c3b43e595ff5dd9a3773453", "8faa2b14fc4f46789b246c4a6cbc8681", "07dbf541fd2f4719bbcc9e517036634d"]} id="c7NvlSml4AgD" outputId="7d1ead0c-0be1-4efa-f3a0-f4c3a7500ec1"
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from torchvision import datasets, transforms
data_path = 'data'
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST(data_path, train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST(data_path, train=False,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1)
val_loader = torch.utils.data.DataLoader(dataset2)

# + colab={"base_uri": "https://localhost:8080/"} id="aP3sE_6l2dIo" outputId="15c1bf67-0ee9-41be-fa2c-6289267ae72c"
set_seed(0)

# Apply the (randomly initialized) model to a minibatch of size 1
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1)
dataiter = iter(train_loader)
img_batch, label_batch = dataiter.next()
img_batch = img_batch.to(device=device)
print(img_batch.shape)

nclasses = 10
model = Net(nclasses, img_batch[0]).to(device=device)
outputs = model(img_batch)
print(outputs.shape)

# + colab={"base_uri": "https://localhost:8080/"} id="qF5XLH4fCxRV" outputId="61ef2a8e-8c5c-4e2a-d2c0-4e511ef5b665"
image_size = img_batch[0].shape
nchannels, nrows, ncols = image_size
print([nchannels, nrows, ncols])

# + colab={"base_uri": "https://localhost:8080/"} id="T8dWHuSGCGJX" outputId="346638a4-610f-459c-ccff-a50a3530ae5e"
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64,
                                           shuffle=True)  

model = Net(nclasses, img_batch[0], nhidden=20).to(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2)  
loss_fn = nn.CrossEntropyLoss() 

training_loop(  
    n_epochs = 20,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    print_every=1
)

# + colab={"base_uri": "https://localhost:8080/"} id="gQOkNR7XFQFi" outputId="a48d74ee-4c6d-4d5c-9035-248d9c729c31"
val_loader = torch.utils.data.DataLoader(dataset2, batch_size=64,
                                           shuffle=True) 

train_acc = accuracy(model, train_loader)
val_acc = accuracy(model, val_loader)
print([train_acc, val_acc])
