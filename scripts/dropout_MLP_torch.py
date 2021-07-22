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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/dropout_MLP_torch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="1WJeMG1Akwvp"
# # Dropout in an MLP
# Based on sec 4.6 of
# http://d2l.ai/chapter_multilayer-perceptrons/dropout.html
#

# + id="VZyBF4-WksTv"
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=1)
import math

import torch
from torch import nn
from torch.nn import functional as F

# !mkdir figures # for saving plots

# !wget https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l/torch.py -q -O d2l.py
import d2l


# + [markdown] id="VzjmKaBXlcY8"
# # Add dropout layer by hand to an MLP

# + id="CY8jl-m7k_mM"


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()
    return mask * X / (1.0 - dropout)


# + colab={"base_uri": "https://localhost:8080/"} id="u62UJnwPlOj4" outputId="12132161-85ef-4982-9572-81dec6c2e4c1"
# quick test
torch.manual_seed(0)
X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))


# + id="RzNntL4slPud"


#  A common trend is to set a lower dropout probability closer to the input layer
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True, dropout1=0.2, dropout2=0.5):
        super(Net, self).__init__()
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # Use dropout only when training the model
        if self.training == True:
            # Add a dropout layer after the first fully connected layer
            H1 = dropout_layer(H1, self.dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # Add a dropout layer after the second fully connected layer
            H2 = dropout_layer(H2, self.dropout2)
        out = self.lin3(H2)
        return out



# + [markdown] id="9SV2d62Dlr28"
# ## Fit to FashionMNIST
#
# Uses the [d2l.load_data_fashion_mnist](https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py#L200) function.

# + colab={"base_uri": "https://localhost:8080/", "height": 471, "referenced_widgets": ["ef31731c8c3341e095f8481db32268e9", "ed68350fce8246c287f5aa93814aa14b", "67fccd492d944dd3a723116b238681c1", "7afc34c5e582451f94a03eee61b01582", "184401772a414e5eae5d6801624f4e98", "0a3298fe24de4f39a44aa88a621173b4", "e9ed770bc63a4a81b96766a8e9a016f8", "51498b301bf14fd3b5528f0eb58e19c8", "a51380f3658440818f3ad7c0135e2b90", "d074c256b9154cd48f36f7d156ac01dd", "e9b3a0832aaa406cbfb94dbaed15d9af", "ae79ba6852ba45deb38c4e753373c6f0", "92939089b55f418ba8c0c35fac364011", "a236377ba48f49a69425708288c1cd41", "ca4ffbd43667438d93d09520b1892d1c", "c071447004d3466db1d6c471e789acf0", "840cd3da067a4842b7ce3c5ef0b39520", "1356126b6b4c4bd6b68a8e2cbee9ca4a", "d945a599158847b3b89a3840eea62d6c", "bae79b1fec4e4aa1b114353f8bb821a8", "b0db1b443e504a62b1720ec317bde753", "fe7b59bc61bc401183ed143ae293d791", "3e7d7a5cfa0147f1853a0c6421f08257", "6a9336f89c6e42dca65603a202e9ec2b", "6184f2aa3cc84abe8d8bf4307037d9ba", "46bb91ac8ce44d328ea7de1f27c9e5d0", "c78c24865191425faea1923118f85e17", "271cb90726df4bb5b325ae606f3d53c9", "acf6d7de62754234b5edd131bae214db", "3363766204fe414dbe40f3f1f6b41da1", "abf7bbeecd9f4ac4b7738687ae6a8828", "3176f07b82a142b48139d65d32fc08e6"]} id="K2Igif6rl3ci" outputId="11eb8519-9d48-4c97-9bec-05be1040e089"
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=256)

# + [markdown] id="gajH4V3gmLVv"
# Fit model using SGD.
# Uses the [d2l.train_ch3](https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py#L326) function.

# + colab={"base_uri": "https://localhost:8080/", "height": 262} id="MrRopaCFlrJB" outputId="d4391e67-8a76-4ec4-f0d2-eee52be0c973"
torch.manual_seed(0)
# We pick a wide model to cause overfitting without dropout
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
          dropout1=0.5, dropout2=0.5)
loss = nn.CrossEntropyLoss()
lr = 0.5
trainer = torch.optim.SGD(net.parameters(), lr=lr)
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# + [markdown] id="Vgi3EjMrrc28"
# When we turn dropout off, we notice a slightly larger gap between train and test accuracy.

# + colab={"base_uri": "https://localhost:8080/", "height": 262} id="-wd8I2-pn69g" outputId="bee0b0aa-6bf6-4b19-9d8d-bfead734c96b"
torch.manual_seed(0)
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
          dropout1=0.0, dropout2=0.0)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# + [markdown] id="UA3JYLAYojwU"
# # Dropout using PyTorch layer

# + id="TmGh7Tf0olsK"
dropout1 = 0.5
dropout2 = 0.5
net = nn.Sequential(
    nn.Flatten(), nn.Linear(num_inputs, num_hiddens1), nn.ReLU(),
    # Add a dropout layer after the first fully connected layer
    nn.Dropout(dropout1), nn.Linear(num_hiddens2, num_hiddens1), nn.ReLU(),
    # Add a dropout layer after the second fully connected layer
    nn.Dropout(dropout2), nn.Linear(num_hiddens2, num_outputs))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

torch.manual_seed(0)
net.apply(init_weights);

# + colab={"base_uri": "https://localhost:8080/", "height": 262} id="YYha_QFlorJr" outputId="e95dc80b-80c1-4692-a489-fcec8f0fdc84"
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


# + [markdown] id="w4_bZQYLs3G7"
# # Visualize some predictions

# + id="2kOtZU7xtErO"
def display_predictions(net, test_iter, n=6):
    # Extract first batch from iterator
    for X, y in test_iter:
        break
    # Get labels
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    # Plot
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(d2l.reshape(X[0:n], (n, 28, 28)), 1, n,
                    titles=titles[0:n])


# + colab={"base_uri": "https://localhost:8080/", "height": 233} id="DE8DD-2Ys42Y" outputId="e0ef855c-bca5-4039-d63e-d18797a9c611"
#d2l.predict_ch3(net, test_iter)
display_predictions(net, test_iter)

# + id="9Tr1jNAIs5Vp"

