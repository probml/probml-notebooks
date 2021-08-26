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
# <a href="https://colab.research.google.com/github/always-newbie161/pyprobml/blob/hermissue185/notebooks/clip_imagenette_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="cRpLbgX759IR"
# This notebook illustrates the following on CLIP extracted features of Imagenette v2 dataset
#
#
# *   Classification using MLP using flax+JAX
# *   Logistic Regression using flax+JAX
# *   Logistic Regression using sklearn
# *   Logistic Regression using pytorch-lightning
#
#
# ---
#
#
# Author : Srikar-Reddy-Jilugu(@always-newbie161)
#
#
#
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="hmwstx0SRbD6" outputId="52b7a657-9bf1-406c-b3fa-693d3def8cc7"
# !nvidia-smi

# + [markdown] id="2VlASTITvnVf"
# ### Required Installations

# + [markdown] id="2PgIgPifJCNc"
# make these dirs just to be consistent with the original repo

# + id="BRO4SSSeCRD0"
# !mkdir data
# !mkdir notebooks

# + colab={"base_uri": "https://localhost:8080/"} id="C6HnKOTkCi3Q" outputId="24eb38af-7f87-43e0-843d-4c59150e69e1"
# cd notebooks

# + [markdown] id="4S2-XMON5t-V"
# Getting the clip_dataloader.py for pyprobml which gets the dataloaders for **CLIP extracted fetures** of imagenette v2 dataset

# + colab={"base_uri": "https://localhost:8080/"} id="sPbuqEalq2k5" outputId="6a996dd2-f28a-46e3-8a4b-cd591d3fe202"
# !wget https://raw.githubusercontent.com/probml/pyprobml/master/scripts/clip_dataloader.py

# + colab={"base_uri": "https://localhost:8080/"} id="gYw6fwZ95Xa8" outputId="7ebfd391-9f38-42ff-cd42-bfad32760abb"
# !pip install flax
# !pip install pytorch-lightning
# !pip install wget

# + [markdown] id="Mm0hdfIE_qeb"
# ### Downloading original Imagenette dataset

# + colab={"base_uri": "https://localhost:8080/", "height": 531, "referenced_widgets": ["c131e3e3ec544adbb633e69d8c2f419b", "4985c9c7623a406d9aab785392302aab", "ef5e23e71968459495fc805c97971bea", "53224f6fc0c2445aac969db4f44c6de5", "75045e0d3b8b44899f8a744e5a22ed72", "6f6e56630b70462a8bff7ee64b9bad9a", "8537d9c57ecb4460a5050d9f85dca94e", "d5af11e794404501b1d3d177d5ef4e94", "88f1f589db5548c592c69e8620348dd5", "1ef4e77d8869407081a4ebd5116af61c", "39c909711d294018964b09ee021d667e", "7a2bb3e43b6c4158981aa002069e8df3", "b47f1414fa014d9495899288d8eff5ee", "446bdaa32b0a4d398fc25d51fd9a7a20", "a360f3e20bbc497c862c5d984ca09523", "03bf43f9eabe4c938affdf2d3d454183", "391bb34fd5774d21acc50626da700304", "b014400669f84d4192acfe325444fd54", "4b4bd859c74f40989e612fd3283157d7", "2da44911dc954bcfa91c7041181b30c1", "577e993a415a46f2be5af6213df1c41c", "d79fae27a8d44cd7b8d039039213c2c5", "ed266e60114a4712be08e43772bd2f01", "99fa3022ca4f4b15abf8e97c708f081e", "97aef278dc52452a91134efa4bd3cd5a", "396b352191374e10b19192153070e8c2", "1abbf3cf49d449d0abffb5f6db815f4d", "5e8514e83653444d96024883c26c620b", "a215a2e7ec144f41a443dee378bc69b4", "6438d884591f480fa4321bad0b3db207", "738d04cd3aa644ad9106dc78e4092b3f", "dd9986385daa4a199632ffb187e0abc9", "820cebc0334f418fb900981f919ba2b0", "de0795e577794a7b82267b9b5b6933b8", "1c308e13df4a4bec9ce3e8859ebf3fa9", "a695824ed6e94a3ab573d645a8137593", "f3b29d68db7e45bab1ac3821e6a3667f", "04e3c2b09e7149cb917c635d4204d1b9", "440fdc133d3f449a988087b205baef38", "51f34083dca0432d9866ad5ba6f0e63a", "855ec80ff10246be86fda848ce55853b", "7f69ac34c8034ffc9391bfe33fc0a5b9", "43dec8ea902748ac9b561ec213a66fab", "679d4d5b24a545b8981ab3078a0c8b40", "1dc151e585ee491da5e15184ad9f6332", "689fb96cc48140fa934036ed1ea4ef8f", "f6f4e48933384d3c99ed9f8693a6e5fe", "53ea77c0555d446395895c3da5d3408d", "02c1f416f7aa488b8a073db917a38b95", "40d8bc9842124246b1b2e665ebccd2f2", "bda382c1dd784cb38cc52dcc27c97112", "237708d49cf64533b267284227e233ef", "4f915025ed8a4293b889c6d9ff6ba5f3", "4c7c8bb4f2ec4317904382263a6a2c65", "790fd453428c434c9a3bf6028319d237", "5b990fe67744407abcee28d2ffa6a4c1", "1f0f8300baa4423fa105187845e0b669", "e7e590ac87394b7ab78bb68c5a269a7a", "71d6456507b249fa80e399c7423fdadd", "15dff46c62fa424c9f3dca420806f778", "f5fb66134efa4c95ab7532a66713b13b", "86faa9666e1d4ed1bb45505a8455c747", "4187071a888e4d54839107eb0e4d0fe7", "0155a9f4004147508f384bde65ffd855", "e67e31c4a7af43ada6b71e6e30d56cb5", "b040d84266cf40c495001ca9b5d1d30a", "b7bd24ab0cef4485b9ffa51465620118", "b2eccbf677e44b16a4eed1376d995ee9", "d8a3e44d7fb840b1b3b60ff9f605351a", "65d26db44c8e4221a46f2a93b7919241", "12da1ddff22141388bf1311f64a53e8e", "9126baa357f6411eb97671800c8668b2", "c4667644328a4dc09c103ad5f1c79c46", "97c190bb65154de19ce25c84d1ae7567", "860359d13e0343fbad20b566ed5a5820", "1175283dc1fa4a389407ca932d4c9024", "9060d1a924ed464e90125465a59ce708", "0967aa8162f44647a1759528499f3e55", "63b965877e18450b97244bc4b47b062e", "25229cde2ced4ba88982ccb7c43e4b0b"]} id="D3fqC8jT_pqH" outputId="0c637ec7-51a1-46cd-ec6f-59e8a65c5f9f"
import tensorflow as tf
import tensorflow_datasets as tfds

try:
  data, info = tfds.load("imagenette/160px-v2", with_info=True, as_supervised=True)
except:
  data, info = tfds.load("imagenette/160px-v2", with_info=True, as_supervised=True)

train_data, test_data = data['train'], data['validation']

# + colab={"base_uri": "https://localhost:8080/"} id="gQbmYDv-kOxb" outputId="9e2d41d9-71c8-406c-ba92-b1927fd23319"
print(info)

# + id="mLQWDJ1RRn0l"
import math
import numpy as np
import matplotlib.pyplot as plt
from clip_dataloader import get_imagenette_clip_loaders

# + [markdown] id="7WLXIBrR2RqL"
# Getting the CLIP-extracted data for the Imagenette dataset.

# + colab={"base_uri": "https://localhost:8080/"} id="QkGSF0ql6TBP" outputId="fa275aa8-cf6f-4051-9bc3-38c97bb8b0b1"
# zip files of the data will be stored in "dir_name" directory.
train_loader, test_loader = get_imagenette_clip_loaders(dir_name='../data')

def convert_dataloader_to_numpy(loader):
    features, labels = [], []

    for batch in loader:
        features.append(batch[0])
        labels.append(batch[1])

    features, labels = np.concatenate(features), np.concatenate(labels)

    return features, labels

train_features, train_labels = convert_dataloader_to_numpy(train_loader)
test_features, test_labels = convert_dataloader_to_numpy(test_loader)

# + [markdown] id="myQ_f6z8q7bI"
# ## Demo of MLP and Logreg using flax+JAX

# + id="60eXi1VS56Jz"
import jax
from typing import Sequence
from jax import random, numpy as jnp
import flax
from flax.core import unfreeze
from flax import linen as nn
from flax.training import train_state
import optax # used for the optimizer
from functools import partial


# + [markdown] id="95EGvSO7qm4k"
# Getting the **CLIP extracted features** data for the Imagenette-160px v2 dataset.

# + [markdown] id="83EcSoTeoLXd"
#  A Simple neural network where all the hidden layers(h) along with the output layer(O) can be specified using
#  
# for MLP:
#
# ```
# model = Simple_nn([h1, h2, h3...,O])
# ```
# for LogReg:
# ```
# model = Simple_nn([O])
# ```
# you may refer to the official docs before proceeding
#
# https://flax.readthedocs.io/en/latest/
#

# + id="9ZANeWXuniOW"
class Simple_nn(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        x = nn.log_softmax(x)
        return x



# + id="BJ5KOWVbiAet"
class Flax_model:

    def __init__(self, layers, train_loader, test_data):
        """
        :type layers: list of no.of.units of hidden_layers and also the output_layer at the end
        :type train_loader: torch dataloader
        :type test_data: tuple of features and label data
        """
        self.train_loader = train_loader
        self.test_features, self.test_labels = test_data
        self.model = Simple_nn(layers)

    '''
    * Loss is used is cross_entropy_loss
    * Optimizer used is SGD with a decay rate(momentum) using optax, 
      the train_state of the Flax_model stores the model params and 
      the state of the model (i.e it stores the updated params and grads while training)
    * Initial params are returned by __init__ method of the nn.module using a random vector of the required features shape
    * Metrics used are __accuracy__ and __loss__
    '''

    def get_initial_params(self, key):
        # initiating the model with random input to kick-off the params.
        kx, kinit = random.split(key, 2)
        x = random.normal(kx, (1, self.test_features.shape[1]))  # shape of a sample of feature data.
        initial_params = self.model.init(kinit, x)['params']
        print('initialized parameter shapes:\n', jax.tree_map(jnp.shape, unfreeze(initial_params)))
        return initial_params

    def __create_train_state(self, init_params, tx):
        """
        tx : optax optmizer
        """
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=init_params,
            tx=tx
        )
    
    '''
    The bottleneck of flax model are the 
      train_step
      eval_step(validation_step)
      computing the metrics (loss and accuracy)
    
    Jitting the corresponding function speeds up the training.
    '''

    # JIT compiled loss and metric functions
    # model(general class) can't be jit complied, so it is passed as a static arg to the jitted funcitons.

    @partial(jax.jit, static_argnums=(0,))
    def __cross_entropy_loss(self, logprobs, labels):
        one_hot_labels = jax.nn.one_hot(labels, num_classes=logprobs.shape[1])
        return -jnp.mean(jnp.sum(one_hot_labels * logprobs, axis=-1))

    @partial(jax.jit, static_argnums=(0,))
    def compute_metrics(self, logprobs, labels):
        loss = self.__cross_entropy_loss(logprobs, labels)
        accuracy = jnp.mean(jnp.argmax(logprobs, -1) == labels)
        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }
        return metrics

    # JIT compiled train and evaluation steps

    @partial(jax.jit, static_argnums=(0,))
    def __train_step(self, state, features, labels):
        def loss_fn(params):
            logprobs = self.model.apply({'params': params}, features)
            loss = self.__cross_entropy_loss(logprobs, labels)
            return loss, logprobs

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, logprobs), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = self.compute_metrics(logprobs, labels)
        return state, metrics

    @partial(jax.jit, static_argnums=(0,))
    def __eval_step(self, params, features, labels):
        logprobs = self.model.apply({'params': params}, features)
        return self.compute_metrics(logprobs, labels)


    '''
    jax.device_get() is used to get the arrays from the Shared Device arrays from the device used.
    jax.tree_map() returns a pytree after doing respective task on the given pytree. 
    Refer JAX Docs for more info
    https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html

    '''

    # train routine for each epoch.
    def __train_epoch(self, state, epoch):
        batch_metrics = []
        for batch in self.train_loader:
            features, labels = batch  # The returned batch are torch tensors.
            features = features.to('cpu').numpy()
            labels = labels.to('cpu').numpy()

            state, metrics = self.__train_step(state, features, labels)
            batch_metrics.append(metrics)

        training_batch_metrics = jax.device_get(batch_metrics)
        training_epoch_metrics = {
            k: np.mean([metrics[k] for metrics in training_batch_metrics])
            for k in training_batch_metrics[0]}

        print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (
            epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))

        return state, training_epoch_metrics


    # Evaluate the model on the test set
    def eval_model(self, params, test_features, test_labels):
        metrics = self.__eval_step(params, test_features, test_labels)
        metrics = jax.device_get(metrics)
        eval_summary = jax.tree_map(lambda x: x.item(), metrics)
        return eval_summary['loss'], eval_summary['accuracy']

    def predict_logproba(self, params, test_features):
        logprobs = self.model.apply({'params': params}, test_features)
        # logprobs returned are log-softmax values.
        return np.array(logprobs)

    def get_params(self, state):
        return state.params

    # runs the flax model and returns the log_softmax values

    def run_flax_demo(self, key, learning_rate=0.1, decay_rate=0.9, num_epochs=10):

        init_rng = key

        init_params = self.get_initial_params(init_rng)

        # Momentum optimizer.
        tx = optax.chain(
            optax.trace(decay=decay_rate, nesterov=False),
            optax.scale(-learning_rate),
        )

        state = self.__create_train_state(init_params, tx)

        for epoch in range(1, num_epochs + 1):
            # Run an optimization step over a training batch
            state, train_metrics = self.__train_epoch(state, epoch)
            # Evaluate on the test set after each training epoch
            test_loss, test_accuracy = self.eval_model(self.get_params(state), self.test_features, self.test_labels)
            print('Testing - epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))

        updated_model_params = self.get_params(state)

        test_pred_logproba_labels = self.predict_logproba(updated_model_params, self.test_features)

        return test_pred_logproba_labels, state.params

# + [markdown] id="1nn2VZ1EVRlN"
# ### MLP 

# + colab={"base_uri": "https://localhost:8080/"} id="MZt7iZDZiz3Y" outputId="b8bcb8b3-1664-43ac-c076-b4bebe837bd3"
key = jax.random.PRNGKey(0)
# specify each layer after input layer with the corresponding no.of.units
layers = [32,10] 

flax_model_mlp = Flax_model(layers, train_loader, (test_features, test_labels))

flags = dict(learning_rate=0.1, decay_rate=0.9, num_epochs=20)
test_preds_logproba_flax_mlp, flax_mlp_params = flax_model_mlp.run_flax_demo(key, **flags)

# + colab={"base_uri": "https://localhost:8080/"} id="CnVK7fU0KeGa" outputId="05a640d4-abac-48d2-c50c-0b3c7c96c348"
print(type(test_preds_logproba_flax_mlp))
print(test_preds_logproba_flax_mlp.shape)

# + id="CnIdeTDwiqN0"
test_preds_flax_mlp = np.argmax(test_preds_logproba_flax_mlp, axis=-1)

# + [markdown] id="Hqb6zDJNWZdP"
# ### Logreg

# + colab={"base_uri": "https://localhost:8080/"} id="aXXtQX0AWe2o" outputId="3e2f8370-f7d8-4683-a962-81c67ef45e47"
key = jax.random.PRNGKey(0)
# As it is logistic regression, there are no hidden-layers, only the output layer is specified.
layers = [10]

flax_model_logreg = Flax_model(layers, train_loader, (test_features, test_labels))

flags = dict(learning_rate=0.1, decay_rate=0.9, num_epochs=40)

test_preds_logproba_flax_logreg, flax_logreg_params = flax_model_logreg.run_flax_demo(key, **flags)

# + colab={"base_uri": "https://localhost:8080/"} id="PykpVFRdG3m_" outputId="91f09a36-b80b-44a2-ff6a-cfaf1c617d8f"
print(type(test_preds_logproba_flax_logreg))
print(test_preds_logproba_flax_logreg.shape)

# + id="HcpuQ7MeqSOd"
test_preds_flax_logreg = np.argmax(test_preds_logproba_flax_logreg, axis=-1)

# + [markdown] id="phnS9UGPW4o5"
# ## Logreg using sklearn

# + id="_i_iZIxzD61-"
from sklearn.linear_model import LogisticRegression


# + id="7PY9-3J2W9Rt"
def run_sklearn():

    clf = LogisticRegression(penalty='none', solver='saga', max_iter=1000, tol=1e-3, verbose=1, multi_class='multinomial', n_jobs=4)

    print('Training sklearn model...')
    clf.fit(train_features, train_labels)
    train_preds_skl = clf.predict(train_features)
    test_preds_skl = clf.predict(test_features)

    test_accuracy = np.mean((test_labels == test_preds_skl).astype(np.float64)) * 100.
    train_accuracy = np.mean((train_labels == train_preds_skl).astype(np.float64)) * 100.
    
    print(f"skl_train_accuracy: {train_accuracy:.3f}, skl_test_accuracy: {test_accuracy:.3f}")

    return clf.predict_proba(test_features)


# + colab={"base_uri": "https://localhost:8080/"} id="jsnecJN7XJIg" outputId="6087f279-1b3a-4bde-baec-5f12e24fd970"
test_preds_proba_skl = run_sklearn()

# + id="U3Qjv0gCtfgr"
test_preds_skl = np.argmax(test_preds_proba_skl, axis=-1)

# + [markdown] id="sbcfPkxjXMM3"
# ## Logreg using Pytorch-Lightning

# + colab={"base_uri": "https://localhost:8080/"} id="ES-6MNwzEfXH" outputId="0942bf6a-5946-4344-a99b-92266c642fb9"
import pytorch_lightning as pl
import torch
import torch.nn as t_nn
pl.seed_everything(0, workers=True)

# + [markdown] id="3Y3OgWcswhKc"
# ### pt_Lightning Model

# + id="dDmtdjtRXiAp"
num_classes = 10
learning_rate = 0.1
decay_rate = 0.9
epochs = 40
n_input_features = train_features[0].shape[0]


# + id="JYe_JLdtv3mB"
class Lit_model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.network = t_nn.Sequential(
            t_nn.Linear(n_input_features, num_classes),
        )

        self.criterion = t_nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=decay_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        out = self.network(x)
        loss = self.criterion(out, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        out = self.network(x)
        criterion = t_nn.CrossEntropyLoss()
        loss = criterion(out, y)
        self.log('val_loss', loss, prog_bar=True)


def lit_predict_proba(model, x):
    x = x.view(x.size(0), -1)
    out = model(x)
    proba = t_nn.Softmax(dim=-1)(out)
    return proba



# + [markdown] id="5AjhC1UW016d"
# ### pt_Lightning_Trainer
# args can be changed accoding to you machine by altering no of GPUs, or using TPUs
#
# (See pl.Trainer doc for all possible arguments)

# + id="xvAhlq1zv7zW"
def run_pt_lightning():
    print('Training pytorch_lightning model...')
    model = Lit_model()

    # training (can be interrupted safely by ctrl+C or "Interrupt execution" on colab)
    trainer = pl.Trainer(gpus=1, max_epochs= epochs, deterministic=True, auto_select_gpus=True)
    trainer.fit(model, train_loader, test_loader)

    model.freeze()
    train_preds_proba_lit = lit_predict_proba(model, torch.Tensor(train_features))
    test_preds_proba_lit = lit_predict_proba(model, torch.Tensor(test_features))

    train_preds_lit = torch.argmax(train_preds_proba_lit, dim=-1)
    test_preds_lit = torch.argmax(test_preds_proba_lit, dim=-1)

    train_accuracy = np.mean((train_labels == train_preds_lit.numpy())) * 100.
    test_accuracy = np.mean((test_labels == test_preds_lit.numpy())) * 100.
    print(f"lit_train_accuracy: {train_accuracy:.3f}, lit_test_accuracy: {test_accuracy:.3f}")

    return test_preds_proba_lit.numpy()



# + colab={"base_uri": "https://localhost:8080/", "height": 458, "referenced_widgets": ["8820ef1c3c90463196bed28c6710cf76", "5c3ea555be0b406e9b402b61bcff3632", "866ea74e5c604987b32b0e482ecfa153", "7ec56b277cd34358bb884bce78185728", "95d2b22a22a84f728750bebb6cb16509", "36183f27dbd747019bc6c6895059deca", "3bdb542f64e84fd5a24baae7b941d8ee", "1d682dd4800a42bc833c265fc19b48c1", "9c567f510c8c403780ad27d724a7bed4", "92b7510fee954438b649f1db51243ca0", "b94217843cec4650a323598545989585", "587c7ca8401c43b9b6384edbf08bf5f2", "ee3280c992944d7697cc539f20fa27fd", "7df25ffe756146e8830cb7614cfdab37", "bc152d1127d049a7a60a02aa0200c1f6", "fc40ed272ef940538bf67cced32b51a0", "4adf761a9ab1432eae582003a4bd2969", "10b944186fe44b3196685ad58c795d62", "17f48a3a90d8497cb9fb17de3ede6ab9", "81514a41d92345b29473a5110c5d4f82", "e538057aeaa6423aa372601ef2c735d2", "e977e76b6ea54ff79ca81b2e1a178cda", "ffc289e01b1d41da8105a733b2cb0808", "46f8fb256b8c4ce986b53cd0e2f732a3", "f8e86cfeb9aa4548bc3c27702852ab7c", "21235db21eac47b687bedde96164b5c8", "0d44db41d02447d39474b559b2bb187e", "0a648a97d40b4e21bacacbda668fb425", "93a45c59325045aebee20a027106f0af", "ece6fd52d4dd409ab18e9dad320b2b41", "e8ce9bd015dc4139b73fa7c8db68572f", "48583404c4c74c25a9f628f613d10223", "c80402bbe424454db39063262da1ea9f", "4408420e53834d5085704a04d7f9326d", "8e6627b6f8a64a2484e038f2c36f79b3", "2863c44561dd4d77be7c2f4aeaf7abc3", "7c4378438902400996f1e0d3e11d67b6", "e96ab7bd90b74d0589b077cac02faaaf", "40f5544def0548f4a292f1d3c1894cdc", "bf774ba5f3f74a98812aa8c405e81720", "ee569f26c84c44909f6d23aa2e2b701e", "0851aa32f0f44e0699b89ad8a1ca0f32", "3f06d4a016d445bb80f9f54c676fd280", "9e9cd4b9a638491dad8b46ef7747687a", "ba479fdf174f4e7da59133005681d19d", "42f4542717fa422d8f8decccf43cb8f5", "0401fa79eec444ccb6394572fddb7759", "c6e6c1328f554344b5952e70372568f4", "f9bf26bda8304dabaff3e8c6d510115c", "94f95d8dd6e2473aaa2aec3894954b36", "289a5c9010864ba3a98786adb853a978", "b4d3cc8cdf1740d1baa94a0a313ef535", "ad61ba7aa9ca444b8f5e231a2ae528b7", "faee8b189578411c98e43ac8add9eaee", "37d028d3442048c9aa879e4f142e6238", "2e6ea1be66864206ab20ad8215a78dbf", "981a27d5ac4e41158890ac7640b32652", "08af36b17aa6454c82b5b4dfb1bea651", "d78fd6ba2ea048de8c5a705fca4299bd", "40de29a2ec5540c6903d4e7cc3dc9014", "61cf3c492db24b5fa10205645eacc1dc", "f68770204b3742b2ad8f4ed934db4b11", "b875999c40a14350b3ec4ad676e2cdaf", "1fc53473d4224c4d99591fb2ab270595", "7113f60c4b3045ca862865ce9b3c3699", "53ed580a31df4aae92bb8f39cea0054d", "d1fe8852ae7f49c4b48a3f81e43ecce0", "ff600323083349fdb7e5064c8372a20c", "e140af2cda3141138c40350f8afcc792", "2a4ce4b61820446e994d41c22eea121b", "786f46ecb86a45fca91c8aedaf0e2086", "e8c53f82a223487f8f3a7c832a180f98", "de30e02b35144929853416064bcb16f1", "2dbdba871dbf43ea80c4cc1b091f8112", "f01cbc1cafc1456c86ead2d12498b15c", "a22fd5d57caf411db83f3010bb351f72", "53d5839bf9d64b03ba6b0c3d246f382b", "5674214edeed456a87f1e8e54455858f", "2c900284fd3745f6b30e008721342089", "2452d2a536634b8e993867adca4747af", "edc60e62549d4b8aa699fda11b4ff1cf", "e5cb7f9a2276466e88ecc699bb149d0b", "9b47f1dfad2842f8a2afaf5b79957e3c", "8d2c868df9b1480ba5ca60d06e8f4970", "002ade5337c540d79c57990cd9f0a6f1", "532b941e51d34c05b50971dedf042ad4", "46d1493dab3f419d9919a539c9802316", "f8ee820cc9a846119a52aa37a11ee205", "453c02bf7e4e4105b02aa02e34ffd206", "7b175951a84043548b9e05e195837f3c", "e7301ffabbdd4fb1b241cc7db8d636d9", "9f77621ae8e2420cb37e625ad0467565", "83ec83f01b4d422d946f5adbb3896414", "c6bbfb19a1e24edea12a08c4b9d95989", "234300319ba545ae87642b09982da8d2", "0084ba66782d4a4eb3d3872e26ad0886", "52eded73e5ac4ca28f969dbfe34246cd", "87fe2ae337c54948b057bc6196e1f526", "ce67f8101b9943a9a09abf282aadbd3d", "5d94069a31184a5f9ff79e638c0fb8f0", "10d873f75374467e8667e9f17956eff0", "468e99f8fa364d288dc4a65a63fe8fc6", "105cd7a607d84fc58e9618b49f48b38b", "fdc5e1537746450e9e2458d75ed1b506", "250953c4f5fd4fdaacb189f3e75ece94", "38d79c6533ab42fa9cf2f769d778d69e", "74c64ef6bc864e34a5a2bf330d9d441b", "69c1bc13cf554ba39d3342672a473aca", "c2d7dc46b90343b092e35ab91f74d4cf", "e74edfd22b1a4660b846319bb3ab2591", "2afa9c84c07f48c582abf7051513c4d7", "2c96aeb72cf5427dbcb9e4529dc208e0", "4c72f0f29b5f47ebb530bc0444859232", "f1cf3e73b7504803a944471401d19474", "1059687e9b84492a93876cd9ce0400a9", "6b24fde015d747ed96debbc2403a0b54", "26bc2c4eb4274e8aa3f0a4cb8100b6a9", "c107267053b94966bf267285b8611164", "6de40aa12403494d9a79e4115215a9e5", "36e3f1dd951b4c92b80507ca4a3c5399", "39b5b21a504d4efc8c486c1e11020d33", "356afcdc772a4d3f929c92b4aacc348c", "335740d3dc614dea916942027e296b65", "b7fc9180e9874d37855526a1d2cbc61f", "965b9b0d8e5543f696466d3173a0a703", "520bd31cf01f41c59b7609b22fff16fe", "b4f5a59be6e141db9ab6839f25221ae2", "515ae396704a43fc880a16d4144389dd", "1379cc67aa0a4a2eae0a0b243c65b5e6", "08a4ff6be72e477fa14158004e73321f", "e2e6ee7b7e9e4133a976972bb8a849ae", "e266185350224bac93e375e4cb477174", "e56b927a15664568b88596eb1ba3e548", "a88987df0d604d1393ae6040f497bc49", "ba27bae94de04fe4b220b555a61c396a", "ca0f9fddd96c45488a240c97813d703e", "bf8595c1d4a7416a9804bab965e75920", "fab1d96f45344b7683188f26c3a94760", "4202a074efda4b44b5ce0a36e854a0bb", "fa64c261282f49b78a3a68944b85ba15", "05691250566a43338d86d1b2b8139cb8", "0d31f2cd73c04621a93a32f98549a378", "23e4cec342c048ea97e3c3b64f4baa1f", "3a2f0ba39178436c9e7f5111fc233931", "74b053912f224ccb8c1a2a2106dea0a8", "eab3511cc5514b1b8dfda7f1212a3f85", "a400f915fc77412ba9389b8eccc1a9bb", "881e89727eed437d89dc58fefb42c7a4", "7b9e513c83e24d69a6e1e2e6a9c4d996", "6196552f119e46e893887165725465f3", "2d9ebacc1e6f40399e7fa3790be22085", "27cfa94379754b87b85c74f874e4cd3a", "997919a2e7524cb7b9e97384e0b9384a", "b96c2b8aaf9e47bd9e05b7400f173262", "11dbd4a41d3b4f4daec69a5b6c75f0b7", "d6cf4e4950294880bf3f470085b4a3e3", "317a575aa9434da5adfaaa4ab6e9fffc", "c11a233e091f485fbe7f059a1930ac90", "790f0f985df84485ad673b2abd809fbb", "daaa136d502a41528a7fc4d51c6a966a", "f3ef32e8d5d44d3ea6d65679fc304b87", "6377caff78ba4058af6882861af58d48", "551354976dfb4c30843c8c2eeb1c5660", "a4fe7824494349969717dc4c19eb154f", "1d06dd8b99c84ab8b285ed45c8e0b13c", "4bd691949f464615bee5f53fff790384", "c7b52c52923740389a2c18f38f6ac902", "84911186e5784ab3b5663775cbacc1bb", "cfae19a0b94a4f69be224d5ec56b0440", "5d43f920ceaa47389426df034db9ed7c", "987bd885423e458d8c73de3d45086294", "d2e28198c1ae4b93b9f58ad5a23112b8", "def14bc07dab4a3fb3351e087114d7cd", "daf94f5c6caf4a96961f156ae0631eae", "452f1d6404534095bae1cd17242ff4ec", "bc49aee3549b449b87ed30a7f4f25849", "780c7c6d5deb468798936052e9116b9a", "b372a40a95bb431e890e17e799149b19", "07fcf75b2d704e1ca5cbb484e8ef6d50", "632c21d29b9a4cb591bdac192be97cc1", "11fbd759be0a4e87898d26b23cb71afb", "a924411596654c3db54acdf6f2e0ab0a", "27d6adf9f74642599177b3ee3f2044cf", "cc7f86536e6c48148de6c7482416905a", "84c8374d19824581ab013380606afc2a", "68dbea0e61324b87bb27e78125aaed5f", "fe423e08f33145a7ae3f91f3006dca0b", "5d945cd02243498385a37470e4ba5e71", "20cf62da10bf46d683c5718d07717499", "d49ed5d72ed146e8b8fd88014f261b1a", "c74156ea98114d66abe64542c04d0bf7", "68ed6421001949598188d491621d2054", "9170a51c5128438a8c4450445f816575", "5f6dc3d13e6645cf9834f4daa825a840", "5267a77540904b48b2cea81d6b3f5573", "00a0e9aa2e7c4f65986279be35640013", "0c3f6274cb714d5998e45c32ffbc7570", "4540455307984e69af6fcdf7f9ade148", "0e60b6a51e984618abfc51ae0608803f", "56d8cf4a7d3a42f98f0efa9772255f95", "f3f9bc92d19b4d448b1c03176a2cb259", "12f1f7bad29c43bfaa767d4c3bf7428e", "3c460f942b84431ba0f604ea21cd4d60", "d9a4281a409943d5aaef131da3728c41", "f0aaed5bc41c4d3f89adfb3cd3a14fc7", "2981c9f2f13c47ba8c38ece5325e0326", "aaa36bd5fabc4e2db16eacfff0cc9117", "231809c0de52455ab0441b7b220722fd", "f06e780e7bf8487b9a6e668c6dd2ffc9", "180b95f709f94b40ad049433590995df", "07e6988946384d23bff18f47fdce3ef0", "e8c6639743fe4999874fd4b7ad7a6e5f", "4093a98c63d54fcd85cc99b0830a4c39", "bf74954323ec42f39934a9ea8294b560", "c5c8d66bbe5048eb9093559e790edec8", "d0bec02d2886449b9907b37c08475b4c", "be23590bb9694f9d8979a2986bb54061", "ef33e7df440b4092a3f1cea1c76e054d", "dc032231b0204b03bc6fca93d62b6d4d", "1aa8279a4a0d435f9a96b656b6c6609a", "18398c8e9568430299c18f2f027e8ea0", "25a2fab81bb0473f80519de370deb338", "a84dd71232d94f629c2e901391659538", "2526a92de859499daf58e605eeba0684", "ab724ecd013941e3ae6a18553fd7e799", "b1f0979663f341789815eff269ba0960", "006c84f3691647fab1159096cbc6c082", "86f1eb6bfac546a891085ed393ba68b2", "9b2225e5342b4c62b7a89894655b7ecd", "cc5b84d18aa64eb69c4e80c124d7064f", "45658af330824bd5a6b8eace98ad95d3", "78f0add20f2f4c7ca0084dfda528f673", "ffab6e378a6f479581a960f22d426efb", "5e211f4a27904072b680ae3a49722cbe", "badc537409744100b0731eecce4d1d51", "ad3eafc1a4eb48e9af9b9ea1672e3c15", "d7ab91f042ca414596f30ab70d0f699f", "b67973b3a6dc46d78e32426457f673f1", "45e91ae941d741288d92bd48834b7cb2", "0fc16aa4757946e894f8773179c09251", "aaa31120d64e4c7bb6530fdcc8a85ce6", "246b145cc23c471abdc6c1da21c08983", "8f4b0987fea24388b98026e68efc1395", "a9b8699d6f0045c7922028d77921a0ff", "20d98f55728a4eca8a30e02fbb85a483", "c86b73d7fb1f47cf8c162f256b50482d", "5390f2d4a6344c38ba757b7d2d373971", "45a4ffee19134c409912ca34c17634f6", "b666ba9fb6ed41ec98328da177fe4d84", "6c79f5d67e0a4bec94558c5125588af5", "9958eb8a4e57493d94a52b5a2cbd0781", "ecd1e2de5a18432dad290ab67e0eafa8", "a9a2864cbe5645c1917d20c1e5f9ff46", "d8f6353167234c07addb696db906f45c", "e0e5f5e41d5841079ace54846117f077", "508e531d3e7048278d7f87e6be75a73b", "1025ea199ce74c8eb370f2200ad42d7e", "bf6a4023d375490d85fa0d0a4337a673", "27b8f976f10b44849933e447c403c582", "116c6fc35e26413fb6c64f65aec7ba0c", "2a031544d3a2475ea737cb19b7226a90", "f71b8022ae3e4e9fb4abca6888bcf091", "03c7197507a7408993b4bbb3951188d6", "b75948942d05444e913cce32da5b7483", "a5bde91eb5024ec3b14729a2160b42dd", "3dc750e70dc4406885a95b23bf3a3f09", "d5dcccbdd60a4db887c71ac5f42c11d8", "f065f01a73d34c3f9bb555f56e53e8a7", "853cf87d5bb1468cbe698a437bf0457d", "0abb6a6d6f61491ea6bb60e91632308a", "d661191b678a43498988944761b064a8", "9ff15aa770d54be093c5275eaf71c5a4", "ccb64f0da34a42ae93e0a725df6f9a87", "257ab8904c01461cb775afb952258a28", "c476910eb48e4af0a4c254ed4fb45c3e", "45cc0089ec8242338bf639f2eeaf9069", "efb2e24c81f94448928e3d5171c7e4e6", "736d1ba3e80a433196c94b200db5fe75", "5faf897347dc474fab43fa024bbea3ca", "c5d8976388424f96bbc0ff2f9c13b836", "a9978d9601284c908438f0adf0ebc884", "7bfb72403700461b9c42f1efaf9874b4", "99dd6017fd514dfa9b1c28c820c1923d", "a32ce74cae454088ab85bbf7d39b1853", "d7c6a42e69d74e6099874772c2865f9c", "4cffdbe447454d669c590b2979edb494", "f877130c06774d8f8ef648578d29b637", "a127d29c88f44ac68d0bab2fe318d24b", "db797e1d23fd463ba0eddd23f68223f9", "149125dff9274387bb7fe90112a82a39", "68e93e64467a44e0979b539bb6d9ea21", "598a023392bc4be3aed9eadf3fcd4db9", "c3bfc41bc45645b7a8c8adf54bd36b9e", "67c9537a7c154dad9310de05036981e8", "ae40b35bd55f429c965113f25afbbcad", "4a0ac270d0ea433b8f3a06b4ddc81d50", "3898a4ab2a8a4551a0605839d6aca6ee", "ef8ee673d6b54e3f881e94e68ded46f5", "113937ed6b854b8c810dd097e590e4b6", "1afb2dd5c9544bda9d9284ae88787fee", "4f9869929753453e8c790634a06fbf58", "93aea90836124e2297b55c68a9aab943", "3308b82aedf54b6ca26e6d43b5105acc", "8d9266017c444bfa9cdfebcd612f8404", "64da8c7b24a4491f8fed3c5246fec005", "7a954dc31d94483f90cd4608b3168b99", "5e8ceb0d34234e6bb859ca6e0890f504", "a36b7d982e854ebeb633e2f2bc0be223", "8be81682a9f440d78ad35f4114c0f2bf", "e3e7e6f363f4419caf8712b4415aaab5", "cda3a6c13b384df3a1dc5f7c66105d3d", "42e69c322cf94fae9be43fcdcba73d3d", "a588543da0a34252acf12399d440c81e", "4c475e9594384eb192268d5929d4654c", "54815566f28b4157b8b569db57c96abf", "40661e9237694ee296d01c9b9e5edd55", "b303cba5f2584a6b81f74ade37776b33", "a117cc9e1be44561a8fd7aea4938bd2d", "8aa67e8392dc4e2a804017b6d9e6cd31", "6d04c9e11fb142eba419158cf3791b2e", "38f1f6a9f9d34352826113adea0a6f1b", "3c0529c5f73147feb51cbbe305358706", "2989031f182f4c7eb82f5becba2cea5f", "0298fddd605745a99f45b4843d6acfe6", "4d391dd06bb249a1bb131ae8965f44dc", "ab82d2c4750941a6912dac79f234dea5", "392cd27cb6b045e7a2cfa4ad1a876d56", "b334cc2b9d6c49cdad258b1e7ed6ba42", "c8416557efaa45dda3d5c69c258dc315", "c184c709cdfc42528422111594054657", "30e9ff45b2e849a9856fa3c96fe6e291", "9dc1ccf5d88446da9c53c89da561a9a3", "165614bdad014a9f89ab4432622f1d74", "789fce1a176540d48ceafaab8fa6bb10", "213acc4cb5024c36bc0e6de233ba6106", "6013a7e298fe4fa983d2ef7d292d495c"]} id="E9YHgTTrYYew" outputId="8ff2da1c-cd20-4274-b40f-85fad72bb481"
test_preds_proba_lit = run_pt_lightning()

# + id="7Vj3PYL_u7zk"
test_preds_lit_logreg = np.argmax(test_preds_proba_lit, axis=-1)

# + [markdown] id="ymbRbhNaBRKa"
# ## Post-Training

# + [markdown] id="j6PeIoNJI8DC"
# ### Testing whether the logreg models are similar

# + id="PtSj7wd1HWed"
test_preds_proba_flax_logreg = np.exp(test_preds_logproba_flax_logreg)


# + id="oKwNnWlqIBjV"
def check_ifsimilar_probs(prob1, prob2):
  assert np.allclose(0, np.max(np.abs(prob1-prob2)), atol=0.4,rtol=0.1)


# + colab={"base_uri": "https://localhost:8080/"} id="sV5s85gSGiRM" outputId="8573bc6a-a2a9-4a33-a423-39fad1bd0e3f"
print('Softmax values difference (max for all classes)')


print('\nskl & lit_logreg: ',np.max(np.abs(test_preds_proba_lit - test_preds_proba_skl)))
print('\nskl & flax_logreg: ',np.max(np.abs(test_preds_proba_flax_logreg - test_preds_proba_skl)))
print('\nlit_logreg & flax_logreg: ',np.max(np.abs(test_preds_proba_flax_logreg - test_preds_proba_lit)))

check_ifsimilar_probs(test_preds_proba_lit, test_preds_proba_skl)
check_ifsimilar_probs(test_preds_proba_flax_logreg, test_preds_proba_skl)
check_ifsimilar_probs(test_preds_proba_lit, test_preds_proba_flax_logreg)


# + id="6RykfjulXIKu"
def check_ifsimilar(pred1, pred2):
  pred_diff = np.mean((pred1 != pred2))
  print(f'{pred_diff:0.04}% difference in prediction')
  assert np.allclose(0, pred_diff, atol=1e-2)


# + [markdown] id="bDYyI9sCGhpm"
# Comparing predictions

# + colab={"base_uri": "https://localhost:8080/"} id="8D1qmLhlJFbr" outputId="76c40800-d9c4-48a0-a6d6-28f2f35d5df4"
check_ifsimilar(test_preds_skl, test_preds_flax_logreg)
check_ifsimilar(test_preds_lit_logreg, test_preds_flax_logreg)

# + [markdown] id="6I_Rk59wBUEW"
# ### Visualizing test Results

# + [markdown] id="WBn_8fk_nviL"
# ### Actual class names for the imagenette v2 dataset

# + colab={"base_uri": "https://localhost:8080/"} id="dxM71npgy3T3" outputId="ff12dd42-441a-4713-e4f5-f29d6acd8546"
# !wget https://raw.githubusercontent.com/probml/probml-data/main/data/imagenette_class_names.csv

# + colab={"base_uri": "https://localhost:8080/"} id="ZF9dv6dsz4ih" outputId="f5ad44a8-7670-49d7-803b-bb03c5520734"
import csv
csv_reader = csv.reader(open('imagenette_class_names.csv')) 
next(csv_reader) # leaving the first row
class_names = {int(k):v for k,v in csv_reader}
print(class_names)


# + [markdown] id="LcK0_bxPCc3I"
# ### First N test images

# + id="GjCF5gPDjtKA"
def plot_first_n_results(test_pred_labels, n_samples=20, name=''):

  first_n_test_samples = list(test_data.take(n_samples).as_numpy_iterator())

  fig = plt.figure(figsize=(15,15))

  rows = math.floor(math.sqrt(n_samples))
  cols = n_samples//rows

  for i in range(rows*cols):
    fig.add_subplot(rows,cols,i+1)
    image, true_label = first_n_test_samples[i]
    pred_label = test_pred_labels[i]
    plt.imshow(image)
    plt.title(f'true: {class_names[true_label]} \npred:{class_names[pred_label]}',fontsize = 10)
    plt.axis('off')
  
  plt.suptitle(name, fontsize='x-large')
  plt.show()



# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="L96_IR04uqtK" outputId="0aabd3e1-e782-423a-9335-6d744903d56f"
plot_first_n_results(test_preds_flax_logreg, name='flax_logreg')
plot_first_n_results(test_preds_flax_mlp, name='flax_mlp')
plot_first_n_results(test_preds_skl, name='sklearn_logreg')
plot_first_n_results(test_preds_lit_logreg, name='pt-lit_logreg')


# + [markdown] id="97I4f75YM8GW"
# ### First N Missclassified test Images

# + id="rmoGHcnwlQSs"
def plot_first_n_mismatched(test_pred_labels, n_samples=20, name=''):

  # Making a dataset which contains **misclassified** samples along with predicted labels
  
  test_pred_labels_ds = tf.data.Dataset.from_tensor_slices(test_pred_labels)
  zipped_ds = tf.data.Dataset.zip((test_data, test_pred_labels_ds))

  def get_mismatched(ds):
    return ds.filter(lambda sample, pred_label: tf.math.not_equal(x=pred_label, y = sample[1]))

  mismatched_ds = zipped_ds.apply(get_mismatched)


  first_n_mismatched_ds = mismatched_ds.take(n_samples)

  first_n_mismatched_data = list(first_n_mismatched_ds.as_numpy_iterator())

  fig = plt.figure(figsize=(17,17))

  rows = math.floor(math.sqrt(n_samples))
  cols = n_samples//rows

  for i in range(rows*cols):
    fig.add_subplot(rows,cols,i+1)
    (image, true_label), pred_label = first_n_mismatched_data[i]
    plt.imshow(image)
    plt.title(f'true: {class_names[true_label]} \npred:{class_names[pred_label]}',fontsize = 10)
    plt.axis('off')
  plt.suptitle(f'{name}_mismatched', fontsize='x-large')
  plt.show()



# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="CDCvR_YkwE-g" outputId="ec2d29c5-c8be-43f0-d4e5-00e37fee190a"
plot_first_n_mismatched(test_preds_flax_logreg, name='flax_logreg')
plot_first_n_mismatched(test_preds_flax_mlp, name='flax_mlp')
plot_first_n_mismatched(test_preds_skl, name='sklearn_logreg')
plot_first_n_mismatched(test_preds_flax_logreg, name='pt-lit_logreg')

# + [markdown] id="4i2QpVrCJv0J"
# ### END
