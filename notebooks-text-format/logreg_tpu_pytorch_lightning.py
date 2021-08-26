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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/logreg_tpu_pytorch_lightning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="2V7qvBWQceLw"
# # Logistic regression on MNIST using TPUs and PyTorch Lightning
#
# Code is from 
# https://lightning-bolts.readthedocs.io/en/latest/introduction_guide.html#logistic-regression
#
# Be sure to select Runtime=TPU in the drop-down menu!
#

# + id="8p_a56gHgbgs"
import matplotlib.pyplot as plt
import numpy as np
import torch

# + colab={"base_uri": "https://localhost:8080/"} id="yFlx8FflcXqU" outputId="26c691cc-6f77-4a72-d6b0-413a7997ff79"
# !pip install -q lightning-bolts

# + id="dIDFTVR5ciGj"
from pl_bolts.models.regression import LogisticRegression
import pytorch_lightning as pl

from pl_bolts.datamodules import MNISTDataModule, FashionMNISTDataModule, CIFAR10DataModule, ImagenetDataModule

# + colab={"base_uri": "https://localhost:8080/", "height": 113, "referenced_widgets": ["bf250e8f4bee4f0bb01a3c146ba71bd2", "3215896d20ff4f2ba8f367160206c706", "440c786b21704a4f969c6b107d365ee5", "bb03348cbf3b492f9b866cad0a965c80", "d073132b62734c658fd9efe7d96d37f0", "a36843d218cf42e8a30bf72ee9ff9f6d", "1fdcbd32e9d84f5da6d0e39f84b209f2", "f331510fe636404ba17a82ab89a9c807"]} id="TS946EHNdsVm" outputId="aeaa0573-a13f-4c5a-e0d2-55bd74303684"
# create dataset
#dm = MNISTDataModule(num_workers=0, data_dir='data')
dm = CIFAR10DataModule(num_workers=0, data_dir='data')
dm.prepare_data() # force download now


# + colab={"base_uri": "https://localhost:8080/"} id="kestqpmxeRb8" outputId="762d564b-556a-4b1a-a5b4-60c71fa624cd"
print(dm.size())
print(dm.num_classes)
ndims = np.prod(dm.size())
nclasses = dm.num_classes
print([ndims, nclasses, ndims*nclasses])

model = LogisticRegression(input_dim=ndims, num_classes=nclasses, learning_rate=0.001)
print(model)

# + colab={"base_uri": "https://localhost:8080/", "height": 361, "referenced_widgets": ["505f6359a56e4ae19dceafbaffa068b8", "31c074cc62c548c6a73f37bb4fe8f65d", "3f14ae7a4e67435b84bee01c235ff35b", "9c48b7eb616a4ca4a19103202797673e", "caa30775ad9b435e9f25e83cac47ecd2", "0f29d99232cd40979bd12ae1f5fd216d", "edb93b0af1f04795b3265bcfcea5d0fb", "87589e57407343f8834fcdd4bb4ef5d6", "b97d3fbdabd44d36a912e64b14d9aa0b", "6c7b73e22ccd47d6a6b82418219a53ab", "c1c2bc34f1d04670ac47a62cb663fcaa", "4ded11f8790641edbfb0f2b48b4ea38d", "64ab0bb8573044a39e1040b62f992ef8", "4598de99ccf740cd89421a0b04a6e075", "8679bb17931f424b9be2c2f9a918ed62", "d6970b54bb324a5b80fca4bf0d2e8df0", "02a53561df7f4d9ebc3d3f39fb484347", "d170b2c621394b18af54b2bd430877f5", "0503043a893843aabb54e4e04de40aa9", "f162c4bd4cd448b7a436eda6ed81d47e", "ea18875525034196a9cc4818f423d037", "ad85f159d6de44e380846772195d351b", "e437e282a73d49ce8625c14407249b9d", "37beff2ceb76440faa4f47331605d4f5", "1ee5fae4eeaa41fc89aa99698eed7ace", "84138e982dc54ac6929cf787dd35d858", "01fd7cf38c014237b688a0bdd8fe89fa", "96badf58f59a4b3980bdcfe8b7da26e3", "963903cab7d6488d8f946a0876fc1694", "66323826f1b44e00be46399a96efda0d", "1480123e957e438d846672665f088dce", "25994d31598f4cb39f779fea3334824d"]} id="QwVKlB_qdt6Y" outputId="c673df8f-1dcc-4adf-f5a5-acc2f56492dd"

trainer = pl.Trainer(max_epochs=2)
trainer.fit(model, datamodule=dm)


# + colab={"base_uri": "https://localhost:8080/", "height": 198, "referenced_widgets": ["92692be6a79943a58a24b7cfdfd8a34b", "7b78f0113a46474aa4a4fda9f9c7b4d6", "faa9f5886fbf46098f346d9fc7c10dbe", "edb3d36e81b34cf6b34bd4c14375a83e", "5399dfc259a94c0a81bf494336656a93", "a6de7a9509fe4b7eba7ae6158d983cd9", "eddc730a260b43868ac414641f582501", "83c50baab99040f1b8ada013074e5e35"]} id="bwG_J2YsgiMz" outputId="70b0a4b5-8b6c-4ff2-eb5e-9af1ccf8f1ec"
trainer.test(model)

