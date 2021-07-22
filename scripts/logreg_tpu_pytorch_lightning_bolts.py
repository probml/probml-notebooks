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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/logreg_tpu_pytorch_lightning_bolts.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="2V7qvBWQceLw"
# # Logistic regression on MNIST using TPUs and PyTorch Lightning
#
# Code is from 
# https://lightning-bolts.readthedocs.io/en/latest/introduction_guide.html#logistic-regression
#
#
#

# + [markdown] id="BAlVldSaxFX8"
# # Setup TPU
#
# Be sure to select Runtime=TPU in the drop-down menu!
#
# See
# https://colab.sandbox.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb#scrollTo=3P6b3uqfzpDI
#
#
# See also 
# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/cifar10-baseline.html#
#

# + id="8p_a56gHgbgs"
import matplotlib.pyplot as plt
import numpy as np

# + id="rwD8v9CKwGU9"
import os
assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'

# + colab={"base_uri": "https://localhost:8080/"} id="4cfxkibaujx7" outputId="b9f0b70e-27af-4aa0-b2af-9cc64569c37a"

# #!pip install -q cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8-cp37-cp37m-linux_x86_64.whl

# !pip install -q cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

# + [markdown] id="0AA-nvEtxQJB"
# # Setup lightning

# + colab={"base_uri": "https://localhost:8080/"} id="VK77VYvyxeQX" outputId="98343365-87d8-428e-942d-16101ab2aca1"
# #!pip install -q lightning-bolts
# !pip install --quiet torchmetrics lightning-bolts torchvision torch pytorch-lightning

# + colab={"base_uri": "https://localhost:8080/"} id="dIDFTVR5ciGj" outputId="3c25a5f8-ffe0-43c1-8127-16c84f692af1"
from pl_bolts.models.regression import LogisticRegression
import pytorch_lightning as pl

from pl_bolts.datamodules import MNISTDataModule, FashionMNISTDataModule, CIFAR10DataModule, ImagenetDataModule

# + [markdown] id="LiWtOtbfiriw"
# # Iris

# + id="wMxR373kitKo"
from sklearn.datasets import load_iris
from pl_bolts.datamodules import SklearnDataModule
import pytorch_lightning as pl

# use any numpy or sklearn dataset
X, y = load_iris(return_X_y=True)
dm = SklearnDataModule(X, y, batch_size=12)

# build model
model = LogisticRegression(input_dim=4, num_classes=3)



# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["b924a5ebb42940bda433eefb9025a36b", "7e71b0ddf54f46f785bc4f3368c78c8b", "4607da79b3494266941feac5771d7a03", "b2705d08877b400494c1110466f62502", "7a146d34220d4a0b9ad8d6cbf5941dee", "938dad840d4441a995a97ad6f359ac2d", "acda2b8258de4673a54646c6ec15db9f", "9facef45ffc14020ba594a990903a35a", "a73b2615a67d4e9286de4c44cdf09b16", "60c9e1858fc743b7b945acba94630b8f", "0a21c0693d5d49229f9940a6316fed89", "4317a64501434487ad9029bd767e2896", "f1f266a2c688440e8b5380c60a3ea387", "f9a90fe9850148278b9425f7efe455dd", "dc95946640c04d77a8c76ad0ae0f25f6", "a7294949757b469cbf0f650693e6d4a0", "b076b8f65c694c5d83be2524a5463c1f", "04fb377ca1bc4c7b99e10e3922893e6d", "03bc872905e74b7a81fb34ad27096840", "7ae5224c111d421aaeeb959311c553a0", "40335da9d3924e7ab4b3f81dd10cef03", "dabf01dba6c6407883b3e30079001124", "f737644ab67440ff859db1d67b288512", "c390bd4f95ec4b72b9af9e9f1a243016"]} id="n__HFjcrr8-H" outputId="2d1133f2-8d36-4f40-aaa0-39c6c6581d7b"

# fit
trainer = pl.Trainer(tpu_cores=8)
trainer.fit(model, train_dataloader=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())


# + id="HQd-zPekr62-"
trainer.test(test_dataloaders=dm.test_dataloader())

# + [markdown] id="kGlURuxsijwx"
# # CIFAR

# + colab={"base_uri": "https://localhost:8080/", "height": 113, "referenced_widgets": ["e5dc024eea0b497d975c1a64b788d596", "7d10ca8321364d1facf3f3bc816bfd66", "dd0ce22e42a1479d922f48e3440747d2", "50d133ed052649aca50687b9efc6b5d5", "7c382921d96d446694fb49da3131a512", "37459cb5998645a2bc675f3867be42c8", "d5a422a80eca4d53a9ca58ab4266c3ab", "998dcee856134303b3987cee0f57a925"]} id="TS946EHNdsVm" outputId="c938d4f8-1651-495d-817f-b149ddaa61d7"
# create dataset
#dm = MNISTDataModule(num_workers=0, data_dir='data')
dm = CIFAR10DataModule(num_workers=0, data_dir='data')
dm.prepare_data() # force download now


# + colab={"base_uri": "https://localhost:8080/"} id="kestqpmxeRb8" outputId="fac42985-7020-4e94-e93e-0122053fcd9a"
print(dm.size())
print(dm.num_classes)
ndims = np.prod(dm.size())
nclasses = dm.num_classes
print([ndims, nclasses, ndims*nclasses])



# + colab={"base_uri": "https://localhost:8080/"} id="Wg0HBrtiiLSi" outputId="1066e7b8-188c-4a92-b3b6-3557a383f051"
model = LogisticRegression(input_dim=ndims, num_classes=nclasses, learning_rate=0.001)
print(model)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["b3a36b9ae01d4b9bade4e6a73e89452a", "b93f348aa03048af9b63b776b41ece9c", "fb2f129324374c6f84835c7b7fcd8acb", "9021dc38e0134da4aa22f47361ecc362", "a8fc2e40a528498f831ff6e3f0009074", "7bede214bc894836966a5fb59bfbce92", "ffbf861fd82b456cbeaf01227fdf4018", "c9037d063f064c93b88cf1010ac60ffc", "8cfc006b0ac34784a2ed783c550356a0", "eea987199c8c42bcbf54c009413c30a5", "cfa289bd8fd041759d86773d55f8bd21", "e345a2cea228475da59e312ddbb3c692", "44e67f40cd4e4ff5a0f5fa5b6efa72d7", "d5fe6a817baf44b0a985caac5d05280a", "82b4bc0fa051428ba6c40f9b64976388", "2c8166ef496349c6b2fdab5697c0b6d0", "46597386182b41d79d3c82a45b81be51", "096e2a3efcc840f5955f51155a1a1e2e", "56677c92446542bdb529e30038d017a3", "50cd48817f1b4ddaa2bc6bd7868a5447", "e115e9f6345245b5bdd0c9587d947ea3", "930ad7db16f3427b884f7f6ef55e18e5", "2d1283ac80e04f8aae6e0cf3a5ac960f", "1814ea1d91da47e598ddbf20e1b5717e"]} id="QwVKlB_qdt6Y" outputId="a5d5de08-72b1-4413-8a22-283a1275a55f"
trainer = pl.Trainer(tpu_cores=8, max_epochs=2)
#trainer = pl.Trainer(max_epochs=2)
trainer.fit(model, datamodule=dm)


# + colab={"base_uri": "https://localhost:8080/", "height": 146, "referenced_widgets": ["aadc9308bc704a5e8ff95e9c8f0a7ac0", "fbabd4d0a43f460fb96b835689e56407", "b2b6bf4d73bc4cdca544a1fa6d9af240", "e7ebb5b343d345e387c23d287c06f846", "1813480685ad47ce8d08d4891459d736", "512ffc142430495d9c9773446d1ddf45", "ee9d1af941004cac9783009b3f4e5cf4", "a58d47aa6b564cb790223786f0f2be04"]} id="bwG_J2YsgiMz" outputId="0f2c3701-45ab-4416-8c39-a52b0a0219d6"
trainer.test(model, test_dataloaders=dm.val_dataloader())


# + id="u55zhmeJiQxj"

