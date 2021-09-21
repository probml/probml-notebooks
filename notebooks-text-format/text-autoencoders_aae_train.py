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

# + id="eUAv0Mzo0Lcm" colab={"base_uri": "https://localhost:8080/"} outputId="9c1b6474-aff7-4a58-b2bf-34481544f615"
from google.colab import drive
drive.mount('/content/drive')

# + id="CTs9XQvFCZa-" colab={"base_uri": "https://localhost:8080/"} outputId="2084e5fc-3005-48c5-dcd8-3dc716220d28"
import torch
from multiprocessing import cpu_count
print(cpu_count())
print(torch.cuda.is_available())

# + id="OHQ1OBH6CbIV" colab={"base_uri": "https://localhost:8080/"} outputId="747bae11-ea87-4107-b5e3-ff71da3d1267"
# !git clone https://github.com/shentianxiao/text-autoencoders.git

# + id="7LeZIGyOCbQe" colab={"base_uri": "https://localhost:8080/"} outputId="d60b9fd8-fece-45ac-f9d6-d1c2f3cd5a3f"
# %cd text-autoencoders

# + [markdown] id="bzZ_0JsyCnAz"
# ##DATA

# + id="GtRrInqSCoLc" colab={"base_uri": "https://localhost:8080/"} outputId="0f2bcb44-891e-4727-c054-1b073be02192"
# !bash download_data.sh

# + [markdown] id="Jj1dupiX0z0b"
# ## Training the AAE model for 30 epochs

# + id="nrcu6QBIhr-5" colab={"base_uri": "https://localhost:8080/"} outputId="5ef8e81e-b5fb-4ffb-a9a2-339ba7354d62"
NUM_EPOCHS = 30 
# !python train.py --epochs $NUM_EPOCHS --train data/yelp/train.txt --valid data/yelp/valid.txt --model_type aae --lambda_adv 10 --noise 0.3,0,0,0 --save-dir checkpoints/yelp/daae

# + colab={"base_uri": "https://localhost:8080/"} id="MlK_CCuVuoLD" outputId="0ee9b4c5-40a6-45a3-976f-b248827895e6"
# !zip -r /content/text-autoencoders/checkpoints.zip /content/text-autoencoders/checkpoints/ 

# + id="mKQqSoV_vEDh"
# !cp /content/text-autoencoders/checkpoints.zip /content/drive/MyDrive/checkpoints
