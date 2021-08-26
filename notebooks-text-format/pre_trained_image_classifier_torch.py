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
#     language: python
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/pre_trained_image_classifier_torch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="Frv977yg2dww"
# # Apply a Resnet image classifier that has been pre-trained on ImageNet 

# + id="z5a1FHe-vdkn"
import torch
from torchvision import models

# + id="Qgi8cSdSvk8z" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="10fb0e09-df03-43a7-dfac-36148deb3363"
torch.__version__

# + [markdown] id="ltduunBdnbKN"
# List of pre-trained models.

# + id="6LVoxV75vdko" colab={"base_uri": "https://localhost:8080/"} outputId="1834691f-4215-4db9-f51f-cc2f57ee827a"
dir(models)

# + id="PsC2lTh5vdkp" colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["23e85808a1c54210b612176371104cde", "2e750d0cfb50481f9d3ef39f635c4532", "d3291c822cd142d5a858b9872ee496e6", "62b84d51ca5b41b9a8f7d4bb659e0e18", "7e827d6e806b418995041d4c71e907b9", "0c8a19c2a83f4eeb8f5cffe1608975b3", "70ee9e93723e4b7da6963de6d1e04e75", "e042f19776db4cf48729524a21487f92"]} outputId="12d1098d-7cd9-48b7-8cb6-d256924ab313"
resnet = models.resnet101(pretrained=True)

# + id="byIForimoiAo" outputId="a33e13f5-826f-45be-a889-c6ef12bd96a9" colab={"base_uri": "https://localhost:8080/"}
resnet

# + id="6Jh6YBKIomQB" outputId="b1c40c7b-5679-4dff-c155-aa1d98361781" colab={"base_uri": "https://localhost:8080/"}
resnet.fc

# + id="N476SFdzvdkq"
from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

# + id="Wytu8FYpn5t3" outputId="f5c5593e-270d-4546-a5d9-2f14f765e0b3" colab={"base_uri": "https://localhost:8080/", "height": 497}
# !wget https://github.com/probml/pyprobml/blob/master/images/cat_dog.jpg?raw=true -q -O img.jpg
from PIL import Image
img = Image.open('img.jpg')
display(img)

# + id="xjlCv-lxvdkq" colab={"base_uri": "https://localhost:8080/"} outputId="1f561125-ce34-4720-9eca-18a727a00334"
print(type(img))
img_t = preprocess(img) # convert to tensor
print(type(img_t))

# + id="3u9oW64rvdkr" colab={"base_uri": "https://localhost:8080/"} outputId="0b6f3482-bd5d-4af5-db2c-0756225a48fa"
import torch
batch_t = torch.unsqueeze(img_t, 0)
print(img_t.shape)
print(batch_t.shape)

# + id="z1Y43v4yvdkr"
resnet.eval(); # set to eval mode (not training)

# + id="57BrgTzavdkr" colab={"base_uri": "https://localhost:8080/"} outputId="c887f8b9-aff7-46fe-e32f-42571077ebf2"
out = resnet(batch_t)
print(out.shape)


# + id="VtzgSke-vdkr" colab={"base_uri": "https://localhost:8080/"} outputId="0324e996-945c-44d3-ef6a-a3f93392fe91"
import urllib
import requests
url = "https://raw.githubusercontent.com/probml/pyprobml/master/data/imagenet_classes.txt"
response = requests.get(url)
file = urllib.request.urlopen(url)
labels = [line.strip() for line in file.readlines()]
print(labels)

# + id="TNjU-5Kwvdks" colab={"base_uri": "https://localhost:8080/"} outputId="fbe9f1c3-5ffd-4364-ec73-782b2c598f39"
# Show top 5 labels
_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

# + id="zuIDA1s4vdks"

