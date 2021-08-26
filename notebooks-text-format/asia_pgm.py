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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/asia_pgm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="caeOoFYBb5p2"
# # The "Asia"  graphical model
#
# We illustrate inference in the "Asia" medical diagnosis network.
# Network is from http://www.bnlearn.com/bnrepository/#asia.
#

# + id="C-zM-WQZFMzv" outputId="445781c6-0c9d-44b1-d729-d10cf7aa34a7" colab={"base_uri": "https://localhost:8080/"}
# !pip install -q causalgraphicalmodels
# !pip install -q pgmpy

# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pyprobml_utils.py
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pgmpy_utils.py
import pyprobml_utils as pml
import pgmpy_utils as pgm

import numpy as np
import matplotlib.pyplot as plt

# + id="Go1UPp8eMIyC"
from causalgraphicalmodels import CausalGraphicalModel
import pgmpy
import numpy as np
import pandas as pd
from graphviz import Digraph

import pgmpy_utils as pgm # from pyprobml


# + [markdown] id="u_-A-Hai4LbM"
# # Model

# + id="Il1OHrxj3e8Y"
#from pgmpy.utils import get_example_model
#asia = get_example_model('asia')
# No such file or directory: 'pgmpy/utils/example_models/asia.bif.gz'

# + id="OGsv1dt81oyI"
# #!wget https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l/torch.py -q -O d2l.py

# !wget https://www.bnlearn.com/bnrepository/asia/asia.bif.gz -q -O asia.bif.gz
# !gunzip asia.bif.gz

# + id="l8xzIiLr0En0" colab={"base_uri": "https://localhost:8080/"} outputId="d52ec9c4-d117-4f08-c322-d10548fe583a"

from pgmpy.readwrite import BIFReader, BIFWriter
reader = BIFReader("asia.bif")
model = reader.get_model()

print("Nodes: ", model.nodes())
print("Edges: ", model.edges())
model.get_cpds()

# + colab={"base_uri": "https://localhost:8080/"} id="d1GFKz5t5kxA" outputId="ea314557-c811-4b92-8cc8-06d521e37f30"
for c in model.get_cpds():
  print(c)

# + colab={"base_uri": "https://localhost:8080/"} id="8jzeJCpE50kw" outputId="771fdaed-4d31-4a21-ac65-293061bf0472"
asia_cpd = model.get_cpds('asia')
print(asia_cpd)
print(asia_cpd.values)



# + colab={"base_uri": "https://localhost:8080/"} id="VDNdkr0_6TsN" outputId="28e96aff-e08a-495e-b66c-8822c31badba"
smoking_cpd = model.get_cpds()[5]
print(smoking_cpd)
print(smoking_cpd.values)
smoking_prior_true = smoking_cpd.values[1]

# + id="K8gIy5Bk2XJR" colab={"base_uri": "https://localhost:8080/", "height": 385} outputId="46b7790f-8a18-462a-a0e1-65f7bf5fd3aa"
asia = CausalGraphicalModel(nodes = model.nodes(), edges=model.edges())

out = asia.draw()
display(out)
out.render()

# + id="hD1xceNIF7zI" outputId="18f390c9-ad89-45d8-c4b8-0b6c6a91d537" colab={"base_uri": "https://localhost:8080/", "height": 1000}

dot = pgm.visualize_model(model)
display(dot)
dot.render('asia_pgm_with_cpt', format='pdf')

# + [markdown] id="5FyYbZfmD-AV"
# # Inference

# + id="dJwoaVgi47Oe"
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)



# + [markdown] id="Fo1Z_lqa4fnB"
# ## Prior marginals

# + colab={"base_uri": "https://localhost:8080/"} id="wIj_U9mC-Q3M" outputId="d54c04d9-3e81-41f8-ea4b-33a92a04ec83"
evidence = {}
marginals = pgm.get_marginals(model, evidence)
print('\n')
for k, v in marginals.items():
  print(k, v)

asia_prior = model.get_cpds('asia').values
assert np.allclose(asia_prior, marginals['asia'])

# + colab={"base_uri": "https://localhost:8080/", "height": 786} id="oC0BWo-PDQU8" outputId="c39cec10-e5a4-42c2-d26e-61dd43807d45"
display(pgm.visualize_marginals(model, evidence, marginals))


# + [markdown] id="StB-XG1z7FEP"
# ## Posterior marginals given dsypnea=yes

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="ZMcpy_pA67bi" outputId="55655b2f-5813-4394-d20f-4b3c75a75363"

evidence  = {'dysp': 0}
marginals = pgm.get_marginals(model, evidence, infer)
print('\n')
for k, v in marginals.items():
  print(k, v)

display(pgm.visualize_marginals(model, evidence, marginals))



# + [markdown] id="4xM1w8qR8wgG"
# ## Posterior marginals given dsypnea=yes, asia=yes

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="eJJhZFWf70pl" outputId="8d9fc8bb-c41e-41e6-abc2-39f5bc4dac73"
evidence  = {'dysp': 'yes', 'asia': 'yes'}
marginals = pgm.get_marginals(model,  evidence, infer)
print('\n')
for k, v in marginals.items():
  print(k, v)

display(pgm.visualize_marginals(model, evidence, marginals))


# + [markdown] id="hyALNuUaCNza"
# ## Posterior marginals given dsypnea=yes, asia=yes, smoking=yes

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="gDydpbab9Svz" outputId="c821e8f4-a005-414d-8e0c-a46733bcec9d"
evidence  = {'dysp': 'yes', 'asia': 'yes', 'smoke': 'yes'}
marginals = pgm.get_marginals(model, evidence, infer)
print('\n')
for k, v in marginals.items():
  print(k, v)


display(pgm.visualize_marginals(model, evidence, marginals))


# + id="24L0MlUXCgDw"

