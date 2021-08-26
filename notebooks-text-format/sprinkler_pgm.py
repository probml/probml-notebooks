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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/sprinkler_pgm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="caeOoFYBb5p2"
# # Directed graphical models
#
# We illustrate some basic properties of DGMs.

# + colab={"base_uri": "https://localhost:8080/"} id="p4SVMjkpbcJ7" outputId="7981ae35-bac7-4def-b85d-7c5d4e0a3c8f"
# !pip install  causalgraphicalmodels
# !pip install pgmpy

# #!ls /usr/local/lib/python3.7/dist-packages/pgmpy/utils

# + id="Go1UPp8eMIyC"
from causalgraphicalmodels import CausalGraphicalModel
import pgmpy
import numpy as np
import pandas as pd

# + [markdown] id="FO0xjHmYcBxI"
# # Make the model

# + id="_W8DPmP7bX-F"
sprinkler = CausalGraphicalModel(
    nodes=["cloudy", "rain", "sprinkler", "wet", "slippery"],
    edges=[
        ("cloudy", "rain"), 
        ("cloudy", "sprinkler"), 
        ("rain", "wet"),
        ("sprinkler", "wet"), 
        ("wet", "slippery")
    ]
)



# + [markdown] id="U3UlRiqWcDgY"
# # Draw the model

# + id="pnMGiQ8pb4Sq"
# draw return a graphviz `dot` object, which jupyter can render
out = sprinkler.draw()

# + colab={"base_uri": "https://localhost:8080/"} id="PCCMSDcLsQDE" outputId="7811c485-5562-477b-bcae-640cd5101196"
type(out)


# + colab={"base_uri": "https://localhost:8080/", "height": 367} id="x8Qs8OSSsIty" outputId="b34bd7f5-c6db-417f-a71a-67bbc8daaba9"
display(out)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="sL02TpzKsg10" outputId="ed5d8de5-940c-467b-fc42-9877bee3f0fc"
out.render()  

# + [markdown] id="ZKK4TvYTcFZi"
# # Display the factorization

# + colab={"base_uri": "https://localhost:8080/"} id="5sqd2omObvG0" outputId="737488e8-a785-4cfb-bbae-2b71f5b7d898"
print(sprinkler.get_distribution())

# + [markdown] id="esxi6RlAcHhD"
# # D-separation

# + colab={"base_uri": "https://localhost:8080/"} id="jZaNeHdQb1Rd" outputId="914b6b74-d3c2-423c-fd5d-6a822d7ae528"

# check for d-seperation of two nodes
sprinkler.is_d_separated("slippery", "cloudy", {"wet"})

# + [markdown] id="p71us5z5cJt3"
# # Extract CI relationships

# + colab={"base_uri": "https://localhost:8080/"} id="UPYs6KMXbvbt" outputId="ee0dbdc7-dab1-431f-c023-31091d100494"
# get all the conditional independence relationships implied by a CGM
CI = sprinkler.get_all_independence_relationships()
print(CI)

# + colab={"base_uri": "https://localhost:8080/", "height": 543} id="70jWucdv2ut4" outputId="2327326f-1a77-4f90-bcdb-73127b95411d"
records = []
for ci in CI:
  record = (ci[0], ci[1], ', '.join(x for x in ci[2]))
  records.append(record)

print(records)
df = pd.DataFrame(records, columns = ('X', 'Y', 'Z'))
display(df)

# + colab={"base_uri": "https://localhost:8080/"} id="E2aCZYTu3n5z" outputId="905c6abb-652d-4947-d4c0-170648041e61"
print(df.to_latex(index=False))

# + [markdown] id="NWhIaqSeu-AY"
# # Parameterize the model

# + colab={"base_uri": "https://localhost:8080/"} id="ni3LOOaQbzIh" outputId="6a871813-bf14-4c47-8908-90f5fc64cbff"


from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianModel([('C', 'S'), ('C', 'R'), ('S', 'W'), ('R', 'W'), ('W', 'L')])

# Defining individual CPDs.
cpd_c = TabularCPD(variable='C', variable_card=2, values=np.reshape([0.5, 0.5],(2,1)))

# In pgmpy the columns are the evidences and rows are the states of the variable.
 
cpd_s = TabularCPD(variable='S', variable_card=2, 
                   values=[[0.5, 0.9],
                           [0.5, 0.1]],
                  evidence=['C'],
                  evidence_card=[2])

cpd_r = TabularCPD(variable='R', variable_card=2, 
                   values=[[0.8, 0.2],
                           [0.2, 0.8]],
                  evidence=['C'],
                  evidence_card=[2])

cpd_w = TabularCPD(variable='W', variable_card=2, 
                   values=[[1.0, 0.1, 0.1, 0.01],
                           [0.0, 0.9, 0.9, 0.99]],
                  evidence=['S', 'R'],
                  evidence_card=[2, 2])

cpd_l = TabularCPD(variable='L', variable_card=2, 
                   values=[[0.9, 0.1],
                           [0.1, 0.9]],
                  evidence=['W'],
                  evidence_card=[2])

# Associating the CPDs with the network
model.add_cpds(cpd_c, cpd_s, cpd_r, cpd_w, cpd_l)

# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
# defined and sum to 1.
model.check_model()




# + [markdown] id="gDGeTduQ4mOp"
# # Inference

# + colab={"base_uri": "https://localhost:8080/"} id="V9-tbp7wx2D3" outputId="e6d47854-ce71-47d9-f851-e1c086d666e2"
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)

# p(R=1)= 0.5*0.2 + 0.5*0.8 = 0.5
probs = infer.query(['R']).values
print('\np(R=1) = ', probs[1])

# P(R=1|W=1) = 0.7079
probs = infer.query(['R'], evidence={'W': 1}).values
print('\np(R=1|W=1) = ', probs[1])


# P(R=1|W=1,S=1) = 0.3204
probs = infer.query(['R'], evidence={'W': 1, 'S': 1}).values
print('\np(R=1|W=1,S=1) = ', probs[1])
