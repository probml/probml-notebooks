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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/student_pgmpy.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="FtkM9t9gXaVB"
# # The (simplified) student Bayes net
#
# This model is from https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/2.%20Bayesian%20Networks.ipynb

# + colab={"base_uri": "https://localhost:8080/"} id="8FEYdsYCXYj5" outputId="92d18d95-7059-4844-ff39-178f6daed5d8"
# !pip install -q causalgraphicalmodels
# !pip install -q pgmpy

# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pyprobml_utils.py
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pgmpy_utils.py
import pyprobml_utils as pml
import pgmpy_utils as pgm

# + id="piOvstm5YQWx"
from causalgraphicalmodels import CausalGraphicalModel
import pgmpy
import numpy as np
import pandas as pd

# + [markdown] id="uF_UjCP-Xi9-"
# # Model

# + [markdown] id="7A8L5MSGcuPC"
#
#
# <img src="https://user-images.githubusercontent.com/4632336/118884310-21bec180-b8ab-11eb-81cf-481553c21d8a.png?raw=true">
#

# + id="Yt2Dql3yXcgC"
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

# Defining the model structure. We can define the network by just passing a list of edges.
#model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])
model = BayesianModel([('Diff', 'Grade'), ('Intel', 'Grade'), ('Grade', 'Letter'), ('Intel', 'SAT')])



# + [markdown] id="KHlBciCF8EUp"
# ## Basic CPDs

# + id="UXxKULXg8Ixx" colab={"base_uri": "https://localhost:8080/"} outputId="a11535e6-86ab-436d-ba2a-7e92d3d7686f"
# Defining individual CPDs.
cpd_d = TabularCPD(variable='Diff', variable_card=2, values=[[0.6], [0.4]])
cpd_i = TabularCPD(variable='Intel', variable_card=2, values=[[0.7], [0.3]])

# The representation of CPD in pgmpy is a bit different than the CPD shown in the above picture. In pgmpy the colums
# are the evidences and rows are the states of the variable. So the grade CPD is represented like this:
#
#    +---------+---------+---------+---------+---------+
#    | diff    | intel_0 | intel_0 | intel_1 | intel_1 |
#    +---------+---------+---------+---------+---------+
#    | intel   | diff_0  | diff_1  | diff_0  | diff_1  |
#    +---------+---------+---------+---------+---------+
#    | grade_0 | 0.3     | 0.05    | 0.9     | 0.5     |
#    +---------+---------+---------+---------+---------+
#    | grade_1 | 0.4     | 0.25    | 0.08    | 0.3     |
#    +---------+---------+---------+---------+---------+
#    | grade_2 | 0.3     | 0.7     | 0.02    | 0.2     |
#    +---------+---------+---------+---------+---------+

cpd_g = TabularCPD(variable='Grade', variable_card=3, 
                   values=[[0.3, 0.05, 0.9,  0.5],
                           [0.4, 0.25, 0.08, 0.3],
                           [0.3, 0.7,  0.02, 0.2]],
                  evidence=['Intel', 'Diff'],
                  evidence_card=[2, 2])

cpd_l = TabularCPD(variable='Letter', variable_card=2, 
                   values=[[0.1, 0.4, 0.99],
                           [0.9, 0.6, 0.01]],
                   evidence=['Grade'],
                   evidence_card=[3])

cpd_s = TabularCPD(variable='SAT', variable_card=2,
                   values=[[0.95, 0.2],
                           [0.05, 0.8]],
                   evidence=['Intel'],
                   evidence_card=[2])

# Associating the CPDs with the network
model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
# defined and sum to 1.
model.check_model()

# + id="XU_P8PWX76bh" colab={"base_uri": "https://localhost:8080/"} outputId="467b027e-a7c3-465b-f82a-bbbeca7e360c"
print(model.get_cpds('Grade'))

# + [markdown] id="1kCNIT4b8Jts"
# ## CPDs with names states

# + colab={"base_uri": "https://localhost:8080/"} id="8c14_W6CX883" outputId="3d2d14ea-1ea4-44af-fa47-a5861701fff3"
# CPDs can also be defined using the state names of the variables. If the state names are not provided
# like in the previous example, pgmpy will automatically assign names as: 0, 1, 2, ....

cpd_d_sn = TabularCPD(variable='Diff', variable_card=2, values=[[0.6], [0.4]], 
                      state_names={'Diff': ['Easy', 'Hard']})
cpd_i_sn = TabularCPD(variable='Intel', variable_card=2, values=[[0.7], [0.3]],
                      state_names={'Intel': ['Dumb', 'Intelligent']})
cpd_g_sn = TabularCPD(variable='Grade', variable_card=3, 
                      values=[[0.3, 0.05, 0.9,  0.5],
                              [0.4, 0.25, 0.08, 0.3],
                              [0.3, 0.7,  0.02, 0.2]],
                      evidence=['Intel', 'Diff'],
                      evidence_card=[2, 2],
                      state_names={'Grade': ['A', 'B', 'C'],
                                   'Intel': ['Dumb', 'Intelligent'],
                                   'Diff': ['Easy', 'Hard']})

cpd_l_sn = TabularCPD(variable='Letter', variable_card=2, 
                      values=[[0.1, 0.4, 0.99],
                              [0.9, 0.6, 0.01]],
                      evidence=['Grade'],
                      evidence_card=[3],
                      state_names={'Letter': ['Bad', 'Good'],
                                   'Grade': ['A', 'B', 'C']})

cpd_s_sn = TabularCPD(variable='SAT', variable_card=2,
                      values=[[0.95, 0.2],
                              [0.05, 0.8]],
                      evidence=['Intel'],
                      evidence_card=[2],
                      state_names={'SAT': ['Bad', 'Good'],
                                   'Intel': ['Dumb', 'Intelligent']})

# These defined CPDs can be added to the model. Since, the model already has CPDs associated to variables, it will
# show warning that pmgpy is now replacing those CPDs with the new ones.
model.add_cpds(cpd_d_sn, cpd_i_sn, cpd_g_sn, cpd_l_sn, cpd_s_sn)
model.check_model()

# + colab={"base_uri": "https://localhost:8080/"} id="PzkidOsEX9H6" outputId="2d05861d-d282-4702-d362-a4ea421b35de"
 #Printing a CPD with it's state names defined.
print(model.get_cpds('Grade'))

# + colab={"base_uri": "https://localhost:8080/"} id="6S3T_rFxYCfR" outputId="903312b9-fa05-49ba-8189-e87d4f46d045"
for cpd in model.get_cpds():
  print(cpd)

# + [markdown] id="mmIYMZVzYuxu"
# # Inference

# + id="PrFI130fYMQ0"
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)


# + [markdown] id="zxIyjd9eYx2S"
# ## Posterior given Grade=C

# + colab={"base_uri": "https://localhost:8080/"} id="oEEU40_pY4C7" outputId="a3e73c3e-24e6-48e8-8fcc-2035066fdcf2"
evidence = {'Grade': 'C'}
postD = infer.query(['Diff'],  evidence=evidence).values
postI = infer.query(['Intel'],  evidence=evidence).values

print('\n')
print('Pr(Difficulty=Hard|Grade=C) = {:0.2f}'.format(postD[1]))
print('Pr(Intelligent=High|Grade=C) = {:0.2f}'.format(postI[1]))

# + [markdown] id="Z2nLT8lYZ2QC"
# ## Posterior given Grade=C, SAT=Good

# + colab={"base_uri": "https://localhost:8080/"} id="NfKdVsChZwrR" outputId="1041bab8-4fe2-4d62-b5e3-f6bbd4513db1"
evidence = {'Grade': 'C', 'SAT': 'Good'}
postD = infer.query(['Diff'],  evidence=evidence).values
postI = infer.query(['Intel'],  evidence=evidence).values

print('\n')
print('Pr(Difficulty=Hard|Grade=C,SAT=Good) = {:0.2f}'.format(postD[1]))
print('Pr(Intelligent=High|Grade=C,SAT=Good) = {:0.2f}'.format(postI[1]))

# + [markdown] id="E5LUO3VSYE5_"
# # Visualization

# + [markdown] id="x5jx0FmAf1t7"
# ## DAG

# + colab={"base_uri": "https://localhost:8080/", "height": 305} id="MDFzI490YFq9" outputId="0ebb3ece-c80b-427f-ad72-cd1d6be69f55"
model2 = CausalGraphicalModel(nodes = model.nodes(), edges=model.edges())

dot = model2.draw()
print(type(dot))
display(dot)
dot.render(filename='student_pgm', format='pdf')
# creates student_pgm (a text file of the graph) and student_pgm.pdf

# + colab={"base_uri": "https://localhost:8080/", "height": 17} id="l2UUtA5_w15d" outputId="cbcc8b5f-e406-491c-caee-6d0350c5121e"
from google.colab import files
files.view('student_pgm') # open text file

# + [markdown] id="LXTnimbBf3yy"
# ## CPTs

# + id="_ujhz9g6eAy6" colab={"base_uri": "https://localhost:8080/", "height": 820} outputId="a5eead04-3fd3-4d68-d001-42ab649facbd"
dot = pgm.visualize_model(model)
display(dot)
dot.render('student_pgm_with_cpt', format='pdf')


# + [markdown] id="GdtP7ww5f5v0"
# ## Marginals

# + colab={"base_uri": "https://localhost:8080/"} id="2rUHEgV8yTCw" outputId="f2e39344-86b4-4228-e8f4-6221ba591502"

evidence = {'Grade': 'C'}
marginals = pgm.get_marginals(model, evidence)
print(marginals)

# + colab={"base_uri": "https://localhost:8080/", "height": 603} id="hIwLouEyyWnQ" outputId="8b70ba1a-86ac-42b9-ed83-d1afb87e9ffe"
dot = pgm.visualize_marginals(model, evidence, marginals)
display(dot)
dot.render('student_pgm_marginals_given_grade', format='pdf')

# + colab={"base_uri": "https://localhost:8080/", "height": 753} id="GvQFdruIvlu8" outputId="cbc1f070-d7c2-4ab1-f51c-944740855911"

evidence = {'Grade': 'C', 'SAT': 'Good'}
marginals = pgm.get_marginals(model, evidence)
print(marginals)

dot = pgm.visualize_marginals(model, evidence, marginals)
display(dot)
dot.render('student_pgm_marginals_given_grade_sat', format='pdf')

# + id="FDxivZgxxJcg"

