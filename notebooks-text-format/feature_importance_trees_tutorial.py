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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/trees/feature_importance_trees_tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="gL5HzLsrDUPM"
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/intro/bagging_boosting_trees_and_forests.ipynb)
#
#

# + [markdown] id="xMEFX8DznQC5"
#
#
# ![GitHub](https://img.shields.io/github/license/probml/pyprobml)
#
# Authors: Kevin P. Murphy (murphyk@gmail.com) and Mahmoud Soliman (mjs@aucegypt.edu)
#

# + [markdown] id="xNvkqv5s8SLU"
# In this notebook we will explore how to use XGBoost and sklearn's random forests to evaluate feature importance. 
#
# **XGBoost**
#
# Support for the following features:
# 1. Vanilla Gradient Boosting algorithm (also known as GBDT (Grandient boosted decisin trees) or GBM(gradient boosting machine) with support to tuning [parameters](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster), parallization and GPU support.
# 2. Stochastic Gradient Boosting with sampling with uniform and gradient-based sampling support as well as sub-sampling at the row, column and column per split levels.
# 3. Regularized Gradient Boosting with support to both L1 and L2 regularization(via alpha and lamda parameters respectively).
# 4. Dropout-esque behaviour via DART booster.
#
# Note that we are using the SKLearn-like api of XGBoost for simplicity.
#
# **SKLearn** 
#
# supports several features for ensemble learning one of which is Random forests, which uses bagging of decision tree classifiers (weak learners) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
#

# + id="LooMB_tChMat"
# Attribution 
#This notebook is based on the following: 
#https://www.kaggle.com/kevalm/xgboost-implementation-on-iris-dataset-python
#https://xgboost.readthedocs.io/en/latest/tutorials/index.html
#https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#https://github.com/slundberg/shap/blob/master/notebooks/kernel_explainer/Iris%20classification%20with%20scikit-learn.ipynb

# + [markdown] id="tEPrhkWco3AU"
# #Setup 
#

# + id="LmGavxNegH4A"
# Imports

import os
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

# + id="OStleC8s7um9" colab={"base_uri": "https://localhost:8080/"} outputId="54774ece-5a33-4855-e139-0de79b3e5831"
# Install the extra required packages if any
# !pip install lime -qq
import lime
import lime.lime_tabular as ll
# !pip install shap -qq
import shap

# + [markdown] id="fCBU5_vt7SCN"
# # Iris dataset
#

# + id="5QDROXli5l1e"
#loading the dataset
iris = datasets.load_iris() 
X = iris.data               
y = iris.target             

# + id="GkwH6S7B5rmM"
#Splitting data into 80/20 training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# + [markdown] id="ih7VTJNQsora"
# # Exploring boosting with xgboost

# + id="GI1SE7VBOd5t"
#@title XGBClassifier


# + id="RVHiiJiiP-_o" colab={"base_uri": "https://localhost:8080/"} outputId="fe76e784-369c-48aa-9b4c-7e586dfc024d"
xgbc = XGBClassifier()
xgbc

# + id="ohJ2-IKA6F6Z" colab={"base_uri": "https://localhost:8080/"} outputId="14eea90e-850e-4bec-e8b4-49ab4900ea8c"
#Training the classifier
xgbc.fit(X_train, y_train)
#Inferencing on testing data
xgbc_y_pred = xgbc.predict(X_test)
#Measuring accuracy
xgbc_acc=metrics.accuracy_score(y_test, xgbc_y_pred)
print('XGBClassifier accuracy is '+str(xgbc_acc))

# + id="_EWy2apMzPxi" cellView="form"
#@title Visualization of boosted tree
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

xgb.plot_tree(xgbc, num_trees=2)
fig = plt.gcf()
fig.set_size_inches(150, 100)
fig.savefig('treeIris.png') 

# + id="W9R6zux9zcHa" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="38fd23b0-9adf-4d7c-a007-3f7e3dc3e4dc"
#@title Feature importance of XGBClassifier

plot_importance(xgbc)
pyplot.show()

#f1 - sepal length in cm
#f2 - sepal width in cm
#f3 - petal length in cm
#f4 - petal width in cm 

# + id="BQ9pWAk2A84_" colab={"base_uri": "https://localhost:8080/", "height": 121} outputId="8308c575-92e1-4033-8889-58b8b4e4fc65"
#@title Explanation of a sample of testing data of XGBClassifier via LIME
xgbc_lime_explainer = ll.LimeTabularExplainer(X_train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)
xgbc_i = np.random.randint(0, X_test.shape[0])
xgbc_exp = xgbc_lime_explainer.explain_instance(X_test[xgbc_i], xgbc.predict_proba, num_features=2, top_labels=1)
xgbc_exp.show_in_notebook(show_table=True, show_all=False)

# + id="d_0qmuMmHB_I" colab={"base_uri": "https://localhost:8080/", "height": 473, "referenced_widgets": ["549b8c9bb0a84691af0057566c20f512", "075ef0e29acf4c4cb5ed7f98a42243d5", "a82c6ab778484379a0d0b884d4c6c09c", "be8e05ebb6b74195b706cb4a363b05af", "b93a165fdb8746aabb86e8af38ff2b34", "c85290c2a61f43e5926b77a33efc7903", "f5c4e083da6440248265db52333215e5", "ae6dd4bfd315443e8430d897c5cfffc7"]} outputId="8a1af0bd-3056-425d-a94b-daf56494bd88"
#@title Explanation of testing data of XGBClassifier via SHAP
shap.initjs()
# explain all the predictions in the test set
xgbc_shap_explainer = shap.KernelExplainer(xgbc.predict_proba, X_train,model_output='probability', feature_perturbation = "interventional")
xgbc_shap_values = xgbc_shap_explainer.shap_values(X_test)
shap.force_plot(xgbc_shap_explainer.expected_value[0], xgbc_shap_values[0], X_test)

# + id="llIFYfwBLic0" colab={"base_uri": "https://localhost:8080/", "height": 232} outputId="cfcdd23e-7f15-4986-9bf0-56f09d79822f"
xgbc_shap_explainer_2=shap.TreeExplainer(xgbc)
xgbc_shap_values_2 = xgbc_shap_explainer_2.shap_values(X_test)
shap.summary_plot(xgbc_shap_values_2, X_test)

# + [markdown] id="MtrfoIpb4vW8"
# # Exploring bagging (Random Forests) with sklearn

# + id="-9wmaLK-OL3b" cellView="form"
#@title RandomForestClassifier


# + id="AnqPwiP_QfIa" colab={"base_uri": "https://localhost:8080/"} outputId="2fe853d4-165d-4305-c790-d362afb4d7f4"
skrfc = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
skrfc.fit(X_train, y_train)

# + id="xV5gz694TUik" colab={"base_uri": "https://localhost:8080/"} outputId="6ac724e7-5034-471d-b802-1d713fb0630f"
skrfc_y_pred = skrfc.predict(X_test)
skrfc_acc=metrics.accuracy_score(y_test, skrfc_y_pred)
print('RandomForestClassifier accuracy is '+str(skrfc_acc))

# + id="NVYnGmXPPTdx" colab={"base_uri": "https://localhost:8080/", "height": 281} outputId="faf30cb4-db85-4743-c195-19314fbfd52f"
#@title Feature importance of RandomForestClassifier
importances = skrfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in skrfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# + id="web-7y5wUWmH" colab={"base_uri": "https://localhost:8080/", "height": 121} outputId="bcbbfeb8-6351-44a0-bc6b-00f163ce2e45"
#@title Explanation of a sample of testing data of RandomForestClassifier via LIME
skrfc_lime_explainer = ll.LimeTabularExplainer(X_train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)
skrfc_i = np.random.randint(0, X_test.shape[0])
skrfc_exp = skrfc_lime_explainer.explain_instance(X_test[xgbc_i], skrfc.predict_proba, num_features=2, top_labels=1)
skrfc_exp.show_in_notebook(show_table=True, show_all=False)

# + id="Ts_TcpnC-4IL" colab={"base_uri": "https://localhost:8080/", "height": 473, "referenced_widgets": ["3546b95356bb4d1eb9a8c10a90d059a8", "ad65536e6e77483a8be25a5d53564a2e", "01d4bbfc6b4142ed98ac679fc24a24a1", "e246bf0bfd6e41458982854dd18e40f8", "1cb35118bdf244b481df0bbad6fc28c2", "8b92c2a2e82c4d6a8f299778694d31e9", "afa2a3bfa0a24fa082204096496ae503", "e2818da23980428dbfd124f6ea99d36c"]} outputId="f9f829dc-8f11-41c7-edf5-d23864f372e8"
#@title Explanation of testing data of RandomForestClassifier via SHAP
shap.initjs()
# explain all the predictions in the test set
skrfc_shap_explainer = shap.KernelExplainer(skrfc.predict_proba, X_train,model_output='probability', feature_perturbation = "interventional")
skrfc_shap_values = skrfc_shap_explainer.shap_values(X_test)
shap.force_plot(skrfc_shap_explainer.expected_value[0], skrfc_shap_values[0], X_test)

# + id="eEBifUbm_Phz" colab={"base_uri": "https://localhost:8080/", "height": 232} outputId="cd49aedd-f46d-4646-caa7-a7d23ea4317f"
skrfc_shap_explainer_2=shap.TreeExplainer(skrfc)
skrfc_shap_values_2 = skrfc_shap_explainer_2.shap_values(X_test)
shap.summary_plot(skrfc_shap_values_2, X_test)
