# -*- coding: utf-8 -*-
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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/iris_dtree.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="pJAXuwceKMxg"
# # Decision tree classifier on Iris data
#
# Based on 
# https://github.com/ageron/handson-ml2/blob/master/06_decision_trees.ipynb

# + id="agyukRFGIDqW"
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import pandas as pd

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
import seaborn as sns


# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt


# + id="uZRZ4wPuV-E5"
# Font sizes
SIZE_SMALL = 18 #14
SIZE_MEDIUM = 20 # 18
SIZE_LARGE = 24

# https://stackoverflow.com/a/39566040
plt.rc('font', size=SIZE_SMALL)          # controls default text sizes
plt.rc('axes', titlesize=SIZE_SMALL)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE_SMALL)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE_SMALL)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE_SMALL)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE_SMALL)    # legend fontsize  
plt.rc('figure', titlesize=SIZE_LARGE)   # fontsize of the figure title

# + [markdown] id="lRYWVyJaKLy8"
# # Data

# + colab={"base_uri": "https://localhost:8080/", "height": 734} id="fd2kv3DxIOeJ" outputId="cd5e5059-d9ce-4b42-9a31-75bcc8f07608"

iris = load_iris()
X = iris.data
y = iris.target
print(iris.feature_names)

# Convert to pandas dataframe 
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['label'] = pd.Series(iris.target_names[y], dtype='category')

# we pick a color map to match that used by decision tree graphviz 
#cmap = ListedColormap(['#fafab0','#a0faa0', '#9898ff']) # orange, green, blue/purple
#cmap = ListedColormap(['orange', 'green', 'purple']) 
palette = {'setosa': 'orange', 'versicolor': 'green', 'virginica': 'purple'}

g = sns.pairplot(df, vars = df.columns[0:4], hue="label", palette=palette)
#g = sns.pairplot(df, vars = df.columns[0:4], hue="label")
plt.savefig("iris_scatterplot_v2.pdf")
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} id="pfBk8QDIIRBs" outputId="8ab79085-4a1f-441a-9f26-e8527dba1c1b"
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.target_names)
print(iris.feature_names)

#ndx = [0, 2] # sepal length, petal length
ndx = [2, 3] # petal lenght and width
X = iris.data[:, ndx] 
y = iris.target
xnames = [iris.feature_names[i] for i in ndx]
ynames = iris.target_names




# + id="26Opc8mnI5g8"
def plot_surface(clf, X, y, xnames, ynames):
    n_classes = 3
    plot_step = 0.02
    markers = [ 'o', 's', '^']
    
    plt.figure(figsize=(10,10))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.xlabel(xnames[0])
    plt.ylabel(xnames[1])

    # we pick a color map to match that used by decision tree graphviz 
    cmap = ListedColormap(['orange', 'green', 'purple']) 
    #cmap = ListedColormap(['blue', 'orange', 'green']) 
    #cmap = ListedColormap(sns.color_palette())
    plot_colors = [cmap(i) for i in range(4)]

    cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5) 
    # Plot the training points
    for i, color, marker in zip(range(n_classes), plot_colors, markers):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], label=ynames[i],
                    edgecolor='black', color = color, s=50, cmap=cmap, 
                    marker = marker)
    plt.legend()



# + [markdown] id="f9dQZFpEKRnF"
# # Depth 2

# + colab={"base_uri": "https://localhost:8080/"} id="MV4wn6aQKIVb" outputId="381d118f-c9f0-4f97-c324-b73554bcde31"
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

# + colab={"base_uri": "https://localhost:8080/", "height": 380} id="YpIKMcF1IV6o" outputId="1575923e-3b33-4a1c-ec3d-71f8c114792c"
from graphviz import Source
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file= "iris_tree.dot",
        feature_names=xnames,
        class_names=ynames,
        rounded=True,
        impurity = False,
        filled=True
    )

Source.from_file("iris_tree.dot")


# + id="N80oHMuhZecS" outputId="995424ee-85f7-4383-e12c-db7d5eb1a42f" colab={"base_uri": "https://localhost:8080/", "height": 34}
plt.savefig("dtree_iris_depth2_tree_v2.pdf")

# + colab={"base_uri": "https://localhost:8080/", "height": 622} id="o4iYj9MyJDes" outputId="d8d9949d-c62e-442a-cb11-d3a6808fc370"
plot_surface(tree_clf, X, y, xnames, ynames)
plt.savefig("dtree_iris_depth2_surface_v2.pdf")

# + [markdown] id="szbqxtLy1V0w"
# # Depth 3

# + colab={"base_uri": "https://localhost:8080/"} id="af6Lep1T1X8s" outputId="c911874a-98eb-4645-a1c0-d638d30f3dd0"
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X, y)



# + colab={"base_uri": "https://localhost:8080/", "height": 520} id="F7jaEWV11azu" outputId="054bc3d9-14c9-4469-ed29-b0eddf9e00f1"
export_graphviz(
        tree_clf,
        out_file= "iris_tree.dot",
        feature_names=xnames,
        class_names=ynames,
        rounded=True,
        impurity = False,
        filled=True
    )

Source.from_file("iris_tree.dot")

# + colab={"base_uri": "https://localhost:8080/", "height": 608} id="eJHigAzb1dD9" outputId="4d92d070-e67e-46f7-92b2-bd3e21f0f663"
plot_surface(tree_clf, X, y, xnames, ynames)

# + [markdown] id="wLturuH-Kcql"
# # Depth unrestricted

# + colab={"base_uri": "https://localhost:8080/"} id="p5bJENQTJDu4" outputId="05e2c26b-eae2-40fd-cbb8-39512b0b516b"

tree_clf = DecisionTreeClassifier(max_depth=None, random_state=42)
tree_clf.fit(X, y)

# + colab={"base_uri": "https://localhost:8080/", "height": 796} id="qgnp_RHYJIyq" outputId="38ffa159-0e83-4dd4-ea5b-a4439803be71"
from graphviz import Source
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file= "iris_tree.dot",
        feature_names=xnames,
        class_names=ynames,
        rounded=True,
        filled=False,
        impurity=False
    )

Source.from_file("iris_tree.dot")

# + colab={"base_uri": "https://localhost:8080/", "height": 608} id="5mlmxuKxJM7u" outputId="048915a4-f92a-4399-e3d8-8a346751383f"
plot_surface(tree_clf, X, y, xnames, ynames)

# + id="z2ibCZ6kJTaW"

