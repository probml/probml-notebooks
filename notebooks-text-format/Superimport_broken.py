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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/Superimport_broken.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="_Qd2diMQZlQL"
# # Superimport demo that does not fully work
#
#
# The [superimport library](https://github.com/probml/superimport) is explained in [this colab](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/Superimport.ipynb#scrollTo=gERNSvUpcru-).
#
# Unfortunately sometimes when you call a script it fails to run, because inspecting the call stack raises an exception (which is [caught by superimport](https://github.com/probml/superimport/blob/e558c33157b32dfd86ad3c7b8d5d4784c6e00e0c/superimport/superimport.py#L163-L165), which prints the error message 'Error importing. Please re-run the cell.'
# Re-running the same cell usually fixes the error but is annoying.
#
#

# + id="1G16DyRzi6kK" colab={"base_uri": "https://localhost:8080/"} outputId="a1434ec0-fae7-4555-ae1e-ed97922631e3"
# !pip install superimport -qqq
# !pip install deimport -qqq

# + id="iiC72KXh5jae"
import superimport
from deimport.deimport import deimport

# + id="4DvNikcygYlC"
# !git clone --depth 1 https://github.com/probml/pyprobml  /pyprobml &> /dev/null 
# %cd -q /pyprobml/scripts


# + id="aOl69rgCgvVX" colab={"base_uri": "https://localhost:8080/", "height": 550} outputId="d774d352-b5f3-45fd-df3d-17be20077294"
deimport(superimport)
# %run linreg_residuals_plot.py

# + id="HpDO7eE1gz-Y" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="3eb974af-cdba-4256-b0b3-e9b41f9ca46c"
deimport(superimport)
# %run linreg_poly_vs_degree.py

# + id="UnKMPSBOh16k" colab={"base_uri": "https://localhost:8080/"} outputId="6c78763b-c6ff-4f11-ce83-9c7bba3e2314"
deimport(superimport)
# %run iris_kmeans.py

# + id="-PzpIRxdrK1r" outputId="2a526b24-33fa-44ea-f195-dca9021488e1" colab={"base_uri": "https://localhost:8080/", "height": 553}
deimport(superimport)
# %run iris_kmeans.py

# + id="2TBfyWOSrK-5"

