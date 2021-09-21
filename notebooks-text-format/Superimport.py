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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/Superimport.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="_Qd2diMQZlQL"
# # Superimport demo
#
# The [superimport library](https://github.com/probml/superimport), written by [Mahmoud Soliman](https://github.com/mjsML), takes care of installing missing python packages for you. All you have to do is type `pip install superimport` (once per colab session), and then add `import superimport` to the top of any of your python files; then, when you run those files, superimport will read the source code, figure out any missing dependencies, install them for you automagically, and then run the rest of your code as usual. We illustrate this below. 
#
#

# + id="1G16DyRzi6kK" colab={"base_uri": "https://localhost:8080/"} outputId="b5baeea5-861e-4436-d126-0d6275a254b4"
# !pip install superimport -qqq
# !pip install deimport -qqq

# + id="iiC72KXh5jae"
import superimport

def try_deimport():
  try: 
    from deimport.deimport import deimport
    deimport(superimport,verbose=False)
  except Exception as e:
    print(e)



# + [markdown] id="V_82RC0lahoP"
# # An example with PgmPy
#
# Colab has most popular ML packages already installed. However, there are a few missing ones, such as [PgmPy](https://github.com/pgmpy/pgmpy). Below we create a short file, called `test.py`, that relies on that missing library. We then show what happens if we try to run the script  without first installing the library. 

# + colab={"base_uri": "https://localhost:8080/"} id="UuJYZu_1kP1B" outputId="1e0b7fc1-98a6-4d54-a92c-ecc378ea54cb"
# %%file test.py
import pgmpy
import numpy
import matplotlib
print('pgmpy ', pgmpy.__version__)

# + [markdown] id="mkwHEjfwknJt"
# Without importing superimport, if you have a missing package your script will fail.

# + colab={"base_uri": "https://localhost:8080/", "height": 215} id="6T6BDgaRkSdM" outputId="437179d6-4d6b-4f40-d570-1283f5108549"
# %run test.py

# + [markdown] id="cZd4ITLYa33l"
#
#
# Now we add one new line to our file: `import superimport`

# + colab={"base_uri": "https://localhost:8080/"} id="fMPOfeIHi9i3" outputId="b78a00c1-cd7b-41da-9287-cd86ea719b4e"
# %%file test.py
import superimport
import pgmpy
import numpy
import matplotlib
print('pgmpy ', pgmpy.__version__)


# + [markdown] id="aIFQaI6pkxIL"
# We can now successfully the script, and it will install any missing packages.
#
#
# Note, however, that we have to deimport the `superimport` symbol before running any code that uses superimport, to force the package to be reloaded (and hence re-executed), otherwise colab will use the cached version (if available) of superimport, which may be stale. 

# + colab={"base_uri": "https://localhost:8080/"} id="tL4_358-jJD8" outputId="a8fe754e-af56-400a-c663-39695ad4862b"
try_deimport()
# %run -n test.py

# + [markdown] id="-IDlmbuCgDMx"
# # An example with NumPyro
#
# This time we make a demo that uses numpyro, that is not installed in colab by default.

# + id="SuSqhaqEgQ3z" colab={"base_uri": "https://localhost:8080/"} outputId="682be66a-a277-44e9-b9b9-23c06c9975d1"
# %%file test.py
import superimport
import numpyro
print('numpyro version ', numpyro.__version__)

# + id="yHihouB-gUJK" colab={"base_uri": "https://localhost:8080/"} outputId="8787fa90-4332-42fd-da3c-6c4ce3657df0"

try_deimport()
# %run -n test.py

# + [markdown] id="KJSga2iNeauy"
# # An example with Pyro
#
# This time we make a demo that uses pyro, that is not installed in colab by default. Furthermore, its package name (pyro-ppl) does not match its import name (pyro).

# + id="Sy7eFOQxfQB6" colab={"base_uri": "https://localhost:8080/"} outputId="b16bde2c-d2ef-4cb2-9292-e2f99f70a3d9"
# %%file test.py
import superimport
import pyro
print('pyro version ', pyro.__version__)

# + id="wMgsJq1ieoeH" colab={"base_uri": "https://localhost:8080/"} outputId="9fb807c0-999f-41b8-aaae-4efdde5fe571"

try_deimport()
# %run -n test.py

# + [markdown] id="KFogkyP8gWZ5"
# # An example from the book

# + id="4DvNikcygYlC"
# !git clone --depth 1 https://github.com/probml/pyprobml  /pyprobml &> /dev/null 
# %cd -q /pyprobml/scripts


# + colab={"base_uri": "https://localhost:8080/", "height": 550} id="aOl69rgCgvVX" outputId="139d0c93-446e-4c9e-a264-11cd353fa1fe"

try_deimport()
# %run -n linreg_residuals_plot.py

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="HpDO7eE1gz-Y" outputId="3e2299dc-01c9-4738-b301-66a05458d3c5"

try_deimport()
# %run -n linreg_poly_vs_degree.py

# + colab={"base_uri": "https://localhost:8080/", "height": 553} id="UnKMPSBOh16k" outputId="5b77a9d2-c1cb-47a3-ea47-c899aea0f19f"

try_deimport()
# %run -n iris_kmeans.py

# + [markdown] id="gERNSvUpcru-"
# # Sharp edges
#
# * There are some packages whose install names differ from their import names  (eg we type `pip install pyro-ppl` but `import pyro`). There is a [public mapping file](https://github.com/bndr/pipreqs/blob/master/pipreqs/mapping) stored by pipreqs. However, this is missing some entries (such as pyro).  These must be manually added to the [mapping2 file](https://github.com/probml/superimport/blob/main/superimport/mapping2). If your favorite package is missing, open a PR on the superimport repo.
#
# * There are some packages that do not list of all of their requirements.txt (eg GPyOpt depends on matplotlib, but does not mention this). If this 'hidden requirement' is missing, superimport cannot find it either. If it is not already installed in colab, then your script will fail, even with superimport.

# + id="GUWz9Gr-d5SO"

