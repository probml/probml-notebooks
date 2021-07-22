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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/colab_intro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="24_OboyL7tqe"
# # Introduction to colab
#
# Kevin Murphy, June 2021.
#
# Colab is Google's version of Jupyter notebooks, but has the following advantages:
# - it runs in the cloud, not locally, so you can use it from a cheap laptop, such as a Chromebook. 
# - The notebook is saved in your Google drive, so you can share your notebook with someone else and work on it collaboratively.
# - it has nearly all of the packages you need for doing ML pre-installed
# - it gives you free access to GPUs
# - it has a [file editor](https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/colab_intro.ipynb#scrollTo=DdXlYCe1AlJa&line=7&uniqifier=1), so you can separate your code from the output of your code, as with other IDEs, such as [Jupyter lab](https://jupyterlab.readthedocs.io/en/stable/).
# - it has various other useful features, such as collapsible sections (cf. code folding), and ways to specify parameters to your functions via [various GUI widgets](https://colab.research.google.com/notebooks/forms.ipynb) for use by non-programmers. (You can automatically execute  parameterized notebooks with different parameters using [papermill](https://papermill.readthedocs.io/en/latest/).)
#
# More details can be found in the [official introduction](https://colab.research.google.com/notebooks/intro.ipynb). Below we describe a few more tips and tricks, focusing on methods that I have found useful when developing the book. (More advanced tricks can be found in [this blog post](https://amitness.com/2020/06/google-colaboratory-tips/) and [this blog post](https://medium.com/@robertbracco1/configuring-google-colab-like-a-pro-d61c253f7573#4cf4).)
#

# + colab={"base_uri": "https://localhost:8080/"} id="ZjFsGQJ41k32" outputId="ba36c760-3d87-4b96-ea62-0fe44d308d95"
IS_COLAB = ('google.colab' in str(get_ipython()))
print(IS_COLAB)


# + id="B4KQOCig_xf1"
# Standard Python libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import glob

from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

# + [markdown] id="8lAbDqny-vDq"
# # How to import and use standard libraries

# + [markdown] id="XHO2_uKXMbD4"
# Colab comes with most of the packages we need pre-installed. 
# You can see them all using this command.
#
#
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="0C54AJx40vJq" outputId="accb047b-6ac3-4e3c-97c5-6f3e1ccf7823"
# !pip list -v 

# + [markdown] id="U9PghW_NT1HY"
# To install a new package called 'foo', use the following (see [this page](https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb) for details):
#
# ```
# # # !pip install foo
# ```
#
#

# + [markdown] id="fOBdg02-_Jws"
# ## Numpy

# + id="AzP2LAtN_L1m" colab={"base_uri": "https://localhost:8080/"} outputId="cbdc24bf-ea76-4192-c17d-a6d97c14f322"
import numpy as np
np.set_printoptions(precision=3)

A = np.random.randn(2,3)
print(A)

# + [markdown] id="76jPgsuk_1IP"
# ## Pandas

# + id="GimloDqo_4No" colab={"base_uri": "https://localhost:8080/", "height": 197} outputId="1ddca722-d652-4b44-c4e0-982c722dabeb"
import pandas as pd
pd.set_option('precision', 2) # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100) # wide windows

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Year', 'Origin', 'Name']
df = pd.read_csv(url, names=column_names, sep='\s+', na_values="?")

df.head()

# + [markdown] id="hUCC261x_7zZ"
# ## Sklearn

# + id="RCSwx_lE_7Jn" colab={"base_uri": "https://localhost:8080/", "height": 285} outputId="4036c988-bb39-4ad4-d9af-f45bc6330231"
import sklearn

from sklearn.datasets import load_iris
iris = load_iris()
# Extract numpy arrays
X = iris.data 
y = iris.target

import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1])

# + [markdown] id="PJXF4csdBhsN"
# ## JAX

# + id="8JiSxcJJ79Bv" colab={"base_uri": "https://localhost:8080/"} outputId="97f3ebb7-93b8-4255-bccb-781d01ccb4cc"
# JAX (https://github.com/google/jax)

import jax
import jax.numpy as jnp
A = jnp.zeros((3,3))

# Check if JAX is using GPU
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))

# + [markdown] id="l99YLyorBdYE"
# ## Tensorflow

# + colab={"base_uri": "https://localhost:8080/"} id="StpReaSICLUm" outputId="5dfac104-8e5a-49cd-8400-eadb56650acd"

import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

print("tf version {}".format(tf.__version__))
print([d for d in tf.config.list_physical_devices()])

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. DNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

# + [markdown] id="grUUK1GrBfIY"
# ## PyTorch

# + id="Oi4Zmzla73A_" colab={"base_uri": "https://localhost:8080/"} outputId="558ed73a-5dfe-491a-be1c-64c949283e48"

import torch
import torchvision
print("torch version {}".format(torch.__version__))
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("Torch cannot find GPU")

# + [markdown] id="M4ayVFuc0FD9"
# # Plotting
#
# Colab has excellent support for plotting. We give some examples below.

# + [markdown] id="ENCT0EqifCDO"
# ## Static plots
#
# Colab lets you make static plots using matplotlib, as shown below.
# Note that plots are displayed inline by default, so
# ```
# # # %matplotlib inline
# ```
# is not needed.
#

# + colab={"base_uri": "https://localhost:8080/", "height": 295} id="j_k4tv4D1VaC" outputId="e1d38639-8a61-43dd-9c85-9091dc2d1f0d"
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(10))
plt.title('my plot')
plt.xlabel('x axis')
plt.savefig('myplot.png')

# + [markdown] id="eUCPm29t6VZf"
# ## Seaborn
#
# Seaborn is a library that makes matplotlib results look prettier. We can also update font size for plots, to make them more suitable for inclusion in papers.

# + id="1w9nRoF96cxy"
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
seaborn.set()
seaborn.set_style("whitegrid")

# Font sizes
SIZE_SMALL = 14
SIZE_MEDIUM = 18
SIZE_LARGE = 24

# https://stackoverflow.com/a/39566040
plt.rc('font', size=SIZE_SMALL)          # controls default text sizes
plt.rc('axes', titlesize=SIZE_SMALL)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE_SMALL)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE_SMALL)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE_SMALL)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE_SMALL)    # legend fontsize  
plt.rc('figure', titlesize=SIZE_LARGE)   # fontsize of the figure title


# + id="LjXEXYe17I2t" colab={"base_uri": "https://localhost:8080/", "height": 307} outputId="a7e2ae2d-67d9-4270-f981-40700c6a9f06"
plt.figure()
plt.plot(range(10))
plt.title('my plot')
plt.xlabel('x axis')
plt.savefig('myplot.png')

# + [markdown] id="iFwo8LA4fIh9"
# ## Interactive plots
#
# Colab also lets you create interactive plots using various javascript libraries - see [here](https://colab.research.google.com/notebooks/charts.ipynb#scrollTo=QSMmdrrVLZ-N) for details.
#
# Below we illustrate how to use the [bokeh library](https://docs.bokeh.org/en/latest/index.html) to create an interactive plot of a  pandas time series, where if you mouse over the plot, it shows the corresponding (x,y) coordinates. (Another option is [plotly](https://plotly.com/graphing-libraries/).)

# + id="SiKRxZyo3Fa1"
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, HoverTool

# Call once to configure Bokeh to display plots inline in the notebook.
output_notebook()


# + id="7_4CJ3mQhLNa" colab={"base_uri": "https://localhost:8080/", "height": 616} outputId="88eb0172-352f-46bd-8303-1baf898e0a79"



np.random.seed(0)
dates = pd.date_range(start='2018-04-24', end='2018-08-27')
N = len(dates)
vals = np.random.standard_t(1, size=N)
dd = pd.DataFrame({'vals': vals, 'dates': dates}, index=dates)
dd['days'] = dd.dates.dt.strftime("%Y-%m-%d")


source = ColumnDataSource(dd)
hover = HoverTool(tooltips=[("Date", "@days"),
                            ("vals", "@vals")],
)
p = figure( x_axis_type="datetime")
p.line(x='dates', y='vals', source=source)
p.add_tools(hover)
show(p)

# + [markdown] id="nHGH3o_R28G9"
# We can also make plots that can you pan and zoom into.

# + id="q1jy_cQl3AYc" colab={"base_uri": "https://localhost:8080/", "height": 616} outputId="ae8fc2f4-a7e5-48e9-f8f0-f9fd32331ded"
N = 4000
np.random.seed(0)
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = ["#%02x%02x%02x" % (r, g, 150) for r, g in zip(np.floor(50+2*x).astype(int), np.floor(30+2*y).astype(int))]

p = figure()
p.circle(x, y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)
show(p)

# + [markdown] id="FwIG5xw8ad8I"
# ## Viewing an image file
#
# You can either use PIL or OpenCV to display (and manipulate) images.
# According to [this notebook](https://www.kaggle.com/vfdev5/pil-vs-opencv), OpenCV is faster, but for a small number of images, it doesn't really matter.

# + id="AqitJZE1bAi8" colab={"base_uri": "https://localhost:8080/", "height": 514} outputId="6e8ebecb-c6ab-45c6-bb2b-782585a3d648"
from PIL import Image
import requests
from io import BytesIO
#url = "https://github.com/probml/probml-notebooks/blob/master/images/cat_dog.jpg?raw=true"
url = "https://raw.githubusercontent.com/probml/probml-notebooks/main/images/cat_dog.jpg"

response = requests.get(url)
img = Image.open(BytesIO(response.content))
print(type(img))
display(img)

# + id="t0EbaZr9bSZL"
# #!wget https://github.com/probml/probml-notebooks/blob/master/images/cat_dog.jpg?raw=true -q -O cat_dog.jpg
# !wget https://raw.githubusercontent.com/probml/probml-notebooks/main/images/cat_dog.jpg -q -O cat_dog.jpg


# + id="Hbb1T75kuaPG" outputId="3c50ec45-c0a7-4268-be21-b6a90fa90467" colab={"base_uri": "https://localhost:8080/"}
# !ls -l

# + id="64X5Xv_Kaf_Q" colab={"base_uri": "https://localhost:8080/", "height": 496} outputId="e6345129-8241-49fd-daa4-f9adf9c749f0"
from google.colab.patches import cv2_imshow
import cv2

def show_image(img_path,size=None,ratio=None):
  img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
  cv2_imshow(img)

show_image('cat_dog.jpg')

# + [markdown] id="y4_LLDBWGkE4"
# ## Visualizing arrays
#
# If you use imshow, be careful of aliasing which can occur for certain figure sizes.

# + id="grpgCI4IG2pb" colab={"base_uri": "https://localhost:8080/", "height": 182} outputId="5455f0f5-7109-49c3-fa64-14bbbbd9b662"
np.random.seed(0)
fig, axs = plt.subplots(1,3,figsize=(8,8))
for t in range(3):
  X = np.random.binomial(1, 0.5, (128, 128))
  axs[t].imshow(X, cmap="Accent")

# + [markdown] id="kKo7IkOzHMRo"
# You can solve this by specifying `interpolation=nearest`:

# + id="4aoE3HkuHQt-" colab={"base_uri": "https://localhost:8080/", "height": 182} outputId="93829355-ebb8-474a-a6ea-e61e18a8d437"
np.random.seed(0)
fig, axs = plt.subplots(1,3,figsize=(8,8))
for t in range(3):
  X = np.random.binomial(1, 0.5, (128, 128))
  axs[t].imshow(X, cmap="Accent", interpolation='nearest')

# + [markdown] id="K7MgGBEvHTlB"
# Alternatively, you can call `matshow`, which is an alias for imshow with `interpolation=nearest`:
#

# + id="M3nsqSA3HZxC" colab={"base_uri": "https://localhost:8080/", "height": 182} outputId="f7600549-b0f3-4fef-8c02-a276a8ea8274"
np.random.seed(0)
fig, axs = plt.subplots(1,3,figsize=(8,8))
for t in range(3):
  X = np.random.binomial(1, 0.5, (128, 128))
  axs[t].matshow(X, cmap="Accent")

# + [markdown] id="qLh3fxl63IHW"
# ## Graphviz
#
# You can use graphviz to layout nodes of a graph and draw the structure.

# + colab={"base_uri": "https://localhost:8080/"} id="33v9WeNjfb2k" outputId="2567027c-8c11-4517-94b8-3ae984b1d328"
# !apt-get -y install python-pydot
# !apt-get -y install python-pydot-ng
# !apt-get -y install graphviz

# + colab={"base_uri": "https://localhost:8080/", "height": 498} id="ZXTgGVkaffjf" outputId="2037bb22-88c1-4013-cf8e-03c8ebe349d5"
from graphviz import Digraph
dot = Digraph(comment='Bayes net')
print(dot)
dot.node('C', 'Cloudy')
dot.node('R', 'Rain')
dot.node('S', 'Sprinkler')
dot.node('W', 'Wet grass')
dot.edge('C', 'R')
dot.edge('C', 'S')
dot.edge('R', 'W')
dot.edge('S', 'W')
print(dot.source) 
dot.render('test-output/graph.jpg', view=True)
dot

# + [markdown] id="DaSY50JzpWnC"
# ## Progress bar
#

# + id="RNoYKY34Iqu6" colab={"base_uri": "https://localhost:8080/"} outputId="07194dcd-35de-47b0-b7b4-8158f83631c0"
from tqdm import tqdm
for i in tqdm(range(20)):
  x = np.random.randn(1000,1000)

# + [markdown] id="lyWwltzKlAHS"
# # Filing system issues
#
#
# Details here:
# - https://colab.research.google.com/notebooks/io.ipynb
# - https://neptune.ai/blog/google-colab-dealing-with-files
#
# Many other sources.

# + [markdown] id="7aCgO-moU2WA"
# ## Accessing local files
#
# Clicking on the file folder icon on the left hand side of colab lets you browse local files. Right clicking on a filename lets you download it to your local machine. Double clicking on a file will open it in the file viewer/ editor, which appears on the right hand side. 
#
# The result should look something like this:
#
# <img src="https://github.com/probml/pyprobml/blob/master/images/colab-file-editor.png?raw=true">
#
#
# You can also use standard unix commands to manipulate files, as we show below.

# + colab={"base_uri": "https://localhost:8080/"} id="dRh4BOIxHpEX" outputId="718b8efd-1988-4633-db29-6a414a350415"
# !pwd

# + colab={"base_uri": "https://localhost:8080/"} id="T7i8bvaghwy7" outputId="bb0012d2-bc82-4d5d-dfe3-acf2ba3e77a7"
# !ls

# + colab={"base_uri": "https://localhost:8080/"} id="HDNijfMPjPsE" outputId="d2bb4d90-86fc-44b2-9900-6050507e082b"
# !echo 'foo bar' > foo.txt
# !cat foo.txt

# + [markdown] id="5bnSVjwTr_bg"
# However, !cd does not work. You need to use the magic %cd.

# + id="voUDbPTUsDI3" colab={"base_uri": "https://localhost:8080/"} outputId="b8e3f19f-244a-407f-9fe7-94b64f60ef4b"
# !pwd
# !mkdir dummy
# %cd dummy
# !ls
# %cd ..

# + [markdown] id="AhyYxQ8mrzev"
# To make a new (local) file in colab's editor, first create the file with the operating system, and then view it using colab.

# + id="LtbGecLQrysw" colab={"base_uri": "https://localhost:8080/", "height": 17} outputId="17613167-0d6d-4f6e-b66f-129eb2552b6a"
from google.colab import files
file = 'bar.py'
# !touch $file 
files.view(file)

# + [markdown] id="PMet3XdcVF9O"
# If you make changes to a file containing code, the new version of the file will not be noticed unless you use the magic below.

# + id="0ufY8AO1VEUh"
# %load_ext autoreload
# %autoreload 2

# + [markdown] id="6a6nkLsKWQpu"
# ## Syncing with Google drive
#
# Files that you generate in, or upload to, colab are ephemeral, since colab is a temporary environment with an idle timeout of 90 minutes and an absolute timeout of 12 hours (24 hours for Colab pro). To save any files permanently, you need to mount your google drive folder as we show below. (Executing this command will open a new window in your browser - you need cut and paste the password that is shown into the prompt box.)
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="cYZpcMiQkl15" outputId="f4cf4b76-100a-428f-8c63-14aad3207235"
from google.colab import drive
drive.mount('/content/gdrive')
# !pwd



# + id="wycWRaVxPh5P" colab={"base_uri": "https://localhost:8080/"} outputId="5571799c-360b-446f-9f70-375f7d14527b"
with open('/content/gdrive/MyDrive/foo.txt', 'w') as f:
  f.write('Hello Google Drive!')
# !cat /content/gdrive/MyDrive/foo.txt

# + [markdown] id="CeGz5P_RUyzG"
# To ensure that local changes are detected by colab, use this piece of magic.

# + id="K6RR9dfoUyMG"
# %load_ext autoreload
# %autoreload 2

# + [markdown] id="_x4_RdSfh272"
# ## Uploading data to colab from your local machine

# + id="IgQe1zgYh4hF"
from google.colab import files
# GUI lets you select the file
# the return value is a dict, mapping filename to bytes
uploaded = files.upload()


# + [markdown] id="2mUVyvAblY3v"
# ## Downloading data from colab to your local machine

# + id="fwdVtkGqlblY"
from google.colab import files
files.download('checkpoints/gan-mlp-mnist-epoch=02.ckpt')

# + [markdown] id="bIqJtkFnlF8I"
# ## Loading data from the web into colab
#
# You can use [wget](https://www.pair.com/support/kb/paircloud-downloading-files-with-wget/) 
#
#

# + id="3oypH8Vclu86"
# !rm timemachine.*

# + colab={"base_uri": "https://localhost:8080/"} id="H0dvvsUclgdu" outputId="924597b5-451d-4eb3-d6dd-6e45763828dc"

# #!wget  https://github.com/probml/pyprobml/blob/master/data/timemachine.txt
# #!wget  https://github.com/probml/pyprobml/blob/master/data/timemachine.txt
# !wget https://raw.githubusercontent.com/probml/probml-data/main/data/timemachine.txt


# + colab={"base_uri": "https://localhost:8080/"} id="kj3fMYuwlyNR" outputId="f4a55c24-854b-4655-a48f-6284f10460a4"
# !head timemachine.txt

# + id="ZawrQx644OyW" outputId="dd6e15a0-71d0-4471-8fe0-06bf3248d144" colab={"base_uri": "https://localhost:8080/"}

datadir = '.'
import re
fname = os.path.join(datadir, 'timemachine.txt')
with open(fname, 'r') as f:
    lines = f.readlines()
    sentences = [re.sub('[^A-Za-z]+', ' ', st).lower().split()
                   for st in lines]
for  i in range(5):
  words = sentences[i]
  print(words)

# + [markdown] id="memojWlxuAyH"
# ## Loading code from the web into colab
#
# We can also download python code and run it locally.

# + id="siXx1f8et98t" colab={"base_uri": "https://localhost:8080/"} outputId="d0a115c5-3a81-4237-bf45-d1c8bd0d9d06"
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pyprobml_utils.py

import pyprobml_utils as pml
pml.test()

# + [markdown] id="57S7dQXbPSh6"
# ## Viewing all your notebooks
#
# You can see the list of colab notebooks that you have saved as shown below.

# + id="cTJGGK29PYM4" colab={"base_uri": "https://localhost:8080/"} outputId="dbb73578-2dd8-4f3c-f270-4dfbdedb60ff"
import re, pathlib, shutil
from pathlib import PosixPath

# Get a list of all your Notebooks
notebooks = [x for x in pathlib.Path("/content/gdrive/MyDrive/Colab Notebooks").iterdir() if 
             re.search(r"\.ipynb", x.name, flags = re.I)]
print(notebooks[:2])

#n = PosixPath('/content/gdrive/MyDrive/Colab Notebooks/covid-open-data-paper.ipynb')

# + [markdown] id="buZsxpmUS37n"
# # Working with github
#
# You can open any jupyter notebook stored in github in a colab by replacing
# https://github.com/probml/.../intro.ipynb with https://colab.research.google.com/github/probml/.../intro.ipynb (see [this blog post](https://amitness.com/2020/06/google-colaboratory-tips/#6-open-notebooks-from-github).
#
# It is possible to download code (or data) from githib into a local directory on this virtual machine.  It is also possible to upload local files back to github, although that is more complex. See details below.

# + [markdown] id="rVvGT6GUBg2Q"
# ## Cloning a repo from github
#
# You can clone a public github repo into your local colab VM, as we show below,
# using the repo for this book as an example.
# (To clone a private repo, you need to specify your password,
# as explained [here](https://medium.com/@robertbracco1/configuring-google-colab-like-a-pro-d61c253f7573#6b70). Alternatively you can use the ssh method we describe below.)

# + id="uVZWqzdW7_ZG" colab={"base_uri": "https://localhost:8080/"} outputId="cbae2714-3da4-405e-93d6-32fe1d4bbf4e"

# !rm -rf pyprobml # Remove any old local directory to ensure fresh install
# !git clone https://github.com/probml/pyprobml


# + id="sL0CLHTm7HSH" colab={"base_uri": "https://localhost:8080/"} outputId="d23ce0d0-3403-4bfd-e666-34f715da765a"
# !pwd

# + colab={"base_uri": "https://localhost:8080/"} id="XdC34HzKT8L8" outputId="49d5d474-7af0-4a4d-adb3-3a11b049f3e3"
# !ls

# + [markdown] id="MNWWINngc5rn"
# We can run any script as shown below.
# (Note we first have to define the environment variable for where the figures will be stored.)

# + colab={"base_uri": "https://localhost:8080/", "height": 873} id="aYXkQP-DdApw" outputId="50b98443-467b-4ab9-ee6c-773cc80530ab"
import os
os.environ['PYPROBML']='pyprobml'

# %run pyprobml/scripts/activation_fun_plot.py

# + id="qItn37RW7R3N" colab={"base_uri": "https://localhost:8080/"} outputId="903c2f05-dafb-4db6-d5b9-9874a3b0cc1d"
# !ls pyprobml/figures

# + [markdown] id="YTT5eJ_qUDFe"
# We can also import code, as we show below.

# + colab={"base_uri": "https://localhost:8080/"} id="0jUdrHWLd95C" outputId="a6ba93bc-118c-42e7-c894-34eeef80c95b"
# !ls

# + colab={"base_uri": "https://localhost:8080/"} id="-NkBCWPdePFj" outputId="b33684a4-cd62-46fb-ab15-ef650f12d5c2"

import pyprobml.scripts.pyprobml_utils as pml
pml.test()

# + [markdown] id="yaISmcnNmnS7"
# ## Pushing local files back to github
#
# You can easily save your entire colab notebook to github by choosing 'Save a copy in github' under the File menu in the top left. But if you want to save individual files (eg code that you edited in the colab file editor, or a bunch of images or data files you created), the process is more complex.
#
# There are two main methods. You can either specify your username and password every time, as explained [here](https://medium.com/@robertbracco1/configuring-google-colab-like-a-pro-d61c253f7573#6b70). Or you can authenticate via ssh. The latter is more secure, but more complex, as we explain below.
#
# You first need to do some setup to create SSH keys on your current colab VM (virtual machine), manually add the keys to your github account, and then copy the keys to your mounted google drive so you can reuse the same keys in the future. This only has to be done once.
#
# After setup, you can use the `git_ssh` function we define below to securely execute git commands. This works by copying your SSH keys from your google drive to the current colab VM, executing the git command, and then deleting the keys from the VM for safety. 
#

# + [markdown] id="0gOzFmcKoUuO"
# ### Setup
#
# Follow these steps. (These instructions are text, not code, since they require user interaction.)
#
# ```
# # # !ssh-keygen -t rsa -b 4096
# # # !ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
# # # !cat /root/.ssh/id_rsa.pub
# ```
# The cat command will display your public key in the colab window.
# Cut and paste this and manually add to your github account following [these instructions](https://github.com/settings/keys).
#
# Test it worked
# ```
# # # !ssh -T git@github.com
# ```
#
# Finally, save the generated keys to your Google drive
#
# ```
# from google.colab import drive
# drive.mount('/content/drive')
# # # !mkdir /content/drive/MyDrive/ssh/
# # # !cp  -r  ~/.ssh/* /content/drive/MyDrive/ssh/
# # # !ls /content/drive/MyDrive/ssh/
# ```
#

# + [markdown] id="oSiVyBG1xm44"
# ### Test previous setup
#
# Let us check that we can see our SSH keys in our mounted google drive.

# + colab={"base_uri": "https://localhost:8080/"} id="cCUxHiHAxcY2" outputId="9555ac68-2dc8-4b9a-9049-b8174ddfbf4c"
from google.colab import drive
drive.mount('/content/drive')

# !ls /content/drive/MyDrive/ssh/



# + [markdown] id="C-lPchgDpD7t"
# ### Executing git commands from colab via SSH
#
# The following function lets you securely doing a git command via SSH.
# It copies the keys from your google drive to the local VM, excecutes the command, then removes the keys.
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="ONJ0Ump4wEho" outputId="ef47e2d3-18fd-4917-c9c5-77315c4fd0b6"
# !rm -rf pyprobml_utils.py # remove any old copies of this file
# !wget https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pyprobml_utils.py  

# + id="FJ1_fCk_K9Mc" colab={"base_uri": "https://localhost:8080/"} outputId="0bd007b2-09c0-4cb1-e295-6f734cf70252"
from google.colab import drive
drive.mount('/content/drive') # must do this before running git_ssh
import pyprobml_utils as pml # import script into namespace

# + [markdown] id="umvOwzMfvpmU"
# Below we clone the pyprobml repo to this colab VM using out github credentials, so we can later check stuff back in. **This is just an example - you should edit the `reponame`, `username` and `email` variables.***

# + colab={"base_uri": "https://localhost:8080/"} id="IbX9-PDpwlO0" outputId="957ae2fd-cab7-4267-8f4d-9709d6554910"


# !rm -rf pyprobml # remove any old copies of this directory
# #!git clone https://github.com/probml/pyprobml.git # clones using wrong credentials
pml.git_ssh("git clone https://github.com/probml/pyprobml.git",
            mail="murphyk@gmail.com", username="probml") # update to use your credentials



# + id="ir4nBvPfLXwF" colab={"base_uri": "https://localhost:8080/"} outputId="00fcd067-3f7f-4ee1-9c54-979b6d9722ce"

reponame = 'pyprobml'
username = 'probml'
email = 'murphyk@gmail.com' # update to use your credentials

# !rm -rf $reponame # remove any old copies of this directory
cmd = f"git clone https://github.com/{username}/{reponame}.git"
pml.git_ssh(cmd, email=email, username=username) 



# + [markdown] id="4-1ZQyFUMq9A"
# Let's check that we can see this repo in our local drive.

# + id="gB5Lx38aMupR" colab={"base_uri": "https://localhost:8080/"} outputId="9c26d870-64bd-4589-b4ec-2607815d8ee5"
# !pwd
# !ls


# + id="od3MymSWNHY6" colab={"base_uri": "https://localhost:8080/"} outputId="02b156ec-c917-4955-bd77-f04177e94abd"
# !ls /content/$reponame/

# + [markdown] id="ft_pJJ4ZTLdl"
# Now we create a dummy file inside our local copy of this repo, and push it back to the  github (public) version of the repo.
#

# + colab={"base_uri": "https://localhost:8080/"} id="3hfluZVYTSNd" outputId="400a5892-fd9d-44ca-e67c-dfdb572a5c00"

# Make the dummy file in the scripts folder of repo
# %cd /content/$reponame
# !echo 'this is a test' > scripts/foo.txt

# Add file to the external repo
cmd = "git add scripts; git commit -m 'push from colab'; git push"
pml.git_ssh(cmd, email=email, username=username)


# + [markdown] id="l8MHjRl5Ptoi"
# We can check that it worked by visiting [this page](https://github.com/probml/pyprobml/blob/master/scripts/foo.txt) on github (note the time stamp on the top right):
#
# <img src="https://github.com/probml/pyprobml/blob/master/images/github-colab-commit-foo.png?raw=true" height=300>
#

# + [markdown] id="DAqqAhzzTS7F"
#
# Finally we clean up our mess.

# + colab={"base_uri": "https://localhost:8080/"} id="r16zYiNTu_Mz" outputId="d10b7525-d203-4622-b0e5-6b7f1f894529"

# %cd /content/$reponame
cmd = "git rm scripts/foo*.txt; git commit -m 'colab cleanup'; git push"
pml.git_ssh(cmd, email=email, username=username, verbose=True)
# %cd /content

# + [markdown] id="q-kRtmdm5d7X"
# # Software engineering tools
#
#  [Joel Grus has argued](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/edit) that notebooks are bad for developing complex software, because they encourage creating monolithic notebooks instead of factoring out code into separate, well-tested files. 
#  
# [Jeremy Howard has responded to Joel's critiques here](https://www.youtube.com/watch?v=9Q6sLbz37gk&feature=youtu.be). In particular, the FastAI organization has created [nbdev](https://github.com/fastai/nbdev) which has various tools that make notebooks more useful.
#
#

# + [markdown] id="KoYhOfdLJOB5"
# ## Argparse
#
# Often code is designed to be run from the command line, and can be configured by passing in arguments and flags. To make this work in colab, you have to use `parse_known_args`, as in  the example below.
#

# + id="B-zBHHR0Ja-C" colab={"base_uri": "https://localhost:8080/"} outputId="a20c460e-76ee-46eb-c840-bddcf4059921"
def main(args):
  print('my awesome function')
  print(args.arg1)
  print(args.arg2)

import argparse
parser = argparse.ArgumentParser(description='My Demo')
parser.add_argument("-arg1", default=1, type=int,  help="An integer to print")
parser.add_argument("-arg2", "--argument2", default="foo", help="A string to print")
parser.add_argument("-f", "--flag", action="store_true", help="Just a flag")

#args = parser.parse_args() # error in colab
args, unused = parser.parse_known_args()

print(args)
print(unused)


# + id="lJhFvBfpJvzX" colab={"base_uri": "https://localhost:8080/"} outputId="263244ba-1e87-4cca-9d45-f703e02bad8e"
args.arg1 = 42
args.arg2 = 'bar'
main(args)

# + id="WY8-ToefJxjU" colab={"base_uri": "https://localhost:8080/"} outputId="b5c789d2-f813-409f-89b5-23c266e2cc89"
args.arg1 = 49
args.arg2 = 'foo'
main(args)

# + [markdown] id="I9YU0L_lKQIc"
# ## YAML files
#
# We show how to create a config file locally, and then pass it to your code.

# + id="cdbgSjt7Kaiv" colab={"base_uri": "https://localhost:8080/"} outputId="9782c31d-bf49-4613-c320-23eefc1032cd"
# %%writefile myconfig.yaml
model_params:
  name: 'VanillaVAE'
  in_channels: 3
  latent_dim: 128

exp_params:
  dataset: celeba
  data_path: "../../shared/Data/"

# + id="9GMLwuvxK8d6" colab={"base_uri": "https://localhost:8080/"} outputId="023cc596-1f65-4416-e208-f8d4ed3558bc"
# !cat myconfig.yaml

# + id="BQQvL8hZLn75" colab={"base_uri": "https://localhost:8080/", "height": 17} outputId="dd022703-d610-4782-d104-b59d1aa9254e"
from google.colab import files
file = 'myconfig.yaml'
# #!touch $file 
files.view(file)

# + id="MFnxc7WGLven" colab={"base_uri": "https://localhost:8080/"} outputId="952c0a8e-5c3d-4c1d-c2fb-1cea85cf58f7"
# !cat myconfig.yaml

# + id="KMw1vYZ1KSLU" colab={"base_uri": "https://localhost:8080/"} outputId="3e5ec991-e4f4-458d-9076-7281a1a98cf4"
import yaml

filename = 'myconfig.yaml'

with open(filename, 'r') as file:
  config = yaml.safe_load(file)

print(type(config))
print(config)
print(config['model_params']['in_channels'])

# + [markdown] id="PuSsmj_fZ106"
# ## Avoiding problems with global state
#
# One of the main drawbacks of colab is that all variables are globally visible, so you may accidently write a function that depends on the current state of the notebook, but which is not passed in as an argument. Such a function may fail if used in a different context.
#
# One solution to this is to put most of your code in files, and then have the notebook simply import the code and run it, like you would from the command line. Then you can always run the notebook from scratch, to ensure consistency.
#
# Another solution is to use the [localscope](https://localscope.readthedocs.io/en/latest/README.html) package can catch some of these errors.
#
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="Q0FmEeIgc0YI" outputId="0d71c4a8-a4fb-45bd-e137-c391e408c83f"
# !pip install localscope


# + id="9zfUiUB8d-jh"
from localscope import localscope

# + colab={"base_uri": "https://localhost:8080/"} id="wI5wXzUPdlOS" outputId="500dfc1f-1cbc-4fa6-a30e-02f32258fa31"
a = 'hello world'
def myfun():
   print(a) # silently accesses global variable

myfun()

# + colab={"base_uri": "https://localhost:8080/", "height": 337} id="T3V1iV32czq8" outputId="b346cb7a-f252-41bc-992b-dd72e5aec825"
a = 'hello world'
@localscope
def myfun():
  print(a)

myfun()


# + colab={"base_uri": "https://localhost:8080/", "height": 337} id="5t48_AMbAN8V" outputId="71fdaacf-4650-4a52-9fb5-da5873c33e73"
def myfun2():
  return 42

@localscope
def myfun3():
  return myfun2()

  

# + colab={"base_uri": "https://localhost:8080/"} id="DAqLZ8PdAquy" outputId="021e8282-a283-414c-ef1d-46521da89e49"
@localscope.mfc # allow for global methods, functions, classes
def myfun4():
  return myfun2()

myfun4()

# + [markdown] id="iZaeVouoAhXP"
# ## Factoring out functionality into files stored on github
#
# The recommended workflow is to  develop your code in the colab in the usual way, and when it is working, to factor out the core code into separate files. You can  edit these files locally in the colab editor, and then push the code to github when ready (see details above). To run functions defined in a local file, just import them. For example, suppose we have created the file /content/pyprobml/scripts/fit_flax.py; we  can use this idiom to run its test suite:
# ```
# import pyprobml.scripts.fit_flax as ff
# ff.test()
# ```
# If you make local edits, you want to be sure
#  that you always import the latest version of the file (not a cached version). So you need to use this piece of colab magic first:
# ```
# # # %load_ext autoreload
# # # %autoreload 2
# ```
#

# + [markdown] id="DdXlYCe1AlJa"
# ## File editors
#
# Colab has a simple file editor, illustrated below for an example file.
# This lets you separate your code from the output of your code, as with other IDEs, such as [Jupyter lab](https://jupyterlab.readthedocs.io/en/stable/).
#
# <img src="https://github.com/probml/probml-notebooks/raw/main/images/colab-file-editor.png">
#
#
#

# + [markdown] id="LgLmSpaHBxLu"
# You can click on a class name when holding Ctrl and the source code will open in the file viewer. (h/t [Amit Choudhary's blog](https://amitness.com/2020/06/google-colaboratory-tips/).
#
# <img src="https://github.com/probml/probml-notebooks/raw/main/images/colab-goto-class.gif">
#
#

# + [markdown] id="4qi2xKMbAnlj"
#
# ## VScode
# The default colab file editor is very primitive.
# See [this article](https://amitness.com/vscode-on-colab/) for how to run VScode
# from inside your Colab browser. Unfortunately this is a bit slow. It is also possible to run VScode locally on your laptop, and have it connect to colab via SSH, but this is more complex (see [this blog post](https://amitness.com/vscode-on-colab/) or [this medium post](https://medium.com/@robertbracco1/configuring-google-colab-like-a-pro-d61c253f7573#4cf4) for details).
#
#

# + [markdown] id="7KrRcbQ71ZyZ"
# # Hardware accelerators
#
# By default, Colab runs on a CPU, but you can select GPU or TPU for extra speed, as we show below. To get access to more powerful machines (with faster processors, more memory, and longer idle timeouts), you can subscript to [Colab Pro](https://colab.research.google.com/signup). At the time of writing (Jan 2021), the cost is $10/month (USD). This is a good deal if you use GPUs a lot. 
#
# <img src="https://github.com/probml/probml-notebooks/raw/main/images/colab-pro-spec-2020.png" height=300>
#
#
#

# + [markdown] id="Qpb9N-c-3RSf"
# ## CPUs
#
# To see what devices you have, use this command.
#

# + colab={"base_uri": "https://localhost:8080/"} id="SZYIR1Kk1ktp" outputId="04ab4999-74d4-47d6-f9cb-004ca13b590b"
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# + colab={"base_uri": "https://localhost:8080/"} id="hs_l1zUY3UuQ" outputId="99a7db7f-6325-452e-ca69-ab7439070d67"
# !cat /proc/version

# + colab={"base_uri": "https://localhost:8080/"} id="WHG13Iwa3Z5k" outputId="ebc74028-163e-48e2-a0c6-56667e4e0c0d"
from psutil import cpu_count

print('num cores', cpu_count())

# !cat /proc/cpuinfo


# + [markdown] id="PivaBI5za45p"
# ## Memory
#

# + colab={"base_uri": "https://localhost:8080/"} id="xg3C0Yrq3job" outputId="0b480191-8489-4b6c-fc26-c6f3b51aecc2"
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('RAM (GB)', ram_gb)


# + colab={"base_uri": "https://localhost:8080/"} id="W_I7m_hUbBpu" outputId="b688a7d0-1c53-4864-c79f-fde42cda4db6"

# !cat /proc/meminfo

# + [markdown] id="v0G2d13kIEz5"
# ## GPUs
#
# If you select the 'Runtime' menu at top left, and then select 'Change runtime type' and then select 'GPU', you can get free access to a GPU. 
#
#
#
#

# + [markdown] id="MGVZB0esI0QG"
#
# To see what kind of GPU you are using, see below.
#

# + id="FikkXWQqBU9O" colab={"base_uri": "https://localhost:8080/"} outputId="ba7b6c8e-5089-4342-f9f4-def0e5ad86a8"
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
print(gpu_info)

# + colab={"base_uri": "https://localhost:8080/"} id="GU6nII1F5S2S" outputId="d220a1a6-cc83-4718-c77e-2df38f13bf9c"
# !grep Model: /proc/driver/nvidia/gpus/*/information | awk '{$1="";print$0}'

# + id="hzi1OpMaZlAc"

