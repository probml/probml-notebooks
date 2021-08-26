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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/intro/intro_figures_matlab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + id="_X2Bi79r1rfW"

import cv2
from google.colab.patches import cv2_imshow
# Helper code to display images
def display_image(image,ratio):
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img=cv2.resize(img, (0,0), fx=ratio, fy=ratio) 
    cv2_imshow(img)
    print("\n")


# + colab={"base_uri": "https://localhost:8080/"} id="KpcxktL-1sov" outputId="5da0ee39-e66a-4df7-c518-a2d2f174e94a"

# !git clone https://github.com/probml/pyprobml/ 

# + colab={"base_uri": "https://localhost:8080/"} id="UDL7RkNf2i1_" outputId="41ed1bab-f0fa-458c-b58d-0e865f6b9be1"

# !apt install octave  -qq > /dev/null


# + id="glanfZME2op0"
# !apt-get install liboctave-dev -qq > /dev/null

# + colab={"base_uri": "https://localhost:8080/", "height": 997} id="6Xhr-JBg1826" outputId="a7555b76-0357-44e6-9bae-e5dca33c8f0f"

# %cd /content/pyprobml/scripts/matlab

# !octave -W '/content/pyprobml/scripts/matlab/maxGMMplot.m' >> _
display_image("./output1.jpg",0.4)
print("\n")
display_image("./output2.jpg",0.4)
# %cd /content/

# + colab={"base_uri": "https://localhost:8080/", "height": 540} id="TmDvRzML39TS" outputId="415cc016-6dcf-4867-964b-2ca44c6e5d50"

# %cd /content/pyprobml/scripts/matlab
# !octave -W '//content/pyprobml/scripts/matlab/regtreeSurfaceDemo.m' >> _
display_image("./output1.jpg",0.4)

# + colab={"base_uri": "https://localhost:8080/", "height": 475} id="ZLmyRu4I5cUK" outputId="f8d4f5cb-fa06-4e1e-aab8-f0a0d000ac2f"
# %cd /content/pyprobml/scripts/matlab

# This fails, since it needs 'load fisheriris', which is part of stats toolbox

# !octave -W '/content/pyprobml/scripts/matlab/dtreeDemoIris.m' >> _
display_image("./output1.jpg",0.4)
