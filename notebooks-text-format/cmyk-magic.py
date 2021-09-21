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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/cmyk-magic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + colab={"base_uri": "https://localhost:8080/", "height": 310} id="lIYdn1woOS1n" outputId="07d63083-8c16-4379-a7b6-58f5926511d1"
import pdf2image

# + colab={"base_uri": "https://localhost:8080/"} id="qD6GDkgWypCM" outputId="71dd05c1-e15b-4556-eb47-80d2e58dadac"
# !pip install pdf2image

# + id="uUfmQqvT3pML" outputId="6784e2ef-c8c8-4af4-875c-b340a659ff50" colab={"base_uri": "https://localhost:8080/"}
from google.colab import drive
drive.mount('/content/drive')

# + id="CgRH6uKy3xkj" outputId="3b0b56a2-7e7e-401a-a99d-6b72a5d4fe89" colab={"base_uri": "https://localhost:8080/"}
# !ls /content/drive/MyDrive/MLAPA

# + id="D3DOb8JlyrjW"
from pdf2image import convert_from_path

# + colab={"base_uri": "https://localhost:8080/"} id="zRyb4TeezRiC" outputId="fc75478b-f8f2-46c8-c603-cf460e0e3108"
# !ls

# + colab={"base_uri": "https://localhost:8080/"} id="aZot-9nDzZSh" outputId="590f61c4-7ad3-48e5-d689-6243e008a3c8"
# !sudo apt-get install poppler-utils


# + id="Tizd6JxIyw_P"
input_path="/content/2dgridDAGa.pdf"; temp_file_path="/content/temp2dgridDAGa";
out = convert_from_path(input_path,output_file=temp_file_path, use_pdftocairo=True, fmt="png",single_file=True)


# + colab={"base_uri": "https://localhost:8080/"} id="RqJH9PDY0A5w" outputId="6dc45038-80c2-476c-9fbc-6cda41536d36"
# !ls

# + id="tcFSkwAC08us"
import os
import shutil
from pdf2image import convert_from_path
from PIL import Image
from PIL import ImageCms
import argparse
from glob import glob
from tqdm import tqdm
import functools
import multiprocessing
import concurrent.futures 





def split_file_name(input_path):
    base_name, dir_name = os.path.basename(input_path),os.path.dirname(input_path)
    file_name,ext = os.path.splitext(os.path.basename(base_name))
    return base_name, dir_name, file_name, ext

def convert(input_path,output_path,color_space="CMYK",input_profile_path=None,output_profile_path=None,quality=100,verbose=False,overwrite=False):
    """ converts an image or pdf into a color space of choice
        for CMYK the default output format is JPG
        Keyword arguments:
        input_path -- the input path of the file
        output_path -- the output path for the result to be written.
        color_space -- the color space to convert to , default value is CMYK
        input_profile_path -- the path to the input profile 
        output_profile_path -- the path to the output profile
    """
    try:
        if not overwrite and os.path.exists(output_path):
            return True
        
        if input_path.endswith(".pdf") or input_path.endswith(".PDF"):
            _, dir_name, file_name, _ =split_file_name(output_path)
            temp_file_name="temp"+file_name
            temp_file_path=os.path.join(dir_name,temp_file_name)

            print('input', input_path)
            print('output', temp_file_path)
            print('call convert ')
            
            #convert_from_path(input_path,output_file=temp_file_path,fmt="png",use_pdftocairo=True,single_file=True)
            convert_from_path(input_path,output_file=temp_file_path,fmt="png",single_file=True)
            temp_file_path+=".png"

            print(temp_file_path)
            
            _convert_profiles(temp_file_path,output_path,color_space=color_space,input_profile_path=input_profile_path,output_profile_path=output_profile_path,quality=quality)
            #os.remove(temp_file_path)
            print('done')
            return True
        elif input_path.endswith(".png") or input_path.endswith(".PNG") or \
            input_path.endswith(".jpg") or input_path.endswith(".JPG") or \
            input_path.endswith(".jpeg") or input_path.endswith(".JPEG") :

            print('else block')
            
            return _convert_profiles(input_path,output_path,color_space=color_space,input_profile_path=input_profile_path,output_profile_path=output_profile_path,quality=quality)
        else:
            print(f"{input_path} is not a valid image file, copying it instead to {output_path}.")
            shutil.copy(input_path,output_path)
            return False
    except Exception as e:
        print('exception')
        if verbose:
            print(f"Error in file: {input_path}\n",e)
        return False





def _convert_profiles(input_path=None,output_path=None,color_space="CMYK",input_profile_path=None,output_profile_path=None,quality="100"):
    try:
        with Image.open(input_path) as im:
            img_cmyk = ImageCms.profileToProfile(im, input_profile_path, output_profile_path, renderingIntent=0,outputMode=color_space)
            quality=int(quality)
            img_cmyk.save(output_path, quality=quality)
            
            return True
    except Exception as e:
        print(e)
        print(f"cannot convert{input_path}, copying it instead.")
        shutil.copy(input_path,output_path)
        return False


# + id="GL3t92sN4QQ2"
import os

# + id="iLrfkajD4GBx" outputId="a836f8a1-4408-4de2-e94c-87f0435e879a" colab={"base_uri": "https://localhost:8080/"}
input_path="/content/2dgridDAGa.pdf";
temp_file_path="/content/2dgridDAGa";
convert(input_path,
        temp_file_path, 
        color_space="RGB", 
        quality=80,
        verbose=True,
        input_profile_path='/content/drive/MyDrive/MLAPA/sRGB Color Space Profile.icm',
        output_profile_path='/content/drive/MyDrive/MLAPA/sRGB Color Space Profile.icm')


# + id="vQk45N3O474s" outputId="465c77a7-385e-47b5-fd43-5860f7f1a3c4" colab={"base_uri": "https://localhost:8080/"}
# !ls

# + id="tIQWPYQN5NLX"

