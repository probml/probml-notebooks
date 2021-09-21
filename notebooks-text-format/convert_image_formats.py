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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/convert_image_formats.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + colab={"base_uri": "https://localhost:8080/"} id="W7b2aANn6nXv" outputId="dbb0343c-7661-418e-c195-90f8a4a213bd"
# !pip install pdf2image

# + id="lIYdn1woOS1n"
import pdf2image

# + colab={"base_uri": "https://localhost:8080/"} id="aZot-9nDzZSh" outputId="fbad35c1-4632-4c98-bd03-88bd85b50944"
# !sudo apt-get install poppler-utils

# + id="uUfmQqvT3pML" colab={"base_uri": "https://localhost:8080/"} outputId="56e593a6-fb22-408e-98b3-2e7a0cba9959"
from google.colab import drive
drive.mount('/content/drive',force_remount=True)

# + id="tcFSkwAC08us"
# pip install pdf2image
# pip install --upgrade pillow

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
            #_, dir_name, file_name, _ =split_file_name(output_path)
            _, dir_name, file_name, _ =split_file_name(input_path)
            temp_file_name="temp"+file_name
            temp_file_path=os.path.join(dir_name,temp_file_name)
            print("converting ", input_path, " to ", temp_file_path)
            convert_from_path(input_path,output_file=temp_file_path,fmt="png",use_pdftocairo=True,single_file=True,
                              use_cropbox=True)
            temp_file_path+=".png"
            print("converting ", temp_file_path, " to ", output_path)
            _convert_profiles(temp_file_path,output_path,color_space=color_space,
                              input_profile_path=input_profile_path,output_profile_path=output_profile_path,quality=quality)
            os.remove(temp_file_path)
            return True
        elif input_path.endswith(".png") or input_path.endswith(".PNG") or \
            input_path.endswith(".jpg") or input_path.endswith(".JPG") or \
            input_path.endswith(".jpeg") or input_path.endswith(".JPEG") :
            return _convert_profiles(input_path,output_path,color_space=color_space,input_profile_path=input_profile_path,output_profile_path=output_profile_path,quality=quality)
        else:
            print(f"{input_path} is not a valid image file, copying it instead to {output_path}.")
            shutil.copy(input_path,output_path)
            return False
    except Exception as e:
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


# from https://pillow.readthedocs.io/en/stable/handbook/tutorial.html?highlight=cmyk#using-the-image-class
def check_image_properties(input_path):
    try:
        with Image.open(input_path) as im:
            print(input_path, im.format, f"{im.size}x{im.mode}")
    except OSError as e:
        print("error opening the image\n",e)



# + colab={"base_uri": "https://localhost:8080/"} id="JI2y1Pax7Zu4" outputId="e8a6cb4e-d8c2-4383-ff50-072f2028f1e9"
from glob import glob
files=glob("/content/drive/MyDrive/MLAPA/book-images-original/*.*")
p=[print(f) for f in files]

filenames = []
for f in files:
  parts = f.split("/")
  fname = parts[-1]
  base = fname.split(".")[:-1][0]
  #filenames.append(base)
  filenames.append(fname)

print(filenames)

# + id="iLrfkajD4GBx" colab={"base_uri": "https://localhost:8080/"} outputId="82700e09-9fec-43c3-c210-8ce8cacbe23c"
in_folder = "/content/drive/MyDrive/MLAPA/book-images-original"
for use_rgb in [False]:
  if use_rgb:
    out_folder = "/content/drive/MyDrive/MLAPA/book-images-rgb-80"
    color_space = "RGB"
    quality = 80
  else:
    out_folder = "/content/drive/MyDrive/MLAPA/book-images-cmyk-100"
    color_space = "CMYK"
    quality = 100

  rgb_profile = 'sRGB Color Space Profile.icm'
  cmyk_profile = 'USWebCoatedSWOP.icc'
  profile_folder = '/content/drive/MyDrive/MLAPA'
  input_profile_path = f'{profile_folder}/{rgb_profile}'
  if color_space == "RGB":
    output_profile_path = f'{profile_folder}/{rgb_profile}'
  else:
    output_profile_path = f'{profile_folder}/{cmyk_profile}'

  for fname in filenames:
    base = fname.split(".")[:-1][0]
    in_name = f'{in_folder}/{fname}'
    #in_name = f'{in_folder}/{fname}.pdf'
    out_name = f'{out_folder}/{base}.jpg'
    print('!converting ', in_name, ' to ', out_name)
    convert(in_name,
            out_name, 
            color_space=color_space, 
            quality=quality,
            verbose=True,
            input_profile_path=input_profile_path,
            output_profile_path=output_profile_path)


# + colab={"base_uri": "https://localhost:8080/"} id="wixLvxK5Astl" outputId="f0f29d99-f609-444e-b431-b73283a9956d"
# !ls /content/drive/MyDrive/MLAPA/book-images-original

# + id="ofDLPAvXAx2_" colab={"base_uri": "https://localhost:8080/"} outputId="5e7ee8d1-f290-481a-8019-0e0b80b489d5"
# !ls /content/drive/MyDrive/MLAPA/book-images-rgb-80

# + colab={"base_uri": "https://localhost:8080/"} id="IzQJPfb-KSzK" outputId="b44a5790-06fa-4c50-915b-517bb0bcb5ff"
# !ls /content/drive/MyDrive/MLAPA/book-images-cmyk-100

# + id="qyhK5xB2KWZx"

