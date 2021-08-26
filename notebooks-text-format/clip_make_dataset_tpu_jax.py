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
# <a href="https://colab.research.google.com/github/always-newbie161/pyprobml/blob/hermissue122/notebooks/clip_make_dataset_tpu_jax.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="oSvmK72aXX2n"
# ### Required Installations and Environment

# + id="PstTJNSF0MEp"
import os
assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'

# + colab={"base_uri": "https://localhost:8080/"} id="ryY4kFg52TgI" outputId="2b140ff2-ad3b-44e9-cbc7-a709a0392a45"
import os
if 'google.colab' in str(get_ipython()) and 'COLAB_TPU_ADDR' in os.environ:
  import jax
  import jax.tools.colab_tpu
  jax.tools.colab_tpu.setup_tpu()
  print('Connected to TPU.')
else:
  print('No TPU detected. Can be changed under "Runtime/Change runtime type".')

# + colab={"base_uri": "https://localhost:8080/"} id="Hvh5IvW71E1z" outputId="528bfaf5-5e90-4f0e-db30-e0a63756ab64"
import jax
print("jax version {}".format(jax.__version__))
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))
print(jax.lib.xla_bridge.device_count())
print(jax.local_device_count())

import jax.numpy as jnp
devices = jax.local_devices()
print(f"jax devices:")
devices

# + [markdown] id="GUUWoVQ70MEt"
# ### Cloning Clip_jax
#
# and loading the jax version of clip_model.
#

# + colab={"base_uri": "https://localhost:8080/"} id="ncTrhbWk1SVc" outputId="d110413c-5119-4bb5-825a-2d487fa2db3d"
# %cd /content/

# + colab={"base_uri": "https://localhost:8080/"} id="AG6JFca30MEu" outputId="d25d2413-ef8e-4d54-a99a-57db7f5eb291"
# !git clone https://github.com/kingoflolz/CLIP_JAX.git

# + colab={"base_uri": "https://localhost:8080/"} id="ESGpTKuj0MEu" outputId="f3247e38-d834-469b-cf9b-6ad19264bdf4"
# cd /content/CLIP_JAX

# + id="Pw293UN0xCpS" colab={"base_uri": "https://localhost:8080/"} outputId="9509a82a-d4a1-4e32-e4c2-d08322172a13"
pip install ftfy regex tqdm dm-haiku

# + colab={"base_uri": "https://localhost:8080/"} id="TRvaiA7i0meM" outputId="6f92a4ae-8b49-453a-b633-ccd56523ec94"
import numpy as np
from PIL import Image
import time

import clip_jax

image_fn, text_fn, jax_params, jax_preprocess = clip_jax.load('ViT-B/32', "cpu", jit=True)


# + [markdown] id="w3SufV5j1p2I"
# pmapping the encoding function and replicating the params.

# + id="mp7wqJU81pNz"
jax_params_repl= jax.device_put_replicated(jax_params, devices)
image_fn_pmapped = jax.pmap(image_fn)

# + [markdown] id="t3ANNd59UKi2"
# ## Dataset

# + [markdown] id="wknyV_ByURRQ"
# **Download the dataset used here** so that it just loads the downloaded dataset when used later.
#
# Change `ds_name` to the dataset required.

# + id="79aR97FZXBH6"
ds_name = 'imagenette/160px-v2'

# + id="d-nuSvkHpR0o"
data_dir = '/root/tensorflow_datasets'

# + id="QruQPROQFA3_" cellView="form"
#@title Choose whether if you want to make a copy of the dataset in the drive
#@markdown Drive can be mounted to download the tfds into the drive for future uses, 
#@markdown downloaded ds can be found in `your_drive_path/MyDrive/$ds_name`

to_load_into_drive = False #@param ["False", "True"] {type:"raw"}

if to_load_into_drive:
  from google.colab import drive
  drive.mount('/content/drive')

  # !mkdir /content/drive/MyDrive/$ds_name # your_drive_path

  data_dir = f'/content/drive/MyDrive/{ds_name}'

# + [markdown] id="Do9GiFpdYCl9"
# ### Loading tfds

# + colab={"base_uri": "https://localhost:8080/", "height": 531, "referenced_widgets": ["5235284b323344a3bc87dbd7412d585f", "72682b11182247f4962e1f31559ca2aa", "1cad5b916e3b4fd3aa058bf3bc589fa2", "13cb91a4595b40878283a5049569bd92", "b61925834adb408fb6240e46cf3ddcbf", "e67ee6b11701428bb977d3999b19c35d", "a58829851ec34bea9ff2e74026d64f7c", "a191d81814784ca58276c029c3264980", "c46f2df5c43d4003bb8dcca0fcba4039", "11650ea55ff244a197b1b4c982aa29cd", "1af2d877748f40a3a587334316609ee1", "492c0065d9e84badafdc41ea4a08aec4", "22c112075dc1484e839e88fce0638e72", "4359abab24014c56be26d0e3e3d5e951", "007d83f29d884a998d3565b071a31cf6", "e8a15d64c07341d9adbbd5bf61f08361", "dfe94ad7b0b44e9e8809711aa96aae32", "de6a89839fa848209ed0f753ab910c2a", "386eaa89254f4c6faadaf7347e265d69", "9ba09916ae9343169c889e69a4740828", "a8867af0df654da4b651e79a1ed75eae", "52970cf2080d4a3dbbce739117d0432b", "15b8b6378a1c4bf3ae09098d5a5b86e7", "e3df81579d3b4a1eb1af7070ad010097", "07008eb4b1a74a2fa272bc8286d6030d", "715f1a1cc8dd4197b4370d397317da74", "14bcb41516534635bc8ab147e37dce6d", "ac1f6ca3460a4eff8bacc196ec011e9c", "4a4004479d42411d9068cf1df1425a62", "9f7e72b88f4e4fdd9a2d388409394eff", "d77702c55ba84ccb8d18f7862d0e1599", "1039d5f8ae6c433a8b96fc031cd4549c", "0ae4c5087f8547439b078590c5550d36", "3222e47384374b239bd9ab2f01b53a23", "5d99b895405f4065a27050777ccf6e2a", "e266e69a9bbf4770aef5bc6fadf94435", "38858bbcddb1409788e8c466a498b569", "6374878ad5df47d989a429728de1e1d5", "d79a3f61b0fc471593eb35a94020a8f6", "6e84dd8f052443a6bf9caaa19ff0a031", "98be7a978fb248129affb5183b3d6271", "44dfcd77ffad40ac8afdf25e18558e80", "e1fbade420a94708a4671d06b23050c7", "e4e94b3da2484d3f9b196042ac295ede", "7ec8d585a3b74a2a907c2fe2ed1c8500", "3a7e071dfa374e8487f5a79a3d5ff500", "4b12be8def7949f09265e6558e35a7d6", "c6fcd5ae1c114c3f872c06286e3b6dc1", "91d860f5f7f94878ba0d8cef97eec8b1", "65edfaaee42e41a9abe47f36e6a267e0", "4d0d81f7436243f59044848faf6e0247", "b2cfc6a54fa64a55af0aab0b0a92d0fc", "e4146ebc55424835910b2d19668cc735", "773e297ca41044f296852f7a29760a17", "dba1a1d22e5149569911c17c93cbc36f", "ce0f1980f8b4485299826050c3fab4f4", "1b9b7519afa9424fb779674a8a7a35a4", "34708342fad4491bb0bb89ff9d21579a", "8a3a6b8e6d314efcb53d1c26f41cde6d", "bf8f2bf0b93f4390ba4a82a5237e0cec", "f0834bd967654528b3fcc2a4bbbd94e5", "58db18eb6a4a423d813f19685128bade", "9d02e5a6d4704446b7cde90137ba080e", "aa68462dbc054c5aa9aa2630c6204df1", "3ed7a642466249a9b5bcbe37468486a1", "28e558285ebd42d6842f6e3c1af5426e", "eac45fbaf3de4767911176ad3c3fbbcc", "8f42966fee084c11b56a5f1a7d9b4f5b", "a50e66e00cac4781b297d72210bafa52", "b8c53d08450d45c195515165d68b3aab", "460cab60adce4c0cbca1c6479319aed9", "a95de664278e46a4b27e118d6b282204", "97d46374e83c43e6ab6d11562d5fba97", "e54ba71b0db84fe78585e0ef89ca1d8e", "d138360b0faf4a318fbfca6d3c7b2ca8", "56bbb212c72b4ed9a86bb2ad6b559f79", "7fc137c5d8224b2792d4e62b693e1f5c", "221381b4575c4e7eb0fda813a633ef82", "94e1f18c994b49539f2bfb13f1e4a7e7", "d65d75e56b1b45c98a525835487e6cac"]} id="ubMMS0XzUNCo" outputId="73fe3339-f5c6-4b11-bf1c-2a78c9d1982b"
import tensorflow as tf
import tensorflow_datasets as tfds

try:
  tfds.load(ds_name, data_dir=data_dir)
except:
  tfds.load(ds_name, data_dir=data_dir)

# + [markdown] id="NL3E44tS0Dcv"
# ## Model

# + colab={"base_uri": "https://localhost:8080/"} id="9EC15b39p_tG" outputId="890cb1cb-3d47-4e88-dee1-3abc26a12d23"
len(devices)


# + [markdown] id="myd-XDIe1UwY"
# Datamodule which makes the numpy dataloaders for the dataset that return batches such that their leading dimension is len(devices)

# + id="y5gzwVSNduJj"
class Tpu_data_loader():
  def __init__(self, loader, split, batch_per_core, no_of_cores):
    self.loader = loader
    self.split = split
    self.batch_size = batch_per_core*no_of_cores

class NumpyDataModule():

    def __init__(self, ds_name: str, data_dir: str):
        self.ds_name = ds_name
        self.data_dir = data_dir
        self.image_size = 224
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.48145466, 0.4578275, 0.40821073]
        self.ds=None

    def preprocess(self, sample):
        image = sample['image']
        """ `uint8` -> `float32`."""
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, self.image_size, self.image_size)
        image = (image-self.mean)/(self.std)
        image = tf.transpose(image, perm = [2, 0 ,1])
        return image
    

    def make_dataset(self, split, batch_per_core, no_of_cores):
        ds = self.ds[split]
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_per_core).batch(no_of_cores)
        return Tpu_data_loader(tfds.as_numpy(ds.prefetch(tf.data.experimental.AUTOTUNE)), split, batch_per_core, no_of_cores)

    def prepare_data(self):
        self.ds, ds_info = tfds.load(
            self.ds_name,
            with_info=True,
            data_dir = self.data_dir,
        )
        return ds_info



# + id="wvfuR3kMwZ2b"
dm = NumpyDataModule(ds_name= ds_name, data_dir=data_dir)

ds_info = dm.prepare_data()

# + [markdown] id="v4NS_YTuiAIX"
# `batch_per_core` should be such that `(n_examples//batch_per_core) % no_of_cores == 0`

# + id="E4x06_6BfCHp"
train_loader = dm.make_dataset('train', batch_per_core=62, no_of_cores=len(devices))
test_loader = dm.make_dataset('validation', batch_per_core=61, no_of_cores=len(devices))

# + colab={"base_uri": "https://localhost:8080/"} id="jF94btwZkSyK" outputId="820632a2-cce7-4b13-f3d0-8b9aebd79383"
print(ds_info.splits[train_loader.split].num_examples)
print(ds_info.splits[test_loader.split].num_examples)

# + id="tom27i1RHh7A"
import tqdm

def clip_extract(tpu_loader):

  clip_features = []

  steps = (ds_info.splits[tpu_loader.split].num_examples // tpu_loader.batch_size)+1

  for i, batch in zip(tqdm.trange(steps), tpu_loader.loader):

    # the last batch is not parallised.
    if i == steps-1:
      clip_encoded_batch = image_fn(jax_params, np.squeeze(batch, axis=0))
    else:
      clip_encoded_batch = image_fn_pmapped(jax_params_repl, batch)

    clip_encoded_batch = jax.device_get(clip_encoded_batch)
    clip_features.append(clip_encoded_batch)

  clip_flattened_features = [fea.reshape(-1,512) for fea in clip_features]
  coco_clip = np.concatenate(clip_flattened_features)

  return coco_clip


# + colab={"base_uri": "https://localhost:8080/"} id="rj239GlVf3wV" outputId="e66cc7ed-0ffc-4c06-b7e6-bbd52d6f70cd"
clip_train = clip_extract(train_loader)

# + id="9-fHuAjrI_z7" colab={"base_uri": "https://localhost:8080/"} outputId="3efd4bd5-922e-4a52-b406-9563b5f0e988"
clip_eval = clip_extract(test_loader)


# + id="kIbTGaQHnEUW"
def make_tfds_and_save(numpy_data, name):
  tf_ds = tf.data.Dataset.from_tensor_slices(numpy_data)
  tf.data.experimental.save(tf_ds,f'/content/{name}')
  return tf_ds


# + id="QU5WLGBKjO2r"
clip_train_ds = make_tfds_and_save(clip_train,'clip_train_ds')
clip_test_ds = make_tfds_and_save(clip_eval,'clip_test_ds')
