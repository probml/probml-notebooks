# Tips and tricks on using Colab, Google Cloud Platform and TPUs

Authors: [murphyk](https://github.com/murphyk), [mjsML](https://github.com/mjsML), [gerdm](https://github.com/gerdm), et al  2021.

## Colab tutorials 

* [Colab tutorial](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/colab_intro.ipynb)
* [Using Google Cloud Storage from Colab](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/GCS_demo_v2.ipynb)
* [Using TPU VMs from Colab](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/tpu_colab_tutorial.ipynb)
* [Accessing colab machine via ssh](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/ssh_tunnels_and_how_to_dig_them.ipynb)


## Using GCP directly

An alternative to using Colab is to get an account on GCP; this can provide much more compute power, and gives you persistent storage.
Follow the setup instructions in the section called "Gcloud VM" in [this colab](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/ssh_tunnels_and_how_to_dig_them.ipynb#scrollTo=TLtWT8vn-Eyh). Now you can connect to your Gcloud account via VScode:
just type in the external IP address into the green icon on the lower left corner. Once connected, you can clone your github repo, edit your source code in a proper IDE, and open a jupyter notebook for interactive development. When you're done, save it all back to github. See the screenshot below for an example.
![](https://github.com/probml/probml-notebooks/raw/main/images/vscode-ssh.png)

#![](https://github.com/probml/probml-notebooks/raw/main/images/github-vscode-browser.png)

## Miscellaneous tricks

- To use TPU VM host machines in CPU mode (ie without using the TPU, just leveraging the ~48 CPU cores), use something like this:
```
import jax
import jax.numpy as jnp

# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')

x = jnp.square(2)
```
