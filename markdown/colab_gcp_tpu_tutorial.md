# Tips and tricks on using Colab, Google Cloud Platform and TPUs

Authors: [murphyk](https://github.com/murphyk), [mjsML](https://github.com/mjsML), [gerdm](https://github.com/gerdm), et al  2021.

## Tutorials in notebook format

* [Colab tutorial](https://github.com/probml/probml-notebooks/blob/main/notebooks/colab_intro.ipynb)
* [Using Google Cloud Storage from Colab](https://github.com/probml/probml-notebooks/blob/main/notebooks/GCS_demo_v2.ipynb)
* [Using TPU VMs from Colab](https://github.com/probml/probml-notebooks/blob/main/notebooks/tpu_colab_tutorial.ipynb)


## Miscellaneous tricks

- To use TPU VM host machines in CPU mode (ie without using the TPU, just leveraging the ~48 CPU cores), use something like this:
```
import jax
import jax.numpy as jnp

# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')

x = jnp.square(2)
```
