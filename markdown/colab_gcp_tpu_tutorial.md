# List of tutorials and tricks on Google Cloud Platform and TPUs

## List of tutorials
- [Colab on how to use Google Cloud Storage from Colab](https://github.com/probml/probml-notebooks/blob/main/notebooks/GCS_demo_v2.ipynb)
- [Colab on how to use TPUs from Colab](https://github.com/probml/probml-notebooks/blob/main/notebooks/tpu_colab_tutorial.ipynb)


## Miscellaneous tricks

- To use TPU VM host machines in CPU mode (ie without using the TPU, just leveraging the ~48 CPU cores), use something like this:
```
import jax
import jax.numpy as jnp

# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')

x = jnp.square(2)
```
