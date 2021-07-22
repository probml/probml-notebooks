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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/hmm_chorales_numpyro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="Tcsyp2vNBl7r"
# HMC infernece for parameters of a discrete observation HMM fit to a dataset of Bach Chorales.
# We marginalize out the discrete latents using  variable elimination.
#
# http://num.pyro.ai/en/stable/examples/hmm_enum.html
#

# + colab={"base_uri": "https://localhost:8080/"} id="SOchf5ozCJT5" outputId="5c867285-9cda-4cce-9d63-0ae3cb354ff5"
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4" # use 2 for regular colab, 4 for high memory (colab pro)
# !pip install -q numpyro@git+https://github.com/pyro-ppl/numpyro
import numpyro
import numpyro.distributions as dist

# + colab={"base_uri": "https://localhost:8080/"} id="ecmWxwg1MJOI" outputId="dad32099-03c7-4319-c4da-b3f2dee15027"
# #!pip install funsor
# !git clone https://github.com/pyro-ppl/funsor.git


# + id="pLZt-eA0BHKy"
import argparse
import logging
import os
import time

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.control_flow import scan
from numpyro.contrib.indexing import Vindex
import numpyro.distributions as dist
from numpyro.examples.datasets import JSB_CHORALES, load_dataset
from numpyro.handlers import mask
from numpyro.infer import HMC, MCMC, NUTS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# + [markdown] id="SY2cYIXeCQcD"
# # Simple HMM

# + id="gr0BlUKEBvhp"
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
#
# This model includes a plate for the data_dim = 44 keys on the piano. This
# model has two "style" parameters probs_x and probs_y that we'll draw from a
# prior. The latent state is x, and the observed state is y.
def hmm_simple(sequences, lengths, hidden_dim, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    with mask(mask=include_prior):
        probs_x = numpyro.sample("probs_x",
                                 dist.Dirichlet(0.9 * jnp.eye(hidden_dim) + 0.1)
                                     .to_event(1))
        probs_y = numpyro.sample("probs_y",
                                 dist.Beta(0.1, 0.9)
                                     .expand([hidden_dim, data_dim])
                                     .to_event(2))

    def transition_fn(carry, y):
        x_prev, t = carry
        with numpyro.plate("sequences", num_sequences, dim=-2):
            with mask(mask=(t < lengths)[..., None]):
                x = numpyro.sample("x", dist.Categorical(probs_x[x_prev]))
                with numpyro.plate("tones", data_dim, dim=-1):
                    numpyro.sample("y", dist.Bernoulli(probs_y[x.squeeze(-1)]), obs=y)
        return (x, t + 1), None

    x_init = jnp.zeros((num_sequences, 1), dtype=jnp.int32)
    # NB swapaxes: we move time dimension of `sequences` to the front to scan over it
    scan(transition_fn, (x_init, 0), jnp.swapaxes(sequences, 0, 1))


# + [markdown] id="4-iVJd8NCYbF"
# # Data

# + colab={"base_uri": "https://localhost:8080/"} id="ZwHUyfLnCZKa" outputId="9f4e148e-aded-493c-d446-7a679b64df95"

args = {'num_sequences': 10,
        'truncate':  50}

_, fetch = load_dataset(JSB_CHORALES, split='train', shuffle=False)
lengths, sequences = fetch()
if args['num_sequences']:
    sequences = sequences[0:args['num_sequences']]
    lengths = lengths[0:args['num_sequences']]

logger.info('-' * 40)
logger.info('Training on {} sequences'.format(len(sequences)))

# find all the notes that are present at least once in the training set
present_notes = ((sequences == 1).sum(0).sum(0) > 0)
# remove notes that are never played (we remove 37/88 notes with default args)
sequences = sequences[..., present_notes]

if args['truncate']:
    lengths = lengths.clip(0, args['truncate'])
    sequences = sequences[:, :args['truncate']]

logger.info('Each sequence has shape {}'.format(sequences[0].shape))


# + [markdown] id="qjGlP8OICde4"
# # Model fitting

# + colab={"base_uri": "https://localhost:8080/", "height": 507} id="JG1Hp1RdCet9" outputId="042d1989-8c39-4974-bae5-2e0d66bb6747"
model = hmm_simple

args = {'num_warmup': 50,
        'num_chains': 2,
        'num_samples': 50,
        'device': 'cpu'}

logger.info('Starting inference...')
rng_key = random.PRNGKey(2)
start = time.time()
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=args['num_warmup'],
            num_samples=args['num_samples'], num_chains=args['num_chains'], progress_bar=True)

nstates = 5
mcmc.run(rng_key, sequences, lengths, hidden_dim=nstates)
mcmc.print_summary()
logger.info('\nMCMC elapsed time: {}'.format(time.time() - start))


# + id="oins4P95LvDN"

