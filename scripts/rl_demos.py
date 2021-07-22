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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book2/rl/rl_demos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="POrA585UFLms"
#
#
# ![GitHub](https://img.shields.io/github/license/probml/pyprobml)
#
# Colab authors: Kevin P. Murphy (murphyk@gmail.com) and Mahmoud Soliman (mjs@aucegypt.edu)
#
#

# + id="I3CEU8u0FQR0"
# Attribution 
# This notebook is based on the following: 
# https://github.com/mjsML/VizDoom-Keras-RL
# https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/actor_critic_cartpole.ipynb

# + id="qEYlbLuzFh_b" colab={"base_uri": "https://localhost:8080/"} outputId="b0661fbe-0e44-4a7a-ae7b-d5c081893257"
# Imports
from tensorflow.python.client import device_lib
from psutil import virtual_memory
import cv2
from google.colab.patches import cv2_imshow
# %tensorflow_version 2.x
import tensorflow as tf
import os


from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score


from sklearn.datasets.samples_generator import make_blobs
from  IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile

from tqdm import tqdm

# + id="X_8KoI6UFkRo" colab={"base_uri": "https://localhost:8080/"} outputId="0055b9e3-d953-4304-830f-ad9fdd54c2e0"
#title Hardware check 



def find_accelerator():
  
  mem = virtual_memory()
  devices=device_lib.list_local_devices()
  RAM="Physical RAM: {:.2f} GB".format(mem.total/(1024*1024*1024))
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  
    device=["TPU at "+str(tpu.cluster_spec().as_dict()['worker'])]  
  except ValueError:
    device =[d.physical_device_desc for d in devices if d.device_type=="GPU"]
  if not device:
    return None, RAM
  return device ,  RAM 

a,r=find_accelerator()
print("Please make sure that the statement below says Accelerator found")
print("Accelerator found:",a,r)



# + id="w3V-stpMFlJN" colab={"base_uri": "https://localhost:8080/"} outputId="5f55dbeb-d950-4b08-b707-018a03c05f3a"
#title Install the extra required packages if any
# Installation of libs as per 
# https://stackoverflow.com/questions/50667565/how-to-install-vizdoom-using-google-colab

# %%bash
# Install deps from 
# https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-linux

apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip
apt-get install libboost-all-dev
apt-get install liblua5.1-dev

# + [markdown] id="M5vmaF61ZvV5"
# #Partially observed Markov decision processes (POMDPs) 
# We will start by exploring POMDPs , the states of the environment, $z_{t}$ , are hidden from the agent. The agent gets to see partial observations derived from the hidden state, which we denote by
# $s_{t} \in \mathcal{S}$ these are sampled from the observation model, $p(s_{t}|z_{t})$.
#
# In this example we will work with ViZDoom and Deep Recurrent Q Network.
#
# Note that this is a quick overview example, the details will be discussed later.

# + [markdown] id="JBTCb79f72TU"
#  ## Deep Recurrent Q Network

# + id="Q0npOg2hPfrF" colab={"base_uri": "https://localhost:8080/"} outputId="70d73c24-44f4-466b-ed6b-a338474c9b2c"
#title Install ViZDoom... takes few mins

# !pip install vizdoom


# + id="5HSk5mAdPbIv" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="8480df6f-678c-46fd-c051-309b828bc49a"
#title Clone ViZDoom-Keras-RL repo and imports
# Clone VizDoom-Keras-RL
# !git clone https://github.com/mjsML/VizDoom-Keras-RL.git
# %cd /content/VizDoom-Keras-RL
from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Flatten, RepeatVector, Masking
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding
#tf.keras.layers.Concatenate(axis=1)([x, y])
from keras.layers.recurrent import LSTM, GRU
#from keras.optimizers import SGD, Adam, rmsprop
from keras.optimizers import SGD, Adam
from keras import backend as K

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import itertools as it
from time import sleep
import tensorflow as tf

from networks import Networks


# + id="7Pqux1c-Trj4"
#title Setup ViZDoom with defend the center scenario

#TF2 TF1 compatibility 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

from drqn import ReplayMemory,DoubleDQNAgent,preprocessImg

game = DoomGame()
game.load_config("/content/VizDoom-Keras-RL/defend_the_center.cfg")
game.set_sound_enabled(True)
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_window_visible(False)
game.init()

game.new_episode()
game_state = game.get_state()

misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
prev_misc = misc

action_size = game.get_available_buttons_size()

img_rows, img_cols = 64, 64
img_channels = 3  # Color channel
trace_length = 4  # Temporal Dimension

state_size = (trace_length, img_rows, img_cols, img_channels)
agent = DoubleDQNAgent(state_size, action_size, trace_length)

agent.model = Networks.drqn(state_size, action_size, agent.learning_rate)
agent.target_model = Networks.drqn(
    state_size, action_size, agent.learning_rate)

s_t = game_state.screen_buffer  # 480 x 640
s_t = preprocessImg(s_t, size=(img_rows, img_cols))

is_terminated = game.is_episode_finished()

# + id="KEOjhw2tR1zl"
#title Start training DRQN Agent
epsilon = agent.initial_epsilon
GAME = 0
t = 0
max_life = 0  # Maximum episode life (Proxy for agent performance)
life = 0
episode_buf = []  # Save entire episode

# Buffer to compute rolling statistics
life_buffer, ammo_buffer, kills_buffer = [], [], []

while not game.is_episode_finished():

    loss = 0
    Q_max = 0
    r_t = 0
    a_t = np.zeros([action_size])

    # Epsilon Greedy
    if len(episode_buf) > agent.trace_length:
        # 1x8x64x64x3
        state_series = np.array(
            [trace[-1] for trace in episode_buf[-agent.trace_length:]])
        state_series = np.expand_dims(state_series, axis=0)
        action_idx = agent.get_action(state_series)
    else:
        action_idx = random.randrange(agent.action_size)
    a_t[action_idx] = 1

    a_t = a_t.astype(int)
    game.set_action(a_t.tolist())
    skiprate = agent.frame_per_action
    game.advance_action(skiprate)

    game_state = game.get_state()  # Observe again after we take the action
    is_terminated = game.is_episode_finished()

    # each frame we get reward of 0.1, so 4 frames will be 0.4
    r_t = game.get_last_reward()

    if (is_terminated):
        if (life > max_life):
            max_life = life
        GAME += 1
        life_buffer.append(life)
        ammo_buffer.append(misc[1])
        kills_buffer.append(misc[0])
        print("Episode Finish ", misc)
        game.new_episode()
        game_state = game.get_state()
        misc = game_state.game_variables
        s_t1 = game_state.screen_buffer

    s_t1 = game_state.screen_buffer
    misc = game_state.game_variables
    s_t1 = preprocessImg(s_t1, size=(img_rows, img_cols))

    r_t = agent.shape_reward(r_t, misc, prev_misc, t)

    if (is_terminated):
        life = 0
    else:
        life += 1

    # update the cache
    prev_misc = misc

    # Update epsilon
    if agent.epsilon > agent.final_epsilon and t > agent.observe:
        agent.epsilon -= (agent.initial_epsilon -
                          agent.final_epsilon) / agent.explore

    # Do the training
    if t > agent.observe:
        Q_max, loss = agent.train_replay()

    # save the sample <s, a, r, s'> to episode buffer
    episode_buf.append([s_t, action_idx, r_t, s_t1])

    if (is_terminated):
        agent.memory.add(episode_buf)
        episode_buf = []  # Reset Episode Buf

    s_t = s_t1
    t += 1

    # save progress every 10000 iterations
    if t % 10000 == 0:
        print("Now we save model")
        agent.model.save_weights("./models/drqn.h5", overwrite=True)

    # print info
    state = ""
    if t <= agent.observe:
        state = "observe"
    elif t > agent.observe and t <= agent.observe + agent.explore:
        state = "explore"
    else:
        state = "train"

    if (is_terminated):
        print("TIME", t, "/ GAME", GAME, "/ STATE", state,
              "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(Q_max), "/ LIFE", max_life, "/ LOSS", loss)

        # Save Agent's Performance Statistics
        if GAME % agent.stats_window_size == 0 and t > agent.observe:
            print("Update Rolling Statistics")
            agent.mavg_score.append(np.mean(np.array(life_buffer)))
            agent.var_score.append(np.var(np.array(life_buffer)))
            agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
            agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

            # Reset rolling stats buffer
            life_buffer, ammo_buffer, kills_buffer = [], [], []

            # Write Rolling Statistics to file
            with open("statistics/drqn_stats.txt", "w") as stats_file:
                stats_file.write('Game: ' + str(GAME) + '\n')
                stats_file.write('Max Score: ' + str(max_life) + '\n')
                stats_file.write('mavg_score: ' +
                                  str(agent.mavg_score) + '\n')
                stats_file.write(
                    'var_score: ' + str(agent.var_score) + '\n')
                stats_file.write('mavg_ammo_left: ' +
                                  str(agent.mavg_ammo_left) + '\n')
                stats_file.write('mavg_kill_counts: ' +
                                  str(agent.mavg_kill_counts) + '\n')


# + [markdown] id="EFHwTMtjtrpI"
# # Fully observed Markov decision processes (MDPs)
# Now we explore fully observed Markov decision process.
#
# In a fully observable problem the observed state is equal to the hidden state (i.e., $s_{t}=z_{t}$). 
#
# In this case, the POMDP reduces to a simpler model known as a Markov decision process or MDP
#

# + [markdown] id="PnFH2ojv7X6i"
#
# ## Actor Critic Method
# As an agent takes actions and moves through an environment, it learns to map the observed state of the environment to two possible outputs:
#
# **Recommended action:** 
#
# A probabiltiy value for each action in the action space. The part of the agent responsible for this output is called the actor.
#
#
# **Estimated rewards in the future:** 
#
# Sum of all rewards it expects to receive in the future. The part of the agent responsible for this output is the critic.
#
# Agent and Critic learn to perform their tasks, such that the recommended actions from the actor maximize the rewards.
#
#
# **CartPole-V0**
#
# A pole is attached to a cart placed on a frictionless track. The agent has to apply force to move the cart. It is rewarded for every time step the pole remains upright. The agent, therefore, must learn to keep the pole from falling over.

# + id="6UfvC4ni0Zan"
#@title Imports
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("CartPole-v0")  # Create the environment
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

# + id="ay_fSFKW0oxp" colab={"base_uri": "https://localhost:8080/", "height": 223} outputId="ff37bd81-65b4-4a63-91c5-c17ef635107a"
#@title Define Model
num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

# + id="uHyJ3w6n6Fdn" cellView="form"
#@title Train model
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break


# + [markdown] id="H0hXYOtJ6NbJ"
# ### Visualizations
#
#
# In early stages of training:
#
#
# ![Imgur](https://i.imgur.com/5gCs5kH.gif)
#
#
# In later stages of training:
#
#
# ![Imgur](https://i.imgur.com/5ziiZUD.gif)
