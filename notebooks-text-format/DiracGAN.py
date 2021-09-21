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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/DiracGAN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="Tioj76xwOWeK"
# ## The DiracGAN example
#
# Author: Mihaela Rosca 
#
# We show DiracGAN (https://arxiv.org/abs/1801.04406), where the true distribution is is Dirac delta distribution with mass at zero. The generator is modeling a Dirac delta distribution with parameter $\theta$: $G_{\theta}(z) = \theta$ and the discriminator is a linear function of the input with learned
# parameter $\phi$: $D_{\phi}(x) = \phi x$. This results in the zero-sum game given by:
# $$ 
# L_D = - l(\theta \phi) - l(0) \\
# L_G = + l(\theta \phi) + l(0) 
# $$
#
# where $l$ depends on the GAN formulation used ($l(z) = - \log (1 + e^{-z})$ for instance). The unique equilibrium point is $\theta = \phi = 0$. 

# + id="snZfbH_TtO5j"
import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.misc import derivative
import seaborn as sns


# + id="u1i8mZWCvugd"
def set_up_fonts():
  sns.reset_orig()

  import matplotlib
  matplotlib.rcParams['pdf.fonttype'] = 42
  matplotlib.rcParams['ps.fonttype'] = 42


# + [markdown] id="OqrocK0iMMqM"
# ### Display variables

# + id="X2bi6yWfI45j"
hw = 10
hl = 6
minshaft = 2
scale = 1.5

# + id="m-FMcNsmInsE"
color=['blue', 'red','green','orange', 'magenta']

# + id="UMdG1ZxrwBEs"
set_up_fonts()


# + [markdown] id="N1S_k02OMHaE"
# ## Defining the Euler updates (gradient descent)

# + id="cPUpPCOGVR1E"
def euler_alternating(fn, v, t):
  last_t = t[0]
  vs = [v]
  num_dims = len(v)
  last_v = list(v)
  for current_t in t[1:]:
    delta_t = current_t - last_t
    for i in range(num_dims):
      interim_v = last_v + delta_t * np.array(fn(current_t, last_v))
      last_v[i] = interim_v[i]
    last_t = current_t
    vs.append(last_v.copy())
  return np.array(vs)


# + id="fEi0ZekkIyXg"
def euler(fn, v, t):
  last_t = t[0]
  vs = [v]
  last_v = v
  for current_t in t[1:]:
    current_v = last_v + (current_t - last_t) * np.array(fn(current_t, last_v))
    last_t = current_t
    last_v = current_v
    vs.append(current_v)
  return np.array(vs)


# + [markdown] id="8p73c5zYhExV"
# # Dirac GAN
#
#

# + id="BkcTXKS76hyV"
grad_f = lambda x:  1. / (1 + np.exp(-x))

vect0 = [(1, 1)]


# + colab={"base_uri": "https://localhost:8080/", "height": 306} id="C8yQukseIcOo" outputId="ebe24249-c6b7-42f7-da6b-61314fbd6829"
# Write the problem in a way compatible with solve_ivp.
# Return the gradients for each player.
def system(t, vect):
  x, y = vect
  return [-grad_f(x * y) * y, grad_f(x * y) * x]

t = np.arange(0, 100, 0.2)
plot = plt.figure()
v = vect0[0]        

sol = solve_ivp(system, (0, 200), v, t_eval=t, dense_output=True, method='RK45')
sol = sol.sol(t).T
widths = np.linspace(0, 2, sol.size)
plt.quiver(sol[:-1, 0], sol[:-1, 1], sol[1:, 0]-sol[:-1, 0], sol[1:, 1]-sol[:-1, 1], scale_units='xy', angles='xy', scale=2, color=color[0], linewidths=widths, edgecolors=color[0], label='Continuous dynamics', headwidth=hw, headlength=hl, minshaft=2)  

plt.title('Dirac GAN', fontsize=16)
plt.plot(v[0], v[1], 'go', markersize=10)
plt.plot(0, 0,'rx', markersize=12) 
plt.plot(0, 0,'rx', markersize=12, label='equilibruim (0, 0)')  
plt.legend(loc='upper right', bbox_to_anchor=(0.8, 1), fontsize=13, framealpha=0)

plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel(r'$\theta$', fontsize=16)

plt.xticks([])
plt.yticks([])
plt.xlim((-4, 4))
plt.ylim((-3, 4.5))

# + colab={"base_uri": "https://localhost:8080/", "height": 306} id="gxMpZMt71ieS" outputId="8584e989-a254-44e9-e057-86d884f9adb7"
disc_lr = 0.1
gen_lr = 0.1
vect0 = [(1, 1)]

t = np.arange(0, 100, disc_lr)
plot = plt.figure()
v = vect0[0]        

   
sol = euler(system, v, t)
widths = np.linspace(0, 2, sol.size)
plt.quiver(sol[:-1, 0], sol[:-1, 1], sol[1:, 0]-sol[:-1, 0], sol[1:, 1]-sol[:-1, 1], scale_units='xy', angles='xy', scale=2, color=color[0], linewidths=widths, edgecolors=color[0], label='Simultaneous gradient descent', headwidth=hw, headlength=hl, minshaft=2)  

plt.title('Dirac GAN', fontsize=16)
plt.plot(v[0], v[1], 'go', markersize=10)
plt.plot(0, 0,'rx', markersize=12, label='equilibruim (0, 0)') 
plt.legend(loc='upper right', bbox_to_anchor=(0.8, 1), fontsize=13, framealpha=0)

plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel(r'$\theta$', fontsize=16)
plt.xticks([])
plt.yticks([])
plt.xlim((-4, 4))
plt.ylim((-3, 4.5))

# + colab={"base_uri": "https://localhost:8080/", "height": 269} id="ZRPE1Fc_73kd" outputId="97dc6f0d-02fa-42e2-d872-02e62de03043"
plt.vlines(0, 0, 10, lw=3, colors='b', label=r'$p^*$')
plt.vlines(2, 0, 10, lw=3, colors='g', label=r'$q_{\theta}$', linestyles='--')
plt.hlines(0, -1, 10, lw=2, colors='k')

xlim = np.linspace(-0.5, 2.5, 50)
plt.plot(xlim, 1.7 * xlim, color='r', label=r'$D_{\phi}(x) = \phi x$', ls='-.')

plt.xlim(-0.5, 2.5)
plt.yticks([])
plt.xticks([])
plt.legend(framealpha=0, loc='upper center', fontsize=14)

# + colab={"base_uri": "https://localhost:8080/", "height": 306} id="OybzsaDz2Nmk" outputId="004dbdb5-f855-43c2-e1a4-471d86067b88"
lr = 0.1
vect0 = [(1, 1)]

t = np.arange(0, 100, lr)
plot = plt.figure()
v = vect0[0]        

sol = euler_alternating(system, v, t)
widths = np.linspace(0, 2, sol.size)
plt.quiver(sol[:-1, 0], sol[:-1, 1], sol[1:, 0]-sol[:-1, 0], sol[1:, 1]-sol[:-1, 1], scale_units='xy', angles='xy', scale=2, color=color[0], linewidths=widths, edgecolors=color[0], label='Alternating gradient descent', headwidth=hw, headlength=hl, minshaft=2)  

plt.title('Dirac GAN', fontsize=16)
plt.plot(v[0], v[1], 'go', markersize=10)
plt.plot(0, 0,'rx', markersize=12, label='equilibruim (0, 0)') 
plt.legend(loc='upper right', bbox_to_anchor=(0.8, 1), fontsize=13, framealpha=0)

plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel(r'$\theta$', fontsize=16)

plt.xticks([])
plt.yticks([])
plt.xlim((-4, 4))
plt.ylim((-3, 4.5))
