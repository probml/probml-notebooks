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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/ssh_tunnels_and_how_to_dig_them.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="NMLDAnET7tTh"
# # Introduction

# + [markdown] id="ilryC0Hc7VAC"
# Secure SHell protocol (or ssh) is a very useful protocol that enables users to execute shell commands on remote hosts. Today this powers a slew of IDEs to enable "remote development" (i.e. you write code on a thin client, and execute the code on a massive server).

# + [markdown] id="M7lNjx0B7xfv"
# # Scenario 1: VSCode to Colab

# + [markdown] id="I87nrN9CQ-wJ"
# ## Check VM type and resources

# + id="yfynSjH6OCeI" colab={"base_uri": "https://localhost:8080/"} outputId="c79f67eb-0889-47e3-f75f-5c5b23fe5942"
# !nvidia-smi
# !apt install htop >/dev/null

# + [markdown] id="kBJ2gyc1RI18"
# ## Install colab-ssh

# + id="508tqLE57Drs" colab={"base_uri": "https://localhost:8080/", "height": 325} outputId="effc8382-0177-46d5-f0e2-e8bcdd8fb32c"
# %%time
# Install colab_ssh on google colab
# !pip install colab_ssh --upgrade

from colab_ssh import launch_ssh_cloudflared, init_git_cloudflared
ssh_tunnel_password = "password" #@param {type: "string"}
launch_ssh_cloudflared(password=ssh_tunnel_password)

# + [markdown] id="TLtWT8vn-Eyh"
# # Scenario 2: VSCode or PyCharm to a gcloud VM

# + [markdown] id="Q72GLf-p-TtZ"
# Steps:
#
#
# 1.   Create an [ssh key](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). 
# 2.   Copy the public key (*.pub) content to your project [meta-data](https://console.cloud.google.com/compute/metadata/sshKeys).
# 3.   Now you can ssh directly to your vm.
#
# 4.   Add the below example config to your ~/.ssh/config (this is a macOS example config file, you can refer to your ssh client manual on how to link your key file and define the host names).
#
# ```bash
# Host *
#   AddKeysToAgent yes
#   UseKeychain yes
#   IdentityFile ~/.ssh/id_ed25519 
# Host <EXTERNAL_IP_ADDRESS>
#   HostName <EXTERNAL_IP_ADDRESS>
#   User <YOUR_USER_NAME>
# ```
# 5.   If you are using VSCode, [this](https://code.visualstudio.com/docs/remote/ssh) help page is very useful. Also VSCode has a "remote++" experience, because it installs a "VSCode server" on the remote machine to take it even further than the standard ssh. The below figure from the microsoft help article illustrates the architecture of the system that enables the remote development experience for VSCode.
# <img src="https://code.visualstudio.com/assets/docs/remote/ssh/architecture-ssh.png">
#
# 6.   If you are using PyCharm, [this](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html#prereq) help page is very useful. PyCharm supports a concept called "remote interpreters via ssh. It makes remote development possible but VSCode provides a richer experience.
#
#
#
#
