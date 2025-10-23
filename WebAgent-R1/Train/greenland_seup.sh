#!/bin/bash

# initialize conda and activate the environment
conda init bash
source ~/.bashrc
conda activate verifiers-new
cd /workspace/verifiers

# change owner of the /workspace directory to the current user
sudo chown -R greenland-user:greenland-users /workspace


# wandb login
wandb login

# huggingface-cli login
huggingface-cli login

playwright install