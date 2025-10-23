
# WebAgent-R1

This repository contains the implementation of **WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning**. The codebase is organized into several main components for environment setup, training, and evaluation.

## 🚀 Quick Start

### Installation

#### 1. WebArena Environment Setup

Before training or evaluation, you need to set up the WebArena environment for the LLM agent to interact with:

```bash
cd WebArena-Env-Setup

# Follow the instructions in WebArena-Env-Setup to download required images

# Edit 00_vars.sh with your ports and hostname (optional)

# Load Docker images and start the environment
# Note: You may need to run with sudo depending on your Docker setup
bash 01_docker_load_images.sh
bash 02_docker_remove_containers.sh
bash 03_docker_create_containers.sh
bash 04_docker_start_containers.sh
bash 05_docker_patch_containers.sh
bash 06_serve_homepage.sh
```

#### 2. Evaluation Setup
```bash
# Create Python environment
conda create -n webagent-r1 python=3.10 -y
conda activate webagent-r1

# Install dependencies for evaluation
cd Eval
pip install -r requirements.txt
playwright install
pip install -e .
pip install lxml dashscope anthropic openai==1.64.0
```

#### 3. Training Setup
```bash
cd Train
conda create -f environment.yml
```

## 📖 Usage

### Evaluation
```bash
cd Eval
bash evaluate.sh
python score.py <path_to_eval_result>
```

### Training
```bash
cd Train/verifiers
conda activate verifiers-new
bash run.sh
```

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Zhepei (zhepei.wei@virginia.edu). If you encounter any problems when using the code, or want to report a bug, feel free to open an issue! Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find this repository helpful in your work:

```bibtex
@inproceedings{
wei2025webagent-r1,
title={{WebAgent-R1}: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning},
author={Wei, Zhepei and Yao, Wenlin and Liu, Yao and Zhang, Weizhi and Lu, Qin and Qiu, Liang and Yu, Changlong and Xu, Puyang and Zhang, Chao and Yin, Bing and Yun, Hyokun and Li, Lihong},
booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)},
year={2025},
url={https://arxiv.org/abs/2505.16421}
}
```