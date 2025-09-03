#!/bin/bash
# re-validate login information

source /home/zhepei/setup_vars.sh

mkdir -p ./.auth
python browser_env/auto_login.py