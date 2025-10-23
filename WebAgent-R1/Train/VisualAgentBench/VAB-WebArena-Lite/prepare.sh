#!/bin/bash
# re-validate login information

source setup_vars.sh

mkdir -p ./.auth
python browser_env/auto_login.py