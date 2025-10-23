#!/bin/bash
# re-validate login information

source setup_vars.sh


mkdir -p ./.auth
python auto_login.py