#!/bin/bash

# stop if any error occur
set -e

source 00_vars.sh

# install flask in a venv
# sudo yum install python3-pip -y
python3.10 -m venv venv_reset
source venv_reset/bin/activate

cd reset_server/
python3.10 server.py --port ${RESET_PORT} 2>&1 | tee -a server.log

# visit http://$PUBLIC_HOSTNAME:7565/reset
