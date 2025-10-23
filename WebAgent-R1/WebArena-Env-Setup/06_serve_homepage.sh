#!/bin/bash

source 00_vars.sh

# install flask in a venv
sudo apt install python3-venv -y
# sudo apt install python3-pip python3-devel -y
python3 -m venv venv
source venv/bin/activate
pip install flask


cd webarena-homepage
cp templates/index.backup templates/index.html
sed -i "s|SHOPPING_URL|${SHOPPING_URL}|g" templates/index.html
sed -i "s|SHOPPING_ADMIN_URL|${SHOPPING_ADMIN_URL}|g" templates/index.html
sed -i "s|GITLAB_URL|${GITLAB_URL}|g" templates/index.html
sed -i "s|REDDIT_URL|${REDDIT_URL}|g" templates/index.html
sed -i "s|MAP_URL|${MAP_URL}|g" templates/index.html
sed -i "s|WIKIPEDIA_URL|${WIKIPEDIA_URL}|g" templates/index.html

flask run --host=0.0.0.0 --port=$HOMEPAGE_PORT
