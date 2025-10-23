#!/bin/bash

# stop if any error occur
set -e

docker start gitlab
docker start shopping
docker start shopping_admin
docker start forum
# docker start kiwix33
docker start wikipedia

cd openstreetmap-website/
docker compose start

echo -n -e "Waiting 60 seconds for all services to start..."
sleep 60
echo -n -e " done\n"

