#!/bin/bash

# stop if any error occur
set -e

cd ..
bash 02_docker_remove_containers.sh
bash 03_docker_create_containers.sh
bash 04_docker_start_containers.sh
bash 05_docker_patch_containers.sh

