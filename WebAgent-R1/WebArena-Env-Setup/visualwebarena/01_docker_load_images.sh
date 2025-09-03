#!/bin/bash

# stop if any error occur
set -e

source 00_vars.sh

assert() {
  if ! "$@"; then
    echo "Assertion failed: $@" >&2
    exit 1
  fi
}

load_docker_image() {
  local IMAGE_NAME="$1"
  local INPUT_FILE="$2"

  if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}:"; then
    echo "Loading Docker image ${IMAGE_NAME} from ${INPUT_FILE}"
    docker load --input "${INPUT_FILE}"
  else
    echo "Docker image ${IMAGE_NAME} is already loaded."
  fi
}

# make sure all required files are here
assert [ -f ${ARCHIVES_LOCATION}/shopping_final_0712.tar ]
assert [ -f ${ARCHIVES_LOCATION}/postmill-populated-exposed-withimg.tar ]
assert [ -f ${ARCHIVES_LOCATION}/classifieds_docker_compose.zip ]
assert [ -f ${ARCHIVES_LOCATION}/wikipedia_en_all_maxi_2022-05.zim ]

# load docker images (if needed)
load_docker_image "shopping_final_0712" "${ARCHIVES_LOCATION}/shopping_final_0712.tar"
load_docker_image "postmill-populated-exposed-withimg" "${ARCHIVES_LOCATION}/postmill-populated-exposed-withimg.tar"

# extract classifieds archive locally (if needed)
if [ ! -d ./classifieds_docker_compose ]; then
  echo "Extracting classifieds archive..."
  unzip ${ARCHIVES_LOCATION}/classifieds_docker_compose.zip
else
  echo "Classifieds archive already extracted."
fi

# copy wikipedia archive to local folder (if needed)
WIKIPEDIA_ARCHIVE=wikipedia_en_all_maxi_2022-05.zim
if [ ! -f ./wiki/${WIKIPEDIA_ARCHIVE} ]; then
  echo "Moving wikipedia archive..."
  mkdir -p ./wiki
  cp ${ARCHIVES_LOCATION}/${WIKIPEDIA_ARCHIVE} ./wiki
else
  echo "Wikipedia archive already present."
fi
