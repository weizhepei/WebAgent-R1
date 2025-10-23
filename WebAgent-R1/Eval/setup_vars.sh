#!/bin/bash


PUBLIC_HOSTNAME="localhost" # 'localhost' / '10.189.112.188'


export DATASET="webarena"


# Actually, the CLASSIFIEDS environment is not included in the WebArena-Lite evaluation, we keep the environment variables here just for consistency.
export CLASSIFIEDS="<your_classifieds_domain>:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"


# Below are the variables you should set for the evaluation.
HOMEPAGE_PORT=8077
SHOPPING_PORT=8082
SHOPPING_ADMIN_PORT=8083
REDDIT_PORT=8080
GITLAB_PORT=9001
WIKIPEDIA_PORT=8087
MAP_PORT=9003


export HOMEPAGE="http://${PUBLIC_HOSTNAME}:${HOMEPAGE_PORT}"
export SHOPPING="http://${PUBLIC_HOSTNAME}:${SHOPPING_PORT}"
export SHOPPING_ADMIN="http://${PUBLIC_HOSTNAME}:${SHOPPING_ADMIN_PORT}/admin"
export REDDIT="http://${PUBLIC_HOSTNAME}:${REDDIT_PORT}"
export GITLAB="http://${PUBLIC_HOSTNAME}:${GITLAB_PORT}"
export WIKIPEDIA="http://${PUBLIC_HOSTNAME}:${WIKIPEDIA_PORT}"
export MAP="http://${PUBLIC_HOSTNAME}:${MAP_PORT}"


export OPENAI_API_KEY="xxxx"
export OPENAI_API_URL="https://api.openai.com/v1"
