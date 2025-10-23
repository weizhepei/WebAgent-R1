#!/bin/bash

PUBLIC_HOSTNAME="localhost"

export DATASET="webarena"

# Actually, the CLASSIFIEDS environment is not included in the WebArena-Lite evaluation, we keep the environment variables here just for consistency.
export CLASSIFIEDS="<your_classifieds_domain>:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"

# Below are the variables you should set for the evaluation.
export SHOPPING="http://${PUBLIC_HOSTNAME}:8082"
export SHOPPING_ADMIN="http://${PUBLIC_HOSTNAME}:8083/admin"
export REDDIT="http://${PUBLIC_HOSTNAME}:8080"
export GITLAB="http://${PUBLIC_HOSTNAME}:9001"
export WIKIPEDIA="http://${PUBLIC_HOSTNAME}:8087"
export MAP="http://${PUBLIC_HOSTNAME}:443"
export HOMEPAGE="http://${PUBLIC_HOSTNAME}:8077"


# export OPENAI_API_KEY="EMPTY" # use with caution: llm-as-judge will use this variable
# export OPENAI_API_URL="http://${PUBLIC_HOSTNAME}:8000/v1" # use with caution: llm-as-judge will use this variable

export OPENAI_API_KEY="xxxxx"
export OPENAI_API_URL="https://api.openai.com/v1"

# gpt-4-0125-preview (base model)
# gpt-4o-mini (chat model)

# from web server on dec desktop
# SHOPPING_PORT=8082
# SHOPPING_ADMIN_PORT=8083
# REDDIT_PORT=8080
# GITLAB_PORT=9001
# WIKIPEDIA_PORT=8081
# MAP_PORT=443
# HOMEPAGE_PORT=8077
# RESET_PORT=7565