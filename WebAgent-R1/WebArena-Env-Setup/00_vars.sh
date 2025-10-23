#!/bin/bash

# PUBLIC_HOSTNAME=$(curl -s ifconfig.me)
# PUBLIC_HOSTNAME="10.189.112.188" # YOUR_HOSTNAME_HERE

PUBLIC_HOSTNAME="localhost"

HOMEPAGE_PORT=8077
SHOPPING_PORT=8082
SHOPPING_ADMIN_PORT=8083
REDDIT_PORT=8080
GITLAB_PORT=9001
WIKIPEDIA_PORT=8087
MAP_PORT=9003
RESET_PORT=7565

SHOPPING_URL="http://${PUBLIC_HOSTNAME}:${SHOPPING_PORT}"
SHOPPING_ADMIN_URL="http://${PUBLIC_HOSTNAME}:${SHOPPING_ADMIN_PORT}/admin"
REDDIT_URL="http://${PUBLIC_HOSTNAME}:${REDDIT_PORT}/forums/all"
GITLAB_URL="http://${PUBLIC_HOSTNAME}:${GITLAB_PORT}/explore"
WIKIPEDIA_URL="http://${PUBLIC_HOSTNAME}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
MAP_URL="http://${PUBLIC_HOSTNAME}:${MAP_PORT}"


# download the archives from the webarena instructions
# https://github.com/web-arena-x/webarena/tree/main/environment_docker
# Download the additional openstreetmap docker files from Zenodo (see README)
#  - shopping_final_0712.tar
#  - shopping_admin_final_0719.tar
#  - postmill-populated-exposed-withimg.tar
#  - gitlab-populated-final-port8023.tar
#  - openstreetmap-website-db.tar.gz
#  - openstreetmap-website-web.tar.gz
#  - openstreetmap-website.tar.gz
#  - wikipedia_en_all_maxi_2022-05.zim

ARCHIVES_LOCATION="./images"
