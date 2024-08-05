#!/usr/bin/env bash

# Usage: ./download_mlir_trt_package.sh token 0.1.29 310

# Remove the requirement for username and token when repository is public.
GITHUB_TOKEN=$1
RELEASE_TAG=$2
PYTHON_VERSION=$3
REPO="NVIDIA/TensorRT-Incubator"
GITHUB="https://api.github.com"

alias errcho='>&2 echo'

# assets list you want to download
List=( 
mlir_tensorrt_compiler-${RELEASE_TAG}+cuda12.trt102-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl 
mlir_tensorrt_runtime-${RELEASE_TAG}+cuda12.trt102-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl
)

function gh_curl() {
  curl -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github.v3+json" \
       $@
}

assets=$(gh_curl -s $GITHUB/repos/$REPO/releases/tags/mlir-tensorrt-v$RELEASE_TAG)
for row in $(echo "${assets}" | jq -c '.assets[]'); do
name=$( jq -r  '.name' <<< "${row}" ) 
    for item in "${List[@]}"; do
        if [[ "$item" == "$name" ]]; then
            id=$( jq -r  '.id' <<< "${row}" ) 
            wget -q --auth-no-challenge --header='Accept:application/octet-stream' \
            https://$GITHUB_TOKEN:@api.github.com/repos/$REPO/releases/assets/$id \
            -O ${name}
        fi
    done
done
