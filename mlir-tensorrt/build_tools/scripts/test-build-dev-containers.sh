#!/usr/bin/env bash
set -ex
set -o pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSPACE_FOLDER="${DIR}/../.."
VERSION="${VERSION:-latest}"
WORKSPACE_FOLDER="${WORKSPACE_FOLDER:-${DEFAULT_WORKSPACE_FOLDER}}"
REGISTRY="ghcr.io"
PUSH="${PUSH:-false}"
INCLUDE_PATTERN="${INCLUDE_PATTERN:-${WORKSPACE_FOLDER}/.devcontainer/**/devcontainer.json}"

MULTI_ARCH="${MULTI_ARCH:-false}"

if [[ "$MULTI_ARCH" == "true" ]]; then
  platform_arg="--platform linux/amd64,linux/arm64"
  load_arg=""
else
  platform_arg=""
  load_arg="--load"
fi


# Iterate over all JSON files in the directory
for file in $INCLUDE_PATTERN; do
  if [[ -f "$file" ]] && [[ ! "$(basename "$(dirname "$file")")" =~ -prebuilt$ ]]; then
      echo "Processing file: $file"
      # Use jq to extract and print the "name" field
      container_suffix=$(jq -r '.name' "$file")

      ARG_BASE_IMAGE=$(jq -r '.build.args.BASE_IMAGE' "$file")
      ARG_USERNAME=$(jq -r '.build.args.USERNAME' "$file")
      ARG_LINUX_DISTRO=$(jq -r '.build.args.LINUX_DISTRO' "$file")
      ARG_LLVM_VERSION=$(jq -r '.build.args.LLVM_VERSION' "$file")
      DOCKERFILE=$(jq -r '.build.dockerfile' "$file" | sed "s|\${localWorkspaceFolder}|$PWD|g")

      tag="${REGISTRY}/nvidia/tensorrt-incubator/mlir-tensorrt:${container_suffix}-${VERSION}"
      push_arg=""
      if [[ "$PUSH" == "true" ]]; then
        push_arg="--push"
      fi
      docker buildx build \
        --build-arg "BASE_IMAGE=${ARG_BASE_IMAGE}" \
        --build-arg "LINUX_DISTRO=${ARG_LINUX_DISTRO}" \
        --build-arg "LLVM_VERSION=${ARG_LLVM_VERSION}" \
        $platform_arg \
        $load_arg \
        $push_arg \
        -t "$tag" -f "$DOCKERFILE" "$WORKSPACE_FOLDER/build_tools/docker"
  fi
done

echo "Script completed successfully."
