#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${1:?workspace root is required}"
USERNAME="${2:?username is required}"
WORKSPACE_NAME="$(basename "${WORKSPACE_ROOT}")"
HOST_SHELL_DIR="${HOST_SHELL_DIR:-/tmp/devcontainer-host-shell-${WORKSPACE_NAME}}"
CONTAINER_HOME="/home/${USERNAME}"

mkdir -p "${CONTAINER_HOME}"

files=(
  ".zshrc"
  ".zshenv"
  ".zprofile"
  ".zaliases"
  ".zsh_history"
  ".bashrc"
  ".bash_profile"
  ".bash_aliases"
  ".bash_history"
  ".profile"
)

for file in "${files[@]}"; do
  src="${HOST_SHELL_DIR}/${file}"
  dst="${CONTAINER_HOME}/${file}"
  if [[ -f "${src}" ]]; then
    ln -sfn "${src}" "${dst}"
  else
    rm -f "${dst}"
  fi
done

if [[ -d "${HOST_SHELL_DIR}/.oh-my-zsh" ]]; then
  ln -sfn "${HOST_SHELL_DIR}/.oh-my-zsh" "${CONTAINER_HOME}/.oh-my-zsh"
else
  rm -rf "${CONTAINER_HOME}/.oh-my-zsh"
fi
