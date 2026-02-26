#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${1:?workspace root is required}"
WORKSPACE_NAME="$(basename "${WORKSPACE_ROOT}")"
HOST_SHELL_DIR="${HOST_SHELL_DIR:-/tmp/devcontainer-host-shell-${WORKSPACE_NAME}}"
HOST_HOME="${HOME:-${USERPROFILE:-}}"

if [[ -z "${HOST_HOME}" ]]; then
  echo "Unable to determine host home directory (HOME/USERPROFILE is unset)."
  exit 1
fi

mkdir -p "${HOST_SHELL_DIR}"

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
  src="${HOST_HOME}/${file}"
  dst="${HOST_SHELL_DIR}/${file}"
  if [[ -f "${src}" ]]; then
    cp "${src}" "${dst}"
  else
    rm -f "${dst}"
  fi
done

if [[ -d "${HOST_HOME}/.oh-my-zsh" ]]; then
  rm -rf "${HOST_SHELL_DIR}/.oh-my-zsh"
  cp -R "${HOST_HOME}/.oh-my-zsh" "${HOST_SHELL_DIR}/.oh-my-zsh"
else
  rm -rf "${HOST_SHELL_DIR}/.oh-my-zsh"
fi
