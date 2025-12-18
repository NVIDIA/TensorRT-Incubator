#!/usr/bin/env bash
set -euo pipefail
set -x

setup_git() {
  cd "${GITHUB_WORKSPACE}"
  git config --global --add safe.directory "${GITHUB_WORKSPACE}"
}

compute_range() {
  local event_name="${GITHUB_EVENT_NAME:-}"
  local default_base="${GITHUB_DEFAULT_BRANCH:-main}"
  local range=""

  if [ "${event_name}" = "push" ]; then
    local from_sha="${GITHUB_BEFORE:-}"
    local to_sha="${GITHUB_SHA:-}"
    if [ -z "${from_sha}" ] || [ "${from_sha}" = "0000000000000000000000000000000000000000" ]; then
      git fetch --no-tags --prune --depth=100 origin "${default_base}"
      range="origin/${default_base}..${to_sha}"
    else
      range="${from_sha}..${to_sha}"
    fi
  else
    local base_ref="${GITHUB_BASE_REF:-${GITHUB_DEFAULT_BRANCH:-main}}"
    git fetch --no-tags --prune --depth=100 origin "${base_ref}"
    if git merge-base "origin/${base_ref}" HEAD >/dev/null 2>&1; then
      range="origin/${base_ref}...HEAD"
    else
      range="origin/${base_ref}..HEAD"
    fi
  fi

  echo "${range}"
}

cmd_detect_code_change() {
  setup_git

  local event_name="${GITHUB_EVENT_NAME:-}"
  local ref_type="${GITHUB_REF_TYPE:-}"
  if [ "${event_name}" = "schedule" ] || [ "${event_name}" = "workflow_dispatch" ] || [ "${ref_type:-}" = "tag" ]; then
    echo "github.event_name: ${event_name} or github.ref_type: ${ref_type}"
    return 0
  fi

  local range
  range="$(compute_range)"

  set +e
  local diff_output
  diff_output="$(git diff --name-only "${range}")"
  local diff_status=$?
  set -e
  if [ ${diff_status} -ne 0 ]; then
    echo "git diff failed for RANGE='${range}'" >&2
    return 0
  else
    if echo "${diff_output}" | grep -Eq '^(mlir-tensorrt/|\.github/workflows/mlir-tensorrt[^/]*\.yml|\.github/workflows/mlir-tensorrt/)'; then
      return 0
    else
      return 1
    fi
  fi
}

cmd_lint_check() {
  setup_git

  local range
  range="$(compute_range)"

  uv tool install black
  uvx black --check --extend-exclude='.*\.pyi' mlir-tensorrt/compiler/ mlir-tensorrt/integrations

  local clang_format_diff
  clang_format_diff=$(git clang-format-20 ${range} --diff -q)
  if [ -n "${clang_format_diff}" ]; then
    echo "${clang_format_diff}"
    echo "Error: C++ files are not properly formatted. Run 'git clang-format ${BASE_COMMIT}' to fix."
    exit 1
  fi
}

cmd_release_check() {
    local ref_type="${GITHUB_REF_TYPE:-}"
    VERSION_FILE="mlir-tensorrt/Version.cmake"
    if [ ! -f "${VERSION_FILE}" ]; then
        echo "Error: ${VERSION_FILE} not found"
        exit 1
    fi
    if [ "${ref_type}" != "tag" ]; then
        echo "Error: Not a release tag"
        exit 1
    fi
    local tag="${GITHUB_REF_NAME:-}"

    major=$(awk -F'"' '/MLIR_TENSORRT_VERSION_MAJOR / {print $2}' "${VERSION_FILE}")
    minor=$(awk -F'"' '/MLIR_TENSORRT_VERSION_MINOR / {print $2}' "${VERSION_FILE}")
    patch=$(awk -F'"' '/MLIR_TENSORRT_VERSION_PATCH / {print $2}' "${VERSION_FILE}")
    expected_tag="mlir-tensorrt-v${major}.${minor}.${patch}"

    case "${tag}" in
        "${expected_tag}"|"${expected_tag}"*)
            echo "Tag ${tag} matches expected tag ${expected_tag} from ${VERSION_FILE}"
            ;;
        *)
            # tag mismatch
            echo "Error: Tag ${tag} does not match expected tag ${expected_tag} from ${VERSION_FILE}"
            exit 1
            ;;
    esac
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    detect-code-change)
      cmd_detect_code_change
      ;;
    lint-check)
      cmd_lint_check
      ;;
    release-check)
      cmd_release_check
      ;;
    *)
      echo "Usage: $0 {detect-code-change|lint-check|release-check}"
      exit 2
      ;;
  esac
}

main "$@"


