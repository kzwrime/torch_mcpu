#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

if ! command -v clang-format >/dev/null 2>&1; then
  echo "error: clang-format is not installed or not in PATH." >&2
  echo "Install clang-format first, then retry the commit." >&2
  exit 1
fi

supported_extensions=(
  c
  cc
  cpp
  cxx
  cu
  h
  hh
  hpp
  hxx
  inc
  inl
)

is_supported_file() {
  local path="$1"
  local ext="${path##*.}"

  case "${path}" in
    build/*|third_party/*)
      return 1
      ;;
  esac

  if [[ "${path}" == "${ext}" ]]; then
    return 1
  fi

  local candidate
  for candidate in "${supported_extensions[@]}"; do
    if [[ "${ext}" == "${candidate}" ]]; then
      return 0
    fi
  done

  return 1
}

mapfile -d '' staged_files < <(
  git diff --cached --name-only --diff-filter=ACMR -z
)

target_files=()
for file in "${staged_files[@]}"; do
  if is_supported_file "${file}"; then
    target_files+=("${file}")
  fi
done

if [[ "${#target_files[@]}" -eq 0 ]]; then
  exit 0
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

formatted_files=()
blocked_files=()

for file in "${target_files[@]}"; do
  if ! git diff --quiet -- "${file}"; then
    blocked_files+=("${file}")
    continue
  fi

  formatted_file="${tmp_dir}/$(basename "${file}").formatted"
  clang-format --assume-filename "${file}" "${file}" > "${formatted_file}"

  if ! cmp -s "${file}" "${formatted_file}"; then
    mv "${formatted_file}" "${file}"
    git add -- "${file}"
    formatted_files+=("${file}")
  else
    rm -f "${formatted_file}"
  fi
done

if [[ "${#blocked_files[@]}" -gt 0 ]]; then
  echo "clang-format check blocked: staged files below also have unstaged edits:" >&2
  printf '  - %s\n' "${blocked_files[@]}" >&2
  echo "Format or fully stage those files before committing." >&2
  exit 1
fi

if [[ "${#formatted_files[@]}" -gt 0 ]]; then
  echo "clang-format updated staged files:" >&2
  printf '  - %s\n' "${formatted_files[@]}" >&2
  echo "Review the formatting changes and re-run git commit." >&2
  exit 1
fi

exit 0
