#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

clang_format_bin=""
for candidate in clang-format-18 clang-format-17 clang-format-16 clang-format-15 clang-format; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    clang_format_bin="${candidate}"
    break
  fi
done

if [[ -z "${clang_format_bin}" ]]; then
  echo "error: clang-format is not installed or not in PATH." >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  ./scripts/format.sh
  ./scripts/format.sh --check
  ./scripts/format.sh <files...>
  ./scripts/format.sh --check <files...>

Behavior:
  - Formats repository C/C++ sources with clang-format.
  - Excludes third_party/googletest when scanning the whole repository.
  - Ignores explicit file arguments under third_party/googletest as well.
EOF
}

check_mode=false
declare -a requested_files=()

for arg in "$@"; do
  case "${arg}" in
    --check)
      check_mode=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      requested_files+=("${arg}")
      ;;
  esac
done

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

is_excluded_path() {
  local path="$1"

  case "${path}" in
    third_party/googletest/*|./third_party/googletest/*)
      return 0
      ;;
  esac

  return 1
}

normalize_path() {
  local path="$1"
  path="${path#./}"
  printf '%s\n' "${path}"
}

declare -a target_files=()

if [[ "${#requested_files[@]}" -eq 0 ]]; then
  while IFS= read -r file; do
    target_files+=("${file}")
  done < <(
    find . \
      \( -path './.git' -o -path './build' -o -path './third_party/googletest' \) -prune \
      -o -type f \
      \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.cu' -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' -o -name '*.inc' -o -name '*.inl' \) \
      -print | sort
  )
else
  for file in "${requested_files[@]}"; do
    normalized_file="$(normalize_path "${file}")"

    if is_excluded_path "${normalized_file}"; then
      echo "skip excluded path: ${file}"
      continue
    fi

    if [[ ! -f "${normalized_file}" ]]; then
      echo "skip missing file: ${file}" >&2
      continue
    fi

    if ! is_supported_file "${normalized_file}"; then
      echo "skip unsupported file: ${file}"
      continue
    fi

    target_files+=("${normalized_file}")
  done
fi

if [[ "${#target_files[@]}" -eq 0 ]]; then
  echo "No matching C/C++ files to process."
  exit 0
fi

if [[ "${check_mode}" == true ]]; then
  echo "Checking ${#target_files[@]} file(s) with ${clang_format_bin}..."
  failed=0
  for file in "${target_files[@]}"; do
    if ! "${clang_format_bin}" --dry-run --Werror --assume-filename "${file}" "${file}" >/dev/null 2>&1; then
      echo "needs formatting: ${file}"
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    exit 1
  fi

  echo "All checked files are properly formatted."
  exit 0
fi

echo "Formatting ${#target_files[@]} file(s) with ${clang_format_bin}..."
for file in "${target_files[@]}"; do
  "${clang_format_bin}" -i --assume-filename "${file}" "${file}"
done

echo "Formatting complete."
