#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hook_path="${repo_root}/.git/hooks/pre-commit"
managed_marker="# torch_mcpu clang-format pre-commit hook"

mkdir -p "$(dirname "${hook_path}")"

if [[ -f "${hook_path}" ]] && ! grep -Fq "${managed_marker}" "${hook_path}"; then
  backup_path="${hook_path}.bak.$(date +%Y%m%d%H%M%S)"
  cp "${hook_path}" "${backup_path}"
  echo "Backed up existing pre-commit hook to ${backup_path}"
fi

cat > "${hook_path}" <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

# torch_mcpu clang-format pre-commit hook
repo_root="$(git rev-parse --show-toplevel)"
exec "${repo_root}/scripts/run-clang-format-staged.sh"
EOF

chmod +x "${hook_path}"

echo "Installed pre-commit hook at ${hook_path}"
echo "The hook formats staged C/C++ files with clang-format before commit."
