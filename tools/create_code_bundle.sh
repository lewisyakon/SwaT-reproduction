#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:?usage: create_code_bundle.sh DESTINATION_DIRECTORY}"
repo_name="$(basename "${repo_root}")"
timestamp="$(date +%Y%m%dT%H%M%S)"

test -d "${repo_root}/.git"
install -d -m 0750 "${destination}"
bundle="${destination}/${repo_name}-${timestamp}.bundle"
git -C "${repo_root}" bundle create "${bundle}" --all
install -m 0640 "${repo_root}/migration/external_assets.json" \
  "${destination}/${repo_name}-${timestamp}.external_assets.json"
sha256sum "${bundle}" >"${bundle}.sha256"
printf 'Created %s\n' "${bundle}"

