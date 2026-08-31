#!/usr/bin/bash

set -Eeuo pipefail
umask 077
export LC_ALL=C
export PATH=/usr/bin:/bin

out=${1:?usage: capture_admission.sh OUTPUT_DIRECTORY}
package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
mapfile -t binding < <(
  /usr/bin/python3 -I "$package_root/scripts/package_manifest.py" \
    "$package_root" --capture-fields
)
[[ ${#binding[@]} -eq 4 ]] || {
  printf 'cannot load package capture binding\n' >&2
  exit 2
}
readonly reservation_id=${binding[0]}
readonly target_entity=${binding[1]}
readonly target_host=${binding[2]}
readonly operator_email=${binding[3]}

[[ $out == /* ]] || {
  printf 'admission output path must be absolute\n' >&2
  exit 2
}
[[ $out =~ ^/[A-Za-z0-9._/-]+$ && $out != *//* && $out != */../* && $out != */./* ]] || {
  printf 'admission output path is not canonical and safe\n' >&2
  exit 2
}
[[ ! -e $out ]] || {
  printf 'admission output already exists: %s\n' "$out" >&2
  exit 2
}

key=${CONDUCTOR_API_KEY:-}
if [[ -z $key && -r $HOME/.bashrc ]]; then
  key=$(awk '/^export CONDUCTOR_API_KEY=/ {
    sub(/^export CONDUCTOR_API_KEY=/, "")
    gsub(/^"|"$/, "")
    print
    exit
  }' "$HOME/.bashrc")
fi
[[ -n $key ]] || {
  printf 'CONDUCTOR_API_KEY is unavailable\n' >&2
  exit 2
}
uv_bin=${UV_BIN:-$HOME/.local/bin/uv}
[[ -x $uv_bin ]] || {
  printf 'uv is unavailable: %s\n' "$uv_bin" >&2
  exit 2
}

tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
conduct=(
  env
  CONDUCTOR_API_KEY_EMAIL="$operator_email"
  CONDUCTOR_EMAIL="$operator_email"
  AMD_EMAIL="$operator_email"
  ATS_EMAIL="$operator_email"
  ATS_SECRET="$key"
  VERIFY_CERTS=False
  REQUEST_TIMEOUT=120
  DISABLE_CLI_VERSION_CHECK=True
  PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring
  "$uv_bin" tool run
  --from amd-conductor-cli
  --extra-index-url
  https://mkmartifactory.amd.com/artifactory/api/pypi/hw-orc3pypi-prod-local/simple
  --allow-insecure-host mkmartifactory.amd.com
  conduct --output json --non-interactive
)

"${conduct[@]}" whoami >"$tmp/whoami.raw"
"${conduct[@]}" list reservation \
  --entity "$target_entity" --state IN_PROGRESS --all --json \
  >"$tmp/reservation.raw"
"${conduct[@]}" list jobs-v2 \
  --system "$target_entity" \
  --state INTERPRETING --state QUEUED --state PENDING --state RUNNING \
  --all --json >"$tmp/jobs.raw"

captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -m 700 "$out"
/usr/bin/python3 -I "$package_root/scripts/normalize_admission.py" \
  "$tmp/whoami.raw" "$tmp/reservation.raw" "$tmp/jobs.raw" \
  "$out" "$captured_utc"
