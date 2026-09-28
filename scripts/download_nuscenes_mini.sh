#!/usr/bin/env bash
set -euo pipefail

project=${1:-$PWD}
data_root="$project/data/nuscenes"
archive="$data_root/v1.0-mini.tgz"
marker="$data_root/.v1.0-mini.complete"
url=https://www.nuscenes.org/data/v1.0-mini.tgz
expected_bytes=4167696325

mkdir -p "$data_root"
if [[ -s "$marker" ]]; then
  echo "nuscenes_mini=already_complete marker=$marker"
  exit 0
fi

curl --fail --location --retry 10 --retry-delay 10 --continue-at - \
  --output "$archive" "$url"
actual_bytes=$(stat -c '%s' "$archive")
if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
  echo "archive size mismatch: expected=$expected_bytes actual=$actual_bytes" >&2
  exit 1
fi

tar -tzf "$archive" >/dev/null
tar -xzf "$archive" -C "$data_root"
test -s "$data_root/v1.0-mini/sample.json"
test -d "$data_root/samples/CAM_FRONT"
sha256=$(sha256sum "$archive" | awk '{print $1}')
{
  echo "url=$url"
  echo "bytes=$actual_bytes"
  echo "sha256=$sha256"
} >"$marker"
echo "nuscenes_mini=complete root=$data_root sha256=$sha256"
