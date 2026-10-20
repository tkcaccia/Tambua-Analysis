#!/usr/bin/env bash
# Download the public HEMIT archive and extract its official test split only.
set -euo pipefail

destination=${1:?Usage: download_hemit.sh DESTINATION_DIR}
mkdir -p "$destination"
archive="$destination/HEMIT.7z"
expected=efe7dd5dc50ec0f789b0ce8a5258f5281a38001ac3dca81620f2a4be7041c943
url=https://data.mendeley.com/public-files/datasets/3gx53zm49d/files/d197d039-397e-4df7-a990-0e7c79a10f47/file_downloaded

if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 3 --continue-at - --output "$archive" "$url"
fi
if command -v sha256sum >/dev/null; then
  actual=$(sha256sum "$archive" | cut -d ' ' -f 1)
else
  actual=$(shasum -a 256 "$archive" | cut -d ' ' -f 1)
fi
if [[ "$actual" != "$expected" ]]; then
  echo "HEMIT archive checksum mismatch: $actual" >&2
  exit 1
fi
7z x -y "$archive" "test/input/*" "test/label/*" -o"$destination/extracted"
echo "Verified public HEMIT archive and extracted its official test split."
