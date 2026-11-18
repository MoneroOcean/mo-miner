#!/usr/bin/env bash
# Publish exactly the two packaged archives whose SHA-256 the package jobs reported, plus .sha256
# sidecars. The tag must equal package.json's version. Re-runs refresh an existing release.
set -euo pipefail

root="${1:?usage: publish-release.sh <download-directory>}"
for expected in "${EXPECTED_LINUX_ARCHIVE_SHA256:-}" "${EXPECTED_WINDOWS_ARCHIVE_SHA256:-}"; do
  if [[ ! "$expected" =~ ^[a-f0-9]{64}$ ]]; then
    printf 'Missing or malformed package archive SHA-256 output.\n' >&2
    exit 1
  fi
done
version="$(node -p 'require("./package.json").version')"
case "$version" in
  ''|*[!0-9A-Za-z.-]*) printf 'Invalid package version: %s\n' "$version" >&2; exit 1 ;;
esac
tag="v$version"
if [[ "${GITHUB_REF_NAME:-}" != "$tag" ]]; then
  printf 'Release tag does not match package version: expected %s\n' "$tag" >&2
  exit 1
fi
linux_name="mom-${tag}-lin.tgz"
windows_name="mom-${tag}-win.zip"
mapfile -d '' archives < <(find "$root" -type f -print0)
mapfile -d '' linux_files < <(find "$root" -type f -name "$linux_name" -print0)
mapfile -d '' windows_files < <(find "$root" -type f -name "$windows_name" -print0)
if (( ${#archives[@]} != 2 || ${#linux_files[@]} != 1 || ${#windows_files[@]} != 1 )); then
  printf 'Expected exactly one unified Linux tgz and one unified Windows zip; found:\n' >&2
  printf '  %s\n' "${archives[@]}" >&2
  exit 1
fi
if [[ "$(sha256sum -- "${linux_files[0]}" | awk '{print $1}')" != "$EXPECTED_LINUX_ARCHIVE_SHA256" ||
      "$(sha256sum -- "${windows_files[0]}" | awk '{print $1}')" != "$EXPECTED_WINDOWS_ARCHIVE_SHA256" ]]; then
  printf 'Downloaded release archive bytes differ from the package workflow outputs.\n' >&2
  exit 1
fi
files=()
for archive in "${linux_files[0]}" "${windows_files[0]}"; do
  directory="$(dirname "$archive")"
  filename="$(basename "$archive")"
  (cd "$directory" && sha256sum "$filename" > "$filename.sha256" && sha256sum -c "$filename.sha256" >/dev/null)
  files+=("$archive" "$archive.sha256")
done
meta=(--repo "$GITHUB_REPOSITORY" --title "$GITHUB_REF_NAME" --notes "mom $GITHUB_REF_NAME")
if gh release view "$GITHUB_REF_NAME" --repo "$GITHUB_REPOSITORY" >/dev/null 2>&1; then
  gh release edit "$GITHUB_REF_NAME" "${meta[@]}"
  gh release upload "$GITHUB_REF_NAME" "${files[@]}" --repo "$GITHUB_REPOSITORY" --clobber
else
  gh release create "$GITHUB_REF_NAME" "${files[@]}" "${meta[@]}"
fi
