#!/usr/bin/env bash
set -euo pipefail

# Install the latest upstream Intel GPU compute runtime used by mom's
# bundled SYCL runtime. This intentionally does not add an apt repository.

github_api_root="https://api.github.com/repos"

if [ "$(id -u)" -ne 0 ]; then
  exec sudo -n env \
    MOM_INTEL_RUNTIME_KEEP_DOWNLOADS="${MOM_INTEL_RUNTIME_KEEP_DOWNLOADS:-}" \
    MOM_COMPUTE_RUNTIME_RELEASE="${MOM_COMPUTE_RUNTIME_RELEASE:-}" \
    MOM_IGC_RELEASE="${MOM_IGC_RELEASE:-}" \
    MOM_LEVEL_ZERO_RELEASE="${MOM_LEVEL_ZERO_RELEASE:-}" \
    "$0" "$@"
fi

if [ "$(uname -m)" != "x86_64" ]; then
  echo "Intel GPU runtime installer supports x86_64 Linux only." >&2
  exit 1
fi

if [ ! -r /etc/os-release ]; then
  echo "/etc/os-release is missing; unable to detect Linux distribution." >&2
  exit 1
fi

. /etc/os-release
if [ "${ID:-}" != "ubuntu" ]; then
  echo "This installer supports Ubuntu 24.04 and 26.04 only; detected ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi

case "${VERSION_ID:-}" in
  24.*|26.*) ;;
  *)
    echo "This installer supports Ubuntu 24.04 and 26.04 only; detected ${PRETTY_NAME:-unknown}." >&2
    exit 1
    ;;
esac

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl python3 ocl-icd-libopencl1

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/mom-intel-runtime.XXXXXX")"
chmod 755 "$work_dir"
if [ "${MOM_INTEL_RUNTIME_KEEP_DOWNLOADS:-}" = "1" ]; then
  echo "Keeping downloads in $work_dir"
else
  trap 'rm -rf "$work_dir"' EXIT
fi

cd "$work_dir"

github_release_json() {
  local repo="$1"
  local release="${2:-}"
  local url
  if [ -n "$release" ]; then
    url="$github_api_root/$repo/releases/tags/$release"
  else
    url="$github_api_root/$repo/releases/latest"
  fi

  curl -fsSL --retry 3 --retry-delay 2 \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "$url"
}

select_asset() {
  local json_file="$1"
  shift
  python3 - "$json_file" "$@" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    release = json.load(fh)

assets = release.get("assets") or []
for pattern in sys.argv[2:]:
    rx = re.compile(pattern)
    for asset in assets:
        name = asset.get("name") or ""
        if rx.fullmatch(name):
            digest = asset.get("digest") or ""
            url = asset.get("browser_download_url") or ""
            if not url:
                continue
            print(f"{name}\t{digest}\t{url}")
            raise SystemExit(0)

print("Unable to find a matching asset in release "
      f"{release.get('tag_name', '<unknown>')}: {', '.join(sys.argv[2:])}",
      file=sys.stderr)
if assets:
    print("Available assets:", file=sys.stderr)
    for asset in assets:
        print("  " + (asset.get("name") or ""), file=sys.stderr)
raise SystemExit(1)
PY
}

download_asset() {
  local name="$1"
  local digest="$2"
  local url="$3"

  echo "Downloading $name" >&2
  curl -fL --retry 3 --retry-delay 2 -o "$name" "$url"
  if [[ "$digest" == sha256:* ]]; then
    printf '%s  %s\n' "${digest#sha256:}" "$name" | sha256sum -c - >&2
  else
    echo "Warning: GitHub did not provide a SHA-256 digest for $name." >&2
  fi
}

download_selected_asset() {
  local json_file="$1"
  shift
  local selected name digest url
  selected="$(select_asset "$json_file" "$@")"
  IFS=$'\t' read -r name digest url <<<"$selected"
  download_asset "$name" "$digest" "$url"
  printf '%s\n' "$name"
}

download_compute_runtime() {
  echo "Resolving Intel compute-runtime release..."
  github_release_json intel/compute-runtime "${MOM_COMPUTE_RUNTIME_RELEASE:-}" > compute-runtime.json
  python3 - <<'PY'
import json
with open("compute-runtime.json", encoding="utf-8") as fh:
    release = json.load(fh)
print("Using intel/compute-runtime " + release.get("tag_name", "<unknown>"))
PY

  compute_debs=()
  compute_debs+=("$(download_selected_asset compute-runtime.json 'libigdgmm[0-9]*_.+_amd64\.deb')")
  compute_debs+=("$(download_selected_asset compute-runtime.json 'intel-ocloc_.+_amd64\.deb')")
  compute_debs+=("$(download_selected_asset compute-runtime.json 'intel-opencl-icd_.+_amd64\.deb')")
  compute_debs+=("$(download_selected_asset compute-runtime.json 'libze-intel-gpu1_.+_amd64\.deb' 'intel-level-zero-gpu_.+_amd64\.deb')")
}

extract_igc_version() {
  local deb="$1"
  python3 - "$deb" <<'PY'
import re
import subprocess
import sys

text = subprocess.check_output(["dpkg-deb", "-I", sys.argv[1]], text=True)
matches = re.findall(r"intel-igc-(?:core|opencl)(?:-[0-9]+)?\s*\(>=\s*([^)]+)\)", text)
if matches:
    print(matches[0])
PY
}

download_igc_runtime() {
  local required_version=""
  local deb version release

  for deb in "${compute_debs[@]}"; do
    version="$(extract_igc_version "$deb" || true)"
    if [ -n "$version" ]; then
      required_version="$version"
      break
    fi
  done

  if [ -n "${MOM_IGC_RELEASE:-}" ]; then
    release="$MOM_IGC_RELEASE"
  elif [ -n "$required_version" ]; then
    release="v$required_version"
  else
    release=""
  fi

  echo "Resolving Intel Graphics Compiler release..."
  if ! github_release_json intel/intel-graphics-compiler "$release" > igc.json; then
    if [ -n "$release" ]; then
      echo "Unable to fetch IGC release $release; falling back to latest IGC release." >&2
      github_release_json intel/intel-graphics-compiler > igc.json
    else
      return 1
    fi
  fi

  python3 - <<'PY'
import json
with open("igc.json", encoding="utf-8") as fh:
    release = json.load(fh)
print("Using intel/intel-graphics-compiler " + release.get("tag_name", "<unknown>"))
PY

  igc_debs=()
  if [ -n "$required_version" ]; then
    igc_debs+=("$(download_selected_asset igc.json "intel-igc-core(?:-[0-9]+)?_${required_version//./\\.}.+_amd64\\.deb" 'intel-igc-core(?:-[0-9]+)?_.+_amd64\.deb')")
    igc_debs+=("$(download_selected_asset igc.json "intel-igc-opencl(?:-[0-9]+)?_${required_version//./\\.}.+_amd64\\.deb" 'intel-igc-opencl(?:-[0-9]+)?_.+_amd64\.deb')")
  else
    igc_debs+=("$(download_selected_asset igc.json 'intel-igc-core(?:-[0-9]+)?_.+_amd64\.deb')")
    igc_debs+=("$(download_selected_asset igc.json 'intel-igc-opencl(?:-[0-9]+)?_.+_amd64\.deb')")
  fi
}

download_level_zero_loader() {
  local ubuntu_asset_suffix="u${VERSION_ID}"

  echo "Resolving oneAPI Level Zero loader release..."
  github_release_json oneapi-src/level-zero "${MOM_LEVEL_ZERO_RELEASE:-}" > level-zero.json
  python3 - <<'PY'
import json
with open("level-zero.json", encoding="utf-8") as fh:
    release = json.load(fh)
print("Using oneapi-src/level-zero " + release.get("tag_name", "<unknown>"))
PY

  level_zero_deb="$(download_selected_asset level-zero.json \
    "libze1_.+\\+${ubuntu_asset_suffix}_amd64\\.deb" \
    "level-zero_.+\\+${ubuntu_asset_suffix}_amd64\\.deb" \
    'libze1_.+\+u24\.04_amd64\.deb' \
    'level-zero_.+\+u24\.04_amd64\.deb' \
    'libze1_.+\+u22\.04_amd64\.deb' \
    'level-zero_.+\+u22\.04_amd64\.deb')"
}

download_compute_runtime
download_igc_runtime
download_level_zero_loader

debs=("$level_zero_deb" "${compute_debs[@]}" "${igc_debs[@]}")

echo "Installing Intel GPU compute runtime packages:"
printf '  %s\n' "${debs[@]}"
apt-get install -y --no-install-recommends "${debs[@]/#/.\/}"
ldconfig

echo "Installed Intel GPU compute runtime:"
dpkg-query -W -f='${db:Status-Abbrev}\t${Package}\t${Version}\n' \
  libze1 level-zero libze-intel-gpu1 intel-level-zero-gpu intel-opencl-icd \
  'intel-igc-core*' 'intel-igc-opencl*' 'libigdgmm*' intel-ocloc 2>/dev/null |
  awk '$1 == "ii" {print $2 "\t" $3}' |
  sort
