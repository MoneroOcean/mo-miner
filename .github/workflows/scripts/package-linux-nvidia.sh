#!/usr/bin/env bash
# Package the Linux/NVIDIA (DPC++ CUDA backend) release. Mirrors package-linux.sh but
# bundles the DPC++ SYCL runtime instead of the Intel oneAPI runtime: libsycl.so.9, the
# CUDA Unified Runtime adapter (libur_adapter_cuda.so.0 + libumf.so.1), and the
# kernel_compiler JIT library (libsycl-jit.so) that the kawpow algo compiles its device
# source through at runtime, plus the device source itself (kawpow_device.inc). The UR
# loader is linked inside libsycl, so no separate loader file is needed. Only the user's
# NVIDIA driver (libcuda.so.1) is required at runtime, not the CUDA toolkit.
set -euo pipefail

version="${1:-}"
if [ -z "$version" ] && [ "${GITHUB_REF_NAME:-}" != "" ] && [[ "${GITHUB_REF_NAME}" =~ ^v?[0-9] ]]; then
  version="$GITHUB_REF_NAME"
fi
if [ -z "$version" ]; then
  version="$(node -p "require('./package.json').version")"
fi
version="${version#v}"

root="mo-miner-v${version}"
archive="${2:-mo-miner-v${version}-lin-nvidia.tgz}"
package_dir="release-nvidia/${root}"
libs_dir="$package_dir/libs"
build_dir="release-nvidia-build"
node_bin="${NODE_BIN:-$(command -v node)}"
image="${MOMINER_NVIDIA_IMAGE:-mo-miner-build-nvidia}"
dpcpp_lib="${MOMINER_DPCPP_LIB:-/opt/dpcpp/lib}"

if [ ! -f build/Release/mo-miner.node ]; then
  echo "build/Release/mo-miner.node is missing; build the native addon (dpcpp-cuda) before packaging." >&2
  exit 1
fi

rm -rf release-nvidia "$build_dir" "$archive"
mkdir -p "$package_dir" "$libs_dir" "$build_dir"

bundle_path="$PWD/$build_dir/mo-miner.bundle.cjs"
blob_path="$PWD/$build_dir/mo-miner.blob"
npx --yes esbuild@0.28.0 mo-miner.js \
  --bundle --platform=node --format=cjs \
  --banner:js="const { createRequire } = require('node:module'); require = createRequire(process.execPath);" \
  --outfile="$bundle_path"
cat >"$build_dir/sea-config.json" <<EOF
{
  "main": "$bundle_path",
  "output": "$blob_path",
  "disableExperimentalSEAWarning": true,
  "useCodeCache": false,
  "useSnapshot": false
}
EOF
"$node_bin" --experimental-sea-config "$build_dir/sea-config.json"
cp "$node_bin" "$package_dir/mo-miner-bin"
npx --yes postject@1.0.0-alpha.6 "$package_dir/mo-miner-bin" NODE_SEA_BLOB "$blob_path" \
  --sentinel-fuse NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2
chmod +x "$package_dir/mo-miner-bin"
cat >"$package_dir/mo-miner" <<'EOF'
#!/usr/bin/env sh
set -eu

case "$0" in
  */*) script_dir=${0%/*} ;;
  *) script_dir=$(dirname "$(command -v "$0")") ;;
esac
script_dir=$(CDPATH= cd -- "$script_dir" && pwd -P)

library_dirs="$script_dir/libs:$script_dir:$(pwd)/libs:$(pwd)"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$library_dirs:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$library_dirs"
fi

exec "$script_dir/mo-miner-bin" "$@"
EOF
chmod +x "$package_dir/mo-miner"

cp package.json README.md LICENSE "$package_dir/"
cp scripts/install-nvidia.sh "$package_dir/install.sh"
cp build/Release/mo-miner.node "$libs_dir/"
# Device source the kawpow kernel_compiler JIT reads at runtime; it resolves beside the
# loaded module (libs/) via dladdr (see sycl/kawpow_jit.inc kawpow_module_dir()).
cp sycl/kawpow_device.inc "$libs_dir/"

container="mo-miner-nvidia-release-libs-$$"
docker rm -f "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" --entrypoint sleep "$image" infinity >/dev/null
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT

# Copy a file out of the build image, dereferencing symlinks to the real target. The dest
# basename is the SONAME (e.g. libsycl.so.9), which is what DT_NEEDED / dlopen() ask for.
copy_image_file() {
  local src="$1" dest="${2:-}"
  [ -n "$src" ] || return 0
  [ -n "$dest" ] || dest="$libs_dir/$(basename "$src")"
  [ -e "$dest" ] && return 0
  docker cp -L "$container:$src" "$dest"
}

# 1. The SYCL runtime + everything libsycl dlopen()s for the CUDA backend. These are loaded
#    by name at runtime, so they are NOT in the addon's ldd output and must be copied
#    explicitly: the SYCL runtime, the CUDA UR adapter (+ its libumf dep), and the
#    kernel_compiler JIT library (kawpow compiles its device source through it).
copy_image_file "$dpcpp_lib/libsycl.so.9"
copy_image_file "$dpcpp_lib/libur_adapter_cuda.so.0"
copy_image_file "$dpcpp_lib/libumf.so.1"
copy_image_file "$dpcpp_lib/libsycl-jit.so"

# 2. Bundle the transitive dependency closure of those libs, resolved INSIDE the build image
#    (ldd is transitive) so the bundled libs match the build environment. This picks up
#    libhwloc, libstdc++/libgcc and libz. Skip glibc-base libs (present on every target) and
#    the user-provided libcuda driver, so the release runs on a machine with just the driver.
deps="$(docker exec "$container" bash -lc "
  set -e
  libs=\"$dpcpp_lib/libsycl.so.9 $dpcpp_lib/libur_adapter_cuda.so.0 $dpcpp_lib/libumf.so.1 $dpcpp_lib/libsycl-jit.so\"
  for l in \$libs; do ldd \"\$l\" 2>/dev/null || true; done \
    | awk '/=>/ && \$3 ~ /^\// {print \$3}' | sort -u
")"
while IFS= read -r path; do
  [ -n "$path" ] || continue
  base="$(basename "$path")"
  case "$base" in
    ld-linux*|libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libresolv.so*|libutil.so*|libcuda.so*|libnvidia-ml.so*) continue ;;
  esac
  copy_image_file "$path"
done <<<"$deps"

# Sanity check: nothing the bundled libs need (other than glibc/libcuda) is absent from
# libs/. Resolved against libs/ only, so it catches gaps a clean target would hit.
unresolved="$(
  while IFS= read -r -d "" file; do
    LD_LIBRARY_PATH="$PWD/$libs_dir" ldd "$file" 2>/dev/null || true
  done < <(find "$libs_dir" -maxdepth 1 -type f \( -name "*.so" -o -name "*.so.*" \) -print0) \
    | awk '/not found/{print $1}' \
    | { grep -vE "^(libcuda\.so|libnvidia-ml\.so|ld-linux|libc\.so|libm\.so|libdl\.so|libpthread\.so|librt\.so|libresolv\.so|libutil\.so)" || true; } \
    | sort -u
)"
if [ -n "$unresolved" ]; then
  echo "Unresolved packaged Linux/NVIDIA dependencies (excluding glibc-base and the user-provided libcuda):" >&2
  echo "$unresolved" >&2
  exit 1
fi

if [ -d "$package_dir/tests" ]; then
  echo "Release package unexpectedly contains tests/." >&2
  exit 1
fi

tar -C release-nvidia -czf "$archive" "$root"
echo "$archive"
