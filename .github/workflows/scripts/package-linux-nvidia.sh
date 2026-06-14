#!/usr/bin/env bash
# Package the Linux/NVIDIA (AdaptiveCpp) release. Mirrors package-linux.sh but
# bundles the AdaptiveCpp + LLVM + CUDA-libdevice runtime instead of the Intel
# oneAPI runtime. The generic-SSCP JIT needs libacpp-rt/common, the hipSYCL
# plugin tree (rt-backend-*, llvm-to-backend/*), libLLVM (IR->PTX at runtime) and
# libdevice.10.bc; only the user's NVIDIA driver (libcuda.so.1) is required at
# runtime, not the CUDA toolkit. libdevice is placed in the acpp redist bitcode
# path (hipSYCL/bitcode/ptx) so it is found without a CUDA install.
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

if [ ! -f build/Release/mo-miner.node ]; then
  echo "build/Release/mo-miner.node is missing; build the native addon (acpp) before packaging." >&2
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

container="mo-miner-nvidia-release-libs-$$"
docker rm -f "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" --entrypoint sleep "$image" infinity >/dev/null
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT

# Copy a file out of the build image, dereferencing symlinks to the real target.
copy_image_file() {
  local src="$1" dest="${2:-}"
  [ -n "$src" ] || return 0
  [ -n "$dest" ] || dest="$libs_dir/$(basename "$src")"
  [ -e "$dest" ] && return 0
  docker cp -L "$container:$src" "$dest"
}

# 1. AdaptiveCpp runtime + the whole hipSYCL plugin/bitcode tree (preserve layout
#    next to libacpp-rt so the runtime finds rt-backend-* / llvm-to-backend / bitcode).
copy_image_file /opt/adaptivecpp/lib/libacpp-rt.so
copy_image_file /opt/adaptivecpp/lib/libacpp-common.so
docker cp -L "$container:/opt/adaptivecpp/lib/hipSYCL" "$libs_dir/hipSYCL"

# 2. libLLVM (the generic JIT lowers IR -> PTX at runtime through it).
llvm_so="$(docker exec "$container" bash -lc 'ls /usr/lib/llvm-*/lib/libLLVM.so.* 2>/dev/null | grep -E "libLLVM\.so\.[0-9]" | head -1')"
copy_image_file "$llvm_so"

# acpp resolves its redistributable-package path as <lib-dir>/hipSYCL/ext, so the
# JIT looks for bitcode under ext/bitcode/<backend> and LLVM tools under
# ext/llvm/bin. Place the bundled libdevice and opt/llc/lld there.
ext="$libs_dir/hipSYCL/ext"

# 3. CUDA libdevice into the acpp redist bitcode path so PTX JIT needs no CUDA toolkit.
mkdir -p "$ext/bitcode/ptx"
libdevice="$(docker exec "$container" bash -lc 'ls /usr/local/cuda*/nvvm/libdevice/libdevice.*.bc 2>/dev/null | head -1')"
copy_image_file "$libdevice" "$ext/bitcode/ptx/$(basename "$libdevice")"

# 3b. LLVM tools (opt/llc/ld.lld) the JIT shells out to at runtime (both PTX and
#     host backends), placed in the acpp LLVM redist path (ext/llvm/bin) so
#     getOptPath/getLLCPath/getLLDPath find them with no system LLVM install.
mkdir -p "$ext/llvm/bin"
for tool in opt llc ld.lld; do
  tsrc="$(docker exec "$container" bash -lc "ls /usr/lib/llvm-*/bin/$tool 2>/dev/null | head -1")"
  copy_image_file "$tsrc" "$ext/llvm/bin/$tool"
done

# 4. Bundle the full transitive dependency closure of the acpp/LLVM libs, resolved
#    INSIDE the build image (ldd is transitive) so the bundled libs match the
#    build environment rather than the packaging host. Skip only glibc-base libs
#    (present on every target) and the user-provided libcuda driver, so the
#    release runs on a clean machine that has just the NVIDIA driver.
deps="$(docker exec "$container" bash -lc '
  set -e
  llvm="$(ls /usr/lib/llvm-*/lib/libLLVM.so.* 2>/dev/null | grep -E "libLLVM\.so\.[0-9]" | head -1)"
  tools="$(ls /usr/lib/llvm-*/bin/opt /usr/lib/llvm-*/bin/llc /usr/lib/llvm-*/bin/ld.lld 2>/dev/null)"
  libs="/opt/adaptivecpp/lib/libacpp-rt.so /opt/adaptivecpp/lib/libacpp-common.so $llvm $tools $(find /opt/adaptivecpp/lib/hipSYCL -name "*.so" 2>/dev/null)"
  for l in $libs; do ldd "$l" 2>/dev/null || true; done \
    | awk "/=>/ && \$3 ~ /^\// {print \$3}" | sort -u
')"
while IFS= read -r path; do
  [ -n "$path" ] || continue
  base="$(basename "$path")"
  case "$base" in
    ld-linux*|libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libresolv.so*|libutil.so*|libcuda.so*) continue ;;
  esac
  copy_image_file "$path"
done <<<"$deps"

# Sanity check: nothing the bundled libs need (other than glibc/libcuda) is absent
# from libs/. Resolved against libs/ only, so it catches gaps a clean target hits.
unresolved="$(
  while IFS= read -r -d "" file; do
    LD_LIBRARY_PATH="$PWD/$libs_dir:$PWD/$libs_dir/hipSYCL:$PWD/$libs_dir/hipSYCL/llvm-to-backend" \
      ldd "$file" 2>/dev/null || true
  done < <(find "$libs_dir" -maxdepth 2 -type f \( -name "*.so" -o -name "*.so.*" \) -print0) \
    | awk '/not found/{print $1}' \
    | { grep -vE "^(libcuda\.so|ld-linux|libc\.so|libm\.so|libdl\.so|libpthread\.so|librt\.so|libresolv\.so|libutil\.so)" || true; } \
    | sort -u
)"
if [ -n "$unresolved" ]; then
  echo "Unresolved packaged Linux/NVIDIA dependencies (excluding glibc-base and the user-provided libcuda):" >&2
  echo "$unresolved" >&2
  exit 1
fi

# opt/llc/ld.lld (at ext/llvm/bin) resolve their libraries through their RUNPATH
# ($ORIGIN/../lib = ext/llvm/lib); point that at the bundled libs via relative
# symlinks so the JIT tools run even when spawned without LD_LIBRARY_PATH.
mkdir -p "$libs_dir/hipSYCL/ext/llvm/lib"
for so in "$libs_dir"/*.so*; do
  [ -e "$so" ] || continue
  ln -sf "../../../../$(basename "$so")" "$libs_dir/hipSYCL/ext/llvm/lib/$(basename "$so")"
done

if [ -d "$package_dir/tests" ]; then
  echo "Release package unexpectedly contains tests/." >&2
  exit 1
fi

tar -C release-nvidia -czf "$archive" "$root"
echo "$archive"
