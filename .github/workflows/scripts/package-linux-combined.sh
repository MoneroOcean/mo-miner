#!/usr/bin/env bash
# Package the Linux/Combined release: ONE mom.node that runs on both Intel and NVIDIA GPUs
# (mom_sycl_impl=dpcpp-combined). It is a dual-compiler build -- the CPU/host objects come from
# oneAPI icx and the SYCL device objects + final link from the intel/llvm nightly clang -- so its
# runtime closure is the UNION of the Intel and NVIDIA packages:
#   * nightly DPC++ (/opt/dpcpp): libsycl.so.9 + the UR adapters it dlopen()s (level_zero[_v2],
#     opencl, cuda) + libumf + the kernel_compiler JIT (libsycl-jit.so) for kawpow.
#   * oneAPI (/opt/intel/oneapi): the icx compiler runtime the host objects need (libimf, libsvml,
#     libintlc, libirc, libirng) AND the OpenCL CPU runtime (libintelocl.so + clbltfn*/cllibrary .rtl
#     device-builtin blobs) that backs the SYCL CPU device and cn/gpu's OpenCL path.
# The nightly libsycl.so.9 is bundled (NOT oneAPI's same-SONAME lib); the user's NVIDIA driver
# (libcuda.so.1 / libnvidia-ml.so.1) and glibc-base libs are provided by the host, never bundled.
set -euo pipefail

version="${1:-}"
if [ -z "$version" ] && [ -n "${GITHUB_REF_NAME:-}" ] && [[ "${GITHUB_REF_NAME}" =~ ^v?[0-9] ]]; then
  version="$GITHUB_REF_NAME"
fi
if [ -z "$version" ]; then
  version="$(node -p "require('./package.json').version")"
fi
version="${version#v}"

root="mom-v${version}"
# This unified build IS the Linux release (one tarball for both Intel and NVIDIA GPUs), so the
# user-facing artifact is just mom-v<ver>-lin.tgz -- no vendor/"combined" suffix.
archive="${2:-mom-v${version}-lin.tgz}"
package_dir="release-combined/${root}"
libs_dir="$package_dir/libs"
build_dir="release-combined-build"
node_bin="${NODE_BIN:-$(command -v node)}"
image="${MOM_COMBINED_IMAGE:-mom-build-combined}"
dpcpp_lib="${MOM_DPCPP_LIB:-/opt/dpcpp/lib}"
# oneAPI compiler lib dir inside the image (icx runtime + OpenCL CPU); 'latest' is a symlink to it.
oneapi_lib="${MOM_ONEAPI_LIB:-/opt/intel/oneapi/compiler/latest/lib}"

if [ ! -f build/Release/mom.node ]; then
  echo "build/Release/mom.node is missing; build the combined native addon before packaging." >&2
  exit 1
fi

rm -rf release-combined "$build_dir" "$archive"
mkdir -p "$package_dir" "$libs_dir" "$build_dir"

bundle_path="$PWD/$build_dir/mom.bundle.cjs"
blob_path="$PWD/$build_dir/mom.blob"
npx --yes esbuild@0.28.0 mom.js \
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
cp "$node_bin" "$package_dir/mom-bin"
npx --yes postject@1.0.0-alpha.6 "$package_dir/mom-bin" NODE_SEA_BLOB "$blob_path" \
  --sentinel-fuse NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2
chmod +x "$package_dir/mom-bin"
# Launcher: bundled libs first on the path (so the nightly libsycl wins), and point the OpenCL ICD
# loader at the bundled libintelocl so the SYCL CPU device / cn/gpu OpenCL path resolve.
cat >"$package_dir/mom" <<'EOF'
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

if [ -z "${OCL_ICD_FILENAMES:-}" ] && [ -f "$script_dir/libs/libintelocl.so" ]; then
  export OCL_ICD_FILENAMES="$script_dir/libs/libintelocl.so"
fi

exec "$script_dir/mom-bin" "$@"
EOF
chmod +x "$package_dir/mom"

cp package.json README.md LICENSE "$package_dir/"
# The combined binary runs on either vendor, so ship BOTH host-runtime installers: install.sh
# (Intel GPU runtime) and install-nvidia.sh (NVIDIA driver). The user runs the one for their GPU.
cp scripts/install.sh scripts/install-nvidia.sh "$package_dir/"
cp build/Release/mom.node "$libs_dir/"
# Device source the kawpow kernel_compiler JIT reads at runtime (resolves beside the loaded module).
cp sycl/kawpow_device.inc "$libs_dir/"

container="mom-combined-release-libs-$$"
docker rm -f "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" --entrypoint sleep "$image" infinity >/dev/null
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT

# Copy a file out of the build image, dereferencing symlinks; dest basename = SONAME.
copy_image_file() {
  local src="$1" dest="${2:-$libs_dir/$(basename "$1")}"
  [ -n "$src" ] && [ ! -e "$dest" ] || return 0
  docker cp -L "$container:$src" "$dest"
}

# Resolve a lib by SONAME in the image: nightly dir first (so its libsycl/adapters win over oneAPI's
# same-SONAME libs), then the oneAPI compiler lib, then a broad search of both trees.
find_image_lib() {
  docker exec "$container" bash -lc "
    for d in '$dpcpp_lib' '$oneapi_lib'; do [ -e \"\$d/$1\" ] && { echo \"\$d/$1\"; exit 0; }; done
    find /opt/dpcpp /opt/intel/oneapi -type f,l -name '$1' ! -name '*-gdb.py' -print -quit 2>/dev/null
  "
}
copy_image_name() {
  local src; src="$(find_image_lib "$1")"
  [ -n "$src" ] || { echo "Unable to find dependency $1 in the build image." >&2; exit 1; }
  copy_image_file "$src" "$libs_dir/$1"
}
# SONAMEs ldd can't resolve against libs/ alone (run on the HOST over the staged files; the packaging
# container has no repo mount, so mom.node is ldd'd here, not in the image).
missing_libraries() {
  local file
  while IFS= read -r -d "" file; do
    LD_LIBRARY_PATH="$PWD/$libs_dir" ldd "$file" 2>/dev/null || true
  done < <(find "$package_dir" "$libs_dir" -maxdepth 1 -type f \
            \( -name mom-bin -o -name mom.node -o -name "*.so" -o -name "*.so.*" \) -print0) \
    | awk '/not found/{print $1}' | sort -u
}
# Pull every still-missing non-base dependency from the image until ldd is clean (newly copied libs
# may add their own deps). Brings in the icx runtime (libimf/svml/intlc/irc/irng -- mom.node's own
# DT_NEEDED) and any transitive deps, from whichever tree they live in.
copy_missing_closure() {
  local copied_any=1 lib
  while [ "$copied_any" -eq 1 ]; do
    copied_any=0
    while IFS= read -r lib; do
      [ -n "$lib" ] || continue
      is_base_lib "$lib" && continue
      [ -e "$libs_dir/$lib" ] && continue
      copy_image_name "$lib"; copied_any=1
    done < <(missing_libraries)
  done
}

# Libs dlopen()'d by name (never in ldd output), so copied explicitly:
#  - nightly DPC++: the SYCL runtime, all UR adapters, libumf, the kawpow JIT lib.
#  - oneAPI: the OpenCL CPU ICD + its device-builtin .rtl blobs (loaded via OCL_ICD_FILENAMES).
dlopen_libs="$dpcpp_lib/libsycl.so.9 \
  $dpcpp_lib/libur_adapter_level_zero.so.0 $dpcpp_lib/libur_adapter_level_zero_v2.so.0 \
  $dpcpp_lib/libur_adapter_opencl.so.0 $dpcpp_lib/libur_adapter_cuda.so.0 \
  $dpcpp_lib/libumf.so.1 $dpcpp_lib/libsycl-jit.so \
  $oneapi_lib/libintelocl.so"

# Basename globs NOT to bundle: glibc-base (present everywhere) and the user-provided NVIDIA driver.
set -f; base_libs=(ld-linux* libc.so* libm.so* libdl.so* libpthread.so* librt.so* libresolv.so* libutil.so* libcuda.so* libnvidia-ml.so*); set +f
is_base_lib() {
  local pat
  for pat in "${base_libs[@]}"; do [[ "$1" == $pat ]] && return 0; done
  return 1
}

# 1. Copy the explicitly-dlopen'd libs.
for lib in $dlopen_libs; do copy_image_file "$lib"; done

# 1b. Bundle those libs' transitive closure resolved INSIDE the image, so deps the packaging HOST
#     happens to have but a clean target may not (notably libhwloc.so.15, which the Level-Zero UR
#     adapters need) get bundled. A host-side ldd masks these; this is what left the L0 adapters
#     unable to load on a clean Ubuntu box. base_libs (glibc + the user-provided driver) are skipped.
in_image_deps="$(docker exec "$container" bash -lc "
  export LD_LIBRARY_PATH='$dpcpp_lib:$oneapi_lib'
  for l in $dlopen_libs; do ldd \"\$l\" 2>/dev/null || true; done \
    | awk '/=>/ && \$3 ~ /^\//{print \$3}' | sort -u
")"
while IFS= read -r path; do
  [ -n "$path" ] || continue
  is_base_lib "$(basename "$path")" && continue
  copy_image_file "$path"
done <<<"$in_image_deps"
# 2. Copy the OpenCL CPU device-builtin blobs the JIT loads at runtime (beside libintelocl).
while IFS= read -r rtl; do
  [ -n "$rtl" ] && copy_image_file "$rtl"
done < <(docker exec "$container" bash -lc "ls $oneapi_lib/clbltfn*.rtl $oneapi_lib/cllibrary*.rtl $oneapi_lib/cllibrary*.o 2>/dev/null || true")

# 3. Resolve the rest of the closure: ldd the staged files on the host against libs/, pull each
#    still-missing non-base SONAME from the image, repeat until clean (see copy_missing_closure).
copy_missing_closure

# Sanity: nothing the bundled libs need (other than base_libs / the user driver) is missing from libs/.
unresolved="$(
  while IFS= read -r -d "" file; do
    LD_LIBRARY_PATH="$PWD/$libs_dir" ldd "$file" 2>/dev/null || true
  done < <(find "$libs_dir" -maxdepth 1 -type f \( -name "*.so" -o -name "*.so.*" -o -name "mom.node" \) -print0) \
    | awk '/not found/{print $1}' | sort -u \
    | while IFS= read -r missing; do is_base_lib "$missing" || printf '%s\n' "$missing"; done
)"
if [ -n "$unresolved" ]; then
  echo "Unresolved packaged Linux/Combined dependencies (excluding glibc-base and the user-provided NVIDIA driver):" >&2
  echo "$unresolved" >&2
  exit 1
fi

if [ -d "$package_dir/tests" ]; then echo "Release package unexpectedly contains tests/." >&2; exit 1; fi

tar -C release-combined -czf "$archive" "$root"
echo "$archive"
