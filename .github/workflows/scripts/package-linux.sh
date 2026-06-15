#!/usr/bin/env bash
# Package the Linux/Intel release: build a single-executable (Node SEA) binary, then bundle
# the Intel oneAPI SYCL/OpenCL runtime closure (extracted from the mom-build image) into
# libs/ so the tarball runs with only the user's GPU driver installed.
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
archive="${2:-mom-v${version}-lin.tgz}"
package_dir="release/${root}"
libs_dir="$package_dir/libs"
build_dir="release-build"
node_bin="${NODE_BIN:-$(command -v node)}"

if [ ! -f build/Release/mom.node ]; then
  echo "build/Release/mom.node is missing; build the native addon before packaging." >&2
  exit 1
fi

rm -rf release "$build_dir" "$archive"
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
cp scripts/install.sh "$package_dir/install.sh"
cp build/Release/mom.node "$libs_dir/"

container="mom-release-libs-$$"
docker rm -f "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" --entrypoint sleep mom-build infinity >/dev/null
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT

find_oneapi_path() {
  local quoted
  quoted="$(printf "%q" "$1")"
  docker exec "$container" bash -lc \
    "find /opt/intel/oneapi -type f,l -name $quoted ! -name '*-gdb.py' -print -quit"
}

# Copy a file out of the build image into libs/ (skipping if it already exists), with
# docker cp -L dereferencing symlinks to their real target.
copy_container_file() {
  local src="$1" dest="${2:-$(basename "$1")}"
  [ -n "$src" ] && [ ! -f "$libs_dir/$dest" ] || return 0
  docker cp -L "$container:$src" "$libs_dir/$dest"
}

# Copy a runtime path, preserving the SONAME symlink layout the loader expects: if the
# source is a symlink, copy its real target under the target basename and recreate the
# symlink in libs/. The -L check (vs -e) guards against an existing broken symlink.
copy_container_runtime_path() {
  local src="$1" dest resolved target quoted
  dest="$(basename "$src")"
  [ -n "$src" ] && [ ! -e "$libs_dir/$dest" ] && [ ! -L "$libs_dir/$dest" ] || return 0

  if docker exec "$container" test -L "$src"; then
    quoted="$(printf "%q" "$src")"
    resolved="$(docker exec "$container" bash -lc "readlink -f $quoted")"
    target="$(basename "$resolved")"
    copy_container_file "$resolved" "$target"
    [ "$dest" = "$target" ] || ln -s "./$target" "$libs_dir/$dest"
  else
    copy_container_file "$src" "$dest"
  fi
}

copy_oneapi_name() {
  local src
  src="$(find_oneapi_path "$1")"
  if [ -z "$src" ]; then
    echo "Unable to find oneAPI runtime dependency $1 in the build image." >&2
    exit 1
  fi
  copy_container_file "$src" "$1"
}

# List SONAMEs that ldd cannot resolve against the bundled dirs alone (libs/ + package
# dir), i.e. the dependencies still needed to make the tarball self-contained.
missing_libraries() {
  local file
  while IFS= read -r -d "" file; do
    LD_LIBRARY_PATH="$PWD/$libs_dir:$PWD/$package_dir" ldd "$file" 2>/dev/null || true
  done < <(
    find "$package_dir" "$libs_dir" -maxdepth 1 -type f \
      \( -name "mom-bin" -o -name "mom.node" -o -name "*.so" -o -name "*.so.*" \) \
      -print0
  ) | awk '/not found/{print $1}' | sort -u
}

# Repeatedly pull every still-unresolved dependency out of the oneAPI tree until ldd
# reports nothing missing, picking up each newly copied lib's own dependencies.
copy_missing_closure() {
  local copied_any=1
  while [ "$copied_any" -eq 1 ]; do
    copied_any=0
    while IFS= read -r lib; do
      [ -n "$lib" ] || continue
      if [ ! -f "$package_dir/$lib" ]; then
        copy_oneapi_name "$lib"
        copied_any=1
      fi
    done < <(missing_libraries)
  done
}

# Copy SYCL runtime pieces that are dlopen()'d by name (so they never appear in ldd
# output) plus the OpenCL device-builtin .rtl/.o blobs the JIT loads at runtime.
copy_optional_sycl_runtime() {
  local paths
  paths="$(docker exec "$container" bash -lc '
    find /opt/intel/oneapi/compiler/latest/lib /opt/intel/oneapi/umf /opt/intel/oneapi/tbb /opt/intel/oneapi/tcm \
      -type f,l \( \
        -name "libur_adapter*.so*" -o \
        -name "libOpenCL.so*" -o \
        -name "libintelocl.so*" -o \
        -name "libocl_svml_*.so" -o \
        -name "libtbbmalloc.so*" -o \
        -name "libtcm.so*" -o \
        -name "libhwloc.so*" -o \
        -name "libiomp5.so" -o \
        -name "clbltfn*.rtl" -o \
        -name "cllibrary*.rtl" -o \
        -name "cllibrary*.o" \
      \) ! -name "*-gdb.py" -print 2>/dev/null | sort
  ')"
  while IFS= read -r src; do
    [ -n "$src" ] || continue
    copy_container_runtime_path "$src"
  done <<<"$paths"
}

# Resolve the addon's closure first, then add the dlopen'd SYCL libs, then close again so
# those libs' own dependencies get pulled in too.
copy_missing_closure
copy_optional_sycl_runtime
copy_missing_closure

unresolved="$(missing_libraries)"
if [ -n "$unresolved" ]; then
  echo "Unresolved packaged Linux dependencies:" >&2
  echo "$unresolved" >&2
  exit 1
fi

if [ -d "$package_dir/tests" ]; then
  echo "Release package unexpectedly contains tests/." >&2
  exit 1
fi

tar -C release -czf "$archive" "$root"
echo "$archive"
