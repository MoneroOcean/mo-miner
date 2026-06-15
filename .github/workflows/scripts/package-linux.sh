#!/usr/bin/env bash
set -euo pipefail

version="${1:-}"
if [ -z "$version" ] && [ "${GITHUB_REF_NAME:-}" != "" ] && [[ "${GITHUB_REF_NAME}" =~ ^v?[0-9] ]]; then
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
  --bundle \
  --platform=node \
  --format=cjs \
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
cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

find_oneapi_path() {
  local name="$1"
  local quoted
  quoted="$(printf "%q" "$name")"
  docker exec "$container" bash -lc \
    "find /opt/intel/oneapi -type f,l -name $quoted ! -name '*-gdb.py' -print -quit"
}

copy_container_file() {
  local source_path="$1"
  local dest_name="${2:-$(basename "$source_path")}"
  if [ -z "$source_path" ] || [ -f "$libs_dir/$dest_name" ]; then
    return
  fi
  docker cp -L "$container:$source_path" "$libs_dir/$dest_name"
}

copy_container_runtime_path() {
  local source_path="$1"
  local dest_name resolved target_name quoted
  dest_name="$(basename "$source_path")"
  if [ -z "$source_path" ] || [ -e "$libs_dir/$dest_name" ] || [ -L "$libs_dir/$dest_name" ]; then
    return
  fi

  if docker exec "$container" test -L "$source_path"; then
    quoted="$(printf "%q" "$source_path")"
    resolved="$(docker exec "$container" bash -lc "readlink -f $quoted")"
    target_name="$(basename "$resolved")"
    copy_container_file "$resolved" "$target_name"
    if [ "$dest_name" != "$target_name" ]; then
      ln -s "./$target_name" "$libs_dir/$dest_name"
    fi
  else
    copy_container_file "$source_path" "$dest_name"
  fi
}

copy_oneapi_name() {
  local name="$1"
  local source_path
  source_path="$(find_oneapi_path "$name")"
  if [ -z "$source_path" ]; then
    echo "Unable to find oneAPI runtime dependency $name in the build image." >&2
    exit 1
  fi
  copy_container_file "$source_path" "$name"
}

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
  while IFS= read -r source_path; do
    [ -n "$source_path" ] || continue
    copy_container_runtime_path "$source_path"
  done <<<"$paths"
}

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
