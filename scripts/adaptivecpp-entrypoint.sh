#!/usr/bin/env bash
set -euo pipefail

host_root=$PWD
build_jobs="${MOM_BUILD_JOBS:-$(nproc)}"
compiler="$host_root/scripts/cxx-adaptivecpp.sh"
host_cc=${MOM_ADAPTIVE_HOST_CC:-clang-21}
host_cxx=${MOM_ADAPTIVE_HOST_CXX:-clang++-21}
backend=${MOM_ADAPTIVE_BACKEND:-hip}
case "$backend" in
  hip|cuda) ;;
  *) echo "Unsupported MOM_ADAPTIVE_BACKEND=$backend" >&2; exit 2 ;;
esac
build_dir=${MOM_ADAPTIVE_BUILD_DIR:-build/cache/acpp-$backend}
if [[ "$build_dir" = /* ]]; then
  selected_build_path="$build_dir"
else
  selected_build_path="$host_root/$build_dir"
fi
cache_dir="$(realpath -m -- "$selected_build_path")"
# multicompiler-entrypoint parks the repository's build/ tree while each worker is built. Treat
# that parked tree as the active build root so its caller-selected cache remains valid, but never
# permit a cache path outside the resolved build tree (or the build root itself).
build_root_path="$host_root/build"
if [ ! -e "$build_root_path" ] && [ -d "$host_root/build-platforms-hold" ]; then
  parked_build_root="$(realpath -m -- "$host_root/build-platforms-hold")"
  case "$cache_dir" in
    "$parked_build_root"/*) build_root_path="$host_root/build-platforms-hold" ;;
  esac
fi
if [ -L "$build_root_path" ]; then
  echo "The repository build root must not be a symbolic link: $build_root_path" >&2
  exit 2
fi
build_root="$(realpath -m -- "$build_root_path")"
case "$cache_dir" in
  "$build_root"/*) ;;
  *) echo "MOM_ADAPTIVE_BUILD_DIR must resolve below the repository build/ directory: $build_dir" >&2; exit 2 ;;
esac
work_root=/tmp/mom-adaptive-build
source_dir="$work_root/source"
MOM_BUILD_LOG="$work_root/build-output.log"
source "$host_root/scripts/build-helpers.sh"
# Include host CPU policy so portable packages cannot inherit cached -march=native objects.
compiler_first_line() {
  "$@" --version 2>&1 | sed -n '1p'
}
host_cc_identity="$(compiler_first_line "$host_cc")"
host_cxx_identity="$(compiler_first_line "$host_cxx")"
acpp_identity="$(compiler_first_line acpp)"
rocm_marker=
if [ "$backend" = hip ]; then
  rocm_root=${MOM_ROCM_ROOT:-${ROCM_PATH:-/opt/rocm}}
  rocm_marker=$'\n'"rocm_root=$rocm_root"
fi
marker="$(printf '%s\n' \
  "node=$(node -p process.version)" \
  "compiler=adaptivecpp-$backend" \
  "targets=${ACPP_TARGETS:-}" \
  "cpu=${MOM_CPU_MARCH:-unset}" \
  "portable=${MOM_PORTABLE_BUILD:-0}" \
  "lto=${MOM_LTO:-auto}" \
  "host_cc=$host_cc" \
  "host_cxx=$host_cxx" \
  "host_cc_id=$host_cc_identity" \
  "host_cxx_id=$host_cxx_identity" \
  "acpp_id=$acpp_identity")$rocm_marker"
if [ "$(cat "$cache_dir/.node-version" 2>/dev/null || true)" != "$marker" ] ||
   [ ! -s "$cache_dir/Release/mom.node" ] ||
   find binding.gyp native sycl xmrig scripts/cpu-cflags.sh scripts/cpu-optflags.sh scripts/cpu-feature.sh scripts/cxx-adaptivecpp.sh -type f \
     ! -name '*.md' \
     -newer "$cache_dir/Release/mom.node" -print -quit | grep -q .; then
  rm -rf "$work_root"
  mkdir -p "$source_dir"
  # Only these tracked inputs participate in the native addon. Copying the entire checkout also
  # copied persistent Linux/Windows compiler caches after they moved under build/, wasting hundreds
  # of megabytes of tmpfs and making an otherwise incremental AdaptiveCpp build needlessly fragile.
  cp -a binding.gyp native scripts sycl xmrig "$source_dir/"
  cd "$source_dir"

  preserve_partial_build() {
    local rc=$?
    trap - EXIT
    set +e
    cd "$host_root"
    if [ -d "$source_dir/build" ]; then
      [ ! -f "$MOM_BUILD_LOG" ] || cp "$MOM_BUILD_LOG" "$source_dir/build/build-output.log"
      rm -rf "$cache_dir"
      mkdir -p "$(dirname "$cache_dir")"
      mv "$source_dir/build" "$cache_dir"
      chown -R --reference="$host_root" "$cache_dir"
    fi
    rm -rf "$work_root"
    exit "$rc"
  }
  trap preserve_partial_build EXIT

  if [ "$(cat "$cache_dir/.node-version" 2>/dev/null || true)" = "$marker" ] &&
     [ -s "$cache_dir/Makefile" ] && [ ! binding.gyp -nt "$cache_dir/Makefile" ] &&
     [ ! scripts/cpu-cflags.sh -nt "$cache_dir/Makefile" ] &&
     [ ! scripts/cpu-optflags.sh -nt "$cache_dir/Makefile" ] &&
     [ ! scripts/cpu-feature.sh -nt "$cache_dir/Makefile" ] &&
     [ ! scripts/cxx-adaptivecpp.sh -nt "$cache_dir/Makefile" ]; then
    mv "$cache_dir" build
  else
    rm -rf "$cache_dir"
  fi
  # node-gyp records the compiler's absolute path. Reconfigure a relocated cache in place so its
  # valid Release objects survive a different constrained-container mount point.
  if [ ! -s build/Makefile ] ||
     ! grep -Fqx "CXX.target ?= $compiler" build/Makefile; then
    mom_run_quiet "[adaptivecpp-$backend] node-gyp configure" env \
      CC="$host_cc" CXX="$compiler" LINK="$compiler" node-gyp configure --nodedir=/usr/local \
      -- -Dmom_sycl_impl="adaptivecpp-$backend"
  fi

  printf '%s\n' "$marker" >build/.node-version
  mom_run_quiet "[adaptivecpp-$backend] node-gyp build" env JOBS="$build_jobs" \
    CC="$host_cc" CXX="$compiler" LINK="$compiler" \
    node-gyp build --nodedir=/usr/local --jobs "$build_jobs"
  [ ! -f "$MOM_BUILD_LOG" ] || cp "$MOM_BUILD_LOG" build/build-output.log
  cd "$host_root"
  rm -rf "$cache_dir"
  mkdir -p "$(dirname "$cache_dir")"
  mv "$source_dir/build" "$cache_dir"
  chown -R --reference="$host_root" "$cache_dir"
  trap - EXIT
  rm -rf "$work_root"
fi
export MOM_NATIVE_PATH="$cache_dir/Release/mom.node"
exec "$@"
