#!/usr/bin/env bash
# Dual-compiler wrapper for the Option C combined build (one mom.node for Intel + NVIDIA).
# node-gyp drives the whole build through a single $CXX/$LINK; this wrapper inspects each
# invocation and routes it to the right toolchain:
#
#   * final link (-shared / -o *.node)  -> nightly clang -fsycl  (only clang embeds the
#                                          spir64 + nvptx device images; also pulls in the
#                                          oneAPI compiler runtime so icx-built objects
#                                          resolve libsvml/libirc/libimf symbols)
#   * a sycl/*.cpp SYCL translation unit -> nightly clang        (binding.gyp already put
#                                          -fsycl on the command for the `sycl` target)
#   * everything else (CPU/host objects) -> oneAPI icpx          (keeps the icx-only
#                                          RandomX codegen advantage)
#
# CC stays icx directly (all .c sources are host code). Both compilers are intel/llvm-based
# and ABI-compatible, so cross-linking icx objects with a clang -fsycl link is sound.
set -euo pipefail

DPCPP="${MOM_DPCPP_ROOT:-/opt/dpcpp}"
CLANGXX="$DPCPP/bin/clang++"
ICPX="${MOM_ICPX:-icpx}"
# AOT targets baked into the SYCL device objects and the final link: Intel SPIR-V + NVIDIA PTX.
# Default to ONE low NVIDIA arch (sm_80): a multi-arch nvptx fatbin is mis-selected at runtime by the
# nightly clang (CUDA_ERROR_NO_BINARY), while a single sm_80 image's forward-compatible PTX is JIT'd
# by the driver to the real GPU at load -- one build runs natively on Ampere/Ada/Hopper. Keep in sync
# with scripts/combined-build.sh's default. Must match between the sycl-TU compile and the link.
TARGETS="${MOM_COMBINED_TARGETS:-spir64,nvidia_gpu_sm_80}"

is_link=0 is_compile=0 src=""
for a in "$@"; do
  case "$a" in
    -shared)               is_link=1 ;;
    *.node)                is_link=1 ;;
    -c)                    is_compile=1 ;;
    *.cpp|*.cc|*.cxx|*.C)  src="$a" ;;
  esac
done

log() { [ -n "${MOM_COMBINED_LOG:-}" ] && echo "$1" >>"$MOM_COMBINED_LOG" || true; }

if [ "$is_link" = 1 ]; then
  log "LINK   -> clang -fsycl"
  # binding.gyp passes -Wl,--disable-new-dtags, so every -rpath becomes DT_RPATH (outranks
  # LD_LIBRARY_PATH, searched in link order). The nightly lib dir MUST come first: the binary
  # is linked against the nightly libsycl.so.9, whose SONAME collides with oneAPI's, and only
  # the nightly one carries the CUDA/Level-Zero UR adapters. The oneAPI compiler dir comes
  # after, only to resolve the libsvml/libimf/libintlc symbols the icx host objects reference.
  rpaths=("-Wl,-rpath,$DPCPP/lib")
  intel_libs=()
  if [ -n "${CMPLR_ROOT:-}" ] && [ -d "$CMPLR_ROOT/lib" ]; then
    rpaths+=("-L$CMPLR_ROOT/lib" "-Wl,-rpath,$CMPLR_ROOT/lib")
    # icx lowers memcpy/memset etc. to libirc helpers (_intel_fast_memcpy), and emits
    # libsvml/libimf/libintlc math calls. clang won't auto-pull these, so link them
    # explicitly AFTER the object group ("$@" ends with gyp's object group + ldflags) so
    # the referenced symbols are resolved. Prefer the STATIC Intel runtime (.a) for the libs
    # that ship one (imf/svml/irng/irc) so mom.node carries no libimf/libsvml/libirng/libirc.so
    # deps to bundle; libintlc ships only a .so, so it stays dynamic. libstdc++/libgcc stay
    # dynamic too -- they share C++ exceptions/RTTI across the boundary with the bundled
    # libsycl.so, where two static copies would break unwinding.
    intel_libs=(-Wl,-Bstatic -limf -lsvml -lirng -lirc -Wl,-Bdynamic -lintlc)
  fi
  exec "$CLANGXX" -fsycl -fsycl-targets="$TARGETS" "${rpaths[@]}" "$@" "${intel_libs[@]}"
fi

# node-gyp runs make from build/, so the SYCL sources arrive as ../sycl/foo.cpp (or an
# absolute path); match any path component "sycl/" so they route to clang, not just a
# leading "sycl/".
if [ "$is_compile" = 1 ] && [[ "$src" == sycl/*.cpp || "$src" == */sycl/*.cpp ]]; then
  # pearl_esimd.cpp is the Intel ESIMD search, which can't share -fsycl-targets with nvptx, so build
  # it spir64-only; everything else gets the full spir64+nvptx target set.
  if [[ "$src" == *pearl_esimd.cpp ]]; then
    log "SYCL   -> clang   : $src (spir64-only ESIMD)"
    exec "$CLANGXX" -fsycl-targets=spir64 "$@"
  fi
  log "SYCL   -> clang   : $src"
  exec "$CLANGXX" -fsycl-targets="$TARGETS" "$@"
fi

[ "$is_compile" = 1 ] && log "HOST   -> icpx    : $src"
exec "$ICPX" "$@"
