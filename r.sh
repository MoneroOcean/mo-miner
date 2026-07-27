#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

backend_was_explicit=0
[ -z "${MOM_GPU_BACKEND+x}" ] || backend_was_explicit=1
backend=${MOM_GPU_BACKEND:-intel}
case "$backend" in intel|nvidia|amd|opencl|all) ;; *) echo "Unknown MOM_GPU_BACKEND: $backend" >&2; exit 2 ;; esac
image=mom-build-multicompiler
name=mom

if ! docker buildx version >/dev/null 2>&1; then
  echo "Docker buildx is required. Install docker-buildx-plugin or see README.md." >&2
  exit 1
fi

# BuildKit component stages are cached; only the final multicompiler image is run. CI points
# MOM_BUILDX_CACHE_DIR at an actions/cache directory. Use independent scopes so an updated compiler
# does not invalidate the others, and replace an imported cache only after its image built cleanly.
build_image() {
  local tag=$1 dockerfile=$2 scope=$3
  local -a cache_args=() build_args=() builder_args=()
  local cache_root=${MOM_BUILDX_CACHE_DIR:-}
  local enabled_scopes=",${MOM_BUILDX_CACHE_SCOPES:-combined,acpp-cuda,acpp-hip,multicompiler},"
  if [ -n "$cache_root" ] && [[ "$enabled_scopes" == *",$scope,"* ]]; then
    local source_cache="$cache_root/$scope" next_cache="$cache_root/.next-$scope"
    mkdir -p "$cache_root"
    rm -rf "$next_cache"
    [ ! -f "$source_cache/index.json" ] || cache_args+=(--cache-from "type=local,src=$source_cache")
    if [ "${MOM_BUILDX_CACHE_READ_ONLY:-0}" != 1 ]; then
      # The payload-only component images are sufficient to skip an unchanged compiler rebuild.
      # Caching every LLVM object or duplicate CUDA/ROCm SDK (mode=max) would exceed GitHub's cache
      # quota before both source toolchains fit.
      cache_args+=(--cache-to "type=local,dest=$next_cache,mode=min")
    fi
  fi
  case "$scope" in
    amd-dpcpp|acpp-cuda|acpp-hip)
      [ -z "${MOM_BUILD_JOBS:-}" ] || build_args+=(--build-arg "MOM_BUILD_JOBS=$MOM_BUILD_JOBS")
      ;;
  esac
  # CI's setup-buildx action selects a docker-container builder. That is useful while importing the
  # source-toolchain caches, but building and --load'ing the 25+ GiB combined/final images there
  # stores their layers once inside BuildKit and again in Docker. Use Docker's default BuildKit
  # store for the uncached final stages after the source images have been materialized.
  if [ "${MOM_BUILDX_CACHE_READ_ONLY:-0}" = 1 ] && [ "${MOM_BUILDX_CACHE_DELETE_AFTER_LOAD:-0}" = 1 ]; then
    case "$scope" in combined|multicompiler) builder_args+=(--builder default) ;; esac
  fi
  docker buildx build "${builder_args[@]}" --load --progress=none -t "$tag" --pull=false \
    "${cache_args[@]}" "${build_args[@]}" -f "$SCRIPT_DIR/$dockerfile" "$SCRIPT_DIR"
  if [ -n "${next_cache:-}" ] && [ -f "$next_cache/index.json" ]; then
    rm -rf "$source_cache"
    mv "$next_cache" "$source_cache"
  fi
  # Final CI assembly imports these caches only to materialize the component images. Once --load
  # succeeds, the restored blob directory can be removed before the large unified image is built;
  # the loaded Docker image is independent of that directory.
  if [ "${MOM_BUILDX_CACHE_DELETE_AFTER_LOAD:-0}" = 1 ] && [ -n "${source_cache:-}" ]; then
    rm -rf "$source_cache"
  fi
}

case "${MOM_BUILD_ONLY_SCOPE:-}" in
  amd-dpcpp) build_image mom-build-amd scripts/build-amd.dockerfile amd-dpcpp; exit ;;
  acpp-cuda)
    build_image mom-build-nvidia-adaptivecpp scripts/build-nvidia-adaptivecpp.dockerfile acpp-cuda
    exit ;;
  acpp-hip)
    build_image mom-build-amd-acpp-baseline scripts/build-amd-adaptivecpp.dockerfile acpp-hip
    exit ;;
  '') ;;
  *) echo "Unknown MOM_BUILD_ONLY_SCOPE: $MOM_BUILD_ONLY_SCOPE" >&2; exit 2 ;;
esac

# A normal source/test iteration uses the existing pinned development image and rebuilds only miner
# objects through the bind mount. Reconfiguring LLVM merely because MOM_BUILD_JOBS or an unrelated
# repository file changed wastes tens of minutes. CI component jobs still take the explicit
# MOM_BUILD_ONLY_SCOPE path above; set MOM_REBUILD_DEV_IMAGE=1 after intentionally changing a
# compiler/runtime Docker stage.
if [ "${MOM_REBUILD_DEV_IMAGE:-0}" != 1 ] && [ -z "${MOM_BUILDX_CACHE_DIR:-}" ] && \
   docker image inspect "$image" >/dev/null 2>&1; then
  : # Reuse the installed compiler runtimes.
else
  # Materialize the cached source toolchains first. In CI this also lets build_image remove each
  # restored cache directory before the much larger oneAPI/CUDA combined image is downloaded.
  build_image mom-build-nvidia-adaptivecpp scripts/build-nvidia-adaptivecpp.dockerfile acpp-cuda
  build_image mom-build-amd-acpp-baseline scripts/build-amd-adaptivecpp.dockerfile acpp-hip
  # The source images are now in Docker and no longer depend on the selected BuildKit worker. On
  # hosted package jobs, discard its imported cache before assembling the two much larger images.
  if [ "${MOM_BUILDX_CACHE_READ_ONLY:-0}" = 1 ] && [ "${MOM_BUILDX_CACHE_DELETE_AFTER_LOAD:-0}" = 1 ]; then
    docker buildx prune --all --force >/dev/null
  fi
  build_image mom-build-combined scripts/build-combined.dockerfile combined
  build_image "$image" scripts/build-multicompiler.dockerfile multicompiler
fi
docker rm -f "$name" >/dev/null 2>&1 || true

docker_flags=(
  --privileged
  --rm
  --name "$name"
  --hostname "$name"
  --env MOM_R_SH=1
  --env "MOM_GPU_BACKEND=$backend"
  --mount "type=bind,source=$SCRIPT_DIR,target=/root/mom"
  # DPC++'s device binaries are keyed by image, driver, and device. Keep that cache across the
  # disposable development containers so large OpenCL kernels are not recompiled on every r.sh run.
  --mount "type=volume,source=mom-sycl-cache,target=/root/.cache/libsycl_cache"
)
if [ -n "${MOM_GPU_TEST_VENDORS:-}" ]; then
  docker_flags+=(--env "MOM_GPU_TEST_VENDORS=$MOM_GPU_TEST_VENDORS")
elif [ "$backend_was_explicit" = 1 ]; then
  # An explicitly selected backend narrows correctness tests; the default development container
  # discovers and tests every supported discrete-GPU vendor visible on the host.
  if [ "$backend" = all ]; then
    docker_flags+=(--env "MOM_GPU_TEST_VENDORS=intel,nvidia,amd")
  else
    docker_flags+=(--env "MOM_GPU_TEST_VENDORS=$backend")
  fi
fi

# The NVIDIA nvptx images are AOT-built (no GPU needed to build); expose the GPU only when a host
# driver is present so GPU runs/tests work (needs nvidia-container-toolkit). CI runners have none
# and just build + package + run the CPU and SYCL-CPU suites. An Intel GPU is reached via --privileged
# (/dev/dri). The build picks dpcpp-combined itself in scripts/combined-build.sh, so no MOM_SYCL_IMPL.
nvidia_gpu_available() {
  nvidia-smi -L >/dev/null 2>&1 && return 0
  command -v nvidia-container-cli >/dev/null 2>&1 || return 1
  [ -e /dev/nvidiactl ] || return 1
  find /sys/bus/pci/drivers/nvidia -maxdepth 1 -type l -name '0000:*' -print -quit 2>/dev/null | grep -q .
}

command_env_value() {
  local name="$1"; shift
  [ "${1:-}" = "env" ] || return 1
  shift
  local arg
  for arg in "$@"; do
    case "$arg" in
      --) continue ;;
      "$name"=*)
        printf '%s\n' "${arg#*=}"
        return 0
        ;;
      *=*) ;;
      *) return 1 ;;
    esac
  done
  return 1
}

selector_requests_cuda() {
  local selector="${ONEAPI_DEVICE_SELECTOR:-}"
  selector="${selector:-$(command_env_value ONEAPI_DEVICE_SELECTOR "$@" || true)}"
  case "$selector" in
    *cuda*) return 0 ;;
    *level_zero*|*opencl*) return 1 ;;
    *) return 2 ;;
  esac
}

case "${MOM_DOCKER_GPUS:-auto}" in
  0) ;;
  1|all) docker_flags+=(--gpus all) ;;
  auto)
    if [ "$backend" = opencl ]; then
      # Generic OpenCL may still be provided by NVIDIA's driver. The container runtime mounts its
      # ICD/user-space libraries only with --gpus, while --privileged already covers DRM GPUs.
      nvidia_gpu_available && docker_flags+=(--gpus all)
    elif [ "$backend" != amd ]; then
      if selector_requests_cuda "$@"; then
        nvidia_gpu_available && docker_flags+=(--gpus all)
      elif [ $? -eq 2 ] && nvidia_gpu_available; then
        docker_flags+=(--gpus all)
      fi
    fi
    ;;
  *)
    docker_flags+=(--gpus "${MOM_DOCKER_GPUS}")
    ;;
esac

# Forward these build-tuning env vars into the container only when set. MOM_COMBINED_TARGETS lets a
# build widen/narrow its AOT arch set; MOM_FORCE_REBUILD forces a clean reconfigure.
for var in \
  MOM_PORTABLE_BUILD MOM_LTO MOM_PERF_SAMPLES MOM_COMBINED_TARGETS MOM_FORCE_REBUILD \
  MOM_BUILD_VERBOSE MOM_BUILD_JOBS MOM_GPU_INDEX MOM_OPENCL_DEVICE_TYPE MOM_CN_GPU_INTENSITY \
  MOM_AUTOLYKOS2_WORKGROUP MOM_AUTOLYKOS2_SPLIT MOM_AUTOLYKOS2_PROFILE MOM_ZELHASH_SLOTS \
  ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS MOM_AMD_TARGET
do
  if [ -n "${!var:-}" ]; then docker_flags+=(--env "$var"); fi
done

# Allocate a TTY only when both stdin and stdout are terminals.
if [ -t 0 ] && [ -t 1 ]; then
  docker_flags+=(-it)
else
  docker_flags+=(-i)
fi

docker run "${docker_flags[@]}" "$image" "$@"
