#!/usr/bin/env bash
# Build/run mom with the DPC++ CUDA backend (NVIDIA), mirroring r.sh but using
# scripts/build-nvidia.dockerfile and mom_sycl_impl=dpcpp-cuda. The Intel r.sh is
# left untouched. AOT-compiles to PTX/SASS, so the image needs no GPU at build time;
# if a host NVIDIA driver is present the container is given GPU access (requires
# nvidia-container-toolkit) so GPU runs/tests work too.
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! docker buildx version >/dev/null 2>&1; then
  echo "Docker buildx is required. Install docker-buildx-plugin or see README.md." >&2
  exit 1
fi

build_iid="$(mktemp)"
trap 'rm -f "$build_iid"' EXIT
docker buildx build --load --progress=none --iidfile "$build_iid" -t mom-build-nvidia --pull=false - <"$SCRIPT_DIR/scripts/build-nvidia.dockerfile"
docker rm -f mom-nvidia >/dev/null 2>&1 || true

docker_flags=(
  --privileged
  --rm
  --name mom-nvidia
  --hostname mom-nvidia
  --env MOM_R_SH=1
  --env MOM_SYCL_IMPL=dpcpp-cuda
  --mount "type=bind,source=$SCRIPT_DIR,target=/root/mom"
)

# Expose the GPU when a host driver is available. The dpcpp-cuda build is CUDA-only
# (no CPU SYCL device), so GPU algos run only with a driver present; CI runners have
# none and only build + package + run the pure-CPU (xmrig) suite.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  docker_flags+=(--gpus all)
fi

if [ -n "${MOM_PORTABLE_BUILD:-}" ]; then
  docker_flags+=(--env MOM_PORTABLE_BUILD)
fi

if [ -n "${MOM_LTO:-}" ]; then
  docker_flags+=(--env MOM_LTO)
fi

if [ -n "${MOM_PERF_SAMPLES:-}" ]; then
  docker_flags+=(--env MOM_PERF_SAMPLES)
fi

if [ -t 0 ] && [ -t 1 ]; then
  docker_flags+=(-it)
else
  docker_flags+=(-i)
fi

if [ $# -ne 0 ]; then
  docker run "${docker_flags[@]}" mom-build-nvidia "$@"
else
  docker run "${docker_flags[@]}" mom-build-nvidia
fi
