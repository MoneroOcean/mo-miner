#!/usr/bin/env bash
# Build/run mo-miner with the DPC++ CUDA backend (NVIDIA), mirroring r.sh but using
# scripts/build-nvidia.dockerfile and mominer_sycl_impl=dpcpp-cuda. The Intel r.sh is
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
docker buildx build --load --progress=none --iidfile "$build_iid" -t mo-miner-build-nvidia --pull=false - <"$SCRIPT_DIR/scripts/build-nvidia.dockerfile"
docker rm -f mo-miner-nvidia >/dev/null 2>&1 || true

docker_flags=(
  --privileged
  --rm
  --name mo-miner-nvidia
  --hostname mo-miner-nvidia
  --env MOMINER_R_SH=1
  --env MOMINER_SYCL_IMPL=dpcpp-cuda
  --mount "type=bind,source=$SCRIPT_DIR,target=/root/mo-miner"
)

# Expose the GPU when a host driver is available. The dpcpp-cuda build is CUDA-only
# (no CPU SYCL device), so GPU algos run only with a driver present; CI runners have
# none and only build + package + run the pure-CPU (xmrig) suite.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  docker_flags+=(--gpus all)
fi

if [ -n "${MOMINER_PORTABLE_BUILD:-}" ]; then
  docker_flags+=(--env MOMINER_PORTABLE_BUILD)
fi

if [ -n "${MOMINER_LTO:-}" ]; then
  docker_flags+=(--env MOMINER_LTO)
fi

if [ -n "${MOMINER_PERF_SAMPLES:-}" ]; then
  docker_flags+=(--env MOMINER_PERF_SAMPLES)
fi

if [ -t 0 ] && [ -t 1 ]; then
  docker_flags+=(-it)
else
  docker_flags+=(-i)
fi

if [ $# -ne 0 ]; then
  docker run "${docker_flags[@]}" mo-miner-build-nvidia "$@"
else
  docker run "${docker_flags[@]}" mo-miner-build-nvidia
fi
