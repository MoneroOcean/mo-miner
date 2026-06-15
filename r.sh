#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Backend select: Intel oneAPI (default) or the DPC++ CUDA backend for NVIDIA. Pass --nvidia
# (or --intel) as the first argument to force one; MOM_NVIDIA=1 also forces NVIDIA. When neither
# is given it auto-selects NVIDIA only if the host has an NVIDIA GPU and no Intel GPU (so a pure
# NVIDIA box "just works" while an Intel box, or a box with both, stays on the Intel build).
nvidia=0; explicit=0
case "${1:-}" in
  --nvidia) nvidia=1; explicit=1; shift ;;
  --intel)  nvidia=0; explicit=1; shift ;;
esac
[ "${MOM_NVIDIA:-}" = 1 ] && { nvidia=1; explicit=1; }

if [ "$explicit" = 0 ] && nvidia-smi -L >/dev/null 2>&1 \
   && ! lspci 2>/dev/null | grep -iE 'vga|3d|display' | grep -qiE 'intel|8086'; then
  nvidia=1
  echo "r.sh: NVIDIA GPU detected and no Intel GPU -> building the DPC++ CUDA backend (use --intel to override)." >&2
fi

if [ "$nvidia" = 1 ]; then
  dockerfile=build-nvidia.dockerfile; image=mom-build-nvidia; name=mom-nvidia
else
  dockerfile=build.dockerfile;        image=mom-build;        name=mom
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "Docker buildx is required. Install docker-buildx-plugin or see README.md." >&2
  exit 1
fi

docker buildx build --load --progress=none -t "$image" --pull=false - <"$SCRIPT_DIR/scripts/$dockerfile"
docker rm -f "$name" >/dev/null 2>&1 || true

docker_flags=(
  --privileged
  --rm
  --name "$name"
  --hostname "$name"
  --env MOM_R_SH=1
  --mount "type=bind,source=$SCRIPT_DIR,target=/root/mom"
)

if [ "$nvidia" = 1 ]; then
  docker_flags+=(--env MOM_SYCL_IMPL=dpcpp-cuda)
  # The CUDA build is AOT (no GPU needed to build); expose the GPU only when a host driver is
  # present so GPU runs/tests work (requires nvidia-container-toolkit). CI runners have none and
  # just build + package + run the pure-CPU suite.
  if nvidia-smi -L >/dev/null 2>&1; then docker_flags+=(--gpus all); fi
fi

# Forward these build-tuning env vars into the container only when set.
for var in MOM_PORTABLE_BUILD MOM_LTO MOM_PERF_SAMPLES; do
  if [ -n "${!var:-}" ]; then docker_flags+=(--env "$var"); fi
done

# Allocate a TTY only when both stdin and stdout are terminals.
if [ -t 0 ] && [ -t 1 ]; then
  docker_flags+=(-it)
else
  docker_flags+=(-i)
fi

docker run "${docker_flags[@]}" "$image" "$@"
