#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# mom builds ONE Linux binary: a unified mom.node that runs on both Intel and NVIDIA GPUs. It is a
# dual-compiler build -- the CPU/host objects come from oneAPI icx (for the faster RandomX codegen)
# and the SYCL device objects + final -fsycl link from the intel/llvm nightly clang (the only
# toolchain that AOTs both spir64 and nvptx into one binary). See scripts/build-combined.dockerfile
# + scripts/combined-build.sh. This is the only build mode -- there is no per-vendor variant to pick.
dockerfile=build-combined.dockerfile; image=mom-build-combined; name=mom

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

# The NVIDIA nvptx images are AOT-built (no GPU needed to build); expose the GPU only when a host
# driver is present so GPU runs/tests work (needs nvidia-container-toolkit). CI runners have none
# and just build + package + run the CPU and SYCL-CPU suites. An Intel GPU is reached via --privileged
# (/dev/dri). The build picks dpcpp-combined itself in scripts/combined-build.sh, so no MOM_SYCL_IMPL.
if nvidia-smi -L >/dev/null 2>&1; then docker_flags+=(--gpus all); fi

# Forward these build-tuning env vars into the container only when set. MOM_COMBINED_TARGETS lets a
# build widen/narrow its AOT arch set; MOM_FORCE_REBUILD forces a clean reconfigure.
for var in MOM_PORTABLE_BUILD MOM_LTO MOM_PERF_SAMPLES MOM_COMBINED_TARGETS MOM_FORCE_REBUILD MOM_BUILD_VERBOSE MOM_BUILD_JOBS; do
  if [ -n "${!var:-}" ]; then docker_flags+=(--env "$var"); fi
done

# Allocate a TTY only when both stdin and stdout are terminals.
if [ -t 0 ] && [ -t 1 ]; then
  docker_flags+=(-it)
else
  docker_flags+=(-i)
fi

docker run "${docker_flags[@]}" "$image" "$@"
