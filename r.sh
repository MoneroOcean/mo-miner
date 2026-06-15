#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! docker buildx version >/dev/null 2>&1; then
  echo "Docker buildx is required. Install docker-buildx-plugin or see README.md." >&2
  exit 1
fi

docker buildx build --load --progress=none -t mom-build --pull=false - <"$SCRIPT_DIR/scripts/build.dockerfile"
docker rm -f mom >/dev/null 2>&1 || true

docker_flags=(
  --privileged
  --rm
  --name mom
  --hostname mom
  --env MOM_R_SH=1
  --mount "type=bind,source=$SCRIPT_DIR,target=/root/mom"
)

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

docker run "${docker_flags[@]}" mom-build "$@"
