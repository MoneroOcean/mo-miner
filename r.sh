#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! docker buildx version >/dev/null 2>&1; then
  echo "Docker buildx is required. Install docker-buildx-plugin or see README.md." >&2
  exit 1
fi

build_iid="$(mktemp)"
trap 'rm -f "$build_iid"' EXIT
docker buildx build --load --progress=none --iidfile "$build_iid" -t mo-miner-build --pull=false - <"$SCRIPT_DIR/scripts/build.dockerfile"
docker rm -f mo-miner >/dev/null 2>&1 || true

docker_flags=(
  --privileged
  --rm
  --name mo-miner
  --hostname mo-miner
  --env MOMINER_R_SH=1
  --mount "type=bind,source=$SCRIPT_DIR,target=/root/mo-miner"
)

if [ -n "${MOMINER_PORTABLE_BUILD:-}" ]; then
  docker_flags+=(--env MOMINER_PORTABLE_BUILD)
fi

if [ -n "${MOMINER_LTO:-}" ]; then
  docker_flags+=(--env MOMINER_LTO)
fi

if [ -t 0 ] && [ -t 1 ]; then
  docker_flags+=(-it)
else
  docker_flags+=(-i)
fi

if [ $# -ne 0 ]; then
  docker run "${docker_flags[@]}" mo-miner-build "$@"
else
  docker run "${docker_flags[@]}" mo-miner-build
fi
