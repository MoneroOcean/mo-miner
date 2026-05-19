#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! docker buildx version >/dev/null 2>&1; then
  echo "Docker buildx is required. Install docker-buildx-plugin or see README.md." >&2
  exit 1
fi

build_iid="$(mktemp)"
trap 'rm -f "$build_iid"' EXIT
docker buildx build --load --progress=none --iidfile "$build_iid" -t mominer-build --pull=false - <"$SCRIPT_DIR/build.dockerfile"
docker rm -f mominer >/dev/null 2>&1 || true

docker_flags=(
  --privileged
  --rm
  --name mominer
  --hostname mominer
  --env MOMINER_R_SH=1
  --mount "type=bind,source=$SCRIPT_DIR,target=/root/mominer"
)

if [ -t 0 ] && [ -t 1 ]; then
  docker_flags+=(-it)
else
  docker_flags+=(-i)
fi

if [ $# -ne 0 ]; then
  docker run "${docker_flags[@]}" mominer-build "$@"
else
  docker run "${docker_flags[@]}" mominer-build
fi
