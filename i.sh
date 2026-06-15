#!/usr/bin/env bash
# Build the image and drop into an interactive shell inside the container.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
docker build -q -t mom-build-combined --pull=false - <"$SCRIPT_DIR/scripts/build-combined.dockerfile" && \
docker run --privileged --rm --name mom --hostname mom \
           --mount "type=bind,source=$SCRIPT_DIR,target=/root/mom" \
           -it --entrypoint "" mom-build-combined bash
