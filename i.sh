#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
docker build -q -t mom-build --pull=false - <$SCRIPT_DIR/scripts/build.dockerfile &&\
docker run --privileged --rm --name mom --hostname mom\
           --mount type=bind,source=$SCRIPT_DIR,target=/root/mom\
           -it --entrypoint "" mom-build bash
