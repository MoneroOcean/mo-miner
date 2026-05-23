#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
docker build -q -t mo-miner-build --pull=false - <$SCRIPT_DIR/build.dockerfile &&\
docker run --privileged --rm --name mo-miner --hostname mo-miner\
           --mount type=bind,source=$SCRIPT_DIR,target=/root/mo-miner\
           -it --entrypoint "" mo-miner-build bash
