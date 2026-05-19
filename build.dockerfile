# https://hub.docker.com/r/intel/oneapi-toolkit/tags
FROM intel/oneapi-toolkit:2026.0.0-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

ARG NODE_VERSION=24.15.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl iputils-ping make python3 sudo xz-utils && \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o /tmp/node.tar.xz && \
    tar -C /usr/local --strip-components=1 -xf /tmp/node.tar.xz && \
    rm /tmp/node.tar.xz && \
    npm install -g node-gyp@12.2.0 && \
    node --version && npm --version && node-gyp --version

# allow root group access to /root
RUN chmod g=u /root
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# replace dash by bash as default /bin/sh shell
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# setup env to do build under user that owns /root/mominer on the host
# runs miner under root anyway (no way to access /dev/cpu/*/msr otherwise)
RUN echo $'#!/usr/bin/env bash\n\
(userdel -r "$(getent passwd $(stat -c "%u" /root/mominer) | cut -d: -f1)" 2>/dev/null || true)\n\
useradd user -u $(stat -c "%g" /root/mominer) -G root,video -m -s /bin/bash;\n\
echo "user ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/user-user\n\
su - user <<EOF\n\
cd /root/mominer # su - resets to home dir and we need to keep /root/mominer pwd\n\
. /opt/intel/oneapi/setvars.sh >/dev/null\n\
{ ping -c1 -W2 8.8.8.8 >/dev/null 2>&1; } && npm update --silent || echo "Skip npm update since there is no internet access"\n\
node_build_version="\$(node -p "process.version"):/usr/local"\n\
if [ ! -s ./build/Release/mominer.node ] || [ "\$(cat ./build/.node-version 2>/dev/null || true)" != "\$node_build_version" ]; then\n\
  CC=icx CXX=icpx node-gyp configure --nodedir=/usr/local\n\
fi &&\n\
JOBS=$(nproc) CC=icx CXX=icpx MAKEFLAGS=-s node-gyp build --nodedir=/usr/local --silent &&\n\
mkdir -p ./build && echo "\$node_build_version" > ./build/.node-version &&\n\
({ test $# -eq 1; } && { echo "One param mode"; sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH -- /bin/bash -c ${*@Q}; } || sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH -- ${*@Q})\n\
EOF' >/root/entrypoint.sh &&\
chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]

# sync user with host, build and run application
WORKDIR /root/mominer
CMD node --require ./tests/common/test_output_buffer.js --test --test-reporter=./tests/common/spec_reporter.js --test-concurrency=1 tests/all.js
