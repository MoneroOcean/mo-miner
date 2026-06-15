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

# Give the root group the same access as the root user to /root, so the
# host-mapped build user (added to group root below) can write into /root/mom.
RUN chmod g=u /root
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# Make /bin/sh bash instead of dash (entrypoint relies on bash features like ${*@Q}).
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Build as the host user that owns /root/mom (so build artifacts stay host-writable),
# but run the miner as root (needed to access /dev/cpu/*/msr).
RUN echo $'#!/usr/bin/env bash\n\
(userdel -r "$(getent passwd $(stat -c "%u" /root/mom) | cut -d: -f1)" 2>/dev/null || true)\n\
useradd user -u $(stat -c "%g" /root/mom) -G root,video -m -s /bin/bash;\n\
echo "user ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/user-user\n\
# Release CI builds use a portable baseline because GitHub can build and test on different x86-64 CPU models.\n\
portable_build="${MOM_PORTABLE_BUILD:-0}"\n\
lto="${MOM_LTO:-auto}"\n\
su - user <<EOF\n\
cd /root/mom # su - resets to home dir and we need to keep /root/mom pwd\n\
. /opt/intel/oneapi/setvars.sh >/dev/null\n\
export MOM_PORTABLE_BUILD="$portable_build"\n\
export MOM_LTO="$lto"\n\
export MOM_PERF_SAMPLES="${MOM_PERF_SAMPLES:-}"\n\
{ ping -c1 -W2 8.8.8.8 >/dev/null 2>&1; } && npm update --silent || echo "Skip npm update since there is no internet access"\n\
node_build_version="\$(node -p "process.version"):/usr/local:portable=$portable_build:ax=1:lto=$lto"\n\
if [ ! -s ./build/Release/mom.node ] || [ "\$(cat ./build/.node-version 2>/dev/null || true)" != "\$node_build_version" ] || ! grep -q "/root/mom" ./build/Makefile 2>/dev/null; then\n\
  rm -rf ./build\n\
  CC=icx CXX=icpx node-gyp configure --nodedir=/usr/local\n\
fi &&\n\
JOBS=$(nproc) CC=icx CXX=icpx MAKEFLAGS=-s node-gyp build --nodedir=/usr/local --silent &&\n\
mkdir -p ./build && echo "\$node_build_version" > ./build/.node-version &&\n\
({ test $# -eq 1; } && { echo "One param mode"; sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH MOM_PERF_SAMPLES="\$MOM_PERF_SAMPLES" -- /bin/bash -c ${*@Q}; } || sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH MOM_PERF_SAMPLES="\$MOM_PERF_SAMPLES" -- ${*@Q})\n\
EOF' >/root/entrypoint.sh &&\
chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]

# Default command (after the entrypoint syncs the user, builds, and elevates): run the test suite.
WORKDIR /root/mom
CMD node --require ./tests/common/test_output_buffer.js --test --test-reporter=./tests/common/spec_reporter.js --test-concurrency=1 tests/all.js
