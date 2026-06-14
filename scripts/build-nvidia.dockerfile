# NVIDIA build image: AdaptiveCpp (acpp) generic-SSCP JIT toolchain.
# Mirrors scripts/build.dockerfile (Intel oneAPI) but compiles the SYCL sources
# with AdaptiveCpp for NVIDIA. The generic SSCP target needs no GPU at build
# time (only the CUDA toolkit), so this builds on a GPU-less CI runner and JITs
# to PTX at runtime; it also exposes an OpenMP host (SYCL CPU) device used for
# the cpu / sycl-cpu verification suites.
# https://hub.docker.com/r/nvidia/cuda/tags
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-c"]

ARG NODE_VERSION=24.15.0
ARG LLVM_VERSION=19
ARG ACPP_REF=v25.10.0

# Base tools + LLVM/Clang (AdaptiveCpp host compiler) + AdaptiveCpp build deps.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg iputils-ping make cmake ninja-build git python3 sudo xz-utils \
      lsb-release software-properties-common pkg-config \
      libboost-context-dev libboost-fiber-dev libnuma-dev zlib1g-dev libffi-dev && \
    curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh && \
    bash /tmp/llvm.sh ${LLVM_VERSION} all && rm /tmp/llvm.sh && \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o /tmp/node.tar.xz && \
    tar -C /usr/local --strip-components=1 -xf /tmp/node.tar.xz && rm /tmp/node.tar.xz && \
    npm install -g node-gyp@12.2.0 && \
    node --version && npm --version && node-gyp --version && clang-${LLVM_VERSION} --version | head -1

# Build + install AdaptiveCpp (CUDA backend + full feature profile = generic JIT).
RUN git clone --branch ${ACPP_REF} --depth 1 https://github.com/AdaptiveCpp/AdaptiveCpp /tmp/acpp && \
    cmake -S /tmp/acpp -B /tmp/acpp/build -G Ninja \
      -DCMAKE_INSTALL_PREFIX=/opt/adaptivecpp \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm \
      -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-${LLVM_VERSION} \
      -DWITH_CUDA_BACKEND=ON \
      -DACPP_COMPILER_FEATURE_PROFILE=full \
      -DCUDAToolkit_ROOT=/usr/local/cuda && \
    cmake --build /tmp/acpp/build -j"$(nproc)" --target install && \
    rm -rf /tmp/acpp

ENV PATH=/opt/adaptivecpp/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/adaptivecpp/lib:/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH
# Referenced (unexpanded) inside the $'...' entrypoint and expanded at runtime,
# so it must not break the ANSI-C quoting the way an inline ${LLVM_VERSION} would.
ENV MOMINER_CC=clang-${LLVM_VERSION}

# allow root group access to /root
RUN chmod g=u /root
# replace dash by bash as default /bin/sh shell
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# setup env to do build under user that owns /root/mo-miner on the host
# runs miner under root anyway (no way to access /dev/cpu/*/msr otherwise)
RUN echo $'#!/usr/bin/env bash\n\
(userdel -r "$(getent passwd $(stat -c "%u" /root/mo-miner) | cut -d: -f1)" 2>/dev/null || true)\n\
useradd user -u $(stat -c "%g" /root/mo-miner) -G root,video -m -s /bin/bash;\n\
echo "user ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/user-user\n\
# Release CI builds use a portable baseline because GitHub can build and test on different x86-64 CPU models.\n\
portable_build="${MOMINER_PORTABLE_BUILD:-0}"\n\
lto="${MOMINER_LTO:-0}"\n\
perf_samples="${MOMINER_PERF_SAMPLES:-}"\n\
su - user <<EOF\n\
cd /root/mo-miner # su - resets to home dir and we need to keep /root/mo-miner pwd\n\
export PATH=/opt/adaptivecpp/bin:$PATH\n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH\n\
export ACPP_TARGETS=generic\n\
export MOMINER_SYCL_IMPL=acpp\n\
export MOMINER_PORTABLE_BUILD="$portable_build"\n\
export MOMINER_LTO="$lto"\n\
export MOMINER_PERF_SAMPLES="$perf_samples"\n\
{ ping -c1 -W2 8.8.8.8 >/dev/null 2>&1; } && npm update --silent || echo "Skip npm update since there is no internet access"\n\
node_build_version="\$(node -p "process.version"):/usr/local:portable=$portable_build:ax=1:lto=$lto:impl=acpp"\n\
if [ ! -s ./build/Release/mo-miner.node ] || [ "\$(cat ./build/.node-version 2>/dev/null || true)" != "\$node_build_version" ] || ! grep -q "/root/mo-miner" ./build/Makefile 2>/dev/null; then\n\
  rm -rf ./build\n\
  CC=$MOMINER_CC CXX=acpp node-gyp configure --nodedir=/usr/local -- -Dmominer_sycl_impl=acpp\n\
fi &&\n\
JOBS=$(nproc) CC=$MOMINER_CC CXX=acpp MAKEFLAGS=-s node-gyp build --nodedir=/usr/local --silent &&\n\
mkdir -p ./build && echo "\$node_build_version" > ./build/.node-version &&\n\
({ test $# -eq 1; } && { echo "One param mode"; sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH MOMINER_PERF_SAMPLES="\$MOMINER_PERF_SAMPLES" -- /bin/bash -c ${*@Q}; } || sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH MOMINER_PERF_SAMPLES="\$MOMINER_PERF_SAMPLES" -- ${*@Q})\n\
EOF' >/root/entrypoint.sh &&\
chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]

# sync user with host, build and run application
WORKDIR /root/mo-miner
CMD node --require ./tests/common/test_output_buffer.js --test --test-reporter=./tests/common/spec_reporter.js --test-concurrency=1 tests/all.js
