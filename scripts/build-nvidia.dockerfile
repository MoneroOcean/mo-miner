# NVIDIA build image: DPC++ CUDA backend (mom_sycl_impl=dpcpp-cuda).
# Mirrors scripts/build.dockerfile (Intel oneAPI) but compiles the SAME SYCL sources
# with the DPC++ CUDA backend instead of AdaptiveCpp. The DPC++ toolchain is a prebuilt
# binary tarball (intel/llvm nightly, which already ships the CUDA/nvptx backend), so the
# Codeplay CUDA plugin does NOT need to be built from source. AOT compiles to PTX/SASS at
# build time for the binding.gyp default arch set (sm_80/89/90); at runtime only the system
# NVIDIA driver (libcuda.so.1) is required, and the CUDA driver JITs PTX where needed.
# https://hub.docker.com/r/nvidia/cuda/tags  https://github.com/intel/llvm/releases
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-c"]

ARG NODE_VERSION=24.15.0
# Prebuilt DPC++ (intel/llvm nightly). The tarball is CUDA-enabled, so this is the "use the
# binary, don't build the plugin" path. Pin a nightly that ships the kernel_compiler SYCL-source
# JIT (libsycl-jit.so) the kawpow algo needs. Bump DPCPP_RELEASE to move forward.
ARG DPCPP_RELEASE=nightly-2026-06-13
ARG DPCPP_ASSET=sycl_linux.tar.gz

# Base tools + libhwloc (UR CUDA adapter dep) + Node. No LLVM/clang apt packages: the DPC++
# tarball below is a self-contained clang + SYCL runtime.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl iputils-ping make python3 sudo xz-utils libhwloc15 && \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o /tmp/node.tar.xz && \
    tar -C /usr/local --strip-components=1 -xf /tmp/node.tar.xz && rm /tmp/node.tar.xz && \
    npm install -g node-gyp@12.2.0 && \
    node --version && npm --version && node-gyp --version

# Install the prebuilt DPC++ CUDA-enabled toolchain into /opt/dpcpp. Self-healing: intel/llvm
# garbage-collects old nightlies, so if the pinned tag has been pruned, resolve the latest
# nightly-YYYY-MM-DD tag from the GitHub releases API and use that instead.
RUN mkdir -p /opt/dpcpp && \
    rel="${DPCPP_RELEASE}" && \
    if ! curl -fsIL "https://github.com/intel/llvm/releases/download/${rel}/${DPCPP_ASSET}" >/dev/null 2>&1; then \
      echo "Pinned ${rel} unavailable (pruned); resolving latest nightly from the GitHub API..." && \
      rel="$(curl -fsSL 'https://api.github.com/repos/intel/llvm/releases?per_page=100' | grep -oE 'nightly-[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -ru | head -1)" && \
      [ -n "${rel}" ] || { echo 'Could not resolve any intel/llvm nightly tag' >&2; exit 1; }; \
    fi && \
    echo "Using DPC++ ${rel}" && \
    curl -fsSL "https://github.com/intel/llvm/releases/download/${rel}/${DPCPP_ASSET}" -o /tmp/dpcpp.tar.gz && \
    tar -C /opt/dpcpp -xf /tmp/dpcpp.tar.gz && rm /tmp/dpcpp.tar.gz && \
    /opt/dpcpp/bin/clang++ --version | head -2

ENV PATH=/opt/dpcpp/bin:$PATH \
    LD_LIBRARY_PATH=/opt/dpcpp/lib:/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH

# Give the root group the same access as the root user to /root, so the
# host-mapped build user (added to group root below) can write into /root/mom.
RUN chmod g=u /root
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
# The DPC++ nightly tarball ships no LLVMgold.so (LTO plugin), so default LTO off.\n\
lto="${MOM_LTO:-0}"\n\
perf_samples="${MOM_PERF_SAMPLES:-}"\n\
su - user <<EOF\n\
cd /root/mom # su - resets to home dir and we need to keep /root/mom pwd\n\
export PATH=/opt/dpcpp/bin:$PATH\n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH\n\
export MOM_SYCL_IMPL=dpcpp-cuda\n\
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-cuda:*}" # DPC++ selects the CUDA backend\n\
export SYCL_CACHE_PERSISTENT=1 # cache the kawpow kernel_compiler JIT between runs\n\
export MOM_PORTABLE_BUILD="$portable_build"\n\
export MOM_LTO="$lto"\n\
export MOM_PERF_SAMPLES="$perf_samples"\n\
{ ping -c1 -W2 8.8.8.8 >/dev/null 2>&1; } && npm update --silent || echo "Skip npm update since there is no internet access"\n\
node_build_version="\$(node -p "process.version"):/usr/local:portable=$portable_build:ax=1:lto=$lto:impl=dpcpp-cuda"\n\
if [ ! -s ./build/Release/mom.node ] || [ "\$(cat ./build/.node-version 2>/dev/null || true)" != "\$node_build_version" ] || ! grep -q "/root/mom" ./build/Makefile 2>/dev/null; then\n\
  rm -rf ./build\n\
  CC=clang CXX=clang++ node-gyp configure --nodedir=/usr/local -- -Dmom_sycl_impl=dpcpp-cuda\n\
fi &&\n\
JOBS=$(nproc) CC=clang CXX=clang++ MAKEFLAGS=-s node-gyp build --nodedir=/usr/local --silent &&\n\
mkdir -p ./build && echo "\$node_build_version" > ./build/.node-version &&\n\
({ test $# -eq 1; } && { echo "One param mode"; sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH MOM_PERF_SAMPLES="\$MOM_PERF_SAMPLES" -- /bin/bash -c ${*@Q}; } || sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH MOM_PERF_SAMPLES="\$MOM_PERF_SAMPLES" -- ${*@Q})\n\
EOF' >/root/entrypoint.sh &&\
chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]

# Default command (after the entrypoint syncs the user, builds, and elevates): run the test suite.
WORKDIR /root/mom
CMD node --require ./tests/common/test_output_buffer.js --test --test-reporter=./tests/common/spec_reporter.js --test-concurrency=1 tests/all.js
