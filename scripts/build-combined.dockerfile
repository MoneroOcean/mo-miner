# Combined build image (Option C): one mom.node that runs on BOTH Intel and NVIDIA GPUs.
# Dual-compiler: the CPU/host objects are built with oneAPI `icx`/`icpx` (keeps the
# icx-only RandomX codegen advantage), while the SYCL device objects + the final link
# use the intel/llvm nightly `clang -fsycl` (the only toolchain that AOTs to BOTH spir64
# and nvptx in one binary). The two compilers are intel/llvm-based and ABI-compatible.
#
# Base = the Intel oneAPI toolkit (icx + the Intel SYCL/OpenCL runtime), with the nightly
# DPC++ tarball laid down beside it in /opt/dpcpp, plus the CUDA toolkit (libdevice + ptxas)
# the nvptx AOT compile/link needs. The actual build steps live in the editable repo script
# scripts/combined-build.sh so they can be iterated without rebuilding this image.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS cuda

FROM intel/oneapi-toolkit:2026.0.0-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

ARG NODE_VERSION=24.15.0
# Pin a nightly that ships the kernel_compiler SYCL-source JIT (libsycl-jit.so) the kawpow
# algo needs, matching scripts/build-nvidia.dockerfile. Bump together with that file.
ARG DPCPP_RELEASE=nightly-2026-06-13
ARG DPCPP_ASSET=sycl_linux.tar.gz

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl iputils-ping make python3 sudo xz-utils libhwloc15 && \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o /tmp/node.tar.xz && \
    tar -C /usr/local --strip-components=1 -xf /tmp/node.tar.xz && \
    rm /tmp/node.tar.xz && \
    npm install -g node-gyp@12.2.0 && \
    node --version && npm --version && node-gyp --version

# Install the prebuilt DPC++ CUDA-enabled nightly toolchain into /opt/dpcpp. Self-healing:
# if the pinned tag was GC'd by intel/llvm, resolve the latest nightly via the releases API.
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

# CUDA toolkit for the nvptx AOT path: clang needs libdevice (device-math bitcode) to compile and
# ptxas to assemble the nvptx images. Copied from the CUDA image (the same toolkit the dedicated
# NVIDIA build uses); no driver is needed at build time. clang auto-detects /usr/local/cuda.
COPY --from=cuda /usr/local/cuda-12.6 /usr/local/cuda-12.6
RUN ln -s /usr/local/cuda-12.6 /usr/local/cuda

RUN chmod g=u /root
ENV PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    MOM_DPCPP_ROOT=/opt/dpcpp \
    CUDA_PATH=/usr/local/cuda
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Build as the host user that owns /root/mom (artifacts stay host-writable), run as root
# (needs /dev/cpu/*/msr). The dual-compiler build itself is in scripts/combined-build.sh.
RUN echo $'#!/usr/bin/env bash\n\
(userdel -r "$(getent passwd $(stat -c "%u" /root/mom) | cut -d: -f1)" 2>/dev/null || true)\n\
useradd user -u $(stat -c "%g" /root/mom) -G root,video -m -s /bin/bash;\n\
echo "user ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/user-user\n\
portable_build="${MOM_PORTABLE_BUILD:-0}"\n\
# AOT device target set for the combined build (e.g. release CI widens to multi-arch NVIDIA via\n\
# -e MOM_COMBINED_TARGETS=spir64,nvidia_gpu_sm_80,nvidia_gpu_sm_89,nvidia_gpu_sm_90). Captured in\n\
# the root shell so the unquoted heredoc below carries it into the user build shell; empty -> the\n\
# combined-build.sh default (spir64,nvidia_gpu_sm_89).\n\
combined_targets="${MOM_COMBINED_TARGETS:-}"\n\
su - user <<EOF\n\
cd /root/mom # su - resets to home dir and we need to keep /root/mom pwd\n\
. /opt/intel/oneapi/setvars.sh >/dev/null\n\
export MOM_DPCPP_ROOT=/opt/dpcpp\n\
# The combined mom.node is linked by the nightly clang, so at RUNTIME it must load the\n\
# nightly libsycl.so.9 AND its libsycl-jit.so (kawpow CUDA JIT), not oneAPIs same-soname\n\
# libs that setvars just put on the path. The heredoc is unquoted, so \$LD_LIBRARY_PATH on\n\
# the sudo line below is expanded by the ROOT shell (oneAPI dirs, no /opt/dpcpp); the inner\n\
# export here does not reach it. So the sudo line prepends /opt/dpcpp/lib literally too --\n\
# otherwise dlopen(libsycl-jit.so) finds oneAPIs clang-22 lib and the kawpow JIT fails\n\
# (Device linking: missing clang/22 libclc) and silently falls back to the ~3x slower kernel.\n\
export LD_LIBRARY_PATH=/opt/dpcpp/lib:$LD_LIBRARY_PATH\n\
export MOM_PORTABLE_BUILD="$portable_build"\n\
export MOM_PERF_SAMPLES="${MOM_PERF_SAMPLES:-}"\n\
export MOM_COMBINED_TARGETS="$combined_targets"\n\
export SYCL_CACHE_PERSISTENT=1\n\
{ ping -c1 -W2 8.8.8.8 >/dev/null 2>&1; } && npm update --silent || echo "Skip npm update since there is no internet access"\n\
bash scripts/combined-build.sh &&\n\
({ test $# -eq 1; } && { echo "One param mode"; sudo LD_LIBRARY_PATH=/opt/dpcpp/lib:$LD_LIBRARY_PATH MOM_PERF_SAMPLES="\$MOM_PERF_SAMPLES" -- /bin/bash -c ${*@Q}; } || sudo LD_LIBRARY_PATH=/opt/dpcpp/lib:$LD_LIBRARY_PATH MOM_PERF_SAMPLES="\$MOM_PERF_SAMPLES" -- ${*@Q})\n\
EOF' >/root/entrypoint.sh &&\
chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]
WORKDIR /root/mom
CMD node --require ./tests/common/test_output_buffer.js --test --test-reporter=./tests/common/spec_reporter.js --test-concurrency=1 tests/all.js
