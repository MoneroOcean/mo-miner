# Combined build image: one mom.node that runs on both Intel and NVIDIA GPUs.
# Dual-compiler: the CPU/host objects are built with oneAPI `icx`/`icpx` (keeps the
# icx-only RandomX codegen advantage), while the SYCL device objects + the final link
# use the intel/llvm nightly `clang -fsycl` (the only toolchain that AOTs to BOTH spir64
# and nvptx in one binary). The two compilers are intel/llvm-based and ABI-compatible.
#
# The canonical development installer supplies the compiler-only oneAPI and CUDA payloads. Starting
# from Ubuntu avoids inheriting the multi-gigabyte MKL/MPI/VTune/Fortran and CUDA BLAS/FFT/profiler
# products that the miner neither builds nor packages.
FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

ARG NODE_VERSION=24.15.0
# Pin a nightly that ships the kernel_compiler SYCL-source JIT (libsycl-jit.so) the kawpow
# algo needs, matching scripts/build-nvidia.dockerfile. Bump together with that file.
ARG DPCPP_RELEASE=nightly-2026-07-11
ARG DPCPP_ASSET=sycl_linux.tar.gz

COPY scripts/install-dev.sh scripts/install-cutlass.sh /tmp/mom-install-dev/

# Install common tooling and every prebuilt compiler dependency in one canonical pass. Expensive
# source-built AdaptiveCpp stages remain independently cached by the multicompiler image.
RUN MOM_NODE_VERSION="$NODE_VERSION" MOM_DPCPP_RELEASE="$DPCPP_RELEASE" \
      MOM_DPCPP_ASSET="$DPCPP_ASSET" bash /tmp/mom-install-dev/install-dev.sh \
      --component base,node,oneapi,cuda,dpcpp && rm -rf /var/lib/apt/lists/*

RUN chmod g=u /root
ENV PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib \
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
export MOM_CN_GPU_INTENSITY="${MOM_CN_GPU_INTENSITY:-}"\n\
export MOM_AUTOLYKOS2_WORKGROUP="${MOM_AUTOLYKOS2_WORKGROUP:-}"\n\
export MOM_AUTOLYKOS2_SPLIT="${MOM_AUTOLYKOS2_SPLIT:-}"\n\
export MOM_AUTOLYKOS2_PROFILE="${MOM_AUTOLYKOS2_PROFILE:-}"\n\
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-}"\n\
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-}"\n\
export MOM_COMBINED_TARGETS="$combined_targets"\n\
export SYCL_CACHE_PERSISTENT=1\n\
{ ping -c1 -W2 8.8.8.8 >/dev/null 2>&1; } && npm update --silent || echo "Skip npm update since there is no internet access"\n\
bash scripts/combined-build.sh &&\n\
sudo_env=(LD_LIBRARY_PATH=/opt/dpcpp/lib:$LD_LIBRARY_PATH MOM_PERF_SAMPLES="\$MOM_PERF_SAMPLES")\n\
for v in MOM_CN_GPU_INTENSITY MOM_AUTOLYKOS2_WORKGROUP MOM_AUTOLYKOS2_SPLIT MOM_AUTOLYKOS2_PROFILE ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK; do\n\
  [ -n "\${!v:-}" ] && sudo_env+=("\$v=\${!v}")\n\
done\n\
({ test $# -eq 1; } && { echo "One param mode"; sudo "\${sudo_env[@]}" -- /bin/bash -c ${*@Q}; } || sudo "\${sudo_env[@]}" -- ${*@Q})\n\
EOF' >/root/entrypoint.sh &&\
chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]
WORKDIR /root/mom
CMD node --require ./tests/common/test_output_buffer.js --test --test-reporter=./tests/common/spec_reporter.js --test-concurrency=1 tests/all.js
