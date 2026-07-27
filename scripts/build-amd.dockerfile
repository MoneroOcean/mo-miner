# AMD build: Intel LLVM/DPC++'s HIP backend, built with AMDGPU enabled. Official nightly archives
# omit AMDGPU and the HIP UR adapter, so this is a source build. Build the deployable runtime on
# ROCm's Ubuntu 24.04 image: compiling libsycl on Ubuntu 26 made it require GLIBC_2.43 and broke the
# advertised Ubuntu 24.04 release even though miner code itself only required GLIBC_2.38.
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.1.1
FROM ${ROCM_IMAGE} AS dpcpp

SHELL ["/bin/bash", "-c"]
ARG DPCPP_COMMIT=eca4d070277a1e62b196a5fddefe72bc7f98ee24
ARG MOM_BUILD_JOBS=6
COPY scripts/install-dev.sh /tmp/mom-install-dev/install-dev.sh
RUN MOM_DPCPP_HIP_COMMIT="$DPCPP_COMMIT" MOM_DPCPP_HIP_BRANCH=nightly-2026-07-11 \
      bash /tmp/mom-install-dev/install-dev.sh --component dpcpp-hip --jobs "$MOM_BUILD_JOBS"

# This tag is an immutable BuildKit input, not a runnable development environment. Keeping only the
# deployed compiler avoids caching ROCm's multi-gigabyte SDK a second time; the single multicompiler
# image supplies host clang, HIP headers/runtime, Node, and device libraries when it builds workers.
FROM ubuntu:24.04
COPY --from=dpcpp /opt/dpcpp-amd/ /opt/dpcpp/
RUN clang_lib="$(find /opt/dpcpp/lib/clang -mindepth 1 -maxdepth 1 -type d -print -quit)/lib" && \
    ln -sfn amdgcn-amd-amdhsa-llvm "$clang_lib/amdgcn-amd-amdhsa"
