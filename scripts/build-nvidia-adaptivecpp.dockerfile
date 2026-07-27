# Comparison/optional NVIDIA backend: the same pinned AdaptiveCpp develop SSCP
# compiler used by the working Windows AMD path. Generic kernels are JIT'd for
# the installed NVIDIA GPU, while ProgPoW keeps the shared SYCL source-JIT path.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS adaptivecpp
SHELL ["/bin/bash", "-c"]
ARG ADAPTIVECPP_COMMIT=da2463e45aa90aa36306c45abcfc05b87de51bc6
ARG MOM_BUILD_JOBS=6
COPY scripts/install-dev.sh /tmp/mom-install-dev/install-dev.sh
COPY scripts/install-cutlass.sh /tmp/mom-install-dev/install-cutlass.sh
COPY scripts/patches/adaptivecpp-cuda-unloading.patch /tmp/mom-install-dev/patches/adaptivecpp-cuda-unloading.patch
RUN MOM_ADAPTIVECPP_COMMIT="$ADAPTIVECPP_COMMIT" \
      bash /tmp/mom-install-dev/install-dev.sh --component acpp-cuda --jobs "$MOM_BUILD_JOBS"
RUN mkdir -p /opt/ubuntu24-libs && \
    for pattern in \
      libLLVM.so.21.1 libclang-cpp.so.21.1 libclang-21.so.21 \
      libffi.so.8 libedit.so.2 libz.so.1 libzstd.so.1 libxml2.so.2 \
      libtinfo.so.6 libbsd.so.0 libicuuc.so.74 liblzma.so.5 libmd.so.0 libicudata.so.74 \
      libomp.so.5 libhwloc.so.15 'libboost_context.so.*' 'libboost_fiber.so.*'; do \
      for library in /usr/lib/x86_64-linux-gnu/$pattern /usr/lib/llvm-21/lib/$pattern; do \
        [ ! -e "$library" ] || cp -L -n "$library" /opt/ubuntu24-libs/; \
      done; \
    done

# Internal cache payload only. CUDA and Node are already present in the single runnable combined
# image; exporting them here made this one cache input roughly 14 GB. Preserve only AdaptiveCpp and
# the Ubuntu-24 LLVM/JIT closure needed to construct a backward-compatible release archive.
FROM ubuntu:24.04
COPY --from=adaptivecpp /opt/adaptivecpp-cuda /opt/adaptivecpp
COPY --from=adaptivecpp /usr/lib/llvm-21 /opt/llvm21-ubuntu24
COPY --from=adaptivecpp /opt/ubuntu24-libs /opt/ubuntu24-libs
