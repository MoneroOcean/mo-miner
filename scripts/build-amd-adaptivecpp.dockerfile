# AdaptiveCpp generic/SSCP toolchain using the Linux HIP runtime.
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.1.1
FROM ${ROCM_IMAGE} AS adaptivecpp
SHELL ["/bin/bash", "-c"]
ARG ADAPTIVECPP_COMMIT=da2463e45aa90aa36306c45abcfc05b87de51bc6
ARG MOM_BUILD_JOBS=6
COPY scripts/install-dev.sh /tmp/mom-install-dev/install-dev.sh
RUN MOM_ADAPTIVECPP_COMMIT="$ADAPTIVECPP_COMMIT" \
      MOM_LLVM_VERSION=20 \
      bash /tmp/mom-install-dev/install-dev.sh --component acpp-hip --jobs "$MOM_BUILD_JOBS"
RUN mkdir -p /opt/ubuntu24-libs && \
    for pattern in \
      libLLVM.so.20.1 libclang-cpp.so.20.1 libffi.so.8 libedit.so.2 libz.so.1 libzstd.so.1 libxml2.so.2 \
      libtinfo.so.6 libbsd.so.0 libicuuc.so.74 liblzma.so.5 libmd.so.0 libicudata.so.74 \
      libomp.so.5 libhwloc.so.15 'libboost_context.so.*' 'libboost_fiber.so.*'; do \
      for library in /usr/lib/x86_64-linux-gnu/$pattern /usr/lib/llvm-20/lib/$pattern; do \
        [ ! -e "$library" ] || cp -L -n "$library" /opt/ubuntu24-libs/; \
      done; \
    done

# Internal cache payload only. The runnable multicompiler image provides the HIP SDK and build tools;
# retaining them in this intermediate tag would duplicate the complete ROCm base in actions/cache.
FROM ubuntu:24.04
COPY --from=adaptivecpp /opt/adaptivecpp-hip /opt/adaptivecpp
COPY --from=adaptivecpp /usr/lib/llvm-20 /opt/llvm20-ubuntu24
COPY --from=adaptivecpp /opt/ubuntu24-libs /opt/ubuntu24-libs
COPY --from=adaptivecpp /opt/rocm-7.1.1/amdgcn/bitcode /opt/rocm-device-libs
