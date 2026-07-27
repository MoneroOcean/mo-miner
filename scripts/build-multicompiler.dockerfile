# One development image containing every SYCL compiler used by the policy table.
# The component images remain BuildKit stages/cache only; users run this final image.
FROM mom-build-combined:latest AS combined
FROM mom-build-amd-acpp-baseline:latest AS acpp_hip
FROM mom-build-nvidia-adaptivecpp:latest AS acpp_cuda

# Keep the combined Ubuntu-24 oneAPI/DPC++/CUDA image as the actual parent. The old Ubuntu-26
# scratch final copied its complete ~25 GiB toolchain closure into a second ~29 GiB image, leaving
# almost no shared layers and exceeding the practical disk budget of GitHub-hosted runners. Adding
# ROCm and the two source-built compiler payloads here preserves the exact compiler binaries while
# sharing all large Intel/NVIDIA layers with mom-build-combined.
FROM combined
SHELL ["/bin/bash", "-c"]
COPY scripts/install-cutlass.sh /tmp/mom-install-dev/install-cutlass.sh
# The combined image supplies CUDA/NVRTC. CUTLASS is a small header-only source-JIT dependency and
# belongs in the final developer image even when its much larger compiler parent is reused from an
# older BuildKit cache.
RUN bash -euc '. /tmp/mom-install-dev/install-cutlass.sh; install_cutlass_headers'
COPY scripts/install-dev.sh /tmp/mom-install-dev/install-dev.sh
RUN MOM_ROCM_VERSION=7.1.1 bash /tmp/mom-install-dev/install-dev.sh --component rocm && \
    rm -rf /var/lib/apt/lists/*

COPY --from=acpp_hip /opt/adaptivecpp /opt/adaptivecpp-hip
COPY --from=acpp_cuda /opt/adaptivecpp /opt/adaptivecpp-cuda
# These relocatable Ubuntu-24 inputs provide AdaptiveCpp's compiler/JIT runtime and its complete
# backward-compatible shared-library closure for the release archive.
COPY --from=acpp_cuda /opt/llvm21-ubuntu24 /opt/llvm21-ubuntu24
COPY --from=acpp_cuda /opt/ubuntu24-libs /opt/ubuntu24-libs
COPY --from=acpp_hip /opt/llvm20-ubuntu24 /opt/llvm20-ubuntu24
COPY --from=acpp_hip /opt/ubuntu24-libs /opt/ubuntu24-libs
COPY --from=acpp_hip /opt/rocm-device-libs /opt/rocm-device-libs-ubuntu24
# AdaptiveCpp first checks this install-relative redistributable path, then falls back to the
# compiled SDK location. Populate it in the development image as well as the release package so the
# payload-only component does not need to retain an absolute /opt/rocm tree.
COPY --from=acpp_hip /opt/rocm-device-libs /opt/adaptivecpp-hip/lib/hipSYCL/ext/bitcode/amdgcn
# AdaptiveCpp's Ubuntu-24 LLVM payload is the same pinned LLVM 21 compiler previously installed
# from apt in the Ubuntu-26 final image. Stable command names keep the existing build wrappers and
# avoid adding a duplicate system LLVM tree.
RUN rm -rf /usr/lib/llvm-20 /usr/lib/llvm-21 && \
    ln -s /opt/llvm20-ubuntu24 /usr/lib/llvm-20 && \
    ln -s /opt/llvm21-ubuntu24 /usr/lib/llvm-21 && \
    ln -sfn /usr/lib/llvm-21/bin/clang /usr/local/bin/clang-21 && \
    ln -sfn /usr/lib/llvm-21/bin/clang++ /usr/local/bin/clang++-21
RUN LD_LIBRARY_PATH=/opt/llvm21-ubuntu24/lib:/opt/ubuntu24-libs \
      /usr/lib/llvm-21/bin/clang++ --version >/dev/null && \
    LD_LIBRARY_PATH=/opt/llvm20-ubuntu24/lib:/opt/ubuntu24-libs \
      /usr/lib/llvm-20/bin/clang++ --version >/dev/null && \
    test -r /usr/local/cuda/include/nvrtc.h && \
    test -r /usr/local/cuda/include/cuda/std/cstdint && \
    test -r /opt/mom/cutlass/include/cute/tensor.hpp

ENV PATH=/opt/dpcpp/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/opt/llvm21-ubuntu24/lib:/opt/ubuntu24-libs:/opt/rocm/lib:/opt/dpcpp/lib:/opt/intel/oneapi/compiler/latest/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
    MOM_ROCM_DEVICE_LIBS=/opt/rocm/amdgcn/bitcode \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm
WORKDIR /root/mom
ENTRYPOINT ["bash", "scripts/multicompiler-entrypoint.sh"]
