// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>
//
// Standalone spir64-only ESIMD pearl kernel TU for the combined (Intel + NVIDIA) build.
//
// ESIMD cannot share a -fsycl-targets with nvptx (SYCL_ESIMD_KERNEL is an Intel-only compilation
// mode), so the combined build compiles the Intel ESIMD search HERE for spir64 only (the
// cxx-combined.sh wrapper gives this file -fsycl-targets=spir64) and links it into the SAME mom.node
// as the rest of pearl -- which is built for spir64 + nvptx and carries the NVIDIA mma.sync path.
// SYCL merges the device images from both objects, so the one binary ends up with the Intel ESIMD
// image AND the NVIDIA PTX image; the runtime dispatch in pearl.cpp picks per device backend.
//
// This TU just #includes pearl.cpp with MOM_PEARL_ESIMD_TU + PEARL_ESIMD set: only the namespace-
// scoped device kernels compile, search_esimd is emitted with EXTERNAL linkage (so the main pearl.o
// can call it), and the host entry point + PlainProof builder are excluded (they live in pearl.o).
#define MOM_PEARL_ESIMD_TU
#define PEARL_ESIMD
#include "pearl.cpp"
