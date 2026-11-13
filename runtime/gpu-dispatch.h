// Native-release routing only. Engineering addons retain their ordinary direct entrypoints.
#pragma once
namespace mom::runtime {
struct GpuTable;
const GpuTable& selectedGpu();
// The event loop must have joined/drained Core workers before retiring their algorithm state.
void cleanupGpu() noexcept;
}
