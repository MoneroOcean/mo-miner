#pragma once
#include <vector>

namespace mom::runtime {
// Physical-device SMs, independent of CUDA visibility/ordering. Empty means unknown; a failed
// device must not leave an optimistic partial list that selects unsupported compiler features.
std::vector<unsigned> nvidiaComputeCapabilities();
}
