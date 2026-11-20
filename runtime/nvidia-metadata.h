#pragma once
#include <vector>

namespace mom::runtime {
// Physical-device SMs, independent of CUDA visibility/ordering. Empty means unknown; a failed
// device must not leave an optimistic partial list that selects unsupported compiler features.
std::vector<unsigned> nvidiaComputeCapabilities();

struct DisplayAdapter {
  unsigned vendor, device, bus;
};
// Present PCI display adapters for single-vendor backend detection; Windows only (Linux reads
// sysfs from JS). Empty means unknown, which leaves the backend unselected.
std::vector<DisplayAdapter> displayAdapters();
}
