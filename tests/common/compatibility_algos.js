"use strict";

// XeLP compatibility is intentionally best-effort. These algorithms are proven through both
// Level Zero and OpenCL on the UHD 750 without multi-minute first-use compilation or allocations
// beyond the integrated GPU's practical limit. CPU and discrete OpenCL lanes still test every
// portable GPU algorithm.
const intelIgpuAlgos = new Set([
  "cn/gpu",
  "pearlhash",
  "zelhash",
  "beamhash3",
]);

module.exports = {intelIgpuAlgos};
