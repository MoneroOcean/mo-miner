"use strict";

const { describe, it } = require("node:test");

const { runMinerTest } = require("./common/miner_command");
const { hashTests } = require("./vectors");

const GPU_TEST_TIMEOUT_MS = 10 * 60 * 1000;

// Force the SYCL OpenCL backend by selecting the device's OpenCL key (`gpu1o`) instead of the default
// (`gpu1`, which is Level-Zero on Intel / CUDA on NVIDIA). On the Intel B580 this runs the SAME spir64
// image through Intel's OpenCL driver -- our stand-in for an AMD / OpenCL-only GPU, since we have no AMD
// hardware. `gpu` stays true so an absent OpenCL device skips (via isMissingGpuOutput), like the gpu suite.
// pearl needs no special knob here: on any OpenCL device the dispatch already takes the portable dp4a
// int8 GEMM (the AMD-representative kernel), the same one the SYCL-CPU suite exercises on cpu1.
function cloneForOpenCL(definition) {
  const copy = JSON.parse(JSON.stringify(definition));
  copy.name = `${copy.name} OpenCL`;
  copy.job.dev = copy.job.dev.replace(/gpu(\d+)/g, "gpu$1o");
  return copy;
}

describe("OpenCL proof-of-work hash vectors", () => {
  for (const definition of hashTests.filter((entry) => entry.gpu)) {
    const opencl = cloneForOpenCL(definition);
    it(opencl.name, { timeout: opencl.timeoutMs || GPU_TEST_TIMEOUT_MS }, async (t) => {
      // runMinerTest throws on failure, so a non-skipped return already means the vector passed.
      const result = await runMinerTest(opencl);
      if (result.skipped) {t.skip(result.reason);}
    });
  }
});
