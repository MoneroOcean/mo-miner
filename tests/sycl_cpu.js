"use strict";

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");

const { getFirstSyclCpuDevice, runMinerTest } = require("./common/miner_command");
const { hashTests } = require("./vectors");

const CPU_TEST_TIMEOUT_MS = 15 * 60 * 1000;

function cloneForSyclCpu(definition, dev) {
  const copy = JSON.parse(JSON.stringify(definition));
  copy.name = `${copy.name.replace(/gpu1/g, dev)} SYCL CPU`;
  copy.gpu = false;
  copy.timeoutMs = CPU_TEST_TIMEOUT_MS;
  copy.job.dev = copy.job.dev.replace(/gpu1/g, dev);
  return copy;
}

function requiredVector(name) {
  const definition = hashTests.find((v) => v.name === name);
  assert.ok(definition, `Missing hash vector: ${name}`);
  return definition;
}

// MOM_SYCL_CPU_SKIP lets a build opt specific algos out of SYCL-CPU hash verification
// (comma-separated algo names) when a host SYCL CPU device miscompiles them. The Intel build
// leaves it unset and runs all five; the NVIDIA build is CUDA-only (no SYCL CPU device), so it
// skips this verification entirely.
const skipAlgos = new Set(
  (process.env.MOM_SYCL_CPU_SKIP || "").split(",").map((s) => s.trim()).filter(Boolean)
);

// These algos are ordinary SYCL kernels (plain integer/FP/memory work; kawpow just swaps its
// warp-shuffle exchange for an SLM/barrier one when is_gpu()==false), so the SAME device code runs
// unchanged on the OpenCL CPU device as on the GPU -- no CPU-specific implementation needed, which
// is what makes this CPU verification possible.
//
// pearl is here too now: the OpenCL CPU device (backend==opencl) takes the same matrix-hardware-free
// portable int8 GEMM (sycl/pearl.cpp search()) that AMD GPUs use -- its 4-wide integer dot product
// degrades to plain scalar int MACs on the CPU. It is the SAME device kernel the `opencl` suite runs on
// the GPU (cross-validated bit-identical to the ESIMD/XMX path), so it is a real regression guard. The
// XMX (ESIMD) and tensor-core (mma.sync) pearl paths still can't run on the CPU, but they don't need to:
// the portable path is what runs on every OpenCL device. It is slow on the CPU but the test shape is tiny.
// firopow/evrprogpow run the kawpow kernel family, so the same SLM/barrier CPU path applies. They use
// the SMALLEST height (epoch 0) to keep the CPU-built DAG as small as possible -- but note chfast init
// sizing makes even epoch 0 large: firopow ~1.5 GiB, evrprogpow ~3 GiB (vs ~1 GiB for etchash/kawpow
// epoch 0), so these are the slowest CPU vectors. The live mainnet-DAG vectors stay GPU-only.
const syclCpuVectors = [
  requiredVector("cn/gpu gpu1*8"),
  requiredVector("kawpow gpu1*256"),
  requiredVector("firopow gpu1*1 height 1"),
  requiredVector("evrprogpow gpu1*1 height 0"),
  requiredVector("etchash gpu1*256"),
  requiredVector("autolykos2 gpu1*1"),
  requiredVector("c29 proofsize 42 gpu1*1"),
  requiredVector("pearl gpu1*1"),
].filter((definition) => !skipAlgos.has(definition.job.algo));

describe("SYCL CPU hash vectors", () => {
  let detectedDevice;

  async function getDevice(t) {
    if (!detectedDevice) detectedDevice = getFirstSyclCpuDevice();
    const result = await detectedDevice;
    if (result.skipped) {
      t.skip(result.reason);
      return null;
    }
    return result.dev;
  }

  for (const definition of syclCpuVectors) {
    it(definition.name.replace(/gpu1/g, "SYCL CPU"), { timeout: CPU_TEST_TIMEOUT_MS }, async (t) => {
      const dev = await getDevice(t);
      if (!dev) return;

      // runMinerTest throws on failure, so a non-skipped return already means the vector passed.
      const result = await runMinerTest(cloneForSyclCpu(definition, dev));
      if (result.skipped) t.skip(result.reason);
    });
  }
});
