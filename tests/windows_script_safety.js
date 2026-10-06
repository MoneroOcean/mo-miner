"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const WINDOWS = process.platform === "win32";
const repo = path.resolve(__dirname, "..");

/** @param {string} driver @param {string[]} args */
function runPowerShell(driver, args) {
  const result = spawnSync("powershell.exe", [
    "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", driver, ...args,
  ], {encoding: "utf8", windowsHide: true, timeout: 30000});
  assert.equal(result.error, undefined, result.error?.message);
  return result;
}

/** @returns {string} */
function makeFixture() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-script-safety-"));
}

/** @param {string} fixture @param {string} source @returns {string} */
function makeDriver(fixture, source) {
  const driver = path.join(fixture, "driver.ps1");
  fs.writeFileSync(driver, source);
  return driver;
}

test("Windows toolchain readiness checks the artifacts consumed by each build", {
  skip: !WINDOWS,
}, () => {
  const fixture = makeFixture();
  const cuda = path.join(fixture, "cuda");
  const hip = path.join(fixture, "hip");
  const dpcpp = path.join(fixture, "dpcpp");
  /** @param {string} root @param {string[]} files */
  const writeFiles = (root, files) => {
    for (const file of files) {
      const destination = path.join(root, ...file.split("/"));
      fs.mkdirSync(path.dirname(destination), {recursive: true});
      fs.writeFileSync(destination, "fixture");
    }
  };
  writeFiles(cuda, [
    "bin/nvcc.exe", "bin/ptxas.exe", "nvvm/libdevice/libdevice.10.bc",
    "include/cuda.h", "include/cuda_runtime.h", "include/nvrtc.h",
    "lib/x64/cuda.lib", "lib/x64/cudart_static.lib", "lib/x64/nvrtc.lib",
  ]);
  writeFiles(hip, [
    "lib/amdhip64.lib", "include/hip/hip_runtime_api.h", "include/hip/hiprtc.h",
    "bin/clang++.exe", "bin/llvm-link.exe", "bin/opt.exe", "bin/llc.exe",
    "bin/lld-link.exe", "amdgcn/bitcode/ockl.bc",
  ]);
  writeFiles(dpcpp, [
    "bin/clang++.exe", "bin/clang.exe", "bin/lld-link.exe", "bin/sycl-post-link.exe",
    "bin/llvm-spirv.exe", "bin/llvm-link.exe", "bin/clang-linker-wrapper.exe",
    "bin/clang-offload-wrapper.exe", "bin/sycl9.dll", "bin/sycl-jit.dll",
    "bin/ur_loader.dll", "bin/ur_adapter_cuda.dll", "bin/ur_adapter_opencl.dll",
    "bin/ur_adapter_level_zero_v2.dll",
  ]);
  const driver = makeDriver(fixture, `
param([string]$Helper, [string]$Cuda, [string]$Hip, [string]$Dpcpp)
$ErrorActionPreference = 'Stop'
. $Helper
[Console]::Write(('cuda={0};hip={1};dpcpp={2}' -f
  (Test-MomCudaCompilerSdk $Cuda), (Test-MomHipSdk $Hip), (Test-MomDpcppToolchain $Dpcpp)))
`);
  /** @returns {string} */
  const status = () => {
    const result = runPowerShell(driver, [
      "-Helper", path.join(repo, "scripts/windows-install-helpers.ps1"),
      "-Cuda", cuda, "-Hip", hip, "-Dpcpp", dpcpp,
    ]);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    return result.stdout.trim();
  };
  try {
    assert.equal(status(), "cuda=True;hip=True;dpcpp=False");
    writeFiles(dpcpp, ["bin/ur_win_proxy_loader.dll"]);
    assert.equal(status(), "cuda=True;hip=True;dpcpp=True");
    fs.rmSync(path.join(cuda, "lib/x64/cudart_static.lib"));
    fs.rmSync(path.join(hip, "include/hip/hiprtc.h"));
    assert.equal(status(), "cuda=False;hip=False;dpcpp=True");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});
