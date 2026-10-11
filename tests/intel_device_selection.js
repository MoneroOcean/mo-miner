"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const {extractCppBlock} = require("./common/cpp_source");

test.describe("Native Intel device selection", {skip: process.platform === "win32"}, () => {
  let temporary = "";
  let executable = "";

  test.before(() => {
    const root = path.join(__dirname, "..");
    const runtime = fs.readFileSync(path.join(root, "sycl/runtime.inc"), "utf8");
    const internal = fs.readFileSync(path.join(root, "sycl/lib-internal.h"), "utf8");
    const params = fs.readFileSync(path.join(root, "sycl/algo_params.inc"), "utf8");
    const discoveryStart = runtime.indexOf("static std::string gpu_identity_key(");
    assert.ok(discoveryStart >= 0, "Missing native discovery functions");
    temporary = fs.mkdtempSync(path.join(os.tmpdir(), "mom-intel-selection-"));
    executable = path.join(temporary, "selection");
    const source = String.raw`
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace sycl {
namespace info {
namespace device { struct name {}; struct vendor {}; struct host_unified_memory {}; }
namespace platform { struct name {}; }
}
struct platform;
struct device {
  std::string name, vendor, api;
  bool integrated = false, cpu = false, fault = false;
  bool is_cpu() const { return cpu; }
  bool is_gpu() const { return !cpu; }
  platform get_platform() const;
  template<class T> auto get_info() const {
    if constexpr (std::is_same_v<T, info::device::host_unified_memory>) {
      if (fault) throw std::runtime_error("metadata fault");
      return integrated;
    } else if constexpr (std::is_same_v<T, info::device::vendor>) {
      return vendor;
    } else {
      return name;
    }
  }
};
struct platform {
  std::string name;
  std::vector<device> devices;
  static std::vector<platform> fixtures;
  static std::vector<platform> get_platforms() { return fixtures; }
  std::vector<device> get_devices() const { return devices; }
  template<class T> std::string get_info() const { return name; }
};
std::vector<platform> platform::fixtures;
platform device::get_platform() const { return {api, {}}; }
}
bool mom_is_cuda(const sycl::device& d) { return d.api == "CUDA"; }
bool mom_is_hip(const sycl::device& d) { return d.api == "HIP"; }
bool mom_is_opencl(const sycl::device& d) { return d.api == "OpenCL"; }
bool sycl_is_level_zero_gpu(const sycl::device& d) { return d.is_gpu() && d.api == "Level-Zero"; }
static std::map<std::string, sycl::device> str2dev;
static std::string available_dev_str();
` + extractCppBlock(internal, "inline bool mom_parse_env_ulong(") +
      extractCppBlock(internal, "inline bool is_integrated_gpu(") +
      runtime.slice(discoveryStart) +
      extractCppBlock(params, "sycl::device get_dev(") + String.raw`
int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "mixed";
  const sycl::device cpu{"CPU", "Host", "OpenCL", false, true};
  // Misleading names ensure memory metadata, not product-name heuristics, decides the split.
  const sycl::device integrated{"Z Discrete-looking", "Intel Corporation", "OpenCL", true};
  const sycl::device discrete{"B Integrated-looking", "Intel Corporation", "OpenCL", false};
  const sycl::device amd{"A AMD", "Advanced Micro Devices", "OpenCL", false};
  const sycl::device nvidia{"N NVIDIA", "NVIDIA Corporation", "CUDA", false};
  sycl::platform::fixtures = {{"OpenCL", {integrated, discrete, amd, cpu}}, {"CUDA", {nvidia}}};
  if (mode == "igpu-only") sycl::platform::fixtures = {{"OpenCL", {integrated, cpu}}};
  if (mode == "dgpu-only") sycl::platform::fixtures = {{"OpenCL", {discrete, cpu}}};
  if (mode == "duplicates") {
    auto preferred = discrete;
    preferred.api = "Level-Zero";
    auto preferred_integrated = integrated;
    preferred_integrated.api = "Level-Zero";
    sycl::platform::fixtures = {{"OpenCL", {integrated, discrete, cpu}},
                                {"Level-Zero", {preferred_integrated, preferred}}};
  }
  if (mode == "fault" || mode == "retry-fault") {
    auto failed = discrete;
    failed.fault = true;
    sycl::platform::fixtures = {{"OpenCL", {failed, cpu}}};
  }
  if (mode == "retry-fault" || mode == "retry-index") {
    try {
      update_str2dev();
    } catch (...) {
      const bool cleared = str2dev.empty();
      try {
        get_dev(mode == "retry-fault" ? "gpu1" : "gpu2");
      } catch (...) {
        std::cout << (cleared ? "cleared;retry rejected\n" : "retained;retry rejected\n");
        return cleared ? 0 : 4;
      }
    }
    std::cerr << "unvalidated device admitted\n";
    return 4;
  }
  try {
    update_str2dev();
    for (const auto& [key, device] : str2dev)
      std::cout << key << ":" << device.name << ":" << device.api << "\n";
  } catch (const std::string& error) {
    std::cerr << error << "\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 3;
  }
}
`;
    const file = path.join(temporary, "selection.cpp");
    fs.writeFileSync(file, source);
    const compile = spawnSync(process.env["CXX"] || "c++", [
      "-std=c++20", "-O0", "-Wall", "-Wextra", "-Werror", file, "-o", executable,
    ], {encoding: "utf8"});
    assert.equal(compile.status, 0, compile.stderr);
  });

  test.after(() => {
    if (temporary) {fs.rmSync(temporary, {recursive: true, force: true});}
  });

  /** @param {string} backend @param {string} mode @param {string} index */
  function run(backend, mode = "mixed", index = "") {
    const env = {...process.env};
    delete env["MOM_GPU_BACKEND"];
    delete env["MOM_GPU_INDEX"];
    if (backend) {env["MOM_GPU_BACKEND"] = backend;}
    if (index) {env["MOM_GPU_INDEX"] = index;}
    return spawnSync(executable, [mode], {env, encoding: "utf8"});
  }

  const cpu = "cpu1:CPU:OpenCL\n";
  const discrete = "gpu2:B Integrated-looking:OpenCL\n";
  const integrated = "gpu4:Z Discrete-looking:OpenCL\n";
  const all = cpu + "gpu1:A AMD:OpenCL\n" + discrete + "gpu3:N NVIDIA:CUDA\n" + integrated;

  for (const backend of ["intel", "INTEL", "intel_igpu", "INTEL_IGPU"]) {
    test(`${backend} selects only the requested Intel memory class`, () => {
      const result = run(backend);
      assert.equal(result.status, 0, result.stderr);
      assert.equal(result.stdout, cpu + (backend.toLowerCase() === "intel" ? discrete : integrated));
    });
  }

  for (const backend of ["", "all", "opencl", "amd", "nvidia", "cpu"]) {
    test(`${backend || "unset"} leaves generic native discovery unchanged`, () => {
      const result = run(backend);
      assert.equal(result.status, 0, result.stderr);
      assert.equal(result.stdout, all);
    });
  }

  test("Intel index selects the original stable name, not a filtered ordinal", () => {
    const result = run("intel", "mixed", "1");
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, cpu + discrete);
  });

  test("uppercase iGPU index selects the original stable name", () => {
    const result = run("INTEL_IGPU", "mixed", "3");
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, cpu + integrated);
  });

  test("excluded and absent indices fail without fallback or remapping", () => {
    /** @type {Array<[string, string]>} */
    const cases = [["intel", "0"], ["intel", "3"], ["intel_igpu", "1"],
      ["intel_igpu", "0"], ["intel", "9"], ["intel_igpu", "9"]];
    for (const [backend, index] of cases) {
      const result = run(backend, "mixed", index);
      assert.equal(result.status, 2, `${backend}:${index} ${result.stderr}`);
      assert.equal(result.stdout, "");
      assert.match(result.stderr, /MOM_GPU_INDEX selects gpu/);
    }
  });

  test("missing Intel memory class never admits the opposite class", () => {
    /** @type {Array<[string, string]>} */
    const cases = [["intel", "igpu-only"], ["intel_igpu", "dgpu-only"]];
    for (const [backend, mode] of cases) {
      const result = run(backend, mode);
      assert.equal(result.status, 0, result.stderr);
      assert.equal(result.stdout, cpu);
      assert.equal(run(backend, mode, "0").status, 2);
    }
  });

  test("duplicate API devices retain one stable identity and Level Zero preference", () => {
    const result = run("intel", "duplicates");
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, cpu + "gpu1:B Integrated-looking:Level-Zero\n");
    const igpu = run("intel_igpu", "duplicates", "1");
    assert.equal(igpu.status, 0, igpu.stderr);
    assert.equal(igpu.stdout, cpu + "gpu2:Z Discrete-looking:Level-Zero\n");
  });

  test("Intel metadata query failure cannot classify a GPU as discrete", () => {
    for (const backend of ["intel", "intel_igpu"]) {
      const result = run(backend, "fault");
      assert.equal(result.status, 3, result.stderr);
      assert.equal(result.stdout, "");
      assert.equal(result.stderr, "metadata fault\n");
    }
    assert.equal(run("opencl", "fault").status, 0);
  });

  test("generic OpenCL index behavior stays stable", () => {
    const result = run("opencl", "mixed", "0");
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, cpu + "gpu1:A AMD:OpenCL\n");
  });

  test("metadata failure clears cached inventory and actual get_dev retries fail closed", () => {
    const result = run("intel", "retry-fault");
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, "cleared;retry rejected\n");
  });

  test("excluded index failure cannot be bypassed by an actual get_dev retry", () => {
    const result = run("intel", "retry-index", "0");
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, "cleared;retry rejected\n");
  });
});
