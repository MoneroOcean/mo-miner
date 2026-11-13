"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {extractCppBlock} = require("./common/cpp_source");
const root = path.resolve(__dirname, "..");

test.describe("Native CUDA image runtime", {concurrency: false}, () => {
  test("native CUDA image selector and module-load retry policy run in a CPU harness", (t) => {
    const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cuda-image-runtime-"));
    t.after(() => fs.rmSync(scratch, {recursive: true, force: true}));
    fs.writeFileSync(path.join(scratch, "cuda.h"), String.raw`
#pragma once
typedef int CUresult;
typedef int CUdevice;
typedef int CUdevice_attribute;
typedef int CUfunction_attribute;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUevent;
typedef void* CUstream;
enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_INVALID_IMAGE = 200,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
  CUDA_ERROR_INVALID_PTX = 218,
  CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
  CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
  CUDA_ERROR_NOT_READY = 600,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
  CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
  CU_EVENT_DEFAULT = 0
};
extern "C" {
CUresult cuInit(unsigned);
CUresult cuDeviceGetAttribute(int*, CUdevice_attribute, CUdevice);
CUresult cuDevicePrimaryCtxRetain(CUcontext*, CUdevice);
CUresult cuDevicePrimaryCtxRelease(CUdevice);
CUresult cuCtxSetCurrent(CUcontext);
CUresult cuCtxSynchronize();
CUresult cuModuleLoadData(CUmodule*, const void*);
CUresult cuModuleUnload(CUmodule);
CUresult cuModuleGetFunction(CUfunction*, CUmodule, const char*);
CUresult cuFuncSetAttribute(CUfunction, CUfunction_attribute, int);
CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                        unsigned, CUstream, void**, void**);
CUresult cuEventCreate(CUevent*, unsigned);
CUresult cuEventDestroy(CUevent);
CUresult cuEventRecord(CUevent, CUstream);
CUresult cuEventQuery(CUevent);
CUresult cuEventElapsedTime(float*, CUevent, CUevent);
CUresult cuGetErrorString(CUresult, const char**);
}
`);
    fs.writeFileSync(path.join(scratch, "cuda-images.inc"), String.raw`
#include <string_view>
static std::string_view embedded_resource(std::string_view name) {
  static constexpr char cubin[] = "registry-cubin";
  static constexpr char ptx[] = "registry-ptx";
  if (name == "registry-sm80-cubin") return {cubin, sizeof(cubin) - 1};
  if (name == "registry-compute75-ptx-compatibility") return {ptx, sizeof(ptx) - 1};
  return {};
}
static constexpr mom::cuda_images::CudaImageRow kCudaImageRows[] = {
  {"registry", 80, true, false, "registry-sm80-cubin"},
  {"registry", 75, false, true, "registry-compute75-ptx-compatibility"},
};
`);
    fs.writeFileSync(path.join(scratch, "runtime.cpp"), String.raw`
#include <cassert>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string_view>
#include "cuda-images.h"
#include "cuda-api.h"

using mom::cuda_images::Blob;
using mom::cuda_images::Candidates;
using mom::cuda_images::CudaImageRow;
using mom::cuda_images::Kind;

static CUmodule fakeModule(std::uintptr_t value) { return reinterpret_cast<CUmodule>(value); }

struct Resource { const char* name; std::string_view bytes; };
static const Resource resources[] = {
  {"rank-primary-80", std::string_view("ptx80\0", 6)},
  {"rank-primary-86", std::string_view("ptx86\0", 6)},
  {"rank-compat-86", std::string_view("compat86\0", 9)},
  {"rank-compat-90", std::string_view("compat90\0", 9)},
  {"rank-cubin-90", std::string_view("cubin90\0", 8)},
  {"fallback-75", std::string_view("fallback75\0", 11)},
  {"fallback-80", std::string_view("fallback80\0", 11)},
  {"shared", std::string_view("shared\0", 7)},
  {"unique-80", std::string_view("unique80\0", 9)},
  {"compact-120", std::string_view("compact120\0", 11)},
};

static std::string_view lookup(std::string_view name) {
  for (const auto& resource : resources)
    if (name == resource.name)
      return resource.bytes;
  return {};
}

static void test_selection() {
  const CudaImageRow rank128[] = {
    {"pearl-rank128", 80, false, false, "rank-primary-80"},
    {"pearl-rank128", 86, false, false, "rank-primary-86"},
    {"pearl-rank128", 86, false, true, "rank-compat-86"},
    {"pearl-rank128", 90, true, false, "rank-cubin-90"},
    {"pearl-rank128", 90, false, true, "rank-compat-90"},
    {"pearl-rank128", 92, false, true, "not-for-sm90"},
  };
  const auto exact = mom::cuda_images::select("pearl-rank128", 90, rank128,
                                              sizeof(rank128) / sizeof(rank128[0]), lookup);
  assert(exact.primary.data && exact.primary.kind == Kind::cubin && exact.primary.arch == 90);
  assert(std::string_view(exact.primary.data, exact.primary.size) == lookup("rank-cubin-90"));
  assert(exact.compatibility.data && exact.compatibility.kind == Kind::ptx &&
         exact.compatibility.arch == 90);

  const CudaImageRow rank256[] = {
    {"pearl-rank256", 80, false, false, "rank-primary-80"},
    {"pearl-rank256", 86, false, false, "rank-primary-86"},
    {"pearl-rank256", 89, false, true, "rank-compat-86"},
    {"pearl-rank256", 90, false, true, "rank-compat-90"},
    {"pearl-rank256", 100, true, false, "rank-cubin-90"},
  };
  const auto ptx = mom::cuda_images::select("pearl-rank256", 90, rank256,
                                            sizeof(rank256) / sizeof(rank256[0]), lookup);
  assert(ptx.primary.data && ptx.primary.kind == Kind::ptx && ptx.primary.arch == 86);
  assert(std::string_view(ptx.primary.data, ptx.primary.size) == lookup("rank-primary-86"));
  assert(ptx.compatibility.data && ptx.compatibility.arch == 90);

  const CudaImageRow fallback[] = {
    {"fallback-test", 75, false, true, "fallback-75"},
    {"fallback-test", 80, false, true, "fallback-80"},
    {"fallback-test", 86, false, true, "above-sm"},
  };
  const auto compatible = mom::cuda_images::select("fallback-test", 80, fallback,
                                                    sizeof(fallback) / sizeof(fallback[0]), lookup);
  assert(compatible.primary.data && compatible.primary.arch == 80);
  assert(compatible.compatibility.data && compatible.compatibility.arch == 75);

  const CudaImageRow duplicate[] = {
    {"duplicate", 80, false, false, "shared"},
    {"duplicate", 90, false, true, "shared"},
    {"duplicate", 80, false, true, "unique-80"},
  };
  const auto distinct = mom::cuda_images::select("duplicate", 90, duplicate,
                                                  sizeof(duplicate) / sizeof(duplicate[0]), lookup);
  assert(distinct.primary.data && distinct.primary.arch == 80);
  assert(distinct.compatibility.data &&
         std::string_view(distinct.compatibility.data, distinct.compatibility.size) == lookup("unique-80"));

  const CudaImageRow compact[] = {
    {"pearl-compact", 120, true, false, "compact-120"},
    {"pearl-compact", 120, false, true, "rank-compat-90"},
  };
  const auto compact_ok = mom::cuda_images::select("pearl-compact", 120, compact,
                                                    sizeof(compact) / sizeof(compact[0]), lookup);
  assert(compact_ok.primary.data && compact_ok.primary.kind == Kind::cubin &&
         compact_ok.primary.arch == 120 && !compact_ok.compatibility.data);
  const auto compact_fail = mom::cuda_images::select("pearl-compact", 121, compact,
                                                      sizeof(compact) / sizeof(compact[0]), lookup);
  assert(!compact_fail.primary.data && !compact_fail.compatibility.data);
  const auto compact_below = mom::cuda_images::select("pearl-compact", 119, compact,
                                                       sizeof(compact) / sizeof(compact[0]), lookup);
  assert(!compact_below.primary.data && !compact_below.compatibility.data);
  const auto unknown = mom::cuda_images::select("missing-profile", 90, rank128,
                                                 sizeof(rank128) / sizeof(rank128[0]), lookup);
  assert(!unknown.primary.data && !unknown.compatibility.data);

  const CudaImageRow native_profiles[] = {
    {"cn-expand", 80, false, false, "fallback-80"},
    {"pearl-rank128", 80, false, false, "rank-primary-80"},
    {"pearl-rank256", 80, false, false, "rank-primary-80"},
  };
  // These images accelerate SM80+ only; older/unknown devices must retain their SYCL route.
  for (const int sm : {0, 50, 52, 60, 61, 70, 75, 79, 1000}) {
    for (const char* profile : {"cn-expand", "pearl-rank128", "pearl-rank256"}) {
      const auto legacy = mom::cuda_images::select(profile, sm, native_profiles,
          sizeof(native_profiles) / sizeof(native_profiles[0]), lookup);
      assert(!legacy.primary.data && !legacy.compatibility.data);
    }
  }
  const auto unknown_sm = mom::cuda_images::select("cn-expand", 0, native_profiles,
      sizeof(native_profiles) / sizeof(native_profiles[0]), lookup);
  assert(!unknown_sm.primary.data && !unknown_sm.compatibility.data);
}

static void test_loader_retries() {
  const Candidates candidates{
    Blob{"primary", 7, Kind::cubin, 90}, Blob{"compatibility", 12, Kind::ptx, 86}};
  const CUresult retryable[] = {CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_NO_BINARY_FOR_GPU,
                                CUDA_ERROR_INVALID_PTX, CUDA_ERROR_UNSUPPORTED_PTX_VERSION};
  for (const CUresult error : retryable) {
    int calls = 0;
    CUmodule module = fakeModule(1);
    const CUresult status = mom::cuda_load_image_candidates(
        candidates, &module, [&](CUmodule* output, const void* image) {
          ++calls;
          if (calls == 1) {
            assert(image == candidates.primary.data);
            *output = fakeModule(2);
            return error;
          }
          assert(image == candidates.compatibility.data);
          assert(*output == nullptr);
          *output = fakeModule(3);
          return static_cast<CUresult>(CUDA_SUCCESS);
        });
    assert(status == CUDA_SUCCESS && calls == 2 && module == fakeModule(3));
  }

  const CUresult stop[] = {CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_INVALID_CONTEXT,
                           CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_JIT_COMPILER_NOT_FOUND, 777};
  for (const CUresult error : stop) {
    int calls = 0;
    CUmodule module = fakeModule(9);
    const CUresult status = mom::cuda_load_image_candidates(
        candidates, &module, [&](CUmodule* output, const void*) {
          ++calls;
          *output = fakeModule(8);
          return error;
        });
    assert(status == error && calls == 1 && module == nullptr);
  }

  int calls = 0;
  CUmodule module = nullptr;
  const CUresult success = mom::cuda_load_image_candidates(
      candidates, &module, [&](CUmodule* output, const void* image) {
        ++calls;
        assert(image == candidates.primary.data);
        *output = fakeModule(4);
        return CUDA_SUCCESS;
      });
  assert(success == CUDA_SUCCESS && calls == 1 && module == fakeModule(4));

  const Candidates second_only{Blob{}, Blob{"only", 4, Kind::ptx, 80}};
  calls = 0;
  const CUresult fallback = mom::cuda_load_image_candidates(
      second_only, &module, [&](CUmodule*, const void* image) {
        ++calls;
        assert(image == second_only.compatibility.data);
        return CUDA_SUCCESS;
      });
  assert(fallback == CUDA_SUCCESS && calls == 1);
  module = fakeModule(7);
  assert(mom::cuda_load_image_candidates(Candidates{}, &module,
      [&](CUmodule*, const void*) { assert(false); return CUDA_SUCCESS; }) == CUDA_ERROR_INVALID_IMAGE);
  assert(module == nullptr);
  assert(mom::cuda_load_image_candidates(candidates, nullptr,
      [&](CUmodule*, const void*) { assert(false); return CUDA_SUCCESS; }) == CUDA_ERROR_INVALID_VALUE);
}

static void test_embedded_registry() {
  const auto selected = mom::cuda_images::select_embedded("registry", 80);
  assert(selected.primary.data && selected.primary.kind == Kind::cubin && selected.primary.arch == 80);
  assert(std::string_view(selected.primary.data, selected.primary.size) == "registry-cubin");
  assert(selected.compatibility.data && selected.compatibility.arch == 75);
  assert(std::string_view(selected.compatibility.data, selected.compatibility.size) == "registry-ptx");

  // The shared C bridge accepts a bounded view, not a C++ string or a NUL-terminated name.
  const char profile[] = {'r', 'e', 'g', 'i', 's', 't', 'r', 'y', 'x'};
  Candidates output;
  mom::cuda_images::mom_cuda_images_select_embedded(profile, 8, 80, &output);
  assert(output.primary.data == selected.primary.data && output.primary.size == selected.primary.size);
  mom::cuda_images::mom_cuda_images_select_embedded(nullptr, 8, 80, &output);
  assert(!output.primary.data && !output.compatibility.data);
  mom::cuda_images::mom_cuda_images_select_embedded(profile, 129, 80, &output);
  assert(!output.primary.data && !output.compatibility.data);
  mom::cuda_images::mom_cuda_images_select_embedded(profile, 0, 80, &output);
  assert(!output.primary.data && !output.compatibility.data);
  mom::cuda_images::mom_cuda_images_select_embedded(profile, 8, 80, nullptr);
}

int main() {
  test_selection();
  test_loader_retries();
  test_embedded_registry();
}
`);

    const candidates = process.env["CXX"] ? [process.env["CXX"]] :
      process.platform === "win32" ? ["clang-cl", "cl", "g++", "clang++"] : ["c++", "clang++", "g++"];
    let compiler = "", msvc = false;
    for (const candidate of candidates) {
      const name = path.basename(candidate).toLowerCase();
      const isMsvc = name === "cl" || name === "cl.exe" || name === "clang-cl" || name === "clang-cl.exe";
      const probe = spawnSync(candidate, [isMsvc && (name === "cl" || name === "cl.exe") ? "/?" : "--version"],
        {encoding: "utf8", timeout: 10000});
      if (!probe.error) {compiler = candidate; msvc = isMsvc; break;}
    }
    if (!compiler) {t.skip("no host C++ compiler is available for the CPU harness"); return;}
    const binary = path.join(scratch, "runtime-test");
    const sources = [path.join(scratch, "runtime.cpp"), path.join(root, "sycl/cuda-images.cpp")];
    const args = msvc ? ["/nologo", "/std:c++17", "/W4", "/DMOM_EMBEDDED_CUDA_IMAGES",
      "/DMOM_SYCL_HAS_CUDA", `/I${scratch}`, `/I${path.join(root, "sycl")}`,
      ...sources, `/Fe:${binary}.exe`] : ["-std=c++17", "-Wall", "-Wextra", "-Werror",
      "-DMOM_EMBEDDED_CUDA_IMAGES", "-DMOM_SYCL_HAS_CUDA", `-I${scratch}`,
      `-I${path.join(root, "sycl")}`, ...sources, "-o", binary,
      ...(process.platform === "win32" ? [] : ["-ldl"] )];
    const compile = spawnSync(compiler, args, {encoding: "utf8", timeout: 30000});
    assert.equal(compile.status, 0, compile.stderr || compile.stdout);
    const run = spawnSync(process.platform === "win32" && msvc ? binary + ".exe" : binary, [],
      {encoding: "utf8", timeout: 10000});
    assert.equal(run.status, 0, run.stderr || run.stdout);
  });

  test("native CUDA image paths do not include NVRTC API, header-pack, or cache code", () => {
    /** @param {string} file */
    const read = file => fs.readFileSync(path.join(root, file), "utf8");
    const api = read("sycl/cuda-api.h"), cn = read("sycl/cn_gpu/cuda_jit.inc");
    const cnCpp = read("sycl/cn_gpu/cn_gpu.cpp"), pearlAlias = read("sycl/pearlhash/cuda_jit.inc");
    const pearl = read("sycl/pearlhash/cuda_search.inc"), pearlCpp = read("sycl/pearlhash/pearlhash.cpp");
    assert.match(api, /#if defined\(MOM_EMBEDDED_CUDA_IMAGES\)[\s\S]*?#include "cuda-images\.h"[\s\S]*?#else[\s\S]*?#include "nvrtc-api\.h"/);
    assert.match(cn, /#if !defined\(MOM_EMBEDDED_CUDA_IMAGES\)[\s\S]*?using CnGpuNvrtcApi = mom::NvrtcApi/);
    assert.match(pearlAlias, /#if !defined\(MOM_EMBEDDED_CUDA_IMAGES\)[\s\S]*?using PearlHashNvrtcApi = mom::NvrtcApi/);
    assert.match(pearl, /#if !defined\(MOM_EMBEDDED_CUDA_IMAGES\)[\s\S]*?compiler_headers\(\)/);
    assert.match(cnCpp, /!defined\(MOM_EMBEDDED_CUDA_IMAGES\)[\s\S]*?#include "\.\.\/jit-cache\.h"/);
    assert.match(pearlCpp, /!defined\(MOM_EMBEDDED_CUDA_IMAGES\)[\s\S]*?#include "\.\.\/jit-cache\.h"/);
  });

  test("embedded accelerators retain the SM80 gate and optional SYCL fallback", () => {
    const cn = fs.readFileSync(path.join(root, "sycl/cn_gpu/cuda_jit.inc"), "utf8");
    const pearl = fs.readFileSync(path.join(root, "sycl/pearlhash/cuda_search.inc"), "utf8");
    const supports = extractCppBlock(pearl,
      "  static bool supports(sycl::queue& q, std::string* const reason = nullptr)");
    assert.match(supports, /if \(major < 8\)\s*return unavailable\(/);
    const pearlEnsure = extractCppBlock(pearl,
      "  bool ensure(sycl::queue& q, const Buffers& b, int m, int n, int k, int rank)");
    assert.ok(pearlEnsure.indexOf("if (!supports(q)") < pearlEnsure.indexOf("select_embedded("));
    const cnImages = cn.slice(cn.indexOf("#if defined(MOM_EMBEDDED_CUDA_IMAGES)"));
    assert.ok(cnImages.indexOf("if (major < 8)") >= 0);
    assert.ok(cnImages.indexOf("if (major < 8)") < cnImages.indexOf("select_embedded("));
    // Optional image rejection must clean up and leave the complete SYCL path usable.
    for (const source of [cn, pearlEnsure]) {
      assert.match(source, /catch \(const std::string& error\) \{\s*release\(\);/);
      assert.match(source, /catch \(const std::exception& error\) \{\s*release\(\);/);
      assert.match(source, /return enabled;/);
    }
  });
});
