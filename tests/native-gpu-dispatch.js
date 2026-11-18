"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const {spawnSync} = require("node:child_process");
const {gpuTable, gpuForwards} = require("../scripts/native-gpu-table");
const {gpuProfile} = require("../scripts/native-gpu-build");

test.describe("Native compiler graph dispatch and generated forwards", {skip: process.platform !== "linux"}, () => {
  let root = "";
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-gpu-dispatch-")); fs.mkdirSync(path.join(root, "sycl"));
    const header = "#pragma once\n#include <cstdint>\n#define MOM_SYCL_API\nMOM_SYCL_API uint64_t work(unsigned a, uint64_t b);\nMOM_SYCL_API void sycl_cleanup() noexcept;\n";
    fs.writeFileSync(path.join(root, "sycl/lib.h"), header);
    fs.writeFileSync(path.join(root, "gpu-table.h"), gpuTable(header, "mom_native_gpu_dpcpp").header);
    fs.writeFileSync(path.join(root, "forwards.cpp"), gpuForwards(header));
    for (const lane of ["both", "acpp", "windows"]) {
      const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", root,
        "-I", path.join(__dirname, ".."), '-DMOM_NATIVE_GPU_DEFAULT="acpp-hip"', "-DMOM_NATIVE_GPU_ACPP_HIP",
        ...(lane === "windows" ? ["-D_WIN32"] : []),
        ...(lane !== "acpp" ? ["-DMOM_NATIVE_GPU_DPCPP", "-DMOM_NATIVE_GPU_ONEAPI", "-DMOM_NATIVE_GPU_DPCPP_OPENCL", "-DMOM_NATIVE_GPU_ACPP_CUDA"] : []), path.join(__dirname, "../runtime/gpu-dispatch.cpp"),
        path.join(root, "forwards.cpp"), path.join(__dirname, "native/gpu-dispatch.cpp"), "-o", path.join(root, lane)],
      {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.error?.message || result.stderr);
    }
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  /** @param {string | undefined} key @param {string} [mode] @param {string} [lane] @param {NodeJS.ProcessEnv} [extra] */
  function run(key, mode = "work", lane = "both", extra = {}) {
    const result = spawnSync(path.join(root, lane), [mode], {encoding: "utf8", timeout: 10000,
      env: {...process.env, MOM_SYCL_COMPILER: key, MOM_TEST_NULL_FACTORY: undefined, ...extra}});
    assert.equal(result.status, 0, result.error?.message || result.stderr); return result.stdout;
  }
  test("a build-selected default works without an addon or compiler environment", () => {assert.match(run(undefined), /immutable route/);});
  test("an empty key uses only the explicit build default", () => {assert.match(run(""), /immutable route/);});
  for (const key of ["dpcpp", "oneapi", "dpcpp-opencl", "acpp-cuda", "acpp-hip"]) {
    test(`${key} selects its linked graph exactly once across concurrent entry`, () => {assert.match(run(key), /first-entry concurrency/);});
  }
  test("unused cleanup cannot initialize a compiler, even with an invalid key", () => {assert.equal(run("bad", "unused"), "");});
  test("Windows addon cleanup does not load an unused GPU DLL and still cleans a loaded one", () => {
    const source = fs.readFileSync(path.join(__dirname, "../native/core.cpp"), "utf8"),
      cleanup = source.slice(source.indexOf("static void cleanup_sycl_runtime"), source.indexOf("static napi_value init_module")),
      output = path.join(root, "addon-cleanup");
    assert(cleanup.startsWith("static void cleanup_sycl_runtime"));
    const fixture = `#include <cwchar>
      bool loaded = false; int calls = 0;
      void* GetModuleHandleW(const wchar_t* name) {
        return loaded && std::wcscmp(name, L"sycl.dll") == 0 ? reinterpret_cast<void*>(1) : nullptr;
      }
      void sycl_cleanup() noexcept { ++calls; }
      ${cleanup}
      int main() {
        cleanup_sycl_runtime(nullptr); if (calls != 0) return 1;
        loaded = true; cleanup_sycl_runtime(nullptr); return calls == 1 ? 0 : 2;
      }`;
    const compiled = spawnSync(process.env["CXX"] || "c++", ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
      "-D_WIN32", "-x", "c++", "-", "-o", output], {input: fixture, encoding: "utf8", timeout: 30000});
    assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
    const result = spawnSync(output, [], {encoding: "utf8", timeout: 10000});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
  });
  test("HIP rejects unusable scratch before initializing its factory or cleanup", () => {
    for (const temporary of [path.join(root, "missing"), path.join(root, "both"), "", "a".repeat(4097)]) {
      assert.equal(run("acpp-hip", "scratch-error", "both", {TMPDIR: temporary}), "");
    }
    assert.equal(run("acpp-hip", "scratch-error", "both", {TMPDIR: undefined, TMP: path.join(root, "missing")}), "");
  });
  test("a scratch failure is retryable and a successful probe leaves no files behind", () => {
    const before = fs.readdirSync(root).sort();
    assert.match(run("acpp-hip", "scratch-retry", "both", {TMPDIR: path.join(root, "missing"), MOM_TEST_WRITABLE_TMP: root}), /immutable route/);
    assert.deepEqual(fs.readdirSync(root).sort(), before);
  });
  test("an unusable HIP scratch directory cannot affect other graphs or unused cleanup", () => {
    const extra = {TMPDIR: path.join(root, "missing")};
    assert.match(run("oneapi", "work", "both", extra), /immutable route/);
    assert.equal(run("acpp-hip", "unused", "both", extra), "");
  });
  for (const key of ["unknown", "/sdk/mom.node"]) {
    test(`unlinked ${key} fails explicitly without fallback or library lookup`, () => {assert.equal(run(key, "reject"), "");});
  }
  test("a null linked factory cannot turn into a silent alternate compiler", () => {
    assert.equal(run("dpcpp", "reject", "both", {MOM_TEST_NULL_FACTORY: "1"}), "");
  });
  test("a compiler absent from this build is rejected even if another graph is available", () => {assert.equal(run("dpcpp", "reject", "acpp"), "");});
  test("Intel cannot silently use a graph built without its fast paths", () => {assert.equal(run("oneapi", "reject", "acpp"), "");});
  test("portable OpenCL cannot silently use a vendor-specific image", () => {assert.equal(run("dpcpp-opencl", "reject", "acpp"), "");});
  test("HIP-only images cannot silently stand in for a CUDA compiler key", () => {assert.equal(run("acpp-cuda", "reject", "acpp"), "");});
  test("failed first initialization leaves selection retryable and cleanup harmless", () => {assert.match(run("unknown", "retry"), /owned cleanup/);});
  test("eager images with identical kernel IDs fail before main, deferred profiles do not", () => {
    const before = spawnSync(path.join(root, "both"), ["work"], {encoding: "utf8", timeout: 10000,
      env: {...process.env, MOM_SYCL_COMPILER: "oneapi", MOM_TEST_EAGER_IMAGES: "1"}});
    assert.equal(before.status, 86, before.stderr);
    assert.match(run("oneapi"), /immutable route/);
  });
  test("Windows retires deferred images before the lazy SYCL runtime shutdown", () => {
    assert.match(run("oneapi", "work", "windows"), /immutable route/);
  });
  test("Windows early cleanup preparation does not discover an unused GPU", () => {
    assert.equal(run("bad", "unused", "windows"), "");
  });
  test("Windows late shutdown never removes a partially registered image", () => {
    assert.equal(run("oneapi", "registration-fault", "windows", {MOM_TEST_REGISTRATION_FAIL: "1"}), "");
  });
  test("a missing image descriptor fails locally without selecting another compiler", () => {
    assert.equal(run("dpcpp", "missing-images", "both", {MOM_TEST_OMIT_IMAGES: "1"}), "");
  });
  test("a descriptor shared across compiler profiles fails before dispatch", () => {
    const result = spawnSync(path.join(root, "both"), ["unused"], {encoding: "utf8", timeout: 10000,
      env: {...process.env, MOM_TEST_ALIAS_IMAGES: "1"}});
    assert.equal(result.status, 88, result.stderr);
  });
  test("a partial registration failure cannot retry the same runtime bookkeeping", () => {
    assert.equal(run("oneapi", "registration-fault", "both", {MOM_TEST_REGISTRATION_FAIL: "1"}), "");
  });
  test("all maintained entrypoints are generated as direct typed forwards", () => {
    const header = fs.readFileSync(path.join(__dirname, "../sycl/lib.h"), "utf8"), source = gpuForwards(header);
    const entries = gpuTable(header, "mom_native_gpu_dpcpp").entries;
    for (const {name} of entries.filter((entry) => entry.name !== "sycl_cleanup")) {
      assert.ok(source.includes(`selectedGpu().${name}(`));
    }
    assert.match(source, /void sycl_cleanup\(\) noexcept \{\n {2}mom::runtime::cleanupGpu\(\)/);
    assert.doesNotMatch(source, /switch|dlopen|LoadLibrary|getenv|fopen|nullptr/);
  });
  test("CUDA and HIP retain the original distinct C30 geometry, without requiring a GPU SDK", () => {
    for (const [profile, bucket, chains] of [["cuda", 13, 16], ["hip", 12, 4]]) {
      const definitions = gpuProfile(String(profile)).definitions;
      const result = spawnSync(process.env["CXX"] || "c++", ["-x", "c++", "-E", "-P", "-DMOM_SYCL_ADAPTIVECPP", ...definitions,
        "-I", path.join(__dirname, ".."), "-"], {input: '#include "sycl/c30/c30-kernels.inc"\n', encoding: "utf8", timeout: 10000});
      assert.equal(result.status, 0, result.error?.message || result.stderr);
      assert.match(result.stdout, new RegExp(`C30_BUCKET_BITS = ${bucket}`));
      assert.match(result.stdout, new RegExp(`C30_SEED_BLOCKS_PER_ITEM = ${chains}u`));
    }
  });
});
