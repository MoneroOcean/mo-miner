"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const {spawnSync} = require("node:child_process");
const {driverImports, driverSignatures, forwardingSource} = require("../scripts/native-cuda-adapter");
const {cudaSdkInput, adapterInputs, sourceJitInput} = require("../scripts/probe-static-dpcpp");

test.describe("Static DPC++ CUDA build admission and original graph", () => {
  const read = (/** @type {string} */ file) => fs.readFileSync(path.join(__dirname, "..", file), "utf8");
  const cmake = read("tests/native/dpcpp-static/CMakeLists.txt"), patch = read("runtime/dpcpp-static-adapters.patch");
  test("CUDA stays disabled unless its SDK is explicitly selected", () => {
    assert.equal(cudaSdkInput({}), null);
    assert.equal(cudaSdkInput({MOM_AOT_DPCPP_CUDA_SDK: "/build/cuda"}), "/build/cuda");
  });
  test("unsafe SDK values cannot become CMake expressions or lists", () => {
    for (const value of ["", "relative", "/sdk;other", "/sdk/$value", '/sdk/"quoted"', "/sdk/line\nbreak", "/sdk/back\\slash"]) {
      assert.throws(() => cudaSdkInput({MOM_AOT_DPCPP_CUDA_SDK: value}), /absolute CUDA SDK/);
    }
  });
  test("CUDA does not implicitly select Level Zero or source JIT", () => {
    const env = {MOM_AOT_DPCPP_CUDA_SDK: "/build/cuda"};
    assert.equal(adapterInputs(env), null); assert.equal(sourceJitInput(env), null);
  });
  test("the original adapter archive and full dispatch table are reused", () => {
    assert.match(cmake, /\$<TARGET_FILE:ur_adapter_cuda>/);
    assert.match(read("tests/native/dpcpp-static/cuda-factory.cpp"), /urAllAddrTable\(UR_API_VERSION_CURRENT, table\)/);
    assert.doesNotMatch(read("tests/native/dpcpp-static/cuda-factory.cpp"), /pfn\w+\s*=/);
  });
  test("private adapter symbols retain upstream's transitive static dependencies", () => {
    assert.match(cmake, /get_target_property\(cuda_dependencies ur_adapter_cuda LINK_LIBRARIES\)/);
    assert.match(cmake, /target_link_libraries\(mom-ur-cuda-static INTERFACE "\$\{cuda_object\}" \$\{cuda_dependencies\}\)/);
    // Imported targets must be visible where the copied interface is evaluated, not only in
    // upstream's nested directory. Otherwise configuration fails before the link can run.
    const parent = cmake.indexOf("find_package(Threads REQUIRED)");
    assert.ok(parent >= 0 && parent < cmake.indexOf("/unified-runtime\" ur-runtime)"));
  });
  test("generated code tracks both SDK headers and its implementation inputs", () => {
    for (const input of ["include/cuda.h", "include/nvml.h", "runtime/system-driver.h", "scripts/native-compiler-isolation.js", "scripts/native-audit.js", "scripts/build-source-inputs.js", "cuda-factory.cpp"]) {
      assert.ok(cmake.includes(input));
    }
    assert.match(cmake, /add_dependencies\(mom-ur-cuda-static mom-ur-cuda-isolation\)/);
  });
  test("CUDA-only allocation admits hwloc without admitting Level Zero", () => {
    assert.equal(adapterInputs({MOM_AOT_DPCPP_CUDA_SDK: "/cuda", MOM_AOT_DPCPP_HWLOC_SOURCE: "/hwloc"}), null);
    assert.throws(() => adapterInputs({MOM_AOT_DPCPP_HWLOC_SOURCE: "/hwloc"}), /explicitly/);
  });
  test("both adapters reuse static hwloc instead of a system application library", () => {
    assert.match(cmake, /if\(MOM_DPCPP_LEVEL_ZERO_SOURCE OR MOM_DPCPP_CUDA_SDK\)[\s\S]*set\(UMF_LINK_HWLOC_STATICALLY ON/);
    assert.match(cmake, /FETCHCONTENT_SOURCE_DIR_HWLOC_TARG "\$\{MOM_DPCPP_HWLOC\}"/);
  });
  test("the normal shared adapter still links its unchanged driver targets", () => {
    assert.match(patch, /\+else\(\)\n\+ {2}target_link_libraries\(\$\{TARGET_NAME\} PRIVATE cudadrv CUDA::nvml\)/);
    assert.match(patch, /\+ {2}set\(ADAPTER_LIB_TYPE STATIC\)/);
  });
  test("the loader registers only the opaque private factory", () => {
    assert.match(patch, /mom_ur_cuda_factory\(&cuda\.dditable\)/);
    assert.match(patch, /if \(result != UR_RESULT_SUCCESS\)\n\+ {8}return result/);
    assert.match(patch, /markAdapterAsStaticallyLoaded\("ur_adapter_cuda"\)/);
  });
  test("no-driver admission explicitly rejects actual devices", () => {
    const proof = read("tests/reference/native_dpcpp_static.cpp");
    assert.match(proof, /platform\.get_backend\(\) != sycl::backend::ext_oneapi_cuda \|\| !platform\.get_devices\(\)\.empty\(\)/);
    assert.match(proof, /cuda_count != 1/);
  });
  test("driver forwarding and isolation are build-only, not Engineering imports", () => {
    for (const file of ["scripts/dev.js", "scripts/build-release.js"]) {
      assert.doesNotMatch(read(file), /native-cuda-adapter|probe-static-dpcpp/);
    }
    const helper = read("scripts/native-cuda-adapter.js");
    assert.match(helper, /entryPoints: \["mom_ur_cuda_factory"\]/);
    assert.doesNotMatch(helper, /nvidia-smi|\/dev\/nvidia|dlopen\(/);
  });
});

test.describe("Native CUDA adapter generated driver boundary", {skip: process.platform !== "linux"}, () => {
  let root = "";
  const entries = [{name: "cuInit", count: 1}, {name: "cuMemAlloc_v2", count: 2}, {name: "cuCtxGetCurrent", count: 1},
    {name: "cuGetErrorString", count: 2}, {name: "nvmlInit_v2", count: 0}, {name: "nvmlErrorString", count: 1}, {name: "nvmlDeviceGetFanSpeed", count: 2}];
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cuda-forwarding-"));
    fs.writeFileSync(path.join(root, "cuda.h"), "#pragma once\n#include <cstddef>\n#define CUDAAPI\n" +
      "enum CUresult {CUDA_SUCCESS,CUDA_ERROR_NOT_INITIALIZED,CUDA_ERROR_NO_DEVICE,CUDA_ERROR_INVALID_VALUE};\n" +
      "using CUdeviceptr=unsigned long long; using CUcontext=struct Context*;\nextern \"C\" {\n" +
      "CUresult cuInit(unsigned); CUresult cuMemAlloc_v2(CUdeviceptr*,size_t); CUresult cuCtxGetCurrent(CUcontext*); CUresult cuGetErrorString(CUresult,const char**);\n}\n");
    fs.writeFileSync(path.join(root, "nvml.h"), "#pragma once\nenum nvmlReturn_t {NVML_SUCCESS,NVML_ERROR_UNINITIALIZED}; using nvmlDevice_t=struct NvmlDevice*;\n" +
      "extern \"C\" { nvmlReturn_t nvmlInit_v2(); const char* nvmlErrorString(nvmlReturn_t); nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t,unsigned*); }\n");
    fs.writeFileSync(path.join(root, "driver-forwarding.cpp"), forwardingSource(entries));
    const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", root,
      "-I", path.join(__dirname, "../runtime"), path.join(__dirname, "native/cuda_driver_forwarding.cpp"), "-o", path.join(root, "forward")],
    {encoding: "utf8", timeout: 30000});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  /** @param {number} mode */
  function execute(mode) {
    const result = spawnSync(path.join(root, "forward"), [String(mode)], {encoding: "utf8", timeout: 5000});
    assert.equal(result.status, 0, result.stderr); assert.match(result.stdout, /ABI, absent\/partial drivers, concurrency and cached lifetime passed/);
  }
  for (const [mode, label] of [[0, "available driver"], [1, "absent drivers"], [2, "missing optional symbol"]]) {
    test(`${label}: exact ABI, typed arguments, cached module and symbol lifetime`, () => {execute(Number(mode));});
  }
  test("only exact imported driver functions are selected from an archive", () => {
    assert.deepEqual(driverImports("file.o:\ncuInit U\ncuInit U\ncuMemAlloc_v2 U\nnvmlInit_v2 U\nstd_func U\ncuInit T 0 12\nnotcuInit U\n"),
      ["cuInit", "cuMemAlloc_v2", "nvmlInit_v2"]);
  });
  test("parameter counts come from original SDK FunctionDecl nodes", () => {
    assert.deepEqual(driverSignatures({inner: [{kind: "FunctionDecl", name: "cuInit", inner: [{kind: "ParmVarDecl"}, {kind: "UnusedAttr"}]}]}, ["cuInit"]), [{name: "cuInit", count: 1}]);
  });
  test("missing declarations cannot silently create an untyped stub", () => {
    assert.throws(() => driverSignatures({inner: []}, ["cuMissing"]), /absent from the selected SDK/);
  });
  test("variadic and ambiguous declarations fail before code generation", () => {
    assert.throws(() => driverSignatures({kind: "FunctionDecl", name: "cuInit", variadic: true}, ["cuInit"]), /Variadic/);
    assert.throws(() => driverSignatures({inner: [{kind: "FunctionDecl", name: "cuInit"},
      {kind: "FunctionDecl", name: "cuInit", inner: [{kind: "ParmVarDecl"}]}]}, ["cuInit"]), /Ambiguous/);
  });
  test("malformed identifiers and unbounded arities cannot enter generated source", () => {
    for (const entry of [{name: "cuBad;evil()", count: 0}, {name: "driverCall", count: 0}, {name: "cuInit", count: -1}, {name: "cuInit", count: 1.5}, {name: "cuInit", count: 33}]) {
      assert.throws(() => forwardingSource([entry]), /Invalid driver forwarding/);
    }
  });
  test("forwarders always use exact header function types and fixed driver identities", () => {
    const source = forwardingSource(entries);
    for (const {name} of entries) {assert.ok(source.includes(`Signature<decltype(&::${name})>`));}
    assert.doesNotMatch(source, /getenv|RTLD_DEFAULT|reinterpret_cast<.*\(\*\)/);
  });
  test("module lookup on Windows is system-directory-only", () => {
    assert.match(fs.readFileSync(path.join(__dirname, "../runtime/system-driver.h"), "utf8"), /LoadLibraryExA\([\s\S]*LOAD_LIBRARY_SEARCH_SYSTEM32\)/);
  });
  test("importing or invoking without build-only admission cannot load a driver", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/native-cuda-adapter.js")],
      {encoding: "utf8", timeout: 5000, env: {...process.env, MOM_AOT_BUILD_ONLY: "0", PATH: "/unavailable"}});
    assert.equal(result.status, 1); assert.match(result.stderr, /MOM_AOT_BUILD_ONLY=1/); assert.doesNotMatch(result.stderr, /ENOENT|spawn/);
  });
});
