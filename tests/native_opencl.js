"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const {spawnSync} = require("node:child_process");
const {openclInputs, pins} = require("../scripts/probe-static-dpcpp");
const read = (/** @type {string} */ name) => fs.readFileSync(path.join(__dirname, "..", name), "utf8");
const patch = read("runtime/dpcpp-static-opencl.patch");

test.describe("Static OpenCL original-source admission", () => {
  // These are CMake inputs: forward slashes are required on either build host.
  const env = {MOM_AOT_DPCPP_OPENCL_SOURCE: path.resolve("/loader").replace(/\\/g, "/"),
    MOM_AOT_DPCPP_OPENCL_HEADERS: path.resolve("/headers").replace(/\\/g, "/")};
  test("OpenCL is not enabled implicitly", () => {assert.equal(openclInputs({}), null);});
  test("both pinned source roots are explicit", () => {
    assert.deepEqual(openclInputs(env), {openclLoader: env.MOM_AOT_DPCPP_OPENCL_SOURCE, openclHeaders: env.MOM_AOT_DPCPP_OPENCL_HEADERS});
  });
  for (const key of Object.keys(env)) {
    test(`missing ${key} cannot select an installed shared library`, () => {assert.throws(() => openclInputs({...env, [key]: undefined}), /both absolute/);});
    for (const value of ["", "relative", "/a;other", "/a\n", '/a"other', "/a$other"]) {
      test(`${key} rejects ${JSON.stringify(value)} before CMake access`, () => {assert.throws(() => openclInputs({...env, [key]: value}), /both absolute/);});
    }
  }
  test("the original loader and headers are revision-pinned independently", () => {
    for (const pin of [pins.openclLoader, pins.openclHeaders]) {assert.match(pin, /^[0-9a-f]{40}$/);}
    assert.notEqual(pins.openclLoader, pins.openclHeaders);
  });
});

test.describe("Static OpenCL linked registry and lifetime", {skip: process.platform === "win32"}, () => {
  let root = "";
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-opencl-registry-"));
    // Compile the actual patch's complete new storage/lifetime fragment, retaining context
    // lines and omitting removed ones. A tiny original-registry stand-in needs no GPU SDK.
    const file = patch.split(/(?=^diff --git )/m).find((part) => part.includes("b/unified-runtime/source/adapters/opencl/ocl_dynamic_lib.cpp\n"));
    assert.ok(file);
    const source = file.split("\n").filter((line) => /^[+ ]/.test(line) && !line.startsWith("+++"))
      .map((line) => line.slice(1)).join("\n");
    const start = source.indexOf("namespace ocl {"), end = source.indexOf("static void *OCLLibHandle");
    assert.ok(start >= 0 && end > start);
    fs.writeFileSync(path.join(root, "registry.inc"), source.slice(start, end) + "\n#endif\n}\n");
    fs.writeFileSync(path.join(root, "ocl_functions.def"), "OCL_FUNC(required)\nOCL_OPTIONAL_FUNC(optional)\n");
    fs.writeFileSync(path.join(root, "probe.cpp"), "#include <cstdint>\n#include <cassert>\n" +
      "static uint64_t required(unsigned a,uint64_t b){return a+b;}\n" +
      "static const char* optional(const char* p){return p;}\n" +
      '#define UR_STATIC_OPENCL_ICD\n#include "registry.inc"\n' +
      "int main(){for(unsigned i=0;i<16;++i){assert(ocl::loadOCLLibrary());" +
      "assert(ocl::required_ptr==&required&&ocl::optional_ptr==&optional);" +
      "assert(ocl::required_ptr(7,0x100000003ULL)==0x10000000aULL);" +
      'const char* value="identity";assert(ocl::optional_ptr(value)==value);' +
      "ocl::unloadOCLLibrary();assert(ocl::required_ptr==&required&&ocl::optional_ptr==&optional);}}\n");
    const built = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      path.join(root, "probe.cpp"), "-o", path.join(root, "probe")], {encoding: "utf8", timeout: 30000});
    assert.equal(built.status, 0, built.error?.message || built.stderr);
    fs.mkdirSync(path.join(root, "CL"));
    fs.writeFileSync(path.join(root, "CL/cl.h"), "#pragma once\ntypedef unsigned cl_uint;\ntypedef struct _cl_platform_id* cl_platform_id;\nextern \"C\" int clGetPlatformIDs(cl_uint,cl_platform_id*,cl_uint*);\n");
    fs.writeFileSync(path.join(root, "CL/cl_ext.h"), "#define CL_PLATFORM_NOT_FOUND_KHR -1001\n");
    const control = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-DMOM_NATIVE_OPENCL_LINKED",
      "-I", root, path.join(__dirname, "logic/opencl_binding_control.cpp"), "-o", path.join(root, "binding-control")], {encoding: "utf8", timeout: 30000});
    assert.equal(control.status, 0, control.error?.message || control.stderr);
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  test("the actual linked storage uses original SDK types, forwards wide values and outlives adapter cleanup", () => {
    const result = spawnSync(path.join(root, "probe"), [], {encoding: "utf8", timeout: 5000});
    assert.equal(result.status, 0, result.error?.message || result.stderr);assert.equal(result.stdout, "");
  });
  test("native pointer definitions reuse the entire original required/optional registry", () => {
    assert.match(patch, /decltype\(::name\) \*name##_ptr = &::name/);
    assert.match(patch, /OCL_OPTIONAL_FUNC\(name\) OCL_FUNC\(name\)/);
    assert.match(patch, /#include "ocl_functions\.def"/);
  });
  test("normal upstream dynamic loading remains behind an off-by-default opt-in", () => {
    assert.match(patch, /option\(UR_STATIC_OPENCL_ICD [^\n]* OFF\)/);
    assert.match(patch, /#else\n static void \*OCLLibHandle = nullptr/);
  });
  test("native mode links the original ICD target, not an application library path", () => {
    assert.match(patch, /target_link_libraries\(\$\{TARGET_NAME\} PRIVATE OpenCL\)/);
    assert.match(patch, /requires the static adapter/);
  });
  /** @type {Array<[string, string]>} */
  const bindingCases = [["bindings", "binding-only mode never queries an installed driver"],
    ["negative", "an unbound original probe fails before any driver query"], ["platforms", "the ordinary probe retains its platform checks"]];
  bindingCases.forEach(([mode, name]) => {
    test(name, () => {
      const result = spawnSync(path.join(root, "binding-control"), [mode], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, result.stderr);
    });
  });
});

test.describe("Static OpenCL full proof boundaries", () => {
  const cmake = read("tests/native/dpcpp-static/CMakeLists.txt"), fixture = read("tests/reference/native_dpcpp_static.cpp");
  test("the upstream ICD is explicitly static and optional plugin layers are disabled", () => {
    for (const option of ["OPENCL_ICD_LOADER_BUILD_SHARED_LIBS", "ENABLE_OPENCL_LAYERS", "ENABLE_OPENCL_LAYERINFO"]) {
      assert.match(cmake, new RegExp(`set\\(${option} OFF CACHE BOOL`));
    }
  });
  test("absence of devices cannot hide an unbound static entrypoint", () => {
    assert.match(fixture, /ocl::clGetPlatformIDs_ptr != &::clGetPlatformIDs/);
    assert.match(fixture, /ocl::unloadOCLLibrary\(\)/);
    assert.match(fixture, /opencl\.Adapter\.pfnGet\(0, nullptr, &adapters\)/);
  });
  test("a loader spy exercises the original ICD path without loading a GPU driver", () => {
    assert.match(cmake, /add_executable\(native-opencl-driver/);
    assert.match(cmake, /target_link_libraries\(native-opencl-driver PRIVATE OpenCL/);
    assert.match(cmake, /MOM_TEST_KHRONOS_ICD/);
    const spy = read("tests/native/driver_library.cpp");
    assert.match(spy, /void\* result = khrIcdOsLibraryLoad\(name\)/);
    assert.match(spy, /__real_dlopen\(const char\* name, int flags\)/);
  });
  test("the same original SYCL host and explicit original ICD sources remain in the graph", () => {
    assert.match(cmake, /add_subdirectory\("\$\{MOM_DPCPP_OPENCL_SOURCE\}" opencl-icd-loader\)/);
    assert.match(cmake, /\$<TARGET_OBJECTS:sycl_object>/);
  });
  test("an unused OpenCL provider stub cannot replace Level Zero's real error translator", () => {
    assert.match(patch, /#ifndef UR_STATIC_OPENCL_ICD\n\+\/\/ OpenCL does not use UMF providers yet/);
    assert.match(fixture, /umf::getProviderNativeError\("Level Zero", UR_RESULT_ERROR_OUT_OF_HOST_MEMORY\)/);
    assert.doesNotMatch(patch, /^diff --git .*adapters\/level_zero\//m);
  });
  test("build inputs cannot implicitly run GPU work or enable OpenCL in normal commands", () => {
    for (const name of ["scripts/dev.js", "scripts/build-release.js"]) {assert.doesNotMatch(read(name), /probe-static-dpcpp/);}
    assert.match(read("scripts/probe-static-dpcpp.js"), /MOM_AOT_BUILD_ONLY/);
  });
  test("Windows validates the actual adapter archive, not a second copy of its binding source", () => {
    const windows = cmake.slice(cmake.indexOf("add_custom_target(mom-static-dpcpp-libs"), cmake.indexOf("# Linux fixtures below"));
    assert.match(windows, /target_link_libraries\(native-opencl-binding-static PRIVATE ur_adapter_opencl OpenCL\)/);
    assert.match(windows, /add_dependencies\(mom-static-dpcpp-libs native-opencl-binding-static\)/);
    assert.doesNotMatch(windows, /ocl_dynamic_lib\.cpp/);
  });
  test("the Windows archive control checks pointers without querying a GPU driver", () => {
    const source = read("tests/reference/native_opencl_binding.cpp");
    assert.match(source, /strcmp\(argv\[1\], "--bindings-only"\)/);
    assert.match(source, /\(required \|\| bindingsOnly\) && \(!loaded \|\| !bound\)/);
    assert.match(source, /if \(!bindingsOnly\) \{\s+cl_uint count/);
  });
  test("the Windows static ICD omits only its separate DLL version resource", () => {
    assert.match(cmake, /get_target_property\(opencl_sources OpenCL SOURCES\)/);
    assert.match(cmake, /list\(FILTER opencl_sources EXCLUDE REGEX "\(\^\|\/\)loader\/windows\/OpenCL\\\\\.rc\$"\)/);
    assert.match(cmake, /set_property\(TARGET OpenCL PROPERTY SOURCES "\$\{opencl_sources\}"\)/);
  });
  test("every present Windows adapter shares the static application CRT", () => {
    assert.match(cmake, /foreach\(target ur_loader ur_common ur_adapter_opencl ur_adapter_cuda ur_adapter_level_zero ur_adapter_level_zero_v2\)/);
    assert.match(cmake, /if\(TARGET \$\{target\}\)\s+set_property\(TARGET \$\{target\} PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded\)/);
  });
});
