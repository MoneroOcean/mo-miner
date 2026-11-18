"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {windowsStaticSdkPlan} = require("../scripts/build-native-windows-sdk");
const directory = path.join(__dirname, "native/dpcpp-static");
const cmake = fs.readFileSync(path.join(directory, "CMakeLists.txt"), "utf8");
const commonPatch = fs.readFileSync(path.join(__dirname, "../runtime/dpcpp-static-adapters.patch"), "utf8");
const patch = fs.readFileSync(path.join(__dirname, "../runtime/dpcpp-static-windows.patch"), "utf8");
const pinnedSource = process.env["MOM_PINNED_WINDOWS_SOURCE"];
const levelZeroPatch = fs.readFileSync(path.join(__dirname, "../runtime/level-zero-static-complete.patch"), "utf8");
const levelZeroWindowsPatch = fs.readFileSync(path.join(__dirname, "../runtime/level-zero-static-windows.patch"), "utf8");
const registryName = "unified-runtime/source/loader/ur_adapter_registry.hpp";
const loaderName = "unified-runtime/source/loader/ur_loader.cpp";
const pinnedFiles = [
  "unified-runtime/cmake/helpers.cmake",
  "sycl/source/detail/global_handler.cpp",
  "sycl/source/detail/windows_ur.cpp",
  "unified-runtime/source/adapters/CMakeLists.txt",
  "unified-runtime/source/adapters/cuda/CMakeLists.txt",
  "unified-runtime/source/adapters/level_zero/CMakeLists.txt",
  "unified-runtime/source/loader/CMakeLists.txt",
  registryName,
  loaderName,
];

test.describe("Windows static SYCL compiler tool selection", () => {
  function descriptor() {
    return {version: 1, tools: {clangCl: "/tools/host", lldLink: "/tools/lld-link", llvmReadobj: "/tools/llvm-readobj",
      llvmNm: "/tools/llvm-nm", shermaot: "/tools/shermes", clangResourceInclude: "/tools/include"},
    hermes: {source: "/hermes/source", build: "/hermes/build", shermaot: "/tools/shermes"},
    windowsSdk: {overlay: "/os/vfs.json", sourceRoot: "/os/sdk", includeRoots: ["/os/include"], libraryRoots: ["/os/lib"],
      msvcLibraries: ["/os/lib/libcmt.lib"], ucrtLibraries: ["/os/lib/libucrt.lib"], importLibraries: ["/os/lib/kernel32.lib"],
      allowedImports: ["kernel32.dll"]},
    libraries: {sourceInclude: "/io/include", ssl: "/io/ssl.lib", crypto: "/io/crypto.lib", libuv: "/io/uv.lib", zlib: "/io/z.lib"}};
  }
  test("Intel direct-LLD builds do not require the unused manifest tool", {skip: process.platform !== "linux"}, () => {
    const plan = windowsStaticSdkPlan({}, descriptor(), "/dpcpp", "/cuda", "/output", 2, undefined, true);
    assert.equal(Object.hasOwn(plan.compilerTools, "CMAKE_MT"), false);
    assert.doesNotMatch(plan.toolchain, /llvm-mt|CMAKE_MT/);
    assert.equal(plan.compilerTools.CMAKE_LINKER, "/tools/lld-link");
    assert.match(plan.ruleOverrideText, /<CMAKE_LINKER> \/NOLOGO/);
  });
  test("the default Clang plan retains its platform manifest tool", {skip: process.platform !== "linux"}, () => {
    const plan = windowsStaticSdkPlan({}, descriptor(), "/dpcpp", "/cuda", "/output", 2);
    assert.equal(plan.compilerTools.CMAKE_MT, "/tools/llvm-mt");
    assert.match(plan.toolchain, /set\(CMAKE_MT /);
  });
});
/** @param {string} name @returns {string} */
function patchBlock(name) {
  const marker = `diff --git a/${name} b/${name}\n`, start = patch.indexOf(marker);
  assert.ok(start >= 0, `Missing Windows patch block: ${name}`);
  const end = patch.indexOf("\ndiff --git ", start + marker.length);
  return patch.slice(start, end < 0 ? patch.length : end);
}
/** @param {import("node:test").TestContext} t */
function temporary(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-static-sycl-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true})); return root;
}
/** @param {import("node:test").TestContext} t */
function applyPinnedSource(t) {
  if (!pinnedSource) {
    t.skip("Set MOM_PINNED_WINDOWS_SOURCE explicitly for pinned-source applicability");
    return null;
  }
  const root = temporary(t);
  for (const name of pinnedFiles) {
    const source = path.join(pinnedSource, name), destination = path.join(root, name);
    assert.ok(fs.existsSync(source), `Missing pinned source file: ${source}`);
    fs.mkdirSync(path.dirname(destination), {recursive: true});
    fs.copyFileSync(source, destination);
  }
  const registryPath = path.join(root, registryName), loaderPath = path.join(root, loaderName);
  const originalRegistry = fs.readFileSync(registryPath, "utf8");
  const originalLoader = fs.readFileSync(loaderPath, "utf8");
  assert.doesNotMatch(originalRegistry, /MOM_NATIVE_STATIC_ADAPTERS|isAdapterEnabled/);
  assert.doesNotMatch(originalLoader, /MOM_NATIVE_STATIC_ADAPTERS|mom_ur_level_zero_v2_factory/);

  for (const input of [commonPatch, patch]) {
    const check = spawnSync("git", ["apply", "--check"], {cwd: root, input, encoding: "utf8", timeout: 30000});
    assert.equal(check.status, 0, check.stderr);
    const apply = spawnSync("git", ["apply"], {cwd: root, input, encoding: "utf8", timeout: 30000});
    assert.equal(apply.status, 0, apply.stderr);
    if (input === commonPatch) {
      assert.match(fs.readFileSync(loaderPath, "utf8"), /UR_STATIC_ADAPTER_LEVEL_ZERO_V2/);
      assert.match(fs.readFileSync(loaderPath, "utf8"), /mom_ur_level_zero_v2_factory/);
    }
  }
  return {root, registryPath, loaderPath};
}

test.describe("Static Windows SYCL linkage boundaries", {skip: process.platform !== "linux"}, () => {
  test("standard CRT lowering is scoped to the Windows SYCL host target", () => {
    assert.match(cmake, /target_compile_options\(sycl_object PRIVATE -fno-builtin\)/);
    const options = cmake.indexOf("target_compile_options(sycl_object PRIVATE -fno-builtin)");
    const windows = cmake.lastIndexOf("if(WIN32)", options);
    assert.ok(windows >= 0 && cmake.indexOf("endif()", windows) > options,
      "The policy belongs to the Windows host runtime, not all compiler targets");
    assert.equal((cmake.match(/target_compile_options\(sycl_object PRIVATE -fno-builtin\)/g) || []).length, 1);
  });
  test("the original static policy recognizes both Level Zero versions and preserves other adapters", t => {
    const root = temporary(t), block = patchBlock("unified-runtime/cmake/helpers.cmake");
    for (const after of [false, true]) {
      const lines = block.split("\n").filter(line => line.startsWith(" ") || (line.startsWith(after ? "+" : "-") && !line.startsWith(after ? "+++" : "---"))).map(line => line.slice(1));
      const file = path.join(root, after ? "after.cmake" : "before.cmake");
      const assertions = ["set(UR_STATIC_ADAPTER_L0 ON)", "set(UR_STATIC_ADAPTER_CUDA ON)",
        ...["LEVEL_ZERO", "LEVEL_ZERO_V2", "CUDA"].flatMap(name => [
          `ur_adapter_is_static(${name} actual)`, `if(NOT actual STREQUAL "${!after && name === "LEVEL_ZERO_V2" ? "FALSE" : "TRUE"}")`,
          `message(FATAL_ERROR "Incorrect static policy for ${name}")`, "endif()"]),
        "set(UR_STATIC_ADAPTER_L0 OFF)", "ur_adapter_is_static(LEVEL_ZERO_V2 actual)", "if(actual)", 'message(FATAL_ERROR "Dynamic V2 became static")', "endif()"];
      fs.writeFileSync(file, [...lines, ...assertions].join("\n"));
      const result = spawnSync("cmake", ["-P", file], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, result.stderr);
    }
  });
  test("embedded Level Zero rejects explicit app layers and never probes a stale optional DLL", t => {
    const root = temporary(t), file = path.join(root, "layers.cpp");
    const added = levelZeroPatch.split("\n").filter(line => /^\+(?!\+)/.test(line)).map(line => line.slice(1)).join("\n");
    const guard = added.match(/#if defined\(MOM_EMBEDDED_LEVEL_ZERO\)\n[\s\S]*?\n#else/)?.[0] || "#if 0\n#else";
    // Execute the real native-profile guard. The excluded shared branch models the
    // observed incompatible DLL; the embedded profile must never reach that lookup.
    fs.writeFileSync(file, `#include <cstring>
enum Result { ZE_RESULT_SUCCESS, ZE_RESULT_ERROR_UNSUPPORTED_FEATURE, StaleLayer };
int selected = -1, loads = 0;
bool getenv_tobool(const char *name) {
  const char *names[] = {"ZE_ENABLE_VALIDATION_LAYER", "ZE_ENABLE_TRACING_LAYER", "ZET_ENABLE_API_TRACING_EXP"};
  return selected >= 0 && std::strcmp(name, names[selected]) == 0;
}
struct Logger { void log_error(const char *) {} } logger;
Logger *zel_logger = &logger;
Result initialize() {
${guard}
  ++loads; return StaleLayer;
#endif
  return ZE_RESULT_SUCCESS;
}
int main() {
#ifdef MOM_EMBEDDED_LEVEL_ZERO
  if (initialize() != ZE_RESULT_SUCCESS || loads) return 1;
  for (selected = 0; selected < 3; ++selected)
    if (initialize() != ZE_RESULT_ERROR_UNSUPPORTED_FEATURE || loads) return 2;
#else
  if (initialize() != StaleLayer || loads != 1) return 3;
#endif
  return 0;
}
`);
    for (const native of [false, true]) {
      const executable = path.join(root, native ? "native" : "shared");
      const built = spawnSync("c++", ["-std=c++14", "-Wall", "-Wextra", "-Werror", ...(native ? ["-DMOM_EMBEDDED_LEVEL_ZERO=1"] : []), file, "-o", executable], {encoding: "utf8", timeout: 30000});
      assert.equal(built.status, 0, built.stderr);
      const run = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
      assert.equal(run.status, 0, native ? "Native mode must not load optional application libraries" : "Shared behavior must remain unchanged");
    }
  });

  test("the image shutdown bridge only registers the original SDK handler", () => {
    const added = patch.split("\n").filter(line => /^\+(?!\+)/.test(line)).map(line => line.slice(1)).join("\n");
    const body = added.match(/extern "C" void mom_native_sycl_prepare_shutdown\(\) \{([\s\S]*?)\}/)?.[1];
    assert.equal(body?.trim(), "GlobalHandler::registerStaticVarShutdownHandler();");
  });

  test("embedded Level Zero owns both contexts without a DLL entrypoint", t => {
    const root = temporary(t), file = path.join(root, "lifetime.cpp"), executable = path.join(root, "lifetime");
    const added = levelZeroPatch.split("\n").filter(line => /^\+(?!\+)/.test(line)).map(line => line.slice(1)).join("\n");
    const body = added.match(/#if defined\(MOM_EMBEDDED_LEVEL_ZERO\) && defined\(_WIN32\)\n([\s\S]*?)\n#endif/)?.[1] || "";
    // Compile the actual lifecycle body, not a second implementation. Its upstream context
    // types are replaced only by counters; a final observer runs after context destruction.
    fs.writeFileSync(file, `#include <memory>
#include <cstdlib>
int live = 0, destroyed = 0; bool initialized = false;
struct Context { Context() { ++live; } ~Context() { --live; ++destroyed; } };
namespace ze_lib { using context_t = Context; context_t *context = nullptr; }
namespace loader { using context_t = Context; context_t *context = nullptr; }
struct Observer { ~Observer() { if (initialized && (live || destroyed != 2 || ze_lib::context || loader::context)) std::abort(); } } observer;
namespace ze_lib { ${body} }
int main() { initialized = live == 2 && ze_lib::context && loader::context; return initialized ? 0 : 1; }
`);
    const build = spawnSync("c++", ["-std=c++14", "-Wall", "-Wextra", "-Werror", file, "-o", executable], {encoding: "utf8", timeout: 30000});
    assert.equal(build.status, 0, build.stderr);
    const run = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
    assert.equal(run.status, 0, "Both contexts must exist before main and be released after the application");
  });

  test("the original loader lookup delegates only in dynamic mode", t => {
    const root = temporary(t), file = path.join(root, "lookup.cpp"), executable = path.join(root, "lookup");
    const before = patch.match(/^-void \*getURLoaderLibrary\(\)[^\n]+/m)?.[0].slice(1);
    const lines = patch.split("\n");
    const start = lines.findIndex(line => line === "+void *getURLoaderLibrary() {");
    const end = lines.findIndex((line, index) => index > start && line === "+}");
    assert.ok(before && start >= 0 && end > start);
    const after = lines.slice(start, end + 1).map(line => line.slice(1)).join("\n");
    for (const [name, body, staticMode, expected] of [["before-static", before, true, 1], ["after-static", after, true, 0], ["after-dynamic", after, false, 0]]) {
      fs.writeFileSync(file, `${staticMode ? "#define SYCL_UR_STATIC_LOADER\n" : ""}static int calls=0,token;\n` +
        `void* getPreloadedURLib(){++calls;return &token;}\n${body}\n` +
        `int main(){auto* result=getURLoaderLibrary();return ${staticMode ? "result!=nullptr || calls!=0" : "result!=&token || calls!=1"};}\n`);
      const compile = spawnSync("c++", ["-std=c++17", "-Wall", "-Wextra", "-Werror", file, "-o", executable], {encoding: "utf8", timeout: 30000});
      assert.equal(compile.status, 0, compile.stderr);
      const run = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
      assert.equal(run.status, expected, String(name));
    }
  });

  test("the maintained common patch then Windows patch apply to the original pinned source", t => {
    const applied = applyPinnedSource(t);
    if (!applied) { return; }
    const {root, registryPath, loaderPath} = applied;
    const loaderBlock = patchBlock("unified-runtime/source/loader/ur_loader.cpp");
    for (const macro of ["LEVEL_ZERO", "LEVEL_ZERO_V2", "OPENCL", "CUDA"]) {
      assert.match(loaderBlock, new RegExp(`UR_STATIC_ADAPTER_${macro}`));
    }
    const registryBlock = patchBlock("unified-runtime/source/loader/ur_adapter_registry.hpp");
    const nativeRegistry = registryBlock.slice(registryBlock.indexOf("+#if defined(MOM_NATIVE_STATIC_ADAPTERS)"),
      registryBlock.indexOf("+#else"));
    assert.doesNotMatch(nativeRegistry, /UR_ADAPTERS_SEARCH_PATH|getEnvAdapterSearchPaths|discoverKnownAdapters/);
    assert.match(loaderBlock, /MOM_NATIVE_STATIC_ADAPTERS/);
    assert.match(loaderBlock, /UR_ADAPTERS_FORCE_LOAD|adaptersForceLoaded/);
    assert.match(loaderBlock, /restoreErrorMode\(\);\n\+\s*return/);
    assert.match(loaderBlock, /#if !defined\(MOM_NATIVE_STATIC_ADAPTERS\)/);
    const appliedLoader = fs.readFileSync(loaderPath, "utf8");
    assert.match(appliedLoader, /LibLoader::loadAdapterLibrary/);
    assert.match(appliedLoader, /mom_ur_level_zero_v2_factory/);
    assert.match(fs.readFileSync(registryPath, "utf8"), /MOM_NATIVE_STATIC_ADAPTERS/);
    const reverseCheck = spawnSync("git", ["apply", "--reverse", "--check"],
      {cwd: root, input: patch, encoding: "utf8", timeout: 30000});
    assert.equal(reverseCheck.status, 0, reverseCheck.stderr);
  });

  test("the applied context_t::init keeps native and engineering decisions distinct", t => {
    const applied = applyPinnedSource(t);
    if (!applied) { return; }
    const root = applied.root, appliedLoader = fs.readFileSync(applied.loaderPath, "utf8");
    const signature = "ur_result_t context_t::init() {", start = appliedLoader.indexOf(signature);
    assert.ok(start >= 0, "Applied loader must contain context_t::init");
    const open = appliedLoader.indexOf("{", start);
    let depth = 0, close = -1;
    for (let index = open; index < appliedLoader.length; ++index) {
      if (appliedLoader[index] === "{") { ++depth; } else if (appliedLoader[index] === "}" && --depth === 0) { close = index; break; }
    }
    assert.ok(close > open, "Could not isolate applied context_t::init body");
    const body = appliedLoader.slice(open + 1, close);
    assert.match(body, /#if defined\(MOM_NATIVE_STATIC_ADAPTERS\)[\s\S]*staticAdapterRegistered/);
    assert.match(body, /#if !defined\(MOM_NATIVE_STATIC_ADAPTERS\)[\s\S]*LibLoader::loadAdapterLibrary/);

    const prefix = String.raw`#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

using ur_result_t = int;
constexpr ur_result_t UR_RESULT_SUCCESS = 0;
constexpr ur_result_t UR_RESULT_ERROR_UNSUPPORTED_FEATURE = 44;
constexpr ur_result_t MOCK_FACTORY_FAILURE = 77;
using UINT = unsigned int;
constexpr UINT SEM_FAILCRITICALERRORS = 1;
static UINT error_mode = 0;
static int set_error_mode_calls = 0;
UINT SetErrorMode(UINT mode) {
  ++set_error_mode_calls;
  const UINT previous = error_mode;
  error_mode = mode;
  return previous;
}
#define MAKE_LIBRARY_NAME(name, version) name

struct Ddi {};
struct Platform {
  Ddi dditable{};
  explicit Platform(void *) {}
};
struct MockPath {
  std::string value;
  std::string string() const { return value; }
};
struct AdapterRegistry {
  bool forceLoaded = false;
  std::set<std::string> enabled;
  std::vector<std::vector<MockPath>> paths;
  std::set<std::string> staticallyLoaded;
  bool adaptersForceLoaded() const { return forceLoaded; }
  bool isAdapterEnabled(const char *name) const { return enabled.count(name) != 0; }
  void markAdapterAsStaticallyLoaded(const char *name) { staticallyLoaded.insert(name); }
  bool isStaticallyLoaded(const MockPath &path) const {
    return staticallyLoaded.count(path.value) != 0;
  }
  auto begin() const { return paths.begin(); }
  auto end() const { return paths.end(); }
};
static int dynamic_calls = 0;
static int factory_calls = 0;
static ur_result_t level_zero_result = UR_RESULT_SUCCESS;
static ur_result_t level_zero_v2_result = UR_RESULT_SUCCESS;
static ur_result_t opencl_result = UR_RESULT_SUCCESS;
static ur_result_t cuda_result = UR_RESULT_SUCCESS;
namespace ur {
namespace level_zero {
ur_result_t urAdapterGetDdiTables(Ddi *tables) {
  (void)tables; ++factory_calls; return level_zero_result;
}
}
namespace opencl {
ur_result_t urAdapterGetDdiTables(Ddi *tables) {
  (void)tables; ++factory_calls; return opencl_result;
}
}
}
extern "C" ur_result_t mom_ur_level_zero_v2_factory(Ddi *tables) {
  (void)tables; ++factory_calls; return level_zero_v2_result;
}
extern "C" ur_result_t mom_ur_cuda_factory(Ddi *tables) {
  (void)tables; ++factory_calls; return cuda_result;
}
static int dynamic_handle = 0;
struct LibLoader {
  static void *loadAdapterLibrary(const char *path) {
    (void)path; ++dynamic_calls; return &dynamic_handle;
  }
};
bool getenv_tobool(const char *, bool fallback = false) { return fallback; }
struct context_t {
  AdapterRegistry adapter_registry;
  std::vector<Platform> platforms;
  bool forceIntercept = false;
  bool intercept_enabled = false;
  ur_result_t init() {
`;
    const suffix = String.raw`
  }
};
int main(int argc, char **argv) {
  context_t context;
  const std::string scenario = argc > 1 ? argv[1] : "all";
  const auto add = [&](const char *name) {
    context.adapter_registry.enabled.insert(name);
    context.adapter_registry.paths.push_back({MockPath{name}});
  };
  if (scenario == "all" || scenario == "search" ||
      scenario == "fail-v2" || scenario == "fail-cuda" ||
      scenario == "force") {
    add("ur_adapter_level_zero");
    add("ur_adapter_level_zero_v2");
    add("ur_adapter_opencl");
    add("ur_adapter_cuda");
  } else if (scenario == "only-v2") {
    add("ur_adapter_level_zero_v2");
  } else if (scenario == "only-cuda") {
    add("ur_adapter_cuda");
  } else if (scenario == "dynamic") {
    context.adapter_registry.paths.push_back({MockPath{"malicious.dll"}});
  }
  if (scenario == "search" || scenario == "dynamic") {
    context.adapter_registry.paths.clear();
    context.adapter_registry.paths.push_back({MockPath{"malicious.dll"}});
  }
  context.adapter_registry.forceLoaded = scenario == "force";
  if (scenario == "fail-v2") level_zero_v2_result = MOCK_FACTORY_FAILURE;
  if (scenario == "fail-cuda") cuda_result = MOCK_FACTORY_FAILURE;
  const ur_result_t result = context.init();
  if (error_mode != 0) return 2;
  std::cout << result << ' ' << dynamic_calls << ' '
            << factory_calls << ' ' << set_error_mode_calls << '\n';
  return 0;
}
`;
    const staticMacros = ["UR_STATIC_ADAPTER_LEVEL_ZERO", "UR_STATIC_ADAPTER_LEVEL_ZERO_V2",
      "UR_STATIC_ADAPTER_OPENCL", "UR_STATIC_ADAPTER_CUDA"];
    /** Compile the actual patched body; mocks supply dependencies, not loader decisions.
     * @param {string} bodyText @param {string[]} defines @param {string} scenario */
    const run = (bodyText, defines, scenario) => {
      const tag = (defines.join("_") || "dynamic") + "-" + scenario +
        "-" + (bodyText === body ? "applied" : "broken");
      const file = path.join(root, "loader-" + tag + ".cpp"), executable = path.join(root, "loader-" + tag);
      fs.writeFileSync(file, prefix + bodyText + suffix);
      const warningFlags = bodyText === body ? [] : ["-Wno-unused-but-set-variable"];
      const compile = spawnSync("c++", ["-std=c++17", "-Wall", "-Wextra", "-Werror", ...warningFlags, "-D_WIN32",
        ...defines.map(value => "-D" + value), file, "-o", executable], {encoding: "utf8", timeout: 30000});
      assert.equal(compile.status, 0, compile.stderr);
      const result = spawnSync(executable, [scenario], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, result.stderr);
      return result.stdout.trim().split(/\s+/).map(Number);
    };
    const native = ["MOM_NATIVE_STATIC_ADAPTERS", ...staticMacros];
    assert.deepEqual(run(body, native, "all"), [0, 0, 4, 2]);
    assert.deepEqual(run(body, native, "only-v2"), [0, 0, 1, 2]);
    assert.deepEqual(run(body, native, "only-cuda"), [0, 0, 1, 2]);
    assert.deepEqual(run(body, native, "search"), [0, 0, 4, 2]);
    assert.deepEqual(run(body, native, "force"), [44, 0, 0, 2]);
    assert.deepEqual(run(body, native, "empty"), [44, 0, 0, 2]);
    assert.deepEqual(run(body, native, "fail-v2"), [77, 0, 2, 2]);
    assert.deepEqual(run(body, native, "fail-cuda"), [77, 0, 4, 2]);
    assert.deepEqual(run(body, staticMacros, "all"), [0, 0, 4, 2]);
    assert.deepEqual(run(body, [], "dynamic"), [0, 1, 0, 2]);

    const nativeClosure = /\n#if defined\(MOM_NATIVE_STATIC_ADAPTERS\)\n {2}\/\/ Native owns the linked factories[\s\S]*?\n#endif\n/;
    assert.match(body, nativeClosure);
    const brokenBody = body.replace(nativeClosure, "\n");
    assert.notEqual(brokenBody, body);
    assert.notDeepEqual(run(brokenBody, native, "empty"), [44, 0, 0, 2],
      "removing the Native closure guard must fail the empty-closure check");
  });

  test("the actual CMake Windows branch removes dynamic CRT options without changing other options", t => {
    const root = temporary(t), source = path.join(root, "source"); fs.mkdirSync(source);
    const runtime = cmake.indexOf("add_library(mom-sycl-static STATIC $<TARGET_OBJECTS:sycl_object>)");
    const start = cmake.indexOf("if(WIN32)", runtime);
    const end = cmake.indexOf("\nendif()", start);
    assert.ok(start >= 0 && end > start);
    fs.writeFileSync(path.join(source, "policy.cmake"), cmake.slice(start, end + "\nendif()".length));
    fs.writeFileSync(path.join(source, "empty.cpp"), "int value;\n");
    fs.writeFileSync(path.join(source, "CMakeLists.txt"), "cmake_minimum_required(VERSION 3.20)\nproject(policy LANGUAGES CXX)\n" +
      "add_library(sycl_object OBJECT empty.cpp)\ntarget_compile_options(sycl_object PRIVATE /MD -O2)\n" +
      "add_library(ur_loader INTERFACE)\nadd_library(ur_common INTERFACE)\ninclude(policy.cmake)\n" +
      "get_target_property(options sycl_object COMPILE_OPTIONS)\nget_target_property(crt sycl_object MSVC_RUNTIME_LIBRARY)\n" +
      // eslint-disable-next-line no-template-curly-in-string -- Expanded by CMake, not JavaScript.
      "file(WRITE \"${CMAKE_BINARY_DIR}/policy.txt\" \"${options}\\n${crt}\\n\")\n");
    for (const windows of [false, true]) {
      const output = path.join(root, windows ? "windows" : "linux");
      const result = spawnSync("cmake", ["-S", source, "-B", output, "-DWIN32=" + (windows ? "ON" : "OFF")], {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.stderr);
      const policy = fs.readFileSync(path.join(output, "policy.txt"), "utf8");
      assert.match(policy, /-O2/);
      if (windows) {
        assert.doesNotMatch(policy, /\/MD/); assert.match(policy, /MultiThreaded/); assert.match(policy, /\/FI.*static-windows\.h/);
        assert.match(policy, /-fno-builtin/);
      } else {assert.match(policy, /\/MD/); assert.doesNotMatch(policy, /MultiThreaded|\/FI|-fno-builtin/);}
    }
  });

  test("static exports compile without DLL annotations", t => {
    const root = temporary(t), file = path.join(root, "exports.cpp");
    fs.writeFileSync(file, "#include <type_traits>\n#include " + JSON.stringify(path.join(directory, "static-windows.h")) + "\n" +
      "__SYCL_EXPORT int exported();\n__SYCL_EXPORT_DEPRECATED(\"old\") int oldExport();\n__SYCL_DLL_LOCAL int local();\n" +
      "static_assert(std::is_same_v<decltype(exported), int()>);\n");
    const result = spawnSync("c++", ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", file], {encoding: "utf8", timeout: 30000});
    assert.equal(result.status, 0, result.stderr);
  });
});

test.describe("Embedded Windows loader artifact ownership", {skip: process.platform !== "linux"}, () => {
  test("the original DLL guard fails before and passes after the embedded ownership fix", t => {
    const root = temporary(t), file = path.join(root, "entry.cpp");
    const guard = levelZeroWindowsPatch.match(/^\+(#if !defined\(L0_STATIC_LOADER_BUILD\)[^\n]+)$/m)?.[1]; assert.ok(guard);
    for (const [name, condition, native, passes] of [["before", "#ifndef L0_STATIC_LOADER_BUILD", true, false], ["native", guard, true, true], ["shared", guard, false, true]]) {
      fs.writeFileSync(file, "#ifdef MOM_EMBEDDED_LEVEL_ZERO\nextern \"C\" int DllMain() {return 1;}\n#endif\n" + condition +
        "\nextern \"C\" int DllMain() {return 2;}\n#endif\nint main() {\n#ifdef MOM_EMBEDDED_LEVEL_ZERO\nreturn DllMain() != 1;\n#else\nreturn DllMain() != 2;\n#endif\n}\n");
      const executable = path.join(root, String(name)), build = spawnSync("c++", ["-std=c++14", "-Wall", "-Wextra", "-Werror", ...(native ? ["-DMOM_EMBEDDED_LEVEL_ZERO=1"] : []), file, "-o", executable], {encoding: "utf8", timeout: 30000});
      if (!passes) {assert.notEqual(build.status, 0); assert.match(build.stderr, /redefinition/); continue;}
      assert.equal(build.status, 0, build.stderr); const run = spawnSync(executable, [], {encoding: "utf8", timeout: 5000}); assert.equal(run.status, 0, run.stderr);
    }
  });
  test("the actual native CMake policy excludes DLL resources without changing shared behavior", t => {
    const root = temporary(t), file = path.join(root, "resource.cmake");
    const section = levelZeroWindowsPatch.split("diff --git a/source/CMakeLists.txt b/source/CMakeLists.txt\n")[1]?.split("\ndiff --git ")[0] || "";
    const body = section.split("\n").filter(line => line.startsWith("+") && !line.startsWith("+++")).map(line => line.slice(1)).join("\n"); assert.match(body, /target_sources/);
    fs.writeFileSync(file, "function(target_sources)\nset(resource_seen TRUE PARENT_SCOPE)\nendfunction()\nset(TARGET_LOADER_NAME ze_loader)\n" + body +
      "\nif(MOM_EMBEDDED_LEVEL_ZERO AND resource_seen)\nmessage(FATAL_ERROR \"Native archive kept a DLL resource\")\nendif()\nif(NOT MOM_EMBEDDED_LEVEL_ZERO AND NOT resource_seen)\nmessage(FATAL_ERROR \"Non-native resource behavior changed\")\nendif()\n");
    for (const native of ["ON", "OFF"]) {
      const run = spawnSync("cmake", ["-DMOM_EMBEDDED_LEVEL_ZERO=" + native, "-P", file], {encoding: "utf8", timeout: 5000}); assert.equal(run.status, 0, run.stderr);
    }
  });
});
