"use strict";

// Portable policy guards complement the opt-in real CMake/compiler/runtime proofs.
const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path");
const os = require("node:os");
const {spawnSync} = require("node:child_process");
const compilerPolicy = require("../compiler-policy");
const {extractCppBlock} = require("./common/cpp-source");
const read = (/** @type {string} */ file) => fs.readFileSync(path.join(__dirname, "..", file), "utf8");
const runtime = read("runtime/adaptivecpp-static.patch"), jit = read("runtime/adaptivecpp-jit.patch");
const added = (/** @type {string} */ patch) => patch.split("\n").filter((line) => /^\+(?!\+)/.test(line)).map((line) => line.slice(1)).join("\n");
const patchBlock = (/** @type {string} */ patch, /** @type {string} */ file) =>
  patch.split(/(?=^diff --git )/m).find((part) => part.startsWith(`diff --git a/${file} `)) || "";

test.describe("Native kernel route policy without addon files", {concurrency: false}, () => {
  test.beforeEach((t) => {
    assert.ok("mock" in t && "after" in t);
    compilerPolicy.selection("octopus", "amd", "linux"); // Prime only the maintained policy table, before blocking addon lookup.
    const previous = globalThis.__momNativeCore;
    globalThis.__momNativeCore = {AsyncWorker: class {sendToCpp() {throw new Error("Policy test must not dispatch work");}}};
    t.after(() => {globalThis.__momNativeCore = previous;});
    t.mock.method(fs, "existsSync", () => {throw new Error("Native kernel policy attempted addon/runtime filesystem admission");});
  });
  /** @type {Array<[string, string]>} */
  const routes = [["octopus", "MOM_OCTOPUS_SYCL_NATIVE"], ["walahash", "MOM_WALAHASH_SYCL_NATIVE"],
    ["xelishashv3", "MOM_XELISHASHV3_SYCL_NATIVE"]];
  for (const [algo, key] of routes) {
    for (const backend of ["auto", "sycl", "sycl-native"]) {
      test(`${algo} ${backend} derives the shared kernel flag without a worker addon`, () => {
        const env = compilerPolicy.workerEnv(algo, {MOM_GPU_BACKEND: "amd"}, "linux", backend);
        const effective = backend === "auto" ? compilerPolicy.selection(algo, "amd", "linux")?.backend : backend;
        assert.equal(env[key], effective === "sycl-native" ? "1" : "0");
        assert.equal(env["ACPP_VISIBILITY_MASK"], "hip");
        assert.equal(env["MOM_SYCL_COMPILER"], "acpp-hip");
        assert.equal(env["MOM_NATIVE_PATH"], undefined); assert.equal(env["LD_LIBRARY_PATH"], undefined);
      });
    }
  }
  test("native mode ignores an inherited Engineering addon override, not kernel selection", () => {
    const env = compilerPolicy.workerEnv("octopus", {MOM_GPU_BACKEND: "amd", MOM_NATIVE_PATH: "/absent/mom.node"}, "linux", "sycl-native");
    assert.equal(env["MOM_OCTOPUS_SYCL_NATIVE"], "1"); assert.equal(env["MOM_NATIVE_PATH"], undefined);
  });
  test("native mode keeps transport validation even though no library needs loading", () => {
    assert.throws(() => compilerPolicy.workerEnv("octopus", {MOM_GPU_BACKEND: "amd"}, "linux", "sycl-opencl"), /incompatible/);
  });
  test("native mode preserves isolated device selection", () => {
    const env = compilerPolicy.workerEnv("octopus", {MOM_GPU_BACKEND: "amd", MOM_GPU_INDEX: "1"}, "linux", "sycl-native");
    assert.equal(env["HIP_VISIBLE_DEVICES"], "1"); assert.equal(env["MOM_GPU_INDEX"], "0"); assert.equal(env["ACPP_VISIBILITY_MASK"], "hip");
  });
  test("native Windows Intel routes preserve selectors without DLL search paths", () => {
    const env = compilerPolicy.workerEnv("octopus", {MOM_GPU_BACKEND: "intel"}, "win32", "sycl-native");
    assert.equal(env["MOM_OCTOPUS_SYCL_NATIVE"], "1"); assert.equal(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu");
    assert.equal(env["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"], "1"); assert.equal(env["PATH"], undefined);
    assert.equal(env["MOM_SYCL_COMPILER"], "oneapi");
  });
  test("modern NVIDIA retains its DPC++ worker key without external metadata commands", () => {
    const env = compilerPolicy.workerEnv("cn/gpu", {MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "120"}, "linux");
    assert.equal(env["MOM_SYCL_COMPILER"], "dpcpp"); assert.equal(env["ONEAPI_DEVICE_SELECTOR"], "cuda:gpu");
  });
  test("older NVIDIA retains its AdaptiveCpp compatibility worker key", () => {
    const env = compilerPolicy.workerEnv("c30", {MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "70"}, "linux");
    assert.equal(env["MOM_SYCL_COMPILER"], "acpp-cuda"); assert.equal(env["ACPP_VISIBILITY_MASK"], "cuda");
  });
  test("portable Nexa selection keeps its compiler/image ABI rather than the native field image", () => {
    const env = compilerPolicy.workerEnv("nexapow", {MOM_GPU_BACKEND: "nvidia", MOM_NVIDIA_COMPUTE_CAPABILITY: "120"}, "linux", "sycl");
    assert.equal(env["MOM_SYCL_COMPILER"], "acpp-cuda"); assert.equal(env["ACPP_VISIBILITY_MASK"], "cuda");
  });
  test("native control discovery uses the same compiler policy without selecting an addon", () => {
    const env = compilerPolicy.workerEnv("__control__", {MOM_GPU_BACKEND: "amd"}, "linux");
    assert.equal(env["MOM_SYCL_COMPILER"], "acpp-hip"); assert.equal(env["MOM_NATIVE_PATH"], undefined);
    assert.match(read("mom.js"), /globalThis\.__momNativeCore !== undefined \|\| !process\.env\["MOM_NATIVE_PATH"\]/);
  });
  test("CPU-only workers do not inherit a manufactured GPU compiler selection", () => {
    assert.deepEqual(compilerPolicy.workerEnv("rx/0", {}, "linux"), {});
  });
  test("Engineering keeps its intentional explicit-addon override", () => {
    globalThis.__momNativeCore = undefined;
    assert.deepEqual(compilerPolicy.workerEnv("octopus", {MOM_GPU_BACKEND: "amd", MOM_NATIVE_PATH: "/explicit/mom.node"}, "linux", "sycl-native"), {});
  });
});

test.describe("Static SYCL build policy", () => {
  test("static runtime remains opt-in", () => {
    assert.match(added(runtime), /option\(ACPP_STATIC_RUNTIME [^\n]* OFF\)/);
  });
  test("the static registry retains the CPU backend required for host-memory support", () => {
    assert.match(added(runtime), /foreach\(backend omp cuda hip ze ocl metal vk\)/);
    assert.doesNotMatch(added(runtime), /WITH_CPU_BACKEND.*OFF/);
  });
  test("Windows static CPU support executes as a direct profile without a public sequential option", () => {
    const cmake = patchBlock(runtime, "CMakeLists.txt");
    const cpuCmake = added(patchBlock(runtime, "src/runtime/CMakeLists.txt"));
    assert.match(cpuCmake, /if\(ACPP_STATIC_RUNTIME AND WIN32\)[\s\S]*set\(OMP_COMPILE_FLAGS\)[\s\S]*set\(OMP_TARGET\)/);
    assert.match(cpuCmake, /elseif\(TARGET omp AND ACPP_LLVM_COMPONENT\)/);
    assert.doesNotMatch(runtime, /ACPP_OMP_SEQUENTIAL/);

    // Execute the exact added profile blocks. The untouched source sentinel is
    // retained for Windows shared and Linux static configurations.
    const profileBlocks = added(cmake).match(/if\(ACPP_STATIC_RUNTIME AND WIN32\)[\s\S]*?endif\(\)/g) || [];
    assert.equal(profileBlocks.length, 2);
    const original = {
      DEFAULT_OMP_FLAG: "-fopenmp", OMP_LINK_LINE: "stale-omp-link",
      SEQUENTIAL_LINK_LINE: "stale-sequential-link", OMP_CXX_FLAGS: "stale-omp-flags"
    };
    const executeProfile = (/** @type {boolean} */ staticRuntime, /** @type {boolean} */ windows) => {
      const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-acpp-static-profile-"));
      const script = path.join(directory, "profile.cmake");
      const result = path.join(directory, "result.txt").replaceAll("\\", "/");
      const body = [
        `set(ACPP_STATIC_RUNTIME ${staticRuntime ? "ON" : "OFF"})`,
        `set(WIN32 ${windows ? "TRUE" : "FALSE"})`,
        `set(DEFAULT_OMP_FLAG "${original.DEFAULT_OMP_FLAG}")`,
        `set(OMP_LINK_LINE "${original.OMP_LINK_LINE}" CACHE STRING "fixture" FORCE)`,
        `set(SEQUENTIAL_LINK_LINE "${original.SEQUENTIAL_LINK_LINE}" CACHE STRING "fixture" FORCE)`,
        `set(OMP_CXX_FLAGS "${original.OMP_CXX_FLAGS}" CACHE STRING "fixture" FORCE)`,
        ...profileBlocks,
        "get_property(omp_link CACHE OMP_LINK_LINE PROPERTY VALUE)",
        "get_property(sequential_link CACHE SEQUENTIAL_LINK_LINE PROPERTY VALUE)",
        "get_property(omp_flags CACHE OMP_CXX_FLAGS PROPERTY VALUE)",
        `file(WRITE "${result}" "DEFAULT_OMP_FLAG=\${DEFAULT_OMP_FLAG}\\nOMP_LINK_LINE=\${omp_link}\\nSEQUENTIAL_LINK_LINE=\${sequential_link}\\nOMP_CXX_FLAGS=\${omp_flags}\\n")`
      ].join("\n");
      try {
        fs.writeFileSync(script, body);
        const run = spawnSync("cmake", ["-P", script], {encoding: "utf8", timeout: 30000});
        assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);
        return Object.fromEntries(fs.readFileSync(result, "utf8").trim().split(/\r?\n/).map((line) => {
          const [key, ...value] = line.split("=");
          return [key, value.join("=")];
        }));
      } finally {
        fs.rmSync(directory, {recursive: true, force: true});
      }
    };
    assert.deepEqual(executeProfile(true, true), {
      DEFAULT_OMP_FLAG: "", OMP_LINK_LINE: "", SEQUENTIAL_LINK_LINE: "",
      OMP_CXX_FLAGS: "-D_ENABLE_EXTENDED_ALIGNED_STORAGE"
    });
    assert.deepEqual(executeProfile(false, true), original);
    assert.deepEqual(executeProfile(true, false), original);
  });
  test("the Windows OMP sources guard the header and use a real host CPU fallback", () => {
    const manager = added(patchBlock(runtime, "src/runtime/omp/omp_hardware_manager.cpp"));
    const queue = patchBlock(runtime, "src/runtime/omp/omp_queue.cpp");
    assert.match(manager, /#include <thread>/);
    assert.match(manager, /#ifdef _OPENMP\s+#include <omp\.h>\s+#endif/);
    assert.match(manager, /std::thread::hardware_concurrency\(\)/);
    assert.match(manager, /return n \? n : 1/);
    assert.match(queue, /\+#ifdef _OPENMP\n #include <omp\.h>\n\+#endif/);
  });
  test("the direct profile remains applicable only to the pinned AdaptiveCpp source", () => {
    const probe = read("scripts/native-sycl-inputs.js");
    assert.match(probe, /const syclRevision = "da2463e45aa90aa36306c45abcfc05b87de51bc6"/);
    assert.match(patchBlock(runtime, "CMakeLists.txt"), /if\(ACPP_STATIC_RUNTIME AND WIN32\)/);
    assert.match(patchBlock(runtime, "src/runtime/CMakeLists.txt"), /if\(ACPP_STATIC_RUNTIME AND WIN32\)/);
    for (const file of ["src/runtime/omp/omp_hardware_manager.cpp", "src/runtime/omp/omp_queue.cpp"]) {
      assert.match(patchBlock(runtime, file), /#ifdef _OPENMP/);
    }
  });
  test("explicit legacy compiler profiles persist through automatic regeneration", () => {
    assert.match(added(runtime), /set\(ACPP_COMPILER_FEATURE_PROFILE "custom-deprecated" CACHE STRING [^\n]* FORCE\)/);
  });
  test("custom feature values are initialized only when absent", () => {
    assert.match(added(runtime), /foreach\(feature SSCP_COMPILER STDPAR_COMPILER ACCELERATED_CPU\)\s+if\(NOT DEFINED ACPP_CUSTOM_PROFILE_WITH_\$\{feature\}\)/);
  });
  test("static CUDA uses its application-runtime archive without a mandatory driver import", () => {
    assert.match(added(jit), /if\(ACPP_STATIC_RUNTIME\)\s+set\(CUDA_LIBS CUDA::cudart_static\)/);
    assert.doesNotMatch(read("scripts/native-sycl-inputs.js"), /CUDA_cuda_driver_LIBRARY|"-lcuda"/);
  });
  test("shared CUDA keeps the existing runtime target", () => {
    assert.match(jit, / set\(CUDA_LIBS CUDA::cudart CUDA::cuda_driver\)/);
  });
  test("resource headers depend on the generator and both bitcode libraries", () => {
    assert.match(added(jit), /DEPENDS "\$\{ACPP_RESOURCE_GENERATOR\}" "\$\{ptx_builtins\}" "\$\{ptx_libdevice\}"/);
  });
  test("generated resources are compiler inputs with quoted non-shell arguments", () => {
    assert.match(added(jit), /VERBATIM\)/);
    assert.match(added(jit), /target_sources\(llvm-to-ptx PRIVATE "\$\{ptx_resources\}"\)/);
  });
  test("the existing runtime factory supplies its linked immutable libraries", () => {
    assert.match(added(jit), /translator->provideBitcodeLibraryResolver\(/);
    assert.match(added(jit), /return embedded_resource\(Name\)/);
  });
  test("static compilers cannot fall back to a disk library resolver", () => {
    assert.match(added(jit), /Static JIT requires an embedded bitcode-library resolver/);
    assert.match(added(jit), /Missing embedded bitcode library:/);
  });
  test("the real probe regenerates and incrementally builds reviewed compiler inputs", () => {
    const probe = read("tests/tools/probe-native-ptx.js");
    assert.match(probe, /run\("cmake", \["-S", source, "-B", build\]/);
    assert.match(probe, /"--target", "llvm-to-ptx", "acpp-clang", "libkernel-sscp-ptx"/);
  });
  test("runtime-factory output is compared with independent reference libraries", () => {
    const probe = read("tests/reference/native-ptx-translator.cpp");
    assert.match(probe, /createLLVMToPtxTranslator\(kernels\)/);
    assert.match(probe, /factory->fullTransformation\(generic, actual\) \|\| actual != ptx/);
  });
  test("the CUDA runtime lane refuses implicit execution before any device access", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/native-sycl-inputs.js")],
      {encoding: "utf8", env: {...process.env, MOM_AOT_SYCL_BACKEND: "cuda", MOM_AOT_BUILD_ONLY: "0"}, timeout: 15000});
    assert.equal(result.status, 1); assert.match(result.stderr, /verify fresh ownership/);
    assert.doesNotMatch(result.stdout, /Static SYCL evidence/);
  });
  test("the combined runtime also requires build-only admission before any tool or device access", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/native-sycl-inputs.js")],
      {encoding: "utf8", env: {...process.env, MOM_AOT_SYCL_BACKEND: "cuda-hip", MOM_AOT_BUILD_ONLY: "0", PATH: "/unavailable"}, timeout: 5000});
    assert.equal(result.status, 1); assert.match(result.stderr, /verify fresh ownership/);
    assert.doesNotMatch(result.stdout, /Static SYCL evidence/); assert.doesNotMatch(result.stderr, /ENOENT|spawn/);
  });
  test("per-backend factories stay isolated on the combined runtime cache", () => {
    const probe = read("scripts/native-sycl-inputs.js");
    assert.match(probe, /validateStaticSyclBackendCache\(cache, backend\)/);
    assert.match(probe, /const backends = backend\.split\("-"\)/);
    assert.match(probe, /\.\.\.backends\.map\(\(name\) => path\.join\(build, `src\/runtime\/librt-backend-\$\{name\}\.a`\)\)/);
  });
  test("a runtime archive cache cannot silently come from another source tree", () => {
    const probe = read("scripts/native-sycl-inputs.js");
    assert.match(probe, /CMAKE_HOME_DIRECTORY:INTERNAL/);
    assert.match(probe, /fs\.realpathSync\(configuredSource\) !== fs\.realpathSync\(source\)/);
  });
  test("complete GPU builds retain the reviewed resource generator and SSCP-only admission", () => {
    const probe = read("scripts/native-sycl-inputs.js");
    assert.match(probe, /ACPP_RESOURCE_GENERATOR:FILEPATH/);
    assert.match(probe, /\["STDPAR_COMPILER", "ACCELERATED_CPU"\]\.some/);
  });
});

test.describe("Optional native CUDA driver boundary", {skip: process.platform !== "linux"}, () => {
  let directory = "";
  test.before(() => {
    directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-driver-boundary-"));
    const block = jit.split(/(?=^diff --git )/m).find((part) => part.startsWith(
      "diff --git a/include/hipSYCL/runtime/cuda/cuda_driver.hpp "));
    assert.ok(block, "Missing reviewed driver boundary");
    fs.writeFileSync(path.join(directory, "cuda_driver.hpp"), added(block) + "\n");
    // Empty SDK includes deliberately prevent the test from using installed CUDA/driver files.
    for (const name of ["cuda_runtime_api.h", "cuda.h", "cudaTypedefs.h"]) {fs.writeFileSync(path.join(directory, name), "");}
    for (const mode of ["static", "shared"]) {
      const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        ...(mode === "static" ? ["-DACPP_STATIC_RUNTIME"] : []), "-I", directory,
        path.join(__dirname, "native/cuda-driver.cpp"), "-o", path.join(directory, mode)], {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.error?.message || result.stderr);
    }
  });
  test.after(() => {if (directory) {fs.rmSync(directory, {recursive: true, force: true});}});
  /** @param {string} mode @param {string} scenario */
  function run(mode, scenario) {
    const result = spawnSync(path.join(directory, mode), [scenario], {encoding: "utf8", timeout: 5000});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
    return result.stdout;
  }
  test("four stable ABIs forward every argument and cache each lookup", () => {
    assert.match(run("static", "success"), /queries=4 launches=20/);
  });
  test("a missing driver is cached and never dereferenced", () => {assert.match(run("static", "no-driver"), /launches=0/);});
  test("success with a null function fails closed", () => {assert.match(run("static", "null"), /launches=0/);});
  test("lookup failure rejects even a non-null returned pointer", () => {assert.match(run("static", "query-error"), /launches=0/);});
  test("symbol-status failure rejects even a non-null returned pointer", () => {assert.match(run("static", "status-error"), /launches=0/);});
  test("actual driver-call errors propagate unchanged", () => {assert.match(run("static", "call-error"), /launches=20/);});
  test("shared mode retains direct calls without runtime lookups", () => {assert.match(run("shared", "success"), /queries=0 launches=20/);});
  test("shared mode error propagation is unchanged", () => {assert.match(run("shared", "call-error"), /queries=0 launches=20/);});
  test("shared mode is independent of native lookup failures", () => {assert.match(run("shared", "no-driver"), /queries=0 launches=20/);});
  test("both real ACPP call sites route through the same boundary", () => {
    for (const name of ["cuModuleLoadDataEx", "cuModuleUnload", "cuModuleGetFunction", "cuLaunchKernel"]) {
      assert.match(added(jit), new RegExp(`cuda_driver::${name}\\(`));
    }
  });
});

test.describe("Static DPC++ core build admission", () => {
  const {pins, sdkIdentity, sourceJitInput} = require("../scripts/native-dpcpp-pins");
  const version = `clang version 23.0.0git (https://github.com/intel/llvm ${pins.source})`;
  test("the matching Engineering SDK/source/archive pair is admitted", () => {sdkIdentity(version, pins.sdk + "\n");});
  test("a matching banner cannot bypass an absent or different archive receipt", () => {
    for (const receipt of ["", "0".repeat(64), pins.sdk.slice(1)]) {assert.throws(() => sdkIdentity(version, receipt), /pinned/);}
  });
  test("an archive receipt cannot bypass a different compiler source revision", () => {
    assert.throws(() => sdkIdentity(version.replace(pins.source, "0".repeat(40)), pins.sdk), /pinned/);
  });
  test("a compiler upgrade is not silently treated as the qualified ABI", () => {
    assert.throws(() => sdkIdentity(version.replace("23.0.0git", "24.0.0git"), pins.sdk), /pinned/);
  });
  test("build-only admission precedes tool lookup and all input access", () => {
    const run = spawnSync(process.execPath, [path.join(__dirname, "../scripts/native-dpcpp-pins.js")], {encoding: "utf8", timeout: 5000,
      env: {...process.env, MOM_AOT_BUILD_ONLY: "0", PATH: "/unavailable"}});
    assert.equal(run.status, 1); assert.match(run.stderr, /MOM_AOT_BUILD_ONLY=1/); assert.doesNotMatch(run.stderr, /ENOENT|spawn/);
  });
  test("relative and CMake-metacharacter paths fail before output mutation", () => {
    for (const unsafe of ["relative", "/tmp/injection;target", "/tmp/$expression", "/tmp/new\nline", '/tmp/quote"']) {
      const run = spawnSync(process.execPath, [path.join(__dirname, "../scripts/native-dpcpp-pins.js"), unsafe,
        "/sdk", "/emhash", "/umf", "/unavailable/output"], {encoding: "utf8", timeout: 5000,
        env: {...process.env, MOM_AOT_BUILD_ONLY: "1", PATH: "/unavailable"}});
      assert.equal(run.status, 1); assert.match(run.stderr, /Usage:/); assert.doesNotMatch(run.stderr, /Input query|ENOENT|mkdir/);
    }
  });
  test("the complete host object target is reused without a second runtime source registry", () => {
    const cmake = read("tests/native/dpcpp-static/CMakeLists.txt");
    assert.match(cmake, /add_subdirectory\("\$\{SYCL_SOURCE_DIR\}\/source" runtime\)/);
    assert.match(cmake, /STATIC \$<TARGET_OBJECTS:sycl_object>/);
    assert.doesNotMatch(cmake, /SYCL_COMMON_SOURCES|\bfile\(\s*GLOB(?:_RECURSE)?\b/);
  });
  test("whole-archive and no-GC linking expose missing static dependencies", () => {
    const cmake = read("tests/native/dpcpp-static/CMakeLists.txt");
    assert.match(cmake, /--whole-archive mom-sycl-static/); assert.match(cmake, /--no-gc-sections/);
    assert.match(cmake, /-static-libstdc\+\+ -static-libgcc/);
  });
  test("the baseline is explicitly adapter/JIT-free and does not fetch mutable build inputs", () => {
    const cmake = read("tests/native/dpcpp-static/CMakeLists.txt");
    assert.match(cmake, /set\(SYCL_ENABLE_EXTENSION_JIT OFF\)/); assert.match(cmake, /UR_BUILD_ADAPTER_\$\{backend\} OFF CACHE BOOL "" FORCE/);
    assert.match(cmake, /set\(FETCHCONTENT_FULLY_DISCONNECTED ON/); assert.match(cmake, /set\(UR_USE_EXTERNAL_UMF OFF/);
  });
  test("the experiment never becomes the Engineering or normal release entrypoint", () => {
    for (const file of ["scripts/dev.js", "scripts/build-release.js"]) {assert.doesNotMatch(read(file), /native-dpcpp-pins/);}
    assert.doesNotMatch(read("scripts/native-dpcpp-pins.js"), /capture\(executable|build\(executable|nvidia-smi/);
  });
  test("source JIT requires one isolated compiler object or Windows archive", () => {
    assert.equal(sourceJitInput({}), null);
    assert.equal(sourceJitInput({MOM_AOT_DPCPP_JIT_OBJECT: "/compiler/isolated.o"}), "/compiler/isolated.o");
    const cmake = read("tests/native/dpcpp-static/CMakeLists.txt");
    assert.match(cmake, /if\(MOM_DPCPP_JIT_OBJECT AND MOM_DPCPP_JIT_ARCHIVE\)[\s\S]*?message\(FATAL_ERROR/);
    assert.match(cmake, /if\(MOM_DPCPP_JIT_OBJECT OR MOM_DPCPP_JIT_ARCHIVE\)[\s\S]*?set\(SYCL_ENABLE_EXTENSION_JIT ON\)/);
    assert.match(cmake, /if\(NOT WIN32 OR NOT IS_ABSOLUTE "\$\{MOM_DPCPP_JIT_ARCHIVE\}"/);
  });
  test("unsafe source-JIT values cannot reach CMake", () => {
    for (const value of ["", "relative.o", "/compiler/shared.so", "/compiler/unsafe;.o", "/compiler/$x.o", "/compiler/line\nbreak.o"]) {
      assert.throws(() => sourceJitInput({MOM_AOT_DPCPP_JIT_OBJECT: value}), /absolute isolated/);
    }
  });
  test("the runtime bridge binds the original isolated compiler rather than a shared SDK", () => {
    const source = read("tests/native/dpcpp-static/CMakeLists.txt");
    assert.match(source, /add_library\(sycl-jit OBJECT IMPORTED\)/);
    assert.match(source, /IMPORTED_OBJECTS "\$\{MOM_DPCPP_JIT_OBJECT\}"/);
    assert.match(source, /set\(SYCL_JIT_STATIC ON\)/);
    assert.match(source, /PRIVATE sycl-jit emhash::emhash/);
  });
  test("the complete host proof exercises the real runtime image owner and retire path", () => {
    const source = read("tests/reference/native-dpcpp-static.cpp");
    assert.match(source, /sycl::detail::jit_compiler::get_instance\(\)/);
    assert.match(source, /compiler\.compileSYCL\(/); assert.match(source, /compiler\.destroyDeviceBinaries\(/);
    assert.match(source, /entry->GetName\(\)/); assert.match(source, /image\.Format != SYCL_DEVICE_BINARY_TYPE_SPIRV/);
  });
});

test.describe("Static DPC++ adapter build admission", () => {
  const {adapterInputs} = require("../scripts/native-dpcpp-pins");
  const roots = {MOM_AOT_DPCPP_LEVEL_ZERO_SOURCE: "/loader", MOM_AOT_DPCPP_COMPUTE_HEADERS: "/headers", MOM_AOT_DPCPP_HWLOC_SOURCE: "/hwloc"};
  test("no adapter is selected implicitly", () => {assert.equal(adapterInputs({}), null);});
  test("explicit none still rejects inherited adapter roots", () => {
    assert.throws(() => adapterInputs({...roots, MOM_AOT_DPCPP_ADAPTER: "none"}), /explicitly/);
  });
  test("one complete reviewed Level Zero profile is admitted", () => {
    assert.deepEqual(adapterInputs({...roots, MOM_AOT_DPCPP_ADAPTER: "level_zero"}), {levelZero: "/loader", computeHeaders: "/headers", hwloc: "/hwloc"});
  });
  for (const name of Object.keys(roots)) {
    test(`missing ${name} cannot partially enable an adapter`, () => {
      assert.throws(() => adapterInputs({...roots, [name]: undefined, MOM_AOT_DPCPP_ADAPTER: "level_zero"}), /all three/);
    });
  }
  test("unknown adapter sets and dynamic profiles fail closed", () => {
    for (const name of ["auto", "cuda", "level_zero,cuda", "level_zero ", "shared"]) {
      assert.throws(() => adapterInputs({...roots, MOM_AOT_DPCPP_ADAPTER: name}), /explicitly/);
    }
  });
  test("unsafe source values cannot reach CMake interpolation", () => {
    for (const value of ["relative", "/tmp/a;b", "/tmp/$variable", "/tmp/line\nfeed"]) {
      assert.throws(() => adapterInputs({...roots, MOM_AOT_DPCPP_ADAPTER: "level_zero", MOM_AOT_DPCPP_LEVEL_ZERO_SOURCE: value}), /absolute/);
    }
  });
  test("hwloc generation mutates a private git snapshot, not the pinned input", () => {
    const source = read("scripts/native-dpcpp-pins.js");
    assert.match(source, /\(hwloc\), "archive", "--format=tar"/);
    assert.match(source, /MOM_DPCPP_HWLOC: snapshot/);
  });
  test("static UMF and hwloc cannot inherit a stale shared-runtime setting", () => {
    const source = read("tests/native/dpcpp-static/CMakeLists.txt");
    assert.match(source, /set\(UMF_STATIC_LEVEL_ZERO ON CACHE BOOL "" FORCE\)/);
    assert.match(source, /set\(UMF_LINK_HWLOC_STATICALLY ON CACHE BOOL "" FORCE\)/);
  });
  test("the native loader archives full discovery, not a shared-loader dispatch stub", () => {
    assert.match(read("tests/native/dpcpp-static/CMakeLists.txt"), /set\(MOM_EMBEDDED_LEVEL_ZERO ON CACHE BOOL "" FORCE\)/);
    const patch = added(read("runtime/level-zero-static-complete.patch"));
    assert.match(patch, /if \(NOT MOM_EMBEDDED_LEVEL_ZERO\)[\s\S]*add_definitions\(-DL0_STATIC_LOADER_BUILD/);
    assert.match(patch, /if \(NOT BUILD_STATIC OR MOM_EMBEDDED_LEVEL_ZERO\)/);
    assert.match(read("scripts/native-dpcpp-pins.js"), /directory === adapter\.levelZero.*level-zero-static-complete\.patch/);
  });
  test("the GPU-free loader fixture rejects an application-loader dependency", () => {
    const source = read("tests/reference/native-level-zero-static.cpp");
    assert.match(source, /zeInit\(ZE_INIT_FLAG_GPU_ONLY\)/);
    assert.match(source, /driverCalls != 1 \|\| loaderCalls/);
    assert.match(read("tests/native/dpcpp-static/CMakeLists.txt"), /add_executable\(native-level-zero-driver/);
  });
  test("the real allocator proof wraps only CPU-backed driver ABI calls", () => {
    const source = read("tests/native/dpcpp-static/CMakeLists.txt");
    for (const name of ["zeMemAllocHost", "zeMemGetAllocProperties", "zeMemFree"]) {assert.ok(source.includes("--wrap=" + name));}
    assert.match(source, /add_dependencies\(native-dpcpp-core native-umf-static native-level-zero-driver\)/);
  });
});

test.describe("Static UMF Level Zero provider boundary", {skip: process.platform !== "linux"}, () => {
  const patch = read("runtime/umf-static-level-zero.patch"), code = added(patch);
  let root = "";
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-umf-static-"));
    const body = code.match(/#ifdef UMF_STATIC_LEVEL_ZERO\n([\s\S]*?)#else/)?.[1];
    assert.ok(body, "Missing actual static provider binding");
    fs.writeFileSync(path.join(root, "bindings.inc"), "static void init_ze_global_state(void) {\n" + body + "}\n");
    const build = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", root,
      path.join(__dirname, "native/umf-static-level-zero.cpp"), "-o", path.join(root, "bindings")], {encoding: "utf8", timeout: 30000});
    assert.equal(build.status, 0, build.error?.message || build.stderr);
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  test("archive entry binding remains opt-in for existing Engineering clients", () => {assert.match(code, /umf_option\(UMF_STATIC_LEVEL_ZERO [^\n]+ OFF\)/);});
  test("a shared library or disabled provider cannot request static binding", () => {
    assert.match(code, /UMF_STATIC_LEVEL_ZERO AND \(UMF_BUILD_SHARED_LIBRARY OR NOT UMF_BUILD_LEVEL_ZERO_PROVIDER\)/);
  });
  test("the selected policy reaches the actual provider translation unit", () => {
    assert.match(code, /target_compile_definitions\(umf PRIVATE UMF_STATIC_LEVEL_ZERO=1\)/);
  });
  test("all required and previously optional entrypoints bind without loader/environment probes", () => {
    const body = code.match(/#ifdef UMF_STATIC_LEVEL_ZERO\n([\s\S]*?)#else/)?.[1] || "";
    const fields = [...body.matchAll(/g_ze_ops\.(\w+) = (\w+);/g)];
    assert.equal(fields.length, 13); assert.equal(new Set(fields.map(field => field[1])).size, 13);
    for (const field of fields) {assert.equal(field[1], field[2]);}
    assert.doesNotMatch(body, /utils_open_library|utils_get_symbol_addr|getenv\(/);
  });
  /** @param {string} override */
  function execute(override) {
    const run = spawnSync(path.join(root, "bindings"), [], {encoding: "utf8", timeout: 5000,
      env: {...process.env, UMF_ZE_LOADER_LIB_NAME: override}});
    assert.equal(run.status, 0, run.error?.message || run.stderr); assert.match(run.stdout, /13 exact callbacks/);
  }
  for (const override of ["", "/unavailable/loader.so", "libze_loader.so.1", "/tmp/hostile-provider.dll"]) {
    test(`actual binding ignores loader override ${override || "(empty)"}`, () => {execute(override);});
  }
});

test.describe("Static DPC++ source-JIT binding", {skip: process.platform !== "linux"}, () => {
  const patch = read("runtime/dpcpp-static-jit.patch"), code = added(patch);
  const constructor = added(patch.slice(patch.indexOf("diff --git a/sycl/source/detail/jit_compiler.cpp")));
  const body = constructor.match(/#ifdef MOM_DPCPP_STATIC_JIT\n([\s\S]*?)#else/)?.[1] || "";
  let root = "";
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-dpcpp-jit-binding-"));
    assert.ok(body); fs.writeFileSync(path.join(root, "bindings.inc"), body);
    const run = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", root,
      path.join(__dirname, "native/dpcpp-jit-binding.cpp"), "-o", path.join(root, "binding")], {encoding: "utf8", timeout: 30000});
    assert.equal(run.status, 0, run.error?.message || run.stderr);
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  test("the original shared-library build remains the default", () => {
    assert.match(code, /option\(SYCL_JIT_STATIC [^\n]+ OFF\)/);
    assert.match(code, /set\(SYCL_JIT_LIBRARY_KIND SHARED\)/);
  });
  test("native mode archives the original implementation instead of replacing its algorithms", () => {
    assert.match(code, /set\(SYCL_JIT_LIBRARY_KIND STATIC\)/);
    assert.match(patch, /\+ {3}\$\{SYCL_JIT_LIBRARY_KIND\}/);
  });
  test("the static binding define and archive dependency travel together", () => {
    assert.match(code, /if\(SYCL_JIT_STATIC\)[\s\S]*MOM_DPCPP_STATIC_JIT[\s\S]*target_link_libraries\(\$\{LIB_NAME\} PRIVATE sycl-jit\)/);
  });
  test("all six existing ABI entries bind directly", () => {
    assert.equal([...body.matchAll(/Handle = &::jit_compiler::/g)].length, 6);
  });
  test("static availability does not depend on external library lookup", () => {
    assert.match(body, /Available = true;/); assert.doesNotMatch(body, /loadOsLibrary|getOsLibraryFuncAddress|reinterpret_cast/);
  });
  test("static mode never invents a loader-owned lifetime", () => {assert.doesNotMatch(body, /LibraryHandle\s*=|LibraryPtr|unloadOsLibrary/);});
  test("shared-only linker exports cannot accidentally apply to the archive", () => {
    assert.match(code, /if\(NOT MSVC AND NOT APPLE AND NOT SYCL_JIT_STATIC\)/);
  });
  test("the executable proof reuses the original compiler target and transitive dependencies", () => {
    const cmake = read("tests/native/dpcpp-jit/CMakeLists.txt");
    assert.match(cmake, /target_link_libraries\(native-dpcpp-jit PRIVATE sycl-jit /);
    assert.match(cmake, /jit_type STREQUAL "STATIC_LIBRARY"/);
    assert.doesNotMatch(cmake, /LLVMCore|clangFrontend|GLOB/);
  });
  test("attaching the proof defers until the maintained target graph exists", () => {
    const attach = read("tests/native/dpcpp-jit/attach.cmake");
    assert.match(attach, /IS_ABSOLUTE "\$\{MOM_SOURCE_REPO\}"/);
    assert.match(attach, /cmake_language\(DEFER CALL include/);
  });
  test("Clang flags cannot be re-rendered as Windows options and input filenames", () => {
    assert.match(patch, /MissingArgCount, llvm::opt::Visibility\(ClangOption\)/);
  });
  test("archive classification honors the supplied driver filesystem", () => {
    assert.match(code, /C\.getDriver\(\)\.getVFS\(\)\.getBufferForFile\(LibName\)/);
    assert.match(code, /llvm::identify_magic\(\(\*Buffer\)->getBuffer\(\)\)/);
  });
  test("the native compiler cannot fall back to host SDK includes", () => {
    assert.match(code, /return getToolchainFS\(\);[\s\S]*#else[\s\S]*return llvm::vfs::getRealFileSystem\(\)/);
    assert.match(code, /OptTable.getOption\(OPT_nostdinc\)/);
    assert.match(code, /getEmbeddedFileSystem\(\)/);
    assert.doesNotMatch(code, /MOM-JIT-ARG/);
  });
  test("the resource overlay is explicit, build-only and tracked by the original generator", () => {
    assert.match(code, /IS_ABSOLUTE "\$\{SYCL_JIT_RESOURCE_OVERLAY\}"/);
    assert.match(code, /SYCL_JIT_RESOURCE_FILES \$\{SYCL_JIT_OVERLAY_FILES\}/);
    assert.match(code, /set\(SYCL_JIT_RESOURCE_INSTALL_DIR "\$\{SYCL_JIT_RESOURCE_OVERLAY\}"\)/);
    assert.doesNotMatch(code, /-E copy_directory/);
  });
  /** @param {string} search */
  function execute(search) {
    const run = spawnSync(path.join(root, "binding"), [], {encoding: "utf8", timeout: 5000,
      env: {...process.env, PATH: "/unavailable", LD_LIBRARY_PATH: search}});
    assert.equal(run.status, 0, run.error?.message || run.stderr); assert.match(run.stdout, /six exact callbacks/);
  }
  for (const search of ["", "/unavailable", "/tmp/hostile-jit"]) {
    test(`actual constructor binding needs no compiler/library search (${search || "empty"})`, () => {execute(search);});
  }
});

test.describe("Native KawPow immutable source selection", {skip: process.platform !== "linux"}, () => {
  let root = "";
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-kawpow-embedded-"));
    const resources = require("../scripts/native-resources");
    fs.writeFileSync(path.join(root, "resources.inc"), resources([
      {name: "kawpow-device.inc", data: Buffer.from(read("sycl/kawpow/device.inc"))},
      {name: "kawpow-keccak.inc", data: Buffer.from(read("sycl/kawpow/keccak.inc"))},
    ]).replace("embedded_resource(", "fixture_resource("));
    fs.writeFileSync(path.join(root, "source.inc"), extractCppBlock(read("sycl/kawpow/jit.inc"), "static const std::string& kawpow_device_src()"));
    const run = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-DMOM_EMBEDDED_SYCL_JIT",
      "-I", root, path.join(__dirname, "native/kawpow-embedded-source.cpp"), "-o", path.join(root, "source")], {encoding: "utf8", timeout: 30000});
    assert.equal(run.status, 0, run.error?.message || run.stderr);
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  /** @param {number} mode @param {string} override */
  function execute(mode, override) {
    const run = spawnSync(path.join(root, "source"), [String(mode)], {encoding: "utf8", timeout: 5000,
      env: {...process.env, PATH: "/unavailable", MOM_KAWPOW_DEVICE_INC: override}});
    assert.equal(run.status, 0, run.error?.message || run.stderr); assert.match(run.stdout, /byte-for-byte, cached and retryable/);
  }
  for (const override of ["", "/unavailable/device.inc", "/tmp/untrusted-device.inc"]) {
    test(`native source bytes ignore disk override ${override || "(empty)"}`, () => {execute(0, override);});
  }
  for (const [mode, label] of [[1, "missing device"], [2, "missing keccak"], [3, "missing include marker"]]) {
    test(`${label} rejects then permits a clean retry`, () => {execute(Number(mode), "");});
  }
  test("file and module lookup helpers are excluded only for the explicit native profile", () => {
    assert.match(read("sycl/kawpow/jit.inc"), /#if !defined\(MOM_EMBEDDED_SYCL_JIT\)[\s\S]*static std::string kawpow_read_file/);
  });
  test("the proof includes original program generation rather than copying its PRNG or kernels", () => {
    const source = read("tests/reference/native-kawpow-source.cpp");
    assert.match(source, /#include "\.\.\/\.\.\/sycl\/kawpow\/program.inc"/);
    assert.doesNotMatch(source, /KawpowProgram make_program\(|inline uint32_t kiss99/);
  });
  test("changing either maintained fragment invalidates its generated resource", () => {
    const source = read("tests/native/dpcpp-jit/CMakeLists.txt");
    for (const name of ["device.inc", "keccak.inc"]) {
      assert.ok(source.includes(`kawpow-${name} "\${MOM_SOURCE_REPO}/sycl/kawpow/${name}"`));
    }
    assert.match(source, /DEPENDS "\$\{MOM_SOURCE_REPO\}\/scripts\/native-resources.js"/);
  });
  test("normal Engineering source lookup remains available", () => {
    assert.match(read("sycl/kawpow/jit.inc"), /std::getenv\("MOM_KAWPOW_DEVICE_INC"\)/);
    assert.doesNotMatch(read("scripts/dev.js"), /MOM_EMBEDDED_SYCL_JIT/);
  });
});

test.describe("Native compiler ELF namespace isolation", {skip: process.platform !== "linux"}, () => {
  const {isolateCompilerObject, groupSignatures, compilerArchives} = require("../scripts/native-compiler-isolation");
  let root = "";
  /** @type {ReturnType<typeof isolateCompilerObject>} */
  let receipt;
  const cxx = process.env["CXX"] || "c++";
  /** @param {string} command @param {string[]} args */
  function capture(command, args) {
    const result = spawnSync(command, args, {encoding: "utf8", timeout: 30000, env: {...process.env, LC_ALL: "C"}});
    assert.equal(result.status, 0, result.error?.message || result.stderr); return result.stdout;
  }
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-compiler-isolation-"));
    const fixture = path.join(__dirname, "native/compiler-isolation.cpp");
    for (const [mode, value] of [["host", 41], ["library", 17]]) {
      capture(cxx, ["-std=c++17", "-O0", "-ffunction-sections", "-fdata-sections", `-DMOM_COMPILER_VALUE=${value}`,
        ...(mode === "host" ? ["-DMOM_COMPILER_HOST"] : []), "-c", fixture, "-o", path.join(root, mode + ".o")]);
    }
    receipt = isolateCompilerObject(path.join(root, "library.o"), path.join(root, "isolated"),
      {prefix: "mom_test_", entryPoints: ["isolated_compiler"]});
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  test("keeps only the original C bridge externally unprefixed", () => {
    assert.deepEqual(receipt.entryPoints, ["isolated_compiler"]);
    const names = capture("nm", ["-P", "-g", "--defined-only", receipt.object]).trim().split("\n").map((line) => line.split(/\s+/)[0] || "");
    assert.ok(names.includes("isolated_compiler"));
    assert.ok(names.every((name) => name === "isolated_compiler" || name.startsWith("mom_test_")));
  });
  test("local constructor COMDAT signatures are renamed in addition to global definitions", () => {
    const signatures = groupSignatures(capture("readelf", ["--wide", "--section-groups", receipt.object]));
    assert.ok(signatures.length && signatures.every((name) => name.startsWith("mom_test_")));
    assert.ok(receipt.groups > 0 && receipt.renamed > receipt.definitions - 1);
  });
  test("both compiler versions retain independent constructor semantics in one executable", () => {
    capture(cxx, [receipt.object, path.join(root, "host.o"), "-Wl,--gc-sections", "-o", path.join(root, "combined")]);
    assert.match(capture(path.join(root, "combined"), []), /Independent compiler constructors/);
  });
  test("the old globals-only isolation fails the same two-version control", () => {
    const definitions = capture("nm", ["-P", "-g", "--defined-only", path.join(root, "library.o")]);
    const map = definitions.trim().split("\n").map((line) => line.split(/\s+/)[0] || "")
      .filter((name) => name !== "isolated_compiler").map((name) => `${name} mom_old_${name}`).join("\n") + "\n";
    fs.writeFileSync(path.join(root, "old.map"), map);
    capture("objcopy", ["--redefine-syms=" + path.join(root, "old.map"), path.join(root, "library.o"), path.join(root, "old.o")]);
    const link = spawnSync(cxx, [path.join(root, "old.o"), path.join(root, "host.o"), "-o", path.join(root, "old")], {encoding: "utf8", timeout: 30000});
    if (link.status === 0) {
      assert.notEqual(spawnSync(path.join(root, "old"), [], {encoding: "utf8", timeout: 5000}).status, 0);
    } else {assert.match(link.stderr, /undefined reference|undefined symbol|discarded section/);}
  });
  test("preserves the original compiler object byte-for-byte", () => {
    const {createHash} = require("node:crypto");
    assert.equal(createHash("sha256").update(fs.readFileSync(path.join(root, "library.o"))).digest("hex"), receipt.inputSha256);
  });
  test("cannot overwrite an earlier isolation receipt", () => {
    assert.throws(() => isolateCompilerObject(path.join(root, "library.o"), path.join(root, "isolated"),
      {prefix: "mom_test_", entryPoints: ["isolated_compiler"]}), /EEXIST/);
  });
  test("requires the declared bridge to be defined by this graph", () => {
    assert.throws(() => isolateCompilerObject(path.join(root, "library.o"), path.join(root, "missing"),
      {prefix: "mom_test_", entryPoints: ["missing_entry"]}), /Missing compiler C bridge/);
  });
  test("rejects non-relocatable executables before objcopy", () => {
    assert.throws(() => isolateCompilerObject("/bin/true", path.join(root, "executable"),
      {prefix: "mom_test_", entryPoints: ["isolated_compiler"]}), /relocatable/);
  });
  test("rejects relative paths and malformed or duplicated bridge names", () => {
    const input = path.join(root, "library.o"), output = path.join(root, "invalid");
    for (const options of [{prefix: "bad prefix_", entryPoints: ["isolated_compiler"]},
      {prefix: "mom_test_", entryPoints: ["isolated_compiler", "isolated_compiler"]},
      {prefix: "mom_test_", entryPoints: []}, {prefix: "mom_test_", entryPoints: ["bad name"]}]) {
      assert.throws(() => isolateCompilerObject(input, output, options), /unique C entries/);
    }
    assert.throws(() => isolateCompilerObject("relative.o", path.join(root, "invalid"),
      {prefix: "mom_test_", entryPoints: ["isolated_compiler"]}), /absolute paths/);
    assert.equal(fs.existsSync(path.join(root, "invalid")), false);
  });
  test("cannot accidentally prefix an already isolated graph twice", () => {
    assert.throws(() => isolateCompilerObject(receipt.object, path.join(root, "twice"),
      {prefix: "mom_test_", entryPoints: ["isolated_compiler"]}), /already isolated/);
  });
  test("incomplete compiler graphs cannot leave unresolved LLVM entries", () => {
    const input = path.join(root, "incomplete.o");
    capture(cxx, ["-std=c++17", "-O0", "-DMOM_COMPILER_VALUE=17", "-DMOM_COMPILER_INCOMPLETE", "-c",
      path.join(__dirname, "native/compiler-isolation.cpp"), "-o", input]);
    assert.throws(() => isolateCompilerObject(input, path.join(root, "incomplete"),
      {prefix: "mom_test_", entryPoints: ["isolated_compiler"]}), /not self-contained/);
  });
  test("group parsing rejects unsupported or malformed selection keys", () => {
    assert.deepEqual(groupSignatures("There are no section groups in this file."), []);
    assert.throws(() => groupSignatures("group section [1] unknown\n"), /Unsupported ELF/);
    assert.throws(() => groupSignatures("COMDAT group section [1] `.group' [bad key] contains 1 sections:\n"), /Unsupported ELF/);
  });
  const archives = ["lib/libsycl-jit.a", ...Array.from({length: 50}, (_, index) => `lib/libLLVM${index}.a`)];
  const link = "clang++ fixture.o -o bin/native-dpcpp-jit " + archives.join(" ");
  test("reads the exact original target archive closure, ignoring other build commands", () => {
    assert.deepEqual(compilerArchives("ar q lib/unrelated.a other.o\n" + link, "/original"),
      archives.map((name) => path.join("/original", name)));
  });
  test("unrecognized or ambiguous compiler graphs fail closed without shell evaluation", () => {
    for (const command of [link + "\n" + link, link + " lib/libLLVM0.a", link.replace("lib/libLLVM0.a", "/external/libLLVM.a"),
      link.replace("lib/libLLVM0.a", "lib/../escape.a"), "clang++ -o bin/other lib/libsycl-jit.a"]) {
      assert.throws(() => compilerArchives(command, "/original"), /Unrecognized/);
    }
  });
  test("CLI admission requires build-only intent before running tools", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/native-compiler-isolation.js"), "/missing", "/missing"],
      {encoding: "utf8", timeout: 5000, env: {...process.env, MOM_AOT_BUILD_ONLY: "0", PATH: "/unavailable"}});
    assert.equal(result.status, 1); assert.match(result.stderr, /MOM_AOT_BUILD_ONLY=1/); assert.doesNotMatch(result.stderr, /ENOENT|spawnSync/);
  });
});

test.describe("Static DPC++ complete-artifact receipts", {skip: process.platform !== "linux"}, () => {
  const {auditBuild} = require("../scripts/native-dpcpp-pins");
  let root = "", serial = 0;
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-dpcpp-receipts-"));
    /** @type {Array<[string, string, string[]]>} */
    const programs = [
      ["clean", "int main(){return 0;}", ["-static-libstdc++", "-static-libgcc"]],
      ["node", "namespace node {void forbidden(){}} int main(){node::forbidden();}", ["-static-libstdc++", "-static-libgcc"]],
      ["shared", '#include <iostream>\nint main(){std::cout << "fixture";}', []],
    ];
    for (const [name, source, flags] of programs) {
      const run = spawnSync(process.env["CXX"] || "c++", ["-x", "c++", "-", ...flags, "-o", path.join(root, name)],
        {input: source, encoding: "utf8", timeout: 30000});
      assert.equal(run.status, 0, run.error?.message || run.stderr);
    }
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  function fixture() {
    const directory = path.join(root, String(++serial)); fs.mkdirSync(directory);
    for (const name of ["native-dpcpp-core", "native-static-adapter-selection", "native-umf-static", "native-level-zero-driver", "native-opencl-binding-static", "native-opencl-binding-dynamic", "native-opencl-driver"]) {
      fs.copyFileSync(path.join(root, "clean"), path.join(directory, name));
    }
    return directory;
  }
  test("the core-only proof does not accidentally admit an adapter artifact", () => {
    assert.equal(auditBuild(fixture(), false).artifacts.length, 1);
  });
  test("Level Zero audits the core, selection, allocator and full-loader boundary", () => {assert.equal(auditBuild(fixture(), true).artifacts.length, 4);});
  test("OpenCL audits both binding controls and the original ICD loader spy", () => {
    const {artifacts, sources} = auditBuild(fixture(), false, false, false, true);
    assert.equal(artifacts.length, 5);
    for (const name of ["runtime/dpcpp-static-opencl.patch", "tests/reference/native-opencl-binding.cpp", "tests/native/driver-library.cpp"]) {
      assert.match(sources[name] || "", /^[a-f0-9]{64}$/);
    }
  });
  test("an absent OpenCL loader spy fails closed", () => {
    const directory = fixture(); fs.unlinkSync(path.join(directory, "native-opencl-driver"));
    assert.throws(() => auditBuild(directory, false, false, false, true), /Input query failed/);
  });
  test("static adapter selection evidence is mandatory for every adapter profile", () => {
    const directory = fixture(); fs.unlinkSync(path.join(directory, "native-static-adapter-selection"));
    for (const profile of [[true, false, false], [false, true, false], [false, false, true]]) {
      assert.throws(() => auditBuild(directory, profile[0] === true, false, profile[1], profile[2]), /Input query failed/);
    }
  });
  test("a forbidden runtime cannot hide in either OpenCL binding control", () => {
    for (const name of ["native-opencl-binding-static", "native-opencl-binding-dynamic"]) {
      const directory = fixture(); fs.copyFileSync(path.join(root, "node"), path.join(directory, name));
      assert.throws(() => auditBuild(directory, false, false, false, true), /Node\/V8\/N-API/);
    }
  });
  test("an OpenCL loader spy cannot depend on application shared libraries", () => {
    const directory = fixture(); fs.copyFileSync(path.join(root, "shared"), path.join(directory, "native-opencl-driver"));
    assert.throws(() => auditBuild(directory, false, false, false, true), /application dependency/);
  });
  test("receipts include complete binary hashes, sizes and OS dependencies", () => {
    for (const artifact of auditBuild(fixture(), true).artifacts) {
      assert.match(artifact.sha256, /^[a-f0-9]{64}$/); assert.equal(artifact.bytes, fs.statSync(artifact.executable).size);
      assert.ok(artifact.needed.includes("libc.so.6"));
    }
  });
  test("provider patch and real allocation fixture are recorded with their inputs", () => {
    const {sources} = auditBuild(fixture(), true);
    for (const name of ["runtime/umf-static-level-zero.patch", "tests/reference/native-umf-static.cpp",
      "runtime/level-zero-static-complete.patch", "tests/reference/native-level-zero-static.cpp"]) {
      assert.match(sources[name] || "", /^[a-f0-9]{64}$/);
    }
  });
  test("source-JIT receipts include the runtime/compiler binding patch", () => {
    assert.match(auditBuild(fixture(), false, true).sources["runtime/dpcpp-static-jit.patch"] || "", /^[a-f0-9]{64}$/);
    assert.equal(auditBuild(fixture(), false).sources["runtime/dpcpp-static-jit.patch"], undefined);
  });
  test("CUDA receipts retain isolation, forwarding and dispatch-factory sources", () => {
    const {sources} = auditBuild(fixture(), false, true, true);
    for (const name of ["runtime/dpcpp-static-adapters.patch", "runtime/system-driver.h", "scripts/native-cuda-adapter.js",
      "scripts/native-compiler-isolation.js", "tests/native/dpcpp-static/cuda-factory.cpp"]) {
      assert.match(sources[name] || "", /^[a-f0-9]{64}$/);
    }
  });
  test("incremental audits preserve their predecessors", () => {
    const directory = fixture(), first = auditBuild(directory, true), second = auditBuild(directory, true);
    assert.notEqual(first.audit, second.audit); assert.deepEqual(JSON.parse(fs.readFileSync(path.join(first.audit, "result.json"), "utf8")), first);
  });
  test("a forbidden runtime in the secondary allocator cannot hide behind a clean core", () => {
    const directory = fixture(); fs.copyFileSync(path.join(root, "node"), path.join(directory, "native-umf-static"));
    assert.throws(() => auditBuild(directory, true), /Node\/V8\/N-API/);
  });
  test("an absent secondary artifact fails closed", () => {
    const directory = fixture(); fs.unlinkSync(path.join(directory, "native-umf-static"));
    assert.throws(() => auditBuild(directory, true), /Input query failed/);
  });
  test("a secondary application shared dependency fails closed", () => {
    const directory = fixture(); fs.copyFileSync(path.join(root, "shared"), path.join(directory, "native-umf-static"));
    assert.throws(() => auditBuild(directory, true), /application dependency/);
  });
  test("malformed executable data is not treated as an empty dependency list", () => {
    const directory = fixture(); fs.writeFileSync(path.join(directory, "native-dpcpp-core"), "not ELF");
    assert.throws(() => auditBuild(directory, false), /Input query failed/);
  });
  test("symbol and dependency evidence is retained for each artifact", () => {
    const {audit} = auditBuild(fixture(), true);
    for (const name of ["native-dpcpp-core", "native-umf-static"]) {
      assert.match(fs.readFileSync(path.join(audit, name + ".dependencies.log"), "utf8"), /NEEDED/);
      assert.match(fs.readFileSync(path.join(audit, name + ".symbols.log"), "utf8"), /\bmain\b/);
    }
  });
});

test.describe("Unused CUDA cleanup isolation", {skip: process.platform !== "linux"}, () => {
  /** @type {Array<[string, string]>} */
  const profiles = [["CN/GPU", "sycl/cn-gpu/cuda-jit.inc"], ["Pearl", "sycl/pearlhash/cuda-search.inc"]];
  for (const [name, file] of profiles) {
    test.describe(name, () => {
      let root = "";
      test.before(() => {
        root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cuda-cleanup-"));
        fs.writeFileSync(path.join(root, "release.inc"), extractCppBlock(read(file), "  void release() noexcept"));
        const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", `-DMOM_TEST_PEARL=${Number(name === "Pearl")}`,
          "-I", root, path.join(__dirname, "native/cuda-unused-cleanup.cpp"), "-o", path.join(root, "cleanup")], {encoding: "utf8", timeout: 30000});
        assert.equal(result.status, 0, result.error?.message || result.stderr);
      });
      test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
      /** @param {number} mode */
      function execute(mode) {
        const run = spawnSync(path.join(root, "cleanup"), [String(mode)], {encoding: "utf8", timeout: 5000});
        assert.equal(run.status, 0, run.error?.message || run.stderr);
      }
      for (const [mode, label] of ["unused", "module only", "completion event only", "context only", "retained primary only", "start event only", "all resources"].entries()) {
        test(`${label}: no foreign-driver probe and idempotent release`, () => {
          execute(mode);
        });
      }
    });
  }
});

test.describe("Native GPU source profiles", () => {
  const {gpuProfile} = require("../scripts/native-gpu-build");
  for (const backend of ["cuda", "hip", "cuda-hip"]) {
    test(`${backend} enables exactly its declared source-JIT APIs`, () => {
      const profile = gpuProfile(backend);
      assert.equal(profile.cuda, backend !== "hip"); assert.equal(profile.hip, backend !== "cuda");
      assert.equal(profile.definitions.includes("-DMOM_STATIC_NVRTC"), profile.cuda);
      assert.equal(profile.definitions.includes("-DMOM_SYCL_ADAPTIVECPP_CUDA"), profile.cuda);
      assert.equal(profile.definitions.includes("-DMOM_STATIC_HIPRTC"), profile.hip);
      assert.equal(profile.definitions.includes("-DMOM_SYCL_HAS_HIP"), profile.hip);
      assert.equal(profile.definitions.includes("-DMOM_OCTOPUS_HAS_SYCL_NATIVE"), profile.hip);
      // Octopus's NVIDIA tensor code still belongs to its existing compiler lane; linking CUDA
      // runtime support alone must not advertise that implementation in an SSCP translation unit.
      assert.ok(!profile.definitions.includes("-DMOM_SYCL_HAS_CUDA"));
      assert.equal(new Set(profile.definitions).size, profile.definitions.length);
    });
  }
  test("unreviewed or ambiguous profiles fail before SDK lookup or output creation", () => {
    for (const backend of ["", "auto", "hip-cuda", "cuda-hip-hip", "cuda,hip", "cuda-hip ", "cuda-hip-opencl"]) {
      assert.throws(() => gpuProfile(backend), /explicit cuda, hip or cuda-hip/);
    }
    if (process.platform !== "linux") {return;} // The executable producer is Linux-only.
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/native-gpu-build.js"),
      "/unavailable/source", "/unavailable/build", "/unavailable/output"], {encoding: "utf8", timeout: 5000,
      env: {...process.env, MOM_AOT_SYCL_BACKEND: "auto", PATH: "/unavailable"}});
    assert.equal(result.status, 1); assert.match(result.stderr, /explicit cuda, hip or cuda-hip/);
    assert.doesNotMatch(result.stderr, /ENOENT|spawn|mkdir/);
  });
});

test.describe("Native GPU Core build boundary", () => {
  const {syclSources} = require("../scripts/native-gpu-build");
  const gyp = read("binding.gyp");
  test("the native build reads the maintained complete base source list", () => {
    const sources = syclSources(gyp);
    assert.equal(sources.length, 21);
    for (const name of ["lib", "pearlhash/pearlhash", "cn-gpu/cn-gpu", "kawpow/kawpow", "c29/c29", "c30/c30"]) {
      assert.ok(sources.includes(`sycl/${name}.cpp`));
    }
  });
  test("gyp's unrelated executable expressions are not evaluated or parsed as JSON", () => {
    assert.equal(syclSources(`arbitrary Python input\n${gyp}`).length, 21);
  });
  test("missing or ambiguous target registries fail closed", () => {
    assert.throws(() => syclSources("{}"), /Unrecognized/);
    assert.throws(() => syclSources(gyp + gyp), /Unrecognized/);
  });
  test("an incomplete source list is not replaced by algorithm stubs", () => {
    assert.throws(() => syclSources('"target_name":"sycl", "sources":["sycl/lib.cpp"]'), /Unrecognized/);
  });
  test("source paths cannot escape the maintained SYCL directory", () => {
    for (const file of ["../sycl/lib.cpp", "sycl/../../lib.cpp", "/tmp/lib.cpp", "sycl/lib.cpp;bad"]) {
      assert.throws(() => syclSources(gyp.replace('"sycl/lib.cpp"', JSON.stringify(file))), /Unrecognized/);
    }
  });
  test("duplicate sources cannot silently introduce duplicate registrations", () => {
    assert.throws(() => syclSources(gyp.replace('"sycl/lib.cpp"', '"sycl/lib.cpp", "sycl/lib.cpp"')), /Unrecognized/);
  });
  test("full Core mode excludes generated CPU-proof GPU stubs", () => {
    const build = read("scripts/build-native-core.js");
    assert.match(build, /if \(!gpu\) \{[\s\S]*gpu-unavailable\.cpp[\s\S]*commands\.push\(/);
    assert.match(build, /const commands = cpuCompileCommands\(repo, output, flags, gpu, compiler\)/);
  });
  test("vendor CPU bitcode is materialized with its original compiler before GPU linking", () => {
    const build = read("scripts/build-native-core.js");
    assert.match(build, /magic !== "4243c0de" && magic !== "dec0170b"/);
    assert.match(build, /run\(compiler, \["-c", "-x", "ir"/);
  });
  test("the GPU linker receives flattened archive members, not a merged relocatable Core", () => {
    const build = read("scripts/build-native-core.js");
    assert.match(build, /gpu \? "cpu-core\.a" : "cpu-core\.o"/);
    assert.match(build, /if \(gpu\) \{\s*if \(!archiver\)[\s\S]*?run\(archiver\.path, \["qcLs"/);
    assert.match(build, /selectCpuArchiver\(compiler/);
    assert.match(build, /verifyCpuArchiver\(archiver\)[\s\S]*?finally \{\s*verifyCpuArchiver\(archiver\);\s*\}/);
  });
  test("controller integration requires explicit build-only admission and stub-free provenance", () => {
    const probe = read("scripts/build-native-application.js");
    assert.match(probe, /gpuManifest && \(!cpuCore \|\| process\.env\["MOM_AOT_BUILD_ONLY"\] !== "1"\)/);
    assert.match(probe, /provenance\.gpu !== true/);
    assert.match(probe, /file\.includes\("gpu-unavailable"\)/);
  });
  test("native mode retires SYCL state after draining Core workers", () => {
    const host = read("runtime/host.cpp");
    assert.match(host, /release_console_bindings\(SHConsoleContext \*context\) \{\s*mom::runtime::closeLoop\(context\);[\s\S]*?sycl_cleanup\(\);/);
  });
  test("the complete graph and primitive use the same reviewed runtime configuration", () => {
    const build = read("scripts/native-gpu-build.js");
    assert.match(build, /require\("\.\/native-sycl-inputs"\)/);
    assert.match(build, /staticSyclInputs\(source, build, output, backend\)/);
    assert.doesNotMatch(build, /spawnSync\([^\n]*(?:docker|nvidia-smi)|-lgomp|-lcudart\b/);
  });
  test("CN/GPU source JIT retains matching static compiler components, not a shared loader", () => {
    const build = read("scripts/native-gpu-build.js");
    assert.match(build, /\["nvrtc_static", "nvrtc-builtins_static", "nvptxcompiler_static"\]/);
    assert.match(build, /cuda && \["sycl\/cn-gpu\/cn-gpu\.cpp", "sycl\/pearlhash\/pearlhash\.cpp"\]\.includes\(file\) \? \["-DMOM_SYCL_HAS_CUDA"\]/);
    assert.match(build, /inputs\.libraries\.push\(file\)/);
  });
  test("every native GPU translation unit uses the same compiler API definition", () => {
    const build = read("scripts/native-gpu-build.js");
    const common = build.slice(build.indexOf("const compileFlags"), build.indexOf("function compile("));
    assert.match(common, /\.\.\.definitions/);
    assert.equal((build.match(/"-DMOM_STATIC_NVRTC"/g) || []).length, 1);
  });
  test("static NVIDIA compiler links retain legacy constructor execution", {
    skip: process.platform !== "linux" ? "ELF linker policy" : false,
  }, (t) => {
    const linker = fs.existsSync("/usr/lib/llvm-21/bin/ld.lld") ? "/usr/lib/llvm-21/bin/ld.lld" : "ld.lld";
    const available = spawnSync(linker, ["--version"], {encoding: "utf8"});
    if (available.error) {t.skip("LLD is not installed"); return;}
    assert.equal(available.status, 0, available.stderr);
    const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "mom-legacy-ctors-"));
    try {
      const executable = path.join(temporary, "constructors");
      const source = path.join(__dirname, "native/static-constructors.cpp");
      for (const fixed of [false, true]) {
        const build = spawnSync("c++", ["-std=c++17", "-O2", ...(path.isAbsolute(linker) ? ["-B", path.dirname(linker)] : []),
          "-fuse-ld=lld", "-Wl,--gc-sections",
          ...(fixed ? ["-Wl,-T," + path.join(__dirname, "../runtime/static-constructors.ld")] : []), source, "-o", executable], {encoding: "utf8", timeout: 30000});
        assert.equal(build.status, 0, build.error?.message || build.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
        // Some newer linkers may already handle .ctors; always require fixed behavior, while
        // recording the observed baseline rather than asserting that every tool must be broken.
        if (fixed) {assert.equal(result.status, 0, result.error?.message || result.stderr);} else {assert.ok(result.status === 0 || result.status === 1, result.stderr);}
      }
    } finally {fs.rmSync(temporary, {recursive: true, force: true});}
    assert.match(read("scripts/native-gpu-build.js"), /inputs\.links\.push\("-Wl,-T," \+ path\.join\(repo, "runtime\/static-constructors\.ld"\)\)/);
  });
  test("the native policy embeds the maintained Markdown, not a parallel configuration table", () => {
    const probe = read("scripts/build-native-application.js");
    assert.match(probe, /fs\.readFileSync\(path\.join\(repo, "GPU-CONFIG\.md"\)\)/);
    assert.match(probe, /nativeResources\(\[\{name: "GPU-CONFIG\.md", data: policy\}\]\)/);
  });
  test("only the exact built-in policy path bypasses filesystem reads", () => {
    const services = read("runtime/services.cpp");
    assert.match(services, /if \(path == kEmbeddedPolicyPath\) \{\s*const auto bytes = embedded_resource\("GPU-CONFIG\.md"\)/);
    assert.doesNotMatch(services, /(?:basename|ends_with)\([^\n]*GPU-CONFIG/);
  });
  test("embedded defaults remain optional outside the native controller build", () => {
    const services = read("runtime/services.cpp");
    assert.match(services, /#ifdef MOM_EMBEDDED_APP_RESOURCES\s*#include "app-resources\.inc"\s*#endif/);
    assert.match(read("scripts/build-native-application.js"), /name === "services" \? \["-DMOM_EMBEDDED_APP_RESOURCES"/);
  });
  test("native source-JIT binds the selected nonzero device in both compiler families", {
    skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
  }, () => {
    const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ordinal-"));
    try {
      const methods = [["CnSelection", "sycl/cn-gpu/cuda-jit.inc"], ["PearlSelection", "sycl/pearlhash/cuda-search.inc"]]
        .map(([name, file]) => {
          if (!file) {throw new Error("Missing native source");}
          const method = read(file).match(/static CUdevice native_device\(const sycl::device& device\) \{[\s\S]*?#endif\s*\}/)?.[0];
          assert.ok(method, "production method not found");
          return `struct ${name} {${method}};`;
        }).join("\n");
      fs.writeFileSync(path.join(temporary, "selection.inc"), methods);
      const executable = path.join(temporary, "selection");
      for (const flags of [[], ["-DMOM_SYCL_ADAPTIVECPP"]]) {
        const build = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", ...flags,
          "-I", temporary, path.join(__dirname, "native/cuda-selected-device.cpp"), "-o", executable], {encoding: "utf8", timeout: 30000});
        assert.equal(build.status, 0, build.error?.message || build.stderr);
        const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
        assert.equal(result.status, 0, result.error?.message || result.stderr);
        assert.match(result.stdout, /Selected native CUDA ordinals/);
      }
    } finally {fs.rmSync(temporary, {recursive: true, force: true});}
  });
});
