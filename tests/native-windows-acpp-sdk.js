"use strict";
const fs = require("node:fs"), path = require("node:path"), assert = require("node:assert/strict"), test = require("node:test");
const {windowsAcppStaticHostPlan, verifyAcppCache, auditAcppCompileCommands, buildWindowsAcppStaticHost,
  verifyIntelCompilerIdentity, verifyWindowsRocmResources, nativeRuntimeInputs, cleanEnvironment, targets, artifacts, forbiddenOutputs, admitSource, archiveObjectInputs} = require("../scripts/build-native-windows-acpp-sdk");
const {commandSwitches} = require("../scripts/build-native-windows-llvm-sdk");
const {windowsIntelRuleOverride} = require("../scripts/prepare-native-jit-toolchain");
const {fileHash} = require("../scripts/build-source-inputs");
const {tempDir} = require("./common/temp-dir");

/** Small explicit inputs; no CMake, compiler, VM, network or GPU is invoked. */
/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = tempDir(t, "mom-windows-acpp-plan-");
  const file = (/** @type {string} */ name) => path.join(root, name);
  const include = file("windows/include"), libraries = file("windows/lib"), resource = file("clang-resource/include");
  for (const directory of [include, libraries, resource]) {fs.mkdirSync(directory, {recursive: true});}
  const header = path.join(include, "windows.h"); fs.writeFileSync(header, "fixture\n");
  const overlay = file("windows/vfs.json");
  fs.writeFileSync(overlay, JSON.stringify({version: 0, "case-sensitive": false, roots: [
    {type: "file", name: path.join(include, "windows.h"), "external-contents": header},
  ]}));
  const lib = (/** @type {string} */ name) => path.join(libraries, name);
  const descriptor = {version: 1,
    tools: {clangCl: "/opt/intel/bin/icx-cl", lldLink: "/not-used/descriptor/lld-link",
      llvmNm: "/tools/llvm-nm", clangResourceInclude: resource},
    windowsSdk: {overlay, sourceRoot: file("windows"), includeRoots: [include], libraryRoots: [libraries],
      msvcLibraries: [lib("libcmt.lib")], ucrtLibraries: [lib("libucrt.lib")], importLibraries: [lib("kernel32.lib")]}};
  const descriptorFile = file("windows-descriptor.json"); fs.writeFileSync(descriptorFile, JSON.stringify(descriptor));
  const input = {source: file("adaptivecpp"), descriptorFile, descriptor, targetLlvm: file("windows-llvm21-output"),
    nativeTools: file("linux-llvm21/bin"), cuda: file("cuda-root"), cudartStatic: file("windows/cuda/cudart_static.lib"),
    cudaDeviceLibs: file("windows/cuda/nvvm/libdevice"), rocm: file("rocm-root"), rocmReceipt: file("rocm-root/result.json"),
    amdhip64: file("rocm-root/lib/amdhip64.lib"), rocmDeviceLibs: file("rocm-root/amdgcn/bitcode"), output: file("new-output"),
    jobs: 2, tools: {cmake: "/tools/cmake", ninja: "/tools/ninja"}};
  return {root, input, descriptor, header};
}

test.describe("Windows AdaptiveCpp static host archive recipe", {skip: process.platform !== "linux" && "Linux Windows cross-build plan"}, () => {
  test("accepts an empty canonical SDK runtime directory without an OS library glob", t => {
    const f = fixture(t), plan = windowsAcppStaticHostPlan(f.input), runtime = plan.nativeRuntimeRoots[0];
    assert.deepEqual(plan.nativeRuntimeRoots, [path.join(path.dirname(f.input.nativeTools), "lib")]);
    assert.ok(runtime);
    fs.mkdirSync(runtime, {recursive: true});
    assert.deepEqual(nativeRuntimeInputs(plan), []);
    const env = cleanEnvironment({LD_LIBRARY_PATH: "/attacker/loader"}, plan.nativeRuntimeRoots);
    assert.equal(env["LD_LIBRARY_PATH"], runtime);
    assert.ok(!env["LD_LIBRARY_PATH"].includes("/opt/ubuntu24-libs"));
    assert.ok(!env["LD_LIBRARY_PATH"].includes("/attacker/loader"));
  });

  test("inventories real native libraries without admitting dangling or escaping aliases", t => {
    const root = tempDir(t, "mom-acpp-runtime-");
    const runtime = path.join(root, "lib"), inside = path.join(runtime, "libLLVM.so.21");
    fs.mkdirSync(runtime); fs.writeFileSync(inside, "runtime payload");
    fs.symlinkSync("absent.so", path.join(runtime, "liboptional.so"));
    const plan = {...windowsAcppStaticHostPlan(fixture(t).input), nativeRuntimeRoots: [runtime]};
    assert.deepEqual(nativeRuntimeInputs(plan), [inside]);
    const cycle = path.join(runtime, "libcycle.so"); fs.symlinkSync("libcycle.so", cycle);
    assert.throws(() => nativeRuntimeInputs(plan), error => error instanceof Error &&
    /** @type {NodeJS.ErrnoException} */ (error).code === "ELOOP");
    fs.unlinkSync(cycle);
    const outside = path.join(root, "outside.so"); fs.writeFileSync(outside, "foreign payload");
    fs.symlinkSync(outside, path.join(runtime, "libescape.so"));
    assert.throws(() => nativeRuntimeInputs(plan), /escapes its selected root/);
  });

  test("selects exactly the nine upstream archives, not the plugin DLL or CLI tools", t => {
    const plan = windowsAcppStaticHostPlan(fixture(t).input);
    assert.deepEqual(plan.targets, ["acpp-rt", "rt-backend-cuda", "rt-backend-hip", "rt-backend-omp", "acpp-common",
      "llvm-to-backend", "llvm-to-ptx", "llvm-to-amdgpu", "llvm-to-host"]);
    assert.deepEqual(plan.artifacts.map(file => path.relative(plan.build, file).split(path.sep).join("/")), [...artifacts]);
    assert.ok(plan.compile.includes("--target"));
    assert.deepEqual(plan.compile.slice(plan.compile.indexOf("--target") + 1, plan.compile.indexOf("--")), [...targets]);
    assert.equal(plan.forbiddenOutputs.length, forbiddenOutputs.length);
    assert.doesNotMatch(plan.compile.join(" "), /acpp-clang|\.exe|--target all/);
  });

  test("keeps the target Windows LLVM exports separate from native Linux LLVM tools", t => {
    const f = fixture(t), plan = windowsAcppStaticHostPlan(f.input);
    assert.equal(plan.targetLlvmDir, path.join(f.input.targetLlvm, "build/lib/cmake/llvm"));
    assert.equal(plan.llvmDir, path.join(f.input.output, "llvm-cmake"));
    assert.ok(plan.llvmConfigText.includes(path.join(plan.targetLlvmDir, "LLVMConfig.cmake")));
    assert.ok(plan.llvmConfigText.includes(`set(LLVM_TOOLS_BINARY_DIR [==[${f.input.nativeTools}]==])`));
    assert.ok(plan.llvmVersionText.includes(path.join(plan.targetLlvmDir, "LLVMConfigVersion.cmake")));
    assert.equal(plan.lldDir, path.join(f.input.targetLlvm, "build/lib/cmake/lld"));
    assert.equal(plan.definitions.LLVM_NATIVE_TOOL_DIR, f.input.nativeTools);
    assert.equal(plan.definitions.CLANG_EXECUTABLE_PATH, path.join(f.input.nativeTools, "clang"));
    assert.equal(plan.definitions.ACPP_LLC_PATH, path.join(f.input.nativeTools, "llc"));
    assert.equal(plan.definitions.ACPP_LLD_PATH, path.join(f.input.nativeTools, "lld-link"));
    assert.equal(plan.definitions.ACPP_OPT_PATH, path.join(f.input.nativeTools, "opt"));
    assert.equal(plan.hostCompilers.cc, f.descriptor.tools.clangCl); assert.equal(plan.hostCompilers.cxx, f.descriptor.tools.clangCl);
    assert.equal(plan.config.tools.lldLink, f.descriptor.tools.lldLink);
    assert.equal(plan.compilerTools["CMAKE_LINKER"], path.join(f.input.nativeTools, "lld-link"));
    assert.ok(plan.toolchain.includes(path.join(f.input.nativeTools, "lld-link")));
    assert.ok(Object.values(plan.native).every(file => file.startsWith(f.input.nativeTools + path.sep)),
      "native tools are never guessed from the target archive-only root");
  });

  test("accepts the thin Windows LLVM descriptor without controller-only Hermes or library fields", t => {
    const f = fixture(t), plan = windowsAcppStaticHostPlan(f.input);
    assert.equal(Object.hasOwn(f.descriptor, "hermes"), false);
    assert.equal(Object.hasOwn(f.descriptor, "libraries"), false);
    assert.deepEqual(Object.keys(plan.config).sort(), ["tools", "version", "windowsSdk"]);
    assert.deepEqual(plan.config.windowsSdk.msvcLibraries, [path.join(f.root, "windows/lib/libcmt.lib")]);
  });

  test("uses the admitted source/build cache separately from a refreshed LLVM output", t => {
    const f = fixture(t), targetLlvmLayout = {source: path.join(f.root, "original-cache", "source"),
        build: path.join(f.root, "original-cache", "build")}, plan = windowsAcppStaticHostPlan(f.input, targetLlvmLayout);
    assert.equal(plan.targetLlvmSource, targetLlvmLayout.source);
    assert.equal(plan.targetBuild, targetLlvmLayout.build);
    assert.equal(plan.targetLlvmDir, path.join(targetLlvmLayout.build, "lib/cmake/llvm"));
    assert.equal(plan.lldDir, path.join(targetLlvmLayout.build, "lib/cmake/lld"));
    assert.ok(plan.llvmConfigText.includes(path.join(targetLlvmLayout.build, "lib/cmake/llvm/LLVMConfig.cmake")));
    assert.ok(plan.llvmVersionText.includes(path.join(targetLlvmLayout.build, "lib/cmake/llvm/LLVMConfigVersion.cmake")));
    assert.ok(plan.includeFlags.includes("-I" + path.join(targetLlvmLayout.source, "clang/include")));
    assert.ok(plan.includeFlags.includes("-I" + path.join(targetLlvmLayout.build, "tools/clang/include")));
    assert.equal(windowsAcppStaticHostPlan(f.input).targetBuild, path.join(f.input.targetLlvm, "build"),
      "ordinary non-refresh admissions keep their existing layout");
  });

  test("uses the shared Intel-ID-gated Windows archive and link rules", t => {
    const f = fixture(t), plan = windowsAcppStaticHostPlan(f.input);
    assert.ok(plan.ruleOverrideText.startsWith(windowsIntelRuleOverride(plan.linkFlags,
      [...plan.config.windowsSdk.msvcLibraries, ...plan.config.windowsSdk.ucrtLibraries, ...plan.config.windowsSdk.importLibraries]) +
      "set(CMAKE_INCLUDE_SYSTEM_FLAG_C /external:I)\nset(CMAKE_INCLUDE_SYSTEM_FLAG_CXX /external:I)\n"));
    assert.match(plan.toolchain, /CMAKE_USER_MAKE_RULES_OVERRIDE_C/);
    assert.match(plan.toolchain, /CMAKE_USER_MAKE_RULES_OVERRIDE_CXX/);
    assert.match(plan.ruleOverrideText, /if\(NOT CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM"/);
    assert.match(plan.ruleOverrideText, /CMAKE_CXX_CREATE_STATIC_LIBRARY/);
    assert.match(plan.ruleOverrideText, /set\(CMAKE_INCLUDE_SYSTEM_FLAG_C \/external:I\)/);
    assert.match(plan.ruleOverrideText, /set\(CMAKE_INCLUDE_SYSTEM_FLAG_CXX \/external:I\)/);
    assert.equal(plan.definitions.CMAKE_INCLUDE_SYSTEM_FLAG_C, "/external:I");
    assert.equal(plan.definitions.CMAKE_INCLUDE_SYSTEM_FLAG_CXX, "/external:I");
    assert.match(plan.toolchain, /CMAKE_INCLUDE_SYSTEM_FLAG_C .*CACHE STRING/);
    assert.match(plan.toolchain, /CMAKE_INCLUDE_SYSTEM_FLAG_CXX .*CACHE STRING/);
    assert.doesNotMatch(plan.toolchain + plan.ruleOverrideText, /-imsvc|-isystem/i);
    assert.doesNotMatch(plan.toolchain + plan.ruleOverrideText, /set\s*\(\s*CMAKE_(?:C|CXX)_COMPILER_ID|SKIP_(?:C|CXX)_COMPILER/);
    assert.equal(plan.compilerTools["CMAKE_C_COMPILER"], plan.hostCompilers.cc);
    assert.equal(plan.compilerTools["CMAKE_CXX_COMPILER"], plan.hostCompilers.cxx);
    assert.equal(plan.compilerTools["CMAKE_AR"], path.join(f.input.output, "bin/llvm-lib"));
  });

  test("pins the static MT Windows ABI, C++ EH/RTTI, and sequential OpenMP profile", t => {
    const f = fixture(t), plan = windowsAcppStaticHostPlan(f.input), configure = plan.configure.join(" ");
    for (const [name, value] of Object.entries({ACPP_STATIC_RUNTIME: "ON", ACPP_COMPILER_FEATURE_PROFILE: "custom-deprecated",
      ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER: "ON", ACPP_CUSTOM_PROFILE_WITH_STDPAR_COMPILER: "OFF",
      ACPP_CUSTOM_PROFILE_WITH_ACCELERATED_CPU: "OFF", WITH_CUDA_BACKEND: "ON", WITH_ROCM_BACKEND: "ON",
      WITH_OPENCL_BACKEND: "OFF", WITH_LEVEL_ZERO_BACKEND: "OFF", WITH_VULKAN_BACKEND: "OFF", WITH_METAL_BACKEND: "OFF",
      CMAKE_MSVC_RUNTIME_LIBRARY: "MultiThreaded", CMAKE_DISABLE_FIND_PACKAGE_HIP: "TRUE",
      CMAKE_DISABLE_FIND_PACKAGE_OpenCL: "TRUE", CMAKE_DISABLE_FIND_PACKAGE_OpenMP: "TRUE"})) {
      assert.equal(plan.definitions[name], value, name);
    }
    assert.match(plan.toolchain, /x86_64-pc-windows-msvc/);
    assert.ok(plan.includeFlags.includes("-DNOMINMAX"), "Windows headers must not replace std::min/max in the OMP backend");
    assert.ok(plan.includeFlags.includes("-DWIN32_LEAN_AND_MEAN"));
    assert.match(plan.toolchain, /"\/MT"/); assert.match(plan.toolchain, /"\/EHsc"/); assert.match(plan.toolchain, /"\/GR"/);
    assert.match(plan.toolchain, /CMAKE_MSVC_RUNTIME_LIBRARY \[==\[MultiThreaded\]==\]/);
    assert.match(plan.toolchain, /CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE/);
    assert.equal(plan.definitions.CMAKE_TRY_COMPILE_CONFIGURATION, "Release");
    assert.match(plan.ruleOverrideText, /set\(CMAKE_C_FLAGS_RELEASE_INIT "\/O2 \/DNDEBUG"\)/);
    assert.match(plan.ruleOverrideText, /set\(CMAKE_CXX_FLAGS_RELEASE_INIT "\/O2 \/DNDEBUG"\)/);
    const exactLibraries = [...f.descriptor.windowsSdk.msvcLibraries, ...f.descriptor.windowsSdk.ucrtLibraries,
      ...f.descriptor.windowsSdk.importLibraries].map(file => '"' + file + '"').join(" ");
    for (const language of ["C", "CXX"]) {
      const link = plan.ruleOverrideText.split("\n").find(line => line.startsWith(`set(CMAKE_${language}_LINK_EXECUTABLE `));
      assert.ok(link?.includes(exactLibraries));
    }
    assert.doesNotMatch(configure + plan.toolchain, /libgomp|fopenmp|\/MD\b| -static-libgcc|--sysroot=/i);
  });

  test("keeps native resource generation on Node and feeds only explicit CUDA/HIP SDK roots", t => {
    const f = fixture(t), plan = windowsAcppStaticHostPlan(f.input);
    assert.equal(plan.definitions.ACPP_RESOURCE_GENERATOR, path.resolve(__dirname, "../scripts/native-resources.js"));
    assert.equal(plan.definitions.ACPP_RESOURCE_NODE, process.execPath);
    assert.equal(plan.definitions.CUDA_TOOLKIT_ROOT_DIR, f.input.cuda);
    assert.equal(plan.definitions.CUDA_CUDART_STATIC_LIBRARY, f.input.cudartStatic);
    assert.equal(plan.definitions.CUDA_CUDART, f.input.cudartStatic);
    assert.equal(plan.definitions.CUDAToolkit_BIN_DIR, path.join(f.input.cuda, "bin"));
    assert.equal(plan.definitions.CUDAToolkit_LIBRARY_ROOT, f.input.cuda);
    assert.equal(plan.definitions.CUDA_DEVICE_LIBS_PATH, f.input.cudaDeviceLibs);
    assert.equal(plan.definitions.ROCM_PATH, f.input.rocm); assert.equal(Object.hasOwn(plan.definitions, "HIPCC_COMPILER"), false);
    assert.equal(plan.definitions.AMDHIP64_LIBRARY, f.input.amdhip64);
    assert.equal(plan.definitions.ROCM_DEVICE_LIBS_PATH, f.input.rocmDeviceLibs);
    assert.ok(plan.toolchain.includes('"-I' + path.join(f.input.cuda, "include") + '"'));
    assert.ok(plan.toolchain.includes('"-I' + path.join(f.input.rocm, "include") + '"'));
    assert.ok(plan.toolchain.includes('"-I' + path.join(f.input.targetLlvm, "source/clang/include") + '"'));
    assert.ok(plan.toolchain.includes('"-I' + path.join(f.input.targetLlvm, "build/tools/clang/include") + '"'));
    assert.deepEqual(plan.compile.slice(-3), ["--", "-k", "0"]);
    assert.doesNotMatch(plan.toolchain, /cuda-root.*cudart_static|rocm-root.*amdhip64/i,
      "runtime archives remain separate explicit inputs, not guessed under toolkit roots");
  });

  test("binds the historical Windows HIP pins and exact bitcode subset of the resource receipt", t => {
    const root = tempDir(t, "mom-windows-rocm-resources-");
    const deviceRoot = path.join(root, "install/amdgcn/bitcode"), configFile = path.join(root, "install/lib/cmake/AMDDeviceLibs/AMDDeviceLibsConfig.cmake");
    fs.mkdirSync(deviceRoot, {recursive: true}); fs.mkdirSync(path.dirname(configFile), {recursive: true});
    const bitcode = ["hip.bc", "ockl.bc", "ocml.bc", "oclc_abi_version_600.bc"].map(name => path.join(deviceRoot, name));
    for (const file of [...bitcode, configFile]) {fs.writeFileSync(file, "fixture " + path.basename(file));}
    /** @param {string[]} files */
    const hashes = files => Object.fromEntries(files.map(file => [file, fileHash(file)]));
    const receiptFile = path.join(root, "result.json"), receipt = {version: 1, target: "win32",
      revision: "27682a16360e33e37c4f3cc6adf9a620733f8fe1", clrRevision: "5b591b2ff64aac4c1c08be879b0d5f89fd713d50",
      source: {"hip/src/hip.cpp": "a".repeat(64)}, inputs: hashes([configFile]), dependencies: hashes([configFile]),
      device: hashes([...bitcode, configFile]), header: {path: path.join(root, "hipRTC_header.h"), sha256: "b".repeat(64)},
      object: {path: path.join(root, "hipRTC_header.obj"), sha256: "c".repeat(64)}, buildOnly: true, publicationQualified: false};
    fs.writeFileSync(receiptFile, JSON.stringify(receipt));
    const verified = verifyWindowsRocmResources(receiptFile, deviceRoot);
    assert.deepEqual(verified.files, [...bitcode].sort());
    assert.equal(Object.keys(verified.device).length, 5, "the unrelated installed CMake config is not device bitcode");
    const changed = bitcode[0];
    if (!changed) {throw new Error("Missing bitcode fixture");}
    fs.appendFileSync(changed, "changed");
    assert.throws(() => verifyWindowsRocmResources(receiptFile, deviceRoot), /differs from its Windows target receipt/);
  });

  test("rejects unsafe paths, overlapping outputs, archive suffixes and job counts", t => {
    const f = fixture(t);
    for (const invalid of ["relative", "/bad;list", "/bad" + String.fromCharCode(36) + "{INJECT}", "/bad\\root", "/bad\nroot"]) {
      assert.throws(() => windowsAcppStaticHostPlan({...f.input, cuda: invalid}), /absolute literal/);
    }
    assert.throws(() => windowsAcppStaticHostPlan({...f.input, cudartStatic: "/cuda/cudart_static.a"}), /COFF archives/);
    assert.throws(() => windowsAcppStaticHostPlan({...f.input, output: path.join(f.input.source, "output")}), /overlaps/);
    for (const jobs of [0, 17, 1.5, NaN, Infinity]) {assert.throws(() => windowsAcppStaticHostPlan({...f.input, jobs}), /jobs/);}
    for (const jobs of [1, 2, 16]) {assert.equal(windowsAcppStaticHostPlan({...f.input, jobs}).compile[4], String(jobs));}
  });

  test("CMake cache proof rejects drift from the pinned Windows static profile", t => {
    const f = fixture(t), plan = windowsAcppStaticHostPlan(f.input), source = f.input.source, build = plan.build;
    fs.mkdirSync(source); fs.mkdirSync(build, {recursive: true});
    fs.writeFileSync(plan.toolchainFile, plan.toolchain);
    const cache = path.join(f.root, "CMakeCache.txt"), values = {...plan.definitions,
      CMAKE_C_COMPILER: plan.hostCompilers.cc, CMAKE_CXX_COMPILER: plan.hostCompilers.cxx,
      CMAKE_C_COMPILER_TARGET: "x86_64-pc-windows-msvc", CMAKE_CXX_COMPILER_TARGET: "x86_64-pc-windows-msvc",
      CMAKE_TRY_COMPILE_TARGET_TYPE: "EXECUTABLE", CMAKE_HOME_DIRECTORY: fs.realpathSync(source),
      ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER: "ON", ACPP_CUSTOM_PROFILE_WITH_STDPAR_COMPILER: "OFF",
      ACPP_CUSTOM_PROFILE_WITH_ACCELERATED_CPU: "OFF", WITH_CUDA_BACKEND: "ON", WITH_ROCM_BACKEND: "ON",
      WITH_OPENCL_BACKEND: "OFF", WITH_VULKAN_BACKEND: "OFF", OMP_LINK_LINE: "", SEQUENTIAL_LINK_LINE: "",
      OMP_CXX_FLAGS: "-D_ENABLE_EXTENDED_ALIGNED_STORAGE", SEQUENTIAL_CXX_FLAGS: "-D_ENABLE_EXTENDED_ALIGNED_STORAGE"};
    fs.writeFileSync(cache, Object.entries(values).map(([name, value]) => `${name}:STRING=${value}`).join("\n") +
      "\nCMAKE_DISABLE_FIND_PACKAGE_HIP:UNINITIALIZED=TRUE\nCMAKE_DISABLE_FIND_PACKAGE_OpenCL:UNINITIALIZED=TRUE\nCMAKE_DISABLE_FIND_PACKAGE_OpenMP:UNINITIALIZED=TRUE\n");
    assert.doesNotThrow(() => verifyAcppCache(cache, plan, source));
    fs.appendFileSync(cache, "CMAKE_SYSTEM_NAME:STRING=Linux\n");
    assert.throws(() => verifyAcppCache(cache, plan, source), /CMAKE_SYSTEM_NAME/);
    fs.writeFileSync(cache, fs.readFileSync(cache, "utf8").replace("CMAKE_SYSTEM_NAME:STRING=Linux\n", ""));
    fs.appendFileSync(cache, "CMAKE_INCLUDE_SYSTEM_FLAG_CXX:STRING=-imsvc\n");
    assert.throws(() => verifyAcppCache(cache, plan, source), /CMAKE_INCLUDE_SYSTEM_FLAG_CXX/);
    fs.writeFileSync(cache, fs.readFileSync(cache, "utf8").replace("CMAKE_INCLUDE_SYSTEM_FLAG_CXX:STRING=-imsvc\n", ""));
    fs.writeFileSync(cache, fs.readFileSync(cache, "utf8").replace("OMP_LINK_LINE:STRING=", "OMP_LINK_LINE:STRING=-fopenmp"));
    assert.throws(() => verifyAcppCache(cache, plan, source), /OMP_LINK_LINE/);
  });

  test("compile database audit enforces COFF target, MT, EH, RTTI and no GNU OpenMP", t => {
    const f = fixture(t), source = f.input.source, build = path.join(f.input.output, "build"), compiler = f.descriptor.tools.clangCl;
    fs.mkdirSync(path.join(source, "src"), {recursive: true}); fs.mkdirSync(build, {recursive: true});
    const cpp = path.join(source, "src", "runtime.cpp"); fs.writeFileSync(cpp, "int f() { return 0; }\n");
    const commandFile = path.join(f.root, "compile_commands.json"), output = path.join(build, "runtime.obj");
    const command = `"${compiler}" --target=x86_64-pc-windows-msvc "/MT" "/EHsc" "/GR" -c "${cpp}" -o "runtime.obj"`;
    fs.writeFileSync(commandFile, JSON.stringify([{directory: build, file: path.relative(build, cpp), output: "runtime.obj", command}]));
    assert.deepEqual(auditAcppCompileCommands(commandFile, source, build, compiler), {
      receipt: {path: commandFile, sha256: fileHash(commandFile), commands: 1},
      objects: ["runtime.obj"],
    });
    fs.writeFileSync(commandFile, JSON.stringify([{directory: build, file: cpp, output,
      command: command + " /EHsc /EHs /GR"}]) + " ".repeat(8 << 20));
    assert.equal(auditAcppCompileCommands(commandFile, source, build, compiler).receipt.commands, 1);
    for (const bad of [command.replace("/MT", "/MD"), command.replace("/MT", "/MTd"), command.replace("/EHsc", ""),
      command.replace("/EHsc", "/EHsc /EH-"), command.replace("/GR", ""), command.replace("/GR", "/GR-"),
      command + " -fopenmp", command + " -fopenmp=libomp", command + " -imsvc " + f.root, command + " -isystem " + f.root,
      command + " -imsvc/" + f.root, command + " -isystem/" + f.root]) {
      fs.writeFileSync(commandFile, JSON.stringify([{directory: build, file: cpp, output, command: bad}]));
      assert.throws(() => auditAcppCompileCommands(commandFile, source, build, compiler));
    }
    fs.writeFileSync(commandFile, JSON.stringify([{directory: build, file: cpp, output: "../outside.obj", command}]));
    assert.throws(() => auditAcppCompileCommands(commandFile, source, build, compiler), /output outside/);
  });

  test("uses a noncolliding CMake literal and an explicit pinned LLVM runtime path", t => {
    const f = fixture(t), nativeTools = path.join(f.root, "linux-llvm]==]tools]=]/bin"), plan = windowsAcppStaticHostPlan({...f.input, nativeTools});
    assert.match(plan.toolchain, /set\(LLVM_NATIVE_TOOL_DIR \[===\[/);
    assert.ok(plan.toolchain.includes("]==]tools]=]/bin]===]"));
    assert.deepEqual(plan.nativeRuntimeRoots, [path.join(path.dirname(nativeTools), "lib")]);
    const env = cleanEnvironment({LD_LIBRARY_PATH: "/attacker/loader", GIT_NO_REPLACE_OBJECTS: "0", GIT_DIR: "/foreign/repository"}, plan.nativeRuntimeRoots);
    assert.equal(env["LD_LIBRARY_PATH"], plan.nativeRuntimeRoots.join(path.delimiter));
    assert.ok(!env["LD_LIBRARY_PATH"].includes("/opt/ubuntu24-libs"));
    assert.equal(env["GIT_NO_REPLACE_OBJECTS"], "1");
    assert.equal(env["GIT_DIR"], undefined);
    assert.deepEqual(commandSwitches('icx-cl "/MT" "/EHsc" "/GR-" -fopenmp=libomp'), ["/MT", "/EHSC", "/GR-", "-FOPENMP=LIBOMP"]);
  });

  test("CMake compiler identity proof refuses a Clang-shaped or ambiguous setup", t => {
    const f = fixture(t), cmakeFiles = path.join(f.root, "build/CMakeFiles"), version = path.join(cmakeFiles, "3.28.3");
    fs.mkdirSync(version, {recursive: true});
    fs.writeFileSync(path.join(version, "CMakeCCompiler.cmake"), 'set(CMAKE_C_COMPILER_ID "IntelLLVM")\n');
    fs.writeFileSync(path.join(version, "CMakeCXXCompiler.cmake"), 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")\n');
    assert.deepEqual(verifyIntelCompilerIdentity(path.join(f.root, "build")), {
      c: path.join(version, "CMakeCCompiler.cmake"), cxx: path.join(version, "CMakeCXXCompiler.cmake"),
    });
    fs.writeFileSync(path.join(version, "CMakeCCompiler.cmake"), 'set(CMAKE_C_COMPILER_ID "Clang")\n');
    assert.throws(() => verifyIntelCompilerIdentity(path.join(f.root, "build")), /IntelLLVM/);
  });

  test("source admission preserves full shared-helper log paths and diagnostics", t => {
    const f = fixture(t), logs = path.join(f.root, "admission"); fs.mkdirSync(logs);
    const patches = ["adaptivecpp-static.patch", "adaptivecpp-jit.patch"].map(name =>
      fs.readFileSync(path.join(__dirname, "../runtime", name), "utf8")).join("");
    const source = "/source", revision = "da2463e45aa90aa36306c45abcfc05b87de51bc6";
    assert.deepEqual(admitSource(source, logs, {}, (command, args) => {
      assert.equal(command, "/usr/bin/git", "source tools cannot be shadowed by SDK directories on PATH");
      return {status: 0, stdout: args.includes("rev-parse") ? revision + "\n" : args.includes("diff") ? patches : ""};
    }), {source, revision});
    assert.deepEqual(fs.readdirSync(logs).sort(), ["revision.log", "runtime-patch.log", "untracked.log"]);
    assert.equal(fs.readFileSync(path.join(logs, "revision.log"), "utf8"), revision + "\n");
    const failed = path.join(f.root, "failed"); fs.mkdirSync(failed);
    assert.throws(() => admitSource(source, failed, {}, () => ({status: 1, stderr: "git input failure\n"})), /pinned-source admission failed/);
    assert.equal(fs.readFileSync(path.join(failed, "revision.log"), "utf8"), "git input failure\n");
  });

  test("build entry rejects missing build-only admission before reading or writing paths", t => {
    const f = fixture(t);
    assert.throws(() => buildWindowsAcppStaticHost(f.input, {MOM_AOT_BUILD_ONLY: "0"}), /MOM_AOT_BUILD_ONLY=1/);
  });

  test("build entry rejects caller tool overrides before expensive LLVM admission", {
    skip: process.platform !== "linux" || process.arch !== "x64"
  }, t => {
    const f = fixture(t);
    assert.throws(() => buildWindowsAcppStaticHost(f.input, {MOM_AOT_BUILD_ONLY: "1"}), /unexpected input fields/);
  });

  test("archive membership comes from the frozen edge, including cross-target CBS objects", () => {
    const artifact = "src/compiler/llvm-to-host.lib", objects = ["src/compiler/CMakeFiles/llvm-to-host.dir/host.cpp.obj",
      "src/compiler/CMakeFiles/acpp-clang-cbs.dir/cbs/SyncDependenceAnalysis.cpp.obj"];
    const graph = `${artifact}:\n  input: CXX_STATIC_LIBRARY_LINKER__llvm-to-host_Release\n` +
      objects.map(file => "    " + file + "\n").join("") + "    | src/libkernel/resources.inc\n    || generated\n  outputs:\n    all\n";
    assert.deepEqual(archiveObjectInputs(graph, artifact, objects), objects);
    assert.throws(() => archiveObjectInputs(graph.replace(objects[1] || "", "../foreign.obj"), artifact, objects), /exact configured/);
    assert.throws(() => archiveObjectInputs(graph.replace(objects[1] || "", objects[0] || ""), artifact, objects), /exact configured/);
    assert.throws(() => archiveObjectInputs(graph, "other.lib", objects), /Unrecognized/);
  });
});
