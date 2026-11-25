"use strict";

// Build AdaptiveCpp's original Windows static host archives against the retained Windows
// LLVM 21 COFF build tree. Linux LLVM executables and Windows target libraries stay separate.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"), {isDeepStrictEqual} = require("node:util");
const {createHash} = require("node:crypto");
const {validateWindowsLlvmSdkConfig, windowsToolEnvironment, vfsFiles} = require("./build-native-windows-controller");
const {admitSyclSource} = require("./native-sycl-inputs");
const {hasExactBuildFields, containsBuildPath, buildPathsOverlap, buildFileInputs, fileHash, readBuildBytes, readBuildJson, sourceInputs, verifySourceInputs, nativeBuildJobs} = require("./build-source-inputs");
const {absolutePath} = require("./build-native-release");
const {isStaticArchive} = require("./native-audit");
const {nativeRuntimeLibraries} = require("./native-sdk-runtime");
const {auditCoffArchiveInputs} = require("./native-coff-archive");
const {windowsIntelRuleOverride} = require("./prepare-native-jit-toolchain");
const {cmake, cmakeCacheValues, commandSwitches, verifyWindowsLlvmAdmission} = require("./build-native-windows-llvm-sdk");
const {sdkStage, selectHostCompilers, hostCompilerEvidence} = require("./build-native-sycl-sdk");
const repo = path.resolve(__dirname, "..");
const targetTriple = "x86_64-pc-windows-msvc";
const gitTool = "/usr/bin/git";
const buildInputKeys = Object.freeze(["amdhip64", "cuda", "cudaDeviceLibs", "cudartStatic", "descriptor", "descriptorFile", "jobs",
  "nativeTools", "output", "rocm", "rocmDeviceLibs", "rocmReceipt", "source", "targetLlvm"].sort());
const windowsRocmRevision = "27682a16360e33e37c4f3cc6adf9a620733f8fe1";
const windowsRocmClrRevision = "5b591b2ff64aac4c1c08be879b0d5f89fd713d50";
const targets = Object.freeze(["acpp-rt", "rt-backend-cuda", "rt-backend-hip", "rt-backend-omp", "acpp-common",
  "llvm-to-backend", "llvm-to-ptx", "llvm-to-amdgpu", "llvm-to-host"]);
const artifacts = Object.freeze([
  "src/runtime/acpp-rt.lib", "src/runtime/rt-backend-cuda.lib", "src/runtime/rt-backend-hip.lib", "src/runtime/rt-backend-omp.lib",
  "src/common/acpp-common.lib", "src/compiler/llvm-to-backend/llvm-to-backend.lib", "src/compiler/llvm-to-backend/llvm-to-ptx.lib",
  "src/compiler/llvm-to-backend/llvm-to-amdgpu.lib", "src/compiler/llvm-to-backend/llvm-to-host.lib",
]);
const forbiddenOutputs = Object.freeze([
  "src/compiler/acpp-clang.dll", "src/compiler/acpp-clang.lib",
  "src/tools/acpp-appdb-tool/acpp-appdb-tool.exe", "src/tools/acpp-hcf-tool/acpp-hcf-tool.exe",
  "src/tools/acpp-info/acpp-info.exe", "src/tools/acpp-pcuda-pp/acpp-pcuda-pp.exe",
  "src/compiler/llvm-to-backend/llvm-to-amdgpu-tool.exe", "src/compiler/llvm-to-backend/llvm-to-host-tool.exe",
  "src/compiler/llvm-to-backend/llvm-to-ptx-tool.exe",
]);
const recipeFiles = Object.freeze(["scripts/build-native-windows-acpp-sdk.js", "scripts/build-native-windows-controller.js",
  "scripts/build-native-sycl-sdk.js", "scripts/build-native-windows-llvm-sdk.js", "scripts/native-jit-stages.js",
  "scripts/build-source-inputs.js", "scripts/build-native-release.js", "scripts/native-audit.js",
  "scripts/native-sycl-inputs.js", "scripts/native-hcf.js", "scripts/prepare-native-jit-toolchain.js", "scripts/native-resources.js", "scripts/native-sdk-runtime.js",
  "scripts/native-coff-archive.js", "scripts/native-compiler-isolation.js",
  "runtime/adaptivecpp-static.patch", "runtime/adaptivecpp-jit.patch"]);

/** Paths passed to CMake must stay literal; spaces are fine, shell/CMake syntax is not.
 * @param {unknown} value @returns {value is string} */
function literal(value) {
  return absolutePath(value) && path.normalize(value) === value && !/[;"$`\\]/.test(value);
}
/** @param {string[]} values @param {string} label */
function requirePaths(values, label) {
  if (!values.every(literal)) {throw new Error("Windows ACPP " + label + " need absolute literal paths");}
}
/** @param {string} root @param {string} name */
function nativeTool(root, name) {return path.join(root, name);}

/** @param {string} file @param {string} label */
function canonicalFile(file, label) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || fs.realpathSync(file) !== file) {throw new Error("Windows ACPP " + label + " must be a canonical regular file: " + file);}
  return file;
}

/** Plan only the nine original static COFF host archives. The selected LLVM producer output
 * supplies its exact CMake/Ninja paths; native Linux tools are a different explicit directory.
 * @param {{source: string, descriptorFile: string, descriptor: unknown, targetLlvm: string, nativeTools: string,
 * cuda: string, cudartStatic: string, cudaDeviceLibs: string, rocm: string, amdhip64: string,
 * rocmReceipt: string,
 * rocmDeviceLibs: string, output: string, jobs: number, tools: {cmake: string, ninja: string}}} input
 * @param {{source: string, build: string}} [admittedLlvmLayout] */
function windowsAcppStaticHostPlan(input, admittedLlvmLayout) {
  if (!hasExactBuildFields(input, ["amdhip64", "cuda", "cudaDeviceLibs", "cudartStatic", "descriptor", "descriptorFile",
    "jobs", "nativeTools", "output", "rocm", "rocmDeviceLibs", "rocmReceipt", "source", "targetLlvm", "tools"])) {
    throw new Error("Windows ACPP plan needs the explicit source, LLVM, CUDA, ROCm and output inputs");
  }
  const config = validateWindowsLlvmSdkConfig(input.descriptor), {tools: windowsTools, windowsSdk} = config;
  const selectedTools = input.tools;
  if (!selectedTools || !hasExactBuildFields(selectedTools, ["cmake", "ninja"])) {throw new Error("Select the LLVM producer's CMake and Ninja executables");}
  const native = {clang: nativeTool(input.nativeTools, "clang"), "clang++": nativeTool(input.nativeTools, "clang++"),
    "llvm-ar": nativeTool(input.nativeTools, "llvm-ar"), llc: nativeTool(input.nativeTools, "llc"),
    "lld-link": nativeTool(input.nativeTools, "lld-link"), opt: nativeTool(input.nativeTools, "opt"),
    "llvm-dis": nativeTool(input.nativeTools, "llvm-dis"), "llvm-link": nativeTool(input.nativeTools, "llvm-link")};
  const nativeLlvmRoot = path.dirname(input.nativeTools), nativeRuntimeRoots = [path.join(nativeLlvmRoot, "lib")];
  const targetLlvmSource = admittedLlvmLayout?.source || path.join(input.targetLlvm, "source"),
    targetBuild = admittedLlvmLayout?.build || path.join(input.targetLlvm, "build");
  if (![targetLlvmSource, targetBuild].every(literal) ||
      admittedLlvmLayout && !hasExactBuildFields(admittedLlvmLayout, ["build", "source"])) {
    throw new Error("Windows ACPP needs the admitted LLVM source/build layout");
  }
  const roots = [input.source, input.descriptorFile, input.targetLlvm, targetLlvmSource, targetBuild, input.nativeTools, input.cuda, input.cudartStatic,
    input.cudaDeviceLibs, input.rocm, input.amdhip64, input.rocmReceipt, input.rocmDeviceLibs, selectedTools.cmake, selectedTools.ninja,
    windowsTools.clangCl, windowsTools.clangResourceInclude, windowsSdk.overlay, windowsSdk.sourceRoot,
    ...windowsSdk.includeRoots, ...windowsSdk.libraryRoots, ...windowsSdk.msvcLibraries, ...windowsSdk.ucrtLibraries, ...nativeRuntimeRoots,
    ...Object.values(native), repo, input.output];
  requirePaths(roots, "roots");
  if (!Number.isInteger(input.jobs) || input.jobs < 1 || input.jobs > 16) {throw new Error("Windows ACPP jobs must be in 1..16");}
  for (const inputRoot of roots.slice(0, -1)) {
    if (buildPathsOverlap(inputRoot, input.output)) {throw new Error("Windows ACPP output overlaps a source, SDK, tool or compiler input");}
  }
  for (const file of [input.cudartStatic, input.amdhip64]) {
    if (!/\.lib$/i.test(file)) {throw new Error("Windows ACPP CUDA and HIP runtimes must be explicit COFF archives");}
  }
  const hostCompilers = selectHostCompilers("", {cc: windowsTools.clangCl, cxx: windowsTools.clangCl});
  const build = path.join(input.output, "build");
  const targetLlvmDir = path.join(targetBuild, "lib/cmake/llvm"), llvmDir = path.join(input.output, "llvm-cmake"),
    lldDir = path.join(targetBuild, "lib/cmake/lld");
  // Upstream libkernel invokes LLVM_TOOLS_BINARY_DIR/llvm-link, ignoring the native-tool
  // setting. Retain the exact target package, changing only its build-time tool directory.
  const llvmConfig = path.join(llvmDir, "LLVMConfig.cmake"), llvmVersion = path.join(llvmDir, "LLVMConfigVersion.cmake");
  const llvmConfigText = `include(${cmake(path.join(targetLlvmDir, "LLVMConfig.cmake"))})\nset(LLVM_TOOLS_BINARY_DIR ${cmake(input.nativeTools)})\n`;
  const llvmVersionText = `include(${cmake(path.join(targetLlvmDir, "LLVMConfigVersion.cmake"))})\n`;
  const overlay = path.join(input.output, "windows-sdk-vfs.json"), aliases = path.join(input.output, "bin");
  const ruleOverride = path.join(input.output, "intel-link-rules.cmake"), toolchainFile = path.join(input.output, "toolchain.cmake");
  const virtual = [...vfsFiles(windowsSdk.overlay)].filter(([name]) => windowsSdk.includeRoots.some(root =>
    name.toLowerCase().startsWith(root.replace(/\/+$/, "").toLowerCase() + "/")));
  if (!virtual.length) {throw new Error("Windows ACPP needs mapped Windows SDK headers");}

  const includeFlags = ["--target=" + targetTriple, "-DNOMINMAX", "-DWIN32_LEAN_AND_MEAN", "-nostdinc", "-I" + windowsTools.clangResourceInclude,
    "-Xclang", "-ivfsoverlay", "-Xclang", overlay, "-I" + path.join(input.cuda, "include"), "-I" + path.join(input.rocm, "include"),
    "-I" + path.join(targetLlvmSource, "clang/include"), "-I" + path.join(targetBuild, "tools/clang/include"),
    ...windowsSdk.includeRoots.map(root => "-I" + root), "/MT", "/O2", "/Gy", "/Gw"];
  const commonFlags = includeFlags.map(value => '"' + value + '"').join(" ");
  const cxxFlags = commonFlags + ' "/EHsc" "/GR" "/std:c++17"';
  const linkFlags = ["/MACHINE:X64", "/INCREMENTAL:NO", ...windowsSdk.libraryRoots
    .map(root => '/LIBPATH:"' + root + '"')].join(" ");
  const runtimeLibraries = [...windowsSdk.msvcLibraries, ...windowsSdk.ucrtLibraries, ...windowsSdk.importLibraries];
  const ruleOverrideText = windowsIntelRuleOverride(linkFlags, runtimeLibraries) +
    "set(CMAKE_INCLUDE_SYSTEM_FLAG_C /external:I)\nset(CMAKE_INCLUDE_SYSTEM_FLAG_CXX /external:I)\n" +
    // Older CMake IntelLLVM platform rules append /MD after toolchain initialization.
    "set(CMAKE_C_FLAGS_RELEASE_INIT \"/O2 /DNDEBUG\")\nset(CMAKE_CXX_FLAGS_RELEASE_INIT \"/O2 /DNDEBUG\")\n";
  const compilerTools = /** @type {Record<string, string>} */ ({CMAKE_C_COMPILER: hostCompilers.cc, CMAKE_CXX_COMPILER: hostCompilers.cxx,
    CMAKE_AR: path.join(aliases, "llvm-lib"), CMAKE_LINKER: native["lld-link"]});
  const toolchain = ["set(CMAKE_SYSTEM_NAME Windows CACHE STRING \"\" FORCE)", "set(CMAKE_SYSTEM_PROCESSOR AMD64)", "set(CMAKE_SYSTEM_VERSION 10.0)",
    "set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE CACHE STRING \"\" FORCE)", "set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)",
    "set(CMAKE_TRY_COMPILE_CONFIGURATION Release CACHE STRING \"\" FORCE)",
    `set(CMAKE_MSVC_RUNTIME_LIBRARY ${cmake("MultiThreaded")} CACHE STRING "" FORCE)`,
    `set(CMAKE_INCLUDE_SYSTEM_FLAG_C ${cmake("/external:I")} CACHE STRING "" FORCE)`,
    `set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX ${cmake("/external:I")} CACHE STRING "" FORCE)`,
    ...Object.entries(compilerTools).map(([name, file]) => `set(${name} ${cmake(file)} CACHE FILEPATH "" FORCE)`),
    `set(CMAKE_C_COMPILER_TARGET ${cmake(targetTriple)} CACHE STRING "" FORCE)`,
    `set(CMAKE_CXX_COMPILER_TARGET ${cmake(targetTriple)} CACHE STRING "" FORCE)`,
    `set(CMAKE_USER_MAKE_RULES_OVERRIDE_C ${cmake(ruleOverride)})`, `set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX ${cmake(ruleOverride)})`,
    `set(CMAKE_C_FLAGS_INIT ${cmake(commonFlags)})`, `set(CMAKE_CXX_FLAGS_INIT ${cmake(cxxFlags)})`,
    `set(CMAKE_C_FLAGS_RELEASE_INIT ${cmake("/O2 /DNDEBUG")})`,
    `set(CMAKE_CXX_FLAGS_RELEASE_INIT ${cmake("/O2 /DNDEBUG")})`,
    `set(CMAKE_EXE_LINKER_FLAGS_INIT ${cmake(linkFlags)})`, `set(CMAKE_SHARED_LINKER_FLAGS_INIT ${cmake(linkFlags)})`,
    `set(CMAKE_MODULE_LINKER_FLAGS_INIT ${cmake(linkFlags)})`, `set(CMAKE_STATIC_LINKER_FLAGS_INIT ${cmake("/MACHINE:X64")})`,
    `set(LLVM_NATIVE_TOOL_DIR ${cmake(input.nativeTools)} CACHE PATH "" FORCE)`,
    `set(CMAKE_FIND_ROOT_PATH ${cmake(windowsSdk.sourceRoot)})`,
    ...["LIBRARY", "INCLUDE", "PACKAGE"].map(name => `set(CMAKE_FIND_ROOT_PATH_MODE_${name} ONLY)`),
    "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)", ""].join("\n");
  const definitions = [
    ["CMAKE_TOOLCHAIN_FILE", toolchainFile], ["CMAKE_MAKE_PROGRAM", selectedTools.ninja], ["CMAKE_BUILD_TYPE", "Release"],
    ["CMAKE_EXPORT_COMPILE_COMMANDS", "ON"], ["CMAKE_SUPPRESS_REGENERATION", "ON"], ["CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded"],
    ["CMAKE_TRY_COMPILE_CONFIGURATION", "Release"],
    ["CMAKE_INCLUDE_SYSTEM_FLAG_C", "/external:I"], ["CMAKE_INCLUDE_SYSTEM_FLAG_CXX", "/external:I"],
    ["CMAKE_DISABLE_FIND_PACKAGE_HIP", "TRUE"], ["CMAKE_DISABLE_FIND_PACKAGE_OpenCL", "TRUE"], ["CMAKE_DISABLE_FIND_PACKAGE_OpenMP", "TRUE"],
    ["ACPP_STATIC_RUNTIME", "ON"], ["ACPP_COMPILER_FEATURE_PROFILE", "custom-deprecated"],
    ["ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER", "ON"], ["ACPP_CUSTOM_PROFILE_WITH_STDPAR_COMPILER", "OFF"],
    ["ACPP_CUSTOM_PROFILE_WITH_ACCELERATED_CPU", "OFF"],
    ["WITH_CUDA_BACKEND", "ON"], ["WITH_ROCM_BACKEND", "ON"], ["WITH_OPENCL_BACKEND", "OFF"],
    ["WITH_LEVEL_ZERO_BACKEND", "OFF"], ["WITH_VULKAN_BACKEND", "OFF"], ["WITH_METAL_BACKEND", "OFF"],
    ["ACPP_SUBPROJECT_PARALLEL_JOBS", "1"], ["ACPP_RESOURCE_GENERATOR", path.join(repo, "scripts/native-resources.js")],
    ["ACPP_RESOURCE_NODE", process.execPath], ["LLVM_DIR", llvmDir], ["LLD_DIR", lldDir],
    ["LLVM_NATIVE_TOOL_DIR", input.nativeTools], ["CLANG_EXECUTABLE_PATH", native.clang],
    ["CLANG_INCLUDE_PATH", path.dirname(windowsTools.clangResourceInclude)], ["ROCM_CLANG_PATH", native["clang++"]],
    ["ACPP_LLC_PATH", native.llc], ["ACPP_LLD_PATH", native["lld-link"]], ["ACPP_OPT_PATH", native.opt],
    // No Windows nvcc runs in this host-only build. Keep CMake on the selected SDK's
    // headers/archive instead of discovering an unrelated Linux nvcc through PATH.
    ["CUDA_TOOLKIT_ROOT_DIR", input.cuda], ["CUDAToolkit_ROOT", input.cuda], ["CUDAToolkit_BIN_DIR", path.join(input.cuda, "bin")],
    ["CUDAToolkit_LIBRARY_ROOT", input.cuda], ["CUDA_CUDART", input.cudartStatic], ["CUDA_CUDART_STATIC_LIBRARY", input.cudartStatic],
    ["CUDA_cudart_static_LIBRARY", input.cudartStatic], ["CUDA_DEVICE_LIBS_PATH", input.cudaDeviceLibs],
    ["ROCM_PATH", input.rocm], ["AMDHIP64_LIBRARY", input.amdhip64],
    ["ROCM_DEVICE_LIBS_PATH", input.rocmDeviceLibs],
  ];
  const configure = [selectedTools.cmake, "-S", input.source, "-B", build, "-G", "Ninja", ...definitions.map(([name, value]) => "-D" + name + "=" + value)];
  const compile = [selectedTools.cmake, "--build", build, "--parallel", String(input.jobs), "--target", ...targets, "--", "-k", "0"];
  return {config, hostCompilers, native, nativeLlvmRoot, nativeRuntimeRoots, compilerTools, targetLlvmSource, targetBuild, build, llvmDir, targetLlvmDir,
    llvmConfig, llvmConfigText, llvmVersion, llvmVersionText, lldDir, aliases, overlay, ruleOverride, ruleOverrideText,
    toolchainFile, toolchain, linkFlags, includeFlags, definitions: Object.fromEntries(definitions), targets: [...targets],
    artifacts: artifacts.map(name => path.join(build, name)), forbiddenOutputs: forbiddenOutputs.map(name => path.join(build, name)),
    configure, compile, virtual, tools: {...selectedTools}};
}

/** Parse CMake's cache once; the exact strings are inputs, not heuristics based on nearby names.
 * @param {string} file @param {ReturnType<typeof windowsAcppStaticHostPlan>} plan @param {string} source */
function verifyAcppCache(file, plan, source) {
  const text = fs.readFileSync(file, "utf8"), values = cmakeCacheValues(text);
  if (values.has("CMAKE_SYSTEM_NAME") && values.get("CMAKE_SYSTEM_NAME") !== "Windows") {
    throw new Error("Windows ACPP CMake configuration drifted from its frozen plan: CMAKE_SYSTEM_NAME");
  }
  if (fs.readFileSync(plan.toolchainFile, "utf8") !== plan.toolchain) {throw new Error("Windows ACPP toolchain file changed after planning");}
  const expected = {...plan.definitions, CMAKE_C_COMPILER: plan.hostCompilers.cc,
    CMAKE_CXX_COMPILER: plan.hostCompilers.cxx, CMAKE_C_COMPILER_TARGET: targetTriple, CMAKE_CXX_COMPILER_TARGET: targetTriple,
    CMAKE_MSVC_RUNTIME_LIBRARY: "MultiThreaded", CMAKE_TRY_COMPILE_TARGET_TYPE: "EXECUTABLE",
    CMAKE_HOME_DIRECTORY: fs.realpathSync(source), ACPP_COMPILER_FEATURE_PROFILE: "custom-deprecated",
    ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER: "ON", ACPP_CUSTOM_PROFILE_WITH_STDPAR_COMPILER: "OFF",
    ACPP_CUSTOM_PROFILE_WITH_ACCELERATED_CPU: "OFF", WITH_CUDA_BACKEND: "ON", WITH_ROCM_BACKEND: "ON",
    WITH_OPENCL_BACKEND: "OFF", WITH_VULKAN_BACKEND: "OFF", OMP_LINK_LINE: "", SEQUENTIAL_LINK_LINE: "",
    OMP_CXX_FLAGS: "-D_ENABLE_EXTENDED_ALIGNED_STORAGE", SEQUENTIAL_CXX_FLAGS: "-D_ENABLE_EXTENDED_ALIGNED_STORAGE"};
  for (const [name, value] of Object.entries(expected)) {
    if (values.get(name) !== value) {throw new Error("Windows ACPP CMake configuration drifted from its frozen plan: " + name);}
  }
  if (!/^CMAKE_DISABLE_FIND_PACKAGE_HIP:[^=]+=TRUE$/m.test(text) || !/^CMAKE_DISABLE_FIND_PACKAGE_OpenCL:[^=]+=TRUE$/m.test(text) ||
      !/^CMAKE_DISABLE_FIND_PACKAGE_OpenMP:[^=]+=TRUE$/m.test(text)) {throw new Error("Windows ACPP must not discover host HIP, OpenCL or OpenMP packages");}
}

/** CMake must identify the actual selected Intel driver before its shared linker rules apply.
 * @param {string} build @returns {{c: string, cxx: string}} */
function verifyIntelCompilerIdentity(build) {
  const directory = path.join(build, "CMakeFiles"), candidates = fs.readdirSync(directory)
    .filter(name => /^\d+\.\d+(?:\.\d+)?$/.test(name)).map(name => path.join(directory, name));
  /** @param {string} language @param {RegExp} identity */
  const read = (language, identity) => {
    const file = candidates.map(root => path.join(root, "CMake" + language + "Compiler.cmake")).find(name => fs.existsSync(name));
    if (!file || fs.statSync(file).size > (1 << 20) || !identity.test(fs.readFileSync(file, "utf8"))) {
      throw new Error("Windows ACPP CMake did not identify the selected " + language + " driver as IntelLLVM");
    }
    return file;
  };
  return {c: read("C", /set\(CMAKE_C_COMPILER_ID "IntelLLVM"\)/),
    cxx: read("CXX", /set\(CMAKE_CXX_COMPILER_ID "IntelLLVM"\)/)};
}

/** Bind archive membership and the receipt hash to the same audited compile database.
 * @param {string} file @param {string} source @param {string} build @param {string} compiler */
function auditAcppCompileCommands(file, source, build, compiler) {
  const bytes = readBuildBytes(file, 64 << 20);
  const entries = JSON.parse(bytes.toString("utf8"));
  if (!Array.isArray(entries) || !entries.length || entries.length > 100000) {throw new Error("Windows ACPP compile database is empty or excessive");}
  const sourceRoot = fs.realpathSync(source) + path.sep, buildRoot = fs.realpathSync(build), outputs = new Set();
  for (const entry of entries) {
    const switches = entry && typeof entry.command === "string" ? commandSwitches(entry.command) : [];
    const runtimeFlags = switches.filter(flag => /^\/(?:MT|MD)/.test(flag));
    const ehFlags = switches.filter(flag => flag.startsWith("/EH")), rttiFlags = switches.filter(flag => flag.startsWith("/GR"));
    const unsupportedSystemInclude = typeof entry?.command === "string" && /(?:^|[\s"])-i(?:system|msvc)(?=$|[\s"/])/i.test(entry.command);
    if (!entry || typeof entry.file !== "string" || typeof entry.output !== "string" || typeof entry.command !== "string" || typeof entry.directory !== "string" ||
        entry.command.length > (1 << 20) || /[\r\n\0]/.test(entry.command) ||
        !(entry.command.startsWith(compiler + " ") || entry.command.startsWith('"' + compiler + '" ')) ||
        !entry.command.includes("--target=" + targetTriple) || runtimeFlags.length !== 1 || runtimeFlags[0] !== "/MT" ||
        switches.some(flag => flag.startsWith("-FOPENMP") || flag.startsWith("/OPENMP")) || unsupportedSystemInclude || /libgomp/i.test(entry.command)) {
      throw new Error("Windows ACPP compile command has the wrong compiler, ABI, backend or output");
    }
    const workingDirectory = path.resolve(entry.directory);
    if (workingDirectory !== buildRoot && !workingDirectory.startsWith(buildRoot + path.sep)) {
      throw new Error("Windows ACPP compile command has an unselected working directory");
    }
    const extension = path.extname(entry.file).toLowerCase();
    // CMake and upstream targets may repeat these options; EHs and EHsc both retain
    // synchronous C++ unwinding. Reject disabling/conflicting variants, not duplicates.
    if ([".cc", ".cpp", ".cxx", ".c++"].includes(extension) &&
        (!ehFlags.length || ehFlags.some(flag => !["/EHS", "/EHSC"].includes(flag)) ||
          !rttiFlags.length || rttiFlags.some(flag => flag !== "/GR"))) {
      throw new Error("Windows ACPP C++ compile command omitted exceptions or RTTI");
    }
    const sourceFile = fs.realpathSync(path.isAbsolute(entry.file) ? entry.file : path.resolve(entry.directory, entry.file));
    if (!sourceFile.startsWith(sourceRoot) && sourceFile !== buildRoot && !sourceFile.startsWith(buildRoot + path.sep)) {
      throw new Error("Windows ACPP compile command references an unselected source");
    }
    // CMake may serialize outputs as absolute or build-relative paths; these records describe
    // every configured target, not just the nine explicitly built archive targets.
    const output = path.resolve(entry.directory, entry.output), relative = path.relative(buildRoot, output);
    const object = relative.split(path.sep).join("/");
    if (!output.endsWith(".obj") || path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep) || outputs.has(object)) {
      throw new Error("Windows ACPP compile command has an output outside the selected build tree");
    }
    outputs.add(object);
  }
  return {receipt: {path: file, sha256: createHash("sha256").update(bytes).digest("hex"), commands: entries.length},
    objects: [...outputs]};
}

/** The shared source admission supplies full log paths, not names relative to output.
 * @param {string} source @param {string} output @param {NodeJS.ProcessEnv} environment
 * @param {(command: string, args: string[]) => {status: number | null, signal?: NodeJS.Signals | null, error?: Error, stdout?: string, stderr?: string}} [execute] */
function admitSource(source, output, environment, execute = (command, args) =>
  spawnSync(command, args, {env: environment, encoding: "utf8", timeout: 30000, maxBuffer: 4 << 20})) {
  return admitSyclSource(source, output, true, (command, args, name) => {
    if (command !== "git") {throw new Error("Unexpected AdaptiveCpp source-admission tool");}
    const result = execute(gitTool, args);
    fs.writeFileSync(name, (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
    if (result.error || result.signal || result.status !== 0) {throw new Error("AdaptiveCpp pinned-source admission failed: " + name, {cause: result.error});}
    return result.stdout || "";
  });
}

/** Only explicit immediate shared-library roots are retained for pinned native LLVM tools.
 * @param {ReturnType<typeof windowsAcppStaticHostPlan>} plan @returns {string[]} */
function nativeRuntimeInputs(plan) {
  return nativeRuntimeLibraries(plan.nativeRuntimeRoots, "Windows ACPP");
}

/** Let the frozen Ninja edge own object-library composition: llvm-to-host also embeds CBS
 * objects from another target. Every explicit member must match the audited compile database.
 * @param {string} text @param {string} artifact @param {string[]} configuredObjects */
function archiveObjectInputs(text, artifact, configuredObjects) {
  if (text.length > (4 << 20)) {throw new Error("Excessive Windows ACPP archive build edge");}
  const lines = text.trimEnd().split(/\r?\n/), known = new Set(configuredObjects), selected = [], seen = new Set();
  if (lines.shift() !== artifact + ":" ||
      !/^ {2}input: CXX_STATIC_LIBRARY_LINKER__[^\r\n]+_Release$/.test(lines.shift() || "")) {
    throw new Error("Unrecognized Windows ACPP archive build edge");
  }
  let outputs = false;
  for (const line of lines) {
    if (line === "  outputs:") {if (outputs) {throw new Error("Duplicate archive build outputs");} outputs = true; continue;}
    if (outputs || /^ {4}\|{1,2} /.test(line)) {continue;}
    const object = line.match(/^ {4}(.+)$/)?.[1];
    if (!object || !known.has(object) || seen.has(object)) {throw new Error("Archive input is not an exact configured COFF object");}
    selected.push(object); seen.add(object);
  }
  if (!outputs || !selected.length) {throw new Error("Windows ACPP archive build edge omitted its objects");}
  return selected;
}

/** @param {NodeJS.ProcessEnv} input @param {string[]} runtimeLibraries */
function cleanEnvironment(input, runtimeLibraries) {
  const env = windowsToolEnvironment(input);
  for (const key of Object.keys(env)) {
    if (/^(?:CMAKE_|LLVM_|CLANG_|CUDA_|CUDAToolkit_|HIP|ROCM_|ACPP_|CC$|CXX$|AR$|RC$|MAKEFLAGS$|NINJAFLAGS$|PKG_CONFIG_PATH$|GIT_)/i.test(key) ||
        /^(?:LD_PRELOAD|LD_LIBRARY_PATH|LD_AUDIT|OMP_)/i.test(key)) {delete env[key];}
  }
  env["GIT_OPTIONAL_LOCKS"] = "0";
  env["GIT_NO_REPLACE_OBJECTS"] = "1";
  env["GIT_TERMINAL_PROMPT"] = "0";
  env["GIT_NO_LAZY_FETCH"] = "1";
  env["LD_LIBRARY_PATH"] = runtimeLibraries.join(path.delimiter);
  return env;
}

/** Keep the Windows-produced device libraries bound to their historical source receipt;
 * old tool paths in that record are evidence, not tools to re-run in this image.
 * @param {string} receiptFile @param {string} deviceRoot */
function verifyWindowsRocmResources(receiptFile, deviceRoot) {
  canonicalFile(receiptFile, "Windows ROCm resource receipt");
  const receipt = readBuildJson(receiptFile), root = path.dirname(receiptFile), expectedRoot = path.join(root, "install/amdgcn/bitcode");
  /** @param {unknown} value @param {boolean} paths @returns {boolean} */
  const validMap = (value, paths) => value !== null && typeof value === "object" && !Array.isArray(value) &&
    Object.keys(value).length > 0 && Object.keys(value).length <= 65536 && Object.entries(value).every(([name, hash]) =>
    typeof name === "string" && name.length <= 4096 && (paths ? path.isAbsolute(name) && path.normalize(name) === name :
      !name.startsWith("/") && !name.split("/").some(part => !part || part === "." || part === "..")) &&
      typeof hash === "string" && /^[0-9a-f]{64}$/.test(hash));
  if (!path.isAbsolute(deviceRoot) || fs.realpathSync(deviceRoot) !== deviceRoot || deviceRoot !== expectedRoot ||
      receipt?.version !== 1 || receipt.target !== "win32" || receipt.revision !== windowsRocmRevision ||
      receipt.clrRevision !== windowsRocmClrRevision || receipt.buildOnly !== true || receipt.publicationQualified !== false ||
      receipt.executed === true || !validMap(receipt.source, false) || !validMap(receipt.inputs, true) ||
      !validMap(receipt.dependencies, true) || !validMap(receipt.device, true) ||
      !receipt.header || typeof receipt.header.path !== "string" || !path.isAbsolute(receipt.header.path) ||
      typeof receipt.header.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(receipt.header.sha256) ||
      !receipt.object || typeof receipt.object.path !== "string" || !path.isAbsolute(receipt.object.path) ||
      typeof receipt.object.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(receipt.object.sha256)) {
    throw new Error("Select the unchanged pinned Windows HIP resource receipt and install directory");
  }
  const entries = fs.readdirSync(deviceRoot, {withFileTypes: true}).filter(entry => entry.name.endsWith(".bc"));
  if (!entries.length || entries.length > 4096 || entries.some(entry => !entry.isFile())) {
    throw new Error("Windows HIP device-library directory is not an exact bounded regular-file set");
  }
  const files = entries.map(entry => path.join(deviceRoot, entry.name)).sort(), recorded = Object.keys(receipt.device)
    .filter(file => file.endsWith(".bc")).sort();
  if (!isDeepStrictEqual(files, recorded) || !files.some(file => path.basename(file) === "ockl.bc") ||
      !files.some(file => path.basename(file) === "ocml.bc") || !files.some(file => path.basename(file) === "hip.bc") ||
      !files.some(file => /^oclc_abi_version_[0-9]+\.bc$/.test(path.basename(file))) ||
      recorded.some(file => !containsBuildPath(deviceRoot, file))) {
    throw new Error("Windows HIP device-library membership differs from its pinned target receipt");
  }
  for (const file of files) {
    canonicalFile(file, "Windows HIP device library");
    if (fileHash(file) !== receipt.device[file]) {throw new Error("Windows HIP device library differs from its Windows target receipt: " + file);}
  }
  return {receipt, path: receiptFile, sha256: fileHash(receiptFile), revision: receipt.revision, clrRevision: receipt.clrRevision,
    device: receipt.device, files};
}

/** Build the nine original archives only. No compiler plugin, AdaptiveCpp command-line tool,
 * installation, Windows execution or GPU execution is part of this producer.
 * @param {Omit<Parameters<typeof windowsAcppStaticHostPlan>[0], "tools">} input @param {NodeJS.ProcessEnv} [environment] */
function buildWindowsAcppStaticHost(input, environment = process.env) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("Windows ACPP cross-build requires Linux x64 and MOM_AOT_BUILD_ONLY=1");
  }
  // Tools come only from the admitted LLVM producer; catch caller/receipt shape drift before building.
  if (!input || !isDeepStrictEqual(Object.keys(input).sort(), buildInputKeys)) {
    throw new Error("Windows ACPP build has unexpected input fields");
  }
  const targetLlvm = verifyWindowsLlvmAdmission(input.targetLlvm), {receipt: targetReceipt, admission: targetAdmission,
    receiptFile: targetReceiptFile, admissionFile: targetAdmissionFile, objectsFile: targetObjectProofFile} = targetLlvm;
  const llvmTools = /** @type {{cmake: string, ninja: string}} */ ({cmake: /** @type {string} */ (targetReceipt.plan?.tools?.cmake),
    ninja: /** @type {string} */ (targetReceipt.plan?.tools?.ninja)});
  const llvmPlan = targetReceipt.plan, admittedLlvmLayout = {source: llvmPlan.source,
    build: path.join(llvmPlan.output, "build")};
  const plan = windowsAcppStaticHostPlan({...input, tools: llvmTools}, admittedLlvmLayout), env = cleanEnvironment(environment, plan.nativeRuntimeRoots);
  const outputParent = fs.realpathSync(path.dirname(input.output)), target = path.join(outputParent, path.basename(input.output));
  if (target !== input.output || fs.existsSync(input.output)) {throw new Error("Select a new canonical Windows ACPP output directory");}
  const inputRoots = [input.source, input.descriptorFile, input.targetLlvm, plan.targetLlvmSource, plan.targetBuild,
    input.nativeTools, ...plan.nativeRuntimeRoots, input.cuda, input.cudartStatic,
    input.cudaDeviceLibs, input.rocm, input.amdhip64, input.rocmReceipt, input.rocmDeviceLibs, plan.tools.cmake, plan.tools.ninja, repo];
  for (const file of inputRoots) {
    const real = fs.realpathSync(file), relative = path.relative(real, target);
    if (!relative || !path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep)) {
      throw new Error("Windows ACPP output overlaps a canonical source, SDK or compiler input");
    }
  }
  const {config, hostCompilers, native, virtual} = plan;
  const rocmResources = verifyWindowsRocmResources(input.rocmReceipt, input.rocmDeviceLibs), rocmBitcode = rocmResources.files;
  const selectedFiles = [gitTool, input.descriptorFile, targetReceiptFile, targetAdmissionFile, targetObjectProofFile, input.rocmReceipt,
    plan.tools.cmake, plan.tools.ninja, config.tools.clangCl,
    path.join(plan.targetLlvmDir, "LLVMConfig.cmake"), path.join(plan.targetLlvmDir, "LLVMConfigVersion.cmake"),
    path.join(plan.targetBuild, "lib/cmake/lld/LLDConfig.cmake"),
    input.cudartStatic, path.join(input.cuda, "include/cuda.h"), path.join(input.cuda, "include/cuda_runtime.h"),
    path.join(input.cuda, "include/cuda_runtime_api.h"), path.join(input.cudaDeviceLibs, "libdevice.10.bc"),
    input.amdhip64, path.join(input.rocm, "include/hip/hip_runtime.h"), path.join(input.rocm, "include/hip/hip_runtime_api.h"),
    path.join(input.rocm, "include/hip/hip_version.h"), native.clang, native["clang++"], native["llvm-ar"], native.llc,
    native["lld-link"], native.opt, native["llvm-dis"], native["llvm-link"], ...targetLlvm.admission.archives.map(target => /** @type {string} */ (target.artifact)),
    config.windowsSdk.overlay, ...virtual.map(([, file]) => file),
    ...config.windowsSdk.msvcLibraries, ...config.windowsSdk.ucrtLibraries, ...config.windowsSdk.importLibraries, ...rocmBitcode, ...nativeRuntimeInputs(plan),
    path.join(repo, "scripts/native-resources.js"), ...recipeFiles.map(name => path.join(repo, name))];
  const recipe = sourceInputs(repo, [], [...recipeFiles]), fingerprints = buildFileInputs(selectedFiles);
  for (const archive of [input.cudartStatic, input.amdhip64]) {if (!isStaticArchive(archive)) {throw new Error("Windows ACPP runtime input is not a regular static archive: " + archive);}}
  for (const file of [path.join(input.cudaDeviceLibs, "libdevice.10.bc"), path.join(plan.targetBuild, "lib/cmake/llvm/LLVMConfig.cmake"),
    path.join(plan.targetBuild, "lib/cmake/lld/LLDConfig.cmake")]) {
    if (!fs.statSync(file, {throwIfNoEntry: false})?.isFile()) {throw new Error("Missing Windows ACPP selected SDK input: " + file);}
  }
  fs.mkdirSync(input.output, {mode: 0o700}); fs.mkdirSync(path.join(input.output, "pre")); fs.mkdirSync(path.join(input.output, "post"));
  fs.mkdirSync(path.join(input.output, "rocm-ir"));
  fs.mkdirSync(plan.aliases);
  fs.mkdirSync(plan.llvmDir);
  fs.writeFileSync(plan.llvmConfig, plan.llvmConfigText, {flag: "wx"});
  fs.writeFileSync(plan.llvmVersion, plan.llvmVersionText, {flag: "wx"});
  fs.symlinkSync(native["llvm-ar"], path.join(plan.aliases, "llvm-lib"));
  const overlayText = JSON.stringify({version: 0, "case-sensitive": false, "use-external-names": true,
    roots: virtual.map(([name, file]) => ({type: "file", name, "external-contents": file}))});
  fs.writeFileSync(plan.overlay, overlayText, {flag: "wx"});
  fs.writeFileSync(plan.ruleOverride, plan.ruleOverrideText, {flag: "wx"});
  fs.writeFileSync(plan.toolchainFile, plan.toolchain, {flag: "wx"});
  for (const file of [plan.overlay, plan.ruleOverride, plan.toolchainFile, plan.llvmConfig, plan.llvmVersion]) {fingerprints[file] = fileHash(file);}
  /** @param {string} command @param {string[]} args @param {string} name */
  function query(command, args, name) {
    const result = spawnSync(command, args, {env, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
    fs.writeFileSync(path.join(input.output, name), (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Windows ACPP input probe failed: " + name, {cause: result.error});}
    return result.stdout || "";
  }
  const admitted = admitSource(input.source, path.join(input.output, "pre"), env);
  const compilerEvidence = hostCompilerEvidence(hostCompilers, query);
  const nativeClang = query(native["clang++"], ["--version"], "native-clang-version.log");
  if (!/\bclang version 21\./i.test(nativeClang)) {throw new Error("Windows ACPP native Clang toolchain must be LLVM 21");}
  const llvmDisVersion = query(native["llvm-dis"], ["--version"], "native-llvm-dis-version.log");
  if (!/\bLLVM version 21\.1\.8\b/.test(llvmDisVersion)) {throw new Error("Windows ACPP device-resource validator must be LLVM 21.1.8");}
  const wcharEvidence = rocmBitcode.map(file => {
    const name = path.basename(file, ".bc"), ir = path.join(input.output, "rocm-ir", name + ".ll");
    query(native["llvm-dis"], [file, "-o", ir], "rocm-bitcode-" + name + ".log");
    const stat = fs.statSync(ir);
    if (!stat.isFile() || stat.size < 1 || stat.size > (128 << 20) ||
        !/!"wchar_size", i32 2\b/.test(fs.readFileSync(ir, "utf8"))) {
      throw new Error("Windows ROCm device bitcode does not retain wchar_size 2: " + file);
    }
    fingerprints[ir] = fileHash(ir);
    return {bitcode: file, sha256: fingerprints[file], llvmIr: ir, llvmIrSha256: fingerprints[ir]};
  });
  for (const [file, hash] of Object.entries(compilerEvidence.inputs)) {fingerprints[file] = hash;}
  fs.writeFileSync(path.join(input.output, "plan.json"), JSON.stringify({version: 1, role: "windows-acpp-static-host", input,
    targetLlvmGeneration: targetAdmission.generationId,
    targetReceiptSha256: fileHash(targetReceiptFile), targetAdmissionSha256: fileHash(targetAdmissionFile),
    targetObjectProofSha256: fileHash(targetObjectProofFile), rocmResourceReceiptSha256: rocmResources.sha256,
    rocmResourceRevision: rocmResources.revision, rocmResourceClrRevision: rocmResources.clrRevision,
    wcharEvidence, admitted, compilerEvidence, fingerprints, recipe, configure: plan.configure,
    compile: plan.compile, artifacts: plan.artifacts, forbiddenOutputs: plan.forbiddenOutputs}, null, 2) + "\n", {flag: "wx"});

  sdkStage(plan.configure, path.join(input.output, "stage-0.log"), env);
  verifyAcppCache(path.join(plan.build, "CMakeCache.txt"), plan, input.source);
  const compilerIdentity = verifyIntelCompilerIdentity(plan.build);
  const {receipt: compileCommands, objects: configuredObjects} =
    auditAcppCompileCommands(path.join(plan.build, "compile_commands.json"), input.source, plan.build, hostCompilers.cxx);
  const configuredFiles = buildFileInputs([path.join(plan.build, "CMakeCache.txt"), path.join(plan.build, "build.ninja"), compileCommands.path,
    compilerIdentity.c, compilerIdentity.cxx]);
  sdkStage(plan.compile, path.join(input.output, "stage-1.log"), env);
  verifyWindowsLlvmAdmission(input.targetLlvm);
  const finalRocmResources = verifyWindowsRocmResources(input.rocmReceipt, input.rocmDeviceLibs);
  if (finalRocmResources.sha256 !== rocmResources.sha256) {throw new Error("Windows ROCm resource receipt changed during AdaptiveCpp build");}
  admitSource(input.source, path.join(input.output, "post"), env);
  verifySourceInputs(recipe, sourceInputs(repo, [], [...recipeFiles]), "Windows ACPP build recipe");
  for (const [file, hash] of Object.entries(fingerprints)) {if (fileHash(file) !== hash) {throw new Error("Windows ACPP input changed during build: " + file);}}
  for (const [file, hash] of Object.entries(configuredFiles)) {if (fileHash(file) !== hash) {throw new Error("Windows ACPP configure graph changed during build: " + file);}}
  const outputArchives = plan.artifacts.map(file => {
    if (!isStaticArchive(file)) {throw new Error("Windows ACPP output is not a static COFF archive: " + file);}
    const relative = path.relative(plan.build, file).split(path.sep).join("/");
    const graph = query(plan.tools.ninja, ["-C", plan.build, "-t", "query", relative], "archive-inputs-" + path.basename(file) + ".log");
    const objects = archiveObjectInputs(graph, relative, configuredObjects);
    const audited = auditCoffArchiveInputs(file, plan.build, objects);
    return {path: file, sha256: audited.archiveSha256, bytes: fs.statSync(file).size,
      members: audited.members, objects: audited.objects};
  });
  const unexpected = plan.forbiddenOutputs.filter(file => fs.existsSync(file));
  if (unexpected.length) {throw new Error("Windows ACPP archive-only build produced a plugin or tool: " + unexpected[0]);}
  const {receipt: finalCompileCommands} = auditAcppCompileCommands(compileCommands.path, input.source, plan.build, hostCompilers.cxx);
  if (!isDeepStrictEqual(compileCommands, finalCompileCommands)) {throw new Error("Windows ACPP compile database changed during build");}
  const result = {version: 1, role: "windows-acpp-static-host", targetLlvmGeneration: targetLlvm.admission.generationId,
    targetReceiptSha256: fileHash(targetReceiptFile), revision: admitted.revision,
    targetAdmissionSha256: fileHash(targetAdmissionFile), targetObjectProofSha256: fileHash(targetObjectProofFile),
    rocmResourceReceiptSha256: rocmResources.sha256, rocmResourceRevision: rocmResources.revision,
    rocmResourceClrRevision: rocmResources.clrRevision, rocmDeviceLibraries: rocmResources.device, wcharEvidence,
    hostCompilers: hostCompilers, compilerEvidence, nativeTools: Object.fromEntries(Object.entries(native).map(([name, file]) => [name, fingerprints[file]])),
    fingerprints, configuredFiles, recipe, compileCommands, targets: plan.targets, artifacts: outputArchives,
    buildOnly: true, compiled: true, executed: false, publicationQualified: false,
    scope: "Original AdaptiveCpp static Windows CUDA+HIP+sequential-OMP host archives and LLVM SSCP backend archives only. No acpp-clang DLL, command-line tools, installation, Windows execution, GPU execution or publication qualification."};
  fs.writeFileSync(path.join(input.output, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  console.log("[native-sdk] Built the nine original Windows AdaptiveCpp static host archives; not executed or publication-qualified.");
  return result;
}

if (require.main === module) {
  try {
    const [source, descriptorFile, targetLlvm, nativeTools, cuda, cudartStatic, cudaDeviceLibs, rocm, amdhip64,
      rocmDeviceLibs, rocmReceipt, output] = process.argv.slice(2);
    if (process.argv.length !== 14 || !source || !descriptorFile || !targetLlvm || !nativeTools || !cuda || !cudartStatic ||
        !cudaDeviceLibs || !rocm || !amdhip64 || !rocmDeviceLibs || !rocmReceipt || !output) {
      throw new Error("Usage: build-native-windows-acpp-sdk.js /AdaptiveCpp /windows-descriptor.json /windows-llvm21-output /linux-llvm21-bin /CUDA /cudart_static.lib /CUDA-device-libs /ROCm /amdhip64.lib /Windows-ROCm-device-libs /Windows-ROCm-result.json /new-output");
    }
    buildWindowsAcppStaticHost({source, descriptorFile, descriptor: readBuildJson(descriptorFile), targetLlvm, nativeTools,
      cuda, cudartStatic, cudaDeviceLibs, rocm, amdhip64, rocmDeviceLibs, rocmReceipt, output,
      jobs: nativeBuildJobs(process.env)});
  } catch (error) {console.error(error instanceof Error ? error.message : "Windows ACPP static host SDK build failed"); process.exitCode = 1;}
}

module.exports = {windowsAcppStaticHostPlan, verifyAcppCache, auditAcppCompileCommands, buildWindowsAcppStaticHost, admitSource, archiveObjectInputs,
  verifyIntelCompilerIdentity, verifyWindowsRocmResources, nativeRuntimeInputs, cleanEnvironment, targets, artifacts, forbiddenOutputs, recipeFiles, buildInputKeys};
