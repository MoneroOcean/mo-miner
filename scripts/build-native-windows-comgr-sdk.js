"use strict";

// Cross-build the original COMGR static COFF target with the selected Intel Windows
// compiler. Linux tablegen tools and the native bc2h generator stay on separate roots.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {validateWindowsLlvmSdkConfig, windowsToolEnvironment, vfsFiles} = require("./build-native-windows-controller");
const {hasExactBuildFields, buildPathsOverlap, fileHash, buildFileInputs, sourceInputs, readBuildJson, readBuildText} = require("./build-source-inputs");
const {windowsIntelRuleOverride} = require("./prepare-native-jit-toolchain");
const {jitBuildOutputs} = require("./native-jit-stages");
const {isStaticArchive} = require("./native-audit");
const {nativeRuntimeLibraries} = require("./native-sdk-runtime");
const {verifyWindowsLlvmAdmission} = require("./build-native-windows-llvm-sdk");

const repo = path.resolve(__dirname, "..");
const revision = "27682a16360e33e37c4f3cc6adf9a620733f8fe1";
const resourceRevision = revision;
const resourceClrRevision = "5b591b2ff64aac4c1c08be879b0d5f89fd713d50";
const phaseTimeout = 3600000;
const targetTriple = "x86_64-pc-windows-msvc";
const patches = Object.freeze(["runtime/rocm-comgr-llvm21.patch", "runtime/rocm-comgr-resources.patch",
  "runtime/rocm-comgr-cross-tools.patch"]);
const hostSources = Object.freeze(["src/comgr-cache.cpp", "src/comgr-cache-command.cpp", "src/comgr-cache-bundler-command.cpp",
  "src/comgr-clang-command.cpp", "src/comgr-compiler.cpp", "src/comgr.cpp", "src/comgr-device-libs.cpp",
  "src/comgr-diagnostic-handler.cpp", "src/comgr-disassembly.cpp", "src/comgr-elfdump.cpp", "src/comgr-env.cpp",
  "src/comgr-metadata.cpp", "src/comgr-objdump.cpp", "src/comgr-signal.cpp", "src/comgr-spirv-command.cpp",
  "src/comgr-symbol.cpp", "src/comgr-symbolizer.cpp", "src/time-stat/time-stat.cpp"]);
const recipeFiles = Object.freeze(["scripts/build-native-windows-comgr-sdk.js", "scripts/build-native-windows-controller.js",
  "scripts/build-native-windows-llvm-sdk.js",
  "scripts/build-native-release.js", "scripts/build-source-inputs.js", "scripts/prepare-native-jit-toolchain.js",
  "scripts/native-jit-stages.js", "scripts/native-audit.js", "scripts/native-comgr-inputs.js", "scripts/native-sdk-runtime.js", ...patches]);
const requiredDeviceLibraries = Object.freeze(["asanrtl.bc", "hip.bc", "ockl.bc", "ocml.bc", "opencl.bc"]);

/** @param {unknown} value @returns {value is string} */
function literal(value) {
  return typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) && path.normalize(value) === value &&
    !/[;"$`\\]/.test(value) && !Array.from(value).some(c => c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127);
}

/** @param {string} command */
function commandSwitches(command) {
  return [...command.matchAll(/(?:^|[\s"])(\/(?:[A-Za-z][A-Za-z0-9_:+.=-]*)|-[A-Za-z][A-Za-z0-9_:+.=-]*)(?=[\s"]|$)/g)]
    .map(match => (match[1] || "").toUpperCase());
}

/** @param {string} value */
function cmake(value) {
  if (typeof value !== "string" || value.length > 65536 || /[\r\n\0]/.test(value)) {throw new Error("Invalid literal COMGR CMake value");}
  let equals = "==";
  for (const match of value.matchAll(/\](=*)\]/g)) {
    if (match[1] !== undefined && match[1].length >= equals.length) {equals = "=".repeat(match[1].length + 1);}
  }
  return "[" + equals + "[" + value + "]" + equals + "]";
}

/** @param {string} file @param {string} label */
function canonicalFile(file, label) {
  const stat = fs.statSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || fs.realpathSync(file) !== file) {throw new Error("Windows COMGR " + label + " is not a canonical regular file: " + file);}
  return file;
}

/** Preserve argv[0] while admitting only the installed native-LLVM clang name or its package alias. @param {string} file @param {string} root */
function canonicalNativeClang(file, root) {
  if (!literal(root) || fs.realpathSync(root) !== root || !fs.statSync(root).isDirectory()) {
    throw new Error("Windows COMGR native LLVM root is not canonical");
  }
  const bin = path.join(root, "bin"), selected = path.join(bin, "clang"), target = path.join(bin, "clang-21");
  if (file !== selected || fs.realpathSync(bin) !== bin || !fs.statSync(bin).isDirectory()) {
    throw new Error("Windows COMGR native Clang is outside the selected LLVM bin directory");
  }
  const stat = fs.lstatSync(selected, {throwIfNoEntry: false});
  if (!stat) {throw new Error("Windows COMGR native Clang is missing: " + selected);}
  if (stat.isFile() && fs.realpathSync(selected) === selected) {
    fs.accessSync(selected, fs.constants.X_OK); return selected;
  }
  if (!stat.isSymbolicLink() || fs.readlinkSync(selected) !== "clang-21" || fs.realpathSync(selected) !== target) {
    throw new Error("Windows COMGR native Clang alias must be the exact relative clang -> clang-21 package link");
  }
  const targetStat = fs.lstatSync(target, {throwIfNoEntry: false});
  if (!targetStat?.isFile() || targetStat.isSymbolicLink() || fs.realpathSync(target) !== target) {
    throw new Error("Windows COMGR native Clang alias target is not a canonical regular file");
  }
  fs.accessSync(target, fs.constants.X_OK);
  return selected;
}

/** @param {Record<string, unknown>} value @param {string} label @param {boolean} absolute */
function validateHashMap(value, label, absolute) {
  if (!value || typeof value !== "object" || Array.isArray(value) || !Object.keys(value).length || Object.keys(value).length > 65536 ||
      Object.entries(value).some(([file, hash]) => (absolute ? !literal(file) : typeof file !== "string" || !file || file.startsWith("/") ||
        file.split("/").some(part => !part || part === "." || part === "..")) || typeof hash !== "string" || !/^[0-9a-f]{64}$/.test(hash))) {
    throw new Error("Invalid Windows COMGR " + label + " hash map");
  }
}

/** Validate the preserved Windows HIP device receipt and all 71 registered bitcode files.
 * Old tool paths are provenance only: this function never runs them or rewrites the receipt.
 * @param {string} receiptFile @param {string} install */
function verifyWindowsDeviceResources(receiptFile, install) {
  canonicalFile(receiptFile, "device-resource receipt");
  const receipt = readBuildJson(receiptFile), root = path.dirname(receiptFile), expectedInstall = path.join(root, "install");
  if (!literal(install) || fs.realpathSync(install) !== install || install !== expectedInstall ||
      receipt?.version !== 1 || receipt.target !== "win32" || receipt.revision !== resourceRevision || receipt.clrRevision !== resourceClrRevision ||
      receipt.buildOnly !== true || receipt.publicationQualified !== false || receipt.executed === true || !receipt.recipe ||
      !receipt.header || !receipt.object) {throw new Error("Select the unchanged pinned Windows HIP resource receipt and install directory");}
  validateHashMap(receipt.source, "device source", false);
  validateHashMap(receipt.inputs, "device input", true);
  validateHashMap(receipt.dependencies, "device dependency", true);
  validateHashMap(receipt.device, "device output", true);
  const sourceRoot = path.join(root, "source"), recipeGraph = receipt.recipe.graph;
  validateHashMap(recipeGraph, "device recipe", false);
  if (!isDeepStrictEqual(receipt.source, sourceInputs(sourceRoot, ["device", "hiprtc"], []).files)) {
    throw new Error("Windows device-library source membership or contents differ from its receipt");
  }
  if (!receipt.recipe.files || typeof receipt.recipe.files !== "object" || Array.isArray(receipt.recipe.files) ||
      Object.entries(receipt.recipe.files).some(([name, hash]) => typeof name !== "string" || !name ||
        typeof hash !== "string" || !/^[0-9a-f]{64}$/.test(hash))) {throw new Error("Invalid historical Windows device-library recipe pins");}
  for (const group of [receipt.inputs, receipt.dependencies]) {
    if (!isDeepStrictEqual(group, buildFileInputs(Object.keys(group)))) {throw new Error("Windows HIP device input changed after its receipt");}
  }
  for (const record of [receipt.header, receipt.object]) {
    if (!record || !literal(record.path) || !record.path.startsWith(root + path.sep) || !/^[0-9a-f]{64}$/.test(String(record.sha256)) ||
        canonicalFile(record.path, "resource output") !== record.path || fileHash(record.path) !== record.sha256) {
      throw new Error("Windows HIP resource output differs from its receipt");
    }
  }
  const bitcodeRoot = path.join(install, "amdgcn/bitcode"), configFile = path.join(install, "lib/cmake/AMDDeviceLibs/AMDDeviceLibsConfig.cmake");
  const stat = fs.lstatSync(bitcodeRoot, {throwIfNoEntry: false});
  if (!stat?.isDirectory() || fs.realpathSync(bitcodeRoot) !== bitcodeRoot) {throw new Error("Windows HIP bitcode root is not canonical");}
  const bitcodeEntries = fs.readdirSync(bitcodeRoot, {withFileTypes: true});
  const actualNames = bitcodeEntries.map(entry => entry.name).sort();
  if (actualNames.length !== 71 || bitcodeEntries.some(entry => !entry.isFile() || !/^[A-Za-z0-9_.-]+\.bc$/.test(entry.name))) {
    throw new Error("Windows HIP device-library membership is not the preserved 71-file set");
  }
  const actualFiles = [...actualNames.map(name => path.join(bitcodeRoot, name)), configFile].sort();
  if (!hasExactBuildFields(receipt.device, actualFiles) || actualNames.some(name => !receipt.device[path.join(bitcodeRoot, name)]) ||
      !requiredDeviceLibraries.every(name => actualNames.includes(name))) {
    throw new Error("Windows HIP device-library membership differs from its pinned receipt");
  }
  for (const file of actualFiles) {
    canonicalFile(file, "device-library file");
    if (fileHash(file) !== receipt.device[file]) {throw new Error("Windows HIP device-library bytes differ from their receipt: " + file);}
  }
  const configText = readBuildText(configFile, 1 << 20), imports = [...configText.matchAll(/IMPORTED_LOCATION "\$\{AMD_DEVICE_LIBS_PREFIX\}\/amdgcn\/bitcode\/([A-Za-z0-9_.-]+\.bc)"/g)]
    .map(match => match[1] || "");
  if (imports.length !== 71 || new Set(imports).size !== imports.length || imports.sort().join("\n") !== actualNames.join("\n")) {
    throw new Error("Windows HIP CMake registry does not describe its exact bitcode set");
  }
  const sha256 = fileHash(receiptFile);
  if (fileHash(receiptFile) !== sha256) {throw new Error("Windows HIP resource receipt changed during admission");}
  return {receiptFile, sha256, revision: receipt.revision, clrRevision: receipt.clrRevision, device: receipt.device, files: actualFiles};
}

/** @param {string} root @param {string} name */
function nativeTool(root, name) {return path.join(root, "bin", name);}

/** Bind the PCH generator and Clang resource directory to native LLVM 21.
 * @param {string} version @param {string} resourceDir @param {ReturnType<typeof windowsComgrSdkPlan>} plan */
function verifyNativeClangIdentity(version, resourceDir, plan) {
  if (typeof version !== "string" || !/\bclang version 21\.1\.8\b/i.test(version) ||
      typeof resourceDir !== "string" || resourceDir !== plan.nativeClangResourceDir) {
    throw new Error("Native COMGR PCH compiler must be LLVM 21.1.8 with its selected resource directory");
  }
  return {version: "21.1.8", resourceDir};
}

/** `llvm-lib` mode rejects `--version`; bind it to the separately version-checked `llvm-ar` bytes.
 * @param {string} llvmLib @param {string} llvmAr @param {string} llvmArVersion */
function verifyNativeLibrarianIdentity(llvmLib, llvmAr, llvmArVersion) {
  if (typeof llvmArVersion !== "string" || !/\bLLVM version 21\.1\.8\b/.test(llvmArVersion)) {
    throw new Error("Native COMGR librarian needs a version-checked LLVM 21.1.8 llvm-ar");
  }
  const llvmLibSha256 = fileHash(llvmLib), llvmArSha256 = fileHash(llvmAr);
  if (llvmLibSha256 !== llvmArSha256) {throw new Error("Native COMGR llvm-lib is not byte-identical to its version-checked LLVM 21.1.8 llvm-ar");}
  return {version: llvmArVersion, sha256: llvmLibSha256};
}

/** Plan only the original static COMGR target and its genuine upstream bc2h host generator.
 * @param {{source: string, descriptorFile: string, descriptor: unknown, targetLlvm: string, nativeLlvm: string, hostCompiler: string,
 * deviceReceipt: string, deviceInstall: string, output: string, jobs: number, tools: {cmake: string, ninja: string}}} input
 * @param {{source: string, build: string}} [admittedLlvmLayout] */
function windowsComgrSdkPlan(input, admittedLlvmLayout) {
  const expectedKeys = ["descriptor", "descriptorFile", "deviceInstall", "deviceReceipt", "hostCompiler", "jobs", "nativeLlvm", "output", "source", "targetLlvm", "tools"];
  if (!input || !hasExactBuildFields(input, expectedKeys)) {throw new Error("Windows COMGR plan needs the explicit source, compiler, SDK and output inputs");}
  const config = validateWindowsLlvmSdkConfig(input.descriptor), {windowsSdk, tools: windowsTools} = config;
  const git = canonicalFile("/usr/bin/git", "trusted Git executable");
  if (!input.tools || !hasExactBuildFields(input.tools, ["cmake", "ninja"]) ||
      ![input.source, input.descriptorFile, input.targetLlvm, input.nativeLlvm, input.hostCompiler, input.deviceReceipt, input.deviceInstall,
        input.output, input.tools.cmake, input.tools.ninja, windowsTools.clangCl, windowsTools.lldLink, windowsTools.clangResourceInclude,
        windowsSdk.overlay, windowsSdk.sourceRoot, ...windowsSdk.includeRoots, ...windowsSdk.libraryRoots,
        ...windowsSdk.msvcLibraries, ...windowsSdk.ucrtLibraries].every(literal) ||
      !Number.isInteger(input.jobs) || input.jobs < 1 || input.jobs > 16) {
    throw new Error("Windows COMGR needs absolute literal paths and jobs in 1..16");
  }
  const native = /** @type {{clang: string, "llvm-lib": string, "llvm-ar": string, "llvm-ranlib": string,
    "llvm-tblgen": string, "clang-tblgen": string, "llvm-config": string}} */ (Object.fromEntries(
      ["clang", "llvm-lib", "llvm-ar", "llvm-ranlib", "llvm-tblgen", "clang-tblgen", "llvm-config"].map(name => [name, nativeTool(input.nativeLlvm, name)])));
  const nativeRuntimeRoots = [path.join(input.nativeLlvm, "lib")];
  const runtimeLibraryRoots = [...new Set([...nativeRuntimeRoots, path.join(path.dirname(path.dirname(input.hostCompiler)), "lib"),
    path.join(path.dirname(path.dirname(windowsTools.clangCl)), "lib")])];
  const targetLlvmSource = admittedLlvmLayout?.source || path.join(input.targetLlvm, "source"),
    targetLlvmBuild = admittedLlvmLayout?.build || path.join(input.targetLlvm, "build");
  if (![targetLlvmSource, targetLlvmBuild].every(literal) ||
      admittedLlvmLayout && !hasExactBuildFields(admittedLlvmLayout, ["build", "source"])) {
    throw new Error("Windows COMGR needs the admitted LLVM source/build layout");
  }
  const nativeBin = path.join(input.nativeLlvm, "bin"), build = path.join(input.output, "build"), sourceRoot = path.join(input.output, "source");
  const nativeClangResourceDir = path.join(input.nativeLlvm, "lib/clang/21");
  const snapshot = path.join(sourceRoot, "amd/comgr"), archive = path.join(input.output, "comgr-source.tar");
  const hostProject = path.join(input.output, "bc2h-host"), hostBuild = path.join(hostProject, "build"), hostHelper = path.join(hostBuild, "bc2h");
  const overlay = path.join(input.output, "windows-sdk-vfs.json"), ruleOverride = path.join(input.output, "intel-link-rules.cmake");
  const toolchainFile = path.join(input.output, "toolchain.cmake"), openclHeader = path.join(nativeClangResourceDir, "include/opencl-c.h"),
    openclBaseHeader = path.join(nativeClangResourceDir, "include/opencl-c-base.h");
  const deviceConfig = path.join(input.deviceInstall, "lib/cmake/AMDDeviceLibs");
  const llvmDir = path.join(targetLlvmBuild, "lib/cmake/llvm"), clangDir = path.join(targetLlvmBuild, "lib/cmake/clang");
  const lldDir = path.join(targetLlvmBuild, "lib/cmake/lld"), deviceDir = path.join(deviceConfig, "AMDDeviceLibsConfig.cmake");
  const virtual = [...vfsFiles(windowsSdk.overlay)].filter(([name]) => windowsSdk.includeRoots.some(root =>
    name.toLowerCase().startsWith(root.replace(/\/+$/, "").toLowerCase() + "/")));
  if (!virtual.length || readBuildJson(windowsSdk.overlay)["case-sensitive"] !== false) {
    throw new Error("Windows COMGR needs a case-insensitive OS SDK VFS overlay");
  }
  const overlayText = JSON.stringify({version: 0, "case-sensitive": false, "use-external-names": true,
    roots: virtual.map(([name, file]) => ({type: "file", name, "external-contents": file}))});
  const includeFlags = ["--target=" + targetTriple, "-nostdinc", "-I" + windowsTools.clangResourceInclude,
    "-Xclang", "-ivfsoverlay", "-Xclang", overlay, ...windowsSdk.includeRoots.map(root => "-I" + root), "/MT", "/O2", "/Gy", "/Gw"];
  const commonFlags = includeFlags.map(value => '"' + value + '"').join(" ");
  const cxxFlags = commonFlags + ' "/EHsc" "/GR" "/std:c++17"';
  const releaseFlags = "/O2 /DNDEBUG";
  const linkFlags = ["/MACHINE:X64", "/INCREMENTAL:NO", ...windowsSdk.libraryRoots.map(root => '/LIBPATH:"' + root + '"')].join(" ");
  const ruleOverrideText = windowsIntelRuleOverride(linkFlags, [...windowsSdk.msvcLibraries, ...windowsSdk.ucrtLibraries, ...windowsSdk.importLibraries]) +
    `set(CMAKE_C_FLAGS_RELEASE_INIT ${cmake(releaseFlags)})\n` +
    `set(CMAKE_CXX_FLAGS_RELEASE_INIT ${cmake(releaseFlags)})\n` +
    "set(CMAKE_INCLUDE_SYSTEM_FLAG_C /external:I)\nset(CMAKE_INCLUDE_SYSTEM_FLAG_CXX /external:I)\n";
  const compilerTools = {CMAKE_C_COMPILER: windowsTools.clangCl, CMAKE_CXX_COMPILER: windowsTools.clangCl,
    CMAKE_LINKER: windowsTools.lldLink, CMAKE_AR: native["llvm-lib"], CMAKE_RANLIB: native["llvm-ranlib"]};
  const toolchain = ["set(CMAKE_SYSTEM_NAME Windows CACHE STRING \"\" FORCE)", "set(CMAKE_SYSTEM_PROCESSOR AMD64)", "set(CMAKE_SYSTEM_VERSION 10.0)",
    "set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY CACHE STRING \"\" FORCE)", "set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)",
    `set(CMAKE_MSVC_RUNTIME_LIBRARY ${cmake("MultiThreaded")} CACHE STRING "" FORCE)`,
    `set(CMAKE_INCLUDE_SYSTEM_FLAG_C ${cmake("/external:I")} CACHE STRING "" FORCE)`,
    `set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX ${cmake("/external:I")} CACHE STRING "" FORCE)`,
    ...Object.entries(compilerTools).map(([name, file]) => `set(${name} ${cmake(file)} CACHE FILEPATH "" FORCE)`),
    `set(CMAKE_C_COMPILER_TARGET ${cmake(targetTriple)} CACHE STRING "" FORCE)`, `set(CMAKE_CXX_COMPILER_TARGET ${cmake(targetTriple)} CACHE STRING "" FORCE)`,
    `set(CMAKE_USER_MAKE_RULES_OVERRIDE_C ${cmake(ruleOverride)})`, `set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX ${cmake(ruleOverride)})`,
    `set(CMAKE_C_FLAGS_INIT ${cmake(commonFlags)})`, `set(CMAKE_CXX_FLAGS_INIT ${cmake(cxxFlags)})`,
    `set(CMAKE_EXE_LINKER_FLAGS_INIT ${cmake(linkFlags)})`, `set(CMAKE_SHARED_LINKER_FLAGS_INIT ${cmake(linkFlags)})`,
    `set(CMAKE_MODULE_LINKER_FLAGS_INIT ${cmake(linkFlags)})`, `set(CMAKE_STATIC_LINKER_FLAGS_INIT ${cmake("/MACHINE:X64")})`,
    `set(LLVM_NATIVE_TOOL_DIR ${cmake(nativeBin)} CACHE PATH "" FORCE)`,
    `set(LLVM_TABLEGEN ${cmake(native["llvm-tblgen"])} CACHE FILEPATH "" FORCE)`,
    `set(CLANG_TABLEGEN ${cmake(native["clang-tblgen"])} CACHE FILEPATH "" FORCE)`,
    `set(CMAKE_FIND_ROOT_PATH ${cmake(windowsSdk.sourceRoot)})`,
    ...["LIBRARY", "INCLUDE", "PACKAGE"].map(name => `set(CMAKE_FIND_ROOT_PATH_MODE_${name} ONLY)`),
    "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)", ""].join("\n");
  const definitions = [
    ["CMAKE_TOOLCHAIN_FILE", toolchainFile], ["CMAKE_MAKE_PROGRAM", input.tools.ninja], ["CMAKE_BUILD_TYPE", "Release"],
    ["CMAKE_EXPORT_COMPILE_COMMANDS", "ON"], ["CMAKE_SUPPRESS_REGENERATION", "ON"], ["CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded"],
    ["CMAKE_TRY_COMPILE_CONFIGURATION", "Release"],
    ["CMAKE_DISABLE_FIND_PACKAGE_ROCM", "ON"], ["CMAKE_DISABLE_FIND_PACKAGE_HIP", "ON"], ["CMAKE_DISABLE_FIND_PACKAGE_OpenCL", "ON"],
    ["COMGR_BUILD_SHARED_LIBS", "OFF"], ["COMGR_EMBEDDED_RESOURCES", "ON"], ["COMGR_DISABLE_SPIRV", "ON"],
    ["LLVM_LINK_LLVM_DYLIB", "OFF"], ["BUILD_SHARED_LIBS", "OFF"], ["BUILD_TESTING", "OFF"], ["ENABLE_ASAN_PACKAGING", "OFF"],
    ["LLVM_DIR", llvmDir], ["Clang_DIR", clangDir], ["LLD_DIR", lldDir], ["AMDDeviceLibs_DIR", deviceConfig],
    ["LLVM_NATIVE_TOOL_DIR", nativeBin], ["LLVM_TABLEGEN", native["llvm-tblgen"]], ["CLANG_TABLEGEN", native["clang-tblgen"]],
    ["COMGR_OPENCL_C_HEADER", openclHeader], ["COMGR_NATIVE_CLANG_EXECUTABLE", native.clang],
    ["COMGR_NATIVE_CLANG_RESOURCE_DIR", nativeClangResourceDir], ["COMGR_BC2H_EXECUTABLE", hostHelper],
    ["CMAKE_INSTALL_PREFIX", path.join(input.output, "unused-install-prefix")],
  ];
  const configure = [input.tools.cmake, "-S", snapshot, "-B", build, "-G", "Ninja",
    ...definitions.map(([name, value]) => "-D" + name + "=" + value)];
  const hostProjectText = ["cmake_minimum_required(VERSION 3.20)", "project(mom_comgr_bc2h_host LANGUAGES C)",
    `include(${cmake(path.join(snapshot, "cmake/bc2h.cmake"))})`, ""].join("\n");
  const hostConfigure = [input.tools.cmake, "-S", hostProject, "-B", hostBuild, "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=" + input.tools.ninja,
    "-DCMAKE_C_COMPILER=" + input.hostCompiler, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"];
  const hostBuildCommand = [input.tools.cmake, "--build", hostBuild, "--parallel", "1", "--target", "bc2h"];
  const nativeQueries = /** @type {Record<string, [string, string[]]>} */ (Object.fromEntries(
    Object.entries(native).filter(([name]) => name !== "llvm-lib").map(([name, file]) => [name, [file, ["--version"]]])));
  return {version: 1, role: "windows-comgr-static-host", revision, patches: [...patches], source: input.source,
    descriptorFile: input.descriptorFile, targetLlvm: input.targetLlvm, targetLlvmSource, targetLlvmBuild,
    nativeLlvm: input.nativeLlvm, hostCompiler: input.hostCompiler,
    deviceReceipt: input.deviceReceipt, deviceInstall: input.deviceInstall, output: input.output, jobs: input.jobs,
    tools: {...input.tools}, config, native, nativeBin, compilerTools, targetTriple, build, sourceRoot, snapshot, archive,
    hostProject, hostProjectText, hostBuild, hostHelper, hostConfigure, hostBuildCommand, nativeQueries, nativeRuntimeRoots, runtimeLibraryRoots,
    overlay, overlayText, ruleOverride, ruleOverrideText, toolchainFile, toolchain, openclHeader, deviceConfig, deviceDir,
    llvmDir, clangDir, lldDir, virtual, nativeClangResourceDir, openclBaseHeader, definitions: Object.fromEntries(definitions), configure,
    git,
    applyCommands: patches.map(file => [git, "apply", "--check", path.join(repo, file)]),
    extract: ["/usr/bin/tar", "--extract", "--file", archive, "--directory", sourceRoot, "--no-same-owner", "--no-same-permissions"],
    archiveCommand: [git, "--no-optional-locks", "-C", input.source, "archive", "--format=tar", "--prefix=amd/comgr/", "--output=" + archive,
      revision + ":amd/comgr"],
    target: "amd_comgr", hostSources: [...hostSources]};
}

/** @param {string} file @param {string} command @param {string[]} args @param {string} cwd
 * @param {NodeJS.ProcessEnv} environment */
function runLogged(file, command, args, cwd, environment) {
  const fd = fs.openSync(file, "wx");
  try {
    const result = spawnSync(command, args, {cwd, env: environment, stdio: ["ignore", fd, fd], timeout: phaseTimeout, maxBuffer: 1 << 20});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Windows COMGR stage failed; inspect " + file, {cause: result.error});}
  } finally {fs.closeSync(fd);}
}

/** @param {string} command @param {string[]} args @param {string} name @param {string} output
 * @param {NodeJS.ProcessEnv} environment @param {string} cwd */
function query(command, args, name, output, environment, cwd) {
  const result = spawnSync(command, args, {cwd, env: environment, encoding: "utf8", timeout: 60000, maxBuffer: 4 << 20});
  fs.writeFileSync(path.join(output, name), (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
  if (result.error || result.signal || result.status !== 0) {throw new Error("Windows COMGR tool probe failed: " + name, {cause: result.error});}
  return ((result.stdout || "") + (result.stderr || "")).trim();
}

/** @param {NodeJS.ProcessEnv} input @param {ReturnType<typeof windowsComgrSdkPlan>} plan @param {string} helperDirectory */
function cleanEnvironment(input, plan, helperDirectory) {
  const environment = windowsToolEnvironment(input);
  for (const key of Object.keys(environment)) {
    if (/^(?:CC$|CXX$|AR$|RC$|CMAKE_|LLVM_|CLANG_|AMD_COMGR|COMGR_|ROCM_|HIP|OPENCL|PKG_CONFIG_PATH$|MAKEFLAGS$|NINJAFLAGS$|GIT_)/i.test(key) ||
        /^(?:LD_PRELOAD|LD_AUDIT|LD_LIBRARY_PATH|OMP_)/i.test(key)) {delete environment[key];}
  }
  environment["GIT_OPTIONAL_LOCKS"] = "0";
  environment["GIT_TERMINAL_PROMPT"] = "0";
  environment["GIT_NO_LAZY_FETCH"] = "1";
  environment["GIT_NO_REPLACE_OBJECTS"] = "1";
  environment["LD_LIBRARY_PATH"] = plan.runtimeLibraryRoots.join(path.delimiter);
  environment["PATH"] = [helperDirectory, plan.nativeBin, path.dirname(plan.hostCompiler), path.dirname(plan.config.tools.clangCl), "/usr/bin", "/bin"].join(path.delimiter);
  return environment;
}

/** Only immediate shared libraries under the pinned native LLVM runtime roots are admitted.
 * @param {ReturnType<typeof windowsComgrSdkPlan>} plan @returns {string[]} */
function nativeRuntimeInputs(plan) {
  return nativeRuntimeLibraries(plan.nativeRuntimeRoots, "Windows COMGR");
}

/** Verify both CMake's compiler identity and the selected oneAPI 2026.2 version.
 * @param {string} build */
function verifyIntelCompilerIdentity(build) {
  const directory = path.join(build, "CMakeFiles"), candidates = fs.readdirSync(directory)
    .filter(name => /^\d+\.\d+(?:\.\d+)?$/.test(name)).map(name => path.join(directory, name));
  /** @param {string} language */
  const read = language => {
    const file = candidates.map(root => path.join(root, "CMake" + language + "Compiler.cmake")).find(name => fs.existsSync(name));
    if (!file) {throw new Error("Windows COMGR CMake omitted its " + language + " compiler identity");}
    const text = readBuildText(file, 1 << 20), id = text.match(new RegExp(`set\\(CMAKE_${language}_COMPILER_ID "([^"]+)"\\)`))?.[1];
    const version = text.match(new RegExp(`set\\(CMAKE_${language}_COMPILER_VERSION "([^"]+)"\\)`))?.[1];
    if (id !== "IntelLLVM" || !/^2026\.2(?:\.|$)/.test(version || "")) {
      throw new Error("Windows COMGR must use actual IntelLLVM 2026.2 for " + language);
    }
    return {file, id, version};
  };
  return {c: read("C"), cxx: read("CXX")};
}

/** @param {string} build @param {string} compiler */
function verifyNativeIntelCompilerIdentity(build, compiler) {
  const candidates = fs.readdirSync(path.join(build, "CMakeFiles"))
    .filter(name => /^\d+\.\d+(?:\.\d+)?$/.test(name)).map(name => path.join(build, "CMakeFiles", name));
  const file = candidates.map(root => path.join(root, "CMakeCCompiler.cmake")).find(name => fs.existsSync(name));
  if (!file) {throw new Error("Native COMGR helper CMake omitted the C compiler identity");}
  const text = readBuildText(file, 1 << 20), id = text.match(/set\(CMAKE_C_COMPILER_ID "([^"]+)"\)/)?.[1],
    version = text.match(/set\(CMAKE_C_COMPILER_VERSION "([^"]+)"\)/)?.[1], selected = text.match(/set\(CMAKE_C_COMPILER "([^"]+)"\)/)?.[1];
  if (id !== "IntelLLVM" || !/^2026\.2(?:\.|$)/.test(version || "") || path.resolve(selected || "") !== compiler) {
    throw new Error("Native bc2h helper must use the selected IntelLLVM 2026.2 icx compiler");
  }
  return {file, id, version};
}

/** @param {string} file @param {string} compiler @param {string} build @param {string} generated @param {string} object */
function auditBc2hCompileCommands(file, compiler, build, generated, object) {
  const entries = JSON.parse(readBuildText(file, 4 << 20));
  if (!Array.isArray(entries) || entries.length !== 1) {throw new Error("Native bc2h build must contain exactly one C translation unit");}
  const entry = entries[0], source = typeof entry?.file === "string" ? path.resolve(entry.directory || build, entry.file) : "",
    output = typeof entry?.output === "string" ? path.resolve(entry.directory || build, entry.output) : "";
  if (typeof entry?.command !== "string" || typeof entry.directory !== "string" ||
      !(entry.command.startsWith(compiler + " ") || entry.command.startsWith('"' + compiler + '" ')) ||
      path.resolve(entry.directory) !== fs.realpathSync(build) || source !== generated || output !== object ||
      !output.startsWith(fs.realpathSync(build) + path.sep) || /--target=|x86_64-pc-windows-msvc/i.test(entry.command)) {
    throw new Error("Native bc2h compile command is not the selected Intel Linux C compilation");
  }
  const stat = fs.statSync(output, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.size < 20) {throw new Error("Native bc2h object is missing");}
  const fd = fs.openSync(output, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK), header = Buffer.alloc(4);
  try {
    if (fs.readSync(fd, header, 0, 4, 0) !== 4 || header.toString("hex") !== "7f454c46") {
    // The CMake object is an ELF relocatable and starts with the ELF magic.
      throw new Error("Native bc2h object is not an ELF Linux object");
    }
  } finally {fs.closeSync(fd);}
  return {path: file, sha256: fileHash(file), source, output, sha256Object: fileHash(output), bytes: stat.size};
}

/** @param {string} file @param {ReturnType<typeof windowsComgrSdkPlan>} plan */
function verifyComgrCache(file, plan) {
  const text = readBuildText(file, 8 << 20), values = new Map();
  for (const line of text.split(/\r?\n/)) {const entry = line.match(/^([^/#][^:]*):[^=]*=(.*)$/); if (entry) {values.set(entry[1], entry[2]);}}
  const expected = {CMAKE_SYSTEM_NAME: "Windows", CMAKE_C_COMPILER: plan.compilerTools.CMAKE_C_COMPILER,
    CMAKE_CXX_COMPILER: plan.compilerTools.CMAKE_CXX_COMPILER, CMAKE_C_COMPILER_TARGET: plan.targetTriple,
    CMAKE_CXX_COMPILER_TARGET: plan.targetTriple, CMAKE_LINKER: plan.compilerTools.CMAKE_LINKER,
    CMAKE_AR: plan.compilerTools.CMAKE_AR, CMAKE_RANLIB: plan.compilerTools.CMAKE_RANLIB,
    CMAKE_INCLUDE_SYSTEM_FLAG_C: "/external:I", CMAKE_INCLUDE_SYSTEM_FLAG_CXX: "/external:I",
    CMAKE_MAKE_PROGRAM: plan.tools.ninja, CMAKE_MSVC_RUNTIME_LIBRARY: "MultiThreaded", CMAKE_BUILD_TYPE: "Release",
    CMAKE_C_FLAGS_RELEASE: "/O2 /DNDEBUG", CMAKE_CXX_FLAGS_RELEASE: "/O2 /DNDEBUG", CMAKE_TRY_COMPILE_CONFIGURATION: "Release",
    COMGR_BUILD_SHARED_LIBS: "OFF", COMGR_EMBEDDED_RESOURCES: "ON", COMGR_DISABLE_SPIRV: "ON",
    COMGR_BC2H_EXECUTABLE: plan.hostHelper, LLVM_LINK_LLVM_DYLIB: "OFF", LLVM_DIR: plan.llvmDir,
    Clang_DIR: plan.clangDir, LLD_DIR: plan.lldDir, AMDDeviceLibs_DIR: plan.deviceConfig,
    COMGR_OPENCL_C_HEADER: plan.openclHeader, COMGR_NATIVE_CLANG_EXECUTABLE: plan.native.clang,
    COMGR_NATIVE_CLANG_RESOURCE_DIR: plan.nativeClangResourceDir, CMAKE_EXPORT_COMPILE_COMMANDS: "ON",
    LLVM_NATIVE_TOOL_DIR: plan.nativeBin, LLVM_TABLEGEN: plan.native["llvm-tblgen"], CLANG_TABLEGEN: plan.native["clang-tblgen"],
    CMAKE_TRY_COMPILE_TARGET_TYPE: "STATIC_LIBRARY"};
  for (const [name, value] of Object.entries(expected)) {if (values.get(name) !== value) {throw new Error("Windows COMGR CMake configuration drifted: " + name);}}
  for (const name of ["CMAKE_DISABLE_FIND_PACKAGE_ROCM", "CMAKE_DISABLE_FIND_PACKAGE_HIP", "CMAKE_DISABLE_FIND_PACKAGE_OpenCL"]) {
    if (values.get(name) !== "ON") {throw new Error("Windows COMGR must not discover host packages: " + name);}
  }
  return file;
}

/** Audit the original COMGR translation units before and after compilation.
 * @param {string} compileCommands @param {string} compiler @param {string} source @param {string} build @param {boolean} built */
function auditComgrCompileCommands(compileCommands, compiler, source, build, built) {
  const stat = fs.statSync(compileCommands);
  if (!stat.isFile() || stat.size < 2 || stat.size > (64 << 20)) {throw new Error("Invalid Windows COMGR compile database");}
  const entries = JSON.parse(fs.readFileSync(compileCommands, "utf8"));
  if (!Array.isArray(entries) || entries.length !== hostSources.length) {throw new Error("Windows COMGR must compile exactly its 18 original host translation units");}
  const sourceRoot = fs.realpathSync(source) + path.sep, buildRoot = fs.realpathSync(build), seenSources = new Set(), seenOutputs = new Set();
  const objects = [];
  for (const entry of entries) {
    const switches = typeof entry?.command === "string" ? commandSwitches(entry.command) : [], runtime = switches.filter(flag => /^(?:\/|-)(?:MT|MD)/.test(flag));
    const file = typeof entry?.file === "string" ? path.resolve(entry.directory || build, entry.file) : "";
    const output = typeof entry?.output === "string" ? path.resolve(entry.directory || build, entry.output) : "";
    const extension = path.extname(file).toLowerCase();
    if (!entry || typeof entry.command !== "string" || typeof entry.directory !== "string" || entry.command.length > (1 << 20) ||
        /[\r\n\0]/.test(entry.command) || !(entry.command.startsWith(compiler + " ") || entry.command.startsWith('"' + compiler + '" ')) ||
        !entry.command.includes("--target=" + targetTriple) || !runtime.length || runtime.some(flag => !["/MT", "-MT"].includes(flag)) ||
        switches.some(flag => flag.startsWith("/MD") || flag.startsWith("-MD") || flag === "-ISYSTEM" || flag.startsWith("-FOPENMP") || flag.startsWith("/OPENMP")) ||
        /libgomp/i.test(entry.command) ||
        ![".cc", ".cpp", ".cxx", ".c++"].includes(extension) || !/^\/EHSC$/.test(switches.find(flag => flag.startsWith("/EH")) || "") ||
        !switches.includes("/GR") || (path.resolve(entry.directory) !== buildRoot && !path.resolve(entry.directory).startsWith(buildRoot + path.sep)) ||
        !file.startsWith(sourceRoot) || !output.startsWith(buildRoot + path.sep) || !output.endsWith(".obj") ||
        seenSources.has(file) || seenOutputs.has(output)) {
      throw new Error("Windows COMGR compile command has the wrong compiler, ABI, source or COFF output");
    }
    const relative = path.relative(sourceRoot, file).split(path.sep).join("/");
    if (!hostSources.includes(relative)) {throw new Error("Windows COMGR compiled a non-original source: " + relative);}
    seenSources.add(file); seenOutputs.add(output);
    const object = fs.statSync(output, {throwIfNoEntry: false});
    if (built) {
      if (!object?.isFile() || object.size < 20) {throw new Error("Windows COMGR COFF object is missing: " + output);}
      const fd = fs.openSync(output, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK), header = Buffer.alloc(2);
      try {if (fs.readSync(fd, header, 0, 2, 0) !== 2 || header.readUInt16LE(0) !== 0x8664) {throw new Error("Windows COMGR output is not an x64 COFF object: " + output);}} finally {fs.closeSync(fd);}
      objects.push({source: file, output, sha256: fileHash(output), bytes: object.size});
    } else if (object) {throw new Error("Windows COMGR object exists before its build phase: " + output);}
  }
  if (hostSources.some(name => !seenSources.has(path.join(sourceRoot, name)))) {throw new Error("Windows COMGR compile database omitted an original host translation unit");}
  return {path: compileCommands, sha256: fileHash(compileCommands), count: entries.length, objects};
}

/** @param {string} listing @param {{output: string}[]} objects @param {string} build */
function auditComgrArchiveMembers(listing, objects, build) {
  const buildRoot = fs.realpathSync(build), members = listing.trim().split(/\r?\n/).filter(Boolean);
  const expected = objects.map(row => path.relative(buildRoot, path.resolve(row.output)).split(path.sep).join("/")).sort();
  if (objects.length !== hostSources.length || members.length !== hostSources.length ||
      expected.some(name => !/^CMakeFiles\/amd_comgr\.dir\/(?:[A-Za-z0-9_.+-]+\/)*[A-Za-z0-9_.+-]+\.obj$/.test(name)) ||
      members.some(name => !/^CMakeFiles\/amd_comgr\.dir\/(?:[A-Za-z0-9_.+-]+\/)*[A-Za-z0-9_.+-]+\.obj$/.test(name)) ||
      new Set(members).size !== members.length || members.sort().join("\n") !== expected.join("\n")) {
    throw new Error("Windows COMGR archive does not contain exactly its 18 original COFF objects");
  }
  return members;
}

/** @param {string} build @param {string} source */
function comgrArchive(build, source) {
  const reply = path.join(build, ".cmake/api/v1/reply"), indexes = fs.readdirSync(reply)
    .filter(name => /^index-[A-Za-z0-9_.-]+\.json$/.test(name));
  if (indexes.length !== 1 || !indexes[0]) {throw new Error("Missing or ambiguous COMGR CMake File API index");}
  const index = readBuildJson(path.join(reply, indexes[0])), modelName = index.reply?.["codemodel-v2"]?.jsonFile;
  if (typeof modelName !== "string" || path.basename(modelName) !== modelName) {throw new Error("COMGR CMake File API omitted its codemodel");}
  const model = readBuildJson(path.join(reply, modelName)), targets = model.configurations?.[0]?.targets;
  if (!Array.isArray(targets) || model.configurations.length !== 1 || path.resolve(model.paths?.source || "") !== fs.realpathSync(source) ||
      path.resolve(model.paths?.build || "") !== fs.realpathSync(build)) {throw new Error("COMGR CMake codemodel has unselected roots");}
  const matches = targets.map(row => {
    if (typeof row?.jsonFile !== "string" || path.basename(row.jsonFile) !== row.jsonFile) {throw new Error("Invalid COMGR File API target reference");}
    return readBuildJson(path.join(reply, row.jsonFile));
  }).filter(row => row?.name === "amd_comgr");
  if (matches.length !== 1) {throw new Error("COMGR CMake File API omitted or duplicated the original amd_comgr target");}
  const target = matches[0], artifact = target.artifacts?.[0]?.path;
  const file = typeof artifact === "string" ? (path.isAbsolute(artifact) ? artifact : path.resolve(build, artifact)) : "";
  if (target.name !== "amd_comgr" || target.type !== "STATIC_LIBRARY" || target.artifacts?.length !== 1 || !file.startsWith(fs.realpathSync(build) + path.sep) ||
      !file.toLowerCase().endsWith(".lib")) {throw new Error("COMGR File API target is not the original static COFF library");}
  return {target, archive: file};
}

/** @param {string} output @param {unknown} value @param {boolean} [replace] */
function writeReceipt(output, value, replace = false) {
  const temporary = output + ".next";
  if (fs.existsSync(temporary) || (!replace && fs.existsSync(output))) {throw new Error("Windows COMGR receipt already exists");}
  fs.writeFileSync(temporary, JSON.stringify(value, null, 2) + "\n", {flag: "wx", mode: 0o600});
  fs.renameSync(temporary, output);
}

/** Query an admitted same-run Windows LLVM build, prepare original source and the native bc2h
 * tool, then configure COMGR. This command never builds the Windows COMGR archive itself.
 * @param {{source: string, descriptorFile: string, targetLlvm: string, nativeLlvm: string, hostCompiler: string, deviceReceipt: string,
 * deviceInstall: string, output: string, jobs: number}} input
 * @param {NodeJS.ProcessEnv} [environment] */
function configureWindowsComgrSdk(input, environment = process.env) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("Windows COMGR cross-build requires Linuxx64 and MOM_AOT_BUILD_ONLY=1");
  }
  const descriptor = validateWindowsLlvmSdkConfig(readBuildJson(input.descriptorFile));
  const llvm = verifyWindowsLlvmAdmission(input.targetLlvm), llvmTools = {cmake: llvm.receipt.plan.tools.cmake, ninja: llvm.receipt.plan.tools.ninja};
  const llvmPlan = llvm.receipt.plan, admittedLlvmLayout = {source: llvmPlan.source,
    build: path.join(llvmPlan.output, "build")};
  const plan = windowsComgrSdkPlan({...input, descriptor, tools: llvmTools}, admittedLlvmLayout);
  const outputParent = fs.realpathSync(path.dirname(input.output)), destination = path.join(outputParent, path.basename(input.output));
  if (destination !== input.output || fs.existsSync(input.output)) {throw new Error("Select a new canonical Windows COMGR output directory");}
  const device = verifyWindowsDeviceResources(input.deviceReceipt, input.deviceInstall);
  if (llvm.receipt.plan.output !== input.targetLlvm) {throw new Error("Windows COMGR target LLVM root differs from its admission");}
  const inputRoots = [repo, input.source, input.descriptorFile, input.targetLlvm, input.nativeLlvm, input.hostCompiler,
    input.deviceReceipt, input.deviceInstall, plan.config.windowsSdk.overlay, plan.config.windowsSdk.sourceRoot,
    ...plan.nativeRuntimeRoots,
    ...plan.config.windowsSdk.includeRoots, ...plan.config.windowsSdk.libraryRoots,
    ...plan.config.windowsSdk.msvcLibraries, ...plan.config.windowsSdk.ucrtLibraries, plan.targetLlvmSource, plan.targetLlvmBuild,
    plan.config.tools.clangCl, plan.config.tools.lldLink, plan.config.tools.clangResourceInclude, plan.tools.cmake, plan.tools.ninja];
  for (const root of inputRoots) {
    const selected = fs.existsSync(root) ? fs.realpathSync(root) : path.resolve(root);
    if (buildPathsOverlap(selected, destination)) {throw new Error("Windows COMGR output overlaps an input");}
  }
  fs.mkdirSync(input.output, {mode: 0o700}); fs.mkdirSync(plan.sourceRoot, {mode: 0o700}); fs.mkdirSync(plan.hostProject, {mode: 0o700});
  fs.mkdirSync(plan.build, {mode: 0o700});
  const env = cleanEnvironment(environment, plan, plan.nativeBin);
  const revisionLog = query(plan.git, ["--no-optional-locks", "-C", input.source, "rev-parse", revision + "^{commit}"], "revision.log", input.output, env, repo);
  if (revisionLog !== revision) {throw new Error("COMGR source does not contain the pinned upstream revision");}
  const includeFiles = /** @type {string[]} */ ([]);
  /** @param {string} root */
  function visit(root) {
    for (const entry of fs.readdirSync(root, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(root, entry.name);
      if (entry.isDirectory()) {visit(file);} else if (entry.isFile() || entry.isSymbolicLink()) {includeFiles.push(file);} else {throw new Error("Invalid Intel Clang resource input: " + file);}
      if (includeFiles.length > 65536) {throw new Error("Intel Clang resource inventory exceeds bounds");}
    }
  }
  visit(plan.config.tools.clangResourceInclude);
  const nativeLibraries = nativeRuntimeInputs(plan);
  const virtual = plan.virtual.map(([, file]) => file);
  const llvmPackages = ["llvm/LLVMConfig.cmake", "clang/ClangConfig.cmake", "lld/LLDConfig.cmake"]
    .map(name => path.join(plan.targetLlvmBuild, "lib/cmake", name));
  if (fs.realpathSync(plan.nativeClangResourceDir) !== plan.nativeClangResourceDir) {throw new Error("Native Clang resource directory is not canonical");}
  canonicalNativeClang(plan.native.clang, input.nativeLlvm);
  canonicalFile(plan.openclHeader, "native LLVM 21 OpenCL header");
  canonicalFile(plan.openclBaseHeader, "native LLVM 21 OpenCL base header");
  const selectedInputs = [input.descriptorFile, llvm.receiptFile, llvm.admissionFile, llvm.objectsFile, input.deviceReceipt, plan.git,
    plan.tools.cmake, plan.tools.ninja, plan.config.tools.clangCl, plan.config.tools.lldLink, plan.config.windowsSdk.overlay,
    ...includeFiles, ...virtual, ...plan.config.windowsSdk.msvcLibraries, ...plan.config.windowsSdk.ucrtLibraries,
    plan.openclHeader, plan.openclBaseHeader, ...nativeLibraries,
    ...Object.values(plan.native), input.hostCompiler, ...Object.keys(device.device), ...llvmPackages,
    ...recipeFiles.map(file => path.join(repo, file))];
  const fingerprints = buildFileInputs(selectedInputs), recipe = sourceInputs(repo, [], [...recipeFiles]);
  if (fs.realpathSync(plan.config.windowsSdk.overlay) !== plan.config.windowsSdk.overlay) {throw new Error("Windows COMGR OS VFS overlay is not canonical");}
  const intel = query(plan.config.tools.clangCl, ["--version"], "intel-windows-version.log", input.output, env, repo);
  const nativeIntel = query(input.hostCompiler, ["--version"], "intel-host-version.log", input.output, env, repo);
  if (!/\b2026\.2(?:\.|\b)/.test(intel) || !/\bIntel|oneAPI\b/i.test(intel) || !/\b2026\.2(?:\.|\b)/.test(nativeIntel) || !/\bIntel|oneAPI\b/i.test(nativeIntel)) {
    throw new Error("Windows COMGR requires the selected Intel 2026.2 cross and native C compilers");
  }
  const nativeVersions = /** @type {Record<string, string>} */ ({});
  for (const [name, [file, args]] of Object.entries(plan.nativeQueries)) {
    const text = query(file, args, "native-" + name + "-version.log", input.output, env, repo);
    if (name === "llvm-config" ? text !== "21.1.8" : name === "clang" ? !/\bclang version 21\.1\.8\b/i.test(text) : !/\bLLVM version 21\.1\.8\b/.test(text)) {
      throw new Error("Native COMGR utility is not LLVM 21.1.8: " + file);
    }
    nativeVersions[file] = text;
  }
  const llvmArVersion = nativeVersions[plan.native["llvm-ar"]];
  if (typeof llvmArVersion !== "string" || !llvmArVersion) {throw new Error("Native COMGR llvm-ar omitted its version evidence");}
  const librarian = verifyNativeLibrarianIdentity(plan.native["llvm-lib"], plan.native["llvm-ar"], llvmArVersion);
  nativeVersions[plan.native["llvm-lib"]] = librarian.version;
  const nativeClangResource = query(plan.native.clang, ["-print-resource-dir"], "native-clang-resource-dir.log", input.output, env, repo);
  const nativeClangVersion = nativeVersions[plan.native.clang];
  if (!nativeClangVersion) {throw new Error("Native COMGR PCH compiler omitted its version evidence");}
  const nativeClangIdentity = verifyNativeClangIdentity(nativeClangVersion, nativeClangResource, plan);
  runLogged(path.join(input.output, "git-archive.log"), plan.archiveCommand[0] || "", plan.archiveCommand.slice(1), repo, env);
  const archiveHash = fileHash(plan.archive);
  runLogged(path.join(input.output, "extract.log"), plan.extract[0] || "", plan.extract.slice(1), repo, env);
  for (const patch of patches) {
    runLogged(path.join(input.output, "patch-check-" + path.basename(patch) + ".log"), plan.git, ["apply", "--check", path.join(repo, patch)], plan.sourceRoot, env);
    runLogged(path.join(input.output, "patch-" + path.basename(patch) + ".log"), plan.git, ["apply", path.join(repo, patch)], plan.sourceRoot, env);
  }
  const sourceState = sourceInputs(plan.sourceRoot, ["amd/comgr"], []);
  if (!fs.existsSync(path.join(plan.snapshot, "LICENSE.txt")) || !fs.existsSync(path.join(plan.snapshot, "NOTICES.txt"))) {
    throw new Error("Pinned COMGR source omitted its license or notices");
  }
  fs.writeFileSync(plan.overlay, plan.overlayText, {flag: "wx"});
  fs.writeFileSync(plan.ruleOverride, plan.ruleOverrideText, {flag: "wx"});
  fs.writeFileSync(plan.toolchainFile, plan.toolchain, {flag: "wx"});
  fs.writeFileSync(path.join(plan.hostProject, "CMakeLists.txt"), plan.hostProjectText, {flag: "wx"});
  fs.mkdirSync(path.join(plan.build, ".cmake/api/v1/query"), {recursive: true});
  const codemodelQuery = path.join(plan.build, ".cmake/api/v1/query/codemodel-v2");
  fs.writeFileSync(codemodelQuery, "", {flag: "wx"});
  Object.assign(fingerprints, buildFileInputs([plan.archive, plan.overlay, plan.ruleOverride, plan.toolchainFile,
    path.join(plan.hostProject, "CMakeLists.txt"), codemodelQuery]));
  const savedPlan = {source: input.source, descriptorFile: input.descriptorFile, targetLlvm: input.targetLlvm,
    nativeLlvm: input.nativeLlvm, hostCompiler: input.hostCompiler, deviceReceipt: input.deviceReceipt, deviceInstall: input.deviceInstall,
    output: input.output, jobs: input.jobs, tools: plan.tools};
  fs.writeFileSync(path.join(input.output, "plan.json"), JSON.stringify({version: 1, role: plan.role, revision, patches,
    savedPlan, configure: plan.configure, hostConfigure: plan.hostConfigure, hostBuild: plan.hostBuildCommand,
    archiveSha256: archiveHash, sourceState, device: {receipt: device.receiptFile, sha256: device.sha256, device: device.device},
    windowsLlvm: {generationId: llvm.admission["generationId"], receipt: llvm.receiptFile, receiptSha256: fileHash(llvm.receiptFile), admission: llvm.admissionFile,
      admissionSha256: fileHash(llvm.admissionFile), objectProof: llvm.objectsFile, objectProofSha256: fileHash(llvm.objectsFile)},
    compilerVersions: {windows: intel, host: nativeIntel, nativeLlvm: nativeVersions}, nativeClangIdentity,
    fingerprints, recipe}, null, 2) + "\n", {flag: "wx"});
  runLogged(path.join(input.output, "bc2h-configure.log"), plan.hostConfigure[0] || "", plan.hostConfigure.slice(1), repo, env);
  const hostCompilerIdentity = verifyNativeIntelCompilerIdentity(plan.hostBuild, input.hostCompiler);
  runLogged(path.join(input.output, "bc2h-build.log"), plan.hostBuildCommand[0] || "", plan.hostBuildCommand.slice(1), repo, env);
  const helperStat = fs.statSync(plan.hostHelper, {throwIfNoEntry: false});
  if (!helperStat?.isFile() || (fs.readFileSync(plan.hostHelper).subarray(0, 4).toString("hex") !== "7f454c46")) {
    throw new Error("Upstream bc2h generator did not produce a native Linux executable");
  }
  const nativeCompile = auditBc2hCompileCommands(path.join(plan.hostBuild, "compile_commands.json"), input.hostCompiler, plan.hostBuild,
    path.join(plan.hostBuild, "bc2h.c"), path.join(plan.hostBuild, "CMakeFiles/bc2h.dir/bc2h.c.o"));
  const nativeFiles = buildFileInputs([path.join(plan.hostProject, "CMakeLists.txt"), hostCompilerIdentity.file, path.join(plan.hostBuild, "CMakeCache.txt"),
    path.join(plan.hostBuild, "build.ninja"), path.join(plan.hostBuild, "compile_commands.json"), path.join(plan.hostBuild, "bc2h.c"),
    nativeCompile.output, plan.hostHelper]);
  runLogged(path.join(input.output, "configure.log"), plan.configure[0] || "", plan.configure.slice(1), repo, env);
  const compilerIdentity = verifyIntelCompilerIdentity(plan.build), cache = verifyComgrCache(path.join(plan.build, "CMakeCache.txt"), plan);
  const compileCommands = auditComgrCompileCommands(path.join(plan.build, "compile_commands.json"), plan.compilerTools.CMAKE_CXX_COMPILER,
    plan.snapshot, plan.build, false);
  const target = comgrArchive(plan.build, plan.snapshot), header = path.join(plan.build, "include/amd_comgr.h");
  const exports = ["amd_comgr-config.cmake", "amd_comgr-targets.cmake", "amd_comgr-config-version.cmake"].map(name =>
    path.join(plan.build, "lib/cmake/amd_comgr", name));
  const configuredFiles = buildFileInputs([cache, path.join(plan.build, "build.ninja"), compileCommands.path,
    ...Object.values(compilerIdentity).map(value => value.file), header, ...exports]);
  if (!fs.statSync(header, {throwIfNoEntry: false})?.isFile() || exports.some(file => !fs.statSync(file, {throwIfNoEntry: false})?.isFile())) {
    throw new Error("Original COMGR header or build-tree CMake exports are missing");
  }
  if (Object.entries(fingerprints).some(([file, hash]) => fileHash(file) !== hash) ||
      !isDeepStrictEqual(sourceState, sourceInputs(plan.sourceRoot, ["amd/comgr"], [])) ||
      !isDeepStrictEqual(device.sha256, verifyWindowsDeviceResources(input.deviceReceipt, input.deviceInstall).sha256)) {
    throw new Error("Windows COMGR input changed during configuration");
  }
  const result = {version: 1, role: plan.role, stage: "configured", plan: savedPlan, planSha256: fileHash(path.join(input.output, "plan.json")),
    revision, sourceState, archive: plan.archive, archiveSha256: archiveHash, patches, recipe, fingerprints, nativeFiles,
    device: {receipt: device.receiptFile, sha256: device.sha256, revision: device.revision, clrRevision: device.clrRevision,
      files: device.files, device: device.device},
    windowsLlvm: {generationId: llvm.admission["generationId"], receipt: llvm.receiptFile, receiptSha256: fileHash(llvm.receiptFile), admission: llvm.admissionFile,
      admissionSha256: fileHash(llvm.admissionFile), objectProof: llvm.objectsFile, objectProofSha256: fileHash(llvm.objectsFile)},
    compilerIdentity, compilerVersions: {windows: intel, host: nativeIntel, nativeLlvm: nativeVersions}, nativeClangIdentity, configure: plan.configure,
    hostConfigure: plan.hostConfigure, hostBuild: plan.hostBuildCommand, hostCompilerIdentity, nativeCompile, compileCommands, configuredFiles,
    configuredOutputs: jitBuildOutputs(plan.build),
    archiveTarget: target.target.name, archivePath: target.archive, header, exports, buildOnly: true, compiled: false, executed: false,
    publicationQualified: false, scope: "Original COMGR static Windows COFF library and exports only; no archive-copy SDK prefix, PE tools, Windows execution, GPU execution or publication qualification."};
  writeReceipt(path.join(input.output, "result.json"), result);
  return result;
}

/** @param {string} output */
function verifySession(output) {
  const receiptFile = path.join(output, "result.json"), receipt = readBuildJson(receiptFile), saved = receipt.plan;
  if (!saved || receipt.version !== 1 || receipt.role !== "windows-comgr-static-host" || receipt.stage !== "configured" ||
      receipt.buildOnly !== true || receipt.compiled !== false || receipt.executed !== false || receipt.publicationQualified !== false ||
      saved.output !== output || path.resolve(path.dirname(saved.output), path.basename(saved.output)) !== output ||
      fileHash(path.join(output, "plan.json")) !== receipt.planSha256) {
    throw new Error("Windows COMGR same-run configure receipt is missing or changed");
  }
  const descriptor = validateWindowsLlvmSdkConfig(readBuildJson(saved.descriptorFile));
  const llvm = verifyWindowsLlvmAdmission(saved.targetLlvm), llvmPlan = llvm.receipt.plan,
    plan = windowsComgrSdkPlan({...saved, descriptor}, {source: llvmPlan.source,
      build: path.join(llvmPlan.output, "build")});
  const device = verifyWindowsDeviceResources(saved.deviceReceipt, saved.deviceInstall);
  if (device.sha256 !== receipt.device?.sha256 || llvm.receiptFile !== receipt.windowsLlvm?.receipt ||
      llvm.admission["generationId"] !== receipt.windowsLlvm?.generationId ||
      fileHash(llvm.receiptFile) !== receipt.windowsLlvm?.receiptSha256 || llvm.admissionFile !== receipt.windowsLlvm?.admission ||
      fileHash(llvm.admissionFile) !== receipt.windowsLlvm?.admissionSha256 || llvm.objectsFile !== receipt.windowsLlvm?.objectProof ||
      fileHash(llvm.objectsFile) !== receipt.windowsLlvm?.objectProofSha256 ||
      !isDeepStrictEqual(receipt.recipe, sourceInputs(repo, [], [...recipeFiles])) ||
      fileHash(receipt.archive) !== receipt.archiveSha256 ||
      !isDeepStrictEqual(receipt.sourceState, sourceInputs(plan.sourceRoot, ["amd/comgr"], [])) ||
      !isDeepStrictEqual(receipt.compileCommands, auditComgrCompileCommands(path.join(plan.build, "compile_commands.json"),
        plan.compilerTools.CMAKE_CXX_COMPILER, plan.snapshot, plan.build, false)) ||
      Object.entries(receipt.nativeFiles).some(([file, hash]) => fileHash(file) !== hash) ||
      Object.entries(receipt.fingerprints).some(([file, hash]) => fileHash(file) !== hash) ||
      Object.entries(receipt.configuredFiles).some(([file, hash]) => fileHash(file) !== hash) ||
      !isDeepStrictEqual(jitBuildOutputs(plan.build), receipt.configuredOutputs)) {
    throw new Error("Windows COMGR source, recipe, SDK or configured build changed; use a new output directory");
  }
  return {receipt, plan, llvm, device};
}

/** Build only the original amd_comgr target; configured inputs and the same-run native bc2h
 * helper must still match. No install/copy stage is performed.
 * @param {string} output @param {NodeJS.ProcessEnv} [environment] */
function buildWindowsComgrSdk(output, environment = process.env) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("Windows COMGR build requires Linuxx64 and MOM_AOT_BUILD_ONLY=1");
  }
  const {receipt, plan, llvm, device} = verifySession(output);
  const env = cleanEnvironment(environment, plan, path.dirname(plan.hostHelper));
  runLogged(path.join(output, "build.log"), plan.tools.cmake,
    ["--build", plan.build, "--parallel", String(receipt.plan.jobs), "--target", "amd_comgr", "--", "-d", "keepdepfile"], repo, env);
  const compileCommands = auditComgrCompileCommands(path.join(plan.build, "compile_commands.json"), plan.compilerTools.CMAKE_CXX_COMPILER,
      plan.snapshot, plan.build, true), target = comgrArchive(plan.build, plan.snapshot), stat = fs.statSync(target.archive);
  if (!isStaticArchive(target.archive) || stat.size < 8) {throw new Error("COMGR target did not produce a regular static COFF archive");}
  const archiveMembers = auditComgrArchiveMembers(query(plan.native["llvm-ar"], ["t", target.archive], "archive-members.log", output,
    cleanEnvironment(environment, plan, path.dirname(plan.hostHelper)), repo), compileCommands.objects, plan.build);
  const unexpected = Object.keys(jitBuildOutputs(plan.build)).filter(name => /\.(?:exe|dll)$/i.test(name));
  if (unexpected.length) {throw new Error("Windows COMGR build produced an executable or DLL: " + unexpected[0]);}
  if (Object.entries(receipt.fingerprints).some(([file, hash]) => fileHash(file) !== hash) ||
      Object.entries(receipt.configuredFiles).some(([file, hash]) => fileHash(file) !== hash) ||
      Object.entries(receipt.nativeFiles).some(([file, hash]) => fileHash(file) !== hash) ||
      fileHash(receipt.archive) !== receipt.archiveSha256 ||
      !isDeepStrictEqual(receipt.sourceState, sourceInputs(plan.sourceRoot, ["amd/comgr"], [])) ||
      fileHash(llvm.receiptFile) !== receipt.windowsLlvm.receiptSha256 || device.sha256 !== verifyWindowsDeviceResources(receipt.plan.deviceReceipt,
    receipt.plan.deviceInstall).sha256) {throw new Error("Windows COMGR input changed during archive build");}
  const exports = receipt.exports;
  const final = {...receipt, stage: "complete", compiled: true, compileCommands, archiveTarget: target.target.name,
    archivePath: target.archive, archiveSha256: fileHash(target.archive), archiveBytes: stat.size,
    archiveMembers, headerSha256: fileHash(receipt.header), exports: buildFileInputs(exports), objects: compileCommands.objects,
    buildOutputs: jitBuildOutputs(plan.build), buildOnly: true, executed: false, publicationQualified: false};
  writeReceipt(path.join(output, "result.json"), final, true);
  return final;
}

module.exports = {revision, patches, hostSources, recipeFiles, windowsComgrSdkPlan, verifyWindowsDeviceResources, verifyIntelCompilerIdentity,
  verifyNativeIntelCompilerIdentity, verifyNativeClangIdentity, canonicalNativeClang, verifyNativeLibrarianIdentity, auditBc2hCompileCommands, verifyComgrCache, auditComgrCompileCommands, comgrArchive,
  auditComgrArchiveMembers, nativeRuntimeInputs, cleanEnvironment, configureWindowsComgrSdk, buildWindowsComgrSdk};

if (require.main === module) {
  try {
    const [phase, ...args] = process.argv.slice(2);
    if (phase === "configure" && args.length === 8) {
      const [source, descriptorFile, targetLlvm, nativeLlvm, hostCompiler, deviceReceipt, output, jobs] = /** @type {[string, string, string, string, string, string, string, string]} */ (args);
      if ([source, descriptorFile, targetLlvm, nativeLlvm, hostCompiler, deviceReceipt, output, jobs].some(value => !value)) {
        throw new Error("Incomplete Windows COMGR configure arguments");
      }
      configureWindowsComgrSdk({source, descriptorFile, targetLlvm, nativeLlvm, hostCompiler, deviceReceipt,
        deviceInstall: path.join(path.dirname(deviceReceipt), "install"), output, jobs: Number(jobs)});
    } else if (phase === "build" && args.length === 1 && args[0]) {buildWindowsComgrSdk(args[0]);} else {throw new Error("Usage: build-native-windows-comgr-sdk.js configure /COMGR-Git /windows-descriptor.json /windows-LLVM-output /native-LLVM-root /Intel-icx /Windows-device-result.json /new-output /jobs | build /configured-output");}
  } catch (error) {console.error(error instanceof Error ? error.message : "Windows COMGR build failed"); process.exitCode = 1;}
}
