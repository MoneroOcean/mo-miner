"use strict";

// Original static Windows SYCL/UR SDK production. The controller's explicit Windows build
// descriptor supplies the same tools and OS headers; no audit-cache paths or installed DLLs.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"), {isDeepStrictEqual} = require("node:util");
const {validateWindowsControllerConfig, windowsToolEnvironment, vfsFiles, parseImports} = require("./build-native-windows-controller");
const {verifyPreparedSdkSources, sdkSourceFiles, sdkSourceEnvironment} = require("./prepare-native-sdk-sources");
const {fileHash, sourceInputs, verifySourceInputs, readBuildJson, readBuildJsonSnapshot, nativeBuildJobs} = require("./build-source-inputs");
const {pins} = require("./native-dpcpp-pins"), {isStaticArchive} = require("./native-audit");
const {dpcppEntries} = require("./native-compiler-isolation");
const {windowsIntelRuleOverride} = require("./prepare-native-jit-toolchain");
const repo = path.resolve(__dirname, "..");

/** Literal cross-toolchain paths must not become CMake lists, interpolation or extra flags.
 * @param {unknown} value @returns {value is string} */
const literal = value => typeof value === "string" && path.isAbsolute(value) && !/[;"$\\]/.test(value) && !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
/** @param {string} value */
const cmake = value => {if (value.includes("]==]")) {throw new Error("Invalid CMake literal");} return "[==[" + value + "]==]";};

/** Admit the complete existing COFF compiler-isolation result, not a caller-curated subset.
 * This proves the private ABI and unchanged compiled inputs, not original SDK production.
 * Every archive stays explicit so the application cannot accidentally omit a compiler leaf.
 * @param {string | undefined} receiptFile */
function windowsSourceJitInputs(receiptFile) {
  if (receiptFile === undefined) {return null;}
  if (!literal(receiptFile)) {throw new Error("Select an absolute isolated Windows source-JIT receipt");}
  const {value: selected, sha256: receiptHash} = readBuildJsonSnapshot(receiptFile), files = /** @type {Record<string, string>} */ ({[receiptFile]: receiptHash});
  if (fileHash(receiptFile) !== receiptHash) {throw new Error("Windows source-JIT receipt changed during admission");}
  if (selected?.version !== 1 || selected.prefix !== "mom_dpcpp23_" || selected.buildOnly !== true || selected.executed !== false || selected.publicationQualified !== false ||
      !isDeepStrictEqual(selected.entryPoints, dpcppEntries) || !Array.isArray(selected.transformed) || selected.transformed.length < 50 || selected.transformed.length > 512 ||
      ["memberOrderPreserved", "nonDirectivePayloadsPreserved", "mappedDirectivesPreserved", "symbolOrderPreserved", "auxiliaryRecordsPreserved", "relocationsPreserved"].some(key => selected.checks?.[key] !== true)) {
    throw new Error("Incomplete isolated Windows source-JIT compiler closure");
  }
  /** @param {unknown} file @param {unknown} hash */
  const admit = (file, hash) => {
    if (!literal(file) || typeof hash !== "string" || !/^[0-9a-f]{64}$/.test(hash) || fileHash(file) !== hash) {throw new Error("Windows source-JIT input changed: " + file);}
    files[file] = hash;
  };
  const sources = ["scripts/native-coff-archive.js", "scripts/native-compiler-isolation.js", "scripts/build-source-inputs.js"].map(name => path.join(repo, name));
  if (!selected.sourceHashes || !isDeepStrictEqual(Object.keys(selected.sourceHashes).sort(), sources.sort()) || !selected.toolHashes || Object.keys(selected.toolHashes).length !== 3) {throw new Error("Incomplete Windows source-JIT isolation recipe");}
  for (const [file, hash] of [...Object.entries(selected.sourceHashes), ...Object.entries(selected.toolHashes)]) {admit(file, hash);}
  admit(selected.map, selected.mapSha256);
  const libraries = [], names = new Set();
  for (const row of selected.transformed) {
    admit(row?.input, row?.inputSha256); admit(row?.output, row?.sha256);
    const name = path.basename(row.output);
    if (!/^[A-Za-z0-9_.-]+\.lib$/.test(name) || name !== path.basename(row.input) || names.has(name) || !isStaticArchive(row.input) || !isStaticArchive(row.output)) {throw new Error("Invalid Windows source-JIT archive selection");}
    names.add(name); libraries.push(row.output);
  }
  if (path.basename(libraries[0] || "") !== "sycl-jit.lib") {throw new Error("Windows source-JIT closure must start with its original entry archive");}
  return {receiptFile, archive: libraries[0] || "", libraries, files};
}

/** One deterministic original-source recipe, without executing CMake or inspecting a GPU.
 * Linux is the cross-build host; the resulting SDK and binding probe are Windows COFF/PE.
 * @param {Record<string, string>} roots @param {unknown} descriptor @param {string} sdk
 * @param {string} cuda @param {string} output @param {number} jobs @param {string} [jitArchive] @param {boolean} [intelHost] */
function windowsStaticSdkPlan(roots, descriptor, sdk, cuda, output, jobs, jitArchive, intelHost = false) {
  const config = validateWindowsControllerConfig(descriptor), {tools, windowsSdk} = config;
  if (![sdk, cuda, output, ...Object.values(roots), tools.clangCl, tools.lldLink, tools.clangResourceInclude,
    windowsSdk.overlay, windowsSdk.sourceRoot, ...windowsSdk.includeRoots, ...windowsSdk.libraryRoots].every(literal) ||
      !Number.isInteger(jobs) || jobs < 1 || jobs > 16 || typeof intelHost !== "boolean") {throw new Error("Windows SDK build needs literal absolute roots and jobs in 1..16");}
  if (jitArchive !== undefined && (!literal(jitArchive) || !/\.lib$/.test(jitArchive))) {throw new Error("Windows source JIT needs an explicit isolated COFF archive");}
  // Intel's compiler need not share a directory with the explicitly selected LLVM utilities.
  const bin = path.dirname(tools.llvmReadobj), build = path.join(output, "build"), aliases = path.join(output, "sdk-aliases");
  const compilerTools = {CMAKE_C_COMPILER: tools.clangCl, CMAKE_CXX_COMPILER: tools.clangCl, CMAKE_LINKER: tools.lldLink,
    CMAKE_RC_COMPILER: path.join(bin, "llvm-rc"), ...(!intelHost ? {CMAKE_MT: path.join(bin, "llvm-mt")} : {}),
    CMAKE_AR: path.join(bin, "llvm-lib"), CMAKE_RANLIB: path.join(bin, "llvm-ranlib")};
  const links = "/machine:x64 /INCREMENTAL:NO " + [aliases, ...windowsSdk.libraryRoots].map(root => '/libpath:"' + root + '"').join(" ");
  const ruleOverride = path.join(output, "intel-link-rules.cmake"), ruleOverrideText = windowsIntelRuleOverride(links,
    [...windowsSdk.msvcLibraries, ...windowsSdk.ucrtLibraries, ...windowsSdk.importLibraries]);
  const flags = ["-nostdinc", "-I" + tools.clangResourceInclude, "-I" + path.join(cuda, "include"), "--target=x86_64-pc-windows-msvc", "/MT", "/O2", "/Gy", "/Gw",
    "-Xclang", "-ivfsoverlay", "-Xclang", windowsSdk.overlay, ...windowsSdk.includeRoots.map(root => "-I" + root)].map(flag => '"' + flag + '"').join(" ");
  const toolchain = ["set(CMAKE_SYSTEM_NAME Windows)", "set(CMAKE_SYSTEM_PROCESSOR AMD64)", "set(CMAKE_SYSTEM_VERSION 10.0)",
    "set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)", "set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)",
    ...Object.entries(compilerTools).map(([name, file]) => `set(${name} ${cmake(file)})`),
    ...["C", "CXX"].map(language => `set(CMAKE_USER_MAKE_RULES_OVERRIDE_${language} ${cmake(ruleOverride)})`),
    `set(CMAKE_C_FLAGS_INIT ${cmake(flags)})`, `set(CMAKE_CXX_FLAGS_INIT ${cmake(flags + " /EHsc /std:c++17")})`,
    `set(CMAKE_FIND_ROOT_PATH ${cmake(windowsSdk.sourceRoot)})`, ...["LIBRARY", "INCLUDE", "PACKAGE"].map(name => `set(CMAKE_FIND_ROOT_PATH_MODE_${name} ONLY)`),
    "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)", ""].join("\n");
  const definitions = {CMAKE_TOOLCHAIN_FILE: path.join(output, "toolchain.cmake"), CMAKE_BUILD_TYPE: "Release", CMAKE_EXPORT_COMPILE_COMMANDS: "ON",
    CMAKE_EXE_LINKER_FLAGS: links,
    MOM_DPCPP_SOURCE: roots["dpcpp"], MOM_DPCPP_EMHASH: roots["emhash"], MOM_DPCPP_UMF: roots["umf"], MOM_SOURCE_REPO: repo, MOM_DPCPP_SDK: sdk,
    MOM_DPCPP_LEVEL_ZERO_SOURCE: roots["levelZero"], MOM_DPCPP_COMPUTE_HEADERS: roots["computeHeaders"], MOM_DPCPP_CUDA_SDK: cuda, MOM_DPCPP_HWLOC: roots["hwloc"],
    MOM_DPCPP_OPENCL_SOURCE: roots["openclLoader"], MOM_DPCPP_OPENCL_HEADERS: roots["openclHeaders"],
    CMAKE_HAVE_LIBC_PTHREAD: "FALSE", CMAKE_HAVE_PTHREADS_CREATE: "FALSE", CMAKE_HAVE_PTHREAD_CREATE: "FALSE", THREADS_HAVE_PTHREAD_ARG: "FALSE"};
  if (jitArchive) {Object.assign(definitions, {MOM_DPCPP_JIT_ARCHIVE: jitArchive});}
  const targets = ["mom-static-dpcpp-libs", "ur_adapter_cuda", "ur_adapter_level_zero", "ur_adapter_level_zero_v2", "ur_adapter_opencl", "ze_loader", "umf", "umf_ba", "umf_coarse", "umf_utils", "hwloc"];
  const artifacts = ["mom-sycl-static.lib", ...["ur_loader", "ur_common", "ur_adapter_cuda", "ur_adapter_level_zero", "ur_adapter_level_zero_v2", "ur_adapter_opencl", "ze_loader", "level_zero_utils", "umf", "umf_ba", "umf_coarse", "umf_utils", "hwloc"].map(name => "lib/" + name + ".lib"), "opencl-icd-loader/OpenCL.lib"];
  return {config, compilerTools, toolchain, ruleOverride, ruleOverrideText, build, aliases, definitions, targets, artifacts, configure: ["-S", path.join(repo, "tests/native/dpcpp-static"), "-B", build, "-G", "Ninja", ...Object.entries(definitions).map(([key, value]) => "-D" + key + "=" + value)],
    compile: ["--build", build, "--parallel", String(jobs), "--target", ...targets, "--", "-k", "0"]};
}

/** Build raw original archives with source/SDK/recipe checks on both sides. COFF factory
 * namespace isolation and the LLVM/COMGR JIT closure remain separate recorded inputs.
 * No compiler installation, VM startup, GPU discovery or execution is performed.
 * @param {string} sourcesFile @param {string} descriptorFile @param {string} sdk
 * @param {string} cuda @param {string} output @param {NodeJS.ProcessEnv} [environment] */
function buildWindowsStaticSdk(sourcesFile, descriptorFile, sdk, cuda, output, environment = process.env) {
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows SDK cross-build requires Linux and MOM_AOT_BUILD_ONLY=1");}
  const selected = verifyPreparedSdkSources(sourcesFile, "windows"), roots = Object.fromEntries(Object.entries(selected.prepared).map(([name, value]) => [name, value.root]));
  const sourceJit = windowsSourceJitInputs(environment["MOM_AOT_DPCPP_JIT_RECEIPT"]);
  const descriptor = validateWindowsControllerConfig(readBuildJson(descriptorFile)), env = sdkSourceEnvironment(windowsToolEnvironment(environment));
  const hostCompilerSha256 = fileHash(descriptor.tools.clangCl);
  const hostVersion = spawnSync(descriptor.tools.clangCl, ["--version"], {env, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
  if (hostVersion.error || hostVersion.signal || hostVersion.status !== 0 || fileHash(descriptor.tools.clangCl) !== hostCompilerSha256) {
    throw new Error("Windows SDK host compiler identity query failed or changed", {cause: hostVersion.error});
  }
  // Intel's direct-LLD rules never invoke the manifest wrapper. Do not require an unused
  // llvm-mt binary in the thin utilities cohort; preserve Clang's original platform rules.
  const intelHost = /Intel\(R\) oneAPI DPC\+\+\/C\+\+ Compiler/.test(hostVersion.stdout + hostVersion.stderr);
  const plan = windowsStaticSdkPlan(roots, descriptor, sdk, cuda, output, nativeBuildJobs(environment), sourceJit?.archive, intelHost);
  const {config} = plan;
  if (fs.existsSync(output) || !fs.statSync(path.dirname(output), {throwIfNoEntry: false})?.isDirectory()) {throw new Error("Select a new Windows SDK build output");}
  const target = path.join(fs.realpathSync(path.dirname(output)), path.basename(output));
  for (const input of [...Object.values(roots), sdk, cuda, config.windowsSdk.sourceRoot, path.dirname(sourcesFile)]) {
    const real = fs.realpathSync(input), relative = path.relative(real, target);
    if (!relative || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep))) {throw new Error("Windows SDK output overlaps a selected source or SDK");}
  }
  const recipeLeaves = ["scripts/build-native-windows-sdk.js", "scripts/prepare-native-jit-toolchain.js", "scripts/prepare-native-sdk-sources.js", "scripts/build-native-windows-controller.js", "scripts/build-environment.js", "scripts/build-source-inputs.js", "scripts/native-audit.js", "scripts/native-dpcpp-pins.js",
    "scripts/native-coff-archive.js", "scripts/native-compiler-isolation.js", "tests/native/dpcpp-static/CMakeLists.txt", "tests/native/dpcpp-static/static-windows.h", "tests/reference/native-opencl-binding.cpp", "runtime/windows-opencl-driver.h"];
  const recipe = sourceInputs(repo, [], recipeLeaves), virtual = vfsFiles(config.windowsSdk.overlay);
  const files = [sourcesFile, descriptorFile, ...Object.values(plan.compilerTools), config.tools.llvmReadobj, config.windowsSdk.overlay,
    path.join(sdk, "bin/clang++"), ...virtual.values(), ...config.windowsSdk.msvcLibraries, ...config.windowsSdk.ucrtLibraries, ...config.windowsSdk.importLibraries];
  const headerRoots = [path.join(sdk, "include"), path.join(cuda, "include"), config.tools.clangResourceInclude];
  const headers = headerRoots.map(root => ({root, files: sdkSourceFiles(root)}));
  const fingerprints = {...Object.fromEntries([...new Set(files)].map(file => [file, fileHash(file)])), ...sourceJit?.files};
  if (fingerprints[descriptor.tools.clangCl] !== hostCompilerSha256) {throw new Error("Windows SDK host compiler changed after its identity query");}
  fs.mkdirSync(output); fs.mkdirSync(plan.aliases);
  /** Logs stream to disk; ordinary Ninja errors do not stop independent jobs. A failed stage
   * never issues a compiled receipt or triggers an automatic execution/retry.
   * @param {string} command @param {string[]} args @param {string} name */
  function run(command, args, name) {
    const fd = fs.openSync(path.join(output, name + ".log"), "wx");
    try {
      const result = spawnSync(command, args, {cwd: repo, env, stdio: ["ignore", fd, fd], timeout: 1800000});
      if (result.error || result.signal || result.status !== 0) {throw new Error("Windows SDK stage failed; inspect " + name + ".log", {cause: result.error});}
    } finally {fs.closeSync(fd);}
  }
  // Match the Windows OS SDK's case-insensitive import-library names on a Linux host.
  const imports = ["kernel32", "user32", "gdi32", "winspool", "shell32", "ole32", "oleaut32", "uuid", "comdlg32", "advapi32", "dbghelp", "cfgmgr32", "runtimeobject"].map(name => name + ".lib");
  for (const name of imports) {
    const file = config.windowsSdk.libraryRoots.flatMap(root => fs.readdirSync(root).filter(entry => entry.toLowerCase() === name).map(entry => path.join(root, entry)))[0];
    if (!file) {throw new Error("Missing Windows OS SDK import: " + name);}
    fingerprints[file] = fileHash(file); fs.symlinkSync(file, path.join(plan.aliases, name));
  }
  const probe = spawnSync(path.join(sdk, "bin/clang++"), ["--version"], {env, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
  // Match the existing Windows producer's source ABI and fingerprint all selected tool/header
  // bytes. A local source-built SDK need not impersonate a downloaded archive receipt.
  if (probe.error || probe.signal || probe.status !== 0 || !probe.stdout.includes(`clang version 23.0.0git (https://github.com/intel/llvm ${pins.source})`)) {throw new Error("DPC++ compiler identity query failed");}
  fs.writeFileSync(path.join(output, "toolchain.cmake"), plan.toolchain, {flag: "wx"});
  fs.writeFileSync(plan.ruleOverride, plan.ruleOverrideText, {flag: "wx"});
  for (const file of [path.join(output, "toolchain.cmake"), plan.ruleOverride]) {fingerprints[file] = fileHash(file);}
  fs.writeFileSync(path.join(output, "before.json"), JSON.stringify({sourcesFile, recipe, fingerprints, headers, configure: plan.configure, compile: plan.compile}, null, 2) + "\n", {flag: "wx"});
  run("cmake", plan.configure, "configure"); run("cmake", plan.compile, "build");
  verifyPreparedSdkSources(sourcesFile, "windows");
  verifySourceInputs(recipe, sourceInputs(repo, [], recipeLeaves), "Windows SDK build recipe");
  for (const [file, hash] of Object.entries(fingerprints)) {if (fileHash(file) !== hash) {throw new Error("Windows SDK input changed: " + file);}}
  for (const header of headers) {if (!isDeepStrictEqual(header.files, sdkSourceFiles(header.root))) {throw new Error("Windows SDK headers changed: " + header.root);}}
  const artifacts = plan.artifacts.map(name => {const file = path.join(plan.build, name); if (!isStaticArchive(file)) {throw new Error("Windows SDK output is not an archive: " + name);} return {path: file, sha256: fileHash(file), bytes: fs.statSync(file).size};});
  const binding = path.join(plan.build, "native-opencl-binding-static.exe"), inspection = spawnSync(config.tools.llvmReadobj, ["--coff-imports", binding], {env, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
  if (inspection.error || inspection.signal || inspection.status !== 0) {throw new Error("Windows SDK binding probe import audit failed");}
  const importsFound = parseImports(inspection.stdout);
  if (importsFound.some(name => !config.windowsSdk.allowedImports.map(value => value.toLowerCase()).includes(name))) {throw new Error("Windows SDK binding probe imports an unapproved DLL");}
  const result = {version: 1, sourcesFile, fingerprints, headers, recipe, hostCompilerVersion: hostVersion.stdout + hostVersion.stderr,
    artifacts, binding: {path: binding, sha256: fileHash(binding), imports: importsFound},
    sourceJit: sourceJit ? {enabled: true, receiptFile: sourceJit.receiptFile, archive: sourceJit.archive, libraries: sourceJit.libraries} : {enabled: false},
    buildOnly: true, executed: false, publicationQualified: false, scope: "Original static Windows SYCL/UR, all adapters and memory/driver loaders. Raw SDK archives only; compiler/factory isolation, JIT closure, runtime and release gates are separate."};
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  console.log("[native-sdk] Built original Windows static archives; not executed or publication-qualified.");
  return result;
}

if (require.main === module) {
  try {
    const [sources, config, sdk, cuda, output, extra] = process.argv.slice(2);
    if (!sources || !config || !sdk || !cuda || !output || extra) {throw new Error("Usage: build-native-windows-sdk.js /sources.json /windows-sdk.json /DPC++ /CUDA /new-output");}
    buildWindowsStaticSdk(sources, config, sdk, cuda, output);
  } catch (error) {console.error(error instanceof Error ? error.message : "Windows SDK build failed"); process.exitCode = 1;}
}
module.exports = {windowsSourceJitInputs, windowsStaticSdkPlan, buildWindowsStaticSdk};
