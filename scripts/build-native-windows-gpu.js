"use strict";

// Compile the original GPU sources for a Windows static application. Each combined object
// supplies both its device image and its retained host half; never recompile that half with
// a different compiler-generated specialization prefix. This builder does not execute GPUs.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {fileHash, nativeSourceInputs, verifySourceInputs, buildFileInputs, readBuildJson, buildDependencies} = require("./build-source-inputs");
const {syclSources, gpuProfile, dpcppGpuFlags, dpcppUniquePrefixFlags, dpcppSourceFlags, portableOpenclTranslatorFlags, nativeIncludeRoots, writeAdaptiveCppUmbrella} = require("./build-native-gpu-probe");
const {admitSyclSource} = require("./probe-static-sycl");
const {windowsDeviceLinkArgs, transformWindowsWrapper, extractSpecializationIds, verifyWindowsHostRetention} = require("./native-windows-wrapper");
const {dependencyHashes, absolutePath, vfsFiles} = require("./build-native-windows-controller");
const {gpuTable} = require("./native-gpu-table"), {isolateCoffObjects} = require("./native-compiler-isolation");
const {pins} = require("./probe-static-dpcpp"), nativeResources = require("./native-resources");
const {releaseBuildEnvironment} = require("./build-environment");
const {validateDirectory} = require("./validate-portable-opencl");
const {cudaHeaderInputs} = require("./native-jit-headers");
const {compileNativeObjects, environmentValue} = require("./native-object-cache");
const repo = path.resolve(__dirname, ".."), target = "--target=x86_64-pc-windows-msvc";
const toolKeys = ["clang", "clangC", "llvmNm", "llvmReadobj", "llvmObjcopy"];
const intelCompileHelpers = ["clang", "clang-offload-bundler", "sycl-post-link", "llvm-spirv", "clang-offload-wrapper",
  "clang-linker-wrapper", "llvm-link", "llvm-foreach", "llc"];

/** CI transport is not a compiler input. Remove it from the actual child environment too,
 * so moving a receipt/output directory cannot invalidate every unchanged GPU object.
 * @param {NodeJS.ProcessEnv} environment */
function windowsGpuBuildEnvironment(environment) {
  const env = releaseBuildEnvironment(environment);
  for (const key of Object.keys(env)) {
    if (/^(?:GITHUB_|RUNNER_)/i.test(key) || /^MOM_NATIVE_WINDOWS_GPU_DIR$/i.test(key)) {delete env[key];}
  }
  return env;
}

/** Honor the caller's memory-budgeted parallelism; default to the previous single-job limit.
 * @param {NodeJS.ProcessEnv} environment @param {NodeJS.Platform} [platform] */
function windowsGpuBuildJobs(environment, platform = process.platform) {
  const value = environmentValue(/** @type {Record<string, string>} */ (environment), "MOM_BUILD_JOBS", platform);
  if (value === undefined || value === "") {return 1;}
  if (!/^[1-9]\d{0,2}$/.test(value) || Number(value) > 256) {throw new Error("MOM_BUILD_JOBS must be an integer in 1..256");}
  return Number(value);
}
/** @typedef {{version: 1, profile: string, tools: {clang: string, clangC: string, llvmNm: string, llvmReadobj: string, llvmObjcopy: string}, includeRoots: string[], overlay: string, compilerRoot?: string, cuda?: {sdk: string, nvrtcInclude?: string, headers?: string, images?: string}, acpp?: {source: string, generatedInclude: string, plugin: string, git?: string}}} Config */
/** @typedef {{status: number | null, error?: Error | undefined, signal: NodeJS.Signals | null, stdout: string}} QueryResult */
/** @typedef {(command: string, args: string[], options: import("node:child_process").SpawnSyncOptionsWithStringEncoding) => QueryResult} CompilerQuery */

/** Reject unsupported policy before creating output or looking up tools. SDK paths are
 * explicit trusted build inputs, not an arbitrary list of compiler flags. @param {unknown} value */
function validateWindowsGpuConfig(value) {
  const config = /** @type {Config} */ (value);
  if (!config || typeof config !== "object" || config.version !== 1 ||
      Object.keys(config).some(key => !["version", "profile", "tools", "includeRoots", "overlay", "compilerRoot", "cuda", "acpp"].includes(key)) ||
      !["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"].includes(config.profile)) {throw new Error("Invalid Windows GPU descriptor/profile");}
  if (!config.tools || Object.keys(config.tools).sort().join(",") !== toolKeys.slice().sort().join(",") ||
      Object.values(config.tools).some(file => !absolutePath(file))) {throw new Error("Windows GPU descriptor needs five absolute tool paths");}
  if (!absolutePath(config.overlay) || !Array.isArray(config.includeRoots) || !config.includeRoots.length || config.includeRoots.length > 32 ||
      config.includeRoots.some(file => !absolutePath(file)) || new Set(config.includeRoots).size !== config.includeRoots.length) {throw new Error("Invalid Windows GPU include roots/overlay");}
  if (config.compilerRoot !== undefined && (config.profile !== "oneapi" || !absolutePath(config.compilerRoot))) {
    throw new Error("Windows Intel compilerRoot is valid only as an absolute oneAPI compiler root");
  }
  if (config.profile === "dpcpp" || config.profile === "acpp-cuda") {
    if (!config.cuda || !["headers,nvrtcInclude,sdk", "images,sdk"].includes(Object.keys(config.cuda).sort().join(",")) || Object.values(config.cuda).some(file => !absolutePath(file))) {throw new Error("Windows DPC++ needs explicit CUDA SDK and image bundle or NVRTC headers");}
  } else if (config.cuda !== undefined) {throw new Error("CUDA inputs are not valid for this Windows GPU profile");}
  if (config.profile.startsWith("acpp-")) {
    if (!config.acpp || !["generatedInclude,plugin,source", "generatedInclude,git,plugin,source"].includes(Object.keys(config.acpp).sort().join(",")) ||
        Object.values(config.acpp).some(file => !absolutePath(file))) {throw new Error("Windows AdaptiveCpp needs explicit source, generatedInclude and plugin paths");}
  } else if (config.acpp !== undefined) {throw new Error("AdaptiveCpp inputs are not valid for this Windows GPU profile");}
  return config;
}

/** @param {string} root @param {string} file */
function contained(root, file) {
  const relative = path.relative(root, file);
  return relative !== "" && !path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep);
}

/** Return the canonical selected path after verifying its target stays under compilerRoot.
 * Keep a driver-reported link name intact: Clang-compatible tools may select behavior by argv[0].
 * @param {string} root @param {string} file @param {string} label @param {boolean} [executable] */
function intelRootFile(root, file, label, executable = true) {
  if (!absolutePath(file)) {throw new Error("Windows oneAPI " + label + " is not an absolute path");}
  const selected = path.resolve(file), stat = fs.statSync(selected, {throwIfNoEntry: false}), real = fs.realpathSync(selected);
  if (!stat?.isFile() || !contained(root, selected) || !contained(root, real)) {throw new Error("Windows oneAPI " + label + " is not a file under compilerRoot: " + file);}
  if (executable) {
    if (process.platform === "win32" ? path.extname(selected).toLowerCase() !== ".exe" : false) {
      throw new Error("Windows oneAPI " + label + " is not an executable .exe: " + file);
    }
    try {fs.accessSync(selected, fs.constants.X_OK);} catch {throw new Error("Windows oneAPI " + label + " is not executable: " + file);}
  }
  return selected;
}

/** @param {string} root @param {string} directory @param {string} label */
function intelRootDirectory(root, directory, label) {
  if (!absolutePath(directory)) {throw new Error("Windows oneAPI " + label + " is not an absolute path");}
  const stat = fs.statSync(directory, {throwIfNoEntry: false}), real = fs.realpathSync(directory);
  if (!stat?.isDirectory() || !contained(root, real)) {throw new Error("Windows oneAPI " + label + " is not a directory under compilerRoot: " + directory);}
  return real;
}

/** Intel's `icpx --version` uses the short `Compiler 2026.2.0` spelling while the shared
 * Windows toolchain verifier accepts Intel's `Compiler ... Version 2026.2.0` spelling.
 * Normalize only that pinned vendor prefix and reuse the shared 2026.2 identity gate.
 * @param {string} text @param {string} label */
function intel2026Identity(text, label) {
  const {verifyIntelCompilerIdentity} = require("./build-native-windows-hiprtc-sdk");
  const normalized = typeof text === "string" ? text.replace(
    /(^|\r?\n)(Intel\(R\) oneAPI DPC\+\+\/C\+\+ Compiler)\s+(?=2026\.2(?:\.\d+)?(?:\s|\(|$))/,
    "$1$2 for applications running on Intel(R) 64, Version ") : "";
  try {return verifyIntelCompilerIdentity(normalized);} catch (error) {throw new Error("Windows oneAPI " + label + " must be Intel 2026.2", {cause: error});}
}

/** Resolve and pin the closed Intel frontend's entire required helper set through public driver
 * queries. The descriptor must name the queried resource include first; flags are not rewritten.
 * @param {Config} config @param {NodeJS.ProcessEnv} environment @param {CompilerQuery} [spawn] */
function resolveOneapiCompiler(config, environment, spawn = spawnSync) {
  if (config.profile !== "oneapi" || !config.compilerRoot) {throw new Error("Select an explicit oneAPI compilerRoot");}
  const requestedRoot = path.resolve(config.compilerRoot), rootStat = fs.statSync(requestedRoot, {throwIfNoEntry: false}), root = fs.realpathSync(requestedRoot);
  if (!rootStat?.isDirectory() || root !== requestedRoot) {throw new Error("Windows oneAPI compilerRoot must be a canonical directory");}
  const extension = process.platform === "win32" ? ".exe" : "";
  const icpx = intelRootFile(root, path.join(root, "bin", "icpx" + extension), "icpx");
  const icx = intelRootFile(root, path.join(root, "bin", "icx" + extension), "icx");
  if (intelRootFile(root, config.tools.clang, "C++ compiler") !== icpx ||
      intelRootFile(root, config.tools.clangC, "C compiler") !== icx) {
    throw new Error("Windows oneAPI tools must select compilerRoot/bin/icpx and compilerRoot/bin/icx");
  }
  /** @param {string} command @param {string[]} args @param {string} label */
  const query = (command, args, label) => {
    const result = spawn(command, args, {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20, env: environment});
    if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string" || result.stdout.length > (1 << 20)) {
      throw new Error("Windows oneAPI " + label + " query failed", {cause: result.error});
    }
    return result.stdout.trim();
  };
  const cxxVersion = query(icpx, ["--version"], "icpx version"), cVersion = query(icx, ["--version"], "icx version");
  intel2026Identity(cxxVersion, "icpx"); intel2026Identity(cVersion, "icx");
  const resourceDir = intelRootDirectory(root, query(icpx, ["--print-resource-dir"], "resource directory"), "resource directory");
  const cResourceDir = intelRootDirectory(root, query(icx, ["--print-resource-dir"], "icx resource directory"), "icx resource directory");
  if (cResourceDir !== resourceDir) {throw new Error("Windows oneAPI icpx and icx resource directories differ");}
  const resourceInclude = intelRootDirectory(root, path.join(resourceDir, "include"), "resource include");
  if (config.includeRoots[0] !== resourceInclude) {throw new Error("Windows oneAPI includeRoots[0] must match the queried compiler resource include");}
  /** @type {Record<string, string>} */
  const helpers = {};
  for (const name of intelCompileHelpers) {
    const executable = name + extension, selected = query(icpx, ["-print-prog-name=" + executable], executable + " path");
    const cSelected = query(icx, ["-print-prog-name=" + executable], "icx " + executable + " path");
    const cppHelper = intelRootFile(root, selected, executable);
    if (path.basename(cppHelper).toLowerCase() !== executable.toLowerCase()) {throw new Error("Windows oneAPI resolved a different helper executable: " + executable);}
    if (intelRootFile(root, cSelected, "icx " + executable) !== cppHelper) {
      throw new Error("Windows oneAPI icpx and icx resolve different helper paths: " + executable);
    }
    helpers[name] = cppHelper;
  }
  /** @type {{clang: string, clangC: string, llvmNm: string, llvmReadobj: string, llvmObjcopy: string}} */
  const tools = {clang: icpx, clangC: icx, llvmNm: config.tools.llvmNm,
    llvmReadobj: config.tools.llvmReadobj, llvmObjcopy: config.tools.llvmObjcopy};
  const configurationFiles = [];
  for (const [label, version] of /** @type {Array<[string, string]>} */ ([
    ["icpx", cxxVersion], ["icx", cVersion],
  ])) {
    const paths = [...version.matchAll(/^\s*Configuration file:\s*(.*?)\s*$/gm)].map(match => match[1] || "");
    if (paths.length > 1 || paths.some(file => !file)) {throw new Error("Windows oneAPI " + label + " has ambiguous compiler configuration evidence");}
    if (paths[0]) {configurationFiles.push(intelRootFile(root, paths[0], label + " configuration", false));}
  }
  const sdkTools = intelCompileHelpers.map(name => helpers[name] || "");
  const identityHelper = path.join(__dirname, "build-native-windows-hiprtc-sdk.js");
  return {root, tools, sdkTools, helperDirectories: [...new Set(Object.values(helpers).map(file => path.dirname(file)))],
    resourceDir, resourceInclude, configurationFiles, cxxVersion, cVersion,
    inputs: buildFileInputs([...Object.values(tools), ...Object.values(helpers), ...configurationFiles, identityHelper])};
}

/** Select an explicit Git for checkout admission; Windows SDK PATHs omit normal Git installs.
 * @param {Config} config @param {NodeJS.Platform} platform */
function windowsGpuGit(config, platform) {
  if (!config.acpp) {return "";}
  const git = config.acpp.git || (platform === "linux" ? "/usr/bin/git" : "");
  if (!absolutePath(git)) {throw new Error("Windows AdaptiveCpp needs an explicit absolute Git executable");}
  return git;
}

/** Reuse the target's Windows-only support inventory instead of maintaining another source
 * registry. Its small literal array admits comments, not executable GYP expressions. @param {string} gyp */
function windowsGpuSources(gyp) {
  const main = syclSources(gyp), source = gyp.slice(gyp.search(/"target_name"\s*:\s*"sycl"/));
  const blocks = [...source.matchAll(/"OS=='win'"\s*,\s*\{[\s\S]*?"sources"\s*:\s*(\[[\s\S]*?\])/g)];
  if (blocks.length !== 1) {throw new Error("Unrecognized maintained Windows GPU source inventory");}
  const extra = JSON.parse((blocks[0]?.[1] || "null").replace(/^\s*#[^\r\n]*/gm, ""));
  if (!Array.isArray(extra) || extra.length < 4 || extra.length > 16 || extra.some(file => typeof file !== "string" || !/^(?:sycl|xmrig)\/[A-Za-z0-9_/-]+\.c(?:pp)?$/.test(file))) {throw new Error("Invalid maintained Windows GPU support sources");}
  const support = [...extra, "sycl/hoohash/host-math.c"];
  if (new Set([...main, ...support]).size !== main.length + support.length) {throw new Error("Duplicated Windows GPU source");}
  return {main, support};
}

/** A single static ABI applies to both halves and the generated factory. Never inherit Node
 * addon DLL annotations, the host's C++ headers, or a dynamic MSVC CRT. @param {Config} config
 * @param {string} output @param {boolean} c */
function windowsGpuHostFlags(config, output, c = false) {
  // Release links are non-incremental; do not embed wall-clock COFF object timestamps.
  return [target, "-mno-incremental-linker-compatible", "-O3", "-DNDEBUG", "-D_MT", "-U_DLL", "-DMOM_NATIVE_RUNTIME", "-D_CRT_SECURE_NO_WARNINGS", "-DNOMINMAX", "-DWIN32_LEAN_AND_MEAN",
    "-Xclang", "--dependent-lib=libcmt", ...(!c ? ["-std=c++20", "-Xclang", "--dependent-lib=libcpmt"] : []),
    "-B", path.dirname(config.tools.clang), "-nostdinc", "-Xclang", "-ivfsoverlay", "-Xclang", config.overlay,
    ...config.includeRoots.flatMap(root => ["-isystem", root]), "-I", repo, "-I", path.join(repo, "xmrig"), "-I", output,
    ...(!c && !config.acpp ? ["-include", path.join(repo, "tests/native/dpcpp-static/static-windows.h")] : []),
    "-fno-strict-aliasing", "-ffunction-sections", "-fdata-sections"];
}

/** The generic AdaptiveCpp pass embeds HCF/LLVM IR directly in its host object. Keep the
 * original per-source CUDA exceptions: only CN/GPU and Pearl use the native CUDA API path.
 * @param {Config} config @param {string[]} host @param {string} gyp @param {string} wrapper */
function windowsGpuDeviceFlags(config, host, gyp, wrapper) {
  if (config.acpp) {
    // SSCP embeds this Windows module for later GPU retargeting. Host autolink directives
    // are not device libraries: their COFF metadata is invalid in AMD's ELF codegen. The
    // final native host link already names the complete static CRT/SDK closure explicitly.
    return [...host, "-fno-autolink", "-ffp-contract=off", "-D_USE_MATH_DEFINES", "-DACPP_STATIC_RUNTIME", "-D__ACPP__", "-DMOM_SYCL_ADAPTIVECPP",
      ...gpuProfile(config.profile.slice(5), Boolean(config.cuda?.images)).definitions, ...(config.profile === "acpp-hip" ? ["-D__HIP_PLATFORM_AMD__"] : []),
      "-isystem", path.join(config.acpp.source, "include"), "-isystem", config.acpp.generatedInclude,
      ...(config.cuda ? ["-isystem", path.join(config.cuda.sdk, "include"), ...(config.cuda.nvrtcInclude ? ["-isystem", config.cuda.nvrtcInclude] : [])] : []),
      "-D__ACPP_ENABLE_LLVM_SSCP_TARGET__", "-include", path.join(config.acpp.source, "include/hipSYCL/glue/llvm-sscp/hcf_registration.hpp"),
      "-fplugin=" + config.acpp.plugin, "-fpass-plugin=" + config.acpp.plugin, "-Xclang", "-disable-O0-optnone", "-mllvm", "-acpp-sscp"];
  }
  // compilerRoot admits icpx; its host-only selector leaves the device builtin policy unchanged.
  return [...host, ...dpcppGpuFlags(gyp, config.profile), "-DMOM_EMBEDDED_SYCL_JIT",
    ...(config.profile === "dpcpp-opencl" ? portableOpenclTranslatorFlags(wrapper) : []),
    ...(config.cuda ? ["--cuda-path=" + config.cuda.sdk, ...(config.cuda.images ? ["-DMOM_EMBEDDED_CUDA_IMAGES"] :
      ["-isystem", config.cuda.nvrtcInclude || "", "-DMOM_STATIC_NVRTC", "-DMOM_EMBEDDED_NVRTC_HEADERS"])] : ["-fsycl-targets=spir64"]),
    ...(config.profile === "oneapi" && config.compilerRoot ? ["-Xarch_host", "-fno-builtin"] : [])];
}
/** Source layout and JIT paths must match the existing compiler profile, not GPU names.
 * @param {Config} config @param {string} file @param {string} wrapper */
function windowsGpuSourceFlags(config, file, wrapper) {
  if (!config.acpp) {
    // Keep host/device specialization names aligned and stable across builds. The shared
    // helper preserves the existing oneAPI ID and qualifies the DPC++ profiles too.
    const prefix = (config.profile === "oneapi" && config.compilerRoot) ||
      config.profile === "dpcpp" || config.profile === "dpcpp-opencl" ? dpcppUniquePrefixFlags(file, config.profile) : [];
    return [...dpcppSourceFlags(wrapper, file, config.profile), ...prefix];
  }
  if (config.profile !== "acpp-cuda" && config.profile !== "acpp-hip") {return [];}
  return [
    ...(config.profile === "acpp-cuda" && ["sycl/cn-gpu/cn-gpu.cpp", "sycl/pearlhash/pearlhash.cpp"].includes(file) ? ["-DMOM_SYCL_HAS_CUDA"] : []),
    ...(config.profile === "acpp-cuda" && !config.cuda?.images && file === "sycl/pearlhash/pearlhash.cpp" ? ["-DMOM_EMBEDDED_NVRTC_HEADERS"] : []),
    ...(file.startsWith("sycl/") && file.endsWith(".cpp") ? ["-mllvm", "-acpp-sscp-hcf-object-id-namespace=" + config.profile] : []),
  ];
}

/** Preserve the established host/device source split, flags and generated-header dependencies.
 * @param {{sourceFiles: string[], deviceCount: number, repo: string, output: string, config: Config,
 * gpuFlags: string[], hostFlags: string[], perSourceFlags: string[][]}} plan */
function windowsGpuCompileCommands(plan) {
  return plan.sourceFiles.map((file, index) => {
    const object = path.join(plan.output, index + ".obj"), depfile = path.join(plan.output, index + ".d");
    const c = file.endsWith(".c"), device = index < plan.deviceCount;
    const flags = device ? [...plan.gpuFlags, ...(plan.perSourceFlags[index] || [])] :
      c ? windowsGpuHostFlags(plan.config, plan.output, true) : plan.hostFlags;
    // Plain support TUs bypass the mixed-SYCL host selector. Keep their memory builtins,
    // but use the public Intel codegen switch instead of requiring libirc helpers.
    const hostRuntime = !device && plan.config.profile === "oneapi" && plan.config.compilerRoot ? ["-i_no-use-libirc"] : [];
    const strict = file === "sycl/hoohash/host-math.c" ? ["-O2", "-fno-fast-math", "-ffp-contract=off", "-fno-builtin"] : [];
    const generatedInputs = [];
    if (plan.config.acpp && file.startsWith("sycl/") && !c) {generatedInputs.push(path.join(plan.output, "sycl/sycl.hpp"));}
    if (!plan.config.acpp && file === "sycl/kawpow/kawpow.cpp") {generatedInputs.push(path.join(plan.output, "kawpow-jit-resources.inc"));}
    if (plan.config.cuda && !plan.config.cuda.images && file === "sycl/pearlhash/pearlhash.cpp") {generatedInputs.push(path.join(plan.output, "nvrtc-headers.inc"));}
    return {command: c ? plan.config.tools.clangC : plan.config.tools.clang,
      args: [...flags, ...hostRuntime, ...strict, "-MD", "-MV", "-MF", depfile, "-c", path.join(plan.repo, file), "-o", object],
      source: path.join(plan.repo, file), object, depfile, generatedInputs,
      dependencyFormat: /** @type {"nmake"} */ ("nmake"), timeout: 300000};
  });
}

/** Discard only the original combined object's offload containers, not code or CRT directives.
 * Device linking consumes the untouched original; specialization IDs in the retained host
 * therefore cannot diverge through a second compilation. @param {string} text */
function windowsHostSections(text) {
  const names = [...text.matchAll(/^\s*Name:\s*(__CLANG_OFFLOAD_BUNDLE__[^\s(]+)/gm)].map(match => match[1] || "");
  if (names.length < 2 || names.length > 4 || new Set(names).size !== names.length ||
      names.filter(name => name.startsWith("__CLANG_OFFLOAD_BUNDLE__host-")).length !== 1 ||
      names.some(name => !/^__CLANG_OFFLOAD_BUNDLE__(?:host-x86_64-pc-windows-msvc(?:19\.33\.0)?|sycl-[A-Za-z0-9_-]+)$/.test(name))) {throw new Error("Unrecognized Windows combined host/device containers");}
  return names;
}

/** Build current maintained sources only; no retained-object or runtime fallback is accepted.
 * The caller supplies a kernel-enforced memory envelope and a fresh external output root.
 * @param {string} configFile @param {string} output @param {NodeJS.ProcessEnv} [environment] */
function buildNativeWindowsGpu(configFile, output, environment = process.env) {
  if (!["linux", "win32"].includes(process.platform) || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows GPU compilation requires MOM_AOT_BUILD_ONLY=1 on Linux or Windows");}
  const jobs = windowsGpuBuildJobs(environment);
  if (!absolutePath(configFile) || !absolutePath(output) || fs.existsSync(output)) {throw new Error("Select absolute descriptor and new Windows GPU output paths");}
  const parent = fs.realpathSync(path.dirname(output)), relative = path.relative(fs.realpathSync(repo), path.join(parent, path.basename(output)));
  if (!relative || (!relative.startsWith(".." + path.sep) && !path.isAbsolute(relative))) {throw new Error("Windows GPU output must be outside the maintained repository");}
  const configSha256 = fileHash(configFile), config = validateWindowsGpuConfig(readBuildJson(configFile));
  const cuda = config.cuda, acpp = config.acpp;
  // A Windows compiler SDK need not contain Git. Never rely on the restricted compiler PATH
  // to select the executable that validates the pinned AdaptiveCpp checkout.
  const git = windowsGpuGit(config, process.platform);
  if (fileHash(configFile) !== configSha256) {throw new Error("Windows GPU descriptor changed during admission");}
  const startedAt = Date.now(), state = nativeSourceInputs(repo, "gpu");
  const gyp = fs.readFileSync(path.join(repo, "binding.gyp"), "utf8"), sources = windowsGpuSources(gyp), profile = config.profile;
  const wrapper = fs.readFileSync(path.join(repo, "scripts/cxx-combined.sh"), "utf8");
  const env = windowsGpuBuildEnvironment(environment);
  for (const key of Object.keys(env)) {if (/^(?:PATH|CL|_CL_|LINK|_LINK_|INCLUDE|LIB|LIBPATH|CPATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH|LIBRARY_PATH|CFLAGS|CXXFLAGS|GIT_.*)$/i.test(key)) {delete env[key];}}
  env["GIT_NO_LAZY_FETCH"] = "1";
  const extension = process.platform === "win32" ? ".exe" : "", bin = path.dirname(config.tools.clang);
  const systemPath = process.platform === "win32" ? [path.join(env["SystemRoot"] || "C:\\Windows", "System32")] : ["/usr/bin", "/bin"];
  if (config.compilerRoot) {
    for (const key of Object.keys(env)) {if (/^(?:ICXCFG|ICPXCFG)$/i.test(key)) {delete env[key];}}
  }
  env["PATH"] = [bin, ...(cuda ? [path.join(cuda.sdk, "bin")] : []), ...systemPath].join(path.delimiter);
  const oneapi = config.compilerRoot ? resolveOneapiCompiler(config, env) : null;
  // This pinned driver does not use optional opt/offload-packager binaries. Every helper it
  // does invoke is mandatory and content-pinned; -B and a restricted PATH prevent SDK fallback.
  const sdkTools = acpp ? [git, acpp.plugin, path.join(acpp.generatedInclude, "hipSYCL/common/config.hpp")] :
    oneapi ? oneapi.sdkTools :
      ["clang-offload-bundler", "sycl-post-link", "llvm-spirv", "clang-offload-wrapper", "clang-linker-wrapper", "llvm-link", "llvm-foreach", "llc"].map(name => path.join(bin, name + extension));
  if (oneapi) {env["PATH"] = [...new Set([bin, ...oneapi.helperDirectories, ...systemPath])].join(path.delimiter);}
  const cudaBin = cuda && !acpp ? path.join(cuda.sdk, "bin") : undefined;
  const deviceLinkEnvironment = cudaBin ? {...env,
    PATH: [cudaBin, ...(env["PATH"] || "").split(path.delimiter).filter(segment => segment && segment !== cudaBin)].join(path.delimiter)} : env;
  const images = cuda?.images;
  const headers = cuda?.headers ? cudaHeaderInputs(cuda.headers).files : {};
  const imageInputs = images ? require("./native-cuda-images").cudaImageInputs(images).files : {};
  const initial = {
    ...(oneapi ? oneapi.inputs : buildFileInputs([...Object.values(config.tools), ...sdkTools])),
    ...buildFileInputs([
      config.overlay,
      ...(cuda ? [path.join(cuda.sdk, "include/cuda.h"),
        ...(!acpp ? [path.join(cuda.sdk, "nvvm/libdevice/libdevice.10.bc"), ...["ptxas", "fatbinary"].map(name => path.join(cuda.sdk, "bin", name + extension))] : []),
        ...(cuda.nvrtcInclude ? [path.join(cuda.nvrtcInclude, "nvrtc.h")] : []), ...Object.keys(headers), ...Object.keys(imageInputs)] : []),
    ]),
  };
  if (Object.entries({...headers, ...imageInputs}).some(([file, sha]) => initial[file] !== sha)) {throw new Error("CUDA compiler inputs changed after admission");}
  const version = oneapi ? {error: undefined, signal: null, status: 0, stdout: oneapi.cxxVersion} :
    spawnSync(config.tools.clang, ["--version"], {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20, env});
  if (version.error || version.signal || version.status !== 0 || typeof version.stdout !== "string" ||
      (!oneapi && (acpp ? !/\bclang version 21\.1\.8\b/.test(version.stdout) : !version.stdout.includes(pins.source)))) {
    throw new Error("Select the matching pinned " + (acpp ? "LLVM21 AdaptiveCpp" : "DPC++ source") + " frontend");
  }
  if (cuda && !acpp) {
    // Windows x64 uses 32-bit long and signed char. The selected frontend links this exact
    // intrinsic library as well as CUDA libdevice; neither is a header dependency in .d files.
    const resource = spawnSync(config.tools.clang, ["--print-resource-dir"], {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20, env});
    if (resource.error || resource.signal || resource.status !== 0 || !absolutePath(resource.stdout.trim())) {throw new Error("Cannot resolve the selected CUDA frontend resource directory");}
    Object.assign(initial, buildFileInputs([path.join(resource.stdout.trim(), "lib/nvptx64-nvidia-cuda/libspirv.l32.signed_char.bc")]));
  }
  const identityHelper = path.join(__dirname, "build-native-windows-hiprtc-sdk.js");
  const compileKeyFiles = [...new Set(Object.keys(initial).filter(file => file !== identityHelper && !Object.hasOwn(imageInputs, file)).map(file => fs.realpathSync(file)))];
  const hostFlags = windowsGpuHostFlags(config, output),
    perSourceFlags = sources.main.map(file => windowsGpuSourceFlags(config, file, wrapper));
  const gpuFlags = windowsGpuDeviceFlags(config, hostFlags, gyp, wrapper);
  const device = path.join(output, "device.obj"), deviceLink = acpp ? [] : windowsDeviceLinkArgs(gyp, cuda?.sdk, sources.main.map((_, index) => path.join(output, index + ".obj")), device, profile, wrapper);
  fs.mkdirSync(output);
  /** @param {string} command @param {string[]} args @param {string} label @param {number} [timeout] @param {NodeJS.ProcessEnv} [runEnvironment] */
  const run = (command, args, label, timeout = 300000, runEnvironment) => {
    fs.writeFileSync(path.join(output, label + ".command.json"), JSON.stringify({command, args}) + "\n", {flag: "wx"});
    const result = spawnSync(command, args, {cwd: repo, encoding: "utf8", timeout, maxBuffer: 16 << 20, env: runEnvironment || env});
    fs.writeFileSync(path.join(output, label + ".log"), (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
    if (result.error || result.signal || result.status === null) {throw new Error("Windows GPU compiler lifecycle unresolved: " + label, {cause: result.error});}
    return result;
  };
  /** @param {string} command @param {string[]} args @param {string} label
   * @param {{cacheHit: boolean, stdout: string, stderr: string, error?: string}} result */
  const recordCompile = (command, args, label, result) => {
    fs.writeFileSync(path.join(output, label + ".command.json"), JSON.stringify({command, args, cacheHit: result.cacheHit}) + "\n", {flag: "wx"});
    fs.writeFileSync(path.join(output, label + ".log"), (result.stdout || "") + (result.stderr || "") + (result.error ? result.error + "\n" : ""), {flag: "wx"});
  };
  fs.writeFileSync(path.join(output, "inputs-before.json"), JSON.stringify({configFile, configSha256, config, startedAt, sourceState: state,
    sdkInputs: initial, hostFlags, gpuFlags, perSourceFlags, deviceLink}, null, 2) + "\n", {flag: "wx"});
  fs.writeFileSync(path.join(output, "compiler-version.log"), version.stdout, {flag: "wx"});
  if (oneapi) {fs.writeFileSync(path.join(output, "c-compiler-version.log"), oneapi.cVersion, {flag: "wx"});}
  const admitAcpp = () => acpp && admitSyclSource(acpp.source, output, true, (command, args, log) => {
    const result = spawnSync(command === "git" ? git : command, args, {env, encoding: "utf8", timeout: 30000, maxBuffer: 4 << 20});
    fs.appendFileSync(log, (result.stdout || "") + (result.stderr || ""));
    if (result.error || result.signal || result.status !== 0) {throw new Error("AdaptiveCpp source admission failed", {cause: result.error});}
    return result.stdout;
  });
  admitAcpp();
  if (config.acpp) {writeAdaptiveCppUmbrella(config.acpp.source, output);}
  fs.writeFileSync(path.join(output, "kawpow-jit-resources.inc"), nativeResources(["device", "keccak"].map(name => ({name: "kawpow-" + name + ".inc", data: fs.readFileSync(path.join(repo, "sycl/kawpow", name + ".inc"))}))), {flag: "wx"});
  if (cuda?.headers) {fs.copyFileSync(cuda.headers, path.join(output, "nvrtc-headers.inc"), fs.constants.COPYFILE_EXCL);}
  const compiled = [], dependencies = [];
  /** @type {Array<{file: string, object: string, status: number | null, sourceSha256: string, flags: string[], objectSha256: string | null, cacheHit: boolean}>} */
  const results = [];
  const sourceFiles = [...sources.main, ...sources.support];
  const compileCommands = windowsGpuCompileCommands({sourceFiles, deviceCount: sources.main.length, repo, output, config,
    gpuFlags, hostFlags, perSourceFlags});
  const vfs = vfsFiles(config.overlay);
  /** @param {{depfileText: string, repo: string}} input @returns {string[]} */
  const resolveWindowsDependencies = ({depfileText, repo: dependencyRepo}) => {
    const dependencies = buildDependencies(depfileText, dependencyRepo, "nmake");
    if (!dependencies?.length) {throw new Error("Incomplete Windows GPU compiler dependency graph");}
    return [...new Set(dependencies.map(dependency => {
      const stat = fs.statSync(dependency, {throwIfNoEntry: false});
      const selected = stat?.isFile() ? dependency : vfs.get(dependency) || vfs.get(dependency.toLowerCase());
      const selectedStat = selected ? fs.statSync(selected, {throwIfNoEntry: false}) : undefined;
      if (!selected || !selectedStat?.isFile()) {throw new Error("Unresolved Windows GPU dependency: " + dependency);}
      return fs.realpathSync(selected);
    }))];
  };
  const objectBatch = compileNativeObjects({repo, output, namespace: `windows/gpu/${profile}`, commands: compileCommands,
    environment: env, includeRoots: [output, ...nativeIncludeRoots(compileCommands.flatMap(item => item.args), repo, output)],
    keyFiles: compileKeyFiles, resolveDependencies: resolveWindowsDependencies, jobs, finalize: false});
  for (const [index, file] of sourceFiles.entries()) {
    const command = compileCommands[index], result = objectBatch.objects[index];
    if (!command || !result) {throw new Error("Native object helper returned an incomplete Windows compile batch");}
    recordCompile(command.command, command.args, "compile-" + index, result);
    results.push({file, object: result.object, status: result.status, sourceSha256: fileHash(path.join(repo, file)), flags: command.args.slice(0, command.args.indexOf("-MD")),
      objectSha256: result.status === 0 ? result.objectSha256 : null, cacheHit: result.cacheHit});
    dependencies.push(result.depfile); compiled.push(result.object);
    console.log("[native-windows] " + file + ": " + (result.status === 0 ? result.cacheHit ? "cached" : "compiled" : "FAILED"));
  }
  fs.writeFileSync(path.join(output, "compilation.json"), JSON.stringify(results, null, 2) + "\n", {flag: "wx"});
  if (results.some(row => row.status !== 0)) {throw new Error("Windows GPU compilation failed; original per-source logs retained");}
  const deviceObjects = compiled.slice(0, sources.main.length);
  const verifyCompiled = () => {
    if (results.some(row => fileHash(row.object) !== row.objectSha256)) {throw new Error("Windows compiled object changed after source compilation");}
  };
  verifyCompiled();
  if (profile === "dpcpp-opencl") {fs.mkdirSync(path.join(output, "device-images"));}
  if (!acpp && run(config.tools.clang, deviceLink, "device-link", 900000, deviceLinkEnvironment).status !== 0) {throw new Error("Windows device linking failed");}
  verifyCompiled();
  if (profile === "dpcpp-opencl") {
    const portability = validateDirectory(path.join(output, "device-images"));
    fs.writeFileSync(path.join(output, "portability.json"), JSON.stringify(portability, null, 2) + "\n", {flag: "wx"});
    if (portability.errors.length) {throw new Error("Non-portable Windows OpenCL images: " + portability.errors.join("; "));}
  }
  const retained = [], retention = [], hostIds = new Set();
  let wrapperReceipt;
  if (acpp) {retained.push(...deviceObjects);} else {
    for (const [index, object] of deviceObjects.entries()) {
      verifyCompiled();
      const inspection = run(config.tools.llvmReadobj, ["--sections", object], "combined-sections-" + index);
      if (inspection.status !== 0) {throw new Error("Cannot inspect Windows combined object");}
      const host = path.join(output, "host-" + index + ".obj"), sections = windowsHostSections(inspection.stdout);
      const removed = [...inspection.stdout.matchAll(/\s+Section \{\r?\n\s+Number:\s+(\d+)\r?\n\s+Name:\s*(\S+)/g)]
        .filter(match => sections.includes(match[2] || "")).map(match => Number(match[1]));
      if (removed.length !== sections.length) {throw new Error("Windows offload section indices are unresolved");}
      verifyCompiled();
      if (run(config.tools.llvmObjcopy, [...sections.map(name => "--remove-section=" + name), object, host], "retain-host-" + index).status !== 0) {throw new Error("Cannot retain original Windows host object");}
      retention.push(verifyWindowsHostRetention(object, host, removed)); verifyCompiled();
      const originalIds = extractSpecializationIds(object), ids = extractSpecializationIds(host);
      if (JSON.stringify(originalIds) !== JSON.stringify(ids)) {throw new Error("Windows host retention changed specialization IDs");}
      for (const id of ids) {hostIds.add(id);} retained.push(host);
    }
    if (JSON.stringify([...hostIds].sort()) !== JSON.stringify(extractSpecializationIds(device))) {throw new Error("Windows full graph specialization IDs do not match its device image");}
    const registered = path.join(output, "device-registration.obj"); wrapperReceipt = transformWindowsWrapper({input: device, output: registered, profile,
      tools: {llvmObjcopy: config.tools.llvmObjcopy, llvmReadobj: config.tools.llvmReadobj, llvmNm: config.tools.llvmNm}});
    retained.push(registered);
  }
  const factoryName = "mom_native_gpu_" + profile.replaceAll("-", "_"), bridge = gpuTable(fs.readFileSync(path.join(repo, "sycl/lib.h"), "utf8"), factoryName);
  fs.writeFileSync(path.join(output, "gpu-table.h"), bridge.header, {flag: "wx"}); fs.writeFileSync(path.join(output, "gpu-factory.cpp"), bridge.definition, {flag: "wx"});
  const factoryObject = path.join(output, "gpu-factory.obj"), factoryDep = path.join(output, "gpu-factory.d");
  const factoryCommand = {command: config.tools.clang, args: [...hostFlags,
    ...(config.profile === "oneapi" && config.compilerRoot ? ["-i_no-use-libirc"] : []), "-MD", "-MV", "-MF", factoryDep,
    "-c", path.join(output, "gpu-factory.cpp"), "-o", factoryObject], source: path.join(output, "gpu-factory.cpp"), object: factoryObject,
  depfile: factoryDep, generatedInputs: [path.join(output, "gpu-table.h")], dependencyFormat: /** @type {"nmake"} */ ("nmake"), timeout: 300000};
  const factoryBatch = compileNativeObjects({repo, output, namespace: `windows/gpu/${profile}`, commands: [factoryCommand], environment: env,
    includeRoots: [output, ...nativeIncludeRoots(factoryCommand.args, repo, output)], resolveDependencies: resolveWindowsDependencies, jobs,
    keyFiles: compileKeyFiles, finalize: true,
    liveObjects: [...compileCommands.map(item => item.object), factoryObject]});
  const factoryResult = factoryBatch.objects[0];
  if (!factoryResult) {throw new Error("Native object helper returned an incomplete Windows factory batch");}
  recordCompile(factoryCommand.command, factoryCommand.args, "factory", factoryResult);
  if (factoryResult.status !== 0 || factoryResult.error || factoryResult.signal) {throw new Error("Windows GPU factory compilation failed");}
  dependencies.push(factoryResult.depfile);
  const objectCache = {cacheHits: objectBatch.cacheHits + factoryBatch.cacheHits,
    cacheMisses: objectBatch.cacheMisses + factoryBatch.cacheMisses, cacheRoot: objectBatch.cacheRoot, ninja: objectBatch.ninja || factoryBatch.ninja, jobs};
  const isolated = isolateCoffObjects({tools: {llvmObjcopy: config.tools.llvmObjcopy, llvmReadobj: config.tools.llvmReadobj, llvmNm: config.tools.llvmNm},
    objects: [...retained, ...compiled.slice(sources.main.length), factoryObject], output: path.join(output, "isolated"),
    prefix: "mom_" + profile.replaceAll("-", "_") + "_gpu_", entryPoints: [factoryName]});
  const sdkInputs = {...initial, ...dependencyHashes(dependencies, config.overlay, output, startedAt)};
  // Hash the admitted tools again independently of dependency discovery; never restamp a
  // changed compiler or source graph as a successful current-source component.
  if (Object.entries(sdkInputs).some(([file, sha]) => fileHash(file) !== sha) || fileHash(configFile) !== configSha256) {throw new Error("Windows GPU SDK/descriptor changed during compilation");}
  verifyCompiled();
  admitAcpp();
  verifySourceInputs(state, nativeSourceInputs(repo, "gpu"), "Windows native GPU");
  const receipt = {version: 1, factory: factoryName, profile, configFile, configSha256, sourceState: state, sdkInputs, compiled: results, retention,
    ...(images ? {cudaImages: images} : {}),
    objects: isolated.outputs, objectCache, factoryCacheHit: factoryResult.cacheHit,
    isolation: path.join(output, "isolated/result.json"), wrapper: wrapperReceipt, specializationIds: [...hostIds].sort(),
    scope: "Original Windows GPU source graph compiled and isolated; build-only, not runtime/GPU/performance/release qualification"};
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(receipt, null, 2) + "\n", {flag: "wx"});
  return receipt;
}

module.exports = {validateWindowsGpuConfig, resolveOneapiCompiler, windowsGpuGit, windowsGpuBuildJobs, windowsGpuBuildEnvironment, windowsGpuSources, windowsGpuHostFlags, windowsGpuDeviceFlags,
  windowsGpuSourceFlags, windowsGpuCompileCommands, windowsHostSections, buildNativeWindowsGpu};
if (require.main === module) {
  try {
    const [config, output, extra] = process.argv.slice(2);
    if (!config || !output || extra) {throw new Error("Usage: build-native-windows-gpu.js /absolute/sdk.json /new/output");}
    buildNativeWindowsGpu(config, output);
  } catch (error) {console.error(error); process.exitCode = 1;}
}
