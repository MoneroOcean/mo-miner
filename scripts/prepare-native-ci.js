"use strict";

// Same-run tar handoffs for manual Linux native prerequisite stages. Producer receipts contain
// absolute source/build/tool paths, so restore each selected state beneath the same /tmp roots.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"), {createHash} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson, readBuildText, sourceInputs, nativeSourceInputs, verifySourceInputs, buildFileInputs} = require("./build-source-inputs");
const {ioPins, nativeIoInputs, validateIoArchive} = require("./build-native-io");
const {nativeOpenSslInputs, opensslPin, opensslBuildRecipe} = require("./build-native-openssl");
const {verifyIoArchives, fetchIoArchives, fetchWindowsOpenSslArchives, verifyNvrtcArchives, gitSourceComponents, sourceRecipe, origins, verifyZstdArchive,
  verifyWindowsHostSdkArchives, windowsHostSdkCatalog} = require("./fetch-native-sdk-sources");
const {hermesPins, hermesSdkPlan} = require("./build-native-hermes");
const {windowsControllerHandoffInputs, windowsLocalControllerHandoffInputs} = require("./native-windows-controller-handoff");
const {windowsNativeCoreHandoffInputs} = require("./native-windows-core-handoff");
const {capturePackageCopyright, auditGnuHeaders} = require("./prepare-native-gnu-headers");
const {safeRelative: safeWindowsSdkRelative, defaultWindowsNativeLibraryPolicy} = require("./prepare-native-windows-sdk");
const {zstdPin, nativeZstdInputs} = require("./build-native-zstd");
const {sdkSourceEnvironment} = require("./build-environment");
const {nativeHermesInputs, nativeStaticRuntimeInputs} = require("./native-release-sdk-inputs");
const {syclRevision, staticSyclInputs, runJitCacheIdentity, runHcfRepro} = require("./probe-static-sycl");
const {nvrtcSdkInputs} = require("./prepare-native-nvrtc-sdk");
const {nvrtcCohortInfo} = require("./probe-static-nvrtc");
const {cudaHeaderInputs} = require("./native-jit-headers");
const {cudaImageInputs} = require("./native-cuda-images");
const {sdkComponents, prepareSdkSources, verifyPreparedSdkSources, sdkSourceFiles, verifySdkExport} = require("./prepare-native-sdk-sources");
const dpcppCudaInputs = require("./native-dpcpp-cuda-inputs");
const {cudaPackageNames, auditCudaPackageCopyrights, copyDpcppBuildInputs, copyCudaBuildInputs, cudaToolkitProbeCMakeLists, probeCudaToolkit,
  inputFileRecord, preparedDpcppCudaSourceOrigin, dpcppCudaInputsManifest, auditDpcppCudaInputs, cudaToolkitProbeRoot, stageDpcppCudaInputs} = dpcppCudaInputs;
const {archiveName: llvmArchiveName, archiveSha256: llvmArchiveSha256} = require("./native-llvm-sdk");
const {nativeOpenmpInputs} = require("./build-native-openmp-sdk");
const {comgrDeviceInputs} = require("./build-native-comgr-sdk");
const {comgrSdkInputs} = require("./native-comgr-inputs");
const {nativeRocmHostInputs} = require("./native-release-sdk-inputs");
const {hiprtcHeaders} = require("./build-native-rocm-resources");
const {jitSdkPlan} = require("./native-jit-sdk");
const {jitBuildStages, jitBuildOutputs} = require("./native-jit-stages");
const {dpcppEntries, compilerArchives: jitCompilerArchives} = require("./native-compiler-isolation");
const {dpcppComponentInputs} = require("./native-dpcpp-component");
const {isStaticArchive} = require("./native-audit");
const {linuxNativeSdkLldPath, linuxNativeSdkDescriptorConfig} = require("./native-linux-sdk-config");
const {REQUIRED: jitBaseRequired, pinnedCommitDate} = require("./prepare-native-jit-base");
const {resourceTarget, verifyJitResources} = require("./prepare-native-sycl-jit-resources");
const tar = "/usr/bin/tar", temp = "/tmp";
const repo = path.resolve(__dirname, "..");
const rocmHipHeaderPatch = "scripts/patches/rocm-hip-headers-only.patch",
  rocmHipHeaderPatchSha256 = "caed551ccaa0d8c3469676d4e3a74c6cd353ed75425297398119d36eca3854e0",
  rocmHipCmakeOriginalSha256 = "4c968219e2a92609160166b73ad924ef4ce1194f451fc7afdd7c11c9c798ced6",
  rocmHipCmakePatchedSha256 = "de18cbc92de9992f90648e7df0ec591c6d23444f4827719227f5683fdcb0a7da";
const cutlassVersion = "v4.6.1", cutlassArchiveSha256 = "455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c",
  cutlassLicenseSha256 = "ad1513ca983da737a6843364ad19e8424a56d860736c2da773b8c5be527e1475";
// Record terms from the runtime, static-development and NVML-header payloads.
// cuda-minimal-build is only a dependency metapackage and ships no copyright file.
/** @param {string} name */
const cudaCopyrightName = name => name.replace(/-12-6$/, "-12.6-copyright.txt");
const rocmResourcesRecipe = ["scripts/build-native-rocm-resources.js", "scripts/native-comgr-inputs.js", "scripts/build-native-comgr-sdk.js",
  "scripts/build-native-sycl-sdk.js", "scripts/prepare-native-sdk-sources.js", "scripts/build-source-inputs.js", "scripts/build-environment.js"];
const rocmParserSource = Object.freeze({url: "https://files.pythonhosted.org/packages/3c/ba/d8d168a4b54cae66eaf13d1d9197ca9349c94653815e061f79e7eed86c01/CppHeaderParser-2.7.4.tar.gz",
  sha256: "382b30416d95b0a5e8502b214810dcac2a56432917e2651447d3abe253e3cc42", bytes: 54445, directory: "CppHeaderParser-2.7.4"});
const rocmPythonDependencies = Object.freeze({ply: "3.11-6", xxd: "2:9.1.0016-1ubuntu7.20"});
const rocmOpenGLDependencies = Object.freeze({"libegl-dev": "1.7.0-1build1", "libopengl-dev": "1.7.0-1build1"});
/** @type {Readonly<Record<string,string>>} */
const rocmPinnedHostPackages = Object.freeze({"python3-ply": rocmPythonDependencies.ply, xxd: rocmPythonDependencies.xxd,
  ...rocmOpenGLDependencies});
const linuxNativeBuildReserveBytes = 4 * 1024 * 1024 * 1024;
const windowsNativeCorePackReserveMs = 7 * 60 * 1000, windowsNativeCoreMinBuildMs = 5 * 60 * 1000,
  windowsNativeCoreMaxFetchMs = 20 * 60 * 1000, windowsNativeCoreMaxBuildMs = 40 * 60 * 1000,
  windowsNativeCoreMaxArchiveStepMs = 2 * 60 * 1000;
const windowsHostSdkArchiveMaxMs = 20 * 60 * 1000, windowsHostSdkPackageReserveMs = 4 * 60 * 1000,
  windowsHostSdkComposeReserveMs = 12 * 60 * 1000, windowsHostSdkMaxJobMs = 60 * 60 * 1000;
/** @typedef {{version:number,compiler:{version:string,sha256:string},libraries:Array<{name:string,source:string,path:string,bytes:number,sha256:string}>,licenses:Array<{source:string,path:string,bytes:number,sha256:string}>}} WindowsIntelRuntimeCatalog */
/** @typedef {{kind:"library",name:string,source:string,path:string,bytes:number,sha256:string}|{kind:"license",source:string,path:string,bytes:number,sha256:string}} WindowsIntelRuntimePin */
/** @typedef {WindowsIntelRuntimePin & {source:string}} WindowsIntelRuntimeEntry */
const windowsIntelRuntimeCatalog = /** @type {WindowsIntelRuntimeCatalog} */ (require("../runtime/windows-intel-runtime.json"));

/** @param {string} name */
function safeRelative(name) {
  return typeof name === "string" && name.length > 0 && name.length <= 4096 && !name.startsWith("/") &&
    !name.includes("\\") && !/[\r\n\0]/.test(name) && name.split("/").every(part => part && part !== "." && part !== "..");
}

/** @param {string} command @param {string[]} args @param {number} timeout */
function runTar(command, args, timeout = 30 * 60 * 1000) {
  const result = spawnSync(tar, args, {encoding: "utf8", timeout, maxBuffer: 128 << 20});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error("Native CI tar " + command + " failed" + (result.stderr ? ": " + result.stderr.slice(0, 2048) : ""), {cause: result.error});
  }
  return result.stdout || "";
}

/** @param {string} base @param {string[]} paths */
function archivePaths(base, paths) {
  const resolved = fs.realpathSync(base);
  if (!Array.isArray(paths) || !paths.length || paths.length > 512 || new Set(paths).size !== paths.length ||
      paths.some(name => !safeRelative(name)) || paths.reduce((bytes, name) => bytes + Buffer.byteLength(name) + 1, 0) > (128 << 10)) {
    throw new Error("Native CI tar needs a bounded set of exact relative state paths");
  }
  if (paths.length > 8 && paths.some(name => !fs.lstatSync(path.resolve(resolved, ...name.split("/"))).isFile())) {
    throw new Error("Native CI tar permits more than eight inputs only as an explicit regular-file selection");
  }
  const roots = new Set(paths.map(name => name.split("/")[0] || ""));
  if (roots.size > 8) {throw new Error("Native CI tar input exceeds the eight-root limit");}
  for (const name of paths) {
    const item = path.resolve(resolved, ...name.split("/")), relative = path.relative(resolved, item), stat = fs.lstatSync(item);
    if (path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep) || stat.isSymbolicLink() ||
        (!stat.isDirectory() && !stat.isFile()) || fs.realpathSync(item) !== item) {
      throw new Error("Native CI tar input escaped or is not a canonical regular state path: " + name);
    }
  }
  return resolved;
}

/** @param {string} archive @param {string} base @param {string[]} paths @param {number} [timeoutMs] @returns {{bytes: number}} */
function packState(archive, base, paths, timeoutMs = 30 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 30 * 60 * 1000) {throw new Error("Native CI tar timeout is outside the bounded range");}
  if (!path.isAbsolute(archive) || /[\r\n\0]/.test(archive)) {throw new Error("Native CI tar output must be an absolute path");}
  const root = archivePaths(base, paths), parent = fs.realpathSync(path.dirname(archive)), target = path.join(parent, path.basename(archive));
  // A sibling under the base is safe; only selected inputs could recursively include the output.
  if (target !== path.resolve(archive) || paths.some(name => {
    const input = path.join(root, name); return target === input || target.startsWith(input + path.sep);
  }) || fs.existsSync(target)) {
    throw new Error("Native CI tar output must be new and outside its inputs");
  }
  const fd = fs.openSync(target, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY, 0o600);
  try {
    const result = spawnSync(tar, ["--create", "--file=-", "--format=posix", "--numeric-owner", "--directory", root, "--", ...paths],
      {stdio: ["ignore", fd, "inherit"], timeout: timeoutMs});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error("Native CI tar create failed", {cause: result.error});
    }
  } catch (error) {
    fs.closeSync(fd);
    fs.unlinkSync(target);
    throw error;
  }
  fs.closeSync(fd);
  const stat = fs.lstatSync(target);
  if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Native CI tar output is not a regular file");}
  return {bytes: stat.size};
}

/** @param {string} archive @param {string[]} expectedRoots @param {number} [timeoutMs] @returns {string[]} */
function listState(archive, expectedRoots, timeoutMs = 5 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 30 * 60 * 1000) {
    throw new Error("Native CI tar listing timeout is outside the bounded range");
  }
  const listing = runTar("list", ["--list", "--file", archive, "--quoting-style=literal"], timeoutMs);
  if (listing.length > 128 << 20 || listing.includes("\0") || /(^|\n)\.[/\n]/.test(listing)) {
    throw new Error("Native CI tar member list is invalid or excessive");
  }
  const names = listing.split("\n").filter(Boolean), roots = new Set();
  if (!names.length || names.length > 500000) {throw new Error("Native CI tar member count is invalid");}
  for (const member of names) {
    const name = member.endsWith("/") ? member.slice(0, -1) : member;
    if (!safeRelative(name)) {throw new Error("Unsafe native CI tar member path");}
    roots.add(name.split("/")[0] || "");
  }
  const expected = [...new Set(expectedRoots)].sort();
  if (!expected.length || expected.some(name => !safeRelative(name) || name.includes("/")) ||
      [...roots].sort().join("\0") !== expected.join("\0")) {throw new Error("Native CI tar has unexpected state roots");}
  return names;
}

/** Extract a same-run state tar into a temporary directory, then rename selected verified roots
 * to their original absolute /tmp names. GNU tar's `--same-permissions` preserves 0444 receipts.
 * @param {string} archive @param {string} base @param {string[]} expectedRoots @param {string[]} [restoreRoots] @param {number} [timeoutMs] */
function extractState(archive, base, expectedRoots, restoreRoots = expectedRoots, timeoutMs = 30 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 30 * 60 * 1000) {
    throw new Error("Native CI tar extraction timeout is outside the bounded range");
  }
  const deadline = Date.now() + timeoutMs;
  const root = fs.realpathSync(base), stat = fs.lstatSync(archive);
  if (!stat.isFile() || stat.isSymbolicLink() || !expectedRoots.length || expectedRoots.length > 8 ||
      new Set(expectedRoots).size !== expectedRoots.length || expectedRoots.some(name => !safeRelative(name) || name.includes("/")) ||
      !Array.isArray(restoreRoots) || !restoreRoots.length || restoreRoots.length > expectedRoots.length ||
      new Set(restoreRoots).size !== restoreRoots.length || restoreRoots.some(name => !expectedRoots.includes(name))) {
    throw new Error("Native CI tar input or expected roots are invalid");
  }
  listState(archive, expectedRoots, Math.min(5 * 60 * 1000, timeoutMs));
  for (const name of restoreRoots) {
    if (fs.existsSync(path.join(root, name))) {throw new Error("Native CI state restore refuses an existing path: " + name);}
  }
  const staging = fs.mkdtempSync(path.join(root, ".native-ci-restore-"));
  try {
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {throw new Error("Native CI tar extraction deadline expired during listing");}
    runTar("extract", ["--extract", "--file", archive, "--directory", staging, "--no-same-owner", "--same-permissions", "--", ...restoreRoots], remainingMs);
    for (const name of restoreRoots) {
      const item = path.join(staging, name), itemStat = fs.lstatSync(item);
      if (!itemStat.isDirectory() || itemStat.isSymbolicLink()) {throw new Error("Native CI tar root is not a real directory: " + name);}
    }
    for (const name of restoreRoots) {fs.renameSync(path.join(staging, name), path.join(root, name));}
  } finally {
    fs.rmSync(staging, {recursive: true, force: true});
  }
}

/** @param {string} file @param {string[]} args @param {NodeJS.ProcessEnv} env @param {number} timeout */
function runNode(file, args, env, timeout = 50 * 60 * 1000) {
  const result = spawnSync(process.execPath, [path.join(repo, file), ...args],
    {cwd: repo, env, stdio: "inherit", timeout});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error("Native CI command failed: " + file + (result.signal ? " (" + result.signal + ")" : ""), {cause: result.error});
  }
}

const linuxLlvmAdmissionTimeoutMs = 10 * 60 * 1000;

/** Re-admit a restored LLVM SDK without rerunning its write-once producer audit.
 * The child performs the full public admission; this parent guard makes the existing result receipt
 * immutable even if a future verifier accidentally writes to it.
 * @param {{output: string, env: NodeJS.ProcessEnv}} paths
 * @param {typeof runNode} [execute] */
function verifyLinuxLlvmAdmissionCi(paths, execute = runNode) {
  const resultFile = path.join(paths.output, "result.json"), beforeStat = fs.lstatSync(resultFile, {throwIfNoEntry: false});
  if (!beforeStat?.isFile() || beforeStat.isSymbolicLink() || fs.realpathSync(resultFile) !== resultFile) {
    throw new Error("Restored Linux LLVM admission requires an existing canonical result receipt");
  }
  const beforeHash = fileHash(resultFile);
  let commandError;
  try {
    execute("scripts/prepare-native-ci.js", ["stage", "linux-llvm-admission"], paths.env, linuxLlvmAdmissionTimeoutMs);
  } catch (error) {commandError = error;}
  const afterStat = fs.lstatSync(resultFile, {throwIfNoEntry: false});
  if (!afterStat?.isFile() || afterStat.isSymbolicLink() || fs.realpathSync(resultFile) !== resultFile ||
      afterStat.dev !== beforeStat.dev || afterStat.ino !== beforeStat.ino || afterStat.size !== beforeStat.size ||
      afterStat.mtimeMs !== beforeStat.mtimeMs || afterStat.ctimeMs !== beforeStat.ctimeMs || fileHash(resultFile) !== beforeHash) {
    throw new Error("Read-only Linux LLVM admission changed its existing result receipt");
  }
  if (commandError) {throw commandError;}
  return {resultSha256: beforeHash};
}

/** Admit either the legacy full LLVM producer tree or a pinned installed-SDK handoff.
 * A handoff marker or an explicitly supplied pin selects the new path; neither can fall back
 * to the legacy verifier if its external pin is absent or malformed.
 * @param {{output: string, suffix: string, env: NodeJS.ProcessEnv}} paths */
function admitLinuxLlvmOutput(paths) {
  const handoffFile = path.join(paths.output, "handoff.json"), handoffStat = fs.lstatSync(handoffFile, {throwIfNoEntry: false}),
    handoffSha256 = paths.env["MOM_NATIVE_LINUX_LLVM_SDK_HANDOFF_SHA256"];
  if (handoffStat || handoffSha256 !== undefined) {
    if (typeof handoffSha256 !== "string" || !/^[a-f0-9]{64}$/.test(handoffSha256) ||
        !/^[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(paths.suffix)) {
      throw new Error("Installed Linux LLVM SDK admission requires its exact run attempt and external handoff pin");
    }
    return require("./native-linux-llvm-sdk-handoff").linuxLlvmSdkHandoffInputs(paths.output,
      {runAttempt: paths.suffix, handoffSha256});
  }
  return require("./build-native-llvm21-sdk").verifyLinuxLlvmAdmission(paths.output);
}

/** @param {string} value */
function jsonHash(value) {return createHash("sha256").update(value).digest("hex");}

/** @param {unknown} value @param {string[]} keys */
function hasExactKeys(value, keys) {
  return !!value && typeof value === "object" && !Array.isArray(value) &&
    Object.keys(value).sort().join("\0") === [...keys].sort().join("\0");
}

/** CI producers and their same-run consumers share the workflow's validated build parallelism.
 * @param {NodeJS.ProcessEnv} [environment] */
function ciBuildJobs(environment = process.env) {
  const jobs = environment["MOM_BUILD_JOBS"] || "";
  if (!["2", "4"].includes(jobs)) {throw new Error("Native CI needs MOM_BUILD_JOBS set to 2 or 4");}
  return jobs;
}

/** @returns {{suffix: string, source: string, sourceName: string, ioSource: string, ioSourceName: string,
 * zstdSource: string, zstdSourceName: string, zstdSdkRoot: string, zstdSdkName: string,
 * linuxJitResourcesRoot: string, linuxJitResourcesName: string,
 * linuxSdkDescriptorRoot: string, linuxSdkDescriptorName: string,
 * linuxSource: string, linuxSourceName: string,
 * sdkRoot: string, sdkName: string, output: string, ioSdkRoot: string, ioSdkName: string,
 * ioSdk: string, opensslSdk: string, opensslTools: string, rocmComgrSdkRoot: string, rocmResourcesRoot: string,
 * rocmHipHeadersRoot: string, rocmHostSdkRoot: string, runnerTemp: string, env: NodeJS.ProcessEnv}} */
function ciPaths() {
  const run = process.env["GITHUB_RUN_ID"] || "", attempt = process.env["GITHUB_RUN_ATTEMPT"] || "", runnerTemp = process.env["RUNNER_TEMP"] || "";
  if (!/^[0-9]{1,20}$/.test(run) || !/^[1-9][0-9]{0,5}$/.test(attempt) || !path.isAbsolute(runnerTemp) ||
      process.platform !== "linux" || process.arch !== "x64") {throw new Error("Native Linux SDK CI needs GitHub run identity and a Linux x64 runner");}
  const suffix = run + "-" + attempt, sourceName = "mom-native-prerequisites-" + suffix, sdkName = "mom-native-llvm21-sdk-" + suffix,
    ioSourceName = "mom-native-io-prerequisites-" + suffix, ioSdkName = "mom-native-io-openssl-sdk-" + suffix,
    linuxSourceName = "mom-native-linux-source-prerequisites-" + suffix,
    zstdSourceName = "mom-native-zstd-prerequisites-" + suffix, zstdSdkName = "mom-native-zstd-sdk-" + suffix,
    linuxJitResourcesName = "mom-native-linux-jit-resources-" + suffix,
    linuxSdkDescriptorName = "mom-native-linux-sdk-descriptor-" + suffix;
  const source = path.join(temp, sourceName), sdkRoot = path.join(temp, sdkName), output = path.join(sdkRoot, "linux-llvm21-sdk"),
    ioSource = path.join(temp, ioSourceName), ioSdkRoot = path.join(temp, ioSdkName),
    linuxSource = path.join(temp, linuxSourceName),
    ioSdk = path.join(ioSdkRoot, "io-sdk"), opensslSdk = path.join(ioSdkRoot, "openssl-sdk"),
    opensslTools = path.join(ioSdkRoot, "openssl-tools.json"), zstdSource = path.join(temp, zstdSourceName),
    zstdSdkRoot = path.join(temp, zstdSdkName), linuxJitResourcesRoot = path.join(temp, linuxJitResourcesName),
    rocmComgrSdkName = "mom-native-rocm-comgr-sdk-" + suffix,
    rocmResourcesName = "mom-native-rocm-resources-" + suffix, rocmHipHeadersName = "mom-native-rocm-hip-headers-" + suffix,
    rocmComgrSdkRoot = path.join(temp, rocmComgrSdkName), rocmResourcesRoot = path.join(temp, rocmResourcesName),
    rocmHipHeadersRoot = path.join(temp, rocmHipHeadersName), rocmHostSdkRoot = path.join(temp, "mom-native-rocm-host-sdk-" + suffix),
    linuxSdkDescriptorRoot = path.join(temp, linuxSdkDescriptorName);
  const jobs = process.env["MOM_BUILD_JOBS"];
  if (process.env["MOM_AOT_BUILD_ONLY"] !== "1" || !["2", "4"].includes(jobs || "") ||
      process.env["MOM_NATIVE_SOURCE_DIR"] !== source || process.env["MOM_NATIVE_LINUX_SDK_DIR"] !== sdkRoot ||
      process.env["MOM_NATIVE_IO_SOURCE_DIR"] !== ioSource || process.env["MOM_NATIVE_IO_SDK_DIR"] !== ioSdkRoot ||
      process.env["MOM_NATIVE_LINUX_SOURCES_DIR"] !== linuxSource ||
      process.env["MOM_NATIVE_ZSTD_SOURCE_DIR"] !== zstdSource || process.env["MOM_NATIVE_ZSTD_SDK_DIR"] !== zstdSdkRoot ||
      process.env["MOM_NATIVE_LINUX_JIT_RESOURCES_DIR"] !== linuxJitResourcesRoot ||
      process.env["MOM_NATIVE_ROCM_COMGR_SDK_DIR"] !== rocmComgrSdkRoot ||
      process.env["MOM_NATIVE_ROCM_RESOURCES_DIR"] !== rocmResourcesRoot ||
      process.env["MOM_NATIVE_ROCM_HIP_HEADERS_DIR"] !== rocmHipHeadersRoot ||
      process.env["MOM_NATIVE_ROCM_HOST_SDK_DIR"] !== rocmHostSdkRoot ||
      process.env["MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR"] !== linuxSdkDescriptorRoot) {
    throw new Error("Native CI paths, bounded jobs, or build-only mode differ from the run plan");
  }
  return {suffix, source, sourceName, ioSource, ioSourceName, linuxSource, linuxSourceName, zstdSource, zstdSourceName, zstdSdkRoot, zstdSdkName,
    sdkRoot, sdkName, output, ioSdkRoot, ioSdkName, ioSdk, opensslSdk, opensslTools,
    linuxJitResourcesRoot, linuxJitResourcesName,
    rocmComgrSdkRoot, rocmResourcesRoot, rocmHipHeadersRoot, rocmHostSdkRoot, linuxSdkDescriptorRoot, linuxSdkDescriptorName,
    runnerTemp: fs.realpathSync(runnerTemp),
    env: {...process.env, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: jobs, OMP_NUM_THREADS: jobs}};
}

/** Export only source-built host tools needed by Windows LLVM, not the Linux build tree.
 * Full SDK admission runs on this producer; the consumer admits an explicitly selected subset.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildLinuxLlvmToolsCi(paths) {
  const name = "native-linux-llvm-tools-" + paths.suffix,
    workRoot = path.join(paths.runnerTemp, "mom-native-linux-llvm-tools-" + paths.suffix),
    bundle = path.join(workRoot, name), archive = path.join(paths.runnerTemp, name + ".tar"),
    {exportLinuxLlvmTools, linuxLlvmToolHandoffInputs} = require("./native-linux-llvm-tools");
  for (const file of [workRoot, archive]) {
    if (fs.lstatSync(file, {throwIfNoEntry: false})) {throw new Error("Linux LLVM tool export refuses an existing same-run output");}
  }
  fs.mkdirSync(workRoot, {mode: 0o700});
  exportLinuxLlvmTools(paths.output, bundle, {runAttempt: paths.suffix});
  const handoffSha256 = fileHash(path.join(bundle, "handoff.json")),
    expectation = {runAttempt: paths.suffix, handoffSha256};
  linuxLlvmToolHandoffInputs(bundle, expectation);
  const packed = packState(archive, workRoot, [name], 5 * 60 * 1000), members = listState(archive, [name]);
  if (packed.bytes > 1024 ** 3 || tarExpandedBytes(archive, members.length) > 1024 ** 3) {
    throw new Error("Linux LLVM tool export exceeded its one-GiB transport bound");
  }
  linuxLlvmToolHandoffInputs(bundle, expectation);
  const values = {tools_bundle_sha256: fileHash(archive), tools_handoff_sha256: handoffSha256};
  appendWorkflowOutputs(paths.env, values);
  fs.appendFileSync(paths.env["GITHUB_STEP_SUMMARY"] || "", "### Selected Linux LLVM host tools\n" +
    "- The fully admitted source-built SDK supplied only Windows LLVM's public host-tool closure, the locked source archive, and unchanged producer receipts.\n" +
    "- The consumer verifies this hash-bound same-run subset; it does not re-admit or receive the complete Linux SDK/build tree. No application or runtime qualification is claimed.\n");
  return {...values, bytes: packed.bytes};
}

/** Export the public installed LLVM prefix and unchanged admission receipts for Linux consumers.
 * The full source/build/SDK state artifact remains separate for Windows host tools.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildLinuxLlvmSdkHandoff(paths) {
  const name = "native-linux-llvm21-installed-sdk-" + paths.suffix,
    workRoot = path.join(paths.runnerTemp, "llvm-installed-sdk-export-" + paths.suffix),
    bundle = path.join(workRoot, paths.sdkName, "linux-llvm21-sdk"),
    archive = path.join(paths.runnerTemp, name + ".tar"),
    {exportLinuxLlvmSdk, linuxLlvmSdkHandoffInputs} = require("./native-linux-llvm-sdk-handoff");
  for (const item of [workRoot, archive]) {
    if (fs.lstatSync(item, {throwIfNoEntry: false})) {throw new Error("Linux installed LLVM SDK handoff refuses an existing same-run output");}
  }
  fs.mkdirSync(path.join(workRoot, paths.sdkName), {recursive: true, mode: 0o700});
  const produced = exportLinuxLlvmSdk(paths.output, bundle, {runAttempt: paths.suffix}),
    expectation = {runAttempt: paths.suffix, handoffSha256: produced.handoffSha256};
  linuxLlvmSdkHandoffInputs(bundle, expectation);
  const packed = packState(archive, workRoot, [paths.sdkName]), members = listState(archive, [paths.sdkName]);
  if (packed.bytes > 16 * 1024 ** 3 || tarExpandedBytes(archive, members.length) > 16 * 1024 ** 3) {
    throw new Error("Linux installed LLVM SDK handoff exceeded its transport bound");
  }
  linuxLlvmSdkHandoffInputs(bundle, expectation);
  const values = {llvm_sdk_handoff_sha256: produced.handoffSha256, bundle_sha256: fileHash(archive)};
  appendWorkflowOutputs(paths.env, values);
  if (paths.env["GITHUB_STEP_SUMMARY"]) {
    fs.appendFileSync(paths.env["GITHUB_STEP_SUMMARY"], "- Linux consumers receive a separate installed-prefix handoff, externally pinned by its unchanged manifest; the full source/build/SDK state remains available to Windows host-tool composition.\n");
  }
  return {...values, bytes: packed.bytes, bundle};
}

/** Isolated source and SDK roots for the same-run static zstd prerequisite branch.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp" | "zstdSource" | "zstdSourceName" | "zstdSdkRoot" | "zstdSdkName">} paths */
function zstdStatePaths(paths) {
  const sourceTar = path.join(paths.runnerTemp, "native-zstd-state-" + paths.suffix + "-sources.tar");
  return {source: paths.zstdSource, sourceName: paths.zstdSourceName, sdkRoot: paths.zstdSdkRoot, sdkName: paths.zstdSdkName,
    archive: path.join(paths.zstdSource, "zstd-" + zstdPin.version + ".tar.gz"), sourceTar,
    incomingTar: path.join(paths.runnerTemp, "incoming", path.basename(sourceTar)),
    sdkTar: path.join(paths.runnerTemp, "native-linux-zstd-sdk-" + paths.suffix + ".tar"),
    toolsFile: path.join(temp, paths.zstdSdkName + "-tools.json")};
}

/** The OpenMP SDK reuses the exact same-run LLVM/OpenSSL source tar and original archive paths.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "source" | "sourceName" | "runnerTemp">} paths
 * @param {NodeJS.ProcessEnv} [environment] */
function openmpStatePaths(paths, environment = process.env) {
  const sdkName = "mom-native-openmp-sdk-" + paths.suffix, sdkRoot = path.join(temp, sdkName),
    sourceTar = path.join(paths.runnerTemp, stateTarName(paths, "sources")),
    toolsFile = path.join(paths.runnerTemp, "native-openmp-tools-" + paths.suffix + ".json");
  if (environment["MOM_NATIVE_OPENMP_SDK_DIR"] !== sdkRoot || !path.isAbsolute(paths.source) ||
      !path.isAbsolute(paths.runnerTemp) || fs.realpathSync(paths.runnerTemp) !== paths.runnerTemp) {
    throw new Error("Native OpenMP SDK root must exactly match this workflow run");
  }
  return {source: paths.source, sourceName: paths.sourceName, archive: path.join(paths.source, llvmArchiveName),
    sdkRoot, sdkName, sourceTar, incomingTar: path.join(paths.runnerTemp, "incoming", path.basename(sourceTar)),
    sdkTar: path.join(paths.runnerTemp, "native-linux-openmp-sdk-" + paths.suffix + ".tar"), toolsFile};
}

/** Restore the exact same-run dependencies for the source-composed static ACPP CUDA/HIP SDK.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp" | "source" | "sourceName" | "ioSourceName" | "sdkRoot" | "sdkName" | "output">} paths
 * @param {NodeJS.ProcessEnv} [environment] */
function acppCudaHipSdkStatePaths(paths, environment = process.env) {
  const acpp = acppStatePaths(paths, environment), openmp = openmpStatePaths(paths, environment), rocm = rocmHostStatePaths(paths, environment),
    cudaInputs = dpcppCudaInputsStatePaths(paths, environment),
    sdkName = "mom-native-acpp-cuda-hip-sdk-" + paths.suffix, sdkRoot = path.join(temp, sdkName),
    sdkTar = path.join(paths.runnerTemp, "native-linux-acpp-cuda-hip-sdk-" + paths.suffix + ".tar");
  if (environment["MOM_NATIVE_ACPP_CUDA_HIP_SDK_DIR"] !== sdkRoot) {
    throw new Error("Native ACPP CUDA/HIP SDK output root must exactly match this workflow run");
  }
  const rocmRoots = [rocm.sourceName, rocm.llvmSourceName, rocm.llvmSdkName, rocm.hipHeadersName,
    rocm.resourcesName, rocm.sdkName, rocm.hostSdkName];
  return {acpp, openmp, rocm, cudaInputs, sdkName, sdkRoot, sdkTar, rocmRoots,
    incomingAcppTar: acpp.incomingTar, incomingOpenmpTar: path.join(paths.runnerTemp, "incoming", path.basename(openmp.sdkTar)),
    incomingRocmHostTar: path.join(paths.runnerTemp, "incoming", path.basename(rocm.hostSdkTar)),
    incomingLinuxTar: path.join(paths.runnerTemp, "incoming", path.basename(cudaInputs.linux.sourceTar)),
    incomingCudaInputsTar: path.join(paths.runnerTemp, "incoming", path.basename(cudaInputs.sdkTar)),
    llvmSdk: rocm.llvmSdk, rocmSdk: path.join(rocm.runtimeOutput, "sdk"),
    comgrReceipt: path.join(rocm.sdkRoot, "result.json"), deviceRoot: path.join(rocm.resourcesRoot, "install"),
    deviceBitcode: path.join(rocm.resourcesRoot, "install/amdgcn/bitcode"), openmpLibrary: path.join(openmp.sdkRoot, "sdk/libomp.a")};
}

/** @param {ReturnType<typeof acppCudaHipSdkStatePaths>} state @param {{cc: string, cxx: string}} tools */
function acppCudaHipSdkBuildCommand(state, tools) {
  return {script: "scripts/build-native-sycl-sdk.js", args: [state.acpp.preparedCheckout, state.llvmSdk,
    state.cudaInputs.cudaRoot, state.sdkRoot, tools.cc, tools.cxx]};
}

/** Keep the release composer’s combined-backend and exact same-run input contract explicit.
 * @param {ReturnType<typeof acppCudaHipSdkStatePaths>} state
 * @param {{hostCompilers?: {cc?: string, cxx?: string}, inputs?: {comgr?: {receipt?: string}}, fingerprints?: Record<string, string>}} record
 * @param {{libraries: string[], backend?: string, cudaVersion?: number, source?: string, build?: string, cxx?: string,
 * comgr?: {receipt?: string, roots?: {llvm?: string, lld?: string, device?: string}}, openmp?: {archive?: string}, files?: Record<string, string>}} sdk
 * @param {{cc: string, cxx: string}} tools
 * @param {{cuda?: {root?: string, headers?: {files?: Record<string, string>}, staticRuntime?: {cudart?: {path?: string, sha256?: string}, culibos?: {path?: string, sha256?: string}}}}} cudaInputs */
function assertAcppCudaHipSdk(state, record, sdk, tools, cudaInputs) {
  const cudaLibraries = cudaInputs.cuda?.staticRuntime,
    cudart = path.join(state.cudaInputs.cudaRoot, "lib64/libcudart_static.a"), culibos = path.join(state.cudaInputs.cudaRoot, "lib64/libculibos.a"),
    cudaHeaders = cudaInputs.cuda?.headers?.files;
  // The SDK receipt's cxx is LLVM clang++ for device compilation; tools stays the oneAPI host pair.
  if (!cudaLibraries || !cudaHeaders || cudaInputs.cuda?.root !== state.cudaInputs.cudaRoot ||
      ["cuda.h", "cuda_runtime_api.h"].some(name => !/^[0-9a-f]{64}$/.test(cudaHeaders[name] || "")) ||
      cudaLibraries.cudart?.path !== cudart || cudaLibraries.culibos?.path !== culibos ||
      !/^[0-9a-f]{64}$/.test(cudaLibraries.cudart.sha256 || "") || !/^[0-9a-f]{64}$/.test(cudaLibraries.culibos.sha256 || "") ||
      sdk.backend !== "cuda-hip" || sdk.cudaVersion !== 12060 || sdk.source !== state.acpp.preparedCheckout ||
      sdk.build !== path.join(state.sdkRoot, "build") || sdk.cxx !== path.join(state.llvmSdk, "bin/clang++") ||
      record.hostCompilers?.cc !== tools.cc || record.hostCompilers?.cxx !== tools.cxx ||
      record.inputs?.comgr?.receipt !== state.comgrReceipt || sdk.comgr?.roots?.llvm !== state.llvmSdk ||
      sdk.comgr?.receipt !== state.comgrReceipt || sdk.comgr?.roots?.lld !== state.llvmSdk || sdk.comgr?.roots?.device !== state.deviceRoot ||
      sdk.openmp?.archive !== state.openmpLibrary ||
      record.fingerprints?.[path.join(state.cudaInputs.cudaRoot, "include/cuda.h")] !== cudaHeaders["cuda.h"] ||
      record.fingerprints?.[path.join(state.cudaInputs.cudaRoot, "include/cuda_runtime_api.h")] !== cudaHeaders["cuda_runtime_api.h"] ||
      record.fingerprints?.[cudart] !== cudaLibraries.cudart.sha256 || record.fingerprints?.[culibos] !== cudaLibraries.culibos.sha256 ||
      sdk.files?.[cudart] !== cudaLibraries.cudart.sha256 || sdk.files?.[culibos] !== cudaLibraries.culibos.sha256) {
    throw new Error("Public ACPP CUDA/HIP SDK admission differs from the exact same-run CI inputs");
  }
  return sdk;
}

/** Admit the public combined SDK and every same-run CUDA, LLVM/ROCm/COMGR and OpenMP input.
 * @param {ReturnType<typeof acppCudaHipSdkStatePaths>} state @param {{cc: string, cxx: string}} tools
 * @param {string} [expectedDpcppManifestSha256] */
function auditAcppCudaHipSdk(state, tools, expectedDpcppManifestSha256) {
  const receiptFile = path.join(state.sdkRoot, "result.json"), record = readBuildJson(receiptFile),
    openmp = nativeOpenmpInputs(state.openmp.sdkRoot), comgr = comgrSdkInputs(state.comgrReceipt),
    cudaInputs = auditDpcppCudaInputs(state.cudaInputs, readBuildJson(state.cudaInputs.manifest), expectedDpcppManifestSha256),
    hostRecord = readBuildJson(path.join(state.rocm.runtimeOutput, "result.json"));
  comgrDeviceInputs(state.deviceRoot);
  if (openmp.archive !== state.openmpLibrary || comgr.roots.llvm !== state.llvmSdk || comgr.roots.lld !== state.llvmSdk ||
      comgr.roots.device !== state.deviceRoot || hostRecord.version !== 1 || hostRecord.buildOnly !== true ||
      hostRecord.executed !== false || hostRecord.publicationQualified !== false || hostRecord.prefix !== state.rocmSdk ||
      hostRecord.comgr?.receipt !== state.comgrReceipt || hostRecord.comgr?.roots?.llvm !== state.llvmSdk ||
      hostRecord.comgr?.roots?.lld !== state.llvmSdk || hostRecord.comgr?.roots?.device !== state.deviceRoot) {
    throw new Error("ACPP CUDA/HIP inputs do not bind the same-run LLVM, CUDA, ROCm, COMGR, resources and OpenMP roots");
  }
  const sdk = nativeStaticRuntimeInputs(receiptFile, {openmpSdkRoot: state.openmp.sdkRoot});
  return assertAcppCudaHipSdk(state, record, sdk, tools, cudaInputs);
}

/** Restore same-run source/SDK roots, build combined CUDA/HIP ACPP offline, then admit and retain its new SDK.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildAcppCudaHipSdk(paths) {
  const state = acppCudaHipSdkStatePaths(paths, paths.env), git = canonicalExecutable("/usr/bin/git");
  if (fs.existsSync(state.sdkRoot) || fs.existsSync(state.sdkTar)) {throw new Error("Native ACPP CUDA/HIP stage refuses an existing output");}
  const externalDpcppPins = dpcppCudaInputPins(paths.env);
  // When supplied by CI, authenticate transport before extraction; preserve the legacy unpinned path.
  if (externalDpcppPins) {dpcppSourceArchivePin(state.incomingCudaInputsTar, externalDpcppPins.bundleSha256);}
  extractState(state.incomingAcppTar, temp, [state.acpp.sourceName]);
  verifyAcppSource(paths, state.acpp, git);
  extractState(state.incomingLinuxTar, temp, [state.cudaInputs.linux.sourceName]);
  verifyLinuxSources(paths, state.cudaInputs.linux, git);
  extractState(state.incomingCudaInputsTar, temp, [state.cudaInputs.name]);
  if (externalDpcppPins) {dpcppSourceArchivePin(state.incomingCudaInputsTar, externalDpcppPins.bundleSha256);}
  auditDpcppCudaInputs(state.cudaInputs, readBuildJson(state.cudaInputs.manifest), externalDpcppPins?.manifestSha256);
  extractState(state.incomingRocmHostTar, temp, state.rocmRoots);
  // OpenMP's historical handoff also carries the shared LLVM source root. That root is already
  // restored and admitted from the ROCm handoff, so validate the complete tar but restore only libomp.
  extractState(state.incomingOpenmpTar, temp, [paths.sourceName, state.openmp.sdkName], [state.openmp.sdkName]);
  verifyRocmSources(paths, state.rocm, git);
  const sourceEnv = {...sdkSourceEnvironment(paths.env), GIT_ALLOW_PROTOCOL: "file", GIT_NO_LAZY_FETCH: "1",
    GIT_TERMINAL_PROMPT: "0", GIT_OPTIONAL_LOCKS: "0"};
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", state.rocm.llvmSource], sourceEnv, 2 * 60 * 1000);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-acpp-cuda-hip-" + paths.suffix, 35 * 60 * 1000);
  nativeOpenmpInputs(state.openmp.sdkRoot); comgrSdkInputs(state.comgrReceipt); comgrDeviceInputs(state.deviceRoot);
  verifyLinuxLlvmAdmissionCi(paths);
  for (const incoming of [state.incomingAcppTar, state.incomingOpenmpTar, state.incomingRocmHostTar,
    state.incomingLinuxTar, state.incomingCudaInputsTar]) {fs.unlinkSync(incoming);}
  const tools = {cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx")};
  const command = acppCudaHipSdkBuildCommand(state, tools), buildEnv = {...sourceEnv, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: ciBuildJobs(paths.env),
    MOM_AOT_SYCL_BACKEND: "cuda-hip", MOM_AOT_ROCM_STATIC_ROOT: state.rocmSdk, MOM_AOT_LLD_ROOT: state.llvmSdk,
    ROCM_DEVICE_LIBS_PATH: state.deviceBitcode, MOM_AOT_COMGR_RECEIPT: state.comgrReceipt, MOM_AOT_OPENMP_LIBRARY: state.openmpLibrary};
  runNode(command.script, command.args, buildEnv, 45 * 60 * 1000);
  const admitted = auditAcppCudaHipSdk(state, tools, externalDpcppPins?.manifestSha256);
  const evidenceParent = fs.realpathSync(paths.runnerTemp), sdkRoot = fs.realpathSync(state.sdkRoot);
  /** @param {string} root @param {string} child */
  const contains = (root, child) => {
    const relative = path.relative(root, child);
    return relative === "" || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep));
  };
  if (contains(sdkRoot, evidenceParent)) {
    throw new Error("ACPP CUDA/HIP regression evidence must stay outside the packaged SDK");
  }
  const evidence = fs.mkdtempSync(path.join(evidenceParent, "native-acpp-cuda-hip-sycl-proof-")),
    probeEnv = {...buildEnv, CXX: tools.cxx, LLVM_CONFIG: path.join(state.llvmSdk, "bin/llvm-config")},
    {source, build, cxx, flags} = staticSyclInputs(state.acpp.preparedCheckout, path.join(state.sdkRoot, "build"), evidence, "hip", probeEnv),
    cacheIdentity = runJitCacheIdentity(tools.cxx, flags, source, build, evidence, repo),
    hcfObjectRepro = runHcfRepro(cxx, flags, source, build, evidence, repo);
  fs.writeFileSync(path.join(evidence, "native-ci-regressions.json"), JSON.stringify({cacheIdentity, hcfObjectRepro}, null, 2) + "\n", {flag: "wx"});
  packState(state.sdkTar, temp, [state.sdkName]); listState(state.sdkTar, [state.sdkName]);
  console.log("[native-ci] ACPP CUDA/HIP SDK admitted: " + admitted.libraries.length + " libraries; HIP HCF and CPU-only JIT-cache proofs passed, no GPU execution or publication qualification. Evidence: " + evidence);
}

/** NVRTC verification keeps both vendor archives and assembled libraries local to one job.
 * Only compact receipts and the Linux CPU proof outputs cross the artifact boundary.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp"> & {env?: NodeJS.ProcessEnv}} paths
 * @param {"linux" | "windows"} platform @param {NodeJS.ProcessEnv} [environment] */
function nvrtcStatePaths(paths, platform, environment = paths.env || process.env) {
  if (platform !== "linux" && platform !== "windows") {throw new Error("Native NVRTC CI target must be linux or windows");}
  const suffix = paths.suffix, key = platform.toUpperCase(), sourceName = `mom-native-nvrtc-${platform}-prerequisites-${suffix}`,
    sdkName = `mom-native-nvrtc-${platform}-sdk-${suffix}`, evidenceName = `mom-native-nvrtc-${platform}-verification-${suffix}`,
    source = path.join(temp, sourceName), sdkRoot = path.join(temp, sdkName), evidence = path.join(temp, evidenceName),
    proofParent = path.join(temp, "mom-native-nvrtc-linux-proof-" + suffix),
    verificationTar = path.join(paths.runnerTemp, `native-nvrtc-${platform}-verification-${suffix}.tar`);
  if (environment[`MOM_NATIVE_NVRTC_${key}_SOURCE_DIR`] !== source || environment[`MOM_NATIVE_NVRTC_${key}_SDK_DIR`] !== sdkRoot) {
    throw new Error(`Native NVRTC ${platform} roots must exactly match this workflow run`);
  }
  return {platform, source, sourceName, sdkRoot, sdkName,
    evidence, evidenceName, proofParent, verificationTar};
}

/** Same-run build-only CUDA image inputs and the admitted image/license-source handoff.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths @param {NodeJS.ProcessEnv} [environment] */
function cudaImagesStatePaths(paths, environment = process.env) {
  const linux = nvrtcStatePaths(paths, "linux", environment), dpcpp = dpcppCudaInputsStatePaths(paths, environment),
    source126Name = `mom-native-nvrtc-linux-12.6-prerequisites-${paths.suffix}`,
    sdk126Name = `mom-native-nvrtc-linux-12.6-sdk-${paths.suffix}`,
    source126 = path.join(temp, source126Name), sdk126 = path.join(temp, sdk126Name),
    name = `mom-native-cuda-images-${paths.suffix}`, root = path.join(temp, name),
    headerRoot = path.join(root, "headers"), cutlassRoot = path.join(root, "cutlass"),
    cutlassLicense = path.join(cutlassRoot, "LICENSE.txt"),
    output = path.join(root, "bundle"), config = path.join(root, "worker-config.json"),
    handoffName = `mom-native-cuda-images-handoff-${paths.suffix}`, handoffRoot = path.join(temp, handoffName),
    noticeRoot = path.join(handoffRoot, "notice-sources"), noticeManifest = path.join(noticeRoot, "manifest.json"),
    tar = path.join(paths.runnerTemp, `native-linux-cuda-images-${paths.suffix}.tar`);
  if (environment["MOM_NATIVE_NVRTC_LINUX_126_SOURCE_DIR"] !== source126 ||
      environment["MOM_NATIVE_NVRTC_LINUX_126_SDK_DIR"] !== sdk126 || environment["MOM_NATIVE_CUDA_IMAGES_DIR"] !== root ||
      !path.isAbsolute(paths.runnerTemp) || fs.realpathSync(paths.runnerTemp) !== paths.runnerTemp) {
    throw new Error("Native CUDA image inputs must exactly match this run's source, SDK and output roots");
  }
  return {linux, source126, source126Name, sdk126, sdk126Name, dpcpp, name, root, headerRoot,
    cutlassRoot, cutlassInclude: path.join(cutlassRoot, "include"), cutlassMarker: path.join(cutlassRoot, ".mom-version"),
    cutlassLicense, noticeRoot, noticeManifest,
    header126: path.join(headerRoot, "cuda-12.6.inc"), header133: path.join(headerRoot, "cuda-13.3.inc"),
    output, config, result: path.join(output, "result.json"), handoffName, handoffRoot, tar,
    incomingDpcppTar: path.join(paths.runnerTemp, "incoming", path.basename(dpcpp.sdkTar))};
}

/** Bind a worker to the exact NVRTC cohort, same-run CUDA 12.6 header root, licenses and compiler.
 * The public image builder then compiles actual kernels with that NVRTC/header pairing.
 * @param {ReturnType<typeof nvrtcSdkInputs>} admitted @param {{cohort: "linux" | "linux-12.6", headerPack: string,
 * cudaInclude: string, cutlassInclude: string, cutlassMarker: string, cutlassLicense: string,
 * cudaCopyrightLicenses: string[], cudaCopyrightManifest: string, compiler: string,
 * compilerIdentity: string, installer: string}} input */
function cudaImageWorkerRecipe(admitted, input) {
  const receipt = admitted.receipt, pins = nvrtcCohortInfo("linux", input.cohort), sdkRoot = receipt.output,
    sdkInclude = path.join(sdkRoot, "include"), sdkLib = path.join(sdkRoot, "lib64");
  if (receipt.platform !== "linux" || receipt.cohort !== input.cohort || receipt.vendorVersion !== pins.version ||
      receipt.vendorPrebuilt !== true || receipt.sourceRebuilt !== false || receipt.compiled !== false ||
      receipt.publicationQualified !== false || receipt.redistributionQualified !== false ||
      !admitted.files || !/^[a-f0-9]{64}$/.test(admitted.receiptSha256 || "") ||
      admitted.files[admitted.receiptFile] !== admitted.receiptSha256) {
    throw new Error("CUDA image worker needs the exact admitted Linux NVRTC cohort and build-only receipt");
  }
  const licenses = [path.join(sdkRoot, "licenses/nvrtc/LICENSE"),
    ...(input.cohort === "linux" ? [path.join(sdkRoot, "licenses/nvptxcompiler/LICENSE")] : [])];
  if (licenses.some(file => !admitted.files[file])) {throw new Error("CUDA image worker inputs must retain every admitted NVRTC license hash");}
  for (const [file, hash] of Object.entries(admitted.files)) {
    if (!/^[a-f0-9]{64}$/.test(hash) || fileHash(file) !== hash) {throw new Error("CUDA image SDK closure changed before worker configuration: " + file);}
  }
  const header = cudaHeaderInputs(input.headerPack), headerRoots = header.manifest["roots"],
    cudaInclude = fs.realpathSync(input.cudaInclude), cutlassInclude = fs.realpathSync(input.cutlassInclude),
    expectedRoots = new Set([cudaInclude, cutlassInclude]), nestedCccl = path.join(cudaInclude, "cccl");
  if (fs.existsSync(path.join(nestedCccl, "cuda/std/cstdint"))) {expectedRoots.add(fs.realpathSync(nestedCccl));}
  if (!Array.isArray(headerRoots) || headerRoots.length !== expectedRoots.size ||
      headerRoots.some(root => typeof root !== "string" || !expectedRoots.has(root)) || headerRoots.includes(fs.realpathSync(sdkInclude))) {
    throw new Error("CUDA image header pack must bind only the same-run CUDA 12.6 and pinned CUTLASS roots, not NVRTC-only headers");
  }
  const extraInputs = [input.compiler, input.compilerIdentity, input.cutlassMarker, input.cutlassLicense,
    ...input.cudaCopyrightLicenses, input.cudaCopyrightManifest, input.installer];
  /** @type {Record<string, string>} */ const inputs = {...admitted.files};
  for (const file of extraInputs) {inputs[fs.realpathSync(file)] = fileHash(file);}
  return {headerPack: input.headerPack, worker: {command: fs.realpathSync(input.compiler),
    args: ["-std=c++20", "-O2", "-I" + sdkInclude, "{source}", "-L" + sdkLib,
      "-Wl,-rpath," + sdkLib, "-lnvrtc", "-pthread", "-o", "{output}"],
    inputs, environment: {LD_LIBRARY_PATH: sdkLib}}};
}

/** @typedef {{path:string,sha256:string}} CudaNoticeSourceReference */
/** @typedef {{version:string,url:string,sha256:string,member:string,installer:CudaNoticeSourceReference,
 * archiveMarker:CudaNoticeSourceReference}} CudaNoticeArchiveSource */
/** @typedef {{component:string,version:string,sourcePath:string,file:string,sha256:string,bytes:number,
 * sourceReceipt:CudaNoticeSourceReference}} CudaNoticeSource */
/** @typedef {{format:"mom-cuda-notice-sources-v1",scope:"CUDA/NVRTC/CUTLASS source license bodies only",
 * completeNativeNotices:false,sourceArchive:CudaNoticeArchiveSource,sources:CudaNoticeSource[]}} CudaNoticeSourcePlan */

/** Bind exact CUDA/NVRTC/CUTLASS license bodies to the receipts that supplied them.
 * This subset is explicitly not a complete native-distribution notice set.
 * @param {Pick<ReturnType<typeof cudaImagesStatePaths>, "sdk126" | "cutlassMarker" | "cutlassLicense"> &
 * {dpcpp:{root:string,manifest:string},linux:{sdkRoot:string}}} state
 * @param {{cuda:{licenses:Array<{package:string,path:string,sha256:string,version:string}>,packages:Record<string,string>}}} dpcpp
 * @param {ReturnType<typeof nvrtcSdkInputs>} baseline @param {ReturnType<typeof nvrtcSdkInputs>} modern
 * @returns {CudaNoticeSourcePlan} */
function cudaNoticeSourcePlan(state, dpcpp, baseline, modern) {
  const installer = path.join(repo, "scripts/install-cutlass.sh"), installerText = fs.readFileSync(installer, "utf8"),
    archiveUrl = `https://github.com/NVIDIA/cutlass/archive/refs/tags/${cutlassVersion}.tar.gz`;
  if (!installerText.includes(`MOM_CUTLASS_VERSION=${cutlassVersion}`) ||
      !installerText.includes(`MOM_CUTLASS_SHA256=${cutlassArchiveSha256}`) ||
      !installerText.includes(`MOM_CUTLASS_LICENSE_SHA256=${cutlassLicenseSha256}`) ||
      !/MOM_CUTLASS_URL="https:\/\/github\.com\/NVIDIA\/cutlass\/archive\/refs\/tags\/\$\{MOM_CUTLASS_VERSION\}\.tar\.gz"/.test(installerText) ||
      fs.readFileSync(state.cutlassMarker, "utf8").trim() !== cutlassArchiveSha256 ||
      fileHash(state.cutlassLicense) !== cutlassLicenseSha256 || baseline.receipt.cohort !== "linux-12.6" ||
      modern.receipt.cohort !== "linux" || dpcpp.cuda.licenses.length !== cudaPackageNames.length ||
      dpcpp.cuda.licenses.some((row, index) => row.package !== cudaPackageNames[index] ||
        row.version !== dpcpp.cuda.packages[row.package] || row.path !== path.join(state.dpcpp.root, "licenses", row.package, "copyright"))) {
    throw new Error("CUDA license sources do not match the pinned same-run package, NVRTC and CUTLASS inputs");
  }
  /** @param {string} component @param {string} version @param {string} sourcePath @param {string} file
   * @param {string} sha256 @param {{path:string,sha256:string}} sourceReceipt */
  const source = (component, version, sourcePath, file, sha256, sourceReceipt) => {
    const stat = fs.lstatSync(sourcePath);
    if (!stat.isFile() || stat.isSymbolicLink() || fs.realpathSync(sourcePath) !== sourcePath || stat.size < 1 || stat.size > (2 << 20)) {
      throw new Error("CUDA notice source must be a bounded canonical regular file: " + file);
    }
    return {component, version, sourcePath, file, sha256, bytes: stat.size, sourceReceipt};
  };
  const baselineLicense = path.join(state.sdk126, "licenses/nvrtc/LICENSE"),
    modernNvrtcLicense = path.join(state.linux.sdkRoot, "licenses/nvrtc/LICENSE"),
    modernPtxLicense = path.join(state.linux.sdkRoot, "licenses/nvptxcompiler/LICENSE");
  if (baseline.files[baselineLicense] !== fileHash(baselineLicense) ||
      modern.files[modernNvrtcLicense] !== fileHash(modernNvrtcLicense) || modern.files[modernPtxLicense] !== fileHash(modernPtxLicense) ||
      dpcpp.cuda.licenses.some(row => row.sha256 !== fileHash(row.path))) {
    throw new Error("CUDA/NVRTC license body is not bound by its admitted producer receipt");
  }
  return {format: "mom-cuda-notice-sources-v1", scope: "CUDA/NVRTC/CUTLASS source license bodies only", completeNativeNotices: false,
    sourceArchive: {version: cutlassVersion.slice(1), url: archiveUrl, sha256: cutlassArchiveSha256,
      member: `cutlass-${cutlassVersion.slice(1)}/LICENSE.txt`, installer: {path: installer, sha256: fileHash(installer)},
      archiveMarker: {path: state.cutlassMarker, sha256: fileHash(state.cutlassMarker)}},
    sources: [
      ...dpcpp.cuda.licenses.map(row => source(row.package.replace(/-12-6$/, ""), row.version, row.path,
        cudaCopyrightName(row.package), row.sha256, {path: state.dpcpp.manifest, sha256: fileHash(state.dpcpp.manifest)})),
      source("nvrtc", baseline.receipt.vendorVersion, baselineLicense, "nvrtc-12.6-LICENSE.txt", baseline.files[baselineLicense],
        {path: baseline.receiptFile, sha256: baseline.receiptSha256}),
      source("nvrtc", modern.receipt.vendorVersion, modernNvrtcLicense, "nvrtc-13.3-LICENSE.txt", modern.files[modernNvrtcLicense],
        {path: modern.receiptFile, sha256: modern.receiptSha256}),
      source("nvptxcompiler", modern.receipt.vendorVersion, modernPtxLicense, "nvptxcompiler-13.3-LICENSE.txt", modern.files[modernPtxLicense],
        {path: modern.receiptFile, sha256: modern.receiptSha256}),
      source("cutlass", cutlassVersion.slice(1), state.cutlassLicense, "cutlass-4.6.1-LICENSE.txt", cutlassLicenseSha256,
        {path: state.cutlassMarker, sha256: fileHash(state.cutlassMarker)}),
    ]};
}

/** @param {Pick<ReturnType<typeof cudaImagesStatePaths>, "noticeRoot" | "noticeManifest">} state @param {CudaNoticeSourcePlan} plan */
function retainCudaNoticeSources(state, plan) {
  if (fs.lstatSync(state.noticeRoot, {throwIfNoEntry: false}) || fs.lstatSync(state.noticeManifest, {throwIfNoEntry: false}) ||
      plan.completeNativeNotices !== false || plan.sources.length !== 7) {throw new Error("CUDA notice-source handoff requires a fresh exact source bundle");}
  fs.mkdirSync(state.noticeRoot, {mode: 0o700});
  let totalBytes = 0;
  for (const row of plan.sources) {
    if (!safeRelative(row.file) || row.file.includes("/") || !path.isAbsolute(row.sourcePath) || !/^[a-f0-9]{64}$/.test(row.sha256) ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > (2 << 20) || !path.isAbsolute(row.sourceReceipt.path) ||
        !/^[a-f0-9]{64}$/.test(row.sourceReceipt.sha256)) {throw new Error("CUDA notice-source row is malformed or oversized");}
    const sourceStat = fs.lstatSync(row.sourcePath), receiptStat = fs.lstatSync(row.sourceReceipt.path);
    if (!sourceStat.isFile() || sourceStat.isSymbolicLink() || fs.realpathSync(row.sourcePath) !== row.sourcePath ||
        sourceStat.size !== row.bytes || fileHash(row.sourcePath) !== row.sha256 || !receiptStat.isFile() || receiptStat.isSymbolicLink() ||
        fileHash(row.sourceReceipt.path) !== row.sourceReceipt.sha256) {throw new Error("CUDA notice source or its receipt changed: " + row.file);}
    totalBytes += row.bytes;
    if (!Number.isSafeInteger(totalBytes) || totalBytes > (8 << 20)) {throw new Error("CUDA notice-source bundle exceeds its total byte bound");}
    const destination = path.join(state.noticeRoot, row.file);
    fs.copyFileSync(row.sourcePath, destination, fs.constants.COPYFILE_EXCL); fs.chmodSync(destination, 0o644);
    if (fileHash(destination) !== row.sha256 || fs.statSync(destination).size !== row.bytes) {throw new Error("CUDA notice copy differs from its admitted source: " + row.file);}
  }
  fs.writeFileSync(state.noticeManifest, JSON.stringify(plan, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return auditCudaNoticeFiles(state, plan);
}

/** @param {Pick<ReturnType<typeof cudaImagesStatePaths>, "noticeRoot" | "noticeManifest">} state @param {CudaNoticeSourcePlan} [expected]
 * @returns {{manifest:CudaNoticeSourcePlan,manifestSha256:string,files:Array<{path:string,sha256:string,bytes:number}>}} */
function auditCudaNoticeFiles(state, expected) {
  const rootStat = fs.lstatSync(state.noticeRoot, {throwIfNoEntry: false}), manifestStat = fs.lstatSync(state.noticeManifest, {throwIfNoEntry: false});
  if (!rootStat?.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(state.noticeRoot) !== state.noticeRoot ||
      !manifestStat?.isFile() || manifestStat.isSymbolicLink() || manifestStat.size > (64 << 10)) {
    throw new Error("CUDA notice-source artifact root or manifest is missing, unsafe or excessive");
  }
  const manifest = readBuildJson(state.noticeManifest), plan = expected || manifest;
  if (!plan || typeof plan !== "object" || Array.isArray(plan) || !Array.isArray(plan.sources) ||
      !isDeepStrictEqual(manifest, plan)) {throw new Error("CUDA notice-source manifest changed during handoff");}
  const names = ["manifest.json", ...plan.sources.map((/** @type {CudaNoticeSource} */ row) => row.file)].sort();
  if (new Set(names).size !== names.length || !isDeepStrictEqual(fs.readdirSync(state.noticeRoot).sort(), names)) {
    throw new Error("CUDA notice-source artifact has missing or unexpected files");
  }
  for (const row of plan.sources) {
    const destination = path.join(state.noticeRoot, row.file), stat = fs.lstatSync(destination);
    if (!safeRelative(row.file) || row.file.includes("/") || !/^[a-f0-9]{64}$/.test(row.sha256 || "") ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > (2 << 20) || !stat.isFile() || stat.isSymbolicLink() ||
        (stat.mode & 0o111) !== 0 || stat.size !== row.bytes || fileHash(destination) !== row.sha256) {
      throw new Error("CUDA notice-source artifact failed exact body verification: " + row.file);
    }
  }
  return {manifest, manifestSha256: fileHash(state.noticeManifest),
    files: plan.sources.map((/** @type {CudaNoticeSource} */ row) => ({path: path.join(state.noticeRoot, row.file), sha256: row.sha256, bytes: row.bytes}))};
}

/** @param {Pick<ReturnType<typeof cudaImagesStatePaths>, "noticeRoot" | "noticeManifest" | "sdk126" | "cutlassMarker" | "cutlassLicense"> &
 * {dpcpp:{root:string,manifest:string},linux:{sdkRoot:string}}} state @param {ReturnType<typeof auditDpcppCudaInputs>} dpcpp
 * @param {unknown} imageManifest @param {CudaNoticeSourcePlan} [expected]
 * @returns {{path:string,manifest:string,manifestSha256:string,files:Array<{path:string,sha256:string,bytes:number}>,completeNativeNotices:false}} */
function auditCudaNoticeSources(state, dpcpp, imageManifest, expected) {
  const filesAudit = auditCudaNoticeFiles(state, expected), manifest = filesAudit.manifest,
    compilers = imageManifest && typeof imageManifest === "object" && Array.isArray(Reflect.get(imageManifest, "compilers")) ?
      /** @type {Array<{major:number,minor:number,inputs:Record<string,string>}>} */ (Reflect.get(imageManifest, "compilers")) : [],
    compiler126 = compilers.find(row => row.major === 12 && row.minor === 6), compiler133 = compilers.find(row => row.major === 13 && row.minor === 3),
    installer = path.join(repo, "scripts/install-cutlass.sh"), installerSha256 = fileHash(installer),
    nvrtc126 = nvrtcCohortInfo("linux", "linux-12.6"), nvrtc133 = nvrtcCohortInfo("linux", "linux"),
    cutlassMarker = state.cutlassMarker, installerHash126 = compiler126?.inputs[installer], installerHash133 = compiler133?.inputs[installer];
  /** @param {Record<string,string> | undefined} inputs @param {string} file */
  const inputHash = (inputs, file) => {
    const value = inputs?.[file];
    if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) {throw new Error("CUDA image input receipt omits notice source: " + file);}
    return value;
  };
  if (compilers.length !== 2 || !compiler126 || !compiler133 || installerHash126 !== installerSha256 || installerHash133 !== installerSha256 ||
      !manifest || typeof manifest !== "object" || Array.isArray(manifest) || manifest.format !== "mom-cuda-notice-sources-v1" ||
      manifest.scope !== "CUDA/NVRTC/CUTLASS source license bodies only" || manifest.completeNativeNotices !== false ||
      !Array.isArray(manifest.sources) || manifest.sources.length !== 7 ||
      !hasExactKeys(manifest.sourceArchive, ["version", "url", "sha256", "member", "installer", "archiveMarker"])) {
    throw new Error("CUDA notice-source artifact is missing or has an invalid/incomplete manifest");
  }
  const specs = [
    ...dpcpp.cuda.licenses.map((/** @type {{package:string,version:string,path:string,sha256:string}} */ row) => ({
      component: row.package.replace(/-12-6$/, ""), version: row.version, sourcePath: row.path,
      file: cudaCopyrightName(row.package), sha256: row.sha256, sourceReceipt: {path: state.dpcpp.manifest,
        sha256: inputHash(compiler126.inputs, state.dpcpp.manifest)}, compilers: [compiler126, compiler133]})),
    {component: "nvrtc", version: nvrtc126.version, sourcePath: path.join(state.sdk126, "licenses/nvrtc/LICENSE"),
      file: "nvrtc-12.6-LICENSE.txt", sha256: inputHash(compiler126.inputs, path.join(state.sdk126, "licenses/nvrtc/LICENSE")),
      sourceReceipt: {path: path.join(state.sdk126, "sdk.json"), sha256: inputHash(compiler126.inputs, path.join(state.sdk126, "sdk.json"))}, compilers: [compiler126]},
    {component: "nvrtc", version: nvrtc133.version, sourcePath: path.join(state.linux.sdkRoot, "licenses/nvrtc/LICENSE"),
      file: "nvrtc-13.3-LICENSE.txt", sha256: inputHash(compiler133.inputs, path.join(state.linux.sdkRoot, "licenses/nvrtc/LICENSE")),
      sourceReceipt: {path: path.join(state.linux.sdkRoot, "sdk.json"), sha256: inputHash(compiler133.inputs, path.join(state.linux.sdkRoot, "sdk.json"))}, compilers: [compiler133]},
    {component: "nvptxcompiler", version: nvrtc133.version, sourcePath: path.join(state.linux.sdkRoot, "licenses/nvptxcompiler/LICENSE"),
      file: "nvptxcompiler-13.3-LICENSE.txt", sha256: inputHash(compiler133.inputs, path.join(state.linux.sdkRoot, "licenses/nvptxcompiler/LICENSE")),
      sourceReceipt: {path: path.join(state.linux.sdkRoot, "sdk.json"), sha256: inputHash(compiler133.inputs, path.join(state.linux.sdkRoot, "sdk.json"))}, compilers: [compiler133]},
    {component: "cutlass", version: cutlassVersion.slice(1), sourcePath: state.cutlassLicense,
      file: "cutlass-4.6.1-LICENSE.txt", sha256: cutlassLicenseSha256,
      sourceReceipt: {path: cutlassMarker, sha256: inputHash(compiler126.inputs, cutlassMarker)}, compilers: [compiler126, compiler133]},
  ];
  const rawSources = manifest.sources;
  if (!hasExactKeys(manifest, ["format", "scope", "completeNativeNotices", "sourceArchive", "sources"]) ||
      !isDeepStrictEqual(manifest.sourceArchive, {version: cutlassVersion.slice(1),
        url: `https://github.com/NVIDIA/cutlass/archive/refs/tags/${cutlassVersion}.tar.gz`, sha256: cutlassArchiveSha256,
        member: `cutlass-${cutlassVersion.slice(1)}/LICENSE.txt`, installer: {path: installer, sha256: installerSha256},
        archiveMarker: {path: cutlassMarker, sha256: inputHash(compiler126.inputs, cutlassMarker)}})) {
    throw new Error("CUDA notice-source manifest differs from the pinned CUTLASS archive or worker recipe");
  }
  let totalBytes = 0;
  const expectedRows = specs.map((spec, index) => {
    // Compiler requirements guide admission; they are not serialized notice-source fields.
    const {compilers: requiredCompilers, ...source} = spec;
    const row = rawSources[index], bytes = row && row.bytes;
    if (!row || typeof bytes !== "number" || !Number.isSafeInteger(bytes) || bytes < 1 || bytes > (2 << 20)) {throw new Error("CUDA notice-source body exceeds its bound");}
    for (const compiler of requiredCompilers) {
      if (inputHash(compiler.inputs, spec.sourcePath) !== spec.sha256 ||
          inputHash(compiler.inputs, spec.sourceReceipt.path) !== spec.sourceReceipt.sha256) {
        throw new Error("CUDA image worker recipe no longer binds notice source and receipt: " + spec.file);
      }
    }
    totalBytes += bytes;
    return {...source, bytes};
  });
  if (!Number.isSafeInteger(totalBytes) || totalBytes > (8 << 20) || !isDeepStrictEqual(rawSources, expectedRows) ||
      expected && !isDeepStrictEqual(manifest, expected)) {throw new Error("CUDA notice-source artifact or worker provenance changed");}
  auditCudaPackageCopyrights(state.dpcpp, dpcpp.cuda.packages, dpcpp.cuda.licenses);
  return {path: state.noticeRoot, manifest: state.noticeManifest, manifestSha256: filesAudit.manifestSha256,
    files: filesAudit.files, completeNativeNotices: false};
}

/** @typedef {Pick<ReturnType<typeof cudaImagesStatePaths>, "root" | "source126" | "sdk126" | "handoffRoot" |
 * "headerRoot" | "header126" | "header133" | "cutlassRoot" | "cutlassInclude" | "cutlassMarker" |
 * "cutlassLicense" | "output" | "config" | "result" | "noticeRoot" | "noticeManifest"> &
 * {linux:{source:string,sdkRoot:string},dpcpp:import("./native-dpcpp-cuda-inputs").DpcppCudaInputState}} LinuxCudaImageTreeState */

/** Build the maintained CUDA image and license-source cohort from admitted directories.
 * Transport, tool installation and memory containment belong to the calling frontdoor.
 * @param {LinuxCudaImageTreeState} state @param {string} compiler
 * @param {NodeJS.ProcessEnv} env @param {string} [manifestSha256] */
function buildLinuxCudaImageTree(state, compiler, env, manifestSha256) {
  const outputs = [state.root, state.source126, state.sdk126, state.linux.source, state.linux.sdkRoot,
    state.handoffRoot];
  if (outputs.some(file => fs.existsSync(file))) {throw new Error("CUDA image producer requires fresh same-run outputs");}
  const dpcppInputs = auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), manifestSha256);
  fs.mkdirSync(state.root, {mode: 0o700});
  runNode("scripts/fetch-native-sdk-sources.js", ["nvrtc-linux", state.linux.source, "--allow-network"], env, 15 * 60 * 1000);
  runNode("scripts/fetch-native-sdk-sources.js", ["nvrtc-linux-12.6", state.source126, "--allow-network"], env, 15 * 60 * 1000);
  for (const cohort of /** @type {const} */ (["linux", "linux-12.6"])) {
    const source = cohort === "linux" ? state.linux.source : state.source126,
      sdk = cohort === "linux" ? state.linux.sdkRoot : state.sdk126;
    verifyNvrtcArchives("linux", source, {cohort});
    runNode("scripts/prepare-native-nvrtc-sdk.js", [cohort, source, sdk], env, 15 * 60 * 1000);
  }
  const modern = nvrtcSdkInputs("linux", state.linux.sdkRoot, {cohort: "linux"}), baseline = nvrtcSdkInputs("linux", state.sdk126, {cohort: "linux-12.6"});
  const version = spawnSync(compiler, ["--version"],
    {encoding: "utf8", env, timeout: 30000, maxBuffer: 1 << 20});
  const identity = (version.stdout || "") + (version.stderr || "");
  if (version.error || version.signal || version.status !== 0 || !/2026\.2/.test(identity)) {
    throw new Error("CUDA image worker requires the existing Intel 2026.2 compiler", {cause: version.error});
  }
  const compilerIdentity = path.join(state.root, "icpx-version.txt");
  fs.writeFileSync(compilerIdentity, identity, {flag: "wx", mode: 0o600});
  const installed = spawnSync("/usr/bin/bash", ["-euc", '. "$1"; install_cutlass_headers "$2"', "install-cutlass",
    path.join(repo, "scripts/install-cutlass.sh"), state.cutlassRoot],
  {cwd: repo, encoding: "utf8", env, timeout: 10 * 60 * 1000, maxBuffer: 4 << 20});
  const cutlassLog = path.join(state.root, "cutlass-install.log");
  fs.writeFileSync(cutlassLog, (installed.stdout || "") + (installed.stderr || ""), {flag: "wx", mode: 0o600});
  if (installed.error || installed.signal || installed.status !== 0 ||
      fs.readFileSync(state.cutlassMarker, "utf8").trim() !== "455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c" ||
      !fs.statSync(path.join(state.cutlassInclude, "cute/tensor.hpp")).isFile() || fileHash(state.cutlassLicense) !== cutlassLicenseSha256) {
    throw new Error("Pinned CUTLASS 4.6.1 header installation failed", {cause: installed.error});
  }
  fs.mkdirSync(state.headerRoot, {mode: 0o700});
  for (const pack of [state.header126, state.header133]) {
    runNode("scripts/native-jit-headers.js", [pack, state.cutlassInclude, path.join(state.dpcpp.cudaRoot, "include")], env, 10 * 60 * 1000);
    const admitted = cudaHeaderInputs(pack);
    if (!admitted.manifest["roots"].includes(fs.realpathSync(path.join(state.dpcpp.cudaRoot, "include")))) {
      throw new Error("CUDA image header pack lost the admitted CUDA 12.6 toolkit root");
    }
  }
  const common = {cudaInclude: path.join(state.dpcpp.cudaRoot, "include"), cutlassInclude: state.cutlassInclude,
    cutlassMarker: state.cutlassMarker, cutlassLicense: state.cutlassLicense,
    cudaCopyrightLicenses: dpcppInputs.cuda.licenses.map((/** @type {{path:string}} */ row) => row.path), cudaCopyrightManifest: state.dpcpp.manifest,
    compiler, compilerIdentity,
    installer: path.join(repo, "scripts/install-cutlass.sh")};
  const config = {repo, output: state.output, compilers: [
    cudaImageWorkerRecipe(baseline, {...common, cohort: "linux-12.6", headerPack: state.header126}),
    cudaImageWorkerRecipe(modern, {...common, cohort: "linux", headerPack: state.header133}),
  ], timeoutMs: 30 * 60 * 1000};
  fs.writeFileSync(state.config, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  runNode("scripts/build-native-cuda-images.js", [state.config], env, 50 * 60 * 1000);
  const admitted = cudaImageInputs(path.join(state.output, "cuda-images.inc"), repo), result = readBuildJson(state.result),
    compilerRows = admitted.manifest["compilers"], imageRows = admitted.manifest["images"];
  if (!Array.isArray(compilerRows) || !Array.isArray(imageRows) ||
      compilerRows.some(row => !row || typeof row !== "object" || typeof row.major !== "number" || typeof row.minor !== "number")) {
    throw new Error("CUDA image bundle compiler or image matrix is malformed");
  }
  const compilerVersions = compilerRows.map(row => `${row.major}.${row.minor}`).sort();
  if (compilerVersions.join(",") !== "12.6,13.3" || result.plannedImages !== imageRows.length ||
      result.sourceSha256 !== admitted.manifest["sourceSha256"]) {
    throw new Error("CUDA image bundle did not admit the exact 12.6/13.3 same-run compiler matrix");
  }
  fs.mkdirSync(state.handoffRoot, {mode: 0o700});
  const noticePlan = cudaNoticeSourcePlan(state, dpcppInputs, baseline, modern),
    retainedNoticeFiles = retainCudaNoticeSources(state, noticePlan),
    noticeSources = auditCudaNoticeSources(state, dpcppInputs, admitted.manifest, noticePlan);
  if (retainedNoticeFiles.manifestSha256 !== noticeSources.manifestSha256) {throw new Error("CUDA notice-source manifest changed after retention");}
  /** @type {Array<[string, string]>} */ const outputFiles = [
    [path.join(state.output, "cuda-images.inc"), "cuda-images.inc"],
    [path.join(state.output, "cuda-images.inc.json"), "cuda-images.inc.json"], [state.result, "result.json"],
  ];
  for (const [source, name] of outputFiles) {
    fs.copyFileSync(source, path.join(state.handoffRoot, name), fs.constants.COPYFILE_EXCL);
  }
  return {noticeSources, images: admitted};
}

/** Restore the pinned CI inputs, invoke the shared producer, then transport its exact outputs.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildLinuxCudaImages(paths) {
  const state = cudaImagesStatePaths(paths, paths.env), env = paths.env;
  const externalDpcppPins = dpcppCudaInputPins(env);
  if ([state.root, state.source126, state.sdk126, state.linux.source, state.linux.sdkRoot, state.handoffRoot, state.tar]
    .some(file => fs.existsSync(file))) {throw new Error("CUDA image producer requires fresh same-run outputs");}
  const incoming = fs.lstatSync(state.incomingDpcppTar, {throwIfNoEntry: false});
  if (!incoming?.isFile() || incoming.isSymbolicLink()) {throw new Error("CUDA image producer needs the same-run DPC++/CUDA 12.6 tar");}
  if (externalDpcppPins) {dpcppSourceArchivePin(state.incomingDpcppTar, externalDpcppPins.bundleSha256);}
  extractState(state.incomingDpcppTar, temp, [state.dpcpp.name]);
  auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), externalDpcppPins?.manifestSha256);
  fs.unlinkSync(state.incomingDpcppTar);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-cuda-images-" + paths.suffix, 25 * 60 * 1000);
  const compiler = canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx"),
    {noticeSources} = buildLinuxCudaImageTree(state, compiler, env, externalDpcppPins?.manifestSha256);
  const packed = packState(state.tar, temp, [state.handoffName]), members = listState(state.tar, [state.handoffName]).map(name => name.replace(/\/$/, "")).sort(),
    expected = [state.handoffName, `${state.handoffName}/notice-sources`,
      ...["cuda-images.inc", "cuda-images.inc.json", "result.json"].map(name => `${state.handoffName}/${name}`),
      `${state.handoffName}/notice-sources/manifest.json`,
      ...noticeSources.files.map((/** @type {{path:string,sha256:string,bytes:number}} */ row) => `${state.handoffName}/notice-sources/${path.basename(row.path)}`)].sort();
  if (packed.bytes <= 0 || !isDeepStrictEqual(members, expected)) {throw new Error("CUDA image handoff tar is not the exact include, sidecar and result");}
  console.log(`[native-ci] CUDA images plus ${noticeSources.files.length} bounded same-run license sources admitted (${packed.bytes} bytes); this subset is not complete native distribution notices or publication qualification.`);
}

/** Hermes uses a separate same-run source and SDK root, independent of the LLVM/I/O graph.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths @param {NodeJS.ProcessEnv} [environment] */
function hermesStatePaths(paths, environment = process.env) {
  const sourceName = "mom-native-hermes-prerequisites-" + paths.suffix, sdkName = "mom-native-hermes-sdk-" + paths.suffix,
    source = path.join(temp, sourceName), sdkRoot = path.join(temp, sdkName);
  if (environment["MOM_NATIVE_HERMES_SOURCE_DIR"] !== source || environment["MOM_NATIVE_HERMES_SDK_DIR"] !== sdkRoot) {
    throw new Error("Native Hermes CI roots must exactly match this workflow run");
  }
  return {source, sourceName, sdkRoot, sdkName,
    sourceTar: path.join(paths.runnerTemp, "native-hermes-state-" + paths.suffix + "-sources.tar"),
    incomingTar: path.join(paths.runnerTemp, "incoming", "native-hermes-state-" + paths.suffix + "-sources.tar"),
    sdkTar: path.join(paths.runnerTemp, "native-linux-hermes-sdk-" + paths.suffix + ".tar")};
}

/** ACPP source inputs are an independent source-only cohort, with the prepared checkout and
 * original bare Git root kept together so all receipt paths survive same-run tar handoffs.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths @param {NodeJS.ProcessEnv} [environment] */
function acppStatePaths(paths, environment = process.env) {
  const sourceName = "mom-native-acpp-prerequisites-" + paths.suffix, source = path.join(temp, sourceName),
    prepared = path.join(source, "prepared"), sourceTar = path.join(paths.runnerTemp, "native-acpp-state-" + paths.suffix + "-sources.tar");
  if (environment["MOM_NATIVE_ACPP_SOURCE_DIR"] !== source) {
    throw new Error("Native ACPP CI source root must exactly match this workflow run");
  }
  return {source, sourceName, prepared, preparedCheckout: path.join(prepared, "acpp"), fetchReceipt: path.join(source, "fetch.json"),
    rootsFile: path.join(source, "roots.json"), preparedReceipt: path.join(prepared, "sources.json"), sourceTar,
    incomingTar: path.join(paths.runnerTemp, "incoming", path.basename(sourceTar))};
}

/** Pinned Linux SDK source and patch state, kept together at its original /tmp paths.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths @param {NodeJS.ProcessEnv} [environment] */
function linuxSourceStatePaths(paths, environment = process.env) {
  const sourceName = "mom-native-linux-source-prerequisites-" + paths.suffix, source = path.join(temp, sourceName),
    prepared = path.join(source, "prepared"), linuxPrepared = path.join(prepared, "linux"), compilerPrepared = path.join(prepared, "compiler"),
    sourceTar = path.join(paths.runnerTemp, "native-linux-sdk-state-" + paths.suffix + "-sources.tar");
  if (environment["MOM_NATIVE_LINUX_SOURCES_DIR"] !== source) {
    throw new Error("Native Linux SDK source root must exactly match this workflow run");
  }
  return {source, sourceName, prepared, rootsFile: path.join(source, "roots.json"), fetchReceipt: path.join(source, "fetch.json"),
    linuxPrepared, compilerPrepared, preparedReceipt: path.join(linuxPrepared, "sources.json"),
    compilerPreparedReceipt: path.join(compilerPrepared, "sources.json"), sourceTar,
    incomingTar: path.join(paths.runnerTemp, "incoming", path.basename(sourceTar))};
}

/** Same-run DPC++23/CUDA 12.6 input handoff. The source tar is independently re-admitted;
 * only the pinned compiler SDK and the small CUDA configure/header/device-resource closure are retained.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths @param {NodeJS.ProcessEnv} [environment] */
function dpcppCudaInputsStatePaths(paths, environment = process.env) {
  const linux = linuxSourceStatePaths(paths, environment), name = "mom-native-dpcpp-cuda-inputs-" + paths.suffix,
    root = path.join(temp, name), sdkTar = path.join(paths.runnerTemp, "native-linux-dpcpp-cuda-inputs-" + paths.suffix + ".tar");
  if (environment["MOM_NATIVE_DPCPP_CUDA_INPUTS_DIR"] !== root || !path.isAbsolute(paths.runnerTemp) ||
      fs.realpathSync(paths.runnerTemp) !== paths.runnerTemp) {
    throw new Error("Native DPC++/CUDA input root must exactly match this workflow run");
  }
  return {linux, name, root, dpcppRoot: path.join(root, "dpcpp"), cudaRoot: path.join(root, "cuda"),
    manifest: path.join(root, "inputs.json"), cudaVersionEvidence: path.join(root, "cuda-nvcc-version.txt"),
    cudaProbeEvidence: path.join(root, "cuda-nvcc-cmake-probe.txt"), cudaCmakeSource: path.join(root, "cuda-cmake-probe"),
    cudaCmakeLists: path.join(root, "cuda-cmake-probe", "CMakeLists.txt"), cudaCmakeEvidence: path.join(root, "cuda-cmake-probe.json"),
    cudaCmakeBuild: path.join(paths.runnerTemp, ".dpcpp-cuda-cmake-probe-" + paths.suffix), sdkTar,
    incomingSourceTar: path.join(paths.runnerTemp, "incoming", path.basename(linux.sourceTar))};
}

/** Keep target GNU headers in a separate small same-run input tar, not in compiler caches.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths @param {NodeJS.ProcessEnv} [environment]
 * @param {"x86_64-unknown-linux-gnu" | "x86_64-w64-windows-gnu"} [target] */
function gnuHeadersStatePaths(paths, environment = process.env, target = "x86_64-unknown-linux-gnu") {
  if (!["x86_64-unknown-linux-gnu", "x86_64-w64-windows-gnu"].includes(target)) {throw new Error("Unsupported GNU header CI target");}
  const windows = target === "x86_64-w64-windows-gnu", name = windows ? "mom-native-windows-gnu-headers-" + paths.suffix : "mom-native-gnu-headers-" + paths.suffix,
    root = path.join(temp, name), variable = windows ? "MOM_NATIVE_WINDOWS_GNU_HEADERS_DIR" : "MOM_NATIVE_GNU_HEADERS_DIR";
  if (environment[variable] !== root || fs.realpathSync(paths.runnerTemp) !== paths.runnerTemp) {
    throw new Error("Native GNU header root must exactly match this workflow run");
  }
  return {target, name, root, sdkTar: path.join(paths.runnerTemp, `native-${windows ? "windows" : "linux"}-gnu-headers-${paths.suffix}.tar`)};
}

/** Preserve every producer's original absolute path while keeping the generated JIT state separate.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths @param {NodeJS.ProcessEnv} [environment] */
function linuxJitResourcesStatePaths(paths, environment = process.env) {
  const linux = linuxSourceStatePaths(paths, environment), dpcpp = dpcppCudaInputsStatePaths(paths, environment),
    gnu = gnuHeadersStatePaths(paths, environment), name = "mom-native-linux-jit-resources-" + paths.suffix,
    root = path.join(temp, name), sdkTar = path.join(paths.runnerTemp, "native-linux-jit-resources-" + paths.suffix + ".tar");
  if (environment["MOM_NATIVE_LINUX_JIT_RESOURCES_DIR"] !== root || !path.isAbsolute(paths.runnerTemp) ||
      fs.realpathSync(paths.runnerTemp) !== paths.runnerTemp) {
    throw new Error("Native Linux JIT resource root must exactly match this workflow run");
  }
  return {linux, dpcpp, gnu, name, root, emitter: path.join(root, "emitter"), bootstrap: path.join(root, "bootstrap"),
    bootstrapBuild: path.join(root, "bootstrap/build"), bootstrapOverlay: path.join(root, "bootstrap-overlay"),
    bootstrapOverlayEvidence: path.join(root, "bootstrap-overlay.json"),
    base: path.join(root, "base"), profile: path.join(root, "resource-profile.json"), resources: path.join(root, "resources"),
    configureEvidence: path.join(root, "bootstrap-configure.json"), result: path.join(root, "ci-result.json"), sdkTar,
    incomingSourceTar: linux.incomingTar, incomingDpcppTar: path.join(paths.runnerTemp, "incoming", path.basename(dpcpp.sdkTar)),
    incomingGnuTar: path.join(paths.runnerTemp, "incoming", path.basename(gnu.sdkTar))};
}

/** @typedef {Pick<ReturnType<typeof linuxJitResourcesStatePaths>, "root" | "emitter" | "bootstrap" | "bootstrapBuild" |
 * "bootstrapOverlay" | "bootstrapOverlayEvidence" | "base" | "profile" | "resources" | "configureEvidence"> &
 * {linux:LinuxPreparedSources,dpcpp:import("./native-dpcpp-cuda-inputs").DpcppCudaInputState,gnu:{root:string}}} LinuxJitResourceTreeState */

/** Pure invocation plan for the existing emitter, headers-only configure and resource producers.
 * @param {LinuxJitResourceTreeState} state @param {Record<string,string>} roots
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools @param {string} sourceDate */
function linuxJitResourceBuildPlan(state, roots, tools, sourceDate) {
  if (!/^[0-9]{8}$/.test(sourceDate)) {throw new Error("JIT bootstrap needs the pinned DPC++ compiler date");}
  const configure = [...jitSdkPlan("linux", roots, state.dpcpp.dpcppRoot, state.bootstrapOverlay, state.bootstrap, tools, 1).configure,
    "-DSYCL_COMPILER_VERSION=" + sourceDate];
  return {sourceDate, emitter: {script: "scripts/native-jit-sdk.js", args: ["emitter", state.dpcpp.dpcppRoot, state.emitter, path.join(state.gnu.root, "target-headers.json")]},
    configure, base: {script: "scripts/prepare-native-jit-base.js", args: ["--sdk", state.dpcpp.dpcppRoot, "--cuda", state.dpcpp.cudaRoot,
      "--build", state.bootstrapBuild, "--sources", state.linux.compilerPreparedReceipt,
      "--dpcpp-git", path.join(state.linux.source, "roots/dpcpp"), "--output", state.base]},
    resources: {script: "scripts/prepare-native-sycl-jit-resources.js", args: ["--profile", state.profile, "--output", state.resources]}};
}

/** Build the CMake probe overlay from real admitted target headers, never placeholder bytes.
 * Directory links keep the bootstrap small; the source/GNU receipts are independently rechecked.
 * @param {LinuxJitResourceTreeState} state @param {{headers:{root:string,files?:unknown[]},notices?:unknown[]}} gnu
 * @param {{root:string,revision:string}} opencl */
function linuxJitBootstrapOverlayPlan(state, gnu, opencl) {
  const rows = [
    {virtual: "include/mom-cxx", source: path.join(gnu.headers.root, "include/mom-cxx"), kind: "gnu"},
    {virtual: "include/mom-cxx-target", source: path.join(gnu.headers.root, "include/mom-cxx-target"), kind: "gnu"},
    {virtual: "include/mom-system-target", source: path.join(gnu.headers.root, "include/mom-system-target"), kind: "gnu"},
    {virtual: "include/mom-system", source: path.join(gnu.headers.root, "include/mom-system"), kind: "gnu"},
    {virtual: "include/CL", source: path.join(opencl.root, "CL"), kind: "opencl"},
  ];
  if (!/^[a-f0-9]{40}$/.test(opencl.revision) || rows.some(row => !path.isAbsolute(row.source) || path.resolve(row.source) !== row.source)) {
    throw new Error("JIT bootstrap overlay needs admitted GNU target roots and pinned OpenCL-Headers");
  }
  return rows.map(row => ({...row, link: path.join(state.bootstrapOverlay, ...row.virtual.split("/"))}));
}

/** Materialize only directory links to admitted GNU/OpenCL header trees and bind their sources.
 * @param {LinuxJitResourceTreeState} state @param {{headers:{root:string,files?:unknown[]},notices?:unknown[]}} gnu
 * @param {{root:string,revision:string}} opencl */
function seedLinuxJitBootstrapOverlay(state, gnu, opencl) {
  if (fs.lstatSync(state.bootstrapOverlay, {throwIfNoEntry: false}) || fs.lstatSync(state.bootstrapOverlayEvidence, {throwIfNoEntry: false})) {
    throw new Error("JIT bootstrap overlay refuses existing state");
  }
  fs.mkdirSync(path.join(state.bootstrapOverlay, "include"), {recursive: true, mode: 0o700});
  const rows = linuxJitBootstrapOverlayPlan(state, gnu, opencl);
  for (const row of rows) {
    const source = fs.realpathSync(row.source), stat = fs.lstatSync(row.source);
    if (!stat.isDirectory() || stat.isSymbolicLink() || source !== row.source ||
        row.kind === "opencl" && !fs.lstatSync(path.join(source, "cl.h"), {throwIfNoEntry: false})?.isFile()) {
      throw new Error("JIT bootstrap overlay source is missing or not canonical: " + row.virtual);
    }
    fs.symlinkSync(source, row.link, "dir");
    if (!fs.lstatSync(row.link).isSymbolicLink() || fs.realpathSync(row.link) !== source) {
      throw new Error("JIT bootstrap overlay link differs from its admitted source: " + row.virtual);
    }
  }
  const record = {version: 1, root: state.bootstrapOverlay,
    roots: rows.map(({virtual, source, link, kind}) => ({virtual, source, link, kind})),
    gnuReceipt: {path: path.join(state.gnu.root, "result.json"), sha256: fileHash(path.join(state.gnu.root, "result.json"))},
    compilerReceipt: {path: state.linux.compilerPreparedReceipt, sha256: fileHash(state.linux.compilerPreparedReceipt)},
    openclRevision: opencl.revision};
  fs.writeFileSync(state.bootstrapOverlayEvidence, JSON.stringify(record, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return record;
}

/** @param {LinuxJitResourceTreeState} state @param {unknown} expected */
function auditLinuxJitBootstrapOverlay(state, expected) {
  const record = readBuildJson(state.bootstrapOverlayEvidence);
  if (!isDeepStrictEqual(record, expected) || record.root !== state.bootstrapOverlay || !Array.isArray(record.roots) ||
      record.roots.length !== 5 || record.gnuReceipt?.sha256 !== fileHash(record.gnuReceipt.path) ||
      record.compilerReceipt?.sha256 !== fileHash(record.compilerReceipt.path)) {
    throw new Error("JIT bootstrap overlay provenance changed");
  }
  for (const row of record.roots) {
    if (!fs.lstatSync(row.link).isSymbolicLink() || fs.readlinkSync(row.link) !== row.source ||
        fs.realpathSync(row.link) !== row.source || !fs.statSync(row.link).isDirectory()) {
      throw new Error("JIT bootstrap overlay link changed: " + row.virtual);
    }
  }
  return record;
}

/** Bind the public resource profile only to admitted emitter, GNU package and JIT-base records.
 * @param {LinuxJitResourceTreeState} state
 * @param {{fixture:string,compiler:string,targetHeaders:string[]}} emitter
 * @param {{headers:{root:string},notices:Array<{package:string,path:string,sha256:string}>}} gnu */
function linuxJitResourceProfile(state, emitter, gnu) {
  const target = resourceTarget("x86_64-unknown-linux-gnu"), base = path.join(state.base, "base");
  return {version: 1, target: target.target, fixture: emitter.fixture, clang: emitter.compiler,
    receipt: path.join(state.emitter, "result.json"), gnuRoot: gnu.headers.root, baseRoot: base,
    noticeRoot: gnu.headers.root, syclInclude: path.join(base, "include"),
    clangBuiltinInclude: path.join(base, "lib/clang/23/include"), baseManifest: path.join(state.base, "base-manifest.json"),
    headerRoots: target.roots.map(virtual => ({path: path.join(gnu.headers.root, ...virtual.split("/")), virtual})),
    required: Object.fromEntries(Object.entries(jitBaseRequired).map(([name, row]) => [name, {virtual: row.virtual, sha256: row.sha256}])),
    notices: gnu.notices.map(notice => ({source: path.relative(gnu.headers.root, notice.path).split(path.sep).join("/"),
      virtual: "usr/share/licenses/" + notice.package + "/copyright", sha256: notice.sha256}))};
}

/** Produce the JIT resource tree from admitted directories, independently of CI transport.
 * Callers provide installed tools and containment; this recipe never installs packages.
 * @param {LinuxJitResourceTreeState} state
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools
 * @param {NodeJS.ProcessEnv} environment @param {string} [expectedDpcppManifestSha256] */
function buildLinuxJitResourceTree(state, tools, environment, expectedDpcppManifestSha256) {
  if (fs.lstatSync(state.root, {throwIfNoEntry: false})) {throw new Error("Linux JIT resource stage refuses an existing output");}
  const git = canonicalExecutable("/usr/bin/git"), paths = {env: environment};
  verifyLinuxSources(paths, state.linux, git);
  auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), expectedDpcppManifestSha256);
  const gnu = auditGnuHeaders(state.gnu),
    compiler = verifyPreparedSdkSources(state.linux.compilerPreparedReceipt, "compiler"),
    compilerRoots = Object.fromEntries(Object.entries(compiler.prepared).map(([name, component]) => [name, component.root]));
  fs.mkdirSync(state.root, {mode: 0o700});
  const bootstrapOverlay = seedLinuxJitBootstrapOverlay(state, gnu, compiler.prepared.openclHeaders),
    sourceDate = pinnedCommitDate(path.join(state.linux.source, "roots/dpcpp"), paths.env);
  if (!sourceDate) {throw new Error("Pinned DPC++ compiler date is missing");}
  const plan = linuxJitResourceBuildPlan(state, compilerRoots, tools, sourceDate);
  runNode(plan.emitter.script, plan.emitter.args, paths.env, 5 * 60 * 1000);
  const emitterReceipt = readBuildJson(path.join(state.emitter, "result.json"));
  const configureLog = path.join(state.root, "bootstrap-configure.log"), descriptor = fs.openSync(configureLog, "wx", 0o600);
  let configured;
  try {
    configured = spawnSync(plan.configure[0] || "", plan.configure.slice(1), {cwd: repo, env: paths.env, stdio: ["ignore", descriptor, descriptor], timeout: 5 * 60 * 1000});
  } finally {fs.closeSync(descriptor);}
  const configureRecord = {version: 1, command: plan.configure, sourceDate, bootstrapOverlay: {path: state.bootstrapOverlayEvidence, sha256: fileHash(state.bootstrapOverlayEvidence)},
    tools: Object.fromEntries(Object.entries(tools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}])),
    status: configured.status, signal: configured.signal || null, error: configured.error?.message || null,
    log: {path: configureLog, sha256: fileHash(configureLog)}};
  fs.writeFileSync(state.configureEvidence, JSON.stringify(configureRecord, null, 2) + "\n", {flag: "wx", mode: 0o600});
  if (configured.error || configured.signal || configured.status !== 0) {throw new Error("Headers-only JIT CMake configure failed; see " + configureLog, {cause: configured.error});}
  auditLinuxJitBootstrapOverlay(state, bootstrapOverlay);
  for (const file of [path.join(state.bootstrapBuild, "CMakeCache.txt"), path.join(state.bootstrapBuild, "build.ninja")]) {
    if (!fs.lstatSync(file).isFile() || fs.realpathSync(file) !== file) {throw new Error("Headers-only JIT configure omitted a regular build record");}
  }
  runNode(plan.base.script, plan.base.args, paths.env, 10 * 60 * 1000);
  const profile = linuxJitResourceProfile(state, emitterReceipt, gnu);
  fs.writeFileSync(state.profile, JSON.stringify(profile, null, 2) + "\n", {flag: "wx", mode: 0o600});
  runNode(plan.resources.script, plan.resources.args, paths.env, 8 * 60 * 1000);
  return auditLinuxJitResourceTree(state, tools, environment, expectedDpcppManifestSha256);
}

/** Bind resource-manifest base rows and required libraries to admitted producer records.
 * @param {unknown} manifest @param {unknown} required
 * @param {Array<{root:"base",source:string,virtual:string,sha256:string,bytes:number}>} expectedBaseRows */
function assertLinuxJitResourceManifest(manifest, required, expectedBaseRows) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest) || !Array.isArray(expectedBaseRows)) {
    throw new Error("Linux JIT resource manifest metadata is malformed");
  }
  const requiredObject = required && typeof required === "object" && !Array.isArray(required) ? required : null,
    requiredRows = /** @type {Array<{virtual?:unknown,sha256?:unknown}>} */ (requiredObject ? Object.values(requiredObject) : []);
  if (!requiredObject || Object.keys(requiredObject).sort().join(",") !== "libdevice,spirv32,spirv64" ||
      requiredRows.some(row => !row || typeof row !== "object" || Array.isArray(row) ||
        !expectedBaseRows.some(base => base.virtual === row.virtual && base.sha256 === row.sha256))) {
    throw new Error("Linux JIT required device libraries are not present in the admitted base payload");
  }
  const files = /** @type {{files?:unknown}} */ (manifest).files;
  if (!Array.isArray(files) || files.some(row => !row || typeof row !== "object" || Array.isArray(row) ||
      row.root !== "base" && row.root !== "gnu")) {
    throw new Error("Linux JIT resource manifest has malformed or unknown file roots");
  }
  const actualBase = /** @type {Array<{root:"base",source:string,virtual:string,sha256:string,bytes:number}>} */ (
      files.filter(row => row.root === "base")),
    byVirtual = (/** @type {Array<{virtual:string}>} */ rows) => [...rows].sort((left, right) =>
      left.virtual < right.virtual ? -1 : left.virtual > right.virtual ? 1 : 0);
  if (!isDeepStrictEqual(/** @type {{required?:unknown}} */ (manifest).required, required) ||
      !isDeepStrictEqual(byVirtual(actualBase), byVirtual(expectedBaseRows))) {
    throw new Error("Linux JIT resource manifest differs from the admitted base payload and required device libraries");
  }
}

/** Re-admit a completed resource tree without running an emitter/compiler or restamping receipts.
 * Returned fingerprints let a local cache bind input/tool and output identities across invocations.
 * @param {LinuxJitResourceTreeState} state
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools
 * @param {NodeJS.ProcessEnv} environment @param {string} [expectedDpcppManifestSha256] */
function auditLinuxJitResourceTree(state, tools, environment, expectedDpcppManifestSha256) {
  const baseRoot = path.join(state.base, "base"), fixture = path.join(state.emitter, "native-kawpow-source"),
    baseStat = fs.lstatSync(baseRoot, {throwIfNoEntry: false}), fixtureStat = fs.lstatSync(fixture, {throwIfNoEntry: false});
  if (!baseStat?.isDirectory() || baseStat.isSymbolicLink() || fs.realpathSync(baseRoot) !== baseRoot ||
      !fixtureStat?.isFile() || fixtureStat.isSymbolicLink() || fs.realpathSync(fixture) !== fixture) {
    throw new Error("JIT resource base and emitter must be canonical real directory/file outputs");
  }
  verifyLinuxSources({env: environment}, state.linux, canonicalExecutable("/usr/bin/git"));
  auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), expectedDpcppManifestSha256);
  const gnu = auditGnuHeaders(state.gnu), compiler = verifyPreparedSdkSources(state.linux.compilerPreparedReceipt, "compiler"),
    sourceDate = pinnedCommitDate(path.join(state.linux.source, "roots/dpcpp"), environment),
    emitterFile = path.join(state.emitter, "result.json"), emitter = readBuildJson(emitterFile),
    baseFile = path.join(state.base, "result.json"), base = readBuildJson(baseFile), baseManifest = path.join(state.base, "base-manifest.json"),
    targetHeaders = readBuildJson(path.join(state.gnu.root, "target-headers.json")),
    resources = verifyJitResources(state.resources, "x86_64-unknown-linux-gnu"), manifest = readBuildJson(path.join(state.resources, "manifest.json"));
  if (!sourceDate || emitter.version !== 1 || emitter.fixture !== path.join(state.emitter, "native-kawpow-source") ||
      emitter.compiler !== path.join(state.dpcpp.dpcppRoot, "bin/clang++") || !Array.isArray(targetHeaders) ||
      !isDeepStrictEqual(emitter.targetHeaders, targetHeaders) || emitter.buildOnly !== true || emitter.executed !== false ||
      emitter.publicationQualified !== false || emitter.fixtureSha256 !== fileHash(emitter.fixture) ||
      emitter.compilerSha256 !== fileHash(emitter.compiler) || manifest.receiptSha256 !== fileHash(emitterFile) ||
      manifest.fixtureSha256 !== emitter.fixtureSha256 || manifest.compilerSha256 !== emitter.compilerSha256) {
    throw new Error("Public Linux JIT emitter did not retain the admitted target-header set or current inputs");
  }
  const baseInventory = require("./prepare-native-jit-base").baseManifest(baseRoot);
  if (base.version !== 1 || base.baseRoot !== path.join(state.base, "base") || base.baseManifest !== baseManifest ||
      base.baseManifestSha256 !== fileHash(baseManifest) || manifest.baseManifestSha256 !== base.baseManifestSha256 ||
      !isDeepStrictEqual(base.required, jitBaseRequired) || base.sourceReceipt?.path !== state.linux.compilerPreparedReceipt ||
      base.dpcpp?.sdkRoot !== state.dpcpp.dpcppRoot || base.dpcpp?.commitDate !== sourceDate ||
      base.cuda?.root !== state.dpcpp.cudaRoot || base.build?.root !== state.bootstrapBuild || base.buildOnly !== true ||
      base.compiled !== false || base.executed !== false || base.publicationQualified !== false ||
      !isDeepStrictEqual(readBuildJson(baseManifest), baseInventory)) {
    throw new Error("Public JIT base receipt does not bind the current inputs and payload");
  }
  const overlay = {version: 1, root: state.bootstrapOverlay,
      roots: linuxJitBootstrapOverlayPlan(state, gnu, compiler.prepared.openclHeaders),
      gnuReceipt: {path: path.join(state.gnu.root, "result.json"), sha256: fileHash(path.join(state.gnu.root, "result.json"))},
      compilerReceipt: {path: state.linux.compilerPreparedReceipt, sha256: fileHash(state.linux.compilerPreparedReceipt)},
      openclRevision: compiler.prepared.openclHeaders.revision},
    roots = Object.fromEntries(Object.entries(compiler.prepared).map(([name, component]) => [name, component.root])),
    plan = linuxJitResourceBuildPlan(state, roots, tools, sourceDate), log = path.join(state.root, "bootstrap-configure.log"),
    configure = {version: 1, command: plan.configure, sourceDate,
      bootstrapOverlay: {path: state.bootstrapOverlayEvidence, sha256: fileHash(state.bootstrapOverlayEvidence)},
      tools: Object.fromEntries(Object.entries(tools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}])),
      status: 0, signal: null, error: null, log: {path: log, sha256: fileHash(log)}};
  auditLinuxJitBootstrapOverlay(state, overlay);
  const expectedProfile = linuxJitResourceProfile(state, emitter, gnu),
    /** @type {Array<{root:"base",source:string,virtual:string,sha256:string,bytes:number}>} */
    expectedBase = baseInventory.files.map(row => ({...row, root: /** @type {"base"} */ ("base"),
      bytes: fs.statSync(path.join(baseRoot, row.source)).size}));
  assertLinuxJitResourceManifest(manifest, expectedProfile.required, expectedBase);
  if (!isDeepStrictEqual(readBuildJson(state.configureEvidence), configure) ||
      !isDeepStrictEqual(readBuildJson(state.profile), expectedProfile)) {
    throw new Error("Linux JIT resource configure/profile differs from admitted inputs and tools");
  }
  const files = [emitterFile, fixture, baseFile, baseManifest, state.profile, state.configureEvidence, state.bootstrapOverlayEvidence,
      path.join(state.bootstrapBuild, "CMakeCache.txt"), path.join(state.bootstrapBuild, "build.ninja"),
      state.linux.fetchReceipt, state.linux.preparedReceipt, state.linux.compilerPreparedReceipt, state.dpcpp.manifest,
      path.join(state.gnu.root, "result.json"), path.join(state.gnu.root, "copyrights.json"), log, __filename],
    fingerprints = {...resources.inputs, ...emitter.inputs, [emitter.fixture]: emitter.fixtureSha256,
      ...Object.fromEntries(Object.values(tools).map(file => [file, fileHash(file)]))};
  for (const file of files) {
    if (!fs.lstatSync(file).isFile() || fs.realpathSync(file) !== file) {throw new Error("JIT resource admission requires regular canonical records: " + file);}
    fingerprints[file] = fileHash(file);
  }
  for (const [file, hash] of Object.entries(fingerprints)) {
    if (!path.isAbsolute(file) || !/^[a-f0-9]{64}$/.test(hash) || fileHash(file) !== hash) {throw new Error("JIT resource input or output changed: " + file);}
  }
  return {resources, sourceDate, fingerprints};
}

/** Restore and re-admit the exact same-run inputs, then produce only the CPU JIT resource pack.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildLinuxJitResources(paths) {
  const state = linuxJitResourcesStatePaths(paths, paths.env);
  if ([state.root, state.sdkTar].some(file => fs.lstatSync(file, {throwIfNoEntry: false}))) {
    throw new Error("Linux JIT resource stage refuses an existing output");
  }
  const externalDpcppPins = dpcppCudaInputPins(paths.env);
  if (externalDpcppPins) {dpcppSourceArchivePin(state.incomingDpcppTar, externalDpcppPins.bundleSha256);}
  const inputArchives = [
    {file: state.incomingSourceTar, root: state.linux.sourceName, role: "linux-sources"},
    {file: state.incomingDpcppTar, root: state.dpcpp.name, role: "dpcpp-cuda-inputs"},
    {file: state.incomingGnuTar, root: state.gnu.name, role: "gnu-headers"},
  ].map(input => {
    const stat = fs.lstatSync(input.file);
    if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Linux JIT resource stage requires all three same-run input tars");}
    return {...input, sha256: fileHash(input.file)};
  });
  for (const input of inputArchives) {extractState(input.file, temp, [input.root]);}
  verifyLinuxSources(paths, state.linux, canonicalExecutable("/usr/bin/git"));
  auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), externalDpcppPins?.manifestSha256);
  auditGnuHeaders(state.gnu);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-linux-jit-" + paths.suffix, 25 * 60 * 1000);
  const tools = {cmake: canonicalExecutable("/usr/bin/cmake"), ninja: canonicalExecutable("/usr/bin/ninja"),
      cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"), cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx")},
    {resources, sourceDate} = buildLinuxJitResourceTree(state, tools, paths.env, externalDpcppPins?.manifestSha256);
  if (inputArchives.some(input => fileHash(input.file) !== input.sha256)) {throw new Error("Same-run JIT input tar changed during production");}
  const baseManifest = path.join(state.base, "base-manifest.json"),
    baseReceiptFile = path.join(state.base, "result.json"), emitterReceiptFile = path.join(state.emitter, "result.json"),
    resourceManifest = path.join(state.resources, "manifest.json");
  const result = {version: 1, target: resources.target, buildOnly: true, executed: false, publicationQualified: false,
    scope: "CPU-only source-JIT resource preparation. It does not build the JIT SDK/compiler object/static runtime or qualify a device/runtime/publication.",
    inputArchives: inputArchives.map(({file, root, role, sha256}) => ({role, name: path.basename(file), root, sha256})),
    emitter: {receipt: emitterReceiptFile, sha256: fileHash(emitterReceiptFile)},
    bootstrap: {configure: state.configureEvidence, configureSha256: fileHash(state.configureEvidence), sourceDate,
      overlay: {path: state.bootstrapOverlayEvidence, sha256: fileHash(state.bootstrapOverlayEvidence)},
      cache: fileHash(path.join(state.bootstrapBuild, "CMakeCache.txt")), ninja: fileHash(path.join(state.bootstrapBuild, "build.ninja"))},
    base: {receipt: baseReceiptFile, sha256: fileHash(baseReceiptFile), manifest: baseManifest, manifestSha256: fileHash(baseManifest)},
    resources: {manifest: resourceManifest, manifestSha256: resources.manifestHash, files: readBuildJson(resourceManifest).files.length,
      notices: readBuildJson(resourceManifest).notices.length}, tools: Object.fromEntries(Object.entries(tools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}]))};
  fs.writeFileSync(state.result, JSON.stringify(result, null, 2) + "\n", {flag: "wx", mode: 0o600});
  packState(state.sdkTar, temp, [state.name]); listState(state.sdkTar, [state.name]);
  for (const input of inputArchives) {fs.unlinkSync(input.file);}
  console.log("[native-ci] Prepared and admitted the same-run Linux JIT resource pack; no JIT SDK, GPU, runtime, or publication qualification.");
}

/** Continue the resource handoff through the public source-JIT compiler phases.
 * Immutable upstream tars are independently restored at their original roots; only the
 * mutable compiler tree moves between jobs. @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths
 * @param {NodeJS.ProcessEnv} [environment] */
function linuxJitSdkStatePaths(paths, environment = process.env) {
  const resources = linuxJitResourcesStatePaths(paths, environment), name = "mom-native-linux-jit-sdk-" + paths.suffix,
    root = path.join(temp, name);
  if (environment["MOM_NATIVE_LINUX_JIT_SDK_DIR"] !== root || !path.isAbsolute(paths.runnerTemp) ||
      fs.realpathSync(paths.runnerTemp) !== paths.runnerTemp) {
    throw new Error("Native Linux source-JIT SDK root must exactly match this workflow run");
  }
  const phaseTar = (/** @type {string} */ phase) => path.join(paths.runnerTemp, "native-linux-jit-sdk-" + paths.suffix + "-" + linuxJitSdkArtifactPhase(phase) + ".tar"),
    phaseIncomingTar = (/** @type {string} */ phase) => path.join(paths.runnerTemp, "incoming", path.basename(phaseTar(phase)));
  return {resources, linux: resources.linux, dpcpp: resources.dpcpp, gnu: resources.gnu, name, root,
    output: path.join(root, "compiler-sdk"), isolation: path.join(root, "compiler-isolation"),
    isolatedObject: path.join(root, "compiler-isolation/isolated/compiler.o"), staticOutput: path.join(root, "static-dpcpp-sdk"),
    staticComponent: path.join(root, "static-dpcpp-sdk.component"), configFile: path.join(paths.runnerTemp, "native-linux-jit-sdk-" + paths.suffix + ".json"),
    ciInputs: path.join(root, "ci-inputs.json"), staticCiInputs: path.join(root, "static-ci-inputs.json"), phaseTar, phaseIncomingTar,
    inputTars: [
      {file: resources.linux.incomingTar, root: resources.linux.sourceName, role: "linux-sources"},
      {file: resources.incomingDpcppTar, root: resources.dpcpp.name, role: "dpcpp-cuda-inputs"},
      {file: resources.incomingGnuTar, root: resources.gnu.name, role: "gnu-headers"},
      {file: path.join(paths.runnerTemp, "incoming", path.basename(resources.sdkTar)), root: resources.name, role: "jit-resources"},
    ]};
}

/** Original same-run tars needed to compose (but not build) the public Linux release descriptor.
 * Every producer remains a separate raw artifact; shared /tmp roots are restored only once.
 * @param {ReturnType<typeof ciPaths>} paths @param {NodeJS.ProcessEnv} [environment] */
function linuxNativeSdkDescriptorStatePaths(paths, environment = process.env) {
  const linux = linuxSourceStatePaths(paths, environment), dpcpp = dpcppCudaInputsStatePaths(paths, environment),
    images = cudaImagesStatePaths(paths, environment), jit = linuxJitSdkStatePaths(paths, environment),
    acpp = acppStatePaths(paths, environment), acppSdk = acppCudaHipSdkStatePaths(paths, environment),
    gnu = gnuHeadersStatePaths(paths, environment), hermes = hermesStatePaths(paths, environment), openmp = openmpStatePaths(paths, environment),
    rocm = rocmHostStatePaths(paths, environment), zstd = zstdStatePaths(paths), name = paths.linuxSdkDescriptorName,
    root = paths.linuxSdkDescriptorRoot, incoming = path.join(paths.runnerTemp, "incoming"),
    archive = path.join(paths.runnerTemp, "native-linux-sdk-descriptor-" + paths.suffix + ".tar");
  if (environment["MOM_NATIVE_LINUX_SDK_DESCRIPTOR_DIR"] !== root || path.dirname(root) !== temp ||
      fs.realpathSync(paths.runnerTemp) !== paths.runnerTemp) {
    throw new Error("Native Linux SDK descriptor root must exactly match this workflow run");
  }
  const input = (/** @type {string} */ file, /** @type {string[]} */ roots, /** @type {string[] | undefined} */ expectedMembers = undefined,
    /** @type {boolean} */ recordMembers = false) => ({
    artifact: path.basename(file), file: path.join(incoming, path.basename(file)), roots,
    ...(expectedMembers ? {expectedMembers} : {}), ...(recordMembers ? {recordMembers: true} : {}),
  });
  const imageMembers = [images.handoffName, images.handoffName + "/notice-sources",
    ...["cuda-images.inc", "cuda-images.inc.json", "result.json"].map(file => images.handoffName + "/" + file),
    ...["manifest.json", ...cudaPackageNames.map(cudaCopyrightName), "nvrtc-12.6-LICENSE.txt", "nvrtc-13.3-LICENSE.txt",
      "nvptxcompiler-13.3-LICENSE.txt", "cutlass-4.6.1-LICENSE.txt"].map(file => images.handoffName + "/notice-sources/" + file)].sort();
  return {name, root, sdkFile: path.join(root, "sdk.json"), admissionFile: path.join(root, "admission.json"), archive, incoming,
    downloads: path.join(paths.runnerTemp, "downloads"),
    llvmSdk: path.join(paths.output, "sdk"),
    linux, dpcpp, gnu, images, jit, acpp, acppSdk, hermes, openmp, rocm, zstd, suffix: paths.suffix,
    ioSdk: paths.ioSdk, opensslSdk: paths.opensslSdk, runnerTemp: paths.runnerTemp,
    inputRows: [input(linux.sourceTar, [linux.sourceName]),
      input(dpcpp.sdkTar, [dpcpp.name]), input(gnu.sdkTar, [gnu.name], undefined, true), input(images.tar, [images.handoffName], imageMembers),
      input(jit.phaseTar("static-dpcpp"), [jit.name]), input(acpp.sourceTar, [acpp.sourceName]), input(acppSdk.sdkTar, [acppSdk.sdkName]),
      input(hermes.sdkTar, [hermes.sourceName, hermes.sdkName]), input(rocm.hostSdkTar,
        [rocm.sourceName, rocm.llvmSourceName, rocm.llvmSdkName, rocm.hipHeadersName, rocm.resourcesName, rocm.sdkName, rocm.hostSdkName]),
      input(openmp.sdkTar, [paths.sourceName, openmp.sdkName]), input(zstd.sdkTar, [zstd.sourceName, zstd.sdkName]),
      input(ioFinalTar(paths), [paths.sourceName, paths.ioSourceName, paths.ioSdkName])],
  };
}

/** Validate the exact current-run input tar path and root membership before restoring it.
 * @param {{artifact: string, file: string, roots: string[], expectedMembers?: string[], recordMembers?: boolean}} row @param {string} incoming */
function auditLinuxNativeSdkDescriptorInputArchive(row, incoming) {
  if (!row || path.dirname(row.file) !== incoming || path.basename(row.file) !== row.artifact || !row.artifact.endsWith(".tar") ||
      !Array.isArray(row.roots) || !row.roots.length || new Set(row.roots).size !== row.roots.length ||
      row.roots.some(name => !safeRelative(name) || name.includes("/"))) {
    throw new Error("Native SDK descriptor input archive path or root plan differs");
  }
  const stat = fs.lstatSync(row.file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || !Number.isSafeInteger(stat.size)) {
    throw new Error("Native SDK descriptor input is not a regular raw tar: " + row.artifact);
  }
  const members = listState(row.file, row.roots).map(name => name.endsWith("/") ? name.slice(0, -1) : name).sort(),
    expandedBytes = tarExpandedBytes(row.file, members.length);
  if (row.expectedMembers && !isDeepStrictEqual(members, [...row.expectedMembers].sort())) {
    throw new Error("Native SDK descriptor CUDA image artifact must contain exactly its include, sidecar and result");
  }
  return {artifact: row.artifact, bytes: stat.size, expandedBytes, sha256: fileHash(row.file), roots: [...row.roots],
    ...(row.expectedMembers || row.recordMembers ? {members} : {})};
}

/** Bound worst-case expanded input size from GNU tar's metadata before extracting same-run inputs.
 * Numeric ownership and full timestamps give a stable size column; member names are checked separately.
 * @param {string} archive @param {number} expectedMembers @param {number} [timeoutMs] */
function tarExpandedBytes(archive, expectedMembers, timeoutMs = 15 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 15 * 60 * 1000) {
    throw new Error("Native CI tar size-listing timeout is outside the bounded range");
  }
  if (!Number.isSafeInteger(expectedMembers) || expectedMembers < 1 || expectedMembers > 500000) {
    throw new Error("Native CI tar size-listing member count is outside the bounded range");
  }
  const listing = runTar("list", ["--list", "--verbose", "--numeric-owner", "--full-time", "--quoting-style=literal", "--file", archive], timeoutMs),
    lines = listing.split("\n").filter(Boolean);
  if (lines.length !== expectedMembers) {
    throw new Error("Native SDK tar size listing differs from its admitted member list");
  }
  let total = 0n;
  for (const line of lines) {
    const match = /^\S+\s+\S+\s+(\d+)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\s|$)/.exec(line);
    if (!match) {throw new Error("Native SDK tar size listing is malformed");}
    total += BigInt(match[1] || "0");
  }
  if (total > BigInt(Number.MAX_SAFE_INTEGER)) {throw new Error("Native SDK tar expanded size exceeds the safe limit");}
  return Number(total);
}

/** `expandedBytes` is the conservative sum of archive member sizes; input tars already occupy disk.
 * The app stage rechecks the free-space reserve after these roots have been restored. */
/** @param {number} expandedBytes @param {number} tempAvailableBytes @param {number} runnerTempAvailableBytes */
function linuxNativeSdkDiskPreflight(expandedBytes, tempAvailableBytes, runnerTempAvailableBytes) {
  if (![expandedBytes, tempAvailableBytes, runnerTempAvailableBytes].every(value => Number.isSafeInteger(value) && value >= 0)) {
    throw new Error("Native SDK disk preflight needs safe nonnegative byte counts");
  }
  const requiredTempBytes = expandedBytes + linuxNativeBuildReserveBytes;
  if (!Number.isSafeInteger(requiredTempBytes) || tempAvailableBytes < requiredTempBytes ||
      runnerTempAvailableBytes < linuxNativeBuildReserveBytes) {
    throw new Error("Insufficient disk for expanded native SDK inputs plus the bounded application-build reserve");
  }
  return {expandedInputBytes: expandedBytes, buildReserveBytes: linuxNativeBuildReserveBytes,
    requiredTempBytes, availableTempBytes: tempAvailableBytes, availableRunnerTempBytes: runnerTempAvailableBytes};
}

/** Read filesystem availability without rounding away the build reserve. @param {string} directory */
function availableDiskBytes(directory) {
  const stat = fs.statfsSync(directory, {bigint: true});
  if (stat.bsize <= 0n) {throw new Error("Cannot determine native SDK filesystem block size");}
  const bytes = stat.bavail * stat.bsize;
  if (bytes > BigInt(Number.MAX_SAFE_INTEGER)) {return Number.MAX_SAFE_INTEGER;}
  return Number(bytes);
}

/** @typedef {{artifact: string, file: string, roots: string[], expectedMembers?: string[], recordMembers?: boolean}} LinuxNativeSdkDescriptorInput */
/** @typedef {import("./native-linux-sdk-config").LinuxNativeSdkConfigState} LinuxNativeSdkConfigState */
/** @typedef {Omit<LinuxNativeSdkConfigState, "rocm"> & {rocm:Pick<ReturnType<typeof rocmHostStatePaths>,
 * "sdkRoot" | "hostSdkName" | "hostSdkRoot" | "hostSdkTar" | "runtimeOutput">, suffix: string, root: string, sdkFile: string, admissionFile: string,
 * gnu:{root:string,name:string,sdkTar:string}, incoming: string, downloads: string, inputRows: LinuxNativeSdkDescriptorInput[]}} LinuxNativeSdkDescriptorState */

/** Ensure the already restored descriptor remains bound to every current-run tar before compilation.
 * @param {LinuxNativeSdkDescriptorState} state
 * @param {{cc: string, cxx: string}} [tools] @param {string} [linker] */
function auditLinuxNativeSdkDescriptorLineage(state, tools = hermesBuildTools(), linker = linuxNativeSdkLldPath(tools.cxx)) {
  const rootStat = fs.lstatSync(state.root), admissionStat = fs.lstatSync(state.admissionFile), descriptorStat = fs.lstatSync(state.sdkFile),
    incomingStat = fs.lstatSync(state.incoming), admission = readBuildJson(state.admissionFile), config = readBuildJson(state.sdkFile);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(state.root) !== state.root ||
      !admissionStat.isFile() || admissionStat.isSymbolicLink() || !descriptorStat.isFile() || descriptorStat.isSymbolicLink() ||
      !incomingStat.isDirectory() || incomingStat.isSymbolicLink() || fs.realpathSync(state.incoming) !== state.incoming ||
      admission.format !== "mom-native-linux-sdk-descriptor-v1" || admission.runAttempt !== state.suffix ||
      admission.buildOnly !== true || admission.compiled !== false || admission.executed !== false || admission.publicationQualified !== false ||
      !hasExactKeys(admission.descriptor, ["path", "sha256"]) || admission.descriptor.path !== state.sdkFile ||
      admission.descriptor.sha256 !== fileHash(state.sdkFile) || !Array.isArray(admission.inputArchives) ||
      admission.inputArchives.length !== state.inputRows.length || !Array.isArray(admission.profiles) ||
      admission.profiles.slice().sort().join(",") !== "acpp-cuda,acpp-hip,dpcpp,dpcpp-opencl,oneapi") {
    throw new Error("Native application build requires the exact admitted same-run SDK descriptor");
  }
  const expectedNames = state.inputRows.map(row => row.artifact).sort();
  if (!isDeepStrictEqual(fs.readdirSync(state.incoming).sort(), expectedNames)) {
    throw new Error("Native application build inputs contain missing, stale or substituted same-run tars");
  }
  const inputArchives = state.inputRows.map((row, index) => {
    const record = admission.inputArchives[index], stat = fs.lstatSync(row.file);
    const hasMembers = Boolean(row.expectedMembers || row.recordMembers);
    if (!stat.isFile() || stat.isSymbolicLink() || !hasExactKeys(record,
      hasMembers ? ["artifact", "bytes", "expandedBytes", "sha256", "roots", "members"] : ["artifact", "bytes", "expandedBytes", "sha256", "roots"]) ||
        record.artifact !== row.artifact || record.bytes !== stat.size || record.sha256 !== fileHash(row.file) ||
        record.expandedBytes < 0 || !Number.isSafeInteger(record.expandedBytes) || !isDeepStrictEqual(record.roots, row.roots) ||
      row.expectedMembers && !isDeepStrictEqual(record.members, [...row.expectedMembers].sort()) ||
      row.recordMembers && !isDeepStrictEqual(record.members, listState(row.file, row.roots).map(name => name.endsWith("/") ? name.slice(0, -1) : name).sort())) {
      throw new Error("Native application build input differs from descriptor admission: " + row.artifact);
    }
    return {artifact: row.artifact, sha256: record.sha256};
  });
  const gnuNoticeSources = linuxNativeSdkGnuNoticeSources(state, admission.inputArchives);
  const rocmNoticeSources = linuxNativeSdkRocmNoticeSources(state, admission.inputArchives);
  if (!isDeepStrictEqual(admission.gnuNoticeSources, gnuNoticeSources) ||
      !isDeepStrictEqual(admission.rocmNoticeSources, rocmNoticeSources)) {
    throw new Error("Same-run package copyright/source-license lineage differs from descriptor admission");
  }
  const expectedConfig = linuxNativeSdkDescriptorConfig(state, tools, linker);
  const selectedLinkInputs = admission.selectedLinkInputs, selectedFiles = selectedLinkInputs?.files,
    selectedPaths = selectedFiles && typeof selectedFiles === "object" && !Array.isArray(selectedFiles) ? Object.keys(selectedFiles) : [];
  if (!hasExactKeys(selectedLinkInputs, ["files", "sha256"]) || !selectedFiles || typeof selectedFiles !== "object" ||
      Array.isArray(selectedFiles) || Object.getPrototypeOf(selectedFiles) !== Object.prototype || selectedPaths.length === 0 ||
      selectedPaths.some(file => file.length > 4096 || !path.isAbsolute(file) || path.resolve(file) !== file) ||
      Object.values(selectedFiles).some(hash => typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash)) ||
      typeof selectedLinkInputs.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(selectedLinkInputs.sha256) ||
      selectedLinkInputs.sha256 !== jsonHash(JSON.stringify(selectedFiles)) || !Object.hasOwn(selectedFiles, linker) ||
      fileHash(linker) !== selectedFiles[linker]) {
    throw new Error("Native application selected linker differs from descriptor admission");
  }
  if (!isDeepStrictEqual(config, expectedConfig) || !hasExactKeys(admission.diskPreflight,
    ["expandedInputBytes", "buildReserveBytes", "requiredTempBytes", "availableTempBytes", "availableRunnerTempBytes"]) ||
      admission.diskPreflight.expandedInputBytes !== admission.inputArchives.reduce((/** @type {number} */ sum,
        /** @type {{expandedBytes: number}} */ row) => sum + row.expandedBytes, 0) ||
      admission.diskPreflight.buildReserveBytes !== linuxNativeBuildReserveBytes ||
      admission.diskPreflight.requiredTempBytes !== admission.diskPreflight.expandedInputBytes + linuxNativeBuildReserveBytes ||
      admission.diskPreflight.availableTempBytes < admission.diskPreflight.requiredTempBytes ||
      admission.diskPreflight.availableRunnerTempBytes < linuxNativeBuildReserveBytes) {
    throw new Error("Native application build descriptor or disk preflight differs from current same-run inputs");
  }
  require("./build-native-release").nativeSdkConfig(config, repo);
  return {admissionHash: fileHash(state.admissionFile), descriptorHash: fileHash(state.sdkFile), inputArchives, gnuNoticeSources,
    rocmNoticeSources, config};
}

/** Bind the existing GNU package-copyright artifact to this descriptor without claiming complete notices.
 * @param {LinuxNativeSdkDescriptorState} state @param {Array<{artifact:string,sha256:string,roots:string[]}>} inputArchives */
function linuxNativeSdkGnuNoticeSources(state, inputArchives) {
  const audited = auditGnuHeaders(state.gnu), artifactName = path.basename(state.gnu.sdkTar),
    archive = inputArchives.find(row => row.artifact === artifactName), resultPath = path.join(state.gnu.root, "result.json"),
    copyrightsPath = path.join(state.gnu.root, "copyrights.json");
  if (!archive || !archive.roots.includes(state.gnu.name) || !/^[a-f0-9]{64}$/.test(archive.sha256 || "")) {
    throw new Error("GNU notice-source handoff is missing its admitted same-run header archive");
  }
  const files = audited.notices.map(notice => {
    const relative = path.relative(state.gnu.root, notice.path).split(path.sep).join("/"), stat = fs.lstatSync(notice.path);
    if (!safeRelative(relative) || relative.startsWith("../") || !stat.isFile() || stat.isSymbolicLink() ||
        fileHash(notice.path) !== notice.sha256) {throw new Error("GNU notice-source path is not a verified package copyright file");}
    return {package: notice.package, version: notice.version, file: relative, sha256: notice.sha256, bytes: stat.size};
  });
  return {format: "mom-linux-gnu-copyright-sources-v1", completeNativeNotices: false,
    artifact: {name: artifactName, sha256: archive.sha256},
    headerReceipt: {file: "result.json", sha256: fileHash(resultPath)},
    copyrightManifest: {file: "copyrights.json", sha256: fileHash(copyrightsPath)}, files};
}

/** Bind package-owned ROCm and hsakmt notice bodies to the admitted same-run ROCm SDK tar.
 * @param {LinuxNativeSdkDescriptorState} state @param {Array<{artifact:string,sha256:string,roots:string[]}>} inputArchives */
function linuxNativeSdkRocmNoticeSources(state, inputArchives) {
  const artifactName = path.basename(state.rocm.hostSdkTar), archive = inputArchives.find(row => row.artifact === artifactName);
  if (!archive || !archive.roots.includes(state.rocm.hostSdkName) || !/^[a-f0-9]{64}$/.test(archive.sha256 || "")) {
    throw new Error("ROCm notice-source handoff is missing its admitted same-run host SDK archive");
  }
  return {...auditRocmHostNoticeSources(state.rocm), artifact: {name: artifactName, sha256: archive.sha256}};
}

/** Fresh output for the public Core/five-factory/application frontdoor; SDK roots stay in place. */
/** @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths */
function linuxNativeApplicationStatePaths(paths) {
  const name = "mom-native-linux-application-" + paths.suffix, root = path.join(temp, name),
    archive = path.join(paths.runnerTemp, "native-linux-application-" + paths.suffix + ".tar"),
    configFile = path.join(paths.runnerTemp, "native-linux-assembly-config-" + paths.suffix + ".json");
  if (path.dirname(root) !== temp || path.dirname(archive) !== paths.runnerTemp || path.dirname(configFile) !== paths.runnerTemp ||
      root === archive || root.startsWith(archive + path.sep) || root === configFile || root.startsWith(configFile + path.sep)) {
    throw new Error("Native Linux application output paths differ from the run-scoped plan");
  }
  return {name, root, archive, configFile};
}

const linuxNativeGpuProfiles = Object.freeze(["acpp-cuda", "acpp-hip", "dpcpp", "dpcpp-opencl", "oneapi"]);
const linuxNativeComponentProfiles = Object.freeze(["core", ...linuxNativeGpuProfiles]);

/** Read the six producer-emitted component archive hashes from job outputs.
 * @param {NodeJS.ProcessEnv} [environment] @returns {Record<string,string>} */
function linuxNativeComponentArchivePins(environment = process.env) {
  const pins = Object.fromEntries(linuxNativeComponentProfiles.map(profile => {
    const key = "MOM_NATIVE_LINUX_COMPONENT_" + profile.replaceAll("-", "_").toUpperCase() + "_SHA256",
      sha256 = environment[key] || "";
    if (!/^[a-f0-9]{64}$/.test(sha256)) {throw new Error("Native application stage requires all six valid external SHA-256 pins");}
    return [profile, sha256];
  }));
  return pins;
}

/** Verify the producer's complete raw component tar before any parser or extractor reads it.
 * @param {string} archive @param {string} expectedSha256 */
function linuxNativeComponentArchiveSha256(archive, expectedSha256) {
  if (!/^[a-f0-9]{64}$/.test(expectedSha256 || "") || !path.isAbsolute(archive) || path.resolve(archive) !== archive) {
    throw new Error("Native component archive requires an exact external SHA-256 pin and canonical path");
  }
  const stat = fs.lstatSync(archive);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size < 1 || stat.size > 16 * 1024 ** 3 ||
      fs.realpathSync(archive) !== archive) {
    throw new Error("Native component archive is not a bounded canonical singly-linked regular file");
  }
  if (fileHash(archive) !== expectedSha256) {throw new Error("Native component archive differs from its external SHA-256 pin");}
  return expectedSha256;
}

/** Verify the external whole-tar pin before checking its exact root and manifest member.
 * @param {{name:string}} state @param {string} archive @param {string} expectedSha256 */
function auditLinuxNativeComponentArchive(state, archive, expectedSha256) {
  const archiveSha256 = linuxNativeComponentArchiveSha256(archive, expectedSha256), members = listState(archive, [state.name]);
  if (!members.includes(state.name + "/component.json")) {throw new Error("Native component archive is missing its producer manifest");}
  return {archiveSha256, members};
}

/** Per-job component roots and handoffs keep producer paths unique across run attempts.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths @param {string} profile */
function linuxNativeComponentStatePaths(paths, profile) {
  if (profile !== "core" && !linuxNativeGpuProfiles.includes(profile)) {throw new Error("Select the Linux Core or one exact GPU factory profile");}
  const name = "mom-native-linux-component-" + profile + "-" + paths.suffix,
    root = path.join(temp, name), archive = path.join(paths.runnerTemp, "native-linux-component-" + profile + "-" + paths.suffix + ".tar");
  if (!/^mom-native-linux-component-[a-z0-9-]+-[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(name) || path.dirname(root) !== temp ||
      path.resolve(root) !== root || path.dirname(archive) !== paths.runnerTemp || path.resolve(archive) !== archive ||
      root === archive || root.startsWith(archive + path.sep)) {throw new Error("Native component paths differ from the run-scoped plan");}
  return {profile, name, root, archive, incoming: path.join(paths.runnerTemp, "incoming"), downloads: path.join(paths.runnerTemp, "downloads")};
}

/** Shrink each synchronous child timeout against the one shared component deadline.
 * @param {number} deadlineMs @param {number} maximumMs @param {number} [nowMs] */
function linuxNativeComponentChildTimeout(deadlineMs, maximumMs, nowMs = Date.now()) {
  if (!Number.isSafeInteger(deadlineMs) || !Number.isSafeInteger(maximumMs) || maximumMs < 1 ||
      !Number.isSafeInteger(nowMs) || nowMs < 0) {throw new Error("Native component child deadline is malformed");}
  const timeout = Math.min(maximumMs, deadlineMs - nowMs);
  if (!Number.isSafeInteger(timeout) || timeout < 1000) {throw new Error("Native component deadline expired before the next child stage");}
  return timeout;
}

/** Canonical component map consumed by the public release frontdoor. @param {string} core @param {Record<string,string>} gpu */
function linuxNativeAssemblyComponents(core, gpu) {
  if (typeof core !== "string" || !path.isAbsolute(core) || !gpu || typeof gpu !== "object" || Array.isArray(gpu) ||
      Object.keys(gpu).sort().join(",") !== linuxNativeGpuProfiles.slice().sort().join(",") ||
      Object.values(gpu).some(file => typeof file !== "string" || !path.isAbsolute(file))) {
    throw new Error("Native assembly requires the exact Core and five same-run GPU factory receipts");
  }
  return {core, gpu: Object.fromEntries(linuxNativeGpuProfiles.map(profile => [profile, gpu[profile] || ""]))};
}

/** Only the complete producer-admitted component map may be added to the original descriptor.
 * @param {unknown} descriptor @param {{core:string,gpu:Record<string,string>}} components */
function linuxNativeAssemblyConfig(descriptor, components) {
  if (!descriptor || typeof descriptor !== "object" || Array.isArray(descriptor) || Object.hasOwn(descriptor, "components")) {
    throw new Error("Native assembly requires the unchanged admitted source SDK descriptor");
  }
  const selected = linuxNativeAssemblyComponents(components.core, components.gpu);
  return {...descriptor, components: selected};
}

/** Hash a bounded regular-file tree without following links or special files. @param {string} root */
function linuxNativeComponentFiles(root) {
  /** @type {Array<{file:string,bytes:number,sha256:string}>} */
  const rows = [];
  /** @param {string} directory @param {string} relative */
  function visit(directory, relative) {
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(directory, entry.name), rel = relative ? relative + "/" + entry.name : entry.name,
        stat = fs.lstatSync(file);
      if (!safeRelative(rel) || stat.isSymbolicLink()) {throw new Error("Native component output contains an unsafe path");}
      if (stat.isDirectory()) {visit(file, rel);} else if (stat.isFile() && Number.isSafeInteger(stat.size)) {
        rows.push({file: rel, bytes: stat.size, sha256: fileHash(file)});
      } else {throw new Error("Native component output contains a non-regular file");}
      if (rows.length > 100000) {throw new Error("Native component output file count exceeds the handoff limit");}
    }
  }
  visit(root, "");
  return rows.sort((a, b) => a.file < b.file ? -1 : a.file > b.file ? 1 : 0);
}

/** Bind one immutable Core/factory result tree to its source descriptor and original SDK tars.
 * @param {{profile:string,name:string,root:string,archive:string}} state @param {string} suffix
 * @param {string} descriptorHash @param {string} admissionHash @param {Array<{artifact:string,sha256:string}>} inputArchives
 * @param {{receipt:string,sha256:string,object:string,objectSha256:string}} producer */
function writeLinuxNativeComponentManifest(state, suffix, descriptorHash, admissionHash, inputArchives, producer) {
  if (fs.lstatSync(state.root).isSymbolicLink() || !fs.statSync(state.root).isDirectory() ||
      !/^[a-f0-9]{64}$/.test(descriptorHash) || !/^[a-f0-9]{64}$/.test(admissionHash) ||
      !producer || !safeRelative(producer.receipt) || !/^[a-f0-9]{64}$/.test(producer.sha256) ||
      !path.isAbsolute(producer.object) || !/^[a-f0-9]{64}$/.test(producer.objectSha256) ||
      !Array.isArray(inputArchives) || inputArchives.length !== 12 || inputArchives.some(row => !row || !/^[a-f0-9]{64}$/.test(row.sha256))) {
    throw new Error("Native component receipt inputs are incomplete or malformed");
  }
  const receipt = path.join(state.root, producer.receipt), object = producer.object, objectRelative = path.relative(state.root, object);
  if (path.relative(state.root, receipt).startsWith("..") || !objectRelative || path.isAbsolute(objectRelative) ||
      objectRelative === ".." || objectRelative.startsWith(".." + path.sep) || fileHash(receipt) !== producer.sha256 ||
      fileHash(object) !== producer.objectSha256) {
    throw new Error("Native component producer receipt or object changed before handoff");
  }
  const manifest = {format: "mom-native-linux-component-v1", runAttempt: suffix, profile: state.profile,
    descriptor: {sha256: descriptorHash}, admission: {sha256: admissionHash},
    inputArchives: inputArchives.map(row => ({artifact: row.artifact, sha256: row.sha256})),
    producer: {receipt: producer.receipt, sha256: producer.sha256, object, objectSha256: producer.objectSha256},
    files: linuxNativeComponentFiles(state.root), buildOnly: true, publicationQualified: false};
  fs.writeFileSync(path.join(state.root, "component.json"), JSON.stringify(manifest, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return manifest;
}

/** Move the exact run-scoped artifact set into disjoint descriptor, SDK and component paths.
 * @param {ReturnType<typeof ciPaths>} paths @param {string[]} [profiles] */
function partitionLinuxNativeDownloads(paths, profiles = []) {
  if (new Set(profiles).size !== profiles.length || profiles.some(profile => profile !== "core" && !linuxNativeGpuProfiles.includes(profile))) {
    throw new Error("Native component download plan contains an unknown or duplicate GPU profile");
  }
  const state = linuxNativeSdkDescriptorStatePaths(paths, paths.env), componentIncoming = path.join(paths.runnerTemp, "component-incoming"),
    descriptorName = path.basename(state.archive), componentNames = profiles.map(profile => "native-linux-component-" + profile + "-" + paths.suffix + ".tar"),
    expected = [descriptorName, ...state.inputRows.map(row => row.artifact), ...componentNames].sort();
  const downloadsStat = fs.lstatSync(state.downloads, {throwIfNoEntry: false});
  if (!downloadsStat?.isDirectory() || downloadsStat.isSymbolicLink() || fs.realpathSync(state.downloads) !== state.downloads ||
      !isDeepStrictEqual(fs.readdirSync(state.downloads).sort(), expected) ||
      [state.incoming, state.archive, componentIncoming].some(file => fs.lstatSync(file, {throwIfNoEntry: false}))) {
    throw new Error("Native run handoff is missing, stale, or contains substituted artifacts");
  }
  fs.mkdirSync(state.incoming, {mode: 0o700});
  if (profiles.length) {fs.mkdirSync(componentIncoming, {mode: 0o700});}
  for (const name of expected) {
    const source = path.join(state.downloads, name), stat = fs.lstatSync(source);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1) {throw new Error("Native artifact handoff contains a non-regular archive: " + name);}
    const destination = name === descriptorName ? state.archive : componentNames.includes(name) ? path.join(componentIncoming, name) : path.join(state.incoming, name);
    fs.renameSync(source, destination);
  }
  fs.rmdirSync(state.downloads);
  return {descriptor: state, componentIncoming, componentArchives: Object.fromEntries(profiles.map(profile => [profile,
    path.join(componentIncoming, "native-linux-component-" + profile + "-" + paths.suffix + ".tar")]))};
}

/** Re-admit descriptor and original input-tar hashes from the preceding same-run producer job.
 * The descriptor job already ran all producer audits; here public builders revalidate their selected inputs.
 * @param {ReturnType<typeof linuxNativeSdkDescriptorStatePaths>} state
 * @param {Pick<ReturnType<typeof ciPaths>, "env" | "runnerTemp">} paths
 * @param {ReturnType<typeof auditLinuxNativeSdkDescriptorInputArchives>} [auditedArchives] */
function restoreLinuxNativeDescriptorHandoff(state, paths, auditedArchives = undefined) {
  const archiveStat = fs.lstatSync(state.archive, {throwIfNoEntry: false});
  if (fs.lstatSync(state.root, {throwIfNoEntry: false}) || !archiveStat?.isFile() || archiveStat.isSymbolicLink() ||
      fs.realpathSync(state.archive) !== state.archive) {throw new Error("Native component stage needs a fresh descriptor root and the exact partitioned descriptor tar");}
  const archives = auditedArchives || auditLinuxNativeSdkDescriptorInputArchives(state), expanded = archives.reduce((sum, row) => sum + row.expandedBytes, 0),
    disk = linuxNativeSdkDiskPreflight(expanded, availableDiskBytes(temp), availableDiskBytes(state.runnerTemp));
  for (const row of archives) {
    const source = state.inputRows.find(item => item.artifact === row.artifact), stat = source && fs.lstatSync(source.file);
    if (!source || !stat?.isFile() || stat.isSymbolicLink() || stat.size !== row.bytes || fileHash(source.file) !== row.sha256) {
      throw new Error("Native SDK handoff changed after preflight: " + row.artifact);
    }
  }
  extractState(state.archive, temp, [state.name]);
  restoreLinuxNativeSdkDescriptorInputs(state);
  const admission = readBuildJson(state.admissionFile), config = readBuildJson(state.sdkFile),
    rowsMatch = Array.isArray(admission.inputArchives) && admission.inputArchives.length === archives.length &&
      archives.every((actual, index) => isDeepStrictEqual(actual, admission.inputArchives[index]));
  if (admission.format !== "mom-native-linux-sdk-descriptor-v1" || admission.runAttempt !== state.suffix ||
      admission.buildOnly !== true || admission.compiled !== false || admission.executed !== false || admission.publicationQualified !== false ||
      !hasExactKeys(admission.descriptor, ["path", "sha256"]) || admission.descriptor.path !== state.sdkFile ||
      admission.descriptor.sha256 !== fileHash(state.sdkFile) || !rowsMatch ||
      !isDeepStrictEqual(admission.profiles, linuxNativeGpuProfiles.slice().sort()) || Object.hasOwn(config, "components")) {
    throw new Error("Native component stage requires the exact admitted same-run SDK descriptor and original archives");
  }
  prepareRocmHostAdmission(paths, state.rocm);
  // Public config admission checks the complete six-profile graph and all shared link inputs;
  // keep the original descriptor intact rather than inventing a weaker per-profile subset.
  const {nativeSdkConfig, nativeSdkLinkInputs} = require("./build-native-release"), selected = nativeSdkConfig(config, repo), linked = nativeSdkLinkInputs(selected.sdk),
    hashes = Object.fromEntries(Object.entries(linked.hashes).sort(([a], [b]) => a.localeCompare(b))),
    gnuNoticeSources = linuxNativeSdkGnuNoticeSources(state, archives),
    rocmNoticeSources = linuxNativeSdkRocmNoticeSources(state, archives);
  if (selected.keys.slice().sort().join(",") !== linuxNativeGpuProfiles.slice().sort().join(",") ||
      !hasExactKeys(admission.selectedLinkInputs, ["files", "sha256"]) ||
      !isDeepStrictEqual(admission.selectedLinkInputs.files, hashes) ||
      admission.selectedLinkInputs.sha256 !== jsonHash(JSON.stringify(hashes)) ||
      !isDeepStrictEqual(admission.sourceReceipts, linked.sources) ||
      !isDeepStrictEqual(admission.gnuNoticeSources, gnuNoticeSources) ||
      !isDeepStrictEqual(admission.rocmNoticeSources, rocmNoticeSources) ||
      !hasExactKeys(admission.diskPreflight, ["expandedInputBytes", "buildReserveBytes", "requiredTempBytes", "availableTempBytes", "availableRunnerTempBytes"]) ||
      admission.diskPreflight.expandedInputBytes !== expanded || admission.diskPreflight.buildReserveBytes !== linuxNativeBuildReserveBytes ||
      admission.diskPreflight.requiredTempBytes !== expanded + linuxNativeBuildReserveBytes) {
    throw new Error("Restored component SDK inputs differ from the original descriptor admission");
  }
  return {state, admission, config, selected, linked, archives, disk, descriptorHash: fileHash(state.sdkFile),
    admissionHash: fileHash(state.admissionFile), rocmNoticeSources};
}

/** Run exactly one maintained Core or GPU factory builder and retain its original outputs unchanged.
 * @param {ReturnType<typeof ciPaths>} paths @param {string} profile */
function buildLinuxNativeComponent(paths, profile) {
  const descriptorState = linuxNativeSdkDescriptorStatePaths(paths, paths.env), component = linuxNativeComponentStatePaths(paths, profile),
    outputs = [component.root, component.archive];
  if (outputs.some(file => fs.lstatSync(file, {throwIfNoEntry: false}))) {throw new Error("Native component output already exists");}
  partitionLinuxNativeDownloads(paths);
  const preflightArchives = auditLinuxNativeSdkDescriptorInputArchives(descriptorState);
  linuxNativeSdkDiskPreflight(preflightArchives.reduce((sum, row) => sum + row.expandedBytes, 0),
    availableDiskBytes(temp), availableDiskBytes(paths.runnerTemp));
  const epoch = paths.env["MOM_NATIVE_COMPONENT_DEADLINE_EPOCH"] || "", remainingMs = Number(epoch) * 1000 - Date.now();
  if (!/^\d{10}$/.test(epoch) || !Number.isSafeInteger(remainingMs) || remainingMs < 10 * 60 * 1000 || remainingMs > 60 * 60 * 1000) {
    throw new Error("Native component job deadline is malformed, expired, or lacks build time");
  }
  provisionOneApi(paths, "/var/tmp/mom-dev-native-component-" + paths.suffix,
    linuxNativeComponentChildTimeout(Number(epoch) * 1000, 15 * 60 * 1000, Date.now() + 10 * 60 * 1000));
  const handoff = restoreLinuxNativeDescriptorHandoff(descriptorState, paths, preflightArchives);
  fs.mkdirSync(component.root, {mode: 0o700}); fs.mkdirSync(path.join(component.root, "logs"));
  const {nativeStage} = require("./build-native-release"), environment = {...paths.env, MOM_BUILD_JOBS: ciBuildJobs(paths.env)},
    buildTimeout = Math.min(35 * 60 * 1000, Number(epoch) * 1000 - Date.now() - 5 * 60 * 1000);
  if (!Number.isSafeInteger(buildTimeout) || buildTimeout < 5 * 60 * 1000) {throw new Error("Native component deadline leaves insufficient bounded build time");}
  const buildDeadline = Date.now() + buildTimeout;
  /** @param {number} maximumMs */
  const timedSpawn = maximumMs => /** @type {typeof spawnSync} */ ((command, args, options) => spawnSync(command, args,
    {...options, timeout: linuxNativeComponentChildTimeout(buildDeadline, maximumMs)}));
  let receipt, object;
  if (profile === "core") {
    const core = path.join(component.root, "core");
    nativeStage({env: {MOM_LINUX_BUILD_COMPILER: "oneapi", MOM_GPU_BACKEND: "intel", MOM_PORTABLE_BUILD: "1"}},
      "bash", [path.join(repo, "scripts/multicompiler-entrypoint.sh"), "true"], environment, repo, component.root, "engineering-cache",
      timedSpawn(Math.min(12 * 60 * 1000, buildTimeout)));
    nativeStage(handoff.config.cpu, process.execPath, ["--max-old-space-size=256", path.join(repo, "scripts/build-native-cpu-probe.js"),
      handoff.config.cpu.cache, core, "--gpu"], environment, repo, component.root, "core", timedSpawn(buildTimeout));
    receipt = path.join(core, "inputs.json"); object = path.join(core, "cpu-core.a");
    const input = readBuildJson(receipt);
    if (input.gpu !== true || input.portable !== true || input.objectSha256 !== fileHash(object)) {
      throw new Error("Native Core producer did not retain the original portable Core receipt");
    }
  } else {
    const output = path.join(component.root, "gpu", profile), selected = handoff.selected, runner = selected.sdk.gpu[profile],
      isAcpp = profile.startsWith("acpp-"), script = isAcpp ? "build-native-gpu-probe.js" : "build-native-dpcpp-probe.js";
    if (!runner) {throw new Error("Admitted SDK descriptor is missing the selected GPU factory: " + profile);}
    const stageRunner = {...runner, env: {...runner.env, ...(isAcpp ? {MOM_AOT_SYCL_BACKEND: profile.slice(5)} : {MOM_AOT_DPCPP_PROFILE: profile})}};
    fs.mkdirSync(path.dirname(output), {recursive: true});
    nativeStage(stageRunner, process.execPath, ["--max-old-space-size=256", path.join(repo, "scripts", script), ...runner.args, output],
      environment, repo, component.root, profile, timedSpawn(buildTimeout));
    receipt = path.join(output, "result.json");
    const record = readBuildJson(receipt);
    if (record.factory !== "mom_native_gpu_" + profile.replaceAll("-", "_") || !path.isAbsolute(record.object) ||
        !fs.statSync(record.object).isFile()) {throw new Error("Native GPU factory producer receipt is incomplete: " + profile);}
    if (handoff.linked.cudaImageInputs && (profile === "dpcpp" || profile === "acpp-cuda")) {
      require("./probe-native-controller").verifyCudaImageReceiptInputs(record, handoff.linked.cudaImageInputs, profile);
    }
    object = record.object;
  }
  const producer = {receipt: path.relative(component.root, receipt), sha256: fileHash(receipt), object,
      objectSha256: fileHash(object)}, inputArchives = handoff.archives.map(row => ({artifact: row.artifact, sha256: row.sha256}));
  writeLinuxNativeComponentManifest(component, paths.suffix, handoff.descriptorHash, handoff.admissionHash, inputArchives, producer);
  packState(component.archive, temp, [component.name]);
  const members = listState(component.archive, [component.name]);
  if (!members.includes(component.name + "/component.json") || fileHash(receipt) !== producer.sha256 || fileHash(object) !== producer.objectSha256) {
    throw new Error("Native component artifact changed during same-run packaging");
  }
  console.log(`[native-ci] Built only ${profile} with its maintained public producer; no executable ran or was publication-qualified (${members.length} members).`);
}

/** Verify an extracted component tree and its exact tar member set before app assembly.
 * @param {{profile:string,name:string,root:string,admissionFile:string}} state
 * @param {{runAttempt:string,descriptor:{sha256:string}}} descriptorAdmission
 * @param {Array<{artifact:string,sha256:string}>} inputArchives @param {string[]} archiveMembers */
function auditLinuxNativeComponentTree(state, descriptorAdmission, inputArchives, archiveMembers) {
  const rootStat = fs.lstatSync(state.root), manifestFile = path.join(state.root, "component.json"), manifestStat = fs.lstatSync(manifestFile),
    manifest = readBuildJson(manifestFile);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(state.root) !== state.root ||
      !manifestStat.isFile() || manifestStat.isSymbolicLink() || !hasExactKeys(manifest,
    ["format", "runAttempt", "profile", "descriptor", "admission", "inputArchives", "producer", "files", "buildOnly", "publicationQualified"]) ||
      manifest.format !== "mom-native-linux-component-v1" || manifest.runAttempt !== descriptorAdmission.runAttempt ||
      manifest.profile !== state.profile || manifest.buildOnly !== true || manifest.publicationQualified !== false ||
      !hasExactKeys(manifest.descriptor, ["sha256"]) || manifest.descriptor.sha256 !== descriptorAdmission.descriptor.sha256 ||
      !hasExactKeys(manifest.admission, ["sha256"]) || manifest.admission.sha256 !== fileHash(state.admissionFile) ||
      !isDeepStrictEqual(manifest.inputArchives, inputArchives) || !Array.isArray(manifest.files) ||
      !hasExactKeys(manifest.producer, ["receipt", "sha256", "object", "objectSha256"]) ||
      typeof manifest.producer.receipt !== "string" || !/^[a-f0-9]{64}$/.test(manifest.producer.sha256 || "") ||
      typeof manifest.producer.object !== "string" || !path.isAbsolute(manifest.producer.object) ||
      !/^[a-f0-9]{64}$/.test(manifest.producer.objectSha256 || "")) {
    throw new Error("Native component tree is not bound to this exact SDK descriptor and run attempt");
  }
  const rows = /** @type {Array<{file:string,bytes:number,sha256:string}>} */ (manifest.files);
  if (rows.some(row => !hasExactKeys(row, ["file", "bytes", "sha256"]) || !safeRelative(row.file) ||
      !Number.isSafeInteger(row.bytes) || row.bytes < 0 || !/^[a-f0-9]{64}$/.test(row.sha256)) ||
      rows.map(row => row.file).sort().join("\0") !== rows.map(row => row.file).join("\0") ||
      rows.some(row => row.file === "component.json")) {
    throw new Error("Native component file manifest is malformed or unsorted");
  }
  const current = linuxNativeComponentFiles(state.root).filter(row => row.file !== "component.json");
  if (!isDeepStrictEqual(current, rows)) {throw new Error("Native component files changed after the producer handoff");}
  const expectedMembers = new Set([state.name, state.name + "/component.json"]);
  for (const row of rows) {
    const parts = row.file.split("/");
    for (let index = 1; index < parts.length; index++) {expectedMembers.add(state.name + "/" + parts.slice(0, index).join("/"));}
    expectedMembers.add(state.name + "/" + row.file);
  }
  const actualMembers = archiveMembers.map(name => name.endsWith("/") ? name.slice(0, -1) : name).sort();
  if (!isDeepStrictEqual(actualMembers, [...expectedMembers].sort())) {throw new Error("Native component tar contains extra, missing or substituted members");}
  const receipt = path.join(state.root, manifest.producer.receipt), object = path.resolve(manifest.producer.object),
    objectRelative = path.relative(state.root, object);
  if (!safeRelative(manifest.producer.receipt) || path.relative(state.root, receipt).startsWith("..") ||
      !objectRelative || path.isAbsolute(objectRelative) || objectRelative === ".." || objectRelative.startsWith(".." + path.sep) ||
      fileHash(receipt) !== manifest.producer.sha256 || fileHash(object) !== manifest.producer.objectSha256) {
    throw new Error("Native component producer receipt or compiled object changed after handoff");
  }
  if (state.profile === "core") {
    const input = readBuildJson(receipt);
    if (manifest.producer.receipt !== "core/inputs.json" || object !== path.join(state.root, "core/cpu-core.a") ||
        input.gpu !== true || input.portable !== true || input.objectSha256 !== manifest.producer.objectSha256) {
      throw new Error("Native Core handoff is not the maintained portable Core producer output");
    }
  } else {
    const record = readBuildJson(receipt);
    if (manifest.producer.receipt !== "gpu/" + state.profile + "/result.json" ||
        record.factory !== "mom_native_gpu_" + state.profile.replaceAll("-", "_") || record.object !== object) {
      throw new Error("Native GPU handoff does not contain its exact maintained factory receipt");
    }
  }
  return {state, manifest, manifestSha256: fileHash(manifestFile), producer: manifest.producer, files: current, members: actualMembers};
}

/** @typedef {{state:ReturnType<typeof linuxNativeComponentStatePaths>,archive:string,members:string[],archiveSha256:string,expandedBytes:number}} LinuxNativeComponentArchiveAdmission */

/** Restore and admit all six original producer outputs without rewriting their receipts.
 * @param {ReturnType<typeof ciPaths>} paths @param {ReturnType<typeof partitionLinuxNativeDownloads>} downloads
 * @param {{state:ReturnType<typeof linuxNativeSdkDescriptorStatePaths>,admission:{runAttempt:string,descriptor:{sha256:string}},
 * descriptorHash:string,admissionHash:string,archives:Array<{artifact:string,sha256:string}>}} descriptor
 * @param {Record<string,string>} archivePins @param {Map<string,LinuxNativeComponentArchiveAdmission>} preflight */
function restoreLinuxNativeComponents(paths, downloads, descriptor, archivePins, preflight) {
  const expectedNames = linuxNativeComponentProfiles.map(profile => "native-linux-component-" + profile + "-" + paths.suffix + ".tar").sort();
  if (!isDeepStrictEqual(fs.readdirSync(downloads.componentIncoming).sort(), expectedNames)) {
    throw new Error("Native assembly requires exactly one Core and five same-run GPU component archives");
  }
  if (!(preflight instanceof Map) || !isDeepStrictEqual([...preflight.keys()].sort(), linuxNativeComponentProfiles.slice().sort())) {
    throw new Error("Native assembly component archive preflight is incomplete");
  }
  let expanded = 0;
  /** @type {Map<string, {state:ReturnType<typeof linuxNativeComponentStatePaths> & {admissionFile:string},archive:string,members:string[],archiveSha256:string,admitted?:ReturnType<typeof auditLinuxNativeComponentTree>}>} */
  const archives = new Map();
  for (const profile of linuxNativeComponentProfiles) {
    const componentPaths = linuxNativeComponentStatePaths(paths, profile),
      state = {...componentPaths, admissionFile: descriptor.state.admissionFile}, archive = path.join(downloads.componentIncoming, path.basename(state.archive)),
      checked = preflight.get(profile);
    if (!checked || checked.archive !== archive || checked.state.name !== state.name || checked.archiveSha256 !== archivePins[profile]) {
      throw new Error("Native component archive preflight differs from its external producer pin: " + profile);
    }
    expanded += checked.expandedBytes;
    archives.set(profile, {state, archive, members: checked.members, archiveSha256: archivePins[profile]});
  }
  const disk = linuxNativeSdkDiskPreflight(expanded, availableDiskBytes(temp), availableDiskBytes(paths.runnerTemp));
  for (const profile of linuxNativeComponentProfiles) {
    const row = archives.get(profile);
    if (!row) {throw new Error("Native component archive table is incomplete");}
    linuxNativeComponentArchiveSha256(row.archive, row.archiveSha256);
    extractState(row.archive, temp, [row.state.name]);
    const manifestFile = path.join(row.state.root, "component.json"), manifest = readBuildJson(manifestFile),
      lineage = descriptor.archives.map(input => ({artifact: input.artifact, sha256: input.sha256}));
    if (!hasExactKeys(manifest, ["format", "runAttempt", "profile", "descriptor", "admission", "inputArchives", "producer", "files", "buildOnly", "publicationQualified"]) ||
        manifest.format !== "mom-native-linux-component-v1" || manifest.profile !== profile ||
        manifest.descriptor?.sha256 !== descriptor.descriptorHash || manifest.admission?.sha256 !== descriptor.admissionHash ||
        !isDeepStrictEqual(manifest.inputArchives, lineage)) {throw new Error("Component archive lineage does not match the same-run SDK descriptor");}
    if (fileHash(row.archive) !== row.archiveSha256) {throw new Error("Native component archive changed during restoration");}
    row.admitted = auditLinuxNativeComponentTree(row.state, descriptor.admission, lineage, row.members);
  }
  return {archives, disk};
}

/** The only compilation entrypoint for this CI stage is the maintained release builder.
 * @param {LinuxNativeSdkDescriptorState} descriptor
 * @param {ReturnType<typeof linuxNativeApplicationStatePaths>} output @param {string} deadlineEpoch @param {number} [nowMs] */
function linuxNativeApplicationBuildCommand(descriptor, output, deadlineEpoch, nowMs = Date.now()) {
  if (!path.isAbsolute(descriptor.sdkFile) || !path.isAbsolute(output.root) || path.dirname(output.root) !== temp ||
      path.resolve(output.root) !== output.root || output.root === descriptor.root || output.root.startsWith(descriptor.root + path.sep) ||
      descriptor.inputRows.some(row => row.roots.some(root => output.root === path.join(temp, root) || output.root.startsWith(path.join(temp, root) + path.sep)))) {
    throw new Error("Native application output must be a new path outside every admitted SDK input");
  }
  if (!/^\d{10}$/.test(deadlineEpoch) || !Number.isSafeInteger(nowMs) || nowMs < 0) {
    throw new Error("Native application stage needs its valid job-start deadline");
  }
  const remainingMs = Number(deadlineEpoch) * 1000 - nowMs;
  if (remainingMs < 5 * 60 * 1000) {throw new Error("Less than five minutes remain in the native application job budget");}
  return {script: "scripts/build-release.js", args: ["--native", descriptor.sdkFile, output.root], timeoutMs: Math.min(50 * 60 * 1000, remainingMs)};
}

/** Admit the unpacked release-frontdoor entrypoint and its original producer receipt.
 * @param {LinuxNativeSdkDescriptorState} descriptor
 * @param {ReturnType<typeof linuxNativeApplicationStatePaths>} output
 * @param {string} expectedConfigHash */
function auditLinuxNativeApplicationEntrypoint(descriptor, output, expectedConfigHash) {
  if (!path.isAbsolute(output.root) || path.dirname(output.root) !== temp || path.resolve(output.root) !== output.root ||
      !safeRelative(output.name) || output.name.includes("/") || output.name.includes("\\") ||
      output.name !== path.basename(output.root) || output.root !== path.join(temp, output.name)) {
    throw new Error("Native application output must be the exact run-scoped canonical directory");
  }
  const outputStat = fs.lstatSync(output.root);
  if (!outputStat.isDirectory() || outputStat.isSymbolicLink() || fs.realpathSync(output.root) !== output.root) {
    throw new Error("Native application output root must be a canonical directory");
  }
  const binary = path.join(output.root, "mom"), producerFile = path.join(output.root, "result.json"),
    entrypointFile = path.join(output.root, "entrypoint.json");
  const descriptorHash = fileHash(descriptor.sdkFile);
  if (descriptorHash !== expectedConfigHash) {throw new Error("Native application descriptor changed during the build");}
  // Source-only prerequisite stages must not import runtime tests or their hash vectors.
  const admitted = require("./check-native-release").admitCandidate(entrypointFile, "linux", "x64"), producer = readBuildJson(producerFile),
    entrypoint = admitted.entryRecord, producerHash = fileHash(producerFile), entrypointHash = admitted.entryHash,
    binaryHash = admitted.artifactHash, recipe = entrypoint["recipe"],
    components = /** @type {Record<string, string>} */ (entrypoint["componentHashes"]);
  if (admitted.artifact !== binary || admitted.source !== binary ||
      entrypoint["configFile"] !== descriptor.sdkFile || entrypoint["configHash"] !== expectedConfigHash ||
      producer.configHash !== expectedConfigHash || producer.buildOnly !== true || producer.publicationQualified !== false ||
      !producer.application || typeof producer.application !== "object" || producer.application.application !== true ||
      producer.application.entrypoint !== "mom.js" || producer.application.buildOnly !== true) {
    throw new Error("Native application receipt must bind the unpacked build-only executable and exact descriptor");
  }
  if (!hasExactKeys(entrypoint, ["version", "target", "configFile", "configHash", "source", "artifact", "recipe", "componentHashes", "producerReceipt", "buildOnly", "publicationQualified"]) ||
      !recipe || typeof recipe !== "object" || Array.isArray(recipe) ||
      !hasExactKeys(components, [producerFile]) || components[producerFile] !== producerHash) {
    throw new Error("Native release entrypoint does not bind the exact unpacked application, descriptor and producer receipt");
  }
  if (fileHash(descriptor.sdkFile) !== descriptorHash || fileHash(producerFile) !== producerHash ||
      fileHash(entrypointFile) !== entrypointHash || fileHash(binary) !== binaryHash) {
    throw new Error("Native application inputs changed during entrypoint admission");
  }
  return {binary, binaryHash, producerFile, producerHash, entrypointFile, entrypointHash};
}

/** Assemble six unchanged producer outputs through the maintained public frontdoor; only the app is built here.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildLinuxNativeApplication(paths) {
  const componentPins = linuxNativeComponentArchivePins(paths.env),
    descriptor = linuxNativeSdkDescriptorStatePaths(paths, paths.env), output = linuxNativeApplicationStatePaths(paths),
    downloads = partitionLinuxNativeDownloads(paths, [...linuxNativeComponentProfiles]);
  if ([output.root, output.archive, output.configFile].some(file => fs.lstatSync(file, {throwIfNoEntry: false}))) {
    throw new Error("Native Linux application stage refuses existing output paths");
  }
  const inputArchives = auditLinuxNativeSdkDescriptorInputArchives(descriptor), expandedInputs = inputArchives.reduce((sum, row) => sum + row.expandedBytes, 0);
  const expectedComponentNames = linuxNativeComponentProfiles.map(profile => "native-linux-component-" + profile + "-" + paths.suffix + ".tar").sort();
  if (!isDeepStrictEqual(fs.readdirSync(downloads.componentIncoming).sort(), expectedComponentNames)) {
    throw new Error("Native application needs exactly the six producer-pinned component archives");
  }
  const componentPreflight = new Map();
  let componentExpanded = 0;
  for (const profile of linuxNativeComponentProfiles) {
    const state = linuxNativeComponentStatePaths(paths, profile), archive = path.join(downloads.componentIncoming, path.basename(state.archive)),
      pin = componentPins[profile];
    if (typeof pin !== "string") {throw new Error("Native application producer pin is absent for profile: " + profile);}
    const admitted = auditLinuxNativeComponentArchive(state, archive, pin), expandedBytes = tarExpandedBytes(archive, admitted.members.length);
    componentExpanded += expandedBytes;
    componentPreflight.set(profile, {state, archive, ...admitted, expandedBytes});
  }
  const disk = linuxNativeSdkDiskPreflight(expandedInputs + componentExpanded, availableDiskBytes(temp), availableDiskBytes(paths.runnerTemp)),
    deadlineEpoch = paths.env["MOM_NATIVE_APPLICATION_DEADLINE_EPOCH"] || "", remainingMs = Number(deadlineEpoch) * 1000 - Date.now();
  if (!/^\d{10}$/.test(deadlineEpoch) || !Number.isSafeInteger(remainingMs) || remainingMs < 20 * 60 * 1000 || remainingMs > 60 * 60 * 1000) {
    throw new Error("Native application deadline is malformed, expired, or cannot cover compiler provisioning and assembly");
  }
  provisionOneApi(paths, "/var/tmp/mom-dev-native-assembly-" + paths.suffix,
    linuxNativeComponentChildTimeout(Number(deadlineEpoch) * 1000, 15 * 60 * 1000, Date.now() + 20 * 60 * 1000));
  extractState(descriptor.archive, temp, [descriptor.name]);
  restoreLinuxNativeSdkDescriptorInputs(descriptor);
  prepareRocmHostAdmission(paths, descriptor.rocm);
  const lineage = auditLinuxNativeSdkDescriptorLineage(descriptor), descriptorHandoff = {state: descriptor, admission: readBuildJson(descriptor.admissionFile),
    descriptorHash: lineage.descriptorHash, admissionHash: lineage.admissionHash, archives: inputArchives};
  if (!isDeepStrictEqual(inputArchives.map(row => ({artifact: row.artifact, sha256: row.sha256})), lineage.inputArchives)) {
    throw new Error("Same-run SDK archives changed between disk preflight and producer admission");
  }
  const componentHandoff = restoreLinuxNativeComponents(paths, downloads, descriptorHandoff, componentPins, componentPreflight),
    core = componentHandoff.archives.get("core"),
    gpu = Object.fromEntries(linuxNativeGpuProfiles.map(profile => {
      const row = componentHandoff.archives.get(profile);
      if (!row?.admitted) {throw new Error("Missing admitted GPU factory: " + profile);}
      return [profile, path.join(row.state.root, "gpu", profile, "result.json")];
    }));
  if (!core?.admitted) {throw new Error("Missing admitted same-run Core output");}
  const components = linuxNativeAssemblyComponents(path.join(core.state.root, "core/cpu-core.a"), gpu),
    config = linuxNativeAssemblyConfig(lineage.config, components), configBytes = JSON.stringify(config, null, 2) + "\n";
  fs.writeFileSync(output.configFile, configBytes, {flag: "wx", mode: 0o600});
  const assemblyDescriptor = {...descriptor, sdkFile: output.configFile},
    command = linuxNativeApplicationBuildCommand(assemblyDescriptor, output, paths.env["MOM_NATIVE_APPLICATION_DEADLINE_EPOCH"] || "");
  runNode(command.script, command.args, paths.env, command.timeoutMs);
  if (fileHash(descriptor.sdkFile) !== lineage.descriptorHash || fileHash(descriptor.admissionFile) !== lineage.admissionHash ||
      fileHash(output.configFile) !== jsonHash(configBytes) ||
      lineage.inputArchives.some(row => fileHash(path.join(descriptor.incoming, row.artifact)) !== row.sha256) ||
      [...componentHandoff.archives.values()].some(row => fileHash(row.archive) !== row.archiveSha256 ||
        !row.admitted || fileHash(path.join(row.state.root, "component.json")) !== row.admitted.manifestSha256 ||
        !isDeepStrictEqual(linuxNativeComponentFiles(row.state.root).filter(file => file.file !== "component.json"), row.admitted?.files))) {
    throw new Error("Same-run SDK inputs, component outputs or derived config changed during application build");
  }
  const admitted = auditLinuxNativeApplicationEntrypoint(assemblyDescriptor, output, fileHash(output.configFile)),
    configCopy = path.join(output.root, "sdk-components.json");
  fs.copyFileSync(output.configFile, configCopy, fs.constants.COPYFILE_EXCL);
  if (fileHash(configCopy) !== fileHash(output.configFile)) {throw new Error("Packaged assembly config differs from the frontdoor input");}
  const assemblyFile = path.join(output.root, "native-assembly.json"), assembly = {format: "mom-native-linux-assembly-v1",
    descriptor: {path: descriptor.sdkFile, sha256: lineage.descriptorHash}, admission: {path: descriptor.admissionFile, sha256: lineage.admissionHash},
    inputArchives: lineage.inputArchives, rocmNoticeSources: lineage.rocmNoticeSources,
    derivedConfig: {file: "sdk-components.json", sha256: fileHash(configCopy)},
    componentArchives: [...componentHandoff.archives.values()].map(row => ({profile: row.state.profile, sha256: row.archiveSha256,
      manifestSha256: row.admitted?.manifestSha256})),
    application: {sha256: admitted.binaryHash, entrypointSha256: admitted.entrypointHash, receiptSha256: admitted.producerHash},
    buildOnly: true, publicationQualified: false};
  fs.writeFileSync(assemblyFile, JSON.stringify(assembly, null, 2) + "\n", {flag: "wx", mode: 0o600});
  packState(output.archive, temp, [output.name]);
  const members = listState(output.archive, [output.name]);
  if (!members.includes(output.name + "/native-assembly.json") || !members.includes(output.name + "/sdk-components.json") ||
      fileHash(admitted.binary) !== admitted.binaryHash || fileHash(admitted.producerFile) !== admitted.producerHash ||
      fileHash(admitted.entrypointFile) !== admitted.entrypointHash || fileHash(descriptor.sdkFile) !== lineage.descriptorHash) {
    throw new Error("Native application output changed during same-run archive staging");
  }
  console.log(`[native-ci] Assembled the same-run Core and five GPU factories through the public build-only frontdoor; only the controller/application was linked, no executable ran or release was published (${members.length} members, ${disk.buildReserveBytes} byte build reserve).`);
}

/** Check every same-run raw tar before restoring any source or SDK root.
 * @param {ReturnType<typeof linuxNativeSdkDescriptorStatePaths>} state */
function auditLinuxNativeSdkDescriptorInputArchives(state) {
  const incoming = fs.lstatSync(state.incoming);
  if (!incoming.isDirectory() || incoming.isSymbolicLink() || fs.realpathSync(state.incoming) !== state.incoming) {
    throw new Error("Native Linux SDK descriptor inputs must be in the canonical run-scoped incoming directory");
  }
  const expected = state.inputRows.map(row => row.artifact).sort(), actual = fs.readdirSync(state.incoming).sort();
  if (!isDeepStrictEqual(actual, expected)) {throw new Error("Native Linux SDK descriptor has missing, stale, or substituted same-run input tars");}
  return state.inputRows.map(row => auditLinuxNativeSdkDescriptorInputArchive(row, state.incoming));
}

/** Restore each canonical root once; later tars still have their full root inventory checked.
 * @param {ReturnType<typeof linuxNativeSdkDescriptorStatePaths>} state */
function restoreLinuxNativeSdkDescriptorInputs(state) {
  const roots = [...new Set(state.inputRows.flatMap(row => row.roots))], restored = new Set();
  for (const name of roots) {
    if (fs.lstatSync(path.join(temp, name), {throwIfNoEntry: false})) {throw new Error("Native Linux SDK descriptor refuses an ambient restored root: " + name);}
  }
  for (const row of state.inputRows) {
    const selected = row.roots.filter(name => !restored.has(name));
    if (selected.length) {extractState(row.file, temp, row.roots, selected);}
    for (const name of selected) {restored.add(name);}
  }
  if (restored.size !== roots.length) {throw new Error("Native Linux SDK descriptor did not restore every exact same-run root");}
}

/** Re-admit the relocatable CUDA-image include; its original header/tool paths remain provenance metadata.
 * @param {ReturnType<typeof linuxNativeSdkDescriptorStatePaths>} state
 * @param {ReturnType<typeof auditDpcppCudaInputs>} dpcppInputs */
function auditLinuxNativeSdkDescriptorCudaImages(state, dpcppInputs) {
  const include = path.join(state.images.handoffRoot, "cuda-images.inc"),
    sidecar = include + ".json", resultFile = path.join(state.images.handoffRoot, "result.json"),
    admitted = cudaImageInputs(include, repo), manifest = admitted.manifest, result = readBuildJson(resultFile),
    versions = Array.isArray(result.compilerVersions) ? [...result.compilerVersions].sort() : [], images = manifest["images"],
    // cudaImageInputs already validated each complete header-pack record; project its provenance only.
    headerPacks = /** @type {Record<string, {path: string, files: Record<string, string>, manifest: {sha256: string, count: number, bytes: number}}>} */ (manifest["headerPacks"]);
  if (!Array.isArray(images) || !hasExactKeys(result, ["format", "include", "sidecar", "compilerVersions", "plannedImages", "embeddedBytes", "sourceSha256"]) ||
      result.format !== "mom-cuda-image-build-v1" || result.include !== path.join(state.images.output, "cuda-images.inc") ||
      result.sidecar !== path.join(state.images.output, "cuda-images.inc.json") || versions.join(",") !== "12.6,13.3" ||
      !Number.isSafeInteger(result.plannedImages) || result.plannedImages !== images.length ||
      !Number.isSafeInteger(result.embeddedBytes) || result.embeddedBytes !== manifest["resourceBytes"] ||
      result.sourceSha256 !== manifest["sourceSha256"] ||
      Object.keys(headerPacks).sort().join(",") !== "12.6,13.3") {
    throw new Error("CUDA image result does not bind the exact same-run 12.6/13.3 sidecar and image matrix");
  }
  const noticeSources = auditCudaNoticeSources(state.images, dpcppInputs, manifest);
  return {include, includeSha256: admitted.files[include], sidecar, sidecarSha256: admitted.files[sidecar],
    result: resultFile, resultSha256: fileHash(resultFile), compilerVersions: versions, plannedImages: result.plannedImages,
    embeddedBytes: result.embeddedBytes, sourceSha256: result.sourceSha256,
    headerPacks: Object.fromEntries(Object.entries(headerPacks).map(([version, row]) => [version, {
      originalPath: row.path, includeSha256: row.files[row.path], sidecarSha256: row.files[row.path + ".json"],
      sha256: row.manifest.sha256, count: row.manifest.count, bytes: row.manifest.bytes,
    }])), headerPackPayloadsRetained: false, noticeSources};
}

/** Compose the maintained release descriptor after all original public producers pass admission.
 * This emits no Core/factory result and is not a built or runtime-qualified application. @param {ReturnType<typeof ciPaths>} paths */
function buildLinuxNativeSdkDescriptor(paths) {
  const state = linuxNativeSdkDescriptorStatePaths(paths, paths.env);
  if ([state.root, state.archive].some(file => fs.lstatSync(file, {throwIfNoEntry: false}))) {
    throw new Error("Native Linux SDK descriptor stage refuses an existing output");
  }
  const externalDpcppPins = dpcppCudaInputPins(paths.env),
    dpcppInputTar = path.join(state.incoming, path.basename(state.dpcpp.sdkTar));
  if (externalDpcppPins) {dpcppSourceArchivePin(dpcppInputTar, externalDpcppPins.bundleSha256);}
  const inputArchives = auditLinuxNativeSdkDescriptorInputArchives(state);
  if (externalDpcppPins && inputArchives.find(row => row.artifact === path.basename(state.dpcpp.sdkTar))?.sha256 !== externalDpcppPins.bundleSha256) {
    throw new Error("DPC++/CUDA descriptor input differs from its external same-run bundle pin");
  }
  const expandedInputBytes = inputArchives.reduce((sum, row) => sum + row.expandedBytes, 0),
    diskPreflight = linuxNativeSdkDiskPreflight(expandedInputBytes, availableDiskBytes(temp), availableDiskBytes(state.runnerTemp));
  restoreLinuxNativeSdkDescriptorInputs(state);
  if (externalDpcppPins) {dpcppSourceArchivePin(dpcppInputTar, externalDpcppPins.bundleSha256);}
  linuxNativeSdkDiskPreflight(0, availableDiskBytes(temp), availableDiskBytes(state.runnerTemp));
  if (state.rocm.llvmSdkRoot !== paths.sdkRoot || state.rocm.llvmSdkName !== paths.sdkName ||
      state.llvmSdk !== state.rocm.llvmSdk || !state.inputRows.some(row => row.roots.includes(state.rocm.llvmSdkName))) {
    throw new Error("Same-run ROCm host handoff must retain the exact audited main LLVM SDK root");
  }
  const git = canonicalExecutable("/usr/bin/git");
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", paths.source], paths.env, 2 * 60 * 1000);
  verifyIoSource(paths);
  verifyLinuxSources(paths, state.linux, git);
  verifyAcppSource(paths, state.acpp, git);
  verifyHermesSource(paths, state.hermes, git);
  verifyRocmSources(paths, state.rocm, git);
  verifyZstdArchive(state.zstd.source);
  const cudaInputs = auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), externalDpcppPins?.manifestSha256),
    cudaImages = auditLinuxNativeSdkDescriptorCudaImages(state, cudaInputs),
    component = dpcppComponentInputs(state.jit.staticComponent);
  // Existing signed Intel installation is needed to re-admit producer tool fingerprints.
  provisionOneApi(paths, "/var/tmp/mom-dev-native-linux-sdk-descriptor-" + paths.suffix, 30 * 60 * 1000);
  verifyLinuxLlvmAdmissionCi(paths);
  prepareRocmHostAdmission(paths, state.rocm);
  const hermesTools = hermesBuildTools(), openmpTools = openmpBuildTools(), acppTools = {cc: hermesTools.cc, cxx: hermesTools.cxx};
  auditHermesSdk(state.hermes, hermesTools);
  auditOpenmpSdk(state.openmp, openmpTools);
  auditAcppCudaHipSdk(state.acppSdk, acppTools, externalDpcppPins?.manifestSha256);
  const ioOpenSsl = auditIoOpenSsl(paths), config = linuxNativeSdkDescriptorConfig(state, hermesTools),
    {nativeSdkConfig, nativeSdkLinkInputs} = require("./build-native-release"), selected = nativeSdkConfig(config, repo),
    linked = nativeSdkLinkInputs(selected.sdk);
  if (selected.keys.slice().sort().join(",") !== "acpp-cuda,acpp-hip,dpcpp,dpcpp-opencl,oneapi" ||
      !isDeepStrictEqual(Object.keys(config.sdkSources).sort(), ["acpp", "cudaImages", "hermes", "openmpSdk", "rocm", "zstd"].sort()) ||
      !linked.sources || !isDeepStrictEqual(Object.keys(linked.sources).sort(), ["acpp", "cudaImages", "hermes", "openmp", "openssl", "rocm", "zstd"].sort()) ||
      !ioOpenSsl.io.libraries.length || !ioOpenSsl.openssl.libraries.length || !component.sdkObjects.length ||
      !component.files[path.join(state.jit.staticComponent, "component.json")] || !cudaInputs.cuda?.headers?.files) {
    throw new Error("Native Linux SDK descriptor did not admit the complete five-profile source cohort");
  }
  for (const row of state.inputRows) {
    const record = inputArchives.find(item => item.artifact === row.artifact);
    if (!record || record.sha256 !== fileHash(row.file)) {throw new Error("Same-run SDK descriptor input tar changed during admission: " + row.artifact);}
  }
  const gnuNoticeSources = linuxNativeSdkGnuNoticeSources(state, inputArchives);
  const rocmNoticeSources = linuxNativeSdkRocmNoticeSources(state, inputArchives);
  const sortedHashes = Object.fromEntries(Object.entries(linked.hashes).sort(([left], [right]) => left.localeCompare(right)));
  fs.mkdirSync(state.root, {mode: 0o700});
  fs.writeFileSync(state.sdkFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const admission = {format: "mom-native-linux-sdk-descriptor-v1", runAttempt: paths.suffix,
    descriptor: {path: state.sdkFile, sha256: fileHash(state.sdkFile)}, inputArchives, diskPreflight,
    profiles: selected.keys.slice().sort(), sourceReceipts: linked.sources,
    selectedLinkInputs: {files: sortedHashes, sha256: jsonHash(JSON.stringify(sortedHashes))},
    dpcppComponent: {root: state.jit.staticComponent, receipt: path.join(state.jit.staticComponent, "component.json"),
      sdkObjects: component.sdkObjects,
      filesSha256: jsonHash(JSON.stringify(Object.fromEntries(Object.entries(component.files).sort(([left], [right]) => left.localeCompare(right)))))},
    dpcppCudaInputs: {manifest: state.dpcpp.manifest, sha256: fileHash(state.dpcpp.manifest), packageVersions: cudaInputs.cuda.packages},
    cudaImages, gnuNoticeSources, rocmNoticeSources, buildOnly: true, compiled: false, executed: false, publicationQualified: false};
  fs.writeFileSync(state.admissionFile, JSON.stringify(admission, null, 2) + "\n", {flag: "wx", mode: 0o600});
  packState(state.archive, temp, [state.name]);
  listState(state.archive, [state.name]);
  console.log("[native-ci] Composed and admitted the same-run five-profile Linux SDK descriptor; no Core/factories were built, no executable ran, no release was published.");
}

/** @param {string} phase */
function linuxJitSdkArtifactPhase(phase) {
  if (phase === "configure") {return "configured";}
  if (phase === "static-dpcpp") {return "static-dpcpp";}
  if (jitBuildStages.includes(phase)) {return phase;}
  throw new Error("Unknown Linux JIT SDK handoff phase: " + phase);
}

/** A tar-level receipt ties unchanged upstream artifacts together. The resource producer
 * independently records the first three hashes; this re-admits rather than restamps them.
 * @param {ReturnType<typeof linuxJitSdkStatePaths>} state @param {unknown} resourceRecord
 * @param {Record<string,string>} archiveHashes */
function linuxJitSdkInputManifest(state, resourceRecord, archiveHashes) {
  if (!resourceRecord || typeof resourceRecord !== "object" || Array.isArray(resourceRecord) ||
      !Array.isArray(Reflect.get(resourceRecord, "inputArchives")) ||
      !hasExactKeys(archiveHashes, state.inputTars.map(row => row.file))) {
    throw new Error("Linux source-JIT SDK inputs omit the original same-run archive proof");
  }
  const archives = state.inputTars.map(row => {
      const sha256 = archiveHashes[row.file];
      if (!/^[a-f0-9]{64}$/.test(sha256 || "")) {throw new Error("Linux source-JIT SDK archive hash is malformed");}
      return {role: row.role, name: path.basename(row.file), root: row.root, sha256};
    }), original = Reflect.get(resourceRecord, "inputArchives");
  if (original.length !== 3 || original.some((/** @type {unknown} */ value, /** @type {number} */ index) =>
    !hasExactKeys(value, ["role", "name", "root", "sha256"]) || !isDeepStrictEqual(value, archives[index]))) {
    throw new Error("JIT resource receipt does not bind the exact three immutable input tars");
  }
  return {version: 1, run: state.resources.linux.sourceName.slice("mom-native-linux-source-prerequisites-".length), archives};
}

/** Pure plans use the existing producer CLIs and exact same-run artifact names.
 * @param {ReturnType<typeof linuxJitSdkStatePaths>} state @param {string} phase
 * @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools */
function linuxJitSdkBuildPlan(state, phase, tools) {
  const phases = ["configure", ...jitBuildStages, "static-dpcpp"], index = phases.indexOf(phase);
  if (index < 0 || !hasExactKeys(tools, ["cc", "cmake", "cxx", "ninja"]) ||
      Object.values(tools).some(file => typeof file !== "string" || !path.isAbsolute(file) || /[\r\n\0]/.test(file))) {
    throw new Error("Linux source-JIT CI needs an existing producer phase and four explicit tools");
  }
  const previous = index ? phases[index - 1] || "" : "", inputTars = state.inputTars.map(row => ({
    name: path.basename(row.file), root: row.root, role: row.role}));
  if (["configure", ...jitBuildStages].includes(phase)) {
    /** @type {import("./build-native-jit-sdk").Config} */
    const config = {version: 1, target: "linux", sources: state.linux.compilerPreparedReceipt, sdk: state.dpcpp.dpcppRoot,
      resources: path.join(state.resources.root, "resources"), tools, jobs: 4};
    return {phase, previous, previousArtifact: previous ? linuxJitSdkArtifactPhase(previous) : "",
      previousTar: previous ? state.phaseIncomingTar(previous) : "", inputTars, config, output: state.output,
      commands: [{script: "scripts/build-native-jit-sdk.js", args: [state.configFile, state.output, phase]}],
      artifact: {name: path.basename(state.phaseTar(phase)), file: state.phaseTar(phase)}};
  }
  return {phase, previous, previousArtifact: "link", previousTar: state.phaseIncomingTar("link"), inputTars,
    output: state.staticOutput, commands: [
      {script: "scripts/native-compiler-isolation.js", args: [path.join(state.output, "build"), state.isolation]},
      {script: "scripts/build-native-dpcpp-sdk.js", args: [state.linux.preparedReceipt, state.dpcpp.dpcppRoot, state.dpcpp.cudaRoot,
        state.isolatedObject, state.staticOutput, tools.cmake, tools.ninja, "4", tools.cc, tools.cxx]},
    ], artifact: {name: path.basename(state.phaseTar(phase)), file: state.phaseTar(phase)}};
}

/** Re-admit the configure receipt's two content hashes, not just its derived archive list.
 * @param {string} output @param {unknown} value */
function auditLinuxJitSdkConfigureArtifacts(output, value) {
  if (!path.isAbsolute(output) || !value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Linux source-JIT configure evidence is incomplete");
  }
  const receipt = /** @type {{configuredSha256?: string, commandsSha256?: string}} */ (value);
  for (const [name, key] of /** @type {Array<[string, "configuredSha256" | "commandsSha256"]>} */ (
    [["configured-tools.json", "configuredSha256"], ["commands.log", "commandsSha256"]])) {
    const file = path.join(output, name), stat = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file ||
        typeof receipt[key] !== "string" || !/^[a-f0-9]{64}$/.test(receipt[key] || "") || fileHash(file) !== receipt[key]) {
      throw new Error("Linux source-JIT configure artifact hash changed: " + name);
    }
  }
}

/** Restore and validate unchanged dependency and prior-phase archives before tool provisioning.
 * @param {ReturnType<typeof ciPaths>} paths
 * @param {ReturnType<typeof linuxJitSdkStatePaths>} state @param {string} phase */
function restoreLinuxJitSdkInputs(paths, state, phase) {
  const phases = ["configure", ...jitBuildStages, "static-dpcpp"], phaseIndex = phases.indexOf(phase);
  if (phaseIndex < 0 || fs.lstatSync(state.root, {throwIfNoEntry: false}) ||
      state.inputTars.some(row => fs.lstatSync(path.join(temp, row.root), {throwIfNoEntry: false}))) {
    throw new Error("Linux source-JIT CI requires new run-scoped roots for this phase");
  }
  for (const row of state.inputTars) {
    const stat = fs.lstatSync(row.file, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(row.file) !== row.file) {
      throw new Error("Linux source-JIT CI requires every named same-run dependency tar");
    }
  }
  const externalDpcppPins = dpcppCudaInputPins(paths.env), dpcppTar = state.inputTars.find(row => row.role === "dpcpp-cuda-inputs");
  if (!dpcppTar) {throw new Error("Linux source-JIT CI is missing its DPC++/CUDA input archive");}
  if (externalDpcppPins) {dpcppSourceArchivePin(dpcppTar.file, externalDpcppPins.bundleSha256);}
  const resourceTar = state.inputTars[3]?.file || "";
  listState(resourceTar, [state.resources.name]); extractState(resourceTar, temp, [state.resources.name]);
  const resourceRecord = readBuildJson(state.resources.result),
    hashes = Object.fromEntries(state.inputTars.map(row => [row.file, fileHash(row.file)])),
    manifest = linuxJitSdkInputManifest(state, resourceRecord, hashes);
  for (const row of state.inputTars.slice(0, 3)) {
    if (row === dpcppTar && externalDpcppPins) {dpcppSourceArchivePin(row.file, externalDpcppPins.bundleSha256);}
    listState(row.file, [row.root]); extractState(row.file, temp, [row.root]);
  }
  if (externalDpcppPins) {dpcppSourceArchivePin(dpcppTar.file, externalDpcppPins.bundleSha256);}
  verifyLinuxSources(paths, state.linux, canonicalExecutable("/usr/bin/git"));
  auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), externalDpcppPins?.manifestSha256); auditGnuHeaders(state.gnu);
  verifyJitResources(state.resources.resources, "x86_64-unknown-linux-gnu");
  let previous = null;
  if (phaseIndex) {
    previous = phases[phaseIndex - 1] || "";
    const priorTar = state.phaseIncomingTar(previous);
    if (!previous) {throw new Error("Linux source-JIT CI phase ancestry is incomplete");}
    listState(priorTar, [state.name]); extractState(priorTar, temp, [state.name]);
    if (!isDeepStrictEqual(readBuildJson(state.ciInputs), manifest)) {
      throw new Error("Linux source-JIT same-run input archive hashes changed between phases");
    }
  } else {
    fs.mkdirSync(state.root, {mode: 0o700});
    fs.writeFileSync(state.ciInputs, JSON.stringify(manifest, null, 2) + "\n", {flag: "wx", mode: 0o600});
  }
  // Prior-phase receipts hash compiler files; callers provision before deep admission.
  return {manifest, hashes, previous};
}

/** Re-admit current source-JIT phase records and the exact original archive closure.
 * @param {ReturnType<typeof linuxJitSdkStatePaths>} state @param {string} phase */
function auditLinuxJitSdkState(state, phase) {
  const expectedConfig = linuxJitSdkBuildPlan(state, "configure", {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja",
    cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx", cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx"}).config;
  if (!expectedConfig) {throw new Error("Linux source-JIT configure plan has no configuration");}
  return auditLinuxJitCompilerState(state.output, expectedConfig, phase);
}

/** Read-only phase admission shared by local builds and CI; transport identities stay with callers.
 * @param {string} output @param {import("./build-native-jit-sdk").Config} expectedConfig @param {string} phase */
function auditLinuxJitCompilerState(output, expectedConfig, phase) {
  if (expectedConfig.target !== "linux" || !["configure", ...jitBuildStages].includes(phase) || !path.isAbsolute(output)) {
    throw new Error("Linux JIT compiler admission requires an absolute output and a supported Linux phase");
  }
  const build = path.join(output, "build"), beforeFile = path.join(output, "before.json"),
    before = readBuildJson(beforeFile), configure = readBuildJson(path.join(output, "stage-configure.json")),
    current = phase === "configure" ? configure : readBuildJson(path.join(output, "stage-" + phase + ".json")),
    libraries = jitCompilerArchives(readBuildText(path.join(output, "commands.log")), build), outputs = jitBuildOutputs(build);
  verifySourceInputs(before.recipe, require("./build-native-jit-sdk").jitSdkRecipe(), "Linux JIT compiler SDK admission");
  auditLinuxJitSdkConfigureArtifacts(output, configure);
  const compilerRoots = Object.fromEntries(Object.entries(readBuildJson(expectedConfig.sources).prepared)
      .map(([name, source]) => [name, source.root])), resources = verifyJitResources(expectedConfig.resources, "x86_64-unknown-linux-gnu"),
    expectedPlan = jitSdkPlan("linux", compilerRoots, expectedConfig.sdk,
      resources.overlay, output, expectedConfig.tools, expectedConfig.jobs);
  if (before.version !== 1 || before.config?.target !== "linux" || before.plan?.target !== "linux" || before.plan.build !== build ||
      !isDeepStrictEqual(before.config, expectedConfig) || !isDeepStrictEqual(before.plan, expectedPlan) ||
      configure.version !== 1 || configure.stage !== "configure" || configure.beforeSha256 !== fileHash(beforeFile) ||
      configure.buildOnly !== true || !isDeepStrictEqual(before.supportLibraries, []) || !isDeepStrictEqual(configure.supportLibraries, []) ||
      !isDeepStrictEqual(configure.libraries, libraries) ||
      current.version !== 1 || current.stage !== phase || current.beforeSha256 !== fileHash(beforeFile) ||
      current.buildOnly !== true || !isDeepStrictEqual(current.supportLibraries, []) ||
      !isDeepStrictEqual(current.libraries, libraries) || !isDeepStrictEqual(current.outputs, outputs) ||
      !isDeepStrictEqual(current.inputs, buildFileInputs(Object.keys(current.inputs || {})))) {
    throw new Error("Linux source-JIT compiler phase receipt or archive graph changed");
  }
  if (phase !== "configure") {
    const phases = ["configure", ...jitBuildStages], index = phases.indexOf(phase), previous = phases[index - 1] || "",
      previousFile = path.join(output, "stage-" + previous + ".json"), prior = readBuildJson(previousFile);
    if (!previous || current.previousSha256 !== fileHash(previousFile) || current.configuredSha256 !== configure.configuredSha256 ||
        current.commandsSha256 !== configure.commandsSha256 || !isDeepStrictEqual(current.libraries, prior.libraries)) {
      throw new Error("Linux source-JIT phase ancestry changed");
    }
  }
  if (phase === "link") {
    const result = readBuildJson(path.join(output, "result.json")), proof = path.join(build, "bin/native-dpcpp-jit"),
      expectedLibraries = libraries.map(file => ({path: file, sha256: fileHash(file)}));
    if (result.version !== 1 || result.kind !== "original-source-jit-sdk" || result.target !== "linux" || result.proof?.path !== proof ||
        result.proof.sha256 !== fileHash(proof) || result.compiled !== true || result.buildOnly !== true || result.executed !== false ||
        result.publicationQualified !== false || result.sourcesFile !== before.config.sources || result.resources !== before.config.resources ||
        !isDeepStrictEqual(result.plan, before.plan) || !isDeepStrictEqual(result.recipe, before.recipe) ||
        !isDeepStrictEqual(result.inputs, current.inputs) || !isDeepStrictEqual(result.supportLibraries, current.supportLibraries) ||
        !isDeepStrictEqual(result.libraries, expectedLibraries)) {
      throw new Error("Linux source-JIT final producer result is incomplete or changed");
    }
  }
  return current;
}

/** Build one existing public source-JIT phase and transfer only its mutable build tree.
 * @param {ReturnType<typeof ciPaths>} paths @param {string} phase */
function buildLinuxJitSdkPhase(paths, phase) {
  const state = linuxJitSdkStatePaths(paths, paths.env), {hashes, previous} = restoreLinuxJitSdkInputs(paths, state, phase);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-linux-jit-sdk-" + paths.suffix, 25 * 60 * 1000);
  if (previous) {auditLinuxJitSdkState(state, previous);}
  const tools = {cmake: canonicalExecutable("/usr/bin/cmake"), ninja: canonicalExecutable("/usr/bin/ninja"),
      cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"), cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx")},
    plan = linuxJitSdkBuildPlan(state, phase, tools);
  const config = require("./build-native-jit-sdk").jitSdkConfig(plan.config);
  fs.writeFileSync(state.configFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const command = plan.commands[0];
  if (!command) {throw new Error("Linux source-JIT phase has no public producer command");}
  runNode(command.script, command.args, paths.env, 50 * 60 * 1000);
  if (["configure", ...jitBuildStages].includes(phase) &&
      !isDeepStrictEqual(readBuildJson(path.join(state.output, "before.json")).config, plan.config)) {
    throw new Error("Public source-JIT producer changed its run-scoped configuration");
  }
  auditLinuxJitSdkState(state, phase);
  for (const [file, sha256] of Object.entries(hashes)) {if (fileHash(file) !== sha256) {throw new Error("Linux source-JIT dependency tar changed during production");}}
  packState(plan.artifact.file, temp, [state.name]); listState(plan.artifact.file, [state.name]);
  console.log("[native-ci] Handed off source-JIT phase " + phase + "; build-only, no GPU/runtime/publication qualification.");
}

/** Run-scoped, proof-only handoff kept separate from the mutable SDK phase tar.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths */
function linuxJitRuntimeProofPaths(paths) {
  const name = "mom-native-linux-jit-runtime-proof-" + paths.suffix,
    archive = path.join(paths.runnerTemp, "native-linux-jit-runtime-proof-" + paths.suffix + ".tar");
  return {name, root: path.join(temp, name), archive,
    incomingArchive: path.join(paths.runnerTemp, "incoming", path.basename(archive)),
    proof: path.join(temp, name, "native-dpcpp-jit"), receipt: path.join(temp, name, "result.json")};
}

/** Package only the linked executable and its original build-only result receipt.
 * @param {string} sourceProof @param {string} sourceReceipt @param {string} base @param {string} name @param {string} archive */
function packLinuxJitRuntimeProofFiles(sourceProof, sourceReceipt, base, name, archive) {
  if (!path.isAbsolute(sourceProof) || !path.isAbsolute(sourceReceipt) || !path.isAbsolute(base) ||
      !safeRelative(name) || name.includes("/") || !path.isAbsolute(archive)) {
    throw new Error("Linux source-JIT runtime proof paths are invalid");
  }
  const proofStat = fs.lstatSync(sourceProof, {throwIfNoEntry: false}), resultStat = fs.lstatSync(sourceReceipt, {throwIfNoEntry: false});
  if (!proofStat?.isFile() || proofStat.isSymbolicLink() || !(proofStat.mode & 0o111) ||
      proofStat.size > 512 * 1024 * 1024 || !resultStat?.isFile() || resultStat.isSymbolicLink() ||
      resultStat.size > 16 * 1024 * 1024) {
    throw new Error("Linux source-JIT runtime proof or original receipt is not a regular executable/file");
  }
  const proofSha256 = fileHash(sourceProof), resultSha256 = fileHash(sourceReceipt), result = readBuildJson(sourceReceipt);
  if (result.version !== 1 || result.kind !== "original-source-jit-sdk" || result.target !== "linux" ||
      result.proof?.path !== sourceProof || result.proof.sha256 !== proofSha256 || result.compiled !== true ||
      result.buildOnly !== true || result.executed !== false || result.publicationQualified !== false) {
    throw new Error("Linux source-JIT runtime proof is not bound to an unchanged build-only receipt");
  }
  const root = fs.realpathSync(base), packageRoot = path.join(root, name);
  if (fs.existsSync(packageRoot)) {throw new Error("Linux source-JIT runtime proof output root already exists");}
  fs.mkdirSync(packageRoot, {mode: 0o700});
  let archiveCreated = false;
  try {
    const proof = path.join(packageRoot, "native-dpcpp-jit"), receipt = path.join(packageRoot, "result.json");
    fs.copyFileSync(sourceProof, proof, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(proof, proofStat.mode & 0o777);
    fs.copyFileSync(sourceReceipt, receipt, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(receipt, resultStat.mode & 0o777);
    if (fileHash(proof) !== proofSha256 || fileHash(receipt) !== resultSha256) {
      throw new Error("Linux source-JIT runtime proof copy differs from its producer bytes");
    }
    const packed = packState(archive, root, [name]); archiveCreated = true;
    const members = listState(archive, [name]).map(member => member.replace(/\/$/, "")).sort(),
      expected = [name, name + "/native-dpcpp-jit", name + "/result.json"].sort();
    if (!isDeepStrictEqual(members, expected)) {throw new Error("Linux source-JIT runtime proof tar has unexpected members");}
    return {proofSha256, resultSha256, bundleSha256: fileHash(archive), bundleBytes: packed.bytes};
  } catch (error) {
    if (archiveCreated) {fs.rmSync(archive, {force: true});}
    throw error;
  } finally {
    fs.rmSync(packageRoot, {recursive: true, force: true});
  }
}

/** Admit an exact same-run proof-only tar before giving its path to the runtime test.
 * @param {string} archive @param {string} base @param {string} name @param {string} originalProofPath
 * @param {{proofSha256: string, resultSha256: string, bundleSha256: string, bundleBytes: string}} expected */
function extractLinuxJitRuntimeProof(archive, base, name, originalProofPath, expected) {
  const hashes = [expected.proofSha256, expected.resultSha256, expected.bundleSha256];
  if (!path.isAbsolute(archive) || !path.isAbsolute(base) || !path.isAbsolute(originalProofPath) ||
      !safeRelative(name) || name.includes("/") || hashes.some(value => !/^[a-f0-9]{64}$/.test(value)) ||
      !/^[1-9][0-9]{0,8}$/.test(expected.bundleBytes) || Number(expected.bundleBytes) > 513 * 1024 * 1024) {
    throw new Error("Linux source-JIT runtime proof expectations are invalid");
  }
  const archiveStat = fs.lstatSync(archive, {throwIfNoEntry: false});
  if (!archiveStat?.isFile() || archiveStat.isSymbolicLink() || archiveStat.size !== Number(expected.bundleBytes) ||
      fileHash(archive) !== expected.bundleSha256) {
    throw new Error("Linux source-JIT runtime proof bundle bytes or hash changed");
  }
  const members = listState(archive, [name]).map(member => member.replace(/\/$/, "")).sort(),
    wanted = [name, name + "/native-dpcpp-jit", name + "/result.json"].sort();
  if (!isDeepStrictEqual(members, wanted)) {throw new Error("Linux source-JIT runtime proof tar is missing or has unexpected members");}
  extractState(archive, base, [name]);
  const root = path.join(fs.realpathSync(base), name), proof = path.join(root, "native-dpcpp-jit"), receipt = path.join(root, "result.json"),
    entries = fs.readdirSync(root).sort(), proofStat = fs.lstatSync(proof), resultStat = fs.lstatSync(receipt);
  if (!isDeepStrictEqual(entries, ["native-dpcpp-jit", "result.json"]) || !proofStat.isFile() || proofStat.isSymbolicLink() ||
      !(proofStat.mode & 0o111) || proofStat.nlink !== 1 || !resultStat.isFile() || resultStat.isSymbolicLink() || resultStat.nlink !== 1 ||
      fileHash(proof) !== expected.proofSha256 || fileHash(receipt) !== expected.resultSha256) {
    throw new Error("Linux source-JIT runtime proof files differ from the same-run outputs");
  }
  const result = readBuildJson(receipt);
  if (result.version !== 1 || result.kind !== "original-source-jit-sdk" || result.target !== "linux" ||
      result.proof?.path !== originalProofPath || result.proof.sha256 !== expected.proofSha256 || result.compiled !== true ||
      result.buildOnly !== true || result.executed !== false || result.publicationQualified !== false) {
    throw new Error("Linux source-JIT runtime proof lost its original build-only receipt");
  }
  return {root, proof, receipt, proofSha256: expected.proofSha256, resultSha256: expected.resultSha256};
}

/** @param {NodeJS.ProcessEnv} environment @param {Record<string, string>} values */
function appendWorkflowOutputs(environment, values) {
  const output = environment["GITHUB_OUTPUT"], stat = output && path.isAbsolute(output) ? fs.lstatSync(output, {throwIfNoEntry: false}) : undefined,
    lines = Object.entries(values);
  if (!output || !stat?.isFile() || stat.isSymbolicLink() || lines.some(([key, value]) =>
    !/^[a-z][a-z0-9_]*$/.test(key) || /[\r\n]/.test(value))) {
    throw new Error("Linux source-JIT runtime handoff requires a regular GitHub output file and bounded fields");
  }
  fs.appendFileSync(output, lines.map(([key, value]) => key + "=" + value).join("\n") + "\n", {encoding: "utf8"});
}

/** Export the linked executable only after the full public producer admission succeeds.
 * @param {ReturnType<typeof ciPaths>} paths */
function packLinuxJitRuntimeProof(paths) {
  const state = linuxJitSdkStatePaths(paths, paths.env), transfer = linuxJitRuntimeProofPaths(paths);
  auditLinuxJitSdkState(state, "link");
  const output = path.join(state.output, "build/bin/native-dpcpp-jit"), result = path.join(state.output, "result.json"),
    packed = packLinuxJitRuntimeProofFiles(output, result, temp, transfer.name, transfer.archive);
  appendWorkflowOutputs(paths.env, {proof_sha256: packed.proofSha256, result_sha256: packed.resultSha256,
    bundle_sha256: packed.bundleSha256, bundle_bytes: String(packed.bundleBytes)});
  console.log("[native-ci] Exported only the same-run source-JIT proof and unchanged build-only receipt; no SDK, GPU, or publication handoff.");
}

/** Validate/extract the separately uploaded proof tar; the compiler SDK is not an input.
 * @param {ReturnType<typeof ciPaths>} paths */
function admitLinuxJitRuntimeProof(paths) {
  const state = linuxJitSdkStatePaths(paths, paths.env), transfer = linuxJitRuntimeProofPaths(paths), env = paths.env,
    expected = {proofSha256: env["MOM_NATIVE_LINUX_JIT_RUNTIME_PROOF_SHA256"] || "",
      resultSha256: env["MOM_NATIVE_LINUX_JIT_RUNTIME_RESULT_SHA256"] || "",
      bundleSha256: env["MOM_NATIVE_LINUX_JIT_RUNTIME_BUNDLE_SHA256"] || "",
      bundleBytes: env["MOM_NATIVE_LINUX_JIT_RUNTIME_BUNDLE_BYTES"] || ""};
  const admitted = extractLinuxJitRuntimeProof(transfer.incomingArchive, temp, transfer.name,
    path.join(state.output, "build/bin/native-dpcpp-jit"), expected);
  appendWorkflowOutputs(env, {fixture_path: admitted.proof, proof_sha256: admitted.proofSha256});
  console.log("[native-ci] Admitted the exact same-run proof-only tar and unchanged receipt; no SDK/compiler inputs restored.");
}

/** Isolate the completed original compiler closure and invoke the existing full static
 * DPC++ producer. Neither step executes a GPU or qualifies runtime/publication.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildLinuxJitStaticSdk(paths) {
  const state = linuxJitSdkStatePaths(paths, paths.env), {manifest, hashes, previous} = restoreLinuxJitSdkInputs(paths, state, "static-dpcpp");
  provisionOneApi(paths, "/var/tmp/mom-dev-native-linux-jit-sdk-" + paths.suffix, 25 * 60 * 1000);
  if (previous) {auditLinuxJitSdkState(state, previous);}
  const linked = readBuildJson(path.join(state.output, "result.json"));
  const tools = {cmake: canonicalExecutable("/usr/bin/cmake"), ninja: canonicalExecutable("/usr/bin/ninja"),
      cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"), cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx")},
    plan = linuxJitSdkBuildPlan(state, "static-dpcpp", tools);
  if (fs.lstatSync(state.isolation, {throwIfNoEntry: false}) || linked.target !== "linux" || linked.compiled !== true ||
      linked.executed !== false || linked.publicationQualified !== false) {
    throw new Error("Linux source-JIT isolation requires an admitted final link tree and new output");
  }
  const isolate = plan.commands[0], build = plan.commands[1];
  if (!isolate || !build) {throw new Error("Static DPC++ handoff omitted an existing public producer command");}
  runNode(isolate.script, isolate.args, paths.env, 15 * 60 * 1000);
  const receipt = readBuildJson(path.join(state.isolation, "result.json")), raw = path.join(state.isolation, "compiler-raw.o");
  if (receipt.input !== raw || receipt.object !== state.isolatedObject || receipt.prefix !== "mom_dpcpp23_" ||
      !isDeepStrictEqual(receipt.entryPoints, dpcppEntries) || receipt.inputSha256 !== fileHash(raw) ||
      receipt.objectSha256 !== fileHash(state.isolatedObject)) {throw new Error("Linux compiler isolation receipt does not bind the linked object");}
  if (fs.lstatSync(state.staticOutput, {throwIfNoEntry: false}) || fs.lstatSync(state.staticComponent, {throwIfNoEntry: false}) ||
      fs.lstatSync(state.staticCiInputs, {throwIfNoEntry: false})) {throw new Error("Linux static DPC++ output refuses existing evidence");}
  const staticInputs = {version: 1, run: manifest.run, sourceJitTar: {name: path.basename(state.phaseTar("link")), sha256: fileHash(state.phaseIncomingTar("link"))},
    isolatedObject: {path: state.isolatedObject, sha256: fileHash(state.isolatedObject)}, dependencies: manifest.archives};
  fs.writeFileSync(state.staticCiInputs, JSON.stringify(staticInputs, null, 2) + "\n", {flag: "wx", mode: 0o600});
  runNode(build.script, build.args, paths.env, 50 * 60 * 1000);
  const produced = readBuildJson(path.join(state.staticOutput, "result.json")), component = dpcppComponentInputs(state.staticComponent);
  if (produced.version !== 1 || produced.target !== "linux" || produced.buildOnly !== true || produced.executed !== false ||
      produced.publicationQualified !== false || produced.archive !== path.join(state.staticOutput, "build/libmom-sycl-static.a") ||
      fileHash(produced.archive) !== produced.sha256 || component.files[path.join(state.staticComponent, "component.json")] !==
      fileHash(path.join(state.staticComponent, "component.json")) || !isDeepStrictEqual(readBuildJson(state.ciInputs), manifest) ||
      !isDeepStrictEqual(readBuildJson(state.staticCiInputs), staticInputs)) {
    throw new Error("Original static DPC++ producer did not preserve a complete admitted build-only component");
  }
  for (const [file, sha256] of Object.entries(hashes)) {if (fileHash(file) !== sha256) {throw new Error("Linux static DPC++ input tar changed during production");}}
  packState(plan.artifact.file, temp, [state.name]); listState(plan.artifact.file, [state.name]);
  console.log("[native-ci] Built and admitted the original static Linux DPC++ component; not executed or runtime/publication qualified.");
}

/** Reuse the public target-aware header producer; package installation belongs to the explicit CI step.
 * @param {ReturnType<typeof ciPaths>} paths @param {"x86_64-unknown-linux-gnu" | "x86_64-w64-windows-gnu"} target */
function buildGnuHeaders(paths, target) {
  const state = gnuHeadersStatePaths(paths, paths.env, target);
  if (fs.lstatSync(state.root, {throwIfNoEntry: false}) || fs.lstatSync(state.sdkTar, {throwIfNoEntry: false})) {
    throw new Error("GNU header stage refuses an existing output");
  }
  runNode("scripts/prepare-native-gnu-headers.js", ["build", state.root, target], paths.env, 5 * 60 * 1000);
  auditGnuHeaders(state, target);
  packState(state.sdkTar, temp, [state.name]);
  listState(state.sdkTar, [state.name]);
  console.log(`[native-ci] Admitted ${target} GNU target headers and bounded package copyright bodies; build-only, not publication qualified.`);
}

/** Preserve the established Linux entry point. @param {ReturnType<typeof ciPaths>} paths */
function buildLinuxGnuHeaders(paths) {buildGnuHeaders(paths, "x86_64-unknown-linux-gnu");}
/** @param {ReturnType<typeof ciPaths>} paths */
function buildWindowsGnuHeaders(paths) {buildGnuHeaders(paths, "x86_64-w64-windows-gnu");}

/** Pinned ROCm source roots remain bare Git repositories because their existing producers
 * consume the exact commits with `git archive`; no second prepared-source format is needed.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp">} paths @param {NodeJS.ProcessEnv} [environment] */
function rocmSourceStatePaths(paths, environment = process.env) {
  const sourceName = "mom-native-rocm-prerequisites-" + paths.suffix, source = path.join(temp, sourceName),
    sourceTar = path.join(paths.runnerTemp, "native-rocm-state-" + paths.suffix + "-sources.tar"),
    pythonSourceRoot = path.join(source, "python-source"), pythonArchive = path.join(pythonSourceRoot, "CppHeaderParser-2.7.4.tar.gz"),
    pythonSourceReceipt = path.join(pythonSourceRoot, "receipt.json");
  if (environment["MOM_NATIVE_ROCM_SOURCE_DIR"] !== source) {
    throw new Error("Native ROCm source root must exactly match this workflow run");
  }
  return {source, sourceName, rootsFile: path.join(source, "roots.json"), fetchReceipt: path.join(source, "fetch.json"), sourceTar,
    pythonSourceRoot, pythonArchive, pythonSourceReceipt,
    incomingTar: path.join(paths.runnerTemp, "incoming", path.basename(sourceTar))};
}

/** Fetch one pinned pure-Python build tool alongside the ROCm source roots; it is installed offline later.
 * @param {Pick<ReturnType<typeof rocmSourceStatePaths>, "pythonSourceRoot" | "pythonArchive" | "pythonSourceReceipt">} state
 * @param {NodeJS.ProcessEnv} env */
function fetchRocmParserSource(state, env) {
  if (fs.existsSync(state.pythonSourceRoot)) {throw new Error("ROCm parser source handoff refuses an existing output");}
  fs.mkdirSync(state.pythonSourceRoot, {mode: 0o700});
  const partial = path.join(state.pythonSourceRoot, ".CppHeaderParser-2.7.4.download"), curl = canonicalExecutable("/usr/bin/curl"),
    result = spawnSync(curl, ["--fail", "--show-error", "--location", "--proto", "=https", "--proto-redir", "=https",
      "--max-redirs", "5", "--retry", "5", "--connect-timeout", "15", "--max-time", "90",
      "--max-filesize", String(rocmParserSource.bytes), "--output", partial, rocmParserSource.url], {env, stdio: "inherit", timeout: 2 * 60 * 1000});
  if (result.error || result.signal || result.status !== 0) {throw new Error("Pinned ROCm parser source fetch failed", {cause: result.error});}
  const stat = fs.lstatSync(partial);
  if (!stat.isFile() || stat.size !== rocmParserSource.bytes || fileHash(partial) !== rocmParserSource.sha256) {
    throw new Error("Pinned ROCm parser source archive failed its size/SHA-256 check");
  }
  fs.renameSync(partial, state.pythonArchive);
  fs.writeFileSync(state.pythonSourceReceipt, JSON.stringify({version: 1, target: "rocm-parser-source", url: rocmParserSource.url,
    archive: {path: state.pythonArchive, bytes: rocmParserSource.bytes, sha256: rocmParserSource.sha256},
    helperSha256: fileHash(__filename), producerSha256: fileHash(path.join(repo, "scripts/build-native-rocm-host-runtime.js"))}, null, 2) + "\n",
  {flag: "wx", mode: 0o600});
}

/** The helper digest records acquisition history, not the current reader's identity.
 * Reuse is bound to the current archive pin and consuming producer; unrelated CI
 * helper edits must not force a refetch of identical source bytes.
 * @param {any} receipt @param {string} pythonArchive @param {string} producerSha256 */
function assertRocmParserReceipt(receipt, pythonArchive, producerSha256) {
  if (!hasExactKeys(receipt, ["version", "target", "url", "archive", "helperSha256", "producerSha256"]) || receipt.version !== 1 ||
      receipt.target !== "rocm-parser-source" || receipt.url !== rocmParserSource.url ||
      !hasExactKeys(receipt.archive, ["path", "bytes", "sha256"]) || receipt.archive.path !== pythonArchive ||
      receipt.archive.bytes !== rocmParserSource.bytes || receipt.archive.sha256 !== rocmParserSource.sha256 ||
      typeof receipt.helperSha256 !== "string" || !/^[a-f0-9]{64}$/.test(receipt.helperSha256) ||
      receipt.producerSha256 !== producerSha256) {
    throw new Error("ROCm parser source receipt does not bind acquisition metadata, current producer, and pinned archive");
  }
}

/** Re-admit the exact same-run parser sdist without changing the public ROCm Git fetch receipt.
 * @param {Pick<ReturnType<typeof rocmSourceStatePaths>, "pythonArchive" | "pythonSourceReceipt">} state */
function verifyRocmParserSource(state) {
  const receipt = readBuildJson(state.pythonSourceReceipt), producer = path.join(repo, "scripts/build-native-rocm-host-runtime.js");
  assertRocmParserReceipt(receipt, state.pythonArchive, fileHash(producer));
  const stat = fs.lstatSync(state.pythonArchive);
  if (!stat.isFile() || fs.realpathSync(state.pythonArchive) !== state.pythonArchive || stat.size !== rocmParserSource.bytes ||
      fileHash(state.pythonArchive) !== rocmParserSource.sha256) {throw new Error("ROCm parser source archive changed after acquisition");}
  return receipt;
}

/** Same-run ROCm/COMGR build roots retain both upstream Git cohorts, the matching LLVM SDK,
 * exported HIP headers, and fresh resource/COMGR outputs under their original /tmp paths.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp" | "source" | "sourceName" | "ioSourceName" | "sdkRoot" | "sdkName" | "output">} paths
 * @param {NodeJS.ProcessEnv} [environment] */
function rocmComgrStatePaths(paths, environment = process.env) {
  const rocm = rocmSourceStatePaths(paths, environment), resourcesName = "mom-native-rocm-resources-" + paths.suffix,
    hipHeadersName = "mom-native-rocm-hip-headers-" + paths.suffix, sdkName = "mom-native-rocm-comgr-sdk-" + paths.suffix,
    resourcesRoot = path.join(temp, resourcesName), hipHeadersRoot = path.join(temp, hipHeadersName), sdkRoot = path.join(temp, sdkName),
    sourceTar = path.join(paths.runnerTemp, "native-rocm-state-" + paths.suffix + "-sources.tar"),
    llvmSdkTar = path.join(paths.runnerTemp, "native-linux-llvm21-installed-sdk-" + paths.suffix + ".tar"),
    llvmSourceTar = path.join(paths.runnerTemp, "native-llvm21-state-" + paths.suffix + "-sources.tar"),
    sdkTar = path.join(paths.runnerTemp, "native-linux-rocm-comgr-sdk-" + paths.suffix + ".tar");
  if (environment["MOM_NATIVE_ROCM_COMGR_SDK_DIR"] !== sdkRoot ||
      environment["MOM_NATIVE_ROCM_RESOURCES_DIR"] !== resourcesRoot ||
      environment["MOM_NATIVE_ROCM_HIP_HEADERS_DIR"] !== hipHeadersRoot) {
    throw new Error("Native ROCm/COMGR output roots must exactly match this workflow run");
  }
  const hipSourceRoot = path.join(hipHeadersRoot, "source"), hipCommonRoot = path.join(hipSourceRoot, "hip-common"),
    clrHipamdRoot = path.join(hipSourceRoot, "clr-hipamd"), hipBuildRoot = path.join(hipHeadersRoot, "build"),
    hipIncludeRoot = path.join(hipHeadersRoot, "include"), hipArchiveRoot = path.join(hipHeadersRoot, "archives"),
    hipInstallPrefix = path.join(hipBuildRoot, "install-prefix"), hipPatchRoot = path.join(hipHeadersRoot, "patches"),
    hipPatchFile = path.join(hipPatchRoot, path.basename(rocmHipHeaderPatch)),
    hipPatchCheckLog = path.join(hipPatchRoot, "check.json"), hipPatchApplyLog = path.join(hipPatchRoot, "apply.json");
  return {...rocm, resourcesName, resourcesRoot, hipHeadersName, hipHeadersRoot, hipIncludeRoot, hipSourceRoot, hipCommonRoot,
    clrHipamdRoot, hipBuildRoot, hipArchiveRoot, hipInstallPrefix, hipPatchRoot, hipPatchFile, hipPatchCheckLog, hipPatchApplyLog, sdkName, sdkRoot,
    sourceTar, llvmSdkTar, llvmSourceTar, incomingSourceTar: path.join(paths.runnerTemp, "incoming", path.basename(sourceTar)),
    incomingLlvmSdkTar: path.join(paths.runnerTemp, "incoming", path.basename(llvmSdkTar)),
    incomingLlvmSourceTar: path.join(paths.runnerTemp, "incoming", path.basename(llvmSourceTar)),
    llvmSourceArchiveRoots: [paths.sourceName, paths.ioSourceName],
    hipCommonTar: path.join(hipArchiveRoot, "hip-common.tar"), clrHipamdTar: path.join(hipArchiveRoot, "clr-hipamd.tar"),
    hipConfigureLog: path.join(hipHeadersRoot, "cmake-configure.log"),
    hipConfigurePlan: path.join(hipHeadersRoot, "configure-plan.json"),
    resourcesConfig: path.join(paths.runnerTemp, "rocm-resources-" + paths.suffix + ".json"),
    sdkTar, llvmSource: paths.source, llvmSourceName: paths.sourceName, llvmSdkRoot: paths.sdkRoot,
    llvmSdkName: paths.sdkName, llvmSdk: path.join(paths.output, "sdk"), output: paths.output};
}

/** Only the already-admitted source/LLVM inputs and paths consumed by the public ROCm recipe.
 * @typedef {Pick<ReturnType<typeof rocmComgrStatePaths>, "source" | "llvmSdk" | "hipHeadersRoot" | "hipSourceRoot" | "hipCommonRoot" |
 * "clrHipamdRoot" | "hipBuildRoot" | "hipArchiveRoot" | "hipInstallPrefix" | "hipIncludeRoot" | "hipPatchRoot" | "hipPatchFile" |
 * "hipPatchCheckLog" | "hipPatchApplyLog" | "hipConfigureLog" | "hipConfigurePlan" | "hipCommonTar" | "clrHipamdTar" |
 * "resourcesRoot" | "resourcesConfig" | "sdkRoot">} RocmComgrBuildState */

/** Restore the admitted COMGR cohort and retain a fresh, separately rooted ROCm host runtime SDK.
 * @param {Pick<ReturnType<typeof ciPaths>, "suffix" | "runnerTemp" | "source" | "sourceName" | "ioSourceName" | "sdkRoot" | "sdkName" | "output">} paths
 * @param {NodeJS.ProcessEnv} [environment] */
function rocmHostStatePaths(paths, environment = process.env) {
  const comgr = rocmComgrStatePaths(paths, environment), hostSdkName = "mom-native-rocm-host-sdk-" + paths.suffix,
    hostSdkRoot = path.join(temp, hostSdkName), runtimeOutput = path.join(hostSdkRoot, "runtime"),
    configFile = path.join(hostSdkRoot, "rocm-host.json"),
    incomingComgrTar = path.join(paths.runnerTemp, "incoming", path.basename(comgr.sdkTar)),
    hostSdkTar = path.join(paths.runnerTemp, "native-linux-rocm-host-sdk-" + paths.suffix + ".tar");
  if (environment["MOM_NATIVE_ROCM_HOST_SDK_DIR"] !== hostSdkRoot || path.dirname(runtimeOutput) !== hostSdkRoot) {
    throw new Error("Native ROCm host SDK output root must exactly match this workflow run");
  }
  return {...comgr, hostSdkName, hostSdkRoot, runtimeOutput, configFile, incomingComgrTar, hostSdkTar};
}

/** Validate acquisition identity without rewriting the immutable fetch receipt.
 * @param {any} receipt @param {any} roots @param {"hermes" | "acpp"} target @param {string} source
 * @param {string} git @param {string} gitSha256 @param {string} fetcherSha256 */
function assertGitFetchProvenance(receipt, roots, target, source, git, gitSha256, fetcherSha256) {
  const root = path.join(source, "roots", target), recipe = sourceRecipe(target), origin = origins[target],
    revision = target === "hermes" ? hermesPins.revision : syclRevision;
  if (!hasExactKeys(receipt, ["version", "target", "status", "networked", "buildOnly", "compiled", "publicationQualified", "git", "fetcherSha256", "recipe", "components"]) ||
      receipt.version !== 1 || receipt.target !== target || receipt.status !== "PASS" || receipt.networked !== true ||
      receipt.buildOnly !== true || receipt.compiled !== false || receipt.publicationQualified !== false ||
      !hasExactKeys(receipt.git, ["path", "sha256"]) || receipt.git.path !== git || receipt.git.sha256 !== gitSha256 ||
      receipt.fetcherSha256 !== fetcherSha256 || !hasExactKeys(receipt.recipe, ["sha256", "files"]) ||
      receipt.recipe.sha256 !== jsonHash(JSON.stringify(recipe)) || !isDeepStrictEqual(receipt.recipe.files, recipe.graph) ||
      !origin || !hasExactKeys(receipt.components, [target]) ||
      !hasExactKeys(receipt.components[target], ["origin", "revision", "root", "gitDirectory", "shallow", "partialClone"]) ||
      receipt.components[target].origin !== origin || receipt.components[target].revision !== revision ||
      receipt.components[target].root !== root || receipt.components[target].gitDirectory !== "bare" ||
      receipt.components[target].shallow !== true || receipt.components[target].partialClone !== false ||
      !hasExactKeys(roots, [target]) || roots[target] !== root) {
    throw new Error(`Native ${target} fetch receipt does not bind this run's original pinned source and producer inputs`);
  }
}

/** Keep the existing Hermes-specific entrypoint stable while sharing its strict receipt checks.
 * @param {any} receipt @param {any} roots @param {string} source @param {string} git @param {string} gitSha256
 * @param {string} fetcherSha256 */
function assertHermesFetchProvenance(receipt, roots, source, git, gitSha256, fetcherSha256) {
  assertGitFetchProvenance(receipt, roots, "hermes", source, git, gitSha256, fetcherSha256);
}

/** @param {any} receipt @param {any} roots @param {string} source @param {string} git @param {string} gitSha256
 * @param {string} fetcherSha256 */
function assertAcppFetchProvenance(receipt, roots, source, git, gitSha256, fetcherSha256) {
  assertGitFetchProvenance(receipt, roots, "acpp", source, git, gitSha256, fetcherSha256);
}

/** Validate original acquisition provenance, not the consuming runner's Git installation.
 * Historical fetcher identity is metadata; callers at a fresh-fetch boundary may additionally
 * require that receipt to name the currently executing fetcher. Consumers independently verify
 * the pinned commits and prepared files with their own trusted Git, without restamping receipts.
 * @param {any} receipt @param {any} roots @param {"linux" | "compiler" | "rocm"} target @param {string} source
 * @param {string} [expectedFetcherSha256] @param {{path:string,sha256:string}} [producerGit] */
function assertSdkFetchProvenance(receipt, roots, target, source, expectedFetcherSha256, producerGit) {
  const selected = gitSourceComponents(target), names = /** @type {(keyof typeof origins)[]} */ (Object.keys(selected).sort()),
    recipe = sourceRecipe(target);
  if (!hasExactKeys(receipt, ["version", "target", "status", "networked", "buildOnly", "compiled", "publicationQualified", "git", "fetcherSha256", "recipe", "components"]) ||
      receipt.version !== 1 || receipt.target !== target || receipt.status !== "PASS" || receipt.networked !== true ||
      receipt.buildOnly !== true || receipt.compiled !== false || receipt.publicationQualified !== false ||
      !hasExactKeys(receipt.git, ["path", "sha256"]) || typeof receipt.git.path !== "string" ||
      !path.posix.isAbsolute(receipt.git.path) || path.posix.normalize(receipt.git.path) !== receipt.git.path ||
      /[\r\n\0]/.test(receipt.git.path) || typeof receipt.git.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(receipt.git.sha256) ||
      producerGit && (receipt.git.path !== producerGit.path || receipt.git.sha256 !== producerGit.sha256) ||
      typeof receipt.fetcherSha256 !== "string" || !/^[a-f0-9]{64}$/.test(receipt.fetcherSha256) ||
      expectedFetcherSha256 !== undefined && (!/^[a-f0-9]{64}$/.test(expectedFetcherSha256) || receipt.fetcherSha256 !== expectedFetcherSha256) ||
      !hasExactKeys(receipt.recipe, ["sha256", "files"]) ||
      receipt.recipe.sha256 !== jsonHash(JSON.stringify(recipe)) || !isDeepStrictEqual(receipt.recipe.files, recipe.graph) ||
      !hasExactKeys(roots, names) || !hasExactKeys(receipt.components, names)) {
    throw new Error(`Native ${target} source fetch receipt does not bind this run's complete pinned source cohort`);
  }
  for (const name of names) {
    const root = path.join(source, "roots", name), component = receipt.components[name], selectedComponent = selected[name];
    if (!selectedComponent || !hasExactKeys(component, ["origin", "revision", "root", "gitDirectory", "shallow", "partialClone"]) ||
        component.origin !== origins[name] || component.revision !== selectedComponent.revision || component.root !== root ||
        component.gitDirectory !== "bare" || component.shallow !== true || component.partialClone !== false || roots[name] !== root) {
      throw new Error(`Native ${target} source fetch receipt changed its pinned component: ` + name);
    }
  }
}

/** Bind an original preparation receipt to the one target and run-scoped output root.
 * Full file/pin/recipe admission remains in verifyPreparedSdkSources().
 * @param {any} receipt @param {"linux" | "compiler"} target @param {string} output */
function assertPreparedSdkPaths(receipt, target, output) {
  const selected = sdkComponents(target), names = /** @type {(keyof typeof selected)[]} */ (Object.keys(selected).sort());
  if (receipt?.version !== 2 || receipt.target !== target || receipt.status !== "PASS" || receipt.compiled !== false ||
      receipt.buildOnly !== true || receipt.publicationQualified !== false || !hasExactKeys(receipt.prepared, names)) {
    throw new Error(`Prepared ${target} source receipt does not bind the pinned source cohort`);
  }
  for (const name of names) {
    const row = receipt.prepared[name], component = selected[name];
    if (!row || row.root !== path.join(output, name) || row.revision !== component?.revision) {
      throw new Error(`Prepared ${target} source receipt changed its run-scoped component: ` + name);
    }
  }
}

/** Share exact root integrity checks between single- and multi-repository source cohorts.
 * @param {string} root @param {string} origin @param {string} revision @param {string} git
 * @param {NodeJS.ProcessEnv} env @param {string} label */
function verifyBareSdkSourceRoot(root, origin, revision, git, env, label) {
  const rootStat = fs.lstatSync(root), shallowStat = fs.lstatSync(path.join(root, "shallow")), config = fs.readFileSync(path.join(root, "config"), "utf8"),
    packs = path.join(root, "objects/pack"), packStat = fs.lstatSync(packs);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(root) !== root ||
      !shallowStat.isFile() || shallowStat.isSymbolicLink() || !packStat.isDirectory() || packStat.isSymbolicLink() ||
      !/\[core\][\s\S]*?\bbare\s*=\s*true/i.test(config) ||
      config.match(/\[remote "origin"\]([\s\S]*?)(?=\n\[|$)/)?.[1]?.match(/^\s*url\s*=\s*(.*?)\s*$/m)?.[1] !== origin ||
      /partialclone|promisor/i.test(config) || fs.readdirSync(packs).some(name => name.endsWith(".promisor"))) {
    throw new Error(`Native ${label} bare source root is incomplete or differs from its pinned fetch receipt`);
  }
  /** @param {string[]} args */
  const gitText = args => {
    const result = spawnSync(git, args, {encoding: "utf8", env, timeout: 60_000, maxBuffer: 8 << 20});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error(`Native ${label} pinned Git root verification failed`, {cause: result.error});
    }
    return result.stdout || "";
  };
  if (gitText(["-C", root, "rev-parse", "--is-bare-repository"]).trim() !== "true" ||
      gitText(["-C", root, "rev-parse", "--verify", revision + "^{commit}"]).trim() !== revision ||
      gitText(["-C", root, "rev-parse", "--verify", "refs/heads/mom-native-source^{commit}"]).trim() !== revision) {
    throw new Error(`Native ${label} Git root is not the fetched pinned shallow revision`);
  }
}

/** Verify a fetched cohort's immutable receipt and bare root without fetching or changing it.
 * @param {Pick<ReturnType<typeof ciPaths>, "env">} paths @param {string} source @param {string} git @param {"hermes" | "acpp"} target */
function verifyGitSource(paths, source, git, target) {
  const root = path.join(source, "roots", target), receipt = readBuildJson(path.join(source, "fetch.json")),
    roots = readBuildJson(path.join(source, "roots.json")), revision = target === "hermes" ? hermesPins.revision : syclRevision,
    origin = origins[target];
  assertGitFetchProvenance(receipt, roots, target, source, git, fileHash(git),
    fileHash(path.join(repo, "scripts/fetch-native-sdk-sources.js")));
  const env = {...sdkSourceEnvironment(paths.env), GIT_ALLOW_PROTOCOL: "file"};
  verifyBareSdkSourceRoot(root, origin, revision, git, env, target);
  return receipt;
}

/** Verify the run-scoped fetched bare Git root before passing it to the original producer.
 * @param {ReturnType<typeof ciPaths>} paths @param {ReturnType<typeof hermesStatePaths>} state @param {string} git */
function verifyHermesSource(paths, state, git) {
  return verifyGitSource(paths, state.source, git, "hermes");
}

/** Check the one ACPP checkout path that the source-only handoff promises to preserve.
 * Full HEAD/diff/file/patch admission remains in verifyPreparedSdkSources().
 * @param {any} receipt @param {Pick<ReturnType<typeof acppStatePaths>, "preparedCheckout">} state */
function assertAcppPreparedSource(receipt, state) {
  if (receipt?.target !== "acpp" || receipt?.prepared?.acpp?.root !== state.preparedCheckout ||
      receipt.prepared.acpp.revision !== syclRevision) {
    throw new Error("Prepared ACPP checkout does not bind this run's pinned source path and revision");
  }
}

/** Verify fetch and original offline-preparation receipts before retaining source-only state.
 * @param {Pick<ReturnType<typeof ciPaths>, "env">} paths
 * @param {Pick<ReturnType<typeof acppStatePaths>, "source" | "preparedReceipt" | "preparedCheckout">} state @param {string} git */
function verifyAcppSource(paths, state, git) {
  const fetchReceipt = verifyGitSource(paths, state.source, git, "acpp"),
    preparedReceipt = verifyPreparedSdkSources(state.preparedReceipt, "acpp");
  assertAcppPreparedSource(preparedReceipt, state);
  return {fetchReceipt, preparedReceipt};
}

/** @typedef {Pick<ReturnType<typeof linuxSourceStatePaths>, "source" | "fetchReceipt" | "rootsFile" |
 * "preparedReceipt" | "compilerPreparedReceipt" | "linuxPrepared" | "compilerPrepared">} LinuxPreparedSources */

/** Re-admit the original multi-repository fetch and prepared source receipts without network access.
 * Only producer inputs belong here; callers need no fabricated CI archive/run fields.
 * @param {Pick<ReturnType<typeof ciPaths>, "env">} paths @param {LinuxPreparedSources} state @param {string} git */
function verifyLinuxSources(paths, state, git) {
  const receipt = readBuildJson(state.fetchReceipt), roots = readBuildJson(state.rootsFile);
  assertSdkFetchProvenance(receipt, roots, "compiler", state.source);
  const env = {...sdkSourceEnvironment(paths.env), GIT_ALLOW_PROTOCOL: "file", GIT_NO_LAZY_FETCH: "1"};
  const components = sdkComponents("compiler");
  const names = /** @type {(keyof typeof components)[]} */ (Object.keys(components));
  for (const name of names) {
    const component = components[name];
    if (!component || !origins[name]) {throw new Error("Native compiler SDK source selection omitted a pinned component: " + name);}
    verifyBareSdkSourceRoot(path.join(state.source, "roots", name), origins[name], component.revision, git, env, "compiler " + name);
  }
  const preparedReceipt = verifyPreparedSdkSources(state.preparedReceipt, "linux"),
    compilerPreparedReceipt = verifyPreparedSdkSources(state.compilerPreparedReceipt, "compiler");
  assertPreparedSdkPaths(preparedReceipt, "linux", state.linuxPrepared);
  assertPreparedSdkPaths(compilerPreparedReceipt, "compiler", state.compilerPrepared);
  return {fetchReceipt: receipt, preparedReceipt, compilerPreparedReceipt};
}

/** Exact source roots and preparation outputs for the shared Linux/compiler source handoff.
 * @param {ReturnType<typeof linuxSourceStatePaths>} state @param {Record<string, string>} roots */
function linuxSourcePreparationPlan(state, roots) {
  const compilerComponents = sdkComponents("compiler"), compilerNames = Object.keys(compilerComponents).sort(),
    linuxComponents = sdkComponents("linux"), linuxNames = Object.keys(linuxComponents).sort();
  if (!hasExactKeys(roots, compilerNames) || compilerNames.some(name => roots[name] !== path.join(state.source, "roots", name))) {
    throw new Error("Native Linux source handoff needs the exact ten compiler-target bare roots");
  }
  const linuxRoots = Object.fromEntries(linuxNames.map(name => [name, roots[name]]));
  return [
    {target: "linux", roots: linuxRoots, output: state.linuxPrepared},
    {target: "compiler", roots: Object.fromEntries(compilerNames.map(name => [name, roots[name]])), output: state.compilerPrepared},
  ];
}

/** The fetcher runs once for the superset cohort; preparation then produces two independent
 * target receipts from the same original bare roots without rewriting either receipt.
 * @param {ReturnType<typeof linuxSourceStatePaths>} state @param {string} git */
function linuxSourceFetchArgs(state, git) {return ["compiler", git, state.source, "--allow-network"];}

/** Fetch the exact pinned shared source cohort and prepare both existing consumer views offline.
 * @param {ReturnType<typeof ciPaths>} paths */
function fetchLinuxSources(paths) {
  const state = linuxSourceStatePaths(paths), git = canonicalExecutable("/usr/bin/git");
  runNode("scripts/fetch-native-sdk-sources.js", linuxSourceFetchArgs(state, git), paths.env, 15 * 60 * 1000);
  const receipt = readBuildJson(state.fetchReceipt), roots = readBuildJson(state.rootsFile), fetcher = path.join(repo, "scripts/fetch-native-sdk-sources.js");
  assertSdkFetchProvenance(receipt, roots, "compiler", state.source, fileHash(fetcher), {path: git, sha256: fileHash(git)});
  const plan = linuxSourcePreparationPlan(state, roots);
  fs.mkdirSync(state.prepared);
  for (const preparation of plan) {prepareSdkSources(preparation.target, preparation.roots, preparation.output, paths.env);}
  verifyLinuxSources(paths, state, git);
  packState(state.sourceTar, temp, [state.sourceName]);
  listState(state.sourceTar, [state.sourceName]);
  const outputs = {bundle_sha256: fileHash(state.sourceTar)};
  appendWorkflowOutputs(paths.env, outputs);
  return outputs;
}

/** @param {Pick<ReturnType<typeof rocmSourceStatePaths>, "source">} state @param {string} git */
function rocmSourceFetchArgs(state, git) {return ["rocm", git, state.source, "--allow-network"];}

/** Re-admit the immutable same-run ROCm acquisition receipt and each pinned bare Git root.
 * @param {Pick<ReturnType<typeof ciPaths>, "env">} paths
 * @param {Pick<ReturnType<typeof rocmSourceStatePaths>, "source" | "rootsFile" | "fetchReceipt" | "pythonArchive" | "pythonSourceReceipt">} state
 * @param {string} git */
function verifyRocmSources(paths, state, git) {
  const receipt = readBuildJson(state.fetchReceipt), roots = readBuildJson(state.rootsFile);
  assertSdkFetchProvenance(receipt, roots, "rocm", state.source);
  const env = {...sdkSourceEnvironment(paths.env), GIT_ALLOW_PROTOCOL: "file", GIT_NO_LAZY_FETCH: "1"};
  for (const [name, component] of Object.entries(gitSourceComponents("rocm"))) {
    const origin = origins[/** @type {keyof typeof origins} */ (name)];
    if (!origin) {throw new Error("Native ROCm source selection omitted its pinned origin: " + name);}
    verifyBareSdkSourceRoot(path.join(state.source, "roots", name), origin, component.revision, git, env, "ROCm " + name);
  }
  verifyRocmParserSource(state);
  return receipt;
}

/** Fetch the four pinned ROCm Git roots once, re-admit them offline, and retain only this source cohort. @param {ReturnType<typeof ciPaths>} paths */
function fetchRocmSources(paths) {
  const state = rocmSourceStatePaths(paths), git = canonicalExecutable("/usr/bin/git"), fetcher = path.join(repo, "scripts/fetch-native-sdk-sources.js");
  runNode("scripts/fetch-native-sdk-sources.js", rocmSourceFetchArgs(state, git), paths.env, 15 * 60 * 1000);
  assertSdkFetchProvenance(readBuildJson(state.fetchReceipt), readBuildJson(state.rootsFile), "rocm", state.source, fileHash(fetcher),
    {path: git, sha256: fileHash(git)});
  fetchRocmParserSource(state, paths.env);
  verifyRocmSources(paths, state, git);
  packState(state.sourceTar, temp, [state.sourceName]);
  listState(state.sourceTar, [state.sourceName]);
}

/** @param {Pick<RocmComgrBuildState, "source" | "llvmSdk" | "hipIncludeRoot">} state
 * @param {{cc: string, cxx: string, cmake: string}} tools @param {"linux" | "win32"} [target] */
function rocmResourcesConfig(state, tools, target = "linux") {
  if (target !== "linux" && target !== "win32") {throw new Error("Unsupported ROCm resources target");}
  return {version: 1, target, source: path.join(state.source, "roots/rocmLlvm"),
    clr: path.join(state.source, "roots/clr"), llvm: state.llvmSdk, cmake: tools.cmake, hipHeaders: state.hipIncludeRoot,
    hostCompilers: {cc: tools.cc, cxx: tools.cxx}};
}

/** Export one pinned Git tree into a new source root and verify every regular file and its membership.
 * @param {string} git @param {string} gitRoot @param {string} treeish @param {string} archive @param {string} destination
 * @param {NodeJS.ProcessEnv} env @param {string} label */
function exportRocmSourceTree(git, gitRoot, treeish, archive, destination, env, label) {
  if ([archive, destination].some(file => fs.existsSync(file))) {throw new Error("ROCm header source export refuses existing output: " + label);}
  const archived = spawnSync(git, ["-C", gitRoot, "archive", "--format=tar", "--output=" + archive, treeish],
    {env, stdio: "inherit", timeout: 5 * 60 * 1000});
  if (archived.error || archived.signal || archived.status !== 0) {throw new Error("Pinned ROCm header source archive failed: " + label, {cause: archived.error});}
  const tree = spawnSync(git, ["-C", gitRoot, "ls-tree", "-rz", treeish],
    {env, encoding: "utf8", timeout: 60_000, maxBuffer: 32 << 20});
  if (tree.error || tree.signal || tree.status !== 0 || typeof tree.stdout !== "string") {
    throw new Error("Pinned ROCm header source tree could not be read: " + label, {cause: tree.error});
  }
  fs.mkdirSync(destination, {mode: 0o700});
  runTar("extract", ["--extract", "--file", archive, "--directory", destination, "--no-same-owner", "--no-same-permissions"]);
  const files = sdkSourceFiles(destination);
  verifySdkExport(destination, files, tree.stdout);
  return {fileCount: Object.keys(files).length, filesSha256: createHash("sha256").update(JSON.stringify(files)).digest("hex"),
    treeSha256: createHash("sha256").update(tree.stdout).digest("hex")};
}

/** The pinned CLR repository stores its HIP AMD CMake project under `hipamd/`. @param {string} revision */
function rocmClrHipamdTreeish(revision) {
  if (!/^[0-9a-f]{40}$/.test(revision)) {throw new Error("Invalid pinned CLR revision");}
  return revision + ":hipamd";
}

/** Apply only the reviewed headers-only packaging guard to the verified CLR HIP subtree.
 * Preserve the exact patch copy, check/apply logs, and before/after target hashes in the tarred state.
 * @param {string} git @param {string} patchFile @param {RocmComgrBuildState} state @param {NodeJS.ProcessEnv} env */
function applyRocmHipHeadersPatch(git, patchFile, state, env) {
  const target = path.join(state.clrHipamdRoot, "CMakeLists.txt");
  if (fileHash(patchFile) !== rocmHipHeaderPatchSha256 || fileHash(target) !== rocmHipCmakeOriginalSha256 ||
      fs.existsSync(state.hipPatchRoot)) {throw new Error("Pinned CLR header-only patch inputs drifted or already exist");}
  fs.mkdirSync(state.hipPatchRoot, {mode: 0o700});
  fs.copyFileSync(patchFile, state.hipPatchFile, fs.constants.COPYFILE_EXCL);
  if (fileHash(state.hipPatchFile) !== rocmHipHeaderPatchSha256) {throw new Error("Saved CLR header-only patch copy changed");}
  /** @param {string[]} args @param {string} log */
  const runPatch = (args, log) => {
    const command = [git, "-C", state.clrHipamdRoot, "apply", "-p2", ...args, "--", state.hipPatchFile],
      result = spawnSync(command[0] || "", command.slice(1), {env, encoding: "utf8", timeout: 60_000, maxBuffer: 8 << 20});
    fs.writeFileSync(log, JSON.stringify({command, status: result.status, signal: result.signal,
      error: result.error?.message || null, stdout: result.stdout || "", stderr: result.stderr || ""}, null, 2) + "\n", {flag: "wx", mode: 0o600});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Pinned CLR header-only patch failed; inspect saved patch log", {cause: result.error});}
  };
  runPatch(["--check"], state.hipPatchCheckLog);
  runPatch([], state.hipPatchApplyLog);
  const patchedSha256 = fileHash(target);
  if (patchedSha256 !== rocmHipCmakePatchedSha256) {throw new Error("CLR header-only patch produced unexpected CMake bytes");}
  /** @param {string} file */
  const relative = file => path.relative(state.hipHeadersRoot, file).split(path.sep).join("/");
  return {path: relative(state.hipPatchFile), sha256: rocmHipHeaderPatchSha256, stripComponents: 2,
    target: relative(target), originalSha256: rocmHipCmakeOriginalSha256, patchedSha256,
    checkLog: relative(state.hipPatchCheckLog), applyLog: relative(state.hipPatchApplyLog)};
}

/** Merge the upstream common and CLR AMD headers without overwrite, then require the generated
 * version header and every source-declared HIPRTC header before the LLVM resource build begins.
 * @param {string} hipCommonRoot @param {string} clrHipamdRoot @param {string} generatedHeader
 * @param {string} output @param {string} hiprtcCmake */
function assembleRocmHipHeaders(hipCommonRoot, clrHipamdRoot, generatedHeader, output, hiprtcCmake) {
  if (![hipCommonRoot, clrHipamdRoot, generatedHeader, output].every(path.isAbsolute) || fs.existsSync(output) ||
      typeof hiprtcCmake !== "string" || hiprtcCmake.length > (8 << 20)) {
    throw new Error("ROCm HIP header package inputs are missing, ambiguous or already exist");
  }
  const commonRoot = path.join(hipCommonRoot, "include/hip"), clrRoot = path.join(clrHipamdRoot, "include/hip/amd_detail"),
    common = sdkSourceFiles(commonRoot), clr = sdkSourceFiles(clrRoot), generated = fs.lstatSync(generatedHeader, {throwIfNoEntry: false});
  if (!Object.keys(common).length || !Object.keys(clr).length || Object.keys(common).some(name => name === "hip_version.h" || name.startsWith("amd_detail/")) ||
      !generated?.isFile() || generated.isSymbolicLink()) {throw new Error("ROCm HIP header package lacks common, CLR or generated-version inputs");}
  const required = hiprtcHeaders(hiprtcCmake, output);
  fs.mkdirSync(output, {mode: 0o700});
  const copy = (/** @type {string} */ sourceRoot, /** @type {string} */ relative, /** @type {string} */ destination) => {
    if (!safeRelative(relative)) {throw new Error("Unsafe pinned ROCm header path");}
    fs.mkdirSync(path.dirname(destination), {recursive: true, mode: 0o700});
    fs.copyFileSync(path.join(sourceRoot, relative), destination, fs.constants.COPYFILE_EXCL);
  };
  for (const name of Object.keys(common)) {copy(commonRoot, name, path.join(output, "hip", name));}
  for (const name of Object.keys(clr)) {copy(clrRoot, name, path.join(output, "hip/amd_detail", name));}
  copy(path.dirname(generatedHeader), "hip_version.h", path.join(output, "hip/hip_version.h"));
  for (const header of required) {
    const stat = fs.lstatSync(header, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink()) {throw new Error("Required HIPRTC header is missing from the pinned HIP/CLR package: " + path.basename(header));}
  }
  return {files: sdkSourceFiles(output), required};
}

/** The upstream configure is intentionally configure-only: the generated HIP version header is
 * required, but HIP runtime libraries are not built or installed in this prerequisite stage.
 * @param {RocmComgrBuildState} state @param {{cc: string, cxx: string, cmake: string}} tools */
function hipHeaderConfigureCommand(state, tools) {
  return [tools.cmake, "-S", state.clrHipamdRoot, "-B", state.hipBuildRoot, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=" + state.hipInstallPrefix, "-DCMAKE_C_COMPILER=" + tools.cc, "-DCMAKE_CXX_COMPILER=" + tools.cxx,
    "-DHIP_COMMON_DIR=" + state.hipCommonRoot, "-DHIP_PLATFORM:STRING=amd", "-DHIP_RUNTIME:STRING=none", "-DHIPCC_BIN_DIR:STRING=",
    "-DHIP_OFFICIAL_BUILD:BOOL=ON", "-DROCM_PATH=" + state.hipInstallPrefix];
}

/** Preserve the upstream producer's native path layout and pass only literal selected roots.
 * @param {RocmComgrBuildState} state @param {{cc: string, cxx: string, cmake: string}} tools */
function rocmComgrBuildCommands(state, tools) {
  const resources = rocmResourcesConfig(state, tools), device = path.join(state.resourcesRoot, "install");
  return [
    {script: "scripts/build-native-rocm-resources.js", args: [state.resourcesConfig, state.resourcesRoot], config: resources},
    {script: "scripts/build-native-comgr-sdk.js", args: [path.join(state.source, "roots/rocmLlvm"), state.llvmSdk, state.llvmSdk,
      device, state.sdkRoot, tools.cc, tools.cxx]},
  ];
}

/** Re-admit the CPU-only HIP/device resource receipt and public COMGR SDK receipt together.
 * @param {RocmComgrBuildState} state @param {{cc: string, cxx: string, cmake: string}} tools */
function auditRocmComgrSdk(state, tools) {
  const resources = readBuildJson(path.join(state.resourcesRoot, "result.json")), device = path.join(state.resourcesRoot, "install"),
    git = gitSourceComponents("rocm"), actualDevice = comgrDeviceInputs(device), sdk = comgrSdkInputs(path.join(state.sdkRoot, "result.json")),
    comgrRecord = readBuildJson(sdk.receipt), expectedHostCompilers = {cc: tools.cc, cxx: tools.cxx};
  if (resources.version !== 1 || resources.target !== "linux" || resources.buildOnly !== true || resources.publicationQualified !== false ||
      resources.revision !== git["rocmLlvm"]?.revision || resources.clrRevision !== git["clr"]?.revision ||
      !isDeepStrictEqual(resources.hostCompilers, expectedHostCompilers) || !isDeepStrictEqual(comgrRecord.hostCompilers, expectedHostCompilers) ||
      !isDeepStrictEqual(resources.device, actualDevice) ||
      !isDeepStrictEqual(resources.source, sdkSourceFiles(path.join(state.resourcesRoot, "source"))) ||
      !resources.header || resources.header.path !== path.join(state.resourcesRoot, "hipRTC") || fileHash(resources.header.path) !== resources.header.sha256 ||
      !resources.object || resources.object.path !== path.join(state.resourcesRoot, "hipRTC_header.o") || fileHash(resources.object.path) !== resources.object.sha256) {
    throw new Error("ROCm resource receipt does not bind the pinned Linux source export and AMD device library tree");
  }
  verifySourceInputs(resources.recipe, sourceInputs(repo, [], rocmResourcesRecipe), "ROCm resource recipe");
  for (const [label, files] of [["inputs", resources.inputs], ["dependencies", resources.dependencies]]) {
    if (!files || typeof files !== "object" || Array.isArray(files) || !Object.keys(files).length || Object.keys(files).length > 65536) {
      throw new Error("ROCm resource receipt has a missing or excessive " + label + " map");
    }
    for (const [file, sha] of Object.entries(files)) {
      if (!path.isAbsolute(file) || typeof sha !== "string" || !/^[0-9a-f]{64}$/.test(sha) || fileHash(file) !== sha) {
        throw new Error("ROCm resource input changed: " + file);
      }
    }
  }
  if (sdk.roots.llvm !== state.llvmSdk || sdk.roots.lld !== state.llvmSdk || sdk.roots.device !== device) {
    throw new Error("COMGR SDK does not use this run's same LLVM/LLD and Linux AMD device resources");
  }
  return {resources, sdk};
}

/** Build the existing CPU-only HIP headers, ROCm resources and COMGR SDK from admitted roots.
 * Source/tar admission and host provisioning remain the caller's responsibility. The LLVM callback
 * is checked before output creation and again between the resource and COMGR producers.
 * @param {RocmComgrBuildState} state @param {{cc:string,cxx:string,cmake:string}} tools
 * @param {NodeJS.ProcessEnv} environment @param {()=>unknown} verifyLlvmAdmission */
function buildRocmComgrFromInputs(state, tools, environment, verifyLlvmAdmission) {
  const freshOutputs = [state.resourcesRoot, state.hipHeadersRoot, state.sdkRoot, state.resourcesConfig];
  for (const output of freshOutputs) {
    if (typeof output !== "string" || !path.isAbsolute(output) || path.resolve(output) !== output ||
        fs.lstatSync(output, {throwIfNoEntry: false}) || fs.realpathSync(path.dirname(output)) !== path.dirname(output)) {
      throw new Error("ROCm/COMGR producer requires fresh canonical output/config paths: " + String(output));
    }
  }
  if (typeof verifyLlvmAdmission !== "function") {throw new Error("ROCm/COMGR producer requires a read-only LLVM admission callback");}
  verifyLlvmAdmission();
  const git = canonicalExecutable("/usr/bin/git"), gitEnvironment = {...sdkSourceEnvironment(environment), GIT_ALLOW_PROTOCOL: "file", GIT_NO_LAZY_FETCH: "1",
    GIT_TERMINAL_PROMPT: "0", GIT_OPTIONAL_LOCKS: "0"};
  fs.mkdirSync(state.hipHeadersRoot, {mode: 0o700});
  fs.mkdirSync(state.hipArchiveRoot, {mode: 0o700});
  fs.mkdirSync(state.hipSourceRoot, {mode: 0o700});
  const hipGitRoot = path.join(state.source, "roots/hip"), clrGitRoot = path.join(state.source, "roots/clr"),
    pins = gitSourceComponents("rocm"), hipRevision = pins["hip"]?.revision, clrRevision = pins["clr"]?.revision;
  if (!hipRevision || !clrRevision) {throw new Error("Pinned ROCm source cohort omitted HIP or CLR");}
  const hipExport = exportRocmSourceTree(git, hipGitRoot, hipRevision, state.hipCommonTar, state.hipCommonRoot, gitEnvironment, "HIP common"),
    clrTreeish = rocmClrHipamdTreeish(clrRevision),
    clrExport = exportRocmSourceTree(git, clrGitRoot, clrTreeish, state.clrHipamdTar, state.clrHipamdRoot, gitEnvironment, "CLR HIP AMD"),
    patchEvidence = applyRocmHipHeadersPatch(git, path.join(repo, rocmHipHeaderPatch), state, gitEnvironment),
    patchedClrFiles = sdkSourceFiles(state.clrHipamdRoot);
  const configure = hipHeaderConfigureCommand(state, tools), configureEnvironment = {...gitEnvironment, GIT_DIR: clrGitRoot};
  fs.writeFileSync(state.hipConfigurePlan, JSON.stringify({version: 1, command: configure, environment: {GIT_DIR: clrGitRoot},
    configureOnly: true, buildInvoked: false, installInvoked: false}, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const configureFd = fs.openSync(state.hipConfigureLog, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY, 0o600);
  let configureResult;
  try {
    configureResult = spawnSync(configure[0] || "", configure.slice(1),
      {cwd: repo, env: configureEnvironment, stdio: ["ignore", configureFd, configureFd], timeout: 15 * 60 * 1000});
  } finally {
    fs.closeSync(configureFd);
  }
  if (configureResult.error || configureResult.signal || configureResult.status !== 0) {
    throw new Error("Pinned upstream HIP header configure failed; inspect its saved configure log", {cause: configureResult.error});
  }
  const generatedHeader = path.join(state.hipBuildRoot, "include/hip/hip_version.h"),
    hiprtcCmake = readBuildText(path.join(state.clrHipamdRoot, "src/hiprtc/CMakeLists.txt"), 1 << 20),
    headerPackage = assembleRocmHipHeaders(state.hipCommonRoot, state.clrHipamdRoot, generatedHeader, state.hipIncludeRoot, hiprtcCmake);
  /** @param {string} file */
  const relative = file => path.relative(state.hipHeadersRoot, file).split(path.sep).join("/");
  fs.writeFileSync(path.join(state.hipHeadersRoot, "header-package.json"), JSON.stringify({version: 1,
    pins: {hip: hipRevision, clr: clrRevision}, sourceExports: {
      hipCommon: {root: relative(state.hipCommonRoot), revision: hipRevision, ...hipExport},
      clrHipamd: {root: relative(state.clrHipamdRoot), revision: clrRevision, treeish: clrTreeish, ...clrExport,
        afterPatch: {fileCount: Object.keys(patchedClrFiles).length,
          filesSha256: createHash("sha256").update(JSON.stringify(patchedClrFiles)).digest("hex")}},
    }, patch: patchEvidence,
    configure: {command: configure, environment: {GIT_DIR: clrGitRoot}, configureOnly: true,
      log: relative(state.hipConfigureLog), logSha256: fileHash(state.hipConfigureLog), status: configureResult.status},
    hiprtcCmake: {path: relative(path.join(state.clrHipamdRoot, "src/hiprtc/CMakeLists.txt")), sha256: fileHash(path.join(state.clrHipamdRoot, "src/hiprtc/CMakeLists.txt"))},
    generatedHeader: {path: relative(generatedHeader), sha256: fileHash(generatedHeader)},
    files: headerPackage.files, requiredHiprtcHeaders: headerPackage.required.map(file => path.relative(state.hipIncludeRoot, file).split(path.sep).join("/")),
    buildOnly: true, publicationQualified: false}, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const commands = rocmComgrBuildCommands(state, tools), buildEnv = {...environment, MOM_BUILD_JOBS: ciBuildJobs(environment),
    GIT_ALLOW_PROTOCOL: "file", GIT_NO_LAZY_FETCH: "1", GIT_TERMINAL_PROMPT: "0", GIT_OPTIONAL_LOCKS: "0"};
  fs.writeFileSync(state.resourcesConfig, JSON.stringify(rocmResourcesConfig(state, tools), null, 2) + "\n", {flag: "wx", mode: 0o600});
  try {
    const resourceCommand = commands[0], comgrCommand = commands[1];
    if (!resourceCommand || !comgrCommand) {throw new Error("Native ROCm/COMGR command plan is incomplete");}
    runNode(resourceCommand.script, resourceCommand.args, buildEnv, 15 * 60 * 1000);
    verifyLlvmAdmission();
    runNode(comgrCommand.script, comgrCommand.args, buildEnv, 20 * 60 * 1000);
    return auditRocmComgrSdk(state, tools);
  } finally {
    if (fs.existsSync(state.resourcesConfig)) {fs.unlinkSync(state.resourcesConfig);}
  }
}

/** Restore/admit the same-run inputs, provision tools, and package the shared CPU-only ROCm producer.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildRocmComgrSdk(paths) {
  const state = rocmComgrStatePaths(paths), git = canonicalExecutable("/usr/bin/git");
  for (const output of [state.resourcesRoot, state.hipHeadersRoot, state.sdkRoot, state.resourcesConfig, state.sdkTar]) {
    if (fs.existsSync(output)) {throw new Error("Native ROCm/COMGR stage refuses an existing output: " + path.basename(output));}
  }
  extractState(state.incomingSourceTar, temp, [state.sourceName]);
  verifyRocmSources(paths, state, git);
  extractState(state.incomingLlvmSourceTar, temp, state.llvmSourceArchiveRoots, [state.llvmSourceName]);
  extractState(state.incomingLlvmSdkTar, temp, [state.llvmSdkName]);
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", state.llvmSource], paths.env, 2 * 60 * 1000);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-rocm-comgr-" + paths.suffix, 15 * 60 * 1000);
  // Retain the original pre-cleanup gate; the reusable producer also checks its own entry boundary.
  verifyLinuxLlvmAdmissionCi(paths);
  for (const file of [state.incomingSourceTar, state.incomingLlvmSourceTar, state.incomingLlvmSdkTar]) {fs.unlinkSync(file);}
  const tools = {cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx"), cmake: canonicalExecutable("/usr/bin/cmake")};
  buildRocmComgrFromInputs(state, tools, paths.env, () => verifyLinuxLlvmAdmissionCi(paths));
  packState(state.sdkTar, temp, [state.sourceName, state.llvmSourceName, state.llvmSdkName, state.hipHeadersName,
    state.resourcesName, state.sdkName]);
  listState(state.sdkTar, [state.sourceName, state.llvmSourceName, state.llvmSdkName, state.hipHeadersName,
    state.resourcesName, state.sdkName]);
}

/** Canonical paths accepted by the public ROCm host-runtime producer.
 * @param {Pick<ReturnType<typeof rocmHostStatePaths>, "llvmSdk">} state */
function rocmHostBuildTools(state) {
  return {ar: canonicalExecutable("/usr/bin/ar"), cmake: canonicalExecutable("/usr/bin/cmake"),
    dpkgQuery: canonicalExecutable("/usr/bin/dpkg-query"), git: canonicalExecutable("/usr/bin/git"),
    llvmObjcopy: canonicalExecutable(path.join(state.llvmSdk, "bin/llvm-objcopy")), ninja: canonicalExecutable("/usr/bin/ninja"),
    perl: canonicalExecutable("/usr/bin/perl"), pkgConfig: canonicalExecutable("/usr/bin/pkg-config"),
    python3: canonicalExecutable("/usr/bin/python3.12"), ranlib: canonicalExecutable("/usr/bin/ranlib"),
    tar: canonicalExecutable("/usr/bin/tar"), xxd: canonicalExecutable("/usr/bin/xxd"),
    cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx")};
}

/** Query the installed libdrm metadata and keep static OS archive paths tied to its selected libdir.
 * @param {Pick<ReturnType<typeof rocmHostStatePaths>, "source" | "llvmSdk" | "sdkRoot" | "resourcesRoot">} state
 * @param {ReturnType<typeof rocmHostBuildTools>} tools */
function rocmHostBuildConfig(state, tools) {
  const environment = {...process.env};
  for (const name of Object.keys(environment)) {
    if (/^PKG_CONFIG_(?:PATH|LIBDIR|SYSROOT_DIR)$/i.test(name)) {delete environment[name];}
  }
  environment["LC_ALL"] = "C";
  /** @param {string[]} args */
  const query = args => {
    const result = spawnSync(tools.pkgConfig, args, {encoding: "utf8", env: environment, timeout: 30000, maxBuffer: 1 << 20});
    if (result.error || result.signal || result.status !== 0 || result.stderr || typeof result.stdout !== "string" || !result.stdout.trim()) {
      throw new Error("Native ROCm host pkg-config query failed: " + args.join(" "), {cause: result.error});
    }
    return result.stdout.trim();
  };
  const pkgConfigDir = query(["--variable=pcfiledir", "libdrm"]), amdgpuPkgConfigDir = query(["--variable=pcfiledir", "libdrm_amdgpu"]),
    libraryDir = query(["--variable=libdir", "libdrm"]), amdgpuLibraryDir = query(["--variable=libdir", "libdrm_amdgpu"]);
  if (![pkgConfigDir, libraryDir].every(path.isAbsolute) || amdgpuPkgConfigDir !== pkgConfigDir || amdgpuLibraryDir !== libraryDir) {
    throw new Error("Native ROCm host libdrm packages do not share one canonical pkg-config/lib directory");
  }
  return {version: 1, sources: {clr: path.join(state.source, "roots/clr"), hip: path.join(state.source, "roots/hip"),
    rocr: path.join(state.source, "roots/rocr")},
  tools: {ar: tools.ar, cmake: tools.cmake, dpkgQuery: tools.dpkgQuery, git: tools.git, llvmObjcopy: tools.llvmObjcopy,
    ninja: tools.ninja, perl: tools.perl, pkgConfig: tools.pkgConfig, python3: tools.python3, ranlib: tools.ranlib, tar: tools.tar, xxd: tools.xxd},
  hostCompilers: {cc: tools.cc, cxx: tools.cxx},
  dependencies: {drm: path.join(libraryDir, "libdrm.a"), drmAmdgpu: path.join(libraryDir, "libdrm_amdgpu.a"),
    elf: path.join(libraryDir, "libelf.a"), numa: path.join(libraryDir, "libnuma.a")},
  pkgConfigDir, comgrReceipt: path.join(state.sdkRoot, "result.json")};
}

/** APT is only for host build prerequisites. The public producer runs afterward from local pinned inputs.
 * @param {Pick<ReturnType<typeof ciPaths>, "env">} paths @param {typeof spawnSync} [runner]
 * @returns {{mode:"already-installed",packages:Record<string,string>}|undefined} */
function provisionRocmHostPackages(paths, runner = spawnSync) {
  const packageNames = ["libdrm-dev", "libelf-dev", "libnuma-dev", ...Object.keys(rocmPinnedHostPackages)],
    // eslint-disable-next-line no-template-curly-in-string -- dpkg-query, not JavaScript, expands these fields.
    format = "${Package}\t${Version}\t${db:Status-Abbrev}\n",
    installed = runner("/usr/bin/dpkg-query", ["-W", "-f=" + format, ...packageNames],
      {encoding: "utf8", env: {...paths.env, LC_ALL: "C"}, timeout: 30000, maxBuffer: 1 << 20});
  if (!installed.error && !installed.signal && installed.status === 0 && typeof installed.stdout === "string") {
    const rows = installed.stdout.trimEnd().split(/\r?\n/), packages = new Map(); let valid = rows.length === packageNames.length;
    for (const row of rows) {
      const [name, version, status, ...extra] = row.split("\t"), expected = name ? rocmPinnedHostPackages[name] : undefined;
      if (!name || !packageNames.includes(name) || !version || status?.trim() !== "ii" || extra.length || packages.has(name) ||
          (expected && version !== expected)) {valid = false; break;}
      packages.set(name, version);
    }
    if (valid && packageNames.every(name => packages.has(name))) {
      return {mode: "already-installed", packages: Object.fromEntries(packages)};
    }
  }
  /** @param {string[]} args @param {number} timeout @param {string} label */
  const apt = (args, timeout, label) => {
    const result = runner("sudo", ["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", ...args],
      {cwd: repo, env: paths.env, stdio: "inherit", timeout});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error("Native ROCm host prerequisite " + label + " failed", {cause: result.error});
    }
  };
  apt(["update", "-qq"], 8 * 60 * 1000, "APT index refresh");
  const catalog = runner("/usr/bin/apt-cache", ["madison", ...Object.keys(rocmPinnedHostPackages)],
    {encoding: "utf8", env: paths.env, timeout: 30000, maxBuffer: 1 << 20});
  const available = new Set((typeof catalog.stdout === "string" ? catalog.stdout : "").split(/\r?\n/)
    .map(line => line.split("|").map(part => part.trim())).filter(parts => parts.length >= 2).map(parts => parts[0] + "=" + parts[1]));
  if (catalog.error || catalog.signal || catalog.status !== 0 ||
      Object.entries(rocmPinnedHostPackages).some(([name, version]) => !available.has(name + "=" + version))) {
    throw new Error("Pinned ROCm host prerequisite package versions are not available from the Ubuntu runner repositories");
  }
  apt(["install", "-y", "--no-install-recommends", "--allow-downgrades", "libdrm-dev", "libelf-dev", "libnuma-dev",
    ...Object.entries(rocmPinnedHostPackages).map(([name, version]) => name + "=" + version)], 10 * 60 * 1000, "pinned OS packages");
  return undefined;
}

/** Admit an existing parser directory only when it is canonical and exactly matches the
 * complete regular-file map extracted from the checksum-pinned source distribution.
 * @param {string} destination @param {Record<string, string>} expectedFiles */
function admitRocmParserInstallation(destination, expectedFiles) {
  const stat = fs.lstatSync(destination, {throwIfNoEntry: false});
  if (!stat) {return undefined;}
  if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(destination) !== destination) {
    throw new Error("Refusing to reuse a noncanonical or redirected CppHeaderParser installation");
  }
  const installed = sdkSourceFiles(destination);
  if (!isDeepStrictEqual(expectedFiles, installed)) {throw new Error("Existing CppHeaderParser bytes differ from the pinned source distribution");}
  return installed;
}

/** Install only the verified CppHeaderParser package directory into the public producer's pinned Python path.
 * @param {Pick<ReturnType<typeof ciPaths>, "runnerTemp" | "env">} paths @param {ReturnType<typeof rocmSourceStatePaths>} state */
function installRocmParserSource(paths, state) {
  const receipt = verifyRocmParserSource(state), extraction = fs.mkdtempSync(path.join(paths.runnerTemp, ".rocm-parser-install-"));
  try {
    extractState(state.pythonArchive, extraction, [rocmParserSource.directory]);
    const sourcePackage = path.join(extraction, rocmParserSource.directory, "CppHeaderParser"), sourceFiles = sdkSourceFiles(sourcePackage),
      required = ["__init__.py", "CppHeaderParser.py"];
    if (required.some(name => !Object.hasOwn(sourceFiles, name))) {throw new Error("Pinned CppHeaderParser source omitted the producer's required modules");}
    const destination = "/usr/local/lib/python3.12/dist-packages/CppHeaderParser";
    if (fs.lstatSync(path.dirname(destination), {throwIfNoEntry: false})?.isSymbolicLink()) {
      throw new Error("Refusing to use a redirected CppHeaderParser installation parent");
    }
    const preinstalled = admitRocmParserInstallation(destination, sourceFiles);
    if (preinstalled) {
      return {version: 1, sourceReceipt: {path: state.pythonSourceReceipt, sha256: fileHash(state.pythonSourceReceipt)},
        source: {path: state.pythonArchive, bytes: receipt.archive.bytes, sha256: fileHash(state.pythonArchive)},
        installed: {path: destination, files: preinstalled}};
    }
    const code = ["import pathlib,shutil,sys", "source=pathlib.Path(sys.argv[1]).resolve(strict=True)",
      "target=pathlib.Path('/usr/local/lib/python3.12/dist-packages/CppHeaderParser')",
      "if target.exists() or target.is_symlink(): raise SystemExit('CppHeaderParser target already exists')",
      "target.parent.mkdir(parents=True,exist_ok=True)", "shutil.copytree(source,target,symlinks=False)"].join("\n");
    const result = spawnSync("sudo", ["/usr/bin/python3.12", "-c", code, sourcePackage],
      {cwd: repo, env: paths.env, stdio: "inherit", timeout: 60_000});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error("Pinned offline CppHeaderParser installation failed", {cause: result.error});
    }
    const installed = admitRocmParserInstallation(destination, sourceFiles);
    if (!installed) {throw new Error("Pinned CppHeaderParser installation did not create its target directory");}
    return {version: 1, sourceReceipt: {path: state.pythonSourceReceipt, sha256: fileHash(state.pythonSourceReceipt)},
      source: {path: state.pythonArchive, bytes: receipt.archive.bytes, sha256: fileHash(state.pythonArchive)},
      installed: {path: destination, files: installed}};
  } finally {fs.rmSync(extraction, {recursive: true, force: true});}
}

/** Restore the original host package and pinned parser inputs required by strict ROCm admission.
 * @param {Pick<ReturnType<typeof ciPaths>, "env" | "runnerTemp">} paths @param {ReturnType<typeof rocmHostStatePaths>} state */
function prepareRocmHostAdmission(paths, state) {
  provisionRocmHostPackages(paths);
  return installRocmParserSource(paths, state);
}

/** Admit the public static ROCm host-runtime SDK against this run's fresh COMGR/LLVM cohort.
 * @param {Pick<ReturnType<typeof rocmHostStatePaths>, "llvmSdk" | "sdkRoot" | "resourcesRoot" | "runtimeOutput">} state
 * @param {ReturnType<typeof rocmHostBuildConfig>} config */
function auditRocmHostSdk(state, config) {
  const receiptFile = path.join(state.runtimeOutput, "result.json"), sdk = nativeRocmHostInputs(receiptFile),
    record = readBuildJson(receiptFile);
  if (sdk.prefix !== path.join(state.runtimeOutput, "sdk") || record.comgr?.receipt !== config.comgrReceipt ||
      record.comgr?.roots?.llvm !== state.llvmSdk || record.comgr?.roots?.lld !== state.llvmSdk ||
      record.comgr?.roots?.device !== path.join(state.resourcesRoot, "install") ||
      !isDeepStrictEqual(record.hostCompilers, config.hostCompilers)) {
    throw new Error("Native ROCm host runtime does not bind this same-run LLVM, device and COMGR cohort");
  }
  return {sdk, record};
}

/** @typedef {{path:string,sha256:string,package:string,architecture:string,version:string,status:string}} RocmNoticeOrigin */
/** @typedef {RocmNoticeOrigin & {staged:string,stagedSha256:string}} RocmNoticeStagedOrigin */
/** @typedef {"drm"|"drmAmdgpu"|"elf"|"numa"} RocmNoticeKey */
/** @typedef {{osOrigins:Record<RocmNoticeKey,RocmNoticeOrigin>,stagedDependencies:Record<RocmNoticeKey,RocmNoticeStagedOrigin>,
 * sources:{files:Record<string,string>},sourceArchives:Record<string,{path:string,sha256:string}>,pins:{rocr:string}}} RocmNoticeReceipt */
/** @typedef {{package:string,version:string,architecture:string,sourcePath:string,archives:Array<{key:RocmNoticeKey,path:string,sha256:string}>}} RocmNoticePackageOrigin */
/** @typedef {RocmNoticePackageOrigin & {file:string,sha256:string,bytes:number}} RocmNoticePackageFile */
/** @typedef {{format:string,completeNativeNotices:boolean,runtimeReceipt:{file:string,sha256:string},packages:RocmNoticePackageFile[],
 * sourceLicense:{component:string,file:string,sha256:string,bytes:number,sourceArchive:{name:string,sha256:string},revision:string}}} RocmNoticeManifest */

/** Bind package-owned static ROCm inputs to their exact Debian copyright bodies and the
 * already-admitted hsakmt source license. This is an incomplete notice-source subset. @param {RocmNoticeReceipt} record */
function rocmHostNoticeRows(record) {
  const packageInputs = /** @type {Array<[RocmNoticeKey,string]>} */ ([["drm", "libdrm-dev"], ["drmAmdgpu", "libdrm-dev"],
    ["elf", "libelf-dev"], ["numa", "libnuma-dev"]]);
  if (!record || !record.osOrigins || !record.stagedDependencies || !record.sourceArchives || !record.sources?.files ||
      !hasExactKeys(record.osOrigins, packageInputs.map(([key]) => key)) ||
      !hasExactKeys(record.stagedDependencies, packageInputs.map(([key]) => key))) {
    throw new Error("ROCm notice sources require the complete admitted static package-origin set");
  }
  /** @type {Map<string,RocmNoticePackageOrigin>} */ const grouped = new Map();
  for (const [key, packageName] of packageInputs) {
    const origin = record.osOrigins[key], staged = record.stagedDependencies[key];
    if (!origin || !staged || !hasExactKeys(origin, ["path", "sha256", "package", "architecture", "version", "status"]) ||
        origin.package !== packageName || origin.architecture !== "amd64" || origin.status !== "ii" ||
        !path.isAbsolute(origin.path) || !/^[a-f0-9]{64}$/.test(origin.sha256 || "") || typeof origin.version !== "string" || !origin.version ||
        !hasExactKeys(staged, ["path", "sha256", "package", "architecture", "version", "status", "staged", "stagedSha256"])) {
      throw new Error("ROCm static archive origin is not bound to its installed package: " + key);
    }
    const {staged: stagedPath, stagedSha256, ...stagedOrigin} = staged;
    if (!isDeepStrictEqual(stagedOrigin, origin) || stagedSha256 !== origin.sha256 || !path.isAbsolute(stagedPath)) {
      throw new Error("ROCm static archive origin is not bound to its staged package input: " + key);
    }
    const row = grouped.get(packageName) || {package: packageName, version: origin.version, architecture: origin.architecture,
      sourcePath: "/usr/share/doc/" + packageName + "/copyright", archives: []};
    if (row.version !== origin.version || row.architecture !== origin.architecture) {
      throw new Error("ROCm static archives disagree on their owning package version: " + packageName);
    }
    row.archives.push({key, path: origin.path, sha256: origin.sha256}); grouped.set(packageName, row);
  }
  for (const row of grouped.values()) {row.archives.sort((a, b) => a.key.localeCompare(b.key));}
  return [...grouped.values()].sort((a, b) => a.package.localeCompare(b.package));
}

/** Capture exact current package docs after the public ROCm producer has recorded archive origins.
 * @param {ReturnType<typeof rocmHostStatePaths>} state @param {ReturnType<typeof auditRocmHostSdk>} admitted
 * @param {NodeJS.ProcessEnv} env */
function captureRocmHostNoticeSources(state, admitted, env = process.env) {
  const packageRows = rocmHostNoticeRows(admitted.record), producer = require("./build-native-rocm-host-runtime"),
    dpkgQuery = canonicalExecutable("/usr/bin/dpkg-query"), copied = [];
  for (const row of packageRows) {
    const originsMatch = () => row.archives.every(item =>
      isDeepStrictEqual(producer.packageOrigin(item.path, env, dpkgQuery, row.package), admitted.record.osOrigins[item.key]));
    if (!row.archives.length || !originsMatch()) {
      throw new Error("ROCm package changed between static archive admission and copyright capture: " + row.package);
    }
    const captured = capturePackageCopyright({root: state.hostSdkRoot}, row.package, row.version), file = path.relative(state.hostSdkRoot, captured.path).split(path.sep).join("/");
    if (!originsMatch()) {
      throw new Error("ROCm package changed while its copyright body was captured: " + row.package);
    }
    copied.push({...row, file, sha256: captured.sha256, bytes: fs.statSync(captured.path).size});
  }
  const licenseRelative = "source/rocr/libhsakmt/LICENSE.md", sourceLicense = path.join(state.runtimeOutput, licenseRelative),
    sourceRecord = admitted.record.sources.files[licenseRelative], sourceArchive = admitted.record.sourceArchives["rocr"];
  if (!sourceRecord || !sourceArchive || !/^[a-f0-9]{64}$/.test(sourceArchive.sha256 || "") ||
      !fs.lstatSync(sourceLicense, {throwIfNoEntry: false})?.isFile() || fs.lstatSync(sourceLicense).isSymbolicLink() ||
      fileHash(sourceLicense) !== sourceRecord || !fs.lstatSync(sourceArchive.path, {throwIfNoEntry: false})?.isFile() ||
      fileHash(sourceArchive.path) !== sourceArchive.sha256) {
    throw new Error("ROCm hsakmt license body is not present in the admitted same-run ROCr source snapshot");
  }
  const recordPath = path.join(state.runtimeOutput, "result.json"), manifest = {format: "mom-linux-rocm-notice-sources-v1",
    completeNativeNotices: false, runtimeReceipt: {file: "runtime/result.json", sha256: fileHash(recordPath)},
    packages: copied, sourceLicense: {component: "hsakmt-staticdrm", file: "runtime/" + licenseRelative,
      sha256: sourceRecord, bytes: fs.statSync(sourceLicense).size, sourceArchive: {name: "rocr", sha256: sourceArchive.sha256},
      revision: admitted.record.pins["rocr"]}};
  const manifestFile = path.join(state.hostSdkRoot, "rocm-notice-sources.json");
  fs.writeFileSync(manifestFile, JSON.stringify(manifest, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return auditRocmHostNoticeSources(state);
}

/** Re-admit notice-source bodies from the transported same-run ROCm host SDK.
 * @param {Pick<ReturnType<typeof rocmHostStatePaths>, "hostSdkRoot" | "runtimeOutput">} state */
function auditRocmHostNoticeSources(state) {
  const recordFile = path.join(state.runtimeOutput, "result.json"), manifestFile = path.join(state.hostSdkRoot, "rocm-notice-sources.json"),
    recordStat = fs.lstatSync(recordFile, {throwIfNoEntry: false}), manifestStat = fs.lstatSync(manifestFile, {throwIfNoEntry: false});
  if (!recordStat?.isFile() || recordStat.isSymbolicLink() || fs.realpathSync(recordFile) !== recordFile || recordStat.size > (16 << 20) ||
      !manifestStat?.isFile() || manifestStat.isSymbolicLink() || fs.realpathSync(manifestFile) !== manifestFile || manifestStat.size > (1 << 20)) {
    throw new Error("ROCm notice-source receipt or manifest is not a bounded canonical file");
  }
  const record = /** @type {RocmNoticeReceipt} */ (readBuildJson(recordFile)),
    manifest = /** @type {RocmNoticeManifest} */ (readBuildJson(manifestFile)), expectedPackages = rocmHostNoticeRows(record),
    receiptHash = fileHash(recordFile);
  if (!hasExactKeys(manifest, ["format", "completeNativeNotices", "runtimeReceipt", "packages", "sourceLicense"]) ||
      manifest.format !== "mom-linux-rocm-notice-sources-v1" || manifest.completeNativeNotices !== false ||
      !isDeepStrictEqual(manifest.runtimeReceipt, {file: "runtime/result.json", sha256: receiptHash}) ||
      !Array.isArray(manifest.packages) || manifest.packages.length !== expectedPackages.length) {
    throw new Error("ROCm notice-source manifest is missing or not bound to the same-run producer receipt");
  }
  const packages = manifest.packages.map((/** @type {RocmNoticePackageFile} */ row, index) => {
    const expected = expectedPackages[index];
    if (!expected) {throw new Error("ROCm package copyright body is absent from its package-origin receipt");}
    const file = path.join(state.hostSdkRoot, "licenses", expected.package, "copyright"), stat = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file || stat.size < 1 || stat.size > (2 << 20) ||
        !hasExactKeys(row, ["package", "version", "architecture", "sourcePath", "archives", "file", "sha256", "bytes"]) ||
        row.package !== expected.package || row.version !== expected.version || row.architecture !== expected.architecture ||
        row.sourcePath !== expected.sourcePath || !isDeepStrictEqual(row.archives, expected.archives) ||
        row.file !== "licenses/" + expected.package + "/copyright" || row.bytes !== stat.size ||
        !/^[a-f0-9]{64}$/.test(row.sha256 || "") || fileHash(file) !== row.sha256) {
      throw new Error("ROCm package copyright body or static archive origin changed: " + (expected?.package || "unknown"));
    }
    return {package: row.package, version: row.version, architecture: row.architecture, sourcePath: row.sourcePath,
      file: row.file, sha256: row.sha256, bytes: row.bytes, archives: row.archives};
  });
  const relative = "source/rocr/libhsakmt/LICENSE.md", sourceFile = path.join(state.runtimeOutput, relative), sourceArchive = record.sourceArchives?.["rocr"],
    source = manifest.sourceLicense, sourceSha256 = record.sources?.files?.[relative];
  if (!hasExactKeys(source, ["component", "file", "sha256", "bytes", "sourceArchive", "revision"]) ||
      source.component !== "hsakmt-staticdrm" || source.file !== "runtime/" + relative || source.sha256 !== sourceSha256 ||
      source.revision !== record.pins?.rocr || !sourceArchive || !isDeepStrictEqual(source.sourceArchive,
    {name: "rocr", sha256: sourceArchive.sha256}) || !/^[a-f0-9]{64}$/.test(sourceSha256 || "") ||
      !fs.lstatSync(sourceFile, {throwIfNoEntry: false})?.isFile() || fs.lstatSync(sourceFile).isSymbolicLink() ||
      fs.statSync(sourceFile).size !== source.bytes || fileHash(sourceFile) !== sourceSha256 ||
      !fs.lstatSync(sourceArchive.path, {throwIfNoEntry: false})?.isFile() || fileHash(sourceArchive.path) !== sourceArchive.sha256) {
    throw new Error("ROCm hsakmt license body differs from its admitted same-run source and archive");
  }
  return {format: manifest.format, completeNativeNotices: false,
    runtimeReceipt: {file: manifest.runtimeReceipt.file, sha256: receiptHash},
    manifest: {file: "rocm-notice-sources.json", sha256: fileHash(manifestFile)}, packages,
    sourceLicense: {component: source.component, file: source.file, sha256: source.sha256, bytes: source.bytes,
      sourceArchive: source.sourceArchive, revision: source.revision}};
}

/** Restore the same-run LLVM/COMGR cohort, provision pinned host inputs, and run the public producer offline.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildRocmHostSdk(paths) {
  const state = rocmHostStatePaths(paths), git = canonicalExecutable("/usr/bin/git"), expectedRoots = [state.sourceName, state.llvmSourceName,
    state.llvmSdkName, state.hipHeadersName, state.resourcesName, state.sdkName];
  for (const output of [state.hostSdkRoot, state.hostSdkTar]) {
    if (fs.existsSync(output)) {throw new Error("Native ROCm host SDK stage refuses an existing output: " + path.basename(output));}
  }
  extractState(state.incomingComgrTar, temp, expectedRoots);
  verifyRocmSources(paths, state, git);
  const env = {...sdkSourceEnvironment(paths.env), GIT_ALLOW_PROTOCOL: "file", GIT_NO_LAZY_FETCH: "1",
    GIT_TERMINAL_PROMPT: "0", GIT_OPTIONAL_LOCKS: "0"};
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", state.llvmSource], paths.env, 2 * 60 * 1000);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-rocm-host-" + paths.suffix, 15 * 60 * 1000);
  verifyLinuxLlvmAdmissionCi(paths);
  fs.unlinkSync(state.incomingComgrTar);
  const parser = prepareRocmHostAdmission(paths, state), tools = rocmHostBuildTools(state), config = rocmHostBuildConfig(state, tools);
  auditRocmComgrSdk(state, {cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake});
  fs.mkdirSync(state.hostSdkRoot, {mode: 0o700});
  fs.writeFileSync(state.configFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  fs.writeFileSync(path.join(state.hostSdkRoot, "python-parser-install.json"), JSON.stringify(parser, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const buildEnv = {...env, MOM_BUILD_JOBS: ciBuildJobs(env), MOM_AOT_BUILD_ONLY: "1"};
  runNode("scripts/build-native-rocm-host-runtime.js", [state.configFile, state.runtimeOutput], buildEnv, 32 * 60 * 1000);
  const admitted = auditRocmHostSdk(state, config);
  captureRocmHostNoticeSources(state, admitted, paths.env);
  packState(state.hostSdkTar, temp, [...expectedRoots, state.hostSdkName]);
  listState(state.hostSdkTar, [...expectedRoots, state.hostSdkName]);
  console.log("[native-ci] ROCm host SDK admitted: " + admitted.sdk.libraries.length + " libraries, CPU-only build-only; no runtime or release qualification.");
}

/** Explicit signed Intel tools for the unchanged public Hermes SDK producer. */
function hermesBuildTools() {
  return {git: canonicalExecutable("/usr/bin/git"), cmake: canonicalExecutable("/usr/bin/cmake"),
    cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx"), ninja: canonicalExecutable("/usr/bin/ninja")};
}

/** Return the exact public producer command for the fetched bare Git root, not its cohort parent.
 * @param {ReturnType<typeof hermesStatePaths>} state @param {ReturnType<typeof hermesBuildTools>} tools */
function hermesBuildCommand(state, tools) {
  const source = path.join(state.source, "roots/hermes");
  hermesSdkPlan(source, state.sdkRoot, tools, 2, "linux");
  return {script: "scripts/build-native-hermes.js", args: [source, state.sdkRoot, tools.git, tools.cmake, tools.cc, tools.cxx, tools.ninja, "2"]};
}

/** Accept only the producer's canonical absolute patch location; tool paths and every hash stay exact.
 * @param {unknown} fingerprints @param {ReturnType<typeof hermesBuildTools>} tools @param {string[]} patches */
function assertHermesSdkFingerprints(fingerprints, tools, patches) {
  const toolNames = ["cc", "cmake", "cxx", "git", "ninja"], knownPatches = ["runtime/hermes-host-rejections.patch"];
  const canonicalAbsolute = (/** @type {unknown} */ value) => typeof value === "string" && value.length <= 4096 &&
    path.isAbsolute(value) && path.resolve(value) === value && !Array.from(value).some(char => char.charCodeAt(0) < 32 ||
      char.charCodeAt(0) >= 127 && char.charCodeAt(0) <= 159);
  if (!fingerprints || typeof fingerprints !== "object" || Array.isArray(fingerprints) ||
      Object.keys(tools).sort().join("\0") !== toolNames.join("\0") ||
      !isDeepStrictEqual(patches, knownPatches) || !Object.values(tools).every(canonicalAbsolute)) {
    throw new Error("Hermes SDK fingerprint plan is not the known bounded tool/patch set");
  }
  const recorded = /** @type {Record<string, unknown>} */ (fingerprints), recordedPaths = Object.keys(recorded),
    toolPaths = Object.values(tools), aliases = /** @type {Record<string, string>} */ ({}), allowed = new Set(toolPaths);
  if (new Set(toolPaths).size !== toolPaths.length || recordedPaths.some(file => !canonicalAbsolute(file))) {
    throw new Error("Hermes SDK fingerprints contain duplicate or noncanonical paths");
  }
  for (const relative of patches) {
    const suffix = "/" + relative, matches = recordedPaths.filter(file => file.endsWith(suffix));
    if (matches.length !== 1) {throw new Error("Hermes SDK must record exactly one canonical path for each source patch");}
    const sourcePath = matches[0] || "";
    if (allowed.has(sourcePath)) {throw new Error("Hermes SDK patch identity collides with a tool fingerprint");}
    allowed.add(sourcePath);
    aliases[relative] = sourcePath;
  }
  if (recordedPaths.length !== allowed.size || recordedPaths.some(file => !allowed.has(file))) {
    throw new Error("Hermes SDK fingerprints contain unknown or missing tool/patch entries");
  }
  for (const file of toolPaths) {
    if (recorded[file] !== fileHash(file)) {throw new Error("Hermes SDK tool fingerprint changed: " + file);}
  }
  for (const [relative, sourcePath] of Object.entries(aliases)) {
    if (recorded[sourcePath] !== fileHash(path.join(repo, relative))) {
      throw new Error("Hermes SDK patch fingerprint changed: " + relative);
    }
  }
}

/** Use the shared strict SDK admission; add only run-specific tool and command-plan identity.
 * @param {ReturnType<typeof hermesStatePaths>} state @param {ReturnType<typeof hermesBuildTools>} tools */
function auditHermesSdk(state, tools) {
  const command = hermesBuildCommand(state, tools), receiptPath = path.join(state.sdkRoot, "result.json"),
    receipt = readBuildJson(receiptPath), before = readBuildJson(path.join(state.sdkRoot, "before.json")),
    plan = hermesSdkPlan(command.args[0] || "", state.sdkRoot, tools, 2, "linux");
  nativeHermesInputs(receiptPath);
  assertHermesSdkFingerprints(receipt.fingerprints, tools, plan.patches);
  if (!isDeepStrictEqual(before.configure, plan.configure) || !isDeepStrictEqual(before.compile, plan.compile)) {
    throw new Error("Native Hermes SDK does not record this run's explicit Intel tools and producer command plan");
  }
  return receipt;
}

/** Use the repository's signed oneAPI package route; its oneapi component also installs the existing
 * base set containing CMake, Ninja, Python, binutils and xz. It does not install umbrella products.
 * @param {ReturnType<typeof ciPaths>} paths */
function provisionOneApi(paths, workspace = "/var/tmp/mom-dev-native-llvm21-" + path.basename(paths.source).replace(/^mom-native-prerequisites-/, ""),
  budgetMs = 90 * 60 * 1000) {
  const installer = path.join(repo, "scripts/install-dev.sh");
  if (!Number.isInteger(budgetMs) || budgetMs < 1 || budgetMs > 90 * 60 * 1000) {throw new Error("Invalid bounded oneAPI provisioning budget");}
  const deadline = Date.now() + budgetMs;
  const validateTimeout = Math.min(2 * 60 * 1000, deadline - Date.now());
  if (validateTimeout <= 0) {throw new Error("Intel 2026.2 provisioning budget expired");}
  const existing = spawnSync(installer, ["--component", "oneapi", "--jobs", "4", "--validate-only"],
    {cwd: repo, env: paths.env, stdio: "inherit", timeout: validateTimeout});
  const installed = !existing.error && !existing.signal && existing.status === 0;
  /** @param {string[]} [extra] */
  const sudo = (extra = []) => {
    const timeout = Math.min(45 * 60 * 1000, deadline - Date.now());
    if (timeout <= 0) {throw new Error("Intel 2026.2 provisioning budget expired");}
    const result = spawnSync("sudo", ["env", "MOM_BUILD_JOBS=4", "MOM_DEV_WORKSPACE=" + workspace, installer,
      "--component", "oneapi", "--jobs", "4", ...extra], {cwd: repo, env: paths.env, stdio: "inherit", timeout});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error("Intel 2026.2 signed APT provisioning or validation failed", {cause: result.error});
    }
  };
  if (!installed) {sudo(); sudo(["--validate-only"]);}
  for (const file of ["/usr/bin/cmake", "/usr/bin/ninja", "/usr/bin/python3", "/usr/bin/ld"]) {
    try {
      if (!fs.statSync(file).isFile()) {throw new Error();}
      fs.accessSync(file, fs.constants.X_OK);
    } catch {
      throw new Error("Missing Linux LLVM build prerequisite: " + file);
    }
  }
  linuxNativeSdkLldPath("/opt/intel/oneapi/compiler/2026.2/bin/icpx");
}

/** Provision only the existing pinned DPC++ and minimal CUDA developer components.
 * @param {ReturnType<typeof ciPaths>} paths @param {string} workspace @param {number} [budgetMs] */
function provisionDpcppCudaInputs(paths, workspace, budgetMs = 40 * 60 * 1000) {
  const installer = path.join(repo, "scripts/install-dev.sh");
  if (!Number.isInteger(budgetMs) || budgetMs < 1 || budgetMs > 45 * 60 * 1000 ||
      ["/opt/dpcpp", "/usr/local/cuda-12.6", "/usr/local/cuda"].some(file => fs.lstatSync(file, {throwIfNoEntry: false}))) {
    throw new Error("DPC++/CUDA provisioning needs fresh install roots and a bounded budget");
  }
  const deadline = Date.now() + budgetMs;
  /** @param {string[]} extra */
  const sudo = extra => {
    const timeout = Math.min(40 * 60 * 1000, deadline - Date.now());
    if (timeout <= 0) {throw new Error("DPC++/CUDA input provisioning budget expired");}
    const result = spawnSync("sudo", ["env", "MOM_BUILD_JOBS=4", "MOM_DEV_WORKSPACE=" + workspace, installer,
      "--component", "dpcpp,cuda", "--jobs", "4", ...extra], {cwd: repo, env: paths.env, stdio: "inherit", timeout});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error("Pinned DPC++/CUDA provisioning or validation failed", {cause: result.error});
    }
  };
  sudo([]); sudo(["--validate-only"]);
}

/** Reject an unpinned or redirected source tar before any extraction or provisioning.
 * @param {string} file @param {string} expectedSha256 */
function dpcppSourceArchivePin(file, expectedSha256) {
  const stat = fs.lstatSync(file);
  if (!/^[a-f0-9]{64}$/.test(expectedSha256) || !path.isAbsolute(file) || fs.realpathSync(file) !== file ||
      !stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size < 1 || stat.size > 16 * 1024 ** 3 ||
      fileHash(file) !== expectedSha256) {
    throw new Error("DPC++/CUDA input stage requires the externally pinned same-run source tar");
  }
  return expectedSha256;
}

/** Use historical acquisition metadata only when both restored DPC++/CUDA inputs are externally sealed.
 * @param {NodeJS.ProcessEnv} environment @returns {{bundleSha256:string,manifestSha256:string}|undefined} */
function dpcppCudaInputPins(environment) {
  const bundleSha256 = environment["MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256"] || "",
    manifestSha256 = environment["MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256"] || "";
  if (!bundleSha256 && !manifestSha256) {return undefined;}
  if (!/^[a-f0-9]{64}$/.test(bundleSha256) || !/^[a-f0-9]{64}$/.test(manifestSha256)) {
    throw new Error("DPC++/CUDA bundle and manifest must both have external SHA-256 pins");
  }
  return {bundleSha256, manifestSha256};
}

/** Restore and verify Linux sources, provision the pinned compiler and minimal CUDA inputs, then retain a raw same-run tar.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildDpcppCudaInputs(paths) {
  const state = dpcppCudaInputsStatePaths(paths, paths.env), git = canonicalExecutable("/usr/bin/git");
  if (fs.lstatSync(state.root, {throwIfNoEntry: false}) || fs.lstatSync(state.sdkTar, {throwIfNoEntry: false})) {
    throw new Error("DPC++/CUDA input stage refuses an existing output");
  }
  const sourceTarSha256 = dpcppSourceArchivePin(state.incomingSourceTar, paths.env["MOM_NATIVE_LINUX_SOURCE_BUNDLE_SHA256"] || "");
  extractState(state.incomingSourceTar, temp, [state.linux.sourceName]);
  verifyLinuxSources(paths, state.linux, git);
  fs.unlinkSync(state.incomingSourceTar);
  const workspace = "/var/tmp/mom-dev-native-dpcpp-cuda-" + paths.suffix;
  if (fs.lstatSync(workspace, {throwIfNoEntry: false})) {throw new Error("DPC++/CUDA installer workspace already exists");}
  provisionDpcppCudaInputs(paths, workspace);
  stageDpcppCudaInputs(state, {dpcpp: "/opt/dpcpp", cuda: "/usr/local/cuda-12.6"},
    {kind: "source-tar", sha256: sourceTarSha256}, paths.env);
  packState(state.sdkTar, temp, [state.name]); listState(state.sdkTar, [state.name]);
  const outputs = {bundle_sha256: fileHash(state.sdkTar), manifest_sha256: fileHash(state.manifest)};
  appendWorkflowOutputs(paths.env, outputs);
  return outputs;
}

/** Use acquisition's shared offline verifier after every transport; the public producer also
 * checks original archive digests before compilation.
 * @param {ReturnType<typeof ciPaths>} paths */
function verifyIoSource(paths) {
  return verifyIoArchives(paths.ioSource);
}

/** Resolve tool paths to canonical executable files because the OpenSSL producer rejects aliases.
 * @param {string} file */
function canonicalExecutable(file) {
  const selected = fs.realpathSync(file), stat = fs.statSync(selected);
  if (!stat.isFile()) {throw new Error("Missing native CI executable: " + file);}
  fs.accessSync(selected, fs.constants.X_OK);
  return selected;
}

/** Recheck the LLVM producer's audited install inventory before using its tools for NVRTC proof.
 * @param {ReturnType<typeof ciPaths>} paths */
function staticNvrtcLlvmConfig(paths) {
  const sdk = path.join(paths.output, "sdk"), llvmConfig = canonicalExecutable(path.join(sdk, "bin/llvm-config")),
    env = {...paths.env, GIT_OPTIONAL_LOCKS: "0"};
  /** @param {string[]} args */
  const query = args => {
    const result = spawnSync(llvmConfig, args, {encoding: "utf8", env, timeout: 30000, maxBuffer: 1 << 20});
    if (result.error || result.signal || result.status !== 0 || result.stderr || typeof result.stdout !== "string") {
      throw new Error("Same-run LLVM 21 tool query failed: llvm-config " + args.join(" "), {cause: result.error});
    }
    return result.stdout.trim();
  };
  const bin = path.join(sdk, "bin"), selectedBin = fs.realpathSync(query(["--bindir"]));
  if (selectedBin !== fs.realpathSync(bin) || query(["--version"]) !== "21.1.8") {
    throw new Error("Same-run LLVM SDK does not identify its own LLVM 21.1.8 tool directory");
  }
  for (const name of ["llvm-config", "clang++", "ld.lld"]) {
    const selected = canonicalExecutable(path.join(selectedBin, name)), relative = path.relative(fs.realpathSync(sdk), selected);
    if (path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep)) {
      throw new Error("NVRTC proof tool escapes the same-run LLVM SDK: " + name);
    }
  }
  return llvmConfig;
}

/** Copy only regular proof receipts, logs and generated source outputs; never SDK binaries.
 * @param {string} source @param {string} destination */
function copyNvrtcEvidenceFile(source, destination) {
  const stat = fs.lstatSync(source);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 32 * 1024 * 1024) {
    throw new Error("NVRTC verification evidence must be a bounded regular file: " + path.basename(source));
  }
  fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
}

/** @param {ReturnType<typeof nvrtcStatePaths>} state @param {string} proofDirectory
 * @param {{receipt: {platform: string, source?: {receiptSha256?: string}}, receiptFile: string, receiptSha256: string}} admitted */
function packNvrtcVerification(state, proofDirectory, admitted) {
  if (fs.existsSync(state.evidence)) {throw new Error("Native NVRTC verification evidence output already exists");}
  fs.mkdirSync(state.evidence, {mode: 0o700});
  copyNvrtcEvidenceFile(path.join(state.source, "fetch.json"), path.join(state.evidence, "fetch.json"));
  copyNvrtcEvidenceFile(admitted.receiptFile, path.join(state.evidence, "sdk.json"));
  if (state.platform === "linux") {
    const proofReceipt = path.join(proofDirectory, "result.json"), proof = readBuildJson(proofReceipt),
      expectedSource = {file: "sycl/cn-gpu/cuda-expand-kernel.inc", sha256: fileHash(path.join(repo, "sycl/cn-gpu/cuda-expand-kernel.inc"))};
    if (proof.sdk !== state.sdkRoot || proof.vendor?.receiptSha256 !== admitted.receiptSha256 ||
        !Array.isArray(proof.files) || proof.files.length !== 4 || !isDeepStrictEqual(proof.source, expectedSource)) {
      throw new Error("NVRTC CPU proof does not bind the admitted run-scoped SDK and maintained source");
    }
    const expectedOutputs = ["compute_120.ptx", "compute_80.ptx", "sm_120.cubin", "sm_80.cubin"].sort();
    if (!isDeepStrictEqual(proof.files.map((/** @type {{name?: string} | null} */ row) => row?.name).sort(), expectedOutputs)) {
      throw new Error("NVRTC CPU proof output matrix is incomplete or contains unexpected files");
    }
    const proofOutput = path.join(state.evidence, "proof"); fs.mkdirSync(proofOutput, {mode: 0o700});
    copyNvrtcEvidenceFile(proofReceipt, path.join(proofOutput, "result.json"));
    for (const directory of ["actual", "expected", "lld"]) {
      const sourceDirectory = path.join(proofDirectory, directory), destination = path.join(proofOutput, directory);
      const stat = fs.lstatSync(sourceDirectory);
      if (!stat.isDirectory() || stat.isSymbolicLink() ||
          !isDeepStrictEqual(fs.readdirSync(sourceDirectory).sort(), expectedOutputs)) {
        throw new Error("NVRTC proof output is not the exact generated-code matrix: " + directory);
      }
      fs.mkdirSync(destination, {mode: 0o700});
      for (const row of proof.files) {
        if (typeof row.name !== "string" || !/^[A-Za-z0-9_.+-]{1,128}$/.test(row.name) ||
            !Number.isSafeInteger(row.bytes) || row.bytes < 1 || !/^[0-9a-f]{64}$/.test(row.sha256 || "")) {
          throw new Error("Invalid NVRTC generated proof output record");
        }
        const file = path.join(sourceDirectory, row.name), outputStat = fs.lstatSync(file);
        if (!outputStat.isFile() || outputStat.isSymbolicLink() || outputStat.size !== row.bytes || fileHash(file) !== row.sha256) {
          throw new Error("NVRTC generated proof output changed before evidence packaging: " + row.name);
        }
        copyNvrtcEvidenceFile(file, path.join(destination, row.name));
      }
    }
    const logs = path.join(proofOutput, "logs"); fs.mkdirSync(logs, {mode: 0o700});
    for (const name of ["build.log", "oracle-build.log", "compile.log", "oracle.log", "llvm-tools.log", "lld-build.log",
      "lld.log", "arguments.log", "dependencies.log", "syscalls.log", "isolated-paths.log"]) {
      copyNvrtcEvidenceFile(path.join(proofDirectory, name), path.join(logs, name));
    }
  }
  const packed = packState(state.verificationTar, temp, [state.evidenceName]);
  if (packed.bytes > 128 * 1024 * 1024) {throw new Error("NVRTC verification evidence exceeded its bounded upload size");}
  return packed;
}

/** Fetch, assemble, and admit one target SDK in the same job; Linux also runs the strict CPU oracle.
 * Vendor archives, assembled libraries, and linked proof executables never enter the uploaded tar.
 * @param {ReturnType<typeof ciPaths>} paths @param {"linux" | "windows"} platform */
function verifyNvrtcTarget(paths, platform) {
  const state = nvrtcStatePaths(paths, platform, paths.env);
  if ([state.source, state.sdkRoot, state.evidence, ...(platform === "linux" ? [state.proofParent] : [])].some(item => fs.existsSync(item))) {
    throw new Error("Native NVRTC verification requires new run-scoped outputs");
  }
  runNode("scripts/fetch-native-sdk-sources.js", ["nvrtc-" + platform, state.source, "--allow-network"], paths.env, 15 * 60 * 1000);
  const fetched = verifyNvrtcArchives(platform, state.source), fetchReceipt = readBuildJson(path.join(state.source, "fetch.json"));
  if (fetched.target !== "nvrtc-" + platform || fetched.status !== "PASS" || fetched.compiled !== false ||
      fetched.publicationQualified !== false || fetchReceipt.target !== "nvrtc-" + platform) {
    throw new Error("Native NVRTC acquisition is not the pinned build-only target cohort");
  }
  runNode("scripts/prepare-native-nvrtc-sdk.js", [platform, state.source, state.sdkRoot], paths.env, 15 * 60 * 1000);
  const admitted = nvrtcSdkInputs(platform, state.sdkRoot), sdk = admitted.receipt;
  if (sdk.platform !== platform || sdk.vendorPrebuilt !== true || sdk.sourceRebuilt !== false || sdk.compiled !== false ||
      sdk.publicationQualified !== false || sdk.redistributionQualified !== false || sdk.source.receiptSha256 !== fileHash(path.join(state.source, "fetch.json"))) {
    throw new Error("Native NVRTC SDK admission must remain vendor-prebuilt and non-publication-qualified");
  }
  let proofDirectory = "";
  if (platform === "linux") {
    provisionOneApi(paths, "/var/tmp/mom-dev-native-nvrtc-" + paths.suffix);
    const tools = hermesBuildTools();
    verifyLinuxLlvmAdmissionCi(paths);
    const llvmConfig = staticNvrtcLlvmConfig(paths);
    fs.mkdirSync(state.proofParent, {mode: 0o700});
    runNode("scripts/probe-static-nvrtc.js", [state.sdkRoot, state.proofParent],
      {...paths.env, CXX: tools.cxx, LLVM_CONFIG: llvmConfig}, 15 * 60 * 1000);
    const children = fs.readdirSync(state.proofParent);
    if (children.length !== 1) {throw new Error("Static NVRTC proof did not produce exactly one evidence directory");}
    proofDirectory = path.join(state.proofParent, children[0] || "");
    const proofStat = fs.lstatSync(proofDirectory);
    if (!proofStat.isDirectory() || proofStat.isSymbolicLink()) {throw new Error("Static NVRTC proof output is not a real directory");}
  }
  const packed = packNvrtcVerification(state, proofDirectory, admitted);
  console.log(`[native-ci] Verified NVRTC ${platform}; vendor SDK stays on-runner and no release qualification is implied (${packed.bytes} evidence bytes).`);
  return packed;
}

/** @typedef {{ar: string, cc: string, make: string, perl: string, ranlib: string, tar: string}} LinuxOpenSslTools */
/** @typedef {{ar: string, cc: string, make: string, mkdir: string, tar: string}} LinuxZstdTools */

/** Use only the existing signed Intel compiler route and stock build tools. @returns {LinuxOpenSslTools} */
function linuxOpenSslTools() {
  return {ar: canonicalExecutable("/usr/bin/ar"), cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    make: canonicalExecutable("/usr/bin/make"), perl: canonicalExecutable("/usr/bin/perl"),
    ranlib: canonicalExecutable("/usr/bin/ranlib"), tar: canonicalExecutable("/usr/bin/tar")};
}

/** Use only the signed Intel compiler route and explicit stock tools required by the zstd producer. @returns {LinuxZstdTools} */
function linuxZstdTools() {
  return {ar: canonicalExecutable("/usr/bin/ar"), cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    make: canonicalExecutable("/usr/bin/make"), mkdir: canonicalExecutable("/usr/bin/mkdir"), tar: canonicalExecutable("/usr/bin/tar")};
}

/** Preserve the producer's SDK-root contract; callers must not substitute lib/libzstd.a.
 * @param {Pick<ReturnType<typeof zstdStatePaths>, "archive" | "sdkRoot">} state @param {string} toolsFile */
function zstdBuildCommand(state, toolsFile) {
  return {script: "scripts/build-native-zstd.js", args: [state.archive, state.sdkRoot, toolsFile]};
}

/** Reuse producer admission, then join its original-input path to this run's verified acquisition.
 * @param {Pick<ReturnType<typeof zstdStatePaths>, "source" | "archive" | "sdkRoot">} state @param {LinuxZstdTools} tools @param {string} toolsFile
 * @param {{verifyArchive?: typeof verifyZstdArchive, sdkInputs?: typeof nativeZstdInputs}} [readers] */
function auditZstdSdk(state, tools, toolsFile, readers = {}) {
  const fetched = (readers.verifyArchive || verifyZstdArchive)(state.source), archive = fetched.archives?.["zstd"],
    sdk = (readers.sdkInputs || nativeZstdInputs)(state.sdkRoot), record = readBuildJson(sdk.receipt),
    expectedTools = Object.fromEntries(Object.entries(tools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}]));
  if (!archive || archive.path !== state.archive || archive.sha256 !== zstdPin.sha256 ||
      record.archive?.path !== archive.path || record.archive.sha256 !== archive.sha256 ||
      sdk.archive !== path.join(state.sdkRoot, "lib/libzstd.a") || record.config?.sha256 !== fileHash(toolsFile) ||
      record.plan?.jobs !== 4 || !isDeepStrictEqual(record.tools, expectedTools) ||
      sdk.files?.[state.archive] !== archive.sha256 || sdk.files?.[sdk.receipt] !== sdk.receiptSha256 ||
      !/^[0-9a-f]{64}$/.test(sdk.files?.[sdk.archive] || "")) {
    throw new Error("Native zstd SDK does not bind this run's pinned archive, descriptor, tools and SDK root");
  }
  return sdk;
}

/** @param {{suffix: string}} paths @param {string} stage */
function stateTarName(paths, stage) {return "native-llvm21-state-" + paths.suffix + "-" + stage + ".tar";}

/** The reusable Linux LLVM jobs can consume only these explicit same-run predecessor states. */
const linuxLlvmCiStageDefinitions = /** @type {Readonly<Record<string, {previousStage: string, artifactStage: string, phases?: readonly string[]}>>} */ (Object.freeze({
  "llvm-common": {previousStage: "configured", artifactStage: "llvm-core-optimization",
    phases: ["llvm-foundation", "llvm-core-analysis", "llvm-core-optimization"]},
  "llvm-core-codegen": {previousStage: "llvm-core-optimization", artifactStage: "llvm-core-codegen"},
  "llvm-backends-clang": {previousStage: "llvm-core-codegen", artifactStage: "clang-sema",
    phases: ["llvm-x86", "llvm-nvptx", "llvm-amdgpu", "clang-base", "clang-sema"]},
  "clang-frontend": {previousStage: "clang-sema", artifactStage: "clang-frontend"},
}));

/** Derive exact incoming/output names for one allowlisted reusable LLVM phase.
 * @param {string} stage @param {NodeJS.ProcessEnv} [environment] */
function linuxLlvmCiStagePlan(stage, environment = process.env) {
  const definition = Object.hasOwn(linuxLlvmCiStageDefinitions, stage) ? linuxLlvmCiStageDefinitions[stage] : undefined,
    run = environment["GITHUB_RUN_ID"] || "", attempt = environment["GITHUB_RUN_ATTEMPT"] || "";
  if (!definition || !/^[0-9]{1,20}$/.test(run) || !/^[1-9][0-9]{0,5}$/.test(attempt)) {
    throw new Error("Linux LLVM CI needs one maintained stage and exact run-attempt identity");
  }
  const suffix = run + "-" + attempt;
  return {previousStage: definition.previousStage, artifactStage: definition.artifactStage,
    inputArtifactName: stateTarName({suffix}, definition.previousStage),
    outputArtifactName: stateTarName({suffix}, definition.artifactStage)};
}

/** Configure emits the configured-state handoff consumed by the first LLVM build.
 * @param {string} stage @returns {string} */
function outputStageName(stage) {return stage === "configure" ? "configured" : stage;}
/** @param {ReturnType<typeof ciPaths>} paths @param {string} stage */
function incomingTar(paths, stage) {return path.join(paths.runnerTemp, "incoming", stateTarName(paths, stage));}
/** @param {ReturnType<typeof ciPaths>} paths @param {string} stage */
function outputTar(paths, stage) {return path.join(paths.runnerTemp, stateTarName(paths, stage));}
/** @param {ReturnType<typeof ciPaths>} paths */
function ioFinalTar(paths) {return path.join(paths.runnerTemp, "native-linux-io-openssl-sdk-" + paths.suffix + ".tar");}

/** Restore an exact prior state, verify its independent archive receipt, then remove only the
 * consumed same-run transfer file to leave room for the next full state tar.
 * @param {ReturnType<typeof ciPaths>} paths @param {string} stage @param {boolean} [includeSdk] @param {string} [deadlineEpoch] */
function restorePrior(paths, stage, includeSdk = true, deadlineEpoch) {
  const file = incomingTar(paths, stage), roots = includeSdk ? [paths.sourceName, paths.sdkName] :
    stage === "sources" ? [paths.sourceName, paths.ioSourceName] : [paths.sourceName];
  extractState(file, temp, roots, roots, deadlineEpoch ? ioOpenSslChildTimeout(deadlineEpoch, 30 * 60 * 1000, Date.now(), 5 * 60 * 1000) : undefined);
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", paths.source], paths.env,
    deadlineEpoch ? ioOpenSslChildTimeout(deadlineEpoch, 2 * 60 * 1000, Date.now(), 5 * 60 * 1000) : 2 * 60 * 1000);
  if (stage === "sources") {verifyIoSource(paths);}
  fs.unlinkSync(file);
}

/** @param {ReturnType<typeof ciPaths>} paths @param {string} stage */
function packCurrent(paths, stage) {packState(outputTar(paths, stage), temp, [paths.sourceName, paths.sdkName]);}

/** Restore/provision once for a short, measured group of existing public build phases.
 * Internal receipts retain their original order; only the last state is transported.
 * @param {ReturnType<typeof ciPaths>} paths @param {string} stage
 * @param {{restore?: (paths: ReturnType<typeof ciPaths>, stage: string) => void,
 *   provision?: (paths: ReturnType<typeof ciPaths>) => void, execute?: typeof runNode,
 *   pack?: (paths: ReturnType<typeof ciPaths>, stage: string) => void}} [operations] */
function buildLinuxLlvmGroupCi(paths, stage, operations = {}) {
  const group = Object.hasOwn(linuxLlvmCiStageDefinitions, stage) ? linuxLlvmCiStageDefinitions[stage] : undefined;
  if (!group?.phases?.length) {throw new Error("Linux LLVM CI needs a maintained phase group");}
  const restore = operations.restore || restorePrior, provision = operations.provision || provisionOneApi,
    execute = operations.execute || runNode, pack = operations.pack || packCurrent,
    builder = "scripts/build-native-llvm21-sdk.js";
  restore(paths, group.previousStage);
  provision(paths);
  for (const phase of group.phases) {execute(builder, ["build", paths.output, phase], paths.env);}
  pack(paths, group.artifactStage);
}

/** Verify both SDK receipts and exact original inputs before any same-run SDK state is retained.
 * @param {ReturnType<typeof ciPaths>} paths @param {number} [verifyTimeoutMs] */
function auditIoOpenSsl(paths, verifyTimeoutMs = 2 * 60 * 1000) {
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", paths.source], paths.env, verifyTimeoutMs);
  verifyIoSource(paths);
  const io = nativeIoInputs(paths.ioSdk, "linux"), openssl = nativeOpenSslInputs(paths.opensslSdk, "linux"),
    ioReceipt = readBuildJson(io.receipt), opensslReceipt = readBuildJson(path.join(paths.opensslSdk, "result.json")),
    toolsDescriptor = readBuildJson(paths.opensslTools), expectedTools = linuxOpenSslTools();
  const cmake = canonicalExecutable("/usr/bin/cmake"), ninja = canonicalExecutable("/usr/bin/ninja"),
    expectedIoFingerprints = Object.fromEntries([cmake, expectedTools.cc, ninja].map(file => [file, fileHash(file)]));
  const expectedIoInputs = {
    [path.join(paths.ioSource, "libuv-v1.51.0.tar.gz")]: ioPins.libuv.sha256,
    [path.join(paths.ioSource, "zlib-1.3.1.tar.gz")]: ioPins.zlib.sha256,
  };
  const opensslArchive = path.join(paths.source, "openssl-" + opensslPin.version + ".tar.gz");
  if (ioReceipt.target !== "linux" || ioReceipt.buildOnly !== true || ioReceipt.publicationQualified !== false ||
      !isDeepStrictEqual(ioReceipt.inputs, expectedIoInputs) || !isDeepStrictEqual(ioReceipt.fingerprints, expectedIoFingerprints) ||
      !opensslReceipt.archive || opensslReceipt.archive.path !== opensslArchive || opensslReceipt.archive.sha256 !== opensslPin.sha256 ||
      !toolsDescriptor || Object.keys(toolsDescriptor).sort().join(",") !== "jobs,tools,version" ||
      toolsDescriptor.version !== 1 || toolsDescriptor.jobs !== 2 || !isDeepStrictEqual(toolsDescriptor.tools, expectedTools) ||
      opensslReceipt.config?.sha256 !== fileHash(paths.opensslTools) ||
      !opensslReceipt.tools || Object.keys(opensslReceipt.tools).sort().join(",") !== Object.keys(expectedTools).sort().join(",")) {
    throw new Error("Native I/O/OpenSSL SDK receipts do not bind this run's pinned sources and tools");
  }
  for (const [name, file] of Object.entries(expectedTools)) {
    const record = opensslReceipt.tools[name];
    if (!record || record.path !== file || record.sha256 !== fileHash(file)) {throw new Error("OpenSSL SDK tool identity changed: " + name);}
  }
  if (fileHash(opensslArchive) !== opensslPin.sha256 ||
      Object.entries(expectedIoInputs).some(([file, hash]) => fileHash(file) !== hash) ||
      io.libraries.length !== 2 || openssl.libraries.length !== 2 || io.libraries.some(file => !fs.statSync(file).isFile()) ||
      openssl.libraries.some(file => !fs.statSync(file).isFile())) {
    throw new Error("Native I/O/OpenSSL SDK source or selected archive changed during admission");
  }
  return {io, openssl};
}

/** Keep the two public producer invocations bound to this run's canonical source and output paths.
 * @param {ReturnType<typeof ciPaths>} paths @param {string} cmake @param {string} ninja @param {LinuxOpenSslTools} tools */
function ioOpenSslBuildCommands(paths, cmake, ninja, tools) {
  return [
    {script: "scripts/build-native-io.js", args: [path.join(paths.ioSource, "libuv-v1.51.0.tar.gz"),
      path.join(paths.ioSource, "zlib-1.3.1.tar.gz"), paths.ioSdk, cmake, tools.cc, ninja, "2"]},
    {script: "scripts/build-native-openssl.js", args: [path.join(paths.source, "openssl-" + opensslPin.version + ".tar.gz"),
      paths.opensslSdk, paths.opensslTools]},
  ];
}

/** Convert the explicit job-start epoch to a bounded child timeout, optionally reserving
 * time for the remaining audit/package stages before the hosted job limit.
 * @param {string} deadlineEpoch @param {number} maximumMs @param {number} [nowMs] @param {number} [reserveMs] */
function ioOpenSslChildTimeout(deadlineEpoch, maximumMs, nowMs = Date.now(), reserveMs = 0) {
  if (!/^\d{10}$/.test(deadlineEpoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 ||
      !Number.isSafeInteger(reserveMs) || reserveMs < 0) {
    throw new Error("Native I/O/OpenSSL CI needs a valid bounded job-start deadline");
  }
  const deadlineMs = Number(deadlineEpoch) * 1000, remainingMs = deadlineMs - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs > 60 * 60 * 1000) {
    throw new Error("Native I/O/OpenSSL deadline exceeds the 60-minute job bound");
  }
  return linuxNativeComponentChildTimeout(deadlineMs - reserveMs, maximumMs, nowMs);
}

/** Keep the public Linux LLVM configure command on the validated run envelope.
 * @param {Pick<ReturnType<typeof ciPaths>, "source" | "output" | "env">} paths */
function linuxLlvmConfigureCommand(paths) {
  return {script: "scripts/build-native-llvm21-sdk.js", args: ["configure", path.join(paths.source, "llvm-project-21.1.8.src.tar.xz"),
    paths.output, "/usr/bin/cmake", "/usr/bin/ninja", "/opt/intel/oneapi/compiler/2026.2/bin/icx",
    "/opt/intel/oneapi/compiler/2026.2/bin/icpx", paths.env["MOM_BUILD_JOBS"] || ""]};
}

/** Restore and verify the same-run IO/OpenSSL sources, then build both original static SDKs.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildIoOpenSsl(paths) {
  const deadlineEpoch = paths.env["MOM_NATIVE_IO_OPENSSL_DEADLINE_EPOCH"] || "";
  ioOpenSslChildTimeout(deadlineEpoch, 60 * 60 * 1000, Date.now(), 5 * 60 * 1000);
  restorePrior(paths, "sources", false, deadlineEpoch);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-io-openssl-" + paths.suffix,
    ioOpenSslChildTimeout(deadlineEpoch, 15 * 60 * 1000, Date.now(), 5 * 60 * 1000));
  const cmake = canonicalExecutable("/usr/bin/cmake"), ninja = canonicalExecutable("/usr/bin/ninja"), tools = linuxOpenSslTools();
  for (const file of ["/usr/bin/python3", "/usr/bin/ld"]) {canonicalExecutable(file);}
  if (fs.existsSync(paths.ioSdkRoot)) {throw new Error("Native I/O/OpenSSL SDK output already exists");}
  fs.mkdirSync(paths.ioSdkRoot, {mode: 0o700});
  const config = {version: 1, jobs: 2, tools};
  fs.writeFileSync(paths.opensslTools, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  for (const command of ioOpenSslBuildCommands(paths, cmake, ninja, tools)) {
    runNodeProcessGroup(command.script, command.args, paths.env,
      ioOpenSslChildTimeout(deadlineEpoch, 45 * 60 * 1000, Date.now(), 5 * 60 * 1000));
  }
  auditIoOpenSsl(paths, ioOpenSslChildTimeout(deadlineEpoch, 2 * 60 * 1000));
  packState(ioFinalTar(paths), temp, [paths.sourceName, paths.ioSourceName, paths.ioSdkName],
    ioOpenSslChildTimeout(deadlineEpoch, 5 * 60 * 1000));
}

/** Fetch, verify, and retain only the pinned bare Hermes source root for its consumer job.
 * @param {ReturnType<typeof ciPaths>} paths */
function fetchHermesSource(paths) {
  const state = hermesStatePaths(paths), git = canonicalExecutable("/usr/bin/git");
  runNode("scripts/fetch-native-sdk-sources.js", ["hermes", git, state.source, "--allow-network"], paths.env, 15 * 60 * 1000);
  verifyHermesSource(paths, state, git);
  packState(state.sourceTar, temp, [state.sourceName]);
}

/** Fetch the pinned ACPP bare Git root, prepare its original working checkout offline, and
 * retain both plus their receipts without compiling or provisioning an SDK.
 * @param {ReturnType<typeof ciPaths>} paths */
function fetchAcppSource(paths) {
  const state = acppStatePaths(paths), git = canonicalExecutable("/usr/bin/git");
  // The fetcher is the only networked step; the existing preparer allows only local Git transport.
  runNode("scripts/fetch-native-sdk-sources.js", ["acpp", git, state.source, "--allow-network"], paths.env, 15 * 60 * 1000);
  verifyGitSource(paths, state.source, git, "acpp");
  prepareSdkSources("acpp", readBuildJson(state.rootsFile), state.prepared, paths.env);
  verifyAcppSource(paths, state, git);
  packState(state.sourceTar, temp, [state.sourceName]);
  listState(state.sourceTar, [state.sourceName]);
}

/** Restore same-run Hermes source and build/audit the original static SDK with explicit tools.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildHermesSdk(paths) {
  const state = hermesStatePaths(paths), git = canonicalExecutable("/usr/bin/git");
  extractState(state.incomingTar, temp, [state.sourceName]);
  verifyHermesSource(paths, state, git);
  fs.unlinkSync(state.incomingTar);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-hermes-" + paths.suffix);
  const tools = hermesBuildTools();
  const command = hermesBuildCommand(state, tools);
  runNode(command.script, command.args, paths.env, 55 * 60 * 1000);
  auditHermesSdk(state, tools);
  packState(state.sdkTar, temp, [state.sourceName, state.sdkName]);
}

/** Fetch only the original pinned source archive and preserve its immutable receipt at the same /tmp path. @param {ReturnType<typeof ciPaths>} paths */
function fetchZstdSource(paths) {
  const state = zstdStatePaths(paths);
  runNode("scripts/fetch-native-sdk-sources.js", ["zstd", state.source, "--allow-network"], paths.env, 15 * 60 * 1000);
  verifyZstdArchive(state.source);
  packState(state.sourceTar, temp, [state.sourceName]);
}

/** Restore the original same-run archive, build with the signed Intel 2026.2 compiler and retain the complete admitted SDK root. @param {ReturnType<typeof ciPaths>} paths */
function buildZstdSdk(paths) {
  const state = zstdStatePaths(paths);
  extractState(state.incomingTar, temp, [state.sourceName]);
  verifyZstdArchive(state.source);
  fs.unlinkSync(state.incomingTar);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-zstd-" + paths.suffix);
  const tools = linuxZstdTools(), config = {version: 1, jobs: 4, tools};
  fs.writeFileSync(state.toolsFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  try {
    const command = zstdBuildCommand(state, state.toolsFile);
    runNode(command.script, command.args, paths.env, 25 * 60 * 1000);
    auditZstdSdk(state, tools, state.toolsFile);
    packState(state.sdkTar, temp, [state.sourceName, state.sdkName]);
  } finally {
    fs.unlinkSync(state.toolsFile);
  }
}

/** Explicit signed Intel 2026.2, CMake/Ninja and OS inspection tools for static OpenMP production. */
function openmpBuildTools() {
  return {cc: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icx"),
    cmake: canonicalExecutable("/usr/bin/cmake"), cxx: canonicalExecutable("/opt/intel/oneapi/compiler/2026.2/bin/icpx"),
    ninja: canonicalExecutable("/usr/bin/ninja"), readelf: canonicalExecutable("/usr/bin/readelf"), tar: canonicalExecutable("/usr/bin/tar")};
}

/** Return the maintained producer invocation for this run's verified original archive.
 * @param {ReturnType<typeof openmpStatePaths>} state */
function openmpBuildCommand(state) {
  return {script: "scripts/build-native-openmp-sdk.js", args: [state.archive, state.sdkRoot, state.toolsFile]};
}

/** Reuse full producer admission, then bind the artifact to this exact same-run source handoff.
 * @param {ReturnType<typeof openmpStatePaths>} state @param {Record<string,string>} tools */
function auditOpenmpSdk(state, tools) {
  const sdk = nativeOpenmpInputs(state.sdkRoot), record = readBuildJson(sdk.receipt),
    expectedTools = Object.fromEntries(Object.entries(tools).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}]));
  if (record.archive.path !== state.archive || record.fetchReceipt.path !== path.join(state.source, "fetch.json") ||
      record.plan.jobs !== 2 || !isDeepStrictEqual(record.tools, expectedTools) ||
      sdk.archive !== path.join(state.sdkRoot, "sdk/libomp.a") || sdk.files[state.archive] !== record.archive.sha256 ||
      sdk.files[sdk.receipt] !== sdk.receiptSha256 || sdk.files[sdk.archive] !== record.sdk.sha256) {
    throw new Error("Static OpenMP SDK does not bind this run's pinned archive, tools and ACPP handoff path");
  }
  return sdk;
}

/** Restore the same-run source state, provision Intel 2026.2 within a separate budget, and build only static libomp.a.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildOpenmpSdk(paths) {
  const state = openmpStatePaths(paths);
  if (fs.existsSync(state.sdkRoot) || fs.existsSync(state.toolsFile)) {throw new Error("Native OpenMP SDK output or tool descriptor already exists");}
  restorePrior(paths, "sources", false);
  provisionOneApi(paths, "/var/tmp/mom-dev-native-openmp-" + paths.suffix, 40 * 60 * 1000);
  const tools = openmpBuildTools(), config = {version: 1, jobs: 2, tools};
  fs.writeFileSync(state.toolsFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  try {
    const command = openmpBuildCommand(state);
    runNode(command.script, command.args, paths.env, 10 * 60 * 1000);
    auditOpenmpSdk(state, tools);
    packState(state.sdkTar, temp, [state.sourceName, state.sdkName]);
    listState(state.sdkTar, [state.sourceName, state.sdkName]);
  } finally {
    fs.unlinkSync(state.toolsFile);
  }
}

/** @typedef {{suffix: string, runnerTemp: string, name: string, workRoot: string, source: string, unpacked: string,
 * output: string, descriptor: string, include: string, bundle: string, archive: string}} WindowsNativeCorePaths */
/** Keep Windows Core on its native runner: its MSBuild/COFF tools and Node-gyp inputs are not
 * interchangeable with the Linux SDK stage. Every path is derived from this run attempt.
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @returns {WindowsNativeCorePaths} */
function windowsNativeCoreStatePaths(environment = process.env, platform = process.platform) {
  const runnerTemp = environment["RUNNER_TEMP"] || "", run = environment["GITHUB_RUN_ID"] || "", attempt = environment["GITHUB_RUN_ATTEMPT"] || "";
  if (platform !== "win32" || !path.isAbsolute(runnerTemp) || !/^[0-9]{1,20}$/.test(run) || !/^[1-9][0-9]{0,5}$/.test(attempt)) {
    throw new Error("Windows native Core CI needs a Windows runner and exact run-attempt paths");
  }
  const suffix = run + "-" + attempt, name = "native-windows-core-" + suffix,
    workRoot = path.join(path.resolve(runnerTemp), "mom-native-windows-core-" + suffix);
  return {suffix, runnerTemp: path.resolve(runnerTemp), name, workRoot, source: path.join(workRoot, "io-source"),
    unpacked: path.join(workRoot, "libuv-unpacked"), output: path.join(workRoot, "producer-output"), descriptor: path.join(workRoot, "windows-core.json"),
    include: path.join(workRoot, "libuv-unpacked", ioPins.libuv.directory, "include"), bundle: path.join(path.resolve(runnerTemp), name),
    archive: path.join(path.resolve(runnerTemp), name + ".tar")};
}

/** Cap each Windows Core child process to the job-start deadline while preserving time for remaining stages.
 * @param {string} deadlineEpoch @param {"fetch" | "listing" | "extract" | "build" | "package"} phase @param {number} [nowMs]
 * @returns {{timeoutMs: number, createTimeoutMs: number, listingTimeoutMs: number, remainingMs: number}} */
function windowsNativeCoreDeadlineBudget(deadlineEpoch, phase, nowMs = Date.now()) {
  if (!/^\d{10}$/.test(deadlineEpoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 ||
      !["fetch", "listing", "extract", "build", "package"].includes(phase)) {
    throw new Error("Windows Core stage needs a valid bounded job-start deadline");
  }
  const remainingMs = Number(deadlineEpoch) * 1000 - nowMs, maximumMs = 60 * 60 * 1000;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > maximumMs) {
    throw new Error("Windows Core job deadline is expired or exceeds the 60-minute job bound");
  }
  const reserve = windowsNativeCorePackReserveMs + windowsNativeCoreMinBuildMs;
  if (phase === "fetch") {
    // Leave four minutes for archive inspection/extraction, a minimal build window, and final handoff.
    const remainingAfterFetch = 2 * windowsNativeCoreMaxArchiveStepMs + reserve;
    if (remainingMs < remainingAfterFetch + 60 * 1000) {
      throw new Error("Insufficient aggregate job time remains before fetching Windows Core inputs");
    }
    return {timeoutMs: Math.min(windowsNativeCoreMaxFetchMs, remainingMs - remainingAfterFetch), createTimeoutMs: 0, listingTimeoutMs: 0, remainingMs};
  }
  if (phase === "listing" || phase === "extract") {
    const futureArchive = phase === "listing" ? windowsNativeCoreMaxArchiveStepMs : 0;
    const reserveAfter = futureArchive + reserve;
    if (remainingMs < reserveAfter + 60 * 1000) {
      throw new Error("Insufficient aggregate job time remains for Windows Core archive preparation");
    }
    return {timeoutMs: Math.min(windowsNativeCoreMaxArchiveStepMs, remainingMs - reserveAfter), createTimeoutMs: 0, listingTimeoutMs: 0, remainingMs};
  }
  if (phase === "build") {
    if (remainingMs < reserve + 60 * 1000) {
      throw new Error("Insufficient aggregate job time remains before the Windows Core producer build");
    }
    return {timeoutMs: Math.min(windowsNativeCoreMaxBuildMs, remainingMs - windowsNativeCorePackReserveMs), createTimeoutMs: 0, listingTimeoutMs: 0, remainingMs};
  }
  if (remainingMs < windowsNativeCorePackReserveMs) {
    throw new Error("Insufficient aggregate job time remains for the Windows Core handoff");
  }
  return {timeoutMs: windowsNativeCorePackReserveMs, createTimeoutMs: 5 * 60 * 1000, listingTimeoutMs: 2 * 60 * 1000, remainingMs};
}

/** The workflow resolves VS, Python, Node-gyp and Intel tools before entering this function.
 * The public producer validates each selected file and verifies the Intel version/target.
 * @param {WindowsNativeCorePaths} state @param {NodeJS.ProcessEnv} [environment] */
function windowsNativeCoreDescriptor(state, environment = process.env) {
  const {validateWindowsCoreDescriptor, validatePortableEnvironment} = require("./build-native-windows-core");
  validatePortableEnvironment(environment);
  return validateWindowsCoreDescriptor({version: 1, nodeGyp: environment["MOM_NATIVE_WINDOWS_CORE_NODE_GYP"],
    msbuild: environment["MOM_NATIVE_WINDOWS_CORE_MSBUILD"], dumpbin: environment["MOM_NATIVE_WINDOWS_CORE_DUMPBIN"],
    python: environment["MOM_NATIVE_WINDOWS_CORE_PYTHON"], compiler: environment["MOM_NATIVE_WINDOWS_CORE_COMPILER"], libuvInclude: state.include});
}

/** Assert the consumer-side portable COFF Core contract before retaining a same-run tar.
 * `inputs.json` remains the public producer's original receipt bytes.
 * @param {WindowsNativeCorePaths} state
 * @returns {{archive: string, receiptFile: string, archiveSha256: string, receiptSha256: string, receipt: Record<string, unknown>, bytes: number}} */
function auditWindowsNativeCoreOutput(state) {
  const archive = path.join(state.output, "cpu-core.lib"), receiptFile = path.join(state.output, "inputs.json"),
    receiptStat = fs.lstatSync(receiptFile), archiveStat = fs.lstatSync(archive);
  if (!receiptStat.isFile() || receiptStat.isSymbolicLink() || !archiveStat.isFile() || archiveStat.isSymbolicLink() || !isStaticArchive(archive)) {
    throw new Error("Windows Core output is not a regular static COFF archive with a regular receipt");
  }
  const receipt = readBuildJson(receiptFile);
  const archiveSha256 = fileHash(archive), receiptSha256 = fileHash(receiptFile), fact = receipt.archive;
  if (receipt.target !== "win32" || receipt.gpu !== true || receipt.portable !== true || receipt.buildOnly !== true ||
      receipt.executed !== false || receipt.publicationQualified !== false || !receipt.compiler ||
      !/^2026\.2(?:\.|$)/.test(receipt.compiler.version || "") || receipt.compiler.target !== "x86_64-pc-windows-msvc" ||
      receipt.objectSha256 !== archiveSha256 || !fact || fact.path !== archive || fact.sha256 !== archiveSha256 || fact.bytes !== archiveStat.size ||
      fact.stubs !== false || !Number.isSafeInteger(fact.coffTableCount) || fact.coffTableCount < 1 ||
      fact.coffTableCount !== fact.machineCount || fact.machineCount !== fact.x64MachineCount) {
    throw new Error("Windows Core receipt fails the portable x64 consumer contract");
  }
  verifySourceInputs(receipt.sourceState, nativeSourceInputs(repo, "cpu"), "Windows Core CI handoff");
  if (!receipt.headerHashes || typeof receipt.headerHashes !== "object" || Array.isArray(receipt.headerHashes) ||
      !Object.keys(receipt.headerHashes).length || Object.keys(receipt.headerHashes).length > 4096) {
    throw new Error("Windows Core receipt has no bounded libuv-header input closure");
  }
  const includeStat = fs.lstatSync(state.include);
  if (!includeStat.isDirectory() || includeStat.isSymbolicLink() || fs.realpathSync(state.include).toLowerCase() !== path.resolve(state.include).toLowerCase()) {
    throw new Error("Windows Core libuv include root is redirected");
  }
  const headers = new Set();
  for (const [file, hash] of Object.entries(receipt.headerHashes)) {
    if (!path.isAbsolute(file) || typeof hash !== "string" || !/^[0-9a-f]{64}$/.test(hash)) {throw new Error("Windows Core header receipt is malformed");}
    const relative = path.relative(state.include, file), normalized = relative.replaceAll("\\", "/"), header = fs.lstatSync(file);
    if (path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep) ||
        !/\.(?:h|hpp|inc)$/i.test(file) || !header.isFile() || header.isSymbolicLink() ||
        fs.realpathSync(file).toLowerCase() !== path.resolve(file).toLowerCase() || fileHash(file) !== hash) {
      throw new Error("Windows Core header input changed or escaped the pinned libuv include root");
    }
    headers.add(normalized);
  }
  if (!headers.has("uv.h") || !headers.has("uv/win.h")) {throw new Error("Windows Core libuv header closure is incomplete");}
  return {archive, receiptFile, archiveSha256, receiptSha256, receipt, bytes: archiveStat.size};
}

/** @typedef {{core_sha256: string, receipt_sha256: string, handoff_sha256: string, bundle_sha256: string, bundle_bytes: number, tar_tool_sha256: string}} WindowsNativeCoreReport */
/** Package only the Core archive, unchanged producer receipt, and a small transport index.
 * The Windows SDK/toolchain and downloaded source archive are intentionally not uploaded.
 * @param {WindowsNativeCorePaths} state @param {string} sourceReceiptSha256 @param {string} libuvArchiveSha256
 * @param {{createTimeoutMs?: number, listingTimeoutMs?: number}} [timeouts]
 * @returns {WindowsNativeCoreReport} */
function packWindowsNativeCoreHandoff(state, sourceReceiptSha256, libuvArchiveSha256, timeouts = {}) {
  const createTimeoutMs = timeouts.createTimeoutMs ?? 5 * 60 * 1000, listingTimeoutMs = timeouts.listingTimeoutMs ?? 2 * 60 * 1000;
  if (!Number.isSafeInteger(createTimeoutMs) || createTimeoutMs < 1 || createTimeoutMs > 5 * 60 * 1000 ||
      !Number.isSafeInteger(listingTimeoutMs) || listingTimeoutMs < 1 || listingTimeoutMs > 2 * 60 * 1000) {
    throw new Error("Windows Core handoff tar timeouts are invalid");
  }
  const audited = auditWindowsNativeCoreOutput(state);
  if (libuvArchiveSha256 !== ioPins.libuv.sha256 || !/^[0-9a-f]{64}$/.test(sourceReceiptSha256 || "")) {
    throw new Error("Windows Core handoff source provenance differs from the pinned libuv input");
  }
  if (fs.existsSync(state.bundle) || fs.existsSync(state.archive)) {throw new Error("Windows Core handoff output already exists");}
  fs.mkdirSync(state.bundle, {mode: 0o700});
  const coreFile = path.join(state.bundle, "cpu-core.lib"), receiptFile = path.join(state.bundle, "inputs.json");
  fs.copyFileSync(audited.archive, coreFile, fs.constants.COPYFILE_EXCL);
  fs.copyFileSync(audited.receiptFile, receiptFile, fs.constants.COPYFILE_EXCL);
  if (fileHash(coreFile) !== audited.archiveSha256 || fileHash(receiptFile) !== audited.receiptSha256) {
    throw new Error("Windows Core handoff changed the producer output bytes");
  }
  const handoff = {format: "mom-native-windows-core-handoff-v1", runAttempt: state.suffix, target: "win32", architecture: "x64",
    archive: {file: "cpu-core.lib", bytes: audited.bytes, sha256: audited.archiveSha256},
    producer: {script: "scripts/build-native-windows-core.js", receipt: "inputs.json", receiptSha256: audited.receiptSha256},
    libuv: {version: "1.51.0", archiveSha256: libuvArchiveSha256, fetchReceiptSha256: sourceReceiptSha256},
    buildOnly: true, executed: false, publicationQualified: false};
  fs.writeFileSync(path.join(state.bundle, "handoff.json"), JSON.stringify(handoff, null, 2) + "\n", {flag: "wx", mode: 0o600});
  if (!isDeepStrictEqual(fs.readdirSync(state.bundle).sort(), ["cpu-core.lib", "handoff.json", "inputs.json"])) {
    throw new Error("Windows Core handoff has unexpected members");
  }
  const handoffFile = path.join(state.bundle, "handoff.json"), handoffSha256 = fileHash(handoffFile),
    admitted = windowsNativeCoreHandoffInputs(state.bundle, {runAttempt: state.suffix, handoffSha256});
  if (admitted.archiveSha256 !== audited.archiveSha256 || admitted.receiptSha256 !== audited.receiptSha256) {
    throw new Error("Windows Core handoff reader rejected the original producer payload");
  }
  const systemRoot = process.env["SystemRoot"] || process.env["WINDIR"] || "", tarTool = path.join(systemRoot, "System32", "tar.exe"),
    tarStat = fs.lstatSync(tarTool);
  if (!path.isAbsolute(tarTool) || !tarStat.isFile() || tarStat.isSymbolicLink() || fs.existsSync(state.archive)) {
    throw new Error("Windows Core handoff requires the Windows inbox tar tool and a new archive path");
  }
  const tarSha256 = fileHash(tarTool), pack = spawnSync(tarTool, ["-cf", state.archive, "-C", state.runnerTemp, state.name],
    {encoding: "utf8", windowsHide: true, timeout: createTimeoutMs, maxBuffer: 4 << 20});
  if (pack.error || pack.signal || pack.status !== 0) {throw new Error("Windows Core handoff tar creation failed", {cause: pack.error});}
  const listing = spawnSync(tarTool, ["-tf", state.archive], {encoding: "utf8", windowsHide: true, timeout: listingTimeoutMs, maxBuffer: 4 << 20});
  if (listing.error || listing.signal || listing.status !== 0 || typeof listing.stdout !== "string") {
    throw new Error("Windows Core handoff tar listing failed", {cause: listing.error});
  }
  const members = listing.stdout.trim().split(/\r?\n/).filter(Boolean).map(name => name.replaceAll("\\", "/").replace(/\/$/, "")).sort(),
    expected = [state.name, state.name + "/cpu-core.lib", state.name + "/handoff.json", state.name + "/inputs.json"].sort();
  if (!isDeepStrictEqual(members, expected) || fileHash(handoffFile) !== admitted.handoffSha256) {
    throw new Error("Windows Core handoff tar has unexpected members or changed source receipt");
  }
  const report = {core_sha256: audited.archiveSha256, receipt_sha256: audited.receiptSha256,
    handoff_sha256: admitted.handoffSha256, bundle_sha256: fileHash(state.archive), bundle_bytes: fs.statSync(state.archive).size, tar_tool_sha256: tarSha256};
  console.log(JSON.stringify(report));
  return report;
}

/** Fetch only the pinned libuv/zlib source cohort, then build and retain the public Windows Core. */
function buildWindowsNativeCoreCi() {
  const state = windowsNativeCoreStatePaths();
  /** @type {NodeJS.ProcessEnv} */
  const environment = {...process.env, MOM_AOT_BUILD_ONLY: "1"};
  const deadlineEpoch = environment["MOM_NATIVE_WINDOWS_CORE_DEADLINE_EPOCH"] || "";
  const githubOutput = environment["GITHUB_OUTPUT"] || "";
  if (!path.isAbsolute(githubOutput) || !fs.lstatSync(githubOutput).isFile() || fs.lstatSync(githubOutput).isSymbolicLink()) {
    throw new Error("Windows Core CI needs the workflow's regular GITHUB_OUTPUT file");
  }
  for (const output of [state.workRoot, state.bundle, state.archive]) {
    if (fs.existsSync(output)) {throw new Error("Windows native Core stage refuses an existing run-attempt output: " + path.basename(output));}
  }
  const tempStat = fs.lstatSync(state.runnerTemp);
  if (!tempStat.isDirectory() || tempStat.isSymbolicLink() || path.resolve(fs.realpathSync(state.runnerTemp)).toLowerCase() !== state.runnerTemp.toLowerCase()) {
    throw new Error("Windows Core RUNNER_TEMP must be canonical and not redirected");
  }
  const fetchBudget = windowsNativeCoreDeadlineBudget(deadlineEpoch, "fetch");
  fs.mkdirSync(state.workRoot, {mode: 0o700});
  runNode("scripts/fetch-native-sdk-sources.js", ["io", state.source, "--allow-network"], environment, fetchBudget.timeoutMs);
  verifyIoArchives(state.source);
  const fetchReceipt = path.join(state.source, "fetch.json"), sourceReceiptSha256 = fileHash(fetchReceipt),
    libuvArchive = path.join(state.source, ioPins.libuv.directory + ".tar.gz"), libuvArchiveSha256 = fileHash(libuvArchive),
    systemRoot = environment["SystemRoot"] || environment["WINDIR"] || "", tarTool = path.join(systemRoot, "System32", "tar.exe");
  if (libuvArchiveSha256 !== ioPins.libuv.sha256 || !path.isAbsolute(tarTool) || !fs.lstatSync(tarTool).isFile()) {
    throw new Error("Windows Core needs the exact libuv pin and Windows inbox tar.exe");
  }
  const listingBudget = windowsNativeCoreDeadlineBudget(deadlineEpoch, "listing");
  const listing = spawnSync(tarTool, ["-tzf", libuvArchive], {encoding: "utf8", windowsHide: true, timeout: listingBudget.timeoutMs, maxBuffer: 4 << 20});
  if (listing.error || listing.signal || listing.status !== 0 || typeof listing.stdout !== "string") {throw new Error("Pinned libuv archive listing failed", {cause: listing.error});}
  validateIoArchive(listing.stdout, ioPins.libuv.directory);
  fs.mkdirSync(state.unpacked, {mode: 0o700});
  const extractBudget = windowsNativeCoreDeadlineBudget(deadlineEpoch, "extract");
  const unpack = spawnSync(tarTool, ["-xzf", libuvArchive, "-C", state.unpacked, ioPins.libuv.directory + "/include"],
    {encoding: "utf8", windowsHide: true, timeout: extractBudget.timeoutMs, maxBuffer: 4 << 20});
  if (unpack.error || unpack.signal || unpack.status !== 0) {throw new Error("Pinned libuv header extraction failed", {cause: unpack.error});}
  const config = windowsNativeCoreDescriptor(state, environment);
  if (fs.existsSync(state.output) || fs.existsSync(state.descriptor)) {throw new Error("Windows Core producer output is not new");}
  fs.writeFileSync(state.descriptor, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  // Fail on tool/layout mistakes before the expensive public producer starts.
  require("./build-native-windows-core").windowsCorePlan(config, state.output, repo);
  const buildBudget = windowsNativeCoreDeadlineBudget(deadlineEpoch, "build");
  runNode("scripts/build-native-windows-core.js", [state.descriptor, state.output], environment, buildBudget.timeoutMs);
  verifyIoArchives(state.source);
  const packageBudget = windowsNativeCoreDeadlineBudget(deadlineEpoch, "package");
  const result = packWindowsNativeCoreHandoff(state, sourceReceiptSha256, libuvArchiveSha256, packageBudget);
  if (fileHash(fetchReceipt) !== sourceReceiptSha256 || fileHash(libuvArchive) !== libuvArchiveSha256) {
    throw new Error("Windows Core source acquisition changed after production");
  }
  fs.appendFileSync(githubOutput, `core_sha256=${result.core_sha256}\nreceipt_sha256=${result.receipt_sha256}\nhandoff_sha256=${result.handoff_sha256}\nbundle_sha256=${result.bundle_sha256}\n`, "utf8");
  return result;
}

/** Same-run source/composition handoff for the selected Windows host-SDK roots. The vendor SDK
 * remains on this disposable runner; only receipts and path/hash metadata cross Actions.
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {string} [architecture]
 * @returns {{suffix:string,runnerTemp:string,workRoot:string,workName:string,archiveRoot:string,archiveName:string,
 * sdkOutput:string,outputName:string,metadataTar:string,deadlineEpoch:string}} */
function windowsHostSdkStatePaths(environment = process.env, platform = process.platform, architecture = process.arch) {
  const runnerTemp = environment["RUNNER_TEMP"] || "", run = environment["GITHUB_RUN_ID"] || "", attempt = environment["GITHUB_RUN_ATTEMPT"] || "";
  if (platform !== "linux" || architecture !== "x64" || !path.isAbsolute(runnerTemp) || !/^[0-9]{1,20}$/.test(run) || !/^[1-9][0-9]{0,5}$/.test(attempt)) {
    throw new Error("Windows host-SDK CI needs a Linux x64 runner and exact run-attempt paths");
  }
  const canonicalTemp = fs.realpathSync(runnerTemp);
  if (canonicalTemp !== path.resolve(runnerTemp) || !fs.lstatSync(canonicalTemp).isDirectory()) {throw new Error("Windows host-SDK RUNNER_TEMP is not canonical");}
  const suffix = run + "-" + attempt, workName = "mom-native-windows-host-sdk-" + suffix,
    outputName = "native-windows-host-sdk-" + suffix, workRoot = path.join(canonicalTemp, workName),
    archiveRoot = path.join(workRoot, "archives"), sdkOutput = path.join(workRoot, outputName),
    metadataTar = path.join(canonicalTemp, outputName + "-metadata.tar");
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1" || environment["MOM_BUILD_JOBS"] !== ciBuildJobs(environment) ||
      environment["MOM_NATIVE_WINDOWS_HOST_SDK_WORK_ROOT"] !== workRoot || environment["MOM_NATIVE_WINDOWS_HOST_SDK_ARCHIVE_DIR"] !== archiveRoot ||
      environment["MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR"] !== sdkOutput || environment["MOM_NATIVE_WINDOWS_HOST_SDK_METADATA_TAR"] !== metadataTar) {
    throw new Error("Windows host-SDK CI paths, job envelope, or build-only mode differ from this run");
  }
  return {suffix, runnerTemp: canonicalTemp, workRoot, workName, archiveRoot, archiveName: "archives", sdkOutput, outputName, metadataTar,
    deadlineEpoch: environment["MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH"] || ""};
}

/** Keep fetch, offline composition and metadata packaging inside one job-start deadline.
 * @param {string} deadlineEpoch @param {"fetch" | "compose" | "package"} phase @param {number} [nowMs] */
function windowsHostSdkDeadlineBudget(deadlineEpoch, phase, nowMs = Date.now()) {
  if (!/^\d{10}$/.test(deadlineEpoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 ||
      !["fetch", "compose", "package"].includes(phase)) {throw new Error("Windows host-SDK CI needs a bounded job-start deadline");}
  const remainingMs = Number(deadlineEpoch) * 1000 - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > windowsHostSdkMaxJobMs) {
    throw new Error("Windows host-SDK job deadline is expired or exceeds the 60-minute job bound");
  }
  if (phase === "fetch") {
    const reserve = windowsHostSdkComposeReserveMs + windowsHostSdkPackageReserveMs;
    if (remainingMs < reserve + 60 * 1000) {throw new Error("Insufficient job time remains for Windows host-SDK acquisition");}
    return {timeoutMs: Math.min(windowsHostSdkArchiveMaxMs, remainingMs - reserve), remainingMs};
  }
  if (phase === "compose") {
    if (remainingMs < windowsHostSdkComposeReserveMs + windowsHostSdkPackageReserveMs) {
      throw new Error("Insufficient job time remains for offline Windows host-SDK composition");
    }
    return {timeoutMs: Math.min(45 * 60 * 1000, remainingMs - windowsHostSdkPackageReserveMs), remainingMs};
  }
  if (remainingMs < 60 * 1000) {throw new Error("Insufficient job time remains for the Windows host-SDK receipt handoff");}
  return {timeoutMs: Math.min(windowsHostSdkPackageReserveMs, remainingMs), remainingMs};
}

/** Run an SDK producer in its own process group so a bounded timeout also contains compiler children.
 * @param {string} script @param {string[]} args @param {NodeJS.ProcessEnv} environment @param {number} timeoutMs */
function runNodeProcessGroup(script, args, environment, timeoutMs) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 45 * 60 * 1000) {throw new Error("SDK producer child timeout is invalid");}
  const timeoutPath = "/usr/bin/timeout", timeoutReal = fs.realpathSync(timeoutPath), timeoutStat = fs.statSync(timeoutReal);
  if (!timeoutStat.isFile()) {throw new Error("GNU timeout is unavailable for bounded SDK production");}
  const timeoutSha256 = fileHash(timeoutReal);
  const version = spawnSync(timeoutReal, ["--version"], {encoding: "utf8", timeout: 30000, maxBuffer: 64 * 1024});
  if (version.error || version.status !== 0 || !((version.stdout || "") + (version.stderr || "")).includes("GNU coreutils")) {
    throw new Error("Bounded SDK production requires GNU coreutils timeout");
  }
  const seconds = Math.max(1, Math.ceil(timeoutMs / 1000)), command = spawnSync(timeoutReal,
    ["--signal=TERM", "--kill-after=5s", seconds + "s", process.execPath, path.join(repo, script), ...args],
    {cwd: repo, env: environment, stdio: "inherit", timeout: timeoutMs + 10000});
  if (command.error || command.signal || command.status !== 0) {
    throw new Error("Bounded SDK producer failed: " + script + (command.status === 124 ? " (deadline)" : ""), {cause: command.error});
  }
  if (timeoutSha256 !== fileHash(timeoutPath)) {throw new Error("GNU timeout changed during SDK production");}
  return {timeoutPath: timeoutReal, timeoutSha256};
}

/** Append hash-bound outputs only after source acquisition and extraction both completed. @param {string} file @param {Record<string,string|number>} values */
function appendWindowsHostSdkOutputs(file, values) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || !path.isAbsolute(file)) {throw new Error("Windows host-SDK job output file is not regular");}
  const lines = Object.entries(values).map(([key, value]) => {
    if (!/^[a-z][a-z0-9_]*$/.test(key) || !/^(?:[a-f0-9]{64}|[0-9]+|native-windows-host-sdk-[0-9]+-[0-9]+-metadata\.tar)$/.test(String(value))) {
      throw new Error("Windows host-SDK output is malformed: " + key);
    }
    return key + "=" + value;
  });
  fs.appendFileSync(file, lines.join("\n") + "\n", "utf8");
}

/** Copy the two small immutable acquisition descriptors beside the composed receipt.
 * @param {ReturnType<typeof windowsHostSdkStatePaths>} state @param {string} fetchReceiptHash */
function retainWindowsHostSdkMetadata(state, fetchReceiptHash) {
  const receipt = path.join(state.archiveRoot, "fetch.json"), receiptStat = fs.lstatSync(receipt), lockSource = path.join(repo, "runtime/windows-host-sdk.json"),
    lockTarget = path.join(state.sdkOutput, "source-lock.json"), receiptTarget = path.join(state.sdkOutput, "source-fetch.json");
  if (!receiptStat.isFile() || receiptStat.isSymbolicLink() || fileHash(receipt) !== fetchReceiptHash ||
      fs.existsSync(lockTarget) || fs.existsSync(receiptTarget)) {throw new Error("Windows host-SDK source receipts changed or collide");}
  const copies = /** @type {Array<[string, string]>} */ ([[receipt, receiptTarget], [lockSource, lockTarget]]);
  for (const [source, target] of copies) {
    fs.copyFileSync(source, target, fs.constants.COPYFILE_EXCL); fs.chmodSync(target, 0o444);
    if (fileHash(target) !== fileHash(source)) {throw new Error("Windows host-SDK metadata copy changed bytes");}
  }
  return {receiptTarget, lockTarget};
}

/** Fetch the 42 pinned vendor archives, compose the selected roots offline, and upload receipts only. */
function buildWindowsHostSdkSourcesCi() {
  const state = windowsHostSdkStatePaths(), environment = {...process.env, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: ciBuildJobs()},
    lock = windowsHostSdkCatalog(), outputFile = process.env["GITHUB_OUTPUT"] || "", summary = process.env["GITHUB_STEP_SUMMARY"] || "";
  if (Object.keys(lock.packages).length + Object.keys(lock.cabinets).length !== 42 || !path.isAbsolute(outputFile) || !path.isAbsolute(summary)) {
    throw new Error("Windows host-SDK CI requires the fixed 42-archive cohort and workflow output files");
  }
  for (const file of [outputFile, summary]) {
    const stat = fs.lstatSync(file); if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Windows host-SDK workflow output is not regular");}
  }
  for (const target of [state.workRoot, state.metadataTar]) {
    try {
      fs.lstatSync(target); throw new Error("Windows host-SDK CI refuses an existing run-attempt output");
    } catch (error) {if (Reflect.get(Object(error), "code") !== "ENOENT") {throw error;}}
  }
  const recipeLeaves = ["scripts/prepare-native-ci.js", "scripts/prepare-native-windows-sdk.js", "scripts/fetch-native-sdk-sources.js",
      "scripts/build-source-inputs.js", "runtime/windows-host-sdk.json"], recipe = sourceInputs(repo, [], recipeLeaves);
  fs.mkdirSync(state.workRoot, {mode: 0o700});
  const fetchBudget = windowsHostSdkDeadlineBudget(state.deadlineEpoch, "fetch");
  runNode("scripts/fetch-native-sdk-sources.js", ["windows-host-sdk", state.archiveRoot, "--allow-network"], environment, fetchBudget.timeoutMs);
  const fetched = verifyWindowsHostSdkArchives(state.archiveRoot), fetchReceiptPath = path.join(state.archiveRoot, "fetch.json"), fetchReceiptSha256 = fileHash(fetchReceiptPath);
  if (fetched.status !== "PASS" || fetched.target !== "windows-host" || fetched.networked !== true || fetched.buildOnly !== true ||
      fetched.compiled !== false || fetched.publicationQualified !== false ||
      Object.keys(fetched.archives || {}).length !== 42) {throw new Error("Windows host-SDK acquisition receipt is not the exact build-only cohort");}
  const composeBudget = windowsHostSdkDeadlineBudget(state.deadlineEpoch, "compose");
  const timeoutIdentity = runNodeProcessGroup("scripts/prepare-native-windows-sdk.js", [state.archiveRoot, state.sdkOutput], environment, composeBudget.timeoutMs);
  const resultFile = path.join(state.sdkOutput, "result.json"), descriptorFile = path.join(state.sdkOutput, "sdk.json"),
    overlayFile = path.join(state.sdkOutput, "vfs-overlay.json"), filesFile = path.join(state.sdkOutput, "files.json"),
    result = /** @type {{status:string,target:string,buildOnly:boolean,compiled:boolean,executed:boolean,publicationQualified:boolean,
      sourceCatalogPublisherAuthenticated:boolean,sourceCatalogBindingVerified:boolean,packageCount:number,cabinetCount:number,
      selectedFileCount:number,tools:unknown[],fetchReceipt:{path:string,sha256:string},sourceCatalog:{sha256:string},roots:string[],
      descriptor:{path:string,sha256:string},overlay:{path:string,sha256:string},provenance:{path:string,sha256:string}}} */ (readBuildJson(resultFile)),
    descriptor = /** @type {{sourceRoot:string,overlay:string,includeRoots:string[],libraryRoots:string[]}} */ (readBuildJson(descriptorFile)),
    files = /** @type {{files:Array<{path:string,sha256:string,bytes:number}>,origins:unknown[]}} */ (readBuildJson(filesFile)), sdkRoot = path.join(state.sdkOutput, "sdk");
  for (const file of [resultFile, descriptorFile, overlayFile, filesFile]) {
    const stat = fs.lstatSync(file); if (!stat.isFile() || stat.isSymbolicLink()) {throw new Error("Windows host-SDK composition output is not a regular file");}
  }
  const descriptorKeys = ["includeRoots", "libraryRoots", "overlay", "sourceRoot"];
  if (!result || result.status !== "PASS" || result.target !== "windows-host" || result.buildOnly !== true || result.compiled !== false ||
      result.executed !== false || result.publicationQualified !== false || result.sourceCatalogPublisherAuthenticated !== false ||
      result.sourceCatalogBindingVerified !== false || result.packageCount !== Object.keys(lock.packages).length ||
      result.cabinetCount !== Object.keys(lock.cabinets).length || !Number.isSafeInteger(result.selectedFileCount) || result.selectedFileCount < 1 ||
      result.fetchReceipt?.path !== fetchReceiptPath || result.fetchReceipt.sha256 !== fetchReceiptSha256 ||
      result.sourceCatalog?.sha256 !== lock.sourceCatalog.sha256 || !isDeepStrictEqual(result.roots, lock.roots) ||
      result.descriptor?.path !== descriptorFile || result.descriptor.sha256 !== fileHash(descriptorFile) ||
      result.overlay?.path !== overlayFile || result.overlay.sha256 !== fileHash(overlayFile) ||
      result.provenance?.path !== filesFile || result.provenance.sha256 !== fileHash(filesFile) ||
      !descriptor || Object.keys(descriptor).sort().join("\0") !== descriptorKeys.sort().join("\0") || descriptor.sourceRoot !== sdkRoot ||
      descriptor.overlay !== overlayFile || !Array.isArray(descriptor.includeRoots) || !descriptor.includeRoots.length ||
      !Array.isArray(descriptor.libraryRoots) || !descriptor.libraryRoots.length || !files || !Array.isArray(files.files) ||
      files.files.length !== result.selectedFileCount || !files.origins || !Array.isArray(files.origins)) {
    throw new Error("Windows host-SDK composition receipt or descriptor fails the source-only contract");
  }
  const seen = new Map();
  for (const row of files.files) {
    if (!row || !safeRelative(row.path) || !/^[0-9a-f]{64}$/.test(row.sha256 || "") || !Number.isSafeInteger(row.bytes) || row.bytes < 0 || seen.has(row.path)) {
      throw new Error("Windows host-SDK per-file provenance is malformed");
    }
    seen.set(row.path, row);
  }
  const actual = require("./prepare-native-windows-sdk").walkTree(sdkRoot);
  if (actual.files.length !== seen.size || actual.bytes !== files.files.reduce((sum, row) => sum + row.bytes, 0)) {
    throw new Error("Windows host-SDK composed tree differs from its selected-file provenance");
  }
  for (const file of actual.files) {
    const row = seen.get(file.path); if (!row || row.bytes !== file.bytes || fileHash(file.full) !== row.sha256) {throw new Error("Windows host-SDK composed file changed after extraction");}
  }
  verifyWindowsHostSdkArchives(state.archiveRoot);
  verifySourceInputs(recipe, sourceInputs(repo, [], recipeLeaves), "Windows host-SDK CI source/composition recipe");
  const retained = retainWindowsHostSdkMetadata(state, fetchReceiptSha256), metadataBudget = windowsHostSdkDeadlineBudget(state.deadlineEpoch, "package");
  const handoffFile = path.join(state.sdkOutput, "handoff.json"), handoff = {version: 1, format: "mom-windows-host-sdk-ci-metadata-v1",
    runAttempt: state.suffix, target: "windows-host", status: "PASS", sourceOnly: true, buildOnly: true, compiled: false, executed: false,
    publicationQualified: false, sourceCatalogPublisherAuthenticated: false, sourceCatalogBindingVerified: false,
    vendorPayloadIncluded: false, sourceLockSha256: fileHash(path.join(repo, "runtime/windows-host-sdk.json")),
    archiveCount: Object.keys(fetched.archives).length, packageCount: result.packageCount, cabinetCount: result.cabinetCount,
    selectedFileCount: result.selectedFileCount, selectedRoots: lock.roots,
    acquisition: {receipt: "source-fetch.json", sha256: fetchReceiptSha256},
    composition: {receipt: "result.json", sha256: fileHash(resultFile), descriptor: "sdk.json", descriptorSha256: fileHash(descriptorFile),
      overlaySha256: fileHash(overlayFile), files: "files.json", filesSha256: fileHash(filesFile)},
    extractionTools: result.tools, timeoutTool: timeoutIdentity, note: "Only source-selection metadata and hashes are retained. Vendor archives and SDK payload files remain runner-local and are not uploaded."};
  fs.writeFileSync(handoffFile, JSON.stringify(handoff, null, 2) + "\n", {flag: "wx", mode: 0o444});
  if (fileHash(retained.receiptTarget) !== fetchReceiptSha256 || fileHash(retained.lockTarget) !== handoff.sourceLockSha256) {
    throw new Error("Windows host-SDK retained source metadata changed");
  }
  const metadataHashes = /** @type {Array<[string,string]>} */ (["sdk.json", "vfs-overlay.json", "files.json", "result.json", "source-fetch.json", "source-lock.json", "handoff.json"]
    .map(name => {const file = path.join(state.sdkOutput, name); return [file, fileHash(file)];}));
  const members = windowsHostSdkPackageMembers(state), packed = packState(state.metadataTar, state.workRoot, members, metadataBudget.timeoutMs),
    listed = listState(state.metadataTar, [state.outputName]), allowed = new Set(["sdk.json", "vfs-overlay.json", "files.json", "result.json", "source-fetch.json", "source-lock.json", "handoff.json"]);
  for (const member of listed) {
    const normalized = member.endsWith("/") ? member.slice(0, -1) : member;
    if (normalized === state.outputName) {continue;}
    const prefix = state.outputName + "/"; if (!normalized.startsWith(prefix)) {throw new Error("Windows host-SDK metadata tar has an unexpected root");}
    const relative = normalized.slice(prefix.length); if (!allowed.has(relative) || relative.includes("/")) {throw new Error("Windows host-SDK tar contains vendor payload or unexpected metadata");}
  }
  if (packed.bytes > 64 * 1024 * 1024 || metadataHashes.some(([file, hash]) => fileHash(file) !== hash)) {
    throw new Error("Windows host-SDK metadata tar is excessive or its receipts changed while packing");
  }
  verifySourceInputs(recipe, sourceInputs(repo, [], recipeLeaves), "Windows host-SDK CI source/composition final verification");
  const bundleSha256 = fileHash(state.metadataTar), output = {metadata_bundle_sha256: bundleSha256,
    metadata_bundle_bytes: packed.bytes, sdk_receipt_sha256: fileHash(resultFile), descriptor_sha256: fileHash(descriptorFile),
    files_provenance_sha256: fileHash(filesFile), handoff_sha256: fileHash(handoffFile), selected_file_count: result.selectedFileCount};
  appendWindowsHostSdkOutputs(outputFile, output);
  fs.appendFileSync(summary, `### Windows host SDK source/composition (metadata only)\n- Acquired and composed ${result.selectedFileCount} selected files from the exact 42-archive lock; raw vendor archives and SDK payloads remained runner-local.\n- The retained artifact contains only receipts, descriptor/overlay, and per-file provenance; no compiler, application, runtime, publication, or publisher-authentication claim is made.\n`, "utf8");
  return output;
}

/** @param {Pick<ReturnType<typeof windowsHostSdkStatePaths>, "outputName">} state */
function windowsHostSdkPackageMembers(state) {
  return ["sdk.json", "vfs-overlay.json", "files.json", "result.json", "source-fetch.json", "source-lock.json", "handoff.json"]
    .map(name => state.outputName + "/" + name);
}

/** Validate the explicit link-order policy before opening any of its input paths.
 * @param {unknown} value @returns {{msvcLibraries:string[],ucrtLibraries:string[],importLibraries:string[]}} */
function validateWindowsLlvmLibraryPolicy(value) {
  const groups = ["msvcLibraries", "ucrtLibraries", "importLibraries"];
  if (!hasExactKeys(value, groups)) {throw new Error("Windows LLVM library policy must name all ordered library groups");}
  const seen = new Set(), policy = /** @type {Record<string,unknown>} */ (value), result = /** @type {Record<string,string[]>} */ ({});
  for (const group of groups) {
    const names = policy[group];
    if (!Array.isArray(names) || names.length < 1 || names.length > 256) {throw new Error("Windows LLVM library policy group is empty or excessive: " + group);}
    result[group] = names.map(name => {
      if (typeof name !== "string" || !/^[A-Za-z0-9_.+-]{1,128}\.lib$/i.test(name) ||
          safeWindowsSdkRelative(name) !== name || name.includes("/") || seen.has(name.toLowerCase())) {
        throw new Error("Windows LLVM library policy contains an unsafe or duplicate basename");
      }
      seen.add(name.toLowerCase()); return name;
    });
  }
  return /** @type {{msvcLibraries:string[],ucrtLibraries:string[],importLibraries:string[]}} */ (result);
}

/** Hash-pin a regular Linux tool file; permit only the explicitly declared sibling alias.
 * @param {{path:string,sha256:string,linkTarget?:string}} pin @param {string} label */
function auditWindowsLlvmToolFile(pin, label) {
  const keys = pin && Object.hasOwn(pin, "linkTarget") ? ["path", "sha256", "linkTarget"] : ["path", "sha256"];
  if (!hasExactKeys(pin, keys) || !path.isAbsolute(pin.path) || /[\r\n\0]/.test(pin.path) || !/^[a-f0-9]{64}$/.test(pin.sha256)) {
    throw new Error("Windows LLVM tool pin is malformed: " + label);
  }
  const stat = fs.lstatSync(pin.path); let file = pin.path;
  if (stat.isSymbolicLink()) {
    if (typeof pin.linkTarget !== "string" || !/^[A-Za-z0-9_.+-]{1,128}$/.test(pin.linkTarget) || fs.readlinkSync(pin.path) !== pin.linkTarget) {
      throw new Error("Windows LLVM tool alias differs from its pin: " + label);
    }
    file = path.join(path.dirname(pin.path), pin.linkTarget);
  } else if (Object.hasOwn(pin, "linkTarget")) {throw new Error("Windows LLVM tool pin declares a missing alias: " + label);}
  const target = fs.lstatSync(file);
  if (!target.isFile() || target.isSymbolicLink() || target.nlink !== 1 || fs.realpathSync(file) !== path.resolve(file) ||
      target.size < 1 || target.size > (512 << 20) || fileHash(file) !== pin.sha256) {
    throw new Error("Windows LLVM tool bytes or canonical path differ from their pin: " + label);
  }
  return {path: pin.path, resolvedPath: file, bytes: target.size, sha256: pin.sha256,
    ...(Object.hasOwn(pin, "linkTarget") ? {linkTarget: pin.linkTarget} : {})};
}

/** Map one ordered basename list to exactly one selected, manifest-bound SDK archive per name.
 * @param {string[]} names @param {string[]} roots @param {Map<string,any>} files @param {string} sdkRoot @param {string} label */
function mapWindowsLlvmSdkLibraries(names, roots, files, sdkRoot, label) {
  return names.map(name => {
    const candidates = roots.map(root => root + "/" + name), found = candidates.map(candidate => files.get(candidate.normalize("NFC").toLowerCase())).filter(Boolean);
    if (found.length !== 1) {throw new Error("Windows LLVM " + label + " library must resolve to exactly one source file: " + name);}
    const row = found[0]; if (!row) {throw new Error("Windows LLVM " + label + " library was not found: " + name);}
    const file = path.join(sdkRoot, ...row.path.split("/")), stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== file ||
        stat.size !== row.bytes || fileHash(file) !== row.sha256 || !isStaticArchive(file)) {
      throw new Error("Windows LLVM selected " + label + " archive changed or is not regular: " + name);
    }
    return {path: file, manifestPath: row.path, bytes: stat.size, sha256: row.sha256,
      origins: row.origins.map((/** @type {any} */ origin) => ({package: origin.package, packageSha256: origin.packageSha256, member: origin.member}))};
  });
}

/** Read one small externally pinned JSON receipt without permitting a redirected or changing file.
 * @param {string} file @param {string} expectedSha256 @param {string} label @returns {any} */
function readWindowsLlvmPinnedJson(file, expectedSha256, label) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== path.resolve(file) ||
      stat.size < 1 || stat.size > (8 << 20) || fileHash(file) !== expectedSha256) {
    throw new Error("Windows LLVM receipt differs from its external pin: " + label);
  }
  const value = readBuildJson(file);
  if (fileHash(file) !== expectedSha256) {throw new Error("Windows LLVM receipt changed while reading: " + label);}
  return value;
}

/** Compose only the descriptor used by the maintained Windows LLVM builder. The host composer
 * remains source-only; this function neither builds LLVM nor asserts publisher authentication.
 * @param {{windowsSdkRoot:string,windowsSdkPins:{resultSha256:string,descriptorSha256:string,filesSha256:string,overlaySha256:string},
 * hermesRoot:string,hermesExpectation:{kind:"hermes",runAttempt:string,handoffSha256:string},
 * tools:{clangCl:{path:string,sha256:string},lldLink:{path:string,sha256:string,linkTarget?:string},llvmNm:{path:string,sha256:string},clangResourceInclude:string},
 * libraryPolicy?:{msvcLibraries:string[],ucrtLibraries:string[],importLibraries:string[]}}} input
 * @returns {{descriptor:Record<string,unknown>,provenance:Record<string,unknown>}} */
function composeWindowsLlvmSdkDescriptor(input) {
  return composeWindowsLlvmDescriptor(input, "runAttempt");
}

/** Local builds retain the same SDK/tool/runtime admission without inventing CI provenance.
 * @param {Omit<Parameters<typeof composeWindowsLlvmSdkDescriptor>[0],"hermesExpectation"> &
 * {hermesExpectation:{kind:"hermes",localBuildId:string,handoffSha256:string}}} input */
function composeLocalWindowsLlvmSdkDescriptor(input) {
  return composeWindowsLlvmDescriptor(input, "localBuildId");
}

/** Only the transport entrypoint differs; the descriptor and source policy stay shared.
 * @param {Parameters<typeof composeWindowsLlvmSdkDescriptor>[0] | Parameters<typeof composeLocalWindowsLlvmSdkDescriptor>[0]} input
 * @param {"runAttempt"|"localBuildId"} identityKey */
function composeWindowsLlvmDescriptor(input, identityKey) {
  const pinKeys = ["resultSha256", "descriptorSha256", "filesSha256", "overlaySha256"], toolKeys = ["clangCl", "lldLink", "llvmNm", "clangResourceInclude"],
    policy = validateWindowsLlvmLibraryPolicy(input?.libraryPolicy === undefined ? defaultWindowsNativeLibraryPolicy() : input.libraryPolicy);
  if (!input || !path.isAbsolute(input.windowsSdkRoot) || /[\r\n\0]/.test(input.windowsSdkRoot) ||
      !hasExactKeys(input.windowsSdkPins, pinKeys) || Object.values(input.windowsSdkPins).some(hash => !/^[a-f0-9]{64}$/.test(hash)) ||
      !hasExactKeys(input.hermesExpectation, ["kind", identityKey, "handoffSha256"]) || input.hermesExpectation.kind !== "hermes" ||
      !hasExactKeys(input.tools, toolKeys) || typeof input.tools.clangResourceInclude !== "string" ||
      !path.isAbsolute(input.tools.clangResourceInclude) || /[\r\n\0]/.test(input.tools.clangResourceInclude)) {
    throw new Error("Windows LLVM descriptor composition requires explicit source pins and tool paths");
  }
  const handoff = "runAttempt" in input.hermesExpectation ?
    windowsControllerHandoffInputs(input.hermesRoot, input.hermesExpectation) :
    windowsLocalControllerHandoffInputs(input.hermesRoot, input.hermesExpectation);
  if (handoff.kind !== "hermes") {throw new Error("Windows LLVM descriptor requires a Hermes handoff");}
  const hermes = handoff, intelRuntime = hermes.intelRuntime;
  if (!intelRuntime) {throw new Error("Windows LLVM descriptor requires Intel runtime inputs in the Hermes handoff");}
  const {root, sdkRoot, resultPath, sdkPath, filesPath, overlayPath, sdk, files, actualTree} =
    require("./native-windows-host-sdk-inputs").windowsHostSdkInputs(input.windowsSdkRoot, input.windowsSdkPins);

  const toolProof = {clangCl: auditWindowsLlvmToolFile(input.tools.clangCl, "clangCl"),
      lldLink: auditWindowsLlvmToolFile(input.tools.lldLink, "lldLink"), llvmNm: auditWindowsLlvmToolFile(input.tools.llvmNm, "llvmNm")},
    resource = input.tools.clangResourceInclude, resourceStat = fs.lstatSync(resource);
  if (!resourceStat.isDirectory() || resourceStat.isSymbolicLink() || fs.realpathSync(resource) !== path.resolve(resource)) {
    throw new Error("Windows LLVM Clang resource include directory is not canonical");
  }
  const compilerRoot = path.resolve(path.dirname(input.tools.clangCl.path), ".."), relativeResource = path.relative(compilerRoot, resource);
  if (path.isAbsolute(relativeResource) || relativeResource === ".." || relativeResource.startsWith(".." + path.sep)) {
    throw new Error("Windows LLVM resource headers are outside the pinned compiler root");
  }
  const runtimeCatalog = windowsIntelRuntimeCatalog.libraries, runtimeByName = new Map(runtimeCatalog.map(row => [row.name.toLowerCase(), row])),
    runtimeMap = intelRuntime.libraries;
  if (!isDeepStrictEqual(Object.keys(runtimeMap).sort(), runtimeCatalog.map(row => row.name).sort())) {
    throw new Error("Windows Hermes handoff Intel library set differs from its shared pin catalog");
  }
  const msvcLibraries = /** @type {any[]} */ (policy.msvcLibraries.map((/** @type {string} */ name) => {
    const pin = runtimeByName.get(name.toLowerCase());
    if (!pin) {
      const row = mapWindowsLlvmSdkLibraries([name], ["msvc/lib"], files, sdkRoot, "MSVC")[0];
      if (!row) {throw new Error("Windows LLVM MSVC library is absent: " + name);} return row;
    }
    const file = runtimeMap[pin.name], expectedPath = path.join(path.resolve(input.hermesRoot), "intel-runtime", "lib", pin.name);
    if (!file) {throw new Error("Windows Intel runtime archive is absent from its handoff: " + pin.name);}
    const stat = fs.lstatSync(file);
    if (file !== expectedPath || !stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== file ||
        stat.size !== pin.bytes || fileHash(file) !== pin.sha256 || !isStaticArchive(file)) {
      throw new Error("Windows Intel runtime archive differs from the admitted Hermes handoff: " + pin.name);
    }
    return {path: file, manifestPath: "intel-runtime/lib/" + pin.name, bytes: stat.size, sha256: pin.sha256,
      origin: {handoffSha256: hermes.handoffSha256, catalog: "runtime/windows-intel-runtime.json"}};
  }));
  if (runtimeCatalog.some(row => !policy.msvcLibraries.some(name => name.toLowerCase() === row.name.toLowerCase()))) {
    throw new Error("Windows LLVM MSVC policy omits a pinned Intel runtime library");
  }
  const ucrtLibraries = mapWindowsLlvmSdkLibraries(policy.ucrtLibraries, ["msvc/lib", "sdk/lib/ucrt"], files, sdkRoot, "UCRT"),
    importLibraries = mapWindowsLlvmSdkLibraries(policy.importLibraries, ["sdk/lib/um"], files, sdkRoot, "Windows import"),
    intelLibraryRoot = intelRuntime.libRoot;
  const expectedSdkLibraryRoots = ["sdk/lib/ucrt", "sdk/lib/um", "msvc/lib"].map(relative => path.join(sdkRoot, ...relative.split("/")));
  if (!Array.isArray(sdk.includeRoots) || !Array.isArray(sdk.libraryRoots) || sdk.libraryRoots.length !== 3 ||
      !isDeepStrictEqual(sdk.libraryRoots, expectedSdkLibraryRoots) ||
      !sdk.libraryRoots.every((/** @type {string} */ rootPath) => path.isAbsolute(rootPath) && fs.lstatSync(rootPath).isDirectory() &&
        !fs.lstatSync(rootPath).isSymbolicLink() && fs.realpathSync(rootPath) === rootPath) ||
      sdk.libraryRoots.some((/** @type {string} */ rootPath) => rootPath.toLowerCase() === intelLibraryRoot.toLowerCase())) {
    throw new Error("Windows host-SDK library roots are redirected, incomplete or collide with Intel runtime");
  }
  const descriptor = {version: 1, tools: {clangCl: input.tools.clangCl.path, lldLink: input.tools.lldLink.path,
    llvmNm: input.tools.llvmNm.path, clangResourceInclude: resource}, windowsSdk: {sourceRoot: sdk.sourceRoot, overlay: sdk.overlay,
    includeRoots: sdk.includeRoots, libraryRoots: [...sdk.libraryRoots, intelLibraryRoot],
    msvcLibraries: msvcLibraries.map(row => row.path), ucrtLibraries: ucrtLibraries.map(row => row.path),
    importLibraries: importLibraries.map(row => row.path)}};
  require("./build-native-windows-controller").validateWindowsLlvmSdkConfig(descriptor);
  const finalSourcePins = /** @type {Array<[string,string]>} */ ([[resultPath, input.windowsSdkPins.resultSha256],
    [sdkPath, input.windowsSdkPins.descriptorSha256], [filesPath, input.windowsSdkPins.filesSha256], [overlayPath, input.windowsSdkPins.overlaySha256]]);
  for (const [file, hash] of finalSourcePins) {
    if (fileHash(file) !== hash) {throw new Error("Windows LLVM descriptor source changed during composition: " + path.basename(file));}
  }
  return {descriptor, provenance: {version: 1, role: "windows-llvm21-descriptor-input-map", status: "PASS", buildOnly: true,
    compiled: false, executed: false, publicationQualified: false,
    sourceTrust: {windowsCatalogPublisherAuthenticated: false, windowsChannelCatalogBindingVerified: false},
    windowsSdk: {root, resultSha256: input.windowsSdkPins.resultSha256, descriptorSha256: input.windowsSdkPins.descriptorSha256,
      filesSha256: input.windowsSdkPins.filesSha256, overlaySha256: input.windowsSdkPins.overlaySha256,
      selectedFileCount: actualTree.files.length, selectedBytes: actualTree.bytes},
    hermes: {root: path.resolve(input.hermesRoot),
      ...("runAttempt" in hermes ? {runAttempt: hermes.runAttempt} : {localBuildId: hermes.localBuildId}), handoffSha256: hermes.handoffSha256,
      intelRuntime: {libraries: windowsIntelRuntimeCatalog.libraries.map(row => ({name: row.name, path: intelRuntime.libraries[row.name],
        manifestPath: row.path, bytes: row.bytes, sha256: row.sha256})),
      licenses: windowsIntelRuntimeCatalog.licenses.map(row => ({path: intelRuntime.licenses[row.path], manifestPath: row.path,
        bytes: row.bytes, sha256: row.sha256}))}},
    tools: {...toolProof, clangResourceInclude: resource}, libraryPolicy: policy,
    libraries: {msvcLibraries, ucrtLibraries, importLibraries}}};
}

/** Fixed run-scoped outputs for composing the Windows LLVM descriptor before configure.
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {string} [architecture] */
function windowsNativeLlvmDescriptorCiStatePaths(environment = process.env, platform = process.platform, architecture = process.arch) {
  const ci = windowsNativeLlvmCiStatePaths(environment, platform, architecture),
    root = path.join(ci.runnerTemp, "mom-native-windows-llvm21-descriptor-" + ci.suffix),
    windowsSdkRoot = path.join(ci.runnerTemp, "mom-native-windows-host-sdk-" + ci.suffix, "native-windows-host-sdk-" + ci.suffix),
    hermesRoot = path.join(ci.runnerTemp, "native-windows-controller-hermes-" + ci.suffix);
  if (environment["MOM_NATIVE_WINDOWS_LLVM_DESCRIPTOR_DIR"] !== root || path.dirname(root) !== ci.runnerTemp ||
      /[\r\n\0]/.test(ci.runnerTemp)) {
    throw new Error("Windows LLVM descriptor output must match the exact same-run path");
  }
  return {...ci, configureRoot: ci.root, root, descriptorOutput: path.join(root, "descriptor.json"), provenanceOutput: path.join(root, "provenance.json"),
    windowsSdkRoot, hermesRoot};
}

/** Validate a pinned composer input without mutating its source or configure roots.
 * @param {string[]} args @param {NodeJS.ProcessEnv} environment @param {NodeJS.Platform} platform @param {string} architecture @param {number} nowMs */
function windowsNativeLlvmDescriptorCiPreflight(args, environment = process.env, platform = process.platform, architecture = process.arch, nowMs = Date.now()) {
  if (args.length !== 2) {throw new Error("Windows LLVM descriptor stage needs one input JSON and its SHA-256 pin");}
  const state = windowsNativeLlvmDescriptorCiStatePaths(environment, platform, architecture),
    reports = windowsNativeLlvmCiReports(environment), budget = windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "configure", nowMs, state.runDeadlineEpoch),
    inputFile = args[0] || "", inputSha256 = args[1] || "";
  if (!/^[a-f0-9]{64}$/.test(inputSha256) || !path.isAbsolute(inputFile) || /[\r\n\0]/.test(inputFile) ||
      !windowsNativeLlvmCiContains(state.runnerTemp, inputFile) || path.resolve(fs.realpathSync(inputFile)) !== path.resolve(inputFile)) {
    throw new Error("Windows LLVM descriptor input must be a canonical pinned same-run JSON file");
  }
  const inputStat = windowsControllerRegularFile(inputFile, "Windows LLVM descriptor composition input");
  if (inputStat.nlink !== 1 || inputStat.size < 1 || inputStat.size > (8 << 20) || fileHash(inputFile) !== inputSha256) {
    throw new Error("Windows LLVM descriptor composition input differs from its external SHA-256 pin");
  }
  const input = readWindowsLlvmPinnedJson(inputFile, inputSha256, "descriptor composition input"),
    required = ["windowsSdkRoot", "windowsSdkPins", "hermesRoot", "hermesExpectation", "tools"], keys = Object.keys(input || {}).sort().join("\0");
  if (keys !== required.sort().join("\0") && keys !== [...required, "libraryPolicy"].sort().join("\0")) {
    throw new Error("Windows LLVM descriptor input has an unexpected composer schema");
  }
  if (input.windowsSdkRoot !== state.windowsSdkRoot || input.hermesRoot !== state.hermesRoot ||
      !hasExactKeys(input.hermesExpectation, ["kind", "runAttempt", "handoffSha256"]) ||
      input.hermesExpectation.kind !== "hermes" || input.hermesExpectation.runAttempt !== state.suffix ||
      !hasExactKeys(input.tools, ["clangCl", "lldLink", "llvmNm", "clangResourceInclude"])) {
    throw new Error("Windows LLVM descriptor SDK/Hermes inputs must be the exact same-run roots and handoff");
  }
  const toolPins = [
    ["clangCl", input.tools.clangCl, ["path", "sha256"]],
    ["lldLink", input.tools.lldLink, Object.hasOwn(input.tools.lldLink || {}, "linkTarget") ? ["path", "sha256", "linkTarget"] : ["path", "sha256"]],
    ["llvmNm", input.tools.llvmNm, ["path", "sha256"]],
  ];
  for (const [label, pin, pinKeys] of toolPins) {
    if (!hasExactKeys(pin, pinKeys)) {throw new Error("Windows LLVM descriptor tool pin is malformed: " + label);}
  }
  if (typeof input.tools.clangResourceInclude !== "string" || !path.isAbsolute(input.tools.clangResourceInclude) ||
      /[\r\n\0]/.test(input.tools.clangResourceInclude)) {throw new Error("Windows LLVM Clang resource include path is malformed");}
  const toolIdentities = toolPins.map(([label, pin]) => auditWindowsLlvmToolFile(pin, label));
  const resourceStat = fs.lstatSync(input.tools.clangResourceInclude);
  if (!resourceStat.isDirectory() || resourceStat.isSymbolicLink() || fs.realpathSync(input.tools.clangResourceInclude) !== path.resolve(input.tools.clangResourceInclude)) {
    throw new Error("Windows LLVM Clang resource include directory is not canonical");
  }
  const inputRoots = [repo, inputFile, state.windowsSdkRoot, state.hermesRoot, ...toolIdentities.map(row => path.dirname(row.resolvedPath)),
    input.tools.clangResourceInclude];
  windowsNativeLlvmCiRejectReportInputOverlap(reports, [...inputRoots, state.root, state.configureRoot, state.output, state.descriptor]);
  const overlapsOutput = inputRoots.some(item => path.isAbsolute(item) &&
    windowsNativeLlvmCiContains(path.resolve(item), state.root) || windowsNativeLlvmCiContains(state.root, path.resolve(item))) ||
    windowsNativeLlvmCiContains(state.root, state.configureRoot) || windowsNativeLlvmCiContains(state.configureRoot, state.root);
  const outputStat = fs.lstatSync(state.root, {throwIfNoEntry: false});
  if (overlapsOutput || outputStat || [state.configureRoot, state.output, state.descriptor].some(file => fs.existsSync(file))) {
    throw new Error("Windows LLVM descriptor output must be fresh and separate from its inputs and configure state");
  }
  return {state, reports, budget, inputFile, inputSha256, inputStat, input, toolPins, toolIdentities};
}

/** Compose and retain the build-only descriptor through the existing full SDK/handoff admission.
 * The composer callback is an internal test seam; the CLI always supplies the maintained composer.
 * @param {string[]} args @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform]
 * @param {string} [architecture] @param {number} [nowMs]
 * @param {(input:any)=>{descriptor:Record<string,unknown>,provenance:Record<string,unknown>}} [composer] */
function buildWindowsNativeLlvmDescriptorCi(args, environment = process.env, platform = process.platform, architecture = process.arch,
  nowMs = Date.now(), composer = composeWindowsLlvmSdkDescriptor) {
  const preflight = windowsNativeLlvmDescriptorCiPreflight(args, environment, platform, architecture, nowMs),
    {state, reports, inputFile, inputSha256, input, toolPins, toolIdentities} = preflight,
    composed = composer(input);
  if (!composed || !composed.descriptor || typeof composed.descriptor !== "object" || Array.isArray(composed.descriptor) ||
      !composed.provenance || typeof composed.provenance !== "object" || Array.isArray(composed.provenance)) {
    throw new Error("Windows LLVM composer returned an invalid descriptor/provenance pair");
  }
  const verifyInputs = () => {
    const currentInput = windowsControllerRegularFile(inputFile, "Windows LLVM descriptor composition input");
    if (currentInput.nlink !== 1 || currentInput.dev !== preflight.inputStat.dev || currentInput.ino !== preflight.inputStat.ino ||
        currentInput.size !== preflight.inputStat.size || fileHash(inputFile) !== inputSha256) {
      throw new Error("Windows LLVM descriptor composition input changed during production");
    }
    const currentTools = toolPins.map(([label, pin]) => auditWindowsLlvmToolFile(pin, label));
    if (!isDeepStrictEqual(currentTools, toolIdentities)) {throw new Error("Windows LLVM descriptor tool identity changed during production");}
  };
  verifyInputs();
  windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "configure", Date.now(), state.runDeadlineEpoch);
  const descriptorBytes = Buffer.from(JSON.stringify(composed.descriptor, null, 2) + "\n"), descriptorSha256 = createHash("sha256").update(descriptorBytes).digest("hex"),
    provenance = {version: 1, format: "mom-windows-llvm21-descriptor-ci-v1", status: "PASS", runAttempt: state.suffix,
      deadlineEpoch: state.deadlineEpoch, buildOnly: true, compiled: false, executed: false, publicationQualified: false,
      input: {path: inputFile, sha256: inputSha256}, descriptor: {path: state.descriptorOutput, sha256: descriptorSha256}, composition: composed.provenance},
    provenanceBytes = Buffer.from(JSON.stringify(provenance, null, 2) + "\n");
  if (descriptorBytes.length > (4 << 20) || provenanceBytes.length > (8 << 20)) {throw new Error("Windows LLVM descriptor provenance exceeds its handoff size bound");}
  fs.mkdirSync(state.root, {mode: 0o700});
  fs.writeFileSync(state.descriptorOutput, descriptorBytes, {flag: "wx", mode: 0o444});
  fs.writeFileSync(state.provenanceOutput, provenanceBytes, {flag: "wx", mode: 0o444});
  const verified = composer(input);
  verifyInputs();
  windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "configure", Date.now(), state.runDeadlineEpoch);
  if (!isDeepStrictEqual(verified, composed) || fileHash(state.descriptorOutput) !== descriptorSha256 ||
      fileHash(state.provenanceOutput) !== createHash("sha256").update(provenanceBytes).digest("hex")) {
    throw new Error("Windows LLVM descriptor source, tool or retained output changed during production");
  }
  const values = {windows_llvm_descriptor_path: state.descriptorOutput, windows_llvm_descriptor_sha256: descriptorSha256,
    windows_llvm_descriptor_provenance_path: state.provenanceOutput, windows_llvm_descriptor_provenance_sha256: fileHash(state.provenanceOutput)};
  windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "outputs", Date.now(), state.runDeadlineEpoch);
  for (const report of reports) {verifyWindowsNativeLlvmCiReport(report);}
  appendWorkflowOutputs(environment, values);
  fs.appendFileSync(environment["GITHUB_STEP_SUMMARY"] || "", `### Windows LLVM 21 descriptor inputs\n- Same-run descriptor: ${state.descriptorOutput} (SHA-256 ${descriptorSha256}).\n` +
    `- Provenance: ${state.provenanceOutput} (SHA-256 ${values.windows_llvm_descriptor_provenance_sha256}).\n` +
    "- Build-only: true; compiled: false; executed: false; publication-qualified: false.\n", "utf8");
  return values;
}

const windowsGpuHandoffProfiles = Object.freeze(["acpp-cuda", "acpp-hip", "dpcpp", "dpcpp-opencl", "oneapi"]);
const windowsGpuHandoffMaxFiles = 512, windowsGpuHandoffMaxBytes = 512 << 20;
const windowsGpuPackReserveMs = 7 * 60 * 1000, windowsGpuMinBuildMs = 5 * 60 * 1000, windowsGpuMaxBuildMs = 45 * 60 * 1000;

/** Keep one Windows GPU factory build and its archive inside the job-start deadline.
 * @param {string} deadlineEpoch @param {"build" | "package"} phase @param {number} [nowMs] */
function windowsGpuDeadlineBudget(deadlineEpoch, phase, nowMs = Date.now()) {
  if (!/^\d{10}$/.test(deadlineEpoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 || !["build", "package"].includes(phase)) {
    throw new Error("Windows GPU profile stage needs a valid bounded job-start deadline");
  }
  const remainingMs = Number(deadlineEpoch) * 1000 - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > 60 * 60 * 1000) {
    throw new Error("Windows GPU profile deadline is expired or exceeds the 60-minute job bound");
  }
  if (phase === "build") {
    if (remainingMs < windowsGpuMinBuildMs + windowsGpuPackReserveMs + 60 * 1000) {
      throw new Error("Insufficient aggregate job time remains before the Windows GPU producer");
    }
    return {timeoutMs: Math.min(windowsGpuMaxBuildMs, remainingMs - windowsGpuPackReserveMs), remainingMs};
  }
  if (remainingMs < 60 * 1000) {throw new Error("Insufficient aggregate job time remains for the Windows GPU handoff");}
  return {timeoutMs: Math.min(windowsGpuPackReserveMs, remainingMs), remainingMs};
}

/** Run-scoped producer paths for one independent Windows GPU factory job.
 * @param {string} profile @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {string} [architecture] */
function windowsGpuHandoffStatePaths(profile, environment = process.env, platform = process.platform, architecture = process.arch) {
  const runnerTemp = environment["RUNNER_TEMP"] || "", run = environment["GITHUB_RUN_ID"] || "", attempt = environment["GITHUB_RUN_ATTEMPT"] || "";
  if (platform !== "linux" || architecture !== "x64" || !windowsGpuHandoffProfiles.includes(profile) ||
      !path.isAbsolute(runnerTemp) || !/^[0-9]{1,20}$/.test(run) || !/^[1-9][0-9]{0,5}$/.test(attempt)) {
    throw new Error("Windows GPU handoff needs a Linux x64 runner, exact run-attempt identity and one maintained profile");
  }
  const canonicalTemp = fs.realpathSync(runnerTemp), suffix = run + "-" + attempt,
    name = "mom-native-windows-gpu-" + profile + "-" + suffix, root = path.join(temp, name), output = path.join(root, "output"),
    isolatedRoot = path.join(output, "isolated"), receipt = path.join(output, "result.json"), isolation = path.join(isolatedRoot, "result.json"),
    configFile = path.join(root, "config.json"), archive = path.join(canonicalTemp, "native-windows-gpu-" + profile + "-" + suffix + ".tar");
  if (canonicalTemp !== path.resolve(runnerTemp) || path.dirname(root) !== temp || path.resolve(root) !== root ||
      environment["MOM_NATIVE_WINDOWS_GPU_DIR"] !== root || path.dirname(archive) !== canonicalTemp ||
      path.resolve(archive) !== archive || root === archive || root.startsWith(archive + path.sep)) {
    throw new Error("Windows GPU producer and archive paths differ from the exact run-attempt plan");
  }
  return {profile, suffix, name, root, output, isolatedRoot, receipt, isolation, configFile, archive, runnerTemp: canonicalTemp};
}

/** Validate one factory and select only its unchanged receipts and isolated objects. The
 * downstream full-application reader still re-admits all five graphs and the complete SDK.
 * @param {string} profile @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform]
 * @param {string} [architecture] @param {string} [sourceRoot] */
function windowsGpuProfileHandoffPlan(profile, environment = process.env, platform = process.platform,
  architecture = process.arch, sourceRoot = repo) {
  const state = windowsGpuHandoffStatePaths(profile, environment, platform, architecture), canonicalSource = path.resolve(sourceRoot);
  if (!path.isAbsolute(sourceRoot) || fs.realpathSync(canonicalSource) !== canonicalSource) {throw new Error("Windows GPU handoff source root must be canonical");}
  for (const directory of [state.root, state.output, state.isolatedRoot]) {
    const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
    if (!stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
      throw new Error("Windows GPU handoff producer root is not a canonical directory");
    }
  }
  const graph = windowsControllerReadJson(state.receipt, 16 << 20, "Windows GPU factory receipt"),
    configStat = windowsControllerRegularFile(state.configFile, "Windows GPU profile configuration"),
    configHash = fileHash(state.configFile), config = windowsControllerReadJson(state.configFile, 4 << 20, "Windows GPU profile configuration");
  if (configStat.nlink !== 1 || graph.version !== 1 || graph.profile !== profile ||
      graph.factory !== "mom_native_gpu_" + profile.replaceAll("-", "_") || graph.configFile !== state.configFile ||
      !/^[a-f0-9]{64}$/.test(graph.configSha256 || "") || configHash !== graph.configSha256 || config.profile !== profile ||
      graph.isolation !== state.isolation || !Array.isArray(graph.objects) || !graph.objects.length || graph.objects.length > 128 ||
      !graph.sourceState || !graph.sdkInputs || typeof graph.sdkInputs !== "object" || Array.isArray(graph.sdkInputs) ||
      Object.keys(graph.sdkInputs).length > 16384 ||
      Object.entries(graph.sdkInputs).some(([file, sha]) => !path.isAbsolute(file) || !/^[a-f0-9]{64}$/.test(sha || ""))) {
    throw new Error("Windows GPU receipt differs from the selected profile or its same-run inputs");
  }
  require("./build-native-windows-gpu").validateWindowsGpuConfig(config);
  verifySourceInputs(graph.sourceState, nativeSourceInputs(canonicalSource, "gpu"), "Windows GPU handoff");
  // Compiler inputs stay producer provenance in the compact handoff, so seal it only
  // while the original SDK cohort still matches; packaging rechecks this plan afterward.
  if (!Object.keys(graph.sdkInputs).length) {throw new Error("Windows GPU handoff omits its SDK input cohort");}
  for (const [file, sha256] of Object.entries(graph.sdkInputs)) {
    // Tool aliases are permitted by the original compiler receipts; bind their resolved
    // bytes rather than imposing a new no-symlink rule on otherwise valid SDK inputs.
    if (!fs.statSync(file).isFile() || fileHash(file) !== sha256) {
      throw new Error("Windows GPU SDK input changed before handoff: " + file);
    }
  }
  const isolatedStat = windowsControllerRegularFile(state.isolation, "Windows GPU isolation receipt"), isolated = windowsControllerReadJson(state.isolation, 16 << 20, "Windows GPU isolation receipt");
  if (isolatedStat.nlink !== 1 || isolated.prefix !== "mom_" + profile.replaceAll("-", "_") + "_gpu_" ||
      !isDeepStrictEqual(isolated.entryPoints, [graph.factory]) || !isDeepStrictEqual(isolated.outputs, graph.objects) ||
      ["nonDirectivePayloadsPreserved", "nonDirectiveMetadataPreserved", "comdatSelectionPreserved", "associativeXdataPreserved",
        "linkerDirectiveReferencesPreserved", "externalUndefinedImportsPreserved", "privateUndefinedReferencesRenamed"]
        .some(key => isolated.checks?.[key] !== true) || isolated.checks?.oldCollisionDefinitions !== 0 || isolated.checks?.oldCollisionReferences !== 0) {
    throw new Error("Windows GPU isolation output differs from the unchanged factory receipt");
  }
  const {coffSectionPayloads} = require("./native-compiler-isolation"),
    files = /** @type {Array<{path:string,member:string,bytes:number,sha256:string,role:string}>} */ ([]), seen = new Set(); let selectedBytes = 0;
  /** @param {string} file @param {string} expected @param {string} role */
  const select = (file, expected, role) => {
    const stat = windowsControllerRegularFile(file, "Windows GPU " + role), relative = path.relative(temp, file).split(path.sep).join("/");
    if (stat.nlink !== 1 || !safeRelative(relative) || !file.startsWith(state.root + path.sep) ||
        !/^[a-f0-9]{64}$/.test(expected || "") || fileHash(file) !== expected || seen.has(relative.toLowerCase())) {
      throw new Error("Windows GPU selected payload is unsafe, duplicated or changed: " + role);
    }
    seen.add(relative.toLowerCase()); selectedBytes += stat.size;
    if (!Number.isSafeInteger(selectedBytes) || selectedBytes > windowsGpuHandoffMaxBytes || files.length >= windowsGpuHandoffMaxFiles) {
      throw new Error("Windows GPU profile payload exceeds its bounded handoff size or file count");
    }
    files.push({path: file, member: relative, bytes: stat.size, sha256: expected, role});
  };
  select(state.receipt, fileHash(state.receipt), "factory receipt");
  select(state.isolation, fileHash(state.isolation), "isolation receipt");
  select(state.configFile, configHash, "profile configuration");
  for (const row of graph.objects) {
    if (!row || typeof row.path !== "string" || path.dirname(row.path) !== state.isolatedRoot ||
        !/^object-[0-9]{4}\.obj$/i.test(path.basename(row.path)) || !Number.isSafeInteger(row.bytes) || row.bytes < 1) {
      throw new Error("Windows GPU isolated object is outside its exact output directory");
    }
    const stat = windowsControllerRegularFile(row.path, "Windows GPU isolated object");
    if (stat.size !== row.bytes) {throw new Error("Windows GPU isolated object size differs from its receipt");}
    coffSectionPayloads(row.path);
    select(row.path, row.sha256, "isolated object");
  }
  return {state, files, members: files.map(row => row.member), selectedBytes, graphSha256: fileHash(state.receipt),
    isolationSha256: fileHash(state.isolation), configSha256: configHash};
}

/** Pack one independently built factory without its unrelated compiler-isolation diagnostics.
 * @param {string} profile @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform]
 * @param {string} [architecture] @param {string} [sourceRoot] */
function packWindowsGpuHandoff(profile, environment = process.env, platform = process.platform,
  architecture = process.arch, sourceRoot = repo, timeoutMs = 30 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 30 * 60 * 1000) {
    throw new Error("Windows GPU profile handoff timeout is outside the bounded range");
  }
  const deadline = Date.now() + timeoutMs;
  /** @param {string} phase @param {number} [capMs] @param {number} [minimumMs] */
  const remaining = (phase, capMs = timeoutMs, minimumMs = 1) => {
    const milliseconds = deadline - Date.now();
    if (!Number.isSafeInteger(milliseconds) || milliseconds < minimumMs) {
      throw new Error("Windows GPU profile handoff deadline expired before " + phase);
    }
    return Math.min(milliseconds, capMs);
  };
  const plan = windowsGpuProfileHandoffPlan(profile, environment, platform, architecture, sourceRoot), state = plan.state,
    before = plan.files.map(({member, bytes, sha256, role}) => ({member, bytes, sha256, role})),
    graph = windowsControllerReadJson(state.receipt, 16 << 20, "Windows GPU factory receipt"),
    handoffFile = path.join(state.root, "handoff.json"),
    handoff = {format: "mom-windows-gpu-handoff-v1", runAttempt: state.suffix, profile,
      files: plan.files.map(row => ({path: path.relative(state.root, row.path).split(path.sep).join("/"), bytes: row.bytes, sha256: row.sha256}))
        .sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0),
      sdkInputs: Object.fromEntries(Object.keys(graph.sdkInputs).sort().map(file => [file, graph.sdkInputs[file]])),
      buildOnly: true, executed: false, publicationQualified: false},
    handoffBytes = Buffer.from(JSON.stringify(handoff, null, 2) + "\n");
  if (handoffBytes.length > 8 << 20) {throw new Error("Windows GPU handoff attestation exceeds its byte bound");}
  // This seals producer-local SDK provenance; the consumer still verifies original graph,
  // source, isolation, config and object bytes using externally supplied handoff hashes.
  fs.writeFileSync(handoffFile, handoffBytes, {flag: "wx", mode: 0o444});
  const handoffSha256 = fileHash(handoffFile), handoffMember = state.name + "/handoff.json",
    packed = packState(state.archive, temp, [...plan.members, handoffMember], remaining("archive creation")), actual = new Set();
  const listing = listState(state.archive, [state.name], remaining("archive listing", 5 * 60 * 1000, 1000));
  if (!Number.isSafeInteger(packed.bytes) || packed.bytes < 1 || packed.bytes > windowsGpuHandoffMaxBytes + (8 << 20)) {
    throw new Error("Windows GPU profile archive exceeds its bounded transfer size");
  }
  for (const raw of listing) {
    const name = raw.endsWith("/") ? raw.slice(0, -1) : raw;
    if (!name || actual.has(name)) {throw new Error("Windows GPU handoff tar contains a duplicate member");}
    actual.add(name);
  }
  const allowed = new Set([...before.map(row => row.member), handoffMember]);
  for (const file of before) {
    let parent = path.posix.dirname(file.member);
    while (parent !== ".") {allowed.add(parent); parent = path.posix.dirname(parent);}
  }
  if (!actual.has(handoffMember) || before.some(row => !actual.has(row.member)) || [...actual].some(name => !allowed.has(name))) {
    throw new Error("Windows GPU handoff tar contains extra, missing or substituted members");
  }
  remaining("final input admission");
  const afterPlan = windowsGpuProfileHandoffPlan(profile, environment, platform, architecture, sourceRoot),
    after = afterPlan.files.map(({member, bytes, sha256, role}) => ({member, bytes, sha256, role}));
  if (!isDeepStrictEqual(before, after) || afterPlan.state.archive !== state.archive || fileHash(handoffFile) !== handoffSha256) {
    throw new Error("Windows GPU source payload changed while packaging its handoff");
  }
  remaining("final input admission");
  return {version: 1, format: "mom-windows-gpu-profile-handoff-v1", profile, runAttempt: state.suffix,
    root: state.name, archive: state.archive, bundleSha256: fileHash(state.archive), bundleBytes: packed.bytes,
    handoffSha256, files: before.length + 1, selectedBytes: plan.selectedBytes + handoffBytes.length,
    buildOnly: true, executed: false, publicationQualified: false};
}

/** Validate every external input and output path before creating the run-scoped producer root.
 * The expected digest is the caller's same-run handoff pin, not publisher authentication.
 * @param {string} profile @param {string} configSource @param {string} expectedConfigSha256
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {string} [architecture] @param {number} [nowMs] */
function windowsGpuProfileCiPreflight(profile, configSource, expectedConfigSha256, environment = process.env,
  platform = process.platform, architecture = process.arch, nowMs = Date.now()) {
  const state = windowsGpuHandoffStatePaths(profile, environment, platform, architecture),
    deadlineEpoch = environment["MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH"] || "";
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows GPU profile CI requires MOM_AOT_BUILD_ONLY=1");}
  const {windowsGpuBuildJobs} = require("./build-native-windows-gpu");
  if (windowsGpuBuildJobs(environment, platform) > 2) {throw new Error("Windows GPU profile CI is limited to two compiler jobs");}
  const buildBudget = windowsGpuDeadlineBudget(deadlineEpoch, "build", nowMs);
  windowsGpuDeadlineBudget(deadlineEpoch, "package", nowMs);
  if (typeof expectedConfigSha256 !== "string" || !/^[a-f0-9]{64}$/.test(expectedConfigSha256)) {
    throw new Error("Windows GPU profile CI requires the explicit same-run configuration SHA-256");
  }
  if (typeof configSource !== "string" || configSource.length > 4096 || /[\0\r\n]/.test(configSource) || !path.isAbsolute(configSource)) {
    throw new Error("Windows GPU profile configuration must be a canonical absolute file path");
  }
  const configStat = windowsControllerRegularFile(configSource, "Same-run Windows GPU configuration"),
    configRelative = path.relative(state.root, path.resolve(configSource));
  if (configStat.nlink !== 1 || configStat.size < 1 || configStat.size > (4 << 20) ||
      configRelative === "" || (!path.isAbsolute(configRelative) && configRelative !== ".." && !configRelative.startsWith(".." + path.sep))) {
    throw new Error("Windows GPU configuration must be a bounded external regular file");
  }
  const configBytes = fs.readFileSync(configSource);
  if (configBytes.length !== configStat.size || configBytes.length < 1 || configBytes.length > (4 << 20)) {
    throw new Error("Windows GPU configuration changed or exceeds its bounded input size");
  }
  const configSha256 = createHash("sha256").update(configBytes).digest("hex");
  if (configSha256 !== expectedConfigSha256) {throw new Error("Windows GPU configuration differs from its same-run SHA-256 pin");}
  let configValue;
  try {
    configValue = JSON.parse(configBytes.toString("utf8"));
  } catch (error) {throw new Error("Same-run Windows GPU configuration is not valid JSON", {cause: error});}
  const config = require("./build-native-windows-gpu").validateWindowsGpuConfig(configValue);
  if (config.profile !== profile) {throw new Error("Windows GPU configuration profile differs from the explicit producer profile");}
  const workflowFiles = /** @type {Array<[string, string]>} */ ([[environment["GITHUB_OUTPUT"] || "", "GitHub output"],
    [environment["GITHUB_STEP_SUMMARY"] || "", "GitHub step summary"]]);
  if (workflowFiles.some(([name]) => !name)) {throw new Error("Windows GPU profile CI requires GitHub output and summary files");}
  for (const [name, label] of workflowFiles) {
    windowsControllerRegularFile(name, label);
  }
  if (path.resolve(environment["GITHUB_OUTPUT"] || "") === path.resolve(environment["GITHUB_STEP_SUMMARY"] || "") ||
      [state.root, state.output, state.archive].some(file => fs.existsSync(file))) {
    throw new Error("Windows GPU profile CI requires distinct workflow reports and fresh producer/output paths");
  }
  return {state, configSource, expectedConfigSha256, configSha256, configBytes, config, deadlineEpoch, buildTimeoutMs: buildBudget.timeoutMs};
}

/** Stage exactly one caller-pinned profile config through the existing public producer and handoff.
 * @param {string} profile @param {string} configSource @param {string} expectedConfigSha256 @param {NodeJS.ProcessEnv} [environment] */
function buildWindowsGpuProfileCi(profile, configSource, expectedConfigSha256, environment = process.env) {
  const input = windowsGpuProfileCiPreflight(profile, configSource, expectedConfigSha256, environment), state = input.state;
  fs.mkdirSync(state.root, {mode: 0o700});
  fs.writeFileSync(state.configFile, input.configBytes, {flag: "wx", mode: 0o444});
  const copiedStat = windowsControllerRegularFile(state.configFile, "Staged Windows GPU configuration");
  if (copiedStat.nlink !== 1 || copiedStat.size < 1 || copiedStat.size > (4 << 20) || fileHash(state.configFile) !== input.configSha256) {
    throw new Error("Staged Windows GPU configuration differs from its same-run pin");
  }
  require("./build-native-windows-gpu").validateWindowsGpuConfig(
    windowsControllerReadJson(state.configFile, 4 << 20, "Staged Windows GPU configuration"));
  if (fileHash(input.configSource) !== input.expectedConfigSha256) {throw new Error("Same-run Windows GPU configuration changed before producer start");}
  const buildEnvironment = {...environment, MOM_AOT_BUILD_ONLY: "1"};
  runNode("scripts/build-native-windows-gpu.js", [state.configFile, state.output], buildEnvironment, input.buildTimeoutMs);
  const producer = windowsControllerReadJson(state.receipt, 16 << 20, "Windows GPU producer receipt"), resultSha256 = fileHash(state.receipt);
  if (producer.version !== 1 || producer.profile !== profile || producer.factory !== "mom_native_gpu_" + profile.replaceAll("-", "_") ||
      producer.configFile !== state.configFile || producer.configSha256 !== input.configSha256 ||
      fileHash(input.configSource) !== input.expectedConfigSha256 || fileHash(state.configFile) !== input.configSha256) {
    throw new Error("Windows GPU producer receipt does not preserve the selected profile and pinned config");
  }
  const packBudget = windowsGpuDeadlineBudget(input.deadlineEpoch, "package"),
    packed = packWindowsGpuHandoff(profile, environment, process.platform, process.arch, repo, packBudget.timeoutMs),
    values = {windows_gpu_profile: profile, windows_gpu_bundle_sha256: packed.bundleSha256,
      windows_gpu_result_sha256: resultSha256, windows_gpu_config_sha256: input.configSha256,
      windows_gpu_handoff_sha256: packed.handoffSha256,
      windows_gpu_bundle_bytes: String(packed.bundleBytes), windows_gpu_files: String(packed.files),
      windows_gpu_build_only: "true", windows_gpu_executed: "false", windows_gpu_publication_qualified: "false"};
  if (packed.buildOnly !== true || packed.executed !== false || packed.publicationQualified !== false || fileHash(state.receipt) !== resultSha256 ||
      fileHash(state.archive) !== packed.bundleSha256) {throw new Error("Windows GPU profile handoff changed or carries an unsafe scope");}
  appendWorkflowOutputs(environment, values);
  const summary = environment["GITHUB_STEP_SUMMARY"] || "", summaryBody = `### Windows GPU profile ${profile}\n` +
    `- Retained the original public producer receipt and ${packed.files} unchanged files (${packed.selectedBytes} bytes).\n` +
    `- Bundle SHA-256: ${packed.bundleSha256}; producer result SHA-256: ${resultSha256}.\n` +
    "- Build-only: true; executed: false; publication-qualified: false. This is not GPU/runtime qualification.\n";
  windowsControllerRegularFile(summary, "GitHub step summary");
  fs.appendFileSync(summary, summaryBody, "utf8");
  console.log(`[native-ci] Built and retained the same-run ${profile} Windows GPU factory; build-only, not executed or publication-qualified (${packed.files} files).`);
  return {...packed, resultSha256, configSha256: input.configSha256};
}

const windowsNativeLlvmPackReserveMs = 60 * 1000, windowsNativeLlvmMinPhaseMs = 5 * 60 * 1000,
  windowsNativeLlvmMaxChildMs = 45 * 60 * 1000, windowsNativeLlvmMaxJobMs = 60 * 60 * 1000,
  windowsNativeLlvmMaxRunMs = 8 * 60 * 60 * 1000;

/** The run deadline is set before configure and must not be recreated by a later job.
 * @param {string} epoch @param {number} [nowMs] */
function windowsNativeLlvmCiRunDeadlineBudget(epoch, nowMs = Date.now()) {
  if (!/^\d{10}$/.test(epoch) || !Number.isSafeInteger(nowMs) || nowMs < 0) {
    throw new Error("Windows LLVM CI needs a valid run-start deadline");
  }
  const remainingMs = Number(epoch) * 1000 - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > windowsNativeLlvmMaxRunMs) {
    throw new Error("Windows LLVM CI run deadline is expired or exceeds the 8-hour run bound");
  }
  return {remainingMs};
}

/** Keep every LLVM phase invocation on the same caller-supplied job-start deadline.
 * @param {string} epoch @param {"configure" | "build" | "audit" | "outputs"} phase @param {number} [nowMs] @param {string} [runEpoch] */
function windowsNativeLlvmCiDeadlineBudget(epoch, phase, nowMs = Date.now(), runEpoch = "") {
  if (!/^\d{10}$/.test(epoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 ||
      !["configure", "build", "audit", "outputs"].includes(phase)) {
    throw new Error("Windows LLVM CI needs a valid shared job-start deadline");
  }
  const remainingMs = Number(epoch) * 1000 - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > windowsNativeLlvmMaxJobMs) {
    throw new Error("Windows LLVM CI deadline is expired or exceeds the 60-minute job bound");
  }
  if (runEpoch) {
    const run = windowsNativeLlvmCiRunDeadlineBudget(runEpoch, nowMs);
    if (Number(epoch) > Number(runEpoch)) {throw new Error("Windows LLVM job deadline exceeds the original run deadline");}
    if (remainingMs > run.remainingMs) {throw new Error("Windows LLVM job deadline exceeds the remaining run limit");}
  }
  if (phase === "outputs") {
    if (remainingMs < 10 * 1000) {throw new Error("Insufficient shared job time remains to report Windows LLVM phase outputs");}
    return {timeoutMs: 0, remainingMs};
  }
  if (remainingMs < windowsNativeLlvmMinPhaseMs + windowsNativeLlvmPackReserveMs) {
    throw new Error("Insufficient shared job time remains for the Windows LLVM phase and final status reporting");
  }
  return {timeoutMs: Math.min(windowsNativeLlvmMaxChildMs, remainingMs - windowsNativeLlvmPackReserveMs), remainingMs};
}

/** @typedef {{suffix:string, runnerTemp:string, root:string, output:string, descriptor:string, deadlineRecord:string, result:string,
 * deadlineEpoch:string, jobId:string, runDeadlineEpoch:string, runDeadlineRecord:string}} WindowsNativeLlvmCiState */

/** Fixed same-run paths keep the builder's absolute receipts valid for each in-job phase call.
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {string} [architecture]
 * @returns {WindowsNativeLlvmCiState} */
function windowsNativeLlvmCiStatePaths(environment = process.env, platform = process.platform, architecture = process.arch) {
  const runnerTemp = environment["RUNNER_TEMP"] || "", run = environment["GITHUB_RUN_ID"] || "", attempt = environment["GITHUB_RUN_ATTEMPT"] || "";
  const runDeadlineEpoch = environment["MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH"] || "", jobId = environment["GITHUB_JOB"] || "";
  if (platform !== "linux" || architecture !== "x64" || !path.isAbsolute(runnerTemp) ||
      !/^[0-9]{1,20}$/.test(run) || !/^[1-9][0-9]{0,5}$/.test(attempt) ||
      environment["MOM_AOT_BUILD_ONLY"] !== "1" || environment["MOM_BUILD_JOBS"] !== ciBuildJobs(environment)) {
    throw new Error("Windows LLVM CI needs a Linux x64 runner, exact run attempt, build-only mode and a valid job count");
  }
  if (!runDeadlineEpoch || !/^[A-Za-z0-9_-]{1,64}$/.test(jobId)) {
    throw new Error("Windows LLVM CI needs its original run deadline and a safe bounded GITHUB_JOB identity");
  }
  const canonicalTemp = fs.realpathSync(runnerTemp), suffix = run + "-" + attempt,
    root = path.join(temp, "mom-native-windows-llvm21-" + suffix), output = path.join(root, "output"),
    descriptor = path.join(root, "descriptor.json");
  if (canonicalTemp !== path.resolve(runnerTemp) || !fs.lstatSync(canonicalTemp).isDirectory() ||
      fs.realpathSync(temp) !== temp || environment["MOM_NATIVE_WINDOWS_LLVM_DIR"] !== root ||
      path.dirname(root) !== temp || path.resolve(root) !== root) {
    throw new Error("Windows LLVM CI paths must match the exact run-attempt plan");
  }
  return {suffix, runnerTemp: canonicalTemp, root, output, descriptor, result: path.join(output, "result.json"),
    deadlineEpoch: environment["MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH"] || "", jobId, runDeadlineEpoch,
    deadlineRecord: path.join(root, "ci-job-deadline-" + jobId + ".json"), runDeadlineRecord: path.join(root, "ci-run-deadline.json")};
}

/** Bind each phase call to the run deadline recorded at configure and to its job's first deadline,
 * without changing public builder receipts. Returns null when the job record is absent and not created.
 * @param {ReturnType<typeof windowsNativeLlvmCiStatePaths>} state @param {boolean} [create] */
function windowsNativeLlvmCiDeadlineRecord(state, create = false) {
  const jobId = state.jobId, runDeadlineRecord = state.runDeadlineRecord;
  const readRecord = (/** @type {string} */ file, /** @type {string} */ label) => {
    const stat = windowsControllerRegularFile(file, label);
    if (stat.nlink !== 1 || stat.size > 1024) {throw new Error(label + " is aliased or invalid");}
    return windowsControllerReadJson(file, 1024, label);
  };
  let runRecord;
  if (!fs.existsSync(runDeadlineRecord)) {
    if (!create || fs.existsSync(state.output)) {throw new Error("Windows LLVM cross-job state is missing its original run deadline record");}
    runRecord = {version: 1, runAttempt: state.suffix, runDeadlineEpoch: state.runDeadlineEpoch};
    fs.writeFileSync(runDeadlineRecord, JSON.stringify(runRecord) + "\n", {flag: "wx", mode: 0o444});
  } else {
    runRecord = readRecord(runDeadlineRecord, "Windows LLVM run deadline record");
    if (runRecord.version !== 1 || runRecord.runAttempt !== state.suffix || runRecord.runDeadlineEpoch !== state.runDeadlineEpoch) {
      throw new Error("Windows LLVM cross-job phase attempted to replace or reset its original run deadline");
    }
  }
  const jobStat = fs.lstatSync(state.deadlineRecord, {throwIfNoEntry: false});
  if (!jobStat) {
    if (!create) {return null;}
    const record = {version: 1, runAttempt: state.suffix, job: jobId,
      deadlineEpoch: state.deadlineEpoch, runDeadlineEpoch: state.runDeadlineEpoch};
    fs.writeFileSync(state.deadlineRecord, JSON.stringify(record) + "\n", {flag: "wx", mode: 0o444});
    return record;
  }
  const record = readRecord(state.deadlineRecord, "Windows LLVM job deadline record");
  if (record.version !== 1 || record.runAttempt !== state.suffix || record.job !== jobId ||
      record.deadlineEpoch !== state.deadlineEpoch || record.runDeadlineEpoch !== state.runDeadlineEpoch) {
    throw new Error("Windows LLVM phase attempted to replace or reset its original job deadline");
  }
  return record;
}

/** Ensure workflow report files cannot redirect phase output or alias each other. @param {NodeJS.ProcessEnv} environment */
function windowsNativeLlvmCiReports(environment) {
  const files = [environment["GITHUB_OUTPUT"] || "", environment["GITHUB_STEP_SUMMARY"] || ""];
  if (files.some(file => !path.isAbsolute(file))) {throw new Error("Windows LLVM CI requires absolute workflow output files");}
  const stats = files.map((file, index) => {
    const stat = windowsControllerRegularFile(file, index === 0 ? "Windows LLVM GitHub output" : "Windows LLVM GitHub summary");
    if (stat.nlink !== 1) {throw new Error("Windows LLVM workflow output must be an unaliased regular file");}
    return {file, dev: stat.dev, ino: stat.ino};
  });
  if (path.resolve(files[0] || "") === path.resolve(files[1] || "")) {throw new Error("Windows LLVM workflow reports must be distinct files");}
  return /** @type {[{file:string,dev:number,ino:number},{file:string,dev:number,ino:number}]} */ (stats);
}

/** Reject a changed workflow file before writing outputs after a long build. @param {{file:string,dev:number,ino:number}} saved */
function verifyWindowsNativeLlvmCiReport(saved) {
  const stat = windowsControllerRegularFile(saved.file, "Windows LLVM workflow report");
  if (stat.nlink !== 1 || stat.dev !== saved.dev || stat.ino !== saved.ino) {
    throw new Error("Windows LLVM workflow report changed during its phase");
  }
}

/** @param {string} root @param {string} child */
function windowsNativeLlvmCiContains(root, child) {
  const relative = path.relative(root, child);
  return relative === "" || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep));
}

/** Keep workflow report files outside the native inputs, including directory roots. @param {ReturnType<typeof windowsNativeLlvmCiReports>} reports @param {string[]} inputs */
function windowsNativeLlvmCiRejectReportInputOverlap(reports, inputs) {
  for (const report of reports) {
    for (const input of inputs) {
      if (typeof input !== "string" || !path.isAbsolute(input) || /[\r\n\0]/.test(input)) {continue;}
      const resolved = path.resolve(input);
      if (windowsNativeLlvmCiContains(resolved, report.file) || windowsNativeLlvmCiContains(report.file, resolved)) {
        throw new Error("Windows LLVM workflow reports must not overlap pinned inputs");
      }
      const stat = fs.lstatSync(resolved, {throwIfNoEntry: false});
      if (stat && stat.dev === report.dev && stat.ino === report.ino) {
        throw new Error("Windows LLVM workflow reports must not alias pinned inputs");
      }
    }
  }
}

/** @param {ReturnType<typeof import("./build-native-windows-controller").validateWindowsLlvmSdkConfig>} descriptor */
function windowsNativeLlvmCiDescriptorRoots(descriptor) {
  return [repo, descriptor.windowsSdk.sourceRoot, descriptor.windowsSdk.overlay, ...descriptor.windowsSdk.includeRoots,
    ...descriptor.windowsSdk.libraryRoots, path.dirname(descriptor.tools.clangCl), path.dirname(descriptor.tools.lldLink),
    path.dirname(descriptor.tools.llvmNm), descriptor.tools.clangResourceInclude];
}

/** Validate one explicit phase, shared deadline and its immutable same-run state before mutation.
 * Configure args are descriptor/path-SHA, source-archive/path-SHA, CMake/path-SHA, Ninja/path-SHA.
 * @param {string[]} args @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform]
 * @param {string} [architecture] @param {number} [nowMs] */
function windowsNativeLlvmCiPreflight(args, environment = process.env, platform = process.platform, architecture = process.arch, nowMs = Date.now()) {
  const phaseValue = args[0] || "";
  if (!["configure", "build", "audit"].includes(phaseValue) || args.length !== (phaseValue === "configure" ? 9 : phaseValue === "build" ? 2 : 1)) {
    throw new Error("Windows LLVM CI needs configure plus eight pinned inputs, build plus one phase, or audit only");
  }
  const phase = /** @type {"configure" | "build" | "audit"} */ (phaseValue),
    {validateWindowsLlvmSdkConfig} = require("./build-native-windows-controller"),
    {windowsLlvmSdkPlan, buildPhases} = require("./build-native-windows-llvm21-sdk");
  const state = windowsNativeLlvmCiStatePaths(environment, platform, architecture), reports = windowsNativeLlvmCiReports(environment),
    budget = windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, phase, nowMs, state.runDeadlineEpoch);
  if (reports.some(report => windowsNativeLlvmCiContains(state.root, report.file))) {
    throw new Error("Windows LLVM CI state and workflow reports must not overlap");
  }
  if (phase !== "configure") {windowsNativeLlvmCiDeadlineRecord(state);}
  /** @param {string} file @param {string} label @param {number} maxBytes */
  const pathStat = (file, label, maxBytes) => {
    const stat = windowsControllerRegularFile(file, label);
    if (stat.nlink !== 1 || stat.size < 1 || stat.size > maxBytes) {throw new Error(label + " is aliased, empty or exceeds its size bound");}
    return stat;
  };
  if (phase === "configure") {
    if (args.length !== 9 || [state.root, state.output, state.descriptor].some(file => fs.existsSync(file))) {
      throw new Error("Windows LLVM configure needs eight pinned inputs and fresh same-run output paths");
    }
    const [descriptorSource = "", descriptorSha256 = "", archive = "", archiveSha256 = "", cmake = "", cmakeSha256 = "", ninja = "", ninjaSha256 = ""] = args.slice(1);
    const pinnedSha = (/** @type {string} */ value, /** @type {string} */ label) => {
      if (!/^[a-f0-9]{64}$/.test(value || "")) {throw new Error("Windows LLVM " + label + " needs a lowercase SHA-256 pin");}
    };
    const regularInput = (/** @type {string} */ file, /** @type {string} */ label, /** @type {number} */ maximum,
      /** @type {boolean} */ underRunnerTemp = false) => {
      if (!path.isAbsolute(file) || /[\r\n\0]/.test(file) || path.resolve(fs.realpathSync(file)).toLowerCase() !== path.resolve(file).toLowerCase() ||
          underRunnerTemp && !windowsNativeLlvmCiContains(state.runnerTemp, file)) {throw new Error(label + " must be a canonical same-run input");}
      return pathStat(file, label, maximum);
    };
    pinnedSha(descriptorSha256 || "", "descriptor"); pinnedSha(archiveSha256 || "", "source archive");
    pinnedSha(cmakeSha256 || "", "CMake"); pinnedSha(ninjaSha256 || "", "Ninja");
    if (archiveSha256 !== llvmArchiveSha256 || path.basename(archive || "") !== llvmArchiveName) {
      throw new Error("Windows LLVM configure requires the exact locked LLVM 21 source archive");
    }
    const descriptorStat = regularInput(descriptorSource || "", "Windows LLVM descriptor", 4 << 20, true),
      archiveStat = regularInput(archive || "", "Pinned LLVM 21 source archive", 2 * 1024 * 1024 * 1024, true),
      cmakeStat = regularInput(cmake || "", "Pinned CMake", 128 << 20), ninjaStat = regularInput(ninja || "", "Pinned Ninja", 128 << 20),
      descriptorBytes = fs.readFileSync(descriptorSource || "");
    const descriptorHash = createHash("sha256").update(descriptorBytes).digest("hex");
    if (descriptorBytes.length !== descriptorStat.size || descriptorHash !== descriptorSha256) {
      throw new Error("Windows LLVM descriptor differs from its external SHA-256 pin");
    }
    windowsNativeLlvmCiRejectReportInputOverlap(reports, [repo, descriptorSource, archive || "", cmake || "", ninja || ""]);
    let descriptorValue;
    try {descriptorValue = JSON.parse(descriptorBytes.toString("utf8"));} catch (error) {
      throw new Error("Pinned Windows LLVM descriptor is invalid JSON", {cause: error});
    }
    const descriptor = validateWindowsLlvmSdkConfig(descriptorValue), tools = {cmake, ninja},
      plan = windowsLlvmSdkPlan(descriptor, archive || "", state.output, tools, 2);
    windowsNativeLlvmCiRejectReportInputOverlap(reports, windowsNativeLlvmCiDescriptorRoots(descriptor));
    const overlapsOutput = [repo, path.dirname(archive || ""), descriptor.windowsSdk.sourceRoot, descriptor.windowsSdk.overlay,
      ...descriptor.windowsSdk.includeRoots, ...descriptor.windowsSdk.libraryRoots, path.dirname(descriptor.tools.clangCl),
      path.dirname(descriptor.tools.lldLink), path.dirname(descriptor.tools.llvmNm), descriptor.tools.clangResourceInclude]
      .some(input => path.resolve(input) === state.root || path.resolve(input).startsWith(state.root + path.sep) ||
        state.root.startsWith(path.resolve(input) + path.sep));
    if (overlapsOutput || descriptorSource === archive || descriptorSource === cmake || descriptorSource === ninja ||
        archive === cmake || archive === ninja || cmake === ninja ||
        fileHash(archive || "") !== archiveSha256 || fileHash(cmake || "") !== cmakeSha256 || fileHash(ninja || "") !== ninjaSha256 ||
        fileHash(descriptorSource || "") !== descriptorSha256) {
      throw new Error("Windows LLVM configure input identity, pin or output isolation changed during preflight");
    }
    const afterBudget = windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "configure", Date.now(), state.runDeadlineEpoch);
    return {phase, state, reports, descriptorSource: descriptorSource || "", descriptorSha256: descriptorSha256 || "",
      descriptorBytes, archive: archive || "", archiveSha256: archiveSha256 || "", tools, toolHashes: {cmake: cmakeSha256 || "", ninja: ninjaSha256 || ""},
      plan, timeoutMs: Math.min(budget.timeoutMs, afterBudget.timeoutMs), sourceStats: {descriptor: descriptorStat, archive: archiveStat,
        cmake: cmakeStat, ninja: ninjaStat}};
  }

  if (args.length !== (phase === "build" ? 2 : 1)) {throw new Error("Windows LLVM phase needs exactly one phase name, or no audit argument");}
  const rootStat = fs.lstatSync(state.root, {throwIfNoEntry: false}), outputStat = fs.lstatSync(state.output, {throwIfNoEntry: false});
  if (!rootStat?.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(state.root) !== state.root ||
      !outputStat?.isDirectory() || outputStat.isSymbolicLink() || fs.realpathSync(state.output) !== state.output) {
    throw new Error("Windows LLVM phase requires its canonical configured same-run output");
  }
  const receiptStat = pathStat(state.result, "Windows LLVM configure receipt", 64 << 20),
    receipt = windowsControllerReadJson(state.result, 64 << 20, "Windows LLVM configure receipt"), descriptorStat = pathStat(state.descriptor, "Windows LLVM descriptor", 4 << 20),
    descriptorHash = fileHash(state.descriptor), plan = receipt.plan;
  windowsNativeLlvmCiRejectReportInputOverlap(reports, [repo, state.descriptor, plan?.archive || "",
    plan?.tools?.cmake || "", plan?.tools?.ninja || ""]);
  if (!plan || plan.output !== state.output || plan.descriptorFile !== state.descriptor || plan.jobs !== Number(ciBuildJobs(environment)) || plan.reuse !== undefined ||
      receipt.role !== "windows-llvm21-host-sdk" || receipt.stage !== "configured" || receipt.buildOnly !== true || receipt.compiled !== false ||
      receipt.executed !== false || receipt.publicationQualified !== false || receipt.fingerprints?.[state.descriptor] !== descriptorHash ||
      descriptorStat.nlink !== 1 || receiptStat.nlink !== 1) {
    throw new Error("Windows LLVM phase receipt does not match the fixed same-run configure plan");
  }
  const descriptor = validateWindowsLlvmSdkConfig(windowsControllerReadJson(state.descriptor, 4 << 20, "Windows LLVM descriptor"));
  windowsNativeLlvmCiRejectReportInputOverlap(reports, windowsNativeLlvmCiDescriptorRoots(descriptor));
  windowsLlvmSdkPlan(descriptor, plan.archive, state.output, plan.tools, 2);
  if (phase === "build") {
    const selected = args[1] || "";
    if (!buildPhases.some(row => row.name === selected)) {throw new Error("Windows LLVM build phase must be one maintained aggregate");}
  } else if ([path.join(state.output, "admitted.json"), path.join(state.output, "admitted-objects.json")].some(file => fs.existsSync(file))) {
    throw new Error("Windows LLVM audit refuses to restamp existing admission receipts");
  }
  const beforeReceiptSha256 = fileHash(state.result), budgetAfter = windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, phase, Date.now(), state.runDeadlineEpoch);
  return {phase, state, reports, receipt, descriptorHash, beforeReceiptSha256, timeoutMs: Math.min(budget.timeoutMs, budgetAfter.timeoutMs)};
}

/** Invoke one original public Windows LLVM builder phase; never restamp its receipts or ship the CMake tree.
 * @param {string[]} args @param {NodeJS.ProcessEnv} [environment] */
function buildWindowsNativeLlvmCi(args, environment = process.env) {
  const input = windowsNativeLlvmCiPreflight(args, environment), {phase, state} = input,
    builder = "scripts/build-native-windows-llvm21-sdk.js", buildEnvironment = {...environment, MOM_AOT_BUILD_ONLY: "1"};
  if (phase === "configure") {
    const descriptorBytes = input.descriptorBytes;
    if (!descriptorBytes) {throw new Error("Windows LLVM configure lost its prevalidated descriptor bytes");}
    fs.mkdirSync(state.root, {mode: 0o700});
    windowsNativeLlvmCiDeadlineRecord(input.state, true);
    fs.writeFileSync(state.descriptor, descriptorBytes, {flag: "wx", mode: 0o444});
    const stat = windowsControllerRegularFile(state.descriptor, "Staged Windows LLVM descriptor");
    if (stat.nlink !== 1 || fileHash(state.descriptor) !== input.descriptorSha256 ||
        fileHash(input.descriptorSource) !== input.descriptorSha256 || fileHash(input.archive) !== input.archiveSha256 ||
        fileHash(input.tools.cmake) !== input.toolHashes.cmake || fileHash(input.tools.ninja) !== input.toolHashes.ninja) {
      throw new Error("Windows LLVM configure inputs changed before the public builder started");
    }
    const childTimeoutMs = Math.min(input.timeoutMs,
      windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, phase, Date.now(), state.runDeadlineEpoch).timeoutMs);
    runNodeProcessGroup(builder, ["configure", state.descriptor, input.archive, state.output,
      input.tools.cmake, input.tools.ninja, ciBuildJobs(buildEnvironment)], buildEnvironment, childTimeoutMs);
    const receipt = windowsControllerReadJson(state.result, 64 << 20, "Windows LLVM configure receipt");
    windowsNativeLlvmCiDeadlineRecord(state);
    if (receipt.role !== "windows-llvm21-host-sdk" || receipt.stage !== "configured" || receipt.plan?.output !== state.output ||
        receipt.plan?.descriptorFile !== state.descriptor || receipt.plan?.jobs !== Number(ciBuildJobs(buildEnvironment)) || receipt.compiled !== false ||
        receipt.buildOnly !== true || receipt.executed !== false || receipt.publicationQualified !== false ||
        fileHash(state.descriptor) !== input.descriptorSha256 || fileHash(input.descriptorSource) !== input.descriptorSha256 ||
        fileHash(input.archive) !== input.archiveSha256 || fileHash(input.tools.cmake) !== input.toolHashes.cmake ||
        fileHash(input.tools.ninja) !== input.toolHashes.ninja) {
      throw new Error("Windows LLVM configure result lost its externally pinned inputs or build-only status");
    }
  } else {
    const childTimeoutMs = Math.min(input.timeoutMs,
      windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, phase, Date.now(), state.runDeadlineEpoch).timeoutMs);
    windowsNativeLlvmCiDeadlineRecord(state, true);
    runNodeProcessGroup(builder, phase === "build" ? ["build", state.output, args[1] || ""] : ["audit", state.output],
      buildEnvironment, childTimeoutMs);
    windowsNativeLlvmCiDeadlineRecord(state);
    if (fileHash(state.result) !== input.beforeReceiptSha256 || fileHash(state.descriptor) !== input.descriptorHash) {
      throw new Error("Windows LLVM builder changed the original configure receipt or descriptor");
    }
  }
  const values = /** @type {Record<string,string>} */ ({windows_llvm_phase: phase === "build" ? "build-" + (args[1] || "") : phase,
    windows_llvm_state_dir: state.output, windows_llvm_result_sha256: fileHash(state.result),
    windows_llvm_build_only: "true", windows_llvm_executed: "false", windows_llvm_publication_qualified: "false"});
  if (phase === "build") {
    const checkpoint = windowsControllerReadJson(path.join(state.output, "build-state.json"), 64 << 20, "Windows LLVM build checkpoint"),
      proof = checkpoint.phaseProofs?.at(-1), proofPath = proof?.path;
    if (proof?.name !== args[1] || typeof proofPath !== "string" || !path.isAbsolute(proofPath) ||
        windowsControllerReadJson(proofPath, 16 << 20, "Windows LLVM phase proof").pass !== true) {
      throw new Error("Windows LLVM producer did not retain the selected successful phase proof");
    }
    values["windows_llvm_phase_proof_sha256"] = fileHash(proofPath);
    values["windows_llvm_checkpoint_sha256"] = fileHash(path.join(state.output, "build-state.json"));
  } else if (phase === "audit") {
    const admission = windowsControllerReadJson(path.join(state.output, "admitted.json"), 64 << 20, "Windows LLVM admission receipt");
    if (admission.role !== "windows-llvm21-static-host-admission" || admission.buildOnly !== true || admission.compiled !== true ||
        admission.executed !== false || admission.publicationQualified !== false) {
      throw new Error("Windows LLVM audit output is not a build-only admission receipt");
    }
    values["windows_llvm_admission_sha256"] = fileHash(path.join(state.output, "admitted.json"));
    values["windows_llvm_objects_sha256"] = fileHash(path.join(state.output, "admitted-objects.json"));
  }
  windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "outputs", Date.now(), state.runDeadlineEpoch);
  for (const report of input.reports) {verifyWindowsNativeLlvmCiReport(report);}
  appendWorkflowOutputs(environment, values);
  const summary = environment["GITHUB_STEP_SUMMARY"] || "";
  fs.appendFileSync(summary, `### Windows LLVM 21 ${values["windows_llvm_phase"]}\n` +
    `- Same-run output: ${state.output}; original configure receipt SHA-256: ${values["windows_llvm_result_sha256"]}.\n` +
    "- Build-only: true; executed: false; publication-qualified: false. No SDK build-tree artifact was exported.\n", "utf8");
  console.log(`[native-ci] Completed the public Windows LLVM 21 ${values["windows_llvm_phase"]} phase; no SDK artifact was exported.`);
  return values;
}

/** Fresh native Windows controller-support outputs; the Hermes source tar is same-run Linux input.
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {"io" | "hermes"} [kind] */
function windowsControllerStatePaths(environment = process.env, platform = process.platform, kind = "io") {
  const runnerTemp = environment["RUNNER_TEMP"] || "", run = environment["GITHUB_RUN_ID"] || "", attempt = environment["GITHUB_RUN_ATTEMPT"] || "";
  if (platform !== "win32" || !["io", "hermes"].includes(kind) || !path.isAbsolute(runnerTemp) ||
      !/^[0-9]{1,20}$/.test(run) || !/^[1-9][0-9]{0,5}$/.test(attempt)) {
    throw new Error("Windows controller-support CI needs a Windows runner and exact run-attempt paths");
  }
  const canonicalTemp = fs.realpathSync(runnerTemp), suffix = run + "-" + attempt,
    workName = "mom-native-windows-controller-" + kind + "-" + suffix, workRoot = path.join(canonicalTemp, workName),
    outputName = "native-windows-controller-" + kind + "-" + suffix,
    outputRoot = path.join(canonicalTemp, outputName), archive = path.join(canonicalTemp, outputName + ".tar"),
    expectedEnvironment = "MOM_NATIVE_WINDOWS_CONTROLLER_" + kind.toUpperCase() + "_DIR";
  const samePath = (/** @type {string} */ left, /** @type {string} */ right) => path.resolve(left).toLowerCase() === path.resolve(right).toLowerCase();
  if (!samePath(canonicalTemp, path.resolve(runnerTemp)) || !fs.lstatSync(canonicalTemp).isDirectory() ||
      !samePath(environment[expectedEnvironment] || "", workRoot) || path.dirname(workRoot) !== canonicalTemp ||
      path.dirname(outputRoot) !== canonicalTemp) {
    throw new Error("Windows controller-support paths must exactly match this run attempt");
  }
  const incomingRoot = path.join(canonicalTemp, "incoming"), outputFile = environment["GITHUB_OUTPUT"] || "",
    summaryFile = environment["GITHUB_STEP_SUMMARY"] || "";
  return {kind, suffix, runnerTemp: canonicalTemp, workName, workRoot, outputName, outputRoot, archive, incomingRoot,
    ioSource: path.join(workRoot, "io-source"), opensslSource: path.join(workRoot, "openssl-source"),
    ioSdk: path.join(workRoot, "io-sdk"), opensslSdk: path.join(workRoot, "openssl-sdk"), opensslToolsFile: path.join(workRoot, "openssl-tools.json"),
    hermesSourceTar: path.join(incomingRoot, "native-hermes-state-" + suffix + "-sources.tar"),
    hermesExtractRoot: path.join(workRoot, "source-input"), hermesSourceName: "mom-native-hermes-prerequisites-" + suffix,
    hermesSource: path.join(workRoot, "source-input", "mom-native-hermes-prerequisites-" + suffix, "roots/hermes"),
    hermesSdk: path.join(workRoot, "hermes-sdk"), deadlineEpoch: environment["MOM_NATIVE_WINDOWS_CONTROLLER_DEADLINE_EPOCH"] || "",
    outputFile, summaryFile};
}

/** @param {string} epoch @param {"fetch" | "extract" | "build" | "package"} phase @param {number} [nowMs] */
function windowsControllerDeadlineBudget(epoch, phase, nowMs = Date.now()) {
  if (!/^\d{10}$/.test(epoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 ||
      !["fetch", "extract", "build", "package"].includes(phase)) {
    throw new Error("Windows controller-support stage needs a valid bounded job-start deadline");
  }
  const remainingMs = Number(epoch) * 1000 - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > 60 * 60 * 1000) {
    throw new Error("Windows controller-support job deadline is expired or exceeds the 60-minute job bound");
  }
  const reserveMs = phase === "package" ? 30 * 1000 : 2 * 60 * 1000,
    maximumMs = {fetch: 10, extract: 5, build: 45, package: 5}[phase] * 60 * 1000;
  if (remainingMs <= reserveMs + 30 * 1000) {throw new Error("Insufficient aggregate job time remains for Windows controller-support " + phase);}
  return {timeoutMs: Math.min(maximumMs, remainingMs - reserveMs), remainingMs};
}

/** @typedef {{ar:string,cc:string,link:string,make:string,perl:string,rc:string,tar:string}} WindowsControllerOpenSslTools */
/** @typedef {{cc:string,cmake:string,ninja:string,openssl:WindowsControllerOpenSslTools}} WindowsControllerIoTools */
/** @typedef {{cc:string,cmake:string,cxx:string,git:string,ninja:string}} WindowsControllerHermesTools */

/** Parse a role-specific explicit tool table before any source acquisition or producer runs.
 * @param {NodeJS.ProcessEnv} environment @param {"io" | "hermes"} kind */
function windowsControllerTools(environment, kind) {
  const text = environment["MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS"] || "";
  if (!text || text.length > 16384) {throw new Error("Windows controller-support tools must be an explicit bounded JSON table");}
  let parsed;
  try {parsed = JSON.parse(text);} catch {throw new Error("Windows controller-support tool table is invalid JSON");}
  const keys = kind === "io" ? ["cc", "cmake", "ninja", "openssl"] : ["cc", "cmake", "cxx", "git", "ninja"];
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed) || Object.keys(parsed).sort().join(",") !== keys.sort().join(",")) {
    throw new Error("Windows controller-support tool table has an unexpected role or key set");
  }
  if (kind === "io" && (!parsed.openssl || typeof parsed.openssl !== "object" || Array.isArray(parsed.openssl) ||
      Object.keys(parsed.openssl).sort().join(",") !== "ar,cc,link,make,perl,rc,tar")) {
    throw new Error("Windows controller I/O/OpenSSL tools must include the exact OpenSSL Windows tool set");
  }
  const files = kind === "io" ? [parsed.cc, parsed.cmake, parsed.ninja, ...Object.values(parsed.openssl)] : Object.values(parsed);
  for (const file of files) {
    if (typeof file !== "string" || file.length > 4096 || !path.isAbsolute(file) || /[\r\n\0]/.test(file)) {
      throw new Error("Windows controller-support tool path must be an explicit absolute path");
    }
    const stat = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || path.resolve(fs.realpathSync(file)).toLowerCase() !== path.resolve(file).toLowerCase()) {
      throw new Error("Windows controller-support tool is missing, redirected or non-regular: " + file);
    }
  }
  return /** @type {WindowsControllerIoTools | WindowsControllerHermesTools} */ (parsed);
}

/** Exact public producer argv; paths stay under the fresh same-run output root.
 * @param {ReturnType<typeof windowsControllerStatePaths>} state @param {WindowsControllerIoTools} tools */
function windowsControllerIoBuildCommands(state, tools) {
  if (state.kind !== "io" || !tools?.openssl) {throw new Error("Windows I/O/OpenSSL command needs its exact role and tools");}
  return [
    {script: "scripts/build-native-io.js", args: [path.join(state.ioSource, "libuv-v1.51.0.tar.gz"),
      path.join(state.ioSource, "zlib-1.3.1.tar.gz"), state.ioSdk, tools.cmake, tools.cc, tools.ninja, "2"]},
    {script: "scripts/build-native-openssl.js", args: [path.join(state.opensslSource, "openssl-" + opensslPin.version + ".tar.gz"),
      state.opensslSdk, state.opensslToolsFile]},
  ];
}

/** @param {ReturnType<typeof windowsControllerStatePaths>} state @param {WindowsControllerHermesTools} tools */
function windowsControllerHermesBuildCommand(state, tools) {
  if (state.kind !== "hermes" || !state.hermesSource || !tools) {throw new Error("Windows Hermes command needs its exact role and tools");}
  return {script: "scripts/build-native-hermes.js", args: [state.hermesSource, state.hermesSdk,
    tools.git, tools.cmake, tools.cc, tools.cxx, tools.ninja, "2"]};
}

/** @param {string} file @param {string} label */
function windowsControllerRegularFile(file, label) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!path.isAbsolute(file) || !stat?.isFile() || stat.isSymbolicLink() ||
      path.resolve(fs.realpathSync(file)).toLowerCase() !== path.resolve(file).toLowerCase()) {
    throw new Error(label + " must be a canonical regular file");
  }
  return stat;
}

/** @param {string} compilerFile @returns {WindowsIntelRuntimeEntry[]} */
function windowsControllerIntelRuntimeEntries(compilerFile) {
  const normalized = typeof compilerFile === "string" ? compilerFile.replaceAll("\\", "/").toLowerCase() : "";
  const catalog = windowsIntelRuntimeCatalog;
  if (!catalog || typeof catalog !== "object" || Array.isArray(catalog) ||
      Object.keys(catalog).sort().join(",") !== "compiler,libraries,licenses,version" || !catalog.compiler ||
      typeof catalog.compiler !== "object" || Array.isArray(catalog.compiler) ||
      Object.keys(catalog.compiler).sort().join(",") !== "sha256,version" || catalog.version !== 1 ||
      !/^\d{4}\.\d+(?:\.\d+)?$/.test(catalog.compiler.version) || !/^[a-f0-9]{64}$/.test(catalog.compiler.sha256) ||
      !Array.isArray(catalog.libraries) || catalog.libraries.length !== 5 || !Array.isArray(catalog.licenses) || catalog.licenses.length !== 3) {
    throw new Error("Intel runtime pin catalog has an unsupported shape or version");
  }
  if (typeof compilerFile !== "string" || !path.isAbsolute(compilerFile) ||
      !normalized.endsWith("/compiler/" + catalog.compiler.version + "/bin/icx-cl.exe")) {
    throw new Error("Intel runtime export requires the selected catalog icx-cl compiler path");
  }
  const compilerRoot = path.dirname(path.dirname(compilerFile));
  const entries = /** @type {WindowsIntelRuntimeEntry[]} */ ([
    ...catalog.libraries.map(entry => ({...entry, kind: "library"})),
    ...catalog.licenses.map(entry => ({...entry, kind: "license"})),
  ]);
  const destinations = new Set();
  for (const entry of entries) {
    if (!safeRelative(entry.source) || !safeRelative(entry.path) || !Number.isSafeInteger(entry.bytes) || entry.bytes < 1 ||
        !/^[a-f0-9]{64}$/.test(entry.sha256) || destinations.has(entry.path.toLowerCase()) ||
        entry.kind === "library" && (Object.keys(entry).sort().join(",") !== "bytes,kind,name,path,sha256,source" ||
          !/^[a-z0-9_.-]+\.lib$/i.test(entry.name) || entry.source !== "lib/" + entry.name || entry.path !== "intel-runtime/lib/" + entry.name) ||
        entry.kind === "license" && (Object.keys(entry).sort().join(",") !== "bytes,kind,path,sha256,source" ||
          entry.path !== "intel-runtime/licenses/" + entry.source)) {
      throw new Error("Intel runtime pin catalog contains an invalid or duplicate file row");
    }
    destinations.add(entry.path.toLowerCase());
  }
  return entries.map(entry => ({...entry, source: path.join(compilerRoot, ...entry.source.split("/"))}));
}

/** @param {string} compilerFile @param {string} outputRoot @param {WindowsControllerFileLedger} ledger */
function copyWindowsControllerIntelRuntime(compilerFile, outputRoot, ledger) {
  const compilerStat = windowsControllerRegularFile(compilerFile, "Intel 2026.2 compiler"), compilerSha256 = fileHash(compilerFile),
    entries = windowsControllerIntelRuntimeEntries(compilerFile);
  if (compilerStat.size < 1 || compilerSha256 !== windowsIntelRuntimeCatalog.compiler.sha256) {
    throw new Error("Intel compiler content differs from the selected 2026.2 runtime pins");
  }
  /** @type {Array<{name:string,path:string,bytes:number,sha256:string}>} */ const libraries = [];
  /** @type {Array<{path:string,bytes:number,sha256:string}>} */ const licenses = [];
  for (const entry of entries) {
    const stat = windowsControllerRegularFile(entry.source, "Intel runtime " + (entry.kind === "library" ? entry.name : entry.path)), sha256 = fileHash(entry.source);
    if (stat.size !== entry.bytes || sha256 !== entry.sha256 || entry.kind === "library" && !isStaticArchive(entry.source)) {
      throw new Error("Intel runtime input differs from its selected 2026.2 pin: " + entry.path);
    }
    copyWindowsControllerPath(entry.source, path.join(outputRoot, ...entry.path.split("/")), entry.path, ledger);
    const row = entry.kind === "library" ? {name: entry.name, path: entry.path, bytes: stat.size, sha256} :
      {path: entry.path, bytes: stat.size, sha256};
    (entry.kind === "library" ? libraries : licenses).push(row);
  }
  const compiler = {path: compilerFile, sha256: compilerSha256}, sourceFiles = [{path: compilerFile, sha256: compilerSha256},
    ...entries.map(entry => ({path: entry.source, sha256: entry.sha256}))];
  if (sourceFiles.some(row => fileHash(row.path) !== row.sha256)) {throw new Error("Intel runtime source changed during handoff copy");}
  return {record: {compiler, libraries, licenses}, sourceFiles};
}

/** @param {string} file @param {number} maximum @param {string} label */
function windowsControllerReadJson(file, maximum, label) {
  const stat = windowsControllerRegularFile(file, label);
  if (stat.size > maximum) {throw new Error(label + " exceeds its size bound");}
  return readBuildJson(file);
}

/** @param {unknown} name @returns {name is string} */
function windowsHermesSourceName(name) {
  return typeof name === "string" && name.length <= 160 && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name) && !/[. ]$/.test(name) &&
    !/^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)/i.test(name);
}

/** @param {string} value */
function windowsHermesHasControl(value) {
  return Array.from(value).some(char => char.charCodeAt(0) < 0x20 || char.charCodeAt(0) === 0x7f);
}

/** @param {unknown} value @returns {value is string} */
function canonicalHermesOriginalRoot(value) {
  return typeof value === "string" && value.length <= 4096 && path.posix.isAbsolute(value) && path.posix.normalize(value) === value &&
    path.posix.resolve(value) === value && !windowsHermesHasControl(value) && !value.includes("\\") && value !== "/" &&
    path.posix.basename(value) === "hermes" && path.posix.basename(path.posix.dirname(value)) === "roots";
}

/** The original WSL fetch receipts bind source paths but are never rewritten for transport.
 * @param {{hermesExtractRoot:string,hermesSourceName:string}} input @param {string} [originalRoot] */
function readWindowsHermesSource(input, originalRoot) {
  if (!input || Object.keys(input).sort().join(",") !== "hermesExtractRoot,hermesSourceName" || !windowsHermesSourceName(input.hermesSourceName) ||
      typeof input.hermesExtractRoot !== "string" || !path.isAbsolute(input.hermesExtractRoot) || path.resolve(input.hermesExtractRoot) !== input.hermesExtractRoot ||
      windowsHermesHasControl(input.hermesExtractRoot)) {
    throw new Error("Windows Hermes source reader needs its exact canonical extracted root and safe archive root name");
  }
  const expectedOriginalRoot = originalRoot === undefined ? "/tmp/" + input.hermesSourceName + "/roots/hermes" : originalRoot;
  if (!canonicalHermesOriginalRoot(expectedOriginalRoot)) {throw new Error("Hermes source original root must be a canonical POSIX roots/hermes path");}
  const extractStat = fs.lstatSync(input.hermesExtractRoot, {throwIfNoEntry: false}), parent = path.dirname(input.hermesExtractRoot),
    parentStat = fs.lstatSync(parent, {throwIfNoEntry: false});
  if (!parentStat?.isDirectory() || parentStat.isSymbolicLink() || fs.realpathSync(parent).toLowerCase() !== parent.toLowerCase() ||
      !extractStat?.isDirectory() || extractStat.isSymbolicLink() || fs.realpathSync(input.hermesExtractRoot).toLowerCase() !== input.hermesExtractRoot.toLowerCase()) {
    throw new Error("Windows Hermes extracted source root or its parent is missing or redirected");
  }
  const source = path.join(input.hermesExtractRoot, input.hermesSourceName), sourceStat = fs.lstatSync(source, {throwIfNoEntry: false});
  if (!sourceStat?.isDirectory() || sourceStat.isSymbolicLink() || fs.realpathSync(source).toLowerCase() !== source.toLowerCase()) {
    throw new Error("Windows Hermes archive root is missing or redirected");
  }
  const rootsFile = path.join(source, "roots.json"), fetchFile = path.join(source, "fetch.json"),
    roots = windowsControllerReadJson(rootsFile, 1 << 20, "Hermes source roots receipt"),
    fetch = windowsControllerReadJson(fetchFile, 4 << 20, "Hermes source fetch receipt"),
    recipe = sourceRecipe("hermes"), expectedRecipeSha = jsonHash(JSON.stringify(recipe));
  windowsControllerRegularFile(path.join(source, "roots/hermes/config"), "Extracted Hermes bare-source configuration");
  if (!hasExactKeys(roots, ["hermes"]) || roots.hermes !== expectedOriginalRoot || fetch.version !== 1 || fetch.target !== "hermes" ||
      fetch.status !== "PASS" || fetch.networked !== true || fetch.buildOnly !== true || fetch.compiled !== false || fetch.publicationQualified !== false ||
      !hasExactKeys(fetch.git, ["path", "sha256"]) || !path.posix.isAbsolute(fetch.git.path) || !/^[a-f0-9]{64}$/.test(fetch.git.sha256 || "") ||
      fetch.fetcherSha256 !== fileHash(path.join(repo, "scripts/fetch-native-sdk-sources.js")) ||
      !hasExactKeys(fetch.recipe, ["sha256", "files"]) || fetch.recipe.sha256 !== expectedRecipeSha || !isDeepStrictEqual(fetch.recipe.files, recipe.graph) ||
      !hasExactKeys(fetch.components, ["hermes"]) || fetch.components.hermes?.origin !== origins.hermes ||
      fetch.components.hermes?.revision !== hermesPins.revision || fetch.components.hermes?.root !== expectedOriginalRoot ||
      fetch.components.hermes?.gitDirectory !== "bare" || fetch.components.hermes?.shallow !== true || fetch.components.hermes?.partialClone !== false) {
    throw new Error("Hermes source receipts do not bind the original pinned bare Git root and current source recipe");
  }
  return {source: path.join(source, "roots/hermes"), rootsFile, fetchFile, fetch, fetchSha256: fileHash(fetchFile), rootsSha256: fileHash(rootsFile)};
}

/** The archive comes from the original Linux Hermes source cohort; do not fetch Git on Windows.
 * @param {Pick<ReturnType<typeof windowsControllerStatePaths>, "kind" | "hermesExtractRoot" | "hermesSourceTar" | "hermesSourceName">} state
 * @param {string} tarTool @param {number} timeoutMs @param {string} [originalRoot] */
function extractWindowsHermesSource(state, tarTool, timeoutMs, originalRoot) {
  if (!state || state.kind !== "hermes" || !windowsHermesSourceName(state.hermesSourceName) ||
      typeof state.hermesExtractRoot !== "string" || !path.isAbsolute(state.hermesExtractRoot) ||
      path.resolve(state.hermesExtractRoot) !== state.hermesExtractRoot || windowsHermesHasControl(state.hermesExtractRoot) ||
      typeof state.hermesSourceTar !== "string" || !path.isAbsolute(state.hermesSourceTar) ||
      typeof tarTool !== "string" || !path.isAbsolute(tarTool)) {
    throw new Error("Windows Hermes source extraction needs its exact role, safe root names and absolute paths");
  }
  if (originalRoot !== undefined && !canonicalHermesOriginalRoot(originalRoot)) {
    throw new Error("Hermes source original root must be a canonical POSIX roots/hermes path");
  }
  const outputParent = path.dirname(state.hermesExtractRoot), outputParentStat = fs.lstatSync(outputParent, {throwIfNoEntry: false}),
    outputStat = fs.lstatSync(state.hermesExtractRoot, {throwIfNoEntry: false});
  if (!outputParentStat?.isDirectory() || outputParentStat.isSymbolicLink() || fs.realpathSync(outputParent).toLowerCase() !== outputParent.toLowerCase() ||
      outputStat || !Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 30 * 60 * 1000) {
    throw new Error("Windows Hermes source extraction needs a safe root, canonical new output and bounded tar timeout");
  }
  windowsControllerRegularFile(tarTool, "Hermes source tar tool");
  const archiveStat = windowsControllerRegularFile(state.hermesSourceTar, "Same-run Hermes source tar");
  if (archiveStat.size < 1024 || archiveStat.size > (2 * 1024 ** 3)) {throw new Error("Same-run Hermes source tar is outside its 2 GiB bound");}
  /** @param {string[]} args @param {number} maxBuffer */
  const runTar = (args, maxBuffer) => {
    const result = spawnSync(tarTool, args, {encoding: "utf8", windowsHide: true, timeout: timeoutMs, maxBuffer});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Same-run Hermes source tar operation failed", {cause: result.error});}
    return result.stdout || "";
  };
  const names = runTar(["-tf", state.hermesSourceTar], 64 << 20).split(/\r?\n/).filter(Boolean),
    verbose = runTar(["-tvf", state.hermesSourceTar], 64 << 20).split(/\r?\n/).filter(Boolean);
  if (!names.length || names.length > 200000 || names.length !== verbose.length || names.some(name => {
    const normalized = name.endsWith("/") ? name.slice(0, -1) : name;
    return !safeRelative(normalized) || normalized.split("/")[0] !== state.hermesSourceName;
  }) || verbose.some(line => !["-", "d"].includes(line[0] || ""))) {
    throw new Error("Same-run Hermes source tar has unsafe paths, links, special files or roots");
  }
  if (!names.includes(state.hermesSourceName + "/roots/hermes/config") || !names.includes(state.hermesSourceName + "/fetch.json") ||
      !names.includes(state.hermesSourceName + "/roots.json")) {throw new Error("Same-run Hermes source tar omits its bare source or original fetch receipts");}
  fs.mkdirSync(state.hermesExtractRoot, {mode: 0o700});
  runTar(["-xf", state.hermesSourceTar, "-C", state.hermesExtractRoot, "--", state.hermesSourceName], 4 << 20);
  return {...readWindowsHermesSource({hermesExtractRoot: state.hermesExtractRoot, hermesSourceName: state.hermesSourceName}, originalRoot),
    tarSha256: fileHash(state.hermesSourceTar)};
}

/** @typedef {{files:Array<{path:string,bytes:number,sha256:string}>,bytes:number}} WindowsControllerFileLedger */
/** @param {string} source @param {string} target @param {string} relative @param {WindowsControllerFileLedger} ledger */
function copyWindowsControllerPath(source, target, relative, ledger) {
  if (!safeRelative(relative)) {throw new Error("Windows controller selected path is unsafe");}
  const stat = fs.lstatSync(source);
  if (stat.isSymbolicLink() || fs.realpathSync(source).toLowerCase() !== path.resolve(source).toLowerCase()) {
    throw new Error("Windows controller SDK selection contains a symlink or redirected path");
  }
  if (stat.isDirectory()) {
    fs.mkdirSync(target, {recursive: true});
    for (const entry of fs.readdirSync(source, {withFileTypes: true}).sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0)) {
      if (!entry.isDirectory() && !entry.isFile()) {throw new Error("Windows controller SDK selection contains a special file");}
      copyWindowsControllerPath(path.join(source, entry.name), path.join(target, entry.name), relative + "/" + entry.name, ledger);
    }
    return;
  }
  if (!stat.isFile() || ledger.files.length >= 50000 || (ledger.bytes += stat.size) > (512 << 20)) {
    throw new Error("Windows controller selected files exceed their regular-file count/size bound");
  }
  fs.mkdirSync(path.dirname(target), {recursive: true});
  const sha256 = fileHash(source);
  fs.copyFileSync(source, target, fs.constants.COPYFILE_EXCL);
  const copied = fs.lstatSync(target);
  if (!copied.isFile() || copied.isSymbolicLink() || copied.size !== stat.size || fileHash(target) !== sha256) {
    throw new Error("Windows controller SDK handoff copy differs from the producer output");
  }
  ledger.files.push({path: relative, bytes: stat.size, sha256});
}

/** @param {string} root @param {string} name */
function windowsControllerPackageInventory(root, name) {
  /** @type {Array<{path:string,bytes:number,sha256:string}>} */
  const files = [], members = [name]; let bytes = 0;
  const visit = (/** @type {string} */ directory, /** @type {string} */ relative) => {
    const stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory).toLowerCase() !== path.resolve(directory).toLowerCase()) {
      throw new Error("Windows controller package contains a redirected directory");
    }
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0)) {
      const item = path.join(directory, entry.name), child = relative ? relative + "/" + entry.name : entry.name;
      if (!safeRelative(child)) {throw new Error("Windows controller package path is unsafe");}
      if (entry.isSymbolicLink()) {throw new Error("Windows controller package contains a symlink");}
      members.push(name + "/" + child);
      if (entry.isDirectory()) {visit(item, child); continue;}
      const itemStat = fs.lstatSync(item);
      if (!entry.isFile() || itemStat.isSymbolicLink() || !itemStat.isFile() || files.length >= 50000 || (bytes += itemStat.size) > (512 << 20)) {
        throw new Error("Windows controller package exceeds its regular-file size/count bound");
      }
      files.push({path: child, bytes: itemStat.size, sha256: fileHash(item)});
    }
  };
  visit(root, "");
  if (!files.length) {throw new Error("Windows controller package is empty");}
  return {files: files.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0), members: members.sort(), bytes};
}

/** @param {Array<{path:string,bytes:number,sha256:string}>} files @param {unknown} sources */
function verifyWindowsHermesSourceHandoff(files, sources) {
  const prefix = "hermes/source/", selected = files.filter(row => row.path.startsWith(prefix));
  if (!selected.length || !sources || typeof sources !== "object" || Array.isArray(sources) || selected.some(row => {
    const sourcePath = row.path.slice(prefix.length);
    return !safeRelative(sourcePath) || !/^[a-f0-9]{64}$/.test(row.sha256) || Reflect.get(sources, sourcePath) !== row.sha256;
  })) {throw new Error("Windows Hermes selected source files do not match the producer's tracked-source hashes");}
  return selected.length;
}

/** Pack only a selected build-only closure and its unchanged producer receipts.
 * @param {Pick<ReturnType<typeof windowsControllerStatePaths>,"archive"|"runnerTemp"|"outputName"|"outputRoot">} state
 * @param {string} tarTool @param {Record<string,unknown>} handoff */
function packWindowsControllerHandoff(state, tarTool, handoff, timeoutMs = 5 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 5 * 60 * 1000) {throw new Error("Windows controller handoff timeout is invalid");}
  if (fs.existsSync(state.archive)) {throw new Error("Windows controller handoff tar already exists");}
  const before = windowsControllerPackageInventory(state.outputRoot, state.outputName), record = {...handoff, files: before.files};
  fs.writeFileSync(path.join(state.outputRoot, "handoff.json"), JSON.stringify(record, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const complete = windowsControllerPackageInventory(state.outputRoot, state.outputName), toolHash = fileHash(tarTool), create = spawnSync(tarTool,
    ["-cf", state.archive, "-C", state.runnerTemp, state.outputName], {encoding: "utf8", windowsHide: true,
      timeout: timeoutMs, maxBuffer: 4 << 20});
  if (create.error || create.signal || create.status !== 0) {throw new Error("Windows controller handoff tar creation failed", {cause: create.error});}
  const listing = spawnSync(tarTool, ["-tf", state.archive], {encoding: "utf8", windowsHide: true, timeout: Math.min(timeoutMs, 2 * 60 * 1000), maxBuffer: 16 << 20}),
    archiveStat = fs.lstatSync(state.archive, {throwIfNoEntry: false});
  if (listing.error || listing.signal || listing.status !== 0 || typeof listing.stdout !== "string" || fileHash(tarTool) !== toolHash ||
      !archiveStat?.isFile() || archiveStat.isSymbolicLink() || archiveStat.size > (576 << 20)) {
    throw new Error("Windows controller handoff tar listing or tool verification failed", {cause: listing.error});
  }
  const actual = listing.stdout.split(/\r?\n/).filter(Boolean).map(name => name.replaceAll("\\", "/").replace(/\/$/, "")).sort();
  if (!isDeepStrictEqual(actual, complete.members) || complete.bytes > (512 << 20)) {
    throw new Error("Windows controller handoff tar has unexpected selected members or size");
  }
  for (const file of complete.files) {
    if (fileHash(path.join(state.outputRoot, ...file.path.split("/"))) !== file.sha256) {throw new Error("Windows controller handoff changed while packing");}
  }
  return {bundleSha256: fileHash(state.archive), bundleBytes: fs.statSync(state.archive).size,
    handoffSha256: fileHash(path.join(state.outputRoot, "handoff.json")), files: before.files.length, selectedBytes: before.bytes};
}

/** @param {NodeJS.ProcessEnv} environment @param {Record<string,string>} values */
function appendWindowsControllerOutputs(environment, values) {
  const output = environment["GITHUB_OUTPUT"] || "", stat = path.isAbsolute(output) ? fs.lstatSync(output, {throwIfNoEntry: false}) : undefined;
  if (!stat?.isFile() || stat.isSymbolicLink() || Object.entries(values).some(([key, value]) =>
    !/^[a-z][a-z0-9_]*$/.test(key) || !/^(?:[a-f0-9]{64}|[0-9]+)$/.test(value))) {
    throw new Error("Windows controller-support GitHub outputs are invalid");
  }
  fs.appendFileSync(output, Object.entries(values).map(([key, value]) => key + "=" + value).join("\n") + "\n", "utf8");
}

/** @param {string} file @param {string[]} args @param {NodeJS.ProcessEnv} environment @param {number} timeoutMs */
function runWindowsControllerProducer(file, args, environment, timeoutMs) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 55 * 60 * 1000) {throw new Error("Windows controller producer timeout is invalid");}
  const result = spawnSync(process.execPath, [path.join(repo, file), ...args], {cwd: repo, env: environment, stdio: "inherit",
    windowsHide: true, timeout: timeoutMs});
  if (result.error || result.signal || result.status !== 0) {throw new Error("Windows controller public producer failed: " + file, {cause: result.error});}
}

/** @param {ReturnType<typeof windowsControllerStatePaths>} state */
function windowsControllerPreflight(state) {
  const outputStat = fs.lstatSync(state.outputFile, {throwIfNoEntry: false}), summaryStat = fs.lstatSync(state.summaryFile, {throwIfNoEntry: false});
  if (!outputStat?.isFile() || outputStat.isSymbolicLink() || !summaryStat?.isFile() || summaryStat.isSymbolicLink() ||
      [state.workRoot, state.outputRoot, state.archive].some(file => fs.existsSync(file))) {
    throw new Error("Windows controller-support stage needs new run-scoped outputs and regular workflow files");
  }
  windowsControllerDeadlineBudget(state.deadlineEpoch, "fetch");
  fs.mkdirSync(state.workRoot, {mode: 0o700});
}

/** Fetch and build original native Windows libuv/zlib and OpenSSL inputs; hand off only selected files. */
async function buildWindowsControllerIoCi() {
  const state = windowsControllerStatePaths(process.env, process.platform, "io"),
    tools = /** @type {WindowsControllerIoTools} */ (windowsControllerTools(process.env, "io")),
    environment = /** @type {NodeJS.ProcessEnv} */ ({...process.env, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2", OMP_NUM_THREADS: "2"});
  windowsControllerPreflight(state);
  const toolHashes = Object.fromEntries([tools.cc, tools.cmake, tools.ninja, ...Object.values(tools.openssl)].map(file => [file, fileHash(file)])),
    ioBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "fetch");
  const ioFetch = await fetchIoArchives(state.ioSource, environment, true, {timeoutMs: Math.min(ioBudget.timeoutMs, 10 * 60 * 1000)}),
    ioFetched = verifyIoArchives(state.ioSource), ioFetchFile = path.join(state.ioSource, "fetch.json"), ioFetchSha = fileHash(ioFetchFile),
    opensslBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "fetch"),
    opensslFetch = await fetchWindowsOpenSslArchives(state.opensslSource, environment, true,
      {timeoutMs: Math.min(opensslBudget.timeoutMs, 10 * 60 * 1000)}),
    opensslFetchFile = path.join(state.opensslSource, "fetch.json"), opensslArchive = path.join(state.opensslSource, "openssl-" + opensslPin.version + ".tar.gz"),
    opensslFetchSha = fileHash(opensslFetchFile);
  if (ioFetch.status !== "PASS" || ioFetch.target !== "io" || ioFetched.status !== "PASS" || ioFetched.target !== "io" ||
      ioFetched.publicationQualified !== false ||
      fileHash(opensslArchive) !== opensslPin.sha256 || opensslFetch.status !== "PASS" || opensslFetch.target !== "windows-openssl" ||
      opensslFetch.archives?.["openssl"]?.sha256 !== opensslPin.sha256 || opensslFetch.archives?.["openssl"]?.path !== opensslArchive) {
    throw new Error("Windows controller source acquisition did not retain its exact pinned inputs");
  }
  const opensslConfig = {version: 1, jobs: 2, tools: tools.openssl};
  fs.writeFileSync(state.opensslToolsFile, JSON.stringify(opensslConfig, null, 2) + "\n", {flag: "wx", mode: 0o600});
  const sourceHashes = {[opensslArchive]: opensslPin.sha256,
      [path.join(state.ioSource, "libuv-v1.51.0.tar.gz")]: ioPins.libuv.sha256,
      [path.join(state.ioSource, "zlib-1.3.1.tar.gz")]: ioPins.zlib.sha256,
      [ioFetchFile]: ioFetchSha, [opensslFetchFile]: opensslFetchSha, [state.opensslToolsFile]: fileHash(state.opensslToolsFile)},
    ioRecipe = sourceInputs(repo, [], ["scripts/build-native-io.js", "scripts/build-environment.js", "scripts/build-source-inputs.js", "scripts/native-audit.js"]),
    opensslRecipe = opensslBuildRecipe();
  const commands = windowsControllerIoBuildCommands(state, tools), ioCommand = commands[0], opensslCommand = commands[1];
  if (!ioCommand || !opensslCommand) {throw new Error("Windows controller I/O/OpenSSL producer command set is incomplete");}
  const ioBuildBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "build");
  runWindowsControllerProducer(ioCommand.script, ioCommand.args, environment, Math.min(ioBuildBudget.timeoutMs, 35 * 60 * 1000));
  const ioAdmitted = nativeIoInputs(state.ioSdk, "win32"), ioReceiptFile = ioAdmitted.receipt, ioReceiptSha = fileHash(ioReceiptFile),
    opensslBuildBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "build");
  runWindowsControllerProducer(opensslCommand.script, opensslCommand.args, environment, Math.min(opensslBuildBudget.timeoutMs, 45 * 60 * 1000));
  const opensslAdmitted = nativeOpenSslInputs(state.opensslSdk, "win32"), opensslReceiptFile = path.join(state.opensslSdk, "result.json"),
    opensslReceiptSha = fileHash(opensslReceiptFile), inputs = {io: readBuildJson(ioReceiptFile), openssl: readBuildJson(opensslReceiptFile)};
  const expectedIoInputs = {
      [path.join(state.ioSource, "libuv-v1.51.0.tar.gz")]: ioPins.libuv.sha256,
      [path.join(state.ioSource, "zlib-1.3.1.tar.gz")]: ioPins.zlib.sha256},
    expectedIoTools = Object.fromEntries([tools.cmake, tools.cc, tools.ninja].map(file => [file, fileHash(file)])),
    expectedOpenSslTools = Object.fromEntries(Object.entries(tools.openssl).map(([name, file]) => [name, {path: file, sha256: fileHash(file)}]));
  if (!isDeepStrictEqual(inputs.io.recipe, ioRecipe) || !isDeepStrictEqual(inputs.io.inputs, expectedIoInputs) ||
      !isDeepStrictEqual(inputs.io.fingerprints, expectedIoTools) || inputs.io.target !== "win32" || inputs.io.buildOnly !== true ||
      !isDeepStrictEqual(inputs.openssl.recipe, opensslRecipe) || inputs.openssl.target !== "VC-WIN64A" ||
      inputs.openssl.platform !== "win32" || inputs.openssl.config?.sha256 !== fileHash(state.opensslToolsFile) ||
      inputs.openssl.archive?.path !== opensslArchive || inputs.openssl.archive?.sha256 !== opensslPin.sha256 ||
      !isDeepStrictEqual(inputs.openssl.tools, expectedOpenSslTools) ||
      Object.entries(sourceHashes).some(([file, sha]) => fileHash(file) !== sha) ||
      Object.entries(toolHashes).some(([file, sha]) => fileHash(file) !== sha)) {
    throw new Error("Windows controller I/O/OpenSSL source, recipe, tool or producer receipt changed");
  }
  const ledger = {files: [], bytes: 0};
  fs.mkdirSync(state.outputRoot, {mode: 0o700});
  for (const [relative, sha] of Object.entries(ioAdmitted.artifacts).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) {
    if (!relative.startsWith("include/") && !relative.startsWith("lib/")) {throw new Error("Native I/O producer selected a non-header/library handoff member");}
    if (fileHash(path.join(state.ioSdk, ...relative.split("/"))) !== sha) {throw new Error("Native I/O selected artifact changed before handoff");}
    copyWindowsControllerPath(path.join(state.ioSdk, ...relative.split("/")), path.join(state.outputRoot, "io-sdk", ...relative.split("/")),
      "io-sdk/" + relative, ledger);
  }
  const sslMembers = Object.keys(opensslAdmitted.artifacts).filter(name => name.startsWith("include/") || name === "lib/libcrypto.lib" || name === "lib/libssl.lib").sort();
  if (!sslMembers.includes("lib/libcrypto.lib") || !sslMembers.includes("lib/libssl.lib") ||
      Object.keys(opensslAdmitted.artifacts).some(name => !name.startsWith("include/") &&
        !["bin/openssl.exe", "lib/libcrypto.lib", "lib/libssl.lib"].includes(name))) {
    throw new Error("OpenSSL selected handoff omits a static library or has unexpected producer artifacts");
  }
  for (const relative of sslMembers) {
    if (fileHash(path.join(state.opensslSdk, ...relative.split("/"))) !== opensslAdmitted.artifacts[relative]) {throw new Error("OpenSSL selected artifact changed before handoff");}
    copyWindowsControllerPath(path.join(state.opensslSdk, ...relative.split("/")), path.join(state.outputRoot, "openssl-sdk", ...relative.split("/")),
      "openssl-sdk/" + relative, ledger);
  }
  copyWindowsControllerPath(path.join(state.opensslSdk, "source", "LICENSE.txt"), path.join(state.outputRoot, "openssl-sdk/source/LICENSE.txt"),
    "openssl-sdk/source/LICENSE.txt", ledger);
  copyWindowsControllerPath(ioReceiptFile, path.join(state.outputRoot, "io-sdk/result.json"), "io-sdk/result.json", ledger);
  copyWindowsControllerPath(opensslReceiptFile, path.join(state.outputRoot, "openssl-sdk/result.json"), "openssl-sdk/result.json", ledger);
  copyWindowsControllerPath(ioFetchFile, path.join(state.outputRoot, "provenance/io-fetch.json"), "provenance/io-fetch.json", ledger);
  copyWindowsControllerPath(opensslFetchFile, path.join(state.outputRoot, "provenance/openssl-fetch.json"), "provenance/openssl-fetch.json", ledger);
  copyWindowsControllerPath(state.opensslToolsFile, path.join(state.outputRoot, "openssl-sdk/producer-tools.json"),
    "openssl-sdk/producer-tools.json", ledger);
  if (fileHash(path.join(state.outputRoot, "io-sdk/result.json")) !== ioReceiptSha ||
      fileHash(path.join(state.outputRoot, "openssl-sdk/result.json")) !== opensslReceiptSha ||
      fileHash(path.join(state.outputRoot, "openssl-sdk/producer-tools.json")) !== inputs.openssl.config.sha256) {
    throw new Error("Windows controller-support producer receipt or tool descriptor changed during handoff");
  }
  if (Object.entries(sourceHashes).some(([file, sha]) => fileHash(file) !== sha) ||
      Object.entries(toolHashes).some(([file, sha]) => fileHash(file) !== sha)) {
    throw new Error("Windows controller-support inputs changed while packaging the handoff");
  }
  const packageBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "package"),
    packaged = packWindowsControllerHandoff(state, tools.openssl.tar, {version: 1, format: "mom-windows-controller-support-handoff-v1",
      runAttempt: state.suffix, target: "win32", architecture: "x64", kind: "io", buildOnly: true, executed: false, publicationQualified: false,
      scope: "Selected public libuv/zlib and OpenSSL headers/static archives plus unchanged receipts; not full SDK re-admission or application qualification.",
      producers: [{script: ioCommand.script, receipt: "io-sdk/result.json", receiptSha256: ioReceiptSha},
        {script: opensslCommand.script, receipt: "openssl-sdk/result.json", receiptSha256: opensslReceiptSha}],
      sourceReceipts: [{file: "provenance/io-fetch.json", sha256: ioFetchSha, archives: {libuv: ioPins.libuv.sha256, zlib: ioPins.zlib.sha256}},
        {file: "provenance/openssl-fetch.json", sha256: opensslFetchSha, archives: {openssl: opensslPin.sha256}}],
      omitted: ["OpenSSL source/build trees", "openssl.exe", "compiler/build tools", "DLLs"]}, packageBudget.timeoutMs);
  appendWindowsControllerOutputs(environment, {bundle_sha256: packaged.bundleSha256, bundle_bytes: String(packaged.bundleBytes),
    io_receipt_sha256: ioReceiptSha, openssl_receipt_sha256: opensslReceiptSha, handoff_sha256: packaged.handoffSha256,
    io_bundle_sha256: packaged.bundleSha256, io_handoff_sha256: packaged.handoffSha256});
  fs.appendFileSync(state.summaryFile, `### Native Windows controller I/O/OpenSSL support inputs\n- Built the pinned same-run Windows static producers and retained ${packaged.files} selected files (${packaged.selectedBytes} bytes).\n- This selected handoff is build-only; no complete-SDK, application, runtime or publication qualification is claimed.\n`, "utf8");
  return packaged;
}

/** Verify and build same-run Hermes source on Windows, retaining only headers/config/static libraries. */
function buildWindowsControllerHermesCi() {
  const state = windowsControllerStatePaths(process.env, process.platform, "hermes"),
    tools = /** @type {WindowsControllerHermesTools} */ (windowsControllerTools(process.env, "hermes")),
    environment = /** @type {NodeJS.ProcessEnv} */ ({...process.env, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2", OMP_NUM_THREADS: "2"});
  windowsControllerPreflight(state);
  const tarTool = path.join(environment["SystemRoot"] || environment["WINDIR"] || "", "System32", "tar.exe");
  windowsControllerRegularFile(tarTool, "Windows inbox tar tool");
  const tarSha = fileHash(tarTool), source = extractWindowsHermesSource(state, tarTool,
      windowsControllerDeadlineBudget(state.deadlineEpoch, "extract").timeoutMs), command = windowsControllerHermesBuildCommand(state, tools),
    plan = hermesSdkPlan(source.source, state.hermesSdk, tools, 2, "win32");
  if (fs.existsSync(state.hermesSdk)) {throw new Error("Windows Hermes SDK output must be new");}
  const producerRecipe = sourceInputs(repo, [], ["scripts/build-native-hermes.js", "scripts/build-source-inputs.js", "scripts/build-environment.js",
      "scripts/native-audit.js", ...plan.patches]), fingerprints = Object.fromEntries([...Object.values(tools), ...plan.patches.map(file => path.join(repo, file))].map(file => [file, fileHash(file)])),
    buildBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "build");
  runWindowsControllerProducer(command.script, command.args, environment, Math.min(buildBudget.timeoutMs, 45 * 60 * 1000));
  const resultFile = path.join(state.hermesSdk, "result.json"), beforeFile = path.join(state.hermesSdk, "before.json"),
    result = windowsControllerReadJson(resultFile, 16 << 20, "Windows Hermes producer receipt"), before = windowsControllerReadJson(beforeFile, 16 << 20, "Windows Hermes producer plan");
  if (result.version !== 1 || result.target !== "win32" || !isDeepStrictEqual(result.pins, hermesPins) ||
      result.source !== plan.source || result.build !== plan.build || result.buildOnly !== true || result.publicationQualified !== false ||
      !isDeepStrictEqual(result.recipe, producerRecipe) || !isDeepStrictEqual(before.configure, plan.configure) || !isDeepStrictEqual(before.compile, plan.compile) ||
      !isDeepStrictEqual(result.fingerprints, fingerprints) || !isDeepStrictEqual(before.fingerprints, fingerprints) ||
      !isDeepStrictEqual(Object.keys(result.artifacts || {}).sort(), Object.keys(plan.artifacts).sort())) {
    throw new Error("Windows Hermes producer receipt differs from the exact pinned source/tool/build-only plan");
  }
  for (const [name, file] of Object.entries(plan.artifacts)) {
    const row = result.artifacts[name], stat = windowsControllerRegularFile(file, "Hermes " + name + " output");
    if (!row || row.path !== file || row.bytes !== stat.size || !/^[a-f0-9]{64}$/.test(row.sha256 || "") || fileHash(file) !== row.sha256 ||
        name !== "compiler" && !isStaticArchive(file)) {throw new Error("Windows Hermes producer artifact changed: " + name);}
  }
  if (fileHash(tarTool) !== tarSha || fileHash(source.fetchFile) !== source.fetchSha256 || fileHash(source.rootsFile) !== source.rootsSha256 ||
      fileHash(state.hermesSourceTar) !== source.tarSha256) {throw new Error("Same-run Hermes source input or Windows tar tool changed during production");}
  const ledger = {files: [], bytes: 0};
  fs.mkdirSync(state.outputRoot, {mode: 0o700});
  const intelRuntime = copyWindowsControllerIntelRuntime(tools.cc, state.outputRoot, ledger);
  if (intelRuntime.record.compiler.sha256 !== fingerprints[tools.cc]) {
    throw new Error("Intel runtime export compiler differs from the same-run Hermes tool fingerprint");
  }
  /** @type {Array<[string,string]>} */
  const selected = [
    [path.join(plan.source, "LICENSE"), "hermes/source/LICENSE"],
    [path.join(plan.source, "include"), "hermes/source/include"], [path.join(plan.source, "API"), "hermes/source/API"],
    [path.join(plan.source, "public"), "hermes/source/public"], [path.join(plan.build, "lib/config"), "hermes/build/lib/config"],
    [plan.artifacts.runtime, "hermes/build/lib/hermesvmlean_a.lib"], [plan.artifacts.jsi, "hermes/build/jsi/jsi.lib"],
    [plan.artifacts.context, "hermes/build/external/boost/boost_1_86_0/libs/context/boost_context.lib"],
  ];
  for (const [sourcePath, relative] of selected) {
    copyWindowsControllerPath(sourcePath, path.join(state.outputRoot, ...relative.split("/")), relative, ledger);
  }
  verifyWindowsHermesSourceHandoff(ledger.files, result.sources);
  const receiptFile = path.join(state.hermesSdk, "result.json"), beforeHash = fileHash(beforeFile), receiptSha = fileHash(receiptFile),
    sourceFetchTarget = path.join(state.outputRoot, "provenance/hermes-fetch.json"), sourceRootsTarget = path.join(state.outputRoot, "provenance/hermes-roots.json");
  copyWindowsControllerPath(receiptFile, path.join(state.outputRoot, "hermes/result.json"), "hermes/result.json", ledger);
  copyWindowsControllerPath(beforeFile, path.join(state.outputRoot, "hermes/before.json"), "hermes/before.json", ledger);
  copyWindowsControllerPath(source.fetchFile, sourceFetchTarget, "provenance/hermes-fetch.json", ledger);
  copyWindowsControllerPath(source.rootsFile, sourceRootsTarget, "provenance/hermes-roots.json", ledger);
  const packageBudget = windowsControllerDeadlineBudget(state.deadlineEpoch, "package"),
    packaged = packWindowsControllerHandoff(state, tarTool, {version: 1, format: "mom-windows-controller-support-handoff-v1",
      runAttempt: state.suffix, target: "win32", architecture: "x64", kind: "hermes", buildOnly: true, executed: false, publicationQualified: false,
      scope: "Selected Hermes root license, headers/config/static libraries and Intel 2026.2 static runtime/license files plus unchanged same-run source and build receipts; no AOT compiler or full-SDK re-admission.",
      producers: [{script: command.script, receipt: "hermes/result.json", receiptSha256: receiptSha, plan: "hermes/before.json", planSha256: beforeHash}],
      sourceReceipts: [{file: "provenance/hermes-fetch.json", sha256: source.fetchSha256, rootsFile: "provenance/hermes-roots.json", rootsSha256: source.rootsSha256,
        revision: hermesPins.revision}], intelRuntime: intelRuntime.record,
      omitted: ["Hermes Git/build trees", "shermes.exe", "hermesPublic.lib", "compiler/build tools", "DLLs"]}, packageBudget.timeoutMs);
  if (fileHash(tarTool) !== tarSha || fileHash(source.fetchFile) !== source.fetchSha256 || fileHash(source.rootsFile) !== source.rootsSha256 ||
      fileHash(state.hermesSourceTar) !== source.tarSha256 || Object.entries(fingerprints).some(([file, sha]) => fileHash(file) !== sha) ||
      intelRuntime.sourceFiles.some(row => fileHash(row.path) !== row.sha256) ||
      [...intelRuntime.record.libraries, ...intelRuntime.record.licenses].some(row => {
        const file = path.join(state.outputRoot, ...row.path.split("/")), stat = fs.lstatSync(file, {throwIfNoEntry: false});
        return !stat?.isFile() || stat.isSymbolicLink() || stat.size !== row.bytes || fileHash(file) !== row.sha256;
      })) {
    throw new Error("Windows Hermes source, tool or patch changed while packaging the handoff");
  }
  appendWindowsControllerOutputs(environment, {bundle_sha256: packaged.bundleSha256, bundle_bytes: String(packaged.bundleBytes),
    hermes_receipt_sha256: receiptSha, source_receipt_sha256: source.fetchSha256, handoff_sha256: packaged.handoffSha256,
    hermes_bundle_sha256: packaged.bundleSha256, hermes_handoff_sha256: packaged.handoffSha256});
  fs.appendFileSync(state.summaryFile, `### Native Windows controller Hermes support inputs\n- Built the pinned same-run Windows Hermes producer and retained ${packaged.files} selected files (${packaged.selectedBytes} bytes).\n- The Linux Hermes AOT tool remains a separate input; this handoff does not include or qualify a Windows executable for Linux.\n`, "utf8");
  return packaged;
}

/** Bound one child by both its own cap and the merged job's deadline (job start + timeout - margin).
 * @param {NodeJS.ProcessEnv} environment @param {number} maximumMs */
function ciJobChildTimeout(environment, maximumMs) {
  const epoch = environment["MOM_CI_DEADLINE_EPOCH"] || "", timeoutMs = Math.min(maximumMs, Number(epoch) * 1000 - Date.now());
  if (!/^\d{10}$/.test(epoch) || !Number.isSafeInteger(timeoutMs) || timeoutMs < 60 * 1000) {
    throw new Error("Native CI job deadline is malformed or leaves no time for the next child");
  }
  return timeoutMs;
}

/** Cold-build the Linux LLVM SDK in one job: one source restore and oneAPI provisioning, then the
 * maintained configure, build-phase, install and audit order without intermediate state transfers.
 * Each child keeps its existing cap and is also bounded by the job deadline.
 * @param {ReturnType<typeof ciPaths>} paths */
function buildLinuxLlvmCi(paths) {
  const builder = "scripts/build-native-llvm21-sdk.js", env = paths.env,
    phases = [...Object.values(linuxLlvmCiStageDefinitions).flatMap(row => row.phases || [row.artifactStage]), "lld", "tools"];
  ciJobChildTimeout(env, 60 * 1000);
  restorePrior(paths, "sources", false);
  provisionOneApi(paths);
  if (fs.existsSync(paths.sdkRoot)) {throw new Error("Native SDK configure output already exists");}
  fs.mkdirSync(paths.sdkRoot, {mode: 0o700});
  const command = linuxLlvmConfigureCommand(paths);
  runNode(command.script, command.args, env, ciJobChildTimeout(env, 50 * 60 * 1000));
  for (const phase of phases) {runNode(builder, ["build", paths.output, phase], env, ciJobChildTimeout(env, 50 * 60 * 1000));}
  runNode(builder, ["install", paths.output], env, ciJobChildTimeout(env, 50 * 60 * 1000));
  runNode(builder, ["audit", paths.output], env, ciJobChildTimeout(env, 15 * 60 * 1000));
  packState(path.join(paths.runnerTemp, "native-linux-llvm21-sdk-" + paths.suffix + ".tar"), temp, [paths.sourceName, paths.sdkName]);
}

/** @param {string[]} args */
async function runStage(args) {
  const stage = args[0] || "";
  if (stage === "linux-native-component") {
    if (args.length !== 2) {throw new Error("Linux native component stage requires exactly core or one GPU profile");}
    buildLinuxNativeComponent(ciPaths(), args[1] || ""); return;
  }
  if (stage === "windows-native-gpu") {
    if (args.length !== 4) {throw new Error("Windows GPU profile stage requires an explicit profile, config path and config SHA-256");}
    buildWindowsGpuProfileCi(args[1] || "", args[2] || "", args[3] || ""); return;
  }
  if (stage === "windows-native-llvm21-descriptor") {
    if (args.length !== 3) {throw new Error("Windows LLVM descriptor stage needs one input JSON and its SHA-256 pin");}
    buildWindowsNativeLlvmDescriptorCi(args.slice(1)); return;
  }
  if (args.length !== 1) {throw new Error("Native CI stage takes no extra arguments");}
  if (stage === "windows-native-core") {buildWindowsNativeCoreCi(); return;}
  if (stage === "windows-host-sdk-sources") {buildWindowsHostSdkSourcesCi(); return;}
  if (stage === "windows-controller-io") {await buildWindowsControllerIoCi(); return;}
  if (stage === "windows-controller-hermes") {buildWindowsControllerHermesCi(); return;}
  const paths = ciPaths(), builder = "scripts/build-native-llvm21-sdk.js";
  if (stage === "linux-llvm-admission") {
    admitLinuxLlvmOutput(paths);
    return;
  }
  if (stage === "linux-llvm") {buildLinuxLlvmCi(paths); return;}
  if (stage === "linux-llvm-installed-sdk") {buildLinuxLlvmSdkHandoff(paths); return;}
  if (stage === "sources") {
    runNode("scripts/fetch-native-sdk-sources.js", ["build", paths.source, "--allow-network"], paths.env, 25 * 60 * 1000);
    runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", paths.source], paths.env, 2 * 60 * 1000);
    runNode("scripts/fetch-native-sdk-sources.js", ["io", paths.ioSource, "--allow-network"], paths.env, 25 * 60 * 1000);
    verifyIoSource(paths);
    packState(outputTar(paths, stage), temp, [paths.sourceName, paths.ioSourceName]);
    return;
  }
  if (stage === "linux-llvm-tools") {buildLinuxLlvmToolsCi(paths); return;}
  if (stage === "acpp-sources") {fetchAcppSource(paths); return;}
  if (stage === "linux-sources") {fetchLinuxSources(paths); return;}
  if (stage === "dpcpp-cuda-inputs") {buildDpcppCudaInputs(paths); return;}
  if (stage === "gnu-headers") {buildLinuxGnuHeaders(paths); return;}
  if (stage === "windows-gnu-headers") {buildWindowsGnuHeaders(paths); return;}
  if (stage === "linux-jit-resources") {buildLinuxJitResources(paths); return;}
  if (stage === "linux-jit-configure" || stage === "linux-jit-support" || stage === "linux-jit-targets" ||
      stage === "linux-jit-frontend" || stage === "linux-jit-link") {
    buildLinuxJitSdkPhase(paths, stage.slice("linux-jit-".length)); return;
  }
  if (stage === "linux-jit-runtime-proof") {packLinuxJitRuntimeProof(paths); return;}
  if (stage === "linux-jit-runtime-admit") {admitLinuxJitRuntimeProof(paths); return;}
  if (stage === "linux-jit-static-dpcpp") {buildLinuxJitStaticSdk(paths); return;}
  if (stage === "rocm-sources") {fetchRocmSources(paths); return;}
  if (stage === "rocm-comgr-sdk") {buildRocmComgrSdk(paths); return;}
  if (stage === "rocm-host-sdk") {buildRocmHostSdk(paths); return;}
  if (stage === "hermes-sources") {fetchHermesSource(paths); return;}
  if (stage === "hermes-sdk") {buildHermesSdk(paths); return;}
  if (stage === "zstd-sources") {fetchZstdSource(paths); return;}
  if (stage === "zstd-sdk") {buildZstdSdk(paths); return;}
  if (stage === "openmp-sdk") {buildOpenmpSdk(paths); return;}
  if (stage === "acpp-cuda-hip-sdk") {buildAcppCudaHipSdk(paths); return;}
  if (stage === "nvrtc-linux") {verifyNvrtcTarget(paths, "linux"); return;}
  if (stage === "nvrtc-windows") {verifyNvrtcTarget(paths, "windows"); return;}
  if (stage === "cuda-images-linux") {buildLinuxCudaImages(paths); return;}
  if (stage === "linux-sdk-descriptor") {buildLinuxNativeSdkDescriptor(paths); return;}
  if (stage === "linux-native-application") {buildLinuxNativeApplication(paths); return;}
  if (stage === "io-openssl") {buildIoOpenSsl(paths); return;}
  if (Object.hasOwn(linuxLlvmCiStageDefinitions, stage) && linuxLlvmCiStageDefinitions[stage]?.phases) {
    buildLinuxLlvmGroupCi(paths, stage); return;
  }
  const previousByStage = /** @type {Record<string, string>} */ ({configure: "sources",
    ...Object.fromEntries(Object.entries(linuxLlvmCiStageDefinitions).map(([name, definition]) => [name, definition.previousStage])),
    "install-audit": "clang-frontend"});
  const previous = previousByStage[stage];
  if (!previous) {throw new Error("Unknown native Linux SDK stage: " + stage);}
  restorePrior(paths, previous, stage !== "configure");
  if (stage === "configure") {
    provisionOneApi(paths);
    if (fs.existsSync(paths.sdkRoot)) {throw new Error("Native SDK configure output already exists");}
    fs.mkdirSync(paths.sdkRoot, {mode: 0o700});
    const command = linuxLlvmConfigureCommand(paths);
    runNode(command.script, command.args, paths.env);
  } else if (["llvm-core-codegen", "clang-frontend"].includes(stage)) {
    provisionOneApi(paths);
    runNode(builder, ["build", paths.output, stage], paths.env);
    packCurrent(paths, stage);
    return;
  } else if (stage === "install-audit") {
    provisionOneApi(paths);
    // The short linker/tool tail shares this restore; original public receipts
    // still gate installation, audit and every downstream tool handoff.
    for (const phase of ["lld", "tools"]) {runNode(builder, ["build", paths.output, phase], paths.env);}
    runNode(builder, ["install", paths.output], paths.env);
    runNode(builder, ["audit", paths.output], paths.env, 15 * 60 * 1000);
    packState(path.join(paths.runnerTemp, "native-linux-llvm21-sdk-" + paths.suffix + ".tar"), temp, [paths.sourceName, paths.sdkName]);
    return;
  } else {throw new Error("Unknown native Linux SDK stage: " + stage);}
  packCurrent(paths, outputStageName(stage));
}

module.exports = {safeRelative, listState, packState, extractState, ciBuildJobs, ciPaths, provisionOneApi, verifyIoSource, verifyLinuxLlvmAdmissionCi, buildLinuxLlvmGroupCi,
  linuxLlvmCiStagePlan,
  admitLinuxLlvmOutput, linuxOpenSslTools, buildLinuxLlvmToolsCi, buildLinuxLlvmSdkHandoff,
  windowsNativeCoreStatePaths, windowsNativeCoreDescriptor, windowsNativeCoreDeadlineBudget, auditWindowsNativeCoreOutput, packWindowsNativeCoreHandoff, buildWindowsNativeCoreCi,
  windowsHostSdkStatePaths, windowsHostSdkDeadlineBudget, windowsHostSdkPackageMembers, buildWindowsHostSdkSourcesCi,
  windowsGpuHandoffStatePaths, windowsGpuProfileHandoffPlan, windowsGpuDeadlineBudget, windowsGpuProfileCiPreflight,
  packWindowsGpuHandoff, buildWindowsGpuProfileCi,
  windowsNativeLlvmCiStatePaths, windowsNativeLlvmCiRunDeadlineBudget, windowsNativeLlvmCiDeadlineBudget,
  windowsNativeLlvmCiDeadlineRecord, windowsNativeLlvmCiPreflight, buildWindowsNativeLlvmCi,
  windowsNativeLlvmDescriptorCiStatePaths, windowsNativeLlvmDescriptorCiPreflight, buildWindowsNativeLlvmDescriptorCi,
  validateWindowsLlvmLibraryPolicy, auditWindowsLlvmToolFile, mapWindowsLlvmSdkLibraries, composeWindowsLlvmSdkDescriptor, composeLocalWindowsLlvmSdkDescriptor,
  windowsControllerStatePaths, windowsControllerDeadlineBudget, windowsControllerTools, windowsControllerIoBuildCommands,
  windowsControllerHermesBuildCommand, windowsHermesSourceName, canonicalHermesOriginalRoot, readWindowsHermesSource, extractWindowsHermesSource,
  windowsControllerPackageInventory, windowsControllerIntelRuntimeEntries,
  copyWindowsControllerIntelRuntime, copyWindowsControllerPath,
  verifyWindowsHermesSourceHandoff, packWindowsControllerHandoff, buildWindowsControllerIoCi, buildWindowsControllerHermesCi,
  auditIoOpenSsl, ioOpenSslBuildCommands, ioOpenSslChildTimeout, linuxLlvmConfigureCommand, ioFinalTar, hermesStatePaths, assertHermesFetchProvenance, verifyHermesSource,
  acppStatePaths, assertAcppFetchProvenance, assertAcppPreparedSource, verifyAcppSource, fetchAcppSource,
  linuxSourceStatePaths, assertSdkFetchProvenance, assertPreparedSdkPaths, linuxSourceFetchArgs, linuxSourcePreparationPlan, verifyLinuxSources, fetchLinuxSources,
  dpcppCudaInputsStatePaths, dpcppSourceArchivePin, dpcppCudaInputPins, cudaPackageNames, auditCudaPackageCopyrights,
  copyDpcppBuildInputs, copyCudaBuildInputs, cudaToolkitProbeCMakeLists, cudaToolkitProbeRoot, probeCudaToolkit, inputFileRecord,
  preparedDpcppCudaSourceOrigin, dpcppCudaInputsManifest, auditDpcppCudaInputs, stageDpcppCudaInputs, buildDpcppCudaInputs,
  gnuHeadersStatePaths, buildLinuxGnuHeaders, buildWindowsGnuHeaders, auditGnuHeaders, capturePackageCopyright,
  linuxJitResourcesStatePaths, linuxJitResourceBuildPlan, linuxJitResourceProfile, buildLinuxJitResourceTree, assertLinuxJitResourceManifest,
  auditLinuxJitResourceTree, buildLinuxJitResources,
  linuxJitBootstrapOverlayPlan, seedLinuxJitBootstrapOverlay, auditLinuxJitBootstrapOverlay,
  linuxJitSdkStatePaths, linuxJitSdkArtifactPhase, linuxJitSdkInputManifest, linuxJitSdkBuildPlan, restoreLinuxJitSdkInputs,
  auditLinuxJitSdkConfigureArtifacts, auditLinuxJitCompilerState, auditLinuxJitSdkState, buildLinuxJitSdkPhase,
  linuxJitRuntimeProofPaths, packLinuxJitRuntimeProofFiles, extractLinuxJitRuntimeProof, buildLinuxJitStaticSdk,
  rocmParserSource, assertRocmParserReceipt, rocmSourceStatePaths, rocmSourceFetchArgs, verifyRocmParserSource, fetchRocmParserSource, verifyRocmSources, fetchRocmSources, rocmComgrStatePaths,
  rocmHostStatePaths, rocmHostBuildTools, rocmHostBuildConfig, provisionRocmHostPackages, admitRocmParserInstallation, installRocmParserSource,
  prepareRocmHostAdmission, auditRocmHostSdk, buildRocmHostSdk,
  rocmResourcesConfig, exportRocmSourceTree, rocmClrHipamdTreeish, applyRocmHipHeadersPatch, assembleRocmHipHeaders, hipHeaderConfigureCommand,
  rocmComgrBuildCommands, auditRocmComgrSdk, buildRocmComgrFromInputs, buildRocmComgrSdk,
  rocmHostNoticeRows, captureRocmHostNoticeSources, auditRocmHostNoticeSources,
  hermesBuildTools, hermesBuildCommand, assertHermesSdkFingerprints, auditHermesSdk, fetchHermesSource, buildHermesSdk,
  zstdStatePaths, linuxZstdTools, zstdBuildCommand, auditZstdSdk, fetchZstdSource, buildZstdSdk,
  openmpStatePaths, openmpBuildTools, openmpBuildCommand, auditOpenmpSdk, buildOpenmpSdk,
  acppCudaHipSdkStatePaths, acppCudaHipSdkBuildCommand, assertAcppCudaHipSdk, auditAcppCudaHipSdk, buildAcppCudaHipSdk,
  nvrtcStatePaths, cudaImagesStatePaths, cudaImageWorkerRecipe, cudaNoticeSourcePlan, retainCudaNoticeSources,
  auditCudaNoticeFiles, auditCudaNoticeSources, buildLinuxCudaImageTree, buildLinuxCudaImages,
  linuxNativeSdkDescriptorStatePaths, linuxNativeSdkLldPath, linuxNativeSdkDescriptorConfig, linuxNativeSdkGnuNoticeSources, linuxNativeSdkRocmNoticeSources, auditLinuxNativeSdkDescriptorInputArchive,
  auditLinuxNativeSdkDescriptorInputArchives, auditLinuxNativeSdkDescriptorCudaImages, buildLinuxNativeSdkDescriptor,
  tarExpandedBytes, linuxNativeSdkDiskPreflight, auditLinuxNativeSdkDescriptorLineage,
  linuxNativeGpuProfiles, linuxNativeComponentProfiles, linuxNativeComponentArchivePins, auditLinuxNativeComponentArchive,
  linuxNativeComponentStatePaths, linuxNativeComponentChildTimeout, linuxNativeAssemblyComponents, linuxNativeAssemblyConfig, linuxNativeComponentFiles,
  writeLinuxNativeComponentManifest, partitionLinuxNativeDownloads, restoreLinuxNativeDescriptorHandoff,
  buildLinuxNativeComponent, auditLinuxNativeComponentTree, restoreLinuxNativeComponents,
  linuxNativeApplicationStatePaths, linuxNativeApplicationBuildCommand, auditLinuxNativeApplicationEntrypoint, buildLinuxNativeApplication,
  staticNvrtcLlvmConfig, packNvrtcVerification, verifyNvrtcTarget, ciJobChildTimeout, buildLinuxLlvmCi, runStage, outputStageName,
  runNodeProcessGroup};

if (require.main === module) {
  try {
    const [command, archive, base, ...paths] = process.argv.slice(2);
    if (command === "pack" && archive && base && paths.length) {
      console.log(JSON.stringify(packState(archive, base, paths)));
    } else if (command === "extract" && archive && base && paths.length) {
      extractState(archive, base, paths);
    } else if (command === "stage" && archive) {
      runStage(process.argv.slice(3)).catch(error => {
        console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
        process.exitCode = 1;
      });
    } else {
      // Linux LLVM stages come from their table; their builder phases are not stages.
      const stages = ["sources", "acpp-sources", "linux-sources", "linux-llvm", "linux-llvm-admission", "linux-llvm-installed-sdk", "linux-llvm-tools",
        "linux-native-component <component>", "windows-native-core", "windows-host-sdk-sources", "windows-controller-io",
        "windows-controller-hermes", "windows-native-gpu <profile> <config.json> <config-sha256>",
        "windows-native-llvm21-descriptor <input.json> <sha256>", "dpcpp-cuda-inputs", "gnu-headers", "windows-gnu-headers",
        "linux-jit-resources", "linux-jit-configure", "linux-jit-support", "linux-jit-targets", "linux-jit-frontend", "linux-jit-link",
        "linux-jit-runtime-proof", "linux-jit-runtime-admit", "linux-jit-static-dpcpp", "rocm-sources", "rocm-comgr-sdk", "rocm-host-sdk",
        "hermes-sources", "hermes-sdk", "zstd-sources", "zstd-sdk", "openmp-sdk", "acpp-cuda-hip-sdk", "nvrtc-linux", "nvrtc-windows",
        "cuda-images-linux", "linux-sdk-descriptor", "linux-native-application", "io-openssl",
        "configure", ...Object.keys(linuxLlvmCiStageDefinitions), "install-audit"];
      throw new Error("Usage: prepare-native-ci.js stage " + stages.join("|") +
        " | pack /new/state.tar /base relative-path... | extract /state.tar /base expected-root...");
    }
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
