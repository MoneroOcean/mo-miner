"use strict";

// Native CI Linux work roots, pinned source fetches and their re-admission, DPC++/CUDA inputs and target GNU headers.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {appendWorkflowOutputs, canonicalExecutable, ciSuffix, hasExactKeys, ioOpenSslChildTimeout, jsonHash, runNode} = require("./native-ci-job");
const {fileHash, readBuildJson} = require("./build-source-inputs");
const {verifyIoArchives, gitSourceComponents, sourceRecipe, origins, verifyZstdArchive} = require("./fetch-native-sdk-sources");
const {hermesPins} = require("./build-native-hermes");
const {auditGnuHeaders} = require("./prepare-native-gnu-headers");
const {zstdPin} = require("./build-native-zstd");
const {sdkSourceEnvironment} = require("./build-environment");
const {syclRevision} = require("./native-sycl-inputs");
const {sdkComponents, prepareSdkSources, verifyPreparedSdkSources} = require("./prepare-native-sdk-sources");
const {stageDpcppCudaInputs} = require("./native-dpcpp-cuda-inputs");
const {archiveName: llvmArchiveName} = require("./native-llvm-sdk");
const {linuxNativeSdkLldPath} = require("./native-linux-sdk-config");
const {extractState, listState, packState, temp} = require("./native-ci-tar");
const repo = path.resolve(__dirname, "..");

const rocmParserSource = Object.freeze({url: "https://files.pythonhosted.org/packages/3c/ba/d8d168a4b54cae66eaf13d1d9197ca9349c94653815e061f79e7eed86c01/CppHeaderParser-2.7.4.tar.gz",
  sha256: "382b30416d95b0a5e8502b214810dcac2a56432917e2651447d3abe253e3cc42", bytes: 54445, directory: "CppHeaderParser-2.7.4"});

const rocmPythonDependencies = Object.freeze({ply: "3.11-6", xxd: "2:9.1.0016-1ubuntu7.20"});

const rocmOpenGLDependencies = Object.freeze({"libegl-dev": "1.7.0-1build1", "libopengl-dev": "1.7.0-1build1"});

/** @type {Readonly<Record<string,string>>} */
const rocmPinnedHostPackages = Object.freeze({"python3-ply": rocmPythonDependencies.ply, xxd: rocmPythonDependencies.xxd,
  ...rocmOpenGLDependencies});

/** @returns {{suffix: string, source: string, sourceName: string, ioSource: string, ioSourceName: string,
 * zstdSource: string, zstdSourceName: string, zstdSdkRoot: string, zstdSdkName: string,
 * linuxJitResourcesRoot: string, linuxJitResourcesName: string,
 * linuxSdkDescriptorRoot: string, linuxSdkDescriptorName: string,
 * linuxSource: string, linuxSourceName: string,
 * sdkRoot: string, sdkName: string, output: string, ioSdkRoot: string, ioSdkName: string,
 * ioSdk: string, opensslSdk: string, opensslTools: string, rocmComgrSdkRoot: string, rocmResourcesRoot: string,
 * rocmHipHeadersRoot: string, rocmHostSdkRoot: string, runnerTemp: string, env: NodeJS.ProcessEnv}} */
function ciPaths() {
  const runnerTemp = process.env["RUNNER_TEMP"] || "";
  if (!path.isAbsolute(runnerTemp) || process.platform !== "linux" || process.arch !== "x64") {
    throw new Error("Native Linux SDK CI needs RUNNER_TEMP and a Linux x64 runner");
  }
  const suffix = ciSuffix(), sourceName = "mom-native-prerequisites-" + suffix, sdkName = "mom-native-llvm21-sdk-" + suffix,
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
    sourceTar = path.join(paths.runnerTemp, sourcesTarName(paths)),
    toolsFile = path.join(paths.runnerTemp, "native-openmp-tools-" + paths.suffix + ".json");
  if (environment["MOM_NATIVE_OPENMP_SDK_DIR"] !== sdkRoot || !path.isAbsolute(paths.source) ||
      !path.isAbsolute(paths.runnerTemp) || fs.realpathSync(paths.runnerTemp) !== paths.runnerTemp) {
    throw new Error("Native OpenMP SDK root must exactly match this workflow run");
  }
  return {source: paths.source, sourceName: paths.sourceName, archive: path.join(paths.source, llvmArchiveName),
    sdkRoot, sdkName, sourceTar, incomingTar: path.join(paths.runnerTemp, "incoming", path.basename(sourceTar)),
    sdkTar: path.join(paths.runnerTemp, "native-linux-openmp-sdk-" + paths.suffix + ".tar"), toolsFile};
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

/** Both restored DPC++/CUDA inputs are externally sealed by the producer job's outputs.
 * @param {NodeJS.ProcessEnv} environment @returns {{bundleSha256:string,manifestSha256:string}} */
function dpcppCudaInputPins(environment) {
  const bundleSha256 = environment["MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256"] || "",
    manifestSha256 = environment["MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256"] || "";
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

/** The LLVM/OpenSSL and libuv/zlib source state tar. @param {{suffix: string}} paths */
function sourcesTarName(paths) {return "native-llvm21-state-" + paths.suffix + "-sources.tar";}

/** Restore the pinned source state, verify it, then remove the consumed transfer file to leave room
 * for the producer's own state tar.
 * @param {ReturnType<typeof ciPaths>} paths @param {string} [deadlineEpoch] */
function restoreSources(paths, deadlineEpoch) {
  const file = path.join(paths.runnerTemp, "incoming", sourcesTarName(paths)), roots = [paths.sourceName, paths.ioSourceName];
  extractState(file, temp, roots, roots, deadlineEpoch ? ioOpenSslChildTimeout(deadlineEpoch, 30 * 60 * 1000, Date.now(), 5 * 60 * 1000) : undefined);
  runNode("scripts/fetch-native-sdk-sources.js", ["verify-build", paths.source], paths.env,
    deadlineEpoch ? ioOpenSslChildTimeout(deadlineEpoch, 2 * 60 * 1000, Date.now(), 5 * 60 * 1000) : 2 * 60 * 1000);
  verifyIoSource(paths);
  fs.unlinkSync(file);
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

/** Fetch only the original pinned source archive and preserve its immutable receipt at the same /tmp path. @param {ReturnType<typeof ciPaths>} paths */
function fetchZstdSource(paths) {
  const state = zstdStatePaths(paths);
  runNode("scripts/fetch-native-sdk-sources.js", ["zstd", state.source, "--allow-network"], paths.env, 15 * 60 * 1000);
  verifyZstdArchive(state.source);
  packState(state.sourceTar, temp, [state.sourceName]);
}

module.exports = {sourcesTarName, acppStatePaths, assertAcppFetchProvenance, assertAcppPreparedSource, assertHermesFetchProvenance,
  assertPreparedSdkPaths, assertRocmParserReceipt, assertSdkFetchProvenance, buildDpcppCudaInputs, buildLinuxGnuHeaders,
  buildWindowsGnuHeaders, ciPaths, dpcppCudaInputPins, dpcppCudaInputsStatePaths, dpcppSourceArchivePin, fetchAcppSource,
  fetchHermesSource, fetchLinuxSources, fetchRocmParserSource, fetchRocmSources, fetchZstdSource, gnuHeadersStatePaths,
  hermesStatePaths, linuxSourceFetchArgs, linuxSourcePreparationPlan, linuxSourceStatePaths, openmpStatePaths, provisionOneApi,
  restoreSources, rocmParserSource, rocmPinnedHostPackages, rocmSourceFetchArgs, rocmSourceStatePaths, verifyAcppSource,
  verifyHermesSource, verifyIoSource, verifyLinuxSources, verifyRocmParserSource, verifyRocmSources, zstdStatePaths};
