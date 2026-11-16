"use strict";

// Shared admission for the small installed DPC++/CUDA input handoff. CI owns tar
// transport and provisioning; local and CI callers use this same producer/auditor.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"), {createHash} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson} = require("./build-source-inputs");
const {capturePackageCopyright} = require("./prepare-native-gnu-headers");
const {pins: dpcppPins, sdkIdentity: dpcppSdkIdentity} = require("./probe-static-dpcpp");
const {sdkSourceFiles} = require("./prepare-native-sdk-sources");
const {isStaticArchive} = require("./native-audit");

const repo = path.resolve(__dirname, "..");
const cudaPackageNames = ["cuda-cudart-12-6", "cuda-cudart-dev-12-6", "cuda-nvml-dev-12-6"];

/** @typedef {import("./prepare-native-ci").LinuxPreparedSources & {sourceName:string,sourceTar?:string}} DpcppPreparedLinuxSourceState */
/** @typedef {{workspaceRoot?:string,name:string,root:string,dpcppRoot:string,cudaRoot:string,manifest:string,cudaVersionEvidence:string,cudaProbeEvidence:string,
 * cudaCmakeSource:string,cudaCmakeBuild:string,cudaCmakeLists:string,cudaCmakeEvidence:string,linux:DpcppPreparedLinuxSourceState}} DpcppCudaInputState */
/** @typedef {{kind:"prepared-receipts",workspaceRoot:string,fetchReceipt:{path:string,sha256:string},linuxPreparation:{path:string,sha256:string},
 * compilerPreparation:{path:string,sha256:string}}} PreparedDpcppCudaSourceOrigin */
/** @typedef {string|{kind:"source-tar",sha256:string}|PreparedDpcppCudaSourceOrigin} DpcppCudaSourceOrigin */

/** @param {string} value */
function jsonHash(value) {return createHash("sha256").update(value).digest("hex");}

/** @param {unknown} value @param {string[]} keys */
function hasExactKeys(value, keys) {
  return !!value && typeof value === "object" && !Array.isArray(value) &&
    Object.keys(value).sort().join("\0") === [...keys].sort().join("\0");
}

/** @param {string} name */
function safeRelative(name) {
  return typeof name === "string" && name.length > 0 && name.length <= 4096 && !name.startsWith("/") &&
    !name.includes("\\") && !/[\r\n\0]/.test(name) && name.split("/").every(part => part && part !== "." && part !== "..");
}

/** @param {string} file */
function canonicalExecutable(file) {
  const selected = fs.realpathSync(file), stat = fs.statSync(selected);
  if (!stat.isFile()) {throw new Error("Missing native CI executable: " + file);}
  fs.accessSync(selected, fs.constants.X_OK);
  return selected;
}

/** Stage the SDK without changing its compiler alias chain. Selected-input admission
 * still rejects aliases that escape the staged root.
 * @param {string} sourceRoot @param {string} outputRoot */
function copyDpcppBuildInputs(sourceRoot, outputRoot) {
  if (!path.isAbsolute(sourceRoot) || !path.isAbsolute(outputRoot) || fs.realpathSync(sourceRoot) !== sourceRoot ||
      !fs.statSync(sourceRoot).isDirectory() || path.resolve(outputRoot) !== outputRoot ||
      fs.realpathSync(path.dirname(outputRoot)) !== path.dirname(outputRoot) || fs.lstatSync(outputRoot, {throwIfNoEntry: false}) ||
      sourceRoot === outputRoot || outputRoot.startsWith(sourceRoot + path.sep) || sourceRoot.startsWith(outputRoot + path.sep)) {
    throw new Error("DPC++ SDK copy needs canonical disjoint roots and a fresh output");
  }
  fs.cpSync(sourceRoot, outputRoot, {recursive: true, errorOnExist: true, force: false, verbatimSymlinks: true});
}

/** Resolve the bounded CUDA input closure used by DPC++ and the source-built ACPP CUDA backend.
 * Follow package symlinks only inside the pinned SDK root; omit install-dev's external NVRTC header link.
 * @param {string} sourceRoot @param {string} outputRoot */
function copyCudaBuildInputs(sourceRoot, outputRoot) {
  const root = fs.realpathSync(sourceRoot);
  /** @type {Array<{path:string,target:string}>} */ const excluded = [];
  if (root !== sourceRoot || !fs.lstatSync(sourceRoot).isDirectory() || fs.lstatSync(outputRoot, {throwIfNoEntry: false})) {
    throw new Error("CUDA input copy requires the exact original SDK root and a new output");
  }
  /** @param {string} relative @param {Set<string>} ancestors */
  function copy(relative, ancestors = new Set()) {
    if (!safeRelative(relative)) {throw new Error("Unsafe CUDA input path");}
    const source = path.join(sourceRoot, ...relative.split("/")), destination = path.join(outputRoot, ...relative.split("/")), link = fs.lstatSync(source);
    if (link.isSymbolicLink()) {
      const target = fs.realpathSync(source), rel = path.relative(root, target);
      // install-dev adds only this NVRTC header as a link to its separate runtime package.
      if (relative === "include/nvrtc.h" && target === "/opt/mom/nvrtc-13.3.33/include/nvrtc.h") {
        excluded.push({path: relative, target}); return;
      }
      if (path.isAbsolute(rel) || rel === ".." || rel.startsWith(".." + path.sep)) {throw new Error("CUDA input symlink escapes the pinned SDK: " + relative);}
    }
    const real = fs.realpathSync(source), stat = fs.statSync(real), rel = path.relative(root, real);
    if (path.isAbsolute(rel) || rel === ".." || rel.startsWith(".." + path.sep)) {throw new Error("CUDA input escaped its SDK root: " + relative);}
    if (stat.isDirectory()) {
      if (ancestors.has(real)) {throw new Error("CUDA input directory symlink cycle: " + relative);}
      fs.mkdirSync(destination, {recursive: true, mode: 0o755});
      const next = new Set(ancestors); next.add(real);
      for (const name of fs.readdirSync(real).sort()) {copy(relative + "/" + name, next);}
    } else if (stat.isFile()) {
      fs.mkdirSync(path.dirname(destination), {recursive: true, mode: 0o755});
      fs.copyFileSync(real, destination, fs.constants.COPYFILE_EXCL);
      fs.chmodSync(destination, stat.mode & 0o755);
    } else {throw new Error("CUDA input is not a regular file or directory: " + relative);}
  }
  fs.mkdirSync(outputRoot, {mode: 0o755});
  for (const relative of ["bin/nvcc", "bin/nvcc.profile", "bin/ptxas", "bin/fatbinary", "include", "nvvm/libdevice/libdevice.10.bc", "lib64/libcudart.so",
    "lib64/libcudart_static.a", "lib64/libculibos.a"]) {copy(relative);}
  return excluded;
}

/** Check every selected CUDA package's transported terms, not just CUDART's similar text.
 * Package-owned doc-directory aliases are captured at origin and retained as data only.
 * @param {{root:string}} state @param {Record<string,string>} packages
 * @param {ReturnType<typeof capturePackageCopyright>[]} notices */
function auditCudaPackageCopyrights(state, packages, notices) {
  if (!Array.isArray(notices) || notices.length !== cudaPackageNames.length) {throw new Error("CUDA package copyrights are incomplete");}
  for (const [index, row] of notices.entries()) {
    const name = cudaPackageNames[index];
    if (!name) {throw new Error("Unexpected CUDA package copyright");}
    const sourcePath = "/usr/share/doc/" + name + "/copyright",
      destination = path.join(state.root, "licenses", name, "copyright"), alias = row?.resolvedSourcePath !== undefined,
      stat = fs.lstatSync(destination, {throwIfNoEntry: false});
    if (!hasExactKeys(row, ["package", "version", "sourcePath", "path", "sha256", ...(alias ? ["resolvedSourcePath", "ownershipPath"] : [])]) ||
        row.package !== name || row.version !== packages[name] || row.sourcePath !== sourcePath || row.path !== destination ||
        !/^[a-f0-9]{64}$/.test(row.sha256 || "") || !stat?.isFile() || stat.isSymbolicLink() ||
        fs.realpathSync(destination) !== destination || stat.size < 1 || stat.size > (2 << 20) || fileHash(destination) !== row.sha256 ||
        alias && (typeof row.resolvedSourcePath !== "string" || !row.resolvedSourcePath.startsWith("/usr/share/doc/") ||
          path.normalize(row.resolvedSourcePath) !== row.resolvedSourcePath || row.ownershipPath !== path.dirname(sourcePath))) {
      throw new Error("CUDA package copyright path, identity or body is missing or changed: " + name);
    }
  }
}

/** A configure-only probe proves the pruned root satisfies the actual CMake CUDA package contract. */
function cudaToolkitProbeCMakeLists() {
  return `cmake_minimum_required(VERSION 3.20)
project(mom_dpcpp_cuda_input_probe NONE)
find_package(CUDAToolkit 10.1 REQUIRED)
if(NOT TARGET CUDA::cudart)
  message(FATAL_ERROR "FindCUDAToolkit did not create CUDA::cudart")
endif()
get_target_property(_mom_cudart CUDA::cudart IMPORTED_LOCATION)
if(NOT _mom_cudart)
  get_target_property(_mom_cudart CUDA::cudart IMPORTED_IMPLIB)
endif()
if(NOT _mom_cudart OR NOT EXISTS "\${_mom_cudart}")
  message(FATAL_ERROR "CUDA::cudart has no usable selected link input")
endif()
file(REAL_PATH "\${CUDA_CUDART}" _mom_cudart_found)
file(REAL_PATH "\${_mom_cudart}" _mom_cudart_imported)
file(REAL_PATH "\${CUDAToolkit_ROOT}/lib64/libcudart.so" _mom_cudart_selected)
if(NOT _mom_cudart_found STREQUAL _mom_cudart_selected OR NOT _mom_cudart_imported STREQUAL _mom_cudart_selected)
  message(FATAL_ERROR "CMake selected a CUDART outside the run-scoped SDK")
endif()
if(NOT EXISTS "\${CUDAToolkit_ROOT}/include/cuda.h" OR NOT EXISTS "\${CUDAToolkit_ROOT}/include/nvml.h" OR NOT EXISTS "\${CUDAToolkit_ROOT}/include/cuda_runtime.h")
  message(FATAL_ERROR "CMake CUDA root is missing required CUDA/NVML headers")
endif()
file(WRITE "\${CMAKE_BINARY_DIR}/mom-cuda-toolkit-result.txt" "\${CUDAToolkit_VERSION}\\n\${CUDA_CUDART}\\n\${_mom_cudart}\\n\${CUDAToolkit_INCLUDE_DIRS}\\n")
message(STATUS "DPCPP_CUDA_TOOLKIT_PROBE=PASS\\n\${CUDAToolkit_VERSION}\\n\${CUDA_CUDART}\\n\${_mom_cudart}\\n\${CUDAToolkit_INCLUDE_DIRS}")
`;
}

/** Run CMake's real FindCUDAToolkit in configure-only mode and retain bounded command/result evidence.
 * @param {DpcppCudaInputState} state @param {{env:NodeJS.ProcessEnv}} paths */
function probeCudaToolkit(state, paths) {
  const cmake = canonicalExecutable("/usr/bin/cmake"), command = [cmake, "-S", state.cudaCmakeSource, "-B", state.cudaCmakeBuild,
      "-DCUDAToolkit_ROOT=" + state.cudaRoot], script = cudaToolkitProbeCMakeLists();
  if (fs.lstatSync(state.cudaCmakeBuild, {throwIfNoEntry: false}) || fs.existsSync(state.cudaCmakeLists) ||
      !path.isAbsolute(state.cudaCmakeSource)) {throw new Error("CUDA toolkit smoke output already exists or is not canonical");}
  fs.mkdirSync(state.cudaCmakeSource, {mode: 0o700});
  fs.writeFileSync(state.cudaCmakeLists, script, {flag: "wx", mode: 0o600});
  try {
    const result = spawnSync(cmake, command.slice(1), {cwd: repo, env: paths.env, encoding: "utf8", timeout: 60_000, maxBuffer: 8 << 20}),
      output = (result.stdout || "") + (result.stderr || ""), resultFile = path.join(state.cudaCmakeBuild, "mom-cuda-toolkit-result.txt");
    if (result.error || result.signal || result.status !== 0) {
      fs.writeFileSync(state.cudaCmakeEvidence, JSON.stringify({version: 1, command, status: result.status, signal: result.signal,
        error: result.error?.message || null, stdout: result.stdout || "", stderr: result.stderr || ""}, null, 2) + "\n", {flag: "wx", mode: 0o600});
      console.error(output.slice(-8192));
      throw new Error("Pruned CUDA inputs failed the actual CMake FindCUDAToolkit smoke", {cause: result.error});
    }
    const fields = fs.readFileSync(resultFile, "utf8").replace(/\r/g, "").trimEnd().split("\n");
    if (fields.length !== 4 || !/^12\.6\./.test(fields[0] || "") || fields[1] !== path.join(state.cudaRoot, "lib64/libcudart.so") ||
        fields[2] !== fields[1] || !(fields[3] || "").split(";").includes(path.join(state.cudaRoot, "include")) ||
        !output.includes("DPCPP_CUDA_TOOLKIT_PROBE=PASS")) {throw new Error("CMake CUDA toolkit probe selected incomplete or ambient inputs");}
    const evidence = {version: 1, command, status: 0, tool: {path: cmake, sha256: fileHash(cmake)},
      script: {path: state.cudaCmakeLists, sha256: fileHash(state.cudaCmakeLists)},
      result: {toolkitVersion: fields[0], cudart: fields[1], importedCudart: fields[2], includeDirectories: fields[3]},
      stdout: result.stdout || "", stderr: result.stderr || ""};
    fs.writeFileSync(state.cudaCmakeEvidence, JSON.stringify(evidence, null, 2) + "\n", {flag: "wx", mode: 0o600});
    return evidence;
  } finally {fs.rmSync(state.cudaCmakeBuild, {recursive: true, force: true});}
}

/** @param {string} root @param {string} relative @param {boolean} [executable] */
function inputFileRecord(root, relative, executable = false) {
  if (!safeRelative(relative)) {throw new Error("Invalid selected SDK input path");}
  if (!fs.lstatSync(root).isDirectory() || fs.realpathSync(root) !== root) {throw new Error("Selected SDK root must be a real canonical directory: " + root);}
  const file = path.join(root, ...relative.split("/")), real = fs.realpathSync(file), rel = path.relative(fs.realpathSync(root), real), stat = fs.statSync(file);
  if (path.isAbsolute(rel) || rel === ".." || rel.startsWith(".." + path.sep) || !stat.isFile() ||
      (executable && (stat.mode & 0o111) === 0)) {throw new Error("Selected SDK input is missing, unsafe, or not executable: " + relative);}
  return {path: file, sha256: fileHash(file)};
}

/** Bind workspace source receipts without inventing a source-tar identity.
 * @param {Pick<DpcppCudaInputState["linux"], "fetchReceipt"|"preparedReceipt"|"compilerPreparedReceipt">} state
 * @param {string} workspaceRoot @returns {PreparedDpcppCudaSourceOrigin} */
function preparedDpcppCudaSourceOrigin(state, workspaceRoot) {
  if (typeof workspaceRoot !== "string" || !path.isAbsolute(workspaceRoot) || path.resolve(workspaceRoot) !== workspaceRoot) {
    throw new Error("Prepared DPC++/CUDA source origin needs a canonical marked workspace root");
  }
  const rootStat = fs.lstatSync(workspaceRoot, {throwIfNoEntry: false}), marker = path.join(workspaceRoot, ".mom-native-sdk-workspace.json"),
    markerStat = fs.lstatSync(marker, {throwIfNoEntry: false});
  if (!rootStat?.isDirectory() ||
      rootStat.isSymbolicLink() || fs.realpathSync(workspaceRoot) !== workspaceRoot || !markerStat?.isFile() || markerStat.isSymbolicLink() ||
      fs.realpathSync(marker) !== marker || fs.readFileSync(marker, "utf8") !== JSON.stringify({version: 1, checkout: repo}) + "\n") {
    throw new Error("Prepared DPC++/CUDA source origin needs a canonical marked workspace root");
  }
  /** @param {string} file @param {string} label */
  const receipt = (file, label) => {
    const stat = fs.lstatSync(file, {throwIfNoEntry: false}), relative = path.relative(workspaceRoot, file);
    if (!path.isAbsolute(file) || path.resolve(file) !== file || Buffer.byteLength(file) > 4096 || /[\r\n\0]/.test(file) ||
        path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep) || !stat?.isFile() ||
        stat.isSymbolicLink() || stat.nlink !== 1 || stat.size > (128 << 20) || fs.realpathSync(file) !== file) {
      throw new Error("Prepared DPC++/CUDA source receipt is unsafe: " + label);
    }
    return {path: file, sha256: fileHash(file)};
  };
  return {kind: "prepared-receipts", workspaceRoot,
    fetchReceipt: receipt(state.fetchReceipt, "fetch"),
    linuxPreparation: receipt(state.preparedReceipt, "Linux"),
    compilerPreparation: receipt(state.compilerPreparedReceipt, "compiler")};
}

/** Capture exact source, installer, compiler, CUDA header, and minimal CMake/device-resource provenance.
 * @param {DpcppCudaInputState} state @param {DpcppCudaSourceOrigin} sourceOrigin
 * @param {Record<string,string>} packageVersions @param {{version:string,top:string,dpcppVersion:string}} cudaProbe
 * @param {ReturnType<typeof capturePackageCopyright>[]} cudaCopyrights @param {Array<{path:string,target:string}>} excluded */
function dpcppCudaInputsManifest(state, sourceOrigin, packageVersions, cudaProbe, cudaCopyrights, excluded) {
  const installer = path.join(repo, "scripts/install-dev.sh"), text = fs.readFileSync(installer, "utf8"),
    release = text.match(/^DPCPP_RELEASE=\$\{MOM_DPCPP_RELEASE-([^}]+)\}$/m)?.[1], dpcppSha = text.match(/^DPCPP_SHA256=\$\{MOM_DPCPP_SHA256-([0-9a-f]{64})\}$/m)?.[1],
    cudaKey = text.match(/^CUDA_KEYRING_SHA256=([0-9a-f]{64})$/m)?.[1], cudaSeries = text.match(/^CUDA_VERSION=\$\{MOM_CUDA_VERSION-([0-9]+-[0-9]+)\}$/m)?.[1],
    cmakeEvidence = readBuildJson(state.cudaCmakeEvidence);
  const source = typeof sourceOrigin === "string" || sourceOrigin?.kind === "source-tar" ?
    {name: state.linux.sourceName, tar: path.basename(state.linux.sourceTar || ""),
      tarSha256: typeof sourceOrigin === "string" ? sourceOrigin : sourceOrigin.sha256,
      fetchReceiptSha256: fileHash(state.linux.fetchReceipt), linuxPreparationSha256: fileHash(state.linux.preparedReceipt),
      compilerPreparationSha256: fileHash(state.linux.compilerPreparedReceipt)} :
    sourceOrigin?.kind === "prepared-receipts" ? {origin: "workspace-prepared-receipts",
      workspaceRoot: sourceOrigin.workspaceRoot, fetchReceipt: sourceOrigin.fetchReceipt,
      linuxPreparation: sourceOrigin.linuxPreparation, compilerPreparation: sourceOrigin.compilerPreparation} : null;
  if (!release || dpcppSha !== dpcppPins.sdk || !cudaKey || cudaSeries !== "12-6" ||
      !source || !/^[0-9a-f]{64}$/.test(source.tarSha256 || "") && source.origin !== "workspace-prepared-receipts") {
    throw new Error("Pinned DPC++/CUDA installer provenance is incomplete or drifted");
  }
  const marker = path.join(state.dpcppRoot, ".mom-toolchain-sha256");
  dpcppSdkIdentity(cudaProbe.dpcppVersion, fs.readFileSync(marker, "utf8"));
  const nvccVersion = cudaProbe.version;
  if (!/\brelease 12\.6,/.test(nvccVersion) || cudaProbe.top !== state.cudaRoot ||
      !hasExactKeys(packageVersions, cudaPackageNames) ||
      Object.values(packageVersions).some(value => typeof value !== "string" || !/^(?:[0-9]+:)?12\.6\.[A-Za-z0-9.+~:-]{1,120}$/.test(value)) ||
      cmakeEvidence.status !== 0 || !hasExactKeys(cmakeEvidence.result, ["toolkitVersion", "cudart", "importedCudart", "includeDirectories"]) ||
      typeof cmakeEvidence.result.toolkitVersion !== "string" ||
      cmakeEvidence.result.toolkitVersion !== "12.6" && !cmakeEvidence.result.toolkitVersion.startsWith("12.6.") ||
      cmakeEvidence.result.cudart !== path.join(state.cudaRoot, "lib64/libcudart.so") ||
      cmakeEvidence.result.importedCudart !== cmakeEvidence.result.cudart ||
      typeof cmakeEvidence.result.includeDirectories !== "string" ||
      !cmakeEvidence.result.includeDirectories.split(";").includes(path.join(state.cudaRoot, "include"))) {
    throw new Error("CUDA compiler query does not identify the run-scoped CUDA 12.6 root");
  }
  const headers = sdkSourceFiles(path.join(state.cudaRoot, "include")), packages = {...packageVersions};
  auditCudaPackageCopyrights(state, packages, cudaCopyrights);
  return {version: 1, target: "linux-dpcpp23-cuda12.6-inputs", buildOnly: true, compiled: false, executed: false, publicationQualified: false,
    scope: "Pinned DPC++ compiler plus CUDA 12.6 headers, nvcc configure probe and ptxas/fatbinary inputs, CUDART and libdevice; exact static CUDA archives are carried for the source-built ACPP runtime, which is built separately.",
    source,
    producer: {installer: {path: installer, sha256: fileHash(installer), dpcppRelease: release, dpcppArchiveSha256: dpcppSha,
      cudaSeries: cudaSeries.replace("-", "."), cudaKeyringSha256: cudaKey}, helperSha256: fileHash(__filename),
    consumerSha256: fileHash(path.join(repo, "scripts/build-native-dpcpp-sdk.js"))},
    dpcpp: {root: state.dpcppRoot, archiveSha256: dpcppPins.sdk, marker: inputFileRecord(state.dpcppRoot, ".mom-toolchain-sha256"),
      clang: inputFileRecord(state.dpcppRoot, "bin/clang", true), clangxx: inputFileRecord(state.dpcppRoot, "bin/clang++", true),
      sycl: inputFileRecord(state.dpcppRoot, "lib/libsycl.so.9"), jit: inputFileRecord(state.dpcppRoot, "lib/libsycl-jit.so"), clangVersion: cudaProbe.dpcppVersion.trim()},
    cuda: {root: state.cudaRoot, series: "12.6", packages, nvcc: inputFileRecord(state.cudaRoot, "bin/nvcc", true),
      profile: inputFileRecord(state.cudaRoot, "bin/nvcc.profile"),
      ptxas: inputFileRecord(state.cudaRoot, "bin/ptxas", true), fatbinary: inputFileRecord(state.cudaRoot, "bin/fatbinary", true),
      nvccVersion, cmakeRoot: cudaProbe.top,
      versionEvidence: {path: state.cudaVersionEvidence, sha256: fileHash(state.cudaVersionEvidence)},
      cmakeProbeEvidence: {path: state.cudaProbeEvidence, sha256: fileHash(state.cudaProbeEvidence)},
      cmakeLists: {path: state.cudaCmakeLists, sha256: fileHash(state.cudaCmakeLists)},
      cmakeEvidence: {path: state.cudaCmakeEvidence, sha256: fileHash(state.cudaCmakeEvidence)},
      headers: {files: headers, sha256: jsonHash(JSON.stringify(headers))}, licenses: cudaCopyrights,
      libdevice: inputFileRecord(state.cudaRoot, "nvvm/libdevice/libdevice.10.bc"), cudart: inputFileRecord(state.cudaRoot, "lib64/libcudart.so"),
      staticRuntime: {cudart: inputFileRecord(state.cudaRoot, "lib64/libcudart_static.a"),
        culibos: inputFileRecord(state.cudaRoot, "lib64/libculibos.a")}, excluded}};
}

/** @param {DpcppCudaInputState} state @param {any} source */
function validDpcppCudaSourceOrigin(state, source) {
  const tarKeys = ["name", "tar", "tarSha256", "fetchReceiptSha256", "linuxPreparationSha256", "compilerPreparationSha256"];
  if (hasExactKeys(source, tarKeys)) {
    return typeof state.linux.sourceTar === "string" && source.name === state.linux.sourceName && source.tar === path.basename(state.linux.sourceTar) &&
      [source.tarSha256, source.fetchReceiptSha256, source.linuxPreparationSha256, source.compilerPreparationSha256].every(value => /^[0-9a-f]{64}$/.test(value || "")) &&
      source.fetchReceiptSha256 === fileHash(state.linux.fetchReceipt) && source.linuxPreparationSha256 === fileHash(state.linux.preparedReceipt) &&
      source.compilerPreparationSha256 === fileHash(state.linux.compilerPreparedReceipt);
  }
  if (!hasExactKeys(source, ["origin", "workspaceRoot", "fetchReceipt", "linuxPreparation", "compilerPreparation"]) ||
      source.origin !== "workspace-prepared-receipts" || state.workspaceRoot !== source.workspaceRoot) {return false;}
  try {
    // Use the producer's path/marker checks again; a second implementation would drift.
    const {kind, ...expected} = preparedDpcppCudaSourceOrigin(state.linux, source.workspaceRoot);
    return kind === "prepared-receipts" && isDeepStrictEqual(source, {origin: "workspace-prepared-receipts", ...expected});
  } catch {return false;}
}

/** Re-admit the small handoff without treating it as a completed DPC++ runtime SDK.
 * Historical installer/helper digests require an external same-run artifact and manifest pin.
 * @param {DpcppCudaInputState} state @param {any} record @param {string} [expectedManifestSha256] */
function auditDpcppCudaInputs(state, record, expectedManifestSha256) {
  for (const directory of [state.root, state.dpcppRoot, state.cudaRoot, path.join(state.cudaRoot, "include")]) {
    if (!fs.lstatSync(directory).isDirectory() || fs.realpathSync(directory) !== directory) {
      throw new Error("DPC++/CUDA handoff roots must be real canonical directories: " + directory);
    }
  }
  const externallyPinned = expectedManifestSha256 !== undefined;
  if (externallyPinned) {
    const manifestStat = fs.lstatSync(state.manifest, {throwIfNoEntry: false});
    if (!/^[a-f0-9]{64}$/.test(expectedManifestSha256) || !manifestStat?.isFile() || manifestStat.isSymbolicLink() ||
        fs.realpathSync(state.manifest) !== state.manifest || fileHash(state.manifest) !== expectedManifestSha256) {
      throw new Error("DPC++/CUDA manifest differs from its external same-run pin");
    }
    const sealedRecord = readBuildJson(state.manifest);
    if (!isDeepStrictEqual(record, sealedRecord) || fileHash(state.manifest) !== expectedManifestSha256) {
      throw new Error("DPC++/CUDA record does not match its externally pinned manifest");
    }
  }
  const producerFile = path.join(repo, "scripts/build-native-dpcpp-sdk.js"), installer = path.join(repo, "scripts/install-dev.sh"), installerText = fs.readFileSync(installer, "utf8"),
    release = installerText.match(/^DPCPP_RELEASE=\$\{MOM_DPCPP_RELEASE-([^}]+)\}$/m)?.[1],
    keyring = installerText.match(/^CUDA_KEYRING_SHA256=([0-9a-f]{64})$/m)?.[1];
  if (!hasExactKeys(record, ["version", "target", "buildOnly", "compiled", "executed", "publicationQualified", "scope", "source", "producer", "dpcpp", "cuda"]) ||
      record.version !== 1 || record.target !== "linux-dpcpp23-cuda12.6-inputs" || record.buildOnly !== true || record.compiled !== false ||
      record.executed !== false || record.publicationQualified !== false ||
      record.scope !== "Pinned DPC++ compiler plus CUDA 12.6 headers, nvcc configure probe and ptxas/fatbinary inputs, CUDART and libdevice; exact static CUDA archives are carried for the source-built ACPP runtime, which is built separately." ||
      !validDpcppCudaSourceOrigin(state, record.source) ||
      !hasExactKeys(record.producer, ["installer", "helperSha256", "consumerSha256"]) ||
      !/^[a-f0-9]{64}$/.test(record.producer.helperSha256 || "") ||
      !externallyPinned && record.producer.helperSha256 !== fileHash(__filename) ||
      !/^[a-f0-9]{64}$/.test(record.producer.consumerSha256 || "") || record.producer.consumerSha256 !== fileHash(producerFile) ||
      !hasExactKeys(record.producer.installer, ["path", "sha256", "dpcppRelease", "dpcppArchiveSha256", "cudaSeries", "cudaKeyringSha256"]) ||
      record.producer.installer.path !== installer || !/^[a-f0-9]{64}$/.test(record.producer.installer.sha256 || "") ||
      !externallyPinned && record.producer.installer.sha256 !== fileHash(installer) ||
      record.producer.installer.dpcppRelease !== release || record.producer.installer.dpcppArchiveSha256 !== dpcppPins.sdk ||
      record.producer.installer.cudaSeries !== "12.6" || record.producer.installer.cudaKeyringSha256 !== keyring ||
      !hasExactKeys(record.dpcpp, ["root", "archiveSha256", "marker", "clang", "clangxx", "sycl", "jit", "clangVersion"]) ||
      record.dpcpp.root !== state.dpcppRoot || record.dpcpp.archiveSha256 !== dpcppPins.sdk ||
      !hasExactKeys(record.cuda, ["root", "series", "packages", "nvcc", "profile", "ptxas", "fatbinary", "nvccVersion", "cmakeRoot", "versionEvidence", "cmakeProbeEvidence",
        "cmakeLists", "cmakeEvidence", "headers", "licenses", "libdevice", "cudart", "staticRuntime", "excluded"]) ||
      record.cuda.root !== state.cudaRoot || record.cuda.series !== "12.6" || record.cuda.cmakeRoot !== state.cudaRoot ||
      !hasExactKeys(record.cuda.staticRuntime, ["cudart", "culibos"]) ||
      !/\brelease 12\.6,/.test(record.cuda.nvccVersion || "") ||
      !hasExactKeys(record.cuda.versionEvidence, ["path", "sha256"]) || record.cuda.versionEvidence.path !== state.cudaVersionEvidence ||
      record.cuda.versionEvidence.sha256 !== fileHash(state.cudaVersionEvidence) || fs.readFileSync(state.cudaVersionEvidence, "utf8") !== record.cuda.nvccVersion ||
      !hasExactKeys(record.cuda.cmakeProbeEvidence, ["path", "sha256"]) || record.cuda.cmakeProbeEvidence.path !== state.cudaProbeEvidence ||
      record.cuda.cmakeProbeEvidence.sha256 !== fileHash(state.cudaProbeEvidence) ||
      !hasExactKeys(record.cuda.cmakeLists, ["path", "sha256"]) || record.cuda.cmakeLists.path !== state.cudaCmakeLists ||
      record.cuda.cmakeLists.sha256 !== fileHash(state.cudaCmakeLists) ||
      !hasExactKeys(record.cuda.cmakeEvidence, ["path", "sha256"]) || record.cuda.cmakeEvidence.path !== state.cudaCmakeEvidence ||
      record.cuda.cmakeEvidence.sha256 !== fileHash(state.cudaCmakeEvidence) ||
      !hasExactKeys(record.cuda.packages, cudaPackageNames) ||
      Object.values(record.cuda.packages).some(value => typeof value !== "string" || !/^(?:[0-9]+:)?12\.6\.[A-Za-z0-9.+~:-]{1,120}$/.test(value)) ||
      !Array.isArray(record.cuda.excluded) || record.cuda.excluded.some((/** @type {{path?:unknown,target?:unknown}} */ item) => !hasExactKeys(item, ["path", "target"]) ||
        item.path !== "include/nvrtc.h" || item.target !== "/opt/mom/nvrtc-13.3.33/include/nvrtc.h")) {
    throw new Error("DPC++/CUDA input handoff provenance is malformed or stale");
  }
  auditCudaPackageCopyrights(state, record.cuda.packages, record.cuda.licenses);
  const marker = fs.readFileSync(path.join(state.dpcppRoot, ".mom-toolchain-sha256"), "utf8");
  dpcppSdkIdentity(record.dpcpp.clangVersion, marker);
  // Probe scratch may move with RUNNER_TEMP; unlike the SDK root, a pinned producer path
  // need only stay canonical, bounded and tied to this run. Unpinned admission stays local.
  const cmakeEvidence = readBuildJson(state.cudaCmakeEvidence), cmake = cmakeEvidence?.tool?.path,
    cmakeBuild = externallyPinned ? cmakeEvidence?.command?.[4] : state.cudaCmakeBuild,
    validCmakeBuild = typeof cmakeBuild === "string" && path.isAbsolute(cmakeBuild) && Buffer.byteLength(cmakeBuild) <= 4096 &&
      !/[\r\n\0]/.test(cmakeBuild) && path.normalize(cmakeBuild) === cmakeBuild &&
      path.basename(cmakeBuild) === path.basename(state.cudaCmakeBuild),
    expectedCmakeCommand = [cmake, "-S", state.cudaCmakeSource, "-B", cmakeBuild, "-DCUDAToolkit_ROOT=" + state.cudaRoot];
  if (!hasExactKeys(cmakeEvidence, ["version", "command", "status", "tool", "script", "result", "stdout", "stderr"]) || cmakeEvidence.version !== 1 ||
      cmakeEvidence.status !== 0 || !validCmakeBuild || !isDeepStrictEqual(cmakeEvidence.command, expectedCmakeCommand) ||
      !hasExactKeys(cmakeEvidence.tool, ["path", "sha256"]) || typeof cmake !== "string" || !path.isAbsolute(cmake) ||
      /[\r\n\0]/.test(cmake) || typeof cmakeEvidence.tool.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(cmakeEvidence.tool.sha256) ||
      !hasExactKeys(cmakeEvidence.script, ["path", "sha256"]) || cmakeEvidence.script.path !== state.cudaCmakeLists ||
      cmakeEvidence.script.sha256 !== fileHash(state.cudaCmakeLists) ||
      !hasExactKeys(cmakeEvidence.result, ["toolkitVersion", "cudart", "importedCudart", "includeDirectories"]) ||
      !/^12\.6\./.test(cmakeEvidence.result.toolkitVersion) || cmakeEvidence.result.cudart !== path.join(state.cudaRoot, "lib64/libcudart.so") ||
      cmakeEvidence.result.importedCudart !== cmakeEvidence.result.cudart ||
      typeof cmakeEvidence.result.includeDirectories !== "string" ||
      !cmakeEvidence.result.includeDirectories.split(";").includes(path.join(state.cudaRoot, "include")) ||
      typeof cmakeEvidence.stdout !== "string" || !cmakeEvidence.stdout.includes("DPCPP_CUDA_TOOLKIT_PROBE=PASS") || typeof cmakeEvidence.stderr !== "string") {
    throw new Error("CUDA handoff is missing the successful same-root CMake toolkit smoke");
  }
  for (const [root, row, relative, executable] of [[state.dpcppRoot, record.dpcpp.marker, ".mom-toolchain-sha256", false],
    [state.dpcppRoot, record.dpcpp.clang, "bin/clang", true], [state.dpcppRoot, record.dpcpp.clangxx, "bin/clang++", true],
    [state.dpcppRoot, record.dpcpp.sycl, "lib/libsycl.so.9", false], [state.dpcppRoot, record.dpcpp.jit, "lib/libsycl-jit.so", false],
    [state.cudaRoot, record.cuda.nvcc, "bin/nvcc", true], [state.cudaRoot, record.cuda.profile, "bin/nvcc.profile", false],
    [state.cudaRoot, record.cuda.ptxas, "bin/ptxas", true], [state.cudaRoot, record.cuda.fatbinary, "bin/fatbinary", true],
    [state.cudaRoot, record.cuda.libdevice, "nvvm/libdevice/libdevice.10.bc", false], [state.cudaRoot, record.cuda.cudart, "lib64/libcudart.so", false],
    [state.cudaRoot, record.cuda.staticRuntime?.cudart, "lib64/libcudart_static.a", false],
    [state.cudaRoot, record.cuda.staticRuntime?.culibos, "lib64/libculibos.a", false]]) {
    if (!hasExactKeys(row, ["path", "sha256"]) || row.path !== path.join(root, relative) || !/^[0-9a-f]{64}$/.test(row.sha256 || "") ||
        inputFileRecord(root, relative, executable).sha256 !== row.sha256) {throw new Error("Selected DPC++/CUDA input changed: " + relative);}
  }
  if (!isStaticArchive(record.cuda.staticRuntime.cudart.path) || !isStaticArchive(record.cuda.staticRuntime.culibos.path)) {
    throw new Error("CUDA static-runtime inputs are not regular static archives");
  }
  const headers = sdkSourceFiles(path.join(state.cudaRoot, "include"));
  const probeText = fs.readFileSync(state.cudaProbeEvidence, "utf8"), probeTop = probeText.match(/^#\$ TOP=(.+)$/m)?.[1]?.trim();
  if (!hasExactKeys(record.cuda.headers, ["files", "sha256"]) || !isDeepStrictEqual(headers, record.cuda.headers.files) ||
      record.cuda.headers.sha256 !== jsonHash(JSON.stringify(headers)) ||
      !["cuda.h", "nvml.h", "cuda_runtime.h"].every(name => Object.hasOwn(headers, name)) || !probeTop || path.resolve(probeTop) !== state.cudaRoot ||
      record.cuda.excluded.length > 1) {
    throw new Error("CUDA headers are incomplete or changed after handoff");
  }
  return record;
}

/** NVCC deliberately rejects CMake's sentinel argument after printing toolkit metadata.
 * Accept that exact diagnostic, not arbitrary failed queries or an ambient toolkit root.
 * @param {Pick<import("node:child_process").SpawnSyncReturns<string>,"status"|"signal"|"error"|"stdout"|"stderr">} version
 * @param {Pick<import("node:child_process").SpawnSyncReturns<string>,"status"|"signal"|"error"|"stdout"|"stderr">} probe
 * @param {string} root */
function cudaToolkitProbeRoot(version, probe, root) {
  const top = ((probe.stdout || "") + (probe.stderr || "")).match(/^#\$ TOP=(.+)$/m)?.[1]?.trim(),
    sentinel = probe.status === 1 && /^nvcc fatal\s*: Don't know what to do with '__cmake_determine_cuda'\s*$/m.test(probe.stderr || "");
  if (version.error || version.signal || version.status !== 0 || probe.error || probe.signal ||
      probe.status !== 0 && !sentinel || !top || path.resolve(top) !== root) {
    throw new Error("Captured CUDA root does not satisfy the public CMake toolkit probe", {cause: probe.error || version.error});
  }
  return root;
}

/** Copy and prove the pinned installed DPC++/CUDA inputs into one fresh stable state root.
 * `sourceOrigin` is the existing CI tar digest or a local prepared-receipt origin; both
 * routes share the exact same vendor copies, probes, manifest payload checks and auditor.
 * @param {DpcppCudaInputState} state @param {{dpcpp:string,cuda:string}} installedRoots
 * @param {DpcppCudaSourceOrigin} sourceOrigin @param {NodeJS.ProcessEnv} environment */
function stageDpcppCudaInputs(state, installedRoots, sourceOrigin, environment) {
  if (!hasExactKeys(installedRoots, ["cuda", "dpcpp"]) ||
      [state.root, state.dpcppRoot, state.cudaRoot].some(file => fs.lstatSync(file, {throwIfNoEntry: false})) ||
      fs.realpathSync(path.dirname(state.root)) !== path.dirname(state.root)) {
    throw new Error("DPC++/CUDA staging needs a fresh output beneath a canonical parent");
  }
  for (const [name, root] of Object.entries(installedRoots)) {
    const stat = typeof root === "string" ? fs.lstatSync(root, {throwIfNoEntry: false}) : undefined;
    if (typeof root !== "string" || !path.isAbsolute(root) || path.resolve(root) !== root || !stat?.isDirectory() ||
        stat.isSymbolicLink() || fs.realpathSync(root) !== root) {throw new Error("Installed DPC++/CUDA root is not a canonical directory: " + name);}
  }
  fs.mkdirSync(state.root, {mode: 0o700});
  copyDpcppBuildInputs(installedRoots.dpcpp, state.dpcppRoot);
  const excluded = copyCudaBuildInputs(installedRoots.cuda, state.cudaRoot), packages = spawnSync("/usr/bin/dpkg-query",
    // These are dpkg's fields, not JavaScript template expressions.
    // eslint-disable-next-line no-template-curly-in-string
    ["-W", "-f=${Package}\t${Version}\n", ...cudaPackageNames],
    {encoding: "utf8", env: environment, timeout: 30000, maxBuffer: 1 << 20});
  if (packages.error || packages.signal || packages.status !== 0) {throw new Error("CUDA package provenance query failed", {cause: packages.error});}
  const packageVersions = Object.fromEntries(packages.stdout.trim().split(/\r?\n/).map(line => line.split("\t"))),
    cudaCopyrights = cudaPackageNames.map(name => capturePackageCopyright(state, name, packageVersions[name])),
    clangVersion = spawnSync(path.join(state.dpcppRoot, "bin/clang++"), ["--version"], {encoding: "utf8", env: environment,
      timeout: 30000, maxBuffer: 1 << 20});
  if (clangVersion.error || clangVersion.signal || clangVersion.status !== 0 || !clangVersion.stdout) {
    throw new Error("Pinned DPC++ compiler identity query failed", {cause: clangVersion.error});
  }
  const nvcc = path.join(state.cudaRoot, "bin/nvcc"), version = spawnSync(nvcc, ["--version"], {encoding: "utf8", env: environment,
      timeout: 30000, maxBuffer: 1 << 20}),
    probe = spawnSync(nvcc, ["-v", "__cmake_determine_cuda"], {encoding: "utf8", env: environment, timeout: 30000, maxBuffer: 8 << 20}),
    probeText = (probe.stdout || "") + (probe.stderr || "");
  fs.writeFileSync(state.cudaVersionEvidence, version.stdout || "", {flag: "wx", mode: 0o600});
  fs.writeFileSync(state.cudaProbeEvidence, probeText, {flag: "wx", mode: 0o600});
  const top = cudaToolkitProbeRoot(version, probe, state.cudaRoot);
  probeCudaToolkit(state, {env: environment});
  const record = dpcppCudaInputsManifest(state, sourceOrigin, packageVersions,
    {version: version.stdout, top, dpcppVersion: clangVersion.stdout}, cudaCopyrights, excluded);
  fs.writeFileSync(state.manifest, JSON.stringify(record, null, 2) + "\n", {flag: "wx", mode: 0o600});
  auditDpcppCudaInputs(state, readBuildJson(state.manifest));
  return record;
}

module.exports = {cudaPackageNames, copyDpcppBuildInputs, copyCudaBuildInputs, auditCudaPackageCopyrights,
  cudaToolkitProbeCMakeLists, probeCudaToolkit, inputFileRecord, preparedDpcppCudaSourceOrigin, dpcppCudaInputsManifest,
  auditDpcppCudaInputs, cudaToolkitProbeRoot, stageDpcppCudaInputs};
