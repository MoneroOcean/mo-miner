"use strict";

// Native CI Linux SDK descriptor, native components and application assembly.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {canonicalExecutable, ciBuildJobs, hasExactKeys, jsonHash, linuxNativeComponentChildTimeout, runNode, validCiSuffix} = require("./native-ci-job");
const {fileHash, readBuildJson} = require("./build-source-inputs");
const {verifyZstdArchive} = require("./fetch-native-sdk-sources");
const {auditGnuHeaders} = require("./prepare-native-gnu-headers");
const {cudaImageInputs} = require("./native-cuda-images");
const {dpcppComponentInputs} = require("./native-dpcpp-component");
const {linuxNativeSdkLldPath, linuxNativeSdkDescriptorConfig} = require("./native-linux-sdk-config");
const {availableDiskBytes, extractState, linuxNativeBuildReserveBytes, linuxNativeSdkDiskPreflight, listState, packState,
  safeRelative, tarExpandedBytes, temp} = require("./native-ci-tar");
const {acppStatePaths, dpcppCudaInputPins, dpcppCudaInputsStatePaths, dpcppSourceArchivePin, gnuHeadersStatePaths,
  hermesStatePaths, linuxSourceStatePaths, openmpStatePaths, provisionOneApi, verifyAcppSource, verifyHermesSource,
  verifyIoSource, verifyLinuxSources, verifyRocmSources, zstdStatePaths} = require("./native-ci-linux-sources");
const {auditDpcppCudaInputs, cudaPackageNames} = require("./native-dpcpp-cuda-inputs");
const {admitLinuxLlvmOutput} = require("./native-ci-linux-llvm");
const {auditHermesSdk, auditIoOpenSsl, auditOpenmpSdk, hermesBuildTools, ioFinalTar, openmpBuildTools} = require("./native-ci-linux-support");
const {linuxSysrootToolchain} = require("./native-linux-sysroot");
const {auditCudaNoticeSources, cudaCopyrightName, cudaImagesStatePaths} = require("./native-ci-linux-cuda");
const {linuxJitSdkStatePaths} = require("./native-ci-linux-jit");
const {acppCudaHipSdkStatePaths, auditAcppCudaHipSdk,
  auditRocmHostNoticeSources, prepareRocmHostAdmission, rocmHostStatePaths} = require("./native-ci-linux-rocm");
const repo = path.resolve(__dirname, "..");

/** Original same-run tars needed to compose (but not build) the public Linux release descriptor.
 * Every producer remains a separate raw artifact; shared /tmp roots are restored only once.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths @param {NodeJS.ProcessEnv} [environment] */
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
      admission.format !== "mom-native-linux-sdk-descriptor-v1" ||
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
/** @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "suffix" | "runnerTemp">} paths */
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

/** Per-job component roots and handoffs keep producer paths fixed per profile.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "suffix" | "runnerTemp">} paths @param {string} profile */
function linuxNativeComponentStatePaths(paths, profile) {
  if (profile !== "core" && !linuxNativeGpuProfiles.includes(profile)) {throw new Error("Select the Linux Core or one exact GPU factory profile");}
  const name = "mom-native-linux-component-" + profile + "-" + paths.suffix,
    root = path.join(temp, name), archive = path.join(paths.runnerTemp, "native-linux-component-" + profile + "-" + paths.suffix + ".tar");
  if (!validCiSuffix(paths.suffix) || path.dirname(root) !== temp ||
      path.resolve(root) !== root || path.dirname(archive) !== paths.runnerTemp || path.resolve(archive) !== archive ||
      root === archive || root.startsWith(archive + path.sep)) {throw new Error("Native component paths differ from the run-scoped plan");}
  return {profile, name, root, archive, incoming: path.join(paths.runnerTemp, "incoming"), downloads: path.join(paths.runnerTemp, "downloads")};
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
 * @param {{profile:string,name:string,root:string,archive:string}} state
 * @param {string} descriptorHash @param {string} admissionHash @param {Array<{artifact:string,sha256:string}>} inputArchives
 * @param {{receipt:string,sha256:string,object:string,objectSha256:string}} producer */
function writeLinuxNativeComponentManifest(state, descriptorHash, admissionHash, inputArchives, producer) {
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
  const manifest = {format: "mom-native-linux-component-v1", profile: state.profile,
    descriptor: {sha256: descriptorHash}, admission: {sha256: admissionHash},
    inputArchives: inputArchives.map(row => ({artifact: row.artifact, sha256: row.sha256})),
    producer: {receipt: producer.receipt, sha256: producer.sha256, object, objectSha256: producer.objectSha256},
    files: linuxNativeComponentFiles(state.root), buildOnly: true, publicationQualified: false};
  fs.writeFileSync(path.join(state.root, "component.json"), JSON.stringify(manifest, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return manifest;
}

/** Move the exact run-scoped artifact set into disjoint descriptor, SDK and component paths.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths @param {string[]} [profiles] */
function partitionLinuxNativeDownloads(paths, profiles = []) {
  if (new Set(profiles).size !== profiles.length || profiles.some(profile => profile !== "core" && !linuxNativeGpuProfiles.includes(profile))) {
    throw new Error("Native component download plan contains an unknown or duplicate GPU profile");
  }
  const state = linuxNativeSdkDescriptorStatePaths(paths, paths.env), componentIncoming = path.join(paths.runnerTemp, "component-incoming"),
    descriptorName = path.basename(state.archive), componentNames = profiles.map(profile => "native-linux-component-" + profile + "-" + paths.suffix + ".tar"),
    expected = [descriptorName, ...state.inputRows.map(row => row.artifact), ...componentNames].sort();
  // native-job.yml downloads every artifact flat into incoming/; move that set aside first so it is
  // partitioned (and checked) exactly like a separate downloads directory.
  if (!fs.lstatSync(state.downloads, {throwIfNoEntry: false}) && fs.lstatSync(state.incoming, {throwIfNoEntry: false})?.isDirectory()) {
    fs.renameSync(state.incoming, state.downloads);
  }
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
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "env" | "runnerTemp">} paths
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
  if (admission.format !== "mom-native-linux-sdk-descriptor-v1" ||
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
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths @param {string} profile */
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
    nativeStage(handoff.config.cpu, process.execPath, ["--max-old-space-size=256", path.join(repo, "scripts/build-native-core.js"),
      handoff.config.cpu.cache, core, "--gpu"], environment, repo, component.root, "core", timedSpawn(buildTimeout));
    receipt = path.join(core, "inputs.json"); object = path.join(core, "cpu-core.a");
    const input = readBuildJson(receipt);
    if (input.gpu !== true || input.portable !== true || input.objectSha256 !== fileHash(object)) {
      throw new Error("Native Core producer did not retain the original portable Core receipt");
    }
  } else {
    const output = path.join(component.root, "gpu", profile), selected = handoff.selected, runner = selected.sdk.gpu[profile],
      isAcpp = profile.startsWith("acpp-"), script = isAcpp ? "native-gpu-build.js" : "build-native-gpu.js";
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
      require("./build-native-application").verifyCudaImageReceiptInputs(record, handoff.linked.cudaImageInputs, profile);
    }
    object = record.object;
  }
  const producer = {receipt: path.relative(component.root, receipt), sha256: fileHash(receipt), object,
      objectSha256: fileHash(object)}, inputArchives = handoff.archives.map(row => ({artifact: row.artifact, sha256: row.sha256}));
  writeLinuxNativeComponentManifest(component, handoff.descriptorHash, handoff.admissionHash, inputArchives, producer);
  packState(component.archive, temp, [component.name]);
  const members = listState(component.archive, [component.name]);
  if (!members.includes(component.name + "/component.json") || fileHash(receipt) !== producer.sha256 || fileHash(object) !== producer.objectSha256) {
    throw new Error("Native component artifact changed during same-run packaging");
  }
  console.log(`[native-ci] Built only ${profile} with its maintained public producer; no executable ran or was publication-qualified (${members.length} members).`);
}

/** Verify an extracted component tree and its exact tar member set before app assembly.
 * @param {{profile:string,name:string,root:string,admissionFile:string}} state
 * @param {{descriptor:{sha256:string}}} descriptorAdmission
 * @param {Array<{artifact:string,sha256:string}>} inputArchives @param {string[]} archiveMembers */
function auditLinuxNativeComponentTree(state, descriptorAdmission, inputArchives, archiveMembers) {
  const rootStat = fs.lstatSync(state.root), manifestFile = path.join(state.root, "component.json"), manifestStat = fs.lstatSync(manifestFile),
    manifest = readBuildJson(manifestFile);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(state.root) !== state.root ||
      !manifestStat.isFile() || manifestStat.isSymbolicLink() || !hasExactKeys(manifest,
    ["format", "profile", "descriptor", "admission", "inputArchives", "producer", "files", "buildOnly", "publicationQualified"]) ||
      manifest.format !== "mom-native-linux-component-v1" ||
      manifest.profile !== state.profile || manifest.buildOnly !== true || manifest.publicationQualified !== false ||
      !hasExactKeys(manifest.descriptor, ["sha256"]) || manifest.descriptor.sha256 !== descriptorAdmission.descriptor.sha256 ||
      !hasExactKeys(manifest.admission, ["sha256"]) || manifest.admission.sha256 !== fileHash(state.admissionFile) ||
      !isDeepStrictEqual(manifest.inputArchives, inputArchives) || !Array.isArray(manifest.files) ||
      !hasExactKeys(manifest.producer, ["receipt", "sha256", "object", "objectSha256"]) ||
      typeof manifest.producer.receipt !== "string" || !/^[a-f0-9]{64}$/.test(manifest.producer.sha256 || "") ||
      typeof manifest.producer.object !== "string" || !path.isAbsolute(manifest.producer.object) ||
      !/^[a-f0-9]{64}$/.test(manifest.producer.objectSha256 || "")) {
    throw new Error("Native component tree is not bound to this exact SDK descriptor");
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
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths @param {ReturnType<typeof partitionLinuxNativeDownloads>} downloads
 * @param {{state:ReturnType<typeof linuxNativeSdkDescriptorStatePaths>,admission:{descriptor:{sha256:string}},
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
    if (!hasExactKeys(manifest, ["format", "profile", "descriptor", "admission", "inputArchives", "producer", "files", "buildOnly", "publicationQualified"]) ||
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
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
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
 * This emits no Core/factory result and is not a built or runtime-qualified application. @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildLinuxNativeSdkDescriptor(paths) {
  const state = linuxNativeSdkDescriptorStatePaths(paths, paths.env);
  if ([state.root, state.archive].some(file => fs.lstatSync(file, {throwIfNoEntry: false}))) {
    throw new Error("Native Linux SDK descriptor stage refuses an existing output");
  }
  const externalDpcppPins = dpcppCudaInputPins(paths.env),
    dpcppInputTar = path.join(state.incoming, path.basename(state.dpcpp.sdkTar));
  dpcppSourceArchivePin(dpcppInputTar, externalDpcppPins.bundleSha256);
  const inputArchives = auditLinuxNativeSdkDescriptorInputArchives(state);
  if (inputArchives.find(row => row.artifact === path.basename(state.dpcpp.sdkTar))?.sha256 !== externalDpcppPins.bundleSha256) {
    throw new Error("DPC++/CUDA descriptor input differs from its external same-run bundle pin");
  }
  const expandedInputBytes = inputArchives.reduce((sum, row) => sum + row.expandedBytes, 0),
    diskPreflight = linuxNativeSdkDiskPreflight(expandedInputBytes, availableDiskBytes(temp), availableDiskBytes(state.runnerTemp));
  restoreLinuxNativeSdkDescriptorInputs(state);
  dpcppSourceArchivePin(dpcppInputTar, externalDpcppPins.bundleSha256);
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
  const cudaInputs = auditDpcppCudaInputs(state.dpcpp, readBuildJson(state.dpcpp.manifest), externalDpcppPins.manifestSha256),
    cudaImages = auditLinuxNativeSdkDescriptorCudaImages(state, cudaInputs),
    component = dpcppComponentInputs(state.jit.staticComponent);
  // Existing signed Intel installation is needed to re-admit producer tool fingerprints.
  provisionOneApi(paths, "/var/tmp/mom-dev-native-linux-sdk-descriptor-" + paths.suffix, 30 * 60 * 1000);
  admitLinuxLlvmOutput(paths);
  prepareRocmHostAdmission(paths, state.rocm);
  const hermesTools = hermesBuildTools(), openmpTools = openmpBuildTools(), acppTools = {cc: hermesTools.cc, cxx: hermesTools.cxx};
  auditHermesSdk(state.hermes, hermesTools, ciBuildJobs(paths.env), linuxSysrootToolchain(paths.env));
  auditOpenmpSdk(state.openmp, openmpTools);
  auditAcppCudaHipSdk(state.acppSdk, acppTools, externalDpcppPins.manifestSha256);
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
  const admission = {format: "mom-native-linux-sdk-descriptor-v1",
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

module.exports = {auditLinuxNativeApplicationEntrypoint, auditLinuxNativeComponentArchive, auditLinuxNativeComponentTree,
  auditLinuxNativeSdkDescriptorCudaImages, auditLinuxNativeSdkDescriptorInputArchive, auditLinuxNativeSdkDescriptorInputArchives,
  auditLinuxNativeSdkDescriptorLineage, buildLinuxNativeApplication, buildLinuxNativeComponent, buildLinuxNativeSdkDescriptor,
  linuxNativeApplicationBuildCommand, linuxNativeApplicationStatePaths, linuxNativeAssemblyComponents, linuxNativeAssemblyConfig,
  linuxNativeComponentArchivePins, linuxNativeComponentFiles, linuxNativeComponentProfiles, linuxNativeComponentStatePaths,
  linuxNativeGpuProfiles, linuxNativeSdkDescriptorStatePaths, linuxNativeSdkGnuNoticeSources, linuxNativeSdkRocmNoticeSources,
  partitionLinuxNativeDownloads, restoreLinuxNativeComponents, restoreLinuxNativeDescriptorHandoff,
  writeLinuxNativeComponentManifest};
