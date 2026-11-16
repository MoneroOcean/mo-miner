"use strict";

// Local/WSL admission feeds the same Windows GPU recipes used by CI, without
// manufacturing CI archives or run identities for workspace-owned producers.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk");
const {verifyLinuxLlvmAdmission} = require("./build-native-llvm21-sdk");
const {verifyPreparedSdkSources} = require("./prepare-native-sdk-sources");
const {auditDpcppCudaInputs} = require("./native-dpcpp-cuda-inputs");
const {windowsHostSdkInputs} = require("./native-windows-host-sdk-inputs");
const images = require("./build-local-native-cuda-images");
const acppSdk = require("./build-local-native-acpp-sdk"), {acppFrontendRows} = require("./native-linux-llvm-tools");
const {composeWindowsDpcppGpuConfig, composeWindowsAcppGpuConfig, resolveAcppFrontend} = require("./native-windows-gpu-config");
const {fileHash, readBuildJson, buildFileInputs} = require("./build-source-inputs");
const repo = fs.realpathSync(path.resolve(__dirname, ".."));

/** Only the current workspace's installed LLVM tools may construct the descriptors.
 * Full SDK source/build/install admission precedes this selected-file check.
 * @param {string} root @param {any} admission */
function localWindowsGpuTools(root, admission) {
  const prefix = path.join(root, "sdk"), toolRoot = path.join(prefix, "bin"), names = ["llvm-nm", "llvm-readobj", "llvm-objcopy"];
  if (admission?.sdk?.path !== prefix || !admission.sdk.snapshot?.files) {
    throw new Error("Local Windows GPU tools require the admitted LLVM installation");
  }
  const files = Object.fromEntries(names.map(name => {
    const file = path.join(toolRoot, name), row = admission.sdk.snapshot.files["bin/" + name],
      stat = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!row || row.type !== "file" || !/^[a-f0-9]{64}$/.test(row.sha256 || "") ||
        !stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file ||
        !(stat.mode & 0o111) || stat.size !== row.bytes || fileHash(file) !== row.sha256) {
      throw new Error("Local Windows GPU tool differs from its LLVM admission: " + name);
    }
    return [file, row.sha256];
  }));
  return {toolRoot, llvmNm: path.join(toolRoot, "llvm-nm"), files};
}

/** Admit the actual producer roots, not a caller's claim that a prior stage succeeded.
 * No source acquisition or compiler build is performed here.
 * @param {{workspace:string,windowsSdkRoot:string,windowsSdkPins:Parameters<typeof windowsHostSdkInputs>[1],
 * windowsSourceSha256:string,tools:ReturnType<typeof local.localTools>}} input
 * @param {{verifyScope?:()=>unknown,resolveTools?:typeof local.localTools,verifyLlvm?:typeof verifyLinuxLlvmAdmission,
 * verifySources?:typeof verifyPreparedSdkSources,auditCuda?:typeof auditDpcppCudaInputs,
 * readHost?:typeof windowsHostSdkInputs,auditImages?:typeof images.auditLocalCudaImages}} [dependencies] */
function admitLocalWindowsDpcppGpuInputs(input, dependencies = {}) {
  (dependencies.verifyScope || local.verifyCurrentMemoryScope)();
  if (process.platform !== "linux" || process.arch !== "x64" || !input ||
      Object.keys(input).sort().join(",") !== "tools,windowsSdkPins,windowsSdkRoot,windowsSourceSha256,workspace" ||
      !/^[a-f0-9]{64}$/.test(input.windowsSourceSha256 || "")) {
    throw new Error("Local Windows DPC++ GPU inputs require Linux x64 and explicit workspace/source pins");
  }
  const workspace = local.admitWorkspace(input.workspace, repo), paths = local.supportPaths(workspace),
    actualTools = (dependencies.resolveTools || local.localTools)(process.env);
  if (!isDeepStrictEqual(input.tools, actualTools)) {throw new Error("Local Windows GPU host tools differ from the current pinned selection");}
  const sourceFile = path.join(workspace, "windows-sources/sources.json"), sourceRoot = path.join(workspace, "windows-sources/dpcpp");
  if (fileHash(sourceFile) !== input.windowsSourceSha256) {throw new Error("Prepared Windows source receipt differs from its pin");}
  const sources = (dependencies.verifySources || verifyPreparedSdkSources)(sourceFile, "windows");
  if (sources.target !== "windows" || sources.prepared?.dpcpp?.root !== sourceRoot) {
    throw new Error("Prepared Windows sources do not belong to the admitted workspace");
  }
  const llvm = (dependencies.verifyLlvm || verifyLinuxLlvmAdmission)(paths.llvmSdk), tools = localWindowsGpuTools(paths.llvmSdk, llvm),
    state = local.localDpcppCudaState(paths), manifest = readBuildJson(state.manifest),
    manifestSha256 = fileHash(state.manifest);
  (dependencies.auditCuda || auditDpcppCudaInputs)(state, manifest, manifestSha256);
  const host = (dependencies.readHost || windowsHostSdkInputs)(input.windowsSdkRoot, input.windowsSdkPins),
    imageState = images.localCudaImagePaths(workspace),
    imageAdmission = (dependencies.auditImages || images.auditLocalCudaImages)(imageState, actualTools.cxx),
    include = path.join(imageState.handoffRoot, "cuda-images.inc"),
    imageFiles = [include, include + ".json", path.join(imageState.handoffRoot, "result.json")],
    selected = [sourceFile, state.manifest, ...Object.keys(tools.files), ...imageFiles],
    files = buildFileInputs(selected);
  if (files[sourceFile] !== input.windowsSourceSha256 || files[state.manifest] !== manifestSha256 ||
      Object.entries(tools.files).some(([file, sha256]) => files[file] !== sha256) ||
      imageFiles.some(file => imageAdmission.fingerprints[file] !== files[file])) {
    throw new Error("Local Windows GPU inputs changed during admission");
  }
  return {workspace, sourceRoot, sources, dpcppRoot: state.dpcppRoot, cudaRoot: state.cudaRoot,
    dpcppManifest: manifest, tools, host, llvm, compilerRoot: path.dirname(path.dirname(actualTools.cxx)),
    cudaImages: {include, hashes: {includeSha256: files[include], sidecarSha256: files[include + ".json"],
      resultSha256: files[path.join(imageState.handoffRoot, "result.json")]}}, imageAdmission, files};
}

/** Extend the shared cohort with the original ACPP/ROCm receipts and fresh-build
 * plugin/header capture. The same LLVM frontend selector/query used by CI applies
 * directly to the admitted installation; no copied frontend handoff is needed.
 * @param {Parameters<typeof admitLocalWindowsDpcppGpuInputs>[0]} input
 * @param {Parameters<typeof admitLocalWindowsDpcppGpuInputs>[1] & {
 * admitBase?:typeof admitLocalWindowsDpcppGpuInputs,auditAcppInputs?:typeof acppSdk.auditLocalAcppSdkInputs,
 * auditAcpp?:typeof acppSdk.auditLocalAcppSdk,readArtifacts?:typeof acppSdk.admitLocalAcppGpuArtifacts,
 * resolveFrontend?:typeof resolveAcppFrontend}} [dependencies] */
function admitLocalWindowsGpuInputs(input, dependencies = {}) {
  const base = (dependencies.admitBase || admitLocalWindowsDpcppGpuInputs)(input, dependencies),
    context = acppSdk.localAcppSdkContextFromWorkspace(base.workspace),
    {git, cmake, ninja, cc, cxx} = input.tools, hostTools = {git, cmake, ninja, cc, cxx},
    inputs = (dependencies.auditAcppInputs || acppSdk.auditLocalAcppSdkInputs)(context, hostTools),
    sdk = (dependencies.auditAcpp || acppSdk.auditLocalAcppSdk)(context, hostTools, {auditInputs: () => inputs}),
    artifacts = (dependencies.readArtifacts || acppSdk.admitLocalAcppGpuArtifacts)(context),
    prefix = base.llvm.sdk.path, snapshot = base.llvm.sdk.snapshot.files,
    frontend = (dependencies.resolveFrontend || resolveAcppFrontend)(prefix, {
      clang: path.join(prefix, "bin/clang"), clangxx: path.join(prefix, "bin/clang++"),
      resourceInclude: path.join(prefix, "lib/clang/21/include"),
      files: Object.fromEntries(acppFrontendRows(snapshot).map(row => [row.path, snapshot[row.path]])),
    });
  if (artifacts.generatedInclude !== path.join(sdk.build, "include") ||
      artifacts.plugin.path !== path.join(sdk.build, "src/compiler/libacpp-clang.so")) {
    throw new Error("Local ACPP GPU artifact capture belongs to a different SDK build");
  }
  return {...base, acpp: {source: sdk.source, generatedInclude: artifacts.generatedInclude, plugin: artifacts.plugin.path},
    acppSdk: sdk, artifacts, frontend, hip: inputs.rocm.comgr.hip};
}

/** Build all five profiles after one cohort admission, then re-admit that
 * cohort before returning. Shared constructors retain compiler/header ABI checks.
 * This is descriptor preparation, not GPU execution or release qualification.
 * @param {Parameters<typeof admitLocalWindowsDpcppGpuInputs>[0]} input
 * @param {{admit?:typeof admitLocalWindowsGpuInputs,compose?:typeof composeWindowsDpcppGpuConfig,
 * composeAcpp?:typeof composeWindowsAcppGpuConfig} & Parameters<typeof admitLocalWindowsGpuInputs>[1]} [dependencies] */
function composeLocalWindowsGpuConfigs(input, dependencies = {}) {
  const admit = dependencies.admit || admitLocalWindowsGpuInputs, admitted = admit(input, dependencies),
    compose = dependencies.compose || composeWindowsDpcppGpuConfig, composeAcpp = dependencies.composeAcpp || composeWindowsAcppGpuConfig;
  const profiles = /** @type {Record<string,Pick<ReturnType<typeof composeWindowsDpcppGpuConfig>,"config"|"configBytes"|"configSha256">>} */
    (Object.fromEntries(["oneapi", "dpcpp", "dpcpp-opencl"].map(profile => [profile, compose(profile, {
      compilerRoot: profile === "oneapi" ? admitted.compilerRoot : admitted.dpcppRoot,
      sourceRoot: admitted.sourceRoot, dpcppRoot: admitted.dpcppRoot, dpcppManifest: admitted.dpcppManifest,
      host: admitted.host, tools: admitted.tools,
      ...(profile === "dpcpp" ? {cudaRoot: admitted.cudaRoot} : {}),
      cudaImages: profile === "dpcpp" ? admitted.cudaImages : null,
    })])));
  for (const profile of /** @type {const} */ (["acpp-cuda", "acpp-hip"])) {
    const built = composeAcpp(profile, {host: admitted.host, tools: admitted.tools, acpp: admitted.acpp, frontend: admitted.frontend,
      ...(profile === "acpp-hip" ? {hipInclude: admitted.hip.include} : {}),
      cuda: profile === "acpp-cuda" ? {sdk: admitted.cudaRoot, images: admitted.cudaImages.include} : null});
    Object.assign(profiles, {[profile]: built});
  }
  if (!isDeepStrictEqual(admitted, admit(input, dependencies))) {throw new Error("Local Windows GPU producer inputs changed while composing profiles");}
  return {profiles, admitted, buildOnly: true, executed: false, publicationQualified: false};
}

module.exports = {localWindowsGpuTools, admitLocalWindowsDpcppGpuInputs, admitLocalWindowsGpuInputs, composeLocalWindowsGpuConfigs};
