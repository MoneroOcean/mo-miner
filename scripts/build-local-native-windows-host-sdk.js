"use strict";

// Local WSL acquisition/extraction adapter for the shared build-only Windows host SDK.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk");
const {fileHash} = require("./build-source-inputs");
const {fetchWindowsHostSdkArchives, verifyWindowsHostSdkArchives} = require("./fetch-native-sdk-sources");
const {prepareWindowsHostSdk} = require("./prepare-native-windows-sdk");
const {windowsHostSdkInputs} = require("./native-windows-host-sdk-inputs");
const repo = fs.realpathSync(path.resolve(__dirname, ".."));

/** @param {string} directory @param {string} label */
function existingDirectory(directory, label) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat) {return false;}
  if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
    throw new Error(label + " is not a canonical directory: " + directory);
  }
  return true;
}

/** Hash the exact public producer receipt names; the shared reader validates their content and tree.
 * @param {string} root @returns {{resultSha256:string,descriptorSha256:string,filesSha256:string,overlaySha256:string}} */
function windowsHostSdkPins(root) {
  return {resultSha256: fileHash(path.join(root, "result.json")), descriptorSha256: fileHash(path.join(root, "sdk.json")),
    filesSha256: fileHash(path.join(root, "files.json")), overlaySha256: fileHash(path.join(root, "vfs-overlay.json"))};
}

/**
 * Prepare or re-admit the immutable Windows build-only SDK in the marked Linux workspace.
 * Acquisition is the only network step; extraction and all receipt/tree admission reuse public APIs.
 * @param {{workspace:string}} input
 * @param {{verifyScope?:()=>unknown,admitWorkspace?:typeof local.admitWorkspace,
 * fetchArchives?:typeof fetchWindowsHostSdkArchives,verifyArchives?:typeof verifyWindowsHostSdkArchives,
 * prepareSdk?:typeof prepareWindowsHostSdk,readSdk?:typeof windowsHostSdkInputs,environment?:NodeJS.ProcessEnv}} [dependencies]
 */
async function buildLocalNativeWindowsHostSdk(input, dependencies = {}) {
  (dependencies.verifyScope || local.verifyCurrentMemoryScope)();
  if (process.platform !== "linux" || process.arch !== "x64" || !input || Object.keys(input).join(",") !== "workspace" ||
      typeof input.workspace !== "string") {
    throw new Error("Local Windows host-SDK preparation requires Linux x64 and one marked workspace");
  }
  const workspace = (dependencies.admitWorkspace || local.admitWorkspace)(input.workspace, repo),
    root = path.join(workspace, "windows-host-sdk"), archiveRoot = path.join(root, "archives"), sdkRoot = path.join(root, "sdk"),
    workRoot = sdkRoot + ".work";
  if (!existingDirectory(root, "Local Windows host-SDK workspace")) {fs.mkdirSync(root, {mode: 0o700});}
  const entries = fs.readdirSync(root).sort();
  if (entries.some(name => !["archives", "sdk"].includes(name))) {
    throw new Error("Local Windows host-SDK workspace contains unknown entries; preserve it and investigate");
  }
  const archivesExist = existingDirectory(archiveRoot, "Local Windows host-SDK archives"), sdkExists = existingDirectory(sdkRoot, "Local Windows host SDK"),
    workStat = fs.lstatSync(workRoot, {throwIfNoEntry: false});
  if (workStat) {throw new Error("Local Windows host-SDK preparation has an incomplete work directory; preserve it and investigate");}
  if (sdkExists && !archivesExist) {throw new Error("Local Windows host SDK exists without its original verified archives; preserve it and investigate");}

  const environment = {...process.env, ...(dependencies.environment || {}), MOM_AOT_BUILD_ONLY: "1"},
    fetchArchives = dependencies.fetchArchives || fetchWindowsHostSdkArchives,
    verifyArchives = dependencies.verifyArchives || verifyWindowsHostSdkArchives,
    prepareSdk = dependencies.prepareSdk || prepareWindowsHostSdk,
    readSdk = dependencies.readSdk || windowsHostSdkInputs;
  if (!archivesExist) {await fetchArchives(archiveRoot, environment, true);}
  const sourceBefore = verifyArchives(archiveRoot), fetchReceipt = path.join(archiveRoot, "fetch.json"),
    fetchReceiptSha256 = fileHash(fetchReceipt);
  if (!sdkExists) {prepareSdk(archiveRoot, sdkRoot, environment);}
  const pins = windowsHostSdkPins(sdkRoot), admitted = readSdk(sdkRoot, pins), sourceAfter = verifyArchives(archiveRoot);
  if (admitted.result?.fetchReceipt?.path !== fetchReceipt || admitted.result.fetchReceipt.sha256 !== fetchReceiptSha256 ||
      !isDeepStrictEqual(sourceBefore, sourceAfter) || fetchReceiptSha256 !== fileHash(fetchReceipt) ||
      !isDeepStrictEqual(pins, windowsHostSdkPins(sdkRoot))) {
    throw new Error("Local Windows host-SDK inputs changed or lost original archive binding during admission; preserve the cache and investigate");
  }
  return {workspace, root: sdkRoot, pins, admitted, archiveRoot, fetchReceipt, fetchReceiptSha256,
    source: sourceAfter, buildOnly: true, compiled: false, publicationQualified: false};
}

module.exports = {buildLocalNativeWindowsHostSdk, windowsHostSdkPins};
