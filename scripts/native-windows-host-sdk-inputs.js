"use strict";

// Shared read-only admission for the composed Microsoft Windows host SDK inputs.
const fs = require("node:fs"), path = require("node:path");
const {hasExactBuildFields, fileHash, readBuildJsonSnapshot} = require("./build-source-inputs");
const {windowsHostSdkCatalog} = require("./fetch-native-sdk-sources");
const {safeRelative, walkTree} = require("./prepare-native-windows-sdk");

/**
 * Re-admit the existing composed Windows host-SDK tree and its package-origin inventory.
 * This stops at SDK inputs; tool, Hermes, and compiler-runtime admission belong to callers.
 * @param {string} windowsSdkRoot
 * @param {{resultSha256:string,descriptorSha256:string,filesSha256:string,overlaySha256:string}} pins
 */
function windowsHostSdkInputs(windowsSdkRoot, pins) {
  const pinKeys = ["resultSha256", "descriptorSha256", "filesSha256", "overlaySha256"];
  if (typeof windowsSdkRoot !== "string" || !path.isAbsolute(windowsSdkRoot) || /[\r\n\0]/.test(windowsSdkRoot) ||
      !hasExactBuildFields(pins, pinKeys) || Object.values(pins).some(hash => typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash))) {
    throw new Error("Windows host-SDK inputs require a canonical root and four explicit file pins");
  }
  const root = path.resolve(windowsSdkRoot), rootStat = fs.lstatSync(root);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || fs.realpathSync(root) !== root) {
    throw new Error("Windows host-SDK composition root is not canonical");
  }
  const filesPath = path.join(root, "files.json"), sdkPath = path.join(root, "sdk.json"),
    overlayPath = path.join(root, "vfs-overlay.json"), resultPath = path.join(root, "result.json"),
    sdkRoot = path.join(root, "sdk"), lock = windowsHostSdkCatalog();
  /** @param {string} file @param {string} expected @param {string} label */
  function readPinnedJson(file, expected, label) {
    const stat = fs.lstatSync(file);
    const message = "Windows host-SDK receipt differs from its external pin: " + label;
    if (!stat.isFile() || stat.nlink !== 1 || fs.realpathSync(file) !== file ||
        stat.size < 1 || stat.size > (8 << 20)) {throw new Error(message);}
    const {value} = readBuildJsonSnapshot(file, undefined, {sha256: expected, message});
    if (fileHash(file) !== expected) {throw new Error("Windows host-SDK receipt changed while reading: " + label);}
    return value;
  }
  const result = readPinnedJson(resultPath, pins.resultSha256, "result"),
    sdk = readPinnedJson(sdkPath, pins.descriptorSha256, "descriptor"),
    manifest = readPinnedJson(filesPath, pins.filesSha256, "file inventory"), overlayStat = fs.lstatSync(overlayPath);
  if (!overlayStat.isFile() || overlayStat.isSymbolicLink() || overlayStat.nlink !== 1 || fs.realpathSync(overlayPath) !== overlayPath ||
      overlayStat.size < 1 || overlayStat.size > (8 << 20) || fileHash(overlayPath) !== pins.overlaySha256) {
    throw new Error("Windows host-SDK VFS overlay differs from its pin");
  }
  if (result.version !== 1 || result.status !== "PASS" || result.target !== "windows-host" || result.buildOnly !== true ||
      result.compiled !== false || result.executed !== false || result.publicationQualified !== false ||
      result.sourceCatalogPublisherAuthenticated !== false || result.sourceCatalogBindingVerified !== false ||
      result.sourceCatalog?.publisherAuthenticated !== false || result.sourceCatalog?.channelCatalogBindingVerified !== false ||
      result.sourceCatalog?.sha256 !== lock.sourceCatalog.sha256 || result.packageCount !== Object.keys(lock.packages).length ||
      result.cabinetCount !== Object.keys(lock.cabinets).length || !Number.isSafeInteger(result.selectedFileCount) ||
      result.selectedFileCount < 1 || result.selectedFileCount > 50000 ||
      !Array.isArray(result.roots) || result.roots.length !== lock.roots.length ||
      result.roots.some((/** @type {unknown} */ name, /** @type {number} */ index) => name !== lock.roots[index]) ||
      result.descriptor?.path !== sdkPath || result.descriptor.sha256 !== pins.descriptorSha256 ||
      result.overlay?.path !== overlayPath || result.overlay.sha256 !== pins.overlaySha256 ||
      result.provenance?.path !== filesPath || result.provenance.sha256 !== pins.filesSha256 ||
      !hasExactBuildFields(sdk, ["sourceRoot", "overlay", "includeRoots", "libraryRoots"]) || sdk.sourceRoot !== sdkRoot || sdk.overlay !== overlayPath ||
      !Array.isArray(sdk.includeRoots) || !sdk.includeRoots.length || !Array.isArray(sdk.libraryRoots) || !sdk.libraryRoots.length ||
      !Array.isArray(manifest.files) || manifest.files.length !== result.selectedFileCount || manifest.files.length > 50000 ||
      !Array.isArray(manifest.origins)) {
    throw new Error("Windows host-SDK source composition receipt or descriptor is outside its build-only contract");
  }

  const actualTree = walkTree(sdkRoot), files = new Map(), originCounts = new Map(); let inventoryBytes = 0;
  for (const row of manifest.files) {
    if (!row || !hasExactBuildFields(row, ["path", "sha256", "bytes", "origins"]) || safeRelative(row.path) !== row.path ||
        !/^[a-f0-9]{64}$/.test(row.sha256 || "") || !Number.isSafeInteger(row.bytes) || row.bytes < 0 ||
        !Array.isArray(row.origins) || !row.origins.length || row.origins.length > 128 ||
        (inventoryBytes += row.bytes) > (2 * 1024 ** 3)) {
      throw new Error("Windows host-SDK file inventory row is invalid");
    }
    const key = row.path.normalize("NFC").toLowerCase();
    if (files.has(key)) {throw new Error("Windows host-SDK inventory has a case alias");}
    files.set(key, row);
    for (const origin of row.origins) {
      const packagePin = lock.packages[origin?.package];
      if (!hasExactBuildFields(origin, ["package", "packageSha256", "member", "sha256", "bytes"]) ||
          !/^[a-z0-9.-]{1,128}$/.test(origin.package || "") || !/^[a-f0-9]{64}$/.test(origin.packageSha256 || "") ||
          !packagePin || origin.packageSha256 !== packagePin.sha256 ||
          safeRelative(origin.member) !== origin.member || origin.sha256 !== row.sha256 || origin.bytes !== row.bytes) {
        throw new Error("Windows host-SDK source origin is malformed");
      }
      const key = JSON.stringify({destination: row.path, ...origin}); originCounts.set(key, (originCounts.get(key) || 0) + 1);
    }
  }
  if (actualTree.files.length !== files.size || actualTree.bytes !== inventoryBytes || manifest.origins.length > 100000) {
    throw new Error("Windows host-SDK tree differs from its bounded pinned inventory");
  }
  for (const file of actualTree.files) {
    const row = files.get(file.path.normalize("NFC").toLowerCase());
    if (!row || row.path !== file.path || row.bytes !== file.bytes || fileHash(file.full) !== row.sha256) {
      throw new Error("Windows host-SDK file differs from its pinned inventory: " + file.path);
    }
  }
  for (const origin of manifest.origins) {
    if (!hasExactBuildFields(origin, ["destination", "package", "packageSha256", "member", "sha256", "bytes"]) ||
        safeRelative(origin.destination) !== origin.destination || !/^[a-f0-9]{64}$/.test(origin.sha256 || "")) {
      throw new Error("Windows host-SDK origin index is malformed");
    }
    const key = JSON.stringify(origin), count = originCounts.get(key) || 0;
    if (!count) {throw new Error("Windows host-SDK origin index is not bound to a selected file");}
    if (count === 1) {originCounts.delete(key);} else {originCounts.set(key, count - 1);}
  }
  if (originCounts.size) {throw new Error("Windows host-SDK selected file is missing a source origin");}
  const metadataPins = /** @type {Array<[string,string]>} */ ([[resultPath, pins.resultSha256], [sdkPath, pins.descriptorSha256],
    [filesPath, pins.filesSha256], [overlayPath, pins.overlaySha256]]);
  for (const [file, expected] of metadataPins) {
    if (fileHash(file) !== expected) {throw new Error("Windows host-SDK inputs changed during admission");}
  }
  return {root, sdkRoot, resultPath, sdkPath, filesPath, overlayPath, pins: {...pins}, result, sdk, manifest, files, actualTree};
}

module.exports = {windowsHostSdkInputs};
