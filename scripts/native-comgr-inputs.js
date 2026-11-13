"use strict";

// Shared producer/consumer contract for a locally built compiler component. This verifies
// content/provenance continuity; a receipt is not a signature for untrusted downloaded code.
const fs = require("node:fs"), path = require("node:path");
const {fileHash, sourceInputs, verifySourceInputs, readBuildJson} = require("./build-source-inputs");
const {isStaticArchive} = require("./native-audit");
const revision = "27682a16360e33e37c4f3cc6adf9a620733f8fe1";
const patches = ["runtime/rocm-comgr-llvm21.patch", "runtime/rocm-comgr-resources.patch"];
const recipeFiles = ["scripts/build-native-comgr-sdk.js", "scripts/build-native-sycl-sdk.js", "scripts/build-source-inputs.js",
  "scripts/build-environment.js", "scripts/probe-native-jit.js", "scripts/native-audit.js", "scripts/native-comgr-inputs.js", ...patches];

/** Read and rehash the exact completed source producer, never copy an archive into a fake
 * SDK prefix or restamp stale provenance. Dependency paths are explicit local build inputs;
 * no recursive SDK/cache discovery, tool execution, driver initialization or downloads.
 * @param {string} receipt */
function comgrSdkInputs(receipt) {
  if (!path.isAbsolute(receipt)) {throw new Error("COMGR receipt must be absolute");}
  const receiptSha = fileHash(receipt), record = readBuildJson(receipt);
  const root = path.dirname(fs.realpathSync(receipt)), repo = path.resolve(__dirname, "..");
  const archive = path.join(root, "build/libamd_comgr.a"), header = path.join(root, "build/include/amd_comgr.h");
  if (record?.version !== 1 || record.revision !== revision || record.llvmVersion !== "21.1.8" ||
      record.buildOnly !== true || record.publicationQualified !== false || record.archive !== archive || record.header !== header ||
      record.license !== path.join(root, "source/amd/comgr/LICENSE.txt") || record.notices !== path.join(root, "source/amd/comgr/NOTICES.txt") ||
      !record.sdkRoots || Object.keys(record.sdkRoots).sort().join(",") !== "device,lld,llvm" ||
      Object.values(record.sdkRoots).some(value => typeof value !== "string" || !path.isAbsolute(value) || fs.realpathSync(value) !== value)) {
    throw new Error("Invalid pinned source COMGR receipt");
  }
  if (fs.realpathSync(archive) !== archive || fs.realpathSync(header) !== header || !isStaticArchive(archive) ||
      fileHash(archive) !== record.sha256 || fileHash(header) !== record.headerSha256) {
    throw new Error("COMGR archive/header differs from its producer receipt");
  }
  const recipe = sourceInputs(repo, [], recipeFiles), snapshot = path.join(root, "source");
  verifySourceInputs(record.recipe, recipe, "COMGR producer recipe");
  const source = sourceInputs(snapshot, ["amd/comgr"], []);
  verifySourceInputs(record.sourceState, source, "COMGR patched source");
  /** @type {Record<string, string>} */
  const files = {[receipt]: receiptSha, [archive]: record.sha256, [header]: record.headerSha256};
  for (const group of [record.sdkInputs, record.dependencies]) {
    if (!group || typeof group !== "object" || Array.isArray(group) || !Object.keys(group).length || Object.keys(group).length > 4096) {
      throw new Error("Missing or excessive COMGR SDK dependency receipt");
    }
    for (const [file, sha] of Object.entries(group)) {
      if (!path.isAbsolute(file) || typeof sha !== "string" || !/^[0-9a-f]{64}$/.test(sha) || fileHash(file) !== sha) {
        throw new Error("COMGR SDK dependency changed: " + file);
      }
      files[file] = sha;
    }
  }
  for (const [file, sha] of Object.entries(source.files)) {files[path.join(snapshot, file)] = sha;}
  for (const [file, sha] of Object.entries(recipe.graph)) {files[path.join(repo, file)] = sha;}
  if (fileHash(receipt) !== receiptSha) {throw new Error("COMGR receipt changed during admission");}
  return {archive, header, receipt, llvmVersion: record.llvmVersion,
    roots: /** @type {{llvm: string, lld: string, device: string}} */ (record.sdkRoots), files};
}

/** The embedded HIPRTC builtins and AdaptiveCpp translator must consume the same SDKs.
 * @param {ReturnType<typeof comgrSdkInputs> | undefined} component
 * @param {string} llvm @param {string} lld @param {string} bitcode @param {string} version */
function verifyComgrSdkSelection(component, llvm, lld, bitcode, version) {
  if (!component || component.llvmVersion !== version || component.roots.llvm !== fs.realpathSync(llvm) ||
      component.roots.lld !== fs.realpathSync(lld) || path.join(component.roots.device, "amdgcn/bitcode") !== fs.realpathSync(bitcode)) {
    throw new Error("COMGR and AdaptiveCpp must use the same LLVM/LLD/device SDKs");
  }
}

module.exports = {revision, patches, recipeFiles, comgrSdkInputs, verifyComgrSdkSelection};
