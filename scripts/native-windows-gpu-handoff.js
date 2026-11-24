"use strict";

// Admit one compact, same-run Windows GPU profile bundle. The producer's original SDK
// files are intentionally not transported; their exact receipt map is pinned here.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const {hasExactBuildFields, fileHash, readCanonicalBuildJson} = require("./build-source-inputs");

const hashPattern = /^[a-f0-9]{64}$/;
const maxFiles = 512, maxBytes = 512 << 20, maxHandoffBytes = 8 << 20, maxGraphBytes = 16 << 20;

/** @param {string} directory @param {string} label */
function requireDirectory(directory, label) {
  const stat = fs.lstatSync(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory || path.resolve(directory) !== directory) {
    throw new Error("Windows GPU handoff " + label + " is not a canonical directory");
  }
}

/** @param {string} file @param {string} root @param {string} relative */
function requireRegularMember(file, root, relative) {
  const stat = fs.lstatSync(file), resolved = path.resolve(root, ...relative.split("/"));
  if (resolved !== file || !file.startsWith(root + path.sep) || !stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 ||
      fs.realpathSync(file) !== file || path.relative(root, file).split(path.sep).join("/") !== relative) {
    throw new Error("Windows GPU handoff member is redirected or not a regular file: " + relative);
  }
  return stat;
}

/** Validate one extracted profile bundle against same-run external pins and its unchanged
 * factory receipt. SDK input paths remain producer provenance; the handoff pins their exact
 * map, while the current source, isolation and object files are independently re-admitted.
 * @param {string} receiptFile @param {{profile:string,receiptSha256:string,configSha256:string,handoffSha256:string}} expected */
function windowsGpuHandoffInputs(receiptFile, expected) {
  if (typeof receiptFile !== "string" || !path.isAbsolute(receiptFile) || path.resolve(receiptFile) !== receiptFile ||
      !expected || !hasExactBuildFields(expected, ["profile", "receiptSha256", "configSha256", "handoffSha256"]) ||
      typeof expected.profile !== "string" || !/^(?:acpp-cuda|acpp-hip|dpcpp|dpcpp-opencl|oneapi)$/.test(expected.profile) ||
      !hashPattern.test(expected.receiptSha256 || "") || !hashPattern.test(expected.configSha256 || "") ||
      !hashPattern.test(expected.handoffSha256 || "")) {
    throw new Error("Windows GPU handoff needs exact profile and external SHA-256 pins");
  }
  const output = path.dirname(receiptFile), root = path.dirname(output);
  requireDirectory(root, "root"); requireDirectory(output, "output");
  if (path.basename(receiptFile) !== "result.json" || path.basename(output) !== "output" ||
      path.relative(root, receiptFile).split(path.sep).join("/") !== "output/result.json") {
    throw new Error("Windows GPU handoff receipt is outside its exact output/result.json location");
  }
  const isolatedRoot = path.join(output, "isolated"), configFile = path.join(root, "config.json"),
    isolationFile = path.join(isolatedRoot, "result.json"), handoffFile = path.join(root, "handoff.json");
  requireDirectory(isolatedRoot, "isolated output");
  const graphRecord = readCanonicalBuildJson(receiptFile, maxGraphBytes, "Windows GPU handoff factory receipt"), graph = graphRecord.value;
  if (graphRecord.sha256 !== expected.receiptSha256 || graph.profile !== expected.profile ||
      graph.factory !== "mom_native_gpu_" + expected.profile.replaceAll("-", "_") ||
      graph.configFile !== configFile || graph.configSha256 !== expected.configSha256 ||
      graph.isolation !== isolationFile || !Array.isArray(graph.objects) || graph.objects.length < 1 || graph.objects.length > 128 ||
      !graph.sdkInputs || typeof graph.sdkInputs !== "object" || Array.isArray(graph.sdkInputs) || Object.keys(graph.sdkInputs).length < 1 ||
      Object.keys(graph.sdkInputs).length > 16384 ||
      Object.entries(graph.sdkInputs).some(([file, sha]) => !path.isAbsolute(file) || /[\0\r\n]/.test(file) || !hashPattern.test(sha || ""))) {
    throw new Error("Windows GPU handoff pins differ from its factory receipt");
  }
  const configRecord = readCanonicalBuildJson(configFile, 4 << 20, "Windows GPU handoff profile configuration");
  if (configRecord.sha256 !== expected.configSha256 || configRecord.value.profile !== expected.profile) {
    throw new Error("Windows GPU configuration differs from its external pin");
  }
  const handoffRecord = readCanonicalBuildJson(handoffFile, maxHandoffBytes, "Windows GPU handoff receipt"), handoff = handoffRecord.value;
  if (handoffRecord.sha256 !== expected.handoffSha256 ||
      !hasExactBuildFields(handoff, ["format", "profile", "files", "sdkInputs", "buildOnly", "executed", "publicationQualified"]) ||
      handoff.format !== "mom-windows-gpu-handoff-v1" || handoff.profile !== expected.profile ||
      handoff.buildOnly !== true || handoff.executed !== false || handoff.publicationQualified !== false ||
      !handoff.sdkInputs || typeof handoff.sdkInputs !== "object" || Array.isArray(handoff.sdkInputs) ||
      !isDeepStrictEqual(Object.keys(handoff.sdkInputs), Object.keys(handoff.sdkInputs).sort()) ||
      !isDeepStrictEqual(handoff.sdkInputs, graph.sdkInputs)) {
    throw new Error("Windows GPU handoff receipt differs from its external pin or factory SDK map");
  }
  const expectedMembers = [
    {path: "config.json", file: configFile},
    {path: "output/result.json", file: receiptFile},
    {path: "output/isolated/result.json", file: isolationFile},
  ];
  const objectNames = new Set();
  for (const row of graph.objects) {
    if (!row || typeof row.path !== "string" || path.dirname(row.path) !== isolatedRoot ||
        !/^object-[0-9]{4}\.obj$/i.test(path.basename(row.path)) || !Number.isSafeInteger(row.bytes) || row.bytes < 1 ||
        !hashPattern.test(row.sha256 || "") || objectNames.has(path.basename(row.path).toLowerCase())) {
      throw new Error("Windows GPU handoff factory object is outside its exact isolated output directory");
    }
    objectNames.add(path.basename(row.path).toLowerCase());
    expectedMembers.push({path: "output/isolated/" + path.basename(row.path), file: row.path});
  }
  expectedMembers.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  if (!Array.isArray(handoff.files) || handoff.files.length !== expectedMembers.length || handoff.files.length > maxFiles) {
    throw new Error("Windows GPU handoff file inventory is incomplete or excessive");
  }
  const files = /** @type {Record<string, string>} */ ({}); let totalBytes = 0;
  for (let index = 0; index < expectedMembers.length; index++) {
    const expectedFile = expectedMembers[index], row = handoff.files[index];
    if (!expectedFile || !hasExactBuildFields(row, ["path", "bytes", "sha256"]) || row.path !== expectedFile.path ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || !hashPattern.test(row.sha256 || "")) {
      throw new Error("Windows GPU handoff file inventory differs from the exact factory cohort");
    }
    const stat = requireRegularMember(expectedFile.file, root, expectedFile.path);
    if (stat.size !== row.bytes || fileHash(expectedFile.file) !== row.sha256) {
      throw new Error("Windows GPU handoff payload differs from its pinned inventory: " + row.path);
    }
    totalBytes += stat.size;
    if (!Number.isSafeInteger(totalBytes) || totalBytes > maxBytes) {throw new Error("Windows GPU handoff exceeds its bounded payload size");}
    files[expectedFile.file] = row.sha256;
  }
  const expectedIsolated = ["result.json"];
  for (const row of graph.objects) {expectedIsolated.push(path.basename(row.path));}
  expectedIsolated.sort();
  if (!isDeepStrictEqual(fs.readdirSync(root).sort(), ["config.json", "handoff.json", "output"]) ||
      !isDeepStrictEqual(fs.readdirSync(output).sort(), ["isolated", "result.json"]) ||
      !isDeepStrictEqual(fs.readdirSync(isolatedRoot).sort(), expectedIsolated)) {
    throw new Error("Windows GPU handoff bundle has extra, missing or substituted members");
  }
  files[handoffFile] = handoffRecord.sha256;
  return {profile: expected.profile, graph, files, handoffFile, handoffSha256: handoffRecord.sha256};
}

module.exports = {windowsGpuHandoffInputs};
