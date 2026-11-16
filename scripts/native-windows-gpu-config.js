"use strict";

// Derive GPU configs from separately admitted same-run source, SDK and tool inputs.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto"), childProcess = require("node:child_process"),
  {spawnSync} = childProcess,
  {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson, verifySourceInputs} = require("./build-source-inputs");
const {currentStaticSyclRecipe} = require("./native-release-sdk-inputs");
const windowsHostSdkReader = require("./native-windows-host-sdk-inputs"), linuxLlvmToolReader = require("./native-linux-llvm-tools");
const {windowsHostSdkInputs} = windowsHostSdkReader;
const {linuxLlvmToolHandoffInputs} = linuxLlvmToolReader;
const {vfsFiles} = require("./build-native-windows-controller");
const {resolveOneapiCompiler, validateWindowsGpuConfig} = require("./build-native-windows-gpu");
const {releaseBuildEnvironment, nativeSdkEnvironment} = require("./build-environment");
const {pins: dpcppPins, sdkIdentity: dpcppSdkIdentity} = require("./probe-static-dpcpp");
const {syclRevision} = require("./probe-static-sycl");
const hipHeaderReader = require("./native-rocm-hip-headers");

const dpcppSdkHeaders = Object.freeze([
  "CL/cl.h", "CL/cl_ext.h", "CL/cl_platform.h", "CL/cl_version.h",
  "sycl/device_aspect_macros.hpp", "sycl/feature_test.hpp", "sycl/version.hpp",
  "unified-runtime/ur_api.h", "unified-runtime/ur_api_funcs.def",
]);
const urHeaders = Object.freeze(["ur_api.h", "ur_api_funcs.def"]);
const windowsSdkVirtualIncludeRoots = Object.freeze([
  "/mom-sdk/msvc/include", "/mom-sdk/sdk/include/ucrt", "/mom-sdk/sdk/include/shared",
  "/mom-sdk/sdk/include/um", "/mom-sdk/sdk/include/winrt",
]);
/** @typedef {ReturnType<typeof import("./prepare-native-ci").dpcppCudaInputsStatePaths>} DpcppCudaInputsState */
/** @typedef {{resourceInclude:string,syclInclude:string,dpcppInclude:string,dpcppSyclInclude:string,urInclude:string,windowsSdkIncludeRoots:string[]}} WindowsGpuIncludePlanInput */
/** @typedef {WindowsGpuIncludePlanInput} WindowsOneapiIncludePlanInput */
/** @typedef {{root:string,archiveSha256:string,marker:{path:string,sha256:string},clang:{path:string,sha256:string},clangxx:{path:string,sha256:string},clangVersion:string}} DpcppCompilerRecord */

/** @param {unknown} value @param {string[]} keys */
function hasExactKeys(value, keys) {
  return !!value && typeof value === "object" && !Array.isArray(value) &&
    Object.keys(value).sort().join("\0") === [...keys].sort().join("\0");
}

/** @param {unknown} value @param {string} label */
function canonicalDirectory(value, label) {
  if (typeof value !== "string" || !path.isAbsolute(value) || path.resolve(value) !== value || /[\r\n\0]/.test(value)) {
    throw new Error(label + " must be an absolute canonical directory");
  }
  const stat = fs.lstatSync(value);
  if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(value) !== value) {throw new Error(label + " is redirected or not a directory");}
  return value;
}

/** @param {unknown} value @returns {value is string} */
function absolutePath(value) {
  return typeof value === "string" && path.isAbsolute(value) && path.resolve(value) === value && !/[\r\n\0]/.test(value);
}

/** @param {string} file @param {string} label @param {number} maximum */
function stableRegular(file, label, maximum) {
  const before = fs.lstatSync(file, {bigint: true});
  if (!before.isFile() || before.isSymbolicLink() || before.nlink !== 1n || before.size < 1n || before.size > BigInt(maximum) ||
      fs.realpathSync(file) !== file) {throw new Error(label + " must be a bounded canonical regular file");}
  const sha256 = fileHash(file), after = fs.lstatSync(file, {bigint: true});
  if (before.dev !== after.dev || before.ino !== after.ino || before.nlink !== after.nlink || before.size !== after.size ||
      before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs || before.mode !== after.mode) {
    throw new Error(label + " changed while being hashed");
  }
  return {bytes: Number(after.size), sha256};
}

/** @param {string} root @param {string} file @param {string} label */
function containedFile(root, file, label) {
  const stat = fs.lstatSync(file);
  const relative = path.relative(root, file);
  if (!path.isAbsolute(file) || path.resolve(file) !== file || !stat.isFile() || stat.isSymbolicLink() ||
      fs.realpathSync(file) !== file || path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep)) {
    throw new Error(label + " is not a regular file beneath its admitted root");
  }
  return file;
}

const maxWindowsAcppSdkArchiveBytes = 16 * 1024 ** 3, maxWindowsAcppSdkExpandedBytes = 32 * 1024 ** 3,
  maxWindowsAcppGeneratedFiles = 64, maxWindowsAcppGeneratedBytes = 16 * 1024 ** 2,
  maxWindowsAcppGeneratedDirectories = 128, maxWindowsAcppGeneratedDepth = 8, maxWindowsAcppTarInspectionMs = 5 * 60 * 1000;

/** Admit only the selected compiler plugin and generated headers from the externally pinned,
 * unchanged same-run ACPP build handoff. The tar SHA is a trusted workflow handoff pin, not publisher authentication.
 * @param {{archive:string,bundleSha256:string,root:string,sourceRoot:string}} input @param {string} suffix */
function windowsNativeAcppInputs(input, suffix) {
  if (process.platform !== "linux" || process.arch !== "x64" || !hasExactKeys(input, ["archive", "bundleSha256", "root", "sourceRoot"]) ||
      !/^[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(suffix) || !/^[a-f0-9]{64}$/.test(input.bundleSha256 || "")) {
    throw new Error("Windows ACPP inputs require the exact same-run Linux handoff and archive pin");
  }
  const runnerTemp = process.env["RUNNER_TEMP"] || "", incoming = path.join(runnerTemp, "incoming"),
    sdkName = "mom-native-acpp-cuda-hip-sdk-" + suffix, sdkRoot = path.join("/tmp", sdkName), build = path.join(sdkRoot, "build"),
    sourceName = "mom-native-acpp-prerequisites-" + suffix, sourceRoot = path.join("/tmp", sourceName, "prepared", "acpp"),
    archive = path.join(incoming, "native-linux-acpp-cuda-hip-sdk-" + suffix + ".tar");
  if (!path.isAbsolute(runnerTemp) || path.resolve(runnerTemp) !== runnerTemp || fs.realpathSync(runnerTemp) !== runnerTemp ||
      fs.realpathSync(incoming) !== incoming || input.archive !== archive || input.root !== sdkRoot || input.sourceRoot !== sourceRoot) {
    throw new Error("Windows ACPP input paths differ from the canonical same-run roots");
  }
  canonicalDirectory(sdkRoot, "Restored ACPP SDK root"); canonicalDirectory(build, "Restored ACPP build root");
  canonicalDirectory(sourceRoot, "Prepared ACPP source root");

  const archivePin = stableRegular(archive, "ACPP SDK bundle", maxWindowsAcppSdkArchiveBytes);
  if (archivePin.sha256 !== input.bundleSha256) {throw new Error("ACPP SDK tar differs from its external same-run pin");}
  const ci = require("./prepare-native-ci"), inspectionDeadline = Date.now() + maxWindowsAcppTarInspectionMs,
    remainingInspectionMs = () => {
      const remaining = inspectionDeadline - Date.now();
      if (remaining < 1000) {throw new Error("ACPP tar inspection exceeded its shared five-minute budget");}
      return remaining;
    }, members = ci.listState(archive, [sdkName], remainingInspectionMs());
  if (members.length > 200000 || new Set(members.map(name => name.endsWith("/") ? name.slice(0, -1) : name)).size !== members.length) {
    throw new Error("ACPP SDK tar has duplicate or excessive members");
  }
  const expandedBytes = ci.tarExpandedBytes(archive, members.length, remainingInspectionMs());
  if (expandedBytes > maxWindowsAcppSdkExpandedBytes) {throw new Error("ACPP SDK tar exceeds its expanded-size bound");}

  const resultFile = path.join(sdkRoot, "result.json"), planFile = path.join(sdkRoot, "plan.json"),
    resultPin = stableRegular(resultFile, "Original ACPP producer result", 16 << 20),
    planPin = stableRegular(planFile, "Original ACPP producer plan", 16 << 20),
    result = readBuildJson(resultFile), plan = readBuildJson(planFile), inputs = result.inputs;
  if (result.version !== 1 || result.backend !== "cuda-hip" || result.revision !== syclRevision || result.cudaVersion !== 12060 ||
      result.buildOnly !== true || result.publicationQualified !== false || result.executed === true ||
      !inputs || typeof inputs !== "object" || Array.isArray(inputs) || inputs.source !== sourceRoot || inputs.build !== build ||
      !absolutePath(inputs.cxx) || !Array.isArray(inputs.flags) || inputs.flags.length > 128 ||
      inputs.flags.some((/** @type {unknown} */ flag) => typeof flag !== "string" || flag.length > 4096 || flag.includes("\0")) ||
      !Array.isArray(inputs.libraries) || !inputs.libraries.length || inputs.libraries.length > 512 ||
      inputs.libraries.some((/** @type {unknown} */ file) => !absolutePath(file)) || !Array.isArray(inputs.links) || !inputs.links.length || inputs.links.length > 4096 ||
      inputs.links.some((/** @type {unknown} */ flag) => typeof flag !== "string" || flag.length > 4096 || flag.includes("\0")) ||
      !plan || typeof plan !== "object" || plan.version !== "21.1.8" ||
      !isDeepStrictEqual(plan.admitted, {source: sourceRoot, revision: syclRevision}) || !Array.isArray(plan.commands) || plan.commands.length !== 2 ||
      plan.commands.some((/** @type {unknown} */ command) => !Array.isArray(command) || !command.length || command.length > 1024 ||
        command.some((/** @type {unknown} */ token) => typeof token !== "string" || token.length > 4096 || token.includes("\0"))) ||
      !isDeepStrictEqual(plan.hostCompilers, result.hostCompilers) || !isDeepStrictEqual(plan.hostCompilerInputs, result.hostCompilerInputs) ||
      !isDeepStrictEqual(plan.compilerVersions, result.compilerVersions) || !isDeepStrictEqual(plan.fingerprints, result.fingerprints) ||
      !isDeepStrictEqual(plan.recipe, result.recipe)) {
    throw new Error("Original ACPP receipt/plan does not bind the pinned same-run source and build");
  }
  verifySourceInputs(result.recipe, currentStaticSyclRecipe(), "Transported AdaptiveCpp SDK recipe");
  const plugin = path.join(build, "src/compiler/libacpp-clang.so"), pluginFlags = inputs.flags.filter((/** @type {string} */ flag) => flag.startsWith("-fplugin=")),
    passPluginFlags = inputs.flags.filter((/** @type {string} */ flag) => flag.startsWith("-fpass-plugin="));
  if (pluginFlags.length !== 1 || passPluginFlags.length !== 1 || pluginFlags[0] !== "-fplugin=" + plugin ||
      passPluginFlags[0] !== "-fpass-plugin=" + plugin) {
    throw new Error("Original ACPP flags do not select the exact in-tree LLVM plugin");
  }
  const pluginFile = containedFile(sdkRoot, plugin, "Restored ACPP LLVM plugin"), pluginPin = stableRegular(pluginFile, "Restored ACPP LLVM plugin", 128 << 20),
    pluginMember = sdkName + "/build/src/compiler/libacpp-clang.so";
  if (!members.includes(pluginMember) || !members.includes(sdkName + "/result.json") || !members.includes(sdkName + "/plan.json")) {
    throw new Error("Pinned ACPP tar omits a selected plugin or original producer receipt");
  }

  const generatedInclude = path.join(build, "include"), generatedRelative = "build/include",
    configMember = sdkName + "/" + generatedRelative + "/hipSYCL/common/config.hpp",
    configFile = path.join(generatedInclude, "hipSYCL/common/config.hpp");
  if (!members.includes(configMember)) {throw new Error("Pinned ACPP tar omits the generated hipSYCL config header");}
  canonicalDirectory(generatedInclude, "Restored ACPP generated include root");
  const configPin = stableRegular(containedFile(sdkRoot, configFile, "Restored ACPP generated config header"),
    "Restored ACPP generated config header", 16 << 20);

  const generatedFiles = generatedIncludeInventory(generatedInclude);
  const generatedMembers = members.filter(name => !name.endsWith("/") && name.startsWith(sdkName + "/" + generatedRelative + "/"))
    .map(name => name.slice((sdkName + "/" + generatedRelative + "/").length)).sort();
  if (!isDeepStrictEqual(generatedMembers, generatedFiles.map(file => file.path))) {
    throw new Error("Restored ACPP generated headers differ from the pinned tar member set");
  }
  for (const file of generatedFiles) {
    const memberPin = tarMemberPin(archive, sdkName + "/" + generatedRelative + "/" + file.path, 16 << 20, remainingInspectionMs());
    if (memberPin.bytes !== file.bytes || memberPin.sha256 !== file.sha256) {
      throw new Error("Restored ACPP generated header differs from the externally pinned tar member");
    }
  }

  const sourceReceiptFile = path.join(path.dirname(sourceRoot), "sources.json"), sourceReceiptPin = stableRegular(sourceReceiptFile,
      "Prepared ACPP source receipt", 16 << 20), sourceHeaderRelative = "include/hipSYCL/glue/llvm-sscp/hcf_registration.hpp",
    sourceHeader = path.join(sourceRoot, ...sourceHeaderRelative.split("/"));
  /** @param {string} file */
  const admitSource = (file) => {
    const sourceReceipt = require("./prepare-native-sdk-sources").verifyPreparedSdkSources(file, "acpp"), acpp = sourceReceipt?.prepared?.acpp,
      headerPin = stableRegular(sourceHeader, "Prepared ACPP HCF registration header", 16 << 20);
    if (sourceReceipt.version !== 2 || sourceReceipt.target !== "acpp" || acpp?.root !== sourceRoot || acpp?.revision !== syclRevision ||
        acpp.files?.[sourceHeaderRelative] !== headerPin.sha256) {
      throw new Error("Prepared ACPP source receipt does not bind the required HCF registration header");
    }
    return {receipt: sourceReceipt, header: headerPin};
  };
  const sourceBefore = admitSource(sourceReceiptFile), tarResultPin = tarMemberPin(archive, sdkName + "/result.json", 16 << 20, remainingInspectionMs()),
    tarPlanPin = tarMemberPin(archive, sdkName + "/plan.json", 16 << 20, remainingInspectionMs()),
    tarPluginPin = tarMemberPin(archive, pluginMember, 128 << 20, remainingInspectionMs());
  if (tarResultPin.bytes !== resultPin.bytes || tarResultPin.sha256 !== resultPin.sha256 ||
      tarPlanPin.bytes !== planPin.bytes || tarPlanPin.sha256 !== planPin.sha256 ||
      tarPluginPin.bytes !== pluginPin.bytes || tarPluginPin.sha256 !== pluginPin.sha256) {
    throw new Error("Restored ACPP receipt, plugin or config header differs from the externally pinned tar member");
  }
  const sourceAfter = admitSource(sourceReceiptFile), generatedAfter = generatedIncludeInventory(generatedInclude);
  if (!isDeepStrictEqual(sourceBefore, sourceAfter) || sourceReceiptPin.sha256 !== stableRegular(sourceReceiptFile, "Prepared ACPP source receipt", 16 << 20).sha256 ||
      sourceBefore.header.sha256 !== fileHash(sourceHeader) || !isDeepStrictEqual(generatedFiles, generatedAfter) ||
      pluginPin.sha256 !== stableRegular(pluginFile, "Restored ACPP LLVM plugin", 128 << 20).sha256 ||
      configPin.sha256 !== stableRegular(configFile, "Restored ACPP generated config header", 16 << 20).sha256 ||
      archivePin.sha256 !== stableRegular(archive, "ACPP SDK bundle", maxWindowsAcppSdkArchiveBytes).sha256 ||
      resultPin.sha256 !== stableRegular(resultFile, "Original ACPP producer result", 16 << 20).sha256 ||
      planPin.sha256 !== stableRegular(planFile, "Original ACPP producer plan", 16 << 20).sha256) {
    throw new Error("ACPP source, bundle, receipt, plugin or generated headers changed during admission");
  }
  return {source: sourceRoot, generatedInclude, plugin,
    hashes: {bundleSha256: archivePin.sha256, bundleBytes: archivePin.bytes, resultSha256: resultPin.sha256, planSha256: planPin.sha256,
      sourceReceiptSha256: sourceReceiptPin.sha256, sourceHeaderSha256: sourceBefore.header.sha256,
      pluginSha256: pluginPin.sha256, generatedConfigSha256: configPin.sha256,
      generatedIncludeSha256: createHash("sha256").update(JSON.stringify(generatedFiles)).digest("hex")},
    generatedFiles};
}

/** Hash one exact selected file from the pinned GNU tar member without extracting anything else.
 * @param {string} archive @param {string} member @param {number} maximum @param {number} timeoutMs */
function tarMemberPin(archive, member, maximum, timeoutMs) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > maxWindowsAcppTarInspectionMs) {
    throw new Error("ACPP tar member read is outside its shared inspection budget");
  }
  const result = spawnSync("/usr/bin/tar", ["--extract", "--to-stdout", "--file", archive, "--", member],
    {timeout: timeoutMs, maxBuffer: maximum});
  if (result.error || result.signal || result.status !== 0 || !Buffer.isBuffer(result.stdout) || result.stdout.length > maximum) {
    throw new Error("Cannot read selected ACPP member from the pinned tar: " + path.posix.basename(member), {cause: result.error});
  }
  return {bytes: result.stdout.length, sha256: createHash("sha256").update(result.stdout).digest("hex")};
}

/** @param {string} root @returns {Array<{path:string,bytes:number,sha256:string}>} */
function generatedIncludeInventory(root) {
  const files = /** @type {Array<{path:string,bytes:number,sha256:string}>} */ ([]); let bytes = 0, directories = 0;
  /** @param {string} directory @param {string} relative @param {number} depth */
  function visit(directory, relative, depth) {
    if (depth > maxWindowsAcppGeneratedDepth) {throw new Error("Generated ACPP include tree exceeds its depth bound");}
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0)) {
      const file = path.join(directory, entry.name), rel = relative ? relative + "/" + entry.name : entry.name,
        stat = fs.lstatSync(file);
      if (stat.isSymbolicLink()) {throw new Error("Generated ACPP include tree contains a symlink");}
      if (stat.isDirectory()) {
        if (++directories > maxWindowsAcppGeneratedDirectories) {throw new Error("Generated ACPP include tree exceeds its directory bound");}
        visit(file, rel, depth + 1); continue;
      }
      if (!stat.isFile() || files.length >= maxWindowsAcppGeneratedFiles) {throw new Error("Generated ACPP include tree is not bounded regular files");}
      const pin = stableRegular(file, "Generated ACPP header", 16 << 20); bytes += pin.bytes;
      if (!Number.isSafeInteger(bytes) || bytes > maxWindowsAcppGeneratedBytes) {throw new Error("Generated ACPP include tree exceeds its byte bound");}
      files.push({path: rel, ...pin});
    }
  }
  visit(root, "", 0);
  return files.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
}

/** Derive ordered include and selected-header paths without treating SDK VFS names as host paths.
 * @param {WindowsGpuIncludePlanInput} input
 */
function windowsNativeGpuIncludePlan(input) {
  const keys = ["resourceInclude", "syclInclude", "dpcppInclude", "dpcppSyclInclude", "urInclude", "windowsSdkIncludeRoots"];
  if (!hasExactKeys(input, keys)) {throw new Error("Windows GPU include roots do not match the pinned physical and SDK VFS layout");}
  const physicalRoots = [input.resourceInclude, input.syclInclude, input.dpcppInclude, input.dpcppSyclInclude, input.urInclude];
  if (physicalRoots.some(value => {
    return typeof value !== "string" || !path.isAbsolute(value) || path.resolve(value) !== value || /[\r\n\0]/.test(value);
  }) || !Array.isArray(input.windowsSdkIncludeRoots) || input.windowsSdkIncludeRoots.length !== windowsSdkVirtualIncludeRoots.length ||
      input.windowsSdkIncludeRoots.some((/** @type {string} */ root, /** @type {number} */ index) => root !== windowsSdkVirtualIncludeRoots[index])) {
    throw new Error("Windows GPU include roots do not match the pinned physical and SDK VFS layout");
  }
  if (input.dpcppSyclInclude !== path.join(input.dpcppInclude, "sycl")) {
    throw new Error("Windows GPU DPC++ SYCL include root is not beneath the admitted SDK include root");
  }
  const includeRoots = [input.resourceInclude, input.syclInclude, input.dpcppInclude, input.dpcppSyclInclude,
    input.urInclude, ...input.windowsSdkIncludeRoots];
  if (new Set(includeRoots).size !== includeRoots.length) {throw new Error("Windows GPU config has duplicate include roots");}
  return {
    includeRoots,
    dpcppHeaders: dpcppSdkHeaders.map(relative => path.join(input.dpcppInclude, ...relative.split("/"))),
    urHeaders: urHeaders.map(relative => path.join(input.urInclude, "unified-runtime", relative)),
  };
}

/** Preserve the original public helper name for callers/tests that explicitly select oneAPI.
 * @param {WindowsOneapiIncludePlanInput} input */
function windowsNativeOneapiIncludePlan(input) {return windowsNativeGpuIncludePlan(input);}

/** Resolve only the pinned DPC++ SDK's two frontend drivers and resource headers.
 * @param {string} root @param {DpcppCompilerRecord} record @param {NodeJS.ProcessEnv} environment
 * @param {(command:string,args:string[],options:import("node:child_process").SpawnSyncOptionsWithStringEncoding)=>import("node:child_process").SpawnSyncReturns<string>} [spawn] */
function resolveDpcppCompiler(root, record, environment, spawn = spawnSync) {
  canonicalDirectory(root, "DPC++ SDK compiler root");
  if (!record || typeof record !== "object") {throw new Error("DPC++ frontend does not match the admitted SDK record");}
  const rows = /** @type {Array<[string, DpcppCompilerRecord["marker"]]>} */ ([
    [".mom-toolchain-sha256", record.marker], ["bin/clang", record.clang], ["bin/clang++", record.clangxx]]);
  if (record.root !== root || record.archiveSha256 !== dpcppPins.sdk || typeof record.clangVersion !== "string" || !record.clangVersion ||
      rows.some(([relative, row]) => !hasExactKeys(row, ["path", "sha256"]) || row.path !== path.join(root, relative) ||
        !/^[a-f0-9]{64}$/.test(row.sha256 || ""))) {throw new Error("DPC++ frontend does not match the admitted SDK record");}
  const inputs = /** @type {Record<string,string>} */ ({});
  for (const [relative, row] of rows) {
    const file = /** @type {string} */ (row.path), before = fs.statSync(file, {bigint: true}), real = fs.realpathSync(file), rel = path.relative(root, real);
    if (!before.isFile() || path.isAbsolute(rel) || rel === ".." || rel.startsWith(".." + path.sep) ||
        (relative.startsWith("bin/") && (before.mode & 0o111n) === 0n)) {
      throw new Error("DPC++ frontend input is missing, non-executable, or outside its SDK: " + relative);
    }
    const sha256 = fileHash(file), after = fs.statSync(file, {bigint: true});
    if (before.dev !== after.dev || before.ino !== after.ino || before.size !== after.size || before.mtimeNs !== after.mtimeNs ||
        before.ctimeNs !== after.ctimeNs || sha256 !== row.sha256) {throw new Error("DPC++ frontend input changed or differs from its receipt: " + relative);}
    if (relative === ".mom-toolchain-sha256") {
      dpcppSdkIdentity(record.clangVersion, fs.readFileSync(file, "utf8"));
    }
    inputs[file] = sha256;
  }
  /** @param {string} command @param {string[]} args @param {string} label */
  const query = (command, args, label) => {
    const result = spawn(command, args, {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20, env: environment});
    if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string" || result.stdout.length > (1 << 20)) {
      throw new Error("DPC++ " + label + " query failed", {cause: result.error});
    }
    return result.stdout.trim();
  };
  const clang = path.join(root, "bin/clang"), clangxx = path.join(root, "bin/clang++");
  for (const [file, label] of /** @type {Array<[string,string]>} */ ([[clangxx, "clang++"], [clang, "clang"]])) {
    const version = query(file, ["--version"], label + " version");
    dpcppSdkIdentity(version, fs.readFileSync(path.join(root, ".mom-toolchain-sha256"), "utf8"));
    if (version !== record.clangVersion) {throw new Error("DPC++ " + label + " version differs from the admitted SDK record");}
  }
  const resourceDirs = /** @type {string[]} */ ([clangxx, clang].map((file, index) => canonicalSdkDirectory(root,
    query(file, ["--print-resource-dir"], index === 0 ? "clang++ resource directory" : "clang resource directory"))));
  const resourceDir = resourceDirs[0] || "";
  if (!resourceDir || resourceDir !== resourceDirs[1]) {throw new Error("DPC++ clang and clang++ resource directories differ");}
  const resourceInclude = canonicalSdkDirectory(root, path.join(resourceDir, "include"));
  for (const [, row] of rows) {
    if (fileHash(row.path) !== row.sha256) {throw new Error("DPC++ frontend input changed during compiler queries: " + row.path);}
  }
  return {root, tools: {clang: clangxx, clangC: clang}, resourceDir, resourceInclude, inputs};
}

/** @param {string} root @param {string} directory */
function canonicalSdkDirectory(root, directory) {
  if (typeof directory !== "string" || !path.isAbsolute(directory) || path.resolve(directory) !== directory || /[\r\n\0]/.test(directory)) {
    throw new Error("DPC++ resource directory is not canonical");
  }
  const stat = fs.statSync(directory), real = fs.realpathSync(directory), relative = path.relative(root, real);
  if (!stat.isDirectory() || path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep)) {
    throw new Error("DPC++ resource directory is outside the admitted SDK");
  }
  return real;
}

/** Validate virtual SDK roots against the admitted VFS and physical file inventory.
 * @param {ReturnType<typeof windowsHostSdkInputs>} host
 */
function validateWindowsSdkVfsRoots(host) {
  const overlay = readBuildJson(host.overlayPath), virtualFiles = vfsFiles(host.overlayPath), insensitive = overlay?.["case-sensitive"] === false;
  if (!overlay || !Array.isArray(overlay.roots) || overlay.roots.length !== virtualFiles.size ||
      !Array.isArray(host.sdk.includeRoots) || host.sdk.includeRoots.length !== windowsSdkVirtualIncludeRoots.length ||
      host.sdk.includeRoots.some((/** @type {string} */ root, /** @type {number} */ index) => root !== windowsSdkVirtualIncludeRoots[index])) {
    throw new Error("Windows host-SDK virtual include roots differ from the pinned VFS layout");
  }
  const rootKeys = host.sdk.includeRoots.map((/** @type {string} */ root) => insensitive ? root.toLowerCase() : root),
    counts = new Map(rootKeys.map((/** @type {string} */ root) => [root, 0]));
  for (const [virtualFile, physicalFile] of virtualFiles) {
    const virtualKey = insensitive ? virtualFile.toLowerCase() : virtualFile;
    const matches = rootKeys.filter((/** @type {string} */ root) => virtualKey.startsWith(root + "/"));
    if (matches.length !== 1 || !path.isAbsolute(physicalFile) || path.resolve(physicalFile) !== physicalFile) {
      throw new Error("Windows host-SDK VFS file is outside its virtual include roots");
    }
    const relative = path.relative(host.sdkRoot, physicalFile), row = host.files.get(relative.normalize("NFC").toLowerCase());
    if (path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep) ||
        !row || row.path !== relative || path.join(host.sdkRoot, row.path) !== physicalFile) {
      throw new Error("Windows host-SDK VFS target is not in its admitted physical inventory");
    }
    counts.set(matches[0], (counts.get(matches[0]) || 0) + 1);
  }
  if ([...counts.values()].some(count => count < 1)) {throw new Error("Windows host-SDK VFS has an empty declared include root");}
  return host.sdk.includeRoots;
}

/**
 * Build a deterministic oneAPI, DPC++, or DPC++ OpenCL GPU descriptor without compiling or writing receipts.
 * @param {string} profile
 * @param {{windowsSdkRoot:string,windowsSdkPins:{resultSha256:string,descriptorSha256:string,filesSha256:string,overlaySha256:string},
 * windowsSourcesFile:string,dpcppState:DpcppCudaInputsState,dpcppManifestSha256:string,
 * toolsRoot:string,toolsExpectation:{runAttempt:string,handoffSha256:string},compilerRoot:string,
 * cudaImages?:{archive:string,bundleSha256:string,root:string}}} input
 */
function windowsNativeGpuConfig(profile, input) {
  const baseInputKeys = ["windowsSdkRoot", "windowsSdkPins", "windowsSourcesFile", "dpcppState", "dpcppManifestSha256",
      "toolsRoot", "toolsExpectation", "compilerRoot"], inputKeys = profile === "dpcpp" ? [...baseInputKeys, "cudaImages"] : baseInputKeys;
  if (!hasExactKeys(input, inputKeys) || typeof input.dpcppManifestSha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(input.dpcppManifestSha256) || !hasExactKeys(input.toolsExpectation, ["runAttempt", "handoffSha256"]) ||
      typeof input.toolsExpectation.runAttempt !== "string" || !/^[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(input.toolsExpectation.runAttempt) ||
      typeof input.toolsExpectation.handoffSha256 !== "string" || !/^[a-f0-9]{64}$/.test(input.toolsExpectation.handoffSha256) ||
      typeof input.windowsSdkRoot !== "string" || !path.isAbsolute(input.windowsSdkRoot) ||
      typeof input.windowsSourcesFile !== "string" || !path.isAbsolute(input.windowsSourcesFile) ||
      typeof input.toolsRoot !== "string" || !path.isAbsolute(input.toolsRoot) ||
      typeof input.compilerRoot !== "string" || !path.isAbsolute(input.compilerRoot)) {
    throw new Error("Windows GPU config requires exact same-run inputs and external pins");
  }
  if (!["oneapi", "dpcpp", "dpcpp-opencl"].includes(profile)) {throw new Error("Unsupported Windows GPU config profile");}
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Windows GPU config is composed on Linux x64 only");}

  const suffix = input.toolsExpectation.runAttempt, expectedDpcppName = "mom-native-dpcpp-cuda-inputs-" + suffix,
    state = input.dpcppState;
  if (!state || typeof state !== "object" || typeof state.root !== "string" || typeof state.name !== "string" ||
      typeof state.dpcppRoot !== "string" || typeof state.manifest !== "string" || !state.linux ||
      typeof state.linux.source !== "string" || typeof state.linux.sourceName !== "string" || typeof state.linux.prepared !== "string" ||
      state.name !== expectedDpcppName || !path.isAbsolute(state.root) ||
      path.basename(state.root) !== expectedDpcppName || state.dpcppRoot !== path.join(state.root, "dpcpp") ||
      state.manifest !== path.join(state.root, "inputs.json") ||
      state.linux.sourceName !== "mom-native-linux-source-prerequisites-" + suffix ||
      path.basename(state.linux.source) !== state.linux.sourceName || state.linux.prepared !== path.join(state.linux.source, "prepared")) {
    throw new Error("Windows GPU DPC++ input paths do not match the selected run attempt");
  }
  canonicalDirectory(state.root, "DPC++/CUDA input root");
  canonicalDirectory(state.dpcppRoot, "DPC++ SDK root");
  canonicalDirectory(state.linux.source, "Linux SDK source root");
  const expectedSourcesFile = path.join(state.linux.prepared, "windows", "sources.json");
  if (input.windowsSourcesFile !== expectedSourcesFile) {throw new Error("Windows SYCL source receipt must be the same-run prepared Windows view");}
  const sourceReceiptPin = stableRegular(input.windowsSourcesFile, "Windows SYCL source receipt", 8 << 20);
  const sources = require("./prepare-native-sdk-sources").verifyPreparedSdkSources(input.windowsSourcesFile, "windows");
  if (fileHash(input.windowsSourcesFile) !== sourceReceiptPin.sha256 || sources.target !== "windows" ||
      sources.prepared?.dpcpp?.root !== path.join(state.linux.prepared, "windows", "dpcpp")) {
    throw new Error("Windows SYCL source receipt changed or has a different run-scoped root");
  }
  const sourceRoot = canonicalDirectory(sources.prepared.dpcpp.root, "Prepared Windows DPC++ source root");

  const manifestPin = stableRegular(state.manifest, "DPC++/CUDA input manifest", 8 << 20);
  if (manifestPin.sha256 !== input.dpcppManifestSha256) {throw new Error("DPC++/CUDA input manifest differs from its external pin");}
  const dpcppManifest = readBuildJson(state.manifest);
  // This is the maintained same-run audit; do not turn the copied SDK into a second receipt.
  const {auditDpcppCudaInputs} = require("./prepare-native-ci");
  auditDpcppCudaInputs(state, dpcppManifest, input.dpcppManifestSha256);
  if (fileHash(state.manifest) !== input.dpcppManifestSha256) {throw new Error("DPC++/CUDA manifest changed during admission");}

  const tools = linuxLlvmToolHandoffInputs(input.toolsRoot, input.toolsExpectation),
    toolFiles = ["bin/llvm-nm", "bin/llvm-readobj", "bin/llvm-objcopy"];
  for (const name of toolFiles) {
    const row = tools.files[name];
    if (!row || row.type !== "file" || !/^[a-f0-9]{64}$/.test(row.sha256 || "")) {
      throw new Error("Linux LLVM handoff is missing a selected Windows GPU tool: " + name);
    }
  }
  // Reject a foreign-run metadata archive before any compiler subprocess is invoked.
  const expectedWindowsSdkBase = "mom-native-windows-host-sdk-" + suffix,
    expectedWindowsSdkLeaf = "native-windows-host-sdk-" + suffix;
  if (path.basename(path.dirname(input.windowsSdkRoot)) !== expectedWindowsSdkBase ||
      path.basename(input.windowsSdkRoot) !== expectedWindowsSdkLeaf) {
    throw new Error("Windows SDK root does not match the selected run attempt");
  }
  const host = windowsHostSdkInputs(input.windowsSdkRoot, input.windowsSdkPins);
  /** @type {ReturnType<typeof windowsNativeGpuCudaImages> | null} */ let cudaImages = null;
  if (profile === "dpcpp") {
    if (!input.cudaImages) {throw new Error("DPC++ requires its same-run CUDA image handoff");}
    cudaImages = windowsNativeGpuCudaImages(input.cudaImages, suffix);
  }
  const built = composeWindowsDpcppGpuConfig(profile, {compilerRoot: input.compilerRoot, sourceRoot,
    dpcppRoot: state.dpcppRoot, cudaRoot: state.cudaRoot, dpcppManifest, host, tools, cudaImages});
  const admittedInputHashes = {
    windowsSdk: {...host.pins},
    windowsSources: {receipt: sourceReceiptPin.sha256, dpcppPreparedInventorySha256:
      createHash("sha256").update(JSON.stringify(sources.prepared.dpcpp.files)).digest("hex")},
    dpcppCuda: {manifestSha256: manifestPin.sha256, sourceTarSha256: dpcppManifest.source.tarSha256,
      sdkArchiveSha256: dpcppManifest.dpcpp.archiveSha256, selectedHeaders: built.selectedHeaders},
    linuxLlvmTools: {runAttempt: input.toolsExpectation.runAttempt, handoffSha256: input.toolsExpectation.handoffSha256,
      files: Object.fromEntries(toolFiles.map(name => [path.join(tools.toolRoot, path.basename(name)), tools.files[name].sha256]))},
    ...(cudaImages ? {cudaImages: cudaImages.hashes} : {}),
    compiler: built.compiler,
  };
  return {config: built.config, configBytes: built.configBytes, configSha256: built.configSha256, admittedInputHashes};
}

/** Shared local/CI construction AFTER the caller admits source, SDK and tool receipts.
 * No transport identity is invented here. Keep header-ABI and compiler checks shared;
 * CI's wrapper additionally binds its exact run-scoped paths and external handoff pins.
 * @param {string} profile
 * @param {{compilerRoot:string,sourceRoot:string,dpcppRoot:string,cudaRoot?:string,dpcppManifest:any,
 * host:ReturnType<typeof windowsHostSdkInputs>,tools:{llvmNm:string,toolRoot:string},
 * cudaImages:Pick<ReturnType<typeof windowsNativeGpuCudaImages>,"include">|null}} input */
function composeWindowsDpcppGpuConfig(profile, input) {
  if (!["oneapi", "dpcpp", "dpcpp-opencl"].includes(profile) || process.platform !== "linux" || process.arch !== "x64") {
    throw new Error("Windows DPC++ GPU descriptors require a maintained profile on Linux x64");
  }
  const {host, tools, dpcppManifest, cudaImages} = input,
    sourceRoot = canonicalDirectory(input.sourceRoot, "Prepared Windows DPC++ source root"),
    dpcppRoot = canonicalDirectory(input.dpcppRoot, "Pinned DPC++ SDK root"),
    compiler = canonicalDirectory(input.compilerRoot, profile === "oneapi" ? "Intel oneAPI compiler root" : "DPC++ compiler root"),
    hostIncludeRoots = validateWindowsSdkVfsRoots(host),
    syclInclude = canonicalDirectory(path.join(sourceRoot, "sycl/include"), "Prepared SYCL headers"),
    urInclude = canonicalDirectory(path.join(sourceRoot, "unified-runtime/include"), "Prepared Unified Runtime headers"),
    dpcppInclude = canonicalDirectory(path.join(dpcppRoot, "include"), "Pinned DPC++ SDK headers"),
    dpcppSyclInclude = canonicalDirectory(path.join(dpcppInclude, "sycl"), "Pinned DPC++ SYCL headers");
  if (profile !== "oneapi" && compiler !== dpcppRoot) {
    throw new Error("DPC++ OpenCL compiler root must be the same-run admitted DPC++ SDK");
  }
  if ((profile === "dpcpp") !== Boolean(cudaImages) || cudaImages && !input.cudaRoot) {
    throw new Error("Only the DPC++ profile requires admitted CUDA images and SDK inputs");
  }
  const compilerTools = profile === "oneapi" ?
    {clang: path.join(compiler, "bin/icpx"), clangC: path.join(compiler, "bin/icx"), llvmNm: tools.llvmNm,
      llvmReadobj: path.join(tools.toolRoot, "llvm-readobj"), llvmObjcopy: path.join(tools.toolRoot, "llvm-objcopy")} :
    {clang: path.join(compiler, "bin/clang++"), clangC: path.join(compiler, "bin/clang"), llvmNm: tools.llvmNm,
      llvmReadobj: path.join(tools.toolRoot, "llvm-readobj"), llvmObjcopy: path.join(tools.toolRoot, "llvm-objcopy")};
  const cleanCompilerEnvironment = () => {
    const environment = releaseBuildEnvironment(process.env);
    for (const key of Object.keys(environment)) {
      if (/^(?:PATH|CL|_CL_|LINK|_LINK_|INCLUDE|LIB|LIBPATH|CPATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH|LIBRARY_PATH|CFLAGS|CXXFLAGS|GIT_.*)$/i.test(key)) {
        delete environment[key];
      }
    }
    environment["PATH"] = [path.join(compiler, "bin"), "/usr/bin", "/bin"].join(path.delimiter);
    environment["LC_ALL"] = "C"; environment["LANG"] = "C";
    return environment;
  };
  /** @type {string} */ let expectedResourceInclude;
  /** @type {Record<string,string>} */ let resolvedCompilerInputs;
  if (profile === "oneapi") {
    const expected = canonicalDirectory(path.join(compiler, "lib/clang/23/include"), "Expected Intel 2026.2 resource headers"),
      draft = {version: /** @type {1} */ (1), profile, tools: compilerTools, includeRoots: [expected], overlay: host.sdk.overlay, compilerRoot: compiler},
      resolved = resolveOneapiCompiler(draft, cleanCompilerEnvironment());
    if (resolved.resourceInclude !== expected) {throw new Error("Intel compiler resource headers differ from the pinned 2026.2 layout");}
    expectedResourceInclude = resolved.resourceInclude; resolvedCompilerInputs = resolved.inputs;
  } else {
    const resolved = resolveDpcppCompiler(compiler, dpcppManifest.dpcpp, cleanCompilerEnvironment());
    expectedResourceInclude = resolved.resourceInclude; resolvedCompilerInputs = resolved.inputs;
  }
  const includePlan = windowsNativeGpuIncludePlan({resourceInclude: expectedResourceInclude,
    syclInclude, dpcppInclude, dpcppSyclInclude, urInclude, windowsSdkIncludeRoots: hostIncludeRoots});
  const includeRoots = includePlan.includeRoots;
  const draft = {version: /** @type {1} */ (1), profile, tools: compilerTools, includeRoots,
    overlay: host.sdk.overlay, ...(profile === "oneapi" ? {compilerRoot: compiler} : {}),
    ...(cudaImages ? {cuda: {sdk: input.cudaRoot, images: cudaImages.include}} : {})};
  const headerHashes = /** @type {Record<string, string>} */ ({});
  for (let index = 0; index < dpcppSdkHeaders.length; index++) {
    const relative = dpcppSdkHeaders[index], selected = includePlan.dpcppHeaders[index];
    if (!relative || !selected) {throw new Error("DPC++ SDK selected header plan is incomplete");}
    const file = containedFile(dpcppRoot, selected, "DPC++ SDK header " + relative);
    headerHashes[file] = stableRegular(file, "DPC++ SDK header " + relative, 16 << 20).sha256;
  }
  for (let index = 0; index < urHeaders.length; index++) {
    const relative = urHeaders[index], selected = includePlan.urHeaders[index];
    if (!relative || !selected) {throw new Error("Prepared Unified Runtime selected header plan is incomplete");}
    const file = containedFile(sourceRoot, selected, "Prepared Unified Runtime header " + relative);
    headerHashes[file] = stableRegular(file, "Prepared Unified Runtime header " + relative, 16 << 20).sha256;
    // The SDK include root wins the compiler's search order; its UR ABI must match our prepared source.
    const sdkHeader = path.join(dpcppInclude, "unified-runtime", relative);
    if (headerHashes[sdkHeader] !== headerHashes[file]) {
      throw new Error("Pinned DPC++ SDK Unified Runtime headers differ from the prepared source");
    }
  }
  const config = validateWindowsGpuConfig(draft), configBytes = Buffer.from(JSON.stringify(config, null, 2) + "\n", "utf8"),
    configSha256 = createHash("sha256").update(configBytes).digest("hex");
  return {config, configBytes, configSha256, selectedHeaders: headerHashes,
    compiler: {root: compiler, resourceInclude: expectedResourceInclude, files: resolvedCompilerInputs}};
}

/** Compose an ACPP device profile from independently admitted same-run inputs.
 * CUDA-only DPC++ state is audited for its retained CUDA payload; its compiler is never queried.
 * @param {"acpp-cuda"|"acpp-hip"} profile
 * @param {any} input */
function windowsNativeAcppGpuConfig(profile, input) {
  if (!["acpp-cuda", "acpp-hip"].includes(profile)) {throw new Error("Unsupported Windows ACPP GPU config profile");}
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Windows ACPP config is composed on Linux x64 only");}
  const commonKeys = ["windowsSdkRoot", "windowsSdkPins", "toolsRoot", "toolsExpectation", "frontendRoot", "frontendExpectation", "acpp"],
    cudaKeys = ["dpcppState", "dpcppManifestSha256", "cudaImages"],
    inputKeys = profile === "acpp-cuda" ? [...commonKeys, ...cudaKeys] : [...commonKeys, "hipHeaders"];
  if (!hasExactKeys(input, inputKeys) || !absolutePath(input.windowsSdkRoot) || !absolutePath(input.toolsRoot) ||
      !absolutePath(input.frontendRoot) || !hasExactKeys(input.toolsExpectation, ["runAttempt", "handoffSha256"]) ||
      profile === "acpp-hip" && !hasExactKeys(input.hipHeaders, ["root", "handoffSha256"]) ||
      !hasExactKeys(input.frontendExpectation, ["runAttempt", "handoffSha256"]) ||
      input.toolsExpectation.runAttempt !== input.frontendExpectation.runAttempt ||
      !/^[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(input.toolsExpectation.runAttempt || "") ||
      !/^[a-f0-9]{64}$/.test(input.toolsExpectation.handoffSha256 || "") ||
      !/^[a-f0-9]{64}$/.test(input.frontendExpectation.handoffSha256 || "")) {
    throw new Error("Windows ACPP config requires exact same-run SDK, tool and frontend inputs");
  }
  const suffix = input.toolsExpectation.runAttempt,
    expectedSdkBase = "mom-native-windows-host-sdk-" + suffix, expectedSdkLeaf = "native-windows-host-sdk-" + suffix;
  if (path.basename(path.dirname(input.windowsSdkRoot)) !== expectedSdkBase || path.basename(input.windowsSdkRoot) !== expectedSdkLeaf) {
    throw new Error("Windows SDK root does not match the selected run attempt");
  }
  const host = windowsHostSdkReader.windowsHostSdkInputs(input.windowsSdkRoot, input.windowsSdkPins),
    tools = linuxLlvmToolReader.linuxLlvmToolHandoffInputs(input.toolsRoot, input.toolsExpectation),
    frontend = linuxLlvmToolReader.linuxLlvmAcppFrontendInputs(input.frontendRoot, input.frontendExpectation),
    acpp = windowsNativeAcppInputs(input.acpp, suffix), resolvedFrontend = resolveAcppFrontend(input.frontendRoot, frontend),
    toolNames = ["bin/llvm-nm", "bin/llvm-readobj", "bin/llvm-objcopy"];
  if (tools.llvmNm !== path.join(tools.toolRoot, "llvm-nm") ||
      toolNames.some(name => !tools.files[name] || tools.files[name].type !== "file" || !/^[a-f0-9]{64}$/.test(tools.files[name].sha256 || ""))) {
    throw new Error("Linux LLVM handoff is missing a selected Windows GPU tool");
  }
  // Native HIP interop consumes upstream HIP declarations as well as SYCL headers. CUDA
  // must not inherit this separate platform's include tree or dependency chain.
  const hipHeaders = profile === "acpp-hip" ? hipHeaderReader.rocmHipHeaderInputs(input.hipHeaders?.root,
    {runAttempt: suffix, handoffSha256: input.hipHeaders?.handoffSha256}) : null;
  /** @type {ReturnType<typeof windowsNativeGpuCudaImages>|null} */ let cudaImages = null, dpcpp = null;
  if (profile === "acpp-cuda") {
    dpcpp = admitAcppCudaInputs(input.dpcppState, input.dpcppManifestSha256, suffix);
    cudaImages = windowsNativeGpuCudaImages(input.cudaImages, suffix);
  }
  const built = composeWindowsAcppGpuConfig(profile, {host, tools, acpp, frontend: resolvedFrontend,
    ...(hipHeaders ? {hipInclude: hipHeaders.include} : {}),
    cuda: dpcpp && cudaImages ? {sdk: dpcpp.state.cudaRoot, images: cudaImages.include} : null});
  const acppFiles = Object.fromEntries(acpp.generatedFiles.map(file => [path.join(acpp.generatedInclude, ...file.path.split("/")), file.sha256]));
  const admittedInputHashes = {
    windowsSdk: {...host.pins},
    acpp: {...acpp.hashes, generatedFiles: acppFiles},
    ...(hipHeaders ? {hipHeaders: {handoffSha256: hipHeaders.handoffSha256,
      manifestSha256: hipHeaders.manifestSha256, files: hipHeaders.files}} : {}),
    linuxLlvmTools: {runAttempt: suffix, handoffSha256: input.toolsExpectation.handoffSha256,
      files: Object.fromEntries(toolNames.map(name => [path.join(tools.toolRoot, path.basename(name)), tools.files[name].sha256]))},
    linuxLlvmAcppFrontend: {runAttempt: suffix, handoffSha256: input.frontendExpectation.handoffSha256,
      version: resolvedFrontend.version, resourceInclude: resolvedFrontend.resourceInclude,
      files: Object.fromEntries(Object.entries(frontend.files).filter(([, row]) => row.type === "file")
        .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0).map(([name, row]) => [path.join(input.frontendRoot, name), row.sha256]))},
    compiler: {version: resolvedFrontend.version, resourceDir: resolvedFrontend.resourceDir,
      queries: resolvedFrontend.queryHashes},
    ...(dpcpp && cudaImages ? {dpcppCuda: {manifestSha256: dpcpp.manifestPin.sha256,
      sourceTarSha256: dpcpp.manifest.source.tarSha256, sdkArchiveSha256: dpcpp.manifest.dpcpp.archiveSha256}, cudaImages: cudaImages.hashes} : {}),
  };
  return {config: built.config, configBytes: built.configBytes, configSha256: built.configSha256, admittedInputHashes};
}

/** Shared descriptor construction from admitted local/CI ACPP, compiler and SDK inputs.
 * Source/transport admission belongs to the caller; backend separation and header
 * selection stay here so a local build cannot silently select a different recipe.
 * @param {string} profile
 * @param {{host:ReturnType<typeof windowsHostSdkInputs>,tools:{llvmNm:string,toolRoot:string},
 * acpp:{source:string,generatedInclude:string,plugin:string},frontend:{clang:string,clangxx:string,resourceInclude:string},
 * hipInclude?:string,cuda:{sdk:string,images:string}|null}} input */
function composeWindowsAcppGpuConfig(profile, input) {
  if (!["acpp-cuda", "acpp-hip"].includes(profile) || process.platform !== "linux" || process.arch !== "x64") {
    throw new Error("Windows ACPP GPU descriptors require a maintained profile on Linux x64");
  }
  const {host, tools, acpp, frontend, hipInclude, cuda} = input;
  if (profile === "acpp-cuda" ? !cuda || Boolean(hipInclude) : Boolean(cuda) || !hipInclude) {
    throw new Error("Windows ACPP descriptor requires only its selected CUDA or HIP inputs");
  }
  const includeRoots = [frontend.resourceInclude, ...(hipInclude ? [hipInclude] : []),
    path.join(acpp.source, "include"), acpp.generatedInclude, ...validateWindowsSdkVfsRoots(host)];
  for (const [directory, label] of /** @type {Array<[string,string]>} */ ([[frontend.resourceInclude, "LLVM 21 resource headers"],
    [path.join(acpp.source, "include"), "Prepared ACPP source headers"], [acpp.generatedInclude, "Generated ACPP headers"],
    ...(hipInclude ? [[hipInclude, "Pinned HIP headers"]] : [])])) {
    canonicalDirectory(directory, label);
  }
  if (new Set(includeRoots).size !== includeRoots.length) {throw new Error("Windows ACPP config has duplicate include roots");}
  const config = validateWindowsGpuConfig({version: 1, profile,
      tools: {clang: frontend.clangxx, clangC: frontend.clang, llvmNm: tools.llvmNm,
        llvmReadobj: path.join(tools.toolRoot, "llvm-readobj"), llvmObjcopy: path.join(tools.toolRoot, "llvm-objcopy")},
      includeRoots, overlay: host.sdk.overlay,
      acpp: {source: acpp.source, generatedInclude: acpp.generatedInclude, plugin: acpp.plugin}, ...(cuda ? {cuda} : {})}),
    configBytes = Buffer.from(JSON.stringify(config, null, 2) + "\n", "utf8"),
    configSha256 = createHash("sha256").update(configBytes).digest("hex");
  return {config, configBytes, configSha256};
}

/** Verify the pinned DPC++/CUDA input handoff; ACPP never uses that DPC++ compiler.
 * @param {any} state @param {string} expectedManifestSha256 @param {string} suffix */
function admitAcppCudaInputs(state, expectedManifestSha256, suffix) {
  const name = "mom-native-dpcpp-cuda-inputs-" + suffix,
    sourceName = "mom-native-linux-source-prerequisites-" + suffix;
  if (!/^[a-f0-9]{64}$/.test(expectedManifestSha256 || "") || !state || typeof state !== "object" ||
      state.name !== name || state.root !== path.join("/tmp", name) || state.dpcppRoot !== path.join(state.root, "dpcpp") ||
      state.cudaRoot !== path.join(state.root, "cuda") || state.manifest !== path.join(state.root, "inputs.json") ||
      state.linux?.sourceName !== sourceName || state.linux.source !== path.join("/tmp", sourceName) ||
      state.linux.prepared !== path.join(state.linux.source, "prepared")) {
    throw new Error("ACPP CUDA inputs do not match the selected DPC++/CUDA run attempt");
  }
  canonicalDirectory(state.root, "DPC++/CUDA input root"); canonicalDirectory(state.dpcppRoot, "DPC++ SDK root");
  canonicalDirectory(state.cudaRoot, "CUDA SDK root"); canonicalDirectory(state.linux.source, "Linux SDK source root");
  const manifestPin = stableRegular(state.manifest, "DPC++/CUDA input manifest", 8 << 20);
  if (manifestPin.sha256 !== expectedManifestSha256) {throw new Error("DPC++/CUDA manifest differs from its external pin");}
  const manifest = readBuildJson(state.manifest), {auditDpcppCudaInputs} = require("./prepare-native-ci");
  // Preserve acquisition provenance across unrelated CI edits, retaining all payload checks.
  auditDpcppCudaInputs(state, manifest, expectedManifestSha256);
  if (fileHash(state.manifest) !== manifestPin.sha256) {throw new Error("DPC++/CUDA manifest changed during admission");}
  return {state, manifest, manifestPin};
}

/** Query the admitted LLVM 21 frontend without allowing ambient include/toolchain variables.
 * @param {string} root @param {Pick<ReturnType<typeof linuxLlvmToolReader.linuxLlvmAcppFrontendInputs>,"clang"|"clangxx"|"resourceInclude"|"files">} frontend
 * @param {(command:string,args:string[],options:import("node:child_process").SpawnSyncOptionsWithStringEncoding)=>import("node:child_process").SpawnSyncReturns<string>} [spawn] */
function resolveAcppFrontend(root, frontend, spawn = childProcess.spawnSync) {
  canonicalDirectory(root, "LLVM 21 ACPP frontend root");
  const clang = path.join(root, "bin/clang"), clangxx = path.join(root, "bin/clang++"), resourceDir = path.join(root, "lib/clang/21"),
    resourceInclude = path.join(resourceDir, "include");
  if (frontend.clang !== clang || frontend.clangxx !== clangxx || frontend.resourceInclude !== resourceInclude ||
      fs.lstatSync(clang).isSymbolicLink() !== true || fs.readlinkSync(clang) !== "clang-21" ||
      fs.lstatSync(clangxx).isSymbolicLink() !== true || fs.readlinkSync(clangxx) !== "clang") {
    throw new Error("LLVM 21 ACPP frontend handoff paths or relative aliases differ");
  }
  const selected = [clangxx, clang];
  for (const file of selected) {
    const stat = fs.statSync(file), real = fs.realpathSync(file), relative = path.relative(root, real);
    if (!stat.isFile() || !(stat.mode & 0o111) || path.isAbsolute(relative) || relative === ".." || relative.startsWith(".." + path.sep)) {
      throw new Error("LLVM 21 ACPP compiler is missing, non-executable, or outside its handoff");
    }
  }
  canonicalDirectory(resourceDir, "LLVM 21 Clang resource directory"); canonicalDirectory(resourceInclude, "LLVM 21 Clang resource headers");
  const environment = nativeSdkEnvironment(process.env);
  for (const key of Object.keys(environment)) {
    if (/^(?:PATH|CL|_CL_|LINK|_LINK_|INCLUDE|LIB|LIBPATH|CPATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH|LIBRARY_PATH|CFLAGS|CXXFLAGS|GIT_.*|LD_.*|DYLD_.*)$/i.test(key)) {
      delete environment[key];
    }
  }
  environment["PATH"] = [path.join(root, "bin"), "/usr/bin", "/bin"].join(path.delimiter);
  environment["LC_ALL"] = "C"; environment["LANG"] = "C";
  /** @param {string} executable @param {string[]} args @param {string} label */
  const query = (executable, args, label) => {
    const result = spawn(executable, args, {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20, env: environment});
    if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string" || result.stdout.length > (1 << 20)) {
      throw new Error("LLVM 21 ACPP " + label + " query failed", {cause: result.error});
    }
    return result.stdout.trim();
  };
  const versions = selected.map((file, index) => query(file, ["--version"], index ? "clang" : "clang++"));
  if (versions.some(version => !/^clang version 21\.1\.8(?:\s|$)/m.test(version)) || versions[0] !== versions[1]) {
    throw new Error("ACPP frontend must be the same pinned LLVM 21.1.8 Clang for C and C++");
  }
  const resourceDirs = selected.map((file, index) => query(file, ["--print-resource-dir"], index ? "clang resource" : "clang++ resource"));
  if (resourceDirs.some(directory => directory !== resourceDir)) {throw new Error("ACPP Clang resource directory differs from its exact LLVM 21 handoff");}
  const queryHashes = Object.fromEntries(selected.flatMap((file, index) => [
    [file + " --version", createHash("sha256").update(versions[index] || "").digest("hex")],
    [file + " --print-resource-dir", createHash("sha256").update(resourceDirs[index] || "").digest("hex")],
  ]));
  for (const [name, row] of Object.entries(frontend.files)) {
    const file = path.join(root, ...name.split("/"));
    if (row.type === "file" && stableRegular(file, "LLVM frontend input " + name, row.bytes).sha256 !== row.sha256) {
      throw new Error("LLVM frontend input changed during compiler queries: " + name);
    }
    if (row.type === "symlink" && (!fs.lstatSync(file).isSymbolicLink() || fs.readlinkSync(file) !== row.target)) {
      throw new Error("LLVM frontend alias changed during compiler queries: " + name);
    }
  }
  return {clang, clangxx, resourceDir, resourceInclude, version: "21.1.8", queryHashes};
}

/** Admit the externally pinned same-run image tar and its restored, source-checked handoff.
 * The maintained job restores the tar with its full root/member checks before this composer runs;
 * the archive pin alone is not treated as proof that the extracted tree is unchanged.
 * @param {{archive:string,bundleSha256:string,root:string}} input @param {string} suffix */
function windowsNativeGpuCudaImages(input, suffix) {
  const run = suffix, runnerTemp = process.env["RUNNER_TEMP"] || "",
    incoming = path.join(runnerTemp, "incoming"), archiveName = "native-linux-cuda-images-" + run + ".tar",
    handoffName = "mom-native-cuda-images-handoff-" + run;
  if (!hasExactKeys(input, ["archive", "bundleSha256", "root"]) ||
      !/^[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(run) ||
      !/^[a-f0-9]{64}$/.test(input.bundleSha256 || "") ||
      !path.isAbsolute(runnerTemp) || fs.realpathSync(runnerTemp) !== runnerTemp ||
      fs.realpathSync(incoming) !== incoming || input.archive !== path.join(incoming, archiveName) ||
      input.root !== path.join("/tmp", handoffName)) {
    throw new Error("Windows CUDA image handoff differs from the exact same-run artifact contract");
  }
  canonicalDirectory(input.root, "CUDA image handoff root");
  const archive = stableRegular(input.archive, "CUDA image archive", 640 * 1024 * 1024);
  if (archive.sha256 !== input.bundleSha256) {throw new Error("CUDA image archive differs from its external same-run pin");}

  const inspectionDeadline = Date.now() + maxWindowsAcppTarInspectionMs,
    remainingInspectionMs = () => {
      const remaining = inspectionDeadline - Date.now();
      if (remaining < 1000) {throw new Error("CUDA image tar inspection exceeded its shared five-minute budget");}
      return remaining;
    }, ci = require("./prepare-native-ci"), members = ci.listState(input.archive, [handoffName], remainingInspectionMs());
  if (members.length > 200000 || new Set(members.map(name => name.endsWith("/") ? name.slice(0, -1) : name)).size !== members.length) {
    throw new Error("CUDA image archive has duplicate or excessive members");
  }
  const selectedMembers = /** @type {[string,string,string]} */ ([handoffName + "/cuda-images.inc", handoffName + "/cuda-images.inc.json",
    handoffName + "/result.json"]);
  if (selectedMembers.some(name => !members.includes(name)) || ci.tarExpandedBytes(input.archive, members.length, remainingInspectionMs()) > 1024 ** 3) {
    throw new Error("CUDA image archive omits a selected result or exceeds its expanded-size bound");
  }

  const include = path.join(input.root, "cuda-images.inc"), sidecar = include + ".json",
    resultFile = path.join(input.root, "result.json"), includePin = stableRegular(include, "CUDA image include", 512 * 1024 * 1024),
    sidecarPin = stableRegular(sidecar, "CUDA image sidecar", 8 << 20), resultPin = stableRegular(resultFile, "CUDA image result", 1 << 20),
    admitted = require("./native-cuda-images").cudaImageInputs(include, path.resolve(__dirname, "..")),
    result = readBuildJson(resultFile), expectedOrigin = path.join("/tmp", "mom-native-cuda-images-" + run, "bundle", "cuda-images.inc"),
    versions = Array.isArray(result.compilerVersions) ? [...result.compilerVersions].sort() : [],
    manifestImages = admitted.manifest["images"], manifestBytes = admitted.manifest["resourceBytes"];
  if (!hasExactKeys(result, ["format", "include", "sidecar", "compilerVersions", "plannedImages", "embeddedBytes", "sourceSha256"]) ||
      result.format !== "mom-cuda-image-build-v1" || result.include !== expectedOrigin || result.sidecar !== expectedOrigin + ".json" ||
      versions.join(",") !== "12.6,13.3" || !Array.isArray(manifestImages) || result.plannedImages !== manifestImages.length ||
      typeof manifestBytes !== "number" || result.embeddedBytes !== manifestBytes ||
      result.sourceSha256 !== includePin.sha256 || result.sourceSha256 !== admitted.manifest["sourceSha256"] ||
      fileHash(input.archive) !== archive.sha256 || fileHash(include) !== includePin.sha256 ||
      fileHash(sidecar) !== sidecarPin.sha256 || fileHash(resultFile) !== resultPin.sha256) {
    throw new Error("CUDA image result, current-source sidecar, and same-run include do not agree");
  }
  const tarPins = [
    tarMemberPin(input.archive, selectedMembers[0], 512 * 1024 * 1024, remainingInspectionMs()),
    tarMemberPin(input.archive, selectedMembers[1], 8 << 20, remainingInspectionMs()),
    tarMemberPin(input.archive, selectedMembers[2], 1 << 20, remainingInspectionMs()),
  ];
  if (!isDeepStrictEqual(tarPins, [includePin, sidecarPin, resultPin])) {
    throw new Error("Restored CUDA image files differ from the original externally pinned tar members");
  }
  if (archive.sha256 !== stableRegular(input.archive, "CUDA image archive", 640 * 1024 * 1024).sha256 ||
      includePin.sha256 !== stableRegular(include, "CUDA image include", 512 * 1024 * 1024).sha256 ||
      sidecarPin.sha256 !== stableRegular(sidecar, "CUDA image sidecar", 8 << 20).sha256 ||
      resultPin.sha256 !== stableRegular(resultFile, "CUDA image result", 1 << 20).sha256) {
    throw new Error("CUDA image tar or restored files changed during admission");
  }
  return {include, hashes: {bundleSha256: archive.sha256, includeSha256: includePin.sha256,
    sidecarSha256: sidecarPin.sha256, resultSha256: resultPin.sha256}};
}

/** @param {Parameters<typeof windowsNativeGpuConfig>[1]} input */
function windowsNativeOneapiGpuConfig(input) {return windowsNativeGpuConfig("oneapi", input);}

module.exports = {windowsNativeGpuConfig, composeWindowsDpcppGpuConfig, composeWindowsAcppGpuConfig, windowsNativeAcppInputs, windowsNativeAcppGpuConfig, windowsNativeOneapiGpuConfig, windowsNativeGpuIncludePlan,
  generatedIncludeInventory,
  windowsNativeOneapiIncludePlan, resolveDpcppCompiler, resolveAcppFrontend, windowsNativeGpuCudaImages};
