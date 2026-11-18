"use strict";

// Local WSL coordinator for the maintained Windows-target LLVM 21 builder.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), nativeCi = require("./native-ci-windows-llvm"),
  linuxLlvm = require("./build-native-llvm-sdk"), windowsLlvm = require("./build-native-windows-llvm-sdk"),
  sourceArchives = require("./fetch-native-sdk-sources");
const {fileHash, readBuildJson, readBuildText} = require("./build-source-inputs"), {archiveName, archiveSha256} = require("./native-llvm-sdk");
const repo = path.resolve(__dirname, ".."), jobs = 2, localBuildIdPattern = /^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/;

/** @param {unknown} value @param {string[]} keys @param {string} label */
function exactObject(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).sort().join(",") !== keys.slice().sort().join(",")) {
    throw new Error(label + " has unknown or missing fields");
  }
  return /** @type {Record<string, any>} */ (value);
}

/** @param {string} directory @param {string} label @param {boolean} [allowMissing] */
function realDirectory(directory, label, allowMissing = false) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat && allowMissing) {return false;}
  if (!stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
    throw new Error(label + " is not a canonical directory: " + directory);
  }
  return true;
}

/** @param {string} file @param {string} label @param {number} [maximum] */
function regularFile(file, label, maximum = 8 << 20) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!path.isAbsolute(file) || path.resolve(file) !== file || !stat?.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 ||
      stat.size < 1 || stat.size > maximum || fs.realpathSync(file) !== file) {
    throw new Error(label + " is not a bounded canonical regular file: " + file);
  }
  return stat;
}

/** Read only the small expected inventory; arbitrary workspace contents are never enumerated unboundedly.
 * @param {string} directory @param {string[]} expected @param {string} label */
function exactEntries(directory, expected, label) {
  const found = [], handle = fs.opendirSync(directory);
  try {
    for (let entry; (entry = handle.readSync()) !== null;) {
      found.push(entry.name);
      if (found.length > expected.length) {throw new Error(label + " contains unknown entries; preserve it for inspection");}
    }
  } finally {handle.closeSync();}
  if (!isDeepStrictEqual(found.sort(), expected.slice().sort())) {throw new Error(label + " is incomplete or contains unknown entries; preserve it for inspection");}
}

/** @param {string} workspace */
function localWindowsLlvmSdkPaths(workspace) {
  if (typeof workspace !== "string" || !path.isAbsolute(workspace) || path.resolve(workspace) !== workspace || /[\r\n\0]/.test(workspace)) {
    throw new Error("Local Windows LLVM workspace must be canonical and absolute");
  }
  const root = path.join(workspace, "windows-llvm-sdk");
  return {root, descriptor: path.join(root, "descriptor.json"), provenance: path.join(root, "provenance.json"),
    output: path.join(root, "output"), archive: path.join(local.supportPaths(workspace).buildInputs, archiveName)};
}

/** @param {string} root @param {{resultSha256:string,descriptorSha256:string,filesSha256:string,overlaySha256:string}} pins
 * @param {string} workspace */
function validateWindowsHostSdk(root, pins, workspace) {
  exactObject(pins, ["resultSha256", "descriptorSha256", "filesSha256", "overlaySha256"], "Local Windows host-SDK pins");
  if (root !== path.join(workspace, "windows-host-sdk", "sdk") || Object.values(pins).some(hash => !/^[a-f0-9]{64}$/.test(hash || ""))) {
    throw new Error("Local Windows LLVM needs the admitted host-SDK root and exact receipt pins");
  }
  realDirectory(root, "Local Windows host SDK");
}

/** @param {{root:string,localBuildId:string,handoffSha256:string}} hermes */
function validateHermes(hermes) {
  exactObject(hermes, ["root", "localBuildId", "handoffSha256"], "Local Windows Hermes input");
  if (typeof hermes.root !== "string" || !path.isAbsolute(hermes.root) || path.resolve(hermes.root) !== hermes.root || /[\r\n\0]/.test(hermes.root) ||
      !localBuildIdPattern.test(hermes.localBuildId || "") || !/^[a-f0-9]{64}$/.test(hermes.handoffSha256 || "")) {
    throw new Error("Local Windows LLVM needs the exact local Hermes handoff identity and hash");
  }
  realDirectory(hermes.root, "Local Windows Hermes handoff");
}

/** Select the actual Intel driver/resource and only LLVM utilities from the admitted Linux SDK.
 * local.localTools has already verified the Intel 2026.2 icx/icpx identities. @param {ReturnType<typeof local.localTools>} tools @param {any} admission */
function localWindowsLlvmDescriptorTools(tools, admission) {
  if (!tools || Object.keys(tools).sort().join(",") !== "cc,cmake,cxx,git,ninja,systemdRun" ||
      path.basename(tools.cc) !== "icx" || path.basename(tools.cxx) !== "icpx" ||
      path.dirname(tools.cc) !== path.dirname(tools.cxx) || path.basename(path.dirname(path.dirname(tools.cc))) !== "2026.2") {
    throw new Error("Local Windows LLVM requires the selected Intel oneAPI 2026.2 toolchain");
  }
  const compilerRoot = path.dirname(path.dirname(tools.cc)), compilerBin = path.dirname(tools.cc), clangCl = path.join(compilerBin, "icx-cl"),
    clangResourceInclude = path.join(compilerRoot, "lib", "clang", "23", "include"), clangClSha256 = fileHash(clangCl);
  const sdk = admission?.sdk, sdkRoot = sdk?.path, snapshot = sdk?.snapshot?.files;
  if (typeof sdkRoot !== "string" || !snapshot || typeof snapshot !== "object" || Array.isArray(snapshot)) {
    throw new Error("Local Windows LLVM needs the public Linux LLVM admission snapshot");
  }
  const toolRoot = path.join(sdkRoot, "bin"), nm = path.join(toolRoot, "llvm-nm"), lldLink = path.join(toolRoot, "lld-link"),
    nmRow = snapshot["bin/llvm-nm"], lldRow = snapshot["bin/lld"], aliasRow = snapshot["bin/lld-link"];
  if (!nmRow || nmRow.type !== "file" || !/^[a-f0-9]{64}$/.test(nmRow.sha256 || "") ||
      !lldRow || lldRow.type !== "file" || !/^[a-f0-9]{64}$/.test(lldRow.sha256 || "")) {
    throw new Error("Selected Linux LLVM utilities differ from the admitted SDK snapshot");
  }
  const aliasStat = fs.lstatSync(lldLink, {throwIfNoEntry: false});
  if (!aliasRow || aliasRow.type !== "symlink" || aliasRow.target !== "lld" || !aliasStat?.isSymbolicLink() ||
      fs.readlinkSync(lldLink) !== "lld" || fs.realpathSync(lldLink) !== path.join(toolRoot, "lld")) {
    throw new Error("Selected Linux LLVM lld-link alias differs from its admitted SDK snapshot");
  }
  return {clangCl: {path: clangCl, sha256: clangClSha256},
    lldLink: {path: lldLink, sha256: lldRow.sha256, linkTarget: "lld"},
    llvmNm: {path: nm, sha256: nmRow.sha256}, clangResourceInclude};
}

/** @param {ReturnType<typeof localWindowsLlvmSdkPaths>} paths @param {boolean} allowMissing */
function checkManagedRoot(paths, allowMissing) {
  if (!realDirectory(paths.root, "Local Windows LLVM managed output", allowMissing)) {return false;}
  exactEntries(paths.root, ["descriptor.json", "output", "provenance.json"], "Local Windows LLVM managed output");
  regularFile(paths.descriptor, "Local Windows LLVM descriptor"); regularFile(paths.provenance, "Local Windows LLVM provenance");
  realDirectory(paths.output, "Local Windows LLVM SDK output");
  return true;
}

/** Reject extra or interrupted direct artifacts without traversing the large source/cache trees.
 * The maintained public admission validates the contents of those trees. @param {string} output @param {any} plan @param {string} runId */
function verifyLocalWindowsLlvmOutput(output, plan, runId) {
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(runId || "")) {
    throw new Error("Local Windows LLVM output has no valid original builder identity");
  }
  const expected = [path.basename(plan.archiveCopy), path.basename(plan.source), path.basename(plan.build), path.basename(plan.overlay),
    path.basename(plan.sourceOptionsFile), path.basename(plan.ruleOverride), path.basename(plan.toolchainFile), "plan.json", "result.json",
    "producer-snapshot.js", "extract.log", "configure.log", "build-state.json", "result-current.json", "admitted.json", "admitted-objects.json",
    ...windowsLlvm.buildPhases.flatMap(phase => ["build-" + phase.name + "-" + runId + ".log", "phase-" + phase.name + "-" + runId + ".json"])].sort();
  if (new Set(expected).size !== expected.length) {throw new Error("Maintained Windows LLVM output inventory has duplicate names");}
  exactEntries(output, expected, "Local Windows LLVM SDK output");
  for (const name of [path.basename(plan.source), path.basename(plan.build)]) {
    realDirectory(path.join(output, name), "Local Windows LLVM " + name);
  }
  for (const name of expected.filter(name => ![path.basename(plan.source), path.basename(plan.build)].includes(name))) {
    const file = path.join(output, name), stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== file) {
      throw new Error("Local Windows LLVM output artifact is redirected or special; preserve it: " + name);
    }
  }
}

/** @param {unknown} value @returns {Buffer} */
function jsonBytes(value) {
  const bytes = Buffer.from(JSON.stringify(value, null, 2) + "\n");
  if (bytes.length < 2 || bytes.length > (8 << 20)) {throw new Error("Local Windows LLVM descriptor or provenance exceeds its bound");}
  return bytes;
}

/** @param {string} file @param {Buffer} expected @param {string} label */
function verifyStableJson(file, expected, label) {
  const before = regularFile(file, label), sha256 = fileHash(file), bytes = Buffer.from(readBuildText(file, 8 << 20)), after = regularFile(file, label);
  if (before.dev !== after.dev || before.ino !== after.ino || before.nlink !== after.nlink || before.size !== after.size ||
      before.mtimeMs !== after.mtimeMs || before.ctimeMs !== after.ctimeMs || sha256 !== fileHash(file) || !bytes.equals(expected)) {
    throw new Error(label + " is stale or changed; preserve the output for inspection");
  }
  return sha256;
}

/** @param {{workspace:string,windowsHostSdk:{root:string,pins:{resultSha256:string,descriptorSha256:string,filesSha256:string,overlaySha256:string}},
 * hermes:{root:string,localBuildId:string,handoffSha256:string},tools:ReturnType<typeof local.localTools>}} input
 * @param {{verifyScope?:()=>unknown,admitWorkspace?:typeof local.admitWorkspace,resolveTools?:typeof local.localTools,
 * verifyArchives?:typeof sourceArchives.verifyBuildArchives,verifyLinuxLlvm?:typeof linuxLlvm.verifyLinuxLlvmAdmission,
 * composeDescriptor?:typeof nativeCi.composeLocalWindowsLlvmSdkDescriptor,
 * configure?:typeof windowsLlvm.configureWindowsLlvmSdk,buildPhase?:typeof windowsLlvm.buildWindowsLlvmSdk,
 * audit?:typeof windowsLlvm.auditWindowsLlvmAdmission,verify?:typeof windowsLlvm.verifyWindowsLlvmAdmission,
 * verifyOutputInventory?:(output:string,result:any)=>unknown,
 * environment?:NodeJS.ProcessEnv}} [dependencies] */
function buildLocalNativeWindowsLlvmSdk(input, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local Windows LLVM SDK production requires Linux x64 (WSL2)");}
  const verifyScope = dependencies.verifyScope || local.verifyCurrentMemoryScope;
  verifyScope();
  if (!input || Object.keys(input).sort().join(",") !== "hermes,tools,windowsHostSdk,workspace" || typeof input.workspace !== "string") {
    throw new Error("Local Windows LLVM SDK needs one workspace, admitted Windows SDK, Hermes handoff and selected tools");
  }
  const workspace = (dependencies.admitWorkspace || local.admitWorkspace)(input.workspace, repo), paths = localWindowsLlvmSdkPaths(workspace),
    sdk = /** @type {typeof input.windowsHostSdk} */ (exactObject(input.windowsHostSdk, ["root", "pins"], "Local Windows host-SDK input")),
    hermes = /** @type {typeof input.hermes} */ (exactObject(input.hermes, ["root", "localBuildId", "handoffSha256"], "Local Windows Hermes input"));
  validateWindowsHostSdk(sdk.root, sdk.pins, workspace); validateHermes(hermes);
  const tools = /** @type {ReturnType<typeof local.localTools>} */ (exactObject(input.tools,
      ["git", "cmake", "ninja", "systemdRun", "cc", "cxx"], "Selected local tools")),
    environment = {...process.env, ...(dependencies.environment || {}), MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: String(jobs)},
    resolveTools = dependencies.resolveTools || local.localTools, selectedTools = resolveTools(environment);
  if (!isDeepStrictEqual(tools, selectedTools)) {throw new Error("Local Windows LLVM tools differ from the current pinned selection");}
  const verifyArchives = dependencies.verifyArchives || sourceArchives.verifyBuildArchives,
    verifyLinux = dependencies.verifyLinuxLlvm || linuxLlvm.verifyLinuxLlvmAdmission,
    compose = dependencies.composeDescriptor || nativeCi.composeLocalWindowsLlvmSdkDescriptor,
    configure = dependencies.configure || windowsLlvm.configureWindowsLlvmSdk,
    buildPhase = dependencies.buildPhase || windowsLlvm.buildWindowsLlvmSdk,
    audit = dependencies.audit || windowsLlvm.auditWindowsLlvmAdmission,
    verify = dependencies.verify || windowsLlvm.verifyWindowsLlvmAdmission,
    verifyOutputInventory = dependencies.verifyOutputInventory || ((output, result) => verifyLocalWindowsLlvmOutput(output,
      windowsLlvm.windowsLlvmSdkPlan(readBuildJson(paths.descriptor), paths.archive, output, {cmake: tools.cmake, ninja: tools.ninja}, jobs), result.runId));

  /** Re-admit original archives, Linux LLVM, compiler tools and the same local descriptor inputs. */
  const admitInputs = () => {
    const archiveReceipt = verifyArchives(local.supportPaths(workspace).buildInputs),
      archive = archiveReceipt?.archives?.["llvm"], linux = verifyLinux(local.supportPaths(workspace).llvmSdk),
      descriptorTools = localWindowsLlvmDescriptorTools(tools, linux),
      composed = compose({windowsSdkRoot: sdk.root, windowsSdkPins: sdk.pins, hermesRoot: hermes.root,
        hermesExpectation: {kind: "hermes", localBuildId: hermes.localBuildId, handoffSha256: hermes.handoffSha256}, tools: descriptorTools});
    if (archiveReceipt?.target !== "build" || archive?.path !== paths.archive || archive.sha256 !== archiveSha256 ||
        linux?.role !== "linux-llvm21-source-sdk" || linux.stage !== "audited" || linux.buildOnly !== true || linux.compiled !== true ||
        linux.executed !== false || linux.publicationQualified !== false || linux.sdk?.path !== path.join(local.supportPaths(workspace).llvmSdk, "sdk") ||
        !composed?.descriptor || typeof composed.descriptor !== "object" || Array.isArray(composed.descriptor) ||
        !composed?.provenance || typeof composed.provenance !== "object" || Array.isArray(composed.provenance)) {
      throw new Error("Local Windows LLVM original archive, Linux SDK or descriptor admission is incomplete");
    }
    return {archiveReceipt, linux, descriptorTools, composed, descriptorBytes: jsonBytes(composed.descriptor), provenanceBytes: jsonBytes(composed.provenance)};
  };

  const existing = checkManagedRoot(paths, true), before = admitInputs();
  if (existing) {
    const descriptorSha256 = verifyStableJson(paths.descriptor, before.descriptorBytes, "Local Windows LLVM descriptor"),
      provenanceSha256 = verifyStableJson(paths.provenance, before.provenanceBytes, "Local Windows LLVM provenance"),
      resultFile = path.join(paths.output, "result.json");
    regularFile(resultFile, "Local Windows LLVM build receipt");
    const result = readBuildJson(resultFile);
    if (result?.plan?.output !== paths.output || result.plan.descriptorFile !== paths.descriptor || result.plan.archive !== paths.archive ||
        result.plan.jobs !== jobs || !isDeepStrictEqual(result.plan.tools, {cmake: tools.cmake, ninja: tools.ninja})) {
      throw new Error("Local Windows LLVM warm output belongs to another descriptor, archive, tools or jobs; preserve it");
    }
    const admission = verify(paths.output, environment);
    verifyOutputInventory(paths.output, result);
    const after = admitInputs();
    if (!isDeepStrictEqual(before.archiveReceipt, after.archiveReceipt) || !isDeepStrictEqual(before.linux, after.linux) ||
        !isDeepStrictEqual(before.descriptorTools, after.descriptorTools) || !isDeepStrictEqual(before.composed, after.composed) ||
        descriptorSha256 !== fileHash(paths.descriptor) || provenanceSha256 !== fileHash(paths.provenance)) {
      throw new Error("Local Windows LLVM inputs changed during warm admission; preserve the output and investigate");
    }
    return {workspace, root: paths.root, output: paths.output, descriptor: paths.descriptor, provenance: paths.provenance,
      archive: paths.archive, descriptorSha256, provenanceSha256, archiveReceipt: after.archiveReceipt, linux: after.linux,
      admission, jobs, buildOnly: true, compiled: true, executed: false, publicationQualified: false, reused: true};
  }

  fs.mkdirSync(paths.root, {mode: 0o700});
  fs.writeFileSync(paths.descriptor, before.descriptorBytes, {flag: "wx", mode: 0o444});
  fs.writeFileSync(paths.provenance, before.provenanceBytes, {flag: "wx", mode: 0o444});
  configure(paths.descriptor, paths.archive, paths.output, {cmake: tools.cmake, ninja: tools.ninja}, jobs, environment);
  if (!Array.isArray(windowsLlvm.buildPhases) || !windowsLlvm.buildPhases.length) {throw new Error("Maintained Windows LLVM builder has no public build phases");}
  for (const phase of windowsLlvm.buildPhases) {buildPhase(paths.output, phase.name, environment);}
  audit(paths.output, environment);
  const admission = verify(paths.output, environment), outputReceiptFile = path.join(paths.output, "result.json");
  regularFile(outputReceiptFile, "Local Windows LLVM build receipt");
  verifyOutputInventory(paths.output, readBuildJson(outputReceiptFile));
  const after = admitInputs();
  checkManagedRoot(paths, false);
  const descriptorSha256 = verifyStableJson(paths.descriptor, before.descriptorBytes, "Local Windows LLVM descriptor"),
    provenanceSha256 = verifyStableJson(paths.provenance, before.provenanceBytes, "Local Windows LLVM provenance");
  if (!isDeepStrictEqual(before.archiveReceipt, after.archiveReceipt) || !isDeepStrictEqual(before.linux, after.linux) ||
      !isDeepStrictEqual(before.descriptorTools, after.descriptorTools) || !isDeepStrictEqual(before.composed, after.composed)) {
    throw new Error("Local Windows LLVM inputs changed during production; preserve the output and investigate");
  }
  return {workspace, root: paths.root, output: paths.output, descriptor: paths.descriptor, provenance: paths.provenance,
    archive: paths.archive, descriptorSha256, provenanceSha256, archiveReceipt: after.archiveReceipt, linux: after.linux,
    admission, jobs, buildOnly: true, compiled: true, executed: false, publicationQualified: false, reused: false};
}

module.exports = {localWindowsLlvmSdkPaths, localWindowsLlvmDescriptorTools, verifyLocalWindowsLlvmOutput, buildLocalNativeWindowsLlvmSdk};
