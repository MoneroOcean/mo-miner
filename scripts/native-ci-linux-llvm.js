"use strict";

// Native CI Linux LLVM SDK: build or reuse the cached SDK, admission, and the installed-SDK and tool handoffs.
const fs = require("node:fs"), path = require("node:path");
const {appendWorkflowOutputs, removeTree, runNode} = require("./native-ci-job");
const {ciDeadline, boundedStepMs} = require("./ci/job");
const {fileHash} = require("./build-source-inputs");
const {listState, packState, tarExpandedBytes, temp} = require("./native-ci-tar");
const {provisionOneApi, restoreSources} = require("./native-ci-linux-sources");

/** Admit the pinned installed-SDK handoff that every LLVM consumer job restores. Consumers call it at
 * their input boundary and again only after an untrusted compiler child has run against the SDK.
 * @param {{output: string, env: NodeJS.ProcessEnv}} paths */
function admitLinuxLlvmOutput(paths) {
  const handoffSha256 = paths.env["MOM_NATIVE_LINUX_LLVM_SDK_HANDOFF_SHA256"] || "";
  if (!/^[a-f0-9]{64}$/.test(handoffSha256)) {throw new Error("Installed Linux LLVM SDK admission requires its external handoff pin");}
  return require("./native-linux-llvm-sdk-handoff").linuxLlvmSdkHandoffInputs(paths.output, {handoffSha256});
}

/** Export only source-built host tools needed by Windows LLVM, not the Linux build tree.
 * Full SDK admission runs on this producer; the consumer admits an explicitly selected subset.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildLinuxLlvmToolsCi(paths) {
  const name = "native-linux-llvm-tools-" + paths.suffix,
    workRoot = path.join(paths.runnerTemp, "mom-native-linux-llvm-tools-" + paths.suffix),
    bundle = path.join(workRoot, name), archive = path.join(paths.runnerTemp, name + ".tar"),
    {exportLinuxLlvmTools, linuxLlvmToolHandoffInputs} = require("./native-linux-llvm-tools");
  for (const file of [workRoot, archive]) {
    if (fs.lstatSync(file, {throwIfNoEntry: false})) {throw new Error("Linux LLVM tool export refuses an existing same-run output");}
  }
  fs.mkdirSync(workRoot, {mode: 0o700});
  exportLinuxLlvmTools(paths.output, bundle);
  const handoffSha256 = fileHash(path.join(bundle, "handoff.json")),
    expectation = {handoffSha256};
  linuxLlvmToolHandoffInputs(bundle, expectation);
  const packed = packState(archive, workRoot, [name], 5 * 60 * 1000), members = listState(archive, [name]);
  if (packed.bytes > 1024 ** 3 || tarExpandedBytes(archive, members.length) > 1024 ** 3) {
    throw new Error("Linux LLVM tool export exceeded its one-GiB transport bound");
  }
  linuxLlvmToolHandoffInputs(bundle, expectation);
  const values = {tools_bundle_sha256: fileHash(archive), tools_handoff_sha256: handoffSha256};
  appendWorkflowOutputs(paths.env, values);
  if (paths.env["GITHUB_STEP_SUMMARY"]) {
    fs.appendFileSync(paths.env["GITHUB_STEP_SUMMARY"], "### Selected Linux LLVM host tools\n" +
      "- The fully admitted source-built SDK supplied only Windows LLVM's public host-tool closure, the locked source archive, and unchanged producer receipts.\n" +
      "- The consumer verifies this hash-bound same-run subset; it does not re-admit or receive the complete Linux SDK/build tree. No application or runtime qualification is claimed.\n");
  }
  return {...values, bytes: packed.bytes};
}

/** Export the public installed LLVM prefix and unchanged admission receipts for Linux consumers.
 * The full source/build/SDK state artifact remains separate for Windows host tools.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildLinuxLlvmSdkHandoff(paths) {
  const name = "native-linux-llvm21-installed-sdk-" + paths.suffix,
    workRoot = path.join(paths.runnerTemp, "llvm-installed-sdk-export-" + paths.suffix),
    bundle = path.join(workRoot, paths.sdkName, "linux-llvm21-sdk"),
    archive = path.join(paths.runnerTemp, name + ".tar"),
    {exportLinuxLlvmSdk, linuxLlvmSdkHandoffInputs} = require("./native-linux-llvm-sdk-handoff");
  for (const item of [workRoot, archive]) {
    if (fs.lstatSync(item, {throwIfNoEntry: false})) {throw new Error("Linux installed LLVM SDK handoff refuses an existing same-run output");}
  }
  fs.mkdirSync(path.join(workRoot, paths.sdkName), {recursive: true, mode: 0o700});
  const produced = exportLinuxLlvmSdk(paths.output, bundle),
    expectation = {handoffSha256: produced.handoffSha256};
  linuxLlvmSdkHandoffInputs(bundle, expectation);
  const packed = packState(archive, workRoot, [paths.sdkName]), members = listState(archive, [paths.sdkName]);
  if (packed.bytes > 16 * 1024 ** 3 || tarExpandedBytes(archive, members.length) > 16 * 1024 ** 3) {
    throw new Error("Linux installed LLVM SDK handoff exceeded its transport bound");
  }
  linuxLlvmSdkHandoffInputs(bundle, expectation);
  const values = {llvm_sdk_handoff_sha256: produced.handoffSha256, bundle_sha256: fileHash(archive)};
  appendWorkflowOutputs(paths.env, values);
  if (paths.env["GITHUB_STEP_SUMMARY"]) {
    fs.appendFileSync(paths.env["GITHUB_STEP_SUMMARY"], "- Linux consumers receive a separate installed-prefix handoff, externally pinned by its unchanged manifest; the full source/build/SDK state remains available to Windows host-tool composition.\n");
  }
  return {...values, bytes: packed.bytes, bundle};
}

/** Build tools of the Linux LLVM SDK; their hashes are part of its admission and of its cache key. */
const linuxLlvmTools = Object.freeze({cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja", cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx",
  cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx"});

/** Keep the public Linux LLVM configure command on the validated run envelope.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "source" | "output" | "env">} paths */
function linuxLlvmConfigureCommand(paths) {
  return {script: "scripts/build-native-llvm-sdk.js", args: ["configure", path.join(paths.source, "llvm-project-21.1.8.src.tar.xz"),
    paths.output, linuxLlvmTools.cmake, linuxLlvmTools.ninja, linuxLlvmTools.cc, linuxLlvmTools.cxx, paths.env["MOM_BUILD_JOBS"] || ""]};
}

/** Cold-build the Linux LLVM SDK in one job: one source restore and oneAPI provisioning, then the
 * maintained configure, build-phase, install and audit order without intermediate state transfers.
 * Each child keeps its existing cap and is also bounded by the job deadline.
 * @param {ReturnType<typeof import("./native-ci-linux-sources").ciPaths>} paths */
function buildLinuxLlvmCi(paths) {
  const builder = "scripts/build-native-llvm-sdk.js", env = paths.env,
    phases = ["llvm-foundation", "llvm-core-analysis", "llvm-core-optimization", "llvm-core-codegen", "llvm-x86", "llvm-nvptx",
      "llvm-amdgpu", "clang-base", "clang-sema", "clang-frontend", "lld", "tools"],
    deadlineMs = ciDeadline(env) * 1000,
    childMs = (/** @type {number} */ capMs) => boundedStepMs(deadlineMs, capMs, 0, "Linux LLVM SDK job deadline leaves no time for the next child");
  restoreSources(paths);
  provisionOneApi(paths);
  if (!reuseCachedLinuxLlvm(paths)) {
    fs.mkdirSync(paths.sdkRoot, {mode: 0o700});
    const command = linuxLlvmConfigureCommand(paths);
    runNode(command.script, command.args, env, childMs(50 * 60 * 1000));
    for (const phase of phases) {runNode(builder, ["build", paths.output, phase], env, childMs(50 * 60 * 1000));}
    runNode(builder, ["install", paths.output], env, childMs(50 * 60 * 1000));
    runNode(builder, ["audit", paths.output], env, childMs(15 * 60 * 1000));
  }
  packState(path.join(paths.runnerTemp, "native-linux-llvm21-sdk-" + paths.suffix + ".tar"), temp, [paths.sourceName, paths.sdkName]);
}

/** An SDK root that actions/cache restored is reused only when the full public admission passes against
 * the current recipe, tools and pinned source archive; otherwise it is removed and rebuilt.
 * @param {Pick<ReturnType<typeof import("./native-ci-linux-sources").ciPaths>, "sdkRoot" | "output">} paths @returns {boolean} */
function reuseCachedLinuxLlvm(paths) {
  if (!fs.existsSync(paths.sdkRoot)) {return false;}
  try {
    require("./build-native-llvm-sdk").verifyLinuxLlvmAdmission(paths.output);
    console.log("[native-ci] Reusing the cached Linux LLVM SDK after full admission");
    return true;
  } catch (error) {
    console.warn("[native-ci] Cached Linux LLVM SDK failed admission and is rebuilt: " + (error instanceof Error ? error.message : String(error)));
    removeTree(paths.sdkRoot);
    return false;
  }
}

module.exports = {admitLinuxLlvmOutput, buildLinuxLlvmCi, buildLinuxLlvmSdkHandoff, buildLinuxLlvmToolsCi,
  linuxLlvmConfigureCommand, linuxLlvmTools, reuseCachedLinuxLlvm};
