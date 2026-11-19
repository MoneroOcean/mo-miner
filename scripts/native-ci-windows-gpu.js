"use strict";

// Native CI Windows GPU factory profiles cross-built on Linux and their handoffs.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");
const {appendWorkflowOutputs, ciBuildJobs, ciSuffix, runNode} = require("./native-ci-job");
const {fileHash, nativeSourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {listState, packState, safeRelative, temp} = require("./native-ci-tar");
const {windowsControllerReadJson, windowsControllerRegularFile} = require("./native-ci-windows-controller");
const repo = path.resolve(__dirname, "..");

const windowsGpuHandoffProfiles = Object.freeze(["acpp-cuda", "acpp-hip", "dpcpp", "dpcpp-opencl", "oneapi"]);

const windowsGpuHandoffMaxFiles = 512, windowsGpuHandoffMaxBytes = 512 << 20;

const windowsGpuPackReserveMs = 7 * 60 * 1000, windowsGpuMinBuildMs = 5 * 60 * 1000, windowsGpuMaxBuildMs = 45 * 60 * 1000;

/** Keep one Windows GPU factory build and its archive inside the job-start deadline.
 * @param {string} deadlineEpoch @param {"build" | "package"} phase @param {number} [nowMs] */
function windowsGpuDeadlineBudget(deadlineEpoch, phase, nowMs = Date.now()) {
  if (!/^\d{10}$/.test(deadlineEpoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 || !["build", "package"].includes(phase)) {
    throw new Error("Windows GPU profile stage needs a valid bounded job-start deadline");
  }
  const remainingMs = Number(deadlineEpoch) * 1000 - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs <= 0 || remainingMs > 60 * 60 * 1000) {
    throw new Error("Windows GPU profile deadline is expired or exceeds the 60-minute job bound");
  }
  if (phase === "build") {
    if (remainingMs < windowsGpuMinBuildMs + windowsGpuPackReserveMs + 60 * 1000) {
      throw new Error("Insufficient aggregate job time remains before the Windows GPU producer");
    }
    return {timeoutMs: Math.min(windowsGpuMaxBuildMs, remainingMs - windowsGpuPackReserveMs), remainingMs};
  }
  if (remainingMs < 60 * 1000) {throw new Error("Insufficient aggregate job time remains for the Windows GPU handoff");}
  return {timeoutMs: Math.min(windowsGpuPackReserveMs, remainingMs), remainingMs};
}

/** Run-scoped producer paths for one independent Windows GPU factory job.
 * @param {string} profile @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {string} [architecture] */
function windowsGpuHandoffStatePaths(profile, environment = process.env, platform = process.platform, architecture = process.arch) {
  const runnerTemp = environment["RUNNER_TEMP"] || "";
  if (platform !== "linux" || architecture !== "x64" || !windowsGpuHandoffProfiles.includes(profile) || !path.isAbsolute(runnerTemp)) {
    throw new Error("Windows GPU handoff needs a Linux x64 runner, RUNNER_TEMP and one maintained profile");
  }
  const canonicalTemp = fs.realpathSync(runnerTemp), suffix = ciSuffix(environment),
    name = "mom-native-windows-gpu-" + profile + "-" + suffix, root = path.join(temp, name), output = path.join(root, "output"),
    isolatedRoot = path.join(output, "isolated"), receipt = path.join(output, "result.json"), isolation = path.join(isolatedRoot, "result.json"),
    configFile = path.join(root, "config.json"), archive = path.join(canonicalTemp, "native-windows-gpu-" + profile + "-" + suffix + ".tar");
  if (canonicalTemp !== path.resolve(runnerTemp) || path.dirname(root) !== temp || path.resolve(root) !== root ||
      environment["MOM_NATIVE_WINDOWS_GPU_DIR"] !== root || path.dirname(archive) !== canonicalTemp ||
      path.resolve(archive) !== archive || root === archive || root.startsWith(archive + path.sep)) {
    throw new Error("Windows GPU producer and archive paths differ from the CI work-root plan");
  }
  return {profile, suffix, name, root, output, isolatedRoot, receipt, isolation, configFile, archive, runnerTemp: canonicalTemp};
}

/** Validate one factory and select only its unchanged receipts and isolated objects. The
 * downstream full-application reader still re-admits all five graphs and the complete SDK.
 * @param {string} profile @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform]
 * @param {string} [architecture] @param {string} [sourceRoot] */
function windowsGpuProfileHandoffPlan(profile, environment = process.env, platform = process.platform,
  architecture = process.arch, sourceRoot = repo) {
  const state = windowsGpuHandoffStatePaths(profile, environment, platform, architecture), canonicalSource = path.resolve(sourceRoot);
  if (!path.isAbsolute(sourceRoot) || fs.realpathSync(canonicalSource) !== canonicalSource) {throw new Error("Windows GPU handoff source root must be canonical");}
  for (const directory of [state.root, state.output, state.isolatedRoot]) {
    const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
    if (!stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
      throw new Error("Windows GPU handoff producer root is not a canonical directory");
    }
  }
  const graph = windowsControllerReadJson(state.receipt, 16 << 20, "Windows GPU factory receipt"),
    configStat = windowsControllerRegularFile(state.configFile, "Windows GPU profile configuration"),
    configHash = fileHash(state.configFile), config = windowsControllerReadJson(state.configFile, 4 << 20, "Windows GPU profile configuration");
  if (configStat.nlink !== 1 || graph.version !== 1 || graph.profile !== profile ||
      graph.factory !== "mom_native_gpu_" + profile.replaceAll("-", "_") || graph.configFile !== state.configFile ||
      !/^[a-f0-9]{64}$/.test(graph.configSha256 || "") || configHash !== graph.configSha256 || config.profile !== profile ||
      graph.isolation !== state.isolation || !Array.isArray(graph.objects) || !graph.objects.length || graph.objects.length > 128 ||
      !graph.sourceState || !graph.sdkInputs || typeof graph.sdkInputs !== "object" || Array.isArray(graph.sdkInputs) ||
      Object.keys(graph.sdkInputs).length > 16384 ||
      Object.entries(graph.sdkInputs).some(([file, sha]) => !path.isAbsolute(file) || !/^[a-f0-9]{64}$/.test(sha || ""))) {
    throw new Error("Windows GPU receipt differs from the selected profile or its same-run inputs");
  }
  require("./build-native-windows-gpu").validateWindowsGpuConfig(config);
  verifySourceInputs(graph.sourceState, nativeSourceInputs(canonicalSource, "gpu"), "Windows GPU handoff");
  // Compiler inputs stay producer provenance in the compact handoff, so seal it only
  // while the original SDK cohort still matches; packaging rechecks this plan afterward.
  if (!Object.keys(graph.sdkInputs).length) {throw new Error("Windows GPU handoff omits its SDK input cohort");}
  for (const [file, sha256] of Object.entries(graph.sdkInputs)) {
    // Tool aliases are permitted by the original compiler receipts; bind their resolved
    // bytes rather than imposing a new no-symlink rule on otherwise valid SDK inputs.
    if (!fs.statSync(file).isFile() || fileHash(file) !== sha256) {
      throw new Error("Windows GPU SDK input changed before handoff: " + file);
    }
  }
  const isolatedStat = windowsControllerRegularFile(state.isolation, "Windows GPU isolation receipt"), isolated = windowsControllerReadJson(state.isolation, 16 << 20, "Windows GPU isolation receipt");
  if (isolatedStat.nlink !== 1 || isolated.prefix !== "mom_" + profile.replaceAll("-", "_") + "_gpu_" ||
      !isDeepStrictEqual(isolated.entryPoints, [graph.factory]) || !isDeepStrictEqual(isolated.outputs, graph.objects) ||
      ["nonDirectivePayloadsPreserved", "nonDirectiveMetadataPreserved", "comdatSelectionPreserved", "associativeXdataPreserved",
        "linkerDirectiveReferencesPreserved", "externalUndefinedImportsPreserved", "privateUndefinedReferencesRenamed"]
        .some(key => isolated.checks?.[key] !== true) || isolated.checks?.oldCollisionDefinitions !== 0 || isolated.checks?.oldCollisionReferences !== 0) {
    throw new Error("Windows GPU isolation output differs from the unchanged factory receipt");
  }
  const {coffSectionPayloads} = require("./native-compiler-isolation"),
    files = /** @type {Array<{path:string,member:string,bytes:number,sha256:string,role:string}>} */ ([]), seen = new Set(); let selectedBytes = 0;
  /** @param {string} file @param {string} expected @param {string} role */
  const select = (file, expected, role) => {
    const stat = windowsControllerRegularFile(file, "Windows GPU " + role), relative = path.relative(temp, file).split(path.sep).join("/");
    if (stat.nlink !== 1 || !safeRelative(relative) || !file.startsWith(state.root + path.sep) ||
        !/^[a-f0-9]{64}$/.test(expected || "") || fileHash(file) !== expected || seen.has(relative.toLowerCase())) {
      throw new Error("Windows GPU selected payload is unsafe, duplicated or changed: " + role);
    }
    seen.add(relative.toLowerCase()); selectedBytes += stat.size;
    if (!Number.isSafeInteger(selectedBytes) || selectedBytes > windowsGpuHandoffMaxBytes || files.length >= windowsGpuHandoffMaxFiles) {
      throw new Error("Windows GPU profile payload exceeds its bounded handoff size or file count");
    }
    files.push({path: file, member: relative, bytes: stat.size, sha256: expected, role});
  };
  select(state.receipt, fileHash(state.receipt), "factory receipt");
  select(state.isolation, fileHash(state.isolation), "isolation receipt");
  select(state.configFile, configHash, "profile configuration");
  for (const row of graph.objects) {
    if (!row || typeof row.path !== "string" || path.dirname(row.path) !== state.isolatedRoot ||
        !/^object-[0-9]{4}\.obj$/i.test(path.basename(row.path)) || !Number.isSafeInteger(row.bytes) || row.bytes < 1) {
      throw new Error("Windows GPU isolated object is outside its exact output directory");
    }
    const stat = windowsControllerRegularFile(row.path, "Windows GPU isolated object");
    if (stat.size !== row.bytes) {throw new Error("Windows GPU isolated object size differs from its receipt");}
    coffSectionPayloads(row.path);
    select(row.path, row.sha256, "isolated object");
  }
  return {state, files, members: files.map(row => row.member), selectedBytes, graphSha256: fileHash(state.receipt),
    isolationSha256: fileHash(state.isolation), configSha256: configHash};
}

/** Pack one independently built factory without its unrelated compiler-isolation diagnostics.
 * @param {string} profile @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform]
 * @param {string} [architecture] @param {string} [sourceRoot] */
function packWindowsGpuHandoff(profile, environment = process.env, platform = process.platform,
  architecture = process.arch, sourceRoot = repo, timeoutMs = 30 * 60 * 1000) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 30 * 60 * 1000) {
    throw new Error("Windows GPU profile handoff timeout is outside the bounded range");
  }
  const deadline = Date.now() + timeoutMs;
  /** @param {string} phase @param {number} [capMs] @param {number} [minimumMs] */
  const remaining = (phase, capMs = timeoutMs, minimumMs = 1) => {
    const milliseconds = deadline - Date.now();
    if (!Number.isSafeInteger(milliseconds) || milliseconds < minimumMs) {
      throw new Error("Windows GPU profile handoff deadline expired before " + phase);
    }
    return Math.min(milliseconds, capMs);
  };
  const plan = windowsGpuProfileHandoffPlan(profile, environment, platform, architecture, sourceRoot), state = plan.state,
    before = plan.files.map(({member, bytes, sha256, role}) => ({member, bytes, sha256, role})),
    graph = windowsControllerReadJson(state.receipt, 16 << 20, "Windows GPU factory receipt"),
    handoffFile = path.join(state.root, "handoff.json"),
    handoff = {format: "mom-windows-gpu-handoff-v1", profile,
      files: plan.files.map(row => ({path: path.relative(state.root, row.path).split(path.sep).join("/"), bytes: row.bytes, sha256: row.sha256}))
        .sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0),
      sdkInputs: Object.fromEntries(Object.keys(graph.sdkInputs).sort().map(file => [file, graph.sdkInputs[file]])),
      buildOnly: true, executed: false, publicationQualified: false},
    handoffBytes = Buffer.from(JSON.stringify(handoff, null, 2) + "\n");
  if (handoffBytes.length > 8 << 20) {throw new Error("Windows GPU handoff attestation exceeds its byte bound");}
  // This seals producer-local SDK provenance; the consumer still verifies original graph,
  // source, isolation, config and object bytes using externally supplied handoff hashes.
  fs.writeFileSync(handoffFile, handoffBytes, {flag: "wx", mode: 0o444});
  const handoffSha256 = fileHash(handoffFile), handoffMember = state.name + "/handoff.json",
    packed = packState(state.archive, temp, [...plan.members, handoffMember], remaining("archive creation")), actual = new Set();
  const listing = listState(state.archive, [state.name], remaining("archive listing", 5 * 60 * 1000, 1000));
  if (!Number.isSafeInteger(packed.bytes) || packed.bytes < 1 || packed.bytes > windowsGpuHandoffMaxBytes + (8 << 20)) {
    throw new Error("Windows GPU profile archive exceeds its bounded transfer size");
  }
  for (const raw of listing) {
    const name = raw.endsWith("/") ? raw.slice(0, -1) : raw;
    if (!name || actual.has(name)) {throw new Error("Windows GPU handoff tar contains a duplicate member");}
    actual.add(name);
  }
  const allowed = new Set([...before.map(row => row.member), handoffMember]);
  for (const file of before) {
    let parent = path.posix.dirname(file.member);
    while (parent !== ".") {allowed.add(parent); parent = path.posix.dirname(parent);}
  }
  if (!actual.has(handoffMember) || before.some(row => !actual.has(row.member)) || [...actual].some(name => !allowed.has(name))) {
    throw new Error("Windows GPU handoff tar contains extra, missing or substituted members");
  }
  remaining("final input admission");
  const afterPlan = windowsGpuProfileHandoffPlan(profile, environment, platform, architecture, sourceRoot),
    after = afterPlan.files.map(({member, bytes, sha256, role}) => ({member, bytes, sha256, role}));
  if (!isDeepStrictEqual(before, after) || afterPlan.state.archive !== state.archive || fileHash(handoffFile) !== handoffSha256) {
    throw new Error("Windows GPU source payload changed while packaging its handoff");
  }
  remaining("final input admission");
  return {version: 1, format: "mom-windows-gpu-profile-handoff-v1", profile,
    root: state.name, archive: state.archive, bundleSha256: fileHash(state.archive), bundleBytes: packed.bytes,
    handoffSha256, files: before.length + 1, selectedBytes: plan.selectedBytes + handoffBytes.length,
    buildOnly: true, executed: false, publicationQualified: false};
}

/** Validate every external input and output path before creating the run-scoped producer root.
 * The expected digest is the caller's same-run handoff pin, not publisher authentication.
 * @param {string} profile @param {string} configSource @param {string} expectedConfigSha256
 * @param {NodeJS.ProcessEnv} [environment] @param {NodeJS.Platform} [platform] @param {string} [architecture] @param {number} [nowMs] */
function windowsGpuProfileCiPreflight(profile, configSource, expectedConfigSha256, environment = process.env,
  platform = process.platform, architecture = process.arch, nowMs = Date.now()) {
  const state = windowsGpuHandoffStatePaths(profile, environment, platform, architecture),
    deadlineEpoch = environment["MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH"] || "";
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows GPU profile CI requires MOM_AOT_BUILD_ONLY=1");}
  const {windowsGpuBuildJobs} = require("./build-native-windows-gpu");
  // Like every CI producer, use the workflow's validated parallelism (2 or 4 inside the 12 GiB scope).
  if (windowsGpuBuildJobs(environment, platform) !== Number(ciBuildJobs(environment))) {
    throw new Error("Windows GPU profile CI must use the workflow's MOM_BUILD_JOBS");
  }
  const buildBudget = windowsGpuDeadlineBudget(deadlineEpoch, "build", nowMs);
  windowsGpuDeadlineBudget(deadlineEpoch, "package", nowMs);
  if (typeof expectedConfigSha256 !== "string" || !/^[a-f0-9]{64}$/.test(expectedConfigSha256)) {
    throw new Error("Windows GPU profile CI requires the explicit same-run configuration SHA-256");
  }
  if (typeof configSource !== "string" || configSource.length > 4096 || /[\0\r\n]/.test(configSource) || !path.isAbsolute(configSource)) {
    throw new Error("Windows GPU profile configuration must be a canonical absolute file path");
  }
  const configStat = windowsControllerRegularFile(configSource, "Same-run Windows GPU configuration"),
    configRelative = path.relative(state.root, path.resolve(configSource));
  if (configStat.nlink !== 1 || configStat.size < 1 || configStat.size > (4 << 20) ||
      configRelative === "" || (!path.isAbsolute(configRelative) && configRelative !== ".." && !configRelative.startsWith(".." + path.sep))) {
    throw new Error("Windows GPU configuration must be a bounded external regular file");
  }
  const configBytes = fs.readFileSync(configSource);
  if (configBytes.length !== configStat.size || configBytes.length < 1 || configBytes.length > (4 << 20)) {
    throw new Error("Windows GPU configuration changed or exceeds its bounded input size");
  }
  const configSha256 = createHash("sha256").update(configBytes).digest("hex");
  if (configSha256 !== expectedConfigSha256) {throw new Error("Windows GPU configuration differs from its same-run SHA-256 pin");}
  let configValue;
  try {
    configValue = JSON.parse(configBytes.toString("utf8"));
  } catch (error) {throw new Error("Same-run Windows GPU configuration is not valid JSON", {cause: error});}
  const config = require("./build-native-windows-gpu").validateWindowsGpuConfig(configValue);
  if (config.profile !== profile) {throw new Error("Windows GPU configuration profile differs from the explicit producer profile");}
  const workflowFiles = /** @type {Array<[string, string]>} */ ([[environment["GITHUB_OUTPUT"] || "", "GitHub output"],
    [environment["GITHUB_STEP_SUMMARY"] || "", "GitHub step summary"]]);
  if (workflowFiles.some(([name]) => !name)) {throw new Error("Windows GPU profile CI requires GitHub output and summary files");}
  for (const [name, label] of workflowFiles) {
    windowsControllerRegularFile(name, label);
  }
  if (path.resolve(environment["GITHUB_OUTPUT"] || "") === path.resolve(environment["GITHUB_STEP_SUMMARY"] || "") ||
      [state.root, state.output, state.archive].some(file => fs.existsSync(file))) {
    throw new Error("Windows GPU profile CI requires distinct workflow reports and fresh producer/output paths");
  }
  return {state, configSource, expectedConfigSha256, configSha256, configBytes, config, deadlineEpoch, buildTimeoutMs: buildBudget.timeoutMs};
}

/** Stage exactly one caller-pinned profile config through the existing public producer and handoff.
 * @param {string} profile @param {string} configSource @param {string} expectedConfigSha256 @param {NodeJS.ProcessEnv} [environment] */
function buildWindowsGpuProfileCi(profile, configSource, expectedConfigSha256, environment = process.env) {
  const input = windowsGpuProfileCiPreflight(profile, configSource, expectedConfigSha256, environment), state = input.state;
  fs.mkdirSync(state.root, {mode: 0o700});
  fs.writeFileSync(state.configFile, input.configBytes, {flag: "wx", mode: 0o444});
  const copiedStat = windowsControllerRegularFile(state.configFile, "Staged Windows GPU configuration");
  if (copiedStat.nlink !== 1 || copiedStat.size < 1 || copiedStat.size > (4 << 20) || fileHash(state.configFile) !== input.configSha256) {
    throw new Error("Staged Windows GPU configuration differs from its same-run pin");
  }
  require("./build-native-windows-gpu").validateWindowsGpuConfig(
    windowsControllerReadJson(state.configFile, 4 << 20, "Staged Windows GPU configuration"));
  if (fileHash(input.configSource) !== input.expectedConfigSha256) {throw new Error("Same-run Windows GPU configuration changed before producer start");}
  const buildEnvironment = {...environment, MOM_AOT_BUILD_ONLY: "1"};
  runNode("scripts/build-native-windows-gpu.js", [state.configFile, state.output], buildEnvironment, input.buildTimeoutMs);
  const producer = windowsControllerReadJson(state.receipt, 16 << 20, "Windows GPU producer receipt"), resultSha256 = fileHash(state.receipt);
  if (producer.version !== 1 || producer.profile !== profile || producer.factory !== "mom_native_gpu_" + profile.replaceAll("-", "_") ||
      producer.configFile !== state.configFile || producer.configSha256 !== input.configSha256 ||
      fileHash(input.configSource) !== input.expectedConfigSha256 || fileHash(state.configFile) !== input.configSha256) {
    throw new Error("Windows GPU producer receipt does not preserve the selected profile and pinned config");
  }
  const packBudget = windowsGpuDeadlineBudget(input.deadlineEpoch, "package"),
    packed = packWindowsGpuHandoff(profile, environment, process.platform, process.arch, repo, packBudget.timeoutMs),
    values = {windows_gpu_profile: profile, windows_gpu_bundle_sha256: packed.bundleSha256,
      windows_gpu_result_sha256: resultSha256, windows_gpu_config_sha256: input.configSha256,
      windows_gpu_handoff_sha256: packed.handoffSha256,
      windows_gpu_bundle_bytes: String(packed.bundleBytes), windows_gpu_files: String(packed.files),
      windows_gpu_build_only: "true", windows_gpu_executed: "false", windows_gpu_publication_qualified: "false"};
  if (packed.buildOnly !== true || packed.executed !== false || packed.publicationQualified !== false || fileHash(state.receipt) !== resultSha256 ||
      fileHash(state.archive) !== packed.bundleSha256) {throw new Error("Windows GPU profile handoff changed or carries an unsafe scope");}
  appendWorkflowOutputs(environment, values);
  const summary = environment["GITHUB_STEP_SUMMARY"] || "", summaryBody = `### Windows GPU profile ${profile}\n` +
    `- Retained the original public producer receipt and ${packed.files} unchanged files (${packed.selectedBytes} bytes).\n` +
    `- Bundle SHA-256: ${packed.bundleSha256}; producer result SHA-256: ${resultSha256}.\n` +
    "- Build-only: true; executed: false; publication-qualified: false. This is not GPU/runtime qualification.\n";
  windowsControllerRegularFile(summary, "GitHub step summary");
  fs.appendFileSync(summary, summaryBody, "utf8");
  console.log(`[native-ci] Built and retained the same-run ${profile} Windows GPU factory; build-only, not executed or publication-qualified (${packed.files} files).`);
  return {...packed, resultSha256, configSha256: input.configSha256};
}

module.exports = {buildWindowsGpuProfileCi, packWindowsGpuHandoff, windowsGpuDeadlineBudget, windowsGpuHandoffStatePaths,
  windowsGpuProfileCiPreflight, windowsGpuProfileHandoffPlan};
