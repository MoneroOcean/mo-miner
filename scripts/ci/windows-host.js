"use strict";

// Windows static host-component CI job: pin validation, same-run input restore and admission,
// COMGR/HIPRTC/ACPP builds, component export, and the archive-only final seal.

const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"),
  {createHash} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const ciJob = require("../native-ci-job"),
  ciRocm = require("../native-ci-linux-rocm"),
  ciSources = require("../native-ci-linux-sources"),
  ciTar = require("../native-ci-tar"),
  ciWindowsLlvm = require("../native-ci-windows-llvm"),
  dpcppInputs = require("../native-dpcpp-cuda-inputs"), {fileHash, readBuildJson, readBuildJsonSnapshot} = require("../build-source-inputs");
const {ciSuffix, validCiSuffix} = require("../native-ci-job");

const repo = path.resolve(__dirname, "../.."), digest = /^[a-f0-9]{64}$/, epochPattern = /^\d{10}$/,
  intelBin = "/opt/intel/oneapi/compiler/2026.2/bin", sourceGit = "/usr/bin/git",
  maxRunSeconds = 8 * 60 * 60, minJobSeconds = 300, minute = 60 * 1000;

/** External same-run artifact pins, by label, as exported in the job environment. */
const inputPins = {
  WINDOWS_STATE_BUNDLE: "MOM_NATIVE_WINDOWS_LLVM_STATE_BUNDLE_SHA256",
  WINDOWS_STATE_MANIFEST: "MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256",
  WINDOWS_ADMISSION: "MOM_NATIVE_WINDOWS_LLVM_ADMISSION_SHA256",
  HERMES_BUNDLE: "MOM_NATIVE_WINDOWS_LLVM_HERMES_BUNDLE_SHA256",
  HERMES_HANDOFF: "MOM_NATIVE_WINDOWS_LLVM_HERMES_HANDOFF_SHA256",
  LINUX_TOOLS_BUNDLE: "MOM_NATIVE_LINUX_LLVM_TOOLS_BUNDLE_SHA256",
  LINUX_TOOLS_HANDOFF: "MOM_NATIVE_LINUX_LLVM_TOOLS_HANDOFF_SHA256",
  LINUX_SDK_BUNDLE: "MOM_NATIVE_LINUX_SDK_BUNDLE_SHA256",
  LINUX_SOURCES_BUNDLE: "MOM_NATIVE_LINUX_SOURCE_BUNDLE_SHA256",
  DPCPP_CUDA_BUNDLE: "MOM_NATIVE_LINUX_DPCPP_CUDA_BUNDLE_SHA256",
  DPCPP_CUDA_MANIFEST: "MOM_NATIVE_LINUX_DPCPP_CUDA_MANIFEST_SHA256",
  ROCM_SOURCES_BUNDLE: "MOM_NATIVE_ROCM_SOURCE_BUNDLE_SHA256",
  ACPP_SOURCES_BUNDLE: "MOM_NATIVE_ACPP_SOURCE_BUNDLE_SHA256",
  HIP_HEADERS_BUNDLE: "MOM_NATIVE_ROCM_HIP_HEADERS_BUNDLE_SHA256",
  HIP_HEADERS_HANDOFF: "MOM_NATIVE_ROCM_HIP_HEADERS_HANDOFF_SHA256",
};

/** @typedef {keyof typeof inputPins} PinLabel */
/** @typedef {Record<PinLabel, string>} Pins */
/** @typedef {{suffix: string, runnerTemp: string, sdkName: string, deadlineMs: number}} RestoreContext */


/** Validate all 15 external SHA-256 pins before any artifact is touched. @param {NodeJS.ProcessEnv} env @returns {Pins} */
function externalPins(env) {
  const pins = /** @type {Pins} */ ({});
  for (const [label, name] of /** @type {Array<[PinLabel, string]>} */ (Object.entries(inputPins))) {
    const value = env[name] || "";
    if (!digest.test(value)) {throw new Error("Missing or malformed " + label + " pin");}
    pins[label] = value;
  }
  return pins;
}

/** Same-run artifact (and incoming file) names keyed by their workflow output names. @param {string} suffix */
function artifactNames(suffix) {
  return {
    windows_state_archive: `native-windows-llvm-state-${suffix}-audited.tar`,
    windows_state_manifest: `native-windows-llvm-state-${suffix}-audited.json`,
    hermes_archive: `native-windows-controller-hermes-${suffix}.tar`,
    linux_tools_archive: `native-linux-llvm-tools-${suffix}.tar`,
    linux_sdk_archive: `native-linux-llvm21-sdk-${suffix}.tar`,
    linux_sources_archive: `native-linux-sdk-state-${suffix}-sources.tar`,
    dpcpp_cuda_archive: `native-linux-dpcpp-cuda-inputs-${suffix}.tar`,
    rocm_sources_archive: `native-rocm-state-${suffix}-sources.tar`,
    acpp_sources_archive: `native-acpp-state-${suffix}-sources.tar`,
    hip_headers_archive: `native-rocm-hip-headers-${suffix}.tar`,
  };
}

/** Flat download location of one same-run artifact. @param {NodeJS.ProcessEnv} env @param {string} name */
function incoming(env, name) {
  const runnerTemp = env["RUNNER_TEMP"] || "";
  if (!path.isAbsolute(runnerTemp)) {throw new Error("Windows host-component job needs an absolute RUNNER_TEMP");}
  return path.join(runnerTemp, "incoming", name);
}

/** The workflow's job deadline (job start + timeout - 300 s). @param {NodeJS.ProcessEnv} env @param {number} nowSeconds */
function ciDeadline(env, nowSeconds) {
  const value = env["MOM_CI_DEADLINE_EPOCH"] || "";
  if (!epochPattern.test(value) || Number(value) <= nowSeconds) {throw new Error("Invalid or expired MOM_CI_DEADLINE_EPOCH");}
  return Number(value);
}

/** Bound the job by both the workflow deadline and the original Windows LLVM run deadline recorded in
 * its admitted state manifest; the state transport rejects any other run deadline.
 * @param {NodeJS.ProcessEnv} env @param {unknown} runDeadlineEpoch @param {number} nowSeconds */
function jobDeadlines(env, runDeadlineEpoch, nowSeconds) {
  const jobLimit = ciDeadline(env, nowSeconds);
  if (typeof runDeadlineEpoch !== "string" || !epochPattern.test(runDeadlineEpoch)) {
    throw new Error("Invalid original Windows LLVM run deadline");
  }
  const runEpoch = Number(runDeadlineEpoch);
  if (runEpoch <= nowSeconds || runEpoch - nowSeconds > maxRunSeconds) {
    throw new Error("Original Windows LLVM run deadline is expired or exceeds eight hours");
  }
  return {runEpoch, jobEpoch: Math.min(jobLimit, runEpoch)};
}

/** Derived per-job environment consumed by the host-SDK, LLVM-state and descriptor helpers.
 * @param {string} runnerTemp @param {string} suffix @param {{runEpoch: number, jobEpoch: number}} deadlines */
function jobEnvironment(runnerTemp, suffix, deadlines) {
  return {
    MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: String(deadlines.runEpoch),
    MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: String(deadlines.jobEpoch),
    ...ciWindowsLlvm.windowsHostSdkEnvironment(runnerTemp, suffix, deadlines.jobEpoch),
    MOM_NATIVE_WINDOWS_LLVM_DESCRIPTOR_DIR: path.join(runnerTemp, "mom-native-windows-llvm21-descriptor-" + suffix),
  };
}

/** Fail fast at job start: validate pins and budget, then publish the exact same-run artifact names.
 * @param {NodeJS.ProcessEnv} env @param {number} [nowSeconds] */
function validatePins(env, nowSeconds = Math.floor(Date.now() / 1000)) {
  const suffix = ciSuffix(env), output = env["GITHUB_OUTPUT"] || "";
  externalPins(env);
  const remaining = ciDeadline(env, nowSeconds) - nowSeconds;
  if (remaining > maxRunSeconds) {throw new Error("MOM_CI_DEADLINE_EPOCH exceeds eight hours");}
  if (remaining < minJobSeconds) {throw new Error("Insufficient budget remains for a bounded host-component job");}
  if (!path.isAbsolute(output)) {throw new Error("Windows host-component pins need an absolute GITHUB_OUTPUT");}
  const names = artifactNames(suffix);
  fs.appendFileSync(output, Object.entries(names).map(([key, value]) => key + "=" + value + "\n").join(""));
  return names;
}

/** Hash-admit one single-link canonical regular file and reject it if it changes while hashed.
 * @param {string} file @param {string} expectedSha256 */
function pinnedFile(file, expectedSha256) {
  const before = fs.lstatSync(file, {bigint: true});
  if (!before.isFile() || before.isSymbolicLink() || before.nlink !== 1n || before.size < 1n || fs.realpathSync(file) !== file ||
      fileHash(file) !== expectedSha256) {
    throw new Error("Same-run input differs from its external SHA-256 pin: " + path.basename(file));
  }
  const after = fs.lstatSync(file, {bigint: true});
  if (before.dev !== after.dev || before.ino !== after.ino || before.nlink !== after.nlink || before.size !== after.size ||
      before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs || fileHash(file) !== expectedSha256) {
    throw new Error("Same-run input changed while being admitted: " + path.basename(file));
  }
  return Number(after.size);
}

/** Read a small pinned manifest once through one descriptor so the parsed bytes are the hashed bytes.
 * @param {string} file @param {string} expectedSha256 @param {number} [maxBytes] */
function pinnedManifest(file, expectedSha256, maxBytes = 1 << 20) {
  if (!digest.test(expectedSha256 || "") || !Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > 1 << 20) {
    throw new Error("Windows LLVM state manifest pin or byte bound is invalid");
  }
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK);
  try {
    const before = fs.fstatSync(fd, {bigint: true});
    if (!before.isFile() || before.nlink !== 1n || before.size < 1n || before.size > BigInt(maxBytes) || fs.realpathSync(file) !== file) {
      throw new Error("Windows LLVM state manifest is not a bounded canonical regular file");
    }
    const bytes = Buffer.alloc(Number(before.size));
    let offset = 0;
    while (offset < bytes.length) {
      const count = fs.readSync(fd, bytes, offset, bytes.length - offset, null);
      if (!count) {break;}
      offset += count;
    }
    const after = fs.fstatSync(fd, {bigint: true}), current = fs.lstatSync(file, {bigint: true}),
      sha256 = createHash("sha256").update(bytes).digest("hex");
    if (offset !== bytes.length || before.dev !== after.dev || before.ino !== after.ino || before.nlink !== after.nlink ||
        before.size !== after.size || before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs ||
        current.dev !== after.dev || current.ino !== after.ino || !current.isFile() || current.isSymbolicLink() ||
        current.nlink !== 1n || fs.realpathSync(file) !== file || sha256 !== expectedSha256) {
      throw new Error("Windows LLVM state manifest changed or differs from its external SHA-256 pin");
    }
    return {bytes, sha256};
  } finally {fs.closeSync(fd);}
}

/** Admit the job identity, pins, Windows LLVM state manifest and the derived deadlines/environment.
 * @param {NodeJS.ProcessEnv} env @param {number} [nowMs] */
function admitHostJob(env, nowMs = Date.now()) {
  const suffix = ciSuffix(env), pins = externalPins(env), names = artifactNames(suffix),
    runnerTemp = env["RUNNER_TEMP"] || "", nowSeconds = Math.floor(nowMs / 1000);
  ciDeadline(env, nowSeconds);
  const manifestPin = pinnedManifest(incoming(env, names.windows_state_manifest), pins.WINDOWS_STATE_MANIFEST),
    manifest = /** @type {unknown} */ (JSON.parse(manifestPin.bytes.toString("utf8"))),
    runDeadline = manifest && typeof manifest === "object" ? /** @type {Record<string, unknown>} */ (manifest)["runDeadlineEpoch"] : undefined,
    deadlines = jobDeadlines(env, runDeadline, nowSeconds);
  return {suffix, pins, names, runnerTemp, manifest, manifestSha256: manifestPin.sha256, deadlines,
    environment: jobEnvironment(runnerTemp, suffix, deadlines)};
}

/** Remaining job time minus a reserve for final reporting. @param {number} deadlineMs @param {number} [reserveMs] */
function remaining(deadlineMs, reserveMs = minute) {
  const ms = deadlineMs - Date.now();
  if (!Number.isSafeInteger(ms) || ms <= reserveMs) {throw new Error("Windows host-component job deadline exhausted");}
  return ms - reserveMs;
}

/** Cap one phase by its own limit and by the job deadline. @param {number} deadlineMs @param {number} capMs @param {number} [reserveMs] */
function boundedTimeout(deadlineMs, capMs, reserveMs = minute) {
  const ms = Math.min(capMs, remaining(deadlineMs, reserveMs));
  if (!Number.isSafeInteger(ms) || ms < 2_000) {throw new Error("Insufficient deadline for the next bounded host-component phase");}
  return ms;
}

/** Run one repository Node script inside the memory-scoped, signal-bounded CI wrapper.
 * @param {number} deadlineMs @param {string} script @param {string[]} args @param {number} capMs */
function runBounded(deadlineMs, script, args, capMs) {
  const timeoutMs = boundedTimeout(deadlineMs, capMs), seconds = Math.floor(timeoutMs / 1000),
    memoryScope = path.join(repo, "scripts/run-bounded-native-ci.sh"),
    result = spawnSync(memoryScope, ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", seconds + "s", process.execPath,
      path.join(repo, script), ...args], {cwd: repo, env: process.env, stdio: "inherit", timeout: timeoutMs + 7_000});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error("Bounded native host-component phase failed: " + script + (result.signal ? " (" + result.signal + ")" : ""), {cause: result.error});
  }
}

/** Free bytes for unprivileged writers on the filesystem holding a directory. @param {string} directory */
function availableBytes(directory) {
  const stat = fs.statfsSync(directory, {bigint: true});
  if (stat.bsize <= 0n) {throw new Error("Cannot determine host-component input filesystem capacity");}
  const bytes = stat.bavail * stat.bsize;
  return bytes > BigInt(Number.MAX_SAFE_INTEGER) ? Number.MAX_SAFE_INTEGER : Number(bytes);
}

/** Restore exact roots from a pinned same-run tar, re-admitting the archive around every tar pass.
 * The Linux LLVM tools must land under their configure-time absolute root, created fresh here.
 * @param {RestoreContext} context @param {string} file @param {string} expectedSha256 @param {string[]} roots
 * @param {string} base @param {number} maxExpandedBytes @param {number} [timeoutCapMs] @param {boolean} [createBase] */
function restore(context, file, expectedSha256, roots, base, maxExpandedBytes, timeoutCapMs = 25 * minute, createBase = false) {
  pinnedFile(file, expectedSha256);
  const members = ciTar.listState(file, roots, boundedTimeout(context.deadlineMs, 5 * minute)),
    expandedBytes = ciTar.tarExpandedBytes(file, members.length, boundedTimeout(context.deadlineMs, 15 * minute));
  if (expandedBytes > maxExpandedBytes) {throw new Error("Same-run input archive exceeds its expanded-size bound: " + path.basename(file));}
  if (roots.includes(context.sdkName)) {
    ciTar.linuxNativeSdkDiskPreflight(expandedBytes, availableBytes(base), availableBytes(context.runnerTemp));
  }
  pinnedFile(file, expectedSha256);
  if (createBase) {
    const expectedBase = path.join(context.runnerTemp, `linux-llvm-tools-extracted-${context.suffix}`);
    if (base !== expectedBase || roots.length !== 1 || roots[0] !== `native-linux-llvm-tools-${context.suffix}` ||
        fs.lstatSync(base, {throwIfNoEntry: false})) {
      throw new Error("Linux LLVM tool restore requires its fresh exact same-run extraction root");
    }
    fs.mkdirSync(base, {mode: 0o700});
  }
  ciTar.extractState(file, base, roots, roots, boundedTimeout(context.deadlineMs, timeoutCapMs, 90_000));
  pinnedFile(file, expectedSha256);
  return {members, expandedBytes};
}

/** Restore all non-LLVM-state same-run inputs to their original absolute roots.
 * @param {RestoreContext} context @param {NodeJS.ProcessEnv} env @param {Pins} pins
 * @param {ReturnType<typeof artifactNames>} names @param {string} sourceName */
function restoreSameRunInputs(context, env, pins, names, sourceName) {
  const suffix = context.suffix, toolName = "native-linux-llvm-tools-" + suffix,
    toolsRoot = path.join(context.runnerTemp, `linux-llvm-tools-extracted-${suffix}`), gib = 1024 ** 3;
  restore(context, incoming(env, names.hermes_archive), pins.HERMES_BUNDLE, ["native-windows-controller-hermes-" + suffix],
    context.runnerTemp, 8 * gib);
  restore(context, incoming(env, names.linux_tools_archive), pins.LINUX_TOOLS_BUNDLE, [toolName], toolsRoot, gib, 25 * minute, true);
  restore(context, incoming(env, names.linux_sdk_archive), pins.LINUX_SDK_BUNDLE, [sourceName, context.sdkName], "/tmp", 64 * gib, 30 * minute);
  restore(context, incoming(env, names.linux_sources_archive), pins.LINUX_SOURCES_BUNDLE,
    ["mom-native-linux-source-prerequisites-" + suffix], "/tmp", 8 * gib);
  restore(context, incoming(env, names.dpcpp_cuda_archive), pins.DPCPP_CUDA_BUNDLE, ["mom-native-dpcpp-cuda-inputs-" + suffix], "/tmp", 32 * gib);
  restore(context, incoming(env, names.rocm_sources_archive), pins.ROCM_SOURCES_BUNDLE, ["mom-native-rocm-prerequisites-" + suffix], "/tmp", 16 * gib);
  restore(context, incoming(env, names.acpp_sources_archive), pins.ACPP_SOURCES_BUNDLE, ["mom-native-acpp-prerequisites-" + suffix], "/tmp", 8 * gib);
  restore(context, incoming(env, names.hip_headers_archive), pins.HIP_HEADERS_BUNDLE, ["native-rocm-hip-headers-" + suffix], "/tmp", 128 * 1024 ** 2);
  return {toolsBundle: path.join(toolsRoot, toolName), hermesRoot: path.join(context.runnerTemp, "native-windows-controller-hermes-" + suffix)};
}

/** Admit every restored same-run input through its public verifier.
 * @param {ReturnType<typeof ciSources.ciPaths>} paths @param {Pins} pins @param {string} suffix
 * @param {{toolsBundle: string, hermesRoot: string}} restored */
function verifySameRunInputs(paths, pins, suffix, restored) {
  const linuxSdk = require("../build-native-llvm-sdk").verifyLinuxLlvmAdmission(paths.output),
    linuxSources = ciSources.linuxSourceStatePaths(paths), dpcpp = ciSources.dpcppCudaInputsStatePaths(paths),
    rocm = ciSources.rocmSourceStatePaths(paths), acpp = ciSources.acppStatePaths(paths),
    toolInputs = require("../native-linux-llvm-tools").linuxLlvmToolHandoffInputs(restored.toolsBundle,
      {handoffSha256: pins.LINUX_TOOLS_HANDOFF}),
    hermes = require("../native-windows-controller-handoff").windowsControllerHandoffInputs(restored.hermesRoot,
      {kind: "hermes", handoffSha256: pins.HERMES_HANDOFF}),
    hipHeaders = require("../native-rocm-hip-headers").rocmHipHeaderInputs(path.join("/tmp", "native-rocm-hip-headers-" + suffix),
      {handoffSha256: pins.HIP_HEADERS_HANDOFF});
  if (!linuxSdk.sdk || !linuxSdk.sdk.path || hermes.kind !== "hermes" || !hermes.intelRuntime ||
      toolInputs.toolRoot !== path.join(restored.toolsBundle, "bin")) {
    throw new Error("Restored same-run SDK, Hermes runtime or thin Linux LLVM tool handoff is incomplete");
  }
  ciSources.verifyLinuxSources(paths, linuxSources, sourceGit);
  const dpcppManifest = readBuildJson(dpcpp.manifest);
  if (ciSources.dpcppSourceArchivePin(dpcpp.incomingSourceTar, pins.LINUX_SOURCES_BUNDLE) !== dpcppManifest.source?.tarSha256) {
    throw new Error("DPC++/CUDA manifest source cohort is not the externally pinned Linux source archive");
  }
  dpcppInputs.auditDpcppCudaInputs(dpcpp, dpcppManifest, pins.DPCPP_CUDA_MANIFEST);
  ciSources.verifyRocmSources(paths, rocm, sourceGit);
  ciSources.verifyAcppSource(paths, acpp, sourceGit);
  return {linuxSdkPath: linuxSdk.sdk.path, dpcpp, rocm, acpp, toolInputs, hipHeaders};
}

/** Reacquire the Windows host SDK and write this job's fresh descriptor recomposition input.
 * @param {string} runnerTemp @param {string} suffix @param {string} hermesRoot @param {string} hermesHandoffSha256
 * @param {{lldLink: string, llvmNm: string}} toolInputs */
function writeDescriptorInput(runnerTemp, suffix, hermesRoot, hermesHandoffSha256, toolInputs) {
  const hostSdkMetadata = ciWindowsLlvm.buildWindowsHostSdkSourcesCi(),
    windowsSdkRoot = ciWindowsLlvm.windowsHostSdkPaths(runnerTemp, suffix).sdkOutput,
    windowsSdkPins = {resultSha256: fileHash(path.join(windowsSdkRoot, "result.json")), descriptorSha256: fileHash(path.join(windowsSdkRoot, "sdk.json")),
      filesSha256: fileHash(path.join(windowsSdkRoot, "files.json")), overlaySha256: fileHash(path.join(windowsSdkRoot, "vfs-overlay.json"))};
  if (!hostSdkMetadata || !Object.values(windowsSdkPins).every(value => digest.test(value))) {
    throw new Error("Locally reacquired Windows host SDK is missing its hash-pinned composition receipts");
  }
  const tools = {clangCl: {path: intelBin + "/icx-cl", sha256: fileHash(intelBin + "/icx-cl")},
      lldLink: {path: toolInputs.lldLink, sha256: fileHash(toolInputs.lldLink), linkTarget: "lld"},
      llvmNm: {path: toolInputs.llvmNm, sha256: fileHash(toolInputs.llvmNm)},
      clangResourceInclude: "/opt/intel/oneapi/compiler/2026.2/lib/clang/23/include"},
    input = {windowsSdkRoot, windowsSdkPins, hermesRoot,
      hermesExpectation: /** @type {{kind: "hermes", handoffSha256: string}} */ (
        {kind: "hermes", handoffSha256: hermesHandoffSha256}), tools},
    file = path.join(runnerTemp, "windows-llvm21-descriptor-input-" + suffix + ".json"),
    bytes = Buffer.from(JSON.stringify(input) + "\n");
  if (fs.existsSync(file) || bytes.length > 8 * 1024 * 1024) {
    throw new Error("Windows LLVM descriptor recomposition input is not fresh and bounded");
  }
  fs.writeFileSync(file, bytes, {flag: "wx", mode: 0o600});
  return {input, file, bytes};
}

/** Restore the audited Windows LLVM state, then prove its admission receipt and descriptor
 * still recompose from this job's SDK, tools and Hermes handoff.
 * @param {ReturnType<typeof admitHostJob>} job @param {string} archive @param {ReturnType<typeof writeDescriptorInput>} descriptor */
function restoreWindowsLlvm(job, archive, descriptor) {
  const windowsState = require("../native-windows-llvm-state").restoreWindowsNativeLlvmState(archive,
      job.pins.WINDOWS_STATE_BUNDLE, job.manifest, job.manifestSha256, process.env).state,
    windowsLlvm = require("../build-native-windows-llvm-sdk").verifyWindowsLlvmAdmission(windowsState.output, process.env),
    admissionSha = fileHash(path.join(windowsState.output, "admitted.json"));
  if (admissionSha !== job.pins.WINDOWS_ADMISSION) {
    throw new Error("Restored Windows LLVM admission receipt differs from the external pin");
  }
  const composed = ciWindowsLlvm.composeWindowsLlvmSdkDescriptor(descriptor.input),
    originalInput = fs.readFileSync(path.join(windowsState.root, "descriptor-input.json")),
    originalProvenance = readBuildJson(path.join(windowsState.root, "descriptor-provenance.json")),
    expectedDescriptor = Buffer.from(JSON.stringify(composed.descriptor, null, 2) + "\n");
  if (!descriptor.bytes.equals(originalInput) || originalProvenance.status !== "PASS" ||
      originalProvenance.input?.sha256 !== fileHash(descriptor.file) ||
      !isDeepStrictEqual(originalProvenance.composition, composed.provenance) ||
      !expectedDescriptor.equals(fs.readFileSync(windowsState.descriptor))) {
    throw new Error("Restored Windows LLVM descriptor no longer recomposes from this job's SDK, tools and Hermes handoff");
  }
  if (!windowsLlvm.admission || fileHash(windowsLlvm.admissionFile) !== admissionSha) {
    throw new Error("Windows LLVM state did not pass the public full admission verifier");
  }
  return windowsState;
}

/** Run git on the local bare ROCm CLR transport with a clean, file-only, non-interactive environment.
 * @param {number} deadlineMs @param {string[]} args @param {number} [capMs] */
function runSourceGit(deadlineMs, args, capMs = 2 * minute) {
  const timeoutMs = boundedTimeout(deadlineMs, capMs), seconds = Math.floor(timeoutMs / 1000),
    sourceEnv = Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith("GIT_")));
  Object.assign(sourceEnv, {GIT_ALLOW_PROTOCOL: "file", GIT_ATTR_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null",
    GIT_CONFIG_NOSYSTEM: "1", GIT_NO_LAZY_FETCH: "1", GIT_NO_REPLACE_OBJECTS: "1", GIT_OPTIONAL_LOCKS: "0",
    GIT_TERMINAL_PROMPT: "0"});
  const result = spawnSync("/usr/bin/timeout", ["--signal=TERM", "--kill-after=5s", seconds + "s", sourceGit, ...args],
    {cwd: repo, env: sourceEnv, encoding: "utf8", maxBuffer: 4 << 20, timeout: timeoutMs + 7_000});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error("Bounded pinned HIPRTC source preparation failed", {cause: result.error});
  }
  return result.stdout || "";
}

/** The source transport is bare; HIPRTC admission requires files in a clean ordinary checkout of the pinned revision.
 * @param {number} deadlineMs @param {string} bareSource @param {string} checkout @param {string} revision */
function checkoutPinnedSource(deadlineMs, bareSource, checkout, revision) {
  if (runSourceGit(deadlineMs, ["-C", bareSource, "rev-parse", "--verify", revision + "^{commit}"]).trim() !== revision) {
    throw new Error("The selected ROCm CLR bare source does not contain the pinned HIPRTC revision");
  }
  runSourceGit(deadlineMs, ["clone", "--shared", "--no-checkout", "--", bareSource, checkout]);
  runSourceGit(deadlineMs, ["-C", checkout, "checkout", "--detach", revision]);
  if (runSourceGit(deadlineMs, ["-C", checkout, "rev-parse", "HEAD"]).trim() !== revision) {
    throw new Error("Windows HIPRTC checkout differs from its pinned ROCm CLR source revision");
  }
}

/** Build the Windows ROCm device resources and the static Windows COMGR archive.
 * @param {number} deadlineMs @param {string} runnerTemp @param {string} suffix
 * @param {{rocmSource: string, linuxSdkPath: string, hipInclude: string, descriptor: string, llvmOutput: string}} input */
function buildComgr(deadlineMs, runnerTemp, suffix, input) {
  const resourcesRoot = path.join(runnerTemp, "native-windows-rocm-resources-" + suffix),
    configFile = path.join(runnerTemp, "native-windows-rocm-resources-config-" + suffix + ".json"),
    config = ciRocm.rocmResourcesConfig({source: input.rocmSource, llvmSdk: input.linuxSdkPath, hipIncludeRoot: input.hipInclude},
      {cmake: "/usr/bin/cmake", cc: intelBin + "/icx", cxx: intelBin + "/icpx"}, "win32");
  fs.writeFileSync(configFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  runBounded(deadlineMs, "scripts/build-native-rocm-resources.js", [configFile, resourcesRoot], 12 * minute);
  const resourceResult = path.join(resourcesRoot, "result.json"), comgrRoot = path.join(runnerTemp, "native-windows-comgr-" + suffix);
  runBounded(deadlineMs, "scripts/build-native-windows-comgr-sdk.js", ["configure", input.rocmSource + "/roots/rocmLlvm", input.descriptor,
    input.llvmOutput, input.linuxSdkPath, intelBin + "/icx", resourceResult, comgrRoot, ciJob.ciBuildJobs()], 12 * minute);
  runBounded(deadlineMs, "scripts/build-native-windows-comgr-sdk.js", ["build", comgrRoot], 12 * minute);
  return {resourcesRoot, resourceResult, comgrResult: path.join(comgrRoot, "result.json")};
}

/** Build the static Windows HIPRTC archive from a fresh checkout of the pinned CLR revision.
 * @param {number} deadlineMs @param {string} runnerTemp @param {string} suffix
 * @param {{rocmSource: string, descriptor: string, hipInclude: string, llvmAr: string}} input
 * @param {ReturnType<typeof buildComgr>} comgr */
function buildHiprtc(deadlineMs, runnerTemp, suffix, input, comgr) {
  const builder = require("../build-native-windows-hiprtc-sdk"),
    checkout = path.join(runnerTemp, "native-windows-hiprtc-source-" + suffix),
    output = path.join(runnerTemp, "native-windows-hiprtc-" + suffix),
    configFile = path.join(runnerTemp, "native-windows-hiprtc-config-" + suffix + ".json");
  if ([checkout, output, configFile].some(file => fs.existsSync(file))) {
    throw new Error("Windows HIPRTC source and output paths must be fresh for this run");
  }
  checkoutPinnedSource(deadlineMs, path.join(input.rocmSource, "roots/clr"), checkout, builder.clrRevision);
  const config = {source: checkout, descriptorFile: input.descriptor, hipInclude: input.hipInclude,
    comgrResult: comgr.comgrResult, deviceReceipt: comgr.resourceResult,
    deviceInstall: path.join(comgr.resourcesRoot, "install"), llvmAr: input.llvmAr, output};
  fs.writeFileSync(configFile, JSON.stringify(config, null, 2) + "\n", {flag: "wx", mode: 0o600});
  runBounded(deadlineMs, "scripts/build-native-windows-hiprtc-sdk.js", [configFile], 5 * minute);
  const result = path.join(output, "result.json"), receipt = readBuildJson(result);
  if (receipt.role !== "windows-hiprtc-static-host" || receipt.sourceRevision !== builder.clrRevision ||
      receipt.buildOnly !== true || receipt.compiled !== true || receipt.executed !== false ||
      receipt.publicationQualified !== false) {
    throw new Error("Windows HIPRTC source builder did not return its expected build-only receipt");
  }
  return result;
}

/** Download the locked Windows CUDA Runtime archive over HTTPS and admit it by exact size and SHA-256.
 * @param {number} deadlineMs @param {string} runnerTemp @param {string} suffix */
function prepareCudart(deadlineMs, runnerTemp, suffix) {
  const cudart = require("../prepare-native-cudart-windows"), catalog = cudart.windowsCudartCatalog(),
    archive = path.join(runnerTemp, catalog.archive.fileName), root = path.join(runnerTemp, "native-windows-cudart-" + suffix),
    timeoutMs = boundedTimeout(deadlineMs, 5 * minute),
    curl = spawnSync("/usr/bin/curl", ["--fail", "--show-error", "--location", "--proto", "=https",
      "--proto-redir", "=https", "--max-redirs", "5", "--retry", "3", "--connect-timeout", "15", "--max-time",
      String(Math.max(1, Math.floor(timeoutMs / 1000))), "--max-filesize", String(catalog.archive.bytes), "--output", archive,
      catalog.archive.url], {cwd: runnerTemp, env: process.env, stdio: "inherit", timeout: timeoutMs + 5_000});
  if (curl.error || curl.signal || curl.status !== 0 || fs.statSync(archive).size !== catalog.archive.bytes ||
      fileHash(archive) !== catalog.archive.sha256) {
    throw new Error("Pinned Windows CUDA Runtime archive download failed size/SHA-256 admission", {cause: curl.error});
  }
  cudart.prepareWindowsCudart(archive, root);
  return cudart.windowsCudartInputs(root);
}

/** Restore, admit and build every producer, then export the relocatable static Windows host component.
 * Mutates process.env with the derived job environment that the shared CI helpers and child scripts read. */
function buildHostComponent() {
  const job = admitHostJob(process.env);
  Object.assign(process.env, job.environment);
  const paths = ciSources.ciPaths(), suffix = job.suffix, runnerTemp = paths.runnerTemp,
    context = {suffix, runnerTemp, sdkName: paths.sdkName, deadlineMs: job.deadlines.jobEpoch * 1000},
    deadlineMs = context.deadlineMs,
    restored = restoreSameRunInputs(context, process.env, job.pins, job.names, paths.sourceName),
    inputs = verifySameRunInputs(paths, job.pins, suffix, restored),
    descriptor = writeDescriptorInput(runnerTemp, suffix, restored.hermesRoot, job.pins.HERMES_HANDOFF, inputs.toolInputs),
    windowsState = restoreWindowsLlvm(job, incoming(process.env, job.names.windows_state_archive), descriptor),
    comgr = buildComgr(deadlineMs, runnerTemp, suffix, {rocmSource: inputs.rocm.source, linuxSdkPath: inputs.linuxSdkPath,
      hipInclude: inputs.hipHeaders.include, descriptor: windowsState.descriptor, llvmOutput: windowsState.output}),
    hiprtcResult = buildHiprtc(deadlineMs, runnerTemp, suffix, {rocmSource: inputs.rocm.source, descriptor: windowsState.descriptor,
      hipInclude: inputs.hipHeaders.include, llvmAr: path.join(inputs.toolInputs.toolRoot, "llvm-ar")}, comgr),
    cudartInputs = prepareCudart(deadlineMs, runnerTemp, suffix),
    amdRoot = path.join(runnerTemp, "native-windows-amdhip-" + suffix);
  runBounded(deadlineMs, "scripts/prepare-native-amdhip-windows.js", [inputs.rocm.source, restored.toolsBundle,
    job.pins.LINUX_TOOLS_HANDOFF, amdRoot], 16 * minute);
  const amdhip = require("../prepare-native-amdhip-windows").windowsAmdhipInputs(amdRoot),
    acppRoot = path.join(runnerTemp, "native-windows-acpp-" + suffix), dpcpp = inputs.dpcpp;
  runBounded(deadlineMs, "scripts/build-native-windows-acpp-sdk.js", [inputs.acpp.preparedCheckout, windowsState.descriptor,
    windowsState.output, path.join(inputs.linuxSdkPath, "bin"), dpcpp.cudaRoot, cudartInputs.cudartStatic,
    path.join(dpcpp.cudaRoot, "nvvm/libdevice"), inputs.hipHeaders.root, amdhip.amdhip64,
    path.join(comgr.resourcesRoot, "install/amdgcn/bitcode"), comgr.resourceResult, acppRoot], 24 * minute);
  require("../native-windows-host-component").exportWindowsHostComponent({llvmOutput: windowsState.output,
    comgrResult: comgr.comgrResult, acppResult: path.join(acppRoot, "result.json"), hiprtcResult,
    output: componentRoot(suffix)});
}

/** Reserved final component root of this CI job. @param {string} suffix */
function componentRoot(suffix) {return path.join("/tmp", "native-windows-host-component-" + suffix);}

/** Every archive member the exact component closure must contain, directories included.
 * @param {string} root @param {Record<string, string>} files */
function expectedMembers(root, files) {
  const name = path.basename(root), expected = new Set([name]);
  for (const file of Object.keys(files)) {
    const relative = path.relative(root, file).split(path.sep).join("/");
    expected.add(name + "/" + relative);
    let parent = path.posix.dirname(relative);
    while (parent !== ".") {
      expected.add(name + "/" + parent);
      parent = path.posix.dirname(parent);
    }
  }
  return [...expected].sort();
}

/** Re-admit, pack and recheck the exact build-only component as the single archive-only artifact,
 * then write bundle_sha256, manifest_sha256 and artifact_name to GITHUB_OUTPUT.
 * @param {{suffix: string, runnerTemp: string, jobDeadlineEpoch: number, outputFile: string}} input @param {number} [nowMs] */
function sealHostComponent(input, nowMs = Date.now()) {
  const {windowsHostComponentInputs} = require("../native-windows-host-component"),
    root = componentRoot(input.suffix), name = path.basename(root), runnerTemp = fs.realpathSync(input.runnerTemp),
    archive = path.join(runnerTemp, name + ".tar"), receipt = path.join(root, "component.json"),
    deadlineMs = Math.min(nowMs + 5 * minute, input.jobDeadlineEpoch * 1000 - 30_000);
  if (!validCiSuffix(input.suffix) || !Number.isSafeInteger(deadlineMs) || deadlineMs - Date.now() < 2_000 ||
      archive.startsWith("/tmp/") || !path.isAbsolute(input.outputFile)) {
    throw new Error("Windows host-component seal paths or inherited job deadline are invalid");
  }
  const operationTimeout = () => boundedTimeout(deadlineMs, Number.MAX_SAFE_INTEGER, 0);
  const before = windowsHostComponentInputs(root), {value: metadata, sha256: manifestSha256} = readBuildJsonSnapshot(receipt);
  if (fs.existsSync(archive) || before.files[receipt] !== manifestSha256 || metadata.buildOnly !== true || metadata.publicationQualified !== false) {
    throw new Error("Windows host component differs from its fresh build-only manifest before sealing");
  }
  const packed = ciTar.packState(archive, path.dirname(root), [name], operationTimeout()),
    listed = ciTar.listState(archive, [name], operationTimeout()),
    actual = listed.map(member => member.endsWith("/") ? member.slice(0, -1) : member).sort(),
    after = windowsHostComponentInputs(root), expandedBytes = ciTar.tarExpandedBytes(archive, listed.length, operationTimeout()),
    bundleSha256 = fileHash(archive);
  if (!packed.bytes || actual.join("\0") !== expectedMembers(root, before.files).join("\0") || expandedBytes < 1 ||
      expandedBytes > 1024 ** 3 || !isDeepStrictEqual(after.files, before.files) || !digest.test(bundleSha256) ||
      fileHash(receipt) !== manifestSha256) {
    throw new Error("Final Windows host-component tar is incomplete, changed, or outside its archive-only closure");
  }
  operationTimeout();
  fs.appendFileSync(input.outputFile, `bundle_sha256=${bundleSha256}\nmanifest_sha256=${manifestSha256}\nartifact_name=${name}.tar\n`);
  return {archive, bundleSha256, manifestSha256, artifactName: name + ".tar"};
}

/** Seal command: derive the job deadline from the admitted job state, then seal. @param {NodeJS.ProcessEnv} env */
function sealFromEnvironment(env) {
  const job = admitHostJob(env);
  return sealHostComponent({suffix: job.suffix, runnerTemp: job.runnerTemp, jobDeadlineEpoch: job.deadlines.jobEpoch,
    outputFile: env["GITHUB_OUTPUT"] || ""});
}

module.exports = {inputPins, externalPins, artifactNames, jobDeadlines, jobEnvironment, validatePins,
  pinnedFile, pinnedManifest, admitHostJob, restore, checkoutPinnedSource, buildHostComponent, sealHostComponent, sealFromEnvironment};

if (require.main === module) {
  try {
    const commands = {pins: () => validatePins(process.env), build: buildHostComponent, seal: () => sealFromEnvironment(process.env)},
      command = process.argv[2] || "";
    if (process.argv.length !== 3 || !Object.hasOwn(commands, command)) {throw new Error("Usage: windows-host.js pins | build | seal");}
    commands[/** @type {keyof typeof commands} */ (command)]();
  } catch (error) {
    console.error("[native-ci] " + (error instanceof Error ? error.message : "Windows host component failed"));
    process.exitCode = 1;
  }
}
