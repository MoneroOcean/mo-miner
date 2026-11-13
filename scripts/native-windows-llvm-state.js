"use strict";

// Same-run Windows LLVM CI state transport. The archive preserves absolute paths and receipts;
// callers must reconstitute the exact descriptor's external SDK/tool inputs before restore.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto");
const {fileHash} = require("./build-source-inputs");
const ci = require("./prepare-native-ci");
const {buildPhases} = require("./build-native-windows-llvm21-sdk");

const maxArchiveBytes = 16 * 1024 ** 3, maxExpandedBytes = 64 * 1024 ** 3,
  maxDeadlineRecords = 256, receiptLimit = 64 << 20, deadlineLimit = 1024;

/** @typedef {{version:1,runAttempt:string,producerJob:string,producerDeadlineEpoch:string,runDeadlineEpoch:string,
 * archive:{sha256:string,bytes:number,members:number,expandedBytes:number},bindings:{descriptorSha256:string,
 * configureReceiptSha256:string,checkpointSha256:string|null,runDeadlineSha256:string,jobDeadlines:Array<{job:string,sha256:string}>}}} WindowsLlvmStateManifest */

/** @param {string} file @param {string} label @param {number} maximum */
function pinnedFile(file, label, maximum) {
  const before = fs.lstatSync(file, {bigint: true, throwIfNoEntry: false});
  if (!path.isAbsolute(file) || /[\r\n\0]/.test(file) || !before?.isFile() || before.isSymbolicLink() ||
      before.nlink !== 1n || before.size < 1n || before.size > BigInt(maximum) ||
      path.resolve(fs.realpathSync(file)) !== path.resolve(file)) {
    throw new Error(label + " must be a canonical singly-linked regular file within its size bound");
  }
  const sha256 = fileHash(file), after = fs.lstatSync(file, {bigint: true});
  if (before.dev !== after.dev || before.ino !== after.ino || before.nlink !== after.nlink ||
      before.size !== after.size || before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs) {
    throw new Error(label + " changed while it was hashed");
  }
  return {sha256, bytes: Number(after.size)};
}

/** @param {string} directory */
function availableBytes(directory) {
  const stat = fs.statfsSync(directory, {bigint: true});
  if (stat.bsize <= 0n) {throw new Error("Cannot determine Windows LLVM state filesystem capacity");}
  const bytes = stat.bavail * stat.bsize;
  return bytes > BigInt(Number.MAX_SAFE_INTEGER) ? Number.MAX_SAFE_INTEGER : Number(bytes);
}

/** Bound the exact state root before creating a potentially multi-gigabyte tar. @param {string} root @param {number} deadline */
function stateTreeSize(root, deadline) {
  if (fs.realpathSync(root) !== root) {throw new Error("Windows LLVM state root is not canonical");}
  let entries = 0, payloadBytes = 0;
  /** @param {string} directory @param {number} depth @param {string} parentRelative */
  const visit = (directory, depth, parentRelative) => {
    if (depth > 128) {throw new Error("Windows LLVM state tree exceeds its depth bound");}
    for (const name of fs.readdirSync(directory)) {
      const relative = parentRelative ? parentRelative + "/" + name : name;
      if (++entries > 500000 || Date.now() >= deadline || /[\\/\r\n\0]/.test(name) || !ci.safeRelative(relative)) {
        throw new Error("Windows LLVM state tree exceeds its entry or time bound");
      }
      const item = path.join(directory, name), stat = fs.lstatSync(item, {bigint: true});
      if (stat.isSymbolicLink()) {
        const targetText = fs.readlinkSync(item);
        if (!targetText || targetText.length > 4096 || /[\r\n\0]/.test(targetText) || path.isAbsolute(targetText)) {
          throw new Error("Windows LLVM state tree has an invalid symlink target");
        }
        const resolved = fs.realpathSync(item), relativeTarget = path.relative(root, resolved);
        if (path.isAbsolute(relativeTarget) || relativeTarget === ".." || relativeTarget.startsWith(".." + path.sep)) {
          throw new Error("Windows LLVM state tree symlink escapes its canonical root");
        }
        payloadBytes += Buffer.byteLength(targetText);
        if (!Number.isSafeInteger(payloadBytes) || payloadBytes > maxExpandedBytes) {
          throw new Error("Windows LLVM state symlink targets exceed their expanded-size bound");
        }
        continue;
      }
      if (stat.isDirectory()) {visit(item, depth + 1, relative); continue;}
      if (!stat.isFile() || stat.nlink !== 1n || stat.size < 0n || stat.size > BigInt(maxArchiveBytes)) {
        throw new Error("Windows LLVM state tree contains a special, aliased or oversized file");
      }
      payloadBytes += Number(stat.size);
      if (!Number.isSafeInteger(payloadBytes) || payloadBytes > maxExpandedBytes) {
        throw new Error("Windows LLVM state tree exceeds its expanded-size bound");
      }
    }
  };
  visit(root, 0, "");
  const archiveUpperBound = payloadBytes + (entries + 1) * 8192;
  if (!Number.isSafeInteger(archiveUpperBound) || archiveUpperBound > maxArchiveBytes) {
    throw new Error("Windows LLVM state tar would exceed its bounded archive size");
  }
  return {entries, payloadBytes, archiveUpperBound};
}

/** @param {ReturnType<typeof ci.windowsNativeLlvmCiStatePaths>} state */
function stateBindings(state) {
  const descriptor = pinnedFile(state.descriptor, "Windows LLVM descriptor", 4 << 20),
    configure = pinnedFile(state.result, "Windows LLVM configure receipt", receiptLimit),
    checkpointFile = path.join(state.output, "build-state.json"), checkpointStat = fs.lstatSync(checkpointFile, {throwIfNoEntry: false}),
    checkpoint = checkpointStat ? pinnedFile(checkpointFile, "Windows LLVM checkpoint", receiptLimit) : null,
    runDeadline = pinnedFile(state.runDeadlineRecord || "", "Windows LLVM run deadline", deadlineLimit);
  const entries = fs.readdirSync(state.root).filter(name => /^ci-job-deadline-[A-Za-z0-9_-]{1,64}\.json$/.test(name)).sort();
  if (!entries.length || entries.length > maxDeadlineRecords || !entries.includes("ci-job-deadline-" + state.jobId + ".json")) {
    throw new Error("Windows LLVM state must retain a bounded set of same-run job deadline records");
  }
  const jobDeadlines = entries.map(name => ({job: name.slice("ci-job-deadline-".length, -".json".length),
    sha256: pinnedFile(path.join(state.root, name), "Windows LLVM job deadline", deadlineLimit).sha256}));
  return {descriptorSha256: descriptor.sha256, configureReceiptSha256: configure.sha256,
    checkpointSha256: checkpoint?.sha256 || null, runDeadlineSha256: runDeadline.sha256, jobDeadlines};
}

/** @param {unknown} value */
function manifestDigest(value) {
  return createHash("sha256").update(JSON.stringify(value) + "\n").digest("hex");
}

/** @param {ReturnType<typeof ci.windowsNativeLlvmCiStatePaths>} state @param {number} capMs */
function operationTimeout(state, capMs) {
  return Math.min(capMs, ci.windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "build", Date.now(), state.runDeadlineEpoch).timeoutMs);
}

/** @param {unknown} value @param {string} expectedSha256 @returns {WindowsLlvmStateManifest} */
function validateManifest(value, expectedSha256) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Windows LLVM state manifest is invalid or differs from its external SHA-256 pin");
  }
  const row = /** @type {WindowsLlvmStateManifest} */ (value);
  if (Object.keys(row).sort().join(",") !== "archive,bindings,producerDeadlineEpoch,producerJob,runAttempt,runDeadlineEpoch,version" ||
      row.version !== 1 || !/^[0-9]{1,20}-[1-9][0-9]{0,5}$/.test(row.runAttempt || "") ||
      !/^[A-Za-z0-9_-]{1,64}$/.test(row.producerJob || "") ||
      !/^\d{10}$/.test(row.producerDeadlineEpoch || "") || !/^\d{10}$/.test(row.runDeadlineEpoch || "") ||
      !row.archive || typeof row.archive !== "object" || Array.isArray(row.archive) ||
      Object.keys(row.archive).sort().join(",") !== "bytes,expandedBytes,members,sha256" ||
      !/^[a-f0-9]{64}$/.test(row.archive.sha256 || "") ||
      ![row.archive.bytes, row.archive.expandedBytes, row.archive.members].every(Number.isSafeInteger) ||
      row.archive.bytes < 1 || row.archive.bytes > maxArchiveBytes || row.archive.expandedBytes < 1 ||
      row.archive.expandedBytes > maxExpandedBytes || row.archive.members < 1 || row.archive.members > 500000 ||
      !row.bindings || typeof row.bindings !== "object" || Array.isArray(row.bindings) ||
      Object.keys(row.bindings).sort().join(",") !== "checkpointSha256,configureReceiptSha256,descriptorSha256,jobDeadlines,runDeadlineSha256" ||
      row.bindings.checkpointSha256 !== null && !/^[a-f0-9]{64}$/.test(row.bindings.checkpointSha256 || "") ||
      ![row.bindings.configureReceiptSha256, row.bindings.descriptorSha256, row.bindings.runDeadlineSha256].every(hash => /^[a-f0-9]{64}$/.test(hash || "")) ||
      !Array.isArray(row.bindings.jobDeadlines) || !row.bindings.jobDeadlines.length ||
      row.bindings.jobDeadlines.length > maxDeadlineRecords || manifestDigest(row) !== expectedSha256) {
    throw new Error("Windows LLVM state manifest is invalid or differs from its external SHA-256 pin");
  }
  let prior = "";
  for (const job of row.bindings.jobDeadlines) {
    if (!job || Object.keys(job).sort().join(",") !== "job,sha256" || !/^[A-Za-z0-9_-]{1,64}$/.test(job.job || "") ||
        !/^[a-f0-9]{64}$/.test(job.sha256 || "") || job.job <= prior) {
      throw new Error("Windows LLVM state manifest has invalid job deadline bindings");
    }
    prior = job.job;
  }
  if (!row.bindings.jobDeadlines.some(job => job.job === row.producerJob)) {
    throw new Error("Windows LLVM state manifest omitted its producer job deadline");
  }
  return row;
}

/** @param {NodeJS.ProcessEnv} environment @param {number} nowMs */
function currentState(environment, nowMs) {
  const state = ci.windowsNativeLlvmCiStatePaths(environment);
  if (!state.runDeadlineEpoch || !state.jobId || !state.runDeadlineRecord) {
    throw new Error("Windows LLVM state transport requires the immutable cross-job run and job contract");
  }
  ci.windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "build", nowMs, state.runDeadlineEpoch);
  return state;
}

/**
 * Pack the complete canonical run/attempt root. The returned manifest is small workflow metadata;
 * consumers must pass its separate output SHA and the archive's separate output SHA back on restore.
 * @param {string} archive @param {NodeJS.ProcessEnv} [environment] @param {number} [nowMs]
 */
function packWindowsNativeLlvmState(archive, environment = process.env, nowMs = Date.now()) {
  const state = currentState(environment, nowMs), phase = buildPhases[0];
  if (!phase) {throw new Error("Windows LLVM state transport has no maintained build phase");}
  ci.windowsNativeLlvmCiPreflight(["build", phase.name], environment, "linux", "x64", nowMs);
  ci.windowsNativeLlvmCiDeadlineRecord(state);
  const before = stateBindings(state), rootName = path.basename(state.root);
  if (path.dirname(state.root) !== "/tmp" || fs.realpathSync("/tmp") !== "/tmp" || rootName !== "mom-native-windows-llvm21-" + state.suffix) {
    throw new Error("Windows LLVM state transport requires the exact canonical /tmp run-attempt root");
  }
  const archiveTarget = path.resolve(archive);
  if (archiveTarget === state.root || archiveTarget.startsWith(state.root + path.sep)) {
    throw new Error("Windows LLVM state archive must be outside the preserved run-attempt root");
  }
  ci.linuxNativeSdkDiskPreflight(0, availableBytes("/tmp"), availableBytes(state.runnerTemp));
  const sourceSize = stateTreeSize(state.root, Date.now() + operationTimeout(state, 30 * 60 * 1000)),
    archiveParent = path.dirname(archive), neededArchiveSpace = sourceSize.archiveUpperBound + 4 * 1024 ** 3;
  if (!path.isAbsolute(archive) || /[\r\n\0]/.test(archive) || fs.realpathSync(archiveParent) !== archiveParent ||
      availableBytes(archiveParent) < neededArchiveSpace) {
    throw new Error("Insufficient canonical runner-temp space for the bounded Windows LLVM state archive and reserve");
  }
  const packed = ci.packState(archive, "/tmp", [rootName], operationTimeout(state, 30 * 60 * 1000)),
    archivePin = pinnedFile(archive, "Windows LLVM state archive", maxArchiveBytes),
    members = ci.listState(archive, [rootName], operationTimeout(state, 5 * 60 * 1000)),
    expandedBytes = ci.tarExpandedBytes(archive, members.length, operationTimeout(state, 15 * 60 * 1000));
  if (packed.bytes !== archivePin.bytes || packed.bytes > sourceSize.archiveUpperBound || expandedBytes > maxExpandedBytes ||
      expandedBytes > sourceSize.payloadBytes || !Number.isSafeInteger(expandedBytes) ||
      !sameBindings(before, stateBindings(state))) {
    throw new Error("Windows LLVM state changed or exceeded bounds while it was packed");
  }
  ci.linuxNativeSdkDiskPreflight(0, availableBytes("/tmp"), availableBytes(state.runnerTemp));
  ci.windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "outputs", Date.now(), state.runDeadlineEpoch);
  const manifest = {version: 1, runAttempt: state.suffix, producerJob: state.jobId,
    producerDeadlineEpoch: state.deadlineEpoch, runDeadlineEpoch: state.runDeadlineEpoch,
    archive: {sha256: archivePin.sha256, bytes: archivePin.bytes, members: members.length, expandedBytes}, bindings: before};
  return {manifest, manifestSha256: manifestDigest(manifest)};
}

/** @param {ReturnType<typeof stateBindings>} left @param {ReturnType<typeof stateBindings>} right */
function sameBindings(left, right) {return JSON.stringify(left) === JSON.stringify(right);}

/** Restore only the fresh canonical same-run root; the public builder remains the checkpoint authority.
 * @param {string} archive @param {string} archiveSha256 @param {unknown} manifest @param {string} manifestSha256
 * @param {NodeJS.ProcessEnv} [environment] @param {number} [nowMs] */
function restoreWindowsNativeLlvmState(archive, archiveSha256, manifest, manifestSha256, environment = process.env, nowMs = Date.now()) {
  const state = currentState(environment, nowMs), pin = validateManifest(manifest, manifestSha256);
  if (!/^[a-f0-9]{64}$/.test(archiveSha256 || "") || archiveSha256 !== pin.archive.sha256 || pin.runAttempt !== state.suffix ||
      pin.runDeadlineEpoch !== state.runDeadlineEpoch || !path.isAbsolute(archive) || path.resolve(archive) !== archive) {
    throw new Error("Windows LLVM restore does not match the external same-run archive and manifest pins");
  }
  const archivePin = pinnedFile(archive, "Windows LLVM incoming state archive", maxArchiveBytes);
  if (archivePin.sha256 !== archiveSha256 || archivePin.bytes !== pin.archive.bytes) {
    throw new Error("Windows LLVM incoming archive differs from its external SHA-256 and size pins");
  }
  const rootName = path.basename(state.root), members = ci.listState(archive, [rootName], operationTimeout(state, 5 * 60 * 1000)),
    expandedBytes = ci.tarExpandedBytes(archive, members.length, operationTimeout(state, 15 * 60 * 1000));
  if (members.length !== pin.archive.members || expandedBytes !== pin.archive.expandedBytes || expandedBytes > maxExpandedBytes) {
    throw new Error("Windows LLVM incoming archive differs from its admitted member or expanded-size bounds");
  }
  ci.linuxNativeSdkDiskPreflight(expandedBytes, availableBytes("/tmp"), availableBytes(state.runnerTemp));
  if (fs.existsSync(state.root)) {throw new Error("Windows LLVM state restore requires a fresh canonical run-attempt destination");}
  ci.extractState(archive, "/tmp", [rootName], [rootName], operationTimeout(state, 30 * 60 * 1000));
  const sourceEnvironment = {...environment, GITHUB_JOB: pin.producerJob,
      MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: pin.producerDeadlineEpoch},
    sourceState = ci.windowsNativeLlvmCiStatePaths(sourceEnvironment, "linux", "x64");
  ci.windowsNativeLlvmCiDeadlineRecord(sourceState);
  const restored = stateBindings(sourceState);
  if (sourceState.suffix !== state.suffix || pin.producerJob !== sourceState.jobId || pin.bindings.descriptorSha256 !== restored.descriptorSha256 ||
      pin.bindings.configureReceiptSha256 !== restored.configureReceiptSha256 || pin.bindings.checkpointSha256 !== restored.checkpointSha256 ||
      pin.bindings.runDeadlineSha256 !== restored.runDeadlineSha256 || !sameBindings(pin.bindings, restored)) {
    throw new Error("Windows LLVM restored configure, checkpoint or deadline records differ from their external bindings");
  }
  ci.windowsNativeLlvmCiDeadlineRecord(state, true);
  const phase = buildPhases[0];
  if (!phase) {throw new Error("Windows LLVM state transport has no maintained build phase");}
  ci.windowsNativeLlvmCiPreflight(["build", phase.name], environment, "linux", "x64", nowMs);
  const finalBindings = stateBindings(sourceState), priorJobs = new Set(pin.bindings.jobDeadlines.map(row => row.job)),
    finalPriorBindings = {...finalBindings, jobDeadlines: finalBindings.jobDeadlines.filter(row => priorJobs.has(row.job))};
  if (archivePin.sha256 !== fileHash(archive) || !sameBindings(pin.bindings, finalPriorBindings) ||
      !finalBindings.jobDeadlines.some(row => row.job === state.jobId)) {
    throw new Error("Windows LLVM state or archive changed during restoration");
  }
  ci.windowsNativeLlvmCiDeadlineBudget(state.deadlineEpoch, "outputs", Date.now(), state.runDeadlineEpoch);
  return {state, archiveSha256, manifestSha256, bindings: restored};
}

module.exports = {packWindowsNativeLlvmState, restoreWindowsNativeLlvmState};
