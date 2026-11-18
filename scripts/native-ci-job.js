"use strict";

// Shared helpers for native CI jobs. Work roots and same-run artifacts carry one fixed suffix instead
// of the GitHub run identity, so "Re-run failed jobs" and restored caches see the same absolute paths
// and artifact names as the first attempt. Hosted runners start empty; a root left by an earlier job
// on the same machine is rejected when the next job starts.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"), {createHash} = require("node:crypto");
const {fileHash} = require("./build-source-inputs");

/** The work-root suffix the workflow exports (`ci`); test fixtures pass a unique value to stay isolated.
 * @param {NodeJS.ProcessEnv} [environment] */
function ciSuffix(environment = process.env) {
  const suffix = environment["MOM_NATIVE_CI_SUFFIX"] || "";
  if (!validCiSuffix(suffix)) {throw new Error("Native CI needs a plain MOM_NATIVE_CI_SUFFIX");}
  return suffix;
}

/** @param {unknown} value @returns {value is string} */
function validCiSuffix(value) {return typeof value === "string" && /^[a-z0-9][a-z0-9-]{0,39}$/.test(value);}

/** CI producers and their same-run consumers share the workflow's validated build parallelism.
 * @param {NodeJS.ProcessEnv} [environment] */
function ciBuildJobs(environment = process.env) {
  const jobs = environment["MOM_BUILD_JOBS"] || "";
  if (!["2", "4"].includes(jobs)) {throw new Error("Native CI needs MOM_BUILD_JOBS set to 2 or 4");}
  return jobs;
}

/** Entries of the given directories whose name carries the CI suffix as a `-<suffix>` component.
 * @param {string[]} roots @param {string} suffix */
function ciScopedEntries(roots, suffix) {
  const pattern = new RegExp("-" + suffix + "(?:$|[-.])");
  return new Set(roots.flatMap(root => fs.readdirSync(root).filter(name => pattern.test(name)).map(name => path.join(root, name))));
}

/** Remove a work root even when a producer made directories read-only. @param {string} target */
function removeTree(target) {
  try {
    fs.rmSync(target, {recursive: true, force: true});
  } catch {
    const chmod = spawnSync("/usr/bin/chmod", ["-R", "u+rwX", target], {stdio: "inherit"});
    if (chmod.error || chmod.status !== 0) {throw new Error("Cannot reset native CI work root: " + target, {cause: chmod.error});}
    fs.rmSync(target, {recursive: true, force: true});
  }
}

const repo = path.resolve(__dirname, "..");

/** @param {string} file @param {string[]} args @param {NodeJS.ProcessEnv} env @param {number} timeout */
function runNode(file, args, env, timeout = 50 * 60 * 1000) {
  const result = spawnSync(process.execPath, [path.join(repo, file), ...args],
    {cwd: repo, env, stdio: "inherit", timeout});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error("Native CI command failed: " + file + (result.signal ? " (" + result.signal + ")" : ""), {cause: result.error});
  }
}

/** @param {string} value */
function jsonHash(value) {return createHash("sha256").update(value).digest("hex");}

/** @param {unknown} value @param {string[]} keys */
function hasExactKeys(value, keys) {
  return !!value && typeof value === "object" && !Array.isArray(value) &&
    Object.keys(value).sort().join("\0") === [...keys].sort().join("\0");
}

/** Shrink each synchronous child timeout against the one shared component deadline.
 * @param {number} deadlineMs @param {number} maximumMs @param {number} [nowMs] */
function linuxNativeComponentChildTimeout(deadlineMs, maximumMs, nowMs = Date.now()) {
  if (!Number.isSafeInteger(deadlineMs) || !Number.isSafeInteger(maximumMs) || maximumMs < 1 ||
      !Number.isSafeInteger(nowMs) || nowMs < 0) {throw new Error("Native component child deadline is malformed");}
  const timeout = Math.min(maximumMs, deadlineMs - nowMs);
  if (!Number.isSafeInteger(timeout) || timeout < 1000) {throw new Error("Native component deadline expired before the next child stage");}
  return timeout;
}

/** @param {NodeJS.ProcessEnv} environment @param {Record<string, string>} values */
function appendWorkflowOutputs(environment, values) {
  const output = environment["GITHUB_OUTPUT"], stat = output && path.isAbsolute(output) ? fs.lstatSync(output, {throwIfNoEntry: false}) : undefined,
    lines = Object.entries(values);
  if (!output || !stat?.isFile() || stat.isSymbolicLink() || lines.some(([key, value]) =>
    !/^[a-z][a-z0-9_]*$/.test(key) || /[\r\n]/.test(value))) {
    throw new Error("Linux source-JIT runtime handoff requires a regular GitHub output file and bounded fields");
  }
  fs.appendFileSync(output, lines.map(([key, value]) => key + "=" + value).join("\n") + "\n", {encoding: "utf8"});
}

/** Resolve tool paths to canonical executable files because the OpenSSL producer rejects aliases.
 * @param {string} file */
function canonicalExecutable(file) {
  const selected = fs.realpathSync(file), stat = fs.statSync(selected);
  if (!stat.isFile()) {throw new Error("Missing native CI executable: " + file);}
  fs.accessSync(selected, fs.constants.X_OK);
  return selected;
}

/** Convert the explicit job-start epoch to a bounded child timeout, optionally reserving
 * time for the remaining audit/package stages before the hosted job limit.
 * @param {string} deadlineEpoch @param {number} maximumMs @param {number} [nowMs] @param {number} [reserveMs] */
function ioOpenSslChildTimeout(deadlineEpoch, maximumMs, nowMs = Date.now(), reserveMs = 0) {
  if (!/^\d{10}$/.test(deadlineEpoch) || !Number.isSafeInteger(nowMs) || nowMs < 0 ||
      !Number.isSafeInteger(reserveMs) || reserveMs < 0) {
    throw new Error("Native I/O/OpenSSL CI needs a valid bounded job-start deadline");
  }
  const deadlineMs = Number(deadlineEpoch) * 1000, remainingMs = deadlineMs - nowMs;
  if (!Number.isSafeInteger(remainingMs) || remainingMs > 60 * 60 * 1000) {
    throw new Error("Native I/O/OpenSSL deadline exceeds the 60-minute job bound");
  }
  return linuxNativeComponentChildTimeout(deadlineMs - reserveMs, maximumMs, nowMs);
}

/** Run an SDK producer in its own process group so a bounded timeout also contains compiler children.
 * @param {string} script @param {string[]} args @param {NodeJS.ProcessEnv} environment @param {number} timeoutMs */
function runNodeProcessGroup(script, args, environment, timeoutMs) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 45 * 60 * 1000) {throw new Error("SDK producer child timeout is invalid");}
  const timeoutPath = "/usr/bin/timeout", timeoutReal = fs.realpathSync(timeoutPath), timeoutStat = fs.statSync(timeoutReal);
  if (!timeoutStat.isFile()) {throw new Error("GNU timeout is unavailable for bounded SDK production");}
  const timeoutSha256 = fileHash(timeoutReal);
  const version = spawnSync(timeoutReal, ["--version"], {encoding: "utf8", timeout: 30000, maxBuffer: 64 * 1024});
  if (version.error || version.status !== 0 || !((version.stdout || "") + (version.stderr || "")).includes("GNU coreutils")) {
    throw new Error("Bounded SDK production requires GNU coreutils timeout");
  }
  const seconds = Math.max(1, Math.ceil(timeoutMs / 1000)), command = spawnSync(timeoutReal,
    ["--signal=TERM", "--kill-after=5s", seconds + "s", process.execPath, path.join(repo, script), ...args],
    {cwd: repo, env: environment, stdio: "inherit", timeout: timeoutMs + 10000});
  if (command.error || command.signal || command.status !== 0) {
    throw new Error("Bounded SDK producer failed: " + script + (command.status === 124 ? " (deadline)" : ""), {cause: command.error});
  }
  if (timeoutSha256 !== fileHash(timeoutPath)) {throw new Error("GNU timeout changed during SDK production");}
  return {timeoutPath: timeoutReal, timeoutSha256};
}

/** Bound one child by both its own cap and the merged job's deadline (job start + timeout - margin).
 * @param {NodeJS.ProcessEnv} environment @param {number} maximumMs */
function ciJobChildTimeout(environment, maximumMs) {
  const epoch = environment["MOM_CI_DEADLINE_EPOCH"] || "", timeoutMs = Math.min(maximumMs, Number(epoch) * 1000 - Date.now());
  if (!/^\d{10}$/.test(epoch) || !Number.isSafeInteger(timeoutMs) || timeoutMs < 60 * 1000) {
    throw new Error("Native CI job deadline is malformed or leaves no time for the next child");
  }
  return timeoutMs;
}

module.exports = {ciSuffix, validCiSuffix, ciBuildJobs, ciScopedEntries, removeTree, runNode, runNodeProcessGroup, jsonHash, hasExactKeys,
  linuxNativeComponentChildTimeout, appendWorkflowOutputs, canonicalExecutable, ioOpenSslChildTimeout, ciJobChildTimeout};
