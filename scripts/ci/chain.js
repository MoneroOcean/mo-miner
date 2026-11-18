"use strict";

// Runs dependent build phases inside one hosted job. Each phase starts from the state a fresh runner
// had in the former one-job-per-phase pipeline: the predecessor's state archives move into incoming/
// and every run-scoped root a phase extracted or created under /tmp or RUNNER_TEMP is removed. The
// unchanged phase scripts therefore keep their fresh-root, pinned-archive and receipt admission.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");

const repo = path.resolve(__dirname, "../.."), phaseBudgetSeconds = 55 * 60, minimumPhaseSeconds = 2 * 60,
  maxReportBytes = 1 << 20;

/** @typedef {{suffix: string, runnerTemp: string, incoming: string, reports: string, deadlineEpoch: number,
 * runDeadlineEpoch: number, baseline: Set<string>, environment: NodeJS.ProcessEnv, step: number}} ChainContext */

/** List run-scoped entries; every CI root embeds the run-attempt suffix.
 * @param {string[]} roots @param {string} suffix */
function runScopedEntries(roots, suffix) {
  return new Set(roots.flatMap(root => fs.readdirSync(root).filter(name => name.includes(suffix)).map(name => path.join(root, name))));
}

/** Validate the job identity and deadline once and record pre-existing run-scoped entries.
 * @param {NodeJS.ProcessEnv} [environment] @returns {ChainContext} */
function chainContext(environment = process.env) {
  const run = environment["GITHUB_RUN_ID"] || "", attempt = environment["GITHUB_RUN_ATTEMPT"] || "",
    runnerTemp = environment["RUNNER_TEMP"] || "", deadline = environment["MOM_CI_DEADLINE_EPOCH"] || "";
  if (!/^[0-9]{1,20}$/.test(run) || !/^[1-9][0-9]{0,5}$/.test(attempt) || !path.isAbsolute(runnerTemp) ||
      fs.realpathSync(runnerTemp) !== runnerTemp || !/^[0-9]{10}$/.test(deadline) || Number(deadline) * 1000 <= Date.now()) {
    throw new Error("Chained CI job needs its run identity, a canonical RUNNER_TEMP and a future MOM_CI_DEADLINE_EPOCH");
  }
  const suffix = run + "-" + attempt, incoming = path.join(runnerTemp, "incoming"), reports = path.join(runnerTemp, "chain-reports");
  fs.mkdirSync(incoming, {recursive: true, mode: 0o700});
  fs.mkdirSync(reports, {mode: 0o700});
  // Downstream jobs re-admit the final state against this run deadline, so it outlives the job (8 h bound).
  return {suffix, runnerTemp, incoming, reports, deadlineEpoch: Number(deadline), runDeadlineEpoch: Math.floor(Date.now() / 1000) + 8 * 60 * 60,
    baseline: runScopedEntries(["/tmp", runnerTemp], suffix), environment, step: 0};
}

/** One phase keeps the former per-job deadline (55 minutes) and never outlives the merged job.
 * @param {ChainContext} context @param {number} [nowMs] */
function phaseDeadline(context, nowMs = Date.now()) {
  const now = Math.floor(nowMs / 1000), epoch = Math.min(now + phaseBudgetSeconds, context.deadlineEpoch);
  if (epoch - now < minimumPhaseSeconds) {throw new Error("Chained CI job deadline leaves no time for the next phase");}
  return String(epoch);
}

/** Parse a workflow command file written by a phase; only single-line KEY=value records are produced.
 * @param {string} file */
function readWorkflowFile(file) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > maxReportBytes) {throw new Error("Phase workflow file is invalid: " + file);}
  /** @type {Record<string, string>} */
  const values = {};
  for (const line of fs.readFileSync(file, "utf8").split("\n").filter(Boolean)) {
    const match = /^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/.exec(line);
    if (!match || !match[1]) {throw new Error("Phase wrote a malformed workflow record to " + path.basename(file));}
    values[match[1]] = match[2] || "";
  }
  return values;
}

/** Run one phase command with fresh workflow files, bounded by the merged job deadline.
 * @param {ChainContext} context @param {string} label @param {string[]} argv @param {NodeJS.ProcessEnv} environment */
function runPhaseCommand(context, label, argv, environment) {
  context.step += 1;
  const prefix = path.join(context.reports, String(context.step).padStart(2, "0") + "-" + label),
    files = {GITHUB_OUTPUT: prefix + ".output", GITHUB_STEP_SUMMARY: prefix + ".summary", GITHUB_ENV: prefix + ".env"},
    timeoutMs = context.deadlineEpoch * 1000 - Date.now(), command = argv[0];
  if (!command || !/^[a-z0-9-]{1,48}$/.test(label) || timeoutMs < 1000) {throw new Error("Chained CI phase is malformed or out of time: " + label);}
  for (const file of Object.values(files)) {fs.writeFileSync(file, "", {flag: "wx", mode: 0o600});}
  console.log("[native-ci] Chained phase " + label + " started");
  const result = spawnSync(command, argv.slice(1), {cwd: repo, env: {...environment, ...files}, stdio: "inherit", timeout: timeoutMs});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error("Chained CI phase failed: " + label + (result.signal ? " (" + result.signal + ")" : ""), {cause: result.error});
  }
  return {outputs: readWorkflowFile(files.GITHUB_OUTPUT), exported: readWorkflowFile(files.GITHUB_ENV)};
}

/** Move produced state archives where the next phase expects its downloaded predecessor.
 * @param {ChainContext} context @param {string[]} names */
function handOff(context, names) {
  for (const name of names) {
    const source = path.join(context.runnerTemp, name), target = path.join(context.incoming, name), stat = fs.lstatSync(source);
    if (path.basename(name) !== name || !stat.isFile() || stat.isSymbolicLink() || fs.existsSync(target)) {
      throw new Error("Chained CI handoff must move one fresh regular file: " + name);
    }
    fs.renameSync(source, target);
  }
}

/** Delete consumed predecessor archives so disk use matches the former single-phase jobs.
 * @param {ChainContext} context @param {string[]} names */
function retire(context, names) {
  for (const name of names) {fs.rmSync(path.join(context.incoming, path.basename(name)));}
}

/** Remove a tree even when a producer made directories read-only. @param {string} target */
function removeTree(target) {
  try {
    fs.rmSync(target, {recursive: true, force: true});
  } catch {
    const chmod = spawnSync("/usr/bin/chmod", ["-R", "u+rwX", target], {stdio: "inherit"});
    if (chmod.error || chmod.status !== 0) {throw new Error("Cannot reset chained CI root: " + target, {cause: chmod.error});}
    fs.rmSync(target, {recursive: true, force: true});
  }
}

/** Return /tmp and RUNNER_TEMP to the fresh-runner state, keeping incoming/ and listed final outputs.
 * @param {ChainContext} context @param {string[]} keep */
function resetRunScopedRoots(context, keep = []) {
  const kept = new Set(keep.map(name => path.join(context.runnerTemp, name)));
  for (const entry of runScopedEntries(["/tmp", context.runnerTemp], context.suffix)) {
    if (!context.baseline.has(entry) && !kept.has(entry)) {removeTree(entry);}
  }
}

/** Publish the final machine outputs of the merged job. @param {NodeJS.ProcessEnv} environment @param {Record<string, string>} values */
function writeWorkflowOutputs(environment, values) {
  const file = environment["GITHUB_OUTPUT"] || "";
  if (!path.isAbsolute(file)) {throw new Error("Chained CI job needs GITHUB_OUTPUT");}
  const lines = Object.entries(values).map(([key, value]) => {
    if (!/^[a-z][a-z0-9_]*$/.test(key) || !/^[A-Za-z0-9._/-]*$/.test(value)) {throw new Error("Chained CI output is malformed: " + key);}
    return key + "=" + value;
  });
  fs.appendFileSync(file, lines.join("\n") + "\n", "utf8");
}

/** @param {...string} parts */
function nodeScript(...parts) {return [process.execPath, path.join(repo, ...parts)];}

/** @param {Record<string, string>} values @param {string} key */
function requiredOutput(values, key) {
  const value = values[key] || "";
  if (!/^[a-f0-9]{64}$/.test(value)) {throw new Error("Chained CI phase did not report " + key);}
  return value;
}

module.exports = {chainContext, phaseDeadline, readWorkflowFile, runPhaseCommand, handOff, retire, resetRunScopedRoots,
  writeWorkflowOutputs, nodeScript, requiredOutput, runScopedEntries};
