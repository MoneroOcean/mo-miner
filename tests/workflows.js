"use strict";

// Structural invariants for every workflow and local action; spelling is free to change.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {describe, test} = require("node:test");
const {parseWorkflowYaml} = require("./common/workflow-yaml");

const repo = path.resolve(__dirname, "..");
const workflowRoot = path.join(repo, ".github", "workflows");
const suffixExpression = /\$\{\{ env\.MOM_NATIVE_CI_SUFFIX \}\}/g;

/** @typedef {import("./common/workflow-yaml").YamlValue} YamlValue */
/** @typedef {{[key: string]: YamlValue}} YamlMap */

/** @param {YamlValue | undefined} value @returns {YamlMap} */
function map(value) {return value && typeof value === "object" && !Array.isArray(value) ? value : {};}
/** @param {YamlValue | undefined} value @returns {YamlValue[]} */
function list(value) {return Array.isArray(value) ? value : value === undefined || value === "" ? [] : [value];}
/** @param {YamlValue | undefined} value */
function text(value) {return typeof value === "string" ? value : "";}

/** @param {string} file */
function load(file) {return map(parseWorkflowYaml(fs.readFileSync(file, "utf8")));}

const workflows = Object.fromEntries(fs.readdirSync(workflowRoot).filter(name => /\.ya?ml$/.test(name)).sort()
  .map(name => [name, load(path.join(workflowRoot, name))]));
/** @type {Array<[string, YamlMap]>} */
const actions = fs.readdirSync(path.join(repo, ".github", "actions")).map(name =>
  /** @type {[string, YamlMap]} */ ([name, load(path.join(repo, ".github", "actions", name, "action.yml"))]));

/** Every step in every job and local composite action.
 * @returns {Array<{where: string, step: YamlMap, job: YamlMap}>} */
function allSteps() {
  const steps = Object.entries(workflows).flatMap(([file, workflow]) => Object.entries(map(workflow["jobs"]))
    .flatMap(([job, value]) => list(map(value)["steps"]).map(step => ({where: file + ":" + job, step: map(step), job: map(value)}))));
  return [...steps, ...actions.flatMap(([name, action]) => list(map(action["runs"])["steps"]).map(step => ({where: name, step: map(step), job: {}})))];
}

/** Substitute ${{ matrix.<key> }} with every value of the job's matrix. @param {string} value @param {YamlMap} job
 * @returns {string[]} */
function matrixValues(value, job) {
  const match = /\$\{\{ matrix\.([A-Za-z0-9_-]+) \}\}/.exec(value);
  if (!match) {return [value];}
  const matrix = map(map(job["strategy"])["matrix"]), key = match[1] || "",
    values = [...list(matrix[key]), ...list(matrix["include"]).map(row => map(row)[key])].map(text).filter(Boolean);
  return values.flatMap(item => matrixValues(value.replace(match[0], item), job));
}

/** Callers of the native job template with their inputs. */
function nativeCalls() {
  return Object.entries(workflows).flatMap(([file, workflow]) => Object.entries(map(workflow["jobs"]))
    .filter(([, job]) => text(map(job)["uses"]) === "./.github/workflows/native-job.yml")
    .map(([id, job]) => ({where: file + ":" + id, with: map(map(job)["with"])})));
}

/** @param {string} value */
function words(value) {return value.split(/\s+/).filter(Boolean);}

/** Artifact names produced anywhere, with "@" for the run suffix. */
function producedArtifacts() {
  const produced = new Set();
  for (const call of nativeCalls()) {
    for (const row of words(text(call.with["upload"]))) {produced.add(row.includes("=") ? row.split("=")[0] : path.posix.basename(row));}
  }
  for (const {step, job} of allSteps().filter(row => text(row.step["uses"]).startsWith("actions/upload-artifact@"))) {
    const options = map(step["with"]), raw = text(options["archive"]) === "false";
    for (const value of matrixValues(text(raw ? options["path"] : options["name"]), job)) {
      const name = value.replace(suffixExpression, "@").replace(/\$\{\{ env\.NATIVE_NAME \}\}/g, "native-linux");
      if (name) {produced.add(raw ? path.posix.basename(name) : name);}
    }
  }
  return produced;
}

/** Expand "{a,b}" download patterns into names. @param {string} pattern @returns {string[]} */
function expand(pattern) {
  const match = /^(.*)\{([^{}]*)\}(.*)$/.exec(pattern);
  return match ? (match[2] || "").split(",").flatMap(item => expand((match[1] || "") + item + (match[3] || ""))) : [pattern];
}

/** Concrete artifact names (or name templates) for a download step. @param {string} name @param {string} pattern */
function downloadedNames(name, pattern) {return name ? [name] : expand(pattern);}

describe("Workflow structure", () => {
  test("every workflow and action parses and every external action is pinned to a full commit", () => {
    assert.ok(Object.keys(workflows).length >= 5);
    for (const {where, step} of allSteps()) {
      const uses = text(step["uses"]);
      if (uses && !uses.startsWith("./")) {assert.match(uses, /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)?@[0-9a-f]{40}$/, where + " " + uses);}
    }
    for (const [file, workflow] of Object.entries(workflows)) {
      for (const [id, job] of Object.entries(map(workflow["jobs"]))) {
        const uses = text(map(job)["uses"]);
        if (uses) {assert.match(uses, /^\.\/\.github\/workflows\/[a-z0-9-]+\.yml$/, file + ":" + id);}
      }
    }
  });

  test("permissions are read-only except the tag publish job", () => {
    for (const [file, workflow] of Object.entries(workflows)) {
      const top = map(workflow["permissions"]);
      assert.ok(Object.keys(top).length, file + " declares workflow permissions");
      for (const [scope, level] of Object.entries(top)) {assert.ok(["actions", "contents"].includes(scope) && level === "read", file + " " + scope);}
      for (const [id, job] of Object.entries(map(workflow["jobs"]))) {
        const permissions = map(map(job)["permissions"]);
        const writes = Object.entries(permissions).filter(([, level]) => level !== "read");
        if (file === "release.yml" && id === "publish") {
          assert.deepEqual(permissions, {contents: "write"});
          assert.match(text(map(job)["if"]), /startsWith\(github\.ref, 'refs\/tags\/'\)/);
        } else {
          assert.deepEqual(writes, [], file + ":" + id);
        }
      }
    }
  });

  test("every job is bounded and named, checkouts drop credentials and uploads expire", () => {
    for (const [file, workflow] of Object.entries(workflows)) {
      for (const [id, job] of Object.entries(map(workflow["jobs"]))) {
        const value = map(job), timeout = text(value["timeout-minutes"]);
        if (!value["uses"]) {
          assert.ok(timeout, file + ":" + id + " needs timeout-minutes");
          if (/^\d+$/.test(timeout)) {assert.ok(Number(timeout) <= 345, file + ":" + id + " stays under the 6 h hosted limit");}
        }
      }
    }
    for (const call of nativeCalls()) {
      const timeout = text(call.with["timeout"]) || "60";
      assert.ok(Number(timeout) >= 10 && Number(timeout) <= 345, call.where);
      // GitHub shows only the reusable job's own name; the caller id keeps each entry distinct.
      assert.equal(text(call.with["name"]), call.where.split(":")[1], call.where + " names its native job");
    }
    for (const {where, step} of allSteps()) {
      const uses = text(step["uses"]);
      if (uses.startsWith("actions/checkout@")) {assert.equal(text(map(step["with"])["persist-credentials"]), "false", where);}
      if (uses.startsWith("actions/upload-artifact@")) {
        // Artifact names are fixed per run, so a re-run attempt replaces what an earlier attempt uploaded.
        assert.equal(text(map(step["with"])["retention-days"]), "3", where);
        assert.equal(text(map(step["with"])["overwrite"]), "true", where);
      }
    }
  });

  test("every downloaded artifact is uploaded by some job", () => {
    const produced = producedArtifacts();
    for (const call of nativeCalls()) {
      for (const name of words(text(call.with["download"]))) {assert.ok(produced.has(name), call.where + " downloads unproduced " + name);}
    }
    for (const {where, step} of allSteps().filter(row => text(row.step["uses"]).startsWith("actions/download-artifact@"))) {
      const options = map(step["with"]);
      const resolve = (/** @type {string} */ value) => value.replace(suffixExpression, "@").replace(/\$\{\{ env\.NATIVE_NAME \}\}/g, "native-linux");
      const name = resolve(text(options["name"])), pattern = resolve(text(options["pattern"]));
      if ((name || pattern).includes("${{")) {continue;}
      const names = downloadedNames(name, pattern);
      for (const name of names.filter(row => !row.includes("*"))) {
        assert.ok(produced.has(name), where + " downloads unproduced " + name);
      }
    }
  });

  test("needs and job-output references resolve, and reusable calls match callee inputs", () => {
    for (const [file, workflow] of Object.entries(workflows)) {
      const jobs = map(workflow["jobs"]);
      for (const [id, job] of Object.entries(jobs)) {
        const value = map(job), needs = list(value["needs"]).map(text), body = JSON.stringify(value);
        for (const need of needs) {assert.ok(Object.hasOwn(jobs, need), file + ":" + id + " needs unknown " + need);}
        for (const match of body.matchAll(/needs\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_-]+)/g)) {
          const producer = match[1] || "", key = match[2] || "", source = map(jobs[producer]);
          assert.ok(needs.includes(producer), file + ":" + id + " reads outputs of a job it does not need: " + producer);
          const callee = text(source["uses"]), outputs = callee ?
            map(map(map(workflows[path.basename(callee)]?.["on"])["workflow_call"])["outputs"]) : map(source["outputs"]);
          assert.ok(Object.hasOwn(outputs, key), file + ":" + id + " reads undefined output " + producer + "." + key);
        }
        const callee = text(value["uses"]);
        if (callee) {
          const inputs = map(map(map(workflows[path.basename(callee)]?.["on"])["workflow_call"])["inputs"]), given = map(value["with"]);
          for (const key of Object.keys(given)) {assert.ok(Object.hasOwn(inputs, key), file + ":" + id + " passes unknown input " + key);}
          for (const [key, input] of Object.entries(inputs)) {
            if (text(map(input)["required"]) === "true") {assert.ok(Object.hasOwn(given, key), file + ":" + id + " omits " + key);}
          }
        }
      }
      const outputs = map(map(map(workflow["on"])["workflow_call"])["outputs"]);
      for (const [key, output] of Object.entries(outputs)) {
        const match = /jobs\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_-]+)/.exec(text(map(output)["value"]));
        assert.ok(match && Object.hasOwn(jobs, match[1] || ""), file + " output " + key);
      }
    }
  });

  test("every script and test named by a workflow or action exists", () => {
    const sources = [...Object.keys(workflows).map(name => path.join(workflowRoot, name)),
      ...actions.map(([name]) => path.join(repo, ".github", "actions", name, "action.yml"))];
    let count = 0;
    for (const file of sources) {
      for (const match of fs.readFileSync(file, "utf8").matchAll(/(?:^|[\s"'=(])((?:scripts|tests)[\\/][A-Za-z0-9_./\\-]+\.(?:js|sh|ps1|bat))/g)) {
        count += 1;
        assert.ok(fs.existsSync(path.join(repo, ...(match[1] || "").split(/[\\/]/))), path.basename(file) + " names missing " + match[1]);
      }
    }
    assert.ok(count > 20);
  });
});
