"use strict";

// Behaviour of the CI job helpers: job preparation (native-job.yml) and chained-phase relays.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {describe, test} = require("node:test");
const job = require("../scripts/ci/job");
const chain = require("../scripts/ci/chain");
const {cacheKey} = require("../scripts/ci/cache");
const {reuseCachedLinuxLlvm} = require("../scripts/native-ci-linux-llvm");

/** @param {import("node:test").TestContext} t */
function workspace(t) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "mom-ci-jobs-")));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const files = {GITHUB_ENV: path.join(root, "env"), GITHUB_OUTPUT: path.join(root, "output")};
  for (const file of Object.values(files)) {fs.writeFileSync(file, "");}
  return {root, files, environment: {MOM_NATIVE_CI_SUFFIX: "ci-123-2", RUNNER_TEMP: root, ...files}};
}

describe("Native CI job preparation", {skip: process.platform !== "linux"}, () => {
  test("exports work roots and pins and plans exact transfers", t => {
    const {root, files, environment} = workspace(t);
    job.prepareJob({...environment, NATIVE_JOB_PINS: "MOM_NATIVE_WINDOWS_GPU_PROFILE=oneapi\nMOM_X_SHA256=" + "a".repeat(64) + "\n",
      NATIVE_JOB_DOWNLOAD: "native-a-@.tar\n native-b-@.tar", NATIVE_JOB_UPLOAD: "native-c-@.tar native-exe-@=out-@/mom",
      NATIVE_JOB_DIAGNOSTICS: "logs-@=/tmp/x-@/*.log"}, root);
    const env = fs.readFileSync(files.GITHUB_ENV, "utf8"), output = fs.readFileSync(files.GITHUB_OUTPUT, "utf8");
    assert.match(env, /^MOM_NATIVE_LINUX_SDK_DIR=\/tmp\/mom-native-llvm21-sdk-ci-123-2$/m);
    assert.match(env, /^MOM_NATIVE_WINDOWS_GPU_DIR=\/tmp\/mom-native-windows-gpu-oneapi-ci-123-2$/m);
    assert.match(env, /^MOM_X_SHA256=a{64}$/m);
    assert.match(output, /^pattern=\{native-a-ci-123-2\.tar,native-b-ci-123-2\.tar\}$/m);
    assert.match(output, new RegExp("^upload_1=" + path.join(root, "native-c-ci-123-2.tar") + "$\\nupload_1_name=$", "m"));
    assert.match(output, new RegExp("^upload_2=" + path.join(root, "out-ci-123-2/mom") + "$\\nupload_2_name=native-exe-ci-123-2$", "m"));
    assert.match(output, /^diagnostics_name=logs-ci-123-2$\ndiagnostics_path=\/tmp\/x-ci-123-2\/\*\.log$/m);
  });

  test("rejects a work root left by an earlier job on the same runner", t => {
    const {root, files, environment} = workspace(t);
    fs.mkdirSync(path.join(root, "mom-native-llvm21-sdk-ci-123-2"));
    assert.throws(() => job.prepareJob(environment, root), /work roots from an earlier job/);
    assert.equal(fs.readFileSync(files.GITHUB_ENV, "utf8"), "");
  });

  test("rejects empty, repeated or unprefixed pins and malformed transfers", () => {
    assert.throws(() => job.parsePins("MOM_A="), /malformed, empty or repeated/);
    assert.throws(() => job.parsePins("MOM_A=1\nMOM_A=2"), /malformed, empty or repeated/);
    assert.throws(() => job.parsePins("PATH=/tmp"), /malformed/);
    assert.throws(() => job.downloadPattern(["../x.tar"]), /plain file names/);
    assert.throws(() => job.uploadPlan(["a", "b", "c", "d", "e"], "/r"), /at most 4/);
    assert.throws(() => job.uploadPlan(["raw-*.tar"], "/r"), /malformed/);
    assert.equal(job.downloadPattern(["one.tar"]), "one.tar");
    assert.throws(() => job.nativeCiEnvironment({MOM_NATIVE_CI_SUFFIX: "../x"}), /MOM_NATIVE_CI_SUFFIX/);
  });
});

describe("Chained CI phases", {skip: process.platform !== "linux"}, () => {
  test("hand off outputs, retire consumed inputs and reset only new work roots", t => {
    const {root, environment} = workspace(t), deadline = String(Math.floor(Date.now() / 1000) + 3600),
      existing = path.join(root, "kept-ci-123-2");
    fs.mkdirSync(existing);
    const context = chain.chainContext({...environment, MOM_CI_DEADLINE_EPOCH: deadline});
    const state = path.join(root, "state-ci-123-2.tar"), extracted = path.join(root, "extracted-ci-123-2"), other = path.join(root, "unrelated");
    fs.writeFileSync(state, "state");
    fs.mkdirSync(path.join(extracted, "ro"), {recursive: true});
    fs.writeFileSync(path.join(extracted, "ro", "file"), "x");
    fs.chmodSync(path.join(extracted, "ro"), 0o500);
    fs.mkdirSync(other);
    chain.handOff(context, ["state-ci-123-2.tar"]);
    assert.equal(fs.readFileSync(path.join(context.incoming, "state-ci-123-2.tar"), "utf8"), "state");
    assert.throws(() => chain.handOff(context, ["state-ci-123-2.tar"]), /ENOENT/);
    fs.writeFileSync(path.join(root, "final-ci-123-2.tar"), "final");
    chain.resetWorkRoots(context, ["final-ci-123-2.tar"]);
    assert.deepEqual([fs.existsSync(extracted), fs.existsSync(existing), fs.existsSync(other), fs.existsSync(path.join(root, "final-ci-123-2.tar"))],
      [false, true, true, true]);
    chain.retire(context, ["state-ci-123-2.tar"]);
    assert.equal(fs.existsSync(path.join(context.incoming, "state-ci-123-2.tar")), false);
  });

  test("bounds each phase by 55 minutes and the job deadline, and runs phases with fresh workflow files", t => {
    const {environment} = workspace(t), now = Date.now(), deadline = Math.floor(now / 1000) + 600;
    const context = chain.chainContext({...environment, MOM_CI_DEADLINE_EPOCH: String(deadline)});
    assert.equal(chain.phaseDeadline(context, now), String(deadline));
    assert.equal(chain.phaseDeadline({...context, deadlineEpoch: deadline + 7200}, now), String(Math.floor(now / 1000) + 3300));
    assert.throws(() => chain.phaseDeadline(context, (deadline - 60) * 1000), /no time for the next phase/);
    assert.ok(context.runDeadlineEpoch > deadline);
    const script = "const fs=require('node:fs');fs.appendFileSync(process.env.GITHUB_OUTPUT,'bundle_sha256=" + "b".repeat(64) + "\\n');" +
      "fs.appendFileSync(process.env.GITHUB_ENV,'MOM_Y=1\\n');";
    const result = chain.runPhaseCommand(context, "phase", [process.execPath, "-e", script], process.env);
    assert.deepEqual(result, {outputs: {bundle_sha256: "b".repeat(64)}, exported: {MOM_Y: "1"}});
    assert.equal(chain.requiredOutput(result.outputs, "bundle_sha256"), "b".repeat(64));
    assert.throws(() => chain.requiredOutput(result.outputs, "manifest_sha256"), /did not report/);
    assert.throws(() => chain.runPhaseCommand(context, "fails", [process.execPath, "-e", "process.exit(3)"], process.env), /phase failed: fails/);
    assert.throws(() => chain.chainContext({...environment, MOM_CI_DEADLINE_EPOCH: "1000000000"}), /future MOM_CI_DEADLINE_EPOCH/);
  });
});

describe("Cached SDK work roots", {skip: process.platform !== "linux"}, () => {
  test("a restored Linux LLVM root that fails admission is removed so the job rebuilds it", t => {
    const {root} = workspace(t), sdkRoot = path.join(root, "mom-native-llvm21-sdk-ci-123-2"), output = path.join(sdkRoot, "linux-llvm21-sdk");
    assert.equal(reuseCachedLinuxLlvm({sdkRoot, output}), false, "a cache miss builds from scratch");
    fs.mkdirSync(path.join(output, "build"), {recursive: true});
    fs.writeFileSync(path.join(output, "result.json"), "{}\n");
    fs.chmodSync(path.join(output, "build"), 0o500);
    assert.equal(reuseCachedLinuxLlvm({sdkRoot, output}), false);
    assert.equal(fs.existsSync(sdkRoot), false);
  });

  test("cache keys exist only for the maintained SDK roots", () => {
    assert.throws(() => cacheKey("windows-everything", {MOM_NATIVE_CI_SUFFIX: "ci", MOM_BUILD_JOBS: "4"}), /Unknown native CI cache/);
  });
});

describe("Workflow stage deadlines", () => {
  test("every stage deadline a CI script reads is set by a workflow or derived by a CI wrapper", () => {
    // The CI restructure dropped the Windows GPU factory's deadline, so every GPU job failed its first check.
    const repo = path.join(__dirname, ".."), workflows = path.join(repo, ".github/workflows"), ci = path.join(repo, "scripts/ci");
    const read = (/** @type {string} */ dir) => fs.readdirSync(dir).filter(name => /\.(js|yml)$/.test(name))
      .map(name => fs.readFileSync(path.join(dir, name), "utf8")).join("\n");
    const scripts = read(path.join(repo, "scripts")), wrappers = read(ci), workflowText = read(workflows);
    const readNames = new Set([...scripts.matchAll(/(?:environment|env|process\.env)\[\s*"(MOM_[A-Z_]*DEADLINE_EPOCH)"\s*\]/g)]
      .map(match => match[1] || ""));
    assert.ok(readNames.has("MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH"));
    for (const name of readNames) {
      const set = new RegExp(`\\b${name}=`).test(workflowText) || new RegExp(`\\b${name}:`).test(wrappers);
      assert.ok(set, `${name} is read by a CI script but no workflow sets it and no CI wrapper derives it`);
    }
  });
});
