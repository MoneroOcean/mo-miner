"use strict";

const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"),
  {spawnSync} = require("node:child_process"), {createHash} = require("node:crypto"), test = require("node:test");

const repository = path.resolve(__dirname, ".."), workflowPath = path.join(repository, ".github/workflows/build-native-prerequisites.yml");

function applicationHandoffScript() {
  const workflow = fs.readFileSync(workflowPath, "utf8"), stepStart = workflow.indexOf("id: native-runtime-input");
  assert.notEqual(stepStart, -1, "Linux application runtime handoff step exists");
  const stepEnd = workflow.indexOf("\n      - name:", stepStart), step = workflow.slice(stepStart, stepEnd === -1 ? undefined : stepEnd),
    match = step.match(/^ {10}node <<'NODE'\n([\s\S]*?)^ {10}NODE\s*$/m);
  assert.ok(match, "Linux application handoff embeds the tested Node implementation");
  return match[1] || "";
}

/** @param {string | Buffer} data */
function hash(data) {return createHash("sha256").update(data).digest("hex");}

/** @param {import('node:test').TestContext} t @param {string} [variant] */
function fixture(t, variant = "valid") {
  const runnerTemp = fs.mkdtempSync(path.join(fs.realpathSync("/var/tmp"), "mom-linux-app-metadata-")), attempt = "1",
    outputFile = path.join(runnerTemp, "github-output"), summaryFile = path.join(runnerTemp, "github-summary");
  let run = String(BigInt(Date.now()) * 100000n + BigInt(process.pid) * 100n), suffix = run + "-" + attempt,
    appRoot = path.join("/tmp", "mom-native-linux-application-" + suffix), evidenceRoot = path.join("/tmp", "native-linux-application-build-" + suffix),
    runtimeDirectory = path.join(runnerTemp, "mom-native-runtime-" + suffix),
    configPath = path.join(runnerTemp, "native-linux-assembly-config-" + suffix + ".json"),
    archive = path.join(runnerTemp, "native-linux-application-build-" + suffix + ".tar");
  for (let index = 0; [appRoot, evidenceRoot, archive].some(file => fs.existsSync(file)); index++) {
    run = String(BigInt(Date.now()) * 100000n + BigInt(process.pid) * 100n + BigInt(index + 1));
    suffix = run + "-" + attempt;
    appRoot = path.join("/tmp", "mom-native-linux-application-" + suffix);
    evidenceRoot = path.join("/tmp", "native-linux-application-build-" + suffix);
    runtimeDirectory = path.join(runnerTemp, "mom-native-runtime-" + suffix);
    configPath = path.join(runnerTemp, "native-linux-assembly-config-" + suffix + ".json");
    archive = path.join(runnerTemp, "native-linux-application-build-" + suffix + ".tar");
  }
  fs.mkdirSync(appRoot, {mode: 0o700});
  const binary = path.join(appRoot, "mom"), receiptPath = path.join(appRoot, "result.json"), entrypointPath = path.join(appRoot, "entrypoint.json"),
    executable = Buffer.alloc(64);
  executable.set([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1]);
  fs.writeFileSync(binary, executable, {flag: "wx", mode: 0o755}); fs.chmodSync(binary, 0o755);
  fs.writeFileSync(configPath, JSON.stringify({version: 1, target: "linux", application: {application: true}}) + "\n", {flag: "wx"});
  fs.writeFileSync(outputFile, "", {flag: "wx"}); fs.writeFileSync(summaryFile, "", {flag: "wx"});
  const executableHash = hash(executable), executableBytes = executable.length, configHash = hash(fs.readFileSync(configPath)),
    receipt = {binary, sha256: executableHash, bytes: executableBytes, configHash, buildOnly: true,
      publicationQualified: false, application: {application: true, entrypoint: "mom.js", buildOnly: true}},
    receiptBytes = Buffer.from(JSON.stringify(receipt) + "\n");
  fs.writeFileSync(receiptPath, receiptBytes, {flag: "wx"});
  const receiptHash = hash(receiptBytes), entrypoint = {version: 1, target: variant === "target" ? "win32" : "linux", configFile: configPath,
    configHash, source: {path: binary, sha256: executableHash}, artifact: {path: binary, bytes: executableBytes, sha256: executableHash},
    recipe: {version: 1}, componentHashes: {[receiptPath]: receiptHash}, producerReceipt: {path: receiptPath, sha256: receiptHash},
    buildOnly: true, publicationQualified: false};
  fs.writeFileSync(entrypointPath, JSON.stringify(entrypoint, null, 2) + "\n", {flag: "wx"});
  if (variant === "raw-hash") {fs.writeFileSync(binary, Buffer.concat([executable, Buffer.from("tamper")]), {mode: 0o755});}
  t.after(() => {
    fs.rmSync(appRoot, {recursive: true, force: true}); fs.rmSync(evidenceRoot, {recursive: true, force: true});
    fs.rmSync(runnerTemp, {recursive: true, force: true});
  });
  const env = {...process.env, RUNNER_TEMP: runnerTemp, NATIVE_RUN_ID: run, NATIVE_RUN_ATTEMPT: attempt,
    NATIVE_APP_ROOT: appRoot, NATIVE_RUNTIME_DIRECTORY: runtimeDirectory, MOM_NATIVE_APPLICATION_DEADLINE_EPOCH: String(Math.floor(Date.now() / 1000) + 600),
    GITHUB_OUTPUT: outputFile, GITHUB_STEP_SUMMARY: summaryFile};
  return {env, runnerTemp, run, attempt, suffix, appRoot, binary, receiptPath, entrypointPath, configPath,
    evidenceRoot, archive, runtimeDirectory, outputFile, summaryFile, executableHash, executableBytes, configHash, receiptHash};
}

/** @param {NodeJS.ProcessEnv} env */
function runHandoff(env) {
  return spawnSync(process.execPath, ["-e", applicationHandoffScript()], {cwd: repository, env, encoding: "utf8", timeout: 15_000});
}

test.describe("Linux compact application build handoff", {skip: process.platform !== "linux"}, () => {
  test("preserves exact raw provenance in a compact four-file bundle", t => {
    const f = fixture(t), originalEntrypointHash = hash(fs.readFileSync(f.entrypointPath)), originalReceiptHash = hash(fs.readFileSync(f.receiptPath)),
      result = runHandoff(f.env);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    assert.equal(hash(fs.readFileSync(f.binary)), f.executableHash);
    assert.equal(hash(fs.readFileSync(f.entrypointPath)), originalEntrypointHash);
    assert.equal(hash(fs.readFileSync(f.receiptPath)), originalReceiptHash);
    assert.equal(hash(fs.readFileSync(path.join(f.runtimeDirectory, "mom"))), f.executableHash);

    const outputs = Object.fromEntries(fs.readFileSync(f.outputFile, "utf8").trim().split("\n").map(line => {
      const index = line.indexOf("="); return [line.slice(0, index), line.slice(index + 1)];
    }));
    assert.equal(outputs["sha256"], f.executableHash);
    assert.equal(outputs["executable_bytes"], String(f.executableBytes));
    assert.equal(outputs["entrypoint_sha256"], originalEntrypointHash);
    assert.equal(outputs["producer_receipt_sha256"], originalReceiptHash);
    assert.equal(outputs["build_handoff_artifact_name"], `native-linux-application-build-${f.suffix}.tar`);
    assert.equal(outputs["build_handoff_bundle_sha256"], hash(fs.readFileSync(f.archive)));

    const ci = require("../scripts/prepare-native-ci.js"),
      rootName = `native-linux-application-build-${f.suffix}`, members = ci.listState(f.archive, [rootName]).map(name => name.replace(/\/$/, "")).sort(),
      expected = [rootName, ...["application-config.json", "entrypoint.json", "handoff.json", "result.json"].map(name => `${rootName}/${name}`)].sort(),
      extraction = fs.mkdtempSync(path.join(f.runnerTemp, "extract-"));
    t.after(() => fs.rmSync(extraction, {recursive: true, force: true}));
    assert.deepEqual(members, expected);
    assert.ok(fs.statSync(f.archive).size < 128 * 1024 * 1024);
    ci.extractState(f.archive, extraction, [rootName]);
    const evidence = path.join(extraction, rootName), handoffPath = path.join(evidence, "handoff.json"), handoff = JSON.parse(fs.readFileSync(handoffPath, "utf8"));
    assert.deepEqual(handoff, {version: 1, format: "mom-linux-application-build-handoff-v1", runAttempt: f.suffix,
      target: "x86_64-unknown-linux-gnu", artifact: {file: "mom", bytes: f.executableBytes, sha256: f.executableHash},
      config: {file: "application-config.json", sha256: f.configHash},
      producerReceipt: {file: "result.json", sha256: f.receiptHash},
      entrypointReceipt: {file: "entrypoint.json", sha256: originalEntrypointHash},
      buildOnly: true, executed: false, publicationQualified: false});
    assert.equal(outputs["build_handoff_sha256"], hash(fs.readFileSync(handoffPath)));
    assert.equal(hash(fs.readFileSync(path.join(evidence, "application-config.json"))), f.configHash);
    assert.equal(hash(fs.readFileSync(path.join(evidence, "entrypoint.json"))), originalEntrypointHash);
    assert.equal(hash(fs.readFileSync(path.join(evidence, "result.json"))), originalReceiptHash);
    assert.match(fs.readFileSync(f.summaryFile, "utf8"), /no executable ran and this is not publication qualification/);
  });

  for (const [variant, reason] of /** @type {[string, RegExp][]} */ ([["raw-hash", /producer receipt/], ["target", /entrypoint, config or producer receipt/]])) {
    test(`refuses ${variant} mismatch before creating output`, t => {
      const f = fixture(t, variant), result = runHandoff(f.env);
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, reason);
      assert.equal(fs.existsSync(f.evidenceRoot), false);
      assert.equal(fs.existsSync(f.archive), false);
      assert.equal(fs.existsSync(f.runtimeDirectory), false);
      assert.equal(fs.readFileSync(f.outputFile, "utf8"), "");
    });
  }
});
