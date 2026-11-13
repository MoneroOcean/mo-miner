"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"),
  path = require("node:path"), {createHash} = require("node:crypto");
const {buildPhases} = require("../scripts/build-native-windows-llvm21-sdk"),
  {windowsLlvmCiJobPlan, windowsLlvmCiJobPreflight, windowsLlvmDescriptorProducerPins} = require("../scripts/run-native-windows-llvm-ci");

let fixtureSequence = 0;

/** @param {string | Buffer} bytes */
function sha256(bytes) {return createHash("sha256").update(bytes).digest("hex");}

/** @param {import('node:test').TestContext} t @param {number} [nowMs] */
function fixture(t, nowMs = Date.now()) {
  const runnerTemp = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-windows-llvm-job-")),
    runId = String(nowMs) + String(++fixtureSequence).padStart(4, "0"), attempt = "1", suffix = runId + "-" + attempt,
    runDeadlineEpoch = String(Math.ceil((nowMs + 7 * 60 * 60 * 1000) / 1000)),
    jobDeadlineEpoch = String(Math.ceil((nowMs + 55 * 60 * 1000) / 1000)),
    outputFile = path.join(runnerTemp, "github-output"), summaryFile = path.join(runnerTemp, "github-summary"),
    environmentFile = path.join(runnerTemp, "github-env"),
    environment = /** @type {NodeJS.ProcessEnv} */ ({RUNNER_TEMP: runnerTemp, GITHUB_RUN_ID: runId, GITHUB_RUN_ATTEMPT: attempt,
      GITHUB_JOB: "windows_llvm", MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2",
      MOM_NATIVE_WINDOWS_LLVM_DIR: path.join("/tmp", "mom-native-windows-llvm21-" + suffix),
      MOM_NATIVE_WINDOWS_LLVM_DESCRIPTOR_DIR: path.join(runnerTemp, "mom-native-windows-llvm21-descriptor-" + suffix),
      MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: runDeadlineEpoch, MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: jobDeadlineEpoch,
      MOM_NATIVE_LINUX_LLVM_TOOLS_BUNDLE_SHA256: "0".repeat(64), MOM_NATIVE_LINUX_LLVM_TOOLS_HANDOFF_SHA256: "0".repeat(64),
      MOM_NATIVE_WINDOWS_LLVM_HERMES_BUNDLE_SHA256: "0".repeat(64), MOM_NATIVE_WINDOWS_LLVM_HERMES_HANDOFF_SHA256: "0".repeat(64),
      GITHUB_OUTPUT: outputFile, GITHUB_STEP_SUMMARY: summaryFile, GITHUB_ENV: environmentFile});
  fs.writeFileSync(outputFile, "", {flag: "wx"}); fs.writeFileSync(summaryFile, "", {flag: "wx"});
  fs.writeFileSync(environmentFile, "", {flag: "wx"});
  const configure = windowsLlvmCiJobPlan("configure", environment, "linux", "x64", nowMs),
    toolsBytes = Buffer.from("small tools archive fixture\n"), hermesBytes = Buffer.from("small Hermes archive fixture\n");
  fs.mkdirSync(path.dirname(configure.incomingToolsArchive), {recursive: true});
  fs.mkdirSync(path.dirname(configure.incomingHermesArchive), {recursive: true});
  fs.writeFileSync(configure.incomingToolsArchive, toolsBytes, {flag: "wx"});
  fs.writeFileSync(configure.incomingHermesArchive, hermesBytes, {flag: "wx"});
  environment["MOM_NATIVE_LINUX_LLVM_TOOLS_BUNDLE_SHA256"] = sha256(toolsBytes);
  environment["MOM_NATIVE_LINUX_LLVM_TOOLS_HANDOFF_SHA256"] = "1".repeat(64);
  environment["MOM_NATIVE_WINDOWS_LLVM_HERMES_BUNDLE_SHA256"] = sha256(hermesBytes);
  environment["MOM_NATIVE_WINDOWS_LLVM_HERMES_HANDOFF_SHA256"] = "2".repeat(64);
  t.after(() => {
    fs.rmSync(runnerTemp, {recursive: true, force: true});
    fs.rmSync(configure.stateRoot, {recursive: true, force: true});
  });
  return {runnerTemp, runId, attempt, suffix, runDeadlineEpoch, jobDeadlineEpoch, outputFile, summaryFile, environmentFile,
    environment, configure};
}

/** @param {ReturnType<typeof fixture>} f @param {string} stage */
function phaseFixture(f, stage) {
  const stageIndex = buildPhases.findIndex(row => row.name === stage),
    previousStage = stage === "audit" ? "lld" : stageIndex < 0 ? "" : stageIndex === 0 ? "configured" : buildPhases[stageIndex - 1]?.name || "",
    expectedProducerStage = previousStage === "configured" ? "configure" : previousStage;
  f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_PREVIOUS_STAGE"] = previousStage;
  f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_BUNDLE_SHA256"] = "3".repeat(64);
  f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256"] = "4".repeat(64);
  const expected = windowsLlvmCiJobPlan(stage, f.environment, "linux", "x64"),
    archiveBytes = Buffer.from("prior state archive for " + expected.previousStage + "\n"),
    manifest = {runAttempt: f.suffix, producerJob: f.configure.workflowJob + "_" + expectedProducerStage,
      archive: {sha256: sha256(archiveBytes)}};
  if (!expected.incomingStateArchive || !expected.incomingStateManifest || !expected.previousStage) {
    throw new Error("Phase fixture did not derive a previous state artifact");
  }
  fs.mkdirSync(path.dirname(expected.incomingStateArchive), {recursive: true});
  const manifestPath = expected.incomingStateManifest;
  if (!manifestPath) {throw new Error("Phase fixture omitted its manifest path");}
  fs.writeFileSync(expected.incomingStateArchive, archiveBytes, {flag: "wx"});
  fs.writeFileSync(manifestPath, JSON.stringify(manifest) + "\n", {flag: "wx"});
  f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_BUNDLE_SHA256"] = sha256(archiveBytes);
  f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256"] = sha256(JSON.stringify(manifest) + "\n");
  f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_PREVIOUS_STAGE"] = expected.previousStage;
  return expected;
}

test.describe("Windows LLVM CI phase job plan", {skip: process.platform !== "linux"}, () => {
  test("maps only the maintained configure, builder, and audit order to distinct job identities", t => {
    const f = fixture(t), names = ["configure", ...buildPhases.map(row => row.name), "audit"], plans = names.map(stage => {
      if (stage !== "configure") {phaseFixture(f, stage);}
      return windowsLlvmCiJobPlan(stage, f.environment, "linux", "x64");
    });
    assert.deepEqual(plans.map(row => row.stage), names);
    assert.deepEqual(plans.slice(1, -1).map(row => row.builderPhase), buildPhases.map(row => row.name));
    assert.deepEqual(plans.map(row => row.statePhase), ["configured", ...buildPhases.map(row => row.name), "audited"]);
    assert.deepEqual(plans.map(row => row.previousStage), [null, "configured", ...buildPhases.slice(0, -1).map(row => row.name), "lld"]);
    assert.equal(new Set(plans.map(row => row.jobId)).size, names.length, "each phase uses a distinct immutable deadline-record identity");
    assert.ok(plans.every(row => row.workflowJob === "windows_llvm"));
    assert.throws(() => windowsLlvmCiJobPlan("llvm-unknown", f.environment, "linux", "x64"), /maintained phase order/);
    assert.throws(() => windowsLlvmCiJobPlan("configure", f.environment, "win32", "x64"), /Linux x64/);
  });

  test("preflights a fresh configure cohort and keeps outputs separate from immutable inputs", t => {
    const f = fixture(t), plan = windowsLlvmCiJobPreflight("configure", f.environment, "linux", "x64");
    assert.equal(plan.previousStage, null);
    assert.equal(plan.builderPhase, null);
    assert.equal(plan.statePhase, "configured");
    assert.equal(plan.incomingStateArchive, null);
    assert.equal(fs.existsSync(plan.stateRoot), false);
    assert.equal(fs.existsSync(plan.archive), false);
    assert.equal(plan.toolsArchive.sha256, f.environment["MOM_NATIVE_LINUX_LLVM_TOOLS_BUNDLE_SHA256"]);
    assert.equal(plan.hermesArchive.sha256, f.environment["MOM_NATIVE_WINDOWS_LLVM_HERMES_BUNDLE_SHA256"]);
    assert.equal(fs.readFileSync(f.outputFile, "utf8"), "");
    assert.equal(fs.readFileSync(f.summaryFile, "utf8"), "");

    const aliased = {...f.environment, GITHUB_OUTPUT: plan.incomingToolsArchive};
    assert.throws(() => windowsLlvmCiJobPreflight("configure", aliased, "linux", "x64"), /alias|overlap|distinct|input/i,
      "workflow output may not overwrite a pinned input archive");
    const envAlias = {...f.environment, GITHUB_ENV: plan.incomingToolsArchive};
    assert.throws(() => windowsLlvmCiJobPreflight("configure", envAlias, "linux", "x64"), /alias|overlap|distinct|input/i,
      "the configure run-deadline record may not overwrite a pinned input archive");
    const envOutputAlias = {...f.environment, GITHUB_ENV: f.outputFile};
    assert.throws(() => windowsLlvmCiJobPreflight("configure", envOutputAlias, "linux", "x64"), /distinct canonical/i,
      "GitHub environment and output channels must remain separate files");
    const missingEnv = {...f.environment}; delete missingEnv["GITHUB_ENV"];
    delete missingEnv["MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH"];
    assert.throws(() => windowsLlvmCiJobPreflight("configure", missingEnv, "linux", "x64"), /GITHUB_ENV|workflow output files/i,
      "configure must have a protected GitHub environment file to persist the minted run deadline");
    assert.equal(fs.readFileSync(plan.incomingToolsArchive, "utf8"), "small tools archive fixture\n");
  });

  test("rejects missing or wrong external bundle pins and mismatched run attempts before mutation", t => {
    const f = fixture(t), before = fs.readFileSync(f.outputFile, "utf8");
    const missingPin = {...f.environment}; delete missingPin["MOM_NATIVE_LINUX_LLVM_TOOLS_HANDOFF_SHA256"];
    assert.throws(() => windowsLlvmCiJobPlan("configure", missingPin, "linux", "x64"), /external artifact SHA-256 pin/);
    const wrongPin = {...f.environment, MOM_NATIVE_LINUX_LLVM_TOOLS_BUNDLE_SHA256: "f".repeat(64)};
    assert.throws(() => windowsLlvmCiJobPreflight("configure", wrongPin, "linux", "x64"), /external SHA-256 pin/);
    const wrongRun = {...f.environment, GITHUB_RUN_ID: String(Number(f.runId) + 1)};
    assert.throws(() => windowsLlvmCiJobPreflight("configure", wrongRun, "linux", "x64"), /exact run-attempt plan/);
    const wrongDescriptorRoot = {...f.environment,
      MOM_NATIVE_WINDOWS_LLVM_DESCRIPTOR_DIR: path.join(f.runnerTemp, "redirected-descriptor")};
    assert.throws(() => windowsLlvmCiJobPlan("configure", wrongDescriptorRoot, "linux", "x64"), /exact same-run path/);
    assert.equal(fs.existsSync(f.configure.stateRoot), false, "invalid input pins and run identity do not create the state root");
    assert.equal(fs.readFileSync(f.outputFile, "utf8"), before);
  });

  test("requires the exact immediately preceding phase artifact and rejects stale manifests", t => {
    const f = fixture(t), configuredInput = phaseFixture(f, "llvm-amdgpu"), configured = windowsLlvmCiJobPreflight("llvm-amdgpu", f.environment, "linux", "x64");
    assert.equal(configured.previousStage, "configured");
    assert.equal(configuredInput.incomingStateManifest && JSON.parse(fs.readFileSync(configuredInput.incomingStateManifest, "utf8")).producerJob,
      "windows_llvm_configure", "the configured artifact is emitted by the configure job, not a synthetic configured job");
    const phase = phaseFixture(f, "llvm-nvptx"), valid = windowsLlvmCiJobPreflight("llvm-nvptx", f.environment, "linux", "x64");
    assert.equal(phase.previousStage, "llvm-amdgpu");
    assert.equal(valid.previousStage, "llvm-amdgpu");
    assert.equal(valid.manifestValue.producerJob, "windows_llvm_llvm-amdgpu");

    const wrongPrevious = {...f.environment, MOM_NATIVE_WINDOWS_LLVM_STATE_PREVIOUS_STAGE: "configured"};
    assert.throws(() => windowsLlvmCiJobPlan("llvm-nvptx", wrongPrevious, "linux", "x64"), /immediately preceding/);
    const wrongProducer = {...valid.manifestValue, producerJob: "windows_llvm_lld"};
    const manifestPath = phase.incomingStateManifest;
    if (!manifestPath) {throw new Error("Phase fixture omitted its manifest path");}
    fs.writeFileSync(manifestPath, JSON.stringify(wrongProducer) + "\n");
    f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256"] = sha256(JSON.stringify(wrongProducer) + "\n");
    assert.throws(() => windowsLlvmCiJobPreflight("llvm-nvptx", f.environment, "linux", "x64"), /immediately preceding stage/);
    assert.equal(fs.existsSync(valid.stateRoot), false, "stale restore artifacts never create a local state root");

    const wrongBundlePin = {...f.environment, MOM_NATIVE_WINDOWS_LLVM_STATE_BUNDLE_SHA256: "e".repeat(64)};
    assert.throws(() => windowsLlvmCiJobPreflight("llvm-nvptx", wrongBundlePin, "linux", "x64"), /external SHA-256 pin/);
  });

  test("rejects expired, extended, or missing original run deadlines and oversized phase identities", t => {
    const f = fixture(t), expired = {...f.environment, MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: "1000000000"};
    assert.throws(() => windowsLlvmCiJobPlan("configure", expired, "linux", "x64"), /expired|60-minute|Insufficient/);
    const extendedRun = {...f.environment,
      MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH: String(Math.ceil((Date.now() + 9 * 60 * 60 * 1000) / 1000))};
    assert.throws(() => windowsLlvmCiJobPlan("configure", extendedRun, "linux", "x64"), /8-hour run bound/);
    const noRunForResume = {...f.environment}; delete noRunForResume["MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH"];
    assert.throws(() => windowsLlvmCiJobPlan("llvm-amdgpu", noRunForResume, "linux", "x64"), /original run deadline/);
    const configureMints = {...f.environment}; delete configureMints["MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH"];
    const minted = windowsLlvmCiJobPlan("configure", configureMints, "linux", "x64");
    assert.ok(Number(minted.runDeadlineEpoch) * 1000 >= Date.now() + 7 * 60 * 60 * 1000);
    assert.ok(Number(minted.runDeadlineEpoch) * 1000 <= Date.now() + 8 * 60 * 60 * 1000);
    const longJob = {...f.environment, GITHUB_JOB: "j".repeat(64)};
    assert.throws(() => windowsLlvmCiJobPlan("llvm-amdgpu", longJob, "linux", "x64"), /workflow job identity/);
    assert.equal(fs.existsSync(f.configure.stateRoot), false);
  });

  test("rejects existing or redirected destination paths before any state mutation", t => {
    const f = fixture(t), plan = windowsLlvmCiJobPlan("configure", f.environment, "linux", "x64");
    fs.mkdirSync(plan.stateRoot, {mode: 0o700}); fs.writeFileSync(path.join(plan.stateRoot, "keep"), "untouched\n");
    assert.throws(() => windowsLlvmCiJobPreflight("configure", f.environment, "linux", "x64"), /fresh local state root/);
    assert.equal(fs.readFileSync(path.join(plan.stateRoot, "keep"), "utf8"), "untouched\n");
    fs.rmSync(plan.stateRoot, {recursive: true});

    const outputLink = path.join(f.runnerTemp, "output-link"); fs.symlinkSync(f.outputFile, outputLink);
    assert.throws(() => windowsLlvmCiJobPreflight("configure", {...f.environment, GITHUB_OUTPUT: outputLink}, "linux", "x64"), /canonical|regular|redirected/i);
    assert.equal(fs.existsSync(plan.stateRoot), false);
  });
});

test.describe("Windows LLVM descriptor producer output pins", {skip: process.platform !== "linux"}, () => {
  /** @param {import('node:test').TestContext} t */
  const fixture = t => {
    const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-windows-llvm-descriptor-pins-")),
      descriptorPath = path.join(root, "descriptor.json"), provenancePath = path.join(root, "provenance.json");
    fs.writeFileSync(descriptorPath, "{\"version\":1}\n", {flag: "wx", mode: 0o444});
    fs.writeFileSync(provenancePath, "{\"status\":\"PASS\"}\n", {flag: "wx", mode: 0o444});
    const generated = /** @type {Record<string,string>} */ ({windows_llvm_descriptor_path: descriptorPath,
      windows_llvm_descriptor_sha256: sha256(fs.readFileSync(descriptorPath)),
      windows_llvm_descriptor_provenance_path: provenancePath,
      windows_llvm_descriptor_provenance_sha256: sha256(fs.readFileSync(provenancePath))});
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    return {root, descriptorPath, provenancePath, generated};
  };

  test("accepts exact producer paths and hashes without modifying either output", t => {
    const f = fixture(t), before = [fs.readFileSync(f.descriptorPath, "utf8"), fs.readFileSync(f.provenancePath, "utf8")];
    assert.deepEqual(windowsLlvmDescriptorProducerPins(f.generated, f.descriptorPath, f.provenancePath), {
      descriptorSha: f.generated["windows_llvm_descriptor_sha256"],
      provenanceSha: f.generated["windows_llvm_descriptor_provenance_sha256"],
    });
    assert.deepEqual([fs.readFileSync(f.descriptorPath, "utf8"), fs.readFileSync(f.provenancePath, "utf8")], before);
  });

  test("rejects descriptor or provenance bytes changed after producer pinning", t => {
    const descriptor = fixture(t), staleDescriptorPin = descriptor.generated["windows_llvm_descriptor_sha256"];
    fs.chmodSync(descriptor.descriptorPath, 0o644);
    fs.writeFileSync(descriptor.descriptorPath, "{\"version\":2}\n");
    const replacementHash = sha256(fs.readFileSync(descriptor.descriptorPath));
    assert.notEqual(replacementHash, staleDescriptorPin);
    assert.equal(sha256(fs.readFileSync(descriptor.descriptorPath)), replacementHash,
      "a caller that re-hashed the changed file would incorrectly accept the replacement as its new pin");
    assert.throws(() => windowsLlvmDescriptorProducerPins(descriptor.generated, descriptor.descriptorPath, descriptor.provenancePath), /external SHA-256 pin/);

    const provenance = fixture(t), staleProvenancePin = provenance.generated["windows_llvm_descriptor_provenance_sha256"];
    fs.chmodSync(provenance.provenancePath, 0o644);
    fs.writeFileSync(provenance.provenancePath, "{\"status\":\"PASS\",\"changed\":true}\n");
    assert.notEqual(sha256(fs.readFileSync(provenance.provenancePath)), staleProvenancePin);
    assert.throws(() => windowsLlvmDescriptorProducerPins(provenance.generated, provenance.descriptorPath, provenance.provenancePath), /external SHA-256 pin/);
  });

  test("rejects missing pins and substituted output paths", t => {
    const f = fixture(t), missing = {...f.generated}; delete missing["windows_llvm_descriptor_sha256"];
    assert.throws(() => windowsLlvmDescriptorProducerPins(missing, f.descriptorPath, f.provenancePath), /inconsistent paths or pins/);
    assert.throws(() => windowsLlvmDescriptorProducerPins(f.generated, path.join(f.root, "other.json"), f.provenancePath), /inconsistent paths or pins/);
    assert.throws(() => windowsLlvmDescriptorProducerPins(f.generated, f.descriptorPath, path.join(f.root, "other-provenance.json")), /inconsistent paths or pins/);
  });
});
