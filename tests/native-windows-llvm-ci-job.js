"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"),
  path = require("node:path"), vm = require("node:vm"), {createHash} = require("node:crypto");
const {buildPhases} = require("../scripts/build-native-windows-llvm21-sdk"),
  {windowsLlvmCiJobPlan, windowsLlvmCiJobPreflight, windowsLlvmDescriptorProducerPins, hostSdkDeadline} = require("../scripts/run-native-windows-llvm-ci");

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
    previousStage = stage === "audit" ? "lld" : stage === "llvm-mid" ? "llvm-amdgpu" : stageIndex < 0 ? "" :
      stageIndex === 0 ? "configured" : buildPhases[stageIndex - 1]?.name || "";
  f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_PREVIOUS_STAGE"] = previousStage;
  f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_BUNDLE_SHA256"] = "3".repeat(64);
  f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256"] = "4".repeat(64);
  const expected = windowsLlvmCiJobPlan(stage, f.environment, "linux", "x64"),
    archiveBytes = Buffer.from("prior state archive for " + expected.previousStage + "\n"),
    producerStage = expected.previousProducerStage || previousStage,
    manifest = {runAttempt: f.suffix, producerJob: f.configure.workflowJob + "_" + (producerStage === "configured" ? "configure" : producerStage),
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
  test("reconstructs external SDK and tool inputs before admitting restored compiler state", () => {
    const source = fs.readFileSync(require.resolve("../scripts/run-native-windows-llvm-ci"), "utf8"),
      restore = source.indexOf("stateTransport.restoreWindowsNativeLlvmState("),
      sdk = source.indexOf("ci.buildWindowsHostSdkSourcesCi();"),
      tools = source.indexOf("const tools = linuxLlvmToolHandoffInputs("),
      hermes = source.indexOf("hermes = windowsControllerHandoffInputs(");
    assert.ok(Math.min(restore, sdk, tools, hermes) >= 0);
    assert.ok(restore > Math.max(sdk, tools, hermes), "restoration validates the external descriptor inputs, not just archived state");
  });

  test("maps only the maintained configure, builder, and audit order to distinct job identities", t => {
    const f = fixture(t), names = ["configure", "llvm-amdgpu", "llvm-mid", "clang", "lld", "audit"], plans = names.map(stage => {
      if (stage !== "configure") {phaseFixture(f, stage);}
      return windowsLlvmCiJobPlan(stage, f.environment, "linux", "x64");
    });
    assert.deepEqual(plans.map(row => row.stage), names);
    assert.deepEqual(plans.map(row => row.builderPhases), [[], ["llvm-amdgpu"], ["llvm-nvptx", "llvm-x86", "llvm-core"], ["clang"], ["lld"], []]);
    assert.deepEqual(plans.map(row => row.statePhase), ["configured", "llvm-amdgpu", "llvm-core", "clang", "lld", "audited"]);
    assert.deepEqual(plans.map(row => row.previousStage), [null, "configured", "llvm-amdgpu", "llvm-core", "clang", "lld"]);
    assert.deepEqual(plans.map(row => row.previousProducerStage), [null, "configured", "llvm-amdgpu", "llvm-mid", "clang", "lld"]);
    assert.ok(plans.every(row => !Object.hasOwn(row, "builderPhase")), "the obsolete single-phase selector is not maintained");
    assert.equal(new Set(plans.map(row => row.jobId)).size, names.length, "each phase uses a distinct immutable deadline-record identity");
    assert.ok(plans.every(row => row.workflowJob === "windows_llvm"));
    for (const retired of ["llvm-nvptx", "llvm-x86", "llvm-core", "llvm-unknown"]) {
      assert.throws(() => windowsLlvmCiJobPlan(retired, f.environment, "linux", "x64"), /maintained phase order/);
    }
    assert.throws(() => windowsLlvmCiJobPlan("configure", f.environment, "win32", "x64"), /Linux x64/);
  });

  test("finishStage accepts each maintained phase and rejects malformed output without appending it", t => {
    const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-windows-llvm-finish-stage-")),
      archive = path.join(root, "bundle.tar"), bundleSha256 = sha256(Buffer.from("small state archive fixture\n")),
      source = fs.readFileSync(require.resolve("../scripts/run-native-windows-llvm-ci"), "utf8"),
      start = source.indexOf("function finishStage("), end = source.indexOf("\nmodule.exports", start);
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(archive, "small state archive fixture\n", {flag: "wx"});
    assert.ok(start >= 0 && end > start, "finishStage remains extractable for direct behavior coverage");
    const finishStage = vm.runInNewContext("(" + source.slice(start, end) + ")", {
      Buffer, createHash, fs, path, buildPhaseOutputValues: new Set(buildPhases.map(row => "build-" + row.name)),
      fileHash: (/** @type {string} */ file) => sha256(fs.readFileSync(file)),
      stateTransport: {packWindowsNativeLlvmState: () => {
        const manifest = {archive: {sha256: bundleSha256}}, manifestSha256 = sha256(JSON.stringify(manifest) + "\n");
        return {manifest, manifestSha256};
      }},
    });
    const names = ["configure", "llvm-amdgpu", "llvm-mid", "clang", "lld", "audit"];
    /** @param {number} index */
    function environmentFor(index) {
      const output = path.join(root, "output-" + index), summary = path.join(root, "summary-" + index);
      fs.writeFileSync(output, "", {flag: "wx"}); fs.writeFileSync(summary, "", {flag: "wx"});
      return {GITHUB_OUTPUT: output, GITHUB_STEP_SUMMARY: summary};
    }
    /** @param {string} stage @param {number} index */
    function planFor(stage, index) {
      return {stage, statePhase: stage === "llvm-mid" ? "llvm-core" : stage, archive,
        manifest: path.join(root, "manifest-" + index), runDeadlineEpoch: "1794850000", environment: {}};
    }
    /** @param {string} stage */
    function valuesFor(stage) {
      const outputPhase = stage === "llvm-mid" ? "llvm-core" : stage;
      return {windows_llvm_phase: stage === "configure" || stage === "audit" ? stage : "build-" + outputPhase,
        windows_llvm_build_only: "true", windows_llvm_executed: "false", windows_llvm_admission_sha256: "a".repeat(64)};
    }
    for (const [index, stage] of names.entries()) {
      const outputPhase = stage === "llvm-mid" ? "llvm-core" : stage,
        environment = environmentFor(index), value = stage === "configure" || stage === "audit" ? stage : "build-" + outputPhase,
        result = finishStage(planFor(stage, index), valuesFor(stage), environment);
      assert.equal(result.windows_llvm_phase, value);
      assert.ok(fs.readFileSync(environment.GITHUB_OUTPUT, "utf8").split("\n").includes("windows_llvm_phase=" + value));
    }
    for (const [index, invalid] of [
      {windows_llvm_phase: "build-llvm-mid\ninjected=value"},
      {windows_llvm_phase: "build-llvm-unmaintained"},
      {windows_llvm_phase: "build-llvm-core", "injected\nkey": "true"},
    ].entries()) {
      const environment = environmentFor(names.length + index), before = "preserve=this\n";
      fs.writeFileSync(environment.GITHUB_OUTPUT, before);
      assert.throws(() => finishStage(planFor("llvm-mid", names.length + index), {...valuesFor("llvm-mid"), ...invalid}, environment),
        /Windows LLVM workflow output is malformed/);
      assert.equal(fs.readFileSync(environment.GITHUB_OUTPUT, "utf8"), before);
    }
    const environment = environmentFor(names.length + 3);
    assert.throws(() => finishStage(planFor("llvm-mid", names.length + 3), {...valuesFor("llvm-mid"), windows_llvm_phase: "build-llvm-x86"}, environment),
      /builder result does not match its maintained phase/);
    assert.equal(fs.readFileSync(environment.GITHUB_OUTPUT, "utf8"), "", "the group cannot emit an intermediate phase as its checkpoint");
  });

  test("preflights a fresh configure cohort and keeps outputs separate from immutable inputs", t => {
    const f = fixture(t), plan = windowsLlvmCiJobPreflight("configure", f.environment, "linux", "x64");
    assert.equal(plan.previousStage, null);
    assert.deepEqual(plan.builderPhases, []);
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
    const wrongRun = {...f.environment, GITHUB_RUN_ID: String(BigInt(f.runId) + 1n)};
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
    const phase = phaseFixture(f, "llvm-mid"), valid = windowsLlvmCiJobPreflight("llvm-mid", f.environment, "linux", "x64");
    assert.equal(phase.previousStage, "llvm-amdgpu");
    assert.equal(valid.previousStage, "llvm-amdgpu");
    assert.equal(valid.manifestValue.producerJob, "windows_llvm_llvm-amdgpu");

    const wrongPrevious = {...f.environment, MOM_NATIVE_WINDOWS_LLVM_STATE_PREVIOUS_STAGE: "configured"};
    assert.throws(() => windowsLlvmCiJobPlan("llvm-mid", wrongPrevious, "linux", "x64"), /immediately preceding/);
    const wrongProducer = {...valid.manifestValue, producerJob: "windows_llvm_lld"};
    const manifestPath = phase.incomingStateManifest;
    if (!manifestPath) {throw new Error("Phase fixture omitted its manifest path");}
    fs.writeFileSync(manifestPath, JSON.stringify(wrongProducer) + "\n");
    f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256"] = sha256(JSON.stringify(wrongProducer) + "\n");
    assert.throws(() => windowsLlvmCiJobPreflight("llvm-mid", f.environment, "linux", "x64"), /immediately preceding stage/);
    assert.equal(fs.existsSync(valid.stateRoot), false, "stale restore artifacts never create a local state root");

    const wrongBundlePin = {...f.environment, MOM_NATIVE_WINDOWS_LLVM_STATE_BUNDLE_SHA256: "e".repeat(64)};
    assert.throws(() => windowsLlvmCiJobPreflight("llvm-mid", wrongBundlePin, "linux", "x64"), /external SHA-256 pin/);
  });

  test("groups NVPTX, X86, and core into one final-core handoff with the original pinned predecessor", t => {
    const f = fixture(t), input = phaseFixture(f, "llvm-mid"), group = windowsLlvmCiJobPreflight("llvm-mid", f.environment, "linux", "x64");
    assert.equal(group.previousStage, "llvm-amdgpu");
    assert.equal(group.manifestValue.producerJob, "windows_llvm_llvm-amdgpu");
    assert.equal(group.statePhase, "llvm-core");
    assert.deepEqual(group.builderPhases, ["llvm-nvptx", "llvm-x86", "llvm-core"]);
    assert.equal(path.basename(group.archive), `native-windows-llvm-state-${f.suffix}-llvm-core.tar`);
    assert.equal(path.basename(group.incomingStateArchive || ""), `native-windows-llvm-state-${f.suffix}-llvm-amdgpu.tar`);
    assert.equal(input.previousStage, "llvm-amdgpu");

    const next = phaseFixture(f, "clang"), clang = windowsLlvmCiJobPreflight("clang", f.environment, "linux", "x64");
    assert.equal(next.previousStage, "llvm-core");
    assert.equal(clang.manifestValue.producerJob, "windows_llvm_llvm-mid");
    assert.equal(clang.previousProducerStage, "llvm-mid");

    const wrongPrevious = {...f.environment, MOM_NATIVE_WINDOWS_LLVM_STATE_PREVIOUS_STAGE: "configured"};
    assert.throws(() => windowsLlvmCiJobPlan("llvm-mid", wrongPrevious, "linux", "x64"), /immediately preceding/);
    const wrongProducer = {...clang.manifestValue, producerJob: "windows_llvm_llvm-core"};
    fs.writeFileSync(next.incomingStateManifest || "", JSON.stringify(wrongProducer) + "\n");
    f.environment["MOM_NATIVE_WINDOWS_LLVM_STATE_MANIFEST_SHA256"] = sha256(JSON.stringify(wrongProducer) + "\n");
    assert.throws(() => windowsLlvmCiJobPreflight("clang", f.environment, "linux", "x64"), /immediately preceding stage/);
  });

  test("gives llvm-mid the shared bounded deadline without minting a new run window", t => {
    const f = fixture(t), nowMs = Date.now(), groupDeadline = String(Math.ceil((nowMs + 55 * 60 * 1000) / 1000));
    phaseFixture(f, "llvm-mid");
    f.environment["MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH"] = groupDeadline;
    const plan = windowsLlvmCiJobPlan("llvm-mid", f.environment, "linux", "x64", nowMs);
    assert.equal(plan.jobDeadlineEpoch, groupDeadline);
    assert.equal(plan.runDeadlineEpoch, f.runDeadlineEpoch, "the immutable original eight-hour run deadline is carried through");
    assert.ok(plan.budget.remainingMs <= 55 * 60 * 1000 + 1000, "rounding the epoch upward adds less than one second");
    assert.ok(plan.budget.timeoutMs >= 5 * 60 * 1000, "the group preflight retains the builder's minimum phase budget");

    const shortDeadline = {...f.environment,
      MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: String(Math.ceil((nowMs + 5 * 60 * 1000) / 1000))};
    assert.throws(() => windowsLlvmCiJobPlan("llvm-mid", shortDeadline, "linux", "x64", nowMs), /Insufficient shared job time/);

  });

  test("cold grouped transfers retain SDK reserves without extending the job or original run", t => {
    // Run 43827203471 spent 6m18s provisioning and transferring same-run inputs.
    const started = Date.now(), f = fixture(t, started), setupFinished = started + (6 * 60 + 18) * 1000;
    phaseFixture(f, "llvm-mid");
    const planFor = (/** @type {number} */ minutes) => windowsLlvmCiJobPlan("llvm-mid", {...f.environment,
      MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: String(Math.floor((started + minutes * 60 * 1000) / 1000))}, "linux", "x64", setupFinished);
    assert.throws(() => hostSdkDeadline(planFor(28), 6 * 60 * 1000, setupFinished), /Insufficient remaining stage budget/);
    const current = planFor(55), deadline = Number(hostSdkDeadline(current, 6 * 60 * 1000, setupFinished)) * 1000;
    assert.ok(deadline - setupFinished >= 17 * 60 * 1000);
    assert.ok(deadline - setupFinished <= 25 * 60 * 1000, "the SDK's own composition window stays capped");
    assert.ok(deadline <= Number(current.jobDeadlineEpoch) * 1000 - 6 * 60 * 1000);
    assert.ok(deadline <= Number(current.runDeadlineEpoch) * 1000 - 6 * 60 * 1000);
    const limitedRun = {...current, runDeadlineEpoch: String(Math.floor((setupFinished + 22 * 60 * 1000) / 1000))};
    assert.throws(() => hostSdkDeadline(limitedRun, 6 * 60 * 1000, setupFinished), /Insufficient remaining stage budget/,
      "a larger job allowance cannot extend the immutable run deadline or consume the phase reserve");
  });

  test("publishes the grouped checkpoint only after all original builder phases succeed", () => {
    const source = fs.readFileSync(require.resolve("../scripts/run-native-windows-llvm-ci"), "utf8"),
      start = source.indexOf("function runBuilderPhases("), end = source.indexOf("\n/**", start);
    assert.ok(start >= 0 && end > start, "group phase runner remains a small independently testable boundary");
    const runBuilderPhases = /** @type {(phases:readonly string[],build:(phase:string)=>{phase:string},
      publish:(values:{phase:string})=>{phase:string})=>{phase:string}} */ (
          vm.runInNewContext("(" + source.slice(start, end) + ")", {})),
      phases = ["llvm-nvptx", "llvm-x86", "llvm-core"];
    /** @type {string[]} */ const completed = [];
    /** @type {{phase:string}[]} */ const published = [];
    const result = runBuilderPhases(phases, phase => {completed.push(phase); return {phase};}, values => {published.push(values); return values;});
    assert.deepEqual(completed, phases);
    assert.deepEqual(result, {phase: "llvm-core"});
    assert.deepEqual(published, [{phase: "llvm-core"}], "one final builder result is packed once");

    completed.length = 0; published.length = 0;
    assert.throws(() => runBuilderPhases(phases, phase => {
      completed.push(phase);
      if (phase === "llvm-x86") {throw new Error("injected phase failure");}
      return {phase};
    }, values => {published.push(values); return values;}), /injected phase failure/);
    assert.deepEqual(completed, ["llvm-nvptx", "llvm-x86"]);
    assert.deepEqual(published, [], "no grouped state handoff is published after a failed phase");
  });

  test("reports a completed stage as passed even when little job time remains", t => {
    const f = fixture(t), expected = phaseFixture(f, "llvm-mid"),
      source = fs.readFileSync(require.resolve("../scripts/run-native-windows-llvm-ci"), "utf8"),
      start = source.indexOf("if (require.main === module) {");
    assert.ok(start >= 0, "the CLI entry remains extractable for direct behavior coverage");
    // The stage finished with three minutes left, below the six-minute budget needed to start a phase.
    const env = /** @type {NodeJS.ProcessEnv} */ ({...f.environment,
        MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH: String(Math.ceil((Date.now() + 3 * 60 * 1000) / 1000))}),
      cli = {argv: ["node", "run-native-windows-llvm-ci.js", "llvm-mid"], env, exitCode: 0, written: "",
        stdout: {write: (/** @type {string} */ text) => {cli.written += text;}}, stderr: {write: (/** @type {string} */ text) => {cli.written += text;}}},
      main = {};
    vm.runInNewContext(source.slice(start), {process: cli, require: {main}, module: main,
      stages: ["configure", "llvm-amdgpu", "llvm-mid", "clang", "lld", "audit"],
      windowsLlvmCiJobPlan: (/** @type {string} */ stage) => windowsLlvmCiJobPlan(stage, cli.env, "linux", "x64"),
      runWindowsLlvmCiJob: () => {
        cli.env["GITHUB_JOB"] = expected.jobId;
        return {windows_llvm_phase: "build-llvm-core", bundle_sha256: "a".repeat(64), manifest_sha256: "b".repeat(64)};
      }});
    assert.equal(cli.exitCode, 0, cli.written);
    assert.equal(JSON.parse(cli.written).output, "build-llvm-core");
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
  test("configure consumes the producer descriptor before the builder retains it in state", t => {
    const f = fixture(t), plan = f.configure,
      source = fs.readFileSync(require.resolve("../scripts/run-native-windows-llvm-ci"), "utf8");
    assert.notEqual(plan.descriptorState.descriptorOutput, path.join(plan.stateRoot, "descriptor.json"));
    assert.match(source, /descriptorPath = plan\.descriptorState\.descriptorOutput, provenancePath = plan\.descriptorState\.provenanceOutput/);
    fs.mkdirSync(plan.descriptorState.root);
    fs.writeFileSync(plan.descriptorState.descriptorOutput, "{}\n");
    fs.writeFileSync(plan.descriptorState.provenanceOutput, "{}\n");
    const generated = {windows_llvm_descriptor_path: plan.descriptorState.descriptorOutput,
      windows_llvm_descriptor_provenance_path: plan.descriptorState.provenanceOutput,
      windows_llvm_descriptor_sha256: sha256("{}\n"), windows_llvm_descriptor_provenance_sha256: sha256("{}\n")};
    assert.doesNotThrow(() => windowsLlvmDescriptorProducerPins(generated,
      plan.descriptorState.descriptorOutput, plan.descriptorState.provenanceOutput));
    assert.throws(() => windowsLlvmDescriptorProducerPins(generated,
      path.join(plan.stateRoot, "descriptor.json"), plan.descriptorState.provenanceOutput), /inconsistent paths or pins/);
    assert.equal(fs.existsSync(plan.stateRoot), false, "the consumer must not require a not-yet-configured state descriptor");
  });

  /** @param {import('node:test').TestContext} t */
  const descriptorFixture = t => {
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
    const f = descriptorFixture(t), before = [fs.readFileSync(f.descriptorPath, "utf8"), fs.readFileSync(f.provenancePath, "utf8")];
    assert.deepEqual(windowsLlvmDescriptorProducerPins(f.generated, f.descriptorPath, f.provenancePath), {
      descriptorSha: f.generated["windows_llvm_descriptor_sha256"],
      provenanceSha: f.generated["windows_llvm_descriptor_provenance_sha256"],
    });
    assert.deepEqual([fs.readFileSync(f.descriptorPath, "utf8"), fs.readFileSync(f.provenancePath, "utf8")], before);
  });

  test("rejects descriptor or provenance bytes changed after producer pinning", t => {
    const descriptor = descriptorFixture(t), staleDescriptorPin = descriptor.generated["windows_llvm_descriptor_sha256"];
    fs.chmodSync(descriptor.descriptorPath, 0o644);
    fs.writeFileSync(descriptor.descriptorPath, "{\"version\":2}\n");
    const replacementHash = sha256(fs.readFileSync(descriptor.descriptorPath));
    assert.notEqual(replacementHash, staleDescriptorPin);
    assert.equal(sha256(fs.readFileSync(descriptor.descriptorPath)), replacementHash,
      "a caller that re-hashed the changed file would incorrectly accept the replacement as its new pin");
    assert.throws(() => windowsLlvmDescriptorProducerPins(descriptor.generated, descriptor.descriptorPath, descriptor.provenancePath), /external SHA-256 pin/);

    const provenance = descriptorFixture(t), staleProvenancePin = provenance.generated["windows_llvm_descriptor_provenance_sha256"];
    fs.chmodSync(provenance.provenancePath, 0o644);
    fs.writeFileSync(provenance.provenancePath, "{\"status\":\"PASS\",\"changed\":true}\n");
    assert.notEqual(sha256(fs.readFileSync(provenance.provenancePath)), staleProvenancePin);
    assert.throws(() => windowsLlvmDescriptorProducerPins(provenance.generated, provenance.descriptorPath, provenance.provenancePath), /external SHA-256 pin/);
  });

  test("rejects missing pins and substituted output paths", t => {
    const f = descriptorFixture(t), missing = {...f.generated}; delete missing["windows_llvm_descriptor_sha256"];
    assert.throws(() => windowsLlvmDescriptorProducerPins(missing, f.descriptorPath, f.provenancePath), /inconsistent paths or pins/);
    assert.throws(() => windowsLlvmDescriptorProducerPins(f.generated, path.join(f.root, "other.json"), f.provenancePath), /inconsistent paths or pins/);
    assert.throws(() => windowsLlvmDescriptorProducerPins(f.generated, f.descriptorPath, path.join(f.root, "other-provenance.json")), /inconsistent paths or pins/);
  });
});
