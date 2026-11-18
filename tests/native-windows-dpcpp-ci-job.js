"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {spawnSync} = require("node:child_process"),
  test = require("node:test"), assert = require("node:assert/strict");
const {fileHash} = require("../scripts/build-source-inputs");
const {stageWindowsDpcppComponent, windowsDpcppComponentInputs} = require("../scripts/native-windows-dpcpp-component");
const {fixture} = require("./common/native-windows-component");
const repo = path.resolve(__dirname, ".."), workflowFile = path.join(repo, ".github/workflows/build-windows-dpcpp-job.yml"),
  workflow = fs.readFileSync(workflowFile, "utf8");

/** Extract a reviewed literal Node heredoc so the production body itself is syntax-checked/executed.
 * @param {string} step */
function nodeBody(step) {
  const marker = "      - name: " + step, position = workflow.indexOf(marker), heredoc = "          node <<'NODE'\n",
    start = workflow.indexOf(heredoc, position) + heredoc.length, end = workflow.indexOf("\n          NODE", start);
  assert.ok(position >= 0 && start >= heredoc.length && end > start, "production Node heredoc exists: " + step);
  return workflow.slice(start, end).split("\n").map(line => line.slice(10)).join("\n");
}

/** Extract the deadline guard's real shell script. */
function deadlineBody() {
  const marker = "      - name: Start within the original Windows JIT run deadline", position = workflow.indexOf(marker),
    heredoc = "        run: |\n", start = workflow.indexOf(heredoc, position) + heredoc.length, end = workflow.indexOf("\n      - ", start);
  assert.ok(position >= 0 && start >= heredoc.length && end > start, "production deadline guard exists");
  return workflow.slice(start, end).split("\n").map(line => line.slice(10)).join("\n");
}

/** @param {string} body @param {NodeJS.ProcessEnv} env */
function execNode(body, env) {
  return spawnSync(process.execPath, ["-e", body], {cwd: repo, env, encoding: "utf8", timeout: 30000, maxBuffer: 2 << 20});
}

test.describe("Windows original static DPC++ reusable job", {skip: process.platform !== "linux" && "Linux-hosted Windows build workflow harness"}, () => {
  test("declares all external handoff pins and keeps the job bounded, build-only and cache-free", () => {
    for (const input of ["windows_jit_run_deadline_epoch", "source_bundle_sha256", "dpcpp_bundle_sha256",
      "dpcpp_manifest_sha256", "windows_jit_bundle_sha256", "linux_tools_bundle_sha256", "linux_tools_handoff_sha256",
      "windows_io_bundle_sha256", "windows_io_handoff_sha256", "windows_hermes_bundle_sha256", "windows_hermes_handoff_sha256",
      "linux_hermes_bundle_sha256"]) {
      assert.match(workflow, new RegExp("^      " + input + ":\\n        required: true$", "m"), input + " is an explicit reusable-job input");
    }
    assert.match(workflow, /^ {2}workflow_call:$/m); assert.doesNotMatch(workflow, /^\s+(?:push|pull_request|workflow_dispatch):/m);
    assert.doesNotMatch(workflow, /windows_llvm_run_deadline_epoch|DPCPP_STATIC_LLVM_RUN_DEADLINE/);
    assert.match(workflow, /^ {4}timeout-minutes: 60$/m); assert.match(workflow, /^ {6}MOM_BUILD_JOBS: "2"$/m);
    assert.doesNotMatch(workflow, /actions\/cache(?:\/|@)|^\s*cache:/m);
    assert.match(workflow, /MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH=/);
    assert.match(workflow, /MOM_NATIVE_WINDOWS_HOST_SDK_WORK_ROOT=/);
    assert.match(workflow, /MOM_NATIVE_WINDOWS_DPCPP_STATIC_DEADLINE_EPOCH=/);
    assert.doesNotMatch(workflow, /dummytool/i);
  });

  test("production build glue is syntactically valid and invokes the existing source-JIT isolator and builders", () => {
    const body = nodeBody("Re-admit inputs, reacquire the local Windows SDK, and run original DPC++ producers"),
      checked = spawnSync(process.execPath, ["--check"], {input: body, encoding: "utf8", timeout: 30000});
    assert.equal(checked.status, 0, checked.stderr || checked.stdout);
    for (const text of ["ci.verifyLinuxSources", "windowsGpuDpcppManifest(dpcpp.manifest, pins.dpcppManifest, pins.source)",
      "ci.auditDpcppCudaInputs", "linuxLlvmToolHandoffInputs", "windowsControllerHandoffInputs",
      "nativeHermesInputs", "ci.buildWindowsHostSdkSourcesCi()", "ci.composeWindowsLlvmSdkDescriptor", "composeWindowsApplicationConfig",
      "scripts/isolate-native-windows-jit.js", "scripts/build-native-windows-sdk.js", "scripts/prepare-native-windows-adapters.js",
      "scripts/native-windows-dpcpp-component.js"]) {assert.ok(body.includes(text), text);}
    assert.match(body, /allowedImports = \["advapi32\.dll", "cfgmgr32\.dll", "dbghelp\.dll", "kernel32\.dll", "ntdll\.dll", "ole32\.dll",\s*"oleaut32\.dll", "setupapi\.dll", "shell32\.dll", "shlwapi\.dll", "version\.dll"\]/);
    assert.match(workflow, /Only the self-contained relocatable DPC\+\+ component was retained/);
  });

  test("job-start shell honors the immutable Windows JIT deadline and checks pins before exporting paths", t => {
    const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-windows-dpcpp-deadline-")),
      output = path.join(root, "github-env"); fs.writeFileSync(output, "", {flag: "wx"}); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const now = Math.floor(Date.now() / 1000), suffix = "20261109-1", good = Object.fromEntries([
      "SOURCE_BUNDLE_SHA256", "DPCPP_BUNDLE_SHA256", "DPCPP_MANIFEST_SHA256", "JIT_BUNDLE_SHA256", "TOOLS_BUNDLE_SHA256",
      "TOOLS_HANDOFF_SHA256", "IO_BUNDLE_SHA256", "IO_HANDOFF_SHA256", "HERMES_BUNDLE_SHA256", "HERMES_HANDOFF_SHA256",
      "LINUX_HERMES_BUNDLE_SHA256",
    ].map(key => ["MOM_NATIVE_WINDOWS_DPCPP_STATIC_" + key, "a".repeat(64)]));
    const baseEnv = {...process.env, ...good, RUNNER_TEMP: root, GITHUB_ENV: output, GITHUB_RUN_ID: suffix.split("-")[0], GITHUB_RUN_ATTEMPT: "1",
      MOM_NATIVE_WINDOWS_DPCPP_STATIC_JIT_RUN_DEADLINE_EPOCH: String(now + 900)};
    const accepted = spawnSync("/bin/bash", ["-euc", deadlineBody()], {cwd: repo, env: baseEnv, encoding: "utf8", timeout: 30000});
    assert.equal(accepted.status, 0, accepted.stderr || accepted.stdout);
    const envFile = fs.readFileSync(output, "utf8");
    assert.match(envFile, new RegExp("MOM_NATIVE_WINDOWS_DPCPP_STATIC_DEADLINE_EPOCH=" + (now + 900)));
    assert.match(envFile, /MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR=.*native-windows-host-sdk-20261109-1/);

    fs.writeFileSync(output, "untouched\n");
    const rejected = spawnSync("/bin/bash", ["-euc", deadlineBody()], {cwd: repo,
      env: {...baseEnv, MOM_NATIVE_WINDOWS_DPCPP_STATIC_SOURCE_BUNDLE_SHA256: "bad"}, encoding: "utf8", timeout: 30000});
    assert.notEqual(rejected.status, 0); assert.equal(fs.readFileSync(output, "utf8"), "untouched\n");
  });

  test("the actual seal heredoc produces and revalidates a transport tar from a structural fixture", t => {
    const f = fixture(t), suffix = `${Date.now()}${process.pid}-1`, name = "native-windows-dpcpp-static-" + suffix,
      root = path.join("/tmp", name), runnerTemp = fs.mkdtempSync(path.join("/var/tmp", "mom-windows-dpcpp-runner-")),
      archive = path.join(runnerTemp, name + ".tar"), output = path.join(runnerTemp, "github-output");
    assert.equal(fs.existsSync(root), false); t.after(() => {if (fs.existsSync(root)) {fs.rmSync(root, {recursive: true, force: true});}});
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    stageWindowsDpcppComponent(f.links, {path: "/original/adapter-receipt.json", sha256: "1".repeat(64)}, root);
    const before = windowsDpcppComponentInputs(root), runId = suffix.split("-")[0], env = {...process.env, RUNNER_TEMP: runnerTemp, GITHUB_RUN_ID: runId,
      GITHUB_RUN_ATTEMPT: "1", GITHUB_OUTPUT: output, MOM_NATIVE_WINDOWS_DPCPP_STATIC_COMPONENT_ROOT: root};
    fs.writeFileSync(output, "", {flag: "wx"});
    const sealed = execNode(nodeBody("Revalidate and seal only the relocatable DPC++ component"), env);
    assert.equal(sealed.status, 0, sealed.stderr || sealed.stdout);
    const values = Object.fromEntries(fs.readFileSync(output, "utf8").trim().split("\n").map(line => line.split("=")));
    assert.equal(values.bundle_sha256, fileHash(archive)); assert.equal(values.component_manifest_sha256, before.files[path.join(root, "component.json")]);
    assert.equal(values.artifact_name, name + ".tar");
    const ci = require("../scripts/prepare-native-ci"), members = ci.listState(archive, [name]).map(file => file.replace(/\/$/, "")).sort(),
      metadata = JSON.parse(fs.readFileSync(path.join(root, "component.json"), "utf8")), rows = [...metadata.link.libraries, ...metadata.link.wholeArchives, ...metadata.link.objects],
      expected = [name, name + "/component.json", name + "/lib", name + "/objects", ...rows.map(row => name + "/" + row)].sort();
    assert.deepEqual(members, expected);
  });

  test("the production seal rejects a redirected root without creating an artifact or output", t => {
    const f = fixture(t), suffix = `${Date.now()}${process.pid}-1`, name = "native-windows-dpcpp-static-" + suffix,
      root = path.join("/tmp", name), runnerTemp = fs.mkdtempSync(path.join("/var/tmp", "mom-windows-dpcpp-runner-")),
      archive = path.join(runnerTemp, name + ".tar"), output = path.join(runnerTemp, "github-output");
    assert.equal(fs.existsSync(root), false); t.after(() => {if (fs.existsSync(root)) {fs.rmSync(root, {recursive: true, force: true});}});
    t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
    stageWindowsDpcppComponent(f.links, {path: "/original/adapter-receipt.json", sha256: "1".repeat(64)}, root);
    fs.writeFileSync(output, "", {flag: "wx"});
    const result = execNode(nodeBody("Revalidate and seal only the relocatable DPC++ component"), {...process.env, RUNNER_TEMP: runnerTemp,
      GITHUB_RUN_ID: suffix.split("-")[0], GITHUB_RUN_ATTEMPT: "1", GITHUB_OUTPUT: output,
      MOM_NATIVE_WINDOWS_DPCPP_STATIC_COMPONENT_ROOT: path.join(f.root, "redirected")});
    assert.notEqual(result.status, 0); assert.equal(fs.existsSync(archive), false); assert.equal(fs.readFileSync(output, "utf8"), "");
  });
});
