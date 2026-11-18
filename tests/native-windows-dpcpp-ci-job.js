"use strict";
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"),
  test = require("node:test"), assert = require("node:assert/strict");
const {fileHash} = require("../scripts/build-source-inputs");
const ci = require("../scripts/native-ci-tar");
const {stageWindowsDpcppComponent, windowsDpcppComponentInputs} = require("../scripts/native-windows-dpcpp-component");
const {jobEnvironment, sealComponent} = require("../scripts/ci/windows-dpcpp");
const {fixture} = require("./common/native-windows-component");
const repo = path.resolve(__dirname, ".."), script = path.join(repo, "scripts/ci/windows-dpcpp.js");
const pinKeys = ["SOURCE_BUNDLE_SHA256", "DPCPP_BUNDLE_SHA256", "DPCPP_MANIFEST_SHA256", "JIT_BUNDLE_SHA256", "TOOLS_BUNDLE_SHA256",
  "TOOLS_HANDOFF_SHA256", "IO_BUNDLE_SHA256", "IO_HANDOFF_SHA256", "HERMES_BUNDLE_SHA256", "HERMES_HANDOFF_SHA256", "LINUX_HERMES_BUNDLE_SHA256"];

/** Valid preflight environment with every external pin present and a deadline 900 s after now.
 * @param {number} now @returns {NodeJS.ProcessEnv} */
function preflightEnv(now) {
  return {...Object.fromEntries(pinKeys.map(key => ["MOM_NATIVE_WINDOWS_DPCPP_STATIC_" + key, "a".repeat(64)])),
    RUNNER_TEMP: "/runner/temp", MOM_NATIVE_CI_SUFFIX: "20261109-1", MOM_CI_DEADLINE_EPOCH: String(now + 900)};
}

/** Stage a fresh same-run component under /tmp plus a private runner temp with an empty GITHUB_OUTPUT.
 * @param {import("node:test").TestContext} t */
function sealFixture(t) {
  const f = fixture(t), suffix = `${Date.now()}${process.pid}-1`, name = "native-windows-dpcpp-static-" + suffix,
    root = path.join("/tmp", name), runnerTemp = fs.mkdtempSync(path.join("/var/tmp", "mom-windows-dpcpp-runner-")),
    archive = path.join(runnerTemp, name + ".tar"), output = path.join(runnerTemp, "github-output");
  assert.equal(fs.existsSync(root), false);
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  t.after(() => fs.rmSync(runnerTemp, {recursive: true, force: true}));
  stageWindowsDpcppComponent(f.links, {path: "/original/adapter-receipt.json", sha256: "1".repeat(64)}, root);
  fs.writeFileSync(output, "", {flag: "wx"});
  const env = {RUNNER_TEMP: runnerTemp, MOM_NATIVE_CI_SUFFIX: suffix.split("-")[0] + "-1", GITHUB_OUTPUT: output,
    MOM_NATIVE_WINDOWS_DPCPP_STATIC_COMPONENT_ROOT: root};
  return {f, name, root, archive, output, env};
}

test.describe("Windows original static DPC++ job script", {skip: process.platform !== "linux" && "Linux-hosted Windows build job harness"}, () => {
  test("preflight accepts the job deadline and derives the host-SDK paths and deadline", () => {
    const now = Math.floor(Date.now() / 1000), {deadline, pins, hostSdk} = jobEnvironment(preflightEnv(now), now);
    assert.equal(deadline, now + 900); assert.equal(pins.linuxHermes, "a".repeat(64));
    assert.deepEqual(hostSdk, {
      MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH: String(now + 900),
      MOM_NATIVE_WINDOWS_HOST_SDK_WORK_ROOT: "/runner/temp/mom-native-windows-host-sdk-20261109-1",
      MOM_NATIVE_WINDOWS_HOST_SDK_ARCHIVE_DIR: "/runner/temp/mom-native-windows-host-sdk-20261109-1/archives",
      MOM_NATIVE_WINDOWS_HOST_SDK_OUTPUT_DIR: "/runner/temp/mom-native-windows-host-sdk-20261109-1/native-windows-host-sdk-20261109-1",
      MOM_NATIVE_WINDOWS_HOST_SDK_METADATA_TAR: "/runner/temp/native-windows-host-sdk-20261109-1-metadata.tar",
    });
  });

  test("preflight rejects malformed, expired and too-short deadlines and every missing or malformed pin", () => {
    const now = Math.floor(Date.now() / 1000), good = preflightEnv(now);
    for (const epoch of ["", "12345", String(now + 900) + "0", "abcdefghij"]) {
      assert.throws(() => jobEnvironment({...good, MOM_CI_DEADLINE_EPOCH: epoch}, now), /Malformed MOM_CI_DEADLINE_EPOCH/);
    }
    for (const epoch of [now, now - 60]) {
      assert.throws(() => jobEnvironment({...good, MOM_CI_DEADLINE_EPOCH: String(epoch)}, now), /expired/);
    }
    assert.throws(() => jobEnvironment({...good, MOM_CI_DEADLINE_EPOCH: String(now + 119)}, now), /Insufficient time/);
    for (const key of pinKeys) {
      const name = "MOM_NATIVE_WINDOWS_DPCPP_STATIC_" + key;
      assert.throws(() => jobEnvironment({...good, [name]: "bad"}, now), /SHA-256 pin/, name);
      assert.throws(() => jobEnvironment({...good, [name]: "A".repeat(64)}, now), /SHA-256 pin/, name);
      assert.throws(() => jobEnvironment({...good, [name]: undefined}, now), /SHA-256 pin/, name);
    }
    assert.throws(() => jobEnvironment({...good, RUNNER_TEMP: undefined}, now), /RUNNER_TEMP/);
  });

  test("the CLI preflight exits non-zero with a native-ci message on a bad pin and rejects unknown commands", () => {
    const now = Math.floor(Date.now() / 1000), base = {...process.env, ...preflightEnv(now)};
    const accepted = spawnSync(process.execPath, [script, "preflight"], {cwd: repo, env: base, encoding: "utf8", timeout: 30000});
    assert.equal(accepted.status, 0, accepted.stderr || accepted.stdout);
    const rejected = spawnSync(process.execPath, [script, "preflight"], {cwd: repo,
      env: {...base, MOM_NATIVE_WINDOWS_DPCPP_STATIC_SOURCE_BUNDLE_SHA256: "bad"}, encoding: "utf8", timeout: 30000});
    assert.equal(rejected.status, 1); assert.match(rejected.stderr, /^\[native-ci\] Missing or malformed external/m);
    const unknown = spawnSync(process.execPath, [script, "publish"], {cwd: repo, env: base, encoding: "utf8", timeout: 30000});
    assert.equal(unknown.status, 1); assert.match(unknown.stderr, /^\[native-ci\] Usage:/m);
  });

  test("seal produces and revalidates a transport tar from a structural fixture", t => {
    const {name, root, archive, output, env} = sealFixture(t), before = windowsDpcppComponentInputs(root);
    sealComponent(env);
    const values = Object.fromEntries(fs.readFileSync(output, "utf8").trim().split("\n").map(line => line.split("=")));
    assert.equal(values.bundle_sha256, fileHash(archive)); assert.equal(values.component_manifest_sha256, before.files[path.join(root, "component.json")]);
    assert.equal(values.artifact_name, name + ".tar");
    const members = ci.listState(archive, [name]).map(file => file.replace(/\/$/, "")).sort(),
      metadata = JSON.parse(fs.readFileSync(path.join(root, "component.json"), "utf8")),
      rows = [...metadata.link.libraries, ...metadata.link.wholeArchives, ...metadata.link.objects],
      expected = [name, name + "/component.json", name + "/lib", name + "/objects", ...rows.map(row => name + "/" + row)].sort();
    assert.deepEqual(members, expected);
  });

  test("seal rejects a redirected root without creating an artifact or output", t => {
    const {f, archive, output, env} = sealFixture(t);
    assert.throws(() => sealComponent({...env, MOM_NATIVE_WINDOWS_DPCPP_STATIC_COMPONENT_ROOT: path.join(f.root, "redirected")}),
      /fresh same-run seal paths/);
    assert.equal(fs.existsSync(archive), false); assert.equal(fs.readFileSync(output, "utf8"), "");
  });

  test("seal refuses an existing run-attempt archive without overwriting it", t => {
    const {archive, output, env} = sealFixture(t);
    fs.writeFileSync(archive, "stale\n", {flag: "wx"});
    assert.throws(() => sealComponent(env), /fresh same-run seal paths/);
    assert.equal(fs.readFileSync(archive, "utf8"), "stale\n"); assert.equal(fs.readFileSync(output, "utf8"), "");
  });
});
