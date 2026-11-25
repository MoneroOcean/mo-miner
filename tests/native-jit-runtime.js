"use strict";

// Optional real compiler proof, not GPU dispatch. The caller supplies containment and an
// SDK-free environment; ordinary source tests neither build nor launch this expensive fixture.
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const {spawnSync} = require("node:child_process");
const fixture = process.env["MOM_NATIVE_JIT_RUNTIME_FIXTURE"];
const bridge = process.env["MOM_NATIVE_JIT_BRIDGE_FIXTURE"];

/** No installed SDK, compiler PATH or inherited include/plugin configuration may assist.
 * The caller still supplies a contained SDK-free filesystem and host memory limit.
 * @param {Record<string,string>} [extra] */
function fixtureEnvironment(extra = {}) {
  return {PATH: process.platform === "win32" ? process.env["SystemRoot"] + "\\System32" : "/nonexistent", HOME: os.tmpdir(),
    TMPDIR: os.tmpdir(), TEMP: os.tmpdir(), TMP: os.tmpdir(), LC_ALL: "C", ...(process.platform === "win32" ? {SystemRoot: process.env["SystemRoot"]} : {}), ...extra};
}

test.describe("Original in-process SYCL compiler runtime", {skip: !fixture && "Set MOM_NATIVE_JIT_RUNTIME_FIXTURE for the contained SDK-free compiler proof"}, () => {
  const variants = ["spirv", "sm_61", "sm_120"].map(target => [target]);
  for (const target of ["sm_61", "sm_120"]) {
    for (const algorithm of ["kawpow", "firopow", "evrprogpow", "meowpow"]) {variants.push([target, algorithm]);}
  }
  for (const args of variants) {
    test(args.join(" ") + " preserves source, cache, error recovery and image lifecycle", {timeout: 310000}, () => {
      assert.ok(fixture && path.isAbsolute(fixture), "Select an absolute original compiler fixture");
      assert.ok(fs.lstatSync(fixture).isFile(), "The fixture must be a regular executable, not a symlink");
      const run = spawnSync(fixture, args, {env: fixtureEnvironment(), encoding: "utf8", timeout: 300000, maxBuffer: 2 << 20});
      assert.equal(run.error, undefined, "Compiler fixture failed to execute: " + run.error?.message);
      assert.equal(run.signal, null, "Compiler fixture was interrupted");
      assert.equal(run.status, 0, "Original compiler failed:\n" + (run.stderr || "") + (run.stdout || ""));
      assert.ok(run.stdout.includes("Maintained static SYCL JIT: " + args.join(" ") + " source/cache/error-recovery/hash/binary-lifecycle passed"), "Missing complete runtime proof");
    });
  }
});

test.describe("Static SYCL runtime source-JIT bridge", {skip: !bridge && "Set MOM_NATIVE_JIT_BRIDGE_FIXTURE for the contained SDK-free runtime proof"}, () => {
  for (const target of ["spirv", "sm_61", "sm_120"]) {
    test(target + " preserves runtime binding, registration and binary retirement", {timeout: 310000}, t => {
      assert.ok(bridge && path.isAbsolute(bridge), "Select an absolute original runtime bridge fixture");
      assert.ok(fs.lstatSync(bridge).isFile(), "The runtime fixture must be a regular executable, not a symlink");
      // The fixture's expectation and upstream compiler target are separate controls. Setting
      // only MOM_DPCPP_JIT_TARGET does not actually select that PTX architecture.
      // CI mounts only this file into the contained runtime proof, so no shared test helper is used.
      const cache = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-bridge-")); t.after(() => fs.rmSync(cache, {recursive: true, force: true}));
      const env = fixtureEnvironment({MOM_DPCPP_JIT_TARGET: target, SYCL_JIT_AMDGCN_PTX_TARGET_CPU: target, SYCL_CACHE_PERSISTENT: "1", SYCL_CACHE_DIR: cache,
        SYCL_JIT_AMDGCN_PTX_TARGET_FEATURES: target === "sm_120" ? "+sm_120,+ptx87" : "+sm_61,+ptx63"});
      const run = spawnSync(bridge, [], {env, encoding: "utf8", timeout: 300000, maxBuffer: 2 << 20});
      assert.equal(run.error, undefined, "Runtime fixture failed to execute: " + run.error?.message);
      assert.equal(run.signal, null, "Runtime fixture was interrupted");
      assert.equal(run.status, 0, "Static runtime bridge failed:\n" + (run.stderr || "") + (run.stdout || ""));
      assert.ok(run.stdout.includes("Static DPC++ host-JIT: " + target + " binding/source/cache/recovery/registration/retirement passed"), "Missing complete runtime bridge proof");
    });
  }
});
