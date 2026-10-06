"use strict";

const assert = require("node:assert/strict");
const childProcess = require("node:child_process");
const path = require("node:path");
const test = require("node:test");

const root = path.join(__dirname, "..");
const platformBuild = process.platform === "win32" ? "win" : "lin";
const addonPath = process.env["MOM_NATIVE_PATH"] ||
  path.join(root, "build", platformBuild, "Release", "mom.node");
const core = require(addonPath);
const noop = () => undefined;

/** @typedef {{message?: string | undefined, nonce?: string | undefined, result?: string | undefined, job_id?: string | undefined, hash?: string | undefined, [key: string]: string | undefined}} NativeValues */

/** @param {unknown} [values] */
function closeWorker(values) {
  return new Promise((resolve, reject) => {
    const worker = new core.AsyncWorker(noop, resolve, reject);
    worker.sendToCpp("close", values);
  });
}

test("native addon validates callback arguments", () => {
  assert.throws(
    () => new core.AsyncWorker(noop, noop),
    /requires progress, complete, and error callbacks/
  );
  assert.throws(
    () => new core.AsyncWorker(1, noop, noop),
    /callbacks must be functions/
  );
});

test("native addon accepts omitted, undefined, and null message values", {timeout: 5000}, async () => {
  await closeWorker();
  await closeWorker(undefined);
  await closeWorker(null);
});

test("native addon rejects invalid message argument types without aborting", {timeout: 5000}, async () => {
  await new Promise((resolve, reject) => {
    const worker = new core.AsyncWorker(noop, resolve, reject);
    assert.throws(() => worker.sendToCpp(1, {}), /message name must be a string/);
    assert.throws(() => worker.sendToCpp("pause", 1), /values must be an object/);
    worker.sendToCpp("close");
  });
});

test("native addon does not exit after an invalid exitNow code", () => {
  assert.throws(() => core.exitNow("bad"), /code must be a number/);
});

test("native worker lifecycle neither hangs idle nor exits before completion", () => {
  const load = `const core = require(${JSON.stringify(addonPath)}); const noop = () => undefined;`;
  const idle = childProcess.spawnSync(
    process.execPath,
    ["-e", `${load} new core.AsyncWorker(noop, noop, noop);`],
    {timeout: 5000}
  );
  assert.equal(idle.error, undefined);
  assert.equal(idle.status, 0);

  const started = childProcess.spawnSync(
    process.execPath,
    ["-e", `${load}
      const worker = new core.AsyncWorker(
        noop,
        () => console.log("complete"),
        (error) => { throw error; }
      );
      worker.sendToCpp("close");`],
    {encoding: "utf8", timeout: 5000}
  );
  assert.equal(started.error, undefined);
  assert.equal(started.status, 0);
  assert.equal(started.stdout.trim(), "complete");
});
