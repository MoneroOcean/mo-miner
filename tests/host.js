"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const path = require("node:path");
const test = require("node:test");

test.describe("Native host correctness", {
  skip: process.platform !== "linux",
}, () => {
  test("production helpers and consensus references", {
    timeout: 5 * 60 * 1000,
  }, () => {
    const root = path.join(__dirname, "..");
    const result = spawnSync("bash", [path.join(root, "scripts/test-host.sh")], {
      cwd: root,
      encoding: "utf8",
      timeout: 5 * 60 * 1000,
      maxBuffer: 2 * 1024 * 1024,
    });
    const diagnostics = [result.error?.message, result.stdout, result.stderr].filter(Boolean).join("\n");
    assert.equal(result.status, 0, `Host correctness runner failed (${result.signal || "no signal"}):\n${diagnostics}`);
    assert.match(result.stdout, /Host correctness tests passed/);
  });
});
