"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const test = require("node:test");

test.describe("Shared Node/native worker lifecycle", {concurrency: false}, () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "mom-worker-"));
  const windows = process.platform === "win32";
  const executable = path.join(temporary, windows ? "worker.exe" : "worker");
  test.before(() => {
    const source = path.join(__dirname, "native", "worker_lifecycle.cpp");
    const result = spawnSync(process.env["CXX"] || (windows ? "cl.exe" : "c++"), windows
      ? ["/nologo", "/std:c++17", "/utf-8", "/EHsc", "/W4", "/WX", "/D_CRT_SECURE_NO_WARNINGS", source,
        `/Fe:${executable}`, `/Fo:${path.join(temporary, "worker.obj")}`]
      : ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread", source, "-o", executable],
    {encoding: "utf8", timeout: 60000});
    assert.equal(result.status, 0, result.error?.message || result.stdout + result.stderr);
  });
  test.after(() => fs.rmSync(temporary, {recursive: true, force: true}));
  for (const name of ["idle", "echo", "payload", "order", "threads", "postClose", "destructor",
    "retainedPayload", "retryStart", "string", "standard", "unknown", "repeatedClose", "independent", "nullChannel", "queue", "terminalPost"]) {
    test(name, () => {
      const result = spawnSync(executable, [name], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, result.error?.message || result.stdout + result.stderr);
    });
  }
});
