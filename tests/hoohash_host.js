"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const test = require("node:test");

const vectors = require("./vectors/memory_hard");

test("HooHash strict host verifier matches the canonical rejected-case vector", {
  skip: process.platform === "win32",
}, () => {
  const root = path.join(__dirname, "..");
  const vector = vectors.find(({name}) => name.includes("hoohash canonical-host") &&
    name.includes("intensity=1"));
  assert.ok(vector, "canonical HooHash vector is missing");

  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hoohash-host-"));
  const executable = path.join(temporary, "hoohash-host");
  try {
    const compile = spawnSync(process.env["CC"] || "cc", [
      "-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off", "-fno-builtin",
      path.join(root, "tests/reference/hoohash_host.c"),
      path.join(root, "sycl/hoohash/host_math.c"),
      "-ldl", "-lpthread", "-Wl,--no-as-needed", "-lm", "-Wl,--as-needed", "-o", executable,
    ], {encoding: "utf8"});
    assert.equal(compile.status, 0, compile.stderr);

    const verify = spawnSync(executable, [], {
      encoding: "utf8",
      input: `${vector.job.blob_hex}\n${vector.expected}\n`,
    });
    assert.equal(verify.status, 0, "strict HooHash host verifier disagrees with canonical output");
  } finally {
    fs.rmSync(temporary, {recursive: true, force: true});
  }
});
