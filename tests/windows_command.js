"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const windowsCommand = require("../scripts/windows-command");

test("Windows command builder quotes safe arguments and rejects unsafe expansion", () => {
  assert.equal(windowsCommand.quoteWindowsCmdArg("C:\\Program Files\\mom.cmd"),
    '"C:\\Program Files\\mom.cmd"');
  assert.equal(windowsCommand.quoteWindowsCmdArg("C:\\Program Files\\mom\\"),
    '"C:\\Program Files\\mom\\\\"');
  assert.equal(windowsCommand.buildWindowsCmd([
    "C:\\Program Files\\mom.cmd", "bench", "fake", "gpu1 & whoami",
  ]), '"C:\\Program Files\\mom.cmd" bench fake "gpu1 & whoami"');
  assert.deepEqual(windowsCommand.windowsCmdArgs([
    "C:\\Program Files\\mom.cmd", "bench",
  ]), ["/d", "/v:off", "/s", "/c", '""C:\\Program Files\\mom.cmd" bench"']);
  for (const unsafe of ["gpu%0", 'gpu"0', "gpu\n0"]) {
    assert.throws(() => windowsCommand.buildWindowsCmd(["mom.cmd", unsafe]),
      /cannot safely preserve/);
  }
});

test("Windows command builder preserves arguments through cmd.exe", {
  skip: process.platform !== "win32",
}, () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-command-"));
  const capture = path.join(temp, "capture.js");
  const batch = path.join(temp, "capture args.cmd");
  const expected = ["plain", "two words", "amp&ersand", "pipe|value", "paren(value)",
    "caret^value", "bang!value", "C:\\path with space\\", ""];
  try {
    fs.writeFileSync(capture,
      '"use strict";\nprocess.stdout.write(JSON.stringify(process.argv.slice(2)));\n');
    fs.writeFileSync(batch,
      '@echo off\r\nsetlocal DisableDelayedExpansion\r\n"%MOM_TEST_NODE%" "%~dp0capture.js" %*\r\n');
    const result = spawnSync(process.env["ComSpec"] || "cmd.exe",
      windowsCommand.windowsCmdArgs([batch, ...expected]), {
        encoding: "utf8",
        env: {...process.env, MOM_TEST_NODE: process.execPath},
        windowsHide: true,
        windowsVerbatimArguments: true,
        timeout: 30000,
      });
    assert.equal(result.status, 0, result.stderr || result.error?.message);
    assert.deepEqual(JSON.parse(result.stdout), expected);
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});
