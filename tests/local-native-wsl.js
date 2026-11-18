"use strict";
const {describe, test} = require("node:test"), assert = require("node:assert/strict");
const {wslText, selectWsl2Distribution, prepareWindowsWslRelease} = require("../scripts/build-local-native-wsl");

describe("Local Windows WSL release boundary", () => {
  test("decodes redirected WSL output without accepting control characters", () => {
    assert.equal(wslText(Buffer.from("\uFEFFUbuntu\r\n", "utf16le")), "Ubuntu\r\n");
    assert.equal(wslText(Buffer.from("Ubuntu\n")), "Ubuntu\n");
    assert.throws(() => wslText("Ubuntu\0"), /Invalid WSL/);
  });
  test("selects only an explicit or unique WSL2 distribution", () => {
    assert.equal(selectWsl2Distribution("Ubuntu\n", " NAME STATE VERSION\n* Ubuntu Stopped 2\n"), "Ubuntu");
    assert.equal(selectWsl2Distribution("Ubuntu\nBuild (24.04)\n", " Ubuntu Stopped 1\n Build (24.04) Running 2\n", "Build (24.04)"), "Build (24.04)");
    assert.throws(() => selectWsl2Distribution("Ubuntu\nOther\n", ""), /Multiple/);
    assert.throws(() => selectWsl2Distribution("Ubuntu\n", " Ubuntu Running 1\n"), /WSL1/);
    assert.throws(() => selectWsl2Distribution("Ubuntu\n", " Ubuntu Running 2\n", "Missing"), /installed/);
    assert.equal(selectWsl2Distribution("Ubuntu\nOther\n", " Ubuntu Running 2\n Other Stopped 2\n", "ubuntu"), "Ubuntu");
    assert.throws(() => selectWsl2Distribution("Ubuntu\nubuntu\n", ""), /installed/);
  });
  test("round-trips a checkout with spaces through literal arguments and proves the runtime", () => {
    const checkout = "C:\\Work\\MoM repo", calls = /** @type {string[][]} */ ([]),
      deps = {platform: /** @type {NodeJS.Platform} */ ("win32"), arch: "x64", realpath: (/** @type {string} */ file) => file,
        spawn: /** @type {typeof import('node:child_process').spawnSync} */ (/** @type {unknown} */ ((/** @type {string} */ exe, /** @type {string[]} */ args, /** @type {any} */ options) => {
          assert.equal(exe, "C:\\Windows\\System32\\wsl.exe"); assert.equal(options.shell, false); calls.push(args);
          const out = args[0] === "--list" ? args[1] === "--quiet" ? "Build\r\n" : "* Build Stopped 2\r\n" :
            args.includes("/usr/bin/uname") ? "6.6.87.2-microsoft-standard-WSL2\n" : args.includes("-u") ? "/mnt/c/Work/MoM repo\n" : checkout + "\n";
          return {status: 0, signal: null, stdout: Buffer.from(out, "utf16le")};
        }))};
    const result = prepareWindowsWslRelease(checkout, {SystemRoot: "C:\\Windows"}, deps);
    assert.equal(result.distribution, "Build"); assert.equal(result.checkout, "/mnt/c/Work/MoM repo");
    assert.deepEqual(calls[3], ["--distribution", "Build", "--exec", "/usr/bin/wslpath", "-u", checkout]);
    assert.throws(() => prepareWindowsWslRelease(checkout, {}, {...deps, platform: "linux"}), /Windows x64/);
    assert.throws(() => prepareWindowsWslRelease("C:\\Work\\Other", {}, deps), /round-trip/);
    const failed = /** @type {typeof import('node:child_process').spawnSync} */ (/** @type {unknown} */ (() => ({status: 1, stdout: Buffer.alloc(0)})));
    assert.throws(() => prepareWindowsWslRelease(checkout, {}, {...deps, spawn: failed}), /preflight failed/);
    const original = deps.spawn, wrongKernel = /** @type {typeof original} */ (/** @type {unknown} */ (
      (/** @type {string} */ exe, /** @type {string[]} */ args, /** @type {any} */ options) => args.includes("/usr/bin/uname") ?
        {status: 0, stdout: Buffer.from("Linux ordinary kernel\n")} : original(exe, args, options)));
    assert.throws(() => prepareWindowsWslRelease(checkout, {}, {...deps, spawn: wrongKernel}), /WSL2 kernel/);
  });
});
