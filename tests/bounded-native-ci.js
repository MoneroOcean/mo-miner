"use strict";

const assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {describe, test} = require("node:test");
const {spawnSync} = require("node:child_process");
const {parseMemAvailableBytes, verifyNativeCiHostHeadroom, verifyNativeCiMemoryEnvelope} = require("../scripts/native-ci-memory-envelope");
const {tempDir} = require("./common/temp-dir");
const repo = path.resolve(__dirname, "..");

/** @param {import('node:test').TestContext} t */
function cgroupFixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-ci-cgroup-")), directory = path.join(root, "system.slice", "run-build.scope");
  fs.mkdirSync(directory, {recursive: true});
  fs.writeFileSync(path.join(directory, "memory.max"), "12884901888\n", {flag: "wx"});
  fs.writeFileSync(path.join(directory, "memory.swap.max"), "0\n", {flag: "wx"});
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, directory};
}

describe("Native CI memory containment", () => {
  test("admits only the exact 12 GiB, swap-zero cgroup-v2 envelope", t => {
    const fixture = cgroupFixture(t);
    assert.equal(verifyNativeCiMemoryEnvelope("0::/system.slice/run-build.scope\n", fixture.root), fixture.directory);
    assert.throws(() => verifyNativeCiMemoryEnvelope("2:memory:/legacy\n", fixture.root), /unified cgroup-v2/);
    assert.throws(() => verifyNativeCiMemoryEnvelope("0::/../outside\n", fixture.root), /invalid unified cgroup path/);
    fs.writeFileSync(path.join(fixture.directory, "memory.max"), "max\n");
    assert.throws(() => verifyNativeCiMemoryEnvelope("0::/system.slice/run-build.scope\n", fixture.root), /memory.max envelope/);
  });

  test("fails closed on missing or redirected cgroup controls", t => {
    const fixture = cgroupFixture(t);
    fs.unlinkSync(path.join(fixture.directory, "memory.swap.max"));
    assert.throws(() => verifyNativeCiMemoryEnvelope("0::/system.slice/run-build.scope\n", fixture.root), /memory.swap.max/);
    const outside = path.join(fixture.root, "outside");
    fs.mkdirSync(outside); fs.symlinkSync(outside, path.join(fixture.root, "linked"));
    assert.throws(() => verifyNativeCiMemoryEnvelope("0::/linked\n", fixture.root), /escaped the cgroup-v2 mount/);
  });

  test("requires strict host MemAvailable headroom before creating the producer scope", () => {
    const requiredBytes = 12 * 1024 ** 3 + 512 * 1024 ** 2,
      atLimit = `MemTotal: 16777216 kB\nMemAvailable: ${requiredBytes / 1024} kB\n`,
      aboveLimit = `MemTotal: 16777216 kB\nMemAvailable: ${requiredBytes / 1024 + 1} kB\n`;
    assert.equal(parseMemAvailableBytes(aboveLimit), requiredBytes + 1024);
    assert.throws(() => verifyNativeCiHostHeadroom(atLimit), /more than 12 GiB plus 512 MiB/);
    assert.deepEqual(verifyNativeCiHostHeadroom(aboveLimit), {availableBytes: requiredBytes + 1024, requiredBytes});
    assert.throws(() => parseMemAvailableBytes("MemAvailable: 10 kB\nMemAvailable: 11 kB\n"), /one MemAvailable/);
    assert.throws(() => parseMemAvailableBytes("MemAvailable: unlimited kB\n"), /malformed/);
    assert.throws(() => parseMemAvailableBytes("MemAvailable: 9007199254740992 kB\n"), /overflows/);
  });

  test("restores multiline environment and literal argv only in the non-root child", {
    skip: process.platform !== "linux" || process.getuid?.() === 0,
  }, t => {
    const root = tempDir(t, "mom-ci-env-pipe-");
    // This fixture bypasses only the kernel check to isolate shell transport. The verifier's
    // real cgroup admission is covered above and by the separately recorded live-scope smoke.
    const verifier = path.join(root, "verifier-node");
    fs.writeFileSync(verifier, "#!/bin/sh\n[ -z \"$NODE_OPTIONS$NODE_PATH\" ] || exit 23\nexit 0\n", {mode: 0o700, flag: "wx"});
    const env = {PATH: root + ":/usr/bin:/bin", MOM_CI_ENV_FIXTURE: "line1\nline2", CI_LITERAL: "$(not-a-command)"};
    const input = Object.entries(env).map(([key, value]) => key + "=" + value + "\0").join("");
    const result = spawnSync("/bin/bash", [path.join(repo, "scripts/run-bounded-native-ci.sh"),
      "--verify", String(process.getuid?.()), verifier, "--", process.execPath, "-e",
      "console.log(JSON.stringify({value:process.env.MOM_CI_ENV_FIXTURE,literal:process.env.CI_LITERAL,arg:process.argv[1],uid:process.getuid()}))",
      "literal $(not-a-command)"], {input, encoding: "utf8"});
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(result.stdout), {value: "line1\nline2", literal: "$(not-a-command)",
      arg: "literal $(not-a-command)", uid: process.getuid?.()});
  });

  test("producer PATH and Node preloads cannot replace the verifier", {
    skip: process.platform !== "linux" || process.getuid?.() === 0,
  }, t => {
    const root = tempDir(t, "mom-ci-verifier-env-");
    const verifier = path.join(root, "verifier-node");
    // Isolate verifier dispatch/environment; actual kernel admission is tested separately.
    fs.writeFileSync(verifier, "#!/bin/sh\n[ -z \"$NODE_OPTIONS$NODE_PATH\" ] || exit 23\nprintf 'verified\\n'\n", {mode: 0o700, flag: "wx"});
    fs.writeFileSync(path.join(root, "node"), "#!/bin/sh\nprintf 'BYPASSED\\n'\nexit 0\n", {mode: 0o700, flag: "wx"});
    const result = spawnSync("/bin/bash", [path.join(repo, "scripts/run-bounded-native-ci.sh"),
      "--verify", String(process.getuid?.()), verifier, "--", "/usr/bin/env"], {
      env: {...process.env, NODE_OPTIONS: "--require=/missing-preload", NODE_PATH: root},
      input: "PATH=" + root + ":/usr/bin:/bin\0NODE_OPTIONS=--require=/producer-only\0", encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /^verified\n/);
    assert.match(result.stdout, /NODE_OPTIONS=--require=\/producer-only/);
    assert.doesNotMatch(result.stdout, /BYPASSED/);
  });
});
