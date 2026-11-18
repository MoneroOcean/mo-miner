"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const nativeProcess = require("../runtime/process");

test.describe("Native process startup", {concurrency: false}, () => {
  const names = ["__momCwd", "__momChdir", "__momExitNow", "__momNextTick", "__momOutputOpen", "__momWrite", "__momTCPWrite"];
  /** @type {unknown[]} */
  let saved;
  /** @type {ReturnType<typeof nativeProcess>} */
  let process;
  /** @type {Array<() => unknown>} */
  let ticks;
  let directory = "/initial";
  test.beforeEach(() => {
    saved = names.map((name) => Reflect.get(globalThis, name));
    ticks = []; directory = "/initial";
    globalThis.__momOutputOpen = () => null;
    globalThis.__momCwd = () => directory;
    globalThis.__momChdir = (value) => {directory = value.split("\0")[0] || "";};
    globalThis.__momNextTick = (callback) => {ticks.push(callback);};
    process = nativeProcess({execPath: "/bin/mom", arguments: ["mine", "é-💎", "--", ""],
      platform: "linux", arch: "x64", pid: 123, ppid: 100});
  });
  test.afterEach(() => {
    names.forEach((name, index) => {
      if (saved[index] === undefined) {Reflect.deleteProperty(globalThis, name);} else {Reflect.set(globalThis, name, saved[index]);}
    });
  });
  test("original CLI receives every argument after its synthetic script slot", () => {
    assert.deepEqual(process.argv, ["/bin/mom", "/bin/mom", "mine", "é-💎", "--", ""]);
  });
  test("startup metadata comes from the OS bridge", () => {
    assert.deepEqual([process.execPath, process.platform, process.arch, process.pid, process.ppid],
      ["/bin/mom", "linux", "x64", 123, 100]);
  });
  test("cwd reads current state instead of a startup snapshot", () => {
    assert.equal(process.cwd(), "/initial"); directory = "/other"; assert.equal(process.cwd(), "/other");
  });
  test("chdir delegates Unicode paths without rewriting them", () => {
    process.chdir("/work/é-💎"); assert.equal(directory, "/work/é-💎");
  });
  test("chdir rejects non-string inputs before the OS call and preserves C-string semantics", () => {
    assert.throws(() => process.chdir(/** @type {string} */ (/** @type {unknown} */ (7))), TypeError);
    assert.equal(directory, "/initial");
    process.chdir("/work\0ignored"); assert.equal(directory, "/work");
  });
  test("OS chdir failures are not reported as a successful change", () => {
    globalThis.__momChdir = () => {throw new Error("denied");};
    assert.throws(() => process.chdir("/denied"), /denied/); assert.equal(directory, "/initial");
  });
  test("exit status is initially unset and nullable assignments reset it", () => {
    assert.equal(process.exitCode, undefined); process.exitCode = 2;
    Reflect.set(process, "exitCode", null); assert.equal(process.exitCode, undefined);
    process.exitCode = 2; process.exitCode = undefined; assert.equal(process.exitCode, undefined);
  });
  test("integer exit codes use Node's signed 32-bit wrapping", () => {
    for (const value of [-2147483649, -1, 0, 2147483648, 4294967295, 4294967296]) {
      process.exitCode = value; assert.equal(process.exitCode, value | 0);
    }
  });
  test("nonempty numeric strings preserve Node coercion", () => {
    for (const value of ["1", " 2 ", "0x10", "1e3"]) {
      Reflect.set(process, "exitCode", value); assert.equal(process.exitCode, Number(value) | 0);
    }
  });
  test("invalid exit code types cannot silently mean success", () => {
    const target = process;
    for (const value of ["", " ", false, {}, 1n, Symbol("bad")]) {
      assert.throws(() => Reflect.set(target, "exitCode", value), {code: "ERR_INVALID_ARG_TYPE"});
    }
  });
  test("nonintegral and nonfinite exit codes are rejected", () => {
    const target = process;
    for (const value of [1.5, NaN, Infinity, "1.5", "no"]) {
      assert.throws(() => Reflect.set(target, "exitCode", value), {code: "ERR_OUT_OF_RANGE"});
    }
  });
  test("failed status validation preserves the preceding status", () => {
    process.exitCode = 7; assert.throws(() => Reflect.set(process, "exitCode", {})); assert.equal(process.exitCode, 7);
  });
  test("forced exit uses stored status unless explicitly replaced", () => {
    globalThis.__momExitNow = (code) => {throw Object.assign(new Error("exit"), {status: code});};
    process.exitCode = 9; assert.throws(() => process.exit(), {status: 9});
    assert.throws(() => process.exit("12"), {status: 12});
    assert.throws(() => process.exit(null), {status: 12});
  });
  test("invalid forced-exit arguments do not reach the OS", () => {
    globalThis.__momExitNow = () => {assert.fail("unexpected exit");};
    assert.throws(() => process.exit(false), TypeError);
  });
  test("nextTick defers invocation and preserves callback arguments", () => {
    /** @type {unknown[][]} */
    const observed = [];
    process.nextTick((...values) => observed.push(values), 3, "é");
    assert.deepEqual(observed, []); assert.equal(ticks.length, 1); ticks[0]?.();
    assert.deepEqual(observed, [[3, "é"]]);
  });
  test("invalid nextTick callbacks are rejected before enqueue", () => {
    assert.throws(() => process.nextTick(/** @type {() => void} */ (/** @type {unknown} */ (4))), TypeError);
    assert.equal(ticks.length, 0);
  });
  test("Windows stdout and stderr finish partial Unicode pipe writes before forced exit", () => {
    /** @type {Array<[number, Buffer]>} */
    const writes = [];
    globalThis.__momOutputOpen = () => {assert.fail("Windows stdio must not queue asynchronous pipe output");};
    globalThis.__momWrite = (fd, data) => {
      const part = Buffer.from(data).subarray(0, 2); writes.push([fd, part]); return part.length;
    };
    const target = nativeProcess({execPath: "C:\\mom.exe", arguments: [], platform: "win32", arch: "x64", pid: 1, ppid: 1});
    let called = false;
    target.stdout.write("help é-💎\n", () => {called = true;});
    target.stderr.write("error\n");
    globalThis.__momExitNow = () => {
      assert.equal(Buffer.concat(writes.filter(([fd]) => fd === 1).map(([, data]) => data)).toString(), "help é-💎\n");
      assert.equal(Buffer.concat(writes.filter(([fd]) => fd === 2).map(([, data]) => data)).toString(), "error\n");
      throw new Error("expected exit");
    };
    assert.throws(() => target.exit(1), /expected exit/);
    assert.equal(called, false); ticks[0]?.(); assert.equal(called, true);
  });
  test("Windows synchronous output rejects stalled or failed writes", () => {
    globalThis.__momOutputOpen = () => {assert.fail("unexpected asynchronous pipe");};
    const target = nativeProcess({execPath: "C:\\mom.exe", arguments: [], platform: "win32", arch: "x64", pid: 1, ppid: 1});
    globalThis.__momWrite = () => 0;
    assert.throws(() => target.stdout.write("help"), /Synchronous output made no progress/);
    globalThis.__momWrite = () => {throw Object.assign(new Error("broken pipe"), {code: "EPIPE"});};
    assert.throws(() => target.stderr.write("error"), {code: "EPIPE"});
  });
  test("POSIX pipes retain asynchronous writes and callback acknowledgements", () => {
    /** @type {Map<number, (event: string, value?: unknown) => void>} */
    const callbacks = new Map();
    /** @type {Array<[number, Buffer]>} */
    const writes = [];
    globalThis.__momOutputOpen = (fd, callback) => {callbacks.set(fd, callback); return fd;};
    globalThis.__momTCPWrite = (id, data) => {writes.push([id, Buffer.from(data)]); return true;};
    globalThis.__momWrite = () => {assert.fail("POSIX pipe output must not block on synchronous writes");};
    const target = nativeProcess({execPath: "/mom", arguments: [], platform: "linux", arch: "x64", pid: 1, ppid: 1});
    let called = false;
    target.stdout.write("help", () => {called = true;}); target.stderr.write("error");
    assert.deepEqual(writes, [[1, Buffer.from("help")], [2, Buffer.from("error")]]);
    assert.equal(called, false); callbacks.get(1)?.("write"); assert.equal(called, true);
  });
});
