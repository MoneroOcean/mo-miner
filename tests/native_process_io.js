"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const Pipe = require("../runtime/pipe");
const {spawn} = require("../runtime/child-process");
const nativeProcess = require("../runtime/process");
const hooks = ["__momPipeRead", "__momStdinOpen", "__momTCPWrite", "__momTCPEnd", "__momHandleClose",
  "__momHandleRef", "__momSpawn", "__momProcessKill", "__momSignalStart", "__momOutputOpen"];
/** @type {unknown[]} */
let saved;
test.beforeEach(() => {saved = hooks.map((name) => Reflect.get(globalThis, name));});
test.afterEach(() => {
  hooks.forEach((name, index) => {
    if (saved[index] === undefined) {Reflect.deleteProperty(globalThis, name);} else {Reflect.set(globalThis, name, saved[index]);}
  });
});

test.describe("Native worker pipe streams", {concurrency: false}, () => {
  /** @type {unknown[]} */
  let actions;
  test.beforeEach(() => {
    actions = [];
    globalThis.__momStdinOpen = () => {actions.push("open"); return 4;};
    globalThis.__momPipeRead = (id, value) => {actions.push(id, value);};
    globalThis.__momTCPEnd = (id) => {actions.push("end", id);};
    globalThis.__momHandleClose = (id) => {actions.push("close", id);};
    globalThis.__momHandleRef = (id, value) => {actions.push("ref", id, value);};
    globalThis.__momTCPWrite = (id, bytes) => {actions.push(id, Buffer.from(bytes)); return bytes.length < 65536;};
  });
  test("unused stdin is lazy and never opens fd 0", () => {
    const pipe = new Pipe(true); assert.equal(pipe.id, null); assert.deepEqual(actions, []);
  });
  test("data subscription opens stdin once and starts reads", () => {
    const pipe = new Pipe(true); pipe.on("data", () => undefined); pipe.resume();
    assert.deepEqual(actions, ["open", 4, true, 4, true]);
  });
  test("end listeners alone do not consume input", () => {new Pipe(true).on("end", () => undefined); assert.deepEqual(actions, []);});
  test("pause stops native reads without destroying the pipe", () => {
    const pipe = new Pipe(true, 5); pipe.pause(); assert.deepEqual(actions, [5, false]); assert.equal(pipe.destroyed, false);
  });
  test("split UTF-8 is decoded and tail is flushed before end", () => {
    /** @type {string[]} */
    const result = []; const pipe = new Pipe(true, 5).setEncoding("utf8");
    pipe.on("data", (value) => result.push(value)); pipe.on("end", () => result.push("end"));
    pipe.receive("data", Uint8Array.from([0xc3]).buffer); pipe.receive("data", Uint8Array.from([0xa9, 0xc3]).buffer);
    pipe.receive("end"); assert.deepEqual(result, ["é", "�", "end"]);
  });
  test("binary reads retain byte ownership and Buffer identity", () => {
    let result; const pipe = new Pipe(true, 5); pipe.on("data", (value) => {result = value;});
    pipe.receive("data", Uint8Array.from([0, 255]).buffer); assert.equal(Buffer.isBuffer(result), true);
    assert.deepEqual(result, Buffer.from([0, 255]));
  });
  test("writes preserve byte ranges and explicit string encoding", () => {
    const pipe = new Pipe(false, 6); pipe.write(Buffer.from("!raw!").subarray(1, 4)); pipe.write("ff", "hex");
    assert.deepEqual(actions, [6, Buffer.from("raw"), 6, Buffer.from([255])]);
  });
  test("backpressure clears only on acknowledged writes", () => {
    const pipe = new Pipe(false, 6); let drained = 0; pipe.on("drain", () => drained++);
    assert.equal(pipe.write(Buffer.alloc(65536)), false); assert.equal(pipe.writableLength, 65536);
    pipe.receive("write"); assert.equal(pipe.writableLength, 0); assert.equal(drained, 1);
  });
  test("write callbacks precede drain and run once", () => {
    /** @type {string[]} */
    const order = []; const pipe = new Pipe(false, 6); pipe.on("drain", () => order.push("drain"));
    pipe.write(Buffer.alloc(65536), () => order.push("callback")); pipe.receive("write"); pipe.receive("write");
    assert.deepEqual(order, ["callback", "drain"]);
  });
  test("end queues shutdown after preceding writes", () => {
    const pipe = new Pipe(false, 6); pipe.end("last"); assert.equal(pipe.writable, false);
    assert.deepEqual(actions, [6, Buffer.from("last"), "end", 6]);
  });
  test("destroy is idempotent and never opens an unused stdin", () => {
    new Pipe(true).destroy().destroy(); assert.deepEqual(actions, []);
    const pipe = new Pipe(true, 5); pipe.destroy().destroy(); assert.deepEqual(actions, ["close", 5]);
  });
  test("close fails pending writes and releases accounting", () => {
    let error; const pipe = new Pipe(false, 6); pipe.write("x", (value) => {error = value;});
    pipe.receive("close"); assert.match(String(error), /closed/); assert.equal(pipe.writes.length, 0);
    assert.equal(pipe.writableLength, 0); assert.equal(pipe.destroyed, true);
  });
  test("ref and unref target only the owned pipe", () => {
    new Pipe(true, 6).ref().unref(); assert.deepEqual(actions, ["ref", 6, true, "ref", 6, false]);
  });
});

test.describe("Native same-executable child workers", {concurrency: false}, () => {
  /** @type {(event: string, value?: unknown) => void} */
  let dispatch;
  /** @type {Array<{args: string[], env: string[]}>} */
  let launches;
  /** @type {Array<[number, string]>} */
  let kills;
  test.beforeEach(() => {
    launches = []; kills = [];
    globalThis.__momSpawn = (args, env, callback) => {
      launches.push({args, env}); dispatch = callback;
      return {id: 1, pid: 100, stdin: 2, stdout: 3, stderr: 4};
    };
    globalThis.__momProcessKill = (id, signal) => {kills.push([id, signal]); return true;};
    globalThis.__momPipeRead = () => undefined;
    globalThis.__momHandleRef = () => undefined;
  });
  test("spawn uses the executable directly without shell or PATH search", () => {
    const worker = spawn(process.execPath, ["", "é-💎", "--"], {env: {FIXTURE: "value"}});
    assert.deepEqual(launches[0], {args: [process.execPath, "", "é-💎", "--"], env: ["FIXTURE=value"]});
    assert.equal(worker.pid, 100);
  });
  test("only explicit worker launches discard the synthetic script slot", () => {
    spawn(process.execPath, [process.execPath, "test", "cn/r"], {env: {MOM_CLUSTER_WORKER: "1"}});
    assert.deepEqual(launches[0]?.args, [process.execPath, "test", "cn/r"]);
  });
  test("non-worker arguments are never heuristically rewritten", () => {
    spawn(process.execPath, [process.execPath, "test"], {env: {}});
    assert.deepEqual(launches[0]?.args, [process.execPath, process.execPath, "test"]);
  });
  test("worker mode requires the original synthetic slot", () => {
    assert.throws(() => spawn(process.execPath, ["test"], {env: {MOM_CLUSTER_WORKER: "1"}}), /script slot/);
    assert.equal(launches.length, 0);
  });
  test("other executables, NUL arguments and shell options are rejected", () => {
    assert.throws(() => spawn("external-program"), /only this executable/);
    assert.throws(() => spawn(process.execPath, ["bad\0arg"]), /string arguments/);
    assert.throws(() => spawn(process.execPath, [], /** @type {object} */ ({shell: true})), /Unsupported/);
    assert.equal(launches.length, 0);
  });
  test("invalid environments cannot cross the native boundary", () => {
    for (const env of [{"BAD=KEY": "x"}, {GOOD: "bad\0value"}, {"": "bad"}]) {
      assert.throws(() => spawn(process.execPath, [], {env}), /environment/);
    }
    assert.equal(launches.length, 0);
  });
  test("undefined environment values are omitted and Unicode stays exact", () => {
    spawn(process.execPath, [], {env: {OMIT: undefined, VALUE: "é-💎"}});
    assert.deepEqual(launches[0]?.env, ["VALUE=é-💎"]);
  });
  test("unsupported stdio is not silently replaced", () => {
    assert.throws(() => spawn(process.execPath, [], {stdio: "inherit"}), /three pipe/);
    assert.equal(launches.length, 0);
  });
  test("kill targets only its native process capability", () => {
    const worker = spawn(process.execPath); assert.equal(worker.kill("SIGTERM"), true);
    assert.equal(worker.killed, true); assert.deepEqual(kills, [[1, "SIGTERM"]]);
  });
  test("exit precedes close and close waits for all three pipes", () => {
    /** @type {unknown[][]} */
    const order = []; const worker = spawn(process.execPath);
    worker.on("exit", (code, signal) => order.push(["exit", code, signal]));
    worker.on("close", (code, signal) => order.push(["close", code, signal]));
    dispatch("exit", {code: 23, signal: null}); dispatch("stdout:close"); dispatch("stderr:close");
    assert.deepEqual(order, [["exit", 23, null]]); dispatch("stdin:close"); dispatch("processClose");
    assert.deepEqual(order, [["exit", 23, null], ["close", 23, null]]);
  });
  test("signal termination reports a signal rather than a successful exit", () => {
    const worker = spawn(process.execPath); dispatch("exit", {code: null, signal: "SIGKILL"});
    assert.equal(worker.signalCode, "SIGKILL"); assert.equal(worker.exitCode, null);
  });
  test("crash signals preserve their names without a platform-number decoder", () => {
    const worker = spawn(process.execPath); dispatch("exit", {code: null, signal: "SIGSEGV"});
    assert.equal(worker.signalCode, "SIGSEGV"); assert.equal(worker.exitCode, null);
  });
  test("failed launch emits error and closes without a fake exit event", () => {
    /** @type {unknown[]} */
    const order = []; const worker = spawn(process.execPath);
    worker.on("error", () => order.push("error")); worker.on("exit", () => order.push("exit"));
    worker.on("close", (code) => order.push(code));
    dispatch("error", Object.assign(new Error("denied"), {errno: -13}));
    for (const name of ["stdin", "stdout", "stderr"]) {dispatch(name + ":close");}
    dispatch("processClose"); assert.deepEqual(order, ["error", -13]);
  });
});

test.describe("Native process signal subscriptions", {concurrency: false}, () => {
  /** @type {Array<[string, number]>} */
  let subscriptions;
  /** @type {Array<() => unknown>} */
  let callbacks;
  /** @type {number[]} */
  let closed;
  /** @type {ReturnType<typeof nativeProcess>} */
  let process;
  test.beforeEach(() => {
    subscriptions = []; callbacks = []; closed = [];
    globalThis.__momOutputOpen = () => null;
    globalThis.__momSignalStart = (name, callback) => {
      callbacks.push(callback); subscriptions.push([name, callbacks.length]); return callbacks.length;
    };
    globalThis.__momHandleClose = (id) => {closed.push(id);};
    process = nativeProcess({execPath: "/mom", arguments: [], platform: "linux", arch: "x64", pid: 1, ppid: 1});
  });
  test("unused signals create no OS handles", () => {assert.deepEqual(subscriptions, []);});
  test("first listener creates one native subscription", () => {
    process.on("SIGTERM", () => undefined); assert.deepEqual(subscriptions, [["SIGTERM", 1]]);
  });
  test("several listeners share that subscription", () => {
    process.on("SIGTERM", () => undefined); process.on("SIGTERM", () => undefined); assert.equal(subscriptions.length, 1);
  });
  test("signal delivery preserves its name argument", () => {
    let observed; process.on("SIGTERM", (name) => {observed = name;}); callbacks[0]?.(); assert.equal(observed, "SIGTERM");
  });
  test("removing the final listener closes the OS handle", () => {
    const handler = () => undefined; process.on("SIGTERM", handler); process.removeListener("SIGTERM", handler);
    assert.deepEqual(closed, [1]);
  });
  test("removing only one listener keeps the shared OS handle", () => {
    const handler = () => undefined; process.on("SIGTERM", handler); process.on("SIGTERM", () => undefined);
    process.removeListener("SIGTERM", handler); assert.deepEqual(closed, []);
  });
  test("once listeners close the subscription after delivery", () => {
    let calls = 0; process.once("SIGTERM", () => calls++); callbacks[0]?.(); callbacks[0]?.();
    assert.equal(calls, 1); assert.deepEqual(closed, [1]);
  });
  test("subscribing again creates a fresh capability", () => {
    process.once("SIGTERM", () => undefined); callbacks[0]?.(); process.on("SIGTERM", () => undefined);
    assert.deepEqual(subscriptions, [["SIGTERM", 1], ["SIGTERM", 2]]);
  });
  test("ordinary message events do not become OS signal handlers", () => {
    let value; process.on("message", (message) => {value = message;}); process.emit("message", 42);
    assert.equal(value, 42); assert.deepEqual(subscriptions, []);
  });
  test("OS subscription errors propagate before a listener is installed", () => {
    globalThis.__momSignalStart = () => {throw new Error("OS error");};
    assert.throws(() => process.on("SIGTERM", () => undefined), /OS error/); assert.equal(process.listenerCount("SIGTERM"), 0);
  });
});
