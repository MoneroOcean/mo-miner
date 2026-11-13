"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const nativeProcess = require("../runtime/process");
const Pipe = require("../runtime/pipe");

test.describe("Native asynchronous standard output", {concurrency: false}, () => {
  const names = ["__momOutputOpen", "__momTCPWrite", "__momTCPEnd", "__momHandleClose", "__momWrite"];
  /** @type {unknown[]} */
  let saved;
  /** @type {Map<number, (event: string, value?: unknown) => void>} */
  let callbacks;
  /** @type {Array<{id: number, bytes: Buffer}>} */
  let written;
  /** @type {Pipe} */
  let stdout;
  /** @type {Pipe} */
  let stderr;
  /** @param {number} id @param {string} event @param {unknown} [value] */
  function dispatch(id, event, value) {
    const callback = callbacks.get(id); if (!callback) {throw new Error("Missing output callback");}
    callback(event, value);
  }
  test.beforeEach(() => {
    saved = names.map((name) => Reflect.get(globalThis, name)); callbacks = new Map(); written = [];
    globalThis.__momOutputOpen = (fd, callback) => {callbacks.set(fd, callback); return fd;};
    globalThis.__momTCPWrite = (id, bytes) => {written.push({id, bytes: Buffer.from(bytes)}); return bytes.length < 65536;};
    globalThis.__momHandleClose = () => undefined;
    globalThis.__momTCPEnd = () => undefined;
    globalThis.__momWrite = () => {assert.fail("Pipe output must not use synchronous fs writes");};
    const process = nativeProcess({execPath: "/mom", arguments: [], platform: "linux", arch: "x64", pid: 1, ppid: 1});
    assert.ok(process.stdout instanceof Pipe); assert.ok(process.stderr instanceof Pipe);
    stdout = process.stdout; stderr = process.stderr;
  });
  test.afterEach(() => {
    names.forEach((name, index) => {
      if (saved[index] === undefined) {Reflect.deleteProperty(globalThis, name);} else {Reflect.set(globalThis, name, saved[index]);}
    });
  });
  test("pipe-backed stdout and stderr use the shared stream queue", () => {
    stdout.write("out"); stderr.write("err");
    assert.deepEqual(written, [{id: 1, bytes: Buffer.from("out")}, {id: 2, bytes: Buffer.from("err")}]);
  });
  test("large writes return backpressure without invoking completion synchronously", () => {
    let complete = false; assert.equal(stdout.write(Buffer.alloc(65536), () => {complete = true;}), false);
    assert.equal(complete, false); assert.equal(stdout.writableLength, 65536);
  });
  test("write acknowledgement updates the exact retained byte count", () => {
    stdout.write("é-💎"); assert.equal(stdout.writableLength, 7); dispatch(1, "write");
    assert.equal(stdout.writableLength, 0);
  });
  test("flush callbacks wait for all preceding queued writes", () => {
    /** @type {string[]} */
    const order = [];
    stdout.write("payload", () => order.push("payload")); stdout.write("", () => order.push("flush"));
    assert.deepEqual(order, []); dispatch(1, "write"); assert.deepEqual(order, ["payload"]);
    dispatch(1, "write"); assert.deepEqual(order, ["payload", "flush"]);
  });
  test("stderr flush ownership is independent from stdout", () => {
    let out = false, err = false;
    stdout.write("out", () => {out = true;}); stderr.write("err", () => {err = true;});
    dispatch(2, "write"); assert.equal(err, true); assert.equal(out, false);
  });
  test("byte views and embedded NULs preserve their exact range", () => {
    stdout.write(Buffer.from("!a\0é!").subarray(1, 5)); assert.equal(written[0]?.bytes.toString("hex"), "6100c3a9");
  });
  test("the stream cap rejects excess data before a partial native write", () => {
    stdout.write(Buffer.alloc(32 * 1024 * 1024)); assert.equal(written.length, 1);
    assert.throws(() => stdout.write("x"), /32 MiB/); assert.equal(written.length, 1);
  });
  test("completion errors are not disguised as successful flushes", () => {
    const result = {error: /** @type {Error | undefined} */ (undefined)};
    stdout.write("pending", (error) => {result.error = error;}); dispatch(1, "close");
    assert.ok(result.error instanceof Error); assert.equal(stdout.writableLength, 0);
  });
  test("native failures are emitted asynchronously and cannot spin in a write retry loop", async () => {
    const expected = Object.assign(new Error("closed"), {code: "EPIPE"});
    globalThis.__momTCPWrite = () => {throw expected;};
    let event, callback;
    stdout.on("error", (error) => {event = error;});
    assert.equal(stdout.write("pending", (error) => {callback = error;}), false);
    assert.equal(event, undefined); assert.equal(callback, undefined);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(event, expected); assert.equal(callback, expected);
  });
  test("file/terminal descriptors retain the synchronous path instead of inventing a pipe", () => {
    globalThis.__momOutputOpen = () => null;
    globalThis.__momWrite = (fd, bytes) => {written.push({id: fd, bytes: Buffer.from(bytes)}); return bytes.length;};
    const process = nativeProcess({execPath: "/mom", arguments: [], platform: "linux", arch: "x64", pid: 1, ppid: 1});
    assert.equal(process.stdout instanceof Pipe, false); process.stdout.write("file");
    assert.equal(written[0]?.bytes.toString(), "file");
  });
});
