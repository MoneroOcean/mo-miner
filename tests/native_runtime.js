"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const nodeNet = require("node:net");
const {Socket, connect, isIP} = require("../runtime/net");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const esbuild = require("esbuild");

test.describe("Native TCP adapter", {concurrency: false}, () => {
  /** @type {(event: string, value?: unknown) => void} */
  let dispatch;
  /** @type {Array<{id: number, data: Buffer}>} */
  let writes;
  /** @type {Array<number | boolean>} */
  let actions;
  /** @type {number | null | undefined} */
  let connectTimeout;
  /** @type {number} */
  let connectCalls;
  /** @type {Record<string, unknown>} */
  const saved = {};
  const names = ["__momTCPConnect", "__momTCPWrite", "__momHandleClose", "__momHandleRef", "__momTCPEnd", "__momIsIP"];

  test.beforeEach(() => {
    for (const name of names) {saved[name] = Reflect.get(globalThis, name);}
    writes = []; actions = []; connectTimeout = undefined; connectCalls = 0;
    globalThis.__momTCPConnect = function(host, port, callback) {
      assert.equal(host, "127.0.0.1"); assert.equal(port, 1234); dispatch = callback;
      connectTimeout = /** @type {number | null | undefined} */ (arguments[3]);
      connectCalls++; return 7;
    };
    globalThis.__momTCPWrite = (id, data) => {
      writes.push({id, data: Buffer.from(data)}); return data.byteLength < 65536;
    };
    globalThis.__momHandleClose = (id) => {actions.push(id);};
    globalThis.__momHandleRef = (id, ref) => {actions.push(id, ref);};
    globalThis.__momTCPEnd = (id) => {actions.push(id);};
  });
  test.afterEach(() => {
    for (const name of names) {
      if (saved[name] === undefined) {
        Reflect.deleteProperty(globalThis, name);
      } else {Reflect.set(globalThis, name, saved[name]);}
    }
  });

  /** @param {{autoSelectFamilyAttemptTimeout?: number | null}=} options */
  function socket(options = {}) {
    return new Socket({host: "127.0.0.1", port: 1234, ...options});
  }

  test("isIP matches native literal semantics without coercing nonstrings", () => {
    const literals = [
      "0.0.0.0", "127.0.0.1", "255.255.255.255", "::", "::1", "2001:db8::1",
      "::ffff:192.0.2.1", "2001:db8::192.0.2.1", "fe80::1%eth0", "fe80::1%zone.with:chars",
      "01.2.3.4", "127.0.0.01", "256.0.0.1", " 127.0.0.1", "127.0.0.1 ", "[::1]",
      "fe80::1%", "fe80::1%bad_zone", "2001:db8:::1", "2001:db8::1\0tail",
    ];
    const calls = [];
    globalThis.__momIsIP = (value) => {calls.push(value); return nodeNet.isIP(value);};
    for (const literal of literals) {assert.equal(isIP(literal), nodeNet.isIP(literal), literal);}
    for (const value of [null, undefined, 1, {}, [], {toString() {throw new Error("must not coerce");}}]) {
      assert.equal(isIP(value), 0);
    }
    assert.equal(calls.length, literals.length);
  });

  test("CLI bracketed IPv6 parsing uses the native isIP adapter", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-cli-ip-"));
    try {
      const bundle = path.join(directory, "cli.cjs");
      esbuild.buildSync({entryPoints: [path.join(__dirname, "../miner/cli.js")], outfile: bundle,
        bundle: true, platform: "node", format: "cjs", external: ["node:*"],
        alias: {"node:net": path.join(__dirname, "../runtime/net.js")}});
      globalThis.__momIsIP = (value) => nodeNet.isIP(value);
      const parseArgs = require(bundle);
      const opt = {pools: [], pool_ids: {primary: null}, job: {algo: null}};
      /** @type {string[]} */
      const help = [];
      /** @param {string} url @param {number} port @param {boolean} tls
       * @param {string} login @param {string} pass */
      const poolCreate = (url, port, tls, login, pass) => ({url, port, is_tls: tls, login, pass});
      /** @param {string} [message] @returns {never} */
      const printHelp = (message) => {help.push(message || ""); throw new Error(message);};
      /** @param {string | null | undefined} algo */
      const normalizeAlgoName = (algo) => algo;
      const parsed = parseArgs({
        o: {
          is_config_file: () => false, load_config: () => undefined, opt_help: {}, parse_opt: () => false,
          pool_create: poolCreate, print_help: printHelp,
        }, opt, normalizeAlgoName,
      })(["node", "mom.js", "mine", "[fe80::1%eth0]:3333tls", "wallet"], {result_hash_hex: null});
      assert.equal(parsed, "mine");
      assert.deepEqual(opt.pools[0], {url: "fe80::1%eth0", port: 3333, is_tls: true, login: "wallet", pass: "", use_subscribe: false});
      assert.deepEqual(help, []);
    } finally {
      fs.rmSync(directory, {recursive: true, force: true});
    }
  });

  test("forwards and clamps the native family-attempt timeout", () => {
    const client = socket({autoSelectFamilyAttemptTimeout: 1});
    assert.equal(connectTimeout, 10); assert.equal(connectCalls, 1);
    dispatch("connect"); assert.equal(client.connecting, false);
  });

  test("rejects invalid family-attempt timeouts before allocating a native handle", () => {
    for (const value of [0, -1, 2147483648, NaN, Infinity, /** @type {number} */ (/** @type {unknown} */ ("1000"))]) {
      assert.throws(() => socket({autoSelectFamilyAttemptTimeout: value}), {code: "ERR_OUT_OF_RANGE"});
    }
    assert.equal(connectCalls, 0);
  });

  test("connect callback waits for the native completion event", () => {
    let connected = false;
    const client = connect({host: "127.0.0.1", port: 1234}, () => {connected = true;});
    assert.equal(connected, false); assert.equal(client.connecting, true);
    dispatch("connect"); assert.equal(connected, true); assert.equal(client.connecting, false);
  });
  test("queued writes retain order before a connect listener writes", () => {
    const client = socket(); client.write("first");
    client.on("connect", () => client.write("second"));
    assert.equal(writes.length, 0); dispatch("connect");
    assert.deepEqual(writes.map((entry) => entry.data.toString()), ["first", "second"]);
  });
  test("writes preserve supplied byte ranges and encoding", () => {
    const client = socket(); dispatch("connect");
    client.write(new Uint8Array([1, 2, 3]).subarray(1)); client.write("é", "latin1");
    assert.deepEqual(writes.map((entry) => [...entry.data]), [[2, 3], [0xe9]]);
  });
  test("writes preserve raw bytes for every ArrayBuffer view", () => {
    const client = socket(); dispatch("connect");
    const bytes = Uint8Array.from([0xaa, 0xbb, 0xcc]);
    const words16 = new Uint16Array([0x1234]);
    const words32 = new Uint32Array([0x01020304]);
    client.write(new DataView(bytes.buffer, 1, 2));
    client.write(words16); client.write(words32);
    assert.deepEqual(writes.map((entry) => entry.data.toString("hex")), [
      "bbcc",
      Buffer.from(words16.buffer, words16.byteOffset, words16.byteLength).toString("hex"),
      Buffer.from(words32.buffer, words32.byteOffset, words32.byteLength).toString("hex"),
    ]);
  });
  test("writes reject arrays and bare ArrayBuffers", () => {
    const client = socket(); dispatch("connect");
    const array = /** @type {string | Uint8Array} */ (/** @type {unknown} */ ([1, 2]));
    const buffer = /** @type {string | Uint8Array} */ (/** @type {unknown} */ (new ArrayBuffer(2)));
    assert.throws(() => client.write(array), {code: "ERR_INVALID_ARG_TYPE"});
    assert.throws(() => client.write(buffer), {code: "ERR_INVALID_ARG_TYPE"});
    assert.equal(writes.length, 0);
  });
  test("binary reads expose a Buffer rather than its backing ArrayBuffer", () => {
    const client = socket(); let received;
    client.on("data", (data) => {received = data;});
    dispatch("data", new Uint8Array([1, 2, 3]).buffer);
    assert.ok(Buffer.isBuffer(received)); assert.deepEqual([...received], [1, 2, 3]);
  });
  test("split UTF-8 characters do not emit empty string chunks", () => {
    const client = socket().setEncoding("utf8");
    /** @type {string[]} */
    const received = [];
    client.on("data", (data) => received.push(data));
    dispatch("data", new Uint8Array([0xc3]).buffer);
    assert.deepEqual(received, []);
    dispatch("data", new Uint8Array([0xa9]).buffer);
    assert.deepEqual(received, ["é"]);
  });
  test("EOF flushes incomplete UTF-8 before the end event", () => {
    const client = socket().setEncoding("utf8");
    /** @type {string[]} */
    const received = [];
    client.on("data", (data) => received.push(data)); client.on("end", () => received.push("end"));
    dispatch("data", new Uint8Array([0xc3]).buffer); dispatch("end");
    assert.deepEqual(received, ["�", "end"]);
  });
  test("backpressure drains only after acknowledged writes", () => {
    const client = socket(); dispatch("connect");
    /** @type {string[]} */
    const order = [];
    client.on("drain", () => order.push("drain"));
    assert.equal(client.write(Buffer.alloc(65536), () => order.push("callback")), false);
    assert.equal(client.writableLength, 65536); dispatch("write");
    assert.equal(client.writableLength, 0); assert.deepEqual(order, ["callback", "drain"]);
  });
  test("each successful write callback runs once", () => {
    const client = socket(); dispatch("connect"); let count = 0;
    client.write("a", () => count++); client.write("b", () => count++);
    dispatch("write"); dispatch("write"); dispatch("write"); assert.equal(count, 2);
  });
  test("close fails pending callbacks with the original native error", () => {
    const client = socket(); const error = Object.assign(new Error("refused"), {code: "ECONNREFUSED"});
    let received; client.on("error", () => undefined);
    client.write("pending", (value) => {received = value;});
    dispatch("error", error); dispatch("close", true);
    assert.equal(received, error); assert.equal(client.destroyed, true);
    assert.equal(client.connecting, false); assert.equal(client.writableLength, 0);
  });
  test("end waits for queued writes before requesting native shutdown", () => {
    const client = socket(); client.end("final");
    assert.deepEqual(actions, []); dispatch("connect");
    assert.equal(writes[0]?.data.toString(), "final"); assert.deepEqual(actions, [7]);
  });
  test("closed writes return false and report errors asynchronously", async () => {
    const client = socket().destroy(); let error;
    const completed = new Promise((resolve) => {
      assert.equal(client.write("late", (value) => {error = value; resolve(undefined);}), false);
    });
    assert.equal(error, undefined); await completed;
    assert.equal(/** @type {NodeJS.ErrnoException | undefined} */ (error)?.code, "ERR_STREAM_DESTROYED");
  });
  test("native write failures become asynchronous socket errors", async () => {
    const client = socket(); dispatch("connect");
    const error = new Error("native failure");
    globalThis.__momTCPWrite = () => {throw error;};
    const failed = new Promise((resolve) => client.once("error", resolve));
    assert.equal(client.write("data"), false); assert.equal(await failed, error);
    assert.equal(client.destroyed, true); assert.equal(client.writableLength, 0);
  });
  test("queue limits are checked before passing writes to native code", () => {
    const client = socket(); client.writableLength = 32 * 1024 * 1024;
    assert.throws(() => client.write("x"), /32 MiB/); assert.equal(writes.length, 0);
  });
  test("reference changes forward only the private handle and requested state", () => {
    const client = socket(); assert.equal(client.unref().ref(), client);
    assert.deepEqual(actions, [7, false, 7, true]);
  });
});
