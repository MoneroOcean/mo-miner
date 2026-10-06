"use strict";

const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
/** @typedef {{filename: string, paths: string[], exports: typeof import("../helper.js"),
 * _compile: (source: string, filename: string) => void}} FixtureModule */
/** @type {{new(filename: string, parent: unknown): FixtureModule,
 * _nodeModulePaths: (directory: string) => string[]}} */
const Module = /** @type {{new(filename: string, parent: unknown): FixtureModule,
 * _nodeModulePaths: (directory: string) => string[]}} */ (/** @type {unknown} */ (require("node:module")));
const {EventEmitter} = require("node:events");
const {PassThrough} = require("node:stream");
const cp = require("node:child_process");
const cluster = /** @type {typeof import("node:cluster").default} */
  (/** @type {unknown} */ (require("node:cluster")));
const {MAX_PEARL_PROOF_BYTES, MAX_PEARL_PROOF_BASE64,
  MAX_PROOF_EVENT_OVERHEAD} = require("../helper/worker-protocol");
const prefix = "MOM_WORKER_MESSAGE ";

/** @typedef {import("node:events").EventEmitter & {id: number,
 * stdout: PassThrough, stderr: PassThrough, stdin: PassThrough,
 * killed?: boolean, kill: () => boolean, isConnected: () => boolean,
 * send: () => boolean}} FixtureWorker */
/** @typedef {{type: string, thread_id: number,
 * value: {plain_proof?: string, message?: string, fatal?: boolean, [key: string]: unknown}}} FixtureEvent */
/** @param {import("node:test").TestContext} t @param {string} [transport] */
function fixture(t, transport = "subprocess") {
  const saved = {spawn: cp.spawn, fork: cluster.fork, env: process.env,
    opt: global.opt, stdout: process.stdout.write};
  assert.ok(cluster.workers);
  const clusterWorkers = cluster.workers;
  process.env = {...process.env};
  delete process.env["MOM_CLUSTER_WORKER"];
  delete process.env["MOM_GPU_BACKEND"];
  if (transport === "subprocess") {process.env["MOM_USE_SUBPROCESS_WORKERS"] = "1";} else {
    delete process.env["MOM_USE_SUBPROCESS_WORKERS"];
  }
  // The isolated helper fixture reads only log_level; it does not configure a live miner.
  global.opt = /** @type {MinerOptions} */ (/** @type {unknown} */ ({log_level: 0}));
  /** @type {FixtureWorker[]} */
  const workers = [];
  function worker() {
    const w = /** @type {FixtureWorker} */ (new EventEmitter());
    w.id = 100 + workers.length;
    w.stdout = new PassThrough();
    w.stderr = new PassThrough();
    w.stdin = new PassThrough();
    w.kill = () => {w.killed = true; return true;};
    w.isConnected = () => true;
    w.send = () => true;
    workers.push(w);
    if (transport === "cluster") {
      assert.equal(Object.hasOwn(clusterWorkers, w.id), false);
      clusterWorkers[w.id] = /** @type {import("node:cluster").Worker} */
        (/** @type {unknown} */ (w));
    }
    return w;
  }
  cp.spawn = /** @type {typeof cp.spawn} */ (/** @type {unknown} */ (worker));
  cluster.fork = /** @type {typeof cluster.fork} */ (/** @type {unknown} */ (worker));
  const filename = path.join(__dirname, "..", "helper.js");
  const source = process.env["MOM_IPC_BASELINE"] ? path.join(__dirname, "..", "before", "helper.js") : filename;
  const mod = new Module(filename, module);
  mod.filename = filename;
  mod.paths = Module._nodeModulePaths(path.dirname(filename));
  mod._compile(fs.readFileSync(source, "utf8"), filename);
  const h = mod.exports;
  h.get_dev_threads = () => 2;
  h.get_thread_dev = () => "cpu";
  /** @type {FixtureEvent[]} */
  const events = [];
  let visible = "";
  process.stdout.write = (chunk) => {visible += chunk; return true;};
  h.recreate_threads("cpu", (message) => events.push(message));
  t.after(() => {
    h.closeWorkers(0);
    for (const w of workers) {
      if (transport === "cluster") {delete clusterWorkers[w.id];}
      w.stdout.destroy(); w.stderr.destroy(); w.stdin.destroy();
    }
    cp.spawn = saved.spawn; cluster.fork = saved.fork;
    process.env = saved.env; global.opt = saved.opt;
    process.stdout.write = saved.stdout;
  });
  /** @param {unknown} event @param {boolean} [split] */
  function wire(event, split = false) {
    const text = prefix + JSON.stringify(event) + "\n";
    if (split) {
      /** @type {FixtureWorker} */ (workers[0]).stdout.write(text.slice(0, 3));
      for (let start = 3; start < text.length; start += 8192) {
        /** @type {FixtureWorker} */ (workers[0]).stdout.write(text.slice(start, start + 8192));
      }
    } else {/** @type {FixtureWorker} */ (workers[0]).stdout.write(text);}
  }
  // recreate_threads is fixed to two logical workers above; tests address only those indices.
  return {h, workers: /** @type {[FixtureWorker, FixtureWorker]} */
    (/** @type {unknown} */ (workers)), events, wire, visible: () => visible};
}

/** @param {unknown} proof @param {number} [thread_id] @param {UnknownRecord} [extra] */
function result(proof, thread_id = 0, extra = {}) {
  return {type: "result", thread_id, value: {plain_proof: proof,
    worker_id: "worker-private", job_id: "job", job_token: "token", ...extra}};
}

for (const transport of ["cluster", "subprocess"]) {
  for (const failure of ["device", "first-env", "last-env"]) {
    test(`${transport} preserves active workers after failed ${failure} preparation`, t => {
      const f = fixture(t, transport);
      const targets = f.h.messageWorkers({type: "pause"});
      assert.equal(targets.length, 2);
      const closeWorkers = f.h.closeWorkers;
      let closed = 0;
      f.h.closeWorkers = (forceAfterMs) => {
        closed++;
        const retired = closeWorkers(forceAfterMs);
        if (closed === 1) {assert.deepEqual(retired, targets);}
        return retired;
      };
      const getDevThreads = f.h.get_dev_threads;
      if (failure === "device") {
        f.h.get_dev_threads = require("../helper").get_dev_threads;
      }
      /** @type {FixtureEvent[]} */
      const replacements = [];
      assert.throws(() => f.h.recreate_threads(failure === "device" ? "cpu^0" : "cpu",
        message => replacements.push(message), (_dev, index) => {
          if (index === (failure === "first-env" ? 0 : 1)) {
            throw new Error("incompatible donation tuning");
          }
          return {};
        }));
      f.h.get_dev_threads = getDevThreads;
      assert.equal(closed, 0);
      assert.equal(f.workers.length, 2);
      assert.deepEqual(f.h.messageWorkers({type: "pause"}), targets);
      const event = result("AA==");
      if (transport === "cluster") {f.workers[0].emit("message", event);} else {f.wire(event);}
      assert.deepEqual(f.events, [event]);

      f.h.recreate_threads("cpu", message => replacements.push(message));
      assert.equal(closed, 1);
      assert.equal(f.workers.length, 4);
      if (transport === "cluster") {f.workers[0].emit("message", event);} else {f.wire(event);}
      assert.deepEqual(f.events, [event]);
      const current = f.h.messageWorkers({type: "pause"})[0];
      assert.ok(current);
      if (current.type === "cluster") {current.worker.emit("message", event);} else {
        assert.ok(current.worker.stdout instanceof PassThrough);
        current.worker.stdout.write(prefix + JSON.stringify(event) + "\n");
      }
      assert.deepEqual(replacements, [event]);
    });
  }
}

test("max-K source-sized result survives fragmented subprocess framing", t => {
  const f = fixture(t);
  const bytes = 433 + 1040 * (65536 / 32) + 32 * 124;
  const proof = Buffer.alloc(bytes).toString("base64");
  assert.ok(proof.length > 1024 * 1024);
  assert.ok(bytes < MAX_PEARL_PROOF_BYTES);
  f.wire(result(proof), true);
  assert.equal(f.events.length, 1);
  assert.equal(/** @type {FixtureEvent} */ (f.events[0]).type, "result");
  assert.equal(/** @type {FixtureEvent} */ (f.events[0]).value.plain_proof, proof);
});

test("existing eight-MiB proof admission budget survives the worker envelope", t => {
  const f = fixture(t);
  const proof = Buffer.alloc(MAX_PEARL_PROOF_BYTES).toString("base64");
  assert.equal(proof.length, MAX_PEARL_PROOF_BASE64);
  f.wire(result(proof), true);
  assert.equal(f.events.length, 1);
  assert.equal(/** @type {FixtureEvent} */ (f.events[0]).type, "result");
  assert.equal(Buffer.from(/** @type {string} */ (/** @type {FixtureEvent} */ (f.events[0]).value.plain_proof), "base64").length, MAX_PEARL_PROOF_BYTES);
});

for (const mode of ["proof-cap", "envelope-cap", "non-proof", "pending-ordinary", "pending-ipc"]) {
  test(`oversize ${mode} is rejected once without payload disclosure`, t => {
    const f = fixture(t);
    if (mode === "proof-cap") {f.wire(result("A".repeat(MAX_PEARL_PROOF_BASE64 + 4)));}
    if (mode === "envelope-cap") {
      f.wire(result("A".repeat(1024 * 1024), 0, {other: "x".repeat(MAX_PROOF_EVENT_OVERHEAD)}));
    }
    if (mode === "non-proof") {
      f.wire({type: "hashrate", thread_id: 0, value: {other: "x".repeat(1024 * 1024)}});
    }
    if (mode === "pending-ordinary") {f.workers[0].stdout.write("x".repeat(1024 * 1024 + 1));}
    if (mode === "pending-ipc") {
      f.workers[0].stdout.write(prefix + "x".repeat(MAX_PEARL_PROOF_BASE64 + MAX_PROOF_EVENT_OVERHEAD));
    }
    f.wire(result("payload-private"));
    assert.equal(f.events.length, 1);
    assert.equal(/** @type {FixtureEvent} */ (f.events[0]).type, "error");
    assert.equal(/** @type {FixtureEvent} */ (f.events[0]).value.fatal, true);
    assert.equal(/** @type {FixtureEvent} */ (f.events[0]).thread_id, 0);
    assert.equal((/** @type {string} */ (/** @type {FixtureEvent} */ (f.events[0]).value.message)).includes("payload-private"), false);
    assert.equal(f.workers[0].killed, true);
  });
}

for (const transport of ["cluster", "subprocess"]) {
  test(`${transport} rejects a current worker impersonating another in-range origin`, t => {
    const f = fixture(t, transport);
    const spoof = result("AA==", 1);
    if (transport === "cluster") {f.workers[0].emit("message", spoof);} else {f.wire(spoof);}
    assert.equal(f.events.length, 1);
    assert.equal(/** @type {FixtureEvent} */ (f.events[0]).type, "error");
    assert.equal(/** @type {FixtureEvent} */ (f.events[0]).thread_id, 0);
    assert.equal(f.workers[0].killed, true);
    if (transport === "cluster") {f.workers[0].emit("message", spoof);} else {f.wire(spoof);}
    assert.equal(f.events.length, 1);
  });
  test(`${transport} accepts its logical index independently of cluster worker ID`, t => {
    const f = fixture(t, transport);
    const own = {type: "hashrate", thread_id: 1, value: {hashrate: 1}};
    if (transport === "cluster") {f.workers[1].emit("message", own);} else {
      f.workers[1].stdout.write(prefix + JSON.stringify(own) + "\n");
    }
    assert.deepEqual(f.events, [own]);
  });
}

for (const mode of ["complete", "partial", "malformed"]) {
  test(`${mode} IPC does not enter unexpected-exit stdout diagnostics`, t => {
    const f = fixture(t);
    f.workers[0].stdout.write("ordinary worker diagnostic\n");
    const privateLine = prefix + JSON.stringify(result("proof-private-secret"));
    if (mode === "complete") {f.workers[0].stdout.write(privateLine + "\n");} else if (mode === "partial") {
      f.workers[0].stdout.write(privateLine);
    } else {f.workers[0].stdout.write(prefix + "{bad-private-secret\n");}
    f.workers[0].stdout.emit("end");
    f.workers[0].emit("exit", 7, null);
    const error = f.events.find(message => message.type === "error");
    assert.ok(error);
    assert.equal((/** @type {string} */ (error.value.message)).includes("private"), false);
    if (mode !== "malformed") {
      assert.ok((/** @type {string} */ (error.value.message)).includes("ordinary worker diagnostic"));
    }
    assert.equal(f.visible(), "ordinary worker diagnostic\n");
  });
}

test("an ordinary partial stdout diagnostic remains useful on exit", t => {
  const f = fixture(t);
  f.workers[0].stdout.write("ordinary partial diagnostic");
  f.workers[0].stdout.emit("end");
  f.workers[0].emit("exit", 7, null);
  assert.ok((/** @type {string} */ (/** @type {FixtureEvent} */ (f.events[0]).value.message)).includes("ordinary partial diagnostic"));
});
