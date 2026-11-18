"use strict";

const assert = require("node:assert/strict");
const {X509Certificate} = require("node:crypto");
const events = require("node:events");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");
const tls = require("node:tls");
const test = require("node:test");

const pool = require("../pool");
const {mockPoolOptions, clearPoolTimers} = require("./logic/support");

/** @type {{opt: ReturnType<typeof mockPoolOptions> | undefined}} */
const testGlobal = /** @type {{opt: ReturnType<typeof mockPoolOptions> | undefined}} */ (
  /** @type {unknown} */ (globalThis));

const tlsFixture = path.join(__dirname, "fixtures");
const tlsCertificate = fs.readFileSync(path.join(tlsFixture, "tls-localhost-cert.pem"), "utf8");
const tlsPrivateKey = fs.readFileSync(path.join(tlsFixture, "tls-localhost-key.pem"), "utf8");
const tlsFingerprint = new X509Certificate(tlsCertificate).fingerprint256;
const tlsFingerprintPlain = tlsFingerprint.replaceAll(":", "");
// None of these fixture pools sends a job; keep the initial-job deadline out of every test.
const poolTime = {connect_throttle: 0, first_job_wait: 60};

/**
 * Wait until `predicate` holds, re-checking whenever `emitter` emits `event`. The deadline only
 * reports a hang; it is far above any loaded-machine latency.
 * @param {events.EventEmitter} emitter @param {string} event @param {() => boolean} predicate
 * @param {string} message @returns {Promise<void>}
 */
function waitForEvent(emitter, event, predicate, message) {
  return new Promise((resolve, reject) => {
    const check = () => {
      if (!predicate()) {return;}
      clearTimeout(deadline);
      emitter.off(event, check);
      resolve();
    };
    const deadline = setTimeout(() => {
      emitter.off(event, check);
      reject(new assert.AssertionError({message}));
    }, 30000);
    emitter.on(event, check);
    check();
  });
}

// Close the pool's client connection the way clear_pool_connection would, so no pool timer or
// socket callback outlives the test, then restore the previous options.
/** @param {ReturnType<typeof mockPoolOptions> | undefined} previousOpt */
function closeTestPool(previousOpt) {
  const poolConfig = testGlobal.opt?.pools[0];
  if (poolConfig) {
    const client = poolConfig.socket;
    poolConfig.socket = null;
    clearPoolTimers(poolConfig);
    client?.destroy?.();
  }
  testGlobal.opt = previousOpt;
}

/** @param {Record<string, unknown>} poolOverrides
 * @param {(state: {request: () => string, switches: () => number, signals: events.EventEmitter}) =>
 *   Promise<unknown>} callback */
async function withLoopbackTlsPool(poolOverrides, callback) {
  const previousOpt = testGlobal.opt, previousSwitch = pool.switch_pool;
  // "data" fires when the server receives bytes, "switch" when the client fails over.
  const signals = new events.EventEmitter();
  /** @type {import("node:tls").TLSSocket[]} */
  const serverSockets = [];
  let request = "";
  let switchCount = 0;
  const server = tls.createServer({cert: tlsCertificate, key: tlsPrivateKey}, (socket) => {
    serverSockets.push(socket);
    socket.on("error", () => undefined);
    socket.on("data", (chunk) => {
      request += chunk.toString("utf8");
      signals.emit("data");
    });
  });
  server.on("tlsClientError", () => undefined);
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen({port: 0, host: "127.0.0.1"}, () => resolve(undefined));
  });
  const address = server.address();
  if (!address || typeof address === "string") {throw new Error("TLS loopback server has no address");}
  testGlobal.opt = mockPoolOptions({
    pool: {url: "127.0.0.1", port: address.port, is_tls: true, tls_verify: false, ...poolOverrides},
    pool_time: poolTime,
  });
  pool.switch_pool = () => {
    ++switchCount;
    signals.emit("switch");
  };
  try {
    pool.connect_pool_throttle(0, () => {throw new Error("Unexpected pool job");});
    return await callback({request: () => request, switches: () => switchCount, signals});
  } finally {
    closeTestPool(previousOpt);
    pool.switch_pool = previousSwitch;
    for (const socket of serverSockets) {socket.destroy();}
    await new Promise((resolve) => server.close(resolve));
  }
}

/** @param {unknown} peerCertificate
 * @param {(state: {socket: events.EventEmitter & {write: (message: string) => boolean, destroy: () => void},
 *   writes: string[], switches: () => number,
 *   poolConfig: ReturnType<typeof mockPoolOptions>["pools"][0]}) => Promise<unknown>} callback */
async function withMockTlsSocket(peerCertificate, callback) {
  const previousOpt = testGlobal.opt, previousConnect = tls.connect, previousSwitch = pool.switch_pool;
  /** @type {string[]} */
  const writes = [];
  let switchCount = 0;
  /** @type {events.EventEmitter & {write: (message: string) => boolean, destroy: () => void,
   * getPeerCertificate: () => unknown}} */
  const socket = Object.assign(new events.EventEmitter(), {
    write: (/** @type {string} */ message) => {writes.push(message); return true;},
    destroy: () => undefined,
    getPeerCertificate: () => peerCertificate,
  });
  tls.connect = /** @type {typeof tls.connect} */ (/** @type {unknown} */ (() => socket));
  pool.switch_pool = () => {++switchCount;};
  testGlobal.opt = mockPoolOptions({
    pool: {url: "127.0.0.1", port: 1, is_tls: true, tls_verify: false, tls_fingerprint: tlsFingerprintPlain},
    pool_time: poolTime,
  });
  const poolConfig = testGlobal.opt.pools[0];
  try {
    pool.connect_pool_throttle(0, () => {throw new Error("Unexpected pool job");});
    return await callback({socket, writes, switches: () => switchCount, poolConfig});
  } finally {
    closeTestPool(previousOpt);
    tls.connect = previousConnect;
    pool.switch_pool = previousSwitch;
  }
}

test.describe("Pool transport", {concurrency: false}, () => {
  test("real loopback pool transport handles fragmented newline-delimited jobs", async () => {
    const previousOpt = testGlobal.opt, previousSwitch = pool.switch_pool;
    /** @type {import("node:net").Socket[]} */
    const serverSockets = [];
    let request = "";
    const server = net.createServer((socket) => {
      serverSockets.push(socket);
      socket.on("error", () => undefined);
      socket.on("data", (chunk) => {
        request += chunk.toString("utf8");
        if (!request.includes("\n")) {return;}
        const workerId = "worker-é";
        const response = Buffer.from(JSON.stringify({
          id: 1,
          jsonrpc: "2.0",
          error: null,
          result: {id: workerId},
        }) + "\n");
        const workerBytes = Buffer.from(workerId);
        const workerOffset = response.indexOf(workerBytes);
        const multibyteOffset = workerBytes.findIndex((byte) => byte > 0x7f);
        if (workerOffset < 0 || multibyteOffset < 0) {
          throw new Error("Loopback response has no multibyte worker id");
        }
        const splitAt = workerOffset + multibyteOffset + 1;
        socket.write(response.subarray(0, splitAt));
        setImmediate(() => socket.write(response.subarray(splitAt)));
      });
    });

    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen({port: 0, host: "127.0.0.1"}, () => resolve(undefined));
    });
    const address = server.address();
    if (!address || typeof address === "string") {throw new Error("Loopback server has no address");}
    const options = mockPoolOptions({pool: {url: "127.0.0.1", port: address.port}, pool_time: poolTime});
    testGlobal.opt = options;
    // A failed connection must surface as a test failure, not a real reconnect loop.
    pool.switch_pool = () => {throw new Error("Unexpected pool failover");};

    try {
      pool.connect_pool_throttle(0, () => {throw new Error("Unexpected pool job");});
      const firstPool = options.pools[0];
      const client = firstPool.socket;
      if (!(client instanceof events.EventEmitter)) {throw new Error("Pool test has no client socket");}
      await waitForEvent(client, "data", () => firstPool.logged_in, "Loopback pool login did not complete");
      assert.match(request, /"method":"login"/);
      assert.equal(firstPool["worker_id"], "worker-é");
    } finally {
      closeTestPool(previousOpt);
      pool.switch_pool = previousSwitch;
      for (const socket of serverSockets) {socket.destroy();}
      await new Promise((resolve) => server.close(resolve));
    }
  });

  test("self-signed loopback TLS remains accepted by default", async () => {
    await withLoopbackTlsPool({}, async ({request, signals}) => {
      await waitForEvent(signals, "data", () => request().includes('"method":"login"'),
        "Default TLS did not send pool login");
      assert.match(request(), /"method":"login"/);
    });
  });

  test("matching unseparated leaf fingerprint sends login", async () => {
    await withLoopbackTlsPool({tls_fingerprint: tlsFingerprintPlain.toLowerCase()}, async ({request, signals}) => {
      await waitForEvent(signals, "data", () => request().includes('"method":"login"'),
        "Matching plain fingerprint did not send pool login");
      assert.match(request(), /"method":"login"/);
    });
  });

  test("matching colon-delimited mixed-case leaf fingerprint sends login", async () => {
    const mixed = tlsFingerprint.toLowerCase().replace(/([0-9a-f]{2})(?=:|$)/g, (byte) => byte.toUpperCase());
    await withLoopbackTlsPool({tls_fingerprint: mixed}, async ({request, signals}) => {
      await waitForEvent(signals, "data", () => request().includes('"method":"login"'),
        "Matching colon fingerprint did not send pool login");
      assert.match(request(), /"method":"login"/);
    });
  });

  test("wrong leaf fingerprint rejects before sending login", async () => {
    const wrong = `00${tlsFingerprintPlain.slice(2)}`;
    await withLoopbackTlsPool({tls_fingerprint: wrong}, async ({request, switches, signals}) => {
      await waitForEvent(signals, "switch", () => switches() > 0, "Wrong TLS fingerprint did not reject the connection");
      assert.equal(request(), "");
    });
  });

  test("a mismatch in the final fingerprint byte rejects before login", async () => {
    const last = tlsFingerprintPlain.at(-1);
    if (!last) {throw new Error("TLS fingerprint is empty");}
    const wrong = tlsFingerprintPlain.slice(0, -1) + (last === "0" ? "1" : "0");
    await withLoopbackTlsPool({tls_fingerprint: wrong}, async ({request, switches, signals}) => {
      await waitForEvent(signals, "switch", () => switches() > 0, "Final-byte TLS fingerprint mismatch was not rejected");
      assert.equal(request(), "");
    });
  });

  test("CA verification rejects the self-signed leaf even with its matching pin", async () => {
    await withLoopbackTlsPool({tls_verify: true, tls_fingerprint: tlsFingerprint}, async ({request, switches, signals}) => {
      await waitForEvent(signals, "switch", () => switches() > 0, "Untrusted TLS certificate was not rejected");
      assert.equal(request(), "");
    });
  });

  test("TCP connect alone never sends credentials before secureConnect", async () => {
    await withMockTlsSocket({fingerprint256: tlsFingerprint}, async ({socket, writes}) => {
      socket.emit("connect");
      assert.deepEqual(writes, []);
      socket.emit("secureConnect");
      assert.equal(writes.length, 1);
      assert.match(writes[0] || "", /"method":"login"/);
    });
  });

  test("missing peer certificate fingerprint fails closed", async () => {
    await withMockTlsSocket({}, async ({socket, writes, switches}) => {
      socket.emit("secureConnect");
      assert.ok(switches() > 0, "Missing peer fingerprint did not reject TLS");
      assert.deepEqual(writes, []);
    });
  });

  test("malformed peer certificate fingerprint fails closed", async () => {
    await withMockTlsSocket({fingerprint256: "not-a-fingerprint"}, async ({socket, writes, switches}) => {
      socket.emit("secureConnect");
      assert.ok(switches() > 0, "Malformed peer fingerprint did not reject TLS");
      assert.deepEqual(writes, []);
    });
  });

  test("a stale TLS socket cannot send login after replacement", async () => {
    await withMockTlsSocket({fingerprint256: tlsFingerprint}, async ({socket, writes, poolConfig}) => {
      poolConfig.socket = {destroy: () => undefined};
      socket.emit("secureConnect");
      assert.deepEqual(writes, []);
    });
  });
});
