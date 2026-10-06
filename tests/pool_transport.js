"use strict";

const assert = require("node:assert/strict");
const net = require("node:net");
const test = require("node:test");

const pool = require("../pool");
const {mockPoolOptions} = require("./logic/support");

/** @typedef {{logged_in: boolean, socket: {destroy?: () => unknown} | null, [key: string]: unknown}} PoolEntry */
/** @typedef {{pools: PoolEntry[], [key: string]: unknown}} PoolTestOptions */
/** @type {{opt: PoolTestOptions | undefined}} */
const testGlobal = /** @type {{opt: PoolTestOptions | undefined}} */ (/** @type {unknown} */ (globalThis));

test("real loopback pool transport handles fragmented newline-delimited jobs", async () => {
  const previousOpt = testGlobal.opt;
  /** @type {import("node:net").Socket | undefined} */
  let acceptedSocket;
  let request = "";
  const server = net.createServer((socket) => {
    acceptedSocket = socket;
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
  const {port} = address;
  testGlobal.opt = mockPoolOptions({
    pool: {url: "127.0.0.1", port},
    pool_time: {connect_throttle: 0, first_job_wait: 0.1},
  });

  try {
    pool.connect_pool_throttle(0, () => {throw new Error("Unexpected pool job");});
    const options = testGlobal.opt;
    if (!options) {throw new Error("Pool test options were not initialized");}
    const firstPool = options.pools[0];
    if (!firstPool) {throw new Error("Pool test has no pool");}
    for (let attempt = 0; attempt < 100 && !firstPool.logged_in; ++attempt) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.match(request, /"method":"login"/);
    assert.equal(firstPool.logged_in, true);
    assert.equal(firstPool["worker_id"], "worker-é");
  } finally {
    const options = testGlobal.opt;
    const firstPool = options?.pools[0];
    const client = firstPool?.socket;
    if (firstPool) {firstPool.socket = null;}
    if (client?.destroy) {client.destroy();}
    if (acceptedSocket) {acceptedSocket.destroy();}
    await new Promise((resolve) => server.close(resolve));
    await new Promise((resolve) => setTimeout(resolve, 125));
    testGlobal.opt = previousOpt;
  }
});
