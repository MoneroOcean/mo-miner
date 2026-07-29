"use strict";

const assert = require("node:assert/strict");
const net = require("node:net");
const test = require("node:test");

const pool = require("../pool");
const {mockPoolOptions} = require("./logic/support");

test("real loopback pool transport handles fragmented newline-delimited jobs", async () => {
  const previousOpt = global.opt;
  let acceptedSocket;
  let request = "";
  const server = net.createServer((socket) => {
    acceptedSocket = socket;
    socket.on("data", (chunk) => {
      request += chunk;
      if (!request.includes("\n")) {return;}
      const response = JSON.stringify({
        id: 1,
        jsonrpc: "2.0",
        error: null,
        result: {id: "worker"},
      }) + "\n";
      socket.write(response.slice(0, 17));
      setImmediate(() => socket.write(response.slice(17)));
    });
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const {port} = server.address();
  global.opt = mockPoolOptions({
    pool: {url: "127.0.0.1", port},
    pool_time: {connect_throttle: 0, first_job_wait: 0.1},
  });

  try {
    pool.connect_pool_throttle(0, () => undefined);
    for (let attempt = 0; attempt < 100 && !global.opt.pools[0].logged_in; ++attempt) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.match(request, /"method":"login"/);
    assert.equal(global.opt.pools[0].logged_in, true);
  } finally {
    const client = global.opt.pools[0].socket;
    global.opt.pools[0].socket = null;
    if (client) {client.destroy();}
    if (acceptedSocket) {acceptedSocket.destroy();}
    await new Promise((resolve) => server.close(resolve));
    await new Promise((resolve) => setTimeout(resolve, 125));
    global.opt = previousOpt;
  }
});
