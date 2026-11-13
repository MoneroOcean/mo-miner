"use strict";

// IPv6-only loopback peer for the native dual-stack AOT fixture. The production resolver is
// redirected to localhost by the test-only native wrapper; this server never exposes IPv4.
const net = require("node:net");

const server = net.createServer((socket) => {
  socket.on("error", () => undefined);
  let header = Buffer.alloc(0);
  let queuedExpected = 0;
  let queuedReceived = 0;
  let queuedResponded = false;
  /** @param {Buffer} data */
  const consume = (data) => {
    if (queuedExpected) {
      queuedReceived += data.length;
      if (queuedReceived >= queuedExpected && !queuedResponded) {
        queuedResponded = true;
        socket.end("queue-ok\n");
      }
      return;
    }
    header = Buffer.concat([header, data]);
    const newline = header.indexOf(10);
    if (newline < 0) {return;}
    const line = header.subarray(0, newline).toString();
    const remainder = header.subarray(newline + 1);
    header = Buffer.alloc(0);
    if (line === "ping") {socket.end("pong\n"); return;}
    if (line.startsWith("queue:")) {
      queuedExpected = Number(line.slice("queue:".length));
      if (!Number.isSafeInteger(queuedExpected) || queuedExpected < 1) {socket.destroy(); return;}
      consume(remainder);
      return;
    }
    socket.destroy();
  };
  socket.on("data", consume);
});

server.on("error", (error) => {console.error(error); process.exitCode = 1;});
server.listen({host: "::1", port: 0, ipv6Only: true}, () => {
  const address = server.address();
  if (!address || typeof address === "string") {throw new Error("Dual-stack fixture has no address");}
  if (!process.send) {throw new Error("Dual-stack fixture needs an IPC parent");}
  process.send({port: address.port});
});

process.on("disconnect", () => server.close());
setTimeout(() => {throw new Error("Dual-stack fixture exceeded 10 seconds");}, 10000).unref();
