"use strict";

// Private, ephemeral loopback fixture for direct-Node/native parity; never a real mining pool.
const net = require("node:net");
const tls = require("node:tls");
const fs = require("node:fs");
const statistics = {tlsLogins: /** @type {string[]} */ ([]), spyConnections: 0, spyBytes: 0, plaintextLeaks: 0};
/** @param {import("node:net").Socket} socket */
function handle(socket) {
  let request = "";
  socket.on("error", () => undefined);
  socket.on("data", (data) => {
    request += data.toString("utf8");
    if (request.length > 2 * 1024 * 1024) {socket.destroy(); return;}
    if (!request.includes("\n")) {return;}
    const login = JSON.parse(request.split("\n")[0] || "null");
    if (login?.method !== "login") {socket.destroy(); return;}
    socket.removeAllListeners("data");
    if (typeof login.params?.case === "string") {
      statistics.tlsLogins.push(login.params.case);
      socket.end(JSON.stringify({accepted: true, case: login.params.case, bytes: Buffer.byteLength(request)}) + "\n");
      return;
    }
    if (login.params?.login === "oversized") {socket.end("x".repeat(65537)); return;}
    const response = Buffer.from('{"id":1,"result":{"worker":"worker-é"}}\n' +
      '{"method":"mining.notify","params":["job",[18446744073709551615,1,2,3],18446744073709551615]}\n' +
      '{"method":"mining.notify","params":["stale",[1,2,3,4],0]}\n');
    const split = response.indexOf(Buffer.from("é")) + 1;
    socket.write(response.subarray(0, split));
    setTimeout(() => socket.end(response.subarray(split)), 10);
  });
}
const server = net.createServer(handle);
const tlsServer = tls.createServer({key: fs.readFileSync(process.argv[2] || ""), cert: fs.readFileSync(process.argv[3] || "")}, handle);
tlsServer.on("tlsClientError", () => undefined);
const expiredServer = tls.createServer({key: fs.readFileSync(process.argv[2] || ""), cert: fs.readFileSync(process.argv[4] || "")}, handle);
expiredServer.on("tlsClientError", () => undefined);
const spyServer = net.createServer((socket) => {
  statistics.spyConnections++;
  let received = Buffer.alloc(0);
  socket.on("error", () => undefined);
  socket.on("data", (data) => {
    statistics.spyBytes += data.length;
    received = Buffer.concat([received, data]);
    if (received.includes("native-parity-sensitive-credential")) {statistics.plaintextLeaks++;}
    if (received.length > 65536) {socket.destroy();}
  });
});
const servers = [server, tlsServer, expiredServer, spyServer];
Promise.all(servers.map((server) => new Promise((resolve, reject) => {
  server.maxConnections = 8;
  server.once("error", reject);
  server.listen({host: "127.0.0.1", port: 0}, () => {
    const address = server.address();
    if (!address || typeof address === "string") {throw new Error("Loopback fixture has no address");}
    resolve(address.port);
  });
}))).then(([port, tlsPort, expiredPort, spyPort]) => process.send?.({port, tlsPort, expiredPort, spyPort}));
process.on("message", () => process.send?.({statistics}));
process.on("disconnect", () => {for (const server of servers) {server.close();}});
setTimeout(() => {throw new Error("Loopback parity fixture exceeded 15 seconds");}, 15000).unref();
