"use strict";

const net = require("node:net");

const mode = process.env["MOM_DUALSTACK_CASE"] || "refused";
const port = Number(process.env["MOM_DUALSTACK_PORT"]);
const cancelDelay = Number(process.env["MOM_DUALSTACK_CANCEL_DELAY_MS"] || 5);
const queuedBytesTarget = 8 * 1024 * 1024;
if (!Number.isInteger(port) || port < 1 || port > 65535) {throw new Error("Missing dual-stack fixture port");}

/** @type {string[]} */
const trace = [];
let connected = false;
let closed = false;
let closeCount = 0;
let failed = false;
let queuedBytes = 0;
let blockedWrites = 0;
let writeCallbacks = 0;
let queuedHeader = false;
const countWrite = () => {writeCallbacks++;};
const socket = net.connect({host: "dualstack.test", port, autoSelectFamilyAttemptTimeout: 25});
if (mode === "queuedwrites") {
  socket.write(`queue:${queuedBytesTarget}\n`);
  queuedHeader = true;
}
socket.on("connect", () => {
  connected = true;
  trace.push("connect");
  if (mode === "queuedwrites") {
    const chunk = Buffer.alloc(64 * 1024, 0x61);
    while (queuedBytes < queuedBytesTarget) {
      const size = Math.min(chunk.length, queuedBytesTarget - queuedBytes);
      const accepted = socket.write(size === chunk.length ? chunk : chunk.subarray(0, size), countWrite);
      queuedBytes += size;
      if (!accepted) {blockedWrites++;}
    }
    socket.end();
  } else {socket.write("ping\n");}
});
socket.on("data", (data) => {trace.push(`data:${data.toString()}`);});
socket.on("end", () => {trace.push("end");});
socket.on("error", (error) => {
  const code = Reflect.get(error, "code");
  trace.push(`error:${typeof code === "string" ? code : "UNKNOWN"}`);
  if (mode === "refused" || mode === "timeout") {failed = true;}
});
socket.on("close", (hadError) => {
  closed = true;
  closeCount++;
  trace.push(`close:${hadError ? "error" : "ok"}`);
  if (mode === "cancel") {failed = connected || closeCount !== 1;} else if (mode === "queuedwrites") {
    failed = !connected || closeCount !== 1 || !queuedHeader || queuedBytes !== queuedBytesTarget || blockedWrites === 0 ||
      writeCallbacks === 0 || !trace.includes("data:queue-ok\n");
  } else {
    failed ||= !connected || closeCount !== 1 || !trace.includes("data:pong\n");
  }
  console.log(JSON.stringify({mode, connected, closeCount, failed, trace, queuedHeader, queuedBytes, blockedWrites, writeCallbacks}));
  process.exitCode = failed ? 2 : 0;
});

if (mode === "cancel") {
  setTimeout(() => {
    if (!closed) {trace.push("cancel"); socket.destroy();}
  }, cancelDelay);
}

setTimeout(() => {
  if (!closed) {
    failed = true;
    trace.push("deadline");
    socket.destroy();
  }
}, 1500).unref();
