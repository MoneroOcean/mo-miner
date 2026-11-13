"use strict";

const {spawn} = require("node:child_process");

// Linker-only fault mode makes uv_spawn initialize its handle then return ENOENT. Verify error
// precedes close without exit/spawn, all pipe handles close, and a fresh child can succeed.
module.exports = function spawnFailure() {
  const worker = spawn(process.execPath, [], {env: {...process.env, MOM_AOT_CHILD_TASK: "echo"}});
  /** @type {string[]} */
  const order = [];
  let code = "";
  const deadline = setTimeout(() => {throw new Error("Spawn failure did not close");}, 5000);
  worker.on("error", (error) => {order.push("error"); code = String(Reflect.get(error, "code"));});
  worker.on("spawn", () => {throw new Error("Failed worker emitted spawn");});
  worker.on("exit", () => {throw new Error("Failed worker emitted exit");});
  worker.on("close", () => {
    if (code !== "ENOENT" || !worker.stdin.destroyed || !worker.stdout.destroyed || !worker.stderr.destroyed) {
      throw new Error("Spawn failure did not close its owned streams");
    }
    order.push("close");
    process.env["MOM_AOT_FAILURE"] = "";
    const recovered = spawn(process.execPath, [], {env: {...process.env, MOM_AOT_CHILD_TASK: "echo"}});
    let stdout = "";
    recovered.on("error", (error) => {throw error;});
    recovered.stdin.on("error", (error) => {throw error;});
    recovered.stdout.setEncoding("utf8");
    recovered.stdout.on("data", (text) => {stdout += text;});
    recovered.stderr.on("data", () => undefined);
    recovered.stdin.end("recovered");
    recovered.on("close", (status, signal) => {
      if (status !== 23 || signal || JSON.parse(stdout).bytes !== 9) {throw new Error("Spawn recovery failed");}
      clearTimeout(deadline);
      console.log(JSON.stringify({spawnFailure: code, order, recovered: true}));
    });
  });
};
