"use strict";

const {spawn} = require("node:child_process");
const path = require("node:path");
const crypto = require("node:crypto");

/** @param {string} mode @param {string[]} args */
function child(mode, args = []) {
  // Node needs its fixture script; the native executable dispatches this same source internally.
  const prefix = typeof globalThis.__momSpawn === "function" ? [] : [path.join(__dirname, "native_child_task.js")];
  return spawn(process.execPath, [...prefix, ...args], {stdio: ["pipe", "pipe", "pipe"],
    env: {...process.env, MOM_AOT_CHILD_TASK: mode, MOM_AOT_CHILD_VALUE: "worker-é-💎"}});
}

/** @param {string} mode @param {Buffer | null} input @param {string[]} [args] */
function exercise(mode, input, args = []) {
  return new Promise((resolve, reject) => {
    const worker = child(mode, args);
    /** @type {string[]} */
    const trace = [];
    let stdout = "", stderr = "", signalled = false;
    const deadline = setTimeout(() => {worker.kill("SIGKILL"); reject(new Error("Native subprocess deadline"));}, 5000);
    worker.on("error", reject);
    worker.stdin.on("error", reject);
    worker.on("spawn", () => trace.push("spawn"));
    worker.stdout.setEncoding("utf8"); worker.stderr.setEncoding("utf8");
    worker.stdout.on("data", (data) => {
      stdout += data;
      if (mode !== "echo" && stdout.includes("ready\n") && !signalled) {
        signalled = true;
        if (!worker.kill(mode === "signal" ? "SIGTERM" : "SIGKILL")) {reject(new Error("Cannot signal child"));}
      }
    });
    worker.stderr.on("data", (data) => {stderr += data;});
    worker.on("exit", () => trace.push("exit"));
    worker.on("close", (code, signal) => {
      clearTimeout(deadline); trace.push("close");
      resolve({stdout, stderr, code, signal, trace, killed: worker.killed, stdinClosed: worker.stdin.destroyed,
        stdoutClosed: worker.stdout.destroyed, stderrClosed: worker.stderr.destroyed});
    });
    if (input) {worker.stdin.end(input);} else {worker.stdin.end();}
  });
}

module.exports = async function childrenFixture() {
  const input = Buffer.concat([Buffer.from("é-💎\0"), Buffer.alloc(256 * 1024, 7)]);
  const echo = /** @type {{stdout: string}} */ (await exercise("echo", input, ["", "é-💎", "--", "space value"]));
  const decoded = JSON.parse(echo.stdout);
  if (decoded.hash !== crypto.createHash("sha256").update(input).digest("hex") || decoded.bytes !== input.length) {
    throw new Error("Subprocess pipes corrupted bytes");
  }
  const handled = await exercise("signal", null);
  const killed = await exercise("hang", null);
  // Linux's per-argument limit forces a real uv_spawn failure after handles were allocated.
  let failed = "NOT_REJECTED";
  try {child("echo", ["x".repeat(256 * 1024)]);} catch (error) {
    failed = error instanceof Error ? String(Reflect.get(error, "code")) : "unknown";
  }
  if (failed !== "E2BIG") {throw new Error("Expected actual argument-limit spawn rejection");}
  const recovered = await exercise("echo", Buffer.from("after-failure"));
  return {echo, handled, killed, failed, recovered};
};
