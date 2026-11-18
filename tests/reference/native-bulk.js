"use strict";

const {spawn} = require("node:child_process");
const path = require("node:path");
const crypto = require("node:crypto");

/** @param {string} [mode] */
module.exports = function bulkFixture(mode = "bulk") {
  return new Promise((resolve, reject) => {
    const prefix = typeof globalThis.__momSpawn === "function" ? [] : [path.join(__dirname, "native-child-task.js")];
    const worker = spawn(process.execPath, prefix, {stdio: ["pipe", "pipe", "pipe"],
      env: {...process.env, MOM_AOT_CHILD_TASK: mode}});
    const hash = crypto.createHash("sha256");
    let size = 0, stderr = "", started = false, reading = false;
    /** @type {ReturnType<typeof setTimeout> | undefined} */
    let resume;
    // This intentionally sends nearly 40 MiB through the language/stream bridge. Bound the test,
    // but allow slower AOT/debug hosts to finish instead of mistaking conversion cost for a hang.
    const deadline = setTimeout(() => {
      worker.kill("SIGKILL"); reject(new Error(`Bulk output deadline (${size} bytes; ${JSON.stringify(stderr)})`));
    }, 30000);
    worker.on("error", reject); worker.stdin.on("error", reject);
    worker.stdout.on("data", (bytes) => {size += bytes.length; hash.update(bytes);});
    worker.stdout.pause();
    worker.stderr.setEncoding("utf8");
    worker.stderr.on("data", (text) => {
      stderr += text;
      if (!started && stderr.includes("ready\n")) {
        started = true;
        worker.stdin.write("go");
      }
      if (started && !reading && (mode === "bulk" || stderr.includes("blocked\n"))) {
        reading = true;
        resume = setTimeout(() => mode === "burstClosed" ? worker.stdout.destroy() : worker.stdout.resume(), 200);
      }
    });
    worker.on("close", (code, signal) => {
      clearTimeout(deadline); clearTimeout(resume);
      const trace = stderr.trim().split("\n").filter((line) => line !== "blocked" && !line.startsWith("peak "));
      resolve({size, hash: hash.digest("hex"), trace, peakQueued: Number(stderr.match(/^peak (\d+)$/m)?.[1] || 0), code, signal,
        pipeClosed: worker.stdout.destroyed});
    });
  });
};
