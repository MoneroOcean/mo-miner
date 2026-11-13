"use strict";

const crypto = require("node:crypto");

if (process.env["MOM_AOT_CHILD_TASK"] === "echo") {
  /** @type {Buffer[]} */
  const input = [];
  process.stdin.on("data", (data) => input.push(Buffer.from(data)));
  process.stdin.on("end", () => {
    const bytes = Buffer.concat(input);
    process.stdout.write(JSON.stringify({args: process.argv.slice(2), bytes: bytes.length,
      hash: crypto.createHash("sha256").update(bytes).digest("hex"), env: process.env["MOM_AOT_CHILD_VALUE"]}) + "\n");
    process.stderr.write("stderr-é-💎\n");
    process.exitCode = 23;
  });
} else if (["bulk", "burst", "burstClosed"].includes(process.env["MOM_AOT_CHILD_TASK"] || "")) {
  // Larger than a maximum Pearl proof's base64 frame. A stalled reader must not stop the
  // worker's timers, discard bytes or run its flush callback before the payload is delivered.
  process.stderr.write("ready\n");
  process.stdin.once("data", () => {
    process.stdin.destroy();
    setTimeout(() => process.stderr.write("timer\n"), 5);
    if (process.env["MOM_AOT_CHILD_TASK"] === "bulk") {
      const payload = Buffer.alloc(12 * 1024 * 1024, 0x61);
      process.stdout.write(payload, () => process.stderr.write("callback\n"));
      process.exitCode = 29;
    } else {
      // Exercise the real Core event channel, not a fake JS producer. Unknown commands emit
      // harmless error events without hashing/GPU work; their combined output exceeds 32 MiB.
      const core = typeof globalThis.__momWorkerCreate === "function" ? require("../../runtime/core")
        : require(process.env["MOM_AOT_CORE_ADDON"] || "../../build/lin/Release/oneapi/mom.node");
      const command = "unknown-" + "x".repeat(96 * 1024);
      let peakQueued = 0, blocked = false, broken = false, completed = false;
      const brokenReader = process.env["MOM_AOT_CHILD_TASK"] === "burstClosed";
      const finish = () => process.stderr.write("peak " + peakQueued + "\ncallback\n");
      if (brokenReader) {
        process.stdout.on("error", () => {
          if (!broken) {
            broken = true; process.stderr.write("broken\n");
            if (completed) {finish();}
          }
        });
      }
      const worker = new core.AsyncWorker((/** @type {string} */ name, /** @type {{message?: string}} */ values) => {
        if (name !== "error" || !values.message) {throw new Error("Unexpected burst event");}
        if (broken) {return;}
        process.stdout.write(values.message + "\n");
        peakQueued = Math.max(peakQueued, process.stdout.writableLength);
        if (!blocked && peakQueued >= 16 * 1024 * 1024) {blocked = true; process.stderr.write("blocked\n");}
      }, () => {
        // Retain the wrapper until completion, then check that the public API rejects new work.
        // Its native finalizer intentionally stops a collected worker, including under pressure.
        let stopped = false;
        try {worker.sendToCpp("pause");} catch {stopped = true;}
        if (!stopped) {throw new Error("Burst worker accepted a post after completion");}
        completed = true;
        // Producer completion and OS pipe failure may arrive in either order. Require both for
        // the cancellation case; the successful case still requires a real final write flush.
        if (brokenReader) {if (broken) {finish();}} else {process.stdout.write("", finish);}
        process.exitCode = 31;
      }, (/** @type {Error} */ error) => {throw error;});
      for (let i = 0; i < 400; i++) {worker.sendToCpp(command);}
      worker.sendToCpp("close");
    }
  });
} else if (process.env["MOM_AOT_CHILD_TASK"] === "signal") {
  const timer = setInterval(() => undefined, 60000);
  process.on("SIGTERM", () => {
    clearInterval(timer); process.stdout.write("handled-SIGTERM\n"); process.exitCode = 19;
  });
  process.stdout.write("ready\n");
} else if (process.env["MOM_AOT_CHILD_TASK"] === "hang") {
  setInterval(() => undefined, 60000);
  process.stdout.write("ready\n");
} else {throw new Error("Unknown native subprocess fixture mode");}
