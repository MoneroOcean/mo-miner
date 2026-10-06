"use strict";

const prefix = "MOM_WORKER_MESSAGE ";
const replacementTimeoutMs = 5000;

/** @param {string} type @param {string} role @param {string} state */
function send(type, role, state) {
  process.stdout.write(prefix + JSON.stringify({
    type,
    value: {role, state, pid: process.pid},
    thread_id: 0,
  }) + "\n");
}

if (process.env["MOM_CLUSTER_WORKER"] === "1") {
  const role = process.env["MOM_TEST_WORKER_ROLE"];
  if (role !== "stale" && role !== "current") {throw new Error("Missing worker replacement role");}
  let input = "";
  send("test", role, role === "stale" ? "ready" : "usable");
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk) => {
    input += chunk;
    let eol;
    while ((eol = input.indexOf("\n")) !== -1) {
      const line = input.slice(0, eol);
      input = input.slice(eol + 1);
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        continue;
      }
      if (!message || message.type !== "close") {continue;}
      if (role === "stale") {
        continue;
      } else {
        process.exit(0);
      }
    }
  });
} else {
  process.env["MOM_USE_SUBPROCESS_WORKERS"] = "1";
  const helper = require("../../helper.js");
  const waitForWorkerCleanup = require("./worker_cleanup.js");
  global.opt = require("../../opts").create_default_opts();
  /** @type {import("node:child_process").ChildProcessWithoutNullStreams | null} */
  let staleWorker = null;
  /** @type {import("node:child_process").ChildProcessWithoutNullStreams | null} */
  let replacementWorker = null;
  let replacementStartedAt = 0;
  let replacementUsable = false;
  /** @type {number | null} */
  let staleExitedAt = null;
  let finished = false;
  /** @type {ReturnType<typeof setTimeout> | null} */
  let watchdog = null;

  /** @param {ReturnType<typeof helper.messageWorkers>[number] | undefined} target */
  function subprocessWorker(target) {
    return target?.type === "subprocess" ? target.worker : null;
  }

  /** @param {import("node:child_process").ChildProcessWithoutNullStreams | null} worker */
  function workerExited(worker) {
    return !worker || worker.exitCode !== null || worker.signalCode !== null;
  }

  /** @param {number} status @param {string} message */
  function finish(status, message = "") {
    if (finished) {return;}
    finished = true;
    if (watchdog) {clearTimeout(watchdog);}

    const targets = helper.closeWorkers(0);
    /** @type {import("node:child_process").ChildProcessWithoutNullStreams[]} */
    const workers = [];
    for (const target of targets) {
      const worker = subprocessWorker(target);
      if (!worker) {throw new Error("Unexpected non-subprocess worker target");}
      workers.push(worker);
    }
    if (staleWorker && !workers.includes(staleWorker)) {
      if (!workerExited(staleWorker)) {staleWorker.kill("SIGKILL");}
      workers.push(staleWorker);
    }

    waitForWorkerCleanup(workers, 1250, (cleanupPassed) => {
      const finalStatus = cleanupPassed ? status : 1;
      const finalMessage = cleanupPassed ? message : "worker cleanup exceeded deadline";
      if (finalStatus === 0) {
        const pids = [staleWorker?.pid, replacementWorker?.pid]
          .filter((pid) => Number.isInteger(pid));
        process.stdout.write(`HELPER_WORKER_REPLACEMENT_PIDS ${JSON.stringify(pids)}\n`);
        process.stdout.write("HELPER_WORKER_REPLACEMENT_OK\n", () => process.exit(0));
      } else {
        process.stderr.write(finalMessage + "\n", () => process.exit(finalStatus));
      }
    });
  }

  /** @param {string} message */
  function fail(message) {
    finish(1, message);
  }

  function checkReplacement() {
    if (finished || !replacementUsable) {return;}
    if (!staleExitedAt) {
      if (Date.now() - replacementStartedAt >= replacementTimeoutMs) {
        fail("stale worker was not force-killed within the replacement deadline");
        return;
      }
      setTimeout(checkReplacement, 20);
      return;
    }
    const elapsed = staleExitedAt - replacementStartedAt;
    if (elapsed < 2500 || elapsed >= replacementTimeoutMs) {
      fail("stale worker exited outside the force-kill deadline: " + elapsed);
      return;
    }
    finish(0);
  }

  /** @param {WorkerEvent} message */
  function handleMessage(message) {
    const role = message.value["role"];
    const state = message.value["state"];
    if (typeof role !== "string" || typeof state !== "string") {return;}
    if (message.type === "test" && role === "stale" && state === "ready") {
      const targets = helper.messageWorkers({type: "pause"});
      const worker = subprocessWorker(targets[0]);
      if (!worker) {return fail("stale worker was not available");}
      staleWorker = worker;
      replacementStartedAt = Date.now();
      helper.recreate_threads("cpu", handleMessage, {
        MOM_TEST_WORKER_ROLE: "current",
      });
      return;
    }
    if (message.type !== "test" || role !== "current" || state !== "usable") {
      return;
    }
    const targets = helper.messageWorkers({type: "pause"});
    const worker = subprocessWorker(targets[0]);
    if (!worker) {
      return fail("replacement worker was not created");
    }
    replacementWorker = worker;
    if (replacementWorker === staleWorker || !replacementWorker.pid) {
      return fail("replacement worker was not created");
    }
    replacementUsable = true;
    checkReplacement();
  }

  if (process.platform !== "win32") {
    process.on("SIGTERM", () => finish(1, "replacement fixture interrupted"));
    process.on("SIGINT", () => finish(1, "replacement fixture interrupted"));
  }
  helper.recreate_threads("cpu", handleMessage, {
    MOM_TEST_WORKER_ROLE: "stale",
  });
  const initialTargets = helper.messageWorkers({type: "pause"});
  const initialWorker = subprocessWorker(initialTargets[0]);
  if (!initialWorker) {
    fail("stale worker was not started");
  } else {
    const worker = initialWorker;
    staleWorker = worker;
    worker.once("exit", () => {
      staleExitedAt = Date.now();
      checkReplacement();
    });
  }
  watchdog = setTimeout(() => fail("replacement fixture timed out"), replacementTimeoutMs + 400);
}
