"use strict";

const prefix = "MOM_WORKER_MESSAGE ";

/** @param {string} type @param {string} source */
function send(type, source) {
  process.stdout.write(prefix + JSON.stringify({type, value: {source}, thread_id: 0}) + "\n");
}

if (process.env["MOM_CLUSTER_WORKER"] === "1") {
  const source = process.env["MOM_TEST_WORKER_GENERATION"];
  if (source !== "stale" && source !== "current") {throw new Error("Missing worker generation role");}
  if (source === "current") {
    for (const type of ["hashrate", "test", "error"]) {send(type, source);}
  }
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (input) => {
    if (!input.includes('"type":"close"')) {return;}
    if (source === "stale") {
      for (const type of ["hashrate", "test", "error", "result", "last_nonce"]) {
        send(type, source);
      }
    }
    setImmediate(() => process.exit(0));
  });
} else {
  process.env["MOM_USE_SUBPROCESS_WORKERS"] = "1";
  const helper = require("../../helper");
  const waitForWorkerCleanup = require("./worker_cleanup");
  global.opt = require("../../opts").create_default_opts();
  /** @type {Array<{type: string, source: string}>} */
  const messages = [];
  /** @type {ReturnType<typeof setTimeout> | null} */
  let settleTimer = null;
  let finished = false;

  function finish() {
    if (finished) {return;}
    finished = true;
    if (settleTimer) {clearTimeout(settleTimer);}
    const targets = helper.closeWorkers(1000);
    const workers = targets
      .filter((target) => target.type === "subprocess")
      .map((target) => target.worker);
    /** @param {boolean} cleanupPassed */
    const report = (cleanupPassed) => {
      const observed = messages.map((message) => `${message.type}:${message.source}`).sort();
      const expected = ["error:current", "hashrate:current", "test:current"];
      const passed = cleanupPassed && JSON.stringify(observed) === JSON.stringify(expected);
      if (passed) {
        process.stdout.write("HELPER_WORKER_GENERATION_OK\n", () => process.exit(0));
      } else {
        const error = cleanupPassed
          ? `unexpected generation messages: ${JSON.stringify(observed)}`
          : "worker cleanup exceeded the force-kill deadline";
        process.stderr.write(error + "\n", () => process.exit(1));
      }
    };

    waitForWorkerCleanup(workers, 1250, report);
  }

  /** @param {WorkerEvent} message */
  function handleMessage(message) {
    const source = message.value["source"];
    if (typeof source !== "string") {return;}
    messages.push({type: message.type, source});
    if (messages.length >= 3 && !settleTimer) {settleTimer = setTimeout(finish, 50);}
  }

  helper.recreate_threads("cpu", handleMessage, {
    MOM_TEST_WORKER_GENERATION: "stale",
  });
  helper.recreate_threads("cpu", handleMessage, {
    MOM_TEST_WORKER_GENERATION: "current",
  });
  setTimeout(finish, 3000);
}
