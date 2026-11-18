"use strict";

const clusterModule = require("node:cluster");
const cluster = clusterModule.default ?? clusterModule;
const fs = require("node:fs");
const failureMode = process.env["MOM_WORKER_FAILURE_MODE"] || "";
const normalCloseMode = /^normal-close-(?:clean|nonzero|signal)$/.test(failureMode);
const workerMode = process.env["MOM_CLUSTER_WORKER"] === "1" ||
  !cluster.isPrimary;

if (workerMode) {
  if (normalCloseMode) {
    const {EventEmitter} = require("node:events");
    const workerHelper = require("../../helper.js");
    const from = new EventEmitter();
    if (failureMode === "normal-close-clean") {
      const markerPath = process.env["MOM_HELPER_BEFORE_EXIT_MARKER"];
      if (!markerPath) {throw new Error("normal-close fixture marker path is missing");}
      process.once("beforeExit", () => {
        const fd = fs.openSync(markerPath, "w", 0o600);
        try {
          fs.writeSync(fd, "beforeExit\n");
        } finally {
          fs.closeSync(fd);
        }
      });
    }
    workerHelper.create_core = () => ({
      from,
      emit_to(type) {
        if (type !== "close") {return;}
        setImmediate(() => {
          from.emit("close");
          if (failureMode === "normal-close-nonzero") {
            process.exit(7);
          } else if (failureMode === "normal-close-signal") {
            process.kill(process.pid, "SIGKILL");
          }
        });
      },
    });
    if (!workerHelper.cluster_process()) {
      throw new Error("normal-close fixture did not enter worker mode");
    }
  } else if (failureMode === "unexpected-exit") {
    setImmediate(() => process.exit(0));
  } else if (failureMode === "oversized") {
    process.stdout.write("x".repeat(1024 * 1024 + 1));
  } else if (failureMode !== "unwritable") {
    process.stdout.write("MOM_WORKER_MESSAGE {malformed\n");
  }
  if (!normalCloseMode) {setInterval(() => process.stdout.write(""), 1000);}
} else if (normalCloseMode) {
  if (process.env["MOM_WORKER_TRANSPORT"] !== "cluster") {
    process.env["MOM_USE_SUBPROCESS_WORKERS"] = "1";
  }
  const parentHelper = require("../../helper.js");
  global.opt = require("../../opts").create_default_opts();
  /** @type {string[]} */
  const loggedErrors = [];
  let workerErrors = 0;
  let finished = false;
  parentHelper.log_err = (/** @type {string} */ message) => loggedErrors.push(message);
  /** @param {number | null} code @param {NodeJS.Signals | null} signal */
  const finish = (code, signal) => {
    if (finished) {return;}
    finished = true;
    const observed = {
      transport: process.env["MOM_WORKER_TRANSPORT"] || "subprocess",
      mode: failureMode,
      workerCode: code,
      workerSignal: signal,
      parentExitCode: process.exitCode == null ? 0 : process.exitCode,
      workerErrors,
      loggedErrors,
    };
    process.stdout.write("HELPER_EXPECTED_CLOSE_RESULT " + JSON.stringify(observed) + "\n");
  };
  parentHelper.recreate_threads("cpu", (message) => {
    if (message.type === "error") {workerErrors++;}
  });
  const targets = parentHelper.messageWorkers({type: "pause"});
  const target = targets[0];
  if (!target) {throw new Error("normal-close fixture worker was not created");}
  target.worker.once("exit", finish);
  const timeout = setTimeout(() => {
    process.stderr.write("normal-close fixture worker did not exit\n", () => process.exit(1));
  }, 2000);
  timeout.unref();
  parentHelper.messageWorkers({type: "close"});
} else {
  process.env["MOM_USE_SUBPROCESS_WORKERS"] = "1";
  const helper = require("../../helper.js");
  global.opt = require("../../opts").create_default_opts();
  let errorCount = 0;
  let finished = false;

  /** @param {ReturnType<typeof helper.messageWorkers>[number] | undefined} target */
  function subprocessWorker(target) {
    return target?.type === "subprocess" ? target.worker : null;
  }

  /** @param {number} status @param {string} message */
  function finish(status, message) {
    if (finished) {return;}
    finished = true;
    if (status === 0) {
      process.stdout.write("HELPER_MALFORMED_WORKER_OK\n");
    } else {
      process.stderr.write(message + "\n");
    }
    setTimeout(() => process.exit(status), 20);
  }

  /** @param {import("node:child_process").ChildProcessWithoutNullStreams} worker
   * @param {number} expectedErrors */
  function finishAfterWorkerExit(worker, expectedErrors) {
    const workerExited = () => worker.exitCode !== null || worker.signalCode !== null;
    const done = () => finish(errorCount === expectedErrors ? 0 : 1,
      "unexpected worker error count: " + errorCount);
    if (workerExited()) {
      done();
    } else {
      worker.once("exit", done);
      setTimeout(() => {
        if (!workerExited()) {finish(1, "worker did not exit promptly");}
      }, 200);
    }
  }

  /** @param {WorkerEvent} msg */
  function handleMessage(msg) {
    if (msg.type !== "error") {return;}
    errorCount++;
    const message = String(msg.value["message"] || "");
    const mode = process.env["MOM_WORKER_FAILURE_MODE"];
    const expected = mode === "oversized" ? "worker output line limit" :
      mode === "unexpected-exit" ? "exited unexpectedly" : "malformed worker output";
    if (errorCount !== 1 || !message.includes(expected) ||
        message.includes("MOM_WORKER_MESSAGE") || message.includes("{malformed") ||
        msg.value["fatal"] !== true) {
      helper.closeWorkers(0);
      finish(1, "unexpected worker error: " + message);
      return;
    }
    if (mode === "unexpected-exit") {return finish(0, "");}
    const targets = helper.closeWorkers(1000);
    const worker = subprocessWorker(targets[0]);
    if (!worker) {
      finish(1, "malformed worker was not available for cleanup");
      return;
    }
    finishAfterWorkerExit(worker, 1);
  }

  helper.recreate_threads("cpu", handleMessage);
  if (process.env["MOM_WORKER_FAILURE_MODE"] === "unwritable") {
    setTimeout(() => {
      const active = helper.messageWorkers({type: "pause"});
      const worker = subprocessWorker(active[0]);
      if (!worker) {return finish(1, "worker was not available");}
      worker.stdin.once("close", () => {
        const targets = helper.closeWorkers(0);
        if (!targets.some((target) => target.worker === worker)) {
          return finish(1, "unwritable worker was omitted from cleanup");
        }
        finishAfterWorkerExit(worker, 0);
      });
      worker.stdin.destroy();
    }, 20);
  }
  setTimeout(() => {
    helper.closeWorkers(0);
    finish(1, "timed out waiting for malformed worker output");
  }, 2000);
}
