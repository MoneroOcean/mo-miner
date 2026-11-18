"use strict";

/**
 * Wait for subprocess workers to be reaped after their owner requested closure.
 * @param {import("node:child_process").ChildProcessWithoutNullStreams[]} workers
 * @param {number} timeoutMs
 * @param {(completed: boolean) => void} done
 */
module.exports = function waitForWorkerCleanup(workers, timeoutMs, done) {
  const deadline = Date.now() + timeoutMs;
  const wait = () => {
    const completed = workers.every((worker) =>
      worker.exitCode !== null || worker.signalCode !== null);
    if (completed || Date.now() >= deadline) {
      done(completed);
      return;
    }
    setTimeout(wait, 10);
  };
  wait();
};
