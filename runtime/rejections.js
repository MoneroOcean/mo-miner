"use strict";

/** Node's default throw policy runs after nextTick/microtasks, not Hermes's diagnostic timer.
 * Track only host notifications: the engine still owns Promise resolution/await semantics.
 * @param {{emit(name: string, ...args: unknown[]): boolean}} process */
module.exports = function rejectionTracker(process) {
  const pending = new Map();
  const reported = new WeakSet();
  /** @type {object[]} */
  let handled = [];
  return {
    /** @param {object} promise @param {unknown} reason @param {boolean} isHandled */
    track(promise, reason, isHandled) {
      if (!isHandled) {pending.set(promise, reason); return;}
      pending.delete(promise);
      if (reported.delete(promise)) {handled.push(promise);}
    },
    // Snapshot notifications: handlers may attach catches or create rejections. New microtasks
    // must run before the next notification batch; never walk a live, reentrantly-growing map.
    drain() {
      const late = handled; handled = [];
      const batch = [...pending];
      for (const promise of late) {process.emit("rejectionHandled", promise);}
      for (const [promise, reason] of batch) {
        if (!pending.delete(promise)) {continue;}
        reported.add(promise);
        if (process.emit("unhandledRejection", reason, promise)) {continue;}
        const error = reason instanceof Error ? reason : Object.assign(new Error(
          `Unhandled promise rejection: ${String(reason)}`), {code: "ERR_UNHANDLED_REJECTION"});
        process.emit("uncaughtExceptionMonitor", error, "unhandledRejection");
        if (!process.emit("uncaughtException", error, "unhandledRejection")) {throw error;}
      }
      return Boolean(late.length || batch.length);
    },
  };
};
