"use strict";

// Shared Node/native fixture. Do not install a catch around intentionally unhandled promises:
// a successful native exit is precisely the failure this lane must detect.
const mode = process.env["MOM_AOT_REJECTION"];
const reason = new Error("native-rejection-fixture");
if (mode === "error") {Promise.reject(reason);} else if (mode === "primitive") {
  // Non-Error rejection is intentional: its failure code and primitive value must survive AOT.
  // eslint-disable-next-line prefer-promise-reject-errors
  Promise.reject("native-rejection-fixture");
} else if (mode === "async") {
  void (async () => {await Promise.resolve(); throw reason;})();
} else if (mode === "then") {
  void Promise.resolve().then(() => {throw reason;});
} else if (mode === "timer") {
  setTimeout(() => Promise.reject(reason), 1);
} else {
  /** @type {string[]} */
  const trace = [];
  const original = Promise.reject(reason);
  const promise = mode === "chain" ? original.then() : original;
  process.on("unhandledRejection", (error, rejected) => {
    if (error !== reason || rejected !== promise) {throw new Error("Rejection identity lost");}
    trace.push("unhandled");
    if (mode === "handlerThrows") {throw new Error("native-rejection-handler-fixture");}
    if (mode === "lateHandled") {setTimeout(() => promise.catch(() => trace.push("caught")), 1);}
  });
  process.on("rejectionHandled", (handled) => {
    if (handled !== promise) {throw new Error("Handled promise identity lost");}
    trace.push("handled");
  });
  if (mode === "caught") {
    promise.catch(() => trace.push("caught"));
  } else if (mode === "nextTick") {
    process.nextTick(() => promise.catch(() => trace.push("caught")));
  } else if (mode === "microtask") {
    Promise.resolve().then(() => promise.catch(() => trace.push("caught")));
  } else if (mode === "tickFromMicrotask") {
    Promise.resolve().then(() => process.nextTick(() => promise.catch(() => trace.push("caught"))));
  } else if (mode === "awaitCaught") {
    void (async () => {try {await promise;} catch {trace.push("caught");}})();
  } else if (!["recovered", "lateHandled", "handlerThrows", "chain"].includes(mode || "")) {throw new Error("Unknown rejection fixture");}
  setTimeout(() => console.log(JSON.stringify(trace)), 20);
}
