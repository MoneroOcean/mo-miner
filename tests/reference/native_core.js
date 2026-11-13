"use strict";

// The same message fixture calls the real Core through development N-API and the native bridge.
// This lane is deliberately CPU-only; it cannot qualify or silently replace any GPU algorithm.
const core = typeof globalThis.__momWorkerCreate === "function" ? require("../../runtime/core")
  : require(process.env["MOM_AOT_CORE_ADDON"] || "../../build/lin/Release/dpcpp/mom.node");
const vectors = require("../vectors/cpu");

/** @param {string} command @param {Record<string, unknown> | null} [values] */
function exchange(command, values) {
  return new Promise((resolve, reject) => {
    /** @type {Array<{name: string, values: Record<string, string>}>} */
    const events = [];
    const deadline = setTimeout(() => reject(new Error("Native CPU proof exceeded deadline")), 5000);
    const worker = new core.AsyncWorker((/** @type {string} */ name, /** @type {Record<string, string>} */ value) => {
      events.push({name, values: value});
      worker.sendToCpp("close");
    }, () => {clearTimeout(deadline); resolve(events);}, (/** @type {Error} */ error) => {clearTimeout(deadline); reject(error);});
    worker.sendToCpp(command, values);
  });
}

module.exports = async function nativeCoreChecks() {
  const rejected = [];
  for (const args of [[], [() => undefined, () => undefined], [1, () => undefined, () => undefined]]) {
    try {
      Reflect.construct(core.AsyncWorker, args); rejected.push("NOT_REJECTED");
    } catch (error) {rejected.push(error instanceof TypeError ? "TypeError" : "wrong-error");}
  }
  if (rejected.some((value) => value !== "TypeError")) {throw new Error("Native worker accepted invalid callbacks");}
  const unknown = await exchange("unknown-command");
  const malformed = await exchange("job", {pool_id: "pool", worker_id: "worker", job_id: "job", job_token: "token", target: "abc"});
  const vector = vectors.find(({name}) => name === "cn/r height 1806260");
  if (!vector) {throw new Error("Missing canonical CPU vector");}
  const result = await exchange("test", {...vector.job, dev: "cpu"});
  const resultEvents = /** @type {Array<{name: string, values: Record<string, string>}>} */ (result);
  if (!resultEvents.some(({name, values}) => name === "test" && values["result"] === vector.expected)) {
    throw new Error("Native CPU proof disagrees with canonical vector");
  }
  return {rejected, unknown, malformed, result};
};
