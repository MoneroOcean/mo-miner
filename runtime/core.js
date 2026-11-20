"use strict";

// Only the native build resolves its compute module here. Keep the public addon contract and
// string-valued message boundary identical; the original controller/worker sources need no fork.
class AsyncWorker {
  /** @param {(name: string, values: Record<string, string>) => void} progress
   * @param {() => void} complete @param {(error: Error) => void} error */
  constructor(progress, complete, error) {
    if (arguments.length < 3) {throw new TypeError("AsyncWorker requires progress, complete, and error callbacks");}
    if (![progress, complete, error].every((callback) => typeof callback === "function")) {
      throw new TypeError("AsyncWorker callbacks must be functions");
    }
    this.handle = globalThis.__momWorkerCreate(progress, complete, error);
  }
  /** @param {string} name @param {object | null} [values] */
  sendToCpp(name, values) {
    if (arguments.length < 1) {throw new TypeError("sendToCpp requires a message name");}
    if (typeof name !== "string") {throw new TypeError("sendToCpp message name must be a string");}
    if (values != null && typeof values !== "object") {throw new TypeError("sendToCpp values must be an object");}
    /** @type {Array<[string, string]>} */
    const entries = [];
    for (const key in values || {}) {
      const value = Reflect.get(values || {}, key);
      if (typeof value === "symbol") {throw new TypeError("Cannot convert a Symbol value to a string");}
      entries.push([key, String(value)]);
    }
    this.handle.post(name, entries);
  }
}

module.exports = {AsyncWorker, nvidiaComputeCapabilities: () => globalThis.__momNvidiaComputeCapabilities(),
  displayAdapters: () => globalThis.__momDisplayAdapters()};
