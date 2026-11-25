// Native-mode language/Node compatibility only; Engineering Mode keeps Node's original builtins.
import {Buffer} from "node:buffer";
import parseJson from "core-js-pure/actual/json/parse.js";
import nativeProcess from "./process.js";
import core from "./core.js";
import rejectionTracker from "./rejections.js";

// Hermes deliberately omits TypedArray subclass propagation. Re-wrap the shared view instead of
// copying bytes: Buffer.subarray aliasing is required by proof encoding and incremental decoding.
if (!Buffer.isBuffer(Buffer.alloc(1).subarray(0))) {
  /** @param {number} [start] @param {number} [end] */
  Buffer.prototype.subarray = function(start, end) {
    const view = Uint8Array.prototype.subarray.call(this, start, end);
    return Buffer.from(view.buffer, view.byteOffset, view.byteLength);
  };
}

// Kaspa job words require JSON reviver context.source; rounding them would silently corrupt jobs.
// The pure standard-library adapter selects native parsing when the engine already implements it.
JSON.parse = parseJson;
globalThis.Buffer = Buffer;
globalThis.global = globalThis;

class Timeout {
  /** @param {number} id */
  constructor(id) {this.id = id;}
  ref() {globalThis.__momHandleRef(this.id, true); return this;}
  unref() {globalThis.__momHandleRef(this.id, false); return this;}
  hasRef() {return globalThis.__momHandleHasRef(this.id);}
  close() {globalThis.__momHandleClose(this.id); return this;}
  [Symbol.toPrimitive]() {return this.id;}
}

/** @param {(...args: unknown[]) => unknown} callback @param {unknown} delay
 * @param {boolean} repeat @param {unknown[]} args */
function timer(callback, delay, repeat, args) {
  if (typeof callback !== "function") {throw new TypeError("Timer callback must be a function");}
  return new Timeout(globalThis.__momTimerStart(() => callback(...args), Number(delay), repeat));
}

/** @param {unknown} value */
function clearTimer(value) {
  const id = value instanceof Timeout ? value.id : Number(value);
  if (Number.isSafeInteger(id) && id > 0 && id <= 0xffffffff) {globalThis.__momHandleClose(id);}
}

Object.defineProperty(globalThis, "process", {value: nativeProcess(globalThis.__momProcessInfo)});
globalThis.__momNativeCore = core;
// There is no cluster IPC here (cluster.js): workers always use the helper's subprocess route.
process.env["MOM_USE_SUBPROCESS_WORKERS"] = "1";
// MoM logs strings (including empty lines), not inspect() graphs or printf-style formats.
const log = (/** @type {{write(value: string): unknown}} */ stream, /** @type {unknown[]} */ values) => {
  if (values.some((value) => typeof value !== "string")) {throw new TypeError("Native console currently accepts strings only");}
  stream.write(values.join(" ") + "\n");
};
globalThis.console = {
  ...globalThis.console,
  log: (...values) => log(process.stdout, values),
  error: (...values) => log(process.stderr, values),
};
Object.assign(globalThis, {
  /** @param {(...args: unknown[]) => unknown} callback @param {unknown[]} args */
  setImmediate(callback, ...args) {
    if (typeof callback !== "function") {throw new TypeError("Immediate callback must be a function");}
    return new Timeout(globalThis.__momImmediateStart(() => callback(...args)));
  },
  clearImmediate: clearTimer,
  /** @param {(...args: unknown[]) => unknown} callback @param {unknown} [delay] @param {unknown[]} args */
  setTimeout: (callback, delay, ...args) => timer(callback, delay, false, args),
  /** @param {(...args: unknown[]) => unknown} callback @param {unknown} [delay] @param {unknown[]} args */
  setInterval: (callback, delay, ...args) => timer(callback, delay, true, args),
  clearTimeout: clearTimer, clearInterval: clearTimer,
});
const timeBase = globalThis.__momNow();
globalThis.performance = /** @type {typeof performance} */ (/** @type {unknown} */ ({
  timeOrigin: Date.now(), now: () => globalThis.__momNow() - timeBase,
}));

const rejections = rejectionTracker(process);
if (globalThis.HermesInternal.enablePromiseRejectionTracker({hostRejectionTracker: rejections.track}) !== true) {
  throw new Error("Native runtime requires the checked Hermes host-rejection patch");
}
globalThis.__momRejectionCheckpoint(rejections.drain);
