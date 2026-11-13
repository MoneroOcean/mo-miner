"use strict";

const environment = require("./environment");
const fs = require("./fs");
const bytes = require("./bytes");
const {EventEmitter} = require("node:events");
const Pipe = require("./pipe");

/** POSIX pipes stay asynchronous under backpressure. Windows pipe writes, like Node's, must
 * finish before returning: CLI process.exit() would otherwise discard queued help/error output.
 * Files/terminals retain the synchronous path. @param {number} fd @param {string} platform */
function output(fd, platform) {
  const pipe = new Pipe(false);
  const id = platform === "win32" ? null : globalThis.__momOutputOpen(fd, (event, value) => pipe.receive(event, value));
  if (id !== null) {pipe.id = id; return pipe;}
  return {
    /** @param {string | Uint8Array} value @param {BufferEncoding | (() => void)} [encoding]
     * @param {() => void} [callback] */
    write(value, encoding, callback) {
      const data = bytes(value, typeof encoding === "string" ? encoding : "utf8");
      let offset = 0;
      while (offset < data.length) {
        const size = fs.writeSync(fd, data.subarray(offset));
        if (!size) {throw new Error("Synchronous output made no progress");}
        offset += size;
      }
      const complete = typeof encoding === "function" ? encoding : callback;
      if (complete) {globalThis.__momNextTick(complete);}
      return true;
    },
  };
}

/** Node accepts integers/numeric strings and stores their signed 32-bit representation.
 * @param {unknown} value */
function exitCode(value) {
  if (value == null) {return undefined;}
  if (typeof value === "string" && value.trim()) {value = Number(value);}
  if (typeof value !== "number") {
    throw Object.assign(new TypeError("Exit code must be a number or numeric string"), {code: "ERR_INVALID_ARG_TYPE"});
  }
  if (!Number.isInteger(value)) {
    throw Object.assign(new RangeError("Exit code must be an integer"), {code: "ERR_OUT_OF_RANGE"});
  }
  return value | 0;
}

/** The synthetic script slot preserves the original CLI's argv.slice(2). No application parsing
 * or controller logic is duplicated here. Child-worker launch must account for that slot.
 * @param {NativeProcessInfo} info */
module.exports = function nativeProcess(info) {
  /** @type {number | undefined} */
  let code;
  const events = new EventEmitter();
  /** @type {Map<string, number>} */
  const signals = new Map();
  const supportedSignals = new Set(["SIGINT", "SIGTERM", info.platform === "win32" ? "SIGBREAK" : "SIGHUP"]);
  events.on("newListener", (name) => {
    if (typeof name === "string" && supportedSignals.has(name) && !signals.has(name)) {
      signals.set(name, globalThis.__momSignalStart(name, () => events.emit(name, name)));
    }
  });
  events.on("removeListener", (name) => {
    if (typeof name === "string" && !events.listenerCount(name) && signals.has(name)) {
      globalThis.__momHandleClose(/** @type {number} */ (signals.get(name))); signals.delete(name);
    }
  });
  const properties = {
    env: environment(), argv: [info.execPath, info.execPath, ...info.arguments],
    execPath: info.execPath, platform: info.platform, arch: info.arch, pid: info.pid, ppid: info.ppid,
    stdout: output(1, info.platform), stderr: output(2, info.platform),
    stdin: new Pipe(true),
    cwd() {return globalThis.__momCwd();},
    /** @param {string} directory */
    chdir(directory) {
      if (typeof directory !== "string") {
        throw Object.assign(new TypeError("Directory must be a string"), {code: "ERR_INVALID_ARG_TYPE"});
      }
      globalThis.__momChdir(directory);
    },
    get exitCode() {return code;},
    set exitCode(value) {code = exitCode(value);},
    /** @param {unknown} [value] @returns {never} */
    exit(value) {
      if (value != null) {code = exitCode(value);}
      return globalThis.__momExitNow(code ?? 0);
    },
    /** @param {(...args: unknown[]) => unknown} callback @param {unknown[]} args */
    nextTick(callback, ...args) {
      if (typeof callback !== "function") {throw new TypeError("nextTick callback must be a function");}
      globalThis.__momNextTick(() => callback(...args));
    },
  };
  return /** @type {EventEmitter & typeof properties} */ (
    Object.defineProperties(events, Object.getOwnPropertyDescriptors(properties)));
};
