"use strict";

const {EventEmitter} = require("node:events");
const Pipe = require("./pipe");

// Process exit and stream close are independent; preserve Node's separate events and retain the
// adapters until every owned stream is closed. The native registry reaps the OS child itself.
class ChildProcess extends EventEmitter {
  /** @param {string[]} args @param {NodeJS.ProcessEnv} env */
  constructor(args, env) {
    super();
    this.stdin = new Pipe(false); this.stdout = new Pipe(true); this.stderr = new Pipe(true);
    /** @type {number | null} */
    this.exitCode = null;
    /** @type {NodeJS.Signals | null} */
    this.signalCode = null;
    this.killed = false;
    this.exited = false;
    this.closed = 0;
    for (const pipe of [this.stdin, this.stdout, this.stderr]) {
      pipe.once("close", () => {this.closed++; this.finish();});
    }
    const entries = Object.entries(env).filter(([, value]) => value !== undefined)
      .map(([name, value]) => {
        if (!name || name.includes("=") || name.includes("\0") || String(value).includes("\0")) {
          throw new TypeError("Invalid child environment");
        }
        return name + "=" + String(value);
      });
    const info = globalThis.__momSpawn([process.execPath, ...args], entries, (event, value) => this.receive(event, value));
    this.id = info.id; this.pid = info.pid;
    this.stdin.id = info.stdin; this.stdout.id = info.stdout; this.stderr.id = info.stderr;
  }
  /** @param {string} event @param {unknown} value */
  receive(event, value) {
    const separator = event.indexOf(":");
    if (separator !== -1) {
      const name = event.slice(0, separator);
      const pipe = name === "stdin" ? this.stdin : name === "stdout" ? this.stdout : this.stderr;
      pipe.receive(event.slice(separator + 1), value);
    } else if (event === "exit") {
      const result = /** @type {{code: number | null, signal: NodeJS.Signals | null}} */ (value);
      this.exitCode = result.code;
      this.signalCode = result.signal;
      this.exited = true;
      this.emit("exit", this.exitCode, this.signalCode);
      this.finish();
    } else if (event === "error") {
      const number = value instanceof Error ? Reflect.get(value, "errno") : undefined;
      this.exitCode = typeof number === "number" ? number : -1; this.exited = true;
      this.emit("error", value);
    } else if (event === "processClose") {this.finish();} else {this.emit(event);}
  }
  finish() {if (this.exited && this.closed === 3) {this.closed++; this.emit("close", this.exitCode, this.signalCode);}}
  /** @param {NodeJS.Signals} [signal] */
  kill(signal = "SIGTERM") {
    const sent = globalThis.__momProcessKill(this.id, signal);
    if (sent) {this.killed = true;}
    return sent;
  }
  ref() {globalThis.__momHandleRef(this.id, true); return this;}
  unref() {globalThis.__momHandleRef(this.id, false); return this;}
}

module.exports = {
  /** @param {string} command @param {string[]} [args]
   * @param {{env?: NodeJS.ProcessEnv, stdio?: unknown}} [options] */
  spawn(command, args = [], options = {}) {
    if (!options || typeof options !== "object" || Object.keys(options).some((name) => !["env", "stdio"].includes(name))) {
      throw new Error("Unsupported native spawn options");
    }
    if (command !== process.execPath || !Array.isArray(args) || args.some((arg) => typeof arg !== "string" || arg.includes("\0"))) {
      throw new Error("Native workers may launch only this executable with string arguments");
    }
    if (options.stdio !== undefined && (!Array.isArray(options.stdio) ||
        options.stdio.length !== 3 || options.stdio.some((entry) => entry !== "pipe"))) {
      throw new Error("Native workers require three pipe streams");
    }
    const env = options.env || process.env;
    // helper.js forwards the original Node script slot. Remove exactly that synthetic slot for
    // our explicit worker role; do not rewrite arbitrary user arguments or controller invocations.
    if (env["MOM_CLUSTER_WORKER"] === "1") {
      if (args[0] !== process.execPath) {throw new Error("Native worker is missing its script slot");}
      args = args.slice(1);
    }
    return new ChildProcess(args, env);
  },
  spawnSync() {throw new Error("Native child_process.spawnSync is not implemented");},
};
