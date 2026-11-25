"use strict";

const bytes = require("./bytes");
const {ByteStream, highWaterMark} = require("./byte-stream");

// Native mode alone resolves node:net here. Controller/pool sources and Engineering Mode remain
// unchanged. OS handles stay private integers; all callbacks run on the language runtime thread.
class Socket extends ByteStream {
  /** @param {{host: string, port: number, autoSelectFamilyAttemptTimeout?: number | null}} options */
  constructor(options) {
    super();
    this.connecting = true;
    this.destroyed = false;
    this.writable = true;
    // Writes made before readiness; their bytes already count toward writableLength.
    /** @type {Array<{data: Buffer, callback?: (error?: Error) => void}>} */
    this.pending = [];
    /** @type {Error | undefined} */
    this.failure = undefined;
    this.ending = false;
    const requestedTimeout = options.autoSelectFamilyAttemptTimeout;
    if (requestedTimeout !== undefined && requestedTimeout !== null &&
        (typeof requestedTimeout !== "number" || !Number.isInteger(requestedTimeout) ||
         requestedTimeout < 1 || requestedTimeout > 2147483647)) {
      const error = Object.assign(new RangeError("The value of autoSelectFamilyAttemptTimeout is out of range"), {
        code: "ERR_OUT_OF_RANGE",
      });
      throw error;
    }
    const attemptTimeout = requestedTimeout === undefined || requestedTimeout === null
      ? requestedTimeout : Math.max(requestedTimeout, 10);
    this.id = globalThis.__momTCPConnect(options.host, options.port,
      (event, value) => this.receive(event, value), attemptTimeout);
  }

  /** @param {string} event @param {unknown} value */
  receive(event, value) {
    if (event === "connect") {
      this.connecting = false;
      this.flushPending();
      this.emit("connect");
    } else if (event === "data") {
      this.receiveData(value);
    } else if (event === "write") {
      this.receiveWrite();
    } else if (event === "error") {
      this.failure = value instanceof Error ? value : new Error(String(value));
      this.emit("error", this.failure);
    } else if (event === "end") {
      this.receiveEnd();
    } else if (event === "finish") {
      this.emit("finish");
    } else if (event === "close") {
      this.connecting = false;
      this.destroyed = true;
      this.writable = false;
      const error = this.failure || new Error("Socket closed before write completion");
      for (const entry of [...this.pending, ...this.writes]) {entry.callback?.(error);}
      this.pending = [];
      this.writes.length = 0;
      this.writableLength = 0;
      this.emit("close", value);
    }
  }

  // TLS overrides readiness until authentication completes, even after TCP emits connect.
  ready() {return !this.connecting;}
  flushPending() {
    while (this.pending.length && !this.destroyed) {
      const entry = this.pending.shift();
      if (entry) {this.send(entry.data, entry.callback);}
    }
    if (!this.destroyed && this.ending) {globalThis.__momTCPEnd(this.id);}
  }

  /** @param {Buffer} data @param {((error?: Error) => void) | undefined} callback */
  send(data, callback) {
    // Match stream error timing even when the OS bridge rejects a write synchronously.
    this.writes.push({size: data.length, ...(callback ? {callback} : {})});
    try {
      return globalThis.__momTCPWrite(this.id, data);
    } catch (error) {
      this.writes.pop();
      this.writableLength -= data.length;
      this.failure = error instanceof Error ? error : new Error(String(error));
      this.destroy();
      process.nextTick(() => {callback?.(this.failure); this.emit("error", this.failure);});
      return false;
    }
  }

  /** @param {string | ArrayBufferView} data
   * @param {BufferEncoding | ((error?: Error) => void)} [encoding]
   * @param {(error?: Error) => void} [callback] */
  write(data, encoding, callback) {
    if (typeof encoding === "function") {callback = encoding;}
    const value = bytes(data, typeof encoding === "string" ? encoding : "utf8");
    if (this.destroyed || !this.writable) {
      const error = Object.assign(new Error("Socket is not writable"), {
        code: this.destroyed ? "ERR_STREAM_DESTROYED" : "ERR_STREAM_WRITE_AFTER_END",
      });
      process.nextTick(() => {
        callback?.(error);
        if (!this.destroyed) {this.destroy(); this.emit("error", error);}
      });
      return false;
    }
    this.reserve(value.length, "Socket");
    if (!this.ready()) {
      this.pending.push({data: value, ...(callback ? {callback} : {})});
      return this.writableLength < highWaterMark;
    }
    return this.send(value, callback);
  }

  /** @param {string | ArrayBufferView} [data] */
  end(data) {
    if (data !== undefined) {this.write(data);}
    this.writable = false;
    this.ending = true;
    if (this.ready()) {globalThis.__momTCPEnd(this.id);}
    return this;
  }

  destroy() {this.destroyed = true; this.writable = false; globalThis.__momHandleClose(this.id); return this;}
  ref() {globalThis.__momHandleRef(this.id, true); return this;}
  unref() {globalThis.__momHandleRef(this.id, false); return this;}
}

/** @param {{host: string, port: number}} options @param {() => void} [callback] */
function connect(options, callback) {
  const socket = new Socket(options);
  if (callback) {socket.once("connect", callback);}
  return socket;
}

/** @param {unknown} value */
function isIP(value) {
  // Native mode deliberately accepts only strings; this avoids coercing arbitrary objects while
  // preserving the maintained controller's literal-host call path.
  return typeof value === "string" ? globalThis.__momIsIP(value) : 0;
}

module.exports = {Socket, connect, createConnection: connect, isIP};
