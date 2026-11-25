"use strict";

const bytes = require("./bytes");
const {ByteStream} = require("./byte-stream");

// One-direction process pipes use the loop's existing owned stream/write machinery. Stdin is
// opened lazily: an unused input stream must neither steal fd 0 nor keep the controller alive.
class Pipe extends ByteStream {
  /** @param {boolean} readable @param {number | null} [id] */
  constructor(readable, id = null) {
    super();
    this.id = id;
    this.readable = readable;
    this.writable = !readable;
    this.destroyed = false;
    this.on("newListener", (name) => {
      if (name === "data") {this.resume();}
    });
  }
  ensure() {
    if (this.id === null) {this.id = globalThis.__momStdinOpen((event, value) => this.receive(event, value));}
    return this.id;
  }
  /** @param {string} event @param {unknown} [value] */
  receive(event, value) {
    if (event === "data") {
      this.receiveData(value);
    } else if (event === "write") {
      this.receiveWrite();
    } else if (event === "end") {
      this.readable = false;
      this.receiveEnd();
    } else if (event === "error") {
      this.emit("error", value);
    } else if (event === "finish") {
      this.emit("finish");
    } else if (event === "close") {
      this.destroyed = true; this.writable = false; this.readable = false;
      const error = new Error("Pipe closed before write completed");
      const pending = this.writes.splice(0); this.writableLength = 0;
      for (const entry of pending) {entry.callback?.(error);}
      this.emit("close");
    }
  }
  /** @param {string | Uint8Array} data @param {BufferEncoding | ((error?: Error) => void)} [encoding]
   * @param {(error?: Error) => void} [callback] */
  write(data, encoding, callback) {
    if (typeof encoding === "function") {callback = encoding;}
    if (!this.writable || this.destroyed) {
      const error = Object.assign(new Error("Pipe is not writable"), {code: "ERR_STREAM_DESTROYED"});
      process.nextTick(() => {callback?.(error); this.emit("error", error);});
      return false;
    }
    const value = bytes(data, typeof encoding === "string" ? encoding : "utf8");
    this.reserve(value.length, "Pipe");
    this.writes.push({size: value.length, ...(callback ? {callback} : {})});
    try {return globalThis.__momTCPWrite(this.ensure(), value);} catch (failure) {
      this.writes.pop(); this.writableLength -= value.length;
      const error = failure instanceof Error ? failure : new Error(String(failure));
      this.destroy();
      process.nextTick(() => {callback?.(error); this.emit("error", error);});
      return false;
    }
  }
  /** @param {string | Uint8Array} [data] */
  end(data) {
    if (data !== undefined) {this.write(data);}
    this.writable = false;
    if (!this.destroyed) {globalThis.__momTCPEnd(this.ensure());}
    return this;
  }
  resume() {
    if (this.readable && !this.destroyed) {globalThis.__momPipeRead(this.ensure(), true);}
    return this;
  }
  pause() {if (this.id !== null && !this.destroyed) {globalThis.__momPipeRead(this.id, false);} return this;}
  destroy() {
    if (!this.destroyed) {
      this.destroyed = true; this.writable = false; this.readable = false;
      if (this.id !== null) {globalThis.__momHandleClose(this.id);}
    }
    return this;
  }
  ref() {if (this.id !== null) {globalThis.__momHandleRef(this.id, true);} return this;}
  unref() {if (this.id !== null) {globalThis.__momHandleRef(this.id, false);} return this;}
}
module.exports = Pipe;
