"use strict";

const {EventEmitter} = require("node:events");
const {StringDecoder} = require("node:string_decoder");
const bytes = require("./bytes");

// One-direction process pipes use the loop's existing owned stream/write machinery. Stdin is
// opened lazily: an unused input stream must neither steal fd 0 nor keep the controller alive.
class Pipe extends EventEmitter {
  /** @param {boolean} readable @param {number | null} [id] */
  constructor(readable, id = null) {
    super();
    this.id = id;
    this.readable = readable;
    this.writable = !readable;
    this.destroyed = false;
    this.writableLength = 0;
    /** @type {StringDecoder | null} */
    this.decoder = null;
    /** @type {Array<{size: number, callback?: (error?: Error) => void}>} */
    this.writes = [];
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
      const buffer = Buffer.from(/** @type {ArrayBuffer} */ (value));
      const decoded = this.decoder ? this.decoder.write(buffer) : buffer;
      if (decoded.length) {this.emit("data", decoded);}
    } else if (event === "write") {
      const entry = this.writes.shift();
      const blocked = this.writableLength >= 65536;
      this.writableLength -= entry?.size || 0;
      entry?.callback?.();
      if (blocked && this.writableLength < 65536) {this.emit("drain");}
    } else if (event === "end") {
      this.readable = false;
      const tail = this.decoder?.end();
      if (tail) {this.emit("data", tail);}
      this.emit("end");
    } else if (event === "error") {
      this.emit("error", value);
    } else if (event === "finish") {
      this.emit("finish");
    } else if (event === "close") {
      this.destroyed = true; this.writable = false; this.readable = false;
      const error = new Error("Pipe closed before write completed");
      const pending = this.writes; this.writes = []; this.writableLength = 0;
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
    if (value.length > 32 * 1024 * 1024 - this.writableLength) {throw new Error("Pipe write queue exceeded 32 MiB");}
    this.writes.push({size: value.length, ...(callback ? {callback} : {})});
    this.writableLength += value.length;
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
  /** @param {BufferEncoding} encoding */
  setEncoding(encoding) {this.decoder = new StringDecoder(encoding); return this;}
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
