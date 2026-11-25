"use strict";

const {EventEmitter} = require("node:events");
const {StringDecoder} = require("node:string_decoder");

// Node's default stream high-water mark: write() reports backpressure at or above it, and "drain"
// follows once acknowledged writes bring the queue back below it.
const highWaterMark = 65536;
// Unacknowledged bytes are owned by the loop until it reports completion. Exceeding this cap is a
// caller bug (it ignored backpressure), so write() throws instead of queueing unbounded memory.
const maxQueuedBytes = 32 * 1024 * 1024;

/** Read decoding and acknowledged-write accounting shared by native sockets and process pipes.
 * The loop acknowledges every accepted write with one "write" event, in submission order, so a
 * FIFO of sizes tracks writableLength. Enqueue, error and close semantics stay per subclass:
 * sockets hold writes until connected/authenticated and fail queued callbacks with the first
 * socket error, while pipes submit at once and fail them with a fixed close error. */
class ByteStream extends EventEmitter {
  constructor() {
    super();
    this.writableLength = 0;
    /** @type {StringDecoder | null} */
    this.decoder = null;
    /** @type {Array<{size: number, callback?: (error?: Error) => void}>} */
    this.writes = [];
  }

  /** Emit one received chunk, decoded when setEncoding selected an encoding. @param {unknown} value */
  receiveData(value) {
    const buffer = Buffer.from(/** @type {ArrayBuffer} */ (value));
    const decoded = this.decoder ? this.decoder.write(buffer) : buffer;
    if (decoded.length) {this.emit("data", decoded);}
  }

  /** Retire the oldest write; its callback runs before any resulting "drain". */
  receiveWrite() {
    const entry = this.writes.shift();
    const blocked = this.writableLength >= highWaterMark;
    this.writableLength -= entry?.size || 0;
    entry?.callback?.();
    if (blocked && this.writableLength < highWaterMark) {this.emit("drain");}
  }

  /** Flush a partial multi-byte character before "end". */
  receiveEnd() {
    const tail = this.decoder?.end();
    if (tail) {this.emit("data", tail);}
    this.emit("end");
  }

  /** Count bytes the caller is queueing. @param {number} size @param {string} label */
  reserve(size, label) {
    if (size > maxQueuedBytes - this.writableLength) {throw new Error(label + " write queue exceeded 32 MiB");}
    this.writableLength += size;
  }

  /** @param {BufferEncoding} encoding */
  setEncoding(encoding) {this.decoder = new StringDecoder(encoding); return this;}
}

module.exports = {ByteStream, highWaterMark};
