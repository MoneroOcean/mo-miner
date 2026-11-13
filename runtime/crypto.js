"use strict";

const bytes = require("./bytes");

module.exports = {
  /** @param {unknown} size */
  randomBytes(size) {
    if (typeof size !== "number") {
      throw Object.assign(new TypeError("Random byte count must be a number"), {code: "ERR_INVALID_ARG_TYPE"});
    }
    if (!Number.isInteger(size) || size < 0 || size > 0x7fffffff) {
      throw Object.assign(new RangeError("Invalid random byte count"), {code: "ERR_OUT_OF_RANGE"});
    }
    return Buffer.from(globalThis.__momRandomBytes(size));
  },
  /** Only SHA-256 is required by the controller; mining hashes use the shared C++/SYCL core.
   * @param {unknown} algorithm */
  createHash(algorithm) {
    if (typeof algorithm !== "string") {
      throw Object.assign(new TypeError("Hash algorithm must be a string"), {code: "ERR_INVALID_ARG_TYPE"});
    }
    if (!["sha256", "sha-256"].includes(algorithm.toLowerCase())) {throw new Error("Unsupported native hash algorithm");}
    const hash = globalThis.__momHashCreate();
    return {
      /** @param {unknown} data @param {BufferEncoding} [encoding] */
      update(data, encoding) {hash.update(bytes(data, encoding)); return this;},
      /** @param {BufferEncoding} [encoding] */
      digest(encoding) {
        const result = Buffer.from(hash.digest());
        return encoding ? result.toString(encoding) : result;
      },
    };
  },
};
