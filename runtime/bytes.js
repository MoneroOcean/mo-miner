"use strict";

/** Normalize the controller's Buffer/string/view inputs without losing a view's offset or size.
 * Only strings and actual ArrayBuffer views may cross the native byte-pointer boundary.
 * @param {unknown} value @param {BufferEncoding} [encoding] */
module.exports = function bytes(value, encoding) {
  if (typeof value === "string") {return Buffer.from(value, encoding);}
  if (Buffer.isBuffer(value)) {return value;}
  if (ArrayBuffer.isView(value)) {return Buffer.from(value.buffer, value.byteOffset, value.byteLength);}
  throw Object.assign(new TypeError("Expected string, Buffer or ArrayBuffer view"), {code: "ERR_INVALID_ARG_TYPE"});
};
