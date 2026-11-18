"use strict";

// Keep proof admission and worker framing on the same raw-byte budget.
const MAX_PEARL_PROOF_BYTES = 8 * 1024 * 1024;
const MAX_PEARL_PROOF_BASE64 = Math.ceil(MAX_PEARL_PROOF_BYTES / 3) * 4;
const MAX_PROOF_EVENT_OVERHEAD = 64 * 1024;

// Pool, worker, and native job identifiers are opaque strings or safe integers.
/** @param {unknown} value @returns {value is string | number} */
function isIdentifier(value) {
  return typeof value === "string" || (typeof value === "number" && Number.isSafeInteger(value));
}

// A pool job ID as received from the wire: non-empty and bounded when it is a string.
/** @param {unknown} value @returns {value is string | number} */
function isJobId(value) {
  return typeof value === "string" ? value.length > 0 && value.length <= 256 : isIdentifier(value);
}

// A nonce or nonce mask in the 1..8-byte hexadecimal form workers report.
/** @param {unknown} value @returns {value is string} */
function isNonceHex(value) {
  return typeof value === "string" && /^[0-9a-f]{1,16}$/i.test(value);
}

// Jobs may also carry a resumable nonce as a non-negative safe integer.
/** @param {unknown} value @returns {value is string | number} */
function isJobNonce(value) {
  return isNonceHex(value) || (typeof value === "number" && Number.isSafeInteger(value) && value >= 0);
}

module.exports = {
  MAX_PEARL_PROOF_BYTES, MAX_PEARL_PROOF_BASE64, MAX_PROOF_EVENT_OVERHEAD,
  isIdentifier, isJobId, isNonceHex, isJobNonce,
};
