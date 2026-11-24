"use strict";

const {isPositiveNumber, isSafeIntegerAtLeast} = require("./validate");

// Keep proof admission and worker framing on the same raw-byte budget.
const MAX_PEARL_PROOF_BYTES = 8 * 1024 * 1024;
const MAX_PEARL_PROOF_BASE64 = Math.ceil(MAX_PEARL_PROOF_BYTES / 3) * 4;
const MAX_PROOF_EVENT_OVERHEAD = 64 * 1024;
const MAX_WORKER_ID_CHARS = 4096;

const jobIntegerMinimums = Object.entries({
  extra_nonce2_size: 0, height: 0, nonce1_len: 0, nonceoffset: 0, thread_id: 0,
  intensity: 1, noncebytes: 1, pearlhash_k: 1, pearlhash_n: 1,
  pearlhash_rank: 1, proofsize: 1, thread_num: 1,
});

// Pool jobs and native worker jobs share these numeric fields. Every field is optional here.
/** @param {Record<string, unknown>} job */
function hasValidJobNumbers(job) {
  for (const [field, minimum] of jobIntegerMinimums) {
    const value = job[field];
    if (value !== undefined && !isSafeIntegerAtLeast(value, minimum)) {return false;}
  }
  const difficulty = job["difficulty"];
  const certVersion = job["pearlhash_cert_version"];
  return (difficulty === undefined || isPositiveNumber(difficulty)) &&
    (certVersion === undefined || certVersion === 3);
}

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
  return isNonceHex(value) || isSafeIntegerAtLeast(value, 0);
}

module.exports = {
  MAX_PEARL_PROOF_BYTES, MAX_PEARL_PROOF_BASE64, MAX_PROOF_EVENT_OVERHEAD, MAX_WORKER_ID_CHARS,
  hasValidJobNumbers, isIdentifier, isJobId, isNonceHex, isJobNonce,
};
