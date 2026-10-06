"use strict";

/** @param {unknown} value */
function hexWithoutPrefix(value) {
  if (value === undefined || value === null) {return "";}
  if (typeof value !== "string") {throw new Error("Invalid hexadecimal value");}
  return value.replace(/^0x/i, "");
}

/** @param {unknown} value */
function normalizedFullNonce(value) {
  const hex = hexWithoutPrefix(value);
  if (!/^[0-9a-f]{1,16}$/i.test(hex)) {throw new Error("Invalid nonce");}
  return hex.padStart(16, "0");
}

/** @param {unknown} value @param {number} noncebytes @param {string} [prefix]
 * @param {boolean} [requireNonzero] */
function isValidNonce(value, noncebytes, prefix = "", requireNonzero = false) {
  if (noncebytes !== 4 && noncebytes !== 8) {return false;}
  if (typeof value === "number" && (!Number.isSafeInteger(value) || value < 0)) {return false;}
  const hex = typeof value === "number" ? value.toString(16) : value;
  const width = noncebytes * 2;
  return typeof hex === "string" && /^[0-9a-f]+$/i.test(hex) && hex.length <= width &&
    (!requireNonzero || /[1-9a-f]/i.test(hex)) &&
    hex.padStart(width, "0").toLowerCase().startsWith(prefix.slice(0, width).toLowerCase());
}

/** @param {unknown} value */
function reverseHexBytes(value) {
  const hex = hexWithoutPrefix(value);
  if (!/^[0-9a-f]+$/i.test(hex) || hex.length % 2 !== 0) {throw new Error("Invalid nonce");}
  const bytes = hex.match(/.{2}/g);
  if (!bytes) {throw new Error("Invalid nonce");}
  return bytes.reverse().join("");
}

/** @param {Pick<PoolConfig, "login">} pool @param {PoolJob} job @param {ShareResult} value */
function ergSubmitParams(pool, job, value) {
  const size = job.extra_nonce2_size;
  if (size === undefined) {return null;}
  const nonce = normalizedFullNonce(value.nonce);
  const extraNonce2HexLength = size * 2;
  const extraNonce2 = extraNonce2HexLength ? nonce.slice(16 - extraNonce2HexLength) : "";
  return [pool.login, job.job_id, extraNonce2, hexWithoutPrefix(job.ntime), nonce];
}

/** @param {Pick<PoolConfig, "login">} pool @param {PoolJob} job @param {ShareResult} value */
function nexaSubmitParams(pool, job, value) {
  const extraNonce = hexWithoutPrefix(job.extra_nonce);
  if (job.extra_nonce2_size === 4) {
    const nonce = normalizedFullNonce(value.nonce);
    if (!/^[0-9a-f]{8}$/i.test(extraNonce)) {throw new Error("Invalid Nexa fixed nonce prefix");}
    // Four-field jobs have no time field. This dialect ignores the timestamp slot and hashes
    // the concatenation of submit fields 2 and 4 (MagicPool's WildRig-compatible ParseWork):
    // fixed extranonce(4) || complete worker nonce(8).
    return [pool.login, job.job_id, extraNonce, "00000000", nonce];
  }
  return [
    pool.login,
    job.job_id,
    extraNonce + normalizedFullNonce(value.nonce),
    hexWithoutPrefix(job.ntime),
  ];
}

// The submit nonce2 = the 32-byte header nonce after the pool's nonce1 prefix. The header nonce is
// nonce1 (nonce1_len bytes) || nonce2; the solver advances an 8-byte counter at the start of nonce2,
// the remaining nonce2 bytes stay as the job delivered them (zeros). Returns wire-order hex.
/** @param {PoolJob} job @param {unknown} nonceHex */
function zelhashNonce2(job, nonceHex) {
  const nonce1_len = job.nonce1_len ?? 0;
  if (!Number.isSafeInteger(nonce1_len) || nonce1_len < 0 || nonce1_len > 24) {
    throw new Error("Invalid ZelHash nonce prefix length");
  }
  // full 32-byte nonce (64 hex) lives at the end of the 280-hex header blob
  const blob = hexWithoutPrefix(job.blob || job.blob_hex || "");
  if (!/^[0-9a-f]{280}$/i.test(blob)) {throw new Error("Invalid ZelHash job blob");}
  const fullNonce = blob.slice(-64);
  // The native prints the counter big-endian; the header stores its eight bytes little-endian.
  const counterLE = reverseHexBytes(normalizedFullNonce(nonceHex));
  // nonce2 = the counter (start of nonce2) || the nonce2 tail past the counter (job-delivered zeros);
  // the nonce1 prefix (fullNonce[0 .. nonce1_len]) is intentionally excluded per ZIP-301.
  const tail = fullNonce.slice(nonce1_len * 2 + 16);
  return (counterLE + tail).padEnd(64 - nonce1_len * 2, "0");
}

/** @param {Pick<PoolConfig, "login">} pool @param {PoolJob} job @param {ShareResult} value */
function verthashSubmitParams(pool, job, value) {
  return [pool.login, job.job_id, job.extranonce2, job.ntime, value.nonce];
}

/** @param {Pick<PoolConfig, "login">} pool @param {PoolJob} job @param {ShareResult} value @param {string} solution */
function zelhashSubmitParams(pool, job, value, solution) {
  return [pool.login, job.job_id, hexWithoutPrefix(job.ntime),
    zelhashNonce2(job, value.nonce), solution];
}

module.exports = {
  ergSubmitParams,
  nexaSubmitParams,
  hexWithoutPrefix,
  isValidNonce,
  reverseHexBytes,
  verthashSubmitParams,
  zelhashSubmitParams,
};
