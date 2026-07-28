"use strict";

function hexWithoutPrefix(value) {
  return String(value || "").replace(/^0x/i, "");
}

function normalizedFullNonce(value) {
  return hexWithoutPrefix(value).padStart(16, "0").slice(-16);
}

function reverseHexBytes(value) {
  const hex = hexWithoutPrefix(value);
  return hex.length % 2 === 0 ? (hex.match(/.{2}/g) || []).reverse().join("") : hex;
}

function ergSubmitMeta(pool, job_id) {
  return (pool.erg_submit_jobs && pool.erg_submit_jobs[job_id]) || {};
}

function ergExtraNonce2Size(pool, meta) {
  const size = Number(meta.extra_nonce2_size);
  if (Number.isInteger(size) && size >= 0 && size <= 8) {return size;}

  const extraNonce = hexWithoutPrefix(meta.extra_nonce || pool.extra_nonce || "");
  return Math.max(0, 8 - Math.ceil(extraNonce.length / 2));
}

function ergSubmitParams(pool, value) {
  const meta = ergSubmitMeta(pool, value.job_id);
  const nonce = normalizedFullNonce(value.nonce);
  const extraNonce2HexLength = ergExtraNonce2Size(pool, meta) * 2;
  const extraNonce2 = extraNonce2HexLength ? nonce.slice(16 - extraNonce2HexLength) : "";
  return [pool.login, value.job_id, extraNonce2, hexWithoutPrefix(meta.ntime), nonce];
}

// Little-endian byte order of an 8-byte counter expressed as big-endian hex (the native emits the
// search counter via %016 PRIx64 = big-endian; the wire/header stores it little-endian, matching the
// memcpy of m_nonce64 into the header at nonceoffset).
function counterHexToWireLE(nonceHex) {
  return normalizedFullNonce(nonceHex).match(/.{2}/g).reverse().join("");
}

// The submit nonce2 = the 32-byte header nonce after the pool's nonce1 prefix. The header nonce is
// nonce1 (nonce1_len bytes) || nonce2; the solver advances an 8-byte counter at the start of nonce2,
// the remaining nonce2 bytes stay as the job delivered them (zeros). Returns wire-order hex.
function zelhashNonce2(pool, nonceHex) {
  const job = (pool && pool.last_job) || {};
  const nonce1_len = Number(job.nonce1_len) || 0;
  // full 32-byte nonce (64 hex) lives at the end of the 280-hex header blob
  const blob = hexWithoutPrefix(job.blob || job.blob_hex || "");
  const fullNonce = blob.slice(-64).padEnd(64, "0");
  const counterLE = counterHexToWireLE(nonceHex);   // 8-byte search counter, wire (LE) order
  // nonce2 = the counter (start of nonce2) || the nonce2 tail past the counter (job-delivered zeros);
  // the nonce1 prefix (fullNonce[0 .. nonce1_len]) is intentionally excluded per ZIP-301.
  const tail = fullNonce.slice(nonce1_len * 2 + 16);
  return (counterLE + tail).padEnd(64 - nonce1_len * 2, "0");
}

function zelhashSubmitNtime(pool) {
  const job = (pool && pool.last_job) || {};
  return hexWithoutPrefix(job.ntime || "");
}

module.exports = {
  ergSubmitParams,
  hexWithoutPrefix,
  reverseHexBytes,
  zelhashNonce2,
  zelhashSubmitNtime,
};
