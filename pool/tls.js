"use strict";

/** SHA-256 of the leaf certificate's DER, not SPKI or a CA. Accept XMRig's plain hex and
 * Node/OpenSSL's colon-separated form, but never discard malformed input characters.
 * @param {unknown} value @returns {string} */
function normalizeFingerprint(value) {
  if (value === undefined || value === "") {return "";}
  if (typeof value !== "string" || !/^(?:[a-f0-9]{64}|(?:[a-f0-9]{2}:){31}[a-f0-9]{2})$/i.test(value)) {
    throw new Error("tls_fingerprint must be a SHA-256 certificate fingerprint (64 hex digits)");
  }
  return value.replace(/:/g, "").toLowerCase();
}

module.exports = {normalizeFingerprint};
