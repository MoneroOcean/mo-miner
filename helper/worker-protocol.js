"use strict";

// Keep proof admission and worker framing on the same raw-byte budget.
const MAX_PEARL_PROOF_BYTES = 8 * 1024 * 1024;
const MAX_PEARL_PROOF_BASE64 = Math.ceil(MAX_PEARL_PROOF_BYTES / 3) * 4;
const MAX_PROOF_EVENT_OVERHEAD = 64 * 1024;

module.exports = {MAX_PEARL_PROOF_BYTES, MAX_PEARL_PROOF_BASE64, MAX_PROOF_EVENT_OVERHEAD};
