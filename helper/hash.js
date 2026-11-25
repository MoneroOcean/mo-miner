"use strict";

const hashrateUnits = [
  {value: 1000000000000000, suffix: "PH/s"},
  {value: 1000000000000, suffix: "TH/s"},
  {value: 1000000000, suffix: "GH/s"},
  {value: 1000000, suffix: "MH/s"},
  {value: 1000, suffix: "KH/s"},
];

/** @param {string | number} hashrate */
module.exports.formatHashrate = function(hashrate) {
  const rate = Number.parseFloat(String(hashrate));
  if (!Number.isFinite(rate)) {return String(hashrate);}
  for (const unit of hashrateUnits) {
    if (Math.abs(rate) >= unit.value) {return (rate / unit.value).toFixed(2) + " " + unit.suffix;}
  }
  return rate.toFixed(2) + " H/s";
};

const hashCountUnits = [
  {value: 1000000000000000000n, suffix: "EH"},
  {value: 1000000000000000n, suffix: "PH"},
  {value: 1000000000000n, suffix: "TH"},
  {value: 1000000000n, suffix: "GH"},
  {value: 1000000n, suffix: "MH"},
  {value: 1000n, suffix: "KH"},
];

/** @param {bigint} count @param {{value: bigint, suffix: string}} unit */
function formatHashCountValue(count, unit) {
  const scaled = (count * 100n + unit.value / 2n) / unit.value;
  const whole = scaled / 100n;
  const fraction = scaled % 100n;
  return whole.toString() + "." + fraction.toString().padStart(2, "0") + " " + unit.suffix;
}

/** @param {bigint | number | string} hashes */
module.exports.formatHashCount = function(hashes) {
  let count = typeof hashes === "bigint" ? hashes : BigInt(Math.round(Number(hashes) || 0));
  if (count < 0n) {count = 0n;} // counts are unsigned; clamp negatives to zero
  for (const unit of hashCountUnits) {
    if (count >= unit.value) {return formatHashCountValue(count, unit);}
  }
  return count.toString() + " H";
};

// pack opt.default_msrs so it can be more easily passed into the compute core
/** @param {Record<string, MsrSetting>} msrs @returns {Record<string, string>} */
module.exports.pack_msr = function(msrs) {
  /** @type {Record<string, string>} */
  const packed = {};
  for (const [key, val] of Object.entries(msrs)) {
    packed["msr:" + key] = val.value + "," + val.mask;
  }
  return packed;
};

/** @param {Record<string, string>} msrs @returns {Record<string, MsrSetting>} */
module.exports.unpack_msr = function(msrs) {
  /** @type {Record<string, MsrSetting>} */
  const unpacked = {};
  for (const [key, val] of Object.entries(msrs)) {
    if (!key.startsWith("msr:0x")) {continue;}
    const parts = val.split(",");
    const [value, mask] = parts;
    if (parts.length !== 2 || value === undefined || mask === undefined) {continue;}
    unpacked[key.substring(4)] = {value, mask};
  }
  return unpacked;
};

/** @param {unknown} target @param {number} maxLength */
function targetHex(target, maxLength) {
  const hex = typeof target === "string" ? target.replace(/^0x/i, "") : "";
  if (!hex || hex.length > maxLength || /[^0-9a-f]/i.test(hex)) {
    throw new Error("Invalid target");
  }
  return hex;
}

// Difficulty of a CryptoNote-style compact target: up to 8 little-endian bytes, where the
// common 4-byte form carries only the high half of the 64-bit boundary.
/** @param {unknown} target */
module.exports.target2diff = function(target) {
  let hex = targetHex(target, 16);
  if (hex.length % 2 !== 0) {throw new Error("Invalid target");}
  if (hex.length === 8) {hex = "00000000" + hex;}
  // Reverse the little-endian bytes into big-endian hex for BigInt.
  const bytes = hex.match(/.{2}/g);
  if (!bytes) {throw new Error("Invalid target");}
  const div = BigInt("0x" + bytes.reverse().join(""));
  if (div === 0n) {return 0;}
  return BigInt("0xFFFFFFFFFFFFFFFF") / div;
};

/** @param {unknown} target */
module.exports.kawpowTarget2diff = function(target) {
  const hex = targetHex(target, 64);
  const div = BigInt("0x" + hex.slice(0, 16).padEnd(16, "0"));
  if (div === 0n) {return 0;}
  return BigInt("0xFFFFFFFFFFFFFFFF") / div;
};

const ETH_STRATUM_DIFF1_TARGET = BigInt("0x00000000ffff0000000000000000000000000000000000000000000000000000");
const UINT256_MAX = (1n << 256n) - 1n;

// Parse a decimal number or string (with optional exponent) exactly as [numerator, denominator].
/** @param {unknown} value @returns {[bigint, bigint]} */
function decimalToRatio(value) {
  if (typeof value !== "string" && typeof value !== "number" && typeof value !== "bigint") {
    throw new Error("Invalid decimal value");
  }
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new Error("Invalid decimal value");
  }
  const raw = String(value);
  if (raw.length > 4096) {throw new Error("Decimal value is too large");}
  const text = raw.trim().toLowerCase();
  const m = text.match(/^([+-])?(?:(\d+)(?:\.(\d*))?|\.(\d+))(?:e([+-]?\d+))?$/);
  if (!m) {throw new Error("Invalid decimal value");}
  const fraction = m[3] ?? m[4] ?? "";
  const digits = ((m[2] || "0") + fraction).replace(/^0+/, "") || "0";
  const exp = BigInt(m[5] || "0");
  // Targets/difficulties are 256-bit (< ~78 decimal digits); bound every decimal component before
  // exponentiation so hostile pool input cannot synchronously exhaust the event loop or memory.
  if (exp > 1000n || exp < -1000n || digits.length > 1000 || fraction.length > 1000) {
    throw new Error("Decimal value is too large");
  }
  const scale = BigInt(fraction.length);
  let numerator = BigInt(digits);
  let denominator = 10n ** scale;
  if (exp > 0n) {
    numerator *= 10n ** exp;
  } else if (exp < 0n) {
    denominator *= 10n ** (-exp);
  }
  if (m[1] === "-") {numerator = -numerator;}
  return [numerator, denominator];
}

// clamp a 256-bit target to UINT256_MAX and render as 64 lowercase hex chars
/** @param {bigint} target */
function target256ToHex(target) {
  return (target > UINT256_MAX ? UINT256_MAX : target).toString(16).padStart(64, "0");
}

// parse a (possibly 0x-prefixed, short) 256-bit target hex string into a BigInt
/** @param {unknown} target */
function parseTarget256(target) {
  const hex = targetHex(target, 64);
  return BigInt("0x" + hex.padStart(64, "0"));
}

/** @param {unknown} diff @param {bigint} [multiplier=1n] */
module.exports.ethDiff2Target = function(diff, multiplier = 1n) {
  if (typeof multiplier !== "bigint" || multiplier <= 0n) {
    throw new Error("Eth difficulty multiplier must be a positive bigint");
  }
  const [numerator, denominator] = decimalToRatio(diff);
  if (numerator <= 0n) {return "0".repeat(64);}
  return target256ToHex((ETH_STRATUM_DIFF1_TARGET * multiplier * denominator) / numerator);
};

/** @param {unknown} diff @param {bigint} [base=UINT256_MAX] */
module.exports.fullDiff2Target = function(diff, base = UINT256_MAX) {
  if (typeof base !== "bigint" || base <= 0n) {
    throw new Error("Full difficulty base must be a positive bigint");
  }
  const [numerator, denominator] = decimalToRatio(diff);
  if (numerator <= 0n) {return target256ToHex(UINT256_MAX);}
  return target256ToHex((base * denominator) / numerator);
};

/** @param {unknown} target */
module.exports.target256ToWork = function(target) {
  const div = parseTarget256(target);
  if (div === 0n) {return 0n;}
  return UINT256_MAX / div;
};

const PEARLHASH_MAX_DIMENSION = 1 << 24;
const PEARLHASH_MAX_K = 1 << 16;
const PEARLHASH_MAX_RANK = 1024;

// The one copy of the PearlHash m/n/k/rank limits, shared by target scaling here and by GPU tuning
// and job shapes in gpu-tuning.js. Absent fields (and the K/rank relation when either is absent)
// are skipped so partial tuning can be checked; complete shapes pass all four. The rank >= 128
// floor is also what certificate v3's rank-128 target normalization relies on.
/** @param {{m?: number, n?: number, k?: number, rank?: number}} shape @param {string} context */
function checkPearlHashLimits(shape, context) {
  /** @type {("m" | "n")[]} */
  const matrixFields = ["m", "n"];
  for (const field of matrixFields) {
    const value = shape[field];
    if (value !== undefined && (!Number.isSafeInteger(value) ||
        value < 128 || value > PEARLHASH_MAX_DIMENSION || value % 32 !== 0)) {
      throw new Error(`${context}.${field} must be a multiple of 32 between 128 and ` +
        PEARLHASH_MAX_DIMENSION);
    }
  }
  const {k, rank} = shape;
  if (k !== undefined &&
      (!Number.isSafeInteger(k) || k < 1024 || k > PEARLHASH_MAX_K || k % 64 !== 0)) {
    throw new Error(`${context}.k must be a multiple of 64 between 1024 and ${PEARLHASH_MAX_K}`);
  }
  if (rank !== undefined && (!Number.isSafeInteger(rank) ||
      rank < 128 || rank > PEARLHASH_MAX_RANK || (rank & (rank - 1)) !== 0)) {
    throw new Error(`${context}.rank must be a power of two between 128 and ${PEARLHASH_MAX_RANK}`);
  }
  if (k !== undefined && rank !== undefined && (k < 16 * rank || k > 4 * rank * rank)) {
    throw new Error(`${context}.k must be between 16*rank and 4*rank^2`);
  }
}
module.exports.checkPearlHashLimits = checkPearlHashLimits;

/** @param {number} k @param {number} rank */
function pearlhashEffectiveK(k, rank) {
  if (k === undefined || rank === undefined) {throw new Error("PearlHash K/rank is missing");}
  checkPearlHashLimits({k, rank}, "PearlHash");
  return k - (k % rank);
}

/** @param {unknown} baseTarget @param {number} k @param {number} rank
 * @param {number} [certVersion] */
module.exports.pearlhashTarget = function(baseTarget, k, rank, certVersion = 3) {
  if (certVersion !== 3) {
    throw new Error("Unsupported PearlHash certificate version");
  }
  const base = parseTarget256(baseTarget);
  // V3 normalizes the target to rank 128. The miner reports rank-128-equivalent GEMM MACs.
  const effectiveK = pearlhashEffectiveK(k, rank);
  const scale = 16n * 16n * BigInt(effectiveK / rank * 128);
  return target256ToHex(base * scale);
};

/** @param {unknown} target @param {number} k @param {number} rank */
module.exports.pearlhashTargetWork = function(target, k, rank) {
  const scale = 16n * 16n * BigInt(pearlhashEffectiveK(k, rank) / rank * 128);
  return module.exports.target256ToWork(target) * scale;
};

// Inverse of target2diff: diff -> little-endian compact target hex, shortened to the 4-byte form
// when the boundary's low four bytes are zero.
/** @param {string | number | bigint} diff */
module.exports.diff2target = function(diff) {
  const d = BigInt(diff);
  if (d <= 0n) {return "0000000000000000";}
  const hexBE = (BigInt("0xFFFFFFFFFFFFFFFF") / d).toString(16).padStart(16, "0");
  const bytes = hexBE.match(/.{2}/g);
  if (!bytes) {throw new Error("Invalid difficulty");}
  const hexLE = bytes.reverse().join("");
  return hexLE.startsWith("00000000") ? hexLE.slice(8) : hexLE;
};

/** @param {string} hex */
module.exports.edge_hex2arr = function(hex) {
  if (!/^(?:[0-9a-f]{8})+$/i.test(hex)) {throw new Error("Invalid edge proof");}
  const pow = [];
  for (let i = 0; i < hex.length; i += 8) {pow.push(Number.parseInt(hex.slice(i, i + 8), 16));}
  return pow;
};
