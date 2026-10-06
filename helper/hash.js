"use strict";

const hashrate_units = [
  { value: 1000000000000000, suffix: "PH/s" },
  { value: 1000000000000, suffix: "TH/s" },
  { value: 1000000000, suffix: "GH/s" },
  { value: 1000000, suffix: "MH/s" },
  { value: 1000, suffix: "KH/s" },
];
const hashrate_unit_multipliers = Object.fromEntries([
  ...hashrate_units.map((unit) => [unit.suffix, unit.value]),
  ["H/s", 1],
]);

module.exports.hashrate_units = Object.keys(hashrate_unit_multipliers);

module.exports.formatHashrate = function(hashrate) {
  const rate = Number.parseFloat(hashrate);
  if (!Number.isFinite(rate)) {return String(hashrate);}
  for (const unit of hashrate_units) {
    if (Math.abs(rate) >= unit.value) {return (rate / unit.value).toFixed(2) + " " + unit.suffix;}
  }
  return rate.toFixed(2) + " H/s";
};

const hash_count_units = [
  { value: 1000000000000000000n, suffix: "EH" },
  { value: 1000000000000000n, suffix: "PH" },
  { value: 1000000000000n, suffix: "TH" },
  { value: 1000000000n, suffix: "GH" },
  { value: 1000000n, suffix: "MH" },
  { value: 1000n, suffix: "KH" },
];

function formatHashCountValue(count, unit) {
  const scaled = (count * 100n + unit.value / 2n) / unit.value;
  const whole = scaled / 100n;
  const fraction = scaled % 100n;
  return whole.toString() + "." + fraction.toString().padStart(2, "0") + " " + unit.suffix;
}

module.exports.formatHashCount = function(hashes) {
  let count = typeof hashes === "bigint" ? hashes : BigInt(Math.round(Number(hashes) || 0));
  if (count < 0n) {count = 0n;} // counts are unsigned; clamp negatives to zero
  for (const unit of hash_count_units) {
    if (count >= unit.value) {return formatHashCountValue(count, unit);}
  }
  return count.toString() + " H";
};

module.exports.parseFormattedHashrate = function(value, unit) {
  const rate = Number.parseFloat(value);
  const multiplier = hashrate_unit_multipliers[unit];
  return Number.isFinite(rate) && multiplier ? rate * multiplier : Number.NaN;
};

// pack opt.default_msrs so it can be more easily passed into the compute core
module.exports.pack_msr = function(default_msr) {
  const packed = {};
  for (const [key, val] of Object.entries(default_msr)) {
    packed["msr:" + key] = val.value + "," + val.mask;
  }
  return packed;
};

module.exports.unpack_msr = function(default_msr) {
  const unpacked = {};
  for (const [key, val] of Object.entries(default_msr)) {
    if (!key.startsWith("msr:0x")) {continue;}
    const parts = val.split(",");
    if (parts.length !== 2) {continue;}
    unpacked[key.substring(4)] = { value: parts[0], mask: parts[1] };
  }
  return unpacked;
};

module.exports.target2diff = function(target) {
  if (target.length === 8) {target = "00000000" + target;}
  // target is stored big-endian; reverse byte pairs to read it as a LE integer
  const div = BigInt("0x" + target.match(/.{2}/g).reverse().join(""));
  if (div === 0n) {return 0;}
  return BigInt("0xFFFFFFFFFFFFFFFF") / div;
};

module.exports.kawpowTarget2diff = function(target) {
  const div = BigInt("0x" + target.slice(0, 16).padEnd(16, "0"));
  if (div === 0n) {return 0;}
  return BigInt("0xFFFFFFFFFFFFFFFF") / div;
};

const ETH_STRATUM_DIFF1_TARGET = BigInt("0x00000000ffff0000000000000000000000000000000000000000000000000000");
const UINT256_MAX = (1n << 256n) - 1n;

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
function target256ToHex(target) {
  return (target > UINT256_MAX ? UINT256_MAX : target).toString(16).padStart(64, "0");
}

function targetHex(target, maxLength) {
  const hex = typeof target === "string" ? target.replace(/^0x/i, "") : "";
  if (!hex || hex.length > maxLength || /[^0-9a-f]/i.test(hex)) {
    throw new Error("Invalid target");
  }
  return hex;
}

// parse a (possibly 0x-prefixed, short) 256-bit target hex string into a BigInt
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

module.exports.decimalTargetToHex = function(value) {
  const [numerator, denominator] = decimalToRatio(value);
  if (numerator <= 0n) {return "0".repeat(64);}
  return target256ToHex(numerator / denominator);
};

module.exports.ethTarget2diff = function(target) {
  const div = parseTarget256(target);
  if (div === 0n) {return 0;}
  return Number(ETH_STRATUM_DIFF1_TARGET) / Number(div);
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

module.exports.target256ToWork = function(target) {
  const div = parseTarget256(target);
  if (div === 0n) {return 0n;}
  return UINT256_MAX / div;
};

function pearlhashEffectiveK(k, rank) {
  if (!Number.isSafeInteger(k) || k < 1024 || k > 0x10000 || k % 64 !== 0) {
    throw new Error("Invalid PearlHash K");
  }
  if (!Number.isSafeInteger(rank) || rank < 128 || rank > 1024 ||
      (rank & (rank - 1)) !== 0) {
    throw new Error("Invalid PearlHash rank");
  }
  if (k < 16 * rank || k > 4 * rank * rank) {
    throw new Error("Invalid PearlHash K/rank");
  }
  return k - (k % rank);
}

/** @param {unknown} baseTarget @param {number} k @param {number} rank
 * @param {number} [certVersion] */
module.exports.pearlhashTarget = function(baseTarget, k, rank, certVersion = 3) {
  if (certVersion !== 3) {
    throw new Error("Unsupported PearlHash certificate version");
  }
  if (rank < 128) {
    throw new Error("PearlHash certificate version 3 requires rank >= 128");
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

// Inverse of target2diff: diff -> compact BE target hex (4 bytes when it fits).
module.exports.diff2target = function(diff) {
  const d = BigInt(diff);
  if (d <= 0n) {return "0000000000000000";}
  const hexLE = (BigInt("0xFFFFFFFFFFFFFFFF") / d).toString(16).padStart(16, "0");
  const hexBE = hexLE.match(/.{2}/g).reverse().join("");
  // Drop the high 4 zero bytes to match the original compact-target style.
  return hexBE.startsWith("00000000") ? hexBE.slice(8) : hexBE;
};

module.exports.edge_hex2arr = function(hex) {
  const pow = [];
  for (let i = 0; i < hex.length; i += 8) {pow.push(Number.parseInt(hex.slice(i, i + 8), 16));}
  return pow;
};
