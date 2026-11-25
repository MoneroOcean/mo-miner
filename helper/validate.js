"use strict";

// Shape checks for unknown values at pool, worker, native-core, and configuration boundaries.
// This module stays side-effect free so pool, miner, and option code can share it without loading
// the worker-process machinery in helper.js.

/** @param {unknown} value @returns {value is UnknownRecord} */
function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Text of a thrown value or a native `{message}` error event.
 * @param {unknown} value @returns {string} */
function errorMessage(value) {
  if (value instanceof Error) {return value.message;}
  return isObject(value) && typeof value["message"] === "string" ? value["message"] : String(value);
}

// Keys that would reach Object.prototype when an external name is used as a record key.
const reservedKeys = new Set(["__proto__", "constructor", "prototype"]);

/** @param {string} key */
function isReservedKey(key) {
  return reservedKeys.has(key);
}

/** @param {unknown} value @returns {value is number} */
function isPositiveNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/** @param {unknown} value @param {number} minimum @returns {value is number} */
function isSafeIntegerAtLeast(value, minimum) {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= minimum;
}

/** Parse a canonical unsigned decimal without rounding large values through Number.
 * @param {unknown} value @param {bigint} maximum @param {string} label */
function parseUnsignedDecimal(value, maximum, label) {
  const text = isSafeIntegerAtLeast(value, 0) ? String(value) : value;
  if (typeof text !== "string" || !/^(0|[1-9]\d*)$/.test(text)) {throw new Error(`Invalid ${label}`);}
  if (text.length > maximum.toString().length) {throw new Error(`${label} is out of range`);}
  const parsed = BigInt(text);
  if (parsed > maximum) {throw new Error(`${label} is out of range`);}
  return parsed;
}

/** @param {unknown} value */
function parseUint64Decimal(value) {
  return parseUnsignedDecimal(value, 0xffffffffffffffffn, "uint64 decimal");
}

module.exports = {
  isObject, errorMessage, isReservedKey, isPositiveNumber, isSafeIntegerAtLeast, parseUnsignedDecimal,
  parseUint64Decimal,
};
