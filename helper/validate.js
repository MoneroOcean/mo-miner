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

/** @param {unknown} value @returns {value is number} */
function isPositiveNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

module.exports = {isObject, errorMessage, isPositiveNumber};
