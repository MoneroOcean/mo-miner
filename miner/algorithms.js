"use strict";

const aliases = new Map([
  ["cuckaroo", "c29"], ["cuckaroo29", "c29"], ["c29xtm", "c29"],
  ["cuckaroo30", "c30"], ["c30ctx", "c30"],
  ["kawpow1", "kawpow"], ["kawpow4", "kawpow"],
  ["xel/2", "xelishashv3"], ["xel/3", "xelishashv3"], ["xel/v3", "xelishashv3"],
]);

/** @param {string | null | undefined} algo */
function normalizeAlgoName(algo) {
  if (algo === null || algo === undefined || algo === "") {return algo;}
  if (typeof algo !== "string") {throw new TypeError("Algorithm name must be a string");}
  const name = algo.toLowerCase();
  return aliases.get(name) || name;
}

module.exports = {normalizeAlgoName};
