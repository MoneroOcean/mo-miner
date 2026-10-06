"use strict";

const aliases = new Map([
  ["cuckaroo", "c29"], ["cuckaroo29", "c29"], ["c29xtm", "c29"],
  ["kawpow1", "kawpow"], ["kawpow4", "kawpow"],
]);

/** @param {string | null | undefined} algo */
function normalizeAlgoName(algo) {
  if (algo === null || algo === undefined || algo === "") {return algo;}
  if (typeof algo !== "string") {throw new TypeError("Algorithm name must be a string");}
  const name = algo.toLowerCase();
  return aliases.get(name) || name;
}

module.exports = {normalizeAlgoName};
