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

// Algorithm families that share a pool dialect, header layout, or GPU tuning surface.
const PROGPOW_ALGOS = new Set(["kawpow", "firopow", "evrprogpow", "meowpow"]);
// 140-byte Zcash-family headers with a 32-byte nonce at offset 108.
const ZCASH_HEADER_ALGOS = new Set(["equihash192_7", "zelhash", "zhash"]);
// 80-byte Kaspa-style headers with an 8-byte little-endian nonce at offset 72.
const KASPA_HEADER_ALGOS = new Set(["karlsenhashv2", "hoohash", "walahash"]);

// Stratum dialect of each algorithm whose pools do not use the default `login` dialect. Together
// with "login", its values are the accepted pool `protocol` overrides.
const POOL_PROTOCOL_BY_ALGO = new Map([
  ["karlsenhashv2", "kaspa"], ["walahash", "kaspa"], ["hoohash", "hoosat"],
  ["etchash", "eth"], ["octopus", "conflux"], ["nexapow", "echelon"], ["c30", "cortex"],
  ["autolykos2", "erg"], ["pearlhash", "pearlhash"], ["fishhash", "ironfish"],
  ["beamhash3", "beam"], ["verthash", "verthash"], ["xelishashv3", "xelis"],
]);
for (const algo of PROGPOW_ALGOS) {POOL_PROTOCOL_BY_ALGO.set(algo, "raven");}
for (const algo of ZCASH_HEADER_ALGOS) {POOL_PROTOCOL_BY_ALGO.set(algo, "zelhash");}

module.exports = {
  normalizeAlgoName, PROGPOW_ALGOS, ZCASH_HEADER_ALGOS, KASPA_HEADER_ALGOS, POOL_PROTOCOL_BY_ALGO,
};
