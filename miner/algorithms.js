"use strict";

function normalizeAlgoName(algo) {
  if (!algo) {return algo;}
  return String(algo).toLowerCase();
}

module.exports = {normalizeAlgoName};
