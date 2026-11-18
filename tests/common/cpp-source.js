"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

// Follow private fragments for source-based fixtures, leaving SDK headers and conditionals to C++.
/** @param {string} file @returns {string} */
function readSyclSource(file) {
  return fs.readFileSync(file, "utf8").replace(/^#include "([^"/]+\.inc)"\s*$/gm,
    (_, include) => readSyclSource(path.join(path.dirname(file), include)));
}

// Extract a declaration's complete body; comments and literal braces must not delimit fixtures.
/** @param {string} source @param {string} declaration @returns {string} */
function extractCppBlock(source, declaration) {
  const escaped = declaration.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const matches = [...source.matchAll(new RegExp(`^${escaped}`, "gm"))];
  assert.equal(matches.length, 1, `Missing or ambiguous C++ declaration: ${declaration}`);
  const match = matches[0];
  assert.ok(match);
  const first = match.index;
  const tokens = /\/\*[\s\S]*?\*\/|\/\/[^\n]*|R"([^()\\\s]{0,16})\([\s\S]*?\)\1"|"(?:\\[\s\S]|[^"\\])*"|'(?:\\[\s\S]|[^'\\])*'|[{}]/g;
  tokens.lastIndex = first;
  let depth = 0;
  for (let token = tokens.exec(source); token; token = tokens.exec(source)) {
    if (token[0] === "{") {
      ++depth;
    } else if (token[0] === "}") {
      assert.ok(depth > 0, declaration);
      if (--depth === 0) {return source.slice(first, tokens.lastIndex) + "\n";}
    }
  }
  assert.fail(`Missing closing C++ body: ${declaration}`);
}

module.exports = {readSyclSource, extractCppBlock};
