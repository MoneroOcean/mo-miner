"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

// Braces in comments, string literals and raw strings must not delimit an extracted fixture.
const literalOrBrace = /\/\*[\s\S]*?\*\/|\/\/[^\n]*|R"([^()\\\s]{0,16})\([\s\S]*?\)\1"|"(?:\\[\s\S]|[^"\\])*"|'(?:\\[\s\S]|[^'\\])*'|[{}]/g;

// Follow private fragments for source-based fixtures, leaving SDK headers and conditionals to C++.
/** @param {string} file @returns {string} */
function readSyclSource(file) {
  return fs.readFileSync(file, "utf8").replace(/^#include "([^"/]+\.inc)"\s*$/gm,
    (_, include) => readSyclSource(path.join(path.dirname(file), include)));
}

// Match a declaration by its tokens so whitespace and line wrapping in the source do not matter.
/** @param {string} declaration @returns {string} */
function declarationPattern(declaration) {
  const tokens = declaration.trim().match(/[A-Za-z0-9_]+|\S/g);
  assert.ok(tokens, "Empty C++ declaration anchor");
  return tokens.map((token, index) => {
    const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    if (index === 0) {return escaped;}
    const previous = tokens[index - 1] ?? "";
    return (/\w$/.test(previous) && /^\w/.test(token) ? "\\s+" : "\\s*") + escaped;
  }).join("");
}

// Raw strings stay searchable because CUDA/HIP device sources are embedded in them.
/** @param {string} source @returns {Array<[number, number]>} */
function commentRanges(source) {
  /** @type {Array<[number, number]>} */
  const ranges = [];
  for (const token of source.matchAll(literalOrBrace)) {
    if (token[0].startsWith("/")) {ranges.push([token.index, token.index + token[0].length]);}
  }
  return ranges;
}

// A function template is only compilable with its header, so include one that precedes it,
// possibly separated by a purpose comment.
/** @param {string} source @param {number} start @returns {number} */
function templateStart(source, start) {
  const header = /template\s*<[^{};]*>(?:\s|\/\/[^\n]*|\/\*[\s\S]*?\*\/)*$/
    .exec(source.slice(Math.max(0, start - 1024), start));
  if (!header) {return start;}
  const begin = start - header[0].length;
  return source.lastIndexOf("\n", begin - 1) + 1;
}

/**
 * Extract one complete declaration body: the anchor (with exactly its own leading indentation)
 * must start one line outside comments, and the result runs from that line, or its
 * template header, to the matching brace. Whitespace between anchor tokens is not significant.
 * @param {string} source @param {string} declaration @returns {string}
 */
function extractCppBlock(source, declaration) {
  const comments = commentRanges(source);
  // Indentation reflects nesting, so the anchor's own leading indentation must match exactly.
  const indentation = /^[ \t]*/.exec(declaration)?.[0] ?? "";
  const anchor = new RegExp(`^${indentation}${declarationPattern(declaration)}`, "gm");
  const matches = [...source.matchAll(anchor)]
    .filter(({index}) => !comments.some(([begin, end]) => index >= begin && index < end));
  assert.equal(matches.length, 1, `Missing or ambiguous C++ declaration: ${declaration}`);
  const match = matches[0];
  assert.ok(match);
  const first = templateStart(source, match.index);
  const tokens = new RegExp(literalOrBrace.source, "g");
  tokens.lastIndex = match.index;
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

// Shell fixtures: `node tests/common/cpp-source.js FILE DECLARATION...` prints each block in order.
if (require.main === module) {
  const [file, ...declarations] = process.argv.slice(2);
  if (!file || declarations.length === 0) {
    process.stderr.write("usage: cpp-source.js FILE DECLARATION...\n");
    process.exit(2);
  }
  const source = fs.readFileSync(file, "utf8");
  process.stdout.write(declarations.map(declaration => extractCppBlock(source, declaration)).join(""));
}

module.exports = {readSyclSource, extractCppBlock};
