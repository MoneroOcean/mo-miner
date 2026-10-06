"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

/** @typedef {{kawpow: string, etchash: string, state: string, program: string, internal: string}} SourceVariant */

const repoRoot = path.join(__dirname, "..");
const nativeFixture = path.join(__dirname, "native", "portable_dag_test.cpp");

/**
 * @param {string} message
 * @returns {never}
 */
function fail(message) {
  throw new Error(`portable DAG fixture: ${message}`);
}

/**
 * @param {string} source
 * @param {string} signature
 * @returns {string}
 */
function extractFunction(source, signature) {
  const start = source.indexOf(signature);
  if (start < 0) {fail(`missing production function ${signature}`);}
  const brace = source.indexOf("{", start);
  if (brace < 0) {fail(`missing function body ${signature}`);}
  let depth = 0;
  for (let index = brace; index < source.length; index++) {
    if (source[index] === "{") {depth++;}
    if (source[index] === "}") {
      depth--;
      if (depth === 0) {return source.slice(start, index + 1);}
    }
  }
  fail(`unterminated production function ${signature}`);
}

/**
 * @param {string} source
 * @param {RegExp} pattern
 * @param {string} label
 * @returns {string}
 */
function extractLine(source, pattern, label) {
  const match = source.match(pattern);
  if (!match) {fail(`missing production line ${label}`);}
  return match[0].replace(/\r$/, "");
}

/**
 * @param {string} source
 * @returns {string}
 */
function extractPredicate(source) {
  const start = source.indexOf("if (epoch == new_epoch");
  if (start < 0) {fail("missing Etchash cache reuse predicate");}
  const open = source.indexOf("(", start);
  let depth = 0;
  for (let index = open; index < source.length; index++) {
    if (source[index] === "(") {depth++;}
    if (source[index] === ")") {
      depth--;
      if (depth === 0) {
        return source.slice(open + 1, index).replace(/\s+/g, " ").trim();
      }
    }
  }
  fail("unterminated Etchash cache reuse predicate");
}

/**
 * @param {SourceVariant} variant
 * @returns {string}
 */
function generatedHeader(variant) {
  const kawpow = variant.kawpow;
  const etchash = variant.etchash;
  const state = variant.state;
  const program = variant.program;
  const internal = variant.internal;
  const kawpowCacheWords = extractLine(
    program, /^constexpr uint32_t KAWPOW_CACHE_WORDS = .*;\r?$/m, "KAWPOW_CACHE_WORDS");
  const kawpowLanes = extractLine(
    program, /^constexpr uint32_t KAWPOW_LANES = .*;\r?$/m, "KAWPOW_LANES");
  const kawpowDagLoads = extractLine(
    program, /^constexpr uint32_t KAWPOW_DAG_LOADS = .*;\r?$/m, "KAWPOW_DAG_LOADS");
  const parser = extractFunction(internal, "inline bool mom_parse_env_ulong");
  const fastModStart = internal.indexOf("struct FastModData");
  const fastModEnd = internal.indexOf("#if defined(_WIN32)", fastModStart);
  if (fastModStart < 0 || fastModEnd < 0) {fail("missing production fast modulus block");}
  const fastMod = internal.slice(fastModStart, fastModEnd);
  const keyPredicate = extractPredicate(state);

  return [
    "#pragma once",
    "",
    "namespace portable_fixture {",
    kawpowCacheWords,
    kawpowLanes,
    kawpowDagLoads,
    extractLine(kawpow, /^constexpr uint64_t KAWPOW_PORTABLE_TEST_CACHE_BYTES = .*;\r?$/m,
      "KawPow cache geometry"),
    extractLine(kawpow, /^constexpr uint64_t KAWPOW_PORTABLE_TEST_DAG_BYTES = .*;\r?$/m,
      "KawPow DAG geometry"),
    extractLine(etchash, /^constexpr uint64_t ETCHASH_PORTABLE_TEST_CACHE_BYTES = .*;\r?$/m,
      "Etchash cache geometry"),
    extractLine(etchash, /^constexpr uint64_t ETCHASH_PORTABLE_TEST_DAG_BYTES = .*;\r?$/m,
      "Etchash DAG geometry"),
    parser,
    fastMod,
    extractFunction(kawpow, "static bool kawpow_portable_test"),
    extractFunction(etchash, "static bool etchash_portable_test"),
    "inline bool etchash_cache_key_matches(",
    "    const uint32_t epoch, const uint32_t new_epoch, const uint32_t seed_epoch,",
    "    const uint32_t new_seed_epoch, const uint64_t light_cache_words,",
    "    const uint64_t new_light_cache_words, const uint64_t dag_words,",
    "    const uint64_t new_dag_words, const void* const light_cache, const void* const dag) {",
    "  if (" + keyPredicate.replace(/\bepoch\b/g, "epoch") + ")",
    "    return true;",
    "  return false;",
    "}",
    "} // namespace portable_fixture",
    "",
  ].join("\n");
}

/**
 * @param {boolean} crlf
 * @returns {SourceVariant}
 */
function readSources(crlf) {
  /**
   * @param {string} relative
   * @returns {string}
   */
  const read = (relative) => {
    const source = fs.readFileSync(path.join(repoRoot, relative), "utf8");
    return crlf ? source.replace(/\n/g, "\r\n") : source;
  };
  return {
    kawpow: read("sycl/kawpow/entry.inc"),
    etchash: read("sycl/etchash/entry.inc"),
    state: read("sycl/etchash/state.inc"),
    program: read("sycl/kawpow/program.inc"),
    internal: read("sycl/lib-internal.h"),
  };
}

/**
 * @param {SourceVariant} variant
 * @param {string} compiler
 * @param {string} root
 * @param {string} label
 * @returns {void}
 */
function runVariant(variant, compiler, root, label) {
  const includeDir = path.join(root, label);
  fs.mkdirSync(includeDir);
  fs.writeFileSync(path.join(includeDir, "portable_dag_generated.inc"), generatedHeader(variant));
  const executable = path.join(includeDir, "portable_dag_test");
  const compile = spawnSync(compiler, [
    "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", nativeFixture,
    "-I", includeDir, "-o", executable,
  ], {cwd: repoRoot, encoding: "utf8", timeout: 10000});
  if (compile.error) {throw compile.error;}
  if (compile.status !== 0) {
    fail(`${label} compile failed\n${compile.stdout}\n${compile.stderr}`);
  }
  const result = spawnSync(executable, [], {cwd: repoRoot, encoding: "utf8", timeout: 5000});
  if (result.error) {throw result.error;}
  if (result.status !== 0) {
    fail(`${label} runtime failed\n${result.stdout}\n${result.stderr}`);
  }
  assert.equal(result.stdout.trim(), "portable DAG fixture PASS", `${label} output`);
}

test("portable DAG modes preserve bounded CPU fixture guards", {
  skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
}, () => {
  const compiler = process.env["CXX"] || "c++";
  const root = fs.mkdtempSync(path.join(repoRoot, ".portable-dag-"));
  try {
    runVariant(readSources(false), compiler, root, "lf");
    runVariant(readSources(true), compiler, root, "crlf");
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});
