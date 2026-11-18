"use strict";

// Many admission checks compare Object.keys(value).sort().join(",") with a hand-written literal.
// A literal that is not itself sorted can never match a genuine receipt (this broke cold local
// ROCm builds once), so require every such literal to be in sorted order.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const repo = path.join(__dirname, "..");
const comparison = /\.sort\(\)\.join\(\s*"([^"]*)"\s*\)\s*[!=]==\s*"([^"]*)"|"([^"]*)"\s*[!=]==\s*[^;\n]*?\.sort\(\)\.join\(\s*"([^"]*)"\s*\)/g;

test("sorted-key comparison literals are themselves sorted", () => {
  // Walk the tree instead of asking git: test runs also use exported archives without .git.
  /** @type {string[]} */
  const files = [];
  /** @param {string} dir */
  const walk = dir => {
    for (const entry of fs.readdirSync(path.join(repo, dir), {withFileTypes: true})) {
      const name = path.posix.join(dir, entry.name);
      if (entry.isDirectory() && !/^(\.|node_modules$|xmrig$|build$|release)/.test(entry.name)) {walk(name);}
      if (entry.isFile() && entry.name.endsWith(".js")) {files.push(name);}
    }
  };
  walk(".");
  const unsorted = [];
  for (const file of files) {
    for (const match of fs.readFileSync(path.join(repo, file), "utf8").matchAll(comparison)) {
      const separator = match[1] ?? match[4] ?? "", literal = match[2] ?? match[3] ?? "";
      if (!separator || !literal) {continue;}
      const keys = literal.split(separator);
      if (keys.join("\0") !== [...keys].sort().join("\0")) {unsorted.push(file + ": " + literal);}
    }
  }
  assert.deepEqual(unsorted, []);
});
