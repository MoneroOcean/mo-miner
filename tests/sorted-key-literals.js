"use strict";

// Many admission checks compare Object.keys(value).sort().join(",") with a hand-written literal.
// A literal that is not itself sorted can never match a genuine receipt (this broke cold local
// ROCm builds once), so require every such literal to be in sorted order.
const assert = require("node:assert/strict");
const {execFileSync} = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const repo = path.join(__dirname, "..");
const comparison = /\.sort\(\)\.join\(\s*"([^"]*)"\s*\)\s*[!=]==\s*"([^"]*)"|"([^"]*)"\s*[!=]==\s*[^;\n]*?\.sort\(\)\.join\(\s*"([^"]*)"\s*\)/g;

test("sorted-key comparison literals are themselves sorted", () => {
  const files = execFileSync("git", ["ls-files", "--", "*.js", ":!xmrig", ":!node_modules"], {cwd: repo, encoding: "utf8"})
    .split("\n").filter(Boolean);
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
