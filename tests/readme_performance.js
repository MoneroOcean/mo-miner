"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const {parseRate, parseReadmePerformance, platformColumns} =
  require("../scripts/readme-performance");
const {gpuAlgos} = require("./common/gpu_test_modes");

test("README contains a measured gate for every supported GPU algorithm and platform", () => {
  const markdown = fs.readFileSync(path.join(__dirname, "..", "README.md"), "utf8");
  const rows = parseReadmePerformance(markdown);
  assert.deepEqual(rows.map((row) => row.algo).sort(), [...gpuAlgos].sort());
  assert.equal(new Set(rows.map((row) => row.algo)).size, rows.length);
  for (const row of rows) {
    assert.deepEqual(Object.keys(row.performance).sort(), Object.keys(platformColumns).sort(),
      `${row.algo} must have all six release performance references`);
    for (const rate of Object.values(row.performance)) {assert.ok(rate.value > 0);}
  }
});

test("README rate parser normalizes mining units", () => {
  assert.equal(parseRate("2.82 g/s (%?)").value, 2.82);
  assert.equal(parseRate("14.74 Sol/s (104%)").value, 14.74);
  assert.equal(parseRate("20.97 MH/s (110%)").value, 20.97e6);
  assert.equal(parseRate("51.97 TH/s (149%)").value, 51.97e12);
  assert.equal(parseRate("-"), null);
});
