#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const reports = process.argv.slice(2).map(file => JSON.parse(fs.readFileSync(file, "utf8")));
if (reports.length < 2) {
  console.error("usage: compare-gpu-benchmarks.js baseline.json candidate.json [...]");
  process.exit(2);
}
const units = { "H/s": 1, "KH/s": 1e3, "MH/s": 1e6, "GH/s": 1e9, "TH/s": 1e12, "PH/s": 1e15 };
const median = values => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const sampleValue = result => {
  const samples = result?.samples || [];
  if (!samples.length || !samples.every(sample => units[sample.unit])) {return null;}
  const normalized = median(samples.map(sample =>
    Number.isFinite(sample.value_per_second) ? sample.value_per_second : sample.value * units[sample.unit]));
  // Render in the last steady sample's prefix while comparing in prefix-independent units. This
  // also reads older reports that predate value_per_second.
  const unit = samples.at(-1).unit;
  return {value: normalized / units[unit], unit, normalized};
};
const value = result => sampleValue(result)?.normalized ?? null;
const byAlgo = reports.map(report => new Map(report.results.map(result => [result.algo, result])));
const algos = [...new Set(reports.flatMap(report => report.results.map(result => result.algo)))].sort();
console.log(`| Algorithm | ${reports.map(report => report.label).join(" | ")} |`);
console.log(`| --- | ${reports.map(() => "---:").join(" | ")} |`);
for (const algo of algos) {
  const base = value(byAlgo[0].get(algo));
  const cells = reports.map((report, index) => {
    const result = byAlgo[index].get(algo);
    const sample = sampleValue(result);
    if (!sample) {return result?.status || "-";}
    const delta = index && base ? ` (${((value(result) / base - 1) * 100).toFixed(1)}%)` : "";
    return `${Number(sample.value.toFixed(2))} ${sample.unit}${delta}`;
  });
  console.log(`| ${algo} | ${cells.join(" | ")} |`);
}
