#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const {isRecord, parseRate} = require("./readme-performance");
/** @typedef {{algo: string, status: string, samples: unknown[]}} ReportResult */
/** @typedef {{label: string, results: ReportResult[]}} Report */
/** @typedef {{value: number, unit: string, normalized: number}} SampleValue */
/** @typedef {Record<string, unknown> & {value_per_second?: unknown, value?: unknown,
 * unit?: unknown, label?: unknown, results?: unknown, algo?: unknown, status?: unknown,
 * samples?: unknown}} JsonObject */

/** @param {number[]} values @returns {number | null} */
const median = (values) => {
  if (!Array.isArray(values) || !values.length || !values.every(Number.isFinite)) {return null;}
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const upper = sorted[middle];
  if (upper === undefined) {return null;}
  if (sorted.length % 2) {return upper;}
  const lower = sorted[middle - 1];
  return lower === undefined ? null : (lower + upper) / 2;
};
/** @param {JsonObject | undefined} result @returns {SampleValue | null} */
const sampleValue = (result) => {
  const samples = result?.samples || [];
  if (!Array.isArray(samples) || !samples.length) {return null;}
  const normalizedSamples = [];
  for (const sample of samples) {
    if (!isRecord(sample)) {return null;}
    const reportedValue = sample["value_per_second"];
    let normalized = typeof reportedValue === "number" ? reportedValue : NaN;
    if (!Number.isFinite(normalized) || normalized <= 0) {
      const value = sample["value"];
      const unit = sample["unit"];
      if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 ||
          typeof unit !== "string") {return null;}
      normalized = parseRate(`${value} ${unit}`)?.value ?? NaN;
    }
    if (!Number.isFinite(normalized) || normalized <= 0) {return null;}
    normalizedSamples.push(normalized);
  }
  const normalized = median(normalizedSamples);
  if (normalized === null) {return null;}
  // Render in the last steady sample's prefix while comparing in prefix-independent units. This
  // also reads older reports that predate value_per_second.
  const lastSample = samples.at(-1);
  if (!isRecord(lastSample) || typeof lastSample["unit"] !== "string") {return null;}
  const unit = lastSample["unit"];
  const unitValue = parseRate(`1 ${unit}`)?.value;
  return unitValue ? {value: normalized / unitValue, unit, normalized} : null;
};

/** @param {unknown} report @returns {Report} */
function validateReport(report) {
  if (!isRecord(report)) {throw new Error("report must be a non-array object");}
  const label = report["label"];
  if (typeof label !== "string" || !label.trim()) {
    throw new Error("report label must be a nonempty string");
  }
  const rawResults = report["results"];
  if (!Array.isArray(rawResults)) {
    throw new Error("report results must be an array");
  }
  const algos = new Set();
  /** @type {ReportResult[]} */
  const results = [];
  for (const result of rawResults) {
    if (!isRecord(result)) {throw new Error("every result must be a non-array object");}
    const algo = result["algo"];
    if (typeof algo !== "string" || !algo.trim()) {
      throw new Error("every result algo must be a nonempty string");
    }
    if (algos.has(algo)) {
      throw new Error(`duplicate result algo: ${algo}`);
    }
    const status = result["status"];
    const samples = result["samples"];
    if (typeof status !== "string" || !status.trim()) {
      throw new Error(`result ${algo} status must be a nonempty string`);
    }
    if (!Array.isArray(samples) ||
        (status === "ok" && !samples.length) ||
        (samples.length && !sampleValue({algo, status, samples}))) {
      throw new Error(`result ${algo} samples are invalid`);
    }
    algos.add(algo);
    results.push({algo, status, samples});
  }
  return {...report, label, results};
}

/** @param {unknown[]} reports @returns {string} */
function renderComparison(reports) {
  if (!Array.isArray(reports) || reports.length < 2) {
    throw new Error("at least two reports are required");
  }
  const validReports = reports.map(validateReport);
  const byAlgo = validReports.map((report) =>
    new Map(report.results.map((result) => [result.algo, result])));
  const algos = [...new Set(validReports.flatMap((report) =>
    report.results.map((result) => result.algo)))].sort();
  const lines = [
    `| Algorithm | ${validReports.map((report) => report.label).join(" | ")} |`,
    `| --- | ${validReports.map(() => "---:").join(" | ")} |`,
  ];
  for (const algo of algos) {
    const base = sampleValue(byAlgo[0]?.get(algo))?.normalized;
    const cells = validReports.map((_, index) => {
      const result = byAlgo[index]?.get(algo);
      const sample = sampleValue(result);
      if (!sample) {return typeof result?.status === "string" ? result.status : "-";}
      const delta = index && base ? ` (${((sample.normalized / base - 1) * 100).toFixed(1)}%)` : "";
      return `${Number(sample.value.toFixed(2))} ${sample.unit}${delta}`;
    });
    lines.push(`| ${algo} | ${cells.join(" | ")} |`);
  }
  return `${lines.join("\n")}\n`;
}

function main() {
  const files = process.argv.slice(2);
  if (files.length < 2) {
    console.error("usage: compare-gpu-benchmarks.js baseline.json candidate.json [...]");
    process.exitCode = 2;
    return;
  }
  try {
    const reports = files.map((file) => JSON.parse(fs.readFileSync(file, "utf8")));
    process.stdout.write(renderComparison(reports));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`compare-gpu-benchmarks: ${message}`);
    process.exitCode = 1;
  }
}

if (require.main === module) {main();}

module.exports = {median, sampleValue, validateReport, renderComparison};
