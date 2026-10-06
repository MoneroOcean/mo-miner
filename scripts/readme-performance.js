"use strict";

const fs = require("node:fs");

/** @typedef {{value: number, displayValue: number, unit: string}} Rate */
/** @typedef {{algo: string, performance: Record<string, Rate>}} PerformanceRow */

/** @type {Record<string, string>} */
const gpuVendors = {"B580": "intel", "5060 Ti": "nvidia", "9060 XT": "amd"};
const maxCapturedOutput = 1024 * 1024;
/** @type {Record<string, true>} */
const platformColumns = Object.fromEntries(Object.values(gpuVendors).flatMap((vendor) =>
  ["linux", "windows"].map((os) => [`${vendor}-${os}`, true])));
/** @type {{os: string, column: "Linux" | "Windows"}[]} */
const performanceColumns = [
  {os: "linux", column: "Linux"}, {os: "windows", column: "Windows"},
];

/** @param {unknown} value @returns {value is Record<string, unknown>} */
function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** @param {string} line @returns {string[]} */
function cells(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

/** @param {string} cell @returns {Rate | null} */
function parseRate(cell) {
  const match = cell.match(/^([0-9]+(?:\.[0-9]+)?)\s+(g\/s|I\/s|Sol\/s|[KMGTP]?H\/s)\b/i);
  if (!match) {return null;}
  const unit = match[2];
  if (!unit || !match[1]) {return null;}
  const multiplier = {
    "g/s": 1,
    "i/s": 1,
    "sol/s": 1,
    "h/s": 1,
    "kh/s": 1e3,
    "mh/s": 1e6,
    "gh/s": 1e9,
    "th/s": 1e12,
    "ph/s": 1e15,
  }[unit.toLowerCase()];
  if (multiplier === undefined) {return null;}
  const displayValue = Number(match[1]);
  const value = displayValue * multiplier;
  return Number.isFinite(value) ? {value, displayValue, unit} : null;
}

/** @param {string} text @param {string} algo @returns {Rate[]} */
function reportedRates(text, algo) {
  const escaped = algo.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return [...text.matchAll(
    new RegExp(`Algo ${escaped} \\([^)]*\\) hashrate: ([0-9]+(?:\\.[0-9]+)?)\\s+` +
      "(g/s|I/s|Sol/s|[KMGTP]?H/s)", "gi")
  )].map((match) => parseRate(`${match[1]} ${match[2]}`))
    .filter((rate) => rate !== null);
}

/** @param {string} current @param {Buffer | string} chunk @returns {string} */
function appendOutputTail(current, chunk) {
  const next = current + chunk.toString();
  return next.length > maxCapturedOutput ? next.slice(-maxCapturedOutput) : next;
}

/** @param {string} markdown @returns {PerformanceRow[]} */
function parseReadmePerformance(markdown) {
  const lines = markdown.split(/\r?\n/);
  const headerIndex = lines.findIndex((line) =>
    /^\|\s*Algo\s*\/\s*coin\s*\/\s*pool\s*\|/.test(line) && line.includes("| GPU"));
  if (headerIndex < 0) {throw new Error("README GPU performance table was not found");}

  const headerLine = lines[headerIndex];
  if (headerLine === undefined) {throw new Error("README GPU performance header is missing");}
  const header = cells(headerLine);
  const indexes = {
    GPU: header.indexOf("GPU"), Linux: header.indexOf("Linux"), Windows: header.indexOf("Windows"),
  };
  if (Object.values(indexes).includes(-1)) {throw new Error("README performance columns are missing");}
  /** @type {PerformanceRow[]} */
  const rows = [];
  /** @type {PerformanceRow | undefined} */
  let parsedRow;

  for (const line of lines.slice(headerIndex + 2)) {
    if (!line.startsWith("|")) {break;}
    const row = cells(line);
    const algoMatch = (row[0] || "").match(/^`([^`]+)`/);
    if (algoMatch) {
      const algo = algoMatch[1];
      if (!algo || !/^[a-z0-9][a-z0-9/_-]*$/i.test(algo)) {
        throw new Error(`README contains an invalid algorithm name: ${algo || ""}`);
      }
      const nextRow = {algo, performance: {}};
      parsedRow = nextRow;
      rows.push(nextRow);
    }
    const vendorKey = row[indexes.GPU];
    const vendor = vendorKey === undefined ? undefined : gpuVendors[vendorKey];
    if (!parsedRow || !vendor) {continue;}
    for (const {os, column} of performanceColumns) {
      const rate = parseRate(row[indexes[column]] || "");
      if (rate) {parsedRow.performance[`${vendor}-${os}`] = rate;}
    }
  }

  return rows;
}

/** @param {string} file @returns {PerformanceRow[]} */
function readPerformanceFile(file) {
  return parseReadmePerformance(fs.readFileSync(file, "utf8"));
}

if (require.main === module) {
  const [file = "README.md", platform] = process.argv.slice(2);
  const rows = readPerformanceFile(file);
  const selected = platform
    ? rows.filter((row) => row.performance[platform])
      .map((row) => ({algo: row.algo, ...row.performance[platform]}))
    : rows;
  process.stdout.write(`${JSON.stringify(selected, null, 2)}\n`);
}

module.exports = {
  appendOutputTail, isRecord, maxCapturedOutput, parseRate, parseReadmePerformance, platformColumns,
  readPerformanceFile, reportedRates,
};
