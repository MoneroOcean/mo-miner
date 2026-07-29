"use strict";

const fs = require("node:fs");

const platformColumns = {
  "intel-linux": "Intel B580 Lin",
  "intel-windows": "Intel B580 Win",
  "nvidia-linux": "RTX 5060 Ti Lin",
  "nvidia-windows": "RTX 5060 Ti Win",
  "amd-linux": "RX 9060 XT Lin",
  "amd-windows": "RX 9060 XT Win",
};

function cells(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

function parseRate(cell) {
  const match = cell.match(/^([0-9]+(?:\.[0-9]+)?)\s+(g\/s|Sol\/s|[KMGT]?H\/s)\b/i);
  if (!match) {return null;}
  const unit = match[2];
  const multiplier = {
    "g/s": 1,
    "sol/s": 1,
    "h/s": 1,
    "kh/s": 1e3,
    "mh/s": 1e6,
    "gh/s": 1e9,
    "th/s": 1e12,
  }[unit.toLowerCase()];
  return {value: Number(match[1]) * multiplier, displayValue: Number(match[1]), unit};
}

function parseReadmePerformance(markdown) {
  const lines = markdown.split(/\r?\n/);
  const headerIndex = lines.findIndex((line) =>
    /^\|\s*Algo\s*\|/.test(line) && line.includes("Intel B580 Lin"));
  if (headerIndex < 0) {throw new Error("README GPU performance table was not found");}

  const header = cells(lines[headerIndex]);
  const indexes = Object.fromEntries(Object.entries(platformColumns).map(([key, title]) => {
    const index = header.indexOf(title);
    if (index < 0) {throw new Error(`README performance column is missing: ${title}`);}
    return [key, index];
  }));
  const supportIndex = header.indexOf("Sup.");
  const rows = [];

  for (const line of lines.slice(headerIndex + 2)) {
    if (!line.startsWith("|")) {break;}
    const row = cells(line);
    const algoMatch = (row[0] || "").match(/^`([^`]+)`$/);
    if (!algoMatch || !/\bmom\b/.test(row[supportIndex] || "")) {continue;}
    const performance = {};
    for (const [platform, index] of Object.entries(indexes)) {
      const parsed = parseRate(row[index] || "");
      if (parsed) {performance[platform] = parsed;}
    }
    rows.push({algo: algoMatch[1], performance});
  }
  return rows;
}

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

module.exports = {parseRate, parseReadmePerformance, platformColumns, readPerformanceFile};
