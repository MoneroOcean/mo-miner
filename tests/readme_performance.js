"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const {parseRate, parseReadmePerformance, platformColumns} =
  require("../scripts/readme-performance");
const {gpuAlgos, requestedAlgos} = require("./common/gpu_test_modes");
const opts = require("../opts");
const createCli = require("../miner/cli");

const alignedMarkdownFiles = [
  "README.md", "DEVELOPMENT.md", "GPU-CONFIG.md",
];

/** @param {string} cell */
function trimCell(cell) {
  return cell.trim();
}

/** @param {string} line */
function splitCells(line) {
  return line.split("|").slice(1, -1).map(trimCell);
}

/** @param {string} text */
function requiredRate(text) {
  const rate = parseRate(text);
  assert.ok(rate, `Expected a numeric rate in ${text}`);
  return rate;
}

/** @param {string} line */
function pipeOffsets(line) {
  const result = [];
  for (let index = 0; index < line.length; ++index) {
    if (line[index] === "|") {result.push(index);}
  }
  return result;
}

/** @param {string} markdown */
function performanceTable(markdown) {
  const start = markdown.indexOf("| Algo / coin / pool");
  assert.notEqual(start, -1, "README performance table must exist");
  return markdown.slice(start, markdown.indexOf("\n\n", start));
}

/** @param {string[]} cells */
function assertRateEvidence(cells) {
  const evidence = cells[4] || "";
  const borrowed = evidence.includes("B580");
  if (borrowed) {
    assert.equal(cells[1], "A770", "borrowed B580 references belong to A770");
    assert.equal(cells[3], "-", "borrowed B580 references are Linux-only");
    assert.match(evidence, /\(B580\)\*$/);
    assert.match(cells[2] || "",
      /^[0-9]+(?:\.[0-9]+)?\s+(g\/s|I\/s|Sol\/s|[KMGTP]?H\/s)(?: \(\d+%\*\))?$/);
  }
  for (const [index, cell] of cells.slice(2, 4).entries()) {
    if (!parseRate(cell)) {continue;}
    if (index === 0 && cells[1] === "A770" && !cell.includes("(")) {
      assert.equal(cells[3], "-", "A770 fallback references are Linux-only");
      assert.match(evidence, /\*$/);
      continue;
    }
    if (cell.endsWith("(TBD)")) {
      assert.match(evidence, /\(rechecking\)$/);
    } else {
      assert.match(cell, /\(\d+%\*?\)$/);
      if (cells[1] === "A770" && evidence.includes("*")) {
        assert.match(cell, /\(\d+%\*\)$/);
      }
    }
  }
}

test("documentation table columns stay source-aligned", () => {
  for (const file of alignedMarkdownFiles) {
    const lines = fs.readFileSync(path.join(__dirname, "..", file), "utf8").split(/\r?\n/);
    for (let start = 0; start < lines.length;) {
      const header = lines[start];
      if (!header || !header.startsWith("|")) {
        ++start;
        continue;
      }
      let end = start + 1;
      while (end < lines.length && lines[end]?.startsWith("|")) {++end;}
      const expected = pipeOffsets(header);
      for (let row = start + 1; row < end; ++row) {
        const rowText = lines[row];
        if (rowText === undefined) {throw new Error(`${file}:${row + 1} table row is missing`);}
        assert.deepEqual(pipeOffsets(rowText), expected,
          `${file}:${row + 1} must align with table header at line ${start + 1}`);
      }
      start = end;
    }
  }
});

test("README contains measured references for every supported GPU algorithm", () => {
  const markdown = fs.readFileSync(path.join(__dirname, "..", "README.md"), "utf8");
  const rows = parseReadmePerformance(markdown);
  // Verthash's 1.20-GiB external dataset is intentionally absent from ordinary deploy tests.
  assert.deepEqual(rows.map((row) => row.algo).sort(), [...new Set([...gpuAlgos, "verthash"])].sort());
  assert.equal(new Set(rows.map((row) => row.algo)).size, rows.length);
  for (const row of rows) {
    const platforms = Object.keys(row.performance);
    assert.ok(platforms.length, `${row.algo} must have a release performance reference`);
    assert.ok(platforms.every((platform) => platformColumns[platform]),
      `${row.algo} has an unknown performance platform`);
    for (const rate of Object.values(row.performance)) {assert.ok(rate.value > 0);}
  }
});

test("README groups A770 immediately after B580", () => {
  const markdown = fs.readFileSync(path.join(__dirname, "..", "README.md"), "utf8");
  const rows = performanceTable(markdown).split("\n").slice(2).map(splitCells);
  for (let index = 0; index < rows.length; index += 4) {
    assert.deepEqual(rows.slice(index, index + 4).map((cells) => cells[1]),
      ["B580", "A770", "5060 Ti", "9060 XT"]);
  }
});

test("numeric README rates retain comparison evidence", () => {
  const markdown = fs.readFileSync(path.join(__dirname, "..", "README.md"), "utf8");
  const table = performanceTable(markdown);
  assert.doesNotMatch(table, /MoM\/search|\/local|\(local reference\)|Reference unavailable/);
  for (const line of table.split("\n").slice(2)) {
    assertRateEvidence(splitCells(line));
  }
});

test("borrowed B580 references retain cross-GPU markers", () => {
  assertRateEvidence(["", "A770", "32.23 TH/s (86%*)", "-", "BZMiner (B580)*"]);
  assertRateEvidence(["", "A770", "2.41 KH/s", "-", "SRBMiner (B580)*"]);
  for (const cells of [
    ["", "A770", "32.23 TH/s (86%)", "-", "BZMiner (B580)*"],
    ["", "A770", "32.23 TH/s (86%*)", "-", "BZMiner (B580)"],
    ["", "A770", "32.23 TH/s (86%*)", "-", "BZMiner (B580*)"],
    ["", "B580", "32.23 TH/s (86%*)", "-", "BZMiner (B580)*"],
    ["", "A770", "32.23 TH/s (86%*)", "31.00 TH/s (83%*)", "BZMiner (B580)*"],
  ]) {
    assert.throws(() => assertRateEvidence(cells));
  }
});

test("approximate A770 measured references mark their percentages", () => {
  for (const evidence of ["WildRig*", "SRBMiner*", "lolMiner*"]) {
    assertRateEvidence(["", "A770", "13.96 MH/s (148%*)", "-", evidence]);
    assert.throws(() =>
      assertRateEvidence(["", "A770", "13.96 MH/s (148%)", "-", evidence]));
  }
});

test("plain README rates require a marked A770 fallback reference", () => {
  assertRateEvidence(["", "A770", "1.41 g/s", "-", "MoM 12-GiB-limited peer*"]);
  for (const gpu of ["B580", "5060 Ti", "9060 XT"]) {
    assert.throws(() => assertRateEvidence(["", gpu, "16.39 MH/s", "-", "miner (B580)*"]));
    assert.throws(() => assertRateEvidence(["", gpu, "16.39 MH/s", "-", "miner"]));
  }
  /** @type {[string, string, string][]} */
  const invalid = [
    ["16.39 MH/s", "17.00 MH/s", "miner (B580)*"],
    ["16.39 MH/s (50%)", "-", "miner (B580)*"],
    ["16.39 nonsense", "-", "miner (B580)*"],
    ["16.39 MH/s", "-", "miner (B580)* extra"],
  ];
  for (const [linux, windows, evidence] of invalid) {
    assert.throws(() => assertRateEvidence(["", "A770", linux, windows, evidence]));
  }
});

test("A770 rows cannot overwrite any of the six existing release platforms", () => {
  const header = [
    "| Algo / coin / pool | GPU | Linux | Windows | Reference / evidence |",
    "| --- | --- | --- | --- | --- |",
  ];
  const original = [
    "| `c29` / TARI / MO | B580 | 4.55 g/s (100%*) | 4.75 g/s (100%*) | best found* |",
    "| | 5060 Ti | 6.32 g/s (85%) | 6.28 g/s (85%) | lolMiner |",
    "| | 9060 XT | 6.15 g/s (106%) | 5.95 g/s (103%) | lolMiner |",
  ];
  const baseline = parseReadmePerformance([...header, ...original, ""].join("\n"));
  assert.equal(Object.keys(baseline[0]?.performance || {}).length, 6);
  const first = original[0];
  assert.ok(first);
  for (let index = 0; index <= original.length; ++index) {
    const rows = [...original];
    const algo = index === 0 ? "`c29` / TARI / MO" : "";
    if (index === 0) {rows[0] = first.replace("`c29` / TARI / MO", "");}
    rows.splice(index, 0, `| ${algo} | A770 | 99.99 g/s | - | Reference unavailable |`);
    assert.deepEqual(parseReadmePerformance([...header, ...rows, ""].join("\n")), baseline);
  }
  const markdown = fs.readFileSync(path.join(__dirname, "..", "README.md"), "utf8");
  const withoutA770 = markdown.split("\n").filter((line) =>
    !line.startsWith("|") || splitCells(line)[1] !== "A770").join("\n");
  assert.deepEqual(parseReadmePerformance(markdown), parseReadmePerformance(withoutA770));
});

test("cross-OS same-GPU references mark only the other-OS cell", () => {
  const markdown = fs.readFileSync(path.join(__dirname, "..", "README.md"), "utf8");
  const rows = performanceTable(markdown).split("\n").slice(2)
    .map(/** @param {string} line */ (line) => splitCells(line));
  const qualified = rows.filter((cells) => {
    const evidence = cells[4];
    return evidence !== undefined && /\((Linux|Windows)\)$/.test(evidence);
  });
  assert.ok(qualified.length, "README must retain controlled cross-OS references");
  for (const cells of qualified) {
    const evidence = cells[4];
    const linux = cells[2];
    const windows = cells[3];
    if (evidence === undefined || linux === undefined || windows === undefined) {
      throw new Error("README cross-OS row is missing a required cell");
    }
    const referenceOs = evidence.endsWith("(Linux)") ? "linux" : "windows";
    assert.match(linux, referenceOs === "linux" ? /\(\d+%\)$/ : /\(\d+%\*\)$/);
    assert.match(windows, referenceOs === "windows" ? /\(\d+%\)$/ : /\(\d+%\*\)$/);
  }
});

test("README lists every supported GPU algorithm in one memory class", () => {
  const markdown = fs.readFileSync(path.join(__dirname, "..", "README.md"), "utf8");
  const start = markdown.indexOf("## GPU memory requirements");
  const end = markdown.indexOf("\n# Install", start);
  assert.ok(start >= 0 && end > start, "GPU memory requirements section must exist");
  const listItems = [...markdown.slice(start, end).matchAll(/^- [^\n]*(?:\n {2}[^\n]*)*/gm)]
    .map((match) => match[0]);
  const listed = listItems.flatMap((item) =>
    [...item.matchAll(/`([^`]+)`/g)].map((match) => match[1]));
  const expected = [...new Set([...gpuAlgos, "verthash"])];
  assert.equal(listed.length, expected.length, "each GPU algorithm must appear exactly once");
  assert.deepEqual([...listed].sort(), [...expected].sort());
});

test("README rate parser normalizes mining units", () => {
  assert.equal(requiredRate("2.82 g/s").value, 2.82);
  assert.equal(requiredRate("36.80 I/s (80%)").value, 36.8);
  assert.equal(requiredRate("14.74 Sol/s (104%)").value, 14.74);
  assert.equal(requiredRate("20.97 MH/s (110%)").value, 20.97e6);
  assert.equal(requiredRate("51.97 TH/s (149%)").value, 51.97e12);
  assert.equal(requiredRate("1.25 PH/s").value, 1.25e15);
  assert.equal(parseRate("-"), null);
  assert.equal(parseRate("unsupported*"), null);
});

test("README performance parser accepts combined algo and coin cells", () => {
  const markdown = [
    "| Algo / coin / pool | GPU | Linux | Windows | Reference / evidence |",
    "| --- | --- | --- | --- | --- |",
    "| `c29` / TARI / MO | B580 | 4.55 g/s (100%*) | 4.75 g/s (100%*) | best found* |",
    "| | 5060 Ti | 6.32 g/s (85%) | 6.28 g/s (85%) | lolMiner |",
    "| | 9060 XT | 6.15 g/s (106%) | 5.95 g/s (103%) | lolMiner |",
    "",
  ].join("\n");
  const [row] = parseReadmePerformance(markdown);
  assert.ok(row);
  assert.equal(row.algo, "c29");
  assert.deepEqual(Object.keys(row.performance).sort(),
    ["amd-linux", "amd-windows", "intel-linux", "intel-windows", "nvidia-linux", "nvidia-windows"]);
  assert.throws(() => parseReadmePerformance(markdown.replace("`c29`", "`c29 & whoami`")),
    /invalid algorithm name/);
});

test("README benchmark examples parse without shell-specific JSON quoting", () => {
  const markdown = fs.readFileSync(path.join(__dirname, "..", "README.md"), "utf8");
  const examples = [...markdown.matchAll(/^\.\/mom bench (\S+) --job\.dev "([^"\n]+)"$/gm)];
  assert.equal(examples.length, 4);
  for (const example of examples) {
    const algo = example[1];
    const dev = example[2];
    if (!algo || !dev) {throw new Error("README benchmark example is incomplete");}
    const opt = opts.create_default_opts();
    const cli = createCli({o: {...opts, print_help: (message) => {
      throw new Error(message || "Invalid README benchmark command");
    }}, opt, normalizeAlgoName: (value) => value});
    assert.equal(cli(["node", "mom.js", "bench", algo, "--job.dev", dev],
      {result: "", result_hash_hex: null, thread_tested: 0}), "bench");
    assert.equal(opt.job.algo, algo);
    assert.equal(opt.job.dev, dev);
  }
});

test("MOM_GPU_TEST_ALGO is validated without leaking the environment", () => {
  const previous = process.env["MOM_GPU_TEST_ALGO"];
  try {
    const firstAlgo = gpuAlgos[0];
    if (!firstAlgo) {throw new Error("GPU algorithm list is empty");}
    process.env["MOM_GPU_TEST_ALGO"] = firstAlgo;
    assert.deepEqual(requestedAlgos(), [gpuAlgos[0]]);
    process.env["MOM_GPU_TEST_ALGO"] = "not-a-gpu-algo";
    assert.throws(() => requestedAlgos(), /Unknown MOM_GPU_TEST_ALGO/);
  } finally {
    if (previous === undefined) {
      delete process.env["MOM_GPU_TEST_ALGO"];
    } else {
      process.env["MOM_GPU_TEST_ALGO"] = previous;
    }
  }
  assert.equal(process.env["MOM_GPU_TEST_ALGO"], previous);
});
