#!/usr/bin/env node
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {normalizeAlgoName} = require("../miner/algorithms");
const miners = require("./sota-miners");

/** @typedef {{epoch: string, devices: string, seconds: string}} BenchmarkOptions */
/** @typedef {import("./sota-miners").BenchmarkSpec} BenchmarkSpec */
/** @typedef {import("./sota-miners").PoolSpec} PoolSpec */
/** @typedef {import("./sota-miners").Qualification} Qualification */
/** @typedef {import("./sota-miners").MinerSpec} MinerSpec */
/** @typedef {{url: string, sha256: string}} ArchiveReference */
/** @typedef {{value: number, unit: string}} Rate */
/** @typedef {{id: string, miner: MinerSpec, spec?: BenchmarkSpec, poolOnly?: PoolSpec}} Comparator */
/** @typedef {{archive: string, image: string, script: string, vendor: string,
 * amdOpencl?: string | undefined, devices?: string}} DockerOptions */
/** @typedef {{error?: Error | null, signal?: string | null, status: number | null}} CurlResult */

const vendors = new Set(["intel", "amd", "nvidia"]);
/** @type {Record<string, MinerSpec>} */
const minerSpecs = miners;
const integerPattern = /^\d+$/;
/** @type {Record<"epoch" | "seconds", number>} */
const limits = {epoch: 1_000_000, seconds: 3600};

/** @param {unknown} value @param {"epoch" | "seconds"} name @returns {string} */
function positiveInteger(value, name) {
  if (typeof value !== "string") {
    throw new Error(`${name} must be a positive integer at most ${limits[name]}`);
  }
  const number = Number(value);
  if (!integerPattern.test(value) || !Number.isSafeInteger(number) ||
      number <= 0 || number > limits[name]) {
    throw new Error(`${name} must be a positive integer at most ${limits[name]}`);
  }
  return String(number);
}

/** @param {unknown} value @returns {value is string} */
function isRateUnit(value) {
  return typeof value === "string" && /^(?:g|i|sol|[kmgtp]?h)\/s$/i.test(value);
}

/** @param {string} value @returns {string} */
function deviceSelector(value) {
  if (typeof value !== "string" || !value.split(",").every((device) => integerPattern.test(device))) {
    throw new Error("devices must be comma-separated nonnegative integers");
  }
  const devices = value.split(",").map(Number);
  if (!devices.length || devices.some((device) => !Number.isSafeInteger(device) || device > 1023)) {
    throw new Error("devices must be comma-separated nonnegative integers");
  }
  return devices.join(",");
}

/** @param {string | number} value @returns {string} */
function shellQuote(value) {
  return `'${String(value).replace(/'/g, "'\\''")}'`;
}

/** @param {(string | number)[]} args @returns {string} */
function shellCommand(args) {
  return args.map(shellQuote).join(" ");
}

/** @param {Partial<Record<"epoch" | "devices" | "seconds", string>>} [options]
 * @returns {BenchmarkOptions} */
function benchmarkOptions(options = {}) {
  return {
    epoch: positiveInteger(options.epoch ?? process.env["MOM_SOTA_EPOCH"] ?? "440", "epoch"),
    devices: deviceSelector(options.devices ?? process.env["MOM_SOTA_DEVICES"] ?? "0"),
    seconds: positiveInteger(options.seconds ?? process.env["MOM_SOTA_SECONDS"] ?? "90", "seconds"),
  };
}

/** @param {MinerSpec} miner @param {string} algo @param {string} [vendor]
 * @returns {BenchmarkSpec | null} */
function benchmarkSpec(miner, algo, vendor) {
  const benchmark = miner.benchmark?.[algo];
  if (!benchmark) {return null;}
  const spec = typeof benchmark === "string" ? {args: ["--benchmark", benchmark]} : benchmark;
  return vendor && spec.vendors && !spec.vendors.includes(vendor) ? null : spec;
}

/** @returns {string[]} */
function comparatorAlgos() {
  return [...new Set(Object.values(minerSpecs).flatMap((miner) => [
    ...Object.keys(miner.benchmark || {}), ...Object.keys(miner.poolOnly || {}),
  ]))].sort();
}

/** @param {string} algo @param {string} [vendor] @returns {Comparator[]} */
function comparatorsFor(algo, vendor) {
  return Object.entries(minerSpecs).flatMap(([id, miner]) => {
    const spec = benchmarkSpec(miner, algo, vendor);
    const poolOnly = miner.poolOnly?.[algo];
    const vendorPool = poolOnly && (!vendor || !poolOnly.vendors || poolOnly.vendors.includes(vendor));
    return [
      ...(spec ? [{id, miner, spec}] : []),
      ...(vendorPool ? [{id, miner, poolOnly}] : []),
    ];
  });
}

/** @param {MinerSpec} miner @returns {string} */
function archiveCommand(miner) {
  return miner.archive === "tar.xz" ? "tar -xJf" : "tar -xzf";
}

/** @param {string} vendor @param {string} [devices] @returns {string[]} */
function deviceArgs(vendor, devices = "0") {
  if (!vendors.has(vendor)) {throw new Error("Unknown GPU vendor");}
  const selector = deviceSelector(devices);
  if (vendor === "nvidia") {return ["--gpus", `device=${selector}`];}
  return ["--device", "/dev/dri", ...(vendor === "amd" ? ["--device", "/dev/kfd"] : [])];
}

/** @param {MinerSpec} miner @param {string} algo @param {string} vendor
 * @param {Partial<Record<"epoch" | "devices" | "seconds", string>>} [options]
 * @returns {string} */
function buildContainerScript(miner, algo, vendor, options = {}) {
  if (!vendors.has(vendor)) {throw new Error("Unknown GPU vendor");}
  const spec = benchmarkSpec(miner, algo, vendor);
  if (!spec) {throw new Error(`${miner.name} has no offline benchmark for ${algo}`);}
  const exe = miner.exe;
  const strip = miner.strip;
  if (typeof exe !== "string" || !/^[a-zA-Z0-9_.-]+(?:\/[a-zA-Z0-9_.-]+)*$/.test(exe) ||
      exe.split("/").some((part) => part === "." || part === "..") ||
      typeof strip !== "number" || !Number.isSafeInteger(strip) || strip < 0 || strip > 16 ||
      !Array.isArray(spec.args) || spec.args.length > 64 ||
      spec.args.some((arg) => typeof arg !== "string" || /[\0\r\n]/.test(arg)) ||
      !Number.isSafeInteger(spec.runs ?? 1) || (spec.runs ?? 1) < 1 || (spec.runs ?? 1) > 12 ||
      (spec.runner !== undefined && spec.runner !== "wildrig") ||
      (miner.exitPolicy?.[algo]?.accepted || []).some((status) =>
        !Number.isSafeInteger(status) || status < 0 || status > 255)) {
    throw new Error(`${miner.name} has invalid offline archive metadata`);
  }
  const {epoch, devices, seconds} = benchmarkOptions(options);
  const minerDevices = vendor === "nvidia" ? devices.split(",").map((_, index) => index).join(",") : devices;
  const executable = `./${exe}`;
  const command = spec.runner === "wildrig" ?
    shellCommand(["timeout", "-s", "INT", "-k", "10s", String(Number(seconds) + 30),
      executable, ...spec.args, "--benchmark-timeout", seconds,
      "--opencl-platforms", vendor, "-d", minerDevices,
      ...(vendor === "amd" ? ["--no-adl"] : []), "--no-color", "--print-time", "5"]) :
    shellCommand(["timeout", "-s", "INT", "-k", "10s", seconds, executable,
      ...spec.args, "--devices", minerDevices, "--benchepoch", epoch, "--nocolor", "--watchdog", "off",
      "--longstats", spec.longStats || 10, "--shortstats", "10", "--log", "off"]);
  const acceptedStatuses = [...new Set([0, 124, ...(miner.exitPolicy?.[algo]?.accepted || [])])];
  const acceptedCheck = acceptedStatuses.map((status) =>
    `[ "$benchmark_status" -eq ${status} ]`).join(" || ");
  const timedCommand = "benchmark_status=0; " + command + " || benchmark_status=$?; " +
    `${acceptedCheck} || exit "$benchmark_status"`;
  return [
    "set -eu",
    "mkdir /tmp/miner",
    shellCommand([...archiveCommand(miner).split(" "), "/archive", "-C", "/tmp/miner",
      `--strip-components=${strip}`, "--no-same-owner", "--no-same-permissions"]),
    // Miner archives include relative configuration and kernel files. Run inside the extracted tree.
    "cd /tmp/miner",
    vendor === "amd" ? "export OCL_ICD_FILENAMES=/sota/libamdocl64.so" : "unset OCL_ICD_FILENAMES",
    ...Array.from({length: spec.runs || 1}, () => timedCommand),
  ].join("; ");
}

/** @param {DockerOptions} options @returns {string[]} */
function buildDockerArgs({archive, image, script, vendor, amdOpencl, devices = "0"}) {
  const gpu = deviceArgs(vendor, devices);
  if (typeof image !== "string" || !/^[a-zA-Z0-9][a-zA-Z0-9_./:@-]*$/.test(image)) {
    throw new Error("Invalid container image");
  }
  // Bind syntax is parsed by Docker even without a shell; reject its separators.
  for (const file of [archive, ...(amdOpencl ? [amdOpencl] : [])]) {
    if (typeof file !== "string" || !path.isAbsolute(file) || /[:,\0\r\n]/.test(file)) {
      throw new Error("Invalid container bind path");
    }
  }
  if (amdOpencl && vendor !== "amd") {throw new Error("AMD loader requires the AMD vendor");}
  if (!/\.(?:tar\.gz|tar\.xz)$/.test(archive) || (amdOpencl && !/\.so(?:\.[0-9]+)*$/.test(amdOpencl))) {
    throw new Error("Container binds must be an archive and an OpenCL library");
  }
  return ["run", "--rm", "--pull", "never", "--user", "65534:65534",
    "--network", "none", "--read-only", "--memory", "4g", "--memory-swap", "4g",
    "--cpus", "4", "--pids-limit", "256", "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--tmpfs", "/tmp:exec,size=1g",
    "--entrypoint", "bash", ...gpu, "-v", `${archive}:/archive:ro`,
    ...(amdOpencl ? ["-v", `${amdOpencl}:/sota/libamdocl64.so:ro`] : []), image, "-lc", script];
}

/** @param {string} archive @param {ArchiveReference} reference */
function verifyArchive(archive, reference) {
  const file = fs.lstatSync(archive);
  if (!file.isFile() || file.size > 512 * 1024 * 1024) {
    throw new Error("Archive must be a regular file at most 512 MiB, not a symlink");
  }
  const digest = crypto.createHash("sha256").update(fs.readFileSync(archive)).digest("hex");
  if (digest !== reference.sha256) {throw new Error(`Bad checksum for ${archive}: ${digest}`);}
}

/** @param {ArchiveReference} reference @param {string} cache
 * @param {(args: string[]) => CurlResult} [runCurl] @returns {string | number} */
function downloadArchive(reference, cache, runCurl = (args) =>
  spawnSync("curl", args, {stdio: "inherit"})) {
  const url = new URL(reference.url);
  const basename = path.basename(url.pathname);
  if (url.protocol !== "https:" || url.username || url.password ||
      !/^[a-zA-Z0-9][a-zA-Z0-9_.-]*\.(?:tar\.gz|tar\.xz)$/.test(basename) ||
      !/^[0-9a-f]{64}$/.test(reference.sha256)) {
    throw new Error("Invalid archive reference");
  }
  cache = path.resolve(cache);
  fs.mkdirSync(cache, {recursive: true, mode: 0o700});
  const directory = fs.lstatSync(cache);
  if (!directory.isDirectory() || fs.realpathSync(cache) !== cache || (directory.mode & 0o077) !== 0 ||
      (typeof process.getuid === "function" && directory.uid !== process.getuid())) {
    throw new Error("Archive cache must be an owned private directory, not a symlink");
  }
  const archive = path.join(cache, basename);
  if (fs.existsSync(archive)) {
    try {
      verifyArchive(archive, reference);
      return archive;
    } catch (error) {
      if (!(error instanceof Error) || !error.message.startsWith(`Bad checksum for ${archive}:`)) {
        throw error;
      }
    }
  }

  const tempDir = fs.mkdtempSync(path.join(cache, ".download-"));
  const tempArchive = path.join(tempDir, path.basename(archive));
  try {
    const download = runCurl(["-q", "-fL", "--proto", "=https", "--proto-redir", "=https",
      "--max-filesize", "536870912", "--retry", "3", "--connect-timeout", "30",
      "--max-time", "600", "-o", tempArchive, reference.url]);
    if (download.error) {throw download.error;}
    if (download.signal) {return 1;}
    if (download.status !== 0) {return download.status || 1;}
    verifyArchive(tempArchive, reference);
    fs.renameSync(tempArchive, archive);
    return archive;
  } finally {
    fs.rmSync(tempDir, {recursive: true, force: true});
  }
}

/** @param {string} output @returns {Rate | null} */
function lastRate(output) {
  const matches = [...output.matchAll(/Average speed \(\d+s\):\s*([0-9.]+)\s*(\S+)/g)];
  if (!matches.length) {return null;}
  const rates = matches.map((match) => {
    const value = match[1];
    const unit = match[2];
    const number = value ? Number(value) : Number.NaN;
    return isRateUnit(unit) && Number.isFinite(number) ? {value: number, unit} : null;
  }).filter((rate) => rate !== null);
  const positive = rates.reverse().find(({value}) => value > 0);
  return positive || rates[0] || null;
}

/** @param {number[]} values @returns {number | null} */
function numericMedian(values) {
  if (!values.length || !values.every(Number.isFinite)) {return null;}
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  const upper = sorted[middle];
  if (upper === undefined) {return null;}
  if (sorted.length & 1) {return upper;}
  const lower = sorted[middle - 1];
  return lower === undefined ? null : (lower + upper) / 2;
}

/** @param {BenchmarkSpec} spec @param {string} output @returns {Rate | null} */
function benchmarkRate(spec, output) {
  if (spec.rate === "lolminer-input-iterations") {
    if (!isRateUnit(spec.unit)) {return null;}
    // lolMiner prints units in the table header; Total's third numeric column is input iterations.
    const values = [...output.matchAll(/^Total\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\b/gm)]
      .map((match) => Number(match[1])).filter((value) => Number.isFinite(value) && value > 0);
    const runs = spec.runs || 1;
    if (values.length < runs) {return null;}
    const value = numericMedian(values.slice(-runs));
    return value === null ? null : {value, unit: spec.unit};
  }
  if (spec.rate === "wildrig-60s") {
    if (!isRateUnit(spec.unit)) {return null;}
    const completedRuns = output.split(/Benchmark finished[^\n]*/).slice(0, -1);
    const values = completedRuns.map((block) => {
      const matches = [...block.matchAll(/^\s*60s:\s*([0-9.]+)\s+(\S+)/gm)]
        .filter((match) => match[2]?.toLowerCase() === spec.unit?.toLowerCase());
      const match = matches.at(-1);
      return match?.[1] ? Number(match[1]) : 0;
    }).filter((value) => value > 0).slice(-(spec.runs || 1)).sort((a, b) => a - b);
    if (values.length < (spec.runs || 1)) {return null;}
    const value = numericMedian(values);
    return value === null ? null : {value, unit: spec.unit};
  }
  if (spec.rate === "gpu-speed") {
    if (!isRateUnit(spec.unit)) {return null;}
    const tailSamples = Math.ceil((spec.runs || 1) / 2);
    const samples = [...output.matchAll(/^GPU\s+\d+\s+.+?\s+([0-9.]+)\s+[0-9.]+\s+\d+\/\d+\/\d+\s+/gm)]
      .map((match) => Number(match[1])).filter((value) => Number.isFinite(value) && value > 0);
    if (samples.length < tailSamples) {return null;}
    const values = samples.slice(-tailSamples).sort((a, b) => a - b);
    const value = numericMedian(values);
    return value === null ? null : {value, unit: spec.unit};
  }
  if (spec.rate === "average-speed-runs") {
    if (!isRateUnit(spec.unit)) {return null;}
    const unit = spec.unit.toLowerCase();
    const tailSamples = spec.tailSamples || 1;
    const runs = spec.runs || 1;
    const blockMedians = output.split("Start Benchmark...").slice(1).flatMap((block) => {
      const rates = [...block.matchAll(/Average speed \(\d+s\):\s*([0-9.]+)\s+(\S+)/g)]
        .map((match) => ({value: Number(match[1]), unit: (match[2] || "").toLowerCase()}))
        .filter(({value, unit: rateUnit}) => value > 0 && rateUnit === unit)
        .map(({value}) => value);
      const value = rates.length >= tailSamples ? numericMedian(rates.slice(-tailSamples)) : null;
      return value === null ? [] : [value];
    });
    if (blockMedians.length < runs) {return null;}
    const value = numericMedian(blockMedians.slice(-runs));
    return value === null ? null : {value, unit: spec.unit};
  }
  return lastRate(output);
}

/** @param {MinerSpec} miner @param {string} algo @param {number | null} status
 * @param {boolean} hasEvidence @returns {boolean} */
function exitAccepted(miner, algo, status, hasEvidence) {
  if (!hasEvidence) {return false;}
  if (status === 0 || status === 124) {return true;}
  if (typeof status !== "number") {return false;}
  return miner.exitPolicy?.[algo]?.accepted?.includes(status) || false;
}

/** @param {Qualification | null | undefined} qualification @param {Rate | null}
 * @returns {{ratio: number, pass: boolean} | null} */
function qualificationResult(qualification, rate) {
  if (!qualification || !rate || rate.value <= 0) {return null;}
  if (rate.unit.toLowerCase() !== qualification.unit.toLowerCase()) {
    return {ratio: 0, pass: false};
  }
  const ratio = qualification.momRate / rate.value;
  return {ratio, pass: ratio >= qualification.minRatio};
}

/** @param {Qualification | null | undefined} qualification @param {string}
 * @returns {Qualification | null} */
function qualificationForVendor(qualification, vendor) {
  if (!qualification || (Object.hasOwn(qualification, "vendor") && qualification.vendor !== vendor)) {
    return null;
  }
  return qualification;
}

function run() {
  if (process.platform !== "linux") {
    console.error("Reference miners require the Linux container runner or a disposable Windows VM");
    return 2;
  }
  const [requestedAlgo, vendor = "intel"] = process.argv.slice(2);
  const algo = normalizeAlgoName(requestedAlgo || "");
  if (!vendors.has(vendor)) {
    console.error(`Unknown GPU vendor: ${vendor}`);
    return 2;
  }
  const candidates = algo ? comparatorsFor(algo, vendor) : [];
  const requestedMiner = process.env["MOM_SOTA_MINER"];
  const selected = candidates.find(({id}) => !requestedMiner || id === requestedMiner);
  if (!algo || !candidates.length || !selected) {
    console.error(`Usage: npm run test:sota -- <${comparatorAlgos().join("|")}> [intel|amd|nvidia]`);
    return 2;
  }
  if (!selected.spec) {
    const poolOnly = selected.poolOnly;
    if (!poolOnly) {
      console.error(`${selected.miner.name} ${algo}: comparator metadata is incomplete`);
      return 3;
    }
    console.error(`${selected.miner.name} ${algo}: unsupported offline comparator`);
    console.error(`Reason: ${poolOnly.reason}`);
    console.error("Pool commands are metadata only; use a constrained container or disposable Windows VM");
    return 3;
  }

  /** @type {BenchmarkOptions | undefined} */
  let options;
  try {
    options = benchmarkOptions();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    return 2;
  }

  const reference = selected.miner;
  const cache = process.env["MOM_SOTA_CACHE"] || path.join(os.tmpdir(), "mom-sota-cache");
  if (typeof reference.url !== "string" || typeof reference.sha256 !== "string") {
    throw new Error(`${reference.name} has no downloadable archive metadata`);
  }
  const archive = downloadArchive({url: reference.url, sha256: reference.sha256}, cache);
  if (typeof archive === "number") {return archive;}

  const image = process.env["MOM_SOTA_IMAGE"] || "mom-build-multicompiler:latest";
  const amdOpencl = process.env["MOM_SOTA_AMD_OPENCL"] || "/usr/lib/x86_64-linux-gnu/libamdocl64.so";
  const loader = vendor === "amd" ? fs.realpathSync(amdOpencl) : undefined;
  if (loader && !fs.statSync(loader).isFile()) {throw new Error("AMD OpenCL loader must be a regular file");}
  const script = buildContainerScript(reference, algo, vendor, options);
  console.log(`${reference.name}: ${algo} (${vendor}, isolated container)`);
  const result = spawnSync("docker", buildDockerArgs({archive, image, script, vendor,
    devices: options.devices, amdOpencl: loader}),
  {encoding: "utf8"});
  if (result.error) {throw result.error;}
  if (result.signal) {throw new Error("Reference container terminated by signal");}
  process.stdout.write(result.stdout || "");
  process.stderr.write(result.stderr || "");

  const output = `${result.stdout || ""}\n${result.stderr || ""}`;
  const rate = benchmarkRate(selected.spec, output);
  if (!rate || rate.value <= 0) {
    console.error(`${reference.name} did not produce a stable ${algo} rate`);
    return result.status || 1;
  }
  if (!exitAccepted(reference, algo, result.status, rate.value > 0)) {return result.status || 1;}

  const selectedQualification = qualificationForVendor(reference.qualification?.[algo], vendor);
  const qualification = qualificationResult(selectedQualification, rate);
  if (qualification && selectedQualification) {
    const percent = (qualification.ratio * 100).toFixed(1);
    console.log(`Qualification: ${percent}% (${selectedQualification.momRate} / ` +
      `${rate.value} ${rate.unit}, minimum ${(selectedQualification.minRatio * 100).toFixed(0)}%)`);
    if (!qualification.pass) {return 1;}
  }
  return 0;
}

if (require.main === module) {process.exitCode = run();}

module.exports = {
  benchmarkSpec,
  benchmarkRate,
  buildContainerScript,
  buildDockerArgs,
  comparatorAlgos,
  comparatorsFor,
  deviceArgs,
  downloadArchive,
  exitAccepted,
  lastRate,
  qualificationResult,
  qualificationForVendor,
};
