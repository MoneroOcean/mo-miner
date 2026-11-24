"use strict";

// Build-only CUDA image planning and immutable-resource generation. This module never starts
// NVRTC itself; callers pass public-API query results and outputs from native-cuda-images.cpp.
const fs = require("node:fs"), path = require("node:path"), crypto = require("node:crypto");
const {hasExactBuildFields, fileHash, readBuildJsonSnapshot} = require("./build-source-inputs");
const {cudaHeaderInputs, collectHeaders, packHeaders} = require("./native-jit-headers");

const maxImageBytes = 64 << 20, maxIncBytes = 512 << 20, maxImages = 2048, maxInputFiles = 4096;
const profiles = [
  {name: "cn-expand", source: "sycl/cn-gpu/cuda-expand-kernel.inc", program: "cn-gpu-expand.cu", exports: ["cn_gpu_expand_cuda"]},
  {name: "pearl-rank128", source: "sycl/pearlhash/cuda-kernel.inc", program: "pearlhash-cute.cu", rank: 128,
    exports: ["pearlhash_cuda_gemm", "pearlhash_cuda_hash"]},
  {name: "pearl-rank256", source: "sycl/pearlhash/cuda-kernel.inc", program: "pearlhash-cute.cu", rank: 256,
    exports: ["pearlhash_cuda_gemm", "pearlhash_cuda_hash"]},
  {name: "pearl-compact", source: "sycl/pearlhash/cuda-compact-kernel.inc", program: "pearlhash-cute.cu", rank: 128,
    compact: true, exports: ["compact_gemm", "selector_prepare"]},
];
const profileOrder = new Map(profiles.map(({name}, index) => [name, index]));
const genericProfiles = profiles.slice(0, 3);
const sourceFiles = ["scripts/build-source-inputs.js", "scripts/native-cuda-images.cpp",
  "scripts/native-cuda-images.js", "scripts/native-jit-headers.js", "scripts/native-resources.js",
  "sycl/cn-gpu/cuda-expand-kernel.inc", "sycl/pearlhash/cuda-kernel.inc", "sycl/pearlhash/cuda-compact-kernel.inc"];

/** @typedef {{major: number, minor: number, supportedArchs: number[]}} NvrtcInfo */
/** @typedef {{profile: string, target: string, format: "cubin" | "ptx", role: "exact" | "primary" | "compatibility-candidate", compiler: NvrtcInfo}} ImageJob */
/** @typedef {{name: string, source: string, program: string, exports: string[], rank?: number, compact?: boolean}} Profile */
/** @typedef {{path: string, manifest: Record<string, unknown>, files: Record<string, string>}} HeaderPack */

/** @param {unknown} value @returns {value is Record<string, unknown>} */
function record(value) {return value !== null && typeof value === "object" && !Array.isArray(value);}

/** Validate only facts obtained from NVRTC's public version and supported-architecture APIs.
 * @param {unknown} value @returns {NvrtcInfo} */
function compilerInfo(value) {
  if (!record(value) || typeof value["major"] !== "number" || !Number.isSafeInteger(value["major"]) || value["major"] < 1 || value["major"] > 99 ||
      typeof value["minor"] !== "number" || !Number.isSafeInteger(value["minor"]) || value["minor"] < 0 || value["minor"] > 99 ||
      !Array.isArray(value["supportedArchs"]) || value["supportedArchs"].length < 1 || value["supportedArchs"].length > 128 ||
      value["supportedArchs"].some((arch) => typeof arch !== "number" || !Number.isSafeInteger(arch) || arch < 50 || arch > 999) ||
      new Set(value["supportedArchs"]).size !== value["supportedArchs"].length) {
    throw new Error("Invalid NVRTC version or supported-architecture query");
  }
  return {major: value["major"], minor: value["minor"],
    supportedArchs: /** @type {number[]} */ ([...value["supportedArchs"]]).sort((a, b) => a - b)};
}

/** Sort compiler queries by actual NVRTC version; never infer supported SMs or PTX ISA.
 * @param {unknown[]} queries @returns {ImageJob[]} */
function planCudaImages(queries) {
  if (!Array.isArray(queries) || queries.length < 1 || queries.length > 16) {throw new Error("Provide bounded NVRTC query results");}
  const compilers = queries.map(compilerInfo).sort(compareCompiler);
  if (new Set(compilers.map(versionKey)).size !== compilers.length) {throw new Error("Duplicate NVRTC compiler versions");}
  /** @param {NvrtcInfo} compiler @param {number} arch */
  const supports = (compiler, arch) => compiler.supportedArchs.includes(arch);
  const has80 = compilers.some((compiler) => supports(compiler, 80));
  if (!has80) {throw new Error("No selected NVRTC compiler supports the fixed compute_80 PTX target");}
  const architectures = [...new Set(compilers.flatMap((compiler) => compiler.supportedArchs.filter((arch) => arch >= 80)))].sort((a, b) => a - b);
  const baseline = compilers.find(({major, minor}) => major === 12 && minor === 6);
  /** @type {ImageJob[]} */
  const jobs = [];
  /** @param {string} profile @param {string} target @param {"cubin" | "ptx"} format
   * @param {ImageJob["role"]} role @param {NvrtcInfo} compiler */
  const add = (profile, target, format, role, compiler) => jobs.push({profile, target, format, role, compiler});
  for (const profile of genericProfiles) {
    for (const arch of architectures) {
      const candidates = compilers.filter((compiler) => supports(compiler, arch));
      const selected = baseline && supports(baseline, arch) ? baseline : candidates.at(-1);
      if (selected) {add(profile.name, `sm_${arch}`, "cubin", "exact", selected);}
      const ptxCompiler = candidates.at(-1);
      if (ptxCompiler) {add(profile.name, `compute_${arch}`, "ptx", "primary", ptxCompiler);}
    }
    const primary80 = jobs.find((job) => job.profile === profile.name && job.target === "compute_80" && job.role === "primary");
    if (baseline && supports(baseline, 80) && primary80 && versionKey(baseline) !== versionKey(primary80.compiler)) {
      add(profile.name, "compute_80", "ptx", "compatibility-candidate", baseline);
    }
  }
  const compactCompiler = compilers.filter(({major, minor, supportedArchs}) =>
    (major > 13 || (major === 13 && minor >= 3)) && supportedArchs.includes(120)).at(-1);
  if (compactCompiler) {add("pearl-compact", "sm_120a", "cubin", "exact", compactCompiler);}
  if (jobs.length > maxImages) {throw new Error("NVRTC image plan exceeds its bounds");}
  return jobs;
}

/** @param {NvrtcInfo} left @param {NvrtcInfo} right */
function compareCompiler(left, right) {return left.major - right.major || left.minor - right.minor;}

/** @param {NvrtcInfo} compiler */
function versionKey(compiler) {return `${compiler.major}.${compiler.minor}`;}

/** @param {Profile} profile @param {string} target @param {"cubin" | "ptx"} format @param {string[]} includeRoots */
function compileOptions(profile, target, format, includeRoots = []) {
  if (includeRoots.length > 16 || includeRoots.some((root) => typeof root !== "string" || !isAbsoluteProducerPath(root))) {
    throw new Error("NVRTC include roots must be bounded absolute directories");
  }
  const options = ["--std=c++17"];
  if (profile.rank) {options.push(`-DPEARLHASH_RANK=${profile.rank}`);}
  options.push(`--gpu-architecture=${target}`);
  for (const root of includeRoots) {options.push(`--include-path=${root}`);}
  options.push("--device-as-default-execution-space");
  if (profile.rank) {options.push("--use_fast_math");}
  if (profile.compact) {options.push("-DDIRECT_RANK_STRIDE=1");}
  if ((format === "cubin") !== target.startsWith("sm_")) {throw new Error("NVRTC format does not match target");}
  return options;
}

/** @param {string} target */
function architectureOf(target) {
  const match = /^(?:sm_|compute_)(\d+)(a?)$/.exec(target);
  if (!match || (match[2] && target !== "sm_120a")) {throw new Error("Invalid planned NVRTC target");}
  return Number(match[1]);
}

/** @param {unknown} image @returns {Buffer} */
function imageBytes(image) {
  if (!Buffer.isBuffer(image) || image.length < 1 || image.length > maxImageBytes) {
    throw new Error("CUDA image is empty or exceeds 64 MiB");
  }
  return image;
}

/** @param {string | Buffer} value */
function digest(value) {return crypto.createHash("sha256").update(value).digest("hex");}

/** @param {unknown} roots @param {Profile} profile @returns {string[]} */
function checkedRoots(roots, profile) {
  if (!Array.isArray(roots) || roots.length > 16 || roots.some((root) => typeof root !== "string" || !isAbsoluteProducerPath(root)) ||
      (!profile.rank && roots.length !== 0) || (Boolean(profile.rank) && roots.length === 0)) {
    throw new Error("CUDA image include roots do not match the maintained profile");
  }
  return /** @type {string[]} */ (roots);
}

/** Match worker stdout and the actual image to the planned source, flags, query, and headers.
 * @param {ImageJob} job @param {unknown} metadata @param {Buffer} data @param {string[]} includeRoots */
function verifyArtifact(job, metadata, data, includeRoots) {
  if (!record(metadata) || metadata["status"] !== "ok" || !record(metadata["nvrtc"]) ||
      metadata["nvrtc"]["major"] !== job.compiler.major || metadata["nvrtc"]["minor"] !== job.compiler.minor ||
      !Array.isArray(metadata["supportedArchs"]) || JSON.stringify(metadata["supportedArchs"]) !== JSON.stringify(job.compiler.supportedArchs)) {
    throw new Error("NVRTC image metadata does not match its admitted compiler query");
  }
  const profile = profileFor(job.profile);
  if (metadata["profile"] !== job.profile || metadata["source"] !== profile.source || metadata["program"] !== profile.program ||
      metadata["target"] !== job.target || metadata["format"] !== job.format || metadata["outputBytes"] !== data.length) {
    throw new Error("NVRTC image metadata does not match its planned profile or target");
  }
  if (!Array.isArray(metadata["includeRoots"]) || JSON.stringify(metadata["includeRoots"]) !== JSON.stringify(includeRoots) ||
      JSON.stringify(metadata["options"]) !== JSON.stringify(compileOptions(profile, job.target, job.format, includeRoots))) {
    throw new Error("NVRTC image options differ from the maintained kernel flags");
  }
  if (!Array.isArray(metadata["exports"]) || metadata["exports"].length !== profile.exports.length ||
      metadata["exports"].some((name, index) => typeof name !== "string" || name !== profile.exports[index])) {
    throw new Error("NVRTC image does not contain the maintained runtime exports");
  }
  if (job.format === "ptx") {
    validatePtx(data, job.target, metadata["ptxVersion"], profile.exports);
    if (job.compiler.major === 12 && job.compiler.minor === 6 && metadata["ptxVersion"] !== "8.5") {
      throw new Error("NVRTC 12.6 PTX must report its actual 8.5 ISA");
    }
  } else {
    if (metadata["ptxVersion"] !== null) {throw new Error("CUBIN unexpectedly reports a PTX ISA");}
    validateCudaCubin(data, profile.exports);
  }
}

/** @param {Buffer} data @param {string} name @param {"cubin" | "ptx"} format */
function imageHasExport(data, name, format) {
  if (format === "cubin") {return data.includes(Buffer.from(name));}
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(?:^|\\n)\\s*(?:\\.visible\\s+)?\\.entry\\s+${escaped}(?=[\\s(])`).test(data.toString("ascii"));
}

/** PTX is passed to cuModuleLoadData as a NUL-terminated byte string.
 * @param {Buffer} data @param {string} target @param {unknown} version @param {string[]} exports */
function validatePtx(data, target, version, exports) {
  if (data.length < 2 || data[data.length - 1] !== 0 || data.subarray(0, -1).includes(0) ||
      data.subarray(0, -1).some((byte) => byte > 0x7f) ||
      typeof version !== "string" || !/^\d{1,3}\.\d{1,3}$/.test(version)) {
    throw new Error("Malformed or unterminated NVRTC PTX image");
  }
  const text = data.toString("ascii", 0, data.length - 1);
  const actualVersion = text.match(/(?:^|\n)\.version\s+(\d{1,3}\.\d{1,3})(?:\s|$)/)?.[1];
  const actualTarget = text.match(/(?:^|\n)\.target\s+(sm_\d{2,3}a?)(?:[\s,]|$)/)?.[1];
  const expectedTarget = `sm_${architectureOf(target)}`;
  if (actualVersion !== version || actualTarget !== expectedTarget || !exports.every((name) => imageHasExport(data.subarray(0, -1), name, "ptx"))) {
    throw new Error("PTX target, ISA, or entry exports differ from the admitted image metadata");
  }
}

/** Inspect the bounded ELF64 section and symbol tables needed to verify a CUDA CUBIN.
 * @param {Buffer} data @param {string[]} exports */
function validateCudaCubin(data, exports) {
  if (data.length < 64 || data[0] !== 0x7f || data.toString("ascii", 1, 4) !== "ELF" ||
      data[4] !== 2 || data[5] !== 1 || data[6] !== 1 || data.readUInt16LE(16) !== 2 || data.readUInt16LE(18) !== 190) {
    throw new Error("NVRTC CUBIN is not ELF64 little-endian EM_CUDA");
  }
  const sectionOffset = data.readBigUInt64LE(40), sectionSize = data.readUInt16LE(58), sectionCount = data.readUInt16LE(60);
  if (sectionSize !== 64 || sectionCount < 1 || sectionCount > 4096 ||
      sectionOffset > BigInt(data.length) || sectionOffset + BigInt(sectionSize) * BigInt(sectionCount) > BigInt(data.length)) {
    throw new Error("NVRTC CUBIN has invalid ELF section bounds");
  }
  /** @param {number} index */
  function section(index) {
    const offset = Number(sectionOffset + BigInt(sectionSize) * BigInt(index));
    const type = data.readUInt32LE(offset + 4), link = data.readUInt32LE(offset + 40);
    const begin = data.readBigUInt64LE(offset + 24), size = data.readBigUInt64LE(offset + 32), entrySize = data.readBigUInt64LE(offset + 56);
    // SHT_NOBITS describes zero-initialized device storage, not bytes in the CUBIN. Its
    // conceptual offset/size may extend beyond EOF; only file-backed sections need a range.
    if (type !== 8 && (begin > BigInt(data.length) || size > BigInt(data.length) || begin + size > BigInt(data.length))) {
      throw new Error("NVRTC CUBIN section data exceeds its bounds");
    }
    return {type, link, begin: Number(begin), size: Number(size), entrySize: Number(entrySize)};
  }
  const sections = Array.from({length: sectionCount}, (_, index) => section(index)), found = new Set();
  for (const table of sections.filter(({type}) => type === 2 || type === 11)) {
    if (table.link >= sections.length || table.entrySize < 24 || table.entrySize > 256 || table.size % table.entrySize) {
      throw new Error("NVRTC CUBIN has an invalid ELF symbol table");
    }
    const strings = sections[table.link];
    if (!strings || strings.type !== 3 || strings.size < 1 || data[strings.begin] !== 0 ||
        data[strings.begin + strings.size - 1] !== 0) {
      throw new Error("NVRTC CUBIN symbol table has no valid string table");
    }
    for (let at = table.begin; at < table.begin + table.size; at += table.entrySize) {
      const nameOffset = data.readUInt32LE(at), info = data.readUInt8(at + 4), binding = info >> 4, type = info & 0x0f;
      const sectionIndex = data.readUInt16LE(at + 6);
      if (!nameOffset || nameOffset >= strings.size || sectionIndex === 0 || sectionIndex >= sections.length ||
          ![1, 2].includes(binding) || ![0, 2].includes(type)) {continue;}
      const start = strings.begin + nameOffset, end = data.indexOf(0, start);
      if (end < 0 || end >= strings.begin + strings.size) {throw new Error("NVRTC CUBIN has an unterminated symbol name");}
      const name = data.toString("utf8", start, end);
      if (exports.includes(name)) {found.add(name);}
    }
  }
  if (!exports.every((name) => found.has(name))) {throw new Error("NVRTC CUBIN is missing a maintained global kernel export");}
}

/** @param {string} name @returns {Profile} */
function profileFor(name) {
  const profile = profiles.find((entry) => entry.name === name);
  if (!profile) {throw new Error("Unknown fixed CUDA image profile");}
  return profile;
}

/** @param {string} name */
function profileRank(name) {
  const rank = profileOrder.get(name);
  if (rank === undefined) {throw new Error("Unknown fixed CUDA image profile");}
  return rank;
}

/** @param {string} left @param {string} right */
function compareIsa(left, right) {
  const a = left.split(".").map(Number), b = right.split(".").map(Number);
  return (a[0] || 0) - (b[0] || 0) || (a[1] || 0) - (b[1] || 0);
}

/** @param {NvrtcInfo} compiler @param {ReadonlyMap<string, string>} inputs */
function sortCompilerRecord(compiler, inputs) {
  return {major: compiler.major, minor: compiler.minor, supportedArchs: compiler.supportedArchs,
    inputs: Object.fromEntries([...inputs].sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0))};
}

/** Validate path/hash maps on the producer, before reading any referenced file.
 * @param {unknown} value @param {string} label @returns {Map<string, string>} */
function producerInputMap(value, label) {
  if (!record(value) || Object.keys(value).length < 1 || Object.keys(value).length > maxInputFiles) {
    throw new Error(`${label} input map is missing or unbounded`);
  }
  const result = new Map();
  for (const [file, hash] of Object.entries(value)) {
    if (!path.isAbsolute(file) || /[\0\r\n]/.test(file) || typeof hash !== "string" || !/^[0-9a-f]{64}$/.test(hash)) {
      throw new Error(`${label} inputs must use absolute paths and SHA-256 digests`);
    }
    result.set(file, hash);
  }
  return result;
}

/** Hash each listed file again; producer inputs are not recursively discovered.
 * @param {ReadonlyMap<string, string>} inputs @param {string} label */
function verifyInputFiles(inputs, label) {
  for (const [file, expected] of inputs) {
    if (fileHash(file) !== expected) {throw new Error(`${label} changed during CUDA image generation: ${file}`);}
  }
}

/** Repack only the already selected, bounded CUDA include roots to detect stale SDK headers.
 * @param {string[]} roots */
function liveHeaderSnapshot(roots) {
  const entries = collectHeaders(roots, true), data = packHeaders(entries);
  return {sha256: digest(data), count: entries.length, bytes: data.length};
}

/** @param {string} repo @returns {Record<string, string>} */
function sourceSnapshot(repo) {
  return Object.fromEntries(sourceFiles.map((relative) => [relative, fileHash(path.join(repo, relative))]));
}

/** @param {string} value */
function isAbsoluteProducerPath(value) {return path.isAbsolute(value) || path.win32.isAbsolute(value);}

/** Validate a stored absolute-file digest map without touching producer files.
 * @param {unknown} value @returns {Record<string, string>} */
function admittedPathHashes(value) {
  if (!record(value) || Object.keys(value).length < 1 || Object.keys(value).length > maxInputFiles) {
    throw new Error("Invalid CUDA image producer input inventory");
  }
  for (const [file, hash] of Object.entries(value)) {
    if (!isAbsoluteProducerPath(file) || /[\0\r\n]/.test(file) || typeof hash !== "string" || !/^[0-9a-f]{64}$/.test(hash)) {
      throw new Error("Invalid CUDA image producer input inventory");
    }
  }
  return /** @type {Record<string, string>} */ (value);
}

/** CUDA CUBIN/PTX registries are large Clang inputs: fixed-width octal string data avoids one
 * integer AST node per byte. Octal escapes preserve each byte even when plain `char` is signed;
 * the runtime uses pointer/length blobs, with sizeof-1 excluding only C++'s final NUL, not any
 * embedded or payload-trailing NUL. Align opaque device images conservatively for module input.
 * @param {Array<{name: string, data: Buffer}>} resources @returns {string} */
function cudaResourceMap(resources) {
  const names = new Set(), octal = Array.from({length: 256}, (_, byte) => "\\" + byte.toString(8).padStart(3, "0"));
  const lines = ["// Generated immutable CUDA image resources.", "#include <string_view>",
    "static std::string_view embedded_resource(std::string_view name) {"];
  let total = 0;
  for (const [index, {name, data}] of resources.entries()) {
    if (typeof name !== "string" || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(name) || names.has(name)) {
      throw new Error("Invalid or duplicate native resource name");
    }
    if (!Buffer.isBuffer(data) || !data.length || data.length > maxImageBytes - total) {
      throw new Error("Native resource data exceeds its bounds");
    }
    total += data.length; names.add(name);
    lines.push(`  alignas(16) static constexpr char resource_${index}[] =`);
    for (let offset = 0; offset < data.length; offset += 256) {
      const escaped = [];
      for (const byte of data.subarray(offset, offset + 256)) {escaped.push(/** @type {string} */ (octal[byte]));}
      lines.push(`    "${escaped.join("")}"`);
    }
    lines.push("    ;", `  if (name == ${JSON.stringify(name)}) return {resource_${index}, sizeof(resource_${index}) - 1};`);
  }
  lines.push("  return {};", "}");
  return lines.join("\n") + "\n";
}

/** Build the generated include and provenance sidecar from worker outputs. `compilers` include
 * actual NVRTC query facts plus absolute files hashed before/after generation. `headerPacks` is
 * keyed by compiler version; consumer admission never opens those producer SDK paths.
 * @param {{repo: string, compilers: unknown[], artifacts: Array<{job: unknown, metadata: unknown, data: Buffer}>, headerPacks: Record<string, string>}} input */
function generateCudaImageBundle({repo, compilers, artifacts, headerPacks}) {
  if (!path.isAbsolute(repo) || !Array.isArray(artifacts)) {throw new Error("CUDA image generation requires an absolute source root and artifact list");}
  const plan = planCudaImages(compilers), planned = new Map(plan.map((job) => [jobKey(job), job]));
  const compilerInfoByVersion = new Map(compilers.map((query) => [versionKey(compilerInfo(query)), compilerInfo(query)]));
  /** @type {Map<string, Map<string, string>>} */
  const compilerInputs = new Map();
  for (const query of compilers) {
    const compiler = compilerInfo(query), files = producerInputMap(record(query) ? query["inputs"] : null, `NVRTC ${versionKey(compiler)}`);
    verifyInputFiles(files, `NVRTC ${versionKey(compiler)} compiler input`);
    compilerInputs.set(versionKey(compiler), files);
  }
  const pearlCompilerVersions = [...new Set(plan.filter((job) => job.profile.startsWith("pearl-"))
    .map((job) => versionKey(job.compiler)))].sort();
  if (!record(headerPacks) || !hasExactBuildFields(headerPacks, pearlCompilerVersions)) {
    throw new Error("Provide one admitted CUDA JIT header pack for each Pearl compiler version");
  }
  const beforeSources = sourceSnapshot(repo);
  /** @type {Record<string, HeaderPack>} */
  const admittedHeaders = {};
  /** @type {Map<string, Map<string, string>>} */
  const headerInputFiles = new Map();
  /** @type {Map<string, string[]>} */
  const headerRoots = new Map();
  for (const version of pearlCompilerVersions) {
    const headerPath = headerPacks[version];
    if (typeof headerPath !== "string" || !path.isAbsolute(headerPath)) {throw new Error(`Missing absolute CUDA JIT header pack for NVRTC ${version}`);}
    const admission = cudaHeaderInputs(headerPath);
    if (!Array.isArray(admission.manifest["roots"]) || !admission.manifest["roots"].length ||
        admission.manifest["roots"].length > 16 || admission.manifest["roots"].some((root) => typeof root !== "string" || !path.isAbsolute(root))) {
      throw new Error(`CUDA JIT header pack lacks bounded absolute source roots for NVRTC ${version}`);
    }
    const files = producerInputMap(admission.files, `NVRTC ${version} header pack`);
    verifyInputFiles(files, `NVRTC ${version} header pack`);
    const roots = checkedRoots(admission.manifest["roots"], profileFor("pearl-rank128"));
    const live = liveHeaderSnapshot(roots);
    if (live.sha256 !== admission.manifest["sha256"] || live.count !== admission.manifest["count"] || live.bytes !== admission.manifest["bytes"]) {
      throw new Error(`NVRTC ${version} include roots no longer match the admitted header pack`);
    }
    admittedHeaders[version] = {path: headerPath, manifest: admission.manifest, files: Object.fromEntries(files)};
    headerInputFiles.set(version, files);
    headerRoots.set(version, roots);
  }
  const seen = new Set();
  /** @type {Array<{job: ImageJob, metadata: Record<string, unknown>, data: Buffer}>} */
  const checked = [];
  if (artifacts.length !== plan.length) {throw new Error("CUDA image outputs do not cover the admitted plan");}
  for (const artifact of artifacts) {
    if (!record(artifact) || !record(artifact.job)) {throw new Error("Invalid CUDA image artifact");}
    const key = jobKey(artifact.job), job = planned.get(key);
    if (!job || seen.has(key)) {throw new Error("Unexpected or duplicate CUDA image output");}
    const data = imageBytes(artifact.data), header = admittedHeaders[versionKey(job.compiler)];
    if (job.profile.startsWith("pearl-") && !header) {throw new Error("Missing admitted header roots for Pearl compiler output");}
    const includeRoots = job.profile.startsWith("pearl-") ? checkedRoots(header?.manifest["roots"], profileFor(job.profile)) : [];
    verifyArtifact(job, artifact.metadata, data, includeRoots);
    seen.add(key); checked.push({job, metadata: /** @type {Record<string, unknown>} */ (artifact.metadata), data});
  }
  if (seen.size !== plan.length) {throw new Error("CUDA image outputs are incomplete");}
  const baseline = compilerInfoByVersion.get("12.6");
  if (!baseline || !baseline.supportedArchs.includes(80)) {
    throw new Error("Release CUDA images require an actual NVRTC 12.6 query supporting compute_80");
  }
  for (const arch of baseline.supportedArchs.filter((value) => value >= 80)) {
    for (const profile of genericProfiles) {
      const exact = checked.find(({job}) => job.profile === profile.name && job.target === `sm_${arch}` && job.format === "cubin");
      if (!exact || versionKey(exact.job.compiler) !== "12.6") {
        throw new Error(`NVRTC 12.6 must own the exact ${profile.name} sm_${arch} image`);
      }
    }
  }
  const primaryPtx = new Map(checked.filter(({job}) => job.format === "ptx" && job.role === "primary")
    .map((item) => [`${item.job.profile}|${item.job.target}`, item]));
  if (primaryPtx.size !== plan.filter((job) => job.format === "ptx" && job.role === "primary").length) {
    throw new Error("Missing primary PTX image for a queried compute target");
  }
  for (const profile of genericProfiles) {
    const legacy = checked.find(({job}) => job.profile === profile.name && job.target === "compute_80" &&
      job.format === "ptx" && versionKey(job.compiler) === "12.6");
    if (!legacy || legacy.metadata["ptxVersion"] !== "8.5") {
      throw new Error(`Release CUDA images require an actual NVRTC 12.6 PTX 8.5 baseline for ${profile.name}`);
    }
  }
  /** @type {Array<{job: ImageJob, metadata: Record<string, unknown>, data: Buffer, profile: Profile,
   * compatibility: boolean, include: boolean, resource: string | null, sha256: string, arch: number}>} */
  const embedded = [];
  for (const item of checked) {
    let compatibility = false, include = true;
    if (item.job.format === "ptx" && item.job.role === "compatibility-candidate") {
      const primary = primaryPtx.get(`${item.job.profile}|${item.job.target}`);
      if (!primary) {throw new Error("Compatibility PTX has no primary image");}
      const comparison = compareIsa(String(item.metadata["ptxVersion"]), String(primary.metadata["ptxVersion"]));
      if (comparison > 0) {throw new Error("Older NVRTC emitted a newer PTX ISA than the primary compiler");}
      // Keep the real 12.6 PTX even if a newer compiler happens to emit the same ISA. It is
      // the admitted compatibility artifact, and its resource identity still records compiler.
      compatibility = comparison < 0 || versionKey(item.job.compiler) === "12.6";
      include = compatibility;
    }
    const profile = profileFor(item.job.profile), name = resourceName(item.job, compatibility);
    embedded.push({...item, profile, compatibility, include, resource: include ? name : null,
      sha256: digest(item.data), arch: architectureOf(item.job.target)});
  }
  for (const profile of genericProfiles) {
    const legacy = embedded.find(({job}) => job.profile === profile.name && job.target === "compute_80" &&
      job.format === "ptx" && versionKey(job.compiler) === "12.6");
    if (!legacy?.include) {throw new Error(`NVRTC 12.6 compatibility PTX is not embedded for ${profile.name}`);}
  }
  /** @type {Array<{name: string, data: Buffer}>} */
  const resources = embedded.filter((item) => item.include).map((item) => ({
    name: /** @type {string} */ (item.resource), data: item.data}));
  const resourceBytes = resources.reduce((sum, item) => sum + item.data.length, 0);
  if (resourceBytes > maxImageBytes) {throw new Error("Aggregate embedded CUDA images exceed 64 MiB");}
  const rows = embedded.filter((item) => item.include).sort((a, b) => profileRank(a.job.profile) - profileRank(b.job.profile) ||
    a.arch - b.arch || Number(b.job.format === "cubin") - Number(a.job.format === "cubin") || Number(b.compatibility) - Number(a.compatibility));
  const registry = cudaResourceMap(resources) + "\nstatic constexpr mom::cuda_images::CudaImageRow kCudaImageRows[] = {\n" +
    rows.map((item) => `  {${JSON.stringify(item.job.profile)}, ${item.arch}, ${item.job.format === "cubin" ? "true" : "false"}, ${item.compatibility ? "true" : "false"}, ${JSON.stringify(item.resource)}},`).join("\n") +
    "\n};\n// Maintained source inputs SHA-256: " + JSON.stringify(beforeSources) + "\n";
  const imageRecords = embedded.map((item) => ({profile: item.job.profile, source: item.profile.source, target: item.job.target,
    arch: item.arch, role: item.job.role, format: item.job.format, compatibility: item.compatibility,
    embedded: item.include, resource: item.resource, compiler: {major: item.job.compiler.major, minor: item.job.compiler.minor},
    includeRoots: item.metadata["includeRoots"], ptxVersion: item.metadata["ptxVersion"],
    options: item.metadata["options"], exports: item.metadata["exports"],
    bytes: item.data.length, sha256: item.sha256}));
  const compilerRecords = compilers.map((query) => {
    const info = compilerInfo(query), inputs = compilerInputs.get(versionKey(info));
    if (!inputs) {throw new Error("NVRTC compiler provenance was lost during image generation");}
    return sortCompilerRecord(info, inputs);
  }).sort(compareCompiler);
  const manifest = {format: "mom-cuda-images-v1", sourceSha256: digest(registry), sources: beforeSources,
    compilers: compilerRecords,
    headerPacks: admittedHeaders, resourceBytes, images: imageRecords};
  verifyInputFiles(new Map(Object.entries(beforeSources).map(([relative, hash]) => [path.join(repo, relative), hash])), "Maintained CUDA source");
  for (const [version, inputs] of compilerInputs) {verifyInputFiles(inputs, `NVRTC ${version} compiler input`);}
  for (const [version, inputs] of headerInputFiles) {verifyInputFiles(inputs, `NVRTC ${version} header pack`);}
  for (const [version, roots] of headerRoots) {
    const live = liveHeaderSnapshot(roots), expected = admittedHeaders[version]?.manifest;
    if (!expected || live.sha256 !== expected["sha256"] || live.count !== expected["count"] || live.bytes !== expected["bytes"]) {
      throw new Error(`NVRTC ${version} include roots changed during CUDA image generation`);
    }
  }
  if (JSON.stringify(sourceSnapshot(repo)) !== JSON.stringify(beforeSources)) {throw new Error("Maintained CUDA sources changed during image generation");}
  return {source: registry, manifest};
}

/** @param {unknown} job */
function jobKey(job) {
  if (!record(job) || typeof job["profile"] !== "string" || typeof job["target"] !== "string" ||
      !["cubin", "ptx"].includes(/** @type {string} */ (job["format"])) ||
      !["exact", "primary", "compatibility-candidate"].includes(/** @type {string} */ (job["role"])) || !record(job["compiler"]) ||
      typeof job["compiler"]["major"] !== "number" || typeof job["compiler"]["minor"] !== "number") {
    throw new Error("Invalid planned CUDA image job");
  }
  return [job["profile"], job["target"], job["format"], job["role"], job["compiler"]["major"], job["compiler"]["minor"]].join("|");
}

/** @param {ImageJob} job @param {boolean} compatibility */
function resourceName(job, compatibility) {
  const role = compatibility ? "compatibility" : job.role === "primary" ? "primary" : "exact";
  return `${job.profile}-${job.target}-${job.format}-${role}-nvrtc-${versionKey(job.compiler)}`;
}

/** Validate compiler records without reading producer paths, as required for cross-host use.
 * @param {unknown} value @returns {NvrtcInfo & {inputs: Record<string, string>}} */
function admittedCompiler(value) {
  const info = compilerInfo(value);
  if (!record(value)) {throw new Error("Invalid CUDA image compiler provenance");}
  const inputs = admittedPathHashes(value["inputs"]);
  return {...info, inputs};
}

/** @param {unknown} value @returns {HeaderPack} */
function admittedHeaderPack(value) {
  if (!record(value) || typeof value["path"] !== "string" || !isAbsoluteProducerPath(value["path"]) || !record(value["manifest"])) {
    throw new Error("Invalid CUDA image header-pack provenance");
  }
  const files = admittedPathHashes(value["files"]), manifest = value["manifest"];
  if (files[value["path"]] !== manifest["sourceSha256"] || files[value["path"] + ".json"] === undefined ||
      !Array.isArray(manifest["requiredHeaders"]) || manifest["requiredHeaders"].join("\n") !== "cuda_runtime.h\ncute/tensor.hpp\ncuda/std/cstdint" ||
      !Number.isSafeInteger(manifest["count"]) || typeof manifest["count"] !== "number" || manifest["count"] < 3 || manifest["count"] > 16384 ||
      !Number.isSafeInteger(manifest["bytes"]) || typeof manifest["bytes"] !== "number" || manifest["bytes"] < 12 || manifest["bytes"] > maxImageBytes ||
      !Number.isSafeInteger(manifest["compressedBytes"]) || typeof manifest["compressedBytes"] !== "number" || manifest["compressedBytes"] < 1 || manifest["compressedBytes"] > maxImageBytes ||
      typeof manifest["sha256"] !== "string" || !/^[0-9a-f]{64}$/.test(manifest["sha256"]) ||
      !Array.isArray(manifest["roots"]) || !manifest["roots"].length || manifest["roots"].length > 16 ||
      manifest["roots"].some((root) => typeof root !== "string" || !isAbsoluteProducerPath(root))) {
    throw new Error("Invalid CUDA image header-pack manifest");
  }
  return {path: /** @type {string} */ (value["path"]), manifest, files};
}

/** @param {unknown} value @param {string} repo @returns {value is Record<string, unknown>} */
function validSources(value, repo) {
  if (!record(value) || Object.keys(value).length !== sourceFiles.length || sourceFiles.some((file) =>
    typeof value[file] !== "string" || !/^[0-9a-f]{64}$/.test(/** @type {string} */ (value[file])) ||
    fileHash(path.join(repo, file)) !== value[file])) {return false;}
  return true;
}

/** Validate a complete sidecar without opening any producer compiler, SDK, or header path.
 * @param {unknown} value @param {string} repo */
function validateManifest(value, repo) {
  if (!record(value)) {throw new Error("Incomplete or changed CUDA image bundle; regenerate it with native-cuda-images.js");}
  const sourceSha = value["sourceSha256"], sourceHashes = value["sources"], resourceBytes = value["resourceBytes"];
  const rawCompilers = value["compilers"], rawImages = value["images"], rawHeaderPacks = value["headerPacks"];
  if (value["format"] !== "mom-cuda-images-v1" || !/^[0-9a-f]{64}$/.test(String(sourceSha)) ||
      !validSources(sourceHashes, repo) || typeof resourceBytes !== "number" || !Number.isSafeInteger(resourceBytes) || resourceBytes < 1 ||
      resourceBytes > maxImageBytes || !Array.isArray(rawCompilers) || !rawCompilers.length || rawCompilers.length > 16 ||
      !Array.isArray(rawImages) || !rawImages.length || rawImages.length > maxImages || !record(rawHeaderPacks)) {
    throw new Error("Incomplete or changed CUDA image bundle; regenerate it with native-cuda-images.js");
  }
  const compilers = rawCompilers.map(admittedCompiler).sort(compareCompiler);
  if (new Set(compilers.map(versionKey)).size !== compilers.length) {throw new Error("Duplicate CUDA image compiler provenance");}
  const compilerByVersion = new Map(compilers.map((compiler) => [versionKey(compiler), compiler]));
  const expectedPlan = planCudaImages(compilers), expectedKeys = new Set(expectedPlan.map(jobKey));
  const headerPacks = new Map(Object.entries(rawHeaderPacks).map(([version, pack]) => [version, admittedHeaderPack(pack)]));
  const usedPearlVersions = new Set();
  const resources = new Set(), keys = new Set();
  let total = 0;
  for (const image of rawImages) {
    if (!record(image) || typeof image["profile"] !== "string" || typeof image["target"] !== "string" ||
        !["cubin", "ptx"].includes(/** @type {string} */ (image["format"])) ||
        !["exact", "primary", "compatibility-candidate"].includes(/** @type {string} */ (image["role"])) ||
        !record(image["compiler"]) || typeof image["compiler"]["major"] !== "number" || typeof image["compiler"]["minor"] !== "number") {
      throw new Error("Invalid CUDA image sidecar entry");
    }
    const profileName = image["profile"], target = image["target"], format = image["format"], roleName = image["role"];
    const imageCompiler = image["compiler"], major = imageCompiler["major"], minor = imageCompiler["minor"];
    const profile = profileFor(profileName), compiler = compilerByVersion.get(`${major}.${minor}`);
    const key = [profileName, target, format, roleName, major, minor].join("|");
    if (!compiler || keys.has(key) || image["source"] !== profile.source || typeof image["arch"] !== "number" ||
        !Number.isSafeInteger(image["arch"]) || image["arch"] !== architectureOf(target) ||
        typeof image["bytes"] !== "number" || !Number.isSafeInteger(image["bytes"]) || image["bytes"] < 1 || image["bytes"] > maxImageBytes ||
        typeof image["sha256"] !== "string" || !/^[0-9a-f]{64}$/.test(image["sha256"]) ||
        typeof image["compatibility"] !== "boolean" || image["embedded"] !== true ||
        image["compatibility"] !== (roleName === "compatibility-candidate") ||
        typeof image["resource"] !== "string") {
      throw new Error("Invalid CUDA image sidecar entry");
    }
    const arch = image["arch"], isPtx = format === "ptx", isCompact = profile.compact === true;
    if (isPtx !== target.startsWith("compute_") || (isPtx && target !== `compute_${arch}`) ||
        (!isPtx && target !== `sm_${arch}` && target !== "sm_120a") ||
        (target === "sm_120a" && (!isCompact || arch !== 120 || compiler.major < 13 ||
          (compiler.major === 13 && compiler.minor < 3))) ||
        (isCompact && target !== "sm_120a") || (roleName === "exact") !== !isPtx || !compiler.supportedArchs.includes(arch)) {
      throw new Error("CUDA image target or compiler query is inconsistent");
    }
    const includeRoots = checkedRoots(image["includeRoots"], profile);
    if (profile.rank && JSON.stringify(includeRoots) !== JSON.stringify(headerPacks.get(versionKey(compiler))?.manifest["roots"])) {
      throw new Error("CUDA image include roots do not match their compiler header pack");
    }
    const ptxVersion = image["ptxVersion"];
    if (JSON.stringify(image["options"]) !== JSON.stringify(compileOptions(profile, target, /** @type {"cubin" | "ptx"} */ (format), includeRoots)) ||
        !Array.isArray(image["exports"]) || JSON.stringify(image["exports"]) !== JSON.stringify(profile.exports) ||
        (isPtx ? (typeof ptxVersion !== "string" || !/^\d{1,3}\.\d{1,3}$/.test(ptxVersion)) : ptxVersion !== null) ||
        (major === 12 && minor === 6 && isPtx && ptxVersion !== "8.5")) {
      throw new Error("CUDA image sidecar options or actual PTX ISA are inconsistent");
    }
    if (profile.rank) {usedPearlVersions.add(versionKey(compiler));}
    const compatibility = image["compatibility"], embedded = image["embedded"];
    const resource = image["resource"], resourceRole = compatibility ? "compatibility" : roleName === "primary" ? "primary" : "exact";
    if (embedded) {
      const expectedResource = `${profileName}-${target}-${format}-${resourceRole}-nvrtc-${versionKey(compiler)}`;
      if (resource !== expectedResource || resources.has(expectedResource)) {throw new Error("CUDA image resource identity is invalid or duplicated");}
      resources.add(expectedResource); total += image["bytes"];
    }
    keys.add(key);
  }
  if (keys.size !== expectedKeys.size || [...expectedKeys].some((key) => !keys.has(key)) || total !== resourceBytes ||
      total > maxImageBytes || !hasExactBuildFields(rawHeaderPacks, [...usedPearlVersions])) {
    throw new Error("CUDA image sidecar resource or header-pack inventory is inconsistent");
  }
  const baseline = compilerByVersion.get("12.6");
  if (!baseline || !baseline.supportedArchs.includes(80)) {
    throw new Error("Release CUDA images require NVRTC 12.6 support for compute_80");
  }
  for (const profile of genericProfiles) {
    const images = rawImages.filter((image) => record(image) && image["profile"] === profile.name);
    const baselinePtx = images.find((image) => record(image) && image["target"] === "compute_80" &&
      record(image["compiler"]) && image["compiler"]["major"] === 12 && image["compiler"]["minor"] === 6 &&
      image["ptxVersion"] === "8.5" && image["embedded"] === true);
    const primary = images.find((image) => record(image) && image["target"] === "compute_80" && image["role"] === "primary");
    if (!baselinePtx || !primary) {throw new Error(`CUDA image release is missing its 12.6 PTX 8.5 baseline for ${profile.name}`);}
    for (const arch of baseline.supportedArchs.filter((value) => value >= 80)) {
      if (!images.some((image) => record(image) && image["target"] === `sm_${arch}` && image["format"] === "cubin" &&
          record(image["compiler"]) && image["compiler"]["major"] === 12 && image["compiler"]["minor"] === 6)) {
        throw new Error(`CUDA image release is missing its NVRTC 12.6 sm_${arch} exact image for ${profile.name}`);
      }
    }
  }
  return true;
}

/**
 * Bind a generated include to its bounded sidecar before it enters an existing build receipt.
 * Producer SDK paths in the sidecar are syntax-checked only; this lets an image bundle move to
 * a different host while the maintained source hashes still have to match that consumer tree.
 * @param {string} file @param {string} [repo]
 */
function cudaImageInputs(file, repo = path.resolve(__dirname, "..")) {
  if (!path.isAbsolute(file) || path.extname(file) !== ".inc" || !path.isAbsolute(repo)) {
    throw new Error("CUDA image include path must be an absolute .inc file");
  }
  const metadata = file + ".json";
  if (!fs.lstatSync(file).isFile() || !fs.lstatSync(metadata).isFile()) {throw new Error("CUDA image include and sidecar must be regular files");}
  const incStat = fs.statSync(file), sidecarStat = fs.statSync(metadata);
  if (incStat.size < 1 || incStat.size > maxIncBytes || sidecarStat.size < 1 || sidecarStat.size > (8 << 20)) {
    throw new Error("CUDA image include or sidecar exceeds bounds");
  }
  const incSha = fileHash(file), {value: manifest, sha256: sidecarSha} = readBuildJsonSnapshot(metadata, 8 << 20);
  validateManifest(manifest, repo);
  if (fileHash(file) !== incSha || fileHash(metadata) !== sidecarSha || manifest["sourceSha256"] !== incSha) {
    throw new Error("Incomplete or changed CUDA image bundle; regenerate it with native-cuda-images.js");
  }
  const manifestRecord = /** @type {Record<string, unknown>} */ (manifest);
  return {files: {[file]: incSha, [metadata]: sidecarSha}, manifest: manifestRecord};
}

/** Return the normal cache command record for the one generated native CUDA registry TU.
 * @param {{repo: string, output: string, compiler: string, flags: string[], windows?: boolean}} input
 * @returns {import("./native-object-cache").NativeCommand} */
function cudaImageCompileCommand({repo, output, compiler, flags, windows = false}) {
  if (!path.isAbsolute(repo) || !path.isAbsolute(output) || typeof compiler !== "string" || !compiler ||
      !Array.isArray(flags) || flags.length > 256 || flags.some((flag) => typeof flag !== "string" || /[\0\r\n]/.test(flag)) ||
      typeof windows !== "boolean") {throw new Error("Invalid CUDA image registry compile command");}
  const source = path.join(repo, "sycl/cuda-images.cpp"), object = path.join(output, windows ? "cuda-images.obj" : "cuda-images.o");
  const depfile = path.join(output, "cuda-images.d");
  return {command: compiler, args: [...flags, "-MD", ...(windows ? ["-MV"] : []), "-MF", depfile, "-I", output,
    "-c", source, "-o", object], source, object, depfile, generatedInputs: [path.join(output, "cuda-images.inc")],
  ...(windows ? {dependencyFormat: "nmake"} : {}), timeout: 300000};
}

module.exports = {planCudaImages, generateCudaImageBundle, cudaImageInputs, cudaImageCompileCommand};
