"use strict";

// Build-only virtual includes. Keep filesystem discovery out of the release executable.
const fs = require("node:fs"), path = require("node:path"), crypto = require("node:crypto");
const {gzipSync} = require("node:zlib");
const nativeResources = require("./native-resources");
const {fileHash, readBuildJson} = require("./build-source-inputs");
const maxBytes = 64 << 20, maxFiles = 16384;
const cudaRequiredHeaders = ["cuda_runtime.h", "cute/tensor.hpp", "cuda/std/cstdint"];
/** @param {unknown} name */
function validName(name) {
  return typeof name === "string" && /^[A-Za-z0-9_][A-Za-z0-9_./+-]{0,254}$/.test(name) &&
    name.split("/").every((part) => part && part !== "." && part !== "..");
}

/** Canonical length-delimited UTF-8/byte payload; NUL terminators serve NVRTC, not framing.
 * @param {Array<{name: string, data: Buffer}>} entries */
function packHeaders(entries) {
  if (!entries.length || entries.length > maxFiles) {throw new Error("Invalid JIT header count");}
  if (entries.some(({name}) => !validName(name))) {throw new Error("Invalid JIT header name");}
  const ordered = [...entries].sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  /** @type {Buffer[]} */
  const chunks = [Buffer.from("MOMHDR01"), Buffer.alloc(4)];
  chunks[1]?.writeUInt32LE(ordered.length);
  let total = 12, previous = "";
  for (const {name, data} of ordered) {
    if (!validName(name) || name === previous) {throw new Error("Invalid or duplicate JIT header name");}
    if (!Buffer.isBuffer(data) || data.includes(0)) {throw new Error("JIT headers require NUL-free byte buffers");}
    const size = 8 + name.length + 1 + data.length + 1;
    if (size > maxBytes - total) {throw new Error("JIT headers exceed size bounds");}
    const lengths = Buffer.alloc(8); lengths.writeUInt32LE(name.length); lengths.writeUInt32LE(data.length, 4);
    chunks.push(lengths, Buffer.from(name + "\0"), data, Buffer.from([0]));
    total += size; previous = name;
  }
  return Buffer.concat(chunks, total);
}

/** Enumerate only explicitly selected SDK include roots; nested links may not escape or cycle.
 * Identical overlapping includes are harmless, conflicting definitions fail rather than shadow.
 * @param {string[]} roots @param {boolean} [canonicalRoots] */
function collectHeaders(roots, canonicalRoots = false) {
  if (!roots.length || roots.length > 8) {throw new Error("Provide bounded JIT include roots");}
  const entries = new Map(), selected = new Set(roots.map(root => fs.realpathSync(root)));
  let total = 12;
  /** @param {string} root @param {string} directory @param {string} prefix @param {Set<string>} parents */
  function visit(root, directory, prefix, parents) {
    const actual = fs.realpathSync(directory), relative = path.relative(root, actual);
    if (relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative) || parents.has(actual) || parents.size > 32) {
      throw new Error("JIT header link escapes its root or cycles");
    }
    const ancestors = new Set(parents).add(actual);
    for (const entry of fs.readdirSync(actual, {withFileTypes: true})) {
      const file = path.join(actual, entry.name), name = prefix + entry.name;
      if (!validName(name)) {throw new Error("Invalid JIT header name");}
      const resolved = fs.realpathSync(file), rel = path.relative(root, resolved);
      if (rel === ".." || rel.startsWith(".." + path.sep) || path.isAbsolute(rel)) {throw new Error("JIT header link escapes its root");}
      const stat = fs.statSync(resolved);
      if (stat.isDirectory()) {
        // CUDA roots overlap at include/cccl. Collect that explicit subroot once under its
        // canonical namespace, after the usual containment check; never hide a self-cycle.
        if (canonicalRoots && resolved !== root && selected.has(resolved)) {continue;}
        visit(root, resolved, name + "/", ancestors); continue;
      }
      if (!stat.isFile() || stat.size > maxBytes - total) {throw new Error("JIT header input exceeds bounds or is not a file");}
      const data = fs.readFileSync(resolved), prior = entries.get(name);
      if (prior) {if (!prior.equals(data)) {throw new Error("Conflicting JIT include: " + name);} continue;}
      total += 10 + name.length + data.length;
      if (total > maxBytes || entries.size >= maxFiles) {throw new Error("JIT headers exceed size/count bounds");}
      entries.set(name, data);
    }
  }
  for (const input of roots) {
    const root = fs.realpathSync(input);
    visit(root, root, "", new Set());
  }
  return [...entries].map(([name, data]) => ({name, data}));
}

/** @param {Array<{name: string, data: Buffer}>} entries */
function headerResources(entries) {
  const data = packHeaders(entries), compressed = gzipSync(data, {level: 9});
  const sha256 = crypto.createHash("sha256").update(data).digest("hex");
  const source = nativeResources([{name: "jit-headers.gz", data: compressed}]) +
    `static constexpr std::size_t kJitHeadersSize = ${data.length};\n` +
    `static constexpr char kJitHeadersSha256[] = "${sha256}";\n`;
  return {source, sourceSha256: crypto.createHash("sha256").update(source).digest("hex"),
    count: entries.length, bytes: data.length, compressedBytes: compressed.length, sha256};
}

/** CUDA 13 archives nest CCCL below include/cccl. Keep runtime headers, expose cuda/std
 * directly, and omit redundant cccl/* aliases. Incomplete packs must fail at SDK creation,
 * not silently select a slower Pearl kernel at runtime. @param {string[]} inputs */
function cudaHeaderResources(inputs) {
  if (!inputs.length || inputs.length > 8) {throw new Error("Provide bounded JIT include roots");}
  const roots = new Set(inputs.map(root => fs.realpathSync(root)));
  for (const root of [...roots]) {
    const nested = path.join(root, "cccl");
    if (!fs.existsSync(path.join(nested, "cuda/std/cstdint"))) {continue;}
    const actual = fs.realpathSync(nested), relative = path.relative(root, actual);
    if (relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative)) {throw new Error("JIT header link escapes its root");}
    roots.add(actual);
  }
  const entries = collectHeaders([...roots], true);
  for (const name of cudaRequiredHeaders) {
    if (!entries.some(entry => entry.name === name && entry.data.length)) {throw new Error("Missing required CUDA JIT header: " + name);}
  }
  return {...headerResources(entries), roots: [...roots], requiredHeaders: [...cudaRequiredHeaders]};
}

/** Prebuilt packs are trusted C++ SDK inputs, but their existing sidecar must bind the
 * exact source to a complete inventory from the producer. No new runtime parser or SDK
 * search path is needed. Freeze both inputs so a changed manifest cannot admit stale data.
 * @param {string} file */
function cudaHeaderInputs(file) {
  if (!path.isAbsolute(file)) {throw new Error("CUDA JIT header pack must be absolute");}
  if (!fs.statSync(file).isFile()) {throw new Error("CUDA JIT header pack is not a regular file");}
  const metadata = file + ".json", metadataSha256 = fileHash(metadata), manifest = readBuildJson(metadata), sha256 = fileHash(file);
  if (!manifest || !Array.isArray(manifest.requiredHeaders) || manifest.requiredHeaders.length !== cudaRequiredHeaders.length ||
      !cudaRequiredHeaders.every(name => manifest.requiredHeaders.includes(name)) || manifest.sourceSha256 !== sha256 ||
      !Number.isSafeInteger(manifest.count) || manifest.count < 3 || manifest.count > maxFiles ||
      !Number.isSafeInteger(manifest.bytes) || manifest.bytes < 12 || manifest.bytes > maxBytes ||
      !Number.isSafeInteger(manifest.compressedBytes) || manifest.compressedBytes < 1 || manifest.compressedBytes > maxBytes ||
      typeof manifest.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(manifest.sha256) || fileHash(metadata) !== metadataSha256) {
    throw new Error("Incomplete or changed CUDA JIT header pack; regenerate it with native-jit-headers.js");
  }
  return {manifest, files: {[file]: sha256, [metadata]: metadataSha256}};
}

module.exports = {validName, packHeaders, collectHeaders, headerResources, cudaHeaderResources, cudaHeaderInputs};
if (require.main === module) {
  try {
    const [output, ...roots] = process.argv.slice(2);
    if (!output || !roots.length) {throw new Error("Usage: native-jit-headers.js <output.inc> <include-root> [include-root ...]");}
    const {source, ...report} = cudaHeaderResources(roots);
    fs.writeFileSync(output, source);
    fs.writeFileSync(output + ".json", JSON.stringify(report, null, 2) + "\n");
  } catch (error) {console.error(error instanceof Error ? error.message : error); process.exitCode = 1;}
}
