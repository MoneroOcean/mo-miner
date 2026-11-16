"use strict";

// Build-only executable compaction, shared by source-built Linux and Windows releases.
// UPX is a pinned build input, not a shipped dependency. Never run the resulting miner.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {fileHash, sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {auditNativePePayload, verifyNativePeMap} = require("./native-pe-audit");

const packingStages = ["pack", "integrity", "restore"];

/** @param {string} output */
function packingLogReceipts(output) {
  return Object.fromEntries(packingStages.map(stage => {
    const file = path.join(output, stage + ".log"), stat = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink()) {throw new Error("Missing native packing stage log: " + stage);}
    return [stage, {path: file, sha256: fileHash(file)}];
  }));
}

/** @param {string} value */
const absolute = value => typeof value === "string" && path.isAbsolute(value) && value.length <= 4096 && !/[\0\r\n]/.test(value);
/** @typedef {{source: string, output: string, upx: string, upxSha256: string}} PackOptions */
/** Validate all paths and content inputs before creating output; retain any failed stage for
 * diagnosis, never retry a weaker/forced packing mode or overwrite an existing artifact.
 * @param {PackOptions} options @param {typeof spawnSync} [spawn] */
function packNativeRelease(options, spawn = spawnSync) {
  const {source, output, upx, upxSha256} = options;
  if (![source, output, upx].every(absolute) || !/^[0-9a-f]{64}$/.test(upxSha256) || fs.existsSync(output)) {throw new Error("Native packing requires explicit absolute inputs and a new output directory");}
  if (fileHash(upx) !== upxSha256) {throw new Error("Native packing tool hash differs from the selected build input");}
  const sourceSha256 = fileHash(source), sourceBytes = fs.statSync(source).size, header = Buffer.alloc(4096), fd = fs.openSync(source, "r");
  try {if (fs.readSync(fd, header, 0, header.length, 0) < 4) {throw new Error("Truncated native packing input");}} finally {fs.closeSync(fd);}
  const windows = header.readUInt16LE() === 0x5a4d;
  if (!windows && header.subarray(0, 4).toString("hex") !== "7f454c46") {throw new Error("Unsupported native executable packing input");}
  if (windows) {
    verifyNativePeMap(source);
    // UPX5.2.1's 768-MiB working-format bound is packing metadata, not image semantics:
    // optional IAT normalization may change this sum without changing any import binding.
    const optional = header.readUInt32LE(0x3c) + 24;
    const upxWorkspaceBytes = header.readUInt32LE(optional + 56) + Math.max(header.readUInt32LE(optional + 8), 65536) +
      [1, 11, 12, 13, 5].reduce((sum, index) => sum + header.readUInt32LE(optional + 112 + index * 8 + 4), 0);
    if (upxWorkspaceBytes > 0x30000000) {
      throw new Error(`Windows UPX workspace ${upxWorkspaceBytes} exceeds 805306368 bytes; reduce embedded resources (not a host OOM)`);
    }
  }
  const repo = path.resolve(__dirname, ".."), recipe = sourceInputs(repo, [], ["scripts/pack-native-release.js", "scripts/native-pe-audit.js", "scripts/native-pe-pages.js", "scripts/native-elf.js", "scripts/build-source-inputs.js"]);
  fs.mkdirSync(output, {mode: 0o700});
  const binary = path.join(output, windows ? "mom.exe" : "mom"), restored = path.join(output, windows ? "restored.exe" : "restored");
  const commands = /** @type {{command: string, args: string[]}[]} */ ([]);
  /** @param {string[]} args @param {string} stage @param {number} timeout */
  const run = (args, stage, timeout) => {
    args = ["--no-env", ...args]; // Ambient UPX options must not change any recorded stage.
    const log = fs.openSync(path.join(output, stage + ".log"), "wx"); commands.push({command: upx, args});
    try {
      const result = spawn(upx, args, {stdio: ["ignore", log, log], timeout, windowsHide: true});
      if (result.status !== 0 || result.signal || result.error) {
        // Preserve the full file, but expose only a bounded tail in ordinary CI logs.
        const reader = fs.openSync(path.join(output, stage + ".log"), "r");
        let tail;
        try {
          const size = fs.fstatSync(reader).size, buffer = Buffer.alloc(Math.min(size, 16384));
          const bytes = fs.readSync(reader, buffer, 0, buffer.length, Math.max(0, size - buffer.length));
          tail = buffer.subarray(0, bytes).toString("utf8");
        } finally {fs.closeSync(reader);}
        throw new Error(`Native packing stage failed: ${stage} (exit ${result.status}, signal ${result.signal || "none"}); inspect its saved log\n${tail}`, {cause: result.error});
      }
    } finally {fs.closeSync(log);}
  };
  console.log("[native] Packing in memory; raw executable is preserved");
  // Pinned UPX 5.2.1's level 10 otherwise grows the dictionary to the entire block,
  // exhausting 5 GiB on our PE. Keep maximum LZMA effort with a fixed 128-MiB dictionary;
  // this CRP option is version-specific and must pass the real-tool round-trip gate.
  run(["--best", "--lzma", "--crp-lzma-ds=134217728", ...(windows ? ["--strip-relocs=0", "--compress-exports=0"] : ["--exact"]), "--no-progress", "-o", binary, source], "pack", 20 * 60 * 1000);
  run(["-t", binary], "integrity", 60000);
  run(["-d", "-o", restored, binary], "restore", 120000);
  const restoredSha256 = fileHash(restored);
  if (!windows && restoredSha256 !== sourceSha256) {throw new Error("Native ELF packing did not restore exact original bytes");}
  const payloadAudit = windows ? auditNativePePayload(source, restored) : {exactOriginalBytes: true};
  const logs = packingLogReceipts(output);
  if (fileHash(source) !== sourceSha256 || fileHash(upx) !== upxSha256) {throw new Error("Native packing inputs changed during compaction");}
  verifySourceInputs(recipe, sourceInputs(repo, [], Object.keys(recipe.graph)), "Native packing recipe");
  fs.chmodSync(binary, 0o755);
  const result = {source: {path: source, bytes: sourceBytes, sha256: sourceSha256}, artifact: {path: binary, bytes: fs.statSync(binary).size, sha256: fileHash(binary)},
    restoredSha256, tool: {path: upx, sha256: upxSha256}, payloadAudit, logs, recipe, commands, buildOnly: true, publicationQualified: false,
    scope: "In-memory executable packing, integrity and offline payload/loader-binding proof; no miner execution or hardware/performance/publication qualification"};
  // The raw source and verification hashes remain; a second full-size raw copy adds no
  // evidence after admission. Keep rejected round-trips above for failure diagnosis.
  fs.unlinkSync(restored);
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  for (const stage of packingStages) {
    const log = logs[stage];
    if (!log || fileHash(log.path) !== log.sha256) {throw new Error("Native packing log changed while writing receipt: " + stage);}
  }
  if (fileHash(source) !== sourceSha256 || fileHash(upx) !== upxSha256) {throw new Error("Native packing inputs changed while writing receipt");}
  return result;
}

if (require.main === module) {
  try {
    const [source, output, upx, upxSha256, extra] = process.argv.slice(2);
    if (!source || !output || !upx || !upxSha256 || extra) {throw new Error("Usage: pack-native-release.js /source/executable /new/output /pinned/upx <sha256>");}
    console.log(JSON.stringify(packNativeRelease({source, output, upx, upxSha256}).artifact));
  } catch (error) {console.error(error instanceof Error ? error.message : error); process.exitCode = 1;}
}

module.exports = {packNativeRelease};
