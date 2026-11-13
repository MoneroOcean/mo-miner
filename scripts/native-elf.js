"use strict";

const fs = require("node:fs"), {createHash} = require("node:crypto"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {fileHash} = require("./build-source-inputs");

/** Build tooling only: inspect bounded binutils metadata, never execute the candidate.
 * @param {string} file @param {string} flag */
function elfReport(file, flag) {
  const result = spawnSync("readelf", ["--wide", flag, file], {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
  if (result.error || result.signal || result.status !== 0) {throw new Error("Cannot inspect native ELF: " + flag, {cause: result.error});}
  return result.stdout;
}

/** Hash one finite regular-file range with bounded storage, including non-ALLOC GPU images.
 * @param {string} file @param {number} offset @param {number} size */
function elfRangeHash(file, offset, size) {
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK);
  try {
    const before = fs.fstatSync(fd, {bigint: true});
    if (!before.isFile() || !Number.isSafeInteger(offset) || !Number.isSafeInteger(size) || offset < 0 || size < 0 ||
        BigInt(offset) + BigInt(size) > before.size) {throw new Error("Invalid native ELF payload range");}
    const data = Buffer.allocUnsafe(64 << 10), sum = createHash("sha256");
    for (let done = 0; done < size;) {
      const n = fs.readSync(fd, data, 0, Math.min(data.length, size - done), offset + done);
      if (!n) {throw new Error("Truncated native ELF payload");}
      sum.update(data.subarray(0, n)); done += n;
    }
    const after = fs.fstatSync(fd, {bigint: true});
    if (after.size !== before.size || after.mtimeNs !== before.mtimeNs || after.ctimeNs !== before.ctimeNs) {throw new Error("Native ELF changed during payload audit");}
    return sum.digest("hex");
  } finally {fs.closeSync(fd);}
}

/** Parse only wide ELF64 section metadata; changed binutils syntax fails closed. Section data
 * stays on disk. Symbol tables may disappear, but images, JIT resources and unwind data may not.
 * @param {string} text @param {(offset: number, size: number) => string} hash */
function elfSections(text, hash) {
  if (Buffer.byteLength(text) > (1 << 20)) {throw new Error("Excessive native ELF metadata");}
  const rows = [], names = new Set();
  for (const line of text.split("\n").filter(line => /^\s*\[\s*[1-9][0-9]*\]/.test(line))) {
    const match = line.match(/^\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([a-f0-9]+)\s+([a-f0-9]+)\s+([a-f0-9]+)\s+([a-f0-9]+)\s+([A-Za-z]*)\s+(\d+)\s+(\d+)\s+(\d+)\s*$/i);
    if (!match || rows.length >= 4096) {throw new Error("Unrecognized native ELF section metadata");}
    const [, section = "", type = "", address = "", offset = "", size = "", entry = "", flags = "", link = "", info = "", align = ""] = match;
    if ([address, offset, size, entry].some(value => !Number.isSafeInteger(parseInt(value, 16)))) {throw new Error("Excessive native ELF section range");}
    if (names.has(section)) {throw new Error("Duplicate native ELF section");}
    names.add(section);
    rows.push({section, type, address, offset, size, entry, flags, link, info, align,
      sha256: type === "NOBITS" ? null : hash(parseInt(offset || "", 16), parseInt(size || "", 16))});
  }
  if (!rows.length) {throw new Error("Missing native ELF sections");}
  return rows;
}

/** Snapshot independently of strip; retain raw metadata in the build receipt and reject a
 * changing input. Every application/runtime payload participates, not just ALLOC sections.
 * @param {string} file */
function nativeElf(file) {
  const sha256 = fileHash(file), programHeaders = elfReport(file, "--program-headers");
  const sections = elfSections(elfReport(file, "--sections"), (offset, size) => elfRangeHash(file, offset, size));
  if (fileHash(file) !== sha256) {throw new Error("Native ELF changed during inspection");}
  return {sha256, bytes: fs.statSync(file).size, programHeaders, sections};
}

/** GNU strip moves nonloaded sections/NOBITS and normalizes ELF64 constructor metadata.
 * Admit only those metadata changes; all remaining bytes, addresses and permissions must match.
 * @param {ReturnType<typeof nativeElf>} before @param {ReturnType<typeof nativeElf>} after */
function verifyNativeStrip(before, after) {
  const symbols = [".symtab", ".strtab", ".shstrtab"];
  const canonical = (/** @type {typeof before.sections} */ rows) => rows.filter(row => !symbols.includes(row.section)).map(row => {
    // LLD can inherit PROGBITS from legacy .ctors collected by our .init_array script.
    // BFD strip restores INIT_ARRAY; loader tags and every pointer byte must still match.
    const type = row.section === ".init_array" && row.type === "PROGBITS" ? "INIT_ARRAY" : row.type;
    let entry = row.entry;
    if (["INIT_ARRAY", "FINI_ARRAY"].includes(type || "")) {
      if (!["00", "08"].includes(entry || "") || parseInt(row.size || "", 16) % 8) {throw new Error("Unrecognized native ELF pointer-array metadata");}
      entry = "08";
    }
    return {...row, type, entry, offset: type === "NOBITS" || !row.flags?.includes("A") ? null : row.offset};
  });
  if (before.programHeaders !== after.programHeaders || !isDeepStrictEqual(canonical(before.sections), canonical(after.sections))) {
    throw new Error("Strip changed native runtime/image payloads or loaded layout");
  }
}

module.exports = {elfReport, elfRangeHash, elfSections, nativeElf, verifyNativeStrip};
