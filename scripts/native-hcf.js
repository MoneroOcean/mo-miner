"use strict";

const fs = require("node:fs"), {createHash} = require("node:crypto");

const appendix = Buffer.from("__acpp_hcf_binary_appendix");
const idToken = Buffer.from("object-id=");
const maxObjectBytes = 64 * 1024 ** 2;
const maxHeaderBytes = 4 * 1024 ** 2;
const maxPayloadBytes = 64 * 1024 ** 2;
const maxHcfRecords = 4096;
const maxHcfFiles = 4096;
const maxAggregateBytes = 256 * 1024 ** 2;

/** @param {Buffer} bytes */
function sha256(bytes) {return createHash("sha256").update(bytes).digest("hex");}

/** @typedef {{id: string, objectBytes: number, objectSha256: string, normalizedHeaderSha256: string,
 * deviceIRBytes: number, deviceIRSha256: string}} HcfObjectInfo */
/** @typedef {{markerAt: number, payloadAt: number, payloadEnd: number, info: HcfObjectInfo}} HcfCandidate */

/** Parse one HCF candidate in an ELF object. Header and payload sizes are bounded before parsing.
 * @param {Buffer} bytes @param {number} markerAt @param {string} objectSha256 @param {number} headerFloor
 * @returns {HcfCandidate} */
function parseHcfCandidate(bytes, markerAt, objectSha256, headerFloor) {
  if (!Number.isSafeInteger(markerAt) || markerAt < 0 || markerAt + appendix.length > bytes.length) {
    throw new Error("Invalid HCF appendix offset");
  }
  const lowerBound = Math.max(headerFloor, markerAt - maxHeaderBytes);
  let idAt = -1, searchAt = lowerBound, roots = 0;
  while ((searchAt = bytes.indexOf(idToken, searchAt)) >= 0 && searchAt < markerAt) {
    if (++roots > 128) {throw new Error("Excessive HCF ID candidates");}
    const root = bytes.subarray(searchAt, Math.min(markerAt, searchAt + 128)).toString("latin1");
    if (/^object-id=[^\n]*\ngenerator=hipSYCL SSCP\n/.test(root)) {
      if (idAt >= 0) {throw new Error("Malformed HCF object ID: multiple record headers");}
      idAt = searchAt;
    }
    searchAt += idToken.length;
  }
  if (idAt < 0) {throw new Error("HCF binary appendix has no object ID");}
  const headerLength = markerAt - idAt;
  if (headerLength <= 0 || headerLength > maxHeaderBytes) {throw new Error("Invalid or excessive HCF header");}
  const header = bytes.subarray(idAt, markerAt).toString("latin1");
  if (!header.endsWith("\n") || header.includes("\0")) {throw new Error("Malformed HCF header terminator");}
  /** A bounded HCF tree may contain many metadata lines, but identity/range fields are unique.
   * @param {string} name */
  const field = name => {
    const matcher = new RegExp(`(?:^|\\n)${name}=([^\\n]*)`, "g");
    let count = 0, value = "";
    for (let match; (match = matcher.exec(header));) {
      if (++count > 1) {throw new Error(`HCF header must contain one bounded ${name} field`);}
      value = match[1] || "";
    }
    if (count !== 1) {throw new Error(`HCF header must contain one bounded ${name} field`);}
    return value;
  };
  const id = field("object-id");
  if (!/^[0-9]{1,20}$/.test(id)) {throw new Error("Malformed HCF object ID");}
  const objectId = BigInt(id);
  if (objectId === 0n || objectId > 0xffffffffffffffffn) {throw new Error("Malformed HCF object ID");}
  if (objectId.toString() !== id) {throw new Error("Noncanonical HCF object ID");}
  if (field("generator") !== "hipSYCL SSCP" || !header.startsWith(`object-id=${id}\ngenerator=hipSYCL SSCP\n`)) {
    throw new Error("Unexpected HCF generator");
  }
  const start = field("start"), size = field("size");
  if (!/^[0-9]{1,20}$/.test(start)) {throw new Error("HCF header must contain one bounded start field");}
  if (!/^[0-9]{1,20}$/.test(size)) {throw new Error("HCF header must contain one bounded size field");}
  const payloadOffset = BigInt(start), payloadBytes = BigInt(size);
  if (payloadOffset !== 0n || payloadBytes <= 0n || payloadBytes > BigInt(maxPayloadBytes)) {
    throw new Error("Invalid HCF payload range");
  }
  const payloadAt = markerAt + appendix.length, payloadEnd = payloadAt + Number(payloadBytes);
  if (payloadEnd > bytes.length) {throw new Error("Truncated HCF payload");}
  const deviceIR = bytes.subarray(payloadAt, payloadEnd);
  if (deviceIR.length < 4 || deviceIR.subarray(0, 4).toString("hex") !== "4243c0de") {
    throw new Error("HCF payload is not LLVM bitcode");
  }
  return {markerAt, payloadAt, payloadEnd, info: {
    id, objectBytes: bytes.length, objectSha256,
    normalizedHeaderSha256: sha256(Buffer.from(header.replace(/^object-id=[0-9]+\n/, "object-id=0\n"), "latin1")),
    deviceIRBytes: deviceIR.length, deviceIRSha256: sha256(deviceIR),
  }};
}

/** Parse all top-level HCF records; appendix-like bytes inside bitcode remain payload data.
 * @param {Buffer} bytes @returns {HcfObjectInfo[]} */
function parseHcfObjects(bytes) {
  if (!Buffer.isBuffer(bytes) || bytes.length === 0 || bytes.length > maxObjectBytes) {
    throw new Error("Invalid or excessive HCF object bytes");
  }
  /** @type {HcfObjectInfo[]} */
  const records = [];
  let markerAt = bytes.indexOf(appendix), payloadEnd = 0, objectSha256 = "";
  while (markerAt >= 0) {
    if (markerAt < payloadEnd) {
      markerAt = bytes.indexOf(appendix, markerAt + appendix.length);
      continue;
    }
    if (records.length >= maxHcfRecords) {throw new Error("Excessive HCF record count");}
    if (!objectSha256) {objectSha256 = sha256(bytes);}
    const candidate = parseHcfCandidate(bytes, markerAt, objectSha256, payloadEnd);
    records.push(candidate.info);
    payloadEnd = candidate.payloadEnd;
    markerAt = bytes.indexOf(appendix, markerAt + appendix.length);
  }
  return records;
}

/** Parse exactly one HCF record; appendix-like bytes wholly inside its bitcode are payload data.
 * @param {Buffer} bytes */
function parseHcfObject(bytes) {
  const records = parseHcfObjects(bytes);
  if (!records.length) {throw new Error("Missing or malformed HCF object record");}
  if (records.length !== 1) {throw new Error("HCF object contains multiple records");}
  return records[0] || (() => {throw new Error("Missing or malformed HCF object record");})();
}

/** Read records through a stable, bounded regular-file descriptor. @param {string} file
 * @param {number} maximumBytes @returns {{bytes: number, records: HcfObjectInfo[]}} */
function readHcfObjects(file, maximumBytes) {
  if (!Number.isSafeInteger(maximumBytes) || maximumBytes < 1 || maximumBytes > maxObjectBytes) {
    throw new Error("Invalid HCF object size bound");
  }
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK | (fs.constants.O_NOFOLLOW || 0));
  try {
    const before = fs.fstatSync(fd, {bigint: true});
    if (!before.isFile() || before.size <= 0n || before.size > BigInt(maximumBytes)) {
      throw new Error("HCF object is not a bounded regular file: " + file);
    }
    const bytes = Buffer.alloc(Number(before.size));
    for (let done = 0; done < bytes.length;) {
      const count = fs.readSync(fd, bytes, done, Math.min(64 << 10, bytes.length - done), done);
      if (!count) {throw new Error("Truncated HCF object file: " + file);}
      done += count;
    }
    const after = fs.fstatSync(fd, {bigint: true});
    if (after.size !== before.size || after.mtimeNs !== before.mtimeNs || after.ctimeNs !== before.ctimeNs) {
      throw new Error("HCF object changed during inspection: " + file);
    }
    return {bytes: bytes.length, records: parseHcfObjects(bytes)};
  } finally {fs.closeSync(fd);}
}

/** Inspect all HCF records in one object; a regular object with no appendix returns an empty list.
 * @param {string} file */
function inspectHcfObjects(file) {return readHcfObjects(file, maxObjectBytes).records;}

/** Preserve the existing one-record API for compile-only fixture inspection.
 * @param {string} file */
function inspectHcfObject(file) {
  const records = inspectHcfObjects(file);
  if (!records.length) {throw new Error("Missing or malformed HCF object record");}
  if (records.length !== 1) {throw new Error("HCF object contains multiple records");}
  const record = records[0];
  if (!record) {throw new Error("Missing or malformed HCF object record");}
  return record;
}

/** Reject collisions across a bounded union of final-link inputs; equal payloads still collide.
 * @param {string[]} files @returns {Array<HcfObjectInfo & {file: string}>} */
function assertUniqueHcfIds(files) {
  if (!Array.isArray(files) || files.length === 0 || files.length > maxHcfFiles ||
      files.some(file => typeof file !== "string" || file.length === 0 || file.length > 4096)) {
    throw new Error("Invalid or empty HCF object file list");
  }
  let totalBytes = 0;
  const ids = new Set(), union = [];
  for (const file of files) {
    const remaining = maxAggregateBytes - totalBytes;
    if (remaining <= 0) {throw new Error("Excessive aggregate HCF object bytes");}
    const inspected = readHcfObjects(file, Math.min(maxObjectBytes, remaining));
    totalBytes += inspected.bytes;
    for (const record of inspected.records) {
      if (ids.has(record.id)) {throw new Error(`Duplicate HCF object ID ${record.id}: ${file}`);}
      ids.add(record.id);
      union.push({file, ...record});
      if (union.length > maxHcfRecords) {throw new Error("Excessive aggregate HCF record count");}
    }
  }
  if (!union.length) {throw new Error("No HCF records in selected object union");}
  return union;
}

module.exports = {parseHcfObject, inspectHcfObject, inspectHcfObjects, assertUniqueHcfIds};
