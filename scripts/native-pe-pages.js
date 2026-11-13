"use strict";

// Build-time page map for in-memory Windows executable packing. Read only bounded PE
// metadata; the application image can be hundreds of MiB and stays outside the JS heap.
const fs = require("node:fs");
const page = 4096, magic = [0x504d4f4d, 0x34393050, 1];

/** Parse only the x64 executable layout supported by the pre-CRT protection guard. Unknown
 * layouts fail before any file mutation, rather than guessing a writable/executable mapping.
 * @param {Buffer} header @param {number} fileSize */
function nativePePages(header, fileSize) {
  const fail = () => {throw new Error("Unsupported native PE page layout");};
  if (!Buffer.isBuffer(header) || header.length !== page || !Number.isSafeInteger(fileSize) || fileSize < page || fileSize > 0x7fffffff || header.readUInt16LE(0) !== 0x5a4d) {return fail();}
  const pe = header.readUInt32LE(0x3c), optional = pe + 24;
  if (pe < 64 || optional + 240 > page || header.readUInt32LE(pe) !== 0x4550 || header.readUInt16LE(pe + 4) !== 0x8664 ||
      header.readUInt16LE(pe + 20) !== 240 || (header.readUInt16LE(pe + 22) & 0x2000) || header.readUInt16LE(optional) !== 0x20b) {return fail();}
  const count = header.readUInt16LE(pe + 6), table = optional + 240, imageSize = header.readUInt32LE(optional + 56),
    fileAlignment = header.readUInt32LE(optional + 36), headerSize = header.readUInt32LE(optional + 60), mitigations = header.readUInt16LE(optional + 70);
  if (!count || count > 64 || table + count * 40 > page || header.readUInt32LE(optional + 32) !== page ||
      ![512, 1024, 2048, 4096].includes(fileAlignment) || headerSize < table + count * 40 || headerSize > page || headerSize % fileAlignment ||
      imageSize < page || imageSize > 0x7fffffff || imageSize % page ||
      (mitigations & 0x160) !== 0x160) {return fail();} // High-entropy ASLR, dynamic base and NX stay enabled.
  const sections = []; let previous = page, rawEnd = headerSize, mapOffset = -1;
  for (let index = 0; index < count; index++) {
    const at = table + index * 40, name = header.subarray(at, at + 8).toString("ascii").replace(/\0+$/, ""),
      rva = header.readUInt32LE(at + 12), rawSize = header.readUInt32LE(at + 16), offset = header.readUInt32LE(at + 20), flags = header.readUInt32LE(at + 36),
      size = Math.ceil(Math.max(header.readUInt32LE(at + 8), rawSize) / page) * page;
    if (!name || name.includes("\0") || rva < previous || rva % page || !size || rva > imageSize || size > imageSize - rva ||
        (!(flags & 0x40000000) && (flags & 0xa0000000)) || ((flags & 0xa0000000) >>> 0) === 0xa0000000 ||
        (rawSize && (offset < rawEnd || offset % fileAlignment || rawSize % fileAlignment || offset > fileSize || rawSize > fileSize - offset))) {return fail();}
    if (name === ".mompp") {
      if (mapOffset !== -1 || size !== page || rawSize !== page || !(flags & 0x40000000) || (flags & 0xa0000000)) {return fail();}
      mapOffset = offset;
    }
    previous = rva + size; if (rawSize) {rawEnd = offset + rawSize;}
    sections.push({rva, size, flags});
  }
  if (mapOffset < 0 || previous !== imageSize) {return fail();}
  return {imageSize, mapOffset, mitigations, sections};
}

/** Fill the linked guard's one reserved read-only page. No executable instructions, relocation,
 * import or application payload is rewritten. A sealed/corrupted slot is never overwritten.
 * @param {string} file */
function sealNativePePages(file) {
  const fd = fs.openSync(file, fs.constants.O_RDWR | fs.constants.O_NONBLOCK);
  try {
    const before = fs.fstatSync(fd, {bigint: true});
    if (!before.isFile()) {throw new Error("Native PE page map requires a regular executable");}
    const header = Buffer.alloc(page);
    if (fs.readSync(fd, header, 0, page, 0) !== page) {throw new Error("Truncated native PE page header");}
    const layout = nativePePages(header, Number(before.size)), map = Buffer.alloc(page);
    if (fs.readSync(fd, map, 0, page, layout.mapOffset) !== page || magic.some((value, i) => map.readUInt32LE(i * 4) !== value) ||
        map.subarray(12).some(value => value !== 0)) {throw new Error("Native PE page map is missing, corrupted or already sealed");}
    const current = fs.fstatSync(fd, {bigint: true});
    if (current.size !== before.size || current.mtimeNs !== before.mtimeNs || current.ctimeNs !== before.ctimeNs) {throw new Error("Native PE changed before sealing");}
    map.writeUInt32LE(layout.sections.length, 12); map.writeUInt32LE(layout.imageSize, 16);
    layout.sections.forEach((section, index) => [section.rva, section.size, section.flags].forEach((value, field) => map.writeUInt32LE(value, 32 + index * 12 + field * 4)));
    if (fs.writeSync(fd, map, 0, page, layout.mapOffset) !== page) {throw new Error("Incomplete native PE page map write");}
    fs.fsyncSync(fd);
    return layout;
  } finally {fs.closeSync(fd);}
}

module.exports = {nativePePages, sealNativePePages};
