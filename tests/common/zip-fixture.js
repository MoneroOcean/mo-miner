"use strict";

/** @param {Uint8Array} data */
function crc32(data) {
  let crc = 0xffffffff;
  for (const byte of data) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) {crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);}
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Build a tiny stored (uncompressed) ZIP with ordinary Unix 0644 file modes. Names are written
 * verbatim, so tests can include duplicate, traversal or otherwise malformed member paths.
 * @param {Array<[string, Uint8Array | string]>} entries @returns {Buffer}
 */
function storedZip(entries) {
  const local = [], central = []; let offset = 0;
  for (const [name, value] of entries) {
    const nameBytes = Buffer.from(name), data = Buffer.from(value), crc = crc32(data), header = Buffer.alloc(30);
    header.writeUInt32LE(0x04034b50, 0); header.writeUInt16LE(20, 4); header.writeUInt32LE(crc, 14);
    header.writeUInt32LE(data.length, 18); header.writeUInt32LE(data.length, 22); header.writeUInt16LE(nameBytes.length, 26);
    local.push(header, nameBytes, data);
    const row = Buffer.alloc(46); row.writeUInt32LE(0x02014b50, 0); row.writeUInt16LE(0x0314, 4); row.writeUInt16LE(20, 6);
    row.writeUInt32LE(crc, 16); row.writeUInt32LE(data.length, 20); row.writeUInt32LE(data.length, 24);
    row.writeUInt16LE(nameBytes.length, 28); row.writeUInt32LE((0o100644 << 16) >>> 0, 38); row.writeUInt32LE(offset, 42);
    central.push(row, nameBytes); offset += header.length + nameBytes.length + data.length;
  }
  const centralBytes = Buffer.concat(central), end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(centralBytes.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...local, centralBytes, end]);
}

module.exports = {storedZip};
