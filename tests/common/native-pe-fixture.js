"use strict";

/** Structural PE fixture, not a compiled executable or an execution claim. */
function nativePeFixture() {
  const bytes = Buffer.alloc(16384), header = bytes.subarray(0, 4096), pe = 128, optional = pe + 24, table = optional + 240;
  header.writeUInt16LE(0x5a4d); header.writeUInt32LE(pe, 0x3c); header.writeUInt32LE(0x4550, pe);
  header.writeUInt16LE(0x8664, pe + 4); header.writeUInt16LE(3, pe + 6); header.writeUInt16LE(240, pe + 20);
  header.writeUInt16LE(0x20b, optional); header.writeUInt32LE(4096, optional + 32); header.writeUInt32LE(512, optional + 36);
  header.writeUInt32LE(16384, optional + 56); header.writeUInt32LE(4096, optional + 60); header.writeUInt16LE(0x160, optional + 70);
  for (const [i, name, flags] of /** @type {[number, string, number][]} */ ([[0, ".text", 0x60000020], [1, ".data", 0xc0000040], [2, ".mompp", 0x40000040]])) {
    const at = table + i * 40; header.write(name, at); header.writeUInt32LE(4096, at + 8); header.writeUInt32LE((i + 1) * 4096, at + 12);
    header.writeUInt32LE(4096, at + 16); header.writeUInt32LE((i + 1) * 4096, at + 20); header.writeUInt32LE(flags, at + 36);
  }
  bytes.writeUInt32LE(0x504d4f4d, 12288); bytes.writeUInt32LE(0x34393050, 12292); bytes.writeUInt32LE(1, 12296);
  return {bytes, header, pe, optional, table};
}

module.exports = {nativePeFixture};
