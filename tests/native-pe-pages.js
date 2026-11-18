"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {nativePePages, sealNativePePages} = require("../scripts/native-pe-pages");
const {verifyNativePeMap} = require("../scripts/native-pe-audit");

const {nativePeFixture: fixture} = require("./common/native-pe-fixture");

test.describe("Native Windows exact page-map admission", () => {
  test("admits original disjoint RX, RW and RO sections without reading payloads", () => {
    const f = fixture(), result = nativePePages(f.header, f.bytes.length);
    assert.deepEqual(result, {imageSize: 16384, mapOffset: 12288, mitigations: 0x160, sections: [
      {rva: 4096, size: 4096, flags: 0x60000020}, {rva: 8192, size: 4096, flags: 0xc0000040}, {rva: 12288, size: 4096, flags: 0x40000040},
    ]});
  });
  test("truncated, non-PE and non-x64 headers fail closed", () => {
    const f = fixture();
    for (const size of [0, 63, 4095]) {assert.throws(() => nativePePages(f.header.subarray(0, size), f.bytes.length), /layout/);}
    for (const offset of [0, f.pe, f.pe + 4, f.optional]) {const bad = Buffer.from(f.header); bad.writeUInt16LE(0, offset); assert.throws(() => nativePePages(bad, f.bytes.length), /layout/);}
  });
  test("file headers smaller than a virtual page retain exact raw offsets", () => {
    const f = fixture(); f.header.writeUInt32LE(1024, f.optional + 60);
    for (let i = 0; i < 3; i++) {f.header.writeUInt32LE(1024 + i * 4096, f.table + i * 40 + 20);}
    assert.equal(nativePePages(f.header, f.bytes.length).mapOffset, 9216);
    f.header.writeUInt32LE(0, f.optional + 60); assert.throws(() => nativePePages(f.header, f.bytes.length), /layout/);
  });
  test("hostile PE/table offsets cannot escape the bounded header", () => {
    for (const value of [0, 4090, 0x7fffffff, 0xffffffff]) {const f = fixture(); f.header.writeUInt32LE(value, 0x3c); assert.throws(() => nativePePages(f.header, f.bytes.length), /layout/);}
    for (const count of [0, 65, 65535]) {const f = fixture(); f.header.writeUInt16LE(count, f.pe + 6); assert.throws(() => nativePePages(f.header, f.bytes.length), /layout/);}
  });
  test("DLL images and foreign section/file alignment are refused", () => {
    const f = fixture(); f.header.writeUInt16LE(0x2000, f.pe + 22); assert.throws(() => nativePePages(f.header, f.bytes.length), /layout/);
    for (const [offset, value] of [[f.optional + 32, 512], [f.optional + 36, 0], [f.optional + 36, 3]]) {
      const g = fixture(); g.header.writeUInt32LE(value || 0, offset); assert.throws(() => nativePePages(g.header, g.bytes.length), /layout/);
    }
  });
  test("ASLR, high-entropy addresses and NX cannot be disabled to permit packing", () => {
    for (const bit of [0x20, 0x40, 0x100]) {const f = fixture(); f.header.writeUInt16LE(0x160 ^ bit, f.optional + 70); assert.throws(() => nativePePages(f.header, f.bytes.length), /layout/);}
  });
  test("executable-writeable or unreadable source pages are never admitted", () => {
    for (const flags of [0xe0000020, 0x20000020, 0x80000040]) {const f = fixture(); f.header.writeUInt32LE(flags, f.table + 36); assert.throws(() => nativePePages(f.header, f.bytes.length), /layout/);}
  });
  test("non-loaded target-symbol metadata retains no-access permissions", () => {
    const f = fixture(); f.header.writeUInt32LE(0x40, f.table + 36);
    assert.equal(nativePePages(f.header, f.bytes.length).sections[0]?.flags, 0x40);
  });
  test("overlapping, unaligned and overflowing image ranges fail", () => {
    for (const [offset, value] of [[12, 1], [12, 8193], [8, 0xffffffff], [12, 0xfffff000]]) {
      const f = fixture(); f.header.writeUInt32LE(value || 0, f.table + 40 + (offset || 0)); assert.throws(() => nativePePages(f.header, f.bytes.length), /layout/);
    }
  });
  test("overlapping and out-of-file raw sections cannot supply the map", () => {
    for (const [offset, value] of [[20, 4096], [20, 0xfffff000], [16, 0xfffff000], [20, 8193]]) {
      const f = fixture(); f.header.writeUInt32LE(value || 0, f.table + 40 + (offset || 0)); assert.throws(() => nativePePages(f.header, f.bytes.length), /layout/);
    }
  });
  test("the manifest is exactly one read-only original page", () => {
    for (const [offset, value] of [[8, 8192], [16, 512], [36, 0xc0000040], [36, 0x60000040], [36, 0x40]]) {
      const f = fixture(); f.header.writeUInt32LE(value || 0, f.table + 80 + (offset || 0)); assert.throws(() => nativePePages(f.header, f.bytes.length), /layout/);
    }
    const f = fixture(); f.header.write(".other", f.table + 80); assert.throws(() => nativePePages(f.header, f.bytes.length), /layout/);
  });
  test("unmapped image tails and unbounded file sizes cannot bypass the guard", () => {
    const f = fixture(); f.header.writeUInt32LE(20480, f.optional + 56); assert.throws(() => nativePePages(f.header, f.bytes.length), /layout/);
    for (const size of [0, NaN, Infinity, 0x80000000]) {assert.throws(() => nativePePages(fixture().header, size), /layout/);}
  });
});

test.describe("Native Windows page-map sealing", () => {
  /** @param {import("node:test").TestContext} t */
  function temporary(t) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pe-pages-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const file = path.join(root, "mom.exe"), data = fixture().bytes; fs.writeFileSync(file, data); return {root, file, data};
  }
  test("only the fixed map page changes; code, imports and other data are byte-identical", t => {
    const f = temporary(t), result = sealNativePePages(f.file), after = fs.readFileSync(f.file);
    assert.ok(after.subarray(0, 12288).equals(f.data.subarray(0, 12288))); assert.equal(after.length, f.data.length);
    assert.equal(after.readUInt32LE(12300), 3); assert.equal(after.readUInt32LE(12304), 16384);
    assert.deepEqual(Array.from({length: 3}, (_, i) => ({rva: after.readUInt32LE(12320 + i * 12), size: after.readUInt32LE(12324 + i * 12), flags: after.readUInt32LE(12328 + i * 12)})), result.sections);
  });
  test("offline PE admission requires the sealed page map", t => {
    const f = temporary(t);
    assert.throws(() => verifyNativePeMap(f.file), /exact sealed pre-CRT page map/);
    const result = sealNativePePages(f.file);
    assert.deepEqual(verifyNativePeMap(f.file), result);
  });
  test("offline PE admission rejects a changed count, section tuple or reserved byte", t => {
    for (const relativeOffset of [12, 32, 4095]) {
      const f = temporary(t), result = sealNativePePages(f.file), bytes = fs.readFileSync(f.file), offset = result.mapOffset + relativeOffset;
      bytes[offset] = (bytes[offset] || 0) ^ 1;
      fs.writeFileSync(f.file, bytes);
      assert.throws(() => verifyNativePeMap(f.file), /exact sealed pre-CRT page map/, "changed map offset " + relativeOffset + " must fail");
    }
  });
  test("sealing twice fails without mutating the first sealed image", t => {
    const f = temporary(t); sealNativePePages(f.file); const first = fs.readFileSync(f.file);
    assert.throws(() => sealNativePePages(f.file), /already sealed/); assert.ok(first.equals(fs.readFileSync(f.file)));
  });
  test("corrupt slot identity or nonzero reserved bytes are not overwritten", t => {
    const f = temporary(t);
    for (const offset of [12288, 12296, 12308, 16383]) {
      const bytes = Buffer.from(f.data); bytes[offset] = (bytes[offset] || 0) ^ 1; fs.writeFileSync(f.file, bytes);
      assert.throws(() => sealNativePePages(f.file), /corrupted/); assert.ok(bytes.equals(fs.readFileSync(f.file)));
    }
  });
  test("a malformed image fails before the map page is written", t => {
    const f = temporary(t), bad = Buffer.from(f.data); bad.writeUInt32LE(0xffffffff, 0x3c); fs.writeFileSync(f.file, bad);
    assert.throws(() => sealNativePePages(f.file), /layout/); assert.ok(bad.equals(fs.readFileSync(f.file)));
  });
  test("empty and directory inputs fail promptly", t => {
    const f = temporary(t); fs.truncateSync(f.file);
    assert.throws(() => sealNativePePages(f.file), /Truncated/); assert.throws(() => sealNativePePages(f.root));
  });
});
