"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {nativePeFixture} = require("./common/native-pe-fixture"), {auditNativePePayload} = require("../scripts/native-pe-audit");

/** Small synthetic loader tables and payloads, not a compiled/mining execution claim. */
function image() {
  const f = nativePeFixture(), b = f.bytes, data = 8192;
  b.writeUInt32LE(0x40000040, f.table + 40 + 36);
  b.writeUInt32LE(4096, f.optional + 16); b.writeBigUInt64LE(0x140000000n, f.optional + 24);
  b.writeUInt32LE(data + 32, f.optional + 112); b.writeUInt32LE(512, f.optional + 116);
  const exp = data + 32; b.writeUInt32LE(data + 176, exp + 12); b.writeUInt32LE(1, exp + 16);
  b.writeUInt32LE(2, exp + 20); b.writeUInt32LE(2, exp + 24);
  b.writeUInt32LE(data + 128, exp + 28); b.writeUInt32LE(data + 144, exp + 32); b.writeUInt32LE(data + 160, exp + 36);
  b.writeUInt32LE(4112, data + 128); b.writeUInt32LE(4128, data + 132);
  b.writeUInt32LE(data + 192, data + 144); b.writeUInt32LE(data + 208, data + 148);
  b.writeUInt16LE(0, data + 160); b.writeUInt16LE(1, data + 162);
  b.write("mom.exe\0", data + 176); b.write("alpha\0", data + 192); b.write("beta\0", data + 208);
  b.writeUInt32LE(data + 768, f.optional + 120); b.writeUInt32LE(40, f.optional + 124);
  b.writeUInt32LE(data + 1280, data + 768); b.writeUInt32LE(data + 1152, data + 780); b.writeUInt32LE(data + 1312, data + 784);
  b.writeUInt32LE(data + 1312, f.optional + 208); b.writeUInt32LE(24, f.optional + 212);
  b.writeUInt16LE(17, data + 1024); b.write("ExitProcess\0", data + 1026);
  b.writeUInt16LE(29, data + 1072); b.write("CloseHandle\0", data + 1074); b.write("KERNEL32.dll\0", data + 1152);
  for (const start of [data + 1280, data + 1312]) {b.writeBigUInt64LE(BigInt(data + 1024), start); b.writeBigUInt64LE(BigInt(data + 1072), start + 8);}
  return f;
}
/** @param {import("node:test").TestContext} t */
function pair(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pe-audit-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const source = path.join(root, "raw.exe"), restored = path.join(root, "restored.exe"), f = image();
  fs.writeFileSync(source, f.bytes); fs.writeFileSync(restored, f.bytes);
  return {...f, source, restored, check: () => auditNativePePayload(source, restored), change: (/** @type {(data: Buffer) => void} */ fn) => {
    const data = Buffer.from(f.bytes); fn(data); fs.writeFileSync(restored, data);
  }};
}

test.describe("Native PE exact loader bindings", () => {
  test("identical initialized payloads and bindings pass", t => {const p = pair(t), result = p.check(); assert.equal(result.payloads.length, 3); assert.equal(result.exportBindings.length, 64);});
  test("equivalent export tables may move inside the exact export directory", t => {
    const p = pair(t); p.change(b => {b.copy(b, 8192 + 256, 8192 + 144, 8192 + 152); b.writeUInt32LE(8192 + 256, 8192 + 64);}); assert.doesNotThrow(p.check);
  });
  test("export name order still satisfies Windows binary lookup", t => {
    const p = pair(t); p.change(b => {b.writeUInt32LE(8192 + 208, 8192 + 144); b.writeUInt32LE(8192 + 192, 8192 + 148);}); assert.throws(p.check, /Unsorted/);
  });
  test("changed export function RVAs are rejected", t => {const p = pair(t); p.change(b => b.writeUInt32LE(4144, 8192 + 128)); assert.throws(p.check, /bindings/);});
  test("export names and ordinals remain exact", t => {
    const p = pair(t); p.change(b => b.write("Alpha", 8192 + 192)); assert.throws(p.check, /bindings/);
    p.change(b => b.writeUInt16LE(1, 8192 + 160)); assert.throws(p.check, /bindings/);
  });
  test("forwarded exports retain the exact target", t => {
    const p = pair(t), data = Buffer.from(p.bytes); data.writeUInt32LE(8192 + 224, 8192 + 128); data.write("OTHER.alpha\0", 8192 + 224);
    fs.writeFileSync(p.source, data); fs.writeFileSync(p.restored, data); assert.doesNotThrow(p.check);
    data.write("OTHER.bravo\0", 8192 + 224); fs.writeFileSync(p.restored, data); assert.throws(p.check, /bindings/);
  });
  test("DLL case, advisory hints and IAT lookup fallback preserve imports", t => {
    const p = pair(t); p.change(b => {
      b.write("kernel32.DLL", 8192 + 1152); b.writeUInt16LE(0, 8192 + 1024); b.writeUInt16LE(0, 8192 + 1072);
      b.writeUInt32LE(0, 8192 + 768); b.fill(0, 8192 + 1280, 8192 + 1304); b.fill(0, p.optional + 208, p.optional + 216);
    }); assert.doesNotThrow(p.check);
  });
  test("imported names stay case-sensitive", t => {const p = pair(t); p.change(b => b.write("exitProcess", 8192 + 1026)); assert.throws(p.check, /bindings/);});
  test("same imported symbols cannot bind to different IAT slots", t => {const p = pair(t); p.change(b => b.writeUInt32LE(8192 + 1320, 8192 + 784)); assert.throws(p.check, /bindings/);});
  test("ordinal imports retain their exact value", t => {
    const p = pair(t), data = Buffer.from(p.bytes); data.writeBigUInt64LE((1n << 63n) | 17n, 8192 + 1280); data.copy(data, 8192 + 1312, 8192 + 1280, 8192 + 1288);
    fs.writeFileSync(p.source, data); fs.writeFileSync(p.restored, data); assert.doesNotThrow(p.check);
    data.writeBigUInt64LE((1n << 63n) | 18n, 8192 + 1280); fs.writeFileSync(p.restored, data); assert.throws(p.check, /bindings/);
  });
  test("excessive export tables fail before allocation", t => {const p = pair(t); p.change(b => b.writeUInt32LE(16385, 8192 + 52)); assert.throws(p.check, /count/);});
  test("images without exports still audit every payload byte", t => {
    const p = pair(t), data = Buffer.from(p.bytes); data.fill(0, p.optional + 112, p.optional + 120);
    fs.writeFileSync(p.source, data); fs.writeFileSync(p.restored, data); assert.doesNotThrow(p.check);
    data[8192 + 128] = (data[8192 + 128] || 0) ^ 1; fs.writeFileSync(p.restored, data); assert.throws(p.check, /payload/);
  });
});

test.describe("Native PE payload and metadata trust boundaries", () => {
  test("real code bytes cannot disappear into a metadata exception", t => {const p = pair(t); p.change(b => {b[4219] = 1;}); assert.throws(p.check, /payload/);});
  test("ordinary read-only resources remain byte-identical", t => {const p = pair(t); p.change(b => {b[8192 + 2304] = 1;}); assert.throws(p.check, /payload/);});
  test("unreferenced bytes inside an export directory are still exact application payload", t => {
    const p = pair(t); p.change(b => {b[8192 + 300] = 1;}); assert.throws(p.check, /payload/);
  });
  test("raw alignment bytes outside VirtualSize do not become application payload", t => {
    const p = pair(t), data = Buffer.from(p.bytes); data.writeUInt32LE(4090, p.table + 8); fs.writeFileSync(p.source, data);
    data[8191] = 1; fs.writeFileSync(p.restored, data); assert.doesNotThrow(p.check);
  });
  test("writable metadata is not a loader-data exception", t => {
    const p = pair(t), data = Buffer.from(p.bytes); data.writeUInt32LE(0xc0000040, p.table + 40 + 36);
    fs.writeFileSync(p.source, data); fs.writeFileSync(p.restored, data); assert.throws(p.check, /read-only/);
  });
  test("out-of-section metadata references are rejected", t => {const p = pair(t); p.change(b => b.writeUInt32LE(0xfffffff0, 8192 + 64)); assert.throws(p.check, /escapes/);});
  test("unterminated export names cannot scan another section", t => {
    const p = pair(t); p.change(b => {b.writeUInt32LE(12287, 8192 + 144); b[12287] = 97;}); assert.throws(p.check, /Unterminated/);
  });
  test("malformed import-directory sizes fail closed", t => {const p = pair(t); p.change(b => b.writeUInt32LE(4095, p.optional + 124)); assert.throws(p.check, /header|directory/);});
  test("metadata references cannot exempt executable pages", t => {
    const p = pair(t), data = Buffer.from(p.bytes); data.writeBigUInt64LE(4352n, 8192 + 1280); data.writeBigUInt64LE(4352n, 8192 + 1312); data.write("ExitProcess\0", 4354);
    fs.writeFileSync(p.source, data); fs.writeFileSync(p.restored, data); assert.throws(p.check, /read-only/);
  });
  test("entrypoint and stack limits are preserved", t => {
    const p = pair(t); p.change(b => b.writeUInt32LE(4128, p.optional + 16)); assert.throws(p.check, /header/);
    p.change(b => b.writeBigUInt64LE(8192n, p.optional + 72)); assert.throws(p.check, /header/);
  });
  test("optional IAT removal is narrow, not an arbitrary directory rewrite", t => {
    const p = pair(t); p.change(b => b.fill(0, p.optional + 208, p.optional + 216)); assert.doesNotThrow(p.check);
    p.change(b => b.writeUInt32LE(8192 + 2304, p.optional + 208)); assert.throws(p.check, /IAT directory/);
  });
  test("permission/mitigation changes fail before hashing application data", t => {
    const p = pair(t); p.change(b => b.writeUInt32LE(0x40000020, p.table + 36)); assert.throws(p.check, /layout/);
    p.change(b => b.writeUInt16LE(0x60, p.optional + 70)); assert.throws(p.check, /layout/);
  });
  test("invalid files fail without modifying the original", t => {
    const p = pair(t); fs.truncateSync(p.restored); assert.throws(p.check, /Missing/); assert.ok(fs.readFileSync(p.source).equals(p.bytes));
    assert.throws(() => auditNativePePayload(path.dirname(p.source), p.restored));
  });
});
