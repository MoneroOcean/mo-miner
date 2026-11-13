"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {auditWindowsLinkMap} = require("../scripts/native-windows-application"), {fileHash} = require("../scripts/build-source-inputs");
const forbidden = /intrinsics[-_]shim|environ[-_]shim|(?:^|[^A-Za-z0-9])(?:_mm_|p__environ)/i;
/** @param {import("node:test").TestContext} t @param {string | Buffer} data */
function fixture(t, data) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-map-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const file = path.join(root, "link.map"); fs.writeFileSync(file, data); return file;
}

test.describe("Bounded Windows link-map audit", () => {
  test("all bytes and lines are inspected with an exact content digest", t => {
    const file = fixture(t, "header\r\nordinary_symbol\r\nlast_symbol"), result = auditWindowsLinkMap(file, forbidden);
    assert.equal(result.sha256, fileHash(file)); assert.equal(result.lines, 3); assert.equal(result.bytes, fs.statSync(file).size);
  });
  test("a clean map above the old whole-file threshold passes under a small JS heap", t => {
    const file = fixture(t, ""), fd = fs.openSync(file, "a"), chunk = Buffer.from("ordinary_native_symbol\n".repeat(4096));
    try {for (let i = 0; i < 400; i++) {fs.writeSync(fd, chunk);}} finally {fs.closeSync(fd);}
    assert.ok(fs.statSync(file).size > 32 << 20);
    const source = "const {auditWindowsLinkMap}=require(" + JSON.stringify(require.resolve("../scripts/native-windows-application")) + ");console.log(JSON.stringify(auditWindowsLinkMap(process.argv[1],/forbidden/)));";
    const result = spawnSync(process.execPath, ["--max-old-space-size=32", "-e", source, file], {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
    assert.equal(result.status, 0, result.stderr); assert.equal(JSON.parse(result.stdout).sha256, fileHash(file));
  });
  test("a forbidden symbol spanning a read boundary is rejected", t => {
    const file = fixture(t, "x".repeat((64 << 10) - 5) + " intrinsics-shim hidden\n");
    assert.throws(() => auditWindowsLinkMap(file, forbidden), /forbidden shim/);
  });
  test("a forbidden symbol after the old threshold is not skipped", t => {
    const file = fixture(t, ""), fd = fs.openSync(file, "a"), chunk = Buffer.from("ordinary_symbol\n".repeat(4096));
    try {for (let i = 0; i < 600; i++) {fs.writeSync(fd, chunk);} fs.writeSync(fd, " environ-shim\n");} finally {fs.closeSync(fd);}
    assert.ok(fs.statSync(file).size > 32 << 20); assert.throws(() => auditWindowsLinkMap(file, forbidden), /forbidden shim/);
  });
  test("the final unterminated line receives the same symbol checks", t => {
    const file = fixture(t, "header\n p__environ"); assert.throws(() => auditWindowsLinkMap(file, forbidden), /forbidden shim/);
  });
  test("intrinsic names retain their boundary rule", t => {
    assert.doesNotThrow(() => auditWindowsLinkMap(fixture(t, "name_mm_suffix\n"), forbidden));
    assert.throws(() => auditWindowsLinkMap(fixture(t, " _mm_helper\n"), forbidden), /forbidden shim/);
  });
  test("UTF-8 split across reads does not change content identity", t => {
    const file = fixture(t, "x".repeat((64 << 10) - 1) + "é-💎\nordinary\n"), result = auditWindowsLinkMap(file, forbidden);
    assert.equal(result.sha256, fileHash(file)); assert.equal(result.lines, 2);
  });
  test("global regular expressions cannot skip checks through lastIndex state", t => {
    const file = fixture(t, "ordinary\nforbidden\n"), pattern = /forbidden/g; pattern.lastIndex = 1000;
    assert.throws(() => auditWindowsLinkMap(file, pattern), /forbidden shim/); assert.equal(pattern.lastIndex, 1000);
  });
  test("unbounded individual symbol lines fail with a bounded diagnostic", t => {
    const file = fixture(t, "x".repeat((1 << 20) + 1));
    assert.throws(() => auditWindowsLinkMap(file, forbidden), /symbol line exceeds bounds/);
  });
  test("oversized sparse files fail before reading or allocating their contents", t => {
    const file = fixture(t, ""), fd = fs.openSync(file, "r+"); try {fs.ftruncateSync(fd, (512 << 20) + 1);} finally {fs.closeSync(fd);}
    assert.throws(() => auditWindowsLinkMap(file, forbidden), /bounded regular file/);
  });
  test("empty maps do not masquerade as completed linker evidence", t => {
    assert.throws(() => auditWindowsLinkMap(fixture(t, ""), forbidden), /bounded regular file/);
  });
  test("a complete application requires actual linked adapter symbol tokens", t => {
    const file = fixture(t, " 0001:00000100 clGetPlatformIDs 0000000140001100 OpenCL:icd.obj\n");
    assert.doesNotThrow(() => auditWindowsLinkMap(file, forbidden, ["clGetPlatformIDs"]));
    assert.throws(() => auditWindowsLinkMap(file, forbidden, ["clGetPlatformIDs", "mom_ur_cuda_factory"]), /omits required adapter symbols: mom_ur_cuda_factory/);
  });
  test("literal text and similarly named private symbols cannot certify an adapter", t => {
    for (const line of [" mom_private_clGetPlatformIDs 000001", " ??_C@clGetPlatformIDs@ 000001", " clGetPlatformIDsSuffix 000001",
      "commentary clGetPlatformIDs must not count", " 0001:00000100 mom_private_clGetPlatformIDs 0000000140001100 obj"]) {
      assert.throws(() => auditWindowsLinkMap(fixture(t, line), forbidden, ["clGetPlatformIDs"]), /omits required adapter/);
    }
  });
  test("unbounded, duplicate and source-injecting required symbol lists fail closed", t => {
    const file = fixture(t, "header\n");
    for (const required of [[""], ["a", "a"], ["bad\nname"], ["x".repeat(4097)], Array.from({length: 33}, (_, i) => "symbol" + i)]) {
      assert.throws(() => auditWindowsLinkMap(file, forbidden, required), /Invalid required Windows link symbols/);
    }
  });
});

test.describe("Windows standalone map rejects Node runtimes", () => {
  for (const symbol of ["napi_create_object", "__imp_napi_create_object", "_napi_create_object",
    "?Start@node@@YAXXZ", "?New@Isolate@v8@@SAPEAV12@XZ", "node::Start()", "v8::Isolate::New()"] ) {
    test("rejects " + symbol, t => {
      const file = fixture(t, " 0001:00000100 " + symbol + " 0000000140001100 runtime:object.obj\n");
      assert.throws(() => auditWindowsLinkMap(file, forbidden), /Node\/V8\/N-API/);
    });
  }
  for (const symbol of ["?walk@node@graph@sycl@@YAXXZ", "sycl::graph::node::run()", "ordinary_napi_suffix"]) {
    test("retains unrelated symbol " + symbol, t => {
      assert.doesNotThrow(() => auditWindowsLinkMap(fixture(t, symbol + "\n"), forbidden));
    });
  }
});
