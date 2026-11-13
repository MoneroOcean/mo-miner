"use strict";
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), test = require("node:test"), assert = require("node:assert/strict");
const {ioPins, nativeIoPlan, validateIoArchive, buildNativeIo, nativeIoInputs} = require("../scripts/build-native-io");
const {fileHash} = require("../scripts/build-source-inputs");
const {spawnSync} = require("node:child_process");
const tools = {cmake: "/tools/cmake", cc: "/tools/cc", ninja: "/tools/ninja"};
/** Synthetic static archives exercise SDK admission, never compilers or mining.
 * @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-io-sdk-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const artifacts = /** @type {Record<string, string>} */ ({});
  for (const [name, data] of Object.entries({"lib/libuv.a": "!<arch>\n", "lib/libz.a": "!<arch>\n", "include/uv.h": "uv", "include/zlib.h": "zlib", "include/zconf.h": "zconf"})) {
    const file = path.join(root, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data); artifacts[name] = fileHash(file);
  }
  const receipt = {version: 1, target: "linux", pins: ioPins, buildOnly: true, artifacts};
  const save = () => fs.writeFileSync(path.join(root, "result.json"), JSON.stringify(receipt)); save();
  return {root, receipt, save};
}

test.describe("Original static I/O SDK production", () => {
  test("only the existing pinned original libuv/zlib cohorts are selected", () => {
    assert.equal(ioPins.libuv.directory, "libuv-v1.51.0"); assert.equal(ioPins.zlib.directory, "zlib-1.3.1");
    assert.ok(Object.isFrozen(ioPins)); assert.ok(Object.isFrozen(ioPins.libuv));
  });
  test("compiler and Ninja paths are explicit and parallelism is bounded", () => {
    for (const p of nativeIoPlan("/new/sdk", tools, 2)) {
      assert.ok(p.configure.includes("-DCMAKE_C_COMPILER:FILEPATH=" + tools.cc)); assert.ok(p.configure.includes("-DCMAKE_MAKE_PROGRAM:FILEPATH=" + tools.ninja));
      assert.deepEqual(p.compile.slice(0, 5), [tools.cmake, "--build", p.build, "--parallel", "2"]);
    }
  });
  test("the build selects static targets, not tests, benchmarks or shared libraries", () => {
    const [uv, z] = nativeIoPlan("/new/sdk", tools, 2); assert.equal(uv?.compile.at(-1), "uv_a"); assert.equal(z?.compile.at(-1), "zlibstatic");
    for (const flag of ["-DLIBUV_BUILD_SHARED:BOOL=OFF", "-DLIBUV_BUILD_TESTS:BOOL=OFF", "-DLIBUV_BUILD_BENCH:BOOL=OFF"]) {assert.ok(uv?.configure.includes(flag));}
  });
  test("Windows keeps the same source with static CRT and native archive names", () => {
    const p = nativeIoPlan("/new/sdk", tools, 2, "win32"); assert.match(p[0]?.archive || "", /libuv\.lib$/); assert.match(p[1]?.archive || "", /zlibstatic\.lib$/);
    assert.equal(p[0]?.installedArchive, path.join("/new/sdk", "lib", "uv_a.lib"));
    assert.equal(p[1]?.installedArchive, path.join("/new/sdk", "lib", "zlibstatic.lib"));
    for (const row of p) {assert.ok(row.configure.includes("-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreaded"));}
  });
  test("invalid jobs, relative tools, unsupported platforms and CMake fragments fail closed", () => {
    for (const jobs of [0, 17, 1.1, NaN]) {assert.throws(() => nativeIoPlan("/new/sdk", tools, jobs), /I\/O SDK needs/);}
    assert.throws(() => nativeIoPlan("/new/sdk", {...tools, ninja: "ninja"}, 2), /I\/O SDK needs/);
    assert.throws(() => nativeIoPlan("/new/sdk;$bad", tools, 2), /I\/O SDK needs/);
    assert.throws(() => nativeIoPlan("/new/sdk", tools, 2, "darwin"), /I\/O SDK needs/);
  });
  test("source extraction accepts only the exact selected cohort's members", () => {
    validateIoArchive("libuv-v1.51.0/\nlibuv-v1.51.0/include/uv.h\n", ioPins.libuv.directory);
    for (const member of ["", "/libuv-v1.51.0/file", "libuv-v1.51.0/../escape", "other/file", "libuv-v1.51.0/a\\b", "libuv-v1.51.0/a\x00b"]) {
      assert.throws(() => validateIoArchive(member, ioPins.libuv.directory), /archive members/);
    }
  });
  test("no source or compiler is touched without the build-only boundary", () => {
    assert.throws(() => buildNativeIo({libuv: "/missing", zlib: "/missing"}, "/new/sdk", tools, 2, {}), /MOM_AOT_BUILD_ONLY=1/);
  });
});

test.describe("Native I/O SDK admission", () => {
  test("a complete SDK resolves exact static files rather than ambient -l lookups", t => {
    const {root} = fixture(t), result = nativeIoInputs(root, "linux"); assert.equal(result.include, path.join(root, "include"));
    assert.deepEqual(result.libraries, [path.join(root, "lib/libuv.a"), path.join(root, "lib/libz.a")]);
    assert.equal(result.receipt, path.join(root, "result.json")); assert.equal(result.receiptSha256, fileHash(result.receipt));
  });
  test("receipt replacement during archive validation is rejected", t => {
    const {root} = fixture(t), archive = path.join(root, "lib/libz.a"), receipt = path.join(root, "result.json"), open = fs.openSync;
    let reads = 0;
    /** @type {typeof fs.openSync} */
    const observeOpen = (file, flags, mode) => {
      const fd = open(file, flags, mode);
      if (file === archive && ++reads === 3) {fs.appendFileSync(receipt, "\n");}
      return fd;
    };
    t.mock.method(fs, "openSync", observeOpen);
    assert.throws(() => nativeIoInputs(root, "linux"), /receipt changed during admission/); assert.equal(reads, 3);
  });
  test("host/target mismatch cannot silently link a different platform's SDK", t => {
    const {root} = fixture(t); assert.throws(() => nativeIoInputs(root, "win32"), /SDK receipt/);
  });
  test("an SDK without the required header cannot get as far as C++ compilation", t => {
    const {root, receipt, save} = fixture(t); delete receipt.artifacts["include/uv.h"]; save(); assert.throws(() => nativeIoInputs(root, "linux"), /Incomplete/);
  });
  test("the actual controller CLI rejects incomplete I/O inputs without creating a compiler workspace", {skip: process.platform !== "linux"}, t => {
    const {root, receipt, save} = fixture(t), output = path.join(root, "output"); fs.mkdirSync(output); delete receipt.artifacts["include/uv.h"]; save();
    const result = spawnSync(process.execPath, [path.resolve(__dirname, "../scripts/probe-native-controller.js"), "/missing/hermes", "/missing/build", output],
      {encoding: "utf8", timeout: 15000, env: {...process.env, MOM_AOT_IO_SDK: root}});
    assert.equal(result.status, 1); assert.match(result.stderr, /Incomplete native I\/O SDK/); assert.deepEqual(fs.readdirSync(output), []);
  });
  test("changed staged library bytes are rejected before linking", t => {
    const {root} = fixture(t); fs.appendFileSync(path.join(root, "lib/libuv.a"), "changed"); assert.throws(() => nativeIoInputs(root, "linux"), /bytes changed/);
  });
  test("unrecorded headers cannot shadow system or original SDK includes", t => {
    const {root} = fixture(t); fs.writeFileSync(path.join(root, "include/stdio.h"), "unexpected");
    assert.throws(() => nativeIoInputs(root, "linux"), /membership or bytes changed/);
  });
  test("extra staged entries cannot turn SDK admission into an unbounded tree scan", t => {
    const {root} = fixture(t); for (let n = 0; n < 257; n++) {fs.writeFileSync(path.join(root, "include/extra" + n + ".h"), "x");}
    assert.throws(() => nativeIoInputs(root, "linux"), /bounded native I\/O SDK tree/);
  });
  test("unrecorded directory symlinks are rejected rather than traversed", {skip: process.platform === "win32"}, t => {
    const {root} = fixture(t); fs.symlinkSync(root, path.join(root, "include/loop"));
    assert.throws(() => nativeIoInputs(root, "linux"), /bounded native I\/O SDK tree/);
  });
  test("receipt hash updates cannot disguise a non-static application library", t => {
    const {root, receipt, save} = fixture(t), file = path.join(root, "lib/libuv.a"); fs.writeFileSync(file, "not an archive");
    receipt.artifacts["lib/libuv.a"] = fileHash(file); save(); assert.throws(() => nativeIoInputs(root, "linux"), /static archives/);
  });
  test("escaping receipt members are rejected without reading outside the SDK", t => {
    const {root, receipt, save} = fixture(t); receipt.artifacts["include/../outside.h"] = "hash"; save(); assert.throws(() => nativeIoInputs(root, "linux"), /SDK member/);
  });
  test("staged symlinks cannot redirect the library admission check", {skip: process.platform === "win32"}, t => {
    const {root} = fixture(t), file = path.join(root, "lib/libuv.a"); fs.unlinkSync(file); fs.symlinkSync(path.join(root, "lib/libz.a"), file);
    assert.throws(() => nativeIoInputs(root, "linux"), /bytes changed/);
  });
  test("receipt configuration cannot masquerade as an unpinned original source cohort", t => {
    const {root, receipt, save} = fixture(t); Object.assign(receipt, {pins: {...ioPins, zlib: {...ioPins.zlib, sha256: "bad"}}}); save();
    assert.throws(() => nativeIoInputs(root, "linux"), /SDK receipt/);
  });
});
