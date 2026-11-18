"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const nodeCrypto = require("node:crypto");
const nodeZlib = require("node:zlib");
const nativeFs = require("../runtime/fs");
const crypto = require("../runtime/crypto");
const zlib = require("../runtime/zlib");
const nativeProcess = require("../runtime/process");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {extractCppBlock} = require("./common/cpp-source");

test.describe("Native service compilation portability", {skip: process.platform !== "linux"}, () => {
  test("production gzip bounds compile with Windows min/max macros", (t) => {
    const source = fs.readFileSync(path.join(__dirname, "../runtime/services.cpp"), "utf8");
    const expressions = [...source.matchAll(/if \((bytes\.size|capacity) > ([^\n]+)\) \{throw JSError\(rt, "Gzip (?:input|output) is too large"\);\}/g)]
      .map(match => match[2]);
    assert.equal(expressions.length, 2);
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-service-windows-macros-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const file = path.join(directory, "bounds.cpp"), object = path.join(directory, "bounds.o");
    fs.writeFileSync(file, "#include <limits>\nusing uInt=unsigned;\n#define max(a,b) not_a_function\n" +
      expressions.map((expression, index) => `static_assert(${expression} == ~uInt(0), "bound ${index}");`).join("\n"));
    const result = spawnSync("c++", ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-c", file, "-o", object], {encoding: "utf8", timeout: 30000});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
  });

  test("production IP helper matches Node literal controls through libuv", (t) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-service-ip-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const source = fs.readFileSync(path.join(__dirname, "../runtime/services.cpp"), "utf8");
    const cases = new Map([
      ["", 0], ["0.0.0.0", 4], ["127.0.0.1", 4], ["255.255.255.255", 4],
      ["::", 6], ["::1", 6], ["2001:db8::1", 6],
      ["::ffff:192.0.2.1", 6], ["2001:db8::192.0.2.1", 6],
      ["fe80::1%eth0", 6], ["fe80::1%zone.with:chars", 6],
      ["01.2.3.4", 0], ["127.0.0.01", 0], ["256.0.0.1", 0],
      [" 127.0.0.1", 0], ["127.0.0.1 ", 0], ["[::1]", 0],
      ["fe80::1%", 0], ["fe80::1%bad_zone", 0], ["fe80::1%one%two", 0],
      ["127.0.0.1%eth0", 0], ["fe80::1%\u00e9", 0], ["2001:db8:::1", 0],
    ]);
    const assertions = [...cases].map(([literal, expected]) =>
      `assert(nativeIsIP(${JSON.stringify(literal)}) == ${expected});`).join("\n");
    const file = path.join(directory, "ip.cpp"), executable = path.join(directory, "ip");
    fs.writeFileSync(file, [
      "#include <cassert>", "#include <string>", "#include <uv.h>",
      extractCppBlock(source, "int nativeIsIP("),
      "int main() {", assertions,
      "std::string embedded = \"2001:db8::1\"; embedded.push_back('\\0'); embedded += \"tail\";",
      "assert(nativeIsIP(embedded) == 0);",
      "assert(nativeIsIP(std::string(4097, 'a')) == 0);", "}", "",
    ].join("\n"));
    const compiler = process.env["CXX"] || "c++";
    const compiled = spawnSync(compiler, ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", file, "-luv", "-o", executable],
      {encoding: "utf8", timeout: 30000});
    assert.equal(compiled.status, 0, compiled.error?.message || compiled.stderr);
    const run = spawnSync(executable, [], {encoding: "utf8", timeout: 30000});
    assert.equal(run.status, 0, run.error?.message || run.stderr);
  });
});

/** @param {Uint8Array} bytes */
const arrayBuffer = (bytes) => Uint8Array.from(bytes).buffer;
const hooks = ["__momHashCreate", "__momRandomBytes", "__momGzip", "__momReadFile", "__momFileExists",
  "__momReadDirectory", "__momWrite", "__momWriteFile", "__momChmod", "__momNextTick", "__momOutputOpen"];
/** @type {unknown[]} */
let saved;
test.beforeEach(() => {saved = hooks.map((name) => Reflect.get(globalThis, name));});
test.afterEach(() => {
  hooks.forEach((name, index) => {
    if (saved[index] === undefined) {Reflect.deleteProperty(globalThis, name);} else {Reflect.set(globalThis, name, saved[index]);}
  });
});

test.describe("Native configuration-file writes", {concurrency: false}, () => {
  test("default text writes use UTF-8 and Node's creation mode", () => {
    globalThis.__momWriteFile = (file, data, mode, exclusive) => {
      assert.equal(file, "config.json"); assert.equal(Buffer.from(data).toString(), "é-💎");
      assert.equal(mode, 0o666); assert.equal(exclusive, false);
    };
    assert.equal(nativeFs.writeFileSync("config.json", "é-💎"), undefined);
  });
  test("the unchanged controller's private config mode reaches native open", () => {
    globalThis.__momWriteFile = (_file, _data, mode) => {assert.equal(mode, 0o600);};
    nativeFs.writeFileSync("config.json", "{}", {mode: 0o600});
  });
  test("Buffer and DataView writes preserve offset/length without text conversion", () => {
    globalThis.__momWriteFile = (_file, data) => {assert.equal(Buffer.from(data).toString("hex"), "00ff80");};
    const value = Buffer.from([1, 0, 255, 128, 2]);
    nativeFs.writeFileSync("binary", value.subarray(1, 4));
    nativeFs.writeFileSync("binary", new DataView(value.buffer, value.byteOffset + 1, 3));
  });
  test("string encoding options are applied before native open", () => {
    globalThis.__momWriteFile = (_file, data) => {assert.equal(Buffer.from(data).toString("hex"), "00ff80");};
    nativeFs.writeFileSync("binary", "00ff80", "hex");
    nativeFs.writeFileSync("binary", "AP+A", {encoding: "base64"});
  });
  test("empty writes still create or truncate a file", () => {
    let writes = 0;
    globalThis.__momWriteFile = (_file, data) => {assert.equal(data.length, 0); ++writes;};
    nativeFs.writeFileSync("empty", ""); nativeFs.writeFileSync("empty", Buffer.alloc(0));
    assert.equal(writes, 2);
  });
  test("NUL and invalid filenames fail before any write", () => {
    globalThis.__momWriteFile = () => {assert.fail("unexpected write");};
    for (const file of ["bad\0path", Buffer.from("bad\0path"), {}, null]) {
      assert.throws(() => nativeFs.writeFileSync(file, "value"), /File path/);
    }
  });
  test("invalid bytes and unsupported encodings cannot truncate an existing file", () => {
    globalThis.__momWriteFile = () => {assert.fail("unexpected write");};
    for (const value of [{}, null, undefined, 42, {buffer: new ArrayBuffer(1), byteLength: 1}]) {
      assert.throws(() => nativeFs.writeFileSync("existing", value), TypeError);
    }
    assert.throws(() => nativeFs.writeFileSync("existing", "value", "unknown-encoding"), /encoding/i);
  });
  test("octal string modes retain Node's public mode representation", () => {
    globalThis.__momWriteFile = (_file, _data, mode) => {assert.equal(mode, 0o640);};
    nativeFs.writeFileSync("file", "data", {mode: "0640"});
  });
  test("malformed and overflowing modes fail before native open", () => {
    globalThis.__momWriteFile = () => {assert.fail("unexpected write");};
    for (const mode of [-1, 1.5, NaN, Infinity, 0x100000000, "0o600", "888", "", true, {}]) {
      assert.throws(() => nativeFs.writeFileSync("file", "data", {mode}), /mode/i);
    }
  });
  test("zero mode is not replaced by a permissive default", () => {
    globalThis.__momWriteFile = (_file, _data, mode) => {assert.equal(mode, 0);};
    nativeFs.writeFileSync("file", "data", {mode: 0});
  });
  test("exclusive creation is not silently downgraded to overwrite", () => {
    globalThis.__momWriteFile = (_file, _data, _mode, exclusive) => {assert.equal(exclusive, true);};
    nativeFs.writeFileSync("file", "data", {flag: "wx"});
  });
  test("unimplemented flags fail explicitly before native I/O", () => {
    globalThis.__momWriteFile = () => {assert.fail("unexpected write");};
    for (const flag of ["a", "r+", "invalid", 0, {}]) {
      assert.throws(() => nativeFs.writeFileSync("file", "data", {flag}), /flag/i);
    }
  });
  test("invalid option types fail before native I/O", () => {
    globalThis.__momWriteFile = () => {assert.fail("unexpected write");};
    for (const options of [1, false, () => undefined]) {
      assert.throws(() => nativeFs.writeFileSync("file", "data", options), /options/i);
    }
  });
  test("native write errors preserve their OS error code", () => {
    globalThis.__momWriteFile = () => {throw Object.assign(new Error("denied"), {code: "EACCES"});};
    assert.throws(() => nativeFs.writeFileSync("file", "data"), {code: "EACCES"});
  });
  test("chmod accepts numeric and octal string modes without changing the path", () => {
    /** @type {number[]} */
    const modes = [];
    globalThis.__momChmod = (file, mode) => {assert.equal(file, "é-💎"); modes.push(mode);};
    nativeFs.chmodSync("é-💎", 0o600); nativeFs.chmodSync("é-💎", "0640");
    assert.deepEqual(modes, [0o600, 0o640]);
  });
  test("invalid chmod mode or path cannot reach the OS", () => {
    globalThis.__momChmod = () => {assert.fail("unexpected chmod");};
    for (const mode of [undefined, null, -1, 1.5, Infinity, 0x100000000, "bad"]) {
      assert.throws(() => nativeFs.chmodSync("file", mode), /mode/i);
    }
    assert.throws(() => nativeFs.chmodSync("file\0tail", 0o600), /NUL/);
  });
  test("chmod does not swallow OS permission errors", () => {
    globalThis.__momChmod = () => {throw Object.assign(new Error("denied"), {code: "EPERM"});};
    assert.throws(() => nativeFs.chmodSync("file", 0o600), {code: "EPERM"});
  });
});

test.describe("Native filesystem admission", {concurrency: false}, () => {
  test("text reads decode actual native bytes", () => {
    globalThis.__momReadFile = (name) => {assert.equal(name, "é/💎"); return arrayBuffer(Buffer.from("value-é"));};
    assert.equal(nativeFs.readFileSync("é/💎", "utf8"), "value-é");
  });
  test("omitted read encoding returns a Buffer without text conversion", () => {
    globalThis.__momReadFile = () => arrayBuffer(Buffer.from([0, 255, 128]));
    const result = nativeFs.readFileSync("binary"); assert.equal(Buffer.isBuffer(result), true);
    assert.equal(result.toString("hex"), "00ff80");
  });
  test("read options preserve null and explicit encoding", () => {
    globalThis.__momReadFile = () => arrayBuffer(Buffer.from([65]));
    assert.equal(nativeFs.readFileSync("file", {encoding: "hex"}), "41");
    assert.equal(Buffer.isBuffer(nativeFs.readFileSync("file", null)), true);
  });
  test("Buffer filenames use only their actual view", () => {
    globalThis.__momReadFile = (name) => {assert.equal(name, "file"); return new ArrayBuffer(0);};
    nativeFs.readFileSync(Buffer.from("!file!").subarray(1, 5));
  });
  test("NUL paths fail before calling any native read", () => {
    globalThis.__momReadFile = () => {assert.fail("unexpected filesystem access");};
    assert.throws(() => nativeFs.readFileSync("file\0tail"), {code: "ERR_INVALID_ARG_VALUE"});
    assert.throws(() => nativeFs.readFileSync(Buffer.from("file\0tail")), {code: "ERR_INVALID_ARG_VALUE"});
  });
  test("non-path inputs cannot be stringified into filesystem access", () => {
    assert.throws(() => nativeFs.readFileSync({toString: () => "file"}), {code: "ERR_INVALID_ARG_TYPE"});
  });
  test("OS read errors retain their code", () => {
    globalThis.__momReadFile = () => {throw Object.assign(new Error("denied"), {code: "EACCES"});};
    assert.throws(() => nativeFs.readFileSync("private"), {code: "EACCES"});
  });
  test("existsSync returns false for missing, invalid and inaccessible paths", () => {
    globalThis.__momFileExists = (name) => name === "present";
    assert.equal(nativeFs.existsSync("present"), true); assert.equal(nativeFs.existsSync("missing"), false);
    assert.equal(nativeFs.existsSync("present\0ignored"), false); assert.equal(nativeFs.existsSync(null), false);
    globalThis.__momFileExists = () => {throw new Error("denied");}; assert.equal(nativeFs.existsSync("private"), false);
  });
  test("directory names are returned unchanged", () => {
    globalThis.__momReadDirectory = (name) => {assert.equal(name, "dir"); return ["a", "é-💎"];};
    assert.deepEqual(nativeFs.readdirSync("dir"), ["a", "é-💎"]);
  });
  test("writeSync validates descriptors and sends exact bytes", () => {
    globalThis.__momWrite = (fd, bytes) => {assert.equal(fd, 2); assert.equal(Buffer.from(bytes).toString(), "é"); return bytes.length;};
    assert.equal(nativeFs.writeSync(2, "é"), 2);
    assert.throws(() => nativeFs.writeSync(-1, "bad"), {code: "ERR_OUT_OF_RANGE"});
    assert.throws(() => nativeFs.writeSync(1.5, "bad"), {code: "ERR_OUT_OF_RANGE"});
  });
});

test.describe("Native crypto and compression admission", {concurrency: false}, () => {
  test.beforeEach(() => {
    globalThis.__momHashCreate = () => {
      const hash = nodeCrypto.createHash("sha256");
      return {update(data) {hash.update(data);}, digest: () => arrayBuffer(hash.digest())};
    };
    globalThis.__momRandomBytes = (size) => arrayBuffer(Buffer.alloc(size, 0xa5));
    globalThis.__momGzip = (data) => arrayBuffer(nodeZlib.gzipSync(data));
  });
  test("incremental SHA-256 and encodings retain the standard digest", () => {
    assert.equal(crypto.createHash("SHA256").update("a").update("6263", "hex").digest("hex"),
      nodeCrypto.createHash("sha256").update("abc").digest("hex"));
  });
  test("Buffer digest output remains usable as another hash input", () => {
    const first = crypto.createHash("sha256").update("abc").digest();
    assert.equal(Buffer.isBuffer(first), true);
    assert.equal(crypto.createHash("sha-256").update(first).digest("hex"),
      nodeCrypto.createHash("sha256").update(nodeCrypto.createHash("sha256").update("abc").digest()).digest("hex"));
  });
  test("typed-array and DataView offsets are preserved", () => {
    const bytes = Uint8Array.from([0, 1, 2, 3, 4]);
    assert.equal(crypto.createHash("sha256").update(new DataView(bytes.buffer, 1, 3)).digest("hex"),
      nodeCrypto.createHash("sha256").update(bytes.subarray(1, 4)).digest("hex"));
  });
  test("non-byte objects cannot forge hash inputs", () => {
    assert.throws(() => crypto.createHash("sha256").update({buffer: new ArrayBuffer(1), byteLength: 1}), TypeError);
  });
  test("unsupported or malformed algorithms fail before native hash allocation", () => {
    globalThis.__momHashCreate = () => {assert.fail("unexpected hash allocation");};
    assert.throws(() => crypto.createHash(1), {code: "ERR_INVALID_ARG_TYPE"});
    assert.throws(() => crypto.createHash("md5"), /Unsupported/);
  });
  test("finalization errors propagate from the native state", () => {
    const hash = crypto.createHash("sha256"); hash.digest();
    assert.throws(() => hash.digest(), {code: "ERR_CRYPTO_HASH_FINALIZED"});
    assert.throws(() => hash.update("late"), /finalized|Digest already/);
  });
  test("random byte buffers use the secure bridge exactly", () => {
    assert.equal(crypto.randomBytes(4).toString("hex"), "a5a5a5a5"); assert.equal(crypto.randomBytes(0).length, 0);
  });
  test("invalid random sizes cannot allocate native memory", () => {
    globalThis.__momRandomBytes = () => {assert.fail("unexpected random allocation");};
    for (const value of [-1, 1.5, NaN, Infinity, 0x80000000]) {
      assert.throws(() => crypto.randomBytes(value), {code: "ERR_OUT_OF_RANGE"});
    }
    assert.throws(() => crypto.randomBytes("4"), {code: "ERR_INVALID_ARG_TYPE"});
  });
  test("entropy failures never fall back to insecure bytes", () => {
    globalThis.__momRandomBytes = () => {throw new Error("entropy unavailable");};
    assert.throws(() => crypto.randomBytes(4), /entropy unavailable/);
  });
  test("gzip returns a valid stream for empty and offset inputs", () => {
    assert.equal(nodeZlib.gunzipSync(zlib.gzipSync("")).length, 0);
    assert.equal(nodeZlib.gunzipSync(zlib.gzipSync(Buffer.from("!raw!").subarray(1, 4))).toString(), "raw");
  });
  test("gzip rejects forged byte objects", () => {
    assert.throws(() => zlib.gzipSync({buffer: new ArrayBuffer(1), byteLength: 1}), TypeError);
  });
  test("compressor errors are surfaced rather than sending an uncompressed proof silently", () => {
    globalThis.__momGzip = () => {throw new Error("compression failed");};
    assert.throws(() => zlib.gzipSync("raw"), /compression failed/);
  });
});

test.describe("Native standard-output ordering", {concurrency: false}, () => {
  /** @type {Array<{fd: number, bytes: Buffer}>} */
  let written;
  /** @type {Array<() => unknown>} */
  let ticks;
  /** @type {ReturnType<typeof nativeProcess>} */
  let process;
  test.beforeEach(() => {
    written = []; ticks = [];
    globalThis.__momOutputOpen = () => null;
    globalThis.__momWrite = (fd, bytes) => {written.push({fd, bytes: Buffer.from(bytes)}); return bytes.length;};
    globalThis.__momNextTick = (callback) => {ticks.push(callback);};
    process = nativeProcess({execPath: "/mom", arguments: [], platform: "linux", arch: "x64", pid: 1, ppid: 1});
  });
  test("stdout uses fd 1 with exact UTF-8 bytes", () => {
    assert.equal(process.stdout.write("é-💎"), true); assert.equal(written[0]?.fd, 1);
    assert.equal(written[0]?.bytes.toString(), "é-💎");
  });
  test("stderr remains separate on fd 2", () => {process.stderr.write("error"); assert.equal(written[0]?.fd, 2);});
  test("Buffer slices never include parent bytes", () => {
    process.stdout.write(Buffer.from("!raw!").subarray(1, 4)); assert.equal(written[0]?.bytes.toString(), "raw");
  });
  test("explicit string encodings are applied before writing", () => {
    process.stdout.write("ff00", "hex"); assert.equal(written[0]?.bytes.toString("hex"), "ff00");
  });
  test("partial synchronous writes finish before a later message", () => {
    /** @type {Array<number | undefined>} */
    const bytes = [];
    globalThis.__momWrite = (_fd, value) => {bytes.push(value[0]); return 1;};
    process.stdout.write("abc"); process.stdout.write("d"); assert.deepEqual(bytes, [97, 98, 99, 100]);
  });
  test("write callbacks are deferred and preserve order", () => {
    /** @type {number[]} */
    const order = [];
    process.stdout.write("a", () => order.push(1)); process.stdout.write("b", "utf8", () => order.push(2));
    assert.deepEqual(order, []); for (const tick of ticks) {tick();} assert.deepEqual(order, [1, 2]);
  });
  test("empty writes still schedule the shutdown flush callback", () => {
    let done = false; process.stdout.write("", () => {done = true;});
    assert.equal(written.length, 0); assert.equal(done, false); ticks[0]?.(); assert.equal(done, true);
  });
  test("zero-progress writes cannot spin forever", () => {
    globalThis.__momWrite = () => 0; assert.throws(() => process.stdout.write("a"), /no progress/);
  });
  test("write failures cannot report success or a completed callback", () => {
    globalThis.__momWrite = () => {throw Object.assign(new Error("closed"), {code: "EPIPE"});};
    assert.throws(() => process.stdout.write("a", () => assert.fail("unexpected completion")), {code: "EPIPE"});
    assert.equal(ticks.length, 0);
  });
  test("embedded NUL is output data, not a path terminator", () => {
    process.stdout.write("a\0b"); assert.equal(written[0]?.bytes.toString("hex"), "610062");
  });
});
