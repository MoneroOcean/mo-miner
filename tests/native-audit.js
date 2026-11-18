"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {nativeDependencies, auditNativeGlibc, auditNativeTrace, auditNativeHeaders, auditNativeSymbols, auditNativeSymbolFile, isStaticArchive} = require("../scripts/native-audit");

/** @param {import('node:test').TestContext} t */
function auditFixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-audit-"));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  return path.join(directory, "evidence");
}

test.describe("Native static archive signature admission", () => {
  for (const content of ["!<arch>\n", "!<arch>\npayload"]) {
    test(`accepts a regular archive signature with ${content.length - 8} payload bytes`, (t) => {
      const file = auditFixture(t); fs.writeFileSync(file, content); assert.equal(isStaticArchive(file), true);
    });
  }
  for (const content of ["", "!<arch>", "!<thin>\n", "\x7fELF", "!<arch>\r"]) {
    test(`rejects signature ${JSON.stringify(content)}`, (t) => {
      const file = auditFixture(t); fs.writeFileSync(file, content); assert.equal(isStaticArchive(file), false);
    });
  }
  test("high-bit bytes cannot alias the ASCII archive marker", (t) => {
    const file = auditFixture(t);
    for (let index = 0; index < 8; index++) {
      const bytes = Buffer.from("!<arch>\n"); bytes[index] = (bytes[index] || 0) | 128;
      fs.writeFileSync(file, bytes); assert.equal(isStaticArchive(file), false);
    }
  });
  test("rejects a non-regular directory", (t) => {
    let accepted = false;
    try {accepted = isStaticArchive(path.dirname(auditFixture(t)));} catch (error) {
      assert.ok(["EISDIR", "EACCES", "EPERM"].includes(/** @type {NodeJS.ErrnoException} */ (error).code || ""));
    }
    assert.equal(accepted, false);
  });
  test("a POSIX FIFO cannot block archive discovery", {skip: process.platform === "win32"}, (t) => {
    const file = auditFixture(t), result = spawnSync("mkfifo", [file], {encoding: "utf8", timeout: 5000});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
    assert.equal(isStaticArchive(file), false);
  });
  test("admission reads only eight bytes and never the whole payload", (t) => {
    const file = auditFixture(t); fs.writeFileSync(file, "!<arch>\npayload");
    const read = t.mock.method(fs, "readSync");
    t.mock.method(fs, "readFileSync", () => {throw new Error("whole-file archive read");});
    try {
      assert.equal(isStaticArchive(file), true); assert.equal(read.mock.callCount(), 1);
      assert.deepEqual(read.mock.calls[0]?.arguments.slice(2), [0, 8, 0]);
    } finally {t.mock.restoreAll();}
  });
  test("short reads fail closed and close their descriptor", (t) => {
    const file = auditFixture(t); fs.writeFileSync(file, "!<arch>\n");
    const close = t.mock.method(fs, "closeSync"); t.mock.method(fs, "readSync", () => 7);
    try {assert.equal(isStaticArchive(file), false); assert.equal(close.mock.callCount(), 1);} finally {t.mock.restoreAll();}
  });
  test("read errors propagate after closing the descriptor", (t) => {
    const file = auditFixture(t); fs.writeFileSync(file, "!<arch>\n");
    const close = t.mock.method(fs, "closeSync"); t.mock.method(fs, "readSync", () => {throw new Error("fixture read");});
    try {assert.throws(() => isStaticArchive(file), /fixture read/); assert.equal(close.mock.callCount(), 1);} finally {t.mock.restoreAll();}
  });
  test("missing input is not accepted", (t) => {assert.throws(() => isStaticArchive(auditFixture(t)), /ENOENT/);});
});

test.describe("Native linked factory completeness", () => {
  const name = "mom_native_gpu_oneapi";
  for (const type of ["T", "W"]) {
    test(`accepts a real global ${type} factory definition`, (t) => {
      const file = auditFixture(t); fs.writeFileSync(file, `0123 ${type} ${name}\n`);
      assert.equal(auditNativeSymbolFile(file, [name]).lines, 1);
    });
  }
  for (const text of [` U ${name}\n`, `0123 T ${name}_other\n`, `0123 t ${name}\n`]) {
    test(`refuses non-definition ${JSON.stringify(text.trim())}`, (t) => {
      const file = auditFixture(t); fs.writeFileSync(file, text);
      assert.throws(() => auditNativeSymbolFile(file, [name]), /Missing linked native symbols/);
    });
  }
  test("every required graph must exist, not just the first", (t) => {
    const file = auditFixture(t); fs.writeFileSync(file, `0123 T ${name}\n`);
    assert.throws(() => auditNativeSymbolFile(file, [name, "mom_native_gpu_dpcpp"]), /mom_native_gpu_dpcpp/);
  });
  test("Node/V8 scanning continues after all required factories are found", (t) => {
    const file = auditFixture(t); fs.writeFileSync(file, `0123 T ${name}\n0124 T node::Start()\n`);
    assert.throws(() => auditNativeSymbolFile(file, [name]), /Node\/V8\/N-API/);
  });
  test("the final unterminated factory is still checked", (t) => {
    const file = auditFixture(t); fs.writeFileSync(file, `0123 T ${name}`);
    assert.equal(auditNativeSymbolFile(file, [name]).lines, 1);
  });
  test("malformed, duplicate or excessive symbol requests fail before opening a file", () => {
    for (const values of [[""], ["bad\nname"], ["a", "a"], ["a".repeat(129)], Array.from({length: 65}, (_, i) => `x${i}`)]) {
      assert.throws(() => auditNativeSymbolFile("/missing-file", values), /Invalid required native symbols/);
    }
  });
  test("an unrelated valid table cannot pass an absent required factory", (t) => {
    const file = auditFixture(t); fs.writeFileSync(file, "0123 T unrelated\n");
    assert.throws(() => auditNativeSymbolFile(file, [name]), /Missing linked native symbols/);
  });
});

test.describe("Native streamed symbol audit", () => {
  const fixture = auditFixture;
  test("accepts a complete table beyond the old 64-MiB admission without a giant string", (t) => {
    const file = fixture(t), fd = fs.openSync(file, "wx"), block = Buffer.alloc(64 << 10, " ");
    for (let i = 511; i < block.length; i += 512) {block[i] = 10;}
    try {for (let i = 0; i < 1025; i++) {fs.writeSync(fd, block);}} finally {fs.closeSync(fd);}
    assert.deepEqual(auditNativeSymbolFile(file), {bytes: 1025 * block.length, lines: 1025 * 128});
  });
  for (const symbol of ["node::Start", "v8::Isolate", "__imp_napi_create_object"]) {
    test(`rejects ${symbol} across a read boundary`, (t) => {
      const file = fixture(t), prefix = " \n".repeat(((64 << 10) - 4) / 2);
      fs.writeFileSync(file, prefix + symbol + "\n");
      assert.throws(() => auditNativeSymbolFile(file), /Node\/V8\/N-API/);
    });
  }
  test("rejects a forbidden final symbol without a newline", (t) => {
    const file = fixture(t); fs.writeFileSync(file, "0000 T safe\n0001 T napi_create_object");
    assert.throws(() => auditNativeSymbolFile(file), /Node\/V8\/N-API/);
  });
  test("preserves UTF-8 boundaries and legitimate nested node classes", (t) => {
    const file = fixture(t), text = " \n".repeat(((64 << 10) - 2) / 2) + "xé sycl::graph::node::run()\n";
    fs.writeFileSync(file, text);
    assert.equal(auditNativeSymbolFile(file).bytes, Buffer.byteLength(text));
  });
  for (const suffix of ["", "\n"]) {
    test(`rejects an oversized ${suffix ? "complete" : "unfinished"} line`, (t) => {
      const file = fixture(t); fs.writeFileSync(file, " ".repeat((64 << 10) + 1) + suffix);
      assert.throws(() => auditNativeSymbolFile(file), /symbol line/);
    });
  }
  test("rejects an excessive sparse file before reading it", (t) => {
    const file = fixture(t); fs.writeFileSync(file, ""); fs.truncateSync(file, (512 << 20) + 1);
    assert.throws(() => auditNativeSymbolFile(file), /excessive native symbol data/);
  });
  test("rejects missing, empty and non-regular evidence", (t) => {
    const file = fixture(t);
    assert.throws(() => auditNativeSymbolFile(file), /ENOENT/);
    fs.writeFileSync(file, ""); assert.throws(() => auditNativeSymbolFile(file), /Missing/);
    assert.throws(() => auditNativeSymbolFile(path.dirname(file)), /Missing|EISDIR/);
  });
  test("counts CRLF rows and the final unterminated row", (t) => {
    const file = fixture(t), text = "0000 T safe\r\n0001 W sycl::node::run()";
    fs.writeFileSync(file, text);
    assert.deepEqual(auditNativeSymbolFile(file), {bytes: Buffer.byteLength(text), lines: 2});
  });
  for (const field of ["size", "mtimeMs", "ctimeMs"]) {
    test(`rejects a concurrent ${field} change even when the scanned byte count matches`, (t) => {
      const file = fixture(t); fs.writeFileSync(file, "0000 T safe\n");
      const stat = fs.statSync(file), changed = Object.assign({}, stat);
      Reflect.set(changed, field, Reflect.get(stat, field) + 1);
      let calls = 0;
      t.mock.method(fs, "fstatSync", () => ++calls === 1 ? stat : changed);
      try {assert.throws(() => auditNativeSymbolFile(file), /changed during audit/);} finally {t.mock.restoreAll();}
    });
  }
});

test.describe("Native Linux deployment ABI baseline", () => {
  test("accepts old numeric glibc symbol versions", () => {
    assert.deepEqual(auditNativeGlibc("Name: GLIBC_2.2.5\nName: GLIBC_2.34\n", "2.39"), ["2.2.5", "2.34"]);
  });
  test("accepts the exact selected baseline", () => {assert.deepEqual(auditNativeGlibc("Name: GLIBC_2.39", "2.39"), ["2.39"]);});
  test("records the glibc RELR ABI feature without treating it as a numeric version", () => {
    assert.deepEqual(auditNativeGlibc("Name: GLIBC_2.38\nName: GLIBC_ABI_DT_RELR", "2.39"), ["2.38", "ABI_DT_RELR"]);
    assert.deepEqual(auditNativeGlibc("Name: GLIBC_ABI_DT_RELR", "2.36"), ["ABI_DT_RELR"]);
  });
  test("RELR ABI admission requires the glibc release that introduced DT_RELR", () => {
    assert.throws(() => auditNativeGlibc("Name: GLIBC_2.34\nName: GLIBC_ABI_DT_RELR", "2.35"), /GLIBC_ABI_DT_RELR requires glibc >= 2\.36/);
  });
  test("rejects a new libm reference even though its SONAME is unchanged", () => {
    nativeDependencies("(NEEDED) [libm.so.6]");
    assert.throws(() => auditNativeGlibc("Name: GLIBC_2.43", "2.39"), /2.43 exceeds.*2.39/);
  });
  test("compares numeric components, not lexical strings", () => {
    auditNativeGlibc("Name: GLIBC_2.9", "2.39");
    assert.throws(() => auditNativeGlibc("Name: GLIBC_2.100", "2.39"), /exceeds/);
  });
  test("handles patch-level baselines explicitly", () => {
    auditNativeGlibc("Name: GLIBC_2.2.5", "2.2.5");
    assert.throws(() => auditNativeGlibc("Name: GLIBC_2.2.5", "2.2.4"), /exceeds/);
  });
  test("a major version increase cannot bypass admission", () => {
    assert.throws(() => auditNativeGlibc("Name: GLIBC_3.0", "2.99"), /exceeds/);
  });
  test("deduplicates requirements without confusing other version namespaces", () => {
    assert.deepEqual(auditNativeGlibc("Name: GLIBC_2.34\nName: GLIBCXX_3.4\nName: GLIBC_2.34", "2.39"), ["2.34"]);
  });
  test("fully static fixtures can have no version requirements", () => {assert.deepEqual(auditNativeGlibc("No version information found in this file.", "2.39"), []);});
  test("private or unrecognized glibc version labels fail closed", () => {
    for (const name of ["PRIVATE", "ABI_NEW", "ABI_DT_RELR_BROKEN", "2.39.0.1", "2.x"]) {
      assert.throws(() => auditNativeGlibc("Name: GLIBC_" + name, "2.39"), /Invalid native glibc/);
    }
  });
  test("requires an explicit bounded numeric baseline", () => {
    for (const value of ["", "latest", "2", "2.39\n", "9999.1"]) {assert.throws(() => auditNativeGlibc("", value), /baseline/);}
  });
  test("empty or truncated tool output cannot prove a deployment ABI floor", () => {
    for (const value of ["", "Version needs section", "Name: GLIBC_2."]) {
      assert.throws(() => auditNativeGlibc(value, "2.39"), /Missing native ELF|Invalid native glibc/);
    }
  });
  test("uses the glibc-compatible linker option, not the loader-incomplete generic spelling", () => {
    const source = fs.readFileSync(path.join(__dirname, "../scripts/build-native-application.js"), "utf8");
    assert.match(source, /"-Wl,-z,pack-relative-relocs"/);
    assert.doesNotMatch(source, /--pack-dyn-relocs=relr/);
  });
});

test.describe("Native demangled symbol audit", () => {
  test("maintained SYCL command-graph node classes are not Node.js", () => {
    auditNativeSymbols("0000549020 T void sycl::_V1::ext::oneapi::experimental::node::update_range<1>(sycl::_V1::range<1>)\n");
  });
  test("AdaptiveCpp's HCF node classes are not Node.js", () => {auditNativeSymbols("0 W hipsycl::common::hcf_container::node::get_subnodes() const\n");});
  for (const symbol of ["node::Start(int, char**)", "vtable for node::Environment", "void run(node::Environment*)",
    "node::Environment::~Environment()", "v8::Isolate::New()", "void v8::HandleScope::Initialize()", "std::vector<v8::Value*>",
    "napi_create_function", "napi_call_function", "__imp_napi_get_value_string_utf8"]) {
    test(`rejects ${symbol}`, () => {assert.throws(() => auditNativeSymbols("00000000 T " + symbol + "\n"), /Node\/V8\/N-API/);});
  }
  test("symbol size admission cannot silently truncate a large archive", () => {
    assert.throws(() => auditNativeSymbols(" ".repeat((64 << 20) + 1)), /excessive/);
  });
});

test.describe("Native dependency and lazy-load audit", () => {
  const start = '100 execve("/mom", ["/mom"], 0x1) = 0\n';
  /** @param {string} file */
  const open = (file) => `100 openat(AT_FDCWD, "${file}", O_RDONLY) = 3\n`;
  test("accepts OS C/math/loader dependencies", () => {
    assert.deepEqual(nativeDependencies("(NEEDED) [libc.so.6]\n(NEEDED) [libm.so.6]\n(NEEDED) [ld-linux-x86-64.so.2]"),
      ["libc.so.6", "libm.so.6", "ld-linux-x86-64.so.2"]);
  });
  test("rejects an explicit compiler runtime dependency", () => {assert.throws(() => nativeDependencies("(NEEDED) [libgcc_s.so.1]"), /dependency/);});
  test("rejects a SYCL application runtime", () => {assert.throws(() => nativeDependencies("(NEEDED) [libacpp-rt.so]"), /dependency/);});
  test("a GPU build must explicitly admit the system CUDA driver", () => {
    assert.throws(() => nativeDependencies("(NEEDED) [libcuda.so.1]"), /dependency/);
    assert.deepEqual(nativeDependencies("(NEEDED) [libcuda.so.1]", {cudaDriver: true}), ["libcuda.so.1"]);
  });
  test("allowing the driver never admits application CUDA or compiler runtimes", () => {
    for (const name of ["libcudart.so.12", "libnvrtc.so.12", "libLLVM.so.21", "libacpp-rt.so"]) {
      assert.throws(() => nativeDependencies(`(NEEDED) [${name}]`, {cudaDriver: true}), /dependency/);
    }
  });
  test("the CUDA trace exception is limited to observed system driver components", () => {
    const trace = start + open("/usr/lib/libcuda.so.1") + open("/usr/lib/libnvidia-ptxjitcompiler.so.1") + open("/usr/lib/libnvidia-ml.so.1");
    assert.throws(() => auditNativeTrace(trace, "/mom"), /external runtime/);
    auditNativeTrace(trace, "/mom", 1, {cudaDriver: true});
    assert.throws(() => auditNativeTrace(trace + open("/sdk/libnvrtc.so.12"), "/mom", 1, {cudaDriver: true}), /external runtime/);
  });
  test("Level Zero admits only its explicitly selected GPU driver identities, not an application loader", () => {
    for (const name of ["libze_intel_gpu.so.1", "libze_intel_gpu_legacy1.so.1"]) {
      assert.throws(() => nativeDependencies(`(NEEDED) [${name}]`), /dependency/);
      assert.deepEqual(nativeDependencies(`(NEEDED) [${name}]`, {levelZeroDriver: true}), [name]);
      auditNativeTrace(start + open("/usr/lib/" + name), "/mom", 1, {levelZeroDriver: true});
    }
    for (const name of ["libze_loader.so.1", "libur_adapter_level_zero.so.0", "libigc.so.2", "libintelocl.so", "libze_intel_npu.so.1"]) {
      assert.throws(() => nativeDependencies(`(NEEDED) [${name}]`, {levelZeroDriver: true}), /dependency/);
      assert.throws(() => auditNativeTrace(start + open("/sdk/" + name), "/mom", 1, {levelZeroDriver: true}), /external runtime/);
    }
  });
  test("POSIX realtime services are OS libraries, not application runtime exemptions", () => {
    assert.deepEqual(nativeDependencies("(NEEDED) [librt.so.1]"), ["librt.so.1"]);
    auditNativeTrace(start + open("/usr/lib/librt.so.1"), "/mom");
  });
  test("does not permit search paths into a development installation", () => {assert.throws(() => nativeDependencies("(RUNPATH) [/opt/mom]"), /search path/);});
  test("does not confuse the OS loader cache with executable library code", () => {
    auditNativeTrace(start + open("/etc/ld.so.cache") + open("/lib/libc.so.6") + open("/proc/self/status"), "/mom");
  });
  test("detects runtime libgcc loads absent from ELF NEEDED entries", () => {
    assert.throws(() => auditNativeTrace(start + open("/lib/libgcc_s.so.1"), "/mom"), /external runtime/);
  });
  test("rejects an unsuccessful optional plugin probe as well", () => {
    assert.throws(() => auditNativeTrace(start + '100 openat(AT_FDCWD, "/opt/plugin.so", O_RDONLY) = -1 ENOENT\n', "/mom"), /external runtime/);
  });
  test("rejects an interpreted/native addon loaded at runtime", () => {
    assert.throws(() => auditNativeTrace(start + open("/build/mom.node"), "/mom"), /external runtime/);
  });
  test("rejects external compiler execution", () => {
    assert.throws(() => auditNativeTrace(start + '101 execve("/usr/bin/llc", [], 0x1) = 0\n', "/mom", 2), /worker launch/);
  });
  test("allows only the exact same image for an explicitly counted worker lane", () => {
    auditNativeTrace(start + start.replace("100", "101"), "/mom", 2);
    assert.throws(() => auditNativeTrace(start, "/mom", 2), /worker launch/);
  });
  test("missing or truncated trace data cannot pass", () => {
    assert.throws(() => auditNativeTrace("", "/mom"), /worker launch/);
    assert.throws(() => auditNativeTrace('execve("/mom"..., [], 0x1) = 0\n', "/mom"), /worker launch/);
    assert.throws(() => auditNativeTrace(start + 'execveat(4, "", [], 0x1, AT_EMPTY_PATH) = 0\n', "/mom"), /worker launch/);
  });
  test("a fully static binary can have no ELF dynamic section", () => {assert.deepEqual(nativeDependencies("There is no dynamic section."), []);});
});

test.describe("Native OpenCL GPU ICD audit boundaries", () => {
  const start = '100 execve("/mom", ["/mom"], 0x1) = 0\n';
  const open = (/** @type {string} */ file) => `100 openat(AT_FDCWD, "${file}", O_RDONLY) = 3\n`;
  for (const name of ["libigdrcl.so", "libigdrcl_legacy1.so", "libnvidia-opencl.so.1", "libamdocl64.so", "libamdocl64.so.2", "libRusticlOpenCL.so.1"]) {
    test(`${name} requires explicit OpenCL driver admission`, () => {
      assert.throws(() => nativeDependencies(`(NEEDED) [${name}]`), /dependency/);
      assert.deepEqual(nativeDependencies(`(NEEDED) [${name}]`, {openclDriver: true}), [name]);
      auditNativeTrace(start + open("/usr/lib/x86_64-linux-gnu/intel-opencl/" + name), "/mom", 1, {openclDriver: true});
    });
  }
  test("ICD admission does not permit an external ICD loader, application runtime or compiler", () => {
    for (const name of ["libOpenCL.so.1", "libur_adapter_opencl.so.0", "libintelocl.so", "libigc.so.2", "libLLVM.so.21", "libstdc++.so.6", "libgcc_s.so.1"]) {
      assert.throws(() => nativeDependencies(`(NEEDED) [${name}]`, {openclDriver: true}), /dependency/);
      assert.throws(() => auditNativeTrace(start + open("/usr/lib/" + name), "/mom", 1, {openclDriver: true}), /external runtime/);
    }
  });
  test("absolute ICD paths must remain within normalized OS library roots", () => {
    for (const prefix of ["/usr/lib/", "/usr/lib64/", "/lib/", "/lib64/"]) {
      auditNativeTrace(start + open(prefix + "libigdrcl.so"), "/mom", 1, {openclDriver: true});
    }
    for (const prefix of ["/tmp/", "/opt/sdk/", "../", "/usr/lib/../tmp/", "/usr/lib//", "/usr/lib/./"]) {
      assert.throws(() => auditNativeTrace(start + open(prefix + "libigdrcl.so"), "/mom", 1, {openclDriver: true}), /external runtime/);
    }
  });
  test("ICD identity is exact, not a filename prefix or case-insensitive match", () => {
    for (const name of ["libigdrcl.so.extra", "libIGDRCL.so", "libnvidia-opencl.so.1.other"]) {
      assert.throws(() => nativeDependencies(`(NEEDED) [${name}]`, {openclDriver: true}), /dependency/);
    }
  });
  test("OpenCL admission cannot waive the independent Level Zero driver selection", () => {
    assert.throws(() => nativeDependencies("(NEEDED) [libze_intel_gpu.so.1]", {openclDriver: true}), /dependency/);
  });
});

test.describe("Native memory-only JIT include audit", () => {
  const open = (/** @type {string} */ file) => `100 openat(AT_FDCWD, "${file}", O_RDONLY) = -1 ENOENT\n`;
  test("OS files are not mistaken for JIT inputs", () => {auditNativeHeaders(open("/etc/ld.so.cache") + open("/lib/libc.so.6"));});
  test("extensionless CUDA standard headers are rejected", () => {assert.throws(() => auditNativeHeaders(open("/sdk/cuda/std/cstdint")), /disk header/);});
  test("absent SDK header probes fail too", () => {assert.throws(() => auditNativeHeaders(open("/sdk/cute/tensor.hpp")), /disk header/);});
  test("any file inside an explicit include root is an input", () => {assert.throws(() => auditNativeHeaders(open("/sdk/include/metadata"), ["/sdk/include"]), /disk header/);});
  test("root matching observes path boundaries", () => {auditNativeHeaders(open("/sdk/includes/output"), ["/sdk/include"]);});
  test("extensionless C++ and generic SDK include files cannot hide outside explicit roots", () => {
    for (const file of ["/usr/include/c++/13/vector", "/usr/include/CL/metadata", "/toolchain/include/c++/v1/array"]) {
      assert.throws(() => auditNativeHeaders(open(file)), /disk header/);
    }
  });
  test("explicit generated CUDA cache roots admit only non-source data", () => {auditNativeHeaders(open("/cache/cuda/index") + open("/cache/cuda/aa/image"), [], ["/cache/cuda"]);});
  test("cache names do not exempt header extensions", () => {assert.throws(() => auditNativeHeaders(open("/cache/cuda/tensor.hpp"), [], ["/cache/cuda"]), /disk header/);});
  test("precompiled headers are source inputs even without an include directory", () => {
    for (const file of ["/sdk/opencl.pch", "/cache/jit/opencl.pch", "/tmp/comgr-12345678/opencl.PCH"]) {
      assert.throws(() => auditNativeHeaders(open(file), [], ["/cache/jit", "/tmp/comgr-12345678"]), /disk header/);
    }
  });
  test("device libraries and intermediate code cannot be external compiler inputs", () => {
    for (const file of ["/sdk/libdevice.10.bc", "/sdk/libkernel.hcf", "/sdk/module.ll", "/sdk/module.spv", "/sdk/kernel.ptx", "/sdk/kernel.cubin", "/sdk/kernel.hsaco"]) {
      assert.throws(() => auditNativeHeaders(open(file)), /disk header/);
    }
  });
  test("explicit generated-code caches admit device intermediates, not source", () => {
    auditNativeHeaders(open("/cache/jit/module.bc") + open("/cache/jit/module.ptx"), [], ["/cache/jit"]);
    assert.throws(() => auditNativeHeaders(open("/cache/jit/source.cpp"), [], ["/cache/jit"]), /disk header/);
  });
  test("generated-code cache boundaries cannot admit nearby SDK bitcode", () => {
    assert.throws(() => auditNativeHeaders(open("/cache/jit-other/library.bc"), [], ["/cache/jit"]), /disk header/);
  });
  test("plain source and uppercase header suffixes also fail closed", () => {
    for (const file of ["/sdk/kernel.c", "/sdk/kernel.cc", "/sdk/kernel.cpp", "/sdk/kernel.cxx", "/sdk/kernel.cu", "/sdk/kernel.hip", "/sdk/Config.H", "/sdk/Header.HPP"]) {
      assert.throws(() => auditNativeHeaders(open(file)), /disk header/);
    }
  });
  test("explicit include roots still override generated-code cache admission", () => {
    assert.throws(() => auditNativeHeaders(open("/cache/jit/module.bc"), ["/cache/jit"], ["/cache/jit"]), /disk header/);
  });
  test("similar cache prefixes are not exempt", () => {assert.throws(() => auditNativeHeaders(open("/cache/cuda/other/cuda/std/cstdint"), [], ["/cache/cud"]), /disk header/);});
  test("include-root admission overrides a cache exemption", () => {assert.throws(() => auditNativeHeaders(open("/cache/cuda/include/cstdint"), ["/cache/cuda/include"], ["/cache/cuda"]), /disk header/);});
  test("escaping or escaped trace paths fail closed", () => {
    for (const file of ["/cache/cuda/../sdk", "/cache/cuda/./data", "/sdk/a\\x2eh"]) {assert.throws(() => auditNativeHeaders(open(file), [], ["/cache/cuda"]), /disk header/);}
  });
  test("hwloc's exact kernel NUMA probes are metadata, not a general traversal exemption", () => {
    for (const entry of [".", ".."]) {auditNativeHeaders(open(`/proc/driver/nvidia/gpus/${entry}/numa_status`));}
    for (const file of ["/proc/driver/nvidia/gpus/../../numa_status", "/proc/driver/nvidia/gpus/../input.bc",
      "/proc/driver/nvidia/gpus/../cuda.h", "/cache/driver/../numa_status", "/sdk/./numa_status"]) {
      assert.throws(() => auditNativeHeaders(open(file), [], ["/cache"]), /disk header/);
    }
    assert.throws(() => auditNativeHeaders(open("/proc/driver/nvidia/gpus/./numa_status"), ["/proc/driver/nvidia"]), /disk header/);
  });
  test("broad and unnormalized exemptions are rejected", () => {
    for (const root of ["/", "relative", "/cache/", "/cache/../sdk"]) {assert.throws(() => auditNativeHeaders("", [], [root]), /exact absolute/);}
  });
});

test.describe("Native syscall file-open coverage", () => {
  const start = '100 execve("/mom", ["/mom"], 0x1) = 0\n';
  /** @param {string} syscall @param {string} file */
  const line = (syscall, file) => {
    const flags = syscall === "creat" ? "0600" : syscall === "openat2" ? "{flags=O_RDONLY, mode=0}, 24" : "O_RDONLY";
    return `100 ${syscall}(${syscall.startsWith("openat") ? "AT_FDCWD, " : ""}"${file}", ${flags}) = 4\n`;
  };
  for (const syscall of ["open", "openat", "openat2", "creat"]) {
    test(`${syscall}: application libraries cannot bypass the syscall audit`, () => {
      assert.throws(() => auditNativeTrace(start + line(syscall, "/sdk/libLLVM.so.23"), "/mom"), /external runtime/);
    });
    test(`${syscall}: source cannot bypass the compiler-input audit`, () => {
      assert.throws(() => auditNativeHeaders(line(syscall, "/sdk/source.hpp")), /disk header/);
    });
    test(`${syscall}: device libraries cannot bypass the compiler-input audit`, () => {
      assert.throws(() => auditNativeHeaders(line(syscall, "/sdk/libdevice.10.bc")), /disk header/);
    });
    test(`${syscall}: escaped library names fail closed`, () => {
      assert.throws(() => auditNativeTrace(start + line(syscall, "/sdk/libLLVM\\x2eso"), "/mom"), /external runtime/);
    });
  }
});
