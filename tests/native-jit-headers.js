"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {gzipSync} = require("node:zlib"), {spawnSync} = require("node:child_process");
const {validName, packHeaders, collectHeaders, headerResources, cudaHeaderResources, cudaHeaderInputs} = require("../scripts/native-jit-headers");
const {tempDir} = require("./common/temp-dir");
const entry = (/** @type {string} */ name, data = "text") => ({name, data: Buffer.from(data)});

test.describe("Native JIT header generation", () => {
  test("canonical payloads sort includes without changing bytes", () => {
    assert.deepEqual(packHeaders([entry("b.h"), entry("a.h", "é-💎")]), packHeaders([entry("a.h", "é-💎"), entry("b.h")]));
  });
  test("gzip and generated source are reproducible", () => {assert.deepEqual(headerResources([entry("a.h")]), headerResources([entry("a.h")]));});
  test("header edits change the complete dependency identity", () => {assert.notEqual(headerResources([entry("a", "old")]).sha256, headerResources([entry("a", "new")]).sha256);});
  test("empty includes remain valid virtual files", () => {assert.ok(packHeaders([entry("empty", "")]).length);});
  test("NUL-containing include text cannot be silently truncated by NVRTC", () => {assert.throws(() => packHeaders([entry("a", "bad\0text")]), /NUL/);});
  test("Buffer subviews retain only their selected byte range", () => {
    assert.deepEqual(packHeaders([{name: "a", data: Buffer.from("before").subarray(1, 3)}]), packHeaders([entry("a", "ef")]));
  });
  test("rooted, escaping, empty, Windows and code-injection names are rejected", () => {
    for (const name of ["", "/a", "../a", "a/../b", "a/./b", "a//b", "a/", "C:/a", "a\\b", 'a"', "💎"]) {
      assert.equal(validName(name), false); assert.throws(() => packHeaders([entry(name)]), /name/);
    }
  });
  test("duplicate names cannot change search precedence", () => {assert.throws(() => packHeaders([entry("a"), entry("a")]), /duplicate/);});
  test("non-string names cannot run coercion hooks during sorting", () => {
    const name = /** @type {string} */ (/** @type {unknown} */ ({toString() {assert.fail("coercion");}}));
    assert.throws(() => packHeaders([entry("a"), {name, data: Buffer.from("x")}]), /name/);
  });
  test("oversized and empty include sets are rejected", () => {
    assert.throws(() => packHeaders([]), /count/);
    assert.throws(() => packHeaders(Array.from({length: 16385}, (_, i) => entry("a" + i))), /count/);
  });
  test("raw byte capacity is bounded before resource generation", () => {assert.throws(() => packHeaders([{name: "a", data: Buffer.alloc(64 << 20, 32)}]), /bounds/);});
  test("forged byte objects cannot supply source data", () => {
    assert.throws(() => packHeaders([{name: "a", data: /** @type {Buffer} */ (/** @type {unknown} */ ({length: 1}))}]), /buffers/);
  });
  test("only explicit include roots are collected", (t) => {
    const root = tempDir(t, "mom-jit-headers-"); fs.mkdirSync(path.join(root, "include"));
    fs.writeFileSync(path.join(root, "outside"), "not packed"); fs.writeFileSync(path.join(root, "include/a"), "packed");
    assert.deepEqual(collectHeaders([path.join(root, "include")]), [entry("a", "packed")]);
  });
  test("identical overlapping includes deduplicate, conflicting ones fail", (t) => {
    const root = tempDir(t, "mom-jit-headers-"), a = path.join(root, "a"), b = path.join(root, "b"); fs.mkdirSync(a); fs.mkdirSync(b);
    fs.writeFileSync(path.join(a, "file"), "same"); fs.writeFileSync(path.join(b, "file"), "same");
    assert.equal(collectHeaders([a, b]).length, 1); fs.writeFileSync(path.join(b, "file"), "different");
    assert.throws(() => collectHeaders([a, b]), /Conflicting/);
  });
  test("nested links cannot escape a root", {skip: process.platform === "win32"}, (t) => {
    const root = tempDir(t, "mom-jit-headers-"), include = path.join(root, "include"); fs.mkdirSync(include);
    fs.writeFileSync(path.join(root, "outside"), "secret"); fs.symlinkSync("../outside", path.join(include, "link"));
    assert.throws(() => collectHeaders([include]), /escapes/);
  });
  test("directory link cycles are rejected", {skip: process.platform === "win32"}, (t) => {
    const root = tempDir(t, "mom-jit-headers-"); fs.symlinkSync(".", path.join(root, "cycle")); assert.throws(() => collectHeaders([root]), /cycles/);
  });
  test("invalid inputs cannot overwrite a generated header", (t) => {
    const root = tempDir(t, "mom-jit-headers-"), output = path.join(root, "headers.inc"); fs.writeFileSync(output, "preserve");
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/native-jit-headers.js"), output, path.join(root, "absent")], {encoding: "utf8"});
    assert.equal(result.status, 1); assert.equal(fs.readFileSync(output, "utf8"), "preserve");
  });
});

test.describe("Native CUDA SDK header generation", () => {
  /** Tiny SDK fixtures exercise the real CLI without a compiler, GPU, VM or network.
   * @param {import('node:test').TestContext} t @param {boolean} [nested] */
  function sdk(t, nested = false) {
    const root = tempDir(t, "mom-jit-headers-"), include = path.join(root, "CUDA include"), cutlass = path.join(root, "CUTLASS include"), output = path.join(root, "headers.inc");
    const write = (/** @type {string} */ name, /** @type {string} */ data) => {const file = path.join(root, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data); return file;};
    write("CUDA include/cuda_runtime.h", "runtime"); write("CUTLASS include/cute/tensor.hpp", "tensor");
    write(`CUDA include/${nested ? "cccl/" : ""}cuda/std/cstdint`, "stdint");
    const run = (roots = [include, cutlass]) => spawnSync(process.execPath,
      [path.join(__dirname, "../scripts/native-jit-headers.js"), output, ...roots], {encoding: "utf8", timeout: 10000});
    return {root, include, cutlass, output, write, run};
  }
  test("missing CUTE fails before overwriting an existing SDK pack", t => {
    const f = sdk(t); fs.unlinkSync(path.join(f.cutlass, "cute/tensor.hpp")); fs.writeFileSync(f.output, "preserve");
    const r = f.run(); assert.equal(r.status, 1); assert.match(r.stderr, /cute\/tensor\.hpp/); assert.equal(fs.readFileSync(f.output, "utf8"), "preserve");
  });
  test("missing standard headers cannot produce a nominally complete CUDA pack", t => {
    const f = sdk(t); fs.unlinkSync(path.join(f.include, "cuda/std/cstdint"));
    const r = f.run(); assert.equal(r.status, 1); assert.match(r.stderr, /cuda\/std\/cstdint/); assert.ok(!fs.existsSync(f.output));
  });
  test("empty required headers are rejected like the runtime decoder rejects them", t => {
    const f = sdk(t); f.write("CUDA include/cuda_runtime.h", "");
    const r = f.run(); assert.equal(r.status, 1); assert.match(r.stderr, /cuda_runtime\.h/);
  });
  test("nested CCCL is exposed at the canonical cuda/std namespace", t => {
    const f = sdk(t, true); const r = f.run(); assert.equal(r.status, 0, r.stderr);
    const report = JSON.parse(fs.readFileSync(f.output + ".json", "utf8"));
    assert.equal(report.sha256, headerResources([entry("cuda_runtime.h", "runtime"), entry("cute/tensor.hpp", "tensor"), entry("cuda/std/cstdint", "stdint")]).sha256);
    assert.equal(report.count, 3);
  });
  test("an explicitly selected CCCL subroot does not duplicate cccl aliases", t => {
    const f = sdk(t, true), r = f.run([f.include, path.join(f.include, "cccl"), f.cutlass]); assert.equal(r.status, 0, r.stderr);
    const report = JSON.parse(fs.readFileSync(f.output + ".json", "utf8")); assert.equal(report.count, 3);
  });
  test("already-flat SDK namespaces retain their exact payload bytes", t => {
    const f = sdk(t), r = f.run(); assert.equal(r.status, 0, r.stderr);
    assert.equal(fs.readFileSync(f.output, "utf8"), headerResources([entry("cuda_runtime.h", "runtime"), entry("cute/tensor.hpp", "tensor"), entry("cuda/std/cstdint", "stdint")]).source);
  });
  test("CUTE is never discovered outside explicitly selected SDK roots", t => {
    const f = sdk(t), r = f.run([f.include]); assert.equal(r.status, 1); assert.match(r.stderr, /cute\/tensor\.hpp/);
  });
  test("conflicting canonical headers cannot silently shadow one another", t => {
    const f = sdk(t, true); f.write("CUDA include/nv/target", "old"); f.write("CUDA include/cccl/nv/target", "different");
    const r = f.run(); assert.equal(r.status, 1); assert.match(r.stderr, /Conflicting JIT include: nv\/target/);
  });
  test("normalization cannot follow a CCCL link outside the selected SDK", {skip: process.platform === "win32"}, t => {
    const f = sdk(t); const target = path.dirname(path.dirname(path.dirname(f.write("outside/cuda/std/cstdint", "not selected"))));
    fs.symlinkSync(target, path.join(f.include, "cccl")); const r = f.run(); assert.equal(r.status, 1); assert.match(r.stderr, /escapes/);
  });
  test("input root count remains bounded before filesystem discovery", t => {
    const f = sdk(t), r = f.run(Array.from({length: 9}, (_, i) => path.join(f.root, "absent" + i)));
    assert.equal(r.status, 1); assert.match(r.stderr, /bounded JIT include roots/);
  });
});

test.describe("Native CUDA prebuilt SDK admission", () => {
  /** @param {import('node:test').TestContext} t */
  function pack(t) {
    const root = tempDir(t, "mom-jit-headers-"), include = path.join(root, "include"), file = path.join(root, "headers.inc"), metadata = file + ".json";
    for (const name of ["cuda_runtime.h", "cute/tensor.hpp", "cuda/std/cstdint"]) {
      const target = path.join(include, name); fs.mkdirSync(path.dirname(target), {recursive: true}); fs.writeFileSync(target, name);
    }
    const {source, ...manifest} = cudaHeaderResources([include]);
    fs.writeFileSync(file, source); fs.writeFileSync(metadata, JSON.stringify(manifest));
    return {root, include, file, metadata, source, manifest};
  }
  test("complete producer inventory freezes both source and sidecar hashes", t => {
    const f = pack(t), result = cudaHeaderInputs(f.file);
    assert.deepEqual(Object.keys(result.files), [f.file, f.metadata]);
    assert.equal(result.files[f.file], f.manifest.sourceSha256);
    assert.match(result.files[f.metadata] || "", /^[0-9a-f]{64}$/);
    assert.deepEqual(result.manifest, f.manifest);
  });
  test("relative packs are refused before filesystem access", () => {assert.throws(() => cudaHeaderInputs("absent.inc"), /absolute/);});
  test("missing inventories cannot admit legacy incomplete packs", t => {
    const f = pack(t); fs.unlinkSync(f.metadata); assert.throws(() => cudaHeaderInputs(f.file), /ENOENT/);
  });
  test("inventories cannot admit a missing source", t => {
    const f = pack(t); fs.unlinkSync(f.file); assert.throws(() => cudaHeaderInputs(f.file), /ENOENT/);
  });
  test("changed generated source cannot reuse a stale inventory", t => {
    const f = pack(t); fs.appendFileSync(f.file, "\n// changed\n"); assert.throws(() => cudaHeaderInputs(f.file), /Incomplete or changed/);
  });
  test("partial or duplicate required-header inventories are refused", t => {
    const f = pack(t);
    for (const requiredHeaders of [[], ["cuda_runtime.h"], ["cuda_runtime.h", "cute/tensor.hpp", "cute/tensor.hpp"], [...f.manifest.requiredHeaders, "other.h"]]) {
      fs.writeFileSync(f.metadata, JSON.stringify({...f.manifest, requiredHeaders})); assert.throws(() => cudaHeaderInputs(f.file), /Incomplete or changed/);
    }
  });
  test("invalid header counts and raw or compressed sizes are bounded", t => {
    const f = pack(t);
    for (const [key, value] of [["count", 2], ["count", 3.5], ["count", 16385], ["bytes", 11], ["bytes", (64 << 20) + 1],
      ["compressedBytes", 0], ["compressedBytes", Infinity], ["compressedBytes", (64 << 20) + 1]]) {
      fs.writeFileSync(f.metadata, JSON.stringify({...f.manifest, [key || ""]: value})); assert.throws(() => cudaHeaderInputs(f.file), /Incomplete or changed/);
    }
  });
  test("malformed and unbound source identities cannot admit a pack", t => {
    const f = pack(t);
    for (const fields of [{sourceSha256: "0".repeat(64)}, {sha256: "not hex"}, {sha256: "A".repeat(64)}]) {
      fs.writeFileSync(f.metadata, JSON.stringify({...f.manifest, ...fields})); assert.throws(() => cudaHeaderInputs(f.file), /Incomplete or changed/);
    }
  });
  test("malformed or oversized metadata fails without unbounded JSON admission", t => {
    const f = pack(t); fs.writeFileSync(f.metadata, "{"); assert.throws(() => cudaHeaderInputs(f.file), SyntaxError);
    fs.writeFileSync(f.metadata, Buffer.alloc((8 << 20) + 1, 32)); assert.throws(() => cudaHeaderInputs(f.file), /excessive/);
  });
  test("nonregular source and metadata paths are never accepted as files", t => {
    const f = pack(t); fs.unlinkSync(f.file); fs.mkdirSync(f.file); assert.throws(() => cudaHeaderInputs(f.file), /regular file/);
    fs.rmdirSync(f.file); fs.writeFileSync(f.file, f.source); fs.unlinkSync(f.metadata); fs.mkdirSync(f.metadata);
    assert.throws(() => cudaHeaderInputs(f.file), /regular file/);
  });
  test("isolating NVRTC exposes only its API, not a foreign runtime header", t => {
    const f = pack(t), api = path.join(f.root, "compiler"), output = path.join(f.root, "output");
    fs.mkdirSync(api); fs.mkdirSync(output);
    fs.writeFileSync(path.join(api, "nvrtc.h"), "selected API"); fs.writeFileSync(path.join(api, "driver_types.h"), "must not shadow runtime");
    const result = require("../scripts/native-gpu-build").cudaBuildHeaders(api, output, f.file, []);
    assert.deepEqual(fs.readdirSync(result.include), ["nvrtc.h"]);
    assert.equal(fs.readFileSync(path.join(result.include, "nvrtc.h"), "utf8"), "selected API");
    assert.equal(fs.readFileSync(path.join(output, "nvrtc-headers.inc"), "utf8"), f.source);
    assert.equal(result.files[f.file], f.manifest.sourceSha256); assert.match(result.files[f.metadata] || "", /^[0-9a-f]{64}$/);
  });
  test("metadata changed while copying cannot be silently rebound to the copied source", t => {
    const f = pack(t), api = path.join(f.root, "api"), output = path.join(f.root, "output");
    fs.mkdirSync(api); fs.mkdirSync(output); fs.writeFileSync(path.join(api, "nvrtc.h"), "API");
    const copy = fs.copyFileSync;
    t.mock.method(fs, "copyFileSync", (/** @type {fs.PathLike} */ source, /** @type {fs.PathLike} */ destination, /** @type {number | undefined} */ mode) => {
      copy(source, destination, mode); if (source === f.file) {fs.appendFileSync(f.metadata, "\n");}
    });
    assert.throws(() => require("../scripts/native-gpu-build").cudaBuildHeaders(api, output, f.file, []), /changed during admission/);
  });
});

test.describe("Native Pearl JIT resource integration", () => {
  const read = (/** @type {string} */ file) => fs.readFileSync(path.join(__dirname, "..", file), "utf8");
  const search = read("sycl/pearlhash/cuda-search.inc"), build = read("scripts/native-gpu-build.js");
  test("GPU builds use the same bounded header generator as compiler proofs", () => {
    assert.match(build, /require\("\.\/native-jit-headers"\)/); assert.match(build, /cudaHeaderResources\(roots\)/);
  });
  test("the CPU compiler oracle accepts every bounded normalized include root", () => {
    const reference = read("tests/reference/native-nvrtc-headers.cpp");
    assert.match(reference, /argc < 5 \|\| argc > 10/);
    assert.match(reference, /for \(int i = 2; i < argc; \+\+i\)/);
  });
  test("compiler proof declares only its fresh cubin output as generated-code storage", () => {
    assert.match(read("tests/tools/probe-native-jit-headers.js"), /auditNativeHeaders\(syscalls, headers\.roots, \[actual\]\)/);
  });
  test("the compiler archive trio comes from one selected SDK", () => {
    assert.match(build, /path\.join\(compilerLib, `lib\$\{name\}\.a`\)/);
  });
  test("the isolated compiler SDK does not replace the application runtime SDK", () => {
    assert.match(build, /process\.env\["MOM_AOT_NVRTC_SDK"\]/); assert.match(build, /"-isystem", selected\.include, "-isystem", cudaInclude/);
  });
  test("native program creation passes memory-only names and exact source arrays", () => {
    assert.match(search, /#if defined\(MOM_EMBEDDED_NVRTC_HEADERS\)\s+includes\.count\(\), includes\.sources\(\), includes\.names\(\)\);/);
  });
  test("Engineering creation retains the original zero-header API", () => {assert.match(search, /#else\s+0, nullptr, nullptr\);/);});
  test("native include admission decodes once without SDK existence probes", () => {
    assert.match(search, /static const mom::JitHeaders& compiler_headers\(\) \{\s+\/\/[^\n]*\n\s+static const mom::JitHeaders headers\(embedded_resource/);
  });
  test("every embedded header participates in the compiled-code cache identity", () => {
    assert.match(search, /key_stream << '\\0' << "embedded-headers" << '\\0' << kJitHeadersSha256/);
  });
  test("incomplete bundles fail admission before NVRTC could probe disk includes", () => {
    assert.match(search, /"cuda_runtime\.h", "cute\/tensor\.hpp", "cuda\/std\/cstdint"/);
    assert.match(search, /headers\.find\(name\)\.empty\(\)/);
  });
  test("Engineering disk include paths remain under the non-embedded branch", () => {
    assert.match(search, /#if !defined\(MOM_EMBEDDED_NVRTC_HEADERS\)\s+"--include-path="/);
  });
  test("AdaptiveCpp compact admission is conditional on the qualified native resource path", () => {
    assert.match(read("sycl/pearlhash/pearlhash.cpp"), /!defined\(MOM_SYCL_ADAPTIVECPP\) \|\| defined\(MOM_EMBEDDED_NVRTC_HEADERS\)/);
    assert.match(search, /require_compact\(driver, rtc, device, major, minor, compiler_major, compiler_minor\)/);
  });
});

test.describe("Native JIT in-memory header decoder", {skip: process.platform !== "linux"}, () => {
  let root = "";
  const canonical = packHeaders([entry("a.h", "é-💎"), entry("z/empty", "")]);
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-jit-decoder-"));
    const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      path.join(__dirname, "native/jit-headers.cpp"), "-lz", "-o", path.join(root, "decoder")], {encoding: "utf8", timeout: 30000});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  /** @param {Buffer} bytes @param {number} [size] @param {boolean} [compressed] */
  function run(bytes, size = bytes.length, compressed = false) {
    const result = spawnSync(path.join(root, "decoder"), [(compressed ? bytes : gzipSync(bytes)).toString("hex"), String(size)], {encoding: "utf8", timeout: 5000});
    assert.equal(result.error, undefined); assert.equal(result.signal, null); return result;
  }
  /** @param {(data: Buffer) => void} edit */
  function rejects(edit) {const bytes = Buffer.from(canonical); edit(bytes); assert.equal(run(bytes).status, 1);}
  test("actual decoder preserves UTF-8, empty files, sorted lookup and pointer ownership", () => {
    const result = run(canonical); assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, "2\na.h c3a92df09f928e\nz/empty \n");
  });
  test("gzip corruption is rejected", () => {const bytes = gzipSync(canonical); bytes[bytes.length - 1] = (bytes[bytes.length - 1] ?? 0) ^ 1; assert.equal(run(bytes, canonical.length, true).status, 1);});
  test("truncated gzip is rejected", () => {assert.equal(run(gzipSync(canonical).subarray(0, 12), canonical.length, true).status, 1);});
  test("extra gzip members or trailing compressed bytes are rejected", () => {assert.equal(run(Buffer.concat([gzipSync(canonical), Buffer.from([1])]), canonical.length, true).status, 1);});
  test("claimed oversized decompression is rejected before allocation", () => {assert.equal(run(canonical, (64 << 20) + 1).status, 1);});
  test("incorrect decompressed byte counts fail closed", () => {assert.equal(run(canonical, canonical.length - 1).status, 1);});
  test("unknown framing cannot reach the JIT", () => {rejects((data) => {data[0] = 0;});});
  test("zero and overflowing file counts are rejected", () => {rejects((data) => data.writeUInt32LE(0, 8)); rejects((data) => data.writeUInt32LE(0xffffffff, 8));});
  test("overflowing name and data lengths are rejected", () => {rejects((data) => data.writeUInt32LE(0xffffffff, 12)); rejects((data) => data.writeUInt32LE(0xffffffff, 16));});
  test("unterminated names cannot read outside the payload", () => {rejects((data) => {data[23] = 65;});});
  test("embedded source NUL cannot silently alter the compile input", () => {rejects((data) => {data[24] = 0;});});
  test("escaping virtual includes are rejected independently of the generator", () => {rejects((data) => data.write("../", 20));});
  test("unordered names cannot corrupt binary lookup", () => {rejects((data) => data.write("zzh", 20));});
  test("trailing uncompressed bytes cannot hide a second payload", () => {assert.equal(run(Buffer.concat([canonical, Buffer.from([0])])).status, 1);});
});

test.describe("Pearl mixed-backend proof geometry", {skip: process.platform !== "linux"}, () => {
  const profiles = [[], ["-DMOM_SYCL_HAS_HIP"], ["-DMOM_SYCL_HAS_CUDA"], ["-DMOM_SYCL_ADAPTIVECPP_CUDA"],
    ["-DMOM_SYCL_HAS_HIP", "-DMOM_SYCL_ADAPTIVECPP_CUDA"]];
  for (const flags of profiles) {
    test(`${flags.join(" ") || "portable"}: dispatch keeps each proof geometry`, (t) => {
      const root = tempDir(t, "mom-jit-headers-");
      const source = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/sycl-search.inc"), "utf8");
      const signature = source.match(/template <int ER, int EC,[\s\S]*?(?=\/\/ Submit the selected portable\/native SYCL tile)/)?.[0];
      const start = source.indexOf("static void search(sycl::queue&"), end = source.indexOf("\n#endif // !MOM_PEARLHASH_ESIMD_TU", start);
      assert.ok(signature && start > 0 && end > start, "Missing maintained search declaration/dispatch");
      // Retain the former conditional declaration too, so the combined-profile negative control
      // fails at its actual template call rather than at an artificial preprocessor boundary.
      fs.writeFileSync(path.join(root, "signature.inc"), (signature.includes("#else") ? "#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)\n" : "") + signature);
      fs.writeFileSync(path.join(root, "dispatch.inc"), source.slice(start, end));
      const binary = path.join(root, "dispatch");
      const build = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", ...flags, "-I", root,
        "-I", path.join(__dirname, "../sycl/pearlhash"), path.join(__dirname, "native/pearlhash-search-profiles.cpp"), "-o", binary], {encoding: "utf8", timeout: 30000});
      assert.equal(build.status, 0, build.error?.message || build.stderr);
      const run = spawnSync(binary, [], {encoding: "utf8", timeout: 5000, env: {...process.env, MOM_PEARLHASH_AMD_DP4A_TILE: ""}});
      assert.equal(run.status, 0, run.error?.message || run.stderr);
    });
  }
});

test.describe("Pearl producer and compiler-feature routing", {skip: process.platform !== "linux"}, () => {
  let root = "";
  const profiles = [
    {name: "generic SSCP", tensor: false, flags: ["-DMOM_SYCL_ADAPTIVECPP", "-D__ACPP_ENABLE_LLVM_SSCP_TARGET__"]},
    {name: "SSCP on NVIDIA", tensor: false, flags: ["-DMOM_SYCL_ADAPTIVECPP", "-D__ACPP_ENABLE_LLVM_SSCP_TARGET__", "-D__NVPTX__"]},
    {name: "legacy SSCP flag", tensor: false, flags: ["-DMOM_SYCL_ADAPTIVECPP", "-D__HIPSYCL_ENABLE_LLVM_SSCP_TARGET__"]},
    {name: "direct NVPTX", tensor: true, flags: ["-D__NVPTX__"]},
    {name: "legacy CUDA", tensor: true, flags: ["-DMOM_SYCL_ADAPTIVECPP", "-D__NVPTX__", "-D__CUDA_ARCH__=800"]},
  ];
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pearl-producer-"));
    const matrix = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/matrix.inc"), "utf8");
    const start = matrix.indexOf("    const size_t idx = id[0], i = idx /", matrix.indexOf("template <bool Dpasw"));
    const packet = matrix.indexOf("template <bool IsA>\nstatic inline size_t nvidia_vector128_offset");
    assert.ok(start > 0 && packet > 0);
    fs.writeFileSync(path.join(root, "producer.inc"), matrix.slice(start, matrix.indexOf("\n  });", start)) + "\n");
    fs.writeFileSync(path.join(root, "packet.inc"), matrix.slice(packet, matrix.indexOf("\n}", packet) + 2) + "\n");
    const state = fs.readFileSync(path.join(__dirname, "../sycl/pearlhash/state.inc"), "utf8");
    const field = state.indexOf("  const bool cuda_tensor"), begin = state.lastIndexOf("#if", field);
    fs.writeFileSync(path.join(root, "tensor.inc"), state.slice(begin, state.indexOf("#endif", field) + 6) + "\n");
    for (const [index, profile] of profiles.entries()) {
      const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-DMOM_SYCL_HAS_CUDA", `-DMOM_TEST_CUDA_TENSOR=${Number(profile.tensor)}`, ...profile.flags,
        "-I", root, path.join(__dirname, "native/pearlhash-compiler-routes.cpp"), "-o", path.join(root, String(index))], {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.error?.message || result.stderr);
    }
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  /** @param {number} index @param {string} mode */
  function execute(index, mode) {
    const result = spawnSync(path.join(root, String(index)), [mode], {encoding: "utf8", timeout: 5000});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
  }
  for (const [index, profile] of profiles.entries()) {
    for (const mode of ["packed", "portable", "tiled"]) {
      test(`${profile.name}: ${mode} producer and executable tensor admission`, () => {
        execute(index, mode);
      });
    }
  }
});
