"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path");
const {createHash} = require("node:crypto"), {spawnSync} = require("node:child_process");
const {buildPathsOverlap, fileHash, walk, sourceInputs, nativeSourceInputs, verifySourceInputs, readBuildJson, readBuildText,
  hasExactBuildFields, requireBuildFields, hasBuildDirectoryEntries, canonicalBuildFile, readCanonicalBuildJson, verifyStableBuildJson,
  buildDependencies, buildFileInputs, completeBuildInputs} = require("../scripts/build-source-inputs");
const {admitCpuCache} = require("../scripts/build-native-core");
const {nativeApplicationInputs, nativeApplicationBuildInputs, nativeApplicationCudaImages, verifyCudaImageReceiptInputs} = require("../scripts/build-native-application");
const {tempDir} = require("./common/temp-dir");
const repo = path.resolve(__dirname, ".."), digest = (/** @type {string | Buffer} */ value) => createHash("sha256").update(value).digest("hex");
const graphNames = [...new Set([sourceInputs(repo), nativeSourceInputs(repo, "cpu"), nativeSourceInputs(repo, "gpu"), nativeApplicationBuildInputs(repo)]
  .flatMap(state => Object.keys(state.graph)))];

/** Miniature source/cache trees, not compiler execution or edits to the real repository.
 * @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = tempDir(t, "mom-native-inputs-");
  /** @param {string} name @param {string | Buffer} value */
  const put = (name, value) => {
    const file = path.join(root, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, value); return file;
  };
  for (const name of graphNames) {put(name, "fixture:" + name);}
  for (const name of ["native/core.cpp", "native/core.h", "sycl/kernel.cpp", "sycl/kernel.inc", "xmrig/hash.h", "runtime/host.cpp", "mom.js"]) {put(name, "fixture:" + name);}
  const cache = path.join(root, "cache"), receipt = path.join(cache, ".mom-content-state.json");
  /** @type {Record<string, string>} */
  const generated = {"mom.target.mk": digest(putText("cache/mom.target.mk"))};
  /** @type {Record<string, string>} */
  const links = {"Release/hash.a": digest(putText("cache/Release/hash.a"))};
  /** @type {Record<string, {content: string}>} */
  const objects = {"Release/core.o": {content: digest(putText("cache/Release/core.o"))}};
  const sdk = put("sdk/header.h", "SDK header"), external = {[sdk]: fileHash(sdk)};
  /** @param {string} name */
  function putText(name) {const value = "fixture:" + name; put(name, value); return value;}
  const state = {schema: 1, portability: {portable: true, cpuMarch: ""}, ...sourceInputs(root), generated, links, objects, external};
  const save = () => fs.writeFileSync(receipt, JSON.stringify(state)); save();
  return {root, put, cache, receipt, state, save};
}

/** Shared miniature application admission fixture; no real SDK, compiler or device.
 * @param {import("node:test").TestContext} t */
function application(t) {
  const f = fixture(t), cpu = f.put("core/cpu-core.a", "core"), gpu = f.put("gpu/compiler.o", "gpu");
  const core = {gpu: true, portable: true, objectSha256: fileHash(cpu), sourceState: nativeSourceInputs(f.root, "cpu")};
  const graph = {factory: "mom_native_gpu_oneapi", object: gpu, objectSha256: fileHash(gpu), sourceState: nativeSourceInputs(f.root, "gpu")};
  const record = f.put("gpu/result.json", JSON.stringify(graph));
  const save = () => {f.put("core/inputs.json", JSON.stringify(core)); f.put("gpu/result.json", JSON.stringify(graph));}; save();
  const inputs = {objects: [{object: gpu}], receipts: [record], sdkObjects: /** @type {{object: string}[]} */ ([])}, factories = [graph.factory];
  const admit = () => nativeApplicationInputs(f.root, cpu, inputs, factories);
  return {...f, cpu, gpu, core, graph, record, save, inputs, factories, admit};
}

test.describe("Shared build source content and membership", () => {
  test("source traversal order is independent of the host locale", t => {
    const f = fixture(t), directory = path.join(f.root, "walk");
    for (const name of ["z", "a", "B"]) {f.put(path.join("walk", name), "fixture");}
    /** @type {string[]} */
    const names = [];
    walk(directory, file => names.push(path.basename(file)));
    assert.deepEqual(names, ["B", "a", "z"]);
  });
  test("unchanged sources produce identical content receipts", t => {const f = fixture(t); assert.deepEqual(sourceInputs(f.root), sourceInputs(f.root));});
  test("JavaScript-only Engineering changes do not enter native source state", t => {
    const f = fixture(t), before = sourceInputs(f.root); f.put("mom.js", "changed controller"); assert.deepEqual(sourceInputs(f.root), before);
  });
  for (const file of ["native/core.cpp", "native/core.h", "sycl/kernel.inc", "xmrig/hash.h", "binding.gyp"]) {
    test(`backdated ${file} edits cannot reuse a content receipt`, t => {
      const f = fixture(t), before = sourceInputs(f.root), name = f.put(file, "changed"); fs.utimesSync(name, 1, 1);
      assert.throws(() => verifySourceInputs(before, sourceInputs(f.root), "test"), /stale.*rebuild/);
    });
  }
  test("new conditional headers change source membership", t => {
    const f = fixture(t), before = sourceInputs(f.root); f.put("sycl/new-header.inc", "new");
    assert.throws(() => verifySourceInputs(before, sourceInputs(f.root), "test"), /stale/);
  });
  test("removed headers invalidate a receipt", t => {
    const f = fixture(t), before = sourceInputs(f.root); fs.unlinkSync(path.join(f.root, "sycl/kernel.inc"));
    assert.throws(() => verifySourceInputs(before, sourceInputs(f.root), "test"), /stale/);
  });
  test("native receipts also exclude fresh application JavaScript", t => {
    const f = fixture(t), before = nativeSourceInputs(f.root, "cpu"); f.put("mom.js", "changed");
    assert.deepEqual(nativeSourceInputs(f.root, "cpu"), before);
  });
  test("GPU kernel edits do not invalidate the distinct CPU Core", t => {
    const f = fixture(t), cpu = nativeSourceInputs(f.root, "cpu"), gpu = nativeSourceInputs(f.root, "gpu");
    f.put("sycl/kernel.inc", "changed"); assert.deepEqual(nativeSourceInputs(f.root, "cpu"), cpu);
    assert.throws(() => verifySourceInputs(gpu, nativeSourceInputs(f.root, "gpu"), "GPU"), /stale/);
  });
  test("shared GPU public ABI edits invalidate both Core and kernels", t => {
    const f = fixture(t), before = ["cpu", "gpu"].map(kind => nativeSourceInputs(f.root, /** @type {"cpu" | "gpu"} */ (kind)));
    f.put("sycl/lib.h", "changed ABI");
    for (const [index, kind] of ["cpu", "gpu"].entries()) {
      assert.throws(() => verifySourceInputs(before[index], nativeSourceInputs(f.root, /** @type {"cpu" | "gpu"} */ (kind)), kind), /stale/);
    }
  });
  test("GPU receipts cover the native headers actually included by maintained kernels", t => {
    const f = fixture(t);
    for (const name of ["native/consts.h", "native/job-boundary.h", "native/jit-headers.h", "native/cpu-scheduling.h"]) {
      f.put(name, "before"); const before = nativeSourceInputs(f.root, "gpu"); f.put(name, "after");
      assert.throws(() => verifySourceInputs(before, nativeSourceInputs(f.root, "gpu"), "GPU"), /stale/);
    }
  });
  test("native build recipe edits invalidate only their selected component", t => {
    const f = fixture(t), before = nativeSourceInputs(f.root, "gpu"); f.put("scripts/native-gpu-build.js", "changed recipe");
    assert.throws(() => verifySourceInputs(before, nativeSourceInputs(f.root, "gpu"), "GPU"), /stale/);
  });
  test("Windows Core producer edits invalidate Core without rebuilding Engineering or GPU kernels", t => {
    const f = fixture(t), cpu = nativeSourceInputs(f.root, "cpu"), gpu = nativeSourceInputs(f.root, "gpu"), engineering = sourceInputs(f.root);
    f.put("scripts/build-native-windows-core.js", "changed standalone Core recipe");
    assert.throws(() => verifySourceInputs(cpu, nativeSourceInputs(f.root, "cpu"), "Core"), /stale/);
    assert.deepEqual(nativeSourceInputs(f.root, "gpu"), gpu);
    assert.deepEqual(sourceInputs(f.root), engineering);
  });
  test("Windows GPU recipe, wrapper, and static ABI edits invalidate only native GPU receipts", t => {
    const f = fixture(t), cpu = nativeSourceInputs(f.root, "cpu"), engineering = sourceInputs(f.root);
    for (const name of ["scripts/build-native-windows-gpu.js", "scripts/native-windows-wrapper.js", "scripts/build-native-windows-controller.js", "scripts/build-native-release.js", "tests/native/dpcpp-static/static-windows.h"]) {
      const before = nativeSourceInputs(f.root, "gpu"); f.put(name, "changed Windows GPU recipe");
      assert.throws(() => verifySourceInputs(before, nativeSourceInputs(f.root, "gpu"), "GPU"), /stale/);
      assert.deepEqual(nativeSourceInputs(f.root, "cpu"), cpu);
      assert.deepEqual(sourceInputs(f.root), engineering);
    }
  });
  test("Windows GPU source receipts cover every directly imported build helper", () => {
    const graph = nativeSourceInputs(repo, "gpu").graph;
    for (const file of ["build-native-gpu.js", "native-gpu-build.js", "build-native-windows-gpu.js", "native-windows-wrapper.js", "native-sycl-inputs.js"]) {
      const source = fs.readFileSync(path.join(repo, "scripts", file), "utf8");
      for (const match of source.matchAll(/require\("\.\/([^"\n]+)"\)/g)) {
        const dependency = "scripts/" + match[1] + ".js";
        assert.ok(Object.hasOwn(graph, dependency), "untracked GPU helper from " + file + ": " + dependency);
      }
    }
  });
  test("shared source-input changes invalidate both CPU and GPU receipts", t => {
    const f = fixture(t), cpu = nativeSourceInputs(f.root, "cpu"), gpu = nativeSourceInputs(f.root, "gpu");
    f.put("scripts/build-source-inputs.js", "changed shared source input module");
    assert.throws(() => verifySourceInputs(cpu, nativeSourceInputs(f.root, "cpu"), "Core"), /stale/);
    assert.throws(() => verifySourceInputs(gpu, nativeSourceInputs(f.root, "gpu"), "GPU"), /stale/);
  });
  for (const file of ["scripts/native-elf.js", "scripts/native-jit-link.js"]) {
    test(`backdated ${file} edits invalidate GPU audits without rebuilding Core or Engineering`, t => {
      const f = fixture(t), gpu = nativeSourceInputs(f.root, "gpu"), cpu = nativeSourceInputs(f.root, "cpu"), engineering = sourceInputs(f.root);
      const changed = f.put(file, "changed GPU audit helper"); fs.utimesSync(changed, 1, 1);
      assert.throws(() => verifySourceInputs(gpu, nativeSourceInputs(f.root, "gpu"), "GPU"), /stale.*rebuild/);
      assert.deepEqual(nativeSourceInputs(f.root, "cpu"), cpu); assert.deepEqual(sourceInputs(f.root), engineering);
    });
  }
  test("shared native GPU-boundary audit recipes also invalidate the Core receipt", t => {
    const f = fixture(t);
    for (const name of ["scripts/native-gpu-table.js", "scripts/native-compiler-isolation.js", "scripts/native-audit.js"]) {
      const before = nativeSourceInputs(f.root, "cpu"); f.put(name, "changed native boundary audit");
      assert.throws(() => verifySourceInputs(before, nativeSourceInputs(f.root, "cpu"), "Core"), /stale/);
    }
  });
  test("unknown scope and unsafe source roots fail before traversal", t => {
    const f = fixture(t); assert.throws(() => nativeSourceInputs(f.root, /** @type {"gpu"} */ ("invalid")), /Unknown/);
    for (const name of ["..", "sycl/../native", "/tmp", "a\\b", "a\0b", "a:b", ""]) {
      assert.throws(() => sourceInputs(f.root, [name], []), /explicit relative/);
    }
  });
  test("large stale receipts produce a small diagnostic, never a source diff", () => {
    assert.throws(() => verifySourceInputs({large: "x".repeat(1 << 20)}, {files: {}, graph: {}}, "Core"), error =>
      error instanceof Error && error.message.length < 100 && /rebuild/.test(error.message));
  });
  test("a source-directory symlink cannot expand discovery outside the selected source tree", {skip: process.platform === "win32"}, t => {
    const f = fixture(t); fs.renameSync(path.join(f.root, "native"), path.join(f.root, "native-original"));
    fs.symlinkSync(path.join(f.root, "native-original"), path.join(f.root, "native"), "dir");
    assert.throws(() => sourceInputs(f.root), /real directory/);
  });
});

test.describe("Bounded native input and receipt IO", {concurrency: false}, () => {
  test("managed paths overlap their ancestor, including the filesystem root, and exclude siblings", () => {
    const input = path.resolve("sdk"), output = path.join(input, "build");
    assert.ok(buildPathsOverlap(input, output));
    assert.ok(buildPathsOverlap(output, path.parse(input).root));
    assert.ok(buildPathsOverlap(input, input));
    assert.ok(!buildPathsOverlap(input, input + "-other"));
  });
  test("streamed hashes match independent SHA256", t => {const f = fixture(t); assert.equal(fileHash(f.put("payload", "hello")), digest("hello"));});
  test("empty regular files are valid source content", t => {const f = fixture(t); assert.equal(fileHash(f.put("empty", "")), digest(""));});
  test("multi-megabyte inputs are read in bounded chunks without readFileSync", t => {
    const f = fixture(t), data = Buffer.alloc((2 << 20) + 7, 0x61), file = f.put("large.a", data);
    const read = fs.readSync, whole = fs.readFileSync; let maximum = 0;
    fs.readSync = /** @type {typeof fs.readSync} */ ((...args) => {maximum = Math.max(maximum, Number(args[3])); return Reflect.apply(read, fs, args);});
    fs.readFileSync = () => {throw new Error("Unbounded read");};
    try {assert.equal(fileHash(file), digest(data)); assert.ok(maximum <= (64 << 10));} finally {fs.readSync = read; fs.readFileSync = whole;}
  });
  test("file content changing during hashing is rejected", t => {
    const f = fixture(t), file = f.put("changing", "old"), read = fs.readSync; let changed = false;
    fs.readSync = /** @type {typeof fs.readSync} */ ((...args) => {
      const size = Reflect.apply(read, fs, args); if (!changed) {changed = true; fs.appendFileSync(file, "more");} return size;
    });
    try {assert.throws(() => fileHash(file), /changed while hashing/);} finally {fs.readSync = read;}
  });
  test("directories cannot be hashed as build files", t => {const f = fixture(t); assert.throws(() => fileHash(f.root), /regular file/);});
  test("missing build files fail rather than manufacturing a digest", t => {const f = fixture(t); assert.throws(() => fileHash(path.join(f.root, "missing")), /ENOENT/);});
  test("malformed JSON is rejected", t => {const f = fixture(t); assert.throws(() => readBuildJson(f.put("bad.json", "{")), SyntaxError);});
  test("empty JSON receipts are not accepted", t => {const f = fixture(t); assert.throws(() => readBuildJson(f.put("empty.json", "")), /Invalid/);});
  test("oversized receipts fail before allocating their contents", t => {
    const f = fixture(t), file = f.put("large.json", "{}"); fs.truncateSync(file, (8 << 20) + 1);
    assert.throws(() => readBuildJson(file), /excessive/);
  });
  test("directory receipts cannot block admission", t => {const f = fixture(t); assert.throws(() => readBuildJson(f.root), /Invalid/);});
  test("valid bounded JSON survives unchanged", t => {const f = fixture(t); assert.deepEqual(readBuildJson(f.put("valid.json", '{"value":7}')), {value: 7});});
  test("field names containing the delimiter cannot impersonate separate required fields", () => {
    const value = {root: "/sdk", receipt: "result.json"};
    assert.equal(requireBuildFields(value, ["root", "receipt"], "SDK"), value);
    for (const delimiter of [",", "\0"]) {
      const malformed = {[["receipt", "root"].join(delimiter)]: value};
      assert.equal(hasExactBuildFields(malformed, ["root", "receipt"]), false);
      assert.throws(() => requireBuildFields(malformed, ["root", "receipt"], "SDK"), /unknown or missing fields/);
    }
  });
  test("a combined filename cannot impersonate two build directory entries", t => {
    const f = fixture(t), directory = path.join(f.root, "managed"), combined = f.put("managed/config.json,result.json", "{}");
    const expected = ["config.json", "result.json"];
    assert.equal(hasBuildDirectoryEntries(directory, expected), false);
    assert.deepEqual(fs.readdirSync(directory), [path.basename(combined)]);
    fs.renameSync(combined, path.join(directory, "config.json"));
    assert.equal(hasBuildDirectoryEntries(directory, expected), false);
    f.put("managed/result.json", "{}");
    assert.equal(hasBuildDirectoryEntries(directory, expected), true);
  });
  test("stable JSON admission returns the digest of the exact serialized bytes", t => {
    const f = fixture(t), bytes = '{\n  "value": 7\n}\n', file = f.put("stable.json", bytes);
    assert.equal(verifyStableBuildJson(file, {value: 7}, "Config"), digest(bytes));
    assert.throws(() => verifyStableBuildJson(file, {value: 8}, "Config"), /Config is stale/);
    fs.linkSync(file, path.join(f.root, "hardlink.json"));
    assert.throws(() => canonicalBuildFile(file, "Config"), /bounded canonical regular file/);
  });
  test("replacing a receipt during admission invalidates the snapshot", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), file = f.put("stable.json", "{}\n"), replacement = f.put("replacement.json", "{}\n"), read = fs.readSync;
    fs.readSync = /** @type {typeof fs.readSync} */ ((...args) => {
      fs.readSync = read;
      fs.renameSync(replacement, file);
      return Reflect.apply(read, fs, args);
    });
    try {assert.throws(() => verifyStableBuildJson(file, {}, "Config"), /changed/);} finally {fs.readSync = read;}
  });
  test("the receipt digest covers the same bytes that supplied the parsed value", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), bytes = '{"value":1}', file = f.put("pinned.json", bytes);
    const pinned = readCanonicalBuildJson(file, 1024, "Config");
    assert.deepEqual(pinned.value, {value: 1});
    assert.equal(pinned.sha256, digest(bytes));

    const replacement = f.put("replacement.json", '{"value":2}'), saved = path.join(f.root, "saved.json"), open = fs.openSync;
    fs.openSync = /** @type {typeof fs.openSync} */ ((...args) => {
      if (args[0] !== file) {return Reflect.apply(open, fs, args);}
      fs.renameSync(file, saved);
      fs.renameSync(replacement, file);
      const fd = Reflect.apply(open, fs, args);
      fs.renameSync(file, replacement);
      fs.renameSync(saved, file);
      return fd;
    });
    try {assert.throws(() => readCanonicalBuildJson(file, 1024, "Config"), /changed before admission/);} finally {fs.openSync = open;}
    assert.equal(fileHash(file), pinned.sha256);
  });
  test("receipt growth during reading fails before parsing a partial object", t => {
    const f = fixture(t), file = f.put("changing.json", "{}"), read = fs.readSync; let changed = false;
    fs.readSync = /** @type {typeof fs.readSync} */ ((...args) => {
      const size = Reflect.apply(read, fs, args); if (!changed) {changed = true; fs.appendFileSync(file, " ");} return size;
    });
    try {assert.throws(() => readBuildJson(file), /changed during admission/);} finally {fs.readSync = read;}
  });
  test("failed hashes close their descriptor", t => {
    const f = fixture(t), file = f.put("fault", "data"), read = fs.readSync, close = fs.closeSync; let closed = false;
    fs.readSync = () => {throw new Error("read fault");};
    fs.closeSync = fd => {closed = true; close(fd);};
    try {assert.throws(() => fileHash(file), /read fault/); assert.equal(closed, true);} finally {fs.readSync = read; fs.closeSync = close;}
  });
  test("FIFO inputs are rejected without waiting for a writer", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), file = path.join(f.root, "pipe"); assert.equal(spawnSync("mkfifo", [file]).status, 0);
    assert.throws(() => fileHash(file), /regular file/); assert.throws(() => readBuildJson(file), /Invalid/);
  });
});

test.describe("Literal compiler dependency rules shared with Engineering", () => {
  const base = path.resolve("/build"), resolved = (/** @type {string} */ file) => path.resolve(base, file);
  test("ordinary compiler dependencies resolve relative to the explicit build cwd", () => {
    assert.deepEqual(buildDependencies("out.o: a.cpp sdk.h\n", base), [resolved("a.cpp"), resolved("sdk.h")]);
  });
  test("line continuations preserve all source and SDK inputs", () => {
    assert.deepEqual(buildDependencies("out.o: a.cpp \\\n sdk.h\n", base), [resolved("a.cpp"), resolved("sdk.h")]);
  });
  test("escaped spaces remain one filename", () => {assert.deepEqual(buildDependencies("out.o: sdk\\ name/header.h", base), [resolved("sdk name/header.h")]);});
  test("escaped target punctuation cannot split the dependency body", () => {assert.deepEqual(buildDependencies("out\\:name.o: sdk.h", base), [resolved("sdk.h")]);});
  test("Gyp command-first metadata cannot masquerade as dependency paths", () => {
    assert.deepEqual(buildDependencies("cmd_obj := clang -c source.cpp\ndeps_obj := a.cpp sdk.h\n", base), [resolved("a.cpp"), resolved("sdk.h")]);
  });
  test("modern Gyp command assignments also precede ordinary rules and -MP targets", () => {
    assert.deepEqual(buildDependencies("cmd_obj := clang -c a.cpp\nout.o: a.cpp sdk.h\nsdk.h:\n", base), [resolved("a.cpp"), resolved("sdk.h")]);
  });
  test("host and device dependency rules are unioned rather than dropping one", () => {
    assert.deepEqual(buildDependencies("out.o: a.cpp host.h\nout.o: a.cpp device.h\n", base), [resolved("a.cpp"), resolved("host.h"), resolved("device.h")]);
  });
  test("compiler -MP phony rules contribute no extra inputs", () => {assert.deepEqual(buildDependencies("out.o: a.cpp sdk.h\nsdk.h:\n", base), [resolved("a.cpp"), resolved("sdk.h")]);});
  test("missing rules or bodies fail closed", () => {for (const text of ["", "out.o:", "no rule", "out.o: a.cpp\nnot a rule"]) {assert.equal(buildDependencies(text, base), null);}});
  test("Make expansion and unescaped comments are not evaluated", () => {for (const text of ["out.o: $(SDK)/x.h", "out.o: a.cpp # hidden.h"]) {assert.equal(buildDependencies(text, base), null);}});
  test("relative base directories never inherit an incidental process cwd", () => {assert.equal(buildDependencies("out.o: a.cpp", "relative"), null);});
  test("oversized dependency metadata is bounded before tokenization", () => {assert.equal(buildDependencies("x".repeat((4 << 20) + 1), base), null);});
});

test.describe("NMake compiler dependencies", () => {
  test("captured native-Windows paths retain backslashes and quoted spaces", () => {
    const originalIsAbsolute = path.isAbsolute, originalResolve = path.resolve;
    path.isAbsolute = path.win32.isAbsolute; path.resolve = path.win32.resolve;
    try {
      const base = String.raw`C:\repo`, source = String.raw`C:\mom-depfile-371\path with spaces\sample.c`;
      const header = String.raw`C:\mom-depfile-371\path with spaces\fixture header.h`, system = String.raw`C:\sdk\include\base.h`;
      const depfile = 'fixture.o: "' + source + '" \\\r\n  "' + header + '" ' + system + "\r\n";
      assert.deepEqual(buildDependencies(depfile, base, "nmake"), [source, header, system]);
    } finally {path.isAbsolute = originalIsAbsolute; path.resolve = originalResolve;}
  });
  test("ordinary cross-host NMake dependencies preserve relative paths", () => {
    const base = path.resolve("/build");
    assert.deepEqual(buildDependencies('output.o: "sdk dir/header.h" source.cpp\n', base, "nmake"),
      [path.resolve(base, "sdk dir/header.h"), path.resolve(base, "source.cpp")]);
  });
  test("clang -MV output resolves a real header whose path contains spaces", {skip: !fs.existsSync("/usr/lib/llvm-21/bin/clang")}, t => {
    const f = fixture(t), header = f.put("sdk dir/header file.h", "#define VALUE 1\n");
    const source = f.put("sdk dir/source.c", '#include "header file.h"\nint value = VALUE;\n');
    const result = spawnSync("/usr/lib/llvm-21/bin/clang", ["-M", "-MV", "-MT", "fixture", source], {encoding: "utf8", timeout: 10000});
    assert.equal(result.status, 0, result.stderr); assert.deepEqual(buildDependencies(result.stdout, f.root, "nmake"), [source, header]);
  });
  test("malformed, concatenated, or expandable NMake entries fail closed", () => {
    const base = path.resolve("/build");
    for (const text of ['out.o: "unterminated', 'out.o: "first"second', 'out.o: ""', "out.o: $(SDK)/header.h", "deps_obj := header.h"]) {
      assert.equal(buildDependencies(text, base, "nmake"), null, text);
    }
  });
});

test.describe("Native SDK dependency receipts capture actual used headers", () => {
  /** @param {import("node:test").TestContext} t */
  function inputs(t) {
    const f = fixture(t), source = f.put("project/source.cpp", "source"), compiler = f.put("sdk/compiler", "tool"), header = f.put("sdk/header.h", "header");
    // GNU depfiles use forward slashes and escape spaces, even on a Windows host.
    const generated = f.put("out/generated.inc", "generated from maintained sources"), depfile = f.put("out/0.d", "out.o: " +
      [source, header, generated].map(file => file.replace(/\\/g, "/").replace(/ /g, "\\ ")).join(" ") + "\n");
    // All fixture inputs exist before compilation; compare against their observed clock,
    // not a small guessed margin over the host's differently rounded wall clock.
    const repo = path.join(f.root, "project"), output = path.join(f.root, "out"),
      startedAt = Math.max(Date.now(), ...[source, compiler, header, generated, depfile].map(file => fs.statSync(file).ctimeMs)),
      before = buildFileInputs([compiler]);
    const finish = (initial = before, started = startedAt) => completeBuildInputs([depfile], repo, output, started, initial);
    return {...f, source, compiler, header, generated, depfile, repo, output, before, startedAt, finish};
  }
  test("only external SDK inputs supplement already tracked source/generated files", t => {
    // Model a wall clock behind filesystem metadata, as can happen with Windows timer rounding.
    t.mock.method(Date, "now", () => 0);
    const f = inputs(t); assert.deepEqual(f.finish(), buildFileInputs([f.compiler, f.header]));
  });
  test("compiler changes cannot pass as source-only parity", t => {const f = inputs(t); fs.writeFileSync(f.compiler, "different tool"); assert.throws(() => f.finish(), /SDK input changed/);});
  test("backdated SDK header edits invalidate a completed input receipt", t => {
    const f = inputs(t), receipt = f.finish(); fs.writeFileSync(f.header, "different header"); fs.utimesSync(f.header, 1, 1);
    assert.throws(() => f.finish(receipt), /SDK input changed/);
  });
  test("a first-seen header modified during the build is not blessed afterward", t => {const f = inputs(t); assert.throws(() => f.finish(f.before, 0), /New native SDK dependency changed/);});
  test("missing compiler dependency files cannot produce incomplete receipts", t => {const f = inputs(t); fs.unlinkSync(f.depfile); assert.throws(() => f.finish(), /ENOENT/);});
  test("missing consumed headers fail before publishing a receipt", t => {const f = inputs(t); fs.unlinkSync(f.header); assert.throws(() => f.finish(), /ENOENT/);});
  test("malformed dependency bodies fail instead of silently excluding headers", t => {const f = inputs(t); fs.writeFileSync(f.depfile, "not a dependency rule"); assert.throws(() => f.finish(), /Incomplete/);});
  test("SDK aliases under output are resolved before excluding generated output", {skip: process.platform === "win32"}, t => {
    const f = inputs(t), alias = path.join(f.output, "sdk"); fs.symlinkSync(path.join(f.root, "sdk"), alias, "dir");
    fs.writeFileSync(f.depfile, "out.o: " + path.join(alias, "header.h")); assert.deepEqual(f.finish(), buildFileInputs([f.compiler, f.header]));
  });
  test("absolute prerequisite paths cannot resolve against the caller cwd", () => {assert.throws(() => buildFileInputs(["relative"]), /absolute/);});
  test("dependency input growth is bounded before reading", t => {
    const f = inputs(t); fs.truncateSync(f.depfile, (4 << 20) + 1); assert.throws(() => f.finish(), /excessive/);
  });
  test("bounded text and JSON receipts share the same regular-file admission", t => {
    const f = inputs(t); assert.equal(readBuildText(f.depfile), fs.readFileSync(f.depfile, "utf8"));
    assert.throws(() => readBuildText(f.output), /Invalid/);
  });
});

test.describe("Native Core admits only completed Engineering cache content", () => {
  test("unchanged completed cache is admitted", t => {const f = fixture(t); assert.equal(admitCpuCache(f.root, f.cache), fileHash(f.receipt));});
  for (const file of ["native/core.h", "sycl/kernel.inc", "xmrig/hash.h", "scripts/cpu-optflags.sh"]) {
    test(`changed ${file} cannot reuse an old cache`, t => {const f = fixture(t); f.put(file, "changed"); assert.throws(() => admitCpuCache(f.root, f.cache), /stale/);});
  }
  for (const file of ["cache/Release/core.o", "cache/Release/hash.a", "cache/mom.target.mk"]) {
    test(`altered ${file} cannot be blessed by source parity`, t => {const f = fixture(t); f.put(file, "changed"); assert.throws(() => admitCpuCache(f.root, f.cache), /content differs/);});
  }
  test("consumed SDK header edits invalidate the cache", t => {const f = fixture(t); f.put("sdk/header.h", "changed"); assert.throws(() => admitCpuCache(f.root, f.cache), /SDK dependency differs/);});
  test("a missing completed receipt fails before compiler lookup", t => {const f = fixture(t); fs.unlinkSync(f.receipt); assert.throws(() => admitCpuCache(f.root, f.cache), /ENOENT/);});
  test("unknown receipt schemas are rejected", t => {const f = fixture(t); f.state.schema = 99; f.save(); assert.throws(() => admitCpuCache(f.root, f.cache), /completed/);});
  test("missing object hashes cannot masquerade as a completed build", t => {const f = fixture(t); f.state.objects = {}; f.save(); assert.throws(() => admitCpuCache(f.root, f.cache), /Incomplete/);});
  test("cache artifact traversal is refused before reading the target", t => {
    const f = fixture(t); f.state.generated = {"../outside": "0".repeat(64)}; f.save(); assert.throws(() => admitCpuCache(f.root, f.cache), /content differs/);
  });
  test("missing consumed SDK metadata cannot weaken cache admission", t => {const f = fixture(t); f.state.external = {}; f.save(); assert.throws(() => admitCpuCache(f.root, f.cache), /SDK dependency receipt/);});
  test("JS-only edits keep the completed Engineering generation usable", t => {
    const f = fixture(t), before = admitCpuCache(f.root, f.cache); f.put("mom.js", "changed controller"); assert.equal(admitCpuCache(f.root, f.cache), before);
  });
});

test.describe("Native Core release portability admission", () => {
  test("a completed portable generation is admitted for release", t => {
    const f = fixture(t); assert.equal(admitCpuCache(f.root, f.cache, true), fileHash(f.receipt));
  });
  test("a host-tuned Engineering generation is not a portable release input", t => {
    const f = fixture(t); f.state.portability.portable = false; f.save();
    assert.throws(() => admitCpuCache(f.root, f.cache, true), /portable.*rebuild/i);
  });
  test("an explicit host CPU override cannot be hidden by the portable flag", t => {
    const f = fixture(t); f.state.portability.cpuMarch = "native"; f.save();
    assert.throws(() => admitCpuCache(f.root, f.cache, true), /portable.*rebuild/i);
  });
  for (const value of [undefined, null, "1", {portable: "1", cpuMarch: ""}, {portable: true}, {portable: true, cpuMarch: "rocketlake"}]) {
    test("legacy or malformed portability metadata is not release evidence: " + JSON.stringify(value), t => {
      const f = fixture(t); Object.assign(f.state, {portability: value}); f.save();
      assert.throws(() => admitCpuCache(f.root, f.cache, true), /portable.*rebuild/i);
    });
  }
  test("standalone Engineering probes can still use a valid host-tuned generation", t => {
    const f = fixture(t); f.state.portability = {portable: false, cpuMarch: "native"}; f.save();
    assert.equal(admitCpuCache(f.root, f.cache), fileHash(f.receipt));
  });
  test("portable metadata never substitutes for verified object contents", t => {
    const f = fixture(t); f.put("cache/Release/core.o", "altered");
    assert.throws(() => admitCpuCache(f.root, f.cache, true), /content differs/);
  });
});

test.describe("Native application source-bound Core and GPU graph admission", () => {
  test("every selected graph and Core must match source and artifact hashes", t => {
    const f = application(t); assert.equal(f.admit().core, f.core.objectSha256); assert.equal(f.admit().portableCore, true);
    assert.equal(f.admit().graphs[0]?.factory, f.graph.factory);
  });
  test("legacy Core proof receipts must be rebuilt, not freshly stamped", t => {
    const f = application(t); f.put("core/inputs.json", '{"gpu":true}'); assert.throws(f.admit, /Core source inputs.*missing/);
  });
  test("CPU-only stub cores cannot enter an application", t => {const f = application(t); f.core.gpu = false; f.save(); assert.throws(f.admit, /Core artifact/);});
  for (const value of [false, undefined, "1"]) {
    test("final links reject a non-portable or unproven Core: " + JSON.stringify(value), t => {
      const f = application(t); Object.assign(f.core, {portable: value}); f.save();
      assert.throws(f.admit, /portable Core.*rebuild/);
    });
  }
  test("changed Core artifact cannot borrow its source receipt", t => {const f = application(t); f.put("core/cpu-core.a", "changed"); assert.throws(f.admit, /Core artifact/);});
  test("changed GPU artifact cannot borrow its source receipt", t => {const f = application(t); f.put("gpu/compiler.o", "changed"); assert.throws(f.admit, /GPU object/);});
  test("changed GPU includes invalidate the application before linking", t => {const f = application(t); f.put("sycl/kernel.inc", "changed"); assert.throws(f.admit, /GPU source inputs.*stale/);});
  test("changed Core includes invalidate the application before linking", t => {const f = application(t); f.put("native/core.h", "changed"); assert.throws(f.admit, /Core source inputs.*stale/);});
  test("missing factory receipts cannot silently drop a compiler", t => {const f = application(t); f.inputs.receipts = []; assert.throws(f.admit, /every GPU factory/);});
  test("unknown factories cannot stand in for the maintained factory set", t => {const f = application(t); f.graph.factory = "unknown"; f.save(); assert.throws(f.admit, /unknown/);});
  test("duplicate factory receipts cannot hide another missing graph", t => {
    const f = application(t); f.factories.push("mom_native_gpu_dpcpp"); f.inputs.receipts.push(f.record); assert.throws(f.admit, /duplicated/);
  });
  test("receipts must refer to the actual selected object", t => {const f = application(t); f.inputs.objects = [{object: f.cpu}]; assert.throws(f.admit, /selected object/);});
  test("relative receipt paths are not resolved through a changing cwd", t => {const f = application(t); f.inputs.receipts = ["gpu/result.json"]; assert.throws(f.admit, /every GPU factory/);});
  test("new conditional GPU includes invalidate old receipts", t => {const f = application(t); f.put("sycl/extra.inc", "new"); assert.throws(f.admit, /GPU source inputs.*stale/);});
  test("JS-only edits do not require recompiling Core or GPU graphs", t => {const f = application(t); const before = f.admit(); f.put("mom.js", "changed"); assert.deepEqual(f.admit(), before);});
  test("an unreceipted cached runtime helper cannot enter the factory object list", t => {
    const f = application(t); f.inputs.objects.push({object: f.put("runtime-helper.o", "stale")});
    assert.throws(f.admit, /runtime helpers are compiled automatically/);
  });
  test("duplicate selected factory objects are rejected before linking", t => {
    const f = application(t); f.inputs.objects.push({object: f.gpu}); assert.throws(f.admit, /unique source-bound factories/);
  });
});

test.describe("Native application separates SDK artifacts from maintained source objects", () => {
  test("final application rejects duplicate HCF registrations in an admitted ACPP graph", t => {
    const f = application(t), bytes = Buffer.concat([Buffer.from(
      "object-id=123\ngenerator=hipSYCL SSCP\nstart=0\nsize=4\n__acpp_hcf_binary_appendix"), Buffer.from([0x42, 0x43, 0xc0, 0xde])]);
    f.put("gpu/compiler.o", bytes); f.graph.objectSha256 = fileHash(f.gpu);
    f.graph.factory = "mom_native_gpu_acpp_cuda"; f.factories[0] = f.graph.factory; f.save();
    assert.doesNotThrow(f.admit);
    f.put("gpu/compiler.o", Buffer.concat([bytes, bytes])); f.graph.objectSha256 = fileHash(f.gpu); f.save();
    assert.throws(f.admit, /Duplicate HCF object ID 123/);
  });
  test("empty explicit SDK lists support toolchains whose runtime is entirely archived", t => {
    const f = application(t); assert.deepEqual(f.admit().sdk, []);
  });
  test("SDK object receipts include content hashes", t => {
    const f = application(t), sdk = f.put("sdk/compiler.o", "SDK object"); f.inputs.sdkObjects.push({object: sdk});
    assert.deepEqual(f.admit().sdk, [{object: sdk, sha256: fileHash(sdk)}]);
  });
  test("a changed SDK object changes final-link admission even with unchanged application sources", t => {
    const f = application(t), sdk = f.put("sdk/compiler.o", "before"); f.inputs.sdkObjects.push({object: sdk});
    const before = f.admit(); f.put("sdk/compiler.o", "after"); assert.notDeepEqual(f.admit(), before);
  });
  test("SDK artifacts cannot also occupy a source-bound factory slot", t => {
    const f = application(t); f.inputs.sdkObjects.push({object: f.gpu}); assert.throws(f.admit, /Invalid or duplicate/);
  });
  test("duplicate SDK objects are rejected", t => {
    const f = application(t), sdk = f.put("sdk/compiler.o", "SDK"); f.inputs.sdkObjects.push({object: sdk}, {object: sdk});
    assert.throws(f.admit, /Invalid or duplicate/);
  });
  test("relative SDK objects cannot resolve through an incidental working directory", t => {
    const f = application(t); f.inputs.sdkObjects.push({object: "relative.o"}); assert.throws(f.admit, /Invalid or duplicate/);
  });
  test("SDK object paths reject control characters", t => {
    const f = application(t); f.inputs.sdkObjects.push({object: path.join(f.root, "newline\n.o")}); assert.throws(f.admit, /Invalid or duplicate/);
  });
  test("excessive SDK paths fail with a bounded diagnostic before filesystem access", t => {
    const f = application(t); f.inputs.sdkObjects.push({object: "/" + "x".repeat(4096)});
    assert.throws(f.admit, error => error instanceof Error && error.message.length < 100 && /Invalid/.test(error.message));
  });
  test("missing SDK objects cannot be silently dropped", t => {
    const f = application(t); f.inputs.sdkObjects.push({object: path.join(f.root, "missing.o")}); assert.throws(f.admit, /ENOENT/);
  });
  test("directory SDK inputs cannot be mistaken for object payloads", t => {
    const f = application(t); f.inputs.sdkObjects.push({object: f.root}); assert.throws(f.admit, /regular file/);
  });
  test("an excessive SDK list fails before opening its objects", t => {
    const f = application(t); f.inputs.sdkObjects = Array(65).fill({object: "missing"}); assert.throws(f.admit, /bounded SDK/);
  });
  test("legacy manifests without an SDK classification fail explicitly", t => {
    const f = application(t); Reflect.deleteProperty(f.inputs, "sdkObjects"); assert.throws(f.admit, /explicit bounded SDK/);
  });
});

test.describe("Native application recipe and runtime freshness", () => {
  test("unchanged runtime sources and recipes produce identical receipts", t => {
    const f = fixture(t); assert.deepEqual(nativeApplicationBuildInputs(f.root), nativeApplicationBuildInputs(f.root));
  });
  for (const file of ["runtime/host.cpp", "scripts/build-native-application.js", "scripts/build-source-inputs.js", "scripts/native-bundle.js", "scripts/native-resources.js", "sycl/cuda-images.cpp", "sycl/cuda-images.h"]) {
    test(`mid-build ${file} edits invalidate application publication`, t => {
      const f = fixture(t), before = nativeApplicationBuildInputs(f.root); f.put(file, "changed");
      assert.throws(() => verifySourceInputs(before, nativeApplicationBuildInputs(f.root), "Native controller build"), /stale.*rebuild/);
    });
  }
  test("backdating the controller recipe cannot keep an old receipt valid", t => {
    const f = fixture(t), before = nativeApplicationBuildInputs(f.root), changed = f.put("scripts/build-native-application.js", "changed");
    fs.utimesSync(changed, 1, 1);
    assert.throws(() => verifySourceInputs(before, nativeApplicationBuildInputs(f.root), "Application"), /stale/);
  });
  test("a newly introduced runtime helper must be compiled into a new application generation", t => {
    const f = fixture(t), before = nativeApplicationBuildInputs(f.root); f.put("runtime/new-helper.cpp", "new");
    assert.throws(() => verifySourceInputs(before, nativeApplicationBuildInputs(f.root), "Application"), /stale/);
  });
  test("controller recipe changes do not invalidate native Core or GPU source receipts", t => {
    const f = fixture(t), cpu = nativeSourceInputs(f.root, "cpu"), gpu = nativeSourceInputs(f.root, "gpu");
    f.put("scripts/build-native-application.js", "changed"); f.put("scripts/native-bundle.js", "changed");
    assert.deepEqual(nativeSourceInputs(f.root, "cpu"), cpu); assert.deepEqual(nativeSourceInputs(f.root, "gpu"), gpu);
  });
  test("controller recipe changes do not enter Engineering's incremental object graph", t => {
    const f = fixture(t), before = sourceInputs(f.root); f.put("scripts/build-native-application.js", "changed");
    assert.deepEqual(sourceInputs(f.root), before);
  });
  test("JS-only edits still flow through the separate exact application-bundle check", t => {
    const f = fixture(t), before = nativeApplicationBuildInputs(f.root); f.put("mom.js", "changed");
    assert.deepEqual(nativeApplicationBuildInputs(f.root), before);
  });
});

test.describe("Shared native CUDA image application admission", () => {
  /** @param {import('node:test').TestContext} t */
  function images(t) {
    const f = fixture(t), include = f.put("images/cuda-images.inc", "immutable image bytes"), sidecar = f.put("images/cuda-images.inc.json", "source admission fixture");
    const files = {[include]: fileHash(include), [sidecar]: fileHash(sidecar)};
    const reader = (/** @type {string} */ file, /** @type {string} */ root) => {
      assert.equal(file, include); assert.equal(root, f.root); return {files};
    };
    return {...f, include, sidecar, files, reader, admit: (/** @type {unknown} */ value = files) => nativeApplicationCudaImages(f.root, value, reader)};
  }
  test("non-image applications never invoke image admission", () => {
    assert.equal(nativeApplicationCudaImages(repo, undefined, () => {throw new Error("unexpected reader");}), null);
  });
  test("one include and adjacent sidecar resolve to one application table", t => {
    const f = images(t); assert.deepEqual(f.admit(), {include: f.include, files: f.files});
  });
  for (const value of [null, [], "bundle.inc"]) {
    test(`rejects a non-map image claim (${JSON.stringify(value)})`, t => {const f = images(t); assert.throws(() => f.admit(value), /exact CUDA image hash map/);});
  }
  test("a missing sidecar cannot qualify image bytes", t => {
    const f = images(t); assert.throws(() => f.admit({[f.include]: f.files[f.include]}), /adjacent sidecar/);
  });
  test("extra files cannot hide among the two admitted image inputs", t => {
    const f = images(t); assert.throws(() => f.admit({...f.files, [path.join(f.root, "extra.o")]: "a".repeat(64)}), /adjacent sidecar/);
  });
  test("malformed hashes are rejected before invoking the image reader", t => {
    const f = images(t); assert.throws(() => f.admit({...f.files, [f.include]: "invalid"}), /adjacent sidecar/);
  });
  test("the selected map must equal source admission, not merely rehash itself", t => {
    const f = images(t); assert.throws(() => f.admit({...f.files, [f.include]: "a".repeat(64)}), /differ from source admission/);
  });
  for (const key of ["include", "sidecar"]) {
    test(`changed ${key} bytes invalidate admitted images`, t => {
      const f = images(t); fs.appendFileSync(key === "include" ? f.include : f.sidecar, "changed");
      assert.throws(() => f.admit(), /differ from source admission/);
    });
  }
  test("each CUDA receipt must bind both identical admitted files", t => {
    const f = images(t), receipt = {cudaImages: f.include, sdkInputs: {...f.files}};
    assert.doesNotThrow(() => verifyCudaImageReceiptInputs(receipt, f.files, "dpcpp"));
    delete receipt.sdkInputs[f.sidecar];
    assert.throws(() => verifyCudaImageReceiptInputs(receipt, f.files, "dpcpp"), /does not bind/);
  });
  test("file hashes alone cannot masquerade as the image-mode selection", t => {
    const f = images(t); assert.throws(() => verifyCudaImageReceiptInputs({sdkInputs: f.files}, f.files, "dpcpp"), /does not bind/);
  });
  test("an image-mode factory cannot omit the shared application bundle", t => {
    const f = application(t); f.graph.factory = "mom_native_gpu_dpcpp"; f.factories[0] = f.graph.factory;
    Object.assign(f.graph, {cudaImages: "/omitted/cuda-images.inc"}); f.save();
    assert.throws(f.admit, /matching shared application image bundle/);
  });
});
