"use strict";

const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {spawnSync} = require("node:child_process");
const test = require("node:test"), assert = require("node:assert/strict");
const {fileHash} = require("../scripts/build-source-inputs");
const {isolateCoffCompilerArchives, verifyCoffArchive, auditCoffArchiveInputs} = require("../scripts/native-coff-archive");
const {validateProducerReceipt, validateArchiveGraph, validateFreshOutput, validateStageAncestry, verifyRecordedArtifacts,
  validateToolsManifest, validateRecipeRecord, readPinnedProducerResult, hashTools, inspectWindowsJitSdk,
  windowsJitIsolationHandoff} = require("../scripts/isolate-native-windows-jit");

/** @param {Array<{name: string, data: Buffer}>} rows */
function makeArchive(rows) {
  const table = Buffer.concat(rows.map(row => Buffer.from(row.name + "/\n"))), chunks = [Buffer.from("!<arch>\n")];
  const header = (/** @type {string} */ token, /** @type {number} */ size) => Buffer.from(token.padEnd(16) + "0".padEnd(12) + "0".padEnd(6) + "0".padEnd(6) +
    "100644".padEnd(8) + String(size).padEnd(10) + "`\n", "ascii");
  chunks.push(header("//", table.length), table, table.length % 2 ? Buffer.from("\n") : Buffer.alloc(0));
  let offset = 0;
  for (const row of rows) {
    chunks.push(header("/" + offset, row.data.length), Buffer.from(row.data), row.data.length % 2 ? Buffer.from("\n") : Buffer.alloc(0));
    offset += Buffer.byteLength(row.name + "/\n");
  }
  return Buffer.concat(chunks);
}

/** @returns {Buffer} Minimal x64 COFF object with one initialized section. */
function minimalCoff() {
  const object = Buffer.alloc(64);
  object.writeUInt16LE(0x8664, 0); object.writeUInt16LE(1, 2);
  object.write(".text", 20, 8, "ascii"); object.writeUInt32LE(4, 36); object.writeUInt32LE(60, 40); object.writeUInt32LE(0x60000020, 56);
  object.set([0xc3, 0x90, 0x90, 0x90], 60);
  return object;
}

/** @param {import("node:test").TestContext} t @param {Array<{name: string, data: Buffer}>} rows @param {Array<{name: string, data: Buffer}>} [objects] */
function auditFixture(t, rows, objects = rows) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-coff-input-audit-")), build = path.join(root, "build");
  fs.mkdirSync(build);
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  for (const row of objects) {
    if (row.name.startsWith("../")) {continue;}
    const file = path.join(build, ...row.name.split("/")); fs.mkdirSync(path.dirname(file), {recursive: true});
    if (!fs.existsSync(file)) {fs.writeFileSync(file, row.data);}
  }
  const archive = path.join(root, "input.lib"); fs.writeFileSync(archive, makeArchive(rows));
  return {root, build, archive};
}

test.describe("Exact COFF archive build-input audit", () => {
  test("matches a headerless x64 object set using safe relative member names", t => {
    const name = "src/runtime/CMakeFiles/acpp-rt.dir/runtime.obj", data = minimalCoff(), selected = auditFixture(t, [{name, data}]);
    const result = auditCoffArchiveInputs(selected.archive, selected.build, [name]);
    assert.equal(result.members, 1); assert.deepEqual(result.objects, [{path: name, bytes: data.length, sha256: fileHash(path.join(selected.build, ...name.split("/")))}]);
    assert.match(result.archiveSha256, /^[0-9a-f]{64}$/);
  });

  test("rejects ELF input, changed member bytes, extra/missing/duplicate members and traversal", t => {
    const name = "owner/expected.obj", original = minimalCoff();
    const elf = Buffer.alloc(64); elf.set([0x7f, 0x45, 0x4c, 0x46]);
    const cases = [
      {rows: [{name, data: elf}], expected: [name], error: /COFF payload header/},
      {rows: [{name, data: Buffer.from(original).fill(0, 60)}], objects: [{name, data: original}], expected: [name], error: /differs from compiled object/},
      {rows: [{name, data: original}, {name: "owner/extra.obj", data: original}], expected: [name], error: /Unexpected COFF archive member/},
      {rows: [{name, data: original}, {name: "../escape.obj", data: original}], expected: [name], error: /Unsafe COFF archive member path/},
      {rows: [{name, data: original}, {name, data: original}], expected: [name], error: /Duplicate COFF archive member/},
    ];
    for (const item of cases) {
      const selected = auditFixture(t, item.rows, item.objects);
      assert.throws(() => auditCoffArchiveInputs(selected.archive, selected.build, item.expected), item.error);
    }
    const other = "owner/other.obj", missing = auditFixture(t, [{name: other, data: original}], [{name, data: original}, {name: other, data: original}]);
    assert.throws(() => auditCoffArchiveInputs(missing.archive, missing.build, [name, other]), /missing expected/);
  });

  test("rejects symlinked expected objects and noncanonical paths without mutation", t => {
    const name = "owner/linked.obj", data = minimalCoff(), selected = auditFixture(t, [{name, data}]), target = path.join(selected.build, "owner", "target.obj");
    fs.renameSync(path.join(selected.build, "owner", "linked.obj"), target); fs.symlinkSync(target, path.join(selected.build, "owner", "linked.obj"));
    assert.throws(() => auditCoffArchiveInputs(selected.archive, selected.build, [name]), /canonical regular file/);
    assert.throws(() => auditCoffArchiveInputs(selected.archive, selected.build, ["../outside.obj"]), /Invalid or duplicate expected/);
    assert.equal(fs.readFileSync(selected.archive).subarray(0, 8).toString(), "!<arch>\n");
  });
});

/** @param {string} name */
function discover(name) {
  const bin = process.env["MOM_TEST_LLVM_BIN"], suffix = process.platform === "win32" ? ".exe" : "";
  if (bin) {const file = path.resolve(bin, name + suffix); if (fs.statSync(file, {throwIfNoEntry: false})?.isFile()) {return file;}}
  const result = spawnSync(process.platform === "win32" ? "where.exe" : "which", [name + suffix], {encoding: "utf8", timeout: 5000});
  return result.status === 0 ? result.stdout.trim().split(/\r?\n/).find(file => path.isAbsolute(file) && fs.statSync(file, {throwIfNoEntry: false})?.isFile()) : undefined;
}
const tools = Object.fromEntries(["clang-cl", "llvm-lib", "llvm-ar", "llvm-nm", "llvm-objcopy", "lld-link"].map(name => [name, discover(name)]));
const skip = Object.values(tools).some(value => !value) ? "headerless COFF LLVM toolchain unavailable; set MOM_TEST_LLVM_BIN" : false;

/** @param {string} name @param {string[]} args @param {string} cwd */
function run(name, args, cwd) {
  const command = tools[name]; assert.ok(command);
  const result = spawnSync(command, args, {cwd, encoding: "utf8", timeout: 30000, maxBuffer: 4 << 20});
  assert.equal(result.status, 0, result.stderr || result.error?.message); return result.stdout || "";
}

/** Tiny real C++ archive graph: cross-member references, long member names, C entries and
 * shared CRT/OS-style imports. No Windows SDK, executable runtime, network or GPU is needed.
 * @param {import("node:test").TestContext} t @param {boolean} [directive] */
function fixture(t, directive = false) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-coff-archive-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const archives = [];
  for (const [name, value] of [["one", 7], ["two", 11]]) {
    const helper = path.join(root, name + "-long-compiler-helper-name.cpp"), factory = path.join(root, name + "-factory.cpp");
    fs.writeFileSync(helper, [
      directive ? '#pragma comment(linker, "/include:?value@llvm@@3HA")' : "",
      `namespace llvm { int value = ${value}; int work() { return value; } }`,
      "struct Plain { static int work(); }; int Plain::work() { return llvm::work(); }",
      'extern "C" int HandleAbort() { return Plain::work(); }',
      'extern "C" int LLVMInitializeFixture() { return 1; }',
      `extern "C" int helper_${name}() { return HandleAbort() + LLVMInitializeFixture(); }`,
    ].join("\n") + "\n");
    fs.writeFileSync(factory, `extern "C" int externalBridge(); extern "C" int helper_${name}();\n` +
      `extern "C" int factory_${name}() { return helper_${name}() + externalBridge(); }\n`);
    for (const file of [helper, factory]) {run("clang-cl", ["--target=x86_64-pc-windows-msvc", "-GS-", "-c", file, "-o", file + ".obj"], root);}
    const archive = path.join(root, name + ".lib");
    run("llvm-lib", ["/out:" + archive, path.basename(helper) + ".obj", path.basename(factory) + ".obj"], root); archives.push(archive);
  }
  const entry = path.join(root, "entry.cpp");
  fs.writeFileSync(entry, 'extern "C" int factory_one(); extern "C" int factory_two(); extern "C" int externalBridge() { return 1; } extern "C" int mainCRTStartup() { return factory_one() + factory_two(); }\n');
  run("clang-cl", ["--target=x86_64-pc-windows-msvc", "-GS-", "-c", entry, "-o", entry + ".obj"], root);
  assert.ok(tools["llvm-objcopy"]); assert.ok(tools["llvm-nm"]); assert.ok(tools["llvm-ar"]);
  const options = {archives: archives.slice(0, 1), tools: {llvmObjcopy: tools["llvm-objcopy"], llvmNm: tools["llvm-nm"], llvmAr: tools["llvm-ar"]}, output: path.join(root, "isolated"), prefix: "mom_compiler_", entryPoints: ["factory_one"]};
  return {root, archives, entry: entry + ".obj", options};
}

test.describe("Source-preserving Windows compiler archives", {skip}, () => {
  test("resolves real duplicate compiler definitions without suppressing linker errors", t => {
    const selected = fixture(t), command = tools["lld-link"]; assert.ok(command);
    const args = ["/nodefaultlib", "/entry:mainCRTStartup", "/subsystem:console", "/out:" + path.join(selected.root, "proof.exe")];
    const before = spawnSync(command, [...args, ...selected.archives, selected.entry], {encoding: "utf8", timeout: 30000});
    assert.notEqual(before.status, 0); assert.match(before.stderr, /duplicate symbol/);
    const result = isolateCoffCompilerArchives(selected.options);
    run("lld-link", [...args, ...result.transformed.map(row => row.output), selected.archives[1] || "", selected.entry], selected.root);
    assert.equal(result.transformed[0]?.members, 2);
  });
  test("preserves the MSVC x64 _setjmp hidden frame argument when builtin lowering is enabled", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-coff-setjmp-abi-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const source = path.join(root, "setjmp-abi.c");
    fs.writeFileSync(source, [
      "extern int _setjmp(void *buffer);",
      "__declspec(noinline) int probe(void *buffer) { return _setjmp(buffer); }",
    ].join("\n") + "\n");
    /** @param {string} label @param {string[]} extraArgs */
    const emit = (label, extraArgs) => {
      const output = path.join(root, label + ".ll");
      run("clang-cl", ["--driver-mode=gcc", "--target=x86_64-pc-windows-msvc", "-O2", "-S", "-emit-llvm", ...extraArgs, source, "-o", output], root);
      const ir = fs.readFileSync(output, "utf8");
      const calls = ir.split(/\r?\n/).filter(line => /\bcall\b/.test(line) && line.includes("@_setjmp("));
      assert.equal(calls.length, 1, "fixture must contain exactly one direct _setjmp call");
      const call = calls[0]; assert.ok(call);
      const start = call.indexOf("@_setjmp(") + "@_setjmp(".length, end = call.indexOf(")", start);
      assert.ok(end > start, "fixture must emit a parseable _setjmp argument list");
      return {ir, args: call.slice(start, end).split(",").map(arg => arg.trim())};
    };
    const builtin = emit("builtin", []), disabled = emit("no-builtin", ["-fno-builtin"]);
    assert.equal(builtin.args.length, 2, "MSVC x64 builtin lowering passes the frame as _setjmp's hidden second argument");
    assert.equal(disabled.args.length, 1, "-fno-builtin is a negative control and removes Clang's MSVC setjmp lowering");
    const frame = builtin.ir.match(/^\s*(%[-\w.$]+)\s*=\s*(?:tail\s+)?call\b[^\n]*@llvm\.frameaddress(?:\.[\w.]+)?\(i32 0\)/m)?.[1];
    assert.ok(frame, "builtin lowering must compute the frame address");
    assert.ok(builtin.args[1]?.includes(frame), "_setjmp's hidden argument must be the computed frame address");
  });
  test("keeps public C bridge and external imports while isolating non-namespace C++ and LLVM C definitions", t => {
    const selected = fixture(t), result = isolateCoffCompilerArchives(selected.options), text = fs.readFileSync(result.map, "utf8");
    assert.match(text, /\?work@Plain@@/); assert.match(text, /^LLVMInitializeFixture mom_compiler_LLVMInitializeFixture$/m);
    assert.match(text, /^HandleAbort mom_compiler_HandleAbort$/m); assert.doesNotMatch(text, /factory_one|externalBridge/);
    const row = result.transformed[0]; assert.ok(row);
    const symbols = run("llvm-nm", ["-P", "-g", row.output], selected.root);
    assert.match(symbols, /^factory_one T /m); assert.match(symbols, /^externalBridge U /m);
    assert.equal(row.inputSha256, fileHash(row.input)); assert.equal(row.sha256, fileHash(row.output));
    assert.equal(result.executed, false); assert.equal(result.publicationQualified, false);
  });
  test("an explicit private C helper does not rename unrelated runtime ABI definitions", t => {
    const selected = fixture(t), result = isolateCoffCompilerArchives({...selected.options, privateSymbols: ["HandleAbort"]});
    assert.equal(result.symbols, 1); assert.equal(fs.readFileSync(result.map, "utf8"), "HandleAbort mom_compiler_HandleAbort\n");
    const row = result.transformed[0]; assert.ok(row);
    const symbols = run("llvm-nm", ["-P", "-g", "--defined-only", row.output], selected.root);
    assert.match(symbols, /^factory_one T /m); assert.match(symbols, /^LLVMInitializeFixture T /m); assert.match(symbols, /^\?work@Plain@@/m);
    const other = path.join(selected.root, "invalid");
    assert.throws(() => isolateCoffCompilerArchives({...selected.options, output: other, privateSymbols: ["factory_one"]}), /Invalid explicit private/);
    assert.equal(fs.existsSync(other), false);
  });
  test("repeated transforms are deterministic and retain only compact evidence", t => {
    const selected = fixture(t), first = isolateCoffCompilerArchives(selected.options);
    const second = isolateCoffCompilerArchives({...selected.options, output: path.join(selected.root, "second")});
    assert.equal(first.mapSha256, second.mapSha256); assert.equal(first.transformed[0]?.sha256, second.transformed[0]?.sha256);
    assert.equal(first.transformed[0]?.checkedMemberSha256, second.transformed[0]?.checkedMemberSha256);
    assert.equal(fs.existsSync(path.join(first.transformed[0]?.output || "", "before.obj")), false);
    assert.equal(fs.existsSync(path.join(selected.options.output, "before.obj")), false);
    assert.equal(Object.keys(first.sourceHashes).length, 3);
    assert.ok(fs.statSync(path.join(selected.options.output, "result.json")).size < 16384);
  });
  test("wrong symbol maps cannot pass the independent member audit", t => {
    const selected = fixture(t), result = isolateCoffCompilerArchives(selected.options), row = result.transformed[0]; assert.ok(row);
    assert.throws(() => verifyCoffArchive(row.input, row.output, new Map(), selected.root), /preservation failed/);
    assert.equal(fs.existsSync(path.join(selected.root, "before.obj")), false);
  });
  test("independent audit never deletes pre-existing scratch files", t => {
    const selected = fixture(t), input = selected.archives[0]; assert.ok(input);
    const marker = path.join(selected.root, "before.obj"); fs.writeFileSync(marker, "preserve");
    assert.throws(() => verifyCoffArchive(input, input, new Map(), selected.root), /EEXIST/);
    assert.equal(fs.readFileSync(marker, "utf8"), "preserve");
  });
  test("rewrites symbol-bearing linker roots without altering other object state", t => {
    const selected = fixture(t, true), result = isolateCoffCompilerArchives(selected.options), row = result.transformed[0]; assert.ok(row);
    assert.equal(row.rewrittenMembers, 1); assert.equal(row.members, 2);
    const raw = path.join(selected.root, "rename-only.lib");
    run("llvm-objcopy", ["--redefine-syms=" + result.map, row.input, raw], selected.root);
    const names = new Map(fs.readFileSync(result.map, "utf8").trim().split("\n").map(line => {const [from, to] = line.split(" "); assert.ok(from && to); return [from, to];}));
    assert.throws(() => verifyCoffArchive(row.input, raw, names, selected.root), /preservation failed/);
    run("lld-link", ["/nodefaultlib", "/entry:mainCRTStartup", "/subsystem:console", "/out:" + path.join(selected.root, "proof.exe"), row.output, selected.archives[1] || "", selected.entry], selected.root);
    assert.equal(fs.existsSync(path.join(selected.options.output, "directives")), false);
  });
  test("preserves duplicate member names and order while rebuilding a GNU archive index", t => {
    const selected = fixture(t, true), input = selected.archives[0]; assert.ok(input);
    const names = run("llvm-ar", ["t", input], selected.root).trim().split(/\r?\n/), files = [];
    for (const [index, name] of names.entries()) {
      const directory = path.join(selected.root, "member-" + index); fs.mkdirSync(directory);
      const file = path.join(directory, "duplicate.obj"); fs.copyFileSync(path.join(selected.root, name), file); files.push(file);
    }
    const duplicate = path.join(selected.root, "duplicate.lib"); run("llvm-ar", ["--format=gnu", "qc", duplicate, ...files], selected.root);
    const result = isolateCoffCompilerArchives({...selected.options, archives: [duplicate]}), row = result.transformed[0]; assert.ok(row);
    assert.equal(row.rewrittenMembers, 1); assert.equal(row.members, 2);
    assert.deepEqual(run("llvm-ar", ["t", row.output], selected.root).trim().split(/\r?\n/), ["duplicate.obj", "duplicate.obj"]);
    run("lld-link", ["/nodefaultlib", "/entry:mainCRTStartup", "/subsystem:console", "/out:" + path.join(selected.root, "proof.exe"), row.output, selected.archives[1] || "", selected.entry], selected.root);
  });
  test("rejects missing bridges and retains failed-build evidence", t => {
    const selected = fixture(t);
    assert.throws(() => isolateCoffCompilerArchives({...selected.options, entryPoints: ["absent"]}), /Missing.*C bridge/);
    assert.equal(fs.existsSync(path.join(selected.options.output, "definitions-0.log")), true);
    assert.equal(fs.existsSync(path.join(selected.options.output, "result.json")), false);
  });
  test("rejects existing outputs without overwriting their contents", t => {
    const selected = fixture(t); fs.mkdirSync(selected.options.output);
    const marker = path.join(selected.options.output, "keep"); fs.writeFileSync(marker, "preserve");
    assert.throws(() => isolateCoffCompilerArchives(selected.options), /Invalid.*options/);
    assert.equal(fs.readFileSync(marker, "utf8"), "preserve");
  });
  test("rejects duplicate physical archives and case-colliding Windows filenames before mutation", t => {
    const selected = fixture(t), input = selected.archives[0]; assert.ok(input);
    assert.throws(() => isolateCoffCompilerArchives({...selected.options, archives: [input, input]}), /distinct paths and names/);
    const other = path.join(selected.root, "ONE.lib"); fs.copyFileSync(input, other);
    assert.throws(() => isolateCoffCompilerArchives({...selected.options, archives: [input, other]}), /distinct paths and names/);
    assert.equal(fs.existsSync(selected.options.output), false);
  });
  test("rejects relative paths, control characters and invalid public symbols before mutation", t => {
    const selected = fixture(t);
    for (const options of [{...selected.options, output: "relative"}, {...selected.options, prefix: "bad prefix"},
      {...selected.options, entryPoints: ["bad\nname"]}, {...selected.options, entryPoints: ["factory_one\n"]},
      {...selected.options, prefix: "mom_\n"}, {...selected.options, entryPoints: ["factory_one", "factory_one"]}]) {
      assert.throws(() => isolateCoffCompilerArchives(options), /Invalid.*options/);
    }
    assert.equal(fs.existsSync(selected.options.output), false);
  });
  test("rejects thin archives and truncated regular archives before mutation", t => {
    const selected = fixture(t), thin = path.join(selected.root, "thin.lib"), short = path.join(selected.root, "short.lib");
    fs.writeFileSync(thin, "!<thin>\n"); fs.writeFileSync(short, "!<ar");
    for (const input of [thin, short]) {assert.throws(() => isolateCoffCompilerArchives({...selected.options, archives: [input]}), /regular|bounds/);}
    assert.equal(fs.existsSync(selected.options.output), false);
  });
  test("independent audit rejects changed member identities", t => {
    const selected = fixture(t), input = selected.archives[0]; assert.ok(input);
    const other = path.join(selected.root, "changed.lib"), data = fs.readFileSync(input);
    // Long names live in the archive name table; changing a harmless member spelling must
    // fail identity validation even if all underlying object code is unchanged.
    const marker = Buffer.from("long-compiler-helper-name"), index = data.indexOf(marker); assert.ok(index > 0);
    data[index] = "X".charCodeAt(0); fs.writeFileSync(other, data);
    assert.throws(() => verifyCoffArchive(input, other, new Map(), selected.root), /identity changed/);
  });
  test("independent audit rejects a code-byte change with unchanged symbols and relocations", t => {
    const selected = fixture(t), input = selected.archives[0]; assert.ok(input);
    const other = path.join(selected.root, "corrupt-code.lib"), data = fs.readFileSync(input), text = data.indexOf(Buffer.from(".text\0"));
    assert.ok(text > 20); const objectStart = text - 20, raw = data.readUInt32LE(text + 20); assert.ok(raw > 0);
    const position = objectStart + raw; assert.ok(position < data.length); data[position] = (data[position] || 0) ^ 1; fs.writeFileSync(other, data);
    assert.throws(() => verifyCoffArchive(input, other, new Map(), selected.root), /preservation failed/);
  });
  test("independent audit rejects malformed member sizes without leaving scratch objects", t => {
    const selected = fixture(t), input = selected.archives[0]; assert.ok(input);
    const other = path.join(selected.root, "truncated.lib"), data = fs.readFileSync(input);
    data.write("9999999999", 8 + 48, "ascii"); fs.writeFileSync(other, data);
    assert.throws(() => verifyCoffArchive(input, other, new Map(), selected.root), /boundary/);
    assert.equal(fs.existsSync(path.join(selected.root, "after.obj")), false);
  });
});

test.describe("Windows native compiler stack policy", {skip}, () => {
  test("native executables inherit LLVM's stack reserve, not the one-MiB PE default", t => {
    const readobj = discover("llvm-readobj"); if (!readobj) {t.skip("llvm-readobj unavailable for PE header inspection"); return;}
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-stack-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const repo = path.resolve(__dirname, ".."), source = path.join(root, "main.cpp");
    fs.writeFileSync(source, "#ifdef MOM_TEST_NATIVE_STACK\n#include " + JSON.stringify(path.join(repo, "runtime/windows-stack.h")) +
      "\n#endif\nextern \"C\" int mainCRTStartup() {return 0;}\n");
    for (const native of [false, true]) {
      const object = path.join(root, native ? "native.obj" : "baseline.obj"), executable = object + ".exe";
      run("clang-cl", ["--target=x86_64-pc-windows-msvc", "-GS-", "-c", ...(native ? ["-DMOM_TEST_NATIVE_STACK"] : []), source, "-o", object], root);
      run("lld-link", ["/nologo", "/machine:x64", "/subsystem:console", "/entry:mainCRTStartup", "/nodefaultlib", "/out:" + executable, object], root);
      /** @type {import("node:child_process").SpawnSyncReturns<string>} */
      const result = spawnSync(readobj, ["--file-headers", executable], {encoding: "utf8", timeout: 5000}); assert.equal(result.status, 0, result.stderr);
      assert.equal(Number(result.stdout.match(/SizeOfStackReserve: (\d+)/)?.[1]), native ? 10000000 : 1048576);
      assert.equal(Number(result.stdout.match(/SizeOfStackCommit: (\d+)/)?.[1]), 4096, "Reserve must not become an eager commit");
    }
    assert.match(fs.readFileSync(path.join(repo, "runtime/main.cpp"), "utf8"), /#include "windows-stack.h"/);
    assert.match(fs.readFileSync(path.join(repo, "tests/reference/native-dpcpp-static.cpp"), "utf8"), /#include "..\/..\/runtime\/windows-stack.h"/);
  });
});

test.describe("Windows source-JIT archive handoff", () => {
  /** @param {import("node:test").TestContext} t */
  function graphFixture(t) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-jit-graph-")), build = path.join(root, "build"); fs.mkdirSync(path.join(build, "lib"), {recursive: true});
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const names = ["sycl-jit", ...Array.from({length: 49}, (_, index) => `llvm-fixture-${String(index).padStart(2, "0")}`)];
    const archives = names.map(name => path.join(build, "lib", name + ".lib"));
    for (const archive of archives) {fs.writeFileSync(archive, "!<arch>\n");}
    fs.mkdirSync(path.join(build, "bin")); const proof = path.join(build, "bin", "native-dpcpp-jit.exe"); fs.writeFileSync(proof, "proof bytes");
    const command = `lld-link /out:bin/native-dpcpp-jit.exe ${names.map(name => `lib/${name}.lib`).join(" ")}`;
    return {root, build, names, archives, proof, command};
  }

  test("derives the complete ordered archive set from the producer link command", t => {
    const selected = graphFixture(t), support = /** @type {string[]} */ ([]), command = selected.command;
    const stageLibraries = selected.archives.slice(), resultLibraries = selected.archives.map(file => ({path: file}));
    const graph = validateArchiveGraph(command, selected.build, support, stageLibraries, resultLibraries);
    assert.deepEqual(graph.archives, selected.archives); assert.equal(graph.linkLine, command);
  });

  test("rejects missing, reordered, subset and duplicate-target graph evidence", t => {
    const selected = graphFixture(t), support = /** @type {string[]} */ ([]), rows = selected.archives.map(file => ({path: file}));
    assert.throws(() => validateArchiveGraph(selected.command, selected.build, support, selected.archives.slice(1), rows), /differs/);
    assert.throws(() => validateArchiveGraph(selected.command, selected.build, support, [...selected.archives].reverse(), rows), /differs/);
    assert.throws(() => validateArchiveGraph(selected.command, selected.build, support, selected.archives, rows.slice(0, -1)), /differs/);
    assert.throws(() => validateArchiveGraph(selected.command + "\n" + selected.command, selected.build, support, selected.archives, rows), /Unrecognized/);
    const omitted = selected.command.replace("lib/llvm-fixture-00.lib ", "");
    assert.throws(() => validateArchiveGraph(omitted, selected.build, support, selected.archives, rows), /Unrecognized/);
  });

  test("accepts only the original completed Windows build-only producer receipt shape", _t => {
    const build = path.resolve(os.tmpdir(), "jit-build"), valid = {version: 1, kind: "original-source-jit-sdk", target: "windows", compiled: true,
      buildOnly: true, executed: false, publicationQualified: false, plan: {target: "windows", build, compile: []}, libraries: [], supportLibraries: [],
      inputs: {[path.join(build, "sources.json")]: "a".repeat(64), [path.join(os.tmpdir(), "absent", "compiler-header.h")]: "b".repeat(64)},
      recipe: {files: {}, graph: {"prior-checkout/recipe.js": "c".repeat(64)}}, proof: {}, sourcesFile: path.join(build, "sources.json"), resources: path.join(build, "resources")};
    assert.doesNotThrow(() => validateProducerReceipt(valid, build));
    for (const changed of [{target: "linux"}, {buildOnly: false}, {compiled: false}, {executed: true}, {publicationQualified: true}, {plan: {target: "linux", build, compile: []}}]) {
      assert.throws(() => validateProducerReceipt({...valid, ...changed}, build), /incomplete, wrong-target or not build-only/);
    }
  });

  test("pins the producer receipt, validates recorded recipe shape and explicit LLVM tools", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-jit-inputs-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const receipt = path.join(root, "result.json"); fs.writeFileSync(receipt, "{\"producer\":true}\n");
    const pinned = readPinnedProducerResult(receipt, fileHash(receipt)); assert.equal(pinned.result.producer, true);
    assert.throws(() => readPinnedProducerResult(receipt, "0".repeat(64)), /does not match/);
    const toolFiles = {llvmAr: path.join(root, "llvmAr"), llvmNm: path.join(root, "llvmNm"), llvmObjcopy: path.join(root, "llvmObjcopy")};
    for (const [name, file] of Object.entries(toolFiles)) {fs.writeFileSync(file, name);}
    const tools = validateToolsManifest({version: 1, tools: toolFiles}), hashes = hashTools(tools);
    assert.deepEqual(tools, toolFiles); fs.writeFileSync(toolFiles.llvmNm, "changed"); assert.notDeepEqual(hashTools(tools), hashes);
    assert.throws(() => validateToolsManifest({version: 1, tools: {...toolFiles, extra: toolFiles.llvmAr}}), /exactly llvmAr/);
    const recipe = /** @type {{files:Record<string,string>,graph:Record<string,string>}} */ (
      {files: {"previous-checkout/header.h": "a".repeat(64)}, graph: {"scripts/native-coff-archive.js": "b".repeat(64)}});
    assert.deepEqual(validateRecipeRecord(recipe), recipe);
    const changed = structuredClone(recipe); changed.graph["../outside.cpp"] = "0".repeat(64);
    assert.throws(() => validateRecipeRecord(changed), /incomplete or unsafe/);
  });

  test("rejects changed phase ancestry and archive/proof bytes", t => {
    const selected = graphFixture(t), support = /** @type {string[]} */ ([]), inputName = path.join(selected.root, "source.h"), inputHashes = {[inputName]: "a".repeat(64)};
    const names = ["configure", "support", "targets", "frontend", "link"], stageHashes = Object.fromEntries(names.map((name, index) => ["stage-" + name + ".json", String(index + 1).repeat(64)]));
    const libraries = selected.archives, stages = names.map((name, index) => ({version: 1, buildOnly: true, stage: name, beforeSha256: "f".repeat(64),
      ...(index ? {previousSha256: stageHashes["stage-" + names[index - 1] + ".json"]} : {}), configuredSha256: "c".repeat(64), commandsSha256: "d".repeat(64),
      libraries, supportLibraries: support, inputs: inputHashes}));
    assert.doesNotThrow(() => validateStageAncestry(stages, stageHashes, "f".repeat(64), "c".repeat(64), "d".repeat(64), libraries, support, inputHashes));
    assert.throws(() => validateStageAncestry(stages.map((row, index) => index === 2 ? {...row, previousSha256: "0".repeat(64)} : row),
      stageHashes, "f".repeat(64), "c".repeat(64), "d".repeat(64), libraries, support, inputHashes), /ancestry changed/);
    const result = {libraries: selected.archives.map(file => ({path: file, sha256: fileHash(file)})), proof: {path: selected.proof, sha256: fileHash(selected.proof)}};
    assert.doesNotThrow(() => verifyRecordedArtifacts(result, selected.archives, selected.build));
    const firstArchive = selected.archives[0]; assert.ok(firstArchive);
    fs.writeFileSync(firstArchive, "!<arch>\nchanged");
    assert.throws(() => verifyRecordedArtifacts(result, selected.archives, selected.build), /archive membership or bytes/);
    fs.writeFileSync(firstArchive, "!<arch>\n"); fs.writeFileSync(selected.proof, "changed proof");
    assert.throws(() => verifyRecordedArtifacts(result, selected.archives, selected.build), /proof executable changed/);
  });

  test("rejects overlapping, existing, and redirected output paths without mutation", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-jit-output-")), sdk = path.join(root, "sdk"), toolsFile = path.join(root, "tools.json");
    fs.mkdirSync(sdk); fs.writeFileSync(toolsFile, "{}\n");
    const selected = {sdkRoot: sdk, protectedPaths: [sdk], tools: {llvmAr: path.join(root, "llvm-ar")}}; fs.writeFileSync(selected.tools.llvmAr, "tool");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    assert.doesNotThrow(() => validateFreshOutput(selected, toolsFile, path.join(root, "fresh")));
    assert.throws(() => validateFreshOutput(selected, toolsFile, path.join(sdk, "isolated")), /overlaps/);
    const occupied = path.join(root, "occupied"); fs.mkdirSync(occupied); fs.writeFileSync(path.join(occupied, "keep"), "preserve");
    assert.throws(() => validateFreshOutput(selected, toolsFile, occupied), /fresh absolute output/);
    assert.equal(fs.readFileSync(path.join(occupied, "keep"), "utf8"), "preserve");
    if (process.platform !== "win32") {
      const alias = path.join(root, "alias"); fs.symlinkSync(root, alias, "dir");
      assert.throws(() => validateFreshOutput(selected, toolsFile, path.join(alias, "new")), /canonical bounded regular directory/);
    }
  });

  /** @param {import("node:test").TestContext} t */
  function completedSdkFixture(t) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-jit-consumer-")), sdkRoot = path.join(root, "sdk"), build = path.join(sdkRoot, "build");
    fs.mkdirSync(path.join(build, "lib"), {recursive: true}); fs.mkdirSync(path.join(build, "bin"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const names = ["sycl-jit", ...Array.from({length: 49}, (_, index) => `llvm-fixture-${String(index).padStart(2, "0")}`)];
    const archives = names.map(name => path.join(build, "lib", name + ".lib"));
    for (const archive of archives) {fs.writeFileSync(archive, "!<arch>\n");}
    const proof = path.join(build, "bin", "native-dpcpp-jit.exe"); fs.writeFileSync(proof, "linked build-only proof bytes\n");
    const sourceRoot = path.join(root, "absent-producer-source"), sourcesFile = path.join(sourceRoot, "sources.json"), resources = path.join(root, "absent-jit-resources");
    const missingHeader = path.join(root, "absent-compiler-sdk", "include", "clang.h"), missingCompiler = path.join(root, "absent-compiler-sdk", "bin", "clang.exe");
    const inputs = {[sourcesFile]: "1".repeat(64), [missingHeader]: "2".repeat(64), [missingCompiler]: "3".repeat(64)};
    const recipe = {files: {"prior-source-tree/include/old.h": "4".repeat(64)}, graph: {"prior-build-recipe/old-producer.js": "5".repeat(64)}};
    const plan = {version: 1, target: "windows", build, compile: ["cmake", "--build", build, "--target", "native-dpcpp-jit"]};
    const supportLibraries = /** @type {string[]} */ ([]), command = `lld-link /out:bin/native-dpcpp-jit.exe ${names.map(name => `lib/${name}.lib`).join(" ")}\n`;
    fs.writeFileSync(path.join(sdkRoot, "configured-tools.json"), "{\"configured\":true}\n");
    fs.writeFileSync(path.join(sdkRoot, "commands.log"), command);
    const configuredSha256 = fileHash(path.join(sdkRoot, "configured-tools.json")), commandsSha256 = fileHash(path.join(sdkRoot, "commands.log"));
    const before = {version: 1, config: {target: "windows", sources: sourcesFile, resources}, plan, recipe,
      inputs: {[sourcesFile]: inputs[sourcesFile]}, supportLibraries};
    const beforePath = path.join(sdkRoot, "before.json"); fs.writeFileSync(beforePath, JSON.stringify(before, null, 2) + "\n");
    const beforeSha256 = fileHash(beforePath), libraries = archives.slice(), stageNames = ["configure", ...require("../scripts/native-jit-stages").jitBuildStages];
    let previousSha256;
    for (const [index, stage] of stageNames.entries()) {
      const receipt = {version: 1, stage, beforeSha256, ...(index ? {previousSha256} : {}), configuredSha256, commandsSha256,
        libraries, supportLibraries, inputs, outputs: {}, buildOnly: true};
      const stagePath = path.join(sdkRoot, stage === "configure" ? "stage-configure.json" : "stage-" + stage + ".json");
      fs.writeFileSync(stagePath, JSON.stringify(receipt, null, 2) + "\n"); previousSha256 = fileHash(stagePath);
    }
    const result = {version: 1, kind: "original-source-jit-sdk", target: "windows", sourcesFile, resources, plan, supportLibraries, inputs, recipe,
      libraries: archives.map(file => ({path: file, sha256: fileHash(file)})), proof: {path: proof, sha256: fileHash(proof)},
      compiled: true, buildOnly: true, executed: false, publicationQualified: false};
    const resultPath = path.join(sdkRoot, "result.json"); fs.writeFileSync(resultPath, JSON.stringify(result, null, 2) + "\n");
    const tools = Object.fromEntries(["llvmAr", "llvmNm", "llvmObjcopy"].map(name => {
      const file = path.join(root, "tools", name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "fixture-only tool identity\n"); return [name, file];
    }));
    const toolsFile = path.join(root, "llvm-tools.json"); fs.writeFileSync(toolsFile, JSON.stringify({version: 1, tools}, null, 2) + "\n");
    return {root, sdkRoot, build, toolsFile, resultPath, result, resultSha256: fileHash(resultPath), archives, proof, sourcesFile, resources,
      missingHeader, missingCompiler};
  }

  test("admits a pinned complete producer handoff without reopening absent source or SDK inputs", t => {
    const fixture = completedSdkFixture(t), selected = inspectWindowsJitSdk(fixture.sdkRoot, fixture.resultSha256, fixture.toolsFile);
    assert.equal(selected.archives.length, 50); assert.equal(selected.resultSha256, fixture.resultSha256);
    assert.deepEqual(selected.result.recipe, fixture.result.recipe);
    assert.equal(fs.existsSync(fixture.sourcesFile), false); assert.equal(fs.existsSync(fixture.resources), false);
    assert.equal(fs.existsSync(fixture.missingHeader), false); assert.equal(fs.existsSync(fixture.missingCompiler), false);
    assert.equal(selected.inputHashes[fixture.missingHeader], "2".repeat(64));
    assert.doesNotThrow(() => validateFreshOutput(selected, fixture.toolsFile, path.join(fixture.root, "isolated")));
    assert.throws(() => validateFreshOutput(selected, fixture.toolsFile, path.join(fixture.sdkRoot, "isolated")), /overlaps/);
  });

  test("shared handoff preserves the inspected producer, tools and transformed archive identities", t => {
    const f = completedSdkFixture(t), prior = inspectWindowsJitSdk(f.sdkRoot, f.resultSha256, f.toolsFile),
      output = path.join(f.root, "isolated"); fs.mkdirSync(output);
    const resultFile = path.join(output, "result.json"); fs.writeFileSync(resultFile, "fixture isolation\n");
    const isolation = /** @type {ReturnType<typeof isolateCoffCompilerArchives>} */ (/** @type {unknown} */ ({
        sourceHashes: {"source.js": "a".repeat(64)}, mapSha256: "b".repeat(64),
        transformed: [{input: prior.archives[0], inputSha256: "c".repeat(64), output: path.join(output, "archive.lib"), sha256: "d".repeat(64)}],
      })), handoff = windowsJitIsolationHandoff(prior, isolation, f.toolsFile, output);
    assert.equal(handoff.kind, "windows-jit-archive-isolation-handoff");
    assert.deepEqual(handoff.producer.result, {path: f.resultPath, sha256: f.resultSha256});
    assert.deepEqual(handoff.producer.inputHashes, prior.inputHashes);
    assert.deepEqual(handoff.producer.sourceRecipe, prior.result.recipe);
    assert.deepEqual(handoff.tools.manifest, {path: f.toolsFile, sha256: fileHash(f.toolsFile)});
    assert.equal(handoff.isolation.result.sha256, fileHash(resultFile));
    assert.deepEqual(handoff.isolation.archives, [{input: prior.archives[0], inputSha256: "c".repeat(64),
      output: path.join(output, "archive.lib"), outputSha256: "d".repeat(64)}]);
    assert.equal(handoff.adapter.sha256, fileHash(require.resolve("../scripts/isolate-native-windows-jit")));
    assert.equal(handoff.publicationQualified, false);
  });

  test("rejects receipt, source-recipe, archive, proof, and phase tampering", t => {
    const fixture = completedSdkFixture(t), originalResult = fs.readFileSync(fixture.resultPath);
    assert.throws(() => inspectWindowsJitSdk(fixture.sdkRoot, "0".repeat(64), fixture.toolsFile), /does not match/);
    const changedRecipe = {...fixture.result, recipe: {files: {}, graph: {"changed-recipe.js": "9".repeat(64)}}};
    fs.writeFileSync(fixture.resultPath, JSON.stringify(changedRecipe) + "\n");
    assert.throws(() => inspectWindowsJitSdk(fixture.sdkRoot, fixture.resultSha256, fixture.toolsFile), /does not match/);
    fs.writeFileSync(fixture.resultPath, originalResult);

    const archive = fixture.archives[0] || "", originalArchive = fs.readFileSync(archive); fs.appendFileSync(archive, "changed");
    assert.throws(() => inspectWindowsJitSdk(fixture.sdkRoot, fixture.resultSha256, fixture.toolsFile), /archive membership or bytes/);
    fs.writeFileSync(archive, originalArchive);
    fs.writeFileSync(fixture.proof, "changed proof\n");
    assert.throws(() => inspectWindowsJitSdk(fixture.sdkRoot, fixture.resultSha256, fixture.toolsFile), /proof executable changed/);
    fs.writeFileSync(fixture.proof, "linked build-only proof bytes\n");
    const stagePath = path.join(fixture.sdkRoot, "stage-link.json"), stage = JSON.parse(fs.readFileSync(stagePath, "utf8"));
    stage.previousSha256 = "0".repeat(64); fs.writeFileSync(stagePath, JSON.stringify(stage) + "\n");
    assert.throws(() => inspectWindowsJitSdk(fixture.sdkRoot, fixture.resultSha256, fixture.toolsFile), /ancestry changed/);
  });
});
