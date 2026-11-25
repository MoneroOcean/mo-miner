"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const resources = require("../scripts/native-resources");
const {tempDir} = require("./common/temp-dir");

/** Create only this test's disposable build inputs. @param {import('node:test').TestContext} t */
function fixture(t) {
  const directory = tempDir(t, "mom-resource-");
  return {input: path.join(directory, "input.bc"), output: path.join(directory, "resources.inc")};
}
/** @param {string[]} args */
function generate(args) {
  return spawnSync(process.execPath, [path.join(__dirname, "../scripts/native-resources.js"), ...args], {encoding: "utf8"});
}

test.describe("Native immutable compiler resources", () => {
  test("preserves binary NUL and high bytes without encoding conversion", () => {
    const output = resources([{name: "input.bc", data: Buffer.from([0, 127, 128, 255])}]);
    assert.match(output, /0x00,0x7f,0x80,0xff/);
  });
  test("preserves exact Buffer view ranges", () => {
    const data = Buffer.from([1, 2, 3]).subarray(1, 2), output = resources([{name: "x", data}]);
    assert.match(output, /0x02,/); assert.doesNotMatch(output, /0x01|0x03/);
  });
  test("emits immutable static data with explicit byte lengths", () => {
    const output = resources([{name: "input.bc", data: Buffer.from("data")}]);
    assert.match(output, /static constexpr unsigned char resource_0\[\]/); assert.match(output, /sizeof\(resource_0\)/);
  });
  test("unknown names return no resource rather than opening a path", () => {
    const output = resources([{name: "x", data: Buffer.from("data")}]);
    assert.match(output, /return \{\};/); assert.doesNotMatch(output, /fopen|ofstream|dlopen|load_library/);
  });
  test("resource names cannot inject generated code", () => {
    assert.throws(() => resources([{name: 'x";evil()', data: Buffer.from("x")}]), /name/);
  });
  test("filesystem paths cannot be used as resource keys", () => {
    assert.throws(() => resources([{name: "../x", data: Buffer.from("x")}]), /name/);
    assert.throws(() => resources([{name: "/x", data: Buffer.from("x")}]), /name/);
  });
  test("resource names cannot rely on implicit string coercion", () => {
    assert.throws(() => resources([{name: /** @type {string} */ (/** @type {unknown} */ (123)), data: Buffer.from("x")}]), /name/);
  });
  test("duplicate symbolic resources are rejected", () => {
    assert.throws(() => resources([{name: "x", data: Buffer.from("a")}, {name: "x", data: Buffer.from("b")}]), /duplicate/);
  });
  test("empty bitcode cannot masquerade as a present resource", () => {
    assert.throws(() => resources([{name: "x", data: Buffer.alloc(0)}]), /bounds/);
  });
  test("arbitrary objects cannot supply forged byte buffers", () => {
    assert.throws(() => resources([{name: "x", data: /** @type {Buffer} */ (/** @type {unknown} */ ({length: 1}))}]), /bounds/);
  });
  test("aggregate size is bounded before generating any C++", () => {
    assert.throws(() => resources([{name: "first", data: Buffer.allocUnsafe(32 << 20)},
      {name: "second", data: Buffer.allocUnsafe((32 << 20) + 1)}]), /bounds/);
  });
  test("input ordering is reproducible and independent resources have unique symbols", () => {
    const input = [{name: "first.bc", data: Buffer.from("one")}, {name: "second.bc", data: Buffer.from("two")}];
    assert.equal(resources(input), resources(input)); assert.match(resources(input), /resource_1\[\]/);
  });
  test("build CLI uses the same binary-preserving generator", (t) => {
    const {input, output} = fixture(t), data = Buffer.from([0, 128, 255]);
    fs.writeFileSync(input, data);
    const result = generate([output, "input.bc", input]);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(fs.readFileSync(output, "utf8"), resources([{name: "input.bc", data}]));
  });
  test("incomplete CLI arguments do not create output", (t) => {
    const {output} = fixture(t), result = generate([output, "input.bc"]);
    assert.equal(result.status, 1); assert.match(result.stderr, /Usage/); assert.equal(fs.existsSync(output), false);
  });
  test("a missing build input cannot replace an existing resource header", (t) => {
    const {input, output} = fixture(t);
    fs.writeFileSync(output, "preserved");
    const result = generate([output, "input.bc", input]);
    assert.equal(result.status, 1); assert.equal(fs.readFileSync(output, "utf8"), "preserved");
  });
  test("changed build inputs regenerate resources without stale bytes", (t) => {
    const {input, output} = fixture(t);
    fs.writeFileSync(input, Buffer.from([1]));
    assert.equal(generate([output, "input.bc", input]).status, 0);
    fs.writeFileSync(input, Buffer.from([2]));
    assert.equal(generate([output, "input.bc", input]).status, 0);
    const text = fs.readFileSync(output, "utf8");
    assert.match(text, /0x02/); assert.doesNotMatch(text, /0x01/);
  });
});

test.describe("Native COMGR memory resources and scratch ownership", () => {
  const patch = fs.readFileSync(path.join(__dirname, "../runtime/rocm-comgr-resources.patch"), "utf8");
  const added = patch.split("\n").filter((line) => /^\+(?!\+)/.test(line)).map((line) => line.slice(1)).join("\n");
  test("the resource policy is opt-in and leaves shared Engineering defaults alone", () => {
    assert.match(added, /option\(COMGR_EMBEDDED_RESOURCES [^\n]* OFF\)/);
    assert.match(added, /\$<\$<BOOL:\$\{COMGR_EMBEDDED_RESOURCES\}>:COMGR_EMBEDDED_RESOURCES>/);
  });
  test("native precompiled headers use one explicit reviewed Clang resource path", () => {
    assert.match(added, /if\(COMGR_EMBEDDED_RESOURCES\)[\s\S]*NOT IS_ABSOLUTE "\$\{COMGR_OPENCL_C_HEADER\}"/);
    assert.match(added, /set\(OPENCL_C_H "\$\{COMGR_OPENCL_C_HEADER\}"\)\nelseif\(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR\)/);
    assert.doesNotMatch(added, /GLOB_RECURSE/);
  });
  test("provided sources and headers are owned by the existing compiler VFS", () => {
    assert.match(added, /MemoryBuffer::getMemBufferCopy\(/);
    assert.match(added, /stageInput\(Input, IncludeFilePath, UseVFS \? InMemoryFS.get\(\) : nullptr\)/);
    assert.match(added, /stageInput\(Input, InputFilePath, UseVFS \? InMemoryFS.get\(\) : nullptr\)/);
  });
  test("precompiled headers use the same VFS staging as source headers", () => {
    assert.match(added, /stageInput\(Input, PrecompiledHeaderPath,\s*UseVFS \? InMemoryFS.get\(\) : nullptr\)/);
    assert.doesNotMatch(added, /outputToFile\(Input, PrecompiledHeaderPath\)/);
  });
  test("forced embedded includes cannot be shadowed by a working-directory file", () => {
    assert.match(added, /StringRef\(Argv\[I \+ 1\]\) == Header->Name/);
    assert.match(added, /Argv\[I \+ 1\] = Saver.save\(StringRef\(FullPath\)\).data\(\)/);
    assert.match(added, /Name.contains\('\\\\'\)/);
    assert.match(added, /Part == "\.\."/);
  });
  test("native compilation uses embedded device libraries and reports handled stream errors", () => {
    assert.match(added, /const bool UseInstalledLibraries = false/);
    assert.match(added, /OS.clear_error\(\)/);
    assert.match(added, /Output->setName\("a.hsaco"\)/);
  });
  const fixture = {directory: "", executable: ""};
  test.before(() => {
    if (process.platform === "win32") {return;}
    fixture.directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-comgr-scratch-"));
    fixture.executable = path.join(fixture.directory, "scratch");
    const start = added.indexOf("  std::error_code EC;"), end = added.indexOf("#else", start);
    assert.ok(start > 0 && end > start);
    fs.writeFileSync(path.join(fixture.directory, "comgr-scratch.inc"), added.slice(start, end));
    const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      "-I", fixture.directory, path.join(__dirname, "native/comgr-scratch.cpp"), "-o", fixture.executable], {encoding: "utf8", timeout: 30000});
    assert.equal(result.status, 0, result.stderr);
  });
  test.after(() => {if (fixture.directory) {fs.rmSync(fixture.directory, {recursive: true, force: true});}});
  for (const [collisions, denied] of [[0, 0], [1, 0], [15, 0], [16, 0], [0, 1], [1, 1]]) {
    test(`acquisition collisions=${collisions} denied=${denied} never transfers foreign-directory ownership`, {
      skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
    }, () => {
      const result = spawnSync(fixture.executable, [String(collisions), String(denied)], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, result.stderr);
    });
  }
});
