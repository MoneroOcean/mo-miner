"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const esbuild = require("esbuild");
const nativeAliases = require("../scripts/native-bundle");
const shim = require("../runtime/path-internals");
const {tempDir} = require("./common/temp-dir");

test.describe("Native path source compatibility", () => {
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-path-"));
  const aliases = nativeAliases(output);
  /** @type {typeof path} */
  let compiled;
  test.before(() => {
    const bundled = path.join(output, "path.cjs");
    esbuild.buildSync({entryPoints: [aliases["node:path"]], outfile: bundled, bundle: true,
      platform: "neutral", format: "cjs", alias: aliases});
    compiled = require(bundled);
  });
  test.after(() => {fs.rmSync(output, {recursive: true, force: true});});
  test("generated source retains its MIT license and exact build-Node provenance", () => {
    assert.match(fs.readFileSync(aliases["node:path"], "utf8"), /Copyright Joyent/);
    const provenance = JSON.parse(fs.readFileSync(path.join(output, "node-path-source.json"), "utf8"));
    assert.equal(provenance.node, process.version); assert.match(provenance.sha256, /^[a-f0-9]{64}$/);
  });
  test("POSIX normalization handles parent segments and trailing separators", () => {
    for (const value of ["", "./", "a/..", "a/../", "///", "/a/../../b"]) {
      assert.equal(compiled.posix.normalize(value), path.posix.normalize(value));
    }
  });
  test("Windows normalization retains drive, UNC and namespace rules", () => {
    for (const value of ["C:", "C:\\", "C:relative", "\\\\host\\share\\a\\..", "\\\\?\\C:\\a"]) {
      assert.equal(compiled.win32.normalize(value), path.win32.normalize(value));
    }
  });
  test("resolve and relative use actual current-directory semantics", () => {
    for (const api of ["posix", "win32"]) {
      const kind = /** @type {"posix" | "win32"} */ (api);
      assert.equal(compiled[kind].resolve("a", "..", "é"), path[kind].resolve("a", "..", "é"));
      assert.equal(compiled[kind].relative("a", "b"), path[kind].relative("a", "b"));
    }
  });
  test("format, parse, basename and extensions match the maintained Node implementation", () => {
    for (const kind of /** @type {const} */ (["posix", "win32"])) {
      for (const value of ["é/💎.txt", "C:\\dir\\file.ext", ".env", "a."]) {
        assert.deepEqual(compiled[kind].parse(value), path[kind].parse(value));
        assert.equal(compiled[kind].basename(value, ".ext"), path[kind].basename(value, ".ext"));
        assert.equal(compiled[kind].format({name: "file", ext: "txt"}), path[kind].format({name: "file", ext: "txt"}));
      }
    }
  });
  test("invalid string arguments retain TypeError and stable error code", () => {
    assert.throws(() => compiled.join(/** @type {string} */ (/** @type {unknown} */ (4))),
      {name: "TypeError", code: "ERR_INVALID_ARG_TYPE"});
  });
  test("invalid format objects cannot be coerced into paths", () => {
    const target = compiled;
    for (const value of [null, [], 1]) {
      assert.throws(() => target.format(/** @type {import("node:path").FormatInputPathObject} */ (value)),
        {code: "ERR_INVALID_ARG_TYPE"});
    }
  });
  test("unimplemented glob semantics fail explicitly rather than returning a guessed match", () => {
    if ("matchesGlob" in compiled) {assert.throws(() => compiled.matchesGlob("a", "*"), /not implemented/);}
  });
  test("internal lazy modules initialize exactly once", () => {
    let calls = 0; const lazy = shim.getLazy(() => ++calls); assert.equal(lazy(), 1); assert.equal(lazy(), 1);
  });
  test("bound builtin methods do not depend on a method receiver", () => {
    assert.equal(shim.primordials.ArrayPrototypeJoin(["a", "b"], "/"), "a/b");
    assert.equal(shim.primordials.StringPrototypeSlice("abcd", 1, 3), "bc");
  });
  test("Node 24's reviewed lazy-glob import binds without adding filesystem glob behavior", t => {
    const root = tempDir(t, "mom-native-path24-");
    const internals = /** @type {{binding: (name: string) => object}} */ (/** @type {unknown} */ (process)), original = internals.binding;
    const source = "const {getLazy}=require('internal/util'); const glob=getLazy(()=>require('internal/fs/glob').matchGlobPattern); module.exports={matchesGlob:(s,p)=>glob()(s,p)};";
    t.mock.method(internals, "binding", (/** @type {string} */ name) => name === "natives" ? {path: source} : original(name));
    const selected = nativeAliases(root), bundled = path.join(root, "path.cjs");
    assert.equal(Reflect.get(selected, "internal/fs/glob"), Reflect.get(selected, "internal/util"));
    esbuild.buildSync({entryPoints: [selected["node:path"]], outfile: bundled, bundle: true, platform: "neutral", format: "cjs", alias: selected});
    assert.throws(() => require(bundled).matchesGlob("a", "*"), /not implemented/);
  });
  test("reviewed lazy aliases do not permit arbitrary Node internals", t => {
    const root = tempDir(t, "mom-native-path-unreviewed-");
    const internals = /** @type {{binding: (name: string) => object}} */ (/** @type {unknown} */ (process)), original = internals.binding;
    t.mock.method(internals, "binding", (/** @type {string} */ name) => name === "natives" ? {path: "require('internal/fs/unreviewed')"} : original(name));
    assert.throws(() => nativeAliases(root), /unreviewed internal dependency/); assert.deepEqual(fs.readdirSync(root), []);
  });
});
