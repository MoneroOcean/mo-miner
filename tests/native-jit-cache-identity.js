"use strict";
const {describe, test} = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {nativeJitCacheTag, nativeJitCacheHeader, admittedJitCacheTag} = require("../scripts/native-jit-cache-tag");

describe("Native JIT closure identity", () => {
  const a = "a".repeat(64), b = "b".repeat(64), recipes = {"runtime/main.cpp": a},
    first = {role: "sdk/0", sha256: a}, second = {role: "gpu/0", sha256: b}, artifacts = [first, second];
  test("same admitted closure is deterministic and emits exactly 32 bytes", () => {
    const tag = nativeJitCacheTag("linux-x64", artifacts, recipes);
    assert.equal(tag, nativeJitCacheTag("linux-x64", artifacts, recipes));
    assert.equal((nativeJitCacheHeader(tag).match(/0x[a-f0-9]{2}/g) || []).length, 32);
  });
  test("SDK contents, link order, target and runtime recipe each affect identity", () => {
    const base = nativeJitCacheTag("linux-x64", artifacts, recipes);
    for (const changed of [nativeJitCacheTag("windows-x64", artifacts, recipes),
      nativeJitCacheTag("linux-x64", [...artifacts].reverse(), recipes),
      nativeJitCacheTag("linux-x64", [{role: "sdk/0", sha256: b}, second], recipes),
      nativeJitCacheTag("linux-x64", artifacts, {"runtime/main.cpp": b})]) {assert.notEqual(changed, base);}
  });
  test("recipe insertion order does not affect identity", () => {
    assert.equal(nativeJitCacheTag("linux-x64", artifacts, {"runtime/main.cpp": a, "scripts/build.js": b}),
      nativeJitCacheTag("linux-x64", artifacts, {"scripts/build.js": b, "runtime/main.cpp": a}));
  });
  test("accepts a runtime recipe tree larger than 128 files", () => {
    const grown = Object.fromEntries(Array.from({length: 200}, (_, index) => ["runtime/file-" + index + ".cpp", a]));
    assert.match(nativeJitCacheTag("linux-x64", artifacts, grown), /^[a-f0-9]{64}$/);
  });
  test("archive selection mode and interleaved linker directives affect identity", () => {
    const base = nativeJitCacheTag("linux-x64", [first], recipes, ["@sdk/0"]);
    assert.notEqual(base, nativeJitCacheTag("linux-x64", [first], recipes, ["--whole-archive", "@sdk/0", "--no-whole-archive"]));
    assert.notEqual(base, nativeJitCacheTag("linux-x64", [{role: "whole/0", sha256: a}], recipes, ["@sdk/0"]));
    assert.throws(() => nativeJitCacheTag("linux-x64", [first], recipes, ["bad\noption"]), /link plan/);
  });
  test("admitted file relocation preserves identity but changed bytes invalidate it", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-jit-tag-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const left = path.join(root, "left.lib"), right = path.join(root, "right.lib"), repo = path.resolve(__dirname, "..");
    fs.writeFileSync(left, "same admitted bytes"); fs.copyFileSync(left, right);
    const tag = admittedJitCacheTag(repo, "windows-x64", {sdk: [left]});
    assert.equal(tag, admittedJitCacheTag(repo, "windows-x64", {sdk: [right]}));
    fs.writeFileSync(right, "changed linked SDK");
    assert.notEqual(tag, admittedJitCacheTag(repo, "windows-x64", {sdk: [right]}));
  });
  test("missing inputs, duplicate roles, paths and malformed hashes fail closed", () => {
    assert.throws(() => nativeJitCacheTag("linux-x64", [], recipes), /closure/);
    assert.throws(() => nativeJitCacheTag("linux-x64", [first, first], recipes), /artifact/);
    assert.throws(() => nativeJitCacheTag("linux-x64", artifacts, {"/tmp/main.cpp": a}), /recipe/);
    assert.throws(() => nativeJitCacheHeader("invalid"), /tag/);
  });
});
