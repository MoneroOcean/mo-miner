"use strict";

const {linuxSysrootRecord, linuxSysrootReceiptSha256} = require("../scripts/native-linux-sysroot");
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {fileHash} = require("../scripts/build-source-inputs");
const {zstdPin, toolNames, zstdSdkPlan, validateArchive, sourceInventory, stagedFiles, zstdBuildRecipe,
  verifyZstdArchiveInput, verifyZstdTools, verifyZstdRecipe, verifyZstdOutputs, nativeZstdNoticeInputs, nativeZstdInputs, buildNativeZstd} = require("../scripts/build-native-zstd");

const linuxTools = {ar: "/tools/ar", cc: "/tools/icx", make: "/tools/make", mkdir: "/tools/mkdir", tar: "/tools/tar"};
const base = "zstd-" + zstdPin.version;

/** Temporary fixtures never invoke a compiler, archive tool or zstd executable. @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-zstd-")); t.after(() => fs.rmSync(root, {recursive: true, force: true})); return root;
}

function archiveFixture() {
  const members = [
    [base + "/", "d"], [base + "/LICENSE", "-"], [base + "/COPYING", "-"], [base + "/lib/", "d"],
    [base + "/lib/Makefile", "-"], [base + "/lib/libzstd.mk", "-"], [base + "/lib/zstd.h", "-"],
    [base + "/lib/zstd_errors.h", "-"], [base + "/lib/zdict.h", "-"],
    [base + "/tests/cli-tests/bin/unzstd", "l"], [base + "/tests/cli-tests/bin/zstdcat", "l"],
  ];
  return {listing: members.map(([name]) => name).join("\n") + "\n",
    verbose: members.map(([name, kind]) => (kind === "d" ? "drwxr-xr-x" : kind === "l" ? "lrwxrwxrwx" : "-rw-r--r--") +
      " user/group 1 2025-01-01 00:00 " + name + (kind === "l" ? " -> zstd" : "")).join("\n") + "\n"};
}

/** @param {string} root */
function sourceTree(root) {
  const source = path.join(root, "source"), library = path.join(source, "lib");
  fs.mkdirSync(library, {recursive: true});
  fs.writeFileSync(path.join(source, "LICENSE"), "BSD-style notice fixture\n");
  fs.writeFileSync(path.join(source, "COPYING"), "GPLv2 notice fixture\n");
  for (const name of ["Makefile", "libzstd.mk", "zstd.h", "zstd_errors.h", "zdict.h"]) {fs.writeFileSync(path.join(library, name), "pinned source fixture: " + name + "\n");}
  return source;
}

/** Build a receipt fixture from inert ar bytes; the reader never executes a program. @param {import("node:test").TestContext} t */
function sdkFixture(t) {
  const root = fixture(t), source = sourceTree(root), library = path.join(root, "lib", "libzstd.a"), buildArchive = path.join(root, "build", "static", "libzstd.a");
  fs.mkdirSync(path.dirname(library), {recursive: true}); fs.mkdirSync(path.dirname(buildArchive), {recursive: true});
  fs.writeFileSync(library, "!<arch>\n"); fs.writeFileSync(buildArchive, "!<arch>\n");
  const inputRoot = path.join(root, "original-inputs"); fs.mkdirSync(inputRoot);
  const archivePath = path.join(inputRoot, "zstd-1.5.7.tar.gz"); fs.writeFileSync(archivePath, "tiny archive fixture, not the pinned release\n");
  const tools = Object.fromEntries(toolNames.map(name => {
    const file = path.join(inputRoot, name); fs.writeFileSync(file, "tool fixture: " + name + "\n"); fs.chmodSync(file, 0o755);
    return [name, {path: file, sha256: fileHash(file)}];
  }));
  const sourceState = sourceInventory(source), toolPaths = Object.fromEntries(Object.entries(tools).map(([name, row]) => [name, row.path]));
  const plan = zstdSdkPlan(root, archivePath, toolPaths, 2, "linux"), artifacts = {"lib/libzstd.a": fileHash(library)};
  const record = {version: 1, zstdVersion: zstdPin.version, platform: "linux", target: "x86_64-linux-gnu",
    archive: {path: archivePath, sha256: zstdPin.sha256, bytes: fs.statSync(archivePath).size, members: 11},
    source: {files: sourceState.count, bytes: sourceState.bytes, sha256: sourceState.sha256,
      notices: {LICENSE: {path: "source/LICENSE", sha256: fileHash(path.join(source, "LICENSE"))}, COPYING: {path: "source/COPYING", sha256: fileHash(path.join(source, "COPYING"))}}},
    config: {sha256: "b".repeat(64)}, tools, compiler: {version: "Intel(R) oneAPI DPC++/C++ Compiler 2026.2.0"}, recipe: zstdBuildRecipe(),
    plan: {jobs: plan.jobs, buildCommand: plan.buildCommand, target: plan.buildArchive, staticMode: plan.staticMode},
    buildOutput: {path: "build/static/libzstd.a", sha256: fileHash(buildArchive), bytes: fs.statSync(buildArchive).size},
    install: {path: "lib/libzstd.a", sha256: fileHash(library), bytes: fs.statSync(library).size}, artifacts,
    sysroot: linuxSysrootRecord({receiptSha256: linuxSysrootReceiptSha256}), buildOnly: true, executed: false, publicationQualified: false,
    scope: "Build-only static multithreaded libzstd 1.5.7 archive; no CLI, tests, shared library or release qualification."};
  fs.writeFileSync(path.join(root, "result.json"), JSON.stringify(record));
  return {root, record, archivePath, save: () => fs.writeFileSync(path.join(root, "result.json"), JSON.stringify(record))};
}

test.describe("Pinned static zstd source production and admission", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("pins the installed package's source version to the official release and builds only the static library", () => {
    assert.deepEqual(zstdPin, {version: "1.5.7", sha256: "eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3"});
    assert.ok(Object.isFrozen(zstdPin));
    for (const jobs of [1, 2, 16]) {
      const plan = zstdSdkPlan("/new/zstd", "/source/zstd-1.5.7.tar.gz", linuxTools, jobs, "linux");
      assert.equal(plan.target, "x86_64-linux-gnu"); assert.equal(plan.staticMode, "ZSTD_MULTITHREAD");
      assert.equal(plan.buildCommand[0], linuxTools.make); assert.ok(plan.buildCommand.includes("CC=" + linuxTools.cc));
      assert.ok(plan.buildCommand.includes("AR=" + linuxTools.ar)); assert.ok(plan.buildCommand.includes("CPPFLAGS=-DZSTD_MULTITHREAD"));
      assert.deepEqual(plan.buildCommand.slice(-1), ["/new/zstd/build/static/libzstd.a"]); assert.equal(plan.jobs, jobs);
      assert.equal(plan.buildCommand.some(arg => /(?:install|test|cli|shared)/i.test(arg)), false);
      assert.deepEqual(plan.extractCommand.slice(-3), [base + "/lib", base + "/LICENSE", base + "/COPYING"]);
    }
  });

  test("plans and tool descriptors reject implicit lookup, malformed jobs and non-Linux targets", () => {
    for (const jobs of [0, 17, 1.5, NaN, Infinity]) {assert.throws(() => zstdSdkPlan("/new/zstd", "/src/zstd.tar.gz", linuxTools, jobs, "linux"), /jobs/);}
    for (const output of ["relative", "/new/bad path", "/new/$PATH", "/new/line\nbreak"]) {
      assert.throws(() => zstdSdkPlan(output, "/src/zstd.tar.gz", linuxTools, 2, "linux"), /absolute paths/);
    }
    assert.throws(() => zstdSdkPlan("/new/zstd", "/src/zstd.tar.gz", linuxTools, 2, "win32"), /Linux/);
    assert.throws(() => zstdSdkPlan("/new/zstd", "/src/zstd.tar.gz", {...linuxTools, cc: "icx"}, 2, "linux"), /explicit absolute/);
    const missing = {...linuxTools}; Reflect.deleteProperty(missing, "tar");
    assert.throws(() => zstdSdkPlan("/new/zstd", "/src/zstd.tar.gz", missing, 2, "linux"), /explicit absolute/);
  });

  test("archive preflight accepts only original regular files/directories and the two excluded CLI-test links", () => {
    const good = archiveFixture(); assert.equal(validateArchive(good.listing, good.verbose), 11);
    const huge = {...good, verbose: good.verbose.replace("user/group 1 2025-01-01 00:00 " + base + "/lib/Makefile",
      "user/group 16777217 2025-01-01 00:00 " + base + "/lib/Makefile")};
    assert.throws(() => validateArchive(huge.listing, huge.verbose), /extraction bounds/);
    const hugeExcluded = {...good, listing: good.listing + base + "/tests/large-fixture\n",
      verbose: good.verbose + "-rw-r--r-- user/group 67108865 2025-01-01 00:00 " + base + "/tests/large-fixture\n"};
    assert.throws(() => validateArchive(hugeExcluded.listing, hugeExcluded.verbose), /expansion bound/);
    const unsafe = /** @type {[string, string][]} */ ([
      [good.listing + base + "/../../escape\n", good.verbose + "-rw-r--r-- user/group 1 2025-01-01 00:00 escape\n"],
      [good.listing.replace(base + "/LICENSE\n", ""), good.verbose.replace(/-rw-r--r-- user\/group 1 2025-01-01 00:00 zstd-1\.5\.7\/LICENSE\n/, "")],
      [good.listing, good.verbose.replace(" -> zstd\n", " -> ../../outside\n")],
      [good.listing.replace("zstd-1.5.7/", "other/"), good.verbose],
      [good.listing, good.verbose.replace("lrwxrwxrwx", "hrw-r--r--")],
    ]);
    for (const [listing, verbose] of unsafe) {assert.throws(() => validateArchive(listing, verbose), /archive|member|link|sources/i);}
  });

  test("the build-only guard, archive pin and canonical input checks fail before output creation", t => {
    const root = fixture(t), output = path.join(root, "new-sdk");
    const tools = Object.fromEntries(toolNames.map(name => {
      const file = path.join(root, name); fs.writeFileSync(file, "#!/bin/sh\nexit 0\n"); fs.chmodSync(file, 0o755); return [name, file];
    }));
    const config = {version: 1, jobs: 2, tools};
    assert.throws(() => buildNativeZstd("/missing.tar.gz", output, config, {}), /MOM_AOT_BUILD_ONLY=1/);
    assert.equal(fs.existsSync(output), false);
    assert.throws(() => buildNativeZstd("relative.tar.gz", output, config, {MOM_AOT_BUILD_ONLY: "1"}), /absolute/);
    assert.equal(fs.existsSync(output), false);
    const bad = path.join(root, "wrong.tar.gz"); fs.writeFileSync(bad, "not the pinned release");
    assert.throws(() => buildNativeZstd(bad, output, config, {MOM_AOT_BUILD_ONLY: "1"}), /SHA256/);
    assert.equal(fs.existsSync(output), false);
  });

  test("output admission binds the build archive and staged archive to one exact identity", t => {
    const f = sdkFixture(t), result = verifyZstdOutputs(f.root, f.record.buildOutput, f.record.install, f.record.artifacts);
    assert.equal(Object.keys(result).length, 1); assert.deepEqual(stagedFiles(f.root), result);
    const changed = sdkFixture(t); fs.appendFileSync(path.join(changed.root, "build", "static", "libzstd.a"), "changed");
    assert.throws(() => verifyZstdOutputs(changed.root, changed.record.buildOutput, changed.record.install, changed.record.artifacts), /build\/install identity differs|build archive changed/);
    const extra = sdkFixture(t); fs.writeFileSync(path.join(extra.root, "lib", "libzstd.so"), "not admitted");
    assert.throws(() => stagedFiles(extra.root), /unexpected staged members/);
  });

  test("source inventory binds notices and detects missing, extra, and linked inputs", {skip: process.platform === "win32"}, t => {
    const clean = sdkFixture(t), initial = sourceInventory(path.join(clean.root, "source"));
    const sourceChanged = sdkFixture(t); fs.appendFileSync(path.join(sourceChanged.root, "source", "LICENSE"), "changed\n");
    assert.notEqual(sourceInventory(path.join(sourceChanged.root, "source")).sha256, initial.sha256);
    const sourceExtra = sdkFixture(t); fs.writeFileSync(path.join(sourceExtra.root, "source", "lib", "shadow.c"), "extra\n");
    assert.notEqual(sourceInventory(path.join(sourceExtra.root, "source")).sha256, initial.sha256);
    const sourceEmptyDir = sdkFixture(t); fs.mkdirSync(path.join(sourceEmptyDir.root, "source", "lib", "empty"));
    assert.notEqual(sourceInventory(path.join(sourceEmptyDir.root, "source")).sha256, initial.sha256);
    const linked = sdkFixture(t), target = path.join(linked.root, "outside"); fs.writeFileSync(target, "outside");
    fs.symlinkSync(target, path.join(linked.root, "source", "lib", "linked.c"));
    assert.throws(() => sourceInventory(path.join(linked.root, "source")), /bounded regular-file tree/);
  });

  test("projects the already-pinned source notices in stable LICENSE/COPYING order", t => {
    const f = sdkFixture(t), source = path.join(f.root, "source"), license = path.join(source, "LICENSE"), copying = path.join(source, "COPYING");
    const notices = nativeZstdNoticeInputs(source, f.record.source.notices);
    assert.deepEqual(notices, [
      {component: "zstd LICENSE", path: license, sha256: fileHash(license)},
      {component: "zstd COPYING", path: copying, sha256: fileHash(copying)},
    ]);
    fs.appendFileSync(license, "changed\n");
    assert.throws(() => nativeZstdNoticeInputs(source, f.record.source.notices), /source notice changed: LICENSE/);
  });

  test("read-only preflight reopens the original archive, exact tools and current recipe", t => {
    const f = sdkFixture(t);
    assert.throws(() => verifyZstdArchiveInput(f.record.archive), /Original pinned zstd archive changed/);
    assert.throws(() => nativeZstdInputs(f.root), /Original pinned zstd archive changed/);
    const compilerTool = f.record.tools["cc"]; assert.ok(compilerTool);
    const alteredTool = compilerTool.path; fs.appendFileSync(alteredTool, "changed\n");
    assert.throws(() => verifyZstdTools(f.record.tools), /tool changed: cc/);
    const staleTool = sdkFixture(t), staleCompiler = staleTool.record.tools["cc"]; assert.ok(staleCompiler);
    fs.appendFileSync(staleCompiler.path, "changed\n");
    assert.throws(() => nativeZstdInputs(staleTool.root), /tool changed: cc/);
    const recipe = zstdBuildRecipe(); verifyZstdRecipe(recipe);
    recipe.graph["scripts/build-native-zstd.js"] = "0".repeat(64);
    assert.throws(() => verifyZstdRecipe(recipe), /source inputs are stale or missing/);
    const mutated = sdkFixture(t); Reflect.set(mutated.record, "zstdVersion", "1.5.6"); mutated.save();
    assert.throws(() => nativeZstdInputs(mutated.root), /Invalid static zstd SDK receipt/);
  });
});
