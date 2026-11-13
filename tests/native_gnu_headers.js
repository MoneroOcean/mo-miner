"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {prepareLinuxGnuHeaders, verifyLinuxGnuHeaders, prepareWindowsGnuHeaders, verifyWindowsGnuHeaders, prepareGnuHeaders, virtualHeader} = require("../scripts/prepare-native-gnu-headers");

const packageNames = ["libstdc++-13-dev", "libc6-dev", "linux-libc-dev"];
const windowsPackageNames = ["mingw-w64-common", "mingw-w64-x86-64-dev", "g++-mingw-w64-x86-64-posix"];
/** @typedef {{name: string, version: string, architecture: string, files: string[]}} FixturePackage */

/** @param {import("node:test").TestContext} context */
function fixture(context) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-gnu-headers-"));
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const include = path.join(root, "include");
  fs.mkdirSync(include);
  const records = /** @type {FixturePackage[]} */ (packageNames.map(name =>
    ({name, version: "1.2.3", architecture: "amd64", files: []})));
  const output = path.join(root, "staged");
  return {root, include, output, records};
}

/** @param {import("node:test").TestContext} context */
function windowsFixture(context) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-gnu-win-headers-"));
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const sysroot = path.join(root, "sysroot");
  fs.mkdirSync(sysroot);
  const records = /** @type {FixturePackage[]} */ ([
    {name: windowsPackageNames[0] || "", version: "13.0.0-2ubuntu1", architecture: "all", files: []},
    {name: windowsPackageNames[1] || "", version: "13.0.0-2ubuntu1", architecture: "all", files: []},
    {name: windowsPackageNames[2] || "", version: "13.2.0-6ubuntu1+26.1", architecture: "amd64", files: []},
  ]);
  return {root, sysroot, output: path.join(root, "staged"), records};
}

/** @param {ReturnType<typeof windowsFixture>} selected @param {string} packageName @param {string} logical
 * @param {string} [contents] */
function addWindowsFile(selected, packageName, logical, contents = logical) {
  const record = selected.records.find(row => row.name === packageName);
  assert.ok(record);
  assert.ok(logical.startsWith("/"));
  const file = path.join(selected.sysroot, logical.slice(1));
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, contents);
  record.files.push(logical);
  return file;
}

/** @param {ReturnType<typeof windowsFixture>} selected @param {boolean} [includeSystem] */
function addWindowsRequiredRoots(selected, includeSystem = true) {
  const gxx = windowsPackageNames[2] || "";
  const common = windowsPackageNames[0] || "";
  addWindowsFile(selected, gxx, "/usr/lib/gcc/x86_64-w64-mingw32/13-posix/include/c++/algorithm");
  addWindowsFile(selected, gxx, "/usr/lib/gcc/x86_64-w64-mingw32/13-posix/include/c++/backward/backward_warning.h");
  addWindowsFile(selected, gxx, "/usr/lib/gcc/x86_64-w64-mingw32/13-posix/include/c++/x86_64-w64-mingw32/bits/c++config.h");
  if (includeSystem) {
    const real = addWindowsFile(selected, common, "/usr/share/mingw-w64/include/windows.h", "windows bytes");
    const alias = path.join(selected.sysroot, "usr/x86_64-w64-mingw32/include/windows.h");
    fs.mkdirSync(path.dirname(alias), {recursive: true});
    fs.symlinkSync("../../share/mingw-w64/include/windows.h", alias);
    assert.ok(fs.realpathSync(alias) === real);
    selected.records[1]?.files.push("/usr/x86_64-w64-mingw32/include/windows.h");
  }
}

/** @param {ReturnType<typeof fixture>} selected @param {string} packageName @param {string} relative
 * @param {string} [contents] */
function addFile(selected, packageName, relative, contents = relative) {
  const record = selected.records.find(row => row.name === packageName);
  assert.ok(record);
  const file = path.join(selected.include, ...relative.split("/"));
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, contents);
  record.files.push(file);
  return file;
}

/** @param {ReturnType<typeof fixture>} selected @param {{backward?: boolean, targetCxx?: boolean, targetSystem?: boolean, system?: boolean}} [options] */
function addRequiredRoots(selected, options = {}) {
  addFile(selected, packageNames[0] || "", "c++/13/bits/c++config.h");
  if (options.backward !== false) {addFile(selected, packageNames[0] || "", "c++/13/backward/hash_fun.h");}
  if (options.targetCxx !== false) {addFile(selected, packageNames[0] || "", "x86_64-linux-gnu/c++/13/bits/os_defines.h");}
  if (options.targetSystem !== false) {addFile(selected, packageNames[1] || "", "x86_64-linux-gnu/sys/types.h");}
  if (options.system !== false) {addFile(selected, packageNames[2] || "", "linux/if.h");}
}

/** @param {FixturePackage[]} records */
function useUnexpectedPackage(records) {if (records[0]) {records[0].name = "unexpected-package";}}
/** @param {FixturePackage[]} records */
function useWrongArchitecture(records) {if (records[1]) {records[1].architecture = "arm64";}}
/** @param {FixturePackage[]} records */
function useMalformedVersion(records) {if (records[2]) {records[2].version = "../bad";}}
/** @param {FixturePackage[]} records */
function removePackage(records) {records.pop();}

test.describe("Pinned GNU header producer fixtures", {concurrency: false}, () => {
  test("ignores dpkg's root-directory marker without normalizing other package paths", context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    for (const row of selected.records) {row.files.unshift("/.");}
    assert.equal(prepareLinuxGnuHeaders(selected.include, selected.output, selected.records).files.length, 5);
    selected.records[0]?.files.push("/usr/include/../escape.h");
    assert.throws(() => prepareLinuxGnuHeaders(selected.include, path.join(selected.root, "invalid"), selected.records), /Malformed GNU package file path/);
  });
  test("maps C++ target paths before the generic architecture/system roots", () => {
    assert.equal(virtualHeader("x86_64-linux-gnu/c++/13/bits/c++config.h"),
      "include/mom-cxx-target/bits/c++config.h");
    assert.equal(virtualHeader("x86_64-linux-gnu/sys/types.h"), "include/mom-system-target/sys/types.h");
    assert.equal(virtualHeader("c++/13/backward/hash_fun.h"), "include/mom-cxx/backward/hash_fun.h");
    assert.equal(virtualHeader("features.h"), "include/mom-system/features.h");
  });

  test("stages the four package-derived roots and re-admits a build-only handoff", context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    const result = prepareLinuxGnuHeaders(selected.include, selected.output, selected.records);
    assert.equal(result.root, selected.output);
    assert.equal(result.files.length, 5);
    assert.equal(result.buildOnly, true);
    assert.equal(result.compiled, false);
    assert.equal(result.executed, false);
    assert.equal(result.publicationQualified, false);
    assert.deepEqual(verifyLinuxGnuHeaders(selected.output), result);
    assert.equal(fs.readFileSync(path.join(selected.output, "include/mom-system/linux/if.h"), "utf8"),
      "linux/if.h");
  });

  test("ignores package-list paths outside the selected include root", context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    const outside = path.join(selected.root, "unrelated.h");
    fs.writeFileSync(outside, "not a staged input");
    selected.records[2]?.files.push(outside);
    const result = prepareLinuxGnuHeaders(selected.include, selected.output, selected.records);
    const rows = /** @type {Array<{source: string}>} */ (result.files);
    assert.equal(rows.some(row => row.source === outside), false);
  });

  test("rejects a missing architecture-specific C++ root", context => {
    const selected = fixture(context);
    addRequiredRoots(selected, {targetCxx: false});
    assert.throws(() => prepareLinuxGnuHeaders(selected.include, selected.output, selected.records),
      /lacks one of the four target roots/);
    assert.equal(fs.existsSync(selected.output), false);
  });

  test("rejects missing backward compatibility headers", context => {
    const selected = fixture(context);
    addRequiredRoots(selected, {backward: false});
    assert.throws(() => prepareLinuxGnuHeaders(selected.include, selected.output, selected.records),
      /backward headers/);
  });

  test("rejects a missing target system root", context => {
    const selected = fixture(context);
    addRequiredRoots(selected, {targetSystem: false});
    assert.throws(() => prepareLinuxGnuHeaders(selected.include, selected.output, selected.records),
      /lacks one of the four target roots/);
  });

  test("rejects duplicate ownership of the same staged header", context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    const existing = selected.records[0]?.files[0];
    assert.ok(existing);
    selected.records[1]?.files.push(existing);
    assert.throws(() => prepareLinuxGnuHeaders(selected.include, selected.output, selected.records),
      /Duplicate GNU package header/);
  });

  test("preserves distinct Linux package headers that differ only by case", {skip: process.platform === "win32"}, context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    addFile(selected, packageNames[2] || "", "misc/Header.h", "upper");
    addFile(selected, packageNames[2] || "", "misc/header.h", "lower");
    const result = prepareLinuxGnuHeaders(selected.include, selected.output, selected.records);
    const virtuals = /** @type {Array<{virtual: string}>} */ (result.files).map(row => row.virtual);
    assert.ok(virtuals.includes("include/mom-system/misc/Header.h"));
    assert.ok(virtuals.includes("include/mom-system/misc/header.h"));
  });

  test("resolves an in-root header symlink to staged regular bytes", {skip: process.platform === "win32"}, context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    const source = addFile(selected, packageNames[2] || "", "misc/real.h", "trusted header");
    const alias = path.join(selected.include, "misc/alias.h");
    fs.symlinkSync(source, alias);
    selected.records[2]?.files.push(alias);
    prepareLinuxGnuHeaders(selected.include, selected.output, selected.records);
    const staged = path.join(selected.output, "include/mom-system/misc/alias.h");
    assert.equal(fs.lstatSync(staged).isSymbolicLink(), false);
    assert.equal(fs.readFileSync(staged, "utf8"), "trusted header");
  });

  test("rejects a package header symlink that escapes the selected include root", {skip: process.platform === "win32"}, context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    const outside = path.join(selected.root, "outside.h");
    fs.writeFileSync(outside, "outside");
    const link = path.join(selected.include, "misc/escape.h");
    fs.mkdirSync(path.dirname(link), {recursive: true});
    fs.symlinkSync(outside, link);
    selected.records[2]?.files.push(link);
    assert.throws(() => prepareLinuxGnuHeaders(selected.include, selected.output, selected.records),
      /special or escapes/);
    assert.equal(fs.existsSync(selected.output), false);
  });

  test("rejects preexisting and overlapping output directories", context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    fs.mkdirSync(selected.output);
    assert.throws(() => prepareLinuxGnuHeaders(selected.include, selected.output, selected.records),
      /canonical, separate and newly staged/);
    const nested = path.join(selected.include, "staged");
    assert.throws(() => prepareLinuxGnuHeaders(selected.include, nested, selected.records),
      /canonical, separate and newly staged/);
  });

  test("rejects output paths whose parent is a symlink alias", {skip: process.platform === "win32"}, context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    const alias = path.join(selected.root, "parent-alias");
    fs.symlinkSync(selected.root, alias, "dir");
    assert.throws(() => prepareLinuxGnuHeaders(selected.include, path.join(alias, "staged"), selected.records),
      /canonical, separate and newly staged/);
  });

  test("rejects malformed package identity, architecture, version, and package list", context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    const mutations = [useUnexpectedPackage, useWrongArchitecture, useMalformedVersion, removePackage];
    for (const mutate of mutations) {
      const records = selected.records.map(record => ({...record, files: [...record.files]}));
      mutate(records);
      assert.throws(() => prepareLinuxGnuHeaders(selected.include, selected.output, records), /explicit installed amd64/);
    }
  });

  test("rejects relative and noncanonical package file paths", context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    for (const malformed of ["relative/header.h", selected.include + "/unused/../bad.h"]) {
      const records = selected.records.map(record => ({...record, files: [...record.files]}));
      records[2]?.files.push(malformed);
      assert.throws(() => prepareLinuxGnuHeaders(selected.include, selected.output, records),
        /Malformed GNU package file path/);
    }
  });

  test("rejects content tampering after staging", context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    prepareLinuxGnuHeaders(selected.include, selected.output, selected.records);
    const staged = path.join(selected.output, "include/mom-system/linux/if.h");
    fs.writeFileSync(staged, "tampered");
    assert.throws(() => verifyLinuxGnuHeaders(selected.output), /content or path changed/);
  });

  test("rejects a relocated handoff whose recorded root no longer matches", context => {
    const selected = fixture(context);
    addRequiredRoots(selected);
    prepareLinuxGnuHeaders(selected.include, selected.output, selected.records);
    const relocated = path.join(selected.root, "relocated");
    fs.renameSync(selected.output, relocated);
    assert.throws(() => verifyLinuxGnuHeaders(relocated), /Malformed GNU target-header handoff/);
  });
});

test.describe("Pinned Windows MinGW GNU header fixtures", {concurrency: false}, () => {
  test("stages only the three Windows MinGW roots and records resolved package ownership", {skip: process.platform !== "linux"}, context => {
    const selected = windowsFixture(context);
    addWindowsRequiredRoots(selected);
    const result = prepareWindowsGnuHeaders(selected.sysroot, selected.output, selected.records);
    assert.equal(result.target, "x86_64-w64-windows-gnu");
    assert.equal(result.files.length, 4);
    assert.equal(result.buildOnly, true);
    assert.equal(result.executed, false);
    assert.deepEqual(verifyWindowsGnuHeaders(selected.output), result);
    const virtuals = result.files.map(row => row.virtual);
    assert.ok(virtuals.includes("include/mom-cxx/algorithm"));
    assert.ok(virtuals.includes("include/mom-cxx/backward/backward_warning.h"));
    assert.ok(virtuals.includes("include/mom-cxx-target/bits/c++config.h"));
    assert.ok(virtuals.includes("include/mom-system-target/windows.h"));
    assert.equal(virtuals.some(name => name.startsWith("include/mom-system/")), false);
    const system = result.files.find(row => row.virtual === "include/mom-system-target/windows.h");
    assert.equal(system?.package, windowsPackageNames[1]);
    assert.equal(system?.resolvedPackage, windowsPackageNames[0]);
    assert.equal(fs.readFileSync(path.join(selected.output, "include/mom-system-target/windows.h"), "utf8"), "windows bytes");
  });

  test("rejects a missing Windows target system header root", {skip: process.platform !== "linux"}, context => {
    const selected = windowsFixture(context);
    addWindowsRequiredRoots(selected, false);
    assert.throws(() => prepareWindowsGnuHeaders(selected.sysroot, selected.output, selected.records), /three target roots/);
    assert.equal(fs.existsSync(selected.output), false);
  });

  test("rejects a Windows header symlink that escapes the declared common-header root", {skip: process.platform !== "linux"}, context => {
    const selected = windowsFixture(context);
    addWindowsRequiredRoots(selected, false);
    const outside = path.join(selected.root, "outside.h");
    fs.writeFileSync(outside, "outside");
    const alias = path.join(selected.sysroot, "usr/x86_64-w64-mingw32/include/escape.h");
    fs.mkdirSync(path.dirname(alias), {recursive: true});
    fs.symlinkSync(outside, alias);
    selected.records[1]?.files.push("/usr/x86_64-w64-mingw32/include/escape.h");
    assert.throws(() => prepareWindowsGnuHeaders(selected.sysroot, selected.output, selected.records), /escapes its package-owned/);
  });

  test("requires every Windows header symlink target to have selected common-package ownership", {skip: process.platform !== "linux"}, context => {
    const selected = windowsFixture(context);
    addWindowsRequiredRoots(selected, false);
    const real = path.join(selected.sysroot, "usr/share/mingw-w64/include/unowned.h");
    fs.mkdirSync(path.dirname(real), {recursive: true});
    fs.writeFileSync(real, "unowned");
    const alias = path.join(selected.sysroot, "usr/x86_64-w64-mingw32/include/unowned.h");
    fs.mkdirSync(path.dirname(alias), {recursive: true});
    fs.symlinkSync("../../share/mingw-w64/include/unowned.h", alias);
    selected.records[1]?.files.push("/usr/x86_64-w64-mingw32/include/unowned.h");
    assert.throws(() => prepareWindowsGnuHeaders(selected.sysroot, selected.output, selected.records), /absent from its selected package file list/);
  });

  test("rejects a Windows profile with a wrong package architecture or Linux package name", {skip: process.platform !== "linux"}, context => {
    const selected = windowsFixture(context);
    addWindowsRequiredRoots(selected);
    const wrongArch = selected.records.map(record => ({...record, files: [...record.files]}));
    if (wrongArch[0]) {wrongArch[0].architecture = "amd64";}
    assert.throws(() => prepareWindowsGnuHeaders(selected.sysroot, selected.output, wrongArch), /explicit installed package records/);
    const wrongPackage = selected.records.map(record => ({...record, files: [...record.files]}));
    if (wrongPackage[0]) {wrongPackage[0].name = packageNames[0] || "";}
    assert.throws(() => prepareWindowsGnuHeaders(selected.sysroot, selected.output, wrongPackage), /explicit installed package records/);
  });

  test("does not admit Windows staged headers through the Linux verifier", {skip: process.platform !== "linux"}, context => {
    const selected = windowsFixture(context);
    addWindowsRequiredRoots(selected);
    prepareWindowsGnuHeaders(selected.sysroot, selected.output, selected.records);
    assert.throws(() => verifyLinuxGnuHeaders(selected.output), /Malformed GNU target-header handoff/);
  });
});

test.describe("GNU header target and CLI guards", {concurrency: false}, () => {
  test("fails closed for inherited profile keys", () => {
    assert.throws(() => prepareGnuHeaders("constructor", "/unused", []), /Unsupported GNU header target/);
  });

  test("CLI rejects extra positional arguments", {skip: process.platform !== "linux"}, () => {
    const script = require.resolve("../scripts/prepare-native-gnu-headers");
    const result = spawnSync(process.execPath, [script, "verify", "/unused", "x86_64-w64-windows-gnu", "extra"], {
      encoding: "utf8", timeout: 5000, env: {PATH: process.env["PATH"] || "/usr/bin:/bin", MOM_AOT_BUILD_ONLY: "1"},
    });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /Usage \(Linux x64/);
  });
});
