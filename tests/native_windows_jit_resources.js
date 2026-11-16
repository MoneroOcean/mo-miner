"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path");
const {fileHash, buildFileInputs, sourceInputs} = require("../scripts/build-source-inputs");
const {REQUIRED} = require("../scripts/prepare-native-jit-base");
const {prepareWindowsGnuHeaders} = require("../scripts/prepare-native-gnu-headers");
const {jitSourceInputs} = require("../scripts/native-jit-sdk");
const {windowsJitResourcePaths, profileFromAdmissions, composeWindowsJitResourceProfile, admitWindowsGnu, admitEmitter, verifyRequiredBaseManifest,
  validateWindowsJitNativeTools} = require("../scripts/native-windows-jit-resources");

const suffix = Date.now().toString() + String(process.pid % 1000).padStart(3, "0") + "-88";
const repo = path.resolve(__dirname, "..");

/** @param {import("node:test").TestContext} context @param {boolean} [gnu] */
function roots(context, gnu = true) {
  const linux = path.join("/tmp", "mom-native-linux-jit-resources-" + suffix),
    windowsGnu = path.join("/tmp", "mom-native-windows-gnu-headers-" + suffix),
    output = path.join("/tmp", "mom-native-windows-jit-resources-" + suffix),
    emitter = path.join(output, "emitter"), dpcpp = path.join("/tmp", "mom-native-dpcpp-sdk-" + suffix);
  for (const root of [linux, output, emitter, dpcpp, ...(gnu ? [windowsGnu] : [])]) {fs.mkdirSync(root, {recursive: true});}
  context.after(() => {for (const root of [linux, windowsGnu, output, dpcpp]) {fs.rmSync(root, {recursive: true, force: true});}});
  return {linux, windowsGnu, output, emitter, dpcpp};
}

/** @param {string} root @param {unknown} value */
function writeJson(root, value) {
  const file = path.join(root, "ci-result.json"); fs.writeFileSync(file, JSON.stringify(value) + "\n", {flag: "wx"}); return file;
}

/** @param {Record<string, unknown>} [changes] */
function linuxResult(changes = {}) {
  return {version: 1, target: "x86_64-unknown-linux-gnu", buildOnly: true, executed: false, publicationQualified: false,
    inputArchives: [], ...changes};
}

/** @param {ReturnType<typeof roots>} selected @param {string} linuxResultSha256 */
function input(selected, linuxResultSha256) {
  return {runAttempt: suffix, linuxResourcesRoot: selected.linux, linuxResultSha256,
    windowsGnuRoot: selected.windowsGnu, windowsGnuResultSha256: "a".repeat(64), windowsGnuCopyrightsSha256: "b".repeat(64),
    emitterRoot: selected.emitter, emitterResultSha256: "c".repeat(64), dpcppRoot: selected.dpcpp,
    outputProfile: path.join(selected.output, "resource-profile.json")};
}

/** @param {import("node:test").TestContext} context */
function windowsGnuFixture(context) {
  const root = fs.mkdtempSync(path.join("/tmp", "mom-jit-windows-gnu-sysroot-")), sysroot = path.join(root, "sysroot"),
    output = path.join("/tmp", "mom-native-windows-gnu-headers-" + suffix),
    names = /** @type {[string,string,string]} */ (["mingw-w64-common", "mingw-w64-x86-64-dev", "g++-mingw-w64-x86-64-posix"]),
    records = /** @type {Array<{name:string,version:string,architecture:string,files:string[]}>} */ ([
      {name: names[0], version: "13.0.0-1", architecture: "all", files: []},
      {name: names[1], version: "13.0.0-1", architecture: "all", files: []},
      {name: names[2], version: "13.2.0-1", architecture: "amd64", files: []},
    ]);
  fs.mkdirSync(sysroot); context.after(() => {fs.rmSync(root, {recursive: true, force: true}); fs.rmSync(output, {recursive: true, force: true});});
  /** @param {number} packageIndex @param {string} logical @param {string} [body] */
  const add = (packageIndex, logical, body = logical) => {
    const file = path.join(sysroot, logical.slice(1)); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, body);
    const row = records[packageIndex]; assert.ok(row); row.files.push(logical); return file;
  };
  add(2, "/usr/lib/gcc/x86_64-w64-mingw32/13-posix/include/c++/algorithm");
  add(2, "/usr/lib/gcc/x86_64-w64-mingw32/13-posix/include/c++/backward/backward_warning.h");
  add(2, "/usr/lib/gcc/x86_64-w64-mingw32/13-posix/include/c++/x86_64-w64-mingw32/bits/c++config.h");
  const common = add(0, "/usr/share/mingw-w64/include/windows.h", "fixture header");
  const alias = path.join(sysroot, "usr/x86_64-w64-mingw32/include/windows.h"); fs.mkdirSync(path.dirname(alias), {recursive: true});
  fs.symlinkSync("../../share/mingw-w64/include/windows.h", alias); assert.equal(fs.realpathSync(alias), common);
  records[1]?.files.push("/usr/x86_64-w64-mingw32/include/windows.h");
  const headers = prepareWindowsGnuHeaders(sysroot, output, records), notices = headers.packages.map(pkg => {
    const file = path.join(output, "licenses", pkg.name, "copyright"); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "license for " + pkg.name + "\n");
    return {package: pkg.name, version: pkg.version, sourcePath: "/usr/share/doc/" + pkg.name + "/copyright", path: file, sha256: fileHash(file)};
  });
  const copyrights = path.join(output, "copyrights.json"); fs.writeFileSync(copyrights, JSON.stringify(notices, null, 2) + "\n");
  return {root: output, headers, notices, resultFile: path.join(output, "result.json"), copyrights};
}

/** @param {import("node:test").TestContext} context @param {ReturnType<typeof roots>} selected */
function emitterFixture(context, selected) {
  const compiler = path.join(selected.dpcpp, "bin/clang++"), sycl = path.join(selected.dpcpp, "include/sycl/sycl.hpp"),
    fixture = path.join(selected.emitter, "native-kawpow-source"), header = path.join(selected.windowsGnu, "include/mom-system-target/windows.h"),
    targetList = path.join(selected.windowsGnu, "target-headers.json");
  fs.mkdirSync(path.dirname(compiler), {recursive: true}); fs.mkdirSync(path.dirname(sycl), {recursive: true});
  fs.mkdirSync(path.dirname(fixture), {recursive: true}); fs.mkdirSync(path.dirname(header), {recursive: true});
  fs.writeFileSync(compiler, "fixture compiler"); fs.writeFileSync(sycl, "fixture SYCL header");
  fs.writeFileSync(fixture, "fixture emitter executable"); fs.writeFileSync(header, "fixture Windows header");
  fs.writeFileSync(targetList, JSON.stringify([header]) + "\n");
  const leaves = [...new Set([...jitSourceInputs, "scripts/native-jit-sdk.js", "scripts/build-source-inputs.js", "scripts/build-environment.js"])],
    requiredInputs = [process.execPath, compiler, sycl, header, ...leaves.map(name => path.join(repo, name))].map(file => fs.realpathSync(file)),
    receipt = {version: 1, fixture, fixtureSha256: fileHash(fixture), compiler, compilerSha256: fileHash(compiler),
      inputs: buildFileInputs(requiredInputs), recipe: sourceInputs(repo, [], leaves), targetHeaders: [header],
      buildOnly: true, executed: false, publicationQualified: false, scope: "unit fixture"},
    resultFile = path.join(selected.emitter, "result.json");
  fs.writeFileSync(resultFile, JSON.stringify(receipt, null, 2) + "\n", {flag: "wx"});
  context.after(() => { /* Parent fixture owns these files and removes its run roots. */ });
  return {receipt, resultFile, compiler, fixture, header, targetList};
}

/** @param {ReturnType<typeof roots>} selected */
function nativeToolsFixture(selected) {
  const nativeRoot = path.join(selected.output, "native-tools"), binRoot = path.join(nativeRoot, "bin"),
    source = {receiptFile: path.join("/tmp", "mom-native-linux-source-prerequisites-" + suffix, "prepared/compiler/sources.json"),
      receiptSha256: "a".repeat(64), dpcppCommit: "b".repeat(40)},
    linux = {root: path.join("/tmp", "mom-native-linux-jit-resources-" + suffix), resultSha256: "c".repeat(64),
      baseReceiptSha256: "d".repeat(64), buildRoot: path.join("/tmp", "mom-native-linux-jit-resources-" + suffix, "bootstrap/build")};
  fs.mkdirSync(binRoot, {recursive: true});
  const files = ["clang-tblgen", "llvm-tblgen"].map((name, index) => {
    const file = path.join(binRoot, name), body = Buffer.from("fixture native TableGen " + index + "\n");
    fs.writeFileSync(file, body, {flag: "wx"}); fs.chmodSync(file, 0o755);
    return {path: "bin/" + name, bytes: body.length, mode: 0o755, sha256: fileHash(file)};
  });
  const manifest = {version: 1, kind: "windows-jit-native-tablegen", runAttempt: suffix, buildOnly: true, executed: false, publicationQualified: false,
      linux, source, files}, manifestFile = path.join(nativeRoot, "manifest.json");
  fs.writeFileSync(manifestFile, JSON.stringify(manifest, null, 2) + "\n", {flag: "wx"});
  return {nativeRoot, binRoot, source, linux, files, manifest, manifestFile};
}

test.describe("Windows GNU source-JIT profile composition", {concurrency: false}, () => {
  test("admits materialized bitcode destinations, not their original SDK source paths", () => {
    const manifest = {version: 1, files: Object.values(REQUIRED).map(row => ({source: row.virtual, virtual: row.virtual, sha256: row.sha256}))};
    verifyRequiredBaseManifest(manifest);
    const libdevice = manifest.files.find(row => row.virtual === REQUIRED.libdevice.virtual);
    assert.ok(libdevice);
    libdevice.source = REQUIRED.libdevice.source;
    assert.throws(() => verifyRequiredBaseManifest(manifest), /pinned libdevice/);
    libdevice.source = REQUIRED.libdevice.virtual; libdevice.sha256 = "0".repeat(64);
    assert.throws(() => verifyRequiredBaseManifest(manifest), /pinned libdevice/);
  });
});

test.describe("Linux-hosted Windows resource filesystem admission", {concurrency: false, skip: process.platform !== "linux"}, () => {
  test("keeps emitter/profile evidence beside the prepared resource child in one run root", () => {
    const paths = windowsJitResourcePaths("9000000000000008-88");
    assert.equal(paths.root, "/tmp/mom-native-windows-jit-resources-9000000000000008-88");
    assert.equal(paths.emitter, path.join(paths.root, "emitter"));
    assert.equal(paths.profile, path.join(paths.root, "resource-profile.json"));
    assert.equal(paths.resources, path.join(paths.root, "resources"));
    assert.equal(paths.nativeTools, path.join(paths.root, "native-tools"));
    assert.throws(() => windowsJitResourcePaths("9000000000000008-0"), /run\/attempt/);
  });

  test("maps admitted Windows roots, existing base headers, required bitcode pins and notices into the established profile", () => {
    const baseRoot = "/tmp/mom-native-linux-jit-resources-9000000000000008-88/base/base",
      gnuRoot = "/tmp/mom-native-windows-gnu-headers-9000000000000008-88",
      emitterRoot = "/tmp/mom-native-windows-jit-resources-9000000000000008-88/emitter",
      noticePath = path.join(gnuRoot, "licenses/mingw-w64-common/copyright"),
      profile = profileFromAdmissions({baseRoot, baseManifest: path.join(path.dirname(baseRoot), "base-manifest.json")},
        {root: gnuRoot, notices: [{package: "mingw-w64-common", path: noticePath, sha256: "d".repeat(64)}]},
        {fixture: path.join(emitterRoot, "native-kawpow-source"), compiler: "/tmp/dpcpp-run/bin/clang++", receiptFile: path.join(emitterRoot, "result.json")});
    assert.equal(profile.target, "x86_64-w64-windows-gnu");
    assert.deepEqual(profile.headerRoots.map(row => row.virtual), ["include/mom-cxx", "include/mom-cxx-target", "include/mom-system-target"]);
    assert.equal(profile.syclInclude, path.join(baseRoot, "include"));
    assert.equal(profile.clangBuiltinInclude, path.join(baseRoot, "lib/clang/23/include"));
    assert.deepEqual(profile.required, Object.fromEntries(Object.entries(REQUIRED).map(([name, row]) => [name, {virtual: row.virtual, sha256: row.sha256}])));
    assert.deepEqual(profile.notices, [{source: "licenses/mingw-w64-common/copyright", virtual: "usr/share/licenses/mingw-w64-common/copyright", sha256: "d".repeat(64)}]);
  });

  test("rejects a wrong-target Linux resource receipt even when its external file hash matches", context => {
    const selected = roots(context), file = writeJson(selected.linux, linuxResult({target: "x86_64-w64-windows-gnu"}));
    assert.throws(() => composeWindowsJitResourceProfile(input(selected, fileHash(file))), /not the same-run build-only/);
  });

  test("rejects a Linux resource result that omits its run-bound source and SDK archive lineage", context => {
    const selected = roots(context), file = writeJson(selected.linux, linuxResult({inputArchives: [{role: "linux-sources", name: "old.tar",
      root: "mom-native-linux-source-prerequisites-1-1", sha256: "a".repeat(64)}]}));
    assert.throws(() => composeWindowsJitResourceProfile(input(selected, fileHash(file))), /exact same-run source/);
  });

  test("rejects a changed Linux result pin before using its contents", context => {
    const selected = roots(context); writeJson(selected.linux, linuxResult());
    assert.throws(() => composeWindowsJitResourceProfile(input(selected, "0".repeat(64))), /differs from its external pin/);
  });

  test("rejects a root from a different run attempt", context => {
    const selected = roots(context); const request = input(selected, "a".repeat(64)); request.runAttempt = "9000000000000008-89";
    assert.throws(() => composeWindowsJitResourceProfile(request), /canonical current-run root/);
  });

  test("fails closed when the same-run Windows GNU header package root is absent", context => {
    const selected = roots(context, false), file = writeJson(selected.linux, linuxResult());
    assert.throws(() => composeWindowsJitResourceProfile(input(selected, fileHash(file))), /canonical directory/);
  });

  test("requires profile output to remain a new sibling in the same Windows resource root", context => {
    const selected = roots(context), file = writeJson(selected.linux, linuxResult()), request = input(selected, fileHash(file));
    request.outputProfile = path.join(selected.linux, "profile.json");
    assert.throws(() => composeWindowsJitResourceProfile(request), /profile path is not bound to this run attempt/);
  });

  test("admitWindowsGnu rechecks exact package versions and each captured notice body", context => {
    const fixture = windowsGnuFixture(context), admitted = admitWindowsGnu(fixture.root, suffix, fileHash(fixture.resultFile), fileHash(fixture.copyrights));
    assert.equal(admitted.headers.target, "x86_64-w64-windows-gnu"); assert.equal(admitted.headers.files.length, 4);
    const notices = JSON.parse(fs.readFileSync(fixture.copyrights, "utf8"));
    notices[0].version = "13.0.0-2"; fs.writeFileSync(fixture.copyrights, JSON.stringify(notices, null, 2) + "\n");
    assert.throws(() => admitWindowsGnu(fixture.root, suffix, fileHash(fixture.resultFile), fileHash(fixture.copyrights)), /copyright content, package or path changed/);
    notices[0].version = admitted.headers.packages[0]?.version; fs.writeFileSync(fixture.copyrights, JSON.stringify(notices, null, 2) + "\n");
    Object.assign(notices[2], {resolvedSourcePath: "/usr/share/doc/unrelated-package/copyright",
      ownershipPath: "/usr/share/doc/g++-mingw-w64-x86-64-posix"});
    fs.writeFileSync(fixture.copyrights, JSON.stringify(notices, null, 2) + "\n");
    assert.throws(() => admitWindowsGnu(fixture.root, suffix, fileHash(fixture.resultFile), fileHash(fixture.copyrights)), /copyright content, package or path changed/);
    delete notices[2].resolvedSourcePath; delete notices[2].ownershipPath;
    fs.writeFileSync(fixture.copyrights, JSON.stringify(notices, null, 2) + "\n");
    fs.appendFileSync(notices[0]?.path || "", "tampered\n");
    assert.throws(() => admitWindowsGnu(fixture.root, suffix, fileHash(fixture.resultFile), fileHash(fixture.copyrights)), /copyright content, package or path changed/);
  });

  test("admitEmitter binds maintained source recipe, SDK compiler and the exact target-header set", context => {
    const selected = roots(context), fixture = emitterFixture(context, selected), gnu = {files: [{virtual: "include/mom-system-target/windows.h"}]};
    const admitted = admitEmitter(selected.emitter, selected.dpcpp, selected.windowsGnu, gnu, fileHash(fixture.resultFile));
    assert.equal(admitted.fixture, fixture.fixture); assert.equal(admitted.compiler, fixture.compiler);
    const receipt = JSON.parse(fs.readFileSync(fixture.resultFile, "utf8")); receipt.targetHeaders = [];
    fs.writeFileSync(fixture.resultFile, JSON.stringify(receipt, null, 2) + "\n");
    assert.throws(() => admitEmitter(selected.emitter, selected.dpcpp, selected.windowsGnu, gnu, fileHash(fixture.resultFile)), /not bound to the fresh/);
  });

  test("admitEmitter rejects input payload drift despite an updated receipt pin", context => {
    const selected = roots(context), fixture = emitterFixture(context, selected), gnu = {files: [{virtual: "include/mom-system-target/windows.h"}]};
    fs.appendFileSync(fixture.header, "changed\n");
    assert.throws(() => admitEmitter(selected.emitter, selected.dpcpp, selected.windowsGnu, gnu, fileHash(fixture.resultFile)), /inputs differ from their receipt/);
  });

  test("admits only the two run-bound native TableGen executables beside runtime resources", context => {
    const selected = roots(context), fixture = nativeToolsFixture(selected),
      admitted = validateWindowsJitNativeTools(selected.output, suffix, fixture.source);
    assert.equal(admitted.toolRoot, fixture.nativeRoot);
    assert.equal(path.dirname(admitted.toolRoot), selected.output);
    assert.notEqual(admitted.toolRoot, path.join(selected.output, "resources"));
    assert.equal(admitted.manifestFile, fixture.manifestFile);
    assert.equal(admitted.manifestSha256, fileHash(fixture.manifestFile));
    assert.equal(admitted.files.clangTableGen, path.join(fixture.binRoot, "clang-tblgen"));
    assert.equal(admitted.files.llvmTableGen, path.join(fixture.binRoot, "llvm-tblgen"));
    assert.equal(admitted.hashes.clangTableGenSha256, fixture.files[0]?.sha256);
    assert.equal(admitted.hashes.llvmTableGenSha256, fixture.files[1]?.sha256);
    assert.equal(admitted.source.commit, fixture.source.dpcppCommit);
  });

  test("rejects a native TableGen payload from another run or source receipt", context => {
    const selected = roots(context), fixture = nativeToolsFixture(selected);
    assert.throws(() => validateWindowsJitNativeTools(selected.output, "9000000000000008-89", fixture.source), /canonical current-run root/);
    assert.throws(() => validateWindowsJitNativeTools(selected.output, suffix, {...fixture.source, receiptSha256: "e".repeat(64)}), /not bound to the current same-run Linux JIT source/);
    assert.throws(() => validateWindowsJitNativeTools(selected.output, suffix, {...fixture.source, dpcppCommit: "f".repeat(40)}), /not bound to the current same-run Linux JIT source/);
  });

  test("rejects modified or non-executable native TableGen files", context => {
    const selected = roots(context), fixture = nativeToolsFixture(selected), changed = path.join(fixture.binRoot, "llvm-tblgen");
    fs.appendFileSync(changed, "tampered");
    assert.throws(() => validateWindowsJitNativeTools(selected.output, suffix, fixture.source), /changed or exceeds bounds/);
    fs.writeFileSync(changed, "fixture native TableGen 1\n"); fs.chmodSync(changed, 0o644);
    const row = fixture.manifest.files[1]; assert.ok(row); row.bytes = fs.statSync(changed).size; row.sha256 = fileHash(changed); row.mode = 0o644;
    fs.writeFileSync(fixture.manifestFile, JSON.stringify(fixture.manifest, null, 2) + "\n");
    assert.throws(() => validateWindowsJitNativeTools(selected.output, suffix, fixture.source), /file row is invalid/);
  });

  test("rejects stale native TableGen hashes and an incomplete two-tool closure", context => {
    const selected = roots(context), fixture = nativeToolsFixture(selected), clangRow = fixture.manifest.files[0];
    assert.ok(clangRow);
    clangRow.sha256 = "0".repeat(64);
    fs.writeFileSync(fixture.manifestFile, JSON.stringify(fixture.manifest, null, 2) + "\n");
    assert.throws(() => validateWindowsJitNativeTools(selected.output, suffix, fixture.source), /changed or exceeds bounds/);
    clangRow.sha256 = fileHash(path.join(fixture.binRoot, "clang-tblgen"));
    fixture.manifest.files.pop();
    fs.writeFileSync(fixture.manifestFile, JSON.stringify(fixture.manifest, null, 2) + "\n");
    assert.throws(() => validateWindowsJitNativeTools(selected.output, suffix, fixture.source), /not bound to the current same-run Linux JIT source/);
  });

  test("rejects native TableGen inventory aliases and path traversal", context => {
    const selected = roots(context), fixture = nativeToolsFixture(selected), clangRow = fixture.manifest.files[0];
    fs.writeFileSync(path.join(fixture.nativeRoot, "extra"), "not allowed");
    assert.throws(() => validateWindowsJitNativeTools(selected.output, suffix, fixture.source), /extra entries/);
    fs.unlinkSync(path.join(fixture.nativeRoot, "extra"));
    assert.ok(clangRow);
    clangRow.path = "../clang-tblgen";
    fs.writeFileSync(fixture.manifestFile, JSON.stringify(fixture.manifest, null, 2) + "\n");
    assert.throws(() => validateWindowsJitNativeTools(selected.output, suffix, fixture.source), /file row is invalid/);
    fs.unlinkSync(path.join(fixture.binRoot, "clang-tblgen"));
    fs.symlinkSync(path.join(fixture.binRoot, "llvm-tblgen"), path.join(fixture.binRoot, "clang-tblgen"));
    clangRow.path = "bin/clang-tblgen";
    fs.writeFileSync(fixture.manifestFile, JSON.stringify(fixture.manifest, null, 2) + "\n");
    assert.throws(() => validateWindowsJitNativeTools(selected.output, suffix, fixture.source), /canonical directory|regular file/);
  });
});
