"use strict";

const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), childProcess = require("node:child_process");
const {test, describe} = require("node:test"), assert = require("node:assert/strict");
const {createHash} = require("node:crypto");
const sourceIO = require("../scripts/build-source-inputs"), llvmTools = require("../scripts/native-linux-llvm-tools");
const sourcePins = require("../scripts/fetch-native-sdk-sources"), llvmSdk = require("../scripts/native-llvm-sdk");
const ci = require("../scripts/native-ci-linux-sources");
const localSdk = require("../scripts/build-local-native-sdk"), localRocm = require("../scripts/build-local-native-rocm-sdk");
const {windowsAmdhipCatalog, prepareWindowsAmdhip, admitWindowsAmdhip, windowsAmdhipInputs, amdhipRecipe,
  deriveDefinition, verifyLinuxToolHandoff, prepareLocalAmdhipWindows} = require("../scripts/prepare-native-amdhip-windows");

/** @param {Buffer|string} value */
function sha256(value) {return createHash("sha256").update(value).digest("hex");}
/** @param {string} file @param {Buffer|string} value */
function writePrivate(file, value) {fs.writeFileSync(file, value, {flag: "wx", mode: 0o600});}
/** @param {string} root */
function makeWritable(root) {
  if (!fs.existsSync(root)) {return;}
  const stat = fs.lstatSync(root);
  if (stat.isSymbolicLink()) {return;}
  if (stat.isDirectory()) {fs.chmodSync(root, 0o700); for (const name of fs.readdirSync(root)) {makeWritable(path.join(root, name));}} else if (stat.isFile()) {fs.chmodSync(root, 0o600);}
}
/** @param {import("node:test").TestContext} t */
function fixtureRoot(t) {
  const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "mom-win-amdhip-"));
  t.after(() => {makeWritable(root); fs.rmSync(root, {recursive: true, force: true});});
  return root;
}
/** @param {string} root @param {Buffer} library @param {Buffer} license */
function receiptFixture(root, library, license) {
  const c = windowsAmdhipCatalog(), recipe = amdhipRecipe(), output = path.join(root, "input"),
    libraryPath = path.join(output, c.library.fileName), licensePath = path.join(output, path.posix.basename(c.source.license.path)), receiptFile = path.join(output, "sdk.json");
  fs.mkdirSync(output, {mode: 0o700}); writePrivate(libraryPath, library); writePrivate(licensePath, license);
  const receipt = {
    version: 2, target: "windows-amdhip64", status: "PASS", networked: false, buildOnly: true, compiled: false, executed: false,
    vendorPrebuilt: false, sourceGenerated: true, publicationQualified: false, redistributionQualified: false, producerSha256: "a".repeat(64),
    recipe: {sha256: sha256(JSON.stringify(recipe)), files: recipe.graph},
    source: {origin: c.source.origin, revision: c.source.revision, sourceRoot: "/tmp/mom-native-rocm-prerequisites-123456-1",
      fetchReceiptSha256: "b".repeat(64), rootsReceiptSha256: "c".repeat(64), git: {path: "/usr/bin/git", bytes: 123456, sha256: "d".repeat(64)},
      definition: {path: c.source.definition.path, blob: c.source.definition.blob, bytes: c.source.definition.bytes,
        sha256: c.source.definition.sha256, actualSha256: c.source.definition.sha256},
      license: {path: c.source.license.path, blob: c.source.license.blob, bytes: c.source.license.bytes,
        sha256: c.source.license.sha256, actualSha256: c.source.license.sha256, note: "Original upstream license text is preserved as evidence; no legal conclusion is asserted."}},
    toolchain: {role: c.toolchain.llvmRole, handoffRoot: "/runner-temp/tools/native-linux-llvm-tools-123456-1", handoffSha256: "e".repeat(64),
      llvmLib: {relative: c.toolchain.llvmLibAlias, aliasTarget: c.toolchain.llvmLibTarget, bytes: 4567, sha256: "f".repeat(64)},
      command: ["/def:<staging>/amdhip.def", "/machine:x64", "/out:<staging>/" + c.library.fileName]},
    definition: {path: c.source.definition.path, sourceBlob: c.source.definition.blob, sourceSha256: c.source.definition.sha256,
      bytes: c.source.definition.bytes, derivedBytes: c.source.definition.derivedBytes, derivedSha256: c.source.definition.derivedSha256,
      transformation: "Only a LIBRARY directive is prepended to the unchanged upstream .def bytes."},
    library: {fileName: c.library.fileName, dllName: c.library.dllName, bytes: c.library.bytes, sha256: c.library.sha256},
    output: "C:\\removed\\amdhip-output", files: {
      [c.library.fileName]: {kind: "file", bytes: c.library.bytes, sha256: c.library.sha256,
        source: {kind: "generated-from-pinned-definition", path: c.source.definition.path, blob: c.source.definition.blob, sha256: c.source.definition.sha256}},
      [path.posix.basename(c.source.license.path)]: {kind: "file", bytes: c.source.license.bytes, sha256: c.source.license.sha256,
        source: {kind: "original-source-license-blob", path: c.source.license.path, blob: c.source.license.blob}}
    },
    note: "Source-generated Windows AMD import library only; build-only and not publication or redistribution qualified."
  };
  writePrivate(receiptFile, JSON.stringify(receipt));
  fs.chmodSync(libraryPath, 0o444); fs.chmodSync(licensePath, 0o444); fs.chmodSync(receiptFile, 0o444); fs.chmodSync(output, 0o555);
  return {output, libraryPath, licensePath, receiptFile, receipt, library, license};
}

/** Install a narrow content-aware hash mock for the two fixture payloads. @param {import("node:test").TestContext} t @param {ReturnType<typeof receiptFixture>} fixture */
function mockFixtureHashes(t, fixture) {
  const real = sourceIO.fileHash;
  /** @param {string} file */
  const fixtureHash = file => {
    if (path.basename(file) === windowsAmdhipCatalog().library.fileName) {
      const bytes = fs.readFileSync(file); return bytes.equals(fixture.library) ? windowsAmdhipCatalog().library.sha256 : sha256(bytes);
    }
    if (path.basename(file) === path.posix.basename(windowsAmdhipCatalog().source.license.path)) {
      const bytes = fs.readFileSync(file); return bytes.equals(fixture.license) ? windowsAmdhipCatalog().source.license.sha256 : sha256(bytes);
    }
    return real(file);
  };
  t.mock.method(sourceIO, "fileHash", fixtureHash);
}

/** @param {import("node:test").TestContext} t */
function localWorkspaceFixture(t) {
  const root = fixtureRoot(t), workspace = path.join(root, "workspace"); localSdk.admitWorkspace(workspace, fs.realpathSync(path.resolve(__dirname, "..")));
  const sourcePaths = localRocm.localRocmSdkPaths(workspace), sourceRoot = sourcePaths.source,
    llvmOutput = path.join(workspace, "llvm21"), sdk = path.join(llvmOutput, "sdk"), bin = path.join(sdk, "bin"),
    llvmAr = path.join(bin, "llvm-ar"), llvmLib = path.join(bin, "llvm-lib"), output = path.join(workspace, "windows-amdhip"),
    localBuildId = "12345678-1234-4234-9234-123456789abc", input = {sourceRoot, llvmOutput, llvmLib, output, localBuildId};
  fs.mkdirSync(path.join(sourceRoot, "roots/clr"), {recursive: true}); fs.mkdirSync(sourcePaths.pythonSourceRoot, {recursive: true});
  fs.mkdirSync(bin, {recursive: true});
  for (const file of [sourcePaths.fetchReceipt, sourcePaths.rootsFile, sourcePaths.pythonArchive, sourcePaths.pythonSourceReceipt, path.join(llvmOutput, "result.json")]) {
    writePrivate(file, path.basename(file) + " fixture\n");
  }
  writePrivate(llvmAr, "admitted llvm-ar fixture\n"); fs.chmodSync(llvmAr, 0o755); fs.symlinkSync("llvm-ar", llvmLib);
  return {root, workspace, sourcePaths, sourceRoot, llvmOutput, sdk, llvmAr, llvmLib, output, localBuildId, input};
}

/** @param {import("node:test").TestContext} t @param {ReturnType<typeof localWorkspaceFixture>} f @param {{mutateTool?:boolean}} [options] */
function mockLocalAmdhipBuild(t, f, options = {}) {
  const c = windowsAmdhipCatalog(), definition = fs.readFileSync(path.join(__dirname, "fixtures/native-windows-amdhip/amdhip.def")),
    license = fs.readFileSync(path.join(__dirname, "fixtures/native-windows-amdhip/LICENSE.md")), fakeLibrary = Buffer.alloc(c.library.bytes, 0x41);
  let sourceVerifications = 0, llvmAdmissions = 0;
  const verifyLlvm = /** @param {string} output */ output => {
      llvmAdmissions++; assert.equal(output, f.llvmOutput);
      const stat = fs.statSync(f.llvmAr);
      return {plan: {output, prefix: f.sdk}, sdk: {path: f.sdk, snapshot: {files: {
        [c.toolchain.llvmLibAlias]: {type: "symlink", target: c.toolchain.llvmLibTarget},
        ["bin/" + c.toolchain.llvmLibTarget]: {type: "file", bytes: stat.size, sha256: sourceIO.fileHash(f.llvmAr)}}}}};
    }, originalFileHash = sourceIO.fileHash;
  assert.equal(sha256(definition), c.source.definition.sha256); assert.equal(sha256(license), c.source.license.sha256);
  t.mock.method(ci, "verifyRocmSources", /** @param {{env:NodeJS.ProcessEnv}} wrapper
   * @param {ReturnType<typeof localRocm.localRocmSdkPaths>} paths @param {string} git */ (wrapper, paths, git) => {
      sourceVerifications++; assert.equal(paths.source, f.sourceRoot); assert.equal(wrapper.env["MOM_AOT_BUILD_ONLY"], "1"); assert.equal(git, "/usr/bin/git");
      return {version: 1, target: "rocm"};
    });
  t.mock.method(sourceIO, "fileHash", /** @param {string} file */ file => {
    if (path.basename(file) === c.library.fileName) {
      const bytes = fs.readFileSync(file); return bytes.equals(fakeLibrary) ? c.library.sha256 : sha256(bytes);
    }
    return originalFileHash(file);
  });
  t.mock.method(childProcess, "spawnSync", /** @param {string} file @param {string[]} args */ (file, args) => {
    if (file === "/usr/bin/git" && args[1] === "show") {
      const spec = args[2] || "", relative = spec.slice(spec.indexOf(":") + 1), bytes = relative === c.source.definition.path ? definition : license;
      return {error: null, signal: null, status: 0, stdout: bytes, stderr: Buffer.alloc(0)};
    }
    if (file === "/usr/bin/git" && args[1] === "rev-parse") {
      const spec = args[args.length - 1] || "", relative = spec.slice(spec.indexOf(":") + 1), blob = relative === c.source.definition.path ? c.source.definition.blob : c.source.license.blob;
      return {error: null, signal: null, status: 0, stdout: blob + "\n", stderr: ""};
    }
    if (file === f.llvmLib) {
      const defArg = args.find(argument => argument.startsWith("/def:")), outArg = args.find(argument => argument.startsWith("/out:"));
      assert.ok(defArg); assert.ok(outArg);
      assert.ok(fs.readFileSync(defArg.slice(5)).equals(Buffer.concat([Buffer.from("LIBRARY " + c.library.dllName + "\n"), definition])));
      fs.writeFileSync(outArg.slice(5), fakeLibrary, {flag: "wx", mode: 0o600});
      if (options.mutateTool) {fs.writeFileSync(f.llvmAr, "drifted llvm-ar fixture\n"); fs.chmodSync(f.llvmAr, 0o755);}
      return {error: null, signal: null, status: 0, stdout: "", stderr: ""};
    }
    throw new Error("Unexpected command in local AMD fixture: " + file);
  });
  return {c, fakeLibrary, verifyLlvm, get sourceVerifications() {return sourceVerifications;}, get llvmAdmissions() {return llvmAdmissions;}};
}

describe("Source-generated Windows AMD import-library input", {skip: process.platform !== "linux", concurrency: false}, () => {
  test("catalog ties source identity to the maintained CLR source pins", () => {
    const c = windowsAmdhipCatalog(), clr = sourcePins.gitSourceComponents("rocm")["clr"];
    assert.ok(clr);
    assert.equal(c.source.origin, sourcePins.origins.clr);
    assert.equal(c.source.revision, clr.revision);
    assert.match(c.source.definition.sha256, /^[a-f0-9]{64}$/);
    assert.match(c.source.license.sha256, /^[a-f0-9]{64}$/);
    assert.match(c.library.sha256, /^[a-f0-9]{64}$/);
    assert.equal(c.source.definition.derivedBytes,
      c.source.definition.bytes + Buffer.byteLength("LIBRARY " + c.library.dllName + "\n"));
    assert.ok(Object.isFrozen(c) && Object.isFrozen(c.source) && Object.isFrozen(c.library));
  });

  test("derives the pinned definition with one DLL directive and no source edits", () => {
    const c = windowsAmdhipCatalog(), source = Buffer.from("EXPORTS\n hipInit\n hipGetDevice\n"),
      prefix = Buffer.from("LIBRARY " + c.library.dllName + "\n"), derived = deriveDefinition(source, c.library.dllName);
    assert.deepEqual(derived.subarray(0, prefix.length), prefix);
    assert.ok(derived.subarray(prefix.length).equals(source));
    assert.throws(() => deriveDefinition(Buffer.from("EXPORTS\0bad"), c.library.dllName), /invalid or excessive/);
    assert.throws(() => deriveDefinition(Buffer.alloc(64 << 10), c.library.dllName), /invalid or excessive/);
  });

  test("rejects wrong roots and missing public source state before creating output", async t => {
    const root = fixtureRoot(t), suffix = "99999999999999999999-4", env = {
        MOM_NATIVE_CI_SUFFIX: suffix, RUNNER_TEMP: root,
        MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2", MOM_NATIVE_ROCM_SOURCE_DIR: "/tmp/mom-native-rocm-prerequisites-" + suffix,
      }, source = env.MOM_NATIVE_ROCM_SOURCE_DIR, bundle = path.join(root, "native-linux-llvm-tools-" + suffix), output = path.join(root, "native-windows-amdhip-" + suffix),
      pin = "f".repeat(64);
    await assert.rejects(prepareWindowsAmdhip(source + "-wrong", bundle, pin, output, env), /original ROCm source/);
    await assert.rejects(prepareWindowsAmdhip(source, bundle, "not-a-hash", output, env), /externally pinned/);
    await assert.rejects(prepareWindowsAmdhip(source, bundle, pin, output, env));
    assert.equal(fs.existsSync(output), false);
  });

  test("removes the temporary generated definition before sealing and admitting the handoff", async t => {
    const root = fixtureRoot(t), c = windowsAmdhipCatalog(), suffix = "99999999999999999998-5", runnerTemp = path.join(root, "runner-temp"),
      sourceRoot = "/tmp/mom-native-rocm-prerequisites-" + suffix, output = path.join(runnerTemp, "native-windows-amdhip-" + suffix),
      bundle = path.join(runnerTemp, "tool-bundle"), stateDir = path.join(root, "source-state"), fetchReceipt = path.join(stateDir, "fetch.json"),
      rootsFile = path.join(stateDir, "roots.json"), definition = fs.readFileSync(path.join(__dirname, "fixtures/native-windows-amdhip/amdhip.def")),
      license = fs.readFileSync(path.join(__dirname, "fixtures/native-windows-amdhip/LICENSE.md")), fakeLibrary = Buffer.alloc(c.library.bytes, 0x41),
      bin = path.join(bundle, "bin"), toolRoot = bin, targetLeaf = c.toolchain.llvmLibTarget,
      handoff = {toolRoot, files: {[c.toolchain.llvmLibAlias]: {type: "symlink", target: targetLeaf},
        ["bin/" + targetLeaf]: {type: "file", bytes: 1234, sha256: "1".repeat(64)}}},
      env = {MOM_NATIVE_CI_SUFFIX: suffix, RUNNER_TEMP: runnerTemp,
        MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: "2", MOM_NATIVE_ROCM_SOURCE_DIR: sourceRoot};
    fs.mkdirSync(runnerTemp); fs.mkdirSync(stateDir); fs.mkdirSync(bin, {recursive: true});
    writePrivate(fetchReceipt, "pinned mocked source acquisition receipt\n"); writePrivate(rootsFile, "pinned mocked root inventory\n");
    assert.equal(definition.length, c.source.definition.bytes); assert.equal(sha256(definition), c.source.definition.sha256);
    assert.equal(license.length, c.source.license.bytes); assert.equal(sha256(license), c.source.license.sha256);

    let generatedDefinition = "", generatedLibrary = "", sourceVerifications = 0;
    t.mock.method(ci, "ciPaths", () => ({suffix, runnerTemp}));
    t.mock.method(ci, "rocmSourceStatePaths", /** @param {{suffix:string,runnerTemp:string}} _paths */ _paths => ({source: sourceRoot, rootsFile, fetchReceipt}));
    t.mock.method(ci, "verifyRocmSources", () => {sourceVerifications++;});
    t.mock.method(llvmTools, "linuxLlvmToolHandoffInputs", () => handoff);
    const originalFileHash = sourceIO.fileHash;
    t.mock.method(sourceIO, "fileHash", /** @param {string} file */ file => {
      if (generatedLibrary && (file === generatedLibrary || file === path.join(output, c.library.fileName))) {
        const bytes = fs.readFileSync(file);
        return bytes.equals(fakeLibrary) ? c.library.sha256 : sha256(bytes);
      }
      return originalFileHash(file);
    });
    t.mock.method(childProcess, "spawnSync", /** @param {string} file @param {string[]} args */ (file, args) => {
      if (file === "/usr/bin/git" && args[1] === "show") {
        const spec = args[2] || "", revisionPath = spec.slice(spec.indexOf(":") + 1);
        if (revisionPath === c.source.definition.path) {return {error: null, signal: null, status: 0, stdout: definition, stderr: Buffer.alloc(0)};}
        if (revisionPath === c.source.license.path) {return {error: null, signal: null, status: 0, stdout: license, stderr: Buffer.alloc(0)};}
        throw new Error("Unexpected mocked Git blob path: " + revisionPath);
      }
      if (file === "/usr/bin/git" && args[1] === "rev-parse") {
        const spec = args[args.length - 1] || "", revisionPath = spec.slice(spec.indexOf(":") + 1);
        const blob = revisionPath === c.source.definition.path ? c.source.definition.blob : c.source.license.blob;
        return {error: null, signal: null, status: 0, stdout: blob + "\n", stderr: ""};
      }
      if (path.basename(file) === "llvm-lib") {
        const defArgument = args.find(/** @param {string} argument */ argument => argument.startsWith("/def:")),
          outArgument = args.find(/** @param {string} argument */ argument => argument.startsWith("/out:"));
        assert.ok(defArgument); assert.ok(outArgument); generatedDefinition = defArgument.slice(5); generatedLibrary = outArgument.slice(5);
        const derived = fs.readFileSync(generatedDefinition), expected = Buffer.concat([Buffer.from("LIBRARY " + c.library.dllName + "\n"), definition]);
        assert.ok(derived.equals(expected)); assert.equal(sha256(derived), c.source.definition.derivedSha256);
        fs.writeFileSync(generatedLibrary, fakeLibrary, {flag: "wx", mode: 0o600});
        return {error: null, signal: null, status: 0, stdout: "", stderr: ""};
      }
      throw new Error("Unexpected external process in staged producer test: " + file);
    });

    const receipt = await prepareWindowsAmdhip(sourceRoot, bundle, "a".repeat(64), output, env);
    assert.equal(sourceVerifications, 2);
    assert.ok(generatedDefinition); assert.equal(fs.existsSync(generatedDefinition), false);
    assert.deepEqual(fs.readdirSync(output).sort(), [path.posix.basename(c.source.license.path), c.library.fileName, "sdk.json"].sort());
    assert.equal(fs.existsSync(path.join(output, "amdhip.def")), false);
    assert.equal(receipt.library.sha256, c.library.sha256);
    assert.equal(admitWindowsAmdhip(output).status, "PASS");
  });

  test("generates a local receipt from the admitted source and LLVM tool without CI identity", async t => {
    const f = localWorkspaceFixture(t), fixture = mockLocalAmdhipBuild(t, f),
      receipt = await prepareLocalAmdhipWindows(f.input, {MOM_AOT_BUILD_ONLY: "1"}, {verifyLlvm: fixture.verifyLlvm}), admitted = windowsAmdhipInputs(f.output);
    assert.equal(receipt.localBuildId, f.localBuildId); assert.equal(Object.hasOwn(receipt, "runAttempt"), false);
    assert.equal(receipt.source.sourceRoot, f.sourceRoot); assert.equal(receipt.toolchain.sdkOutput, f.llvmOutput);
    assert.equal(fixture.sourceVerifications, 2); assert.equal(fixture.llvmAdmissions, 2);
    assert.equal(admitted.amdhip64, path.join(f.output, fixture.c.library.fileName));
    assert.deepEqual(fs.readdirSync(f.output).sort(), [path.basename(fixture.c.source.license.path), fixture.c.library.fileName, "sdk.json"].sort());
  });

  test("rejects a malformed local identity and preserves existing unknown output", async t => {
    const f = localWorkspaceFixture(t), env = {MOM_AOT_BUILD_ONLY: "1"};
    await assert.rejects(prepareLocalAmdhipWindows({...f.input, localBuildId: "42-1"}, env), /UUID localBuildId/);
    assert.equal(fs.existsSync(f.output), false);
    fs.mkdirSync(f.output); writePrivate(path.join(f.output, "keep.bin"), "preserve\n");
    await assert.rejects(prepareLocalAmdhipWindows(f.input, env), /must be new/);
    assert.equal(fs.readFileSync(path.join(f.output, "keep.bin"), "utf8"), "preserve\n");
    assert.deepEqual(fs.readdirSync(f.output), ["keep.bin"]);
  });

  test("rejects Linux LLVM tool drift before publishing local output", async t => {
    const f = localWorkspaceFixture(t), fixture = mockLocalAmdhipBuild(t, f, {mutateTool: true});
    await assert.rejects(prepareLocalAmdhipWindows(f.input, {MOM_AOT_BUILD_ONLY: "1"}, {verifyLlvm: fixture.verifyLlvm}), /changed during generation/);
    assert.equal(fixture.sourceVerifications, 2); assert.equal(fixture.llvmAdmissions, 2);
    assert.equal(fs.existsSync(f.output), false);
    assert.equal(fs.readdirSync(f.workspace).some(name => name.startsWith(".native-windows-amdhip-local-")), false);
  });

  test("admits only an externally pinned same-run thin LLVM tool handoff", t => {
    const root = fixtureRoot(t), bundle = path.join(root, "bundle"), bin = path.join(bundle, "bin"), receipts = path.join(bundle, "receipts"), c = windowsAmdhipCatalog();
    fs.mkdirSync(bin, {recursive: true}); fs.mkdirSync(receipts);
    const aliasLeaf = path.posix.basename(c.toolchain.llvmLibAlias), targetLeaf = c.toolchain.llvmLibTarget,
      archiveName = path.basename(llvmSdk.archiveName), ar = path.join(bin, targetLeaf), alias = path.join(bin, aliasLeaf), sourceArchive = path.join(bundle, archiveName);
    writePrivate(ar, "admitted dispatcher"); writePrivate(sourceArchive, "source archive");
    fs.chmodSync(ar, 0o555); fs.symlinkSync(targetLeaf, alias);
    writePrivate(path.join(receipts, "result.json"), JSON.stringify({role: c.toolchain.llvmRole, stage: "audited", buildOnly: true,
      compiled: true, executed: false, publicationQualified: false, plan: {output: "/tmp/sdk-output"}, sdk: {path: "/tmp/sdk-output/sdk"}}));
    const input = {toolRoot: bin, lldLink: path.join(bin, "lld-link"), llvmNm: path.join(bin, "llvm-nm"), sourceArchive,
      files: {[c.toolchain.llvmLibAlias]: {type: "symlink", target: targetLeaf},
        ["bin/" + targetLeaf]: {type: "file", bytes: Buffer.byteLength("admitted dispatcher"), sha256: sha256("admitted dispatcher")},
        [archiveName]: {type: "file", bytes: Buffer.byteLength("source archive"), sha256: sha256("source archive")},
        "receipts/result.json": {type: "file", bytes: fs.statSync(path.join(receipts, "result.json")).size,
          sha256: sourceIO.fileHash(path.join(receipts, "result.json"))}}};
    t.mock.method(llvmTools, "linuxLlvmToolHandoffInputs", () => input);
    const selected = verifyLinuxToolHandoff(bundle, "0".repeat(64));
    assert.equal(selected.llvmLib, alias);
    const aliasProof = input.files[c.toolchain.llvmLibAlias];
    assert.ok(aliasProof);
    aliasProof.target = "other-ar";
    assert.throws(() => verifyLinuxToolHandoff(bundle, "0".repeat(64)), /stale or does not contain/);
  });

  test("admits a sealed generated payload, returns its inputs and allows relocation", t => {
    const root = fixtureRoot(t), c = windowsAmdhipCatalog(), licenseName = path.posix.basename(c.source.license.path), fixture = receiptFixture(root,
      Buffer.alloc(c.library.bytes, 0x41), Buffer.alloc(c.source.license.bytes, 0x42));
    mockFixtureHashes(t, fixture);
    const admitted = admitWindowsAmdhip(fixture.output), first = windowsAmdhipInputs(fixture.output);
    assert.equal(admitted.vendorPrebuilt, false); assert.equal(admitted.sourceGenerated, true);
    assert.equal(first.amdhip64, fixture.libraryPath); assert.equal(first.files[first.amdhip64], c.library.sha256);
    assert.equal(first.files[fixture.licensePath], c.source.license.sha256);
    const moved = path.join(root, "moved"); fs.mkdirSync(moved, {mode: 0o700});
    for (const name of [c.library.fileName, licenseName, "sdk.json"]) {
      fs.copyFileSync(path.join(fixture.output, name), path.join(moved, name), fs.constants.COPYFILE_EXCL); fs.chmodSync(path.join(moved, name), 0o444);
    }
    fs.chmodSync(moved, 0o555);
    assert.equal(windowsAmdhipInputs(moved).amdhip64, path.join(moved, c.library.fileName));
  });

  test("rejects a same-size modified import library after relocation", t => {
    const root = fixtureRoot(t), c = windowsAmdhipCatalog(), fixture = receiptFixture(root,
      Buffer.alloc(c.library.bytes, 0x41), Buffer.alloc(c.source.license.bytes, 0x42));
    mockFixtureHashes(t, fixture); fs.chmodSync(fixture.output, 0o700); fs.chmodSync(fixture.libraryPath, 0o600);
    const altered = Buffer.alloc(c.library.bytes, 0x43); fs.writeFileSync(fixture.libraryPath, altered); fs.chmodSync(fixture.libraryPath, 0o444); fs.chmodSync(fixture.output, 0o555);
    assert.throws(() => admitWindowsAmdhip(fixture.output), /differs from its pinned bytes/);
  });

  test("rejects changed original license evidence", t => {
    const root = fixtureRoot(t), c = windowsAmdhipCatalog(), fixture = receiptFixture(root,
      Buffer.alloc(c.library.bytes, 0x41), Buffer.alloc(c.source.license.bytes, 0x42));
    mockFixtureHashes(t, fixture); fs.chmodSync(fixture.output, 0o700); fs.chmodSync(fixture.licensePath, 0o600);
    fs.writeFileSync(fixture.licensePath, Buffer.alloc(c.source.license.bytes, 0x43)); fs.chmodSync(fixture.licensePath, 0o444); fs.chmodSync(fixture.output, 0o555);
    assert.throws(() => admitWindowsAmdhip(fixture.output), /differs from its pinned bytes/);
  });

  test("rejects extra files and malformed or stale producer receipts", t => {
    const root = fixtureRoot(t), c = windowsAmdhipCatalog(), fixture = receiptFixture(root,
      Buffer.alloc(c.library.bytes, 0x41), Buffer.alloc(c.source.license.bytes, 0x42));
    mockFixtureHashes(t, fixture); fs.chmodSync(fixture.output, 0o700); writePrivate(path.join(fixture.output, "extra.lib"), "extra");
    fs.chmodSync(fixture.output, 0o555);
    assert.throws(() => admitWindowsAmdhip(fixture.output), /invalid or writable top-level layout/);
    const stale = path.join(root, "stale"); fs.mkdirSync(stale, {mode: 0o700});
    for (const name of [c.library.fileName, path.posix.basename(c.source.license.path)]) {
      fs.copyFileSync(path.join(fixture.output, name), path.join(stale, name)); fs.chmodSync(path.join(stale, name), 0o444);
    }
    const receipt = {...fixture.receipt, recipe: {...fixture.receipt.recipe, files: {}}};
    writePrivate(path.join(stale, "sdk.json"), JSON.stringify(receipt)); fs.chmodSync(path.join(stale, "sdk.json"), 0o444); fs.chmodSync(stale, 0o555);
    assert.throws(() => admitWindowsAmdhip(stale), /receipt is stale, malformed or incomplete/);
  });

  test("rejects linked payloads and hard-linked files", {skip: process.platform === "win32"}, t => {
    const root = fixtureRoot(t), c = windowsAmdhipCatalog(), fixture = receiptFixture(root,
      Buffer.alloc(c.library.bytes, 0x41), Buffer.alloc(c.source.license.bytes, 0x42));
    mockFixtureHashes(t, fixture); fs.chmodSync(fixture.output, 0o700);
    const original = path.join(root, "amdhip64.original"); fs.renameSync(fixture.libraryPath, original); fs.symlinkSync(original, fixture.libraryPath); fs.chmodSync(fixture.output, 0o555);
    assert.throws(() => admitWindowsAmdhip(fixture.output), /bounded canonical regular file/);
    fs.chmodSync(fixture.output, 0o700); fs.unlinkSync(fixture.libraryPath); fs.renameSync(original, fixture.libraryPath);
    fs.linkSync(fixture.libraryPath, path.join(root, "hardlink")); fs.chmodSync(fixture.output, 0o555);
    assert.throws(() => admitWindowsAmdhip(fixture.output), /bounded canonical regular file/);
  });
});
