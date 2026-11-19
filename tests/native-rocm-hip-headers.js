"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"),
  {createHash} = require("node:crypto");
const {exportRocmHipHeaders, rocmHipHeaderInputs} = require("../scripts/native-rocm-hip-headers");
const nativeCi = require("../scripts/native-ci-tar");
const {gitSourceComponents} = require("../scripts/fetch-native-sdk-sources");
const {sdkSourceFiles} = require("../scripts/prepare-native-sdk-sources");

const repo = path.resolve(__dirname, ".."), patchSource = path.join(repo, "scripts/patches/rocm-hip-headers-only.patch");
let serial = 0;

/** @param {string} file @returns {string} */
function hash(file) {return createHash("sha256").update(fs.readFileSync(file)).digest("hex");}

/** @param {string} file @param {string} contents */
function write(file, contents) {fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, contents, {flag: "wx"});}

/** @param {ReturnType<typeof fixture>} f */
function rewritePackage(f) {fs.writeFileSync(f.packagePath, JSON.stringify(f.package, null, 2) + "\n");}

/** @param {Record<string,string>} files */
function mapSha(files) {return createHash("sha256").update(JSON.stringify(files)).digest("hex");}

/** @param {import("node:test").TestContext} t @param {boolean} [duplicateRequired=false] */
function fixture(t, duplicateRequired = false) {
  const suffix = "ci-" + String(Date.now()).slice(-8) + String(process.pid).slice(-4) + String(++serial) + "-88",
    sourceRoot = path.join("/tmp", "mom-native-rocm-hip-headers-" + suffix),
    sourceCohortRoot = path.join("/tmp", "mom-native-rocm-prerequisites-" + suffix),
    consumerRoot = path.join("/tmp", "native-rocm-hip-headers-" + suffix),
    outputParent = fs.mkdtempSync(path.join("/tmp", "mom-rocm-hip-headers-test-")),
    outputRoot = path.join(outputParent, path.basename(consumerRoot)), archivePath = outputParent + ".tar", pins = gitSourceComponents("rocm"),
    hipPin = pins["hip"], clrPin = pins["clr"];
  assert.ok(hipPin && clrPin);
  const hipRevision = hipPin.revision, clrRevision = clrPin.revision;
  for (const root of [sourceRoot, sourceCohortRoot, consumerRoot]) {assert.equal(fs.existsSync(root), false, "fixture root collision");}
  fs.mkdirSync(path.join(sourceRoot, "source/hip-common"), {recursive: true});
  fs.mkdirSync(path.join(sourceRoot, "source/clr-hipamd/src/hiprtc"), {recursive: true});
  fs.mkdirSync(path.join(sourceRoot, "source/roots"), {recursive: true});
  fs.mkdirSync(path.join(sourceCohortRoot, "roots/clr"), {recursive: true});
  fs.mkdirSync(path.join(sourceRoot, "build/include/hip"), {recursive: true});
  fs.mkdirSync(path.join(sourceRoot, "patches"), {recursive: true});
  fs.mkdirSync(path.join(sourceRoot, "include/hip"), {recursive: true});
  const requiredFiles = Array.from({length: 16}, (_, index) => "hip/required" + String(index).padStart(2, "0") + ".h"),
    required = duplicateRequired ? [...requiredFiles, ...requiredFiles.slice(0, 1)] : requiredFiles,
    cmake = "set(HIPRTC_HEADERS\n" + required.map(name => String.fromCharCode(36) + "{HIP_COMMON_INCLUDE_DIR}/" + name).join("\n") + "\n)\n",
    cmakeFile = path.join(sourceRoot, "source/clr-hipamd/src/hiprtc/CMakeLists.txt");
  write(cmakeFile, cmake);
  write(path.join(sourceRoot, "source/clr-hipamd/CMakeLists.txt"), "patched HIP CMake fixture\n");
  for (const name of ["hip/hip_runtime_api.h", "hip/hiprtc.h", "hip/amd_detail/amd_fixture.h", ...requiredFiles]) {
    write(path.join(sourceRoot, "include", name), "/* " + name + " */\n");
  }
  for (const name of ["hip/hip_runtime_api.h", "hip/hiprtc.h", ...requiredFiles]) {
    write(path.join(sourceRoot, "source/hip-common/include", name), fs.readFileSync(path.join(sourceRoot, "include", name), "utf8"));
  }
  write(path.join(sourceRoot, "source/clr-hipamd/include/hip/amd_detail/amd_fixture.h"),
    fs.readFileSync(path.join(sourceRoot, "include/hip/amd_detail/amd_fixture.h"), "utf8"));
  const generated = path.join(sourceRoot, "build/include/hip/hip_version.h");
  write(generated, "#define HIP_VERSION_MAJOR 7\n");
  write(path.join(sourceRoot, "include/hip/hip_version.h"), fs.readFileSync(generated, "utf8"));
  const patchCopy = path.join(sourceRoot, "patches/rocm-hip-headers-only.patch"); fs.copyFileSync(patchSource, patchCopy);
  const patchSha = hash(patchSource), sourceFiles = sdkSourceFiles(path.join(sourceRoot, "include"));
  const patchTarget = path.join(sourceRoot, "source/clr-hipamd/CMakeLists.txt"), patchTargetSha = hash(patchTarget),
    commonMap = sdkSourceFiles(path.join(sourceRoot, "source/hip-common")), clrMap = sdkSourceFiles(path.join(sourceRoot, "source/clr-hipamd")),
    checkCommand = ["/usr/bin/git", "-C", path.join(sourceRoot, "source/clr-hipamd"), "apply", "-p2", "--check", "--", patchCopy],
    applyCommand = ["/usr/bin/git", "-C", path.join(sourceRoot, "source/clr-hipamd"), "apply", "-p2", "--", patchCopy];
  write(path.join(sourceRoot, "patches/check.json"), JSON.stringify({command: checkCommand, status: 0, signal: null, error: null, stdout: "", stderr: ""}) + "\n");
  write(path.join(sourceRoot, "patches/apply.json"), JSON.stringify({command: applyCommand, status: 0, signal: null, error: null, stdout: "", stderr: ""}) + "\n");
  write(path.join(sourceRoot, "cmake-configure.log"), "configure succeeded\n");
  const prefix = path.join(sourceRoot, "build/install-prefix"), command = ["/usr/bin/cmake", "-S", path.join(sourceRoot, "source/clr-hipamd"),
      "-B", path.join(sourceRoot, "build"), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_INSTALL_PREFIX=" + prefix,
      "-DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.2/bin/icx", "-DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.2/bin/icpx",
      "-DHIP_COMMON_DIR=" + path.join(sourceRoot, "source/hip-common"), "-DHIP_PLATFORM:STRING=amd", "-DHIP_RUNTIME:STRING=none",
      "-DHIPCC_BIN_DIR:STRING=", "-DHIP_OFFICIAL_BUILD:BOOL=ON", "-DROCM_PATH=" + prefix],
    pkg = {version: 1, pins: {hip: hipRevision, clr: clrRevision}, sourceExports: {
      hipCommon: {root: "source/hip-common", revision: hipRevision, fileCount: Object.keys(commonMap).length,
        filesSha256: mapSha(commonMap), treeSha256: "2".repeat(64)},
      clrHipamd: {root: "source/clr-hipamd", revision: clrRevision, treeish: clrRevision + ":hipamd", fileCount: Object.keys(clrMap).length,
        filesSha256: "3".repeat(64), treeSha256: "4".repeat(64), afterPatch: {fileCount: Object.keys(clrMap).length, filesSha256: mapSha(clrMap)}}},
    patch: {path: "patches/rocm-hip-headers-only.patch", sha256: patchSha, stripComponents: 2,
      target: "source/clr-hipamd/CMakeLists.txt", originalSha256: "6".repeat(64), patchedSha256: patchTargetSha,
      checkLog: "patches/check.json", applyLog: "patches/apply.json"},
    configure: {command, environment: {GIT_DIR: path.join(sourceCohortRoot, "roots/clr")}, configureOnly: true,
      log: "cmake-configure.log", logSha256: hash(path.join(sourceRoot, "cmake-configure.log")), status: 0},
    hiprtcCmake: {path: "source/clr-hipamd/src/hiprtc/CMakeLists.txt", sha256: hash(cmakeFile)},
    generatedHeader: {path: "build/include/hip/hip_version.h", sha256: hash(generated)}, files: sourceFiles,
    requiredHiprtcHeaders: required, buildOnly: true, publicationQualified: false};
  const packagePath = path.join(sourceRoot, "header-package.json"); write(packagePath, JSON.stringify(pkg, null, 2) + "\n");
  t.after(() => {
    fs.rmSync(archivePath, {force: true});
    for (const root of [sourceRoot, sourceCohortRoot, consumerRoot, outputParent]) {fs.rmSync(root, {recursive: true, force: true});}
  });
  return {suffix, sourceRoot, sourceCohortRoot, consumerRoot, outputRoot, outputParent, archivePath, packagePath, package: pkg, required, sourceFiles};
}

/** @param {ReturnType<typeof fixture>} f */
function exportAndRestore(f) {
  const exported = exportRocmHipHeaders(f.sourceRoot, f.outputRoot, f.suffix);
  const rootName = path.basename(f.outputRoot);
  nativeCi.packState(f.archivePath, f.outputParent, [rootName]);
  fs.rmSync(f.outputRoot, {recursive: true});
  nativeCi.extractState(f.archivePath, "/tmp", [rootName]);
  return {exported, admitted: rocmHipHeaderInputs(f.consumerRoot, {handoffSha256: exported.handoffSha256})};
}

test.describe("ROCm HIP header handoff", {skip: process.platform !== "linux" && "Linux cross-build header transport"}, () => {
  test("exports original manifest and headers byte-for-byte and admits the same run", t => {
    const f = fixture(t), {exported, admitted} = exportAndRestore(f);
    assert.equal(admitted.include, path.join(f.consumerRoot, "include"));
    assert.equal(admitted.manifestSha256, hash(path.join(f.consumerRoot, "header-package.json")));
    assert.equal(admitted.handoffSha256, exported.handoffSha256);
    assert.equal(admitted.sourceRoot, f.sourceRoot);
    assert.equal(admitted.fileCount, Object.keys(f.sourceFiles).length);
    for (const [name, sha256] of Object.entries(f.sourceFiles)) {
      assert.equal(admitted.files[path.join(admitted.include, ...name.split("/"))], sha256);
    }
    assert.equal(admitted.files[path.join(f.consumerRoot, "header-package.json")], admitted.manifestSha256);
    assert.equal(fs.readFileSync(path.join(f.consumerRoot, "header-package.json"), "utf8"), fs.readFileSync(f.packagePath, "utf8"));
  });

  test("rejects writing the state tar inside a selected input before creating it", t => {
    const f = fixture(t), rootName = path.basename(f.outputRoot);
    exportRocmHipHeaders(f.sourceRoot, f.outputRoot, f.suffix);
    const archiveInsideInput = path.join(f.outputRoot, "inside.tar");
    assert.throws(() => nativeCi.packState(archiveInsideInput, f.outputParent, [rootName]), /outside its inputs/);
    assert.equal(fs.existsSync(archiveInsideInput), false);
  });

  test("preserves a repeated CMake-required header without duplicating its payload file", t => {
    const f = fixture(t, true), {exported, admitted} = exportAndRestore(f);
    assert.equal(f.required.length, 17);
    assert.equal(new Set(f.required).size, 16);
    assert.equal(f.required[f.required.length - 1], f.required[0]);
    assert.equal(admitted.fileCount, Object.keys(f.sourceFiles).length);
    assert.equal(Object.keys(admitted.files).length, Object.keys(f.sourceFiles).length + 1);
    assert.equal(admitted.handoffSha256, exported.handoffSha256);
  });

  test("rejects an unpinned handoff or a noncanonical consumer root", t => {
    const f = fixture(t), {exported} = exportAndRestore(f);
    assert.throws(() => rocmHipHeaderInputs(f.consumerRoot, {handoffSha256: "0".repeat(64)}), /external hash pin/);
    assert.throws(() => rocmHipHeaderInputs(f.outputParent, {handoffSha256: exported.handoffSha256}), /canonical run path/);
  });

  test("rejects mismatched upstream revisions and changed HIPRTC CMake source before copying", t => {
    const f = fixture(t); f.package.pins.hip = "0".repeat(40);
    fs.writeFileSync(f.packagePath, JSON.stringify(f.package));
    assert.throws(() => exportRocmHipHeaders(f.sourceRoot, f.outputRoot, f.suffix), /pinned build-only/);
    const other = fixture(t), cmake = path.join(other.sourceRoot, "source/clr-hipamd/src/hiprtc/CMakeLists.txt");
    fs.appendFileSync(cmake, "# stale\n");
    assert.throws(() => exportRocmHipHeaders(other.sourceRoot, other.outputRoot, other.suffix), /source exports/);
  });

  test("rejects a changed header, package receipt, and unlisted extra payload", t => {
    const changed = fixture(t), {exported: changedExport} = exportAndRestore(changed), header = path.join(changed.consumerRoot, "include/hip/hiprtc.h");
    fs.appendFileSync(header, "changed\n");
    assert.throws(() => rocmHipHeaderInputs(changed.consumerRoot, {handoffSha256: changedExport.handoffSha256}), /package|payload/);
    const manifest = fixture(t), {exported: manifestExport} = exportAndRestore(manifest);
    fs.appendFileSync(path.join(manifest.consumerRoot, "header-package.json"), " ");
    assert.throws(() => rocmHipHeaderInputs(manifest.consumerRoot, {handoffSha256: manifestExport.handoffSha256}), /external hash pin|package/);
    const extra = fixture(t), {exported: extraExport} = exportAndRestore(extra);
    write(path.join(extra.consumerRoot, "include/hip/unlisted.h"), "extra\n");
    assert.throws(() => rocmHipHeaderInputs(extra.consumerRoot, {handoffSha256: extraExport.handoffSha256}), /manifest/);
  });

  test("rejects rewritten manifests even when a caller supplies their new digest", t => {
    const f = fixture(t), {exported} = exportAndRestore(f), handoffFile = path.join(f.consumerRoot, "handoff.json"),
      handoff = JSON.parse(fs.readFileSync(handoffFile, "utf8"));
    handoff.recipe.graph["scripts/native-rocm-hip-headers.js"] = "0".repeat(64);
    fs.chmodSync(handoffFile, 0o644); fs.writeFileSync(handoffFile, JSON.stringify(handoff, null, 2) + "\n");
    assert.notEqual(hash(handoffFile), exported.handoffSha256);
    assert.throws(() => rocmHipHeaderInputs(f.consumerRoot, {handoffSha256: hash(handoffFile)}), /recipe/);
  });

  test("rejects a top-level payload outside the exact handoff root", t => {
    const f = fixture(t), {exported} = exportAndRestore(f);
    write(path.join(f.consumerRoot, "unexpected.json"), "{}\n");
    assert.throws(() => rocmHipHeaderInputs(f.consumerRoot, {handoffSha256: exported.handoffSha256}), /top-level members/);
    const linked = fixture(t), {exported: linkedExport} = exportAndRestore(linked);
    try {
      fs.symlinkSync(path.join(linked.consumerRoot, "header-package.json"), path.join(linked.consumerRoot, "unexpected-link"));
    } catch {
      t.skip("symbolic links are unavailable on this filesystem");
      return;
    }
    assert.throws(() => rocmHipHeaderInputs(linked.consumerRoot, {handoffSha256: linkedExport.handoffSha256}), /top-level members/);
  });

  test("rejects a configure receipt that did not succeed", t => {
    const f = fixture(t);
    f.package.configure.status = 1;
    rewritePackage(f);
    assert.throws(() => exportRocmHipHeaders(f.sourceRoot, f.outputRoot, f.suffix), /successful configure-only/);
  });

  test("requires clean process evidence for a status-zero patch check", t => {
    const f = fixture(t), log = path.join(f.sourceRoot, "patches/check.json"), valid = JSON.parse(fs.readFileSync(log, "utf8")),
      invalidRows = [{signal: "SIGTERM"}, {error: "spawn failed"}, {stdout: 0}, {stderr: null}];
    for (const invalid of invalidRows) {
      fs.writeFileSync(log, JSON.stringify({...valid, ...invalid}) + "\n");
      assert.throws(() => exportRocmHipHeaders(f.sourceRoot, f.outputRoot, f.suffix), /patch receipt did not pass/);
    }
  });

  test("rejects a patch pin for different source bytes", t => {
    const f = fixture(t);
    f.package.patch.sha256 = "0".repeat(64);
    rewritePackage(f);
    assert.throws(() => exportRocmHipHeaders(f.sourceRoot, f.outputRoot, f.suffix), /reviewed source patch/);
  });

  test("rejects duplicate CMake overrides instead of accepting a later runtime value", t => {
    const f = fixture(t);
    f.package.configure.command.push("-DHIP_RUNTIME:STRING=hip");
    rewritePackage(f);
    assert.throws(() => exportRocmHipHeaders(f.sourceRoot, f.outputRoot, f.suffix), /configure command differs/);
  });

  test("rejects a package that omits a HIPRTC-required header", t => {
    const f = fixture(t), requiredHeader = f.required[0];
    assert.ok(requiredHeader);
    delete f.package.files[requiredHeader];
    rewritePackage(f);
    assert.throws(() => exportRocmHipHeaders(f.sourceRoot, f.outputRoot, f.suffix), /not covered by the package map/);
  });

  test("rejects symbolic links, hard links, excessive file size, and overdeep trees", t => {
    const symlink = fixture(t), {exported: symlinkExport} = exportAndRestore(symlink), target = path.join(symlink.consumerRoot, "include/hip/hip_runtime_api.h"),
      linked = path.join(symlink.consumerRoot, "include/hip/hiprtc.h");
    fs.unlinkSync(linked); fs.symlinkSync(path.basename(target), linked);
    assert.throws(() => rocmHipHeaderInputs(symlink.consumerRoot, {handoffSha256: symlinkExport.handoffSha256}), /symbolic link/);

    const hardlink = fixture(t), {exported: hardlinkExport} = exportAndRestore(hardlink), source = path.join(hardlink.consumerRoot, "include/hip/hip_runtime_api.h"),
      alias = path.join(hardlink.consumerRoot, "include/hip/alias.h");
    try {fs.linkSync(source, alias);} catch {t.skip("hard links are unavailable on this filesystem"); return;}
    assert.throws(() => rocmHipHeaderInputs(hardlink.consumerRoot, {handoffSha256: hardlinkExport.handoffSha256}), /singly-linked/);

    const oversized = fixture(t), {exported: oversizedExport} = exportAndRestore(oversized), large = path.join(oversized.consumerRoot, "include/hip/hiprtc.h");
    fs.truncateSync(large, 8 * 1024 * 1024 + 1);
    assert.throws(() => rocmHipHeaderInputs(oversized.consumerRoot, {handoffSha256: oversizedExport.handoffSha256}), /bounded/);

    const deep = fixture(t), {exported: deepExport} = exportAndRestore(deep); let nested = path.join(deep.consumerRoot, "include");
    for (let index = 0; index < 17; index++) {nested = path.join(nested, "d" + index);}
    write(path.join(nested, "too-deep.h"), "x\n");
    assert.throws(() => rocmHipHeaderInputs(deep.consumerRoot, {handoffSha256: deepExport.handoffSha256}), /depth/);
  });
});
