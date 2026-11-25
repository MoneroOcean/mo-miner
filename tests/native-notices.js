"use strict";

const fs = require("node:fs"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict"),
  {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const {assembleNativeNotices, projectNativeArchiveNotices, readNativeNoticeReceipt} = require("../scripts/prepare-native-notices");
const {fileHash, readBuildJson} = require("../scripts/build-source-inputs");
const {tempDir} = require("./common/temp-dir");

/** @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = tempDir(t, "mom-native-notices-");
  const openmpBody = Buffer.from("The LLVM Project license header.\n\nOpenMP Runtime Team\nCopyright (c) 1997-2019 Intel Corporation\n\nSoftware Grant License Agreement (\"Agreement\")\nFull preserved text.\n", "utf8");
  const acppBody = Buffer.from("\nCopyright The AdaptiveCpp Contributors\nAll rights reserved.\n", "utf8");
  const openmp = path.join(root, "openmp-license.txt"), acpp = path.join(root, "acpp-license.txt");
  fs.writeFileSync(openmp, openmpBody, {mode: 0o644}); fs.writeFileSync(acpp, acppBody, {mode: 0o644});
  return {root, openmp, acpp, openmpBody, acppBody,
    inputs: [{component: "LLVM OpenMP runtime", path: openmp, sha256: fileHash(openmp)}, {component: "AdaptiveCpp", path: acpp, sha256: fileHash(acpp)}]};
}

/** @param {string[]} args */
function tarFixture(args) {
  const result = spawnSync("tar", args, {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
  assert.equal(result.error, undefined, result.error?.message);
  assert.equal(result.signal, null, result.stderr);
  assert.equal(result.status, 0, result.stderr);
}

/** @param {import('node:test').TestContext} t */
function archiveFixture(t) {
  const f = fixture(t), source = path.join(f.root, "archive-source"), member = "vendor/LICENSE", archive = path.join(f.root, "source.tar"),
    admission = path.join(f.root, "component-result.json"), licenseBody = Buffer.from("Pinned upstream license body.\nSecond complete line.\n", "utf8");
  fs.mkdirSync(path.join(source, "vendor"), {recursive: true}); fs.writeFileSync(path.join(source, member), licenseBody);
  fs.writeFileSync(admission, JSON.stringify({componentArchiveSha256: "component-output-pin"}) + "\n");
  tarFixture(["-cf", archive, "-C", source, member]);
  return {root: f.root, source, member, archive, admission, licenseBody,
    rows: [{component: "Pinned test component", archive: {path: archive, sha256: fileHash(archive)}, member,
      admissionFiles: {[admission]: fileHash(admission)}}]};
}

test.describe("Maintained native notice assembly", () => {
  test("preserves whole ordered source bodies and stable source fingerprints", t => {
    const f = fixture(t), first = assembleNativeNotices(f.inputs, path.join(f.root, "notice-a")), second = assembleNativeNotices(f.inputs, path.join(f.root, "notice-b"));
    const firstBytes = fs.readFileSync(first.notice.path), secondBytes = fs.readFileSync(second.notice.path);
    assert.deepEqual(firstBytes, secondBytes); assert.equal(first.notice.sha256, fileHash(first.notice.path));
    assert.equal(first.notice.bytes, firstBytes.length); assert.equal(first.fingerprint, second.fingerprint);
    assert.deepEqual(first.inputs.map(row => row.component), ["LLVM OpenMP runtime", "AdaptiveCpp"]);
    assert.ok(firstBytes.indexOf(f.openmpBody) >= 0, "the complete OpenMP-specific license block remains contiguous");
    assert.ok(firstBytes.indexOf(f.acppBody) > firstBytes.indexOf(f.openmpBody), "input ordering is retained");
    assert.equal(first.buildOnly, true); assert.equal(first.publicationQualified, false);
    if (process.platform !== "win32") {assert.equal(fs.statSync(first.notice.path).mode & 0o777, 0o644);}
    const receipt = readBuildJson(path.join(f.root, "notice-a", "native-notices.json"));
    assert.equal(receipt.notice.sha256, first.notice.sha256); assert.equal(receipt.fingerprint, first.fingerprint);
    assert.deepEqual(receipt.inputs, first.inputs);
    assert.deepEqual(fs.readdirSync(path.join(f.root, "notice-a")).sort(), ["THIRD-PARTY-NOTICES.txt", "native-notices.json"]);
    const reversed = assembleNativeNotices([...f.inputs].reverse(), path.join(f.root, "notice-reversed"));
    assert.notEqual(reversed.fingerprint, first.fingerprint); assert.notDeepEqual(fs.readFileSync(reversed.notice.path), firstBytes);
  });

  test("stages ordered literal inputs and re-admits a round-tripped bundle at its canonical destination", t => {
    const f = fixture(t), firstInput = f.inputs[0]; assert.ok(firstInput);
    const inputs = [firstInput, {...firstInput, component: "Second OpenMP consumer"}],
      output = path.join(f.root, "native-release-notices-123-1"), inputDirectory = path.join(output, "inputs"),
      assembled = assembleNativeNotices(inputs, output, {bundleInputs: true}), receiptPath = path.join(output, "native-notices.json"),
      expectedNames = inputs.map((input, index) => String(index).padStart(3, "0") + "-" + input.sha256 + ".txt").sort();
    assert.deepEqual(fs.readdirSync(output).sort(), ["THIRD-PARTY-NOTICES.txt", "inputs", "native-notices.json"]);
    assert.deepEqual(fs.readdirSync(inputDirectory).sort(), expectedNames, "duplicate bodies get distinct ordinal-and-hash paths");
    assert.deepEqual(assembled.inputs.map(row => row.path), expectedNames.map(name => path.join(inputDirectory, name)));
    assert.deepEqual(assembled.inputs.map(row => row.sha256), inputs.map(row => row.sha256));
    assert.deepEqual(assembled.inputs.map(row => row.bytes), inputs.map(() => f.openmpBody.length));
    assert.deepEqual(fs.readFileSync(assembled.inputs[0]?.path || ""), f.openmpBody);
    assert.deepEqual(fs.readFileSync(assembled.inputs[1]?.path || ""), f.openmpBody);

    const bundleTar = path.join(f.root, "notices.tar");
    tarFixture(["-cf", bundleTar, "-C", f.root, path.basename(output)]);
    fs.rmSync(output, {recursive: true, force: true});
    tarFixture(["-xf", bundleTar, "-C", f.root]);
    const receiptHash = fileHash(receiptPath), admitted = readNativeNoticeReceipt(receiptPath, receiptHash);
    assert.equal(admitted.receiptSha256, receiptHash);
    assert.equal(admitted.notice.path, path.join(output, "THIRD-PARTY-NOTICES.txt"));
    assert.deepEqual(admitted.inputs.map(row => row.path), assembled.inputs.map(row => row.path));
    assert.equal(admitted.fingerprint, assembled.fingerprint);

    const relocated = path.join(f.root, "other-destination", path.basename(output));
    fs.mkdirSync(path.dirname(relocated));
    tarFixture(["-xf", bundleTar, "-C", path.dirname(relocated)]);
    assert.throws(() => readNativeNoticeReceipt(path.join(relocated, "native-notices.json"), fileHash(path.join(relocated, "native-notices.json"))),
      /invalid notice binding/, "receipt paths intentionally bind to the canonical absolute destination");
  });

  test("rejects malformed bundle options before output creation", t => {
    const f = fixture(t), output = path.join(f.root, "malformed-options");
    for (const options of [null, [], {bundleInputs: "true"}, {unknown: true}, {bundleInputs: true, unknown: false}]) {
      assert.throws(() => assembleNativeNotices(f.inputs, output, options), /assembly options/);
      assert.equal(fs.existsSync(output), false);
    }
    assert.throws(() => assembleNativeNotices(f.inputs, "relative-output", {bundleInputs: true}), /absolute directory/);
  });

  test("rechecks source immutability and staged-body hashes in bundle mode", t => {
    const f = fixture(t), output = path.join(f.root, "mutating-source-bundle"), inputDirectory = path.join(output, "inputs"),
      originalMkdir = fs.mkdirSync;
    try {
      fs.mkdirSync = function(directory, options) {
        const result = originalMkdir.call(fs, directory, options);
        if (directory === inputDirectory) {fs.appendFileSync(f.openmp, "changed during assembly\n");}
        return result;
      };
      assert.throws(() => assembleNativeNotices(f.inputs, output, {bundleInputs: true}), /input changed during assembly/);
    } finally {fs.mkdirSync = originalMkdir;}
    assert.equal(fs.existsSync(path.join(output, "native-notices.json")), false, "a changed source never receives an assembly receipt");

    const stable = fixture(t), stableOutput = path.join(stable.root, "staged-mutation"),
      assembled = assembleNativeNotices(stable.inputs, stableOutput, {bundleInputs: true}),
      receiptPath = path.join(stableOutput, "native-notices.json"), receiptHash = fileHash(receiptPath);
    const stagedPath = assembled.inputs[0]?.path;
    assert.ok(stagedPath);
    fs.appendFileSync(stagedPath, "changed after assembly\n");
    assert.throws(() => readNativeNoticeReceipt(receiptPath, receiptHash), /hash differs/);
  });

  test("accepts 65 unique notice inputs and rejects 129 before creating output", t => {
    const f = fixture(t), inputs = Array.from({length: 65}, (_, index) => {
        const bodyPath = path.join(f.root, "notice-" + index + ".txt"); fs.writeFileSync(bodyPath, "Unique notice body " + index + "\n");
        return {component: "Additional component " + index, path: bodyPath, sha256: fileHash(bodyPath)};
      }), output = path.join(f.root, "notice-65"), assembled = assembleNativeNotices(inputs, output),
      receiptPath = path.join(output, "native-notices.json"), receiptHash = fileHash(receiptPath), admitted = readNativeNoticeReceipt(receiptPath, receiptHash);
    assert.equal(assembled.inputs.length, 65); assert.equal(admitted.inputs.length, 65);
    assert.deepEqual(assembled.inputs.map(row => row.component), inputs.map(row => row.component));
    const tooMany = Array.from({length: 129}, (_, index) => ({...f.inputs[0], component: "Over-limit component " + index})), rejectedOutput = path.join(f.root, "notice-129");
    assert.throws(() => assembleNativeNotices(tooMany, rejectedOutput), /bounded ordered array/);
    assert.equal(fs.existsSync(rejectedOutput), false, "over-limit input count fails before creating output");
  });

  test("admits an existing assembly receipt only with its external pin and unchanged bodies", t => {
    const f = fixture(t), output = path.join(f.root, "notice-admission"), assembled = assembleNativeNotices(f.inputs, output),
      receiptPath = path.join(output, "native-notices.json"), receiptHash = fileHash(receiptPath), admitted = readNativeNoticeReceipt(receiptPath, receiptHash);
    assert.equal(admitted.receiptPath, receiptPath);
    assert.equal(admitted.receiptSha256, receiptHash);
    assert.equal(admitted.fingerprint, assembled.fingerprint);
    assert.deepEqual(admitted.notice, assembled.notice);
    assert.deepEqual(admitted.inputs, assembled.inputs);
    assert.equal(admitted.buildOnly, true);
    assert.equal(admitted.publicationQualified, false);
    assert.throws(() => readNativeNoticeReceipt(receiptPath, "0".repeat(64)), /external SHA-256/);
    fs.writeFileSync(f.openmp, f.openmpBody.toString("utf8") + "changed\n");
    assert.throws(() => readNativeNoticeReceipt(receiptPath, receiptHash), /hash differs/);
  });

  test("rejects changed notice bytes and a receipt with a forged fingerprint", t => {
    const f = fixture(t), output = path.join(f.root, "notice-admission"), assembled = assembleNativeNotices(f.inputs, output),
      receiptPath = path.join(output, "native-notices.json"), receiptHash = fileHash(receiptPath);
    fs.appendFileSync(assembled.notice.path, "extra text\n");
    assert.throws(() => readNativeNoticeReceipt(receiptPath, receiptHash), /notice bytes differ/);
    fs.writeFileSync(assembled.notice.path, fs.readFileSync(assembled.notice.path).subarray(0, -11));
    const receipt = readBuildJson(receiptPath); receipt.fingerprint = "0".repeat(64); fs.writeFileSync(receiptPath, JSON.stringify(receipt));
    assert.throws(() => readNativeNoticeReceipt(receiptPath, fileHash(receiptPath)), /fingerprint differs/);
  });

  test("keeps notice bytes and content fingerprint independent of checkout paths", t => {
    const f = fixture(t), otherRoot = path.join(f.root, "other-checkout"); fs.mkdirSync(otherRoot);
    const otherInputs = f.inputs.map(({component, path: sourcePath, sha256}) => {
      const copy = path.join(otherRoot, path.basename(sourcePath)); fs.copyFileSync(sourcePath, copy);
      return {component, path: copy, sha256};
    });
    const original = assembleNativeNotices(f.inputs, path.join(f.root, "notice-original"));
    const relocated = assembleNativeNotices(otherInputs, path.join(f.root, "notice-relocated"));
    assert.deepEqual(fs.readFileSync(relocated.notice.path), fs.readFileSync(original.notice.path));
    assert.equal(relocated.notice.sha256, original.notice.sha256);
    assert.equal(relocated.fingerprint, original.fingerprint);
    const text = fs.readFileSync(relocated.notice.path, "utf8");
    assert.equal(text.includes(f.root), false, "host input and output paths stay out of the shipped notice");
    assert.ok(text.includes("Components: LLVM OpenMP runtime\n"));
  });

  test("rejects malformed rows, labels, hashes, invalid UTF-8 and NUL input", t => {
    const f = fixture(t), output = path.join(f.root, "bad-output");
    assert.throws(() => assembleNativeNotices([{...f.inputs[0], sha256: "0".repeat(64)}], output), /hash differs/);
    assert.throws(() => assembleNativeNotices([{...f.inputs[0], extra: true}], output), /input row/);
    assert.throws(() => assembleNativeNotices([{...f.inputs[0], component: "bad\nlabel"}], output), /labels/);
    assert.throws(() => assembleNativeNotices([f.inputs[0], {...f.inputs[1], component: "llvm openmp RUNTIME"}], output), /labels/);
    assert.throws(() => assembleNativeNotices([{...f.inputs[0], sha256: "A".repeat(64)}], output), /SHA-256/);
    fs.writeFileSync(f.openmp, Buffer.from([0xff]));
    assert.throws(() => assembleNativeNotices([{...f.inputs[0], sha256: fileHash(f.openmp)}], output), /UTF-8/);
    fs.writeFileSync(f.openmp, Buffer.from([0x61, 0x00, 0x62]));
    assert.throws(() => assembleNativeNotices([{...f.inputs[0], sha256: fileHash(f.openmp)}], output), /NUL/);
    assert.equal(fs.existsSync(output), false, "invalid inputs fail before creating output");
  });

  test("rejects non-file and over-limit inputs", t => {
    const f = fixture(t), directory = path.join(f.root, "directory"); fs.mkdirSync(directory);
    assert.throws(() => assembleNativeNotices([{component: "directory", path: directory, sha256: "0".repeat(64)}], path.join(f.root, "out-directory")), /regular file/);
    const large = path.join(f.root, "large.txt"); fs.writeFileSync(large, Buffer.alloc((8 << 20) + 1, 0x61));
    assert.throws(() => assembleNativeNotices([{component: "large", path: large, sha256: createHash("sha256").update("unused").digest("hex")}], path.join(f.root, "out-large")), /bounded/);
  });

  test("rejects hard-linked inputs", t => {
    const f = fixture(t), hardlink = path.join(f.root, "hardlink.txt"); fs.linkSync(f.openmp, hardlink);
    assert.throws(() => assembleNativeNotices([{component: "unsafe", path: hardlink, sha256: fileHash(f.openmp)}], path.join(f.root, "out-hardlink")), /regular file/);
  });

  test("rejects symlinked inputs", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), link = path.join(f.root, "symlink.txt"); fs.symlinkSync(f.openmp, link);
    assert.throws(() => assembleNativeNotices([{component: "unsafe", path: link, sha256: fileHash(f.openmp)}], path.join(f.root, "out-symlink")), /canonical|symlink|regular/);
  });

  test("requires a fresh canonical output path and refuses clobbering", t => {
    const f = fixture(t), existing = path.join(f.root, "existing"); fs.mkdirSync(existing); fs.writeFileSync(path.join(existing, "keep"), "unchanged");
    assert.throws(() => assembleNativeNotices(f.inputs, existing), /fresh absolute directory/);
    assert.equal(fs.readFileSync(path.join(existing, "keep"), "utf8"), "unchanged");
    assert.throws(() => assembleNativeNotices(f.inputs, path.join(f.root, "missing-parent", "new")));
    assert.throws(() => assembleNativeNotices(f.inputs, "relative-output"), /absolute directory/);
    assert.throws(() => assembleNativeNotices(f.inputs, /** @type {string} */ (/** @type {unknown} */ (null))), /absolute directory/);
  });

  test("rejects symlinked output parents", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), parentLink = path.join(f.root, "parent-link"); fs.symlinkSync(f.root, parentLink, "dir");
    assert.throws(() => assembleNativeNotices(f.inputs, path.join(parentLink, "new-output")), /symlink/);
    assert.throws(() => assembleNativeNotices(f.inputs, path.join(parentLink, "new-bundle"), {bundleInputs: true}), /symlink/);
  });
});

test.describe("GNU tar native notice projection", {skip: process.platform !== "linux" && "Linux CI archive projection"}, () => {
  test("rejects other hosts before reading inputs or creating output", () => {
    const platform = Object.getOwnPropertyDescriptor(process, "platform");
    assert.ok(platform);
    try {
      Object.defineProperty(process, "platform", {...platform, value: "win32"});
      assert.throws(() => projectNativeArchiveNotices([], "not-an-output-path"), /requires the Linux CI GNU tar/);
    } finally {Object.defineProperty(process, "platform", platform);}
  });

  test("projects one pinned regular archive member and feeds the ordinary assembler", t => {
    const f = archiveFixture(t), projected = projectNativeArchiveNotices(f.rows, path.join(f.root, "projected")),
      projection = readBuildJson(projected.provenance.path), projectedInput = projected.inputs[0], projectionInput = projection.inputs[0];
    assert.ok(projectedInput); assert.ok(projectionInput);
    assert.deepEqual(projected.inputs, [{component: "Pinned test component", path: path.join(f.root, "projected/000-pinned-test-component.txt"),
      sha256: createHash("sha256").update(f.licenseBody).digest("hex")}]);
    assert.deepEqual(fs.readFileSync(projectedInput.path), f.licenseBody);
    assert.equal(projected.provenance.sha256, fileHash(projected.provenance.path));
    assert.equal(projectionInput.archive.path, f.archive);
    assert.equal(projectionInput.archive.sha256, fileHash(f.archive));
    assert.equal(projectionInput.member, f.member);
    assert.deepEqual(projectionInput.admissionFiles, [{path: f.admission, sha256: fileHash(f.admission)}]);
    const assembled = assembleNativeNotices(projected.inputs, path.join(f.root, "assembled-from-archive"));
    assert.ok(fs.readFileSync(assembled.notice.path).includes(f.licenseBody));
    assert.equal(assembled.inputs[0]?.sha256, projectedInput.sha256);
    assert.equal(projection.buildOnly, true); assert.equal(projection.publicationQualified, false);
  });

  test("projects multiple pinned members from one admitted archive in order", t => {
    const f = archiveFixture(t), secondMember = "vendor/SECOND-LICENSE", secondBody = Buffer.from("Second complete upstream license.\n", "utf8");
    fs.writeFileSync(path.join(f.source, secondMember), secondBody);
    tarFixture(["-cf", f.archive, "-C", f.source, f.member, secondMember]);
    const first = f.rows[0]; assert.ok(first);
    const archive = {path: f.archive, sha256: fileHash(f.archive)}, rows = [{...first, archive},
        {component: "Second pinned test component", archive, member: secondMember, admissionFiles: first.admissionFiles}],
      output = path.join(f.root, "projected-multiple"),
      projected = projectNativeArchiveNotices(rows, output), projection = readBuildJson(projected.provenance.path);
    assert.deepEqual(projected.inputs.map(row => row.component), ["Pinned test component", "Second pinned test component"]);
    assert.deepEqual(projected.inputs.map(row => fs.readFileSync(row.path)), [f.licenseBody, secondBody]);
    assert.deepEqual(projection.inputs.map((/** @type {{member:string}} */ row) => row.member), [f.member, secondMember]);
    const assembled = assembleNativeNotices(projected.inputs, path.join(f.root, "assembled-from-multiple-archive-members"));
    const notice = fs.readFileSync(assembled.notice.path);
    assert.ok(notice.includes(f.licenseBody)); assert.ok(notice.includes(secondBody));
    assert.ok(notice.indexOf(f.licenseBody) < notice.indexOf(secondBody), "archive member order survives projection and assembly");
  });

  test("rejects archive or admission pins that do not match current bytes", t => {
    const f = archiveFixture(t), output = path.join(f.root, "bad-pin"), row = f.rows[0];
    assert.ok(row);
    const badArchive = [{...row, archive: {...row.archive, sha256: "0".repeat(64)}}];
    assert.throws(() => projectNativeArchiveNotices(badArchive, output), /source archive SHA-256 differs/);
    assert.equal(fs.existsSync(output), false);
    fs.appendFileSync(f.admission, "changed\n");
    assert.throws(() => projectNativeArchiveNotices(f.rows, output), /admission file SHA-256 differs/);
    assert.equal(fs.existsSync(output), false);
  });

  test("projects pinned Windows PAX members without importing platform xattrs", t => {
    const f = archiveFixture(t), archive = path.join(f.root, "windows-pax.tar");
    tarFixture(["--format=pax", "--pax-option=SCHILY.fflags:=arch", "-cf", archive, "-C", f.source, f.member]);
    const rows = [{...f.rows[0], archive: {path: archive, sha256: fileHash(archive)}}],
      projected = projectNativeArchiveNotices(rows, path.join(f.root, "pax-output"));
    assert.deepEqual(fs.readFileSync(projected.inputs[0]?.path || ""), f.licenseBody);
  });

  test("stops projecting as soon as aggregate notice bytes exceed the limit", t => {
    const f = archiveFixture(t), output = path.join(f.root, "aggregate-output"), row = f.rows[0];
    assert.ok(row);
    fs.writeFileSync(path.join(f.source, f.member), Buffer.alloc(3 << 20, 0x61));
    fs.unlinkSync(f.archive);
    tarFixture(["-cf", f.archive, "-C", f.source, f.member]);
    const pinned = {...row, archive: {...row.archive, sha256: fileHash(f.archive)}},
      rows = [0, 1, 2].map(index => ({...pinned, component: "Component " + index}));
    // If all members were buffered before the total-size check, this missing fourth member
    // would mask the aggregate failure and permit memory to grow with the input count.
    rows.push({...pinned, component: "Unread component", member: "vendor/MISSING"});
    assert.throws(() => projectNativeArchiveNotices(rows, output), /bodies exceed the total size limit/);
    assert.equal(fs.existsSync(output), false);
  });

  test("rejects duplicate archive members and unsafe member paths", t => {
    const f = archiveFixture(t), duplicateSource = path.join(f.root, "duplicate-source"), duplicateArchive = path.join(f.root, "duplicate.tar");
    fs.mkdirSync(path.join(duplicateSource, "vendor"), {recursive: true});
    fs.writeFileSync(path.join(duplicateSource, f.member), "second license body\n");
    fs.copyFileSync(f.archive, duplicateArchive);
    tarFixture(["-rf", duplicateArchive, "-C", duplicateSource, f.member]);
    const duplicateRows = [{...f.rows[0], archive: {path: duplicateArchive, sha256: fileHash(duplicateArchive)}}];
    assert.throws(() => projectNativeArchiveNotices(duplicateRows, path.join(f.root, "duplicate-output")), /exactly one matching member/);
    for (const member of ["../escape", "/absolute", "vendor/*", "vendor/../LICENSE", "C:/absolute"]) {
      const rows = [{...f.rows[0], member}];
      assert.throws(() => projectNativeArchiveNotices(rows, path.join(f.root, "unsafe-" + String(member.length))), /fixed relative path/);
    }
  });

  test("rejects non-UTF-8 and NUL archive bodies before staging", t => {
    const f = archiveFixture(t), invalidBodies = /** @type {[string, Buffer, RegExp][]} */ ([
      ["invalid-utf8", Buffer.from([0xff]), /valid UTF-8/], ["nul-body", Buffer.from([0x61, 0x00, 0x62]), /NUL bytes/]]);
    for (const [name, body, expected] of invalidBodies) {
      const source = path.join(f.root, name), archive = path.join(f.root, name + ".tar");
      fs.mkdirSync(path.join(source, "vendor"), {recursive: true}); fs.writeFileSync(path.join(source, f.member), body);
      tarFixture(["-cf", archive, "-C", source, f.member]);
      const rows = [{...f.rows[0], archive: {path: archive, sha256: fileHash(archive)}}], output = path.join(f.root, name + "-output");
      assert.throws(() => projectNativeArchiveNotices(rows, output), expected);
      assert.equal(fs.existsSync(output), false);
    }
  });

  test("rejects symlink, directory and hardlink tar entries", t => {
    const f = archiveFixture(t), symlinkSource = path.join(f.root, "symlink-source"), symlinkArchive = path.join(f.root, "symlink.tar");
    fs.mkdirSync(path.join(symlinkSource, "vendor"), {recursive: true}); fs.writeFileSync(path.join(symlinkSource, "vendor/target"), "target\n");
    fs.symlinkSync("target", path.join(symlinkSource, f.member));
    tarFixture(["-cf", symlinkArchive, "-C", symlinkSource, f.member]);
    assert.throws(() => projectNativeArchiveNotices([{...f.rows[0], archive: {path: symlinkArchive, sha256: fileHash(symlinkArchive)}}],
      path.join(f.root, "symlink-member-output")), /regular file/);

    const directorySource = path.join(f.root, "directory-source"), directoryArchive = path.join(f.root, "directory.tar");
    fs.mkdirSync(path.join(directorySource, f.member), {recursive: true}); fs.writeFileSync(path.join(directorySource, f.member, "child"), "child\n");
    tarFixture(["-cf", directoryArchive, "-C", directorySource, f.member]);
    assert.throws(() => projectNativeArchiveNotices([{...f.rows[0], archive: {path: directoryArchive, sha256: fileHash(directoryArchive)}}],
      path.join(f.root, "directory-member-output")), /exactly one matching member|regular file/);

    const hardlinkSource = path.join(f.root, "hardlink-source"), hardlinkArchive = path.join(f.root, "hardlink.tar");
    fs.mkdirSync(path.join(hardlinkSource, "vendor"), {recursive: true});
    fs.writeFileSync(path.join(hardlinkSource, "vendor/original"), "license\n"); fs.linkSync(path.join(hardlinkSource, "vendor/original"), path.join(hardlinkSource, f.member));
    tarFixture(["-cf", hardlinkArchive, "-C", hardlinkSource, "vendor/original", f.member]);
    assert.throws(() => projectNativeArchiveNotices([{...f.rows[0], archive: {path: hardlinkArchive, sha256: fileHash(hardlinkArchive)}}],
      path.join(f.root, "hardlink-member-output")), /regular file/);
  });

  test("rejects symlinked and hard-linked archive or admission inputs", {skip: process.platform === "win32"}, t => {
    const f = archiveFixture(t), symlinkArchive = path.join(f.root, "source-link.tar"), hardlinkArchive = path.join(f.root, "source-hardlink.tar");
    fs.symlinkSync(f.archive, symlinkArchive);
    assert.throws(() => projectNativeArchiveNotices([{...f.rows[0], archive: {path: symlinkArchive, sha256: fileHash(f.archive)}}],
      path.join(f.root, "symlink-archive-output")), /symlink|regular file/);
    fs.linkSync(f.archive, hardlinkArchive);
    assert.throws(() => projectNativeArchiveNotices([{...f.rows[0], archive: {path: hardlinkArchive, sha256: fileHash(f.archive)}}],
      path.join(f.root, "hardlink-archive-output")), /regular file/);
  });

  test("enforces the aggregate projection deadline", t => {
    const f = archiveFixture(t), originalNow = Date.now; let calls = 0;
    try {
      Date.now = () => calls++ === 0 ? 1000 : 1000000000;
      assert.throws(() => projectNativeArchiveNotices(f.rows, path.join(f.root, "deadline-output")), /aggregate deadline/);
      assert.equal(fs.existsSync(path.join(f.root, "deadline-output")), false);
    } finally {Date.now = originalNow;}
  });

  test("passes only the remaining aggregate deadline to tar", {skip: process.platform === "win32"}, t => {
    const f = archiveFixture(t), fakeBin = path.join(f.root, "fake-bin"), fakeTar = path.join(fakeBin, "tar"), originalPath = process.env["PATH"],
      originalNow = Date.now; let calls = 0;
    fs.mkdirSync(fakeBin); fs.writeFileSync(fakeTar, "#!/bin/sh\nexec /bin/sleep 5\n"); fs.chmodSync(fakeTar, 0o755);
    try {
      process.env["PATH"] = fakeBin + path.delimiter + (originalPath || "");
      Date.now = () => {
        const call = calls++;
        return call === 0 ? 1000 : call < 7 ? 1001 : 120999;
      };
      assert.throws(() => projectNativeArchiveNotices(f.rows, path.join(f.root, "tar-timeout-output")), /could not be read safely/);
      assert.equal(fs.existsSync(path.join(f.root, "tar-timeout-output")), false);
    } finally {process.env["PATH"] = originalPath; Date.now = originalNow;}
  });
});
