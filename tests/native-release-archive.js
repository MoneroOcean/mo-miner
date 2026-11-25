"use strict";
const fs = require("node:fs"), path = require("node:path"), assert = require("node:assert/strict"), test = require("node:test");
const {packageNativeRelease} = require("../scripts/package-native-release"), {fileHash} = require("../scripts/build-source-inputs");
const {auditNativeRelease} = require("../scripts/check-native-release");
const {createHash} = require("node:crypto");
const {setTimeout: delay} = require("node:timers/promises");
const {assembleNativeNotices} = require("../scripts/prepare-native-notices");
const {tempDir} = require("./common/temp-dir");
const noticeName = "THIRD-PARTY-NOTICES.txt";
const skipLinuxCandidateArchives = process.platform !== "linux" && "Linux candidate archives require ELF and POSIX executable metadata";

/** Exercise the actual archive orchestrator; only CMake is stubbed, never execute the image.
 * @param {import('node:test').TestContext} t @param {boolean} [windows] */
function fixture(t, windows = process.platform === "win32") {
  const root = tempDir(t, "mom-native-archive-");
  const name = windows ? "mom.exe" : "mom", binary = path.join(root, name), notice = path.join(root, noticeName), tool = path.join(root, "cmake"), entry = path.join(root, "entrypoint.json"),
    archive = path.join(root, `mom-v0.9.4-${windows ? "win.zip" : "lin.tgz"}`);
  fs.writeFileSync(binary, windows ? Buffer.from("4d5a000066697874757265", "hex") : Buffer.from("7f454c4666697874757265", "hex"), {mode: 0o755});
  fs.writeFileSync(notice, "Third-party notices fixture.\n", {mode: 0o644});
  fs.writeFileSync(tool, "fixture tool");
  const receipt = /** @type {{version: number, target: string, buildOnly: true, publicationQualified?: false,
    artifact: {path: string, sha256: string, bytes: number}, notices?: {path: string, sha256: string, bytes: number},
    noticeAssembly?: {receiptPath: string, receiptSha256: string, fingerprint: string, notice: {path: string, sha256: string, bytes: number}},
    configFile?: string, configHash?: string, producerReceipt?: {path: string, sha256: string},
    componentHashes?: Record<string, string>, source?: {path: string, sha256: string, bytes: number}}} */ ({version: 1, target: windows ? "win32" : "linux", buildOnly: true,
      artifact: {path: binary, sha256: fileHash(binary), bytes: fs.statSync(binary).size},
      notices: {path: notice, sha256: fileHash(notice), bytes: fs.statSync(notice).size}});
  const write = () => fs.writeFileSync(entry, JSON.stringify(receipt)); write();
  const state = {fail: "", list: name + "\n" + noticeName + "\n", corrupt: false, corruptNotice: false, executableNotice: false,
      edit: "", noticeForArchive: notice, captureModes: false, stagedModes: /** @type {number[]} */ ([])}, calls = /** @type {string[][]} */ ([]);
  const spawn = /** @type {typeof import('node:child_process').spawnSync} */ (/** @type {unknown} */ (
    (/** @type {string} */ command, /** @type {string[]} */ args, /** @type {{cwd: string, env?: NodeJS.ProcessEnv}} */ options) => {
      assert.equal(command, tool); calls.push(args); assert.deepEqual(args.slice(0, 2), ["-E", "tar"]);
      if (options.env) {assert.equal(options.env["TZ"], "UTC"); assert.equal(options.env["SOURCE_DATE_EPOCH"], "315532800");}
      const operation = args[2], archive = args[3]; assert.ok(operation && archive);
      if (state.fail === operation) {return {status: 3, stdout: "", stderr: "fixture failure"};}
      if (operation === "cf" || operation === "czf") {
        if (state.captureModes) {state.stagedModes = [fs.statSync(path.join(options.cwd, name)).mode & 0o777, fs.statSync(path.join(options.cwd, noticeName)).mode & 0o777];}
        fs.writeFileSync(archive, "archive fixture");
      }
      if (operation === "tvf") {
        const executableMode = windows ? "-rw-r--r--" : "-rwxr-xr-x";
        return {status: 0, stdout: `${executableMode} 0 user group ${fs.statSync(binary).size} Jan 01 00:00 ${name}\n-rw-r--r-- 0 user group ${fs.statSync(state.noticeForArchive).size} Jan 01 00:00 ${noticeName}\n`, stderr: ""};
      }
      if (operation === "xf") {
        fs.copyFileSync(binary, path.join(options.cwd, name)); fs.copyFileSync(state.noticeForArchive, path.join(options.cwd, noticeName));
        if (state.corrupt) {fs.appendFileSync(path.join(options.cwd, name), "bad");}
        if (state.corruptNotice) {fs.appendFileSync(path.join(options.cwd, noticeName), "bad");}
        if (state.executableNotice) {fs.chmodSync(path.join(options.cwd, noticeName), 0o755);}
      }
      if (operation === "tf" && state.edit) {fs.appendFileSync(state.edit, " ");}
      return {status: 0, stdout: operation === "tf" ? state.list : "", stderr: ""};
    }));
  return {root, binary, name, windows, notice, tool, entry, archive, receipt, calls, state, write, spawn, run: (version = "0.9.4") => packageNativeRelease(entry, tool, version, spawn)};
}

/** Add the complete Linux provenance shape needed by archive admission without executing this fixture image.
 * @param {ReturnType<typeof fixture>} f */
function candidateFixture(f) {
  const elf = Buffer.alloc(64); Buffer.from("7f454c46", "hex").copy(elf); elf[4] = 2; elf[5] = 1; elf[6] = 1;
  elf.writeUInt16LE(3, 16); elf.writeUInt16LE(62, 18); elf.writeUInt32LE(1, 20); elf.writeUInt16LE(64, 52); elf.writeUInt16LE(56, 54); elf.writeUInt16LE(1, 56);
  fs.writeFileSync(f.binary, elf, {mode: 0o755});
  f.receipt.artifact.sha256 = fileHash(f.binary); f.receipt.artifact.bytes = fs.statSync(f.binary).size;
  const config = path.join(f.root, "candidate-config.json"), producer = path.join(f.root, "result.json");
  fs.writeFileSync(config, "{}\n");
  fs.writeFileSync(producer, JSON.stringify({buildOnly: true, publicationQualified: false, configHash: fileHash(config),
    target: "linux", binary: f.binary, sha256: fileHash(f.binary), bytes: fs.statSync(f.binary).size}));
  f.receipt["publicationQualified"] = false;
  f.receipt["configFile"] = config; f.receipt["configHash"] = fileHash(config);
  f.receipt["producerReceipt"] = {path: producer, sha256: fileHash(producer)};
  f.receipt["componentHashes"] = {[producer]: fileHash(producer)};
  f.receipt["source"] = {path: f.binary, sha256: fileHash(f.binary), bytes: fs.statSync(f.binary).size};
  f.write();
}

/** Attach a real assembly proof whose body is staged separately beside the candidate.
 * @param {ReturnType<typeof fixture>} f */
function bindNoticeAssembly(f) {
  const input = path.join(f.root, "reviewed-license.txt"), output = path.join(f.root, "notice-assembly");
  fs.writeFileSync(input, "Reviewed upstream license text.\n");
  const assembled = assembleNativeNotices([{component: "fixture", path: input, sha256: fileHash(input)}], output),
    receiptPath = path.join(output, "native-notices.json"), receiptSha256 = fileHash(receiptPath);
  fs.copyFileSync(assembled.notice.path, f.notice);
  f.receipt.notices = {path: f.notice, sha256: fileHash(f.notice), bytes: fs.statSync(f.notice).size};
  f.receipt.noticeAssembly = {receiptPath, receiptSha256, fingerprint: assembled.fingerprint, notice: assembled.notice};
  f.write();
  return {input, receiptPath, receiptSha256, binding: f.receipt.noticeAssembly};
}

/** @param {string} raw @param {string} target */
function fixtureAudit(raw, target) {assert.equal(target, "linux"); assert.ok(fs.statSync(raw).isFile());}

/** @param {import('node:test').TestContext} t @param {ReturnType<typeof fixture>} f */
function latePackageInputs(t, f) {
  const output = path.join(path.dirname(f.root), path.basename(f.root) + "-late-package");
  const draftRoot = path.join(f.root, "draft"); fs.mkdirSync(draftRoot);
  const notices = path.join(draftRoot, noticeName); fs.writeFileSync(notices, "Reviewed final notice fixture.\n", {mode: 0o644});
  t.after(() => fs.rmSync(output, {recursive: true, force: true}));
  return {output, notices};
}

/** @param {import('node:test').TestContext} t @param {ReturnType<typeof fixture>} f @param {string} [body] */
function assembledLatePackageInputs(t, f, body = "Late reviewed license text.\n") {
  const source = path.join(f.root, "late-reviewed-license.txt"), assemblyDirectory = path.join(f.root, "late-notice-assembly");
  fs.writeFileSync(source, body);
  const assembled = assembleNativeNotices([{component: "late fixture", path: source, sha256: fileHash(source)}], assemblyDirectory),
    receiptPath = path.join(assemblyDirectory, "native-notices.json"), late = latePackageInputs(t, f);
  return {packageInputs: {...late, notices: assembled.notice.path, receipt: {path: receiptPath, sha256: fileHash(receiptPath)}}, source, assembled};
}

/** @typedef {{output: string, notices: string, receipt: {path: string, sha256: string}}} AssembledLatePackageInputs */

test.describe("Standalone native release archives", () => {
  for (const windows of [false, true]) {
    test(`${windows ? "Windows" : "Linux"} ships its executable and data-only notices with exact restored bytes`, {
      skip: !windows && process.platform === "win32" && "Linux archive validation requires POSIX executable metadata",
    }, t => {
      const f = fixture(t, windows), r = f.run(), name = windows ? "mom.exe" : "mom";
      assert.deepEqual(r.members, [name, noticeName]); assert.equal(r.exactOriginalBytes, true); assert.equal(r.publicationQualified, false);
      assert.deepEqual(r.notices, {path: f.notice, sha256: fileHash(f.notice), bytes: fs.statSync(f.notice).size});
      assert.equal(r.archive.path, path.join(f.root, "mom-v0.9.4-" + (windows ? "win.zip" : "lin.tgz")));
      assert.equal(fileHash(r.archive.path), r.archive.sha256); assert.equal(fileHash(f.binary), r.executable.sha256);
      assert.equal(f.calls[0]?.includes("--mtime=1980-01-01 00:00:00 UTC"), !windows,
        "ZIP must use SOURCE_DATE_EPOCH to normalize all timestamps, not --mtime alone");
      assert.equal(f.calls.length, 3); assert.deepEqual(f.calls[0]?.slice(-4), ["--format=" + (windows ? "zip" : "gnutar"), "--", name, noticeName]);
      assert.equal(fs.readdirSync(f.root).some(n => n.startsWith(".package-native-")), false);
    });
  }
  test("SDK files, Node, logs and libraries beside the image never enter archive arguments", t => {
    const f = fixture(t); for (const name of ["node", "mom.node", "libsycl.so", "compiler.exe", "secret.json"]) {fs.writeFileSync(path.join(f.root, name), "not shipped");}
    f.run(); assert.deepEqual(f.calls[0]?.slice(-3), ["--", f.name, noticeName]);
    assert.equal(f.calls[0]?.some(value => ["node", "mom.node", "libsycl.so", "compiler.exe", "secret.json"].includes(value)), false);
  });
  test("existing archives are never overwritten", t => {
    const f = fixture(t); fs.writeFileSync(f.archive, "preserve");
    assert.throws(f.run, /already exists/); assert.equal(fs.readFileSync(f.archive, "utf8"), "preserve"); assert.equal(f.calls.length, 0);
  });
  test("an existing package receipt is refused before archive mutation", t => {
    const f = fixture(t); fs.writeFileSync(path.join(f.root, "native-package.json"), "preserve");
    assert.throws(f.run, /already exists/); assert.equal(f.calls.length, 0);
  });
  test("invalid or escaping release labels fail before commands", t => {
    const f = fixture(t); for (const version of ["", "../outside", "0.9.4/other", "0.9.4\n", "0.9.4;echo bad"]) {assert.throws(() => f.run(version), /release version/);}
    assert.equal(f.calls.length, 0);
  });
  test("relative receipt and tool paths cannot invoke a command", t => {
    const f = fixture(t); assert.throws(() => packageNativeRelease("relative", f.tool, "0.9.4", f.spawn), /absolute/);
    assert.throws(() => packageNativeRelease(f.entry, "cmake", "0.9.4", f.spawn), /absolute/); assert.equal(f.calls.length, 0);
  });
  test("receipt hashes and byte counts must describe the actual image", t => {
    const f = fixture(t); f.receipt.artifact.sha256 = "0".repeat(64); f.write(); assert.throws(f.run, /bind/);
    f.receipt.artifact.sha256 = fileHash(f.binary); f.receipt.artifact.bytes++; f.write(); assert.throws(f.run, /bind/); assert.equal(f.calls.length, 0);
  });
  test("a notice receipt is required and must bind its exact adjacent filename, bytes and hash", t => {
    const f = fixture(t); Reflect.deleteProperty(f.receipt, "notices"); f.write(); assert.throws(f.run, /receipt does not bind its notices/);
    f.receipt.notices = {path: path.join(f.root, "other.txt"), sha256: fileHash(f.notice), bytes: fs.statSync(f.notice).size};
    f.write(); assert.throws(f.run, /receipt does not bind its notices/);
    f.receipt.notices = {path: f.notice, sha256: "0".repeat(64), bytes: fs.statSync(f.notice).size};
    f.write(); assert.throws(f.run, /receipt does not bind its notices/); assert.equal(f.calls.length, 0);
  });
  test("a missing notice file fails before archive creation", t => {
    const f = fixture(t); fs.unlinkSync(f.notice); assert.throws(f.run, /notice/); assert.equal(f.calls.length, 0);
  });
  test("notice size, NULs and invalid text fail before archive creation", t => {
    const f = fixture(t); fs.writeFileSync(f.notice, Buffer.from([0x61, 0x00, 0x62]));
    f.receipt.notices = {path: f.notice, sha256: fileHash(f.notice), bytes: fs.statSync(f.notice).size}; f.write();
    assert.throws(f.run, /NUL/); assert.equal(f.calls.length, 0);
    fs.writeFileSync(f.notice, Buffer.from([0xff]));
    f.receipt.notices = {path: f.notice, sha256: fileHash(f.notice), bytes: fs.statSync(f.notice).size}; f.write();
    assert.throws(f.run, /UTF-8/); assert.equal(f.calls.length, 0);
    fs.writeFileSync(f.notice, Buffer.alloc((8 << 20) + 1, 0x61));
    f.receipt.notices = {path: f.notice, sha256: fileHash(f.notice), bytes: fs.statSync(f.notice).size}; f.write();
    assert.throws(f.run, /notice/); assert.equal(f.calls.length, 0);
  });
  test("symlinked notice inputs cannot redirect the package", {skip: process.platform === "win32"}, t => {
    const f = fixture(t); fs.renameSync(f.notice, f.notice + ".actual"); fs.symlinkSync(f.notice + ".actual", f.notice);
    assert.throws(f.run, /notice/); assert.equal(f.calls.length, 0);
  });
  test("equal-length UTF-8 replacement cannot certify bytes different from the archive member", t => {
    const f = fixture(t); fs.writeFileSync(f.notice, Buffer.from([0xf0, 0x90, 0x80]));
    f.receipt.notices = {path: f.notice, bytes: 3, sha256: createHash("sha256").update("\ufffd").digest("hex")}; f.write();
    assert.throws(f.run, /UTF-8/); assert.equal(f.calls.length, 0);
  });
  test("executable notice files are rejected as data", {skip: process.platform === "win32"}, t => {
    const f = fixture(t); fs.chmodSync(f.notice, 0o755); assert.throws(f.run, /notice/); assert.equal(f.calls.length, 0);
  });
  test("target mismatch cannot rename a non-native payload into a release", t => {
    const f = fixture(t); fs.writeFileSync(f.binary, "node archive"); f.receipt.artifact.sha256 = fileHash(f.binary); f.receipt.artifact.bytes = fs.statSync(f.binary).size; f.write();
    assert.throws(f.run, /format differs/); assert.equal(f.calls.length, 0);
  });
  test("a receipt cannot select a different filename or sibling payload", t => {
    const f = fixture(t); f.receipt.artifact.path += ".other"; f.write(); assert.throws(f.run, /bind/); assert.equal(f.calls.length, 0);
  });
  test("symlinked executable inputs cannot redirect the package", {skip: process.platform === "win32"}, t => {
    const f = fixture(t); fs.renameSync(f.binary, f.binary + ".actual"); fs.symlinkSync(f.binary + ".actual", f.binary);
    assert.throws(f.run, /bind/); assert.equal(f.calls.length, 0);
  });
  test("unexpected archive members fail before extraction or publication", t => {
    const f = fixture(t); f.state.list += "libsycl.so\n"; assert.throws(f.run, /contain only/);
    assert.equal(f.calls.length, 2); assert.ok(!fs.existsSync(f.archive));
  });
  test("altered extracted bytes cannot receive an archive receipt", t => {
    const f = fixture(t); f.state.corrupt = true; assert.throws(f.run, /changed executable bytes/); assert.ok(!fs.existsSync(path.join(f.root, "native-package.json")));
  });
  test("altered extracted notice bytes cannot receive an archive receipt", t => {
    const f = fixture(t); f.state.corruptNotice = true; assert.throws(f.run, /changed notice bytes/); assert.ok(!fs.existsSync(f.archive));
  });
  test("executable notice archive members cannot receive an archive receipt", {skip: process.platform === "win32"}, t => {
    const f = fixture(t); f.state.executableNotice = true; assert.throws(f.run, /notice/); assert.ok(!fs.existsSync(f.archive));
  });
  for (const input of /** @type {("entry" | "tool" | "notice")[]} */ (["entry", "tool", "notice"])) {
    test(`changed ${input} cannot be paired with an earlier artifact hash`, t => {
      const f = fixture(t); f.state.edit = f[input]; assert.throws(f.run, input === "notice" ? /notice/ : /inputs changed/);
      assert.ok(!fs.existsSync(f.archive));
    });
  }
  test("failed archive stages retain diagnostics without publishing or retrying", t => {
    const f = fixture(t); f.state.fail = f.windows ? "cf" : "czf"; assert.throws(f.run, /stage failed: create/); assert.equal(f.calls.length, 1);
    assert.ok(!fs.existsSync(f.archive)); assert.ok(fs.readdirSync(f.root).some(n => n.startsWith(".package-native-")));
  });
  test.describe("Linux candidate archive admission", {skip: skipLinuxCandidateArchives}, () => {
    test("late notice packaging preserves the admitted candidate and v2 archive admission binds both notices", t => {
      const f = fixture(t); candidateFixture(f); const inputs = latePackageInputs(t, f), entryHash = fileHash(f.entry), binaryHash = fileHash(f.binary), oldNoticeHash = fileHash(f.notice);
      f.state.noticeForArchive = inputs.notices; f.state.captureModes = true;
      const oldUmask = process.umask(0o077); let packaged;
      try {packaged = packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, inputs);} finally {process.umask(oldUmask);}
      assert.equal(packaged.version, 2); assert.equal(packaged.entryHash, entryHash);
      assert.deepEqual(packaged.originNotices, {path: f.notice, sha256: oldNoticeHash, bytes: fs.statSync(f.notice).size});
      assert.deepEqual(packaged.members, ["mom", noticeName]); assert.equal(packaged.publicationQualified, false);
      assert.equal(fileHash(f.entry), entryHash); assert.equal(fileHash(f.binary), binaryHash); assert.equal(fileHash(f.notice), oldNoticeHash);
      assert.equal(fileHash(path.join(inputs.output, noticeName)), fileHash(inputs.notices));
      if (process.platform !== "win32") {
        assert.deepEqual(f.state.stagedModes, [fs.statSync(f.binary).mode & 0o777, 0o644]);
        assert.equal(fs.statSync(path.join(inputs.output, noticeName)).mode & 0o777, 0o644);
      }
      const checked = auditNativeRelease(f.entry, {archive: {path: packaged.archive.path, tool: f.tool}, spawn: f.spawn, audit: fixtureAudit});
      assert.equal(checked.archiveChecked, true); assert.equal(checked.sha256, binaryHash); assert.equal(checked.publicationQualified, false);
    });
    test("normal packaging re-admits notice source proof and clean-host archive checks need only its candidate binding", t => {
      const f = fixture(t); candidateFixture(f); const proof = bindNoticeAssembly(f), packaged = f.run(), packagePath = path.join(f.root, "native-package.json");
      assert.deepEqual(packaged.noticeAssembly, proof.binding);
      fs.unlinkSync(proof.input); fs.unlinkSync(proof.receiptPath);
      const checked = auditNativeRelease(f.entry, {archive: {path: packaged.archive.path, tool: f.tool}, spawn: f.spawn, audit: fixtureAudit});
      assert.equal(checked.archiveChecked, true);
      assert.equal(checked.sha256, fileHash(f.binary));
      assert.deepEqual(JSON.parse(fs.readFileSync(packagePath, "utf8")).noticeAssembly, proof.binding);
    });
    test("a changed assembly receipt is rejected before archive commands", t => {
      const f = fixture(t); candidateFixture(f); const proof = bindNoticeAssembly(f);
      f.receipt.noticeAssembly = {...proof.binding, receiptSha256: "0".repeat(64)}; f.write();
      assert.throws(f.run, /external SHA-256/); assert.equal(f.calls.length, 0);
    });
    test("late notice overrides do not inherit proof for different original bytes", t => {
      const f = fixture(t); candidateFixture(f); bindNoticeAssembly(f);
      const inputs = latePackageInputs(t, f); f.state.noticeForArchive = inputs.notices;
      const packaged = packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, inputs);
      assert.equal(Object.hasOwn(packaged, "noticeAssembly"), false);
      const checked = auditNativeRelease(f.entry, {archive: {path: packaged.archive.path, tool: f.tool}, spawn: f.spawn, audit: fixtureAudit});
      assert.equal(checked.archiveChecked, true);
    });
    test("late notice packaging retains proof only when the selected bytes still match", t => {
      const f = fixture(t); candidateFixture(f); const proof = bindNoticeAssembly(f), inputs = latePackageInputs(t, f);
      fs.copyFileSync(f.notice, inputs.notices); f.state.noticeForArchive = inputs.notices;
      const packaged = packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, inputs);
      assert.deepEqual(packaged.noticeAssembly, proof.binding);
      const checked = auditNativeRelease(f.entry, {archive: {path: packaged.archive.path, tool: f.tool}, spawn: f.spawn, audit: fixtureAudit});
      assert.equal(checked.archiveChecked, true);
    });
    test("late notice packaging binds a pinned assembler receipt for changed notice bytes", t => {
      const f = fixture(t); candidateFixture(f);
      const {packageInputs, assembled} = assembledLatePackageInputs(t, f);
      f.state.noticeForArchive = packageInputs.notices;
      const packaged = packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, packageInputs);
      assert.deepEqual(packaged.originNotices, f.receipt.notices, "the candidate's original notice proof remains separate");
      assert.deepEqual(packaged.notices, {path: path.join(packageInputs.output, noticeName), sha256: assembled.notice.sha256, bytes: assembled.notice.bytes});
      assert.deepEqual(packaged.noticeAssembly, {receiptPath: packageInputs.receipt.path, receiptSha256: packageInputs.receipt.sha256,
        fingerprint: assembled.fingerprint, notice: assembled.notice});
      const checked = auditNativeRelease(f.entry, {archive: {path: packaged.archive.path, tool: f.tool}, spawn: f.spawn, audit: fixtureAudit});
      assert.equal(checked.archiveChecked, true);
      assert.equal(checked.publicationQualified, false);
      const packageReceipt = path.join(packageInputs.output, "native-package.json"), record = JSON.parse(fs.readFileSync(packageReceipt, "utf8"));
      record.noticeAssembly.notice.sha256 = "0".repeat(64); fs.writeFileSync(packageReceipt, JSON.stringify(record));
      assert.throws(() => auditNativeRelease(f.entry, {archive: {path: packaged.archive.path, tool: f.tool}, spawn: f.spawn, audit: fixtureAudit}), /does not bind/);
    });
    test("late notice assembly binding rejects missing, malformed, mismatched and mutated inputs", t => {
      const invalidPins = /** @type {{label: string, change: (inputs: AssembledLatePackageInputs, source: string) => AssembledLatePackageInputs}[]} */ ([
        {label: "missing receipt", change: inputs => ({...inputs, receipt: {path: path.join(path.dirname(inputs.receipt.path), "missing", "native-notices.json"), sha256: "a".repeat(64)}})},
        {label: "malformed receipt hash", change: inputs => ({...inputs, receipt: {...inputs.receipt, sha256: "A".repeat(64)}})},
        {label: "wrong receipt pin", change: inputs => ({...inputs, receipt: {...inputs.receipt, sha256: "f".repeat(64)}})},
        {label: "malformed receipt body", change: inputs => {
          fs.writeFileSync(inputs.receipt.path, "not a native notice receipt\n");
          return {...inputs, receipt: {...inputs.receipt, sha256: fileHash(inputs.receipt.path)}};
        }},
        {label: "changed assembled body", change: inputs => {fs.appendFileSync(inputs.notices, "tampered\n"); return inputs;}},
        {label: "different notice path", change: (inputs, source) => {
          const directory = path.join(path.dirname(source), "same-bytes-notice"); fs.mkdirSync(directory);
          const copy = path.join(directory, noticeName); fs.copyFileSync(inputs.notices, copy);
          return {...inputs, notices: copy};
        }},
      ]);
      for (const {label, change} of invalidPins) {
        const f = fixture(t); candidateFixture(f); const {packageInputs, source} = assembledLatePackageInputs(t, f), selected = change(packageInputs, source);
        f.state.noticeForArchive = selected.notices;
        assert.throws(() => packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, selected), /./, label);
        assert.equal(f.calls.length, 0, label + " is rejected before archive commands");
        assert.equal(fs.existsSync(path.join(selected.output, "native-package.json")), false, label + " cannot write a package receipt");
      }
      const f = fixture(t); candidateFixture(f); const {packageInputs, source} = assembledLatePackageInputs(t, f);
      f.state.noticeForArchive = packageInputs.notices; f.state.edit = source;
      assert.throws(() => packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, packageInputs), /Notice input hash differs/);
      assert.equal(fs.existsSync(path.join(packageInputs.output, "native-package.json")), false, "a changed source cannot qualify a package");
    });
    test("archive admission rejects an optional notice binding that differs from the candidate", t => {
      const f = fixture(t); candidateFixture(f); bindNoticeAssembly(f);
      const packaged = f.run(), receiptPath = path.join(f.root, "native-package.json"), receipt = JSON.parse(fs.readFileSync(receiptPath, "utf8"));
      receipt.noticeAssembly.fingerprint = "0".repeat(64); fs.writeFileSync(receiptPath, JSON.stringify(receipt));
      assert.throws(() => auditNativeRelease(f.entry, {archive: {path: packaged.archive.path, tool: f.tool}, spawn: f.spawn, audit: fixtureAudit}), /does not bind the selected candidate/);
    });
  });
  test("late notice packaging supports a Windows candidate whose original entrypoint has no notice", t => {
    const f = fixture(t, true); Reflect.deleteProperty(f.receipt, "notices"); f.write(); const originalEntryHash = fileHash(f.entry);
    const inputs = latePackageInputs(t, f); f.state.noticeForArchive = inputs.notices;
    const packaged = packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, inputs);
    assert.equal(packaged.version, 2); assert.equal(packaged.originNotices, null);
    assert.equal(packaged.notices.path, path.join(inputs.output, noticeName)); assert.equal(packaged.executable.sha256, fileHash(f.binary));
    assert.equal(fileHash(f.entry), originalEntryHash);
  });
  test("late notice mode rejects nested, pre-existing and unbound output or notice paths before commands", t => {
    const f = fixture(t), inputs = latePackageInputs(t, f);
    assert.throws(() => packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, {output: path.join(f.root, "nested"), notices: f.notice}), /sibling or external/);
    fs.mkdirSync(inputs.output); assert.throws(() => packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, inputs), /fresh absolute/);
    const wrongName = path.join(f.root, "draft", "other.txt"); fs.writeFileSync(wrongName, "no");
    assert.throws(() => packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, {output: path.join(path.dirname(f.root), path.basename(f.root) + "-other"), notices: wrongName}), /named/);
    assert.equal(f.calls.length, 0);
  });
  test.describe("Linux v2 candidate admission", {skip: skipLinuxCandidateArchives}, () => {
    test("v2 archive admission refuses an altered source-notice origin binding", t => {
      const f = fixture(t); candidateFixture(f); const inputs = latePackageInputs(t, f); f.state.noticeForArchive = inputs.notices;
      const packaged = packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, inputs), receiptPath = path.join(inputs.output, "native-package.json");
      const receipt = JSON.parse(fs.readFileSync(receiptPath, "utf8")); receipt.originNotices.sha256 = "0".repeat(64);
      fs.writeFileSync(receiptPath, JSON.stringify(receipt));
      assert.throws(() => auditNativeRelease(f.entry, {archive: {path: packaged.archive.path, tool: f.tool}, spawn: f.spawn, audit: fixtureAudit}), /does not bind the selected candidate/);
    });
    test("v2 archive admission accepts a Linux candidate without an origin notice", t => {
      const f = fixture(t); Reflect.deleteProperty(f.receipt, "notices"); f.write(); candidateFixture(f);
      const inputs = latePackageInputs(t, f); f.state.noticeForArchive = inputs.notices;
      const packaged = packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, inputs);
      assert.equal(packaged.originNotices, null);
      const checked = auditNativeRelease(f.entry, {archive: {path: packaged.archive.path, tool: f.tool}, spawn: f.spawn, audit: fixtureAudit});
      assert.equal(checked.archiveChecked, true); assert.equal(checked.sha256, fileHash(f.binary));
    });
    test("v2 archive admission rejects an archive nested inside its admitted candidate directory", t => {
      const f = fixture(t); candidateFixture(f); const inputs = latePackageInputs(t, f); f.state.noticeForArchive = inputs.notices;
      const packaged = packageNativeRelease(f.entry, f.tool, "0.9.4", f.spawn, inputs), nested = path.join(f.root, "nested-package"); fs.mkdirSync(nested);
      const nestedArchive = path.join(nested, path.basename(packaged.archive.path)), nestedNotice = path.join(nested, noticeName), nestedReceipt = path.join(nested, "native-package.json");
      fs.copyFileSync(packaged.archive.path, nestedArchive); fs.copyFileSync(path.join(inputs.output, noticeName), nestedNotice);
      const receipt = JSON.parse(fs.readFileSync(path.join(inputs.output, "native-package.json"), "utf8"));
      receipt.archive.path = nestedArchive; receipt.archive.sha256 = fileHash(nestedArchive); receipt.archive.bytes = fs.statSync(nestedArchive).size;
      receipt.notices.path = nestedNotice; fs.writeFileSync(nestedReceipt, JSON.stringify(receipt));
      assert.throws(() => auditNativeRelease(f.entry, {archive: {path: nestedArchive, tool: f.tool}, spawn: f.spawn, audit: fixtureAudit}), /fresh notice package/);
    });
  });
});

test.describe("Native release archive reproducibility", () => {
  for (const windows of [false, true]) {
    test(`${windows ? "Windows ZIP" : "Linux TGZ"} repeat packaging ignores source mtimes and local timezone`, {
      skip: !process.env["MOM_TEST_CMAKE"] && "Set MOM_TEST_CMAKE to exercise the real archive tool",
    }, async t => {
      const timezone = process.env["TZ"];
      t.after(() => {if (timezone === undefined) {delete process.env["TZ"];} else {process.env["TZ"] = timezone;}});
      const cmake = process.env["MOM_TEST_CMAKE"] || "", first = fixture(t, windows), second = fixture(t, windows),
        times = [new Date("2001-02-03T04:05:06Z"), new Date("2025-07-08T09:10:12Z")],
        accesses = [new Date("2031-02-03T04:05:06Z"), new Date("2041-07-08T09:10:12Z")];
      assert.ok(path.isAbsolute(cmake), "The real CMake tool must be selected explicitly");
      const archives = [];
      for (const [index, f] of [first, second].entries()) {
        // Separate ZIP passes beyond DOS timestamps' two-second resolution. Otherwise
        // an accidentally retained current-time/ctime field can match by coincidence.
        if (windows && index) {await delay(2100);}
        const time = times[index], access = accesses[index]; assert.ok(time && access);
        // Future atime > mtime/ctime survives ordinary reads on relatime filesystems.
        // Back-to-back fixtures otherwise hide ZIP's current-time atime/ctime fields.
        fs.utimesSync(f.binary, access, time); fs.utimesSync(f.notice, access, time);
        const metadata = () => [f.binary, f.notice].map(file => {
            const stat = fs.statSync(file); return {mtime: stat.mtimeMs, atime: stat.atimeMs};
          }), before = metadata();
        assert.ok(before.every(stat => Math.abs(stat.atime - access.getTime()) < 2000 && Math.abs(stat.mtime - time.getTime()) < 2000),
          "The filesystem must preserve the deliberately different fixture timestamps");
        process.env["TZ"] = index ? "America/Los_Angeles" : "UTC";
        const result = packageNativeRelease(f.entry, cmake, "0.9.4");
        // Reads may update atime on strictatime filesystems; only explicit metadata
        // rewriting is forbidden. The effective pre-pack atimes above still differ.
        assert.deepEqual(metadata().map(stat => stat.mtime), before.map(stat => stat.mtime),
          "Reproducible metadata must not rewrite the original inputs");
        archives.push(result.archive);
      }
      assert.equal(archives[0]?.sha256, archives[1]?.sha256,
        "Identical executable/notice bytes must produce byte-identical archives in fresh directories");
    });
  }
});
