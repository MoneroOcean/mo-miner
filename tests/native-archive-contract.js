"use strict";

const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), assert = require("node:assert/strict"), test = require("node:test");
const {checkNativeRelease, auditNativeRelease, suiteTimeoutMs} = require("../scripts/check-native-release");
const {fileHash} = require("../scripts/build-source-inputs");
const {packageNativeRelease} = require("../scripts/package-native-release");
const {hashTests} = require("./vectors");
const {spawnSync} = require("node:child_process");
const {tempDir} = require("./common/temp-dir");
const {setStableMtime, transientRewriteRestore} = require("./common/file-times");

const noticeName = "THIRD-PARTY-NOTICES.txt", executableName = "mom";
const hostPlatform = process.platform === "win32" ? "win32" : "linux";
const cpuVectorCount = hashTests.filter(definition => !definition.gpu).length;
const successfulCpuSummary = `ℹ tests ${cpuVectorCount}\nℹ suites 1\nℹ pass ${cpuVectorCount}\nℹ fail 0\nℹ cancelled 0\nℹ skipped 0\n`;

/** Structural candidate/archive fixture; all process execution is stubbed.
 * @param {import('node:test').TestContext} t */
function fixture(t, platform = hostPlatform) {
  const root = tempDir(t, "mom-native-archive-contract-");
  const windows = platform === "win32", archiveExecutableName = windows ? "mom.exe" : executableName;
  const executable = path.join(root, archiveExecutableName), notice = path.join(root, noticeName), config = path.join(root, "config.json"),
    producerFile = path.join(root, "result.json"), entrypoint = path.join(root, "entrypoint.json"), tool = path.join(root, "cmake"),
    archive = path.join(root, windows ? "mom-v0.9.4-win.zip" : "mom-v0.9.4-lin.tgz"), packageFile = path.join(root, "native-package.json");
  const image = Buffer.alloc(windows ? 4096 : 128);
  if (windows) {
    image.writeUInt16LE(0x5a4d, 0); image.writeUInt32LE(0x80, 0x3c); image.write("PE\0\0", 0x80, "binary");
    image.writeUInt16LE(0x8664, 0x84); image.writeUInt16LE(1, 0x86); image.writeUInt16LE(240, 0x94); image.writeUInt16LE(0x20b, 0x98);
  } else {
    image.write("\x7fELF", 0, "binary"); image[4] = 2; image[5] = 1; image[6] = 1;
    image.writeUInt16LE(3, 16); image.writeUInt16LE(62, 18); image.writeUInt32LE(1, 20); image.writeUInt16LE(64, 52);
    image.writeUInt16LE(56, 54); image.writeUInt16LE(1, 56);
  }
  fs.writeFileSync(executable, image, {mode: 0o755}); fs.writeFileSync(notice, "Fixture notice.\n");
  fs.writeFileSync(config, JSON.stringify({version: 1})); fs.writeFileSync(tool, "trusted tool fixture"); fs.writeFileSync(archive, "archive fixture");
  const producer = {configHash: fileHash(config), buildOnly: true, publicationQualified: false,
    ...(windows ? {target: "x86_64-pc-windows-msvc", artifact: {path: executable, bytes: fs.statSync(executable).size, sha256: fileHash(executable)}} :
      {binary: executable, bytes: fs.statSync(executable).size, sha256: fileHash(executable)})};
  fs.writeFileSync(producerFile, JSON.stringify(producer));
  const candidateNotices = {path: notice, sha256: fileHash(notice), bytes: fs.statSync(notice).size};
  const candidate = {version: 1, target: platform, buildOnly: true, publicationQualified: false, configFile: config,
    configHash: fileHash(config), componentHashes: {[producerFile]: fileHash(producerFile)},
    producerReceipt: {path: producerFile, sha256: fileHash(producerFile)}, notices: candidateNotices,
    source: {path: executable, sha256: fileHash(executable)},
    artifact: {path: executable, bytes: fs.statSync(executable).size, sha256: fileHash(executable)}};
  fs.writeFileSync(entrypoint, JSON.stringify(candidate));
  const packageReceipt = {version: 1, entrypoint, entryHash: fileHash(entrypoint), executable: candidate.artifact,
    tool: {path: tool, sha256: fileHash(tool)}, archive: {path: archive, bytes: fs.statSync(archive).size, sha256: fileHash(archive)},
    notices: candidateNotices, members: [archiveExecutableName, noticeName], exactOriginalBytes: true, buildOnly: true, publicationQualified: false};
  const writePackage = () => fs.writeFileSync(packageFile, JSON.stringify(packageReceipt)); writePackage();
  const state = /** @type {{listing: string[], types: string[], corrupt: string, mutateArchive: boolean, mutationAfterTf: string, crlfListings: boolean, toolCommands: Set<string>}} */ ({
    listing: [archiveExecutableName, noticeName], types: [`-rw-r--r-- 0 fixture fixture ${fs.statSync(executable).size} Jun 1 00:00 ${archiveExecutableName}`,
      `-rw-r--r-- 0 fixture fixture ${fs.statSync(notice).size} Jun 1 00:00 ${noticeName}`], corrupt: "", mutateArchive: false, mutationAfterTf: "", crlfListings: false, toolCommands: new Set([tool])});
  /** @type {{command: string, args: string[], options: {cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}}[]} */
  const calls = [];
  /** @type {typeof import('node:child_process').spawnSync} */
  const spawn = /** @type {typeof import('node:child_process').spawnSync} */ (/** @type {unknown} */ (
    (/** @type {string} */ command, /** @type {string[]} */ args, /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      calls.push({command, args, options});
      if (state.toolCommands.has(command)) {
        const operation = args[2];
        if (operation === "cf" || operation === "czf") {fs.writeFileSync(args[3] || "", "archive fixture"); return {status: 0, stdout: "", stderr: ""};}
        if (operation === "tf") {
          const stdout = state.listing.join(state.crlfListings ? "\r\n" : "\n") + (state.crlfListings ? "\r\n" : "\n");
          if (state.mutationAfterTf === "archive") {fs.appendFileSync(archive, "changed");}
          if (state.mutationAfterTf === "receipt") {fs.appendFileSync(packageFile, "changed");}
          if (state.mutationAfterTf === "tool") {fs.appendFileSync(command, "changed");}
          return {status: 0, stdout, stderr: ""};
        }
        if (operation === "tvf") {return {status: 0, stdout: state.types.join(state.crlfListings ? "\r\n" : "\n") + (state.crlfListings ? "\r\n" : "\n"), stderr: ""};}
        assert.equal(operation, "xf"); assert.ok(options.cwd);
        fs.copyFileSync(executable, path.join(options.cwd, archiveExecutableName));
        fs.copyFileSync(notice, path.join(options.cwd, noticeName));
        if (state.corrupt === "link") {
          fs.unlinkSync(path.join(options.cwd, noticeName)); fs.linkSync(path.join(options.cwd, archiveExecutableName), path.join(options.cwd, noticeName));
        } else if (state.corrupt === "symlink") {
          fs.unlinkSync(path.join(options.cwd, noticeName)); fs.symlinkSync(archiveExecutableName, path.join(options.cwd, noticeName));
        } else if (state.corrupt === "setuid") {
          fs.chmodSync(path.join(options.cwd, archiveExecutableName), 0o4755);
        }
        return {status: 0, stdout: "", stderr: ""};
      }
      if (command === process.execPath) {
        assert.ok(options.env);
        if (args[0] !== path.join(path.dirname(require.resolve("./vectors")), "run-hash.js")) {
          const directive = args.at(-1), output = directive === "invalid-directive" ? "Unknown directive invalid-directive\n--save_config:\n" : "No directive specified\n--save_config:\n";
          assert.equal(options.env["MOM_TEST_NO_POOL_NETWORK"], "1");
          return {status: 1, stdout: output, stderr: ""};
        }
        assert.equal(options.timeout, suiteTimeoutMs);
        assert.equal(path.basename(options.env["MOM_TEST_EXECUTABLE"] || ""), archiveExecutableName);
        assert.equal(options.env["MOM_TEST_NO_POOL_NETWORK"], "1");
        assert.equal(options.env["MOM_SKIP_MSR"], "1"); assert.equal(options.env["MOM_GPU_BACKEND"], "");
        if (state.mutateArchive) {fs.appendFileSync(archive, "changed");}
        return {status: 0, stdout: successfulCpuSummary, stderr: ""};
      }
      if (path.basename(command) === archiveExecutableName) {
        const directive = args.at(-1), output = directive === "invalid-directive" ? "Unknown directive invalid-directive\n--save_config:\n" : "No directive specified\n--save_config:\n";
        assert.equal(options.env?.["MOM_TEST_NO_POOL_NETWORK"], "1");
        assert.equal(options.env?.["MOM_COMMAND"], "./mom");
        return {status: 1, stdout: output, stderr: ""};
      }
      throw new Error("Unexpected command: " + command);
    }));
  return {root, executable, notice, config, producerFile, entrypoint, tool, archive, archiveExecutableName, packageFile, candidate, candidateNotices,
    packageReceipt, writePackage, state, calls, spawn};
}

test.describe("Flat native archive qualification", () => {
  test("CPU vectors run against the extracted member with pool networking disabled", t => {
    const f = fixture(t), platform = f.candidate.target, result = checkNativeRelease(f.entrypoint, "cpu", {platform, arch: "x64", audit: () => undefined,
      archive: {path: f.archive, tool: f.tool}, spawn: f.spawn,
      env: {MOM_TEST_EXECUTABLE: "/old/mom", MOM_TEST_NO_POOL_NETWORK: "0", NODE_OPTIONS: "--require=untrusted.js", LD_LIBRARY_PATH: "/sdk/lib"}});
    assert.equal(result.suitePassed, true); assert.equal(result.archive, f.archive); assert.equal(result.executable, f.executable);
    assert.equal(f.calls.length, platform === "win32" ? 5 : 4); assert.deepEqual(f.calls.slice(0, 3).map(call => call.args[2]), ["tf", "tvf", "xf"]);
    const run = f.calls.at(-1); assert.equal(run?.command, process.execPath); assert.deepEqual(run?.args, [path.join(path.dirname(require.resolve("./vectors")), "run-hash.js"), "cpu"]);
    assert.equal(run?.options.env?.["MOM_TEST_NO_POOL_NETWORK"], "1");
    for (const key of ["NODE_OPTIONS", "LD_LIBRARY_PATH"]) {assert.equal(Object.hasOwn(run?.options.env || {}, key), false);}
    assert.equal(fs.existsSync(path.dirname(run?.options.env?.["MOM_TEST_EXECUTABLE"] || "")), false, "private extraction is removed after qualification");
  });

  test("Windows PE candidates use the flat ZIP member names and CRLF listing format", t => {
    const f = fixture(t, "win32"); f.state.crlfListings = true;
    f.packageReceipt.tool.path = f.tool.toUpperCase(); f.writePackage();
    const result = checkNativeRelease(f.entrypoint, "cpu", {platform: "win32", arch: "x64", audit: () => undefined,
      archive: {path: f.archive, tool: f.tool}, spawn: f.spawn});
    assert.equal(result.suitePassed, true); assert.equal(result.archive, f.archive); assert.equal(result.executable, f.executable);
    assert.deepEqual(f.calls.slice(0, 3).map(call => call.args[2]), ["tf", "tvf", "xf"]);
    assert.deepEqual(f.calls[2]?.args.slice(-2), ["mom.exe", noticeName]);
    assert.equal(f.calls.length, 5);
    const cold = f.calls[3]; assert.equal(path.basename(cold?.command || ""), "mom.exe");
    assert.deepEqual(cold?.args, ["invalid-directive"]);
    assert.equal(result.coldStartPreflight?.executable, cold?.command);
    const run = f.calls[4]; assert.equal(run?.command, process.execPath);
    assert.equal(path.basename(run?.options.env?.["MOM_TEST_EXECUTABLE"] || ""), "mom.exe");
  });

  test("complete CLI parity also launches the extracted flat member", t => {
    const f = fixture(t), platform = f.candidate.target, result = checkNativeRelease(f.entrypoint, "cli", {platform, arch: "x64", audit: () => undefined,
      archive: {path: f.archive, tool: f.tool}, spawn: f.spawn});
    assert.equal(result.suitePassed, true); assert.equal(result.archive, f.archive);
    const nativeRuns = f.calls.filter(call => path.basename(call.command) === f.archiveExecutableName &&
      call.options.env?.["MOM_TEST_EXECUTABLE"]?.startsWith(os.tmpdir()));
    assert.equal(nativeRuns.length, platform === "win32" ? 3 : 2); assert.ok(nativeRuns.every(call => call.options.env?.["MOM_TEST_EXECUTABLE"]?.startsWith(os.tmpdir())));
    assert.ok(nativeRuns.every(call => call.options.env?.["MOM_TEST_NO_POOL_NETWORK"] === "1"));
  });

  test("traversal, duplicate and foreign members fail before extraction", t => {
    const f = fixture(t), platform = f.candidate.target, name = f.archiveExecutableName;
    for (const listing of [[name, "../outside"], [name, name], [name, "README.md"]]) {
      f.state.listing = listing;
      assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform, arch: "x64", audit: () => undefined,
        archive: {path: f.archive, tool: f.tool}, spawn: f.spawn}), /two members|foreign|flat/);
      assert.deepEqual(f.calls.map(call => call.args[2]), ["tf"]);
      f.calls.length = 0;
    }
  });

  for (const [kind, prefix] of [["symlink", "l"], ["hard link", "h"]]) {
    test(`${kind} members are rejected from verbose listing before extraction`, t => {
      const f = fixture(t), platform = f.candidate.target; f.state.types[1] = prefix + "rw-r--r-- 0 fixture fixture 0 Jun 1 00:00 " + noticeName + (prefix === "h" ? " link to " + f.archiveExecutableName : " -> " + f.archiveExecutableName);
      assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform, arch: "x64", audit: () => undefined,
        archive: {path: f.archive, tool: f.tool}, spawn: f.spawn}), /regular files/);
      assert.deepEqual(f.calls.map(call => call.args[2]), ["tf", "tvf"]);
    });
  }

  test("oversized declared members fail before extraction", t => {
    const f = fixture(t), platform = f.candidate.target, size = String(fs.statSync(f.executable).size);
    f.state.types[0] = f.state.types[0]?.replace(` ${size} `, " 1048576 ") || "";
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform, arch: "x64", audit: () => undefined,
      archive: {path: f.archive, tool: f.tool}, spawn: f.spawn}), /declared member size/);
    assert.deepEqual(f.calls.map(call => call.args[2]), ["tf", "tvf"]);
  });

  test("hard-linked or symlinked extracted files are rejected", t => {
    for (const kind of ["link", "symlink"]) {
      const f = fixture(t), platform = f.candidate.target; f.state.corrupt = kind;
      assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform, arch: "x64", audit: () => undefined,
        archive: {path: f.archive, tool: f.tool}, spawn: f.spawn}), /hard-linked|regular file/);
      assert.equal(f.calls.some(call => call.command === process.execPath), false);
    }
  });

  test("setuid executable members are rejected before vector launch", {skip: process.platform === "win32"}, t => {
    for (const location of ["listing", "extracted"]) {
      const f = fixture(t);
      if (location === "listing") {f.state.types[0] = f.state.types[0]?.replace(/^-[rwxStTs-]{9}/, "-rwsr-xr-x") || "";} else {f.state.corrupt = "setuid";}
      assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: "linux", arch: "x64", audit: () => undefined,
        archive: {path: f.archive, tool: f.tool}, spawn: f.spawn}), /permission bits/);
      assert.equal(f.calls.some(call => call.command === process.execPath), false);
      assert.equal(f.calls.some(call => call.args[2] === "xf"), location === "extracted");
    }
  });

  test("CMake tool symlinks are admitted only through the caller-selected resolved target", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), selectedTool = path.join(f.root, "cmake-selected");
    fs.unlinkSync(f.archive); fs.unlinkSync(f.packageFile); fs.symlinkSync(f.tool, selectedTool); f.state.toolCommands.add(selectedTool);
    const packaged = packageNativeRelease(f.entrypoint, selectedTool, "0.9.4", f.spawn);
    assert.equal(packaged.tool.path, selectedTool); assert.equal(packaged.tool.sha256, fileHash(f.tool));
    const result = checkNativeRelease(f.entrypoint, "cpu", {platform: f.candidate.target, arch: "x64", audit: () => undefined,
      archive: {path: f.archive, tool: selectedTool}, spawn: f.spawn});
    assert.equal(result.suitePassed, true); assert.equal(result.archive, f.archive);
    assert.ok(f.calls.some(call => call.command === selectedTool));
    assert.ok(f.calls.some(call => call.command === fs.realpathSync(selectedTool)));
  });

  test("qualification uses the caller's tool, not the original build-only packaging tool", t => {
    const f = fixture(t), selected = path.join(f.root, "qualification-cmake");
    fs.writeFileSync(selected, "independently selected qualification tool"); f.state.toolCommands.add(selected);
    // Producer paths are historical data and need not exist on the clean test host.
    f.packageReceipt.tool.path = "C:\\build-only\\cmake.exe"; f.writePackage();
    const original = fs.readFileSync(f.packageFile);
    const result = checkNativeRelease(f.entrypoint, "cpu", {platform: f.candidate.target, arch: "x64", audit: () => undefined,
      archive: {path: f.archive, tool: selected}, spawn: f.spawn});
    assert.equal(result.suitePassed, true);
    assert.ok(f.calls.slice(0, 3).every(call => call.command === selected));
    assert.deepEqual(fs.readFileSync(f.packageFile), original, "qualification must not restamp producer metadata");
  });

  test("real CMake archives can be qualified through a different tool installation path", {skip: process.platform !== "linux" || !fs.existsSync("/usr/bin/cmake")}, t => {
    const f = fixture(t), selected = path.join(f.root, "qualification-cmake");
    fs.unlinkSync(f.archive); fs.unlinkSync(f.packageFile); fs.symlinkSync("/usr/bin/cmake", selected);
    packageNativeRelease(f.entrypoint, "/usr/bin/cmake", "0.9.4");
    const original = fs.readFileSync(f.packageFile);
    // Exercise real create/list/extract commands, but never execute the structural ELF fixture.
    const run = /** @type {typeof spawnSync} */ (/** @type {unknown} */ (
      (/** @type {string} */ command, /** @type {string[]} */ args, /** @type {import("node:child_process").SpawnSyncOptions} */ options) =>
        command === process.execPath ? f.spawn(command, args, options) : spawnSync(command, args, options)));
    const result = checkNativeRelease(f.entrypoint, "cpu", {platform: "linux", arch: "x64", audit: () => undefined,
      archive: {path: f.archive, tool: selected}, spawn: run});
    assert.equal(result.suitePassed, true);
    assert.deepEqual(fs.readFileSync(f.packageFile), original);
  });

  test("malformed packaging-tool provenance fails before any tool or candidate launch", t => {
    for (const change of [{path: "relative-cmake"}, {path: "/tool\n"}, {sha256: "not-a-hash"}]) {
      const f = fixture(t); Object.assign(f.packageReceipt.tool, change); f.writePackage();
      assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: f.candidate.target, arch: "x64", audit: () => undefined,
        archive: {path: f.archive, tool: f.tool}, spawn: f.spawn}), /does not bind|provenance/);
      assert.equal(f.calls.length, 0);
    }
  });

  test("a different qualification tool still cannot change during archive inspection", t => {
    const f = fixture(t), selected = path.join(f.root, "qualification-cmake");
    fs.writeFileSync(selected, "independently selected qualification tool"); f.state.toolCommands.add(selected);
    f.state.mutationAfterTf = "tool";
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: f.candidate.target, arch: "x64", audit: () => undefined,
      archive: {path: f.archive, tool: selected}, spawn: f.spawn}), /tool changed during qualification/);
    assert.deepEqual(f.calls.map(call => call.command), [selected]);
  });

  test("archive, receipt or tool mutation after listing fails before extraction", t => {
    for (const item of ["archive", "receipt", "tool"]) {
      const f = fixture(t); f.state.mutationAfterTf = item;
      assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: f.candidate.target, arch: "x64", audit: () => undefined,
        archive: {path: f.archive, tool: f.tool}, spawn: f.spawn}), /changed during qualification/);
      assert.deepEqual(f.calls.map(call => call.args[2]), ["tf"]);
    }
  });

  test("archive mutation during the suite invalidates an otherwise passing vector summary", t => {
    const f = fixture(t); f.state.mutateArchive = true;
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: f.candidate.target, arch: "x64", audit: () => undefined,
      archive: {path: f.archive, tool: f.tool}, spawn: f.spawn}), /file changed during qualification/);
    assert.equal(f.calls.some(call => call.command === process.execPath), true);
  });
});

test.describe("Build-only candidate and archive audit", () => {
  test("audits a Windows archive on Linux without launching the candidate or rewriting receipts", t => {
    const f = fixture(t, "win32"), stages = /** @type {string[]} */ ([]), originalEntry = fs.readFileSync(f.entrypoint), originalPackage = fs.readFileSync(f.packageFile);
    const spawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      if (args[2] === "xf") {stages.push(options.cwd || "");}
      return f.spawn(command, args, options);
    }));
    const auditTargets = /** @type {string[]} */ ([]);
    const result = auditNativeRelease(f.entrypoint, {audit: (raw, target) => {
      assert.equal(raw, f.executable); auditTargets.push(target);
    }, archive: {path: f.archive, tool: f.tool}, spawn});
    assert.equal(result.scope, "build-audit"); assert.equal(result.target, "win32");
    assert.equal(result.buildEvidencePassed, true); assert.equal(result.runtimeTested, false);
    assert.equal(result.executable, f.executable); assert.equal(result.sha256, fileHash(f.executable));
    assert.equal(result.sourceSha256, fileHash(f.executable)); assert.equal(result.bytes, fs.statSync(f.executable).size);
    assert.equal(result.archive, f.archive); assert.equal(result.archiveSha256, fileHash(f.archive));
    assert.deepEqual(auditTargets, ["win32"]); assert.equal(f.calls.length, 3);
    assert.deepEqual(f.calls.map(call => call.args[2]), ["tf", "tvf", "xf"]);
    assert.equal(f.calls.some(call => call.command === process.execPath), false, "audit mode must not launch vectors");
    assert.deepEqual(fs.readFileSync(f.entrypoint), originalEntry); assert.deepEqual(fs.readFileSync(f.packageFile), originalPackage);
    assert.equal(stages.length, 1); assert.equal(fs.existsSync(stages[0] || ""), false, "audit extraction must be removed on success");
  });

  test("archive mutation fails closed before extraction and never launches vectors", t => {
    const f = fixture(t, "win32"); f.state.mutationAfterTf = "archive";
    assert.throws(() => auditNativeRelease(f.entrypoint, {audit: () => undefined,
      archive: {path: f.archive, tool: f.tool}, spawn: f.spawn}), /changed during qualification/);
    assert.deepEqual(f.calls.map(call => call.args[2]), ["tf"]);
    assert.equal(f.calls.some(call => call.command === process.execPath), false);
  });

  test("archive and tool transient rewrites remain detectable after restoring bytes and mtime", t => {
    for (const kind of ["archive", "tool"]) {
      const f = fixture(t, "win32"), file = kind === "archive" ? f.archive : f.tool;
      setStableMtime(file);
      const beforeHash = fileHash(file), before = fs.statSync(file, {bigint: true});
      const spawn = /** @type {typeof f.spawn} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
        /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
        if (args[2] === "tf") {transientRewriteRestore(file);}
        return f.spawn(command, args, options);
      });
      assert.throws(() => auditNativeRelease(f.entrypoint, {audit: () => undefined,
        archive: {path: f.archive, tool: f.tool}, spawn}), /changed during qualification/i);
      const after = fs.statSync(file, {bigint: true});
      assert.equal(fileHash(file), beforeHash, `${kind} bytes are restored`);
      assert.equal(after.mtimeNs, before.mtimeNs, `${kind} mtime is restored`);
      assert.notEqual(after.ctimeNs, before.ctimeNs, `${kind} ctime still records the rewrite`);
      assert.deepEqual(f.calls.map(call => call.args[2]), ["tf"]);
      assert.equal(f.calls.some(call => call.command === process.execPath), false);
    }
  });
});
