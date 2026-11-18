"use strict";

const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const assert = require("node:assert/strict"), test = require("node:test");
const {checkNativeRelease, auditNativeRelease, checkNativeExecutable, suiteTimeoutMs} = require("../scripts/check-native-release");
const {fileHash} = require("../scripts/build-source-inputs");
const gpuTuning = require("../gpu-tuning");
const {hashTests} = require("../tests/vectors");
const {nativePeFixture} = require("./common/native-pe-fixture");
const {sealNativePePages} = require("../scripts/native-pe-pages");
const {packNativeEntrypoint} = require("../scripts/build-release");
const {openclSyclEnv} = require("./common/gpu-test-modes");
const cpuVectorCount = hashTests.filter(definition => !definition.gpu).length;
const portableCpuVectors = hashTests.filter(definition => definition.syclCpu);
const portableCpuAlgos = new Set(portableCpuVectors.map(definition => definition.job.algo));
const portableCpuTestCount = portableCpuVectors.length + portableCpuAlgos.size * 2 + 1;
const successfulCpuSummary = `ℹ tests ${cpuVectorCount}\nℹ suites 1\nℹ pass ${cpuVectorCount}\nℹ fail 0\nℹ cancelled 0\nℹ skipped 0\n`;
const hostTarget = process.platform === "win32" ? "win32" : "linux";
const syntheticTargets = /** @type {("linux" | "win32")[]} */ (process.platform === "win32" ? ["win32"] : ["linux", "win32"]);

/** @typedef {{path: string, sha256: string}} FixtureBinding */
/** @typedef {{path: string, bytes: number, sha256: string}} FixtureArtifact */
/** @typedef {{version: number, graph: Record<string, unknown>}} FixtureRecipe */
/** @typedef {{version: number, target: "linux" | "win32", buildOnly: true, publicationQualified: false,
 * configFile: string, configHash: string, componentHashes: Record<string, string>, producerReceipt: FixtureBinding,
 * packingReceipt: FixtureBinding | undefined, generatedDescriptor: FixtureBinding | undefined,
 * notices: FixtureArtifact | undefined,
 * noticeAssembly?: {receiptPath: string, receiptSha256: string, fingerprint: string, notice: FixtureArtifact},
 * source: FixtureBinding & {bytes?: number}, artifact: FixtureArtifact, recipe?: FixtureRecipe,
 * sourceEntrypoint?: FixtureBinding}} FixtureRecord */
/** @typedef {{configHash: string, buildOnly: boolean, publicationQualified: boolean, target?: string,
 * artifact?: FixtureArtifact, binary?: string, bytes?: number, sha256?: string, nativeRuntime?: boolean, executed?: boolean}} FixtureProducer */
/** @typedef {{source: FixtureArtifact, artifact: FixtureArtifact, restoredSha256: string,
 * payloadAudit: {exactOriginalBytes: boolean}, logs: Record<string, FixtureBinding>,
 * buildOnly: boolean, publicationQualified: boolean}} FixturePacking */

/** A structural fixture only. No candidate or hash vector is ever executed here.
 * @param {import('node:test').TestContext} t @param {"linux" | "win32"} [target] @param {boolean} [packed] */
function fixture(t, target = hostTarget, packed = false) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-qualification-")), name = target === "win32" ? "mom.exe" : "mom",
    executable = path.join(root, name), sourceDirectory = root + ".unpacked", source = packed ? path.join(sourceDirectory, name) : executable,
    entrypoint = path.join(root, "entrypoint.json");
  t.after(() => {fs.rmSync(root, {recursive: true, force: true}); fs.rmSync(sourceDirectory, {recursive: true, force: true});});
  /** @param {string} output */
  const writeExecutable = output => {
    if (target === "win32") {
      const data = nativePeFixture().bytes; fs.writeFileSync(output, data, {mode: 0o755}); sealNativePePages(output);
    } else {
      const data = Buffer.alloc(128); data.write("\x7fELF", 0, "binary"); data[4] = 2; data[5] = 1; data[6] = 1;
      data.writeUInt16LE(3, 16); data.writeUInt16LE(62, 18); data.writeUInt32LE(1, 20); data.writeUInt16LE(64, 52);
      data.writeUInt16LE(56, 54); data.writeUInt16LE(1, 56); fs.writeFileSync(output, data, {mode: 0o755});
    }
  };
  if (packed) {fs.mkdirSync(sourceDirectory); writeExecutable(source);}
  writeExecutable(executable);
  const configFile = path.join(root, "config.json"), producerFile = path.join(path.dirname(source), "result.json");
  fs.writeFileSync(configFile, JSON.stringify({version: 1}));
  const producer = /** @type {FixtureProducer} */ ({configHash: fileHash(configFile), buildOnly: true, publicationQualified: false,
    ...(target === "win32" ? {target: "x86_64-pc-windows-msvc", artifact: {path: source, bytes: fs.statSync(source).size, sha256: fileHash(source)}} :
      {binary: source, bytes: fs.statSync(source).size, sha256: fileHash(source)})});
  fs.writeFileSync(producerFile, JSON.stringify(producer));
  const packingFile = path.join(root, "result.json"), logs = Object.fromEntries(["pack", "integrity", "restore"].map(stage => {
    const file = path.join(root, stage + ".log");
    if (packed) {fs.writeFileSync(file, stage + " successful\n");}
    return [stage, {path: file, sha256: packed ? fileHash(file) : ""}];
  }));
  const packing = /** @type {FixturePacking} */ ({source: {path: source, bytes: fs.statSync(source).size, sha256: fileHash(source)},
    artifact: {path: executable, bytes: fs.statSync(executable).size, sha256: fileHash(executable)},
    restoredSha256: fileHash(source), payloadAudit: {exactOriginalBytes: true}, logs, buildOnly: true, publicationQualified: false});
  if (packed) {fs.writeFileSync(packingFile, JSON.stringify(packing));}
  const record = /** @type {FixtureRecord} */ ({version: 1, target, buildOnly: true, publicationQualified: false,
    configFile, configHash: fileHash(configFile), componentHashes: {[producerFile]: fileHash(producerFile)},
    producerReceipt: {path: producerFile, sha256: fileHash(producerFile)},
    packingReceipt: packed ? {path: packingFile, sha256: fileHash(packingFile)} : undefined,
    generatedDescriptor: /** @type {{path: string, sha256: string} | undefined} */ (undefined),
    notices: /** @type {{path: string, sha256: string, bytes: number} | undefined} */ (undefined),
    source: {path: source, sha256: fileHash(source)},
    artifact: {path: executable, bytes: fs.statSync(executable).size, sha256: fileHash(executable)}});
  const write = () => fs.writeFileSync(entrypoint, JSON.stringify(record));
  write();
  const calls = /** @type {{command: string, args: string[], options: {cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}}[]} */ ([]);
  /** @type {typeof import('node:child_process').spawnSync} */
  const spawn = /** @type {typeof import('node:child_process').spawnSync} */ (/** @type {unknown} */ (
    (/** @type {string} */ command, /** @type {string[]} */ args, /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      calls.push({command, args, options});
      if (target === "win32" && args.length === 1 && args[0] === "invalid-directive") {
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
          stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      return {status: 0, signal: null, error: undefined, stdout: successfulCpuSummary, stderr: ""};
    }));
  return {root, executable, source, entrypoint, record, write, calls, spawn, producerFile, producer, packingFile, packing};
}

/** Use the real producer handoff API with a mock UPX process; neither candidate is launched.
 * @param {import('node:test').TestContext} t @param {"linux" | "win32"} target */
function transportedFixture(t, target) {
  const original = fixture(t, target), name = target === "win32" ? "mom.exe" : "mom",
    packedDirectory = original.root + ".packed", sourceDirectory = original.root + ".transported";
  fs.mkdirSync(sourceDirectory);
  t.after(() => {fs.rmSync(packedDirectory, {recursive: true, force: true}); fs.rmSync(sourceDirectory, {recursive: true, force: true});});
  original.record.recipe = {version: 1, graph: {fixture: target}};
  if (target === "win32") {
    const pe = nativePeFixture(), data = pe.bytes;
    data.writeUInt32LE(0x40000040, pe.table + 40 + 36);
    data.writeUInt32LE(8192, pe.optional + 120); data.writeUInt32LE(40, pe.optional + 124);
    data.writeUInt32LE(8256, 8192); data.writeUInt32LE(8352, 8204); data.writeUInt32LE(8288, 8208);
    data.writeBigUInt64LE(8320n, 8256); data.writeBigUInt64LE(8320n, 8288);
    data.write("ExitProcess\0", 8322); data.write("KERNEL32.dll\0", 8352);
    fs.writeFileSync(original.executable, data); sealNativePePages(original.executable);
    original.producer.artifact = {path: original.executable, bytes: fs.statSync(original.executable).size, sha256: fileHash(original.executable)};
    original.producer.nativeRuntime = true; original.producer.executed = false;
  }
  if (target === "linux") {
    original.producer.binary = original.executable;
    original.producer.bytes = fs.statSync(original.executable).size;
    original.producer.sha256 = fileHash(original.executable);
  }
  fs.writeFileSync(original.producerFile, JSON.stringify(original.producer));
  original.record.source.sha256 = fileHash(original.executable);
  original.record.artifact = {path: original.executable, bytes: fs.statSync(original.executable).size, sha256: fileHash(original.executable)};
  original.record.producerReceipt.sha256 = fileHash(original.producerFile);
  original.record.componentHashes[original.producerFile] = original.record.producerReceipt.sha256;
  original.write();

  const source = path.join(sourceDirectory, name), producerFile = path.join(sourceDirectory, "result.json"),
    executable = path.join(packedDirectory, name), packingFile = path.join(packedDirectory, "result.json"),
    entrypoint = path.join(packedDirectory, "entrypoint.json"), upx = path.join(original.root, "fixture-upx");
  fs.copyFileSync(original.executable, source); fs.chmodSync(source, fs.statSync(original.executable).mode & 0o777);
  fs.copyFileSync(original.producerFile, producerFile); fs.writeFileSync(upx, "mock pinned UPX\n", {mode: 0o755});
  const pack = /** @type {typeof import('node:child_process').spawnSync} */ (/** @type {unknown} */ ((
    /** @type {string} */ command, /** @type {string[]} */ args) => {
    assert.equal(command, upx);
    if (args.includes("--best")) {
      const output = args[args.indexOf("-o") + 1], input = args.at(-1);
      if (typeof output !== "string" || typeof input !== "string") {throw new Error("Mock pack invocation lacks explicit paths");}
      fs.copyFileSync(input, output); fs.appendFileSync(output, "packed fixture bytes");
    } else if (args.includes("-d")) {
      const output = args[args.indexOf("-o") + 1];
      if (typeof output !== "string") {throw new Error("Mock restore invocation lacks an output path");}
      fs.copyFileSync(source, output);
    } else {assert.ok(args.includes("-t"), "only pack, integrity and restore stages are expected");}
    return {status: 0, signal: null, error: undefined, stdout: "", stderr: ""};
  }));
  const packed = packNativeEntrypoint({entrypointPath: original.entrypoint, entrypointSha256: fileHash(original.entrypoint),
    sourcePath: source, sourceSha256: fileHash(source), producerReceiptPath: producerFile, target, output: packedDirectory,
    upx, upxSha256: fileHash(upx)}, pack);
  assert.equal(packed.entrypoint, entrypoint); assert.equal(packed.source.path, source);
  const record = /** @type {FixtureRecord} */ (JSON.parse(fs.readFileSync(entrypoint, "utf8"))),
    packing = /** @type {FixturePacking} */ (JSON.parse(fs.readFileSync(packingFile, "utf8"))),
    calls = /** @type {{command: string, args: string[], options: {cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}}[]} */ ([]),
    spawn = /** @type {typeof original.spawn} */ (/** @type {unknown} */ ((
      /** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      calls.push({command, args, options});
      if (target === "win32" && command === executable && args.length === 1 && args[0] === "invalid-directive") {
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
          stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      return {status: 0, signal: null, error: undefined, stdout: successfulCpuSummary, stderr: ""};
    }));
  return {...original, executable, source, entrypoint, record, producerFile, producer: original.producer,
    packingFile, packing, originEntrypoint: original.entrypoint, originRecord: original.record, calls, spawn,
    write: () => fs.writeFileSync(entrypoint, JSON.stringify(record))};
}

const stableFileTime = new Date(1600000000000);
/** @param {string} file */
function setStableMtime(file) {fs.utimesSync(file, stableFileTime, stableFileTime);}
/** @param {string} file @returns {{mtimeNs: bigint, ctimeNs: bigint}} */
function transientRewriteRestore(file) {
  const original = fs.readFileSync(file), before = fs.statSync(file, {bigint: true}), changed = Buffer.from(original);
  changed[0] = (changed[0] || 0) ^ 0xff;
  fs.writeFileSync(file, changed);
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 25);
  fs.writeFileSync(file, original);
  fs.utimesSync(file, stableFileTime, stableFileTime);
  const after = fs.statSync(file, {bigint: true});
  assert.equal(after.mtimeNs, before.mtimeNs, "restore the original mtime");
  assert.notEqual(after.ctimeNs, before.ctimeNs, "the write must still advance ctime");
  return {mtimeNs: after.mtimeNs, ctimeNs: after.ctimeNs};
}

/** @param {string[]} args */
function runReleaseCli(args) {
  return spawnSync(process.execPath, [path.resolve(__dirname, "../scripts/check-native-release.js"), ...args], {
    cwd: path.resolve(__dirname, ".."), env: {...process.env, NODE_OPTIONS: ""}, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20,
  });
}

function nativeTraceRoot() {return fs.realpathSync(process.env["RUNNER_TEMP"] || os.tmpdir());}

/** A strace fixture that writes only synthetic records; the structural candidate is never run.
 * @param {ReturnType<typeof fixture>} f @param {(isCli: boolean) => string} [extraTrace] @param {boolean} [oversized] */
function nativeTraceRunner(f, extraTrace = () => "", oversized = false) {
  return /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
    /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
    f.calls.push({command, args, options});
    assert.equal(command, "/usr/bin/strace");
    assert.deepEqual(args.slice(0, 6), ["-f", "-qq", "-s", "4096", "-e", "trace=file,process"]);
    const traceOption = args.indexOf("-o"), boundary = args.indexOf("--"), tracePath = args[traceOption + 1];
    assert.ok(traceOption > 0 && boundary === traceOption + 2);
    if (typeof tracePath !== "string") {throw new Error("Trace fixture omitted its output path");}
    assert.equal(fs.existsSync(tracePath), false, "each invocation starts with a fresh trace file");
    assert.equal(fs.statSync(path.dirname(tracePath)).mode & 0o077, 0, "trace is in a private directory");
    assert.equal(args[boundary + 1], f.executable, "the tracee is the candidate itself, not a wrapper");
    const invocation = args.slice(boundary + 2), isCli = invocation.length === 1 && invocation[0] === "invalid-directive";
    const execCount = isCli ? 1 : 2, launches = Array.from({length: execCount}, (_, index) =>
      `${100 + index} execve(${JSON.stringify(f.executable)}, [${JSON.stringify(f.executable)}], 0x1) = 0`);
    if (oversized && isCli) {
      const fd = fs.openSync(tracePath, "wx", 0o600);
      try {fs.ftruncateSync(fd, (32 << 20) + 1);} finally {fs.closeSync(fd);}
    } else {fs.writeFileSync(tracePath, launches.join("\n") + "\n" + extraTrace(isCli));}
    assert.equal(options.env?.["PATH"], "/unavailable-native-runtime-tools");
    assert.equal(options.env?.["MOM_TEST_EXECUTABLE"], f.executable);
    assert.equal(options.env?.["MOM_TEST_NO_POOL_NETWORK"], "1");
    assert.equal(options.env?.["LD_PRELOAD"], undefined);
    assert.equal(options.env?.["NODE_OPTIONS"], undefined);
    return isCli ? {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
      stderr: "ERROR: Unknown directive invalid-directive\n"} : {status: 0, signal: null, error: undefined, stdout: "PASSED\n", stderr: ""};
  }));
}

test.describe("Native release provenance admission", () => {
  test("runtime qualification imports do not load JavaScript build tooling", () => {
    const script = `const assert = require('node:assert/strict'), Module = require('node:module');
const load = Module._load;
Module._load = function(name, ...args) {
  if (name === 'esbuild') throw new Error('Runtime checks must not load the JavaScript bundler');
  return load.call(this, name, ...args);
};
assert.equal(typeof require(${JSON.stringify(path.resolve(__dirname, "../scripts/check-native-release.js"))}).checkNativeExecutable, 'function');
assert.equal(typeof require(${JSON.stringify(path.resolve(__dirname, "../scripts/package-native-release.js"))}).nativeNoticeFile, 'function');`;
    const child = spawnSync(process.execPath, ["--max-old-space-size=128", "-e", script], {encoding: "utf8", timeout: 30000, maxBuffer: 64 << 10});
    assert.equal(child.status, 0, child.stderr); assert.equal(child.signal, null);
  });
  test("descriptor and producer changes fail before any candidate launch", t => {
    for (const kind of ["config", "producer"]) {
      const f = fixture(t);
      fs.appendFileSync(kind === "config" ? f.record.configFile : f.producerFile, " ");
      assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn}), /provenance|receipt|descriptor/i);
      assert.equal(f.calls.length, 0);
    }
  });

  test("a rehashed producer must still bind the raw image and selected descriptor", t => {
    const f = fixture(t); f.producer.configHash = "0".repeat(64);
    fs.writeFileSync(f.producerFile, JSON.stringify(f.producer));
    f.record.producerReceipt.sha256 = fileHash(f.producerFile);
    f.record.componentHashes[f.producerFile] = f.record.producerReceipt.sha256; f.write();
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn}), /producer receipt/i);
    assert.equal(f.calls.length, 0);
  });

  test("every explicit metadata component must be hash-bound and the producer included", t => {
    const f = fixture(t), extra = path.join(f.root, "gpu-descriptor.json");
    fs.writeFileSync(extra, "{}"); f.record.componentHashes[extra] = "0".repeat(64); f.write();
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn}), /provenance|component/i);
    delete f.record.componentHashes[extra]; delete f.record.componentHashes[f.producerFile]; f.write();
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn}), /producer|component/i);
    assert.equal(f.calls.length, 0);
  });

  test("packed candidates require a separate packing receipt", t => {
    const f = fixture(t, hostTarget, true); f.record.packingReceipt = undefined; f.write();
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn}), /packing receipt/i);
    assert.equal(f.calls.length, 0);
  });

  test("rehashed packing evidence cannot name different source or artifact bytes", t => {
    const f = fixture(t, hostTarget, true); f.packing.source.sha256 = "0".repeat(64);
    fs.writeFileSync(f.packingFile, JSON.stringify(f.packing));
    assert.ok(f.record.packingReceipt); f.record.packingReceipt.sha256 = fileHash(f.packingFile); f.write();
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn}), /packing receipt/i);
    assert.equal(f.calls.length, 0);
  });

  test("packing stage logs must exist and retain their bound bytes", t => {
    const f = fixture(t, hostTarget, true); assert.ok(f.packing.logs["restore"]);
    fs.appendFileSync(f.packing.logs["restore"].path, "changed");
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn}), /provenance|packing/i);
    assert.equal(f.calls.length, 0);
  });

  test("unpacked images cannot claim unrelated packing evidence", t => {
    const f = fixture(t); f.record.packingReceipt = {path: f.producerFile, sha256: fileHash(f.producerFile)}; f.write();
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn}), /cannot claim a packing receipt/);
    assert.equal(f.calls.length, 0);
  });

  test("producer receipt paths cannot be redirected to an unrelated matching file", t => {
    const f = fixture(t), extra = path.join(f.root, "unrelated.json");
    fs.copyFileSync(f.producerFile, extra); f.record.producerReceipt.path = extra; f.write();
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn}), /unexpected producer receipt/);
    assert.equal(f.calls.length, 0);
  });

  test("Windows generated descriptors are checked without opening their SDK libraries", t => {
    const f = fixture(t, "win32"), generated = path.join(f.root, "generated-controller.json");
    fs.writeFileSync(generated, JSON.stringify({sdk: {libraries: ["/not-mounted/build-only.lib"]}}));
    f.record.generatedDescriptor = {path: generated, sha256: fileHash(generated)};
    f.producer.configHash = f.record.generatedDescriptor.sha256; fs.writeFileSync(f.producerFile, JSON.stringify(f.producer));
    f.record.producerReceipt.sha256 = fileHash(f.producerFile); f.record.componentHashes[f.producerFile] = fileHash(f.producerFile); f.write();
    const result = checkNativeRelease(f.entrypoint, "cpu", {platform: "win32", arch: "x64", spawn: f.spawn});
    assert.equal(result.suitePassed, true); assert.equal(result.coldStartPreflight?.phase, "windows-cold-start-preflight");
    fs.appendFileSync(generated, " ");
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: "win32", arch: "x64", spawn: f.spawn}), /provenance|descriptor/i);
    assert.equal(f.calls.length, 2);
  });

  test("metadata changed by the runner invalidates an otherwise passing vector suite", t => {
    const f = fixture(t), mutate = /** @type {typeof f.spawn} */ ((command, args, options) => {
      fs.appendFileSync(f.producerFile, " "); return f.spawn(command, args, options);
    });
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: mutate}), /provenance.*changed|producer.*changed/i);
    assert.equal(f.calls.length, 1);
  });

  test("selected notice data is checked without becoming a runtime dependency", t => {
    const f = fixture(t), file = path.join(f.root, "THIRD-PARTY-NOTICES.txt"); fs.writeFileSync(file, "fixture notices\n");
    f.record.notices = {path: file, sha256: fileHash(file), bytes: fs.statSync(file).size}; f.write();
    assert.equal(checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn}).suitePassed, true);
    fs.appendFileSync(file, "changed");
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn}), /provenance|notice/i);
    assert.equal(f.calls.length, hostTarget === "win32" ? 2 : 1);
  });

  test("clean-host candidate admission checks notice proof metadata without opening original sources", t => {
    for (const target of syntheticTargets) {
      const f = fixture(t, target), file = path.join(f.root, "THIRD-PARTY-NOTICES.txt"); fs.writeFileSync(file, "fixture notices\n");
      const hash = fileHash(file), bytes = fs.statSync(file).size;
      f.record.notices = {path: file, sha256: hash, bytes};
      f.record.noticeAssembly = {receiptPath: path.join(f.root, "unavailable", "native-notices.json"), receiptSha256: "a".repeat(64),
        fingerprint: "b".repeat(64), notice: {path: path.join(f.root, "unavailable", "THIRD-PARTY-NOTICES.txt"), sha256: hash, bytes}};
      f.write();
      const check = () => checkNativeRelease(f.entrypoint, "cpu", {platform: target, arch: "x64",
        audit: raw => assert.ok(path.isAbsolute(raw)), spawn: f.spawn});
      assert.equal(check().suitePassed, true);
      const expectedCalls = target === "win32" ? 2 : 1;
      assert.equal(f.calls.length, expectedCalls);
      f.record.noticeAssembly.notice.path = path.join(f.root, "other-source", "THIRD-PARTY-NOTICES.txt"); f.write();
      assert.throws(check, /sibling/);
      f.record.noticeAssembly.notice.path = path.join(f.root, "unavailable", "THIRD-PARTY-NOTICES.txt");
      f.record.noticeAssembly.notice.sha256 = "0".repeat(64); f.write();
      assert.throws(check, /does not bind the candidate notice bytes/);
      assert.equal(f.calls.length, expectedCalls, "malformed receipt metadata must fail before runtime execution");
    }
  });
});

test.describe("Native CPU release qualification entrypoint", () => {
  test("Windows cold-start ctime change is isolated before the strict CPU baseline", t => {
    const f = fixture(t, "win32"); setStableMtime(f.executable);
    const beforeCtime = fs.statSync(f.executable, {bigint: true}).ctimeNs.toString();
    const calls = /** @type {{command: string, args: string[], options: {cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}}[]} */ ([]);
    const spawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      calls.push({command, args, options});
      if (command === f.executable && args.length === 1 && args[0] === "invalid-directive") {
        transientRewriteRestore(f.executable);
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
          stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      return f.spawn(command, args, options);
    }));
    const result = checkNativeRelease(f.entrypoint, "cpu", {platform: "win32", arch: "x64", spawn});
    const beforeFormal = calls[0]; assert.ok(beforeFormal, "cold-start preflight must precede the formal vector process");
    assert.equal(beforeFormal.command, f.executable); assert.deepEqual(beforeFormal.args, ["invalid-directive"]);
    assert.equal(result.suitePassed, true);
    assert.equal(result.stdout, successfulCpuSummary, "the preflight is not folded into the vector summary");
    assert.deepEqual(result.coldStartPreflight, {phase: "windows-cold-start-preflight", executable: f.executable, args: ["invalid-directive"],
      status: 1, expectedStatus: 1, outputMarkers: ["Unknown directive invalid-directive", "--save_config:"],
      sha256: fileHash(f.executable), bytes: fs.statSync(f.executable).size,
      identity: {mode: fs.statSync(f.executable).mode, dev: fs.statSync(f.executable, {bigint: true}).dev.toString(),
        ino: fs.statSync(f.executable, {bigint: true}).ino.toString(), nlink: 1, mtimeNs: fs.statSync(f.executable, {bigint: true}).mtimeNs.toString()},
      ctimeNs: {before: beforeCtime, after: fs.statSync(f.executable, {bigint: true}).ctimeNs.toString()}});
    assert.notEqual(result.coldStartPreflight?.ctimeNs.after, beforeCtime);
    assert.equal(calls.length, 2); assert.equal(calls[1]?.command, process.execPath);
    const preflightEnv = beforeFormal.options.env || {};
    assert.equal(preflightEnv["MOM_COMMAND"], "./mom"); assert.equal(preflightEnv["MOM_TEST_EXECUTABLE"], f.executable);
    assert.equal(preflightEnv["MOM_TEST_NO_POOL_NETWORK"], "1"); assert.equal(preflightEnv["MOM_SKIP_MSR"], "1");
    for (const key of ["NODE_OPTIONS", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "MOM_NATIVE_RUNTIME"]) {
      assert.equal(Object.hasOwn(preflightEnv, key), false, key + " must not leak into cold-start preflight");
    }
  });

  test("Windows packed candidate keeps the separate raw source baseline strict", t => {
    const f = fixture(t, "win32", true); setStableMtime(f.executable);
    const sourceHash = fileHash(f.source), sourceBefore = fs.statSync(f.source, {bigint: true}), artifactCtime = fs.statSync(f.executable, {bigint: true}).ctimeNs.toString();
    const spawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      if (command === f.executable && args.length === 1 && args[0] === "invalid-directive") {
        transientRewriteRestore(f.executable);
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
          stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      return f.spawn(command, args, options);
    }));
    const result = checkNativeRelease(f.entrypoint, "cpu", {platform: "win32", arch: "x64", spawn});
    assert.equal(result.suitePassed, true); assert.equal(result.coldStartPreflight?.executable, f.executable);
    assert.notEqual(result.coldStartPreflight?.ctimeNs.after, artifactCtime);
    const sourceAfter = fs.statSync(f.source, {bigint: true});
    assert.equal(fileHash(f.source), sourceHash); assert.equal(sourceAfter.ctimeNs, sourceBefore.ctimeNs);
    assert.equal(sourceAfter.ino, sourceBefore.ino); assert.equal(sourceAfter.mtimeNs, sourceBefore.mtimeNs);
  });

  test("Windows cold-start preflight requires the bounded invalid-directive response", t => {
    const valid = {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
      stderr: "ERROR: Unknown directive invalid-directive\n"};
    const invalid = /** @type {[string, unknown][]} */ ([
      ["zero status", {...valid, status: 0}], ["signal", {...valid, signal: "SIGTERM"}],
      ["spawn error", {...valid, error: new Error("fixture spawn failure")}],
      ["missing marker", {...valid, stderr: "ERROR: startup failed\n"}],
      ["missing footer", {...valid, stdout: "Options:\n"}],
      ["non-string output", {...valid, stdout: undefined}],
    ]);
    for (const [name, response] of invalid) {
      const f = fixture(t, "win32"); let formalCalls = 0, preflightCalls = 0;
      const spawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
        /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
        if (command === f.executable && args[0] === "invalid-directive") {preflightCalls++; return response;}
        formalCalls++; return f.spawn(command, args, options);
      }));
      assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: "win32", arch: "x64", spawn}), error => {
        const failure = /** @type {Error & {stdout?: string, stderr?: string}} */ (error);
        const result = /** @type {{status: number, signal?: string, error?: Error, stdout?: string, stderr?: string}} */ (response);
        assert.match(failure.message, /cold-start preflight/i);
        assert.match(failure.message, result.error ? /fixture spawn failure/ : new RegExp("exit " + result.status));
        assert.equal(failure.stdout, result.stdout || ""); assert.equal(failure.stderr, result.stderr || "");
        return true;
      }, name);
      assert.equal(preflightCalls, 1, name); assert.equal(formalCalls, 0, name); assert.equal(f.calls.length, 0, name);
    }
  });

  test("Windows preflight cannot rebase any executable change besides ctime", t => {
    const cases = /** @type {[string, (file: string) => void][]} */ ([
      ["bytes", file => {const data = fs.readFileSync(file); data[0] = (data[0] || 0) ^ 0xff; fs.writeFileSync(file, data); setStableMtime(file);}],
      ["mtime", file => fs.utimesSync(file, new Date(1700000000000), new Date(1700000000000))],
      ["mode", file => fs.chmodSync(file, fs.statSync(file).mode ^ 0o100)],
      ["link count", file => fs.linkSync(file, file + ".extra-link")],
      ["file identity", file => {
        const moved = file + ".old", before = fs.statSync(file); fs.renameSync(file, moved); fs.copyFileSync(moved, file);
        fs.chmodSync(file, before.mode & 0o777); fs.utimesSync(file, before.atime, before.mtime);
      }],
    ]).filter(([name]) => process.platform !== "win32" || name !== "mode");
    for (const [name, mutate] of cases) {
      const f = fixture(t, "win32"); setStableMtime(f.executable); let formalCalls = 0;
      const spawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
        /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
        if (command === f.executable && args[0] === "invalid-directive") {
          mutate(f.executable);
          return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
            stderr: "ERROR: Unknown directive invalid-directive\n"};
        }
        formalCalls++; return f.spawn(command, args, options);
      }));
      assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: "win32", arch: "x64", spawn}),
        /changed executable|regular file|symlink|link/i, name);
      assert.equal(formalCalls, 0, name); assert.equal(f.calls.length, 0, name);
    }
  });

  test("Windows receipt mutation during cold-start remains outside the ctime allowance", t => {
    const f = fixture(t, "win32"); let formalCalls = 0;
    const spawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      if (command === f.executable && args[0] === "invalid-directive") {
        fs.appendFileSync(f.producerFile, " ");
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
          stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      formalCalls++; return f.spawn(command, args, options);
    }));
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: "win32", arch: "x64", spawn}), /provenance changed/i);
    assert.equal(formalCalls, 0); assert.equal(f.calls.length, 0);
  });

  test("Windows formal suite ctime changes remain strict after the preflight", t => {
    const f = fixture(t, "win32"); setStableMtime(f.executable); let formalCalls = 0;
    const spawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      if (command === f.executable && args[0] === "invalid-directive") {
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
          stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      formalCalls++; const result = f.spawn(command, args, options); transientRewriteRestore(f.executable); return result;
    }));
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: "win32", arch: "x64", spawn}), /changed during qualification/i);
    assert.equal(formalCalls, 1); assert.equal(f.calls.length, 1);
  });

  test("uses maintained CPU vectors, explicit candidate hook and guarded environment", {skip: process.platform === "win32" && "This Linux packed-candidate check requires POSIX executable metadata"}, t => {
    const f = fixture(t, "linux", true), auditPaths = /** @type {string[]} */ ([]);
    const spawn = /** @type {typeof f.spawn} */ ((command, args, options) => {
      auditPaths.push("launch"); return f.spawn(command, args, options);
    });
    const result = checkNativeRelease(f.entrypoint, "cpu", {
      platform: "linux", arch: "x64", audit: raw => auditPaths.push(raw), spawn,
      env: {MOM_TEST_EXECUTABLE: "/old/mom", MOM_GPU_BACKEND: "nvidia", MOM_GPU_TEST_VENDORS: "nvidia",
        MOM_SKIP_MSR: "0", MOM_TEST_NO_POOL_NETWORK: "0", NODE_OPTIONS: "--require=untrusted.js",
        CUDA_VISIBLE_DEVICES: "0", LD_LIBRARY_PATH: "/sdk/lib", FORCE_COLOR: "1", NO_COLOR: "0", PATH: "/sdk/compiler/bin:/usr/bin"},
    });
    assert.deepEqual(auditPaths, [f.source, "launch"]);
    assert.equal(result.suitePassed, true); assert.equal(result.executable, f.executable);
    const call = f.calls[0]; assert.ok(call); assert.equal(call.command, process.execPath);
    assert.deepEqual(call.args, [path.resolve(__dirname, "..", "tests", "run-hash.js"), "cpu"]);
    assert.equal(call.options.cwd, path.resolve(__dirname, "..")); assert.equal(call.options.timeout, suiteTimeoutMs);
    assert.ok(suiteTimeoutMs < 30 * 60 * 1000);
    const env = call.options.env || {};
    assert.equal(env["MOM_TEST_EXECUTABLE"], f.executable); assert.equal(env["MOM_SKIP_MSR"], "1");
    assert.equal(env["MOM_TEST_NO_POOL_NETWORK"], "1"); assert.equal(env["MOM_GPU_BACKEND"], "");
    assert.equal(env["MOM_SKIP_SYCL_ALGO_PARAMS"], "1"); assert.equal(env["NO_COLOR"], "1");
    assert.equal(env["PATH"], "/unavailable-native-runtime-tools");
    for (const key of ["MOM_GPU_TEST_VENDORS", "NODE_OPTIONS", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "FORCE_COLOR"]) {
      assert.equal(Object.hasOwn(env, key), false, key + " must not leak into the candidate test process");
    }
  });

  test("audits the raw linked executable while launching the packed artifact", {skip: process.platform === "win32" && "This Linux packed-candidate check requires POSIX executable metadata"}, t => {
    const f = fixture(t, "linux", true), audited = /** @type {string[]} */ ([]);
    const result = checkNativeRelease(f.entrypoint, "cpu", {platform: "linux", arch: "x64", audit: raw => audited.push(raw), spawn: f.spawn});
    assert.deepEqual(audited, [f.source]); assert.equal(result.executable, f.executable);
    assert.equal(Object.hasOwn(result, "coldStartPreflight"), false);
    assert.notEqual(f.source, f.executable);
  });

  test("Windows PE fixtures pass the sealed-map audit without real execution", t => {
    const f = fixture(t, "win32");
    const result = checkNativeRelease(f.entrypoint, "cpu", {platform: "win32", arch: "x64", spawn: f.spawn});
    assert.equal(result.suitePassed, true); assert.equal(result.gpu, undefined); assert.equal(f.calls.length, 2);
    assert.equal(result.coldStartPreflight?.phase, "windows-cold-start-preflight");
  });

  test("Windows archive qualification warms only its extracted executable", t => {
    const f = fixture(t, "win32"), noticeName = "THIRD-PARTY-NOTICES.txt", noticePath = path.join(f.root, noticeName),
      archivePath = path.join(f.root, "mom-v0.9.4-win.zip"), toolPath = path.join(f.root, "cmake.exe"),
      packageReceiptPath = path.join(f.root, "native-package.json");
    fs.writeFileSync(noticePath, "Synthetic archive notice.\n");
    const notices = {path: noticePath, bytes: fs.statSync(noticePath).size, sha256: fileHash(noticePath)};
    f.record.notices = notices; f.write();
    fs.writeFileSync(archivePath, "synthetic zip fixture"); fs.writeFileSync(toolPath, "synthetic cmake fixture");
    fs.writeFileSync(packageReceiptPath, JSON.stringify({version: 1, entrypoint: f.entrypoint, entryHash: fileHash(f.entrypoint),
      archive: {path: archivePath, bytes: fs.statSync(archivePath).size, sha256: fileHash(archivePath)},
      executable: {path: f.executable, bytes: fs.statSync(f.executable).size, sha256: fileHash(f.executable)},
      tool: {path: "/historical/cmake.exe", sha256: "a".repeat(64)}, notices, members: ["mom.exe", noticeName],
      exactOriginalBytes: true, buildOnly: true, publicationQualified: false}));
    const protectedFiles = [f.entrypoint, f.producerFile, f.executable, archivePath, packageReceiptPath, toolPath, noticePath],
      before = new Map(protectedFiles.map(file => [file, fs.statSync(file, {bigint: true}).ctimeNs.toString()]));
    let extractedExecutable = "", preflightPath = "";
    const spawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      if (command === toolPath) {
        if (args[2] === "tf") {return {status: 0, signal: null, error: undefined, stdout: "mom.exe\n" + noticeName + "\n", stderr: ""};}
        if (args[2] === "tvf") {
          return {status: 0, signal: null, error: undefined,
            stdout: "-rwxr-xr-x 1 user group " + fs.statSync(f.executable).size + " date mom.exe\n" +
              "-rw-r--r-- 1 user group " + notices.bytes + " date " + noticeName + "\n", stderr: ""};
        }
        if (args[2] === "xf") {
          extractedExecutable = path.join(options.cwd || "", "mom.exe");
          fs.copyFileSync(f.executable, extractedExecutable); setStableMtime(extractedExecutable);
          fs.copyFileSync(noticePath, path.join(options.cwd || "", noticeName));
          return {status: 0, signal: null, error: undefined, stdout: "", stderr: ""};
        }
      }
      if (command === extractedExecutable && args.length === 1 && args[0] === "invalid-directive") {
        preflightPath = command; transientRewriteRestore(command);
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
          stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      return f.spawn(command, args, options);
    }));
    const result = checkNativeRelease(f.entrypoint, "cpu", {platform: "win32", arch: "x64",
      archive: {path: archivePath, tool: toolPath}, spawn});
    assert.equal(result.suitePassed, true); assert.equal(result.archive, archivePath);
    assert.equal(result.coldStartPreflight?.executable, extractedExecutable);
    assert.equal(preflightPath, extractedExecutable); assert.notEqual(preflightPath, f.executable);
    for (const file of protectedFiles) {
      assert.equal(fs.statSync(file, {bigint: true}).ctimeNs.toString(), before.get(file), path.basename(file) + " retains original ctime");
    }
  });

  test("malformed receipts and changed artifacts fail before the runner", t => {
    const malformed = fixture(t); fs.writeFileSync(malformed.entrypoint, "{");
    assert.throws(() => checkNativeRelease(malformed.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: malformed.spawn}), /JSON|receipt/i);
    assert.equal(malformed.calls.length, 0);

    const changed = fixture(t); fs.appendFileSync(changed.executable, "changed");
    assert.throws(() => checkNativeRelease(changed.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: changed.spawn}), /hash\/size/);
    assert.equal(changed.calls.length, 0);
  });

  test("format and target mismatches fail before audit or launch", t => {
    const wrongFormat = fixture(t), windows = wrongFormat.record.target === "win32", invalidImage = Buffer.alloc(windows ? 4096 : 128);
    invalidImage.write(windows ? "not a PE image" : "not an ELF");
    fs.writeFileSync(wrongFormat.executable, invalidImage, {mode: 0o755});
    wrongFormat.record.artifact.bytes = fs.statSync(wrongFormat.executable).size;
    wrongFormat.record.artifact.sha256 = fileHash(wrongFormat.executable); wrongFormat.record.source.sha256 = wrongFormat.record.artifact.sha256;
    wrongFormat.write();
    assert.throws(() => checkNativeRelease(wrongFormat.entrypoint, "cpu", {audit: raw => assert.ok(path.isAbsolute(raw)), spawn: wrongFormat.spawn}),
      windows ? /Windows x86-64 PE executable/ : /Linux x86-64 ELF executable/);
    assert.equal(wrongFormat.calls.length, 0);

    const wrongTarget = fixture(t, "win32");
    assert.throws(() => checkNativeRelease(wrongTarget.entrypoint, "cpu", {platform: "linux", arch: "x64", audit: raw => assert.ok(path.isAbsolute(raw)), spawn: wrongTarget.spawn}), /wrong-target/);
    assert.equal(wrongTarget.calls.length, 0);
  });

  test("matching provenance cannot qualify a fixed-address Linux executable", {skip: process.platform === "win32" && "Linux ELF type is not representable by the Windows fixture host"}, t => {
    const f = fixture(t, "linux"), data = fs.readFileSync(f.executable); data.writeUInt16LE(2, 16); fs.writeFileSync(f.executable, data);
    const imageHash = fileHash(f.executable);
    assert.ok("sha256" in f.producer);
    f.record.artifact.sha256 = imageHash; f.record.source.sha256 = imageHash; f.producer.sha256 = imageHash;
    fs.writeFileSync(f.producerFile, JSON.stringify(f.producer));
    const receiptHash = fileHash(f.producerFile);
    f.record.componentHashes[f.producerFile] = receiptHash; f.record.producerReceipt.sha256 = receiptHash; f.write();
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {audit: () => undefined, spawn: f.spawn}), /position-independent/);
    assert.equal(f.calls.length, 0);
  });

  test("unsupported targets and unconfigured GPU suites are rejected", t => {
    const f = fixture(t);
    assert.throws(() => checkNativeRelease(f.entrypoint, "gpu", {spawn: f.spawn}), /explicit supported vendor and algorithm/);
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: "darwin", arch: "x64", spawn: f.spawn}), /supports Linux\/Windows/);
    assert.equal(f.calls.length, 0);
  });

  test("nonzero or indeterminate test-runner status never reports a pass", t => {
    for (const target of syntheticTargets) {
      const failed = fixture(t, target), failSpawn = /** @type {typeof failed.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command,
        /** @type {string[]} */ args, /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
        failed.calls.push({command, args, options});
        if (target === "win32" && command === failed.executable && args.length === 1 && args[0] === "invalid-directive") {
          return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n", stderr: "ERROR: Unknown directive invalid-directive\n"};
        }
        return {status: 7, signal: null, error: undefined, stdout: "FAIL algo-a\nFAIL algo-b\n", stderr: ""};
      }));
      assert.throws(() => checkNativeRelease(failed.entrypoint, "cpu", {platform: target, audit: raw => assert.ok(path.isAbsolute(raw)), spawn: failSpawn}), error => {
        assert.ok(error instanceof Error);
        const details = /** @type {Error & {stdout: string}} */ (error);
        assert.match(details.message, /failed \(exit 7\)/); assert.match(details.stdout, /algo-a/); assert.match(details.stdout, /algo-b/); return true;
      });

      const unknown = fixture(t, target), unknownSpawn = /** @type {typeof unknown.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command,
        /** @type {string[]} */ args, /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
        unknown.calls.push({command, args, options});
        if (target === "win32" && command === unknown.executable && args.length === 1 && args[0] === "invalid-directive") {
          return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n", stderr: "ERROR: Unknown directive invalid-directive\n"};
        }
        return {status: null, signal: null, error: undefined, stdout: "PASSED\n", stderr: ""};
      }));
      assert.throws(() => checkNativeRelease(unknown.entrypoint, "cpu", {platform: target, audit: raw => assert.ok(path.isAbsolute(raw)), spawn: unknownSpawn}), /exit null/);
    }
  });

  test("a zero exit or PASSED token without the vector reporter summary is not a pass", t => {
    for (const target of syntheticTargets) {
      const f = fixture(t, target), incomplete = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command,
        /** @type {string[]} */ args, /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
        f.calls.push({command, args, options});
        if (target === "win32" && command === f.executable && args.length === 1 && args[0] === "invalid-directive") {
          return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n", stderr: "ERROR: Unknown directive invalid-directive\n"};
        }
        return {status: 0, signal: null, error: undefined, stdout: "PASSED\n", stderr: ""};
      }));
      assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: target, audit: raw => assert.ok(path.isAbsolute(raw)), spawn: incomplete}), error => {
        assert.ok(error instanceof Error); assert.match(error.message, /every maintained vector passing without skips/); return true;
      });
      assert.equal(f.record.publicationQualified, false);
    }
  });

  test("zero exit summaries must count every CPU vector and contain no skips or failures", t => {
    for (const target of syntheticTargets) {
      const f = fixture(t, target), summary = (/** @type {number} */ tests, /** @type {number} */ passed,
      /** @type {number} */ failed, /** @type {number} */ skipped) =>
        `ℹ tests ${tests}\nℹ suites 1\nℹ pass ${passed}\nℹ fail ${failed}\nℹ cancelled 0\nℹ skipped ${skipped}\n`;
      const invalid = [summary(0, 0, 0, 0), summary(cpuVectorCount - 1, cpuVectorCount - 1, 0, 0),
        summary(cpuVectorCount, cpuVectorCount - 1, 1, 0), summary(cpuVectorCount, cpuVectorCount - 1, 0, 1)];
      for (const stdout of invalid) {
        const spawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command,
        /** @type {string[]} */ args, /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
          f.calls.push({command, args, options});
          if (target === "win32" && command === f.executable && args.length === 1 && args[0] === "invalid-directive") {
            return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n", stderr: "ERROR: Unknown directive invalid-directive\n"};
          }
          return {status: 0, signal: null, error: undefined, stdout, stderr: ""};
        }));
        assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {platform: target, audit: raw => assert.ok(path.isAbsolute(raw)), spawn}), /every maintained vector passing/);
      }
      assert.equal(f.calls.length, invalid.length * (target === "win32" ? 2 : 1));
    }
  });

  test("artifact mutation during audit or the vector run invalidates the result", t => {
    const beforeLaunch = fixture(t);
    assert.throws(() => checkNativeRelease(beforeLaunch.entrypoint, "cpu", {
      audit: () => fs.appendFileSync(beforeLaunch.executable, "changed during audit"), spawn: beforeLaunch.spawn,
    }), /artifact changed during qualification/);
    assert.equal(beforeLaunch.calls.length, 0);

    const duringRun = fixture(t), mutateRunner = /** @type {typeof duringRun.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command,
      /** @type {string[]} */ args, /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      duringRun.calls.push({command, args, options});
      if (duringRun.record.target === "win32" && command === duringRun.executable && args.length === 1 && args[0] === "invalid-directive") {
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n", stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      fs.appendFileSync(duringRun.executable, "changed during run");
      return {status: 0, signal: null, error: undefined, stdout: successfulCpuSummary, stderr: ""};
    }));
    assert.throws(() => checkNativeRelease(duringRun.entrypoint, "cpu", {
      audit: raw => assert.ok(path.isAbsolute(raw)), spawn: mutateRunner,
    }), /artifact changed during qualification/);
    assert.equal(duringRun.calls.length, duringRun.record.target === "win32" ? 2 : 1);
  });
});

test.describe("Native release single-algorithm GPU qualification", () => {
  const gpu = {vendor: "intel", algo: "pearlhash"};
  const vectors = hashTests.filter(test => test.gpu && !test.portableOnly && test.job.algo === gpu.algo).length;
  const summary = (tests = vectors + 2, pass = tests, skipped = 0, failed = 0) =>
    `ℹ tests ${tests}\nℹ suites 1\nℹ pass ${pass}\nℹ fail ${failed}\nℹ cancelled 0\nℹ skipped ${skipped}\n`;
  /** Synthetic runner only; no hardware discovery or hashing.
   * @param {ReturnType<typeof fixture>} f @param {string} output */
  const runner = (f, output) => /** @type {typeof f.spawn} */ (/** @type {unknown} */ (
    (/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      f.calls.push({command, args, options});
      if (f.record.target === "win32" && command === f.executable && args?.length === 1 && args[0] === "invalid-directive") {
        return {status: 1, signal: null, stdout: "Options:\n--save_config: file\n", stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      return {status: 0, signal: null, stdout: output, stderr: ""};
    }));
  test("reuses the discrete vector runner with one explicit vendor and algorithm", t => {
    const f = fixture(t), result = checkNativeRelease(f.entrypoint, "gpu", {
      gpu, audit: () => undefined, spawn: runner(f, summary()),
      env: {MOM_GPU_TEST_VENDORS: "nvidia,amd", MOM_GPU_TEST_ALGO: "c29", MOM_GPU_INDEX: "99",
        MOM_SKIP_SYCL_ALGO_PARAMS: "1", MOM_NATIVE_PATH: "/old/addon", NODE_OPTIONS: "--require=untrusted.js"},
    });
    assert.equal(result.suitePassed, true); assert.deepEqual(result.gpu, gpu); assert.notEqual(result.gpu, gpu);
    assert.equal(f.calls.length, f.record.target === "win32" ? 2 : 1);
    const call = f.calls.at(-1); assert.ok(call);
    assert.deepEqual(call.args, [path.resolve(__dirname, "run-hash.js"), "gpu-discrete"]);
    const env = call.options.env || {};
    assert.equal(env["MOM_TEST_EXECUTABLE"], f.executable);
    assert.equal(env["MOM_GPU_BACKEND"], "intel"); assert.equal(env["MOM_GPU_TEST_VENDORS"], "intel");
    assert.equal(env["MOM_GPU_TEST_ALGO"], "pearlhash"); assert.equal(env["MOM_REQUIRE_GPU_TESTS"], "1");
    assert.equal(env["MOM_TEST_NO_POOL_NETWORK"], "1"); assert.equal(env["MOM_SKIP_MSR"], "1");
    for (const name of ["MOM_SKIP_SYCL_ALGO_PARAMS", "MOM_GPU_INDEX", "MOM_NATIVE_PATH", "NODE_OPTIONS"]) {
      assert.equal(Object.hasOwn(env, name), false, name);
    }
    assert.equal(call.options.timeout, suiteTimeoutMs);
  });
  test("missing, mixed, unknown and CPU-only selections fail before candidate access", t => {
    const f = fixture(t);
    for (const selected of [undefined, {...gpu, vendor: "all"}, {...gpu, vendor: "intel,nvidia"},
      {...gpu, algo: "rx/0"}, {...gpu, algo: "unknown"}]) {
      assert.throws(() => checkNativeRelease(f.entrypoint, "gpu", {...(selected ? {gpu: selected} : {}), spawn: f.spawn}), /explicit supported vendor and algorithm/);
    }
    assert.throws(() => checkNativeRelease(f.entrypoint, "cpu", {gpu, spawn: f.spawn}), /only to the gpu suite/);
    assert.equal(f.calls.length, 0);
  });
  test("empty, truncated, skipped and failed lanes never qualify as a pass", t => {
    const f = fixture(t);
    for (const stdout of ["PASSED\n", summary(0), summary(vectors + 1),
      summary(vectors + 2, vectors + 1, 1), summary(vectors + 2, vectors + 1, 0, 1)]) {
      assert.throws(() => checkNativeRelease(f.entrypoint, "gpu", {gpu, audit: () => undefined, spawn: runner(f, stdout)}),
        /selected algorithm passing without skips/);
    }
  });
  test("a candidate changed during GPU vectors cannot receive a passing result", t => {
    const f = fixture(t), base = runner(f, summary());
    const spawn = /** @type {typeof f.spawn} */ ((command, args, options) => {
      const argv = args || [];
      if (f.record.target === "win32" && command === f.executable && argv.length === 1 && argv[0] === "invalid-directive") {
        return base(command, argv, options);
      }
      fs.appendFileSync(f.executable, "changed"); return base(command, argv, options);
    });
    assert.throws(() => checkNativeRelease(f.entrypoint, "gpu", {gpu, audit: () => undefined, spawn}), /artifact changed/);
  });
});

test.describe("Native GPU private AdaptiveCpp cache lifecycle", () => {
  const gpu = {vendor: "nvidia", algo: "pearlhash"};
  const vectors = hashTests.filter(test => test.gpu && !test.portableOnly && test.job.algo === gpu.algo).length;
  const summary = `ℹ tests ${vectors + 2}\nℹ suites 1\nℹ pass ${vectors + 2}\nℹ fail 0\nℹ cancelled 0\nℹ skipped 0\n`;
  const target = process.platform === "win32" ? "win32" : "linux";

  /** Synthetic runner only; no candidate or hash vector is executed.
   * @param {ReturnType<typeof fixture>} f @param {(env: NodeJS.ProcessEnv) => void} inspect
   * @param {{status?: number | null, signal?: NodeJS.Signals | null, error?: Error, stdout?: string}} [outcome] */
  const runner = (f, inspect, outcome = {}) => /** @type {typeof f.spawn} */ (/** @type {unknown} */ (
    (/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      if (command !== process.execPath) {return f.spawn(command, args, options);}
      f.calls.push({command, args, options}); inspect(options.env || {});
      return {status: outcome.status === undefined ? 0 : outcome.status, signal: outcome.signal || null,
        error: outcome.error, stdout: outcome.stdout === undefined ? summary : outcome.stdout, stderr: "synthetic GPU fixture"};
    }));

  test("uses a unique private writable cache and deletes it only after a complete pass", t => {
    const f = fixture(t, target), inherited = path.join(os.tmpdir(), "caller-selected-acpp-cache");
    /** @type {string[]} */
    const created = [];
    t.after(() => {for (const directory of created) {if (fs.existsSync(directory)) {fs.rmSync(directory, {recursive: true});}}});
    for (let index = 0; index < 2; index++) {
      const env = {ACPP_APPDB_DIR: inherited};
      const spawn = runner(f, childEnv => {
        const directory = childEnv["ACPP_APPDB_DIR"];
        assert.ok(directory); assert.notEqual(directory, inherited);
        assert.equal(path.dirname(directory), fs.realpathSync(os.tmpdir()));
        const info = fs.lstatSync(directory);
        assert.ok(info.isDirectory()); assert.equal(info.isSymbolicLink(), false);
        assert.deepEqual(fs.readdirSync(directory), []);
        if (process.platform !== "win32") {assert.equal(info.mode & 0o077, 0);}
        fs.writeFileSync(path.join(directory, "writable-probe"), "owned cache\n", {flag: "wx"});
        created.push(directory);
      });
      const result = checkNativeRelease(f.entrypoint, "gpu", {gpu, platform: target, audit: () => undefined, spawn, env});
      assert.equal(result.suitePassed, true);
      assert.deepEqual(env, {ACPP_APPDB_DIR: inherited}, "the caller environment remains untouched");
      const completed = created[index]; assert.ok(completed);
      assert.equal(fs.existsSync(completed), false, "only the completed run's cache is removed");
    }
    assert.notEqual(created[0], created[1], "each discrete GPU qualification receives a fresh cache");
  });

  test("retains and reports the exact cache after process, timeout, summary, or unknown failures", t => {
    const f = fixture(t, target);
    /** @type {string[]} */
    const created = [];
    t.after(() => {for (const directory of created) {if (fs.existsSync(directory)) {fs.rmSync(directory, {recursive: true});}}});
    const failures = /** @type {{name: string, outcome?: {status?: number | null, signal?: NodeJS.Signals | null, error?: Error, stdout?: string}, throws?: boolean}[]} */ ([
      {name: "nonzero exit", outcome: {status: 7}},
      {name: "signal", outcome: {status: null, signal: "SIGTERM"}},
      {name: "timeout", outcome: {status: null, signal: "SIGTERM", error: Object.assign(new Error("timed out"), {code: "ETIMEDOUT"})}},
      {name: "invalid summary", outcome: {status: 0, stdout: "not a test summary\n"}},
      {name: "unknown runner exception", throws: true},
    ]);
    for (const failure of failures) {
      const spawn = runner(f, childEnv => {
        const directory = childEnv["ACPP_APPDB_DIR"];
        assert.ok(directory); created.push(directory);
        fs.writeFileSync(path.join(directory, "failure-evidence"), failure.name, {flag: "wx"});
      }, failure.outcome);
      if (failure.throws) {
        const throwing = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command,
          /** @type {string[]} */ args, /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
          if (command !== process.execPath) {return f.spawn(command, args, options);}
          f.calls.push({command, args, options});
          const directory = options.env?.["ACPP_APPDB_DIR"];
          assert.ok(directory); created.push(directory);
          fs.writeFileSync(path.join(directory, "failure-evidence"), failure.name, {flag: "wx"});
          throw new Error("unexpected synthetic runner failure");
        }));
        assert.throws(() => checkNativeRelease(f.entrypoint, "gpu", {gpu, platform: target, audit: () => undefined, spawn: throwing}), error => {
          assert.ok(error instanceof Error);
          assert.match(error.message, /unexpected synthetic runner failure/);
          const retained = created.at(-1); assert.ok(retained); assert.ok(error.message.includes(retained));
          return true;
        }, failure.name);
      } else {
        assert.throws(() => checkNativeRelease(f.entrypoint, "gpu", {gpu, platform: target, audit: () => undefined, spawn}), error => {
          assert.ok(error instanceof Error);
          assert.match(error.message, /ACPP AppDB cache preserved at/);
          const retained = created.at(-1); assert.ok(retained); assert.ok(error.message.includes(retained));
          return true;
        }, failure.name);
      }
      const retained = created.at(-1);
      assert.ok(retained); assert.ok(fs.statSync(retained).isDirectory(), failure.name + " preserves its cache");
      assert.ok(fs.existsSync(path.join(retained, "failure-evidence")), failure.name + " preserves cache contents");
      fs.rmSync(retained, {recursive: true});
    }
  });

  test("CPU and portable-CPU suites scrub current and legacy AdaptiveCpp settings", t => {
    const f = fixture(t, target), inherited = {ACPP_APPDB_DIR: "/caller/cache", ACPP_CACHE_DIR: "/caller/other",
      HIPSYCL_APPDB_DIR: "/caller/legacy-cache", HIPSYCL_DEBUG_LEVEL: "0"};
    /** @param {"cpu" | "gpu-portable-cpu"} suite @param {string} output */
    const check = (suite, output) => {
      const spawn = runner(f, childEnv => {
        assert.deepEqual(Object.keys(childEnv).filter(key => /^(?:ACPP|HIPSYCL)_/i.test(key)), []);
      }, {stdout: output});
      return checkNativeRelease(f.entrypoint, suite, {platform: target, audit: () => undefined, spawn, env: inherited});
    };
    assert.equal(check("cpu", successfulCpuSummary).suitePassed, true);
    assert.equal(check("gpu-portable-cpu", `ℹ tests ${portableCpuTestCount}\nℹ suites 1\nℹ pass ${portableCpuTestCount}\nℹ fail 0\nℹ cancelled 0\nℹ skipped 0\n`).suitePassed, true);
    assert.deepEqual(inherited, {ACPP_APPDB_DIR: "/caller/cache", ACPP_CACHE_DIR: "/caller/other",
      HIPSYCL_APPDB_DIR: "/caller/legacy-cache", HIPSYCL_DEBUG_LEVEL: "0"});
  });
});

test.describe("Native release portable SYCL/OpenCL CPU qualification", () => {
  const summary = (tests = portableCpuTestCount, pass = tests, skipped = 0, failed = 0) =>
    `ℹ tests ${tests}\nℹ suites 1\nℹ pass ${pass}\nℹ fail ${failed}\nℹ cancelled 0\nℹ skipped ${skipped}\n`;

  test("standalone candidate uses its native OpenCL runtime without a development addon", t => {
    const f = fixture(t, hostTarget), keys = ["MOM_TEST_EXECUTABLE", "MOM_NATIVE_DIR", "MOM_NATIVE_PATH",
        "MOM_NATIVE_PATH_LAUNCHER_DEFAULT", "MOM_GPU_BACKEND", "MOM_OPENCL_DEVICE_TYPE", "ONEAPI_DEVICE_SELECTOR"],
      saved = new Map(keys.map(key => [key, process.env[key]])), missingRoot = path.join(f.root, "absent-native-sdk");
    try {
      process.env["MOM_TEST_EXECUTABLE"] = f.executable;
      process.env["MOM_NATIVE_DIR"] = missingRoot;
      process.env["MOM_NATIVE_PATH"] = path.join(missingRoot, "stale", "mom.node");
      process.env["MOM_NATIVE_PATH_LAUNCHER_DEFAULT"] = path.join(missingRoot, "stale", "mom.node");
      process.env["MOM_GPU_BACKEND"] = "nvidia";
      process.env["MOM_OPENCL_DEVICE_TYPE"] = "gpu";
      process.env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu";
      assert.equal(fs.existsSync(missingRoot), false);
      const originalExistsSync = fs.existsSync;
      t.mock.method(fs, "existsSync", /** @param {import("node:fs").PathLike} file */ file =>
        String(file).endsWith(path.join("dpcpp-opencl", "mom.node")) ? false : originalExistsSync(file));
      const env = openclSyclEnv("cpu");
      assert.equal(env["MOM_GPU_BACKEND"], "opencl");
      assert.equal(env["MOM_OPENCL_DEVICE_TYPE"], "cpu");
      assert.equal(env["MOM_SYCL_COMPILER"], "dpcpp-opencl");
      assert.equal(env["ONEAPI_DEVICE_SELECTOR"], "opencl:cpu");
      assert.equal(env["MOM_NATIVE_PATH"], undefined);
      assert.equal(Object.hasOwn(env, "MOM_RUNTIME_DIR"), false);
      assert.equal(Object.hasOwn(env, "LD_LIBRARY_PATH"), false);
    } finally {
      for (const [key, value] of saved) {
        if (value === undefined) {delete process.env[key];} else {process.env[key] = value;}
      }
    }
  });

  /** Synthetic runner only; no OpenCL device discovery or hashing.
   * @param {ReturnType<typeof fixture>} f @param {string} output */
  const runner = (f, output) => /** @type {typeof f.spawn} */ (/** @type {unknown} */ (
    (/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      f.calls.push({command, args, options});
      if (f.record.target === "win32" && command === f.executable && args.length === 1 && args[0] === "invalid-directive") {
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
          stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      return {status: 0, signal: null, error: undefined, stdout: output, stderr: ""};
    }));

  test("uses the same candidate with fixed CPU OpenCL selectors and no inherited narrowing", t => {
    const target = process.platform === "win32" ? "win32" : "linux", f = fixture(t, target), env = {
      MOM_GPU_BACKEND: "nvidia", MOM_OPENCL_DEVICE_TYPE: "gpu", ONEAPI_DEVICE_SELECTOR: "level_zero:gpu",
      MOM_GPU_TEST_VENDORS: "nvidia", MOM_GPU_TEST_ALGO: "c29", MOM_GPU_INDEX: "99",
      MOM_SKIP_SYCL_ALGO_PARAMS: "1", MOM_NATIVE_PATH: "/old/addon", NODE_OPTIONS: "--require=untrusted.js",
      SYCL_DEVICE_FILTER: "cuda", OCL_ICD_VENDORS: "/untrusted/vendor.icd",
    };
    const result = checkNativeRelease(f.entrypoint, "gpu-portable-cpu", {
      platform: target, arch: "x64", audit: () => undefined, spawn: runner(f, summary()), env,
    });
    assert.equal(result.suite, "gpu-portable-cpu"); assert.equal(result.suitePassed, true);
    assert.equal(result.sha256, f.record.artifact.sha256); assert.equal(result.executable, f.executable);
    assert.equal(f.calls.length, target === "win32" ? 2 : 1);
    const call = f.calls.at(-1); assert.ok(call);
    assert.deepEqual(call.args, [path.resolve(__dirname, "run-hash.js"), "gpu-portable-cpu"]);
    assert.equal(call.options.timeout, suiteTimeoutMs);
    const clean = call.options.env || {};
    assert.equal(clean["MOM_TEST_EXECUTABLE"], f.executable);
    assert.equal(clean["MOM_TEST_NO_POOL_NETWORK"], "1"); assert.equal(clean["MOM_SKIP_MSR"], "1");
    assert.equal(clean["MOM_GPU_BACKEND"], "opencl"); assert.equal(clean["MOM_OPENCL_DEVICE_TYPE"], "cpu");
    assert.equal(clean["ONEAPI_DEVICE_SELECTOR"], "opencl:cpu"); assert.equal(clean["MOM_REQUIRE_PORTABLE_CPU_TESTS"], "1");
    for (const name of ["MOM_SKIP_SYCL_ALGO_PARAMS", "MOM_GPU_TEST_VENDORS", "MOM_GPU_TEST_ALGO", "MOM_GPU_INDEX",
      "MOM_NATIVE_PATH", "NODE_OPTIONS", "SYCL_DEVICE_FILTER", "OCL_ICD_VENDORS"]) {
      assert.equal(Object.hasOwn(clean, name), false, name);
    }
  });

  test("requires every maintained portable algorithm/vector test and no skips", t => {
    const target = process.platform === "win32" ? "win32" : "linux", f = fixture(t, target), invalid = [
      "PASSED\n", summary(0), summary(portableCpuTestCount - 1), summary(portableCpuTestCount, portableCpuTestCount - 1, 1),
      summary(portableCpuTestCount, portableCpuTestCount - 1, 0, 1),
    ];
    for (const stdout of invalid) {
      assert.throws(() => checkNativeRelease(f.entrypoint, "gpu-portable-cpu", {
        platform: target, arch: "x64", audit: () => undefined, spawn: runner(f, stdout),
      }), /portable SYCL\/OpenCL CPU.*complete maintained vector coverage passing without skips/);
    }
    const failSpawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command,
      /** @type {string[]} */ args, /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      f.calls.push({command, args, options});
      if (target === "win32" && command === f.executable && args.length === 1 && args[0] === "invalid-directive") {
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n", stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      return {status: 7, signal: null, error: undefined, stdout: summary(), stderr: ""};
    }));
    assert.throws(() => checkNativeRelease(f.entrypoint, "gpu-portable-cpu", {
      platform: target, arch: "x64", audit: () => undefined, spawn: failSpawn,
    }), /failed \(exit 7\)/);
    const signalSpawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command,
      /** @type {string[]} */ args, /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      f.calls.push({command, args, options});
      if (target === "win32" && command === f.executable && args.length === 1 && args[0] === "invalid-directive") {
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n", stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      return {status: 0, signal: "SIGTERM", error: undefined, stdout: summary(), stderr: ""};
    }));
    assert.throws(() => checkNativeRelease(f.entrypoint, "gpu-portable-cpu", {
      platform: target, arch: "x64", audit: () => undefined, spawn: signalSpawn,
    }), /signal SIGTERM/);
  });

  test("runtime-only lane rechecks the exact executable hash and accepts the public CLI name", {skip: !["linux", "win32"].includes(process.platform) || process.arch !== "x64"}, t => {
    const target = process.platform === "win32" ? "win32" : "linux", f = fixture(t, target), expected = fileHash(f.executable),
      spawn = runner(f, summary());
    const result = checkNativeExecutable(f.executable, expected, "gpu-portable-cpu", {spawn});
    assert.equal(result.scope, "runtime-only"); assert.equal(result.suitePassed, true);
    assert.equal(result.sha256, expected); assert.equal(f.calls.length, target === "win32" ? 2 : 1);
    assert.equal(f.calls.at(-1)?.options.env?.["MOM_TEST_EXECUTABLE"], f.executable);
    const before = f.calls.length;
    assert.throws(() => checkNativeExecutable(f.executable, "0".repeat(64), "gpu-portable-cpu", {spawn}), /SHA256/);
    assert.equal(f.calls.length, before, "a mismatched candidate hash must fail before the runner");
    const missing = path.join(f.root, "not-created");
    const parsed = runReleaseCli([missing, "gpu-portable-cpu", "--sha256", "0".repeat(64)]);
    assert.equal(parsed.status, 1); assert.doesNotMatch(parsed.stderr, /Usage:/i);
    assert.match(parsed.stderr, /ENOENT|no such file/i);
    const narrowed = runReleaseCli([missing, "gpu-portable-cpu", "--sha256", "0".repeat(64),
      "--vendor", "intel", "--algo", "pearlhash"]);
    assert.equal(narrowed.status, 1); assert.match(narrowed.stderr, /Usage:/i);
  });

  test("Linux runtime trace checks CLI and one CPU vector with exact same-executable workers", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {
    const f = fixture(t), vector = hashTests.find(definition => !definition.gpu && definition.job.algo === "cn/0");
    assert.ok(vector);
    if (vector.expected === undefined) {throw new Error("Maintained CPU vector has no expected hash");}
    const workers = gpuTuning.parseDeviceList("cpu", vector.job.algo).reduce((total, entry) => total + entry.processes, 0);
    assert.equal(workers, 1, "one low-memory CPU worker exercises the same executable-launch boundary");
    const result = checkNativeExecutable(f.executable, fileHash(f.executable), "trace", {spawn: nativeTraceRunner(f),
      env: {PATH: "/tmp/untrusted", LD_PRELOAD: "/tmp/untrusted.so", NODE_OPTIONS: "--require=untrusted.js",
        MOM_TEST_NO_POOL_NETWORK: "0", MOM_GPU_BACKEND: "nvidia"}});
    const traceMatch = result.stdout.match(/trace logs: (.+)$/m);
    assert.ok(traceMatch, "successful trace log directory is reported for artifact collection");
    const traceDirectory = traceMatch?.[1];
    if (!traceDirectory) {throw new Error("Successful trace log directory was empty");}
    assert.equal(path.dirname(traceDirectory), nativeTraceRoot());
    assert.match(path.basename(traceDirectory), /^mom-native-trace-(?:\d{1,20}-\d{1,4}|local)-/);
    t.after(() => {fs.rmSync(traceDirectory, {recursive: true});});
    assert.equal(result.scope, "runtime-only"); assert.equal(result.suite, "trace"); assert.equal(result.suitePassed, true);
    assert.equal(result.sha256, fileHash(f.executable)); assert.equal(f.calls.length, 2);
    assert.ok(f.calls.every(call => call.command === "/usr/bin/strace"));
    assert.equal(f.calls[0]?.args.slice(f.calls[0].args.indexOf("--") + 1).join(" "), `${f.executable} invalid-directive`);
    const cpuInvocation = f.calls[1]?.args.slice(f.calls[1].args.indexOf("--") + 1);
    assert.equal(cpuInvocation?.[0], f.executable); assert.equal(cpuInvocation?.[1], "test");
    assert.equal(cpuInvocation?.[2], vector.job.algo);
    assert.equal(cpuInvocation?.[3], Array.isArray(vector.expected) ? vector.expected.join("|") : vector.expected);
    assert.equal(cpuInvocation?.[4], "--job");
    assert.deepEqual(JSON.parse(cpuInvocation?.[5] || "null"), {...vector.job, dev: "cpu"});
    assert.deepEqual(fs.readdirSync(traceDirectory), ["cli.strace", "cpu.strace"]);
    assert.match(fs.readFileSync(path.join(traceDirectory, "cli.strace"), "utf8"), /execve\(/);
    assert.match(fs.readFileSync(path.join(traceDirectory, "cpu.strace"), "utf8"), /execve\(/);
    assert.throws(() => checkNativeRelease(f.entrypoint, "trace"), /runtime-only.*SHA256/i,
      "trace cannot be substituted for the provenance-bearing build check");
  });

  test("Linux runtime trace fails closed on external exec, DSO load and DSO extraction", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {
    const f = fixture(t);
    const attacks = /** @type {[string, string][]} */ ([
      ["external exec", '101 execve("/usr/bin/llc", ["llc"], 0x1) = 0\n'],
      ["DSO load", '101 openat(AT_FDCWD, "/opt/plugin.so", O_RDONLY) = 4\n'],
      ["DSO extraction", '101 creat("/tmp/extracted.so", 0600) = 4\n'],
    ]);
    for (const [label, line] of attacks) {
      let preserved;
      try {
        assert.throws(() => checkNativeExecutable(f.executable, fileHash(f.executable), "trace", {
          spawn: nativeTraceRunner(f, isCli => isCli ? line : ""),
        }), error => {
          if (!(error instanceof Error)) {return false;}
          assert.match(error.message, /unexpected worker launch|external runtime\/compiler/i, label);
          const match = error.message.match(/syscall traces preserved at (.+)$/);
          const logDirectory = match?.[1];
          if (!logDirectory) {return false;}
          preserved = logDirectory;
          assert.deepEqual(fs.readdirSync(preserved), ["cli.strace"]);
          assert.match(fs.readFileSync(path.join(preserved, "cli.strace"), "utf8"), label === "external exec" ? /\/usr\/bin\/llc/ : /\.so/);
          return true;
        });
      } finally {
        if (preserved) {
          assert.equal(path.dirname(preserved), nativeTraceRoot());
          assert.match(path.basename(preserved), /^mom-native-trace-/);
          fs.rmSync(preserved, {recursive: true});
        }
      }
    }
    assert.equal(f.calls.length, 3, "each attack is rejected in the first trace before CPU execution");
  });

  test("Linux runtime trace rejects and preserves an oversized fresh log", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {
    const f = fixture(t); let preserved;
    try {
      assert.throws(() => checkNativeExecutable(f.executable, fileHash(f.executable), "trace", {
        spawn: nativeTraceRunner(f, () => "", true),
      }), error => {
        if (!(error instanceof Error)) {return false;}
        assert.match(error.message, /post-run read limit|bounded regular file/i);
        const match = error.message.match(/syscall traces preserved at (.+)$/);
        const logDirectory = match?.[1];
        if (!logDirectory) {return false;}
        preserved = logDirectory;
        assert.deepEqual(fs.readdirSync(preserved), ["cli.strace"]);
        assert.equal(fs.statSync(path.join(preserved, "cli.strace")).size, (32 << 20) + 1);
        return true;
      });
    } finally {
      if (preserved) {
        assert.equal(path.dirname(preserved), nativeTraceRoot());
        assert.match(path.basename(preserved), /^mom-native-trace-/);
        fs.rmSync(preserved, {recursive: true});
      }
    }
    assert.equal(f.calls.length, 1, "an oversized CLI trace stops before the CPU vector");
  });
});

test.describe("Native release CLI transcript qualification", () => {
  const help = "Options:\n--save_config: file\n";
  /** Structural runner fixture; never execute its synthetic ELF/PE. The one Windows preflight is mocked too.
   * @param {ReturnType<typeof fixture>} f @param {(role: string, index: number) => object} [change] */
  function cliRunner(f, change = () => ({})) {
    let index = 0, preflightDone = false;
    return /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      f.calls.push({command, args, options});
      if (f.record.target === "win32" && !preflightDone && command === f.executable && args.length === 1 && args[0] === "invalid-directive") {
        preflightDone = true;
        return {status: 1, signal: null, stdout: help, stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      const role = command === process.execPath ? "node" : "native", current = Math.floor(index++ / 2);
      return {status: 1, signal: null, stdout: help, stderr: "2026-11-03 12:34:56 ERROR: " + (current === 0 ? "No directive specified" : "Unknown directive invalid-directive") + "\n",
        ...change(role, current)};
    }));
  }
  for (const target of /** @type {("linux" | "win32")[]} */ (["linux", "win32"])) {
    test("compares complete original Node/controller transcripts without invoking hashing on " + target, {
      skip: target === "linux" && process.platform === "win32" && "Linux ELF fixtures require POSIX executable metadata",
    }, t => {
      const f = fixture(t, target), result = checkNativeRelease(f.entrypoint, "cli", {platform: target, arch: "x64", audit: () => undefined, spawn: cliRunner(f)});
      assert.equal(result.suitePassed, true); assert.equal(f.calls.length, target === "win32" ? 5 : 4);
      assert.deepEqual(f.calls.filter(c => c.command === f.executable).map(c => c.args), target === "win32" ?
        [["invalid-directive"], [], ["invalid-directive"]] : [[], ["invalid-directive"]]);
      if (target === "win32") {assert.equal(result.coldStartPreflight?.phase, "windows-cold-start-preflight");} else {assert.equal(result.coldStartPreflight, undefined);}
      for (const call of f.calls) {
        assert.ok(call.options.timeout && call.options.timeout <= 30000);
        assert.equal(call.options.env?.["MOM_COMMAND"], "./mom"); assert.equal(call.options.env?.["MOM_TEST_NO_POOL_NETWORK"], "1");
      }
      const nodeCalls = f.calls.filter(c => c.command === process.execPath);
      assert.deepEqual(nodeCalls[0]?.args, ["--max-old-space-size=128", "--require", path.resolve(__dirname, "common/no-pool-network.js"), path.resolve(__dirname, "../mom.js")]);
    });
  }
  for (const target of /** @type {("linux" | "win32")[]} */ (["linux", "win32"])) {
    test("rejects a truncated formal invalid-directive transcript after any Windows preflight on " + target, {
      skip: target === "linux" && process.platform === "win32" && "Linux ELF fixtures require POSIX executable metadata",
    }, t => {
      const f = fixture(t, target);
      assert.throws(() => checkNativeRelease(f.entrypoint, "cli", {platform: target, arch: "x64", audit: () => undefined,
        spawn: cliRunner(f, (role, index) => role === "native" && index === 1 ? {stdout: "Options:\n"} : {})}), /CLI/);
      assert.equal(f.calls.filter(c => c.command === f.executable).length, target === "win32" ? 3 : 2);
    });
  }
  for (const [name, change] of /** @type {[string, object][]} */ ([
    ["lost stdout", {stdout: ""}], ["truncated stdout", {stdout: "Options:\n"}],
    ["lost stderr", {stderr: ""}], ["wrong status", {status: 0}], ["indeterminate status", {status: null}],
    ["signal", {signal: "SIGTERM"}], ["spawn failure", {error: new Error("fixture failure")}],
  ])) {
    test("rejects " + name + " even if other CLI output looks correct", t => {
      const f = fixture(t);
      assert.throws(() => checkNativeRelease(f.entrypoint, "cli", {audit: () => undefined,
        spawn: cliRunner(f, role => role === "native" ? change : {})}), /CLI/);
    });
  }
  test("normalizes only CRLF and logger timestamps, not missing or different text", t => {
    const f = fixture(t);
    assert.equal(checkNativeRelease(f.entrypoint, "cli", {audit: () => undefined,
      spawn: cliRunner(f, (role, index) => role === "native" ? {stdout: help.replaceAll("\n", "\r\n"),
        stderr: "2026-11-04 05:06:07 ERROR: " + (index === 0 ? "No directive specified" : "Unknown directive invalid-directive") + "\r\n"} : {})}).suitePassed, true);
  });
  test("an incomplete oracle cannot bless identically truncated native help", t => {
    const f = fixture(t);
    assert.throws(() => checkNativeRelease(f.entrypoint, "cli", {audit: () => undefined,
      spawn: cliRunner(f, () => ({stdout: "Options:\n"}))}), /CLI/);
  });
  test("candidate mutation during a CLI run is detected before the next process", t => {
    const f = fixture(t), runner = cliRunner(f);
    const mutate = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args = [],
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      if (f.record.target === "win32" && command === f.executable && args.length === 1 && args[0] === "invalid-directive") {
        return runner(command, args, options);
      }
      fs.appendFileSync(f.executable, "changed"); return runner(command, args, options);
    }));
    assert.throws(() => checkNativeRelease(f.entrypoint, "cli", {audit: () => undefined, spawn: mutate}), /changed during qualification/);
    assert.equal(f.calls.length, f.record.target === "win32" ? 2 : 1);
  });
});

test.describe("Transported packed provenance", {
  skip: process.platform !== "linux" && "UPX transport fixtures require a POSIX executable-mode host; Linux CI covers both targets",
}, () => {
  for (const target of /** @type {const} */ (["linux", "win32"])) {
    test.describe(target, () => {
      const check = (/** @type {ReturnType<typeof transportedFixture>} */ f,
        /** @type {typeof f.spawn} */ spawn = f.spawn) => checkNativeRelease(f.entrypoint, "cpu", {
        platform: target, arch: "x64", audit: () => undefined, spawn});

      test("admits a real packNativeEntrypoint handoff with its raw source relocated", t => {
        const f = transportedFixture(t, target), audited = /** @type {string[]} */ ([]);
        const result = checkNativeRelease(f.entrypoint, "cpu", {platform: target, arch: "x64",
          audit: raw => audited.push(raw), spawn: f.spawn});
        assert.equal(result.suitePassed, true); assert.equal(result.executable, f.executable);
        assert.deepEqual(audited, [f.source]);
        assert.notEqual(f.source, path.join(path.dirname(f.entrypoint) + ".unpacked", path.basename(f.executable)));
      });

      test("rejects a relocated raw image without its original-entrypoint link", t => {
        const f = transportedFixture(t, target); delete f.record["sourceEntrypoint"]; f.write();
        assert.throws(() => check(f), /original entrypoint binding/i); assert.equal(f.calls.length, 0);
      });

      test("rejects a changed original entrypoint", t => {
        const f = transportedFixture(t, target); fs.appendFileSync(f.originEntrypoint, " ");
        assert.throws(() => check(f), /original entrypoint/i); assert.equal(f.calls.length, 0);
      });

      test("rejects a recursively packed original entrypoint", t => {
        const f = transportedFixture(t, target), original = /** @type {FixtureRecord} */ (JSON.parse(fs.readFileSync(f.originEntrypoint, "utf8")));
        original.sourceEntrypoint = {path: f.originEntrypoint, sha256: "0".repeat(64)};
        fs.writeFileSync(f.originEntrypoint, JSON.stringify(original));
        const originLink = f.record.sourceEntrypoint; assert.ok(originLink);
        originLink.sha256 = fileHash(f.originEntrypoint); f.write();
        assert.throws(() => check(f), /original entrypoint does not bind/i); assert.equal(f.calls.length, 0);
      });

      test("rejects a relocated raw source under a symlinked parent when supported", t => {
        const f = transportedFixture(t, target), realParent = path.dirname(f.source), aliasParent = realParent + ".alias";
        try {
          fs.symlinkSync(realParent, aliasParent, process.platform === "win32" ? "junction" : "dir");
        } catch {t.skip("directory symlinks are unavailable"); return;}
        t.after(() => fs.rmSync(aliasParent, {recursive: true, force: true}));
        f.record.source.path = path.join(aliasParent, path.basename(f.source)); f.write();
        assert.throws(() => check(f), /symlinked parent/i); assert.equal(f.calls.length, 0);
      });

      test("rejects a Windows-target raw source under a case-only alias on Linux", {
        skip: process.platform !== "linux" || target !== "win32",
      }, t => {
        const f = transportedFixture(t, target), realParent = path.dirname(f.source), leaf = path.basename(realParent),
          aliasLeaf = leaf.replace(/[a-z]/, letter => letter.toUpperCase()), aliasParent = path.join(path.dirname(realParent), aliasLeaf);
        assert.notEqual(aliasLeaf, leaf, "the fixture path has a lowercase character to flip");
        try {
          fs.symlinkSync(realParent, aliasParent, "dir");
        } catch {t.skip("case-sensitive directory symlinks are unavailable"); return;}
        t.after(() => fs.rmSync(aliasParent, {recursive: true, force: true}));
        f.record.source.path = path.join(aliasParent, path.basename(f.source)); f.write();
        assert.throws(() => check(f), /symlinked parent/i); assert.equal(f.calls.length, 0);
      });

      test("rejects an original entrypoint under a symlinked parent when supported", t => {
        const f = transportedFixture(t, target), aliasParent = f.root + ".origin-alias";
        try {
          fs.symlinkSync(f.root, aliasParent, process.platform === "win32" ? "junction" : "dir");
        } catch {t.skip("directory symlinks are unavailable"); return;}
        t.after(() => fs.rmSync(aliasParent, {recursive: true, force: true}));
        const originLink = f.record.sourceEntrypoint; assert.ok(originLink);
        originLink.path = path.join(aliasParent, "entrypoint.json"); f.write();
        assert.throws(() => check(f), /canonical file/i); assert.equal(f.calls.length, 0);
      });

      test("rejects a Windows-target original entrypoint under a case-only alias on Linux", {
        skip: process.platform !== "linux" || target !== "win32",
      }, t => {
        const f = transportedFixture(t, target), leaf = path.basename(f.root),
          aliasLeaf = leaf.replace(/[a-z]/, letter => letter.toUpperCase()), aliasParent = path.join(path.dirname(f.root), aliasLeaf);
        assert.notEqual(aliasLeaf, leaf, "the fixture path has a lowercase character to flip");
        try {
          fs.symlinkSync(f.root, aliasParent, "dir");
        } catch {t.skip("case-sensitive directory symlinks are unavailable"); return;}
        t.after(() => fs.rmSync(aliasParent, {recursive: true, force: true}));
        const originLink = f.record.sourceEntrypoint; assert.ok(originLink);
        originLink.path = path.join(aliasParent, "entrypoint.json"); f.write();
        assert.throws(() => check(f), /canonical file/i); assert.equal(f.calls.length, 0);
      });

      test("rejects rehashed transported raw bytes that differ from the original raw entrypoint", t => {
        const f = transportedFixture(t, target), changedRaw = fs.readFileSync(f.source);
        const changedIndex = changedRaw.length - 1;
        changedRaw[changedIndex] = (changedRaw[changedIndex] ?? 0) ^ 1; fs.writeFileSync(f.source, changedRaw);
        const hash = fileHash(f.source), source = f.record.source, packingSource = f.packing.source;
        source.sha256 = hash; source.bytes = fs.statSync(f.source).size;
        packingSource.sha256 = hash; packingSource.bytes = fs.statSync(f.source).size; f.packing.restoredSha256 = hash;
        fs.writeFileSync(f.packingFile, JSON.stringify(f.packing));
        const packingReceipt = f.record.packingReceipt; assert.ok(packingReceipt);
        packingReceipt.sha256 = fileHash(f.packingFile); f.write();
        assert.throws(() => check(f), /original entrypoint|provenance chain/i); assert.equal(f.calls.length, 0);
      });

      test("rejects a rehashed transported producer receipt that differs from the original pin", t => {
        const f = transportedFixture(t, target);
        f.producer["configHash"] = "0".repeat(64); fs.writeFileSync(f.producerFile, JSON.stringify(f.producer));
        const hash = fileHash(f.producerFile);
        f.record.producerReceipt.sha256 = hash; f.record.componentHashes[f.producerFile] = hash; f.write();
        assert.throws(() => check(f), /original raw provenance chain/i); assert.equal(f.calls.length, 0);
      });

      test("rejects a rehashed descriptor substitution", t => {
        const f = transportedFixture(t, target), alternate = f.record["configFile"] + ".alternate";
        fs.copyFileSync(f.record["configFile"], alternate);
        fs.appendFileSync(alternate, " ");
        f.record["configFile"] = alternate; f.record["configHash"] = fileHash(alternate); f.write();
        assert.throws(() => check(f), /original raw provenance chain/i); assert.equal(f.calls.length, 0);
      });

      test("rejects a component map that drops the relocated producer", t => {
        const f = transportedFixture(t, target); delete f.record["componentHashes"][f.producerFile]; f.write();
        assert.throws(() => check(f), /component hashes do not preserve/i); assert.equal(f.calls.length, 0);
      });

      test("rejects an extra rehashed component absent from the original map", t => {
        const f = transportedFixture(t, target), extra = path.join(f.root, "extra-component.json");
        fs.writeFileSync(extra, "{}"); f.record["componentHashes"][extra] = fileHash(extra); f.write();
        assert.throws(() => check(f), /component hashes do not preserve/i); assert.equal(f.calls.length, 0);
      });

      test("rejects a changed build recipe", t => {
        const f = transportedFixture(t, target);
        f.record["recipe"] = {version: 1, graph: {fixture: "substituted"}}; f.write();
        assert.throws(() => check(f), /original raw provenance chain/i); assert.equal(f.calls.length, 0);
      });

      test("rejects a generated-descriptor claim absent from the original entrypoint", t => {
        const f = transportedFixture(t, target);
        f.record["generatedDescriptor"] = {path: f.record["configFile"], sha256: f.record["configHash"]}; f.write();
        assert.throws(() => check(f), /original raw provenance chain/i); assert.equal(f.calls.length, 0);
      });

      test("detects original-entrypoint mutation during the vector run", t => {
        const f = transportedFixture(t, target); let mutated = false;
        const spawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((
          /** @type {string} */ command, /** @type {string[]} */ args,
          /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
          const result = f.spawn(command, args, options);
          if (command === process.execPath && !mutated) {mutated = true; fs.appendFileSync(f.originEntrypoint, " ");}
          return result;
        }));
        assert.throws(() => check(f, spawn), /provenance changed during qualification/i);
        assert.equal(mutated, true); assert.equal(f.calls.length, target === "win32" ? 2 : 1);
      });
    });
  }
});

test.describe("Native build-only audit and exact-hash runtime split", () => {
  test("audits a Windows PE candidate on Linux from its target metadata without launching it", t => {
    const f = fixture(t, "win32");
    const result = auditNativeRelease(f.entrypoint);
    assert.equal(result.scope, "build-audit"); assert.equal(result.target, "win32");
    assert.equal(result.entrypoint, f.entrypoint); assert.equal(result.entryHash, fileHash(f.entrypoint));
    assert.equal(result.sourceSha256, fileHash(f.source)); assert.equal(result.sha256, fileHash(f.executable));
    assert.equal(result.bytes, fs.statSync(f.executable).size); assert.equal(result.executable, f.executable);
    assert.equal(result.buildEvidencePassed, true); assert.equal(result.runtimeTested, false);
    assert.equal(Object.hasOwn(result, "coldStartPreflight"), false);
    assert.equal(f.calls.length, 0);
  });

  test("build audit reports the shipped packed hash separately from the raw linked-image hash", t => {
    const f = fixture(t, "win32", true), producerBefore = fs.readFileSync(f.producerFile);
    fs.appendFileSync(f.executable, "packed fixture payload");
    f.packing.artifact = {path: f.executable, bytes: fs.statSync(f.executable).size, sha256: fileHash(f.executable)};
    fs.writeFileSync(f.packingFile, JSON.stringify(f.packing));
    f.record.artifact = f.packing.artifact;
    f.record.packingReceipt = {path: f.packingFile, sha256: fileHash(f.packingFile)};
    f.write();

    const result = auditNativeRelease(f.entrypoint);
    assert.equal(result.scope, "build-audit"); assert.equal(result.target, "win32");
    assert.equal(result.sourceSha256, fileHash(f.source)); assert.equal(result.sha256, fileHash(f.executable));
    assert.notEqual(result.sha256, result.sourceSha256);
    assert.equal(result.bytes, fs.statSync(f.executable).size); assert.equal(result.executable, f.executable);
    assert.equal(result.buildEvidencePassed, true); assert.equal(result.runtimeTested, false);
    assert.deepEqual(fs.readFileSync(f.producerFile), producerBefore);
    assert.equal(f.calls.length, 0);
  });

  test("malformed or stale build evidence fails before raw-image audit", t => {
    const malformed = fixture(t, "win32"); fs.writeFileSync(malformed.entrypoint, "{");
    let audited = false;
    assert.throws(() => auditNativeRelease(malformed.entrypoint, {audit: () => {audited = true;}}), /JSON|receipt/i);
    assert.equal(audited, false); assert.equal(malformed.calls.length, 0);

    const stale = fixture(t, "win32"); fs.appendFileSync(stale.record.configFile, " "); audited = false;
    assert.throws(() => auditNativeRelease(stale.entrypoint, {audit: () => {audited = true;}}), /provenance|descriptor/i);
    assert.equal(audited, false); assert.equal(stale.calls.length, 0);
  });

  test("build audit catches mutation during PE audit and never restamps producer evidence", t => {
    const f = fixture(t, "win32"), beforeEntry = fs.readFileSync(f.entrypoint), beforeProducer = fs.readFileSync(f.producerFile);
    assert.throws(() => auditNativeRelease(f.entrypoint, {audit: raw => fs.appendFileSync(raw, "changed during audit")}), /changed during qualification/);
    assert.deepEqual(fs.readFileSync(f.entrypoint), beforeEntry); assert.deepEqual(fs.readFileSync(f.producerFile), beforeProducer);
    assert.equal(f.calls.length, 0);
  });

  test("build audit detects transient candidate and provenance rewrites after mtime restoration", t => {
    for (const kind of ["artifact", "provenance"]) {
      const f = fixture(t, "win32"), file = kind === "artifact" ? f.executable : f.producerFile;
      setStableMtime(file);
      const beforeHash = fileHash(file), before = fs.statSync(file, {bigint: true});
      assert.throws(() => auditNativeRelease(f.entrypoint, {audit: () => transientRewriteRestore(file)}), /changed during qualification/i);
      const after = fs.statSync(file, {bigint: true});
      assert.equal(fileHash(file), beforeHash, `${kind} bytes are restored`);
      assert.equal(after.mtimeNs, before.mtimeNs, `${kind} mtime is restored`);
      assert.notEqual(after.ctimeNs, before.ctimeNs, `${kind} ctime still records the rewrite`);
      assert.equal(f.calls.length, 0);
    }
  });

  test("build audit refuses platform and runtime-suite overrides", t => {
    const f = fixture(t, "win32");
    for (const options of [{platform: "linux"}, {arch: "arm64"}, {gpu: {vendor: "intel", algo: "pearlhash"}, audit: () => undefined},
      {env: {}}]) {
      assert.throws(() => auditNativeRelease(f.entrypoint, options), /option|unsupported|only|audit/i);
    }
    assert.equal(f.calls.length, 0);
  });

  test("runtime-only CPU checks a relocated executable by explicit SHA without provenance metadata", {skip: !["linux", "win32"].includes(process.platform) || process.arch !== "x64"}, t => {
    const target = process.platform === "win32" ? "win32" : "linux", f = fixture(t, target), relocated = path.join(f.root, target === "win32" ? "transported.exe" : "transported-mom");
    fs.copyFileSync(f.executable, relocated); const expected = fileHash(relocated);
    for (const file of [f.entrypoint, f.record.configFile, f.producerFile]) {fs.unlinkSync(file);}
    assert.equal([f.entrypoint, f.record.configFile, f.producerFile].some(file => fs.existsSync(file)), false);
    const result = checkNativeExecutable(relocated, expected, "cpu", {spawn: f.spawn,
      env: {MOM_TEST_EXECUTABLE: "/old/candidate", MOM_TEST_NO_POOL_NETWORK: "0", MOM_GPU_BACKEND: "nvidia", NODE_OPTIONS: "--require=untrusted.js"}});
    assert.equal(result.scope, "runtime-only"); assert.equal(result.target, target);
    assert.equal(result.buildProvenanceChecked, false); assert.equal(result.publicationQualified, false);
    assert.equal(result.suitePassed, true); assert.equal(result.executable, relocated); assert.equal(result.sha256, expected);
    assert.equal(f.calls.length, target === "win32" ? 2 : 1);
    if (target === "win32") {assert.equal(result.coldStartPreflight?.executable, relocated);}
    const call = f.calls.find(row => row.command === process.execPath); assert.ok(call);
    assert.deepEqual(call.args, [path.resolve(__dirname, "run-hash.js"), "cpu"]);
    const env = call.options.env || {};
    assert.equal(env["MOM_TEST_EXECUTABLE"], relocated); assert.equal(env["MOM_TEST_NO_POOL_NETWORK"], "1");
    assert.equal(env["MOM_SKIP_MSR"], "1"); assert.equal(env["MOM_GPU_BACKEND"], "");
    assert.equal(Object.hasOwn(env, "NODE_OPTIONS"), false);
  });

  test("missing, malformed or mismatched expected hashes fail before runtime invocation", {skip: !["linux", "win32"].includes(process.platform) || process.arch !== "x64"}, t => {
    const target = process.platform === "win32" ? "win32" : "linux", f = fixture(t, target);
    const missingHash = /** @type {string} */ (/** @type {unknown} */ (undefined));
    for (const hash of [missingHash, "", "A".repeat(64), "g".repeat(64), "0".repeat(64)]) {
      assert.throws(() => checkNativeExecutable(f.executable, hash, "cpu", {spawn: f.spawn}), /SHA|hash|digest/i);
    }
    assert.equal(f.calls.length, 0);
  });

  test("wrong host format and symlinked runtime inputs fail before launch", {skip: !["linux", "win32"].includes(process.platform) || process.arch !== "x64"}, t => {
    const target = process.platform === "win32" ? "win32" : "linux", f = fixture(t, target), wrong = path.join(f.root, "wrong-format");
    fs.writeFileSync(wrong, "not a native executable");
    assert.throws(() => checkNativeExecutable(wrong, fileHash(wrong), "cpu", {spawn: f.spawn}), /ELF|PE|executable/i);
    if (process.platform !== "win32") {
      const link = path.join(f.root, "linked-runtime"); fs.symlinkSync(f.executable, link);
      assert.throws(() => checkNativeExecutable(link, fileHash(f.executable), "cpu", {spawn: f.spawn}), /regular|symlink|link/i);
    }
    assert.equal(f.calls.length, 0);
  });

  test("a runtime executable changed by the runner cannot retain a pass", {skip: !["linux", "win32"].includes(process.platform) || process.arch !== "x64"}, t => {
    const target = process.platform === "win32" ? "win32" : "linux", f = fixture(t, target), relocated = path.join(f.root, target === "win32" ? "mutable.exe" : "mutable-mom");
    fs.copyFileSync(f.executable, relocated);
    const mutate = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args = [],
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      const result = f.spawn(command, args, options);
      if (!(command === relocated && args?.length === 1 && args[0] === "invalid-directive")) {fs.appendFileSync(relocated, "changed during run");}
      return result;
    }));
    assert.throws(() => checkNativeExecutable(relocated, fileHash(relocated), "cpu", {spawn: mutate}), /changed during qualification/i);
    assert.equal(f.calls.length, target === "win32" ? 2 : 1);
  });

  test("runtime executable transient rewrite is detected after restoring bytes and mtime", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {
    const f = fixture(t), relocated = path.join(f.root, "transient-mom");
    fs.copyFileSync(f.executable, relocated); setStableMtime(relocated);
    const expected = fileHash(relocated), before = fs.statSync(relocated, {bigint: true});
    const mutate = /** @type {typeof f.spawn} */ ((command, args, options) => {
      const result = f.spawn(command, args, options); transientRewriteRestore(relocated); return result;
    });
    assert.throws(() => checkNativeExecutable(relocated, expected, "cpu", {spawn: mutate}), /changed during qualification/i);
    const after = fs.statSync(relocated, {bigint: true});
    assert.equal(fileHash(relocated), expected, "runtime bytes are restored");
    assert.equal(after.mtimeNs, before.mtimeNs, "runtime mtime is restored");
    assert.notEqual(after.ctimeNs, before.ctimeNs, "runtime ctime still records the rewrite");
    assert.equal(f.calls.length, 1);
  });

  test("runtime-only GPU lane still requires one explicit supported vendor and algorithm", {skip: !["linux", "win32"].includes(process.platform) || process.arch !== "x64"}, t => {
    const target = process.platform === "win32" ? "win32" : "linux", f = fixture(t, target), vectors = hashTests.filter(row => row.gpu && !row.portableOnly && row.job.algo === "pearlhash").length,
      summary = `ℹ tests ${vectors + 2}\nℹ suites 1\nℹ pass ${vectors + 2}\nℹ fail 0\nℹ cancelled 0\nℹ skipped 0\n`;
    const spawn = /** @type {typeof f.spawn} */ (/** @type {unknown} */ ((/** @type {string} */ command, /** @type {string[]} */ args,
      /** @type {{cwd?: string, env?: NodeJS.ProcessEnv, timeout?: number}} */ options) => {
      f.calls.push({command, args, options});
      if (target === "win32" && command === f.executable && args.length === 1 && args[0] === "invalid-directive") {
        return {status: 1, signal: null, error: undefined, stdout: "Options:\n--save_config: file\n",
          stderr: "ERROR: Unknown directive invalid-directive\n"};
      }
      return {status: 0, signal: null, stdout: summary, stderr: ""};
    }));
    const selected = {vendor: "intel", algo: "pearlhash"};
    const result = checkNativeExecutable(f.executable, fileHash(f.executable), "gpu", {gpu: selected, spawn});
    assert.equal(result.suitePassed, true); assert.deepEqual(result.gpu, selected); assert.equal(f.calls.length, target === "win32" ? 2 : 1);
    if (target === "win32") {assert.equal(result.coldStartPreflight?.executable, f.executable);}
    assert.equal(f.calls.at(-1)?.options.env?.["MOM_REQUIRE_GPU_TESTS"], "1");
    assert.throws(() => checkNativeExecutable(f.executable, fileHash(f.executable), "gpu", {spawn}), /explicit supported vendor and algorithm/);
    for (const suite of ["cpu", "cli"]) {
      assert.throws(() => checkNativeExecutable(f.executable, fileHash(f.executable), suite, {gpu: selected, spawn}), /only to the gpu suite/);
    }
    assert.throws(() => checkNativeExecutable(f.executable, fileHash(f.executable), "gpu", {gpu: {vendor: "bad", algo: "pearlhash"}, spawn}), /explicit supported vendor and algorithm/);
    assert.throws(() => checkNativeExecutable(f.executable, fileHash(f.executable), "gpu", {gpu: {vendor: "intel", algo: "unknown"}, spawn}), /explicit supported vendor and algorithm/);
    assert.equal(f.calls.length, target === "win32" ? 2 : 1);
  });

  test("runtime-only API rejects build-audit and target override options", {skip: !["linux", "win32"].includes(process.platform) || process.arch !== "x64"}, t => {
    const target = process.platform === "win32" ? "win32" : "linux", f = fixture(t, target), hash = fileHash(f.executable);
    for (const options of [{platform: "win32"}, {arch: "arm64"}, {audit: () => undefined}, {archive: {path: "/tmp/a.zip", tool: "/tmp/cmake"}}]) {
      assert.throws(() => checkNativeExecutable(f.executable, hash, "cpu", {...options, spawn: f.spawn}), /option|unsupported|runtime/i);
    }
    assert.equal(f.calls.length, 0);
  });

  test("CLI audit and runtime hash parsing fail closed without running the structural fixture", t => {
    const f = fixture(t, "win32");
    const audit = runReleaseCli([f.entrypoint, "audit"]);
    assert.equal(audit.status, 0, audit.stderr);
    const report = JSON.parse(audit.stdout);
    assert.equal(report.scope, "build-audit"); assert.equal(report.runtimeTested, false);
    assert.equal(report.archiveChecked, false); assert.equal(report.publicationQualified, false);
    assert.equal(report.sha256, fileHash(f.executable));

    const invalidHash = runReleaseCli([f.executable, "cpu", "--sha256", "g".repeat(64)]);
    assert.equal(invalidHash.status, 1);
    assert.match(invalidHash.stderr, /explicit expected SHA256/i);
    assert.doesNotMatch(invalidHash.stdout + invalidHash.stderr, /suite passed/i);

    const traceWithoutHash = runReleaseCli([f.executable, "trace"]);
    assert.equal(traceWithoutHash.status, 1);
    assert.match(traceWithoutHash.stderr, /trace is runtime-only.*SHA256/i);
    const malformedTraceHash = runReleaseCli([f.executable, "trace", "--sha256", "g".repeat(64)]);
    assert.equal(malformedTraceHash.status, 1);
    assert.match(malformedTraceHash.stderr, process.platform === "win32" ? /supports Linux only/i : /explicit expected SHA256/i);
    assert.doesNotMatch(malformedTraceHash.stdout + malformedTraceHash.stderr, /suite passed/i);

    const malformed = [
      [f.executable, "cpu", "0".repeat(64), "--sha256"],
      [f.entrypoint, "audit", "--tool", "/tmp/cmake", "--archive", "/tmp/package.zip"],
      [f.executable, "gpu", "--vendor", "intel", "--algo", "pearlhash", "--sha256", "g".repeat(64)],
    ];
    for (const args of malformed) {
      const result = runReleaseCli(args);
      assert.equal(result.status, 1);
      assert.match(result.stderr, /Usage:/i);
      assert.doesNotMatch(result.stdout + result.stderr, /suite passed/i);
    }
    assert.equal(f.calls.length, 0, "the structural PE fixture is never executed");
  });
});
