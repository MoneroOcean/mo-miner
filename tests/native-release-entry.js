"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {buildNativeArtifact, nativeReleaseArguments} = require("../scripts/build-release"), {fileHash} = require("../scripts/build-source-inputs");
const {assembleNativeNotices} = require("../scripts/prepare-native-notices");
const linux = require("../scripts/build-native-release"), windows = require("../scripts/build-native-windows-controller"), packer = require("../scripts/pack-native-executable");

/** Stub only existing compiler/packer boundaries; the public orchestration is real.
 * No compiler, VM, GPU, network connection or resulting miner is executed.
 * @param {import("node:test").TestContext} t @param {boolean} [win] */
function fixture(t, win = false) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-entry-$-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const config = path.join(root, "sdk.json"), output = path.join(root, "output"), tool = path.join(root, "packer");
  fs.writeFileSync(config, JSON.stringify(win ? {version: 1, windowsSdk: {}, application: {}} : {version: 1})); fs.writeFileSync(tool, "pinned fixture");
  const packing = {upx: tool, upxSha256: fileHash(tool)}, state = {fail: false, changed: false, probe: false, executed: false, badOutput: false,
    missingReceipt: false, badReceiptHash: false, wrongReceiptTarget: false, receiptChangedDuringPack: false,
    packFail: false, packChanged: false, changedSource: false, changedArtifact: false, packMissingReceipt: false,
    packMissingLog: false, packBadLogHash: false, packMismatchReceipt: false, noticeChanged: ""};
  const calls = /** @type {{kind: string, config: string, output: string, env?: NodeJS.ProcessEnv}[]} */ ([]);
  const packs = /** @type {Parameters<typeof packer.packNativeRelease>[0][]} */ ([]);
  const build = (/** @type {string} */ kind) => (/** @type {string} */ input, /** @type {string} */ out, /** @type {NodeJS.ProcessEnv} */ env) => {
    calls.push({kind, config: input, output: out, env}); if (state.fail) {throw new Error("fixture compile failed");}
    fs.mkdirSync(out); const file = path.join(out, kind === "windows" ? "mom.exe" : "mom"); fs.writeFileSync(file, "source-built fixture");
    if (state.changed) {fs.appendFileSync(config, " ");}
    if (state.noticeChanged) {fs.appendFileSync(state.noticeChanged, " ");}
    if (!state.missingReceipt) {
      const sha256 = fileHash(file), receipt = kind === "windows" ? {version: 1, configHash: fileHash(input),
        artifact: {path: file, bytes: fs.statSync(file).size, sha256}, target: state.wrongReceiptTarget ? "x86_64-w64-windows-gnu" : "x86_64-pc-windows-msvc",
        buildOnly: true, publicationQualified: false, executed: false} : {version: require("../package.json").version,
        configHash: fileHash(input), binary: file, sha256: state.badReceiptHash ? "0".repeat(64) : sha256, bytes: fs.statSync(file).size, buildOnly: true, publicationQualified: false};
      fs.writeFileSync(path.join(out, "result.json"), JSON.stringify(receipt) + "\n");
    }
    return kind === "windows" ? {artifact: {path: state.badOutput ? "relative" : file}, application: state.probe ? null : {}, executed: state.executed, buildOnly: true} : {binary: state.badOutput ? "relative" : file};
  };
  t.mock.method(linux, "buildNativeRelease", /** @type {typeof linux.buildNativeRelease} */ (/** @type {unknown} */ (build("linux"))));
  t.mock.method(windows, "buildNativeWindowsController", /** @type {typeof windows.buildNativeWindowsController} */ (/** @type {unknown} */ (build("windows"))));
  t.mock.method(packer, "packNativeRelease", /** @type {typeof packer.packNativeRelease} */ (/** @type {unknown} */ ((/** @type {Parameters<typeof packer.packNativeRelease>[0]} */ options) => {
    packs.push(options); if (state.packFail) {throw new Error("fixture packing failed");}
    fs.mkdirSync(options.output); const file = path.join(options.output, win ? "mom.exe" : "mom"); fs.copyFileSync(options.source, file);
    if (state.receiptChangedDuringPack) {fs.appendFileSync(path.join(path.dirname(options.source), "result.json"), " ");}
    if (state.packChanged) {fs.appendFileSync(config, " ");}
    const sha256 = fileHash(file), sourceSha256 = fileHash(options.source), logs = Object.fromEntries(["pack", "integrity", "restore"]
      .filter(stage => !(state.packMissingLog && stage === "restore")).map(stage => {
        const log = path.join(options.output, stage + ".log"); fs.writeFileSync(log, stage + " fixture\n");
        return [stage, {path: log, sha256: state.packBadLogHash && stage === "integrity" ? "0".repeat(64) : fileHash(log)}];
      }));
    const result = {source: {path: options.source, bytes: fs.statSync(options.source).size, sha256: sourceSha256},
      artifact: {path: file, bytes: fs.statSync(file).size, sha256}, restoredSha256: sourceSha256,
      tool: {path: options.upx, sha256: options.upxSha256}, payloadAudit: {exactOriginalBytes: true}, logs,
      recipe: {files: {}, graph: {}}, commands: [], buildOnly: true, publicationQualified: false};
    if (!state.packMissingReceipt) {
      const receipt = state.packMismatchReceipt ? {...result, scope: "tampered"} : result;
      fs.writeFileSync(path.join(options.output, "result.json"), JSON.stringify(receipt) + "\n");
    }
    if (state.changedSource) {fs.appendFileSync(options.source, " ");} if (state.changedArtifact) {fs.appendFileSync(file, " ");}
    return result;
  })));
  return {root, config, output, packing, calls, packs, state, run: (pack = false, env = {}) => buildNativeArtifact(config, output, pack ? packing : undefined, env)};
}

test.describe("Native release entrypoint SDK admission", () => {
  test("relative SDK paths fail before either compiler", t => {const f = fixture(t); assert.throws(() => buildNativeArtifact("relative", f.output), /absolute/); assert.equal(f.calls.length, 0);});
  test("control characters in output paths are refused", t => {const f = fixture(t); assert.throws(() => buildNativeArtifact(f.config, f.output + "\n"), /absolute/); assert.equal(f.calls.length, 0);});
  test("an existing destination is never overwritten", t => {const f = fixture(t); fs.mkdirSync(f.output); assert.throws(f.run, /new output/); assert.equal(f.calls.length, 0);});
  test("a symlinked missing output ancestor cannot redirect into maintained sources", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-output-alias-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const source = fs.realpathSync(path.join(__dirname, "..", "runtime")), alias = path.join(root, "source-alias"),
      relative = path.join("native-output-" + path.basename(root), "deep", "candidate"), output = path.join(alias, relative);
    fs.symlinkSync(source, alias, "dir");
    assert.throws(() => linux.buildNativeRelease(path.join(root, "missing-sdk.json"), output), /cannot enter a maintained source tree/);
    assert.equal(fs.lstatSync(path.join(source, relative), {throwIfNoEntry: false}), undefined);
  });
  test("malformed JSON cannot start a compiler", t => {const f = fixture(t); fs.writeFileSync(f.config, "{"); assert.throws(f.run); assert.equal(f.calls.length, 0);});
  test("unknown SDK versions are rejected before creating output", t => {const f = fixture(t); fs.writeFileSync(f.config, '{"version":2}'); assert.throws(f.run, /descriptor/); assert.ok(!fs.existsSync(f.output));});
  test("Windows controller-only probes cannot become native releases", t => {const f = fixture(t, true); fs.writeFileSync(f.config, '{"version":1,"windowsSdk":{}}'); assert.throws(f.run, /Core and all GPU/); assert.equal(f.calls.length, 0);});
  test("Windows release output names are fixed before any compiler", t => {const f = fixture(t, true); fs.writeFileSync(f.config, '{"version":1,"windowsSdk":{},"application":{},"outputName":"other.exe"}'); assert.throws(f.run, /output must be mom.exe/); assert.equal(f.calls.length, 0);});
  test("host-specific build overrides use the existing release policy", t => {const f = fixture(t); assert.throws(() => f.run(false, {MOM_CPU_MARCH: "native"}), /MOM_CPU_MARCH/); assert.equal(f.calls.length, 0);});
  test("packing requires an absolute tool path before compilation", t => {const f = fixture(t); f.packing.upx = "relative"; assert.throws(() => f.run(true), /pinned tool/); assert.equal(f.calls.length, 0);});
  test("null packing options cannot silently select the uncompressed route", t => {
    const f = fixture(t); assert.throws(() => buildNativeArtifact(f.config, f.output, /** @type {{upx: string, upxSha256: string}} */ (/** @type {unknown} */ (null))), /pinned tool/);
    assert.equal(f.calls.length, 0);
  });
  test("packing pins cannot be empty or stale", t => {const f = fixture(t); f.packing.upxSha256 = ""; assert.throws(() => f.run(true), /pinned tool/); assert.equal(f.calls.length, 0);});
  test("changed tool bytes fail before an expensive build", t => {const f = fixture(t); fs.appendFileSync(f.packing.upx, "changed"); assert.throws(() => f.run(true), /pinned tool/); assert.equal(f.calls.length, 0);});
  test("old unpacked evidence is preserved instead of silently reused", t => {const f = fixture(t); fs.mkdirSync(f.output + ".unpacked"); assert.throws(() => f.run(true), /new unpacked/); assert.equal(f.calls.length, 0);});
  test("packing options cannot override the source-built artifact or destination", t => {
    const f = fixture(t); Object.assign(f.packing, {source: "/unrelated", output: "/unrelated"});
    assert.throws(() => f.run(true), /pinned tool/); assert.equal(f.calls.length, 0);
  });
});

test.describe("Native distribution notice source binding", () => {
  /** @param {ReturnType<typeof fixture>} f @param {string} [content] */
  const select = (f, content = "Synthetic third-party fixture notice\n") => {
    const source = path.join(f.root, "selected-notices.txt"); fs.writeFileSync(source, content);
    const config = JSON.parse(fs.readFileSync(f.config, "utf8"));
    config.notices = {path: source, sha256: fileHash(source)}; fs.writeFileSync(f.config, JSON.stringify(config));
    return source;
  };
  /** @param {ReturnType<typeof fixture>} f */
  const assembledFixture = f => {
    const source = path.join(f.root, "reviewed-license.txt"), directory = path.join(f.root, "notice-assembly");
    fs.writeFileSync(source, "Synthetic reviewed license text\n");
    const assembly = assembleNativeNotices([{component: "Synthetic component", path: source, sha256: fileHash(source)}], directory);
    return {assembly, receiptPath: path.join(directory, "native-notices.json")};
  };
  for (const win of [false, true]) {
    test((win ? "Windows" : "Linux") + " copies only the selected notice data and binds its bytes", t => {
      const f = fixture(t, win), source = select(f); f.run(true);
      const entry = JSON.parse(fs.readFileSync(path.join(f.output, "entrypoint.json"), "utf8"));
      const output = path.join(f.output, "THIRD-PARTY-NOTICES.txt");
      assert.deepEqual(entry.notices, {path: output, sha256: fileHash(source), bytes: fs.statSync(source).size});
      assert.equal(fileHash(output), fileHash(source));
    });
  }
  test("Windows assembled notice receipts are staged and bound to the entrypoint", t => {
    const f = fixture(t, true), {assembly, receiptPath} = assembledFixture(f), receiptSha256 = fileHash(receiptPath),
      config = JSON.parse(fs.readFileSync(f.config, "utf8"));
    config.notices = {path: assembly.notice.path, sha256: assembly.notice.sha256, receipt: {path: receiptPath, sha256: receiptSha256}};
    fs.writeFileSync(f.config, JSON.stringify(config));

    f.run();

    const entry = JSON.parse(fs.readFileSync(path.join(f.output, "entrypoint.json"), "utf8")),
      output = path.join(f.output, "THIRD-PARTY-NOTICES.txt");
    assert.equal(f.calls[0]?.kind, "windows");
    assert.deepEqual(entry.notices, {path: output, sha256: assembly.notice.sha256, bytes: assembly.notice.bytes});
    assert.equal(fileHash(output), assembly.notice.sha256);
    assert.deepEqual(entry.noticeAssembly, {receiptPath, receiptSha256, fingerprint: assembly.fingerprint, notice: assembly.notice});
  });
  test("Windows assembled notice receipt pin tampering fails before the build", t => {
    const f = fixture(t, true), {assembly, receiptPath} = assembledFixture(f), config = JSON.parse(fs.readFileSync(f.config, "utf8"));
    config.notices = {path: assembly.notice.path, sha256: assembly.notice.sha256, receipt: {path: receiptPath, sha256: "0".repeat(64)}};
    fs.writeFileSync(f.config, JSON.stringify(config));

    assert.throws(f.run, /external SHA-256/);
    assert.equal(f.calls.length, 0);
    assert.equal(fs.existsSync(f.output), false);
  });
  test("unselected notices are absent rather than presented as a complete inventory", t => {
    const f = fixture(t); f.run(); const entry = JSON.parse(fs.readFileSync(path.join(f.output, "entrypoint.json"), "utf8"));
    assert.equal(Object.hasOwn(entry, "notices"), false);
  });
  test("stale notice hashes fail before any expensive compilation", t => {
    const f = fixture(t), source = select(f); fs.appendFileSync(source, "changed");
    assert.throws(f.run, /notice/i); assert.equal(f.calls.length, 0); assert.equal(fs.existsSync(f.output), false);
  });
  test("empty and binary notices cannot masquerade as distribution text", t => {
    for (const content of ["", "notices\0payload"]) {
      const f = fixture(t); select(f, content); assert.throws(f.run, /notice/i); assert.equal(f.calls.length, 0);
    }
  });
  test("malformed UTF-8 and oversized notice sources fail before compilation", t => {
    for (const oversized of [false, true]) {
      const f = fixture(t), source = select(f);
      if (oversized) {fs.truncateSync(source, (8 << 20) + 1);} else {fs.writeFileSync(source, Buffer.from([0xff, 0xff]));}
      const config = JSON.parse(fs.readFileSync(f.config, "utf8"));
      config.notices.sha256 = fileHash(source); fs.writeFileSync(f.config, JSON.stringify(config));
      assert.throws(f.run, /notice/i); assert.equal(f.calls.length, 0);
    }
  });
  test("notice bytes changing during compilation prevent an entrypoint", t => {
    const f = fixture(t); f.state.noticeChanged = select(f);
    assert.throws(f.run, /notice/i); assert.equal(fs.existsSync(path.join(f.output, "entrypoint.json")), false);
  });
  test("relative notice paths cannot be inherited from a build working directory", t => {
    const f = fixture(t); select(f); const config = JSON.parse(fs.readFileSync(f.config, "utf8"));
    config.notices.path = "relative.txt"; fs.writeFileSync(f.config, JSON.stringify(config));
    assert.throws(f.run, /notice/i); assert.equal(f.calls.length, 0);
  });
  test("notice symlinks cannot redirect the selected source", {skip: process.platform === "win32"}, t => {
    const f = fixture(t), source = select(f); fs.renameSync(source, source + ".real"); fs.symlinkSync(source + ".real", source);
    assert.throws(f.run, /notice/i); assert.equal(f.calls.length, 0);
  });
});

test.describe("Native release entrypoint shared source builders", () => {
  for (const win of [false, true]) {
    test((win ? "Windows" : "Linux") + " dispatch reuses its original source builder and explicit paths", t => {
      const f = fixture(t, win), result = f.run(); assert.equal(f.calls.length, 1); assert.equal(f.calls[0]?.kind, win ? "windows" : "linux");
      assert.equal(f.calls[0]?.config, f.config); assert.equal(f.calls[0]?.output, f.output); assert.equal(f.calls[0]?.env?.["MOM_AOT_BUILD_ONLY"], "1");
      assert.equal(f.packs.length, 0); assert.equal(result.buildOnly, true); assert.equal(result.publicationQualified, false);
    });
    test((win ? "Windows" : "Linux") + " packs only the newly source-built executable and retains unpacked evidence", t => {
      const f = fixture(t, win), result = f.run(true); assert.equal(f.calls[0]?.output, f.output + ".unpacked");
      assert.equal(f.packs[0]?.source, path.join(f.output + ".unpacked", win ? "mom.exe" : "mom")); assert.equal(f.packs[0]?.upxSha256, f.packing.upxSha256);
      assert.equal(result.artifact.path, path.join(f.output, win ? "mom.exe" : "mom")); assert.ok(fs.existsSync(f.output + ".unpacked"));
      const receipt = JSON.parse(fs.readFileSync(path.join(f.output, "entrypoint.json"), "utf8"));
      assert.equal(receipt.configHash, fileHash(f.config)); assert.equal(receipt.source.sha256, fileHash(f.packs[0]?.source || ""));
      assert.equal(receipt.artifact.sha256, fileHash(result.artifact.path)); assert.equal(receipt.publicationQualified, false);
      assert.deepEqual(receipt.producerReceipt, {path: path.join(f.output + ".unpacked", "result.json"), sha256: fileHash(path.join(f.output + ".unpacked", "result.json"))});
      assert.equal(receipt.componentHashes[receipt.producerReceipt.path], receipt.producerReceipt.sha256);
      assert.deepEqual(receipt.packingReceipt, {path: path.join(f.output, "result.json"), sha256: fileHash(path.join(f.output, "result.json"))});
    });
  }
  test("unpacked entrypoints do not claim packing evidence", t => {
    const f = fixture(t); f.run(false);
    const receipt = JSON.parse(fs.readFileSync(path.join(f.output, "entrypoint.json"), "utf8"));
    assert.ok(!Object.hasOwn(receipt, "packingReceipt"));
  });
  test("missing producer receipts fail before packing", t => {const f = fixture(t); f.state.missingReceipt = true; assert.throws(() => f.run(true), /did not write its build receipt/); assert.equal(f.packs.length, 0);});
  test("producer receipt binary hashes and Windows targets must match before packing", t => {
    const badHash = fixture(t); badHash.state.badReceiptHash = true;
    assert.throws(() => badHash.run(true), /producer receipt does not match/); assert.equal(badHash.packs.length, 0);
    const badTarget = fixture(t, true); badTarget.state.wrongReceiptTarget = true;
    assert.throws(() => badTarget.run(true), /producer receipt does not match/); assert.equal(badTarget.packs.length, 0);
  });
  test("producer receipt mutation during packing prevents entrypoint publication", t => {
    const f = fixture(t); f.state.receiptChangedDuringPack = true;
    assert.throws(() => f.run(true), /producer receipt changed during packing/); assert.equal(f.packs.length, 1);
    assert.ok(!fs.existsSync(path.join(f.output, "entrypoint.json")));
  });
  test("failed compilation never reaches packing", t => {const f = fixture(t); f.state.fail = true; assert.throws(() => f.run(true), /compile failed/); assert.equal(f.packs.length, 0);});
  test("a returned controller probe is not mistaken for the full Windows application", t => {const f = fixture(t, true); f.state.probe = true; assert.throws(f.run, /full application/); assert.equal(f.packs.length, 0);});
  test("a builder which executed its result violates the build-only boundary", t => {const f = fixture(t, true); f.state.executed = true; assert.throws(f.run, /full application/);});
  test("relative builder output cannot become a native release", t => {const f = fixture(t); f.state.badOutput = true; assert.throws(f.run, /expected executable/);});
  test("a changed SDK descriptor fails before packing", t => {const f = fixture(t); f.state.changed = true; assert.throws(() => f.run(true), /descriptor changed/); assert.equal(f.packs.length, 0);});
  test("packing failures propagate without fallback to raw publication", t => {const f = fixture(t); f.state.packFail = true; assert.throws(() => f.run(true), /packing failed/); assert.ok(!fs.existsSync(f.output));});
  test("a descriptor changed during packing cannot pass final admission", t => {const f = fixture(t); f.state.packChanged = true; assert.throws(() => f.run(true), /descriptor changed during packing/);});
  test("a changed unpacked input cannot be paired with an earlier packed artifact", t => {const f = fixture(t); f.state.changedSource = true; assert.throws(() => f.run(true), /artifact chain/);});
  test("a changed packed artifact cannot be returned using an earlier digest", t => {const f = fixture(t); f.state.changedArtifact = true; assert.throws(() => f.run(true), /artifact chain/);});
  test("missing packing receipts or logs cannot publish an entrypoint", t => {
    const noReceipt = fixture(t); noReceipt.state.packMissingReceipt = true;
    assert.throws(() => noReceipt.run(true), /packer did not write a regular result receipt/);
    assert.ok(!fs.existsSync(path.join(noReceipt.output, "entrypoint.json")));
    const noLog = fixture(t); noLog.state.packMissingLog = true;
    assert.throws(() => noLog.run(true), /incomplete log evidence/);
    assert.ok(!fs.existsSync(path.join(noLog.output, "entrypoint.json")));
  });
  test("tampered pack log hashes or a mismatched saved result cannot publish an entrypoint", t => {
    const badLog = fixture(t); badLog.state.packBadLogHash = true;
    assert.throws(() => badLog.run(true), /packing log is missing or changed: integrity/);
    assert.ok(!fs.existsSync(path.join(badLog.output, "entrypoint.json")));
    const badReceipt = fixture(t); badReceipt.state.packMismatchReceipt = true;
    assert.throws(() => badReceipt.run(true), /packing result receipt differs from the packed candidate/);
    assert.ok(!fs.existsSync(path.join(badReceipt.output, "entrypoint.json")));
  });
  for (const [at, phase] of /** @type {[number, string][]} */ ([[2, "before"], [3, "after"]])) {
    test("entrypoint recipe changes " + phase + " packing fail closed", t => {
      const f = fixture(t), inputs = require("../scripts/build-source-inputs"), original = inputs.sourceInputs; let calls = 0;
      t.mock.method(inputs, "sourceInputs", (/** @type {string} */ root, /** @type {string[]} */ dirs = [], /** @type {string[]} */ leaves = []) => {
        const result = original(root, dirs, leaves); if (++calls === at) {result.graph["scripts/build-release.js"] = "0".repeat(64);} return result;
      });
      assert.throws(() => f.run(true), /Native release entrypoint source inputs are stale/); assert.equal(f.packs.length, phase === "before" ? 0 : 1);
    });
  }
});

test.describe("Native release command-line dispatch", () => {
  test("default native preparation rejects invalid versions without a legacy archive fallback", t => {
    const f = fixture(t), preload = path.join(f.root, "no-compiler.cjs"), cli = path.join(__dirname, "../scripts/build-release.js");
    fs.writeFileSync(preload, 'require("node:child_process").spawnSync=()=>{console.log("LEGACY_ARCHIVE");return {status:0};};' +
      'if(process.env.FIXTURE_RELEASE_PLATFORM)Object.defineProperty(process,"platform",{value:process.env.FIXTURE_RELEASE_PLATFORM});');
    /** @type {NodeJS.ProcessEnv} */
    const env = {...process.env, MOM_RELEASE_VERSION: "../bad"};
    for (const key of ["MOM_NATIVE_SDK", "MOM_NATIVE_OUTPUT", "MOM_NATIVE_UPX", "MOM_NATIVE_UPX_SHA256"]) {delete env[key];}
    for (const platform of ["", "win32"]) {
      const result = spawnSync(process.execPath, ["--require", preload, cli], {
        encoding: "utf8", timeout: 10000, env: {...env, FIXTURE_RELEASE_PLATFORM: platform}});
      assert.notEqual(result.status, 0); assert.match(result.stderr, /Invalid release version/i);
      assert.doesNotMatch(result.stdout, /LEGACY_ARCHIVE|Built mom-v/);
    }
  });
  for (const mode of ["explicit", "environment", "environment-unpacked"]) {
    test(`the actual CLI uses ${mode} native inputs without an archive-builder fallback`, t => {
      const f = fixture(t), preload = path.join(f.root, "preload.cjs"), cli = path.join(__dirname, "../scripts/build-release.js");
      const source = [
        '"use strict";',
        'const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");',
        `const native=require(${JSON.stringify(require.resolve("../scripts/build-native-release"))});`,
        'native.buildNativeRelease=(config,out)=>{fs.mkdirSync(out);const binary=path.join(out,"mom");fs.writeFileSync(binary,"fixture");const hash=file=>crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");const receipt={version:require(' + JSON.stringify(require.resolve("../package.json")) + ').version,configHash:hash(config),binary,sha256:hash(binary),bytes:fs.statSync(binary).size,buildOnly:true,publicationQualified:false};fs.writeFileSync(path.join(out,"result.json"),JSON.stringify(receipt));return {binary};};',
        `const pack=require(${JSON.stringify(require.resolve("../scripts/pack-native-executable"))});`,
        'pack.packNativeRelease=({source,output,upx,upxSha256})=>{fs.mkdirSync(output);const file=path.join(output,"mom");fs.copyFileSync(source,file);const hash=f=>crypto.createHash("sha256").update(fs.readFileSync(f)).digest("hex"),sourceHash=hash(source),logs={};for(const stage of ["pack","integrity","restore"]){const log=path.join(output,stage+".log");fs.writeFileSync(log,stage+" fixture\\n");logs[stage]={path:log,sha256:hash(log)};}const result={source:{path:source,bytes:fs.statSync(source).size,sha256:sourceHash},artifact:{path:file,bytes:fs.statSync(file).size,sha256:hash(file)},restoredSha256:sourceHash,tool:{path:upx,sha256:upxSha256},payloadAudit:{exactOriginalBytes:true},logs,recipe:{files:{},graph:{}},commands:[],buildOnly:true,publicationQualified:false};fs.writeFileSync(path.join(output,"result.json"),JSON.stringify(result));return result;};',
        'require("node:child_process").spawnSync=()=>{throw Error("Legacy archive producer must not run for a configured native build");};',
      ].join("\n");
      fs.writeFileSync(preload, source);
      const packed = mode !== "environment-unpacked", env = {...process.env, MOM_NATIVE_SDK: f.config, MOM_NATIVE_OUTPUT: f.output,
        MOM_NATIVE_UPX: packed ? f.packing.upx : undefined, MOM_NATIVE_UPX_SHA256: packed ? f.packing.upxSha256 : undefined};
      const args = mode === "explicit" ? ["--native", f.config, f.output, "--pack", f.packing.upx, f.packing.upxSha256] : [];
      const result = spawnSync(process.execPath, ["--require", preload, cli, ...args], {encoding: "utf8", timeout: 10000, env});
      assert.equal(result.status, 0, result.stderr); assert.match(result.stdout, /NOT executed or publication-qualified/);
      const receipt = JSON.parse(fs.readFileSync(path.join(f.output, "entrypoint.json"), "utf8"));
      assert.equal(receipt.source.path, path.join(packed ? f.output + ".unpacked" : f.output, "mom")); assert.equal(receipt.artifact.path, path.join(f.output, "mom"));
      assert.equal(receipt.publicationQualified, false);
      assert.equal(receipt.producerReceipt.path, path.join(packed ? f.output + ".unpacked" : f.output, "result.json"));
      assert.equal(receipt.componentHashes[receipt.producerReceipt.path], receipt.producerReceipt.sha256);
      if (packed) {assert.equal(receipt.packingReceipt.path, path.join(f.output, "result.json")); assert.equal(receipt.packingReceipt.sha256, fileHash(receipt.packingReceipt.path));} else {assert.ok(!Object.hasOwn(receipt, "packingReceipt"));}
    });
  }
  test("unexpected native flags fail before touching a descriptor or output", t => {
    const f = fixture(t), cli = path.join(__dirname, "../scripts/build-release.js");
    const result = spawnSync(process.execPath, [cli, "--native", f.config, f.output, "--unexpected"], {encoding: "utf8", timeout: 10000});
    assert.notEqual(result.status, 0); assert.match(result.stderr, /Usage: npm run build:release -- --native/); assert.ok(!fs.existsSync(f.output));
  });
});

test.describe("Native release environment selection", () => {
  const env = {MOM_NATIVE_SDK: "/sdk/selected.json", MOM_NATIVE_OUTPUT: "/new/output"};
  test("configured source builds do not require repeating the SDK on the command line", () => {
    assert.deepEqual(nativeReleaseArguments([], env), ["--native", env.MOM_NATIVE_SDK, env.MOM_NATIVE_OUTPUT]);
  });
  test("both pinned packing inputs are carried unchanged", () => {
    assert.deepEqual(nativeReleaseArguments([], {...env, MOM_NATIVE_UPX: "/tools/upx", MOM_NATIVE_UPX_SHA256: "a".repeat(64)}),
      ["--native", env.MOM_NATIVE_SDK, env.MOM_NATIVE_OUTPUT, "--pack", "/tools/upx", "a".repeat(64)]);
  });
  test("explicit native CLI arguments override installed environment defaults", () => {
    const args = ["--native", "/other/sdk.json", "/other/output"];
    assert.deepEqual(nativeReleaseArguments(args, {...env, MOM_NATIVE_UPX: "unselected"}), args);
  });
  for (const key of ["MOM_NATIVE_SDK", "MOM_NATIVE_OUTPUT", "MOM_NATIVE_UPX", "MOM_NATIVE_UPX_SHA256"]) {
    test(`an isolated ${key} cannot start a legacy archive build`, () => {assert.throws(() => nativeReleaseArguments([], {[key]: "only"}), /Native release settings/);});
  }
  test("empty explicit native settings cannot enable a legacy fallback", () => {
    for (const key of Object.keys(env)) {assert.throws(() => nativeReleaseArguments([], {...env, [key]: ""}), /Native release settings/);}
  });
  test("partial or empty packer settings are not silently dropped", () => {
    for (const fields of [{MOM_NATIVE_UPX: "/upx"}, {MOM_NATIVE_UPX_SHA256: "a".repeat(64)}, {MOM_NATIVE_UPX: "", MOM_NATIVE_UPX_SHA256: ""}]) {
      assert.throws(() => nativeReleaseArguments([], {...env, ...fields}), /both MOM_NATIVE_UPX/);
    }
  });
  test("archive-only flags cannot override a configured native SDK", () => {
    for (const flag of ["--clean", "--package-only", "--unknown"]) {assert.throws(() => nativeReleaseArguments([flag], env), /do not mix archive flags/);}
  });
});
