"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {packNativeRelease} = require("../scripts/pack-native-release"), {fileHash} = require("../scripts/build-source-inputs");
const {sealNativePePages} = require("../scripts/native-pe-pages"), {nativePeFixture} = require("./common/native-pe-fixture");

/** Synthetic command-boundary fixture only; no compiler, packer or miner is executed.
 * @param {import("node:test").TestContext} t @param {boolean} [windows] */
function fixture(t, windows = false) {
  t.mock.method(console, "log", () => undefined);
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-pack fixture-$-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const source = path.join(root, windows ? "raw.exe" : "raw"), upx = path.join(root, "packer"), output = path.join(root, "output");
  fs.writeFileSync(source, windows ? nativePeFixture().bytes : Buffer.concat([Buffer.from("7f454c46", "hex"), Buffer.alloc(4092, 42)]));
  if (windows) {
    const data = fs.readFileSync(source), f = nativePeFixture();
    // One real-shaped read-only OS import: page-map fixtures alone have no loader tables.
    data.writeUInt32LE(0x40000040, f.table + 40 + 36);
    data.writeUInt32LE(8192, f.optional + 120); data.writeUInt32LE(40, f.optional + 124);
    data.writeUInt32LE(8256, 8192); data.writeUInt32LE(8352, 8204); data.writeUInt32LE(8288, 8208);
    data.writeBigUInt64LE(8320n, 8256); data.writeBigUInt64LE(8320n, 8288);
    data.write("ExitProcess\0", 8322); data.write("KERNEL32.dll\0", 8352);
    fs.writeFileSync(source, data); sealNativePePages(source);
  }
  fs.writeFileSync(upx, "synthetic build tool, never executed");
  const options = {source, output, upx, upxSha256: fileHash(upx)};
  const calls = /** @type {{command: string, args: string[], timeout: number | undefined}[]} */ ([]);
  const state = {failure: "", signal: false, error: false, corrupt: false, changeSource: false, changeTool: false, removeLog: "", failureLog: ""};
  /** @param {string} command @param {string[]} args @param {import("node:child_process").SpawnSyncOptions} settings */
  function spawn(command, args, settings) {
    calls.push({command, args, timeout: settings.timeout});
    const stage = args.includes("--best") ? "pack" : args.includes("-t") ? "integrity" : "restore";
    if (state.failure === stage) {
      const log = settings.stdio?.[1]; if (typeof log === "number") {fs.writeSync(log, state.failureLog);}
      return {status: state.signal || state.error ? null : 1, signal: state.signal ? "SIGTERM" : null, error: state.error ? new Error("fixture timeout") : undefined};
    }
    const log = settings.stdio?.[1]; if (typeof log === "number") {fs.writeSync(log, stage + " fixture output\n");}
    const destination = args[args.indexOf("-o") + 1], input = args.at(-1);
    if (stage !== "integrity") {
      assert.ok(destination && input); fs.copyFileSync(input, destination);
      if (stage === "restore" && state.corrupt) {const data = fs.readFileSync(destination), index = data.length > 4200 ? 4200 : 100; data[index] = (data[index] || 0) ^ 1; fs.writeFileSync(destination, data);}
    }
    if (stage === "restore" && state.removeLog) {fs.unlinkSync(path.join(output, state.removeLog + ".log"));}
    if (stage === "restore" && state.changeSource) {fs.appendFileSync(source, "changed");}
    if (stage === "restore" && state.changeTool) {fs.appendFileSync(upx, "changed");}
    return {status: 0, signal: null};
  }
  const run = () => packNativeRelease(options, /** @type {typeof import("node:child_process").spawnSync} */ (/** @type {unknown} */ (spawn)));
  return {root, options, calls, state, run};
}

test.describe("Native release packing admission", () => {
  test("relative input paths fail before any command or output creation", t => {
    const f = fixture(t); f.options.source = "relative"; assert.throws(f.run, /absolute/); assert.equal(f.calls.length, 0); assert.ok(!fs.existsSync(f.options.output));
  });
  test("control characters are refused in trusted tool paths", t => {const f = fixture(t); f.options.upx += "\n"; assert.throws(f.run, /absolute/); assert.equal(f.calls.length, 0);});
  test("an existing output directory and its data are preserved", t => {
    const f = fixture(t); fs.mkdirSync(f.options.output); const keep = path.join(f.options.output, "keep"); fs.writeFileSync(keep, "keep");
    assert.throws(f.run, /new output/); assert.equal(fs.readFileSync(keep, "utf8"), "keep"); assert.equal(f.calls.length, 0);
  });
  test("a malformed tool digest is not accepted as an optional pin", t => {const f = fixture(t); f.options.upxSha256 = ""; assert.throws(f.run, /explicit/); assert.equal(f.calls.length, 0);});
  test("different tool contents fail before output mutation", t => {const f = fixture(t); fs.appendFileSync(f.options.upx, "changed"); assert.throws(f.run, /tool hash/); assert.ok(!fs.existsSync(f.options.output));});
  test("missing input files never invoke the packer", t => {const f = fixture(t); fs.unlinkSync(f.options.source); assert.throws(f.run, /ENOENT/); assert.equal(f.calls.length, 0);});
  test("directory inputs are not treated as executable files", t => {const f = fixture(t); f.options.source = f.root; assert.throws(f.run, /regular file/); assert.equal(f.calls.length, 0);});
  test("truncated inputs fail before output creation", t => {const f = fixture(t); fs.writeFileSync(f.options.source, "ELF"); assert.throws(f.run, /Truncated/); assert.ok(!fs.existsSync(f.options.output));});
  test("unsupported executable formats are not guessed", t => {const f = fixture(t); fs.writeFileSync(f.options.source, "not an executable"); assert.throws(f.run, /Unsupported/); assert.equal(f.calls.length, 0);});
  test("unsealed Windows protection maps are refused", t => {const f = fixture(t, true); fs.writeFileSync(f.options.source, nativePeFixture().bytes); assert.throws(f.run, /map/); assert.equal(f.calls.length, 0);});
  test("a corrupted Windows protection map is never repaired silently", t => {
    const f = fixture(t, true), data = fs.readFileSync(f.options.source); data[12328] = (data[12328] || 0) ^ 1; fs.writeFileSync(f.options.source, data);
    assert.throws(f.run, /map/); assert.equal(f.calls.length, 0); assert.ok(data.equals(fs.readFileSync(f.options.source)));
  });
  test("Windows workspace overflow fails before UPX or destination creation, not as a host OOM", t => {
    const f = fixture(t, true), data = fs.readFileSync(f.options.source), {optional} = nativePeFixture();
    data.writeUInt32LE(0x30000000 - 16384 - 40 + 1, optional + 8); fs.writeFileSync(f.options.source, data); const before = fileHash(f.options.source);
    assert.throws(f.run, /UPX workspace.*805306369.*805306368/); assert.equal(f.calls.length, 0);
    assert.ok(!fs.existsSync(f.options.output)); assert.equal(fileHash(f.options.source), before);
  });
  test("Windows workspace exactly at the pinned UPX bound remains admissible", t => {
    const f = fixture(t, true), data = fs.readFileSync(f.options.source), {optional} = nativePeFixture();
    data.writeUInt32LE(0x30000000 - 16384 - 40, optional + 8); fs.writeFileSync(f.options.source, data);
    assert.equal(f.run().buildOnly, true); assert.equal(f.calls.length, 3);
  });
  test("Windows workspace includes every saved loader directory without integer wrapping", t => {
    const f = fixture(t, true), data = fs.readFileSync(f.options.source), {optional} = nativePeFixture();
    data.writeUInt32LE(0x30000000 - 16384 - 5 * 1024 + 1, optional + 8);
    for (const index of [1, 11, 12, 13, 5]) {data.writeUInt32LE(1024, optional + 112 + index * 8 + 4);}
    fs.writeFileSync(f.options.source, data); assert.throws(f.run, /UPX workspace 805306369/); assert.equal(f.calls.length, 0);
  });
  test("unsigned PE size fields cannot wrap around the workspace bound", t => {
    const f = fixture(t, true), data = fs.readFileSync(f.options.source), {optional} = nativePeFixture();
    data.writeUInt32LE(0xffffffff, optional + 8); fs.writeFileSync(f.options.source, data);
    assert.throws(f.run, /UPX workspace/); assert.equal(f.calls.length, 0); assert.ok(!fs.existsSync(f.options.output));
  });
});

test.describe("Native release packing command and payload boundaries", () => {
  test("Linux uses maximum LZMA effort with a fixed dictionary, exact restoration and literal paths only", t => {
    const f = fixture(t), r = f.run(); assert.equal(f.calls.length, 3);
    assert.deepEqual(f.calls[0]?.args, ["--no-env", "--best", "--lzma", "--crp-lzma-ds=134217728", "--exact", "--no-progress", "-o", r.artifact.path, f.options.source]);
    assert.deepEqual(f.calls.map(c => c.timeout), [1200000, 60000, 120000]);
    assert.deepEqual(f.calls[1]?.args, ["--no-env", "-t", r.artifact.path]); assert.deepEqual(f.calls[2]?.args, ["--no-env", "-d", "-o", path.join(f.options.output, "restored"), r.artifact.path]);
    assert.ok(f.calls.every(c => c.command === f.options.upx));
  });
  test("Windows retains relocations and exports and audits every initialized payload", t => {
    const f = fixture(t, true), r = f.run();
    assert.deepEqual(f.calls[0]?.args.slice(0, 7), ["--no-env", "--best", "--lzma", "--crp-lzma-ds=134217728", "--strip-relocs=0", "--compress-exports=0", "--no-progress"]);
    assert.ok(r.artifact.path.endsWith("mom.exe")); assert.ok("payloads" in r.payloadAudit); assert.equal(r.restoredSha256, r.source.sha256);
  });
  test("the raw source is retained and receipts do not claim runtime or publication qualification", t => {
    const f = fixture(t), before = fileHash(f.options.source), r = f.run(); assert.equal(fileHash(f.options.source), before);
    assert.equal(r.buildOnly, true); assert.equal(r.publicationQualified, false); assert.equal(r.source.sha256, before);
    assert.equal(r.artifact.sha256, fileHash(r.artifact.path)); assert.equal(r.tool.sha256, f.options.upxSha256);
    assert.equal(JSON.parse(fs.readFileSync(path.join(f.options.output, "result.json"), "utf8")).artifact.sha256, r.artifact.sha256);
    assert.deepEqual(r.logs, Object.fromEntries(["pack", "integrity", "restore"].map(stage => {
      const file = path.join(f.options.output, stage + ".log"); return [stage, {path: file, sha256: fileHash(file)}];
    })));
    assert.match(r.scope, /no miner execution/);
  });
  for (const windows of [false, true]) {
    test((windows ? "Windows" : "Linux") + " successful packing discards only its verified round-trip copy", t => {
      const f = fixture(t, windows), r = f.run();
      assert.ok(fs.existsSync(f.options.source)); assert.ok(fs.existsSync(r.artifact.path));
      assert.ok(!fs.existsSync(path.join(f.options.output, windows ? "restored.exe" : "restored")));
      for (const name of ["pack.log", "integrity.log", "restore.log", "result.json"]) {assert.ok(fs.existsSync(path.join(f.options.output, name)));}
    });
  }
  test("failed round-trip verification retains the rejected bytes for diagnosis", t => {
    const f = fixture(t); f.state.corrupt = true;
    assert.throws(f.run, /exact original bytes/);
    assert.ok(fs.existsSync(path.join(f.options.output, "restored")));
    assert.ok(!fs.existsSync(path.join(f.options.output, "result.json")));
  });
  test("missing saved stage logs cannot receive a packing receipt", t => {
    const f = fixture(t); f.state.removeLog = "integrity";
    assert.throws(f.run, /Missing native packing stage log: integrity/);
    assert.ok(!fs.existsSync(path.join(f.options.output, "result.json")));
  });
  for (const [stage, count] of /** @type {[string, number][]} */ ([["pack", 1], ["integrity", 2], ["restore", 3]])) {
    test(stage + " failure is preserved without retry, fallback or successful receipt", t => {
      const f = fixture(t); f.state.failure = stage; assert.throws(f.run, /stage failed/); assert.equal(f.calls.length, count);
      assert.ok(!fs.existsSync(path.join(f.options.output, "result.json"))); assert.ok(fs.existsSync(path.join(f.options.output, stage + ".log")));
    });
  }
  test("failure messages contain a bounded diagnostic tail while preserving the full log", t => {
    const f = fixture(t); f.state.failure = "pack"; f.state.failureLog = "discarded-prefix" + "x".repeat(20000) + "CantPackException: file not executable\n";
    assert.throws(f.run, error => {
      assert.ok(error instanceof Error); assert.match(error.message, /exit 1, signal none/);
      assert.match(error.message, /CantPackException: file not executable/);
      assert.ok(!error.message.includes("discarded-prefix")); assert.ok(error.message.length < 17000); return true;
    });
    assert.equal(fs.readFileSync(path.join(f.options.output, "pack.log"), "utf8"), f.state.failureLog);
    assert.ok(!fs.existsSync(path.join(f.options.output, "result.json")));
  });
  test("signals are not mistaken for successful pack completion", t => {const f = fixture(t); f.state.failure = "pack"; f.state.signal = true; assert.throws(f.run, /signal SIGTERM/); assert.equal(f.calls.length, 1);});
  test("timed-out tools fail without running the miner or a second packing attempt", t => {const f = fixture(t); f.state.failure = "pack"; f.state.error = true; assert.throws(f.run, /stage failed/); assert.equal(f.calls.length, 1);});
  test("nonidentical Linux restoration fails even after tool integrity succeeds", t => {const f = fixture(t); f.state.corrupt = true; assert.throws(f.run, /exact original bytes/); assert.ok(!fs.existsSync(path.join(f.options.output, "result.json")));});
  test("changed Windows code bytes cannot pass semantic restoration checks", t => {const f = fixture(t, true); f.state.corrupt = true; assert.throws(f.run, /payload/); assert.ok(!fs.existsSync(path.join(f.options.output, "result.json")));});
  test("a source changed during packing cannot receive a fresh receipt", t => {const f = fixture(t); f.state.changeSource = true; assert.throws(f.run, /inputs changed/);});
  test("a tool changed during packing cannot receive a fresh receipt", t => {const f = fixture(t); f.state.changeTool = true; assert.throws(f.run, /inputs changed/);});
});
