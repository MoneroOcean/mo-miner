"use strict";
const {describe, test} = require("node:test"), assert = require("node:assert/strict"),
  fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {Readable} = require("node:stream"), {spawnSync} = require("node:child_process");
const {fileHash} = require("../scripts/build-source-inputs"), {opensslPin} = require("../scripts/build-native-openssl"),
  {buildLocalWindowsSupportCommand, runLocalWindowsSupportCommandCli} = require("../scripts/build-local-native-windows-support-command");

/** Inert native-stage fixtures. Never invokes Windows, WSL, tools, networking or mining.
 * @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-support-command-")), workspace = path.join(root, "support"),
    archive = path.join(root, "hermes.tar"), calls = /** @type {string[]} */ ([]);
  fs.mkdirSync(workspace); fs.writeFileSync(archive, Buffer.alloc(1024));
  const input = {workspace, exportWorkspace: path.join(root, "exports"), tools: /** @type {any} */ ({}),
      hermesArchive: {path: archive, sha256: fileHash(archive), rootName: "hermes-source", originalRoot: "/verified/workspace/hermes-source/roots/hermes"}},
    io = /** @type {Record<string,{path:string}>} */ ({}), ssl = /** @type {Record<string,{path:string}>} */ ({});
  let hermes = /** @type {any} */ (undefined);
  const deps = /** @type {Parameters<typeof buildLocalWindowsSupportCommand>[1]} */ ({platform: "win32", environment: {SystemRoot: "C:\\Windows"},
    validateTools() {return input.tools;},
    admitWorkspace() {
      return {workspace, marker: path.join(workspace, "marker"), artifacts: path.join(workspace, "artifacts"),
        io: path.join(workspace, "io"), openssl: path.join(workspace, "openssl"), hermes: path.join(workspace, "hermes"),
        coreRuns: path.join(workspace, "core-runs"), descriptors: path.join(workspace, "descriptors")};
    },
    /** @param {string} output @param {NodeJS.ProcessEnv} _env @param {boolean} allowed */
    async fetchIo(output, _env, allowed) {
      calls.push("fetch-io"); assert.equal(allowed, true); fs.mkdirSync(output);
      for (const name of ["libuv", "zlib"]) {const file = path.join(output, name + ".tar.gz"); fs.writeFileSync(file, name); io[name] = {path: file};}
      fs.writeFileSync(path.join(output, "fetch.json"), "original Windows io receipt\n");
      return /** @type {any} */ ({});
    },
    verifyIo() {calls.push("verify-io"); return /** @type {any} */ ({archives: io});},
    /** @param {string} output */
    async fetchOpenSsl(output) {
      calls.push("fetch-openssl"); fs.mkdirSync(output); const file = path.join(output, "openssl-" + opensslPin.version + ".tar.gz");
      fs.writeFileSync(file, "ssl"); ssl["openssl"] = {path: file}; fs.writeFileSync(path.join(output, "fetch.json"), "original Windows ssl receipt\n");
      return /** @type {any} */ ({});
    },
    verifyOpenSsl() {calls.push("verify-openssl"); return /** @type {any} */ ({archives: ssl});},
    /** @param {any} state @param {string} _tar @param {number} timeout @param {string} originalRoot */
    extractHermes(state, _tar, timeout, originalRoot) {
      calls.push("extract-hermes"); assert.equal(timeout, 300000); assert.equal(originalRoot, input.hermesArchive.originalRoot);
      fs.mkdirSync(state.hermesExtractRoot); const fetchFile = path.join(state.hermesExtractRoot, "fetch.json"), rootsFile = path.join(state.hermesExtractRoot, "roots.json");
      fs.writeFileSync(fetchFile, "original POSIX fetch receipt\n"); fs.writeFileSync(rootsFile, originalRoot + "\n");
      hermes = {source: state.hermesExtractRoot, fetchFile, rootsFile}; return hermes;
    },
    /** @param {any} state @param {string} originalRoot */
    readHermes(state, originalRoot) {
      calls.push("read-hermes"); assert.deepEqual(Object.keys(state).sort(), ["hermesExtractRoot", "hermesSourceName"]);
      assert.equal(originalRoot, input.hermesArchive.originalRoot); return hermes;
    },
    /** @param {any} value @param {any} options */
    buildSupport(value, options) {
      calls.push("support"); assert.equal(options.environment.MOM_BUILD_JOBS, "2"); assert.equal(options.environment.MOM_AOT_BUILD_ONLY, "1");
      assert.equal(value.sources.hermes, hermes.source);
      const coreReceipt = path.join(workspace, "inputs.json"); fs.writeFileSync(coreReceipt, "original core receipt\n");
      return /** @type {any} */ ({core: {archive: path.join(workspace, "core.lib"), receipt: coreReceipt},
        managedOutputCandidates: {io: [], openssl: [], hermes: [], core: []}});
    },
    /** @param {any} _value @param {any} options */
    exportSupport(_value, options) {
      calls.push("export"); assert.equal(fs.readFileSync(options.sourceReceipts.hermesRoots, "utf8"), input.hermesArchive.originalRoot + "\n");
      assert.match(fs.readFileSync(options.sourceReceipts.ioFetch, "utf8"), /original Windows/);
      return {localBuildId: "12345678-1234-4234-9234-123456789abc", io: {root: "io", handoffSha256: "1".repeat(64)},
        hermes: {root: "hermes", handoffSha256: "2".repeat(64)}, retirementCandidates: []};
    },
  });
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, workspace, archive, input, deps, calls};
}

describe("Native Windows support command composition", () => {
  test("preserves original source receipts and reuses source caches on a warm run", async t => {
    const f = fixture(t), first = await buildLocalWindowsSupportCommand(f.input, f.deps), before = f.calls.length,
      second = await buildLocalWindowsSupportCommand(f.input, f.deps);
    assert.deepEqual(first.controllerSupport, second.controllerSupport);
    assert.deepEqual(f.calls.slice(0, before), ["fetch-io", "verify-io", "fetch-openssl", "verify-openssl", "extract-hermes", "support", "export"]);
    assert.deepEqual(f.calls.slice(before), ["verify-io", "verify-openssl", "read-hermes", "support", "export"]);
    assert.equal(first.coreReceipt.sha256, fileHash(first.coreReceipt.path));
    assert.deepEqual(first.controllerSupport.hermes.admitted, {kind: "hermes", localBuildId: first.controllerSupport.localBuildId});
  });
  test("rejects archive substitution before any source acquisition or producer call", async t => {
    const f = fixture(t); fs.appendFileSync(f.archive, "changed");
    await assert.rejects(buildLocalWindowsSupportCommand(f.input, f.deps), /original WSL pin/);
    assert.deepEqual(f.calls, []);
  });
  test("rejects unsupported hosts and malformed source roots without acquiring inputs", async t => {
    const f = fixture(t);
    await assert.rejects(buildLocalWindowsSupportCommand(f.input, {...f.deps, platform: "linux"}), /Windows x64/);
    await assert.rejects(buildLocalWindowsSupportCommand({...f.input, hermesArchive: {...f.input.hermesArchive, originalRoot: "/a/../b"}}, f.deps), /explicit workspaces/);
    await assert.rejects(buildLocalWindowsSupportCommand({...f.input, exportWorkspace: f.workspace}, f.deps), /must be disjoint/);
    assert.deepEqual(f.calls, []);
  });
});

/** @param {ReturnType<typeof fixture>} f */
function cliFiles(f) {
  const input = {workspace: f.input.workspace, exportWorkspace: f.input.exportWorkspace, hermesArchive: f.input.hermesArchive},
    inputFile = path.join(f.root, "command-input.json"), outputFile = path.join(f.root, "command-result.json");
  fs.writeFileSync(inputFile, JSON.stringify(input) + "\n");
  const tools = {core: {nodeGyp: "C:\\Tools\\node-gyp.js", msbuild: "C:\\Tools\\MSBuild.exe", dumpbin: "C:\\Tools\\dumpbin.exe",
    python: "C:\\Tools\\python.exe", compiler: "C:\\Tools\\icx-cl.exe"},
  io: {cmake: "C:\\Tools\\cmake.exe", cc: "C:\\Tools\\icx-cl.exe", ninja: "C:\\Tools\\ninja.exe"},
  openssl: {ar: "C:\\Tools\\lib.exe", cc: "C:\\Tools\\icx-cl.exe", link: "C:\\Tools\\link.exe", make: "C:\\Tools\\nmake.exe",
    perl: "C:\\Tools\\perl.exe", rc: "C:\\Tools\\rc.exe", tar: "C:\\Windows\\System32\\tar.exe"},
  hermes: {git: "C:\\Tools\\git.exe", cmake: "C:\\Tools\\cmake.exe", cc: "C:\\Tools\\icx-cl.exe", cxx: "C:\\Tools\\icx-cl.exe",
    ninja: "C:\\Tools\\ninja.exe"}};
  return {input, inputFile, outputFile, tools};
}
/** @typedef {Awaited<ReturnType<typeof buildLocalWindowsSupportCommand>>} LocalWindowsSupportResult */
/** @param {ReturnType<typeof fixture>} f @returns {LocalWindowsSupportResult} */
function cliResult(f) {
  const localBuildId = "12345678-1234-4234-9234-123456789abc";
  return {controllerSupport: {localBuildId,
    io: {root: path.join(f.root, "export/io"), handoffSha256: "1".repeat(64), admitted: {kind: "io", localBuildId}},
    hermes: {root: path.join(f.root, "export/hermes"), handoffSha256: "2".repeat(64), admitted: {kind: "hermes", localBuildId}}},
  coreArchive: path.join(f.root, "support/core.lib"), coreReceipt: {path: path.join(f.root, "support/inputs.json"), sha256: "3".repeat(64)},
  supportRetirementCandidates: {io: [], openssl: [], hermes: [], core: []}, exportRetirementCandidates: []};
}

describe("Native Windows support command CLI", () => {
  test("composes exact stdin tools and writes only the successful minimal result", async t => {
    const f = fixture(t), files = cliFiles(f), result = cliResult(f), inputHash = fileHash(files.inputFile); let calls = 0;
    const returned = await runLocalWindowsSupportCommandCli(["--run", files.inputFile, files.outputFile], {
      stdin: Readable.from([Buffer.from(JSON.stringify(files.tools) + "\n")]),
      async buildCommand(request) {
        calls++;
        assert.deepEqual(request, {...files.input, tools: files.tools});
        return result;
      },
    });
    assert.equal(calls, 1); assert.deepEqual(returned, result); assert.equal(fileHash(files.inputFile), inputHash);
    assert.deepEqual(JSON.parse(fs.readFileSync(files.outputFile, "utf8")), result);
    assert.equal(fs.lstatSync(files.outputFile).nlink, 1);
  });

  test("rejects malformed arguments, input schemas, and aliased input before building", async t => {
    const f = fixture(t), files = cliFiles(f); let calls = 0;
    const deps = {stdin: Readable.from([JSON.stringify(files.tools)]), buildCommand: async () => {calls++; return cliResult(f);}};
    await assert.rejects(runLocalWindowsSupportCommandCli([files.inputFile, files.outputFile], deps), /Usage/);
    fs.writeFileSync(files.inputFile, JSON.stringify({...files.input, tools: files.tools}));
    await assert.rejects(runLocalWindowsSupportCommandCli(["--run", files.inputFile, files.outputFile], deps), /unknown or missing fields/);
    const alias = path.join(f.root, "input-hardlink.json"); fs.linkSync(files.inputFile, alias);
    fs.writeFileSync(files.inputFile, JSON.stringify(files.input));
    await assert.rejects(runLocalWindowsSupportCommandCli(["--run", alias, files.outputFile], deps), /bounded, canonical, unaliased/);
    assert.equal(calls, 0); assert.equal(fs.existsSync(files.outputFile), false);
  });

  test("keeps reads bounded when the input grows after its opening snapshot", async t => {
    const f = fixture(t), files = cliFiles(f), original = fs.readSync;
    let grew = false, largest = 0;
    t.mock.method(fs, "readSync", /** @param {...any} args */ (...args) => {
      largest = Math.max(largest, args[1].length);
      if (!grew) {grew = true; fs.appendFileSync(files.inputFile, " ".repeat(128 << 10));}
      return Reflect.apply(original, fs, args);
    });
    await assert.rejects(runLocalWindowsSupportCommandCli(["--run", files.inputFile, files.outputFile], {
      stdin: Readable.from([JSON.stringify(files.tools)]), buildCommand: async () => cliResult(f),
    }), /changed during its bounded read/);
    assert.ok(largest > 0 && largest <= (64 << 10) + 1);
    assert.equal(fs.existsSync(files.outputFile), false);
  });

  test("rejects oversized or invalid UTF-8 tools JSON without creating output", async t => {
    const f = fixture(t), files = cliFiles(f), inputHash = fileHash(files.inputFile);
    await assert.rejects(runLocalWindowsSupportCommandCli(["--run", files.inputFile, files.outputFile], {
      stdin: Readable.from([Buffer.alloc(128 * 1024 + 1)]), buildCommand: async () => cliResult(f),
    }), /exceeds its byte bound/);
    await assert.rejects(runLocalWindowsSupportCommandCli(["--run", files.inputFile, files.outputFile], {
      stdin: Readable.from([Buffer.from([0xff])]), buildCommand: async () => cliResult(f),
    }), /valid UTF-8 JSON/);
    assert.equal(fileHash(files.inputFile), inputHash); assert.equal(fs.existsSync(files.outputFile), false);
  });

  test("leaves no result after producer failure, input drift, or an existing output target", async t => {
    const f = fixture(t), files = cliFiles(f), tools = Readable.from([JSON.stringify(files.tools)]);
    await assert.rejects(runLocalWindowsSupportCommandCli(["--run", files.inputFile, files.outputFile], {
      stdin: tools, buildCommand: async () => {throw new Error("fixture producer failure");},
    }), /fixture producer failure/);
    assert.equal(fs.existsSync(files.outputFile), false);
    await assert.rejects(runLocalWindowsSupportCommandCli(["--run", files.inputFile, files.outputFile], {
      stdin: Readable.from([JSON.stringify(files.tools)]),
      async buildCommand() {fs.appendFileSync(files.inputFile, " "); return cliResult(f);},
    }), /input changed during the native build/);
    assert.equal(fs.existsSync(files.outputFile), false);
    fs.writeFileSync(files.outputFile, "keep me\n");
    await assert.rejects(runLocalWindowsSupportCommandCli(["--run", files.inputFile, files.outputFile], {
      stdin: Readable.from([JSON.stringify(files.tools)]), buildCommand: async () => cliResult(f),
    }), /result path must be new/);
    assert.equal(fs.readFileSync(files.outputFile, "utf8"), "keep me\n");
  });

  test("the real CLI fails closed on unsupported hosts without output", t => {
    if (process.platform === "win32" && process.arch === "x64") {t.skip("Windows x64 is a supported host"); return;}
    const f = fixture(t), files = cliFiles(f), cli = path.resolve(__dirname, "../scripts/build-local-native-windows-support-command.js"),
      result = spawnSync(process.execPath, [cli, "--run", files.inputFile, files.outputFile], {input: JSON.stringify(files.tools), encoding: "utf8"});
    assert.equal(result.status, 1); assert.match(result.stderr, /requires Windows x64/); assert.equal(fs.existsSync(files.outputFile), false);
  });
});
