"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"),
  path = require("node:path"), {createHash, randomUUID} = require("node:crypto");
const {fileHash, sourceInputs} = require("../scripts/build-source-inputs"),
  {linuxLlvmToolHandoffInputs, exportLinuxLlvmTools, linuxLlvmAcppFrontendInputs, exportLinuxLlvmAcppFrontend} = require("../scripts/native-linux-llvm-tools");
const {windowsLlvmSdkPlan} = require("../scripts/build-native-windows-llvm21-sdk"), {archiveName} = require("../scripts/native-llvm-sdk");

const recipeFiles = ["scripts/native-linux-llvm-tools.js", "scripts/build-native-llvm21-sdk.js", "scripts/native-llvm-sdk.js",
  "scripts/native-audit.js", "scripts/build-source-inputs.js"];
const toolFiles = ["bin/lld", "bin/llvm-nm", "bin/llvm-ar", "bin/llvm-readobj", "bin/llvm-objcopy",
  "bin/llvm-rc", "bin/llvm-ml", "bin/llvm-tblgen", "bin/clang-tblgen"];
const aliases = {"bin/lld-link": "lld", "bin/llvm-lib": "llvm-ar", "bin/llvm-ranlib": "llvm-ar"};
const receiptNames = ["result.json", "configured.json", "install.json", "build-complete.json"];
const fixtureArchiveName = "llvm-project-fixture.tar.xz";

/** @param {import('node:test').TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-linux-llvm-tools-test-")), admitted = path.join(root, "admitted"),
    sdkRoot = path.join(admitted, "sdk"), bin = path.join(sdkRoot, "bin"), bundle = path.join(root, "bundle"),
    resourceRoot = path.join(sdkRoot, "lib/clang/21/include"),
    attempt = String(Date.now()) + String(process.pid) + "-1", runId = randomUUID(), archivePath = path.join(admitted, fixtureArchiveName),
    archiveBytes = Buffer.from("fixture-only locked archive bytes\n"), archiveSha256 = createHash("sha256").update(archiveBytes).digest("hex"),
    role = "linux-llvm21-source-sdk", plan = {output: admitted, prefix: sdkRoot, archive: archivePath}, planSha256 = "a".repeat(64),
    configure = {fixture: "configure"}, sourceState = {fixture: "source"}, inputFiles = {fixture: "inputs"}, archives = 0;
  const policy = {testOnlySourceArchiveName: fixtureArchiveName, testOnlySourceArchiveSha256: archiveSha256};
  fs.mkdirSync(bin, {recursive: true}); fs.mkdirSync(path.join(admitted, "receipts"), {recursive: true});
  fs.writeFileSync(archivePath, archiveBytes, {flag: "wx", mode: 0o644});

  /** @type {Record<string, any>} */ const snapshotFiles = {};
  for (const name of toolFiles) {
    const content = Buffer.from("fixture executable " + path.basename(name) + "\n"), file = path.join(sdkRoot, name);
    fs.writeFileSync(file, content, {flag: "wx", mode: 0o755});
    snapshotFiles[name] = {type: "file", bytes: content.length, sha256: createHash("sha256").update(content).digest("hex")};
  }
  for (const [name, target] of Object.entries(aliases)) {
    fs.symlinkSync(target, path.join(sdkRoot, name)); snapshotFiles[name] = {type: "symlink", target};
  }
  const clang = Buffer.from("fixture LLVM 21 clang frontend\n");
  fs.writeFileSync(path.join(bin, "clang-21"), clang, {flag: "wx", mode: 0o755});
  snapshotFiles["bin/clang-21"] = {type: "file", bytes: clang.length, sha256: createHash("sha256").update(clang).digest("hex")};
  const frontendAliases = /** @type {Array<[string, string]>} */ ([ ["bin/clang", "clang-21"], ["bin/clang++", "clang"] ]);
  for (const [name, target] of frontendAliases) {
    fs.symlinkSync(target, path.join(sdkRoot, name)); snapshotFiles[name] = {type: "symlink", target};
  }
  const resourceFiles = {"stddef.h": "/* resource header fixture */\n", "include-fixed/limits.h": "/* nested resource fixture */\n"};
  for (const [name, content] of Object.entries(resourceFiles)) {
    const file = path.join(resourceRoot, name), bytes = Buffer.from(content); fs.mkdirSync(path.dirname(file), {recursive: true});
    fs.writeFileSync(file, bytes, {flag: "wx", mode: 0o644});
    snapshotFiles["lib/clang/21/include/" + name] = {type: "file", bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex")};
  }
  const sdkBytes = Object.values(snapshotFiles).reduce((sum, row) => sum + (row.type === "file" ? row.bytes : Buffer.byteLength(row.target)), 0),
    sdkSnapshot = {files: snapshotFiles, bytes: sdkBytes}, sdk = {path: sdkRoot, files: Object.keys(snapshotFiles).length, bytes: sdkBytes, snapshot: sdkSnapshot};
  const writeJson = (/** @type {string} */ name, /** @type {unknown} */ value) => {
    const file = path.join(admitted, name); fs.writeFileSync(file, JSON.stringify(value) + "\n", {flag: "wx", mode: 0o444}); return fileHash(file);
  };
  const complete = {version: 1, role, runId, phases: [{name: "fixture-phase", pass: true}]}, completeSha256 = writeJson("build-complete.json", complete);
  const audit = {files: sdk.files, bytes: sdk.bytes, archives, snapshot: sdk.snapshot},
    install = {version: 1, role, runId, buildCompleteSha256: completeSha256, audit}, installSha256 = writeJson("install.json", install),
    configured = {version: 1, role, runId, stage: "configured", archiveSha256, planSha256, plan, configure, sourceState, inputFiles};
  writeJson("configured.json", configured);
  const result = {version: 1, role, runId, stage: "audited", archiveSha256, plan, planSha256, configure, sourceState,
    sourceInputs: inputFiles, sdk, archives, buildComplete: {path: path.join(admitted, "build-complete.json"), sha256: completeSha256},
    install: {path: path.join(admitted, "install.json"), sha256: installSha256}, buildOnly: true, compiled: true,
    executed: false, publicationQualified: false};
  writeJson("result.json", result);
  const runAttempt = attempt, recipe = sourceInputs(path.resolve(__dirname, ".."), [], recipeFiles), files = [];
  const pinFile = (/** @type {string} */ relative, /** @type {string} */ file) => files.push({path: relative, type: "file", bytes: fs.statSync(file).size, sha256: fileHash(file),
    mode: fs.statSync(file).mode & 0o777});
  for (const name of toolFiles) {pinFile(name, path.join(sdkRoot, name));}
  for (const [name, target] of Object.entries(aliases)) {files.push({path: name, type: "symlink", target});}
  pinFile(fixtureArchiveName, archivePath);
  for (const name of receiptNames) {pinFile("receipts/" + name, path.join(admitted, name));}
  const handoff = {version: 1, format: "mom-linux-llvm-tools-handoff-v1", runAttempt, runId, buildOnly: true, compiled: true,
    publicationQualified: false, archiveSha256, sourceArchive: {path: fixtureArchiveName, bytes: archiveBytes.length, sha256: archiveSha256},
    receipts: Object.fromEntries(receiptNames.map(name => [name, {path: "receipts/" + name, sha256: fileHash(path.join(admitted, name))}])),
    files: files.sort((left, right) => left.path.localeCompare(right.path)), recipe};
  fs.mkdirSync(bundle, {recursive: true});
  for (const relative of toolFiles) {
    const destination = path.join(bundle, relative); fs.mkdirSync(path.dirname(destination), {recursive: true});
    fs.copyFileSync(path.join(sdkRoot, relative), destination); fs.chmodSync(destination, 0o755);
  }
  for (const [relative, target] of Object.entries(aliases)) {fs.symlinkSync(target, path.join(bundle, relative));}
  fs.copyFileSync(archivePath, path.join(bundle, fixtureArchiveName));
  fs.mkdirSync(path.join(bundle, "receipts"));
  for (const name of receiptNames) {fs.copyFileSync(path.join(admitted, name), path.join(bundle, "receipts", name)); fs.chmodSync(path.join(bundle, "receipts", name), 0o444);}
  const handoffPath = path.join(bundle, "handoff.json"); fs.writeFileSync(handoffPath, JSON.stringify(handoff, null, 2) + "\n", {flag: "wx", mode: 0o444});
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, admitted, sdkRoot, bundle, resourceRoot, attempt, runId, archivePath, archiveSha256, policy, handoffPath, handoff,
    sdkSnapshot, result, inputFiles: [archivePath, ...toolFiles.map(name => path.join(sdkRoot, name)), ...receiptNames.map(name => path.join(admitted, name))]};
}

/** @param {string} handoffPath @param {Record<string, any>} handoff */
function replaceHandoff(handoffPath, handoff) {
  fs.chmodSync(handoffPath, 0o644); fs.writeFileSync(handoffPath, JSON.stringify(handoff, null, 2) + "\n"); fs.chmodSync(handoffPath, 0o444);
  return fileHash(handoffPath);
}

test.describe("Linux LLVM tool handoff", {skip: process.platform !== "linux"}, () => {
  test("reads a tiny externally pinned bundle with all nine tools, three aliases, source, and original receipts", t => {
    const f = fixture(t), hashes = new Map(f.inputFiles.map(file => [file, fileHash(file)])),
      result = linuxLlvmToolHandoffInputs(f.bundle, {runAttempt: f.attempt, handoffSha256: fileHash(f.handoffPath)}, f.policy);
    assert.equal(result.toolRoot, path.join(f.bundle, "bin"));
    assert.equal(result.lldLink, path.join(f.bundle, "bin/lld-link"));
    assert.equal(result.llvmNm, path.join(f.bundle, "bin/llvm-nm"));
    assert.equal(result.sourceArchive, path.join(f.bundle, fixtureArchiveName));
    assert.equal(Object.keys(result.files).length, 17);
    for (const name of ["llvm-nm", "llvm-readobj", "llvm-objcopy"]) {
      const relative = "bin/" + name;
      assert.equal(result.files[relative]?.sha256, fileHash(path.join(result.toolRoot, name)),
        "Windows GPU-factory inspection and isolation tools retain original producer pins");
      fs.accessSync(path.join(result.toolRoot, name), fs.constants.X_OK);
    }
    assert.deepEqual(fs.readlinkSync(path.join(f.bundle, "bin/lld-link")), "lld");
    assert.deepEqual(fs.readlinkSync(path.join(f.bundle, "bin/llvm-lib")), "llvm-ar");
    assert.deepEqual(fs.readlinkSync(path.join(f.bundle, "bin/llvm-ranlib")), "llvm-ar");
    for (const [file, hash] of hashes) {assert.equal(fileHash(file), hash, "reading the handoff leaves original admissions untouched");}
  });

  test("supplies every host utility required by the maintained Windows LLVM planner", t => {
    const f = fixture(t), admitted = linuxLlvmToolHandoffInputs(f.bundle, {runAttempt: f.attempt, handoffSha256: fileHash(f.handoffPath)}, f.policy),
      include = path.join(f.root, "include"), header = path.join(f.root, "windows.h"), overlay = path.join(f.root, "overlay.json");
    fs.writeFileSync(header, "/* planner fixture */\n", {flag: "wx"});
    fs.writeFileSync(overlay, JSON.stringify({version: 0, roots: [{type: "file", name: path.join(include, "windows.h"),
      "external-contents": header}]}) + "\n", {flag: "wx"});
    const descriptor = {version: 1, tools: {clangCl: path.join(f.root, "icx-cl"), lldLink: admitted.lldLink,
        llvmNm: admitted.llvmNm, clangResourceInclude: path.join(f.root, "resource")},
      windowsSdk: {overlay, sourceRoot: f.root, includeRoots: [include], libraryRoots: [path.join(f.root, "lib")],
        msvcLibraries: [path.join(f.root, "lib/libcmt.lib")], ucrtLibraries: [path.join(f.root, "lib/libucrt.lib")],
        importLibraries: [path.join(f.root, "lib/kernel32.lib")]}},
      plan = windowsLlvmSdkPlan(descriptor, path.join(f.root, archiveName), path.join(f.root, "configured"),
        {cmake: path.join(f.root, "cmake"), ninja: path.join(f.root, "ninja")}, 2);
    const hostTools = /** @type {Record<string, string>} */ (plan.nativeTools);
    for (const file of [...Object.values(hostTools), admitted.lldLink, admitted.llvmNm]) {
      assert.ok(admitted.files["bin/" + path.basename(file)], "thin handoff omitted a maintained host tool: " + file);
      fs.accessSync(file, fs.constants.X_OK);
    }
  });

  test("rejects wrong external pins and run identity before accepting bundle contents", t => {
    const f = fixture(t), correctSha = fileHash(f.handoffPath);
    assert.throws(() => linuxLlvmToolHandoffInputs(f.bundle, {runAttempt: f.attempt, handoffSha256: "0".repeat(64)}, f.policy), /external hash pin|another run attempt/);
    assert.throws(() => linuxLlvmToolHandoffInputs(f.bundle, {runAttempt: f.attempt + "-other", handoffSha256: correctSha}, f.policy), /exact same-run attempt/);
    assert.throws(() => linuxLlvmToolHandoffInputs(f.bundle, {runAttempt: "987654-2", handoffSha256: correctSha}, f.policy), /another run attempt/);
    assert.throws(() => linuxLlvmToolHandoffInputs(f.bundle, {runAttempt: f.attempt, handoffSha256: correctSha}, {...f.policy, testOnlySourceArchiveSha256: "0".repeat(64)}), /external hash pin|source archive/);
    const oversized = structuredClone(f.handoff);
    oversized.sourceArchive.bytes = 2 * 1024 ** 3 + 1;
    const oversizedSha = replaceHandoff(f.handoffPath, oversized);
    assert.throws(() => linuxLlvmToolHandoffInputs(f.bundle, {runAttempt: f.attempt, handoffSha256: oversizedSha}, f.policy), /invalid or differs from its external hash pin/);
  });

  test("rejects missing or extra payloads, escaping aliases, hardlinks, and file-byte drift", t => {
    const missing = fixture(t); fs.unlinkSync(path.join(missing.bundle, "bin/llvm-ar"));
    assert.throws(() => linuxLlvmToolHandoffInputs(missing.bundle, {runAttempt: missing.attempt,
      handoffSha256: fileHash(missing.handoffPath)}, missing.policy), /missing or extra payload/);

    const extra = fixture(t), extraFile = path.join(extra.bundle, "bin/unselected-tool"); fs.writeFileSync(extraFile, "extra\n", {flag: "wx"});
    assert.throws(() => linuxLlvmToolHandoffInputs(extra.bundle, {runAttempt: extra.attempt,
      handoffSha256: fileHash(extra.handoffPath)}, extra.policy), /missing or extra payload/);

    const alias = fixture(t), aliasPath = path.join(alias.bundle, "bin/lld-link"); fs.unlinkSync(aliasPath); fs.symlinkSync("../../outside", aliasPath);
    assert.throws(() => linuxLlvmToolHandoffInputs(alias.bundle, {runAttempt: alias.attempt,
      handoffSha256: fileHash(alias.handoffPath)}, alias.policy), /alias differs/);

    const hardlink = fixture(t), linked = path.join(hardlink.root, "external-link"); fs.linkSync(path.join(hardlink.bundle, "bin/lld"), linked);
    assert.throws(() => linuxLlvmToolHandoffInputs(hardlink.bundle, {runAttempt: hardlink.attempt,
      handoffSha256: fileHash(hardlink.handoffPath)}, hardlink.policy), /singly-linked/);

    const drift = fixture(t); fs.appendFileSync(path.join(drift.bundle, "bin/llvm-nm"), "changed\n");
    assert.throws(() => linuxLlvmToolHandoffInputs(drift.bundle, {runAttempt: drift.attempt,
      handoffSha256: fileHash(drift.handoffPath)}, drift.policy), /differs from its external manifest/);
  });

  test("rejects re-pinned tool substitution against unchanged admitted SDK inventory", t => {
    const f = fixture(t), tool = path.join(f.bundle, "bin/lld"); fs.appendFileSync(tool, "substituted\n");
    const handoff = structuredClone(f.handoff), row = /** @type {{path:string,type:string,bytes:number,sha256:string,mode:number}|undefined} */
      (handoff.files.find(item => item.path === "bin/lld"));
    if (!row) {throw new Error("Fixture omitted the selected lld manifest row");}
    row.bytes = fs.statSync(tool).size; row.sha256 = fileHash(tool); row.mode = fs.statSync(tool).mode & 0o777;
    const handoffSha256 = replaceHandoff(f.handoffPath, handoff);
    assert.throws(() => linuxLlvmToolHandoffInputs(f.bundle, {runAttempt: f.attempt, handoffSha256}, f.policy), /snapshot|receipt|admitted|original/i,
      "a freshly rebound handoff manifest cannot authorize tool bytes absent from the copied SDK snapshot");
  });

  test("rejects receipt cross-reference drift even after payload and handoff pins are recomputed", t => {
    const f = fixture(t), configuredPath = path.join(f.bundle, "receipts/configured.json"), configured = JSON.parse(fs.readFileSync(configuredPath, "utf8"));
    configured.planSha256 = "f".repeat(64); fs.chmodSync(configuredPath, 0o644); fs.writeFileSync(configuredPath, JSON.stringify(configured) + "\n");
    fs.chmodSync(configuredPath, 0o444);
    const handoff = structuredClone(f.handoff), pin = /** @type {{path:string,type:string,bytes:number,sha256:string,mode:number}|undefined} */
        (handoff.files.find(item => item.path === "receipts/configured.json")),
      receiptHash = fileHash(configuredPath);
    if (!pin) {throw new Error("Fixture omitted the configured receipt manifest row");}
    pin.bytes = fs.statSync(configuredPath).size; pin.sha256 = receiptHash; pin.mode = fs.statSync(configuredPath).mode & 0o777;
    const configuredReceipt = handoff.receipts["configured.json"];
    if (!configuredReceipt) {throw new Error("Fixture omitted the configured receipt cross-reference");}
    configuredReceipt.sha256 = receiptHash;
    const handoffSha256 = replaceHandoff(f.handoffPath, handoff);
    assert.throws(() => linuxLlvmToolHandoffInputs(f.bundle, {runAttempt: f.attempt, handoffSha256}, f.policy), /cross-references/);
  });

  test("keeps production export bound to full public admission and the locked archive", t => {
    const f = fixture(t), output = path.join(f.root, "failed-export"), calls = [];
    const fakeReader = (/** @type {string} */ root) => {
      calls.push(root);
      return {...JSON.parse(fs.readFileSync(path.join(root, "result.json"), "utf8")),
        sdk: {...f.sdkSnapshot, path: path.join(root, "sdk")}, plan: {output: root, prefix: path.join(root, "sdk")}, archiveSha256: "0".repeat(64)};
    };
    assert.throws(() => exportLinuxLlvmTools(f.admitted, output, {runAttempt: f.attempt}, fakeReader), /full public SDK admission/);
    assert.equal(calls.length, 1); assert.equal(fs.existsSync(output), false, "a mismatched locked source archive is rejected before output creation");
    const noAdmission = path.join(f.root, "no-admission"); fs.mkdirSync(noAdmission);
    assert.throws(() => exportLinuxLlvmTools(noAdmission, path.join(f.root, "default-reader-output"), {runAttempt: f.attempt}), /LLVM|receipt|admission/i);
    assert.equal(fs.existsSync(path.join(f.root, "default-reader-output")), false, "the public verifier remains the default export reader");
  });

  test("exports only the admitted Clang 21 drivers and bounded resource headers for ACPP", t => {
    const f = fixture(t), output = path.join(f.root, "acpp-frontend"), reader = () => f.result,
      exported = exportLinuxLlvmAcppFrontend(f.admitted, output, {runAttempt: f.attempt}, reader, f.policy),
      frontend = linuxLlvmAcppFrontendInputs(output, {runAttempt: f.attempt, handoffSha256: exported.handoffSha256}, f.policy),
      handoff = JSON.parse(fs.readFileSync(exported.handoffPath, "utf8"));
    assert.equal(frontend.clang, path.join(output, "bin/clang"));
    assert.equal(frontend.clangxx, path.join(output, "bin/clang++"));
    assert.equal(frontend.resourceInclude, path.join(output, "lib/clang/21/include"));
    assert.equal(fs.realpathSync(frontend.clangxx), fs.realpathSync(frontend.clang));
    assert.equal(fs.realpathSync(frontend.clang), path.join(output, "bin/clang-21"));
    assert.equal(frontend.files["bin/clang-21"]?.sha256, f.sdkSnapshot.files["bin/clang-21"].sha256);
    assert.equal(frontend.files["lib/clang/21/include/stddef.h"]?.sha256,
      f.sdkSnapshot.files["lib/clang/21/include/stddef.h"].sha256);
    assert.equal(frontend.files["lib/clang/21/include/include-fixed/limits.h"]?.sha256,
      f.sdkSnapshot.files["lib/clang/21/include/include-fixed/limits.h"].sha256);
    assert.equal(handoff.format, "mom-linux-llvm-acpp-frontend-handoff-v1");
    assert.deepEqual(handoff.frontend, {clang: "bin/clang", clangxx: "bin/clang++", resourceInclude: "lib/clang/21/include", version: "21.1.8"});
    assert.equal(Object.keys(frontend.files).some(name => name.endsWith(".a") || name.startsWith("include/llvm/")), false,
      "the ACPP role carries neither static LLVM archives nor LLVM development headers");

    const toolOutput = path.join(f.root, "default-tools"), toolExport = exportLinuxLlvmTools(f.admitted, toolOutput,
        {runAttempt: f.attempt}, reader, f.policy), defaultHandoff = JSON.parse(fs.readFileSync(toolExport.handoffPath, "utf8"));
    assert.equal(defaultHandoff.format, "mom-linux-llvm-tools-handoff-v1");
    assert.equal(Object.hasOwn(defaultHandoff, "frontend"), false);
    assert.equal((/** @type {{path:string}[]} */ (defaultHandoff.files)).some(row => row.path.startsWith("lib/clang/21/include/")), false,
      "adding the frontend role leaves the default nine-tool payload selection unchanged");
  });

  test("rejects missing, changed, unsafe, or over-bound frontend resources before export mutation", t => {
    const missing = fixture(t), output = path.join(missing.root, "missing-output");
    fs.unlinkSync(path.join(missing.resourceRoot, "stddef.h"));
    assert.throws(() => exportLinuxLlvmAcppFrontend(missing.admitted, output, {runAttempt: missing.attempt}, () => missing.result, missing.policy), /resource include tree/);
    assert.equal(fs.existsSync(output), false, "resource completeness is established before creating the handoff root");

    const unsafe = fixture(t), symlink = path.join(unsafe.resourceRoot, "unsafe.h");
    fs.symlinkSync("../../../../../../outside.h", symlink);
    unsafe.result.sdk.snapshot.files["lib/clang/21/include/unsafe.h"] = {type: "symlink", target: "../../../../../../outside.h"};
    assert.throws(() => exportLinuxLlvmAcppFrontend(unsafe.admitted, path.join(unsafe.root, "unsafe-output"),
      {runAttempt: unsafe.attempt}, () => unsafe.result, unsafe.policy), /unsupported or invalid member/);

    const oversized = fixture(t), largeSnapshot = structuredClone(oversized.result), prefix = "lib/clang/21/include/";
    for (let i = 0; i < 4095; i++) {
      largeSnapshot.sdk.snapshot.files[prefix + "synthetic-" + String(i).padStart(4, "0")] =
        {type: "file", bytes: 1, sha256: "0".repeat(64)};
    }
    const oversizedOutput = path.join(oversized.root, "oversized-output");
    assert.throws(() => exportLinuxLlvmAcppFrontend(oversized.admitted, oversizedOutput, {runAttempt: oversized.attempt},
      () => largeSnapshot, oversized.policy), /exceed the member bound/);
    assert.equal(fs.existsSync(oversizedOutput), false, "the fixed 4096-member limit fails before output mutation");
  });

  test("rejects stale frontend source recipes and cross-role bundles", t => {
    const f = fixture(t), output = path.join(f.root, "acpp-frontend"), exported = exportLinuxLlvmAcppFrontend(
        f.admitted, output, {runAttempt: f.attempt}, () => f.result, f.policy),
      defaultOutput = path.join(f.root, "default-tools"), defaultExport = exportLinuxLlvmTools(
        f.admitted, defaultOutput, {runAttempt: f.attempt}, () => f.result, f.policy);
    assert.throws(() => linuxLlvmToolHandoffInputs(output, {runAttempt: f.attempt, handoffSha256: exported.handoffSha256}, f.policy), /invalid or differs/);
    assert.throws(() => linuxLlvmAcppFrontendInputs(defaultOutput,
      {runAttempt: f.attempt, handoffSha256: defaultExport.handoffSha256}, f.policy), /invalid or differs/);

    const handoff = JSON.parse(fs.readFileSync(exported.handoffPath, "utf8"));
    handoff.recipe.graph["scripts/native-linux-llvm-tools.js"] = "0".repeat(64);
    replaceHandoff(exported.handoffPath, handoff);
    assert.throws(() => linuxLlvmAcppFrontendInputs(output,
      {runAttempt: f.attempt, handoffSha256: fileHash(exported.handoffPath)}, f.policy), /source inputs are stale/);
  });

  test("rejects frontend resource substitution against the producer snapshot", t => {
    const f = fixture(t), output = path.join(f.root, "acpp-frontend"), exported = exportLinuxLlvmAcppFrontend(
        f.admitted, output, {runAttempt: f.attempt}, () => f.result, f.policy),
      resource = path.join(output, "lib/clang/21/include/stddef.h");
    fs.appendFileSync(resource, "changed\n");
    const handoff = /** @type {{files:{path:string,bytes:number,sha256:string,mode:number}[]}} */ (JSON.parse(fs.readFileSync(exported.handoffPath, "utf8"))),
      row = handoff.files.find(item => item.path === "lib/clang/21/include/stddef.h");
    if (!row) {throw new Error("Fixture omitted its selected Clang resource header");}
    row.bytes = fs.statSync(resource).size; row.sha256 = fileHash(resource); row.mode = fs.statSync(resource).mode & 0o777;
    const handoffSha256 = replaceHandoff(exported.handoffPath, handoff);
    assert.throws(() => linuxLlvmAcppFrontendInputs(output, {runAttempt: f.attempt, handoffSha256}, f.policy), /original admitted SDK snapshot/);
  });
});
