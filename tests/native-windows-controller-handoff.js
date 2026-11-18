"use strict";
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), {createHash} = require("node:crypto");
const test = require("node:test"), assert = require("node:assert/strict");
const {windowsControllerHandoffInputs, windowsLocalControllerHandoffInputs, validateIntelRuntimeMetadata} = require("../scripts/native-windows-controller-handoff");
const {fileHash, sourceInputs} = require("../scripts/build-source-inputs");
const {ioPins} = require("../scripts/build-native-io"), {opensslPin, opensslBuildRecipe} = require("../scripts/build-native-openssl");
const {hermesPins} = require("../scripts/build-native-hermes");
const {ioSourceRecipe, sourceRecipe, ioArchiveUrls, buildArchiveUrls} = require("../scripts/fetch-native-sdk-sources");
const intelRuntimeCatalog = require("../runtime/windows-intel-runtime.json");

const repo = path.resolve(__dirname, ".."), suffix = "12345-1", sha = (/** @type {string} */ text) => createHash("sha256").update(text).digest("hex");
/** @param {string} root @param {string} relative @param {string} data @returns {string} */
function put(root, relative, data) {
  const file = path.join(root, ...relative.split("/")); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data); return file;
}
/** @param {string} root @param {string} relative @param {any} value @returns {string} */
function save(root, relative, value) {return put(root, relative, JSON.stringify(value, null, 2) + "\n");}
/** @param {string} root @returns {Array<{path:string,bytes:number,sha256:string}>} */
function hashes(root) {
  const rows = /** @type {Array<{path:string,bytes:number,sha256:string}>} */ ([]);
  /** @param {string} directory @param {string} [prefix] */
  const visit = (directory, prefix = "") => {
    for (const entry of fs.readdirSync(directory, {withFileTypes: true})) {
      const relative = prefix ? prefix + "/" + entry.name : entry.name, file = path.join(directory, entry.name);
      if (relative === "handoff.json") {continue;}
      if (entry.isDirectory()) {visit(file, relative);} else {rows.push({path: relative, bytes: fs.statSync(file).size, sha256: fileHash(file)});}
    }
  };
  visit(root); return rows.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
}
/** @param {string} root @param {any} handoff @returns {string} */
function seal(root, handoff) {handoff.files = hashes(root); save(root, "handoff.json", handoff); return fileHash(path.join(root, "handoff.json"));}
/** @param {{after:(fn:()=>void)=>void}} t @param {string} name @returns {string} */
function createRoot(t, name) {
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-handoff-")), root = path.join(parent, name);
  fs.mkdirSync(root); t.after(() => fs.rmSync(parent, {recursive: true, force: true})); return root;
}
/** @param {string} target @param {any} recipe @param {Record<string,any>} archives @param {string} fetcherSha256 @returns {any} */
function fetchRecord(target, recipe, archives, fetcherSha256) {
  return {version: 1, target, status: "PASS", networked: true, buildOnly: true, compiled: false, publicationQualified: false,
    fetcherSha256, recipe: {sha256: sha(JSON.stringify(recipe)), files: recipe.graph}, archives};
}
/** @param {{after:(fn:()=>void)=>void}} t @returns {any} */
function ioFixture(t) {
  const root = createRoot(t, "io"), staticArchive = "!<arch>\nfixture\n";
  const files = {
    "io-sdk/include/uv.h": "uv header", "io-sdk/include/zlib.h": "zlib header", "io-sdk/include/zconf.h": "generated zlib header",
    "io-sdk/lib/uv_a.lib": staticArchive, "io-sdk/lib/zlibstatic.lib": staticArchive,
    "openssl-sdk/include/openssl/opensslv.h": '#define OPENSSL_VERSION_STR "3.5.9"\n',
    "openssl-sdk/include/openssl/ssl.h": "openssl public header", "openssl-sdk/lib/libssl.lib": staticArchive,
    "openssl-sdk/lib/libcrypto.lib": staticArchive, "openssl-sdk/source/LICENSE.txt": "OpenSSL license body\n",
  };
  for (const [name, body] of Object.entries(files)) {put(root, name, body);}
  const ioFetchRecipe = ioSourceRecipe(), sslRecipe = opensslBuildRecipe(), fetcherSha256 = fileHash(path.join(repo, "scripts/fetch-native-sdk-sources.js")),
    libuvPath = "C:\\runner\\io-source\\libuv-v1.51.0.tar.gz", zlibPath = "C:\\runner\\io-source\\zlib-1.3.1.tar.gz",
    sslArchivePath = "C:\\runner\\openssl-source\\openssl-3.5.9.tar.gz",
    ioFetch = fetchRecord("io", ioFetchRecipe, {
      libuv: {path: libuvPath, url: ioArchiveUrls.libuv, sha256: ioPins.libuv.sha256, bytes: 1000},
      zlib: {path: zlibPath, url: ioArchiveUrls.zlib, sha256: ioPins.zlib.sha256, bytes: 2000},
    }, fetcherSha256), sslFetch = fetchRecord("windows-openssl", sslRecipe, {
      openssl: {path: sslArchivePath, url: buildArchiveUrls.openssl, sha256: opensslPin.sha256, bytes: 3000},
    }, fetcherSha256);
  save(root, "provenance/io-fetch.json", ioFetch); save(root, "provenance/openssl-fetch.json", sslFetch);
  const ioArtifacts = Object.fromEntries(Object.keys(files).filter(name => name.startsWith("io-sdk/")).map(name => {
    const relative = name.slice("io-sdk/".length); return [relative, fileHash(path.join(root, ...name.split("/")))];
  }));
  const toolPaths = {cmake: "C:\\Tools\\cmake.exe", cc: "C:\\Tools\\icx-cl.exe", ninja: "C:\\Tools\\ninja.exe"},
    ioRecipe = sourceInputs(repo, [], ["scripts/build-native-io.js", "scripts/build-environment.js", "scripts/build-source-inputs.js", "scripts/native-audit.js"]);
  save(root, "io-sdk/result.json", {version: 1, target: "win32", pins: ioPins, buildOnly: true, publicationQualified: false,
    inputs: {[libuvPath]: ioPins.libuv.sha256, [zlibPath]: ioPins.zlib.sha256}, fingerprints: Object.fromEntries(Object.values(toolPaths).map(file => [file, sha(file)])),
    recipe: ioRecipe, artifacts: ioArtifacts});
  const opensslTools = {ar: "C:\\Tools\\llvm-ar.exe", cc: "C:\\Tools\\icx-cl.exe", link: "C:\\Tools\\lld-link.exe", make: "C:\\Tools\\nmake.exe",
    perl: "C:\\Tools\\perl.exe", rc: "C:\\Tools\\llvm-rc.exe", tar: "C:\\Windows\\System32\\tar.exe"};
  save(root, "openssl-sdk/producer-tools.json", {version: 1, jobs: 2, tools: opensslTools});
  const sslArtifacts = {
    "include/openssl/opensslv.h": fileHash(path.join(root, "openssl-sdk/include/openssl/opensslv.h")),
    "include/openssl/ssl.h": fileHash(path.join(root, "openssl-sdk/include/openssl/ssl.h")),
    "lib/libssl.lib": fileHash(path.join(root, "openssl-sdk/lib/libssl.lib")),
    "lib/libcrypto.lib": fileHash(path.join(root, "openssl-sdk/lib/libcrypto.lib")), "bin/openssl.exe": sha("omitted OpenSSL build CLI"),
  };
  const sslTools = Object.fromEntries(Object.entries(opensslTools).map(([name, file]) => [name, {path: file, sha256: sha(file)}]));
  const licenseSha256 = fileHash(path.join(root, "openssl-sdk/source/LICENSE.txt"));
  save(root, "openssl-sdk/result.json", {version: 1, platform: "win32", target: "VC-WIN64A", opensslVersion: opensslPin.version,
    sourceDateEpoch: opensslPin.sourceDateEpoch,
    archive: {path: sslArchivePath, sha256: opensslPin.sha256, bytes: 3000, members: 200}, source: {files: 100, bytes: 10000, sha256: sha("source"),
      notice: {path: "source/LICENSE.txt", sha256: licenseSha256}}, config: {sha256: fileHash(path.join(root, "openssl-sdk/producer-tools.json"))},
    tools: sslTools, recipe: sourceInputs(repo, [], ["scripts/build-native-openssl.js", "scripts/build-source-inputs.js", "scripts/build-environment.js", "scripts/native-audit.js"]),
    artifacts: sslArtifacts, buildOnly: true, executed: false, publicationQualified: false});
  const handoff = {version: 1, format: "mom-windows-controller-support-handoff-v1", target: "win32", architecture: "x64", kind: "io",
    buildOnly: true, executed: false, publicationQualified: false, scope: "Selected build-only I/O/OpenSSL source closure.",
    producers: [{script: "scripts/build-native-io.js", receipt: "io-sdk/result.json", receiptSha256: fileHash(path.join(root, "io-sdk/result.json"))},
      {script: "scripts/build-native-openssl.js", receipt: "openssl-sdk/result.json", receiptSha256: fileHash(path.join(root, "openssl-sdk/result.json"))}],
    sourceReceipts: [{file: "provenance/io-fetch.json", sha256: fileHash(path.join(root, "provenance/io-fetch.json")), archives: {libuv: ioPins.libuv.sha256, zlib: ioPins.zlib.sha256}},
      {file: "provenance/openssl-fetch.json", sha256: fileHash(path.join(root, "provenance/openssl-fetch.json")), archives: {openssl: opensslPin.sha256}}],
    omitted: ["OpenSSL source/build trees", "openssl.exe", "compiler/build tools", "DLLs"]};
  return {root, handoff, expected: {kind: "io", handoffSha256: seal(root, handoff)}, files};
}
/** @param {{after:(fn:()=>void)=>void}} t @param {boolean} [shareCCompiler] @param {boolean} [includeRootLicense] @returns {any} */
function hermesFixture(t, shareCCompiler = false, includeRootLicense = false) {
  const root = createRoot(t, "hermes"), sourceFiles = {LICENSE: "Hermes root license\n", "include/hermes.h": "include", "API/jsi.h": "API",
    "public/hermes.h": "public", "lib/internal.h": "not handed off"};
  for (const [name, body] of Object.entries(sourceFiles)) {
    if (!name.startsWith("lib/") && (name !== "LICENSE" || includeRootLicense)) {put(root, "hermes/source/" + name, body);}
  }
  const archiveNames = {runtime: "hermes/build/lib/hermesvmlean_a.lib", jsi: "hermes/build/jsi/jsi.lib",
      context: "hermes/build/external/boost/boost_1_86_0/libs/context/boost_context.lib"}, archiveData = "!<arch>\nfixture\n";
  for (const relative of Object.values(archiveNames)) {put(root, relative, archiveData);}
  put(root, "hermes/build/lib/config/config.h", "generated config\n");
  const resultPath = "hermes/result.json", beforePath = "hermes/before.json", fetchPath = "provenance/hermes-fetch.json", rootsPath = "provenance/hermes-roots.json",
    win = path.win32, base = "C:\\agent\\hermes-sdk", source = win.join(base, "source"), build = win.join(base, "build"),
    toolPaths = {git: "C:\\Tools\\git.exe", cmake: "C:\\Tools\\cmake.exe", cc: "C:\\Tools\\icx-cl.exe",
      cxx: shareCCompiler ? "C:\\Tools\\icx-cl.exe" : "C:\\Tools\\icx.exe", ninja: "C:\\Tools\\ninja.exe"},
    patches = ["runtime/hermes-host-rejections.patch", "runtime/hermes-clang-cl.patch"], fingerprints = Object.fromEntries([
      ...Object.values(toolPaths).map(file => [file, sha(file)]), ...patches.map(name => ["C:\\repo\\" + name.replaceAll("/", "\\"), fileHash(path.join(repo, name))]),
    ]), sources = Object.fromEntries(Object.entries(sourceFiles).map(([name, body]) => [name, name.startsWith("lib/") || name === "LICENSE" && !includeRootLicense ? sha(body) :
      fileHash(path.join(root, "hermes/source", ...name.split("/")))])),
    recipe = sourceInputs(repo, [], ["scripts/build-native-hermes.js", "scripts/build-source-inputs.js", "scripts/build-environment.js", "scripts/native-audit.js",
      "runtime/hermes-host-rejections.patch", "runtime/hermes-clang-cl.patch"]),
    artifacts = {compiler: {path: win.join(build, "bin/shermes.exe"), sha256: sha("compiler"), bytes: 10},
      runtime: {path: win.join(build, "lib/hermesvmlean_a.lib"), sha256: fileHash(path.join(root, archiveNames.runtime)), bytes: fs.statSync(path.join(root, archiveNames.runtime)).size},
      jsi: {path: win.join(build, "jsi/jsi.lib"), sha256: fileHash(path.join(root, archiveNames.jsi)), bytes: fs.statSync(path.join(root, archiveNames.jsi)).size},
      context: {path: win.join(build, "external/boost/boost_1_86_0/libs/context/boost_context.lib"), sha256: fileHash(path.join(root, archiveNames.context)), bytes: fs.statSync(path.join(root, archiveNames.context)).size},
      public: {path: win.join(build, "public/hermes/Public/hermesPublic.lib"), sha256: sha("omitted public library"), bytes: 11}},
    configure = ["C:\\Tools\\cmake.exe", "-DCMAKE_C_COMPILER:FILEPATH=" + toolPaths.cc.replaceAll("\\", "/"),
      "-DCMAKE_CXX_COMPILER:FILEPATH=" + toolPaths.cxx.replaceAll("\\", "/"), "-DCMAKE_MAKE_PROGRAM:FILEPATH=C:/Tools/ninja.exe"],
    compile = ["C:\\Tools\\cmake.exe", "--build", build, "--parallel", "2"];
  const before = {pins: hermesPins, recipe, fingerprints, sources, configure, compile};
  save(root, beforePath, before);
  save(root, resultPath, {version: 1, target: "win32", pins: hermesPins, source, build, artifacts, fingerprints, sources, recipe,
    buildOnly: true, publicationQualified: false});
  const gitRoot = "/tmp/mom-native-hermes-prerequisites-" + suffix + "/roots/hermes", fetchRecipe = sourceRecipe("hermes"),
    fetch = fetchRecord("hermes", fetchRecipe, {}, fileHash(path.join(repo, "scripts/fetch-native-sdk-sources.js")));
  Object.assign(fetch, {git: {path: "/usr/bin/git", sha256: sha("bare git payload")}, components: {hermes: {origin: "https://github.com/facebook/hermes.git",
    revision: hermesPins.revision, root: gitRoot, gitDirectory: "bare", shallow: true, partialClone: false}}});
  save(root, fetchPath, fetch); save(root, rootsPath, {hermes: gitRoot});
  const handoff = {version: 1, format: "mom-windows-controller-support-handoff-v1", target: "win32", architecture: "x64", kind: "hermes",
    buildOnly: true, executed: false, publicationQualified: false, scope: "Selected build-only Hermes source/config/static closure; AOT compiler is separate.",
    producers: [{script: "scripts/build-native-hermes.js", receipt: resultPath, receiptSha256: fileHash(path.join(root, resultPath)), plan: beforePath,
      planSha256: fileHash(path.join(root, beforePath))}], sourceReceipts: [{file: fetchPath, sha256: fileHash(path.join(root, fetchPath)), rootsFile: rootsPath,
      rootsSha256: fileHash(path.join(root, rootsPath)), revision: hermesPins.revision}],
    omitted: ["Hermes Git/build trees", "shermes.exe", "hermesPublic.lib", "compiler/build tools", "DLLs"]};
  return {root, handoff, expected: {kind: "hermes", handoffSha256: seal(root, handoff)}};
}
/** @param {string} root @param {any} handoff @returns {string} */
function reseal(root, handoff) {
  handoff.producers[0].receiptSha256 = fileHash(path.join(root, ...handoff.producers[0].receipt.split("/")));
  if (handoff.producers[1]) {handoff.producers[1].receiptSha256 = fileHash(path.join(root, ...handoff.producers[1].receipt.split("/")));}
  return seal(root, handoff);
}

/** @param {any} fixture @param {(fetch:any)=>void} update @param {boolean} [rebindHandoff] */
function updateHermesFetch(fixture, update, rebindHandoff = true) {
  const relative = "provenance/hermes-fetch.json", file = path.join(fixture.root, relative),
    fetch = JSON.parse(fs.readFileSync(file, "utf8"));
  update(fetch); save(fixture.root, relative, fetch);
  if (rebindHandoff) {
    fixture.handoff.sourceReceipts[0].sha256 = fileHash(file);
    fixture.expected.handoffSha256 = seal(fixture.root, fixture.handoff);
  }
}

/** @param {any} fixture @param {string} relative @param {(fetch:any)=>void} update */
function updateIoFetch(fixture, relative, update) {
  const file = path.join(fixture.root, ...relative.split("/")), fetch = JSON.parse(fs.readFileSync(file, "utf8")),
    sourceReceipt = fixture.handoff.sourceReceipts.find((/** @type {{file:string,sha256:string}} */ row) => row.file === relative);
  if (!sourceReceipt) {throw new Error("Fixture is missing its pinned Windows source receipt: " + relative);}
  update(fetch); save(fixture.root, relative, fetch);
  sourceReceipt.sha256 = fileHash(file); fixture.expected.handoffSha256 = seal(fixture.root, fixture.handoff);
}

/** @param {any} fixture @param {(result:any,before:any)=>void} update */
function updateHermesProducerRecipe(fixture, update) {
  const resultPath = "hermes/result.json", beforePath = "hermes/before.json",
    result = JSON.parse(fs.readFileSync(path.join(fixture.root, resultPath), "utf8")),
    before = JSON.parse(fs.readFileSync(path.join(fixture.root, beforePath), "utf8"));
  update(result, before);
  save(fixture.root, resultPath, result); save(fixture.root, beforePath, before);
  fixture.handoff.producers[0].planSha256 = fileHash(path.join(fixture.root, beforePath));
  fixture.expected.handoffSha256 = reseal(fixture.root, fixture.handoff);
}

/** @param {any} fixture @param {(fingerprints:Record<string,string>)=>Record<string,string>} update */
function updateHermesFingerprints(fixture, update) {
  const resultPath = path.join(fixture.root, "hermes/result.json"), beforePath = path.join(fixture.root, "hermes/before.json"),
    result = JSON.parse(fs.readFileSync(resultPath, "utf8")), before = JSON.parse(fs.readFileSync(beforePath, "utf8")),
    fingerprints = update({...result.fingerprints});
  result.fingerprints = fingerprints; before.fingerprints = fingerprints;
  save(fixture.root, "hermes/result.json", result); save(fixture.root, "hermes/before.json", before);
  fixture.handoff.producers[0].planSha256 = fileHash(beforePath);
  fixture.expected.handoffSha256 = reseal(fixture.root, fixture.handoff);
}

/** @param {string} cc @param {Record<string,string>} fingerprints @returns {any} */
function intelRuntimeMetadata(cc, fingerprints) {
  return {compiler: {path: cc, sha256: fingerprints[cc]},
    libraries: intelRuntimeCatalog.libraries.map((/** @type {any} */ row) => ({name: row.name, path: row.path, bytes: row.bytes, sha256: row.sha256})),
    licenses: intelRuntimeCatalog.licenses.map((/** @type {any} */ row) => ({path: row.path, bytes: row.bytes, sha256: row.sha256}))};
}

/** @param {any} metadata @returns {Array<{path:string,bytes:number,sha256:string}>} */
function intelRuntimeRows(metadata) {
  return [...metadata.libraries, ...metadata.licenses].map((/** @type {any} */ row) => ({path: row.path, bytes: row.bytes, sha256: row.sha256}));
}

/** @param {{after:(fn:()=>void)=>void}} t @returns {any} */
function hermesIntelRuntimeFixture(t) {
  const f = hermesFixture(t, true), cc = "C:\\Tools\\icx-cl.exe";
  updateHermesFingerprints(f, fingerprints => {fingerprints[cc] = intelRuntimeCatalog.compiler.sha256; return fingerprints;});
  const result = JSON.parse(fs.readFileSync(path.join(f.root, "hermes/result.json"), "utf8"));
  return {...f, cc, result, metadata: intelRuntimeMetadata(cc, result.fingerprints)};
}

test.describe("Windows controller support handoff admission", () => {
  test.describe("local Windows-to-WSL transport identity", () => {
    const localBuildId = "12345678-1234-4234-9234-123456789abc";
    for (const [kind, fixture] of /** @type {const} */ ([["io", ioFixture], ["hermes", hermesFixture]])) {
      test(kind + " uses the same source/content admission without rewriting native receipts", t => {
        const f = fixture(t), ci = windowsControllerHandoffInputs(f.root, f.expected),
          receipts = f.handoff.producers.map((/** @type {{receipt:string}} */ row) =>
            [row.receipt, fileHash(path.join(f.root, row.receipt))]);
        f.handoff.localBuildId = localBuildId;
        const handoffSha256 = seal(f.root, f.handoff),
          local = windowsLocalControllerHandoffInputs(f.root, {kind, localBuildId, handoffSha256});
        assert.equal(local.localBuildId, localBuildId); assert.equal(local.kind, ci.kind);
        assert.deepEqual(local.files, {...ci.files, [ci.handoff]: handoffSha256}, "only the explicit transport identity manifest changes");
        for (const [file, hash] of receipts) {assert.equal(fileHash(path.join(f.root, file)), hash);}
        assert.throws(() => windowsControllerHandoffInputs(f.root, {...f.expected, handoffSha256}), /role or build-only boundary/);
      });
    }
    test("rejects mixed, malformed or foreign local identities", t => {
      const f = ioFixture(t);
      f.handoff.localBuildId = localBuildId;
      let handoffSha256 = seal(f.root, f.handoff);
      assert.throws(() => windowsLocalControllerHandoffInputs(f.root,
        {kind: "io", localBuildId: "not-a-local-build", handoffSha256}), /local build identity/);
      assert.throws(() => windowsLocalControllerHandoffInputs(f.root,
        {kind: "io", localBuildId: "12345678-1234-4234-9234-123456789abd", handoffSha256}), /role or build-only boundary/);
      delete f.handoff.localBuildId; handoffSha256 = seal(f.root, f.handoff);
      assert.throws(() => windowsLocalControllerHandoffInputs(f.root, {kind: "io", localBuildId, handoffSha256}), /role or build-only boundary/);
    });
    test("local identity does not bypass selected-file hash checks", t => {
      const f = ioFixture(t);
      f.handoff.localBuildId = localBuildId;
      const handoffSha256 = seal(f.root, f.handoff);
      fs.appendFileSync(path.join(f.root, "io-sdk/lib/uv_a.lib"), "corrupted");
      assert.throws(() => windowsLocalControllerHandoffInputs(f.root, {kind: "io", localBuildId, handoffSha256}));
    });
  });

  test.describe("optional Intel runtime manifest", () => {
    test("admits exact catalog pins bound to the shared Hermes C/C++ driver fingerprint", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
      const f = hermesIntelRuntimeFixture(t), admitted = validateIntelRuntimeMetadata({intelRuntime: f.metadata, files: intelRuntimeRows(f.metadata)}, f.cc, f.result.fingerprints);
      assert.ok(admitted);
      assert.equal(admitted.compiler.sha256, intelRuntimeCatalog.compiler.sha256);
      assert.deepEqual(admitted.libraries, f.metadata.libraries);
      assert.deepEqual(admitted.licenses, f.metadata.licenses);
      const firstLibrary = admitted.libraries[0]; assert.ok(firstLibrary);
      assert.equal(Object.hasOwn(firstLibrary, "source"), false);
    });
    test("keeps legacy handoffs valid when the optional Intel runtime manifest is absent", () => {
      assert.equal(validateIntelRuntimeMetadata({files: []}, "C:\\Tools\\icx-cl.exe", {}), undefined);
    });
    test("rejects an optional runtime compiler hash not matching both the producer and catalog", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
      const f = hermesIntelRuntimeFixture(t); f.metadata.compiler.sha256 = sha("different compiler");
      assert.throws(() => validateIntelRuntimeMetadata({intelRuntime: f.metadata, files: intelRuntimeRows(f.metadata)}, f.cc, f.result.fingerprints),
        /compiler or payload pins/);
    });
    test("rejects a missing or byte-substituted runtime payload row", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
      const f = hermesIntelRuntimeFixture(t), rows = intelRuntimeRows(f.metadata);
      assert.throws(() => validateIntelRuntimeMetadata({intelRuntime: f.metadata, files: rows.slice(1)}, f.cc, f.result.fingerprints), /payload differs/);
      const firstRow = rows[0]; assert.ok(firstRow); firstRow.bytes++;
      assert.throws(() => validateIntelRuntimeMetadata({intelRuntime: f.metadata, files: rows}, f.cc, f.result.fingerprints), /payload differs/);
    });
    test("rejects extra and path-aliased runtime metadata rows", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
      const f = hermesIntelRuntimeFixture(t), rows = intelRuntimeRows(f.metadata);
      f.metadata.libraries.push({...f.metadata.libraries[0], name: "extra.lib", path: "intel-runtime/lib/extra.lib"});
      assert.throws(() => validateIntelRuntimeMetadata({intelRuntime: f.metadata, files: rows}, f.cc, f.result.fingerprints), /compiler or payload pins/);
      f.metadata.libraries.pop(); f.metadata.licenses[0].path = f.metadata.licenses[0].path.toUpperCase();
      assert.throws(() => validateIntelRuntimeMetadata({intelRuntime: f.metadata, files: rows}, f.cc, f.result.fingerprints), /compiler or payload pins/);
    });
  });
  test("admits the exact selected I/O/OpenSSL closure without claiming full OpenSSL SDK admission", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t), result = windowsControllerHandoffInputs(f.root, f.expected);
    if (result.kind !== "io") {throw new Error("Expected the I/O handoff result");}
    assert.equal(result.kind, "io"); assert.equal(result.handoffSha256, f.expected.handoffSha256);
    assert.equal(result.libraries.ssl, path.join(f.root, "openssl-sdk/lib/libssl.lib"));
    assert.equal(result.libraries.libuv, path.join(f.root, "io-sdk/lib/uv_a.lib"));
    assert.equal(result.libraries.zlibGeneratedInclude, path.join(f.root, "io-sdk/include"));
    assert.equal(Object.hasOwn(result.libraries, "sourceHeaders"), false);
  });
  test("admits historical fetcher hashes when exact I/O and OpenSSL source recipes remain current", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t), historicalFetcherSha256 = "235a3d8ca4cd3b78fd5f1c5f7da74db4241acf501e2e84d47fb8470253fa1ac6";
    assert.notEqual(historicalFetcherSha256, fileHash(path.join(repo, "scripts/fetch-native-sdk-sources.js")));
    updateIoFetch(f, "provenance/io-fetch.json", fetch => {fetch.fetcherSha256 = historicalFetcherSha256;});
    updateIoFetch(f, "provenance/openssl-fetch.json", fetch => {fetch.fetcherSha256 = historicalFetcherSha256;});
    const result = windowsControllerHandoffInputs(f.root, f.expected);
    assert.equal(result.kind, "io");
  });
  test("rejects missing and malformed historical fetcher hashes for both I/O source receipts", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    for (const row of [
      {relative: "provenance/io-fetch.json", target: "io", value: undefined},
      {relative: "provenance/io-fetch.json", target: "io", value: "g".repeat(64)},
      {relative: "provenance/openssl-fetch.json", target: "windows-openssl", value: undefined},
      {relative: "provenance/openssl-fetch.json", target: "windows-openssl", value: "g".repeat(64)},
    ]) {
      const f = ioFixture(t);
      updateIoFetch(f, row.relative, fetch => {
        if (row.value === undefined) {delete fetch.fetcherSha256;} else {fetch.fetcherSha256 = row.value;}
      });
      assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected),
        new RegExp("Windows controller source receipt is stale or incomplete: " + row.target));
    }
  });
  test("keeps the current I/O/OpenSSL acquisition recipes exact with historical fetcher hashes", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    for (const row of [
      {relative: "provenance/io-fetch.json", target: "io", file: "scripts/build-native-io.js",
        fileSha256: sha("stale I/O source recipe"), recipeSha256: sha("stale I/O source recipe")},
      {relative: "provenance/openssl-fetch.json", target: "windows-openssl", file: "scripts/build-native-openssl.js",
        fileSha256: "d5d299d070357e3323446b50cd392861ba00e42a44f4c719c39595ec930cdacc",
        recipeSha256: "16f09bd7a6c6154aefec51c1ab61d19735a84ab137c735f21869e0d2a51a70f9"},
    ]) {
      const f = ioFixture(t);
      updateIoFetch(f, row.relative, fetch => {
        fetch.fetcherSha256 = "235a3d8ca4cd3b78fd5f1c5f7da74db4241acf501e2e84d47fb8470253fa1ac6";
        fetch.recipe.files[row.file] = row.fileSha256; fetch.recipe.sha256 = row.recipeSha256;
      });
      assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected),
        new RegExp("Windows controller source receipt is stale or incomplete: " + row.target));
    }
  });
  test("keeps pinned I/O and OpenSSL archive identities exact with historical fetcher hashes", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    for (const row of [
      {relative: "provenance/io-fetch.json", target: "io", archive: "libuv"},
      {relative: "provenance/openssl-fetch.json", target: "windows-openssl", archive: "openssl"},
    ]) {
      const f = ioFixture(t);
      updateIoFetch(f, row.relative, fetch => {
        fetch.fetcherSha256 = "235a3d8ca4cd3b78fd5f1c5f7da74db4241acf501e2e84d47fb8470253fa1ac6";
        fetch.archives[row.archive].sha256 = sha("substituted source archive");
      });
      assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected),
        new RegExp("Windows controller source receipt pin differs: " + row.target + "/" + row.archive));
    }
  });
  test("rejects a stale OpenSSL producer recipe even when its acquisition receipt uses a historical fetcher hash", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t), resultPath = "openssl-sdk/result.json", result = JSON.parse(fs.readFileSync(path.join(f.root, resultPath), "utf8"));
    result.recipe.graph["scripts/build-native-openssl.js"] = sha("stale OpenSSL producer recipe"); save(f.root, resultPath, result);
    f.expected.handoffSha256 = reseal(f.root, f.handoff);
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /Windows OpenSSL producer source inputs are stale/);
  });
  test("admits the same-run Hermes source/build subset and leaves the Linux AOT tool separate", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = hermesFixture(t), result = windowsControllerHandoffInputs(f.root, f.expected);
    if (result.kind !== "hermes") {throw new Error("Expected the Hermes handoff result");}
    assert.equal(result.hermes.source, path.join(f.root, "hermes/source")); assert.equal(result.hermes.build, path.join(f.root, "hermes/build"));
    assert.equal(result.hermes.license, undefined, "legacy handoffs remain readable without transported root LICENSE");
    assert.equal(Object.hasOwn(result.hermes, "shermaot"), false);
  });
  test("admits the historical fetcher hash when the offline Hermes source recipe remains current", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = hermesFixture(t), historicalFetcherSha256 = "235a3d8ca4cd3b78fd5f1c5f7da74db4241acf501e2e84d47fb8470253fa1ac6";
    assert.notEqual(historicalFetcherSha256, fileHash(path.join(repo, "scripts/fetch-native-sdk-sources.js")));
    updateHermesFetch(f, fetch => {fetch.fetcherSha256 = historicalFetcherSha256;});
    const result = windowsControllerHandoffInputs(f.root, f.expected);
    assert.equal(result.kind, "hermes");
  });
  test("rejects missing and malformed historical Hermes fetcher hashes", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    for (const digest of [undefined, "g".repeat(64)]) {
      const f = hermesFixture(t);
      updateHermesFetch(f, fetch => {
        if (digest === undefined) {delete fetch.fetcherSha256;} else {fetch.fetcherSha256 = digest;}
      });
      assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /Windows Hermes source receipt/);
    }
  });
  test("keeps the pinned Hermes commit and current source recipe exact", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const changedCommit = hermesFixture(t);
    updateHermesFetch(changedCommit, fetch => {fetch.components.hermes.revision = "d".repeat(40);});
    assert.throws(() => windowsControllerHandoffInputs(changedCommit.root, changedCommit.expected), /Windows Hermes source receipt/);

    const changedRecipe = hermesFixture(t);
    updateHermesFetch(changedRecipe, fetch => {fetch.recipe.files["scripts/build-native-hermes.js"] = sha("changed Hermes source recipe");});
    assert.throws(() => windowsControllerHandoffInputs(changedRecipe.root, changedRecipe.expected), /Windows Hermes source receipt/);
  });
  test("keeps Hermes source-root and Git metadata shape pinned", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const staleRoot = hermesFixture(t);
    updateHermesFetch(staleRoot, fetch => {fetch.components.hermes.root += "/stale";});
    assert.throws(() => windowsControllerHandoffInputs(staleRoot.root, staleRoot.expected), /Windows Hermes source receipt/);

    const malformedGit = hermesFixture(t);
    updateHermesFetch(malformedGit, fetch => {fetch.git.sha256 = "not-a-git-hash";});
    assert.throws(() => windowsControllerHandoffInputs(malformedGit.root, malformedGit.expected), /Windows Hermes source receipt/);
  });
  test("keeps the compiled Hermes producer recipe current", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = hermesFixture(t);
    updateHermesProducerRecipe(f, (result, before) => {
      result.recipe.graph["scripts/build-native-hermes.js"] = sha("stale compiled Hermes recipe");
      before.recipe = result.recipe;
    });
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /Windows Hermes producer source inputs are stale/);
  });
  test("rejects changed Hermes source receipt bytes without an updated external inventory", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = hermesFixture(t);
    updateHermesFetch(f, fetch => {fetch.fetcherSha256 = "g".repeat(64);}, false);
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /handoff tree differs from its pinned inventory/);
  });
  test("admits and exposes Hermes root LICENSE only when it matches the original source inventory", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = hermesFixture(t, false, true), result = windowsControllerHandoffInputs(f.root, f.expected),
      license = path.join(f.root, "hermes/source/LICENSE"), source = JSON.parse(fs.readFileSync(path.join(f.root, "hermes/result.json"), "utf8"));
    if (result.kind !== "hermes") {throw new Error("Expected the Hermes handoff result");}
    assert.equal(result.hermes.license, license);
    assert.equal(fileHash(license), source.sources.LICENSE);
    assert.equal(result.files[license], source.sources.LICENSE);
  });
  test("rejects a re-sealed Hermes root LICENSE that differs from the producer source inventory", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = hermesFixture(t, false, true), license = path.join(f.root, "hermes/source/LICENSE");
    fs.writeFileSync(license, "tampered root license\n");
    f.expected.handoffSha256 = seal(f.root, f.handoff);
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /root LICENSE differs from the same-run source inventory/);
  });
  test("admits one fingerprint when the recorded C and C++ roles share a driver", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = hermesFixture(t, true), result = windowsControllerHandoffInputs(f.root, f.expected);
    assert.equal(result.kind, "hermes");
  });
  test("rejects a missing Hermes tool fingerprint", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = hermesFixture(t);
    updateHermesFingerprints(f, fingerprints => {delete fingerprints["C:\\Tools\\icx.exe"]; return fingerprints;});
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /tool fingerprints do not match/);
  });
  test("rejects an extra Hermes tool fingerprint", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = hermesFixture(t, true);
    updateHermesFingerprints(f, fingerprints => {fingerprints["C:\\Tools\\extra.exe"] = sha("extra tool"); return fingerprints;});
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /tool fingerprints do not match/);
  });
  test("rejects case-aliased Hermes tool fingerprints", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = hermesFixture(t);
    updateHermesFingerprints(f, fingerprints => {
      const driverHash = fingerprints["C:\\Tools\\icx-cl.exe"];
      if (!driverHash) {throw new Error("Fixture must have its C compiler fingerprint");}
      delete fingerprints["C:\\Tools\\git.exe"]; fingerprints["c:\\TOOLS\\ICX-CL.EXE"] = driverHash;
      return fingerprints;
    });
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /case alias/);
  });
  test("rejects a substituted file even when the remaining paths still look valid", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t); fs.appendFileSync(path.join(f.root, "io-sdk/include/uv.h"), " substituted");
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /tree differs/);
  });
  test("rejects extra package files", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t); put(f.root, "openssl-sdk/bin/unexpected.exe", "extra");
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /tree differs/);
  });
  test("bounds empty-directory floods as well as file counts", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t), originalOpen = fs.opendirSync, originalLstat = fs.lstatSync;
    const emptyHandle = () => ({readSync: () => null, closeSync: () => false});
    Object.defineProperty(fs, "opendirSync", {configurable: true, value: (/** @type {string} */ directory) => {
      if (directory !== f.root) {return emptyHandle();}
      let index = 0;
      return {readSync: () => index < 100001 ? {name: "empty-" + index++, isSymbolicLink: () => false, isDirectory: () => true, isFile: () => false} : null,
        closeSync: () => false};
    }});
    Object.defineProperty(fs, "lstatSync", {configurable: true, value: (/** @type {string} */ file, /** @type {any} */ options) => file.startsWith(f.root + path.sep + "empty-") ?
      {isDirectory: () => true} : originalLstat(file, options)});
    try {
      assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /entry or nesting bound/);
    } finally {
      Object.defineProperty(fs, "opendirSync", {configurable: true, value: originalOpen});
      Object.defineProperty(fs, "lstatSync", {configurable: true, value: originalLstat});
    }
  });
  test("rejects directory nesting beyond the fixed depth limit", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t); fs.mkdirSync(path.join(f.root, ...Array(65).fill("d")), {recursive: true});
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /entry or nesting bound/);
  });
  test("rejects control characters in externally re-pinned paths", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t); f.handoff.files[0].path = "io-sdk/include/uv\u0001.h"; save(f.root, "handoff.json", f.handoff);
    f.expected.handoffSha256 = fileHash(path.join(f.root, "handoff.json"));
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /Invalid .*inventory/);
  });
  test("rejects symlinked handoff members", {skip: process.platform === "win32"}, (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t), target = path.join(f.root, "io-sdk/include/uv.h"); fs.unlinkSync(target); fs.symlinkSync("zlib.h", target);
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /unsafe or case-aliased/);
  });
  test("rejects traversal in an externally re-pinned handoff manifest", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t); f.handoff.files[0].path = "../escape"; save(f.root, "handoff.json", f.handoff);
    f.expected.handoffSha256 = fileHash(path.join(f.root, "handoff.json"));
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /Invalid .*inventory/);
  });
  test("rejects a forged producer SDK pin even when a new handoff hash is supplied", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t), receipt = JSON.parse(fs.readFileSync(path.join(f.root, "io-sdk/result.json"), "utf8"));
    receipt.pins.zlib.sha256 = sha("substituted source pin"); save(f.root, "io-sdk/result.json", receipt);
    f.expected.handoffSha256 = reseal(f.root, f.handoff);
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /SDK receipt/);
  });
  test("rejects an incomplete selected static-library closure", (/** @type {{after:(fn:()=>void)=>void}} */ t) => {
    const f = ioFixture(t); fs.unlinkSync(path.join(f.root, "openssl-sdk/lib/libssl.lib"));
    f.expected.handoffSha256 = seal(f.root, f.handoff);
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected));
  });
  test("rejects a re-sealed OpenSSL receipt with an unpinned build epoch", t => {
    const f = ioFixture(t), receipt = JSON.parse(fs.readFileSync(path.join(f.root, "openssl-sdk/result.json"), "utf8"));
    receipt.sourceDateEpoch = 1; save(f.root, "openssl-sdk/result.json", receipt);
    f.expected.handoffSha256 = reseal(f.root, f.handoff);
    assert.throws(() => windowsControllerHandoffInputs(f.root, f.expected), /OpenSSL producer receipt/);
  });
});
