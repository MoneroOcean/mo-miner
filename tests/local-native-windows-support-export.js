"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), os = require("node:os"),
  {createHash} = require("node:crypto");
const {exportLocalWindowsSupport, exportOrAdmitLocalWindowsSupport} = require("../scripts/build-local-native-windows-support-export"), {fileHash, sourceInputs} = require("../scripts/build-source-inputs");
const {ioPins, ioRecipeFiles} = require("../scripts/build-native-io"), {opensslPin, opensslBuildRecipe} = require("../scripts/build-native-openssl"),
  {hermesPins, hermesRecipeFiles} = require("../scripts/build-native-hermes"), {ioSourceRecipe, sourceRecipe, ioArchiveUrls, buildArchiveUrls} = require("../scripts/fetch-native-sdk-sources");
const repo = path.resolve(__dirname, "..");
/** @param {string} text */
const sha = text => createHash("sha256").update(text).digest("hex");

/** @param {string} file @param {string} text */
function write(file, text) {
  fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, text); return file;
}
/** @param {string} file @param {any} value */
function save(file, value) {return write(file, JSON.stringify(value, null, 2) + "\n");}
/** @param {string} root */
function treeState(root) {
  const rows = /** @type {Array<[string,number,number,number,number]>} */ ([]);
  /** @param {string} directory */
  const visit = directory => {
    for (const entry of fs.readdirSync(directory, {withFileTypes: true})) {
      const file = path.join(directory, entry.name), stat = fs.lstatSync(file);
      rows.push([path.relative(root, file).split(path.sep).join("/"), stat.dev, stat.ino, stat.size, stat.mtimeMs]);
      if (entry.isDirectory()) {visit(file);}
    }
  };
  visit(root); return rows.sort(([a], [b]) => a.localeCompare(b));
}
/** @param {{after:(fn:()=>void)=>void}} t */
function fixture(t) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-win-export-")), producer = path.join(base, "producer"),
    ioRoot = path.join(producer, "io-sdk"), sslRoot = path.join(producer, "openssl-sdk"), hermesRoot = path.join(producer, "hermes-sdk"),
    sourceRoot = path.join(producer, "hermes-bare"), sourceFiles = {LICENSE: "Hermes root license\n", "include/hermes.h": "hermes header\n",
      "API/jsi.h": "Hermes API\n", "public/hermes.h": "public header\n", "lib/internal.h": "unselected source\n"};
  fs.mkdirSync(sourceRoot, {recursive: true});
  const archives = {libuv: "C:\\native-fetch\\libuv-v1.51.0.tar.gz", zlib: "C:\\native-fetch\\zlib-1.3.1.tar.gz",
      openssl: "C:\\native-fetch\\openssl-3.5.9.tar.gz"},
    toolPaths = {cmake: "C:\\Tools\\cmake.exe", cc: "C:\\Tools\\icx-cl.exe", ninja: "C:\\Tools\\ninja.exe"},
    ioArtifactFiles = {"include/uv.h": "uv", "include/zlib.h": "zlib", "include/zconf.h": "zconf",
      "lib/uv_a.lib": "!<arch>\nuv", "lib/zlibstatic.lib": "!<arch>\nzlib"};
  for (const [name, body] of Object.entries(ioArtifactFiles)) {write(path.join(ioRoot, ...name.split("/")), body);}
  const ioArtifacts = Object.fromEntries(Object.keys(ioArtifactFiles).map(name => [name, fileHash(path.join(ioRoot, ...name.split("/")))])),
    ioRecipe = sourceInputs(repo, [], ioRecipeFiles),
    ioReceipt = path.join(ioRoot, "result.json"), ioRecord = {version: 1, target: "win32", pins: ioPins, buildOnly: true, publicationQualified: false,
      inputs: {[archives.libuv]: ioPins.libuv.sha256, [archives.zlib]: ioPins.zlib.sha256},
      fingerprints: Object.fromEntries(Object.values(toolPaths).map(file => [file, sha(file)])), recipe: ioRecipe, artifacts: ioArtifacts};
  save(ioReceipt, ioRecord);

  const sslTools = {ar: "C:\\Tools\\lib.exe", cc: toolPaths.cc, link: "C:\\Tools\\link.exe", make: "C:\\Tools\\nmake.exe",
      perl: "C:\\Tools\\perl.exe", rc: "C:\\Tools\\rc.exe", tar: "C:\\Windows\\System32\\tar.exe"},
    sslConfig = {version: 1, jobs: 2, tools: sslTools},
    sslArtifactBodies = {"include/openssl/opensslv.h": '#define OPENSSL_VERSION_STR "3.5.9"\n', "include/openssl/ssl.h": "ssl header\n",
      "lib/libcrypto.lib": "!<arch>\ncrypto", "lib/libssl.lib": "!<arch>\nssl", "bin/openssl.exe": "not included\n"};
  for (const [name, body] of Object.entries(sslArtifactBodies)) {write(path.join(sslRoot, ...name.split("/")), body);}
  const license = write(path.join(sslRoot, "source/LICENSE.txt"), "OpenSSL license\n"),
    sslArtifacts = Object.fromEntries(Object.keys(sslArtifactBodies).map(name => [name, fileHash(path.join(sslRoot, ...name.split("/")))])),
    sslRecipe = opensslBuildRecipe(),
    sslReceipt = path.join(sslRoot, "result.json"), sslRecord = {version: 1, platform: "win32", target: "VC-WIN64A", opensslVersion: opensslPin.version,
      sourceDateEpoch: opensslPin.sourceDateEpoch, archive: {path: archives.openssl, sha256: opensslPin.sha256, bytes: 3000, members: 20},
      source: {files: 10, bytes: 10000, sha256: sha("source tree"), notice: {path: "source/LICENSE.txt", sha256: fileHash(license)}},
      config: {sha256: sha(JSON.stringify(sslConfig)), tools: sslTools},
      tools: Object.fromEntries(Object.entries(sslTools).map(([name, file]) => [name, {path: file, sha256: sha(file)}])),
      recipe: sslRecipe, artifacts: sslArtifacts, buildOnly: true, executed: false, publicationQualified: false};
  save(sslReceipt, sslRecord);
  const sslAdmissionFiles = /** @type {Record<string, string>} */ ({});
  for (const [name, hash] of Object.entries(sslArtifacts)) {sslAdmissionFiles[path.join(sslRoot, ...name.split("/"))] = hash;}
  sslAdmissionFiles[license] = fileHash(license); sslAdmissionFiles[sslReceipt] = fileHash(sslReceipt);

  for (const [name, body] of Object.entries(sourceFiles)) {write(path.join(hermesRoot, "source", ...name.split("/")), body);}
  const hermesSelected = ["source/LICENSE", "source/include", "source/API", "source/public", "build/lib/config", "build/lib/hermesvmlean_a.lib",
      "build/jsi/jsi.lib", "build/external/boost/boost_1_86_0/libs/context/boost_context.lib"],
    hermesBuildFiles = {"build/bin/shermes.exe": "shermes", "build/public/hermes/Public/hermesPublic.lib": "!<arch>\npublic",
      "build/lib/config/config.h": "config", "build/lib/hermesvmlean_a.lib": "!<arch>\nhermes", "build/jsi/jsi.lib": "!<arch>\njsi",
      "build/external/boost/boost_1_86_0/libs/context/boost_context.lib": "!<arch>\ncontext"};
  for (const [name, body] of Object.entries(hermesBuildFiles)) {write(path.join(hermesRoot, ...name.split("/")), body);}
  const hermesTools = {git: "C:\\Tools\\git.exe", cmake: "C:\\Tools\\cmake.exe", cc: "C:\\Tools\\icx-cl.exe",
      cxx: "C:\\Tools\\icx-cl.exe", ninja: "C:\\Tools\\ninja.exe"},
    patches = ["runtime/hermes-host-rejections.patch", "runtime/hermes-clang-cl.patch"],
    fingerprints = Object.fromEntries([
      ...[...new Set(Object.values(hermesTools))].map(file => [file, sha(file)]),
      ...patches.map(name => ["C:\\repo\\" + name.replaceAll("/", "\\"), fileHash(path.join(repo, name))]),
    ]),
    hermesSources = Object.fromEntries(Object.keys(sourceFiles).map(name => [name, fileHash(path.join(hermesRoot, "source", ...name.split("/")))])),
    hermesRecipe = sourceInputs(repo, [], hermesRecipeFiles("win32")),
    windowsBuild = "C:\\producer\\hermes-sdk\\build", windowsSource = "C:\\producer\\hermes-sdk\\source",
    hermesArtifactPaths = {compiler: "build/bin/shermes.exe", context: "build/external/boost/boost_1_86_0/libs/context/boost_context.lib",
      jsi: "build/jsi/jsi.lib", public: "build/public/hermes/Public/hermesPublic.lib", runtime: "build/lib/hermesvmlean_a.lib"},
    hermesArtifacts = Object.fromEntries(Object.entries(hermesArtifactPaths).map(([name, relative]) => {
      const file = path.join(hermesRoot, ...relative.split("/"));
      return [name, {path: path.win32.join(windowsBuild, ...relative.slice("build/".length).split("/")), sha256: fileHash(file), bytes: fs.statSync(file).size}];
    })),
    configure = [hermesTools.cmake, "-DCMAKE_C_COMPILER:FILEPATH=C:/Tools/icx-cl.exe", "-DCMAKE_CXX_COMPILER:FILEPATH=C:/Tools/icx-cl.exe",
      "-DCMAKE_MAKE_PROGRAM:FILEPATH=C:/Tools/ninja.exe"], compile = [hermesTools.cmake, "--build", windowsBuild, "--parallel", "2"],
    hermesResult = {version: 1, target: "win32", pins: hermesPins, source: windowsSource, build: windowsBuild, artifacts: hermesArtifacts,
      fingerprints, sources: hermesSources, recipe: hermesRecipe, buildOnly: true, publicationQualified: false},
    hermesBefore = {pins: hermesPins, recipe: hermesRecipe, fingerprints, sources: hermesSources, configure, compile},
    hermesReceipt = path.join(hermesRoot, "result.json"), hermesBeforeFile = path.join(hermesRoot, "before.json");
  save(hermesReceipt, hermesResult); save(hermesBeforeFile, hermesBefore);

  const fetcherHash = fileHash(path.join(repo, "scripts/fetch-native-sdk-sources.js")), ioFetchRecipe = ioSourceRecipe(),
    sslFetchRecipe = opensslBuildRecipe(), hermesFetchRecipe = sourceRecipe("hermes"), ioFetch = path.join(base, "provenance/io-fetch.json"),
    sslFetch = path.join(base, "provenance/openssl-fetch.json"), hermesFetch = path.join(base, "wsl-receipts/hermes-fetch.json"),
    hermesRoots = path.join(base, "wsl-receipts/hermes-roots.json"), wslGitRoot = "/tmp/mom-native-hermes-source/roots/hermes";
  save(ioFetch, {version: 1, target: "io", status: "PASS", networked: true, buildOnly: true, compiled: false, publicationQualified: false,
    fetcherSha256: fetcherHash, recipe: {sha256: sha(JSON.stringify(ioFetchRecipe)), files: ioFetchRecipe.graph}, archives: {
      libuv: {path: archives.libuv, url: ioArchiveUrls.libuv, sha256: ioPins.libuv.sha256, bytes: 1000},
      zlib: {path: archives.zlib, url: ioArchiveUrls.zlib, sha256: ioPins.zlib.sha256, bytes: 2000}}});
  save(sslFetch, {version: 1, target: "windows-openssl", status: "PASS", networked: true, buildOnly: true, compiled: false, publicationQualified: false,
    fetcherSha256: fetcherHash, recipe: {sha256: sha(JSON.stringify(sslFetchRecipe)), files: sslFetchRecipe.graph},
    archives: {openssl: {path: archives.openssl, url: buildArchiveUrls.openssl, sha256: opensslPin.sha256, bytes: 3000}}});
  save(hermesFetch, {version: 1, target: "hermes", status: "PASS", networked: true, buildOnly: true, compiled: false, publicationQualified: false,
    fetcherSha256: fetcherHash, recipe: {sha256: sha(JSON.stringify(hermesFetchRecipe)), files: hermesFetchRecipe.graph}, git: {path: "/usr/bin/git", sha256: sha("git")},
    components: {hermes: {origin: "https://github.com/facebook/hermes.git", revision: hermesPins.revision, root: wslGitRoot, gitDirectory: "bare", shallow: true, partialClone: false}}});
  save(hermesRoots, {hermes: wslGitRoot});

  const support = {workspace: producer,
      plan: {sources: {libuv: path.join(base, "libuv.tar.gz"), zlib: path.join(base, "zlib.tar.gz"), openssl: path.join(base, "openssl.tar.gz"), hermes: sourceRoot},
        io: {output: ioRoot, key: sha("io generation")}, openssl: {output: sslRoot, key: sha("OpenSSL generation"), config: sslConfig},
        hermes: {output: hermesRoot, key: sha("Hermes generation"), runtimeEntries: []},
        tools: {hermes: hermesTools}, selected: {hermes: [...hermesSelected, "before.json", "result.json"]}},
      io: {root: ioRoot, receipt: ioReceipt, receiptSha256: fileHash(ioReceipt), artifacts: ioArtifacts},
      openssl: {root: sslRoot, receipt: sslReceipt, receiptSha256: fileHash(sslReceipt), artifacts: sslArtifacts, files: sslAdmissionFiles},
      hermes: {root: hermesRoot, receipt: hermesReceipt, before: hermesBeforeFile, result: hermesResult, runtimeEntries: []}},
    roots = {io: path.join(base, "export/io"), hermes: path.join(base, "export/hermes")},
    cache = path.join(base, "support-cache"),
    sourceReceipts = {ioFetch, opensslFetch: sslFetch, hermesFetch, hermesRoots};
  fs.mkdirSync(path.dirname(roots.io), {recursive: true});
  for (const file of [support.plan.sources.libuv, support.plan.sources.zlib, support.plan.sources.openssl]) {write(file, "archive placeholder");}
  t.after(() => fs.rmSync(base, {recursive: true, force: true}));
  return {base, support, options: {localBuildId: "00000000-0000-4000-8000-000000000001", roots, sourceReceipts}, cache,
    ioReceipt, sslReceipt, hermesReceipt, hermesBeforeFile, sourceFiles: sourceReceipts};
}

/** @param {ReturnType<typeof fixture>} f */
function admissionDependencies(f) {
  return /** @type {any} */ ({planSupport: () => f.support.plan,
    admitIo: () => ({receipt: f.ioReceipt, receiptSha256: fileHash(f.ioReceipt), artifacts: f.support.io.artifacts}),
    admitOpenSsl: () => ({receipt: f.sslReceipt, receiptSha256: fileHash(f.sslReceipt), artifacts: f.support.openssl.artifacts, files: f.support.openssl.files}),
    admitHermes: () => ({receipt: f.hermesReceipt, before: f.hermesBeforeFile, result: f.support.hermes.result, runtimeEntries: []}),
    id: () => "00000000-0000-4000-8000-000000000002"});
}

test.describe("local Windows support export", () => {
  test("exports selected I/O and Hermes closures with unchanged producer/source receipts", t => {
    const f = fixture(t), originalHashes = Object.fromEntries(Object.values(f.sourceFiles).map(file => [file, fileHash(file)])),
      result = exportLocalWindowsSupport(f.support, f.options);
    assert.equal(result.localBuildId, f.options.localBuildId);
    assert.equal(result.io.admitted.kind, "io"); assert.equal(result.hermes.admitted.kind, "hermes");
    assert.equal(result.io.admitted.localBuildId, f.options.localBuildId); assert.equal(result.hermes.admitted.localBuildId, f.options.localBuildId);
    assert.equal(fileHash(f.ioReceipt), result.io.admitted.files[path.join(result.io.root, "io-sdk/result.json")]);
    assert.equal(fileHash(f.sslReceipt), result.io.admitted.files[path.join(result.io.root, "openssl-sdk/result.json")]);
    assert.equal(fileHash(f.hermesReceipt), result.hermes.admitted.files[path.join(result.hermes.root, "hermes/result.json")]);
    for (const [file, hash] of Object.entries(originalHashes)) {assert.equal(fileHash(file), hash);}
    assert.equal(fs.existsSync(path.join(result.io.root, "openssl-sdk/bin/openssl.exe")), false);
    assert.equal(fs.existsSync(path.join(result.hermes.root, "hermes/build/bin/shermes.exe")), false);
  });

  test("rejects changed admitted SDK outputs and missing source proofs before export", t => {
    const changed = fixture(t);
    write(path.join(changed.support.io.root, "lib/uv_a.lib"), "changed archive");
    assert.throws(() => exportLocalWindowsSupport(changed.support, changed.options), /changed/);
    assert.equal(fs.existsSync(changed.options.roots.io), false);

    const missing = fixture(t); fs.unlinkSync(missing.options.sourceReceipts.hermesRoots);
    assert.throws(() => exportLocalWindowsSupport(missing.support, missing.options), /Original source receipt/);
    assert.equal(fs.existsSync(missing.options.roots.hermes), false);
  });

  test("rejects a Hermes runtime plan that differs from the admitted producer result", t => {
    const f = fixture(t);
    f.support.plan.hermes.runtimeEntries = /** @type {any} */ ([{kind: "library", name: "libiomp5md.lib", source: "C:\\runtime\\libiomp5md.lib",
      path: "intel-runtime/lib/libiomp5md.lib", bytes: 10, sha256: sha("runtime")}]);
    assert.throws(() => exportLocalWindowsSupport(f.support, f.options), /Hermes support plan/);
    assert.equal(fs.existsSync(f.options.roots.io), false);
    assert.equal(fs.existsSync(f.options.roots.hermes), false);
  });

  test("rejects a source receipt whose pinned Hermes revision is false", t => {
    const f = fixture(t), receipt = JSON.parse(fs.readFileSync(f.options.sourceReceipts.hermesFetch, "utf8"));
    receipt.components.hermes.revision = "0".repeat(40); save(f.options.sourceReceipts.hermesFetch, receipt);
    assert.throws(() => exportLocalWindowsSupport(f.support, f.options), /Hermes source receipt|pinned original source/);
  });

  test("exports once, then readmits the stable generation without changing its handoffs", t => {
    const f = fixture(t), dependencies = admissionDependencies(f), options = {workspace: f.cache, sourceReceipts: f.options.sourceReceipts},
      fresh = exportOrAdmitLocalWindowsSupport(f.support, options, dependencies),
      before = treeState(f.cache),
      warm = exportOrAdmitLocalWindowsSupport(f.support, options, dependencies);
    assert.equal(fresh.reused, false); assert.equal(warm.reused, true);
    assert.match(fresh.generationKey, /^[a-f0-9]{64}$/); assert.equal(warm.generationKey, fresh.generationKey);
    assert.equal(warm.localBuildId, dependencies.id());
    assert.equal(warm.io.admitted.kind, "io"); assert.equal(warm.hermes.admitted.kind, "hermes");
    assert.equal(warm.io.admitted.localBuildId, warm.localBuildId); assert.equal(warm.hermes.admitted.localBuildId, warm.localBuildId);
    assert.deepEqual(treeState(f.cache), before);
    assert.deepEqual(warm.retirementCandidates, []);
  });

  test("retires an earlier complete generation after exporting a new one without following its links", t => {
    const f = fixture(t), dependencies = admissionDependencies(f), options = {workspace: f.cache, sourceReceipts: f.options.sourceReceipts},
      first = exportOrAdmitLocalWindowsSupport(f.support, options, dependencies),
      fetchReceipt = JSON.parse(fs.readFileSync(f.options.sourceReceipts.ioFetch, "utf8")),
      outside = write(path.join(f.base, "outside/keep.txt"), "not owned by the export cache\n");
    fs.symlinkSync(path.dirname(outside), path.join(first.io.root, "outside-link"), "junction");
    fetchReceipt.fetcherSha256 = sha("a newer source-fetch run"); save(f.options.sourceReceipts.ioFetch, fetchReceipt);
    // An older payload may no longer pass today's source recipe. It is never selected or
    // freshly qualified, and must not prevent creating a new key.
    dependencies.readHandoff = (/** @type {string} */ root, /** @type {any} */ expected) => {
      if (root.includes(first.generationKey)) {throw new Error("historical recipe is no longer current");}
      return require("../scripts/native-windows-controller-handoff").windowsLocalControllerHandoffInputs(root, expected);
    };
    const second = exportOrAdmitLocalWindowsSupport(f.support, options, dependencies);
    assert.notEqual(second.generationKey, first.generationKey);
    assert.deepEqual(second.retirementCandidates.map(row => row.generationKey), [first.generationKey]);
    assert.equal(fs.existsSync(path.join(f.cache, first.generationKey)), false);
    assert.equal(fs.readFileSync(outside, "utf8"), "not owned by the export cache\n");
    assert.equal(fs.existsSync(path.join(f.cache, second.generationKey, "commit.json")), true);
    assert.equal(exportOrAdmitLocalWindowsSupport(f.support, options, dependencies).reused, true);
  });

  test("rejects a self-consistent cached payload that no longer matches current producer bytes", t => {
    const f = fixture(t), dependencies = admissionDependencies(f), options = {workspace: f.cache, sourceReceipts: f.options.sourceReceipts},
      cached = exportOrAdmitLocalWindowsSupport(f.support, options, dependencies), root = cached.io.root,
      artifact = path.join(root, "io-sdk/lib/uv_a.lib"), receiptFile = path.join(root, "io-sdk/result.json"), handoffFile = cached.io.handoff,
      receipt = JSON.parse(fs.readFileSync(receiptFile, "utf8"));
    write(artifact, "!<arch>\nchanged cached bytes");
    receipt.artifacts["lib/uv_a.lib"] = fileHash(artifact); save(receiptFile, receipt);
    const handoff = /** @type {any} */ (JSON.parse(fs.readFileSync(handoffFile, "utf8"))),
      rows = new Map(handoff.files.map((/** @type {any} */ row) => [row.path, row]));
    rows.set("io-sdk/lib/uv_a.lib", {path: "io-sdk/lib/uv_a.lib", bytes: fs.statSync(artifact).size, sha256: fileHash(artifact)});
    rows.set("io-sdk/result.json", {path: "io-sdk/result.json", bytes: fs.statSync(receiptFile).size, sha256: fileHash(receiptFile)});
    handoff.producers[0].receiptSha256 = fileHash(receiptFile);
    handoff.files = [...rows.values()].sort((/** @type {any} */ a, /** @type {any} */ b) => a.path.localeCompare(b.path)); save(handoffFile, handoff);
    const commitFile = path.join(f.cache, cached.generationKey, "commit.json"), commit = JSON.parse(fs.readFileSync(commitFile, "utf8"));
    commit.handoffSha256.io = fileHash(handoffFile); save(commitFile, commit);
    assert.throws(() => exportOrAdmitLocalWindowsSupport(f.support, options, dependencies), /differs from current original producer inputs/);
  });

  test("fails closed on unknown or partial cache generations", t => {
    const f = fixture(t), dependencies = admissionDependencies(f), options = {workspace: f.cache, sourceReceipts: f.options.sourceReceipts};
    exportOrAdmitLocalWindowsSupport(f.support, options, dependencies);
    write(path.join(f.cache, "unfinished/partial.json"), "{}");
    assert.throws(() => exportOrAdmitLocalWindowsSupport(f.support, options, dependencies), /unknown or excessive entries/);

    const partial = fixture(t), partialDependencies = admissionDependencies(partial), partialOptions = {workspace: partial.cache, sourceReceipts: partial.options.sourceReceipts};
    exportOrAdmitLocalWindowsSupport(partial.support, partialOptions, partialDependencies);
    const partialKey = sha("incomplete generation"); fs.mkdirSync(path.join(partial.cache, partialKey));
    assert.throws(() => exportOrAdmitLocalWindowsSupport(partial.support, partialOptions, partialDependencies), /incomplete or contains unknown entries/);
  });

  test("rejects cache workspaces that overlap original producers or receipts", t => {
    const f = fixture(t), dependencies = admissionDependencies(f);
    assert.throws(() => exportOrAdmitLocalWindowsSupport(f.support, {workspace: path.join(f.support.workspace, "exports"), sourceReceipts: f.options.sourceReceipts}, dependencies),
      /outside the checkout, producers and source receipts/);
    assert.equal(fs.existsSync(path.join(f.support.workspace, "exports")), false);
  });

  test("rejects a selected payload with an external hard-link alias", t => {
    const f = fixture(t), dependencies = admissionDependencies(f), options = {workspace: f.cache, sourceReceipts: f.options.sourceReceipts},
      cached = exportOrAdmitLocalWindowsSupport(f.support, options, dependencies), artifact = path.join(cached.io.root, "io-sdk/lib/uv_a.lib");
    fs.linkSync(artifact, path.join(f.base, "cached-alias.lib"));
    assert.throws(() => exportOrAdmitLocalWindowsSupport(f.support, options, dependencies), /regular-file bounds/);
  });
});
