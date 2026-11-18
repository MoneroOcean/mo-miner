"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {fileHash} = require("../scripts/build-source-inputs");
const {archiveName, archiveSha256, toolNames, openmpBuildRecipe, openmpSdkPlan, validateOpenmpArchiveListing, sourceInventory, verifyCMakeCache,
  validateTools, regularInput, verifyOpenmpArchiveInput, verifyStaticOutput, verifyOpenmpDynamicDependencies, assertOpenmpReceiptShape,
  openmpSdkEnvironment, toolRecord, buildNativeOpenmpSdk, verifyOpenmpSourceInventory, verifyOpenmpNoticeInput, nativeOpenmpInputs} = require("../scripts/build-native-openmp-sdk");

const tools = {cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx", cmake: "/usr/bin/cmake",
  cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx", ninja: "/usr/bin/ninja", readelf: "/usr/bin/readelf", tar: "/usr/bin/tar"};

/** Tests use only synthetic files and never launch compilers, tar, readelf, or OpenMP code. @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-openmp-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return root;
}

function listingFixture() {
  const prefix = "llvm-project-21.1.8.src";
  return [prefix + "/", prefix + "/openmp/", prefix + "/openmp/CMakeLists.txt", prefix + "/cmake/",
    prefix + "/cmake/modules/", prefix + "/cmake/modules/FindLLVM.cmake"].join("\n") + "\n";
}

/** @param {string} root */
function sourceFixture(root) {
  const source = path.join(root, "source"); fs.mkdirSync(path.join(source, "cmake/modules"), {recursive: true});
  fs.mkdirSync(path.join(source, "openmp/runtime/src"), {recursive: true});
  fs.writeFileSync(path.join(source, "cmake/modules/FindLLVM.cmake"), "cmake fixture\n");
  fs.writeFileSync(path.join(source, "openmp/CMakeLists.txt"), "openmp fixture\n");
  fs.writeFileSync(path.join(source, "openmp/LICENSE.TXT"), "LLVM OpenMP license fixture\n");
  fs.writeFileSync(path.join(source, "openmp/runtime/src/omp.h"), "header fixture\n");
  return source;
}

test.describe("Pinned static LLVM OpenMP source SDK", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("the compilation recipe excludes acquisition implementation but retains producer and pin inputs", () => {
    const recipe = openmpBuildRecipe();
    assert.equal(Object.hasOwn(recipe.graph, "scripts/fetch-native-sdk-sources.js"), false);
    for (const file of ["scripts/build-native-openmp-sdk.js", "scripts/native-llvm-sdk.js", "scripts/build-source-inputs.js"]) {
      assert.equal(recipe.graph[file], fileHash(path.resolve(__dirname, "..", file)));
    }
  });
  test("pins the LLVM archive and plans a static host-only target plus closure smoke", () => {
    assert.equal(archiveName, "llvm-project-21.1.8.src.tar.xz");
    assert.equal(archiveSha256, "4633a23617fa31a3ea51242586ea7fb1da7140e426bd62fc164261fe036aa142");
    assert.deepEqual(toolNames, ["cc", "cmake", "cxx", "ninja", "readelf", "tar"]);
    const plan = openmpSdkPlan("/input/" + archiveName, "/tmp/openmp sdk", tools, 2, "linux");
    assert.deepEqual(plan.extract.slice(-2), ["llvm-project-21.1.8.src/openmp/", "llvm-project-21.1.8.src/cmake/"]);
    assert.ok(plan.configure.includes("-DLIBOMP_ENABLE_SHARED=OFF"));
    assert.ok(plan.configure.includes("-DOPENMP_ENABLE_LIBOMPTARGET=OFF"));
    assert.ok(plan.configure.includes("-DOPENMP_ENABLE_OMPT_TOOLS=OFF"));
    assert.ok(plan.configure.includes("-DBUILD_TESTING=OFF"));
    assert.ok(plan.configure.includes("-DLIBOMP_USE_QUAD_PRECISION=ON"));
    assert.ok(plan.configure.includes("-DCMAKE_POSITION_INDEPENDENT_CODE=ON"));
    const prefixMap = '-ffile-prefix-map="/tmp/openmp sdk=/mom/openmp-sdk"',
      compileFlags = "-fno-builtin -ffunction-sections -fdata-sections " + prefixMap;
    assert.ok(plan.configure.includes("-DCMAKE_C_FLAGS=" + compileFlags));
    assert.ok(plan.configure.includes("-DCMAKE_CXX_FLAGS=" + compileFlags));
    assert.ok(plan.configure.includes("-DCMAKE_ASM_FLAGS=" + prefixMap));
    assert.ok(plan.configure.includes("-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF"));
    assert.ok(plan.configure.includes("-DCMAKE_MAKE_PROGRAM=/usr/bin/ninja"));
    assert.deepEqual(plan.buildCommand.slice(-5), ["--target", "omp", "--parallel", "2", "--verbose"]);
    assert.ok(plan.smokeCompile.includes("-ffreestanding"));
    assert.ok(plan.smokeCompile.includes("-fno-builtin"));
    assert.ok(plan.smokeCompile.includes("-ffunction-sections"));
    assert.ok(plan.smokeCompile.includes("-fdata-sections"));
    assert.ok(plan.smokeCompile.includes("-fopenmp"));
    assert.ok(plan.smokeLink.includes("-no-intel-lib"));
    assert.ok(plan.smokeLink.includes("-Wl,--gc-sections"));
    assert.equal(plan.smokeLink.includes("-fopenmp"), false, "the smoke link must not auto-select shared iomp");
    assert.deepEqual(plan.smokeRun, [plan.smokeBinary]);
    assert.equal(plan.cache["LIBOMP_ENABLE_SHARED"], "OFF");
    assert.equal(plan.cache["LIBOMP_USE_QUAD_PRECISION"], "ON");
    assert.equal(plan.cache["OPENMP_ENABLE_LIBOMPTARGET"], "OFF");
    assert.equal(plan.cache["CMAKE_C_FLAGS"], compileFlags);
    assert.equal(plan.cache["CMAKE_CXX_FLAGS"], compileFlags);
    assert.equal(plan.cache["CMAKE_ASM_FLAGS"], prefixMap);
    assert.deepEqual(plan.sourceEnvironment, {PYTHONDONTWRITEBYTECODE: "1"});
    assert.equal(plan.configure.some((arg) => /CMAKE_(?:C|CXX)_FLAGS=.*-ffreestanding/.test(arg)), false,
      "CMake compiler probes must retain normal hosted startup assumptions");
    assert.throws(() => openmpSdkPlan("/input/other.tar.xz", "/tmp/openmp", tools), /pinned LLVM/);
    assert.throws(() => openmpSdkPlan("/input/" + archiveName, "relative", tools), /canonical Linux/);
    assert.throws(() => openmpSdkPlan("/input/" + archiveName, "/tmp/map=ambiguous", tools), /canonical Linux/);
    assert.throws(() => openmpSdkPlan("/input/" + archiveName, "/tmp/openmp", tools, 4), /exactly two jobs/);
    assert.throws(() => openmpSdkPlan("/input/" + archiveName, "/tmp/openmp", tools, 2, "win32"), /canonical Linux/);
  });

  test("forces the receipt-bound Python source policy in every producer child environment", () => {
    const environment = openmpSdkEnvironment({PATH: "/usr/bin", PYTHONDONTWRITEBYTECODE: "0", PYTHONPATH: "/tmp/injected"});
    assert.equal(environment["PYTHONDONTWRITEBYTECODE"], "1");
    assert.equal(environment["PYTHONPATH"], undefined);
    assert.equal(environment["PATH"], "/usr/bin");
  });

  test("tool identity and version maps roundtrip through the JSON receipt", () => {
    const identity = toolRecord(name => ({path: "/tools/" + name, sha256: "a".repeat(64)})),
      versions = toolRecord(name => ({firstLine: name, outputSha256: "b".repeat(64)})),
      receipt = JSON.parse(JSON.stringify({identity, versions}));
    assert.deepEqual(Object.keys(identity), toolNames);
    assert.deepEqual(Object.keys(versions), toolNames);
    assert.equal(Object.getPrototypeOf(identity), Object.prototype);
    assert.equal(Object.getPrototypeOf(versions), Object.prototype);
    assert.deepEqual(receipt.identity, identity);
    assert.deepEqual(receipt.versions, versions);
    assert.notDeepStrictEqual(Object.assign(Object.create(null), identity), receipt.identity,
      "null-prototype maps are not equivalent to JSON-parsed receipt maps");
  });

  test("requires the exact explicit tool set and canonical Intel/readelf/tar identities", () => {
    assert.doesNotThrow(() => validateTools(tools));
    assert.throws(() => validateTools({...tools, compiler: "/opt/icx"}), /exact explicit/);
    assert.throws(() => validateTools({...tools, cc: "icx"}), /explicit Linux tool set|canonical/);
    assert.throws(() => validateTools({...tools, cc: "/opt/intel/oneapi/compiler/2025.0/bin/icx"}), /Intel 2026.2/);
    assert.throws(() => validateTools({...tools, readelf: "/tmp/readelf"}), /canonical/);
    assert.throws(() => validateTools({...tools, tar: "/tmp/tar"}), /canonical/);
  });

  test("validates the full LLVM member catalog and requires only the selected OpenMP/CMake roots", () => {
    const listing = listingFixture(), catalog = validateOpenmpArchiveListing(listing);
    assert.equal(catalog.members, 6);
    assert.equal(catalog.selectedMembers, 5);
    assert.match(catalog.sha256, /^[0-9a-f]{64}$/);
    assert.match(catalog.selectedSha256, /^[0-9a-f]{64}$/);
    for (const altered of [
      listing.replace("llvm-project-21.1.8.src/cmake/\n", ""),
      listing + "llvm-project-21.1.8.src/openmp/../../outside\n",
      listing + "llvm-project-21.1.8.src/openmp/CMakeLists.txt\n",
      listing.replace("llvm-project-21.1.8.src/openmp/", "other/openmp/"),
    ]) {assert.throws(() => validateOpenmpArchiveListing(altered), /archive|member|root/i);}
  });

  test("source inventory binds contents, membership and safe symlink text", {skip: process.platform === "win32"}, t => {
    const root = fixture(t), source = sourceFixture(root), before = sourceInventory(source);
    assert.equal(before.count, 4);
    assert.equal(before.entries, 7);
    assert.equal(before.bytes, Buffer.byteLength("cmake fixture\n") + Buffer.byteLength("openmp fixture\n") +
      Buffer.byteLength("LLVM OpenMP license fixture\n") + Buffer.byteLength("header fixture\n"));
    fs.appendFileSync(path.join(source, "openmp/CMakeLists.txt"), "changed\n");
    assert.notEqual(sourceInventory(source).sha256, before.sha256);
    const linked = path.join(root, "linked"); sourceFixture(linked);
    const outside = path.join(root, "outside"); fs.writeFileSync(outside, "outside\n");
    fs.symlinkSync(path.relative(path.dirname(path.join(linked, "source/openmp/runtime/src/escape")), outside),
      path.join(linked, "source/openmp/runtime/src/escape"));
    assert.throws(() => sourceInventory(path.join(linked, "source")), /unsafe symlink/);
    const extra = path.join(root, "extra"); sourceFixture(extra); fs.writeFileSync(path.join(extra, "source/unselected.txt"), "extra\n");
    assert.throws(() => sourceInventory(path.join(extra, "source")), /only the selected/);
    const bytecode = path.join(root, "bytecode"), bytecodeSource = sourceFixture(bytecode), beforeBytecode = sourceInventory(bytecodeSource),
      cache = path.join(bytecodeSource, "openmp/runtime/tools/__pycache__"); fs.mkdirSync(cache, {recursive: true});
    fs.writeFileSync(path.join(cache, "libomputils.cpython-312.pyc"), "unexpected bytecode");
    const afterBytecode = sourceInventory(bytecodeSource);
    assert.notEqual(afterBytecode.sha256, beforeBytecode.sha256, "the strict before/after source guard must detect Python bytecode additions");
    assert.ok(afterBytecode.entries > beforeBytecode.entries);
  });

  test("receipt admission exposes only the exact regular OpenMP license hashed by the source inventory", t => {
    const root = fixture(t), source = sourceFixture(root), before = sourceInventory(source), expected = {
      path: "source", files: before.count, entries: before.entries, bytes: before.bytes, sha256: before.sha256};
    assert.deepEqual(before.license, {path: "source/openmp/LICENSE.TXT", sha256: fileHash(path.join(source, "openmp/LICENSE.TXT")),
      bytes: Buffer.byteLength("LLVM OpenMP license fixture\n")});
    assert.deepEqual(verifyOpenmpSourceInventory(source, expected).license, before.license);
    assert.deepEqual(verifyOpenmpNoticeInput(root, before.license), {file: path.join(root, before.license.path), notice: before.license});

    fs.appendFileSync(path.join(source, "openmp/LICENSE.TXT"), "changed\n");
    assert.throws(() => verifyOpenmpSourceInventory(source, expected), /membership or bytes changed/);
    assert.throws(() => verifyOpenmpNoticeInput(root, before.license), /source license changed/);

    const missingRoot = fixture(t), missingSource = sourceFixture(missingRoot); fs.unlinkSync(path.join(missingSource, "openmp/LICENSE.TXT"));
    assert.throws(() => sourceInventory(missingSource), /missing its top-level license/);
    assert.throws(() => verifyOpenmpNoticeInput(missingRoot, before.license), /canonical regular file/);
    if (process.platform !== "win32") {
      const linkedRoot = fixture(t), linkedSource = sourceFixture(linkedRoot), license = path.join(linkedSource, "openmp/LICENSE.TXT");
      fs.unlinkSync(license); fs.symlinkSync("CMakeLists.txt", license);
      assert.throws(() => sourceInventory(linkedSource), /license must be a regular canonical file/);
      assert.throws(() => verifyOpenmpNoticeInput(linkedRoot, before.license), /canonical regular file/);
    }

    const excessiveRoot = fixture(t), excessiveSource = sourceFixture(excessiveRoot);
    fs.truncateSync(path.join(excessiveSource, "openmp/LICENSE.TXT"), (8 << 20) + 1);
    assert.throws(() => sourceInventory(excessiveSource), /license must be bounded/);
    assert.throws(() => verifyOpenmpNoticeInput(excessiveRoot, before.license), /source license changed/);
  });

  test("CMake cache admission rejects shared, offload, tests, and compiler/configuration drift", t => {
    const root = fixture(t), plan = openmpSdkPlan("/input/" + archiveName, path.join(root, "sdk"), tools, 2, "linux"),
      cache = path.join(root, "CMakeCache.txt");
    fs.writeFileSync(cache, Object.entries(plan.cache).map(([name, value]) => name + ":STRING=" + value).join("\n") + "\n");
    assert.deepEqual(verifyCMakeCache(cache, plan.cache), plan.cache);
    const changedCases = /** @type {[string, string][]} */ ([
      ["LIBOMP_ENABLE_SHARED", "ON"], ["LIBOMP_USE_QUAD_PRECISION", "OFF"],
      ["OPENMP_ENABLE_LIBOMPTARGET", "ON"], ["BUILD_TESTING", "ON"],
      ["CMAKE_CXX_COMPILER", "/tmp/clang++"], ["CMAKE_CXX_FLAGS", "-O3"], ["CMAKE_C_FLAGS", "-O3"],
      ["CMAKE_POSITION_INDEPENDENT_CODE", "OFF"],
    ]);
    for (const [name, value] of changedCases) {
      const changed = Object.entries(plan.cache).map(([key, selected]) => key + ":STRING=" + (key === name ? value : selected)).join("\n") + "\n";
      fs.writeFileSync(cache, changed);
      assert.throws(() => verifyCMakeCache(cache, plan.cache), new RegExp(name));
    }
    fs.writeFileSync(cache, "BUILD_TESTING:BOOL=OFF\nBUILD_TESTING:STRING=ON\n");
    assert.throws(() => verifyCMakeCache(cache, plan.cache), /Duplicate/);
    const directory = path.join(root, "cache-directory"); fs.mkdirSync(directory);
    assert.throws(() => verifyCMakeCache(directory, plan.cache), /Invalid or excessive/);
    if (process.platform !== "win32") {
      const alias = path.join(root, "cache-link"); fs.symlinkSync(cache, alias);
      assert.throws(() => verifyCMakeCache(alias, plan.cache), /Invalid or excessive/);
    }
  });

  test("producer metadata requires canonical regular files, not symlink aliases", t => {
    const root = fixture(t), file = path.join(root, "metadata.json"), directory = path.join(root, "metadata-directory");
    fs.writeFileSync(file, "{}\n"); assert.equal(regularInput(file, "fixture metadata").size, 3);
    fs.mkdirSync(directory); assert.throws(() => regularInput(directory, "fixture metadata"), /canonical regular file/);
    if (process.platform !== "win32") {
      const alias = path.join(root, "metadata-link"); fs.symlinkSync(file, alias);
      assert.throws(() => regularInput(alias, "fixture metadata"), /canonical regular file/);
      const receiptRoot = path.join(root, "receipt-root"); fs.mkdirSync(receiptRoot);
      fs.symlinkSync(file, path.join(receiptRoot, "result.json"));
      assert.throws(() => nativeOpenmpInputs(receiptRoot), /producer receipt must be a canonical regular file/);
    }
  });

  test("rejects changed or misnamed archive inputs before trusting a fetch receipt", t => {
    const root = fixture(t), wrongPath = path.join(root, "wrong.tar.xz"), source = path.join(root, "source");
    fs.mkdirSync(source); fs.writeFileSync(wrongPath, "not LLVM");
    assert.throws(() => verifyOpenmpArchiveInput(wrongPath), /exact pinned/);
    const archive = path.join(source, archiveName); fs.writeFileSync(archive, "not LLVM");
    const fakeReceipt = {archives: {llvm: {path: archive, sha256: archiveSha256, bytes: fs.statSync(archive).size}}};
    assert.throws(() => verifyOpenmpArchiveInput(archive, archiveSha256, {verify: () => fakeReceipt}), /differs from the immutable/);
    assert.throws(() => verifyOpenmpArchiveInput(archive, "0".repeat(64), {verify: () => fakeReceipt}), /exact pinned/);
  });

  test("admits only bounded regular static archives and rejects compiler runtime dependencies", t => {
    const root = fixture(t), archive = path.join(root, "libomp.a");
    fs.writeFileSync(archive, "!<arch>\nfixture");
    assert.equal(verifyStaticOutput(archive, "fixture").sha256, fileHash(archive));
    fs.writeFileSync(archive, "not an archive");
    assert.throws(() => verifyStaticOutput(archive, "fixture"), /static archive/);
    assert.deepEqual(verifyOpenmpDynamicDependencies(" 0x1 (NEEDED) Shared library: [libc.so.6]\n 0x2 (NEEDED) Shared library: [libm.so.6]\n"),
      ["libc.so.6", "libm.so.6"]);
    for (const library of ["libiomp5.so", "libimf.so", "libomp.so"]) {
      assert.throws(() => verifyOpenmpDynamicDependencies(" 0x1 (NEEDED) Shared library: [" + library + "]\n"), /Unexpected native|proprietary or shared/);
    }
    assert.throws(() => verifyOpenmpDynamicDependencies(" 0x1 (RPATH) Library rpath: [/tmp/compiler]\n"), /Unexpected native/);
  });

  test("receipt and producer guards fail closed without touching an output root", t => {
    const good = {version: 1, role: "linux-static-openmp-sdk", platform: "linux", output: "/tmp/openmp",
      archive: {sha256: archiveSha256}, fetchReceipt: {sha256: "a".repeat(64)}, config: {path: "tools.json", sha256: "b".repeat(64)},
      source: {}, plan: {}, tools: {}, toolVersions: {}, recipe: {files: {}, graph: {}}, buildOutput: {}, sdk: {},
      smoke: {run: {status: 0, signal: null}}, logs: {}, buildOnly: true, compiled: true, executed: true, publicationQualified: false};
    assert.equal(assertOpenmpReceiptShape(good), good);
    for (const altered of [
      {...good, publicationQualified: true},
      {...good, archive: {sha256: "0".repeat(64)}},
      {...good, config: {...good.config, path: "other.json"}},
      {...good, smoke: {run: {status: 1, signal: null}}},
      {...good, extra: "unexpected"},
    ]) {assert.throws(() => assertOpenmpReceiptShape(altered), /Invalid or stale/);}
    const root = fixture(t), output = path.join(root, "new-sdk"), configFile = path.join(root, "tools.json");
    fs.writeFileSync(configFile, "{}\n");
    const config = {version: 1, jobs: 2, tools};
    assert.throws(() => buildNativeOpenmpSdk("/missing/" + archiveName, output, config, {}, configFile, fileHash(configFile)), /Linux x64 and MOM_AOT_BUILD_ONLY/);
    assert.equal(fs.existsSync(output), false);
    assert.throws(() => nativeOpenmpInputs(root), /producer receipt must be a canonical regular file/);
  });
});
