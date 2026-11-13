"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {archiveSha256, toolsComponents, metadataInstallTargets, installTargets, linuxLlvmSdkPlan, verifySourceVersion, llvmSourceSnapshot,
  verifyArchiveCatalog, llvmStaticTargets, linuxDistributionSelection, isRefreshablePriorToolClosure, verifyDistributionGraph, cmakeCache,
  auditLinuxLlvmPrefix, phaseBuildArgs, findClangLinkCommand, buildInputSetsMatch, verifyReuseLayout,
  isReuseProvenance, verifyReuseProvenance, verifyReuseSourceState, refreshLinuxLlvmSdk, verifyAmdgpulibCallsCompileFlag,
  isLegacyLlvmConfigureReceipt} = require("../scripts/build-native-llvm21-sdk");
const {sourceDirectories, archiveName, llvmAmdgpulibCallsCompileOptionsShim, verifyCmakeVersion} = require("../scripts/native-llvm-sdk");
const {fileHash} = require("../scripts/build-source-inputs");

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-linux-llvm21-sdk-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const source = path.join(root, "source"), build = path.join(root, "build"), output = path.join(root, "output with spaces"),
    archive = path.join(root, archiveName), tools = {cmake: "/tools/cmake", ninja: "/tools/ninja", cc: "/intel/icx", cxx: "/intel/icpx"};
  for (const name of sourceDirectories) {fs.mkdirSync(path.join(source, name), {recursive: true});}
  fs.mkdirSync(build, {recursive: true});
  return {root, source, build, output, archive, tools};
}

/** @param {string} file @param {unknown} value */
function writeJson(file, value) {fs.writeFileSync(file, JSON.stringify(value) + "\n");}

/** @param {ReturnType<typeof fixture>} f */
function targetFixture(f) {
  const rows = [
    ["LLVMCore", "llvm", "lib/IR", "STATIC_LIBRARY", "lib/libLLVMCore.a"],
    ["LLVMX86CodeGen", "llvm", "lib/Target/X86", "STATIC_LIBRARY", "lib/libLLVMX86CodeGen.a"],
    ["LLVMPrivate", "llvm", "lib/Private", "STATIC_LIBRARY", "lib/libLLVMPrivate.a"],
    ["clangAAA", "clang", "../clang/lib/Tooling/Refactoring/Rename", "STATIC_LIBRARY", "lib/libclangAAA.a"],
    ["clangBasic", "clang", "../clang/lib/Basic", "STATIC_LIBRARY", "lib/libclangBasic.a"],
    ["clangAST", "clang", "../clang/lib/AST", "STATIC_LIBRARY", "lib/libclangAST.a"],
    ["clangSema", "clang", "../clang/lib/Sema", "STATIC_LIBRARY", "lib/libclangSema.a"],
    ["clangFrontend", "clang", "../clang/lib/Frontend", "STATIC_LIBRARY", "lib/libclangFrontend.a"],
    ["lldCommon", "lld", "../lld/Common", "STATIC_LIBRARY", "lib/liblldCommon.a"],
    ["lld", "lld", "../lld/tools/lld", "EXECUTABLE", "bin/lld"],
    ["clang", "clang", "../clang/tools/driver", "EXECUTABLE", "bin/clang-21"],
    ["llvm-config", "llvm", "../llvm/tools/llvm-config", "EXECUTABLE", "bin/llvm-config"],
    ["llvm-ar", "llvm", "../llvm/tools/llvm-ar", "EXECUTABLE", "bin/llvm-ar"],
    ["llvm-nm", "llvm", "../llvm/tools/llvm-nm", "EXECUTABLE", "bin/llvm-nm"],
    ["llvm-rc", "llvm", "../llvm/tools/llvm-rc", "EXECUTABLE", "bin/llvm-rc"],
    ["llvm-ml", "llvm", "../llvm/tools/llvm-ml", "EXECUTABLE", "bin/llvm-ml"],
    ["llc", "llvm", "../llvm/tools/llc", "EXECUTABLE", "bin/llc"],
    ["opt", "llvm", "../llvm/tools/opt", "EXECUTABLE", "bin/opt"],
    ["llvm-link", "llvm", "../llvm/tools/llvm-link", "EXECUTABLE", "bin/llvm-link"],
    ["llvm-mc", "llvm", "../llvm/tools/llvm-mc", "EXECUTABLE", "bin/llvm-mc"],
    ["llvm-dis", "llvm", "../llvm/tools/llvm-dis", "EXECUTABLE", "bin/llvm-dis"],
    ["llvm-objcopy", "llvm", "../llvm/tools/llvm-objcopy", "EXECUTABLE", "bin/llvm-objcopy"],
    ["llvm-readobj", "llvm", "../llvm/tools/llvm-readobj", "EXECUTABLE", "bin/llvm-readobj"],
    ["llvm-tblgen", "llvm", "../llvm/utils/TableGen", "EXECUTABLE", "bin/llvm-tblgen"],
    ["clang-tblgen", "clang", "../clang/utils/TableGen", "EXECUTABLE", "bin/clang-tblgen"],
  ];
  const reply = path.join(f.build, ".cmake/api/v1/reply"); fs.mkdirSync(reply, {recursive: true});
  const refs = rows.map(([name, , sourcePath, type, artifact], index) => {
    const jsonFile = "target-" + index + ".json";
    writeJson(path.join(reply, jsonFile), {name, type, paths: {source: sourcePath, build: "target-" + index}, artifacts: [{path: artifact}]});
    return {id: name + ":" + type, jsonFile};
  });
  writeJson(path.join(reply, "index-test.json"), {reply: {"codemodel-v2": {jsonFile: "codemodel.json"}}});
  writeJson(path.join(reply, "codemodel.json"), {paths: {source: path.join(f.source, "llvm"), build: f.build}, configurations: [{targets: refs}]});
  const staticNames = rows.filter(([name, , , type]) => type === "STATIC_LIBRARY" && name !== "LLVMPrivate").map(([name]) => "install-" + name);
  const aliasNames = ["llvm-lib", "llvm-ranlib"];
  const targets = new Map([...staticNames, ...toolsComponents.map(name => "install-" + name), ...aliasNames.map(name => "install-" + name),
    ...metadataInstallTargets, "install-mom-distribution"].map(name => [name, []]));
  return {rows, graphTargets: llvmStaticTargets(f.build, f.source), targets};
}

/** @param {ReturnType<typeof fixture>} f */
function writeBuildGraphFixture(f) {
  const model = targetFixture(f), query = path.join(f.build, ".cmake", "api", "v1", "query");
  fs.mkdirSync(query, {recursive: true}); fs.writeFileSync(path.join(query, "codemodel-v2"), "");
  fs.mkdirSync(path.join(f.build, "CMakeFiles"), {recursive: true});
  fs.writeFileSync(path.join(f.build, "CMakeCache.txt"), "cache\n");
  fs.writeFileSync(path.join(f.build, "build.ninja"), "build sample: phony\n");
  fs.writeFileSync(path.join(f.build, "CMakeFiles", "rules.ninja"), "rules\n");
  fs.writeFileSync(path.join(f.build, "compile_commands.json"), "[]\n");
  return model;
}

/** @param {ReturnType<typeof linuxDistributionSelection>} selection @param {string[]} names @param {boolean} [preserveAliases] */
function selectionWithoutTools(selection, names, preserveAliases = false) {
  const removed = new Set(names), phases = selection.phases.map(phase => {
    /** @type {{name: string, artifact: string}[]} */ const retained = [];
    for (const [index, name] of phase.targets.entries()) {
      const artifact = phase.artifacts[index];
      if (typeof artifact !== "string") {throw new Error("Fixture phase is missing an artifact path");}
      if (!removed.has(name)) {retained.push({name, artifact});}
    }
    return {...phase, targets: retained.map(row => row.name), artifacts: retained.map(row => row.artifact)};
  }).filter(phase => phase.targets.length > 0);
  const {distributionComponents: _distributionComponents, ...legacySelection} = selection;
  /** @type {Omit<ReturnType<typeof linuxDistributionSelection>, "distributionComponents"> & {distributionComponents?: string[]}} */
  const prior = {...legacySelection, components: selection.components.filter(name => !removed.has(name)),
    tools: selection.tools.filter(row => !removed.has(row.name)), phases};
  if (preserveAliases) {prior.distributionComponents = selection.distributionComponents.filter(name => !removed.has(name));}
  return prior;
}

test.describe("Public Linux LLVM 21 source SDK producer", () => {
  test("pins the official source/compiler recipe and keeps the ABI export flags", t => {
    const f = fixture(t), plan = linuxLlvmSdkPlan(f.archive, f.output, f.tools, 8);
    assert.equal(archiveSha256, "4633a23617fa31a3ea51242586ea7fb1da7140e426bd62fc164261fe036aa142");
    assert.equal(path.basename(plan.archive), archiveName);
    assert.equal(plan.jobs, 8);
    assert.equal(plan.sourceOptionsFile, path.join(f.output, "llvm-host-source-options.cmake"));
    assert.ok(plan.configure.includes("-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=" + plan.sourceOptionsFile));
    assert.ok(plan.configure.includes("-DLLVM_ENABLE_PROJECTS=clang;lld"));
    assert.ok(plan.configure.includes("-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU"));
    assert.equal(plan.configure.some(value => value.startsWith("-DLLD_INCLUDE_TESTS=")), false);
    assert.ok(plan.configure.includes("-DCMAKE_EXE_LINKER_FLAGS=-Wl,--export-dynamic"));
    assert.ok(plan.configure.includes("-DLLVM_ENABLE_PIC=ON"));
    assert.ok(plan.configure.includes("-DLLVM_ENABLE_PLUGINS=OFF"));
    assert.ok(plan.configure.includes("-DLLVM_EXPORT_SYMBOLS_FOR_PLUGINS=OFF"));
    assert.ok(plan.configure.includes("-DLLVM_ENABLE_LTO=OFF"));
    assert.ok(plan.selectedConfigure.at(-1)?.includes("<generated-components>"));
    assert.throws(() => linuxLlvmSdkPlan(f.archive, f.output, {...f.tools, cc: "/intel/clang"}, 2), /Intel tools/);
    assert.throws(() => linuxLlvmSdkPlan(f.archive, f.output, f.tools, 33), /jobs in 1\.\.32/);
  });

  test("generates a deferred source-scoped host shim with safe literal path quoting", t => {
    const f = fixture(t), source = path.join(f.root, "source with spaces ]]"), shim = llvmAmdgpulibCallsCompileOptionsShim(source);
    assert.match(shim, /cmake_language\(DEFER CALL _mom_llvm_amdgpulibcalls_host_option\)/);
    assert.match(shim, /SOURCE \[=\[/);
    assert.ok(shim.includes("APPEND PROPERTY COMPILE_OPTIONS [[-fno-builtin]]"));
    assert.match(shim, /source with spaces/);
    assert.throws(() => llvmAmdgpulibCallsCompileOptionsShim(path.join(f.root, "bad;source")), /canonical absolute/);
    assert.doesNotThrow(() => verifyCmakeVersion("cmake version 3.24.0\n"));
    assert.doesNotThrow(() => verifyCmakeVersion("cmake version 4.2.1\n"));
    assert.throws(() => verifyCmakeVersion("cmake version 3.23.9\n"), /3.24 or newer/);
    assert.throws(() => verifyCmakeVersion("unrecognized output"), /3.24 or newer/);
  });

  test("requires exactly AMDGPULibCalls.cpp to receive the host-only option", t => {
    const f = fixture(t), sourceFile = path.join(f.source, "llvm/lib/Target/AMDGPU/AMDGPULibCalls.cpp"),
      otherFile = path.join(f.source, "llvm/lib/Support/Other.cpp"), database = path.join(f.build, "compile_commands.json");
    /** @param {string} file @param {boolean} flagged */
    const row = (file, flagged) => ({directory: f.build, file,
      command: "icpx " + (flagged ? "-fno-builtin " : "") + "-c " + file});
    fs.writeFileSync(database, JSON.stringify([row(sourceFile, true), row(otherFile, false)]) + "\n");
    assert.equal(verifyAmdgpulibCallsCompileFlag(f.build, f.source).compileCommands, 2);
    fs.writeFileSync(database, JSON.stringify([row(sourceFile, false), row(otherFile, false)]) + "\n");
    assert.throws(() => verifyAmdgpulibCallsCompileFlag(f.build, f.source), /missing its unique/);
    fs.writeFileSync(database, JSON.stringify([row(sourceFile, true), row(otherFile, true)]) + "\n");
    assert.throws(() => verifyAmdgpulibCallsCompileFlag(f.build, f.source), /escaped/);
    fs.writeFileSync(database, JSON.stringify([row(sourceFile, true), row(sourceFile, true)]) + "\n");
    assert.throws(() => verifyAmdgpulibCallsCompileFlag(f.build, f.source), /exactly one/);
  });

  test("permits old LLVM configure receipts only in the explicit refresh path", () => {
    const output = "/private/old-sdk", producer = "/repo/scripts/build-native-llvm21-sdk.js", oldHash = "a".repeat(64),
      shim = path.join(output, "llvm-host-source-options.cmake"), currentHash = "b".repeat(64), legacy = {
        plan: {output}, configure: {files: {}}, producerSnapshot: {path: path.join(output, "producer-snapshot.js"), sha256: oldHash},
        inputFiles: {[producer]: oldHash},
      };
    assert.equal(isLegacyLlvmConfigureReceipt(legacy, output, producer, currentHash), true);
    assert.equal(isLegacyLlvmConfigureReceipt(legacy, output, producer, oldHash), false, "a current recipe cannot opt out of the shim");
    assert.equal(isLegacyLlvmConfigureReceipt({...legacy, plan: {...legacy.plan, sourceOptionsFile: shim}}, output, producer, currentHash), false);
    assert.equal(isLegacyLlvmConfigureReceipt({...legacy, configure: {...legacy.configure, hostSourceFlag: {flag: "-fno-builtin"}}}, output, producer, currentHash), false);
    assert.equal(isLegacyLlvmConfigureReceipt({...legacy, inputFiles: {...legacy.inputFiles, [shim]: "c".repeat(64)}}, output, producer, currentHash), false);
  });

  test("rejects out-of-range refresh jobs before probing prior metadata or creating output", {skip: process.platform !== "linux" || process.arch !== "x64"}, t => {
    const f = fixture(t), prior = path.join(f.root, "missing-prior"), output = path.join(f.root, "not-created"),
      environment = /** @type {NodeJS.ProcessEnv} */ ({MOM_AOT_BUILD_ONLY: "1"});
    for (const jobs of [0, 33, 1.5, Number.NaN]) {
      assert.throws(() => refreshLinuxLlvmSdk(prior, output, environment, jobs), /jobs must be an integer in 1\.\.32/);
    }
    assert.equal(fs.existsSync(output), false);
  });

  test("creates a fresh SDK prefix while keeping only the validated source and Ninja paths", t => {
    const f = fixture(t), oldSource = path.join(f.root, "old", "source"), oldBuild = path.join(f.root, "old", "build"),
      plan = linuxLlvmSdkPlan(f.archive, f.output, f.tools, 3, {source: oldSource, build: oldBuild});
    assert.equal(plan.source, oldSource);
    assert.equal(plan.build, oldBuild);
    assert.equal(plan.prefix, path.join(f.output, "sdk"));
    assert.ok(plan.configure.includes("-S"));
    assert.ok(plan.configure.includes(path.join(oldSource, "llvm")));
    assert.ok(plan.configure.includes("-B"));
    assert.ok(plan.configure.includes(oldBuild));
    assert.ok(plan.configure.includes("-DCMAKE_INSTALL_PREFIX=" + path.join(f.output, "sdk")));
    assert.throws(() => linuxLlvmSdkPlan(f.archive, f.output, f.tools, 3, {source: oldSource, build: oldSource}), /separate/);
    assert.throws(() => linuxLlvmSdkPlan(f.archive, f.output, f.tools, 3, {source: oldSource, build: f.output}), /new output/);
  });

  test("cache adoption permits only byte drift in an existing recipe input", () => {
    const saved = {"/repo/scripts/build-source-inputs.js": "old-recipe", "/input/llvm.tar.xz": "archive"},
      current = {"/repo/scripts/build-source-inputs.js": "new-recipe", "/input/llvm.tar.xz": "archive"};
    assert.equal(buildInputSetsMatch(saved, current), false, "ordinary public admission stays strict");
    assert.equal(buildInputSetsMatch(saved, current, ["/repo/scripts/build-source-inputs.js"]), true,
      "explicit cache adoption may refresh a changed current recipe");
    assert.equal(buildInputSetsMatch(saved, {...current, "/input/llvm.tar.xz": "changed"}, ["/repo/scripts/build-source-inputs.js"]), false,
      "archive/tool/source inputs never inherit the recipe exception");
    assert.equal(buildInputSetsMatch(saved, {...current, "/repo/scripts/new-recipe.js": "new"}, ["/repo/scripts/build-source-inputs.js"]), false,
      "adding a new dependency cannot be hidden by cache adoption");
  });

  test("requires an owned, non-symlink old cache and a separate new output", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), old = path.join(f.root, "prior"), parent = path.join(f.root, "fresh"), output = path.join(parent, "sdk");
    fs.mkdirSync(old, {mode: 0o700}); fs.mkdirSync(parent, {mode: 0o700});
    for (const name of ["source", "build", "sdk"]) {fs.mkdirSync(path.join(old, name), {mode: 0o700});}
    const priorGraph = {...f, source: path.join(old, "source"), build: path.join(old, "build")};
    writeBuildGraphFixture(priorGraph);
    const priorPlan = linuxLlvmSdkPlan(f.archive, old, f.tools, 2);
    assert.equal(verifyReuseLayout(old, output, priorPlan), parent);
    assert.throws(() => verifyReuseLayout(old, output, {...priorPlan, build: path.join(f.root, "other-build")}), /original canonical/);
    assert.throws(() => verifyReuseLayout(old, path.join(old, "new-output"), priorPlan), /overlaps/);
    const ninjaLog = path.join(f.root, "outside-ninja-log"); fs.writeFileSync(ninjaLog, "state\n");
    fs.symlinkSync(ninjaLog, path.join(old, "build", ".ninja_log"));
    assert.throws(() => verifyReuseLayout(old, output, priorPlan), /symlink|escapes/);
    assert.equal(fs.existsSync(output), false, "state-link rejection happens before creating a refresh output");
    fs.rmSync(path.join(old, "build", ".ninja_log"));
    const outsideCmake = path.join(f.root, "outside-cmake"); fs.mkdirSync(outsideCmake); fs.writeFileSync(path.join(outsideCmake, "rules.ninja"), "rules\n");
    fs.rmSync(path.join(old, "build", "CMakeFiles"), {recursive: true});
    fs.symlinkSync(outsideCmake, path.join(old, "build", "CMakeFiles"));
    assert.throws(() => verifyReuseLayout(old, output, priorPlan), /symlink|escapes/);
    assert.equal(fs.existsSync(output), false, "graph-ancestor rejection happens before creating a refresh output");
    fs.rmSync(path.join(old, "build", "CMakeFiles"));
    fs.chmodSync(old, 0o777);
    assert.throws(() => verifyReuseLayout(old, output, priorPlan), /privately owned/);
    fs.chmodSync(old, 0o700);
    fs.rmSync(path.join(old, "source"), {recursive: true}); fs.symlinkSync(f.root, path.join(old, "source"));
    assert.throws(() => verifyReuseLayout(old, output, priorPlan), /privately owned/);
  });

  test("reuses the same canonical Ninja cache across refresh generations", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), cacheRoot = path.join(f.root, "cache-root"), first = path.join(f.root, "first-generation"),
      secondParent = path.join(f.root, "second-generation"), second = path.join(secondParent, "sdk");
    for (const directory of [cacheRoot, first, secondParent]) {fs.mkdirSync(directory, {mode: 0o700});}
    for (const name of ["source", "build", "sdk"]) {fs.mkdirSync(path.join(cacheRoot, name), {mode: 0o700});}
    fs.mkdirSync(path.join(first, "sdk"), {mode: 0o700});
    f.source = path.join(cacheRoot, "source"); f.build = path.join(cacheRoot, "build");
    writeBuildGraphFixture(f);
    const plan = linuxLlvmSdkPlan(f.archive, first, f.tools, 1,
      {source: path.join(cacheRoot, "source"), build: path.join(cacheRoot, "build")});
    assert.equal(verifyReuseLayout(first, second, plan), secondParent, "the latest prefix may live outside the original cache root");
    assert.throws(() => verifyReuseLayout(first, second,
      {...plan, source: path.join(first, "source")}), /original canonical/);
    assert.throws(() => verifyReuseLayout(first, second,
      {...plan, build: path.join(first, "build")}), /original canonical/);
    const linked = path.join(f.root, "cache-link"); fs.symlinkSync(cacheRoot, linked);
    const linkedPlan = linuxLlvmSdkPlan(f.archive, first, f.tools, 1,
      {source: path.join(linked, "source"), build: path.join(linked, "build")});
    assert.throws(() => verifyReuseLayout(first, second, linkedPlan), /privately owned/);
  });

  test("an interrupted configured cache need not pretend to have an installed SDK", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), prior = path.join(f.root, "prior"), parent = path.join(f.root, "fresh"), output = path.join(parent, "output");
    fs.mkdirSync(prior, {mode: 0o700}); fs.mkdirSync(parent, {mode: 0o700});
    for (const name of ["source", "build"]) {fs.mkdirSync(path.join(prior, name), {mode: 0o700});}
    writeBuildGraphFixture({...f, source: path.join(prior, "source"), build: path.join(prior, "build")});
    const plan = linuxLlvmSdkPlan(f.archive, prior, f.tools, 1);
    assert.throws(() => verifyReuseLayout(prior, output, plan), /ENOENT/);
    assert.equal(verifyReuseLayout(prior, output, plan, false), parent);
    fs.symlinkSync(f.source, plan.prefix);
    assert.throws(() => verifyReuseLayout(prior, output, plan, false), /privately owned/);
  });

  test("rejects symlinked build artifacts", t => {
    const f = fixture(t), outside = path.join(f.root, "outside-file"); fs.writeFileSync(outside, "outside\n");
    targetFixture(f);
    fs.mkdirSync(path.join(f.build, "lib"), {recursive: true});
    fs.symlinkSync(outside, path.join(f.build, "lib", "libLLVMCore.a"));
    assert.throws(() => llvmStaticTargets(f.build, f.source), /symlink|escapes/);
  });

  test("admits LLVM's sibling executable alias without permitting escaping or chained links", t => {
    const f = fixture(t); targetFixture(f);
    const targetFile = path.join(f.build, ".cmake/api/v1/reply/target-10.json"),
      target = JSON.parse(fs.readFileSync(targetFile, "utf8")), binary = path.join(f.build, "bin/clang-21"),
      alias = path.join(f.build, "bin/clang");
    target.artifacts = [{path: "bin/clang"}]; writeJson(targetFile, target);
    fs.mkdirSync(path.dirname(binary)); fs.writeFileSync(binary, "compiler\n");
    fs.symlinkSync("clang-21", alias);
    assert.equal(llvmStaticTargets(f.build, f.source).find(row => row.name === "clang")?.artifact, alias);
    fs.rmSync(alias); fs.symlinkSync("../../outside", alias);
    assert.throws(() => llvmStaticTargets(f.build, f.source), /alias|symlink|escapes/);
    fs.rmSync(alias); fs.symlinkSync("clang-next", alias); fs.symlinkSync("clang-21", path.join(f.build, "bin/clang-next"));
    assert.throws(() => llvmStaticTargets(f.build, f.source), /alias|symlink|escapes/);
  });

  test("binds refresh provenance to unchanged prior receipt files and exact source state", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), result = path.join(f.root, "result.json"), configured = path.join(f.root, "configured.json");
    fs.writeFileSync(result, "prior result\n"); fs.writeFileSync(configured, "prior configure\n");
    const provenance = {from: f.root, priorResultSha256: fileHash(result), priorConfiguredSha256: fileHash(configured)};
    assert.equal(isReuseProvenance(provenance), true);
    assert.equal(isReuseProvenance({...provenance, extra: "unreviewed"}), false);
    assert.doesNotThrow(() => verifyReuseProvenance(provenance));
    fs.writeFileSync(configured, "changed configure\n");
    assert.throws(() => verifyReuseProvenance(provenance), /metadata changed/);
    fs.writeFileSync(configured, "prior configure\n"); fs.writeFileSync(result, "changed result\n");
    assert.throws(() => verifyReuseProvenance(provenance), /metadata changed/);
    const partial = {...provenance, priorResultSha256: null};
    fs.rmSync(result);
    assert.doesNotThrow(() => verifyReuseProvenance(partial));
    fs.writeFileSync(result, "unexpected completion\n");
    assert.throws(() => verifyReuseProvenance(partial), /metadata changed/);
    const sourceState = {sha256: "a".repeat(64), files: 2, bytes: 9};
    assert.doesNotThrow(() => verifyReuseSourceState(sourceState, {...sourceState}));
    assert.throws(() => verifyReuseSourceState(sourceState, {...sourceState, sha256: "b".repeat(64)}), /source changed/);
  });

  test("reads the pinned LLVM version through the upstream CMake include chain", t => {
    const f = fixture(t), llvmCmake = path.join(f.source, "llvm", "CMakeLists.txt"), versionModule = path.join(f.source, "cmake", "Modules", "LLVMVersion.cmake");
    fs.writeFileSync(llvmCmake, [
      // eslint-disable-next-line no-template-curly-in-string -- Literal upstream CMake expansion.
      "set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)",
      // eslint-disable-next-line no-template-curly-in-string -- Literal upstream CMake expansion.
      "include(${LLVM_COMMON_CMAKE_UTILS}/Modules/LLVMVersion.cmake)", "",
    ].join("\n"));
    fs.mkdirSync(path.dirname(versionModule), {recursive: true});
    fs.writeFileSync(versionModule, "if(NOT DEFINED LLVM_VERSION_MAJOR)\n  set(LLVM_VERSION_MAJOR 21)\nendif()\n" +
      "if(NOT DEFINED LLVM_VERSION_MINOR)\n  set(LLVM_VERSION_MINOR 1)\nendif()\n" +
      "if(NOT DEFINED LLVM_VERSION_PATCH)\n  set(LLVM_VERSION_PATCH 8)\nendif()\n");
    assert.doesNotThrow(() => verifySourceVersion(f.source));
    fs.writeFileSync(versionModule, fs.readFileSync(versionModule, "utf8").replace("LLVM_VERSION_MAJOR 21", "LLVM_VERSION_MAJOR 22"));
    assert.throws(() => verifySourceVersion(f.source), /not 21\.1\.8: LLVM_VERSION_MAJOR/);
  });

  test("shares the exact pinned source membership and source hash across platforms", t => {
    const f = fixture(t);
    for (const name of sourceDirectories) {fs.writeFileSync(path.join(f.source, name, "input.txt"), name + "\n");}
    fs.writeFileSync(path.join(f.source, "LICENSE.TXT"), "license\n");
    const before = llvmSourceSnapshot(f.source);
    fs.writeFileSync(path.join(f.source, "libunwind/include", "compact_unwind_encoding.h"), "required source header\n");
    assert.notEqual(llvmSourceSnapshot(f.source).sha256, before.sha256);
    assert.throws(() => llvmSourceSnapshot(path.join(f.root, "missing-source")), /ENOENT/);
    fs.writeFileSync(path.join(f.source, "llvm", "new-member.txt"), "membership changed\n");
    assert.notEqual(llvmSourceSnapshot(f.source).sha256, before.sha256, "new source members invalidate the frozen input snapshot");
  });

  test("rejects missing source roots and marks content changes stale", t => {
    const f = fixture(t);
    fs.writeFileSync(path.join(f.source, "LICENSE.TXT"), "license\n");
    fs.writeFileSync(path.join(f.source, "llvm", "input.txt"), "pinned\n");
    const pinned = llvmSourceSnapshot(f.source);
    fs.writeFileSync(path.join(f.source, "llvm", "input.txt"), "changed\n");
    assert.notDeepEqual(llvmSourceSnapshot(f.source), pinned);
    fs.rmSync(path.join(f.source, "libunwind/include"), {recursive: true});
    assert.throws(() => llvmSourceSnapshot(f.source), /Missing original LLVM 21 source directory: libunwind\/include/);
  });

  test("rejects unsafe archive catalogs and requires original libunwind headers", t => {
    const f = fixture(t), upstream = path.join(f.root, "llvm-project-21.1.8.src");
    for (const name of sourceDirectories) {fs.mkdirSync(path.join(upstream, name), {recursive: true});}
    fs.writeFileSync(path.join(upstream, "LICENSE.TXT"), "license\n");
    const archive = path.join(f.root, "catalog.tar");
    const create = spawnSync("/usr/bin/tar", ["-cf", archive, "--directory", f.root, "llvm-project-21.1.8.src"], {encoding: "utf8"});
    assert.equal(create.status, 0, create.stderr || create.error?.message);
    assert.ok(verifyArchiveCatalog(archive).members > 0);
    fs.rmSync(path.join(upstream, "libunwind/include"), {recursive: true});
    const missing = spawnSync("/usr/bin/tar", ["-cf", archive, "--directory", f.root, "llvm-project-21.1.8.src"], {encoding: "utf8"});
    assert.equal(missing.status, 0, missing.stderr || missing.error?.message);
    assert.throws(() => verifyArchiveCatalog(archive), /omitted required source: libunwind\/include/);
  });

  test("derives installable archives, phase groups, and tool closure from CMake target metadata", t => {
    const f = fixture(t), model = targetFixture(f), selection = linuxDistributionSelection(model.graphTargets, model.targets, f.build);
    assert.deepEqual(selection.components.includes("LLVMPrivate"), false, "uninstallable internals stay outside the SDK");
    assert.deepEqual(selection.components.filter(name => name.startsWith("LLVM")), ["LLVMCore", "LLVMX86CodeGen"]);
    assert.deepEqual(selection.tools.map(row => row.name), [...toolsComponents]);
    assert.equal(toolsComponents.includes("llvm-mt"), false, "llvm-mt is not built with the pinned no-LibXml2 configuration");
    assert.deepEqual(selection.distributionComponents.filter(name => ["llvm-lib", "llvm-ranlib"].includes(name)),
      ["llvm-lib", "llvm-ranlib"], "the MoM distribution explicitly installs the source-defined archive aliases");
    assert.ok(selection.tools.some(row => row.name === "lld"), "ROCr assembly and AdaptiveCpp host links need the LLD driver, not just its archives");
    assert.ok(selection.tools.some(row => row.name === "llvm-readobj"), "the Windows native component admission needs LLVM's COFF reader");
    assert.equal(selection.components.includes("llvm-mca"), false);
    assert.deepEqual(selection.phases.map(row => row.name), ["llvm-targets", "clang-base", "clang-sema", "clang-frontend", "lld", "tools"]);
    assert.deepEqual(selection.phases.flatMap(row => row.targets).sort(), selection.components);
    assert.deepEqual(selection.phases.find(row => row.name === "clang-base")?.targets, ["clangAST", "clangBasic"]);
    assert.deepEqual(selection.phases.find(row => row.name === "clang-sema")?.targets, ["clangSema"]);
    assert.deepEqual(selection.phases.find(row => row.name === "clang-frontend")?.targets, ["clangAAA", "clangFrontend"]);
    assert.ok(selection.archives.every(row => row.relativeArtifact.startsWith("lib/")));
    for (const name of toolsComponents) {
      const missingTool = new Map(model.targets); missingTool.delete("install-" + name);
      assert.throws(() => linuxDistributionSelection(model.graphTargets, missingTool, f.build), new RegExp("required source tool: " + name));
    }
    assert.throws(() => linuxDistributionSelection(model.graphTargets.filter(row => row.name !== "llvm-readobj"), model.targets, f.build),
      /required source tool: llvm-readobj/, "a missing CMake target must not create a partial SDK");
    const wrongReadobjType = model.graphTargets.map(row => row.name === "llvm-readobj" ? {...row, type: "STATIC_LIBRARY"} : row);
    assert.throws(() => linuxDistributionSelection(wrongReadobjType, model.targets, f.build),
      /required source tool: llvm-readobj/, "a non-executable target cannot stand in for the source reader");
  });

  test("accepts only exact prior tool closures as private-refresh input", t => {
    const f = fixture(t), model = targetFixture(f), current = linuxDistributionSelection(model.graphTargets, model.targets, f.build),
      latestR5 = selectionWithoutTools(current, ["llvm-ar", "llvm-nm", "llvm-rc", "llvm-ml", "llvm-readobj"]),
      previous = selectionWithoutTools(current, ["llvm-ar", "llvm-nm", "llvm-rc", "llvm-ml", "llvm-objcopy", "llvm-readobj"]),
      historical = selectionWithoutTools(current, ["llvm-ar", "llvm-nm", "llvm-rc", "llvm-ml", "llvm-mc", "llvm-dis", "llvm-objcopy", "llvm-readobj"]),
      latestR9 = selectionWithoutTools(current, ["llvm-readobj"], true),
      latestR5ToolNames = ["clang", "llvm-config", "llvm-tblgen", "clang-tblgen", "lld", "llc", "opt", "llvm-link",
        "llvm-mc", "llvm-dis", "llvm-objcopy"],
      latestR9ToolNames = ["clang", "llvm-config", "llvm-ar", "llvm-nm", "llvm-rc", "llvm-ml", "llvm-tblgen", "clang-tblgen",
        "lld", "llc", "opt", "llvm-link", "llvm-mc", "llvm-dis", "llvm-objcopy"],
      latestR5PhaseToolNames = ["clang", "clang-tblgen", "llc", "lld", "llvm-config", "llvm-dis", "llvm-link", "llvm-mc",
        "llvm-objcopy", "llvm-tblgen", "opt"];
    assert.deepEqual(latestR5.tools.map(row => row.name), latestR5ToolNames);
    assert.deepEqual(latestR5.phases.find(row => row.name === "tools")?.targets, latestR5PhaseToolNames);
    assert.equal(Object.hasOwn(latestR5, "distributionComponents"), false, "the prior receipt has no new alias-component field");
    assert.equal(isRefreshablePriorToolClosure(latestR5, current), true, "the exact latest admitted 11-tool SDK remains refreshable");
    assert.equal(isRefreshablePriorToolClosure(previous, current), true, "the immediate pre-utility predecessor without objcopy can be refreshed");
    assert.equal(isRefreshablePriorToolClosure(historical, current), true, "the older pre-utility mc/dis predecessor remains explicitly refreshable");
    assert.deepEqual(latestR9.tools.map(row => row.name), latestR9ToolNames);
    assert.deepEqual(latestR9.distributionComponents?.filter(name => ["llvm-lib", "llvm-ranlib"].includes(name)), ["llvm-lib", "llvm-ranlib"]);
    assert.equal(isRefreshablePriorToolClosure(latestR9, current), true, "the exact R9 alias-aware closure can receive llvm-readobj incrementally");
    assert.equal(isRefreshablePriorToolClosure(current, current), false, "the refresh exception is not a public admission mode");
    const prefix = path.join(f.root, "prior-sdk");
    for (const row of previous.archives) {
      const file = path.join(prefix, row.relativeArtifact); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "!<arch>\n");
    }
    for (const name of ["lib/cmake/llvm/LLVMConfig.cmake", "lib/cmake/clang/ClangConfig.cmake", "lib/cmake/lld/LLDConfig.cmake",
      "include/llvm/ADT/StringRef.h", "include/clang/AST/ASTContext.h", "include/lld/Common/Driver.h", "bin/clang", "bin/clang++"]) {
      const file = path.join(prefix, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "present\n");
    }
    for (const row of previous.tools) {
      const file = path.join(prefix, row.relativeArtifact); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "tool\n");
    }
    fs.symlinkSync("llvm-ar", path.join(prefix, "bin/llvm-lib"));
    fs.symlinkSync("llvm-ar", path.join(prefix, "bin/llvm-ranlib"));
    fs.symlinkSync("lld", path.join(prefix, "bin/lld-link"));
    fs.symlinkSync("lld", path.join(prefix, "bin/ld.lld"));
    assert.doesNotThrow(() => auditLinuxLlvmPrefix(prefix, previous), "a privately verified old SDK remains usable as refresh input");
    assert.throws(() => auditLinuxLlvmPrefix(prefix, current), /omitted source tool output/, "current admission requires llvm-objcopy");
    const oldTool = path.join(prefix, "bin/llvm-link"), oldToolBytes = fs.readFileSync(oldTool); fs.rmSync(oldTool);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, previous), /omitted source tool output/, "the prior closure itself stays strict");
    fs.writeFileSync(oldTool, oldToolBytes);
    for (const name of ["llvm-config", "llc", "opt", "llvm-link"]) {
      const otherMissing = selectionWithoutTools(current, [name]);
      assert.equal(isRefreshablePriorToolClosure(otherMissing, current), false, "unrelated omitted tool is not refreshable: " + name);
    }
    assert.equal(isRefreshablePriorToolClosure(selectionWithoutTools(current, ["llvm-readobj", "llvm-nm"], true), current), false,
      "the R9 exception does not permit any additional missing tool");
    const r9WithoutAliases = {...latestR9}; delete r9WithoutAliases.distributionComponents;
    assert.equal(isRefreshablePriorToolClosure(r9WithoutAliases, current), false, "R9's explicit alias metadata is part of its exact predecessor closure");
    const reordered = {...previous, tools: previous.tools.slice().reverse()};
    assert.equal(isRefreshablePriorToolClosure(reordered, current), false, "tool order/closure is hash-bound");
    const reorderedR9 = {...latestR9, tools: latestR9.tools.slice().reverse()};
    assert.equal(isRefreshablePriorToolClosure(reorderedR9, current), false, "the R9 tool order is also exact");
  });

  for (const name of ["llvm-mc", "llvm-dis", "llvm-objcopy"]) {
    test("ROCm resource generation requires source tool " + name, t => {
      const f = fixture(t), model = targetFixture(f), selection = linuxDistributionSelection(model.graphTargets, model.targets, f.build);
      assert.ok(selection.tools.some(row => row.name === name), "the current closure includes " + name);
      const missing = new Map(model.targets); missing.delete("install-" + name);
      assert.throws(() => linuxDistributionSelection(model.graphTargets, missing, f.build), new RegExp("required source tool: " + name));
    });
  }

  test("captures the actual clang link command needed by the ABI evidence", t => {
    const f = fixture(t), phase = {name: "tools", targets: ["clang"], artifacts: []}, args = phaseBuildArgs(f.build, 8, phase),
      log = path.join(f.root, "build-tools.log"), linkCommand = f.tools.cxx + " -fPIC -Wl,--export-dynamic -o " + path.join(f.build, "bin/clang-21");
    assert.ok(args.includes("--verbose"), "the final tools phase must make Ninja print compiler/link commands");
    assert.deepEqual(args.slice(-3), ["--", "-k", "0"]);
    fs.writeFileSync(log, "[1/1] Linking CXX executable bin/clang-21\n");
    assert.throws(() => findClangLinkCommand(f.root, [phase], f.tools.cxx), /exactly one PIC Intel clang link/);
    fs.writeFileSync(log, linkCommand + "\n");
    assert.equal(findClangLinkCommand(f.root, [phase], f.tools.cxx).command, linkCommand);
  });

  test("records a cache-hit link plan without claiming that Ninja executed a relink", t => {
    const f = fixture(t), phase = {name: "tools", targets: ["clang"], artifacts: []},
      log = path.join(f.root, "build-tools.log"), query = path.join(f.root, "clang-link-plan.log"), artifact = path.join(f.root, "clang-21"),
      command = f.tools.cxx + " -fPIC -Wl,--export-dynamic -o " + path.join(f.build, "bin/clang-21");
    fs.writeFileSync(artifact, "compiler\n");
    const cached = {log: query, command, artifact, sha256: fileHash(artifact)};
    fs.writeFileSync(log, "ninja: no work to do.\n"); fs.writeFileSync(query, command + "\n");
    const link = findClangLinkCommand(f.root, [phase], f.tools.cxx, cached);
    assert.equal(link.command, command); assert.equal(link.source, "ninja-plan");
    assert.equal(link.executedThisGeneration, false);
    assert.equal(findClangLinkCommand(f.root, [phase], f.tools.cxx, {...cached, command: "[289/291] " + command}).source, "ninja-plan",
      "Ninja progress decoration does not change the admitted shell command");
    assert.throws(() => findClangLinkCommand(f.root, [phase], f.tools.cxx, {...cached, command: command + " altered"}), /relink required/);
    fs.writeFileSync(artifact, "changed compiler\n");
    assert.throws(() => findClangLinkCommand(f.root, [phase], f.tools.cxx, cached), /relink required/);
    fs.writeFileSync(artifact, "compiler\n");
    fs.writeFileSync(log, command + "\n");
    assert.equal(findClangLinkCommand(f.root, [phase], f.tools.cxx, cached).source, undefined,
      "an actually observed link is not described as a cache hit");
    fs.writeFileSync(log, "ninja: no work to do.\n"); fs.writeFileSync(query, "ninja: no work to do.\n");
    assert.throws(() => findClangLinkCommand(f.root, [phase], f.tools.cxx, cached), /exactly one PIC Intel clang link/);
    fs.writeFileSync(query, command + "\n" + command + "\n");
    assert.throws(() => findClangLinkCommand(f.root, [phase], f.tools.cxx, cached), /exactly one PIC Intel clang link/);
  });

  test("requires source-consumer LLVM tools in the generated graph", t => {
    const f = fixture(t), model = targetFixture(f), selection = linuxDistributionSelection(model.graphTargets, model.targets, f.build);
    for (const name of ["llc", "opt", "llvm-link", "llvm-mc", "llvm-dis", "llvm-objcopy"]) {
      assert.ok(selection.tools.some(row => row.name === name), "the SDK requires its selected LLVM " + name);
      const missing = new Map(model.targets); missing.delete("install-" + name);
      assert.throws(() => linuxDistributionSelection(model.graphTargets, missing, f.build), new RegExp("required source tool: " + name));
    }
  });

  test("checks generated distribution and install metadata targets before any build", t => {
    const f = fixture(t), model = targetFixture(f), selection = linuxDistributionSelection(model.graphTargets, model.targets, f.build),
      aliases = {"llvm-lib": "tools/llvm-ar/llvm-lib", "llvm-ranlib": "tools/llvm-ar/llvm-ranlib"},
      outputs = [...selection.archives, ...selection.tools].map(row => row.relativeArtifact).concat(Object.values(aliases)).sort();
    const aliasRows = Object.entries(aliases).flatMap(([name, edge]) => [
      `build ${name}: phony ${edge}`,
      `build ${edge}: phony bin/${name} bin/llvm-ar`,
    ]);
    const rows = ["build mom-distribution: phony " + outputs.join(" "), "build install-mom-distribution: phony install-all",
      ...selection.distributionComponents.map(name => "build install-" + name + ": phony install-" + name),
      ...aliasRows,
      ...metadataInstallTargets.map(name => "build " + name + ": phony install-metadata")];
    fs.writeFileSync(path.join(f.build, "build.ninja"), rows.join("\n") + "\n");
    assert.equal(verifyDistributionGraph(f.build, selection).install, "install-mom-distribution");
    assert.deepEqual(installTargets, ["install-mom-distribution", ...metadataInstallTargets]);
    const omitted = rows.filter(row => !row.startsWith("build install-clang-headers:"));
    fs.writeFileSync(path.join(f.build, "build.ninja"), omitted.join("\n") + "\n");
    assert.throws(() => verifyDistributionGraph(f.build, selection), /does not match its target\/install graph/);
    const stale = rows.filter(row => !row.startsWith("build install-clang-headers:")).concat("build install-new-component: phony install-new-component");
    fs.writeFileSync(path.join(f.build, "build.ninja"), stale.join("\n") + "\n");
    assert.throws(() => verifyDistributionGraph(f.build, selection), /does not match its target\/install graph/);
  });

  test("audits only regular static archives and requires installed upstream headers and exports", t => {
    const f = fixture(t), model = targetFixture(f), selection = linuxDistributionSelection(model.graphTargets, model.targets, f.build),
      prefix = path.join(f.root, "sdk");
    for (const row of selection.archives) {
      const file = path.join(prefix, row.relativeArtifact); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "!<arch>\n");
    }
    for (const name of ["lib/cmake/llvm/LLVMConfig.cmake", "lib/cmake/clang/ClangConfig.cmake", "lib/cmake/lld/LLDConfig.cmake",
      "include/llvm/ADT/StringRef.h", "include/clang/AST/ASTContext.h", "include/lld/Common/Driver.h", "bin/clang", "bin/clang++"]) {
      const file = path.join(prefix, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "present\n");
    }
    for (const row of selection.tools) {
      const file = path.join(prefix, row.relativeArtifact); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "tool\n");
    }
    fs.symlinkSync("llvm-ar", path.join(prefix, "bin/llvm-lib"));
    fs.symlinkSync("llvm-ar", path.join(prefix, "bin/llvm-ranlib"));
    fs.symlinkSync("lld", path.join(prefix, "bin/lld-link"));
    const linkerAlias = path.join(prefix, "bin/ld.lld");
    fs.symlinkSync("lld", linkerAlias);
    assert.equal(auditLinuxLlvmPrefix(prefix, selection).archives, selection.archives.length);
    for (const name of ["llc", "opt", "llvm-link", "llvm-mc", "llvm-dis", "llvm-objcopy"]) {
      const file = path.join(prefix, "bin", name), contents = fs.readFileSync(file);
      fs.rmSync(file);
      assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /omitted source tool output/);
      fs.writeFileSync(file, contents);
    }
    fs.rmSync(linkerAlias);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /linker driver alias/);
    fs.symlinkSync("clang-21", linkerAlias);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /linker driver alias/);
    fs.rmSync(linkerAlias); fs.symlinkSync("lld", linkerAlias);
    const llvmLibAlias = path.join(prefix, "bin/llvm-lib"), llvmLibTarget = fs.readlinkSync(llvmLibAlias);
    fs.rmSync(llvmLibAlias);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /required tool alias/);
    fs.symlinkSync("../llvm-ar", llvmLibAlias);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /required tool alias/);
    fs.rmSync(llvmLibAlias); fs.symlinkSync(llvmLibTarget, llvmLibAlias);
    const unexpected = path.join(prefix, "lib/libUnexpected.a");
    fs.writeFileSync(unexpected, "!<arch>\n");
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /archive set differs/);
    fs.rmSync(unexpected);
    const header = path.join(prefix, "include/lld/Common/Driver.h"), headerContents = fs.readFileSync(header);
    fs.rmSync(header);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /omitted required upstream export\/header/);
    fs.writeFileSync(header, headerContents);
    const archive = path.join(prefix, selection.archives[0]?.relativeArtifact || "");
    assert.ok(archive.startsWith(prefix + path.sep));
    const archiveContents = fs.readFileSync(archive);
    fs.writeFileSync(archive, "corrupt\n");
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /not a regular static archive/);
    fs.writeFileSync(archive, archiveContents);
    const escaped = path.join(prefix, "escaping-link");
    fs.symlinkSync(path.join(f.root, "outside"), escaped);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /escaping symlink/);
    fs.rmSync(escaped);
    let deep = prefix;
    for (let index = 0; index < 65; index++) {deep = path.join(deep, "d"); fs.mkdirSync(deep);}
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /output inventory exceeds bounds/);
  });

  test("parses CMake cache values without losing list order", t => {
    const f = fixture(t), file = path.join(f.root, "CMakeCache.txt");
    fs.writeFileSync(file, "# generated\nLLVM_ENABLE_PROJECTS:STRING=clang;lld\nCMAKE_EXE_LINKER_FLAGS:STRING=-Wl,--export-dynamic\n");
    const cache = cmakeCache(file);
    assert.equal(cache.get("LLVM_ENABLE_PROJECTS"), "clang;lld");
    assert.equal(cache.get("CMAKE_EXE_LINKER_FLAGS"), "-Wl,--export-dynamic");
  });
});
