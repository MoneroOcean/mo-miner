"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {jitBuildStages, jitBuildStage, jitBuildOutputs, jitBuildHandoff} = require("../scripts/native-jit-stages");
const {buildJitSdk} = require("../scripts/build-native-jit-sdk");
const {fileHash} = require("../scripts/build-source-inputs");
const {jitSdkPlan} = require("../scripts/native-jit-sdk"), {sdkComponents} = require("../scripts/prepare-native-sdk-sources");
const roots = Object.fromEntries(Object.keys(sdkComponents("compiler")).map(name => [name, "/source/" + name]));
const plan = () => jitSdkPlan("linux", roots, "/sdk", "/resources", "/output", {cmake: "/tools/cmake", ninja: "/tools/ninja"}, 2);

test.describe("Bounded original compiler CI partitions", {skip: process.platform !== "linux" && "Linux-hosted compiler build plans"}, () => {
  test("uses four stable sequential phases", () => {assert.deepEqual(jitBuildStages, ["support", "targets", "frontend", "link"]);});
  test("support builds original shared IR/object/analysis foundations", () => {
    assert.deepEqual(jitBuildStage(plan(), "support").slice(4, -5), ["LLVMSupport", "LLVMCore", "LLVMAnalysis", "LLVMObject", "LLVMMC"]);
  });
  test("target stage preserves X86, NVPTX and SPIR-V plus optimization and translator code", () => {
    assert.deepEqual(jitBuildStage(plan(), "targets").slice(4, -5), ["LLVMX86CodeGen", "LLVMNVPTXCodeGen", "LLVMSPIRVCodeGen", "LLVMPasses", "LLVMSPIRVLib"]);
  });
  test("frontend builds original language semantics, parsing and serialization", () => {
    assert.deepEqual(jitBuildStage(plan(), "frontend").slice(4, -5), ["clangSema", "clangAST", "clangSerialization", "clangParse"]);
  });
  test("final stage is exactly the complete upstream proof build", () => {assert.deepEqual(jitBuildStage(plan(), "link"), plan().compile);});
  test("all stages retain the selected tools, build root, job bound and collect-all-errors policy", () => {
    const p = plan();
    for (const stage of jitBuildStages) {const command = jitBuildStage(p, stage); assert.deepEqual(command.slice(0, 4), p.compile.slice(0, 4)); assert.deepEqual(command.slice(-5), p.compile.slice(-5));}
  });
  test("explicit host compilers remain separate from the embedded SDK across every stage", () => {
    const p = jitSdkPlan("linux", roots, "/sdk", "/resources", "/output",
      {cmake: "/tools/cmake", ninja: "/tools/ninja", cc: "/intel/icx", cxx: "/intel/icpx"}, 2);
    assert.deepEqual(p.compilerIdentity, {host: {mode: "explicit-host-override", cc: "/intel/icx", cxx: "/intel/icpx"}, embeddedJit23: "/sdk/bin/clang++"});
    assert.equal(p.definitions["CMAKE_C_COMPILER"], p.compilerIdentity.host.cc);
    assert.equal(p.definitions["CMAKE_CXX_COMPILER"], p.compilerIdentity.host.cxx);
    for (const stage of jitBuildStages) {assert.equal(jitBuildStage(p, stage)[0], p.compile[0]);}
  });
  test("Windows COFF uses the same partitions without a second compiler source implementation", () => {
    const p = jitSdkPlan("windows", roots, "/sdk", "/resources", "/output", {cmake: "/tools/cmake", ninja: "/tools/ninja", toolchain: "/tools/windows.cmake"}, 2);
    for (const stage of jitBuildStages) {assert.deepEqual(jitBuildStage(p, stage), jitBuildStage(plan(), stage));}
  });
  test("the original plan is never mutated", () => {
    const p = plan(), before = JSON.stringify(p); for (const stage of jitBuildStages) {jitBuildStage(p, stage);} assert.equal(JSON.stringify(p), before);
  });
  test("unknown and prototype stage names cannot select arbitrary build targets", () => {
    for (const stage of ["", "all", "constructor", "__proto__", "link;other"]) {assert.throws(() => jitBuildStage(plan(), stage), /named CI/);}
  });
  test("a changed target, extra flag or missing error-collection flag fails closed", () => {
    for (const mutate of [(/** @type {string[]} */ command) => {command[4] = "other";},
      (/** @type {string[]} */ command) => {command.push("--unsafe");}, (/** @type {string[]} */ command) => {command.pop();}]) {
      const p = plan(); mutate(p.compile); assert.throws(() => jitBuildStage(p, "support"));
    }
  });
  test("unbounded or ineffective build jobs cannot be smuggled into a stage", () => {
    for (const jobs of ["0", "17", "2.5", "NaN", "--other"]) {const p = plan(); p.compile[6] = jobs; assert.throws(() => jitBuildStage(p, "support"));}
  });
  test("relative compiler/build roots are not accepted", () => {
    const p = plan(); p.compile[0] = "cmake"; assert.throws(() => jitBuildStage(p, "support"));
    const q = plan(); q.build = "relative"; q.compile[2] = q.build; assert.throws(() => jitBuildStage(q, "support"));
  });
  test("unknown producer phases fail before SDK discovery or output mutation", async () => {
    const c = {version: 1, target: "linux", sources: "/not-used.json", sdk: "/not-used", resources: "/not-used-resources",
      tools: {cmake: "/cmake", ninja: "/ninja"}, jobs: 2};
    await assert.rejects(buildJitSdk(c, "/not-used-output", {MOM_AOT_BUILD_ONLY: "1"}, "other"), /Select configure/);
  });
});

test.describe("Original compiler phase receipt admission", () => {
  const expected = () => ({stage: "support", beforeSha256: "b".repeat(64), configuredSha256: "c".repeat(64), commandsSha256: "d".repeat(64),
    inputs: {"/sdk/compiler": "e".repeat(64)}, outputs: {"lib/a.a": "sha256:" + "a".repeat(64)}});
  const receipt = () => ({version: 1, buildOnly: true, ...expected(), libraries: [path.resolve("/output/build/lib/a.a")]});
  test("admits only the exact previous phase with its bound objects and tools", () => {
    assert.deepEqual(jitBuildHandoff(receipt(), expected(), path.resolve("/output/build")), [path.resolve("/output/build/lib/a.a")]);
  });
  test("a missing phase receipt cannot skip compilation", () => {
    for (const value of [null, undefined, {}, []]) {assert.throws(() => jitBuildHandoff(value, expected(), "/output/build"), /handoff/);}
  });
  test("changed configuration and source receipt hashes invalidate the handoff", () => {
    const r = receipt(); r.beforeSha256 = "0".repeat(64); assert.throws(() => jitBuildHandoff(r, expected(), "/output/build"), /handoff/);
  });
  test("earlier phases cannot masquerade as the required predecessor", () => {
    const r = receipt(); r.stage = "configure"; assert.throws(() => jitBuildHandoff(r, expected(), "/output/build"), /handoff/);
  });
  test("an unknown expected phase is never accepted", () => {
    const r = receipt(), e = expected(); r.stage = "other"; e.stage = "other"; assert.throws(() => jitBuildHandoff(r, e, "/output/build"), /handoff/);
  });
  test("modified generated objects are rejected even with identical source hashes", () => {
    const r = receipt(); r.outputs["lib/a.a"] = "sha256:" + "f".repeat(64); assert.throws(() => jitBuildHandoff(r, expected(), "/output/build"), /handoff/);
  });
  test("a changed external SDK dependency is rejected", () => {
    const r = receipt(); r.inputs["/sdk/compiler"] = "f".repeat(64); assert.throws(() => jitBuildHandoff(r, expected(), "/output/build"), /handoff/);
  });
  test("changed configured tools and archive graph are rejected", () => {
    for (const key of ["configuredSha256", "commandsSha256"]) {const r = receipt(); Reflect.set(r, key, "f".repeat(64)); assert.throws(() => jitBuildHandoff(r, expected(), "/output/build"), /handoff/);}
  });
  test("an archive outside the explicit build directory cannot enter the SDK", () => {
    for (const library of ["/external/a.a", "/output/build/../escape.a", "relative.a", "/output/build/a.so"]) {
      const r = receipt(); r.libraries = [library]; assert.throws(() => jitBuildHandoff(r, expected(), "/output/build"), /handoff/);
    }
  });
  test("empty, duplicated or excessive archive lists are rejected", () => {
    for (const libraries of [[], ["/output/build/a.a", "/output/build/a.a"], Array.from({length: 513}, (_, i) => "/output/build/" + i + ".a")]) {
      const r = receipt(); r.libraries = libraries; assert.throws(() => jitBuildHandoff(r, expected(), "/output/build"), /handoff/);
    }
  });
});

test.describe("Compiler same-workflow output handoffs", () => {
  /** @param {import('node:test').TestContext} t */
  function fixture(t) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-jit-stages-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const file = path.join(root, "object.o"); fs.writeFileSync(file, "original"); return {root, file};
  }
  test("binds generated objects to their bytes, not timestamps", t => {
    const f = fixture(t); assert.deepEqual(jitBuildOutputs(f.root), {"object.o": "mode:" + (fs.statSync(f.file).mode & 0o777).toString(8) + ":sha256:" + fileHash(f.file)});
  });
  test("lost executable permissions on generated host tools invalidate the handoff", {skip: process.platform === "win32" && "POSIX CI build outputs"}, t => {
    const f = fixture(t); fs.chmodSync(f.file, 0o755); const before = jitBuildOutputs(f.root);
    fs.chmodSync(f.file, 0o644); assert.notDeepEqual(jitBuildOutputs(f.root), before);
  });
  test("backdated changed objects invalidate a transferred build", t => {
    const f = fixture(t), old = jitBuildOutputs(f.root), stat = fs.statSync(f.file);
    fs.writeFileSync(f.file, "modified"); fs.utimesSync(f.file, stat.atime, stat.mtime); assert.notDeepEqual(jitBuildOutputs(f.root), old);
  });
  test("added files cannot hide behind an unchanged Ninja log", t => {
    const f = fixture(t), old = jitBuildOutputs(f.root); fs.writeFileSync(path.join(f.root, "unexpected.o"), "extra");
    assert.notDeepEqual(jitBuildOutputs(f.root), old);
  });
  test("removed generated files also invalidate a handoff", t => {
    const f = fixture(t); fs.writeFileSync(path.join(f.root, "other.o"), "other"); const old = jitBuildOutputs(f.root);
    fs.unlinkSync(f.file); assert.notDeepEqual(jitBuildOutputs(f.root), old);
  });
  test("nested outputs use deterministic relative names", t => {
    const f = fixture(t); fs.mkdirSync(path.join(f.root, "lib")); fs.writeFileSync(path.join(f.root, "lib/a.a"), "archive");
    assert.deepEqual(Object.keys(jitBuildOutputs(f.root)), ["lib/a.a", "object.o"]);
  });
  test("an external directory link is recorded but never traversed", t => {
    const f = fixture(t), outside = fs.mkdtempSync(path.join(os.tmpdir(), "mom-jit-outside-"));
    t.after(() => fs.rmSync(outside, {recursive: true, force: true})); fs.writeFileSync(path.join(outside, "unrelated"), "secret");
    try {fs.symlinkSync(outside, path.join(f.root, "alias"), "dir");} catch (error) {
      if (process.platform === "win32" && error instanceof Error && "code" in error && ["EPERM", "EACCES"].includes(String(error.code))) {t.skip("Windows symlink privilege unavailable"); return;}
      throw error;
    }
    const result = jitBuildOutputs(f.root); assert.equal(result["alias"], "link:" + outside); assert.equal(Object.hasOwn(result, "alias/unrelated"), false);
  });
  test("relative directories and file roots are rejected", t => {
    const f = fixture(t); assert.throws(() => jitBuildOutputs("relative"), /real absolute/); assert.throws(() => jitBuildOutputs(f.file), /real absolute/);
  });
  test("an empty directory cannot stand in for configured compiler outputs", t => {
    const f = fixture(t); fs.unlinkSync(f.file); assert.throws(() => jitBuildOutputs(f.root), /empty/);
  });
  test("excessive nesting cannot exhaust the stack", t => {
    const f = fixture(t); fs.mkdirSync(path.join(f.root, ...Array(34).fill("nested")), {recursive: true});
    assert.throws(() => jitBuildOutputs(f.root), /nesting/);
  });
  test("control-containing output names are not admitted to receipts", {skip: process.platform === "win32" && "Windows forbids these filenames"}, t => {
    const f = fixture(t); fs.writeFileSync(path.join(f.root, "bad\nname"), "bad"); assert.throws(() => jitBuildOutputs(f.root), /bounds/);
  });
});
