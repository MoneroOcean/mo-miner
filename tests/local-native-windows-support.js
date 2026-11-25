"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), os = require("node:os"),
  {createHash} = require("node:crypto");
const local = require("../scripts/build-local-native-windows-support"), io = require("../scripts/build-native-io"),
  openssl = require("../scripts/build-native-openssl"), hermes = require("../scripts/build-native-hermes"),
  {fileHash} = require("../scripts/build-source-inputs");
const {tempDir} = require("./common/temp-dir");

/** @param {string} file @param {string} [text] */
function write(file, text = "fixture\n") {
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, text);
  return file;
}
/** @param {string} root @param {string} name */
function tool(root, name) {return write(path.join(root, "tools", name));}
/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-win-support-test-")),
    workspace = path.join(root, "workspace"), hermesSource = path.join(root, "hermes-source");
  fs.mkdirSync(hermesSource);
  const archiveFiles = {libuv: write(path.join(root, "source/libuv.tar.gz")), zlib: write(path.join(root, "source/zlib.tar.gz")),
      openssl: write(path.join(root, "source/openssl.tar.gz"))},
    files = {nodeGyp: tool(root, "node-gyp.js"), msbuild: tool(root, "msbuild.exe"), dumpbin: tool(root, "dumpbin.exe"),
      python: tool(root, "python.exe"), compiler: tool(root, "icx-cl.exe"), cmake: tool(root, "cmake.exe"), ninja: tool(root, "ninja.exe"),
      git: tool(root, "git.exe"), ar: tool(root, "lib.exe"), link: tool(root, "link.exe"), make: tool(root, "nmake.exe"),
      perl: tool(root, "perl.exe"), rc: tool(root, "rc.exe"), tar: tool(root, "tar.exe")},
    runtimeLibrary = write(path.join(root, "intel/compiler/lib/libircmt.lib"), "!<arch>\n"),
    runtimeLicense = write(path.join(root, "intel/compiler/licenses/license.txt")),
    runtimeEntries = [{name: "libircmt.lib", source: runtimeLibrary, path: "intel-runtime/lib/libircmt.lib", bytes: fs.statSync(runtimeLibrary).size,
      sha256: fileHash(runtimeLibrary), kind: "library"},
    {source: runtimeLicense, path: "intel-runtime/licenses/license.txt", bytes: fs.statSync(runtimeLicense).size,
      sha256: fileHash(runtimeLicense), kind: "license"}],
    options = {workspace, sources: {libuv: archiveFiles.libuv, zlib: archiveFiles.zlib, openssl: archiveFiles.openssl, hermes: hermesSource},
      tools: {core: {nodeGyp: files.nodeGyp, msbuild: files.msbuild, dumpbin: files.dumpbin, python: files.python, compiler: files.compiler},
        io: {cmake: files.cmake, cc: files.compiler, ninja: files.ninja},
        openssl: {ar: files.ar, cc: files.compiler, link: files.link, make: files.make, perl: files.perl, rc: files.rc, tar: files.tar},
        hermes: {git: files.git, cmake: files.cmake, cc: files.compiler, cxx: files.compiler, ninja: files.ninja}}},
    hash = (/** @type {string} */ file) => archiveFiles.libuv === file ? io.ioPins.libuv.sha256 : archiveFiles.zlib === file ? io.ioPins.zlib.sha256 :
      archiveFiles.openssl === file ? openssl.opensslPin.sha256 : files.compiler === file ? require("../runtime/windows-intel-runtime.json").compiler.sha256 : fileHash(file),
    dependencies = /** @type {Parameters<typeof local.buildLocalNativeWindowsSupport>[1]} */ ({
      platform: "win32", arch: "x64", hash, compilerProbe() {return undefined;}, gitProbe() {return undefined;}, runtimeEntries: () => runtimeEntries});
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, options, files, runtimeEntries, dependencies, hash};
}

test.describe("local native Windows support", () => {
  test("Windows support plan binds canonical source/tool paths and current public producer recipes", t => {
    const f = fixture(t), plan = local.windowsSupportPlan(f.options, f.dependencies);
    assert.equal(plan.paths.workspace, f.options.workspace);
    assert.equal(plan.hashes.libuv, io.ioPins.libuv.sha256);
    assert.equal(plan.hashes.zlib, io.ioPins.zlib.sha256);
    assert.equal(plan.hashes.openssl, openssl.opensslPin.sha256);
    assert.equal(plan.io.output, path.join(plan.paths.io, plan.io.key));
    assert.equal(plan.openssl.output, path.join(plan.paths.openssl, plan.openssl.key));
    assert.equal(plan.hermes.output, path.join(plan.paths.hermes, plan.hermes.key));
    assert.ok(plan.hermes.runtimeEntries.length === 2);
    assert.ok(plan.selected.hermes.includes("build/lib/hermesvmlean_a.lib"));
    assert.throws(() => local.windowsSupportPlan({...f.options, workspace: f.root + "/../not-canonical"},
      f.dependencies), /canonical native workspace/);
    assert.throws(() => local.windowsSupportPlan({...f.options, tools: {...f.options.tools, hermes: {...f.options.tools.hermes, cc: f.files.cmake}}},
      f.dependencies), /one selected Intel/);
    let probed = false;
    const badCompilerHash = {...f.dependencies, hash(/** @type {string} */ file) {return file === f.files.compiler ? "0".repeat(64) : f.hash(file);},
      compilerProbe() {probed = true;}};
    assert.throws(() => local.windowsSupportPlan(f.options, badCompilerHash), /runtime catalog/);
    assert.equal(probed, false);
  });

  test("producer admits unchanged content-addressed SDKs and keeps Core's existing cache lane", t => {
    const f = fixture(t), plan = local.windowsSupportPlan(f.options, f.dependencies),
      calls = /** @type {{io:number,openssl:number,hermes:number,core:number}} */ ({io: 0, openssl: 0, hermes: 0, core: 0}),
      coreEnvironments = /** @type {NodeJS.ProcessEnv[]} */ ([]), defaultEnvironment = {PATH: process.env["PATH"] || ""};
    let currentCore = /** @type {{identity:string,stateSha256:string,archiveSha256:string}|null} */ (null), coreGeneration = 0;
    /** @param {string} descriptor @param {NodeJS.ProcessEnv} env */
    const coreIdentity = (descriptor, env) => createHash("sha256").update(JSON.stringify([fileHash(descriptor), fileHash(f.files.msbuild), env["PATH"]])).digest("hex");
    /** @type {Parameters<typeof local.buildLocalNativeWindowsSupport>[1]} */
    const dependencies = {
      ...f.dependencies,
      environment: defaultEnvironment,
      id: (() => {let n = 0; return () => `00000000-0000-4000-8000-${String(++n).padStart(12, "0")}`;})(),
      admitIo(output) {
        return {receipt: path.join(output, "result.json"), libraries: [path.join(output, "lib/uv_a.lib"), path.join(output, "lib/zlibstatic.lib")],
          include: path.join(output, "include")};
      },
      admitOpenSsl(output) {
        return {receipt: path.join(output, "result.json"), include: path.join(output, "include"),
          libraries: [path.join(output, "lib/libcrypto.lib"), path.join(output, "lib/libssl.lib")]};
      },
      buildIo(_archives, output, _tools, _jobs, env) {
        calls.io++; assert.equal(env["MOM_AOT_BUILD_ONLY"], "1");
        assert.equal(env["MOM_BUILD_JOBS"], "2"); assert.equal(env["OMP_NUM_THREADS"], "2");
        fs.mkdirSync(path.join(output, "include/uv"), {recursive: true}); fs.mkdirSync(path.join(output, "lib"), {recursive: true});
        write(path.join(output, "include/uv.h")); write(path.join(output, "include/uv/win.h"));
        write(path.join(output, "lib/uv_a.lib"), "!<arch>\n"); write(path.join(output, "lib/zlibstatic.lib"), "!<arch>\n");
        write(path.join(output, "result.json"), JSON.stringify({inputs: {[f.options.sources.libuv]: io.ioPins.libuv.sha256,
          [f.options.sources.zlib]: io.ioPins.zlib.sha256}, fingerprints: plan.io.tools, recipe: plan.recipes.io, target: "win32", publicationQualified: false}));
      },
      buildOpenSsl(_archive, output, config, env) {
        calls.openssl++; assert.equal(env["MOM_AOT_BUILD_ONLY"], "1");
        assert.equal(env["MOM_BUILD_JOBS"], "2"); assert.equal(env["OMP_NUM_THREADS"], "2");
        fs.mkdirSync(output, {recursive: true}); write(path.join(output, "result.json"), JSON.stringify({archive: {path: f.options.sources.openssl,
          sha256: openssl.opensslPin.sha256}, config: {sha256: require("node:crypto").createHash("sha256").update(JSON.stringify(config)).digest("hex")},
        tools: plan.openssl.tools, recipe: plan.recipes.openssl, publicationQualified: false}));
      },
      buildHermes(source, output, tools, _jobs, env) {
        calls.hermes++; assert.equal(env["MOM_AOT_BUILD_ONLY"], "1");
        assert.equal(env["MOM_BUILD_JOBS"], "2"); assert.equal(env["OMP_NUM_THREADS"], "2");
        const build = hermes.hermesSdkPlan(source, output, tools, 2, "win32");
        fs.mkdirSync(build.source, {recursive: true}); fs.mkdirSync(build.build, {recursive: true});
        fs.mkdirSync(path.join(output, "logs")); write(path.join(build.source, "LICENSE"), "test Hermes license\n");
        for (const [name, artifact] of Object.entries(build.artifacts)) {
          write(artifact, name === "compiler" ? "fixture exe\n" : "!<arch>\n");
        }
        const sourceHash = fileHash(path.join(build.source, "LICENSE")), sources = {LICENSE: sourceHash}, artifacts = Object.fromEntries(Object.entries(build.artifacts)
            .map(([name, file]) => [name, {path: file, sha256: fileHash(file), bytes: fs.statSync(file).size}])),
          cache = new Map([["CMAKE_HOME_DIRECTORY", build.source], ["CMAKE_CACHEFILE_DIR", build.build], ["CMAKE_GENERATOR", "Ninja"]]);
        for (const arg of build.configure) {
          const match = arg.match(/^-D([^:=]+)(?::[^=]+)?=(.*)$/);
          if (match?.[1] && match[2] !== undefined) {cache.set(match[1], match[2]);}
        }
        write(path.join(build.build, "CMakeCache.txt"), [...cache].map(([key, value]) => key + ":UNINITIALIZED=" + value).join("\n") + "\n");
        write(path.join(build.build, "build.ninja"));
        const receipt = {version: 1, target: "win32", pins: hermes.hermesPins, source: build.source, build: build.build, artifacts,
            fingerprints: plan.hermes.tools, sources, recipe: plan.recipes.hermes, buildOnly: true, publicationQualified: false},
          before = {pins: hermes.hermesPins, recipe: plan.recipes.hermes, fingerprints: plan.hermes.tools, sources,
            configure: build.configure, compile: build.compile};
        write(path.join(output, "result.json"), JSON.stringify(receipt)); write(path.join(output, "before.json"), JSON.stringify(before));
      },
      gitCommand(_git, args) {
        const text = args.at(-1) === "HEAD" ? hermes.hermesPins.revision : args.includes("--others") ? "" : args.includes("-z") ? "LICENSE\0" : "";
        return {status: 0, signal: null, stdout: text};
      },
      buildCore(_descriptor, output, env) {
        calls.core++; assert.equal(env["MOM_AOT_BUILD_ONLY"], "1");
        assert.equal(env["MOM_BUILD_JOBS"], "2"); assert.equal(env["OMP_NUM_THREADS"], "2"); coreEnvironments.push(env);
        fs.mkdirSync(output); const archive = write(path.join(output, "cpu-core.lib"), "!<arch>\n1"), archiveSha256 = fileHash(archive),
          identity = coreIdentity(_descriptor, env), stateSha256 = createHash("sha256").update(identity + ":" + (++coreGeneration)).digest("hex"),
          producer = {descriptorSha256: fileHash(_descriptor), objectSha256: archiveSha256,
            archive: {path: archive, sha256: archiveSha256}, incremental: {stateSha256, reason: env["MOM_NATIVE_CACHE"] || env["MOM_FORCE_REBUILD"] || "inputs unchanged"}};
        currentCore = {identity, stateSha256, archiveSha256}; write(path.join(output, "inputs.json"), JSON.stringify(producer) + "\n");
        return producer;
      },
      currentCoreAdmission(descriptor, _prospectiveOutput, env) {
        const identity = coreIdentity(descriptor, env), reusable = currentCore?.identity === identity;
        return {reusable, descriptorSha256: fileHash(descriptor), ...(reusable && currentCore ? currentCore : {})};
      },
      auditCore({output}) {
        const archive = path.join(output, "cpu-core.lib"), receiptFile = path.join(output, "inputs.json");
        return {archive, receiptFile, archiveSha256: fileHash(archive), receiptSha256: fileHash(receiptFile)};
      },
    };
    const first = local.buildLocalNativeWindowsSupport(f.options, dependencies),
      second = local.buildLocalNativeWindowsSupport(f.options, dependencies);
    assert.deepEqual(calls, {io: 1, openssl: 1, hermes: 1, core: 1});
    assert.deepEqual(first.reused, {io: false, openssl: false, hermes: false, core: false});
    assert.deepEqual(second.reused, {io: true, openssl: true, hermes: true, core: true});
    assert.equal(second.core.output, first.core.output);
    assert.equal(fs.readFileSync(second.core.receipt, "utf8"), fs.readFileSync(first.core.receipt, "utf8"), "warm admission preserves original producer receipt bytes");
    assert.equal(first.coreCache.reason, "inputs unchanged");
    assert.equal(second.coreCache.reason, "inputs unchanged");
    assert.equal(coreEnvironments[0]?.["MOM_NATIVE_CACHE"], undefined);
    assert.equal(coreEnvironments[0]?.["MOM_FORCE_REBUILD"], undefined);
    assert.deepEqual(second.io.libraries, [path.join(plan.io.output, "lib/uv_a.lib"), path.join(plan.io.output, "lib/zlibstatic.lib")]);
    assert.equal(second.hermes.hermes.source, path.join(plan.hermes.output, "source"));
    assert.equal(second.hermes.intelRuntime.libraries["libircmt.lib"], f.runtimeEntries[0]?.source);
    assert.ok(second.managedOutputCandidates.io.some(row => row.current));
    assert.equal(second.managedOutputCandidates.core.length, 1, "identical calls do not accumulate Core output directories");
    assert.equal(second.managedOutputCandidates.core.filter(row => row.current).length, 1);
    write(f.files.msbuild, "changed MSBuild tool input\n");
    const changed = local.buildLocalNativeWindowsSupport(f.options, dependencies), changedWarm = local.buildLocalNativeWindowsSupport(f.options, dependencies);
    assert.equal(calls.core, 2, "changed Core tool input produces exactly one new Core output");
    assert.notEqual(changed.core.output, second.core.output);
    assert.equal(changedWarm.core.output, changed.core.output);
    assert.equal(changedWarm.reused.core, true);
    assert.equal(local.managedCoreCandidates(plan.paths.coreRuns, changedWarm.core.output).length, 2);
    const stableAdmission = dependencies.currentCoreAdmission;
    assert.ok(stableAdmission); let admissionCalls = 0;
    dependencies.currentCoreAdmission = (descriptor, prospectiveOutput, env) => {
      const admitted = stableAdmission(descriptor, prospectiveOutput, env);
      return ++admissionCalls === 2 ? {...admitted, stateSha256: "d".repeat(64)} : admitted;
    };
    assert.throws(() => local.buildLocalNativeWindowsSupport(f.options, dependencies), /cache changed during/i,
      "Core cache identity must remain stable across public output admission");
    assert.equal(calls.core, 2);
    dependencies.currentCoreAdmission = stableAdmission;
    /** @param {string} outputFile */
    const checkOutputMutation = outputFile => {
      const original = fs.readFileSync(outputFile); admissionCalls = 0;
      dependencies.currentCoreAdmission = (descriptor, prospectiveOutput, env) => {
        const admitted = stableAdmission(descriptor, prospectiveOutput, env);
        if (++admissionCalls === 2) {fs.appendFileSync(outputFile, "tampered during re-admission\n");}
        return admitted;
      };
      assert.throws(() => local.buildLocalNativeWindowsSupport(f.options, dependencies), /changed during cache re-admission/i,
        "archive and original receipt bytes must be checked again after the second cache admission");
      assert.equal(calls.core, 2);
      fs.writeFileSync(outputFile, original);
    };
    checkOutputMutation(changedWarm.core.archive); checkOutputMutation(changedWarm.core.receipt);
    dependencies.currentCoreAdmission = stableAdmission;
    fs.unlinkSync(changedWarm.core.archive);
    assert.throws(() => local.buildLocalNativeWindowsSupport(f.options, dependencies), /ENOENT|archive|output|admission/i,
      "a cache-matching but damaged Core output must not fall back to another build");
    assert.equal(calls.core, 2);
    write(changedWarm.core.archive, "!<arch>\n1"); fs.unlinkSync(changedWarm.core.receipt);
    assert.throws(() => local.buildLocalNativeWindowsSupport(f.options, dependencies), /ENOENT|receipt|output/i,
      "a partial managed Core output must fail closed rather than start a duplicate build");
    assert.equal(calls.core, 2);
    const explicitControls = /** @type {Array<[string, string]>} */ ([
      ["MOM_NATIVE_CACHE", "0"], ["MOM_FORCE_REBUILD", "1"],
    ]);
    for (const [control, value] of explicitControls) {
      dependencies.environment = {...defaultEnvironment, [control]: value};
      const explicit = local.buildLocalNativeWindowsSupport(f.options, dependencies);
      assert.equal(explicit.coreCache.reason, value);
      assert.equal(coreEnvironments.at(-1)?.[control], value);
    }
    assert.equal(calls.core, 4);
    assert.equal(local.managedCoreCandidates(plan.paths.coreRuns, second.core.output).length, 4);
  });

  test("retires stale support outputs and Core runs without following links out of the cache", t => {
    const root = tempDir(t, "mom-native-win-support-retire-");
    const outside = write(path.join(root, "outside", "keep.txt")),
      ioParent = path.join(root, "io"), coreParent = path.join(root, "core");
    const current = write(path.join(ioParent, "a".repeat(64), "receipt.json")),
      stale = write(path.join(ioParent, "b".repeat(64), "receipt.json")),
      currentCore = write(path.join(coreParent, "00000000-0000-4000-8000-000000000001", "inputs.json")),
      staleCore = write(path.join(coreParent, "00000000-0000-4000-8000-000000000002", "inputs.json"));
    fs.symlinkSync(path.dirname(outside), path.join(path.dirname(stale), "link"), "dir");
    const retired = local.retireManagedOutputs({
      io: local.managedOutputCandidates(ioParent, path.dirname(current)), openssl: [], hermes: [],
      core: local.managedCoreCandidates(coreParent, path.dirname(currentCore)),
    });
    assert.deepEqual(retired.io.map(row => row.root), [path.dirname(stale)]);
    assert.deepEqual(retired.core.map(row => row.root), [path.dirname(staleCore)]);
    assert.ok(fs.existsSync(current) && fs.existsSync(currentCore) && fs.existsSync(outside));
    assert.ok(!fs.existsSync(path.dirname(stale)) && !fs.existsSync(path.dirname(staleCore)));
  });

  test("does not reuse an occupied SDK path when its public receipt admission fails", t => {
    const f = fixture(t), plan = local.windowsSupportPlan(f.options, f.dependencies);
    local.admitWindowsSupportWorkspace(plan.paths.workspace, "win32");
    fs.mkdirSync(plan.paths.io, {recursive: true});
    fs.mkdirSync(plan.io.output);
    const calls = /** @type {string[]} */ ([]), dependencies = /** @type {Parameters<typeof local.buildLocalNativeWindowsSupport>[1]} */ ({
      ...f.dependencies, environment: {PATH: process.env["PATH"] || ""}, buildIo() {calls.push("build"); return undefined;},
    });
    assert.throws(() => local.buildLocalNativeWindowsSupport(f.options, dependencies));
    assert.deepEqual(calls, []);
  });
});
