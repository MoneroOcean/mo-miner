"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {
  windowsCoreCacheRoot, acquireWindowsCoreCacheLock, windowsCoreIncludeMembership, windowsCoreIncludeRoots,
  windowsCoreEnvironmentSha256, windowsCoreCacheDecision, windowsCoreCurrentCacheAdmission, clearWindowsCoreRelease, sanitizeBuildEnvironment, windowsCoreTrackerRowsScript,
  windowsCoreTrackerEnvironment, verifyWindowsCoreTrackerInputs,
} = require("../scripts/build-native-windows-core");
const {fileHash} = require("../scripts/build-source-inputs");

test.describe("Windows Core stable MSBuild work graph cache", () => {
  test("uses one guarded cache root and refuses a concurrent owner", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-core-cache-root-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const cache = windowsCoreCacheRoot(root);
    assert.equal(cache, windowsCoreCacheRoot(root));
    const release = acquireWindowsCoreCacheLock(cache);
    assert.throws(() => acquireWindowsCoreCacheLock(cache), /already locked/);
    release();
    assert.doesNotThrow(() => acquireWindowsCoreCacheLock(cache)());
  });

  test("rejects symlinked cache roots and include trees", {skip: process.platform === "win32"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-core-cache-link-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const outside = path.join(root, "outside"), redirected = path.join(root, "redirected");
    fs.mkdirSync(outside); fs.symlinkSync(outside, path.join(root, "build"), "dir");
    assert.throws(() => windowsCoreCacheRoot(root), /symlink|redirected/);
    fs.mkdirSync(redirected); fs.writeFileSync(path.join(redirected, "header.h"), "header\n");
    fs.symlinkSync(redirected, path.join(root, "include-link"), "dir");
    assert.equal(windowsCoreIncludeMembership([path.join(root, "include-link")]).complete, false);
  });

  test("hashes selected MSVC, INCLUDE, and Windows SDK headers and fails closed on missing SDK identity", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-core-includes-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const msvcInclude = path.join(root, "msvc", "include"), libuvInclude = path.join(root, "libuv", "include");
    const sdk = path.join(root, "windows-sdk"), version = "10.0.1.0";
    for (const leaf of ["ucrt", "shared", "um", "winrt", "cppwinrt"]) {
      const directory = path.join(sdk, "Include", version, leaf); fs.mkdirSync(directory, {recursive: true});
      fs.writeFileSync(path.join(directory, leaf + ".h"), leaf + "\n");
    }
    fs.mkdirSync(msvcInclude, {recursive: true}); fs.writeFileSync(path.join(msvcInclude, "vcruntime.h"), "runtime\n");
    fs.mkdirSync(libuvInclude, {recursive: true}); fs.writeFileSync(path.join(libuvInclude, "uv.h"), "uv\n");
    const config = /** @type {import("../scripts/build-native-windows-core").WindowsCoreConfig} */ ({version: 1, nodeGyp: path.join(root, "node-gyp.js"),
      msbuild: path.join(root, "MSBuild.exe"), dumpbin: path.join(root, "dumpbin.exe"), python: path.join(root, "python.exe"), libuvInclude});
    const msvc = {cl: "cl.exe", lib: "lib.exe", ml64: "ml64.exe", version: "14.44.35207", include: msvcInclude};
    const environment = {INCLUDE: msvcInclude + ";" + path.join(sdk, "Include", version, "ucrt"), WindowsSdkDir: sdk, WindowsSDKVersion: version};
    const first = windowsCoreIncludeRoots(config, msvc, environment);
    assert.equal(first.complete, true);
    assert.ok(first.roots.includes(msvcInclude));
    assert.ok(first.roots.includes(path.join(sdk, "Include", version, "um")));
    const rootKeys = first.roots.map(value => process.platform === "win32" ? value.toLowerCase() : value);
    assert.equal(new Set(rootKeys).size, first.roots.length, "canonical aliases are inventoried only once");
    const alias = windowsCoreIncludeMembership([msvcInclude, msvcInclude + path.sep]);
    assert.deepEqual(alias, windowsCoreIncludeMembership([msvcInclude]));
    if (process.platform === "win32") {
      assert.equal(windowsCoreIncludeMembership([msvcInclude, msvcInclude.toUpperCase()]).roots.length, 1);
    }
    fs.writeFileSync(path.join(msvcInclude, "new-shadow.h"), "new header\n");
    const second = windowsCoreIncludeRoots(config, msvc, environment);
    assert.equal(second.complete, true);
    assert.notEqual(second.sha256, first.sha256, "new include membership/content must change the identity");
    assert.equal(windowsCoreIncludeRoots(config, msvc, {}).complete, false);
  });

  test("keeps fresh output and cold-control values out of compiler environment identity without mutating callers", () => {
    const first = {PATH: "tool-path", MOM_NATIVE_OUTPUT: "fresh-a", MOM_NATIVE_BUILD_OUTPUT: "a", MOM_NATIVE_BUILD_STAGE: "a",
      MOM_NATIVE_SDK: "sdk", MOM_NATIVE_UPX: "upx", MOM_NATIVE_UPX_SHA256: "hash-a", HOSTNAME: "host-a", MOM_FORCE_REBUILD: "1", MOM_NATIVE_CACHE: "0"};
    const second = {...first, MOM_NATIVE_OUTPUT: "fresh-b", MOM_NATIVE_BUILD_OUTPUT: "b", MOM_NATIVE_BUILD_STAGE: "b",
      MOM_NATIVE_UPX_SHA256: "hash-b", HOSTNAME: "host-b", MOM_FORCE_REBUILD: "0", MOM_NATIVE_CACHE: "1"};
    const sanitizedFirst = sanitizeBuildEnvironment(first, "python.exe"), sanitizedSecond = sanitizeBuildEnvironment(second, "python.exe");
    assert.equal(windowsCoreEnvironmentSha256(sanitizedFirst), windowsCoreEnvironmentSha256(sanitizedSecond));
    for (const key of ["MOM_NATIVE_OUTPUT", "MOM_NATIVE_BUILD_OUTPUT", "MOM_NATIVE_BUILD_STAGE", "MOM_NATIVE_SDK", "MOM_NATIVE_UPX",
      "MOM_NATIVE_UPX_SHA256", "HOSTNAME", "MOM_FORCE_REBUILD", "MOM_NATIVE_CACHE"]) {
      assert.equal(Object.hasOwn(sanitizedFirst, key), false, key);
    }
    assert.equal(first.MOM_NATIVE_OUTPUT, "fresh-a");
    assert.equal(first.MOM_FORCE_REBUILD, "1");
  });

  test("clears ambient LIB only for the static Core tracker census, not build identity or MSBuild", () => {
    const first = {PATH: "tool-path", LIB: "sdk-lib-a;intel-lib"}, second = {...first, LIB: "sdk-lib-b;intel-lib"};
    const trackerFirst = windowsCoreTrackerEnvironment(first), trackerSecond = windowsCoreTrackerEnvironment(second);
    assert.equal(trackerFirst["LIB"], ""); assert.equal(trackerSecond["LIB"], "");
    assert.equal(windowsCoreEnvironmentSha256(trackerFirst), windowsCoreEnvironmentSha256(trackerSecond));
    const buildFirst = sanitizeBuildEnvironment(first, "python.exe"), buildSecond = sanitizeBuildEnvironment(second, "python.exe");
    assert.equal(buildFirst["LIB"], first["LIB"]); assert.equal(buildSecond["LIB"], second["LIB"]);
    assert.notEqual(windowsCoreEnvironmentSha256(buildFirst), windowsCoreEnvironmentSha256(buildSecond));
    assert.equal(first.LIB, "sdk-lib-a;intel-lib", "tracker preparation must not mutate the real build environment");
    assert.deepEqual(trackerFirst, {PATH: first.PATH, LIB: ""});
  });

  test("uses the OS Windows PowerShell module path instead of inherited PowerShell 7 paths", () => {
    const systemRoot = path.resolve(os.tmpdir(), "mom-windows-system-root"), environment = {PATH: "tool-path", lib: "sdk-lib", SystemRoot: systemRoot,
        pSmOdUlEpAtH: "C:\\PowerShell\\Modules", PSMODULEPATH: "C:\\WindowsPowerShell\\Modules"}, tracker = windowsCoreTrackerEnvironment(environment);
    assert.equal(tracker["PSModulePath"], path.join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "Modules"));
    assert.deepEqual(Object.keys(tracker).filter(key => key.toUpperCase() === "PSMODULEPATH"), ["PSModulePath"]);
    assert.equal(tracker["LIB"], ""); assert.equal(environment.lib, "sdk-lib", "tracker setup must not mutate caller variables");
    assert.equal(windowsCoreTrackerEnvironment({WINDIR: systemRoot})["PSModulePath"], tracker["PSModulePath"]);
    assert.equal(Object.hasOwn(windowsCoreTrackerEnvironment({...environment, SystemRoot: "relative"}), "PSModulePath"), false);
  });

  test("tracker row projection sorts hashtable values explicitly", () => {
    const script = windowsCoreTrackerRowsScript();
    assert.equal((script.match(/Sort-Object -Property \{ \[string\] \$_\['Path'\] \}/g) || []).length, 3);
    assert.match(script, /\{ \[string\] \$_\['Hash'\] \} -Unique/);
    assert.doesNotMatch(script, /Sort-Object Path -Unique/);
  });

  test("PowerShell 5 keeps every distinct tracked input, output, and toolchain row", {skip: process.platform !== "win32"}, () => {
    const systemRoot = process.env["SystemRoot"] || process.env["WINDIR"];
    assert.ok(systemRoot, "Windows must expose SystemRoot for the PowerShell fixture");
    const powershell = path.join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
    const script = ["$ErrorActionPreference = 'Stop'", "$ProgressPreference = 'SilentlyContinue'",
      "$OutputEncoding = New-Object System.Text.UTF8Encoding($false)", "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)",
      "$hash = Get-FileHash -LiteralPath (Join-Path $PSHOME 'powershell.exe') -Algorithm SHA256; if ($hash.Hash.Length -ne 64) { throw 'Get-FileHash is unavailable.' }",
      "$prefix = 'C:\\repo\\'",
      "$outputs = @(@{ Path = 'C:\\build\\a.obj'; Hash = 'out-a'; Inputs = @(@{ Path = 'C:\\repo\\a.cpp'; Hash = 'src-a' }, @{ Path = 'C:\\sdk\\shared.h'; Hash = 'hdr' }, @{ Path = 'C:\\sdk\\core.lib'; Hash = 'lib' }) }, @{ Path = 'C:\\build\\b.obj'; Hash = 'out-b'; Inputs = @(@{ Path = 'C:\\repo\\b.cpp'; Hash = 'src-b' }, @{ Path = 'C:\\sdk\\shared.h'; Hash = 'hdr' }) })",
      "$script:MomBuildSnapshot = @(@{ Path = 'C:\\tools\\cl.exe'; Hash = 'tool-a' }, @{ Path = 'C:\\tools\\lib.exe'; Hash = 'tool-b' }, @{ Path = 'C:\\repo\\native\\core.cpp'; Hash = 'repo-src' })",
      windowsCoreTrackerRowsScript(),
      "$snapshot = @{ Inputs = $inputs; Artifacts = $artifacts; Toolchain = $toolchain }",
      "[Console]::Out.WriteLine((ConvertTo-Json -InputObject $snapshot -Compress -Depth 8))"].join("\n");
    const result = spawnSync(powershell, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
      {encoding: "utf8", timeout: 10000, windowsHide: true, env: windowsCoreTrackerEnvironment(process.env)});
    assert.equal(result.status, 0, String(result.stderr || result.error?.message || "PowerShell fixture failed"));
    const snapshot = /** @type {{Inputs: Array<{Path: string}>, Artifacts: Array<{Path: string}>, Toolchain: Array<{Path: string}>}} */ (JSON.parse(result.stdout || "{}"));
    assert.equal(snapshot.Inputs.length, 4);
    assert.equal(snapshot.Artifacts.length, 2);
    assert.equal(snapshot.Toolchain.length, 2);
    assert.ok(snapshot.Inputs.some(row => row.Path === "C:\\repo\\b.cpp"));
    assert.ok(snapshot.Inputs.some(row => row.Path === "C:\\sdk\\core.lib"), "actual TLog libraries are not filtered from tracker inputs");
    assert.ok(snapshot.Artifacts.some(row => row.Path === "C:\\build\\b.obj"));
    assert.ok(snapshot.Toolchain.every(row => !row.Path.startsWith("C:\\repo\\")));
  });

  test("permits changed generated objects but rejects changed source, SDK header, and TLog library inputs", () => {
    const source = path.join(os.tmpdir(), "core.cpp").toLowerCase(), header = path.join(os.tmpdir(), "windows.h").toLowerCase();
    const library = path.join(os.tmpdir(), "core.lib").toLowerCase(), generatedBefore = path.join(os.tmpdir(), "core.obj").toLowerCase();
    const generatedAfter = path.join(os.tmpdir(), "new-core.obj").toLowerCase();
    const before = {inputs: {[source]: "source-a", [header]: "header-a", [library]: "library-a", [generatedBefore]: "obj-a",
      [generatedAfter]: "missing-before"}, artifacts: {[generatedBefore]: "obj-a"}};
    const after = {inputs: {[source]: "source-a", [header]: "header-a", [library]: "library-a", [generatedBefore]: "obj-b",
      [generatedAfter]: "obj-new"}, artifacts: {[generatedBefore]: "obj-b", [generatedAfter]: "obj-new"}};
    assert.doesNotThrow(() => verifyWindowsCoreTrackerInputs(before, after));
    for (const file of [source, header, library]) {
      const changed = structuredClone(after); changed.inputs[file] = "changed-during-build";
      assert.throws(() => verifyWindowsCoreTrackerInputs(before, changed), /compiler dependency changed during compilation/);
    }
  });

  test("allows only tracked newer source edits to use MSBuild Build; other drift and removed sources Rebuild", () => {
    const repo = path.resolve(os.tmpdir(), "mom-win-core-repo").toLowerCase(), key = path.join(repo, "native", "core.cpp");
    const current = {
      repo, staticIdentity: {sourceMembership: ["native/core.cpp"], recipe: "fixed"}, incrementalSafe: true,
      projectSha256: "project-hash", sourceState: {files: {"native/core.cpp": "source-a", "native/core.h": "header-a"}, graph: {}},
      sourceMtimes: {"native/core.cpp": 300, "native/core.h": 10},
      tracker: {complete: true, artifacts: {[path.join(repo, "build", "Release", "core.obj")]: "obj-hash"}, toolchain: {[path.join(repo, "tools", "cl.exe")]: "tool-hash"},
        inputs: {[key]: "source-a", [path.join(repo, "native", "core.h")]: "header-a"}},
    };
    const state = {...current, version: 1, lastSuccessMs: 200, archiveSha256: "archive-hash", sourceMtimes: {...current.sourceMtimes},
      sourceState: {files: {...current.sourceState.files}, graph: {}}, tracker: {...current.tracker, artifacts: {...current.tracker.artifacts},
        toolchain: {...current.tracker.toolchain}, inputs: {...current.tracker.inputs}}};
    const decide = (next = current, extra = {}) => windowsCoreCacheDecision({state, current: next, marker: false, projectExists: true, archiveValid: true, ...extra});
    assert.equal(decide().target, "Build");

    const edited = structuredClone(current);
    edited.sourceState.files["native/core.cpp"] = "source-b";
    edited.sourceMtimes["native/core.cpp"] = 400;
    edited.tracker.inputs[key] = "source-b";
    assert.deepEqual(decide(edited), {configure: false, target: "Build", reason: "tracked source-only edit"});

    const backdated = structuredClone(edited); backdated.sourceMtimes["native/core.cpp"] = 150;
    assert.equal(decide(backdated).target, "Rebuild");
    const headerEdit = structuredClone(current);
    headerEdit.sourceState.files["native/core.h"] = "header-b";
    headerEdit.tracker.inputs[path.join(repo, "native", "core.h")] = "header-b";
    assert.equal(decide(headerEdit).target, "Rebuild");
    const sourceRemoval = structuredClone(current);
    sourceRemoval.staticIdentity.sourceMembership = [];
    sourceRemoval.sourceState.files = /** @type {typeof current.sourceState.files} */ ({"native/core.h": "header-a"});
    assert.equal(decide(sourceRemoval).target, "Rebuild", "removed-source membership rebuild clears stale Release objects");
    assert.equal(decide(current, {marker: true}).configure, true);
    assert.equal(decide(current, {forceRebuild: true}).target, "Rebuild");
    assert.equal(decide(current, {archiveValid: false}).target, "Rebuild");
  });

  test("Rebuild cleanup removes only validated Release outputs and preserves the rest of the graph", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-core-release-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const build = path.join(root, "source", "build"), release = path.join(build, "Release"), project = path.join(build, "mom_native_core.vcxproj");
    fs.mkdirSync(path.join(release, "obj"), {recursive: true}); fs.mkdirSync(path.join(build, "Debug"), {recursive: true});
    fs.writeFileSync(path.join(release, "obj", "orphan.obj"), "orphan");
    fs.writeFileSync(path.join(release, "outputs.write.1.tlog"), "tracker");
    fs.writeFileSync(project, "project"); fs.writeFileSync(path.join(build, "Debug", "keep.obj"), "debug");
    clearWindowsCoreRelease(build);
    assert.equal(fs.existsSync(release), false);
    assert.equal(fs.readFileSync(project, "utf8"), "project");
    assert.equal(fs.readFileSync(path.join(build, "Debug", "keep.obj"), "utf8"), "debug");

  });

  test("Rebuild cleanup refuses escaping links", {skip: process.platform === "win32"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-core-release-link-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const build = path.join(root, "source", "build"), release = path.join(build, "Release");
    fs.mkdirSync(release, {recursive: true});
    const outside = path.join(root, "outside"); fs.mkdirSync(outside);
    const sentinel = path.join(outside, "keep.txt"); fs.writeFileSync(sentinel, "keep");
    fs.symlinkSync(outside, path.join(release, "escape"), "dir");
    assert.throws(() => clearWindowsCoreRelease(build), /unsafe path/);
    assert.equal(fs.readFileSync(sentinel, "utf8"), "keep");
  });

  test("read-only current-cache admission requires an unchanged tracked Build state", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-win-core-admission-")), tools = path.join(root, "tools"),
      dumpbinDirectory = path.join(tools, "MSVC", "14.44", "bin", "HostX64", "x64"), include = path.join(tools, "MSVC", "14.44", "include"),
      libuvInclude = path.join(root, "libuv"), cacheRoot = path.join(root, "cache", "native-windows-core", "v1"), prospective = path.join(root, "prospective"),
      descriptor = path.join(root, "descriptor.json"), currentEnv = {PATH: process.env["PATH"] || ""};
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.mkdirSync(dumpbinDirectory, {recursive: true}); fs.mkdirSync(include, {recursive: true});
    fs.mkdirSync(path.join(libuvInclude, "uv"), {recursive: true}); fs.mkdirSync(path.join(cacheRoot, "source", "build", "Release"), {recursive: true});
    for (const file of ["dumpbin.exe", "cl.exe", "lib.exe", "ml64.exe"]) {fs.writeFileSync(path.join(dumpbinDirectory, file), file);}
    for (const file of [path.join(root, "node-gyp.js"), path.join(tools, "msbuild.exe"), path.join(tools, "python.exe")]) {fs.writeFileSync(file, "tool");}
    fs.writeFileSync(path.join(libuvInclude, "uv.h"), "uv\n"); fs.writeFileSync(path.join(libuvInclude, "uv", "win.h"), "win\n");
    const config = {version: 1, nodeGyp: path.join(root, "node-gyp.js"), msbuild: path.join(tools, "msbuild.exe"),
      dumpbin: path.join(dumpbinDirectory, "dumpbin.exe"), python: path.join(tools, "python.exe"), libuvInclude};
    fs.writeFileSync(descriptor, JSON.stringify(config));
    const archive = path.join(cacheRoot, "source", "build", "Release", "mom_native_core.lib"), project = path.join(cacheRoot, "source", "build", "mom_native_core.vcxproj");
    fs.writeFileSync(archive, "!<arch>\n1"); fs.writeFileSync(project, "project\n");
    const sdkHeader = path.join(include, "windows.h"); fs.writeFileSync(sdkHeader, "Windows SDK header\n");
    const tracker = {complete: true, membership: "membership", inputs: {[sdkHeader.toLowerCase()]: fileHash(sdkHeader)}, artifacts: {}, toolchain: {}, powershellSha256: "a".repeat(64)},
      includeState = {complete: true, roots: [include, libuvInclude], sha256: "b".repeat(64)},
      current = {repo: path.resolve(__dirname, ".."), staticIdentity: {fixture: "identity"}, projectSha256: "project-hash",
        archiveSha256: fileHash(archive), sourceState: {files: {}, graph: {}}, sourceMtimes: {}, tracker, incrementalSafe: true},
      statePath = path.join(cacheRoot, "core-state.json"), state = {...structuredClone(current), version: 1, lastSuccessMs: Date.now()};
    let includeCalls = 0, trackerCalls = 0, viewCalls = 0;
    const hooks = {root: path.resolve(__dirname, ".."), cacheRoot, includeRoots() {includeCalls++; return includeState;},
      trackerSnapshot() {trackerCalls++; return tracker;}, cacheView() {viewCalls++; return current;}};
    assert.equal(windowsCoreCurrentCacheAdmission(descriptor, prospective, currentEnv, hooks).reusable, false,
      "an existing cache directory without admitted state remains on the producer's fresh path");
    assert.equal(viewCalls, 0, "missing state does not inspect or mutate the cache view");
    fs.writeFileSync(statePath, JSON.stringify(state));
    const stateBytes = fs.readFileSync(statePath);
    const admitted = windowsCoreCurrentCacheAdmission(descriptor, prospective, currentEnv, hooks);
    assert.deepEqual(admitted, {reusable: true, reason: "inputs unchanged", descriptorSha256: fileHash(descriptor),
      stateSha256: fileHash(statePath), archiveSha256: fileHash(archive)});
    assert.equal(fs.existsSync(prospective), false, "admission must not create an output directory");
    assert.deepEqual(fs.readFileSync(statePath), stateBytes, "admission must not rewrite the producer's cache state");
    assert.deepEqual([includeCalls, trackerCalls, viewCalls], [1, 1, 1], "admission delegates to the selected include, tracker, and cache-view readers");

    const markerPath = path.join(cacheRoot, "build-in-progress.json"); fs.writeFileSync(markerPath, "{}\n");
    assert.equal(windowsCoreCurrentCacheAdmission(descriptor, prospective, currentEnv, hooks).reusable, false,
      "a producer in-progress marker prevents reuse");
    fs.unlinkSync(markerPath);
    const lockPath = path.join(cacheRoot, "build.lock"); fs.writeFileSync(lockPath, "{}\n");
    assert.equal(windowsCoreCurrentCacheAdmission(descriptor, prospective, currentEnv, hooks).reusable, false,
      "a cache lock prevents reuse");
    fs.unlinkSync(lockPath);

    tracker.complete = false;
    const incomplete = windowsCoreCurrentCacheAdmission(descriptor, prospective, currentEnv, hooks);
    assert.equal(incomplete.reusable, false, "an incomplete external-header tracker cannot authorize reuse");
    tracker.complete = true;
    fs.writeFileSync(sdkHeader, "Changed Windows SDK header\n"); tracker.inputs[sdkHeader.toLowerCase()] = fileHash(sdkHeader);
    assert.equal(windowsCoreCurrentCacheAdmission(descriptor, prospective, currentEnv, hooks).reusable, false,
      "an external Windows SDK header hash change cannot authorize reuse");
    fs.writeFileSync(sdkHeader, "Windows SDK header\n"); tracker.inputs[sdkHeader.toLowerCase()] = fileHash(sdkHeader);
    const originalHash = current.archiveSha256; current.archiveSha256 = "c".repeat(64);
    assert.equal(windowsCoreCurrentCacheAdmission(descriptor, prospective, currentEnv, hooks).reusable, false,
      "a cache archive that no longer matches admitted state cannot authorize reuse");
    current.archiveSha256 = originalHash;
    const changed = windowsCoreCurrentCacheAdmission(descriptor, prospective, {...currentEnv, MOM_FORCE_REBUILD: "1"}, hooks);
    assert.equal(changed.reusable, false, "explicit cold-build controls bypass warm admission");

    fs.writeFileSync(statePath, stateBytes);
    hooks.cacheView = () => {fs.appendFileSync(statePath, " "); return current;};
    assert.equal(windowsCoreCurrentCacheAdmission(descriptor, prospective, currentEnv, hooks).reusable, false,
      "a cache-state replacement during admission cannot authorize reuse");
  });
});
