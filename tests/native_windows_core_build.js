"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {fileHash} = require("../scripts/build-source-inputs");
const {validateWindowsCoreDescriptor, intelCoreCompilerVersion, validatePortableEnvironment, sanitizeBuildEnvironment, windowsCorePlan, runTool, parseDumpbinSymbols, parseDumpbinArchive, inspectCoreProject, stageSourceTree, cleanupCoreSourceJunctions, buildNativeWindowsCore} = require("../scripts/build-native-windows-core");

const root = path.resolve(__dirname, "..");

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-core-build-"));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const nodeGyp = path.join(directory, "node-gyp.js"), msbuild = path.join(directory, "MSBuild.exe"), python = path.join(directory, "python.exe");
  const msvcDirectory = path.join(directory, "MSVC", "14.44.35207", "bin", "HostX64", "x64"), dumpbin = path.join(msvcDirectory, "dumpbin.exe");
  fs.mkdirSync(msvcDirectory, {recursive: true});
  const cl = path.join(msvcDirectory, "cl.exe"), lib = path.join(msvcDirectory, "lib.exe"), ml64 = path.join(msvcDirectory, "ml64.exe");
  for (const file of [nodeGyp, msbuild, dumpbin, cl, lib, ml64, python]) {fs.writeFileSync(file, "tool\n", {flag: "wx"});}
  const libuvInclude = path.join(directory, "libuv", "include");
  fs.mkdirSync(path.join(libuvInclude, "uv"), {recursive: true});
  fs.writeFileSync(path.join(libuvInclude, "uv.h"), "#pragma once\n", {flag: "wx"});
  fs.writeFileSync(path.join(libuvInclude, "uv", "win.h"), "#pragma once\n", {flag: "wx"});
  for (const name of ["errno.h", "tree.h", "threadpool.h", "version.h"]) {
    fs.writeFileSync(path.join(libuvInclude, "uv", name), "#pragma once\n", {flag: "wx"});
  }
  const config = {version: 1, nodeGyp, msbuild, dumpbin, python, libuvInclude}, descriptor = path.join(directory, "windows-core.json"), output = path.join(directory, "output");
  fs.writeFileSync(descriptor, JSON.stringify(config) + "\n", {flag: "wx"});
  return {directory, config, descriptor, output, nodeGyp, msbuild, dumpbin, cl, lib, ml64, python, libuvInclude};
}

test.describe("Windows Core producer admission and plan", () => {
  test("admits only the explicit v1 tool/header descriptor", t => {
    const f = fixture(t);
    assert.deepEqual(validateWindowsCoreDescriptor(f.config), f.config);
    const {python: _python, ...withoutPython} = f.config;
    assert.throws(() => validateWindowsCoreDescriptor(withoutPython), /python|descriptor/);
    for (const invalid of [
      {...f.config, version: 2},
      {...f.config, unknown: f.nodeGyp},
      {...f.config, nodeGyp: path.join(f.directory, "node-gyp.cmd")},
      {...f.config, msbuild: f.nodeGyp},
      {...f.config, dumpbin: f.nodeGyp},
      {...f.config, python: "relative-python.exe"},
      {...f.config, libuvInclude: path.join(f.directory, "missing")},
    ]) {assert.throws(() => validateWindowsCoreDescriptor(invalid), /descriptor|nodeGyp|msbuild|dumpbin|python|libuv/);}
    fs.unlinkSync(path.join(f.libuvInclude, "uv", "win.h"));
    assert.throws(() => validateWindowsCoreDescriptor(f.config), /win\.h/);
  });

  test("plans literal private configure and Core-only MSBuild argv", t => {
    const f = fixture(t), plan = windowsCorePlan(f.config, f.output, root);
    assert.equal(plan.configure.command, process.execPath);
    assert.deepEqual(plan.configure.args.slice(0, 3), [f.nodeGyp, "configure", `--devdir=${plan.paths.devdir}`]);
    assert.ok(plan.configure.args.includes(`--python=${f.python}`));
    assert.ok(plan.configure.args.includes("--msvs-version"));
    assert.ok(plan.configure.args.includes("2022"));
    for (const flag of ["-Dmom_native_core=1", `-Dmom_native_uv_include=${f.libuvInclude}`, "-Dtarget_arch=x64", "-Dlibrary=shared_library"]) {
      assert.ok(plan.configure.args.includes(flag), flag);
    }
    assert.equal(plan.build.command, f.msbuild);
    assert.equal(plan.build.args[0], plan.paths.project);
    for (const flag of ["/m:2", "/p:BuildProjectReferences=false", "/p:Configuration=Release", "/p:Platform=x64", "/p:VCToolsVersion=14.44.35207",
      `/p:DirectoryBuildPropsPath=${path.join(plan.paths.msbuildUserExtensions, "Directory.Build.props")}`,
      `/p:DirectoryBuildTargetsPath=${path.join(plan.paths.msbuildUserExtensions, "Directory.Build.targets")}`,
      `/p:MSBuildUserExtensionsPath=${plan.paths.msbuildUserExtensions}`]) {
      assert.ok(plan.build.args.includes(flag), flag);
    }
    assert.ok(plan.build.args.every(arg => !/\.sln$/i.test(arg)));
    assert.ok(plan.sourceInventory.includes("native/core.cpp"));
    assert.ok(plan.sourceInventory.includes("sycl/lib.h"));
    assert.ok(plan.sourceInventory.includes("xmrig/crypto/randomx/randomx.cpp"));
    assert.match(plan.paths.archive, /[\\/]source[\\/]build[\\/]Release[\\/]mom_native_core\.lib$/);
    assert.equal(path.dirname(plan.paths.exportedArchive), plan.output);
    assert.equal(Object.keys(plan.headerHashes).length, 6);
    assert.ok(Object.keys(plan.headerHashes).some(file => file.endsWith(path.join("uv", "version.h"))));
    for (const tool of [f.cl, f.lib, f.ml64]) {assert.ok(plan.toolHashes[tool], tool);}
    assert.equal(plan.config.version, 1);
  });

  test("rejects an existing or source-owned output before mutation", t => {
    const f = fixture(t);
    assert.throws(() => windowsCorePlan(f.config, path.join(root, ".windows-core-output"), root), /outside/);
    fs.mkdirSync(f.output);
    assert.throws(() => windowsCorePlan(f.config, f.output, root), /new absolute/);
    const alias = path.join(f.directory, "repo-alias"); fs.symlinkSync(root, alias, process.platform === "win32" ? "junction" : "dir");
    assert.throws(() => windowsCorePlan(f.config, path.join(alias, "output"), root), /outside/);
  });

  test("stages only private junctions and a binding copy", t => {
    const f = fixture(t), plan = windowsCorePlan(f.config, f.output, root), staged = stageSourceTree(plan);
    for (const name of ["native", "sycl", "xmrig"]) {
      assert.equal(fs.realpathSync(path.join(plan.paths.source, name)), fs.realpathSync(path.join(root, name)));
      assert.equal(staged.junctions[name], path.join(root, name));
    }
    assert.equal(fileHash(plan.paths.binding), fileHash(path.join(root, "binding.gyp")));
    cleanupCoreSourceJunctions(plan, staged);
  });

  test("rolls back known junctions when private staging fails after link creation", t => {
    const f = fixture(t), plan = windowsCorePlan(f.config, f.output, root);
    fs.mkdirSync(plan.paths.source, {recursive: true}); fs.mkdirSync(plan.paths.binding);
    assert.throws(() => stageSourceTree(plan), /staged binding\.gyp is unsafe/);
    for (const name of ["native", "sycl", "xmrig"]) {assert.equal(fs.existsSync(path.join(plan.paths.source, name)), false);}
    assert.equal(fs.statSync(plan.paths.binding).isDirectory(), true);
  });

  test("recovers exact restored trees into canonical junctions and preserves cache outputs", t => {
    const f = fixture(t), plan = windowsCorePlan(f.config, f.output, root);
    const sourceFile = path.join(root, "native", "core.cpp"), sourceHash = fileHash(sourceFile), sourceMtime = fs.statSync(sourceFile).mtimeMs;
    fs.mkdirSync(plan.paths.source, {recursive: true});
    for (const name of ["native", "sycl", "xmrig"]) {
      fs.cpSync(path.join(root, name), path.join(plan.paths.source, name), {recursive: true, preserveTimestamps: true});
    }
    const staged = stageSourceTree(plan);
    assert.deepEqual(staged.recoveredCopies, ["native", "sycl", "xmrig"]);
    for (const name of ["native", "sycl", "xmrig"]) {
      assert.equal(fs.lstatSync(path.join(plan.paths.source, name)).isSymbolicLink(), true);
      assert.equal(fs.realpathSync(path.join(plan.paths.source, name)), fs.realpathSync(path.join(root, name)));
    }
    const marker = path.join(plan.paths.build, "Release", "retained.obj");
    fs.mkdirSync(path.dirname(marker), {recursive: true}); fs.writeFileSync(marker, "cached object\n", {flag: "wx"});
    cleanupCoreSourceJunctions(plan, staged);
    for (const name of ["native", "sycl", "xmrig"]) {assert.equal(fs.existsSync(path.join(plan.paths.source, name)), false);}
    assert.equal(fs.readFileSync(marker, "utf8"), "cached object\n");
    assert.equal(fileHash(sourceFile), sourceHash); assert.equal(fs.statSync(sourceFile).mtimeMs, sourceMtime);
    const restaged = stageSourceTree(plan);
    assert.deepEqual(restaged.recoveredCopies, []);
    for (const name of ["native", "sycl", "xmrig"]) {assert.equal(fs.realpathSync(path.join(plan.paths.source, name)), fs.realpathSync(path.join(root, name)));}
    cleanupCoreSourceJunctions(plan, restaged);
    assert.equal(fs.readFileSync(marker, "utf8"), "cached object\n");
  });

  test("rejects changed and extra restored files without deleting them", t => {
    for (const mutation of ["changed", "extra"]) {
      const f = fixture(t), plan = windowsCorePlan(f.config, f.output, root), restored = path.join(plan.paths.source, "sycl");
      fs.mkdirSync(plan.paths.source, {recursive: true});
      fs.cpSync(path.join(root, "sycl"), restored, {recursive: true, preserveTimestamps: true});
      const file = mutation === "changed" ? path.join(restored, "lib.h") : path.join(restored, "unexpected.txt");
      if (mutation === "changed") {fs.appendFileSync(file, "// unexpected\n");} else {fs.writeFileSync(file, "foreign cache data\n", {flag: "wx"});}
      const observed = fileHash(file);
      assert.throws(() => stageSourceTree(plan), /differs from maintained source/);
      assert.equal(fileHash(file), observed);
    }
  });

  test("rejects restored links and hardlinks without following or deleting their targets", t => {
    const f = fixture(t), plan = windowsCorePlan(f.config, f.output, root), restored = path.join(plan.paths.source, "sycl");
    fs.mkdirSync(plan.paths.source, {recursive: true});
    fs.cpSync(path.join(root, "sycl"), restored, {recursive: true});
    const outside = path.join(f.directory, "outside"); fs.mkdirSync(outside);
    fs.writeFileSync(path.join(outside, "keep.txt"), "keep outside\n", {flag: "wx"});
    fs.symlinkSync(outside, path.join(restored, "redirect"), process.platform === "win32" ? "junction" : "dir");
    assert.throws(() => stageSourceTree(plan), /link/);
    assert.equal(fs.readFileSync(path.join(outside, "keep.txt"), "utf8"), "keep outside\n");
    if (process.platform === "win32") {fs.rmdirSync(path.join(restored, "redirect"));} else {fs.unlinkSync(path.join(restored, "redirect"));}
    const sourceFile = path.join(root, "sycl", "lib.h"), linkedFile = path.join(restored, "lib.h"), duplicate = path.join(f.directory, "duplicate.h");
    fs.copyFileSync(sourceFile, duplicate); fs.unlinkSync(linkedFile); fs.linkSync(duplicate, linkedFile);
    assert.throws(() => stageSourceTree(plan), /hard-linked/);
    assert.equal(fs.readFileSync(sourceFile, "utf8"), fs.readFileSync(linkedFile, "utf8"));
    fs.unlinkSync(linkedFile);
  });

  test("refuses wrong-target cleanup before removing any staged junction", t => {
    const f = fixture(t), plan = windowsCorePlan(f.config, f.output, root), staged = stageSourceTree(plan);
    const outside = path.join(f.directory, "outside"); fs.mkdirSync(outside); fs.writeFileSync(path.join(outside, "keep.txt"), "keep\n", {flag: "wx"});
    const redirected = path.join(plan.paths.source, "native");
    if (process.platform === "win32") {fs.rmdirSync(redirected);} else {fs.unlinkSync(redirected);}
    fs.symlinkSync(outside, redirected, process.platform === "win32" ? "junction" : "dir");
    assert.throws(() => cleanupCoreSourceJunctions(plan, staged), /refusing to remove a changed source junction/);
    assert.equal(fs.existsSync(path.join(plan.paths.source, "sycl")), true);
    assert.equal(fs.existsSync(path.join(plan.paths.source, "xmrig")), true);
    assert.equal(fs.readFileSync(path.join(outside, "keep.txt"), "utf8"), "keep\n");
  });
});

test.describe("Windows Core explicit Intel compiler selection", () => {
  /** @param {import("node:test").TestContext} t */
  const intelFixture = t => {
    const f = fixture(t), compiler = path.join(f.directory, "Intel Compiler", "icx-cl.exe");
    fs.mkdirSync(path.dirname(compiler)); fs.writeFileSync(compiler, "Intel compiler fixture\n", {flag: "wx"});
    return {...f, config: {...f.config, compiler}, compiler};
  };
  test("admits an explicit Intel frontend without changing the GYP or assembly graph", t => {
    const f = intelFixture(t), {compiler: _compiler, ...plain} = f.config, baseline = windowsCorePlan(plain, f.output, root);
    const plan = windowsCorePlan(f.config, f.output, root);
    assert.equal(validateWindowsCoreDescriptor(f.config).compiler, f.compiler);
    assert.deepEqual(plan.configure.args.filter(arg => arg !== "-Dmom_native_intel=1"), baseline.configure.args);
    assert.ok(plan.configure.args.includes("-Dmom_native_intel=1"));
    assert.deepEqual(plan.sourceInventory, baseline.sourceInventory);
    assert.equal(plan.toolHashes[f.compiler], fileHash(f.compiler));
    assert.ok(plan.toolHashes[f.ml64]); assert.ok(plan.toolHashes[f.lib]);
  });
  test("pins the MSBuild frontend using literal argv, including a path containing spaces", t => {
    const f = intelFixture(t), plan = windowsCorePlan(f.config, f.output, root);
    assert.ok(plan.build.args.includes("/p:CLToolExe=icx-cl.exe"));
    assert.ok(plan.build.args.includes(`/p:CLToolPath=${path.dirname(f.compiler)}${path.sep}`));
    assert.ok(plan.build.args.includes("/p:UseMultiToolTask=false"));
    assert.ok(plan.compiler); assert.equal(plan.compiler.command, f.compiler); assert.deepEqual(plan.compiler.args, ["--version"]);
  });
  test("does not inject Intel overrides into the existing MSVC baseline", t => {
    const f = fixture(t), plan = windowsCorePlan(f.config, f.output, root);
    assert.equal(plan.compiler, null); assert.ok(!plan.build.args.some(arg => /CLTool|UseMultiToolTask/.test(arg)));
    assert.ok(!plan.configure.args.some(arg => arg.startsWith("-Dmom_native_intel=")));
  });
  test("rejects missing, unrelated or MSBuild-interpreted compiler paths", t => {
    const f = intelFixture(t);
    for (const compiler of ["icx-cl.exe", path.join(f.directory, "missing", "icx-cl.exe"), f.cl, null, ""]) {
      assert.throws(() => validateWindowsCoreDescriptor({...f.config, compiler}), /compiler|Intel/);
    }
    if (process.platform !== "win32") {
      for (const name of ["semi;colon", "percent%3B", "macro$(extra)"]) {
        const compiler = path.join(f.directory, name, "icx-cl.exe"); fs.mkdirSync(path.dirname(compiler)); fs.writeFileSync(compiler, "fixture");
        assert.throws(() => validateWindowsCoreDescriptor({...f.config, compiler}), /compiler|Intel/);
      }
    }
  });
  test("removes inherited frontend overrides without mutating the caller environment", t => {
    const f = fixture(t), env = {CLToolExe: "other.exe", cltoolpath: "bad", UseMultiToolTask: "true", Keep: "yes"};
    const result = sanitizeBuildEnvironment(env, f.python);
    assert.equal(result["CLToolExe"], undefined); assert.equal(result["cltoolpath"], undefined); assert.equal(result["UseMultiToolTask"], undefined);
    assert.equal(result["Keep"], "yes"); assert.equal(env.CLToolExe, "other.exe");
  });
  test("Intel Core disables proprietary memory helpers without disabling language builtins", t => {
    const f = fixture(t), env = {cl: "", Keep: "yes"};
    const intel = sanitizeBuildEnvironment(env, f.python, true);
    assert.equal(intel["CL"], "/Q_no-use-libirc");
    assert.equal(intel["cl"], undefined);
    assert.equal(intel["Keep"], "yes");
    assert.equal(sanitizeBuildEnvironment({}, f.python)["CL"], undefined);
    assert.deepEqual(env, {cl: "", Keep: "yes"});
  });
  test("requires the real Intel Windows x64 identity rather than trusting the filename", () => {
    const version = "Intel(R) oneAPI DPC++/C++ Compiler 2026.2.0 (2026.2.0.20261001)\r\nTarget: x86_64-pc-windows-msvc\r\n";
    assert.deepEqual(intelCoreCompilerVersion(version), {version: "2026.2.0", target: "x86_64-pc-windows-msvc"});
    for (const invalid of ["", "icx-cl.exe", version.replace("Intel(R)", "Clang"), version.replace("x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu")]) {
      assert.throws(() => intelCoreCompilerVersion(invalid), /Intel Windows x64/);
    }
  });
});

test.describe("Windows Core bounded tool and project gates", () => {
  test("passes literal argv/cwd and bounded timeout to the child", t => {
    const f = fixture(t), log = path.join(f.directory, "tool.log");
    /** @type {Array<{command: string, args: readonly string[], options: import("node:child_process").SpawnSyncOptions}>} */
    const calls = [];
    /** @param {string} command @param {readonly string[]} args @param {import("node:child_process").SpawnSyncOptions} options */
    const mockSpawn = (command, args, options) => {calls.push({command, args, options}); return {status: 0, signal: null, error: undefined};};
    runTool(f.msbuild, ["project.vcxproj", "/m:2"], f.directory, log, mockSpawn, {MOM_AOT_BUILD_ONLY: "1"}, 1234);
    assert.equal(calls.length, 1);
    const first = calls[0]; assert.ok(first);
    assert.deepEqual(first.args, ["project.vcxproj", "/m:2"]);
    assert.equal(first.options.cwd, f.directory);
    assert.equal(first.options.timeout, 1234);
    assert.ok(first.options.stdio);
    assert.deepEqual(first.options.stdio.slice(0, 2), ["ignore", first.options.stdio[1]]);
    assert.ok(fs.statSync(log).isFile());
  });

  test("rejects Node/V8/NAPI/SYCL project dependencies before MSBuild", t => {
    const f = fixture(t), project = path.join(f.directory, "mom_native_core.vcxproj");
    const valid = `<Project><ConfigurationType>StaticLibrary</ConfigurationType><PlatformToolset>v143</PlatformToolset><RuntimeLibrary>MultiThreaded</RuntimeLibrary><ExceptionHandling>Sync</ExceptionHandling><LanguageStandard>stdcpp20</LanguageStandard><WholeProgramOptimization>false</WholeProgramOptimization><LinkTimeCodeGeneration>false</LinkTimeCodeGeneration><PreprocessorDefinitions>MOM_NATIVE_RUNTIME</PreprocessorDefinitions><AdditionalIncludeDirectories>${f.libuvInclude};%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories><ClCompile Include="native/core.cpp"/><MASM Include="native/core.asm"/></Project>`;
    fs.writeFileSync(project, valid, {flag: "wx"});
    assert.equal(inspectCoreProject(project, undefined, f.libuvInclude).bytes, Buffer.byteLength(valid));
    for (const forbidden of ["<ProjectReference Include=\"sycl.vcxproj\" />", "node.lib", "NAPI_VERSION", "v8.lib", "_HAS_EXCEPTIONS=0", "<WholeProgramOptimization>true</WholeProgramOptimization>", "/LTCG"]) {
      const bad = path.join(f.directory, "bad-" + Buffer.byteLength(forbidden) + ".vcxproj");
      fs.writeFileSync(bad, valid.replace("</Project>", forbidden + "</Project>"), {flag: "wx"});
      assert.throws(() => inspectCoreProject(bad), /Node\/V8\/NAPI\/SYCL|unexpected WholeProgramOptimization/);
    }
    const implicitNode = path.join(f.directory, "node", "include", "node");
    const badInclude = path.join(f.directory, "bad-implicit-node.vcxproj");
    fs.writeFileSync(badInclude, valid.replace("</AdditionalIncludeDirectories>", `;${implicitNode}</AdditionalIncludeDirectories>`), {flag: "wx"});
    assert.throws(() => inspectCoreProject(badInclude, undefined, f.libuvInclude), /implicit Node/);
    const missingCxx20 = path.join(f.directory, "bad-no-cxx20.vcxproj");
    fs.writeFileSync(missingCxx20, valid.replace(/<LanguageStandard>.*?<\/LanguageStandard>/, ""), {flag: "wx"});
    assert.throws(() => inspectCoreProject(missingCxx20), /LanguageStandard/);
  });

  test("rejects inherited compiler/linker tuning before output or tools", t => {
    const f = fixture(t);
    for (const key of ["CL", "_CL_", "LINK", "_LINK_", "MOM_CPU_MARCH", "cl", "_cl_", "link", "_link_", "mom_cpu_march",
      "ICXCFG", "__INTEL_PRE_CFLAGS", "__INTEL_POST_CFLAGS", "icxcfg", "__intel_pre_cflags", "__intel_post_cflags"]) {
      assert.throws(() => validatePortableEnvironment({[key]: "unexpected"}), new RegExp(`portable.*${key}`));
    }
    assert.doesNotThrow(() => validatePortableEnvironment({CL: "", _CL_: "", LINK: "", _LINK_: "", MOM_CPU_MARCH: ""}));
    const sanitized = sanitizeBuildEnvironment({GYP_CONFIG_DIR: "bad", gyp_generator_output: "bad", Node_Options: "--require bad", node_path: "bad",
      NODE_GYP_FORCE_PYTHON: "bad", pythonhome: "bad", PYTHONPATH: "bad", npm_config_python: "bad", DirectoryBuildPropsPath: "bad", Keep: "yes"}, f.python);
    for (const key of ["GYP_CONFIG_DIR", "gyp_generator_output", "Node_Options", "node_path", "NODE_GYP_FORCE_PYTHON", "pythonhome", "PYTHONPATH",
      "npm_config_python", "DirectoryBuildPropsPath"]) {assert.equal(Object.keys(sanitized).some(name => name.toLowerCase() === key.toLowerCase()), false, key);}
    assert.equal(sanitized["PYTHON"], f.python);
    assert.equal(sanitized["Keep"], "yes");
    assert.equal(fs.existsSync(f.output), false);
  });

  test("guards Windows build-only entry before descriptor/output/tool access", t => {
    const f = fixture(t);
    /** @type {unknown[][]} */
    const calls = [];
    /** @param {string} _command @param {readonly string[]} _args @param {import("node:child_process").SpawnSyncOptions} _options */
    const mockSpawn = (_command, _args, _options) => {calls.push([_command, _args, _options]); return {status: 0, signal: null, error: undefined};};
    assert.throws(() => buildNativeWindowsCore(f.descriptor, f.output, {MOM_AOT_BUILD_ONLY: "0", CL: "/bad"}, {spawn: mockSpawn}), /Windows.*MOM_AOT_BUILD_ONLY/);
    assert.equal(calls.length, 0);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("does not execute the build-only lane on Linux", {skip: process.platform === "win32"}, t => {
    const f = fixture(t);
    /** @type {unknown[][]} */
    const calls = [];
    /** @param {string} _command @param {readonly string[]} _args @param {import("node:child_process").SpawnSyncOptions} _options */
    const mockSpawn = (_command, _args, _options) => {calls.push([_command, _args, _options]); return {status: 0, signal: null, error: undefined};};
    assert.throws(() => buildNativeWindowsCore(f.descriptor, f.output, {MOM_AOT_BUILD_ONLY: "1"}, {spawn: mockSpawn}), /Windows.*MOM_AOT_BUILD_ONLY/);
    assert.equal(calls.length, 0);
    assert.equal(fs.existsSync(f.output), false);
  });
});

test.describe("Windows dumpbin symbol identity gate", () => {
  const good = "00A 00000000 UNDEF notype External | ?algo_params@@YAXXZ\n00B 00000000 UNDEF notype External | ?pearlhash_claim@@YAXXZ\n00C 00000001 SECT3 notype External | ?local_core@@YAXXZ\n";

  test("accepts only real unresolved GPU API references", () => {
    assert.deepEqual(parseDumpbinSymbols(good).gpuUnresolvedReferences, ["algo_params", "pearlhash_claim"]);
    assert.doesNotThrow(() => parseDumpbinSymbols(good + "00D 00000001 SECT4 notype External | ?helper@gpu@@YAXXZ (void helper(gpu::type))\n"));
    assert.doesNotThrow(() => parseDumpbinSymbols(good + "00E 00000000 SECT9 notype () External | ?_Insert_node@?$_Tree_val@std@@QEAAXXZ\n"));
  });

  test("rejects missing, defined, import-thunk, and Node/V8/NAPI symbols", () => {
    for (const bad of [
      good.replace(/00B[^\n]+\n/, ""),
      good + "00D 00000001 SECT4 notype External | ?algo_params@@YAXXZ\n",
      good + "00D 00000001 SECT3F2 notype External | ?c29@@YAXXZ\n",
      good + "00D 00000000 UNDEF notype External | __imp_?algo_params@@YAXXZ\n",
      good + "00D 00000000 UNDEF notype External | __imp_?c29@@YAXXZ\n",
      good + "00D 00000000 UNDEF notype External | ?v8_call@@YAXXZ\n",
      good + "00D 00000000 UNDEF notype External | ?Call@internal@node@@YAXXZ\n",
      good + "00D 00000000 UNDEF notype External | napi_create_object\n",
    ]) {assert.throws(() => parseDumpbinSymbols(bad), /Windows Core/);}
    assert.throws(() => parseDumpbinSymbols("x".repeat((16 << 20) + 1)), /malformed|excessive/);
  });

  test("audits one combined x64/header/directive/symbol dump", () => {
    const dump = ["Dump of file core.lib", "File Type: LIBRARY", "FILE HEADER VALUES", "             8664 machine (x64)",
      "COFF SYMBOL TABLE", "  /DEFAULTLIB:LIBCMT", good].join("\n");
    const audit = parseDumpbinArchive(dump, 1);
    assert.equal(audit.coffTableCount, 1);
    assert.equal(audit.machineCount, 1);
    assert.equal(audit.x64MachineCount, 1);
    assert.deepEqual(audit.gpuUnresolvedReferences, ["algo_params", "pearlhash_claim"]);
    assert.throws(() => parseDumpbinArchive(dump.replace("8664 machine (x64)", "014C machine (x86)"), 1), /count mismatch/);
    assert.throws(() => parseDumpbinArchive(dump.replace("/DEFAULTLIB:LIBCMT", "/DEFAULTLIB:node.lib"), 1), /defaultlib/);
  });
});
