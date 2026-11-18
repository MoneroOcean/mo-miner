"use strict";
const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {compileNativeObjects, environmentValue} = require("../scripts/native-object-cache");
const {buildDependencies} = require("../scripts/build-source-inputs");
const {describe} = test;

/** @typedef {Parameters<typeof compileNativeObjects>[0]} CacheOptions */
/** @typedef {CacheOptions["commands"][number]} CompileCommand */
/** @typedef {ReturnType<typeof compileNativeObjects>} CacheResult */
const ninja = (() => {
  const result = spawnSync("ninja", ["--version"], {encoding: "utf8"});
  return !result.error && result.status === 0 ? "ninja" : undefined;
})();

/** Fake compiler that emits object bytes and a Make depfile from the arguments it receives. */
/** @param {import("node:test").TestContext} t @param {string} [prefix] @returns {any} */
function fixture(t, prefix = "mom native object cache-") {
  const repo = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  t.after(() => fs.rmSync(repo, {recursive: true, force: true}));
  /** @param {string} name @param {string | Buffer} value */
  const put = (name, value) => {
    const file = path.join(repo, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, value); return file;
  };
  const count = put("compile-count", "0");
  const compiler = put("fake-compiler.js", `#!${process.execPath}\n"use strict";
const fs=require("node:fs"),path=require("node:path"),a=process.argv.slice(2);
const source=a[a.indexOf("-c")+1], oi=a.indexOf("-o"), object=oi>=0?a[oi+1]:a.find(x=>x.startsWith("-Fo"))?.slice(3);
const mi=a.indexOf("-MF"), xi=a.indexOf("-dependency-file"), dep=mi>=0?a[mi+1]:xi>=0?a[xi+2]:undefined;
if(!source||!object||!dep)process.exit(41);
const count=${JSON.stringify(count)}, headers=[];
const envNames=["MOM_NATIVE_BUILD_OUTPUT","MOM_NATIVE_BUILD_STAGE","MOM_NATIVE_OUTPUT","MOM_NATIVE_SDK","MOM_NATIVE_UPX","MOM_NATIVE_UPX_SHA256","HOSTNAME","MOM_FORCE_REBUILD","GITHUB_JOB","MOM_NATIVE_APPLICATION_DEADLINE_EPOCH","MOM_NATIVE_COMPONENT_DEADLINE_EPOCH","MOM_NATIVE_WINDOWS_CONTROLLER_DEADLINE_EPOCH","MOM_NATIVE_WINDOWS_CORE_DEADLINE_EPOCH","MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH","MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH","MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH","MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH","CMAKE_BUILD_PARALLEL_LEVEL","NATIVE_CACHE_TEST"];
const src=fs.readFileSync(source), probe=src.toString().includes('__has_include("a.inc")');
for(const arg of a){if(arg==="-I"||arg.startsWith("-I")){const dir=arg.length>2?arg.slice(2):a[a.indexOf(arg)+1];for(const name of ["kernel.h","generated.inc",...(probe?["a.inc"]:[])]){const file=path.join(dir,name);if(fs.existsSync(file)&&!headers.includes(file))headers.push(file);}}}
const deps=[source,...headers];
const escape=x=>x.replaceAll("\\\\","\\\\\\\\").replaceAll(" ","\\\\ ").replaceAll("#","\\\\#").replaceAll(":","\\\\:");
const token=x=>a.includes("-MV")?'"'+x+'"':escape(x);
const previous=Number(fs.readFileSync(count,"utf8"));fs.writeFileSync(count,String(previous+1));
fs.mkdirSync(path.dirname(object),{recursive:true});fs.mkdirSync(path.dirname(dep),{recursive:true});
fs.writeFileSync(object,Buffer.concat([src,...headers.map(x=>fs.readFileSync(x)),...(probe?[Buffer.from(headers.some(x=>path.basename(x)==="a.inc")?"HAS_A_INC\\n":"NO_A_INC\\n")]:[]),
Buffer.from(JSON.stringify({args:a,env:Object.fromEntries(envNames.map(name=>[name,process.env[name]??null]))}))]));
fs.writeFileSync(dep,token(object)+": "+deps.map(token).join(" ")+"\\n");
const mutate=process.env["NATIVE_CACHE_MUTATE_KEY"];if(mutate)fs.writeFileSync(mutate,"mutated during compile");
console.log("fake compiled");`);
  fs.chmodSync(compiler, 0o755);
  const source = put("native/kernel.cpp", "int kernel() { return 1; }\n"), header = put("native/kernel.h", "#define KERNEL_VALUE 1\n");
  const include = path.join(repo, "native");
  const environment = {...process.env}; delete environment["MOM_NATIVE_CACHE"]; delete environment["MOM_FORCE_REBUILD"];
  /** @param {string} output @param {string} [src] @param {string[]} [extra] */
  const command = (output, src = source, extra = []) => {
    const object = path.join(output, "kernel.o"), depfile = path.join(output, "kernel.d");
    return {command: compiler, args: ["-I", include, ...extra, "-MF", depfile, "-c", src, "-o", object], source: src, object, depfile};
  };
  /** @param {string} output @param {CompileCommand[]} [commands] @param {Partial<CacheOptions>} [extra] */
  const run = (output, commands = [command(output)], extra = {}) => {
    fs.mkdirSync(output, {recursive: true});
    const executableCommands = commands.map(entry => entry.command === compiler ? {...entry, command: process.execPath,
      args: [compiler, ...entry.args]} : entry);
    return compileNativeObjects({repo, output, namespace: "linux/x64/gpu/test", commands: executableCommands, environment,
      includeRoots: [include], ...(ninja ? {ninja} : {}), ...extra,
      keyFiles: [...new Set([compiler, ...(extra.keyFiles || [])])]});
  };
  return {repo, put, count, compiler, source, header, include, environment, command, run};
}

/** @param {CacheResult} result */
function firstObject(result) {
  const item = result.objects[0];
  if (!item) {throw new Error("Native object helper returned no command result");}
  return item;
}

describe("native object cache", () => {
  test("tool lookup honors Windows environment key casing and rejects ambiguous aliases", () => {
    assert.equal(environmentValue({Path: "C:\\tools"}, "PATH", "win32"), "C:\\tools");
    assert.equal(environmentValue({Pathext: ".EXE;.COM"}, "PATHEXT", "win32"), ".EXE;.COM");
    assert.equal(environmentValue({PATH: "C:\\tools", Path: "C:\\tools"}, "PATH", "win32"), "C:\\tools");
    assert.throws(() => environmentValue({PATH: "C:\\first", Path: "C:\\second"}, "PATH", "win32"),
      /Conflicting case variants for Windows environment variable: PATH/);
    assert.equal(environmentValue({Path: "/tools"}, "PATH", "linux"), undefined);
    assert.equal(environmentValue({Path: "/tools"}, "Path", "linux"), "/tools");
  });

  test("Ninja restores cached objects into each fresh candidate output", {skip: !ninja && "Ninja is not installed"}, (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), out1 = path.join(f.repo, "candidate-one"), out2 = path.join(f.repo, "candidate-two");
    const first = f.run(out1, [f.command(out1)], {finalize: true, liveObjects: [path.join(out1, "kernel.o")]});
    const object = fs.readFileSync(firstObject(first).object);
    assert.ok(first.ninja, "the cold build must select Ninja rather than silently test its fallback");
    assert.equal(first.cacheMisses, 1); assert.equal(firstObject(first).status, 0, JSON.stringify(first)); assert.equal(firstObject(first).cacheHit, false);
    const second = f.run(out2, [f.command(out2)], {finalize: true, liveObjects: [path.join(out2, "kernel.o")]});
    assert.equal(second.cacheHits, 1); assert.equal(firstObject(second).status, 0); assert.equal(firstObject(second).cacheHit, true);
    assert.deepEqual(fs.readFileSync(firstObject(second).object), object); assert.equal(fs.readFileSync(f.count, "utf8"), "1");
    assert.deepEqual(fs.readdirSync(second.cacheRoot).filter(name => name.startsWith(".run-") || name === ".ninja_log"), []);
  });

  test("a missing Ninja preserves direct compilation and object reuse", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), environment = {...f.environment};
    for (const key of Object.keys(environment)) {
      if (["PATH", "MOM_NATIVE_NINJA"].includes(key.toUpperCase())) {delete environment[key];}
    }
    environment["PATH"] = "";
    const first = f.run(path.join(f.repo, "direct-one"), undefined, {ninja: undefined, environment});
    assert.equal(first.ninja, null); assert.equal(first.cacheMisses, 1); assert.equal(firstObject(first).status, 0);
    const second = f.run(path.join(f.repo, "direct-two"), undefined, {ninja: undefined, environment});
    assert.equal(second.ninja, null); assert.equal(second.cacheHits, 1); assert.equal(firstObject(second).status, 0);
    assert.equal(fs.readFileSync(f.count, "utf8"), "1");
    assert.deepEqual(fs.readdirSync(second.cacheRoot).filter(name => name.startsWith(".run-") || name === ".ninja_log"), []);
  });

  test("backdated source and dependency edits invalidate only the affected object", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), run = () => {
      const output = path.join(f.repo, "candidate" + Number(fs.readFileSync(f.count, "utf8")));
      return f.run(output, [f.command(output)], {finalize: true, liveObjects: [path.join(output, "kernel.o")]});
    };
    assert.equal(firstObject(run()).cacheHit, false);
    fs.writeFileSync(f.source, "int kernel() { return 2; }\n"); fs.utimesSync(f.source, 1, 1);
    assert.equal(firstObject(run()).cacheHit, false);
    fs.writeFileSync(f.header, "#define KERNEL_VALUE 2\n"); fs.utimesSync(f.header, 1, 1);
    assert.equal(firstObject(run()).cacheHit, false); assert.equal(fs.readFileSync(f.count, "utf8"), "3");
  });

  test("a changed compile signature skips obsolete dependency scans but checks newly built inputs", t => {
    const f = fixture(t), first = path.join(f.repo, "signature-one"), second = path.join(f.repo, "signature-two");
    assert.equal(f.run(first).cacheMisses, 1);
    let headerReads = 0;
    const open = fs.openSync;
    fs.openSync = (file, flags, mode) => {
      if (file === f.header) {headerReads++;}
      return open(file, flags, mode);
    };
    try {
      const result = f.run(second, [f.command(second, f.source, ["-DCHANGED=1"])], {
        resolveDependencies: (/** @type {{depfileText: string, repo: string}} */ info) => {
          assert.equal(headerReads, 0, "a signature mismatch already proves the cached object cannot be reused");
          return buildDependencies(info.depfileText, info.repo) || [];
        },
      });
      assert.equal(result.cacheMisses, 1); assert.equal(firstObject(result).status, 0);
      assert.equal(headerReads, 1, "the freshly compiled dependency must still be content-verified");
      assert.equal(fs.readFileSync(f.count, "utf8"), "2");
    } finally {fs.openSync = open;}
  });

  test("include membership detects a newly added shadowing header", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), firstOutput = path.join(f.repo, "candidate-one"), first = f.run(firstOutput, [f.command(firstOutput)],
      {includeRoots: [f.include, f.repo], finalize: true, liveObjects: [path.join(firstOutput, "kernel.o")]});
    assert.equal(firstObject(first).cacheHit, false);
    f.put("cstdint", "#pragma once\n");
    const secondOutput = path.join(f.repo, "candidate-two"), second = f.run(secondOutput, [f.command(secondOutput)],
      {includeRoots: [f.include, f.repo], finalize: true, liveObjects: [path.join(secondOutput, "kernel.o")]});
    assert.equal(firstObject(second).cacheHit, false); assert.equal(fs.readFileSync(f.count, "utf8"), "2");
  });

  test("generated source and declared generated headers are stable across random output roots", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t);
    /** @param {string} output @param {string} headerText */
    const make = (output, headerText) => {
      const source = path.join(output, "factory.cpp"), header = path.join(output, "generated.inc");
      fs.mkdirSync(output, {recursive: true}); fs.writeFileSync(source, "int generated() { return GENERATED_VALUE; }\n"); fs.writeFileSync(header, headerText);
      const object = path.join(output, "factory.o"), depfile = path.join(output, "factory.d");
      const entry = {command: f.compiler, source, object, depfile, generatedInputs: [header],
        args: ["-I", output, "-MF", depfile, "-c", source, "-o", object]};
      return {entry, result: f.run(output, [entry], {includeRoots: [output]})};
    };
    const first = make(path.join(f.repo, "generated-one"), "#define GENERATED_VALUE 1\n");
    const second = make(path.join(f.repo, "generated-two"), "#define GENERATED_VALUE 1\n");
    assert.equal(firstObject(first.result).cacheHit, false); assert.equal(firstObject(second.result).cacheHit, true);
    const third = make(path.join(f.repo, "generated-three"), "#define GENERATED_VALUE 2\n");
    assert.equal(firstObject(third.result).cacheHit, false); assert.equal(fs.readFileSync(f.count, "utf8"), "2");
  });

  test("every object sees source-like files in the shared generated include namespace", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), probeSource = f.put("native/probe.cpp", '#if __has_include("a.inc")\n#include "a.inc"\n#endif\nint probe() { return 1; }\n');
    /** @param {string} output */
    const build = output => {
      fs.mkdirSync(output, {recursive: true});
      const header = path.join(output, "a.inc"); fs.writeFileSync(header, "#define GENERATED_A 1\n");
      const first = {...f.command(output), generatedInputs: [header]};
      const object = path.join(output, "probe.o"), depfile = path.join(output, "probe.d");
      const second = {command: f.compiler, source: probeSource, object, depfile,
        args: ["-I", output, "-MF", depfile, "-c", probeSource, "-o", object]};
      return f.run(output, [first, second], {includeRoots: [output]});
    };
    const first = build(path.join(f.repo, "generated-shared-one"));
    assert.equal(first.objects[1]?.cacheHit, false);
    assert.ok(fs.readFileSync(path.join(f.repo, "generated-shared-one", "probe.o")).includes(Buffer.from("HAS_A_INC")),
      "the second object must observe the generated header even though it is not individually declared");
    const second = build(path.join(f.repo, "generated-shared-two"));
    assert.equal(second.objects[0]?.cacheHit, true); assert.equal(second.objects[1]?.cacheHit, true);
    assert.ok(fs.readFileSync(path.join(f.repo, "generated-shared-two", "probe.o")).includes(Buffer.from("HAS_A_INC")));
    assert.equal(fs.readFileSync(f.count, "utf8"), "2");
  });

  test("generated include-name membership invalidates conditional probes without an output include root", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), probeSource = f.put("native/probe.cpp", '#if __has_include("a.inc")\n#include "a.inc"\n#endif\nint probe() { return 1; }\n');
    /** @param {string} output @param {boolean} withHeader */
    const build = (output, withHeader) => {
      fs.mkdirSync(output, {recursive: true});
      if (withHeader) {fs.writeFileSync(path.join(output, "a.inc"), "#define GENERATED_A 1\n");}
      const object = path.join(output, "probe.o"), depfile = path.join(output, "probe.d");
      const command = {command: f.compiler, source: probeSource, object, depfile,
        args: ["-I", output, "-MF", depfile, "-c", probeSource, "-o", object]};
      return f.run(output, [command]);
    };
    const first = build(path.join(f.repo, "conditional-absent"), false);
    assert.equal(firstObject(first).cacheHit, false);
    assert.ok(fs.readFileSync(path.join(f.repo, "conditional-absent", "probe.o")).includes(Buffer.from("NO_A_INC")));
    const second = build(path.join(f.repo, "conditional-present"), true);
    assert.equal(firstObject(second).cacheHit, false);
    assert.ok(fs.readFileSync(path.join(f.repo, "conditional-present", "probe.o")).includes(Buffer.from("HAS_A_INC")));
    const third = build(path.join(f.repo, "conditional-present-again"), true);
    assert.equal(firstObject(third).cacheHit, true); assert.equal(fs.readFileSync(f.count, "utf8"), "2");
  });

  test("CMAKE_BUILD_PARALLEL_LEVEL is scheduling-only for compiler object reuse", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), firstOutput = path.join(f.repo, "parallel-two"), secondOutput = path.join(f.repo, "parallel-unset");
    const withParallelism = {...f.environment, CMAKE_BUILD_PARALLEL_LEVEL: "2"}, withoutParallelism = {...f.environment};
    delete withoutParallelism.CMAKE_BUILD_PARALLEL_LEVEL;
    const first = f.run(firstOutput, [f.command(firstOutput)], {environment: withParallelism});
    const second = f.run(secondOutput, [f.command(secondOutput)], {environment: withoutParallelism});
    assert.equal(firstObject(first).cacheHit, false);
    assert.equal(firstObject(second).cacheHit, true, "changing only CMake build scheduling must reuse the object");
    assert.equal(fs.readFileSync(f.count, "utf8"), "1");
    assert.ok(fs.readFileSync(firstObject(first).object, "utf8").includes('"CMAKE_BUILD_PARALLEL_LEVEL":null'),
      "the scheduling-only variable must not reach the compiler child");
    const compilerEnv = {...withoutParallelism, NATIVE_CACHE_TEST: "different"}, thirdOutput = path.join(f.repo, "semantic-env-change");
    const third = f.run(thirdOutput, [f.command(thirdOutput)], {environment: compilerEnv});
    assert.equal(firstObject(third).cacheHit, false, "a compiler-relevant environment change must still miss");
    assert.equal(fs.readFileSync(f.count, "utf8"), "2");
  });

  test("changed flags, tool bytes, and environment cannot hit an old object", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), first = f.run(path.join(f.repo, "candidate-one")); assert.equal(firstObject(first).cacheHit, false);
    const second = f.run(path.join(f.repo, "candidate-two"), [f.command(path.join(f.repo, "candidate-two"), f.source, ["-DTEST_FLAG=1"]) ]);
    assert.equal(firstObject(second).cacheHit, false);
    f.put("fake-compiler.js", fs.readFileSync(f.compiler, "utf8") + "\n// tool identity changed\n"); fs.chmodSync(f.compiler, 0o755);
    const third = f.run(path.join(f.repo, "candidate-three")); assert.equal(firstObject(third).cacheHit, false);
    const nextEnv = {...f.environment, NATIVE_CACHE_TEST: "different"};
    const fourth = f.run(path.join(f.repo, "candidate-four"), [f.command(path.join(f.repo, "candidate-four"))], {environment: nextEnv});
    assert.equal(firstObject(fourth).cacheHit, false); assert.equal(fs.readFileSync(f.count, "utf8"), "4");
  });

  test("producer paths, job identity and deadlines neither reach the compiler nor spoil reuse", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), names = ["MOM_NATIVE_BUILD_OUTPUT", "MOM_NATIVE_BUILD_STAGE", "MOM_NATIVE_OUTPUT", "MOM_NATIVE_SDK",
      "MOM_NATIVE_UPX", "MOM_NATIVE_UPX_SHA256", "HOSTNAME", "GITHUB_JOB", "MOM_NATIVE_APPLICATION_DEADLINE_EPOCH",
      "MOM_NATIVE_COMPONENT_DEADLINE_EPOCH", "MOM_NATIVE_WINDOWS_CONTROLLER_DEADLINE_EPOCH", "MOM_NATIVE_WINDOWS_CORE_DEADLINE_EPOCH",
      "MOM_NATIVE_WINDOWS_GPU_DEADLINE_EPOCH", "MOM_NATIVE_WINDOWS_HOST_SDK_DEADLINE_EPOCH", "MOM_NATIVE_WINDOWS_LLVM_DEADLINE_EPOCH",
      "MOM_NATIVE_WINDOWS_LLVM_RUN_DEADLINE_EPOCH", "CMAKE_BUILD_PARALLEL_LEVEL"];
    /** @param {string} output @param {string} runId @param {string} [compilerEnv] */
    const build = (output, runId, compilerEnv = "stable") => {
      const environment = {...f.environment};
      for (const name of names) {environment[name] = `${name}-${runId}-${output}`;}
      environment.NATIVE_CACHE_TEST = compilerEnv;
      const before = {...environment}, result = f.run(output, [f.command(output)], {environment});
      assert.deepEqual(environment, before, "the helper must not mutate caller-owned environment");
      return result;
    };
    const first = build(path.join(f.repo, "orchestrator-one"), "one"), second = build(path.join(f.repo, "orchestrator-two"), "two");
    assert.equal(firstObject(first).cacheHit, false); assert.equal(firstObject(second).cacheHit, true);
    const third = build(path.join(f.repo, "orchestrator-three"), "three", "compiler-relevant-change");
    assert.equal(firstObject(third).cacheHit, false, "compiler-relevant environment must still invalidate reuse");
    assert.equal(fs.readFileSync(f.count, "utf8"), "2");
    const objectText = fs.readFileSync(firstObject(first).object, "utf8"), jsonStart = objectText.lastIndexOf("{\"args\":");
    const compilerReceipt = JSON.parse(objectText.slice(jsonStart));
    assert.deepEqual(compilerReceipt.env, {...Object.fromEntries([...names, "MOM_FORCE_REBUILD"].map(name => [name, null])),
      NATIVE_CACHE_TEST: "stable"});
    const changedText = fs.readFileSync(firstObject(third).object, "utf8"), changedStart = changedText.lastIndexOf("{\"args\":");
    assert.equal(JSON.parse(changedText.slice(changedStart)).env.NATIVE_CACHE_TEST, "compiler-relevant-change");
  });

  test("non-depfile key inputs affect cache identity and are rechecked after compilation", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), keyFile = f.put("sdk/plugin.bc", "plugin bytes one");
    /** @param {string} name */
    const run = name => {
      const output = path.join(f.repo, name);
      return f.run(output, [f.command(output)], {keyFiles: [keyFile]});
    };
    assert.equal(firstObject(run("key-one")).cacheHit, false);
    assert.equal(firstObject(run("key-two")).cacheHit, true);
    fs.writeFileSync(keyFile, "plugin bytes two");
    assert.equal(firstObject(run("key-three")).cacheHit, false);
    assert.throws(() => f.run(path.join(f.repo, "key-missing"), undefined, {keyFiles: [path.join(f.repo, "sdk/missing.bc")]}), /Native key file/);
    const changing = f.put("sdk/changing.bc", "before compile");
    const environment = {...f.environment}; environment["NATIVE_CACHE_MUTATE_KEY"] = changing;
    assert.throws(() => f.run(path.join(f.repo, "key-midcompile"), undefined, {keyFiles: [changing], environment}),
      /key files changed during compilation/);
  });

  test("symlinked non-depfile key inputs fail closed", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), keyFile = f.put("sdk/plugin.bc", "plugin bytes"), alias = path.join(f.repo, "sdk/plugin-alias.bc");
    try {fs.symlinkSync(keyFile, alias, "file");} catch (error) {
      if (process.platform === "win32" && error instanceof Error && "code" in error && error.code === "EPERM") {
        t.skip("Windows symbolic-link privilege is unavailable"); return;
      }
      throw error;
    }
    assert.throws(() => f.run(path.join(f.repo, "key-symlink"), undefined, {keyFiles: [alias]}), /Native key file/);
  });

  test("corrupt object data and missing state rebuild safely", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), first = f.run(path.join(f.repo, "candidate-one"));
    const id = fs.readdirSync(path.join(first.cacheRoot, "slots"))[0];
    if (!id) {throw new Error("Expected one cached object slot");}
    const slot = path.join(first.cacheRoot, "slots", id);
    fs.writeFileSync(path.join(slot, "object.o"), "corrupt");
    const second = f.run(path.join(f.repo, "candidate-two")); assert.equal(firstObject(second).cacheHit, false);
    fs.unlinkSync(path.join(slot, "state.json"));
    const third = f.run(path.join(f.repo, "candidate-three")); assert.equal(firstObject(third).cacheHit, false);
    assert.equal(fs.readFileSync(f.count, "utf8"), "3");
  });

  test("symlinked cache objects are rejected without following or changing their targets", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), first = f.run(path.join(f.repo, "candidate-one"));
    const id = fs.readdirSync(path.join(first.cacheRoot, "slots"))[0];
    if (!id) {throw new Error("Expected one cached object slot");}
    const slot = path.join(first.cacheRoot, "slots", id);
    const target = f.put("outside-object", "preserve me"), object = path.join(slot, "object.o"); fs.unlinkSync(object);
    try {fs.symlinkSync(target, object, "file");} catch (error) {
      if (process.platform === "win32" && error instanceof Error && "code" in error && error.code === "EPERM") {
        t.skip("Windows symbolic-link privilege is unavailable"); return;
      }
      throw error;
    }
    assert.throws(() => f.run(path.join(f.repo, "candidate-two")), /symlink|regular file|cache artifact/i);
    assert.equal(fs.readFileSync(target, "utf8"), "preserve me");
  });

  test("Windows-style dependency-file and /Fo arguments are rewritten as tokens", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), output = path.join(f.repo, "windows candidate"); fs.mkdirSync(output);
    const object = path.join(output, "kernel.obj"), depfile = path.join(output, "kernel.d");
    const entry = {command: f.compiler, source: f.source, object, depfile, dependencyFormat: "nmake",
      args: ["-I", f.include, "-Xclang", "-dependency-file", "-Xclang", depfile, "-Xclang", "-MT", "-Xclang", object,
        "-Xclang", "-MV", "-Fo" + object, "-c", f.source]};
    const result = f.run(output, [entry], {resolveDependencies: (/** @type {{command: CompileCommand, depfileText: string}} */ input) => {
      const {command, depfileText} = input;
      assert.equal(command.depfile, depfile); assert.match(depfileText, /kernel\.obj/); return [f.source, f.header];
    }});
    assert.equal(firstObject(result).status, 0); assert.ok(fs.existsSync(object)); assert.ok(fs.existsSync(depfile));
  });

  test("cache disable executes exact fresh-output command paths without populating slots", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), output = path.join(f.repo, "cold"), environment = {...f.environment};
    environment["MOM_NATIVE_CACHE"] = "0";
    const result = f.run(output, [f.command(output)], {environment, finalize: true, liveObjects: []});
    assert.equal(result.cacheHits, 0); assert.equal(firstObject(result).cacheHit, false); assert.equal(firstObject(result).status, 0);
    assert.ok(fs.existsSync(firstObject(result).object));
    assert.ok(!fs.existsSync(path.join(result.cacheRoot, "slots")) || fs.readdirSync(path.join(result.cacheRoot, "slots")).length === 0);

    const seeded = f.run(path.join(f.repo, "seeded")), beforeIndex = fs.readFileSync(path.join(seeded.cacheRoot, "slots.json"), "utf8");
    const beforeSlots = fs.readdirSync(path.join(seeded.cacheRoot, "slots")).sort();
    const coldFinalizeOutput = path.join(f.repo, "cold-finalize");
    f.run(coldFinalizeOutput, [f.command(coldFinalizeOutput)], {environment, finalize: true, liveObjects: []});
    assert.equal(fs.readFileSync(path.join(seeded.cacheRoot, "slots.json"), "utf8"), beforeIndex);
    assert.deepEqual(fs.readdirSync(path.join(seeded.cacheRoot, "slots")).sort(), beforeSlots);
  });

  test("MOM_FORCE_REBUILD=1 is cold and non-mutating while 0 remains reusable", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), seeded = f.run(path.join(f.repo, "force-seed"));
    const zeroOutput = path.join(f.repo, "force-zero"), zeroEnv = {...f.environment, MOM_FORCE_REBUILD: "0"};
    const zero = f.run(zeroOutput, [f.command(zeroOutput)], {environment: zeroEnv, finalize: true, liveObjects: [path.join(zeroOutput, "kernel.o")]});
    assert.equal(firstObject(zero).cacheHit, true); assert.equal(fs.readFileSync(f.count, "utf8"), "1");
    const beforeIndex = fs.readFileSync(path.join(seeded.cacheRoot, "slots.json"), "utf8"), beforeSlots = fs.readdirSync(path.join(seeded.cacheRoot, "slots")).sort();

    const forceOutput = path.join(f.repo, "force-one"), forceEnv = {...f.environment, MOM_FORCE_REBUILD: "1"};
    const forced = f.run(forceOutput, [f.command(forceOutput)], {environment: forceEnv, finalize: true, liveObjects: []});
    assert.equal(firstObject(forced).cacheHit, false); assert.equal(firstObject(forced).status, 0);
    assert.equal(fs.readFileSync(f.count, "utf8"), "2");
    assert.equal(fs.readFileSync(path.join(seeded.cacheRoot, "slots.json"), "utf8"), beforeIndex);
    assert.deepEqual(fs.readdirSync(path.join(seeded.cacheRoot, "slots")).sort(), beforeSlots);
    const objectText = fs.readFileSync(firstObject(forced).object, "utf8"), jsonStart = objectText.lastIndexOf("{\"args\":");
    assert.equal(JSON.parse(objectText.slice(jsonStart)).env["MOM_FORCE_REBUILD"], null);
    assert.throws(() => f.run(path.join(f.repo, "force-invalid"), undefined,
      {environment: {...f.environment, MOM_FORCE_REBUILD: "yes"}}), /MOM_FORCE_REBUILD must be 0 or 1/);
  });

  test("Ninja graph and manifest safely handle dollar, hash and colon in repository paths", {skip: process.platform === "win32"}, (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t, "mom native $#: object cache-"), output = path.join(f.repo, "candidate");
    const result = f.run(output, [f.command(output)], {resolveDependencies: (/** @type {{command: CompileCommand}} */ input) => [input.command.source, f.header]});
    assert.equal(firstObject(result).status, 0); assert.ok(fs.existsSync(firstObject(result).object));
  });

  test("path rewriting preserves longer sibling paths in joined and Windows-style arguments", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), output = path.join(f.repo, "native-out"), siblingSdk = path.join(f.repo, "native-out-sdk", "include");
    fs.mkdirSync(siblingSdk, {recursive: true});
    const object = path.join(output, "kernel.o"), depfile = path.join(output, "kernel.d");
    const siblingWindows = output.replaceAll(path.sep, "\\") + "-sdk\\include";
    const windowsOutput = output.replaceAll(path.sep, "\\");
    const entry = {command: f.compiler, source: f.source, object, depfile,
      args: ["-isystem" + siblingSdk, "--sdk-root=" + siblingSdk, "-imsvc" + siblingWindows, "--windows-sdk=" + siblingWindows,
        "-I" + output, "-I", output, "--generated-root=" + output, "-include" + windowsOutput,
        "--windows-generated=" + windowsOutput, "-MF", depfile, "-c", f.source, "-o", object]};
    const result = f.run(output, [entry], {includeRoots: [output],
      resolveDependencies: (/** @type {{command: CompileCommand}} */ input) => [input.command.source, f.header]});
    const objectText = fs.readFileSync(object, "utf8"), jsonStart = objectText.lastIndexOf("{\"args\":");
    const argv = JSON.parse(objectText.slice(jsonStart)).args;
    const generatedRoot = path.join(result.cacheRoot, "generated");
    assert.equal(argv[0], "-isystem" + siblingSdk); assert.equal(argv[1], "--sdk-root=" + siblingSdk);
    assert.equal(argv[2], "-imsvc" + siblingWindows); assert.equal(argv[3], "--windows-sdk=" + siblingWindows);
    assert.equal(argv[4], "-I" + generatedRoot); assert.equal(argv[5], "-I"); assert.equal(argv[6], generatedRoot);
    assert.equal(argv[7], "--generated-root=" + generatedRoot);
    assert.equal(argv[8], "-include" + generatedRoot); assert.equal(argv[9], "--windows-generated=" + generatedRoot);
  });

  test("finalizing a graph prunes removed object slots without pruning intermediate batches", (/** @type {import("node:test").TestContext} */ t) => {
    const f = fixture(t), firstOut = path.join(f.repo, "graph-one"), secondOut = path.join(f.repo, "graph-two");
    const source2 = f.put("native/other.cpp", "int other() { return 2; }\n");
    const first = f.run(firstOut, [f.command(firstOut), {...f.command(firstOut, source2), object: path.join(firstOut, "other.o"), depfile: path.join(firstOut, "other.d"),
      args: ["-I", f.include, "-MF", path.join(firstOut, "other.d"), "-c", source2, "-o", path.join(firstOut, "other.o")]}]);
    assert.equal(fs.readdirSync(path.join(first.cacheRoot, "slots")).length, 2);
    const firstObj = path.join(secondOut, "kernel.o");
    f.run(secondOut, [f.command(secondOut)], {finalize: false});
    assert.equal(fs.readdirSync(path.join(first.cacheRoot, "slots")).length, 2);
    const factory = f.command(secondOut, source2);
    factory.object = path.join(secondOut, "factory.o"); factory.depfile = path.join(secondOut, "factory.d");
    factory.args = ["-I", f.include, "-MF", factory.depfile, "-c", source2, "-o", factory.object];
    f.run(secondOut, [factory], {finalize: true, liveObjects: [firstObj, factory.object]});
    assert.equal(fs.readdirSync(path.join(first.cacheRoot, "slots")).length, 2);
    assert.ok(!fs.readdirSync(path.join(first.cacheRoot, "slots")).some(name =>
      fs.readFileSync(path.join(first.cacheRoot, "slots", name, "state.json"), "utf8").includes("other.o")));
  });
});
