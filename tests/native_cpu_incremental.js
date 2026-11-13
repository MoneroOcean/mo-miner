"use strict";
const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {compileNativeObjects} = require("../scripts/native-object-cache");
const {cpuCompileCommands, cpuIncludeRoots, selectCpuArchiver, verifyCpuArchiver} = require("../scripts/build-native-cpu-probe");
const {fileHash} = require("../scripts/build-source-inputs");

const repoRoot = path.resolve(__dirname, "..");

/** A fake compiler emits regular objects and actual Make depfiles; no native compiler is launched. */
/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom cpu incremental-")), repo = path.join(root, "repo");
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  /** @param {string} name @param {string | Buffer} value */
  const put = (name, value) => {
    const file = path.join(repo, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, value); return file;
  };
  const sourceFiles = /** @type {Array<[string, string]>} */ ([
    ["native/core.cpp", "int core() { return 1; }\n"],
    ["native/job.cpp", "int job() { return 2; }\n"],
    ["xmrig/crypto/common/VirtualMemory.cpp", "int memory() { return 3; }\n"],
    ["xmrig/hw/msr/Msr_linux.cpp", "int msr() { return 4; }\n"],
    ["sycl/lib.h", "void sycl_cleanup();\n"],
  ]);
  for (const [file, body] of sourceFiles) {put(file, body);}
  const sdkInclude = path.join(repo, "sdk/include"); fs.mkdirSync(sdkInclude, {recursive: true});
  const count = path.join(root, "compile-count"); fs.writeFileSync(count, "0");
  const compiler = path.join(root, "fake-icpx.js");
  fs.writeFileSync(compiler, `#!${process.execPath}
"use strict";
const fs=require("node:fs"),path=require("node:path"),a=process.argv.slice(2);
const source=a[a.indexOf("-c")+1], object=a[a.indexOf("-o")+1], dep=a[a.indexOf("-MF")+1];
if(!source||!object||!dep||!a.includes("-MD"))process.exit(41);
const count=${JSON.stringify(count)}, headers=[];
const sourceText=fs.readFileSync(source,"utf8"), roots=[];
for(let i=0;i<a.length;i++){if(a[i]==="-I"||a[i]==="-isystem"){roots.push(a[++i]);}else if(a[i].startsWith("-I")){roots.push(a[i].slice(2));}}
for(const match of sourceText.matchAll(/^\\s*#include\\s+"([^"]+)"/gm)){
  const name=match[1]; for(const base of [path.dirname(source),...roots]){const candidate=path.resolve(base,name);if(fs.existsSync(candidate)&&!headers.includes(candidate)){headers.push(candidate);break;}}
}
const escape=x=>x.replaceAll("\\\\","\\\\\\\\").replaceAll(" ","\\\\ ").replaceAll("#","\\\\#").replaceAll(":","\\\\:");
fs.writeFileSync(count,String(Number(fs.readFileSync(count,"utf8"))+1));
fs.mkdirSync(path.dirname(object),{recursive:true});fs.mkdirSync(path.dirname(dep),{recursive:true});
fs.writeFileSync(object,Buffer.concat([Buffer.from(sourceText),Buffer.from(JSON.stringify(a))]));
fs.writeFileSync(dep,escape(object)+": "+[source,...headers].map(escape).join(" ")+"\\n");
console.log("mock compiled");`);
  fs.chmodSync(compiler, 0o755);
  const environment = {...process.env}; delete environment["MOM_NATIVE_CACHE"];
  return {root, repo, put, sdkInclude, count, compiler, environment};
}

/** @param {string} output */
function writeStub(output) {
  fs.mkdirSync(output, {recursive: true});
  fs.writeFileSync(path.join(output, "gpu-unavailable.cpp"), '#include "sycl/lib.h"\nvoid unavailable() {}\n');
}

/** Run the real object-cache path with a portable Node launcher for the fake compiler. */
/** @param {string} repo @param {string} output @param {ReturnType<typeof cpuCompileCommands>} commands
 * @param {boolean} gpu @param {NodeJS.ProcessEnv} environment @param {string} compilerScript */
function compileFixtureBatch(repo, output, commands, gpu, environment, compilerScript) {
  const executableCommands = commands.map(command => command.command === compilerScript ? {...command, command: process.execPath,
    args: [compilerScript, ...command.args]} : command);
  return compileNativeObjects({repo, output, namespace: `linux/core/${gpu ? "gpu" : "cpu"}`, commands: executableCommands, environment,
    includeRoots: cpuIncludeRoots(commands, repo, output), keyFiles: [compilerScript], jobs: 1, finalize: true,
    liveObjects: commands.map(command => command.object)});
}

test.describe("Incremental Linux native CPU Core objects", () => {
  test("CPU cache plan preserves Core argv/order and marks its generated proof stub", t => {
    const f = fixture(t), output = path.join(f.root, "candidate-plan"), flags = ["-std=c++17", "-O3", "-DCORE_FIXTURE=1", "-I", f.sdkInclude,
      "-DMOM_NATIVE_RUNTIME", "-fno-lto", "-ffunction-sections", "-fdata-sections"];
    const gpu = cpuCompileCommands(f.repo, output, flags, true, f.compiler), cpu = cpuCompileCommands(f.repo, output, flags, false, f.compiler);
    const files = ["native/core.cpp", "native/job.cpp", "xmrig/crypto/common/VirtualMemory.cpp", "xmrig/hw/msr/Msr_linux.cpp"];
    assert.deepEqual(gpu.map(command => path.relative(f.repo, command.source).split(path.sep).join("/")), files);
    assert.deepEqual(cpu.slice(0, 4), gpu);
    for (const [index, command] of gpu.entries()) {
      const source = path.join(f.repo, files[index] || ""), object = path.join(output, ["core.o", "job.o", "VirtualMemory.o", "Msr_linux.o"][index] || "");
      const depfile = path.join(output, ["core.d", "job.d", "VirtualMemory.d", "Msr_linux.d"][index] || "");
      assert.equal(command.command, f.compiler);
      assert.deepEqual(command.args, [...flags, "-MD", "-MF", depfile, "-c", source, "-o", object]);
      assert.deepEqual(command.generatedInputs || [], []);
    }
    assert.ok(gpu[3]?.args.includes("-DMOM_NATIVE_RUNTIME"), "MSR helper policy must use the native build mode");
    const stub = cpu[4];
    assert.ok(stub);
    assert.deepEqual(stub.generatedInputs, [path.join(output, "gpu-unavailable.cpp")]);
    assert.deepEqual(stub.args, ["-std=c++20", "-O2", "-fPIC", "-I", f.repo, "-MD", "-MF", path.join(output, "gpu-unavailable.d"),
      "-c", path.join(output, "gpu-unavailable.cpp"), "-o", path.join(output, "gpu-unavailable.o")]);
    const roots = cpuIncludeRoots(cpu, f.repo, output);
    assert.ok(roots.includes(f.repo)); assert.ok(roots.includes(output)); assert.ok(roots.includes(f.sdkInclude));
  });

  test("independent CPU bridge objects batch once and restore a fresh ordered graph", t => {
    const f = fixture(t), flags = ["-std=c++17", "-O3", "-I", f.sdkInclude, "-DMOM_NATIVE_RUNTIME", "-fno-lto"];
    /** @param {string} output */
    const compile = output => {
      writeStub(output);
      const commands = cpuCompileCommands(f.repo, output, flags, false, f.compiler);
      return {commands, result: compileFixtureBatch(f.repo, output, commands, false, f.environment, f.compiler)};
    };
    const firstOutput = path.join(f.root, "candidate-one"), first = compile(firstOutput);
    assert.equal(first.result.cacheMisses, 5); assert.equal(first.result.cacheHits, 0);
    assert.deepEqual(first.result.objects.map(item => path.basename(item.object)), ["core.o", "job.o", "VirtualMemory.o", "Msr_linux.o", "gpu-unavailable.o"]);
    assert.deepEqual(first.commands.map(command => path.basename(command.source)), ["core.cpp", "job.cpp", "VirtualMemory.cpp", "Msr_linux.cpp", "gpu-unavailable.cpp"]);
    for (const [index, item] of first.result.objects.entries()) {
      assert.equal(item.status, 0, JSON.stringify(item)); assert.ok(fs.existsSync(item.object)); assert.ok(fs.existsSync(item.depfile));
      assert.match(fs.readFileSync(item.depfile, "utf8"), /\.o:\\?\s/);
      const objectText = fs.readFileSync(item.object, "utf8"), argvStart = objectText.lastIndexOf("\n[");
      assert.ok(argvStart >= 0);
      const actualArgs = JSON.parse(objectText.slice(argvStart + 1)), expectedArgs = first.commands[index]?.args || [];
      /** @param {string[]} args */
      const normalizePaths = (args) => args.map((value, position) => ["-MF", "-c", "-o"].includes(args[position - 1] || "") ? "<path>" : value);
      assert.deepEqual(normalizePaths(actualArgs), normalizePaths(expectedArgs));
      if (index === 4) {assert.equal(first.commands[index]?.source, path.join(firstOutput, "gpu-unavailable.cpp"));}
    }
    assert.equal(fs.readFileSync(f.count, "utf8"), "5");
    const secondOutput = path.join(f.root, "candidate-two"), second = compile(secondOutput);
    assert.equal(second.result.cacheHits, 5); assert.equal(second.result.cacheMisses, 0);
    assert.deepEqual(second.result.objects.map(item => path.dirname(item.object)), Array(5).fill(secondOutput));
    assert.equal(fs.readFileSync(f.count, "utf8"), "5");
    assert.ok(second.result.cacheRoot.endsWith(path.join("linux", "core", "cpu")));
  });

  test("fresh archive uses Engineering objects first and appends the ordered bridge batch before materialization", () => {
    const source = fs.readFileSync(path.join(repoRoot, "scripts/build-native-cpu-probe.js"), "utf8");
    const engineeringObjects = source.indexOf('const objects = variable("OBJS")');
    const archiverSelection = source.indexOf('selectCpuArchiver("icpx"');
    const batch = source.indexOf("const compiled = compileCpuBatch");
    const append = source.indexOf("objects.push(...compiled.objects.map", batch);
    const materialize = source.indexOf("const materialized = []", append);
    const combined = source.indexOf('const combined = path.join(output, gpu ? "cpu-core.a"', materialize);
    assert.ok(engineeringObjects >= 0 && engineeringObjects < batch && archiverSelection >= 0 && archiverSelection < batch &&
      batch < append && append < materialize && materialize < combined);
    assert.ok(source.includes("(VirtualMemory|Msr_linux)"), "native-only MSR object must replace its Engineering object");
    assert.match(source, /run\(archiver\.path, \["qcLs", combined, \.\.\.objects, \.\.\.archives\]/);
    assert.match(source, /-print-prog-name=llvm-ar/);
    assert.match(source, /archiver: \{path: archiver\.path, sha256: archiver\.sha256\}/);
    assert.match(source, /compileNativeObjects\(\{repo, output, namespace: `linux\/core\/\$\{gpu \? "gpu" : "cpu"\}`, commands, environment/);
    assert.match(source, /includeRoots: cpuIncludeRoots\(commands, repo, output\), jobs: 1, finalize: true,/);
    assert.match(source, /liveObjects: commands\.map\(command => command\.object\)/);
  });

  test("GPU Core pins Intel's selected canonical archiver and detects replacement", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), tools = path.join(f.root, "toolchain"); fs.mkdirSync(tools);
    const archiver = path.join(tools, "llvm-ar"); fs.writeFileSync(archiver, "#!/bin/sh\nexit 0\n"); fs.chmodSync(archiver, 0o755);
    const compiler = path.join(f.root, "fake-icpx");
    /** @param {string} selected */
    const writeCompiler = selected => {
      fs.writeFileSync(compiler, `#!${process.execPath}\n"use strict";\n` +
        "if (process.argv.slice(2).join(\" \") !== \"-print-prog-name=llvm-ar\") process.exit(41);\n" +
        `process.stdout.write(${JSON.stringify(selected)} + "\\n");\n`);
      fs.chmodSync(compiler, 0o755);
    };
    writeCompiler(archiver);
    const pin = selectCpuArchiver(compiler, path.join(f.root, "archiver-query.log"));
    assert.equal(pin.path, archiver);
    assert.equal(pin.sha256, fileHash(archiver));
    assert.doesNotThrow(() => verifyCpuArchiver(pin));
    fs.appendFileSync(archiver, "# replacement\n");
    assert.throws(() => verifyCpuArchiver(pin), /changed during CPU archive creation/);

    const alias = path.join(tools, "llvm-ar-alias"); fs.symlinkSync(archiver, alias);
    writeCompiler(alias);
    assert.throws(() => selectCpuArchiver(compiler, path.join(f.root, "symlink-query.log")), /canonical regular executable/);
    writeCompiler("llvm-ar");
    assert.throws(() => selectCpuArchiver(compiler, path.join(f.root, "relative-query.log")), /absolute canonical/);
  });
});
