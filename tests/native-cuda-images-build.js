"use strict";

const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test");
const assert = require("node:assert/strict");
const {fileHash} = require("../scripts/build-source-inputs");
const {cudaHeaderResources} = require("../scripts/native-jit-headers");
const {buildCudaImages, validateConfig} = require("../scripts/build-native-cuda-images");
const {cudaImageInputs} = require("../scripts/native-cuda-images");

const repo = path.resolve(__dirname, "..");
const exportsByProfile = {
  "cn-expand": ["cn_gpu_expand_cuda"], "pearl-rank128": ["pearlhash_cuda_gemm", "pearlhash_cuda_hash"],
  "pearl-rank256": ["pearlhash_cuda_gemm", "pearlhash_cuda_hash"], "pearl-compact": ["compact_gemm", "selector_prepare"],
};

/** @param {import("node:test").TestContext} t */
function tempDirectory(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cuda-image-build-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return root;
}

/** @param {string} file @param {string | Buffer} data @param {number} [mode] */
function write(file, data, mode) {
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, data, {mode, flag: "wx"});
}

/** @param {string} root */
function makeHeaderPack(root) {
  const cutlass = path.join(root, "cutlass"), cuda = path.join(root, "cuda");
  write(path.join(cutlass, "cute/tensor.hpp"), "// CUTE fixture\n");
  write(path.join(cuda, "cuda_runtime.h"), "// CUDA runtime fixture\n");
  write(path.join(cuda, "cccl/cuda/std/cstdint"), "// CCCL fixture\n");
  const file = path.join(root, "nvrtc-headers.inc"), {source, ...manifest} = cudaHeaderResources([cutlass, cuda]);
  write(file, source); write(file + ".json", JSON.stringify(manifest, null, 2) + "\n");
  return {file, roots: manifest.roots};
}

/** Build the bounded ELF64 symbol fixture used by the fake worker.
 * @param {string[]} names @returns {Buffer} */
function cubin(names) {
  const chunks = [Buffer.from([0])], offsets = new Map();
  let stringBytes = 1;
  for (const name of names) {
    offsets.set(name, stringBytes);
    const encoded = Buffer.from(name + "\0"); chunks.push(encoded); stringBytes += encoded.length;
  }
  const strings = Buffer.concat(chunks), symbols = Buffer.alloc((names.length + 1) * 24);
  for (const [index, name] of names.entries()) {
    const offset = (index + 1) * 24;
    symbols.writeUInt32LE(offsets.get(name) || 0, offset); symbols[offset + 4] = 0x12; symbols.writeUInt16LE(1, offset + 6);
  }
  const sectionOffset = 64, count = 4, textOffset = sectionOffset + count * 64, stringsOffset = textOffset + 1;
  const symbolsOffset = stringsOffset + strings.length, data = Buffer.alloc(symbolsOffset + symbols.length);
  data.set([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1]);
  data.writeUInt16LE(2, 16); data.writeUInt16LE(190, 18); data.writeUInt32LE(1, 20);
  data.writeBigUInt64LE(BigInt(sectionOffset), 40); data.writeUInt16LE(64, 52);
  data.writeUInt16LE(64, 58); data.writeUInt16LE(count, 60);
  /** @param {number} index @param {number} type @param {number} offset @param {number} size @param {number} [link] @param {number} [entrySize] */
  const section = (index, type, offset, size, link = 0, entrySize = 0) => {
    const at = sectionOffset + index * 64; data.writeUInt32LE(type, at + 4);
    data.writeBigUInt64LE(BigInt(offset), at + 24); data.writeBigUInt64LE(BigInt(size), at + 32);
    data.writeUInt32LE(link, at + 40); data.writeBigUInt64LE(BigInt(entrySize), at + 56);
  };
  section(1, 1, textOffset, 1); section(2, 3, stringsOffset, strings.length);
  section(3, 2, symbolsOffset, symbols.length, 2, 24);
  data[textOffset] = 0; strings.copy(data, stringsOffset); symbols.copy(data, symbolsOffset);
  return data;
}

/** @param {string} version */
function versionLiteral(version) {return JSON.stringify(version);}

/** Create a fake public-API worker with optional deterministic failures, caps and mutation hooks.
 * @param {string} root @param {{major: number, minor: number, supportedArchs: number[]}} compiler
 * @param {{failTargets?: string[], mutateTarget?: string, mutateFile?: string, oversizeTarget?: string, hangQuery?: boolean, flatQuery?: boolean}} [mode] */
function fakeToolchain(root, compiler, mode = {}) {
  if (process.platform === "win32") {throw new Error("The fake native-worker fixture uses POSIX shebang execution");}
  const sdk = path.join(root, `sdk-${compiler.major}.${compiler.minor}`), headers = makeHeaderPack(path.join(sdk, "headers"));
  const inputFiles = [];
  for (const name of ["nvrtc.h", "libnvrtc.so", "libnvrtc-builtins.so", "libnvptxcompiler.so", "libdevice.10.bc"]) {
    const file = path.join(sdk, name); write(file, `${compiler.major}.${compiler.minor}:${name}\n`); inputFiles.push(file);
  }
  const template = path.join(sdk, "worker-template.js"), shim = path.join(sdk, "fake-cxx.js");
  const workerSource = [
    "#!/usr/bin/env node", '"use strict";',
    'const fs=require("node:fs"),path=require("node:path");',
    `const compiler=${JSON.stringify(compiler)}, failTargets=new Set(${JSON.stringify(mode.failTargets || [])});`,
    `const mutateTarget=${JSON.stringify(mode.mutateTarget || "")}, mutateFile=${JSON.stringify(mode.mutateFile || "")};`,
    `const oversizeTarget=${JSON.stringify(mode.oversizeTarget || "")}, hangQuery=${Boolean(mode.hangQuery)}, flatQuery=${Boolean(mode.flatQuery)};`,
    `const exportsByProfile=${JSON.stringify(exportsByProfile)};`,
    cubin.toString(),
    "const args=process.argv.slice(2);",
    'if(args.length===1&&args[0]==="--query"){if(hangQuery){setInterval(()=>{},60000);}else{const report=flatQuery?{status:"ok",...compiler}:{status:"ok",nvrtc:{major:compiler.major,minor:compiler.minor},supportedArchs:compiler.supportedArchs};process.stdout.write(JSON.stringify(report)+"\\n");process.exit(0);}}',
    'if(args.length!==1||args[0]!=="--query"){',
    "const [profile,target,format,output]=args,includeRoots=[];",
    'for(let i=4;i<args.length;i++){if(args[i]!=="--include"||!args[i+1]){process.stderr.write("bad repeated include args\\n");process.exit(4);}includeRoots.push(args[++i]);}',
    'const names=exportsByProfile[profile];if(!names){process.stderr.write("unknown profile\\n");process.exit(5);}',
    'if(failTargets.has(target)){process.stderr.write("fixture failure "+target+"\\n");process.exit(13);}',
    'if(target===mutateTarget&&mutateFile){fs.appendFileSync(mutateFile,"changed\\n");}',
    'const arch=Number(target.match(/\\d+/)[0]),ptxVersion=format==="ptx"?(compiler.major===12?"8.5":"9.3"):null;',
    'const options=["--std=c++17"];if(profile==="pearl-rank128"||profile==="pearl-compact")options.push("-DPEARLHASH_RANK=128");',
    'if(profile==="pearl-rank256")options.push("-DPEARLHASH_RANK=256");options.push("--gpu-architecture="+target);',
    'for(const root of includeRoots)options.push("--include-path="+root);options.push("--device-as-default-execution-space");',
    'if(profile!=="cn-expand")options.push("--use_fast_math");if(profile==="pearl-compact")options.push("-DDIRECT_RANK_STRIDE=1");',
    'let image;if(target===oversizeTarget){const fd=fs.openSync(output,"wx");fs.ftruncateSync(fd,67108865);fs.closeSync(fd);image=null;}',
    'else if(format==="ptx"){const text=["// fixture",".version "+ptxVersion,".target sm_"+arch,".address_size 64",...names.map(name=>".visible .entry "+name+"() { ret; }")].join("\\n")+"\\n\\0";image=Buffer.from(text,"ascii");fs.writeFileSync(output,image,{flag:"wx"});}',
    'else{image=cubin(names);fs.writeFileSync(output,image,{flag:"wx"});}',
    'const size=fs.statSync(output).size,metadata={status:"ok",nvrtc:{major:compiler.major,minor:compiler.minor},supportedArchs:compiler.supportedArchs,',
    'profile,source:profile==="cn-expand"?"sycl/cn-gpu/cuda-expand-kernel.inc":profile==="pearl-compact"?"sycl/pearlhash/cuda-compact-kernel.inc":"sycl/pearlhash/cuda-kernel.inc",',
    'program:profile==="cn-expand"?"cn-gpu-expand.cu":"pearlhash-cute.cu",target,format,ptxVersion,includeRoots,options,exports:names,outputBytes:size};',
    'process.stdout.write(JSON.stringify(metadata)+"\\n");',
    "}",
  ].join("\n");
  write(template, workerSource, 0o700);
  const shimSource = [
    "#!/usr/bin/env node", '"use strict";', 'const fs=require("node:fs");',
    'const args=process.argv.slice(2),at=args.indexOf("-o");if(at<0||!args[at+1])process.exit(8);',
    `fs.copyFileSync(${versionLiteral(template)},args[at+1]);fs.chmodSync(args[at+1],0o700);`,
  ].join("\n");
  write(shim, shimSource, 0o700);
  return {headers, template, shim, inputFiles};
}

/** @param {string} root @param {{major: number, minor: number, supportedArchs: number[]}} compiler
 * @param {Parameters<typeof fakeToolchain>[2]} [mode] */
function recipe(root, compiler, mode) {
  const fixture = fakeToolchain(root, compiler, mode);
  return {fixture, value: {headerPack: fixture.headers.file, worker: {command: fixture.shim,
    args: ["-std=c++20", "{source}", "-o", "{output}"],
    inputs: Object.fromEntries([fixture.template, ...fixture.inputFiles].map((file) => [file, fileHash(file)]))}}};
}

/** @param {string} root @param {{major: number, minor: number, supportedArchs: number[]}} oldCompiler
 * @param {{major: number, minor: number, supportedArchs: number[]}} currentCompiler
 * @param {Parameters<typeof fakeToolchain>[2]} [mode] */
function buildConfig(root, oldCompiler, currentCompiler, mode) {
  const old = recipe(root, oldCompiler, mode), current = recipe(root, currentCompiler, mode);
  return {root, config: {repo, output: path.join(root, "bundle"), compilers: [old.value, current.value]}};
}

const baselineCompiler = {major: 12, minor: 6, supportedArchs: [80, 86, 90]};
const currentCompiler = {major: 13, minor: 3, supportedArchs: [80, 86, 90, 100, 120]};

test.describe("Build-only native CUDA image bundle", () => {
  test("validates trusted explicit recipes before creating output", (t) => {
    const root = tempDirectory(t), config = {repo, output: path.join(root, "bundle"), compilers: [{
      headerPack: path.join(root, "sdk-headers.inc"), worker: {command: path.join(root, "icpx"),
        args: ["-std=c++20", "{source}", "-o", "{output}"], inputs: {[path.join(root, "libnvrtc.so")]: "0".repeat(64)}}}],
    };
    assert.throws(() => validateConfig({...config, output: "relative"}), /absolute/);
    assert.throws(() => validateConfig({...config, compilers: []}), /1–16/);
    assert.throws(() => validateConfig({...config, compilers: [{...config.compilers[0], worker: {...config.compilers[0]?.worker,
      args: ["{source}", "-o", "{output}"]}}]}), /C\+\+20/);
    assert.ok(!fs.existsSync(config.output));
  });

  test("modern-only query cannot create a release bundle without the 12.6 compatibility baseline", {skip: process.platform === "win32"}, (t) => {
    const root = tempDirectory(t), onlyModern = recipe(root, currentCompiler);
    const output = path.join(root, "modern-only-bundle"), config = {repo, output, compilers: [onlyModern.value]};
    assert.throws(() => buildCudaImages(config), /actual NVRTC 12\.6 compute_80/);
    assert.ok(!fs.existsSync(path.join(output, "cuda-images.inc")));
    assert.ok(fs.existsSync(path.join(output, "compiler-00", "query.log")));
  });

  test("query admission accepts only the worker's nested public NVRTC record", {skip: process.platform === "win32"}, (t) => {
    const root = tempDirectory(t), onlyModern = recipe(root, currentCompiler, {flatQuery: true});
    const output = path.join(root, "flat-query-bundle"), config = {repo, output, compilers: [onlyModern.value]};
    assert.throws(() => buildCudaImages(config), /valid public compiler query/);
    assert.ok(fs.existsSync(path.join(output, "compiler-00", "query.log")));
    assert.ok(!fs.existsSync(path.join(output, "cuda-images.inc")));
  });

  test("fake 12.6/13.3 workers query the public API, compile the entire bounded plan, and preserve provenance", {skip: process.platform === "win32"}, (t) => {
    const root = tempDirectory(t), {config} = buildConfig(root, baselineCompiler, currentCompiler);
    const result = buildCudaImages(config), admitted = cudaImageInputs(result.include, repo);
    const manifest = /** @type {{images: Array<{profile: string, target: string, compiler: {major: number}, ptxVersion: string | null, compatibility: boolean}>,
   * compilers: Array<{inputs: Record<string, string>}>}} */ (/** @type {unknown} */ (admitted.manifest));
    assert.equal(result.include, path.join(config.output, "cuda-images.inc"));
    assert.equal(result.sidecar, path.join(config.output, "cuda-images.inc.json"));
    assert.equal(manifest.images.length, 34);
    assert.ok(manifest.images.some((image) => image.profile === "pearl-rank128" && image.target === "compute_80" &&
    image.compiler.major === 12 && image.ptxVersion === "8.5" && image.compatibility));
    assert.ok(manifest.images.some((image) => image.profile === "pearl-rank128" && image.target === "sm_100" && image.compiler.major === 13));
    assert.ok(manifest.images.some((image) => image.profile === "pearl-compact" && image.target === "sm_120a"));
    for (const compiler of manifest.compilers) {
      assert.ok(Object.keys(compiler.inputs).some((file) => file.endsWith("native-cuda-images")));
      assert.ok(Object.keys(compiler.inputs).some((file) => file.endsWith("recipe.json")));
      assert.ok(Object.keys(compiler.inputs).some((file) => file.endsWith("libnvrtc-builtins.so")));
    }
    assert.ok(fs.existsSync(path.join(config.output, "compiler-00", "query.log")));
    assert.ok(fs.existsSync(path.join(config.output, "compiler-01", "recipe.json")));
    assert.ok(fs.existsSync(path.join(config.output, "result.json")));
    const oldHash = fileHash(result.include);
    assert.throws(() => buildCudaImages(config), /already exists/);
    assert.equal(fileHash(result.include), oldHash);
  });

  test("canonicalizes a compiler alias for both execution and provenance", {skip: process.platform === "win32"}, (t) => {
    const root = tempDirectory(t), {config} = buildConfig(root, baselineCompiler, currentCompiler);
    const recipe = config.compilers[0];
    assert.ok(recipe);
    const alias = path.join(root, "compiler-alias");
    fs.symlinkSync(recipe.worker.command, alias);
    recipe.worker.command = alias;

    buildCudaImages(config);
    const resolved = fs.realpathSync(alias);
    const record = JSON.parse(fs.readFileSync(path.join(config.output, "compiler-00", "recipe.json"), "utf8"));
    const buildCommand = JSON.parse(fs.readFileSync(path.join(config.output, "compiler-00", "build.command.json"), "utf8"));
    assert.equal(record.command, resolved);
    assert.equal(buildCommand.command, resolved);
    assert.equal(record.inputs[resolved], fileHash(resolved));
  });

  test("ordinary image failures are collected across the plan and preserve worker evidence", {skip: process.platform === "win32"}, (t) => {
    const root = tempDirectory(t), failing = {...currentCompiler, supportedArchs: [...currentCompiler.supportedArchs]};
    const {config} = buildConfig(root, baselineCompiler, failing, {failTargets: ["sm_86", "compute_100"]});
    assert.throws(() => buildCudaImages(config), /CUDA image compilation failed[\s\S]*sm_86[\s\S]*compute_100/);
    assert.ok(fs.existsSync(path.join(config.output, "failures.json")));
    assert.ok(fs.existsSync(path.join(config.output, "compiler-01", "image-0007.command.json")));
    assert.ok(!fs.existsSync(path.join(config.output, "cuda-images.inc")));
  });

  test("input drift and image-size caps fail closed without publishing a bundle", {skip: process.platform === "win32"}, (t) => {
    const changedRoot = tempDirectory(t), mutation = path.join(changedRoot, "sdk-13.3", "libnvrtc.so");
    const changedConfig = buildConfig(changedRoot, baselineCompiler, currentCompiler, {mutateTarget: "compute_100", mutateFile: mutation}).config;
    assert.throws(() => buildCudaImages(changedConfig), /changed during CUDA image production/);
    assert.ok(!fs.existsSync(path.join(changedConfig.output, "cuda-images.inc")));

    const cappedRoot = tempDirectory(t), {config} = buildConfig(cappedRoot, baselineCompiler, currentCompiler, {oversizeTarget: "sm_80"});
    assert.throws(() => buildCudaImages(config), /64 MiB image data bound/);
    assert.equal(fs.statSync(path.join(config.output, "compiler-00", "images", "0000.cubin")).size, 67108865);
    assert.ok(!fs.existsSync(path.join(config.output, "cuda-images.inc")));
  });

  test("unresolved worker timeout stops later launches but leaves command and log evidence", {skip: process.platform === "win32"}, (t) => {
    const root = tempDirectory(t), {config} = buildConfig(root, baselineCompiler, currentCompiler, {hangQuery: true});
    const bounded = {...config, timeoutMs: 1000};
    assert.throws(() => buildCudaImages(bounded), /Unresolved CUDA worker lifecycle/);
    assert.ok(fs.existsSync(path.join(config.output, "compiler-00", "query.command.json")));
    assert.ok(fs.existsSync(path.join(config.output, "compiler-00", "query.log")));
    assert.ok(!fs.existsSync(path.join(config.output, "compiler-01")));
  });

});
