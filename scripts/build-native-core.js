"use strict";

// Release `core` stage of the standalone native build (spawned by build-native-release.js), run in the
// oneAPI development environment. Reuse its generated source/flag graph and unchanged CPU objects;
// never replace Engineering artifacts.
const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {fileHash, sourceInputs, nativeSourceInputs, verifySourceInputs, readBuildJsonSnapshot, compilerIncludePaths, isBuildIncludeDirectory} = require("./build-source-inputs");
const {compileNativeObjects} = require("./native-object-cache");

/** Reuse only a completed Engineering generation whose sources, recipes, objects, archives
 * and consumed SDK headers still match. Reading old Makefiles alone cannot prove cache freshness.
 * @param {string} repo @param {string} cache @param {boolean} [portable] */
function admitCpuCache(repo, cache, portable = false) {
  const receipt = path.join(cache, ".mom-content-state.json"), {value: state, sha256: generation} = readBuildJsonSnapshot(receipt);
  if (state.schema !== 1) {throw new Error("Missing completed Engineering cache receipt");}
  // An output/source hash says nothing about the build CPU. Release admission needs the
  // completed portable profile; do not bless legacy or host-tuned objects by adding metadata.
  if (portable && (state.portability?.portable !== true || state.portability.cpuMarch !== "")) {
    throw new Error("Engineering cache is not proven portable; rebuild it with MOM_PORTABLE_BUILD=1 and no MOM_CPU_MARCH");
  }
  verifySourceInputs({files: state.files, graph: state.graph}, sourceInputs(repo), "Engineering cache");
  for (const entries of [state.generated, state.links, Object.fromEntries(Object.entries(state.objects || {}).map(([name, value]) =>
    [name, /** @type {{content?: unknown}} */ (value)?.content]))]) {
    if (!entries || !Object.keys(entries).length) {throw new Error("Incomplete Engineering cache content receipt");}
    for (const [name, expected] of Object.entries(entries)) {
      if (path.isAbsolute(name) || /[\\\0]/.test(name) || name.split("/").some(part => !part || part === "." || part === "..") ||
          typeof expected !== "string" || !/^[0-9a-f]{64}$/.test(expected) || fileHash(path.join(cache, name)) !== expected) {
        throw new Error("Engineering cache content differs; rebuild it before native compilation");
      }
    }
  }
  if (!state.external || !Object.keys(state.external).length) {throw new Error("Missing Engineering SDK dependency receipt");}
  for (const [name, expected] of Object.entries(state.external)) {
    if (!path.isAbsolute(name) || fileHash(name) !== expected) {throw new Error("Engineering SDK dependency differs; rebuild this cache");}
  }
  if (fileHash(receipt) !== generation) {throw new Error("Engineering cache generation changed during admission");}
  return generation;
}

/** @param {string} command @param {string[]} args @param {string} log */
function run(command, args, log) {
  const result = spawnSync(command, args, {encoding: "utf8", timeout: 120000, maxBuffer: 8 << 20});
  fs.writeFileSync(log, (result.stdout || "") + (result.stderr || ""));
  if (result.error || result.signal || result.status !== 0) {throw new Error(`CPU bridge build failed: ${log}`, {cause: result.error});}
  return result.stdout;
}

/** Pin Intel's archiver instead of depending on an ambient (or missing) PATH entry.
 * @param {string} compiler @param {string} log */
function selectCpuArchiver(compiler, log) {
  const output = run(compiler, ["-print-prog-name=llvm-ar"], log);
  const selected = output.trim();
  if (!selected || selected.includes("\n") || selected.includes("\r") || !path.isAbsolute(selected) ||
      path.resolve(selected) !== selected) {throw new Error("Intel compiler did not select an absolute canonical llvm-ar");}
  return pinCpuArchiver(selected);
}

/** @param {string} executable */
function pinCpuArchiver(executable) {
  const link = fs.lstatSync(executable, {bigint: true});
  if (!link.isFile() || (link.mode & 0o111n) === 0n || fs.realpathSync(executable) !== executable) {
    throw new Error("Selected llvm-ar is not a canonical regular executable");
  }
  const hash = fileHash(executable);
  return {path: executable, sha256: hash, dev: String(link.dev), ino: String(link.ino), size: String(link.size),
    mtimeNs: String(link.mtimeNs), ctimeNs: String(link.ctimeNs)};
}

/** @param {ReturnType<typeof pinCpuArchiver>} expected */
function verifyCpuArchiver(expected) {
  const actual = pinCpuArchiver(expected.path);
  if (actual.path !== expected.path || actual.sha256 !== expected.sha256 || actual.dev !== expected.dev ||
      actual.ino !== expected.ino || actual.size !== expected.size || actual.mtimeNs !== expected.mtimeNs ||
      actual.ctimeNs !== expected.ctimeNs) {throw new Error("Selected llvm-ar changed during CPU archive creation");}
}

/** Compile mode-sensitive Core units with the exact generated flags, rather than reusing
 * Engineering objects whose N-API, allocation or OS-helper policy differs.
 * The optional stub source is a fresh output-local generated input, not a production algorithm.
 * @param {string} repo @param {string} output @param {string[]} flags @param {boolean} gpu @param {string} [compiler] */
function cpuCompileCommands(repo, output, flags, gpu, compiler = "icpx") {
  const sources = /** @type {Array<[string, string]>} */ (
    [["core", "native/core.cpp"], ["job", "native/job.cpp"], ["VirtualMemory", "xmrig/crypto/common/VirtualMemory.cpp"],
      ["Msr_linux", "xmrig/hw/msr/Msr_linux.cpp"]]);
  const commands = /** @type {Array<Parameters<typeof compileNativeObjects>[0]["commands"][number]>} */ (
    sources.map(([name, file]) => {
      const source = path.join(repo, file), object = path.join(output, `${name}.o`), depfile = path.join(output, `${name}.d`);
      return {command: compiler, source, object, depfile,
        args: [...flags, "-MD", "-MF", depfile, "-c", source, "-o", object], timeout: 120000};
    }));
  if (!gpu) {
    const source = path.join(output, "gpu-unavailable.cpp"), object = path.join(output, "gpu-unavailable.o"), depfile = path.join(output, "gpu-unavailable.d");
    commands.push({command: compiler, source, object, depfile, generatedInputs: [source],
      args: ["-std=c++20", "-O2", "-fPIC", "-I", repo, "-MD", "-MF", depfile, "-c", source, "-o", object], timeout: 120000});
  }
  return commands;
}

/** Include the bounded repository view plus every existing selected include directory.
 * Relative include flags resolve from the same repository cwd used by the cache runner.
 * @param {{args: string[]}[]} commands @param {string} repo @param {string} output */
function cpuIncludeRoots(commands, repo, output) {
  const roots = new Set([path.resolve(repo), path.resolve(output)]);
  for (const {args} of commands) {
    for (const selected of compilerIncludePaths(args)) {
      const root = path.resolve(repo, selected);
      if (isBuildIncludeDirectory(root)) {roots.add(root);}
    }
  }
  return [...roots];
}

/** Compile only the bridge-owned translation units; Engineering objects remain untouched.
 * @param {string} repo @param {string} output @param {ReturnType<typeof cpuCompileCommands>} commands
 * @param {boolean} gpu @param {NodeJS.ProcessEnv} [environment] */
function compileCpuBatch(repo, output, commands, gpu, environment = process.env) {
  return compileNativeObjects({repo, output, namespace: `linux/core/${gpu ? "gpu" : "cpu"}`, commands, environment,
    includeRoots: cpuIncludeRoots(commands, repo, output), jobs: 1, finalize: true,
    liveObjects: commands.map(command => command.object)});
}

function main() {
  const [cacheArg, outputArg, mode] = process.argv.slice(2);
  const gpu = mode === "--gpu";
  if (process.platform !== "linux" || !cacheArg || !outputArg || process.argv.length > 5 || (mode && !gpu)) {
    throw new Error("Usage: node scripts/build-native-core.js <oneAPI cache> <new output directory> [--gpu]");
  }
  const repo = path.resolve(__dirname, "..");
  const cache = path.resolve(cacheArg);
  const output = path.resolve(outputArg);
  const portable = process.env["MOM_PORTABLE_BUILD"] === "1";
  // The release descriptor selects Intel explicitly; do not rely on a parent's shell PATH.
  const compiler = process.env["CXX"] || "icpx";
  const sourceState = nativeSourceInputs(repo, "cpu"), cacheReceiptSha256 = admitCpuCache(repo, cache, portable);
  fs.mkdirSync(output); // Refuse to overwrite evidence or an existing build directory.
  const archiver = gpu ? selectCpuArchiver(compiler, path.join(output, "archiver.log")) : undefined;
  const makefile = fs.readFileSync(path.join(cache, "mom.target.mk"), "utf8").replace(/\\\r?\n/g, " ");
  /** @param {string} name */
  function variable(name) {
    const text = makefile.match(new RegExp(`^${name}\\s*:=([^\\n]*)`, "m"))?.[1];
    if (!text) {throw new Error(`Missing generated build variable: ${name}`);}
    return text.trim().split(/\s+/).map((value) => value.replace(/^'(.*)'$/, "$1").replaceAll("$(srcdir)", repo));
  }
  const objects = variable("OBJS").filter((name) => !/\/native\/(core|job)\.o$|\/(VirtualMemory|Msr_linux)\.o$/.test(name))
    .map((name) => path.join(cache, name.replace("$(obj)", "Release/obj").replace("$(TARGET)", "mom")));
  for (const object of objects) {if (!fs.existsSync(object)) {throw new Error(`Missing cached CPU object: ${object}`);}}
  const flags = [...variable("DEFS_Release"), ...variable("CFLAGS_Release"),
    ...variable("CFLAGS_CC_Release"), ...variable("INCS_Release"), "-DMOM_NATIVE_RUNTIME", "-fno-lto",
    "-ffunction-sections", "-fdata-sections"];
  // GPU entrypoints fail explicitly in this CPU-only proof. These are generated signatures,
  // not replacement algorithms, and must never enter the production build.
  if (!gpu) {
    const header = fs.readFileSync(path.join(repo, "sycl", "lib.h"), "utf8");
    const declarations = [...header.matchAll(/^MOM_SYCL_API ([\s\S]*?);/gm)].map((match) => match[1]);
    if (declarations.length < 20) {throw new Error("Incomplete GPU declaration graph");}
    const stubs = path.join(output, "gpu-unavailable.cpp");
    fs.writeFileSync(stubs, '#include "sycl/lib.h"\n#include <stdexcept>\n' + declarations.map((declaration) => {
      if (!declaration) {throw new Error("Empty GPU declaration");}
      return declaration.replace(/\s*=\s*nullptr/g, "") +
      (declaration.includes("sycl_cleanup()") ? " {}" : ' {throw std::runtime_error("GPU unavailable in CPU-only native bridge proof");}');
    }).join("\n"));
  }
  // XMRig's unused application-level VirtualMemory::init references an omitted MemoryPool source.
  // Keep it in its own section so the final application can discard it, not invent a stub allocator.
  const commands = cpuCompileCommands(repo, output, flags, gpu, compiler);
  const compiled = compileCpuBatch(repo, output, commands, gpu);
  const compileFailures = [];
  for (const [index, command] of commands.entries()) {
    const result = compiled.objects[index];
    if (!result) {throw new Error("Native CPU object cache returned an incomplete batch");}
    const label = path.basename(command.source, path.extname(command.source));
    fs.writeFileSync(path.join(output, `${label}.log`), (result.stdout || "") + (result.stderr || "") + (result.error ? result.error + "\n" : ""));
    fs.writeFileSync(path.join(output, `${label}.command.json`), JSON.stringify({command: command.command, args: command.args,
      cacheHit: result.cacheHit, status: result.status, signal: result.signal}) + "\n", {flag: "wx"});
    if (result.status !== 0 || result.error || result.signal) {compileFailures.push(`${label}.log`);}
  }
  if (compileFailures.length) {throw new Error("CPU bridge build failed: " + compileFailures.join(", "));}
  objects.push(...compiled.objects.map(result => result.object));
  const archives = ["argon2_x86_sse2", "argon2_x86_ssse3", "argon2_x86_xop", "argon2_x86_avx2",
    "argon2_x86_avx512f", "randomx_blake2b_sse41", "randomx_blake2b_avx2"]
    .map((name) => path.join(cache, "Release", `${name}.a`));
  for (const name of ["libirc.a", ...(gpu ? ["libsvml.a"] : [])]) {
    const runtime = run(compiler, [`-print-file-name=${name}`], path.join(output, `${name}.log`)).trim();
    if (!path.isAbsolute(runtime) || !fs.existsSync(runtime)) {throw new Error(`Static Intel CPU runtime is missing: ${name}`);}
    archives.push(runtime);
  }
  const materialized = [];
  if (gpu) {
    for (let index = 0; index < objects.length; ++index) {
      const input = objects[index];
      if (!input) {throw new Error("Missing CPU object");}
      const fd = fs.openSync(input, "r"), header = Buffer.alloc(4);
      try {fs.readSync(fd, header, 0, header.length, 0);} finally {fs.closeSync(fd);}
      const magic = header.toString("hex");
      if (magic !== "4243c0de" && magic !== "dec0170b") {continue;}
      // Engineering's Intel LTO IR is newer than the GPU LLVM reader. Materialize each module
      // with its original compiler, preserving target attributes and COMDAT groups. Do not feed
      // vendor bitcode to an incompatible LLVM linker or flatten it into one relocatable object.
      const object = path.join(output, `cpu-${index}.o`);
      run(compiler, ["-c", "-x", "ir", "-O3", "-fPIC", "-fno-lto", "-ffunction-sections", "-fdata-sections",
        input, "-o", object], path.join(output, `cpu-${index}.log`));
      objects[index] = object;
      materialized.push({input, object});
    }
  }
  const combined = path.join(output, gpu ? "cpu-core.a" : "cpu-core.o");
  // Keep original COMDAT groups for the GPU/LLVM linker. An Intel -r merge works in the old
  // CPU-only BFD proof but can retain relocations into groups discarded by LLD. Flatten archive
  // members, not object sections; this also lets the final linker discard unused CPU runtime code.
  if (gpu) {
    if (!archiver) {throw new Error("Intel-selected llvm-ar is missing");}
    verifyCpuArchiver(archiver);
    try {
      run(archiver.path, ["qcLs", combined, ...objects, ...archives], path.join(output, "combine.log"));
    } finally {
      verifyCpuArchiver(archiver);
    }
  } else {
    run(compiler, ["-r", "-nostdlib", "-flto", "-fPIC", ...objects, ...archives, "-o", combined], path.join(output, "combine.log"));
  }
  verifySourceInputs(sourceState, nativeSourceInputs(repo, "cpu"), "Native Core");
  if (cacheReceiptSha256 !== admitCpuCache(repo, cache, portable)) {throw new Error("Engineering cache generation changed during native compilation");}
  const objectCache = {cacheHits: compiled.cacheHits, cacheMisses: compiled.cacheMisses, cacheRoot: compiled.cacheRoot, ninja: compiled.ninja};
  fs.writeFileSync(path.join(output, "inputs.json"), JSON.stringify({objects, archives, flags, materialized, sourceState, objectCache,
    objectSha256: fileHash(combined), cacheReceiptSha256, ...(archiver ? {archiver: {path: archiver.path, sha256: archiver.sha256}} : {}),
    gpu, portable, scope: gpu ? "Real Core with unresolved GPU entrypoints; must link maintained SYCL objects" :
      "Real CPU Core; generated GPU entrypoints throw; not a production worker"}, null, 2) + "\n");
  console.log(combined);
}

module.exports = {admitCpuCache, cpuCompileCommands, cpuIncludeRoots, compileCpuBatch, selectCpuArchiver, pinCpuArchiver, verifyCpuArchiver};
if (require.main === module) {
  try {main();} catch (error) {console.error(error); process.exitCode = 1;}
}
