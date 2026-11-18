"use strict";

// CPU-only: compile the existing Pearl standard/compact sources, never dispatch their images.
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), crypto = require("node:crypto");
const {spawnSync} = require("node:child_process");
const {cudaHeaderResources} = require("../../scripts/native-jit-headers");
const {nativeDependencies, auditNativeTrace, auditNativeHeaders} = require("../../scripts/native-audit");

/** @param {string} command @param {string[]} args @param {string} log @param {NodeJS.ProcessEnv} [env] */
function run(command, args, log, env = {}) {
  const result = spawnSync(command, args, {encoding: "utf8", timeout: 180000, maxBuffer: 8 << 20,
    env: {...process.env, ...env}});
  fs.writeFileSync(log, (result.stdout || "") + (result.stderr || ""));
  if (result.error || result.signal || result.status) {throw new Error(`JIT header proof failed: ${log}`, {cause: result.error});}
  return result.stdout;
}

function main() {
  const [sdkArg, cudaArg, cutlassArg, ccclArg, outputArg] = process.argv.slice(2);
  if (process.platform !== "linux" || !sdkArg || !cudaArg || !cutlassArg || !ccclArg || process.argv.length > 7) {
    throw new Error("Usage (Linux): probe-native-jit-headers.js <compiler SDK> <CUDA include> <CUTLASS include> <CCCL include> [evidence parent]");
  }
  const sdk = fs.realpathSync(sdkArg), roots = [cutlassArg, cudaArg, ccclArg].map((file) => fs.realpathSync(file));
  const output = fs.mkdtempSync(path.join(path.resolve(outputArg || os.tmpdir()), "mom-jit-headers-"));
  console.log(`Memory-only JIT header evidence: ${output}`);
  const {source, ...headers} = cudaHeaderResources(roots);
  fs.writeFileSync(path.join(output, "nvrtc-headers.inc"), source);
  fs.writeFileSync(path.join(output, "headers.json"), JSON.stringify(headers, null, 2) + "\n");
  const repo = path.resolve(__dirname, "../.."), cxx = "/usr/lib/llvm-21/bin/clang++";
  const archives = ["nvrtc_static", "nvrtc-builtins_static", "nvptxcompiler_static"].map((name) => path.join(sdk, `lib64/lib${name}.a`));
  for (const file of archives) {if (!fs.existsSync(file)) {throw new Error("Missing compiler archive: " + file);}}
  const flags = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-ffunction-sections", "-fdata-sections",
    "-isystem", path.join(sdk, "include"), "-I", output, path.join(repo, "tests/reference/native-nvrtc-headers.cpp"),
    "--ld-path=/usr/lib/llvm-21/bin/ld.lld", "-Wl,--threads=4", "-Wl,--gc-sections",
    "-static-libstdc++", "-static-libgcc", "-ldl", "-lpthread", "-lm"];
  const executable = path.join(output, "nvrtc-headers"), oracle = path.join(output, "nvrtc-oracle");
  run(cxx, [...flags, "-DMOM_STATIC_NVRTC", "-Wl,-T," + path.join(repo, "runtime/static-constructors.ld"),
    "-Wl,--start-group", ...archives, "-Wl,-Bstatic", "-lz", "-Wl,-Bdynamic", "-Wl,--end-group", "-o", executable], path.join(output, "build.log"));
  run(cxx, [...flags, "-DMOM_NVRTC_DISK_HEADERS", "-L", path.join(sdk, "lib64"), "-lnvrtc", "-o", oracle], path.join(output, "oracle-build.log"));
  const actual = path.join(output, "actual"), expected = path.join(output, "expected"); fs.mkdirSync(actual); fs.mkdirSync(expected);
  run(executable, [actual], path.join(output, "compile.log"));
  run(oracle, [expected, ...headers.roots], path.join(output, "oracle.log"), {LD_LIBRARY_PATH: path.join(sdk, "lib64")});
  const files = fs.readdirSync(actual).map((name) => {
    const data = fs.readFileSync(path.join(actual, name));
    if (!data.equals(fs.readFileSync(path.join(expected, name)))) {throw new Error("Memory/disk Pearl compiler output differs: " + name);}
    return {name, bytes: data.length, sha256: crypto.createHash("sha256").update(data).digest("hex")};
  });
  if (files.length !== 5) {throw new Error("Incomplete Pearl compiler matrix");}
  const trace = path.join(output, "syscalls.log");
  run("strace", ["-f", "-s", "1024", "-e", "trace=openat,execve,execveat", "-E", "PATH=/unavailable", "-E", "LD_LIBRARY_PATH=/unavailable",
    "-o", trace, executable, actual], path.join(output, "isolated-paths.log"));
  const syscalls = fs.readFileSync(trace, "utf8"); auditNativeTrace(syscalls, executable, 1, {cudaDriver: true});
  // These cubins are this proof's generated outputs, not external compiler inputs.
  // Keep every SDK/source prohibition; exempt only the freshly created output directory.
  auditNativeHeaders(syscalls, headers.roots, [actual]);
  const needed = nativeDependencies(run("readelf", ["-d", executable], path.join(output, "dependencies.log")));
  const hash = (/** @type {string} */ file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({headers, roots, files, needed, bytes: fs.statSync(executable).size,
    sha256: hash(executable), archives: archives.map((file) => ({file, sha256: hash(file)})),
    sources: ["sycl/pearlhash/cuda-kernel.inc", "sycl/pearlhash/cuda-compact-kernel.inc"].map((file) => ({file, sha256: hash(path.join(repo, file))})),
    scope: "Maintained Pearl standard/compact sources, memory-only vs filesystem includes; byte-exact cubins. CPU compiler only, no GPU correctness/performance qualification."}, null, 2) + "\n");
  console.log("Maintained Pearl in-memory headers match filesystem compiler output; GPU dispatch remains unqualified.");
}

if (require.main === module) {try {main();} catch (error) {console.error(error); process.exitCode = 1;}}
