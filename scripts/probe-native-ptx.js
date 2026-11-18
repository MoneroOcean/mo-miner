"use strict";

// Opt-in, CPU-only real AdaptiveCpp generic lowering + in-process PTX proof, not GPU dispatch.
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const resources = require("./native-resources");
const {nativeDependencies, auditNativeTrace} = require("./native-audit");

/** @param {string} command @param {string[]} args @param {string} log @param {number} [status] */
function run(command, args, log, status = 0) {
  const result = spawnSync(command, args, {encoding: "utf8", timeout: 120000, maxBuffer: 4 << 20});
  fs.writeFileSync(log, (result.stdout || "") + (result.stderr || ""));
  if (result.error || result.signal || result.status !== status) {throw new Error(`Native PTX proof failed; see ${log}`, {cause: result.error});}
  return result.stdout;
}

function main() {
  const [sourceArg, buildArg, outputArg] = process.argv.slice(2);
  if (process.platform !== "linux" || !sourceArg || !buildArg || process.argv.length > 5) {
    throw new Error("Usage (Linux): node scripts/probe-native-ptx.js <patched AdaptiveCpp> <static JIT build> [evidence-parent]");
  }
  const source = path.resolve(sourceArg), build = path.resolve(buildArg), repo = path.resolve(__dirname, "..");
  const output = fs.mkdtempSync(path.join(path.resolve(outputArg || os.tmpdir()), "mom-ptx-"));
  console.log(`In-process PTX evidence: ${output}`);
  const revision = "da2463e45aa90aa36306c45abcfc05b87de51bc6";
  const canonical = (/** @type {string} */ patch) => patch.replace(/^ $/gm, "").split(/(?=^diff --git )/m).filter(Boolean).sort().join("");
  if (run("git", ["-C", source, "rev-parse", "HEAD"], path.join(output, "revision.log")).trim() !== revision ||
      canonical(run("git", ["-C", source, "diff", "--no-ext-diff", "--no-textconv", "HEAD"], path.join(output, "patch.log"))) !==
      canonical(["adaptivecpp-static.patch", "adaptivecpp-jit.patch"].map((file) => fs.readFileSync(path.join(repo, "runtime", file), "utf8")).join(""))) {
    throw new Error("Use the pinned AdaptiveCpp source with only the reviewed static/JIT patches");
  }
  // Automatic regeneration is an input-preservation check too: a custom SSCP-only profile
  // must not silently turn stdpar/accelerated CPU back on when no flags are repeated.
  run("cmake", ["-S", source, "-B", build], path.join(output, "regenerate.log"));
  const cache = fs.readFileSync(path.join(build, "CMakeCache.txt"), "utf8");
  const configuredSource = cache.match(/^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$/m)?.[1];
  if (!/^ACPP_STATIC_RUNTIME:BOOL=(?:ON|TRUE|1)$/mi.test(cache) || !/^ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER:INTERNAL=(?:ON|TRUE|1)$/mi.test(cache) ||
      ["STDPAR_COMPILER", "ACCELERATED_CPU"].some((feature) => !new RegExp(`^ACPP_CUSTOM_PROFILE_WITH_${feature}:INTERNAL=(?:OFF|FALSE|0)$`, "mi").test(cache)) ||
      !configuredSource || fs.realpathSync(configuredSource) !== fs.realpathSync(source)) {
    throw new Error("Use a static AdaptiveCpp SSCP build configured from the reviewed source");
  }
  const generator = cache.match(/^ACPP_RESOURCE_GENERATOR:FILEPATH=(.+)$/m)?.[1];
  if (!generator || fs.realpathSync(generator) !== fs.realpathSync(path.join(repo, "scripts/native-resources.js"))) {
    throw new Error("Static PTX must use the reviewed build-only resource generator");
  }
  run("cmake", ["--build", build, "--target", "llvm-to-ptx", "acpp-clang", "libkernel-sscp-ptx", "--parallel", "4"], path.join(output, "incremental-build.log"));
  const config = process.env["LLVM_CONFIG"] || "/usr/lib/llvm-21/bin/llvm-config";
  const version = run(config, ["--version"], path.join(output, "llvm-version.log")).trim();
  if (!version.startsWith("21.")) {throw new Error("The reviewed native PTX proof requires LLVM 21");}
  const tools = run(config, ["--bindir"], path.join(output, "llvm-tools.log")).trim();
  const include = run(config, ["--includedir"], path.join(output, "llvm-include.log")).trim();
  const library = run(config, ["--libdir"], path.join(output, "llvm-lib.log")).trim();
  const libs = run(config, ["--link-static", "--libs", "core", "support", "irreader", "passes", "nvptxcodegen"],
    path.join(output, "llvm-components.log")).trim().split(/\s+/);
  const clang = path.join(tools, "clang++"), plugin = path.join(build, "src/compiler/libacpp-clang.so");
  const includes = ["-isystem", path.join(source, "include"), "-isystem", path.join(build, "include")];
  for (const [file, mode] of [["native-ptx-kernel.cpp", "device"], ["native-sscp-kernel.cpp", "generic"]]) {
    if (!file || !mode) {throw new Error("Missing kernel fixture");}
    const input = path.join(output, file);
    fs.writeFileSync(input, fs.readFileSync(path.join(repo, "tests/reference", file), "utf8")
      .replace('"../../sycl/blake3-device.inc"', JSON.stringify(path.join(repo, "sycl/blake3-device.inc"))));
    const flags = mode === "device" ? ["-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib", "--cuda-gpu-arch=sm_61", "-S", "-emit-llvm"] :
      ["-D__ACPP__", "-D__ACPP_ENABLE_LLVM_SSCP_TARGET__", ...includes, "-include", path.join(source, "include/hipSYCL/glue/llvm-sscp/hcf_registration.hpp"),
        `-fplugin=${plugin}`, `-fpass-plugin=${plugin}`, "-Xclang", "-disable-O0-optnone", "-mllvm", "-acpp-sscp", "-mllvm", "-acpp-sscp-emit-hcf", "-c"];
    run(clang, ["-std=c++17", "-O2", ...flags, input, "-o", path.join(output, mode === "device" ? "device.ll" : "generic.o")], path.join(output, `${mode}-compile.log`));
  }
  const libdevice = cache.match(/^CUDA_DEVICE_LIBS_PATH:PATH=(.+)$/m)?.[1];
  if (!libdevice) {throw new Error("Build cache has no pinned CUDA device library path");}
  const inputs = [{name: "device.ll", file: path.join(output, "device.ll")},
    {name: "generic.hcf", file: path.join(output, "native-sscp-kernel.cpp.hcf")},
    {name: "libdevice.10.bc", file: path.join(libdevice, "libdevice.10.bc")},
    {name: "libkernel-sscp-ptx-full.bc", file: path.join(build, "src/libkernel/sscp/ptx/libkernel-sscp-ptx-full.bc")}]
    .map(({name, file}) => ({name, data: fs.readFileSync(file)}));
  fs.writeFileSync(path.join(output, "ptx-resources.inc"), resources(inputs));
  const provenance = inputs.map(({name, data}) => ({name, bytes: data.length, sha256: createHash("sha256").update(data).digest("hex")}));
  const executable = path.join(output, "ptx-native");
  run(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-ffunction-sections", "-fdata-sections", "-DHIPSYCL_COMPILER_COMPONENT", "-DACPP_STATIC_RUNTIME",
    ...includes, "-isystem", include, "-I", output, path.join(repo, "tests/reference/native-ptx-translator.cpp"), path.join(repo, "runtime/compiler-gate.cpp"),
    "-Wl,--gc-sections", "-static-libstdc++", "-static-libgcc",
    "-Wl,--start-group", ...["llvm-to-ptx", "llvm-to-backend"].map((name) => path.join(build, "src/compiler/llvm-to-backend", `lib${name}.a`)),
    path.join(build, "src/common/libacpp-common.a"), "-L", library, ...libs, "-Wl,--end-group", "-Wl,-Bstatic", "-lz", "-lzstd", "-Wl,-Bdynamic", "-lm", "-ldl", "-lpthread", "-o", executable], path.join(output, "link.log"));
  run(executable, [output], path.join(output, "compile.log"));
  run(executable, [], path.join(output, "missing-arguments.log"), 2);
  run(executable, [path.join(output, "absent", "directory")], path.join(output, "missing-output.log"), 1);
  if (!/Cannot preserve pre-emission IR/.test(fs.readFileSync(path.join(output, "missing-output.log"), "utf8"))) {
    throw new Error("Missing-output case failed for an unexpected reason");
  }
  // Build-time oracle only. The isolated executable below must never invoke these tools.
  // Feed both pipelines identical pre-emission bitcode, including legacy kernel metadata.
  const hashes = [];
  for (const full of [false, true]) {
    for (const fast of [false, true]) {
      for (const [sm, ptx] of [[61, 64], [120, 87]]) {
        const base = path.join(output, `${full ? "generic-" : ""}sm_${sm}${fast ? "-fast" : ""}`);
        run(path.join(tools, "opt"), [`--mcpu=sm_${sm}`, "-O3", base + ".bc", "-o", base + "-oracle.bc"], base + "-opt.log");
        const flags = fast ? ["--enable-unsafe-fp-math", "--enable-no-infs-fp-math", "--enable-no-nans-fp-math",
          "--enable-no-signed-zeros-fp-math", "--enable-no-trapping-fp-math"] : [];
        run(path.join(tools, "llc"), ["--mtriple=nvptx64-nvidia-cuda", "--march=nvptx64", "--frame-pointer=none",
          `--mattr=+ptx${ptx}`, `--mcpu=sm_${sm}`, "-O3", "-o", base + "-oracle.ptx", base + "-oracle.bc", ...flags], base + "-llc.log");
        const actual = fs.readFileSync(base + ".ptx"), expected = fs.readFileSync(base + "-oracle.ptx");
        hashes.push({sm, ptx, genericSycl: full, fastMath: fast, matches: actual.equals(expected),
          sha256: createHash("sha256").update(actual).digest("hex"), expected: createHash("sha256").update(expected).digest("hex")});
      }
    }
  }
  fs.writeFileSync(path.join(output, "oracle.json"), JSON.stringify(hashes, null, 2) + "\n");
  if (hashes.some((entry) => !entry.matches)) {throw new Error(`In-process PTX differs from opt/llc oracle; see ${path.join(output, "oracle.json")}`);}
  for (const sm of [61, 120]) {
    const variants = hashes.filter((entry) => entry.sm === sm && entry.genericSycl);
    if (variants.length !== 2 || variants[0]?.sha256 === variants[1]?.sha256) {throw new Error("Device-library fast-math policy had no effect");}
  }
  const needed = nativeDependencies(run("readelf", ["-d", executable], path.join(output, "dependencies.log")));
  const trace = path.join(output, "syscalls.log");
  run("strace", ["-f", "-s", "1024", "-e", "trace=openat,execve,execveat", "-E", "PATH=/unavailable", "-E", "LD_LIBRARY_PATH=/unavailable", "-o", trace, executable, output], path.join(output, "isolated-paths.log"));
  auditNativeTrace(fs.readFileSync(trace, "utf8"), executable);
  if (fs.readFileSync(trace, "utf8").split("\n").some((line) =>
    /openat\([^\n]*"[^"\n]*\.(?:bc|hcf|ll)"/.test(line) && !/", O_WRONLY\b/.test(line))) {
    throw new Error("Native JIT read external compiler resources");
  }
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({revision, version, needed, bytes: fs.statSync(executable).size, resources: provenance,
    targets: [61, 120], genericSycl: true, embeddedLibraries: true, oracle: hashes,
    scope: "CPU-only generic SYCL lowering/PTX generation; GPU execution/performance unqualified"}, null, 2) + "\n");
  console.log("Static AdaptiveCpp generic SYCL/PTX proof passed without external compiler or resource files.");
}

if (require.main === module) {try {main();} catch (error) {console.error(error); process.exitCode = 1;}}
