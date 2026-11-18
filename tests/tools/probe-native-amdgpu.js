"use strict";

// Opt-in CPU-only qualification of real generic SYCL AMD lowering; never discovers a GPU.
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const resources = require("../../scripts/native-resources");
const {lldInputs} = require("../../scripts/native-jit-link");
const {nativeDependencies, auditNativeTrace, auditNativeHeaders} = require("../../scripts/native-audit");

/** @param {string} command @param {string[]} args @param {string} log @param {number} [status] */
function run(command, args, log, status = 0) {
  const result = spawnSync(command, args, {encoding: "utf8", timeout: 180000, maxBuffer: 8 << 20});
  fs.writeFileSync(log, (result.stdout || "") + (result.stderr || ""));
  if (result.error || result.signal || result.status !== status) {throw new Error(`Native AMD proof failed; see ${log}`, {cause: result.error});}
  return result.stdout;
}

function main() {
  const [sourceArg, buildArg, outputArg] = process.argv.slice(2);
  if (process.platform !== "linux" || !sourceArg || !buildArg || process.argv.length > 5) {
    throw new Error("Usage (Linux): probe-native-amdgpu.js <patched AdaptiveCpp> <static AMD JIT build> [evidence-parent]");
  }
  const repo = path.resolve(__dirname, "../.."), source = path.resolve(sourceArg), build = path.resolve(buildArg);
  const output = fs.mkdtempSync(path.join(path.resolve(outputArg || os.tmpdir()), "mom-amdgpu-"));
  console.log(`In-process AMD SYCL evidence: ${output}`);
  const canonical = (/** @type {string} */ patch) => patch.replace(/^ $/gm, "").split(/(?=^diff --git )/m).filter(Boolean).sort().join("");
  if (run("git", ["-C", source, "rev-parse", "HEAD"], path.join(output, "revision.log")).trim() !== "da2463e45aa90aa36306c45abcfc05b87de51bc6" ||
      canonical(run("git", ["-C", source, "diff", "--no-ext-diff", "--no-textconv", "HEAD"], path.join(output, "patch.log"))) !==
      canonical(["adaptivecpp-static.patch", "adaptivecpp-jit.patch"].map((file) => fs.readFileSync(path.join(repo, "runtime", file), "utf8")).join(""))) {
    throw new Error("Use the pinned AdaptiveCpp source with only the reviewed static/JIT patches");
  }
  run("cmake", ["-S", source, "-B", build], path.join(output, "regenerate.log"));
  const cache = fs.readFileSync(path.join(build, "CMakeCache.txt"), "utf8");
  /** @param {string} name */
  const setting = (name) => cache.match(new RegExp(`^${name}:[^=]+=(.+)$`, "m"))?.[1];
  if (setting("ACPP_STATIC_RUNTIME") !== "ON" || setting("WITH_ROCM_BACKEND") !== "ON" ||
      setting("ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER") !== "ON" ||
      ["STDPAR_COMPILER", "ACCELERATED_CPU"].some((name) => setting("ACPP_CUSTOM_PROFILE_WITH_" + name) !== "OFF") ||
      fs.realpathSync(setting("CMAKE_HOME_DIRECTORY") || "/unavailable") !== fs.realpathSync(source) ||
      fs.realpathSync(setting("ACPP_RESOURCE_GENERATOR") || "/unavailable") !== fs.realpathSync(path.join(repo, "scripts/native-resources.js"))) {
    throw new Error("Use a static AMD SSCP build and the reviewed resource generator");
  }
  run("cmake", ["--build", build, "--target", "llvm-to-amdgpu", "acpp-clang", "--parallel", "4"], path.join(output, "incremental-build.log"));
  const config = process.env["LLVM_CONFIG"] || "/usr/lib/llvm-21/bin/llvm-config";
  const version = run(config, ["--version"], path.join(output, "llvm-version.log")).trim();
  if (version !== "21.1.8") {throw new Error("The reviewed native AMD proof requires LLVM 21.1.8");}
  const tool = run(config, ["--bindir"], path.join(output, "llvm-tools.log")).trim();
  const include = run(config, ["--includedir"], path.join(output, "llvm-include.log")).trim();
  const library = run(config, ["--libdir"], path.join(output, "llvm-lib.log")).trim();
  const libs = run(config, ["--link-static", "--libs", "all"], path.join(output, "llvm-libraries.log")).trim().split(/\s+/);
  const includes = ["-isystem", path.join(source, "include"), "-isystem", path.join(build, "include")];
  const input = path.join(output, "native-sscp-kernel.cpp");
  fs.writeFileSync(input, fs.readFileSync(path.join(repo, "tests/reference/native-sscp-kernel.cpp"), "utf8")
    .replace('"../../sycl/blake3-device.inc"', JSON.stringify(path.join(repo, "sycl/blake3-device.inc"))));
  const plugin = path.join(build, "src/compiler/libacpp-clang.so");
  run(path.join(tool, "clang++"), ["-std=c++17", "-O2", "-D__ACPP__", "-D__ACPP_ENABLE_LLVM_SSCP_TARGET__", ...includes,
    "-include", path.join(source, "include/hipSYCL/glue/llvm-sscp/hcf_registration.hpp"), `-fplugin=${plugin}`, `-fpass-plugin=${plugin}`,
    "-Xclang", "-disable-O0-optnone", "-mllvm", "-acpp-sscp", "-mllvm", "-acpp-sscp-emit-hcf", "-c", input, "-o", path.join(output, "generic.o")],
  path.join(output, "generic-build.log"));
  const deviceLibs = setting("ROCM_DEVICE_LIBS_PATH");
  if (!deviceLibs) {throw new Error("Missing selected ROCm device libraries");}
  const values = [{name: "generic.hcf", file: input + ".hcf"},
    {name: "libkernel-sscp-amdgpu-amdhsa-full.bc", file: path.join(build, "src/libkernel/sscp/amdgpu/libkernel-sscp-amdgpu-amdhsa-full.bc")},
    ...fs.readdirSync(deviceLibs).filter((name) => name.endsWith(".bc")).sort().map((name) => ({name, file: path.join(deviceLibs, name)}))]
    .map(({name, file}) => ({name, data: fs.readFileSync(file)}));
  fs.writeFileSync(path.join(output, "amdgpu-resources.inc"), resources(values));
  const executable = path.join(output, "amdgpu-native"), lld = path.resolve(setting("LLD_DIR") || "/unavailable", "../../..");
  run(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-DHIPSYCL_COMPILER_COMPONENT", "-DACPP_STATIC_RUNTIME",
    ...includes, "-isystem", include, "-I", output, path.join(repo, "tests/reference/native-amdgpu-translator.cpp"), path.join(repo, "runtime/compiler-gate.cpp"),
    "-Wl,--gc-sections", "-static-libstdc++", "-static-libgcc", "-Wl,-Bstatic,--start-group",
    ...["llvm-to-amdgpu", "llvm-to-backend"].map((name) => path.join(build, "src/compiler/llvm-to-backend", `lib${name}.a`)),
    path.join(build, "src/common/libacpp-common.a"), ...lldInputs(lld, version), path.join(library, "libclangBasic.a"),
    "-L", library, ...libs, "-Wl,--end-group", "-lz", "-lzstd", "-Wl,-Bdynamic", "-lm", "-ldl", "-lpthread", "-o", executable], path.join(output, "link.log"));
  run(executable, [output], path.join(output, "translate.log"));
  run(executable, [], path.join(output, "missing-arguments.log"), 2);
  run(executable, [path.join(output, "absent", "directory")], path.join(output, "missing-output.log"), 1);
  if (!/Cannot preserve pre-emission IR/.test(fs.readFileSync(path.join(output, "missing-output.log"), "utf8"))) {
    throw new Error("Missing-output case failed for an unexpected reason");
  }
  // Independent build-only oracle: feed command-line LLVM the exact pre-emission module and
  // original SDK library selection. The runtime below may execute none of these tools.
  const abis = fs.readdirSync(deviceLibs).map((name) => Number(name.match(/^oclc_abi_version_([0-9]+)\.bc$/)?.[1]) || 0);
  const abi = Math.max(...abis);
  /** @type {Array<{cpu: string, wave: number, fast: boolean, matches: boolean, bytes: number, sha256: string}>} */
  const oracle = [];
  if (!abi) {throw new Error("No device-library ABI found");}
  /** @type {Array<[string, number, string]>} */
  const variants = [["gfx900", 64, ""], ["gfx906", 64, ",+sramecc,-xnack"], ["gfx1200", 32, ""], ["gfx1200", 64, ""]];
  let index = 0;
  for (const fast of [false, true]) {
    for (const [cpu, wave, extra] of variants) {
      const base = path.join(output, String(index++));
      const features = (wave === 32 ? "+wavefrontsize32,-wavefrontsize64" : "+wavefrontsize64,-wavefrontsize32") + extra;
      const selected = ["hip.bc", "ockl.bc", "ocml.bc", `oclc_isa_version_${cpu.slice(3)}.bc`,
        `oclc_wavefrontsize64_${wave === 64 ? "on" : "off"}.bc`, `oclc_abi_version_${abi}.bc`,
        `oclc_correctly_rounded_sqrt_${fast ? "off" : "on"}.bc`, ...["daz_opt", "finite_only", "unsafe_math"].map((name) => `oclc_${name}_${fast ? "on" : "off"}.bc`)];
      run(path.join(tool, "llvm-link"), [base + ".bc", ...selected.map((name) => path.join(deviceLibs, name)), "-o", base + "-oracle-linked.bc"], base + "-oracle-link.log");
      run(path.join(tool, "opt"), ["-mtriple=amdgcn-amd-amdhsa", `-mcpu=${cpu}`, `-mattr=${features}`, "-relocation-model=pic", "-O3",
        base + "-oracle-linked.bc", "-o", base + "-oracle-opt.bc"], base + "-oracle-opt.log");
      const flags = fast ? ["--enable-unsafe-fp-math", "--enable-no-infs-fp-math", "--enable-no-nans-fp-math", "--enable-no-signed-zeros-fp-math", "--enable-no-trapping-fp-math"] : [];
      run(path.join(tool, "llc"), ["-mtriple=amdgcn-amd-amdhsa", `-mcpu=${cpu}`, `-mattr=${features}`, "-relocation-model=pic", "-O3", "-filetype=obj",
        base + "-oracle-opt.bc", "-o", base + "-oracle.o", ...flags], base + "-oracle-llc.log");
      run(path.join(tool, "ld.lld"), ["-m", "elf64_amdgpu", "--shared", "--no-undefined", "--build-id=none", "--threads=1", "-o", base + "-oracle.hsaco", base + "-oracle.o"],
        base + "-oracle-elf.log");
      const binary = fs.readFileSync(base + ".hsaco"), expected = fs.readFileSync(base + "-oracle.hsaco");
      const metadata = run(path.join(tool, "llvm-readelf"), ["--notes", base + ".hsaco"], base + "-metadata.log");
      if ((metadata.match(new RegExp(`\\.wavefront_size:\\s+${wave}\\b`, "g")) || []).length !== 4 ||
          (metadata.match(/\.max_flat_workgroup_size:\s+64\b/g) || []).length !== 4) {
        throw new Error("AMD image lost wave/workgroup specialization");
      }
      oracle.push({cpu, wave, fast, matches: binary.equals(expected), bytes: binary.length, sha256: createHash("sha256").update(binary).digest("hex")});
    }
  }
  fs.writeFileSync(path.join(output, "oracle.json"), JSON.stringify(oracle, null, 2) + "\n");
  if (oracle.some((entry) => !entry.matches)) {throw new Error("Native AMD images differ from the independent LLVM tool oracle; inspect oracle.json");}
  if (oracle.slice(0, 4).some((entry, index) => entry.sha256 === oracle[index + 4]?.sha256)) {throw new Error("Fast-math device libraries had no effect");}
  const needed = nativeDependencies(run("readelf", ["-d", executable], path.join(output, "dependencies.log")));
  const trace = path.join(output, "syscalls.log");
  run("strace", ["-f", "-s", "1024", "-e", "trace=open,openat,openat2,creat,execve,execveat,mkdir,mkdirat,rmdir", "-E", "PATH=/unavailable", "-E", "LD_LIBRARY_PATH=/unavailable",
    "-o", trace, executable, output], path.join(output, "isolated-paths.log"));
  const captured = fs.readFileSync(trace, "utf8");
  auditNativeTrace(captured, executable);
  const scratch = captured.split("\n").filter((line) => /\bmkdir(?:at)?\([^\n]*"[^"\n]*\/acpp-amdgpu-/.test(line));
  if (scratch.length < 23 || scratch.some((line) => !/", 0700\)\s+= 0$/.test(line))) {
    throw new Error("AMD generated-data scratch was not created privately");
  }
  const scratchRoots = [];
  for (const line of scratch) {
    const directory = line.match(/"([^"\n]+)"/)?.[1];
    if (!directory || !captured.split("\n").some((entry) => entry.includes(`rmdir("${directory}")`) && /\)\s+= 0$/.test(entry))) {
      throw new Error("AMD generated-data scratch was not cleaned");
    }
    scratchRoots.push(directory);
  }
  auditNativeHeaders(captured, [], [output, ...scratchRoots]);
  if (captured.split("\n").some((line) => /openat\([^\n]*"[^"\n]*\.(?:bc|hcf|ll)"/.test(line) && !/", O_WRONLY\b/.test(line))) {
    throw new Error("Native AMD JIT read external compiler resources");
  }
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({version, needed, bytes: fs.statSync(executable).size,
    resources: values.map(({name, data}) => ({name, bytes: data.length, sha256: createHash("sha256").update(data).digest("hex")})), oracle,
    scope: "CPU-only generic SYCL AMD lowering/linking; no GPU dispatch, performance or full-worker qualification"}, null, 2) + "\n");
  console.log("Static AdaptiveCpp generic SYCL/AMD proof passed without external compiler or device-library files.");
}

if (require.main === module) {try {main();} catch (error) {console.error(error); process.exitCode = 1;}}
