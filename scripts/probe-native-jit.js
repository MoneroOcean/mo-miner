"use strict";

// Opt-in, CPU-only LLVM API/linking experiment. Never part of Engineering Mode or GPU discovery.
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const {nativeDependencies, auditNativeTrace, isStaticArchive} = require("./native-audit");

/** Admit only a matching static linker SDK; importing the probe must never compile or run it.
 * @param {string | undefined} root @param {string} version */
function lldInputs(root, version) {
  if (!root) {return [];}
  if (!path.isAbsolute(root)) {throw new Error("LLD SDK root must be absolute");}
  const metadata = path.join(root, "lib/cmake/lld/LLDConfigVersion.cmake");
  const libraries = ["ELF", "Common"].map((name) => path.join(root, "lib", `liblld${name}.a`));
  if (!fs.existsSync(path.join(root, "include/lld/Common/Driver.h")) || !fs.existsSync(metadata) ||
      fs.readFileSync(metadata, "utf8").match(/set\(PACKAGE_VERSION "([0-9.]+)"\)/)?.[1] !== version ||
      libraries.some((file) => !fs.existsSync(file) || !isStaticArchive(file))) {
    throw new Error("LLD SDK requires matching LLVM version, headers, and self-contained static ELF/Common archives");
  }
  return libraries;
}

/** @param {string} command @param {string[]} args @param {string} log @param {number} [expected] */
function run(command, args, log, expected = 0) {
  const result = spawnSync(command, args, {encoding: "utf8", timeout: 120000, maxBuffer: 4 << 20});
  fs.writeFileSync(log, (result.stdout || "") + (result.stderr || ""));
  if (result.error || result.signal || result.status !== expected) {throw new Error(`JIT probe failed; see ${log}`, {cause: result.error});}
  return result.stdout;
}

function main() {
  if (process.platform !== "linux" || process.argv.length > 3) {throw new Error("Usage (Linux): node scripts/probe-native-jit.js [evidence parent]");}
  const output = fs.mkdtempSync(path.join(path.resolve(process.argv[2] || os.tmpdir()), "mom-jit-"));
  console.log(`In-process LLVM evidence: ${output}`);
  const config = process.env["LLVM_CONFIG"] || "/usr/lib/llvm-21/bin/llvm-config";
  const version = run(config, ["--version"], path.join(output, "version.log")).trim();
  if (!version.startsWith("21.")) {throw new Error("This initial in-process JIT probe requires LLVM 21");}
  const include = run(config, ["--includedir"], path.join(output, "includes.log")).trim();
  const library = run(config, ["--libdir"], path.join(output, "library.log")).trim();
  const lldRoot = process.env["MOM_AOT_LLD_ROOT"];
  const lldLibraries = lldInputs(lldRoot, version);
  // The distro linker archive initializes all target registries. This initial opt-in proof must
  // retain them; prune that build policy before calling the combined executable compact.
  const libs = run(config, ["--link-static", "--libs", ...(lldRoot ? ["all"] : ["core", "asmparser", "nvptxcodegen", "amdgpucodegen", "passes"])],
    path.join(output, "llvm-libraries.log")).trim().split(/\s+/);
  const executable = path.join(output, "jit-native");
  run(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
    ...(lldRoot ? ["-DMOM_NATIVE_AMDGPU_LINK_PROBE", "-isystem", path.join(lldRoot, "include")] : []),
    "-isystem", include, path.join(__dirname, "..", "tests", "reference", "native-jit.cpp"),
    "-L", library, "-Wl,--gc-sections", "-static-libstdc++", "-static-libgcc", "-Wl,-Bstatic", ...lldLibraries, ...libs,
    "-lz", "-lzstd", "-Wl,-Bdynamic", "-lm", "-ldl", "-lpthread", "-o", executable], path.join(output, "build.log"));
  run(executable, [output], path.join(output, "jit.log"));
  const linkedImages = [];
  if (lldRoot) {
    for (const target of ["gfx900", "gfx1200"]) {
      const actual = fs.readFileSync(path.join(output, target + ".hsaco"));
      const oracle = path.join(output, target + "-oracle.hsaco");
      run(path.join(run(config, ["--bindir"], path.join(output, "tools.log")).trim(), "ld.lld"),
        ["-m", "elf64_amdgpu", "--shared", "--no-undefined", "--build-id=none", "--threads=1", "-o", oracle,
          path.join(output, target + ".o")], path.join(output, target + "-oracle.log"));
      const matches = actual.equals(fs.readFileSync(oracle));
      if (!matches) {throw new Error("In-process ELF output differs from external linker oracle");}
      linkedImages.push({target, bytes: actual.length, matches});
    }
  }
  run(executable, [], path.join(output, "missing-arguments.log"), 2);
  const missingOutput = path.join(output, "missing-output.log");
  run(executable, [path.join(output, "missing", "directory")], missingOutput, 1);
  if (!fs.readFileSync(missingOutput, "utf8").includes("Cannot preserve JIT output")) {
    throw new Error("JIT fixture failed for an unexpected reason");
  }
  const dependencies = run("readelf", ["-d", executable], path.join(output, "dependencies.log"));
  const needed = nativeDependencies(dependencies);
  if (lldRoot) {
    const trace = path.join(output, "syscalls.log");
    run("strace", ["-f", "-s", "1024", "-e", "trace=openat,execve,execveat", "-E", "PATH=/unavailable",
      "-E", "LD_LIBRARY_PATH=/unavailable", "-o", trace, executable, output], path.join(output, "isolated-link.log"));
    auditNativeTrace(fs.readFileSync(trace, "utf8"), executable);
  }
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({version, needed,
    bytes: fs.statSync(executable).size, targets: ["sm_61", "sm_120", "gfx900", "gfx1200"], lldRoot,
    linkerArchives: lldLibraries.map((file) => ({file, sha256: createHash("sha256").update(fs.readFileSync(file)).digest("hex")})), linkedImages,
    scope: "In-process IR optimization/codegen and optional ELF linking; no GPU, SYCL integration or device-library qualification"}, null, 2) + "\n");
  console.log("Static in-process device code generation/linking passed; runtime integration remains unqualified.");
}

module.exports = {lldInputs};
if (require.main === module) {
  try {main();} catch (error) {console.error(error); process.exitCode = 1;}
}
