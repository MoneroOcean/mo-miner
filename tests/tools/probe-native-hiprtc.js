"use strict";

// Explicit BUILD-ONLY compiler fixture. A GPU-free clean container executes it separately;
// importing/building this helper never discovers a device or runs a compiler job at runtime.
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {nativeDependencies} = require("../../scripts/native-audit");
const resources = require("../../scripts/native-resources");

function main() {
  const [manifest, parent] = process.argv.slice(2);
  if (process.platform !== "linux" || !manifest || process.argv.length > 4) {
    throw new Error("Usage (Linux): probe-native-hiprtc.js <native HIP inputs.json> [evidence-parent]");
  }
  const inputs = JSON.parse(fs.readFileSync(manifest, "utf8"));
  if (inputs.version !== 1 || inputs.backend !== "hip" || !inputs.compileFlags.includes("-DMOM_STATIC_HIPRTC")) {
    throw new Error("Use the complete static HIP GPU source manifest");
  }
  const repo = path.resolve(__dirname, "../.."), output = fs.mkdtempSync(path.join(path.resolve(parent || os.tmpdir()), "mom-hiprtc-"));
  const caller = fs.readFileSync(path.join(repo, "sycl/octopus/amd-wmma.inc"), "utf8");
  const start = caller.indexOf("    const std::string source ="), end = caller.indexOf("    const std::string arch_option", start);
  if (start < 0 || end < start) {throw new Error("Missing maintained Octopus source expression");}
  fs.writeFileSync(path.join(output, "octopus-wmma-source.inc"),
    "static std::string octopus_source(unsigned rows) {\n" + caller.slice(start, end) + "\nreturn source;\n}\n");
  const executable = path.join(output, "hiprtc-native");
  const mixed = process.env["MOM_AOT_MIXED_HCF"];
  /** @type {string[]} */
  const extra = [];
  if (mixed) {
    fs.writeFileSync(path.join(output, "mixed-compiler-resources.inc"), resources([{name: "generic.hcf", data: fs.readFileSync(mixed)}]));
    const config = spawnSync(process.env["LLVM_CONFIG"] || "/usr/lib/llvm-21/bin/llvm-config", ["--includedir"], {encoding: "utf8", timeout: 15000});
    if (config.status || !config.stdout.trim()) {throw new Error("Missing matching LLVM headers");}
    const lld = inputs.libraries.find((/** @type {string} */ file) => file.endsWith("/liblldCommon.a"));
    if (!lld) {throw new Error("Missing matching LLD SDK");}
    extra.push("-isystem", config.stdout.trim(), "-isystem", path.resolve(lld, "../../include"),
      "-Wl,--wrap=_ZN3lld7lldMainEN4llvm8ArrayRefIPKcEERNS0_11raw_ostreamES6_NS1_INS_9DriverDefEEE");
  }
  const fixture = path.join(repo, mixed ? "tests/reference/native-compiler-concurrency.cpp" : "tests/reference/native-hiprtc.cpp");
  fs.copyFileSync(fixture, path.join(output, "fixture-source.cpp")); // Preserve the exact negative/positive control input.
  const args = [...inputs.compileFlags, ...extra, "-I", output, fixture,
    path.join(repo, "runtime/driver-library.cpp"), path.join(repo, "runtime/compiler-gate.cpp"),
    ...inputs.flags.filter((/** @type {string} */ flag) => flag.startsWith("-Wl,") || flag.startsWith("--ld-path=")),
    "-Wl,--gc-sections", "-static-libstdc++", "-static-libgcc", "-Wl,--start-group", ...inputs.libraries, ...inputs.links,
    "-Wl,--end-group", "-o", executable];
  fs.writeFileSync(path.join(output, "command.json"), JSON.stringify({command: inputs.cxx, args}, null, 2) + "\n");
  const result = spawnSync(inputs.cxx, args, {encoding: "utf8", timeout: 180000, maxBuffer: 8 << 20});
  fs.writeFileSync(path.join(output, "build.log"), (result.stdout || "") + (result.stderr || ""));
  if (result.error || result.signal || result.status) {throw new Error(`Static HIPRTC build failed: ${output}`, {cause: result.error});}
  const elf = spawnSync("readelf", ["-d", executable], {encoding: "utf8", timeout: 15000});
  if (elf.error || elf.status) {throw new Error("Cannot audit HIPRTC executable dependencies");}
  const needed = nativeDependencies(elf.stdout);
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({needed, bytes: fs.statSync(executable).size,
    scope: `CPU-only ${mixed ? "mixed SYCL/HIPRTC concurrency" : "HIPRTC"} fixture built, NOT executed; no GPU or mining performance claim`}, null, 2) + "\n");
  console.log(`Static HIPRTC fixture built, NOT executed: ${output}`);
}

if (require.main === module) {try {main();} catch (error) {console.error(error); process.exitCode = 1;}}
