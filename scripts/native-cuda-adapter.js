"use strict";

// Build-only adapter closure: original APIs/implementation, generated SDK-typed system-driver
// shims, and a single opaque dispatch factory. No GPU lookup or work is performed by this tool.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const {isolateCompilerObject} = require("./native-compiler-isolation");

/** Keep only exact unmangled imported driver entries, not a handwritten API registry.
 * @param {string} symbols */
function driverImports(symbols) {
  return [...new Set([...symbols.matchAll(/^(cu[A-Z]\w*|nvml[A-Z]\w*) U(?:\s|$)/gm)].map((match) => match[1] || ""))].sort();
}

/** Count parameters from the compiler's view of the selected SDK, not parsed C declarators.
 * Generated definitions derive every parameter/return type through decltype of that same API.
 * @param {unknown} ast @param {string[]} names */
function driverSignatures(ast, names) {
  const selected = new Set(names), counts = new Map();
  /** @param {unknown} value */
  function visit(value) {
    if (!value || typeof value !== "object") {return;}
    const node = /** @type {{kind?: string, name?: string, variadic?: boolean, inner?: unknown[]}} */ (value);
    if (node.kind === "FunctionDecl" && node.name && selected.has(node.name)) {
      if (node.variadic) {throw new Error("Variadic driver entry is not supported");}
      const count = (node.inner || []).filter((child) => child && typeof child === "object" && Reflect.get(child, "kind") === "ParmVarDecl").length;
      if (count > 32 || (counts.has(node.name) && counts.get(node.name) !== count)) {throw new Error("Ambiguous or excessive driver arguments");}
      counts.set(node.name, count);
    }
    for (const child of node.inner || []) {visit(child);}
  }
  visit(ast);
  if (!names.length || names.some((name) => !counts.has(name))) {throw new Error("Imported driver entry is absent from the selected SDK");}
  return names.map((name) => ({name, count: /** @type {number} */ (counts.get(name))}));
}

/** Missing drivers/symbols fail locally without touching output storage or accepting compute.
 * Error-description helpers supply safe diagnostics even when no driver can describe the error.
 * @param {Array<{name: string, count: number}>} entries */
function forwardingSource(entries) {
  const lines = ["// Generated from exact adapter imports and compiler-validated SDK declarations.",
    "#include <cuda.h>", "#include <nvml.h>", "#include <tuple>", "#include <type_traits>",
    '#include "system-driver.h"', "namespace {",
    "template<class> struct Signature;",
    "template<class R, class... A> struct Signature<R(*)(A...)> { using Result=R; template<size_t I> using Arg=std::tuple_element_t<I,std::tuple<A...>>; };", "}"];
  for (const [index, {name, count}] of entries.entries()) {
    if (!/^(cu|nvml)[A-Z]\w*$/.test(name) || !Number.isSafeInteger(count) || count < 0 || count > 32) {throw new Error("Invalid driver forwarding entry");}
    const type = `Entry${index}`, nvml = name.startsWith("nvml"), args = Array.from({length: count}, (_, i) => `a${i}`);
    const declarations = args.map((arg, i) => `${type}::Arg<${i}> ${arg}`).join(", ");
    const expected = name === "nvmlErrorString" ? "const char*" : nvml ? "nvmlReturn_t" : "CUresult";
    const fallback = name === "nvmlErrorString" ? '"NVIDIA management driver unavailable"' : nvml ? "NVML_ERROR_UNINITIALIZED" : name === "cuInit" ? "CUDA_ERROR_NO_DEVICE" : "CUDA_ERROR_NOT_INITIALIZED";
    lines.push(`using ${type} = Signature<decltype(&::${name})>;`, `static_assert(std::is_same_v<${type}::Result, ${expected}>);`,
      `extern "C" ${type}::Result ${nvml ? "" : "CUDAAPI "}${name}(${declarations}) {`,
      `  static const auto function = mom::runtime::systemDriverSymbol<mom::runtime::SystemDriver::${nvml ? "nvml" : "cuda"}, decltype(&::${name})>("${name}");`,
      `  if (function) return function(${args.join(", ")});`);
    if (name === "cuGetErrorName" || name === "cuGetErrorString") {lines.push('  if (a1) *a1 = "CUDA driver unavailable";');}
    lines.push(`  return ${fallback};`, "}");
  }
  return lines.join("\n") + "\n";
}

function main() {
  const args = process.argv.slice(2);
  if (process.platform !== "linux" || process.env["MOM_AOT_BUILD_ONLY"] !== "1" || args.length !== 5 ||
      !args.every((value) => path.isAbsolute(value))) {
    throw new Error("Usage (Linux, MOM_AOT_BUILD_ONLY=1): native-cuda-adapter.js <archive> <CUDA SDK> <clang++> <original UR headers> <new output>");
  }
  const [archive, sdk, compiler, urHeaders, root] = /** @type {[string, string, string, string, string]} */ (args);
  const repo = path.resolve(__dirname, "..");
  fs.mkdirSync(root, {recursive: true});
  const output = fs.mkdtempSync(path.join(root, "attempt-"));
  /** @param {string} command @param {string[]} args @param {string} label @param {number} [maxBuffer] */
  function run(command, args, label, maxBuffer = 64 << 20) {
    fs.writeFileSync(path.join(output, label + ".command.json"), JSON.stringify({command, args}) + "\n");
    const result = spawnSync(command, args, {encoding: "utf8", timeout: 180000, maxBuffer});
    fs.writeFileSync(path.join(output, label + ".log"), (result.stdout || "") + (result.stderr || ""));
    if (result.error || result.signal || result.status) {throw new Error(`Static CUDA adapter failed: ${label}`, {cause: result.error});}
    return result.stdout;
  }
  const names = driverImports(run("nm", ["-P", "-u", archive], "imports"));
  if (names.length < 20 || names.length > 512 || !names.includes("cuInit") || !names.includes("nvmlErrorString")) {throw new Error("Incomplete original CUDA driver closure");}
  const header = path.join(output, "driver-headers.cpp");
  fs.writeFileSync(header, "#include <cuda.h>\n#include <nvml.h>\n");
  const includes = ["-isystem", path.join(sdk, "include"), "-I", path.join(repo, "runtime")];
  const ast = JSON.parse(run(compiler, ["-x", "c++", "-std=c++17", ...includes, "-fsyntax-only", "-Xclang", "-ast-dump=json", header], "sdk-ast"));
  const entries = driverSignatures(ast, names), source = path.join(output, "driver-forwarding.cpp");
  fs.writeFileSync(source, forwardingSource(entries));
  fs.writeFileSync(path.join(output, "entries.json"), JSON.stringify(entries, null, 2) + "\n");
  const flags = ["-std=c++17", "-O2", "-fPIC", "-ffunction-sections", "-fdata-sections", ...includes];
  run(compiler, [...flags, "-c", source, "-o", path.join(output, "drivers.o")], "driver-build");
  run(compiler, [...flags, "-isystem", urHeaders, "-c", path.join(repo, "tests/native/dpcpp-static/cuda-factory.cpp"), "-o", path.join(output, "factory.o")], "factory-build");
  const raw = path.join(output, "adapter-raw.o");
  run("ld", ["-r", "--whole-archive", archive, "--no-whole-archive", path.join(output, "drivers.o"), path.join(output, "factory.o"), "-o", raw], "partial-link");
  const result = isolateCompilerObject(raw, path.join(output, "isolated"), {prefix: "mom_cuda_", entryPoints: ["mom_ur_cuda_factory"]});
  const inputs = Object.fromEntries([archive, path.join(sdk, "include/cuda.h"), path.join(sdk, "include/nvml.h"),
    path.join(urHeaders, "unified-runtime/ur_api.h"), path.join(urHeaders, "unified-runtime/ur_ddi.h")].map((file) =>
    [file, createHash("sha256").update(fs.readFileSync(file)).digest("hex")]));
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({entries, inputs, ...result,
    scope: "Original static CUDA adapter and typed optional driver forwarding; BUILD ONLY, NOT execution or GPU qualification"}, null, 2) + "\n");
  // CMake consumes a stable artifact; preserve failed/prior receipts and publish only a fully
  // qualified new object. Readers never observe a partially copied relocatable object.
  const staged = path.join(root, "adapter.o.tmp-" + process.pid);
  fs.copyFileSync(result.object, staged, fs.constants.COPYFILE_EXCL);
  fs.renameSync(staged, path.join(root, "adapter.o"));
  console.log(`Original static CUDA adapter isolated; NOT executed: ${result.object}`);
}

module.exports = {driverImports, driverSignatures, forwardingSource};
if (require.main === module) {try {main();} catch (error) {console.error(error); process.exitCode = 1;}}
