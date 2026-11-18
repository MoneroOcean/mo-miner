"use strict";
const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {coffExternalFallbacks} = require("../scripts/native-compiler-isolation");
const {adapterDefinitions, windowsDriverRequirements, windowsRuntimeLinks, adapterSdkInputs, prepareWindowsAdapters} = require("../scripts/prepare-native-windows-adapters");
const {windowsDriverBindings, windowsTypedDriverAliases} = require("../scripts/native-windows-driver");

test.describe("Windows original adapter namespace boundary", () => {
  test("CRT aliases retain their external override and weak fallback definitions", () => {
    const definitions = new Set(["weakCRT", "private"]), names = coffExternalFallbacks(definitions, [["externalCRT", "weakCRT"]]);
    assert.deepEqual([...names].sort(), ["externalCRT", "weakCRT"]); assert.equal(names.has("private"), false);
  });
  test("an externally visible alias chain is preserved until its defined fallback", () => {
    assert.deepEqual([...coffExternalFallbacks(new Set(["leaf"]), [["outer", "inner"], ["inner", "leaf"]])].sort(), ["inner", "leaf", "outer"]);
  });
  test("private defined aliases do not become external fallback roots", () => {
    assert.deepEqual([...coffExternalFallbacks(new Set(["private", "other"]), [["private", "other"]])], []);
  });
  test("alias cycles terminate without losing external names", () => {
    assert.deepEqual([...coffExternalFallbacks(new Set(), [["left", "right"], ["right", "left"]])].sort(), ["left", "right"]);
  });
  test("contradictory CRT aliases cannot be silently overwritten", () => {
    assert.throws(() => coffExternalFallbacks(new Set(), [["external", "one"], ["external", "two"]]), /Conflicting/);
    assert.deepEqual([...coffExternalFallbacks(new Set(), [["external", "one"], ["external", "one"]])].sort(), ["external", "one"]);
  });
  test("fallback metadata shares the existing bounded symbol budget", () => {
    assert.throws(() => coffExternalFallbacks(new Set(), [["external", "one"]], 0), /metadata budget/);
    for (const value of [-1, NaN, Infinity, 0.5]) {assert.throws(() => coffExternalFallbacks(new Set(), [], value), /fallback budget/);}
  });
  test("real archive headings are ignored but symbol identities are exact", () => {
    const names = adapterDefinitions("\nfactory.cpp.obj:\nmom_ur_cuda_factory T 0 0\n?urSymbol@@ T 20 0\n__real_data D 0 4\n");
    assert.deepEqual([...names], ["mom_ur_cuda_factory", "?urSymbol@@", "__real_data"]);
  });
  test("malformed or excessive definition metadata fails closed", () => {
    for (const text of ["not a symbol", "A".repeat(4097) + " T 0 0", "one T not-hex"]) {assert.throws(() => adapterDefinitions(text), /Malformed/);}
    assert.throws(() => adapterDefinitions(" ".repeat((16 << 20) + 1)), /exceeds bounds/);
  });
  test("adapter inputs must be absolute before opening an SDK receipt", () => {
    for (const row of [["relative", "/descriptor", "/cuda"], ["/sdk", "relative", "/cuda"], ["/sdk", "/descriptor", "/cuda\n"]]) {
      assert.throws(() => adapterSdkInputs(row[0] || "", row[1] || "", row[2] || ""), /absolute SDK/);
    }
  });
  test("preparation cannot execute tools without build-only admission", () => {
    assert.throws(() => prepareWindowsAdapters("/absent", "/absent", "/absent", "/absent", {}), /MOM_AOT_BUILD_ONLY/);
  });
  test("direct calls and Windows import pointers select the same exact driver APIs", () => {
    assert.deepEqual(windowsDriverRequirements("cuInit U 0 0\n__imp_nvmlInit_v2 U 0 0\n__imp_nvmlErrorString U 0 0\n__imp_nvmlInit_v2 U 0 0\n__imp_other U 0 0\n__imp_nvmlUnused T 0 4\n"),
      {names: ["cuInit", "nvmlErrorString", "nvmlInit_v2"], pointers: ["nvmlErrorString", "nvmlInit_v2"]});
  });
  test("only actual import pointers acquire a typed alias, without an NVML application DLL", () => {
    const source = windowsDriverBindings(["cuInit", "nvmlInit_v2"], "cuda");
    assert.ok(source.includes("WindowsCudaCall<decltype(&::nvmlInit_v2)>::template call<mom_cuda_name_1>"));
    assert.ok(source.includes("decltype(&::cuInit) __imp_cuInit"));
    assert.ok(source.includes('asm("mom_cuda_thunk_1")'));
    assert.doesNotMatch(source, /LoadLibrary|nvml\.dll|ast-dump|Signature</);
  });
  test("unknown, duplicate and malformed import-pointer aliases fail before compilation", () => {
    for (const names of [[], ["wrong"], ["cuInit", "cuInit"], ["cuInit;bad"]]) {assert.throws(() => windowsDriverBindings(names, "cuda"), /Invalid Windows driver/);}
  });
  test("generated driver labels compile and link direct/import calls without C++ ABI-name parsing", t => {
    const suffix = process.platform === "win32" ? ".exe" : "", bin = process.env["MOM_TEST_LLVM_BIN"];
    const tools = ["clang-cl", "llvm-nm", "lld-link"].map(name => {
      if (bin) {return path.join(bin, name + suffix);}
      const found = spawnSync(process.platform === "win32" ? "where.exe" : "which", [name], {encoding: "utf8", timeout: 5000});
      return found.status === 0 ? found.stdout.trim().split(/\r?\n/)[0] : undefined;
    });
    if (tools.some(file => !file || !fs.statSync(file, {throwIfNoEntry: false})?.isFile())) {t.skip("COFF LLVM tools unavailable; set MOM_TEST_LLVM_BIN"); return;}
    const [clang, nm, linker] = tools, root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-driver-labels-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    /** @param {string | undefined} command @param {string[]} args */
    const run = (command, args) => {
      assert.ok(command);
      const result = spawnSync(command, args, {cwd: root, encoding: "utf8", timeout: 30000, maxBuffer: 4 << 20});
      assert.equal(result.status, 0, result.error?.message || result.stderr); return result.stdout;
    };
    const objects = [], aliases = [], declarations = [], comparisons = [];
    fs.mkdirSync(path.join(root, "runtime"));
    for (const kind of /** @type {const} */ (["hip", "cuda"])) {
      const names = kind === "hip" ? ["hipGetDeviceCount", "hipGetErrorString", "hipCopyPayload"] : ["cuInit", "nvmlErrorString"];
      fs.copyFileSync(path.join(__dirname, "logic/windows-driver-bindings.h"), path.join(root, "runtime/windows-" + kind + "-driver.h"));
      const source = path.join(root, kind + ".cpp"), object = path.join(root, kind + ".obj");
      fs.writeFileSync(source, windowsDriverBindings(names, kind));
      run(clang, ["--target=x86_64-pc-windows-msvc", "/nologo", "/std:c++17", "/O2", "/GS-", "/c", "-I" + root, source, "/Fo" + object]);
      aliases.push(...windowsTypedDriverAliases(run(nm, ["-P", "--defined-only", object]), names, kind)); objects.push(object);
      declarations.push(...names.map(name => `extern "C" decltype(&::${name}) __imp_${name};`));
      comparisons.push(...names.map(name => `&::${name} == __imp_${name}`));
    }
    const caller = path.join(root, "caller.cpp"), callerObject = path.join(root, "caller.obj");
    fs.writeFileSync(caller, ['#include "runtime/windows-hip-driver.h"', ...declarations,
      'extern "C" __declspec(dllexport) bool binding_probe() { return ' + comparisons.join(" && ") + "; }"].join("\n"));
    run(clang, ["--target=x86_64-pc-windows-msvc", "/nologo", "/std:c++17", "/O2", "/GS-", "/c", caller, "/Fo" + callerObject]);
    run(linker, ["/dll", "/noentry", "/nodefaultlib", "/out:" + path.join(root, "bindings.dll"), ...objects, callerObject, ...aliases]);
  });
  for (const [mode, name] of [[0, "available driver"], [1, "absent drivers"], [2, "missing optional exports"]]) {
    test(`real Windows CUDA/NVML template: ${name}, typed ABI and cached lookup`, {skip: process.platform !== "linux"}, t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-cuda-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      const executable = path.join(root, "probe"), repo = path.resolve(__dirname, "..");
      const built = spawnSync("c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-D_WIN32", "-I", path.join(__dirname, "logic/cuda-driver"), "-I", repo,
        path.join(__dirname, "logic/windows-cuda-driver.cpp"), "-o", executable], {encoding: "utf8", timeout: 30000});
      assert.equal(built.status, 0, built.stderr);
      const result = spawnSync(executable, [String(mode)], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, result.stderr); assert.match(result.stdout, /ABI, fallback and cached lifetime passed/);
    });
  }
});

/** Shape-only inputs: the producer verifies original source/archive hashes before selection. */
function linksFixture() {
  return {artifacts: ["mom-sycl-static", "ur_loader", "ur_common", "ur_adapter_cuda", "ur_adapter_level_zero", "ur_adapter_level_zero_v2", "ur_adapter_opencl",
    "ze_loader", "level_zero_utils", "umf", "umf_ba", "umf_coarse", "umf_utils", "hwloc", "OpenCL"].map(name => ({path: "/sdk/" + name + ".lib"})),
  adapters: ["cuda", "level-zero-v2"].map(kind => ({kind, factory: "mom_ur_" + kind.replaceAll("-", "_") + "_factory", archive: "/isolated/" + kind + ".lib"})),
  objects: ["drivers", "driver-aliases"].map(name => ({path: "/isolated/" + name + ".obj"}))};
}

test.describe("Source-built Windows runtime link selection", () => {
  test("includes every original archive exactly once, replacing only isolated adapters", () => {
    const f = linksFixture(), link = windowsRuntimeLinks(f.artifacts, f.adapters, f.objects);
    assert.equal(link.libraries.length, 13); assert.equal(link.wholeArchives.length, 2);
    const all = [...link.libraries, ...link.wholeArchives]; assert.equal(new Set(all).size, 15);
    for (const row of f.artifacts) {
      if (/ur_adapter_(?:cuda|level_zero_v2)\.lib$/.test(row.path)) {assert.equal(all.includes(row.path), false);} else {assert.ok(all.includes(row.path));}
    }
    for (const row of f.adapters) {assert.ok(all.includes(row.archive));}
  });
  test("constructor-bearing archives retain whole-archive semantics", () => {
    const f = linksFixture(), link = windowsRuntimeLinks(f.artifacts, f.adapters, f.objects);
    assert.deepEqual(link.wholeArchives, ["/sdk/mom-sycl-static.lib", "/sdk/ze_loader.lib"]);
  });
  test("both NVIDIA binding and alias objects are explicit, never archive-only inputs", () => {
    const f = linksFixture(), link = windowsRuntimeLinks(f.artifacts, f.adapters, f.objects);
    assert.deepEqual(link.objects, ["/isolated/drivers.obj", "/isolated/driver-aliases.obj"]);
  });
  test("selection does not mutate the verified input cohort", () => {
    const f = linksFixture(), original = JSON.stringify(f); windowsRuntimeLinks(f.artifacts, f.adapters, f.objects); assert.equal(JSON.stringify(f), original);
  });
  test("partial and malformed original cohorts are rejected", () => {
    const f = linksFixture(); f.artifacts.pop(); assert.throws(() => windowsRuntimeLinks(f.artifacts, f.adapters, f.objects), /Incomplete Windows runtime/);
  });
  test("duplicate original archive basenames cannot hide a missing component", () => {
    const f = linksFixture(); f.artifacts[0] = {path: "/other/ur_loader.lib"}; assert.throws(() => windowsRuntimeLinks(f.artifacts, f.adapters, f.objects), /Duplicate Windows runtime/);
  });
  test("a missing or repeated adapter cannot replace the required pair", () => {
    const f = linksFixture(); assert.ok(f.adapters[0]); f.adapters[1] = {...f.adapters[0]}; assert.throws(() => windowsRuntimeLinks(f.artifacts, f.adapters, f.objects), /Incomplete isolated/);
  });
  test("factory identity stays coupled to the selected original adapter", () => {
    const f = linksFixture(); assert.ok(f.adapters[0]); f.adapters[0].factory = "mom_ur_wrong_factory"; assert.throws(() => windowsRuntimeLinks(f.artifacts, f.adapters, f.objects), /Incomplete isolated.*cuda/);
  });
  test("the original unisolated archive cannot stand in for its private factory", () => {
    const f = linksFixture(); assert.ok(f.adapters[0]); f.adapters[0].archive = "/sdk/ur_adapter_cuda.lib"; assert.throws(() => windowsRuntimeLinks(f.artifacts, f.adapters, f.objects), /Incomplete isolated.*cuda/);
  });
  test("two adapters cannot alias the same transformed archive", () => {
    const f = linksFixture(); assert.ok(f.adapters[0] && f.adapters[1]); f.adapters[1].archive = f.adapters[0].archive; assert.throws(() => windowsRuntimeLinks(f.artifacts, f.adapters, f.objects), /Incomplete isolated.*level-zero-v2/);
  });
  test("missing constructor archives are not silently downgraded", () => {
    const f = linksFixture(); f.artifacts[0] = {path: "/sdk/unexpected.lib"}; assert.throws(() => windowsRuntimeLinks(f.artifacts, f.adapters, f.objects), /Missing constructor-bearing/);
  });
  test("missing, duplicate and non-object driver bindings fail closed", () => {
    for (const objects of [[{path: "/one.obj"}], [{path: "/one.obj"}, {path: "/one.obj"}], [{path: "/one.obj"}, {path: "/two.lib"}]]) {
      const f = linksFixture(); assert.throws(() => windowsRuntimeLinks(f.artifacts, f.adapters, objects), /Incomplete Windows runtime|Duplicate Windows runtime/);
    }
  });
  test("relative paths, newline injection and shared-library spellings are not accepted", () => {
    for (const file of ["relative.lib", "/bad\nname.lib", "/bad\0name.lib", "/bad.dll"]) {
      const f = linksFixture(); f.artifacts[0] = {path: file}; assert.throws(() => windowsRuntimeLinks(f.artifacts, f.adapters, f.objects), /Incomplete Windows runtime/);
    }
  });
  test("an explicitly selected source-JIT closure retains every compiler archive", () => {
    const f = linksFixture(), jit = ["/compiler/sycl-jit.lib", "/compiler/clangTooling.lib", "/compiler/LLVMCore.lib"];
    const link = windowsRuntimeLinks(f.artifacts, f.adapters, f.objects, jit);
    assert.deepEqual(link.libraries.slice(-3), jit); assert.equal(link.libraries.length, 16);
  });
  test("JIT and runtime archives cannot be linked twice", () => {
    const f = linksFixture(); assert.throws(() => windowsRuntimeLinks(f.artifacts, f.adapters, f.objects, ["/sdk/ur_loader.lib"]), /Duplicate Windows runtime\/JIT/);
  });
  test("JIT closure entries must remain bounded explicit static Windows libraries", () => {
    const f = linksFixture();
    for (const libraries of [["/compiler/jit.dll"], ["relative.lib"], Array(513).fill("/compiler/jit.lib")]) {
      assert.throws(() => windowsRuntimeLinks(f.artifacts, f.adapters, f.objects, libraries), /Incomplete Windows runtime/);
    }
  });
});
