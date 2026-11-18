"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {spawnSync} = require("node:child_process");
const test = require("node:test"), assert = require("node:assert/strict");
const {windowsHipBindings, windowsHipAliases, windowsTypedDriverAliases} = require("../scripts/native-windows-driver");
const repo = path.resolve(__dirname, ".."), source = path.join(__dirname, "logic/windows-hip-driver.cpp");

test.describe("Native Windows HIP system-driver boundary", () => {
  const fixture = {root: "", executable: ""};
  const args = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-cast-function-type", "-D_WIN32", "-I", path.join(__dirname, "logic/windows-driver"), "-I", repo, source];
  test.before(() => {
    if (process.platform !== "linux") {return;}
    fixture.root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-hip-driver-")); fixture.executable = path.join(fixture.root, "fixture");
    const built = spawnSync("c++", [...args, "-o", fixture.executable], {encoding: "utf8", timeout: 30000});
    assert.equal(built.status, 0, built.stderr);
  });
  test.after(() => {if (fixture.root) {fs.rmSync(fixture.root, {recursive: true, force: true});}});
  for (const [mode, name] of [
    "static initialization performs no driver lookup", "absent driver reports no device without changing outputs",
    "absent driver returns a stable diagnostic string", "absent export never fabricates success",
    "real status and output pass through unchanged", "real diagnostic strings pass through unchanged",
    "one System32 module serves all HIP exports", "resolved exports are cached", "failed module lookup stays cached",
    "CUDA and NVML still resolve their separate fixed System32 libraries",
  ].entries()) {
    test(name, {skip: process.platform !== "linux"}, () => {
      const result = spawnSync(fixture.executable, [String(mode)], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, result.stderr || name);
    });
  }
  test("new result types require an explicit reviewed missing-driver policy", {skip: process.platform !== "linux"}, () => {
    const built = spawnSync("c++", [...args, "-DMOM_TEST_UNSUPPORTED_RESULT", "-o", path.join(fixture.root, "unsupported")], {encoding: "utf8", timeout: 30000});
    assert.notEqual(built.status, 0); assert.match(built.stderr, /Review the missing-driver fallback/);
  });
  test("Linux cannot accidentally opt into dynamic HIP application-library loading", {skip: process.platform !== "linux"}, () => {
    const built = spawnSync("c++", [...args.filter(arg => arg !== "-D_WIN32"), "-o", path.join(fixture.root, "linux")], {encoding: "utf8", timeout: 30000});
    assert.notEqual(built.status, 0); assert.match(built.stderr, /HIP system-driver bridge is Windows-only/);
  });
  const metadata = "Import {\n  Name: amdhip64_7.dll\n  ImportLookupTableRVA: 0x123\n  ImportAddressTableRVA: 0x456\n  Symbol: hipGetDeviceCount (1)\n  Symbol: hipGetErrorString (2)\n}\n";
  test("bindings use official SDK signatures rather than maintaining driver prototypes", () => {
    const result = windowsHipBindings(metadata);
    assert.deepEqual(result.names, ["hipGetDeviceCount", "hipGetErrorString"]);
    assert.match(result.source, /decltype\(&::hipGetDeviceCount\) __imp_hipGetDeviceCount/);
    assert.match(result.source, /WindowsHipCall<decltype\(&::hipGetErrorString\)>/);
    assert.ok(result.source.indexOf("#define __HIP_DISABLE_CPP_FUNCTIONS__") < result.source.indexOf("#include"));
    assert.doesNotMatch(result.source, /LoadLibrary|GetProcAddress|\.dll|hipError_t/);
  });
  test("ordinary OS imports do not become driver bindings or lookup paths", () => {
    assert.deepEqual(windowsHipBindings(metadata + "Import {\n  Name: KERNEL32.dll\n  Symbol: Sleep (42)\n}\n").names, windowsHipBindings(metadata).names);
  });
  test("missing, duplicate, old-version and delayed driver imports fail closed", () => {
    for (const text of ["", metadata + metadata, metadata.replace("amdhip64_7.dll", "amdhip64_6.dll"), metadata.replace("Import {", "DelayImport {")]) {
      assert.throws(() => windowsHipBindings(text), /Windows HIP|AMD HIP/);
    }
  });
  test("unknown, ordinal-only, duplicate and source-injecting symbols cannot produce code", () => {
    for (const symbol of ["Symbol: #1 (1)", "Symbol: otherApi (1)", "Symbol: hipBad;code (1)", "Symbol: hipGetDeviceCount (1)\n  Symbol: hipGetDeviceCount (1)"]) {
      assert.throws(() => windowsHipBindings(metadata.replace("Symbol: hipGetDeviceCount (1)", symbol)), /Unsupported Windows HIP/);
    }
  });
  test("overlarge metadata and missing discovery are rejected", () => {
    assert.throws(() => windowsHipBindings(" ".repeat((4 << 20) + 1)), /Invalid Windows HIP/);
    assert.throws(() => windowsHipBindings(metadata.replace("Symbol: hipGetDeviceCount (1)\n", "")), /retain device discovery/);
  });
  const symbols = [0, 1].map(index => "mom_hip_thunk_" + index);
  test("direct C calls and import pointers use the exact same generated typed thunk", () => {
    const names = windowsHipBindings(metadata).names, aliases = windowsHipAliases(symbols.map(symbol => symbol + " T 0 0").join("\n"), names);
    assert.deepEqual(aliases, names.map((name, index) => "/alternatename:" + name + "=" + symbols[index]));
  });
  test("missing, duplicated, foreign or excessive thunk mappings fail closed", () => {
    const names = windowsHipBindings(metadata).names, text = symbols.map(symbol => symbol + " T 0 0").join("\n");
    for (const invalid of ["", text + "\n" + symbols[0] + " T 0 0", text.replace("mom_hip_thunk_0", "foreign_thunk_0"),
      text.replace("mom_hip_thunk_0", "mom_hip_thunk_999"), text.replace(" T 0 0", "=injected T 0 0"),
      text.replace(" T 0 0", " D 0 0"), "??$call@$1?mom_hip_name_0@@ T 0 0", " ".repeat((4 << 20) + 1)]) {
      assert.throws(() => windowsHipAliases(invalid, names), /HIP|thunk/);
    }
  });
  test("driver aliases use source-owned names without depending on compiler C++ mangling", () => {
    for (const kind of /** @type {const} */ (["hip", "cuda"])) {
      const names = kind === "hip" ? ["hipGetDeviceCount", "hipGetErrorString"] : ["cuInit", "nvmlErrorString"];
      const symbols = names.map((_, index) => "mom_" + kind + "_thunk_" + index);
      const definitions = ["?unrelated_compiler_private_function@@ T 0 8", ...symbols.map(symbol => symbol + " T 0 8")].join("\n");
      assert.deepEqual(windowsTypedDriverAliases(definitions, names, kind), names.map((name, index) => "/alternatename:" + name + "=" + symbols[index]));
    }
  });
});
