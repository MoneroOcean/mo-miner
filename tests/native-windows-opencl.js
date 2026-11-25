"use strict";
const fs = require("node:fs"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {tempDir} = require("./common/temp-dir");
const repo = path.resolve(__dirname, ".."), read = (/** @type {string} */ name) => fs.readFileSync(path.join(repo, name), "utf8");
const patch = read("runtime/opencl-static-windows.patch");

test.describe("Native Windows OpenCL driver-only boundary", () => {
  test("the reviewed SDK patch has valid unified-diff boundaries", () => {
    const result = spawnSync("git", ["apply", "--numstat", path.join(repo, "runtime/opencl-static-windows.patch")], {cwd: repo, encoding: "utf8", timeout: 5000});
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout.trim().split(/\r?\n/).length, 3);
  });
  for (const [compiler, language, standard] of [["cc", "c", "c11"], ["c++", "c++", "c++17"]]) {
    test(`real driver path boundary rejects traversal, search paths and unsafe aliases (${language})`, {skip: process.platform !== "linux"}, t => {
      const root = tempDir(t, "mom-windows-opencl-");
      const executable = path.join(root, "probe");
      const built = spawnSync(compiler || "", ["-x", language || "", "-std=" + standard, "-O2", "-Wall", "-Wextra", "-Werror", "-D_WIN32",
        "-I", path.join(__dirname, "logic/opencl-driver"), "-I", repo, path.join(__dirname, "logic/windows-opencl-driver.c"), "-o", executable], {encoding: "utf8", timeout: 30000});
      assert.equal(built.status, 0, built.error?.message || built.stderr);
      const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000}); assert.equal(result.status, 0, result.stderr);
    });
  }
  test("the ICD cannot enumerate environment-selected application DLLs", () => {
    assert.match(patch, /void khrIcdVendorsEnumerateEnv\(void\)\n \{\n\+#if defined\(MOM_NATIVE_STATIC_OPENCL_WINDOWS\)[\s\S]*?\+ {4}return;\n\+#else/);
    assert.match(patch, /khrIcd_free_getenv\(icdFilenames\);\n {5}}\n\+#endif/);
  });
  test("the original ICD driver load is replaced only in native Windows mode", () => {
    assert.match(patch, /\+#if defined\(MOM_NATIVE_STATIC_OPENCL_WINDOWS\)\n\+ {4}return mom_native_opencl_load_driver\(libraryName\);\n\+#else/);
    assert.match(patch, /return \(void\*\)hTemp;\n\+#endif/);
  });
  for (const name of ["dxgi.dll", "gdi32.dll"]) {
    test(`${name} uses OS-only search, never the current directory`, () => {
      const line = patch.split("\n").find(line => line.startsWith("+") && line.includes(JSON.stringify(name)));
      assert.ok(line); assert.match(line, /LoadLibraryExA\(.*NULL, LOAD_LIBRARY_SEARCH_SYSTEM32\)/);
    });
  }
  test("native SDK configuration rejects an unpatched ICD before certifying its archive", () => {
    const cmake = read("tests/native/dpcpp-static/CMakeLists.txt");
    assert.match(cmake, /if\(opencl_env_guard EQUAL -1 OR opencl_driver_guard EQUAL -1\)/);
    assert.match(cmake, /target_compile_definitions\(OpenCL PRIVATE MOM_NATIVE_STATIC_OPENCL_WINDOWS\)/);
    assert.match(cmake, /target_include_directories\(OpenCL PRIVATE "\$\{MOM_SOURCE_REPO\}\/runtime"\)/);
  });
  test("optional ICD plugin layers remain disabled", () => {
    const cmake = read("tests/native/dpcpp-static/CMakeLists.txt");
    for (const name of ["ENABLE_OPENCL_LAYERS", "ENABLE_OPENCL_LAYERINFO"]) {assert.match(cmake, new RegExp("set\\(" + name + " OFF CACHE BOOL"));}
  });
  test("Linux cannot opt into the Windows loader exception", {skip: process.platform !== "linux"}, t => {
    const root = tempDir(t, "mom-windows-opencl-platform-");
    const result = spawnSync("cc", ["-std=c11", "-I", path.join(__dirname, "logic/opencl-driver"), "-I", repo,
      "-c", path.join(__dirname, "logic/windows-opencl-driver.c"), "-o", path.join(root, "bad.o")], {encoding: "utf8", timeout: 30000});
    assert.notEqual(result.status, 0); assert.match(result.stderr, /Windows-only/);
  });
});
