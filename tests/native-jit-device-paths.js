"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path");
const {spawnSync} = require("node:child_process");

const sourcePath = "sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp";
const patch = fs.readFileSync(path.join(__dirname, "../runtime/dpcpp-static-jit.patch"), "utf8");
/** @typedef {{windows: boolean, staticJit: boolean}} MacroSet */

function patchedFunction() {
  const marker = `diff --git a/${sourcePath} b/${sourcePath}`;
  const start = patch.indexOf(marker);
  assert.notEqual(start, -1, "static-JIT patch must change DeviceCompilation.cpp");
  const nextFile = patch.indexOf("\ndiff --git ", start + marker.length);
  const filePatch = patch.slice(start, nextFile < 0 ? patch.length : nextFile);
  const hunk = filePatch.split(/(?=^@@ )/m).find(chunk => chunk.includes("getLibPathSuffix"));
  assert.ok(hunk, "patch must include the getLibPathSuffix hunk");
  const source = hunk.split(/\r?\n/).flatMap(line => {
    if (line.startsWith("+") && !line.startsWith("+++")) {return [line.slice(1)];}
    if (line.startsWith(" ")) {return [line.slice(1)];}
    return [];
  }).join("\n");
  assert.match(source, /#if defined\(_WIN32\) && !defined\(MOM_DPCPP_STATIC_JIT\)/);
  assert.match(source, /DPCPP_VERSION_MAJOR/);
  assert.match(source, /Static overlays share one virtual layout/);
  return source;
}

/** @param {string} source @param {MacroSet} macros */
function selectedFormat(source, {windows, staticJit}) {
  const args = ["-E", "-P", "-x", "c++", '-DCLANG_INSTALL_LIBDIR_BASENAME="lib"', "-DDPCPP_VERSION_MAJOR=7"];
  if (windows) {args.push("-D_WIN32");}
  if (staticJit) {args.push("-DMOM_DPCPP_STATIC_JIT");}
  args.push("-");
  const result = spawnSync(process.env["MOM_TEST_CXX"] || "c++", args, {
    input: source, encoding: "utf8", timeout: 10000, maxBuffer: 64 * 1024,
  });
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
  const formatString = result.stdout.match(/return llvm::formatv\("([^"]+)"/)?.[1];
  assert.ok(formatString, "preprocessed branch must return its library suffix");
  return formatString.replace("{0}", "lib").replace("{1}", "7");
}

test.describe("Static JIT device library paths", () => {
  test("static source-JIT path matches the embedded Windows PTX manifest", {skip: process.platform !== "linux" && "uses the host C++ preprocessor"}, () => {
    const revised = patchedFunction();
    const original = `std::string getLibPathSuffix() {
#ifdef _WIN32
  return llvm::formatv("/{0}/", CLANG_INSTALL_LIBDIR_BASENAME);
#else
  return llvm::formatv("/{0}/dpcpp-{1}/sycl/", CLANG_INSTALL_LIBDIR_BASENAME,
                       DPCPP_VERSION_MAJOR);
#endif
}`;
    const manifest = new Set(["lib/dpcpp-7/sycl/devicelib-nvptx64-nvidia-cuda.bc"]);
    const library = "devicelib-nvptx64-nvidia-cuda.bc";
    /** @param {string} format @param {MacroSet} macros */
    const lookup = (format, macros) => manifest.has((selectedFormat(format, macros) + library).replace(/^\/+/, ""));

    assert.equal(selectedFormat(original, {windows: true, staticJit: true}), "/lib/");
    assert.equal(lookup(original, {windows: true, staticJit: true}), false, "old Windows static path misses the manifest entry");
    assert.equal(selectedFormat(revised, {windows: true, staticJit: false}), "/lib/", "dynamic Windows layout stays unchanged");
    assert.equal(selectedFormat(revised, {windows: true, staticJit: true}), "/lib/dpcpp-7/sycl/");
    assert.equal(selectedFormat(revised, {windows: false, staticJit: true}), "/lib/dpcpp-7/sycl/", "Linux layout stays unchanged");
    assert.equal(lookup(revised, {windows: true, staticJit: true}), true, "static Windows PTX lookup resolves the manifest path");
  });
});
