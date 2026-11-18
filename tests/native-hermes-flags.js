"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs");
const path = require("node:path"), os = require("node:os"), {spawnSync} = require("node:child_process");

test.describe("Native Hermes MSVC-front-end exception and RTTI flags", () => {
  const fixture = {root: ""};
  const cmake = spawnSync("cmake", ["--version"], {encoding: "utf8", timeout: 5000});
  test.before(() => {
    if (cmake.status !== 0) {return;}
    fixture.root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hermes-flags-"));
    for (const mode of ["before", "after"]) {
      const root = path.join(fixture.root, mode);
      fs.mkdirSync(path.join(root, "cmake/modules"), {recursive: true});
      fs.mkdirSync(path.join(root, "API/jsi/jsi"), {recursive: true});
      fs.copyFileSync(path.join(__dirname, "native/hermes-flags/Hermes.cmake"), path.join(root, "cmake/modules/Hermes.cmake"));
      fs.copyFileSync(path.join(__dirname, "native/hermes-flags/jsi.cmake"), path.join(root, "API/jsi/jsi/CMakeLists.txt"));
      if (mode === "after") {
        // Exercise the flag hunks against these original fixtures; DateUtil's separate
        // source-optimization hunk is qualified by the native Hermes build tests.
        const applied = spawnSync("git", ["apply", "--include=API/jsi/jsi/CMakeLists.txt", "--include=cmake/modules/Hermes.cmake",
          path.join(__dirname, "../runtime/hermes-clang-cl.patch")], {cwd: root, encoding: "utf8", timeout: 5000});
        assert.equal(applied.status, 0, applied.error?.message || applied.stderr);
      }
    }
  });
  test.after(() => {if (fixture.root) {fs.rmSync(fixture.root, {recursive: true, force: true});}});
  for (const id of ["GNU", "Clang", "MSVC", "clang-cl"]) {
    for (const enabled of [false, true]) {
      test(`${id}: ${enabled ? "enabled" : "disabled"} exceptions/RTTI and required JSI unwinding`, (t) => {
        if (cmake.status !== 0) {t.skip("CMake unavailable"); return;}
        const microsoft = id === "MSVC" || id === "clang-cl", flags = [];
        for (const mode of ["before", "after"]) {
          const root = path.join(fixture.root, mode), output = path.join(root, `${id}-${enabled}.txt`), script = output + ".cmake";
          const quote = (/** @type {string} */ value) => JSON.stringify(value.replace(/\\/g, "/"));
          // Match upstream policies: OLD CMP0054 dereferences the quoted "MSVC" match token.
          fs.writeFileSync(script, `cmake_minimum_required(VERSION 3.20)\nset(CMAKE_CXX_COMPILER_ID ${id === "clang-cl" ? "Clang" : id})\n` +
            `set(GCC_COMPATIBLE ${id !== "MSVC"})\nset(CLANG_CL ${id === "clang-cl"})\nset(MSVC ${microsoft})\n` +
            `set(HERMES_ENABLE_EH ${enabled})\nset(HERMES_ENABLE_RTTI ${enabled})\n` +
            `function(target_compile_options)\nfile(APPEND ${quote(output)} "\u0024{ARGV}\\n")\nendfunction()\n` +
            ["add_library", "target_include_directories"].map((name) => `function(${name})\nendfunction()\n`).join("") +
            `include(${quote(path.join(root, "cmake/modules/Hermes.cmake"))})\nhermes_update_cxx_flags(mom)\n` +
            `include(${quote(path.join(root, "API/jsi/jsi/CMakeLists.txt"))})\n`);
          const result = spawnSync("cmake", ["-P", script], {encoding: "utf8", timeout: 5000});
          assert.equal(result.status, 0, result.error?.message || result.stderr);
          flags.push(fs.readFileSync(output, "utf8"));
        }
        const expected = microsoft ? [enabled ? "/EHsc" : "/EHs-c-", enabled ? "/GR" : "/GR-"] :
          [enabled ? "-fexceptions" : "-fno-exceptions", enabled ? "-frtti" : "-fno-rtti"];
        for (const value of expected) {assert.ok(flags[1]?.includes(value), value);}
        if (id === "clang-cl") {assert.notEqual(flags[0], flags[1]); assert.match(flags[1] || "", /jsi;PRIVATE;.*\/EHsc/);} else {assert.equal(flags[0], flags[1]);}
      });
    }
  }
});
