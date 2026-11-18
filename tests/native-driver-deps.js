"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {dynamicInfo, systemPath, searchPaths, inspectDrivers} = require("./tools/inspect-native-driver-deps");
const {isGpuDriver, nativeDependencies} = require("../scripts/native-audit");

test.describe("Native system-driver ELF metadata, not application exemptions", () => {
  const section = "Dynamic section at offset 0x123 contains 3 entries:\n" +
    "0x01 (NEEDED) Shared library: [libstdc++.so.6]\n0x0e (SONAME) Library soname: [libigdrcl.so]\n0x00 (NULL) 0x0\n";
  test("records exact driver SONAME and NEEDED metadata without loading a library", () => {
    assert.deepEqual(dynamicInfo(section), {soname: "libigdrcl.so", needed: ["libstdc++.so.6"], search: [], glibc: []});
  });
  test("a transitive dependency report does not weaken the miner's own ELF dependency audit", () => {
    assert.throws(() => nativeDependencies("(NEEDED) [libstdc++.so.6]", {openclDriver: true}), /application dependency/);
    assert.equal(isGpuDriver("libstdc++.so.6"), false);
  });
  test("only known GPU entrypoint identities can root a driver report", () => {
    for (const name of ["libcuda.so.1", "libze_intel_gpu.so.1", "libigdrcl.so", "libnvidia-opencl.so.1", "libamdocl64.so.2"]) {assert.equal(isGpuDriver(name), true);}
    for (const name of ["libOpenCL.so.1", "libLLVM.so.21", "libigc.so.2", "libintelocl.so", "mom.node"]) {assert.equal(isGpuDriver(name), false);}
  });
  test("missing and truncated readelf output is never interpreted as an empty closure", () => {
    for (const text of ["", "There is no dynamic section", section.replace("(NULL)", "(OTHER)"), section.replace("Dynamic section", "Other section")]) {
      assert.throws(() => dynamicInfo(text), /complete driver ELF/);
    }
  });
  test("malformed tags cannot disappear from the dependency graph", () => {
    assert.throws(() => dynamicInfo(section.replace("[libstdc++.so.6]", "[missing bracket")), /dependency identity/);
  });
  test("duplicate or path-valued SONAMEs fail explicitly", () => {
    assert.throws(() => dynamicInfo(section + "0x0e (SONAME) [other.so]\n"), /dependency identity/);
    assert.throws(() => dynamicInfo(section.replace("[libigdrcl.so]", "[/tmp/libigdrcl.so]")), /dependency identity/);
  });
  test("relative, absolute and escaped dependency paths are not SONAMEs", () => {
    for (const value of ["../libevil.so", "/tmp/libevil.so", "lib\\evil.so", "libevil.so\nother"]) {
      assert.throws(() => dynamicInfo(section.replace("libstdc++.so.6", value)), /dependency identity/);
    }
  });
  test("an actual empty NEEDED set remains distinct from missing tool output", () => {
    assert.deepEqual(dynamicInfo("Dynamic section at offset 0x10 contains 1 entries:\n0x00 (NULL) 0x0\n"), {soname: null, needed: [], search: [], glibc: []});
  });
  test("driver search metadata remains data, not evaluated commands", () => {
    assert.deepEqual(dynamicInfo(section + "0x01 (RUNPATH) [$ORIGIN/driver]\n").search, ["$ORIGIN/driver"]);
  });
  test("only imported glibc versions determine driver OS requirements", () => {
    const versions = "Version definition section '.gnu.version_d':\nName: GLIBC_2.99\n" +
      "Version needs section '.gnu.version_r':\nName: GLIBC_2.38\nName: GLIBCXX_3.4\nName: GLIBC_2.38\n";
    assert.deepEqual(dynamicInfo(section + versions).glibc, ["2.38"]);
  });
  test("non-numeric glibc ABI features remain visible rather than guessed into a version", () => {
    assert.deepEqual(dynamicInfo(section + "Version needs section '.gnu.version_r':\nName: GLIBC_ABI_DT_RELR\n").glibc, ["ABI_DT_RELR"]);
  });
});

test.describe("Native system-driver path and input boundaries", () => {
  test("only normalized OS library paths are admitted", () => {
    for (const file of ["/usr/lib/a.so", "/usr/lib64/a.so", "/lib/a.so", "/lib64/a.so"]) {assert.equal(systemPath(file), true);}
    for (const file of ["/tmp/a.so", "/opt/sdk/a.so", "/usr/library/a.so", "/usr/lib/../tmp/a.so", "/usr/lib//a.so", "/usr/lib/a\nb.so", "/usr/lib/a\\b.so"]) {
      assert.equal(systemPath(file), false, file);
    }
  });
  test("literal origin forms resolve within OS libraries", () => {
    // eslint-disable-next-line no-template-curly-in-string -- Literal ELF loader syntax, not JS interpolation.
    assert.deepEqual(searchPaths(["$ORIGIN/driver:${ORIGIN}/../compiler"], "/usr/lib/vendor"), ["/usr/lib/vendor/driver", "/usr/lib/compiler"]);
  });
  test("a normalized origin path may select the OS library root itself", () => {
    assert.deepEqual(searchPaths(["$ORIGIN/.."], "/usr/lib/vendor"), ["/usr/lib"]);
  });
  test("unknown variables, command syntax and empty/current-directory search entries fail", () => {
    for (const value of ["$HOME/driver", "$(pwd)/driver", "$ORIGINevil", "", ":/usr/lib", "/usr/lib:", "relative"]) {
      assert.throws(() => searchPaths([value], "/usr/lib/vendor"), /search path/);
    }
  });
  test("empty driver RUNPATH entries require an explicit root cwd assumption", {skip: process.platform !== "linux"}, () => {
    assert.deepEqual(searchPaths(["$ORIGIN/../lib:"], "/usr/lib/vendor", "/"), ["/usr/lib/lib", "/"]);
    for (const cwd of [undefined, "/tmp", "/usr/lib", "/some/user/directory"]) {
      assert.throws(() => searchPaths(["$ORIGIN/../lib:"], "/usr/lib/vendor", cwd), /search path/);
    }
  });
  test("origin traversal cannot escape to a user or SDK directory", () => {
    for (const value of ["$ORIGIN/../../../tmp", "/opt/rocm/lib"]) {assert.throws(() => searchPaths([value], "/usr/lib/vendor"), /search path/);}
  });
  test("empty, duplicate, excessive and non-driver roots fail before file/tool access", () => {
    for (const roots of [[], ["/usr/lib/libcuda.so.1", "/usr/lib/libcuda.so.1"], Array.from({length: 13}, (_, i) => `/usr/lib/${i}/libcuda.so.1`),
      ["/usr/lib/libLLVM.so.21"], ["/usr/lib/libOpenCL.so.1"], ["/tmp/libcuda.so.1"]]) {
      assert.throws(() => inspectDrivers(roots), /unique absolute system GPU-driver/);
    }
  });
  test("the CLI rejects untrusted roots with all external tools unavailable", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "./tools/inspect-native-driver-deps.js"), "/tmp/libcuda.so.1"],
      {encoding: "utf8", timeout: 5000, env: {...process.env, PATH: "/unavailable"}});
    assert.equal(result.status, 1); assert.match(result.stderr, /unique absolute system GPU-driver/); assert.doesNotMatch(result.stderr, /spawn|ENOENT/);
  });
  test("late components cannot turn an arbitrary system/compiler library into a driver root", {skip: process.platform !== "linux"}, () => {
    for (const lateLibraries of [["/usr/lib/libLLVM.so.21"], ["/tmp/libigc.so.2"], ["/usr/lib/libigc.so.2", "/usr/lib/libigc.so.2"]]) {
      assert.throws(() => inspectDrivers(["/usr/lib/libze_intel_gpu.so.1"], {lateLibraries}), /only explicit system Intel/);
    }
    assert.throws(() => inspectDrivers(["/usr/lib/libcuda.so.1"], {lateLibraries: ["/usr/lib/libigc.so.2"]}), /only explicit system Intel/);
    assert.throws(() => inspectDrivers(["/usr/lib/libze_intel_gpu.so.1"], {workingDirectory: "/tmp"}), /verified native worker root/);
  });
});
