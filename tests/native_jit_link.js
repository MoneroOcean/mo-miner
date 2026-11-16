"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {lldInputs} = require("../scripts/probe-native-jit");
const {rocmInputs} = require("../scripts/probe-static-sycl");

test.describe("Native AMD linker target registration", {skip: process.platform === "win32"}, () => {
  // Compile the actual patched function in both modes; fake registrations never load a driver.
  const patch = fs.readFileSync(path.join(__dirname, "../runtime/lld-native-targets.patch"), "utf8");
  const functionText = patch.split("\n").filter((line) => /^[ +]/.test(line) && !line.startsWith("+++"))
    .map((line) => line.slice(1)).join("\n").match(/static void initLLVM\(\) \{[\s\S]*?\n\}/)?.[0];
  const amd = ["TargetInfo", "Target", "TargetMC", "AsmPrinter", "AsmParser"].map((name) => "LLVMInitializeAMDGPU" + name);
  const all = ["Targets", "TargetMCs", "AsmPrinters", "AsmParsers"].map((name) => "InitializeAll" + name);
  for (const specialized of [false, true]) {
    for (const calls of [1, 2, 3, 4]) {
      test(`${specialized ? "native AMD" : "unchanged general"} mode registers only its own targets on ${calls} calls`, (t) => {
        assert.ok(functionText, "Missing patched original initializer");
        const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-lld-targets-"));
        t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
        // Do not even declare other-target initializers in native mode: an accidental reference
        // would retain those backends in the real link and must fail this fixture's compilation.
        const names = specialized ? amd : all;
        const code = "static unsigned counts[5]{};\n" +
          names.map((name, i) => `static void ${name}() {++counts[${i}];}`).join("\n") + "\n" + functionText +
          `\nint main() {for (unsigned i=0;i<${calls};++i) initLLVM();` +
          names.map((_, i) => `if(counts[${i}]!=${calls})return 86;`).join("") + "return 0;}\n";
        const source = path.join(directory, "registry.cpp"), executable = path.join(directory, "registry");
        fs.writeFileSync(source, code);
        const build = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-DNDEBUG", "-Wall", "-Wextra", "-Werror",
          ...(specialized ? ["-DMOM_NATIVE_LLD_AMDGPU_ONLY"] : []), source, "-o", executable], {encoding: "utf8", timeout: 30000});
        assert.equal(build.status, 0, build.error?.message || build.stderr);
        const run = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
        assert.equal(run.status, 0, run.error?.message || run.stderr);
      });
    }
  }
});

test.describe("Native in-process ELF linker SDK admission", () => {
  /** @param {import('node:test').TestContext} t */
  function sdk(t) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-lld-inputs-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    for (const name of ["include/lld/Common", "lib/cmake/lld"]) {fs.mkdirSync(path.join(root, name), {recursive: true});}
    fs.writeFileSync(path.join(root, "include/lld/Common/Driver.h"), "fixture\n");
    fs.writeFileSync(path.join(root, "lib/cmake/lld/LLDConfigVersion.cmake"), 'set(PACKAGE_VERSION "21.1.8")\n');
    for (const name of ["ELF", "Common"]) {fs.writeFileSync(path.join(root, "lib", `liblld${name}.a`), "!<arch>\n");}
    return root;
  }
  test("the original primitive has no additional SDK requirement", () => {assert.deepEqual(lldInputs(undefined, "21.1.8"), []);});
  test("matching self-contained archives are explicit link inputs", (t) => {
    const root = sdk(t); assert.deepEqual(lldInputs(root, "21.1.8"), ["ELF", "Common"].map((name) => path.join(root, "lib", `liblld${name}.a`)));
  });
  test("linker SDK admission never loads entire archives to inspect eight-byte signatures", (t) => {
    const root = sdk(t);
    t.mock.method(fs, "readFileSync", (/** @type {unknown} */ file) => {
      assert.equal(file, path.join(root, "lib/cmake/lld/LLDConfigVersion.cmake"));
      return 'set(PACKAGE_VERSION "21.1.8")\n';
    });
    try {assert.equal(lldInputs(root, "21.1.8").length, 2);} finally {t.mock.restoreAll();}
  });
  test("relative SDK roots cannot depend on the caller's working directory", () => {assert.throws(() => lldInputs("relative", "21.1.8"), /absolute/);});
  for (const file of ["include/lld/Common/Driver.h", "lib/liblldELF.a", "lib/liblldCommon.a", "lib/cmake/lld/LLDConfigVersion.cmake"]) {
    test(`missing ${file} fails closed`, (t) => {
      const root = sdk(t); fs.unlinkSync(path.join(root, file)); assert.throws(() => lldInputs(root, "21.1.8"), /matching LLVM/);
    });
  }
  for (const version of ["20.1.8", "21.0.8", "21.1.7"]) {
    test(`mixed LLVM version ${version} is rejected`, (t) => {assert.throws(() => lldInputs(sdk(t), version), /matching LLVM/);});
  }
  /** @type {Array<[string, string]>} */
  const invalidArchives = [["thin", "!<thin>\n"], ["ELF shared object", "\x7fELF"], ["empty", ""]];
  for (const [name, signature] of invalidArchives) {
    test(`non-self-contained ${name} archive is rejected`, (t) => {
      const root = sdk(t); fs.writeFileSync(path.join(root, "lib/liblldELF.a"), signature);
      assert.throws(() => lldInputs(root, "21.1.8"), /static ELF\/Common/);
    });
  }
  test("importing SDK validation cannot start a compiler or implicitly run a GPU", () => {
    const result = spawnSync(process.execPath, ["-e", 'require("./scripts/probe-native-jit")'],
      {cwd: path.join(__dirname, ".."), encoding: "utf8", timeout: 5000,
        env: {...process.env, LLVM_CONFIG: "/unavailable", MOM_AOT_LLD_ROOT: "/unavailable"}});
    assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, ""); assert.equal(result.stderr, "");
  });
  test("importing the AMD qualification helper never starts its build or workload", () => {
    const result = spawnSync(process.execPath, ["-e", 'require("./scripts/probe-native-amdgpu")'],
      {cwd: path.join(__dirname, ".."), encoding: "utf8", timeout: 5000,
        env: {...process.env, LLVM_CONFIG: "/unavailable", PATH: "/unavailable"}});
    assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, ""); assert.equal(result.stderr, "");
  });
  test("AMD qualification requires explicit source and build inputs before invoking tools", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/probe-native-amdgpu.js")],
      {encoding: "utf8", timeout: 5000, env: {...process.env, LLVM_CONFIG: "/unavailable", PATH: "/unavailable"}});
    assert.equal(result.status, 1); assert.equal(result.stdout, ""); assert.match(result.stderr, /Usage \(Linux\)/);
    assert.doesNotMatch(result.stderr, /ENOENT|spawn/);
  });
});

test.describe("Windows Intel direct-LLD delay-load adaptation", () => {
  test("the maintained adapter removes only the known driver wrappers from LLVMSupport links", (t) => {
    const cmake = fs.readFileSync(path.join(__dirname, "native/dpcpp-jit/CMakeLists.txt"), "utf8");
    const start = cmake.indexOf("function(mom_strip_direct_lld_delayload_wrappers target)"), end = cmake.indexOf("\nendfunction()", start);
    assert.ok(start >= 0 && end > start, "Missing maintained LLVMSupport link-property adapter");
    const adapter = cmake.slice(start, end + "\nendfunction()".length);
    const wrapped = "$<$<OR:$<LINK_LANG_AND_ID:C,IntelLLVM>,$<LINK_LANG_AND_ID:CXX,IntelLLVM>,$<LINK_LANG_AND_ID:Fortran,IntelLLVM>>:-Qoption,link,>";
    const delayload = "$<$<NOT:$<LINK_LANGUAGE:Swift>>:delayimp " + wrapped + "-delayload:shell32.dll " +
      wrapped + "-delayload:ole32.dll " + wrapped + "-delayload:kernel32.dll>";
    const unrelated = "-Qoption,link,-INCLUDE:malloc";
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-intel-lld-adapter-"));
    t.after(() => fs.rmSync(fixture, {recursive: true, force: true}));
    fs.writeFileSync(path.join(fixture, "CMakeLists.txt"), `cmake_minimum_required(VERSION 3.20)
project(MomIntelLldAdapter NONE)
add_library(LLVMSupport INTERFACE)
set(wrapped_links [==[${delayload};${unrelated};ordinary.lib]==])
set_property(TARGET LLVMSupport PROPERTY LINK_LIBRARIES "\${wrapped_links}")
set_property(TARGET LLVMSupport PROPERTY INTERFACE_LINK_LIBRARIES "\${wrapped_links}")
${adapter}
mom_strip_direct_lld_delayload_wrappers(LLVMSupport)
mom_strip_direct_lld_delayload_wrappers(LLVMSupport)
get_target_property(local_links LLVMSupport LINK_LIBRARIES)
get_target_property(interface_links LLVMSupport INTERFACE_LINK_LIBRARIES)
file(WRITE "\${CMAKE_BINARY_DIR}/link-properties.txt" "\${local_links}\\n\${interface_links}\\n")
`);
    const cmakeBin = process.env["CMAKE"] || "cmake", configured = spawnSync(cmakeBin, ["-G", "Ninja", "-S", fixture, "-B", path.join(fixture, "build")],
      {encoding: "utf8", timeout: 30000});
    assert.equal(configured.status, 0, configured.error?.message || configured.stdout + configured.stderr);
    const properties = fs.readFileSync(path.join(fixture, "build/link-properties.txt"), "utf8");
    assert.doesNotMatch(properties, /-Qoption,link,>-delayload:(?:shell32|ole32)\.dll/);
    assert.equal((properties.match(/>-delayload:(?:shell32|ole32)\.dll/g) || []).length, 4);
    assert.match(properties, /-Qoption,link,>-delayload:kernel32\.dll/, "unknown delay-load libraries are unchanged");
    assert.equal((properties.match(/-Qoption,link,-INCLUDE:malloc/g) || []).length, 2);
    assert.equal((properties.match(/ordinary\.lib/g) || []).length, 2);
  });
  test("the target rewrite is guarded by Windows, IntelLLVM, and direct lld-link rules", () => {
    const cmake = fs.readFileSync(path.join(__dirname, "native/dpcpp-jit/CMakeLists.txt"), "utf8");
    const start = cmake.indexOf("get_filename_component(mom_jit_linker_name"), end = cmake.indexOf("set(native_dpcpp_jit_sources", start);
    assert.ok(start >= 0 && end > start);
    const gate = cmake.slice(start, end);
    assert.match(gate, /if\(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"/);
    assert.match(gate, /mom_jit_linker_name MATCHES "\^lld-link/);
    assert.match(gate, /CMAKE_C_LINK_EXECUTABLE MATCHES "<CMAKE_LINKER>"/);
    assert.match(gate, /CMAKE_CXX_LINK_EXECUTABLE MATCHES "<CMAKE_LINKER>"/);
    assert.match(gate, /mom_strip_direct_lld_delayload_wrappers\(LLVMSupport\)/);
  });
});

test.describe("Native application-library load boundary", {skip: process.platform === "win32"}, () => {
  const files = {directory: "", executable: ""};
  test.before(() => {
    files.directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-driver-library-"));
    files.executable = path.join(files.directory, "boundary");
    const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-DNDEBUG", "-Wall", "-Wextra", "-Werror",
      path.join(__dirname, "native/driver_library.cpp"), path.join(__dirname, "../runtime/driver-library.cpp"), "-o", files.executable],
    {encoding: "utf8", timeout: 30000});
    assert.equal(result.status, 0, result.stderr);
  });
  test.after(() => {if (files.directory) {fs.rmSync(files.directory, {recursive: true, force: true});}});
  test("loader-spy failures stay observable under NDEBUG", () => {
    for (const args of [["libcuda.so.1", "denied"], ["libOpenCL.so.1", "allowed"]]) {
      const result = spawnSync(files.executable, args, {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 86, result.stderr);
    }
  });
  test("rejected libraries provide consumable thread-local loader diagnostics without loading code", () => {
    const executable = path.join(files.directory, "error-contract");
    const built = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      path.join(__dirname, "native/driver_library_error.cpp"), path.join(__dirname, "../runtime/driver-library.cpp"),
      "-Wl,--wrap=dlopen", "-Wl,--wrap=dlerror", "-ldl", "-pthread", "-o", executable], {encoding: "utf8", timeout: 30000});
    assert.equal(built.status, 0, built.stderr);
    const run = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
    assert.equal(run.status, 0, run.stderr);
  });
  const allowed = ["<self>", "libcuda.so.1", "libnvidia-ptxjitcompiler.so.1", "libnvidia-ml.so.1"];
  const denied = ["", "libhsa-amd-aqlprofile64.so", "libhiprtc.so.7", "libamd_comgr.so.3", "libamdhip64.so.7", "libLLVM.so.21",
    "mom.node", "/tmp/libcuda.so.1", "../libcuda.so.1", "libcuda.so", "libcuda.so.1.extra", "libCUDA.so.1",
    "/tmp/libnvidia-ml.so.1", "libnvidia-ml.so"];
  for (const name of [...allowed, ...denied]) {
    test(`${JSON.stringify(name)} is ${allowed.includes(name) ? "forwarded unchanged" : "rejected before loader access"}`, () => {
      const result = spawnSync(files.executable, [name, allowed.includes(name) ? "allowed" : "denied"], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, "");
    });
  }
  test.describe("Level Zero system GPU driver identities", () => {
    const intel = ["libze_intel_gpu.so.1", "libze_intel_gpu_legacy1.so.1"];
    for (const name of [...intel, "libze_loader.so.1", "libze_intel_npu.so.1", "libze_intel_vpu.so.1",
      "libze_intel_gpu.so", "libze_intel_gpu.so.1.extra", "libZE_intel_gpu.so.1", "/tmp/libze_intel_gpu.so.1",
      "../libze_intel_gpu.so.1", "/usr/lib/libze_intel_gpu.so.1", "libur_adapter_level_zero.so.0"]) {
      test(`${name} is ${intel.includes(name) ? "forwarded unchanged" : "rejected"}`, () => {
        const result = spawnSync(files.executable, [name, intel.includes(name) ? "allowed" : "denied"], {encoding: "utf8", timeout: 5000});
        assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, "");
      });
    }
  });
  test.describe("OpenCL system GPU ICD identities", () => {
    const drivers = ["libigdrcl.so", "libigdrcl_legacy1.so", "libnvidia-opencl.so.1",
      "libamdocl64.so", "libamdocl64.so.2", "libRusticlOpenCL.so.1"];
    for (const name of drivers) {
      test(`${name} supports SONAME and OS-library ICD registry paths`, () => {
        for (const prefix of ["", "/usr/lib/x86_64-linux-gnu/intel-opencl/", "/usr/lib64/", "/lib/", "/lib64/"]) {
          const result = spawnSync(files.executable, [prefix + name, "allowed"], {encoding: "utf8", timeout: 5000});
          assert.equal(result.status, 0, result.stderr);
        }
      });
    }
    for (const name of ["/tmp/libigdrcl.so", "../libigdrcl.so", "/opt/sdk/libigdrcl.so",
      "/usr/lib/../tmp/libigdrcl.so", "/usr/lib/./libigdrcl.so", "/usr/lib//libigdrcl.so",
      "/usr/lib/a\n/libigdrcl.so", "/usr/lib/a\\b/libigdrcl.so",
      "libigdrcl.so.extra", "libIGDRCL.so", "libOpenCL.so.1", "libintelocl.so",
      "libur_adapter_opencl.so.0", "/usr/lib/libLLVM.so.21"]) {
      test(`${name} cannot turn the ICD exception into an application/plugin loader`, () => {
        const result = spawnSync(files.executable, [name, "denied"], {encoding: "utf8", timeout: 5000});
        assert.equal(result.status, 0, result.stderr);
      });
    }
    test("the Intel CPU ICD path is accepted only for the exact explicit portable-CPU opt-in", () => {
      const cpuPath = "/opt/intel/oneapi/redist/lib/libintelocl.so";
      /** @param {string} name @param {boolean} expected @param {string} [optIn] */
      const invoke = (name, expected, optIn) => {
        const env = {...process.env};
        delete env["MOM_REQUIRE_PORTABLE_CPU_TESTS"];
        if (optIn !== undefined) {env["MOM_REQUIRE_PORTABLE_CPU_TESTS"] = optIn;}
        const result = spawnSync(files.executable, [name, expected ? "allowed" : "denied"],
          {encoding: "utf8", timeout: 5000, env});
        assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, "");
      };
      invoke(cpuPath, false);
      invoke(cpuPath, true, "1");
      for (const value of ["", "0", "true", "01", "1 "]) {invoke(cpuPath, false, value);}
      for (const name of ["libintelocl.so", "/opt/intel/oneapi/redist/lib/libintelocl.so.2026.22.10.0",
        "/opt/intel/oneapi/compiler/latest/lib/libintelocl.so", "/opt/intel/oneapi/redist/lib/../lib/libintelocl.so"]) {
        invoke(name, false, "1");
      }
    });
  });
});

test.describe("Windows OpenCL loader portable-CPU opt-in (mock Win32 loader)", {skip: process.platform === "win32"}, () => {
  test("only the standalone Intel CPU ICD install path is added by the exact opt-in", (t) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-opencl-policy-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const windowsHeader = path.join(directory, "windows.h"), source = path.join(directory, "policy.cpp"), executable = path.join(directory, "policy");
    fs.writeFileSync(windowsHeader, `#pragma once
typedef unsigned int DWORD;
typedef unsigned int UINT;
typedef const char* LPCSTR;
typedef void* HANDLE;
typedef void* HMODULE;
#define MAX_PATH 260
#define LOAD_LIBRARY_SEARCH_SYSTEM32 0x00000800
#define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR 0x00000100
DWORD GetEnvironmentVariableA(LPCSTR, char*, DWORD);
UINT GetSystemDirectoryA(char*, UINT);
HMODULE LoadLibraryExA(LPCSTR, HANDLE, DWORD);
`);
    fs.writeFileSync(source, `#define _WIN32
#include <stdlib.h>
#include <string.h>
#include "runtime/windows-opencl-driver.h"
static unsigned loads = 0;
static char loaded[32768];
static DWORD loadFlags = 0;
DWORD GetEnvironmentVariableA(LPCSTR name, char* value, DWORD capacity) {
  const char* source = getenv(name);
  if (!source) return 0;
  size_t length = strlen(source);
  if (length >= capacity) return (DWORD)(length + 1);
  memcpy(value, source, length + 1);
  return (DWORD)length;
}
UINT GetSystemDirectoryA(char* value, UINT capacity) {
  const char* system = "C:\\\\Windows\\\\System32";
  size_t length = strlen(system);
  if (length + 1 > capacity) return (UINT)(length + 1);
  memcpy(value, system, length + 1);
  return (UINT)length;
}
HMODULE LoadLibraryExA(LPCSTR name, HANDLE, DWORD flags) {
  ++loads;
  strncpy(loaded, name, sizeof(loaded) - 1);
  loaded[sizeof(loaded) - 1] = 0;
  loadFlags = flags;
  return (HMODULE)1;
}
int main(int argc, char** argv) {
  if (argc != 3) return 86;
  const int expected = strcmp(argv[2], "allowed") == 0;
  HMODULE result = (HMODULE)mom_native_opencl_load_driver(argv[1]);
  if (loads != (unsigned)expected || result != (expected ? (HMODULE)1 : NULL)) return 86;
  if (expected && strcmp(loaded, argv[1]) != 0) return 86;
  const DWORD expectedFlags = strchr(argv[1], ':') ?
    LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR : LOAD_LIBRARY_SEARCH_SYSTEM32;
  if (expected && loadFlags != expectedFlags) return 86;
  return 0;
}
`);
    const built = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
      "-I", directory, "-I", path.join(__dirname, ".."), source, "-o", executable], {encoding: "utf8", timeout: 30000});
    assert.equal(built.status, 0, built.error?.message || built.stderr);
    /** @param {string} name @param {boolean} expected @param {string} [optIn] */
    const invoke = (name, expected, optIn) => {
      const env = {...process.env};
      delete env["MOM_REQUIRE_PORTABLE_CPU_TESTS"];
      if (optIn !== undefined) {env["MOM_REQUIRE_PORTABLE_CPU_TESTS"] = optIn;}
      const result = spawnSync(executable, [name, expected ? "allowed" : "denied"], {encoding: "utf8", timeout: 5000, env});
      assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, "");
    };
    const cpu = "C:\\Program Files (x86)\\Common Files\\Intel\\Shared Libraries\\bin\\intelocl64.dll";
    invoke(cpu, false); invoke(cpu, true, "1");
    for (const value of ["", "0", "true", "01", "1 "]) {invoke(cpu, false, value);}
    for (const name of ["C:\\Temp\\intelocl64.dll", "C:\\Program Files (x86)\\Common Files\\Intel\\Shared Libraries evil\\bin\\intelocl64.dll",
      "C:\\Program Files (x86)\\Common Files\\Intel\\Shared Libraries\\bin\\..\\intelocl64.dll"]) {invoke(name, false, "1");}
    invoke("intelocl64.dll", true); invoke("intelocl64.dll", true, "1");
    invoke("nvcuda.dll", true); invoke("C:\\Windows\\System32\\nvcuda.dll", true);
  });
});

test.describe("Native static ROCm SDK admission", () => {
  const names = ["amdhip64", "hiprtc", "amd_comgr", "hsakmt-staticdrm", "elf", "drm_amdgpu", "drm", "numa"];
  /** @param {import('node:test').TestContext} t */
  function sdk(t) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-rocm-inputs-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    for (const name of ["include/hip", "lib/cmake/hsa-runtime64"]) {fs.mkdirSync(path.join(root, name), {recursive: true});}
    for (const name of ["hip_runtime_api.h", "hiprtc.h"]) {fs.writeFileSync(path.join(root, "include/hip", name), "fixture\n");}
    for (const name of [...names, "hsa-runtime64"]) {fs.writeFileSync(path.join(root, "lib", `lib${name}.a`), "!<arch>\n");}
    fs.writeFileSync(path.join(root, "lib/cmake/hsa-runtime64/hsacore.so.link"), "hsa_signal_load_acquire = hsa_signal_load_scacquire;\n");
    return root;
  }
  test("AMD requires an explicit absolute SDK prefix", () => {
    for (const root of [undefined, "", "relative"]) {assert.throws(() => rocmInputs(root), /absolute/);}
  });
  test("all application dependencies are explicit self-contained archives", (t) => {
    const root = sdk(t), inputs = rocmInputs(root);
    assert.equal(inputs.include, path.join(root, "include"));
    assert.deepEqual(inputs.libraries, names.map((name) => path.join(root, "lib", `lib${name}.a`)));
  });
  test("HSA is retained exactly once with its original alias script", (t) => {
    const root = sdk(t), inputs = rocmInputs(root), hsa = path.join(root, "lib/libhsa-runtime64.a");
    assert.equal([...inputs.libraries, ...inputs.links].filter((file) => file === hsa).length, 1);
    assert.deepEqual(inputs.links, ["-Wl,--whole-archive", hsa, "-Wl,--no-whole-archive",
      "-Wl," + path.join(root, "lib/cmake/hsa-runtime64/hsacore.so.link"), "-Wl,--export-dynamic-symbol=amdgpu_device_get_fd"]);
  });
  for (const file of ["include/hip/hip_runtime_api.h", "include/hip/hiprtc.h", "lib/libhiprtc.a", "lib/libhsa-runtime64.a", "lib/libamd_comgr.a"]) {
    test(`missing ${file} fails closed`, (t) => {
      const root = sdk(t); fs.unlinkSync(path.join(root, file)); assert.throws(() => rocmInputs(root), /static ROCm/);
    });
  }
  for (const signature of ["!<thin>\n", "\x7fELF", "", "!<arch>"]) {
    test(`non-archive signature ${JSON.stringify(signature)} cannot import an external component`, (t) => {
      const root = sdk(t); fs.writeFileSync(path.join(root, "lib/libhiprtc.a"), signature);
      assert.throws(() => rocmInputs(root), /static ROCm/);
    });
  }
  for (const script of ["", 'INPUT("/tmp/untrusted.so")\n', "hsa_a = hsa_b;\nINCLUDE extra.ld\n"]) {
    test(`HSA alias script rejects unexpected input ${JSON.stringify(script)}`, (t) => {
      const root = sdk(t); fs.writeFileSync(path.join(root, "lib/cmake/hsa-runtime64/hsacore.so.link"), script);
      assert.throws(() => rocmInputs(root), /HSA alias/);
    });
  }
  test("the HIP runtime lane refuses implicit execution before any device access", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/probe-static-sycl.js")],
      {encoding: "utf8", timeout: 5000, env: {...process.env, MOM_AOT_SYCL_BACKEND: "hip", MOM_AOT_BUILD_ONLY: "0", PATH: "/unavailable"}});
    assert.equal(result.status, 1); assert.equal(result.stdout, ""); assert.match(result.stderr, /verify fresh ownership/);
    assert.doesNotMatch(result.stderr, /ENOENT|spawn/);
  });
  test("the same HIPRTC adapter binds static symbols without breaking optional Engineering loading", {
    skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
  }, (t) => {
    const root = sdk(t), header = fs.readFileSync(path.join(__dirname, "../sycl/hiprtc-api.h"), "utf8");
    const start = header.indexOf("class HiprtcApi {"), end = header.lastIndexOf("} // namespace mom");
    assert.ok(start > 0 && end > start);
    fs.writeFileSync(path.join(root, "hiprtc-contract.inc"), "namespace mom {\n" + header.slice(start, end) + "}\n");
    for (const mode of ["static", "engineering"]) {
      const executable = path.join(root, mode);
      const compiled = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        ...(mode === "static" ? ["-DMOM_STATIC_HIPRTC"] : []), "-I", root, path.join(__dirname, "native/hiprtc_static.cpp"), "-o", executable],
      {encoding: "utf8", timeout: 30000});
      assert.equal(compiled.status, 0, compiled.stderr);
      const result = spawnSync(executable, [], {encoding: "utf8", timeout: 5000});
      assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, "");
    }
  });
  test("importing or invoking the HIPRTC fixture without inputs cannot build or execute a workload", () => {
    for (const args of [["-e", 'require("./scripts/probe-native-hiprtc")'], [path.join(__dirname, "../scripts/probe-native-hiprtc.js")]]) {
      const result = spawnSync(process.execPath, args, {cwd: path.join(__dirname, ".."), encoding: "utf8", timeout: 5000,
        env: {...process.env, PATH: "/unavailable"}});
      assert.equal(result.status, args[0] === "-e" ? 0 : 1);
      assert.equal(result.stdout, ""); assert.doesNotMatch(result.stderr, /ENOENT|spawn/);
      if (args[0] !== "-e") {assert.match(result.stderr, /Usage \(Linux\)/);}
    }
  });
});

test.describe("Native process-wide compiler gate", {skip: process.platform === "win32"}, () => {
  const files = {directory: "", executable: ""};
  test.before(() => {
    files.directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-compiler-gate-"));
    files.executable = path.join(files.directory, "gate");
    const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
      path.join(__dirname, "native/compiler_gate.cpp"), path.join(__dirname, "../runtime/compiler-gate.cpp"), "-o", files.executable],
    {encoding: "utf8", timeout: 30000});
    assert.equal(result.status, 0, result.stderr);
  });
  test.after(() => {if (files.directory) {fs.rmSync(files.directory, {recursive: true, force: true});}});
  for (const mode of ["nest", "throw", "threads"]) {
    for (const count of [1, 2, 8]) {
      test(`${mode} (${count}) preserves recursion, exclusion and release`, () => {
        const result = spawnSync(files.executable, [mode, String(count)], {encoding: "utf8", timeout: 5000});
        assert.equal(result.status, 0, result.stderr); assert.equal(result.signal, null);
      });
    }
  }
  test("all four public SYCL transformation entries acquire and unwind the same native-only gate", () => {
    const patch = fs.readFileSync(path.join(__dirname, "../runtime/adaptivecpp-jit.patch"), "utf8");
    for (const entry of ["partialTransformation", "fullTransformation", "prepareIR", "translatePreparedIR"]) {
      assert.match(patch, new RegExp(`bool LLVMToBackendTranslator::${entry}[^\\n]*\\{\\n` +
        String.raw`\+#ifdef ACPP_STATIC_RUNTIME\n\+  mom_native_compiler_lock\(\);\n\+  auto compilerGuard = llvm::make_scope_exit\(mom_native_compiler_unlock\);\n\+  resetStaticCompilerOptions\(\);`));
    }
  });
  test("native translators reset foreign CLI options only after owning the common gate", () => {
    const patch = fs.readFileSync(path.join(__dirname, "../runtime/adaptivecpp-jit.patch"), "utf8");
    assert.match(patch, /static void resetStaticCompilerOptions\(\) \{\n\+ {2}llvm::cl::ResetAllOptionOccurrences\(\);/);
    assert.match(patch, /getRegisteredSubcommands\(\)[\s\S]*?entry.second->setDefault\(\)/);
    assert.equal((patch.match(/^\+ {2}resetStaticCompilerOptions\(\);$/gm) || []).length, 4);
  });
  test("COMGR target initialization and actions acquire the common gate before local exclusion", () => {
    const patch = fs.readFileSync(path.join(__dirname, "../runtime/rocm-comgr-resources.patch"), "utf8");
    assert.match(patch, /void COMGR::ensureLLVMInitialized\(\) \{\n\+#ifdef COMGR_EMBEDDED_RESOURCES\n\+ {2}mom_native_compiler_lock\(\);/);
    assert.match(patch, /amd_comgr_status_t ActionStatus;\n\+#ifdef COMGR_EMBEDDED_RESOURCES\n\+ {2}mom_native_compiler_lock\(\);[\s\S]*?Comgr actions in a mutally excusive/);
    assert.equal((patch.match(/make_scope_exit\(mom_native_compiler_unlock\)/g) || []).length, 2);
  });
  test("every static GPU/compiler proof links the same gate implementation", () => {
    for (const file of ["build-native-gpu-probe.js", "probe-static-sycl.js", "probe-native-amdgpu.js", "probe-native-ptx.js", "probe-native-hiprtc.js"]) {
      assert.match(fs.readFileSync(path.join(__dirname, "../scripts", file), "utf8"), /runtime\/compiler-gate\.cpp/);
    }
  });
});

test.describe("Native ROCm self-symbol preservation", {skip: process.platform !== "linux"}, () => {
  test("one narrow self export survives GC without admitting an application shared library", (t) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-drm-self-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const link = "-Wl,--export-dynamic-symbol=amdgpu_device_get_fd";
    for (const exported of [false, true]) {
      const executable = path.join(directory, exported ? "after" : "before");
      const result = spawnSync(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-ffunction-sections",
        path.join(__dirname, "native/drm_self_export.cpp"), "-Wl,--gc-sections", ...(exported ? [link] : []), "-ldl", "-o", executable],
      {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.stderr);
      const ran = spawnSync(executable, ["probe"], {encoding: "utf8", timeout: 5000});
      assert.equal(ran.status, exported ? 0 : 3, ran.stderr);
    }
  });
});
