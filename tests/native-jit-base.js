"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {createHash} = require("node:crypto");
const {fileHash} = require("../scripts/build-source-inputs");
const {pins} = require("../scripts/native-dpcpp-pins");
const {prepareNativeJitBase, parseCache, parseInstallComponent, baseManifest, verifyBitcode, pinnedBitcodeInput, pinnedCommitDate, sourceTreeFiles,
  jitPlanCompilerTools, buildLlvmTableGen, REQUIRED} = require("../scripts/prepare-native-jit-base");
const {jitSdkPlan} = require("../scripts/native-jit-sdk");
const {sdkComponents} = require("../scripts/prepare-native-sdk-sources");
const INSTALL_PREFIX = String.fromCharCode(36) + "{CMAKE_INSTALL_PREFIX}";

/** @param {import("node:test").TestContext} t */
function temp(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-jit-base-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return root;
}

/** @param {string} file @param {string | Buffer} value */
function write(file, value) {
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, value, {flag: "wx"});
}

/** @param {string} component @param {string} body */
function installScript(component, body) {
  return `# Install script for fixture\nif(CMAKE_INSTALL_COMPONENT STREQUAL "${component}" OR NOT CMAKE_INSTALL_COMPONENT)\n${body}\nendif()\n`;
}

/** @param {string} destination @param {string} type @param {string} files */
function installOperation(destination, type, files) {
  return `  file(INSTALL DESTINATION "${INSTALL_PREFIX}/${destination}" TYPE ${type} FILES ${files})`;
}

test.describe("Pinned SYCL JIT base producer fixtures", () => {
  test("parses unique typed CMake cache values", () => {
    const cache = parseCache("// generated\nCMAKE_HOME_DIRECTORY:INTERNAL=/src/dpcpp/llvm\nSYCL_COMPILER_VERSION:UNINITIALIZED=20260711\n");
    assert.equal(cache.get("CMAKE_HOME_DIRECTORY"), "/src/dpcpp/llvm");
    assert.equal(cache.get("SYCL_COMPILER_VERSION"), "20260711");
  });

  test("rejects malformed and duplicate cache keys", () => {
    for (const value of ["not a cache entry\n", "A:STRING=1\nA:STRING=2\n"]) {assert.throws(() => parseCache(value), /Malformed or duplicate/);}
  });

  test("builds llvm-tblgen from the selected CMake tree and records its output hash", t => {
    const root = temp(t), build = path.join(root, "build"); fs.mkdirSync(build);
    const cmake = "/usr/bin/cmake", cwd = path.join(root, "stage"), log = path.join(root, "llvm-tablegen.log");
    fs.mkdirSync(cwd);
    /** @type {{command?: string, args?: string[], options?: Record<string, unknown>} | undefined} */ let invocation;
    const result = buildLlvmTableGen(cmake, build, cwd, {PATH: "/usr/bin", LC_ALL: "C"}, log,
      (command, args, options) => {
        invocation = {command, args, options};
        const file = path.join(build, "bin/llvm-tblgen"); write(file, "source-matched TableGen\n"); fs.chmodSync(file, 0o755);
        return {status: 0};
      });
    assert.equal(invocation?.command, cmake);
    assert.deepEqual(invocation?.args, ["--build", build, "--target", "llvm-tblgen", "--parallel", "1"]);
    assert.equal(invocation?.options?.["cwd"], cwd);
    assert.deepEqual(result, {path: path.join(build, "bin/llvm-tblgen"), sha256: fileHash(path.join(build, "bin/llvm-tblgen"))});
  });

  test("does not admit a stale or missing llvm-tblgen after a failed or empty build", t => {
    const root = temp(t), build = path.join(root, "build"), cwd = path.join(root, "stage");
    fs.mkdirSync(path.join(build, "bin"), {recursive: true}); fs.mkdirSync(cwd);
    const stale = path.join(build, "bin/llvm-tblgen"); write(stale, "stale tool\n"); fs.chmodSync(stale, 0o755);
    const failing = () => buildLlvmTableGen("/usr/bin/cmake", build, cwd, {PATH: "/usr/bin"}, path.join(root, "failed.log"),
      () => ({status: 1}));
    assert.throws(failing, /Build-only cmake failed/);
    fs.unlinkSync(stale);
    assert.throws(() => buildLlvmTableGen("/usr/bin/cmake", build, cwd, {PATH: "/usr/bin"}, path.join(root, "empty.log"),
      () => ({status: 0})), /ENOENT/);
  });

  test("headers-only base reconstructs the exact configured host compiler plan", {skip: process.platform !== "linux"}, () => {
    const sdk = "/tmp/pinned-dpcpp-sdk", overlay = "/tmp/jit-placeholder-overlay", output = "/tmp/jit-configure-root",
      roots = Object.fromEntries(Object.keys(sdkComponents("compiler")).map(name => [name, "/tmp/pinned-compiler-sources/" + name])),
      cmake = "/usr/bin/cmake", ninja = "/usr/bin/ninja", dpcppC = path.join(sdk, "bin/clang"), dpcppCxx = path.join(sdk, "bin/clang++"),
      defaultTools = jitPlanCompilerTools(dpcppC, dpcppCxx, sdk);
    assert.deepEqual(defaultTools, {}, "the original embedded DPC++ compiler mode remains unchanged");
    const defaultPlan = jitSdkPlan("linux", roots, sdk, overlay, output, {cmake, ninja, ...defaultTools}, 1);
    assert.equal(defaultPlan.definitions["CMAKE_C_COMPILER"], dpcppC);
    assert.equal(defaultPlan.definitions["CMAKE_CXX_COMPILER"], dpcppCxx);
    assert.equal(defaultPlan.compilerIdentity.host.mode, "matching-clang23-bootstrap");

    const icx = "/opt/intel/oneapi/compiler/2026.2/bin/icx", icpx = "/opt/intel/oneapi/compiler/2026.2/bin/icpx",
      intelTools = jitPlanCompilerTools(icx, icpx, sdk), intelPlan = jitSdkPlan("linux", roots, sdk, overlay, output,
        {cmake, ninja, ...intelTools}, 1);
    assert.deepEqual(intelTools, {cc: icx, cxx: icpx});
    assert.equal(intelPlan.definitions["CMAKE_C_COMPILER"], icx);
    assert.equal(intelPlan.definitions["CMAKE_CXX_COMPILER"], icpx);
    assert.equal(intelPlan.compilerIdentity.host.mode, "explicit-host-override");
    assert.throws(() => jitPlanCompilerTools("relative/icx", icpx, sdk), /explicit absolute paths/);
  });

  test("extracts only the selected upstream component's file map", () => {
    const text = installScript("sycl-headers", `  file(INSTALL DESTINATION "${INSTALL_PREFIX}/include" TYPE DIRECTORY FILES\n    "/src/sycl/include/sycl"\n    "/src/ur/include/unified-runtime"\n    )`);
    assert.deepEqual(parseInstallComponent(text, "sycl-headers"), [{destination: "include", type: "DIRECTORY",
      sources: ["/src/sycl/include/sycl", "/src/ur/include/unified-runtime"]}]);
  });

  test("supports multiple generated install operations without a handwritten file list", () => {
    const text = installScript("libsycldevice", [
      installOperation("lib/dpcpp-7/sycl", "FILE", '"/build/lib/dpcpp-7/sycl/a.bc"'),
      installOperation("lib", "FILE", '"/build/lib/libsycl-devicelib-host.a"'),
    ].join("\n"));
    const operations = parseInstallComponent(text, "libsycldevice");
    assert.equal(operations.length, 2);
    assert.equal(operations[1]?.sources[0], "/build/lib/libsycl-devicelib-host.a");
  });

  test("rejects components with unreviewed CMake commands", () => {
    assert.throws(() => parseInstallComponent(installScript("sycl-headers", "  execute_process(COMMAND /bin/true)"), "sycl-headers"), /unreviewed/);
  });

  test("rejects file operations other than CMake INSTALL", () => {
    assert.throws(() => parseInstallComponent(installScript("sycl-headers", '  file(WRITE "/tmp/unsafe" "x")'), "sycl-headers"), /unreviewed/);
  });

  test("rejects a component that is absent from generated install metadata", () => {
    assert.throws(() => parseInstallComponent(installScript("sycl-headers", installOperation("include", "FILE", '"/src/a.h"')), "libsycldevice"), /missing or excessive/);
  });

  test("rejects parent traversal in install destinations", () => {
    const text = installScript("libsycldevice", installOperation("../escape", "FILE", '"/build/lib/a.bc"'));
    assert.throws(() => parseInstallComponent(text, "libsycldevice"), /relative install destination/);
  });

  test("rejects expressions and residue in generated source lists", () => {
    const text = installScript("libsycldevice", installOperation("lib", "FILE", "$<TARGET_FILE:bad>"));
    assert.throws(() => parseInstallComponent(text, "libsycldevice"), /Malformed CMake install source list/);
  });

  test("emits the resource-preparer's exact explicit manifest schema", t => {
    const root = temp(t), base = path.join(root, "base");
    write(path.join(base, "include/sycl/a.hpp"), "A\n"); write(path.join(base, "lib/device.bc"), "B\n");
    const manifest = baseManifest(base);
    assert.deepEqual(Object.keys(manifest), ["version", "files"]);
    assert.equal(manifest.version, 1);
    assert.deepEqual(manifest.files.map(row => [row.source, row.virtual]), [
      ["include/sycl/a.hpp", "include/sycl/a.hpp"], ["lib/device.bc", "lib/device.bc"],
    ]);
    assert.equal(manifest.files[0]?.sha256, fileHash(path.join(base, "include/sycl/a.hpp")));
  });

  test("base manifest rejects case-folded path aliases", {skip: process.platform === "win32"}, t => {
    const base = path.join(temp(t), "base");
    write(path.join(base, "include/CL/A.h"), "A"); write(path.join(base, "include/CL/a.h"), "a");
    assert.throws(() => baseManifest(base), /duplicate paths/);
  });

  test("base manifest rejects symlinks and special files", {skip: process.platform === "win32"}, t => {
    const root = temp(t), base = path.join(root, "base");
    write(path.join(root, "external.h"), "outside"); fs.mkdirSync(base);
    fs.symlinkSync(path.join(root, "external.h"), path.join(base, "escape.h"));
    assert.throws(() => baseManifest(base), /link, special file/);
  });

  test("base manifest rejects forbidden native payloads", t => {
    const base = path.join(temp(t), "base"); write(path.join(base, "bin/clang.exe"), "not a resource");
    assert.throws(() => baseManifest(base), /forbidden native payload/);
  });

  test("base byte limit is checked before hashing oversized inputs", t => {
    const base = path.join(temp(t), "base"), file = path.join(base, "large.bc");
    write(file, "x"); fs.truncateSync(file, (64 << 20) + 1);
    assert.throws(() => baseManifest(base), /exceeds byte bounds/);
  });

  test("pinned required resources are LLVM bitcode with fixed hashes", t => {
    const root = temp(t), file = path.join(root, "resource.bc"), bytes = Buffer.concat([Buffer.from([0x42, 0x43, 0xc0, 0xde]), Buffer.alloc(12)]);
    write(file, bytes); verifyBitcode(file, createHash("sha256").update(bytes).digest("hex"), "fixture");
    assert.equal(REQUIRED.spirv32.virtual, "lib/clang/23/lib/nvptx64-nvidia-cuda/libspirv.l32.signed_char.bc");
    assert.throws(() => verifyBitcode(file, "0".repeat(64), "fixture"), /pinned SHA-256/);
  });

  test("admits relocated CUDA bitcode by its canonical in-root path and pinned bytes", t => {
    const root = temp(t), file = path.join(root, "nvvm/libdevice/libdevice.10.bc"),
      bytes = Buffer.concat([Buffer.from([0x42, 0x43, 0xc0, 0xde]), Buffer.alloc(12)]), sha256 = createHash("sha256").update(bytes).digest("hex"),
      required = {source: "nvvm/libdevice/libdevice.10.bc", virtual: "cuda/nvvm/libdevice/libdevice.10.bc", sha256};
    write(file, bytes);
    assert.deepEqual(pinnedBitcodeInput(root, required, "CUDA libdevice"), {file, sha256});
    assert.throws(() => pinnedBitcodeInput(root, Object.assign({}, required, {package: "cuda-nvvm-12-6"}), "CUDA libdevice"), /pin is malformed/);
    assert.throws(() => pinnedBitcodeInput(root, {...required, sha256: "0".repeat(64)}, "CUDA libdevice"), /pinned SHA-256/);
    assert.throws(() => pinnedBitcodeInput(root, {...required, source: "../outside.bc"}, "CUDA libdevice"), /Invalid relative/);
    fs.writeFileSync(file, Buffer.from("changed"));
    assert.throws(() => pinnedBitcodeInput(root, required, "CUDA libdevice"), /pinned SHA-256/);
  });

  test("relocated CUDA bitcode cannot escape its selected root", {skip: process.platform === "win32"}, t => {
    const parent = temp(t), root = path.join(parent, "cuda"), external = path.join(parent, "outside.bc"), bytes = Buffer.concat([Buffer.from([0x42, 0x43, 0xc0, 0xde]), Buffer.alloc(12)]),
      sha256 = createHash("sha256").update(bytes).digest("hex"), required = {source: "nvvm/libdevice/libdevice.10.bc",
        virtual: "cuda/nvvm/libdevice/libdevice.10.bc", sha256};
    write(external, bytes); fs.mkdirSync(path.dirname(path.join(root, required.source)), {recursive: true});
    fs.symlinkSync(external, path.join(root, required.source));
    assert.throws(() => pinnedBitcodeInput(root, required, "CUDA libdevice"), /inside its admitted root/);
  });

  test("pinned Git commit date is read from the exact SHA without lazy fetch", () => {
    /** @type {{command?: string, args?: string[], options?: Record<string, unknown>} | undefined} */ let invocation;
    const date = pinnedCommitDate("/git/dpcpp.git", {MOM_AOT_BUILD_ONLY: "1"}, (command, args, options) => {
      invocation = {command, args, options}; return {status: 0, stdout: `${pins.source} 20260711\n`};
    });
    assert.equal(date, "20260711");
    assert.equal(invocation?.command, "git");
    assert.ok(invocation?.args?.includes(pins.source));
    const commandEnv = invocation?.options?.["env"];
    assert.equal(typeof commandEnv === "object" && commandEnv !== null ?
      /** @type {Record<string, unknown>} */ (commandEnv)["GIT_NO_LAZY_FETCH"] : undefined, "1");
  });

  test("rejects mismatched and unavailable pinned Git commits", () => {
    assert.throws(() => pinnedCommitDate("/git/dpcpp.git", {}, () => ({status: 0, stdout: `${"0".repeat(40)} 20260711\n`})), /pinned source commit/);
    assert.throws(() => pinnedCommitDate("/git/dpcpp.git", {}, () => ({status: 128, stderr: "missing object"})), /could not be read offline/);
  });

  test("admits the pinned OpenCL source directory only when it is regular", t => {
    const root = temp(t), headers = path.join(root, "CL"); fs.mkdirSync(headers);
    write(path.join(headers, "cl.h"), "#pragma once\n");
    const result = sourceTreeFiles(root, "CL", "OpenCL-Headers");
    assert.equal(result.files.length, 1); assert.equal(result.files[0]?.sha256, fileHash(path.join(headers, "cl.h")));
  });

  test("OpenCL source directory symlinks cannot escape the admitted root", {skip: process.platform === "win32"}, t => {
    const root = temp(t), headers = path.join(root, "CL"); fs.mkdirSync(headers); write(path.join(root, "outside.h"), "outside");
    fs.symlinkSync(path.join(root, "outside.h"), path.join(headers, "cl.h"));
    assert.throws(() => sourceTreeFiles(root, "CL", "OpenCL-Headers"), /link, special file/);
  });
});

test.describe("Opt-in actual pinned SYCL JIT base checks", () => {
  test("rejects the actual headers-only build dated from wall clock, not the pinned commit", {timeout: 600000}, t => {
    const options = actualOptions(t, "MOM_AOT_JIT_BASE_STALE_"); if (!options) {return;}
    assert.throws(() => prepareNativeJitBase(options), /SYCL_COMPILER_VERSION must be 20260711/);
    assert.equal(fs.existsSync(options.output), false);
  });

  test("stages an actual pinned headers-only resource base without publication claims", {timeout: 600000}, t => {
    const options = actualOptions(t, "MOM_AOT_JIT_BASE_PROOF_"); if (!options) {return;}
    const result = prepareNativeJitBase(options);
    /** @type {{files: Array<{virtual: string, sha256: string}>}} */ const manifest = JSON.parse(fs.readFileSync(result.baseManifest, "utf8"));
    assert.equal(result.buildOnly, true); assert.equal(result.compiled, false); assert.equal(result.executed, false);
    assert.equal(result.publicationQualified, false); assert.ok(manifest.files.length > 700);
    for (const required of Object.values(REQUIRED)) {
      assert.ok(manifest.files.some(row => row.virtual === required.virtual && row.sha256 === required.sha256));
    }
  });
});

/** @param {import("node:test").TestContext} t @param {string} prefix */
function actualOptions(t, prefix) {
  const names = {sdkRoot: "SDK", cudaRoot: "CUDA", buildRoot: "BUILD", sourceReceipt: "SOURCES", dpcppGitRoot: "GIT", output: "OUTPUT"};
  const options = Object.fromEntries(Object.entries(names).map(([key, name]) => [key, process.env[prefix + name]]));
  if (Object.values(options).some(value => !value) || process.env["MOM_AOT_BUILD_ONLY"] !== "1") {
    t.skip("actual resource-base proof inputs are not selected"); return null;
  }
  return /** @type {Parameters<typeof prepareNativeJitBase>[0]} */ (options);
}
