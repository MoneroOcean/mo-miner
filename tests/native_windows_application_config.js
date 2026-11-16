"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const os = require("node:os");
const path = require("node:path");
const {composeWindowsApplicationConfig, composeWindowsApplicationDescriptor} = require("../scripts/native-windows-application-config");

const runAttempt = "42-1", sha = "a".repeat(64), fixtureRoot = path.join(os.tmpdir(), "native-windows-controller-" + runAttempt),
  at = (/** @type {string} */ root, /** @type {string} */ file) => path.join(root, file);

/** Reader-result fixtures; malformed cases intentionally mutate this shape. @returns {any} */
function inputs() {
  const ioRoot = at(fixtureRoot, "io"), hermesRoot = at(fixtureRoot, "hermes"),
    sdkRoot = at(fixtureRoot, "windows-host-sdk"), intelRoot = at(hermesRoot, "intel-runtime/lib"),
    intelLibraries = Object.fromEntries(["libircmt.lib", "libirc.lib", "libmmt.lib", "svml_dispmt.lib", "libdecimal.lib"]
      .map(name => [name, at(intelRoot, name)])), hermesBuild = at(hermesRoot, "hermes/build");
  return {
    runAttempt,
    llvmDescriptor: {version: 1, tools: {clangCl: "/tools/icx-cl.exe", lldLink: "/tools/lld-link.exe", llvmNm: "/tools/llvm-nm.exe",
      clangResourceInclude: "/tools/resource/include"}, windowsSdk: {sourceRoot: sdkRoot, overlay: at(sdkRoot, "overlay.json"),
      includeRoots: [at(sdkRoot, "msvc/include"), at(sdkRoot, "sdk/include")],
      libraryRoots: [at(sdkRoot, "sdk/lib/ucrt"), at(sdkRoot, "sdk/lib/um"), at(sdkRoot, "msvc/lib"), intelRoot],
      msvcLibraries: [at(sdkRoot, "msvc/lib/libcmt.lib"), ...Object.values(intelLibraries)],
      ucrtLibraries: [at(sdkRoot, "sdk/lib/ucrt/libvcruntime.lib")], importLibraries: [at(sdkRoot, "sdk/lib/um/kernel32.lib")]}},
    io: {kind: "io", runAttempt, handoff: at(ioRoot, "handoff.json"), handoffSha256: sha, files: {},
      libraries: {sourceInclude: at(ioRoot, "openssl-sdk/include"), ssl: at(ioRoot, "openssl-sdk/lib/libssl.lib"),
        crypto: at(ioRoot, "openssl-sdk/lib/libcrypto.lib"), libuv: at(ioRoot, "io-sdk/lib/uv_a.lib"),
        zlib: at(ioRoot, "io-sdk/lib/zlibstatic.lib"), libuvInclude: at(ioRoot, "io-sdk/include"),
        zlibInclude: at(ioRoot, "io-sdk/include"), zlibGeneratedInclude: at(ioRoot, "io-sdk/include")}},
    hermes: {kind: "hermes", runAttempt, handoff: at(hermesRoot, "handoff.json"), handoffSha256: sha, files: {},
      hermes: {source: at(hermesRoot, "hermes/source"), build: hermesBuild},
      intelRuntime: {libRoot: intelRoot, libraries: intelLibraries, licenses: {license: at(hermesRoot, "intel-runtime/licenses/LICENSE.rtf")}}},
    hermesAot: {artifacts: {compiler: "/tmp/linux-hermes-sdk/build/bin/shermes"},
      files: {"/tmp/linux-hermes-sdk/build/bin/shermes": sha}},
    tools: {shermaot: "/tmp/linux-hermes-sdk/build/bin/shermes", llvmReadobj: "/tools/llvm-readobj.exe"},
    allowedImports: ["kernel32.dll", "ws2_32.dll"],
  };
}

function application() {
  return {core: "/app/core.lib", gpuReceipts: ["/app/gpu-result.json"], defaultCompiler: "acpp-hip",
    tools: {llvmAr: "/tools/llvm-ar.exe", llvmObjcopy: "/tools/llvm-objcopy.exe"},
    sdk: {libraries: ["/app/gpu.lib"], wholeArchives: ["/app/runtime.lib"], objects: [], includeRoots: ["/app/include"],
      nvmlHeader: "/app/include/nvml.h", hipDriverLibrary: "/app/amdhip64.lib"}};
}

test.describe("Windows full-controller descriptor composer", () => {
  test("local reader results compose the same descriptor without fabricated CI identity", () => {
    const ci = inputs(), local = structuredClone(ci);
    delete local.runAttempt;
    local.io = {libraries: local.io.libraries};
    local.hermes = {hermes: local.hermes.hermes, intelRuntime: local.hermes.intelRuntime};
    assert.deepEqual(composeWindowsApplicationDescriptor(local), composeWindowsApplicationConfig(ci));
    assert.throws(() => composeWindowsApplicationConfig(local), /input is incomplete/);
    assert.throws(() => composeWindowsApplicationDescriptor(ci), /unexpected shape/);
  });

  test("CI transport identity remains explicit and cannot be inherited or omitted", () => {
    const selected = inputs();
    delete selected.runAttempt;
    Object.setPrototypeOf(selected, {runAttempt});
    assert.throws(() => composeWindowsApplicationConfig(selected), /input is incomplete/);
    selected.runAttempt = runAttempt;
    delete selected.io.handoffSha256;
    assert.throws(() => composeWindowsApplicationConfig(selected), /same-run admitted io handoff/);
  });

  test("local composition retains tool, Intel runtime, import and application guards", () => {
    for (const mutate of [
      (/** @type {any} */ row) => {delete row.allowedImports;},
      (/** @type {any} */ row) => {row.allowedImports = ["unreviewed-runtime.dll"];},
      (/** @type {any} */ row) => {row.llvmDescriptor.windowsSdk.libraryRoots.pop();},
      (/** @type {any} */ row) => {row.hermesAot.files = {};},
      (/** @type {any} */ row) => {row.application = {core: "/app/core.lib"};},
    ]) {
      const selected = inputs(); delete selected.runAttempt; mutate(selected);
      assert.throws(() => composeWindowsApplicationDescriptor(selected));
    }
  });

  test("preserves the admitted LLVM arrays and composes same-run IO/Hermes values", () => {
    const selected = inputs(), original = structuredClone(selected.llvmDescriptor), config = composeWindowsApplicationConfig(selected);
    assert.deepEqual(config.tools, {...original.tools, ...selected.tools});
    assert.deepEqual(config.hermes, {...selected.hermes.hermes, shermaot: selected.tools.shermaot});
    assert.deepEqual(config.libraries, selected.io.libraries);
    assert.deepEqual(config.windowsSdk, {...original.windowsSdk, allowedImports: selected.allowedImports});
    assert.equal(Object.hasOwn(config, "application"), false);
  });

  test("passes an optional application descriptor only through the full validator", () => {
    const selected = inputs(), app = application();
    selected.application = app;
    selected.tools.llvmObjdump = "/tools/llvm-objdump.exe";
    const config = composeWindowsApplicationConfig(selected);
    assert.deepEqual(config.application, app);
    assert.equal(config.tools.llvmObjdump, selected.tools.llvmObjdump);
  });

  test("rejects a support reader result for the wrong role", () => {
    const selected = inputs(); selected.io.kind = "hermes";
    assert.throws(() => composeWindowsApplicationConfig(selected), /same-run admitted io handoff/);
  });

  test("rejects IO and Hermes reader results from different attempts", () => {
    const selected = inputs(); selected.hermes.runAttempt = "43-1";
    assert.throws(() => composeWindowsApplicationConfig(selected), /same-run admitted hermes handoff/);
  });

  test("requires the base LLVM descriptor to retain every admitted Intel runtime path", () => {
    const missingRoot = inputs(); missingRoot.llvmDescriptor.windowsSdk.libraryRoots.pop();
    assert.throws(() => composeWindowsApplicationConfig(missingRoot), /does not retain the admitted Hermes Intel runtime/);
    const missingLibrary = inputs(); missingLibrary.llvmDescriptor.windowsSdk.msvcLibraries.pop();
    assert.throws(() => composeWindowsApplicationConfig(missingLibrary), /does not retain the admitted Hermes Intel runtime/);
  });

  test("binds shermaot to the separately admitted Linux Hermes compiler", () => {
    const wrongTool = inputs(); wrongTool.tools.shermaot = "/tools/other-shermaot";
    assert.throws(() => composeWindowsApplicationConfig(wrongTool), /not bound to the admitted Linux Hermes/);
    const unbound = inputs(); unbound.hermesAot.files = {};
    assert.throws(() => composeWindowsApplicationConfig(unbound), /not bound to the admitted Linux Hermes/);
  });

  test("requires explicit allowed imports and delegates the OS-DLL policy to the controller validator", () => {
    const missing = inputs(); delete missing.allowedImports;
    assert.throws(() => composeWindowsApplicationConfig(missing), /input is incomplete/);
    const invalid = inputs(); invalid.allowedImports = ["kernel32.dll", "unreviewed-runtime.dll"];
    assert.throws(() => composeWindowsApplicationConfig(invalid), /ordinary system DLLs/);
  });

  test("does not create a partial or dummy application descriptor by default", () => {
    const config = composeWindowsApplicationConfig(inputs());
    assert.equal(Object.hasOwn(config, "application"), false);
    const invalid = inputs(); invalid.application = {core: "/app/core.lib"};
    assert.throws(() => composeWindowsApplicationConfig(invalid), /Invalid Windows application Core/);
  });
});
