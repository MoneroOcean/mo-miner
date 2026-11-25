"use strict";
const fs = require("node:fs"), path = require("node:path");
const test = require("node:test"), assert = require("node:assert/strict");
const {fileHash, sourceInputs} = require("../scripts/build-source-inputs");
const {revision: comgrRevision} = require("../scripts/build-native-windows-comgr-sdk");
const {dependencyHashes} = require("../scripts/build-native-windows-controller");
const {sourceFiles, recipeFiles, windowsHiprtcPlan, verifyComgrHeader, verifyIntelCompilerIdentity, verifyLlvmToolIdentity, auditHiprtcSymbolClosure} = require("../scripts/build-native-windows-hiprtc-sdk");
const {tempDir} = require("./common/temp-dir");

const resourceRevision = comgrRevision;
const clrRevision = "5b591b2ff64aac4c1c08be879b0d5f89fd713d50";
const requiredBitcode = ["asanrtl.bc", "hip.bc", "ockl.bc", "ocml.bc", "opencl.bc"];
const linuxPlanSkip = process.platform !== "linux" ? "requires Linux-hosted POSIX cross-build paths" : false;

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.realpathSync(tempDir(t, "mom-win-hiprtc-sdk-"));
  const mkdir = (/** @type {string} */ directory) => fs.mkdirSync(directory, {recursive: true});
  const write = (/** @type {string} */ file, /** @type {string} */ text = "fixture\n") => {
    mkdir(path.dirname(file)); fs.writeFileSync(file, text); return file;
  };
  const source = path.join(root, "clr"), hipInclude = path.join(root, "hip-sdk", "include"), tools = path.join(root, "tools"),
    descriptorFile = path.join(root, "windows-sdk.json"), windowsSource = path.join(root, "windows-source"),
    include = path.join(root, "clang", "include"), overlay = path.join(root, "windows-vfs.json"), llvmAr = path.join(tools, "llvm-ar"),
    clangCl = path.join(tools, "icx-cl"), llvmNm = path.join(tools, "llvm-nm"), sdkHeader = write(path.join(windowsSource, "msvc/include/vector"));
  for (const name of sourceFiles) {write(path.join(source, name), "// pinned source fixture: " + name + "\n");}
  write(path.join(hipInclude, "hip/hip_common.h")); write(path.join(hipInclude, "amd_comgr/amd_comgr.h"), "/* official header fixture */\n");
  write(clangCl, "intel compiler fixture\n"); write(llvmAr, "llvm librarian fixture\n"); write(llvmNm, "llvm nm fixture\n"); write(path.join(include, "stddef.h"));
  const sdk = (/** @type {string} */ directory) => "/mom-sdk/" + directory;
  fs.writeFileSync(overlay, JSON.stringify({version: 0, "case-sensitive": false, "use-external-names": true,
    roots: [{type: "file", name: sdk("msvc/include/vector"), "external-contents": sdkHeader}]}));
  const lib = (/** @type {string} */ name) => path.join(root, "windows-libs", name),
    config = {version: 1, tools: {clangCl, lldLink: lib("lld-link"), llvmNm, clangResourceInclude: include},
      windowsSdk: {overlay, sourceRoot: windowsSource,
        includeRoots: [sdk("msvc/include"), sdk("sdk/include/ucrt"), sdk("sdk/include/shared"), sdk("sdk/include/um"), sdk("sdk/include/winrt")],
        libraryRoots: [lib("msvc/lib"), lib("sdk/lib/ucrt"), lib("sdk/lib/um")],
        msvcLibraries: [lib("msvc/lib/libcmt.lib"), lib("msvc/lib/libcpmt.lib"), lib("msvc/lib/oldnames.lib")],
        ucrtLibraries: [lib("sdk/lib/ucrt/libucrt.lib")], importLibraries: [lib("sdk/lib/um/kernel32.lib")]}};
  fs.writeFileSync(descriptorFile, JSON.stringify(config, null, 2) + "\n");

  const resourceRoot = path.join(root, "resources"), resourceSource = path.join(resourceRoot, "source"),
    deviceInstall = path.join(resourceRoot, "install"), deviceReceipt = path.join(resourceRoot, "result.json"),
    bitcodeRoot = path.join(deviceInstall, "amdgcn/bitcode"), deviceConfig = path.join(deviceInstall, "lib/cmake/AMDDeviceLibs/AMDDeviceLibsConfig.cmake"),
    hipHeader = write(path.join(resourceRoot, "hipRTC"), "upstream generated HIPRTC header\n"),
    builtinObject = write(path.join(resourceRoot, "hipRTC_header.obj"), "generated COFF object fixture\n"),
    selectedBitcode = [...requiredBitcode, ...Array.from({length: 66}, (_, index) => "fixture-" + String(index).padStart(2, "0") + ".bc")];
  write(path.join(resourceSource, "device/pinned.txt")); write(path.join(resourceSource, "hiprtc/pinned.txt"));
  const deviceFiles = /** @type {Record<string, string>} */ ({});
  for (const name of selectedBitcode) {
    const file = write(path.join(bitcodeRoot, name), "bitcode " + name + "\n"); deviceFiles[file] = fileHash(file);
  }
  const registry = selectedBitcode.map(name => `set_property(TARGET "AMDDeviceLibs::${name.slice(0, -3)}" PROPERTY IMPORTED_LOCATION "\${AMD_DEVICE_LIBS_PREFIX}/amdgcn/bitcode/${name}")`).join("\n") + "\n";
  write(deviceConfig, registry); deviceFiles[deviceConfig] = fileHash(deviceConfig);
  const sourceRootState = sourceInputs(resourceSource, ["device", "hiprtc"], []),
    resourceInputs = {[path.join(hipInclude, "hip/hip_common.h")]: fileHash(path.join(hipInclude, "hip/hip_common.h"))};
  const deviceReceiptValue = {version: 1, target: "win32", revision: resourceRevision, clrRevision,
    buildOnly: true, publicationQualified: false, executed: false, source: sourceRootState.files,
    recipe: {graph: {"resource-recipe": "a".repeat(64)}, files: {}}, inputs: resourceInputs, dependencies: resourceInputs,
    device: deviceFiles, header: {path: hipHeader, sha256: fileHash(hipHeader)}, object: {path: builtinObject, sha256: fileHash(builtinObject)}};
  fs.writeFileSync(deviceReceipt, JSON.stringify(deviceReceiptValue, null, 2) + "\n");

  const comgrRoot = path.join(root, "comgr"), comgrBuild = path.join(comgrRoot, "build"), comgrHeader = write(path.join(comgrBuild, "include/amd_comgr.h"), "/* generated Windows COMGR header */\n"),
    comgrPlan = {source: path.join(root, "comgr-source"), descriptorFile, targetLlvm: path.join(root, "llvm-sdk"), nativeLlvm: "/opt/llvm21-ubuntu24",
      hostCompiler: "/opt/intel/oneapi/compiler/2026.2/bin/icx", deviceReceipt, deviceInstall, output: comgrRoot, jobs: 2,
      tools: {cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja"}},
    planFile = path.join(comgrRoot, "plan.json"), comgrResult = path.join(comgrRoot, "result.json");
  write(planFile, "{\"version\":1}\n");
  const comgrReceiptValue = {version: 1, role: "windows-comgr-static-host", stage: "complete", revision: comgrRevision,
    buildOnly: true, compiled: true, executed: false, publicationQualified: false, plan: comgrPlan, planSha256: fileHash(planFile),
    fingerprints: {[descriptorFile]: fileHash(descriptorFile)}, header: comgrHeader, headerSha256: fileHash(comgrHeader)};
  fs.writeFileSync(comgrResult, JSON.stringify(comgrReceiptValue, null, 2) + "\n");
  const input = {source, descriptorFile, hipInclude, comgrResult, deviceReceipt, deviceInstall, llvmAr, output: path.join(root, "out")};
  return {root, input, config, sdkHeader, source, hipInclude, descriptorFile, deviceReceipt, deviceInstall, builtinObject,
    hipHeader, comgrRoot, comgrHeader, comgrResult, resourceInputs, write};
}

/** @param {string} file @param {(value: any) => void} change */
function editJson(file, change) {
  const value = JSON.parse(fs.readFileSync(file, "utf8")); change(value); fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n");
}

test.describe("Windows HIPRTC source SDK admission", () => {
  test("fingerprints the producer recipe for receipt provenance", () => {
    const root = path.resolve(__dirname, ".."), recipe = sourceInputs(root, [], [...recipeFiles]);
    assert.deepEqual(Object.keys(recipe.graph).sort(), [...recipeFiles].sort());
    assert.equal(recipe.graph["scripts/build-native-windows-hiprtc-sdk.js"], fileHash(path.join(root, "scripts/build-native-windows-hiprtc-sdk.js")));
  });

  test("accepts the thin Windows LLVM descriptor and plans ten host TUs plus the generated builtins object", {skip: linuxPlanSkip}, t => {
    const f = fixture(t), plan = windowsHiprtcPlan(f.input);
    assert.deepEqual(Object.keys(f.config).sort(), ["tools", "version", "windowsSdk"]);
    assert.deepEqual(Object.keys(f.config.tools).sort(), ["clangCl", "clangResourceInclude", "lldLink", "llvmNm"]);
    assert.deepEqual(Object.keys(f.config.windowsSdk).sort(), ["importLibraries", "includeRoots", "libraryRoots", "msvcLibraries", "overlay", "sourceRoot", "ucrtLibraries"]);
    assert.equal(plan.sourceFiles.length, 10);
    assert.equal(plan.objects.length, 11);
    assert.equal(path.basename(plan.objects[10] || ""), "hipRTC_header.obj");
    assert.equal(plan.deviceObject.path, f.builtinObject);
    assert.equal(plan.deviceHeader.path, f.hipHeader);
    assert.equal(plan.comgr.header, f.comgrHeader);
    assert.ok(plan.includeRoots.indexOf(path.join(f.input.output, "include")) < plan.includeRoots.indexOf(f.hipInclude));
  });

  test("rejects missing inputs from the thin Windows LLVM descriptor", {skip: linuxPlanSkip}, t => {
    const f = fixture(t), descriptor = JSON.parse(fs.readFileSync(f.descriptorFile, "utf8"));
    delete descriptor.windowsSdk.importLibraries;
    fs.writeFileSync(f.descriptorFile, JSON.stringify(descriptor));
    assert.throws(() => windowsHiprtcPlan(f.input), /import libraries/);
  });

  test("uses static CRT, the selected target, and a direct COMGR declaration override", {skip: linuxPlanSkip}, t => {
    const plan = windowsHiprtcPlan(fixture(t).input);
    for (const flag of ["--target=x86_64-pc-windows-msvc", "/MT", "/EHsc", "/GR", "/DAMD_COMGR_IMPORT_DECORATOR="]) {
      assert.ok(plan.flags.includes(flag), flag);
    }
    assert.equal(plan.flags.some(flag => /^\/(?:MD|MDd)$/.test(flag)), false);
  });

  test("writes system-header dependencies in the existing NMake format", {skip: linuxPlanSkip}, t => {
    assert.deepEqual(windowsHiprtcPlan(fixture(t).input).nmakeDependencyArgs, ["-Xclang", "-MV"]);
  });

  test("resolves Intel compiler VFS dependency names through the selected Windows SDK overlay", {skip: linuxPlanSkip}, t => {
    const f = fixture(t), output = path.join(f.root, "out"), depfile = path.join(f.root, "hiprtc.d");
    fs.mkdirSync(output);
    const virtual = "/mom-sdk/msvc/include/vector";
    fs.writeFileSync(depfile, path.join(output, "hiprtc.obj") + ": \\\n  " + virtual + "\n");
    const dependencies = dependencyHashes([depfile], f.config.windowsSdk.overlay, output, Date.now() + 1000);
    assert.equal(dependencies[fs.realpathSync(f.sdkHeader)], fileHash(f.sdkHeader));
    assert.equal(Object.hasOwn(dependencies, virtual), false, "receipts use the resolved SDK header path, not the compiler's virtual name");
  });

  test("requires every root explicitly and rejects noncanonical path spellings", {skip: linuxPlanSkip}, t => {
    const f = fixture(t), incomplete = /** @type {Record<string, string | undefined>} */ ({...f.input}); delete incomplete["deviceReceipt"];
    assert.throws(() => windowsHiprtcPlan(/** @type {any} */ (incomplete)), /explicit canonical absolute source/);
    assert.throws(() => windowsHiprtcPlan({...f.input, source: f.source + "/../clr"}), /explicit canonical absolute/);
  });

  test("rejects an existing output instead of overwriting producer evidence", {skip: linuxPlanSkip}, t => {
    const f = fixture(t); fs.mkdirSync(f.input.output);
    assert.throws(() => windowsHiprtcPlan(f.input), /new canonical/);
  });

  test("rejects outputs inside source or selected SDK roots", {skip: linuxPlanSkip}, t => {
    const f = fixture(t);
    assert.throws(() => windowsHiprtcPlan({...f.input, output: path.join(f.source, "out")}), /overlaps/);
    assert.throws(() => windowsHiprtcPlan({...f.input, output: path.join(f.hipInclude, "out")}), /overlaps/);
  });

  test("rejects symlinked source and HIP include roots", {skip: linuxPlanSkip}, t => {
    const f = fixture(t), alias = path.join(f.root, "clr-alias"); fs.symlinkSync(f.source, alias);
    assert.throws(() => windowsHiprtcPlan({...f.input, source: alias}), /canonical directory/);
    const hipAlias = path.join(f.root, "hip-alias"); fs.symlinkSync(f.hipInclude, hipAlias);
    assert.throws(() => windowsHiprtcPlan({...f.input, hipInclude: hipAlias}), /canonical directory/);
  });

  test("resolves selected compiler, librarian, and symbol-auditor aliases once", {skip: linuxPlanSkip}, t => {
    const f = fixture(t), compilerAlias = path.join(f.root, "icx-cl"), nmAlias = path.join(f.root, "llvm-nm"), arAlias = path.join(f.root, "llvm-ar");
    fs.symlinkSync(f.config.tools.clangCl, compilerAlias); fs.symlinkSync(f.config.tools.llvmNm, nmAlias); fs.symlinkSync(f.input.llvmAr, arAlias);
    const config = JSON.parse(fs.readFileSync(f.descriptorFile, "utf8")); config.tools.clangCl = compilerAlias; config.tools.llvmNm = nmAlias;
    fs.writeFileSync(f.descriptorFile, JSON.stringify(config, null, 2) + "\n");
    editJson(f.comgrResult, value => {value.fingerprints[f.descriptorFile] = fileHash(f.descriptorFile);});
    const plan = windowsHiprtcPlan({...f.input, llvmAr: arAlias});
    assert.equal(plan.compiler, fs.realpathSync(compilerAlias)); assert.equal(plan.llvmNm, fs.realpathSync(nmAlias)); assert.equal(plan.llvmAr, fs.realpathSync(arAlias));
    assert.deepEqual(plan.toolPaths, {compiler: compilerAlias, llvmNm: nmAlias, llvmAr: arAlias});
  });

  test("admits only the completed pinned COMGR result and generated header path", t => {
    const f = fixture(t), selected = verifyComgrHeader(f.comgrResult, f.descriptorFile);
    assert.equal(selected.header, f.comgrHeader);
    assert.equal(selected.headerSha256, fileHash(f.comgrHeader));
    editJson(f.comgrResult, value => {value.revision = "0".repeat(40);});
    assert.throws(() => verifyComgrHeader(f.comgrResult, f.descriptorFile), /completed pinned/);
  });

  test("rejects a configured, uncompiled or publication-qualified COMGR header receipt", t => {
    const f = fixture(t);
    const changes = /** @type {Array<(value: any) => void>} */ ([value => {value.stage = "configured";}, value => {value.compiled = false;},
      value => {value.publicationQualified = true;}]);
    for (const change of changes) {
      editJson(f.comgrResult, change);
      assert.throws(() => verifyComgrHeader(f.comgrResult, f.descriptorFile), /completed pinned/);
      const baseline = fixture(t); // Each mutation gets a fresh, independently valid receipt.
      f.comgrResult = baseline.comgrResult; f.descriptorFile = baseline.descriptorFile;
    }
  });

  test("binds COMGR to the selected Windows descriptor and its plan receipt", t => {
    const f = fixture(t); editJson(f.comgrResult, value => {value.plan.descriptorFile = path.join(f.root, "other.json");});
    assert.throws(() => verifyComgrHeader(f.comgrResult, f.descriptorFile), /completed pinned/);
    const g = fixture(t); editJson(g.comgrResult, value => {value.fingerprints[g.descriptorFile] = "0".repeat(64);});
    assert.throws(() => verifyComgrHeader(g.comgrResult, g.descriptorFile), /completed pinned/);
  });

  test("rejects a changed or symlinked generated COMGR header", t => {
    const f = fixture(t); fs.appendFileSync(f.comgrHeader, "changed\n");
    assert.throws(() => verifyComgrHeader(f.comgrResult, f.descriptorFile), /generated header differs/);
    const g = fixture(t), original = g.comgrHeader, target = g.write(path.join(g.root, "other-comgr.h"));
    fs.unlinkSync(original); fs.symlinkSync(target, original);
    assert.throws(() => verifyComgrHeader(g.comgrResult, g.descriptorFile), /canonical regular file/);
  });

  test("requires the HIPRTC builtins receipt to use the same official HIP header root", {skip: linuxPlanSkip}, t => {
    const f = fixture(t), other = path.join(f.root, "other-hip"); f.write(path.join(other, "hip/hip_common.h"));
    assert.throws(() => windowsHiprtcPlan({...f.input, hipInclude: other}), /selected official HIP headers/);
  });

  test("rechecks the original builtins header and object hashes on admission", {skip: linuxPlanSkip}, t => {
    const f = fixture(t); fs.appendFileSync(f.builtinObject, "changed\n");
    assert.throws(() => windowsHiprtcPlan(f.input), /resource output differs/);
    const g = fixture(t); fs.appendFileSync(g.hipHeader, "changed\n");
    assert.throws(() => windowsHiprtcPlan(g.input), /resource output differs/);
  });

  test("rejects an invalid device install root", {skip: linuxPlanSkip}, t => {
    const f = fixture(t), wrong = path.join(f.root, "not-install"); fs.mkdirSync(wrong);
    assert.throws(() => windowsHiprtcPlan({...f.input, deviceInstall: wrong}), /Windows HIP resource receipt and install directory/);
  });

  test("accepts only Intel 2026.2 and rejects a compatible but unselected compiler", () => {
    assert.match(verifyIntelCompilerIdentity("Intel(R) oneAPI DPC++/C++ Compiler for applications running on Intel(R) 64, Version 2026.2.0 Build 20261001").version, /2026\.2/);
    for (const version of ["Ubuntu clang version 21.1.8", "clang version 23", "Intel compiler 2025.1",
      "Unrelated text ending in Intel, Version 2026.2.0", "Intel(R) oneAPI DPC++/C++ Compiler, Version 2026.20.0"]) {
      assert.throws(() => verifyIntelCompilerIdentity(version), /Intel 2026\.2/);
    }
  });

  test("pins the archive and symbol-audit tools to LLVM 21.1.8", () => {
    assert.equal(verifyLlvmToolIdentity("LLVM version 21.1.8", "librarian").version, "21.1.8");
    assert.throws(() => verifyLlvmToolIdentity("LLVM version 22.0.0", "symbol auditor"), /LLVM 21\.1\.8/);
  });

  test("U-minus-defined audit allows system/driver externals but requires direct COMGR and generated builtins", () => {
    const undefinedText = ["hiprtc.obj:", "         U amd_comgr_create_data", "         U hipGetDevicePropertiesR0600", "         U __imp_CreateFileA",
      "         U public: bool __cdecl hip::LinkProgram::run(void)"].join("\n");
    const definedText = ["hiprtc.obj:", "00000000 T public: bool __cdecl hip::LinkProgram::run(void)",
      "00000000 D __hipRTC_header", "00000010 D __hipRTC_header_size"].join("\n");
    const audit = auditHiprtcSymbolClosure(undefinedText, definedText);
    assert.equal(audit.undefinedCount, 4);
    assert.deepEqual(audit.comgrReferences, ["amd_comgr_create_data"]);
    assert.deepEqual(audit.driverReferences, ["hipGetDevicePropertiesR0600"]);
    assert.ok(audit.unresolved.includes("__imp_CreateFileA"));
  });

  test("deduplicates per-member references before closure comparison", () => {
    const audit = auditHiprtcSymbolClosure(["         U amd_comgr_create_data", "         U amd_comgr_create_data", "         U hipGetDevicePropertiesR0600"].join("\n"),
      ["00000000 D __hipRTC_header", "00000010 D __hipRTC_header_size"].join("\n"));
    assert.equal(audit.undefinedCount, 2);
    assert.equal(audit.unresolved.filter(name => name === "amd_comgr_create_data").length, 1);
  });

  test("rejects COMGR import-address-table references rather than allowing a DLL", () => {
    const undefinedText = ["         U amd_comgr_create_data", "         U __imp_amd_comgr_create_data"].join("\n"),
      definedText = "00000000 D __hipRTC_header\n00000010 D __hipRTC_header_size";
    assert.throws(() => auditHiprtcSymbolClosure(undefinedText, definedText), /COMGR DLL imports/);
  });

  test("rejects internal HIPRTC/ROCclr unresolved symbols", () => {
    const undefinedText = "         U amd_comgr_create_data\n         U public: bool __cdecl amd::Os::commitMemory(void)";
    const definedText = "00000000 D __hipRTC_header\n00000010 D __hipRTC_header_size";
    assert.throws(() => auditHiprtcSymbolClosure(undefinedText, definedText), /missing HIPRTC\/ROCclr/);
  });

  test("rejects missing generated HIPRTC builtins and missing direct COMGR use", () => {
    const noBuiltins = "00000000 D __hipRTC_header";
    assert.throws(() => auditHiprtcSymbolClosure("         U amd_comgr_create_data", noBuiltins), /missing HIPRTC\/ROCclr/);
    assert.throws(() => auditHiprtcSymbolClosure("         U __imp_CreateFileA", "00000000 D __hipRTC_header\n00000010 D __hipRTC_header_size"),
      /no direct COMGR references/);
  });
});
