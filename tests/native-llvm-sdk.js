"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {archiveSha256, toolsComponents, metadataInstallTargets, installTargets, linuxLlvmSdkPlan, verifySourceVersion, llvmSourceSnapshot,
  verifyArchiveCatalog, llvmStaticTargets, linuxDistributionSelection, verifyDistributionGraph, cmakeCache,
  auditLinuxLlvmPrefix, phaseBuildArgs, findClangLinkCommand, verifyAmdgpulibCallsCompileFlag} = require("../scripts/build-native-llvm-sdk");
const {sourceDirectories, archiveName, llvmAmdgpulibCallsCompileOptionsShim, verifyCmakeVersion} = require("../scripts/native-llvm-sdk");

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-linux-llvm21-sdk-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const source = path.join(root, "source"), build = path.join(root, "build"), output = path.join(root, "output with spaces"),
    archive = path.join(root, archiveName), tools = {cmake: "/tools/cmake", ninja: "/tools/ninja", cc: "/intel/icx", cxx: "/intel/icpx"};
  for (const name of sourceDirectories) {fs.mkdirSync(path.join(source, name), {recursive: true});}
  fs.mkdirSync(build, {recursive: true});
  return {root, source, build, output, archive, tools};
}

/** @param {string} file @param {unknown} value */
function writeJson(file, value) {fs.writeFileSync(file, JSON.stringify(value) + "\n");}

/** @param {ReturnType<typeof fixture>} f */
function targetFixture(f) {
  const rows = [
    ["LLVMCore", "llvm", "lib/IR", "STATIC_LIBRARY", "lib/libLLVMCore.a"],
    ["LLVMSupport", "llvm", "lib/Support", "STATIC_LIBRARY", "lib/libLLVMSupport.a"],
    ["LLVMDemangle", "llvm", "lib/Demangle", "STATIC_LIBRARY", "lib/libLLVMDemangle.a"],
    ["LLVMBinaryFormat", "llvm", "lib/BinaryFormat", "STATIC_LIBRARY", "lib/libLLVMBinaryFormat.a"],
    ["LLVMRemarks", "llvm", "lib/Remarks", "STATIC_LIBRARY", "lib/libLLVMRemarks.a"],
    ["LLVMTargetParser", "llvm", "lib/TargetParser", "STATIC_LIBRARY", "lib/libLLVMTargetParser.a"],
    ["LLVMBitstreamReader", "llvm", "lib/Bitstream/Reader", "STATIC_LIBRARY", "lib/libLLVMBitstreamReader.a"],
    ["LLVMSupportParallel", "llvm", "lib/Support/Parallel", "STATIC_LIBRARY", "lib/libLLVMSupportParallel.a"],
    ["LLVMNotSupport", "llvm", "lib/Supportive", "STATIC_LIBRARY", "lib/libLLVMNotSupport.a"],
    ["LLVMAnalysis", "llvm", "lib/Analysis", "STATIC_LIBRARY", "lib/libLLVMAnalysis.a"],
    ["LLVMBitReader", "llvm", "lib/Bitcode/Reader", "STATIC_LIBRARY", "lib/libLLVMBitReader.a"],
    ["LLVMObject", "llvm", "lib/Object", "STATIC_LIBRARY", "lib/libLLVMObject.a"],
    ["LLVMTransformUtils", "llvm", "lib/Transforms/Utils", "STATIC_LIBRARY", "lib/libLLVMTransformUtils.a"],
    ["LLVMipo", "llvm", "lib/IPO", "STATIC_LIBRARY", "lib/libLLVMipo.a"],
    ["LLVMCodeGen", "llvm", "lib/CodeGen", "STATIC_LIBRARY", "lib/libLLVMCodeGen.a"],
    ["LLVMPasses", "llvm", "lib/Passes", "STATIC_LIBRARY", "lib/libLLVMPasses.a"],
    ["LLVMLTO", "llvm", "lib/LTO", "STATIC_LIBRARY", "lib/libLLVMLTO.a"],
    ["LLVMX86CodeGen", "llvm", "lib/Target/X86", "STATIC_LIBRARY", "lib/libLLVMX86CodeGen.a"],
    ["LLVMX86AsmParser", "llvm", "lib/Target/X86/AsmParser", "STATIC_LIBRARY", "lib/libLLVMX86AsmParser.a"],
    ["LLVMNVPTXCodeGen", "llvm", "lib/Target/NVPTX", "STATIC_LIBRARY", "lib/libLLVMNVPTXCodeGen.a"],
    ["LLVMNVPTXDesc", "llvm", "lib/Target/NVPTX/MCTargetDesc", "STATIC_LIBRARY", "lib/libLLVMNVPTXDesc.a"],
    ["LLVMAMDGPUCodeGen", "llvm", "lib/Target/AMDGPU", "STATIC_LIBRARY", "lib/libLLVMAMDGPUCodeGen.a"],
    ["LLVMAMDGPUUtils", "llvm", "lib/Target/AMDGPU/Utils", "STATIC_LIBRARY", "lib/libLLVMAMDGPUUtils.a"],
    ["LLVMPrivate", "llvm", "lib/Private", "STATIC_LIBRARY", "lib/libLLVMPrivate.a"],
    ["clangAAA", "clang", "../clang/lib/Tooling/Refactoring/Rename", "STATIC_LIBRARY", "lib/libclangAAA.a"],
    ["clangBasic", "clang", "../clang/lib/Basic", "STATIC_LIBRARY", "lib/libclangBasic.a"],
    ["clangAST", "clang", "../clang/lib/AST", "STATIC_LIBRARY", "lib/libclangAST.a"],
    ["clangSema", "clang", "../clang/lib/Sema", "STATIC_LIBRARY", "lib/libclangSema.a"],
    ["clangFrontend", "clang", "../clang/lib/Frontend", "STATIC_LIBRARY", "lib/libclangFrontend.a"],
    ["lldCommon", "lld", "../lld/Common", "STATIC_LIBRARY", "lib/liblldCommon.a"],
    ["lld", "lld", "../lld/tools/lld", "EXECUTABLE", "bin/lld"],
    ["clang", "clang", "../clang/tools/driver", "EXECUTABLE", "bin/clang-21"],
    ["llvm-config", "llvm", "../llvm/tools/llvm-config", "EXECUTABLE", "bin/llvm-config"],
    ["llvm-ar", "llvm", "../llvm/tools/llvm-ar", "EXECUTABLE", "bin/llvm-ar"],
    ["llvm-nm", "llvm", "../llvm/tools/llvm-nm", "EXECUTABLE", "bin/llvm-nm"],
    ["llvm-rc", "llvm", "../llvm/tools/llvm-rc", "EXECUTABLE", "bin/llvm-rc"],
    ["llvm-ml", "llvm", "../llvm/tools/llvm-ml", "EXECUTABLE", "bin/llvm-ml"],
    ["llc", "llvm", "../llvm/tools/llc", "EXECUTABLE", "bin/llc"],
    ["opt", "llvm", "../llvm/tools/opt", "EXECUTABLE", "bin/opt"],
    ["llvm-link", "llvm", "../llvm/tools/llvm-link", "EXECUTABLE", "bin/llvm-link"],
    ["llvm-mc", "llvm", "../llvm/tools/llvm-mc", "EXECUTABLE", "bin/llvm-mc"],
    ["llvm-dis", "llvm", "../llvm/tools/llvm-dis", "EXECUTABLE", "bin/llvm-dis"],
    ["llvm-objcopy", "llvm", "../llvm/tools/llvm-objcopy", "EXECUTABLE", "bin/llvm-objcopy"],
    ["llvm-readobj", "llvm", "../llvm/tools/llvm-readobj", "EXECUTABLE", "bin/llvm-readobj"],
    ["llvm-tblgen", "llvm", "../llvm/utils/TableGen", "EXECUTABLE", "bin/llvm-tblgen"],
    ["clang-tblgen", "clang", "../clang/utils/TableGen", "EXECUTABLE", "bin/clang-tblgen"],
  ];
  const reply = path.join(f.build, ".cmake/api/v1/reply"); fs.mkdirSync(reply, {recursive: true});
  const refs = rows.map(([name, , sourcePath, type, artifact], index) => {
    const jsonFile = "target-" + index + ".json";
    writeJson(path.join(reply, jsonFile), {name, type, paths: {source: sourcePath, build: "target-" + index}, artifacts: [{path: artifact}]});
    return {id: name + ":" + type, jsonFile};
  });
  writeJson(path.join(reply, "index-test.json"), {reply: {"codemodel-v2": {jsonFile: "codemodel.json"}}});
  writeJson(path.join(reply, "codemodel.json"), {paths: {source: path.join(f.source, "llvm"), build: f.build}, configurations: [{targets: refs}]});
  const staticNames = rows.filter(([name, , , type]) => type === "STATIC_LIBRARY" && name !== "LLVMPrivate").map(([name]) => "install-" + name);
  const aliasNames = ["llvm-lib", "llvm-ranlib"];
  const targets = new Map([...staticNames, ...toolsComponents.map(name => "install-" + name), ...aliasNames.map(name => "install-" + name),
    ...metadataInstallTargets, "install-mom-distribution"].map(name => [name, []]));
  return {rows, graphTargets: llvmStaticTargets(f.build, f.source), targets};
}

test.describe("Public Linux LLVM 21 source SDK producer", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("pins the official source/compiler recipe and keeps the ABI export flags", t => {
    const f = fixture(t), plan = linuxLlvmSdkPlan(f.archive, f.output, f.tools, 8, "/sysroot/toolchain.cmake");
    assert.equal(archiveSha256, "4633a23617fa31a3ea51242586ea7fb1da7140e426bd62fc164261fe036aa142");
    assert.equal(path.basename(plan.archive), archiveName);
    assert.equal(plan.jobs, 8);
    assert.equal(plan.sourceOptionsFile, path.join(f.output, "llvm-host-source-options.cmake"));
    assert.ok(plan.configure.includes("-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=" + plan.sourceOptionsFile));
    assert.ok(plan.configure.includes("-DLLVM_ENABLE_PROJECTS=clang;lld"));
    assert.ok(plan.configure.includes("-DCMAKE_TOOLCHAIN_FILE=/sysroot/toolchain.cmake"), "the glibc sysroot selects headers and libraries");
    assert.ok(plan.configure.includes("-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU"));
    assert.equal(plan.configure.some(value => value.startsWith("-DLLD_INCLUDE_TESTS=")), false);
    assert.ok(plan.configure.includes("-DCMAKE_EXE_LINKER_FLAGS=-Wl,--export-dynamic"));
    assert.ok(plan.configure.includes("-DLLVM_ENABLE_PIC=ON"));
    assert.ok(plan.configure.includes("-DLLVM_ENABLE_PLUGINS=OFF"));
    assert.ok(plan.configure.includes("-DLLVM_EXPORT_SYMBOLS_FOR_PLUGINS=OFF"));
    assert.ok(plan.configure.includes("-DLLVM_ENABLE_LTO=OFF"));
    assert.ok(plan.selectedConfigure.at(-1)?.includes("<generated-components>"));
    assert.throws(() => linuxLlvmSdkPlan(f.archive, f.output, {...f.tools, cc: "/intel/clang"}, 2, "/sysroot/toolchain.cmake"), /Intel tools/);
    assert.throws(() => linuxLlvmSdkPlan(f.archive, f.output, f.tools, 33, "/sysroot/toolchain.cmake"), /jobs in 1\.\.32/);
  });

  test("generates a deferred source-scoped host shim with safe literal path quoting", t => {
    const f = fixture(t), source = path.join(f.root, "source with spaces ]]"), shim = llvmAmdgpulibCallsCompileOptionsShim(source);
    assert.match(shim, /cmake_language\(DEFER CALL _mom_llvm_amdgpulibcalls_host_option\)/);
    assert.match(shim, /SOURCE \[=\[/);
    assert.ok(shim.includes("APPEND PROPERTY COMPILE_OPTIONS [[-fno-builtin]]"));
    assert.match(shim, /source with spaces/);
    assert.throws(() => llvmAmdgpulibCallsCompileOptionsShim(path.join(f.root, "bad;source")), /canonical absolute/);
    assert.doesNotThrow(() => verifyCmakeVersion("cmake version 3.24.0\n"));
    assert.doesNotThrow(() => verifyCmakeVersion("cmake version 4.2.1\n"));
    assert.throws(() => verifyCmakeVersion("cmake version 3.23.9\n"), /3.24 or newer/);
    assert.throws(() => verifyCmakeVersion("unrecognized output"), /3.24 or newer/);
  });

  test("requires exactly AMDGPULibCalls.cpp to receive the host-only option", t => {
    const f = fixture(t), sourceFile = path.join(f.source, "llvm/lib/Target/AMDGPU/AMDGPULibCalls.cpp"),
      otherFile = path.join(f.source, "llvm/lib/Support/Other.cpp"), database = path.join(f.build, "compile_commands.json");
    /** @param {string} file @param {boolean} flagged */
    const row = (file, flagged) => ({directory: f.build, file,
      command: "icpx " + (flagged ? "-fno-builtin " : "") + "-c " + file});
    fs.writeFileSync(database, JSON.stringify([row(sourceFile, true), row(otherFile, false)]) + "\n");
    assert.equal(verifyAmdgpulibCallsCompileFlag(f.build, f.source).compileCommands, 2);
    fs.writeFileSync(database, JSON.stringify([row(sourceFile, false), row(otherFile, false)]) + "\n");
    assert.throws(() => verifyAmdgpulibCallsCompileFlag(f.build, f.source), /exactly one host-only/);
    const duplicate = row(sourceFile, false);
    duplicate.command += " -fno-builtin -fno-builtin";
    fs.writeFileSync(database, JSON.stringify([duplicate]) + "\n");
    assert.throws(() => verifyAmdgpulibCallsCompileFlag(f.build, f.source), /exactly one host-only/);
    const quoted = row(otherFile, false);
    quoted.command += ' -DOTHER="text -fno-builtin text"';
    fs.writeFileSync(database, JSON.stringify([row(sourceFile, true), quoted]) + "\n");
    assert.equal(verifyAmdgpulibCallsCompileFlag(f.build, f.source).compileCommands, 2);
    const escaped = row(sourceFile, false);
    escaped.command += String.raw` "\-fno-builtin"`;
    fs.writeFileSync(database, JSON.stringify([escaped]) + "\n");
    assert.throws(() => verifyAmdgpulibCallsCompileFlag(f.build, f.source), /exactly one host-only/);
    fs.writeFileSync(database, JSON.stringify([row(sourceFile, true), row(otherFile, true)]) + "\n");
    assert.throws(() => verifyAmdgpulibCallsCompileFlag(f.build, f.source), /escaped/);
    fs.writeFileSync(database, JSON.stringify([row(sourceFile, true), row(sourceFile, true)]) + "\n");
    assert.throws(() => verifyAmdgpulibCallsCompileFlag(f.build, f.source), /exactly one/);
  });

  test("rejects symlinked build artifacts", t => {
    const f = fixture(t), outside = path.join(f.root, "outside-file"); fs.writeFileSync(outside, "outside\n");
    targetFixture(f);
    fs.mkdirSync(path.join(f.build, "lib"), {recursive: true});
    fs.symlinkSync(outside, path.join(f.build, "lib", "libLLVMCore.a"));
    assert.throws(() => llvmStaticTargets(f.build, f.source), /symlink|escapes/);
  });

  test("admits LLVM's sibling executable alias without permitting escaping or chained links", t => {
    const f = fixture(t), model = targetFixture(f), targetIndex = model.rows.findIndex(([name]) => name === "clang"),
      targetFile = path.join(f.build, ".cmake/api/v1/reply/target-" + targetIndex + ".json"),
      target = JSON.parse(fs.readFileSync(targetFile, "utf8")), binary = path.join(f.build, "bin/clang-21"),
      alias = path.join(f.build, "bin/clang");
    target.artifacts = [{path: "bin/clang"}]; writeJson(targetFile, target);
    fs.mkdirSync(path.dirname(binary)); fs.writeFileSync(binary, "compiler\n");
    fs.symlinkSync("clang-21", alias);
    assert.equal(llvmStaticTargets(f.build, f.source).find(row => row.name === "clang")?.artifact, alias);
    fs.rmSync(alias); fs.symlinkSync("../../outside", alias);
    assert.throws(() => llvmStaticTargets(f.build, f.source), /alias|symlink|escapes/);
    fs.rmSync(alias); fs.symlinkSync("clang-next", alias); fs.symlinkSync("clang-21", path.join(f.build, "bin/clang-next"));
    assert.throws(() => llvmStaticTargets(f.build, f.source), /alias|symlink|escapes/);
  });

  test("reads the pinned LLVM version through the upstream CMake include chain", t => {
    const f = fixture(t), llvmCmake = path.join(f.source, "llvm", "CMakeLists.txt"), versionModule = path.join(f.source, "cmake", "Modules", "LLVMVersion.cmake");
    fs.writeFileSync(llvmCmake, [
      // eslint-disable-next-line no-template-curly-in-string -- Literal upstream CMake expansion.
      "set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)",
      // eslint-disable-next-line no-template-curly-in-string -- Literal upstream CMake expansion.
      "include(${LLVM_COMMON_CMAKE_UTILS}/Modules/LLVMVersion.cmake)", "",
    ].join("\n"));
    fs.mkdirSync(path.dirname(versionModule), {recursive: true});
    fs.writeFileSync(versionModule, "if(NOT DEFINED LLVM_VERSION_MAJOR)\n  set(LLVM_VERSION_MAJOR 21)\nendif()\n" +
      "if(NOT DEFINED LLVM_VERSION_MINOR)\n  set(LLVM_VERSION_MINOR 1)\nendif()\n" +
      "if(NOT DEFINED LLVM_VERSION_PATCH)\n  set(LLVM_VERSION_PATCH 8)\nendif()\n");
    assert.doesNotThrow(() => verifySourceVersion(f.source));
    fs.writeFileSync(versionModule, fs.readFileSync(versionModule, "utf8").replace("LLVM_VERSION_MAJOR 21", "LLVM_VERSION_MAJOR 22"));
    assert.throws(() => verifySourceVersion(f.source), /not 21\.1\.8: LLVM_VERSION_MAJOR/);
  });

  test("shares the exact pinned source membership and source hash across platforms", t => {
    const f = fixture(t);
    for (const name of sourceDirectories) {fs.writeFileSync(path.join(f.source, name, "input.txt"), name + "\n");}
    fs.writeFileSync(path.join(f.source, "LICENSE.TXT"), "license\n");
    const before = llvmSourceSnapshot(f.source);
    fs.writeFileSync(path.join(f.source, "libunwind/include", "compact_unwind_encoding.h"), "required source header\n");
    assert.notEqual(llvmSourceSnapshot(f.source).sha256, before.sha256);
    assert.throws(() => llvmSourceSnapshot(path.join(f.root, "missing-source")), /ENOENT/);
    fs.writeFileSync(path.join(f.source, "llvm", "new-member.txt"), "membership changed\n");
    assert.notEqual(llvmSourceSnapshot(f.source).sha256, before.sha256, "new source members invalidate the frozen input snapshot");
  });

  test("rejects missing source roots and marks content changes stale", t => {
    const f = fixture(t);
    fs.writeFileSync(path.join(f.source, "LICENSE.TXT"), "license\n");
    fs.writeFileSync(path.join(f.source, "llvm", "input.txt"), "pinned\n");
    const pinned = llvmSourceSnapshot(f.source);
    fs.writeFileSync(path.join(f.source, "llvm", "input.txt"), "changed\n");
    assert.notDeepEqual(llvmSourceSnapshot(f.source), pinned);
    fs.rmSync(path.join(f.source, "libunwind/include"), {recursive: true});
    assert.throws(() => llvmSourceSnapshot(f.source), /Missing original LLVM 21 source directory: libunwind\/include/);
  });

  test("rejects unsafe archive catalogs and requires original libunwind headers", t => {
    const f = fixture(t), upstream = path.join(f.root, "llvm-project-21.1.8.src");
    for (const name of sourceDirectories) {fs.mkdirSync(path.join(upstream, name), {recursive: true});}
    fs.writeFileSync(path.join(upstream, "LICENSE.TXT"), "license\n");
    const archive = path.join(f.root, "catalog.tar");
    const create = spawnSync("/usr/bin/tar", ["-cf", archive, "--directory", f.root, "llvm-project-21.1.8.src"], {encoding: "utf8"});
    assert.equal(create.status, 0, create.stderr || create.error?.message);
    assert.ok(verifyArchiveCatalog(archive).members > 0);
    fs.rmSync(path.join(upstream, "libunwind/include"), {recursive: true});
    const missing = spawnSync("/usr/bin/tar", ["-cf", archive, "--directory", f.root, "llvm-project-21.1.8.src"], {encoding: "utf8"});
    assert.equal(missing.status, 0, missing.stderr || missing.error?.message);
    assert.throws(() => verifyArchiveCatalog(archive), /omitted required source: libunwind\/include/);
  });

  test("derives installable archives, phase groups, and tool closure from CMake target metadata", t => {
    const f = fixture(t), model = targetFixture(f), selection = linuxDistributionSelection(model.graphTargets, model.targets, f.build);
    assert.deepEqual(selection.components.includes("LLVMPrivate"), false, "uninstallable internals stay outside the SDK");
    assert.deepEqual(selection.components.filter(name => name.startsWith("LLVM")), ["LLVMAMDGPUCodeGen", "LLVMAMDGPUUtils", "LLVMAnalysis", "LLVMBinaryFormat",
      "LLVMBitReader", "LLVMBitstreamReader", "LLVMCodeGen", "LLVMCore", "LLVMDemangle", "LLVMLTO", "LLVMNVPTXCodeGen", "LLVMNVPTXDesc", "LLVMNotSupport",
      "LLVMObject", "LLVMPasses", "LLVMRemarks", "LLVMSupport", "LLVMSupportParallel", "LLVMTargetParser", "LLVMTransformUtils", "LLVMX86AsmParser", "LLVMX86CodeGen", "LLVMipo"]);
    assert.deepEqual(selection.tools.map(row => row.name), [...toolsComponents]);
    assert.equal(toolsComponents.includes("llvm-mt"), false, "llvm-mt is not built with the pinned no-LibXml2 configuration");
    assert.deepEqual(selection.distributionComponents.filter(name => ["llvm-lib", "llvm-ranlib"].includes(name)),
      ["llvm-lib", "llvm-ranlib"], "the MoM distribution explicitly installs the source-defined archive aliases");
    assert.ok(selection.tools.some(row => row.name === "lld"), "ROCr assembly and AdaptiveCpp host links need the LLD driver, not just its archives");
    assert.ok(selection.tools.some(row => row.name === "llvm-readobj"), "the Windows native component admission needs LLVM's COFF reader");
    assert.equal(selection.components.includes("llvm-mca"), false);
    assert.deepEqual(selection.phases.map(row => row.name), ["llvm-foundation", "llvm-core-analysis", "llvm-core-optimization", "llvm-core-codegen",
      "llvm-x86", "llvm-nvptx", "llvm-amdgpu", "clang-base", "clang-sema", "clang-frontend", "lld", "tools"]);
    const phaseTargets = selection.phases.flatMap(row => row.targets);
    assert.deepEqual(phaseTargets.slice().sort(), selection.components);
    assert.equal(new Set(phaseTargets).size, phaseTargets.length, "every selected component belongs to exactly one phase");
    assert.deepEqual(selection.phases.find(row => row.name === "llvm-foundation")?.targets, ["LLVMBinaryFormat", "LLVMBitstreamReader", "LLVMCore",
      "LLVMDemangle", "LLVMRemarks", "LLVMSupport", "LLVMSupportParallel", "LLVMTargetParser"], "the specified directories and descendants build first");
    assert.deepEqual(selection.phases.find(row => row.name === "llvm-core-analysis")?.targets, ["LLVMAnalysis", "LLVMBitReader", "LLVMObject"],
      "analysis and object archives build after the foundation layer");
    assert.deepEqual(selection.phases.find(row => row.name === "llvm-core-optimization")?.targets, ["LLVMipo", "LLVMTransformUtils"],
      "the transform and IPO libraries build after their analysis inputs");
    assert.deepEqual(selection.phases.find(row => row.name === "llvm-core-codegen")?.targets, ["LLVMCodeGen", "LLVMLTO", "LLVMNotSupport", "LLVMPasses"],
      "CodeGen, Passes, LTO and remaining common archives stay in the final common layer");
    assert.deepEqual(selection.phases.find(row => row.name === "llvm-x86")?.targets, ["LLVMX86AsmParser", "LLVMX86CodeGen"],
      "backend descendants remain in their architecture phase");
    assert.deepEqual(selection.phases.find(row => row.name === "llvm-nvptx")?.targets, ["LLVMNVPTXCodeGen", "LLVMNVPTXDesc"]);
    assert.deepEqual(selection.phases.find(row => row.name === "llvm-amdgpu")?.targets, ["LLVMAMDGPUCodeGen", "LLVMAMDGPUUtils"]);
    assert.deepEqual(selection.phases.find(row => row.name === "clang-base")?.targets, ["clangAST", "clangBasic"]);
    assert.deepEqual(selection.phases.find(row => row.name === "clang-sema")?.targets, ["clangSema"]);
    assert.deepEqual(selection.phases.find(row => row.name === "clang-frontend")?.targets, ["clangAAA", "clangFrontend"]);
    assert.ok(selection.archives.every(row => row.relativeArtifact.startsWith("lib/")));
    for (const name of toolsComponents) {
      const missingTool = new Map(model.targets); missingTool.delete("install-" + name);
      assert.throws(() => linuxDistributionSelection(model.graphTargets, missingTool, f.build), new RegExp("required source tool: " + name));
    }
    assert.throws(() => linuxDistributionSelection(model.graphTargets.filter(row => row.name !== "llvm-readobj"), model.targets, f.build),
      /required source tool: llvm-readobj/, "a missing CMake target must not create a partial SDK");
    const escapedSource = model.graphTargets.map(row => row.name === "LLVMX86CodeGen" ? {...row, sourceDirectory: f.root} : row);
    assert.throws(() => linuxDistributionSelection(escapedSource, model.targets, f.build), /target source roots are inconsistent/,
      "architecture grouping must not admit LLVM target paths outside the selected source root");
    const wrongReadobjType = model.graphTargets.map(row => row.name === "llvm-readobj" ? {...row, type: "STATIC_LIBRARY"} : row);
    assert.throws(() => linuxDistributionSelection(wrongReadobjType, model.targets, f.build),
      /required source tool: llvm-readobj/, "a non-executable target cannot stand in for the source reader");
  });

  test("captures the actual clang link command needed by the ABI evidence", t => {
    const f = fixture(t), phase = {name: "tools", targets: ["clang"], artifacts: []}, args = phaseBuildArgs(f.build, 8, phase),
      log = path.join(f.root, "build-tools.log"), linkCommand = f.tools.cxx + " -fPIC -Wl,--export-dynamic -o " + path.join(f.build, "bin/clang-21");
    assert.ok(args.includes("--verbose"), "the final tools phase must make Ninja print compiler/link commands");
    assert.deepEqual(args.slice(-3), ["--", "-k", "0"]);
    fs.writeFileSync(log, "[1/1] Linking CXX executable bin/clang-21\n");
    assert.throws(() => findClangLinkCommand(f.root, [phase], f.tools.cxx), /exactly one PIC Intel clang link/);
    fs.writeFileSync(log, linkCommand + "\n");
    assert.equal(findClangLinkCommand(f.root, [phase], f.tools.cxx).command, linkCommand);

    const read = fs.readSync;
    fs.readSync = /** @type {typeof fs.readSync} */ ((...args) => {
      fs.readSync = read;
      fs.appendFileSync(log, "more output\n");
      return Reflect.apply(read, fs, args);
    });
    try {
      assert.throws(() => findClangLinkCommand(f.root, [phase], f.tools.cxx), /changed during admission/);
    } finally {fs.readSync = read;}
  });

  test("requires source-consumer LLVM tools in the generated graph", t => {
    const f = fixture(t), model = targetFixture(f), selection = linuxDistributionSelection(model.graphTargets, model.targets, f.build);
    for (const name of ["llc", "opt", "llvm-link", "llvm-mc", "llvm-dis", "llvm-objcopy"]) {
      assert.ok(selection.tools.some(row => row.name === name), "the SDK requires its selected LLVM " + name);
      const missing = new Map(model.targets); missing.delete("install-" + name);
      assert.throws(() => linuxDistributionSelection(model.graphTargets, missing, f.build), new RegExp("required source tool: " + name));
    }
  });

  test("checks generated distribution and install metadata targets before any build", t => {
    const f = fixture(t), model = targetFixture(f), selection = linuxDistributionSelection(model.graphTargets, model.targets, f.build),
      aliases = {"llvm-lib": "tools/llvm-ar/llvm-lib", "llvm-ranlib": "tools/llvm-ar/llvm-ranlib"},
      outputs = [...selection.archives, ...selection.tools].map(row => row.relativeArtifact).concat(Object.values(aliases)).sort();
    const aliasRows = Object.entries(aliases).flatMap(([name, edge]) => [
      `build ${name}: phony ${edge}`,
      `build ${edge}: phony bin/${name} bin/llvm-ar`,
    ]);
    const rows = ["build mom-distribution: phony " + outputs.join(" "), "build install-mom-distribution: phony install-all",
      ...selection.distributionComponents.map(name => "build install-" + name + ": phony install-" + name),
      ...aliasRows,
      ...metadataInstallTargets.map(name => "build " + name + ": phony install-metadata")];
    fs.writeFileSync(path.join(f.build, "build.ninja"), rows.join("\n") + "\n");
    assert.equal(verifyDistributionGraph(f.build, selection).install, "install-mom-distribution");
    assert.deepEqual(installTargets, ["install-mom-distribution", ...metadataInstallTargets]);
    const omitted = rows.filter(row => !row.startsWith("build install-clang-headers:"));
    fs.writeFileSync(path.join(f.build, "build.ninja"), omitted.join("\n") + "\n");
    assert.throws(() => verifyDistributionGraph(f.build, selection), /does not match its target\/install graph/);
    const stale = rows.filter(row => !row.startsWith("build install-clang-headers:")).concat("build install-new-component: phony install-new-component");
    fs.writeFileSync(path.join(f.build, "build.ninja"), stale.join("\n") + "\n");
    assert.throws(() => verifyDistributionGraph(f.build, selection), /does not match its target\/install graph/);
  });

  test("audits only regular static archives and requires installed upstream headers and exports", t => {
    const f = fixture(t), model = targetFixture(f), selection = linuxDistributionSelection(model.graphTargets, model.targets, f.build),
      prefix = path.join(f.root, "sdk");
    for (const row of selection.archives) {
      const file = path.join(prefix, row.relativeArtifact); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "!<arch>\n");
    }
    for (const name of ["lib/cmake/llvm/LLVMConfig.cmake", "lib/cmake/clang/ClangConfig.cmake", "lib/cmake/lld/LLDConfig.cmake",
      "include/llvm/ADT/StringRef.h", "include/clang/AST/ASTContext.h", "include/lld/Common/Driver.h", "bin/clang", "bin/clang++"]) {
      const file = path.join(prefix, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "present\n");
    }
    for (const row of selection.tools) {
      const file = path.join(prefix, row.relativeArtifact); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "tool\n");
    }
    fs.symlinkSync("llvm-ar", path.join(prefix, "bin/llvm-lib"));
    fs.symlinkSync("llvm-ar", path.join(prefix, "bin/llvm-ranlib"));
    fs.symlinkSync("lld", path.join(prefix, "bin/lld-link"));
    const linkerAlias = path.join(prefix, "bin/ld.lld");
    fs.symlinkSync("lld", linkerAlias);
    assert.equal(auditLinuxLlvmPrefix(prefix, selection).archives, selection.archives.length);
    for (const name of ["llc", "opt", "llvm-link", "llvm-mc", "llvm-dis", "llvm-objcopy"]) {
      const file = path.join(prefix, "bin", name), contents = fs.readFileSync(file);
      fs.rmSync(file);
      assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /omitted source tool output/);
      fs.writeFileSync(file, contents);
    }
    fs.rmSync(linkerAlias);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /linker driver alias/);
    fs.symlinkSync("clang-21", linkerAlias);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /linker driver alias/);
    fs.rmSync(linkerAlias); fs.symlinkSync("lld", linkerAlias);
    const llvmLibAlias = path.join(prefix, "bin/llvm-lib"), llvmLibTarget = fs.readlinkSync(llvmLibAlias);
    fs.rmSync(llvmLibAlias);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /required tool alias/);
    fs.symlinkSync("../llvm-ar", llvmLibAlias);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /required tool alias/);
    fs.rmSync(llvmLibAlias); fs.symlinkSync(llvmLibTarget, llvmLibAlias);
    const unexpected = path.join(prefix, "lib/libUnexpected.a");
    fs.writeFileSync(unexpected, "!<arch>\n");
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /archive set differs/);
    fs.rmSync(unexpected);
    const header = path.join(prefix, "include/lld/Common/Driver.h"), headerContents = fs.readFileSync(header);
    fs.rmSync(header);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /omitted required upstream export\/header/);
    fs.writeFileSync(header, headerContents);
    const archive = path.join(prefix, selection.archives[0]?.relativeArtifact || "");
    assert.ok(archive.startsWith(prefix + path.sep));
    const archiveContents = fs.readFileSync(archive);
    fs.writeFileSync(archive, "corrupt\n");
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /not a regular static archive/);
    fs.writeFileSync(archive, archiveContents);
    const escaped = path.join(prefix, "escaping-link");
    fs.symlinkSync(path.join(f.root, "outside"), escaped);
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /escaping symlink/);
    fs.rmSync(escaped);
    let deep = prefix;
    for (let index = 0; index < 65; index++) {deep = path.join(deep, "d"); fs.mkdirSync(deep);}
    assert.throws(() => auditLinuxLlvmPrefix(prefix, selection), /output inventory exceeds bounds/);
  });

  test("parses CMake cache values without losing list order", t => {
    const f = fixture(t), file = path.join(f.root, "CMakeCache.txt");
    fs.writeFileSync(file, "# generated\nLLVM_ENABLE_PROJECTS:STRING=clang;lld\nCMAKE_EXE_LINKER_FLAGS:STRING=-Wl,--export-dynamic\n");
    const cache = cmakeCache(file);
    assert.equal(cache.get("LLVM_ENABLE_PROJECTS"), "clang;lld");
    assert.equal(cache.get("CMAKE_EXE_LINKER_FLAGS"), "-Wl,--export-dynamic");
  });
});
