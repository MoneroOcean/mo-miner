"use strict";

// Export the verified Windows LLVM/COMGR/AdaptiveCpp/HIPRTC archives as one relocatable link fragment.
// Producer receipts stay untouched; the component is transport, not runtime qualification.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson, readBuildText, sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {isStaticArchive} = require("./native-audit");
const {auditCoffArchiveInputs} = require("./native-coff-archive");
const {verifyWindowsLlvmAdmission, archiveSha256} = require("./build-native-windows-llvm-sdk");
const {revision: comgrRevision, recipeFiles: comgrRecipeFiles, hostSources, verifyWindowsDeviceResources,
  windowsComgrSdkPlan, auditComgrCompileCommands, auditComgrArchiveMembers, comgrArchive} = require("./build-native-windows-comgr-sdk");
const {admitSyclSource} = require("./native-sycl-inputs");
const {windowsAcppStaticHostPlan, artifacts: acppArtifacts, targets: acppTargets, auditAcppCompileCommands,
  forbiddenOutputs: acppForbiddenOutputs, recipeFiles: acppRecipeFiles, verifyWindowsRocmResources, buildInputKeys} = require("./build-native-windows-acpp-sdk");
const {clrRevision: hiprtcSourceRevision, sourceFiles: hiprtcSourceFiles, recipeFiles: hiprtcRecipeFiles,
  verifyIntelCompilerIdentity, verifyLlvmToolIdentity, auditHiprtcSymbolClosure} = require("./build-native-windows-hiprtc-sdk");
const {jitBuildOutputs} = require("./native-jit-stages");

const repo = path.resolve(__dirname, "..");
const targetTriple = "x86_64-pc-windows-msvc";
const llvmArchiveName = "llvm-project-21.1.8.src.tar.xz";
const llvmArchiveCount = 164, comgrArchiveCount = 1, acppArchiveCount = 9;
const acppRevision = "da2463e45aa90aa36306c45abcfc05b87de51bc6";
const hiprtcArchiveName = "hiprtc-host-intel.lib";
const hiprtcArchiveMembers = Object.freeze([...hiprtcSourceFiles.map(file => path.basename(file, ".cpp") + ".obj"), "hipRTC_header.obj"]);
const componentPins = Object.freeze({version: 3, targetTriple, linkOrder: ["acpp", "hiprtc", "comgr", "llvm"],
  llvm: {sourceArchiveSha256: archiveSha256, archives: llvmArchiveCount},
  comgr: {sourceRevision: comgrRevision, archives: comgrArchiveCount, objects: hostSources.length},
  acpp: {sourceRevision: acppRevision, archives: acppArchiveCount},
  hiprtc: {sourceRevision: hiprtcSourceRevision, comgrRevision, archiveName: hiprtcArchiveName, archives: 1, objects: hiprtcArchiveMembers.length}});
const groups = Object.freeze(["acpp", "hiprtc", "comgr", "llvm"]);
const runtimeInputPaths = Object.freeze({cudartStatic: "runtime/cudart_static.lib", amdhip64: "runtime/amdhip64.lib"});
const archiveBytesLimit = 256 * 1024 * 1024, componentBytesLimit = 1024 * 1024 * 1024;
const archiveName = /^[A-Za-z0-9_.+-]{1,160}\.lib$/;
const hashPattern = /^[a-f0-9]{64}$/;
const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/** @param {unknown} value @returns {value is string} */
function absolute(value) {
  return typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) && path.normalize(value) === value &&
    !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
}

/** @param {string} file @param {string} label */
function canonicalFile(file, label) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!absolute(file) || !stat?.isFile() || fs.realpathSync(file) !== file) {
    throw new Error("Windows host component " + label + " must be a canonical regular file: " + file);
  }
  return file;
}

/** @param {string} root @param {string} file */
function contained(root, file) {
  const relative = path.relative(root, file);
  return relative !== "" && !path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep);
}

/** @param {unknown} value @param {string} label */
function verifyHashMap(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value) || !Object.keys(value).length || Object.keys(value).length > 65536 ||
      Object.entries(value).some(([file, hash]) => !absolute(file) || typeof hash !== "string" || !hashPattern.test(hash))) {
    throw new Error("Invalid Windows host component " + label + " hash map");
  }
  for (const [file, hash] of Object.entries(value)) {
    if (fileHash(file) !== hash) {throw new Error("Windows host component " + label + " changed: " + file);}
  }
}

/** @param {string} output @param {string[]} inputs */
function requireDisjointOutput(output, inputs) {
  if (!absolute(output) || fs.lstatSync(output, {throwIfNoEntry: false}) ||
      path.join(fs.realpathSync(path.dirname(output)), path.basename(output)) !== output) {
    throw new Error("Windows host component output must be new and canonical");
  }
  for (const input of inputs) {
    const selected = fs.realpathSync(input), relative = path.relative(selected, output), reverse = path.relative(output, selected);
    if (relative === "" || reverse === "" || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep)) ||
        (!path.isAbsolute(reverse) && reverse !== ".." && !reverse.startsWith(".." + path.sep))) {
      throw new Error("Windows host component output overlaps a producer input");
    }
  }
}

/** @param {string} file @param {string} sha256 @param {number} bytes @param {string} label */
function archiveRecord(file, sha256, bytes, label) {
  canonicalFile(file, label);
  if (!/\.lib$/i.test(file) || !hashPattern.test(sha256) || !Number.isSafeInteger(bytes) || bytes < 8 || bytes > archiveBytesLimit ||
      fs.statSync(file).size !== bytes || !isStaticArchive(file) || fileHash(file) !== sha256) {
    throw new Error("Windows host component archive differs from its producer receipt: " + file);
  }
  return {file, sha256, bytes};
}

/** Resolve the immutable source/build pair from a verified Windows LLVM receipt.
 * A refresh output may name a new identity root while reusing the original configured tree.
 * @param {ReturnType<typeof verifyWindowsLlvmAdmission>} llvm */
function windowsHostComponentLlvmLayout(llvm) {
  const plan = llvm.receipt.plan;
  return {source: plan.source, build: path.join(plan.output, "build")};
}

/** @param {string} resultFile @param {ReturnType<typeof verifyWindowsLlvmAdmission>} llvm */
function verifyComgrResult(resultFile, llvm) {
  canonicalFile(resultFile, "COMGR result receipt");
  const receiptSha256 = fileHash(resultFile), receipt = readBuildJson(resultFile), saved = receipt.plan,
    planProof = saved ? readBuildJson(path.join(saved.output, "plan.json")) : undefined;
  if (!saved || receipt.version !== 1 || receipt.role !== "windows-comgr-static-host" || receipt.stage !== "complete" ||
      receipt.buildOnly !== true || receipt.compiled !== true || receipt.executed !== false || receipt.publicationQualified !== false ||
      receipt.revision !== comgrRevision ||
      saved.output !== path.dirname(resultFile) || path.join(saved.output, "result.json") !== resultFile ||
      fileHash(path.join(saved.output, "plan.json")) !== receipt.planSha256 || saved.targetLlvm !== llvm.receipt.plan.output ||
      !planProof || planProof.revision !== receipt.revision || !hashPattern.test(planProof.archiveSha256) ||
      !isDeepStrictEqual(planProof.sourceState, receipt.sourceState)) {
    throw new Error("Windows COMGR result is incomplete or does not match its frozen output plan");
  }
  const descriptor = readBuildJson(saved.descriptorFile), plan = windowsComgrSdkPlan({...saved, descriptor}, windowsHostComponentLlvmLayout(llvm));
  const device = verifyWindowsDeviceResources(saved.deviceReceipt, saved.deviceInstall);
  const windowsLlvm = receipt.windowsLlvm;
  if (device.sha256 !== receipt.device?.sha256 || receipt.device?.receipt !== device.receiptFile ||
      receipt.device?.revision !== device.revision || receipt.device?.clrRevision !== device.clrRevision ||
      !isDeepStrictEqual(receipt.device?.files, device.files) || !isDeepStrictEqual(receipt.device?.device, device.device) ||
      !isDeepStrictEqual(receipt.patches, plan.patches) || !windowsLlvm || windowsLlvm.generationId !== llvm.admission.generationId ||
      windowsLlvm.receipt !== llvm.receiptFile || windowsLlvm.receiptSha256 !== fileHash(llvm.receiptFile) ||
      windowsLlvm.admission !== llvm.admissionFile || windowsLlvm.admissionSha256 !== fileHash(llvm.admissionFile) ||
      windowsLlvm.objectProof !== llvm.objectsFile || windowsLlvm.objectProofSha256 !== fileHash(llvm.objectsFile)) {
    throw new Error("Windows COMGR result is not bound to the selected LLVM admission");
  }
  verifySourceInputs(receipt.recipe, sourceInputs(repo, [], [...comgrRecipeFiles]), "Windows COMGR recipe");
  if (!isDeepStrictEqual(receipt.sourceState, sourceInputs(plan.sourceRoot, ["amd/comgr"], []))) {
    throw new Error("Windows COMGR source files changed after their receipt");
  }
  for (const map of [receipt.nativeFiles, receipt.fingerprints, receipt.configuredFiles, receipt.exports]) {verifyHashMap(map, "COMGR input");}
  const buildOutputs = jitBuildOutputs(plan.build), configuredOutputs = receipt.configuredOutputs;
  if (receipt.archive !== plan.archive || fileHash(receipt.archive) !== planProof.archiveSha256 ||
      !configuredOutputs || typeof configuredOutputs !== "object" || Array.isArray(configuredOutputs) || !Object.keys(configuredOutputs).length ||
      Object.entries(configuredOutputs).some(([name, value]) => typeof value !== "string" || buildOutputs[name] !== value) ||
      !isDeepStrictEqual(receipt.buildOutputs, buildOutputs) || Object.keys(buildOutputs).some(name => /\.(?:exe|dll)$/i.test(name))) {
    throw new Error("Windows COMGR archive, source tar or build outputs changed");
  }
  const compileCommands = auditComgrCompileCommands(path.join(plan.build, "compile_commands.json"),
    plan.compilerTools.CMAKE_CXX_COMPILER, plan.snapshot, plan.build, true);
  if (!isDeepStrictEqual(receipt.compileCommands, compileCommands) ||
      !isDeepStrictEqual(receipt.objects, compileCommands.objects) ||
      receipt.archiveTarget !== "amd_comgr" || receipt.header !== path.join(plan.build, "include/amd_comgr.h") ||
      fileHash(receipt.header) !== receipt.headerSha256 ||
      Object.keys(receipt.exports).length !== 3 || Object.keys(receipt.exports).some(file => !contained(plan.build, file))) {
    throw new Error("Windows COMGR compile, header or export receipt changed");
  }
  const target = comgrArchive(plan.build, plan.snapshot);
  if (receipt.archivePath !== target.archive || !isStaticArchive(target.archive) ||
      fs.statSync(target.archive).size !== receipt.archiveBytes || fileHash(target.archive) !== receipt.archiveSha256 ||
      !Array.isArray(receipt.archiveMembers) || !Array.isArray(compileCommands.objects) ||
      receipt.archiveMembers.length !== hostSources.length || receipt.archiveBytes < 8) {
    throw new Error("Windows COMGR target archive differs from its CMake target");
  }
  auditComgrArchiveMembers(receipt.archiveMembers.join("\n"), compileCommands.objects, plan.build);
  const expectedObjects = compileCommands.objects.map(row => {
    const relative = path.relative(plan.build, row.output).split(path.sep).join("/");
    if (!contained(plan.build, row.output)) {throw new Error("Windows COMGR object escaped its build tree");}
    return relative;
  });
  const audited = auditCoffArchiveInputs(target.archive, plan.build, expectedObjects);
  if (audited.archiveSha256 !== receipt.archiveSha256 || audited.members !== receipt.archiveMembers.length ||
      audited.objects.some((row, index) => row.sha256 !== compileCommands.objects[index]?.sha256 || row.bytes !== compileCommands.objects[index]?.bytes)) {
    throw new Error("Windows COMGR archive members differ from their compiled object receipt");
  }
  if (fileHash(resultFile) !== receiptSha256) {throw new Error("Windows COMGR result receipt changed during admission");}
  return {receipt, receiptFile: resultFile, receiptSha256, archive: archiveRecord(target.archive, receipt.archiveSha256, receipt.archiveBytes, "COMGR")};
}

/** Read-only git callback used by the shared AdaptiveCpp source admission. @param {string} command @param {string[]} args */
function readOnlyGit(command, args) {
  if (command !== "git") {throw new Error("Unexpected AdaptiveCpp provenance tool");}
  const result = spawnSync("/usr/bin/git", args, {encoding: "utf8", timeout: 30000, maxBuffer: 8 << 20,
    env: {PATH: "/usr/bin:/bin", LC_ALL: "C", GIT_NO_LAZY_FETCH: "1", GIT_NO_REPLACE_OBJECTS: "1",
      GIT_OPTIONAL_LOCKS: "0", GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null", GIT_ATTR_NOSYSTEM: "1",
      GIT_TERMINAL_PROMPT: "0"}});
  if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string") {
    throw new Error("AdaptiveCpp source provenance check failed", {cause: result.error});
  }
  return result.stdout;
}

/** @param {string} resultFile @param {ReturnType<typeof verifyWindowsLlvmAdmission>} llvm */
function verifyAcppResult(resultFile, llvm) {
  canonicalFile(resultFile, "AdaptiveCpp result receipt");
  const receiptSha256 = fileHash(resultFile), receipt = readBuildJson(resultFile), output = path.dirname(resultFile),
    saved = readBuildJson(path.join(output, "plan.json"));
  const artifacts = /** @type {Array<{path: string, sha256: string, bytes: number, members: number,
    objects: Array<{path: string, sha256: string, bytes: number}>}>} */ (receipt.artifacts);
  const build = path.join(output, "build");
  if (receipt.version !== 1 || receipt.role !== "windows-acpp-static-host" || receipt.buildOnly !== true || receipt.compiled !== true ||
      receipt.executed !== false || receipt.publicationQualified !== false || receipt.revision !== acppRevision ||
      receipt.targetLlvmGeneration !== llvm.admission.generationId || saved.version !== 1 || saved.role !== "windows-acpp-static-host" ||
      !saved.input || !isDeepStrictEqual(Object.keys(saved.input).sort(), buildInputKeys) ||
      saved.input?.output !== output || saved.input?.targetLlvm !== llvm.receipt.plan.output ||
      saved.targetLlvmGeneration !== receipt.targetLlvmGeneration || saved.targetReceiptSha256 !== receipt.targetReceiptSha256 ||
      saved.targetAdmissionSha256 !== receipt.targetAdmissionSha256 || saved.targetObjectProofSha256 !== receipt.targetObjectProofSha256 ||
      !isDeepStrictEqual(saved.recipe, receipt.recipe) || !isDeepStrictEqual(saved.wcharEvidence, receipt.wcharEvidence) ||
      !isDeepStrictEqual(saved.compilerEvidence, receipt.compilerEvidence) ||
      saved.rocmResourceReceiptSha256 !== receipt.rocmResourceReceiptSha256 || saved.rocmResourceRevision !== receipt.rocmResourceRevision ||
      saved.rocmResourceClrRevision !== receipt.rocmResourceClrRevision ||
      saved.admitted?.revision !== receipt.revision || !isDeepStrictEqual(receipt.targets, acppTargets) ||
      !Array.isArray(receipt.artifacts) || artifacts.length !== acppArchiveCount ||
      !isDeepStrictEqual(artifacts.map(row => path.relative(build, row.path).split(path.sep).join("/")), acppArtifacts) ||
      !isDeepStrictEqual(saved.artifacts, acppArtifacts.map(name => path.join(build, name))) ||
      !isDeepStrictEqual(saved.forbiddenOutputs, acppForbiddenOutputs.map(name => path.join(build, name)))) {
    throw new Error("Windows AdaptiveCpp result is incomplete or does not match its frozen build plan");
  }
  const llvmPlan = llvm.receipt.plan, descriptor = readBuildJson(saved.input.descriptorFile),
    plan = windowsAcppStaticHostPlan({...saved.input, descriptor, tools: llvmPlan.tools}, windowsHostComponentLlvmLayout(llvm));
  if (!isDeepStrictEqual(saved.input.descriptor, descriptor) || !isDeepStrictEqual(saved.configure, plan.configure) ||
      !isDeepStrictEqual(saved.compile, plan.compile) || !isDeepStrictEqual(saved.artifacts, plan.artifacts) ||
      !isDeepStrictEqual(saved.forbiddenOutputs, plan.forbiddenOutputs)) {
    throw new Error("Windows AdaptiveCpp frozen plan does not use the admitted LLVM source/build layout");
  }
  const targetReceipt = path.join(llvm.receipt.plan.output, "result.json"), targetAdmission = path.join(llvm.receipt.plan.output, "admitted.json"),
    targetObjects = path.join(llvm.receipt.plan.output, "admitted-objects.json");
  if (receipt.targetReceiptSha256 !== fileHash(targetReceipt) || receipt.targetAdmissionSha256 !== fileHash(targetAdmission) ||
      receipt.targetObjectProofSha256 !== fileHash(targetObjects)) {throw new Error("Windows AdaptiveCpp target LLVM evidence changed");}
  verifySourceInputs(receipt.recipe, sourceInputs(repo, [], [...acppRecipeFiles]), "Windows AdaptiveCpp recipe");
  const source = saved.input.source;
  if (!absolute(source) || !saved.fingerprints || !isDeepStrictEqual(saved.fingerprints, Object.fromEntries(Object.entries(receipt.fingerprints)
    .filter(([file]) => Object.hasOwn(saved.fingerprints || {}, file))))) {
    throw new Error("Windows AdaptiveCpp source or input provenance is incomplete");
  }
  const admitted = admitSyclSource(source, output, true, readOnlyGit);
  if (admitted.revision !== receipt.revision) {throw new Error("Windows AdaptiveCpp source revision changed");}
  const compilerPaths = Object.keys(receipt.compilerEvidence?.inputs || {});
  const compilerVersion = receipt.compilerEvidence?.versions?.[compilerPaths[0] || ""];
  if (!receipt.hostCompilers || compilerPaths.length !== 1 || receipt.hostCompilers.cc !== compilerPaths[0] ||
      receipt.hostCompilers.cxx !== compilerPaths[0] || typeof compilerVersion !== "string" ||
      !/^Intel\(R\)/.test(compilerVersion) || !/\bTarget: x86_64-pc-windows-msvc\b/.test(compilerVersion)) {
    throw new Error("Windows AdaptiveCpp compiler identity is incomplete or not Windows-targeted");
  }
  for (const map of [receipt.fingerprints, receipt.configuredFiles, receipt.rocmDeviceLibraries]) {verifyHashMap(map, "AdaptiveCpp input");}
  const runtimeInputs = Object.fromEntries(Object.entries(runtimeInputPaths).map(([name, relative]) => {
    const file = saved.input[name], expectedName = path.basename(relative), sha256 = receipt.fingerprints?.[file];
    if (!absolute(file) || path.basename(file) !== expectedName || !hashPattern.test(String(sha256)) ||
        saved.fingerprints?.[file] !== sha256) {
      throw new Error("Windows AdaptiveCpp producer does not bind its selected runtime input: " + name);
    }
    return [name, archiveRecord(file, sha256, fs.statSync(file).size, "AdaptiveCpp runtime input")];
  }));
  const rocm = verifyWindowsRocmResources(saved.input.rocmReceipt, saved.input.rocmDeviceLibs);
  if (rocm.sha256 !== receipt.rocmResourceReceiptSha256 || rocm.revision !== receipt.rocmResourceRevision ||
      rocm.clrRevision !== receipt.rocmResourceClrRevision || !isDeepStrictEqual(rocm.device, receipt.rocmDeviceLibraries)) {
    throw new Error("Windows AdaptiveCpp ROCm device-library receipt changed");
  }
  for (const file of saved.forbiddenOutputs || []) {
    if (fs.existsSync(file)) {throw new Error("Windows AdaptiveCpp output contains a forbidden tool or DLL: " + file);}
  }
  const compileFile = path.join(build, "compile_commands.json"),
    commands = auditAcppCompileCommands(compileFile, source, build, receipt.hostCompilers?.cxx);
  if (!isDeepStrictEqual(receipt.compileCommands, commands) || commands.path !== receipt.compileCommands?.path) {
    throw new Error("Windows AdaptiveCpp compile database changed after its receipt");
  }
  for (const [index, item] of artifacts.entries()) {
    const expected = acppArtifacts[index];
    if (!expected || path.relative(build, item.path).split(path.sep).join("/") !== expected ||
        !Array.isArray(item.objects) || item.objects.length !== item.members || item.members < 1) {
      throw new Error("Windows AdaptiveCpp archive membership is incomplete");
    }
    const audited = auditCoffArchiveInputs(item.path, build, item.objects.map((row) => row.path));
    if (audited.archiveSha256 !== item.sha256 || audited.members !== item.members || fs.statSync(item.path).size !== item.bytes ||
        audited.objects.some((row, objectIndex) => row.sha256 !== item.objects[objectIndex]?.sha256 || row.bytes !== item.objects[objectIndex]?.bytes)) {
      throw new Error("Windows AdaptiveCpp archive members differ from their object receipt: " + item.path);
    }
  }
  if (fileHash(resultFile) !== receiptSha256) {throw new Error("Windows AdaptiveCpp result receipt changed during admission");}
  return {receipt, receiptFile: resultFile, receiptSha256, output,
    archives: artifacts.map(item => archiveRecord(item.path, item.sha256, item.bytes, "AdaptiveCpp")), runtimeInputs};
}

/** Admit the producer's exact static HIPRTC closure before transporting its archive.
 * @param {string} resultFile @param {ReturnType<typeof verifyComgrResult>} comgr
 * @param {ReturnType<typeof verifyAcppResult>} acpp */
function verifyHiprtcResult(resultFile, comgr, acpp) {
  canonicalFile(resultFile, "HIPRTC result receipt");
  const receiptSha256 = fileHash(resultFile), receipt = readBuildJson(resultFile), output = path.dirname(resultFile), rows = receipt.sources;
  if (receipt.version !== 1 || receipt.role !== "windows-hiprtc-static-host" || receipt.targetTriple !== targetTriple ||
      receipt.sourceRevision !== hiprtcSourceRevision || receipt.comgrRevision !== comgrRevision || receipt.buildOnly !== true ||
      receipt.compiled !== true || receipt.executed !== false || receipt.publicationQualified !== false || !Array.isArray(rows) ||
      rows.length !== hiprtcSourceFiles.length || receipt.archive !== path.join(output, hiprtcArchiveName) ||
      !isDeepStrictEqual(receipt.archiveMembers, hiprtcArchiveMembers)) {
    throw new Error("Windows HIPRTC result is incomplete or has the wrong static source closure");
  }
  verifySourceInputs(receipt.recipe, sourceInputs(repo, [], [...hiprtcRecipeFiles]), "Windows HIPRTC recipe");
  const firstSource = rows[0]?.source, firstRelative = hiprtcSourceFiles[0];
  if (!absolute(firstSource) || !firstRelative) {throw new Error("Windows HIPRTC source root is invalid");}
  const sourceRoot = path.resolve(firstSource, ...firstRelative.split("/").map(() => ".."));
  if (fs.realpathSync(sourceRoot) !== sourceRoot) {throw new Error("Windows HIPRTC source root is not canonical");}
  verifySourceInputs(receipt.sourceState, sourceInputs(sourceRoot, ["hipamd", "rocclr"], []), "Windows HIPRTC");
  verifyHashMap(receipt.inputs, "HIPRTC input"); verifyHashMap(receipt.dependencies, "HIPRTC dependency");
  const descriptor = receipt.descriptor, builtins = receipt.builtins, producerComgr = receipt.comgr;
  if (!descriptor || descriptor.path !== comgr.receipt.plan.descriptorFile || descriptor.sha256 !== fileHash(descriptor.path) ||
      !builtins || builtins.kind !== "original-upstream-hiprtc-generated-object" || !absolute(builtins.receipt) ||
      fileHash(builtins.receipt) !== builtins.receiptSha256 || builtins.receiptSha256 !== acpp.receipt.rocmResourceReceiptSha256 ||
      !absolute(builtins.preprocessedHeader) || fileHash(builtins.preprocessedHeader) !== builtins.preprocessedHeaderSha256 ||
      !producerComgr || producerComgr.receipt !== comgr.receiptFile || producerComgr.receiptSha256 !== comgr.receiptSha256 ||
      producerComgr.header !== comgr.receipt.header ||
      producerComgr.headerSha256 !== comgr.receipt.headerSha256 || producerComgr.staticImportOverride !== "AMD_COMGR_IMPORT_DECORATOR=") {
    throw new Error("Windows HIPRTC receipt is not bound to the selected COMGR, descriptor and ROCm resources");
  }
  const deviceReceipt = readBuildJson(builtins.receipt), resource = verifyWindowsDeviceResources(builtins.receipt,
      path.join(path.dirname(builtins.receipt), "install")), descriptorConfig = readBuildJson(descriptor.path);
  if (resource.sha256 !== builtins.receiptSha256 || resource.revision !== acpp.receipt.rocmResourceRevision ||
      resource.clrRevision !== acpp.receipt.rocmResourceClrRevision || builtins.preprocessedHeader !== deviceReceipt.header?.path ||
      builtins.preprocessedHeaderSha256 !== deviceReceipt.header?.sha256 || builtins.sha256 !== deviceReceipt.object?.sha256 ||
      !receipt.compilerIdentity?.version || !isDeepStrictEqual(receipt.compilerIdentity, verifyIntelCompilerIdentity(receipt.compilerIdentity.version)) ||
      receipt.compiler !== descriptorConfig.tools?.clangCl || receipt.compiler !== receipt.toolPaths?.compiler ||
      receipt.compilerSha256 !== receipt.inputs?.[receipt.compiler] || receipt.archiver?.path !== receipt.toolPaths?.llvmAr ||
      receipt.archiver?.sha256 !== receipt.inputs?.[receipt.archiver?.path] || receipt.symbolAuditor?.path !== receipt.toolPaths?.llvmNm ||
      receipt.symbolAuditor?.path !== descriptorConfig.tools?.llvmNm || receipt.symbolAuditor?.sha256 !== receipt.inputs?.[receipt.symbolAuditor?.path] ||
      receipt.archiver?.version !== verifyLlvmToolIdentity(receipt.archiver?.text, "librarian").version ||
      receipt.symbolAuditor?.version !== verifyLlvmToolIdentity(receipt.symbolAuditor?.text, "symbol auditor").version) {
    throw new Error("Windows HIPRTC compiler, archive tools or generated resource inputs changed");
  }
  const expectedObjects = [];
  for (let index = 0; index < rows.length; index++) {
    const name = hiprtcSourceFiles[index] || "", row = rows[index], object = path.join(output, path.basename(name, ".cpp") + ".obj");
    if (!row || row.source !== path.join(sourceRoot, name) || row.object !== object || !hashPattern.test(row.sha256 || "") ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 20) {
      throw new Error("Windows HIPRTC source/object receipt differs from its pinned translation units");
    }
    canonicalFile(row.source, "HIPRTC source file");
    expectedObjects.push({path: path.basename(object), sha256: row.sha256, bytes: row.bytes});
  }
  if (builtins.object !== path.join(output, "hipRTC_header.obj") || !hashPattern.test(builtins.sha256 || "")) {
    throw new Error("Windows HIPRTC generated builtins object is not in the archive output");
  }
  const builtinsBytes = fs.statSync(builtins.object).size;
  expectedObjects.push({path: "hipRTC_header.obj", sha256: builtins.sha256, bytes: builtinsBytes});
  const audited = auditCoffArchiveInputs(receipt.archive, output, expectedObjects.map(row => row.path));
  if (audited.archiveSha256 !== receipt.archiveSha256 || receipt.archiveBytes !== fs.statSync(receipt.archive).size ||
      !isDeepStrictEqual(audited.objects, expectedObjects) ||
      !isDeepStrictEqual(auditHiprtcSymbolClosure(readBuildText(path.join(output, "symbols-undefined.log")),
        readBuildText(path.join(output, "symbols-defined.log"))), receipt.symbols)) {
    throw new Error("Windows HIPRTC archive members or audited symbol closure changed");
  }
  if (fileHash(resultFile) !== receiptSha256) {throw new Error("Windows HIPRTC producer receipt changed during admission");}
  return {receipt, receiptFile: resultFile, receiptSha256,
    archive: archiveRecord(receipt.archive, receipt.archiveSha256, receipt.archiveBytes, "HIPRTC")};
}

/** @param {{sha256: string, bytes: number}} producer @param {string} source @param {string} group @param {number} index
 * @param {string} output @param {Record<string, {sha256: string, bytes: number}>} files */
function copyArchive(producer, source, group, index, output, files) {
  const relative = "lib/" + group + "/" + String(index).padStart(3, "0") + "-" + path.basename(source);
  const destination = path.join(output, relative), expected = archiveRecord(source, producer.sha256, producer.bytes, group);
  fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
  canonicalFile(destination, "staged archive");
  if (fs.statSync(destination).size !== expected.bytes || fileHash(destination) !== expected.sha256 ||
      fileHash(source) !== expected.sha256) {throw new Error("Windows host component archive changed while staging: " + source);}
  files[relative] = {sha256: expected.sha256, bytes: expected.bytes};
  return relative;
}

/** Copy a producer-fingerprinted vendor runtime to its fixed role path. @param {{file:string,sha256:string,bytes:number}} producer
 * @param {string} relative @param {string} output @param {Record<string,{sha256:string,bytes:number}>} files
 * @returns {{path:string,sha256:string,bytes:number}} */
function copyRuntimeInput(producer, relative, output, files) {
  const destination = path.join(output, ...relative.split("/"));
  if (!Object.values(runtimeInputPaths).some(file => file === relative) || !contained(output, destination)) {
    throw new Error("Invalid Windows host runtime input path");
  }
  archiveRecord(producer.file, producer.sha256, producer.bytes, "AdaptiveCpp runtime input");
  fs.copyFileSync(producer.file, destination, fs.constants.COPYFILE_EXCL);
  canonicalFile(destination, "staged runtime input");
  if (fs.statSync(destination).size !== producer.bytes || fileHash(destination) !== producer.sha256 ||
      fileHash(producer.file) !== producer.sha256 || !isStaticArchive(destination)) {
    throw new Error("Windows host runtime input changed while staging: " + path.basename(destination));
  }
  const proof = {path: relative, sha256: producer.sha256, bytes: producer.bytes};
  files[relative] = {sha256: proof.sha256, bytes: proof.bytes};
  return proof;
}

/** Re-admit every original producer without copying, executing, or restamping it.
 * Local warm builds and fresh CI exports share this exact source/object verification.
 * @param {{llvmOutput: string, comgrResult: string, acppResult: string, hiprtcResult: string}} input
 * @param {NodeJS.ProcessEnv} [environment] */
function verifyWindowsHostComponentProducers(input, environment = process.env) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1" || !input ||
      Object.keys(input).sort().join(",") !== "acppResult,comgrResult,hiprtcResult,llvmOutput" ||
      !Object.values(input).every(absolute)) {
    throw new Error("Windows host component export requires Linuxx64, MOM_AOT_BUILD_ONLY=1 and explicit roots");
  }
  const llvmOutput = input.llvmOutput, llvm = verifyWindowsLlvmAdmission(llvmOutput, environment);
  if (llvm.admission.archives.length !== llvmArchiveCount || path.basename(llvm.receipt.plan.archive) !== llvmArchiveName ||
      fileHash(llvm.receipt.plan.archive) !== archiveSha256) {throw new Error("Windows LLVM admission does not match the reviewed 21.1.8 target closure");}
  const comgr = verifyComgrResult(input.comgrResult, llvm), acpp = verifyAcppResult(input.acppResult, llvm),
    hiprtc = verifyHiprtcResult(input.hiprtcResult, comgr, acpp);
  if (llvm.admission.archives.length !== llvmArchiveCount || acpp.archives.length !== acppArchiveCount || !hiprtc.archive) {
    throw new Error("Windows host component archive counts differ from the reviewed closures");
  }
  const llvmArchives = llvm.admission.archives.map(row => archiveRecord(row.artifact, row.sha256, row.bytes, "LLVM"));
  const sources = /** @type {{acpp: Array<{file: string, sha256: string, bytes: number}>,
    hiprtc: Array<{file: string, sha256: string, bytes: number}>, comgr: Array<{file: string, sha256: string, bytes: number}>,
    llvm: Array<{file: string, sha256: string, bytes: number}>}} */
    ({acpp: acpp.archives, hiprtc: [hiprtc.archive], comgr: [comgr.archive], llvm: llvmArchives});
  if (sources.acpp.length !== acppArchiveCount || sources.hiprtc.length !== 1 ||
      sources.comgr.length !== comgrArchiveCount || sources.llvm.length !== llvmArchiveCount) {
    throw new Error("Windows host component has an incomplete archive group");
  }
  const provenance = windowsHostProducerProvenance(llvm, comgr, acpp, hiprtc),
    files = Object.fromEntries([llvm.receiptFile, llvm.admissionFile, llvm.objectsFile, comgr.receiptFile, acpp.receiptFile, hiprtc.receiptFile,
      ...Object.values(sources).flat().map(row => row.file), ...Object.values(acpp.runtimeInputs).map(row => row.file)]
      .map(file => [file, fileHash(file)]));
  return {llvm, comgr, acpp, hiprtc, sources, provenance, files};
}

/** Keep the producer identity shared by fresh exports and warm component matching.
 * @param {ReturnType<typeof verifyWindowsLlvmAdmission>} llvm @param {ReturnType<typeof verifyComgrResult>} comgr
 * @param {ReturnType<typeof verifyAcppResult>} acpp @param {ReturnType<typeof verifyHiprtcResult>} hiprtc */
function windowsHostProducerProvenance(llvm, comgr, acpp, hiprtc) {
  return {llvm: {generationId: llvm.admission.generationId, receiptSha256: fileHash(llvm.receiptFile),
    admissionSha256: fileHash(llvm.admissionFile), objectProofSha256: fileHash(llvm.objectsFile)},
  comgr: {receiptSha256: comgr.receiptSha256, archiveSha256: comgr.archive.sha256, llvmGenerationId: llvm.admission.generationId},
  acpp: {receiptSha256: acpp.receiptSha256, sourceRevision: acppRevision, llvmGenerationId: llvm.admission.generationId,
    rocmResourceReceiptSha256: acpp.receipt.rocmResourceReceiptSha256},
  hiprtc: {receiptSha256: hiprtc.receiptSha256, archiveSha256: hiprtc.archive.sha256, sourceRevision: hiprtcSourceRevision,
    comgrRevision, comgrReceiptSha256: hiprtc.receipt.comgr.receiptSha256, comgrArchiveSha256: comgr.archive.sha256,
    resourceReceiptSha256: hiprtc.receipt.builtins.receiptSha256, archiveMembers: hiprtcArchiveMembers}};
}

/** Join a warm transport component to freshly admitted original producers, without restamping it.
 * @param {string} root @param {ReturnType<typeof verifyWindowsHostComponentProducers>} original */
function matchWindowsHostComponent(root, original) {
  const selected = windowsHostComponentInputs(root), manifest = readBuildJson(path.join(root, "component.json"));
  if (!isDeepStrictEqual(selected.provenance, original.provenance)) {
    throw new Error("Windows host component differs from its original producer provenance");
  }
  for (const group of /** @type {Array<keyof typeof original.sources>} */ (groups)) {
    const rows = original.sources[group], expected = rows.map((row, index) =>
      "lib/" + group + "/" + String(index).padStart(3, "0") + "-" + path.basename(row.file));
    if (!isDeepStrictEqual(manifest.link[group], expected) || rows.some((row, index) => {
      const relative = expected[index];
      return !relative || selected.files[path.join(root, relative)] !== row.sha256 || manifest.files[relative]?.bytes !== row.bytes;
    })) {throw new Error("Windows host component archive order or payload differs from its original producer: " + group);}
  }
  for (const [name, relative] of Object.entries(runtimeInputPaths)) {
    const row = original.acpp.runtimeInputs[name], installed = path.join(root, relative);
    if (!row || selected.runtimeInputs[name] !== installed || selected.files[installed] !== row.sha256 ||
        manifest.files[relative]?.bytes !== row.bytes) {
      throw new Error("Windows host component runtime differs from its original producer: " + name);
    }
  }
  verifyHashMap(original.files, "original producer");
  return selected;
}

/** Verify and stage the actual producer archives into a relocatable component.
 * @param {{llvmOutput: string, comgrResult: string, acppResult: string, hiprtcResult: string, output: string}} input
 * @param {NodeJS.ProcessEnv} [environment] */
function exportWindowsHostComponent(input, environment = process.env) {
  if (!input || Object.keys(input).sort().join(",") !== "acppResult,comgrResult,hiprtcResult,llvmOutput,output" || !absolute(input.output)) {
    throw new Error("Windows host component export requires Linuxx64, MOM_AOT_BUILD_ONLY=1 and explicit roots");
  }
  const {output, ...producerInput} = input, original = verifyWindowsHostComponentProducers(producerInput, environment),
    {llvm, comgr, acpp, hiprtc, sources, provenance} = original;
  requireDisjointOutput(output, [input.llvmOutput, path.dirname(input.comgrResult), acpp.output, path.dirname(input.hiprtcResult),
    ...Object.values(acpp.runtimeInputs).map(row => row.file)]);
  fs.mkdirSync(input.output, {mode: 0o700}); fs.mkdirSync(path.join(input.output, "lib")); fs.mkdirSync(path.join(input.output, "runtime"));
  for (const group of groups) {fs.mkdirSync(path.join(input.output, "lib", group));}
  const files = /** @type {Record<string, {sha256: string, bytes: number}>} */ ({}),
    link = /** @type {Record<string, string[]>} */ ({});
  for (const group of groups) {
    const rows = group === "acpp" ? sources.acpp : group === "hiprtc" ? sources.hiprtc : group === "comgr" ? sources.comgr : sources.llvm;
    link[group] = rows.map((row, index) => copyArchive(row, row.file, group, index, input.output, files));
  }
  const runtimeInputs = Object.fromEntries(Object.entries(runtimeInputPaths).map(([name, relative]) => {
    const producer = acpp.runtimeInputs[name];
    if (!producer) {throw new Error("Windows AdaptiveCpp producer runtime input is missing: " + name);}
    return [name, copyRuntimeInput(producer, relative, input.output, files)];
  }));
  const manifest = {version: 3, kind: "windows-native-host-sdk", pins: componentPins, link, runtimeInputs, files, provenance,
    buildOnly: true, publicationQualified: false};
  fs.writeFileSync(path.join(input.output, "component.json"), JSON.stringify(manifest, null, 2) + "\n", {flag: "wx", mode: 0o600});
  if (fileHash(comgr.receiptFile) !== comgr.receiptSha256 || fileHash(acpp.receiptFile) !== acpp.receiptSha256 ||
      fileHash(hiprtc.receiptFile) !== hiprtc.receiptSha256 ||
      Object.values(acpp.runtimeInputs).some(row => fileHash(row.file) !== row.sha256) ||
      fileHash(llvm.receiptFile) !== provenance.llvm.receiptSha256 || fileHash(llvm.admissionFile) !== provenance.llvm.admissionSha256 ||
      fileHash(llvm.objectsFile) !== provenance.llvm.objectProofSha256) {
    throw new Error("Windows host producer receipt changed during export");
  }
  return matchWindowsHostComponent(input.output, original);
}

/** @param {unknown} value @param {string} label */
function validRelative(value, label) {
  if (typeof value !== "string" || value.length > 4096 || value.startsWith("/") || /[\\:\0\r\n]/.test(value) ||
      value.split("/").some(part => !part || part === "." || part === "..") || path.posix.normalize(value) !== value) {
    throw new Error("Invalid Windows host component " + label);
  }
  return value;
}

/** Rehash the exact bounded relative host archive closure on any build host.
 * @param {string} root */
function windowsHostComponentInputs(root) {
  if (!absolute(root) || fs.realpathSync(root) !== root) {throw new Error("Select a canonical Windows host component root");}
  const manifestFile = path.join(root, "component.json"); canonicalFile(manifestFile, "component receipt");
  const manifestSha256 = fileHash(manifestFile), manifest = readBuildJson(manifestFile);
  if (manifest.version !== 3 || manifest.kind !== "windows-native-host-sdk" || !isDeepStrictEqual(manifest.pins, componentPins) ||
      manifest.buildOnly !== true || manifest.publicationQualified !== false || !manifest.link || !manifest.files ||
      typeof manifest.files !== "object" || Array.isArray(manifest.files) || !manifest.provenance ||
      !isDeepStrictEqual(Object.keys(manifest).sort(), ["buildOnly", "files", "kind", "link", "pins", "provenance", "publicationQualified", "runtimeInputs", "version"].sort()) ||
      !manifest.runtimeInputs || typeof manifest.runtimeInputs !== "object" || Array.isArray(manifest.runtimeInputs) ||
      !isDeepStrictEqual(Object.keys(manifest.runtimeInputs).sort(), Object.keys(runtimeInputPaths).sort()) ||
      !isDeepStrictEqual(Object.keys(manifest.link).sort(), [...groups].sort())) {
    throw new Error("Invalid Windows host component receipt");
  }
  const provenance = manifest.provenance, llvm = provenance.llvm, comgr = provenance.comgr, acpp = provenance.acpp, hiprtc = provenance.hiprtc;
  if (!llvm || !uuidPattern.test(llvm.generationId) || !hashPattern.test(llvm.receiptSha256) || !hashPattern.test(llvm.admissionSha256) ||
      !hashPattern.test(llvm.objectProofSha256) || !comgr || !hashPattern.test(comgr.receiptSha256) ||
      !hashPattern.test(comgr.archiveSha256) || comgr.llvmGenerationId !== llvm.generationId || !acpp ||
      !hashPattern.test(acpp.receiptSha256) || acpp.sourceRevision !== acppRevision || acpp.llvmGenerationId !== llvm.generationId ||
      !hashPattern.test(acpp.rocmResourceReceiptSha256) || !hiprtc || !hashPattern.test(hiprtc.receiptSha256) ||
      !hashPattern.test(hiprtc.archiveSha256) || hiprtc.sourceRevision !== hiprtcSourceRevision || hiprtc.comgrRevision !== comgrRevision ||
      hiprtc.comgrReceiptSha256 !== comgr.receiptSha256 || hiprtc.comgrArchiveSha256 !== comgr.archiveSha256 ||
      hiprtc.resourceReceiptSha256 !== acpp.rocmResourceReceiptSha256 || !isDeepStrictEqual(hiprtc.archiveMembers, hiprtcArchiveMembers) ||
      !isDeepStrictEqual(Object.keys(provenance).sort(), ["acpp", "comgr", "hiprtc", "llvm"]) ||
      !isDeepStrictEqual(Object.keys(llvm).sort(), ["admissionSha256", "generationId", "objectProofSha256", "receiptSha256"]) ||
      !isDeepStrictEqual(Object.keys(comgr).sort(), ["archiveSha256", "llvmGenerationId", "receiptSha256"]) ||
      !isDeepStrictEqual(Object.keys(acpp).sort(), ["llvmGenerationId", "receiptSha256", "rocmResourceReceiptSha256", "sourceRevision"]) ||
      !isDeepStrictEqual(Object.keys(hiprtc).sort(), ["archiveMembers", "archiveSha256", "comgrArchiveSha256", "comgrReceiptSha256",
        "comgrRevision", "receiptSha256", "resourceReceiptSha256", "sourceRevision"]) ||
      !isDeepStrictEqual(Object.keys(manifest.link).sort(), [...groups].sort())) {
    throw new Error("Windows host component producer provenance is incomplete");
  }
  const expectedFiles = /** @type {Record<string, {sha256: string, bytes: number}>} */ ({}),
    libraries = /** @type {string[]} */ ([]), runtimeInputs = /** @type {Record<string,string>} */ ({});
  let totalBytes = 0;
  for (const group of groups) {
    const rows = manifest.link[group];
    const expectedCount = group === "llvm" ? llvmArchiveCount : group === "comgr" ? comgrArchiveCount : group === "hiprtc" ? 1 : acppArchiveCount;
    if (!Array.isArray(rows) || rows.length !== expectedCount) {throw new Error("Incomplete Windows host component " + group + " archive group");}
    for (const name of rows) {
      const relative = validRelative(name, "archive path");
      if (!relative.startsWith("lib/" + group + "/") || !archiveName.test(path.basename(relative)) || Object.hasOwn(expectedFiles, relative)) {
        throw new Error("Invalid or duplicate Windows host component archive path");
      }
      if (group === "hiprtc" && relative !== "lib/hiprtc/000-" + hiprtcArchiveName) {
        throw new Error("Windows HIPRTC component archive differs from its producer provenance");
      }
      const proof = manifest.files[relative];
      if (!proof || Object.keys(proof).sort().join(",") !== "bytes,sha256" || !hashPattern.test(proof.sha256) ||
          !Number.isSafeInteger(proof.bytes) || proof.bytes < 8 || proof.bytes > archiveBytesLimit ||
          (totalBytes += proof.bytes) > componentBytesLimit) {throw new Error("Invalid or oversized Windows host component archive proof");}
      if (group === "hiprtc" && proof.sha256 !== hiprtc.archiveSha256) {
        throw new Error("Windows HIPRTC component archive differs from its producer provenance");
      }
      expectedFiles[relative] = proof; libraries.push(path.join(root, relative));
    }
  }
  for (const [name, relative] of Object.entries(runtimeInputPaths)) {
    const proof = manifest.runtimeInputs[name], fileProof = manifest.files[relative];
    if (!proof || !isDeepStrictEqual(Object.keys(proof).sort(), ["bytes", "path", "sha256"]) || proof.path !== relative ||
        !hashPattern.test(proof.sha256) || !Number.isSafeInteger(proof.bytes) || proof.bytes < 8 || proof.bytes > archiveBytesLimit ||
        (totalBytes += proof.bytes) > componentBytesLimit || Object.hasOwn(expectedFiles, relative) ||
        !fileProof || !isDeepStrictEqual(fileProof, {sha256: proof.sha256, bytes: proof.bytes})) {
      throw new Error("Invalid Windows host component runtime input proof: " + name);
    }
    expectedFiles[relative] = {sha256: proof.sha256, bytes: proof.bytes};
    runtimeInputs[name] = path.join(root, ...relative.split("/"));
    if (name === "cudartStatic") {libraries.push(runtimeInputs[name] || "");}
  }
  if (!isDeepStrictEqual(Object.keys(manifest.files).sort(), Object.keys(expectedFiles).sort())) {
    throw new Error("Windows host component file membership differs from its archive groups");
  }
  const files = /** @type {Record<string, string>} */ ({[manifestFile]: manifestSha256});
  for (const [relative, proof] of Object.entries(expectedFiles)) {
    const file = path.join(root, relative);
    canonicalFile(file, "archive");
    if (!contained(root, file) || fs.statSync(file).size !== proof.bytes || !isStaticArchive(file) || fileHash(file) !== proof.sha256) {
      throw new Error("Windows host component archive changed: " + relative);
    }
    files[file] = proof.sha256;
  }
  const expectedDirectories = ["lib", ...groups.map(group => "lib/" + group), "runtime"];
  if (!isDeepStrictEqual(fs.readdirSync(root).sort(), ["component.json", "lib", "runtime"]) ||
      !isDeepStrictEqual(fs.readdirSync(path.join(root, "lib")).sort(), [...groups].sort())) {
    throw new Error("Unrecorded Windows host component entry");
  }
  if (!isDeepStrictEqual(fs.readdirSync(path.join(root, "runtime")).sort(), Object.values(runtimeInputPaths).map(relative => path.basename(relative)).sort())) {
    throw new Error("Unrecorded Windows host runtime input");
  }
  for (const group of groups) {
    const relativeNames = /** @type {string[]} */ (manifest.link[group]);
    const directory = path.join(root, "lib", group), names = relativeNames.map(name => path.basename(name)).sort();
    if (!isDeepStrictEqual(fs.readdirSync(directory).sort(), names)) {throw new Error("Unrecorded Windows host component archive");}
  }
  for (const relative of expectedDirectories) {
    const directory = path.join(root, relative), stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || fs.realpathSync(directory) !== directory) {throw new Error("Windows host component directory is not canonical");}
  }
  if (fileHash(manifestFile) !== manifestSha256) {throw new Error("Windows host component receipt changed during admission");}
  return {link: {libraries}, runtimeInputs, files, provenance};
}

module.exports = {exportWindowsHostComponent, verifyWindowsHostComponentProducers, matchWindowsHostComponent,
  windowsHostComponentInputs, windowsHostComponentLlvmLayout, copyRuntimeInput};

if (require.main === module) {
  try {
    const [llvmOutput, comgrResult, acppResult, hiprtcResult, output, extra] = process.argv.slice(2);
    if (!llvmOutput || !comgrResult || !acppResult || !hiprtcResult || !output || extra) {
      throw new Error("Usage: native-windows-host-component.js /llvm-output /comgr/result.json /acpp/result.json /hiprtc/result.json /new/component");
    }
    const component = exportWindowsHostComponent({llvmOutput, comgrResult, acppResult, hiprtcResult, output});
    console.log("[native-sdk] Exported " + component.link.libraries.length + " Windows host archives; not runtime-qualified.");
  } catch (error) {console.error(error instanceof Error ? error.message : "Windows host component export failed"); process.exitCode = 1;}
}
