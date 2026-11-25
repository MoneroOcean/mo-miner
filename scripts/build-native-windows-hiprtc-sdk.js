"use strict";

// Build the original Windows HIPRTC host source closure and its upstream-generated RTC header.
// This is a static source component only; it neither installs nor executes HIPRTC.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {validateWindowsLlvmSdkConfig, windowsToolEnvironment, vfsFiles, dependencyHashes} = require("./build-native-windows-controller");
const {revision: comgrRevision, verifyWindowsDeviceResources} = require("./build-native-windows-comgr-sdk");
const {hasExactBuildFields, buildPathsOverlap, fileHash, buildFileInputs, readBuildJson, readBuildJsonSnapshot, readBuildText, sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {auditCoffArchiveInputs} = require("./native-coff-archive");
const {isStaticArchive} = require("./native-audit");

const repo = path.resolve(__dirname, "..");
const clrRevision = "5b591b2ff64aac4c1c08be879b0d5f89fd713d50";
const targetTriple = "x86_64-pc-windows-msvc";
const commandTimeout = 180000, commandOutputLimit = 4 << 20;
const sourceFiles = Object.freeze([
  "hipamd/src/hiprtc/hiprtc.cpp", "hipamd/src/hiprtc/hiprtcInternal.cpp", "hipamd/src/hip_comgr_helper.cpp",
  "rocclr/device/comgrctx.cpp", "rocclr/os/os.cpp", "rocclr/os/os_win32.cpp", "rocclr/os/alloc.cpp",
  "rocclr/thread/thread.cpp", "rocclr/utils/debug.cpp", "rocclr/utils/flags.cpp",
]);
const recipeFiles = Object.freeze(["scripts/build-native-windows-hiprtc-sdk.js", "scripts/build-native-windows-controller.js",
  "scripts/build-native-windows-comgr-sdk.js", "scripts/build-native-rocm-resources.js", "scripts/native-comgr-inputs.js",
  "scripts/build-source-inputs.js", "scripts/native-coff-archive.js", "scripts/native-compiler-isolation.js", "scripts/native-audit.js"]);
const archiveName = "hiprtc-host-intel.lib";

/** Paths are argv values, not shell fragments; still reject controls and noncanonical spellings. @param {unknown} value */
function literal(value) {
  return typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) && path.normalize(value) === value &&
    !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
}

/** @param {string} file @param {string} label */
function canonicalFile(file, label) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!literal(file) || !stat?.isFile() || fs.realpathSync(file) !== file) {throw new Error("Windows HIPRTC " + label + " must be a canonical regular file: " + file);}
  return file;
}

/** Resolve selected tool aliases once, then run and hash only their canonical targets.
 * @param {string} file @param {string} label */
function canonicalTool(file, label) {
  if (!literal(file)) {throw new Error("Windows HIPRTC " + label + " must be an absolute canonical path: " + file);}
  return canonicalFile(fs.realpathSync(file), label);
}

/** @param {string} directory @param {string} label */
function canonicalDirectory(directory, label) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!literal(directory) || !stat?.isDirectory() || fs.realpathSync(directory) !== directory) {
    throw new Error("Windows HIPRTC " + label + " must be a canonical directory: " + directory);
  }
  return directory;
}

/** Inventory one caller-selected SDK/tool include root, so added headers cannot change precedence silently.
 * @param {string} root @param {string} label */
function directoryInputs(root, label) {
  canonicalDirectory(root, label);
  const files = /** @type {string[]} */ ([]);
  /** @param {string} directory */
  function visit(directory) {
    const entries = fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const file = path.join(directory, entry.name), stat = fs.lstatSync(file);
      if (stat.isSymbolicLink()) {throw new Error("Windows HIPRTC " + label + " contains a symbolic link: " + file);}
      if (stat.isDirectory()) {visit(file);} else if (stat.isFile()) {files.push(file);} else {throw new Error("Invalid Windows HIPRTC " + label + " entry: " + file);}
      if (files.length > 65536) {throw new Error("Windows HIPRTC " + label + " inventory exceeds bounds");}
    }
  }
  visit(root);
  return buildFileInputs(files);
}

/** Admit the completed Windows COMGR producer's generated header, not a HIP SDK or host LLVM copy.
 * @param {string} resultFile @param {string} descriptorFile */
function verifyComgrHeader(resultFile, descriptorFile) {
  canonicalFile(resultFile, "COMGR result receipt");
  const {value: receipt, sha256: receiptSha256} = readBuildJsonSnapshot(resultFile), plan = receipt.plan;
  if (!plan || receipt.version !== 1 || receipt.role !== "windows-comgr-static-host" || receipt.stage !== "complete" ||
      receipt.buildOnly !== true || receipt.compiled !== true || receipt.executed !== false || receipt.publicationQualified !== false ||
      receipt.revision !== comgrRevision || plan.output !== path.dirname(resultFile) || path.join(plan.output, "result.json") !== resultFile ||
      plan.descriptorFile !== descriptorFile ||
      fileHash(path.join(plan.output, "plan.json")) !== receipt.planSha256 || receipt.fingerprints?.[descriptorFile] !== fileHash(descriptorFile)) {
    throw new Error("Select the completed pinned Windows COMGR header receipt for this Windows SDK descriptor");
  }
  const header = path.join(plan.output, "build/include/amd_comgr.h");
  canonicalFile(header, "generated COMGR header");
  if (receipt.header !== header || !/^[a-f0-9]{64}$/.test(String(receipt.headerSha256)) ||
      fs.statSync(header).size > (8 << 20) || fileHash(header) !== receipt.headerSha256 || fileHash(resultFile) !== receiptSha256) {
    throw new Error("Windows COMGR generated header differs from its producer receipt");
  }
  return {receipt, receiptFile: resultFile, receiptSha256, header, headerSha256: receipt.headerSha256};
}

/** Validate the one-stage, explicit-root recipe before creating a private output tree.
 * @param {{source: string, descriptorFile: string, hipInclude: string, comgrResult: string,
 * deviceReceipt: string, deviceInstall: string, llvmAr: string, output: string}} input */
function windowsHiprtcPlan(input) {
  const keys = ["comgrResult", "descriptorFile", "deviceInstall", "deviceReceipt", "hipInclude", "llvmAr", "output", "source"];
  if (!input || !hasExactBuildFields(input, keys) || !Object.values(input).every(literal)) {
    throw new Error("Windows HIPRTC needs explicit canonical absolute source, descriptor, HIP/COMGR/resource SDK, librarian and new output roots");
  }
  canonicalDirectory(input.source, "CLR source root"); canonicalDirectory(input.hipInclude, "official HIP include root");
  canonicalFile(input.descriptorFile, "controller descriptor");
  const parent = canonicalDirectory(path.dirname(input.output), "output parent");
  if (fs.lstatSync(input.output, {throwIfNoEntry: false}) || path.join(parent, path.basename(input.output)) !== input.output) {
    throw new Error("Select a new canonical Windows HIPRTC output directory");
  }
  const config = validateWindowsLlvmSdkConfig(readBuildJson(input.descriptorFile));
  const compiler = canonicalTool(config.tools.clangCl, "Intel clang-cl"), llvmNm = canonicalTool(config.tools.llvmNm, "LLVM nm"),
    llvmAr = canonicalTool(input.llvmAr, "LLVM librarian");
  canonicalDirectory(config.tools.clangResourceInclude, "Intel Clang resource include");
  canonicalFile(config.windowsSdk.overlay, "Windows SDK VFS overlay");
  canonicalDirectory(input.deviceInstall, "HIP device install root");
  const resourceReceipt = readBuildJson(input.deviceReceipt);
  const device = verifyWindowsDeviceResources(input.deviceReceipt, input.deviceInstall),
    deviceHeader = resourceReceipt.header, deviceObject = resourceReceipt.object;
  if (!deviceHeader || !deviceObject) {throw new Error("Windows HIPRTC needs the original generated builtins header and COFF object receipt");}
  const hipCommon = path.join(input.hipInclude, "hip/hip_common.h");
  if (!Object.hasOwn(resourceReceipt.inputs, hipCommon)) {
    throw new Error("Windows HIPRTC header object was not generated from the selected official HIP headers");
  }
  const comgr = verifyComgrHeader(input.comgrResult, input.descriptorFile);
  const includeRoots = [input.source, path.join(input.source, "rocclr"), path.join(input.source, "rocclr/elf"),
    path.join(input.source, "rocclr/include"), path.join(input.source, "hipamd/src"), path.join(input.output, "include"),
    input.hipInclude, path.join(input.source, "hipamd/include"), path.join(input.source, "opencl/khronos/headers/opencl2.2"),
    path.join(input.source, "opencl"), ...config.windowsSdk.includeRoots];
  const virtual = [...vfsFiles(config.windowsSdk.overlay)].filter(([name]) => config.windowsSdk.includeRoots.some(root =>
    name.toLowerCase().startsWith(root.replace(/\/+$/, "").toLowerCase() + "/")));
  if (!virtual.length) {throw new Error("Windows HIPRTC needs mapped Windows SDK headers");}
  const excluded = [repo, input.source, input.hipInclude, input.descriptorFile, input.comgrResult, path.dirname(input.comgrResult),
    input.deviceReceipt, path.dirname(input.deviceReceipt), input.deviceInstall, input.llvmAr, compiler, llvmNm, llvmAr, config.tools.clangCl, config.tools.llvmNm,
    config.tools.clangResourceInclude, config.windowsSdk.overlay, config.windowsSdk.sourceRoot, ...config.windowsSdk.includeRoots,
    ...config.windowsSdk.libraryRoots, ...config.windowsSdk.msvcLibraries, ...config.windowsSdk.ucrtLibraries,
    ...virtual.map(([, file]) => file), comgr.header, deviceHeader.path, deviceObject.path];
  if (excluded.some(root => buildPathsOverlap(root, input.output))) {throw new Error("Windows HIPRTC output overlaps a source, receipt, tool or SDK input");}
  const stagedComgrHeader = path.join(input.output, "include/amd_comgr/amd_comgr.h");
  const objects = [...sourceFiles.map(name => path.join(input.output, path.basename(name, ".cpp") + ".obj")),
    path.join(input.output, "hipRTC_header.obj")];
  const flags = ["--target=" + targetTriple, "/nologo", "/c", "/MT", "/EHsc", "/GR", "/std:c++17", "/O2", "/Gy", "/Gw", "/DNDEBUG",
    "/DWIN32", "/D_WIN32", "/D_WIN64", "/D__HIP_PLATFORM_AMD__", "/DWITH_LIGHTNING_COMPILER", "/DUSE_COMGR_LIBRARY", "/D__HIP_ENABLE_RTC",
    "/DAMD_COMGR_IMPORT_DECORATOR=", "-fms-compatibility-version=19.44", "-nostdinc", "-I" + config.tools.clangResourceInclude,
    "-Xclang", "-ivfsoverlay", "-Xclang", config.windowsSdk.overlay, ...includeRoots.map(root => "-I" + root)];
  return {input: {...input}, config, device, deviceHeader, deviceObject, comgr, sourceFiles: [...sourceFiles], includeRoots, virtual: virtual.map(([, file]) => file),
    compiler, llvmNm, llvmAr, toolPaths: {compiler: config.tools.clangCl, llvmNm: config.tools.llvmNm, llvmAr: input.llvmAr}, targetTriple, flags,
    nmakeDependencyArgs: ["-Xclang", "-MV"], objects,
    stagedComgrHeader, archive: path.join(input.output, archiveName)};
}

/** Require the selected Intel 2026.2 frontend, not compatible-mode Clang/GCC.
 * @param {string} version */
function verifyIntelCompilerIdentity(version) {
  if (typeof version !== "string" || version.length > 65536 ||
      !/(?:^|\r?\n)Intel\(R\) oneAPI DPC\+\+\/C\+\+ Compiler\b[^\r\n]*\bVersion 2026\.2(?:\.\d+)?(?:\s|$)/i.test(version)) {
    throw new Error("Windows HIPRTC requires the selected Intel 2026.2 clang-cl");
  }
  return {version: version.trim()};
}

/** @param {string} version @param {string} tool */
function verifyLlvmToolIdentity(version, tool) {
  if (typeof version !== "string" || version.length > 65536 || !/\bLLVM version 21\.1\.8\b/.test(version)) {
    throw new Error("Windows HIPRTC " + tool + " must be LLVM 21.1.8");
  }
  return {version: "21.1.8", text: version.trim()};
}

/** Keep one direct undefined/defined symbol comparison for the full archive; repeated per-object U rows
 * are expected and must not be mistaken for missing HIPRTC/ROCclr definitions.
 * @param {string} undefinedText @param {string} definedText */
function auditHiprtcSymbolClosure(undefinedText, definedText) {
  if (undefinedText.length > (16 << 20) || definedText.length > (64 << 20)) {throw new Error("Excessive Windows HIPRTC symbol audit output");}
  const undefinedSymbols = /** @type {Set<string>} */ (new Set()), definedSymbols = /** @type {Set<string>} */ (new Set());
  for (const line of undefinedText.split(/\r?\n/)) {const symbol = line.match(/^\s+U\s+(.+)$/)?.[1]; if (symbol) {undefinedSymbols.add(symbol);}}
  for (const line of definedText.split(/\r?\n/)) {const symbol = line.match(/^\s*[0-9a-fA-F]+\s+[A-Za-z?]\s+(.+)$/)?.[1]; if (symbol) {definedSymbols.add(symbol);}}
  if (!undefinedSymbols.size || !definedSymbols.size) {throw new Error("Windows HIPRTC symbol audit omitted undefined or defined archive symbols");}
  const unresolved = [...undefinedSymbols].filter(name => !definedSymbols.has(name)).sort();
  const comgrImports = unresolved.filter(name => /^__imp_+amd_comgr_/i.test(name));
  const internal = unresolved.filter(name => /\b(?:hiprtc|hip|amd|rocclr)::/i.test(name) || /^__hipRTC/i.test(name) ||
    /^HIPRTC_/i.test(name) || /^AMD_(?!COMGR_)/i.test(name) || /^GPU_/i.test(name));
  const comgrReferences = unresolved.filter(name => /^_?amd_comgr_[A-Za-z0-9_]+$/.test(name));
  if (comgrImports.length || internal.length || !comgrReferences.length ||
      !definedSymbols.has("__hipRTC_header") || !definedSymbols.has("__hipRTC_header_size")) {
    throw new Error("Windows HIPRTC archive has COMGR DLL imports, missing HIPRTC/ROCclr members, or no direct COMGR references");
  }
  return {undefinedCount: undefinedSymbols.size, definedCount: definedSymbols.size, unresolved,
    comgrReferences: comgrReferences.sort(), driverReferences: unresolved.filter(name => name === "hipGetDevicePropertiesR0600")};
}

/** @param {string} log @param {string} command @param {string[]} args @param {string} cwd
 * @param {NodeJS.ProcessEnv} env @param {number} [timeout] */
function runLogged(log, command, args, cwd, env, timeout = commandTimeout) {
  const result = spawnSync(command, args, {cwd, env, encoding: "utf8", timeout, maxBuffer: commandOutputLimit});
  const output = (result.stdout || "") + (result.stderr || "");
  fs.writeFileSync(log, output.slice(0, commandOutputLimit * 2) || (result.status === 0 ? "" : "No command output\n"), {flag: "wx", mode: 0o600});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error("Windows HIPRTC command failed; inspect " + log, {cause: result.error});
  }
  return {command, args: [...args], log, logSha256: fileHash(log)};
}

/** @param {string} source */
function verifyPinnedClrSource(source) {
  const env = {PATH: "/usr/bin:/bin", LC_ALL: "C", GIT_NO_LAZY_FETCH: "1", GIT_NO_REPLACE_OBJECTS: "1",
    GIT_OPTIONAL_LOCKS: "0", GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null", GIT_ATTR_NOSYSTEM: "1", GIT_TERMINAL_PROMPT: "0"};
  /** @param {string[]} args */
  function git(args) {
    const result = spawnSync("/usr/bin/git", ["--no-optional-locks", "-C", source, ...args], {encoding: "utf8", env, timeout: 30000, maxBuffer: 8 << 20});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Pinned Windows HIPRTC CLR source check failed", {cause: result.error});}
    return result.stdout || "";
  }
  if (git(["rev-parse", "HEAD"]).trim() !== clrRevision || git(["diff", "--quiet", "HEAD", "--", "hipamd", "rocclr"]) !== "" ||
      git(["status", "--porcelain=v1", "--untracked-files=all", "--", "hipamd", "rocclr"]).trim()) {
    throw new Error("Select the unchanged pinned ROCm CLR 7.1.1 HIPRTC source tree");
  }
}

/** Copy an already-admitted object without mutating its producing receipt.
 * @param {string} source @param {string} destination @param {string} sha256 */
function stageVerified(source, destination, sha256) {
  canonicalFile(source, "staged source");
  fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
  canonicalFile(destination, "staged file");
  if (fileHash(source) !== sha256 || fileHash(destination) !== sha256) {throw new Error("Windows HIPRTC staged input differs from its receipt");}
}

/** Build the original ten Windows HIPRTC/ROCclr TUs plus the original generated builtins object.
 * The one-stage output is new-only; it is not a runtime or publication qualification.
 * @param {Parameters<typeof windowsHiprtcPlan>[0]} input @param {NodeJS.ProcessEnv} [environment] */
function buildNativeWindowsHiprtcSdk(input, environment = process.env) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("Windows HIPRTC source production requires Linux x64 and MOM_AOT_BUILD_ONLY=1");
  }
  const plan = windowsHiprtcPlan(input), sourceState = sourceInputs(input.source, ["hipamd", "rocclr"], []);
  verifyPinnedClrSource(input.source);
  const git = canonicalFile("/usr/bin/git", "trusted Git executable"), hipFiles = directoryInputs(input.hipInclude, "official HIP include"),
    resourceFiles = directoryInputs(plan.config.tools.clangResourceInclude, "Intel Clang resource include"),
    virtualFiles = buildFileInputs(plan.virtual);
  const selected = [git, input.descriptorFile, input.comgrResult, plan.comgr.header, input.deviceReceipt, plan.deviceHeader.path,
    plan.deviceObject.path, plan.compiler, plan.llvmAr, plan.llvmNm, plan.config.windowsSdk.overlay,
    ...Object.keys(hipFiles), ...Object.keys(resourceFiles), ...Object.keys(virtualFiles)];
  const inputs = buildFileInputs(selected), recipe = sourceInputs(repo, [], [...recipeFiles]);
  const outputParent = fs.realpathSync(path.dirname(input.output));
  if (path.join(outputParent, path.basename(input.output)) !== input.output || fs.lstatSync(input.output, {throwIfNoEntry: false})) {
    throw new Error("Windows HIPRTC output appeared during admission; use another new output path");
  }
  fs.mkdirSync(input.output, {mode: 0o700});
  fs.mkdirSync(path.dirname(plan.stagedComgrHeader), {recursive: true, mode: 0o700});
  stageVerified(plan.comgr.header, plan.stagedComgrHeader, plan.comgr.headerSha256);
  stageVerified(plan.deviceObject.path, plan.objects[plan.objects.length - 1] || "", plan.deviceObject.sha256);
  const env = windowsToolEnvironment(environment);
  for (const key of Object.keys(env)) {
    if (/^(?:CC$|CXX$|AR$|RC$|CL$|_CL_$|LINK$|INCLUDE$|LIB$|LIBPATH$|CMAKE_|LLVM_|CLANG_|HIP|ROCM_|AMD_COMGR|COMGR_)/i.test(key)) {delete env[key];}
  }
  env["PATH"] = [path.dirname(plan.llvmAr), path.dirname(plan.compiler), "/usr/bin", "/bin"].join(path.delimiter);
  env["LC_ALL"] = "C";
  const commands = [];
  const versionFile = path.join(input.output, "intel-clang-version.log"), versionCommand = runLogged(versionFile, plan.compiler, ["--version"], repo, env, 60000),
    versionText = readBuildText(versionFile, 8 << 20);
  const compilerIdentity = verifyIntelCompilerIdentity(versionText);
  const arVersionFile = path.join(input.output, "llvm-ar-version.log"), nmVersionFile = path.join(input.output, "llvm-nm-version.log"),
    arVersionCommand = runLogged(arVersionFile, plan.llvmAr, ["--version"], repo, env, 60000),
    nmVersionCommand = runLogged(nmVersionFile, plan.llvmNm, ["--version"], repo, env, 60000),
    arIdentity = verifyLlvmToolIdentity(readBuildText(arVersionFile, 8 << 20), "librarian"),
    nmIdentity = verifyLlvmToolIdentity(readBuildText(nmVersionFile, 8 << 20), "symbol auditor");
  commands.push({probe: "intel-clang", args: ["--version"], logSha256: versionCommand.logSha256},
    {probe: "llvm-ar", args: ["--version"], logSha256: arVersionCommand.logSha256},
    {probe: "llvm-nm", args: ["--version"], logSha256: nmVersionCommand.logSha256});
  const startTime = Date.now(), depfiles = [], objects = [];
  for (let index = 0; index < sourceFiles.length; index++) {
    const name = sourceFiles[index] || "", object = plan.objects[index] || "", source = path.join(input.source, name),
      depfile = path.join(input.output, path.basename(name, ".cpp") + ".d"), log = path.join(input.output, path.basename(name, ".cpp") + ".log");
    canonicalFile(source, "original translation unit");
    const args = [...plan.flags, "-Xclang", "-dependency-file", "-Xclang", depfile, "-Xclang", "-MT", "-Xclang", object,
      ...plan.nmakeDependencyArgs, "-Xclang", "-sys-header-deps", source, "/Fo" + object];
    const command = runLogged(log, plan.compiler, args, repo, env);
    const stat = fs.statSync(object, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.size < 20) {throw new Error("Windows HIPRTC compiler omitted its COFF object: " + name);}
    depfiles.push(depfile); objects.push({source, object, sha256: fileHash(object), bytes: stat.size});
    commands.push({source, object, args, logSha256: command.logSha256});
  }
  const archiveLog = path.join(input.output, "archive.log");
  const archiveCommand = runLogged(archiveLog, plan.llvmAr, ["rcs", plan.archive, ...plan.objects], repo, env);
  commands.push({archive: plan.archive, args: archiveCommand.args, logSha256: archiveCommand.logSha256});
  if (!isStaticArchive(plan.archive)) {throw new Error("Windows HIPRTC source archive is not a regular ar archive");}
  const audited = auditCoffArchiveInputs(plan.archive, input.output, plan.objects.map(file => path.basename(file))),
    undefinedLog = path.join(input.output, "symbols-undefined.log"), definedLog = path.join(input.output, "symbols-defined.log");
  const undefinedCommand = runLogged(undefinedLog, plan.llvmNm, ["-C", "--undefined-only", plan.archive], repo, env),
    definedCommand = runLogged(definedLog, plan.llvmNm, ["-C", "--defined-only", plan.archive], repo, env),
    symbolAudit = auditHiprtcSymbolClosure(readBuildText(undefinedLog, 8 << 20), readBuildText(definedLog, 8 << 20));
  commands.push({symbols: "undefined", args: undefinedCommand.args, logSha256: undefinedCommand.logSha256},
    {symbols: "defined", args: definedCommand.args, logSha256: definedCommand.logSha256});
  const expectedObjects = [...objects.map(row => ({path: path.basename(row.object), sha256: row.sha256, bytes: row.bytes})),
    {path: "hipRTC_header.obj", sha256: plan.deviceObject.sha256, bytes: fs.statSync(plan.deviceObject.path).size}];
  if (!isDeepStrictEqual(audited.objects, expectedObjects) || fileHash(plan.archive) !== audited.archiveSha256) {
    throw new Error("Windows HIPRTC archive members changed after compilation or differ from the admitted generated object");
  }
  const discoveredDependencies = dependencyHashes(depfiles, plan.config.windowsSdk.overlay, input.output, startTime);
  for (const [file, hash] of Object.entries(discoveredDependencies)) {
    if (Object.hasOwn(inputs, file) && inputs[file] !== hash) {throw new Error("Windows HIPRTC SDK dependency changed during compilation: " + file);}
  }
  const dependencies = {...inputs, ...discoveredDependencies};
  verifyPinnedClrSource(input.source);
  verifySourceInputs(recipe, sourceInputs(repo, [], [...recipeFiles]), "Windows HIPRTC recipe");
  if (!isDeepStrictEqual(sourceState, sourceInputs(input.source, ["hipamd", "rocclr"], [])) ||
      Object.entries(inputs).some(([file, hash]) => fileHash(file) !== hash) ||
      !isDeepStrictEqual(hipFiles, directoryInputs(input.hipInclude, "official HIP include")) ||
      !isDeepStrictEqual(resourceFiles, directoryInputs(plan.config.tools.clangResourceInclude, "Intel Clang resource include")) ||
      !isDeepStrictEqual(virtualFiles, buildFileInputs(plan.virtual)) ||
      fileHash(input.comgrResult) !== plan.comgr.receiptSha256 || fileHash(plan.comgr.header) !== plan.comgr.headerSha256 ||
      fileHash(input.deviceReceipt) !== plan.device.sha256 || fileHash(plan.deviceObject.path) !== plan.deviceObject.sha256 ||
      fileHash(plan.deviceHeader.path) !== plan.deviceHeader.sha256 || fileHash(plan.stagedComgrHeader) !== plan.comgr.headerSha256 ||
      fileHash(plan.objects[plan.objects.length - 1] || "") !== plan.deviceObject.sha256 || fileHash(plan.archive) !== audited.archiveSha256) {
    throw new Error("Windows HIPRTC source or SDK inputs changed during compilation");
  }
  const deviceFinal = verifyWindowsDeviceResources(input.deviceReceipt, input.deviceInstall);
  if (deviceFinal.sha256 !== plan.device.sha256) {throw new Error("Windows HIPRTC resource receipt changed during compilation");}
  const result = {version: 1, role: "windows-hiprtc-static-host", targetTriple, sourceRevision: clrRevision,
    comgrRevision, compiler: plan.compiler, compilerIdentity, compilerSha256: inputs[plan.compiler], toolPaths: plan.toolPaths,
    archiver: {path: plan.llvmAr, sha256: inputs[plan.llvmAr], ...arIdentity}, symbolAuditor: {path: plan.llvmNm, sha256: inputs[plan.llvmNm], ...nmIdentity},
    sources: objects, builtins: {kind: "original-upstream-hiprtc-generated-object", object: plan.objects[plan.objects.length - 1],
      sha256: plan.deviceObject.sha256, receipt: input.deviceReceipt, receiptSha256: plan.device.sha256,
      preprocessedHeader: plan.deviceHeader.path, preprocessedHeaderSha256: plan.deviceHeader.sha256},
    comgr: {receipt: input.comgrResult, receiptSha256: plan.comgr.receiptSha256, header: plan.comgr.header, headerSha256: plan.comgr.headerSha256,
      staticImportOverride: "AMD_COMGR_IMPORT_DECORATOR="},
    recipe,
    descriptor: {path: input.descriptorFile, sha256: inputs[input.descriptorFile]}, archive: plan.archive,
    archiveSha256: audited.archiveSha256, archiveBytes: fs.statSync(plan.archive).size, archiveMembers: audited.objects.map(row => row.path),
    sourceState, inputs, dependencies, symbols: symbolAudit, commands, buildOnly: true, compiled: true, executed: false,
    publicationQualified: false, scope: "Original ten Windows HIPRTC/ROCclr translation units plus the admitted upstream HIPRTC builtins COFF object; not runtime-qualified."};
  fs.writeFileSync(path.join(input.output, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return result;
}

module.exports = {clrRevision, sourceFiles, recipeFiles, windowsHiprtcPlan, verifyComgrHeader, verifyIntelCompilerIdentity,
  verifyLlvmToolIdentity, auditHiprtcSymbolClosure, buildNativeWindowsHiprtcSdk};

if (require.main === module) {
  try {
    const [configFile, extra] = process.argv.slice(2);
    if (!configFile || extra) {throw new Error("Usage: build-native-windows-hiprtc-sdk.js /explicit-hiprtc-config.json");}
    buildNativeWindowsHiprtcSdk(readBuildJson(configFile));
  } catch (error) {console.error(error instanceof Error ? error.message : "Windows HIPRTC source production failed"); process.exitCode = 1;}
}
