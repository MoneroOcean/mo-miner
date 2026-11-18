"use strict";

// Produce the original Windows LLVM 21 host-library closure from its pinned source archive.
// GPU frontend/plugin artifacts and final application linking are deliberately separate lanes.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {createHash, randomUUID} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const {validateWindowsLlvmSdkConfig, windowsToolEnvironment, vfsFiles} = require("./build-native-windows-controller");
const {fileHash, buildFileInputs, readBuildJson} = require("./build-source-inputs");
const {windowsIntelRuleOverride} = require("./prepare-native-jit-toolchain");
const {jitBuildOutputs} = require("./native-jit-stages");
const {isStaticArchive} = require("./native-audit");
const {archiveName, archiveSha256, tarTool, sourceDirectories, llvmSourceSnapshot, verifyArchiveCatalog,
  llvmAmdgpulibCallsCompileOptionsShim, verifyCmakeVersion} = require("./native-llvm-sdk");

const repo = path.resolve(__dirname, "..");
const targetTriple = "x86_64-pc-windows-msvc";
const expectedStaticArchiveCount = 164;
const sourceProjects = Object.freeze(["llvm", "clang", "lld"]);
const recipeFiles = Object.freeze(["scripts/build-native-windows-llvm-sdk.js", "scripts/build-native-windows-controller.js",
  "scripts/build-source-inputs.js", "scripts/prepare-native-jit-toolchain.js", "scripts/native-jit-stages.js", "scripts/native-audit.js",
  "scripts/native-llvm-sdk.js"]);
const phaseTimeout = 3600000;
const buildPhases = Object.freeze([
  Object.freeze({name: "llvm-amdgpu", target: "lib/Target/AMDGPU/all"}),
  Object.freeze({name: "llvm-nvptx", target: "lib/Target/NVPTX/all"}),
  Object.freeze({name: "llvm-x86", target: "lib/Target/X86/all"}),
  Object.freeze({name: "llvm-core", target: "llvm-libraries"}),
  Object.freeze({name: "clang", target: "tools/clang/clang-libraries"}),
  Object.freeze({name: "lld", target: "tools/lld/lld-libraries"}),
]);

/** @typedef {{name: string, project: string, sourceDirectory: string, artifact: string, id: string, sources: number}} StaticTarget */
/** @typedef {{name: string, target: string}} BuildPhase */
/** @typedef {{sha256: string, files: number, bytes: number}} SourceState */
/** @typedef {{members: number, sha256: string}} ArchiveCatalog */
/** @typedef {{descriptorFile: string, archive: string, source: string, output: string, tools: {cmake: string, ninja: string}, jobs: number,
 * sourceOptionsFile: string, sourceOptionsText: string}} SavedPlan */
/** @typedef {StaticTarget & {sha256: string, bytes: number}} AuditedArchive */
/** @typedef {{version: number, role: string, runId: string, stage: string, plan: SavedPlan, planSha256: string,
 * sourceState: SourceState, catalog: ArchiveCatalog, fingerprints: Record<string, string>, recipeSha256: string,
 * configure: string[],
 * buildOnly: boolean, compiled: boolean, executed: boolean, publicationQualified: boolean, targets: StaticTarget[], phases: BuildPhase[],
 * outputs: Record<string, string>, producerSnapshot: {path: string, sha256: string}, sourceOptions?: {path: string, sha256: string},
 * hostSourceFlag?: {path: string, flag: string, compileCommands: number}, configuredFiles?: Record<string, string>, compileCommandsSha256?: string,
 * archives?: AuditedArchive[], objectProof?: {path: string, sha256: string, count: number},
 * admission?: {path: string, sha256: string},
 * admittedObjects?: {path: string, sha256: string, count: number}}} WindowsLlvmReceipt */

/** CMake paths are literals, not list separators, interpolation or quoted-flag fragments.
 * @param {unknown} value @returns {value is string} */
const literal = value => typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) &&
  !/[;"$`\\]/.test(value) && !Array.from(value).some(c => c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127);
/** @param {string} value */
const cmake = value => {
  if (typeof value !== "string" || value.length > 65536 || /[\r\n\0]/.test(value)) {
    throw new Error("Windows LLVM CMake values must be bounded literal strings");
  }
  let equals = "==";
  for (const match of value.matchAll(/\](=*)\]/g)) {
    if (match[1] !== undefined && match[1].length >= equals.length) {equals = "=".repeat(match[1].length + 1);}
  }
  return "[" + equals + "[" + value + "]" + equals + "]";
};
/** @param {string} root @param {string} child */
function contained(root, child) {
  const relative = path.relative(root, child);
  return relative === "" || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep));
}
/** @param {string} left @param {string} right */
function overlaps(left, right) {return contained(left, right) || contained(right, left);}

/** Check each existing cache path component without following symlinks out of the Ninja tree.
 * @param {string} build @param {string} file @param {"file" | "directory"} finalKind @param {boolean} [optional] */
function verifyWindowsBuildPath(build, file, finalKind, optional = false) {
  const root = fs.realpathSync(build);
  if (root !== build || path.resolve(file) !== file || !contained(build, file)) {
    throw new Error("Windows LLVM build path escaped or aliased its selected tree: " + file);
  }
  const parts = path.relative(build, file).split(path.sep); let current = build;
  for (const [index, part] of parts.entries()) {
    current = path.join(current, part);
    const stat = fs.lstatSync(current, {throwIfNoEntry: false});
    if (!stat) {
      if (optional) {return false;}
      throw new Error("Windows LLVM build path is missing: " + file);
    }
    const final = index === parts.length - 1;
    if (stat.isSymbolicLink() || (finalKind === "directory" || !final) && !stat.isDirectory() || final && finalKind === "file" && !stat.isFile()) {
      throw new Error("Windows LLVM build path is not a regular in-tree " + (final ? finalKind : "directory") + ": " + file);
    }
  }
  if (!contained(root, fs.realpathSync(file))) {throw new Error("Windows LLVM build path resolves outside its selected tree: " + file);}
  return true;
}

/** @param {string} directory @param {string} label @param {boolean} [ownerOnly] */
function verifyWindowsPrivateDirectory(directory, label, ownerOnly = false) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false}), uid = typeof process.getuid === "function" ? process.getuid() : -1;
  if (!stat || stat.isSymbolicLink() || !stat.isDirectory() || fs.realpathSync(directory) !== directory || stat.uid !== uid ||
      (stat.mode & (ownerOnly ? 0o077 : 0o022)) !== 0) {
    throw new Error("Windows LLVM cache " + label + " must be a canonical, privately owned directory");
  }
}

/** Re-derive the source/build layout before any read from a recorded build path. @param {ReturnType<typeof windowsLlvmSdkPlan>} plan */
function verifyWindowsCacheRoots(plan) {
  if (!literal(plan.output) || path.normalize(plan.output) !== plan.output || plan.source !== path.join(plan.output, "source") ||
      plan.build !== path.join(plan.output, "build") || [repo, plan.archive, ...Object.values(plan.tools)].some(input => overlaps(input, plan.output))) {
    throw new Error("Windows LLVM build roots do not match their recorded generation");
  }
  verifyWindowsPrivateDirectory(plan.output, "generation output", true);
  verifyWindowsPrivateDirectory(plan.source, "source"); verifyWindowsPrivateDirectory(plan.build, "build");
  verifyWindowsConfiguration(plan);
}

/** The generated configure helper files still hold the bytes the frozen plan derives.
 * @param {ReturnType<typeof windowsLlvmSdkPlan>} plan */
function verifyWindowsConfiguration(plan) {
  const inputs = /** @type {[string, string][]} */ ([[plan.overlay, plan.overlayText], [plan.ruleOverride, plan.ruleOverrideText],
    [plan.toolchainFile, plan.toolchain]]);
  for (const [file, expected] of inputs) {
    const stat = fs.lstatSync(file, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file || stat.size !== Buffer.byteLength(expected) ||
        fs.readFileSync(file, "utf8") !== expected) {
      throw new Error("Windows LLVM configure inputs changed; a full new configure/build is required");
    }
  }
  verifyWindowsSourceOptionsFile(plan);
}

/** Bind the host-only source hook to the selected generation root. @param {ReturnType<typeof windowsLlvmSdkPlan>} plan */
function verifyWindowsSourceOptionsFile(plan) {
  if (plan.sourceOptionsFile !== path.join(plan.output, "llvm-host-source-options.cmake") ||
      plan.sourceOptionsText !== llvmAmdgpulibCallsCompileOptionsShim(plan.source)) {
    throw new Error("Windows LLVM source-options hook differs from the frozen plan");
  }
  const stat = fs.lstatSync(plan.sourceOptionsFile, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(plan.sourceOptionsFile) !== plan.sourceOptionsFile ||
      stat.size !== Buffer.byteLength(plan.sourceOptionsText) || fs.readFileSync(plan.sourceOptionsFile, "utf8") !== plan.sourceOptionsText) {
    throw new Error("Windows LLVM source-options hook is not the expected generated build input");
  }
  return evidenceFile(plan.sourceOptionsFile);
}

/** Every recorded input still has its recorded bytes. @param {Record<string, string>} expected */
function windowsBuildInputsCurrent(expected) {return isDeepStrictEqual(expected, buildFileInputs(Object.keys(expected || {})));}

/** One explicit static Windows COFF toolchain. The C and C++ drivers are the same selected
 * Intel executable; tablegen stays on the recorded native LLVM 21 utility root.
 * @param {unknown} descriptor @param {string} archive @param {string} output
 * @param {{cmake: string, ninja: string}} tools @param {number} jobs */
function windowsLlvmSdkPlan(descriptor, archive, output, tools, jobs) {
  const config = validateWindowsLlvmSdkConfig(descriptor), {windowsSdk, tools: selected} = config;
  if (!tools || Object.keys(tools).sort().join(",") !== "cmake,ninja" ||
      ![archive, output, tools.cmake, tools.ninja, selected.clangCl, selected.lldLink, selected.llvmNm, tarTool,
        selected.clangResourceInclude, windowsSdk.overlay, windowsSdk.sourceRoot, ...windowsSdk.includeRoots,
        ...windowsSdk.libraryRoots, ...windowsSdk.msvcLibraries, ...windowsSdk.ucrtLibraries].every(literal) ||
      path.basename(archive) !== archiveName ||
      path.normalize(output) !== output || !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {
    throw new Error("Windows LLVM 21 SDK needs the pinned source, explicit tools/roots and jobs in 1..16");
  }
  const generatedRoot = output;
  const source = path.join(output, "source"), archiveCopy = path.join(output, archiveName);
  const utilityRoot = path.dirname(selected.llvmNm), build = path.join(output, "build"),
    overlay = path.join(generatedRoot, "windows-sdk-vfs.json"),
    sourceOptionsFile = path.join(generatedRoot, "llvm-host-source-options.cmake"),
    sourceOptionsText = llvmAmdgpulibCallsCompileOptionsShim(source);
  const ruleOverride = path.join(generatedRoot, "intel-link-rules.cmake"), toolchainFile = path.join(generatedRoot, "toolchain.cmake");
  const nativeTools = /** @type {Record<"llvm-lib" | "llvm-ranlib" | "llvm-rc" | "llvm-ml" | "llvm-tblgen" | "clang-tblgen", string>} */
    (Object.fromEntries(["llvm-lib", "llvm-ranlib", "llvm-rc", "llvm-ml", "llvm-tblgen", "clang-tblgen"]
      .map(name => [name, path.join(utilityRoot, name)])));
  const compilerTools = {CMAKE_C_COMPILER: selected.clangCl, CMAKE_CXX_COMPILER: selected.clangCl,
    CMAKE_LINKER: selected.lldLink, CMAKE_AR: nativeTools["llvm-lib"], CMAKE_RANLIB: nativeTools["llvm-ranlib"],
    CMAKE_RC_COMPILER: nativeTools["llvm-rc"], CMAKE_ASM_MASM_COMPILER: nativeTools["llvm-ml"]};
  const links = ["/MACHINE:X64", "/INCREMENTAL:NO", ...windowsSdk.libraryRoots.map(root => '/LIBPATH:"' + root + '"')].join(" ");
  const runtimeLibraries = [...windowsSdk.msvcLibraries, ...windowsSdk.ucrtLibraries, ...windowsSdk.importLibraries];
  // CMake concatenates this flag with its directory. Intel 2026.2 ignores -imsvc under -nostdinc,
  // while the joined /external:Ipath form remains an explicit system include.
  const ruleOverrideText = windowsIntelRuleOverride(links, runtimeLibraries) +
    "set(CMAKE_INCLUDE_SYSTEM_FLAG_C /external:I)\nset(CMAKE_INCLUDE_SYSTEM_FLAG_CXX /external:I)\n";
  const virtual = [...vfsFiles(windowsSdk.overlay)].filter(([name]) => windowsSdk.includeRoots.some(root =>
    name.toLowerCase().startsWith(root.replace(/\/+$/, "").toLowerCase() + "/")));
  if (!virtual.length) {throw new Error("Windows LLVM 21 needs mapped OS SDK headers");}
  const overlayText = JSON.stringify({version: 0, "case-sensitive": false, "use-external-names": true,
    roots: virtual.map(([name, file]) => ({type: "file", name, "external-contents": file}))});
  const includeFlags = ["--target=" + targetTriple, "/MT", "/Gy", "/Gw", "-nostdinc", "-I" + selected.clangResourceInclude,
    "-Xclang", "-ivfsoverlay", "-Xclang", overlay, ...windowsSdk.includeRoots.map(root => "-I" + root)];
  const commonFlags = includeFlags.map(value => '"' + value + '"').join(" ");
  const toolchain = ["set(CMAKE_SYSTEM_NAME Windows)", "set(CMAKE_SYSTEM_PROCESSOR AMD64)", "set(CMAKE_SYSTEM_VERSION 10.0)",
    "set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)", `set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded CACHE STRING ${cmake("")} FORCE)`,
    // The public target properties let CMake identify the compiler from a COFF object
    // before its link rules are known. ABI/feature probes below still link real executables.
    ...["C", "CXX"].map(language => `set(CMAKE_${language}_COMPILER_TARGET ${cmake(targetTriple)} CACHE STRING "" FORCE)`),
    `set(CMAKE_INCLUDE_SYSTEM_FLAG_C ${cmake("/external:I")} CACHE STRING ${cmake("")} FORCE)`,
    `set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX ${cmake("/external:I")} CACHE STRING ${cmake("")} FORCE)`,
    // LLVM doesn't enable ASM_MASM as a CMake language, so set its verifier-owned flag directly in the cache.
    `set(CMAKE_ASM_MASM_FLAGS ${cmake("/nologo --m64")} CACHE STRING ${cmake("")} FORCE)`,
    ...Object.entries(compilerTools).map(([name, file]) => `set(${name} ${cmake(file)} CACHE FILEPATH "" FORCE)`),
    `set(CMAKE_USER_MAKE_RULES_OVERRIDE_C ${cmake(ruleOverride)})`,
    `set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX ${cmake(ruleOverride)})`,
    `set(CMAKE_C_FLAGS_INIT ${cmake(commonFlags)})`, `set(CMAKE_CXX_FLAGS_INIT ${cmake(commonFlags + ' "/EHsc" "/GR" "/std:c++17"')})`,
    `set(CMAKE_EXE_LINKER_FLAGS_INIT ${cmake(links)})`, `set(CMAKE_SHARED_LINKER_FLAGS_INIT ${cmake(links)})`,
    `set(CMAKE_MODULE_LINKER_FLAGS_INIT ${cmake(links)})`, `set(LLVM_NATIVE_TOOL_DIR ${cmake(utilityRoot)} CACHE PATH "" FORCE)`,
    `set(LLVM_TABLEGEN ${cmake(nativeTools["llvm-tblgen"])} CACHE FILEPATH "" FORCE)`,
    `set(CLANG_TABLEGEN ${cmake(nativeTools["clang-tblgen"])} CACHE FILEPATH "" FORCE)`,
    `set(CMAKE_FIND_ROOT_PATH ${cmake(windowsSdk.sourceRoot)})`,
    ...["LIBRARY", "INCLUDE", "PACKAGE"].map(name => `set(CMAKE_FIND_ROOT_PATH_MODE_${name} ONLY)`),
    "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)", ""].join("\n");
  // MinSizeRel limits embedded compiler-host LLVM only; Core/mining code and device kernels are separate lanes, and JIT performance still needs qualification.
  const definitions = ["-DCMAKE_TOOLCHAIN_FILE=" + toolchainFile, "-DCMAKE_MAKE_PROGRAM=" + tools.ninja,
    "-DCMAKE_BUILD_TYPE=MinSizeRel", "-DCMAKE_INSTALL_PREFIX=" + path.join(generatedRoot, "unused-install-prefix"),
    "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=" + sourceOptionsFile,
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_SUPPRESS_REGENERATION=ON",
    "-DLLVM_ENABLE_PROJECTS=clang;lld", "-DLLVM_TARGETS_TO_BUILD=AMDGPU;NVPTX;X86",
    "-DLLVM_NATIVE_TOOL_DIR=" + utilityRoot, "-DLLVM_TABLEGEN=" + nativeTools["llvm-tblgen"],
    "-DCLANG_TABLEGEN=" + nativeTools["clang-tblgen"], "-DLLVM_BUILD_TOOLS=OFF", "-DLLVM_INCLUDE_TOOLS=ON",
    "-DCLANG_BUILD_TOOLS=OFF", "-DLLD_BUILD_TOOLS=OFF", "-DLLVM_INCLUDE_UTILS=OFF", "-DLLVM_INCLUDE_TESTS=OFF",
    "-DCLANG_INCLUDE_TESTS=OFF", "-DLLVM_INCLUDE_EXAMPLES=OFF",
    "-DLLVM_INCLUDE_BENCHMARKS=OFF", "-DLLVM_BUILD_RUNTIME=OFF", "-DLLVM_ENABLE_RUNTIMES=",
    "-DLLVM_ENABLE_BINDINGS=OFF", "-DLLVM_ENABLE_EH=ON", "-DLLVM_ENABLE_RTTI=ON", "-DLLVM_ENABLE_PIC=OFF",
    "-DLLVM_ENABLE_PLUGINS=OFF", "-DBUILD_SHARED_LIBS=OFF", "-DLLVM_BUILD_LLVM_DYLIB=OFF",
    "-DLLVM_BUILD_LLVM_C_DYLIB=OFF", "-DLLVM_LINK_LLVM_DYLIB=OFF", "-DLIBCLANG_BUILD_STATIC=ON",
    "-DCLANG_ENABLE_ARCMT=OFF", "-DCLANG_ENABLE_STATIC_ANALYZER=OFF",
    "-DLLVM_ENABLE_ZLIB=OFF", "-DLLVM_ENABLE_ZSTD=OFF", "-DLLVM_ENABLE_LIBXML2=OFF",
    "-DLLVM_ENABLE_LIBEDIT=OFF", "-DLLVM_ENABLE_LIBPFM=OFF", "-DLLVM_ENABLE_CURL=OFF"];
  const configure = [tools.cmake, "-G", "Ninja", "-S", path.join(source, "llvm"), "-B", build, ...definitions];
  return {version: 1, role: "windows-llvm21-host-sdk", archive, archiveSha256, archiveCopy, source,
    sourceProjects: [...sourceProjects], output, build, jobs, tools: {...tools}, tarTool, utilityRoot, nativeTools,
    compilerTools, overlay, overlayText, sourceOptionsFile, sourceOptionsText, toolchainFile, toolchain, ruleOverride, ruleOverrideText, configure,
    definitions, targetTriple, phases: [...buildPhases], phaseTimeout};
}

/** CMake File API target inventory: only upstream static libraries from the three selected
 * projects become build targets. No hand-maintained LLVM archive-name list is used.
 * @param {string} build @param {string} source */
function llvmStaticArchiveTargets(build, source) {
  if (!literal(build) || !literal(source)) {throw new Error("Select explicit LLVM build/source roots");}
  const reply = path.join(build, ".cmake", "api", "v1", "reply");
  verifyWindowsBuildPath(build, reply, "directory");
  const indexes = fs.readdirSync(reply).filter(name => /^index-[A-Za-z0-9_.-]+\.json$/.test(name));
  if (indexes.length !== 1) {throw new Error("Missing or ambiguous LLVM 21 CMake File API index");}
  const indexFile = indexes[0];
  if (!indexFile) {throw new Error("Missing LLVM 21 CMake File API index");}
  const indexPath = path.join(reply, indexFile); verifyWindowsBuildPath(build, indexPath, "file");
  const index = /** @type {{reply?: {"codemodel-v2"?: {jsonFile?: string}}}} */ (readBuildJson(indexPath));
  const modelRef = index.reply?.["codemodel-v2"];
  if (!modelRef || typeof modelRef.jsonFile !== "string" || path.basename(modelRef.jsonFile) !== modelRef.jsonFile) {
    throw new Error("LLVM 21 CMake File API omitted its codemodel");
  }
  const modelFile = path.join(reply, modelRef.jsonFile); verifyWindowsBuildPath(build, modelFile, "file");
  const model = /** @type {{paths?: {source?: string, build?: string}, configurations?: {targets?: {id?: string, jsonFile?: string}[]}[]}} */
    (readBuildJson(modelFile));
  const configurations = model.configurations, targetReferences = configurations?.[0]?.targets;
  if (!model.paths || typeof model.paths.source !== "string" || !path.isAbsolute(model.paths.source) ||
      typeof model.paths.build !== "string" || !path.isAbsolute(model.paths.build) ||
      !Array.isArray(configurations) || configurations.length !== 1 || !Array.isArray(targetReferences)) {
    throw new Error("LLVM 21 CMake File API has an unexpected configuration set");
  }
  const sourceRoot = fs.realpathSync(source), buildRoot = fs.realpathSync(build);
  const modelSource = path.resolve(model.paths.source), modelBuild = path.resolve(model.paths.build);
  if (!contained(sourceRoot, modelSource) || !contained(buildRoot, modelBuild)) {throw new Error("LLVM 21 CMake codemodel escaped its source/build roots");}
  /** @type {{name: string, project: string, sourceDirectory: string, artifact: string, id: string, sources: number}[]} */ const targets = [];
  for (const reference of targetReferences) {
    if (!reference || typeof reference.id !== "string" || typeof reference.jsonFile !== "string" ||
        path.basename(reference.jsonFile) !== reference.jsonFile) {throw new Error("Invalid LLVM CMake target reference");}
    const targetFile = path.join(reply, reference.jsonFile); verifyWindowsBuildPath(build, targetFile, "file");
    const target = /** @type {{name?: string, type?: string, paths?: {source?: string, build?: string}, artifacts?: {path?: string}[], sources?: unknown[]}} */
      (readBuildJson(targetFile));
    const targetSource = typeof target?.paths?.source === "string" ? path.resolve(modelSource, target.paths.source) : "";
    const targetBuild = typeof target?.paths?.build === "string" ? path.resolve(modelBuild, target.paths.build) : "";
    const project = sourceProjects.find(name => contained(path.join(sourceRoot, name), targetSource));
    if (!target || !project) {continue;}
    if (!contained(buildRoot, targetBuild)) {throw new Error("LLVM 21 target build directory escaped its output");}
    // File API lists configured tools even when they are excluded from every requested
    // aggregate. Admit only static libraries here; the output audit rejects PE tools/DLLs
    // if one actually enters the selected build closure.
    if (target.type !== "STATIC_LIBRARY") {continue;}
    if (typeof target.name !== "string" || !/^[A-Za-z0-9_.+-]{1,160}$/.test(target.name) ||
        !Array.isArray(target.artifacts) || target.artifacts.length !== 1 || typeof target.artifacts[0]?.path !== "string") {
      throw new Error("Invalid LLVM 21 static target artifact");
    }
    const artifact = path.isAbsolute(target.artifacts[0].path) ? target.artifacts[0].path : path.resolve(modelBuild, target.artifacts[0].path);
    if (!contained(buildRoot, artifact) || !artifact.endsWith(".lib")) {throw new Error("LLVM 21 target is not a static COFF archive: " + artifact);}
    targets.push({name: target.name, project, sourceDirectory: targetSource, artifact, id: reference.id,
      sources: Array.isArray(target.sources) ? target.sources.length : 0});
  }
  targets.sort((a, b) => a.project.localeCompare(b.project) || a.artifact.localeCompare(b.artifact));
  if (!targets.length || targets.length > 512 || new Set(targets.map(row => row.name)).size !== targets.length ||
      new Set(targets.map(row => row.artifact)).size !== targets.length || sourceProjects.some(project => !targets.some(row => row.project === project))) {
    throw new Error("LLVM 21 static target inventory is incomplete or ambiguous");
  }
  for (const backend of ["AMDGPU", "NVPTX", "X86"]) {
    const backendRoot = path.join(sourceRoot, "llvm", "lib", "Target", backend);
    if (!targets.some(row => row.project === "llvm" && contained(backendRoot, row.sourceDirectory))) {
      throw new Error("LLVM 21 static target inventory omitted the " + backend + " backend");
    }
  }
  return targets;
}

/** Verify only the selected upstream aggregates exist; each is one explicit <=60-minute phase.
 * @param {string} build */
function verifyBuildPhases(build) {
  const file = path.join(build, "build.ninja"), stat = fs.statSync(file);
  if (!stat.isFile() || stat.size > (64 << 20)) {throw new Error("Invalid LLVM 21 Ninja target graph");}
  const text = fs.readFileSync(file, "utf8"), targets = new Set(text.split(/\r?\n/).map(line => line.match(/^build ([^ :]+):/)?.[1]).filter(Boolean));
  for (const phase of buildPhases) {if (!targets.has(phase.target)) {throw new Error("CMake omitted LLVM 21 phase target: " + phase.target);}}
  return {buildNinjaSha256: fileHash(file), phases: buildPhases.map(phase => ({...phase}))};
}

/** Parse CMake's POSIX-quoted command string without invoking a shell. @param {string} command */
function commandArguments(command) {
  if (command.length > (1 << 20) || /[\r\n\0]/.test(command)) {throw new Error("Invalid LLVM 21 compile command length");}
  const args = []; let value = "", quote = "", escaped = false, active = false;
  for (const char of command) {
    if (escaped) {value += char; escaped = false; active = true; continue;}
    if (char === "\\" && quote !== "'") {escaped = true; active = true; continue;}
    if (quote) {if (char === quote) {quote = "";} else {value += char;} active = true; continue;}
    if (char === "'" || char === '"') {quote = char; active = true; continue;}
    if (/[;&|<>]/.test(char)) {throw new Error("LLVM 21 compile command contains a shell operator");}
    if (/\s/.test(char)) {if (active) {args.push(value); value = ""; active = false;} continue;}
    if (char === "$" || char === "`") {throw new Error("LLVM 21 compile command contains shell expansion");}
    value += char; active = true;
  }
  if (escaped || quote) {throw new Error("LLVM 21 compile command has incomplete quoting");}
  if (active) {args.push(value);}
  if (!args.length || args.length > 100000) {throw new Error("LLVM 21 compile command is empty or excessive");}
  return args;
}

/** Require the compiler-generated host option on only the AMDGPU library helper TU.
 * @param {string} build @param {string} source */
function verifyAmdgpulibCallsCompileFlag(build, source) {
  return verifyAmdgpulibCallsCompileFlagInternal(build, source, true);
}

/** @param {string} build @param {string} source @param {boolean} required
 * @returns {{path: string, flag: string, compileCommands: number} | undefined} */
function verifyAmdgpulibCallsCompileFlagInternal(build, source, required) {
  const file = path.join(build, "compile_commands.json"); verifyWindowsBuildPath(build, file, "file");
  const stat = fs.statSync(file);
  if (!stat.isFile() || stat.size < 2 || stat.size > (256 << 20)) {throw new Error("Invalid or excessive LLVM 21 compile_commands.json");}
  const rows = readBuildJson(file);
  if (!Array.isArray(rows) || !rows.length || rows.length > 200000) {throw new Error("LLVM compile command inventory is empty or excessive");}
  const buildRoot = fs.realpathSync(build), sourceRoot = fs.realpathSync(source),
    target = fs.realpathSync(path.join(sourceRoot, "llvm", "lib", "Target", "AMDGPU", "AMDGPULibCalls.cpp"));
  if (!contained(sourceRoot, target)) {throw new Error("AMDGPULibCalls.cpp escaped the frozen LLVM source tree");}
  let targetRows = 0, flaggedRows = 0;
  for (const row of rows) {
    if (!row || typeof row.directory !== "string" || !path.isAbsolute(row.directory) || typeof row.file !== "string" || !row.file) {
      throw new Error("LLVM compile command row omitted its directory or source");
    }
    const directory = fs.realpathSync(row.directory);
    if (!contained(buildRoot, directory)) {throw new Error("LLVM compile command directory escaped its build tree");}
    const sourceFile = path.resolve(directory, row.file);
    if (!contained(sourceRoot, sourceFile) && !contained(buildRoot, sourceFile)) {
      throw new Error("LLVM compile command references a source outside its frozen trees");
    }
    const actualSource = fs.realpathSync(sourceFile);
    if (!contained(sourceRoot, actualSource) && !contained(buildRoot, actualSource)) {
      throw new Error("LLVM compile command source escaped its frozen trees");
    }
    /** @type {number[]} */ const counts = [];
    if (row.arguments !== undefined) {
      if (!Array.isArray(row.arguments) || !row.arguments.every((/** @type {unknown} */ value) => typeof value === "string")) {
        throw new Error("LLVM compile command arguments are malformed");
      }
      counts.push(row.arguments.filter((/** @type {string} */ value) => value === "-fno-builtin").length);
    }
    if (row.command !== undefined) {
      if (typeof row.command !== "string") {throw new Error("LLVM compile command text is malformed");}
      counts.push(commandArguments(row.command).filter(value => value === "-fno-builtin").length);
    }
    if (!counts.length || counts.some(value => value !== counts[0])) {throw new Error("LLVM compile command option representations disagree");}
    const count = counts[0] || 0;
    if (actualSource === target) {
      targetRows++;
      if (count !== Number(required)) {throw new Error("AMDGPULibCalls.cpp has an unexpected host-only -fno-builtin state");}
    }
    if (count) {
      flaggedRows++;
      if (actualSource !== target || count !== 1) {throw new Error("-fno-builtin escaped the AMDGPULibCalls.cpp source row");}
    }
  }
  if (targetRows !== 1 || flaggedRows !== Number(required)) {
    throw new Error("LLVM compile database must contain exactly one AMDGPULibCalls.cpp host-option row");
  }
  return required ? {path: target, flag: "-fno-builtin", compileCommands: rows.length} : undefined;
}

/** Read the COFF machine field, including the standard bigobj header variant. @param {string} file */
function coffObjectMachine(file) {
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK), header = Buffer.alloc(20);
  try {
    if (!fs.fstatSync(fd).isFile() || fs.readSync(fd, header, 0, header.length, 0) !== header.length) {return null;}
    const machine = header.readUInt16LE(0);
    if (machine === 0x8664) {return machine;}
    return machine === 0 && header.readUInt16LE(2) === 0xffff && header.readUInt16LE(6) === 0x8664 ? 0x8664 : null;
  } finally {fs.closeSync(fd);}
}

/** @param {string} build @param {string} directory @param {string} value */
function buildPath(build, directory, value) {
  const file = path.resolve(directory, value), root = fs.realpathSync(build);
  if (!contained(root, file)) {throw new Error("LLVM 21 compile command output escaped its build tree");}
  return file;
}

/** @param {string[]} args */
function commandOutput(args) {
  for (let index = 0; index < args.length; index++) {
    const arg = args[index] || "";
    if (arg === "-o" || /^[-/]Fo$/i.test(arg || "")) {return args[index + 1] || "";}
    if (/^[-/]Fo.+/i.test(arg || "")) {return arg.slice(3);}
  }
  return "";
}

/** Validate the full CMake database, but require object files only for the selected Ninja closure.
 * This intentionally excludes configured tool EXE rows that are not inputs to admitted archives.
 * @param {string} compileCommands @param {string} compiler @param {string} build @param {string} source
 * @param {string[]} [objectClosure] */
function auditCompileCommands(compileCommands, compiler, build, source, objectClosure = []) {
  const stat = fs.statSync(compileCommands);
  if (!stat.isFile() || stat.size < 2 || stat.size > (256 << 20)) {throw new Error("Invalid or excessive LLVM 21 compile_commands.json");}
  const bytes = fs.readFileSync(compileCommands), entries = JSON.parse(bytes.toString("utf8"));
  if (!Array.isArray(entries) || !entries.length || entries.length > 100000) {throw new Error("LLVM 21 compile command inventory is empty or excessive");}
  const buildRoot = fs.realpathSync(build), sourceRoot = fs.realpathSync(source);
  /** @type {Record<string, {source: string, output: string, compiler: string, compilerIdentity: string, language: string,
   * commandSha256: string, abiSha256: string, sha256?: string, bytes?: number}>} */ const commands = {};
  for (const entry of entries) {
    if (!entry || typeof entry.directory !== "string" || typeof entry.file !== "string" || typeof entry.output !== "string" ||
        typeof entry.command !== "string" || !path.isAbsolute(entry.directory)) {throw new Error("LLVM 21 compile command omitted its directory, source or output");}
    const directory = fs.realpathSync(entry.directory);
    if (!contained(buildRoot, directory)) {throw new Error("LLVM 21 compile command directory escaped its build tree");}
    const output = buildPath(build, directory, entry.output);
    const sourceFile = path.resolve(directory, entry.file);
    if (!contained(sourceRoot, sourceFile) && !contained(buildRoot, sourceFile)) {throw new Error("LLVM 21 compile command references a source outside its frozen trees");}
    const actualSource = fs.realpathSync(sourceFile);
    if (!contained(sourceRoot, actualSource) && !contained(buildRoot, actualSource)) {throw new Error("LLVM 21 compile command source escaped its frozen trees");}
    if (!output.endsWith(".obj")) {throw new Error("LLVM 21 compile command output is not a COFF object");}
    const args = commandArguments(entry.command), extension = path.extname(sourceFile).toLowerCase();
    const language = extension === ".asm" ? "ASM_MASM" : [".c", ".m"].includes(extension) ? "C" :
      [".cc", ".cpp", ".cxx", ".c++", ".mm"].includes(extension) ? "CXX" : "";
    if (!language) {throw new Error("LLVM 21 compile command uses an unsupported source language: " + sourceFile);}
    if (!args.some(arg => arg === sourceFile || (!arg.startsWith("-") && path.resolve(directory, arg) === sourceFile)) ||
        path.resolve(directory, commandOutput(args)) !== output) {
      throw new Error("LLVM 21 compile command has the wrong source or object output");
    }
    if (language !== "ASM_MASM") {
      const target = args.some((arg, index) => arg === "--target=" + targetTriple || arg === "--target" && args[index + 1] === targetTriple);
      if (args[0] !== compiler || !target || !args.some(arg => /^\/MTd?$/i.test(arg)) ||
          args.some(arg => /^\/(?:MD|MDd)$/i.test(arg)) || args.some(arg => arg === "-isystem" || arg.startsWith("-isystem") || arg === "-imsvc" || arg.startsWith("-imsvc"))) {
        throw new Error("LLVM 21 compile command has the wrong compiler, Windows ABI or system include flag");
      }
      for (let index = 0; index < args.length; index++) {
        const arg = args[index] || "";
        if (/^\/external:I$/i.test(arg)) {
          if (!path.isAbsolute(args[index + 1] || "")) {throw new Error("LLVM 21 external include path is malformed");}
        } else if (/^\/external:I/i.test(arg) && !path.isAbsolute(arg.slice(11))) {
          throw new Error("LLVM 21 external include path is malformed");
        }
      }
      if (language === "CXX" && (!args.includes("/EHsc") || !args.includes("/GR"))) {throw new Error("LLVM 21 C++ compile command omitted exceptions or RTTI");}
    }
    if (Object.hasOwn(commands, output)) {throw new Error("LLVM 21 compile command output is duplicated");}
    const commandSha256 = createHash("sha256").update(JSON.stringify(args)).digest("hex");
    const commandCompiler = args[0] || "";
    if (!commandCompiler) {throw new Error("LLVM 21 compile command omitted its compiler");}
    commands[output] = {source: sourceFile, output, compiler: language === "ASM_MASM" ? commandCompiler : compiler,
      compilerIdentity: language === "ASM_MASM" ? "LLVM llvm-ml" : "IntelLLVM", language, commandSha256,
      abiSha256: createHash("sha256").update(JSON.stringify({compiler: language === "ASM_MASM" ? commandCompiler : compiler,
        compilerIdentity: language === "ASM_MASM" ? "LLVM llvm-ml" : "IntelLLVM", targetTriple,
        runtime: "MultiThreaded", language, exceptions: language === "CXX", rtti: language === "CXX",
        mode: language === "ASM_MASM" ? "x64" : undefined, objectFormat: language === "ASM_MASM" ? "COFF" : undefined})).digest("hex")};
  }
  /** @type {typeof commands[string][]} */ const objects = [], missing = [];
  for (const output of [...new Set(objectClosure.map(file => path.resolve(file)))].sort()) {
    const command = commands[output];
    if (!command) {missing.push(output); continue;}
    const object = fs.lstatSync(output, {throwIfNoEntry: false});
    if (!object?.isFile() || object.isSymbolicLink() || object.size < 20 ||
        !contained(buildRoot, fs.realpathSync(output)) || (command.language !== "ASM_MASM" && coffObjectMachine(output) !== 0x8664)) {
      throw new Error("LLVM 21 x64 COFF object is missing or invalid: " + output);
    }
    if (command.language !== "ASM_MASM") {objects.push({...command, sha256: fileHash(output), bytes: object.size});}
  }
  return {sha256: fileHash(compileCommands), count: entries.length, commands, objects, missing};
}

/** Read the exact direct COFF object inputs of a selected static Ninja target. @param {string} ninja @param {string} build @param {string} artifact */
function queryNinjaArchiveObjects(ninja, build, artifact) {
  const target = path.relative(build, artifact).split(path.sep).join("/");
  if (!target || target.startsWith("../") || path.isAbsolute(target)) {throw new Error("LLVM 21 archive escaped its Ninja build root");}
  const result = spawnSync(ninja, ["-C", build, "-t", "query", target], {encoding: "utf8", timeout: 30000, maxBuffer: 32 << 20});
  if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string") {throw new Error("LLVM 21 Ninja archive query failed: " + target, {cause: result.error});}
  const lines = result.stdout.split(/\r?\n/);
  if (lines[0] !== target + ":") {throw new Error("LLVM 21 Ninja query returned another target: " + target);}
  const members = []; let inputs = false;
  for (const line of lines.slice(1)) {
    if (line === "  input: " || line.startsWith("  input: ")) {inputs = true; continue;}
    if (line === "  outputs:") {inputs = false; continue;}
    if (!inputs || !line.startsWith("    ")) {continue;}
    const value = line.trim();
    if (value.startsWith("||")) {continue;}
    if (!value.endsWith(".obj")) {throw new Error("LLVM 21 static target has a non-object Ninja input: " + value);}
    const file = path.resolve(build, value);
    if (!contained(fs.realpathSync(build), file)) {throw new Error("LLVM 21 Ninja object escaped its build root");}
    members.push(file);
  }
  if (!members.length || new Set(members).size !== members.length) {throw new Error("LLVM 21 static target has no unique object closure: " + target);}
  return members.sort();
}

/** Verify every selected archive member is a real x64 COFF object, not ELF/ar data or an import lib.
 * @param {string} archive */
function verifyCoffArchive(archive) {
  if (!isStaticArchive(archive)) {throw new Error("LLVM 21 static target did not produce an ar/COFF archive: " + archive);}
  const fd = fs.openSync(archive, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK), size = fs.fstatSync(fd).size;
  let position = 8, members = 0;
  try {
    while (position < size) {
      const header = Buffer.alloc(60);
      if (fs.readSync(fd, header, 0, 60, position) !== 60 || header.toString("latin1", 58, 60) !== "`\n") {throw new Error("Malformed LLVM 21 COFF archive member header");}
      const name = header.toString("latin1", 0, 16).trim(), lengthText = header.toString("latin1", 48, 58).trim();
      if (!/^\d+$/.test(lengthText)) {throw new Error("Malformed LLVM 21 COFF archive member size");}
      const length = Number(lengthText), data = position + 60;
      if (!Number.isSafeInteger(length) || length < 0 || data + length > size) {throw new Error("LLVM 21 COFF archive member escaped its file");}
      if (name !== "/" && name !== "//" && name !== "/SYM64/" && !name.startsWith("__.SYMDEF")) {
        let offset = data, objectBytes = length;
        if (name.startsWith("#1/")) {const prefix = Number(name.slice(3)); if (!Number.isSafeInteger(prefix) || prefix < 1 || prefix >= length) {throw new Error("Malformed extended COFF archive name");} offset += prefix; objectBytes -= prefix;}
        const object = Buffer.alloc(20);
        if (objectBytes < object.length || fs.readSync(fd, object, 0, object.length, offset) !== object.length ||
            (object.readUInt16LE(0) !== 0x8664 && !(object.readUInt16LE(0) === 0 && object.readUInt16LE(2) === 0xffff && object.readUInt16LE(6) === 0x8664))) {
          throw new Error("LLVM 21 archive contains a non-x64-COFF member: " + name);
        }
        members++;
      }
      position = data + length + (length & 1);
    }
  } finally {fs.closeSync(fd);}
  if (position !== size || members < 1) {throw new Error("LLVM 21 COFF archive is empty or malformed: " + archive);}
  return {members, bytes: size, sha256: fileHash(archive)};
}

/** Verify an assembler object against the exact Ninja edge and 64-bit llvm-ml command.
 * @param {string} ninja @param {string} build @param {string} sourceRoot @param {string} output @param {string} compiler */
function auditNinjaAssemblyObject(ninja, build, sourceRoot, output, compiler) {
  const target = path.relative(build, output).split(path.sep).join("/");
  const query = spawnSync(ninja, ["-C", build, "-t", "query", target], {encoding: "utf8", timeout: 30000, maxBuffer: 4 << 20});
  if (query.error || query.signal || query.status !== 0) {throw new Error("LLVM 21 MASM object graph query failed: " + target, {cause: query.error});}
  const lines = String(query.stdout).split(/\r?\n/), rule = lines.find(line => line.startsWith("  input: "))?.slice(9) || "";
  const source = lines.slice(lines.indexOf(lines.find(line => line.startsWith("  input: ")) || "") + 1)
    .map(line => line.trim()).find(value => value.endsWith(".asm") && !value.startsWith("||")) || "";
  if (lines[0] !== target + ":" || !rule.startsWith("ASM_MASM_COMPILER") || !path.isAbsolute(source) ||
      !contained(fs.realpathSync(sourceRoot), source)) {throw new Error("LLVM 21 object is not a frozen-source MASM edge: " + target);}
  const result = spawnSync(ninja, ["-C", build, "-t", "commands", target], {encoding: "utf8", timeout: 30000, maxBuffer: 4 << 20});
  if (result.error || result.signal || result.status !== 0) {throw new Error("LLVM 21 MASM command query failed: " + target, {cause: result.error});}
  const commands = String(result.stdout).split(/\r?\n/).filter(line => /(?:^|\/)llvm-ml(?:\s|$)/.test(line));
  if (commands.length !== 1) {throw new Error("LLVM 21 MASM object has an ambiguous llvm-ml command: " + target);}
  const [command] = commands;
  if (!command) {throw new Error("LLVM 21 MASM object omitted its llvm-ml command: " + target);}
  const args = commandArguments(command);
  if (!args[0]) {throw new Error("LLVM 21 MASM command omitted llvm-ml: " + target);}
  const compilerReal = fs.realpathSync(compiler), selectedReal = fs.realpathSync(args[0] || ""), outputFlag = args.indexOf("-Fo");
  const emitted = outputFlag >= 0 ? args[outputFlag + 1] || "" : commandOutput(args);
  if (selectedReal !== compilerReal || !(args.includes("--m64") || args.includes("-m64")) ||
      !(args.includes("-c") || args.includes("/c")) || !emitted || path.resolve(build, emitted) !== output || !args.includes(source)) {
    throw new Error("LLVM 21 MASM command has the wrong compiler, x64 mode or object output: " + target);
  }
  const stat = fs.statSync(output);
  if (!stat.isFile() || stat.size < 20 || coffObjectMachine(output) !== 0x8664) {throw new Error("LLVM 21 MASM x64 COFF object is missing or invalid: " + output);}
  return {source, output, sha256: fileHash(output), bytes: stat.size, compiler: args[0], compilerIdentity: "LLVM llvm-ml",
    commandSha256: createHash("sha256").update(JSON.stringify(args)).digest("hex"),
    abiSha256: createHash("sha256").update(JSON.stringify({compiler: args[0], compilerIdentity: "LLVM llvm-ml", targetTriple, mode: "x64", objectFormat: "COFF"})).digest("hex")};
}

/** Derive the exact object closure from selected Ninja archive edges and bind every object to
 * its CMake command (or separate x64 llvm-ml edge) and every archive to COFF.
 * @param {string} ninja @param {string} build @param {string} source @param {string} compiler
 * @param {string} masm @param {string} compileCommands @param {StaticTarget[]} targets */
function auditWindowsLlvmArchives(ninja, build, source, compiler, masm, compileCommands, targets) {
  if (!targets.length || targets.length > 512) {throw new Error("LLVM 21 static archive inventory is empty or excessive");}
  verifyWindowsTargetPaths(build, targets);
  const buildRoot = fs.realpathSync(build), archiveInputs = new Map();
  for (const target of targets) {
    if (!isStaticArchive(target.artifact) || !contained(buildRoot, path.resolve(target.artifact))) {
      throw new Error("LLVM 21 archive target escaped its build tree: " + target.artifact);
    }
    archiveInputs.set(target.artifact, queryNinjaArchiveObjects(ninja, build, target.artifact));
  }
  const closure = [...new Set([...archiveInputs.values()].flat())].sort();
  const compile = auditCompileCommands(compileCommands, compiler, build, source, closure);
  if (compile.missing.length || compile.objects.length + closure.filter(file => compile.commands[file]?.language === "ASM_MASM").length !== closure.length) {
    throw new Error("LLVM 21 archive object closure is not completely represented in compile_commands.json");
  }
  const objects = new Map(compile.objects.map(row => [row.output, row]));
  for (const output of closure.filter(file => compile.commands[file]?.language === "ASM_MASM")) {
    const command = compile.commands[output];
    if (!command) {throw new Error("LLVM 21 MASM object is missing its compiler command: " + output);}
    const row = auditNinjaAssemblyObject(ninja, build, source, output, masm);
    if (row.source !== command.source || row.output !== command.output) {
      throw new Error("LLVM 21 MASM Ninja edge disagrees with compile_commands.json: " + output);
    }
    objects.set(output, {...row, language: "ASM_MASM"});
  }
  if (objects.size !== closure.length) {throw new Error("LLVM 21 archive object closure has duplicate or unverified commands");}
  const archives = targets.map(target => {
    const file = target.artifact, stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || !contained(buildRoot, fs.realpathSync(file))) {
      throw new Error("LLVM 21 static archive is missing or escaped its build tree: " + file);
    }
    const proof = verifyCoffArchive(file), members = archiveInputs.get(file);
    return {...target, sha256: proof.sha256, bytes: stat.size, members: proof.members,
      objects: members, objectCount: members.length};
  });
  const objectRows = closure.map(file => {
    const row = objects.get(file);
    if (!row) {throw new Error("LLVM 21 archive object has no verified compiler ABI: " + file);}
    return row;
  });
  return {compileCommands: {path: path.resolve(compileCommands), sha256: compile.sha256, count: compile.count},
    archives, objects: objectRows};
}

/** @param {string} build @returns {{c: string, cxx: string}} */
function verifyIntelCompilerIdentity(build) {
  const directory = path.join(build, "CMakeFiles");
  verifyWindowsBuildPath(build, directory, "directory");
  const candidates = fs.readdirSync(directory).filter(name => /^\d+\.\d+(?:\.\d+)?$/.test(name)).map(name => path.join(directory, name));
  /** @param {string} language @param {RegExp} pattern */
  const read = (language, pattern) => {
    const file = candidates.map(root => path.join(root, "CMake" + language + "Compiler.cmake")).find(name => fs.existsSync(name));
    if (!file) {throw new Error("LLVM 21 CMake omitted its " + language + " compiler identity");}
    verifyWindowsBuildPath(build, file, "file");
    const text = fs.readFileSync(file, "utf8");
    if (!pattern.test(text)) {throw new Error("LLVM 21 CMake compiler is not the selected IntelLLVM driver");}
    return file;
  };
  return {c: read("C", /set\(CMAKE_C_COMPILER_ID "IntelLLVM"\)/),
    cxx: read("CXX", /set\(CMAKE_CXX_COMPILER_ID "IntelLLVM"\)/)};
}

/** @param {string} build @param {ReturnType<typeof windowsLlvmSdkPlan>} plan */
function verifyCMakeCache(build, plan) {
  const file = path.join(build, "CMakeCache.txt"), text = fs.readFileSync(file, "utf8"), values = new Map();
  for (const line of text.split(/\r?\n/)) {
    const entry = line.match(/^([^/#][^:]*):[^=]*=(.*)$/);
    if (entry) {values.set(entry[1], entry[2]);}
  }
  const expected = {CMAKE_C_COMPILER: plan.compilerTools.CMAKE_C_COMPILER, CMAKE_CXX_COMPILER: plan.compilerTools.CMAKE_CXX_COMPILER,
    CMAKE_LINKER: plan.compilerTools.CMAKE_LINKER, CMAKE_AR: plan.compilerTools.CMAKE_AR, CMAKE_RANLIB: plan.compilerTools.CMAKE_RANLIB,
    CMAKE_MAKE_PROGRAM: plan.tools.ninja, CMAKE_BUILD_TYPE: "MinSizeRel", CMAKE_MSVC_RUNTIME_LIBRARY: "MultiThreaded",
    CMAKE_PROJECT_TOP_LEVEL_INCLUDES: plan.sourceOptionsFile,
    CMAKE_C_COMPILER_TARGET: targetTriple, CMAKE_CXX_COMPILER_TARGET: targetTriple,
    CMAKE_ASM_MASM_COMPILER: plan.nativeTools["llvm-ml"], CMAKE_ASM_MASM_FLAGS: "/nologo --m64",
    CMAKE_INCLUDE_SYSTEM_FLAG_C: "/external:I", CMAKE_INCLUDE_SYSTEM_FLAG_CXX: "/external:I", LLVM_ENABLE_PROJECTS: "clang;lld",
    LLVM_TARGETS_TO_BUILD: "AMDGPU;NVPTX;X86", LLVM_ENABLE_EH: "ON", LLVM_ENABLE_RTTI: "ON", LLVM_ENABLE_PLUGINS: "OFF",
    LLVM_BUILD_TOOLS: "OFF", CLANG_BUILD_TOOLS: "OFF", LLD_BUILD_TOOLS: "OFF", BUILD_SHARED_LIBS: "OFF",
    LLVM_BUILD_LLVM_DYLIB: "OFF", LLVM_BUILD_LLVM_C_DYLIB: "OFF", LLVM_LINK_LLVM_DYLIB: "OFF", LIBCLANG_BUILD_STATIC: "ON"};
  for (const [name, value] of Object.entries(expected)) {
    if (values.get(name) !== value) {throw new Error("LLVM 21 CMake configuration drifted from its frozen plan: " + name);}
  }
  return file;
}

/** @param {string} output @param {unknown} value @param {boolean} [replace] */
function writeReceipt(output, value, replace = false) {
  const temporary = output + ".next";
  if (fs.existsSync(temporary) || (!replace && fs.existsSync(output))) {throw new Error("LLVM 21 receipt already exists");}
  fs.writeFileSync(temporary, JSON.stringify(value, null, 2) + "\n", {flag: "wx", mode: 0o600});
  fs.renameSync(temporary, output);
}

/** @param {string} file @param {string} command @param {string[]} args @param {string} cwd
 * @param {NodeJS.ProcessEnv} environment @param {number} [timeout] */
function runLogged(file, command, args, cwd, environment, timeout = phaseTimeout) {
  const fd = fs.openSync(file, "wx");
  try {
    const result = spawnSync(command, args, {cwd, env: environment, stdio: ["ignore", fd, fd], timeout, maxBuffer: 1 << 20});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error("Windows LLVM 21 phase failed; inspect " + file, {cause: result.error});
    }
  } finally {fs.closeSync(fd);}
}

/** @param {string} source @returns {string[]} */
function directoryFiles(source) {
  /** @type {string[]} */ const files = [];
  for (const root of [source]) {
    if (!fs.lstatSync(root).isDirectory()) {throw new Error("LLVM 21 include root is not a real directory: " + root);}
    /** @param {string} directory */
    const visit = directory => {
      for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
        const file = path.join(directory, entry.name);
        if (entry.isDirectory()) {visit(file);} else if (entry.isFile() || entry.isSymbolicLink()) {files.push(file);} else {throw new Error("Invalid LLVM 21 include input");}
        if (files.length > 65536) {throw new Error("LLVM 21 include inventory exceeds bounds");}
      }
    };
    visit(root);
  }
  return files;
}

/** Collect source and tool inputs frozen into the Windows LLVM SDK configure receipt.
 * @param {string} descriptorFile @param {string} archive @param {ReturnType<typeof windowsLlvmSdkPlan>} plan */
function windowsLlvmSdkInputFiles(descriptorFile, archive, plan) {
  const descriptor = validateWindowsLlvmSdkConfig(readBuildJson(descriptorFile));
  const includeFiles = directoryFiles(descriptor.tools.clangResourceInclude);
  const virtual = [...vfsFiles(descriptor.windowsSdk.overlay)].filter(([name]) => descriptor.windowsSdk.includeRoots.some(root =>
    name.toLowerCase().startsWith(root.replace(/\/+$/, "").toLowerCase() + "/")));
  return [descriptorFile, archive, tarTool, plan.tools.cmake, plan.tools.ninja, descriptor.tools.clangCl,
    descriptor.tools.lldLink, descriptor.windowsSdk.overlay, ...Object.values(plan.nativeTools), ...virtual.map(([, file]) => file),
    ...includeFiles, ...descriptor.windowsSdk.msvcLibraries, ...descriptor.windowsSdk.ucrtLibraries, ...descriptor.windowsSdk.importLibraries,
    ...recipeFiles.map(name => path.join(repo, name))];
}

/** Keep source/tool inputs and configure-generated frozen inputs identical across configure and audit.
 * @param {string} descriptorFile @param {string} archive @param {ReturnType<typeof windowsLlvmSdkPlan>} plan */
function windowsLlvmConfiguredInputFiles(descriptorFile, archive, plan) {
  return {base: windowsLlvmSdkInputFiles(descriptorFile, archive, plan), generated: {
    plan: path.join(plan.output, "plan.json"), archiveCopy: plan.archiveCopy, overlay: plan.overlay,
    ruleOverride: plan.ruleOverride, toolchainFile: plan.toolchainFile, sourceOptionsFile: plan.sourceOptionsFile}};
}

/** The build graph is part of the admission: Ninja rules and both compiler identities must
 * still be byte-for-byte current when a downstream SDK consumes the archive set.
 * @param {ReturnType<typeof windowsLlvmSdkPlan>} plan */
function windowsLlvmConfiguredFiles(plan) {
  const identity = verifyIntelCompilerIdentity(plan.build);
  const files = [path.join(plan.build, "CMakeCache.txt"), path.join(plan.build, "build.ninja"),
    path.join(plan.build, "CMakeFiles", "rules.ninja"), path.join(plan.build, "compile_commands.json"), identity.c, identity.cxx,
    plan.sourceOptionsFile];
  verifyWindowsSourceOptionsFile(plan);
  for (const file of files) {if (file !== plan.sourceOptionsFile) {verifyWindowsBuildPath(plan.build, file, "file");}}
  return buildFileInputs(files);
}

/** Ensure selected archive artifacts stay as ordinary files inside the original Ninja cache.
 * @param {string} build @param {StaticTarget[]} targets @param {boolean} [allowMissing] */
function verifyWindowsTargetPaths(build, targets, allowMissing = false) {
  for (const target of targets) {verifyWindowsBuildPath(build, target.artifact, "file", allowMissing);}
}

/** Create a source/tool/config-bound configure receipt from a fresh extraction of the pinned archive.
 * @param {string} descriptorFile @param {string} archive @param {string} output
 * @param {{cmake: string, ninja: string}} tools @param {number} jobs @param {NodeJS.ProcessEnv} environment */
function configureWindowsLlvmSdk(descriptorFile, archive, output, tools, jobs, environment = process.env) {
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows LLVM 21 cross-build requires Linux and MOM_AOT_BUILD_ONLY=1");}
  const descriptor = validateWindowsLlvmSdkConfig(readBuildJson(descriptorFile)), plan = windowsLlvmSdkPlan(descriptor, archive, output, tools, jobs);
  const target = path.join(fs.realpathSync(path.dirname(output)), path.basename(output));
  if (target !== output || fs.existsSync(output)) {throw new Error("Select a new canonical Windows LLVM 21 output directory");}
  for (const input of [repo, path.dirname(archive), descriptor.windowsSdk.sourceRoot, ...descriptor.windowsSdk.libraryRoots,
    path.dirname(descriptor.tools.clangCl), path.dirname(descriptor.tools.lldLink)]) {
    const real = fs.realpathSync(input);
    if (overlaps(real, target)) {throw new Error("Windows LLVM 21 output overlaps a source, tool or SDK input");}
  }
  const cleanEnvironment = windowsToolEnvironment(environment);
  for (const key of ["CC", "CXX", "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE", "CMAKE_GENERATOR", "CMAKE_MAKE_PROGRAM", "PKG_CONFIG_PATH"]) {delete cleanEnvironment[key];}
  const cmakeVersion = spawnSync(tools.cmake, ["--version"], {cwd: repo, env: cleanEnvironment, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 16});
  if (cmakeVersion.error || cmakeVersion.signal || cmakeVersion.status !== 0) {
    throw new Error("Windows LLVM 21 could not verify its selected CMake version", {cause: cmakeVersion.error});
  }
  verifyCmakeVersion((cmakeVersion.stdout || "") + (cmakeVersion.stderr || ""));
  if (fileHash(archive) !== archiveSha256) {throw new Error("Windows LLVM 21 source archive SHA256 mismatch");}
  const configuredInputFiles = windowsLlvmConfiguredInputFiles(descriptorFile, archive, plan),
    fingerprints = buildFileInputs(configuredInputFiles.base), planFile = configuredInputFiles.generated.plan,
    resultFile = path.join(output, "result.json"), build = plan.build;
  fs.mkdirSync(output, {mode: 0o700});
  const producerFile = path.join(repo, "scripts/build-native-windows-llvm-sdk.js"), producerSnapshotFile = path.join(output, "producer-snapshot.js");
  fs.copyFileSync(producerFile, producerSnapshotFile, fs.constants.COPYFILE_EXCL);
  const producerSnapshot = evidenceFile(producerSnapshotFile);
  if (producerSnapshot.sha256 !== fingerprints[producerFile]) {throw new Error("Windows LLVM 21 producer changed while taking its configure snapshot");}
  fs.mkdirSync(plan.source, {mode: 0o700});
  fs.copyFileSync(archive, plan.archiveCopy, fs.constants.COPYFILE_EXCL);
  if (fileHash(plan.archiveCopy) !== archiveSha256 || fileHash(archive) !== archiveSha256) {throw new Error("Pinned LLVM 21 archive changed while copying");}
  const catalog = verifyArchiveCatalog(plan.archiveCopy);
  const members = [...sourceDirectories.map(name => "llvm-project-21.1.8.src/" + name + "/"), "llvm-project-21.1.8.src/LICENSE.TXT"];
  runLogged(path.join(output, "extract.log"), plan.tarTool,
    ["-xJf", plan.archiveCopy, "--directory", plan.source, "--strip-components=1", "--no-same-owner", "--no-same-permissions", "--", ...members],
    output, cleanEnvironment, 1200000);
  if (fileHash(plan.archiveCopy) !== archiveSha256 || fileHash(archive) !== archiveSha256) {throw new Error("Pinned LLVM 21 archive changed during extraction");}
  const sourceState = llvmSourceSnapshot(plan.source);
  fs.writeFileSync(plan.overlay, plan.overlayText, {flag: "wx"});
  fs.writeFileSync(plan.ruleOverride, plan.ruleOverrideText, {flag: "wx"});
  fs.writeFileSync(plan.toolchainFile, plan.toolchain, {flag: "wx"});
  fs.writeFileSync(plan.sourceOptionsFile, plan.sourceOptionsText, {flag: "wx", mode: 0o600});
  const query = path.join(build, ".cmake", "api", "v1", "query", "codemodel-v2");
  fs.mkdirSync(path.dirname(query), {recursive: true});
  fs.writeFileSync(query, "", {flag: "wx"});
  const savedPlan = {descriptorFile, archive, source: plan.source, output, tools: {...tools}, jobs,
    sourceOptionsFile: plan.sourceOptionsFile, sourceOptionsText: plan.sourceOptionsText};
  fs.writeFileSync(planFile, JSON.stringify(savedPlan, null, 2) + "\n", {flag: "wx"});
  for (const file of Object.values(configuredInputFiles.generated)) {fingerprints[file] = fileHash(file);}
  const recipeSha256 = recipeHash(plan, sourceState, catalog);
  const receipt = {version: 1, role: plan.role, runId: randomUUID(), stage: "configuring", plan: savedPlan,
    planSha256: fileHash(planFile), sourceState, catalog, fingerprints, recipeSha256, configure: plan.configure,
    producerSnapshot, buildOnly: true, compiled: false, executed: false, publicationQualified: false};
  writeReceipt(resultFile, receipt);
  runLogged(path.join(output, "configure.log"), tools.cmake, plan.configure.slice(1), repo, cleanEnvironment, 1200000);
  const compilerIdentity = verifyIntelCompilerIdentity(build), targets = llvmStaticArchiveTargets(build, plan.source);
  const targetGraph = verifyBuildPhases(build), compileCommands = path.join(build, "compile_commands.json");
  verifyCMakeCache(build, plan);
  const sourceOptions = verifyWindowsSourceOptionsFile(plan), hostSourceFlag = verifyAmdgpulibCallsCompileFlag(build, plan.source);
  const configuredFiles = windowsLlvmConfiguredFiles(plan);
  const result = {...receipt, stage: "configured", compilerIdentity, targets, phases: targetGraph.phases,
    sourceOptions, hostSourceFlag, configuredFiles, compileCommandsSha256: fileHash(compileCommands),
    outputs: jitBuildOutputs(build)};
  if (!isDeepStrictEqual(sourceState, llvmSourceSnapshot(plan.source)) ||
      Object.entries(fingerprints).some(([file, hash]) => fileHash(file) !== hash) || fileHash(archive) !== archiveSha256) {
    throw new Error("LLVM 21 source or frozen input changed during CMake configuration");
  }
  writeReceipt(resultFile, result, true);
  return result;
}

/** @param {ReturnType<typeof windowsLlvmSdkPlan>} plan @param {SourceState} sourceState @param {ArchiveCatalog} catalog */
function recipeHash(plan, sourceState, catalog) {
  return createHash("sha256").update(JSON.stringify({configure: plan.configure, toolchain: plan.toolchain,
    rules: plan.ruleOverrideText, overlay: plan.overlayText, sourceOptionsFile: plan.sourceOptionsFile,
    sourceOptionsText: plan.sourceOptionsText, sourceState, catalog})).digest("hex");
}

/** @param {WindowsLlvmReceipt} receipt @param {ReturnType<typeof windowsLlvmSdkPlan>} plan
 * @param {Record<string, string>} [expectedOutputs] @returns {{receipt: WindowsLlvmReceipt}} */
function verifySession(receipt, plan, expectedOutputs = receipt.outputs) {
  verifyWindowsCacheRoots(plan);
  for (const file of Object.keys(receipt.configuredFiles || {})) {
    if (file === plan.sourceOptionsFile) {verifyWindowsSourceOptionsFile(plan);} else {verifyWindowsBuildPath(plan.build, file, "file");}
  }
  verifyWindowsBuildPath(plan.build, path.join(plan.build, "compile_commands.json"), "file");
  const planMatches = isDeepStrictEqual(receipt.plan, {descriptorFile: receipt.plan.descriptorFile, archive: plan.archive, source: plan.source,
    output: plan.output, tools: plan.tools, jobs: plan.jobs, sourceOptionsFile: plan.sourceOptionsFile, sourceOptionsText: plan.sourceOptionsText});
  if (receipt.version !== 1 || receipt.role !== plan.role || receipt.stage !== "configured" || receipt.buildOnly !== true ||
      receipt.compiled === true || receipt.executed !== false ||
      receipt.publicationQualified !== false || !receipt.runId || !receipt.fingerprints || !receipt.outputs ||
      !Array.isArray(receipt.targets) || !Array.isArray(receipt.phases) || !receipt.sourceState || !receipt.catalog ||
      !planMatches) {
    throw new Error("Windows LLVM 21 same-run phase receipt is missing or does not match its frozen plan");
  }
  if (!windowsBuildInputsCurrent(receipt.fingerprints)) {throw new Error("Windows LLVM 21 input changed; do not restamp or resume");}
  const producerFile = path.join(repo, "scripts/build-native-windows-llvm-sdk.js");
  verifyEvidenceFile(receipt.producerSnapshot, "configure producer");
  if (receipt.fingerprints[producerFile] !== receipt.producerSnapshot.sha256) {
    throw new Error("Windows LLVM 21 producer snapshot does not match the configure input");
  }
  if (fileHash(path.join(plan.output, "plan.json")) !== receipt.planSha256) {
    throw new Error("Windows LLVM 21 build outputs changed since the last completed phase; use a new output directory");
  }
  if (!isDeepStrictEqual(jitBuildOutputs(plan.build), expectedOutputs)) {
    throw new Error("Windows LLVM 21 build outputs changed since the last completed phase; use a new output directory");
  }
  if (!isDeepStrictEqual(receipt.sourceOptions, verifyWindowsSourceOptionsFile(plan)) ||
      !isDeepStrictEqual(receipt.hostSourceFlag, verifyAmdgpulibCallsCompileFlag(plan.build, plan.source))) {
    throw new Error("Windows LLVM 21 source-specific host option is missing or changed");
  }
  if (!isDeepStrictEqual(llvmSourceSnapshot(plan.source), receipt.sourceState)) {throw new Error("LLVM 21 sources changed between build phases");}
  if (receipt.recipeSha256 !== recipeHash(plan, receipt.sourceState, receipt.catalog) ||
      !isDeepStrictEqual(receipt.phases, buildPhases)) {throw new Error("Windows LLVM 21 recipe or phase list changed; use a new output directory");}
  return {receipt};
}

/** @param {string} output @param {BuildCheckpoint} value */
function writeBuildCheckpoint(output, value) {writeReceipt(path.join(output, "build-state.json"), value, true);}

/** @param {string} output @param {WindowsLlvmReceipt} receipt @param {ReturnType<typeof windowsLlvmSdkPlan>} plan
 * @returns {BuildCheckpoint} */
function readBuildCheckpoint(output, receipt, plan) {
  const file = path.join(output, "build-state.json");
  if (!fs.existsSync(file)) {
    return {version: 1, role: "windows-llvm21-build-checkpoint", runId: receipt.runId,
      configuredReceipt: evidenceFile(path.join(output, "result.json")), producerSnapshot: receipt.producerSnapshot,
      nextPhase: 0, completedPhases: [], phaseProofs: [], outputs: receipt.outputs, stage: "configured"};
  }
  const state = /** @type {BuildCheckpoint} */ (readBuildJson(file));
  if (state.version !== 1 || state.role !== "windows-llvm21-build-checkpoint" || state.runId !== receipt.runId ||
      !isDeepStrictEqual(state.configuredReceipt, evidenceFile(path.join(output, "result.json"))) ||
      !isDeepStrictEqual(state.producerSnapshot, receipt.producerSnapshot) ||
      !Number.isInteger(state.nextPhase) || (state.nextPhase || 0) < 0 || (state.nextPhase || 0) > buildPhases.length ||
      !Array.isArray(state.completedPhases) || !Array.isArray(state.phaseProofs) ||
      state.completedPhases.length !== state.nextPhase || state.phaseProofs.length !== state.nextPhase ||
      !isDeepStrictEqual(state.completedPhases, buildPhases.slice(0, Number(state.nextPhase)).map(row => row.name)) ||
      !["configured", "building", "built", "failed"].includes(String(state.stage)) ||
      state.stage === "failed" || state.stage !== (state.nextPhase === 0 ? "configured" :
    state.nextPhase === buildPhases.length ? "built" : "building")) {
    throw new Error("Windows LLVM 21 build checkpoint is invalid or terminal; use a new output directory");
  }
  verifyCurrentPhaseProofs(state.phaseProofs, plan, Number(state.nextPhase));
  return state;
}

/** @param {string} output @param {WindowsLlvmReceipt} receipt @param {ReturnType<typeof windowsLlvmSdkPlan>} plan
 * @param {BuildCheckpoint} state */
function writeCurrentResult(output, receipt, plan, state) {
  if (state.stage !== "built" || state.nextPhase !== buildPhases.length || !Array.isArray(state.phaseProofs) ||
      state.phaseProofs.length !== buildPhases.length || !receipt.producerSnapshot) {
    throw new Error("Windows LLVM 21 cannot publish a current result before all six phases pass");
  }
  const compileCommandsFile = path.join(plan.build, "compile_commands.json"), compileCommands = readBuildJson(compileCommandsFile);
  const sourceOptions = verifyWindowsSourceOptionsFile(plan), hostSourceFlag = verifyAmdgpulibCallsCompileFlag(plan.build, plan.source);
  if (!isDeepStrictEqual(sourceOptions, receipt.sourceOptions) || !isDeepStrictEqual(hostSourceFlag, receipt.hostSourceFlag)) {
    throw new Error("Windows LLVM 21 current result source-option evidence changed after configuration");
  }
  const current = {version: 1, role: "windows-llvm21-current-result", generationId: receipt.runId, output,
    originalReceipt: evidenceFile(path.join(output, "result.json")), producerSnapshot: receipt.producerSnapshot,
    sourceState: receipt.sourceState, inputs: receipt.fingerprints, configuredFiles: receipt.configuredFiles,
    sourceOptions, hostSourceFlag,
    compileCommands: {path: compileCommandsFile, sha256: fileHash(compileCommandsFile), count: compileCommands.length},
    phaseProofs: state.phaseProofs, buildOnly: true, compiled: true, executed: false, publicationQualified: false};
  writeReceipt(path.join(output, "result-current.json"), current);
  return current;
}

/** @param {string} output @param {WindowsLlvmReceipt} receipt @param {ReturnType<typeof windowsLlvmSdkPlan>} plan
 * @param {BuildPhase} phase @param {NodeJS.ProcessEnv} environment */
function runBuildPhase(output, receipt, plan, phase, environment) {
  const args = ["--build", plan.build, "--target", phase.target, "--parallel", String(plan.jobs), "--", "-k", "0"];
  const logFile = path.join(output, "build-" + phase.name + "-" + receipt.runId + ".log");
  const fd = fs.openSync(logFile, "wx"); let result;
  try {result = spawnSync(plan.tools.cmake, args, {cwd: repo, env: environment, stdio: ["ignore", fd, fd], timeout: phaseTimeout, maxBuffer: 1 << 20});} finally {fs.closeSync(fd);}
  const record = {phase, command: plan.tools.cmake, args, status: result.status, signal: result.signal || null,
    ...(result.error ? {error: String(result.error.message)} : {}), pass: !result.error && !result.signal && result.status === 0,
    log: evidenceFile(logFile), buildOnly: true, executed: false, publicationQualified: false};
  const proofFile = path.join(output, "phase-" + phase.name + "-" + receipt.runId + ".json");
  writeReceipt(proofFile, record);
  return {record, reference: {...evidenceFile(proofFile), name: phase.name, target: phase.target, log: record.log}};
}

/** Run exactly one named upstream aggregate per invocation. Completed phases can be resumed;
 * interrupted phases have no receipt and cannot be blessed in place.
 * @param {string} output @param {string} phaseName @param {NodeJS.ProcessEnv} environment */
function buildWindowsLlvmSdk(output, phaseName, environment = process.env) {
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows LLVM 21 build requires Linux and MOM_AOT_BUILD_ONLY=1");}
  const resultFile = path.join(output, "result.json"), current = /** @type {WindowsLlvmReceipt} */ (readBuildJson(resultFile)), saved = current.plan;
  if (!saved || typeof saved.descriptorFile !== "string") {throw new Error("Windows LLVM 21 configure receipt is missing its source descriptor");}
  const descriptor = validateWindowsLlvmSdkConfig(readBuildJson(saved.descriptorFile));
  const plan = windowsLlvmSdkPlan(descriptor, saved.archive, saved.output, saved.tools, saved.jobs);
  const checkpoint = readBuildCheckpoint(output, current, plan), expectedOutputs = checkpoint.outputs;
  const {receipt} = verifySession(current, plan, expectedOutputs);
  const cleanEnvironment = windowsToolEnvironment(environment);
  for (const key of ["CC", "CXX", "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE", "CMAKE_GENERATOR", "CMAKE_MAKE_PROGRAM", "PKG_CONFIG_PATH"]) {delete cleanEnvironment[key];}
  if (!["configured", "building"].includes(String(checkpoint.stage))) {throw new Error("Select a completed LLVM 21 configure phase before building");}
  const configFiles = receipt.configuredFiles;
  if (!configFiles || Object.entries(configFiles).some(([file, hash]) => fileHash(file) !== hash) ||
      fileHash(path.join(plan.build, "compile_commands.json")) !== receipt.compileCommandsSha256) {
    throw new Error("LLVM 21 configure, compile database or CMake target graph changed");
  }
  const phases = receipt.phases, phaseIndex = Number(checkpoint.nextPhase || 0), selected = phases[phaseIndex];
  if (!Number.isInteger(phaseIndex) || !selected || selected.name !== phaseName || !buildPhases.some(row => row.name === phaseName)) {
    throw new Error("Select the next named LLVM 21 aggregate phase in order");
  }
  const {record, reference} = runBuildPhase(output, receipt, plan, selected, cleanEnvironment);
  const nextPhase = Number(checkpoint.nextPhase || 0) + 1, completedPhases = [.../** @type {string[]} */ (checkpoint.completedPhases), phaseName];
  const phaseProofs = [.../** @type {CurrentPhaseProof[]} */ (checkpoint.phaseProofs), reference];
  const outputs = jitBuildOutputs(plan.build);
  verifyWindowsTargetPaths(plan.build, receipt.targets, true);
  const inputsCurrent = isDeepStrictEqual(llvmSourceSnapshot(plan.source), receipt.sourceState) &&
    Object.entries(receipt.fingerprints).every(([file, hash]) => fileHash(file) === hash) &&
    Object.entries(configFiles).every(([file, hash]) => fileHash(file) === hash);
  const noPeTools = !Object.keys(outputs).some(name => /\.(?:exe|dll)$/i.test(name));
  const phasePass = record.pass && inputsCurrent && noPeTools;
  const nextCheckpoint = {...checkpoint, nextPhase, completedPhases, phaseProofs, outputs,
    stage: /** @type {BuildCheckpoint["stage"]} */ (phasePass ? (nextPhase === phases.length ? "built" : "building") : "failed"),
    ...(phasePass ? {} : {error: !record.pass ? (record.error || (record.signal ? "signal " + record.signal : "exit " + record.status)) :
      !inputsCurrent ? "LLVM 21 source, tool inputs or configure outputs changed during a build phase" : "unexpected PE tool or DLL in build output"})};
  writeBuildCheckpoint(output, nextCheckpoint);
  if (!record.pass) {throw new Error("Windows LLVM 21 phase failed; inspect " + reference.log.path);}
  if (!inputsCurrent) {
    throw new Error("LLVM 21 inputs or configure outputs changed during a build phase");
  }
  if (!noPeTools) {
    throw new Error("Windows LLVM 21 aggregate produced a target executable or DLL");
  }
  if (nextCheckpoint.stage === "built") {
    verifyWindowsTargetPaths(plan.build, receipt.targets);
    writeCurrentResult(output, receipt, plan, nextCheckpoint);
  }
  return nextCheckpoint;
}

/** Collect names only during revalidation; the object/archive receipts below carry content
 * hashes, so repeatedly reading every compiler object just to detect a newly added PE tool is
 * unnecessary. @param {string} build */
function windowsLlvmOutputNames(build) {
  if (!literal(build) || !fs.lstatSync(build).isDirectory()) {throw new Error("Select a real LLVM 21 build directory");}
  /** @type {string[]} */ const names = []; let entries = 0;
  /** @param {string} directory @param {number} depth */
  function visit(directory, depth) {
    if (depth > 32) {throw new Error("LLVM 21 build output nesting exceeds bounds");}
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(directory, entry.name), name = path.relative(build, file).split(path.sep).join("/");
      if (++entries > 65536 || name.length > 4096 || /[\r\n\0]/.test(name)) {throw new Error("LLVM 21 build output inventory exceeds bounds");}
      if (entry.isDirectory()) {visit(file, depth + 1);} else if (entry.isFile() || entry.isSymbolicLink()) {names.push(name);} else {throw new Error("LLVM 21 build output is not a file or directory");}
    }
  }
  visit(build, 0);
  return names.sort();
}

/** Return a stable path/hash record for one generated evidence file. @param {string} file */
function evidenceFile(file) {return {path: file, sha256: fileHash(file)};}

/** @typedef {{path: string, sha256: string}} EvidenceFile */
/** @typedef {EvidenceFile & {name: string, target: string, log: EvidenceFile}} CurrentPhaseProof */
/** @typedef {{version: number, role: string, runId: string, configuredReceipt: EvidenceFile, producerSnapshot: EvidenceFile,
 * nextPhase: number, completedPhases: string[], phaseProofs: CurrentPhaseProof[], outputs: Record<string, string>,
 * stage: "configured" | "building" | "built" | "failed", error?: string}} BuildCheckpoint */
/** @typedef {{version: number, role: string, generationId: string, output: string, originalReceipt: EvidenceFile,
 * producerSnapshot: EvidenceFile, sourceState: SourceState, inputs: Record<string, string>, configuredFiles: Record<string, string>,
 * sourceOptions: EvidenceFile, hostSourceFlag: {path: string, flag: string, compileCommands: number},
 * compileCommands: {path: string, sha256: string, count: number}, phaseProofs: CurrentPhaseProof[], buildOnly: boolean,
 * compiled: boolean, executed: boolean, publicationQualified: boolean}} CurrentLlvmResult */
/** @typedef {{version: number, generationId: string, sourceState: SourceState,
 * objects: ReturnType<typeof auditWindowsLlvmArchives>["objects"]}} WindowsLlvmObjectProof */
/** @typedef {{version: number, role: string, generationId: string, originalReceipt: EvidenceFile, currentResult: EvidenceFile,
 * sourceState: SourceState, producerSnapshot: EvidenceFile, phaseProofs: CurrentPhaseProof[], configuredFiles: Record<string, string>,
 * sourceOptions: EvidenceFile, hostSourceFlag: {path: string, flag: string, compileCommands: number},
 * compileCommands: {path: string, sha256: string, count: number}, archiveCount: number,
 * archives: ReturnType<typeof auditWindowsLlvmArchives>["archives"], objectProof: EvidenceFile & {count: number},
 * compilerAbi: Record<string, {count: number, compilerIdentity: string, abiSha256: string[]}>,
 * outputInventory: {count: number, sha256: string}, buildOnly: boolean, compiled: boolean, executed: boolean,
 * publicationQualified: boolean}} WindowsLlvmAdmission */

/** Accept only explicit, immutable evidence references; generation layout belongs to the runner.
 * @param {unknown} value @param {string} label @returns {EvidenceFile} */
function verifyEvidenceFile(value, label) {
  const record = /** @type {EvidenceFile | null} */ (value);
  if (!record || !literal(record.path) || path.normalize(record.path) !== record.path || !/^[0-9a-f]{64}$/.test(record.sha256)) {
    throw new Error("Invalid Windows LLVM 21 " + label + " reference");
  }
  const stat = fs.lstatSync(record.path, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || fileHash(record.path) !== record.sha256) {
    throw new Error("Windows LLVM 21 " + label + " changed: " + record.path);
  }
  return record;
}

/** @param {CurrentPhaseProof[]} phaseProofs @param {ReturnType<typeof windowsLlvmSdkPlan>} plan @param {number} [phaseCount] */
function verifyCurrentPhaseProofs(phaseProofs, plan, phaseCount = buildPhases.length) {
  if (!Array.isArray(phaseProofs) || phaseProofs.length !== phaseCount) {
    throw new Error("Windows LLVM 21 current result omitted one or more build phases");
  }
  for (const [index, phase] of buildPhases.slice(0, phaseCount).entries()) {
    const reference = phaseProofs[index];
    if (!reference || reference.name !== phase.name || reference.target !== phase.target) {throw new Error("Windows LLVM 21 phase order changed");}
    const proof = verifyEvidenceFile(reference, phase.name + " phase proof");
    const record = /** @type {{phase?: BuildPhase, command?: string, args?: string[], status?: number | null, signal?: string | null,
      error?: string, pass?: boolean, log?: EvidenceFile, buildOnly?: boolean, executed?: boolean, publicationQualified?: boolean}} */
      (readBuildJson(proof.path));
    const expectedArgs = ["--build", plan.build, "--target", phase.target, "--parallel", String(plan.jobs), "--", "-k", "0"];
    if (!isDeepStrictEqual(record.phase, phase) || record.command !== plan.tools.cmake || !isDeepStrictEqual(record.args, expectedArgs) ||
        record.status !== 0 || record.signal !== null || record.error || record.pass !== true || record.buildOnly !== true ||
        record.executed !== false || record.publicationQualified !== false ||
        !isDeepStrictEqual(verifyEvidenceFile(reference.log, phase.name + " log"),
          verifyEvidenceFile(record.log, phase.name + " phase log"))) {
      throw new Error("Windows LLVM 21 phase proof is incomplete or unsuccessful: " + phase.name);
    }
  }
}

/** Verify the current-run record without depending on a runner's private generation directory.
 * @param {string} output @param {WindowsLlvmReceipt} receipt @param {ReturnType<typeof windowsLlvmSdkPlan>} plan
 */
function verifyWindowsLlvmCurrentResult(output, receipt, plan) {
  verifyWindowsCacheRoots(plan);
  const configuredFiles = windowsLlvmConfiguredFiles(plan), compileCommandsFile = path.join(plan.build, "compile_commands.json");
  verifyWindowsBuildPath(plan.build, compileCommandsFile, "file");
  const resultFile = path.join(output, "result-current.json"), current = /** @type {CurrentLlvmResult} */ (readBuildJson(resultFile));
  if (current.version !== 1 || current.role !== "windows-llvm21-current-result" || current.output !== output ||
      !/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(current.generationId) || current.buildOnly !== true || current.compiled !== true ||
      current.executed !== false || current.publicationQualified !== false || receipt.stage !== "configured" ||
      receipt.buildOnly !== true || receipt.compiled === true || receipt.executed !== false || receipt.publicationQualified !== false ||
      !isDeepStrictEqual(current.originalReceipt, evidenceFile(path.join(output, "result.json"))) ||
      !isDeepStrictEqual(current.sourceState, receipt.sourceState) ||
      !isDeepStrictEqual(current.sourceOptions, receipt.sourceOptions) ||
      !isDeepStrictEqual(current.hostSourceFlag, receipt.hostSourceFlag) ||
      !isDeepStrictEqual(current.sourceOptions, verifyWindowsSourceOptionsFile(plan)) ||
      !isDeepStrictEqual(current.hostSourceFlag, verifyAmdgpulibCallsCompileFlag(plan.build, plan.source)) ||
      !isDeepStrictEqual(current.compileCommands, {path: path.join(plan.build, "compile_commands.json"),
        sha256: fileHash(path.join(plan.build, "compile_commands.json")), count: readBuildJson(path.join(plan.build, "compile_commands.json")).length}) ||
      !isDeepStrictEqual(current.configuredFiles, configuredFiles)) {
    throw new Error("Windows LLVM 21 current result does not match its original configured receipt or build graph");
  }
  verifyEvidenceFile(current.producerSnapshot, "frozen producer");
  if (!isDeepStrictEqual(llvmSourceSnapshot(plan.source), current.sourceState)) {throw new Error("LLVM 21 source tree changed after compilation");}
  verifyCMakeCache(plan.build, plan);
  const producerFile = path.join(repo, "scripts/build-native-windows-llvm-sdk.js"), recordedInputs = current.inputs;
  const inputFiles = windowsLlvmConfiguredInputFiles(receipt.plan.descriptorFile, plan.archive, plan);
  if (!recordedInputs || !Object.hasOwn(recordedInputs, producerFile) || recordedInputs[producerFile] !== current.producerSnapshot.sha256) {
    throw new Error("Windows LLVM 21 selected inputs are not bound to the frozen producer");
  }
  const otherInputs = Object.fromEntries(Object.entries(recordedInputs).filter(([file]) => file !== producerFile));
  const selectedInputs = buildFileInputs([...inputFiles.base, ...Object.values(inputFiles.generated)]
    .filter(file => file !== producerFile));
  if (!isDeepStrictEqual(otherInputs, selectedInputs)) {throw new Error("Windows LLVM 21 selected source or tool input changed");}
  verifyCurrentPhaseProofs(current.phaseProofs, plan);
  const targets = llvmStaticArchiveTargets(plan.build, plan.source);
  if (targets.length !== expectedStaticArchiveCount || !isDeepStrictEqual(targets, receipt.targets)) {
    throw new Error("Windows LLVM 21 configured static target graph changed");
  }
  return {current, resultFile, targets};
}

/** Summarize ABI classes while each admitted object retains its detailed ABI digest.
 * @param {ReturnType<typeof auditWindowsLlvmArchives>["objects"]} objects */
function windowsLlvmCompilerAbi(objects) {
  return Object.fromEntries([...new Set(objects.map(row => row.language))].sort().map(language => {
    const rows = objects.filter(row => row.language === language);
    const first = rows[0];
    if (!first) {throw new Error("Windows LLVM 21 compiler ABI class is empty");}
    return [language, {count: rows.length, compilerIdentity: first.compilerIdentity,
      abiSha256: [...new Set(rows.map(row => row.abiSha256))].sort()}];
  }));
}

/** Create the COFF archive/object receipt from the current result record; result.json is preserved.
 * @param {string} output @param {NodeJS.ProcessEnv} [environment] */
function auditWindowsLlvmAdmission(output, environment = process.env) {
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows LLVM 21 audit requires Linux and MOM_AOT_BUILD_ONLY=1");}
  if (!literal(output) || path.normalize(output) !== output || fs.realpathSync(output) !== output) {throw new Error("Select a canonical Windows LLVM 21 output directory");}
  const receiptFile = path.join(output, "result.json"), currentFile = path.join(output, "result-current.json"),
    admissionFile = path.join(output, "admitted.json"), objectsFile = path.join(output, "admitted-objects.json");
  if ([admissionFile, objectsFile, admissionFile + ".next", objectsFile + ".next"].some(file => fs.existsSync(file))) {
    throw new Error("Windows LLVM 21 admission already exists; do not restamp it");
  }
  const receipt = /** @type {WindowsLlvmReceipt} */ (readBuildJson(receiptFile));
  if (!receipt.plan || receipt.plan.output !== output) {throw new Error("Windows LLVM 21 configured receipt is missing its plan");}
  const descriptor = validateWindowsLlvmSdkConfig(readBuildJson(receipt.plan.descriptorFile));
  const plan = windowsLlvmSdkPlan(descriptor, receipt.plan.archive, output, receipt.plan.tools, receipt.plan.jobs);
  verifyWindowsCacheRoots(plan);
  if (!fs.existsSync(currentFile)) {
    const checkpoint = readBuildCheckpoint(output, receipt, plan);
    if (checkpoint.stage !== "built" || !isDeepStrictEqual(checkpoint.outputs, jitBuildOutputs(plan.build))) {
      throw new Error("Windows LLVM 21 audit needs a complete successful six-phase checkpoint");
    }
    writeCurrentResult(output, receipt, plan, checkpoint);
  }
  const {current} = verifyWindowsLlvmCurrentResult(output, receipt, plan), names = windowsLlvmOutputNames(plan.build);
  if (names.some(name => /\.(?:exe|dll)$/i.test(name))) {throw new Error("Windows LLVM 21 output contains a PE tool or DLL");}
  const audited = auditWindowsLlvmArchives(plan.tools.ninja, plan.build, plan.source, descriptor.tools.clangCl,
    plan.nativeTools["llvm-ml"], path.join(plan.build, "compile_commands.json"), receipt.targets);
  if (audited.archives.length !== expectedStaticArchiveCount || !audited.objects.length) {throw new Error("Windows LLVM 21 archive closure is incomplete");}
  const objectData = {version: 1, generationId: current.generationId, sourceState: current.sourceState, objects: audited.objects};
  writeReceipt(objectsFile, objectData);
  const admission = {version: 1, role: "windows-llvm21-static-host-admission", generationId: current.generationId,
    originalReceipt: current.originalReceipt, currentResult: evidenceFile(currentFile), producerSnapshot: current.producerSnapshot,
    sourceState: current.sourceState, phaseProofs: current.phaseProofs, configuredFiles: current.configuredFiles,
    sourceOptions: current.sourceOptions, hostSourceFlag: current.hostSourceFlag,
    compileCommands: current.compileCommands, archiveCount: audited.archives.length, archives: audited.archives,
    objectProof: {...evidenceFile(objectsFile), count: audited.objects.length}, compilerAbi: windowsLlvmCompilerAbi(audited.objects),
    outputInventory: {count: names.length, sha256: createHash("sha256").update(JSON.stringify(names)).digest("hex")},
    buildOnly: true, compiled: true, executed: false, publicationQualified: false};
  writeReceipt(admissionFile, admission);
  return {receipt, admission, receiptFile, admissionFile, objectsFile};
}

/** Re-derive the real Ninja object closure and verify every admitted object/archive byte.
 * @param {string} output @param {NodeJS.ProcessEnv} [environment] */
function verifyWindowsLlvmAdmission(output, environment = process.env) {
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows LLVM 21 admission requires Linux and MOM_AOT_BUILD_ONLY=1");}
  if (!literal(output) || path.normalize(output) !== output || fs.realpathSync(output) !== output) {throw new Error("Select a canonical Windows LLVM 21 output directory");}
  const receiptFile = path.join(output, "result.json"), currentFile = path.join(output, "result-current.json"),
    admissionFile = path.join(output, "admitted.json"), objectsFile = path.join(output, "admitted-objects.json");
  const receipt = /** @type {WindowsLlvmReceipt} */ (readBuildJson(receiptFile));
  if (!receipt.plan || receipt.plan.output !== output) {throw new Error("Windows LLVM 21 configured receipt is missing its plan");}
  const descriptor = validateWindowsLlvmSdkConfig(readBuildJson(receipt.plan.descriptorFile));
  const plan = windowsLlvmSdkPlan(descriptor, receipt.plan.archive, output, receipt.plan.tools, receipt.plan.jobs);
  const checkpoint = readBuildCheckpoint(output, receipt, plan);
  if (checkpoint.stage !== "built") {throw new Error("Windows LLVM 21 admission needs a complete successful build checkpoint");}
  // The configure snapshot predates compiled objects; only the completed checkpoint can
  // describe a built cache. Recheck its phase proofs and bytes, never the old empty tree.
  verifySession(receipt, plan, checkpoint.outputs);
  const {current, targets} = verifyWindowsLlvmCurrentResult(output, receipt, plan);
  const admission = /** @type {WindowsLlvmAdmission} */ (readBuildJson(admissionFile));
  const objects = /** @type {WindowsLlvmObjectProof} */ (readBuildJson(objectsFile));
  if (admission.version !== 1 || admission.role !== "windows-llvm21-static-host-admission" ||
      admission.generationId !== current.generationId || admission.buildOnly !== true || admission.compiled !== true ||
      admission.executed !== false || admission.publicationQualified !== false ||
      !isDeepStrictEqual(admission.originalReceipt, current.originalReceipt) ||
      !isDeepStrictEqual(admission.currentResult, evidenceFile(currentFile)) ||
      !isDeepStrictEqual(admission.producerSnapshot, current.producerSnapshot) ||
      !isDeepStrictEqual(admission.sourceState, current.sourceState) || !isDeepStrictEqual(admission.phaseProofs, current.phaseProofs) ||
      !isDeepStrictEqual(admission.sourceOptions, current.sourceOptions) || !isDeepStrictEqual(admission.hostSourceFlag, current.hostSourceFlag) ||
      !isDeepStrictEqual(admission.configuredFiles, current.configuredFiles) || !isDeepStrictEqual(admission.compileCommands, current.compileCommands) ||
      admission.archiveCount !== expectedStaticArchiveCount || !Array.isArray(admission.archives) || admission.archives.length !== expectedStaticArchiveCount ||
      objects.version !== 1 || objects.generationId !== current.generationId || !isDeepStrictEqual(objects.sourceState, current.sourceState) ||
      !Array.isArray(objects.objects) || objects.objects.length !== admission.objectProof.count ||
      !isDeepStrictEqual(admission.objectProof, {...evidenceFile(objectsFile), count: objects.objects.length})) {
    throw new Error("Windows LLVM 21 current-result and admission records do not form one proof");
  }
  const names = windowsLlvmOutputNames(plan.build);
  if (names.some(name => /\.(?:exe|dll)$/i.test(name))) {throw new Error("Windows LLVM 21 output tree contains a PE tool or DLL");}
  if (!isDeepStrictEqual({count: names.length, sha256: createHash("sha256").update(JSON.stringify(names)).digest("hex")}, admission.outputInventory)) {
    throw new Error("Windows LLVM 21 output name inventory changed from its admitted checkpoint");
  }
  const audited = auditWindowsLlvmArchives(plan.tools.ninja, plan.build, plan.source, descriptor.tools.clangCl,
    plan.nativeTools["llvm-ml"], path.join(plan.build, "compile_commands.json"), targets);
  if (!isDeepStrictEqual(audited.archives, admission.archives) || !isDeepStrictEqual(audited.objects, objects.objects) ||
      !isDeepStrictEqual(windowsLlvmCompilerAbi(audited.objects), admission.compilerAbi)) {
    throw new Error("Windows LLVM 21 archive, object or compiler ABI proof changed");
  }
  return {receipt, admission, receiptFile, admissionFile, objectsFile};
}

module.exports = {archiveSha256, buildPhases, windowsLlvmSdkPlan, llvmSourceSnapshot, verifyArchiveCatalog,
  llvmStaticArchiveTargets, verifyBuildPhases, auditCompileCommands, verifyIntelCompilerIdentity, verifyCMakeCache,
  configureWindowsLlvmSdk, buildWindowsLlvmSdk, auditWindowsLlvmAdmission,
  windowsLlvmSdkInputFiles, windowsLlvmConfiguredInputFiles, windowsLlvmConfiguredFiles,
  verifyAmdgpulibCallsCompileFlag, verifyWindowsLlvmAdmission};

if (require.main === module) {
  try {
    const [phase, ...args] = process.argv.slice(2);
    if (phase === "configure" && args.length === 6) {
      const [descriptor, archive, output, cmake, ninja, jobs] = args;
      if (!descriptor || !archive || !output || !cmake || !ninja || !jobs) {throw new Error("Incomplete Windows LLVM 21 configure arguments");}
      configureWindowsLlvmSdk(descriptor, archive, output, {cmake, ninja}, Number(jobs));
    } else if (phase === "build" && args.length === 2) {
      const [output, name] = args;
      if (!output || !name) {throw new Error("Incomplete Windows LLVM 21 build arguments");}
      buildWindowsLlvmSdk(output, name);
    } else if (phase === "audit" && args.length === 1) {
      const output = args[0];
      if (!output) {throw new Error("Missing Windows LLVM 21 audit output");}
      auditWindowsLlvmAdmission(output);
    } else {throw new Error("Usage: build-native-windows-llvm-sdk.js configure /descriptor.json /llvm-project-21.1.8.src.tar.xz /new-output /cmake /ninja /jobs | build /configured-output phase | audit /configured-output");}
  } catch (error) {console.error(error instanceof Error ? error.message : "Windows LLVM 21 SDK build failed"); process.exitCode = 1;}
}
