"use strict";

// Build the public Linux LLVM 21 static SDK from its pinned upstream source archive.
// Distribution membership and phase targets come from CMake's generated graph.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {createHash, randomUUID} = require("node:crypto"), {isDeepStrictEqual} = require("node:util");
const {hasExactBuildFields, buildPathsOverlap, containsBuildPath, fileHash, buildFileInputs,
  readBuildBytes, readBuildJson, verifyStableBuildBytes} = require("./build-source-inputs");
const {isStaticArchive} = require("./native-audit");
const {archiveName, archiveSha256, tarTool, sourceDirectories, amdgpulibCallsSource, llvmSourceSnapshot, verifyArchiveCatalog,
  llvmAmdgpulibCallsCompileOptionsShim, verifyCmakeVersion, verifyLlvmHostCompileFlag} = require("./native-llvm-sdk");
const {applyLinuxSysrootEnvironment, requireLinuxSysroot, linuxSysrootRecord, verifyLinuxSysrootRecord, compilerIdentityText} = require("./native-linux-sysroot");

const repo = path.resolve(__dirname, "..");
// Source consumers need drivers and utilities as well as archives. These SDK tools are not release files.
// llvm-objcopy supports the ROCm producer; llvm-readobj is needed to inspect Windows COFF inputs; llvm-mca stays out.
const toolsComponents = Object.freeze(["clang", "llvm-config", "llvm-ar", "llvm-nm", "llvm-rc", "llvm-ml",
  "llvm-tblgen", "clang-tblgen", "lld", "llc", "opt", "llvm-link", "llvm-mc", "llvm-dis", "llvm-objcopy", "llvm-readobj"]);
// These are source-defined sibling aliases, not independent compiler targets.
const toolAliases = Object.freeze({"bin/llvm-lib": "llvm-ar", "bin/llvm-ranlib": "llvm-ar", "bin/lld-link": "lld"});
// llvm-lib/ranlib are separate install components; lld-link is installed as part of the lld component.
const installAliasComponents = Object.freeze(["llvm-lib", "llvm-ranlib"]);
const metadataInstallTargets = Object.freeze(["install-mom-cmake-exports", "install-clang-mom-cmake-exports", "install-lld-mom-cmake-exports",
  "install-cmake-exports", "install-clang-cmake-exports", "install-lld-cmake-exports", "install-llvm-headers", "install-clang-headers",
  "install-clang-resource-headers", "install-lld-headers"]);
const installTargets = Object.freeze(["install-mom-distribution", ...metadataInstallTargets]);
const recipeFiles = Object.freeze(["scripts/build-native-llvm-sdk.js", "scripts/build-source-inputs.js", "scripts/native-audit.js", "scripts/native-llvm-sdk.js",
  "scripts/native-linux-sysroot.js"]);
const configureTimeout = 1200000, extractTimeout = 600000, installTimeout = 1200000, phaseTimeout = 2700000;

/** @typedef {{name: string, project: "llvm" | "clang" | "lld", projectRoot: string, sourceDirectory: string, artifact: string, type: string}} LlvmTarget */
/** @typedef {{name: string, targets: string[], artifacts: string[]}} BuildPhase */
/** @typedef {{components: string[], distributionComponents: string[], archives: {name: string, artifact: string, relativeArtifact: string}[],
 * tools: {name: string, artifact: string, relativeArtifact: string}[], phases: BuildPhase[]}} DistributionSelection */
/** @typedef {{archive: string, output: string, source: string, build: string, prefix: string, jobs: number,
 * tools: {cmake: string, ninja: string, cc: string, cxx: string}, toolchain: string, sourceOptionsFile: string, configure: string[], selectedConfigure: string[],
 * selectedComponents: string[], phases: BuildPhase[]}} LlvmPlan */
/** @typedef {Omit<LlvmPlan, "selectedComponents" | "phases">} LlvmBasePlan */

/** CMake paths and arguments are passed directly, never through a shell. @param {unknown} value @returns {value is string} */
const literal = value => typeof value === "string" && value.length > 0 && value.length <= 4096 && path.isAbsolute(value) &&
  !/[;"'$`\\\r\n\0]/.test(value) && !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);

/** Check each lexical component without following symlinks outside the selected Ninja tree.
 * Only executable leaves may be a direct relative sibling alias, as with clang -> clang-21.
 * @param {string} build @param {string} file @param {"file" | "directory" | "executable"} finalKind @param {boolean} [optional] */
function verifyBuildPath(build, file, finalKind, optional = false) {
  const root = fs.realpathSync(build);
  if (root !== build || path.resolve(file) !== file || !containsBuildPath(build, file)) {
    throw new Error("LLVM build path escaped or aliased its selected tree: " + file);
  }
  const parts = path.relative(build, file).split(path.sep);
  let current = build;
  for (const [index, part] of parts.entries()) {
    current = path.join(current, part);
    const stat = fs.lstatSync(current, {throwIfNoEntry: false});
    if (!stat) {
      if (optional) {return false;}
      throw new Error("LLVM build path is missing: " + file);
    }
    const final = index === parts.length - 1;
    if (final && finalKind === "executable" && stat.isSymbolicLink()) {
      const target = fs.readlinkSync(current);
      if (!target || target === "." || target === ".." || path.basename(target) !== target ||
          path.dirname(current) !== path.join(build, "bin")) {throw new Error("LLVM executable alias escapes its sibling directory: " + current);}
      return verifyBuildPath(build, path.join(path.dirname(current), target), "file");
    }
    const real = fs.realpathSync(current);
    const wrongKind = final ? (finalKind === "directory" ? !stat.isDirectory() : !stat.isFile()) : !stat.isDirectory();
    if (stat.isSymbolicLink() || real !== current || !containsBuildPath(root, real) || wrongKind) {
      throw new Error("LLVM build path is a symlink, special file or escapes its selected tree: " + current);
    }
  }
  return true;
}

/** The sysroot toolchain file keeps CMake's header/library searches inside the glibc 2.31 sysroot.
 * @param {string} archive @param {string} output @param {{cmake: string, ninja: string, cc: string, cxx: string}} tools @param {number} jobs
 * @param {string} toolchain @returns {LlvmBasePlan} */
function linuxLlvmSdkPlan(archive, output, tools, jobs, toolchain) {
  if (![archive, output, toolchain, ...Object.values(tools || {})].every(literal) || !tools ||
      !hasExactBuildFields(tools, ["cc", "cmake", "cxx", "ninja"]) || path.basename(archive) !== archiveName ||
      path.normalize(output) !== output || path.basename(tools.cc) !== "icx" || path.basename(tools.cxx) !== "icpx" ||
      path.basename(tools.cmake) !== "cmake" || path.basename(tools.ninja) !== "ninja" ||
      !Number.isInteger(jobs) || jobs < 1 || jobs > 32) {
    throw new Error("Linux LLVM 21 SDK needs its pinned archive, explicit CMake/Ninja/Intel tools and jobs in 1..32");
  }
  const source = path.join(output, "source"), build = path.join(output, "build");
  for (const input of [repo, archive, ...Object.values(tools)]) {
    if (buildPathsOverlap(input, output)) {throw new Error("Linux LLVM 21 output must not overlap a source, tool or repository input");}
  }
  const prefix = path.join(output, "sdk"), sourceOptionsFile = path.join(output, "llvm-host-source-options.cmake");
  const definitions = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_INSTALL_PREFIX=" + prefix,
    "-DCMAKE_C_COMPILER=" + tools.cc, "-DCMAKE_CXX_COMPILER=" + tools.cxx, "-DCMAKE_TOOLCHAIN_FILE=" + toolchain,
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_SUPPRESS_REGENERATION=ON",
    "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=" + sourceOptionsFile,
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--export-dynamic", "-DLLVM_ENABLE_PROJECTS=clang;lld",
    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU", "-DLLVM_DISTRIBUTIONS=MoM", "-DLLVM_STRICT_DISTRIBUTIONS=ON",
    "-DBUILD_SHARED_LIBS=OFF", "-DCLANG_LINK_CLANG_DYLIB=OFF",
    "-DLLVM_BUILD_LLVM_DYLIB=OFF", "-DLLVM_LINK_LLVM_DYLIB=OFF", "-DLLVM_BUILD_RUNTIME=ON",
    "-DLLVM_BUILD_TOOLS=ON", "-DCLANG_BUILD_TOOLS=ON", "-DLLD_BUILD_TOOLS=ON",
    "-DLLVM_ENABLE_RTTI=ON", "-DLLVM_ENABLE_EH=ON", "-DLLVM_ENABLE_PIC=ON", "-DLLVM_ENABLE_LTO=OFF",
    "-DLLVM_ENABLE_PLUGINS=OFF", "-DLLVM_EXPORT_SYMBOLS_FOR_PLUGINS=OFF",
    "-DLLVM_INCLUDE_TESTS=OFF", "-DCLANG_INCLUDE_TESTS=OFF",
    "-DLLVM_INCLUDE_DOCS=OFF", "-DCLANG_INCLUDE_DOCS=OFF", "-DLLVM_INCLUDE_EXAMPLES=OFF",
    "-DCLANG_BUILD_EXAMPLES=OFF", "-DLLVM_INCLUDE_BENCHMARKS=OFF",
    "-DLLVM_ENABLE_ZLIB=OFF", "-DLLVM_ENABLE_ZSTD=OFF", "-DLLVM_ENABLE_CURL=OFF",
    "-DLLVM_ENABLE_LIBXML2=OFF", "-DLLVM_ENABLE_LIBEDIT=OFF", "-DLLVM_ENABLE_LIBPFM=OFF",
    "-DLLVM_USE_PERF=OFF"];
  const configure = [tools.cmake, "-G", "Ninja", "-S", path.join(source, "llvm"), "-B", build, ...definitions];
  return {archive, output, source, build, prefix, jobs, tools: {...tools}, toolchain, sourceOptionsFile, configure,
    selectedConfigure: [...configure, "-DLLVM_MoM_DISTRIBUTION_COMPONENTS=<generated-components>"]};
}

/** @param {string} source */
function verifySourceVersion(source) {
  const llvmCmake = readBounded(path.join(source, "llvm", "CMakeLists.txt"), 2 << 20);
  if (!/set\(\s*LLVM_COMMON_CMAKE_UTILS\s+\$\{CMAKE_CURRENT_SOURCE_DIR\}\/\.\.\/cmake\s*\)/.test(llvmCmake) ||
      !/include\(\s*\$\{LLVM_COMMON_CMAKE_UTILS\}\/Modules\/LLVMVersion\.cmake\s*\)/.test(llvmCmake)) {
    throw new Error("LLVM CMake does not include the pinned version module");
  }
  const text = readBounded(path.join(source, "cmake", "Modules", "LLVMVersion.cmake"), 2 << 20);
  for (const [name, value] of [["LLVM_VERSION_MAJOR", "21"], ["LLVM_VERSION_MINOR", "1"], ["LLVM_VERSION_PATCH", "8"]]) {
    if (!new RegExp("^\\s*set\\(\\s*" + name + "\\s+" + value + "\\s*\\)", "m").test(text)) {
      throw new Error("Pinned LLVM source version is not 21.1.8: " + name);
    }
  }
}

/** @param {string} file @param {number} maximum */
function readBounded(file, maximum) {
  const bytes = readBuildBytes(file, maximum), text = bytes.toString("utf8");
  if (Buffer.byteLength(text) !== bytes.length) {throw new Error("Invalid UTF-8 in LLVM build metadata: " + file);}
  return text;
}

/** @param {string} build */
function ninjaPhonyTargets(build) {
  const file = path.join(build, "build.ninja"); verifyBuildPath(build, file, "file");
  const text = readBounded(file, 64 << 20), rows = new Map();
  for (const line of text.split(/\r?\n/)) {
    const match = line.match(/^build ([^ :]+): phony(?: (.*))?$/);
    if (match) {
      if (rows.has(match[1] || "")) {throw new Error("Duplicate generated Ninja target: " + match[1]);}
      rows.set(match[1] || "", (match[2] || "").split(/\s+/).filter(Boolean));
    }
  }
  if (!rows.size) {throw new Error("LLVM Ninja graph has no phony targets");}
  return rows;
}

/** @param {string} build @param {string} source */
function llvmStaticTargets(build, source) {
  const reply = path.join(build, ".cmake", "api", "v1", "reply"); verifyBuildPath(build, reply, "directory");
  const indexNames = fs.readdirSync(reply).filter(name => /^index-[A-Za-z0-9_.-]+\.json$/.test(name));
  if (indexNames.length !== 1) {throw new Error("Missing or ambiguous LLVM 21 CMake File API index");}
  const indexName = indexNames[0]; if (!indexName) {throw new Error("Missing LLVM 21 CMake File API index");}
  const indexFile = path.join(reply, indexName); verifyBuildPath(build, indexFile, "file");
  const index = /** @type {{reply?: {"codemodel-v2"?: {jsonFile?: string}}}} */ (readBuildJson(indexFile));
  const modelName = index.reply?.["codemodel-v2"]?.jsonFile;
  if (typeof modelName !== "string" || path.basename(modelName) !== modelName) {throw new Error("LLVM 21 File API omitted its codemodel");}
  const modelFile = path.join(reply, modelName); verifyBuildPath(build, modelFile, "file");
  const model = /** @type {{paths?: {source?: string, build?: string}, configurations?: {targets?: {id?: string, jsonFile?: string}[]}[]}} */
    (readBuildJson(modelFile));
  const configuration = model.configurations?.[0], refs = configuration?.targets;
  const modelSourcePath = model.paths?.source, modelBuildPath = model.paths?.build;
  if (typeof modelSourcePath !== "string" || !path.isAbsolute(modelSourcePath) || typeof modelBuildPath !== "string" || !path.isAbsolute(modelBuildPath) ||
      model.configurations?.length !== 1 || !Array.isArray(refs)) {throw new Error("LLVM 21 CMake File API has an unexpected model");}
  const sourceRoot = fs.realpathSync(source), buildRoot = fs.realpathSync(build), modelSource = path.resolve(modelSourcePath),
    modelBuild = path.resolve(modelBuildPath);
  if (!containsBuildPath(sourceRoot, modelSource) || !containsBuildPath(buildRoot, modelBuild)) {throw new Error("LLVM 21 CMake model escaped its source/build roots");}
  /** @type {LlvmTarget[]} */ const targets = [];
  for (const reference of refs) {
    if (typeof reference?.id !== "string" || typeof reference.jsonFile !== "string" || path.basename(reference.jsonFile) !== reference.jsonFile) {
      throw new Error("Invalid LLVM 21 CMake target reference");
    }
    const targetFile = path.join(reply, reference.jsonFile); verifyBuildPath(build, targetFile, "file");
    const target = /** @type {{name?: string, type?: string, paths?: {source?: string, build?: string}, artifacts?: {path?: string}[]}} */
      (readBuildJson(targetFile));
    const targetSource = typeof target.paths?.source === "string" ? path.resolve(modelSource, target.paths.source) : "";
    const project = /** @type {"llvm" | "clang" | "lld" | undefined} */ (sourceDirectories.slice(0, 3).find(name =>
      containsBuildPath(path.join(sourceRoot, name), targetSource)));
    if (!project || !["STATIC_LIBRARY", "EXECUTABLE"].includes(target.type || "")) {continue;}
    const targetBuild = typeof target.paths?.build === "string" ? path.resolve(modelBuild, target.paths.build) : "";
    const targetArtifact = target.artifacts?.[0]?.path;
    if (!containsBuildPath(buildRoot, targetBuild) || typeof target.name !== "string" || !/^[A-Za-z0-9_.+-]{1,160}$/.test(target.name || "") ||
        !Array.isArray(target.artifacts) || target.artifacts.length !== 1 || typeof targetArtifact !== "string") {
      throw new Error("Invalid LLVM 21 target metadata: " + (target.name || "unnamed"));
    }
    const artifact = path.isAbsolute(targetArtifact) ? targetArtifact : path.resolve(modelBuild, targetArtifact);
    if (!containsBuildPath(buildRoot, artifact) || target.type === "STATIC_LIBRARY" && !artifact.endsWith(".a") ||
        target.type === "EXECUTABLE" && !artifact.startsWith(path.join(buildRoot, "bin") + path.sep)) {
      throw new Error("LLVM 21 target artifact escaped its selected build kind: " + artifact);
    }
    verifyBuildPath(build, artifact, target.type === "EXECUTABLE" ? "executable" : "file", true);
    targets.push({name: target.name, project, projectRoot: path.join(sourceRoot, project), sourceDirectory: targetSource, artifact, type: target.type || ""});
  }
  targets.sort((a, b) => a.project.localeCompare(b.project) || a.name.localeCompare(b.name));
  if (!targets.length || targets.length > 1024 || new Set(targets.map(row => row.name)).size !== targets.length ||
      ["llvm", "clang", "lld"].some(project => !targets.some(row => row.project === project && row.type === "STATIC_LIBRARY"))) {
    throw new Error("LLVM 21 static target graph is incomplete or ambiguous");
  }
  return targets;
}

/** The public distribution is exactly installable static targets plus tools used by source consumers.
 * @param {LlvmTarget[]} targets @param {Map<string, string[]>} ninjaTargets @param {string} build @returns {DistributionSelection} */
function linuxDistributionSelection(targets, ninjaTargets, build) {
  const buildRoot = fs.realpathSync(build);
  const byName = new Map(targets.map(row => [row.name, row]));
  const archives = targets.filter(row => row.type === "STATIC_LIBRARY" && ninjaTargets.has("install-" + row.name));
  if (!archives.length) {throw new Error("LLVM 21 distribution graph has no installable static targets");}
  const selectedTools = toolsComponents.map(name => {
    const target = byName.get(name);
    if (!target || target.type !== "EXECUTABLE" || !ninjaTargets.has("install-" + name)) {
      throw new Error("LLVM 21 distribution omitted required source tool: " + name);
    }
    return target;
  });
  const components = [...new Set([...archives, ...selectedTools].map(row => row.name))].sort();
  /** @param {LlvmTarget} row */ const relativeArtifact = row => {
    const relative = path.relative(buildRoot, row.artifact);
    if (!containsBuildPath(buildRoot, row.artifact) || path.isAbsolute(relative) || relative.startsWith("..")) {throw new Error("LLVM distribution artifact escaped its build tree");}
    return relative.split(path.sep).join("/");
  };
  const phases = /** @type {BuildPhase[]} */ ([]);
  /** @param {string} name @param {LlvmTarget[]} rows */
  const add = (name, rows) => {
    const selected = rows.filter(row => components.includes(row.name)).sort((a, b) => a.name.localeCompare(b.name));
    if (selected.length) {phases.push({name, targets: selected.map(row => row.name), artifacts: selected.map(row => row.artifact)});}
  };
  const llvm = archives.filter(row => row.project === "llvm"), llvmRoot = llvm[0]?.projectRoot;
  if (!llvmRoot || llvm.some(row => row.projectRoot !== llvmRoot || !containsBuildPath(llvmRoot, row.sourceDirectory))) {
    throw new Error("LLVM 21 target source roots are inconsistent");
  }
  /** @param {LlvmTarget} row */ const llvmSource = row => path.relative(llvmRoot, row.sourceDirectory).split(path.sep).join("/");
  /** @param {LlvmTarget} row @param {string} target */ const belongsToTarget = (row, target) => {
    const prefix = "lib/Target/" + target, relative = llvmSource(row);
    return relative === prefix || relative.startsWith(prefix + "/");
  };
  // Split common libraries to keep each hosted cold build within its deadline, using public source directories rather than target names.
  const llvmFoundationDirectories = ["lib/Support", "lib/Demangle", "lib/IR", "lib/BinaryFormat", "lib/Remarks",
    "lib/TargetParser", "lib/Bitstream/Reader"];
  /** @param {LlvmTarget} row */ const belongsToFoundation = row => llvmFoundationDirectories.some(directory => {
    const relative = llvmSource(row);
    return relative === directory || relative.startsWith(directory + "/");
  });
  const llvmCoreAnalysisDirectories = ["lib/Analysis", "lib/Bitcode", "lib/DebugInfo", "lib/IRReader", "lib/Linker",
    "lib/Object", "lib/ObjectYAML", "lib/ProfileData"];
  const llvmCoreOptimizationDirectories = ["lib/IPO", "lib/Transforms"];
  /** @param {LlvmTarget} row @param {string[]} directories */ const belongsToDirectories = (row, directories) => directories.some(directory => {
    const relative = llvmSource(row);
    return relative === directory || relative.startsWith(directory + "/");
  });
  const commonCore = llvm.filter(row => !["X86", "NVPTX", "AMDGPU"].some(target => belongsToTarget(row, target)) && !belongsToFoundation(row)),
    coreAnalysis = commonCore.filter(row => belongsToDirectories(row, llvmCoreAnalysisDirectories)),
    coreOptimization = commonCore.filter(row => belongsToDirectories(row, llvmCoreOptimizationDirectories));
  const llvmPhases = /** @type {[string, LlvmTarget[]][]} */ ([
    ["llvm-foundation", llvm.filter(belongsToFoundation)],
    ["llvm-core-analysis", coreAnalysis],
    ["llvm-core-optimization", coreOptimization],
    // Passes and LTO link CodeGen, so keep them and all remaining common libraries after the optimization layer.
    ["llvm-core-codegen", commonCore.filter(row => !coreAnalysis.includes(row) && !coreOptimization.includes(row))],
    ["llvm-x86", llvm.filter(row => belongsToTarget(row, "X86"))],
    ["llvm-nvptx", llvm.filter(row => belongsToTarget(row, "NVPTX"))],
    ["llvm-amdgpu", llvm.filter(row => belongsToTarget(row, "AMDGPU"))]
  ]);
  if (llvmPhases.some(([, rows]) => !rows.length)) {throw new Error("LLVM 21 target phase group is empty");}
  for (const [name, rows] of llvmPhases) {add(name, rows);}
  const clang = archives.filter(row => row.project === "clang"), clangRoot = clang[0]?.projectRoot;
  if (!clangRoot || clang.some(row => row.projectRoot !== clangRoot || !containsBuildPath(clangRoot, row.sourceDirectory))) {
    throw new Error("LLVM 21 clang source roots are inconsistent");
  }
  /** @param {LlvmTarget} row */ const relativeClang = row => path.relative(clangRoot, row.sourceDirectory).split(path.sep).join("/");
  add("clang-base", clang.filter(row => ["lib/Basic", "lib/AST", "lib/Lex"].includes(relativeClang(row))));
  add("clang-sema", clang.filter(row => ["lib/Analysis", "lib/Parse", "lib/Sema"].includes(relativeClang(row))));
  add("clang-frontend", clang.filter(row => !["lib/Basic", "lib/AST", "lib/Lex", "lib/Analysis", "lib/Parse", "lib/Sema"].includes(relativeClang(row))));
  add("lld", archives.filter(row => row.project === "lld"));
  add("tools", selectedTools);
  const seen = phases.flatMap(phase => phase.targets);
  if (!isDeepStrictEqual([...seen].sort(), components.slice().sort()) || new Set(seen).size !== seen.length) {
    throw new Error("LLVM 21 generated phase groups do not cover the distribution exactly once");
  }
  const distributionComponents = [...new Set([...components, ...installAliasComponents])].sort();
  return {components, distributionComponents,
    archives: archives.map(row => ({name: row.name, artifact: row.artifact, relativeArtifact: relativeArtifact(row)})),
    tools: selectedTools.map(row => ({name: row.name, artifact: row.artifact, relativeArtifact: relativeArtifact(row)})), phases};
}

/** @param {string} build @param {DistributionSelection} selection */
function verifyDistributionGraph(build, selection) {
  const targets = ninjaPhonyTargets(build), distribution = targets.get("mom-distribution"), install = targets.get("install-mom-distribution");
  const expected = [...selection.archives, ...selection.tools].map(row => row.relativeArtifact);
  const distributionComponents = selection.distributionComponents;
  for (const aliasName of installAliasComponents) {
    const aliasPath = "bin/" + aliasName;
    const backingName = Object.hasOwn(toolAliases, aliasPath) ?
      toolAliases[/** @type {keyof typeof toolAliases} */ (aliasPath)] : undefined;
    if (!distributionComponents.includes(aliasName)) {continue;}
    const aliasTarget = targets.get(aliasName), aliasEdgeName = aliasTarget?.length === 1 ? aliasTarget[0] : undefined,
      aliasEdge = aliasEdgeName ? targets.get(aliasEdgeName) : undefined,
      backingPath = selection.tools.find(row => row.name === backingName)?.relativeArtifact;
    if (!aliasEdgeName || !aliasEdge?.includes(aliasPath) || !backingPath || !aliasEdge.includes(backingPath) ||
        !targets.has("install-" + aliasName)) {
      throw new Error("Generated LLVM distribution omitted a selected source tool alias: " + aliasName);
    }
    // CMake represents a custom alias component by its directory-qualified phony edge.
    // Keep the observed form exact while accepting Ninja's top-level alias spelling too.
    const distributionEdge = distribution?.includes(aliasEdgeName) ? aliasEdgeName :
      distribution?.includes(aliasName) ? aliasName : undefined;
    if (!distributionEdge) {throw new Error("Generated LLVM distribution omitted its source tool alias edge: " + aliasName);}
    expected.push(distributionEdge);
  }
  expected.sort();
  if (!distribution || !install || !isDeepStrictEqual(distribution.slice().sort(), expected) ||
      distributionComponents.some(name => !targets.has("install-" + name)) || metadataInstallTargets.some(name => !targets.has(name))) {
    throw new Error("Generated LLVM MoM distribution does not match its target/install graph");
  }
  return {buildNinjaSha256: fileHash(path.join(build, "build.ninja")), distribution: "mom-distribution",
    install: "install-mom-distribution", components: distributionComponents};
}

/** @param {string} build */
function configuredGraphFiles(build) {
  const reply = path.join(build, ".cmake", "api", "v1", "reply"); verifyBuildPath(build, reply, "directory");
  const names = fs.readdirSync(reply).filter(name => /^index-[A-Za-z0-9_.-]+\.json$/.test(name));
  if (names.length !== 1 || !names[0]) {throw new Error("LLVM 21 File API index changed after configure");}
  const indexFile = path.join(reply, names[0]); verifyBuildPath(build, indexFile, "file");
  const index = /** @type {{reply?: {"codemodel-v2"?: {jsonFile?: string}}}} */ (readBuildJson(indexFile)),
    modelName = index.reply?.["codemodel-v2"]?.jsonFile;
  if (typeof modelName !== "string" || path.basename(modelName) !== modelName) {throw new Error("LLVM 21 File API codemodel is invalid");}
  const modelFile = path.join(reply, modelName); verifyBuildPath(build, modelFile, "file");
  const model = /** @type {{configurations?: {targets?: {jsonFile?: string}[]}[]}} */ (readBuildJson(modelFile)),
    refs = model.configurations?.[0]?.targets;
  if (!Array.isArray(refs) || refs.length > 2048) {throw new Error("LLVM 21 File API target set is invalid");}
  const files = [path.join(build, "CMakeCache.txt"), path.join(build, "build.ninja"), path.join(build, "CMakeFiles", "rules.ninja"),
    path.join(build, "compile_commands.json"), indexFile, modelFile];
  for (const reference of refs) {
    if (typeof reference?.jsonFile === "string" && path.basename(reference.jsonFile) === reference.jsonFile) {
      const file = path.join(reply, reference.jsonFile); verifyBuildPath(build, file, "file"); files.push(file);
    }
  }
  if (files.length > 2048) {throw new Error("LLVM 21 generated graph exceeds bounds");}
  for (const file of files) {verifyBuildPath(build, file, "file");}
  return [...new Set(files)];
}

/** @param {string} file */
function cmakeCache(file) {
  const values = new Map();
  for (const line of readBounded(file, 2 << 20).split(/\r?\n/)) {
    const match = line.match(/^([^#/:][^:=]*):[^=]*=(.*)$/);
    if (match) {values.set(match[1] || "", match[2] || "");}
  }
  return values;
}

/** @param {string} build @param {Pick<LlvmPlan, "tools" | "prefix" | "sourceOptionsFile" | "toolchain">} plan @param {string[]} components */
function verifyCMakeCache(build, plan, components) {
  const values = cmakeCache(path.join(build, "CMakeCache.txt")), expected = {
    CMAKE_BUILD_TYPE: "Release", CMAKE_C_COMPILER: plan.tools.cc, CMAKE_CXX_COMPILER: plan.tools.cxx, CMAKE_TOOLCHAIN_FILE: plan.toolchain,
    CMAKE_INSTALL_PREFIX: plan.prefix, CMAKE_EXE_LINKER_FLAGS: "-Wl,--export-dynamic",
    LLVM_ENABLE_PROJECTS: "clang;lld", LLVM_TARGETS_TO_BUILD: "X86;NVPTX;AMDGPU", LLVM_DISTRIBUTIONS: "MoM",
    LLVM_MoM_DISTRIBUTION_COMPONENTS: components.join(";"), LLVM_ENABLE_PIC: "ON", LLVM_ENABLE_PLUGINS: "OFF",
    LLVM_EXPORT_SYMBOLS_FOR_PLUGINS: "OFF", LLVM_ENABLE_RTTI: "ON", LLVM_ENABLE_EH: "ON", LLVM_ENABLE_LTO: "OFF",
    BUILD_SHARED_LIBS: "OFF", LLVM_BUILD_LLVM_DYLIB: "OFF", LLVM_LINK_LLVM_DYLIB: "OFF", CLANG_LINK_CLANG_DYLIB: "OFF",
    LLVM_INCLUDE_TESTS: "OFF", CLANG_INCLUDE_TESTS: "OFF", LLVM_INCLUDE_DOCS: "OFF",
    CLANG_INCLUDE_DOCS: "OFF", LLVM_INCLUDE_EXAMPLES: "OFF", CLANG_BUILD_EXAMPLES: "OFF", LLVM_INCLUDE_BENCHMARKS: "OFF",
    LLVM_ENABLE_ZLIB: "OFF", LLVM_ENABLE_ZSTD: "OFF", LLVM_ENABLE_CURL: "OFF", LLVM_ENABLE_LIBXML2: "OFF",
    LLVM_ENABLE_LIBEDIT: "OFF", LLVM_ENABLE_LIBPFM: "OFF", LLVM_USE_PERF: "OFF",
    LLVM_BUILD_TOOLS: "ON", CLANG_BUILD_TOOLS: "ON", LLVM_BUILD_RUNTIME: "ON", CMAKE_PROJECT_TOP_LEVEL_INCLUDES: plan.sourceOptionsFile,
  };
  for (const [name, value] of Object.entries(expected)) {if (values.get(name) !== value) {throw new Error("LLVM 21 CMake cache changed required value: " + name);}}
  return Object.fromEntries(Object.keys(expected).sort().map(name => [name, values.get(name) || ""]));
}

/** @param {string} file @param {string} command @param {string[]} args @param {string} cwd @param {NodeJS.ProcessEnv} env @param {number} timeout */
function runLogged(file, command, args, cwd, env, timeout) {
  const fd = fs.openSync(file, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY, 0o600);
  let result;
  try {result = spawnSync(command, args, {cwd, env, stdio: ["ignore", fd, fd], timeout, maxBuffer: 1 << 20});} finally {fs.closeSync(fd);}
  const record = {command, args, cwd, timeoutMs: timeout, status: result.status, signal: result.signal || null, error: result.error?.message || null,
    log: file, logSha256: fileHash(file)};
  if (result.error || result.signal || result.status !== 0) {throw new Error("LLVM 21 phase failed; inspect " + file, {cause: result.error});}
  return record;
}

/** @param {string} build @param {number} jobs @param {BuildPhase} phase */
function phaseBuildArgs(build, jobs, phase) {
  const args = ["--build", build, "--target", ...phase.targets, "--parallel", String(jobs)];
  if (phase.name === "tools") {args.push("--verbose");}
  return [...args, "--", "-k", "0"];
}

/** @param {string} file @param {unknown} value */
function writeJsonExclusive(file, value) {fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n", {flag: "wx", mode: 0o600});}
/** Every saved build input still has its recorded bytes. @param {Record<string, string>} expected */
function buildInputsCurrent(expected) {return isDeepStrictEqual(expected, buildFileInputs(Object.keys(expected || {})));}

/** @param {string} file @param {number} bytes */
function evidenceFile(file, bytes) {return {path: file, sha256: fileHash(file), bytes};}

/** @param {string} tool @param {string[]} args */
function toolVersion(tool, args = ["--version"]) {
  const result = spawnSync(tool, args, {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
  if (result.error || result.signal || result.status !== 0) {throw new Error("Cannot identify selected LLVM build tool: " + tool, {cause: result.error});}
  return (result.stdout || "") + (result.stderr || "");
}
/** @param {LlvmPlan["tools"]} tools */
function verifyToolIdentity(tools) {
  /** @type {Record<string, {path: string, realpath: string, linkTarget: string | null, sha256: string, version: string}>} */ const rows = {};
  for (const [name, file] of Object.entries(tools)) {
    const realpath = fs.realpathSync(file), stat = fs.statSync(realpath);
    if (!stat.isFile()) {throw new Error("LLVM build tool is not a regular file: " + file);}
    const version = compilerIdentityText(toolVersion(file));
    if (name === "cmake") {verifyCmakeVersion(version);}
    if (["cc", "cxx"].includes(name) && !/2026\.2(?:\.|\b)/.test(version)) {throw new Error("LLVM 21 SDK requires Intel compiler 2026.2: " + file);}
    rows[name] = {path: file, realpath, linkTarget: fs.lstatSync(file).isSymbolicLink() ? fs.readlinkSync(file) : null,
      sha256: fileHash(realpath), version: version.trim().slice(0, 4096)};
  }
  return rows;
}

/** Require exactly one compile row, the AMDGPU host helper TU, to carry the host-only option.
 * @param {string} build @param {string} source */
function verifyAmdgpulibCallsCompileFlag(build, source) {
  const file = path.join(build, "compile_commands.json"); verifyBuildPath(build, file, "file");
  const rows = JSON.parse(readBounded(file, 64 << 20)), target = path.resolve(source, amdgpulibCallsSource);
  return verifyLlvmHostCompileFlag(rows, target, path.resolve);
}

/** The selected sysroot supplies ICXCFG/ICPXCFG and pkg-config paths; other compiler hooks are dropped.
 * @param {LlvmPlan["tools"]} tools @param {NodeJS.ProcessEnv} environment */
function cleanBuildEnvironment(tools, environment) {
  const env = applyLinuxSysrootEnvironment(environment);
  for (const name of Object.keys(env)) {
    if (/^(?:CC|CXX|CPP|CFLAGS|CXXFLAGS|CPPFLAGS|LDFLAGS|CMAKE_|GIT_|LD_PRELOAD$|LD_AUDIT$)/.test(name)) {delete env[name];}
  }
  env["PATH"] = [path.dirname(tools.cmake), path.dirname(tools.ninja), path.dirname(tools.cc), path.dirname(tools.cxx), env["PATH"] || "/usr/bin:/bin"].join(path.delimiter);
  env["LC_ALL"] = "C"; env["TZ"] = "UTC";
  return env;
}

/** @param {string} output @param {string} archive @param {LlvmPlan["tools"]} tools @param {number} jobs @param {NodeJS.ProcessEnv} environment */
function configureLinuxLlvmSdk(output, archive, tools, jobs, environment = process.env) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("Linux LLVM 21 source build requires Linux x64 and MOM_AOT_BUILD_ONLY=1");
  }
  // The sysroot compiler configuration reaches icx/icpx through ICXCFG/ICPXCFG (cleanBuildEnvironment).
  const sysroot = requireLinuxSysroot(environment), plan = linuxLlvmSdkPlan(archive, output, tools, jobs, sysroot.toolchain),
    outputParent = fs.realpathSync(path.dirname(output));
  if (path.join(outputParent, path.basename(output)) !== output || fs.existsSync(output)) {throw new Error("Select a new canonical LLVM 21 output directory");}
  for (const input of [repo, archive, ...Object.values(tools)]) {
    if (buildPathsOverlap(fs.realpathSync(input), path.join(outputParent, path.basename(output)))) {throw new Error("LLVM 21 output overlaps a frozen input");}
  }
  if (fileHash(archive) !== archiveSha256) {throw new Error("LLVM 21.1.8 source archive SHA256 mismatch");}
  const toolIdentity = verifyToolIdentity(tools), env = cleanBuildEnvironment(tools, environment);
  const inputFiles = buildFileInputs([archive, tarTool, ...Object.values(tools).map(file => fs.realpathSync(file)),
    ...recipeFiles.map(name => path.join(repo, name)), sysroot.toolchain, sysroot.compilerConfig, sysroot.receipt]);
  fs.mkdirSync(output, {mode: 0o700});
  const sourceOptionsShim = Buffer.from(llvmAmdgpulibCallsCompileOptionsShim(plan.source));
  fs.writeFileSync(plan.sourceOptionsFile, sourceOptionsShim, {flag: "wx", mode: 0o600});
  inputFiles[plan.sourceOptionsFile] = verifyStableBuildBytes(plan.sourceOptionsFile, sourceOptionsShim, "LLVM source-options shim");
  fs.mkdirSync(plan.source, {mode: 0o700});
  fs.copyFileSync(archive, path.join(output, archiveName), fs.constants.COPYFILE_EXCL);
  const archiveCopy = path.join(output, archiveName);
  if (fileHash(archiveCopy) !== archiveSha256 || fileHash(archive) !== archiveSha256) {throw new Error("Pinned LLVM archive changed while copying");}
  inputFiles[archiveCopy] = fileHash(archiveCopy);
  const catalog = verifyArchiveCatalog(archiveCopy);
  const members = [...sourceDirectories.map(name => "llvm-project-21.1.8.src/" + name + "/"), "llvm-project-21.1.8.src/LICENSE.TXT"];
  runLogged(path.join(output, "extract.log"), tarTool,
    ["-xJf", archiveCopy, "--directory", plan.source, "--strip-components=1", "--no-same-owner", "--no-same-permissions", "--", ...members], output, env, extractTimeout);
  const sourceState = llvmSourceSnapshot(plan.source); verifySourceVersion(plan.source);
  const producerSnapshot = path.join(output, "producer-snapshot.js");
  fs.copyFileSync(path.join(repo, "scripts/build-native-llvm-sdk.js"), producerSnapshot, fs.constants.COPYFILE_EXCL);
  if (fileHash(producerSnapshot) !== inputFiles[path.join(repo, "scripts/build-native-llvm-sdk.js")]) {throw new Error("LLVM producer changed during configure");}
  const queryDirectory = path.join(plan.build, ".cmake", "api", "v1", "query"), queryFile = path.join(queryDirectory, "codemodel-v2");
  fs.mkdirSync(queryDirectory, {recursive: true});
  fs.writeFileSync(queryFile, "", {flag: "wx"});
  const discoveryCommand = [...plan.configure];
  const discovery = runLogged(path.join(output, "configure-discovery.log"), tools.cmake, discoveryCommand.slice(1), repo, env, configureTimeout);
  const discoveryHostSourceFlag = verifyAmdgpulibCallsCompileFlag(plan.build, plan.source);
  const targets = llvmStaticTargets(plan.build, plan.source), firstNinja = ninjaPhonyTargets(plan.build),
    selection = linuxDistributionSelection(targets, firstNinja, plan.build);
  const selectedConfigure = [...plan.configure, "-DLLVM_MoM_DISTRIBUTION_COMPONENTS=" + selection.distributionComponents.join(";")];
  const selected = runLogged(path.join(output, "configure-selected.log"), tools.cmake, selectedConfigure.slice(1), repo, env, configureTimeout);
  const hostSourceFlag = verifyAmdgpulibCallsCompileFlag(plan.build, plan.source);
  if (!isDeepStrictEqual(hostSourceFlag, discoveryHostSourceFlag)) {throw new Error("LLVM host source-option row changed between configure passes");}
  const postConfigureSourceState = llvmSourceSnapshot(plan.source);
  if (!isDeepStrictEqual(sourceState, postConfigureSourceState)) {throw new Error("LLVM source changed during configure");}
  const finalTargets = llvmStaticTargets(plan.build, plan.source), finalNinja = ninjaPhonyTargets(plan.build),
    finalSelection = linuxDistributionSelection(finalTargets, finalNinja, plan.build);
  if (!isDeepStrictEqual(selection, finalSelection)) {throw new Error("LLVM 21 generated distribution selection changed on reconfigure");}
  const graphFiles = configuredGraphFiles(plan.build), configuredFiles = buildFileInputs(graphFiles);
  const distributionGraph = verifyDistributionGraph(plan.build, selection), cache = verifyCMakeCache(plan.build, plan, selection.distributionComponents);
  const generated = {...plan, selectedComponents: selection.distributionComponents, phases: selection.phases,
    selectedConfigure: [...plan.configure, "-DLLVM_MoM_DISTRIBUTION_COMPONENTS=" + selection.distributionComponents.join(";")], installTargets: [...installTargets]};
  const savedPlan = path.join(output, "plan.json"); writeJsonExclusive(savedPlan, generated);
  const record = {version: 1, role: "linux-llvm21-source-sdk", runId: randomUUID(), stage: "configured", plan: generated,
    planSha256: fileHash(savedPlan), archiveSha256, catalog, sourceState, inputFiles, toolIdentity, sysroot: linuxSysrootRecord(sysroot),
    producerSnapshot: evidenceFile(producerSnapshot, fs.statSync(producerSnapshot).size),
    configure: {discovery, selected, cache, distributionGraph, targetCount: targets.length, selection, hostSourceFlag,
      files: configuredFiles, compileCommands: evidenceFile(path.join(plan.build, "compile_commands.json"), fs.statSync(path.join(plan.build, "compile_commands.json")).size)},
    buildOnly: true, compiled: false, executed: false, publicationQualified: false};
  writeJsonExclusive(path.join(output, "configured.json"), record);
  if (!isDeepStrictEqual(sourceState, llvmSourceSnapshot(plan.source)) || fileHash(archive) !== archiveSha256 ||
      !isDeepStrictEqual(inputFiles, buildFileInputs(Object.keys(inputFiles)))) {throw new Error("LLVM 21 source/tool inputs changed during configure");}
  return record;
}

/** @param {string} output */
function verifyConfigured(output) {
  const file = path.join(output, "configured.json"), receipt = readBuildJson(file), plan = receipt.plan;
  const producerFile = path.join(repo, "scripts/build-native-llvm-sdk.js"), producerSnapshot = path.join(output, "producer-snapshot.js");
  if (!plan || !receipt.inputFiles || !receipt.configure?.files) {throw new Error("LLVM 21 configure receipt is incomplete");}
  verifyLinuxSysrootRecord(receipt.sysroot, "LLVM 21 SDK");
  if (plan.sourceOptionsFile !== path.join(output, "llvm-host-source-options.cmake")) {throw new Error("LLVM source-options shim is outside its output");}
  const sourceOptionsSha256 = verifyStableBuildBytes(plan.sourceOptionsFile,
    Buffer.from(llvmAmdgpulibCallsCompileOptionsShim(plan.source)), "LLVM source-options shim");
  if (receipt.inputFiles[plan.sourceOptionsFile] !== sourceOptionsSha256) {
    throw new Error("LLVM source-options shim is not the expected generated build input");
  }
  verifySourceVersion(plan.source);
  for (const graphFile of Object.keys(receipt.configure.files)) {verifyBuildPath(plan.build, graphFile, "file");}
  if (receipt.version !== 1 || receipt.role !== "linux-llvm21-source-sdk" || receipt.stage !== "configured" ||
      receipt.buildOnly !== true || receipt.compiled !== false || receipt.executed !== false || receipt.publicationQualified !== false ||
      !plan || plan.output !== output || fileHash(path.join(output, "plan.json")) !== receipt.planSha256 ||
      fileHash(plan.archive) !== archiveSha256 || fileHash(path.join(output, archiveName)) !== archiveSha256 ||
      receipt.producerSnapshot?.path !== producerSnapshot || receipt.producerSnapshot.sha256 !== fileHash(producerSnapshot) ||
      receipt.inputFiles[producerFile] !== receipt.producerSnapshot.sha256 ||
      !isDeepStrictEqual(receipt.sourceState, llvmSourceSnapshot(plan.source)) ||
      !buildInputsCurrent(receipt.inputFiles) ||
      !isDeepStrictEqual(Object.fromEntries(Object.entries(receipt.toolIdentity || {}).map(([name, row]) =>
        [name, {...row, version: compilerIdentityText(String(row?.version))}])), verifyToolIdentity(plan.tools)) ||
      !isDeepStrictEqual(receipt.configure.files, buildFileInputs(Object.keys(receipt.configure.files || {})))) {
    throw new Error("LLVM 21 configure receipt is stale or incomplete; use a new output directory");
  }
  const targets = llvmStaticTargets(plan.build, plan.source), selection = linuxDistributionSelection(targets, ninjaPhonyTargets(plan.build), plan.build),
    hostSourceFlag = verifyAmdgpulibCallsCompileFlag(plan.build, plan.source);
  if (!isDeepStrictEqual(selection, receipt.configure.selection) || !isDeepStrictEqual(plan.phases, selection.phases) ||
      !isDeepStrictEqual(verifyDistributionGraph(plan.build, selection), receipt.configure.distributionGraph) ||
      !isDeepStrictEqual(verifyCMakeCache(plan.build, plan, selection.distributionComponents), receipt.configure.cache) ||
      !isDeepStrictEqual(hostSourceFlag, receipt.configure.hostSourceFlag)) {
    throw new Error("LLVM 21 configure graph or distribution selection changed");
  }
  return {file, receipt, plan, selection, targets};
}

/** @param {string} output @param {string} phaseName @param {NodeJS.ProcessEnv} environment */
function buildLinuxLlvmSdkPhase(output, phaseName, environment = process.env) {
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Linux LLVM 21 build requires MOM_AOT_BUILD_ONLY=1");}
  const {receipt, plan, selection} = verifyConfigured(output), phaseIndex = selection.phases.findIndex(row => row.name === phaseName);
  if (phaseIndex < 0) {throw new Error("Select a generated LLVM 21 build phase");}
  for (let index = 0; index < phaseIndex; index++) {
    const prior = selection.phases[index];
    if (!prior) {throw new Error("LLVM 21 build phase order is invalid");}
    verifyPhaseReceipt(output, plan, receipt, prior);
  }
  if (fs.existsSync(path.join(output, "phase-" + phaseName + ".json"))) {throw new Error("LLVM 21 phase already has an immutable result");}
  const phase = selection.phases[phaseIndex];
  if (!phase) {throw new Error("LLVM 21 build phase order is invalid");}
  const env = cleanBuildEnvironment(plan.tools, environment);
  const graphFiles = Object.keys(receipt.configure.files), before = buildFileInputs(graphFiles), args = phaseBuildArgs(plan.build, plan.jobs, phase);
  const phaseStartedAt = Date.now();
  console.log(`[LLVM 21] ${phase.name}: starting ${phase.targets.length} targets`);
  let run;
  try {
    run = runLogged(path.join(output, "build-" + phase.name + ".log"), plan.tools.cmake, args, repo, env, phaseTimeout);
  } catch (error) {
    console.log(`[LLVM 21] ${phase.name}: failed after ${Date.now() - phaseStartedAt} ms`);
    throw error;
  }
  console.log(`[LLVM 21] ${phase.name}: completed in ${Date.now() - phaseStartedAt} ms`);
  const targets = llvmStaticTargets(plan.build, plan.source), byName = new Map(targets.map(row => [row.name, row]));
  const outputs = phase.targets.map(name => {
    const target = byName.get(name);
    if (!target) {throw new Error("LLVM target disappeared after its build phase: " + name);}
    const artifact = target.artifact; verifyBuildPath(plan.build, artifact, target.type === "EXECUTABLE" ? "executable" : "file");
    const stat = fs.statSync(artifact);
    if (!stat.isFile() || target.type === "STATIC_LIBRARY" && !isStaticArchive(artifact)) {throw new Error("LLVM build phase output failed audit: " + artifact);}
    return {name, path: artifact, bytes: stat.size, sha256: fileHash(artifact)};
  });
  if (!isDeepStrictEqual(before, buildFileInputs(graphFiles)) || !isDeepStrictEqual(receipt.sourceState, llvmSourceSnapshot(plan.source)) ||
      !isDeepStrictEqual(receipt.inputFiles, buildFileInputs(Object.keys(receipt.inputFiles)))) {throw new Error("LLVM inputs or generated graph changed during build");}
  const link = phaseIndex === selection.phases.length - 1 ? findClangLinkCommand(output, selection.phases, plan.tools.cxx) : undefined;
  const phaseReceipt = {version: 1, role: receipt.role, runId: receipt.runId, phase, run, outputs, sourceSha256: receipt.sourceState.sha256,
    graphFiles: receipt.configure.files, buildOnly: true, compiled: true, executed: false};
  writeJsonExclusive(path.join(output, "phase-" + phaseName + ".json"), phaseReceipt);
  if (link) {
    writeJsonExclusive(path.join(output, "build-complete.json"), {version: 1, role: receipt.role, runId: receipt.runId,
      phases: selection.phases.map(row => evidenceFile(path.join(output, "phase-" + row.name + ".json"), fs.statSync(path.join(output, "phase-" + row.name + ".json")).size)),
      clangLinkCommand: link, buildOnly: true, compiled: true, executed: false});
  }
  return phaseReceipt;
}

/** @param {string} output @param {LlvmPlan} plan @param {ReturnType<typeof readBuildJson>} receipt @param {BuildPhase} phase */
function verifyPhaseReceipt(output, plan, receipt, phase) {
  const file = path.join(output, "phase-" + phase.name + ".json"), saved = readBuildJson(file),
    log = path.join(output, "build-" + phase.name + ".log"), graphFiles = Object.keys(receipt.configure.files);
  const expectedArgs = phaseBuildArgs(plan.build, plan.jobs, phase);
  if (saved.version !== 1 || saved.role !== receipt.role || saved.runId !== receipt.runId ||
      !isDeepStrictEqual(saved.phase, phase) || saved.buildOnly !== true || saved.compiled !== true || saved.executed !== false ||
      saved.run?.command !== plan.tools.cmake || saved.run.timeoutMs !== phaseTimeout || !isDeepStrictEqual(saved.run.args, expectedArgs) || saved.run.status !== 0 ||
      saved.run.signal !== null || saved.run.log !== log || saved.run.logSha256 !== fileHash(log) ||
      saved.sourceSha256 !== receipt.sourceState.sha256 || !isDeepStrictEqual(saved.graphFiles, receipt.configure.files) ||
      !isDeepStrictEqual(buildFileInputs(graphFiles), receipt.configure.files)) {
    throw new Error("LLVM 21 phase receipt is missing or stale: " + phase.name);
  }
  const targets = llvmStaticTargets(plan.build, plan.source), byName = new Map(targets.map(row => [row.name, row]));
  if (!Array.isArray(saved.outputs) || saved.outputs.length !== phase.targets.length) {throw new Error("LLVM 21 phase output inventory is incomplete");}
  for (const [index, name] of phase.targets.entries()) {
    const target = byName.get(name), row = saved.outputs[index];
    if (!target || !row || row.name !== name || row.path !== target.artifact) {throw new Error("LLVM 21 phase artifact changed: " + name);}
    verifyBuildPath(plan.build, row.path, target.type === "EXECUTABLE" ? "executable" : "file");
    if (fileHash(row.path) !== row.sha256 ||
        target.type === "STATIC_LIBRARY" && !isStaticArchive(row.path)) {throw new Error("LLVM 21 phase artifact changed: " + name);}
  }
  if (!isDeepStrictEqual(receipt.sourceState, llvmSourceSnapshot(plan.source)) ||
      !buildInputsCurrent(receipt.inputFiles)) {throw new Error("LLVM 21 phase inputs changed");}
  return saved;
}

/** @param {string} output @param {BuildPhase[]} phases @param {string} compiler */
function findClangLinkCommand(output, phases, compiler) {
  /** @param {string} text */
  const commands = text => text.split(/\r?\n/).filter(line => line.includes(compiler) && line.includes("-Wl,--export-dynamic") &&
    /(?:^|\s)-o\s+\S*bin\/clang(?:-21)?(?:\s|$)/.test(line));
  const lines = phases.flatMap(phase => commands(readBounded(path.join(output, "build-" + phase.name + ".log"), 256 << 20)));
  if (lines.length !== 1 || !lines[0]?.includes("-fPIC") || lines[0]?.includes("-shared")) {
    throw new Error("LLVM tools phase did not record exactly one PIC Intel clang link with export-dynamic");
  }
  return {command: lines[0], sha256: createHash("sha256").update(lines[0]).digest("hex")};
}

/** @param {string} output */
function verifyBuildComplete(output) {
  const configured = verifyConfigured(output), {receipt, plan, selection} = configured,
    file = path.join(output, "build-complete.json"), result = readBuildJson(file);
  if (result.version !== 1 || result.role !== receipt.role || result.runId !== receipt.runId || result.buildOnly !== true ||
      result.compiled !== true || result.executed !== false || result.phases?.length !== selection.phases.length) {
    throw new Error("LLVM 21 build phase receipts are incomplete or stale");
  }
  for (const [index, phase] of selection.phases.entries()) {
    const file = path.join(output, "phase-" + phase.name + ".json"), reference = result.phases[index];
    if (!reference || reference.path !== file || reference.sha256 !== fileHash(file)) {
      throw new Error("LLVM 21 phase evidence is incomplete or changed: " + phase.name);
    }
    verifyPhaseReceipt(output, plan, receipt, phase);
  }
  const currentGraph = buildFileInputs(Object.keys(receipt.configure.files));
  if (!isDeepStrictEqual(currentGraph, receipt.configure.files) || !isDeepStrictEqual(receipt.sourceState, llvmSourceSnapshot(receipt.plan.source)) ||
      !buildInputsCurrent(receipt.inputFiles)) {
    throw new Error("LLVM 21 inputs changed after the final build phase");
  }
  const link = findClangLinkCommand(output, selection.phases, receipt.plan.tools.cxx);
  if (!isDeepStrictEqual(link, result.clangLinkCommand)) {throw new Error("LLVM 21 clang linker evidence changed");}
  return {configured, result};
}

/** @param {string} root */
function treeSnapshot(root) {
  const base = fs.realpathSync(root), /** @type {Record<string, {type: "symlink", target: string} | {type: "file", bytes: number, sha256: string}>} */ files = {};
  let bytes = 0, entries = 0;
  /** @param {string} directory @param {number} depth */
  function visit(directory, depth) {
    if (depth > 64) {throw new Error("LLVM SDK output inventory exceeds bounds");}
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(directory, entry.name), relative = path.relative(base, file).split(path.sep).join("/");
      if (++entries > 200000 || relative.length > 4096) {throw new Error("LLVM SDK output inventory exceeds bounds");}
      if (entry.isDirectory()) {visit(file, depth + 1); continue;}
      if (entry.isSymbolicLink()) {
        const link = fs.readlinkSync(file), resolved = path.resolve(path.dirname(file), link);
        if (path.isAbsolute(link) || /[\r\n\0]/.test(link) || !containsBuildPath(base, resolved)) {throw new Error("LLVM SDK contains an escaping symlink: " + relative);}
        files[relative] = {type: "symlink", target: link}; bytes += Buffer.byteLength(link);
      } else if (entry.isFile()) {
        const size = fs.statSync(file).size; files[relative] = {type: "file", bytes: size, sha256: fileHash(file)}; bytes += size;
      } else {throw new Error("LLVM SDK contains a special file: " + relative);}
      if (bytes > (8 * 1024 * 1024 * 1024)) {throw new Error("LLVM SDK output exceeds byte limit");}
    }
  }
  visit(base, 0);
  return {files, bytes};
}

/** @param {string} prefix @param {DistributionSelection} selection */
function auditLinuxLlvmPrefix(prefix, selection) {
  const root = fs.realpathSync(prefix), snapshot = treeSnapshot(root), actualArchives = Object.keys(snapshot.files).filter(name => name.endsWith(".a")).sort();
  const expectedArchives = selection.archives.map(row => row.relativeArtifact).sort();
  if (!actualArchives.length || !isDeepStrictEqual(actualArchives, expectedArchives)) {throw new Error("LLVM installed static archive set differs from its generated distribution");}
  for (const name of actualArchives) {
    const file = path.join(root, name), row = snapshot.files[name];
    if (row?.type !== "file" || !isStaticArchive(file)) {throw new Error("LLVM installed archive is missing or not a regular static archive: " + name);}
  }
  for (const name of ["lib/cmake/llvm/LLVMConfig.cmake", "lib/cmake/clang/ClangConfig.cmake", "lib/cmake/lld/LLDConfig.cmake",
    "include/llvm/ADT/StringRef.h", "include/clang/AST/ASTContext.h", "include/lld/Common/Driver.h"]) {
    if (snapshot.files[name]?.type !== "file") {throw new Error("LLVM 21 SDK omitted required upstream export/header: " + name);}
  }
  for (const component of selection.tools) {
    const targetRelative = component.relativeArtifact;
    if (!snapshot.files[targetRelative]) {throw new Error("LLVM SDK omitted source tool output: " + targetRelative);}
  }
  /** @param {string} alias @param {string} target */
  const verifyAlias = (alias, target, description = "required tool alias") => {
    const row = snapshot.files[alias], targetRelative = selection.tools.find(tool => tool.name === target)?.relativeArtifact;
    if (!targetRelative || row?.type !== "symlink" || row.target !== path.basename(targetRelative) ||
        fs.realpathSync(path.join(root, alias)) !== fs.realpathSync(path.join(root, targetRelative))) {
      throw new Error("LLVM SDK omitted or redirected the " + description + ": " + alias);
    }
  };
  if (selection.distributionComponents) {
    for (const [alias, target] of Object.entries(toolAliases)) {verifyAlias(alias, target);}
  }
  for (const name of ["bin/clang", "bin/clang++"]) {
    if (!snapshot.files[name]) {throw new Error("LLVM SDK omitted the clang driver alias: " + name);}
  }
  verifyAlias("bin/ld.lld", "lld", "linker driver alias");
  if (snapshot.bytes < 1) {throw new Error("LLVM SDK installation is empty");}
  return {files: Object.keys(snapshot.files).length, bytes: snapshot.bytes, archives: actualArchives.length, snapshot};
}

/** @param {string} output @param {NodeJS.ProcessEnv} environment */
function installLinuxLlvmSdk(output, environment = process.env) {
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("LLVM SDK installation requires MOM_AOT_BUILD_ONLY=1");}
  const {configured} = verifyBuildComplete(output), {receipt, plan, selection} = configured;
  const resultFile = path.join(output, "install.json");
  if (fs.existsSync(resultFile) || fs.existsSync(plan.prefix)) {throw new Error("LLVM 21 install output must be new and immutable");}
  const env = cleanBuildEnvironment(plan.tools, environment), before = buildFileInputs(Object.keys(receipt.configure.files));
  const runs = [];
  for (const target of installTargets) {
    const args = ["--build", plan.build, "--target", target, "--parallel", String(plan.jobs), "--", "-k", "0"];
    runs.push({target, ...runLogged(path.join(output, target + ".log"), plan.tools.cmake, args, repo, env, installTimeout)});
  }
  const audit = auditLinuxLlvmPrefix(plan.prefix, selection);
  if (!isDeepStrictEqual(before, buildFileInputs(Object.keys(receipt.configure.files))) ||
      !isDeepStrictEqual(receipt.sourceState, llvmSourceSnapshot(plan.source))) {throw new Error("LLVM 21 source or graph changed during install");}
  const installed = {version: 1, role: receipt.role, runId: receipt.runId, installTargets, runs, audit,
    buildCompleteSha256: fileHash(path.join(output, "build-complete.json")),
    graphFiles: receipt.configure.files, buildOnly: true, compiled: true, executed: false};
  writeJsonExclusive(resultFile, installed);
  return installed;
}

/** @param {string} output @param {ReturnType<typeof verifyConfigured>} configured */
function verifyInstallReceipt(output, configured) {
  const {receipt, plan} = configured, file = path.join(output, "install.json"), saved = readBuildJson(file);
  if (saved.version !== 1 || saved.role !== receipt.role || saved.runId !== receipt.runId ||
      !isDeepStrictEqual(saved.installTargets, installTargets) || !Array.isArray(saved.runs) || saved.runs.length !== installTargets.length ||
      saved.buildOnly !== true || saved.compiled !== true || saved.executed !== false ||
      saved.buildCompleteSha256 !== fileHash(path.join(output, "build-complete.json")) ||
      !isDeepStrictEqual(saved.graphFiles, receipt.configure.files)) {throw new Error("LLVM 21 install receipt is incomplete or stale");}
  for (const [index, target] of installTargets.entries()) {
    const run = saved.runs[index], log = path.join(output, target + ".log"), args = ["--build", plan.build, "--target", target,
      "--parallel", String(plan.jobs), "--", "-k", "0"];
    if (!run || run.target !== target || run.command !== plan.tools.cmake || run.timeoutMs !== installTimeout ||
        !isDeepStrictEqual(run.args, args) || run.status !== 0 || run.signal !== null || run.log !== log || run.logSha256 !== fileHash(log)) {
      throw new Error("LLVM 21 install target evidence is missing or stale: " + target);
    }
  }
  return saved;
}

/** @param {string} output */
function auditLinuxLlvmSdk(output) {
  const {configured, result: buildComplete} = verifyBuildComplete(output), {receipt, selection, plan} = configured;
  const installed = verifyInstallReceipt(output, configured), currentAudit = auditLinuxLlvmPrefix(plan.prefix, selection),
    snapshot = currentAudit.snapshot;
  if (installed.version !== 1 || installed.role !== receipt.role || installed.runId !== receipt.runId ||
      !isDeepStrictEqual(installed.installTargets, installTargets) || !Array.isArray(installed.runs) || installed.runs.length !== installTargets.length ||
      installed.buildOnly !== true ||
      installed.compiled !== true || installed.executed !== false || installed.buildCompleteSha256 !== fileHash(path.join(output, "build-complete.json")) ||
      !isDeepStrictEqual(installed.audit, currentAudit) || !isDeepStrictEqual(installed.graphFiles, receipt.configure.files) ||
      !isDeepStrictEqual(receipt.sourceState, llvmSourceSnapshot(plan.source)) ||
      !isDeepStrictEqual(receipt.inputFiles, buildFileInputs(Object.keys(receipt.inputFiles)))) {
    throw new Error("LLVM 21 installed SDK or source receipt is stale");
  }
  const finalFile = path.join(output, "result.json");
  if (fs.existsSync(finalFile)) {throw new Error("LLVM 21 final admission is immutable; use a new output directory");}
  const final = {version: 1, role: receipt.role, runId: receipt.runId, stage: "audited", archiveSha256, catalog: receipt.catalog,
    sourceState: receipt.sourceState, sourceInputs: receipt.inputFiles, toolIdentity: receipt.toolIdentity, plan: receipt.plan,
    planSha256: receipt.planSha256, configure: receipt.configure, buildComplete: {path: path.join(output, "build-complete.json"), sha256: fileHash(path.join(output, "build-complete.json"))},
    install: {path: path.join(output, "install.json"), sha256: fileHash(path.join(output, "install.json"))},
    clangLinkCommand: buildComplete.clangLinkCommand, archives: currentAudit.archives, sdk: {path: plan.prefix, files: currentAudit.files,
      bytes: currentAudit.bytes, snapshot}, buildOnly: true, compiled: true, executed: false, publicationQualified: false};
  writeJsonExclusive(finalFile, final);
  return final;
}

/** Public SDK qualification: every receipt, input and the installed prefix are current. @param {string} output */
function verifyLinuxLlvmAdmission(output) {
  const final = readBuildJson(path.join(output, "result.json")), configured = verifyConfigured(output),
    audit = auditLinuxLlvmPrefix(configured.plan.prefix, configured.selection);
  const {receipt, plan} = configured;
  if (final.version !== 1 || final.role !== "linux-llvm21-source-sdk" || final.stage !== "audited" || final.plan.output !== output ||
      final.archiveSha256 !== archiveSha256 || final.buildOnly !== true || final.compiled !== true || final.executed !== false ||
      final.publicationQualified !== false || !isDeepStrictEqual(final.sdk.snapshot, audit.snapshot) ||
      final.buildComplete.path !== path.join(output, "build-complete.json") || final.install.path !== path.join(output, "install.json") ||
      final.buildComplete.sha256 !== fileHash(final.buildComplete.path) || final.install.sha256 !== fileHash(final.install.path) ||
      !isDeepStrictEqual(final.sourceState, llvmSourceSnapshot(configured.plan.source)) ||
      !buildInputsCurrent(final.sourceInputs) ||
      !isDeepStrictEqual(final.plan, plan) || final.planSha256 !== receipt.planSha256 ||
      !isDeepStrictEqual(final.configure, receipt.configure) || !isDeepStrictEqual(final.sourceInputs, receipt.inputFiles) ||
      !isDeepStrictEqual(final.sourceState, receipt.sourceState) || !isDeepStrictEqual(final.toolIdentity, receipt.toolIdentity) ||
      !isDeepStrictEqual(final.catalog, receipt.catalog)) {
    throw new Error("Linux LLVM 21 SDK admission is stale or incomplete");
  }
  const {result: buildComplete} = verifyBuildComplete(output);
  const installed = verifyInstallReceipt(output, configured);
  if (!isDeepStrictEqual(installed.audit, audit) || final.archives !== audit.archives || !isDeepStrictEqual(final.sdk, {path: configured.plan.prefix,
    files: audit.files, bytes: audit.bytes, snapshot: audit.snapshot}) || !isDeepStrictEqual(final.clangLinkCommand, buildComplete.clangLinkCommand)) {
    throw new Error("Linux LLVM 21 SDK admission no longer matches its installed/build evidence");
  }
  return final;
}

module.exports = {archiveSha256, recipeFiles, toolsComponents, metadataInstallTargets, installTargets, linuxLlvmSdkPlan, verifySourceVersion,
  llvmSourceSnapshot, verifyArchiveCatalog, llvmStaticTargets, linuxDistributionSelection, verifyDistributionGraph, cmakeCache, verifyCMakeCache,
  verifyAmdgpulibCallsCompileFlag, auditLinuxLlvmPrefix, phaseBuildArgs, findClangLinkCommand, verifyLinuxLlvmAdmission};

if (require.main === module) {
  try {
    const [command, ...args] = process.argv.slice(2);
    if (command === "configure" && args.length === 7) {
      const [archive, output, cmake, ninja, cc, cxx, jobs] = args;
      configureLinuxLlvmSdk(output || "", archive || "", {cmake: cmake || "", ninja: ninja || "", cc: cc || "", cxx: cxx || ""}, Number(jobs));
    } else if (command === "build" && args.length === 2) {
      buildLinuxLlvmSdkPhase(args[0] || "", args[1] || "");
    } else if (command === "install" && args.length === 1) {
      installLinuxLlvmSdk(args[0] || "");
    } else if (command === "audit" && args.length === 1) {
      auditLinuxLlvmSdk(args[0] || "");
    } else {throw new Error("Usage: build-native-llvm-sdk.js configure /archive.tar.xz /new-output /cmake /ninja /icx /icpx /jobs | build /configured-output phase | install /built-output | audit /installed-output");}
  } catch (error) {
    console.error("[native-llvm21] " + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  }
}
