"use strict";

// Build-only SYCL JIT resource base producer. Use upstream CMake install metadata for
// source headers, the stamped SDK only for its pinned device-library files, and keep
// admission in prepare-native-sycl-jit-resources.js.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {fileHash, readBuildText, buildFileInputs, sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {verifyPreparedSdkSources} = require("./prepare-native-sdk-sources");
const {sdkIdentity, pins} = require("./probe-static-dpcpp");
const {jitSdkPlan} = require("./native-jit-sdk");
const {absolutePath} = require("./build-native-release");
const {validName} = require("./native-jit-headers");

/** @typedef {(command: string, args: string[], options: Record<string, unknown>) =>
 * {error?: Error, signal?: NodeJS.Signals | null, status: number | null, stdout?: string | null, stderr?: string | null}} ToolRunner */
const defaultRunner = /** @type {ToolRunner} */ (/** @type {unknown} */ (spawnSync));
const repo = path.resolve(__dirname, ".."), MAX_FILES = 16384, MAX_BYTES = 64 << 20, MAX_INSTALLS = 4096;
const REQUIRED = Object.freeze({
  spirv32: {virtual: "lib/clang/23/lib/nvptx64-nvidia-cuda/libspirv.l32.signed_char.bc",
    source: "lib/clang/23/lib/nvptx64-nvidia-cuda/libspirv.l32.signed_char.bc", sha256: "ce08e9d3a5212d95988dddffcf79687d83c5f38bac6afcc2cec64e0bcd2cc76a"},
  spirv64: {virtual: "lib/clang/23/lib/nvptx64-nvidia-cuda/libspirv.l64.signed_char.bc",
    source: "lib/clang/23/lib/nvptx64-nvidia-cuda/libspirv.l64.signed_char.bc", sha256: "132ba9b7a41fe158528400e636f9be6ba66779c35712a2296679fd096c43ba83"},
  libdevice: {virtual: "cuda/nvvm/libdevice/libdevice.10.bc", source: "nvvm/libdevice/libdevice.10.bc",
    sha256: "8c593e48547f85458b6bfbed93a2e6b05484644ccc6b429bb0d8424e06a68b6c"},
});
const RECIPE_FILES = ["scripts/prepare-native-jit-base.js", "scripts/prepare-native-sdk-sources.js", "scripts/probe-static-dpcpp.js",
  "scripts/build-source-inputs.js", "scripts/build-environment.js", "scripts/build-native-release.js", "scripts/native-jit-headers.js", "scripts/native-jit-sdk.js",
  "scripts/prepare-native-sycl-jit-resources.js"];
const FORBIDDEN = /(?:^|[/\\])(?:msvc(?:[._-]|$)|msvcp(?:[._-]|$)|ucrt(?:[._-]|$)|vcruntime(?:[._-]|$)|api-ms-win(?:[._-]|$))|\.(?:dll|exe|obj|pdb|ilk)$/i;

/** @param {string} root @param {string} file */
function within(root, file) {
  const relative = path.relative(root, file);
  return relative === "" || (relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative));
}

/** @param {unknown} value @param {string} label */
function absolute(value, label) {
  if (!absolutePath(value)) {throw new Error("Expected an absolute " + label);}
  return /** @type {string} */ (value);
}

/** @param {string} directory @param {string} label */
function inputDirectory(directory, label) {
  const selected = fs.realpathSync(absolute(directory, label)), stat = fs.statSync(selected);
  if (!stat.isDirectory()) {throw new Error(label + " is not a directory");}
  return selected;
}

/** @param {string} file @param {string} root @param {string} label */
function inputFile(file, root, label) {
  const selected = fs.realpathSync(absolute(file, label));
  if (!within(root, selected) || !fs.statSync(selected).isFile()) {throw new Error(label + " is not a regular file inside its admitted root");}
  return selected;
}

/** @param {string} value @param {string} label */
function relativeName(value, label) {
  if (!validName(value) || value.includes(":") || value.includes("\\") || value.split("/").some(part => !part || part === "." || part === "..")) {
    throw new Error("Invalid relative " + label);
  }
  return value;
}

/** Keep the default embedded compiler identity when configured paths match it; otherwise
 * reconstruct the public JIT plan with the exact explicit CMake host compiler pair.
 * @param {string | undefined} cc @param {string | undefined} cxx @param {string} sdkRoot */
function jitPlanCompilerTools(cc, cxx, sdkRoot) {
  if (!cc || !cxx || !absolutePath(cc) || !absolutePath(cxx) || !absolutePath(sdkRoot)) {
    throw new Error("Configured JIT host compilers must be explicit absolute paths");
  }
  if (cc === path.join(sdkRoot, "bin/clang") && cxx === path.join(sdkRoot, "bin/clang++")) {return {};}
  return {cc, cxx};
}

/** Parse CMake's generated cache without accepting duplicate or malformed keys.
 * @param {string} text */
function parseCache(text) {
  /** @type {Map<string, string>} */ const values = new Map();
  for (const line of text.split(/\r?\n/)) {
    if (!line || line.startsWith("#") || line.startsWith("//")) {continue;}
    const match = line.match(/^([^:=]+):[^=]+=([^\r\n]*)$/), key = match?.[1], value = match?.[2];
    if (!key || value === undefined || values.has(key)) {throw new Error("Malformed or duplicate CMake cache entry");}
    values.set(key, value);
  }
  return values;
}

/** Check the restricted CMake-generated install language before executing it.
 * @param {string} text @param {string} component @returns {Array<{destination: string, type: string, sources: string[]}>} */
function parseInstallComponent(text, component) {
  if (!/^[a-z][a-z0-9-]{0,63}$/.test(component) || text.length > (8 << 20)) {throw new Error("Invalid install script or component");}
  const commandNames = [...text.matchAll(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(/gm)].map(match => (match[1] || "").toLowerCase());
  if (commandNames.some(name => !["if", "else", "endif", "set", "string", "message", "file"].includes(name)) ||
      [...text.matchAll(/\bfile\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)/gi)].some(match => (match[1] || "").toUpperCase() !== "INSTALL")) {
    throw new Error("Install script contains an unreviewed CMake command");
  }
  const blockPattern = new RegExp(`if\\(CMAKE_INSTALL_COMPONENT STREQUAL "${component}" OR NOT CMAKE_INSTALL_COMPONENT\\)\\s*([\\s\\S]*?)\\nendif\\(\\)`, "g");
  const operations = [];
  for (const block of text.matchAll(blockPattern)) {
    const body = block[1] || "", installPattern = /file\(INSTALL\s+DESTINATION\s+"\$\{CMAKE_INSTALL_PREFIX\}([^"\r\n]*)"\s+TYPE\s+(FILE|DIRECTORY)\s+FILES([\s\S]*?)\)/g;
    let last = 0;
    for (const operation of body.matchAll(installPattern)) {
      if (/\S/.test(body.slice(last, operation.index))) {throw new Error("Install component contains unexpected CMake statements");}
      const destination = relativeName((operation[1] || "").replace(/^\/+|\/+$/g, ""), "install destination");
      const sources = [...(operation[3] || "").matchAll(/"([^"\r\n]+)"/g)].map(match => match[1] || "");
      const residue = (operation[3] || "").replace(/"[^"\r\n]+"/g, "");
      if (!sources.length || sources.length > MAX_INSTALLS || /\S/.test(residue)) {throw new Error("Malformed CMake install source list");}
      operations.push({destination, type: (operation[2] || "").toUpperCase(), sources});
      last = (operation.index || 0) + operation[0].length;
    }
    if (/\S/.test(body.slice(last))) {throw new Error("Install component contains unexpected CMake statements");}
  }
  if (!operations.length || operations.length > MAX_INSTALLS) {throw new Error("Generated CMake component is missing or excessive");}
  return operations;
}

/** Create the explicit manifest format consumed by the existing resource preparer.
 * @param {string} root @returns {{version: 1, files: Array<{source: string, virtual: string, sha256: string}>}} */
function baseManifest(root) {
  /** @type {Array<{source: string, virtual: string, sha256: string}>} */ const files = [];
  const names = new Set(), folded = new Set();
  let bytes = 0;
  /** @param {string} directory */
  function visit(directory) {
    for (const item of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(directory, item.name), relative = path.relative(root, file).split(path.sep).join("/");
      relativeName(relative, "base file");
      if (FORBIDDEN.test(relative)) {throw new Error("Base contains a forbidden native payload: " + relative);}
      if (item.isDirectory()) {visit(file); continue;}
      if (!item.isFile() || files.length >= MAX_FILES) {throw new Error("Base contains a link, special file, or too many entries");}
      const key = relative.toLowerCase(), stat = fs.statSync(file);
      if (names.has(relative) || folded.has(key) || (bytes += stat.size) > MAX_BYTES) {throw new Error("Base contains duplicate paths or exceeds byte bounds");}
      names.add(relative); folded.add(key);
      files.push({source: relative, virtual: relative, sha256: fileHash(file)});
    }
  }
  visit(root);
  if (!files.length) {throw new Error("Base resource tree is empty");}
  return {version: 1, files};
}

/** @param {string} file @param {string} expected @param {string} label */
function verifyBitcode(file, expected, label) {
  if (fileHash(file) !== expected || fs.statSync(file).size < 16) {throw new Error(label + " differs from its pinned SHA-256");}
  const fd = fs.openSync(file, "r"), magic = Buffer.alloc(4);
  try {fs.readSync(fd, magic, 0, 4, 0);} finally {fs.closeSync(fd);}
  if (!magic.equals(Buffer.from([0x42, 0x43, 0xc0, 0xde]))) {throw new Error(label + " is not LLVM bitcode");}
}

/** Bind a staged device-library input by its canonical in-root path and pinned bytes.
 * Package-manager ownership is not stable after a verified SDK subtree is transported.
 * @param {string} root @param {{source: string, virtual: string, sha256: string}} required @param {string} label
 * @returns {{file: string, sha256: string}} */
function pinnedBitcodeInput(root, required, label) {
  if (!required || typeof required !== "object" || Array.isArray(required) || Object.keys(required).sort().join(",") !== "sha256,source,virtual" ||
      !/^[a-f0-9]{64}$/.test(required.sha256 || "")) {throw new Error(label + " pin is malformed");}
  const selectedRoot = inputDirectory(root, label + " root"), source = relativeName(required.source, label + " source"),
    virtual = relativeName(required.virtual, label + " virtual path");
  const file = inputFile(path.join(selectedRoot, ...source.split("/")), selectedRoot, label);
  verifyBitcode(file, required.sha256, virtual);
  return {file, sha256: fileHash(file)};
}

/** @param {string} gitRoot @param {NodeJS.ProcessEnv} environment @param {ToolRunner} [run] */
function pinnedCommitDate(gitRoot, environment, run = defaultRunner) {
  const env = {PATH: "/usr/bin:/bin", LC_ALL: "C", GIT_NO_LAZY_FETCH: "1", GIT_NO_REPLACE_OBJECTS: "1",
    GIT_OPTIONAL_LOCKS: "0", GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null", GIT_ATTR_NOSYSTEM: "1"};
  const result = run("git", ["--git-dir", gitRoot, "show", "-s", "--format=%H %cd", "--date=format:%Y%m%d", pins.source],
    {env: {...env, MOM_AOT_BUILD_ONLY: environment["MOM_AOT_BUILD_ONLY"] || ""}, encoding: "utf8", timeout: 30000, maxBuffer: 4096});
  if (result.error || result.signal || result.status !== 0) {throw new Error("Pinned DPC++ Git commit date could not be read offline");}
  const match = (result.stdout || "").trim().match(/^([a-f0-9]{40}) (\d{8})$/);
  if (!match || match[1] !== pins.source) {throw new Error("DPC++ Git root does not contain the pinned source commit");}
  return match[2];
}

/** @param {string} root @param {string} relative @param {string} label
 * @returns {{directory: string, files: Array<{file: string, name: string, sha256: string}>}} */
function sourceTreeFiles(root, relative, label) {
  const directory = fs.realpathSync(path.join(root, ...relative.split("/")));
  if (!within(root, directory) || !fs.statSync(directory).isDirectory()) {throw new Error(label + " directory escapes its admitted root");}
  const files = [];
  for (const item of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
    const file = path.join(directory, item.name), real = fs.realpathSync(file);
    if (!within(root, real) || !item.isFile() || !fs.statSync(real).isFile() || files.length >= MAX_FILES) {
      throw new Error(label + " contains a link, special file, or too many entries");
    }
    files.push({file: real, name: item.name, sha256: fileHash(real)});
  }
  if (!files.length) {throw new Error(label + " source directory is empty");}
  return {directory, files};
}

/** @param {string} command @param {string[]} args @param {{cwd: string, env: NodeJS.ProcessEnv, log: string, timeout?: number}} options
 * @param {ToolRunner} [run] */
function runLogged(command, args, options, run = defaultRunner) {
  const fd = fs.openSync(options.log, "wx");
  let result;
  try {
    result = run(command, args, {cwd: options.cwd, env: options.env, encoding: "utf8", timeout: options.timeout || 300000,
      maxBuffer: 8 << 20, stdio: ["ignore", fd, fd]});
  } finally {fs.closeSync(fd);}
  if (result.error || result.signal || result.status !== 0) {throw new Error(`Build-only ${path.basename(command)} failed (${result.signal || result.status}); log: ${options.log}`);}
}

/** Build and pin the matching LLVM TableGen executable from the configured source tree.
 * @param {string} cmake @param {string} buildRoot @param {string} cwd @param {NodeJS.ProcessEnv} env @param {string} log
 * @param {ToolRunner} [run] @returns {{path: string, sha256: string}} */
function buildLlvmTableGen(cmake, buildRoot, cwd, env, log, run = defaultRunner) {
  const root = inputDirectory(buildRoot, "LLVM build root");
  runLogged(cmake, ["--build", root, "--target", "llvm-tblgen", "--parallel", "1"], {cwd, env, log}, run);
  const file = inputFile(path.join(root, "bin/llvm-tblgen"), root, "built LLVM TableGen tool");
  fs.accessSync(file, fs.constants.X_OK);
  return {path: file, sha256: fileHash(file)};
}

/** Build one pinned, explicit SYCL JIT resource base without compiling a compiler or
 * executing a device. The output contains base/, base-manifest.json and result.json.
 * @param {{sdkRoot: string, cudaRoot: string, buildRoot: string, sourceReceipt: string, dpcppGitRoot: string, output: string}} options
 * @param {NodeJS.ProcessEnv} [environment]
 * @param {{verifySources?: typeof verifyPreparedSdkSources, run?: ToolRunner}} [dependencies] */
function prepareNativeJitBase(options, environment = process.env, dependencies = {}) {
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("JIT base production requires Linux and MOM_AOT_BUILD_ONLY=1");}
  const selected = {
    sdkRoot: inputDirectory(options?.sdkRoot, "DPC++ SDK root"), cudaRoot: inputDirectory(options?.cudaRoot, "CUDA SDK root"),
    buildRoot: inputDirectory(options?.buildRoot, "CMake build root"), dpcppGitRoot: inputDirectory(options?.dpcppGitRoot, "pinned DPC++ Git root"),
  };
  const sourceReceipt = fs.realpathSync(absolute(options?.sourceReceipt, "compiler source receipt"));
  if (!fs.statSync(sourceReceipt).isFile()) {throw new Error("Compiler source receipt is not a regular file");}
  const output = absolute(options?.output, "new output directory"), parent = fs.realpathSync(path.dirname(output));
  const outputPath = path.join(parent, path.basename(output));
  if (fs.lstatSync(outputPath, {throwIfNoEntry: false})) {throw new Error("JIT base output already exists");}
  const sourceVerifier = dependencies.verifySources || verifyPreparedSdkSources, run = dependencies.run || defaultRunner;
  const sourceReceiptSha256 = fileHash(sourceReceipt), source = sourceVerifier(sourceReceipt, "compiler");
  if (!source.prepared?.dpcpp || !source.prepared?.openclHeaders || source.prepared.dpcpp.revision !== pins.source ||
      source.prepared.openclHeaders.revision !== pins.openclHeaders) {throw new Error("Compiler source receipt lacks the pinned DPC++ or OpenCL-Headers trees");}
  const sourceRoots = Object.values(source.prepared).map(component => inputDirectory(component.root, "prepared compiler source root"));
  const dpcppRoot = inputDirectory(source.prepared.dpcpp.root, "prepared DPC++ source root");
  const openclRoot = inputDirectory(source.prepared.openclHeaders.root, "prepared OpenCL-Headers source root");
  const inputsToAvoid = [repo, selected.sdkRoot, selected.cudaRoot, selected.buildRoot, selected.dpcppGitRoot, ...sourceRoots];
  if (inputsToAvoid.some(root => within(root, outputPath) || within(outputPath, root)) || outputPath === sourceReceipt) {
    throw new Error("JIT base output must be outside every source, SDK, Git, build and repository input");
  }
  const sourceRecipe = sourceInputs(repo, [], RECIPE_FILES);
  const sourceDate = pinnedCommitDate(selected.dpcppGitRoot, environment, run);
  const sdkCompiler = inputFile(path.join(selected.sdkRoot, "bin/clang++"), selected.sdkRoot, "DPC++ compiler");
  const sdkStamp = inputFile(path.join(selected.sdkRoot, ".mom-toolchain-sha256"), selected.sdkRoot, "DPC++ archive stamp");
  const version = run(sdkCompiler, ["--version"], {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20,
    env: {PATH: "/usr/bin:/bin", LC_ALL: "C"}});
  if (version.error || version.signal || version.status !== 0) {throw new Error("Pinned DPC++ compiler version query failed");}
  sdkIdentity(version.stdout || "", fs.readFileSync(sdkStamp, "utf8"));
  const cacheFile = inputFile(path.join(selected.buildRoot, "CMakeCache.txt"), selected.buildRoot, "CMake cache");
  const cache = parseCache(readBuildText(cacheFile, 8 << 20));
  const expectedCache = {
    CMAKE_HOME_DIRECTORY: path.join(dpcppRoot, "llvm"),
    LLVM_EXTERNAL_LIBDEVICE_SOURCE_DIR: path.join(dpcppRoot, "libdevice"),
    LLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR: path.join(dpcppRoot, "llvm-spirv"),
    LLVM_EXTERNAL_SYCL_JIT_SOURCE_DIR: path.join(dpcppRoot, "sycl-jit"),
    LLVM_EXTERNAL_SYCL_SOURCE_DIR: path.join(dpcppRoot, "sycl"),
    SYCL_HEADERS_ONLY: "ON", SYCL_JIT_STATIC: "ON", SYCL_COMPILER_VERSION: sourceDate,
  };
  for (const [key, value] of Object.entries(expectedCache)) {
    if (cache.get(key) !== value) {throw new Error(`${key} must be ${value}; reconfigure the private headers-only build with -D${key}=${value}`);}
  }
  const cmake = absolute(cache.get("CMAKE_COMMAND"), "configured CMake executable"), realCmake = fs.realpathSync(cmake);
  const cmakeFile = inputFile(realCmake, path.dirname(realCmake), "CMake executable");
  const ninja = cache.get("CMAKE_MAKE_PROGRAM");
  if (!ninja || !absolutePath(ninja)) {throw new Error("CMake cache does not select an absolute Ninja executable");}
  const realNinja = fs.realpathSync(ninja), ninjaFile = inputFile(realNinja, path.dirname(realNinja), "Ninja executable");
  const configuredOverlay = absolute(cache.get("SYCL_JIT_RESOURCE_OVERLAY"), "SYCL JIT resource overlay");
  const resourceOverlay = inputDirectory(configuredOverlay, "SYCL JIT resource overlay"), configureRoot = path.dirname(selected.buildRoot);
  if (resourceOverlay !== configuredOverlay || [configureRoot, resourceOverlay].some(root => within(root, outputPath) || within(outputPath, root))) {
    throw new Error("JIT base output must be outside the configured build and resource overlay inputs");
  }
  const configuredCc = cache.get("CMAKE_C_COMPILER"), configuredCxx = cache.get("CMAKE_CXX_COMPILER"),
    ccReal = fs.realpathSync(absolute(configuredCc, "configured C compiler")), cxxReal = fs.realpathSync(absolute(configuredCxx, "configured C++ compiler")),
    ccFile = inputFile(ccReal, path.dirname(ccReal), "configured C compiler"), cxxFile = inputFile(cxxReal, path.dirname(cxxReal), "configured C++ compiler");
  fs.accessSync(ccFile, fs.constants.X_OK); fs.accessSync(cxxFile, fs.constants.X_OK);
  const planRoots = Object.fromEntries(Object.entries(source.prepared).map(([name, component]) => [name, component.root])),
    compilerTools = jitPlanCompilerTools(configuredCc, configuredCxx, selected.sdkRoot),
    plan = jitSdkPlan("linux", planRoots, selected.sdkRoot, resourceOverlay, configureRoot, {cmake, ninja, ...compilerTools}, 1);
  if (plan.build !== selected.buildRoot || cache.get("CMAKE_GENERATOR") !== "Ninja" || cache.get("CMAKE_BUILD_TYPE") !== "Release") {
    throw new Error("Headers-only build directory does not match the maintained Linux JIT SDK plan");
  }
  for (const [key, value] of Object.entries(plan.definitions)) {
    if (cache.get(key) !== value) {throw new Error(`CMake cache does not match maintained JIT plan: ${key}`);}
  }
  /** @type {string[]} */ const buildToolFiles = [];
  for (const key of ["CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_AR", "CMAKE_C_COMPILER_AR", "CMAKE_CXX_COMPILER_AR",
    "CMAKE_LINKER", "CMAKE_NM", "CMAKE_OBJCOPY", "CMAKE_OBJDUMP", "CMAKE_RANLIB", "CMAKE_STRIP"]) {
    const tool = cache.get(key);
    if (!tool || tool.endsWith("-NOTFOUND")) {continue;}
    const realTool = fs.realpathSync(absolute(tool, "configured " + key));
    buildToolFiles.push(inputFile(realTool, path.dirname(realTool), "configured " + key));
  }
  const buildNinja = inputFile(path.join(selected.buildRoot, "build.ninja"), selected.buildRoot, "Ninja build graph");
  const generatedVersion = inputFile(path.join(selected.buildRoot, "include/sycl/version.hpp"), selected.buildRoot, "generated SYCL version header");
  const generatedDate = readBuildText(generatedVersion, 1 << 20).match(/^#define __SYCL_COMPILER_VERSION (\d{8})$/m)?.[1];
  if (generatedDate !== sourceDate) {throw new Error(`Generated SYCL version is ${generatedDate || "missing"}, pinned DPC++ commit date is ${sourceDate}; reconfigure with -DSYCL_COMPILER_VERSION=${sourceDate}`);}
  const syclScript = inputFile(path.join(selected.buildRoot, "tools/sycl/cmake_install.cmake"), selected.buildRoot, "SYCL install script");
  const clangScript = inputFile(path.join(selected.buildRoot, "tools/clang/lib/Headers/cmake_install.cmake"), selected.buildRoot, "Clang resource install script");
  const deviceScript = inputFile(path.join(selected.buildRoot, "tools/libdevice/cmake_install.cmake"), selected.buildRoot, "libsycldevice install script");
  const syclOps = parseInstallComponent(readBuildText(syclScript, 8 << 20), "sycl-headers");
  const clangOps = parseInstallComponent(readBuildText(clangScript, 8 << 20), "clang-resource-headers");
  const deviceOps = parseInstallComponent(readBuildText(deviceScript, 8 << 20), "libsycldevice");
  const allowedHeaderRoots = [selected.buildRoot, dpcppRoot];
  const generatedSyclHeaders = new Set(["feature_test.hpp", "version.hpp", "device_aspect_macros.hpp"]
    .map(name => path.join(selected.buildRoot, "include/sycl", name)));
  const syclSourceDirectories = new Set([
    path.join(dpcppRoot, "sycl/include/sycl"), path.join(dpcppRoot, "sycl/include/CL"), path.join(dpcppRoot, "sycl/include/std"),
    path.join(dpcppRoot, "unified-runtime/include/unified-runtime"),
  ]);
  const clangSourceDirectory = path.join(dpcppRoot, "clang/lib/Headers"), clangGeneratedDirectory = path.join(selected.buildRoot, "tools/clang/lib/Headers");
  /** @type {string[]} */ const clangSourceFiles = [];
  /** @type {string[]} */ const generatedClangFiles = [];
  const headerComponents = [{name: "SYCL", operations: syclOps, expectedDestinations: ["include", "include/sycl"]},
    {name: "Clang", operations: clangOps, expectedDestinations: ["lib/clang/23/include"]}];
  for (const {name, operations, expectedDestinations} of headerComponents) {
    for (const operation of operations) {
      const clangResourceRoot = "lib/clang/23/include";
      const allowedDestination = name === "Clang" ? operation.destination === clangResourceRoot || operation.destination.startsWith(clangResourceRoot + "/") :
        expectedDestinations.includes(operation.destination);
      if (!allowedDestination) {throw new Error(`${name} install destination is outside the JIT resource layout`);}
      if (name === "Clang" && operation.type !== "FILE" || operation.type !== "FILE" && operation.type !== "DIRECTORY") {
        throw new Error(`${name} install component has an unexpected file type`);
      }
      for (const file of operation.sources) {
        const selectedFile = absolute(file, name + " install source"), resolved = path.resolve(selectedFile);
        const isDirectory = operation.type === "DIRECTORY";
        const generated = !isDirectory && (name === "SYCL" && generatedSyclHeaders.has(resolved) ||
          name === "Clang" && within(clangGeneratedDirectory, resolved));
        const real = generated ? resolved : fs.realpathSync(selectedFile);
        const existingGenerated = generated ? fs.lstatSync(real, {throwIfNoEntry: false}) : undefined;
        const permitted = name === "Clang" ? !isDirectory && (generated || within(clangSourceDirectory, real)) :
          isDirectory ? syclSourceDirectories.has(real) : generated;
        if (!allowedHeaderRoots.some(root => within(root, real)) || !permitted ||
            generated && existingGenerated && !existingGenerated.isFile() ||
            !generated && !fs.statSync(real)[isDirectory ? "isDirectory" : "isFile"]()) {
          throw new Error(`${name} install source escapes its admitted build/source roots`);
        }
        if (name === "Clang") {(generated ? generatedClangFiles : clangSourceFiles).push(real);}
      }
    }
  }
  /** @type {Array<{file: string, virtual: string}>} */ const deviceFiles = [];
  for (const operation of deviceOps) {
    if (operation.type !== "FILE" || !["lib", "lib/dpcpp-7/sycl"].includes(operation.destination)) {throw new Error("libsycldevice install metadata has an unexpected destination or type");}
    for (const sourcePath of operation.sources) {
      const sourceFile = path.resolve(sourcePath), relative = path.relative(selected.buildRoot, sourceFile).split(path.sep).join("/");
      const virtual = path.posix.join(operation.destination, path.posix.basename(sourceFile));
      if (!within(selected.buildRoot, sourceFile) || relative !== virtual || deviceFiles.some(row => row.virtual.toLowerCase() === virtual.toLowerCase())) {
        throw new Error("libsycldevice install metadata escapes or aliases its upstream SDK path");
      }
      const sdkFile = inputFile(path.join(selected.sdkRoot, ...virtual.split("/")), selected.sdkRoot, "pinned libsycldevice input");
      deviceFiles.push({file: sdkFile, virtual});
    }
  }
  if (!deviceFiles.length || deviceFiles.length > MAX_INSTALLS) {throw new Error("Generated libsycldevice whitelist is empty or excessive");}
  const openclCmake = inputFile(path.join(openclRoot, "CMakeLists.txt"), openclRoot, "OpenCL-Headers CMake source");
  const openclCmakeText = readBuildText(openclCmake, 1 << 20);
  if (!/install\s*\(\s*DIRECTORY\s+CL\s+DESTINATION\s+\$\{CMAKE_INSTALL_INCLUDEDIR\}/m.test(openclCmakeText)) {
    throw new Error("Pinned OpenCL-Headers source no longer declares its upstream CL directory install");
  }
  const opencl = sourceTreeFiles(openclRoot, "CL", "OpenCL-Headers");
  const cudaLibdevice = pinnedBitcodeInput(selected.cudaRoot, REQUIRED.libdevice, "CUDA libdevice"), cudaFile = cudaLibdevice.file;
  const dpcppRequired = [REQUIRED.spirv32, REQUIRED.spirv64].map(row => ({row,
    file: inputFile(path.join(selected.sdkRoot, ...row.source.split("/")), selected.sdkRoot, "pinned SPIR-V bitcode")}));
  for (const {row, file} of dpcppRequired) {verifyBitcode(file, row.sha256, row.virtual);}
  verifySourceInputs(sourceRecipe, sourceInputs(repo, [], RECIPE_FILES), "JIT base producer recipe");
  if (fileHash(sourceReceipt) !== sourceReceiptSha256) {throw new Error("Compiler source receipt changed during admission");}

  const paths = [sourceReceipt, cmakeFile, ninjaFile, cacheFile, syclScript, clangScript, deviceScript, generatedVersion,
    buildNinja, ccFile, cxxFile, ...buildToolFiles, ...clangSourceFiles, openclCmake, ...opencl.files.map(file => file.file), sdkCompiler, sdkStamp,
    ...deviceFiles.map(file => file.file),
    ...dpcppRequired.map(item => item.file), cudaFile];
  const initialInputs = buildFileInputs(paths.map(file => fs.realpathSync(file)));
  const parentStat = fs.statSync(parent);
  if (!parentStat.isDirectory()) {throw new Error("JIT base output parent is not a directory");}
  const staging = fs.mkdtempSync(path.join(parent, ".mom-native-jit-base-")), base = path.join(staging, "base"), logs = path.join(staging, "logs");
  fs.mkdirSync(base); fs.mkdirSync(logs);
  const env = {PATH: path.dirname(cmakeFile), LC_ALL: "C", LANG: "C"};
  try {
    const headersTargetLog = path.join(logs, "sycl-headers-build.log");
    runLogged(cmakeFile, ["--build", selected.buildRoot, "--target", "sycl-headers", "--parallel", "1"],
      {cwd: staging, env, log: headersTargetLog}, run);
    runLogged(cmakeFile, ["--build", selected.buildRoot, "--target", "clang-resource-headers", "--parallel", "1"],
      {cwd: staging, env, log: path.join(logs, "clang-resource-headers-build.log")}, run);
    const llvmTableGen = buildLlvmTableGen(cmakeFile, selected.buildRoot, staging, env,
      path.join(logs, "llvm-tablegen-build.log"), run);
    const generatedHeaders = ["feature_test.hpp", "version.hpp", "device_aspect_macros.hpp"].map(name =>
      inputFile(path.join(selected.buildRoot, "include/sycl", name), selected.buildRoot, "generated SYCL header"));
    const generatedClangHeaders = [...new Set(generatedClangFiles)].map(file =>
      inputFile(file, selected.buildRoot, "generated Clang resource header"));
    const clangTableGen = inputFile(path.join(selected.buildRoot, "bin/clang-tblgen"), selected.buildRoot, "built Clang TableGen tool");
    const generatedAfterBuild = buildFileInputs([...generatedHeaders, ...generatedClangHeaders, clangTableGen, llvmTableGen.path]);
    runLogged(cmakeFile, ["-D", "CMAKE_INSTALL_COMPONENT=sycl-headers", "-D", "CMAKE_INSTALL_PREFIX=" + base,
      "-D", "CMAKE_INSTALL_CONFIG_NAME=Release", "-D", "CMAKE_INSTALL_MESSAGE=NEVER", "-P", syclScript],
    {cwd: staging, env, log: path.join(logs, "sycl-headers-install.log")}, run);
    runLogged(cmakeFile, ["-D", "CMAKE_INSTALL_COMPONENT=clang-resource-headers", "-D", "CMAKE_INSTALL_PREFIX=" + base,
      "-D", "CMAKE_INSTALL_CONFIG_NAME=Release", "-D", "CMAKE_INSTALL_MESSAGE=NEVER", "-P", clangScript],
    {cwd: staging, env, log: path.join(logs, "clang-resource-headers-install.log")}, run);
    // OpenCL-Headers has no generated component in SYCL_HEADERS_ONLY configurations;
    // mirror its pinned upstream install(DIRECTORY CL DESTINATION include) declaration.
    fs.mkdirSync(path.join(base, "include/CL"), {recursive: true});
    for (const item of opencl.files) {
      const destination = path.join(base, "include/CL", item.name);
      fs.copyFileSync(item.file, destination, fs.constants.COPYFILE_EXCL);
      fs.chmodSync(destination, 0o644);
    }
    for (const item of deviceFiles) {
      const destination = path.join(base, ...item.virtual.split("/"));
      fs.mkdirSync(path.dirname(destination), {recursive: true});
      fs.copyFileSync(item.file, destination, fs.constants.COPYFILE_EXCL); fs.chmodSync(destination, 0o644);
    }
    for (const {row, file} of [...dpcppRequired, {row: REQUIRED.libdevice, file: cudaFile}]) {
      const destination = path.join(base, ...row.virtual.split("/"));
      fs.mkdirSync(path.dirname(destination), {recursive: true});
      fs.copyFileSync(file, destination, fs.constants.COPYFILE_EXCL); fs.chmodSync(destination, 0o644);
    }
    const manifest = baseManifest(base), manifestFile = path.join(staging, "base-manifest.json");
    const manifestText = JSON.stringify(manifest, null, 2) + "\n";
    fs.writeFileSync(manifestFile, manifestText, {flag: "wx"});
    for (const {row} of [...dpcppRequired, {row: REQUIRED.libdevice, file: cudaFile}]) {
      const entry = manifest.files.find(file => file.virtual === row.virtual);
      if (!entry || entry.sha256 !== row.sha256) {throw new Error("Required pinned bitcode is absent from generated base manifest");}
    }
    const trackedInputs = [...paths, ...generatedHeaders, ...generatedClangHeaders, clangTableGen, llvmTableGen.path].map(file => fs.realpathSync(file));
    const currentInputs = buildFileInputs(trackedInputs);
    const initialCurrentInputs = buildFileInputs(paths.map(file => fs.realpathSync(file)));
    for (const [file, hash] of Object.entries(initialInputs)) {
      if (initialCurrentInputs[file] !== hash || currentInputs[file] !== hash) {throw new Error("JIT base input changed during staging: " + file);}
    }
    for (const [file, hash] of Object.entries(generatedAfterBuild)) {
      if (currentInputs[file] !== hash) {throw new Error("Generated JIT header changed during install: " + file);}
    }
    const sourceAfter = sourceVerifier(sourceReceipt, "compiler");
    if (fileHash(sourceReceipt) !== sourceReceiptSha256 || sourceAfter.prepared.dpcpp.revision !== pins.source) {
      throw new Error("Compiler source cohort changed during JIT base staging");
    }
    verifySourceInputs(sourceRecipe, sourceInputs(repo, [], RECIPE_FILES), "JIT base producer recipe");
    const cmakeVersion = run(cmakeFile, ["--version"], {encoding: "utf8", timeout: 10000, maxBuffer: 4096, env});
    if (cmakeVersion.error || cmakeVersion.signal || cmakeVersion.status !== 0) {throw new Error("CMake version query failed");}
    /** @type {Record<string, {path: string, firstLine: string, sha256: string}>} */ const hostCompilerVersions = {};
    for (const [name, file] of Object.entries({cc: ccFile, cxx: cxxFile})) {
      const identity = run(file, ["--version"], {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20, env});
      if (identity.error || identity.signal || identity.status !== 0 || !identity.stdout?.trim() || identity.stdout.includes("\0")) {
        throw new Error("Configured JIT host compiler identity query failed: " + name);
      }
      hostCompilerVersions[name] = {path: file, firstLine: identity.stdout.trim().split(/\r?\n/, 1)[0] || "", sha256: fileHash(file)};
    }
    const result = {version: 1, baseRoot: path.join(outputPath, "base"), baseManifest: path.join(outputPath, "base-manifest.json"),
      baseManifestSha256: fileHash(manifestFile), files: manifest.files.length, bytes: manifest.files.reduce((total, file) => total + fs.statSync(path.join(base, file.source)).size, 0),
      sourceReceipt: {path: sourceReceipt, sha256: sourceReceiptSha256, target: source.target,
        revisions: Object.fromEntries(Object.entries(source.prepared).map(([name, component]) => [name, component.revision]))},
      dpcpp: {sourceCommit: pins.source, commitDate: sourceDate, sdkRoot: selected.sdkRoot, sdkArchiveSha256: pins.sdk, compilerVersion: (version.stdout || "").split(/\r?\n/)[0],
        stampSha256: fileHash(sdkStamp)},
      cuda: {root: selected.cudaRoot, libdevice: {path: cudaFile, sha256: cudaLibdevice.sha256}},
      build: {root: selected.buildRoot, cmake: cmakeFile, cmakeVersion: (cmakeVersion.stdout || "").split(/\r?\n/)[0],
        hostCompilerVersions,
        cmakeCacheSha256: fileHash(cacheFile), ninjaGraphSha256: fileHash(buildNinja), clangTableGenSha256: fileHash(clangTableGen),
        llvmTableGenSha256: llvmTableGen.sha256,
        buildTools: Object.fromEntries([...new Set(buildToolFiles)].map(file => [file, fileHash(file)])),
        syclInstallSha256: fileHash(syclScript), clangHeadersInstallSha256: fileHash(clangScript),
        libsycldeviceInstallSha256: fileHash(deviceScript), libsycldeviceFiles: deviceFiles.map(item => item.virtual).sort()},
      inputs: currentInputs, recipe: sourceRecipe, required: REQUIRED, buildOnly: true, compiled: false, executed: false,
      publicationQualified: false, scope: "Pinned, build-only SYCL JIT resource base staging. No compiler SDK qualification, device execution, GPU or publication claim."};
    fs.writeFileSync(path.join(staging, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
    if (fs.lstatSync(outputPath, {throwIfNoEntry: false})) {throw new Error("JIT base output appeared during staging");}
    fs.renameSync(staging, outputPath);
    return result;
  } catch (error) {
    // Preserve command logs and partial stage as evidence; the requested output path stays absent.
    try {
      fs.writeFileSync(path.join(staging, "failure.txt"), (error instanceof Error ? error.message : "JIT base staging failed") + "\n", {flag: "wx"});
    } catch {
      // A partial stage may already contain its first failure record.
    }
    throw new Error((error instanceof Error ? error.message : "JIT base staging failed") + "; staged evidence: " + staging, {cause: error});
  }
}

module.exports = {prepareNativeJitBase, parseCache, parseInstallComponent, baseManifest, verifyBitcode, pinnedBitcodeInput, pinnedCommitDate, sourceTreeFiles,
  jitPlanCompilerTools, buildLlvmTableGen, REQUIRED};

if (require.main === module) {
  try {
    const args = process.argv.slice(2), flags = ["--sdk", "--cuda", "--build", "--sources", "--dpcpp-git", "--output"];
    if (args.length !== flags.length * 2) {throw new Error("Usage: prepare-native-jit-base.js --sdk <DPC++> --cuda <CUDA> --build <headers-only-build> --sources <compiler-sources.json> --dpcpp-git <bare-DPC++-Git-root> --output <new-directory>");}
    /** @type {Record<string, string>} */ const options = {};
    const optionNames = {"--sdk": "sdkRoot", "--cuda": "cudaRoot", "--build": "buildRoot", "--sources": "sourceReceipt",
      "--dpcpp-git": "dpcppGitRoot", "--output": "output"};
    for (let index = 0; index < args.length; index += 2) {
      const flag = args[index], value = args[index + 1], key = optionNames[/** @type {keyof typeof optionNames} */ (flag || "")];
      if (!key || !value || Object.hasOwn(options, key)) {throw new Error("Invalid or duplicate JIT base option");}
      options[key] = value;
    }
    if (Object.keys(options).length !== flags.length) {throw new Error("Missing JIT base options");}
    prepareNativeJitBase(/** @type {Parameters<typeof prepareNativeJitBase>[0]} */ (options));
    console.log("[native-sdk] Staged pinned SYCL JIT resources; no compiler/device/publication qualification.");
  } catch (error) {console.error(error instanceof Error ? error.message : "SYCL JIT base production failed"); process.exitCode = 1;}
}
