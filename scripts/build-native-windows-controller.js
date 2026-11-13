"use strict";

// Cross-build the maintained JavaScript controller and its original C++ host for Windows.
// This is deliberately a build-only lane: it never executes the PE or publishes an archive.
// An explicit application descriptor adds source-bound Core/all GPU graphs to the same host
// recipe. Every tool, SDK root, library and import-library path comes from
// the caller's descriptor; the audit cache is not a default and is never discovered here.
const fs = require("node:fs");
const path = require("node:path");
const tls = require("node:tls");
const crypto = require("node:crypto");
const {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const esbuild = require("esbuild");
const {absolutePath} = require("./build-native-release");
const {controllerBundleOptions, controllerAotCommand, verifyBundle, nativeApplicationRoot} = require("./native-bundle");
const {controllerEntry, nativeApplicationBuildInputs} = require("./probe-native-controller");
const nativeResources = require("./native-resources");
const {nativeJitCacheHeader} = require("./native-jit-cache-tag");
const {buildDependencies, fileHash, readBuildJson, readBuildText} = require("./build-source-inputs");
const {validateWindowsApplication, windowsApplicationInputs, verifyWindowsApplicationFiles, rejectStaticNvrtcArchives, windowsResponseFile, auditWindowsLinkMap} = require("./native-windows-application");
const {gpuTable, gpuForwards} = require("./native-gpu-table");
const {isolateCoffCompilerArchives} = require("./native-coff-archive");
const {windowsHipBindings, windowsHipAliases} = require("./native-windows-driver");
const {sealNativePePages} = require("./native-pe-pages");
const {windowsDpcppSdkInputs} = require("./native-windows-dpcpp-component");
const {compileNativeObjects} = require("./native-object-cache");

const repo = path.resolve(__dirname, "..");
const MAX_WINDOWS_LIBRARY_ROOT_ENTRIES = 8192;
const windowsLongPathManifest = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings xmlns:ws2="http://schemas.microsoft.com/SMI/2016/WindowsSettings">
      <ws2:longPathAware>true</ws2:longPathAware>
    </windowsSettings>
  </application>
</assembly>
`;
// Fixed Win32 .res records: null header, then neutral RT_MANIFEST (24), executable ID 1.
// LLD reads .res directly, avoiding an external mt.exe/resource compiler for this fixed XML.
function windowsLongPathResource() {
  const xml = Buffer.from(windowsLongPathManifest), resource = Buffer.alloc(64 + Math.ceil(xml.length / 4) * 4);
  for (const offset of [0, 32]) {
    resource.writeUInt32LE(32, offset + 4);
    resource.writeUInt16LE(0xffff, offset + 8); resource.writeUInt16LE(0xffff, offset + 12);
  }
  resource.writeUInt32LE(xml.length, 32); resource.writeUInt16LE(24, 42);
  resource.writeUInt16LE(1, 46); resource.writeUInt16LE(0x1030, 52);
  xml.copy(resource, 64);
  return resource;
}
const windowsLongPathManifestSha256 = crypto.createHash("sha256").update(windowsLongPathResource()).digest("hex");
const sourceLeaves = [
  "scripts/build-native-windows-controller.js",
  "scripts/build-source-inputs.js",
  "scripts/native-bundle.js",
  "scripts/native-resources.js",
  "scripts/probe-native-controller.js",
  "scripts/native-windows-application.js", "scripts/native-windows-driver.js", "scripts/native-pe-pages.js",
  "scripts/native-coff-archive.js", "scripts/native-compiler-isolation.js", "scripts/native-gpu-table.js",
  "scripts/native-windows-dpcpp-component.js", "scripts/native-windows-host-component.js",
];
const runtimeSources = ["main", "host", "process", "event-loop", "services", "tls", "nvidia-metadata"];
const forbiddenSymbols = /(?:intrinsics[-_]shim|environ[-_]shim|(?:^|[^A-Za-z0-9])(?:_mm_|p__environ))/i;
const windowsTarget = "x86_64-pc-windows-msvc";
// The descriptor may narrow this set for a particular import-library closure, but cannot add
// a runtime such as node.dll or an application-provided DLL to the PE trust boundary.
const ordinaryWindowsImports = new Set([
  "advapi32.dll", "bcrypt.dll", "cfgmgr32.dll", "crypt32.dll", "dbghelp.dll", "dnsapi.dll", "gdi32.dll", "iphlpapi.dll",
  "kernel32.dll", "ncrypt.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll", "powrprof.dll", "psapi.dll", "rpcrt4.dll",
  "secur32.dll", "setupapi.dll", "shell32.dll", "shlwapi.dll", "user32.dll", "userenv.dll", "uuid.dll",
  "version.dll", "winhttp.dll", "winmm.dll", "wsock32.dll", "ws2_32.dll",
]);

/** @typedef {{clangCl: string, lldLink: string, shermaot: string, llvmNm: string,
 * llvmReadobj: string, llvmObjdump?: string, clangResourceInclude: string}} WindowsTools
 * @typedef {{overlay: string, sourceRoot: string, includeRoots: string[], libraryRoots: string[],
 * msvcLibraries: string[], ucrtLibraries: string[], importLibraries: string[], allowedImports: string[]}} WindowsSdk
 * @typedef {Pick<WindowsSdk, "overlay" | "sourceRoot" | "includeRoots" | "libraryRoots" | "msvcLibraries" | "ucrtLibraries" | "importLibraries">} WindowsSdkInputs
 * @typedef {Pick<WindowsTools, "clangCl" | "lldLink" | "llvmNm" | "clangResourceInclude">} WindowsLlvmTools
 * @typedef {{version: number, tools: WindowsLlvmTools, windowsSdk: WindowsSdkInputs}} WindowsLlvmSdkConfig
 * @typedef {{sourceInclude: string, sourceHeaders?: string, libuvInclude?: string, zlibInclude?: string, zlibGeneratedInclude?: string, ssl: string, crypto: string, libuv: string, zlib: string}} StaticLibraries
 * @typedef {{source: string, build: string, shermaot: string}} Hermes
 * @typedef {{version: number, tools: WindowsTools, hermes: Hermes, windowsSdk: WindowsSdk,
 * libraries: StaticLibraries, outputName?: string, target?: string,
 * application?: import("./native-windows-application").WindowsApplication}} WindowsControllerConfig */

/** Keep diagnostics and child-process tokens finite; a path may contain spaces and Unicode but
 * never controls a command or filesystem traversal. */
/** @param {unknown} value @param {string} name */
function requirePath(value, name) {
  if (!absolutePath(value)) {throw new Error(`Invalid absolute ${name}`);}
  return /** @type {string} */ (value);
}

/** @param {unknown} value @param {string} name @returns {string[]} */
function requirePathList(value, name) {
  if (!Array.isArray(value) || !value.length || value.length > 256) {throw new Error(`Invalid ${name}`);}
  const result = value.map((item, index) => requirePath(item, `${name}[${index}]`));
  if (new Set(result).size !== result.length) {throw new Error(`Duplicate ${name}`);}
  return result;
}

/** @param {unknown} value @param {string} name @returns {string[]} */
function requireLibraryList(value, name) {
  const result = requirePathList(value, name);
  if (result.some(file => !/\.(?:lib|a)$/i.test(file))) {throw new Error(`${name} must contain archives`);}
  return result;
}

/** @param {unknown} value @param {string} name @returns {string[]} */
function requireImportList(value, name) {
  const result = requireLibraryList(value, name);
  if (result.some(file => !/\.lib$/i.test(file))) {throw new Error(`${name} must contain Windows import libraries`);}
  return result;
}

/** Validate the OS SDK roots shared by the controller and cross-built LLVM host libraries.
 * @param {unknown} value @returns {WindowsSdkInputs} */
function validateWindowsSdkInputs(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {throw new Error("Invalid Windows SDK inputs");}
  const sdk = /** @type {WindowsSdkInputs} */ (value);
  requirePath(sdk.overlay, "Windows SDK VFS overlay"); requirePath(sdk.sourceRoot, "Windows SDK source root");
  requirePathList(sdk.includeRoots, "Windows SDK include roots"); requirePathList(sdk.libraryRoots, "Windows SDK library roots");
  requireImportList(sdk.msvcLibraries, "MSVC libraries"); requireImportList(sdk.ucrtLibraries, "UCRT libraries");
  requireImportList(sdk.importLibraries, "Windows SDK import libraries");
  return sdk;
}

/** Validate only the explicit tools and OS SDK inputs consumed by the Windows LLVM 21 cross-build.
 * Extra controller/application fields may be present, but do not become LLVM build inputs.
 * @param {unknown} value @returns {WindowsLlvmSdkConfig} */
function validateWindowsLlvmSdkConfig(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {throw new Error("Invalid Windows LLVM SDK descriptor");}
  const descriptor = /** @type {{version?: unknown, tools?: unknown, windowsSdk?: unknown}} */ (value);
  if (descriptor.version !== 1 || !descriptor.tools || typeof descriptor.tools !== "object" || Array.isArray(descriptor.tools)) {
    throw new Error("Invalid Windows LLVM SDK descriptor");
  }
  const selected = /** @type {Record<string, unknown>} */ (descriptor.tools);
  const tools = /** @type {WindowsLlvmTools} */ ({
    clangCl: requirePath(selected["clangCl"], "tool clangCl"),
    lldLink: requirePath(selected["lldLink"], "tool lldLink"),
    llvmNm: requirePath(selected["llvmNm"], "tool llvmNm"),
    clangResourceInclude: requirePath(selected["clangResourceInclude"], "tool clangResourceInclude"),
  });
  return {version: 1, tools, windowsSdk: validateWindowsSdkInputs(descriptor.windowsSdk)};
}

/** Validate a complete, explicit Windows controller SDK descriptor before output creation. */
/** @param {unknown} value @returns {WindowsControllerConfig} */
function validateWindowsControllerConfig(value) {
  const config = /** @type {WindowsControllerConfig} */ (value);
  if (!config || config.version !== 1 || !config.tools || !config.hermes || !config.windowsSdk || !config.libraries) {
    throw new Error("Invalid Windows controller SDK descriptor");
  }
  const tools = config.tools;
  for (const [name, value] of [["clangCl", tools.clangCl], ["lldLink", tools.lldLink], ["shermaot", tools.shermaot],
    ["llvmNm", tools.llvmNm], ["llvmReadobj", tools.llvmReadobj], ["clangResourceInclude", tools.clangResourceInclude]]) {
    requirePath(value, `tool ${name}`);
  }
  if (tools.llvmObjdump !== undefined) {requirePath(tools.llvmObjdump, "tool llvmObjdump");}
  const hermes = config.hermes;
  for (const [name, value] of [["source", hermes.source], ["build", hermes.build], ["shermaot", hermes.shermaot]]) {
    requirePath(value, `Hermes ${name}`);
  }
  if (hermes.shermaot !== tools.shermaot) {throw new Error("Hermes shermaot must match tools.shermaot");}
  const sdk = /** @type {WindowsSdk} */ (config.windowsSdk);
  validateWindowsSdkInputs(sdk);
  if (!Array.isArray(sdk.allowedImports) || !sdk.allowedImports.length || sdk.allowedImports.length > 64 ||
      sdk.allowedImports.some(name => typeof name !== "string" || !/^[A-Za-z0-9_.-]+\.dll$/i.test(name)) ||
      new Set(sdk.allowedImports.map(name => name.toLowerCase())).size !== sdk.allowedImports.length ||
      sdk.allowedImports.some(name => !ordinaryWindowsImports.has(name.toLowerCase()))) {
    throw new Error("Allowed Windows imports must be ordinary system DLLs");
  }
  const libraries = config.libraries;
  for (const [name, value] of [["sourceInclude", libraries.sourceInclude], ["ssl", libraries.ssl], ["crypto", libraries.crypto],
    ["libuv", libraries.libuv], ["zlib", libraries.zlib]]) {requirePath(value, `static library ${name}`);}
  if (libraries.sourceHeaders !== undefined) {requirePath(libraries.sourceHeaders, "OpenSSL source headers");}
  if (libraries.libuvInclude !== undefined) {requirePath(libraries.libuvInclude, "libuv headers");}
  if (libraries.zlibInclude !== undefined) {requirePath(libraries.zlibInclude, "zlib headers");}
  if (libraries.zlibGeneratedInclude !== undefined) {requirePath(libraries.zlibGeneratedInclude, "generated zlib headers");}
  if (!/\.(?:lib|a)$/i.test(libraries.ssl) || !/\.(?:lib|a)$/i.test(libraries.crypto) ||
      !/\.lib$/i.test(libraries.libuv) || !/\.lib$/i.test(libraries.zlib)) {throw new Error("Invalid Windows static library suffix");}
  if (config.outputName !== undefined && (typeof config.outputName !== "string" ||
      !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.exe$/i.test(config.outputName))) {throw new Error("Invalid Windows controller output name");}
  if (config.target !== undefined && config.target !== windowsTarget) {
    throw new Error("Windows controller target must be x86_64-pc-windows-msvc");
  }
  // Legacy flags cannot turn the host placeholder into an unreviewed mining build. Only the
  // explicit application descriptor admits the complete source-bound Core/GPU set.
  if (Object.hasOwn(config, "nativeRuntime") || Object.hasOwn(config, "cpuCore") || Object.hasOwn(config, "gpuInputs")) {
    throw new Error("Windows controller host build does not accept Core/GPU inputs");
  }
  if (config.application !== undefined) {validateWindowsApplication(config.application, repo);}
  return config;
}

/** @param {string} file */
function regularFile(file) {
  const stat = fs.statSync(file, {throwIfNoEntry: false});
  return Boolean(stat?.isFile() && stat.size >= 0);
}

/** @param {string[]} files @param {string} label */
function validateInputFiles(files, label) {
  for (const file of [...new Set(files)]) {
    if (!regularFile(file)) {throw new Error(`${label} is missing or not a regular file: ${file}`);}
  }
}

/** @param {WindowsControllerConfig} config */
function explicitInputFiles(config) {
  const {tools, hermes, windowsSdk: sdk, libraries} = config;
  const files = [tools.clangCl, tools.lldLink, tools.shermaot, tools.llvmNm, tools.llvmReadobj,
    ...(tools.llvmObjdump ? [tools.llvmObjdump] : []), sdk.overlay, libraries.ssl, libraries.crypto,
    libraries.libuv, libraries.zlib];
  for (const file of [
    path.join(hermes.build, "jsi", "jsi.lib"), path.join(hermes.build, "lib", "hermesvmlean_a.lib"),
    path.join(hermes.build, "external", "boost", "boost_1_86_0", "libs", "context", "boost_context.lib"),
    ...sdk.msvcLibraries, ...sdk.ucrtLibraries, ...sdk.importLibraries,
  ]) {files.push(file);}
  if (config.application) {
    const {sdk, tools} = config.application;
    const selected = windowsDpcppSdkInputs(sdk);
    files.push(...selected.link.libraries, ...selected.link.wholeArchives, ...selected.link.objects, ...Object.keys(selected.files), sdk.nvmlHeader, sdk.hipDriverLibrary, ...Object.values(tools));
  }
  // LLD may resolve /DEFAULTLIB directives from these exact /libpath roots. Hash all direct
  // archive candidates without recursively inventorying the SDK; unused archives are possible
  // inputs, not claims that the linker consumed them. Keep root order in linkCommand unchanged.
  files.push(...windowsSdkLibraryCandidates(sdk.libraryRoots));
  return [...new Set(files)];
}

/** @param {string[]} libraryRoots */
function windowsSdkLibraryCandidates(libraryRoots) {
  if (!Array.isArray(libraryRoots) || libraryRoots.length > 256 ||
      libraryRoots.some(root => typeof root !== "string" || !absolutePath(root))) {
    throw new Error("Invalid Windows SDK library roots");
  }
  const candidates = [];
  let entriesSeen = 0;
  for (const root of libraryRoots) {
    const entries = fs.readdirSync(root, {withFileTypes: true});
    entriesSeen += entries.length;
    if (entriesSeen > MAX_WINDOWS_LIBRARY_ROOT_ENTRIES) {
      throw new Error("Windows SDK library roots exceed the bounded entry limit");
    }
    for (const entry of entries) {
      if (!/\.lib$/i.test(entry.name)) {continue;}
      const file = path.join(root, entry.name), stat = fs.lstatSync(file);
      if (stat.isSymbolicLink() || !stat.isFile()) {
        throw new Error("Windows SDK library candidate is not a regular file: " + file);
      }
      candidates.push(file);
    }
  }
  return candidates;
}

/** @param {string[]} files */
function hashFiles(files) {
  return Object.fromEntries([...new Set(files)].sort().map(file => [file, fileHash(file)]));
}

/** @param {Record<string, string>} before @param {Record<string, string>} after */
function assertStableInputHashes(before, after) {
  if (!isDeepStrictEqual(before, after)) {
    throw new Error("Windows controller SDK input set or contents changed during compilation");
  }
}

/** @param {string} overlay @returns {Map<string, string>} */
function vfsFiles(overlay) {
  const value = readBuildJson(overlay);
  const roots = /** @type {{type: string, name: string, "external-contents": string}[]} */ (value?.roots);
  if (!value || !Array.isArray(roots) || roots.length > 65536 ||
      roots.some((/** @type {{type: string, name: string, "external-contents": string}} */ root) => !root || root.type !== "file" || !absolutePath(root.name) ||
        !absolutePath(root["external-contents"]))) {
    throw new Error("Invalid Windows SDK VFS overlay");
  }
  const insensitive = value["case-sensitive"] === false;
  const result = new Map(roots.map(root => [insensitive ? root.name.toLowerCase() : root.name, root["external-contents"]]));
  if (result.size !== roots.length) {throw new Error("Duplicate Windows SDK VFS overlay path");}
  return result;
}

/** Hash every header discovered by the compiler dependency graph. Virtual VFS names are resolved
 * through the selected overlay; source/output paths are already covered by their own receipts. */
/** @param {string[]} depfiles @param {string} overlay @param {string} output @param {number} startedAt */
function dependencyHashes(depfiles, overlay, output, startedAt) {
  const files = new Set(), virtual = vfsFiles(overlay);
  const roots = [repo, output].map(directory => fs.realpathSync(directory) + path.sep);
  for (const depfile of depfiles) {
    for (const real of resolveWindowsDependencyPaths(readBuildText(depfile, 4 << 20), repo, virtual)) {
      if (!roots.some(root => real.startsWith(root))) {
        if (fs.statSync(real).ctimeMs > startedAt) {throw new Error("Windows SDK dependency changed during compilation: " + real);}
        files.add(real);
      }
    }
  }
  return hashFiles([...files]);
}

/** @param {string} text @param {string} root @param {Map<string, string>} virtual */
function resolveWindowsDependencyPaths(text, root, virtual) {
  const dependencies = buildDependencies(text, root, "nmake");
  if (!dependencies?.length) {throw new Error("Incomplete Windows controller compiler dependency graph");}
  return [...new Set(dependencies.map(dependency => {
    const selected = regularFile(dependency) ? dependency : virtual.get(dependency) || virtual.get(dependency.toLowerCase());
    if (!selected || !regularFile(selected)) {throw new Error("Unresolved Windows controller dependency: " + dependency);}
    return fs.realpathSync(selected);
  }))];
}

/** Output-local headers are explicit inputs; the source itself is staged by the cache helper. */
/** @param {string} name @param {string} output */
function windowsGeneratedCompileInputs(name, output) {
  const files = [];
  if (name === "services") {files.push(path.join(output, "app-resources.inc"));}
  if (name === "tls") {files.push(path.join(output, "tls-roots.inc"));}
  if (name === "gpu-dispatch" || name === "gpu-forwards") {files.push(path.join(output, "gpu-table.h"));}
  if (name === "cuda-images") {files.push(path.join(output, "cuda-images.inc"));}
  if (name === "main") {files.push(path.join(output, "mom-jit-cache-tag.h"));}
  return files.filter(file => fs.existsSync(file));
}

/** @param {Array<{args: string[]}>} commands @param {string} output */
function windowsObjectIncludeRoots(commands, output) {
  const roots = new Set(), outputRoot = path.resolve(output);
  for (const {args} of commands) {
    for (let i = 0; i < args.length; ++i) {
      const arg = args[i];
      let selected;
      if (arg === "-I" || arg === "-isystem" || arg === "-imsvc") {selected = args[++i];} else if (arg?.startsWith("-I") && arg.length > 2) {selected = arg.slice(2);}
      if (!selected || !path.isAbsolute(selected)) {continue;}
      const root = path.resolve(selected);
      if (root === outputRoot || root.startsWith(outputRoot + path.sep)) {continue;}
      try {if (fs.statSync(root).isDirectory()) {roots.add(root);}} catch (error) {
        if (!(error instanceof Error) || !("code" in error) || !["ENOENT", "ENOTDIR"].includes(String(error.code))) {throw error;}
      }
    }
  }
  return [...roots];
}

/** Compile application translation units while retaining the same VFS-aware admission used by the final receipt. */
/** @param {{output: string, namespace: string, entries: Array<{name: string, command: string, args: string[], source: string, object: string, depfile: string, log: string, generatedInputs?: string[]}>, overlay: string, environment: NodeJS.ProcessEnv, finalize?: boolean, liveObjects?: string[]}} batch */
function compileWindowsObjectBatch(batch) {
  const virtual = vfsFiles(batch.overlay);
  const commands = /** @type {Parameters<typeof compileNativeObjects>[0]["commands"]} */ (
    batch.entries.map(({name: _name, log: _log, ...entry}) => ({...entry, timeout: 300000, dependencyFormat: /** @type {"nmake"} */ ("nmake")})));
  const result = compileNativeObjects({repo, output: batch.output, namespace: batch.namespace, commands,
    environment: windowsToolEnvironment(batch.environment), jobs: 1, includeRoots: windowsObjectIncludeRoots(batch.entries, batch.output),
    keyFiles: [batch.overlay],
    resolveDependencies: ({depfileText, repo: root}) => resolveWindowsDependencyPaths(depfileText, root, virtual),
    ...(batch.finalize ? {finalize: true, liveObjects: batch.liveObjects || []} : {})});
  if (!Array.isArray(result.objects) || result.objects.length !== batch.entries.length) {throw new Error("Native object cache returned an incomplete Windows batch");}
  const objects = result.objects.map((object, index) => {
    const entry = batch.entries[index];
    if (!entry || path.resolve(object.object) !== path.resolve(entry.object) || path.resolve(object.depfile) !== path.resolve(entry.depfile)) {
      throw new Error("Native object cache returned an unexpected Windows output path");
    }
    fs.writeFileSync(entry.log, String(object.stdout || "") + String(object.stderr || ""), {flag: "wx"});
    if (object.status !== 0) {throw new Error("Windows native object compile failed: " + path.basename(entry.log), {cause: object.error});}
    return {name: entry.name, source: entry.source, log: entry.log, command: entry.command, args: entry.args, object: object.object, depfile: object.depfile,
      status: object.status, cacheHit: object.cacheHit, objectSha256: object.objectSha256, depfileSha256: object.depfileSha256};
  });
  return {cacheRoot: result.cacheRoot, ninja: result.ninja, cacheHits: result.cacheHits, cacheMisses: result.cacheMisses, objects};
}

/** @param {string} file */
function regularDirectory(file) {
  const stat = fs.statSync(file, {throwIfNoEntry: false});
  return Boolean(stat?.isDirectory());
}

/** Validate path-bearing directories after descriptor admission, before any output is made. */
/** @param {WindowsControllerConfig} config */
function validateInputDirectories(config) {
  const {tools, hermes, windowsSdk: sdk, libraries} = config;
  const virtual = vfsFiles(sdk.overlay);
  const directories = [tools.clangResourceInclude, hermes.source, hermes.build, sdk.sourceRoot,
    ...sdk.includeRoots, ...sdk.libraryRoots, libraries.sourceInclude];
  if (libraries.sourceHeaders) {directories.push(libraries.sourceHeaders);}
  if (libraries.libuvInclude) {directories.push(libraries.libuvInclude);}
  if (libraries.zlibInclude) {directories.push(libraries.zlibInclude);}
  if (libraries.zlibGeneratedInclude) {directories.push(libraries.zlibGeneratedInclude);}
  if (config.application) {directories.push(...config.application.sdk.includeRoots);}
  for (const directory of directories) {
    const prefix = directory.replace(/\/+$/, "") + "/";
    if (!regularDirectory(directory) && ![...virtual.keys()].some(file => file.startsWith(prefix))) {
      throw new Error("Windows controller SDK directory is missing: " + directory);
    }
  }
}

/** Hash the maintained source graph and the exact controller recipe. Runtime is intentionally
 * inventoried as a whole so an unrelated maintained host edit cannot be hidden by a stale PE. */
/** @param {string} root @param {string[]} [bundleInputs] */
function sourceSnapshot(root, bundleInputs = []) {
  const state = nativeApplicationBuildInputs(root);
  const files = new Set(Object.keys(state.files).map(file => path.join(root, file)));
  files.add(path.join(root, "mom.js")); files.add(path.join(root, "GPU-CONFIG.md"));
  for (const file of sourceLeaves.map(name => path.join(root, name))) {files.add(file);}
  for (const input of bundleInputs) {
    if (!absolutePath(input) || !path.isAbsolute(input) || input.includes("/node_modules/")) {continue;}
    const relative = path.relative(root, input);
    if (relative && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative)) {files.add(input);}
  }
  validateInputFiles([...files], "Maintained source input");
  return Object.fromEntries([...files].sort().map(file => [path.relative(root, file).split(path.sep).join("/"), fileHash(file)]));
}

/** @param {string} file @param {string[]} includeRoots @param {string} target @param {boolean} c */
function commonCompileArgs(file, includeRoots, target, c) {
  // This recipe owns host controller/compatibility glue; mining Core and device kernels arrive separately.
  // MSVC setjmp needs compiler lowering for its hidden frame argument. Generated C uses
  // it for JS exceptions; -fno-builtin there makes longjmp fail with STATUS_BAD_STACK.
  // Disable only Intel's special memset mapping; use the ordinary static UCRT implementation.
  return ["--target=" + target, "/nologo", "/c", "/MT", "/O2", "/DNDEBUG", "/Brepro", ...(c ? ["/Qno-builtin-memset"] : ["-fno-builtin"]), "-D_WIN32", "-D_WIN64",
    "-fms-compatibility-version=19.44", "-nostdinc", ...includeRoots.flatMap(root => ["-I", root]), c ? "/Tc" : "/Tp", file];
}

/** @param {WindowsControllerConfig} config @param {string} source @param {string} object @param {string} dep @param {boolean} c */
function compileCommand(config, source, object, dep, c) {
  const target = config.target || windowsTarget, {tools, hermes, windowsSdk: sdk, libraries} = config;
  const includes = [tools.clangResourceInclude, path.join(hermes.source, "include"), path.join(hermes.build, "lib", "config"),
    ...sdk.includeRoots, ...(c ? [] : [path.join(hermes.source, "API"), path.join(hermes.source, "API", "jsi"),
      path.join(hermes.source, "public"), libraries.sourceInclude, ...(libraries.sourceHeaders ? [libraries.sourceHeaders] : []),
      ...(libraries.libuvInclude ? [libraries.libuvInclude] : []),
      ...(libraries.zlibInclude ? [libraries.zlibInclude] : []),
      ...(libraries.zlibGeneratedInclude ? [libraries.zlibGeneratedInclude] : []),
      path.join(repo, "runtime"), ...(config.application ? [repo, path.dirname(object), ...config.application.sdk.includeRoots] : [])])];
  const args = commonCompileArgs(source, includes, target, c);
  // Explicit language operands consume POSIX absolute filenames. Otherwise icx-cl can
  // interpret /workspace/... as a /w warning option and report "no input files".
  args.splice(args.length - 2, 0, "-Xclang", "-ivfsoverlay", "-Xclang", sdk.overlay);
  if (!c) {
    args.splice(3, 0, "/EHsc", config.application ? "/std:c++20" : "/std:c++17");
    if (config.application) {
      args.push("-DMOM_NATIVE_RUNTIME");
      if (path.basename(source) === "nvidia-metadata.cpp") {args.push("-DMOM_NATIVE_NVML_METADATA", "-I", path.dirname(config.application.sdk.nvmlHeader));}
      // This entry runs before the CRT initializes its security cookie; all other objects
      // retain their normal stack protection and runtime initialization.
      if (path.basename(source) === "packed-windows.cpp") {args.push("/GS-");}
    }
  }
  args.push("-Xclang", "-dependency-file", "-Xclang", dep, "-Xclang", "-MT", "-Xclang", object, "-Xclang", "-MV", "-Fo" + object);
  return {command: tools.clangCl, args};
}

/** @param {WindowsControllerConfig} config @param {string[]} objects @param {string} output @param {string} map @param {string} [manifestInput] */
function linkCommand(config, objects, output, map, manifestInput = undefined) {
  const {tools, hermes, windowsSdk: sdk, libraries} = config;
  if (config.application) {
    manifestInput ??= path.join(path.dirname(output), "long-path-aware.res");
    requirePath(manifestInput, "Windows application manifest resource");
  }
  let applicationSdk = null;
  if (config.application) {
    const sdkDescriptor = {...config.application.sdk}; delete sdkDescriptor.cudaImages;
    applicationSdk = windowsDpcppSdkInputs(sdkDescriptor).link;
  }
  const archives = [path.join(hermes.build, "jsi", "jsi.lib"), path.join(hermes.build, "lib", "hermesvmlean_a.lib"),
    path.join(hermes.build, "external", "boost", "boost_1_86_0", "libs", "context", "boost_context.lib"),
    libraries.libuv, libraries.zlib, libraries.ssl, libraries.crypto];
  const applicationArchives = config.application && applicationSdk ? [config.application.core, ...applicationSdk.wholeArchives,
    ...applicationSdk.libraries] : [];
  // Retain admission of pinned SDK inputs, but use UCRT for host memory and standard math.
  const linkMsvcLibraries = sdk.msvcLibraries.filter(file => !["libircmt.lib", "libmmt.lib"].includes(path.basename(file).toLowerCase()));
  if (config.application?.sdk.cudaImages) {
    rejectStaticNvrtcArchives([...archives, ...sdk.msvcLibraries, ...sdk.ucrtLibraries, ...sdk.importLibraries, ...applicationArchives]);
  }
  const args = ["/machine:x64", "/subsystem:console", config.application ? "/entry:momPackedEntry" : "/entry:wmainCRTStartup", "/nologo", "/out:" + output, "/map:" + map,
    "/NXCOMPAT", "/DYNAMICBASE", "/HIGHENTROPYVA",
    // Fix link time without /Brepro's REPRO debug marker, which UPX erases on restore.
    // Compile objects still use /Brepro; the strict payload audit remains unchanged.
    "/timestamp:0",
    ...(config.application ? ["/manifest:no", /** @type {string} */ (manifestInput)] : []),
    // The exact static CRT archives follow explicitly; case-varying defaultlib hints must
    // not select another SDK or fail lookup on a case-sensitive cross-compilation host.
    "/nodefaultlib:LIBCMT.lib", "/nodefaultlib:OLDNAMES.lib", "/nodefaultlib:libircmt.lib", "/nodefaultlib:libmmt.lib",
    // SYCL emits this hint even when these images need none of its host helpers. A future
    // real dependency still fails as an undefined symbol; never suppress unresolved errors.
    ...(config.application ? ["/nodefaultlib:sycl-devicelib-host.lib"] : []),
    "/opt:ref", "/opt:icf", "/incremental:no", ...sdk.libraryRoots.flatMap(root => ["/libpath:" + root]),
    ...objects, ...archives, "/defaultlib:libvcruntime.lib", "/defaultlib:libucrt.lib", ...linkMsvcLibraries,
    ...sdk.ucrtLibraries, ...sdk.importLibraries,
    ...(config.application && applicationSdk ? [config.application.core, ...applicationSdk.wholeArchives.map(file => "/wholearchive:" + file),
      ...applicationSdk.libraries, ...applicationSdk.objects] : [])];
  return {command: tools.lldLink, args};
}

/** @param {string} text @returns {string[]} */
function parseImports(text) {
  const imports = new Set();
  // llvm-readobj prints `Name: KERNEL32.dll` under each Import section. Keep parsing narrow;
  // if its format changes, the build fails closed instead of claiming an import audit.
  for (const match of text.matchAll(/^\s*Name:\s*([^\s]+\.dll)\s*$/gim)) {
    const name = match[1] || ""; if (name) {imports.add(name.toLowerCase());}
  }
  if (!imports.size) {throw new Error("Windows PE import inspection returned no DLL names");}
  return [...imports].sort();
}

/** Validate the PE shape and fail closed if delay imports could hide an application DLL. */
/** @param {string} text */
function validatePeHeaders(text) {
  if (!/^Format:[ \t]+COFF-x86-64[ \t]*$/m.test(text) ||
      !/^[ \t]*Magic:[ \t]+0x20B[ \t]*$/m.test(text)) {
    throw new Error("Windows controller output is not a PE32+ x86-64 executable");
  }
  const fields = [...text.matchAll(/^[ \t]*(DelayImportDescriptorRVA|DelayImportDescriptorSize):[ \t]*0x([0-9a-f]+)[ \t]*$/gim)];
  const names = new Set(fields.map(match => match[1]));
  if (fields.length !== 2 || names.size !== 2) {
    throw new Error("Windows PE header omitted the delay-import directory");
  }
  if (fields.some(match => {
    const value = Number.parseInt(match[2] || "", 16);
    return !Number.isSafeInteger(value) || value !== 0;
  })) {
    throw new Error("Windows PE delay-import directory is not supported");
  }
}

/** Re-run the canonical bundle recipe read-only and compare bytes with the bundle that fed AOT.
 * The injected build function is only for bounded unit testing; production uses esbuild directly. */
/** @param {string} entry @param {string} bundle @param {Uint8Array} expected
 * @param {typeof esbuild.buildSync} [build] */
function verifyCanonicalBundle(entry, bundle, expected, build = esbuild.buildSync) {
  const rebuilt = build(controllerBundleOptions(entry, bundle, true, "win32"));
  if (!rebuilt.metafile) {throw new Error("Read-only Windows controller bundle did not produce a source graph");}
  if (!Array.isArray(rebuilt.outputFiles) || rebuilt.outputFiles.length !== 1) {
    throw new Error("Read-only Windows controller bundle did not produce exactly one output");
  }
  const output = rebuilt.outputFiles[0];
  if (!output || typeof output.path !== "string" || path.resolve(output.path) !== path.resolve(bundle) ||
      !(output.contents instanceof Uint8Array)) {
    throw new Error("Read-only Windows controller bundle produced an unexpected output");
  }
  return verifyBundle(output.contents, expected);
}

/** @param {string} command @param {string[]} args @param {string} cwd @param {string} log
 * @param {typeof spawnSync} [spawn] */
function runTool(command, args, cwd, log, spawn = spawnSync) {
  const fd = fs.openSync(log, "wx");
  try {
    const result = spawn(command, args, {cwd, stdio: ["ignore", fd, fd], timeout: 300000,
      env: windowsToolEnvironment()});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error(`Windows native build failed: ${path.basename(log)}`, {cause: result.error});
    }
    return result;
  } finally {fs.closeSync(fd);}
}

/** Explicit SDK descriptors own every include/library/ABI choice. Do not let inherited
 * Windows compiler flags or SDK search roots silently turn a portable build into another ABI.
 * @param {NodeJS.ProcessEnv} [environment] */
function windowsToolEnvironment(environment = process.env) {
  const selected = {...environment};
  for (const key of Object.keys(selected)) {
    if (/^(?:MOM_NATIVE_RUNTIME|CL|_CL_|LINK|_LINK_|INCLUDE|LIB|LIBPATH|CPATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH|LIBRARY_PATH|CFLAGS|CXXFLAGS)$/i.test(key)) {delete selected[key];}
  }
  return selected;
}

/** @param {string} command @param {string[]} args @param {string} cwd */
function inspectTool(command, args, cwd) {
  const result = spawnSync(command, args, {cwd, encoding: "utf8", timeout: 120000, maxBuffer: 8 << 20,
    env: windowsToolEnvironment()});
  if (result.error || result.signal || result.status !== 0) {throw new Error(`Windows PE inspection failed: ${path.basename(command)}`, {cause: result.error});}
  return result.stdout;
}

/** @param {string} object @param {string} text */
function rejectForbiddenObjectSymbols(object, text) {
  if (forbiddenSymbols.test(text)) {throw new Error(`Forbidden intrinsic/environment shim symbol in ${object}`);}
}

/** @param {string} output @param {Record<string, unknown>} value */
function writeJson(output, value) {fs.writeFileSync(output, JSON.stringify(value, null, 2) + "\n", {flag: "wx"});}

/** @param {WindowsControllerConfig} config @param {string} output @param {string} [root] */
function windowsControllerPlan(config, output, root = repo) {
  const selected = validateWindowsControllerConfig(config);
  requirePath(output, "Windows controller output"); requirePath(root, "repository root");
  const generated = {
    entry: path.join(output, "entry.js"), bundle: path.join(output, "controller.js"), controllerC: path.join(output, "controller.c"),
    resources: path.join(output, "app-resources.inc"), tlsRoots: path.join(output, "tls-roots.inc"), cudaImages: path.join(output, "cuda-images.inc"),
    ...(config.application ? {manifest: path.join(output, "long-path-aware.res")} : {}),
    executable: path.join(output, selected.outputName || (config.application ? "mom.exe" : "mom-controller.exe")), map: path.join(output, config.application ? "mom.map" : "mom-controller.map"),
  };
  const sources = [...runtimeSources, ...(config.application ? ["gpu-dispatch", "gpu-forwards", "compiler-gate", "packed-windows",
    ...(config.application.sdk.cudaImages ? ["cuda-images"] : [])] : [])];
  const objects = /** @type {Record<string, string>} */ (Object.fromEntries(["controller", ...sources].map(name => [name, path.join(output, name + ".obj")])));
  const deps = /** @type {Record<string, string>} */ (Object.fromEntries(["controller", ...sources].map(name => [name, path.join(output, name + ".d")])));
  const controllerObject = objects["controller"], controllerDep = deps["controller"];
  if (!controllerObject || !controllerDep) {throw new Error("Missing planned Windows controller C output");}
  const c = compileCommand(selected, generated.controllerC, controllerObject, controllerDep, true);
  const cpp = sources.map(name => {
    const object = objects[name], dep = deps[name];
    if (!object || !dep) {throw new Error("Missing planned Windows controller C++ output: " + name);}
    const source = name === "gpu-forwards" ? path.join(output, "gpu-forwards.cpp") :
      name === "cuda-images" ? path.join(root, "sycl/cuda-images.cpp") : path.join(root, "runtime", name + ".cpp");
    return compileCommand(selected, source, object, dep, false);
  });
  const link = linkCommand(selected, Object.values(objects), generated.executable, generated.map, generated.manifest);
  const aot = controllerAotCommand(selected.tools.shermaot, generated.bundle, generated.controllerC);
  // The bundled path module follows the runtime OS, not the cross-build host. Keep the
  // immutable resource's exact identity equal to the controller's Windows path.join result.
  const policyPath = path.win32.join(nativeApplicationRoot("win32"), "GPU-CONFIG.md");
  return {config: selected, root, output, generated, objects, deps, sources, aot, compile: [c, ...cpp], link, policyPath};
}

/** @param {string} configFile @param {string} output @param {NodeJS.ProcessEnv} [environment] */
function buildNativeWindowsController(configFile, output, environment = process.env) {
  if (!absolutePath(configFile) || !absolutePath(output) || fs.existsSync(output)) {
    throw new Error("Select an absolute SDK descriptor and a new absolute Windows controller output directory");
  }
  if (environment["MOM_NATIVE_RUNTIME"] || environment["MOM_AOT_CPU_CORE"] || environment["MOM_AOT_GPU_INPUTS"]) {
    throw new Error("Windows controller host build refuses Core/GPU runtime environment inputs");
  }
  const configHash = fileHash(configFile), config = validateWindowsControllerConfig(readBuildJson(configFile));
  if (fileHash(configFile) !== configHash) {throw new Error("Windows SDK descriptor changed during admission");}
  // Component manifests own link order: freeze each selected root before planning, not after.
  const componentState = /** @type {Record<string, string>} */ ({});
  for (const component of [config.application?.sdk.dpcppComponent, config.application?.sdk.hostComponent]) {
    if (component) {const manifest = path.join(component, "component.json"); componentState[manifest] = fileHash(manifest);}
  }
  const plan = windowsControllerPlan(config, output, repo);
  const relative = path.relative(repo, output);
  if (!relative || (!relative.startsWith(".." + path.sep) && !path.isAbsolute(relative))) {
    throw new Error("Windows controller output cannot enter the maintained source tree");
  }
  validateInputDirectories(config);
  const explicitInputs = explicitInputFiles(config);
  validateInputFiles(explicitInputs, "Windows controller SDK input");
  const inputHashes = hashFiles(explicitInputs);
  verifyWindowsApplicationFiles(componentState);
  const application = config.application ? windowsApplicationInputs(config.application, repo) : null;
  for (const file of [path.join(repo, "mom.js"), ...runtimeSources.map(name => path.join(repo, "runtime", name + ".cpp"))]) {
    if (!regularFile(file)) {throw new Error("Missing maintained Windows controller source: " + file);}
  }
  if (config.application?.sdk.cudaImages && !regularFile(path.join(repo, "sycl/cuda-images.cpp"))) {
    throw new Error("Missing maintained Windows CUDA image registry source");
  }
  const startedAt = Date.now();
  fs.mkdirSync(output, {recursive: true});
  if (plan.generated.manifest) {
    fs.writeFileSync(plan.generated.manifest, windowsLongPathResource(), {flag: "wx"});
    if (fileHash(plan.generated.manifest) !== windowsLongPathManifestSha256) {throw new Error("Generated Windows long-path manifest changed while writing");}
  }
  if (application) {fs.writeFileSync(path.join(output, "mom-jit-cache-tag.h"), nativeJitCacheHeader(application.jitCacheTag), {flag: "wx"});}
  /** @type {{file: string, sha256: string} | undefined} */ let copiedCudaImages;
  if (config.application?.sdk.cudaImages) {
    const source = config.application.sdk.cudaImages, expected = application?.files[source];
    if (!expected) {throw new Error("Windows application did not admit the selected CUDA image include");}
    fs.copyFileSync(source, plan.generated.cudaImages, fs.constants.COPYFILE_EXCL);
    if (fileHash(plan.generated.cudaImages) !== expected) {throw new Error("Windows CUDA image include changed while copying into the application build");}
    copiedCudaImages = {file: plan.generated.cudaImages, sha256: expected};
  }
  const verifyCopiedCudaImages = () => {
    if (copiedCudaImages && fileHash(copiedCudaImages.file) !== copiedCudaImages.sha256) {
      throw new Error("Windows application CUDA image copy changed during compilation");
    }
  };
  const before = sourceSnapshot(repo);
  writeJson(path.join(output, "source-hashes.before.json"), before);
  writeJson(path.join(output, "config.json"), {version: 1, configHash});
  fs.writeFileSync(plan.generated.entry, controllerEntry(repo, true), {flag: "wx"});
  const bundled = esbuild.buildSync(controllerBundleOptions(plan.generated.entry, plan.generated.bundle, false, "win32"));
  const bundleBytes = fs.readFileSync(plan.generated.bundle);
  if (!bundled.metafile) {throw new Error("Native controller bundle did not produce a source graph");}
  writeJson(path.join(output, "source-graph.json"), /** @type {Record<string, unknown>} */ (/** @type {unknown} */ (bundled.metafile)));
  const bundleInputs = Object.keys(bundled.metafile.inputs || {}).map(file => path.resolve(repo, file));
  const beforeWithBundle = sourceSnapshot(repo, bundleInputs);
  // The bundle graph is discovered only after esbuild runs, so it legitimately adds JS leaves
  // that were not in the maintained native snapshot. Every pre-existing leaf must still match.
  if (Object.keys(before).some(file => before[file] !== beforeWithBundle[file])) {
    writeJson(path.join(output, "source-freshness-failure.json"), {
      before, after: beforeWithBundle, reason: "Maintained source changed while creating the Windows controller bundle",
    });
    throw new Error("Maintained source changed while creating the Windows controller bundle; freshness receipt recorded");
  }
  writeJson(path.join(output, "source-hashes.bundle.json"), beforeWithBundle);
  fs.writeFileSync(plan.generated.resources, nativeResources([{name: "GPU-CONFIG.md", data: fs.readFileSync(path.join(repo, "GPU-CONFIG.md"))}]) +
    `static constexpr std::string_view kEmbeddedPolicyPath = ${JSON.stringify(plan.policyPath)};\n`, {flag: "wx"});
  fs.writeFileSync(plan.generated.tlsRoots, `static constexpr char kTlsRoots[] = ${JSON.stringify(tls.rootCertificates.join("\n"))};\n`, {flag: "wx"});
  if (application) {
    const api = fs.readFileSync(path.join(repo, "sycl/lib.h"), "utf8");
    fs.writeFileSync(path.join(output, "gpu-table.h"), gpuTable(api, application.graphs[0]?.factory || "").header, {flag: "wx"});
    fs.writeFileSync(path.join(output, "gpu-forwards.cpp"), gpuForwards(api), {flag: "wx"});
  }
  const aot = plan.aot;
  runTool(aot.command, aot.args, aot.cwd, path.join(output, "aot.log"));
  const compileCommands = [];
  const pendingObjects = [];
  for (const [index, command] of plan.compile.entries()) {
    const name = index === 0 ? "controller" : plan.sources[index - 1];
    if (!name) {throw new Error("Missing planned Windows controller source name");}
    const args = command.args.slice();
    if (name === "services") {args.splice(args.length - 1, 0, "-DMOM_EMBEDDED_APP_RESOURCES", "-I", output);}
    if (name === "tls") {args.splice(args.length - 1, 0, "-I", output);}
    if (application && name === "gpu-dispatch") {args.push(...application.definitions);}
    if (application && name === "main") {args.push("-DMOM_NATIVE_ACPP_CACHE", "-I", output);}
    compileCommands.push({command: command.command, args});
    const object = plan.objects[name];
    if (!object) {throw new Error("Missing planned Windows controller object: " + name);}
    const depfile = plan.deps[name];
    if (!depfile) {throw new Error("Missing planned Windows controller dependency file: " + name);}
    if (application) {
      const source = index === 0 ? plan.generated.controllerC : name === "gpu-forwards" ? path.join(output, "gpu-forwards.cpp") :
        name === "cuda-images" ? path.join(repo, "sycl/cuda-images.cpp") : path.join(repo, "runtime", name + ".cpp");
      pendingObjects.push({name, command: command.command, args, source, object, depfile,
        log: path.join(output, name + ".compile.log"), generatedInputs: windowsGeneratedCompileInputs(name, output)});
    } else {
      runTool(command.command, args, repo, path.join(output, name + ".compile.log"));
    }
  }
  let objectCache = null;
  if (application && config.application) {
    const profile = config.application.defaultCompiler.replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 48) || "default";
    const namespace = `windows/controller/app-${profile}`;
    const firstBatch = compileWindowsObjectBatch({output, namespace, entries: pendingObjects, overlay: config.windowsSdk.overlay, environment});
    objectCache = {namespace, cacheRoot: firstBatch.cacheRoot, ninja: firstBatch.ninja, cacheHits: firstBatch.cacheHits,
      cacheMisses: firstBatch.cacheMisses, batches: [firstBatch]};
  }
  verifyCopiedCudaImages();
  for (const name of plan.sources.length ? ["controller", ...plan.sources] : ["controller"]) {
    const object = plan.objects[name];
    if (!object) {throw new Error("Missing planned Windows controller object: " + name);}
    const undefinedText = inspectTool(config.tools.llvmNm, ["--undefined-only", object], repo);
    rejectForbiddenObjectSymbols(name, undefinedText);
    fs.writeFileSync(path.join(output, name + ".undefined.txt"), undefinedText, {flag: "wx"});
  }
  const link = {...plan.link, args: plan.link.args.slice()};
  /** Keep large all-profile links below Windows' command-line limit without a shell.
   * @param {string[]} args @param {string} label */
  const runLink = (args, label) => {
    const response = path.join(output, label + ".rsp");
    fs.writeFileSync(response, windowsResponseFile(args), {flag: "wx"});
    runTool(link.command, ["@" + response], repo, path.join(output, label + ".log"));
  };
  let driverBinding = null;
  const generatedInputs = /** @type {Record<string, string>} */ ({});
  if (plan.generated.manifest) {generatedInputs[plan.generated.manifest] = windowsLongPathManifestSha256;}
  if (application) {generatedInputs[path.join(output, "mom-jit-cache-tag.h")] = fileHash(path.join(output, "mom-jit-cache-tag.h"));}
  if (application && config.application) {
    const app = config.application, hermesArchive = path.join(config.hermes.build, "lib/hermesvmlean_a.lib");
    const hermesPrivate = isolateCoffCompilerArchives({archives: [hermesArchive], tools: {...app.tools, llvmNm: config.tools.llvmNm},
      output: path.join(output, "hermes-private"), prefix: "mom_hermes_", entryPoints: ["_sh_init_with_error"], privateSymbols: ["HandleAbort"]});
    const selected = hermesPrivate.transformed[0]?.output; if (!selected) {throw new Error("Missing isolated Hermes runtime");}
    generatedInputs[selected] = fileHash(selected);
    link.args = link.args.map(file => file === hermesArchive ? selected : file);
    link.args.push(...application.objects);
    const probe = path.join(output, "driver-discovery.exe"), probeMap = path.join(output, "driver-discovery.map");
    runLink([...link.args.map(arg => arg.startsWith("/out:") ? "/out:" + probe : arg.startsWith("/map:") ? "/map:" + probeMap : arg), app.sdk.hipDriverLibrary], "driver-discovery");
    const discovered = inspectTool(config.tools.llvmReadobj, ["--file-headers", "--coff-imports", probe], repo);
    validatePeHeaders(discovered);
    if (parseImports(discovered).some(name => !ordinaryWindowsImports.has(name) && name !== "amdhip64_7.dll")) {throw new Error("Unexpected application DLL in Windows driver-discovery link");}
    fs.writeFileSync(path.join(output, "driver-discovery.txt"), discovered, {flag: "wx"});
    const binding = windowsHipBindings(discovered), source = path.join(output, "hip-driver-bindings.cpp"), object = path.join(output, "hip-driver-bindings.obj"), dep = path.join(output, "hip-driver-bindings.d");
    fs.writeFileSync(source, binding.source, {flag: "wx"});
    const compiled = compileCommand(config, source, object, dep, false); compiled.args.push("-D__HIP_PLATFORM_AMD__");
    if (objectCache) {
      const batch = compileWindowsObjectBatch({output, namespace: objectCache.namespace, overlay: config.windowsSdk.overlay, environment,
        entries: [{name: "hip-driver-bindings", command: compiled.command, args: compiled.args, source, object, depfile: dep,
          log: path.join(output, "hip-driver-bindings.compile.log")}], finalize: true, liveObjects: [...Object.values(plan.objects), object]});
      objectCache.cacheHits += batch.cacheHits; objectCache.cacheMisses += batch.cacheMisses; objectCache.ninja ||= batch.ninja;
      objectCache.batches.push(batch);
    } else {
      runTool(compiled.command, compiled.args, repo, path.join(output, "hip-driver-bindings.compile.log"));
    }
    compileCommands.push(compiled);
    const definitions = inspectTool(config.tools.llvmNm, ["-P", "-g", "--defined-only", object], repo);
    fs.writeFileSync(path.join(output, "hip-driver-bindings.symbols.txt"), definitions, {flag: "wx"});
    link.args.push(object, ...windowsHipAliases(definitions, binding.names));
    plan.objects["hip-driver-bindings"] = object; plan.deps["hip-driver-bindings"] = dep;
    driverBinding = {names: binding.names, sourceSha256: fileHash(source), objectSha256: fileHash(object)};
  }
  if (objectCache) {writeJson(path.join(output, "object-cache.json"), objectCache);}
  runLink(link.args, "link");
  verifyCopiedCudaImages();
  // Seal only this newly linked, unpublished executable. It remains runnable unpacked;
  // in-memory packing is a separate qualification step, never an Engineering requirement.
  const pageProtectionMap = application ? sealNativePePages(plan.generated.executable) : null;
  verifyWindowsApplicationFiles(generatedInputs);
  const dependencyFiles = Object.values(plan.deps);
  const dependencies = dependencyHashes(dependencyFiles, config.windowsSdk.overlay, output, startedAt);
  const objectFiles = Object.values(plan.objects);
  // Linking five kernel graphs does not prove that their transports are reachable: a
  // Level-Zero-only UR loader can discard OpenCL/CUDA adapters without a link error.
  const mapAudit = auditWindowsLinkMap(plan.generated.map, forbiddenSymbols, application ? [
    "mom_ur_level_zero_v2_factory", "mom_ur_cuda_factory", "clGetPlatformIDs",
    "?urAdapterGetDdiTables@opencl@ur@@YA?AW4ur_result_t@@PEAUur_dditable_t@@@Z",
  ] : []);
  const peText = inspectTool(config.tools.llvmReadobj, ["--file-headers", "--sections", "--coff-imports", plan.generated.executable], repo);
  validatePeHeaders(peText);
  const imports = parseImports(peText);
  const allowed = new Set(config.windowsSdk.allowedImports.map(name => name.toLowerCase()));
  if (imports.some(name => !allowed.has(name))) {throw new Error("Windows PE imports an unapproved DLL: " + imports.join(", "));}
  fs.writeFileSync(path.join(output, "pe-readobj.txt"), peText, {flag: "wx"});
  if (config.tools.llvmObjdump) {
    const dump = inspectTool(config.tools.llvmObjdump, ["-p", plan.generated.executable], repo);
    fs.writeFileSync(path.join(output, "pe-objdump.txt"), dump, {flag: "wx"});
  }
  verifyCanonicalBundle(plan.generated.entry, plan.generated.bundle, bundleBytes);
  const after = sourceSnapshot(repo, bundleInputs);
  writeJson(path.join(output, "source-hashes.after.json"), after);
  if (!isDeepStrictEqual(beforeWithBundle, after)) {
    writeJson(path.join(output, "source-freshness-failure.json"), {before: beforeWithBundle, after, reason: "Maintained source changed during Windows controller build"});
    throw new Error("Maintained source changed during Windows controller build; freshness receipt recorded");
  }
  if (application && config.application) {
    verifyWindowsApplicationFiles(application.files);
    if (!isDeepStrictEqual(application, windowsApplicationInputs(config.application, repo))) {throw new Error("Windows application components changed during link");}
  }
  if (fileHash(configFile) !== configHash) {throw new Error("Windows SDK descriptor changed during compilation");}
  const finalExplicitHashes = hashFiles(explicitInputFiles(config));
  assertStableInputHashes(inputHashes, finalExplicitHashes);
  if (plan.generated.manifest && fileHash(plan.generated.manifest) !== windowsLongPathManifestSha256) {throw new Error("Generated Windows manifest resource changed during link");}
  const finalInputs = {...finalExplicitHashes, ...hashFiles(objectFiles),
    ...(plan.generated.manifest ? {[plan.generated.manifest]: generatedInputs[plan.generated.manifest]} : {})};
  const dependenciesAfter = dependencyHashes(dependencyFiles, config.windowsSdk.overlay, output, startedAt);
  if (!isDeepStrictEqual(dependencies, dependenciesAfter)) {
    throw new Error("Windows SDK dependency changed after compilation");
  }
  const artifact = {path: plan.generated.executable, bytes: fs.statSync(plan.generated.executable).size, sha256: fileHash(plan.generated.executable)};
  const receipt = {
    version: 1, configHash, artifact, map: {path: plan.generated.map, ...mapAudit},
    sourceHashes: {before: beforeWithBundle, after}, inputHashes: finalInputs, dependencyHashes: dependenciesAfter,
    bundle: {bytes: bundleBytes.length, sha256: crypto.createHash("sha256").update(bundleBytes).digest("hex")},
    commands: [aot, ...compileCommands, link],
    target: config.target || windowsTarget, runtimeLibrary: "MT", nativeRuntime: Boolean(application), application, driverBinding, pageProtectionMap, objectCache,
    shimFree: {intrinsics: true, environment: true, sseHelpers: true},
    buildOnly: true, publicationQualified: false, executed: false,
    windowsSdk: {sourceRoot: config.windowsSdk.sourceRoot, overlay: config.windowsSdk.overlay,
      includeRoots: config.windowsSdk.includeRoots, libraryRoots: config.windowsSdk.libraryRoots},
    scope: application ? "Windows maintained JS/Core/all-GPU source-bound application link; build-only, no execution, SDK source-production, hardware/performance or publication qualification." :
      "Windows maintained AOT controller-host compile/link only; no MOM_NATIVE_RUNTIME Core, CPU/GPU mining, PE execution, VM validation or publication claim.",
    imports,
  };
  writeJson(path.join(output, "result.json"), receipt);
  console.log(`[native-windows] Built ${artifact.path}; build-only, not executed or publication-qualified.`);
  return receipt;
}

if (require.main === module) {
  try {
    const [configFile, output, extra] = process.argv.slice(2);
    if (!configFile || !output || extra) {throw new Error("Usage: node scripts/build-native-windows-controller.js /absolute/sdk.json /new/absolute/output");}
    buildNativeWindowsController(configFile, output);
  } catch (error) {console.error(error instanceof Error ? error.message : "Windows native controller build failed"); process.exitCode = 1;}
}

module.exports = {
  absolutePath, validateWindowsControllerConfig, validateWindowsSdkInputs, validateWindowsLlvmSdkConfig,
  explicitInputFiles, hashFiles, assertStableInputHashes, sourceSnapshot, dependencyHashes,
  compileCommand, linkCommand, parseImports, validatePeHeaders, verifyCanonicalBundle, runTool, windowsToolEnvironment, windowsControllerPlan, vfsFiles,
  windowsLongPathManifest, windowsLongPathResource, buildNativeWindowsController,
};
