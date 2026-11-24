"use strict";

// Original COFF adapters share C++/C names; isolate their implementations and retain one
// original opaque dispatch factory each. Driver thunks derive signatures from official headers.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {fileHash, sourceInputs, verifySourceInputs, readBuildJsonSnapshot} = require("./build-source-inputs");
const {verifyPreparedSdkSources, sdkSourceFiles, sdkSourceEnvironment} = require("./prepare-native-sdk-sources");
const {compileCommand, dependencyHashes, validateWindowsControllerConfig, windowsToolEnvironment, parseImports} = require("./build-native-windows-controller");
const {windowsResponseFile} = require("./native-windows-application");
const {isolateCoffCompilerArchives} = require("./native-coff-archive");
const {coffExternalFallbacks, rewriteCoffDirectives} = require("./native-compiler-isolation");
const {driverImports} = require("./native-cuda-adapter");
const {windowsDriverBindings, windowsTypedDriverAliases} = require("./native-windows-driver");
const {windowsSourceJitInputs} = require("./build-native-windows-sdk");
const repo = path.resolve(__dirname, "..");

/** Parse only actual llvm-nm definition rows; archive headings do not name definitions.
 * @param {string} text */
function adapterDefinitions(text) {
  if (Buffer.byteLength(text) > (16 << 20)) {throw new Error("Adapter definition metadata exceeds bounds");}
  const names = new Set();
  for (const row of text.split(/\r?\n/)) {
    if (!row.trim() || /:\s*$/.test(row)) {continue;}
    const name = row.match(/^(\S+) [A-Za-z] (?:[0-9a-f]|$)/)?.[1];
    if (!name || name.length > 4096 || names.size >= 1000000) {throw new Error("Malformed adapter definition metadata");}
    names.add(name);
  }
  return names;
}

/** Windows SDK callers may use either a direct function or its dllimport pointer. Preserve
 * both spellings while deriving one underlying API from the actual undefined-symbol rows.
 * @param {string} text */
function windowsDriverRequirements(text) {
  const pointers = [...new Set([...text.matchAll(/^__imp_((?:cu|nvml)[A-Z]\w*) U(?:\s|$)/gm)].map(match => match[1] || ""))].sort();
  const names = driverImports(text.replace(/^__imp_((?:cu|nvml)[A-Z]\w*) U(?=\s|$)/gm, "$1 U"));
  return {names, pointers};
}

/** Select the already-verified original runtime, replacing only the two isolated adapters.
 * Constructor-only SYCL/Level Zero members need whole-archive linking; both explicit CUDA
 * objects must enter the link, since an archive-only alias member can be silently skipped.
 * Append an explicitly verified complete JIT closure when selected; the raw-runtime-only
 * mode remains distinguishable and cannot imply compiler or runtime qualification.
 * @param {{path: string}[]} artifacts
 * @param {{kind: string, factory: string, archive?: string | undefined}[]} adapters
 * @param {{path: string}[]} driverObjects @param {string[]} [jitLibraries] */
function windowsRuntimeLinks(artifacts, adapters, driverObjects, jitLibraries = []) {
  const absolute = (/** @type {unknown} */ file) => typeof file === "string" && path.isAbsolute(file) && !/[\r\n\0]/.test(file);
  if (!Array.isArray(artifacts) || artifacts.length !== 15 || artifacts.some(row => !row || !absolute(row.path) || !/\.lib$/i.test(row.path)) ||
      !Array.isArray(adapters) || adapters.length !== 2 || !Array.isArray(driverObjects) || driverObjects.length !== 2 ||
      driverObjects.some(row => !row || !absolute(row.path) || !/\.obj$/i.test(row.path)) || !Array.isArray(jitLibraries) || jitLibraries.length > 512 ||
      jitLibraries.some(file => !absolute(file) || !/\.lib$/i.test(file))) {throw new Error("Incomplete Windows runtime link selection");}
  const libraries = new Map(artifacts.map(row => [path.basename(row.path), row.path]));
  if (libraries.size !== artifacts.length || new Set(driverObjects.map(row => row.path)).size !== 2) {throw new Error("Duplicate Windows runtime link input");}
  for (const kind of ["cuda", "level-zero-v2"]) {
    const selected = adapters.filter(row => row?.kind === kind), archive = selected[0]?.archive;
    const original = kind === "cuda" ? "ur_adapter_cuda.lib" : "ur_adapter_level_zero_v2.lib";
    if (selected.length !== 1 || selected[0]?.factory !== "mom_ur_" + kind.replaceAll("-", "_") + "_factory" || !absolute(archive) ||
        !/\.lib$/i.test(archive || "") || !libraries.has(original) || [...libraries.values()].includes(archive || "")) {throw new Error("Incomplete isolated Windows adapter link input: " + kind);}
    libraries.set(original, archive || "");
  }
  const wholeArchives = ["mom-sycl-static.lib", "ze_loader.lib"].map(name => {
    const file = libraries.get(name); if (!file) {throw new Error("Missing constructor-bearing Windows runtime: " + name);} libraries.delete(name); return file;
  });
  const all = [...libraries.values(), ...wholeArchives, ...jitLibraries];
  if (new Set(all).size !== all.length) {throw new Error("Duplicate Windows runtime/JIT archive");}
  return {libraries: [...libraries.values(), ...jitLibraries], wholeArchives, objects: driverObjects.map(row => row.path)};
}

/** Verify this exact original SDK cohort, selected OS descriptor and official CUDA headers.
 * Build receipts are local provenance, not authentication of arbitrary third-party SDKs.
 * @param {string} sdkFile @param {string} descriptorFile @param {string} cuda */
function adapterSdkInputs(sdkFile, descriptorFile, cuda) {
  if (![sdkFile, descriptorFile, cuda].every(file => path.isAbsolute(file) && !/[\r\n\0]/.test(file))) {throw new Error("Windows adapters need absolute SDK/descriptor/CUDA inputs");}
  const {value: sdk, sha256: sdkSha256} = readBuildJsonSnapshot(sdkFile),
    descriptorSha256 = sdk?.fingerprints?.[descriptorFile], message = "Incomplete original Windows SDK build receipt";
  if (sdk?.version !== 1 || sdk.buildOnly !== true || sdk.executed !== false || sdk.publicationQualified !== false ||
      !Array.isArray(sdk.artifacts) || sdk.artifacts.length !== 15 ||
      typeof descriptorSha256 !== "string" || !/^[0-9a-f]{64}$/.test(descriptorSha256)) {throw new Error(message);}
  const {value: descriptor} = readBuildJsonSnapshot(descriptorFile, undefined, {sha256: descriptorSha256, message});
  const config = validateWindowsControllerConfig(descriptor);
  const source = verifyPreparedSdkSources(sdk.sourcesFile, "windows");
  verifySourceInputs(sdk.recipe, sourceInputs(repo, [], Object.keys(sdk.recipe.graph)), "Windows SDK producer");
  const paths = new Map();
  for (const artifact of sdk.artifacts) {
    if (!artifact || !path.isAbsolute(artifact.path) || !/\.lib$/i.test(artifact.path) || paths.has(path.basename(artifact.path)) || fileHash(artifact.path) !== artifact.sha256) {throw new Error("Windows SDK archive changed or duplicated");}
    paths.set(path.basename(artifact.path), artifact.path);
  }
  for (const name of ["ur_adapter_cuda.lib", "ur_adapter_level_zero.lib", "ur_adapter_level_zero_v2.lib", "ur_loader.lib"]) {if (!paths.has(name)) {throw new Error("Missing original Windows SDK adapter archive: " + name);}}
  for (const [file, hash] of Object.entries(sdk.fingerprints)) {if (fileHash(file) !== hash) {throw new Error("Windows SDK input changed: " + file);}}
  if (!sdk.headers?.some((/** @type {{root: string}} */ header) => fs.realpathSync(header.root) === fs.realpathSync(path.join(cuda, "include")))) {throw new Error("CUDA headers differ from the admitted SDK");}
  for (const header of sdk.headers) {if (!isDeepStrictEqual(header.files, sdkSourceFiles(header.root))) {throw new Error("Windows SDK header tree changed");}}
  if (sdk.sourceJit?.enabled === true) {
    const jit = windowsSourceJitInputs(sdk.sourceJit.receiptFile);
    if (!jit || jit.archive !== sdk.sourceJit.archive || !isDeepStrictEqual(jit.libraries, sdk.sourceJit.libraries)) {throw new Error("Windows SDK source-JIT selection changed");}
  }
  return {sdk, config, source, paths, sdkSha256, descriptorSha256};
}

/** Build-only adaptation of the original static SDK, never driver discovery or execution.
 * The caller enforces a host memory cgroup; no compiler AST or GPU workload is required.
 * Independent CUDA/V2 failures are collected, without accepting a partial final receipt.
 * @param {string} sdkFile @param {string} descriptorFile @param {string} cuda @param {string} output
 * @param {NodeJS.ProcessEnv} [environment] */
function prepareWindowsAdapters(sdkFile, descriptorFile, cuda, output, environment = process.env) {
  if (process.platform !== "linux" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("Windows adapter preparation requires Linux and MOM_AOT_BUILD_ONLY=1");}
  const before = adapterSdkInputs(sdkFile, descriptorFile, cuda), config = {...before.config}; delete config.application;
  if (!path.isAbsolute(output) || /[\r\n\0]/.test(output) || fs.existsSync(output)) {throw new Error("Select a new absolute Windows adapter output");}
  const target = path.join(fs.realpathSync(path.dirname(output)), path.basename(output));
  for (const root of [path.dirname(sdkFile), cuda, ...Object.values(before.source.prepared).map((/** @type {{root: string}} */ value) => value.root)]) {
    const relative = path.relative(fs.realpathSync(root), target);
    if (!relative || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep))) {throw new Error("Windows adapter output overlaps source or SDK inputs");}
  }
  const env = sdkSourceEnvironment(windowsToolEnvironment(environment)), startedAt = Date.now(), bin = path.dirname(config.tools.llvmReadobj);
  const tools = {llvmNm: config.tools.llvmNm, llvmObjcopy: path.join(bin, "llvm-objcopy"), llvmAr: path.join(bin, "llvm-ar")};
  const urHeaders = path.join(before.source.prepared.dpcpp.root, "unified-runtime/include");
  const leaves = ["scripts/prepare-native-windows-adapters.js", "scripts/prepare-native-sdk-sources.js", "scripts/build-native-windows-sdk.js", "scripts/build-native-windows-controller.js",
    "scripts/native-coff-archive.js", "scripts/native-compiler-isolation.js", "scripts/native-cuda-adapter.js", "scripts/native-windows-driver.js", "scripts/native-windows-application.js", "scripts/build-source-inputs.js", "scripts/build-environment.js",
    "tests/native/dpcpp-static/cuda-factory.cpp", "tests/native/dpcpp-static/level-zero-v2-factory.cpp", "tests/reference/native-dpcpp-static.cpp",
    "tests/native/dpcpp-static/static-windows.h", "runtime/system-driver.h", "runtime/windows-cuda-driver.h"];
  const recipe = sourceInputs(repo, [], leaves), inputs = {[sdkFile]: before.sdkSha256, [descriptorFile]: before.descriptorSha256,
    ...Object.fromEntries([config.tools.clangCl, config.tools.llvmReadobj, path.join(bin, "llvm-lib"), ...Object.values(tools)].map(file => [file, fileHash(file)]))};
  fs.mkdirSync(output);
  /** @param {string} command @param {string[]} args @param {string} label */
  function run(command, args, label) {
    fs.writeFileSync(path.join(output, label + ".command.json"), JSON.stringify({command, args}) + "\n", {flag: "wx"});
    const result = spawnSync(command, args, {cwd: repo, env, encoding: "utf8", timeout: 120000, maxBuffer: 16 << 20});
    fs.writeFileSync(path.join(output, label + ".log"), (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Windows adapter stage failed: " + label, {cause: result.error});}
    return result.stdout;
  }
  const dependencies = [], adapters = [], failures = [];
  for (const kind of ["cuda", "level-zero-v2"]) {
    try {
      const factory = "mom_ur_" + kind.replaceAll("-", "_") + "_factory", bridge = path.join(output, kind + "-factory.obj"), dep = path.join(output, kind + "-factory.d");
      const command = compileCommand(config, path.join(repo, "tests/native/dpcpp-static/" + kind + "-factory.cpp"), bridge, dep, false);
      command.args.push("-I", urHeaders); run(command.command, command.args, kind + "-compile"); dependencies.push(dep);
      const archive = before.paths.get(kind === "cuda" ? "ur_adapter_cuda.lib" : "ur_adapter_level_zero_v2.lib"), merged = path.join(output, kind + ".lib");
      if (!archive) {throw new Error("Original adapter missing");}
      run(path.join(bin, "llvm-lib"), ["/nologo", "/out:" + merged, archive, bridge], kind + "-archive");
      const definitions = adapterDefinitions(run(tools.llvmNm, ["-P", "-g", "--defined-only", merged], kind + "-definitions"));
      if (!definitions.has(factory) || definitions.size < 100) {throw new Error("Incomplete Windows adapter factory closure");}
      const alternatives = rewriteCoffDirectives(run(config.tools.llvmReadobj, ["--coff-directives", merged], kind + "-directives"), new Map()).alternates;
      const shared = coffExternalFallbacks(definitions, alternatives);
      const isolated = isolateCoffCompilerArchives({archives: [merged], tools, output: path.join(output, kind + "-isolated"), prefix: "mom_ur_" + kind.replaceAll("-", "_") + "_private_",
        entryPoints: [factory], privateSymbols: [...definitions].filter(name => name !== factory && !shared.has(name))});
      adapters.push({kind, factory, isolation: path.join(output, kind + "-isolated/result.json"), archive: isolated.transformed[0]?.output, sha256: isolated.transformed[0]?.sha256});
    } catch (error) {failures.push(kind + ": " + (error instanceof Error ? error.message : "Windows adapter preparation failed"));}
  }
  fs.writeFileSync(path.join(output, "adapters.json"), JSON.stringify({adapters, failures}, null, 2) + "\n", {flag: "wx"});
  if (failures.length) {throw new Error(failures.join("\n"));}
  const archive = adapters.find(value => value.kind === "cuda")?.archive;
  if (!archive) {throw new Error("Missing isolated CUDA adapter");}
  const {names, pointers} = windowsDriverRequirements(run(tools.llvmNm, ["-P", "--undefined-only", archive], "driver-imports"));
  if (names.length < 20 || names.length > 512 || !names.includes("cuInit") || !names.includes("nvmlErrorString")) {throw new Error("Incomplete original CUDA driver imports");}
  const source = path.join(output, "drivers.cpp"), object = path.join(output, "drivers.obj"), dep = path.join(output, "drivers.d");
  fs.writeFileSync(source, windowsDriverBindings(names, "cuda"), {flag: "wx"});
  const command = compileCommand(config, source, object, dep, false); command.args.push("-I", path.join(cuda, "include"), "-I", repo);
  run(command.command, command.args, "driver-compile"); dependencies.push(dep);
  const aliases = windowsTypedDriverAliases(run(tools.llvmNm, ["-P", "--defined-only", object], "driver-symbols"), names, "cuda");
  // Keep linker aliases in an explicit object, not an archive-only member that could be
  // skipped before resolving a direct CUDA call. Both objects enter the final application.
  const aliasSource = path.join(output, "driver-aliases.c"), aliasObject = path.join(output, "driver-aliases.obj"), aliasDep = path.join(output, "driver-aliases.d");
  fs.writeFileSync(aliasSource, aliases.map(alias => "#pragma comment(linker, " + JSON.stringify(alias) + ")").join("\n") + "\n", {flag: "wx"});
  const aliasCommand = compileCommand(config, aliasSource, aliasObject, aliasDep, true); run(aliasCommand.command, aliasCommand.args, "driver-alias-compile"); dependencies.push(aliasDep);
  // Link every generated direct/import spelling. The probe compares addresses only: it
  // never calls a driver, and its PE imports must stay ordinary OS libraries.
  const bindingSource = path.join(output, "binding.cpp"), bindingObject = path.join(output, "binding.obj"), bindingDep = path.join(output, "binding.d"), binding = path.join(output, "driver-binding.exe");
  fs.writeFileSync(bindingSource, ["#define NVML_STATIC_IMPORT", "#include <cuda.h>", "#include <nvml.h>", "#include <cstdio>",
    ...names.map(name => `extern "C" decltype(&::${name}) __imp_${name};`), "int main() {",
    ...names.map(name => `if (&::${name} != __imp_${name}) return 86;`), 'std::puts("All direct/import NVIDIA driver bindings match without loading a driver"); return 0; }', ""].join("\n"), {flag: "wx"});
  const bindingCommand = compileCommand(config, bindingSource, bindingObject, bindingDep, false); bindingCommand.args.push("-I", path.join(cuda, "include"));
  run(bindingCommand.command, bindingCommand.args, "binding-compile"); dependencies.push(bindingDep);
  const response = path.join(output, "binding-link.rsp"), sdk = config.windowsSdk;
  fs.writeFileSync(response, windowsResponseFile(["/nologo", "/machine:x64", "/subsystem:console", "/entry:mainCRTStartup", "/opt:ref", "/opt:icf", "/incremental:no", "/out:" + binding,
    ...sdk.libraryRoots.map(root => "/libpath:" + root), bindingObject, object, aliasObject, ...sdk.msvcLibraries, ...sdk.ucrtLibraries, ...sdk.importLibraries]), {flag: "wx"});
  run(config.tools.lldLink, ["@" + response], "binding-link");
  const imports = parseImports(run(config.tools.llvmReadobj, ["--coff-imports", binding], "binding-imports"));
  if (imports.some(name => !sdk.allowedImports.map(name => name.toLowerCase()).includes(name))) {throw new Error("Driver binding probe imports an application DLL");}
  const driver = {objects: [object, aliasObject].map(path => ({path, sha256: fileHash(path)})), names, pointers, aliases};
  const link = windowsRuntimeLinks(before.sdk.artifacts, adapters, driver.objects, before.sdk.sourceJit?.enabled ? before.sdk.sourceJit.libraries : []);
  let hostJitBinding = null;
  if (before.sdk.sourceJit?.enabled === true) {
    // Link the SAME host-JIT lifecycle/source/cache/recovery checks as Linux. The dedicated
    // compile flag excludes device discovery; build success is not Windows execution proof.
    const source = path.join(repo, "tests/reference/native-dpcpp-static.cpp"), object = path.join(output, "host-jit.obj"), dep = path.join(output, "host-jit.d");
    const headers = path.join(before.source.prepared.dpcpp.root, "sycl-jit/jit-compiler/include");
    const command = compileCommand(config, source, object, dep, false);
    command.args.push("-DMOM_NATIVE_DPCPP_JIT", "-DMOM_NATIVE_DPCPP_JIT_ONLY", "-DSYCL_EXT_JIT_ENABLE=1", "-DNOMINMAX", "/FI" + path.join(repo, "tests/native/dpcpp-static/static-windows.h"),
      "-I", path.join(before.source.prepared.dpcpp.root, "sycl/source"), "-I", before.source.prepared.emhash.root, "-I", headers, "-I", before.sdk.headers[0].root);
    run(command.command, command.args, "host-jit-compile"); dependencies.push(dep);
    const executable = path.join(output, "host-jit.exe"), response = path.join(output, "host-jit-link.rsp");
    fs.writeFileSync(response, windowsResponseFile(["/nologo", "/machine:x64", "/subsystem:console", "/entry:mainCRTStartup", "/opt:ref", "/opt:icf", "/incremental:no", "/errorlimit:0", "/threads:2", "/out:" + executable,
      ...sdk.libraryRoots.map(root => "/libpath:" + root), object, ...link.objects, ...link.wholeArchives.map(file => "/wholearchive:" + file), ...link.libraries,
      ...sdk.msvcLibraries, ...sdk.ucrtLibraries, ...sdk.importLibraries]), {flag: "wx"});
    run(config.tools.lldLink, ["@" + response], "host-jit-link");
    const imports = parseImports(run(config.tools.llvmReadobj, ["--coff-imports", executable], "host-jit-imports"));
    if (imports.some(name => !sdk.allowedImports.map(name => name.toLowerCase()).includes(name))) {throw new Error("Host-JIT probe imports an application DLL");}
    hostJitBinding = {path: executable, sha256: fileHash(executable), imports, executed: false};
  }
  const headers = dependencyHashes(dependencies, config.windowsSdk.overlay, output, startedAt);
  adapterSdkInputs(sdkFile, descriptorFile, cuda); verifySourceInputs(recipe, sourceInputs(repo, [], leaves), "Windows adapter recipe");
  for (const [file, hash] of Object.entries(inputs)) {if (fileHash(file) !== hash) {throw new Error("Windows adapter input changed: " + file);}}
  for (const row of driver.objects) {if (fileHash(row.path) !== row.sha256) {throw new Error("Windows driver binding object changed");}}
  for (const adapter of adapters) {if (!adapter.archive || fileHash(adapter.archive) !== adapter.sha256) {throw new Error("Isolated Windows adapter changed");}}
  const receipt = {version: 1, sdkFile, descriptorFile, inputs, recipe, adapters, headers, driver, link, hostJitBinding,
    binding: {path: binding, sha256: fileHash(binding), imports, executed: false},
    buildOnly: true, executed: false, publicationQualified: false, scope: "Original isolated Windows CUDA/L0v2 adapters and SDK-typed lazy CUDA/NVML thunks. No GPU/runtime/performance or complete release qualification."};
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(receipt, null, 2) + "\n", {flag: "wx"});
  console.log("[native-sdk] Prepared original Windows adapters and " + names.length + " typed driver entries; not executed.");
  return receipt;
}

if (require.main === module) {
  try {
    const [sdk, descriptor, cuda, output, extra] = process.argv.slice(2);
    if (!sdk || !descriptor || !cuda || !output || extra) {throw new Error("Usage: prepare-native-windows-adapters.js /sdk-result.json /windows-build.json /CUDA /new-output");}
    prepareWindowsAdapters(sdk, descriptor, cuda, output);
  } catch (error) {console.error(error instanceof Error ? error.message : "Windows adapter preparation failed"); process.exitCode = 1;}
}
module.exports = {adapterDefinitions, windowsDriverRequirements, windowsRuntimeLinks, adapterSdkInputs, prepareWindowsAdapters};
