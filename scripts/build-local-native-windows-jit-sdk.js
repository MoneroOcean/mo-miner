"use strict";

// Continue admitted local inputs through the original Windows source-JIT producer and
// COFF isolation pipeline. This is a build-only WSL adapter, not a release command.
const fs = require("node:fs"), path = require("node:path"), {isDeepStrictEqual} = require("node:util");
const local = require("./build-local-native-sdk"), localJit = require("./build-local-native-jit-sdk"),
  localWindowsLlvm = require("./build-local-native-windows-llvm-sdk"), localWindowsResources = require("./build-local-native-windows-jit-resources"),
  nativeCi = require("./native-ci-linux-jit"),
  windowsToolchain = require("./prepare-native-jit-toolchain"), jitSdk = require("./build-native-jit-sdk"),
  windowsIsolation = require("./isolate-native-windows-jit"),
  {dpcppEntries} = require("./native-compiler-isolation"), {jitBuildStages} = require("./native-jit-stages"),
  {vfsFiles} = require("./build-native-windows-controller");
const {fileHash, readBuildJson, buildFileInputs, verifySourceInputs} = require("./build-source-inputs");

const repo = fs.realpathSync(path.resolve(__dirname, "..")), hashPattern = /^[a-f0-9]{64}$/;

/** @param {unknown} value @param {string[]} keys @param {string} label */
function exactObject(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).sort().join(",") !== keys.slice().sort().join(",")) {
    throw new Error(label + " has unknown or missing fields");
  }
  return /** @type {Record<string, any>} */ (value);
}

/** @param {string} file @param {string} label @param {number} [maximum] */
function regularFile(file, label, maximum = 8 << 20) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!path.isAbsolute(file) || path.resolve(file) !== file || !stat?.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 ||
      stat.size < 1 || stat.size > maximum || fs.realpathSync(file) !== file) {
    throw new Error(label + " is not a bounded canonical regular file: " + file);
  }
  return stat;
}

/** @param {string} directory @param {string} label @param {boolean} [allowMissing] */
function realDirectory(directory, label, allowMissing = false) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  if (!stat && allowMissing) {return false;}
  if (!stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
    throw new Error(label + " is not a canonical directory: " + directory);
  }
  return true;
}

/** Read only a bounded direct inventory; interrupted or unknown state is preserved. @param {string} directory
 * @param {string[]} expected @param {string} label */
function exactEntries(directory, expected, label) {
  const found = [], handle = fs.opendirSync(directory);
  try {
    for (let entry; (entry = handle.readSync()) !== null;) {
      found.push(entry.name);
      if (found.length > expected.length) {throw new Error(label + " contains unknown entries; preserve it for inspection");}
    }
  } finally {handle.closeSync();}
  if (!isDeepStrictEqual(found.sort(), expected.slice().sort())) {throw new Error(label + " is incomplete or contains unknown entries; preserve it for inspection");}
}

/** @param {unknown} value @returns {value is Record<string, any>} */
function record(value) {return value !== null && typeof value === "object" && !Array.isArray(value);}

/** @param {string} workspace */
function localWindowsJitSdkPaths(workspace) {
  if (typeof workspace !== "string" || !path.isAbsolute(workspace) || path.resolve(workspace) !== workspace || /[\r\n\0]/.test(workspace)) {
    throw new Error("Local Windows JIT workspace must be canonical and absolute");
  }
  const root = path.join(workspace, "windows-jit-sdk");
  return {root, descriptor: path.join(root, "descriptor.json"), tools: path.join(root, "llvm-tools.json"),
    config: path.join(root, "config.json"), toolchain: path.join(root, "toolchain"), output: path.join(root, "compiler-sdk"),
    isolation: path.join(root, "isolation"), provenance: path.join(root, "provenance.json")};
}

/** @param {ReturnType<typeof localWindowsJitSdkPaths>} paths @param {boolean} allowMissing */
function checkManagedRoot(paths, allowMissing) {
  if (!realDirectory(paths.root, "Local Windows JIT managed output", allowMissing)) {return false;}
  exactEntries(paths.root, ["config.json", "descriptor.json", "compiler-sdk", "isolation", "llvm-tools.json", "provenance.json", "toolchain"],
    "Local Windows JIT managed output");
  for (const file of [paths.config, paths.descriptor, paths.tools, paths.provenance]) {regularFile(file, "Local Windows JIT " + path.basename(file));}
  for (const directory of [paths.toolchain, paths.output, paths.isolation]) {realDirectory(directory, "Local Windows JIT " + path.basename(directory));}
  return true;
}

/** @param {ReturnType<typeof local.supportPaths>} support @param {ReturnType<typeof jitSdk.jitSdkConfig>} config */
function localWindowsJitCompilerConfig(support, config) {
  const value = jitSdk.jitSdkConfig(config);
  if (value.target !== "windows" || value.sources !== path.join(support.compilerSources, "sources.json") ||
      value.sdk !== path.join(support.dpcppCudaRoot, "dpcpp") || value.resources !== path.join(support.workspace, "windows-jit-resources", "resources") ||
      value.tools.toolchain !== path.join(support.workspace, "windows-jit-sdk", "toolchain", "toolchain.cmake") ||
      !Array.isArray(value.inputs) || !value.inputs.includes(value.tools.toolchain)) {
    throw new Error("Local Windows JIT config differs from the current admitted source and toolchain roots");
  }
  return value;
}

/** @param {unknown} value @returns {Buffer} */
function jsonBytes(value) {return Buffer.from(JSON.stringify(value, null, 2) + "\n");}

/** @param {string} file @param {unknown} expected @param {string} label */
function verifyStableJson(file, expected, label) {
  const bytes = jsonBytes(expected), stat = regularFile(file, label);
  if (stat.size !== bytes.length || !fs.readFileSync(file).equals(bytes)) {throw new Error(label + " is stale; preserve it for inspection");}
  return fileHash(file);
}

/** Verify the generated Windows aliases/toolchain against the public pure plan and its own original receipt.
 * All external inputs named by the original producer are re-hashed; generated files are derived from that plan. @param {string} descriptorFile
 * @param {string} sdk @param {string} nativeTools @param {string} output @param {{cc:string,cxx:string}} hostCompilers */
function verifyLocalWindowsJitToolchain(descriptorFile, sdk, nativeTools, output, hostCompilers) {
  const resultFile = path.join(output, "result.json"), result = /** @type {any} */ (readBuildJson(resultFile)), plan = windowsToolchain.windowsJitToolchainPlan(
    readBuildJson(descriptorFile), sdk, nativeTools, output, hostCompilers);
  if (!record(result) || Object.keys(result).sort().join(",") !== "buildOnly,compiled,executed,inputs,publicationQualified,toolchain,version" ||
      result["version"] !== 1 || result["buildOnly"] !== true || result["compiled"] !== false || result["executed"] !== false ||
      result["publicationQualified"] !== false || result["toolchain"] !== path.join(output, "toolchain.cmake") || !record(result["inputs"])) {
    throw new Error("Windows JIT toolchain producer receipt is incomplete or stale");
  }
  const virtual = [...vfsFiles(plan.config.windowsSdk.overlay)].filter(([name]) => plan.config.windowsSdk.includeRoots.some(root =>
    name.toLowerCase().startsWith(root.replace(/\/+$/, "").toLowerCase() + "/")));
  if (!virtual.length || virtual.length > 65536) {throw new Error("Windows JIT toolchain has no bounded OS header mapping");}
  const imports = new Map();
  for (const root of plan.config.windowsSdk.libraryRoots) {
    const names = fs.readdirSync(root);
    if (names.length > 65536) {throw new Error("Windows JIT library root inventory exceeds bounds");}
    for (const name of names) {
      if (!/^[A-Za-z0-9_.-]+\.lib$/i.test(name)) {continue;}
      const file = path.join(root, name), key = name.toLowerCase(), previous = imports.get(key);
      if (previous && fileHash(previous) !== fileHash(file)) {throw new Error("Conflicting Windows JIT OS library aliases: " + name);}
      if (!previous) {imports.set(key, file);}
    }
  }
  const aliases = path.join(output, "bin"), libraries = path.join(output, "lib"), llvmAr = path.join(sdk, "bin/llvm-ar");
  exactEntries(aliases, ["llvm-lib", "llvm-ranlib"], "Windows JIT LLVM aliases");
  for (const name of ["llvm-lib", "llvm-ranlib"]) {
    const alias = path.join(aliases, name), stat = fs.lstatSync(alias);
    if (!stat.isSymbolicLink() || fs.readlinkSync(alias) !== llvmAr || fs.realpathSync(alias) !== llvmAr) {
      throw new Error("Windows JIT LLVM alias differs from its admitted SDK archiver: " + name);
    }
  }
  exactEntries(libraries, [...imports.keys()].sort(), "Windows JIT import-library aliases");
  for (const [name, source] of imports) {
    const alias = path.join(libraries, name), stat = fs.lstatSync(alias);
    if (!stat.isSymbolicLink() || fs.readlinkSync(alias) !== source || fs.realpathSync(alias) !== source) {
      throw new Error("Windows JIT import-library alias differs from its admitted SDK library: " + name);
    }
  }
  const overlayFile = JSON.stringify({version: 0, "case-sensitive": false, "use-external-names": true,
      roots: virtual.map(([name, file]) => ({type: "file", name, "external-contents": file}))}),
    expectedGenerated = new Map([[plan.overlay, overlayFile], [plan.sourceOptionsFile, plan.sourceOptionsText], [path.join(output, "toolchain.cmake"), plan.toolchain]]);
  if (plan.ruleOverride && plan.ruleOverrideText) {expectedGenerated.set(plan.ruleOverride, plan.ruleOverrideText);}
  const requiredInputs = [descriptorFile, ...virtual.map(([, file]) => file), ...Object.values(plan.compilerTools).filter(file => !file.startsWith(output + path.sep)),
    llvmAr, ...imports.values(), plan.config.windowsSdk.overlay];
  for (const file of requiredInputs) {
    if (typeof result["inputs"][file] !== "string" || !hashPattern.test(result["inputs"][file] || "") || fileHash(file) !== result["inputs"][file]) {
      throw new Error("Windows JIT toolchain omitted or changed an original producer input: " + file);
    }
  }
  for (const [file, contents] of expectedGenerated) {
    regularFile(file, "Windows JIT generated toolchain file");
    if (fs.readFileSync(file, "utf8") !== contents || result["inputs"][file] !== fileHash(file)) {
      throw new Error("Windows JIT generated toolchain file differs from its public plan: " + file);
    }
  }
  const currentInputs = buildFileInputs(Object.keys(result["inputs"]));
  if (!isDeepStrictEqual(currentInputs, result["inputs"])) {throw new Error("Windows JIT toolchain input fingerprints are stale");}
  const expectedEntries = ["bin", "lib", "vfs.json", "heatutils-source-options.cmake", "intel-link-rules.cmake", "toolchain.cmake", "result.json"];
  exactEntries(output, expectedEntries, "Windows JIT toolchain output");
  return result;
}

/** @param {any} linuxAdmission */
function localWindowsJitIsolationTools(linuxAdmission) {
  const sdk = linuxAdmission?.sdk, root = sdk?.path, snapshot = sdk?.snapshot?.files;
  if (typeof root !== "string" || !record(snapshot)) {throw new Error("Local Windows JIT isolation needs the admitted Linux LLVM tools snapshot");}
  const tools = {llvmAr: path.join(root, "bin/llvm-ar"), llvmNm: path.join(root, "bin/llvm-nm"), llvmObjcopy: path.join(root, "bin/llvm-objcopy")};
  for (const [name, file] of Object.entries(tools)) {
    const row = snapshot["bin/" + path.basename(file)];
    regularFile(file, "Admitted Linux LLVM " + name, 256 << 20);
    if (!row || row.type !== "file" || row.sha256 !== fileHash(file)) {throw new Error("Selected Linux LLVM isolation tool differs from its admission: " + name);}
  }
  return tools;
}

/** @param {ReturnType<typeof localWindowsJitIsolationTools>} tools */
function localWindowsJitToolsManifest(tools) {
  const manifest = {version: 1, tools};
  windowsIsolation.validateToolsManifest(manifest);
  return manifest;
}

/** Reconstruct the original isolation handoff without inventing another archive/phase protocol.
 * @param {ReturnType<typeof localWindowsJitSdkPaths>} paths @param {any} prior @param {any} isolation */
function expectedIsolationHandoff(paths, prior, isolation) {
  return windowsIsolation.windowsJitIsolationHandoff(prior, isolation, paths.tools, paths.isolation);
}

/** Admit the original isolation result, all transformed archives, tool/source pins and exact output inventory.
 * @param {ReturnType<typeof localWindowsJitSdkPaths>} paths @param {any} prior */
function verifyLocalWindowsJitIsolation(paths, prior) {
  const resultFile = path.join(paths.isolation, "result.json"), handoffFile = path.join(paths.isolation, "handoff.json");
  regularFile(resultFile, "Windows JIT isolation result", 16 << 20); regularFile(handoffFile, "Windows JIT isolation handoff", 16 << 20);
  const result = /** @type {any} */ (readBuildJson(resultFile)), expectedTools = Object.fromEntries(Object.values(prior.tools).map(file => [file, fileHash(file)])),
    expectedSources = Object.fromEntries(
      [require.resolve("./native-coff-archive"), require.resolve("./native-compiler-isolation"), require.resolve("./build-source-inputs")]
        .map(file => [file, fileHash(file)]));
  if (!record(result) || Object.hasOwn(result, "kind") || result["version"] !== 1 || result["prefix"] !== "mom_dpcpp23_" ||
      Object.keys(result).sort().join(",") !== "buildOnly,checks,entryPoints,executed,map,mapSha256,prefix,publicationQualified,sourceHashes,symbols,toolHashes,transformed,version" ||
      !isDeepStrictEqual(result["entryPoints"], dpcppEntries) || !Number.isInteger(result["symbols"]) || result["symbols"] < 1 ||
      result["map"] !== path.join(paths.isolation, "rename.map") || result["mapSha256"] !== fileHash(result["map"]) ||
      !isDeepStrictEqual(result["toolHashes"], expectedTools) || !isDeepStrictEqual(result["sourceHashes"], expectedSources) ||
      !isDeepStrictEqual(result["checks"], {memberOrderPreserved: true, nonDirectivePayloadsPreserved: true, mappedDirectivesPreserved: true,
        symbolOrderPreserved: true, auxiliaryRecordsPreserved: true, relocationsPreserved: true}) ||
      result["buildOnly"] !== true || result["executed"] !== false || result["publicationQualified"] !== false || !Array.isArray(result["transformed"]) ||
      result["transformed"].length !== prior.archives.length) {
    throw new Error("Windows JIT isolation result is incomplete or no longer matches the public COFF producer");
  }
  regularFile(result["map"], "Windows JIT COFF symbol map", 64 << 20);
  const expectedNames = ["result.json", "handoff.json", "rename.map"];
  for (const [index, input] of prior.archives.entries()) {
    const row = /** @type {any[]} */ (result["transformed"])[index], output = path.join(paths.isolation, path.basename(input));
    if (!row || row.input !== input || row.inputSha256 !== prior.result.libraries[index]?.sha256 || row.output !== output ||
        !hashPattern.test(row.sha256 || "") || !Number.isSafeInteger(row.rewrittenMembers) || row.rewrittenMembers < 0 ||
        !Number.isSafeInteger(row.members) || row.members < 1 || !hashPattern.test(row.checkedMemberSha256 || "") ||
        fileHash(input) !== row.inputSha256) {
      throw new Error("Windows JIT isolated archive order or original hash changed");
    }
    regularFile(output, "Windows JIT isolated archive", 256 << 20);
    if (fileHash(output) !== row.sha256) {throw new Error("Windows JIT isolated archive bytes changed");}
    expectedNames.push(path.basename(input));
    for (const label of ["definitions-" + index, "isolate-" + index, ...(row.rewrittenMembers ? ["directives-" + index] : [])]) {
      expectedNames.push(label + ".command.json", label + ".log");
    }
  }
  for (let index = 0; index < Object.keys(prior.tools).length; index++) {expectedNames.push("tool-" + index + ".command.json", "tool-" + index + ".log");}
  exactEntries(paths.isolation, expectedNames, "Windows JIT isolation output");
  for (const name of expectedNames.filter(name => name.endsWith(".log") || name.endsWith(".command.json"))) {
    regularFile(path.join(paths.isolation, name), "Windows JIT isolation evidence", 64 << 20);
  }
  const expectedHandoff = expectedIsolationHandoff(paths, prior, result);
  if (!isDeepStrictEqual(readBuildJson(handoffFile), expectedHandoff)) {throw new Error("Windows JIT isolation handoff does not bind the inspected producer and archive transformation");}
  return {result, handoff: expectedHandoff, handoffSha256: fileHash(handoffFile), outputSha256: fileHash(resultFile),
    files: expectedNames.slice().sort().map((/** @type {string} */ name) => ({name, sha256: fileHash(path.join(paths.isolation, name))}))};
}

/** @param {unknown} input @param {NodeJS.ProcessEnv} environment @param {ReturnType<typeof local.localTools>} tools
 * @param {any} dependencies */
function admitLocalWindowsJitInputs(input, environment, tools, dependencies) {
  if (dependencies.admitInputs) {return dependencies.admitInputs(input, environment, tools);}
  if (!record(input)) {throw new Error("Local Windows JIT inputs are malformed");}
  const args = /** @type {Record<string, any>} */ (input), workspace = local.admitWorkspace(args["workspace"], repo), support = local.supportPaths(workspace), jitTools = {
      cmake: tools.cmake, ninja: tools.ninja, cc: tools.cc, cxx: tools.cxx}, llvmPaths = localWindowsLlvm.localWindowsLlvmSdkPaths(workspace),
    resourcesPaths = localWindowsResources.localWindowsJitResourcePaths(workspace);
  if (!fs.lstatSync(llvmPaths.root, {throwIfNoEntry: false}) || !fs.lstatSync(resourcesPaths.root, {throwIfNoEntry: false})) {
    throw new Error("Local Windows JIT needs completed Windows LLVM and JIT-resource outputs; it will not create prerequisites");
  }
  const sdkInput = {workspace, windowsHostSdk: args["windowsHostSdk"], hermes: args["hermes"], tools},
    checkedLlvm = localWindowsLlvm.buildLocalNativeWindowsLlvmSdk(sdkInput, {environment, verifyScope() {return true;}}),
    suppliedLlvm = /** @type {any} */ (args["windowsLlvm"]);
  if (!record(suppliedLlvm) || suppliedLlvm["workspace"] !== workspace || suppliedLlvm["root"] !== llvmPaths.root ||
      suppliedLlvm["output"] !== llvmPaths.output || suppliedLlvm["descriptor"] !== llvmPaths.descriptor ||
      suppliedLlvm["provenance"] !== llvmPaths.provenance || suppliedLlvm["descriptorSha256"] !== checkedLlvm.descriptorSha256 ||
      suppliedLlvm["provenanceSha256"] !== checkedLlvm.provenanceSha256) {
    throw new Error("Local Windows LLVM handoff differs from the current full warm admission");
  }
  const localSdk = {workspace, compilerFetch: support.compilerFetch, compilerSources: support.compilerSources,
      buildInputs: support.buildInputs, llvmSdk: support.llvmSdk},
    checkedResources = localWindowsResources.buildLocalWindowsJitResources(localSdk, jitTools, {environment, verifyScope() {return true;}}),
    suppliedResources = /** @type {any} */ (args["windowsJitResources"]);
  if (!record(suppliedResources) || suppliedResources["workspace"] !== workspace || suppliedResources["root"] !== resourcesPaths.root ||
      suppliedResources["target"] !== checkedResources.target || suppliedResources["resources"]?.root !== resourcesPaths.resources ||
      suppliedResources["resources"]?.manifestSha256 !== checkedResources.resources.manifestSha256 ||
      suppliedResources["profile"]?.sha256 !== checkedResources.profile.sha256 ||
      suppliedResources["emitter"]?.receiptSha256 !== checkedResources.emitter.receiptSha256) {
    throw new Error("Local Windows JIT resources differ from the current public producer admission");
  }
  const sourceReceipt = path.join(support.compilerSources, "sources.json"), source = require("./prepare-native-sdk-sources").verifyPreparedSdkSources(sourceReceipt, "compiler"),
    linuxResources = nativeCi.auditLinuxJitResourceTree(local.localLinuxJitResourceState(support), jitTools, environment),
    linuxBaseReceipt = /** @type {any} */ (readBuildJson(path.join(support.linuxJitResources, "base/result.json"))),
    nativeToolRoot = path.join(support.linuxJitResources, "bootstrap/build/bin");
  if (checkedResources.linux.root !== support.linuxJitResources || linuxResources.resources?.target !== "x86_64-unknown-linux-gnu" ||
      linuxResources.resources.directory !== path.join(support.linuxJitResources, "resources") ||
      checkedResources.linux.manifestSha256 !== linuxResources.resources.manifestHash ||
      linuxBaseReceipt.build?.root !== path.join(support.linuxJitResources, "bootstrap/build")) {
    throw new Error("Local Windows JIT native generators differ from the admitted Linux JIT resource bootstrap");
  }
  for (const [name, key] of /** @type {Array<[string,string]>} */ ([
    ["clang-tblgen", "clangTableGenSha256"], ["llvm-tblgen", "llvmTableGenSha256"]])) {
    const file = path.join(nativeToolRoot, name), stat = regularFile(file, "Admitted native " + name, 512 << 20);
    if ((stat.mode & 0o777) !== 0o755 || fileHash(file) !== linuxBaseReceipt.build?.[key]) {
      throw new Error("Local Windows JIT native generator differs from its original base receipt: " + name);
    }
  }
  const descriptor = readBuildJson(llvmPaths.descriptor), descriptorTools = localWindowsLlvm.localWindowsLlvmDescriptorTools(tools, checkedLlvm.linux);
  if (!isDeepStrictEqual(descriptor.tools, descriptorTools)) {throw new Error("Windows JIT compiler/toolchain differs from the admitted Intel/LLVM descriptor");}
  const isolationTools = localWindowsJitIsolationTools(checkedLlvm.linux), toolManifest = localWindowsJitToolsManifest(isolationTools);
  return {workspace, support, paths: localWindowsJitSdkPaths(workspace), windowsLlvm: checkedLlvm, windowsResources: checkedResources,
    descriptor, descriptorFile: llvmPaths.descriptor, hostCompiler: descriptor.tools.clangCl.path,
    hostCompilers: {cc: descriptor.tools.clangCl.path, cxx: descriptor.tools.clangCl.path},
    sdk: path.join(support.dpcppCudaRoot, "dpcpp"), nativeTools: nativeToolRoot, sources: sourceReceipt,
    resources: resourcesPaths.resources, isolationTools, toolManifest, linuxResources, source,
    windowsHostSdk: args["windowsHostSdk"], hermes: args["hermes"], tools};
}

/** @param {ReturnType<typeof localWindowsJitSdkPaths>} paths @param {any} context */
function localWindowsJitDescriptor(paths, context) {
  const configSha256 = fileHash(paths.config);
  return {version: 1, kind: "local-windows-jit-sdk", target: "windows", workspace: context.workspace,
    windowsLlvm: {root: context.windowsLlvm.root, output: context.windowsLlvm.output,
      descriptorSha256: context.windowsLlvm.descriptorSha256, provenanceSha256: context.windowsLlvm.provenanceSha256},
    windowsJitResources: {root: context.windowsResources.root, resources: context.resources,
      profileSha256: context.windowsResources.profile.sha256, emitterReceiptSha256: context.windowsResources.emitter.receiptSha256,
      manifestSha256: context.windowsResources.resources.manifestSha256, linuxManifestSha256: context.windowsResources.linux.manifestSha256},
    windowsHostSdk: context.windowsHostSdk, hermes: context.hermes, tools: context.tools,
    toolchainResultSha256: fileHash(path.join(paths.toolchain, "result.json")), configSha256};
}

/** @param {ReturnType<typeof localWindowsJitSdkPaths>} paths @param {any} prior @param {any} isolated */
function localWindowsJitProvenance(paths, prior, isolated) {
  return {version: 1, kind: "local-windows-jit-sdk-result", target: "windows", buildOnly: true, compiled: true,
    executed: false, publicationQualified: false, descriptorSha256: fileHash(paths.descriptor), configSha256: fileHash(paths.config),
    toolsManifestSha256: fileHash(paths.tools), toolchainResultSha256: fileHash(path.join(paths.toolchain, "result.json")),
    producerResultSha256: prior.resultSha256, isolationHandoffSha256: isolated.handoffSha256, isolationFiles: isolated.files};
}

/** @param {unknown} input @param {{verifyScope?:()=>unknown,resolveTools?:typeof local.localTools,environment?:NodeJS.ProcessEnv,
 * admitInputs?:(input:any,environment:NodeJS.ProcessEnv,tools:ReturnType<typeof local.localTools>)=>any,
 * prepareToolchain?:typeof windowsToolchain.prepareWindowsJitToolchain,buildPhase?:typeof jitSdk.buildJitSdk,
 * inspect?:typeof windowsIsolation.inspectWindowsJitSdk,verifyToolchain?:typeof verifyLocalWindowsJitToolchain,
 * runIsolation?:(prior:any,toolsFile:string,output:string,environment:NodeJS.ProcessEnv)=>unknown}} [dependencies]
 * @returns {Promise<any>} */
async function buildLocalNativeWindowsJitSdk(input, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local Windows JIT SDK production requires Linux x64 (WSL2)");}
  (dependencies.verifyScope || local.verifyCurrentMemoryScope)();
  if (!record(input) || Object.keys(input).sort().join(",") !== "hermes,tools,windowsHostSdk,windowsJitResources,windowsLlvm,workspace" ||
      typeof input["workspace"] !== "string") {
    throw new Error("Local Windows JIT needs the marked workspace, admitted Windows LLVM/resources, Hermes/host-SDK pins and selected tools");
  }
  const jobs = local.localBuildJobs({...process.env, ...(dependencies.environment || {})}),
    environment = {...process.env, ...(dependencies.environment || {}), MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: String(jobs)},
    resolveTools = dependencies.resolveTools || local.localTools, tools = input["tools"], selectedTools = resolveTools(environment);
  exactObject(tools, ["cc", "cmake", "cxx", "git", "ninja", "systemdRun"], "Selected local tools");
  if (!isDeepStrictEqual(tools, selectedTools)) {throw new Error("Local Windows JIT tools differ from the current pinned selection");}
  const admit = () => admitLocalWindowsJitInputs(input, environment, tools, dependencies), before = admit(), paths = before.paths,
    toolManifestBytes = jsonBytes(before.toolManifest),
    verifyToolchain = dependencies.verifyToolchain || verifyLocalWindowsJitToolchain,
    prepareToolchain = dependencies.prepareToolchain || windowsToolchain.prepareWindowsJitToolchain,
    buildPhase = dependencies.buildPhase || jitSdk.buildJitSdk,
    inspect = dependencies.inspect || windowsIsolation.inspectWindowsJitSdk,
    runIsolation = dependencies.runIsolation || ((prior, toolsFile, output, env) => local.runCommand(process.execPath,
      [require.resolve("./isolate-native-windows-jit"), prior.sdkRoot, prior.resultSha256, toolsFile, output],
      {label: "Local Windows JIT COFF archive isolation", env, cwd: repo}));
  const existing = checkManagedRoot(paths, true);
  if (!existing) {
    fs.mkdirSync(paths.root, {mode: 0o700});
    fs.writeFileSync(paths.tools, toolManifestBytes, {flag: "wx", mode: 0o444});
    prepareToolchain(before.descriptorFile, before.sdk, before.nativeTools, paths.toolchain, environment, before.hostCompilers);
  }
  if (existing) {verifyStableJson(paths.tools, before.toolManifest, "Local Windows JIT LLVM tools manifest");}
  // A warm compiler keeps the job count its config recorded; that count is provenance, not an output input.
  const toolchainResult = verifyToolchain(before.descriptorFile, before.sdk, before.nativeTools, paths.toolchain, before.hostCompilers),
    config = localWindowsJitCompilerConfig(before.support, {version: 1, target: "windows", sources: before.sources, sdk: before.sdk,
      resources: before.resources, tools: {cmake: tools.cmake, ninja: tools.ninja, toolchain: toolchainResult["toolchain"]},
      inputs: Object.keys(toolchainResult["inputs"]), jobs: existing ? readBuildJson(paths.config)?.jobs : jobs});
  if (!existing) {
    fs.writeFileSync(paths.config, jsonBytes(config), {flag: "wx", mode: 0o444});
    fs.writeFileSync(paths.descriptor, jsonBytes(localWindowsJitDescriptor(paths, before)), {flag: "wx", mode: 0o444});
  } else {
    const configSha256 = verifyStableJson(paths.config, config, "Local Windows JIT config"),
      descriptorSha256 = verifyStableJson(paths.descriptor, localWindowsJitDescriptor(paths, before), "Local Windows JIT descriptor");
    const saved = readBuildJson(paths.provenance);
    if (saved?.descriptorSha256 !== descriptorSha256 || saved.configSha256 !== configSha256 || !hashPattern.test(saved.producerResultSha256 || "")) {
      throw new Error("Local Windows JIT result does not match its original descriptor/config; preserve it");
    }
  }

  const buildEnv = {...environment, MOM_AOT_BUILD_ONLY: "1"};
  if (!existing) {
    await buildPhase(config, paths.output, buildEnv, "configure");
    for (const phase of jitBuildStages) {await buildPhase(config, paths.output, buildEnv, phase);}
  }
  if (localJit.localJitSdkPhasePrefix(paths.output) !== "link") {throw new Error("Local Windows JIT output is incomplete; preserve it for inspection");}
  regularFile(path.join(paths.output, "result.json"), "Original Windows JIT result");
  const producerResultSha256 = fileHash(path.join(paths.output, "result.json"));
  if (existing && readBuildJson(paths.provenance).producerResultSha256 !== producerResultSha256) {
    throw new Error("Original Windows JIT result changed after its local handoff; preserve it");
  }
  const readmittedProducer = () => {
    const current = inspect(paths.output, producerResultSha256, paths.tools), beforeBuild = readBuildJson(path.join(paths.output, "before.json"));
    verifySourceInputs(current.result.recipe, jitSdk.jitSdkRecipe(), "Local Windows JIT producer");
    if (!isDeepStrictEqual(beforeBuild.config, config) || !isDeepStrictEqual(current.inputHashes, buildFileInputs(Object.keys(current.inputHashes || {}))) ||
        current.resultSha256 !== producerResultSha256) {
      throw new Error("Original Windows JIT input hashes or config changed after compilation; preserve the output");
    }
    return current;
  };
  const prior = readmittedProducer();
  const isolationExists = fs.lstatSync(paths.isolation, {throwIfNoEntry: false});
  if (!isolationExists) {
    windowsIsolation.validateFreshOutput(prior, paths.tools, paths.isolation);
    runIsolation(prior, paths.tools, paths.isolation, buildEnv);
  }
  const checkedPrior = readmittedProducer();
  if (!isDeepStrictEqual(prior, checkedPrior)) {throw new Error("Windows JIT producer or its inputs changed during COFF isolation");}
  const isolated = verifyLocalWindowsJitIsolation(paths, checkedPrior), after = admit();
  if (!isDeepStrictEqual(before, after)) {throw new Error("Local Windows JIT inputs changed during production; preserve the output");}
  const descriptorSha256 = fileHash(paths.descriptor), configSha256 = fileHash(paths.config),
    provenance = localWindowsJitProvenance(paths, checkedPrior, isolated);
  if (!existing) {
    fs.writeFileSync(paths.provenance, jsonBytes(provenance), {flag: "wx", mode: 0o444});
  } else {verifyStableJson(paths.provenance, provenance, "Local Windows JIT provenance");}
  checkManagedRoot(paths, false);
  return {workspace: before.workspace, root: paths.root, descriptor: paths.descriptor, descriptorSha256, config: paths.config,
    configSha256, toolchain: paths.toolchain, compilerSdk: paths.output, compilerResultSha256: checkedPrior.resultSha256,
    isolation: paths.isolation, isolationHandoffSha256: isolated.handoffSha256, provenance: paths.provenance,
    jobs, buildOnly: true, compiled: true, executed: false, publicationQualified: false, reused: existing};
}

module.exports = {localWindowsJitSdkPaths, localWindowsJitCompilerConfig, verifyLocalWindowsJitToolchain,
  localWindowsJitIsolationTools, localWindowsJitToolsManifest, expectedIsolationHandoff, verifyLocalWindowsJitIsolation,
  admitLocalWindowsJitInputs, buildLocalNativeWindowsJitSdk};
