"use strict";

// Shared Linux AOT controller producer, invoked explicitly by the native source builder.
// Engineering Mode never invokes it or pays the AOT/relink cost.
// Use a pinned, separately built Static Hermes; do not download or mutate toolchains here.
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {fork, spawn, spawnSync} = require("node:child_process");
const {once} = require("node:events");
const tls = require("node:tls");
const {createHash} = require("node:crypto");
const {isDeepStrictEqual} = require("node:util");
const {controllerBundleOptions, controllerAotCommand, verifyBundle, nativeApplicationRoot} = require("./native-bundle");
const {fileHash, sourceInputs, nativeSourceInputs, verifySourceInputs, readBuildJson} = require("./build-source-inputs");
const nativeResources = require("./native-resources");
const {admittedJitCacheTag, nativeJitCacheHeader} = require("./native-jit-cache-tag");
const {assertUniqueHcfIds} = require("./native-hcf");
const {nativeDependencies, auditNativeGlibc, auditNativeVersionNeeds, auditNativeSymbolFile} = require("./native-audit");
const {linuxGlibcBaseline} = require("./native-linux-sysroot");
const {gpuTable, gpuForwards} = require("./native-gpu-table");
const {comgrSdkInputs} = require("./native-comgr-inputs");
const {nativeIoInputs} = require("./build-native-io");
const {compileNativeObjects} = require("./native-object-cache");

const revision = "26fbb82fad39d9d06901adb1d6d630dacc84c877";
/** @type {null | {repo: string, output: string, namespace: string,
 * commands: Array<{command: string, args: string[], source: string, object: string, depfile: string,
 * timeout: number, generatedInputs: string[], log: string}>, includeRoots: Set<string>}} */
let activeObjectBatch = null;

/** @param {string[]} args @param {string} source @param {string} output */
function linuxObjectIncludeRoots(args, source, output) {
  const roots = new Set(), outputRoot = path.resolve(output);
  for (let i = 0; i < args.length; ++i) {
    const arg = args[i] || "";
    let root;
    if (arg === "-I" || arg === "-isystem") {root = args[++i];} else if (arg.startsWith("-I") && arg.length > 2) {root = arg.slice(2);}
    if (!root || !path.isAbsolute(root)) {continue;}
    const selected = path.resolve(root);
    if (selected === outputRoot || selected.startsWith(outputRoot + path.sep)) {continue;}
    roots.add(selected);
  }
  const sourceDirectory = path.dirname(path.resolve(source));
  if (sourceDirectory !== outputRoot && !sourceDirectory.startsWith(outputRoot + path.sep)) {roots.add(sourceDirectory);}
  return [...roots];
}

/** Output-local generated headers are explicit cache inputs; the source itself is staged by the cache helper. */
/** @param {string} source @param {string} output */
function linuxGeneratedCompileInputs(source, output) {
  const name = path.basename(source), files = [];
  if (name === "services.cpp") {files.push(path.join(output, "app-resources.inc"));}
  if (name === "tls.cpp") {files.push(path.join(output, "tls-roots.inc"));}
  if (name === "gpu-dispatch.cpp" || name === "gpu-forwards.cpp") {files.push(path.join(output, "gpu-table.h"));}
  if (name === "nvidia-metadata.cpp") {files.push(path.join(output, "nvml", "nvml.h"));}
  if (name === "cuda-images.cpp") {files.push(path.join(output, "cuda-images.inc"));}
  if (name === "main.cpp") {files.push(path.join(output, "mom-jit-cache-tag.h"));}
  return files.filter(file => fs.existsSync(file));
}

/** Match the environment used by run() while making inherited AOT test controls inert. */
function linuxCompileEnvironment() {
  const environment = {...process.env};
  for (const name of ["MOM_AOT_PROCESS_EXIT", "MOM_AOT_RUN_CONTROLLER", "MOM_AOT_CHILD_TASK", "MOM_AOT_BULK",
    "MOM_AOT_REJECTION", "MOM_AOT_PROGPOW", "MOM_AOT_FILE_FAULT", "MOM_AOT_FILE_FAULT_PATH",
    "MOM_AOT_CPU_CORE", "MOM_AOT_GPU_INPUTS", "MOM_AOT_IO_SDK", "MOM_AOT_NVML_HEADER", "MOM_AOT_OPENSSL_ROOT",
    "MOM_AOT_BUILD_ONLY"]) {delete environment[name];}
  environment["MOM_TEST_NO_POOL_NETWORK"] = "1";
  environment["NODE_TLS_REJECT_UNAUTHORIZED"] = "1";
  return environment;
}

/** Keep compiler diagnostics bounded in thrown errors while the complete command log remains on disk.
 * @param {string | null | undefined} stdout @param {string | null | undefined} stderr */
function boundedOutputExcerpt(stdout, stderr) {
  const limit = 16 * 1024, bytes = Buffer.byteLength(stdout || "") + Buffer.byteLength(stderr || ""),
    captured = Buffer.from((stdout || "").slice(-limit) + (stderr || "").slice(-limit)), start = Math.max(0, captured.length - limit);
  return captured.length ?
    `\n${bytes > limit ? "Last " + limit + " bytes (earlier output omitted)" : "Output"}:\n${captured.subarray(start).toString("utf8")}` : "";
}

/** Compile queued application objects from a fresh output through the shared content cache. */
function flushLinuxObjectBatch(batch = activeObjectBatch, compile = compileNativeObjects) {
  if (!batch) {return null;}
  if (batch === activeObjectBatch) {activeObjectBatch = null;}
  const {commands, output, repo, namespace, includeRoots} = batch;
  const result = compile({repo, output, namespace, commands,
    environment: linuxCompileEnvironment(), jobs: 1, includeRoots: [...includeRoots], finalize: true,
    liveObjects: commands.map(command => command.object)});
  if (!Array.isArray(result.objects) || result.objects.length !== commands.length) {throw new Error("Native object cache returned an incomplete Linux batch");}
  // Ninja may invalidate earlier successful edges too; retain every log before reporting a later failure.
  const failures = /** @type {Array<{object: (typeof result.objects)[number], command: (typeof commands)[number]}>} */ ([]);
  const objects = result.objects.map((object, index) => {
    const command = commands[index];
    if (!command || path.resolve(object.object) !== path.resolve(command.object) || path.resolve(object.depfile) !== path.resolve(command.depfile)) {
      throw new Error("Native object cache returned an unexpected Linux output path");
    }
    fs.writeFileSync(command.log, String(object.stdout || "") + String(object.stderr || ""), {flag: "wx"});
    if (object.status !== 0) {failures.push({object, command});}
    return {source: command.source, log: command.log, command: command.command, args: command.args, object: object.object, depfile: object.depfile,
      status: object.status, cacheHit: object.cacheHit, objectSha256: object.objectSha256, depfileSha256: object.depfileSha256};
  });
  if (failures.length) {
    const diagnostics = failures.map(({object, command}) => `${command.log} (exit ${object.status}, signal ${object.signal || "none"}` +
      `${object.error ? ", error " + object.error : ""})${boundedOutputExcerpt(object.stdout, object.stderr)}\nFull log: ${command.log}`).join("\n");
    const first = failures[0];
    throw new Error(`Native Linux object compile failed (exit ${first?.object.status}, signal ${first?.object.signal || "none"}): ${first?.command.log}` +
      boundedOutputExcerpt(diagnostics, ""), {cause: first?.object.error});
  }
  const summary = {namespace, cacheRoot: result.cacheRoot, ninja: result.ninja, cacheHits: result.cacheHits,
    cacheMisses: result.cacheMisses, objects};
  fs.writeFileSync(path.join(output, "object-cache.json"), JSON.stringify(summary, null, 2) + "\n", {flag: "wx"});
  return summary;
}

/** Optional metadata uses the same reviewed NVML header as the static CUDA adapter. Freeze
 * verified bytes before compilation; production packages no SDK, only uses the system driver.
 * @param {unknown} file */
function nvmlMetadataInput(file) {
  if (file === undefined) {return null;}
  if (typeof file !== "string" || !path.isAbsolute(file) || Array.from(file).some((char) => char.charCodeAt(0) < 32)) {throw new Error("Select an absolute NVML metadata header");}
  const stat = fs.statSync(file);
  if (!stat.isFile() || stat.size < 1 || stat.size > (2 << 20)) {throw new Error("NVML metadata header must be a bounded regular file");}
  const data = fs.readFileSync(file), sha256 = createHash("sha256").update(data).digest("hex");
  if (sha256 !== "2ec2431ac69cf205ecf6cd2bf24d0402997975505d574cee5aec6a85a4c520c9") {throw new Error("NVML metadata header does not match the reviewed 12.6.77 input");}
  return {source: file, sha256, data};
}

/** @param {string} command @param {string[]} args @param {string} output @param {number} [expectedStatus]
 * @param {NodeJS.ProcessEnv} [extraEnv] @param {string} [cwd] @param {typeof spawnSync} [spawn] */
function run(command, args, output, expectedStatus = 0, extraEnv = {}, cwd = process.cwd(), spawn = spawnSync) {
  if (activeObjectBatch && args.includes("-c")) {
    const sourceIndex = args.indexOf("-c"), outputIndex = args.indexOf("-o");
    const source = args[sourceIndex + 1], object = args[outputIndex + 1];
    if (!source || outputIndex < 0 || !object) {throw new Error("Incomplete queued Linux object compile");}
    const depfile = object + ".d";
    const compileArgs = [...args.slice(0, outputIndex), "-MD", "-MF", depfile, ...args.slice(outputIndex)];
    const roots = linuxObjectIncludeRoots(args, source, activeObjectBatch.output);
    for (const root of roots) {activeObjectBatch.includeRoots.add(root);}
    activeObjectBatch.commands.push({command, args: compileArgs, source, object, depfile, timeout: 120000,
      generatedInputs: linuxGeneratedCompileInputs(source, activeObjectBatch.output), log: output});
    return "";
  }
  const result = spawn(command, args, {cwd, encoding: "utf8", timeout: 120000, maxBuffer: 8 << 20,
    env: {...process.env, MOM_AOT_PROCESS_EXIT: undefined, MOM_AOT_RUN_CONTROLLER: undefined,
      MOM_AOT_CHILD_TASK: undefined, MOM_AOT_BULK: undefined, MOM_AOT_REJECTION: undefined,
      MOM_AOT_PROGPOW: undefined,
      MOM_AOT_FILE_FAULT: undefined, MOM_AOT_FILE_FAULT_PATH: undefined, ...extraEnv,
      MOM_TEST_NO_POOL_NETWORK: "1", NODE_TLS_REJECT_UNAUTHORIZED: "1"}});
  fs.writeFileSync(output, (result.stdout || "") + (result.stderr || ""));
  if (result.error || result.signal || result.status !== expectedStatus) {
    const excerpt = boundedOutputExcerpt(result.stdout, result.stderr);
    throw new Error(`${command} failed (exit ${result.status}, signal ${result.signal}, expected ${expectedStatus}); see ${output}${excerpt}`, {cause: result.error});
  }
  return result.stdout;
}

/** Large static compiler archives have a large demangled symbol table. Preserve it directly
 * rather than hitting a child stdout buffer cap or treating a truncated audit as a pass.
 * @param {string} executable @param {string} output @param {string[]} [requiredSymbols] */
function auditSymbols(executable, output, requiredSymbols = []) {
  const fd = fs.openSync(output, "wx");
  try {
    const result = spawnSync("nm", ["-gC", executable], {stdio: ["ignore", fd, fd], timeout: 120000});
    if (result.error || result.signal || result.status !== 0) {throw new Error(`Symbol audit failed: ${output}`, {cause: result.error});}
  } finally {fs.closeSync(fd);}
  auditNativeSymbolFile(output, requiredSymbols);
}

/** The application always executes maintained mom.js; test-entry switches exist only in probes.
 * @param {string} repo @param {boolean} [application] */
function controllerEntry(repo, application = false) {
  if (application) {return `require(${JSON.stringify(path.join(repo, "mom.js"))});\n`;}
  return `if (process.env.MOM_AOT_CHILD_TASK) {require(${JSON.stringify(path.join(repo, "tests/reference/native-child-task.js"))});}` +
    ` else if (process.env.MOM_AOT_FILE_FAULT) {require(${JSON.stringify(path.join(repo, "tests/reference/native-filesystem-fault.js"))});}` +
    ` else if (process.env.MOM_AOT_REJECTION) {require(${JSON.stringify(path.join(repo, "tests/reference/native-rejections.js"))});}` +
    ` else if (process.env.MOM_AOT_PROGPOW === "1") {require(${JSON.stringify(path.join(repo, "tests/reference/native-progpow.js"))}).probe()` +
    ".then((value) => console.log(JSON.stringify(value))).catch((error) => setTimeout(() => {throw error;}, 0));}" +
    ` else if (process.env.MOM_AOT_BULK) {require(${JSON.stringify(path.join(repo, "tests/reference/native-bulk.js"))})(process.env.MOM_AOT_BULK)` +
    ".then((value) => console.log(JSON.stringify(value))).catch((error) => setTimeout(() => {throw error;}, 0));}" +
    ` else if (process.env.MOM_AOT_FAILURE === "spawn") {require(${JSON.stringify(path.join(repo, "tests/reference/native-spawn-failure.js"))})();}` +
    ` else if (["childTeardown", "promiseTeardown"].includes(process.env.MOM_AOT_FAILURE)) {require(${JSON.stringify(path.join(repo, "tests/reference/native-child-failure.js"))})(process.env.MOM_AOT_FAILURE === "promiseTeardown");}` +
    ` else if (process.env.MOM_AOT_RUN_CONTROLLER === "1") {require(${JSON.stringify(path.join(repo, "mom.js"))});}` +
    ` else {require(${JSON.stringify(path.join(repo, "tests/reference/native-controller.js"))});}\n`;
}

/** Derive the complete factory set from maintained dispatch declarations, not a parallel list.
 * A real application must link every supported graph and must not inherit a probe's main wrapper.
 * @param {string} source @param {{defaultCompiler?: unknown, flags: string[], links: string[]}} inputs */
function nativeApplicationGpu(source, inputs) {
  const factories = [...source.matchAll(/^extern "C" const mom::runtime::GpuTable\* (mom_native_gpu_[a-z0-9_]+)\(\);$/gm)].map((match) => match[1] || "");
  if (!factories.length || factories.length > 64 || new Set(factories).size !== factories.length ||
      factories.length !== (source.match(/extern "C" const mom::runtime::GpuTable/g) || []).length) {
    throw new Error("Unrecognized maintained native GPU factory declarations");
  }
  const keys = factories.map((name) => name.replace("mom_native_gpu_", "").replaceAll("_", "-"));
  if (typeof inputs.defaultCompiler !== "string" || !keys.includes(inputs.defaultCompiler) ||
      [...inputs.flags, ...inputs.links].some((flag) => flag.includes("--wrap=main"))) {
    throw new Error("Native application needs an explicit linked default compiler and no main wrapper");
  }
  return {factories, definitions: ["-DMOM_NATIVE_GPU_DEFAULT=" + JSON.stringify(inputs.defaultCompiler),
    ...factories.map((name) => "-DMOM_NATIVE_GPU_" + name.replace("mom_native_gpu_", "").toUpperCase())],
  // Driver-only loading is application policy, never an optional SDK manifest setting.
  linkFlags: ["-Wl,--wrap=dlopen", "-Wl,--wrap=dlerror", "-Wl,--wrap=dlsym,--wrap=dlvsym", ...factories.map(name => "-Wl,--undefined=" + name)]};
}

/** Static SYCL runtimes must initialize before regular-priority image registrations. Keep
 * their original library closure before the image-bearing objects; a linker group resolves
 * circular references without moving registrations ahead of their owning runtime's lifetime.
 * @param {{flags: string[], libraries: string[], links: string[], objects: {object: string}[], sdkObjects?: {object: string}[]}} inputs */
function nativeGpuLinkArgs(inputs) {
  // The full Level Zero loader has constructor-only context objects. A normal archive
  // search discards them and leaves zeInit with null state, despite a successful link.
  const links = [...inputs.libraries, ...inputs.links].flatMap(flag => path.basename(flag) === "libze_loader.a" ?
    ["-Wl,--push-state,--whole-archive", flag, "-Wl,--pop-state"] : [flag]);
  return [...inputs.flags.filter((flag) => flag.startsWith("--ld-path=") || flag.startsWith("-Wl,")),
    "-Wl,--start-group", ...links,
    ...(inputs.sdkObjects || []).map((item) => item.object),
    ...inputs.objects.map((item) => item.object), "-Wl,--end-group"];
}

/** Retain a path-bound map only for the final application link. Pass the map path as one literal
 * linker argument so commas or spaces in the output root are not parsed by the compiler driver.
 * @param {string} output @param {string[]} args @param {boolean} application */
function nativeApplicationLinkPlan(output, args, application) {
  if (!application) {return {args, map: null};}
  const map = path.join(output, "controller-native.map");
  return {args: ["-Xlinker", "-Map=" + map, ...args], map};
}

/** Both CUDA graphs must bind the exact image files selected for the shared application table.
 * @param {unknown} receipt @param {Record<string, string>} inputs @param {string} profile */
function verifyCudaImageReceiptInputs(receipt, inputs, profile) {
  if (!receipt || typeof receipt !== "object" || Array.isArray(receipt)) {
    throw new Error(`Native ${profile} GPU receipt does not bind the admitted CUDA image bundle`);
  }
  const record = /** @type {{sdkInputs?: Record<string, unknown>, cudaImages?: unknown}} */ (receipt), files = record.sdkInputs;
  const include = Object.keys(inputs).find(file => file.endsWith(".inc"));
  if (!include || Object.keys(inputs).length !== 2 || !Object.hasOwn(inputs, include + ".json") || record.cudaImages !== include ||
      !files || typeof files !== "object" || Array.isArray(files) ||
      Object.entries(inputs).some(([file, sha256]) => files[file] !== sha256)) {
    throw new Error(`Native ${profile} GPU receipt does not bind the admitted CUDA image bundle`);
  }
}

/** Resolve a source-admitted bundle from the final SDK's exact two-file hash claim. The
 * reader is injectable only for unit fixtures; production always uses the maintained reader.
 * @param {string} repo @param {unknown} value
 * @param {(file: string, repo: string) => {files: Record<string, string>}} [read] */
function nativeApplicationCudaImages(repo, value, read = require("./native-cuda-images").cudaImageInputs) {
  if (value === undefined) {return null;}
  if (!value || typeof value !== "object" || Array.isArray(value)) {throw new Error("Native application requires an exact CUDA image hash map");}
  const files = /** @type {Record<string, unknown>} */ (value), names = Object.keys(files);
  const include = names.find(file => path.isAbsolute(file) && file.endsWith(".inc"));
  if (!include || names.length !== 2 || !Object.hasOwn(files, include + ".json") ||
      Object.values(files).some(hash => typeof hash !== "string" || !/^[0-9a-f]{64}$/.test(hash))) {
    throw new Error("Native application requires the CUDA include and adjacent sidecar hashes");
  }
  const admitted = read(include, repo);
  if (!isDeepStrictEqual(files, admitted.files) || Object.entries(admitted.files).some(([file, hash]) => fileHash(file) !== hash)) {
    throw new Error("Native application CUDA image files differ from source admission");
  }
  return {include, files: admitted.files};
}

/** Hash every absolute linked file while replacing transport-specific library search paths.
 * @param {string[]} links @returns {{files:string[], linkPlan:string[]}} */
function nativeApplicationLinkCacheInputs(links) {
  if (!Array.isArray(links) || links.length > 32768 || links.some(token => typeof token !== "string" || !token.length || token.length > 4096 ||
      Array.from(token).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127))) {
    throw new Error("Invalid native JIT link closure");
  }
  const files = [], linkPlan = [];
  let searchIndex = 0;
  /** @param {string | undefined} directory */
  const admitSearchDirectory = directory => {
    if (typeof directory !== "string" || !path.isAbsolute(directory) || path.resolve(directory) !== directory) {
      throw new Error("Native application link search path must be an absolute canonical directory");
    }
    let stat;
    try {stat = fs.lstatSync(directory);} catch {throw new Error("Native application link search directory is missing: " + directory);}
    if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(directory) !== directory) {
      throw new Error("Native application link search path must be a canonical directory: " + directory);
    }
  };
  for (let index = 0; index < links.length; index++) {
    const token = links[index] || "";
    if (token.startsWith("-L")) {
      const directory = token === "-L" ? links[++index] : token.slice(2);
      admitSearchDirectory(directory);
      linkPlan.push("-L", "@link-search/" + searchIndex++);
    } else if (path.isAbsolute(token)) {
      linkPlan.push("@link/" + files.length);
      files.push(token);
    } else {linkPlan.push(token);}
  }
  return {files, linkPlan};
}

/** A final application may consume only source-bound Core and factory objects. Older proof
 * receipts deliberately need a rebuild, not a newly stamped digest of unproven cached code.
 * @param {string} repo @param {string} cpuCore
 * @param {{objects: {object: string}[], receipts?: unknown, sdkObjects?: unknown, libraries?: unknown, links?: unknown, cudaImageInputs?: unknown}} inputs @param {string[]} factories
 * @param {Parameters<typeof nativeApplicationCudaImages>[2]} [readImages] */
function nativeApplicationInputs(repo, cpuCore, inputs, factories, readImages) {
  const images = nativeApplicationCudaImages(repo, inputs.cudaImageInputs, readImages), imageFactories = new Set();
  const core = readBuildJson(path.join(path.dirname(cpuCore), "inputs.json"));
  verifySourceInputs(core?.sourceState, nativeSourceInputs(repo, "cpu"), "Native Core");
  if (core.gpu !== true || core.objectSha256 !== fileHash(cpuCore)) {throw new Error("Native Core artifact differs from its source receipt");}
  // The final link is also an admission boundary: direct callers must not bypass the
  // orchestrator's portable-cache policy with a valid but host-tuned Core build.
  if (core.portable !== true) {throw new Error("Native application requires a portable Core; rebuild from a completed portable Engineering cache");}
  if (!Array.isArray(inputs.receipts) || inputs.receipts.length !== factories.length ||
      inputs.receipts.some(file => typeof file !== "string" || !path.isAbsolute(file))) {
    throw new Error("Native application requires source receipts for every GPU factory");
  }
  const current = nativeSourceInputs(repo, "gpu"), remaining = new Set(factories), objects = new Set(inputs.objects.map(row => row.object));
  if (objects.size !== inputs.objects.length) {throw new Error("Native GPU objects must be unique source-bound factories");}
  const hcfObjects = /** @type {string[]} */ ([]);
  const graphs = inputs.receipts.map(file => {
    const receipt = readBuildJson(file);
    if (!receipt || typeof receipt.factory !== "string" || !remaining.delete(receipt.factory) || !objects.delete(receipt.object)) {
      throw new Error("Native GPU factory receipt is duplicated, unknown or not a selected object");
    }
    verifySourceInputs(receipt.sourceState, current, "Native GPU");
    if (receipt.objectSha256 !== fileHash(receipt.object)) {throw new Error("Native GPU object differs from its source receipt");}
    if (receipt.factory.startsWith("mom_native_gpu_acpp_")) {hcfObjects.push(receipt.object);}
    if (receipt.cudaImages !== undefined && (!images || !["mom_native_gpu_dpcpp", "mom_native_gpu_acpp_cuda"].includes(receipt.factory))) {
      throw new Error("Image-mode CUDA factory requires its matching shared application image bundle");
    }
    if (images && ["mom_native_gpu_dpcpp", "mom_native_gpu_acpp_cuda"].includes(receipt.factory)) {
      verifyCudaImageReceiptInputs(receipt, images.files, receipt.factory);
      imageFactories.add(receipt.factory);
    }
    const comgr = receipt.factory === "mom_native_gpu_acpp_hip" ? nativeComgrLinkInputs(receipt, inputs) : undefined;
    return {factory: receipt.factory, objectSha256: receipt.objectSha256, ...(comgr ? {comgr} : {})};
  });
  if (hcfObjects.length) {assertUniqueHcfIds(hcfObjects);}
  if (images && imageFactories.size !== 2) {throw new Error("Shared CUDA images require both source-bound CUDA factories");}
  if (objects.size) {throw new Error("Native application objects must contain only source-bound GPU factories; runtime helpers are compiled automatically");}
  // SDK objects have a distinct role from maintained application sources. Hash them across
  // the link too; never let an extra cached MoM helper hide among admitted factory objects.
  if (!Array.isArray(inputs.sdkObjects) || inputs.sdkObjects.length > 64) {throw new Error("Native application requires an explicit bounded SDK object list");}
  const sdkPaths = new Set();
  const sdk = inputs.sdkObjects.map(row => {
    if (!row || typeof row.object !== "string" || row.object.length > 4096 || !path.isAbsolute(row.object) ||
        Array.from(row.object).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127) ||
        sdkPaths.has(row.object) || inputs.objects.some(item => item.object === row.object)) {throw new Error("Invalid or duplicate native SDK object");}
    sdkPaths.add(row.object);
    return {object: row.object, sha256: fileHash(row.object)};
  });
  const libraries = inputs.libraries === undefined ? [] : inputs.libraries, links = inputs.links === undefined ? [] : inputs.links;
  if (!Array.isArray(libraries) || libraries.some(file => typeof file !== "string" || !path.isAbsolute(file)) ||
      !Array.isArray(links) || links.some(file => typeof file !== "string")) {throw new Error("Invalid native JIT link closure");}
  const linkInputs = nativeApplicationLinkCacheInputs(links);
  const jitCacheTag = admittedJitCacheTag(repo, "linux-x64", {
    library: libraries, link: linkInputs.files,
    sdk: sdk.map(row => row.object), gpu: inputs.objects.map(row => row.object),
  }, linkInputs.linkPlan);
  return {core: core.objectSha256, portableCore: true, graphs, sdk, jitCacheTag, ...(images ? {cudaImages: images} : {})};
}

/** A verified producer at GPU compilation is insufficient if the final SDK silently links
 * another COMGR. Bind that receipt and the graph's consumed archive to exactly one final
 * archive token, revalidating the source producer before AND after the application link.
 * @param {unknown} value @param {{libraries?: unknown, links?: unknown}} sdk
 * @param {typeof comgrSdkInputs} [read] */
function nativeComgrLinkInputs(value, sdk, read = comgrSdkInputs) {
  const record = /** @type {{comgr?: {receipt?: unknown, receiptSha256?: unknown, archive?: unknown, sha256?: unknown}, sdkInputs?: Record<string, string>}} */ (value);
  const claim = record?.comgr;
  if (!claim || typeof claim.receipt !== "string" || !path.isAbsolute(claim.receipt) ||
      typeof claim.archive !== "string" || !path.isAbsolute(claim.archive) ||
      typeof claim.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(claim.sha256) ||
      typeof claim.receiptSha256 !== "string" || !/^[0-9a-f]{64}$/.test(claim.receiptSha256) ||
      ![sdk.libraries, sdk.links].every(values => Array.isArray(values) && values.every(token => typeof token === "string"))) {
    throw new Error("Native HIP graph requires a complete COMGR producer and final SDK link claim");
  }
  const component = read(claim.receipt);
  if (component.archive !== claim.archive || component.files[claim.archive] !== claim.sha256 ||
      component.files[claim.receipt] !== claim.receiptSha256 || record.sdkInputs?.[claim.archive] !== claim.sha256 ||
      record.sdkInputs?.[claim.receipt] !== claim.receiptSha256) {
    throw new Error("Native HIP COMGR provenance differs from its consumed SDK inputs");
  }
  const tokens = [.../** @type {string[]} */ (sdk.libraries), .../** @type {string[]} */ (sdk.links)
    .flatMap(flag => flag.startsWith("-Wl,") ? flag.slice(4).split(",") : [flag])];
  const archives = tokens.filter(token => path.basename(token) === "libamd_comgr.a");
  if (archives.length !== 1 || archives[0] !== claim.archive) {throw new Error("Final native SDK must link exactly the source-receipted COMGR archive");}
  return {...claim};
}

/** Controller-only build inputs never invalidate Engineering/Core/kernel caches. Freeze both
 * maintained runtime code and its compilation recipe so a mid-build edit cannot be stamped
 * as current merely because the final JavaScript bundle and GPU objects still match.
 * @param {string} repo */
function nativeApplicationBuildInputs(repo) {
  return sourceInputs(repo, ["runtime"], ["package.json", "package-lock.json", "scripts/build-native-application.js", "scripts/native-bundle.js",
    "scripts/native-resources.js", "scripts/native-gpu-table.js", "scripts/native-audit.js", "scripts/build-source-inputs.js",
    "scripts/build-native-io.js", "scripts/build-environment.js", "scripts/native-object-cache.js", "scripts/native-jit-cache-tag.js", "scripts/native-hcf.js",
    "scripts/native-linux-sysroot.js", "sycl/cuda-images.cpp", "sycl/cuda-images.h"]);
}

/** Inspect actual outputs outside either runtime; text parity cannot prove file modes.
 * @param {string} directory */
function filesystemReport(directory) {
  return ["binary", "empty", "private-é-💎.json"].map((name) => {
    const file = path.join(directory, name), stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.size > 1024 * 1024) {throw new Error("Unexpected native filesystem output");}
    return {name, mode: stat.mode & 0o7777, bytes: fs.readFileSync(file).toString("hex")};
  });
}

/** Compile actual source modules and compare their native transcript with direct Node execution. */
async function main() {
  const args = process.argv.slice(2), application = args.at(-1) === "--app";
  if (application) {args.pop();}
  const [sourceArg, buildArg, outputArg] = args;
  if (process.platform !== "linux" || !sourceArg || !buildArg || args.length > 3 || args.some((arg) => arg.startsWith("--"))) {
    throw new Error("Usage (Linux): node scripts/build-native-application.js <Hermes source> <Hermes build> [output parent] [--app]");
  }
  if (application && (!process.env["MOM_AOT_CPU_CORE"] || !process.env["MOM_AOT_GPU_INPUTS"] || !process.env["MOM_AOT_IO_SDK"] || process.env["MOM_AOT_BUILD_ONLY"] !== "1")) {
    throw new Error("Application build requires the full CPU/GPU/I/O SDK inputs and MOM_AOT_BUILD_ONLY=1");
  }
  const ioRoot = process.env["MOM_AOT_IO_SDK"], io = ioRoot ? nativeIoInputs(ioRoot) : null;
  const ioFlags = io ? ["-isystem", io.include] : [];
  const nvml = nvmlMetadataInput(process.env["MOM_AOT_NVML_HEADER"]);
  const source = path.resolve(sourceArg);
  const build = path.resolve(buildArg);
  const output = fs.mkdtempSync(path.join(path.resolve(outputArg || os.tmpdir()), "mom-aot-"));
  if (nvml) {
    fs.mkdirSync(path.join(output, "nvml"));
    fs.writeFileSync(path.join(output, "nvml/nvml.h"), nvml.data, {flag: "wx"});
    fs.writeFileSync(path.join(output, "nvml-metadata-input.json"), JSON.stringify({source: nvml.source, sha256: nvml.sha256}) + "\n", {flag: "wx"});
  }
  console.log(`AOT feasibility evidence: ${output}`);
  if (run("git", ["-C", source, "rev-parse", "HEAD"], path.join(output, "revision.log")).trim() !== revision) {
    throw new Error(`Expected Static Hermes revision ${revision}`);
  }
  const changed = run("git", ["-C", source, "diff", "--name-only", "HEAD"], path.join(output, "toolchain-changed.log")).trim();
  const promiseSource = "lib/InternalJavaScript/01-Promise.js";
  if (changed !== promiseSource || createHash("sha256").update(fs.readFileSync(path.join(source, promiseSource))).digest("hex") !==
      "071f4322f9dba4fa06992065bc5f1eb9b11e49c9b1b6336d31f377b3e9801184") {
    throw new Error("Apply only runtime/hermes-host-rejections.patch to the pinned Hermes source and rebuild its static runtime");
  }
  run("git", ["-C", source, "diff", "HEAD"], path.join(output, "toolchain-diff.log"));
  const cache = fs.readFileSync(path.join(build, "CMakeCache.txt"), "utf8");
  if (!/^HERMESVM_INTERNAL_JAVASCRIPT_NATIVE:BOOL=ON$/m.test(cache)) {
    throw new Error("Rebuild Hermes with HERMESVM_INTERNAL_JAVASCRIPT_NATIVE=ON (no interpreted builtin JavaScript)");
  }
  const repo = path.resolve(__dirname, "..");
  const policy = fs.readFileSync(path.join(repo, "GPU-CONFIG.md"));
  fs.writeFileSync(path.join(output, "app-resources.inc"), nativeResources([{name: "GPU-CONFIG.md", data: policy}]) +
    `static constexpr std::string_view kEmbeddedPolicyPath = ${JSON.stringify(path.posix.join(nativeApplicationRoot("linux"), "GPU-CONFIG.md"))};\n`);
  const appResources = [{name: "GPU-CONFIG.md", bytes: policy.length, sha256: createHash("sha256").update(policy).digest("hex")}];
  const cpuCore = process.env["MOM_AOT_CPU_CORE"];
  if (cpuCore && (!path.isAbsolute(cpuCore) || !fs.existsSync(cpuCore))) {throw new Error("Missing native CPU proof object");}
  const gpuManifest = process.env["MOM_AOT_GPU_INPUTS"];
  if (gpuManifest && (!cpuCore || process.env["MOM_AOT_BUILD_ONLY"] !== "1")) {
    throw new Error("GPU Core integration requires a stub-free Core and MOM_AOT_BUILD_ONLY=1; verify ownership before execution");
  }
  if (gpuManifest && cpuCore) {
    const provenance = readBuildJson(path.join(path.dirname(cpuCore), "inputs.json"));
    if (provenance.gpu !== true || !Array.isArray(provenance.objects) ||
        provenance.objects.some((/** @type {string} */ file) => file.includes("gpu-unavailable"))) {
      throw new Error("GPU controller cannot include CPU-proof stubs; build Core with --gpu");
    }
  }
  const gpuInputs = gpuManifest ? readBuildJson(gpuManifest) : null;
  if (gpuInputs && (gpuInputs.version !== 1 || !Array.isArray(gpuInputs.objects) || !gpuInputs.objects.length ||
      ![gpuInputs.flags, gpuInputs.libraries, gpuInputs.links].every((values) => Array.isArray(values) && values.every((value) => typeof value === "string")) ||
      gpuInputs.objects.some((/** @type {{object: string}} */ item) => !path.isAbsolute(item.object) || !fs.existsSync(item.object)))) {
    throw new Error("Missing maintained GPU build inputs");
  }
  const applicationGpu = application ? nativeApplicationGpu(fs.readFileSync(path.join(repo, "runtime/gpu-dispatch.cpp"), "utf8"), gpuInputs) : null;
  const sourceFreshness = applicationGpu ? nativeApplicationInputs(repo, cpuCore || "", gpuInputs, applicationGpu.factories) : null;
  if (sourceFreshness) {fs.writeFileSync(path.join(output, "mom-jit-cache-tag.h"), nativeJitCacheHeader(sourceFreshness.jitCacheTag), {flag: "wx"});}
  const verifyCopiedCudaImages = () => {
    if (sourceFreshness && fs.readFileSync(path.join(output, "mom-jit-cache-tag.h"), "utf8") !== nativeJitCacheHeader(sourceFreshness.jitCacheTag)) {
      throw new Error("Generated native JIT cache identity changed during application compilation");
    }
    const images = sourceFreshness?.cudaImages;
    if (images && fileHash(path.join(output, "cuda-images.inc")) !== images.files[images.include]) {
      throw new Error("Shared CUDA image copy changed during application compilation");
    }
  };
  const runtimeState = application ? nativeApplicationBuildInputs(repo) : null;
  /** @type {NodeJS.ProcessEnv} */
  const oracleEnv = {};
  if (cpuCore && !gpuInputs) {
    const addon = process.env["MOM_AOT_CORE_ADDON"];
    if (!addon || !path.isAbsolute(addon) || !fs.existsSync(addon)) {throw new Error("Select an existing Node CPU oracle with MOM_AOT_CORE_ADDON");}
    // Development builds temporarily park build/. Pin this test input before compilation so a
    // concurrent cache move cannot replace/disappear beneath the child Node oracle.
    const pinnedAddon = path.join(output, "oracle-addon.node");
    fs.copyFileSync(addon, pinnedAddon);
    oracleEnv["MOM_AOT_CORE_ADDON"] = pinnedAddon;
  }
  const opensslRoot = path.resolve(process.env["MOM_AOT_OPENSSL_ROOT"] || "build/cache/native-openssl-3.5.9");
  for (const archive of ["libssl.a", "libcrypto.a"]) {
    if (!fs.existsSync(path.join(opensslRoot, "lib", archive))) {
      throw new Error(`Missing pinned static OpenSSL; set MOM_AOT_OPENSSL_ROOT (${archive})`);
    }
  }
  const openssl = path.join(opensslRoot, "bin", "openssl");
  if (!/^OpenSSL 3\.5\.9\b/.test(run(openssl, ["version"], path.join(output, "openssl-version.log")))) {
    throw new Error("Native TLS probe requires pinned OpenSSL 3.5.9");
  }
  const files = path.join(output, "filesystem");
  fs.mkdirSync(files);
  for (const name of ["node-writes", "native-writes"]) {fs.mkdirSync(path.join(files, name));}
  fs.writeFileSync(path.join(files, "config-é-💎.json"), '{"fixture":"é-💎"}\n');
  fs.writeFileSync(path.join(files, "binary"), Buffer.from([0, 255, 128, 10]));
  fs.writeFileSync(path.join(files, "empty"), "");
  oracleEnv["MOM_AOT_FS_DIRECTORY"] = files;
  const fixture = path.join(repo, "tests", "reference", "native-controller.js");
  const entry = path.join(output, "entry.js");
  fs.writeFileSync(entry, controllerEntry(repo, application));
  const bundle = path.join(output, "controller.js");
  // Exported admission helpers are also used on SDK-free QA hosts; only a build needs the bundler.
  const esbuild = require("esbuild");
  const bundled = esbuild.buildSync(controllerBundleOptions(entry, bundle, false, "linux"));
  fs.writeFileSync(path.join(output, "source-graph.json"), JSON.stringify(bundled.metafile, null, 2) + "\n");
  const cSource = path.join(output, "controller.c");
  const aot = controllerAotCommand(path.join(build, "bin", "shermes"), bundle, cSource);
  run(aot.command, aot.args, path.join(output, "aot.log"), 0, {}, aot.cwd);
  const includes = [path.join(source, "include"), path.join(build, "lib", "config"), ...(io ? [io.include] : [])];
  if (application) {
    const profile = String(gpuInputs?.defaultCompiler || "default").replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 48) || "default";
    activeObjectBatch = {repo, output, namespace: `linux/controller/app-${profile}`, commands: [], includeRoots: new Set()};
  }
  const object = path.join(output, "controller.o");
  // Intel's driver need not inherit a distro compiler's PIE default. All host objects must
  // remain relocatable when the final executable explicitly enables ASLR below.
  run(process.env["CC"] || "cc", ["-std=gnu11", "-O2", "-fPIC", "-DNDEBUG", "-fno-strict-aliasing",
    "-fno-strict-overflow", ...includes.flatMap((dir) => ["-I", dir]),
    "-c", cSource, "-o", object], path.join(output, "compile.log"));
  const host = path.join(output, "console.o");
  const cxx = gpuInputs?.cxx || process.env["CXX"] || "c++";
  run(cxx, ["-std=c++17", "-O2", "-fPIC", "-DNDEBUG", "-Wall", "-Wextra", "-Werror",
    ...(cpuCore ? ["-DMOM_NATIVE_RUNTIME"] : []),
    ...[...includes, path.join(source, "API"), path.join(source, "API", "jsi"),
      path.join(source, "public")].flatMap((dir) => ["-isystem", dir]),
    "-c", path.join(repo, "runtime", "host.cpp"), "-o", host],
  path.join(output, "host.log"));
  const loop = path.join(output, "event-loop.o");
  const processObjects = ["process", "main", "services", "nvidia-metadata"].map((name) => {
    const outputObject = path.join(output, `${name}.o`);
    run(cxx, ["-std=c++17", "-O2", "-fPIC", "-DNDEBUG", "-Wall", "-Wextra", "-Werror",
      ...(name === "services" ? ["-DMOM_EMBEDDED_APP_RESOURCES", "-I", output] : []),
      ...(name === "main" && sourceFreshness ? ["-DMOM_NATIVE_ACPP_CACHE", "-I", output] : []),
      ...(name === "nvidia-metadata" && nvml ? ["-DMOM_NATIVE_NVML_METADATA", "-I", path.join(output, "nvml")] : []),
      ...[...includes, path.join(source, "API"), path.join(source, "API", "jsi"),
        path.join(source, "public"), path.join(opensslRoot, "include")].flatMap((dir) => ["-isystem", dir]),
      "-c", path.join(repo, "runtime", `${name}.cpp`), "-o", outputObject], path.join(output, `${name}.log`));
    return outputObject;
  });
  run(cxx, ["-std=c++17", "-O2", "-fPIC", "-DNDEBUG", "-Wall", "-Wextra", "-Werror", "-isystem", path.join(source, "API", "jsi"),
    "-isystem", path.join(opensslRoot, "include"),
    ...ioFlags,
    "-c", path.join(repo, "runtime", "event-loop.cpp"), "-o", loop], path.join(output, "loop.log"));
  fs.writeFileSync(path.join(output, "tls-roots.inc"),
    `static constexpr char kTlsRoots[] = ${JSON.stringify(tls.rootCertificates.join("\n"))};\n`);
  const crypto = path.join(output, "tls.o");
  run(cxx, ["-std=c++17", "-O2", "-fPIC", "-DNDEBUG", "-Wall", "-Wextra", "-Werror", "-I", output,
    "-isystem", path.join(opensslRoot, "include"),
    "-c", path.join(repo, "runtime", "tls.cpp"), "-o", crypto], path.join(output, "tls.log"));
  const executable = path.join(output, "controller-native");
  const applicationObjects = [];
  if (applicationGpu) {
    const api = fs.readFileSync(path.join(repo, "sycl/lib.h"), "utf8");
    fs.writeFileSync(path.join(output, "gpu-table.h"), gpuTable(api, applicationGpu.factories[0] || "").header);
    fs.writeFileSync(path.join(output, "gpu-forwards.cpp"), gpuForwards(api));
    for (const [name, input] of [["gpu-dispatch", path.join(repo, "runtime/gpu-dispatch.cpp")], ["gpu-forwards", path.join(output, "gpu-forwards.cpp")],
      ["driver-library", path.join(repo, "runtime/driver-library.cpp")], ["compiler-gate", path.join(repo, "runtime/compiler-gate.cpp")]]) {
      if (!name || !input) {throw new Error("Missing native application bridge");}
      const object = path.join(output, `${name}.o`);
      run(cxx, ["-std=c++20", "-O2", "-fPIC", ...applicationGpu.definitions, "-I", repo, "-I", output, "-c", input, "-o", object], path.join(output, `${name}.log`));
      applicationObjects.push(object);
    }
    if (sourceFreshness?.cudaImages) {
      const images = sourceFreshness.cudaImages, include = path.join(output, "cuda-images.inc");
      fs.copyFileSync(images.include, include, fs.constants.COPYFILE_EXCL);
      if (fileHash(include) !== images.files[images.include]) {throw new Error("Shared CUDA image copy changed before application compilation");}
      const object = path.join(output, "cuda-images.o");
      run(cxx, ["-std=c++20", "-O2", "-fPIC", "-ffunction-sections", "-fdata-sections", "-I", output,
        "-c", path.join(repo, "sycl/cuda-images.cpp"), "-o", object], path.join(output, "cuda-images.log"));
      applicationObjects.push(object);
    }
  }
  const objectCache = application ? flushLinuxObjectBatch() : null;
  verifyCopiedCudaImages();
  // No DT_RELR (-z pack-relative-relocs): it adds a GLIBC_ABI_DT_RELR need that glibc < 2.36 refuses to load.
  const linkArgs = ["-pie", "-Wl,--gc-sections", "-static-libstdc++", "-static-libgcc", object, host, loop, crypto, ...processObjects, ...applicationObjects,
    ...(applicationGpu?.linkFlags || []),
    ...(cpuCore ? [cpuCore] : []),
    ...(gpuInputs ? nativeGpuLinkArgs(gpuInputs) : []),
    path.join(build, "lib", "libhermesvmlean_a.a"), path.join(build, "jsi", "libjsi.a"),
    path.join(build, "public", "hermes", "Public", "libhermesPublic.a"),
    path.join(build, "external", "boost", "boost_1_86_0", "libs", "context", "libboost_context.a"),
    path.join(opensslRoot, "lib", "libssl.a"), path.join(opensslRoot, "lib", "libcrypto.a"),
    // librt: shm_open lives there before glibc 2.34; newer glibc keeps it as a compatibility stub.
    ...(io ? io.libraries : ["-Wl,-Bstatic", "-luv", "-lz", "-Wl,-Bdynamic"]), "-lm", "-ldl", "-lpthread", "-lrt"];
  const linkPlan = nativeApplicationLinkPlan(output, linkArgs, application);
  fs.writeFileSync(path.join(output, "link-inputs.json"), JSON.stringify({cxx, args: [...linkPlan.args, "-o", executable],
    ...(linkPlan.map ? {linkMap: linkPlan.map} : {})}, null, 2) + "\n");
  run(cxx, [...linkPlan.args, "-o", executable], path.join(output, "link.log"));
  let linkMapReceipt;
  if (linkPlan.map) {
    const stat = fs.lstatSync(linkPlan.map);
    if (!stat.isFile() || !stat.size) {throw new Error("Native application link did not emit a regular nonempty link map");}
    linkMapReceipt = {path: linkPlan.map, bytes: stat.size, sha256: fileHash(linkPlan.map)};
  }
  verifyCopiedCudaImages();
  if (io && ioRoot && !isDeepStrictEqual(io, nativeIoInputs(ioRoot))) {throw new Error("Native I/O SDK changed during compilation");}
  if (applicationGpu) {
    if (!isDeepStrictEqual(sourceFreshness, nativeApplicationInputs(repo, cpuCore || "", gpuInputs, applicationGpu.factories))) {
      throw new Error("Native Core/GPU artifacts changed during compilation");
    }
    verifySourceInputs(runtimeState, nativeApplicationBuildInputs(repo), "Native controller build");
    const fresh = esbuild.buildSync(controllerBundleOptions(entry, bundle, true, "linux"));
    verifyBundle(fresh.outputFiles?.[0]?.contents || new Uint8Array(), fs.readFileSync(bundle));
    if (fileHash(path.join(repo, "GPU-CONFIG.md")) !== appResources[0]?.sha256) {throw new Error("Embedded GPU policy changed during compilation");}
  }
  if (gpuInputs) {
    const needed = nativeDependencies(run("readelf", ["-d", executable], path.join(output, "dependencies.log")), {cudaDriver: true});
    auditSymbols(executable, path.join(output, "runtime-symbols.log"), applicationGpu ? [...applicationGpu.factories, "__wrap_dlopen",
      ...(sourceFreshness?.cudaImages ? ["mom_cuda_images_select_embedded"] : [])] : []);
    const versions = application ? run("readelf", ["--version-info", executable], path.join(output, "versions.log")) : "",
      glibc = application ? auditNativeGlibc(versions, linuxGlibcBaseline) : null;
    if (application) {auditNativeVersionNeeds(versions);}
    fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({revision, needed, bytes: fs.statSync(executable).size,
      gpuManifest, cpuCore, appResources, nvmlMetadataSha256: nvml?.sha256 || null,
      application, entrypoint: application ? "mom.js" : "probe", gpuFactories: applicationGpu?.factories || [], glibc, sourceFreshness,
      ...(linkMapReceipt ? {linkMap: linkMapReceipt} : {}),
      applicationInputs: runtimeState, ioSdk: io, objectCache,
      buildOnly: true, scope: "Original AOT controller + maintained Core/GPU sources; NOT executed or qualified"}, null, 2) + "\n");
    console.log("Native GPU controller built; NOT executed. Verify fresh device ownership before a bounded run.");
    return;
  }
  const rejectionModes = ["error", "primitive", "async", "then", "timer", "caught", "nextTick", "microtask", "tickFromMicrotask",
    "awaitCaught", "chain", "recovered", "lateHandled", "handlerThrows"];
  for (const mode of rejectionModes) {
    const fails = ["error", "primitive", "async", "then", "timer", "handlerThrows"].includes(mode);
    const oracle = run(process.execPath, [path.join(repo, "tests/reference/native-rejections.js")],
      path.join(output, `rejection-${mode}-node.log`), fails ? 1 : 0, {MOM_AOT_REJECTION: mode});
    const actual = run(executable, [], path.join(output, `rejection-${mode}-native.log`), fails ? 1 : 0, {MOM_AOT_REJECTION: mode});
    const marker = mode === "handlerThrows" ? "native-rejection-handler-fixture" : "native-rejection-fixture";
    if (!fails && actual !== oracle || fails && !fs.readFileSync(path.join(output, `rejection-${mode}-native.log`), "utf8").includes(marker)) {
      throw new Error(`Native Promise rejection behavior differs (${mode})`);
    }
  }
  for (const mode of cpuCore ? ["bulk", "burst", "burstClosed"] : ["bulk"]) {
    const bulkOracle = run(process.execPath, ["-e", `require(${JSON.stringify(path.join(repo, "tests/reference/native-bulk.js"))})(${JSON.stringify(mode)}).then((value) => console.log(JSON.stringify(value))).catch((error) => {console.error(error); process.exitCode = 1;});`],
      path.join(output, `${mode}-node.log`), 0, oracleEnv);
    const bulkNative = run(executable, [], path.join(output, `${mode}-native.log`), 0, {...oracleEnv, MOM_AOT_BULK: mode});
    const actual = JSON.parse(bulkNative), expected = JSON.parse(bulkOracle);
    const peakQueued = actual.peakQueued;
    delete actual.peakQueued; delete expected.peakQueued;
    // A deliberately broken reader discards a scheduling-dependent partial prefix. Compare its
    // failure/cleanup behavior, never mistake those discarded bytes for a valid delivery result.
    if (mode === "burstClosed") {delete actual.size; delete expected.size; delete actual.hash; delete expected.hash;}
    const trace = mode === "burstClosed" ? "ready,timer,broken,callback" : "ready,timer,callback";
    if (JSON.stringify(actual) !== JSON.stringify(expected) || actual.trace.join(",") !== trace ||
        (mode !== "bulk" && (peakQueued < 16 * 1024 * 1024 || peakQueued > 32 * 1024 * 1024))) {
      throw new Error(`Native ${mode} output/parity differs; see ${mode}-{node,native}.log`);
    }
  }
  const faultObject = path.join(output, "write-failure.o");
  const faultExecutable = path.join(output, "controller-write-failure");
  run(cxx, ["-std=c++17", "-O2", "-fPIC", "-Wall", "-Wextra", "-Werror", "-c",
    ...ioFlags, path.join(repo, "tests", "reference", "native-write-failure.cpp"), "-o", faultObject], path.join(output, "fault-compile.log"));
  run(cxx, [...linkArgs, faultObject, "-Wl,--wrap=uv_write", "-Wl,--wrap=uv_spawn", "-o", faultExecutable],
    path.join(output, "fault-link.log"));
  const fileFaultObject = path.join(output, "filesystem-faults.o"), fileFaultExecutable = path.join(output, "controller-filesystem-faults");
  run(cxx, ["-std=c++17", "-O2", "-fPIC", "-Wall", "-Wextra", "-Werror", "-c",
    ...ioFlags, path.join(repo, "tests/native/filesystem-faults.cpp"), "-o", fileFaultObject], path.join(output, "filesystem-fault-compile.log"));
  run(cxx, [...linkArgs, fileFaultObject, ...["open", "write", "close"].map((name) => `-Wl,--wrap=uv_fs_${name}`),
    "-o", fileFaultExecutable], path.join(output, "filesystem-fault-link.log"));
  const fileFaults = [];
  for (const [mode, code, syscall] of [["partial", "", ""], ["zero", "EIO", "write"],
    ["space", "ENOSPC", "write"], ["close", "EIO", "close"], ["open", "EACCES", "open"]]) {
    if (!mode || code === undefined || syscall === undefined) {throw new Error("Incomplete file-fault fixture");}
    const file = path.join(output, `fault-${mode}.json`), log = path.join(output, `filesystem-fault-${mode}.log`);
    fs.writeFileSync(file, "original", {mode: 0o600});
    const text = run(fileFaultExecutable, [], log, 0, {...oracleEnv, MOM_AOT_FILE_FAULT: mode, MOM_AOT_FILE_FAULT_PATH: file});
    const report = JSON.parse(text), diagnostic = fs.readFileSync(log, "utf8");
    const counts = diagnostic.match(/FILESYSTEM_FAULT opened=(\d+) writes=(\d+) closed=(\d+) live=(-?\d+)/);
    const opens = mode === "open" ? 0 : 1;
    if (!counts || Number(counts[1]) !== opens || Number(counts[3]) !== opens || counts[4] !== "-1" ||
        (mode === "partial" && Number(counts[2]) < 2) ||
        (code ? report.error?.code !== code || report.error?.syscall !== syscall : report.error !== undefined) ||
        ((mode === "partial" || mode === "close") && report.bytes !== Buffer.from("partial-é-💎-write").toString("hex")) ||
        (mode === "open" && report.bytes !== Buffer.from("original").toString("hex"))) {
      throw new Error(`Native filesystem fault/cleanup contract failed: ${log}`);
    }
    fileFaults.push({mode, report, opened: Number(counts[1]), writes: Number(counts[2]), closed: Number(counts[3])});
  }
  fs.writeFileSync(path.join(output, "filesystem-faults.json"), JSON.stringify(fileFaults, null, 2) + "\n");
  const spawnFault = run(faultExecutable, [], path.join(output, "spawn-failure.log"), 0,
    {...oracleEnv, MOM_AOT_FAILURE: "spawn"});
  if (spawnFault.trim() !== '{"spawnFailure":"ENOENT","order":["error","close"],"recovered":true}') {
    throw new Error("Native failed-spawn recovery did not match its contract");
  }
  // Execute the unchanged application entrypoint, not a reimplementation of its CLI/config code.
  // The only normalized data is log wall time; options, output and exit statuses must match.
  const config = path.join(output, "cli-é-💎.json");
  const invalidConfig = path.join(output, "cli-invalid.json");
  fs.writeFileSync(config, JSON.stringify({log_level: 0, pools: [{url: "127.0.0.1", port: 1, login: "fixture"}],
    pool_ids: {primary: 0, donate: null}}));
  fs.writeFileSync(invalidConfig, '{"log_level":{}}');
  const cliCases = [
    {name: "help", args: [], status: 1, marker: "No directive specified"},
    {name: "unknown", args: ["invalid-directive"], status: 1, marker: "Unknown directive"},
    {name: "config", args: ["mine", config, "--invalid", "x"], status: 1, marker: "Unparsed option"},
    {name: "invalid-config", args: ["mine", invalidConfig], status: 1, marker: "must be a scalar"},
    ...(cpuCore ? [{name: "algos", args: ["algos"], status: 0, marker: "MOM_ALGORITHMS "}] : []),
  ];
  if (cpuCore) {
    const vector = require("../tests/vectors/cpu").find((item) => item.name === "cn/r height 1806260");
    if (!vector || !vector.job.blob_hex || !vector.job.algo) {throw new Error("Missing controller CPU vector");}
    const args = ["test", vector.job.algo, vector.expected, "--job.dev", "cpu", "--job.height", String(vector.job.height),
      "--job.blob_hex", vector.job.blob_hex];
    cliCases.push({name: "cpu-worker", args, status: 0, marker: "PASSED"},
      {name: "cpu-multi-worker", args: args.map((value) => value === "cpu" ? "cpu^2" : value), status: 0, marker: "PASSED"});
    cliCases.push({name: "cpu-mismatch", args: args.map((value) => value === vector.expected ? "00".repeat(32) : value),
      status: 1, marker: "FAILED"});
  }
  const cliEnv = {...oracleEnv, MOM_COMMAND: "./mom", MOM_GPU_BACKEND: "", MOM_SKIP_MSR: "1",
    MOM_SKIP_SYCL_ALGO_PARAMS: "1", MOM_USE_SUBPROCESS_WORKERS: "1",
    MOM_NATIVE_PATH: oracleEnv["MOM_AOT_CORE_ADDON"], MOM_AOT_FAILURE: undefined};
  for (const {name, args, status, marker} of cliCases) {
    const nodeLog = path.join(output, `cli-${name}-node.log`);
    const nativeLog = path.join(output, `cli-${name}-native.log`);
    run(process.execPath, [path.join(repo, "mom.js"), ...args], nodeLog, status, cliEnv);
    run(executable, args, nativeLog, status, {...cliEnv, MOM_AOT_RUN_CONTROLLER: "1"});
    const normalize = (/** @type {string} */ value) => value.replace(/^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d /gm, "");
    const expected = normalize(fs.readFileSync(nodeLog, "utf8"));
    const actual = normalize(fs.readFileSync(nativeLog, "utf8"));
    if (!actual.includes(marker) || actual !== expected || /Native (loop did not drain|cleanup retained handles)/.test(actual)) {
      throw new Error(`Original controller CLI parity failed (${name}); see ${nativeLog}`);
    }
  }
  if (cpuCore) {
    for (const [name, command, args] of /** @type {Array<[string, string, string[]]>} */ ([
      ["node", process.execPath, [path.join(repo, "mom.js"), "algos"]], ["native", executable, ["algos"]],
    ])) {
      const result = await new Promise((resolve) => {
        let stderr = "";
        let failure = "";
        const child = spawn(command, args, {stdio: ["ignore", "pipe", "pipe"],
          env: {...process.env, ...cliEnv, MOM_TEST_NO_POOL_NETWORK: "1",
            MOM_AOT_RUN_CONTROLLER: name === "native" ? "1" : undefined},
          signal: AbortSignal.timeout(10000), killSignal: "SIGKILL"});
        child.stderr.on("data", (bytes) => {stderr = (stderr + bytes).slice(-65536);});
        child.on("error", (error) => {failure = error.message;});
        child.on("close", (status, signal) => resolve({status, signal, failure, stderr}));
        child.stdout.destroy(); // Reproduce a real broken pipe, not a mocked failed write.
      });
      fs.writeFileSync(path.join(output, `cli-broken-output-${name}.log`), JSON.stringify(result, null, 2) + "\n");
      if (result.status !== 1 || result.signal || result.failure || !/EPIPE|broken pipe/.test(result.stderr) ||
          /DEP0168|Native (loop did not drain|cleanup retained handles)/.test(result.stderr)) {
        throw new Error(`Controller broken-output handling failed (${name})`);
      }
    }
  }
  for (const failure of ["childTeardown", "promiseTeardown"]) {
    const childFailure = path.join(output, `${failure}-failure.log`);
    run(executable, [], childFailure, 1, {...oracleEnv, MOM_AOT_FAILURE: failure});
    const childDiagnostic = fs.readFileSync(childFailure, "utf8");
    const childPid = Number(childDiagnostic.match(/^AOT_CHILD_TEARDOWN_PID (\d+)$/m)?.[1]);
    if (!Number.isSafeInteger(childPid) || childPid <= 0 || !childDiagnostic.includes("EXPECTED_CHILD_TEARDOWN_FAILURE") ||
        /Native (loop did not drain|cleanup retained handles)|termination failed/.test(childDiagnostic)) {
      throw new Error("Invalid child teardown failure evidence");
    }
    let reaped = false;
    try {process.kill(childPid, 0);} catch (error) {
      if (error instanceof Error && Reflect.get(error, "code") === "ESRCH") {reaped = true;} else {throw error;}
    }
    if (!reaped) {throw new Error(`Native teardown left its child ${childPid} alive`);}
  }
  const key = path.join(output, "fixture-key.pem");
  const certificate = path.join(output, "fixture-cert.pem");
  const expired = path.join(output, "fixture-expired.pem");
  const opensslConfig = path.join(output, "fixture-openssl.cnf");
  fs.writeFileSync(opensslConfig, "[req]\ndistinguished_name=dn\n[dn]\nCN=localhost\n");
  run(openssl, ["req", "-x509", "-newkey", "rsa:2048", "-noenc", "-days", "1",
    "-config", opensslConfig,
    "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
    "-keyout", key, "-out", certificate], path.join(output, "certificate.log"));
  run(openssl, ["x509", "-in", certificate, "-signkey", key,
    "-not_before", "20200101000000Z", "-not_after", "20200102000000Z", "-out", expired],
  path.join(output, "expired-certificate.log"));
  const server = fork(path.join(repo, "tests", "reference", "native-pool-server.js"), [key, certificate, expired], {
    stdio: ["ignore", "ignore", "pipe", "ipc"],
  });
  server.stderr?.on("data", (data) => fs.appendFileSync(path.join(output, "server.log"), data));
  const previousPort = process.env["MOM_AOT_POOL_PORT"];
  const previousFailure = process.env["MOM_AOT_FAILURE"];
  const tlsVariables = ["MOM_AOT_TLS_PORT", "MOM_AOT_TLS_EXPIRED_PORT", "MOM_AOT_TLS_SPY_PORT",
    "MOM_AOT_TLS_CA", "MOM_AOT_TLS_EXPIRED_CA", "MOM_AOT_TLS_FINGERPRINT"];
  const previousTls = tlsVariables.map((name) => process.env[name]);
  /** @type {unknown[]} */
  const failures = [];
  try {
    delete process.env["MOM_AOT_FAILURE"];
    const [message] = await once(server, "message", {signal: AbortSignal.timeout(5000)});
    if (!message || typeof message !== "object" || !("port" in message) ||
        typeof message.port !== "number" || !Number.isInteger(message.port) || message.port < 1 || message.port > 65535) {
      throw new Error("Invalid loopback server port");
    }
    process.env["MOM_AOT_POOL_PORT"] = String(message.port);
    for (const [name, field] of [["MOM_AOT_TLS_PORT", "tlsPort"], ["MOM_AOT_TLS_EXPIRED_PORT", "expiredPort"],
      ["MOM_AOT_TLS_SPY_PORT", "spyPort"]]) {
      if (!name || !field) {throw new Error("Missing TLS fixture field");}
      const port = Reflect.get(message, field);
      if (!Number.isInteger(port) || port < 1 || port > 65535) {throw new Error("Invalid TLS fixture port");}
      process.env[name] = String(port);
    }
    process.env["MOM_AOT_TLS_CA"] = certificate;
    process.env["MOM_AOT_TLS_EXPIRED_CA"] = expired;
    process.env["MOM_AOT_TLS_FINGERPRINT"] = new (require("node:crypto").X509Certificate)(fs.readFileSync(certificate)).fingerprint256;
    const fixtureArgs = ["--", "é-💎", "space value", "", "--unrecognized-runtime-flag"];
    const oracle = run(process.execPath, [fixture, ...fixtureArgs], path.join(output, "node.json"), 0, oracleEnv);
    const actual = run(executable, fixtureArgs, path.join(output, "native.json"), 0, oracleEnv);
    if (actual !== oracle) {throw new Error(`Native/Node behavior differs: ${output}`);}
    const nodeFiles = filesystemReport(path.join(files, "node-writes"));
    const nativeFiles = filesystemReport(path.join(files, "native-writes"));
    if (JSON.stringify(nodeFiles) !== JSON.stringify(nativeFiles) ||
        nodeFiles.find(({name}) => name === "private-é-💎.json")?.mode !== 0o640) {
      throw new Error("Native filesystem bytes or permissions differ from Node");
    }
    fs.writeFileSync(path.join(output, "filesystem-parity.json"), JSON.stringify({nodeFiles, nativeFiles}, null, 2) + "\n");
    for (const [mode, status] of /** @type {Array<[string, number]>} */ ([["graceful", 37], ["forced", 41]])) {
      const env = {...oracleEnv, MOM_AOT_PROCESS_EXIT: mode};
      const expected = run(process.execPath, [fixture], path.join(output, `node-exit-${mode}.log`), status, {...oracleEnv, ...env});
      const actual = run(executable, [], path.join(output, `native-exit-${mode}.log`), status, env);
      if (actual !== expected || /Native (loop did not drain|cleanup retained handles)/.test(actual)) {
        throw new Error(`Process exit parity failed (${mode})`);
      }
    }
    for (const phase of ["timer", "nextTick", "oversized"]) {
      process.env["MOM_AOT_FAILURE"] = phase;
      for (const [name, command, args] of /** @type {Array<[string, string, string[]]>} */ ([
        ["node", process.execPath, [fixture]], ["native", executable, []],
      ])) {
        const log = path.join(output, `${name}-${phase}-failure.log`);
        run(command, args, log, 1, oracleEnv);
        const text = fs.readFileSync(log, "utf8");
        const marker = phase === "oversized" ? "Pool message buffer limit exceeded" : "EXPECTED_AOT_ASYNC_FAILURE";
        if (!text.includes(marker) || /Native (loop did not drain|cleanup retained handles)/.test(text)) {
          throw new Error(`Unexpected failure reason: ${log}`);
        }
      }
    }
    process.env["MOM_AOT_FAILURE"] = "tlsWrite";
    const recovered = run(faultExecutable, [], path.join(output, "native-write-failure.log"), 0, oracleEnv);
    if (!recovered.includes('"code":"EPIPE"') || !recovered.includes('"case":"recovered"')) {
      throw new Error("Native TLS write failure did not preserve socket recovery");
    }
    server.send({statistics: true});
    const [report] = await once(server, "message", {signal: AbortSignal.timeout(5000)});
    const statistics = report?.statistics;
    if (!statistics || statistics.plaintextLeaks || statistics.spyConnections < 2 || statistics.spyBytes < 1 ||
        statistics.tlsLogins.some((/** @type {string} */ name) => ["defaultReject", "wrongIdentity", "expired"].includes(name))) {
      throw new Error("TLS fixture observed unauthorized plaintext or incomplete security checks");
    }
    fs.writeFileSync(path.join(output, "tls-server-statistics.json"), JSON.stringify(statistics, null, 2) + "\n");
  } catch (error) {failures.push(error);}
  if (previousPort === undefined) {
    delete process.env["MOM_AOT_POOL_PORT"];
  } else {process.env["MOM_AOT_POOL_PORT"] = previousPort;}
  if (previousFailure === undefined) {
    delete process.env["MOM_AOT_FAILURE"];
  } else {process.env["MOM_AOT_FAILURE"] = previousFailure;}
  tlsVariables.forEach((name, index) => {
    const value = previousTls[index];
    if (value === undefined) {delete process.env[name];} else {process.env[name] = value;}
  });
  try {
    if (server.exitCode === null && server.signalCode === null) {
      const ended = once(server, "exit", {signal: AbortSignal.timeout(5000)});
      if (server.connected) {server.disconnect();}
      try {
        await ended;
      } catch (error) {
        if (server.exitCode === null && server.signalCode === null) {
          server.kill("SIGTERM");
          await once(server, "exit", {signal: AbortSignal.timeout(5000)});
        }
        throw error;
      }
    }
    if (server.exitCode !== 0 || server.signalCode) {throw new Error("Loopback fixture did not shut down cleanly");}
  } catch (error) {failures.push(error);}
  if (failures.length) {throw new AggregateError(failures, "Native parity or fixture teardown failed");}
  run("readelf", ["-sW", object], path.join(output, "native-symbols.log"));
  const dependencies = run("readelf", ["-d", executable], path.join(output, "dependencies.log"));
  auditSymbols(executable, path.join(output, "runtime-symbols.log"));
  const needed = nativeDependencies(dependencies);
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({
    revision, bytes: fs.statSync(executable).size, needed, parity: true, appResources, ioSdk: io,
    nvmlMetadataSha256: nvml?.sha256 || null,
    failureCleanup: ["timer", "nextTick", "oversized", "tlsWrite"],
    filesystem: {bytesAndModesMatch: true, faults: fileFaults.map(({mode}) => mode), descriptorsDrained: true},
    tls: {openssl: "3.5.9", cases: 9, plaintextLeaks: 0}, cpuCore: Boolean(cpuCore),
    process: {arguments: true, cwd: true, exit: ["graceful", "forced"]}, path: ["posix", "win32"],
    controller: {entrypoint: "mom.js", cases: cliCases.map((test) => test.name),
      brokenOutput: Boolean(cpuCore), gpuDiscovery: false},
    childProcesses: {sameExecutable: true, failureTeardownReaped: true, spawnFailureRecovery: true},
    output: {slowReader: true, workerBurst: Boolean(cpuCore), brokenReaderRecovery: Boolean(cpuCore), streamCap: 32 * 1024 * 1024},
    promiseRejections: rejectionModes,
    scope: `Controller modules, loopback TCP/TLS${cpuCore ? " and real CPU Core vector" : ""}; not full miner qualification`,
  }, null, 2) + "\n");
  console.log(`Native module/TCP/TLS${cpuCore ? "/CPU-core" : ""} parity passed; full controller, process workers, SYCL and Windows remain unqualified.`);
}

module.exports = {nvmlMetadataInput, controllerEntry, run, nativeApplicationGpu, nativeGpuLinkArgs, nativeApplicationLinkPlan, nativeApplicationInputs, nativeApplicationLinkCacheInputs, nativeApplicationBuildInputs, nativeComgrLinkInputs,
  nativeApplicationCudaImages, verifyCudaImageReceiptInputs, flushLinuxObjectBatch,
  linuxCompileEnvironment};
if (require.main === module) {
  main().catch((error) => {console.error(error); process.exitCode = 1;});
}
