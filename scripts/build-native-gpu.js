"use strict";

// Build-only maintained Intel/NVIDIA/portable OpenCL graphs for the standalone release.
// Engineering keeps its addon build; this producer must not discover or execute any device.
const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto"), {spawnSync} = require("node:child_process");
const {fileHash, nativeSourceInputs, verifySourceInputs, buildFileInputs, completeBuildInputs} = require("./build-source-inputs");
const {syclSources, dpcppGpuFlags, dpcppUniquePrefixFlags, dpcppSourceFlags, dpcppDeviceLinkFlags, portableOpenclTranslatorFlags, nativeIncludeRoots} = require("./native-gpu-build");
const {validateDirectory} = require("./check-portable-opencl");
const {sdkIdentity} = require("./native-dpcpp-pins");
const nativeResources = require("./native-resources");
const {isolateGpuObjects} = require("./native-gpu-table");
const {compileNativeObjects} = require("./native-object-cache");

const oneapiMixedFrontendSources = Object.freeze([
  "sycl/beamhash3/beamhash3.cpp", "sycl/c29/c29.cpp", "sycl/c30/c30.cpp", "sycl/etchash/etchash.cpp",
  "sycl/kawpow/kawpow.cpp", "sycl/nexapow/nexapow.cpp", "sycl/nexapow/validation.cpp", "sycl/octopus/octopus.cpp", "sycl/pearlhash/pearlhash.cpp",
  "sycl/xelishashv3/xelishashv3.cpp", "sycl/zelhash/zelhash.cpp",
]);
const oneapiCompileHelpers = Object.freeze(["clang", "clang-offload-bundler", "sycl-post-link", "llvm-spirv", "clang-offload-wrapper",
  "clang-linker-wrapper", "llvm-link", "llvm-foreach", "llc"]);
const oneapiCompilerEnv = "MOM_AOT_ONEAPI_CXX";

/** This supported mixed frontend uses Intel's bundled SYCL headers for qualified TUs;
 * other TUs, device linking, and the later static runtime remain on the selected DPC++ SDK.
 * On DG2, matched vectors/generation checks fail with the nightly frontend but pass with
 * Intel, including against the old shared runtime. Nexa's separate SHA-49 validation TU
 * needs the same selection as its main TU. Xelis ESIMD also needs Intel's original
 * wide-product lowering. Keep compiler workarounds here, not in consensus
 * arithmetic or GPU-name-specific kernel branches.
 * @param {NodeJS.ProcessEnv} env @param {string} profile */
function oneapiCompilerPath(env, profile) {
  const value = env[oneapiCompilerEnv];
  if (value === undefined) {return undefined;}
  if (profile !== "oneapi") {throw new Error(`${oneapiCompilerEnv} is only valid for the oneapi profile`);}
  if (typeof value !== "string" || value.length > 4096 || !path.isAbsolute(value) ||
      Array.from(value).some((char) => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) {
    throw new Error(`${oneapiCompilerEnv} must be an absolute executable path without control characters`);
  }
  return value;
}

/** Query the admitted vendor frontend before output creation. Its version is provenance evidence;
 * the executable itself is added to the existing SDK input receipt. @param {string} compiler */
function queryOneapiCompiler(compiler) {
  let stat;
  try {
    stat = fs.statSync(compiler);
    fs.accessSync(compiler, fs.constants.X_OK);
  } catch (error) {
    throw new Error("Selected oneAPI compiler must be a regular executable", {cause: error});
  }
  if (!stat.isFile() || (stat.mode & 0o111) === 0) {throw new Error("Selected oneAPI compiler must be a regular executable");}
  const result = spawnSync(compiler, ["--version"], {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
  const text = (result.stdout || "") + (result.stderr || "");
  if (result.error || result.signal || result.status !== 0 || !text.trim()) {
    throw new Error("Cannot query selected oneAPI compiler", {cause: result.error});
  }
  if (!/\bIntel\b/i.test(text)) {throw new Error("Selected oneAPI compiler did not identify as Intel");}
  return {text, sha256: createHash("sha256").update(text).digest("hex")};
}

/** Validate and query the optional per-TU frontend before any output or upstream tool launch.
 * @param {NodeJS.ProcessEnv} env @param {string} profile */
function admitOneapiCompiler(env, profile) {
  const compiler = oneapiCompilerPath(env, profile);
  if (compiler === undefined) {return undefined;}
  const version = queryOneapiCompiler(compiler);
  return {compiler, version: version.text, versionSha256: version.sha256};
}

/** Hash the bounded helper/config closure actually selected by Intel's driver. Dependencies
 * consumed by individual TUs are still captured from their depfiles. @param {{compiler: string, version: string}} admitted
 * @param {(command: string, args: string[], options: import("node:child_process").SpawnSyncOptionsWithStringEncoding) => import("node:child_process").SpawnSyncReturns<string>} [spawn] */
function oneapiCompilerInputs(admitted, spawn = spawnSync) {
  const realCompiler = fs.realpathSync(admitted.compiler), root = fs.realpathSync(path.resolve(path.dirname(realCompiler), ".."));
  /** @param {string} base @param {string} file */
  const within = (base, file) => {
    const relative = path.relative(base, file);
    return relative !== "" && !path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep);
  };
  if (!fs.statSync(root).isDirectory() || !within(root, realCompiler)) {throw new Error("Selected oneAPI compiler has no bounded compiler root");}
  /** @param {string} file @param {string} label @param {boolean} executable */
  const admittedFile = (file, label, executable) => {
    if (!path.isAbsolute(file) || /[\r\n\0]/.test(file)) {throw new Error("Selected oneAPI " + label + " path is not absolute");}
    const selected = path.resolve(file), selectedStat = fs.statSync(selected), real = fs.realpathSync(selected);
    if (!selectedStat.isFile() || !within(root, selected) || !within(root, real)) {
      throw new Error("Selected oneAPI " + label + " is not a regular compiler-root file: " + file);
    }
    if (executable) {fs.accessSync(selected, fs.constants.X_OK);}
    return real;
  };
  /** @param {string} name */
  const queryHelper = name => {
    const result = spawn(admitted.compiler, ["-print-prog-name=" + name], {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
    if (result.error || result.signal || result.status !== 0 || typeof result.stdout !== "string" || result.stdout.length > (1 << 20)) {
      throw new Error("Selected oneAPI helper query failed: " + name, {cause: result.error});
    }
    const selected = result.stdout.trim();
    if (path.basename(selected) !== name) {throw new Error("Selected oneAPI helper has an unexpected name: " + name);}
    return admittedFile(selected, name, true);
  };
  const files = oneapiCompileHelpers.map(queryHelper);
  const configs = [...admitted.version.matchAll(/^\s*Configuration file:\s*(.*?)\s*$/gm)].map(match => match[1] || "");
  if (configs.length > 1 || configs.some(file => !file)) {throw new Error("Selected oneAPI compiler has ambiguous configuration evidence");}
  if (configs.length) {files.push(admittedFile(configs[0] || "", "configuration", false));} else {
    const adjacent = path.join(path.dirname(realCompiler), "icpx.cfg");
    if (fs.existsSync(adjacent)) {files.push(admittedFile(adjacent, "configuration", false));}
  }
  return [...new Set(files)];
}

/** Keep the vendor frontend source-limited; flags and device-link inputs remain the DPC++ recipe.
 * @param {string} compiler @param {string} file @param {string} profile @param {string | undefined} override */
function sourceCompiler(compiler, file, profile, override) {
  return profile === "oneapi" && oneapiMixedFrontendSources.includes(file) && override !== undefined ? override : compiler;
}

/** Preserve the maintained mixed-frontend source selection and per-TU flags as literal argv.
 * @param {{sources: string[], repo: string, output: string, sdk: string, compiler: string,
 * flags: string[], wrapper: string, profile: string, cuda?: string, images?: string, oneapiCompiler?: string}} plan */
function dpcppCompileCommands(plan) {
  return plan.sources.map((file, index) => {
    const object = path.join(plan.output, `${index}.o`), depfile = path.join(plan.output, `${index}.d`), c = file.endsWith(".c");
    const sourceFlags = c ? [] : [...dpcppSourceFlags(plan.wrapper, file, plan.profile), ...dpcppUniquePrefixFlags(file, plan.profile)];
    const args = [...(c ? ["-x", "c", "-O3", "-ffp-contract=off", "-fPIC"] : [...plan.flags, ...sourceFlags]),
      "-MD", "-MF", depfile, "-c", path.join(plan.repo, file), "-o", object];
    const compiler = c ? path.join(plan.sdk, "bin/clang") : sourceCompiler(plan.compiler, file, plan.profile, plan.oneapiCompiler);
    const generatedInputs = [];
    if (file === "sycl/kawpow/kawpow.cpp") {generatedInputs.push(path.join(plan.output, "kawpow-jit-resources.inc"));}
    if (plan.cuda && !plan.images && file === "sycl/pearlhash/pearlhash.cpp") {generatedInputs.push(path.join(plan.output, "nvrtc-headers.inc"));}
    return {command: compiler, args, source: path.join(plan.repo, file), object, depfile, generatedInputs,
      timeout: 300000, sourceFlags};
  });
}

/** Keep the admitted compiler identity stable for the entire source/device-link build.
 * @param {{compiler: string, versionSha256: string} | undefined} admitted */
function verifyOneapiCompiler(admitted) {
  if (admitted && queryOneapiCompiler(admitted.compiler).sha256 !== admitted.versionSha256) {
    throw new Error("Selected oneAPI compiler version changed during compilation");
  }
}

/** Admit explicit build inputs before touching output or looking up any compiler.
 * Generated NVRTC headers are trusted build source, like the selected SDK, not runtime input.
 * @param {NodeJS.ProcessEnv} env @param {string[]} args
 * @returns {{sdk: string, output: string, cuda: string | undefined, nvrtc: string | undefined,
 * headers: string | undefined, images?: string}} */
function buildInputs(env, args) {
  const profile = env["MOM_AOT_DPCPP_PROFILE"] || "dpcpp";
  if (!["dpcpp", "oneapi", "dpcpp-opencl"].includes(profile)) {throw new Error("Select the explicit dpcpp, oneapi or dpcpp-opencl compiler profile");}
  // Reject a vendor frontend on another profile before touching any SDK/output path.
  oneapiCompilerPath(env, profile);
  const images = env["MOM_AOT_CUDA_IMAGES"];
  if (images !== undefined && profile !== "dpcpp") {throw new Error("MOM_AOT_CUDA_IMAGES is only valid for the dpcpp CUDA profile");}
  const roots = profile !== "dpcpp" ? args : [...args, ...(images !== undefined ? [images] : [env["MOM_AOT_NVRTC_SDK"], env["MOM_AOT_NVRTC_HEADERS"]])];
  if (args.length !== (profile !== "dpcpp" ? 2 : 3) || !roots.every((root) => typeof root === "string" && path.isAbsolute(root) &&
      !Array.from(root).some((char) => char.charCodeAt(0) < 32))) {
    throw new Error("Select absolute SDK and new output roots; dpcpp additionally requires a CUDA SDK and MOM_AOT_CUDA_IMAGES or MOM_AOT_NVRTC_SDK and MOM_AOT_NVRTC_HEADERS");
  }
  if (profile !== "dpcpp") {return {sdk: args[0] || "", output: args[1] || "", cuda: undefined, nvrtc: undefined, headers: undefined};}
  if (images !== undefined) {return {sdk: args[0] || "", cuda: args[1] || "", output: args[2] || "", images, nvrtc: undefined, headers: undefined};}
  const [sdk, cuda, output, nvrtc, headers] = /** @type {[string, string, string, string, string]} */ (roots);
  return {sdk, cuda, output, nvrtc, headers};
}

/** Device post-link has its own options. Passing the compile-only -fsycl-embed-ir here makes
 * the pinned wrapper read an incompatible metadata table and abort, even for one algorithm.
 * Keep code-generation/target settings, including large-ID metadata, but never compiler
 * defines or dynamic SYCL libraries. Dropping the ID range restores DPC++'s int default.
 * @param {string} gyp @param {string | undefined} cuda @param {string[]} objects @param {string} output @param {string} [profile] @param {string} [wrapper] */
function deviceLinkArgs(gyp, cuda, objects, output, profile = "dpcpp", wrapper = "") {
  if ((profile === "dpcpp" ? !cuda : cuda !== undefined) ||
      ![...(cuda ? [cuda] : []), output, ...objects].every((value) => path.isAbsolute(value)) || !objects.length || new Set(objects).size !== objects.length) {
    throw new Error("Device linking needs unique absolute objects, SDK and output paths");
  }
  const codegen = dpcppGpuFlags(gyp, profile).filter((flag) => /^-std=|^-O[0-3s]$|^-ffp-contract=|^-fsycl-targets=|^-fno-sycl-(?:rdc|instrument-device-code|id-queries-fit-in-int)$/.test(flag));
  const portable = profile === "dpcpp-opencl" ? portableOpenclTranslatorFlags(wrapper) : [];
  return ["-fsycl", "-fPIC", ...codegen, ...dpcppDeviceLinkFlags(gyp), ...portable,
    ...(profile === "dpcpp-opencl" ? ["-fsycl-dump-device-code=" + path.join(path.dirname(output), "device-images")] : []),
    ...(cuda ? [`--cuda-path=${cuda}`] : ["-fsycl-targets=spir64"]), "-fsycl-link", "-nolibsycl",
    "-fsycl-max-parallel-link-jobs=2", ...objects, "-o", output];
}

/** nvptx foreach invokes ptxas by name instead of honoring --cuda-path. Admit the exact
 * selected SDK tools and put their directory first only for the device-link subprocess.
 * @param {string} cuda @param {NodeJS.ProcessEnv} [environment]
 * @returns {{files: string[], inputs: Record<string, string>, environment: NodeJS.ProcessEnv}} */
function cudaDeviceLinkSetup(cuda, environment = process.env) {
  if (!path.isAbsolute(cuda) || /[\r\n\0]/.test(cuda)) {throw new Error("Selected CUDA SDK path must be absolute");}
  const bin = path.join(cuda, "bin"), files = ["ptxas", "fatbinary"].map(name => path.join(bin, name));
  for (const file of files) {
    const stat = fs.statSync(file, {throwIfNoEntry: false});
    if (!stat || !stat.isFile() || (stat.mode & 0o111) === 0) {throw new Error("Selected CUDA SDK needs executable " + path.basename(file));}
    try {fs.accessSync(file, fs.constants.X_OK);} catch {throw new Error("Selected CUDA SDK needs executable " + path.basename(file));}
  }
  // Hash before output creation; completeBuildInputs checks the same files again after linking.
  const inputs = buildFileInputs(files);
  const inheritedPath = typeof environment["PATH"] === "string" ? environment["PATH"].split(path.delimiter).filter(Boolean) : [];
  return {files, inputs, environment: {...environment, PATH: [bin, ...inheritedPath].join(path.delimiter)}};
}

/** The original DSO initialized before priority-1 application image registration. In one
 * static image, retain the exact registration bodies but use regular constructor/destructor
 * sections; link the host runtime before this object. Unregistration then precedes the
 * original runtime's priority-110 late shutdown, instead of calling a destroyed handler.
 * @param {string} sections @param {string} [profile] */
function deviceRegistrationArgs(sections, profile = "dpcpp") {
  if (!["dpcpp", "oneapi", "dpcpp-opencl"].includes(profile)) {throw new Error("Unknown static device registration profile");}
  for (const [name, type] of [["init", "INIT"], ["fini", "FINI"]]) {
    const pattern = new RegExp(`\\]\\s+\\.${name}_array\\.1\\s+${type}_ARRAY\\s`, "g");
    if ((sections.match(pattern) || []).length !== 1) {throw new Error("Unrecognized pinned device registration lifetime");}
  }
  const key = profile.replaceAll("-", "_");
  return ["--rename-section", ".init_array.1=.init_array", "--rename-section", ".fini_array.1=.fini_array",
    "--redefine-sym", "__sycl_register_lib=mom_native_register_" + key,
    "--redefine-sym", "__sycl_unregister_lib=mom_native_unregister_" + key];
}

/** The pinned merged wrapper has exactly one import for each static-image callback. Do not
 * guess at another compiler ABI or silently leave an eager registration in the combined app.
 * @param {string} symbols @param {string} [profile] */
function deviceRegistrationImports(symbols, profile) {
  if (profile !== undefined && !["dpcpp", "oneapi", "dpcpp-opencl"].includes(profile)) {
    throw new Error("Unknown static device registration profile");
  }
  const names = profile === undefined ? ["__sycl_register_lib", "__sycl_unregister_lib"] :
    ["mom_native_register_", "mom_native_unregister_"].map(name => name + profile.replaceAll("-", "_"));
  for (const name of names) {
    if ((symbols.match(new RegExp("^" + name + " U(?:\\s|$)", "gm")) || []).length !== 1) {
      throw new Error("Unrecognized static device registration imports");
    }
  }
}

/** Finalize already-built original objects without recompiling unchanged kernels. The normal
 * builder and retained-input qualification share this exact registration/isolation operation.
 * @param {string} sdk @param {string} output @param {string[]} objects @param {string} device
 * @param {(command: string, args: string[], label: string) => import('node:child_process').SpawnSyncReturns<string>} run @param {string} [profile] */
function isolateGpuGraph(sdk, output, objects, device, run, profile = "dpcpp") {
  const hash = fileHash;
  const sections = run("readelf", ["--sections", "--wide", device], "device-sections");
  if (sections.status !== 0) {throw new Error("Cannot inspect device registration lifetime");}
  const imports = run("nm", ["-P", "-u", device], "device-registration-imports");
  if (imports.status !== 0) {throw new Error("Cannot inspect static device registration imports");}
  deviceRegistrationImports(imports.stdout);
  const registeredDevice = path.join(output, "device-registration.o");
  if (run("objcopy", [...deviceRegistrationArgs(sections.stdout, profile), device, registeredDevice], "registration-lifetime").status !== 0) {
    throw new Error("Cannot preserve static device registration lifetime");
  }
  const hostObjects = [];
  for (const [index, input] of objects.entries()) {
    const object = path.join(output, `host-${index}.o`);
    const unbundled = run(path.join(sdk, "bin/clang-offload-bundler"), ["-type=o", "-targets=host-x86_64-unknown-linux-gnu",
      "-input=" + input, "-output=" + object, "-unbundle", "-allow-missing-bundles"], `unbundle-${index}`);
    if (unbundled.status !== 0) {throw new Error("Cannot retain original DPC++ host object");}
    hostObjects.push(object);
  }
  const receipt = isolateGpuObjects(path.join(sdk, "bin/clang++"), output, [...hostObjects, registeredDevice], "mom_native_gpu_" + profile.replaceAll("-", "_"), run);
  const remaining = run("nm", ["-P", "-u", receipt.object], "isolated-registration-imports");
  if (remaining.status !== 0 || /^__sycl_(?:un)?register_lib\s/m.test(remaining.stdout)) {
    throw new Error("Native GPU graph retained an eager static-image registration");
  }
  deviceRegistrationImports(remaining.stdout, profile);
  return {...receipt, deviceSha256: hash(device), registrationSha256: hash(registeredDevice)};
}

function main() {
  if (process.platform !== "linux" || process.env["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("This build-only DPC++ proof requires Linux and MOM_AOT_BUILD_ONLY=1; no execution is implicit");
  }
  const {sdk, cuda, output, nvrtc, headers, images} = buildInputs(process.env, process.argv.slice(2));
  const headerInputs = headers ? require("./native-jit-headers").cudaHeaderInputs(headers).files : {};
  const imageInputs = images ? require("./native-cuda-images").cudaImageInputs(images).files : {};
  const cudaLink = cuda ? cudaDeviceLinkSetup(cuda) : undefined;
  const profile = process.env["MOM_AOT_DPCPP_PROFILE"] || "dpcpp";
  const oneapi = admitOneapiCompiler(process.env, profile);
  const oneapiInputs = oneapi ? oneapiCompilerInputs(oneapi) : [];
  const repo = path.resolve(__dirname, ".."), compiler = path.join(sdk, "bin/clang++");
  const wrapperPath = path.join(repo, "scripts/cxx-combined.sh");
  const version = spawnSync(compiler, ["--version"], {encoding: "utf8", timeout: 10000});
  if (version.status !== 0) {throw new Error("Cannot query selected DPC++ compiler", {cause: version.error});}
  sdkIdentity(version.stdout, fs.readFileSync(path.join(sdk, ".mom-toolchain-sha256"), "utf8"));
  const sdkFiles = cuda ? [path.join(cuda, "include/cuda.h"), path.join(cuda, "nvvm/libdevice/libdevice.10.bc"), ...(cudaLink?.files || []),
    ...(images ? Object.keys(imageInputs) : nvrtc ? [path.join(nvrtc, "include/nvrtc.h"), ...Object.keys(headerInputs)] : [])] : [];
  if (profile === "dpcpp-opencl") {sdkFiles.push(wrapperPath);}
  for (const file of sdkFiles) {if (!fs.statSync(file).isFile()) {throw new Error("Missing regular SDK/header input: " + file);}}
  const gyp = fs.readFileSync(path.join(repo, "binding.gyp"), "utf8");
  const sourceState = nativeSourceInputs(repo, "gpu"), startedAt = Date.now();
  const sources = [...syclSources(gyp), "sycl/hoohash/host-math.c"], engineeringFlags = dpcppGpuFlags(gyp, profile);
  if (oneapi && oneapiMixedFrontendSources.some((file) => !sources.includes(file))) {
    throw new Error("The maintained oneAPI mixed frontend source graph is incomplete");
  }
  const wrapper = profile === "dpcpp" || profile === "dpcpp-opencl" ? fs.readFileSync(wrapperPath, "utf8") : "";
  const translator = profile === "dpcpp-opencl" ? portableOpenclTranslatorFlags(wrapper) : [];
  fs.mkdirSync(output); // Never overwrite evidence, source objects or an Engineering cache.
  fs.writeFileSync(path.join(output, "compiler-version.log"), version.stdout, {flag: "wx"});
  if (oneapi) {fs.writeFileSync(path.join(output, "oneapi-compiler-version.log"), oneapi.version, {flag: "wx"});}
  if (headers) {fs.copyFileSync(headers, path.join(output, "nvrtc-headers.inc"));}
  const fragments = ["device", "keccak"].map((name) => ({name: `kawpow-${name}.inc`,
    data: fs.readFileSync(path.join(repo, `sycl/kawpow/${name}.inc`))}));
  fs.writeFileSync(path.join(output, "kawpow-jit-resources.inc"), nativeResources(fragments), {flag: "wx"});
  const flags = [...engineeringFlags, ...translator, ...(cuda ? [`--cuda-path=${cuda}`,
    ...(images ? ["-DMOM_EMBEDDED_CUDA_IMAGES"] : nvrtc ? ["-DMOM_STATIC_NVRTC", "-DMOM_EMBEDDED_NVRTC_HEADERS", "-isystem", path.join(nvrtc, "include")] : [])] : ["-fsycl-targets=spir64"]), "-DMOM_EMBEDDED_SYCL_JIT",
  "-fPIC", "-ffunction-sections", "-fdata-sections", "-fno-strict-aliasing", "-I", output, "-I", path.join(repo, "xmrig")];
  const hash = fileHash;
  const tools = ["clang", "clang++", "clang-offload-bundler", "sycl-post-link", "llvm-spirv", "clang-offload-wrapper",
    "clang-linker-wrapper", "clang-offload-packager", "llvm-link", "llc", "opt"].map(name => path.join(sdk, "bin", name)).filter(file => fs.existsSync(file));
  const explicitSdkFiles = [...sdkFiles, ...(oneapi ? [oneapi.compiler] : []), ...oneapiInputs];
  const sdkBefore = buildFileInputs([...explicitSdkFiles, ...tools, path.join(sdk, ".mom-toolchain-sha256")]);
  // Embedded NVRTC pack inputs and parsed wrapper flags are represented per TU; the selected
  // frontend binary is keyed by the command itself. Assemblers affect only the separately
  // executed device link; their hashes stay receipt-bound without invalidating source IR objects.
  const compileKeyFiles = Object.keys(sdkBefore).filter(file => !Object.hasOwn(headerInputs, file) && !Object.hasOwn(imageInputs, file) && file !== wrapperPath &&
    !Object.hasOwn(cudaLink?.inputs || {}, file) && (!oneapi || file !== oneapi.compiler))
    .map(file => fs.realpathSync(file));
  if (Object.entries({...headerInputs, ...imageInputs, ...(cudaLink?.inputs || {})}).some(([file, sha]) => sdkBefore[file] !== sha) ||
      headers && fileHash(path.join(output, "nvrtc-headers.inc")) !== headerInputs[headers]) {
    throw new Error("CUDA compiler inputs changed after admission");
  }
  fs.writeFileSync(path.join(output, "sdk-inputs-before.json"), JSON.stringify({version: 1, startedAt, compiler,
    oneapiCompiler: oneapi ? {path: oneapi.compiler, versionSha256: oneapi.versionSha256} : undefined,
    flags, files: sdkBefore}, null, 2) + "\n", {flag: "wx"});
  /** @type {Array<{file: string, compiler: string, object: string, sourceSha256: string, sourceFlags: string[], status: number | null, cacheHit: boolean}>} */
  const results = [];
  /** @param {string} command @param {string[]} args @param {string} label @param {number} [timeout] @param {NodeJS.ProcessEnv} [environment] */
  function run(command, args, label, timeout = 300000, environment) {
    fs.writeFileSync(path.join(output, label + ".command.json"), JSON.stringify({command, args}) + "\n", {flag: "wx"});
    const result = spawnSync(command, args, {encoding: "utf8", timeout, maxBuffer: 32 << 20, ...(environment ? {env: environment} : {})});
    fs.writeFileSync(path.join(output, label + ".log"), (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
    if (result.error || result.signal || result.status === null) {throw new Error(`Compiler lifecycle unresolved: ${label}; inspect this build before retrying`, {cause: result.error});}
    return result;
  }
  const commands = dpcppCompileCommands({sources, repo, output, sdk, compiler, flags, wrapper, profile,
    ...(images ? {images} : {}),
    ...(cuda ? {cuda} : {}), ...(oneapi ? {oneapiCompiler: oneapi.compiler} : {})});
  const compiled = compileNativeObjects({repo, output, namespace: `linux/gpu/dpcpp-${profile}`, commands,
    includeRoots: [output, ...nativeIncludeRoots(commands.flatMap(item => item.args), repo, output)], jobs: 1,
    keyFiles: compileKeyFiles,
    finalize: true, liveObjects: commands.map(item => item.object)});
  const objectCache = {cacheHits: compiled.cacheHits, cacheMisses: compiled.cacheMisses, cacheRoot: compiled.cacheRoot, ninja: compiled.ninja};
  for (const [index, file] of sources.entries()) {
    const command = commands[index], result = compiled.objects[index];
    if (!command || !result) {throw new Error("Native object helper returned an incomplete DPC++ compile batch");}
    fs.writeFileSync(path.join(output, `${index}.command.json`), JSON.stringify({command: command.command, args: command.args,
      cacheHit: result.cacheHit, status: result.status, signal: result.signal}) + "\n", {flag: "wx"});
    fs.writeFileSync(path.join(output, `${index}.log`), (result.stdout || "") + (result.stderr || "") + (result.error ? result.error + "\n" : ""));
    results.push({file, compiler: command.command, object: result.object, sourceSha256: hash(path.join(repo, file)),
      sourceFlags: command.sourceFlags, status: result.status, cacheHit: result.cacheHit});
    fs.writeFileSync(path.join(output, "inputs.json"), JSON.stringify({profile, engineeringFlags, flags, results, objectCache, sourceState,
      // Progress uses the admitted snapshot; completeBuildInputs rehashes it after device linking.
      sdkFiles: Object.fromEntries(explicitSdkFiles.map((file) => [file, sdkBefore[file]])),
      oneapiCompiler: oneapi ? {path: oneapi.compiler, versionSha256: oneapi.versionSha256} : undefined,
      mixedFrontend: oneapi ? {sources: oneapiMixedFrontendSources,
        caveat: "Intel SYCL headers are used for these TUs; all other TUs and device linking stay on the selected DPC++ SDK, with its runtime supplied by the later static-link recipe."} : undefined,
      scope: "Complete maintained compiler-specific source graph, BUILD ONLY. Device-image/static runtime linking and GPU/mining/performance qualification remain separate gates."}, null, 2) + "\n");
    console.log(`${file}: ${result.status === 0 ? result.cacheHit ? "cached" : "compiled" : "FAILED"}`);
  }
  if (results.some((result) => result.status !== 0)) {throw new Error("DPC++ source failures retained; see per-source logs and inputs.json");}
  const device = path.join(output, "device-linked.o");
  if (profile === "dpcpp-opencl") {fs.mkdirSync(path.join(output, "device-images"));}
  // Mixed oneAPI frontends are limited to source compilation; the selected DPC++ SDK owns device linking.
  const linked = run(compiler, deviceLinkArgs(gyp, cuda, results.map((result) => result.object), device, profile, wrapper), "device-link", 900000,
    cudaLink?.environment);
  if (linked.status !== 0) {throw new Error("DPC++ device-image linking failed; see device-link.log");}
  if (profile === "dpcpp-opencl") {
    const report = validateDirectory(path.join(output, "device-images"));
    fs.writeFileSync(path.join(output, "portability.json"), JSON.stringify(report, null, 2) + "\n", {flag: "wx"});
    if (report.errors.length) {throw new Error("Non-portable OpenCL image: " + report.errors.join("; "));}
  }
  fs.writeFileSync(path.join(output, "device-images.json"), JSON.stringify({device, bytes: fs.statSync(device).size, sha256: hash(device),
    scope: "Maintained DPC++ source/device-image graph linked. Host/runtime integration, GPU correctness/mining/performance and release qualification remain separate gates."}, null, 2) + "\n", {flag: "wx"});
  // Unbundle actual host objects and retain only the profile's typed factory. CUDA image
  // selection remains an unresolved C import for the one shared application registry object.
  const receipt = isolateGpuGraph(sdk, output, results.map((result) => result.object), device, run, profile);
  const sdkInputs = completeBuildInputs(results.map((_, index) => path.join(output, `${index}.d`)), repo, output, startedAt, sdkBefore);
  verifyOneapiCompiler(oneapi);
  verifySourceInputs(sourceState, nativeSourceInputs(repo, "gpu"), "Native GPU");
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({...receipt, profile, sources, sdk, sourceState, sdkInputs, objectCache,
    ...(images ? {cudaImages: images} : {}),
    oneapiCompiler: oneapi ? {path: oneapi.compiler, versionSha256: oneapi.versionSha256} : undefined,
    mixedFrontend: oneapi ? {sources: oneapiMixedFrontendSources,
      caveat: "Intel SYCL headers are used for these TUs; all other TUs and device linking stay on the selected DPC++ SDK, with its runtime supplied by the later static-link recipe."} : undefined,
    scope: "Complete maintained DPC++ host/device graph with typed private factory, BUILD ONLY. Static runtime integration and GPU/mining/performance/release qualification remain separate gates."}, null, 2) + "\n", {flag: "wx"});
  console.log("Original DPC++ algorithm host/device graph isolated; NOT executed or runtime-qualified");
}

module.exports = {buildInputs, oneapiCompilerPath, admitOneapiCompiler, oneapiCompilerInputs, verifyOneapiCompiler, sourceCompiler, deviceLinkArgs,
  cudaDeviceLinkSetup, dpcppCompileCommands, deviceRegistrationArgs, deviceRegistrationImports, isolateGpuGraph};
if (require.main === module) {try {main();} catch (error) {console.error(error); process.exitCode = 1;}}
