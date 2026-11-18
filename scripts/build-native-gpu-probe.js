"use strict";

// Build-only experiment: reuse the maintained gyp SYCL source list and the reviewed static
// runtime/JIT recipe. Neither compilation nor linking may implicitly launch a GPU workload.
const fs = require("node:fs"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const {fileHash, nativeSourceInputs, verifySourceInputs, buildFileInputs, completeBuildInputs} = require("./build-source-inputs");
const {staticSyclInputs} = require("./probe-static-sycl");
const {cudaHeaderResources, cudaHeaderInputs} = require("./native-jit-headers");
const {isolateGpuObjects} = require("./native-gpu-table");
const {compileNativeObjects} = require("./native-object-cache");

/** Keep the original SDK umbrella (including its namespace alias) at its original path.
 * An explicit path avoids both relocated relative includes and Windows case-folding recursion.
 * @param {string} source @param {string} output */
function writeAdaptiveCppUmbrella(source, output) {
  const header = fs.realpathSync(path.join(source, "include/SYCL/sycl.hpp")).replaceAll("\\", "/");
  if (Array.from(header).some(char => char === '"' || char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) {
    throw new Error("AdaptiveCpp umbrella path cannot be represented in an include");
  }
  fs.mkdirSync(path.join(output, "sycl"));
  fs.writeFileSync(path.join(output, "sycl/sycl.hpp"), '#include "' + header + '"\n', {flag: "wx"});
}

/** Hash membership only for selected external include roots. Maintained source membership is
 * already captured by nativeSourceInputs, while generated output headers are declared per TU.
 * @param {string[]} args @param {string} repo @param {string} output */
function nativeIncludeRoots(args, repo, output) {
  const selected = [];
  for (let index = 0; index < args.length; index++) {
    const value = args[index] || "";
    if (["-I", "-isystem", "-iquote"].includes(value)) {selected.push(args[++index] || "");} else if (value.startsWith("-I") && value.length > 2) {selected.push(value.slice(2));} else if (value.startsWith("--cuda-path=")) {selected.push(path.join(value.slice("--cuda-path=".length), "include"));}
  }
  const outputRoot = path.resolve(output), sourceRoot = path.resolve(repo);
  return [...new Set(selected.filter(Boolean).map(root => path.resolve(repo, root)).filter(root => {
    const relOutput = path.relative(outputRoot, root), relSource = path.relative(sourceRoot, root);
    if (!relOutput.startsWith(".." + path.sep) && !path.isAbsolute(relOutput)) {return false;}
    if (relSource !== "" && (!relSource.startsWith(".." + path.sep) && !path.isAbsolute(relSource))) {return false;}
    try {return fs.statSync(root).isDirectory();} catch {return false;}
  }))];
}

/** Preserve the maintained AdaptiveCpp per-source exceptions in one testable command plan.
 * @param {string[]} sourceFiles @param {string} compiler @param {string} repo @param {string} output
 * @param {string[]} compileFlags @param {boolean} cuda @param {string} backend
 * @param {boolean} [images] */
function adaptiveCppCompileCommands(sourceFiles, compiler, repo, output, compileFlags, cuda, backend, images = false) {
  if (backend !== "cuda" && backend !== "hip" && backend !== "cuda-hip") {throw new Error("Unknown AdaptiveCpp HCF identity profile");}
  return sourceFiles.map((file, index) => {
    const object = path.join(output, `${index}.o`), depfile = path.join(output, `${index}.d`);
    const flags = file.endsWith(".c") ? ["-x", "c", "-O3", "-ffp-contract=off", "-fPIC"] :
      [...compileFlags, ...(cuda && ["sycl/cn-gpu/cn-gpu.cpp", "sycl/pearlhash/pearlhash.cpp"].includes(file) ? ["-DMOM_SYCL_HAS_CUDA"] : []),
        ...(cuda && !images && file === "sycl/pearlhash/pearlhash.cpp" ? ["-DMOM_EMBEDDED_NVRTC_HEADERS"] : [])];
    const hcfNamespace = file.startsWith("sycl/") && file.endsWith(".cpp") ?
      ["-mllvm", "-acpp-sscp-hcf-object-id-namespace=acpp-" + backend] : [];
    const generatedInputs = [];
    if (file.startsWith("sycl/") && !file.endsWith(".c")) {generatedInputs.push(path.join(output, "sycl/sycl.hpp"));}
    if (cuda && !images && ["sycl/cn-gpu/cn-gpu.cpp", "sycl/pearlhash/pearlhash.cpp"].includes(file)) {
      generatedInputs.push(path.join(output, "nvrtc-api/nvrtc.h"));
    }
    if (cuda && !images && file === "sycl/pearlhash/pearlhash.cpp") {generatedInputs.push(path.join(output, "nvrtc-headers.inc"));}
    return {command: compiler, args: [...flags, ...hcfNamespace, "-MD", "-MF", depfile, "-c", path.join(repo, file), "-o", object],
      source: path.join(repo, file), object, depfile, generatedInputs, timeout: 180000};
  });
}

/** A newer NVRTC API must not shadow the host runtime's cuda.h/driver_types.h. Isolate its
 * single API header; a separately selected compiler uses an explicit matching device-header
 * pack, just like DPC++. Default same-SDK builds retain the existing header collector.
 * @param {string} compilerInclude @param {string} output @param {string | undefined} prebuilt
 * @param {string[]} roots */
function cudaBuildHeaders(compilerInclude, output, prebuilt, roots) {
  const api = path.join(compilerInclude, "nvrtc.h"), sha256 = fileHash(api), include = path.join(output, "nvrtc-api");
  const files = {[api]: sha256}; fs.mkdirSync(include);
  const copy = path.join(include, "nvrtc.h"); fs.copyFileSync(api, copy, fs.constants.COPYFILE_EXCL);
  if (fileHash(copy) !== sha256 || fileHash(api) !== sha256) {throw new Error("NVRTC API changed during admission");}
  let headers;
  if (prebuilt) {
    if (!path.isAbsolute(prebuilt)) {throw new Error("NVRTC device-header pack must be absolute");}
    const admitted = cudaHeaderInputs(prebuilt), sha = admitted.files[prebuilt]; Object.assign(files, admitted.files);
    fs.copyFileSync(prebuilt, path.join(output, "nvrtc-headers.inc"), fs.constants.COPYFILE_EXCL);
    if (fileHash(path.join(output, "nvrtc-headers.inc")) !== sha ||
        Object.entries(admitted.files).some(([file, hash]) => fileHash(file) !== hash)) {throw new Error("NVRTC device headers changed during admission");}
    headers = {source: prebuilt, sha256: sha};
  } else {
    const {source, ...manifest} = cudaHeaderResources(roots); headers = manifest;
    fs.writeFileSync(path.join(output, "nvrtc-headers.inc"), source, {flag: "wx"});
  }
  fs.writeFileSync(path.join(output, "headers.json"), JSON.stringify(headers, null, 2) + "\n", {flag: "wx"});
  return {include, headers, files};
}

/** Read only the maintained base target list, not gyp's executable Python expressions.
 * @param {string} gyp @returns {string[]} */
function syclSources(gyp) {
  const targets = [...gyp.matchAll(/"target_name"\s*:\s*"sycl"[\s\S]*?"sources"\s*:\s*(\[[\s\S]*?\])/g)];
  const sources = targets.length === 1 ? JSON.parse(targets[0]?.[1] || "null") : null;
  if (!Array.isArray(sources) || sources.length < 20 || new Set(sources).size !== sources.length || sources.some((file) =>
    typeof file !== "string" || !/^sycl\/[A-Za-z0-9_/-]+\.cpp$/.test(file))) {
    throw new Error("Unrecognized maintained SYCL source graph");
  }
  return sources;
}

/** Admit only reviewed backend sets; keep source-JIT API definitions uniform across the graph.
 * @param {string} backend @param {boolean} [images] */
function gpuProfile(backend, images = false) {
  if (!["cuda", "hip", "cuda-hip"].includes(backend)) {
    throw new Error("Select the explicit cuda, hip or cuda-hip GPU build profile");
  }
  const cuda = backend.includes("cuda"), hip = backend.includes("hip");
  return {cuda, hip, definitions: [
    ...(cuda ? [images ? "-DMOM_EMBEDDED_CUDA_IMAGES" : "-DMOM_STATIC_NVRTC", "-DMOM_SYCL_ADAPTIVECPP_CUDA"] : []),
    ...(hip ? ["-DMOM_STATIC_HIPRTC", "-DMOM_SYCL_HAS_HIP", "-DMOM_OCTOPUS_HAS_SYCL_NATIVE"] : []),
  ]};
}

/** Reuse the combined worker's literal target policy without evaluating shell or accepting
 * host-specific environment overrides. Intel AOT is optional; portable SPIR-V is mandatory.
 * @param {string} source */
function combinedGpuTargets(source) {
  const matches = [...source.matchAll(/^export MOM_COMBINED_TARGETS="\$\{MOM_COMBINED_TARGETS:-([^"\r\n{}$]+)\}"$/gm)];
  const targets = matches.length === 1 ? (matches[0]?.[1] || "").split(",") : [];
  if ((source.match(/^export MOM_COMBINED_TARGETS=/gm) || []).length !== 1 ||
      targets.length < 2 || targets.length > 3 || new Set(targets).size !== targets.length ||
      !targets.includes("spir64") || targets.filter(target => /^nvidia_gpu_sm_\d+$/.test(target)).length !== 1 ||
      targets.some(target => !/^(?:spir64(?:_gen)?|nvidia_gpu_sm_\d+)$/.test(target))) {
    throw new Error("Unrecognized maintained combined GPU targets");
  }
  return targets.filter(target => target !== "spir64_gen").join(",");
}

/** Preserve the selected maintained compiler recipe: Intel fast paths, the combined worker's
 * SPIR-V plus single forward-compatible NVIDIA image, or portable OpenCL.
 * This reads trusted source configuration, never evaluates gyp/Python or shell expressions.
 * @param {string} gyp @param {string} [profile] @param {string} [combined] */
function dpcppGpuFlags(gyp, profile = "dpcpp", combined) {
  if (!["dpcpp", "oneapi", "dpcpp-opencl"].includes(profile)) {throw new Error("Unrecognized maintained DPC++ compiler profile");}
  syclSources(gyp); // Also require one unambiguous maintained SYCL target.
  const target = gyp.slice(gyp.search(/"target_name"\s*:\s*"sycl"/));
  const implementation = profile === "oneapi" ? "dpcpp" : profile === "dpcpp" ? "dpcpp-cuda" : "dpcpp-opencl";
  // Stay inside this condition's direct body; a missing field must not borrow another profile.
  const matches = [...target.matchAll(new RegExp(`mom_sycl_impl=='${implementation}'"\\s*,\\s*\\{[^{}]*?"cflags\\+"\\s*:\\s*(\\[[\\s\\S]*?\\])`, "g"))];
  const architectures = [...gyp.matchAll(/"mom_cuda_arch%"\s*:\s*"([^"]+)"/g)];
  if (matches.length !== 1 || (profile === "dpcpp" &&
      (architectures.length !== 1 || !/^nvidia_gpu_sm_\d+$/.test(architectures[0]?.[1] || "")))) {
    throw new Error("Unrecognized maintained DPC++ compiler profile");
  }
  const values = JSON.parse(matches[0]?.[1] || "null");
  if (!Array.isArray(values) || !values.length || values.some((value) => typeof value !== "string" || !value.trim())) {
    throw new Error("Invalid maintained DPC++ compiler flags");
  }
  const flags = values.flatMap((value) => (profile === "dpcpp" ? value.replaceAll("<(mom_cuda_arch)", architectures[0]?.[1] || "") : value).trim().split(/\s+/));
  if (!flags.includes("-fsycl") || flags.includes("-DMOM_SYCL_HAS_CUDA") !== (profile === "dpcpp") || flags.some((flag) => /[<>$]/.test(flag))) {
    throw new Error("Incomplete or unresolved maintained DPC++ compiler flags");
  }
  if (profile !== "dpcpp") {return /** @type {string[]} */ (flags);}
  // GPU-CONFIG also selects dpcpp for Intel ZHash. The ordinary combined wrapper supplies
  // its extra SPIR-V target outside gyp's CUDA condition; preserve that policy here too.
  const targets = combinedGpuTargets(combined ?? fs.readFileSync(path.join(__dirname, "combined-build.sh"), "utf8"));
  if (!targets.split(",").includes(architectures[0]?.[1] || "") ||
      flags.filter(flag => flag.startsWith("-fsycl-targets=")).length !== 1) {
    throw new Error("Maintained combined GPU architecture differs from the gyp compiler profile");
  }
  return /** @type {string[]} */ (flags).map(flag => flag.startsWith("-fsycl-targets=") ? "-fsycl-targets=" + targets : flag);
}

/** Forward a stable translation-unit identity after the driver's random default.
 * Keep the established oneAPI namespace unchanged; other DPC++ profiles get their own IDs.
 * @param {string} file @param {string} profile @returns {string[]} */
function dpcppUniquePrefixFlags(file, profile = "dpcpp") {
  const segments = typeof file === "string" ? file.split("/") : [];
  const source = segments.pop() || "";
  if (!["oneapi", "dpcpp", "dpcpp-opencl"].includes(profile) || segments[0] !== "sycl" ||
      segments.some(segment => !/^[A-Za-z0-9_-]+$/.test(segment)) || !/^[A-Za-z0-9_-]+\.cpp$/.test(source)) {return [];}
  const identity = createHash("sha256").update("mom-sycl-unique-prefix-v1\n" + profile + "\n" + file + "\n").digest("hex").slice(0, 16);
  return ["-Xclang", "-fsycl-unique-prefix=uid" + identity];
}

/** Read the maintained combined-worker source override for one native DPC++ TU.
 * The worker gates this override on an Intel AOT image; the native mixed graph
 * deliberately has no spir64_gen image, but reuses the accepted Intel arena
 * geometry. Keep the define on both host/device compilations of this TU so their
 * layout agrees. This restores geometry only; it is not an AOT-profile claim.
 * @param {string} source @param {string} file @param {string} [profile] @returns {string[]} */
function dpcppSourceFlags(source, file, profile = "dpcpp") {
  if (!["dpcpp", "oneapi", "dpcpp-opencl"].includes(profile)) {throw new Error("Unrecognized maintained DPC++ compiler profile");}
  if (profile !== "dpcpp" || file !== "sycl/zhash/zhash.cpp") {return [];}
  const blocks = [...source.matchAll(/^ {2}source_args=\(\)\n {2}if \[ -n "\$intel_aot_device" \] && \[\[ "\$src" == \*sycl\/zhash\/zhash\.cpp \]\]; then\n(?: {4}#[^\r\n]*\r?\n)* {4}source_args=\((-[A-Za-z0-9_]+)\)\n {2}fi$/gm)];
  const flag = blocks.length === 1 ? blocks[0]?.[1] : undefined;
  // A second malformed override must not disappear merely because its body fails the parser.
  if ((source.match(/sycl\/zhash\/zhash\.cpp/g) || []).length !== 1 ||
      blocks.length !== 1 || flag !== "-DMOM_ZHASH_INTEL_LATE_BUCKETS") {
    throw new Error("Unrecognized maintained DPC++ source flag policy");
  }
  return [flag];
}

/** Reuse the normal wrapper's portable translator policy, without evaluating shell source.
 * Only its explicit OpenCL block and literal flag pairs are admitted; changed syntax must be
 * reviewed rather than silently dropping portability restrictions or borrowing another profile.
 * @param {string} source */
function portableOpenclTranslatorFlags(source) {
  const blocks = [...source.matchAll(/^if \[ "\$\{MOM_DPCPP_IMPL:-dpcpp-combined\}" = dpcpp-opencl \]; then\n([\s\S]*?)^fi$/gm)];
  const arrays = blocks.length === 1 ? [...(blocks[0]?.[1] || "").matchAll(/^\s*spirv_translator_args=\(([^)]*)\)/gm)] : [];
  const flags = arrays.length === 1 ? (arrays[0]?.[1] || "").trim().split(/\s+/) : [];
  const keys = flags.filter((_, index) => index % 2).map((flag) => flag.split("=")[0]);
  if (flags.length !== 4 || new Set(keys).size !== 2 || flags.some((flag, index) => index % 2 === 0 ?
    flag !== "-Xspirv-translator=spir64" : !/^--spirv-(?:max-version=1\.\d|ext=[-+]SPV_\w+(?:,[-+]SPV_\w+)*)$/.test(flag))) {
    throw new Error("Unrecognized maintained portable OpenCL translator policy");
  }
  return flags;
}

/** Reuse the normal combined worker's final per-kernel split for static device linking.
 * Compile flags alone do not describe this step: omitting it can merge unrelated algorithms
 * into one costly code-generation module. Never infer a fallback from an unrecognized graph.
 * @param {string} gyp */
function dpcppDeviceLinkFlags(gyp) {
  const matches = [...gyp.matchAll(/mom_sycl_impl=='dpcpp' or mom_sycl_impl=='dpcpp-combined'[\s\S]*?"ldflags\+"\s*:\s*(\[[\s\S]*?\])/g)];
  const flags = matches.length === 1 ? JSON.parse(matches[0]?.[1] || "null") : null;
  if (!Array.isArray(flags) || flags.length !== 1 || flags[0] !== "-fsycl-device-code-split=per_kernel") {
    throw new Error("Unrecognized maintained DPC++ device-link profile");
  }
  return /** @type {string[]} */ (flags);
}

function main() {
  const [source, build, outputArg] = process.argv.slice(2);
  if (process.platform !== "linux" || !source || !build || !outputArg || process.argv.length !== 5) {
    throw new Error("Usage: node scripts/build-native-gpu-probe.js <AdaptiveCpp source> <static GPU/JIT build> <new output directory>");
  }
  const backend = process.env["MOM_AOT_SYCL_BACKEND"] || "cuda";
  const images = process.env["MOM_AOT_CUDA_IMAGES"];
  if (images !== undefined && (!backend.includes("cuda") || !path.isAbsolute(images))) {throw new Error("MOM_AOT_CUDA_IMAGES requires a CUDA profile and absolute image bundle");}
  const {cuda, definitions} = gpuProfile(backend, images !== undefined);
  const imageInputs = images ? require("./native-cuda-images").cudaImageInputs(images).files : {};
  const output = path.resolve(outputArg), repo = path.resolve(__dirname, "..");
  const sourceState = nativeSourceInputs(repo, "gpu"), startedAt = Date.now();
  fs.mkdirSync(output); // Never overwrite earlier evidence or Engineering build products.
  const inputs = staticSyclInputs(source, build, output, backend);
  const sources = syclSources(fs.readFileSync(path.join(repo, "binding.gyp"), "utf8"));
  writeAdaptiveCppUmbrella(inputs.source, output);
  const cache = fs.readFileSync(path.join(inputs.build, "CMakeCache.txt"), "utf8");
  /** @type {object | undefined} */
  let headers;
  /** @type {string[]} */
  let vendorIncludes = [];
  let compilerFiles = /** @type {Record<string, string>} */ ({});
  if (cuda) {
    const cudart = cache.match(/^CUDA_cudart_static_LIBRARY:FILEPATH=(.+)$/m)?.[1];
    const cudaInclude = cudart && path.join(path.dirname(cudart), "../include");
    if (!cudaInclude || !fs.existsSync(path.join(cudaInclude, "cuda.h"))) {throw new Error("Missing reviewed CUDA SDK headers");}
    if (images) {
      compilerFiles = imageInputs;
      vendorIncludes = ["-isystem", cudaInclude];
    } else {
    // One matching compiler/builtins/PTX archive set. A newer isolated compiler may coexist with
    // the CUDA application runtime SDK, as in Engineering; never mutate that installed SDK.
      const compilerSdk = process.env["MOM_AOT_NVRTC_SDK"];
      const compilerLib = compilerSdk ? path.join(fs.realpathSync(compilerSdk), "lib64") : path.dirname(cudart);
      const compilerInclude = compilerSdk ? path.join(fs.realpathSync(compilerSdk), "include") : cudaInclude;
      const prebuilt = process.env["MOM_AOT_NVRTC_HEADERS"];
      if (compilerSdk && !prebuilt) {throw new Error("A separate NVRTC compiler requires MOM_AOT_NVRTC_HEADERS for its reviewed device-header pack");}
      for (const name of ["nvrtc_static", "nvrtc-builtins_static", "nvptxcompiler_static"]) {
        const file = path.join(compilerLib, `lib${name}.a`);
        if (!fs.existsSync(file)) {throw new Error(`Missing static NVIDIA compiler component: ${file}`);}
        inputs.libraries.push(file);
      }
      const headerRoots = [process.env["MOM_CUTLASS_INCLUDE_DIR"] || "/opt/mom/cutlass/include", cudaInclude,
        process.env["MOM_CCCL_INCLUDE_DIR"] || path.join(cudaInclude, "cccl")];
      const selected = cudaBuildHeaders(compilerInclude, output, prebuilt, headerRoots);
      headers = selected.headers; compilerFiles = selected.files;
      vendorIncludes = ["-isystem", selected.include, "-isystem", cudaInclude];
      inputs.links.push("-Wl,-T," + path.join(repo, "runtime/static-constructors.ld"));
    }
  }
  const compileFlags = ["-std=c++20", "-O3", "-ffp-contract=off", "-DNDEBUG", "-fPIC",
    "-ffunction-sections", "-fdata-sections", "-fno-strict-aliasing", "-DACPP_STATIC_RUNTIME", "-D__ACPP__",
    "-DMOM_SYCL_ADAPTIVECPP", ...definitions,
    ...inputs.flags.filter((flag) => !flag.startsWith("-Wl,") && !flag.startsWith("--ld-path=")),
    "-isystem", path.join(inputs.source, "include"), "-isystem", path.join(inputs.build, "include"),
    ...vendorIncludes, "-I", output,
    "-I", path.join(repo, "xmrig")];
  const sdkBefore = buildFileInputs([inputs.cxx, path.join(inputs.build, "CMakeCache.txt"),
    path.join(inputs.build, "src/compiler/libacpp-clang.so"), path.join(inputs.source, "include/SYCL/sycl.hpp"),
    ...inputs.libraries, ...Object.keys(inputs.comgr?.files || {}), ...Object.keys(compilerFiles)]);
  const comgrSourceRoot = inputs.comgr && path.join(path.dirname(fs.realpathSync(inputs.comgr.receipt)), "source");
  const comgrNonCompilerInputs = new Set(Object.keys(inputs.comgr?.files || {}).filter(file => {
    const inRoot = (/** @type {string} */ root) => {
      const relative = path.relative(root, file);
      return relative === "" || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(".." + path.sep));
    };
    return file === inputs.comgr?.receipt || inRoot(repo) || Boolean(comgrSourceRoot && inRoot(comgrSourceRoot));
  }));
  // Generated NVRTC packs and the shared SYCL umbrella are represented per TU; do not make
  // a change to one source's header invalidate the rest of the graph.
  const compileKeyFiles = [...new Set(Object.keys(sdkBefore).filter(file => !Object.hasOwn(compilerFiles, file) &&
    file !== path.join(inputs.source, "include/SYCL/sycl.hpp") && !comgrNonCompilerInputs.has(file)).map(file => fs.realpathSync(file)))];
  fs.writeFileSync(path.join(output, "sdk-inputs-before.json"), JSON.stringify({version: 1, startedAt, compiler: inputs.cxx,
    flags: compileFlags, files: sdkBefore}, null, 2) + "\n", {flag: "wx"});
  // The same strict-FP host helper is already a separate gyp dependency; it is not a GPU rewrite.
  const sourceFiles = [...sources, "sycl/hoohash/host-math.c", "runtime/driver-library.cpp", "runtime/compiler-gate.cpp"];
  const commands = adaptiveCppCompileCommands(sourceFiles, inputs.cxx, repo, output, compileFlags, cuda, backend, Boolean(images));
  const compiled = compileNativeObjects({repo, output, namespace: `linux/gpu/acpp-${backend}`, commands,
    includeRoots: [output, ...nativeIncludeRoots(commands.flatMap(item => item.args), repo, output)], jobs: 1,
    keyFiles: compileKeyFiles,
    finalize: true, liveObjects: commands.map(item => item.object)});
  /** @type {Array<{file: string, object: string, sha256: string, cacheHit: boolean}>} */
  const objects = sourceFiles.map((file, index) => {
    const command = commands[index], result = compiled.objects[index], log = path.join(output, `${index}.log`);
    if (!command || !result) {throw new Error("Native object helper returned an incomplete GPU compile batch");}
    fs.writeFileSync(log, (result.stdout || "") + (result.stderr || "") + (result.error ? result.error + "\n" : ""));
    fs.writeFileSync(path.join(output, `${index}.command.json`), JSON.stringify({command: command.command, args: command.args,
      cacheHit: result.cacheHit, status: result.status, signal: result.signal}) + "\n", {flag: "wx"});
    console.log(`${result.cacheHit ? "Cached" : "Compiled"} ${file}`);
    if (result.status !== 0 || result.error || result.signal) {throw new Error(`GPU source build failed: ${log}`, {cause: result.error});}
    return {file, object: result.object, sha256: fileHash(path.join(repo, file)), cacheHit: result.cacheHit};
  });
  const objectCache = {cacheHits: compiled.cacheHits, cacheMisses: compiled.cacheMisses, cacheRoot: compiled.cacheRoot, ninja: compiled.ninja};
  fs.writeFileSync(path.join(output, "inputs.json"), JSON.stringify({version: 1, ...inputs, compileFlags, objects, objectCache, headers, sourceState,
    backend, scope: "Complete maintained AdaptiveCpp GPU SYCL source graph; BUILD ONLY, no correctness/performance claim"}, null, 2) + "\n");
  const graph = objects.filter(({file}) => file.startsWith("sycl/")).map(({object}) => object);
  /** @param {string} command @param {string[]} args @param {string} label */
  function run(command, args, label) {
    const result = spawnSync(command, args, {encoding: "utf8", timeout: 180000, maxBuffer: 64 << 20});
    fs.writeFileSync(path.join(output, label + ".command.json"), JSON.stringify({command, args}) + "\n", {flag: "wx"});
    fs.writeFileSync(path.join(output, label + ".log"), (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
    if (result.error || result.signal || result.status !== 0) {throw new Error("GPU graph isolation failed: " + label, {cause: result.error});}
    return result;
  }
  // Even with a shared runtime, compiler defines change algorithm layouts (notably C30). Keep
  // each maintained backend's table distinct; a combined build is not a routing substitute.
  const factory = "mom_native_gpu_acpp_" + backend.replaceAll("-", "_");
  const receipt = isolateGpuObjects(inputs.cxx, output, graph, factory, run);
  for (const [file, sha256] of Object.entries(compilerFiles)) {if (fileHash(file) !== sha256) {throw new Error("NVRTC API/device headers changed during compilation");}}
  const sdkInputs = completeBuildInputs(objects.map((_, index) => path.join(output, `${index}.d`)), repo, output, startedAt, sdkBefore);
  verifySourceInputs(sourceState, nativeSourceInputs(repo, "gpu"), "Native GPU");
  const comgr = inputs.comgr ? {receipt: inputs.comgr.receipt, receiptSha256: fileHash(inputs.comgr.receipt),
    archive: inputs.comgr.archive, sha256: fileHash(inputs.comgr.archive)} : undefined;
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({...receipt, backend, sourceState, sdkInputs, comgr, objectCache,
    ...(images ? {cudaImages: images} : {}),
    scope: "Original AdaptiveCpp GPU graph isolated behind its generated typed factory; BUILD ONLY, no GPU/mining/performance qualification."}, null, 2) + "\n", {flag: "wx"});
  console.log(`GPU graph isolated, NOT executed: ${output}`);
}

module.exports = {syclSources, gpuProfile, dpcppGpuFlags, dpcppUniquePrefixFlags, dpcppSourceFlags, dpcppDeviceLinkFlags, portableOpenclTranslatorFlags, writeAdaptiveCppUmbrella,
  cudaBuildHeaders, combinedGpuTargets, nativeIncludeRoots, adaptiveCppCompileCommands};
if (require.main === module) {try {main();} catch (error) {console.error(error); process.exitCode = 1;}}
