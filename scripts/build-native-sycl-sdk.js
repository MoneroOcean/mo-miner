"use strict";

// Build reviewed static CUDA and/or HIP SYCL SDKs beside installed Engineering toolchains. This is
// CPU-only compilation, not GPU execution or SDK installation over an existing prefix.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {admitSyclSource, cudaRuntimeInputs, rocmInputs, staticSyclInputs} = require("./native-sycl-inputs");
const {lldInputs} = require("./native-jit-link");
const {hasExactBuildFields, buildPathsOverlap, buildFileInputs, fileHash, sourceInputs, verifySourceInputs, nativeBuildJobs} = require("./build-source-inputs");
const {absolutePath} = require("./build-native-release");
const {nativeSdkEnvironment} = require("./build-environment");
const {requireLinuxSysroot, linuxSysrootRecord} = require("./native-linux-sysroot");
const {isStaticArchive} = require("./native-audit");
const {verifyComgrSdkSelection} = require("./native-comgr-inputs");

// NVIDIA cuda-cudart-dev12.6.77-1, verified against its official repository SHA256 package
// c9920d6ff4368b657e91ef405b5283c3e85f7507d2420e4220282685efb959e2. Header macros alone
// cannot distinguish a12.x header tree accidentally paired with13.x runtime archives.
const cudaPins = Object.freeze({
  "libcudart_static.a": "5a90415e481a710fa00676f9fe1bbae703f33450c4fd3f35146a3f2980567b06",
  "libculibos.a": "5349773e44c22fd17aefd46f9f595e07eb033bd04017423f41298bf5f935900a",
});

/** CMake's -D values are list/interpolation syntax; compiler paths must stay literal argv.
 * @param {unknown} file @returns {file is string} */
function cmakeCompilerPath(file) {
  return absolutePath(file) && file.length <= 4096 && !/[;"$\\]/.test(file);
}

/** Keep host compilation selectable without changing the LLVM compiler used for SYCL JIT ABI.
 * @param {string} llvm @param {{cc: string, cxx: string} | undefined} selected */
function selectHostCompilers(llvm, selected) {
  const compilers = selected === undefined ? {cc: path.join(llvm, "bin/clang"), cxx: path.join(llvm, "bin/clang++")} : selected;
  if (!hasExactBuildFields(compilers, ["cc", "cxx"]) ||
      !cmakeCompilerPath(compilers.cc) || !cmakeCompilerPath(compilers.cxx)) {
    throw new Error("Host C and C++ compilers must be an absolute, literal pair");
  }
  return {cc: compilers.cc, cxx: compilers.cxx};
}

/** Keep Autoconf's deferred $CC expansion to safe command names; quotes injected through
 * the environment become literal filename characters.
 * @param {{cc: string, cxx: string}} compilers */
function compilerSelection(compilers) {
  const selected = selectHostCompilers("", compilers), names = /** @type {Map<string, string>} */ (new Map()), directories = /** @type {string[]} */ ([]);
  for (const file of [selected.cc, selected.cxx]) {
    const name = path.posix.basename(file), directory = path.posix.dirname(file), prior = names.get(name);
    const unsafeDirectory = directory.includes(":") || [...directory].some(character => {
      const code = character.charCodeAt(0); return code < 0x20 || code === 0x7f;
    });
    if (path.posix.resolve(file) !== file || !/^[A-Za-z0-9_][A-Za-z0-9_.+-]*$/.test(name) || unsafeDirectory ||
        (prior && prior !== file)) {throw new Error("Host compilers need unique safe names in PATH-safe directories");}
    names.set(name, file);
    if (!directories.includes(directory)) {directories.push(directory);}
  }
  return {selected, names: {CC: path.basename(selected.cc), CXX: path.basename(selected.cxx)}, directories};
}

/** @param {{cc: string, cxx: string}} compilers */
function compilerEnvironment(compilers) {
  const selection = compilerSelection(compilers);
  return {CC: selection.names.CC, CXX: selection.names.CXX};
}

/** @param {NodeJS.ProcessEnv} environment @param {{cc: string, cxx: string}} compilers @returns {NodeJS.ProcessEnv} */
function sdkHostEnvironment(environment, compilers) {
  const env = nativeSdkEnvironment(environment), selection = compilerSelection(compilers);
  return {...env, CC: selection.names.CC, CXX: selection.names.CXX, PATH: [
    ...selection.directories, env["PATH"] || ""].filter(Boolean).join(path.posix.delimiter)};
}

/** Resolve each Autotools command through the exact selected compiler directories.
 * @param {{cc: string, cxx: string}} compilers @param {NodeJS.ProcessEnv} environment */
function verifyCompilerEnvironment(compilers, environment) {
  const selection = compilerSelection(compilers);
  if (environment["CC"] !== selection.names.CC || environment["CXX"] !== selection.names.CXX) {
    throw new Error("Host compiler environment differs from its admitted command names");
  }
  for (const compiler of new Set(Object.values(selection.selected))) {
    const name = path.posix.basename(compiler), result = spawnSync("/bin/sh", ["-c", "command -v \"$1\"", "mom-host-compiler", name],
      {env: environment, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
    if (result.status !== 0 || result.error || result.signal || result.stdout?.trim() !== compiler) {
      throw new Error("Host compiler PATH does not resolve to its admitted input: " + compiler);
    }
  }
}

/** Validation helpers execute CXX directly; use the selected absolute compiler path.
 * @param {NodeJS.ProcessEnv} environment @param {string} cxx */
function staticSyclValidationEnvironment(environment, cxx) {return {...environment, CXX: cxx};}

/** Bind each queried compiler identity to the exact file bytes used by the build.
 * @param {{cc: string, cxx: string}} compilers
 * @param {(command: string, args: string[], name: string) => string} query */
function hostCompilerEvidence(compilers, query) {
  const inputs = buildFileInputs([...new Set(Object.values(compilers))]), versions = /** @type {Record<string, string>} */ ({});
  for (const [name, compiler] of Object.entries(compilers)) {
    if (Object.hasOwn(versions, compiler)) {continue;}
    const version = query(compiler, ["--version"], "host-" + name + "-version.log").trim();
    if (!version || version.length > (1 << 20) || version.includes("\0")) {throw new Error("Selected host compiler returned an invalid version: " + compiler);}
    versions[compiler] = version;
  }
  if (!isDeepStrictEqual(inputs, buildFileInputs(Object.keys(inputs)))) {throw new Error("Selected host compiler changed during identity query");}
  return {inputs, versions};
}

/** Only the reviewed runtime pair may enter this reproducible SDK build.
 * @param {string[]} libraries */
function admitCudaLibraries(libraries) {
  if (libraries.length !== 2 || new Set(libraries.map(file => path.basename(file))).size !== 2 || libraries.some(file =>
    fileHash(file) !== Reflect.get(cudaPins, path.basename(file)))) {throw new Error("CUDA runtime libraries do not match the reviewed 12.6.77 pair");}
}

/** One literal CMake recipe; no shell expansion, host-SDK fallback or backend auto-detection.
 * Source preparation uses the same pinned revision/patches as the consumer.
 * @param {string} source @param {string} llvm @param {string} cuda @param {string} output @param {number} jobs
 * @param {NodeJS.ProcessEnv} [environment] @param {{cc: string, cxx: string}} [selectedHostCompilers] */
function staticSyclSdkPlan(source, llvm, cuda, output, jobs, environment = {}, selectedHostCompilers) {
  const backend = environment["MOM_AOT_SYCL_BACKEND"] || "cuda";
  if (!["cuda", "hip", "cuda-hip"].includes(backend)) {throw new Error("Static SDK backend must be cuda, hip or cuda-hip");}
  const usesCuda = backend !== "hip";
  const hip = backend !== "cuda" ? {rocm: environment["MOM_AOT_ROCM_STATIC_ROOT"] || "",
    lld: environment["MOM_AOT_LLD_ROOT"] || "", bitcode: environment["ROCM_DEVICE_LIBS_PATH"] || "",
    comgr: environment["MOM_AOT_COMGR_RECEIPT"] || ""} : undefined;
  const repo = path.resolve(__dirname, ".."), roots = [source, llvm, cuda, repo, ...Object.values(hip || {})];
  if (![...roots, output].every(absolutePath) || !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {
    throw new Error("Static SYCL SDK needs absolute roots and jobs in 1..16");
  }
  const hostCompilers = selectHostCompilers(llvm, selectedHostCompilers);
  const intelOneApiHostPair = /(?:^|\/)intel\/oneapi\/compiler\/[^/]+\/bin\/icx$/.test(hostCompilers.cc) &&
    hostCompilers.cxx === path.join(path.dirname(hostCompilers.cc), "icpx");
  for (const input of [...roots, ...Object.values(hostCompilers)]) {
    if (buildPathsOverlap(input, output)) {throw new Error("Static SDK output must not enter or enclose source, SDK or compiler inputs");}
  }
  const build = path.join(output, "build"), cxx = hostCompilers.cxx;
  const cache = usesCuda ? ["cudart_static", "culibos"].map(name => `CUDA_${name}_LIBRARY:FILEPATH=${path.join(cuda, "lib64/lib" + name + ".a")}`).join("\n") : "";
  // Static archives still enter a PIE executable; Intel host code needs explicit PIC.
  const configure = ["-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    "-DCMAKE_C_COMPILER=" + hostCompilers.cc, "-DCMAKE_CXX_COMPILER=" + cxx,
    // Intel's shared ACPP plugin must not retain dynamically loaded Intel runtime DSOs.
    // Clang's plugin is a CMake MODULE, which does not inherit SHARED linker flags.
    ...(intelOneApiHostPair ? ["-DCMAKE_SHARED_LINKER_FLAGS=-static-intel", "-DCMAKE_MODULE_LINKER_FLAGS=-static-intel"] : []),
    "-DLLVM_DIR=" + path.join(llvm, "lib/cmake/llvm"), ...(usesCuda ? ["-DCUDA_TOOLKIT_ROOT_DIR=" + cuda, "-DCUDAToolkit_ROOT=" + cuda] : []),
    "-DACPP_STATIC_RUNTIME=ON", "-DWITH_SSCP_COMPILER=ON", "-DWITH_STDPAR_COMPILER=OFF", "-DWITH_ACCELERATED_CPU=OFF",
    "-DWITH_CUDA_BACKEND=" + (usesCuda ? "ON" : "OFF"), "-DWITH_ROCM_BACKEND=" + (hip ? "ON" : "OFF"),
    ...["OPENCL", "LEVEL_ZERO", "VULKAN"].map(name => "-DWITH_" + name + "_BACKEND=OFF"),
    "-DACPP_RESOURCE_GENERATOR=" + path.resolve(__dirname, "native-resources.js"), "-DACPP_RESOURCE_NODE=" + process.execPath,
    "-DACPP_SUBPROJECT_PARALLEL_JOBS=1"];
  const targets = ["acpp-rt", ...(usesCuda ? ["rt-backend-cuda"] : []), "rt-backend-omp", "acpp-common", "acpp-clang", "llvm-to-backend", "llvm-to-host", ...(usesCuda ? ["llvm-to-ptx"] : [])];
  if (hip) {
    // Do not let a coinstalled HIPConfig import a different shared runtime/header tree.
    configure.push("-DCMAKE_DISABLE_FIND_PACKAGE_HIP=ON", "-DROCM_PATH=" + hip.rocm, "-DROCM_DEVICE_LIBS_PATH=" + hip.bitcode,
      "-DLLD_DIR=" + path.join(hip.lld, "lib/cmake/lld"), "-DAMDHIP64_LIBRARY=" + path.join(hip.rocm, "lib/libamdhip64.a"),
      "-DHIPRTC_LIBRARY=" + path.join(hip.rocm, "lib/libhiprtc.a"));
    targets.push("rt-backend-hip", "llvm-to-amdgpu");
  }
  return {build, cxx, hostCompilers, compilerEnvironment: compilerEnvironment(hostCompilers), cache, backend, hip, usesCuda,
    commands: [["cmake", ...configure], ["cmake", "--build", build, "--parallel", String(jobs), "--target", ...targets]]};
}

/** Stream noisy build output to its own fd; retain all failures, never retry unknown jobs.
 * @param {string[]} command @param {string} log @param {NodeJS.ProcessEnv} env */
function sdkStage(command, log, env) {
  const fd = fs.openSync(log, "wx+");
  try {
    const result = spawnSync(command[0] || "", command.slice(1), {env, stdio: ["ignore", fd, fd]});
    if (result.error || result.signal || result.status !== 0) {
      const size = fs.fstatSync(fd).size, length = Math.min(size, 16 * 1024), tail = Buffer.alloc(length);
      let read = 0;
      while (read < length) {
        const count = fs.readSync(fd, tail, read, length - read, size - length + read);
        if (count === 0) {break;}
        read += count;
      }
      process.stderr.write(`[native-sdk] Failed stage; showing last ${read} of ${size} log bytes from ${log}:\n`);
      if (read > 0) {
        const output = tail.subarray(0, read);
        process.stderr.write(output);
        if (output[read - 1] !== 0x0a) {process.stderr.write("\n");}
      }
      throw new Error("Static SDK stage failed; inspect " + log + " before retrying", {cause: result.error});
    }
  } finally {fs.closeSync(fd);}
}

/** Build only a new explicit SDK tree. The caller owns host cgroup/resource enforcement;
 * nothing starts a VM, probes hardware, downloads tools or changes installed SDKs.
 * @param {string} source @param {string} llvm @param {string} cuda @param {string} output
 * @param {NodeJS.ProcessEnv} [environment] @param {{cc: string, cxx: string}} [selectedHostCompilers] */
function buildStaticSyclSdk(source, llvm, cuda, output, environment = process.env, selectedHostCompilers) {
  if (process.platform !== "linux" || process.arch !== "x64" || environment["MOM_AOT_BUILD_ONLY"] !== "1") {
    throw new Error("Static SYCL SDK compilation requires Linux x64 and MOM_AOT_BUILD_ONLY=1; no GPU execution is performed");
  }
  const env = nativeSdkEnvironment(environment), plan = staticSyclSdkPlan(source, llvm, cuda, output,
    nativeBuildJobs(env), env, selectedHostCompilers);
  if (fs.existsSync(output)) {throw new Error("Select a new static SDK output; existing SDKs/evidence are never overwritten");}
  const openmp = env["MOM_AOT_OPENMP_LIBRARY"];
  if (!absolutePath(openmp) || !isStaticArchive(openmp)) {throw new Error("Select the matching static OpenMP archive with MOM_AOT_OPENMP_LIBRARY");}
  const canonicalOutput = path.join(fs.realpathSync(path.dirname(output)), path.basename(output));
  const repo = path.resolve(__dirname, ".."), canonicalInputs = [source, llvm, cuda, repo, openmp, ...Object.values(plan.hostCompilers),
    ...(plan.hip ? [plan.hip.rocm, plan.hip.lld, plan.hip.bitcode, plan.hip.comgr] : [])].map(file => fs.realpathSync(file));
  if (canonicalInputs.some(file => buildPathsOverlap(file, canonicalOutput))) {throw new Error("Static SDK output overlaps a canonical source, SDK or compiler input");}
  Object.assign(env, sdkHostEnvironment(env, plan.hostCompilers), {LLVM_CONFIG: path.join(llvm, "bin/llvm-config")});
  verifyCompilerEnvironment(plan.hostCompilers, env);
  const sysroot = requireLinuxSysroot(env);
  const recipe = sourceInputs(repo, [], ["scripts/build-native-sycl-sdk.js", "scripts/native-sycl-inputs.js", "scripts/native-hcf.js", "scripts/native-linux-sysroot.js",
    "tests/reference/native-sycl-hcf-repro.cpp",
    "tests/reference/native-jit-cache.cpp",
    "scripts/native-jit-link.js", "scripts/build-environment.js", "scripts/native-audit.js", "scripts/native-resources.js",
    "runtime/adaptivecpp-static.patch", "runtime/adaptivecpp-jit.patch"]);
  fs.mkdirSync(output); fs.mkdirSync(path.join(output, "pre")); fs.mkdirSync(path.join(output, "post"));
  /** Compiler probes have bounded stdout; CMake compilation uses streamed sdkStage instead.
   * @param {string} command @param {string[]} args @param {string} name */
  function query(command, args, name) {
    const result = spawnSync(command, args, {env, encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
    fs.writeFileSync(path.join(output, name), (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
    if (result.error || result.signal || result.status !== 0) {throw new Error("Static SDK input probe failed: " + name, {cause: result.error});}
    return result.stdout;
  }
  const admitted = admitSyclSource(source, path.join(output, "pre"), true);
  const version = query(path.join(llvm, "bin/llvm-config"), ["--version"], "llvm-version.log").trim();
  if (!/^21\./.test(version)) {throw new Error("Static SYCL SDK currently requires the reviewed LLVM21 toolchain");}
  const {inputs: hostCompilerInputs, versions: compilerVersions} = hostCompilerEvidence(plan.hostCompilers, query);
  const cudaInputs = plan.usesCuda ? cudaRuntimeInputs(plan.cache, plan.cxx, query) : undefined;
  if (cudaInputs) {admitCudaLibraries(cudaInputs.libraries);}
  const hipFiles = [];
  if (plan.hip) {
    const selected = plan.hip, hip = rocmInputs(selected.rocm, selected.comgr), lld = lldInputs(selected.lld, version);
    if (!hip.comgr) {throw new Error("Missing verified COMGR component");}
    verifyComgrSdkSelection(hip.comgr, llvm, selected.lld, selected.bitcode, version);
    const bitcode = fs.readdirSync(selected.bitcode).filter(name => /^[A-Za-z0-9_.-]+\.bc$/.test(name));
    if (!["ockl.bc", "ocml.bc"].every(name => bitcode.includes(name)) || !bitcode.some(name => /^oclc_abi_version_[0-9]+\.bc$/.test(name))) {
      throw new Error("Static HIP JIT needs explicit ROCm builtins and ABI bitcode");
    }
    hipFiles.push(...hip.libraries, ...Object.keys(hip.comgr.files), ...lld, ...bitcode.map(name => path.join(selected.bitcode, name)),
      ...["hip_runtime.h", "hip_runtime_api.h", "hiprtc.h", "hip_version.h"].map(name => path.join(hip.include, "hip", name)),
      path.join(selected.rocm, "lib/libhsa-runtime64.a"), path.join(selected.rocm, "lib/cmake/hsa-runtime64/hsacore.so.link"),
      path.join(selected.lld, "lib/cmake/lld/LLDConfigVersion.cmake"), path.join(selected.lld, "include/lld/Common/Driver.h"));
  }
  const fingerprints = {
    ...Object.fromEntries([path.join(llvm, "bin/clang"), path.join(llvm, "bin/clang++"), path.join(llvm, "bin/llvm-config")].map(file => [file, fileHash(file)])),
    ...hostCompilerInputs,
    ...Object.fromEntries([...(cudaInputs ? [path.join(cudaInputs.include, "cuda.h"), path.join(cudaInputs.include, "cuda_runtime_api.h"), ...cudaInputs.libraries] : []),
      openmp, ...hipFiles].map(file => [file, fileHash(file)])),
  };
  fs.writeFileSync(path.join(output, "plan.json"), JSON.stringify({admitted, version, hostCompilers: plan.hostCompilers, hostCompilerInputs, compilerVersions,
    commands: plan.commands, fingerprints, recipe}, null, 2) + "\n", {flag: "wx"});
  console.log("[native-sdk] configure"); sdkStage(plan.commands[0] || [], path.join(output, "stage-0.log"), env);
  const cache = fs.readFileSync(path.join(plan.build, "CMakeCache.txt"), "utf8");
  if (cudaInputs) {
    const configured = cudaRuntimeInputs(cache, plan.cxx, (command, args, name) => query(command, args, "post-" + name));
    if (!isDeepStrictEqual(cudaInputs, configured)) {throw new Error("Configured CUDA SDK differs from the admitted runtime/header pair");}
  }
  if (plan.hip) {
    for (const [name, expected] of Object.entries({ROCM_PATH: plan.hip.rocm, ROCM_DEVICE_LIBS_PATH: plan.hip.bitcode,
      LLD_DIR: path.join(plan.hip.lld, "lib/cmake/lld")})) {
      const values = [...cache.matchAll(new RegExp(`^${name}:[^=]+=(.+)$`, "gm"))];
      if (values.length !== 1 || fs.realpathSync(values[0]?.[1] || "") !== fs.realpathSync(expected)) {throw new Error("Configured HIP SDK differs: " + name);}
    }
    if (!/^CMAKE_DISABLE_FIND_PACKAGE_HIP:UNINITIALIZED=ON$/m.test(cache)) {throw new Error("Static HIP SDK must not import a host HIPConfig");}
  }
  console.log("[native-sdk] build"); sdkStage(plan.commands[1] || [], path.join(output, "stage-1.log"), env);
  // The existing consumer rechecks source patches, static/SSCP settings and backend set.
  const inputs = staticSyclInputs(source, plan.build, path.join(output, "post"), plan.backend, staticSyclValidationEnvironment(env, plan.cxx));
  for (const [file, sha256] of Object.entries(fingerprints)) {if (fileHash(file) !== sha256) {throw new Error("Static SDK input changed during compilation");}}
  verifySourceInputs(recipe, sourceInputs(repo, [], Object.keys(recipe.graph)), "Static SDK recipe");
  const libraries = Object.fromEntries(inputs.libraries.map(file => [file, fileHash(file)]));
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({version: 1, backend: plan.backend, ...(cudaInputs ? {cudaVersion: cudaInputs.version} : {}),
    revision: admitted.revision, inputs, hostCompilers: plan.hostCompilers, hostCompilerInputs, compilerVersions, fingerprints, libraries, recipe,
    sysroot: linuxSysrootRecord(sysroot), buildOnly: true, publicationQualified: false,
    scope: "Reviewed static CUDA12.6 and/or HIP AdaptiveCpp runtime/plugin/SSCP SDK source recipe. No installation, GPU execution, performance or full SDK provenance qualification."}, null, 2) + "\n", {flag: "wx"});
  console.log("[native-sdk] Built " + plan.build + "; NOT executed or publication-qualified.");
}

if (require.main === module) {
  try {
    const [source, llvm, cuda, output, cc, cxx, extra] = process.argv.slice(2);
    if (!source || !llvm || !cuda || !output || extra || Boolean(cc) !== Boolean(cxx)) {
      throw new Error("Usage: build-native-sycl-sdk.js /AdaptiveCpp /LLVM21 /CUDA12.6 /new-output [/host-cc /host-cxx]");
    }
    buildStaticSyclSdk(source, llvm, cuda, output, process.env, cc && cxx ? {cc, cxx} : undefined);
  } catch (error) {console.error(error instanceof Error ? error.message : "Static SDK build failed"); process.exitCode = 1;}
}
module.exports = {staticSyclSdkPlan, sdkStage, buildStaticSyclSdk, admitCudaLibraries, cmakeCompilerPath, selectHostCompilers,
  compilerEnvironment, sdkHostEnvironment, verifyCompilerEnvironment, staticSyclValidationEnvironment, hostCompilerEvidence};
