"use strict";

// Pinned AdaptiveCpp source admission and static SYCL inputs shared by the release producers, plus a
// CPU-only runtime proof. GPU backends are explicit BUILD-ONLY lanes; ownership and bounded execution
// must be verified separately before running its artifact.
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {nativeDependencies, auditNativeTrace, isStaticArchive} = require("./native-audit");
const {lldInputs} = require("./native-jit-link");
const {comgrSdkInputs, verifyComgrSdkSelection} = require("./native-comgr-inputs");
const {elfReport} = require("./native-elf");
const {inspectHcfObject} = require("./native-hcf");
const {fileHash} = require("./build-source-inputs");
// Shared upstream revision for the source-only fetch target and static-SYCL admission.
const syclRevision = "da2463e45aa90aa36306c45abcfc05b87de51bc6";
const backend = process.env["MOM_AOT_SYCL_BACKEND"] || "omp";

/** @param {string} command @param {string[]} args @param {string} log @param {number} [status] */
function run(command, args, log, status = 0) {
  const result = spawnSync(command, args, {encoding: "utf8", timeout: 120000, maxBuffer: 4 << 20,
    env: {...process.env, ACPP_VISIBILITY_MASK: backend, OMP_NUM_THREADS: "2"}});
  fs.writeFileSync(log, (result.stdout || "") + (result.stderr || ""));
  if (result.error || result.signal || result.status !== status) {throw new Error(`Static SYCL probe failed; see ${log}`, {cause: result.error});}
  return result.stdout;
}

/** A build-only ROCm prefix, never a runtime search path. HSA's upstream aliases require its
 * entire archive exactly once. Reject thin/shared libraries and executable linker directives.
 * The optional component argument keeps primitive SDK inspection available; complete HIP
 * builders below REQUIRE a verified source receipt instead of the legacy prefix archive.
 * @param {string | undefined} root @param {string} [receipt] */
function rocmInputs(root, receipt) {
  if (!root || !path.isAbsolute(root)) {throw new Error("Static ROCm SDK root must be absolute");}
  const comgr = receipt ? comgrSdkInputs(receipt) : undefined;
  const include = path.join(root, "include");
  const libraries = ["amdhip64", "hiprtc", "amd_comgr", "hsakmt-staticdrm", "elf", "drm_amdgpu", "drm", "numa"]
    .map((name) => name === "amd_comgr" && comgr ? comgr.archive : path.join(root, "lib", `lib${name}.a`));
  const hsa = path.join(root, "lib/libhsa-runtime64.a"), aliases = path.join(root, "lib/cmake/hsa-runtime64/hsacore.so.link");
  for (const file of [...libraries, hsa]) {
    if (!fs.existsSync(file)) {throw new Error(`Missing static ROCm archive: ${file}`);}
    if (!isStaticArchive(file)) {throw new Error(`Invalid static ROCm archive: ${file}`);}
  }
  if (["hip_runtime_api.h", "hiprtc.h"].some((name) => !fs.existsSync(path.join(include, "hip", name)))) {
    throw new Error("Missing static ROCm SDK headers");
  }
  if (!fs.existsSync(aliases) || !/^(?:hsa_[a-z0-9_]+ = hsa_[a-z0-9_]+;\r?\n)+$/.test(fs.readFileSync(aliases, "utf8"))) {
    throw new Error("Missing or unexpected HSA alias script");
  }
  // ROCr queries this statically linked libdrm entry through dlsym(RTLD_DEFAULT).
  // Export only that entry: omitting it silently disables VM support and managed USM.
  return {include, libraries, comgr, links: ["-Wl,--whole-archive", hsa, "-Wl,--no-whole-archive", "-Wl," + aliases,
    "-Wl,--export-dynamic-symbol=amdgpu_device_get_fd"]};
}

/** The SDK producer and consumers share one upstream revision/patch admission boundary.
 * This never patches a tree or silently repairs an unreviewed SDK checkout.
 * @param {string} sourceArg @param {string} output @param {boolean} gpu
 * @param {(command: string, args: string[], log: string) => string} [execute] */
function admitSyclSource(sourceArg, output, gpu, execute = run) {
  const source = path.resolve(sourceArg), repo = path.resolve(__dirname, "..");
  // Git accepts empty context lines with or without their single-space marker. A matching
  // tracked diff alone misses extra (even ignored) headers/CMake files. Builds stay outside
  // this pinned checkout; reject all untracked entries without traversing their directories.
  const canonical = (/** @type {string} */ patch) => patch.replace(/^ $/gm, "").split(/(?=^diff --git )/m).filter(Boolean).sort().join("");
  if (execute("git", ["--no-optional-locks", "-C", source, "rev-parse", "HEAD"], path.join(output, "revision.log")).trim() !== syclRevision ||
      canonical(execute("git", ["--no-optional-locks", "-C", source, "diff", "--no-ext-diff", "--no-textconv", "HEAD"], path.join(output, "runtime-patch.log"))) !==
      canonical(["adaptivecpp-static.patch", ...(gpu ? ["adaptivecpp-jit.patch"] : [])].map((file) => fs.readFileSync(path.join(repo, "runtime", file), "utf8")).join("")) ||
      execute("git", ["--no-optional-locks", "-C", source, "ls-files", "--others", "--directory", "--"], path.join(output, "untracked.log")).length !== 0) {
    throw new Error("Use the pinned AdaptiveCpp source with only the reviewed runtime/JIT patches");
  }
  return {source, revision: syclRevision};
}

/** CUDA runtime headers and libraries are one ABI. Keep the legacy-capable12.6 runtime
 * separate from newer isolated NVRTC; preprocessing verifies the actual selected headers,
 * not a directory name or nvcc banner. This never loads a driver or queries a device.
 * @param {string} cache @param {string} cxx
 * @param {(command: string, args: string[], label: string) => string} execute */
function cudaRuntimeInputs(cache, cxx, execute) {
  const libraries = ["cudart_static", "culibos"].map(name => {
    const file = cache.match(new RegExp(`^CUDA_${name}_LIBRARY:FILEPATH=(.+)$`, "m"))?.[1];
    if (!file || !path.isAbsolute(file) || !isStaticArchive(file)) {throw new Error("Missing regular static CUDA library " + name);}
    return fs.realpathSync(file);
  });
  const directory = path.dirname(libraries[0] || "");
  if (libraries.some(file => path.dirname(file) !== directory)) {throw new Error("CUDA runtime libraries must belong to one SDK");}
  const include = path.resolve(directory, "../include");
  const macros = execute(cxx, ["-E", "-dM", "-x", "c++", "-I", include, "-include", "cuda.h", "-include", "cuda_runtime_api.h", "/dev/null"], "cuda-runtime-headers.log");
  for (const name of ["CUDA_VERSION", "CUDART_VERSION"]) {
    const definitions = [...macros.matchAll(new RegExp(`^#define ${name} ([0-9]+)$`, "gm"))];
    if (definitions.length !== 1 || definitions[0]?.[1] !== "12060") {throw new Error("Native CUDA runtime requires matching CUDA12.6 headers and static libraries; keep modern NVRTC isolated");}
  }
  return {include, libraries, version: 12060};
}

/** A GPU consumer requires its selected backend. The other vendor backend may be present in
 * a shared cache; the combined consumer explicitly requires both. Each consumer below still
 * selects only its own generated backend archives and compiler components.
 * @param {string} cache @param {string} backend */
function validateStaticSyclBackendCache(cache, backend) {
  if (!["omp", "cuda", "hip", "cuda-hip"].includes(backend)) {throw new Error("Unknown static SYCL backend");}
  const selected = backend === "omp" ? [] : backend.split("-").map(name => name === "cuda" ? "CUDA" : "ROCM"),
    disabled = backend === "omp" ? ["CUDA", "ROCM", "OPENCL", "LEVEL_ZERO", "VULKAN"] : ["OPENCL", "LEVEL_ZERO", "VULKAN"];
  if (["CUDA", "ROCM"].some(name => !new RegExp(`^WITH_${name}_BACKEND:(?:BOOL|UNINITIALIZED)=(?:ON|OFF)$`, "m").test(cache)) ||
      selected.some(name => !new RegExp(`^WITH_${name}_BACKEND:BOOL=ON$`, "m").test(cache)) ||
      disabled.some(name => !new RegExp(`^WITH_${name}_BACKEND:(?:BOOL|UNINITIALIZED)=OFF$`, "m").test(cache))) {
    throw new Error("Use an isolated CPU-only build, or the explicitly selected GPU+host-support SSCP build");
  }
}

/** Reviewed static-runtime/compiler inputs, shared by primitive and complete Core builds.
 * This function builds nothing and never queries a GPU.
 * @param {string} sourceArg @param {string} buildArg @param {string} output @param {string} backend
 * @param {NodeJS.ProcessEnv} [environment] */
function staticSyclInputs(sourceArg, buildArg, output, backend, environment = process.env) {
  if (!["omp", "cuda", "hip", "cuda-hip"].includes(backend)) {throw new Error("Unknown static SYCL backend");}
  const backends = backend.split("-"), gpu = backend !== "omp", hip = backends.includes("hip"), cuda = backends.includes("cuda");
  if (hip && !environment["MOM_AOT_COMGR_RECEIPT"]) {throw new Error("Native HIP requires MOM_AOT_COMGR_RECEIPT from the pinned source producer");}
  if (gpu && !environment["MOM_AOT_OPENMP_LIBRARY"]) {
    throw new Error("Set MOM_AOT_OPENMP_LIBRARY to the static LLVM OpenMP archive required by the GPU host-support backend");
  }
  const {source, revision} = admitSyclSource(sourceArg, output, gpu);
  const build = path.resolve(buildArg), repo = path.resolve(__dirname, "..");
  const cache = fs.readFileSync(path.join(build, "CMakeCache.txt"), "utf8");
  const configuredSource = cache.match(/^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$/m)?.[1];
  if (!configuredSource || fs.realpathSync(configuredSource) !== fs.realpathSync(source)) {
    throw new Error("Static runtime build must match the reviewed source tree");
  }
  if (!/^ACPP_STATIC_RUNTIME:BOOL=(?:ON|TRUE|1)$/mi.test(cache) ||
      (gpu ? !/^ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER:INTERNAL=(?:ON|TRUE|1)$/mi.test(cache) :
        !/^ACPP_COMPILER_FEATURE_PROFILE:STRING=none$/m.test(cache))) {
    throw new Error("Use an isolated CPU-only build, or the explicitly selected GPU+host-support SSCP build");
  }
  validateStaticSyclBackendCache(cache, backend);
  if (gpu) {
    const generator = cache.match(/^ACPP_RESOURCE_GENERATOR:FILEPATH=(.+)$/m)?.[1];
    if (!generator || fs.realpathSync(generator) !== fs.realpathSync(path.join(repo, "scripts/native-resources.js")) ||
        ["STDPAR_COMPILER", "ACCELERATED_CPU"].some((feature) =>
          !new RegExp(`^ACPP_CUSTOM_PROFILE_WITH_${feature}:INTERNAL=(?:OFF|FALSE|0)$`, "mi").test(cache))) {
      throw new Error("Use the reviewed immutable resource generator and SSCP-only compiler profile");
    }
  }
  let cxx = environment["CXX"] || "c++";
  const openmp = environment["MOM_AOT_OPENMP_LIBRARY"] ||
    run(cxx, ["-print-file-name=libgomp.a"], path.join(output, "openmp.log")).trim();
  if (!path.isAbsolute(openmp) || !fs.existsSync(openmp)) {throw new Error("A static OpenMP archive is required");}
  const numa = cache.match(/^NUMA_LIBRARY:FILEPATH=(.+)$/m)?.[1];
  const libraries = [path.join(build, "src/runtime/libacpp-rt.a"), ...backends.map((name) => path.join(build, `src/runtime/librt-backend-${name}.a`)),
    path.join(build, "src/common/libacpp-common.a"), openmp,
    ...(numa && !numa.endsWith("-NOTFOUND") ? [numa.replace(/\.so(?:\..*)?$/, ".a")] : [])];
  let flags = ["-D__ACPP_ENABLE_OMPHOST_TARGET__", "-fopenmp"], links = ["-ldl", "-lpthread", "-lm"];
  /** @type {ReturnType<typeof comgrSdkInputs> | undefined} */
  let comgr;
  if (gpu) {
    const config = environment["LLVM_CONFIG"] || "/usr/lib/llvm-21/bin/llvm-config";
    const tools = run(config, ["--bindir"], path.join(output, "llvm-tools.log")).trim();
    const lib = run(config, ["--libdir"], path.join(output, "llvm-lib.log")).trim();
    const components = run(config, ["--link-static", "--libs", ...(hip ? ["all"] :
      ["core", "support", "irreader", "passes", "nvptxcodegen", "nativecodegen", "orcjit"])], path.join(output, "llvm-components.log")).trim().split(/\s+/);
    cxx = path.join(tools, "clang++");
    const plugin = path.join(build, "src/compiler/libacpp-clang.so");
    // The real runtime archives use Clang ThinLTO; bind the matching linker, not host BFD.
    flags = [`--ld-path=${path.join(tools, "ld.lld")}`, "-Wl,--threads=4", "-Wl,--thinlto-jobs=4", "-Wl,--wrap=dlopen", "-Wl,--wrap=dlerror",
      "-D__ACPP_ENABLE_LLVM_SSCP_TARGET__", "-include", path.join(source, "include/hipSYCL/glue/llvm-sscp/hcf_registration.hpp"),
      `-fplugin=${plugin}`, `-fpass-plugin=${plugin}`, "-Xclang", "-disable-O0-optnone", "-mllvm", "-acpp-sscp"];
    libraries.push(path.join(build, "src/runtime/librt-backend-omp.a"),
      ...[...(hip ? ["llvm-to-amdgpu"] : []), ...(cuda ? ["llvm-to-ptx"] : []), "llvm-to-host", "llvm-to-backend"]
        .map((name) => path.join(build, `src/compiler/llvm-to-backend/lib${name}.a`)));
    if (hip) {
      const rocm = rocmInputs(environment["MOM_AOT_ROCM_STATIC_ROOT"], environment["MOM_AOT_COMGR_RECEIPT"]);
      comgr = rocm.comgr;
      flags.push("-D__HIP_PLATFORM_AMD__", "-isystem", rocm.include);
      libraries.push(...rocm.libraries);
      const version = run(config, ["--version"], path.join(output, "llvm-version.log")).trim();
      const lldRoot = environment["MOM_AOT_LLD_ROOT"] || path.resolve(cache.match(/^LLD_DIR:[^=]+=(.+)$/m)?.[1] || "/unavailable", "../../..");
      verifyComgrSdkSelection(comgr, path.dirname(tools), lldRoot,
        cache.match(/^ROCM_DEVICE_LIBS_PATH:[^=]+=(.+)$/m)?.[1] || "/unavailable", version);
      libraries.push(...lldInputs(lldRoot, version), ...fs.readdirSync(lib).filter((name) => /^libclang.*\.a$/.test(name)).map((name) => path.join(lib, name)));
      if (!libraries.some((file) => file.endsWith("/liblldELF.a"))) {throw new Error("Static AMD JIT requires matching LLD archives");}
      // These are link inputs only: do not repeat HSA as an ordinary archive, and never add
      // ROCm SDK rpaths. LLVM/Clang and HIPRTC share the already-qualified compiler version.
      links.push(...rocm.links);
    }
    if (cuda) {
      libraries.push(...cudaRuntimeInputs(cache, cxx, (command, args, label) => run(command, args, path.join(output, label))).libraries);
    }
    // GPU runtimes resolve their system drivers only when needed; do not import a vendor driver
    // into the executable, so machines without that GPU can still start the combined image.
    links = ["-L", lib, ...components, ...links, "-Wl,-Bstatic", "-lz", "-lzstd", "-Wl,-Bdynamic", "-ldl", "-lpthread", "-lm"];
  }
  for (const file of libraries) {if (!fs.existsSync(file)) {throw new Error(`Missing static archive ${file}`);}}
  return {source, build, revision, cxx, flags, libraries, links, comgr};
}

/** Count unique named-kernel globals from bounded readelf symbol output.
 * @param {string} text */
function countHcfKernelGlobals(text) {
  if (Buffer.byteLength(text) > (1 << 20)) {throw new Error("Excessive HCF symbol metadata");}
  const suffix = "__acpp_sscp_kernel_name.initialized", names = new Set();
  for (const line of text.split("\n")) {
    const match = line.match(/^\s*\d+:\s+[a-f0-9]+\s+\d+\s+OBJECT\s+\S+\s+DEFAULT\s+\S+\s+(\S+)$/i);
    if (match?.[1]?.endsWith(suffix)) {names.add(match[1]);}
  }
  return names.size;
}

/** @typedef {{name: string, value: number, object: string, depfile: string, log: string, cxx: string, args: string[]}} HcfReproPlan */
/** @typedef {{name: string, id?: string, normalizedHeaderSha256?: string, deviceIRSha256?: string, objectSha256?: string,
 * kernelNameGlobals?: number, status?: string, log?: string, error?: string}} HcfReproRecord */
/** @typedef {"id" | "normalizedHeaderSha256" | "deviceIRSha256" | "objectSha256"} HcfReproDigestField */

/** Stable three-object compile plan for the HIP build-only reproducibility regression.
 * @param {string} cxx @param {string[]} flags @param {string} source @param {string} build
 * @param {string} output @param {string} repo @returns {HcfReproPlan[]} */
function hcfReproPlan(cxx, flags, source, build, output, repo) {
  const fixture = path.join(repo, "tests/reference/native-sycl-hcf-repro.cpp"),
    compilerFlags = flags.filter(flag => !flag.startsWith("-Wl,") && !flag.startsWith("--ld-path=")),
    common = ["-std=c++17", "-O2", "-fPIC", "-ffunction-sections", "-fdata-sections", "-DACPP_STATIC_RUNTIME", "-D__ACPP__",
      ...compilerFlags, "-isystem", path.join(source, "include"), "-isystem", path.join(build, "include")];
  return [
    {name: "same-a", value: 1}, {name: "same-b", value: 1}, {name: "payload-v2", value: 2},
  ].map(({name, value}) => {
    const object = path.join(output, `${name}.o`), depfile = path.join(output, `${name}.d`), log = path.join(output, `${name}.log`);
    return {name, value, object, depfile, log, cxx, args: [...common, `-DHCF_PROOF_VALUE=${value}`, "-mllvm",
      "-acpp-sscp-hcf-object-id-namespace=acpp-hip", "-MD", "-MF", depfile, "-c", fixture, "-o", object]};
  });
}

/** Compare the two same-input objects strictly while allowing changed payload metadata to grow.
 * @param {HcfReproRecord[]} records */
function hcfReproFailures(records) {
  const failures = [], byName = new Map(records.map(record => [record.name, record]));
  if (records.length !== 3 || byName.size !== 3 || !byName.get("same-a") || !byName.get("same-b") || !byName.get("payload-v2")) {
    return ["HCF regression did not produce all three object records"];
  }
  const a = byName.get("same-a"), b = byName.get("same-b"), changed = byName.get("payload-v2");
  if (!a || !b || !changed) {return ["HCF regression did not produce all three object records"];}
  if ([a, b, changed].some(record => typeof record.kernelNameGlobals !== "number" ||
      !Number.isSafeInteger(record.kernelNameGlobals) || record.kernelNameGlobals < 20)) {
    failures.push("HCF fixture emitted fewer than 20 named kernel globals");
  }
  /** @type {HcfReproDigestField[]} */
  const identicalFields = ["id", "normalizedHeaderSha256", "deviceIRSha256", "objectSha256"];
  for (const field of identicalFields) {
    if (a[field] !== b[field]) {failures.push(`identical HCF inputs differ in ${field}`);}
  }
  /** @type {HcfReproDigestField[]} */
  const changedFields = ["id", "deviceIRSha256", "objectSha256"];
  for (const field of changedFields) {
    if (a[field] === changed[field]) {failures.push(`changed HCF payload did not change ${field}`);}
  }
  return failures;
}

/** Compile and inspect a fixture with the admitted plugin; this never links or executes it.
 * @param {string} cxx @param {string[]} flags @param {string} source @param {string} build
 * @param {string} output @param {string} repo */
function runHcfRepro(cxx, flags, source, build, output, repo) {
  const fixture = path.join(repo, "tests/reference/native-sycl-hcf-repro.cpp");
  const plugin = path.join(build, "src/compiler/libacpp-clang.so");
  const reader = path.join(__dirname, "native-hcf.js");
  const compilerSha256 = fileHash(cxx), fixtureSha256 = fileHash(fixture), pluginSha256 = fileHash(plugin), readerSha256 = fileHash(reader);
  const plans = hcfReproPlan(cxx, flags, source, build, output, repo), records = /** @type {HcfReproRecord[]} */ ([]), failures = [];
  for (const plan of plans) {
    try {
      run(cxx, plan.args, plan.log);
      const hcf = inspectHcfObject(plan.object);
      const kernelNameGlobals = countHcfKernelGlobals(elfReport(plan.object, "--symbols"));
      records.push({name: plan.name, ...hcf, kernelNameGlobals});
    } catch (error) {
      failures.push(`${plan.name}: ${error instanceof Error ? error.message : String(error)}`);
      records.push({name: plan.name, status: "failed", log: plan.log,
        error: error instanceof Error ? error.message : String(error)});
    }
  }
  const complete = records.filter(record => record.status !== "failed");
  failures.push(...hcfReproFailures(complete));
  const result = {status: failures.length ? "FAIL" : "PASS", backend: "hip", namespace: "acpp-hip",
    compiler: cxx, compilerSha256, fixture, fixtureSha256, plugin, pluginSha256, reader, readerSha256, failures, cases: records};
  if (fileHash(cxx) !== compilerSha256 || fileHash(fixture) !== fixtureSha256 || fileHash(plugin) !== pluginSha256 || fileHash(reader) !== readerSha256) {
    result.status = "FAIL"; failures.push("compiler, fixture, plugin, or HCF reader changed during regression");
  }
  const resultPath = path.join(output, "hcf-object-repro.json");
  fs.writeFileSync(resultPath, JSON.stringify(result, null, 2) + "\n");
  if (failures.length) {throw new Error(`HIP HCF object reproducibility failed; see ${resultPath}`);}
  return result;
}

/** Exercise only CPU storage code from the actual common archive; no GPU runtime is opened.
 * The explicit root keeps generated databases out of the caller's normal JIT cache.
 * @param {string} cxx @param {string[]} flags @param {string} source @param {string} build
 * @param {string} output @param {string} repo */
function runJitCacheIdentity(cxx, flags, source, build, output, repo) {
  const executable = path.join(output, "cache-identity"), root = path.join(output, "cache-identity-data");
  run(cxx, ["-std=c++17", "-O1", "-ffunction-sections", "-fdata-sections",
    ...flags.filter(flag => flag.startsWith("--ld-path=")),
    "-isystem", path.join(source, "include"), "-isystem", path.join(build, "include"),
    path.join(repo, "tests/reference/native-jit-cache.cpp"), path.join(build, "src/common/libacpp-common.a"),
    "-Wl,--gc-sections", "-pthread", "-ldl", "-o", executable], path.join(output, "cache-identity-build.log"));
  const records = Object.fromEntries(["untagged", "a", "b", "a-repeat"].map(name => [name,
    run(executable, [name === "a-repeat" ? "a" : name, root], path.join(output, `cache-identity-${name}.log`)).trim()]));
  if (records["a"] !== records["a-repeat"] || records["a"] === records["b"] || records["a"] === records["untagged"]) {
    throw new Error("Native JIT cache identities were not isolated and repeatable");
  }
  return {status: "PASS", cases: 4, scope: "CPU-only persistent storage; no GPU execution"};
}

function main() {
  const [sourceArg, buildArg, outputArg] = process.argv.slice(2);
  const gpu = backend !== "omp", buildOnly = process.env["MOM_AOT_BUILD_ONLY"] === "1";
  if (!["omp", "cuda", "hip", "cuda-hip"].includes(backend) || (gpu && !buildOnly)) {
    throw new Error("GPU probes require MOM_AOT_SYCL_BACKEND=cuda|hip|cuda-hip MOM_AOT_BUILD_ONLY=1; verify fresh ownership before executing the built artifact");
  }
  if (process.platform !== "linux" || !sourceArg || !buildArg || process.argv.length > 5) {
    throw new Error("Usage (Linux): node scripts/native-sycl-inputs.js <AdaptiveCpp source> <static build> [evidence-parent]");
  }
  const repo = path.resolve(__dirname, "..");
  const output = fs.mkdtempSync(path.join(path.resolve(outputArg || os.tmpdir()), "mom-sycl-"));
  console.log(`Static SYCL evidence: ${output}`);
  const {source, build, revision, cxx, flags, libraries, links} = staticSyclInputs(sourceArg, buildArg, output, backend);
  // Device IR uses the LLVM plugin ABI; the common archive uses its configured host compiler.
  const hostCxx = fs.readFileSync(path.join(build, "CMakeCache.txt"), "utf8")
    .match(/^CMAKE_CXX_COMPILER:[^=]+=(.+)$/m)?.[1];
  if (!hostCxx || !path.isAbsolute(hostCxx)) {throw new Error("Missing configured SDK host compiler for CPU cache proof");}
  const cacheIdentity = runJitCacheIdentity(hostCxx, flags, source, build, output, repo);
  const hcfObjectRepro = backend === "hip" ? runHcfRepro(cxx, flags, source, build, output, repo) : undefined;
  const executable = path.join(output, "static-sycl");
  run(cxx, ["-std=c++17", "-O2", "-fPIC", "-ffunction-sections", "-fdata-sections", "-DACPP_STATIC_RUNTIME", "-D__ACPP__",
    ...(gpu ? [backend === "cuda-hip" ? "-DMOM_NATIVE_GPU_PROBE" : backend === "hip" ? "-DMOM_NATIVE_HIP_PROBE" : "-DMOM_NATIVE_CUDA_PROBE"] : []),
    ...flags, "-isystem", path.join(source, "include"), "-isystem", path.join(build, "include"),
    path.join(repo, "tests/reference/native-sycl-static.cpp"), ...(gpu ? [path.join(repo, "runtime/driver-library.cpp"), path.join(repo, "runtime/compiler-gate.cpp")] : []),
    "-pie", "-Wl,--gc-sections", "-static-libstdc++", "-static-libgcc",
    "-Wl,--start-group", ...libraries, ...links, "-Wl,--end-group", "-o", executable], path.join(output, "build.log"));
  if (buildOnly) {
    const needed = nativeDependencies(run("readelf", ["-d", executable], path.join(output, "dependencies.log")), {cudaDriver: gpu});
    fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({revision, needed, bytes: fs.statSync(executable).size, backend, buildOnly: true,
      hcfObjectRepro, cacheIdentity,
      scope: "Build/dependency proof only; not executed, device correctness/performance unqualified"}, null, 2) + "\n");
    console.log("Static SYCL artifact built; NOT executed. Verify ownership before a bounded GPU run.");
    return;
  }
  const transcript = run(executable, [], path.join(output, "correctness.log"));
  if (!transcript.includes("BLAKE3 parity passed")) {throw new Error("Static runtime did not reach kernel verification");}
  run(executable, ["mismatch"], path.join(output, "mismatch.log"), 1);
  const dependencies = run("readelf", ["-d", executable], path.join(output, "dependencies.log"));
  const needed = nativeDependencies(dependencies);
  const symbols = run("nm", ["-C", executable], path.join(output, "symbols.log"));
  // libgomp contains optional offload-loader functions even in a statically linked CPU lane.
  // Their presence is not an app-library dependency; retain it as an unresolved pruning gate.
  // The exact source patch removes AdaptiveCpp's loader; syscall/clean-container checks must
  // independently verify which files this particular lane actually loads at execution time.
  const optionalLoaderSymbols = /\b(dlopen|dlsym|LoadLibrary|GetProcAddress)\b/.test(symbols);
  const syscalls = path.join(output, "syscalls.log");
  run("strace", ["-f", "-s", "1024", "-e", "trace=openat,execve,execveat", "-E", "PATH=/unavailable", "-E", "LD_LIBRARY_PATH=/unavailable",
    "-o", syscalls, executable], path.join(output, "isolated-paths.log"));
  auditNativeTrace(fs.readFileSync(syscalls, "utf8"), executable);
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({revision, needed, bytes: fs.statSync(executable).size,
    cases: 10, backend: "omp", optionalLoaderSymbols, cacheIdentity,
    scope: "Static runtime/registration/USM/queue/event/device primitive; no GPU images or JIT"}, null, 2) + "\n");
  console.log("Static AdaptiveCpp runtime and maintained SYCL device primitive passed; GPU/JIT integration remains unqualified.");
}

module.exports = {staticSyclInputs, rocmInputs, admitSyclSource, cudaRuntimeInputs, validateStaticSyclBackendCache, countHcfKernelGlobals,
  hcfReproPlan, hcfReproFailures, runHcfRepro, runJitCacheIdentity, syclRevision};
if (require.main === module) {try {main();} catch (error) {console.error(error); process.exitCode = 1;}}
