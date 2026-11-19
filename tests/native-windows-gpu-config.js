"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), path = require("node:path"), fs = require("node:fs"),
  os = require("node:os"), {createHash} = require("node:crypto");
const gpuConfig = require("../scripts/native-windows-gpu-config"),
  {windowsNativeOneapiIncludePlan, windowsNativeGpuIncludePlan, resolveDpcppCompiler, windowsNativeGpuCudaImages, windowsNativeGpuConfig, windowsNativeAcppInputs,
    windowsNativeAcppGpuConfig} = gpuConfig;
const {validateWindowsGpuConfig} = require("../scripts/build-native-windows-gpu");
const cudaImages = require("../scripts/native-cuda-images"), childProcess = require("node:child_process"),
  windowsHostSdkReader = require("../scripts/native-windows-host-sdk-inputs"), linuxLlvmToolReader = require("../scripts/native-linux-llvm-tools");
const {pins} = require("../scripts/native-dpcpp-pins");
const preparedSdkSources = require("../scripts/prepare-native-sdk-sources"), ciTar = require("../scripts/native-ci-tar"),
  dpcppInputs = require("../scripts/native-dpcpp-cuda-inputs"),
  {syclRevision} = require("../scripts/native-sycl-inputs");
const {cudaHeaderResources} = require("../scripts/native-jit-headers"),
  {planCudaImages, generateCudaImageBundle} = cudaImages;

const sdkRoots = ["/mom-sdk/msvc/include", "/mom-sdk/sdk/include/ucrt", "/mom-sdk/sdk/include/shared",
  "/mom-sdk/sdk/include/um", "/mom-sdk/sdk/include/winrt"];
/** @returns {import("../scripts/native-windows-gpu-config").WindowsOneapiIncludePlanInput} */
function input() {
  return {resourceInclude: "/opt/intel/compiler/lib/clang/23/include", syclInclude: "/src/prepared/dpcpp/sycl/include",
    dpcppInclude: "/sdk/dpcpp/include", dpcppSyclInclude: "/sdk/dpcpp/include/sycl", urInclude: "/src/prepared/dpcpp/unified-runtime/include",
    windowsSdkIncludeRoots: [...sdkRoots]};
}

/** @param {string} file @returns {string} */
function sha256(file) {return createHash("sha256").update(fs.readFileSync(file)).digest("hex");}

let cudaImageFixtureCounter = 0;
/** @param {import("node:test").TestContext} t @param {{suffix?:string,runnerTemp?:string}} [options] */
function cudaImagesFixture(t, options = {}) {
  const suffix = options.suffix || "ci-" + String(Date.now()) + "-" + String(++cudaImageFixtureCounter),
    ownsRunnerTemp = !options.runnerTemp, runnerTemp = options.runnerTemp || fs.mkdtempSync(path.join("/var/tmp", "mom-windows-cuda-images-ci-")),
    incoming = path.join(runnerTemp, "incoming"),
    root = path.join("/tmp", "mom-native-cuda-images-handoff-" + suffix),
    archive = path.join(incoming, "native-linux-cuda-images-" + suffix + ".tar"), include = path.join(root, "cuda-images.inc"),
    sidecar = include + ".json", resultFile = path.join(root, "result.json"), originalRunnerTemp = process.env["RUNNER_TEMP"];
  if (!fs.existsSync(incoming)) {fs.mkdirSync(incoming, {recursive: true});}
  fs.mkdirSync(root);
  fs.writeFileSync(include, "image include fixture\n", {flag: "wx"});
  fs.writeFileSync(sidecar, "sidecar fixture\n", {flag: "wx"});
  const includeSha = sha256(include), sidecarSha = sha256(sidecar), manifest = {
      sourceSha256: includeSha, images: [{}], resourceBytes: Buffer.byteLength("image include fixture\n"),
    }, producerInclude = path.join("/tmp", "mom-native-cuda-images-" + suffix, "bundle", "cuda-images.inc"),
    writeResult = (sourceSha256 = includeSha) => fs.writeFileSync(resultFile, JSON.stringify({format: "mom-cuda-image-build-v1",
      include: producerInclude, sidecar: producerInclude + ".json", compilerVersions: ["12.6", "13.3"], plannedImages: 1,
      embeddedBytes: manifest.resourceBytes, sourceSha256}) + "\n");
  writeResult();
  ciTar.packState(archive, "/tmp", [path.basename(root)]);
  if (ownsRunnerTemp) {process.env["RUNNER_TEMP"] = runnerTemp;}
  t.after(() => {
    if (ownsRunnerTemp) {
      if (originalRunnerTemp === undefined) {delete process.env["RUNNER_TEMP"];} else {process.env["RUNNER_TEMP"] = originalRunnerTemp;}
      fs.rmSync(runnerTemp, {recursive: true, force: true});
    }
    fs.rmSync(root, {recursive: true, force: true});
  });
  return {suffix, archive, root, include, sidecar, resultFile, includeSha, sidecarSha, manifest, writeResult,
    input: () => ({archive, bundleSha256: sha256(archive), root})};
}

/** Build a genuinely admissible tiny image bundle for testing the tar/restored-tree join.
 * @param {import("node:test").TestContext} t @param {boolean} [mutate]
 * @returns {{source:string,manifest:Record<string, any>}} */
function validCudaImageBundle(t, mutate = false) {
  const root = fs.mkdtempSync(path.join("/var/tmp", "mom-windows-cuda-valid-")), cutlass = path.join(root, "cutlass"), cuda = path.join(root, "cuda");
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  /** @param {string} file @param {string | Buffer} data */
  const write = (file, data) => {fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data, {flag: "wx"});};
  write(path.join(cutlass, "cute/tensor.hpp"), "// CUTE fixture\n");
  write(path.join(cuda, "cuda_runtime.h"), "// runtime fixture\n");
  write(path.join(cuda, "cccl/cuda/std/cstdint"), "// CCCL fixture\n");
  const headerPack = cudaHeaderResources([cutlass, cuda]), headerFile = path.join(root, "headers.inc"),
    headerManifest = Object.fromEntries(Object.entries(headerPack).filter(([key]) => key !== "source"));
  write(headerFile, headerPack.source); write(headerFile + ".json", JSON.stringify(headerManifest, null, 2) + "\n");
  const compilers = [
    {major: 12, minor: 6, supportedArchs: [80]},
    {major: 13, minor: 3, supportedArchs: [80, 120]},
  ].map((query) => {
    const inputs = /** @type {Record<string, string>} */ ({}), file = path.join(root, `nvrtc-${query.major}.${query.minor}.bin`);
    write(file, `NVRTC ${query.major}.${query.minor}\n`); inputs[file] = sha256(file);
    inputs[headerFile] = sha256(headerFile); inputs[headerFile + ".json"] = sha256(headerFile + ".json");
    return {...query, inputs};
  });
  const exportsByProfile = /** @type {Record<string, string[]>} */ ({
    "cn-expand": ["cn_gpu_expand_cuda"], "pearl-rank128": ["pearlhash_cuda_gemm", "pearlhash_cuda_hash"],
    "pearl-rank256": ["pearlhash_cuda_gemm", "pearlhash_cuda_hash"], "pearl-compact": ["compact_gemm", "selector_prepare"],
  });
  const profileSource = (/** @type {string} */ profile) => profile === "cn-expand" ? "sycl/cn-gpu/cuda-expand-kernel.inc" :
    profile === "pearl-compact" ? "sycl/pearlhash/cuda-compact-kernel.inc" : "sycl/pearlhash/cuda-kernel.inc";
  const program = (/** @type {string} */ profile) => profile === "cn-expand" ? "cn-gpu-expand.cu" : "pearlhash-cute.cu";
  const options = (/** @type {ReturnType<typeof planCudaImages>[number]} */ job, /** @type {string[]} */ includeRoots) => {
    const values = ["--std=c++17"], rank = job.profile === "pearl-rank128" || job.profile === "pearl-compact" ? 128 :
      job.profile === "pearl-rank256" ? 256 : undefined;
    if (rank) {values.push(`-DPEARLHASH_RANK=${rank}`);}
    values.push(`--gpu-architecture=${job.target}`); for (const include of includeRoots) {values.push(`--include-path=${include}`);}
    values.push("--device-as-default-execution-space"); if (rank) {values.push("--use_fast_math");}
    if (job.profile === "pearl-compact") {values.push("-DDIRECT_RANK_STRIDE=1");}
    return values;
  };
  /** @param {string[]} names */
  const cubin = (/** @type {string[]} */ names) => {
    const strings = [Buffer.from([0])], offsets = new Map(); let stringBytes = 1;
    for (const name of names) {offsets.set(name, stringBytes); strings.push(Buffer.from(name + "\0")); stringBytes += name.length + 1;}
    const stringTable = Buffer.concat(strings), symbols = Buffer.alloc((names.length + 1) * 24), sectionOffset = 64, sectionCount = 4,
      textOffset = sectionOffset + sectionCount * 64, stringsOffset = textOffset + 1, symbolsOffset = stringsOffset + stringTable.length,
      data = Buffer.alloc(symbolsOffset + symbols.length);
    data.set([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1], 0); data.writeUInt16LE(2, 16); data.writeUInt16LE(190, 18);
    data.writeUInt32LE(1, 20); data.writeBigUInt64LE(BigInt(sectionOffset), 40); data.writeUInt16LE(64, 52);
    data.writeUInt16LE(64, 58); data.writeUInt16LE(sectionCount, 60);
    const section = (/** @type {number} */ index, /** @type {number} */ type, /** @type {number} */ offset,
      /** @type {number} */ size, /** @type {number} */ link = 0, /** @type {number} */ entrySize = 0) => {
      const at = sectionOffset + index * 64; data.writeUInt32LE(type, at + 4); data.writeBigUInt64LE(BigInt(offset), at + 24);
      data.writeBigUInt64LE(BigInt(size), at + 32); data.writeUInt32LE(link, at + 40); data.writeBigUInt64LE(BigInt(entrySize), at + 56);
    };
    section(1, 1, textOffset, 1); section(2, 3, stringsOffset, stringTable.length); section(3, 2, symbolsOffset, symbols.length, 2, 24);
    for (const [index, name] of names.entries()) {const at = (index + 1) * 24; symbols.writeUInt32LE(offsets.get(name) || 0, at); symbols[at + 4] = 0x12; symbols.writeUInt16LE(1, at + 6);}
    data[textOffset] = 0; stringTable.copy(data, stringsOffset); symbols.copy(data, symbolsOffset); return data;
  };
  const plan = planCudaImages(compilers), artifacts = plan.map((/** @type {ReturnType<typeof planCudaImages>[number]} */ job) => {
    const exports = exportsByProfile[job.profile];
    if (!exports) {throw new Error("Unexpected CUDA profile in fixture plan");}
    const includeRoots = job.profile.startsWith("pearl-") ? headerPack.roots : [],
      ptxVersion = job.format === "ptx" ? job.compiler.major === 12 ? "8.5" : "9.3" : null;
    let data;
    if (job.format === "cubin") {data = cubin(exports);} else {
      const names = exports.map(name => `.visible .entry ${name}() { ret; }`), target = `sm_${job.target.match(/\d+/)?.[0]}`;
      data = Buffer.from([`// NVRTC fixture\n.version ${ptxVersion}\n.target ${target}\n.address_size 64`, ...names].join("\n") + "\n\0", "ascii");
    }
    return {job, data, metadata: {status: "ok", nvrtc: {major: job.compiler.major, minor: job.compiler.minor},
      supportedArchs: job.compiler.supportedArchs, profile: job.profile, source: profileSource(job.profile), program: program(job.profile),
      target: job.target, format: job.format, ptxVersion, includeRoots, options: options(job, includeRoots), exports, outputBytes: data.length}};
  });
  if (mutate) {
    const selected = artifacts.find((/** @type {(typeof artifacts)[number]} */ artifact) => artifact.job.format === "ptx");
    assert.ok(selected); selected.data = Buffer.from(selected.data.toString("ascii").replace("NVRTC fixture", "NVRTC changed fixture"), "ascii");
    selected.metadata.outputBytes = selected.data.length;
  }
  const bundle = generateCudaImageBundle({repo: path.resolve(__dirname, ".."), compilers, artifacts,
    headerPacks: {"12.6": headerFile, "13.3": headerFile}});
  return bundle;
}

/** @param {string} suffix @param {string} runnerTemp */
function windowsSdkFixture(suffix, runnerTemp) {
  const root = path.join(runnerTemp, "mom-native-windows-host-sdk-" + suffix, "native-windows-host-sdk-" + suffix),
    sdkRoot = path.join(root, "sdk"), overlayPath = path.join(root, "vfs-overlay.json"), files = new Map(), roots = [];
  for (const [index, virtual] of sdkRoots.entries()) {
    const relative = ["msvc/include", "sdk/include/ucrt", "sdk/include/shared", "sdk/include/um", "sdk/include/winrt"][index] + "/fixture.h",
      physical = path.join(sdkRoot, ...relative.split("/"));
    fs.mkdirSync(path.dirname(physical), {recursive: true}); fs.writeFileSync(physical, "#pragma once\n", {flag: "wx"});
    roots.push({type: "file", name: virtual + "/fixture.h", "external-contents": physical});
    files.set(relative.toLowerCase(), {path: relative, sha256: sha256(physical), bytes: fs.statSync(physical).size});
  }
  fs.mkdirSync(root, {recursive: true});
  fs.writeFileSync(overlayPath, JSON.stringify({"case-sensitive": false, roots}) + "\n", {flag: "wx"});
  const pins = {resultSha256: "1".repeat(64), descriptorSha256: "2".repeat(64), filesSha256: "3".repeat(64), overlaySha256: sha256(overlayPath)},
    sdk = {sourceRoot: sdkRoot, overlay: overlayPath, includeRoots: [...sdkRoots], libraryRoots: [path.join(sdkRoot, "msvc/lib")]};
  return {root, sdkRoot, overlayPath, files, pins, sdk,
    admitted: {root, sdkRoot, overlayPath, pins, sdk, files}};
}

/** @param {import("node:test").TestContext} t */
function acppGpuFixture(t) {
  const acpp = acppFixture(t), suffix = acpp.suffix, toolsRoot = path.join("/tmp", "mom-native-llvm-tools-fixture-" + suffix),
    frontendRoot = path.join("/tmp", "mom-native-llvm-acpp-frontend-fixture-" + suffix),
    windowsSdk = windowsSdkFixture(suffix, acpp.runnerTemp), originalTemp = process.env["RUNNER_TEMP"],
    hipRoot = path.join("/tmp", "native-rocm-hip-headers-" + suffix), hipInclude = path.join(hipRoot, "include");
  fs.mkdirSync(path.join(hipInclude, "hip"), {recursive: true});
  const hipHeader = path.join(hipInclude, "hip/hip_runtime_api.h"); fs.writeFileSync(hipHeader, "/* HIP fixture */\n");
  for (const root of [toolsRoot, frontendRoot]) {fs.mkdirSync(path.join(root, "bin"), {recursive: true});}
  /** @param {string} file @param {string} body */
  const addExecutable = (file, body) => fs.writeFileSync(file, body, {mode: 0o755, flag: "wx"}),
    clang21 = path.join(frontendRoot, "bin/clang-21"), clang = path.join(frontendRoot, "bin/clang"), clangxx = path.join(frontendRoot, "bin/clang++"),
    resourceInclude = path.join(frontendRoot, "lib/clang/21/include"), resourceHeader = path.join(resourceInclude, "stddef.h");
  addExecutable(clang21, "fixture clang executable\n"); fs.symlinkSync("clang-21", clang); fs.symlinkSync("clang", clangxx);
  fs.mkdirSync(resourceInclude, {recursive: true}); fs.writeFileSync(resourceHeader, "/* fixture resource */\n", {flag: "wx"});
  const toolFiles = /** @type {Record<string, any>} */ ({});
  for (const name of ["llvm-nm", "llvm-readobj", "llvm-objcopy"]) {
    const file = path.join(toolsRoot, "bin", name); addExecutable(file, "fixture " + name + "\n");
    toolFiles["bin/" + name] = {path: "bin/" + name, type: "file", sha256: sha256(file), bytes: fs.statSync(file).size, mode: 0o755};
  }
  const frontendFiles = /** @type {Record<string, any>} */ ({
    "bin/clang": {path: "bin/clang", type: "symlink", target: "clang-21"},
    "bin/clang++": {path: "bin/clang++", type: "symlink", target: "clang"},
    "bin/clang-21": {path: "bin/clang-21", type: "file", sha256: sha256(clang21), bytes: fs.statSync(clang21).size, mode: 0o755},
    "lib/clang/21/include/stddef.h": {path: "lib/clang/21/include/stddef.h", type: "file", sha256: sha256(resourceHeader),
      bytes: fs.statSync(resourceHeader).size, mode: 0o644},
  });
  const dpcppRoot = path.join("/tmp", "mom-native-dpcpp-cuda-inputs-" + suffix),
    linuxSource = path.join("/tmp", "mom-native-linux-source-prerequisites-" + suffix),
    dpcppState = {name: "mom-native-dpcpp-cuda-inputs-" + suffix, root: dpcppRoot, dpcppRoot: path.join(dpcppRoot, "dpcpp"),
      cudaRoot: path.join(dpcppRoot, "cuda"), manifest: path.join(dpcppRoot, "inputs.json"),
      linux: {source: linuxSource, sourceName: path.basename(linuxSource), prepared: path.join(linuxSource, "prepared")}},
    dpcppManifest = {source: {tarSha256: "5".repeat(64)}, dpcpp: {archiveSha256: "6".repeat(64)}};
  fs.mkdirSync(dpcppState.dpcppRoot, {recursive: true}); fs.mkdirSync(path.join(dpcppState.cudaRoot, "include"), {recursive: true});
  fs.writeFileSync(dpcppState.manifest, JSON.stringify(dpcppManifest) + "\n", {flag: "wx"}); fs.mkdirSync(linuxSource, {recursive: true});
  t.after(() => {
    for (const root of [windowsSdk.root, path.dirname(windowsSdk.root), toolsRoot, frontendRoot, dpcppRoot, linuxSource, hipRoot]) {
      fs.rmSync(root, {recursive: true, force: true});
    }
    if (process.env["RUNNER_TEMP"] !== originalTemp) {
      if (originalTemp === undefined) {delete process.env["RUNNER_TEMP"];} else {process.env["RUNNER_TEMP"] = originalTemp;}
    }
  });
  const expectations = {handoffSha256: "7".repeat(64)},
    input = {windowsSdkRoot: windowsSdk.root, windowsSdkPins: windowsSdk.pins,
      toolsRoot, toolsExpectation: expectations, frontendRoot, frontendExpectation: {...expectations, handoffSha256: "8".repeat(64)},
      acpp: acpp.input(), suffix};
  return {suffix, acpp, toolsRoot, frontendRoot, windowsSdk, clang, clangxx, resourceDir: path.join(frontendRoot, "lib/clang/21"),
    resourceInclude, toolFiles, frontendFiles, dpcppState, dpcppManifest, input, hipRoot, hipInclude, hipHeader,
    hipInput: {...input, hipHeaders: {root: hipRoot, handoffSha256: "9".repeat(64)}}};
}

/** @param {import("node:test").TestContext} t @param {ReturnType<typeof acppGpuFixture>} fixture
 * @param {{version?:string,resourceDir?:string}} [queryOptions] */
function mockAcppGpuReaders(t, fixture, queryOptions = {}) {
  t.mock.method(require("../scripts/native-rocm-hip-headers"), "rocmHipHeaderInputs",
    (/** @type {string} */ root, /** @type {any} */ expectation) => {
      assert.equal(root, fixture.hipRoot);
      assert.deepEqual(expectation, {handoffSha256: "9".repeat(64)});
      return {include: fixture.hipInclude, files: {[fixture.hipHeader]: sha256(fixture.hipHeader)},
        manifestSha256: "a".repeat(64), handoffSha256: expectation.handoffSha256};
    });
  t.mock.method(windowsHostSdkReader, "windowsHostSdkInputs", (/** @type {string} */ root, /** @type {any} */ pins) => {
    assert.equal(root, fixture.windowsSdk.root); assert.deepEqual(pins, fixture.windowsSdk.pins);
    return fixture.windowsSdk.admitted;
  });
  t.mock.method(linuxLlvmToolReader, "linuxLlvmToolHandoffInputs", (/** @type {string} */ root, /** @type {any} */ expectation) => {
    assert.equal(root, fixture.toolsRoot); assert.deepEqual(expectation, fixture.input.toolsExpectation);
    return {toolRoot: path.join(root, "bin"), llvmNm: path.join(root, "bin/llvm-nm"), files: fixture.toolFiles};
  });
  t.mock.method(linuxLlvmToolReader, "linuxLlvmAcppFrontendInputs", (/** @type {string} */ root, /** @type {any} */ expectation) => {
    assert.equal(root, fixture.frontendRoot); assert.deepEqual(expectation, fixture.input.frontendExpectation);
    return {clang: fixture.clang, clangxx: fixture.clangxx, resourceInclude: fixture.resourceInclude,
      sourceArchive: path.join(root, "llvm-project-21.1.8.src.tar.xz"), files: fixture.frontendFiles};
  });
  t.mock.method(preparedSdkSources, "verifyPreparedSdkSources", (/** @type {string} */ file) => JSON.parse(fs.readFileSync(file, "utf8")));
  const queries = /** @type {Array<{command:string,args:string[]}>} */ ([]);
  t.mock.method(childProcess, "spawnSync", (/** @type {string} */ command, /** @type {string[]} */ args, /** @type {any} */ options) => {
    queries.push({command, args});
    assert.equal(options.timeout, 10000); assert.equal(options.maxBuffer, 1 << 20);
    assert.equal(options.env.INCLUDE, undefined); assert.equal(options.env.LIB, undefined);
    return {error: undefined, signal: null, status: 0, stdout: args[0] === "--version" ?
      (queryOptions.version || "clang version 21.1.8\n" ) : (queryOptions.resourceDir || fixture.resourceDir) + "\n", stderr: ""};
  });
  return queries;
}

let acppFixtureCounter = 0;
/** @param {import("node:test").TestContext} t */
function acppFixture(t) {
  const suffix = "ci-" + String(Date.now()).slice(-11) + String(process.pid % 100000).padStart(5, "0") +
      String(++acppFixtureCounter).padStart(3, "0") + "-77",
    runnerTemp = fs.realpathSync(fs.mkdtempSync(path.join("/var/tmp", "mom-windows-acpp-ci-"))), incoming = path.join(runnerTemp, "incoming"),
    sdkName = "mom-native-acpp-cuda-hip-sdk-" + suffix, root = path.join("/tmp", sdkName), build = path.join(root, "build"),
    sourceState = path.join("/tmp", "mom-native-acpp-prerequisites-" + suffix), sourceRoot = path.join(sourceState, "prepared/acpp"),
    archive = path.join(incoming, "native-linux-acpp-cuda-hip-sdk-" + suffix + ".tar"),
    generatedInclude = path.join(build, "include"), configFile = path.join(generatedInclude, "hipSYCL/common/config.hpp"),
    plugin = path.join(build, "src/compiler/libacpp-clang.so"), sourceHeaderRelative = "include/hipSYCL/glue/llvm-sscp/hcf_registration.hpp",
    sourceHeader = path.join(sourceRoot, ...sourceHeaderRelative.split("/")), sourceReceiptFile = path.join(sourceState, "prepared/sources.json"),
    oldRunnerTemp = process.env["RUNNER_TEMP"];
  t.after(() => {
    if (oldRunnerTemp === undefined) {delete process.env["RUNNER_TEMP"];} else {process.env["RUNNER_TEMP"] = oldRunnerTemp;}
    fs.rmSync(root, {recursive: true, force: true}); fs.rmSync(sourceState, {recursive: true, force: true});
    fs.rmSync(runnerTemp, {recursive: true, force: true});
  });
  for (const selected of [root, sourceState]) {
    if (fs.existsSync(selected)) {throw new Error("ACPP fixture path unexpectedly exists");}
  }
  fs.mkdirSync(incoming); fs.mkdirSync(path.dirname(configFile), {recursive: true}); fs.mkdirSync(path.dirname(plugin), {recursive: true});
  fs.mkdirSync(path.dirname(sourceHeader), {recursive: true});
  fs.writeFileSync(plugin, "fixture ACPP plugin\n", {flag: "wx"});
  fs.writeFileSync(configFile, "#define MOM_ACPP_CONFIG_FIXTURE 1\n", {flag: "wx"});
  fs.writeFileSync(path.join(generatedInclude, "hipSYCL/common/generated.hpp"), "#pragma once\n", {flag: "wx"});
  fs.writeFileSync(sourceHeader, "#pragma once\n", {flag: "wx"});
  const sourceReceipt = {version: 2, target: "acpp", status: "PASS", compiled: false, buildOnly: true, publicationQualified: false,
    prepared: {acpp: {root: sourceRoot, revision: syclRevision, files: {[sourceHeaderRelative]: sha256(sourceHeader)}}}};
  fs.mkdirSync(path.dirname(sourceReceiptFile), {recursive: true});
  fs.writeFileSync(sourceReceiptFile, JSON.stringify(sourceReceipt) + "\n", {flag: "wx"});
  const inputs = {source: sourceRoot, build, cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx",
      flags: ["--ld-path=/llvm/bin/ld.lld", "-fplugin=" + plugin, "-fpass-plugin=" + plugin],
      libraries: [path.join(build, "src/runtime/libacpp-rt.a")], links: ["-ldl", "-lpthread"]},
    recipe = require("../scripts/native-release-sdk-inputs").currentStaticSyclRecipe(),
    fingerprints = {[inputs.cxx]: "3".repeat(64)}, hostCompilers = {cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx", cxx: inputs.cxx},
    hostCompilerInputs = {[hostCompilers.cc]: "4".repeat(64), [hostCompilers.cxx]: "5".repeat(64)},
    compilerVersions = {[hostCompilers.cc]: "Intel 2026.2", [hostCompilers.cxx]: "Intel 2026.2"};
  const result = {version: 1, backend: "cuda-hip", cudaVersion: 12060, revision: syclRevision, inputs, hostCompilers, hostCompilerInputs,
    compilerVersions, fingerprints, libraries: {[inputs.libraries[0] || ""]: "6".repeat(64)}, recipe, buildOnly: true, executed: false, publicationQualified: false};
  const plan = {version: "21.1.8", admitted: {source: sourceRoot, revision: syclRevision}, commands: [["cmake", "configure"], ["cmake", "build"]],
    hostCompilers, hostCompilerInputs, compilerVersions, fingerprints, recipe};
  const writeProducer = () => {
    fs.writeFileSync(path.join(root, "result.json"), JSON.stringify(result) + "\n", {flag: "wx"});
    fs.writeFileSync(path.join(root, "plan.json"), JSON.stringify(plan) + "\n", {flag: "wx"});
  };
  const rewriteProducer = () => {
    fs.writeFileSync(path.join(root, "result.json"), JSON.stringify(result) + "\n");
    fs.writeFileSync(path.join(root, "plan.json"), JSON.stringify(plan) + "\n");
  };
  const repack = () => {
    if (fs.existsSync(archive)) {fs.unlinkSync(archive);}
    ciTar.packState(archive, "/tmp", [sdkName]);
  };
  writeProducer(); repack(); process.env["RUNNER_TEMP"] = runnerTemp;
  return {suffix, runnerTemp, archive, root, build, sourceRoot, sourceReceiptFile, sourceHeader, sourceReceipt, result, plan, inputs,
    plugin, configFile, generatedInclude, repack, rewriteProducer,
    input: () => ({archive, bundleSha256: sha256(archive), root, sourceRoot})};
}

/** @returns {{root:string,record:import("../scripts/native-windows-gpu-config").DpcppCompilerRecord,version:string}} */
function dpcppFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-windows-dpcpp-config-")), version =
    `clang version 23.0.0git (https://github.com/intel/llvm ${pins.source})`;
  fs.mkdirSync(path.join(root, "bin")); fs.mkdirSync(path.join(root, "lib/clang/23/include"), {recursive: true});
  const marker = path.join(root, ".mom-toolchain-sha256"), clang = path.join(root, "bin/clang"), clangxx = path.join(root, "bin/clang++");
  fs.writeFileSync(marker, pins.sdk + "\n");
  fs.writeFileSync(path.join(root, "bin/clang-23"), "fixture compiler\n", {mode: 0o755});
  fs.symlinkSync("clang-23", clang); fs.symlinkSync("clang", clangxx);
  return {root, version, record: {root, archiveSha256: pins.sdk,
    marker: {path: marker, sha256: sha256(marker)}, clang: {path: clang, sha256: sha256(clang)},
    clangxx: {path: clangxx, sha256: sha256(clangxx)}, clangVersion: version}};
}

/** Reader-result fixtures for local construction: no CI path or run identity is supplied.
 * The tiny compiler only answers identity/resource queries; it never compiles or uses a GPU.
 * @param {import("node:test").TestContext} t */
function localDpcppConfigFixture(t) {
  const sdk = dpcppFixture(), root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-windows-gpu-config-")),
    sourceRoot = path.join(root, "source"), host = windowsSdkFixture("local", root).admitted,
    resourceInclude = path.join(sdk.root, "lib/clang/23/include");
  t.after(() => {fs.rmSync(sdk.root, {recursive: true, force: true}); fs.rmSync(root, {recursive: true, force: true});});
  const replies = {"--version": sdk.version, "--print-resource-dir": path.dirname(resourceInclude)};
  fs.writeFileSync(path.join(sdk.root, "bin/clang-23"), "#!" + process.execPath + "\nconst replies=" + JSON.stringify(replies) +
    ";\nif (!Object.hasOwn(replies,process.argv[2])) process.exit(1);\nconsole.log(replies[process.argv[2]]);\n");
  sdk.record.clang.sha256 = sha256(sdk.record.clang.path);
  sdk.record.clangxx.sha256 = sha256(sdk.record.clangxx.path);
  const plan = windowsNativeGpuIncludePlan({resourceInclude, syclInclude: path.join(sourceRoot, "sycl/include"),
    urInclude: path.join(sourceRoot, "unified-runtime/include"), dpcppInclude: path.join(sdk.root, "include"),
    dpcppSyclInclude: path.join(sdk.root, "include/sycl"), windowsSdkIncludeRoots: sdkRoots});
  for (const file of [...plan.dpcppHeaders, ...plan.urHeaders]) {
    fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, "#pragma once\n");
  }
  fs.mkdirSync(path.join(sourceRoot, "sycl/include"), {recursive: true});
  fs.mkdirSync(path.join(sdk.root, "include/sycl"), {recursive: true});
  for (const name of ["llvm-nm", "llvm-readobj", "llvm-objcopy"]) {fs.writeFileSync(path.join(sdk.root, "bin", name), "fixture\n");}
  return {plan, input: {compilerRoot: sdk.root, sourceRoot, dpcppRoot: sdk.root, dpcppManifest: {dpcpp: sdk.record},
    host: /** @type {any} */ (host), tools: {llvmNm: path.join(sdk.root, "bin/llvm-nm"), toolRoot: path.join(sdk.root, "bin")},
    cudaImages: null}};
}

let windowsGpuConfigFixtureCounter = 0;
/** @param {import("node:test").TestContext} t */
function dpcppWindowsGpuConfigFixture(t) {
  const suffix = `ci-${Date.now()}-${++windowsGpuConfigFixtureCounter}`, runnerTemp = fs.mkdtempSync(path.join("/var/tmp", "mom-windows-gpu-config-")),
    taskRoot = fs.mkdtempSync(path.join("/var/tmp", "mom-windows-gpu-config-task-")),
    dpcppRoot = path.join("/tmp", "mom-native-dpcpp-cuda-inputs-" + suffix),
    linuxSource = path.join("/tmp", "mom-native-linux-source-prerequisites-" + suffix),
    prepared = path.join(linuxSource, "prepared"), sourceRoot = path.join(prepared, "windows/dpcpp"),
    sourcesFile = path.join(prepared, "windows/sources.json"), manifest = path.join(dpcppRoot, "inputs.json"),
    state = {name: path.basename(dpcppRoot), root: dpcppRoot, dpcppRoot: path.join(dpcppRoot, "dpcpp"), manifest,
      linux: {source: linuxSource, sourceName: path.basename(linuxSource), prepared}},
    manifestRecord = {source: {tarSha256: "1".repeat(64)}, dpcpp: {archiveSha256: "2".repeat(64)}};
  for (const directory of [path.join(state.dpcppRoot, "include/sycl"), path.join(sourceRoot, "sycl/include"),
    path.join(sourceRoot, "unified-runtime/include")]) {fs.mkdirSync(directory, {recursive: true});}
  fs.writeFileSync(manifest, JSON.stringify(manifestRecord) + "\n", {flag: "wx"});
  fs.writeFileSync(sourcesFile, JSON.stringify({target: "windows", prepared: {dpcpp: {root: sourceRoot}}}) + "\n", {flag: "wx"});
  const windowsSdk = windowsSdkFixture(suffix, runnerTemp), toolsRoot = path.join(taskRoot, "tools"),
    manifestSha256 = sha256(manifest), toolsExpectation = {handoffSha256: "3".repeat(64)},
    input = {windowsSdkRoot: windowsSdk.root, windowsSdkPins: windowsSdk.pins, windowsSourcesFile: sourcesFile,
      dpcppState: state, dpcppManifestSha256: manifestSha256, toolsRoot, toolsExpectation, compilerRoot: state.dpcppRoot, suffix};
  t.after(() => {
    for (const root of [runnerTemp, taskRoot, dpcppRoot, linuxSource]) {fs.rmSync(root, {recursive: true, force: true});}
  });
  return {suffix, runnerTemp, state, manifestRecord, manifestSha256, sourcesFile, sourceRoot, input};
}

/** @param {string} root @param {{clangVersion?:string,clangResource?:string,clangxxResource?:string,status?:number}} [options]
 * @returns {(command:string,args:string[])=>{status:number|null,signal:NodeJS.Signals|null,stdout:string,error?:Error}} */
function fakeDpcppQuery(root, options = {}) {
  const version = `clang version 23.0.0git (https://github.com/intel/llvm ${pins.source})`, resource = path.join(root, "lib/clang/23");
  return (command, args) => {
    if (options.status !== undefined && options.status !== 0) {return {status: options.status, signal: null, stdout: "", error: new Error("fixture query failed")};}
    if (args[0] === "--version") {
      const text = command.endsWith("clang++") ? options.clangVersion || version : version;
      return {status: 0, signal: null, stdout: text + "\n"};
    }
    if (args[0] === "--print-resource-dir") {
      return {status: 0, signal: null, stdout: (command.endsWith("clang++") ? options.clangxxResource : options.clangResource) || resource};
    }
    return {status: 1, signal: null, stdout: ""};
  };
}

test.describe("Windows oneAPI GPU config path derivation", {skip: process.platform !== "linux" && "Linux Windows cross-build config"}, () => {
  test("preserves physical and Windows SDK virtual include order", () => {
    const plan = windowsNativeGpuIncludePlan(input());
    assert.deepEqual(plan.includeRoots, ["/opt/intel/compiler/lib/clang/23/include", "/src/prepared/dpcpp/sycl/include",
      "/sdk/dpcpp/include", "/sdk/dpcpp/include/sycl", "/src/prepared/dpcpp/unified-runtime/include", ...sdkRoots]);
    assert.deepEqual(plan, windowsNativeOneapiIncludePlan(input()));
  });

  test("selects nine SDK headers and two matching Unified Runtime source files under their actual layouts", () => {
    const plan = windowsNativeOneapiIncludePlan(input());
    assert.deepEqual(plan.dpcppHeaders, ["CL/cl.h", "CL/cl_ext.h", "CL/cl_platform.h", "CL/cl_version.h",
      "sycl/device_aspect_macros.hpp", "sycl/feature_test.hpp", "sycl/version.hpp",
      "unified-runtime/ur_api.h", "unified-runtime/ur_api_funcs.def"].map(relative => path.join("/sdk/dpcpp/include", relative)));
    assert.deepEqual(plan.urHeaders, ["ur_api.h", "ur_api_funcs.def"].map(relative =>
      path.join("/src/prepared/dpcpp/unified-runtime/include/unified-runtime", relative)));
  });

  test("rejects substituted, duplicated, or noncanonical SDK virtual roots", () => {
    const changed = input(); changed.windowsSdkIncludeRoots[0] = "/physical/sdk/msvc/include";
    assert.throws(() => windowsNativeOneapiIncludePlan(changed), /pinned physical and SDK VFS layout/);
    const duplicate = input(); duplicate.windowsSdkIncludeRoots[1] = "/mom-sdk/msvc/include";
    assert.throws(() => windowsNativeOneapiIncludePlan(duplicate), /pinned physical and SDK VFS layout/);
    const relative = input(); relative.windowsSdkIncludeRoots[2] = "mom-sdk/sdk/include/shared";
    assert.throws(() => windowsNativeOneapiIncludePlan(relative), /pinned physical and SDK VFS layout/);
  });

  test("rejects a DPC++ SYCL root detached from its admitted include tree", () => {
    const changed = input(); changed.dpcppSyclInclude = "/other/sycl";
    assert.throws(() => windowsNativeGpuIncludePlan(changed), /beneath the admitted SDK include root/);
  });

  test("admits the same-run DPC++ frontend and canonical resource headers without executing the compiler", () => {
    const fixture = dpcppFixture();
    try {
      const result = resolveDpcppCompiler(fixture.root, fixture.record, {}, /** @type {any} */ (fakeDpcppQuery(fixture.root)));
      assert.deepEqual(result.tools, {clang: path.join(fixture.root, "bin/clang++"), clangC: path.join(fixture.root, "bin/clang")});
      assert.equal(result.resourceInclude, path.join(fixture.root, "lib/clang/23/include"));
      assert.equal(result.inputs[fixture.record.clang.path], fixture.record.clang.sha256);
      assert.equal(result.inputs[fixture.record.clangxx.path], fixture.record.clangxx.sha256);
    } finally {fs.rmSync(fixture.root, {recursive: true, force: true});}
  });

  test("rejects changed recorded frontend bytes and a compiler version outside the pinned SDK", () => {
    const fixture = dpcppFixture();
    try {
      fs.appendFileSync(fixture.record.clang.path, "changed\n");
      assert.throws(() => resolveDpcppCompiler(fixture.root, fixture.record, {}, /** @type {any} */ (fakeDpcppQuery(fixture.root))), /differs from its receipt/);
    } finally {fs.rmSync(fixture.root, {recursive: true, force: true});}
    const second = dpcppFixture();
    try {
      const query = fakeDpcppQuery(second.root, {clangVersion: "clang version 99.0"});
      assert.throws(() => resolveDpcppCompiler(second.root, second.record, {}, /** @type {any} */ (query)), /Use the matching pinned DPC\+\+ Engineering SDK/);
    } finally {fs.rmSync(second.root, {recursive: true, force: true});}
  });

  test("rejects resource headers outside the SDK or differing between C and C++ drivers", () => {
    const fixture = dpcppFixture(), outside = fs.mkdtempSync(path.join(os.tmpdir(), "mom-dpcpp-outside-"));
    try {
      assert.throws(() => resolveDpcppCompiler(fixture.root, fixture.record, {}, /** @type {any} */
        (fakeDpcppQuery(fixture.root, {clangxxResource: outside}))), /outside the admitted SDK/);
      assert.throws(() => resolveDpcppCompiler(fixture.root, fixture.record, {}, /** @type {any} */
        (fakeDpcppQuery(fixture.root, {clangResource: path.join(fixture.root, "lib/clang/23/include")}))), /resource directories differ/);
    } finally {
      fs.rmSync(fixture.root, {recursive: true, force: true}); fs.rmSync(outside, {recursive: true, force: true});
    }
  });

  test("rejects frontend mutation during the bounded version/resource queries", () => {
    const fixture = dpcppFixture(), ordinary = fakeDpcppQuery(fixture.root);
    try {
      const query = (/** @type {string} */ command, /** @type {string[]} */ args) => {
        if (command.endsWith("clang++") && args[0] === "--print-resource-dir") {fs.appendFileSync(fixture.record.clang.path, "race\n");}
        return ordinary(command, args);
      };
      assert.throws(() => resolveDpcppCompiler(fixture.root, fixture.record, {}, /** @type {any} */ (query)), /changed during compiler queries/);
    } finally {fs.rmSync(fixture.root, {recursive: true, force: true});}
  });

  test("admits the exact DPC++ CUDA SDK and image config shape without changing other profiles", () => {
    const config = {version: 1, profile: "dpcpp", tools: {clang: "/sdk/clang++", clangC: "/sdk/clang",
      llvmNm: "/tools/llvm-nm", llvmReadobj: "/tools/llvm-readobj", llvmObjcopy: "/tools/llvm-objcopy"},
    includeRoots: ["/sdk/include"], overlay: "/sdk/overlay.yaml", cuda: {sdk: "/cuda", images: "/images/cuda-images.inc"}};
    assert.equal(validateWindowsGpuConfig(config), config);
    assert.throws(() => validateWindowsGpuConfig({...config, cuda: {...config.cuda, bundle: "/untrusted.tar"}}), /needs explicit CUDA/);
    assert.throws(() => validateWindowsGpuConfig({...config, profile: "dpcpp-opencl"}), /CUDA inputs are not valid/);
    assert.throws(() => validateWindowsGpuConfig({...config, profile: "oneapi"}), /CUDA inputs are not valid/);
  });

  test("binds the CUDA image result to the external artifact pin and current-source validator", t => {
    const fixture = cudaImagesFixture(t), validator = cudaImages.cudaImageInputs;
    let called = false;
    t.mock.method(cudaImages, "cudaImageInputs", (/** @type {string} */ file, /** @type {string} */ sourceRoot) => {
      called = true;
      assert.equal(file, fixture.include);
      assert.equal(sourceRoot, path.resolve(__dirname, ".."));
      return {files: {[file]: fixture.includeSha, [file + ".json"]: fixture.sidecarSha}, manifest: fixture.manifest};
    });
    const admitted = windowsNativeGpuCudaImages(fixture.input(), fixture.suffix);
    assert.equal(called, true);
    assert.equal(admitted.include, fixture.include);
    assert.equal(admitted.hashes.bundleSha256, sha256(fixture.archive));
    assert.equal(admitted.hashes.sidecarSha256, fixture.sidecarSha);
    // The existing native-cuda-images suite verifies that this maintained validator rejects stale sources.
    assert.equal(typeof validator, "function");
  });

  test("rejects changed CUDA image artifact pins, mismatched producer source hashes, and extra keys", t => {
    const fixture = cudaImagesFixture(t);
    t.mock.method(cudaImages, "cudaImageInputs", () => ({files: {[fixture.include]: fixture.includeSha,
      [fixture.include + ".json"]: fixture.sidecarSha}, manifest: fixture.manifest}));
    assert.throws(() => windowsNativeGpuCudaImages({...fixture.input(), bundleSha256: "0".repeat(64)}, fixture.suffix), /external same-run pin/);
    assert.throws(() => windowsNativeGpuCudaImages(Object.assign(fixture.input(), {unexpected: true}), fixture.suffix), /exact same-run artifact contract/);
    fixture.writeResult("0".repeat(64));
    assert.throws(() => windowsNativeGpuCudaImages(fixture.input(), fixture.suffix), /result, current-source sidecar/);
  });

  test("propagates stale-source rejection from the maintained CUDA image validator", t => {
    const fixture = cudaImagesFixture(t);
    t.mock.method(cudaImages, "cudaImageInputs", () => {throw new Error("stale maintained CUDA source fixture");});
    assert.throws(() => windowsNativeGpuCudaImages(fixture.input(), fixture.suffix), /stale maintained CUDA source fixture/);
  });

  test("rejects a regenerated valid sidecar when restored bytes differ from the original tar", t => {
    const fixture = cudaImagesFixture(t), writeBundle = (/** @type {ReturnType<typeof validCudaImageBundle>} */ bundle) => {
      fs.writeFileSync(fixture.include, bundle.source);
      fs.writeFileSync(fixture.sidecar, JSON.stringify(bundle.manifest, null, 2) + "\n");
      fs.writeFileSync(fixture.resultFile, JSON.stringify({format: "mom-cuda-image-build-v1",
        include: path.join("/tmp", "mom-native-cuda-images-" + fixture.suffix, "bundle", "cuda-images.inc"),
        sidecar: path.join("/tmp", "mom-native-cuda-images-" + fixture.suffix, "bundle", "cuda-images.inc.json"),
        compilerVersions: bundle.manifest["compilers"].map((/** @type {{major:number,minor:number}} */ row) => `${row.major}.${row.minor}`).sort(),
        plannedImages: bundle.manifest["images"].length, embeddedBytes: bundle.manifest["resourceBytes"],
        sourceSha256: bundle.manifest["sourceSha256"]}) + "\n");
    };
    writeBundle(validCudaImageBundle(t));
    fs.unlinkSync(fixture.archive); ciTar.packState(fixture.archive, "/tmp", [path.basename(fixture.root)]);
    writeBundle(validCudaImageBundle(t, true));
    assert.doesNotThrow(() => cudaImages.cudaImageInputs(fixture.include, path.resolve(__dirname, "..")));
    assert.throws(() => windowsNativeGpuCudaImages(fixture.input(), fixture.suffix),
      /Restored CUDA image files differ from the original externally pinned tar members/);
  });

  test.describe("Windows ACPP full-tar input admission", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
    test("rejects a stale SDK recipe even when the original plan and result agree", t => {
      const fixture = acppFixture(t);
      fixture.result.recipe.graph["scripts/native-sycl-inputs.js"] = "0".repeat(64);
      fixture.plan.recipe = fixture.result.recipe;
      fixture.rewriteProducer(); fixture.repack();
      assert.throws(() => windowsNativeAcppInputs(fixture.input(), fixture.suffix), /Transported AdaptiveCpp SDK recipe source inputs are stale/);
    });
    test("joins the pinned archive, original result/plan, plugin and prepared source header", t => {
      const fixture = acppFixture(t); let calls = 0;
      t.mock.method(preparedSdkSources, "verifyPreparedSdkSources", (/** @type {string} */ file, /** @type {string} */ target) => {
        calls++; assert.equal(file, fixture.sourceReceiptFile); assert.equal(target, "acpp");
        return JSON.parse(fs.readFileSync(file, "utf8"));
      });
      const admitted = windowsNativeAcppInputs(fixture.input(), fixture.suffix);
      assert.equal(calls, 2);
      assert.equal(admitted.source, fixture.sourceRoot);
      assert.equal(admitted.generatedInclude, fixture.generatedInclude);
      assert.equal(admitted.plugin, fixture.plugin);
      assert.equal(admitted.hashes.bundleSha256, sha256(fixture.archive));
      assert.equal(admitted.hashes.pluginSha256, sha256(fixture.plugin));
      assert.equal(admitted.hashes.generatedConfigSha256, sha256(fixture.configFile));
      assert.equal(admitted.hashes.sourceHeaderSha256, sha256(fixture.sourceHeader));
      assert.deepEqual(admitted.generatedFiles.map(file => file.path), ["hipSYCL/common/config.hpp", "hipSYCL/common/generated.hpp"]);
    });

    test("rejects malformed pins, another run's root, and extra caller paths", t => {
      const fixture = acppFixture(t);
      assert.throws(() => windowsNativeAcppInputs({...fixture.input(), bundleSha256: "0".repeat(64)}, fixture.suffix), /external same-run pin/);
      assert.throws(() => windowsNativeAcppInputs(fixture.input(), fixture.suffix + "1"), /canonical same-run roots/);
      assert.throws(() => windowsNativeAcppInputs(/** @type {any} */ ({...fixture.input(), unexpected: "/tmp/other"}), fixture.suffix), /same-run Linux handoff/);
    });

    test("rejects wrong producer backend, revision and plugin-selection flags", t => {
      /** @type {Array<(fixture: ReturnType<typeof acppFixture>) => void>} */ const changes = [
        (fixture) => {fixture.result.backend = "hip";},
        (fixture) => {fixture.result.revision = "0".repeat(40);},
        (fixture) => {fixture.result.inputs.flags[1] = "-fplugin=/tmp/other.so";},
        (fixture) => {fixture.plan.admitted.revision = "0".repeat(40);},
      ];
      for (const change of changes) {
        const fixture = acppFixture(t); change(fixture); fixture.rewriteProducer(); fixture.repack();
        t.mock.method(preparedSdkSources, "verifyPreparedSdkSources", (/** @type {string} */ file) => JSON.parse(fs.readFileSync(file, "utf8")));
        assert.throws(() => windowsNativeAcppInputs(fixture.input(), fixture.suffix), /Original ACPP receipt\/plan|Original ACPP flags/);
      }
    });

    test("requires the generated config header at the consumed build/include path", t => {
      const missing = acppFixture(t); fs.rmSync(missing.configFile); missing.repack();
      assert.throws(() => windowsNativeAcppInputs(missing.input(), missing.suffix), /omits the generated hipSYCL config/);
      const misplaced = acppFixture(t), duplicate = path.join(misplaced.build, "alternate/include/hipSYCL/common/config.hpp");
      fs.rmSync(misplaced.configFile); fs.mkdirSync(path.dirname(duplicate), {recursive: true});
      fs.writeFileSync(duplicate, "#pragma once\n"); misplaced.repack();
      assert.throws(() => windowsNativeAcppInputs(misplaced.input(), misplaced.suffix), /omits the generated hipSYCL config/);
    });

    test("compares the original tar bytes with restored receipt, plugin and config files", t => {
      for (const selected of ["result", "plan", "plugin", "config"]) {
        const fixture = acppFixture(t), file = selected === "result" ? path.join(fixture.root, "result.json") :
          selected === "plan" ? path.join(fixture.root, "plan.json") : selected === "plugin" ? fixture.plugin : fixture.configFile;
        fs.appendFileSync(file, "\n");
        t.mock.method(preparedSdkSources, "verifyPreparedSdkSources", (/** @type {string} */ sourceFile) => JSON.parse(fs.readFileSync(sourceFile, "utf8")));
        assert.throws(() => windowsNativeAcppInputs(fixture.input(), fixture.suffix), /differs from the externally pinned tar member/);
      }
    });

    test("rejects symlinks in the generated header tree", t => {
      const fixture = acppFixture(t), extra = path.join(fixture.generatedInclude, "hipSYCL/common/generated.hpp");
      fs.rmSync(extra); fs.symlinkSync(path.basename(fixture.configFile), extra);
      assert.throws(() => windowsNativeAcppInputs(fixture.input(), fixture.suffix), /contains a symlink/);
    });

    test("rejects changed bytes in a non-config generated header", t => {
      const fixture = acppFixture(t), extra = path.join(fixture.generatedInclude, "hipSYCL/common/generated.hpp");
      fs.appendFileSync(extra, "changed after archive creation\n");
      t.mock.method(preparedSdkSources, "verifyPreparedSdkSources", (/** @type {string} */ file) => JSON.parse(fs.readFileSync(file, "utf8")));
      assert.throws(() => windowsNativeAcppInputs(fixture.input(), fixture.suffix), /generated header differs from the externally pinned tar member/);
    });

    test("bounds generated include depth and directory count", t => {
      const deep = acppFixture(t), deepPath = path.join(deep.generatedInclude, ...Array.from({length: 9}, (_, index) => "d" + index));
      fs.mkdirSync(deepPath, {recursive: true}); fs.writeFileSync(path.join(deepPath, "leaf.hpp"), "#pragma once\n"); deep.repack();
      assert.throws(() => windowsNativeAcppInputs(deep.input(), deep.suffix), /depth bound/);
      const wide = acppFixture(t);
      for (let index = 0; index < 129; index++) {fs.mkdirSync(path.join(wide.generatedInclude, "empty" + index));}
      wide.repack();
      assert.throws(() => windowsNativeAcppInputs(wide.input(), wide.suffix), /directory bound/);
    });

    test("shares one five-minute budget across tar listing and member reads", t => {
      const fixture = acppFixture(t), originalList = ciTar.listState, originalSizeScan = ciTar.tarExpandedBytes;
      let clock = 1000000000000;
      t.mock.method(Date, "now", () => clock);
      t.mock.method(ciTar, "listState", (/** @type {string} */ archive, /** @type {string[]} */ roots, /** @type {number} */ timeout) => {
        assert.equal(timeout, 5 * 60 * 1000);
        const result = originalList(archive, roots, timeout); clock += 3 * 60 * 1000; return result;
      });
      t.mock.method(ciTar, "tarExpandedBytes", (/** @type {string} */ archive, /** @type {number} */ count, /** @type {number} */ timeout) => {
        assert.equal(timeout, 2 * 60 * 1000);
        const result = originalSizeScan(archive, count, timeout); clock += 119500; return result;
      });
      assert.throws(() => windowsNativeAcppInputs(fixture.input(), fixture.suffix), /shared five-minute budget/);
    });

    test("rechecks ACPP source identity and required header membership before returning", t => {
      const fixture = acppFixture(t);
      t.mock.method(preparedSdkSources, "verifyPreparedSdkSources", (/** @type {string} */ file) => {
        const result = JSON.parse(fs.readFileSync(file, "utf8"));
        result.prepared.acpp.root = "/tmp/other-acpp-source";
        return result;
      });
      assert.throws(() => windowsNativeAcppInputs(fixture.input(), fixture.suffix), /does not bind the required HCF registration header/);
    });

    test("fails if the required source header changes between the before/after verification", t => {
      const fixture = acppFixture(t); let calls = 0;
      t.mock.method(preparedSdkSources, "verifyPreparedSdkSources", (/** @type {string} */ file) => {
        if (++calls === 2) {fs.appendFileSync(fixture.sourceHeader, "changed\n");}
        return JSON.parse(fs.readFileSync(file, "utf8"));
      });
      assert.throws(() => windowsNativeAcppInputs(fixture.input(), fixture.suffix), /does not bind the required HCF registration header/);
      assert.equal(calls, 2);
    });
  });

  test.describe("Windows SYCL GPU config manifest admission", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
    test("local descriptor construction shares compiler/header checks without a CI transport identity", t => {
      const fixture = localDpcppConfigFixture(t), result = gpuConfig.composeWindowsDpcppGpuConfig("dpcpp-opencl", fixture.input);
      assert.equal(result.config.profile, "dpcpp-opencl");
      assert.deepEqual(result.config.includeRoots, fixture.plan.includeRoots);
      assert.equal(result.compiler.root, fixture.input.compilerRoot);
      assert.equal(Object.keys(result.selectedHeaders).length, 11);
      assert.equal(Object.hasOwn(result.config, "runAttempt"), false);
      assert.equal(result.configSha256, createHash("sha256").update(result.configBytes).digest("hex"));
    });
    test("local construction still rejects mismatched Unified Runtime headers and cross-profile CUDA", t => {
      const fixture = localDpcppConfigFixture(t), header = fixture.plan.urHeaders[0];
      if (!header) {throw new Error("Missing UR fixture header");}
      fs.appendFileSync(header, "// changed\n");
      assert.throws(() => gpuConfig.composeWindowsDpcppGpuConfig("dpcpp-opencl", fixture.input), /Unified Runtime headers differ/);
      assert.throws(() => gpuConfig.composeWindowsDpcppGpuConfig("dpcpp", fixture.input), /requires admitted CUDA images/);
    });
    test("forwards the externally verified DPC++ manifest pin to the maintained audit", t => {
      const fixture = dpcppWindowsGpuConfigFixture(t), auditReached = new Error("DPC++ audit reached"),
        sources = JSON.parse(fs.readFileSync(fixture.sourcesFile, "utf8"));
      t.mock.method(preparedSdkSources, "verifyPreparedSdkSources", (/** @type {string} */ file, /** @type {string} */ target) => {
        assert.equal(file, fixture.sourcesFile); assert.equal(target, "windows"); return sources;
      });
      let audited = false;
      t.mock.method(dpcppInputs, "auditDpcppCudaInputs", (/** @type {any} */ state, /** @type {any} */ manifest,
        /** @type {string | undefined} */ pin) => {
        assert.equal(state, fixture.state); assert.deepEqual(manifest, fixture.manifestRecord);
        assert.equal(pin, fixture.manifestSha256, "pass the validated same-run hash, not a newly inferred provenance value");
        audited = true; throw auditReached;
      });

      // This fixture intentionally stops at the audit boundary, so fields consumed only by
      // later compiler/handoff stages are absent.
      assert.throws(() => windowsNativeGpuConfig("dpcpp-opencl", /** @type {any} */ (fixture.input)), error => error === auditReached);
      assert.equal(audited, true);
    });
  });

  test.describe("Windows ACPP GPU config composition", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
    test("composes the complete HIP profile from same-run ACPP, LLVM 21 and Windows SDK inputs", t => {
      const fixture = acppGpuFixture(t), queries = mockAcppGpuReaders(t, fixture);
      t.mock.method(dpcppInputs, "auditDpcppCudaInputs", () => {throw new Error("HIP must not admit or query DPC++/CUDA inputs");});
      const result = windowsNativeAcppGpuConfig("acpp-hip", fixture.hipInput), config = result.config;
      const localInput = {host: /** @type {any} */ (fixture.windowsSdk.admitted),
          tools: {llvmNm: path.join(fixture.toolsRoot, "bin/llvm-nm"), toolRoot: path.join(fixture.toolsRoot, "bin")},
          acpp: {source: fixture.acpp.sourceRoot, generatedInclude: fixture.acpp.generatedInclude, plugin: fixture.acpp.plugin},
          frontend: {clang: fixture.clang, clangxx: fixture.clangxx, resourceInclude: fixture.resourceInclude},
          hipInclude: fixture.hipInclude, cuda: null}, local = gpuConfig.composeWindowsAcppGpuConfig("acpp-hip", localInput);
      assert.deepEqual(local.configBytes, result.configBytes, "local and CI construction produce identical descriptor bytes");
      assert.equal(local.configSha256, result.configSha256);
      assert.throws(() => gpuConfig.composeWindowsAcppGpuConfig("acpp-cuda", localInput), /only its selected CUDA or HIP/);
      assert.equal(config.profile, "acpp-hip");
      assert.deepEqual(Object.keys(config).sort(), ["acpp", "includeRoots", "overlay", "profile", "tools", "version"]);
      assert.deepEqual(config.acpp, {source: fixture.acpp.sourceRoot, generatedInclude: fixture.acpp.generatedInclude, plugin: fixture.acpp.plugin});
      assert.equal(config.tools.clang, fixture.clangxx); assert.equal(config.tools.clangC, fixture.clang);
      assert.equal(config.tools.llvmNm, path.join(fixture.toolsRoot, "bin/llvm-nm"));
      assert.equal(config.tools.llvmReadobj, path.join(fixture.toolsRoot, "bin/llvm-readobj"));
      assert.equal(config.tools.llvmObjcopy, path.join(fixture.toolsRoot, "bin/llvm-objcopy"));
      assert.deepEqual(config.includeRoots, [fixture.resourceInclude, fixture.hipInclude, path.join(fixture.acpp.sourceRoot, "include"),
        fixture.acpp.generatedInclude, ...sdkRoots]);
      assert.equal("cuda" in config, false); assert.equal("compilerRoot" in config, false);
      assert.equal(result.configSha256, createHash("sha256").update(result.configBytes).digest("hex"));
      assert.equal(result.configBytes.toString("utf8"), JSON.stringify(config, null, 2) + "\n");
      assert.equal(result.admittedInputHashes.acpp.bundleSha256, sha256(fixture.acpp.archive));
      assert.equal(result.admittedInputHashes.linuxLlvmAcppFrontend.version, "21.1.8");
      assert.deepEqual(result.admittedInputHashes.hipHeaders?.files, {[fixture.hipHeader]: sha256(fixture.hipHeader)});
      assert.equal(queries.length, 4);
      assert.ok(queries.every(query => query.command === fixture.clang || query.command === fixture.clangxx));
    });

    test("queries LLVM without inherited LD_ or DYLD_ loader controls", t => {
      const fixture = acppGpuFixture(t), keys = ["LD_TRACE_LOADED_OBJECTS", "LD_PROFILE", "DYLD_PRINT_LIBRARIES"],
        old = Object.fromEntries(keys.map(key => [key, process.env[key]]));
      t.after(() => {for (const key of keys) {if (old[key] === undefined) {delete process.env[key];} else {process.env[key] = old[key];}}});
      for (const key of keys) {process.env[key] = "1";}
      const frontend = {clang: fixture.clang, clangxx: fixture.clangxx, resourceInclude: fixture.resourceInclude,
        sourceArchive: path.join(fixture.frontendRoot, "llvm-project-21.1.8.src.tar.xz"), files: fixture.frontendFiles};
      let queries = 0;
      gpuConfig.resolveAcppFrontend(fixture.frontendRoot, frontend, (/** @type {string} */ command,
        /** @type {string[]} */ args, /** @type {any} */ options) => {
        queries++;
        assert.ok(command === fixture.clang || command === fixture.clangxx);
        for (const key of keys) {assert.equal(options.env[key], undefined, key + " must be removed from the LLVM query environment");}
        const stdout = args[0] === "--version" ? "clang version 21.1.8\n" : fixture.resourceDir;
        return {pid: 1, signal: null, status: 0, stdout, stderr: "", output: [null, stdout, ""]};
      });
      assert.equal(queries, 4);
    });

    test("HIP refuses missing or rejected header provenance instead of compiling with ambient headers", t => {
      const fixture = acppGpuFixture(t); mockAcppGpuReaders(t, fixture);
      assert.throws(() => windowsNativeAcppGpuConfig("acpp-hip", fixture.input), /same-run SDK, tool and frontend/);
      t.mock.method(require("../scripts/native-rocm-hip-headers"), "rocmHipHeaderInputs", () => {throw new Error("HIP header receipt changed");});
      assert.throws(() => windowsNativeAcppGpuConfig("acpp-hip", fixture.hipInput), /HIP header receipt changed/);
    });

    test("composes CUDA only after the existing DPC++/CUDA auditor and CUDA-image validator", t => {
      const fixture = acppGpuFixture(t), queries = mockAcppGpuReaders(t, fixture),
        images = cudaImagesFixture(t, {suffix: fixture.suffix, runnerTemp: fixture.acpp.runnerTemp});
      let auditCalls = 0;
      t.mock.method(require("../scripts/native-rocm-hip-headers"), "rocmHipHeaderInputs", () => {throw new Error("CUDA must not admit HIP headers");});
      t.mock.method(dpcppInputs, "auditDpcppCudaInputs", (/** @type {any} */ state, /** @type {any} */ manifest, /** @type {string | undefined} */ pin) => {
        auditCalls++; assert.equal(state, fixture.dpcppState); assert.deepEqual(manifest, fixture.dpcppManifest);
        assert.equal(pin, sha256(fixture.dpcppState.manifest), "forward the verified external pin without restamping acquisition provenance");
        return manifest;
      });
      t.mock.method(cudaImages, "cudaImageInputs", (/** @type {string} */ file) => {
        assert.equal(file, images.include);
        return {files: {[file]: images.includeSha, [file + ".json"]: images.sidecarSha},
          manifest: {sourceSha256: images.includeSha, images: [{}], resourceBytes: Buffer.byteLength("image include fixture\n")}};
      });
      const input = {...fixture.input, dpcppState: fixture.dpcppState, dpcppManifestSha256: sha256(fixture.dpcppState.manifest),
          cudaImages: images.input()}, result = windowsNativeAcppGpuConfig("acpp-cuda", input);
      assert.equal(result.config.profile, "acpp-cuda"); assert.equal(auditCalls, 1);
      assert.deepEqual(result.config.cuda, {sdk: fixture.dpcppState.cudaRoot, images: images.include});
      assert.deepEqual(result.admittedInputHashes.cudaImages, {bundleSha256: sha256(images.archive), includeSha256: images.includeSha,
        sidecarSha256: images.sidecarSha, resultSha256: sha256(images.resultFile)});
      assert.equal(result.admittedInputHashes.dpcppCuda?.manifestSha256, sha256(fixture.dpcppState.manifest));
      assert.equal(queries.length, 4); assert.ok(queries.every(query => query.command === fixture.clang || query.command === fixture.clangxx));
    });

    test("rejects another run, altered pins and cross-profile CUDA fields before config output", t => {
      const fixture = acppGpuFixture(t); mockAcppGpuReaders(t, fixture);
      t.mock.method(dpcppInputs, "auditDpcppCudaInputs", () => fixture.dpcppManifest);
      const cudaInput = {...fixture.input, dpcppState: fixture.dpcppState, dpcppManifestSha256: sha256(fixture.dpcppState.manifest),
        cudaImages: {archive: "/tmp/not-the-run.tar", bundleSha256: "0".repeat(64), root: "/tmp/not-the-run"}};
      assert.throws(() => windowsNativeAcppGpuConfig("acpp-hip", {...fixture.input, dpcppState: fixture.dpcppState}), /same-run SDK, tool and frontend/);
      assert.throws(() => windowsNativeAcppGpuConfig("acpp-cuda", {...cudaInput, suffix: "../77"}), /same-run SDK, tool and frontend/);
      assert.throws(() => windowsNativeAcppGpuConfig("acpp-cuda", {...cudaInput,
        frontendExpectation: {...fixture.input.frontendExpectation, handoffSha256: "bad"}}), /same-run SDK, tool and frontend/);
      assert.throws(() => windowsNativeAcppGpuConfig("acpp-cuda", {...cudaInput, dpcppManifestSha256: "bad"}), /ACPP CUDA inputs/);
      assert.throws(() => windowsNativeAcppGpuConfig(/** @type {any} */ ("dpcpp"), fixture.input), /Unsupported Windows ACPP/);
    });

    test("rejects an unpinned LLVM version or wrong Clang resource directory", t => {
      const wrongVersion = acppGpuFixture(t); mockAcppGpuReaders(t, wrongVersion, {version: "clang version 22.0.0\n"});
      assert.throws(() => windowsNativeAcppGpuConfig("acpp-hip", wrongVersion.hipInput), /same pinned LLVM 21.1.8/);
      const wrongResource = acppGpuFixture(t), queries = mockAcppGpuReaders(t, wrongResource,
        {resourceDir: path.join(wrongResource.frontendRoot, "lib/clang/22")});
      assert.throws(() => windowsNativeAcppGpuConfig("acpp-hip", wrongResource.hipInput), /resource directory differs/);
      assert.ok(queries.length >= 3);
    });
  });
});
