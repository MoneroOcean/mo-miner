"use strict";

const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const {after, describe, test} = require("node:test");
const {spawnSync} = require("node:child_process");
const {fileHash, sourceInputs} = require("../scripts/build-source-inputs");
const {hermesPins} = require("../scripts/build-native-hermes");
const rocmProducer = require("../scripts/build-native-rocm-host-runtime");
const {nativeStaticRuntimeInputs, nativeHermesInputs, nativeRocmHostInputs, nativeNvrtcInputs, gitOutput} = require("../scripts/native-release-sdk-inputs");

const repo = path.resolve(__dirname, ".."), scratch = fs.mkdtempSync(path.join(os.tmpdir(), "native-release-sdk-inputs-"));
after(() => fs.rmSync(scratch, {recursive: true, force: true}));

/** @typedef {Record<string, unknown> & {inputs?: any, libraries?: any, revision?: any, fingerprints?: any,
 * sdkInputs?: any, dependencies?: any, archives?: any}} MutableReceipt */

const {linuxSysrootRecord, linuxSysrootReceiptSha256} = require("../scripts/native-linux-sysroot");
const sysroot = linuxSysrootRecord({receiptSha256: linuxSysrootReceiptSha256});
const syclRecipe = ["scripts/build-native-sycl-sdk.js", "scripts/native-sycl-inputs.js", "scripts/native-hcf.js", "scripts/native-jit-link.js",
  "scripts/native-linux-sysroot.js", "tests/reference/native-sycl-hcf-repro.cpp",
  "tests/reference/native-jit-cache.cpp",
  "scripts/build-environment.js", "scripts/native-audit.js", "scripts/native-resources.js",
  "runtime/adaptivecpp-static.patch", "runtime/adaptivecpp-jit.patch"];
const hermesRecipe = ["scripts/build-native-hermes.js", "scripts/build-source-inputs.js", "scripts/build-environment.js",
  "scripts/native-audit.js", "scripts/native-linux-sysroot.js", "runtime/hermes-host-rejections.patch"];
const rocmRecipe = ["scripts/build-native-rocm-host-runtime.js", "scripts/build-native-sycl-sdk.js", "scripts/build-native-comgr-sdk.js",
  "scripts/native-comgr-inputs.js", "scripts/prepare-native-sdk-sources.js", "scripts/build-native-release.js",
  "scripts/build-source-inputs.js", "scripts/build-environment.js", "scripts/native-audit.js", "scripts/native-sycl-inputs.js", "scripts/native-hcf.js",
  "scripts/native-linux-sysroot.js", "scripts/build-native-libdrm.js"];
const syclRevision = "da2463e45aa90aa36306c45abcfc05b87de51bc6";

/** @param {string} file */
function directory(file) {fs.mkdirSync(file, {recursive: true}); return file;}
/** @param {string} file @param {string | Buffer} [data] */
function write(file, data = "fixture\n") {directory(path.dirname(file)); fs.writeFileSync(file, data); return file;}
/** @param {string} file */
function archive(file) {return write(file, Buffer.concat([Buffer.from("!<arch>\n"), Buffer.from("fixture\n")]));}
/** @param {string} file @param {unknown} value */
function json(file, value) {return write(file, JSON.stringify(value) + "\n");}
/** @param {string[]} files @returns {Record<string, string>} */
function hashes(files) {return Object.fromEntries(files.map(file => [file, fileHash(file)]));}
/** @param {string[]} files */
function recipe(files) {return sourceInputs(repo, [], files);}
/** @param {string} file */
function copyX64(file) {directory(path.dirname(file)); fs.copyFileSync("/bin/true", file); return file;}

/** @param {string} parent @param {boolean} [cuda] */
function staticFixture(parent, cuda = false) {
  const root = directory(path.join(parent, "acpp")), receipt = path.join(root, "result.json"), source = directory(path.join(parent, "acpp-source"));
  const build = directory(path.join(root, "build")), rocm = directory(path.join(parent, "rocm")), hip = archive(path.join(rocm, "lib/libamdhip64.a"));
  let cudaInputs;
  if (cuda) {
    const sdk = directory(path.join(parent, "cuda-12.6")), libraryRoot = directory(path.join(sdk, "targets/x86_64-linux/lib")), alias = path.join(parent, "cuda");
    const libraries = [archive(path.join(libraryRoot, "libcudart_static.a")), archive(path.join(libraryRoot, "libculibos.a"))];
    fs.symlinkSync(sdk, alias, "dir"); fs.symlinkSync(libraryRoot, path.join(sdk, "lib64"), "dir");
    cudaInputs = {libraries, cachePaths: libraries.map(file => path.join(alias, "lib64", path.basename(file)))};
  }
  const cc = write(path.join(parent, "tools/icx")), cxx = write(path.join(parent, "tools/icpx"));
  const device = directory(path.join(parent, "device")), lld = directory(path.join(parent, "lld"));
  const comgrArchive = archive(path.join(parent, "comgr/libamd_comgr.a")), comgrHeader = write(path.join(parent, "comgr/include/amd_comgr.h"));
  const comgrReceipt = json(path.join(parent, "comgr/result.json"), {version: 1});
  const comgr = {archive: comgrArchive, header: comgrHeader, receipt: comgrReceipt, llvmVersion: "21.1.8",
    roots: {llvm: lld, lld, device}, files: hashes([comgrReceipt, comgrArchive, comgrHeader])};
  const hostCompilers = {cc, cxx}, hostCompilerInputs = hashes([cc, cxx]), compilerVersions = {[cc]: "Intel C fixture", [cxx]: "Intel C++ fixture"};
  const fingerprints = {...hostCompilerInputs, [cxx]: fileHash(cxx), ...comgr.files};
  const inputs = {source, build, revision: syclRevision, cxx, flags: [], libraries: [hip, ...(cudaInputs?.libraries || [])], links: ["-ldl"], comgr};
  const producerRecipe = recipe(syclRecipe), plan = {admitted: {source, revision: syclRevision}, version: "21.1.8",
    hostCompilers, hostCompilerInputs, compilerVersions, commands: [["cmake", "-S", source], ["cmake", "--build", build]], fingerprints, recipe: producerRecipe};
  const cache = [`CMAKE_HOME_DIRECTORY:INTERNAL=${source}`, "ACPP_STATIC_RUNTIME:BOOL=ON",
    `WITH_CUDA_BACKEND:BOOL=${cuda ? "ON" : "OFF"}`, "WITH_ROCM_BACKEND:BOOL=ON", "WITH_OPENCL_BACKEND:BOOL=OFF",
    "WITH_LEVEL_ZERO_BACKEND:BOOL=OFF", "WITH_VULKAN_BACKEND:BOOL=OFF",
    "ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER:BOOL=ON", "ACPP_CUSTOM_PROFILE_WITH_STDPAR_COMPILER:BOOL=OFF",
    "ACPP_CUSTOM_PROFILE_WITH_ACCELERATED_CPU:BOOL=OFF", `ACPP_RESOURCE_GENERATOR:FILEPATH=${path.join(repo, "scripts/native-resources.js")}`,
    ...(cudaInputs ? [`CUDA_cudart_static_LIBRARY:FILEPATH=${cudaInputs.cachePaths[0]}`,
      `CUDA_culibos_LIBRARY:FILEPATH=${cudaInputs.cachePaths[1]}`] : []),
    "CMAKE_DISABLE_FIND_PACKAGE_HIP:UNINITIALIZED=ON", `ROCM_PATH:PATH=${rocm}`,
    `ROCM_DEVICE_LIBS_PATH:PATH=${path.join(device, "amdgcn/bitcode")}`, `LLD_DIR:PATH=${path.join(lld, "lib/cmake/lld")}`].join("\n") + "\n";
  const cacheFile = path.join(build, "CMakeCache.txt"); write(cacheFile, cache);
  const record = /** @type {MutableReceipt} */ ({version: 1, backend: cuda ? "cuda-hip" : "hip", ...(cuda ? {cudaVersion: 12060} : {}),
    revision: syclRevision, inputs, hostCompilers, hostCompilerInputs, compilerVersions,
    fingerprints, libraries: hashes(inputs.libraries), recipe: producerRecipe, sysroot, buildOnly: true, publicationQualified: false,
    scope: "Fixture of the current build-only AdaptiveCpp result schema."});
  json(path.join(root, "plan.json"), plan); json(receipt, record);
  const patchText = ["runtime/adaptivecpp-static.patch", "runtime/adaptivecpp-jit.patch"].map(name =>
    fs.readFileSync(path.join(repo, name), "utf8")).join("");
  const gitRunner = /** @type {(command: string, args: string[]) => string} */ (function(_command, args) {
    if (args.includes("rev-parse")) {return syclRevision + "\n";}
    if (args.includes("diff")) {return patchText;}
    if (args.includes("ls-files")) {return "";}
    throw new Error("Unexpected Git fixture request");
  });
  return {root, receipt, record, comgr, gitRunner, readComgr: () => comgr, ...(cudaInputs ? {cuda: {...cudaInputs, cacheFile}} : {})};
}

/** @param {string} parent */
function openmpFixture(parent) {
  const root = directory(path.join(parent, "openmp-sdk")), receipt = json(path.join(root, "result.json"), {version: 1}),
    archiveFile = archive(path.join(root, "sdk/libomp.a")), files = hashes([receipt, archiveFile]);
  const admitted = {receipt, receiptSha256: fileHash(receipt), archive: archiveFile, files};
  return {root, receipt, archive: archiveFile, files, admitted,
    read: (/** @type {string} */ selectedRoot) => {assert.equal(selectedRoot, root); return admitted;}};
}

/** @param {string} parent @param {string} promiseSource */
function hermesFixture(parent, promiseSource) {
  const root = directory(path.join(parent, "hermes")), receipt = path.join(root, "result.json"), source = directory(path.join(root, "source")),
    build = directory(path.join(root, "build")), promise = "lib/InternalJavaScript/01-Promise.js";
  write(path.join(source, promise), fs.readFileSync(promiseSource));
  const cache = ["HERMESVM_INTERNAL_JAVASCRIPT_NATIVE:BOOL=ON", "HERMESVM_ALLOW_JIT:STRING=0", "HERMES_ENABLE_NAPI:BOOL=OFF",
    "HERMES_BUILD_SHARED_JSI:BOOL=OFF", "CMAKE_BUILD_TYPE:STRING=Release"].join("\n") + "\n";
  write(path.join(build, "CMakeCache.txt"), cache);
  const tools = Object.fromEntries(["git", "cmake", "cc", "cxx", "ninja"].map(name => [name, write(path.join(parent, "tools", name))]));
  const fingerprints = {...hashes(Object.values(tools)), [path.join(repo, "runtime/hermes-host-rejections.patch")]:
    fileHash(path.join(repo, "runtime/hermes-host-rejections.patch"))};
  const artifacts = {compiler: path.join(build, "bin/shermes"), context: path.join(build, "external/boost/boost_1_86_0/libs/context/libboost_context.a"),
    jsi: path.join(build, "jsi/libjsi.a"), public: path.join(build, "public/hermes/Public/libhermesPublic.a"), runtime: path.join(build, "lib/libhermesvmlean_a.a")};
  const artifactRecords = /** @type {Record<string, any>} */ ({});
  for (const [name, file] of Object.entries(artifacts)) {
    if (name === "compiler") {copyX64(file);} else {archive(file);}
    artifactRecords[name] = {path: file, sha256: fileHash(file), bytes: fs.statSync(file).size};
  }
  const record = /** @type {MutableReceipt} */ ({version: 1, target: "linux", pins: hermesPins, source, build, artifacts: artifactRecords, fingerprints,
    sources: {[promise]: fileHash(path.join(source, promise))}, recipe: recipe(hermesRecipe), sysroot, buildOnly: true, publicationQualified: false});
  json(receipt, record);
  const gitRunner = /** @type {(command: string, args: string[]) => string} */ (function(_command, args) {
    if (args.includes("rev-parse")) {return hermesPins.revision + "\n";}
    if (args.includes("diff")) {return promise + "\n";}
    if (args.includes("--others")) {return "";}
    if (args.includes("-z")) {return promise + "\0";}
    throw new Error("Unexpected Git fixture request");
  });
  return {receipt, promise, gitRunner};
}

/** @param {string} parent */
function rocmFixture(parent) {
  const root = directory(path.join(parent, "rocm-host")), prefix = directory(path.join(root, "sdk")),
    lib = directory(path.join(prefix, "lib")), include = directory(path.join(prefix, "include/hip")), build = directory(path.join(root, "build"));
  const sourcesRoot = directory(path.join(root, "source"));
  for (const name of ["clr", "hip", "rocr"]) {write(path.join(sourcesRoot, name, "CMakeLists.txt"));}
  const sourceArchives = /** @type {Record<string, any>} */ ({});
  for (const name of ["clr", "hip", "rocr"]) {
    const file = write(path.join(root, `source-${name}.tar`), `source archive ${name}\n`);
    sourceArchives[name] = {path: file, bytes: fs.statSync(file).size, sha256: fileHash(file)};
  }
  const beforePaths = /** @type {Record<string, string>} */ ({amdhip64: path.join(build, "clr/hipamd/lib", rocmProducer.archiveNames.amdhip64),
    hiprtc: path.join(build, "clr/hipamd/lib", rocmProducer.archiveNames.hiprtc),
    hsa: path.join(build, "rocr/runtime/hsa-runtime", rocmProducer.archiveNames.hsa),
    hsakmt: path.join(build, "rocr/libhsakmt", rocmProducer.archiveNames.hsakmt)});
  const archivesBefore = /** @type {Record<string, any>} */ ({}), archivesAfter = /** @type {Record<string, any>} */ ({});
  for (const [name, basename] of Object.entries(rocmProducer.archiveNames)) {
    const beforePath = beforePaths[name];
    if (!beforePath) {throw new Error("Missing ROCm test archive path: " + name);}
    const before = archive(beforePath), after = archive(path.join(lib, basename));
    archivesBefore[name] = {path: before, bytes: fs.statSync(before).size, sha256: fileHash(before)};
    archivesAfter[name] = {path: after, bytes: fs.statSync(after).size, sha256: fileHash(after)};
  }
  const installerOnly = archive(path.join(build, "rocr/libhsakmt/archive", rocmProducer.hsakmtInstallArchiveName));
  const installerOnlyArchives = {hsakmt: {path: installerOnly, bytes: fs.statSync(installerOnly).size, sha256: fileHash(installerOnly)}};
  for (const name of ["hip_runtime.h", "hip_runtime_api.h", "hiprtc.h", "hip_version.h"]) {write(path.join(include, name));}
  const dependencyNames = /** @type {Record<string, string>} */ ({drm: "libdrm.a", drmAmdgpu: "libdrm_amdgpu.a", elf: "libelf.a", numa: "libnuma.a"});
  const packageOwners = /** @type {Record<string, string>} */ ({drm: "libdrm", drmAmdgpu: "libdrm", elf: "libelf-dev", numa: "libnuma-dev"}),
    stagedDependencies = /** @type {Record<string, any>} */ ({}), originalDependencies = /** @type {string[]} */ ([]);
  for (const [name, basename] of Object.entries(dependencyNames)) {
    const original = archive(path.join(parent, "os", basename)), staged = archive(path.join(lib, basename)), sha256 = fileHash(original);
    originalDependencies.push(original);
    stagedDependencies[name] = {path: original, sha256, package: packageOwners[name], architecture: "amd64", version: "1.2.3-1",
      status: packageOwners[name] === "libdrm" ? "source" : "sysroot", staged, stagedSha256: fileHash(staged)};
  }
  const comgrArchive = archive(path.join(parent, "comgr/libamd_comgr.a")), comgrHeader = write(path.join(parent, "comgr/include/amd_comgr.h"));
  const comgrReceipt = json(path.join(parent, "comgr/result.json"), {version: 1});
  const roots = {llvm: directory(path.join(parent, "llvm")), lld: directory(path.join(parent, "lld")), device: directory(path.join(parent, "device"))};
  const comgr = {receipt: comgrReceipt, archive: comgrArchive, header: comgrHeader, llvmVersion: "21.1.8", roots,
    files: hashes([comgrReceipt, comgrArchive, comgrHeader])};
  const overlayRoot = path.join(root, "comgr-install-include"), overlayDestination = path.join(overlayRoot, "amd_comgr/amd_comgr.h");
  directory(path.dirname(overlayDestination)); fs.copyFileSync(comgrHeader, overlayDestination);
  const comgrHeaderOverlay = {source: comgrHeader, destination: overlayDestination, includeRoot: overlayRoot, sha256: fileHash(comgrHeader)};
  const compilerDir = directory(path.join(parent, "tools")), hostCompilers = {cc: write(path.join(compilerDir, "icx")), cxx: write(path.join(compilerDir, "icpx"))};
  const sdkInputs = {...comgr.files, ...hashes([...originalDependencies, ...Object.values(hostCompilers)])};
  const extraInput = write(path.join(parent, "external/include/fixture.h")), dependencies = hashes([extraInput]);
  const consumerAdmission = {include: path.join(prefix, "include"), libraries: Object.values(archivesAfter).map(row => row.path),
    comgr, links: ["-Wl,--whole-archive", archivesAfter["hsa"].path, "-Wl,--no-whole-archive"]};
  const configured = [];
  for (const name of ["rocr", "clr"]) {
    const buildRoot = directory(path.join(build, name)), cache = write(path.join(buildRoot, "CMakeCache.txt"), "fixture cache\n"),
      database = path.join(buildRoot, "compile_commands.json"), depfiles = [], rows = [];
    for (let i = 0; i < 8; i++) {
      const output = `object${i}.o`, depfile = path.join(buildRoot, output + ".d");
      write(depfile, `${output}: fixture.c\n`); depfiles.push(depfile);
      rows.push({directory: buildRoot, output, command: `${hostCompilers.cxx} -c fixture.c -o ${output}`});
    }
    json(database, rows);
    configured.push({name, cache, database, depfiles, commandCount: rows.length});
  }
  const osOrigins = Object.fromEntries(Object.entries(stagedDependencies).map(([name, item]) => [name, {
    path: item.path, sha256: item.sha256, package: item.package, architecture: item.architecture, version: item.version, status: item.status}]));
  const receipt = path.join(root, "result.json"), record = /** @type {MutableReceipt} */ ({version: 1, pins: rocmProducer.pins,
    sources: sourceInputs(root, ["source/clr", "source/hip", "source/rocr"], []), sourceArchives,
    hostCompilers, hostCompilerInputs: hashes(Object.values(hostCompilers)), compilerVersions: {[hostCompilers.cc]: "Intel(R) 2026.2", [hostCompilers.cxx]: "Intel(R) 2026.2"},
    toolVersions: {}, toolOrigins: {}, osOrigins, python: {}, hipPchTools: {}, comgrHeaderOverlay, comgr: {
      receipt: comgr.receipt, archive: comgr.archive, llvmVersion: comgr.llvmVersion, roots: comgr.roots}, sdkInputs,
    dependencies, configured, archivesBefore, archivesAfter, installerOnlyArchives, stagedDependencies, consumerAdmission,
    prefix, recipe: recipe(rocmRecipe), sysroot, buildOnly: true, executed: false, publicationQualified: false, scope: "ROCm host runtime fixture."});
  json(receipt, record);
  return {receipt, record, comgr, consumerAdmission};
}

/** @param {string} parent */
function nvrtcFixture(parent) {
  const root = directory(path.join(parent, "nvrtc")), sdk = directory(path.join(root, "sdk")), lib64 = directory(path.join(sdk, "lib64")),
    output = directory(path.join(root, "proof")), names = ["libnvrtc_static.a", "libnvrtc-builtins_static.a", "libnvptxcompiler_static.a"];
  const archives = names.map(name => archive(path.join(lib64, name))), files = ["compute_120.ptx", "compute_80.ptx", "sm_120.cubin", "sm_80.cubin"];
  const rows = files.map(name => {
    const data = `output ${name}\n`;
    for (const dir of ["actual", "expected", "lld"]) {write(path.join(output, dir, name), data);}
    const file = path.join(output, "actual", name);
    return {name, bytes: fs.statSync(file).size, sha256: fileHash(file)};
  });
  const executable = copyX64(path.join(output, "nvrtc-native")), lld = copyX64(path.join(output, "nvrtc-lld"));
  const sourceFile = "sycl/cn-gpu/cuda-expand-kernel.inc", constructorScript = "runtime/static-constructors.ld";
  const receipt = path.join(output, "result.json"), record = /** @type {MutableReceipt} */ ({sdk, bytes: fs.statSync(executable).size, needed: ["libm.so.6", "libc.so.6"],
    archives: archives.map(file => ({file, sha256: fileHash(file)})), files: rows,
    source: {file: sourceFile, sha256: fileHash(path.join(repo, sourceFile))},
    lld: {bytes: fs.statSync(lld).size, constructorScript: fileHash(path.join(repo, constructorScript)), byteExactOracle: true},
    scope: "Maintained CN/GPU source compiled statically, byte-exact shared compiler oracle; CPU only, no GPU/runtime/performance qualification"});
  json(receipt, record);
  return {receipt, record, output, sdk, archives};
}

/** @param {ReturnType<typeof nvrtcFixture>} fixture */
function nvrtcVendorFixture(fixture) {
  const source = directory(path.join(path.dirname(fixture.sdk), "vendor-source")), sourceReceipt = path.join(source, "fetch.json"),
    tool = write(path.join(source, "tools/tar")), archiveNames = ["nvrtc", "nvptxcompiler"],
    sourceArchives = archiveNames.map(name => {
      const file = write(path.join(source, name + ".tar.xz"), "original " + name + " archive\n");
      return file;
    });
  const sdkOutputs = [...fixture.archives,
    .../** @type {Array<[string, string]>} */ ([["include/nvrtc.h", "NVRTC header\n"], ["include/nvPTXCompiler.h", "NVPTX header\n"],
      ["licenses/nvrtc/LICENSE", "NVRTC license\n"], ["licenses/nvptxcompiler/LICENSE", "NVPTX license\n"]]).map(([relative, data]) =>
      write(path.join(fixture.sdk, relative), data))];
  json(sourceReceipt, {version: 1, archives: sourceArchives.map(file => ({file, sha256: fileHash(file)}))});
  const sdkReceipt = path.join(fixture.sdk, "sdk.json");
  json(sdkReceipt, {version: 1, source: "fixture"});
  const recipe = sourceInputs(repo, [], ["scripts/prepare-native-nvrtc-sdk.js", "scripts/fetch-native-sdk-sources.js",
      "scripts/probe-static-nvrtc.js", "scripts/build-source-inputs.js"]).graph,
    recipeFiles = Object.keys(recipe).map(relative => path.join(repo, ...relative.split("/"))),
    files = Object.fromEntries([...new Set([sdkReceipt, sourceReceipt, tool, ...sourceArchives, ...sdkOutputs, ...recipeFiles])]
      .map(file => [file, fileHash(file)]));
  const inputs = {receipt: {version: 1}, receiptFile: sdkReceipt, receiptSha256: "a".repeat(64), files};
  return {inputs, sourceReceipt, sourceArchives, tool, sdkReceipt, sdkOutputs, recipe};
}

describe("Linux native release SDK receipt admission", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("source Git probes ignore ambient repository overrides and PATH shadowing", () => {
    const source = directory(path.join(scratch, "git-probe")), shadow = directory(path.join(scratch, "shadow"));
    const created = spawnSync("/usr/bin/git", ["-c", "init.defaultBranch=main", "init", source], {encoding: "utf8"});
    assert.equal(created.status, 0, created.stderr);
    const fake = write(path.join(shadow, "git"), "#!/bin/sh\nprintf 'shadowed\\n'\n"); fs.chmodSync(fake, 0o700);
    const previous = {PATH: process.env["PATH"], GIT_DIR: process.env["GIT_DIR"], GIT_CONFIG_COUNT: process.env["GIT_CONFIG_COUNT"]};
    try {
      Object.assign(process.env, {PATH: shadow, GIT_DIR: path.join(scratch, "missing"), GIT_CONFIG_COUNT: "invalid"});
      assert.equal(gitOutput("git", ["-C", source, "rev-parse", "--show-toplevel"]).trim(), source);
    } finally {
      for (const [key, value] of Object.entries(previous)) {
        if (value === undefined) {delete process.env[key];} else {process.env[key] = value;}
      }
    }
  });

  test("AdaptiveCpp reader binds its result, plan, current source recipe and output archive hashes", () => {
    const fixture = staticFixture(path.join(scratch, "static-positive")), before = fs.readdirSync(fixture.root).sort();
    const admitted = nativeStaticRuntimeInputs(fixture.receipt, {gitRunner: fixture.gitRunner, readComgr: fixture.readComgr});
    assert.deepEqual(admitted.libraries, fixture.record.inputs.libraries);
    assert.equal(admitted.backend, "hip");
    assert.equal(admitted.files[fixture.record.inputs.libraries[0]], fixture.record.libraries[fixture.record.inputs.libraries[0]]);
    assert.deepEqual(fs.readdirSync(fixture.root).sort(), before);
  });

  test("AdaptiveCpp reader binds only the OpenMP SDK archive and rewrites the final direct link", () => {
    const fixture = staticFixture(path.join(scratch, "static-openmp")), openmp = openmpFixture(path.dirname(fixture.receipt));
    fixture.record.inputs.libraries.push("/openmp/libomp.a");
    fixture.record.inputs.links.push("/openmp/libomp.a");
    fixture.record.libraries["/openmp/libomp.a"] = fileHash(openmp.archive);
    json(fixture.receipt, fixture.record);
    const admitted = nativeStaticRuntimeInputs(fixture.receipt, {openmpSdkRoot: openmp.root, readOpenmp: openmp.read,
      gitRunner: fixture.gitRunner, readComgr: fixture.readComgr});
    assert.equal(admitted.libraries.at(-1), openmp.archive);
    assert.equal(admitted.links.at(-1), openmp.archive);
    assert.equal(admitted.pathBindings["/openmp/libomp.a"], openmp.archive);
    assert.equal(admitted.files[openmp.archive], fileHash(openmp.archive));
    assert.deepEqual(admitted.openmp, openmp.admitted);
  });

  test("AdaptiveCpp reader accepts the admitted canonical OpenMP path without an alias remap", () => {
    const fixture = staticFixture(path.join(scratch, "static-openmp-direct")), openmp = openmpFixture(path.dirname(fixture.receipt));
    fixture.record.inputs.libraries.push(openmp.archive);
    fixture.record.inputs.links.push(openmp.archive, `-Wl,--whole-archive,${openmp.archive},--no-whole-archive`);
    fixture.record.libraries[openmp.archive] = fileHash(openmp.archive);
    json(fixture.receipt, fixture.record);
    const admitted = nativeStaticRuntimeInputs(fixture.receipt, {openmpSdkRoot: openmp.root, readOpenmp: openmp.read,
      gitRunner: fixture.gitRunner, readComgr: fixture.readComgr});
    assert.equal(admitted.libraries.at(-1), openmp.archive);
    assert.deepEqual(admitted.links.slice(-2), [openmp.archive, `-Wl,--whole-archive,${openmp.archive},--no-whole-archive`]);
    assert.equal(admitted.pathBindings["/openmp/libomp.a"], undefined);
    assert.equal(admitted.files[openmp.archive], openmp.files[openmp.archive]);
  });

  test("AdaptiveCpp OpenMP binding rejects a second provider in the declared libraries", () => {
    for (const [index, name] of ["libiomp5.a", "libgomp.a"].entries()) {
      const fixture = staticFixture(path.join(scratch, "static-openmp-second-library-" + index)),
        openmp = openmpFixture(path.dirname(fixture.receipt)), alternate = archive(path.join(path.dirname(fixture.receipt), "alternate-provider", name));
      fixture.record.inputs.libraries.push(openmp.archive, alternate);
      fixture.record.libraries[openmp.archive] = fileHash(openmp.archive);
      fixture.record.libraries[alternate] = fileHash(alternate);
      json(fixture.receipt, fixture.record);
      assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {openmpSdkRoot: openmp.root, readOpenmp: openmp.read,
        gitRunner: fixture.gitRunner, readComgr: fixture.readComgr}), /must select exactly the admitted OpenMP SDK archive/, name);
    }
  });

  test("AdaptiveCpp OpenMP binding rejects alternate direct, wrapped, and short-name providers", () => {
    const cases = [
      {name: "plain libomp archive outside an openmp directory", file: "libomp.a", mode: "path"},
      {name: "wrapped shared libgomp path", file: "libgomp.so.1", mode: "wrappedPath"},
      {name: "shared libiomp5 path", file: "libiomp5.so", mode: "path"},
      {name: "-liomp5", file: "libiomp5.a", mode: "short"},
      {name: "-lgomp", file: "libgomp.a", mode: "short"},
      {name: "-l:libomp.a", file: "libomp.a", mode: "colon"},
      {name: "wrapped -liomp5", file: "libiomp5.a", mode: "wrappedShort"},
      {name: "wrapped -lgomp", file: "libgomp.a", mode: "wrappedShort"},
      {name: "wrapped -l:libomp.a", file: "libomp.a", mode: "wrappedColon"},
      {name: "wrapped -l,omp", file: "libomp.a", mode: "wrappedSeparated"},
    ];
    for (const [index, entry] of cases.entries()) {
      const fixture = staticFixture(path.join(scratch, "static-openmp-alternate-link-" + index)),
        openmp = openmpFixture(path.dirname(fixture.receipt)), providerRoot = directory(path.join(path.dirname(fixture.receipt), "alternate-provider"));
      fixture.record.inputs.libraries.push("/openmp/libomp.a");
      fixture.record.libraries["/openmp/libomp.a"] = fileHash(openmp.archive);
      const file = path.join(providerRoot, entry.file), provider = entry.file.endsWith(".a") ? archive(file) : write(file);
      fixture.record.fingerprints[provider] = fileHash(provider);
      const stem = entry.file.match(/^lib(omp|iomp5|gomp)/)?.[1];
      switch (entry.mode) {
        case "path": fixture.record.inputs.links.push(provider); break;
        case "wrappedPath": fixture.record.inputs.links.push("-Wl," + provider); break;
        case "short": fixture.record.inputs.links.push("-L", providerRoot, "-l" + stem); break;
        case "colon": fixture.record.inputs.links.push("-L", providerRoot, "-l:" + entry.file); break;
        case "wrappedShort": fixture.record.inputs.links.push("-Wl,-l" + stem); break;
        case "wrappedColon": fixture.record.inputs.links.push("-Wl,-l:" + entry.file); break;
        default: fixture.record.inputs.links.push("-Wl,-l,omp");
      }
      const planFile = path.join(path.dirname(fixture.receipt), "plan.json"), plan = JSON.parse(fs.readFileSync(planFile, "utf8"));
      plan.fingerprints = fixture.record.fingerprints; json(planFile, plan);
      json(fixture.receipt, fixture.record);
      assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {openmpSdkRoot: openmp.root, readOpenmp: openmp.read,
        gitRunner: fixture.gitRunner, readComgr: fixture.readComgr}), /AdaptiveCpp link bypasses the admitted OpenMP archive/, entry.name);
    }
  });

  test("an ambient OpenMP mount with matching bytes cannot substitute for the admitted SDK path", () => {
    const fixture = staticFixture(path.join(scratch, "static-openmp-ambient")), openmp = openmpFixture(path.dirname(fixture.receipt)),
      ambient = archive(path.join(path.dirname(fixture.receipt), "ambient-libomp.a"));
    fs.copyFileSync(openmp.archive, ambient);
    fixture.record.inputs.libraries.push("/openmp/libomp.a");
    fixture.record.libraries["/openmp/libomp.a"] = fileHash(openmp.archive);
    json(fixture.receipt, fixture.record);
    const originalLstat = fs.lstatSync, originalStat = fs.statSync, originalRealpath = fs.realpathSync;
    /** @param {string} file @param {{bigint?: boolean, throwIfNoEntry?: boolean}} [options] */
    const lstatMount = (file, options) => file === "/openmp/libomp.a" ? originalLstat(ambient, options) : originalLstat(file, options);
    /** @param {string} file @param {{bigint?: boolean, throwIfNoEntry?: boolean}} [options] */
    const statMount = (file, options) => file === "/openmp/libomp.a" ? originalStat(ambient, options) : originalStat(file, options);
    /** @param {string} file */
    const realpathMount = file => file === "/openmp/libomp.a" ? originalRealpath(ambient) : originalRealpath(file);
    Object.assign(fs, {lstatSync: lstatMount, statSync: statMount, realpathSync: realpathMount});
    try {
      assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {openmpSdkRoot: openmp.root, readOpenmp: openmp.read,
        gitRunner: fixture.gitRunner, readComgr: fixture.readComgr}), /Ambient OpenMP mount conflicts with the admitted SDK archive/);
    } finally {Object.assign(fs, {lstatSync: originalLstat, statSync: originalStat, realpathSync: originalRealpath});}
  });

  test("AdaptiveCpp OpenMP binding rejects a stale SDK receipt, another archive path, or absent mount membership", () => {
    const fixture = staticFixture(path.join(scratch, "static-openmp-provenance")), openmp = openmpFixture(path.dirname(fixture.receipt));
    fixture.record.inputs.libraries.push("/openmp/libomp.a");
    fixture.record.libraries["/openmp/libomp.a"] = fileHash(openmp.archive);
    json(fixture.receipt, fixture.record);
    const options = {openmpSdkRoot: openmp.root, readOpenmp: openmp.read, gitRunner: fixture.gitRunner, readComgr: fixture.readComgr};
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {...options, readOpenmp: () => ({...openmp.admitted, receiptSha256: "0".repeat(64)})}),
      /does not bind the admitted OpenMP SDK receipt and archive/);
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {...options, readOpenmp: () => ({...openmp.admitted, archive: openmp.archive + ".other"})}),
      /does not bind the admitted OpenMP SDK receipt and archive/);
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {...options,
      readOpenmp: (/** @type {string} */ root) => ({...openmp.admitted, receipt: path.join(root, "other.json")})}),
    /does not bind the admitted OpenMP SDK receipt and archive/);
    fixture.record.inputs.libraries = fixture.record.inputs.libraries.filter((/** @type {string} */ file) => file !== "/openmp/libomp.a");
    delete fixture.record.libraries["/openmp/libomp.a"]; json(fixture.receipt, fixture.record);
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, options), /must select exactly the admitted OpenMP SDK archive/);
  });

  test("AdaptiveCpp reader rejects a direct archive with wrong bytes or a second libomp provider", () => {
    const fixture = staticFixture(path.join(scratch, "static-openmp-direct-negative")), openmp = openmpFixture(path.dirname(fixture.receipt));
    fixture.record.inputs.libraries.push(openmp.archive);
    fixture.record.libraries[openmp.archive] = "0".repeat(64);
    json(fixture.receipt, fixture.record);
    const options = {openmpSdkRoot: openmp.root, readOpenmp: openmp.read, gitRunner: fixture.gitRunner, readComgr: fixture.readComgr};
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, options), /must select exactly the admitted OpenMP SDK archive/);
    const other = archive(path.join(path.dirname(openmp.root), "other/libomp.a"));
    fixture.record.libraries[openmp.archive] = fileHash(openmp.archive);
    fixture.record.inputs.libraries.push(other); fixture.record.libraries[other] = fileHash(other); json(fixture.receipt, fixture.record);
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, options), /must select exactly the admitted OpenMP SDK archive/);
  });

  test("AdaptiveCpp reader admits a pinned CUDA target through SDK symlinks, not another path or changed bytes", () => {
    const fixture = staticFixture(path.join(scratch, "static-cuda-symlink"), true);
    const cuda = fixture.cuda;
    assert.ok(cuda);
    const runtimeArchive = cuda.libraries[0];
    assert.ok(runtimeArchive);
    const syclBuilder = require("../scripts/build-native-sycl-sdk"), pinnedAdmission = syclBuilder.admitCudaLibraries;
    // These synthetic ar files cannot be the pinned NVIDIA payloads. Mock only the existing
    // package-pin predicate so this public receipt-reader fixture can isolate CMake path identity.
    syclBuilder.admitCudaLibraries = libraries => assert.deepEqual(libraries, cuda.libraries);
    try {
      const options = {gitRunner: fixture.gitRunner, readComgr: fixture.readComgr};
      const admitted = nativeStaticRuntimeInputs(fixture.receipt, options);
      assert.equal(admitted.backend, "cuda-hip");
      assert.deepEqual(admitted.libraries.slice(1), cuda.libraries);

      const otherSdk = directory(path.join(fixture.root, "cuda-13.2")), otherLibraries = directory(path.join(otherSdk, "targets/x86_64-linux/lib"));
      fs.copyFileSync(runtimeArchive, path.join(otherLibraries, "libcudart_static.a"));
      fs.symlinkSync(otherSdk, path.join(fixture.root, "cuda-other"), "dir");
      fs.symlinkSync(otherLibraries, path.join(otherSdk, "lib64"), "dir");
      const original = fs.readFileSync(cuda.cacheFile, "utf8"), wrongTarget = path.join(fixture.root, "cuda-other/lib64/libcudart_static.a");
      write(cuda.cacheFile, original.replace(/^CUDA_cudart_static_LIBRARY:[^\r\n]+$/m,
        `CUDA_cudart_static_LIBRARY:FILEPATH=${wrongTarget}`));
      assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, options), /Native SDK CUDA cache differs: CUDA_cudart_static_LIBRARY/);
    } finally {syclBuilder.admitCudaLibraries = pinnedAdmission;}

    const changed = staticFixture(path.join(scratch, "static-cuda-changed"), true);
    const changedCuda = changed.cuda;
    assert.ok(changedCuda);
    const changedRuntimeArchive = changedCuda.libraries[0];
    assert.ok(changedRuntimeArchive);
    fs.writeFileSync(changedRuntimeArchive, Buffer.from("!<arch>\nchanged target bytes\n"));
    assert.throws(() => nativeStaticRuntimeInputs(changed.receipt, {gitRunner: changed.gitRunner, readComgr: changed.readComgr}),
      /SDK input differs from its receipt/);
  });

  test("AdaptiveCpp reader rejects stale revision, output hash and unadmitted OpenMP archive bytes", () => {
    const fixture = staticFixture(path.join(scratch, "static-negative"));
    fixture.record.revision = "0".repeat(40); json(fixture.receipt, fixture.record);
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {gitRunner: fixture.gitRunner, readComgr: fixture.readComgr}), /Invalid static AdaptiveCpp/);
    fixture.record.revision = syclRevision;
    const library = fixture.record.inputs.libraries[0], libraryHash = fileHash(library);
    fixture.record.libraries[library] = "0".repeat(64); json(fixture.receipt, fixture.record);
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {gitRunner: fixture.gitRunner, readComgr: fixture.readComgr}), /differs from its receipt/);
    const openmp = openmpFixture(path.dirname(fixture.receipt));
    fixture.record.libraries[library] = libraryHash;
    fixture.record.inputs.libraries.push("/openmp/libomp.a"); fixture.record.libraries["/openmp/libomp.a"] = "0".repeat(64); json(fixture.receipt, fixture.record);
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {openmpSdkRoot: openmp.root, readOpenmp: openmp.read,
      gitRunner: fixture.gitRunner, readComgr: fixture.readComgr}), /must select exactly the admitted OpenMP SDK archive/);
  });

  test("AdaptiveCpp reader rejects malformed fingerprint maps", () => {
    const fixture = staticFixture(path.join(scratch, "static-malformed-map")), file = Object.keys(fixture.record.libraries)[0];
    assert.ok(file);
    fixture.record.libraries[file] = "not-a-sha256";
    json(fixture.receipt, fixture.record);
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {gitRunner: fixture.gitRunner, readComgr: fixture.readComgr}), /Invalid SDK input hash row/);
  });

  test("AdaptiveCpp admits an unchanged resource generator from its original checkout, never changed bytes", () => {
    const fixture = staticFixture(path.join(scratch, "static-generator-checkout")),
      generator = write(path.join(scratch, "original-checkout/scripts/native-resources.js"), fs.readFileSync(path.join(repo, "scripts/native-resources.js"))),
      cache = path.join(fixture.record.inputs.build, "CMakeCache.txt");
    write(cache, fs.readFileSync(cache, "utf8").replace(/^ACPP_RESOURCE_GENERATOR:[^\r\n]+$/m,
      `ACPP_RESOURCE_GENERATOR:FILEPATH=${generator}`));
    const originalReceipt = fs.readFileSync(fixture.receipt);
    assert.equal(nativeStaticRuntimeInputs(fixture.receipt, {gitRunner: fixture.gitRunner, readComgr: fixture.readComgr}).backend, "hip");
    assert.deepEqual(fs.readFileSync(fixture.receipt), originalReceipt, "admission must not rewrite producer evidence");
    fs.appendFileSync(generator, "\n// different generator\n");
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {gitRunner: fixture.gitRunner, readComgr: fixture.readComgr}), /resource generator differs/);
    fs.unlinkSync(generator); fs.symlinkSync(path.join(repo, "scripts/native-resources.js"), generator);
    assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {gitRunner: fixture.gitRunner, readComgr: fixture.readComgr}), /canonical/);
  });

  test("AdaptiveCpp reader requires the exact COMGR device bitcode subpath", () => {
    const fixture = staticFixture(path.join(scratch, "static-device-path")), cacheFile = path.join(fixture.record.inputs.build, "CMakeCache.txt"),
      original = fs.readFileSync(cacheFile, "utf8"), deviceRoot = fixture.comgr.roots.device;
    for (const wrong of [deviceRoot, path.join(deviceRoot, "amdgcn/not-bitcode")]) {
      write(cacheFile, original.replace(/^ROCM_DEVICE_LIBS_PATH:[^\r\n]+$/m, `ROCM_DEVICE_LIBS_PATH:PATH=${wrong}`));
      assert.throws(() => nativeStaticRuntimeInputs(fixture.receipt, {gitRunner: fixture.gitRunner, readComgr: fixture.readComgr}),
        /Native SDK CMake cache differs: ROCM_DEVICE_LIBS_PATH/);
    }
  });

  test("Hermes reader rejects maintained patch text in place of its pinned patched Promise source", () => {
    const fixture = hermesFixture(path.join(scratch, "hermes-promise-mismatch"), path.join(repo, "runtime/hermes-host-rejections.patch"));
    assert.throws(() => nativeHermesInputs(fixture.receipt, {gitRunner: fixture.gitRunner}), /Hermes source tree differs/);
  });

  test("Hermes binds original-checkout patch bytes without requiring the consumer's installation path", () => {
    const fixture = hermesFixture(path.join(scratch, "hermes-patch-checkout"), path.join(repo, "runtime/hermes-host-rejections.patch")),
      record = JSON.parse(fs.readFileSync(fixture.receipt, "utf8")), patch = path.join(repo, "runtime/hermes-host-rejections.patch"),
      original = write(path.join(scratch, "original-hermes/runtime/hermes-host-rejections.patch"), fs.readFileSync(patch));
    record.fingerprints[original] = record.fingerprints[patch]; delete record.fingerprints[patch]; json(fixture.receipt, record);
    const before = fs.readFileSync(fixture.receipt);
    // Deliberately invalid Promise fixture: pass the patch path gate, then reject source contents.
    assert.throws(() => nativeHermesInputs(fixture.receipt, {gitRunner: fixture.gitRunner}), /Hermes source tree differs/);
    assert.deepEqual(fs.readFileSync(fixture.receipt), before);
    fs.appendFileSync(original, "\nchanged patch\n"); record.fingerprints[original] = fileHash(original); json(fixture.receipt, record);
    assert.throws(() => nativeHermesInputs(fixture.receipt, {gitRunner: fixture.gitRunner}), /tool\/patch fingerprints are incomplete/);
    fs.writeFileSync(original, fs.readFileSync(patch)); record.fingerprints[original] = fileHash(original);
    const other = write(path.join(scratch, "duplicate-hermes/hermes-host-rejections.patch"), fs.readFileSync(patch));
    const ninja = Object.keys(record.fingerprints).find(file => path.basename(file) === "ninja"); assert.ok(ninja);
    delete record.fingerprints[ninja];
    record.fingerprints[other] = fileHash(other); json(fixture.receipt, record);
    assert.equal(Object.keys(record.fingerprints).length, 6);
    assert.throws(() => nativeHermesInputs(fixture.receipt, {gitRunner: fixture.gitRunner}), /tool\/patch fingerprints are incomplete/);
    delete record.fingerprints[other];
    record.fingerprints[ninja] = fileHash(ninja);
    fs.unlinkSync(original); fs.symlinkSync(patch, original); json(fixture.receipt, record);
    assert.throws(() => nativeHermesInputs(fixture.receipt, {gitRunner: fixture.gitRunner}), /Hermes patch must be an absolute canonical path|Hermes patch must be a canonical/);
  });

  test("Hermes reader rejects a receipt with a different source revision", () => {
    const root = directory(path.join(scratch, "hermes-negative")), receipt = path.join(root, "result.json");
    json(receipt, {version: 1, target: "linux", pins: {revision: "0".repeat(40), promiseSha256: hermesPins.promiseSha256},
      buildOnly: true, publicationQualified: false, source: path.join(root, "source"), build: path.join(root, "build")});
    assert.throws(() => nativeHermesInputs(receipt), /Invalid static Hermes SDK receipt/);
  });

  test("ROCm host reader binds source snapshots, staged archives, package hashes and consumer admission", () => {
    const fixture = rocmFixture(path.join(scratch, "rocm-positive")), before = fs.readdirSync(path.dirname(fixture.receipt)).sort();
    const admitted = nativeRocmHostInputs(fixture.receipt, {readComgr: () => fixture.comgr, readConsumer: () => fixture.consumerAdmission});
    assert.equal(admitted.prefix, path.join(path.dirname(fixture.receipt), "sdk"));
    assert.equal(admitted.libraries.length, 4);
    assert.ok(admitted.files[path.join(admitted.prefix, "lib/libnuma.a")]);
    assert.deepEqual(fs.readdirSync(path.dirname(fixture.receipt)).sort(), before);
  });

  test("ROCm host reader rejects conflicting hashes for one canonical input", () => {
    const fixture = rocmFixture(path.join(scratch, "rocm-negative")), collision = Object.keys(fixture.record.sdkInputs)[0];
    assert.ok(collision);
    fixture.record.dependencies[collision] = "0".repeat(64); json(fixture.receipt, fixture.record);
    assert.throws(() => nativeRocmHostInputs(fixture.receipt, {readComgr: () => fixture.comgr, readConsumer: () => fixture.consumerAdmission}), /Conflicting SDK hashes/);
  });

  test("NVRTC reader admits its exact static/shared/LLD proof and rejects extra outputs or stale hashes", () => {
    const fixture = nvrtcFixture(path.join(scratch, "nvrtc-positive")), before = fs.readdirSync(fixture.output).sort();
    const admitted = nativeNvrtcInputs(fixture.receipt);
    assert.deepEqual(admitted.archives, fixture.archives);
    assert.equal(Object.keys(admitted.files).length, 3);
    assert.throws(() => nativeNvrtcInputs(fixture.receipt, {requireVendorProvenance: true}), /require NVRTC vendor provenance/);
    assert.deepEqual(fs.readdirSync(fixture.output).sort(), before);

    const firstArchive = fixture.record.archives[0];
    assert.ok(firstArchive);
    const originalHash = firstArchive.sha256;
    firstArchive.sha256 = "0".repeat(64); json(fixture.receipt, fixture.record);
    assert.throws(() => nativeNvrtcInputs(fixture.receipt), /NVRTC archive differs/);
    firstArchive.sha256 = originalHash; json(fixture.receipt, fixture.record);
    write(path.join(fixture.output, "actual/unrecorded.ptx"));
    assert.throws(() => nativeNvrtcInputs(fixture.receipt), /output membership differs: actual/);
  });

  test("NVRTC CLI admits a supplied SDK before invoking a compiler or creating proof output", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-nvrtc-cli-admission-")), sdk = path.join(root, "sdk"),
      evidence = path.join(root, "evidence"), compiler = path.join(root, "compiler"), invoked = path.join(root, "compiler-invoked");
    t.after(() => {fs.chmodSync(sdk, 0o700); fs.rmSync(root, {recursive: true, force: true});});
    fs.mkdirSync(sdk); fs.mkdirSync(evidence);
    fs.writeFileSync(path.join(sdk, "sdk.json"), "{}\n", {mode: 0o444}); fs.chmodSync(sdk, 0o555);
    fs.writeFileSync(compiler, "#!/bin/sh\nprintf invoked > " + JSON.stringify(invoked) + "\nexit 77\n", {mode: 0o755});
    const result = spawnSync(process.execPath, [path.join(repo, "scripts/probe-static-nvrtc.js"), sdk, evidence],
      {encoding: "utf8", timeout: 10000, maxBuffer: 65536, env: {...process.env, CXX: compiler}});
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /ENOENT|NVRTC SDK receipt/);
    assert.doesNotMatch(result.stderr, /Cannot read properties of undefined/);
    assert.equal(fs.existsSync(invoked), false, "unadmitted vendor code must not reach the compiler/oracle");
    assert.deepEqual(fs.readdirSync(evidence), [], "rejected SDKs must not publish partial compiler evidence");
  });

  test("NVRTC source-composed admission binds shared SDK closure and requires its receipt hash", t => {
    const fixture = nvrtcFixture(path.join(scratch, "nvrtc-vendor")), vendorFixture = nvrtcVendorFixture(fixture),
      producer = require("../scripts/prepare-native-nvrtc-sdk");
    fixture.record["vendor"] = {receiptSha256: vendorFixture.inputs.receiptSha256}; json(fixture.receipt, fixture.record);
    let currentInputs = vendorFixture.inputs, calls = 0;
    /** @param {"linux" | "windows"} platform @param {string} root */
    function readNvrtcSdkInputs(platform, root) {
      assert.equal(platform, "linux"); assert.equal(root, fixture.sdk); calls++;
      for (const [file, sha256] of Object.entries(currentInputs.files)) {
        if (fileHash(file) !== sha256) {throw new Error("mock admission reports changed SDK input: " + file);}
      }
      return currentInputs;
    }
    t.mock.method(producer, "nvrtcSdkInputs", readNvrtcSdkInputs);
    const admitted = nativeNvrtcInputs(fixture.receipt, {requireVendorProvenance: true});
    assert.deepEqual(admitted.archives, fixture.archives);
    for (const [file, sha256] of Object.entries(vendorFixture.inputs.files)) {
      assert.equal(admitted.files[file], sha256, "vendor closure omits " + file);
    }
    assert.equal(Object.keys(vendorFixture.recipe).length, 4);
    for (const [relative, sha256] of Object.entries(vendorFixture.recipe)) {
      assert.equal(vendorFixture.inputs.files[path.join(repo, ...relative.split("/"))], sha256,
        "vendor recipe input omitted from final-link snapshot: " + relative);
    }
    assert.equal(calls, 1);

    const license = vendorFixture.sdkOutputs.find(file => file.endsWith("/licenses/nvrtc/LICENSE"));
    assert.ok(license);
    const original = fs.readFileSync(license);
    try {
      write(license, Buffer.concat([original, Buffer.from("changed\n")]));
      assert.throws(() => nativeNvrtcInputs(fixture.receipt, {requireVendorProvenance: true}), /mock admission reports changed SDK input/);
    } finally {
      fs.writeFileSync(license, original);
    }

    currentInputs = {...vendorFixture.inputs, receiptSha256: "0".repeat(64)};
    assert.throws(() => nativeNvrtcInputs(fixture.receipt, {requireVendorProvenance: true}), /NVRTC vendor SDK receipt differs from compiler proof/);
    currentInputs = vendorFixture.inputs;
    fixture.record["vendor"] = {receiptSha256: "0".repeat(64)};
    json(fixture.receipt, fixture.record);
    assert.throws(() => nativeNvrtcInputs(fixture.receipt, {requireVendorProvenance: true}),
      /NVRTC vendor SDK receipt differs from compiler proof/);
  });

  test("NVRTC reader rejects a static archive symlink that escapes the selected SDK input root", () => {
    const fixture = nvrtcFixture(path.join(scratch, "nvrtc-escape")), outside = archive(path.join(scratch, "outside-sdk.a"));
    const firstArchive = fixture.archives[0];
    assert.ok(firstArchive);
    fs.unlinkSync(firstArchive); fs.symlinkSync(outside, firstArchive);
    assert.throws(() => nativeNvrtcInputs(fixture.receipt), /escapes its selected input root/);
  });

  test("NVRTC reader rejects an archive path swapped to another in-root archive", () => {
    const fixture = nvrtcFixture(path.join(scratch, "nvrtc-path-swap")), replacement = path.join(fixture.sdk, "lib64/replacement.a");
    write(replacement, Buffer.concat([Buffer.from("!<arch>\n"), Buffer.from("different archive bytes\n")]));
    const firstArchive = fixture.archives[0];
    assert.ok(firstArchive);
    fs.unlinkSync(firstArchive); fs.symlinkSync(replacement, firstArchive);
    assert.throws(() => nativeNvrtcInputs(fixture.receipt), /NVRTC archive differs/);
  });
});
