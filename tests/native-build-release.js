"use strict";
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), assert = require("node:assert/strict"), test = require("node:test");
const {spawnSync} = require("node:child_process");
const {runInNewContext} = require("node:vm");
const {absolutePath, nativeSdkConfig, nativeSdkLinkInputs, nativeSdkSourceInputs, nativeReleaseComponents, nativeStage, compactNativeRelease, buildNativeRelease} = require("../scripts/build-native-release");
const {run: runNativeControllerCommand, flushLinuxObjectBatch} = require("../scripts/build-native-application");
const {buildNativeArtifact, nativeReleaseArguments, packNativeEntrypoint} = require("../scripts/build-release");
const {packageNativeRelease} = require("../scripts/package-native-release");
const {ioPins} = require("../scripts/build-native-io");
const {fileHash, nativeSourceInputs} = require("../scripts/build-source-inputs");
const {elfSections, elfRangeHash, nativeElf, verifyNativeStrip} = require("../scripts/native-elf");
const {linuxSysrootRecord, linuxSysrootReceiptSha256} = require("../scripts/native-linux-sysroot");
const {tempDir} = require("./common/temp-dir");
const sysroot = linuxSysrootRecord({receiptSha256: linuxSysrootReceiptSha256});
const repo = path.resolve(__dirname, "..");
/** @typedef {typeof import('../scripts/native-release-sdk-inputs') & {nativeZstdInputs: typeof import('../scripts/build-native-zstd').nativeZstdInputs,
 * nativeOpenSslInputs: typeof import('../scripts/build-native-openssl').nativeOpenSslInputs,
 * cudaImageInputs: typeof import('../scripts/native-cuda-images').cudaImageInputs}} ImageSourceReaders */
function configuration() {
  return {version: 1, cpu: {cache: "/sdk/cache", prepareCache: false},
    controller: {source: "/sdk/hermes", build: "/sdk/hermes-build", openssl: "/sdk/openssl", io: "/sdk/io", cc: "/sdk/cc", nvml: "/sdk/nvml.h"},
    sdk: {defaultCompiler: "oneapi", cxx: "/sdk/cxx", flags: /** @type {string[]} */ ([]), libraries: /** @type {string[]} */ ([]),
      links: /** @type {string[]} */ ([]), sdkObjects: /** @type {{object: string}[]} */ ([])},
    gpu: {oneapi: {args: ["/sdk/dpcpp"]}, dpcpp: {args: ["/sdk/dpcpp", "/sdk/cuda"]}, "dpcpp-opencl": {args: ["/sdk/dpcpp"]},
      "acpp-cuda": {args: ["/sdk/acpp", "/sdk/cuda-runtime"]}, "acpp-hip": {args: ["/sdk/acpp", "/sdk/hip-runtime"]}}};
}

/** A relocated raw build handoff; the recorded SDK/config paths intentionally are not present.
 * @param {import("node:test").TestContext} t */
function packingFixture(t) {
  const root = tempDir(t, "mom-native-release-"), origin = path.join(root, "origin"), transport = path.join(root, "transport"), output = path.join(root, "packed");
  fs.mkdirSync(origin); fs.mkdirSync(transport);
  const originalBinary = path.join(origin, "mom"), sourcePath = path.join(transport, "mom"),
    originalProducer = path.join(origin, "result.json"), producerReceiptPath = path.join(transport, "result.json"),
    entrypointPath = path.join(origin, "entrypoint.json"), upx = path.join(root, "upx"), configFile = path.join(origin, "missing-sdk.json");
  const raw = Buffer.alloc(64, 0x5a); Buffer.from("7f454c46", "hex").copy(raw);
  fs.writeFileSync(originalBinary, raw, {mode: 0o755}); fs.writeFileSync(sourcePath, raw, {mode: 0o755});
  fs.writeFileSync(upx, "fixture UPX tool\n", {mode: 0o755});
  const sourceHash = fileHash(sourcePath), configHash = "a".repeat(64), producer = {version: "0.9.4", configHash,
    binary: originalBinary, bytes: raw.length, sha256: sourceHash, buildOnly: true, publicationQualified: false};
  fs.writeFileSync(originalProducer, JSON.stringify(producer) + "\n"); fs.copyFileSync(originalProducer, producerReceiptPath);
  const producerHash = fileHash(originalProducer), entry = {version: 1, configFile, configHash, target: "linux",
    source: {path: originalBinary, sha256: sourceHash}, artifact: {path: originalBinary, bytes: raw.length, sha256: sourceHash},
    recipe: {graph: {}}, componentHashes: {[originalProducer]: producerHash},
    producerReceipt: {path: originalProducer, sha256: producerHash}, buildOnly: true, publicationQualified: false};
  const writeEntry = () => fs.writeFileSync(entrypointPath, JSON.stringify(entry, null, 2) + "\n", {flag: "w"});
  writeEntry();
  const calls = /** @type {string[][]} */ ([]);
  const packSpawn = /** @type {typeof import("node:child_process").spawnSync} */ (/** @type {unknown} */ (
    (/** @type {string} */ command, /** @type {string[]} */ args, /** @type {import("node:child_process").SpawnSyncOptions} */ options) => {
      assert.equal(command, upx); calls.push(args.slice());
      const stage = args.includes("--best") ? "pack" : args.includes("-t") ? "integrity" : "restore";
      if (stage === "pack") {
        const destination = args[args.indexOf("-o") + 1], source = args.at(-1);
        if (!destination || !source) {throw new Error("Fixture pack paths are missing");}
        if (process.platform === "linux") {assert.equal(fs.statSync(source).mode & 0o111, 0o111);}
        fs.writeFileSync(destination, Buffer.concat([fs.readFileSync(source).subarray(0, 4), Buffer.from("packed fixture\n")]), {mode: 0o600});
      } else if (stage === "restore") {
        const destination = args[args.indexOf("-o") + 1];
        if (!destination) {throw new Error("Fixture restore path is missing");}
        fs.writeFileSync(destination, raw, {mode: 0o600});
      }
      if (Array.isArray(options.stdio) && typeof options.stdio[2] === "number") {fs.writeSync(options.stdio[2], stage + " fixture\n");}
      return {status: 0};
    }));
  const options = () => /** @type {{entrypointPath:string, entrypointSha256:string, sourcePath:string, sourceSha256:string,
    producerReceiptPath:string, target:"linux"|"win32", output:string, upx:string, upxSha256:string}} */ ({
      entrypointPath, entrypointSha256: fileHash(entrypointPath), sourcePath, sourceSha256: fileHash(sourcePath),
      producerReceiptPath, target: "linux", output, upx, upxSha256: fileHash(upx),
    });
  return {root, origin, transport, output, originalBinary, sourcePath, originalProducer, producerReceiptPath, entrypointPath, upx,
    configFile, configHash, sourceHash, producerHash, entry, calls, packSpawn, writeEntry, options};
}

test.describe("Post-assembly native entrypoint packing", () => {
  test("transported Linux executable permissions are restored without changing pinned bytes", {skip: process.platform !== "linux"}, t => {
    const f = packingFixture(t); fs.chmodSync(f.sourcePath, 0o644);
    const before = fileHash(f.sourcePath);
    packNativeEntrypoint(f.options(), f.packSpawn);
    assert.equal(fs.statSync(f.sourcePath).mode & 0o777, 0o755);
    assert.equal(fileHash(f.sourcePath), before);
  });
  test("rejected transported inputs retain their original permissions", {skip: process.platform !== "linux"}, t => {
    const f = packingFixture(t); fs.chmodSync(f.sourcePath, 0o644);
    fs.appendFileSync(f.producerReceiptPath, "tampered\n");
    assert.throws(() => packNativeEntrypoint(f.options(), f.packSpawn), /producer receipt differs/);
    assert.equal(fs.statSync(f.sourcePath).mode & 0o777, 0o644);
    assert.equal(f.calls.length, 0);
  });
  test("a rejected packer never changes transported executable permissions", {skip: process.platform !== "linux"}, t => {
    const f = packingFixture(t); fs.chmodSync(f.sourcePath, 0o644);
    const options = f.options(); fs.appendFileSync(f.upx, "tampered\n");
    assert.throws(() => packNativeEntrypoint(options, f.packSpawn), /packing tool differs/);
    assert.equal(fs.statSync(f.sourcePath).mode & 0o777, 0o644);
    assert.equal(f.calls.length, 0);
  });
  test("external entrypoint and raw executable pins fail before creating output", t => {
    const f = packingFixture(t), options = f.options(); options.entrypointSha256 = "0".repeat(64);
    assert.throws(() => packNativeEntrypoint(options, f.packSpawn), /entrypoint.*external SHA-256/);
    assert.equal(fs.existsSync(f.output), false); assert.equal(f.calls.length, 0);
    options.entrypointSha256 = fileHash(f.entrypointPath); options.sourceSha256 = "0".repeat(64);
    assert.throws(() => packNativeEntrypoint(options, f.packSpawn), /raw executable.*external SHA-256/);
    assert.equal(fs.existsSync(f.output), false); assert.equal(f.calls.length, 0);
  });
  test("selected target, entrypoint flags and transported producer bytes are re-admitted", t => {
    const f = packingFixture(t), options = f.options(); options.target = "win32";
    assert.throws(() => packNativeEntrypoint(options, f.packSpawn));
    assert.equal(fs.existsSync(f.output), false); assert.equal(f.calls.length, 0);
    options.target = "linux";
    fs.appendFileSync(f.producerReceiptPath, "tampered\n");
    assert.throws(() => packNativeEntrypoint(options, f.packSpawn), /producer receipt differs/);
    assert.equal(fs.existsSync(f.output), false); assert.equal(f.calls.length, 0);
    fs.copyFileSync(f.originalProducer, f.producerReceiptPath);
    f.entry.publicationQualified = true; f.writeEntry(); options.entrypointSha256 = fileHash(f.entrypointPath);
    assert.throws(() => packNativeEntrypoint(options, f.packSpawn), /build-only candidate/);
    assert.equal(fs.existsSync(f.output), false); assert.equal(f.calls.length, 0);
  });
  test("relocated source packs once, chains the raw handoff and remains packageable", {skip: process.platform !== "linux"}, t => {
    const f = packingFixture(t), beforeEntry = fileHash(f.entrypointPath), beforeSource = fileHash(f.sourcePath), options = f.options(),
      packed = packNativeEntrypoint(options, f.packSpawn), entry = JSON.parse(fs.readFileSync(packed.entrypoint, "utf8")),
      packReceiptPath = path.join(f.output, "result.json"), packReceipt = JSON.parse(fs.readFileSync(packReceiptPath, "utf8"));
    assert.equal(f.calls.length, 3); assert.equal(fileHash(f.entrypointPath), beforeEntry); assert.equal(fileHash(f.sourcePath), beforeSource);
    assert.deepEqual(entry.sourceEntrypoint, {path: f.entrypointPath, sha256: beforeEntry});
    assert.deepEqual(entry.source, {path: f.sourcePath, bytes: fs.statSync(f.sourcePath).size, sha256: beforeSource});
    assert.deepEqual(entry.producerReceipt, {path: f.producerReceiptPath, sha256: f.producerHash});
    assert.deepEqual(entry.packingReceipt, {path: packReceiptPath, sha256: fileHash(packReceiptPath)});
    assert.equal(entry.target, "linux"); assert.equal(entry.version, 1); assert.equal(entry.buildOnly, true); assert.equal(entry.publicationQualified, false);
    assert.deepEqual(entry.artifact, packReceipt.artifact); assert.equal(entry.artifact.path, path.join(f.output, "mom"));
    assert.equal(packReceipt.source.sha256, beforeSource); assert.equal(packReceipt.tool.sha256, options.upxSha256);

    const cmake = path.join(f.root, "cmake"), noticeDirectory = path.join(f.root, "notices"), notice = path.join(noticeDirectory, "THIRD-PARTY-NOTICES.txt"),
      packageDirectory = path.join(f.root, "package");
    fs.writeFileSync(cmake, "fixture CMake archiver\n"); fs.mkdirSync(noticeDirectory); fs.writeFileSync(notice, "Reviewed notice fixture.\n", {mode: 0o644});
    const packageSpawn = /** @type {typeof import("node:child_process").spawnSync} */ (/** @type {unknown} */ (
      (/** @type {string} */ command, /** @type {string[]} */ args, /** @type {import("node:child_process").SpawnSyncOptions} */ spawnOptions) => {
        assert.equal(command, cmake);
        const operation = args[2], archive = args[3];
        if (operation === "czf" || operation === "cf") {
          if (!archive) {throw new Error("Fixture archive path is missing");}
          fs.writeFileSync(archive, "fixture archive\n");
        }
        if (operation === "tf") {return {status: 0, stdout: "mom\nTHIRD-PARTY-NOTICES.txt\n", stderr: ""};}
        if (operation === "xf") {
          const cwd = typeof spawnOptions.cwd === "string" ? spawnOptions.cwd : "";
          if (!cwd) {throw new Error("Archive working directory is missing");}
          const input = path.join(path.dirname(cwd), "input");
          for (const name of ["mom", "THIRD-PARTY-NOTICES.txt"]) {fs.copyFileSync(path.join(input, name), path.join(cwd, name));}
          fs.chmodSync(path.join(cwd, "mom"), 0o755); fs.chmodSync(path.join(cwd, "THIRD-PARTY-NOTICES.txt"), 0o644);
        }
        return {status: 0, stdout: "", stderr: ""};
      }));
    const result = packageNativeRelease(packed.entrypoint, cmake, "0.9.4", packageSpawn, {output: packageDirectory, notices: notice});
    assert.deepEqual(result.members, ["mom", "THIRD-PARTY-NOTICES.txt"]);
    assert.equal(result.entryHash, packed.entryHash); assert.equal(result.executable.sha256, packed.artifact.sha256);
    assert.equal(path.basename(result.archive.path), "mom-v0.9.4-lin.tgz");
    assert.equal(fs.existsSync(path.join(packageDirectory, "native-package.json")), true);
  });
  test("the existing source-build API and explicit native CLI selection remain available", () => {
    assert.equal(typeof buildNativeArtifact, "function");
    assert.deepEqual(nativeReleaseArguments(["--native", "/sdk.json", "/output"], {}), ["--native", "/sdk.json", "/output"]);
  });
});

test.describe("Native source-build SDK descriptor admission", () => {
  test("complete SDK describes exactly every maintained compiler", () => {assert.equal(nativeSdkConfig(configuration(), repo).keys.length, 5);});
  test("unknown SDK versions fail before any compiler", () => {const c = configuration(); c.version = 2; assert.throws(() => nativeSdkConfig(c, repo), /descriptor/);});
  test("a missing compiler cannot silently narrow the release", () => {const c = configuration(); Reflect.deleteProperty(c.gpu, "acpp-hip"); assert.throws(() => nativeSdkConfig(c, repo), /every maintained/);});
  test("unrecognized extra profiles are rejected", () => {const c = configuration(); Object.assign(c.gpu, {unknown: {args: []}}); assert.throws(() => nativeSdkConfig(c, repo), /every maintained/);});
  test("SDK link descriptors cannot carry prebuilt application objects", () => {const c = configuration(); Object.assign(c.sdk, {objects: []}); assert.throws(() => nativeSdkConfig(c, repo), /descriptor/);});
  test("SDK templates cannot carry old factory receipts", () => {const c = configuration(); Object.assign(c.sdk, {receipts: []}); assert.throws(() => nativeSdkConfig(c, repo), /descriptor/);});
  test("profile SDK root counts follow the actual producer", () => {const c = configuration(); c.gpu.oneapi.args.push("/unexpected"); assert.throws(() => nativeSdkConfig(c, repo), /source roots/);});
  test("relative SDK roots cannot resolve through an incidental cwd", () => {const c = configuration(); c.gpu.dpcpp.args[0] = "relative"; assert.throws(() => nativeSdkConfig(c, repo), /source roots/);});
  test("runner prefixes require an explicit absolute executable", () => {const c = configuration(); Object.assign(c.cpu, {runner: ["relative"]}); assert.throws(() => nativeSdkConfig(c, repo), /runner/);});
  test("SDK environment values cannot inject NUL tokens", () => {const c = configuration(); Object.assign(c.cpu, {env: {SDK: "bad\0value"}}); assert.throws(() => nativeSdkConfig(c, repo), /environment/);});
  test("core cache preparation is an explicit boolean", () => {const c = configuration(); Object.assign(c.cpu, {prepareCache: "false"}); assert.throws(() => nativeSdkConfig(c, repo), /roots/);});
  test("automatic preparation cannot accidentally build a different cache than the one consumed", () => {
    const c = configuration(); c.cpu.prepareCache = true; assert.throws(() => nativeSdkConfig(c, repo), /shared Engineering cache/);
    c.cpu.cache = path.join(repo, "build/cache/oneapi"); assert.ok(nativeSdkConfig(c, repo));
  });
  test("controller SDK roots are absolute too", () => {const c = configuration(); c.controller.openssl = "relative"; assert.throws(() => nativeSdkConfig(c, repo), /roots/);});
  test("a DPC++ component selector is optional but must be an explicit absolute root", () => {
    const c = configuration(); Object.assign(c.sdk, {dpcppComponent: "/sdk/component"}); assert.ok(nativeSdkConfig(c, repo));
    Object.assign(c.sdk, {dpcppComponent: "relative"}); assert.throws(() => nativeSdkConfig(c, repo), /descriptor/);
  });
  test("CUDA image selection requires a source-admitted image bundle", () => {
    const c = configuration(); Object.assign(c.gpu["dpcpp"], {env: {MOM_AOT_CUDA_IMAGES: "/sdk/cuda-images.inc"}});
    assert.throws(() => nativeSdkConfig(c, repo), /admitted sdkSources.cudaImages/);
  });
  test("native releases cannot silently find libuv/zlib through the build host", () => {
    const c = configuration(); Reflect.deleteProperty(c.controller, "io"); assert.throws(() => nativeSdkConfig(c, repo), /roots/);
  });
  test("control and oversized output paths are refused", () => {assert.equal(absolutePath("/tmp/a\n"), false); assert.equal(absolutePath("/" + "x".repeat(4096)), false);});
  test("malformed descriptors never create the output directory", {skip: process.platform !== "linux"}, t => {
    const root = tempDir(t, "mom-native-release-"), file = path.join(root, "sdk.json"), output = path.join(root, "output");
    fs.writeFileSync(file, "{}"); assert.throws(() => buildNativeRelease(file, output), /descriptor/); assert.equal(fs.existsSync(output), false);
  });
  test("existing artifacts are never overwritten", {skip: process.platform !== "linux"}, t => {
    const root = tempDir(t, "mom-native-release-"); assert.throws(() => buildNativeRelease(path.join(root, "missing.json"), root), /new absolute/);
  });
  test("outputs cannot contaminate maintained source inventories", {skip: process.platform !== "linux"}, () => {
    assert.throws(() => buildNativeRelease("/missing/sdk.json", path.join(repo, "native/new-build-output")), /maintained source tree/);
  });
  test("npm native-mode arity fails before SDK or compiler access", () => {
    const r = spawnSync(process.execPath, [path.join(repo, "scripts/build-release.js"), "--native"], {encoding: "utf8", timeout: 10000});
    assert.equal(r.status, 1); assert.match(r.stderr, /Usage: npm run build:release -- --native/); assert.equal(r.stdout, "");
  });
});

test.describe("Parallel native component assembly admission", () => {
  function config() {const c = configuration(); return {...c, components: {core: "/components/cpu-core.a", gpu: Object.fromEntries(Object.keys(c.gpu).map(key => [key, "/components/" + key + "/result.json"]))}};}
  /** @param {import('node:test').TestContext} t */
  function fixture(t) {
    const root = tempDir(t, "mom-native-release-"), core = path.join(root, "cpu-core.a"), object = path.join(root, "gpu.o"), receipt = path.join(root, "result.json");
    fs.writeFileSync(core, "!<arch>\n"); fs.writeFileSync(object, "object");
    const cpu = {gpu: true, portable: true, objectSha256: fileHash(core), sourceState: nativeSourceInputs(repo, "cpu")};
    const gpu = {factory: "mom_native_gpu_oneapi", object, objectSha256: fileHash(object), sourceState: nativeSourceInputs(repo, "gpu")};
    const save = () => {fs.writeFileSync(path.join(root, "inputs.json"), JSON.stringify(cpu)); fs.writeFileSync(receipt, JSON.stringify(gpu));};
    save();
    const components = {core, gpu: {oneapi: receipt}}, sdk = configuration().sdk;
    return {root, core, object, receipt, cpu, gpu, save, invoke: () => nativeReleaseComponents(components, ["oneapi"], sdk, repo)};
  }
  test("a complete explicit component set preserves every maintained profile", () => {assert.equal(nativeSdkConfig(config(), repo).keys.length, 5);});
  test("partial GPU component sets cannot narrow the final release", () => {
    const c = config(); delete c.components.gpu["dpcpp"]; assert.throws(() => nativeSdkConfig(c, repo), /every GPU profile/);
  });
  test("unknown component profiles fail before file access", () => {
    const c = config(); c.components.gpu["other"] = "/extra.json"; assert.throws(() => nativeSdkConfig(c, repo), /every GPU profile/);
  });
  test("relative component paths cannot depend on the assembly working directory", () => {
    const c = config(); c.components.core = "relative.a"; assert.throws(() => nativeSdkConfig(c, repo), /explicit Core/);
  });
  test("extra component controls cannot bypass receipt admission", () => {
    const c = config(); Object.assign(c.components, {trust: true}); assert.throws(() => nativeSdkConfig(c, repo), /explicit Core/);
  });
  test("existing source receipts are consumed without being rewritten", t => {
    const f = fixture(t), before = fileHash(f.receipt), r = f.invoke();
    assert.equal(r.provenance.portableCore, true); assert.equal(r.hashes[f.receipt], before); assert.equal(fileHash(f.receipt), before);
    assert.deepEqual(r.objects, [{object: f.object}]); assert.equal(r.core, f.core);
  });
  test("CUDA component receipts must bind the admitted image include and sidecar", t => {
    const f = fixture(t), bundle = path.join(f.root, "cuda-images.inc"), sidecar = bundle + ".json";
    fs.writeFileSync(bundle, "image include"); fs.writeFileSync(sidecar, "image sidecar");
    const imageInputs = {[bundle]: fileHash(bundle), [sidecar]: fileHash(sidecar)};
    const gpuRecord = /** @type {Record<string, unknown>} */ (f.gpu);
    f.gpu.factory = "mom_native_gpu_dpcpp"; gpuRecord["sdkInputs"] = {...imageInputs}; gpuRecord["cudaImages"] = bundle; f.save();
    const secondObject = path.join(f.root, "cuda-second.o"), secondReceipt = path.join(f.root, "cuda-second.json");
    // AdaptiveCpp admission inspects its embedded HCF even when CUDA images are shared.
    fs.writeFileSync(secondObject, Buffer.concat([Buffer.from(
      "object-id=123\ngenerator=hipSYCL SSCP\nstart=0\nsize=4\n__acpp_hcf_binary_appendix"),
    Buffer.from([0x42, 0x43, 0xc0, 0xde])]));
    fs.writeFileSync(secondReceipt, JSON.stringify({...f.gpu, factory: "mom_native_gpu_acpp_cuda", object: secondObject, objectSha256: fileHash(secondObject)}));
    const components = {core: f.core, gpu: {dpcpp: f.receipt, "acpp-cuda": secondReceipt}}, sdk = configuration().sdk, keys = ["dpcpp", "acpp-cuda"];
    const reader = (/** @type {string} */ file, /** @type {string} */ selectedRepo) => {
      assert.equal(file, bundle); assert.equal(selectedRepo, repo); return {files: imageInputs};
    };
    assert.doesNotThrow(() => nativeReleaseComponents(components, keys, sdk, repo, imageInputs, reader));
    assert.throws(() => nativeReleaseComponents(components, keys, sdk, repo), /matching shared application image bundle/);
    const recordedInputs = /** @type {Record<string, string>} */ (gpuRecord["sdkInputs"]);
    for (const file of [bundle, sidecar]) {
      delete recordedInputs[file]; f.save();
      assert.throws(() => nativeReleaseComponents(components, keys, sdk, repo, imageInputs, reader), /does not bind/);
      recordedInputs[file] = fileHash(file);
    }
  });
  test("host-specific Core objects are refused even when their source hash matches", t => {
    const f = fixture(t); f.cpu.portable = false; f.save(); assert.throws(f.invoke, /portable Core/);
  });
  test("stale source recipes request an actual rebuild instead of a restamp", t => {
    const f = fixture(t); f.gpu.sourceState.graph["scripts/build-native-release.js"] = "0".repeat(64); f.save(); assert.throws(f.invoke, /source inputs are stale/);
  });
  test("modified source-bound objects cannot enter assembly", t => {
    const f = fixture(t); fs.writeFileSync(f.object, "modified"); assert.throws(f.invoke, /object differs/);
  });
  test("a receipt cannot substitute another factory", t => {
    const f = fixture(t); f.gpu.factory = "mom_native_gpu_dpcpp"; f.save(); assert.throws(f.invoke, /assembled GPU factory/);
  });
  test("component continuity covers Core, receipts and object bytes", t => {
    const f = fixture(t), r = f.invoke();
    assert.deepEqual(Object.keys(r.hashes).sort(), [f.core, f.object, f.receipt, path.join(f.root, "inputs.json")].sort());
  });
});

test.describe("Receipt-derived Linux SDK assembly", {skip: process.platform !== "linux"}, () => {
  /** Readers are separately tested against real producer schemas. These small admitted
   * values isolate assembly/order checks without a compiler, SDK, network or GPU.
   * @param {import('node:test').TestContext} t */
  function fixture(t) {
    const root = tempDir(t, "mom-native-release-"), c = configuration();
    const put = (/** @type {string} */ name, data = "!<arch>\n") => {
      const file = path.join(root, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data); return file;
    };
    const hash = (/** @type {string[]} */ files) => Object.fromEntries(files.map(file => [file, fileHash(file)]));
    const acppReceipt = put("acpp.json", "{}"), hermesReceipt = put("hermes.json", "{}"), rocmReceipt = put("rocm.json", "{}"), nvrtcReceipt = put("nvrtc.json", "{}"),
      openmpReceipt = put("openmp-sdk/result.json", "{}"), openmpArchive = put("openmp-sdk/sdk/libomp.a"),
      openmpLicense = put("openmp-sdk/source/openmp/LICENSE.TXT", "LLVM OpenMP license fixture\n"),
      openmpNotice = {path: "source/openmp/LICENSE.TXT", sha256: fileHash(openmpLicense), bytes: fs.statSync(openmpLicense).size},
      openmp = {receipt: openmpReceipt, receiptSha256: fileHash(openmpReceipt), archive: openmpArchive, notice: openmpNotice,
        files: hash([openmpReceipt, openmpArchive, openmpLicense])};
    const original = put("system/libnuma.a"), staged = put("rocm/libnuma.a"), hip = put("rocm/libamdhip64.a"), runtimeLib = put("runtime.a"), nvrtcLib = put("nvrtc.a"), llvm = put("libLLVMFixture.a");
    c.controller.io = path.join(root, "io");
    const ioFiles = [put("io/lib/libz.a"), put("io/lib/libuv.a"), put("io/include/uv.h", "uv"), put("io/include/zlib.h", "zlib"), put("io/include/zconf.h", "zconf")];
    put("io/result.json", JSON.stringify({version: 1, target: "linux", pins: ioPins, buildOnly: true, sysroot,
      artifacts: Object.fromEntries(ioFiles.map(file => [path.relative(c.controller.io, file), fileHash(file)]))}));
    const zstdRoot = path.join(root, "zstd-sdk"), zstdArchive = put("zstd-sdk/lib/libzstd.a"), zstdBuild = put("zstd-sdk/build/static/libzstd.a"),
      zstdReceipt = put("zstd-sdk/result.json", "{}"), zstdOriginal = put("zstd-input/zstd-1.5.7.tar.gz", "original source archive"),
      zstdLicense = put("zstd-sdk/source/LICENSE", "zstd license fixture\n"), zstdCopying = put("zstd-sdk/source/COPYING", "zstd COPYING fixture\n"),
      zstdSource = put("zstd-sdk/source/lib/zstd.h", "zstd source"), zstdTools = ["ar", "cc", "make", "mkdir", "tar"].map(name => put("zstd-tools/" + name, "tool identity: " + name));
    const zstdFiles = hash([zstdArchive, zstdBuild, zstdReceipt, zstdOriginal, zstdLicense, zstdCopying, zstdSource, ...zstdTools]);
    const zstd = {archive: zstdArchive, artifacts: {"lib/libzstd.a": fileHash(zstdArchive)}, files: zstdFiles,
      receipt: zstdReceipt, receiptSha256: fileHash(zstdReceipt), notices: [
        {component: "zstd LICENSE", path: zstdLicense, sha256: fileHash(zstdLicense)},
        {component: "zstd COPYING", path: zstdCopying, sha256: fileHash(zstdCopying)},
      ]};
    const sources = {acpp: acppReceipt, hermes: hermesReceipt, rocm: rocmReceipt, nvrtc: nvrtcReceipt, openmpSdk: path.join(root, "openmp-sdk"), zstd: zstdRoot};
    const config = /** @type {typeof c & {sdkSources: typeof sources | {acpp: string, hermes: string, rocm: string, cudaImages: string, openmpSdk: string, zstd: string}}} */
      ({...c, sdkSources: sources});
    c.controller.openssl = path.join(root, "openssl-sdk");
    const opensslReceipt = put("openssl-sdk/result.json", "{}"), opensslLicense = put("openssl-sdk/source/LICENSE.txt", "OpenSSL license fixture\n"),
      opensslCrypto = put("openssl-sdk/lib/libcrypto.a"), opensslSsl = put("openssl-sdk/lib/libssl.a"),
      opensslHeader = put("openssl-sdk/include/openssl/ssl.h", "OpenSSL header fixture\n"), opensslCli = put("openssl-sdk/bin/openssl", "OpenSSL CLI fixture\n"),
      opensslFiles = hash([opensslReceipt, opensslLicense, opensslCrypto, opensslSsl, opensslHeader, opensslCli]),
      openssl = {include: path.join(c.controller.openssl, "include"), headers: path.join(c.controller.openssl, "include/openssl"),
        executable: opensslCli, receipt: opensslReceipt, receiptSha256: fileHash(opensslReceipt), files: opensslFiles, libraries: [opensslCrypto, opensslSsl], artifacts: Object.fromEntries(
          [opensslCrypto, opensslSsl, opensslHeader, opensslCli].map(file => [path.relative(c.controller.openssl, file), fileHash(file)])),
        notices: [{component: "OpenSSL LICENSE", path: opensslLicense, sha256: fileHash(opensslLicense)}]};
    c.sdk.cxx = put("cxx", "compiler"); c.controller.cc = put("cc", "compiler"); c.controller.nvml = put("nvml.h", "header");
    const runtime = {receipt: acppReceipt, receiptSha256: fileHash(acppReceipt), source: c.gpu["acpp-cuda"].args[0] || "", backend: "cuda-hip",
      revision: "da2463e45aa90aa36306c45abcfc05b87de51bc6", cudaVersion: 12060, build: c.controller.build, cxx: c.sdk.cxx,
      libraries: [runtimeLib, openmpArchive, original, hip, staged], links: ["-lLLVMFixture", "-Wl,--whole-archive", hip, "-Wl,--no-whole-archive", "-lz", "-lzstd", "-ldl"],
      resolvedLinks: {"-lLLVMFixture": llvm}, files: hash([runtimeLib, openmpArchive, original, hip, staged, llvm]),
      pathBindings: /** @type {Record<string, string>} */ ({"/openmp/libomp.a": openmpArchive}), openmp,
      comgr: {archive: "/sdk/comgr.a", header: "/sdk/comgr.h", receipt: "/sdk/comgr.json", llvmVersion: "fixture",
        roots: {llvm: "/sdk/llvm", lld: "/sdk/lld", device: "/sdk/device"}, files: {}}};
    const hermes = {receipt: hermesReceipt, receiptSha256: fileHash(hermesReceipt), source: c.controller.source, build: c.controller.build, files: hash([hermesReceipt])};
    const rocm = {receipt: rocmReceipt, receiptSha256: fileHash(rocmReceipt), libraries: [hip, staged], files: hash([hip, staged]), comgr: runtime.comgr, numa: {original, staged}};
    const nvrtc = {receipt: nvrtcReceipt, receiptSha256: fileHash(nvrtcReceipt), archives: [nvrtcLib], files: hash([nvrtcLib])};
    const readers = /** @type {typeof import('../scripts/native-release-sdk-inputs') & {nativeZstdInputs: typeof import('../scripts/build-native-zstd').nativeZstdInputs,
      nativeOpenSslInputs: typeof import('../scripts/build-native-openssl').nativeOpenSslInputs}} */ (/** @type {unknown} */ ({
        nativeStaticRuntimeInputs: (/** @type {string} */ receipt, /** @type {{openmpSdkRoot: string}} */ options) => {
          assert.equal(receipt, sources.acpp); assert.deepEqual(options, {openmpSdkRoot: sources.openmpSdk}); return runtime;
        }, nativeHermesInputs: () => hermes, nativeRocmHostInputs: () => rocm,
        nativeNvrtcInputs: (/** @type {string} */ receipt, /** @type {{requireVendorProvenance?: boolean}} */ options) => {
          assert.equal(receipt, sources.nvrtc); assert.deepEqual(options, {requireVendorProvenance: true}); return nvrtc;
        },
        nativeZstdInputs: (/** @type {string} */ sdkRoot) => {if (sdkRoot !== zstdRoot) {throw new Error("Unadmitted zstd SDK root");} return zstd;},
        nativeOpenSslInputs: (/** @type {string} */ sdkRoot, /** @type {string} */ platform) => {
          assert.equal(sdkRoot, c.controller.openssl); assert.equal(platform, "linux"); return openssl;
        },
      }));
    /** Configure the alternative embedded-image cohort without admitting NVRTC archives. */
    function imageMode() {
      const bundle = put("cuda/cuda-images.inc", "embedded image include\n"), sidecar = put("cuda/cuda-images.inc.json", "{}\n");
      const imageFiles = hash([bundle, sidecar]);
      config["sdkSources"] = {acpp: sources.acpp, hermes: sources.hermes, rocm: sources.rocm, cudaImages: bundle,
        openmpSdk: sources.openmpSdk, zstd: sources.zstd};
      Object.assign(config.gpu["dpcpp"], {env: {MOM_AOT_CUDA_IMAGES: bundle}});
      Object.assign(config.gpu["acpp-cuda"], {env: {MOM_AOT_CUDA_IMAGES: bundle}});
      const imageReaders = /** @type {ImageSourceReaders} */ (/** @type {unknown} */ ({
        ...readers,
        nativeNvrtcInputs: () => {throw new Error("NVRTC reader must not run for embedded images");},
        cudaImageInputs: (/** @type {string} */ file, /** @type {string | undefined} */ selectedRepo) => {
          assert.equal(file, bundle); assert.equal(selectedRepo, repo); return {files: imageFiles, manifest: {}};
        },
      }));
      return {bundle, sidecar, imageFiles, readers: imageReaders, invoke: () => nativeSdkSourceInputs(config, imageReaders)};
    }
    return {root, config, runtime, openmp, hermes, rocm, nvrtc, sources, zstd, openssl, put, llvm, readers, imageMode,
      invoke: () => nativeSdkSourceInputs(config, readers), captureLinks: () => nativeSdkLinkInputs(config, readers)};
  }
  test("accepts only a complete explicit source cohort distinct from GPU receipts", t => {
    const f = fixture(t); assert.equal(nativeSdkConfig(f.config, repo).sdk.sdkSources, f.sources);
    for (const invalid of [{}, {...f.sources, acpp: "relative"}, {...f.sources, unused: "/other"}]) {
      assert.throws(() => nativeSdkConfig({...f.config, sdkSources: invalid}, repo), /SDK sources/);
    }
    const {openmpSdk, ...withoutOpenmp} = f.sources;
    assert.ok(openmpSdk);
    assert.throws(() => nativeSdkConfig({...f.config, sdkSources: withoutOpenmp}, repo), /SDK sources/);
    assert.throws(() => nativeSdkConfig({...f.config, sdkSources: {...withoutOpenmp, openmpArchive: f.openmp.archive}}, repo), /SDK sources/);
  });
  test("embedded CUDA images replace only NVRTC and preserve every other admitted archive", t => {
    const f = fixture(t), images = f.imageMode(), extra = f.put("application-extra.a");
    f.config.sdk.libraries.push(extra);
    assert.equal(nativeSdkConfig(f.config, repo).sdk.sdkSources, f.config["sdkSources"]);
    const result = images.invoke();
    assert.deepEqual(result.libraries, [f.runtime.libraries[0], f.openmp.archive, ...f.rocm.libraries, extra]);
    assert.ok(!result.libraries.includes(f.nvrtc.archives[0] || ""));
    assert.equal(result.hashes[images.bundle], images.imageFiles[images.bundle]);
    assert.equal(result.hashes[images.sidecar], images.imageFiles[images.sidecar]);
    assert.deepEqual(result.receipts["cudaImages"], {path: images.sidecar, sha256: images.imageFiles[images.sidecar]});
  });
  test("embedded CUDA images reject a leftover static NVRTC archive", t => {
    const f = fixture(t), images = f.imageMode();
    f.config.sdk.libraries.push(f.put("sdk/libnvrtc_static.a"));
    assert.throws(images.invoke, /static NVRTC archives/);
  });
  for (const token of ["-lnvrtc_static", "-l:libnvrtc-builtins_static.a", "-Wl,--whole-archive,/sdk/libnvptxcompiler_static.a,--no-whole-archive"]) {
    test(`image-only SDK links reject hidden NVRTC (${token})`, t => {
      const f = fixture(t), images = f.imageMode(); f.config.sdk.links.push(token);
      assert.throws(images.invoke, /static NVRTC archives/);
    });
    test(`image-only runtime closure rejects hidden NVRTC (${token})`, t => {
      const f = fixture(t), images = f.imageMode(); f.runtime.links.push(token);
      assert.throws(images.invoke, /static NVRTC archives/);
    });
  }
  test("image mode requires the same bundle on both CUDA-bearing profiles", t => {
    const f = fixture(t), images = f.imageMode();
    Object.assign(f.config.gpu["acpp-cuda"], {env: {}});
    assert.throws(() => nativeSdkConfig(f.config, repo), /dpcpp and acpp-cuda/);
    Object.assign(f.config.gpu["acpp-cuda"], {env: {MOM_AOT_CUDA_IMAGES: images.bundle + ".other"}});
    assert.throws(() => nativeSdkConfig(f.config, repo), /dpcpp and acpp-cuda/);
  });
  test("a CUDA image selector cannot mix with source-composed NVRTC", t => {
    const f = fixture(t), bundle = f.put("cuda/mixed.inc", "images");
    Object.assign(f.config.gpu["dpcpp"], {env: {MOM_AOT_CUDA_IMAGES: bundle}});
    Object.assign(f.config.gpu["acpp-cuda"], {env: {MOM_AOT_CUDA_IMAGES: bundle}});
    assert.throws(() => nativeSdkConfig(f.config, repo), /sdkSources.cudaImages/);
    f.config["sdkSources"] = {...f.sources, cudaImages: bundle};
    assert.throws(() => nativeSdkConfig(f.config, repo), /SDK sources/);
  });
  test("CUDA image inputs are rechecked after admission", t => {
    const f = fixture(t), images = f.imageMode(), readers = /** @type {typeof images.readers} */ ({...images.readers});
    readers.cudaImageInputs = (/** @type {string} */ file, /** @type {string | undefined} */ selectedRepo) => {
      const admitted = images.readers.cudaImageInputs(file, selectedRepo);
      fs.appendFileSync(images.bundle, "tampered after admission");
      return admitted;
    };
    assert.throws(() => nativeSdkSourceInputs(f.config, readers), /current include and sidecar/);
  });
  test("invalid image profile selection fails before creating release output", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), images = f.imageMode(), configFile = path.join(f.root, "image-sdk.json"), output = path.join(f.root, "output");
    Object.assign(f.config.gpu["acpp-cuda"], {env: {}});
    fs.writeFileSync(configFile, JSON.stringify(f.config));
    assert.throws(() => buildNativeRelease(configFile, output), /dpcpp and acpp-cuda/);
    assert.equal(fs.existsSync(output), false);
    assert.ok(images.bundle);
  });
  test("derives library order from the producer and removes only its proven NUMA duplicate", t => {
    const f = fixture(t), result = f.invoke();
    assert.deepEqual(result.libraries, [f.runtime.libraries[0], f.openmp.archive, ...f.rocm.libraries, ...f.nvrtc.archives]);
    assert.ok(!result.libraries.includes(f.rocm.numa.original));
  });
  test("projects only admitted OpenMP, zstd and OpenSSL bodies into final link hashes", t => {
    const f = fixture(t), result = f.captureLinks(), license = path.join(f.root, "openmp-sdk/source/openmp/LICENSE.TXT"),
      zstdLicense = path.join(f.root, "zstd-sdk/source/LICENSE"), zstdCopying = path.join(f.root, "zstd-sdk/source/COPYING"),
      opensslLicense = path.join(f.root, "openssl-sdk/source/LICENSE.txt"), ssl = path.join(f.root, "openssl-sdk/lib/libssl.a"),
      crypto = path.join(f.root, "openssl-sdk/lib/libcrypto.a"), opensslReceipt = path.join(f.root, "openssl-sdk/result.json");
    assert.deepEqual(result.noticeSources, [
      {component: "LLVM OpenMP runtime", path: license, sha256: fileHash(license)},
      {component: "zstd LICENSE", path: zstdLicense, sha256: fileHash(zstdLicense)},
      {component: "zstd COPYING", path: zstdCopying, sha256: fileHash(zstdCopying)},
      {component: "OpenSSL LICENSE", path: opensslLicense, sha256: fileHash(opensslLicense)},
    ]);
    for (const row of result.noticeSources || []) {assert.equal(result.hashes[row.path], row.sha256);}
    assert.equal(result.hashes[ssl], fileHash(ssl)); assert.equal(result.hashes[crypto], fileHash(crypto));
    assert.equal(result.sources?.["openssl"]?.path, opensslReceipt); assert.equal(result.hashes[opensslReceipt], fileHash(opensslReceipt));
    assert.equal(result.hashes[path.join(f.root, "openssl-sdk/include/openssl/ssl.h")], fileHash(path.join(f.root, "openssl-sdk/include/openssl/ssl.h")));
  });
  test("missing or wrong selected notice rows fail closed", t => {
    const f = fixture(t);
    const missingZstd = {...f.readers, nativeZstdInputs: () => ({...f.zstd, notices: []})};
    assert.throws(() => nativeSdkSourceInputs(f.config, missingZstd), /zstd notice rows/);
    const wrongZstd = {...f.readers, nativeZstdInputs: () => ({...f.zstd, notices: f.zstd.notices.map((row, index) =>
      index === 0 ? {...row, sha256: "0".repeat(64)} : row)})};
    assert.throws(() => nativeSdkSourceInputs(f.config, wrongZstd), /zstd notice rows/);
    const missingOpenSsl = {...f.readers, nativeOpenSslInputs: () => ({...f.openssl, notices: []})};
    assert.throws(() => nativeSdkSourceInputs(f.config, missingOpenSsl), /OpenSSL notice rows/);
    const wrongOpenSsl = {...f.readers, nativeOpenSslInputs: () => ({...f.openssl, notices: f.openssl.notices.map(row =>
      ({...row, path: path.join(f.root, "unselected-license.txt")}))})};
    assert.throws(() => nativeSdkSourceInputs(f.config, wrongOpenSsl), /OpenSSL notice rows/);
    const wrongOpenSslHash = {...f.readers, nativeOpenSslInputs: () => ({...f.openssl, notices: f.openssl.notices.map(row =>
      ({...row, sha256: "0".repeat(64)}))})};
    assert.throws(() => nativeSdkSourceInputs(f.config, wrongOpenSslHash), /OpenSSL notice rows/);
    const otherArchive = f.put("openssl-sdk/lib/libother.a"), wrongOpenSslArchives = {...f.readers,
      nativeOpenSslInputs: () => ({...f.openssl, libraries: [path.join(f.config.controller.openssl, "lib/libcrypto.a"), otherArchive]})};
    assert.throws(() => nativeSdkSourceInputs(f.config, wrongOpenSslArchives), /controller link inputs/);
    const missingOpenSslArchive = {...f.readers, nativeOpenSslInputs: () => ({...f.openssl, files: Object.fromEntries(
      Object.entries(f.openssl.files).filter(([file]) => file !== f.openssl.libraries[0]))})};
    assert.throws(() => nativeSdkSourceInputs(f.config, missingOpenSslArchive), /omit a controller-selected static archive/);
    const missingOpenmp = {...f.readers, nativeStaticRuntimeInputs: () => ({...f.runtime,
      openmp: {...f.openmp, notice: undefined}})};
    assert.throws(() => nativeSdkSourceInputs(f.config, missingOpenmp), /OpenMP notice/);
  });
  test("source bodies and OpenSSL receipt cannot change after their readers return", t => {
    const f = fixture(t), license = path.join(f.root, "openssl-sdk/source/LICENSE.txt"), readers = {...f.readers};
    readers.nativeOpenSslInputs = () => {fs.appendFileSync(license, "changed\n"); return f.openssl;};
    assert.throws(() => nativeSdkLinkInputs(f.config, readers), /absent from admitted link hashes/);
    const g = fixture(t), receipt = path.join(g.root, "openssl-sdk/result.json"), receiptReaders = {...g.readers};
    receiptReaders.nativeOpenSslInputs = () => {fs.appendFileSync(receipt, "changed\n"); return g.openssl;};
    assert.throws(() => nativeSdkLinkInputs(g.config, receiptReaders), /changed during assembly/);
  });
  test("preserves static groups and substitutes exact bound LLVM and compression archives", t => {
    const f = fixture(t), result = f.invoke();
    assert.deepEqual(result.links, [f.llvm, "-Wl,--whole-archive", f.rocm.libraries[0], "-Wl,--no-whole-archive",
      path.join(f.config.controller.io, "lib/libz.a"), f.zstd.archive, "-ldl"]);
  });
  test("raw zstd archives are not accepted as source SDK descriptors", t => {
    const f = fixture(t); f.config.sdkSources.zstd = f.zstd.archive;
    assert.throws(f.invoke, /producer-admitted SDK root, not a raw archive/);
  });
  test("source-reader injection cannot omit zstd or OpenSSL receipt admission", t => {
    const f = fixture(t), readers = {...f.readers}; Reflect.deleteProperty(readers, "nativeZstdInputs");
    assert.throws(() => nativeSdkSourceInputs(f.config, readers), /require zstd and OpenSSL receipt admission/);
    const noOpenSsl = {...f.readers}; Reflect.deleteProperty(noOpenSsl, "nativeOpenSslInputs");
    assert.throws(() => nativeSdkSourceInputs(f.config, noOpenSsl), /require zstd and OpenSSL receipt admission/);
  });
  test("final source closure requires the admitted OpenMP path to be the actual ACPP library", t => {
    const f = fixture(t), ambient = f.put("ambient/libomp.a");
    f.runtime.pathBindings["/openmp/libomp.a"] = ambient;
    assert.throws(f.invoke, /does not resolve ACPP's OpenMP input to the admitted SDK archive/);
    f.runtime.pathBindings["/openmp/libomp.a"] = f.openmp.archive;
    f.runtime.libraries = f.runtime.libraries.filter(file => file !== f.openmp.archive);
    assert.throws(f.invoke, /does not resolve ACPP's OpenMP input to the admitted SDK archive/);
  });
  test("source composer accepts an admitted canonical OpenMP archive without a container alias", t => {
    const f = fixture(t);
    f.runtime.pathBindings = {};
    f.runtime.links.push(f.openmp.archive);
    const result = f.invoke();
    assert.ok(result.libraries.includes(f.openmp.archive));
    assert.ok(result.links.includes(f.openmp.archive));
    assert.equal(result.hashes[f.openmp.archive], f.openmp.files[f.openmp.archive]);
    assert.deepEqual(result.receipts["openmp"], {path: f.openmp.receipt, sha256: f.openmp.receiptSha256});
  });
  test("keeps application linker controls before the closure and other additions after it", t => {
    const f = fixture(t); f.config.sdk.links = ["-Wl,--threads=2", "-Wl,-T,/app/constructors.ld"];
    assert.equal(f.invoke().links[0], "-Wl,--threads=2");
    assert.equal(f.invoke().links.at(-1), "-Wl,-T,/app/constructors.ld");
  });
  test("retains all source receipt and dependency hashes for final continuity checks", t => {
    const f = fixture(t), result = f.invoke();
    for (const file of [f.sources.acpp, f.sources.hermes, f.sources.rocm, f.sources.nvrtc, ...f.runtime.libraries,
      f.openmp.receipt, f.openmp.archive]) {
      assert.equal(result.hashes[file], fileHash(file));
    }
    for (const [file, hash] of Object.entries(f.zstd.files)) {assert.equal(result.hashes[file], hash);}
    for (const name of ["result.json", "include/uv.h", "include/zlib.h", "include/zconf.h"]) {
      const file = path.join(f.config.controller.io, name); assert.equal(result.hashes[file], fileHash(file));
    }
    assert.deepEqual(Object.keys(result.receipts).sort(), ["acpp", "hermes", "nvrtc", "openmp", "openssl", "rocm", "zstd"]);
    assert.deepEqual(result.receipts["openmp"], {path: f.openmp.receipt, sha256: f.openmp.receiptSha256});
    const opensslReceipt = path.join(f.config.controller.openssl, "result.json");
    assert.deepEqual(result.receipts["openssl"], {path: opensslReceipt, sha256: fileHash(opensslReceipt)});
  });
  test("does not refresh admitted I/O hashes after a file swap during link planning", t => {
    const f = fixture(t), zlib = path.join(f.config.controller.io, "lib/libz.a"), receipt = path.join(f.config.controller.io, "result.json"),
      expected = {[zlib]: fileHash(zlib), [receipt]: fileHash(receipt)}, open = fs.openSync;
    let changed = false;
    /** @type {typeof fs.openSync} */
    const observeOpen = (file, flags, mode) => {
      const fd = open(file, flags, mode);
      if (file === f.zstd.archive && !changed) {
        changed = true; fs.appendFileSync(zlib, "changed"); fs.appendFileSync(receipt, "\n");
      }
      return fd;
    };
    t.mock.method(fs, "openSync", observeOpen);
    const result = f.invoke(); assert.equal(changed, true);
    for (const [file, hash] of Object.entries(expected)) {assert.equal(result.hashes[file], hash); assert.notEqual(hash, fileHash(file));}
  });
  test("rejects manual archive duplication instead of silently deduplicating by basename", t => {
    const f = fixture(t); f.config.sdk.libraries.push(f.rocm.libraries[0] || ""); assert.throws(f.invoke, /duplicated/);
  });
  test("does not delete unrelated same-basename application archives", t => {
    const f = fixture(t), other = f.put("other/libnuma.a"); f.config.sdk.libraries.push(other); assert.ok(f.invoke().libraries.includes(other));
  });
  test("rejects conflicting SDK dependency hashes", t => {
    const f = fixture(t); f.rocm.files[f.rocm.libraries[0] || ""] = "0".repeat(64); assert.throws(f.invoke, /Conflicting/);
  });
  test("rejects a Hermes checkout not used by the controller", t => {
    const f = fixture(t); f.hermes.source = "/other/hermes"; assert.throws(f.invoke, /selected controller/);
  });
  test("requires the combined CUDA and HIP runtime", t => {
    const f = fixture(t); f.runtime.backend = "cuda"; assert.throws(f.invoke, /CUDA\/HIP runtime cohort/);
  });
  test("requires the same source and COMGR selection", t => {
    const f = fixture(t); f.runtime.comgr = {...f.runtime.comgr, receipt: "/other/comgr.json"}; assert.throws(f.invoke, /runtime cohort/);
  });
  test("rejects a missing staged ROCm library", t => {
    const f = fixture(t); f.runtime.libraries = f.runtime.libraries.filter(file => file !== f.rocm.numa.staged); assert.throws(f.invoke, /runtime cohort/);
  });
  test("rejects a shared or thin compression archive", t => {
    const f = fixture(t); fs.writeFileSync(f.zstd.archive, "!<thin>\n");
    assert.throws(f.invoke, /admitted static zstd SDK archive/);
  });
  test("low-level explicit SDK probes stay unchanged without source selectors", () => {
    const c = configuration(), result = nativeSdkSourceInputs(c); assert.equal(result.libraries, c.sdk.libraries); assert.equal(result.links, c.sdk.links);
  });
  test("CLI exports path helpers before lazy SDK readers load original producer modules", t => {
    const f = fixture(t), config = path.join(f.root, "config.json"), preload = path.join(f.root, "preload.cjs");
    fs.writeFileSync(config, JSON.stringify(f.config));
    fs.writeFileSync(preload, `const Module=require('module'),original=Module._load;
Module._load=function(name,parent,main){
if(name==='./build-native-io' && parent?.filename.endsWith('build-native-release.js'))return {nativeIoInputs(){return {}}};
if(name==='./native-release-sdk-inputs')return {nativeStaticRuntimeInputs(){
 const r=require(${JSON.stringify(path.join(repo, "scripts/build-native-release.js"))});
 if(typeof r.absolutePath!=='function')throw Error('CLI_HELPERS_NOT_EXPORTED');
 require(${JSON.stringify(path.join(repo, "scripts/build-native-sycl-sdk.js"))});
 throw Error('CLI_SDK_READER_REACHED');}};
return original.apply(this,arguments);};\n`);
    const result = spawnSync(process.execPath, ["--require", preload, path.join(repo, "scripts/build-native-release.js"), config,
      path.join(f.root, "output")], {encoding: "utf8", timeout: 10000});
    assert.equal(result.status, 1); assert.match(result.stderr, /CLI_SDK_READER_REACHED/);
    assert.doesNotMatch(result.stderr, /CLI_HELPERS_NOT_EXPORTED|circular dependency/);
    assert.equal(fs.existsSync(path.join(f.root, "output")), false);
  });
});

test.describe("Native SDK link files are explicit and content-bound", () => {
  /** @param {import("node:test").TestContext} t */
  function fixture(t) {
    const root = tempDir(t, "mom-native-release-"), c = configuration();
    const put = (/** @type {string} */ name, /** @type {string} */ value) => {const file = path.join(root, name); fs.writeFileSync(file, value); return file;};
    c.sdk.cxx = put("cxx", "compiler"); c.controller.cc = put("cc", "compiler"); c.controller.nvml = put("nvml.h", "header");
    c.sdk.links = ["-L", root, "-lLLVMFixture"]; put("libLLVMFixture.a", "!<arch>\n");
    return {root, c, put, capture: () => nativeSdkLinkInputs(c)};
  }
  test("SDK short names resolve to exact archive paths", t => {const f = fixture(t), r = f.capture(); assert.equal(r.links[2], path.join(f.root, "libLLVMFixture.a"));});
  test("explicit SDK archives participate in the hash receipt", t => {const f = fixture(t), file = f.put("explicit.a", "!<arch>\n"); f.c.sdk.libraries.push(file); assert.ok(f.capture().hashes[file]);});
  test("linker scripts participate in the hash receipt", t => {const f = fixture(t), file = f.put("script.ld", "SECTIONS {}"); f.c.sdk.links.push("-Wl,-T," + file); assert.ok(f.capture().hashes[file]);});
  test("SDK object changes alter the content receipt", t => {const f = fixture(t), file = f.put("compiler.o", "before"); f.c.sdk.sdkObjects.push({object: file}); const before = f.capture(); f.put("compiler.o", "after"); assert.notEqual(f.capture().hashes[file], before.hashes[file]);});
  test("repeated SDK objects fail before costly compilation instead of a final duplicate-symbol link", t => {
    const f = fixture(t), file = f.put("compiler.o", "fixture"); f.c.sdk.sdkObjects.push({object: file}, {object: file}); assert.throws(f.capture, /duplicated/);
  });
  test("SDK activation wrapper bytes are pinned too", t => {const f = fixture(t), file = f.put("runner", "activate SDK"); Object.assign(f.c.cpu, {runner: [file]}); assert.ok(f.capture().hashes[file]);});
  test("missing named archives fail before compilation", t => {const f = fixture(t); f.c.sdk.links.push("-lMissing"); assert.throws(f.capture, /Missing static/);});
  test("relative SDK search roots cannot depend on cwd", t => {const f = fixture(t); f.c.sdk.links = ["-Lrelative"]; assert.throws(f.capture, /search root/);});
  test("a missing -L value is rejected", t => {const f = fixture(t); f.c.sdk.links = ["-L"]; assert.throws(f.capture, /search root/);});
  test("SDK library names cannot traverse the search roots", t => {const f = fixture(t); f.c.sdk.links.push("-l../outside"); assert.throws(f.capture, /library name/);});
  test("shared or thin libraries cannot replace static SDK archives", t => {const f = fixture(t); f.put("libLLVMFixture.a", "!<thin>\n"); assert.throws(f.capture, /static archive/);});
  test("OS and compiler support lookup remains owned by the baseline toolchain", t => {const f = fixture(t); f.c.sdk.links.push("-ldl", "-lm", "-lstdc++fs"); assert.deepEqual(f.capture().links.slice(-3), ["-ldl", "-lm", "-lstdc++fs"]);});
  test("directories are not recorded as opaque file hashes", t => {const f = fixture(t), r = f.capture(); assert.equal(r.hashes[f.root], undefined);});
  test("assembly preserves the C++ driver name through the actual manifest and link", {skip: process.platform !== "linux"}, t => {
    const found = spawnSync("sh", ["-c", "command -v clang++ || command -v clang++-21 || command -v clang++-20"], {encoding: "utf8", timeout: 5000});
    if (found.status !== 0) {t.skip("Clang C++ driver unavailable"); return;}
    const output = tempDir(t, "mom-native-release-"), c = configuration(); c.sdk.cxx = found.stdout.trim();
    const source = fs.readFileSync(path.join(repo, "scripts/build-native-release.js"), "utf8");
    const alias = source.indexOf("  for (const [name, target]"), manifest = source.indexOf('  const manifest = path.join(output, "gpu-inputs.json");');
    const end = source.indexOf("\n  node({...sdk.controller", manifest);
    assert.ok(manifest >= 0 && end > manifest);
    const imageFiles = {[path.join(output, "cuda-images.inc")]: "a".repeat(64), [path.join(output, "cuda-images.inc.json")]: "b".repeat(64)};
    const file = runInNewContext(source.slice(alias >= 0 ? alias : manifest, end) + "\nmanifest;",
      {fs, path, output, sdk: c, sdkInputs: {cudaImageInputs: imageFiles}, objects: [], receipts: []});
    const selected = JSON.parse(fs.readFileSync(file, "utf8"));
    assert.deepEqual(selected.cudaImageInputs, imageFiles, "The application must receive the admitted image hashes, not merely its compiler objects");
    const input = path.join(output, "source.cpp");
    fs.writeFileSync(input, "#include <string>\nint main(int argc, char **argv) { return std::string(argv[0]).size() < static_cast<unsigned>(argc); }\n");
    const linked = spawnSync(selected.cxx, [input, "-o", path.join(output, "linked")], {encoding: "utf8", timeout: 30000});
    assert.equal(linked.status, 0, linked.stderr);
    assert.equal(selected.cxx, c.sdk.cxx, "Compiler argv[0] must preserve the selected SDK driver, not a renamed alias");
  });
});

test.describe("Native release and SDK stages share the portable release policy", () => {
  for (const [name, value] of [["MOM_CPU_MARCH", "native"], ["MOM_DPCPP_IMPL", "custom"],
    ["MOM_COMBINED_TARGETS", "local"], ["MOM_LINUX_BUILD_PLAN_ONLY", "1"], ["MOM_FORCE_REBUILD", "yes"]]) {
    test("inherited " + name + " fails before descriptor access or output mutation", {skip: process.platform !== "linux"}, t => {
      const root = tempDir(t, "mom-native-release-"), output = path.join(root, "output");
      assert.throws(() => buildNativeRelease(path.join(root, "missing.json"), output, {[name || ""]: value}), new RegExp(name || ""));
      assert.equal(fs.existsSync(output), false);
    });
    test("SDK " + name + " cannot bypass release admission", () => {
      const c = configuration(); Object.assign(c.controller, {env: {[name || ""]: value}});
      assert.throws(() => nativeSdkConfig(c, repo), new RegExp(name || ""));
    });
  }
  test("archive and native builders use the very same policy function", () => {
    assert.equal(require("../scripts/build-environment").releaseBuildEnvironment, require("../scripts/build-release").releaseBuildEnvironment);
  });
});

test.describe("Native build stages retain literal commands and bounded logging", () => {
  /** @param {import("node:test").TestContext} t @param {{runner?: string[], env?: NodeJS.ProcessEnv}} runner @param {{status: number | null, signal?: string, error?: Error}} result */
  function stage(t, runner, result = {status: 0}, logOutput = "") {
    const output = tempDir(t, "mom-native-release-"); fs.mkdirSync(path.join(output, "logs"));
    const calls = /** @type {{command: string, args: string[], options: import("node:child_process").SpawnSyncOptions}[]} */ ([]);
    const fake = /** @type {typeof spawnSync} */ ((command, args, options) => {
      calls.push({command, args: [...(args || [])], options: options || {}});
      const fd = options?.stdio?.[1]; if (typeof fd === "number") {fs.writeSync(fd, logOutput);}
      return result;
    });
    const invoke = () => {
      const log = console.log; console.log = () => undefined;
      try {nativeStage(runner, "/sdk/node", ["literal;$(false)", "space name"], {MOM_AOT_BUILD_ONLY: "0"}, repo, output, "fixture", fake);} finally {console.log = log;}
    };
    return {calls, output, invoke};
  }
  test("SDK runners receive tokens without a shell", t => {const f = stage(t, {runner: ["/sdk/runner", "arg"]}); f.invoke(); assert.equal(f.calls[0]?.command, "/sdk/runner"); assert.deepEqual(f.calls[0]?.args, ["arg", "/sdk/node", "literal;$(false)", "space name"]); assert.equal(f.calls[0]?.options.shell, undefined);});
  test("the normal path invokes the maintained command directly", t => {const f = stage(t, {}); f.invoke(); assert.equal(f.calls[0]?.command, "/sdk/node");});
  test("build-only mode cannot be disabled by inherited or profile environment", t => {const f = stage(t, {env: {MOM_AOT_BUILD_ONLY: "0"}}); f.invoke(); assert.equal(f.calls[0]?.options.env?.["MOM_AOT_BUILD_ONLY"], "1");});
  test("SDK environments cannot disable the portable build profile", t => {const f = stage(t, {env: {MOM_PORTABLE_BUILD: "0", MOM_REUSE_BUILT_WORKER: "1"}}); f.invoke(); assert.equal(f.calls[0]?.options.env?.["MOM_PORTABLE_BUILD"], "1"); assert.equal(f.calls[0]?.options.env?.["MOM_REUSE_BUILT_WORKER"], "0");});
  test("SDK host CPU overrides fail before spawn", t => {const f = stage(t, {env: {MOM_CPU_MARCH: "native"}}); assert.throws(f.invoke, /MOM_CPU_MARCH/); assert.equal(f.calls.length, 0);});
  test("the shared Engineering cache still prepares only the intended oneAPI stage", t => {const f = stage(t, {env: {MOM_GPU_BACKEND: "intel", MOM_LINUX_BUILD_COMPILER: "oneapi"}}); f.invoke(); assert.equal(f.calls[0]?.options.env?.["MOM_GPU_BACKEND"], "intel"); assert.equal(f.calls[0]?.options.env?.["MOM_LINUX_BUILD_COMPILER"], "oneapi");});
  test("runner environment retains SDK activation data", t => {const f = stage(t, {env: {SDK: "/sdk"}}); f.invoke(); assert.equal(f.calls[0]?.options.env?.["SDK"], "/sdk");});
  test("output and stage identity are explicit to containment wrappers", t => {const f = stage(t, {}); f.invoke(); assert.equal(f.calls[0]?.options.env?.["MOM_NATIVE_BUILD_OUTPUT"], f.output); assert.equal(f.calls[0]?.options.env?.["MOM_NATIVE_BUILD_STAGE"], "fixture");});
  test("compiler output uses a file descriptor, not a parent stdout buffer", t => {const f = stage(t, {}); f.invoke(); const stdio = /** @type {unknown[]} */ (f.calls[0]?.options.stdio || []); assert.equal(typeof stdio[1], "number"); assert.equal(f.calls[0]?.options.encoding, undefined);});
  test("a nonzero compiler result stops the lifecycle", t => {const f = stage(t, {}, {status: 1}); assert.throws(f.invoke, /inspect.*before retrying/); assert.equal(f.calls.length, 1);});
  test("a failed compiler stage emits only a bounded tail while retaining its log and failure", t => {
    const output = "early compiler context\n" + "x".repeat(20000) + "\nlate compiler reason\n", f = stage(t, {}, {status: 2}, output),
      errors = /** @type {string[]} */ ([]), oldError = console.error;
    console.error = message => {errors.push(String(message));};
    try {assert.throws(f.invoke, /Native build failed: fixture; inspect .*fixture\.log before retrying/);} finally {console.error = oldError;}
    const diagnostic = errors.join("\n");
    assert.match(diagnostic, /exit 2, signal none/); assert.match(diagnostic, /late compiler reason/); assert.doesNotMatch(diagnostic, /early compiler context/);
    assert.ok(Buffer.byteLength(diagnostic) <= 17000); assert.equal(fs.readFileSync(path.join(f.output, "logs/fixture.log"), "utf8"), output);
    assert.equal(f.calls.length, 1);
  });
  test("a compiler signal is unresolved even with status zero", t => {
    const f = stage(t, {}, {status: 0, signal: "SIGTERM"}), errors = /** @type {string[]} */ ([]), oldError = console.error;
    console.error = message => {errors.push(String(message));};
    try {assert.throws(f.invoke, /Native build failed/);} finally {console.error = oldError;}
    assert.match(errors.join("\n"), /exit 0, signal SIGTERM/);
  });
  test("a missing compiler returns a clear stage failure", t => {
    const f = stage(t, {}, {status: null, error: new Error("ENOENT")}), errors = /** @type {string[]} */ ([]), oldError = console.error;
    console.error = message => {errors.push(String(message));};
    try {assert.throws(f.invoke, /Native build failed/);} finally {console.error = oldError;}
    assert.match(errors.join("\n"), /exit null, signal none, error ENOENT/);
  });
  test("a stage cannot overwrite earlier diagnostic evidence", t => {const f = stage(t, {}); f.invoke(); assert.throws(f.invoke, /EEXIST/); assert.equal(f.calls.length, 1);});
  test("a failed controller command preserves its status and bounded compiler output", t => {
    const root = tempDir(t, "mom-native-release-"), log = path.join(root, "controller.log"), output = "early controller context\n" + "x".repeat(40000) + "\nactual compiler error\n";
    const cause = new Error("fixture spawn error"), result = {status: 2, stdout: output.slice(0, output.length - 24), stderr: "actual compiler error\n", error: cause};
    let calls = 0;
    const fake = /** @type {typeof spawnSync} */ (/** @type {unknown} */ (() => {calls++; return result;}));
    let failure;
    try {runNativeControllerCommand("fixture-compiler", ["-c"], log, 0, {}, root, fake);} catch (error) {failure = error;}
    assert.ok(failure instanceof Error); assert.match(failure.message, /exit 2/); assert.match(failure.message, /actual compiler error/);
    assert.doesNotMatch(failure.message, /early controller context/); assert.ok(Buffer.byteLength(failure.message) < 18000);
    assert.equal(fs.readFileSync(log, "utf8"), result.stdout + result.stderr); assert.equal(failure.cause, cause); assert.equal(calls, 1);
  });
  test("a failed controller object compile keeps its full log and emits a bounded tail once", t => {
    const root = tempDir(t, "mom-native-release-"), output = path.join(root, "application"), object = path.join(output, "gpu-dispatch.o"),
      depfile = object + ".d", log = path.join(output, "gpu-dispatch.log"), fullOutput = "early object compiler context\n" +
        "x".repeat(40000) + "\nactual object compiler error\n", cause = "fixture compiler launch failure";
    fs.mkdirSync(output);
    const command = {command: "/sdk/cxx", args: ["-c", path.join(repo, "runtime/gpu-dispatch.cpp"), "-o", object],
      source: path.join(repo, "runtime/gpu-dispatch.cpp"), object, depfile, timeout: 120000, generatedInputs: [], log};
    const result = {objects: [{object, depfile, cacheHit: false, status: 2, stdout: fullOutput, stderr: "", error: cause,
      signal: null, objectSha256: null, depfileSha256: null}], cacheRoot: path.join(root, "cache"), cacheHits: 0, cacheMisses: 1, ninja: "/usr/bin/ninja"};
    const calls = /** @type {Array<Parameters<typeof import("../scripts/native-object-cache").compileNativeObjects>[0]>} */ ([]),
      fake = (/** @type {Parameters<typeof import("../scripts/native-object-cache").compileNativeObjects>[0]} */ options) => {calls.push(options); return result;},
      batch = {repo, output, namespace: "linux/controller/test", commands: [command], includeRoots: new Set(["/sdk/include"])};
    let failure;
    try {flushLinuxObjectBatch(batch, fake);} catch (error) {failure = error;}
    assert.ok(failure instanceof Error); assert.match(failure.message, /Native Linux object compile failed \(exit 2, signal none\)/);
    assert.match(failure.message, /actual object compiler error/); assert.doesNotMatch(failure.message, /early object compiler context/);
    assert.ok(Buffer.byteLength(failure.message) < 18000); assert.equal(failure.cause, cause);
    assert.equal(fs.readFileSync(log, "utf8"), fullOutput);
    assert.equal(calls.length, 1); assert.deepEqual(calls[0]?.commands, [command]); assert.equal(calls[0]?.jobs, 1);
    assert.equal(calls[0]?.finalize, true); assert.deepEqual(calls[0]?.liveObjects, [object]);
    assert.deepEqual(calls[0]?.includeRoots, ["/sdk/include"]);
  });
  test("a failed Ninja batch retains every object log and exposes a later edge error", t => {
    const root = tempDir(t, "mom-native-release-"), output = path.join(root, "application"), firstObject = path.join(output, "gpu-dispatch.o"),
      secondObject = path.join(output, "driver-library.o"), firstLog = path.join(output, "gpu-dispatch.log"),
      secondLog = path.join(output, "driver-library.log"), firstDepfile = firstObject + ".d", secondDepfile = secondObject + ".d",
      secondOutput = "early edge context\n" + "x".repeat(40000) + "\nactual later compiler error\n", cause = "Ninja failed with status 1";
    fs.mkdirSync(output);
    const commands = [
      {command: "/sdk/cxx", args: [], source: path.join(repo, "runtime/gpu-dispatch.cpp"), object: firstObject, depfile: firstDepfile, timeout: 120000, generatedInputs: [], log: firstLog},
      {command: "/sdk/cxx", args: [], source: path.join(repo, "runtime/driver-library.cpp"), object: secondObject, depfile: secondDepfile, timeout: 120000, generatedInputs: [], log: secondLog},
    ];
    const result = {objects: [
      {object: firstObject, depfile: firstDepfile, cacheHit: false, status: null, stdout: "", stderr: "", error: cause,
        signal: null, objectSha256: null, depfileSha256: null},
      {object: secondObject, depfile: secondDepfile, cacheHit: false, status: null, stdout: secondOutput, stderr: "", error: cause,
        signal: null, objectSha256: null, depfileSha256: null},
    ], cacheRoot: path.join(root, "cache"), cacheHits: 0, cacheMisses: 2, ninja: "/usr/bin/ninja"};
    const calls = /** @type {Array<Parameters<typeof import("../scripts/native-object-cache").compileNativeObjects>[0]>} */ ([]),
      fake = (/** @type {Parameters<typeof import("../scripts/native-object-cache").compileNativeObjects>[0]} */ options) => {calls.push(options); return result;},
      batch = {repo, output, namespace: "linux/controller/test", commands, includeRoots: new Set(["/sdk/include"])};
    let failure;
    try {flushLinuxObjectBatch(batch, fake);} catch (error) {failure = error;}
    assert.ok(failure instanceof Error); assert.match(failure.message, /actual later compiler error/);
    assert.ok(Buffer.byteLength(failure.message) < 18000); assert.equal(failure.cause, cause);
    assert.equal(fs.readFileSync(firstLog, "utf8"), ""); assert.equal(fs.readFileSync(secondLog, "utf8"), secondOutput);
    assert.equal(calls.length, 1); assert.deepEqual(calls[0]?.commands, commands); assert.equal(calls[0]?.jobs, 1);
  });
});

test.describe("Native ELF strip payload boundaries", () => {
  const section = "  [ 1] .text PROGBITS 0000000000001000 001000 000008 00 AX 0 0 16\n";
  const parsed = () => elfSections(section, () => "hash");
  test("wide section metadata preserves address, flags and payload digest", () => {const r = parsed()[0]; assert.equal(r?.section, ".text"); assert.equal(r?.sha256, "hash"); assert.equal(r?.flags, "AX");});
  test("malformed section syntax fails closed", () => {assert.throws(() => elfSections("[1] .text unexpected", () => ""), /metadata/);});
  test("missing section metadata is not an empty success", () => {assert.throws(() => elfSections("nothing", () => ""), /Missing/);});
  test("excessive metadata fails before payload hashing", () => {assert.throws(() => elfSections("x".repeat((1 << 20) + 1), () => {throw new Error("unexpected");}), /Excessive/);});
  test("duplicate section names cannot hide a replaced image", () => {assert.throws(() => elfSections(section + section, () => ""), /Duplicate/);});
  test("oversized payload ranges cannot lose integer precision", () => {assert.throws(() => elfSections(section.replace("000008", "ffffffffffffffff"), () => ""), /Excessive/);});
  test("NOBITS metadata never reads nonexistent file payload", () => {const r = elfSections(section.replace("PROGBITS", "NOBITS"), () => {throw new Error("unexpected");}); assert.equal(r[0]?.sha256, null);});
  test("identical strip payloads pass", () => {const r = {sha256: "", bytes: 1, programHeaders: "layout", sections: parsed()}; verifyNativeStrip(r, r);});
  test("changed loaded layout is rejected", () => {const r = {sha256: "", bytes: 1, programHeaders: "layout", sections: parsed()}; assert.throws(() => verifyNativeStrip(r, {...r, programHeaders: "changed"}), /loaded layout/);});
  test("changed non-ALLOC GPU metadata is rejected too", () => {const r = {sha256: "", bytes: 1, programHeaders: "layout", sections: parsed().map(row => ({...row, flags: ""}))}; assert.throws(() => verifyNativeStrip(r, {...r, sections: r.sections.map(row => ({...row, sha256: "changed"}))}), /payload/);});
  test("removed runtime sections cannot pass as compactness", () => {const r = {sha256: "", bytes: 1, programHeaders: "layout", sections: parsed()}; assert.throws(() => verifyNativeStrip(r, {...r, sections: []}), /payload/);});
  test("out-of-bounds regular-file reads fail before allocating a payload", t => {const file = path.join(tempDir(t, "mom-native-release-"), "file"); fs.writeFileSync(file, "123"); assert.throws(() => elfRangeHash(file, 0, 4), /Invalid/); assert.throws(() => elfRangeHash(file, -1, 1), /Invalid/);});
});

test.describe("Legacy constructor metadata normalization preserves runtime payloads", () => {
  const original = () => ({sha256: "", bytes: 1, programHeaders: "layout", sections:
    elfSections("  [ 1] .init_array PROGBITS 0000000000001000 001000 000008 00 WA 0 0 8\n", () => "hash")});
  /** @param {ReturnType<typeof original>} r */
  const section = r => {const row = r.sections[0]; assert.ok(row); return row;};
  const stripped = () => {const r = original(); r.sections = r.sections.map(row => ({...row, type: "INIT_ARRAY", entry: "08"})); return r;};
  test("GNU strip may canonicalize legacy .ctors collected into PROGBITS .init_array", () => {verifyNativeStrip(original(), stripped());});
  test("unchanged legacy constructor sections remain admissible", () => {verifyNativeStrip(original(), original());});
  test("an unrelated PROGBITS section cannot change type", () => {
    const a = original(), b = stripped(); section(a).section = ".data"; section(b).section = ".data";
    assert.throws(() => verifyNativeStrip(a, b), /payload/);
  });
  for (const [field, value] of /** @type {[string, string][]} */ ([["sha256", "changed"], ["address", "0000000000002000"], ["flags", "A"], ["offset", "002000"], ["align", "16"]])) {
    test("constructor normalization cannot conceal changed " + field, () => {
      const b = stripped(); Reflect.set(section(b), field, value); assert.throws(() => verifyNativeStrip(original(), b), /payload/);
    });
  }
  test("malformed pointer-array length is not normalized", () => {
    const a = original(), b = stripped(); section(a).size = "000007"; section(b).size = "000007";
    assert.throws(() => verifyNativeStrip(a, b), /pointer-array/);
  });
  test("unexpected legacy pointer entry sizes fail closed", () => {
    const a = original(); section(a).entry = "04"; assert.throws(() => verifyNativeStrip(a, stripped()), /pointer-array/);
  });
});

test.describe("Native release compaction uses actual ELF tooling", {skip: process.platform !== "linux", concurrency: false}, () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-elf-")), original = path.join(root, "original");
  test.before(() => {
    // The release audit admits only glibc 2.31 symbol versions; a host C runtime start file would
    // need __libc_start_main@GLIBC_2.34, so this never-executed fixture provides its own entry point.
    const source = path.join(root, "source.cpp");
    fs.writeFileSync(source, "#include <unistd.h>\nstatic int touched;\n__attribute__((constructor)) static void touch() {touched = 1;}\n" +
      "extern \"C\" void _start() {_exit(touched ? 0 : 1);}\n");
    const r = spawnSync("c++", ["-fPIC", "-pie", "-nostartfiles", source, "-o", original], {encoding: "utf8", timeout: 30000}); assert.equal(r.status, 0, r.stderr);
  });
  test.after(() => fs.rmSync(root, {recursive: true, force: true}));
  test("a real stripped executable retains every required section byte", () => {
    const output = path.join(root, "out"); fs.mkdirSync(output); const r = compactNativeRelease(original, output);
    assert.equal(fs.existsSync(r.binary), false); assert.equal(fs.existsSync(r.pendingBinary), true);
    assert.ok(r.bytes < fs.statSync(original).size); assert.ok(r.needed.includes("libc.so.6"));
  });
  test("compaction ignores PATH binutils and records the exact strip and readelf identities", t => {
    const shadow = path.join(root, "shadow-bin"), output = path.join(root, "pinned-tools"), savedPath = process.env["PATH"];
    fs.mkdirSync(shadow); fs.mkdirSync(output);
    for (const name of ["strip", "readelf"]) {fs.writeFileSync(path.join(shadow, name), "#!/bin/sh\nexit 7\n", {mode: 0o755});}
    process.env["PATH"] = shadow + path.delimiter + savedPath;
    t.after(() => {process.env["PATH"] = savedPath;});
    const r = compactNativeRelease(original, output);
    for (const tool of [r.tools.strip, r.tools.readelf]) {
      assert.ok(path.isAbsolute(tool.path) && !tool.path.startsWith(shadow), tool.path);
      assert.equal(tool.sha256, fileHash(tool.path)); assert.match(tool.version, /\S/);
    }
  });
  test("compaction does not leave a redundant compressed intermediate", () => {
    const output = path.join(root, "single"); fs.mkdirSync(output); compactNativeRelease(original, output);
    assert.deepEqual(fs.readdirSync(output), ["mom.pending"]);
  });
  for (const [name, flags] of /** @type {[string, string[]][]} */ ([["fixed-address executable", ["-fno-pie", "-no-pie"]], ["shared library", ["-fPIC", "-shared"]]])) {
    test(`release admission rejects a real ${name} before stripping`, () => {
      const input = path.join(root, name), output = path.join(root, name + "-output"); fs.mkdirSync(output);
      const built = spawnSync("c++", [...flags, "-nostartfiles", path.join(root, "source.cpp"), "-o", input], {encoding: "utf8", timeout: 30000});
      assert.equal(built.status, 0, built.stderr);
      assert.throws(() => compactNativeRelease(input, output), /position-independent executable/);
      assert.deepEqual(fs.readdirSync(output), [], "rejected input must not create a release candidate");
    });
  }
  test("native snapshots do not execute their input", () => {const r = nativeElf(original); assert.ok(r.sections.length > 10); assert.ok(r.sha256.length === 64);});
  test("malformed regular files cannot masquerade as an executable", () => {const file = path.join(root, "bad"); fs.writeFileSync(file, "not an ELF"); assert.throws(() => nativeElf(file), /inspect native ELF/);});
  test("directory inputs fail before running tools", () => {assert.throws(() => nativeElf(root), /regular file/);});
  test("FIFO inputs cannot block waiting for a producer", () => {const file = path.join(root, "pipe"); assert.equal(spawnSync("mkfifo", [file]).status, 0); assert.throws(() => nativeElf(file), /regular file/);});
  test("already published executables are never replaced", () => {
    const output = path.join(root, "occupied"); fs.mkdirSync(output); fs.writeFileSync(path.join(output, "mom"), "preserve");
    assert.throws(() => compactNativeRelease(original, output), /fresh output/); assert.equal(fs.readFileSync(path.join(output, "mom"), "utf8"), "preserve");
  });
  test("failed-build pending artifacts are preserved too", () => {
    const output = path.join(root, "pending"); fs.mkdirSync(output); fs.writeFileSync(path.join(output, "mom.pending"), "preserve");
    assert.throws(() => compactNativeRelease(original, output), /fresh output/);
  });
  test("stripping preserves executable permissions without a symlink", () => {
    const output = path.join(root, "mode"); fs.mkdirSync(output); const r = compactNativeRelease(original, output), stat = fs.lstatSync(r.pendingBinary);
    assert.equal(stat.isFile(), true); assert.ok(stat.mode & 0o111);
  });
  test("non-ALLOC GPU image metadata survives real compaction", () => {
    const file = path.join(root, "image.bin"), input = path.join(root, "with-image"), output = path.join(root, "image");
    fs.writeFileSync(file, "immutable GPU image fixture"); fs.mkdirSync(output);
    const r = spawnSync("objcopy", ["--add-section", ".mom_image=" + file, "--set-section-flags", ".mom_image=readonly", original, input], {encoding: "utf8"});
    assert.equal(r.status, 0, r.stderr); const artifact = compactNativeRelease(input, output), image = artifact.sections.find(row => row.section === ".mom_image");
    assert.ok(image); assert.equal(image.flags.includes("A"), false);
  });
  test("a real legacy constructor type is repaired by strip without changing its runtime bytes", () => {
    const legacy = path.join(root, "legacy-constructors"), output = path.join(root, "legacy-output"), bytes = fs.readFileSync(original);
    // LLD may inherit PROGBITS from a .ctors input in our constructor collection script.
    // Reproduce only those section-header fields; do not alter any loader segment/payload.
    assert.equal(bytes[4], 2); assert.equal(bytes[5], 1);
    const table = Number(bytes.readBigUInt64LE(40)), size = bytes.readUInt16LE(58), count = bytes.readUInt16LE(60);
    let found = false;
    for (let i = 1; i < count; i++) {
      const offset = table + i * size;
      if (bytes.readUInt32LE(offset + 4) === 14) {
        bytes.writeUInt32LE(1, offset + 4); bytes.writeBigUInt64LE(0n, offset + 56); found = true; break;
      }
    }
    assert.equal(found, true); fs.writeFileSync(legacy, bytes, {mode: 0o755}); fs.mkdirSync(output);
    const before = nativeElf(legacy), result = compactNativeRelease(legacy, output), after = nativeElf(result.pendingBinary);
    assert.equal(before.sections.find(row => row.section === ".init_array")?.type, "PROGBITS");
    assert.equal(after.sections.find(row => row.section === ".init_array")?.type, "INIT_ARRAY");
    assert.equal(before.sections.find(row => row.section === ".init_array")?.sha256,
      after.sections.find(row => row.section === ".init_array")?.sha256);
    verifyNativeStrip(before, after);
  });
});
