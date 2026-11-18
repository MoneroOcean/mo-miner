"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path");
const local = require("../scripts/build-local-native-sdk"), jit = require("../scripts/build-local-native-jit-sdk"),
  rocm = require("../scripts/build-local-native-rocm-sdk"), cuda = require("../scripts/build-local-native-cuda-images"),
  acpp = require("../scripts/build-local-native-acpp-sdk"), nativeRelease = require("../scripts/build-native-release"),
  {noticeInput} = require("../scripts/build-release"),
  {dependencyPins} = require("../scripts/check-native-notices"),
  {fileHash} = require("../scripts/build-source-inputs"),
  {preflightLocalLinuxSdk, buildLocalNativePrerequisites, buildLocalLinuxSdk, writeOrVerifyDescriptor} = require("../scripts/build-local-native-linux-sdk");

const repo = fs.realpathSync(path.resolve(__dirname, "..")), target = "x86_64-unknown-linux-gnu";
const noticeFile = path.join(repo, "THIRD-PARTY-NOTICES.txt");
function fixtureNoticeReview() {
  return {version: 1, notice: {bytes: fs.statSync(noticeFile).size, sha256: fileHash(noticeFile)},
    redistributionEvidence: {}, dependencies: dependencyPins()};
}

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(path.dirname(repo), ".local-linux-sdk-test-")), workspace = path.join(root, "native-sdk");
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, workspace};
}

/** @param {string} workspace
 * @returns {{order:string[],builders:NonNullable<NonNullable<Parameters<typeof buildLocalLinuxSdk>[1]>["builders"]>}} */
function graphBuilders(workspace) {
  const paths = local.supportPaths(workspace), jitPaths = jit.localJitSdkPaths(workspace),
    imagePaths = cuda.localCudaImagePaths(workspace), acppPaths = acpp.localAcppSdkPaths(workspace),
    order = /** @type {string[]} */ ([]);
  const builders = /** @type {NonNullable<NonNullable<Parameters<typeof buildLocalLinuxSdk>[1]>["builders"]>} */ ({
    source(options) {
      order.push("source");
      assert.equal(options?.workspace, workspace);
      return {workspace, compilerFetch: paths.compilerFetch, compilerSources: paths.compilerSources,
        buildInputs: paths.buildInputs, llvmSdk: paths.llvmSdk, sourcesReused: false, llvmReused: false,
        llvmResumed: false, jobs: 2, memoryMax: 4 * 1024 ** 3};
    },
    compilerInputs(source) {
      order.push("compiler-inputs"); assert.equal(source.workspace, workspace);
      return {workspace, dpcppRoot: path.join(paths.dpcppCudaRoot, "dpcpp"), cudaRoot: path.join(paths.dpcppCudaRoot, "cuda"),
        manifest: path.join(paths.dpcppCudaRoot, "inputs.json"), reused: false};
    },
    gnuHeaders(source) {
      order.push("gnu-headers"); assert.equal(source.workspace, workspace);
      return {workspace, root: paths.gnuHeadersRoot, target, reused: false};
    },
    support(source) {
      order.push("support"); assert.equal(source.workspace, workspace);
      return {workspace, hermesSdk: paths.hermesSdk, ioSdk: paths.ioSdk, opensslSdk: paths.opensslSdk,
        zstdSdk: paths.zstdSdk, openmpSdk: paths.openmpSdk,
        linuxSources: paths.linuxSources, reused: {hermes: false, io: false, openssl: false, zstd: false, openmp: false}};
    },
    jitResources(source) {
      order.push("jit-resources"); assert.equal(source.workspace, workspace);
      return {workspace, root: paths.linuxJitResources, target, reused: false};
    },
    jitSdk(context) {
      const selected = /** @type {{source:{workspace:string},resources:{root:string}}} */ (/** @type {unknown} */ (context));
      order.push("jit-sdk"); assert.equal(selected.source.workspace, workspace); assert.equal(selected.resources.root, paths.linuxJitResources);
      return {workspace, compilerSdk: jitPaths.output, compilerIsolation: jitPaths.isolation, isolatedObject: jitPaths.isolatedObject,
        staticDpcppSdk: jitPaths.staticOutput, staticComponent: jitPaths.staticComponent,
        reused: {compilerSdk: false, compilerIsolation: false, staticDpcppSdk: false, staticComponent: false},
        buildOnly: true, executed: false, publicationQualified: false};
    },
    rocmSdk(context) {
      order.push("rocm-sdk"); const selected = rocm.localRocmSdkContext(context);
      assert.equal(selected.workspace, workspace);
      return rocm.rocmSdkResult(selected, {reusedSources: false, reusedComgr: false, reusedHost: false});
    },
    cudaImages(selectedWorkspace) {
      order.push("cuda-images"); assert.equal(selectedWorkspace, workspace);
      return {workspace, handoffRoot: imagePaths.handoffRoot, image: path.join(imagePaths.handoffRoot, "cuda-images.inc")};
    },
    acppSdk(context) {
      const selected = /** @type {{paths:{workspace:string}}} */ (/** @type {unknown} */ (context));
      order.push("acpp-sdk"); assert.equal(selected.paths.workspace, workspace);
      return {workspace, sources: acppPaths.preparedCheckout, sdk: acppPaths.sdk,
        reused: {sources: false, prepared: false, sdk: false}, buildOnly: true, executed: false, publicationQualified: false};
    },
  });
  return {order, builders};
}

function fakeTools() {
  return {git: "/usr/bin/git", systemdRun: "/usr/bin/systemd-run", cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja",
    cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx", cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx"};
}

/** @param {string} root */
function executableTools(root) {
  const bin = path.join(root, "tools"); fs.mkdirSync(bin, {recursive: true});
  const executable = (/** @type {string} */ name) => {
    const file = path.join(bin, name); fs.writeFileSync(file, "fixture executable\n"); fs.chmodSync(file, 0o755); return file;
  };
  return {git: executable("git"), systemdRun: executable("systemd-run"), cmake: executable("cmake"), ninja: executable("ninja"),
    cc: executable("icx"), cxx: executable("icpx")};
}

function fakeSupportTools() {
  const cc = "/opt/intel/oneapi/compiler/2026.2/bin/icx", cxx = "/opt/intel/oneapi/compiler/2026.2/bin/icpx";
  return {hermes: {git: "/usr/bin/git", cmake: "/usr/bin/cmake", cc, cxx, ninja: "/usr/bin/ninja"},
    cmake: "/usr/bin/cmake", ninja: "/usr/bin/ninja",
    openssl: {ar: "/usr/bin/ar", cc, make: "/usr/bin/make", perl: "/usr/bin/perl", ranlib: "/usr/bin/ranlib", tar: "/usr/bin/tar"},
    zstd: {ar: "/usr/bin/ar", cc, make: "/usr/bin/make", mkdir: "/usr/bin/mkdir", tar: "/usr/bin/tar"},
    openmp: {cc, cmake: "/usr/bin/cmake", cxx, ninja: "/usr/bin/ninja", readelf: "/usr/bin/readelf", tar: "/usr/bin/tar"}};
}

function ready(tools = fakeTools()) {
  return {environment: {...process.env, MOM_AOT_BUILD_ONLY: "1"}, tools, supportTools: fakeSupportTools(),
    installedRoots: {dpcpp: "/opt/dpcpp", cuda: "/usr/local/cuda-12.6"}, linker: "/opt/intel/oneapi/compiler/2026.2/lib/ld.lld",
    noticeReview: fixtureNoticeReview()};
}

/** Offline stand-in for the workspace sysroot preparation. @param {string} _root @param {NodeJS.ProcessEnv} environment */
const keepEnvironment = (_root, environment) => environment;

/** @param {string[]} calls @param {boolean} exists */
function pythonFilesystem(calls, exists) {
  return /** @type {Pick<typeof fs,"statSync"|"accessSync">} */ ({
    statSync(file) {
      if (file !== "/usr/bin/python3.12") {calls.push("tool-stat " + file); return /** @type {fs.Stats} */ (/** @type {unknown} */ ({isFile: () => true}));}
      calls.push("python-stat");
      if (!exists) {throw new Error("missing fixture Python");}
      return /** @type {fs.Stats} */ (/** @type {unknown} */ ({isFile: () => true}));
    },
    accessSync(file, mode) {
      if (file !== "/usr/bin/python3.12") {calls.push("tool-access " + file); return;}
      calls.push("python-access"); assert.equal(mode, fs.constants.X_OK);
    },
  });
}

test.describe("Local Linux SDK frontdoor", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("preflights small read-only host prerequisites before workspace work", t => {
    const f = fixture(t), dpcpp = path.join(f.root, "dpcpp"), cudaRoot = path.join(f.root, "cuda"), calls = /** @type {string[]} */ ([]);
    fs.mkdirSync(dpcpp); fs.mkdirSync(cudaRoot);
    const tools = fakeTools(), selected = preflightLocalLinuxSdk({
      nodeVersion: dependencyPins().buildNode.version, environment: {PATH: "/usr/bin"}, tools, supportTools: fakeSupportTools(), installedRoots: {dpcpp, cuda: cudaRoot},
      installedGnuPackages() {calls.push("gnu"); return [];},
      pythonFilesystem: pythonFilesystem(calls, true),
      linker: "/opt/intel/oneapi/compiler/2026.2/lib/ld.lld",
      checkNotices() {calls.push("notices"); return fixtureNoticeReview();},
    });
    assert.deepEqual(calls, ["gnu", "python-stat", "python-access", "tool-stat /usr/bin/dpkg-deb", "tool-access /usr/bin/dpkg-deb",
      "tool-stat /usr/bin/meson", "tool-access /usr/bin/meson", "notices"], "sysroot extraction and libdrm tools are checked before any acquisition");
    assert.deepEqual(selected.installedRoots, {dpcpp, cuda: cudaRoot});
    assert.equal(selected.tools, tools);
    assert.equal(fs.existsSync(f.workspace), false, "preflight does not create the SDK workspace");
  });

  test("a missing GNU prerequisite fails before workspace creation or later preflights", t => {
    const f = fixture(t), dpcpp = path.join(f.root, "dpcpp"), cudaRoot = path.join(f.root, "cuda");
    fs.mkdirSync(dpcpp); fs.mkdirSync(cudaRoot);
    let later = false, workspaceResolved = false;
    assert.throws(() => buildLocalLinuxSdk({}, {
      nodeVersion: dependencyPins().buildNode.version,
      resolveTools: () => fakeTools(), resolveSupportTools: () => fakeSupportTools(), installedRoots: {dpcpp, cuda: cudaRoot},
      installedGnuPackages() {throw new Error("GNU package is not installed: libstdc++-13-dev");},
      checkNotices() {later = true; return fixtureNoticeReview();},
      defaultWorkspace() {workspaceResolved = true; return f.workspace;},
    }), /GNU package is not installed/);
    assert.equal(later, false); assert.equal(workspaceResolved, false); assert.equal(fs.existsSync(f.workspace), false);
  });

  test("a missing fixed ROCm Python executable fails before workspace creation or LLVM acquisition", t => {
    const f = fixture(t), dpcpp = path.join(f.root, "dpcpp"), cudaRoot = path.join(f.root, "cuda"), calls = /** @type {string[]} */ ([]);
    fs.mkdirSync(dpcpp); fs.mkdirSync(cudaRoot);
    let sourceStarted = false, workspaceResolved = false, noticesChecked = false;
    assert.throws(() => buildLocalLinuxSdk({}, {
      nodeVersion: dependencyPins().buildNode.version,
      resolveTools: () => fakeTools(), resolveSupportTools: () => fakeSupportTools(), installedRoots: {dpcpp, cuda: cudaRoot},
      installedGnuPackages() {calls.push("gnu"); return [];},
      pythonFilesystem: pythonFilesystem(calls, false),
      linker: "/opt/intel/oneapi/compiler/2026.2/lib/ld.lld",
      checkNotices() {noticesChecked = true; return fixtureNoticeReview();},
      defaultWorkspace() {workspaceResolved = true; return f.workspace;},
      builders: {source() {sourceStarted = true; throw new Error("LLVM source builder must not run");}},
    }), /Pinned ROCm host Python 3\.12 is missing or not executable/);
    assert.deepEqual(calls, ["gnu", "python-stat"]);
    assert.equal(noticesChecked, false); assert.equal(workspaceResolved, false); assert.equal(sourceStarted, false);
    assert.equal(fs.existsSync(f.workspace), false);
  });

  test("Windows prerequisites reuse the same producers without building an unused Linux JIT SDK", t => {
    const f = fixture(t), graph = graphBuilders(f.workspace), tools = fakeTools(),
      result = buildLocalNativePrerequisites({workspace: f.workspace}, {ensureSysroot: keepEnvironment, preflight: () => ready(tools),
        preflightRocmHost() {return undefined;}, builders: graph.builders});
    assert.deepEqual(graph.order, ["source", "compiler-inputs", "gnu-headers", "support", "jit-resources", "rocm-sdk", "cuda-images", "acpp-sdk"]);
    assert.equal(result.source.llvmSdk, path.join(f.workspace, "llvm21"));
    assert.equal(result.jitSdk, undefined); assert.deepEqual(result.tools, tools);
    assert.equal(fs.existsSync(path.join(f.workspace, "native-linux-sdk.json")), false);
    assert.throws(() => buildLocalNativePrerequisites({linuxJit: /** @type {any} */ ("yes")}), /must be boolean/);
  });

  test("calls the actual producer graph in dependency order and admits a real source-composed descriptor", t => {
    const f = fixture(t), graph = graphBuilders(f.workspace), tools = fakeTools(), order = graph.order;
    const result = buildLocalLinuxSdk({}, {
      preflight() {order.push("preflight"); return ready(tools);},
      ensureSysroot(root, environment) {order.push("sysroot"); assert.equal(root, f.workspace); return environment;},
      defaultWorkspace() {order.push("workspace"); return f.workspace;},
      preflightRocmHost(context, selectedTools) {
        order.push("rocm-host-preflight");
        assert.equal(context.workspace, f.workspace); assert.deepEqual(selectedTools, {cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake});
      },
      builders: graph.builders,
      checkNotices: fixtureNoticeReview,
      admitConfig(config, selectedRepo) {order.push("config-admission"); return nativeRelease.nativeSdkConfig(config, selectedRepo);},
      admitLinkInputs(sdk) {
        order.push("link-input-admission"); assert.ok(sdk.sdkSources?.acpp);
        return {links: [], libraries: [], sdkObjects: [], hashes: {}};
      },
    });
    assert.deepEqual(order, ["preflight", "workspace", "sysroot", "source", "rocm-host-preflight", "compiler-inputs", "gnu-headers",
      "support", "jit-resources", "jit-sdk", "rocm-sdk", "cuda-images", "acpp-sdk", "config-admission", "link-input-admission"]);
    assert.equal(result.workspace, f.workspace); assert.equal(result.configFile, path.join(f.workspace, "native-linux-sdk.json"));
    assert.equal(result.sdkPaths.llvm, path.join(f.workspace, "llvm21/sdk"));
    assert.equal(result.sdkPaths.rocmRuntime, path.join(f.workspace, "rocm-host-sdk/runtime/sdk"));
    const config = JSON.parse(fs.readFileSync(result.configFile, "utf8"));
    assert.equal(config.gpu.dpcpp.args[0], path.join(f.workspace, "dpcpp-cuda-inputs/dpcpp"));
    assert.equal(config.gpu.dpcpp.args[1], path.join(f.workspace, "dpcpp-cuda-inputs/cuda"));
    assert.equal(config.sdkSources.cudaImages, path.join(f.workspace, "cuda-images-handoff/cuda-images.inc"));
    assert.equal(config.sdkSources.rocm, path.join(f.workspace, "rocm-host-sdk/runtime/result.json"));
    assert.deepEqual(config.notices, {path: noticeFile, sha256: fileHash(noticeFile)});
    assert.doesNotThrow(() => noticeInput(config.notices));
    assert.deepEqual(result.notices, config.notices);
    assert.equal(result.buildOnly, true); assert.equal(result.executed, false); assert.equal(result.publicationQualified, false);
  });

  test("forwards admitted DPC++ and GNU records through the actual JIT resource builder boundary", t => {
    const f = fixture(t), graph = graphBuilders(f.workspace), tools = executableTools(f.root), calls = /** @type {string[]} */ ([]),
      paths = local.supportPaths(f.workspace), compilerInputs = {workspace: f.workspace, dpcppRoot: path.join(paths.dpcppCudaRoot, "dpcpp"),
        cudaRoot: path.join(paths.dpcppCudaRoot, "cuda"), manifest: path.join(paths.dpcppCudaRoot, "inputs.json"), reused: false},
      gnuHeaders = {workspace: f.workspace, root: paths.gnuHeadersRoot, target, reused: false};
    let compilerAdmissions = 0, gnuAdmissions = 0;
    graph.builders.compilerInputs = source => {
      compilerAdmissions++; assert.equal(source.workspace, f.workspace); return compilerInputs;
    };
    graph.builders.gnuHeaders = source => {
      gnuAdmissions++; assert.equal(source.workspace, f.workspace); return gnuHeaders;
    };
    graph.builders.jitResources = (source, dependencies = {}) => {
      const prepareInputs = dependencies.prepareInputs, prepareHeaders = dependencies.prepareHeaders;
      assert.ok(prepareInputs); assert.ok(prepareHeaders);
      assert.equal(prepareInputs(source, dependencies), compilerInputs);
      assert.equal(prepareHeaders(source, dependencies), gnuHeaders);
      return local.buildLocalLinuxJitResources(source, dependencies);
    };
    /** @type {typeof local.runCommand} */
    const run = (command, args, settings) => {
      const split = args.indexOf("--"), child = split < 0 ? args : args.slice(split + 1);
      calls.push(settings.label);
      assert.equal(command, tools.systemdRun); assert.ok(args.includes("--property=MemoryMax=" + local.localMemoryMax(settings.env)));
      assert.ok(args.includes("--property=MemorySwapMax=0")); assert.equal(child[0], process.execPath);
      if (settings.label === "Linux JIT resource tree producer") {
        assert.deepEqual(child.slice(1), [path.join(repo, "scripts/build-local-native-sdk.js"), "--local-jit-resources", "build", f.workspace,
          tools.cmake, tools.ninja, tools.cc, tools.cxx]);
        fs.mkdirSync(paths.linuxJitResources);
      }
      const empty = Buffer.alloc(0);
      return {pid: 1, output: [empty, empty, empty], stdout: empty, stderr: empty, status: 0, signal: null};
    };
    const result = buildLocalLinuxSdk({}, {ensureSysroot: keepEnvironment, preflight: () => ready(tools), defaultWorkspace: () => f.workspace, run,
      preflightRocmHost() {calls.push("rocm-host-preflight");}, builders: graph.builders, checkNotices: fixtureNoticeReview,
      admitConfig: nativeRelease.nativeSdkConfig,
      admitLinkInputs(sdk) {
        assert.ok(sdk.sdkSources?.acpp);
        return {links: [], libraries: [], sdkObjects: [], hashes: {}};
      }});
    assert.equal(compilerAdmissions, 1); assert.equal(gnuAdmissions, 1);
    assert.deepEqual(calls, ["rocm-host-preflight", "systemd memory-limit preflight", "LLVM 21 source SDK re-admission",
      "Linux JIT resource tree producer"]);
    assert.equal(result.sdkPaths.gnuHeaders, paths.gnuHeadersRoot);
    assert.ok(fs.existsSync(paths.linuxJitTools), "the actual JIT resource builder admitted its tool descriptor");
  });

  test("a failed stage stops downstream builders and leaves no descriptor", t => {
    const f = fixture(t), graph = graphBuilders(f.workspace), calls = graph.order;
    graph.builders.jitResources = () => {calls.push("jit-resources"); throw new Error("fixture JIT failure");};
    assert.throws(() => buildLocalLinuxSdk({}, {ensureSysroot: keepEnvironment, preflight: () => ready(), defaultWorkspace: () => f.workspace,
      preflightRocmHost() {calls.push("rocm-host-preflight");}, builders: graph.builders}), /fixture JIT failure/);
    assert.deepEqual(calls, ["source", "rocm-host-preflight", "compiler-inputs", "gnu-headers", "support", "jit-resources"]);
    assert.equal(fs.existsSync(path.join(f.workspace, "native-linux-sdk.json")), false);
  });

  test("public descriptor admission failure stops before link admission or writing", t => {
    const f = fixture(t), graph = graphBuilders(f.workspace); let linked = false;
    assert.throws(() => buildLocalLinuxSdk({}, {ensureSysroot: keepEnvironment, preflight: () => ready(), defaultWorkspace: () => f.workspace,
      preflightRocmHost() {assert.ok(true);}, builders: graph.builders, checkNotices: fixtureNoticeReview,
      admitConfig() {throw new Error("invalid maintained release schema");},
      admitLinkInputs() {linked = true; return {links: [], libraries: [], sdkObjects: [], hashes: {}};}}),
    /invalid maintained release schema/);
    assert.equal(linked, false); assert.equal(fs.existsSync(path.join(f.workspace, "native-linux-sdk.json")), false);
  });

  test("descriptor reuse is exact and changed output is preserved", t => {
    const f = fixture(t), file = path.join(f.root, "descriptor.json"), config = {version: 1};
    writeOrVerifyDescriptor(file, config);
    assert.doesNotThrow(() => writeOrVerifyDescriptor(file, config));
    fs.writeFileSync(file, "{\"version\":2}\n");
    assert.throws(() => writeOrVerifyDescriptor(file, config), /different or aliased content/);
    assert.equal(fs.readFileSync(file, "utf8"), "{\"version\":2}\n");
  });
});
