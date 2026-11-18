"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), os = require("node:os");
const local = require("../scripts/build-local-native-sdk"), nativeCi = require("../scripts/native-ci-linux-rocm"),
  {localRocmSdkPaths, localRocmSdkContextFromWorkspace, localRocmSdkContext, localRocmBuildState, localRocmHostState,
    localRocmTools, localRocmHostInputs, runLocalRocmSdkChild, buildLocalLinuxRocmSdk, main} = require("../scripts/build-local-native-rocm-sdk");

const repo = fs.realpathSync(path.resolve(__dirname, "..")), linuxX64 = process.platform === "linux" && process.arch === "x64";

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-local-native-rocm-sdk-test-")), workspace = path.join(root, "native-sdk");
  local.admitWorkspace(workspace, repo);
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, workspace, context: localRocmSdkContextFromWorkspace(workspace), paths: localRocmSdkPaths(workspace)};
}

/** @param {string} file @param {string} [text] */
function write(file, text = "fixture\n") {
  fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, text, {flag: "wx", mode: 0o600});
}

/** @param {string} root */
function fakeTools(root) {
  const bin = path.join(root, "tools"); fs.mkdirSync(bin);
  /** @param {string} name */
  const executable = name => {const file = path.join(bin, name); write(file); fs.chmodSync(file, 0o755); return file;};
  return {git: executable("git"), systemdRun: executable("systemd-run"), cmake: executable("cmake"), ninja: executable("ninja"),
    cc: executable("icx"), cxx: executable("icpx"), ar: executable("ar"), dpkgQuery: executable("dpkg-query"),
    llvmObjcopy: executable("llvm-objcopy"), perl: executable("perl"), pkgConfig: executable("pkg-config"),
    python3: executable("python3.12"), ranlib: executable("ranlib"), tar: executable("tar"), xxd: executable("xxd")};
}

/** @param {string} workspace @param {ReturnType<typeof fakeTools>} tools @param {string} root */
function hostInputs(workspace, tools, root) {
  const paths = localRocmSdkPaths(workspace), dependencies = {
    drm: path.join(root, "deps/libdrm.a"), drmAmdgpu: path.join(root, "deps/libdrmAmdgpu.a"),
    elf: path.join(root, "deps/libelf.a"), numa: path.join(root, "deps/libnuma.a")};
  for (const file of Object.values(dependencies)) {write(file, "!<arch>\n");}
  const hostTools = {ar: tools.ar, cmake: tools.cmake, dpkgQuery: tools.dpkgQuery, git: tools.git,
    llvmObjcopy: tools.llvmObjcopy, ninja: tools.ninja, perl: tools.perl, pkgConfig: tools.pkgConfig,
    python3: tools.python3, ranlib: tools.ranlib, tar: tools.tar, xxd: tools.xxd, cc: tools.cc, cxx: tools.cxx};
  const config = {version: 1, sources: {clr: path.join(paths.source, "roots/clr"), hip: path.join(paths.source, "roots/hip"),
    rocr: path.join(paths.source, "roots/rocr")},
  tools: {ar: tools.ar, cmake: tools.cmake, dpkgQuery: tools.dpkgQuery, git: tools.git, llvmObjcopy: tools.llvmObjcopy,
    ninja: tools.ninja, perl: tools.perl, pkgConfig: tools.pkgConfig, python3: tools.python3, ranlib: tools.ranlib, tar: tools.tar, xxd: tools.xxd},
  hostCompilers: {cc: tools.cc, cxx: tools.cxx}, dependencies, libdrm: path.join(root, "libdrm"), pkgConfigDir: path.join(root, "pkg-config"),
  comgrReceipt: path.join(paths.sdkRoot, "result.json")};
  return {hostTools, config, verifyHostPrerequisites() {return undefined;}};
}

/** @returns {import("node:child_process").SpawnSyncReturns<Buffer<ArrayBuffer>>} */
function successfulRun() {
  return {pid: 1, output: [null, Buffer.alloc(0), Buffer.alloc(0), Buffer.alloc(0)], stdout: Buffer.alloc(0),
    stderr: Buffer.alloc(0), status: 0, signal: null};
}

/** Seed only receipt path shapes; source validation is injected so no network or Git fetch occurs. @param {ReturnType<typeof localRocmSdkPaths>} paths */
function sourceFixture(paths) {
  fs.mkdirSync(paths.source, {mode: 0o700}); fs.mkdirSync(paths.pythonSourceRoot, {mode: 0o700});
  write(paths.fetchReceipt, "{}\n"); write(paths.rootsFile, "{}\n");
  write(paths.pythonArchive); write(paths.pythonSourceReceipt, "{}\n");
}

test.describe("Local Linux ROCm SDK slice", {skip: linuxX64 ? false : "Linux x64 build-only ROCm adapter"}, () => {
  test("re-admits its own admitted context, as the auditors receive it, but not a redirected one", t => {
    const f = fixture(t), context = localRocmSdkContext(f.context);
    assert.deepEqual(localRocmSdkContext(context), context);
    assert.throws(() => localRocmSdkContext({...context, source: {...context.source, llvmSdk: "/elsewhere"}}),
      /do not match the exact admitted workspace paths/);
    assert.throws(() => localRocmSdkContext({...context, extra: true}), /needs only an admitted source-LLVM stage result/);
  });

  test("uses stable marked-workspace paths and admits only the exact source-LLVM context", t => {
    const f = fixture(t), context = localRocmSdkContext(f.context);
    assert.equal(context.workspace, f.workspace);
    assert.deepEqual(context.paths, f.paths);
    assert.deepEqual(localRocmSdkPaths(f.workspace), f.paths, "local cache paths do not grow by run or timestamp");
    assert.equal(context.source.llvmSdk, path.join(f.workspace, "llvm21"), "the LLVM source/output root is the admission target");
    assert.equal(f.paths.source, path.join(f.workspace, "rocm-sources"));
    assert.equal(f.paths.runtimeOutput, path.join(f.workspace, "rocm-host-sdk/runtime"));
    assert.equal(path.join(f.paths.runtimeOutput, "sdk"), path.join(f.workspace, "rocm-host-sdk/runtime/sdk"));
    assert.throws(() => localRocmSdkContext({...f.context, source: {...f.context.source, llvmSdk: path.join(f.root, "other")}}),
      /exact admitted workspace paths/);
    assert.throws(() => localRocmSdkContext({source: f.context.source, fakeCiTar: "/tmp/fake.tar"}), /only an admitted source-LLVM/);
  });

  test("passes the exact shared ROCm resource/COMGR recipe and no CI transport fields", t => {
    const f = fixture(t), state = localRocmBuildState(localRocmSdkContext(f.context)), tools = {
        cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx", cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx", cmake: "/usr/bin/cmake"},
      config = nativeCi.rocmResourcesConfig(state, tools), commands = nativeCi.rocmComgrBuildCommands(state, tools), host = localRocmHostState(localRocmSdkContext(f.context));
    assert.deepEqual(config, {version: 1, target: "linux", source: path.join(f.paths.source, "roots/rocmLlvm"),
      clr: path.join(f.paths.source, "roots/clr"), llvm: path.join(f.workspace, "llvm21/sdk"), cmake: tools.cmake,
      hipHeaders: f.paths.hipIncludeRoot, hostCompilers: {cc: tools.cc, cxx: tools.cxx}});
    assert.deepEqual(commands, [
      {script: "scripts/build-native-rocm-resources.js", args: [f.paths.resourcesConfig, f.paths.resourcesRoot], config},
      {script: "scripts/build-native-comgr-sdk.js", args: [path.join(f.paths.source, "roots/rocmLlvm"), path.join(f.workspace, "llvm21/sdk"),
        path.join(f.workspace, "llvm21/sdk"), path.join(f.paths.resourcesRoot, "install"), f.paths.sdkRoot, tools.cc, tools.cxx]},
    ]);
    assert.equal(host.runtimeOutput, f.paths.runtimeOutput);
    assert.equal(host.sdkRoot, f.paths.sdkRoot);
    assert.equal(host.llvmSdk, path.join(f.workspace, "llvm21/sdk"));
    assert.notEqual(host.llvmSdk, localRocmSdkContext(f.context).source.llvmSdk, "ROCm consumers bind the installed LLVM SDK prefix, not the output root");
    for (const forbidden of ["suffix", "runnerTemp", "sourceTar", "incomingComgrTar", "hostSdkTar"]) {
      assert.equal(Object.hasOwn(host, forbidden), false, "local state does not fabricate CI field " + forbidden);
    }
  });

  test("preserves an icpx symlink for COMGR and admits the same canonical compiler for the public host producer", t => {
    const f = fixture(t), tools = fakeTools(f.root), compilerBin = path.join(f.root, "oneapi/compiler/2026.2/bin"),
      compiler = path.join(compilerBin, "icx"), invocationAlias = path.join(compilerBin, "icpx");
    fs.mkdirSync(compilerBin, {recursive: true}); write(compiler); fs.chmodSync(compiler, 0o755); fs.symlinkSync("icx", invocationAlias);
    const selected = localRocmTools({cc: compiler, cxx: invocationAlias, cmake: tools.cmake}, compilerBin);
    assert.equal(selected.cxx, invocationAlias, "COMGR retains the selected C++ driver spelling");
    assert.equal(fs.realpathSync(selected.cxx), compiler);

    const configured = hostInputs(f.workspace, tools, f.root), canonicalTools = {...configured.hostTools, cc: compiler, cxx: compiler},
      canonicalConfig = {...configured.config, hostCompilers: {cc: compiler, cxx: compiler}},
      host = localRocmHostInputs(localRocmSdkContext(f.context), selected, process.env, {
        hostTools: () => canonicalTools, hostConfig: () => canonicalConfig, verifyHostPrerequisites() {return undefined;},
      });
    assert.equal(host.tools.cxx, compiler, "the shared public host producer receives its required canonical path");
    assert.equal(host.config.hostCompilers.cxx, compiler);
    assert.equal(fs.realpathSync(host.config.hostCompilers.cxx), host.config.hostCompilers.cxx);

    const external = path.join(f.root, "outside/icpx"); write(external); fs.chmodSync(external, 0o755);
    fs.unlinkSync(invocationAlias); fs.symlinkSync(external, invocationAlias);
    assert.throws(() => localRocmTools({cc: compiler, cxx: invocationAlias, cmake: tools.cmake}, compilerBin), /outside the pinned Intel 2026\.2 SDK/);
  });

  test("runs both public entry commands only inside the bounded systemd scope", t => {
    const f = fixture(t), tools = fakeTools(f.root),
      /** @type {Array<{command:string,args:string[],options:{label:string,env:NodeJS.ProcessEnv,cwd:string}}>} */ calls = [];
    /** @param {string} command @param {string[]} args @param {{label:string,env:NodeJS.ProcessEnv,cwd:string}} options */
    function run(command, args, options) {
      calls.push({command, args, options}); assert.equal(command, tools.systemdRun);
      assert.ok(args.includes("--user")); assert.ok(args.includes("--scope")); assert.ok(args.includes("--quiet"));
      assert.ok(args.includes("--property=MemoryMax=" + local.localMemoryMax(options.env))); assert.ok(args.includes("--property=MemorySwapMax=0"));
      assert.equal(options.env["MOM_AOT_BUILD_ONLY"], "1");
      return successfulRun();
    }
    const result = buildLocalLinuxRocmSdk(f.context, {tools, run, validateTools: () => ({cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake})});
    assert.deepEqual(calls.map(row => row.options.label), ["systemd memory-limit preflight", "Local Linux ROCm/COMGR and host SDK build-only slice"]);
    const probe = calls[0]?.args.slice(calls[0].args.indexOf("--") + 1) || [];
    assert.deepEqual(probe.slice(1), [path.join(repo, "scripts/build-local-native-rocm-sdk.js"), "--memory-probe"]);
    const build = calls[1]?.args.slice(calls[1].args.indexOf("--") + 1) || [];
    assert.deepEqual(build.slice(1), [path.join(repo, "scripts/build-local-native-rocm-sdk.js"), "--local-rocm-sdk", f.workspace,
      tools.cmake, tools.cc, tools.cxx]);
    assert.equal(result.rocmSdk, path.join(f.paths.runtimeOutput, "sdk"));
    assert.equal(result.reused.reusedHost, false);
    for (const output of [f.paths.source, f.paths.hipHeadersRoot, f.paths.resourcesRoot, f.paths.sdkRoot, f.paths.hostSdkRoot]) {
      assert.equal(fs.existsSync(output), false, output + " is created only by the scoped child");
    }
  });

  test("forwards an enclosing-scope marker to the shared invoker and keeps both children in that scope", t => {
    const f = fixture(t), tools = fakeTools(f.root), environment = {...process.env, MOM_LOCAL_NATIVE_SCOPE: "1"},
      calls = /** @type {Array<{command:string,args:string[],options:{label:string,env:NodeJS.ProcessEnv,cwd:string}}>} */ ([]);
    let proofs = 0;
    const invoke = local.scopedInvoker;
    /** @type {typeof local.scopedInvoker} */
    const scopedInvoker = (selectedTools, selectedEnvironment, run) => {
      assert.equal(selectedEnvironment["MOM_LOCAL_NATIVE_SCOPE"], "1");
      return invoke(selectedTools, selectedEnvironment, run, () => {proofs++;});
    };
    t.mock.method(local, "scopedInvoker", scopedInvoker);
    /** @type {typeof local.runCommand} */
    const run = (command, args, options) => {
      calls.push({command, args, options});
      assert.equal(command, process.execPath, "the child runs directly instead of creating a sibling systemd scope");
      assert.equal(options.env["MOM_LOCAL_NATIVE_SCOPE"], "1");
      return successfulRun();
    };
    buildLocalLinuxRocmSdk(f.context, {tools, run, environment,
      validateTools: () => ({cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake})});
    assert.equal(proofs, 2);
    assert.deepEqual(calls.map(row => row.options.label), ["systemd memory-limit preflight", "Local Linux ROCm/COMGR and host SDK build-only slice"]);
  });

  test("propagates a failed systemd child and does not stage in the wrapper", t => {
    const f = fixture(t), tools = fakeTools(f.root); let count = 0;
    /** @param {string} _command @param {string[]} _args @param {{label:string,env:NodeJS.ProcessEnv,cwd:string}} options */
    function run(_command, _args, options) {
      if (++count === 2) {throw new Error(options.label + " failed (7)");}
      return successfulRun();
    }
    assert.throws(() => buildLocalLinuxRocmSdk(f.context, {tools, run, validateTools: () => ({cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake})}),
      /build-only slice failed \(7\)/);
    assert.equal(count, 2);
    assert.equal(fs.existsSync(f.paths.source), false);
    assert.equal(fs.existsSync(f.paths.sdkRoot), false);
  });

  test("the internal producer refuses to proceed before proving its own cgroup scope", t => {
    const f = fixture(t); let validated = false, ran = false;
    assert.throws(() => runLocalRocmSdkChild(f.context, {}, {
      verifyScope() {throw new Error("cgroup cap missing");},
      validateTools() {validated = true; throw new Error("tool validation must not run");},
      run() {ran = true; throw new Error("source fetch must not run");},
    }), /cgroup cap missing/);
    assert.equal(validated, false); assert.equal(ran, false);
    for (const output of [f.paths.source, f.paths.hipHeadersRoot, f.paths.resourcesRoot, f.paths.sdkRoot, f.paths.hostSdkRoot]) {
      assert.equal(fs.existsSync(output), false, output);
    }
  });

  test("the direct producer CLI proves the cap before resolving/versioning tools", t => {
    const f = fixture(t);
    assert.throws(() => main(["--local-rocm-sdk", f.workspace, "/usr/bin/cmake",
      "/opt/intel/oneapi/compiler/2026.2/bin/icx", "/opt/intel/oneapi/compiler/2026.2/bin/icpx"],
    () => {throw new Error("cgroup cap missing");}), /cgroup cap missing/);
  });

  test("missing host prerequisites fail before network acquisition or COMGR production", t => {
    const f = fixture(t), tools = fakeTools(f.root), host = hostInputs(f.workspace, tools, f.root); let ran = false, built = false;
    assert.throws(() => runLocalRocmSdkChild(f.context, {cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake}, {
      verifyScope() {return undefined;}, validateTools: () => ({cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake}), verifyInputs() {return undefined;},
      hostTools: () => host.hostTools, hostConfig: () => host.config,
      verifyHostPrerequisites() {throw new Error("pinned Python or package prerequisites missing");},
      run() {ran = true; throw new Error("source fetch must not run");}, buildComgr() {built = true; throw new Error("COMGR build must not run");},
    }), /pinned Python or package prerequisites missing/);
    assert.equal(ran, false, "does not fetch ROCm sources until host prerequisites pass");
    assert.equal(built, false);
    assert.equal(fs.existsSync(f.paths.source), false);
    assert.equal(fs.existsSync(f.paths.sdkRoot), false);
  });

  test("reuses the exact fetched-source state and hands the public COMGR core its LLVM callback", t => {
    const f = fixture(t), tools = fakeTools(f.root), host = hostInputs(f.workspace, tools, f.root), selectedJobs = process.env["MOM_BUILD_JOBS"];
    sourceFixture(f.paths);
    // The shared CI COMGR recipe admits only 2 or 4 jobs; a larger local selection maps to 4.
    process.env["MOM_BUILD_JOBS"] = "16";
    t.after(() => {if (selectedJobs === undefined) {delete process.env["MOM_BUILD_JOBS"];} else {process.env["MOM_BUILD_JOBS"] = selectedJobs;}});
    let inputChecks = 0, coreCalls = 0, sourceChecks = 0;
    assert.throws(() => runLocalRocmSdkChild(f.context, {cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake}, {
      verifyScope() {return undefined;}, validateTools: () => ({cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake}), verifyInputs() {inputChecks++; return undefined;},
      verifySources(_paths, state, git) {
        sourceChecks++; assert.equal(state.source, f.paths.source); assert.equal(state.rootsFile, f.paths.rootsFile);
        assert.equal(state.pythonArchive, f.paths.pythonArchive); assert.equal(git, "/usr/bin/git");
      }, hostTools: () => host.hostTools, hostConfig: () => host.config, verifyHostPrerequisites() {return undefined;},
      buildComgr(state, buildTools, env, verifyLlvm) {
        coreCalls++;
        assert.equal(state.source, f.paths.source); assert.equal(state.llvmSdk, path.join(f.workspace, "llvm21/sdk"));
        assert.equal(state.hipHeadersRoot, f.paths.hipHeadersRoot); assert.equal(state.resourcesRoot, f.paths.resourcesRoot);
        assert.equal(state.resourcesConfig, f.paths.resourcesConfig); assert.equal(state.sdkRoot, f.paths.sdkRoot);
        for (const forbidden of ["sourceTar", "incomingSourceTar", "sdkTar", "runnerTemp", "suffix"]) {
          assert.equal(Object.hasOwn(state, forbidden), false, "no fake CI transport field " + forbidden);
        }
        assert.deepEqual(buildTools, {cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake});
        assert.equal(env["MOM_AOT_BUILD_ONLY"], "1"); assert.equal(env["MOM_BUILD_JOBS"], "4");
        verifyLlvm();
        throw new Error("fixture stop before public producers");
      },
    }), /fixture stop before public producers/);
    assert.equal(inputChecks, 2, "the public builder's preflight callback re-admits LLVM");
    assert.equal(coreCalls, 1); assert.equal(sourceChecks, 1);
    assert.equal(fs.existsSync(f.paths.resourcesConfig), false);
    assert.equal(fs.existsSync(f.paths.comgrAdmission), false);
    assert.equal(fs.existsSync(f.paths.hostSdkRoot), false);
  });

  test("source-fetch failure stops before parser, COMGR, or host output creation", t => {
    const f = fixture(t), tools = fakeTools(f.root), host = hostInputs(f.workspace, tools, f.root); let fetches = 0, parser = 0, built = 0;
    assert.throws(() => runLocalRocmSdkChild(f.context, {cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake}, {
      verifyScope() {return undefined;}, validateTools: () => ({cc: tools.cc, cxx: tools.cxx, cmake: tools.cmake}), verifyInputs() {return undefined;},
      hostTools: () => host.hostTools, hostConfig: () => host.config, verifyHostPrerequisites() {return undefined;},
      run(command, args, options) {
        fetches++;
        assert.equal(command, process.execPath);
        assert.equal(args[0], path.join(repo, "scripts/fetch-native-sdk-sources.js"));
        assert.deepEqual(args.slice(1), ["rocm", "/usr/bin/git", f.paths.source, "--allow-network"]);
        assert.equal(options.env["MOM_AOT_BUILD_ONLY"], "1");
        throw new Error("mock source fetch failed");
      },
      fetchParser() {parser++; throw new Error("parser fetch must not run");}, buildComgr() {built++; throw new Error("COMGR build must not run");},
    }), /mock source fetch failed/);
    assert.equal(fetches, 1); assert.equal(parser, 0); assert.equal(built, 0);
    for (const output of [f.paths.source, f.paths.hipHeadersRoot, f.paths.resourcesRoot, f.paths.sdkRoot, f.paths.hostSdkRoot]) {
      assert.equal(fs.existsSync(output), false, output);
    }
  });
});
