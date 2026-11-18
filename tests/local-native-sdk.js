"use strict";

const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"),
  {spawnSync} = require("node:child_process");
const {buildLocalNativeSdk, buildLocalNativeSupportSdk, buildLocalCompilerInputs, buildLocalGnuHeaders, stageLocalCompilerInputs,
  prepareOrAdmitLinuxSources, prepareOrAdmitWindowsSources, main,
  buildLocalLinuxJitResources, localLinuxJitResourceState, runLocalJitResourceChild, localJitToolsReceipt,
  workspaceOption, localDpcppCudaRoots, admitWorkspace, readConfigureState, cgroupMemoryPaths, verifyMemoryLimits, trustedSystemdRun, supportPaths, memoryMax, scopedInvoker} =
  require("../scripts/build-local-native-sdk");
const {archiveName} = require("../scripts/native-llvm-sdk");
const {sdkComponents} = require("../scripts/prepare-native-sdk-sources");
const ciSupport = require("../scripts/native-ci-linux-support");

const repo = fs.realpathSync(path.resolve(__dirname, "..")), node = process.execPath;

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(path.dirname(repo), ".local-native-sdk-test-")), workspace = path.join(root, "native-sdk");
  admitWorkspace(workspace, repo);
  const compilerSources = path.join(workspace, "compiler-sources"), buildInputs = path.join(workspace, "llvm-inputs");
  fs.mkdirSync(compilerSources);
  for (const name of Object.keys(sdkComponents("compiler"))) {fs.mkdirSync(path.join(compilerSources, name));}
  writeJson(path.join(compilerSources, "sources.json"), {version: 2, target: "compiler", status: "PASS",
    prepared: Object.fromEntries(Object.keys(sdkComponents("compiler")).map(name => [name, {root: path.join(compilerSources, name)}]))});
  fs.mkdirSync(buildInputs); fs.writeFileSync(path.join(buildInputs, archiveName), "fixture archive\n");
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, workspace, compilerSources, buildInputs, llvmOutput: path.join(workspace, "llvm21")};
}

/** @param {string} output */
function configuredReceipt(output) {
  return {version: 1, role: "linux-llvm21-source-sdk", stage: "configured", plan: {output, prefix: path.join(output, "sdk"), phases: [
    {name: "llvm-foundation", targets: ["Support"], artifacts: [path.join(output, "build", "libSupport.a")]},
    {name: "tools", targets: ["clang"], artifacts: [path.join(output, "build", "bin", "clang")]}
  ]}};
}

/** @param {string} file @param {unknown} value */
function writeJson(file, value) {fs.writeFileSync(file, JSON.stringify(value) + "\n");}

/** @param {{git: string, systemdRun: string, cmake: string, ninja: string, cc: string, cxx: string}} tools
 * @param {{failLabel?: string,afterRun?:(label:string,child:string[])=>void}} [options] */
function mockRun(tools, options = {}) {
  /** @type {{command: string, args: string[], label: string}[]} */ const calls = [];
  /** @param {string} command @param {string[]} args @param {{label: string, env: NodeJS.ProcessEnv, cwd: string}} settings
   * @returns {import("node:child_process").SpawnSyncReturns<import("node:buffer").NonSharedBuffer>} */
  function run(command, args, settings) {
    const label = settings.label, child = args.includes("--") ? args.slice(args.indexOf("--") + 1) : args;
    calls.push({command, args, label});
    assert.equal(command, tools.systemdRun);
    assert.ok(args.includes("--scope")); assert.ok(args.includes("--quiet")); assert.ok(args.includes("--property=MemoryMax=" + memoryMax));
    assert.ok(args.includes("--property=MemorySwapMax=0")); assert.ok(!args.includes("--wait")); assert.equal(child[0], node);
    if (label === "systemd memory-limit preflight") {
      assert.equal(child[1], path.join(repo, "scripts/build-local-native-sdk.js"));
      assert.equal(child[2], "--memory-probe");
    }
    if (label === options.failLabel) {throw new Error(label + " fixture failure");}
    if (label === "LLVM 21 configure") {
      const output = child[4] || ""; fs.mkdirSync(output, {recursive: true}); writeJson(path.join(output, "configured.json"), configuredReceipt(output));
    } else if (label.startsWith("LLVM 21 build phase ")) {
      const phase = label.slice("LLVM 21 build phase ".length), output = child[3] || "";
      fs.writeFileSync(path.join(output, "phase-" + phase + ".json"), "public phase command fixture\n");
      if (phase === "tools") {fs.writeFileSync(path.join(output, "build-complete.json"), "public build completion fixture\n");}
    } else if (label === "LLVM 21 install") {
      fs.writeFileSync(path.join(child[3] || "", "install.json"), "public install command fixture\n");
    } else if (label === "LLVM 21 audit") {
      fs.writeFileSync(path.join(child[3] || "", "result.json"), "public admission command fixture\n");
    }
    options.afterRun?.(label, child);
    const empty = /** @type {import("node:buffer").NonSharedBuffer} */ (Buffer.alloc(0));
    return {pid: 1, output: [empty, empty, empty], stdout: empty, stderr: empty, status: 0, signal: null};
  }
  return {run, calls};
}

/** @param {string} root */
function fakeTools(root) {
  const bin = path.join(root, "tools"); fs.mkdirSync(bin, {recursive: true});
  const actualCxx = path.join(bin, "icx-real"), cxx = path.join(bin, "icpx");
  fs.writeFileSync(actualCxx, "fixture compiler\n"); fs.symlinkSync(actualCxx, cxx);
  return {git: path.join(bin, "git"), systemdRun: "/usr/bin/systemd-run", cmake: path.join(bin, "cmake"), ninja: path.join(bin, "ninja"),
    cc: path.join(bin, "icx"), cxx};
}

/** @param {string} root @param {ReturnType<typeof fakeTools>} [baseTools] */
function fakeSupportTools(root, baseTools = fakeTools(root)) {
  const base = baseTools, bin = path.dirname(base.git), tool = (/** @type {string} */ name) => path.join(bin, name);
  return {hermes: {git: base.git, cmake: base.cmake, cc: base.cc, cxx: base.cxx, ninja: base.ninja}, cmake: base.cmake, ninja: base.ninja,
    openssl: {ar: tool("ar"), cc: base.cc, make: tool("make"), perl: tool("perl"), ranlib: tool("ranlib"), tar: tool("tar")},
    zstd: {ar: tool("ar"), cc: base.cc, make: tool("make"), mkdir: tool("mkdir"), tar: tool("tar")},
    openmp: {cc: base.cc, cmake: base.cmake, cxx: base.cxx, ninja: base.ninja, readelf: tool("readelf"), tar: tool("tar")}};
}

/** @param {ReturnType<typeof fakeTools>} tools @param {{failLabel?: string}} [options] */
function mockSupportRun(tools, options = {}) {
  /** @type {{label: string, child: string[]}[]} */ const calls = [];
  /** @param {string} command @param {string[]} args @param {{label: string, env: NodeJS.ProcessEnv, cwd: string}} settings
   * @returns {import("node:child_process").SpawnSyncReturns<import("node:buffer").NonSharedBuffer>} */
  function run(command, args, settings) {
    const split = args.indexOf("--"), child = split < 0 ? args : args.slice(split + 1);
    calls.push({label: settings.label, child});
    assert.equal(command, tools.systemdRun); assert.ok(args.includes("--scope")); assert.ok(args.includes("--quiet"));
    assert.ok(args.includes("--property=MemoryMax=" + memoryMax)); assert.ok(args.includes("--property=MemorySwapMax=0"));
    assert.ok(!args.includes("--wait")); assert.equal(child[0], node); assert.equal(settings.env["MOM_AOT_BUILD_ONLY"], "1");
    if (settings.label === options.failLabel) {throw new Error(settings.label + " fixture failure");}
    const mkdir = (/** @type {string | undefined} */ output) => {if (output) {fs.mkdirSync(output, {recursive: true});}};
    if (settings.label === "pinned Linux source preparation and re-admission") {
      const output = supportPaths(child[3] || "").linuxSources; mkdir(output);
      if (!fs.existsSync(path.join(output, "sources.json"))) {writeJson(path.join(output, "sources.json"), {version: 2, target: "linux", status: "PASS", prepared: {}});}
    } else if (settings.label === "pinned Hermes source fetch") {mkdir(child[4]);} else if (settings.label === "pinned I/O source archive fetch" ||
        settings.label === "pinned zstd source archive fetch") {mkdir(child[3]);} else if (settings.label === "static Hermes SDK producer") {mkdir(child[3]);} else if (
      settings.label === "static I/O SDK producer") {mkdir(child[4]);} else if (["static OpenSSL SDK producer", "static zstd SDK producer", "static OpenMP SDK producer"].includes(settings.label)) {mkdir(child[3]);}
    const empty = /** @type {import("node:buffer").NonSharedBuffer} */ (Buffer.alloc(0));
    return {pid: 1, output: [empty, empty, empty], stdout: empty, stderr: empty, status: 0, signal: null};
  }
  return {run, calls};
}

/** @param {ReturnType<typeof fakeTools>} tools @param {{failLabel?:string}} [options] */
function mockJitRun(tools, options = {}) {
  /** @type {{label:string,child:string[]}[]} */ const calls = [];
  /** @param {string} command @param {string[]} args @param {{label:string,env:NodeJS.ProcessEnv,cwd:string}} settings
   * @returns {import("node:child_process").SpawnSyncReturns<import("node:buffer").NonSharedBuffer>} */
  function run(command, args, settings) {
    const split = args.indexOf("--"), child = split < 0 ? args : args.slice(split + 1);
    calls.push({label: settings.label, child});
    assert.equal(command, tools.systemdRun); assert.ok(args.includes("--scope")); assert.ok(args.includes("--quiet"));
    assert.ok(args.includes("--property=MemoryMax=" + memoryMax)); assert.ok(args.includes("--property=MemorySwapMax=0"));
    assert.ok(!args.includes("--wait")); assert.equal(child[0], node); assert.equal(settings.env["MOM_AOT_BUILD_ONLY"], "1");
    if (settings.label === options.failLabel) {throw new Error(settings.label + " fixture failure");}
    if (settings.label === "Linux JIT resource tree producer") {
      assert.equal(child[2], "--local-jit-resources"); assert.equal(child[3], "build");
      const workspace = child[4] || "";
      fs.mkdirSync(path.join(workspace, "linux-jit-resources"), {recursive: true});
    }
    const empty = /** @type {import("node:buffer").NonSharedBuffer} */ (Buffer.alloc(0));
    return {pid: 1, output: [empty, empty, empty], stdout: empty, stderr: empty, status: 0, signal: null};
  }
  return {run, calls};
}

/** @param {ReturnType<typeof fakeTools>} tools */
function makeJitToolsExecutable(tools) {
  for (const file of [tools.cmake, tools.ninja, tools.cc, tools.cxx]) {
    fs.writeFileSync(file, "fixture tool\n"); fs.chmodSync(file, 0o755);
  }
}

/** @param {ReturnType<typeof fakeTools>} tools */
function jitTools(tools) {return {cmake: tools.cmake, ninja: tools.ninja, cc: tools.cc, cxx: tools.cxx};}

test("checks cgroup-v2 limits and maps the current process into its mounted scope", () => {
  const paths = cgroupMemoryPaths("0::/user.slice/user-1000.slice/app.scope\n",
    "31 22 0:30 / /sys/fs/cgroup rw,nosuid - cgroup2 cgroup2 rw\n");
  assert.deepEqual(paths, {memory: "/sys/fs/cgroup/user.slice/user-1000.slice/app.scope/memory.max",
    swap: "/sys/fs/cgroup/user.slice/user-1000.slice/app.scope/memory.swap.max"});
  assert.throws(() => cgroupMemoryPaths("0::/system.slice/worker.scope\n",
    "31 22 0:30 /user.slice /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n"), /outside the mounted/);
  assert.throws(() => cgroupMemoryPaths("0::/user.slice/../../system.slice\n",
    "31 22 0:30 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n"), /not canonical/);
  assert.doesNotThrow(() => verifyMemoryLimits(String(4 * 1024 ** 3), "0"));
  assert.throws(() => verifyMemoryLimits("max", "0"), /memory.max <= 4 GiB/);
  assert.throws(() => verifyMemoryLimits(String(4 * 1024 ** 3 + 1), "0"), /memory.max <= 4 GiB/);
  assert.throws(() => verifyMemoryLimits("2147483648", "1"), /memory.max <= 4 GiB/);
});

test("shares stable --workspace parsing across explicit follow-up commands", () => {
  assert.equal(workspaceOption(["--workspace", "/cache/native-sdk"], "support"), "/cache/native-sdk");
  assert.equal(workspaceOption(["--workspace", "/old", "--workspace", "/new"], "gnu-headers"), "/new");
  assert.throws(() => workspaceOption(["--workspace"], "compiler-inputs"), /Unknown or incomplete compiler-inputs option: --workspace/);
  assert.throws(() => workspaceOption(["--jobs", "2"], "jit-resources"), /Unknown or incomplete jit-resources option: --jobs/);
});

test.describe("local Linux source/LLVM bootstrap slice", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  test("full release children retain the proven aggregate scope instead of creating sibling units", () => {
    const order = /** @type {string[]} */ ([]), env = {MOM_LOCAL_NATIVE_SCOPE: "1"};
    const run = /** @type {typeof import('../scripts/build-local-native-sdk').runCommand} */ ((command, args, settings) => {
      order.push("run"); assert.equal(command, process.execPath); assert.deepEqual(args, ["--fixture"]);
      assert.equal(settings.env, env);
      return {pid: 1, output: [], stdout: Buffer.alloc(0), stderr: Buffer.alloc(0), status: 0, signal: null};
    });
    scopedInvoker({systemdRun: "/must-not-launch"}, env, run, () => {order.push("prove");})("fixture", process.execPath, ["--fixture"]);
    assert.deepEqual(order, ["prove", "run"]);
    assert.throws(() => scopedInvoker({systemdRun: "/must-not-launch"}, env, run, () => {throw new Error("unbounded parent");})(
      "fixture", process.execPath, ["--fixture"]), /unbounded parent/);
    assert.deepEqual(order, ["prove", "run"]);
  });
  test("ignores PATH shims for the root-owned systemd resource controller", t => {
    const f = fixture(t), shimDirectory = path.join(f.root, "path-shim"), shim = path.join(shimDirectory, "systemd-run");
    fs.mkdirSync(shimDirectory); fs.writeFileSync(shim, "#!/bin/sh\nexit 0\n"); fs.chmodSync(shim, 0o755);
    const originalPath = process.env["PATH"];
    process.env["PATH"] = shimDirectory;
    try {
      let selected;
      try {selected = trustedSystemdRun();} catch (error) {
        t.skip(error instanceof Error ? error.message : String(error)); return;
      }
      assert.equal(selected, "/usr/bin/systemd-run");
    } finally {
      if (originalPath === undefined) {delete process.env["PATH"];} else {process.env["PATH"] = originalPath;}
    }
  });

  test("parses the public configured receipt phase shape without inventing build evidence", t => {
    const f = fixture(t); fs.mkdirSync(f.llvmOutput);
    const file = path.join(f.llvmOutput, "configured.json"); writeJson(file, configuredReceipt(f.llvmOutput));
    assert.deepEqual(readConfigureState(file, f.llvmOutput), {phases: ["llvm-foundation", "tools"], prefix: path.join(f.llvmOutput, "sdk")});
    const bad = configuredReceipt(f.llvmOutput), second = bad.plan.phases[1]; assert.ok(second);
    second.name = "llvm-foundation"; writeJson(file, bad);
    assert.throws(() => readConfigureState(file, f.llvmOutput), /generated build-phase list/);
    fs.unlinkSync(file);
    assert.throws(() => readConfigureState(file, f.llvmOutput), /partial output is preserved/);
  });

  test("continues from a valid configured tree through public phase, install and audit commands", t => {
    const f = fixture(t), tools = fakeTools(f.root); fs.mkdirSync(f.llvmOutput);
    writeJson(path.join(f.llvmOutput, "configured.json"), configuredReceipt(f.llvmOutput));
    fs.writeFileSync(path.join(f.llvmOutput, "phase-llvm-foundation.json"), "admitted earlier phase fixture\n");
    const sentinel = path.join(f.llvmOutput, "unknown-preserved.log"); fs.writeFileSync(sentinel, "preserve stale/unrecognized data\n");
    const mocked = mockRun(tools);
    const result = buildLocalNativeSdk({workspace: f.workspace}, {tools, run: mocked.run, environment: {PATH: "/usr/bin"}});
    assert.equal(result.llvmResumed, true); assert.equal(result.llvmReused, false);
    assert.deepEqual(mocked.calls.filter(row => row.label.startsWith("LLVM 21 build phase ")).map(row => row.label), ["LLVM 21 build phase tools"]);
    assert.equal(mocked.calls.at(-1)?.label, "LLVM 21 source SDK re-admission");
    assert.equal(fs.readFileSync(path.join(f.llvmOutput, "phase-llvm-foundation.json"), "utf8"), "admitted earlier phase fixture\n");
    assert.equal(fs.readFileSync(sentinel, "utf8"), "preserve stale/unrecognized data\n");
  });

  test("rejects compiler source-root symlink substitution and mismatched receipt roots", t => {
    const linked = fixture(t), linkedTools = fakeTools(linked.root), linkedRun = mockRun(linkedTools), component = "dpcpp",
      source = path.join(linked.compilerSources, component), external = path.join(linked.root, "external-dpcpp");
    fs.mkdirSync(external); fs.rmSync(source, {recursive: true, force: true}); fs.symlinkSync(external, source, "dir");
    assert.throws(() => buildLocalNativeSdk({workspace: linked.workspace}, {tools: linkedTools, run: linkedRun.run,
      environment: {PATH: "/usr/bin"}}), /expected real directory: dpcpp/);

    const mismatched = fixture(t), tools = fakeTools(mismatched.root), run = mockRun(tools), sourceReceipt = path.join(mismatched.compilerSources, "sources.json"),
      receipt = JSON.parse(fs.readFileSync(sourceReceipt, "utf8"));
    receipt.prepared[component].root = external; writeJson(sourceReceipt, receipt);
    assert.throws(() => buildLocalNativeSdk({workspace: mismatched.workspace}, {tools, run: run.run,
      environment: {PATH: "/usr/bin"}}), /expected real directory: dpcpp/);
  });

  test("rechecks compiler source roots after the capped inventory admission", t => {
    const f = fixture(t), tools = fakeTools(f.root), component = "dpcpp",
      source = path.join(f.compilerSources, component), external = path.join(f.root, "external-dpcpp");
    fs.mkdirSync(external);
    const mocked = mockRun(tools, {afterRun(label) {
      if (label === "compiler source re-admission") {fs.rmSync(source, {recursive: true, force: true}); fs.symlinkSync(external, source, "dir");}
    }});
    assert.throws(() => buildLocalNativeSdk({workspace: f.workspace}, {tools, run: mocked.run, environment: {PATH: "/usr/bin"}}),
      /expected real directory: dpcpp/);
    assert.deepEqual(mocked.calls.map(row => row.label), ["systemd memory-limit preflight", "compiler source re-admission"]);
  });

  test("rejects out-of-order interrupted phase state before starting any remaining build", t => {
    const f = fixture(t), tools = fakeTools(f.root); fs.mkdirSync(f.llvmOutput);
    writeJson(path.join(f.llvmOutput, "configured.json"), configuredReceipt(f.llvmOutput));
    const later = path.join(f.llvmOutput, "phase-tools.json"); fs.writeFileSync(later, "preserve out-of-order receipt\n");
    const mocked = mockRun(tools);
    assert.throws(() => buildLocalNativeSdk({workspace: f.workspace}, {tools, run: mocked.run,
      environment: {PATH: "/usr/bin"}}), /not a contiguous prefix/);
    assert.deepEqual(mocked.calls.filter(row => row.label.startsWith("LLVM 21 build phase ")), []);
    assert.equal(fs.readFileSync(later, "utf8"), "preserve out-of-order receipt\n");
  });

  test("runs a fresh slice with the icpx alias intact and admits its completed output", t => {
    const f = fixture(t), tools = fakeTools(f.root), mocked = mockRun(tools);
    const result = buildLocalNativeSdk({workspace: f.workspace, jobs: 1}, {tools, run: mocked.run, environment: {PATH: "/usr/bin"}});
    const configure = mocked.calls.find(row => row.label === "LLVM 21 configure");
    assert.ok(configure);
    const child = configure.args.slice(configure.args.indexOf("--") + 1);
    assert.equal(child[8], tools.cxx); assert.equal(path.basename(child[8] || ""), "icpx");
    assert.equal(result.llvmResumed, false); assert.equal(result.llvmReused, false);
    assert.deepEqual(mocked.calls.filter(row => row.label.startsWith("LLVM 21 build phase ")).map(row => row.label),
      ["LLVM 21 build phase llvm-foundation", "LLVM 21 build phase tools"]);
  });

  test("reuses an admitted LLVM output and stops immediately on a failed source fetch", t => {
    const reused = fixture(t), tools = fakeTools(reused.root), mocked = mockRun(tools);
    fs.mkdirSync(reused.llvmOutput); fs.writeFileSync(path.join(reused.llvmOutput, "result.json"), "admitted output fixture\n");
    const result = buildLocalNativeSdk({workspace: reused.workspace}, {tools, run: mocked.run,
      environment: {PATH: "/usr/bin"}});
    assert.equal(result.llvmReused, true);
    assert.deepEqual(mocked.calls.map(row => row.label), ["systemd memory-limit preflight", "compiler source re-admission",
      "LLVM 21 source SDK re-admission"]);

    const failed = fixture(t), failedTools = fakeTools(failed.root), stopped = mockRun(failedTools, {failLabel: "pinned compiler source fetch"});
    fs.rmSync(failed.compilerSources, {recursive: true, force: true}); fs.rmSync(failed.buildInputs, {recursive: true, force: true});
    assert.throws(() => buildLocalNativeSdk({workspace: failed.workspace}, {tools: failedTools, run: stopped.run,
      environment: {PATH: "/usr/bin"}}), /pinned compiler source fetch fixture failure/);
    assert.deepEqual(stopped.calls.map(row => row.label), ["systemd memory-limit preflight", "pinned compiler source fetch"]);
    assert.equal(fs.existsSync(path.join(failed.workspace, "llvm-inputs")), false);
  });

  test("source and LLVM admission failures stay capped and stop before downstream writes", t => {
    const source = fixture(t), sourceTools = fakeTools(source.root), sourceRun = mockRun(sourceTools,
      {failLabel: "compiler source re-admission"});
    assert.throws(() => buildLocalNativeSdk({workspace: source.workspace}, {tools: sourceTools, run: sourceRun.run,
      environment: {PATH: "/usr/bin"}}), /compiler source re-admission fixture failure/);
    assert.deepEqual(sourceRun.calls.map(row => row.label), ["systemd memory-limit preflight", "compiler source re-admission"]);
    const sourceAudit = sourceRun.calls[1]; assert.ok(sourceAudit);
    assert.deepEqual(sourceAudit.args.slice(sourceAudit.args.indexOf("--") + 1),
      [node, path.join(repo, "scripts/build-local-native-sdk.js"), "--audit-support", "compiler-sources", source.workspace]);
    assert.equal(fs.existsSync(source.llvmOutput), false, "LLVM output is not created after source admission fails");

    const llvm = fixture(t), llvmTools = fakeTools(llvm.root), llvmOutput = path.join(llvm.llvmOutput, "result.json");
    fs.mkdirSync(llvm.llvmOutput); fs.writeFileSync(llvmOutput, "preserve admitted LLVM receipt\n");
    const before = fs.readdirSync(llvm.workspace).sort(), llvmRun = mockRun(llvmTools, {failLabel: "LLVM 21 source SDK re-admission"});
    assert.throws(() => buildLocalNativeSdk({workspace: llvm.workspace}, {tools: llvmTools, run: llvmRun.run,
      environment: {PATH: "/usr/bin"}}), /LLVM 21 source SDK re-admission fixture failure/);
    assert.deepEqual(llvmRun.calls.map(row => row.label), ["systemd memory-limit preflight", "compiler source re-admission",
      "LLVM 21 source SDK re-admission"]);
    const llvmAudit = llvmRun.calls[2]; assert.ok(llvmAudit);
    assert.deepEqual(llvmAudit.args.slice(llvmAudit.args.indexOf("--") + 1),
      [node, path.join(repo, "scripts/build-local-native-sdk.js"), "--audit-support", "llvm-sdk", llvm.workspace]);
    assert.equal(fs.readFileSync(llvmOutput, "utf8"), "preserve admitted LLVM receipt\n");
    assert.deepEqual(fs.readdirSync(llvm.workspace).sort(), before, "failed re-admission creates no downstream files");
  });

  test("a failed resource-limit preflight stops before source acquisition", t => {
    const f = fixture(t), tools = fakeTools(f.root), stopped = mockRun(tools, {failLabel: "systemd memory-limit preflight"});
    assert.throws(() => buildLocalNativeSdk({workspace: f.workspace}, {tools, run: stopped.run,
      environment: {PATH: "/usr/bin"}}), /preflight fixture failure/);
    assert.deepEqual(stopped.calls.map(row => row.label), ["systemd memory-limit preflight"]);
    assert.equal(fs.existsSync(path.join(f.workspace, "compiler-fetch")), false);
    assert.equal(fs.existsSync(f.llvmOutput), false);
  });

  test("the real preflight observes bounded cgroup-v2 memory and propagates scope exit status", t => {
    const systemdRun = "/usr/bin/systemd-run",
      options = /** @type {import("node:child_process").SpawnSyncOptionsWithStringEncoding} */ ({cwd: repo, encoding: "utf8", timeout: 30000});
    const probe = spawnSync(systemdRun, ["--user", "--scope", "--quiet", "--property=MemoryMax=2G", "--property=MemorySwapMax=0", "--",
      node, path.join(repo, "scripts/build-local-native-sdk.js"), "--memory-probe"], options);
    const probeError = /** @type {NodeJS.ErrnoException | undefined} */ (probe.error);
    if (probeError?.code === "ENOENT" || /Failed to connect to bus|not been booted with systemd/i.test(probe.stderr || "")) {
      t.skip("a user systemd manager is unavailable: " + (probe.error?.message || probe.stderr || "no exit status")); return;
    }
    assert.equal(probe.status, 0, probe.stderr || "memory probe did not verify its own systemd cgroup-v2 limits");
    assert.match(probe.stdout || "", /\[native-sdk\] cgroup-v2 memory\.max=2147483648 memory\.swap\.max=0/);
    const failed = spawnSync(systemdRun, ["--user", "--scope", "--quiet", "--property=MemoryMax=2G", "--property=MemorySwapMax=0", "--",
      node, "-e", "process.exit(23)"], options);
    assert.equal(failed.status, 23, failed.stderr || "systemd scope did not propagate the child status");
  });
});

test.describe("local Linux static support SDK bootstrap slice", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  /** @param {ReturnType<typeof fixture>} f */
  function context(f) {
    fs.mkdirSync(f.llvmOutput); writeJson(path.join(f.llvmOutput, "result.json"), {version: 1, role: "immutable LLVM fixture"});
    fs.writeFileSync(path.join(f.llvmOutput, "sentinel.txt"), "completed LLVM state remains immutable\n");
    return {workspace: f.workspace, compilerFetch: path.join(f.workspace, "compiler-fetch"), compilerSources: f.compilerSources,
      buildInputs: f.buildInputs, llvmSdk: f.llvmOutput};
  }

  /** @param {ReturnType<typeof fakeTools>} runner @param {ReturnType<typeof fakeSupportTools>} support
   * @param {ReturnType<typeof mockSupportRun>} mocked @param {string[]} audits */
  function dependencies(runner, support, mocked, audits) {
    return {tools: runner, supportTools: support, run: mocked.run, environment: {PATH: "/usr/bin"},
      /** @param {unknown} _paths @param {unknown} _tools */ verifyIo(_paths, _tools) {audits.push("io-sdk");},
      /** @param {unknown} _paths @param {unknown} _tools */ verifyOpenSsl(_paths, _tools) {audits.push("openssl-sdk");},
      /** @param {string} kind */ auditSupport(kind) {audits.push(kind);}};
  }

  test("uses maintained CI commands/configs inside bounded scopes and keeps the admitted LLVM result immutable", t => {
    const f = fixture(t), ctx = context(f), runner = fakeTools(f.root), support = fakeSupportTools(f.root, runner), mocked = mockSupportRun(runner);
    /** @type {string[]} */ const audits = [];
    const result = buildLocalNativeSupportSdk(ctx, dependencies(runner, support, mocked, audits)), paths = supportPaths(f.workspace);
    assert.deepEqual(result.reused, {hermes: false, io: false, openssl: false, zstd: false, openmp: false});
    const call = (/** @type {string} */ label) => mocked.calls.find(row => row.label === label)?.child;
    const hermes = ciSupport.hermesBuildCommand(/** @type {Parameters<typeof ciSupport.hermesBuildCommand>[0]} */ (/** @type {unknown} */
        ({source: paths.hermesSource, sdkRoot: paths.hermesSdk})), support.hermes, "2"),
      hermesCall = call("static Hermes SDK producer");
    assert.equal(hermesCall?.[1], path.join(repo, "scripts", hermes.script)); assert.deepEqual(hermesCall?.slice(2), hermes.args);
    const ioPaths = /** @type {Parameters<typeof ciSupport.ioOpenSslBuildCommands>[0]} */ (/** @type {unknown} */
        ({env: {}, source: f.buildInputs, ioSource: paths.ioSource, ioSdk: paths.ioSdk,
          opensslSdk: paths.opensslSdk, opensslTools: paths.opensslTools})),
      ioCommands = ciSupport.ioOpenSslBuildCommands(ioPaths, support.cmake, support.ninja, support.openssl, "2"),
      ioCall = call("static I/O SDK producer"), opensslCall = call("static OpenSSL SDK producer");
    assert.equal(ioCall?.[1], path.join(repo, "scripts", ioCommands[0]?.script || "")); assert.deepEqual(ioCall?.slice(2), ioCommands[0]?.args);
    assert.equal(opensslCall?.[1], path.join(repo, "scripts", ioCommands[1]?.script || "")); assert.deepEqual(opensslCall?.slice(2), ioCommands[1]?.args);
    assert.deepEqual(JSON.parse(fs.readFileSync(paths.opensslTools, "utf8")), {version: 1, jobs: 2, tools: support.openssl});
    const zstdState = /** @type {Parameters<typeof ciSupport.zstdBuildCommand>[0]} */ ({source: paths.zstdSource,
        archive: path.join(paths.zstdSource, "zstd-1.5.7.tar.gz"), sdkRoot: paths.zstdSdk, toolsFile: paths.zstdTools}),
      zstd = ciSupport.zstdBuildCommand(zstdState, paths.zstdTools), zstdCall = call("static zstd SDK producer");
    assert.equal(zstdCall?.[1], path.join(repo, "scripts", zstd.script)); assert.deepEqual(zstdCall?.slice(2), zstd.args);
    assert.deepEqual(JSON.parse(fs.readFileSync(paths.zstdTools, "utf8")), {version: 1, jobs: 4, tools: support.zstd});
    const openmpState = /** @type {Parameters<typeof ciSupport.openmpBuildCommand>[0]} */ ({source: f.buildInputs,
        archive: path.join(f.buildInputs, archiveName), sdkRoot: paths.openmpSdk, toolsFile: paths.openmpTools}),
      openmp = ciSupport.openmpBuildCommand(openmpState), openmpCall = call("static OpenMP SDK producer");
    assert.equal(openmpCall?.[1], path.join(repo, "scripts", openmp.script)); assert.deepEqual(openmpCall?.slice(2), openmp.args);
    assert.deepEqual(JSON.parse(fs.readFileSync(paths.openmpTools, "utf8")), {version: 1, jobs: 2, tools: support.openmp});
    assert.ok(mocked.calls.every(row => row.child[0] === node));
    assert.ok(mocked.calls.some(row => row.label === "LLVM 21 source SDK re-admission" && row.child[2] === "--audit-support" && row.child[3] === "llvm-sdk"));
    assert.ok(mocked.calls.some(row => row.label === "pinned Linux source preparation and re-admission" &&
      row.child[1] === path.join(repo, "scripts", "build-local-native-sdk.js") && row.child[2] === "--prepare-linux-sources" && row.child[3] === f.workspace));
    assert.equal(fs.readFileSync(path.join(f.llvmOutput, "sentinel.txt"), "utf8"), "completed LLVM state remains immutable\n");
    assert.deepEqual(audits, ["hermes-source", "hermes-sdk", "io-sdk", "openssl-sdk", "io-openssl", "zstd-sdk", "openmp-sdk"]);
  });

  test("prepares Linux sources from the exact compiler-root subset and reuses only admitted real roots", t => {
    const f = fixture(t), paths = supportPaths(f.workspace), roots = /** @type {Record<string, string>} */ ({}), fetch = paths.compilerFetch;
    fs.mkdirSync(path.join(fetch, "roots"), {recursive: true});
    for (const name of Object.keys(sdkComponents("compiler"))) {
      const root = path.join(fetch, "roots", name); fs.mkdirSync(root); roots[name] = root;
    }
    writeJson(path.join(fetch, "roots.json"), roots); writeJson(path.join(fetch, "fetch.json"), {fixture: true});
    let prepared = 0, verified = 0;
    const deps = {assertFetch: (/** @type {unknown} */ receipt, /** @type {Record<string, string>} */ selected,
      /** @type {string} */ target, /** @type {string} */ source) => {
      assert.deepEqual(receipt, {fixture: true}); assert.deepEqual(selected, roots); assert.equal(target, "compiler"); assert.equal(source, fetch);
    }, verifyCompiler: (/** @type {string} */ file, /** @type {string} */ output) => {
      assert.equal(file, path.join(paths.compilerSources, "sources.json")); assert.equal(output, paths.compilerSources);
    }, prepare: (/** @type {string} */ target, /** @type {Record<string, string>} */ selected,
      /** @type {string} */ output, /** @type {NodeJS.ProcessEnv} */ env) => {
      prepared++; assert.equal(target, "linux"); assert.deepEqual(Object.keys(selected).sort(), Object.keys(sdkComponents("linux")).sort());
      assert.equal(env["MOM_AOT_BUILD_ONLY"], "1"); fs.mkdirSync(output);
      const components = Object.fromEntries(Object.keys(sdkComponents("linux")).map(name => {
        const root = path.join(output, name); fs.mkdirSync(root); return [name, {root}];
      }));
      writeJson(path.join(output, "sources.json"), {version: 2, target: "linux", status: "PASS", prepared: components});
    }, verifyLinux: (/** @type {{env: NodeJS.ProcessEnv}} */ pathsArg, /** @type {{source: string}} */ state,
      /** @type {string} */ git) => {
      verified++; assert.equal(pathsArg.env["MOM_AOT_BUILD_ONLY"], "1"); assert.equal(state.source, fetch); assert.ok(path.isAbsolute(git));
    }};
    assert.equal(prepareOrAdmitLinuxSources(f.workspace, deps), paths.linuxSources);
    const sentinel = path.join(paths.linuxSources, "unknown-preserved.log"); fs.writeFileSync(sentinel, "preserve\n");
    assert.equal(prepareOrAdmitLinuxSources(f.workspace, deps), paths.linuxSources);
    assert.equal(prepared, 1); assert.equal(verified, 2); assert.equal(fs.readFileSync(sentinel, "utf8"), "preserve\n");

    const sourceRoot = path.join(paths.linuxSources, "dpcpp"), external = path.join(f.root, "outside-dpcpp");
    fs.mkdirSync(external); fs.rmdirSync(sourceRoot); fs.symlinkSync(external, sourceRoot, "dir");
    assert.throws(() => prepareOrAdmitLinuxSources(f.workspace, deps), /expected real directory: dpcpp/);
    assert.equal(prepared, 1); assert.equal(verified, 2);
  });

  test("Windows source preparation reuses the compiler cohort without restamping receipts or Linux patches", t => {
    const f = fixture(t), paths = supportPaths(f.workspace), roots = Object.fromEntries(Object.keys(sdkComponents("compiler"))
      .map(name => [name, path.join(paths.compilerFetch, "roots", name)]));
    fs.mkdirSync(paths.compilerFetch);
    writeJson(path.join(paths.compilerFetch, "roots.json"), roots);
    writeJson(path.join(paths.compilerFetch, "fetch.json"), {unchanged: "fetch"});
    let prepares = 0, admissions = 0, verifies = 0;
    const deps = {
      admitLinux(/** @type {string} */ workspace) {assert.equal(workspace, f.workspace); admissions++; return paths.linuxSources;},
      prepare(/** @type {string} */ target, /** @type {Record<string,string>} */ selected,
        /** @type {string} */ output, /** @type {NodeJS.ProcessEnv} */ env) {
        prepares++; assert.equal(target, "windows"); assert.equal(env["MOM_AOT_BUILD_ONLY"], "1");
        assert.equal(output, path.join(f.workspace, "windows-sources"));
        assert.deepEqual(selected, Object.fromEntries(Object.keys(sdkComponents("windows")).map(name => [name, roots[name]])));
        fs.mkdirSync(output);
        const prepared = Object.fromEntries(Object.keys(selected).map(name => {
          const root = path.join(output, name); fs.mkdirSync(root); return [name, {root}];
        }));
        writeJson(path.join(output, "sources.json"), {version: 2, target, status: "PASS", prepared});
        return {};
      },
      verify(/** @type {string} */ file, /** @type {string} */ target) {
        verifies++; assert.equal(file, path.join(f.workspace, "windows-sources/sources.json")); assert.equal(target, "windows"); return {};
      },
    };
    const injected = /** @type {any} */ ({...deps}), first = prepareOrAdmitWindowsSources(f.workspace, injected);
    assert.deepEqual(prepareOrAdmitWindowsSources(f.workspace, injected), first);
    assert.equal(prepares, 1); assert.equal(admissions, 4); assert.equal(verifies, 4);
    assert.equal(first.receiptSha256, require("../scripts/build-source-inputs").fileHash(first.receipt));
    injected.verify = () => {throw new Error("changed Windows patch receipt");};
    assert.throws(() => prepareOrAdmitWindowsSources(f.workspace, injected), /changed Windows patch receipt/);
    injected.verify = deps.verify;
    injected.admitLinux = () => {
      if (++admissions % 2 === 1) {writeJson(path.join(paths.compilerFetch, "fetch.json"), {changed: true});}
      return paths.linuxSources;
    };
    assert.throws(() => prepareOrAdmitWindowsSources(f.workspace, injected), /Original compiler source receipt changed/);
    assert.equal(prepares, 1);
  });

  test("re-admits stable caches without rerunning completed support producers", t => {
    const f = fixture(t), ctx = context(f), runner = fakeTools(f.root), support = fakeSupportTools(f.root, runner), first = mockSupportRun(runner);
    /** @type {string[]} */ const audits = [];
    buildLocalNativeSupportSdk(ctx, dependencies(runner, support, first, audits));
    const second = mockSupportRun(runner), result = buildLocalNativeSupportSdk(ctx, dependencies(runner, support, second, audits));
    assert.deepEqual(result.reused, {hermes: true, io: true, openssl: true, zstd: true, openmp: true});
    assert.deepEqual(second.calls.filter(row => row.label.endsWith("SDK producer")), []);
    assert.equal(second.calls.some(row => row.label.includes("source fetch")), false);
    assert.deepEqual(second.calls.filter(row => row.label === "LLVM 21 source SDK re-admission").length, 1);
  });

  test("memory preflight and first fetch failures stop the support slice before later inputs", t => {
    const f = fixture(t), ctx = context(f), runner = fakeTools(f.root), support = fakeSupportTools(f.root, runner), stopped = mockSupportRun(runner, {failLabel: "systemd memory-limit preflight"});
    assert.throws(() => buildLocalNativeSupportSdk(ctx, dependencies(runner, support, stopped, [])), /preflight fixture failure/);
    assert.deepEqual(stopped.calls.map(row => row.label), ["systemd memory-limit preflight"]);
    assert.equal(fs.existsSync(supportPaths(f.workspace).hermesSource), false);

    const failed = fixture(t), failedContext = context(failed), failedRunner = fakeTools(failed.root), failedSupport = fakeSupportTools(failed.root, failedRunner),
      fetchStopped = mockSupportRun(failedRunner, {failLabel: "pinned Hermes source fetch"});
    assert.throws(() => buildLocalNativeSupportSdk(failedContext, dependencies(failedRunner, failedSupport, fetchStopped, [])), /Hermes source fetch fixture failure/);
    assert.deepEqual(fetchStopped.calls.map(row => row.label), ["systemd memory-limit preflight", "offline LLVM/OpenSSL source archive verification",
      "LLVM 21 source SDK re-admission", "pinned Linux source preparation and re-admission", "pinned Hermes source fetch"]);
    assert.equal(fs.existsSync(supportPaths(failed.workspace).ioSource), false);
  });

  test("preserves interrupted SDK outputs and rejects a mismatched source-LLVM context", t => {
    const f = fixture(t), ctx = context(f), runner = fakeTools(f.root), support = fakeSupportTools(f.root, runner), mocked = mockSupportRun(runner), stale = supportPaths(f.workspace),
      sentinel = path.join(stale.opensslSdk, "unknown-partial-state.log");
    fs.mkdirSync(stale.opensslSdk); fs.writeFileSync(sentinel, "do not overwrite or delete\n");
    assert.throws(() => buildLocalNativeSupportSdk(ctx, dependencies(runner, support, mocked, [])), /without its original tool descriptor/);
    assert.equal(fs.readFileSync(sentinel, "utf8"), "do not overwrite or delete\n");
    assert.equal(mocked.calls.some(row => row.label === "static OpenSSL SDK producer"), false);

    const malformed = fixture(t), badContext = context(malformed), badRunner = fakeTools(malformed.root), badSupport = fakeSupportTools(malformed.root, badRunner), noRun = mockSupportRun(badRunner);
    badContext.llvmSdk = path.join(malformed.root, "elsewhere");
    assert.throws(() => buildLocalNativeSupportSdk(badContext, dependencies(badRunner, badSupport, noRun, [])), /workspace's admitted compiler sources, LLVM inputs and SDK paths/);
    assert.deepEqual(noRun.calls, []);
  });
});

test.describe("local Linux DPC++/CUDA compiler-input bootstrap slice", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  /** @param {ReturnType<typeof fixture>} f */
  function context(f) {
    fs.mkdirSync(f.llvmOutput); writeJson(path.join(f.llvmOutput, "result.json"), {version: 1, role: "immutable LLVM fixture"});
    const paths = supportPaths(f.workspace);
    return {workspace: f.workspace, compilerFetch: paths.compilerFetch, compilerSources: paths.compilerSources,
      buildInputs: paths.buildInputs, llvmSdk: paths.llvmSdk};
  }

  test("compiler-inputs runs only scoped public verification and the explicit local source/staging child", t => {
    const f = fixture(t), ctx = context(f), runner = fakeTools(f.root), mocked = mockRun(runner), roots = {
      dpcpp: path.join(f.root, "installed-dpcpp"), cuda: path.join(f.root, "installed-cuda")};
    fs.mkdirSync(roots.dpcpp); fs.mkdirSync(roots.cuda);
    const result = buildLocalCompilerInputs(ctx, {tools: runner, run: mocked.run, environment: {PATH: "/usr/bin"}, installedRoots: roots});
    assert.deepEqual(result, {workspace: f.workspace, dpcppRoot: path.join(f.workspace, "dpcpp-cuda-inputs/dpcpp"),
      cudaRoot: path.join(f.workspace, "dpcpp-cuda-inputs/cuda"), manifest: path.join(f.workspace, "dpcpp-cuda-inputs/inputs.json"), reused: false});
    assert.deepEqual(mocked.calls.map(row => row.label), ["systemd memory-limit preflight", "offline LLVM/OpenSSL source archive verification",
      "LLVM 21 source SDK re-admission", "local DPC++/CUDA input staging and source re-admission"]);
    const lastCall = mocked.calls[mocked.calls.length - 1]; assert.ok(lastCall);
    const split = lastCall.args.indexOf("--"); assert.ok(split >= 0);
    const staged = lastCall.args.slice(split + 1);
    assert.deepEqual(staged, [node, path.join(repo, "scripts/build-local-native-sdk.js"), "--stage-local-compiler-inputs", f.workspace]);
  });

  test("internal source preparation, staging, and admission require their own active memory scope", t => {
    const f = fixture(t), paths = supportPaths(f.workspace), failScope = () => {throw new Error("scope limit missing");};
    assert.throws(() => main(["--prepare-linux-sources", f.workspace], failScope), /scope limit missing/);
    assert.throws(() => main(["--stage-local-compiler-inputs", f.workspace], failScope), /scope limit missing/);
    assert.throws(() => main(["--audit-support", "compiler-sources", f.workspace], failScope), /scope limit missing/);
    assert.equal(fs.existsSync(paths.linuxSources), false, "source prep does not start before the cap proof");
    assert.equal(fs.existsSync(paths.dpcppCudaRoot), false, "DPC++/CUDA staging does not start before the cap proof");
  });

  test("source receipts are verified before/after staging and reused inputs are read-only re-admitted", t => {
    const f = fixture(t), paths = supportPaths(f.workspace), roots = {dpcpp: path.join(f.root, "installed-dpcpp"), cuda: path.join(f.root, "installed-cuda")};
    fs.mkdirSync(roots.dpcpp); fs.mkdirSync(roots.cuda);
    let prepared = 0, verified = 0, staged = 0, audits = 0;
    const deps = /** @type {Parameters<typeof stageLocalCompilerInputs>[1]} */ ({installedRoots: roots,
      prepareSources(workspace) {
        assert.equal(workspace, f.workspace); prepared++;
        fs.mkdirSync(paths.compilerFetch, {recursive: true});
        fs.mkdirSync(paths.linuxSources, {recursive: true});
        writeJson(path.join(paths.compilerFetch, "fetch.json"), {fixture: "fetch"});
        writeJson(path.join(paths.linuxSources, "sources.json"), {fixture: "linux"});
        writeJson(path.join(paths.compilerSources, "sources.json"), {fixture: "compiler"});
      },
      verifyLinux(_paths, state, git) {
        verified++; assert.equal(state.source, paths.compilerFetch); assert.ok(path.isAbsolute(git));
        assert.equal(state.linuxPrepared, paths.linuxSources); assert.equal(state.compilerPrepared, paths.compilerSources);
      },
      stageInputs(state, installed, origin, env) {
        staged++; assert.deepEqual(installed, roots); assert.equal(origin.kind, "prepared-receipts");
        assert.equal(origin.workspaceRoot, f.workspace); assert.equal(env["MOM_AOT_BUILD_ONLY"], "1");
        assert.equal(Object.hasOwn(origin, "tarSha256"), false);
        fs.mkdirSync(state.root); writeJson(state.manifest, {fixture: "admitted"});
      },
      auditInputs(state, record) {audits++; assert.equal(record.fixture, "admitted"); assert.equal(state.root, paths.dpcppCudaRoot);}});
    const first = stageLocalCompilerInputs(f.workspace, deps), second = stageLocalCompilerInputs(f.workspace, deps);
    assert.equal(first.reused, false); assert.equal(second.reused, true);
    assert.equal(prepared, 2); assert.equal(verified, 4); assert.equal(staged, 1); assert.equal(audits, 3);
  });

  test("missing installed inputs produce an explicit installer command without running producers", t => {
    const f = fixture(t), ctx = context(f), runner = fakeTools(f.root), mocked = mockRun(runner), roots = {
      dpcpp: path.join(f.root, "missing-dpcpp"), cuda: path.join(f.root, "missing-cuda")};
    assert.throws(() => buildLocalCompilerInputs(ctx, {tools: runner, run: mocked.run, environment: {PATH: "/usr/bin"}, installedRoots: roots}),
      /install them explicitly with scripts\/install-dev\.sh --component dpcpp,cuda --jobs 4 --keep-workspace/);
    assert.deepEqual(mocked.calls, []);
    assert.throws(() => localDpcppCudaRoots(roots), /install them explicitly/);
  });
});

test.describe("local Linux GNU target-header bootstrap slice", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  /** @param {ReturnType<typeof fixture>} f */
  function context(f) {
    fs.mkdirSync(f.llvmOutput); writeJson(path.join(f.llvmOutput, "result.json"), {version: 1, role: "immutable LLVM fixture"});
    const paths = supportPaths(f.workspace);
    return {workspace: f.workspace, compilerFetch: paths.compilerFetch, compilerSources: paths.compilerSources,
      buildInputs: paths.buildInputs, llvmSdk: paths.llvmSdk};
  }

  test("runs the shared Linux GNU build/audit CLI inside the bounded scope and reuses by audit", t => {
    const f = fixture(t), ctx = context(f), runner = fakeTools(f.root), mocked = mockRun(runner), paths = supportPaths(f.workspace);
    const fresh = buildLocalGnuHeaders(ctx, {tools: runner, run: mocked.run, environment: {PATH: "/usr/bin"}});
    assert.deepEqual(fresh, {workspace: f.workspace, root: paths.gnuHeadersRoot, target: "x86_64-unknown-linux-gnu", reused: false});
    const first = mocked.calls[mocked.calls.length - 1]; assert.ok(first);
    const firstChild = first.args.slice(first.args.indexOf("--") + 1);
    assert.deepEqual(firstChild.slice(1), [path.join(repo, "scripts/prepare-native-gnu-headers.js"), "build", paths.gnuHeadersRoot, "x86_64-unknown-linux-gnu"]);

    fs.mkdirSync(paths.gnuHeadersRoot);
    const rechecked = mockRun(runner), reused = buildLocalGnuHeaders(ctx, {tools: runner, run: rechecked.run, environment: {PATH: "/usr/bin"}}),
      last = rechecked.calls[rechecked.calls.length - 1];
    assert.equal(reused.reused, true); assert.ok(last);
    assert.deepEqual(last.args.slice(last.args.indexOf("--") + 1).slice(1),
      [path.join(repo, "scripts/prepare-native-gnu-headers.js"), "audit", paths.gnuHeadersRoot, "x86_64-unknown-linux-gnu"]);
  });

  test("preserves invalid header state and stops immediately when the shared producer fails", t => {
    const f = fixture(t), ctx = context(f), paths = supportPaths(f.workspace), runner = fakeTools(f.root), failed = mockRun(runner,
      {failLabel: "GNU target-header handoff producer"});
    assert.throws(() => buildLocalGnuHeaders(ctx, {tools: runner, run: failed.run, environment: {PATH: "/usr/bin"}}), /producer fixture failure/);
    assert.deepEqual(failed.calls.map(row => row.label), ["systemd memory-limit preflight", "LLVM 21 source SDK re-admission", "GNU target-header handoff producer"]);

    fs.mkdirSync(paths.gnuHeadersRoot); const sentinel = path.join(paths.gnuHeadersRoot, "unknown-partial-state.log");
    fs.writeFileSync(sentinel, "preserve unknown state\n");
    const noRun = mockRun(runner, {failLabel: "GNU target-header handoff re-admission"});
    assert.throws(() => buildLocalGnuHeaders(ctx, {tools: runner, run: noRun.run, environment: {PATH: "/usr/bin"}}), /fixture failure/);
    assert.equal(fs.readFileSync(sentinel, "utf8"), "preserve unknown state\n");
  });
});

test.describe("local Linux JIT resource bootstrap slice", {skip: process.platform !== "linux" || process.arch !== "x64"}, () => {
  /** @param {ReturnType<typeof fixture>} f */
  function context(f) {
    fs.mkdirSync(f.llvmOutput); writeJson(path.join(f.llvmOutput, "result.json"), {version: 1, role: "immutable LLVM fixture"});
    const paths = supportPaths(f.workspace);
    return {workspace: f.workspace, compilerFetch: paths.compilerFetch, compilerSources: paths.compilerSources,
      buildInputs: paths.buildInputs, llvmSdk: paths.llvmSdk};
  }

  test("routes a fresh tree through the shared builder inside a bounded scope", t => {
    const f = fixture(t), ctx = context(f), paths = supportPaths(f.workspace), tools = fakeTools(f.root), mocked = mockJitRun(tools);
    makeJitToolsExecutable(tools);
    const options = {tools, run: mocked.run, environment: {PATH: "/usr/bin"},
      prepareInputs(/** @type {any} */ selected) {
        assert.equal(selected.workspace, f.workspace);
        return {workspace: f.workspace, dpcppRoot: path.join(paths.dpcppCudaRoot, "dpcpp"), cudaRoot: path.join(paths.dpcppCudaRoot, "cuda"),
          manifest: path.join(paths.dpcppCudaRoot, "inputs.json")};
      },
      prepareHeaders(/** @type {any} */ selected) {
        assert.equal(selected.workspace, f.workspace);
        return {workspace: f.workspace, root: paths.gnuHeadersRoot, target: "x86_64-unknown-linux-gnu"};
      }};
    const result = buildLocalLinuxJitResources(ctx, options);
    assert.deepEqual(result, {workspace: f.workspace, root: paths.linuxJitResources, target: "x86_64-unknown-linux-gnu", reused: false});
    assert.deepEqual(mocked.calls.map(row => row.label), ["systemd memory-limit preflight", "LLVM 21 source SDK re-admission", "Linux JIT resource tree producer"]);
    const child = mocked.calls.at(-1)?.child;
    assert.deepEqual(child?.slice(1), [path.join(repo, "scripts/build-local-native-sdk.js"), "--local-jit-resources", "build", f.workspace,
      tools.cmake, tools.ninja, tools.cc, tools.cxx]);
    assert.equal(path.basename(JSON.parse(fs.readFileSync(paths.linuxJitTools, "utf8")).tools.cxx.path), "icpx");
    assert.equal(fs.existsSync(paths.linuxJitAdmission), false);
  });

  test("binds a completed local admission to the shared auditor fingerprints and leaves warm receipts unchanged", t => {
    const f = fixture(t), runner = fakeTools(f.root), tools = jitTools(runner), paths = supportPaths(f.workspace), state = localLinuxJitResourceState(paths);
    makeJitToolsExecutable(runner); fs.writeFileSync(paths.linuxJitTools, JSON.stringify(localJitToolsReceipt(tools)) + "\n");
    const fingerprintPath = path.join(repo, "scripts/native-ci-linux-jit.js"), admission = {resources: {target: "x86_64-unknown-linux-gnu",
      directory: state.resources, overlay: path.join(state.resources, "overlay"), manifestHash: "a".repeat(64), inputs: {[fingerprintPath]: "b".repeat(64)}},
    sourceDate: "20260711", fingerprints: {[fingerprintPath]: "b".repeat(64)}};
    let builds = 0, audits = 0;
    const built = runLocalJitResourceChild("build", f.workspace, tools, {verifyScope() {assert.ok(true);},
      validateTools(selected) {assert.deepEqual(selected, tools);}, buildTree(received) {
        builds++; assert.equal(received.root, state.root); fs.mkdirSync(received.root); return admission;
      }, auditTree() {audits++; throw new Error("fresh builder already returned audited admission");}});
    assert.equal(builds, 1); assert.equal(audits, 0); assert.equal(built.manifestHash, admission.resources.manifestHash);
    const receiptBytes = fs.readFileSync(paths.linuxJitAdmission), reused = runLocalJitResourceChild("audit", f.workspace, tools,
      {verifyScope() {assert.ok(true);}, validateTools(selected) {assert.deepEqual(selected, tools);},
        buildTree() {throw new Error("warm reuse must not rerun producers");}, auditTree() {audits++; return admission;}});
    assert.equal(audits, 1); assert.equal(reused.sourceDate, "20260711"); assert.deepEqual(fs.readFileSync(paths.linuxJitAdmission), receiptBytes);
    const stale = {...admission, fingerprints: {[fingerprintPath]: "c".repeat(64)}};
    assert.throws(() => runLocalJitResourceChild("audit", f.workspace, tools,
      {verifyScope() {assert.ok(true);}, validateTools(selected) {assert.deepEqual(selected, tools);}, auditTree() {return stale;}}), /fingerprints changed/);
    assert.deepEqual(fs.readFileSync(paths.linuxJitAdmission), receiptBytes);

    const empty = fixture(t), emptyRunner = fakeTools(empty.root), emptyTools = jitTools(emptyRunner), emptyPaths = supportPaths(empty.workspace);
    makeJitToolsExecutable(emptyRunner); fs.writeFileSync(emptyPaths.linuxJitTools, JSON.stringify(localJitToolsReceipt(emptyTools)) + "\n");
    const emptyState = localLinuxJitResourceState(emptyPaths), noFingerprints = {...admission, resources: {...admission.resources,
      directory: emptyState.resources}, fingerprints: {}};
    assert.throws(() => runLocalJitResourceChild("build", empty.workspace, emptyTools,
      {verifyScope() {assert.ok(true);}, validateTools(selected) {assert.deepEqual(selected, emptyTools);}, buildTree(received) {
        fs.mkdirSync(received.root); return noFingerprints;
      }}), /incomplete local admission/);
    assert.equal(fs.existsSync(emptyState.root), true); assert.equal(fs.existsSync(emptyPaths.linuxJitAdmission), false);
  });

  test("preserves interrupted output and stops before the JIT producer when prerequisites fail", t => {
    const partial = fixture(t), partialRunner = fakeTools(partial.root), tools = jitTools(partialRunner), paths = supportPaths(partial.workspace);
    makeJitToolsExecutable(partialRunner); fs.writeFileSync(paths.linuxJitTools, JSON.stringify(localJitToolsReceipt(tools)) + "\n");
    const state = localLinuxJitResourceState(paths); fs.mkdirSync(state.root); fs.writeFileSync(path.join(state.root, "unknown-partial.log"), "preserve\n");
    let audited = false;
    assert.throws(() => runLocalJitResourceChild("audit", partial.workspace, tools,
      {verifyScope() {assert.ok(true);}, validateTools(selected) {assert.deepEqual(selected, tools);}, auditTree() {
        audited = true; return /** @type {ReturnType<typeof import("../scripts/native-ci-linux-jit").auditLinuxJitResourceTree>} */ (/** @type {unknown} */ ({}));
      }}), /incomplete or out of order/);
    assert.equal(audited, false); assert.equal(fs.readFileSync(path.join(state.root, "unknown-partial.log"), "utf8"), "preserve\n");

    const capped = fixture(t), cappedRunner = fakeTools(capped.root), cappedTools = jitTools(cappedRunner), cappedPaths = supportPaths(capped.workspace);
    makeJitToolsExecutable(cappedRunner); fs.writeFileSync(cappedPaths.linuxJitTools, JSON.stringify(localJitToolsReceipt(cappedTools)) + "\n");
    let produced = false;
    assert.throws(() => runLocalJitResourceChild("build", capped.workspace, cappedTools,
      {verifyScope() {throw new Error("scope limit missing");}, validateTools(selected) {assert.deepEqual(selected, cappedTools);},
        buildTree() {produced = true; throw new Error("must not build");}}), /scope limit missing/);
    assert.equal(produced, false); assert.equal(fs.existsSync(cappedPaths.linuxJitResources), false);

    const f = fixture(t), ctx = context(f), runner = fakeTools(f.root), mocked = mockJitRun(runner); makeJitToolsExecutable(runner);
    assert.throws(() => buildLocalLinuxJitResources(ctx, {tools: runner, run: mocked.run, environment: {PATH: "/usr/bin"},
      prepareInputs() {throw new Error("DPC++ input admission failed");}, prepareHeaders() {throw new Error("must not reach GNU headers");}}),
    /DPC\+\+ input admission failed/);
    assert.deepEqual(mocked.calls.map(row => row.label), ["systemd memory-limit preflight", "LLVM 21 source SDK re-admission"]);
    assert.equal(fs.existsSync(supportPaths(f.workspace).linuxJitResources), false);
  });
});
