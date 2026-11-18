"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {verify, mine, miningServer, terminateChild, savedConfigPermissions} = require("./tools/probe-native-mining");
const {nativeCgroupParent, nativeContainerLaunch, nativeDeviceArgs, nativePearlFixtureTarget, nativePearlStraceArgs,
  nativeVerifierModuleFiles, probeProfile} = require("./tools/probe-native-pearl");
const {EventEmitter} = require("node:events");
const {spawnSync} = require("node:child_process"), path = require("node:path");
const net = require("node:net"), tls = require("node:tls");
/** @typedef {{once: (event: string, listener: (...args: unknown[]) => void) => unknown}} TerminationKiller */
/** @typedef {EventEmitter & TerminationKiller} TestKiller */
/** @typedef {(command: string, args: string[], options: {stdio: "ignore", windowsHide: true, timeout: number}) => TerminationKiller} TerminationSpawn */
/** @typedef {(pid: number, signal: "SIGKILL") => void} ProcessGroupKill */
/** @typedef {{pid?: number | undefined, exitCode: number | null, signalCode: string | null,
 * kill: (signal?: number | NodeJS.Signals) => void}} TerminationChild */

test.describe("Native mining parity share verifier", {concurrency: false}, () => {
  test("plaintext mining keeps the original private TCP fixture", () => {
    const handle = () => undefined, expected = new net.Server(), original = net.createServer;
    let calls = 0;
    net.createServer = /** @type {typeof net.createServer} */ (listener => {calls++; assert.equal(listener, handle); return expected;});
    try {assert.equal(miningServer(undefined, handle), expected); assert.equal(calls, 1);} finally {net.createServer = original;}
  });
  test("TLS mining reuses the same share handler with exact certificate bytes", () => {
    const handle = () => undefined, expected = new net.Server(), original = tls.createServer;
    const certificate = {key: Buffer.from("private test key"), cert: Buffer.from("private test certificate")};
    let calls = 0;
    tls.createServer = /** @type {typeof tls.createServer} */ ((options, listener) => {
      calls++; assert.equal(listener, handle);
      assert.deepEqual(options, {...certificate, minVersion: "TLSv1.2"}); return expected;
    });
    try {assert.equal(miningServer(certificate, handle), expected); assert.equal(calls, 1);} finally {tls.createServer = original;}
  });
  test("malformed TLS material fails locally rather than falling back to TCP", () => {
    assert.throws(() => miningServer({key: "invalid key", cert: "invalid certificate"}, () => undefined));
  });
  test("keeps strict Unix 0600 validation and marks Windows mode unverified", () => {
    const regular600 = {isFile: () => true, mode: 0o600};
    const regular644 = {isFile: () => true, mode: 0o644};
    const directory = {isFile: () => false, mode: 0o600};
    assert.deepEqual(savedConfigPermissions(regular600, "linux"),
      {mode: 0o600, posixModeApplicable: true, posixModeVerified: true});
    assert.deepEqual(savedConfigPermissions(regular600, "win32"),
      {mode: null, posixModeApplicable: false, posixModeVerified: false});
    assert.throws(() => savedConfigPermissions(regular644, "linux"), /mode 0600/);
    assert.throws(() => savedConfigPermissions(directory, "linux"), /regular file/);
    assert.throws(() => savedConfigPermissions(directory, "win32"), /regular file/);
  });
  test("uses Windows taskkill tree termination only for forced cleanup", () => {
    /** @type {unknown[]} */
    const calls = [];
    const killer = /** @type {TestKiller} */ (new EventEmitter());
    const child = /** @type {TerminationChild} */ ({pid: 37, exitCode: null, signalCode: null,
      kill: (signal) => {calls.push(["fallback", signal]);}});
    const spawnProcess = /** @type {TerminationSpawn} */ ((command, args, options) => {
      calls.push([command, args, options]); return killer;
    });
    terminateChild(child, "win32", spawnProcess);
    assert.deepEqual(calls[0], [path.win32.join(process.env["SystemRoot"] || "C:\\Windows", "System32", "taskkill.exe"), ["/pid", "37", "/t", "/f"],
      {stdio: "ignore", windowsHide: true, timeout: 5000}]);
    killer.emit("close", 0); assert.equal(calls.length, 1);
    const failedKiller = /** @type {TestKiller} */ (new EventEmitter());
    const failedChild = /** @type {TerminationChild} */ ({pid: 38, exitCode: null, signalCode: null,
      kill: (signal) => {calls.push(["fallback", signal]);}});
    const failedSpawn = /** @type {TerminationSpawn} */ (() => failedKiller);
    terminateChild(failedChild, "win32", failedSpawn);
    failedKiller.emit("close", 1); assert.deepEqual(calls.at(-1), ["fallback", "SIGKILL"]);
  });
  test("falls back after Windows taskkill spawn error without launching taskkill", () => {
    /** @type {unknown[]} */
    const calls = [];
    const killer = /** @type {TestKiller} */ (new EventEmitter());
    const child = /** @type {TerminationChild} */ ({pid: 39, exitCode: null, signalCode: null,
      kill: (signal) => {calls.push(signal);}});
    const spawnProcess = /** @type {TerminationSpawn} */ (() => killer);
    terminateChild(child, "win32", spawnProcess);
    killer.emit("error", new Error("fixture taskkill failure"));
    assert.deepEqual(calls, ["SIGKILL"]);
  });
  test("does not terminate an already closed or pid-less child", () => {
    /** @type {unknown[]} */
    const calls = [];
    const spawnProcess = /** @type {TerminationSpawn} */ (() => {
      calls.push("spawn"); return /** @type {TestKiller} */ (new EventEmitter());
    });
    terminateChild(/** @type {TerminationChild} */ ({pid: 40, exitCode: 0, signalCode: null, kill: () => {calls.push("kill");}}), "win32", spawnProcess);
    terminateChild(/** @type {TerminationChild} */ ({pid: undefined, exitCode: null, signalCode: null, kill: () => {calls.push("kill");}}), "win32", spawnProcess);
    assert.deepEqual(calls, []);
  });
  test("retains the Unix negative-PID process-group boundary", () => {
    /** @type {unknown[]} */
    const calls = [];
    const child = /** @type {TerminationChild} */ ({pid: 41, exitCode: null, signalCode: null, kill: () => assert.fail("direct kill bypassed Unix group")});
    const noSpawn = /** @type {TerminationSpawn} */ (() => assert.fail("taskkill used on Unix"));
    const killGroup = /** @type {ProcessGroupKill} */ ((pid, signal) => {calls.push([pid, signal]);});
    terminateChild(child, "linux", noSpawn, killGroup);
    assert.deepEqual(calls, [[-41, "SIGKILL"]]);
  });
  test("private mining cannot bind wildcard or public endpoints", async () => {
    const core = {AsyncWorker: class {sendToCpp() {throw new Error("must not dispatch");}}};
    for (const host of ["0.0.0.0", "::", "localhost", "192.0.2.1", "::ffff:127.0.0.1"]) {
      await assert.rejects(mine({command: "must-not-launch", args: [], env: {}}, core,
        "/must-not-write", "invalid", undefined, host), /literal loopback/);
    }
  });
  /** @param {string} [result] @param {string} [event] @param {Error} [error] */
  function oracle(result = "ab".repeat(32), event = "test", error) {
    const commands = /** @type {Array<{name: string, values?: Record<string, string>}>} */ ([]);
    return {commands, AsyncWorker: class {
      /** @param {(name: string, values: Record<string, string>) => void} progress
       * @param {() => void} complete @param {(error: Error) => void} failed */
      constructor(progress, complete, failed) {this.progress = progress; this.complete = complete; this.failed = failed;}
      /** @param {string} name @param {Record<string, string>} [values] */
      sendToCpp(name, values) {
        commands.push({name, ...(values ? {values} : {})});
        setImmediate(() => {
          if (name === "close") {this.complete();} else if (error) {this.failed(error);} else {this.progress(event, {result});}
        });
      }
    }};
  }
  test("reuses the existing Core test command and CPU execution path", async () => {
    const core = oracle(); await verify(core, "cn/0", "01020304", "ab".repeat(32));
    assert.deepEqual(core.commands.map(({name}) => name), ["test", "close"]);
    assert.equal(core.commands[0]?.values?.["dev"], "cpu"); assert.equal(core.commands[0]?.values?.["algo"], "cn/0");
  });
  test("wire nonce bytes are not integer-reversed", async () => {
    const core = oracle(); await verify(core, "cn/1", "01020304", "ab".repeat(32));
    assert.equal(core.commands[0]?.values?.["blob_hex"]?.slice(78, 86), "01020304");
  });
  test("rejects an independently recomputed hash mismatch", async () => {
    await assert.rejects(verify(oracle("cd".repeat(32)), "cn/0", "00000000", "ab".repeat(32)), /Invalid cn\/0 submitted hash/);
  });
  test("rejects an unexpected compute event even with a matching result field", async () => {
    await assert.rejects(verify(oracle(undefined, "error"), "cn/0", "00000000", "ab".repeat(32)), /Verifier event: error/);
  });
  test("propagates actual Core failures", async () => {
    const error = new Error("fixture");
    await assert.rejects(verify(oracle(undefined, undefined, error), "cn/0", "00000000", "ab".repeat(32)), (value) => value === error);
  });
  test("unsupported algorithms cannot accidentally launch GPU work", async () => {
    const core = oracle(); await assert.rejects(verify(core, "pearlhash", "00000000", "ab".repeat(32)), /Malformed/);
    assert.equal(core.commands.length, 0);
  });
  test("rejects a truncated nonce before execution", async () => {
    const core = oracle(); await assert.rejects(verify(core, "cn/0", "0000", "ab".repeat(32)), /Malformed/);
    assert.equal(core.commands.length, 0);
  });
  test("rejects nonhex nonce data instead of silently truncating Buffer input", async () => {
    await assert.rejects(verify(oracle(), "cn/0", "zz000000", "ab".repeat(32)), /Malformed/);
  });
  test("rejects an invalid submitted hash before creating a worker", async () => {
    const core = oracle(); await assert.rejects(verify(core, "cn/0", "00000000", "ab"), /Malformed/); assert.equal(core.commands.length, 0);
  });
  test("synchronous dispatch failure does not leave a pending deadline", async () => {
    const error = new Error("dispatch");
    const core = {AsyncWorker: class {sendToCpp() {throw error;}}};
    await assert.rejects(verify(core, "cn/0", "00000000", "ab".repeat(32)), (value) => value === error);
  });
  test("native Pearl execution cannot implicitly launch from ordinary source tests", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "./tools/probe-native-pearl.js")],
      {encoding: "utf8", timeout: 5000, env: {...process.env, MOM_AOT_GPU_OWNERSHIP_VERIFIED: undefined}});
    assert.equal(result.status, 1); assert.match(result.stderr, /Verify fresh GPU ownership/);
    assert.doesNotMatch(result.stdout, /evidence/);
  });
  test("the explicit execution marker never admits malformed device identity", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "./tools/probe-native-pearl.js"),
      "/nonexistent/mom", "/nonexistent/verifier", "all"], {encoding: "utf8", timeout: 5000,
      env: {...process.env, MOM_AOT_GPU_OWNERSHIP_VERIFIED: "1"}});
    assert.equal(result.status, 1); assert.match(result.stderr, /Verify fresh GPU ownership/);
    assert.doesNotMatch(result.stderr, /Cannot find module/);
  });
  test("optional Docker parent accepts only one simple systemd slice name", () => {
    assert.equal(nativeCgroupParent(undefined), undefined);
    assert.equal(nativeCgroupParent("mom094native.slice"), "mom094native.slice");
    for (const value of ["", "/mom094native.slice", "mom094native.slice/child", "mom094native.service", "bad slice.slice",
      "mom094native.slice\n--privileged", "x".repeat(256) + ".slice"]) {
      assert.throws(() => nativeCgroupParent(value), /MOM_NATIVE_CGROUP_PARENT/);
    }
  });
  test("unset Docker parent preserves the original strace entrypoint and argv", () => {
    const straceArgs = ["-f", "-s", "1024", "-e", "trace=connect", "-o", "/dev/stderr", "/mom", "mine", "/config.json"];
    assert.deepEqual(nativeContainerLaunch(undefined, straceArgs),
      {dockerArgs: [], entrypoint: "/usr/bin/strace", commandArgs: straceArgs});
  });
  test("configured Docker parent guards the actual 4GiB/no-swap cgroup before execing strace", () => {
    const straceArgs = ["-f", "-s", "1024", "-e", "trace=connect", "-o", "/dev/stderr", "/mom", "mine", "/config.json"];
    const launch = nativeContainerLaunch(nativeCgroupParent("mom094native.slice"), straceArgs);
    assert.deepEqual(launch.dockerArgs, ["--cgroup-parent", "mom094native.slice"]);
    assert.equal(launch.entrypoint, "/bin/sh");
    assert.equal(launch.commandArgs[0], "-ec");
    const guard = launch.commandArgs[1] ?? "";
    assert.match(guard, /read -r memory_max < \/sys\/fs\/cgroup\/memory\.max/);
    assert.match(guard, /test "\$memory_max" = 4294967296/);
    assert.match(guard, /read -r swap_max < \/sys\/fs\/cgroup\/memory\.swap\.max/);
    assert.match(guard, /test "\$swap_max" = 0/);
    assert.match(guard, /exec \/usr\/bin\/strace "\$@"$/);
    assert.equal(launch.commandArgs[2], "mom-native-cgroup-guard");
    assert.deepEqual(launch.commandArgs.slice(3), straceArgs);
  });
  test("generated cgroup guard reads its limits with shell builtins when PATH is unavailable", {skip: process.platform !== "linux"}, (t) => {
    const fs = require("node:fs"), os = require("node:os");
    const cgroup = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-cgroup-guard-"));
    t.after(() => fs.rmSync(cgroup, {recursive: true, force: true}));
    fs.writeFileSync(path.join(cgroup, "memory.max"), "4294967296\n");
    fs.writeFileSync(path.join(cgroup, "memory.swap.max"), "0\n");
    const launch = nativeContainerLaunch("mom094native.slice", []);
    const guard = (launch.commandArgs[1] ?? "").replaceAll("/sys/fs/cgroup/", cgroup + "/")
      .replace('exec /usr/bin/strace "$@"', 'printf "GUARD_OK:%s\\n" "$@"');
    const run = () => spawnSync("/bin/sh", ["-ec", guard, "mom-native-cgroup-guard", "sentinel"],
      {encoding: "utf8", timeout: 5000, env: {PATH: "/unavailable"}});
    let result = run();
    assert.equal(result.status, 0); assert.equal(result.stdout, "GUARD_OK:sentinel\n");
    fs.writeFileSync(path.join(cgroup, "memory.max"), "4294967295\n");
    result = run(); assert.notEqual(result.status, 0); assert.equal(result.stdout, "");
    fs.writeFileSync(path.join(cgroup, "memory.max"), "4294967296\n");
    fs.writeFileSync(path.join(cgroup, "memory.swap.max"), "1\n");
    result = run(); assert.notEqual(result.status, 0); assert.equal(result.stdout, "");
  });
  test("invalid Docker parent is rejected before loading the verifier", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "./tools/probe-native-pearl.js"),
      "/nonexistent/mom", "/nonexistent/verifier", "GPU-376cdd2a-9343-0add-67da-9d230a3638df"],
    {encoding: "utf8", timeout: 5000, env: {...process.env,
      MOM_AOT_GPU_OWNERSHIP_VERIFIED: "1", MOM_NATIVE_CGROUP_PARENT: "/mom094native.slice"}});
    assert.equal(result.status, 1);
    assert.match(result.stderr, /MOM_NATIVE_CGROUP_PARENT/);
    assert.doesNotMatch(result.stderr, /Cannot find module/);
  });
});

test.describe("Native Pearl execution profiles", () => {
  const nvidia = "GPU-376cdd2a-9343-0add-67da-9d230a3638df", amd = "/dev/dri/renderD130";
  test("the qualified compact NVIDIA profile stays unchanged", () => {
    assert.deepEqual(probeProfile(nvidia), {gpu: "nvidia", shape: {m: 768, n: 512, k: 65536, rank: 128}, rows: 24, columns: 8,
      marker: "PEARLHASH_COMPACT admitted=1"});
  });
  test("AMD exercises its source-JIT row-slice traversal and native proof pattern", () => {
    assert.deepEqual(probeProfile(amd), {gpu: "amd", shape: {m: 2048, n: 2048, k: 2048, rank: 128}, rows: 4, columns: 64,
      marker: "PEARLHASH_HIP_STATS search="});
  });
  test("NVIDIA container reads the private config as the invoking host user", () => {
    const profile = probeProfile(nvidia); assert.ok(profile);
    assert.deepEqual(nativeDeviceArgs(profile, nvidia, 1000, 1007), ["--user", "1000:1007", "--gpus", `device=${nvidia}`]);
    assert.throws(() => nativeDeviceArgs(profile, nvidia, undefined, 1007), /invoking host UID:GID/);
  });
  test("AMD retains its explicit root and KFD-device launch contract", () => {
    const profile = probeProfile(amd); assert.ok(profile);
    assert.deepEqual(nativeDeviceArgs(profile, amd, 1000, 1007),
      ["--runtime=runc", "--user", "0:0", "--device", "/dev/kfd", "--device", amd]);
  });
  test("private compact proofs use a bounded target 16x easier than the first fixture", () => {
    const profile = probeProfile(nvidia); assert.ok(profile);
    const fixture = nativePearlFixtureTarget(profile.shape);
    const previous = ((1n << 256n) - 1n) / BigInt(16 * 16 * (profile.shape.k / profile.shape.rank) * 128 * 16777216);
    assert.equal(BigInt("0x" + fixture.target) / previous, 16n);
    assert.equal(fixture.baseTarget.length, 32);
    assert.equal(Buffer.from(fixture.baseTarget).reverse().toString("hex"), fixture.target);
  });
  test("optional IO tracing timestamps the network read/write evidence", () => {
    const suffix = ["-o", "/dev/stderr", "/mom", "mine", "/config.json"];
    const syscalls = "trace=open,openat,openat2,creat,execve,execveat,connect";
    assert.deepEqual(nativePearlStraceArgs(false), ["-f", "-s", "1024", "-e", syscalls, ...suffix]);
    assert.deepEqual(nativePearlStraceArgs(true), ["-tt", "-f", "-s", "1024", "-e", syscalls + ",read,write", ...suffix]);
  });
  test("saved verifier inventory includes its entry module and packaged native add-on", () => {
    const root = path.resolve("/pool", "node_modules", "node-powhash"), entry = path.join(root, "index.js"),
      addon = path.join(root, "build", "Release", "node-powhash.node");
    assert.deepEqual(nativeVerifierModuleFiles([entry, addon, path.join(root, "helper.js"),
      path.join("/pool", "node_modules", "blake2", "build", "Release", "blake2.node")], entry), [entry, addon]);
  });
  test("expected claims derive from the admitted pattern/configuration, not a common hardcoded factor", () => {
    const factors = [nvidia, amd].map((device) => {
      const profile = probeProfile(device); assert.ok(profile);
      return profile.rows * profile.columns * (profile.shape.k / profile.shape.rank) * 128;
    });
    assert.deepEqual(factors, [12582912, 524288]);
  });
  test("each caller gets an independent profile rather than mutable standing admission", () => {
    const profile = probeProfile(amd); assert.ok(profile); profile.shape.k = 1;
    assert.equal(probeProfile(amd)?.shape.k, 2048);
  });
  for (const device of [undefined, "", "all", "GPU-", "GPU-376CDD2A-9343-0add-67da-9d230a3638df", "/dev/kfd", "/dev/dri/card2",
    "/tmp/renderD130", "/dev/dri/renderD130/..", "/dev/dri/renderD130-extra", "/dev/dri/renderD130\n"]) {
    test(`malformed or overbroad device ${JSON.stringify(device)} is rejected without probing`, () => {
      assert.equal(probeProfile(device), null);
    });
  }
});
