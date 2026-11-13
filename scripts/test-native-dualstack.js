"use strict";

// Audit-only, Linux-only native dual-stack regression. It builds two otherwise-identical AOT
// fixtures: the historical first-address event loop from 09eb680 and the current event loop.
// The test wrapper changes only resolver ordering and, for one case, simulates a pending first
// connect. Production code remains libuv-only; no DNS implementation is copied into runtime JS.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {fork, spawnSync} = require("node:child_process");
const {once} = require("node:events");
const {createHash} = require("node:crypto");
const esbuild = require("esbuild");
const {controllerBundleOptions} = require("./native-bundle");
const nativeResources = require("./native-resources");

const BEFORE_REVISION = "09eb680";
const HERMES_REVISION = "26fbb82fad39d9d06901adb1d6d630dacc84c877";
const MAX_OUTPUT = 8 << 20;

/** @param {string} file */
function hash(file) {
  return createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

/** @param {string} file @param {unknown} value */
function save(file, value) {
  fs.writeFileSync(file, typeof value === "string" ? value : JSON.stringify(value, null, 2) + "\n", {flag: "wx"});
}

/** @param {string} command @param {string[]} args @param {string} log @param {string} cwd */
function run(command, args, log, cwd) {
  const result = spawnSync(command, args, {cwd, encoding: "utf8", timeout: 120000, maxBuffer: MAX_OUTPUT});
  save(log, (result.stdout || "") + (result.stderr || ""));
  if (result.error || result.signal || result.status !== 0) {
    throw new Error(`${command} failed (exit ${result.status}, signal ${result.signal}); see ${log}`, {cause: result.error});
  }
  return result.stdout || "";
}

/** @param {string} directory */
function cgroupReport(directory) {
  const mount = "/sys/fs/cgroup";
  const cgroupLine = fs.readFileSync("/proc/self/cgroup", "utf8").split("\n")
    .find((line) => line.startsWith("0::"));
  assert.ok(cgroupLine, "native dual-stack audit needs a unified cgroup");
  const relative = cgroupLine.slice(3);
  assert.ok(relative.startsWith("/") && !relative.split("/").includes(".."),
    "native dual-stack audit received an invalid cgroup path");
  const cgroup = path.join(mount, relative.slice(1));
  /** @param {string} name */
  const read = (name) => fs.readFileSync(path.join(cgroup, name), "utf8").trim();
  const memory = read("memory.max"), swap = read("memory.swap.max"), pids = read("pids.max");
  assert.equal(memory, String(2 * 1024 * 1024 * 1024), "native dual-stack audit needs a 2 GiB cgroup");
  assert.equal(swap, "0", "native dual-stack audit needs swap disabled");
  assert.equal(pids, "96", "native dual-stack audit needs pids.max=96");
  const [quota, period] = read("cpu.max").split(/\s+/).map(Number);
  assert.equal(quota, 200000, "native dual-stack audit needs two CPUs");
  assert.equal(period, 100000, "native dual-stack audit needs a 100 ms CPU period");
  const result = {path: relative, memory, swap, pids, cpu: read("cpu.max")};
  save(path.join(directory, "cgroup.json"), result);
  return result;
}

/** @param {string} output @param {string} repo */
function bundleFixture(output, repo) {
  const entry = path.join(output, "entry.js"), bundle = path.join(output, "dualstack.js");
  save(entry, `require(${JSON.stringify(path.join(repo, "tests/reference/native_dualstack_fixture.js"))});\n`);
  const result = esbuild.buildSync(controllerBundleOptions(entry, bundle));
  save(path.join(output, "bundle.json"), result.metafile || {});
  return bundle;
}

/** @param {string} repo @param {string} output */
function writeResources(repo, output) {
  const policy = fs.readFileSync(path.join(repo, "GPU-CONFIG.md"));
  save(path.join(output, "app-resources.inc"), nativeResources([{name: "GPU-CONFIG.md", data: policy}]) +
    `static constexpr std::string_view kEmbeddedPolicyPath = ${JSON.stringify(path.join(repo, "GPU-CONFIG.md"))};\n`);
}

/** @typedef {{repo: string, hermes: string, hermesBuild: string, openssl: string, uvInclude: string,
 * uvLibrary: string, output: string, source: string, cxx: string, cc: string}} BuildOptions */
/** @param {BuildOptions} options */
function buildObjects(options) {
  const {repo, hermes, hermesBuild, openssl, uvInclude, output, source, cxx} = options;
  const includes = [path.join(hermes, "include"), path.join(hermesBuild, "lib/config")];
  const commonCxx = ["-std=c++17", "-O2", "-DNDEBUG", "-Wall", "-Wextra", "-Werror",
    ...includes.flatMap((dir) => ["-isystem", dir]),
    "-isystem", path.join(hermes, "API"), "-isystem", path.join(hermes, "API/jsi"),
    "-isystem", path.join(hermes, "public"), "-isystem", path.join(openssl, "include"),
    // /usr/include is also GCC's next-search target for <cstdlib>; marking it -isystem moves it
    // ahead of libstdc++ and makes cstdlib's include_next(stdlib.h) miss the multiarch headers.
    "-I", uvInclude, "-I", repo, "-I", path.join(repo, "runtime")];
  /** @param {string} input @param {string} name @param {string[]} [extra] */
  const compile = (input, name, extra = []) => {
    const object = path.join(output, `${name}.o`);
    run(cxx, [...commonCxx, ...extra, "-c", input, "-o", object], path.join(output, `${name}.log`), repo);
    return object;
  };
  const processObjects = ["process", "main", "services", "nvidia-metadata"].map((name) => compile(
    path.join(repo, "runtime", `${name}.cpp`), name,
    name === "services" ? ["-DMOM_EMBEDDED_APP_RESOURCES", "-I", path.dirname(output)] : []));
  const host = compile(path.join(repo, "runtime/host.cpp"), "console");
  const tlsRoots = path.join(output, "tls-roots.inc");
  save(tlsRoots, `static constexpr char kTlsRoots[] = ${JSON.stringify(require("node:tls").rootCertificates.join("\n"))};\n`);
  const crypto = compile(path.join(repo, "runtime/tls.cpp"), "tls", ["-I", output]);
  const wrapper = compile(path.join(repo, "tests/reference/native_dualstack.cpp"), "dualstack-wrapper");
  const eventLoop = compile(source, path.basename(output) === "after" ? "event-loop-after" : "event-loop-before");
  return {processObjects, crypto, wrapper, eventLoop, host, commonCxx};
}

/** @param {BuildOptions & {objects: ReturnType<typeof buildObjects>}} options */
function linkFixture(options) {
  const {hermesBuild, openssl, uvLibrary, output, objects, cxx} = options;
  const executable = path.join(output, "dualstack-native");
  const controller = path.join(output, "controller.o");
  const includes = [path.join(options.hermes, "include"), path.join(hermesBuild, "lib/config")];
  run(options.cc, ["-std=gnu11", "-O2", "-DNDEBUG", "-fno-strict-aliasing", "-fno-strict-overflow",
    ...includes.flatMap((dir) => ["-I", dir]), "-c", path.join(output, "controller.c"), "-o", controller],
  path.join(output, "controller-compile.log"), options.repo);
  const args = ["-Wl,--gc-sections", "-static-libstdc++", "-static-libgcc", controller,
    objects.host, objects.eventLoop, objects.crypto, ...objects.processObjects, objects.wrapper,
    path.join(hermesBuild, "lib/libhermesvmlean_a.a"), path.join(hermesBuild, "jsi/libjsi.a"),
    path.join(hermesBuild, "public/hermes/Public/libhermesPublic.a"),
    path.join(hermesBuild, "external/boost/boost_1_86_0/libs/context/libboost_context.a"),
    path.join(openssl, "lib/libssl.a"), path.join(openssl, "lib/libcrypto.a"),
    "-Wl,-Bstatic", uvLibrary, "/usr/lib/x86_64-linux-gnu/libz.a", "-Wl,-Bdynamic", "-lm", "-ldl", "-lpthread", "-lrt",
    "-Wl,--wrap=uv_getaddrinfo", "-Wl,--wrap=uv_cancel", "-Wl,--wrap=uv_tcp_connect", "-Wl,--wrap=uv_close", "-o", executable];
  save(path.join(output, "link-inputs.json"), {cxx, args});
  run(cxx, args, path.join(output, "link.log"), options.repo);
  return executable;
}

/** @param {BuildOptions} options */
function build(options) {
  const {output, repo, hermesBuild} = options;
  const bundle = bundleFixture(output, repo);
  run(path.join(hermesBuild, "bin/shermes"), ["-Xenable-tdz", "-Xes6-block-scoping", "-exported-unit=mom_entry", "-emit-c", "-o", path.join(output, "controller.c"), bundle], path.join(output, "aot.log"), repo);
  const after = buildObjects({...options, output: path.join(output, "after"), source: path.join(repo, "runtime/event-loop.cpp")});
  const beforeSource = path.join(output, "event-loop-before.cpp");
  const before = spawnSync("git", ["-C", repo, "show", `${BEFORE_REVISION}:runtime/event-loop.cpp`], {encoding: "utf8", maxBuffer: 4 << 20});
  assert.equal(before.status, 0, before.stderr);
  save(beforeSource, before.stdout);
  const beforeObjects = buildObjects({...options, output: path.join(output, "before"), source: beforeSource});
  /** @param {string} name @param {ReturnType<typeof buildObjects>} objects */
  const buildOne = (name, objects) => linkFixture({...options, output: path.join(output, name), objects});
  // The controller C object, entry bundle and support objects are shared by both links; event-loop
  // objects are compiled independently from the historical and current source snapshots.
  fs.copyFileSync(path.join(output, "controller.c"), path.join(output, "after/controller.c"));
  fs.copyFileSync(path.join(output, "controller.c"), path.join(output, "before/controller.c"));
  const afterBinary = buildOne("after", after), beforeBinary = buildOne("before", beforeObjects);
  save(path.join(output, "objects.json"), {before: {eventLoop: hash(beforeObjects.eventLoop), source: hash(beforeSource)},
    after: {eventLoop: hash(after.eventLoop), source: hash(path.join(repo, "runtime/event-loop.cpp"))},
    binaries: {before: hash(beforeBinary), after: hash(afterBinary)}});
  return {beforeBinary, afterBinary};
}

/** @param {string} executable @param {string} output @param {number} port @param {string} mode @param {string} label */
function runCase(executable, output, port, mode, label) {
  /** @type {NodeJS.ProcessEnv} */
  const env = {...process.env, MOM_DUALSTACK_REORDER: "1", MOM_DUALSTACK_PORT: String(port), MOM_DUALSTACK_CASE: mode,
    MOM_DUALSTACK_TIMEOUT_FIRST: mode === "timeout" ? "1" : undefined,
    MOM_DUALSTACK_DELAY_FIRST_CLOSE_MS: mode === "cancel" ? "100" : undefined,
    MOM_DUALSTACK_CANCEL_DELAY_MS: mode === "cancel" ? "5" : undefined};
  if (mode !== "timeout") {Reflect.deleteProperty(env, "MOM_DUALSTACK_TIMEOUT_FIRST");}
  if (mode !== "cancel") {
    Reflect.deleteProperty(env, "MOM_DUALSTACK_DELAY_FIRST_CLOSE_MS");
    Reflect.deleteProperty(env, "MOM_DUALSTACK_CANCEL_DELAY_MS");
  }
  const result = spawnSync(executable, [], {env, encoding: "utf8", timeout: 6000, maxBuffer: MAX_OUTPUT});
  const text = (result.stdout || "") + (result.stderr || "") +
    (result.error ? `\n[spawn error] ${result.error.stack || result.error.message}\n` : "");
  save(path.join(output, `${label}.log`), text);
  const json = text.trim().split("\n").map((line) => {
    try {return JSON.parse(line);} catch {return undefined;}
  }).find((value) => value && typeof value === "object" && "mode" in value);
  return {status: result.status, signal: result.signal,
    timedOut: (/** @type {NodeJS.ErrnoException | undefined} */ (result.error))?.code === "ETIMEDOUT", json, text};
}

async function main() {
  assert.equal(process.platform, "linux", "The dual-stack AOT audit is Linux-only");
  const [hermes, hermesBuild, openssl, outputParent = os.tmpdir()] = process.argv.slice(2);
  assert.ok(hermes && hermesBuild && openssl, "Usage: test-native-dualstack.js <Hermes> <Hermes build> <OpenSSL> [output parent]");
  const repo = path.resolve(__dirname, "..");
  const output = fs.mkdtempSync(path.join(path.resolve(outputParent), "native-dualstack-2-"));
  cgroupReport(output);
  assert.equal(spawnSync("git", ["-C", hermes, "rev-parse", "HEAD"], {encoding: "utf8"}).stdout.trim(), HERMES_REVISION);
  assert.match(fs.readFileSync(path.join(hermesBuild, "CMakeCache.txt"), "utf8"), /^HERMESVM_INTERNAL_JAVASCRIPT_NATIVE:BOOL=ON$/m);
  assert.match(run(path.join(openssl, "bin/openssl"), ["version"], path.join(output, "openssl-version.log"), repo), /^OpenSSL 3\.5\.9\b/);
  writeResources(repo, output);
  fs.mkdirSync(path.join(output, "before")); fs.mkdirSync(path.join(output, "after"));
  const uvInclude = process.env["MOM_DUALSTACK_UV_INCLUDE"] || "/usr/include";
  const uvLibrary = process.env["MOM_DUALSTACK_UV_LIBRARY"] || "/usr/lib/x86_64-linux-gnu/libuv.a";
  const cxx = process.env["CXX"] || "c++", cc = process.env["CC"] || "cc";
  const binaries = build({repo, hermes, hermesBuild, openssl, uvInclude, uvLibrary, output,
    source: path.join(repo, "runtime/event-loop.cpp"), cxx, cc});
  const server = fork(path.join(repo, "tests/reference/native_dualstack_server.js"), [], {stdio: ["ignore", "ignore", "pipe", "ipc"]});
  const serverLog = path.join(output, "server.log");
  server.stderr?.on("data", (data) => fs.appendFileSync(serverLog, data));
  /** @param {Error} error */
  const serverError = (error) => fs.appendFileSync(serverLog, `${error.stack || error.message}\n`);
  server.on("error", serverError);
  const failures = [];
  try {
    const [message] = await once(server, "message", {signal: AbortSignal.timeout(5000)});
    assert.ok(message && Number.isInteger(message.port) && message.port > 0 && message.port <= 65535);
    /** @type {Record<string, ReturnType<typeof runCase>>} */
    const cases = {};
    /** @param {string} name */
    const getCase = (name) => {
      const value = cases[name];
      assert.ok(value);
      return value;
    };
    for (const mode of ["refused", "timeout", "cancel", "queuedwrites"]) {
      cases[`before-${mode}`] = runCase(binaries.beforeBinary, output, message.port, mode, `before-${mode}`);
      cases[`after-${mode}`] = runCase(binaries.afterBinary, output, message.port, mode, `after-${mode}`);
    }
    for (const [name, report] of Object.entries(cases)) {
      assert.equal(report.timedOut, false, `${name} was killed by the harness timeout`);
      assert.ok(!report.json?.trace.includes("deadline"), `${name} hit the fixture deadline`);
    }
    assert.equal(getCase("before-refused").status, 2);
    assert.equal(getCase("before-refused").json?.connected, false);
    assert.equal(getCase("before-timeout").status, 2);
    assert.equal(getCase("before-timeout").json?.connected, false);
    assert.equal(getCase("after-refused").status, 0);
    assert.equal(getCase("after-timeout").status, 0);
    assert.equal(getCase("after-cancel").status, 0);
    assert.equal(getCase("before-cancel").json?.closeCount, 1);
    for (const mode of ["refused", "timeout"]) {
      const report = getCase(`after-${mode}`).json;
      assert.equal(report?.connected, true);
      assert.equal(report?.closeCount, 1);
      assert.ok(report?.trace.includes("data:pong\n"));
    }
    const cancelReport = getCase("after-cancel").json;
    assert.deepEqual(cancelReport && {
      connected: cancelReport.connected,
      closeCount: cancelReport.closeCount,
      failed: cancelReport.failed,
    }, {connected: false, closeCount: 1, failed: false});
    assert.equal(getCase("before-queuedwrites").status, 2);
    assert.equal(getCase("before-queuedwrites").json?.connected, false);
    assert.equal(getCase("after-queuedwrites").status, 0);
    const queuedReport = getCase("after-queuedwrites").json;
    assert.deepEqual(queuedReport && {
      connected: queuedReport.connected,
      closeCount: queuedReport.closeCount,
      failed: queuedReport.failed,
      queuedHeader: queuedReport.queuedHeader,
      queuedBytes: queuedReport.queuedBytes,
    }, {connected: true, closeCount: 1, failed: false, queuedHeader: true, queuedBytes: 8 * 1024 * 1024});
    assert.ok(queuedReport?.blockedWrites > 0 && queuedReport.writeCallbacks > 0);
    save(path.join(output, "cases.json"), cases);
    save(path.join(output, "result.json"), {status: "PASS", output, cases: Object.keys(cases),
      beforeRevision: BEFORE_REVISION, currentSource: path.join(repo, "runtime/event-loop.cpp"),
      injected: {resolver: "two libuv requests (localhost + ip6-localhost), merged/reordered IPv4-first", timeout: "simulated pending IPv4 uv_tcp_connect", cancellation: "delayed first TCP close callback"},
      bounds: {addresses: 64, attemptTimeoutMs: 25, fixtureDeadlineMs: 1500},
      scope: "CPU-only AOT production event-loop object before/after; no GPU/Core, no JS DNS implementation"});
  } catch (error) {failures.push(error);} finally {
    if (server.exitCode === null && server.signalCode === null) {
      const exited = once(server, "exit", {signal: AbortSignal.timeout(5000)});
      if (server.connected) {server.disconnect();}
      try {
        await exited;
      } catch (error) {
        if (server.exitCode === null && server.signalCode === null) {server.kill("SIGTERM");}
        try {await once(server, "exit", {signal: AbortSignal.timeout(5000)});} catch { /* preserve the original timeout */ }
        failures.push(error);
      }
    }
    if (server.exitCode !== 0 || server.signalCode) {
      failures.push(new Error(`Dual-stack server exited (exit ${server.exitCode}, signal ${server.signalCode}); see ${serverLog}`));
    }
    server.off("error", serverError);
  }
  if (failures.length) {throw new AggregateError(failures, `Native dual-stack audit failed; evidence: ${output}`);}
  console.log(JSON.stringify({status: "PASS", output, before: binaries.beforeBinary, after: binaries.afterBinary}));
}

if (require.main === module) {main().catch((error) => {console.error(error); process.exitCode = 1;});}

module.exports = {cgroupReport};
