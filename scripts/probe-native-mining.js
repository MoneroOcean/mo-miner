"use strict";

// Opt-in CPU-only end-to-end probe, not a public-pool miner or a replacement controller.
// Both runners execute maintained mom.js, and every accepted sample is rehashed by the Node
// oracle's existing CPU Core. All sockets are private ephemeral loopback listeners.
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const net = require("node:net");
const tls = require("node:tls");
const {spawn} = require("node:child_process");
const {once} = require("node:events");
const {isDeepStrictEqual} = require("node:util");
const repo = path.resolve(__dirname, "..");
const blob = String(require("../opts").opt_help.job.blob_hex[0]);
/** @typedef {{key: string | Buffer, cert: string | Buffer}} PrivateTls */
/** @typedef {{AsyncWorker: new(progress: (name: string, values: Record<string, string>) => void,
 * complete: () => void, error: (error: Error) => void) =>
 * {sendToCpp(name: string, values?: Record<string, string>): void}}} CpuOracle */
/** @typedef {{once: (event: string, listener: (...args: unknown[]) => void) => unknown}} TerminationKiller */
/** @typedef {(command: string, args: string[], options: {stdio: "ignore", windowsHide: true, timeout: number}) => TerminationKiller} TerminationSpawn */
/** @typedef {(pid: number, signal: "SIGKILL") => void} ProcessGroupKill */
/** @typedef {{pid?: number | undefined, exitCode: number | null, signalCode: string | null,
 * kill: (signal?: number | NodeJS.Signals) => void}} TerminationChild */

/** Reuse the actual independent execution mode, not another hash implementation.
 * @param {CpuOracle} core @param {string} algo @param {string} nonce @param {string} expected */
function verify(core, algo, nonce, expected) {
  if (!["cn/0", "cn/1"].includes(algo) || !/^[0-9a-f]{8}$/.test(nonce) || !/^[0-9a-f]{64}$/.test(expected)) {
    return Promise.reject(new Error("Malformed private share-verifier input"));
  }
  return new Promise((resolve, reject) => {
    // Submit nonce hex is already in blob byte order; integer parsing would reverse it.
    const input = Buffer.from(blob, "hex"); Buffer.from(nonce, "hex").copy(input, 39);
    let result = "", failure = "";
    const worker = new core.AsyncWorker((name, values) => {
      if (name === "test") {result = values["result"] || "";} else {failure = `Verifier event: ${name}`;}
      worker.sendToCpp("close");
    }, () => {
      clearTimeout(timer);
      if (!failure && result === expected) {resolve(undefined);} else {reject(new Error(failure || `Invalid ${algo} submitted hash`));}
    }, (error) => {clearTimeout(timer); reject(error);});
    const timer = setTimeout(() => {worker.sendToCpp("close"); reject(new Error("Share verifier deadline"));}, 5000);
    try {worker.sendToCpp("test", {algo, dev: "cpu", blob_hex: input.toString("hex")});} catch (error) {clearTimeout(timer); reject(error);}
  });
}

/** Use the same private protocol fixture for plaintext and authenticated TLS. The caller
 * supplies only disposable certificate material; binding remains loopback-only in mine().
 * @param {PrivateTls | undefined} options @param {(socket: import("node:net").Socket) => void} handle */
function miningServer(options, handle) {
  return options ? tls.createServer({key: options.key, cert: options.cert, minVersion: "TLSv1.2"}, handle) : net.createServer(handle);
}

/**
 * Terminate only the controller process tree after cooperative close has already failed. Unix
 * keeps the process-group boundary; Windows has no negative-PID process groups, so use the
 * system tree terminator and retain a direct-process fallback if it cannot be started.
 * @param {TerminationChild} child
 * @param {string} [platform] @param {TerminationSpawn} [spawnProcess] @param {ProcessGroupKill} [killProcess]
 */
function terminateChild(child, platform = process.platform,
  spawnProcess = /** @type {TerminationSpawn} */ (spawn),
  killProcess = /** @type {ProcessGroupKill} */ (process.kill.bind(process))) {
  if (!child?.pid || child.exitCode !== null || child.signalCode !== null) {return;}
  if (platform !== "win32") {
    try {killProcess(-child.pid, "SIGKILL");} catch (error) {
      if (Reflect.get(Object(error), "code") !== "ESRCH") {throw error;}
    }
    return;
  }
  const running = () => child.exitCode === null && child.signalCode === null;
  const systemRoot = process.env["SystemRoot"] || "C:\\Windows";
  const killer = spawnProcess(path.win32.join(systemRoot, "System32", "taskkill.exe"),
    ["/pid", String(child.pid), "/t", "/f"], {stdio: "ignore", windowsHide: true, timeout: 5000});
  const fallback = () => {if (running()) {try {child.kill("SIGKILL");} catch { /* process is already gone */ }}};
  killer.once("error", fallback);
  killer.once("close", (code, signal) => {if (code || signal) {fallback();}});
}

/** @param {{isFile: () => boolean, mode: number}} stat @param {string} [platform] */
function savedConfigPermissions(stat, platform = process.platform) {
  if (!stat.isFile()) {throw new Error("Saved config is not a regular file");}
  if (platform === "win32") {
    // Node deliberately skips chmod on Windows. Do not turn Windows mode bits into a fake
    // POSIX 0600 result; the caller still checks the serialized file and save diagnostics.
    return {mode: null, posixModeApplicable: false, posixModeVerified: false};
  }
  const mode = stat.mode & 0o777;
  if (mode !== 0o600) {throw new Error("Saved config is not mode 0600");}
  return {mode, posixModeApplicable: true, posixModeVerified: true};
}

/** @param {{command: string, args: string[], env: NodeJS.ProcessEnv}} runner
 * @param {CpuOracle} core @param {string} output @param {string} name @param {PrivateTls} [secure]
 * @param {string} [host] */
async function mine(runner, core, output, name, secure, host = "127.0.0.1") {
  // Both address families use the same maintained controller/share oracle. Never widen this
  // private fixture to a wildcard listener or a public mining endpoint.
  if (host !== "127.0.0.1" && host !== "::1") {throw new Error("Mining parity requires a literal loopback host");}
  /** @type {Set<import("node:net").Socket>} */
  const sockets = new Set();
  /** @type {string[]} */
  const trace = [];
  const tlsVersions = /** @type {string[]} */ ([]);
  let logins = 0, phase = 0, rejected = false, stopping = false, verifiedSamples = 0;
  let failure = /** @type {Error | null} */ (null);
  let child = /** @type {import("node:child_process").ChildProcessWithoutNullStreams | undefined} */ (undefined);
  let force = /** @type {ReturnType<typeof setTimeout> | undefined} */ (undefined);
  /** @param {Error} [error] */
  function stop(error) {
    if (error) {failure ||= error;}
    if (stopping || !child) {return;}
    stopping = true;
    child.stdin.write("close\n");
    // This new process group contains only our controller and its explicit worker children.
    force = setTimeout(() => {
      failure ||= new Error("Mining controller did not close cooperatively");
      if (child?.pid) {
        try {terminateChild(child);} catch (error) {failure = error instanceof Error ? error : new Error(String(error));}
      }
    }, 7500); // The maintained controller permits up to five seconds for final share replies.
  }
  /** @param {import("node:net").Socket} socket @param {object} value */
  const send = (socket, value) => {if (!socket.destroyed) {socket.write(JSON.stringify(value) + "\n");}};
  const server = miningServer(secure, (socket) => {
    if (secure) {
      const protocol = socket instanceof tls.TLSSocket && socket.getProtocol();
      if (protocol !== "TLSv1.2" && protocol !== "TLSv1.3") {socket.destroy(); stop(new Error("Private TLS fixture is not encrypted")); return;}
      tlsVersions.push(protocol);
    }
    sockets.add(socket); socket.on("close", () => sockets.delete(socket)); socket.on("error", () => undefined);
    let input = "", workerId = "", verifying = false;
    /** @type {Map<string, string>} */
    const jobs = new Map();
    /** @param {string} algo */
    function job(algo) {
      const value = {algo, job_id: `${logins}-${phase}`, blob, height: 0, target: "ffffffff"};
      jobs.set(value.job_id, algo); return value;
    }
    /** @param {string} line */
    function handle(line) {
      if (socket.destroyed) {return;}
      try {
        const message = JSON.parse(line);
        if (message.method === "login") {
          if (workerId || !["cn/0", "cn/1"].every((algo) => message.params.algo.includes(algo))) {
            throw new Error("Mining login did not advertise the selected CPU algorithms");
          }
          if (++logins > 2) {throw new Error("Unexpected repeated reconnect");}
          workerId = "loopback-" + logins;
          if (logins === 2) {trace.push("reconnected"); phase = 2;}
          send(socket, {id: message.id, result: {id: workerId, status: "OK", extensions: ["mo-native"], job: job("cn/0")}});
        } else if (message.method === "submit") {
          const params = message.params;
          const algo = jobs.get(params?.job_id);
          if (!algo || params.id !== workerId || !/^[0-9a-f]{8}$/.test(params.nonce) || !/^[0-9a-f]{64}$/.test(params.result)) {
            throw new Error("Malformed or wrong-connection share");
          }
          // Old work can be in flight at a switch. Acknowledge it, but only a newly verified
          // sample for this phase may advance the fixture. Never label a stale sample a pass.
          if (verifying || stopping || algo !== (phase === 1 ? "cn/1" : "cn/0")) {
            send(socket, {id: message.id, result: {status: "OK"}}); return;
          }
          verifying = true;
          verify(core, algo, params.nonce, params.result).then(() => {
            verifiedSamples++;
            if (!rejected) {
              rejected = true; trace.push("rejected");
              send(socket, {id: message.id, error: {code: -1, message: "Expected fixture rejection"}});
            } else {
              trace.push("accepted:" + algo); send(socket, {id: message.id, result: {status: "OK"}});
              if (phase === 0) {phase = 1; send(socket, {method: "job", params: job("cn/1")});} else if (phase === 1) {socket.destroy();} else {stop();}
            }
            verifying = false;
          }).catch((error) => stop(error));
        } else {throw new Error(`Unexpected loopback method: ${message.method}`);}
      } catch (error) {stop(error instanceof Error ? error : new Error(String(error)));}
    }
    socket.on("data", (bytes) => {
      input += bytes.toString("utf8");
      if (input.length > 1024 * 1024) {socket.destroy(); stop(new Error("Loopback request limit")); return;}
      let end;
      while ((end = input.indexOf("\n")) !== -1) {
        const line = input.slice(0, end); input = input.slice(end + 1); handle(line);
      }
    });
  });
  if (secure) {
    // Track pre-handshake sockets too: a failed/stalled TLS peer never reaches the shared
    // application handler, but must not keep server.close() pending during fixture cleanup.
    server.on("connection", socket => {sockets.add(socket); socket.on("close", () => sockets.delete(socket)); socket.on("error", () => undefined);});
    server.on("tlsClientError", (error) => stop(error));
  }
  server.maxConnections = 4;
  server.listen({host, port: 0}); await once(server, "listening");
  const address = server.address();
  if (!address || typeof address === "string") {throw new Error("No private mining endpoint");}
  const config = path.join(output, `${name}-config.json`);
  const savedConfig = path.join(output, `${name}-saved.json`);
  // A preexisting permissive file proves the controller's separate chmod, not just open's mode.
  fs.writeFileSync(savedConfig, "replace this fixture", {mode: 0o644}); fs.chmodSync(savedConfig, 0o644);
  fs.writeFileSync(config, JSON.stringify({bench_algo_params: 0, log_level: 0, save_config: savedConfig,
    pools: [{url: host, port: address.port, login: "loopback-fixture", use_subscribe: false, is_keepalive: false,
      ...(secure ? {is_tls: true, tls_verify: true} : {})}],
    pool_ids: {primary: 0, donate: null}, pool_time: {connect_throttle: 0.05, first_job_wait: 3, close_wait: 0.01},
    algo_params: {"cn/0": {dev: "cpu^1", perf: 1}, "cn/1": {dev: "cpu^1", perf: 1}}}));
  let stdout = "", stderr = "";
  // A packed native executable must decompress again for each replacement worker. Keep this
  // correctness deadline bounded without treating that separately measured startup cost as a
  // protocol failure when the fixture switches algorithms and reconnects.
  const deadline = setTimeout(() => stop(new Error("Private mining probe exceeded 60 seconds")), 60000);
  try {
    child = spawn(runner.command, [...runner.args, "mine", config], {env: {...runner.env,
      MOM_BENCHMARK_CONTROL_STDIN: "1", MOM_TEST_NO_POOL_NETWORK: "1", MOM_SKIP_MSR: "1", MOM_SKIP_SYCL_ALGO_PARAMS: "1",
      MOM_USE_SUBPROCESS_WORKERS: "1", MOM_GPU_BACKEND: ""}, detached: process.platform !== "win32", windowsHide: true});
    child.on("error", (error) => {failure = error;}); child.stdin.on("error", (error) => {if (!stopping) {stop(error);}});
    child.stdout.on("data", (value) => {
      stdout += value;
      if (stdout.length > 1024 * 1024) {stdout = stdout.slice(0, 1024 * 1024); stop(new Error("Mining log limit"));}
    });
    child.stderr.on("data", (value) => {
      stderr += value;
      if (stderr.length > 1024 * 1024) {stderr = stderr.slice(0, 1024 * 1024); stop(new Error("Mining error log limit"));}
    });
    const [status, signal] = await once(child, "close");
    fs.writeFileSync(path.join(output, `${name}.log`), stdout + stderr);
    fs.writeFileSync(path.join(output, `${name}-result.json`), JSON.stringify({status, signal, logins, trace, verifiedSamples,
      failure: failure?.message}, null, 2) + "\n");
    const expected = ["rejected", "accepted:cn/0", "accepted:cn/1", "reconnected", "accepted:cn/0"];
    if (status !== 0 || signal || verifiedSamples !== 4 || !isDeepStrictEqual(trace, expected) ||
        !stdout.includes("Share accepted by the pool") || !(stdout + stderr).includes("Share rejected by the pool") ||
        /Native (cleanup retained handles|loop did not drain)/.test(stderr)) {
      failure ||= new Error(`Mining parity failed (${name} status=${status}, signal=${signal}); see ${name}.log`);
    }
    if (failure) {throw failure;}
    const stat = fs.lstatSync(savedConfig), saved = JSON.parse(fs.readFileSync(savedConfig, "utf8"));
    const permissions = savedConfigPermissions(stat);
    if (/Error saving/.test(stdout + stderr) ||
        saved.bench_algo_params !== 0 || saved.pools?.[0]?.login !== "loopback-fixture" ||
        saved.algo_params?.["cn/0"]?.dev !== "cpu" || saved.algo_params?.["cn/1"]?.dev !== "cpu") {
      throw new Error("Original controller did not save its private config correctly");
    }
    return {logins, trace, verifiedSamples, cooperativeClose: true, savedConfig: {...permissions, valid: true},
      ...(secure ? {tlsVersions} : {})};
  } finally {
    clearTimeout(deadline); clearTimeout(force);
    for (const socket of sockets) {socket.destroy();}
    await new Promise((resolve) => server.close(resolve));
  }
}

async function main() {
  const [executable, parent] = process.argv.slice(2), addon = process.env["MOM_AOT_CORE_ADDON"];
  if (!["linux", "win32"].includes(process.platform) || !executable || !path.isAbsolute(executable) ||
      !addon || !path.isAbsolute(addon) || process.argv.length > 4) {
    throw new Error("Usage (Linux/Windows, MOM_AOT_CORE_ADDON required): node scripts/probe-native-mining.js /absolute/native-mom [evidence-parent]");
  }
  const output = fs.mkdtempSync(path.join(path.resolve(parent || os.tmpdir()), "mom-mine-"));
  console.log(`Private mining parity evidence: ${output}`);
  const core = require(addon);
  const env = {...process.env, MOM_NATIVE_PATH: addon, MOM_AOT_REJECTION: undefined, MOM_AOT_BULK: undefined,
    MOM_AOT_CHILD_TASK: undefined, MOM_AOT_FAILURE: undefined, MOM_AOT_FILE_FAULT: undefined, MOM_AOT_FILE_FAULT_PATH: undefined};
  const node = await mine({command: process.execPath, args: [path.join(repo, "mom.js")], env: {...env, MOM_AOT_RUN_CONTROLLER: undefined}}, core, output, "node");
  const nativeEnv = /** @type {NodeJS.ProcessEnv} */ ({...env, MOM_AOT_RUN_CONTROLLER: "1", MOM_NATIVE_PATH: ""});
  if (process.platform === "win32") {
    delete nativeEnv["Path"]; delete nativeEnv["LD_LIBRARY_PATH"]; delete nativeEnv["MOM_AOT_CORE_ADDON"];
    const systemRoot = process.env["SystemRoot"] || "C:\\Windows";
    nativeEnv["PATH"] = `${systemRoot}\\System32;${systemRoot}`;
  } else {
    nativeEnv["LD_LIBRARY_PATH"] = "/unavailable"; nativeEnv["PATH"] = "/unavailable";
  }
  const native = await mine({command: executable, args: [], env: nativeEnv}, core, output, "native");
  if (!isDeepStrictEqual(node, native)) {throw new Error("Native mining transcript differs from Node");}
  const saved = ["node", "native"].map((name) => {
    const config = JSON.parse(fs.readFileSync(path.join(output, `${name}-saved.json`), "utf8"));
    // Each private listener has its own ephemeral port. Normalize only that transport detail;
    // compare the entire maintained serializer output, not a hand-picked subset of options.
    config.pools[0].port = 0;
    return config;
  });
  if (!isDeepStrictEqual(saved[0], saved[1])) {throw new Error("Saved native config differs from Node");}
  fs.writeFileSync(path.join(output, "saved-config-parity.json"), JSON.stringify(saved[0], null, 2) + "\n");
  fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({node, native, scope: "Private CPU-loopback only; no public pool or GPU qualification"}, null, 2) + "\n");
  console.log("Original controller mining/switch/reconnect/rejection/close parity passed.");
}
if (require.main === module) {main().catch((error) => {console.error(error); process.exitCode = 1;});}
module.exports = {verify, mine, miningServer, terminateChild, savedConfigPermissions};
