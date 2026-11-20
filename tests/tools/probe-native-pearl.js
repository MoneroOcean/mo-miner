"use strict";

// Explicit GPU execution lane, not a release default. Verify fresh device ownership first.
// Only our native image/config enter the constrained container. The host's private loopback
// fixture checks real submissions with node-powhash, never a copied Pearl implementation.
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), {createHash} = require("node:crypto");
const net = require("node:net"), {once} = require("node:events"), {spawn, spawnSync} = require("node:child_process");
const {gunzipSync} = require("node:zlib");
const {nativeDependencies, auditNativeTrace, auditNativeHeaders} = require("../../scripts/native-audit");

/** @param {string | undefined} value @returns {string | undefined} */
function nativeCgroupParent(value) {
  if (value === undefined) {return undefined;}
  if (value.length > 255 || !/^[A-Za-z0-9][A-Za-z0-9_.-]*\.slice$/.test(value)) {
    throw new Error("MOM_NATIVE_CGROUP_PARENT must be one systemd *.slice name");
  }
  return value;
}

/** @param {string | undefined} parent @param {string[]} straceArgs */
function nativeContainerLaunch(parent, straceArgs) {
  if (!parent) {return {dockerArgs: [], entrypoint: "/usr/bin/strace", commandArgs: straceArgs};}
  return {
    dockerArgs: ["--cgroup-parent", parent], entrypoint: "/bin/sh",
    commandArgs: ["-ec", "set -eu; read -r memory_max < /sys/fs/cgroup/memory.max; " +
      "test \"$memory_max\" = 4294967296; read -r swap_max < /sys/fs/cgroup/memory.swap.max; " +
      "test \"$swap_max\" = 0; exec /usr/bin/strace \"$@\"",
    "mom-native-cgroup-guard", ...straceArgs],
  };
}

/** @param {{gpu: string}} profile @param {string} device @param {number | undefined} uid @param {number | undefined} gid @returns {string[]} */
function nativeDeviceArgs(profile, device, uid, gid) {
  if (profile.gpu === "amd") {return ["--runtime=runc", "--user", "0:0", "--device", "/dev/kfd", "--device", device];}
  if (profile.gpu !== "nvidia" || !Number.isSafeInteger(uid) || (uid ?? -1) < 0 ||
      !Number.isSafeInteger(gid) || (gid ?? -1) < 0) {
    throw new Error("NVIDIA container requires the invoking host UID:GID");
  }
  // With capabilities dropped, container root cannot read this caller-owned bind-mounted config.
  return ["--user", `${uid}:${gid}`, "--gpus", "device=" + device];
}

/** @param {{k: number, rank: number}} shape @returns {{target: string, baseTarget: Buffer}} */
function nativePearlFixtureTarget(shape) {
  // Keep share odds bounded while making two fixture proofs practical after cold GPU initialization.
  const baseValue = ((1n << 256n) - 1n) / BigInt(16 * 16 * (shape.k / shape.rank) * 128 * 1048576);
  const target = baseValue.toString(16).padStart(64, "0");
  return {target, baseTarget: Buffer.from(target, "hex").reverse()};
}

/** @param {boolean} traceIO @returns {string[]} */
function nativePearlStraceArgs(traceIO) {
  return [...(traceIO ? ["-tt"] : []), "-f", "-s", "1024", "-e",
    "trace=open,openat,openat2,creat,execve,execveat,connect" + (traceIO ? ",read,write" : ""),
    "-o", "/dev/stderr", "/mom", "mine", "/config.json"];
}

/** @param {string[]} cacheFiles @param {string} entryFile @returns {string[]} */
function nativeVerifierModuleFiles(cacheFiles, entryFile) {
  const entry = path.resolve(entryFile), addonRoot = path.dirname(entry) + path.sep;
  return cacheFiles.filter((file) => file === entry || (file.startsWith(addonRoot) && file.endsWith(".node")));
}

/** Small valid source-JIT profiles, not performance defaults. Identity admission is pure;
 * fresh occupancy/ownership remains the caller's responsibility before executing this helper.
 * @param {string | undefined} device */
function probeProfile(device) {
  if (!device || /\s/.test(device)) {return null;}
  if (/^GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(device)) {
    return {gpu: "nvidia", shape: {m: 768, n: 512, k: 65536, rank: 128}, rows: 24, columns: 8,
      marker: "PEARLHASH_COMPACT admitted=1"};
  }
  if (/^\/dev\/dri\/renderD[0-9]+$/.test(device)) {
    // m is whole 208-row lane-proof groups (kAmdLaneRows), so every computed row is opened.
    return {gpu: "amd", shape: {m: 2080, n: 2048, k: 2048, rank: 128}, rows: 52, columns: 4,
      marker: "PEARLHASH_HIP_STATS search="};
  }
  return null;
}

async function main() {
  const [executable, verifier, device, parent] = process.argv.slice(2), profile = probeProfile(device);
  const cgroupParent = nativeCgroupParent(process.env["MOM_NATIVE_CGROUP_PARENT"]);
  if (process.platform !== "linux" || !executable || !verifier || !profile || process.argv.length > 6 ||
      !path.isAbsolute(executable) || !path.isAbsolute(verifier) ||
      process.env["MOM_AOT_GPU_OWNERSHIP_VERIFIED"] !== "1") {
    throw new Error("Verify fresh GPU ownership, set MOM_AOT_GPU_OWNERSHIP_VERIFIED=1, then: probe-native-pearl.js <absolute-native-image> <absolute-node-powhash> <NVIDIA GPU-uuid | AMD /dev/dri/renderDnumber> [evidence-parent]");
  }
  const powhash = require(verifier);
  if (typeof powhash.pearl_v3 !== "function") {throw new Error("Missing independent Pearl verifier");}
  const output = fs.mkdtempSync(path.join(path.resolve(parent || os.tmpdir()), "mom-pearl-native-"));
  console.log(`Private native Pearl proof evidence: ${output}`);
  // Small valid compact geometry, not a performance shape. Two seeds cover retained-B reuse.
  const {shape} = profile, header = Buffer.alloc(76); header.writeUInt32LE(3);
  const expectedFactor = profile.rows * profile.columns * (shape.k / shape.rank) * 128;
  // Leave room for the verifier's consensus adjustment multiplication; an all-FF base target
  // overflows upstream's fixed-width target arithmetic and is not a valid easy-share fixture.
  // Keep target odds far from the queue-flooding all-winning case.
  const {target, baseTarget} = nativePearlFixtureTarget(shape);
  const container = "mom-094-pearl-proof-" + path.basename(output);
  /** @type {import('node:child_process').ChildProcessWithoutNullStreams | undefined} */
  let child;
  /** @type {Error | undefined} */
  let failure;
  let stopping = false, authorized = false, loginCount = 0;
  /** @type {Array<Record<string, unknown>>} */
  const shares = [];
  /** @type {Set<import('node:net').Socket>} */
  const sockets = new Set();
  /** @type {NodeJS.Timeout | undefined} */
  let force;
  /** @param {Error} [error] */
  function stop(error) {
    failure ||= error;
    if (stopping || !child) {return;}
    stopping = true;
    fs.writeFileSync(path.join(output, "close-request.json"), JSON.stringify({at: new Date().toISOString(), error: error?.message,
      shares: shares.length, stdoutBytes: stdout.length, stderrBytes: stderr.length}) + "\n");
    child.stdin.end("close\n");
    // Stop only this owned container if cooperative shutdown fails, never a shared GPU reset.
    force = setTimeout(() => {
      failure ||= new Error("Native Pearl did not close cooperatively");
      spawnSync("docker", ["stop", "--time", "5", container], {encoding: "utf8", timeout: 15000});
    }, 60000); // A full bounded proof/output queue may need longer than a single attempt to drain.
  }
  /** @param {import('node:net').Socket} socket @param {object} value */
  const send = (socket, value) => {if (!socket.destroyed) {socket.write(JSON.stringify(value) + "\n");}};
  const server = net.createServer((socket) => {
    sockets.add(socket); socket.on("close", () => sockets.delete(socket)); socket.on("error", () => undefined);
    let pending = "";
    socket.on("data", (bytes) => {
      pending += bytes.toString("utf8");
      if (pending.length > 16 << 20) {stop(new Error("Private Pearl request limit")); socket.destroy(); return;}
      let end;
      while ((end = pending.indexOf("\n")) >= 0) {
        const line = pending.slice(0, end); pending = pending.slice(end + 1);
        try {
          const message = JSON.parse(line);
          if (message.method === "login") {
            if (++loginCount !== 1 || !message.params?.algo?.includes("pearlhash")) {throw new Error("Unexpected Pearl capability/reconnect");}
            send(socket, {id: message.id, result: {id: "private-pearl", status: "OK", extensions: ["mo-native"],
              job: {job_id: "private-compact", algo: "pearlhash", header: header.toString("hex"),
                target, cert_version: 3, proof_encodings: ["none", "gzip"]}}});
          } else if (message.method === "mining.submit") {
            const params = message.params;
            if (params?.job_id !== "private-compact" || typeof params.plain_proof !== "string") {throw new Error("Malformed/wrong-job Pearl submission");}
            if (shares.length < 2) {
              let proof = Buffer.from(params.plain_proof, "base64");
              if (proof.toString("base64") !== params.plain_proof) {throw new Error("Noncanonical proof base64");}
              const uploadBytes = proof.length;
              if (params.proof_encoding === "gzip") {proof = gunzipSync(proof, {maxOutputLength: 8 << 20});} else if (params.proof_encoding && params.proof_encoding !== "none") {throw new Error("Unknown proof encoding");}
              fs.writeFileSync(path.join(output, `proof-${shares.length}.bin`), proof, {mode: 0o600});
              const result = powhash.pearl_v3(header, proof.toString("base64"), baseTarget);
              const proofSha256 = createHash("sha256").update(proof).digest("hex");
              fs.writeFileSync(path.join(output, `verdict-${shares.length}.json`), JSON.stringify({valid: result.valid, candidate: result.candidate,
                error: result.error, config: result.config, expectedJackpot: result.jackpot?.toString("hex"),
                claimedJackpot: params.jackpot, claimedFactor: params.adjustment_factor, proofSha256}, null, 2) + "\n");
              if (!result.valid || !result.candidate || result.config.adjustment_factor !== expectedFactor ||
                  params.adjustment_factor !== expectedFactor || params.jackpot !== result.jackpot.toString("hex") ||
                  shares.some((prior) => prior["proofSha256"] === proofSha256) || params.proof_encoding !== "gzip") {
                throw new Error("Independent native verifier/claim mismatch; see retained proof and verdict in " + output);
              }
              shares.push({config: result.config, jackpot: params.jackpot, adjustment_factor: params.adjustment_factor,
                proofSha256, encoding: params.proof_encoding || "none", uploadBytes, rawBytes: proof.length});
            }
            send(socket, {id: message.id, result: {status: "OK"}});
          } else {throw new Error("Unexpected private Pearl method: " + message.method);}
        } catch (error) {stop(error instanceof Error ? error : new Error(String(error)));}
      }
    });
  });
  server.maxConnections = 2; server.listen({host: "127.0.0.1", port: 0}); await once(server, "listening");
  const address = server.address(); if (!address || typeof address === "string") {throw new Error("No private listener");}
  const config = path.join(output, "config.json");
  fs.writeFileSync(config, JSON.stringify({bench_algo_params: 0, log_level: 0,
    pools: [{url: "127.0.0.1", port: address.port, login: "private-proof-fixture", use_subscribe: false, is_keepalive: false, protocol: "pearlhash"}],
    pool_ids: {primary: 0, donate: null}, pool_time: {first_job_wait: 10},
    algo_params: {pearlhash: {dev: `gpu1*[m=${shape.m};n=${shape.n};k=${shape.k};rank=${shape.rank}]`, backend: "native", perf: 1}}}));
  const env = {PATH: "/unavailable", LD_LIBRARY_PATH: "/unavailable", MOM_AOT_RUN_CONTROLLER: "1", MOM_BENCHMARK_CONTROL_STDIN: "1",
    MOM_GPU_BACKEND: profile.gpu, MOM_SKIP_SYCL_ALGO_PARAMS: "1", MOM_SKIP_MSR: "1", MOM_PEARLHASH_STATS: "1",
    ACPP_VISIBILITY_MASK: profile.gpu === "amd" ? "hip" : "cuda", OMP_NUM_THREADS: "2", XDG_DATA_HOME: "/cache",
    CUDA_CACHE_PATH: "/cache/cuda", MOM_JIT_CACHE_DIR: "/cache/jit", TMPDIR: "/scratch",
    ...(profile.gpu === "nvidia" ? {MOM_NVIDIA_COMPUTE_CAPABILITY: "120"} : {})};
  const devices = nativeDeviceArgs(profile, String(device), process.getuid?.(), process.getgid?.());
  const straceArgs = nativePearlStraceArgs(process.env["MOM_AOT_TRACE_IO"] === "1");
  const launch = nativeContainerLaunch(cgroupParent, straceArgs);
  const args = ["run", "--interactive", "--name", container, ...devices, ...launch.dockerArgs, "--network", "host", "--read-only", "--ulimit", "core=0",
    "--cpus", "4", "--memory", "4g", "--memory-swap", "4g", "--pids-limit", "128", "--cap-drop", "ALL", "--cap-add", "SYS_PTRACE",
    "--security-opt", "no-new-privileges", "--tmpfs", "/cache:rw,noexec,nosuid,size=256m,mode=1777",
    "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=256m", "--tmpfs", "/root:rw,noexec,nosuid,nodev,size=128m",
    "--tmpfs", "/scratch:rw,noexec,nosuid,nodev,size=128m,mode=1777",
    "--mount", `type=bind,src=${executable},dst=/mom,readonly`, "--mount", `type=bind,src=${config},dst=/config.json,readonly`,
    ...Object.entries(env).flatMap(([key, value]) => ["--env", `${key}=${value}`]), "--entrypoint", launch.entrypoint,
    "mom-native-audit:ubuntu24", ...launch.commandArgs];
  fs.writeFileSync(path.join(output, "command.json"), JSON.stringify({command: "docker", args}, null, 2) + "\n");
  let stdout = "", stderr = "";
  const interrupt = () => stop(new Error("Private Pearl probe interrupted"));
  process.once("SIGINT", interrupt); process.once("SIGTERM", interrupt);
  const deadline = setTimeout(() => stop(new Error("Private Pearl proof exceeded 120 seconds")), 120000);
  try {
    child = spawn("docker", args);
    child.on("error", stop); child.stdin.on("error", (error) => {if (!stopping) {stop(error);}});
    child.stdout.on("data", (data) => {
      stdout += data;
      const accepted = (stdout.match(/Share accepted by the pool/g) || []).length;
      if (accepted >= 2 && shares.length >= 2) {authorized = true; stop();}
      if (stdout.length > 1 << 20) {stop(new Error("Pearl log limit"));}
    });
    child.stderr.on("data", (data) => {stderr += data; if (stderr.length > 8 << 20) {stop(new Error("Pearl trace limit"));}});
    const [status, signal] = await once(child, "close");
    fs.writeFileSync(path.join(output, "stdout.log"), stdout); fs.writeFileSync(path.join(output, "trace.log"), stderr);
    if (failure) {throw failure;}
    if (status || signal || !authorized || shares.length !== 2 || !stderr.includes(profile.marker) ||
        /unavailable, using|ERROR:|Native (cleanup retained handles|loop did not drain)/.test(stdout + stderr)) {
      throw new Error("Native mining/shutdown failed; see " + output);
    }
    const cudaDriver = profile.gpu === "nvidia";
    // /scratch is a fresh bounded tmpfs, never a mounted SDK or persistent input directory.
    auditNativeTrace(stderr, "/mom", 2, {cudaDriver});
    auditNativeHeaders(stderr, [], ["/scratch", ...(cudaDriver ? ["/cache/cuda"] : [])]);
    const dynamic = spawnSync("readelf", ["-d", executable], {encoding: "utf8"}); if (dynamic.status) {throw new Error("ELF audit failed");}
    const needed = nativeDependencies(dynamic.stdout);
    const inspected = spawnSync("docker", ["inspect", container], {encoding: "utf8"}); if (inspected.status) {throw new Error("Container audit failed");}
    const [details] = JSON.parse(inspected.stdout);
    if (details.State.Running || details.State.OOMKilled || details.State.ExitCode !== 0) {throw new Error("Container failed to drain");}
    fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({gpu: profile.gpu, shape, expectedFactor, loginCount, shares, needed,
      executable: {file: executable, bytes: fs.statSync(executable).size, sha256: createHash("sha256").update(fs.readFileSync(executable)).digest("hex")},
      verifier: nativeVerifierModuleFiles(Object.keys(require.cache), require.resolve(verifier)).map((file) => ({file,
        sha256: createHash("sha256").update(fs.readFileSync(file)).digest("hex")})),
      cooperativeClose: true, state: details.State, hostConfig: details.HostConfig, mounts: details.Mounts,
      scope: "Actual native controller mining, two independent verifier-accepted private loopback proofs, including trusted claims/gzip. Not external pool or performance qualification."}, null, 2) + "\n");
    console.log(`Two ${profile.gpu} native Pearl proofs independently verified; controller closed cooperatively.`);
  } finally {
    process.removeListener("SIGINT", interrupt); process.removeListener("SIGTERM", interrupt);
    clearTimeout(deadline); clearTimeout(force); for (const socket of sockets) {socket.destroy();}
    await new Promise((resolve) => server.close(resolve));
    // A lost Docker client is not proof that its container stopped. Preserve state and retire
    // only this uniquely named job on error/interruption; never leave compute running unseen.
    const inspected = spawnSync("docker", ["inspect", container], {encoding: "utf8", timeout: 10000});
    if (!inspected.status) {
      const [details] = JSON.parse(inspected.stdout);
      fs.writeFileSync(path.join(output, "final-container-state.json"), JSON.stringify(details.State, null, 2) + "\n");
      if (details.State.Running) {
        spawnSync("docker", ["stop", "--time", "5", container], {encoding: "utf8", timeout: 15000});
        console.error("Private Pearl container required final cleanup; check " + output);
        process.exitCode = 1;
      }
    }
  }
}

module.exports = {nativeCgroupParent, nativeContainerLaunch, nativeDeviceArgs, nativePearlFixtureTarget, nativePearlStraceArgs,
  nativeVerifierModuleFiles, probeProfile};
if (require.main === module) {main().catch((error) => {console.error(error); process.exitCode = 1;});}
